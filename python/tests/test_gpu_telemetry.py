import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.common.telemetry import (
    load_complete_telemetry,
    monitored_trial_command,
)
from scripts.common.gpu_monitor import (
    INVALID_EXIT,
    compute_processes,
    is_exact_owned_process,
    is_owned_process_session,
    is_previously_observed_owned_process,
    is_unresolved_owned_teardown_process,
    process_metadata_identity,
    telemetry_status,
    CohortPeers,
    wait_for_identifiable_exit,
)


def _process(
    pid: int,
    *,
    name: str,
    args: str | None,
    user: str | None = None,
    ppid: int | None = None,
    session_id: int | None = None,
    start_time_ticks: int | None = None,
) -> dict:
    return {
        "pid": pid,
        "process_name": name,
        "args": args,
        "user": user,
        "ppid": ppid,
        "session_id": session_id,
        "start_time_ticks": start_time_ticks,
    }


class GpuTelemetryOwnershipTests(unittest.TestCase):
    def test_direct_child_is_not_reaped_until_nvml_and_peers_settle(self):
        from unittest.mock import Mock
        import os
        child = Mock(pid=42)
        child.wait.return_value = 0
        events = []
        def query(_uuid):
            child.wait.assert_not_called()
            events.append('query')
            return [{'pid':42}] if len(events)==1 else []
        with patch('scripts.common.gpu_monitor.os.waitid') as waitid, \
             patch('scripts.common.gpu_monitor.compute_processes', side_effect=query), \
             patch('scripts.common.gpu_monitor.time.sleep') as sleep:
            self.assertEqual(wait_for_identifiable_exit(child, 'gpu', quiescence_s=10.25), 0)
            waitid.assert_called_once_with(os.P_PID,42,os.WEXITED|os.WNOWAIT)
            self.assertEqual([call.args[0] for call in sleep.call_args_list], [0.1,10.25])
            child.wait.assert_called_once()

    def test_teardown_query_failure_is_not_hidden(self):
        from unittest.mock import Mock
        child = Mock(pid=42)
        with patch('scripts.common.gpu_monitor.os.waitid'), \
             patch('scripts.common.gpu_monitor.compute_processes', side_effect=RuntimeError('query failed')):
            with self.assertRaisesRegex(RuntimeError, 'query failed'):
                wait_for_identifiable_exit(child, 'gpu')
        child.wait.assert_called_once()

    def test_cohort_does_not_relax_default_or_accept_own_leftovers(self):
        process = dict(pid=17, session_id=9, start_time_ticks=20)
        self.assertFalse(CohortPeers(None).accepts(process))
        with patch('scripts.common.gpu_monitor.process_start_time_ticks', return_value=10), \
             patch('scripts.common.gpu_monitor.is_descendant', return_value=True):
            peers = CohortPeers('8:10')
            self.assertTrue(peers.accepts(process, own_session=30))
            self.assertFalse(peers.accepts(process, own_session=9))
            self.assertFalse(peers.accepts(dict(process, start_time_ticks=None), own_session=30))
        with patch('scripts.common.gpu_monitor.process_start_time_ticks', return_value=11):
            self.assertFalse(peers.accepts(process, own_session=30))

    def test_cohort_rejects_unrelated_and_changed_peer_identity(self):
        with patch('scripts.common.gpu_monitor.process_start_time_ticks', return_value=10), \
             patch('scripts.common.gpu_monitor.is_descendant', return_value=True):
            peers = CohortPeers('8:10')
            self.assertTrue(peers.accepts(dict(pid=17, session_id=9, start_time_ticks=20)))
        with patch('scripts.common.gpu_monitor.process_start_time_ticks', return_value=10), \
             patch('scripts.common.gpu_monitor.is_descendant', return_value=False):
            self.assertFalse(peers.accepts(dict(pid=18, session_id=9, start_time_ticks=20)))
            self.assertFalse(peers.accepts(dict(pid=17, session_id=9, start_time_ticks=21)))

    def test_fresh_session_identifies_child_after_parent_chain_disappears(self) -> None:
        zombie = _process(
            17,
            name="/tmp/model_server",
            args="[model_server] <defunct>",
            user="test-user",
            ppid=16,
            session_id=10,
            start_time_ticks=None,
        )
        self.assertTrue(is_owned_process_session(zombie, 10))
        self.assertFalse(is_owned_process_session(zombie, 11))

    def test_reparented_process_requires_the_exact_proved_identity(self) -> None:
        process = _process(
            17,
            name="/tmp/benchmark_child",
            args="benchmark_child",
            user="test-user",
            ppid=1,
            start_time_ticks=1234,
        )
        self.assertTrue(is_exact_owned_process(process, {(17, 1234)}))
        self.assertFalse(is_exact_owned_process(process, {(17, 1235)}))
        self.assertFalse(is_exact_owned_process(process, {(18, 1234)}))
        process["start_time_ticks"] = None
        self.assertFalse(is_exact_owned_process(process, {(17, 1234)}))

    def test_vanished_process_is_ambiguous_and_rejected(self) -> None:
        process = _process(
            17,
            name="/tmp/benchmark_child",
            args="benchmark_child",
            user="test-user",
            ppid=1,
            start_time_ticks=None,
        )
        self.assertFalse(is_exact_owned_process(process, {(17, 1234)}))
        self.assertFalse(is_exact_owned_process(process, {(18, 1234)}))
        process["start_time_ticks"] = 5678
        self.assertFalse(is_exact_owned_process(process, {(17, 1234)}))

    def test_reused_pid_is_foreign_even_if_name_looks_stale(self) -> None:
        process = _process(
            17,
            name="[No data]",
            args="[benchmark_child] <defunct>",
            start_time_ticks=5678,
        )
        self.assertFalse(is_exact_owned_process(process, {(17, 1234)}))

    def test_live_or_never_owned_process_is_not_owned(self) -> None:
        live = _process(
            17,
            name="/tmp/benchmark_child",
            args="benchmark_child",
            user="test-user",
            ppid=10,
            start_time_ticks=5678,
        )
        foreign = _process(
            18,
            name="[No data]",
            args="[python] <defunct>",
            start_time_ticks=None,
        )
        self.assertFalse(is_exact_owned_process(live, {(17, 1234)}))
        self.assertFalse(is_exact_owned_process(foreign, {(17, 1234)}))

    def test_vanished_owned_process_requires_exact_previous_metadata(self) -> None:
        observed = _process(
            17,
            name="/tmp/benchmark_child",
            args="/tmp/benchmark_child",
            user="test-user",
            ppid=10,
            start_time_ticks=1234,
        )
        owned_metadata = {process_metadata_identity(observed)}
        vanished = dict(observed, start_time_ticks=None)
        self.assertTrue(
            is_previously_observed_owned_process(vanished, owned_metadata)
        )
        for field, value in (
            ("pid", 18),
            ("ppid", 11),
            ("user", "other"),
            ("process_name", "/tmp/other"),
            ("args", "/tmp/other"),
        ):
            with self.subTest(field=field):
                changed = dict(vanished, **{field: value})
                self.assertFalse(
                    is_previously_observed_owned_process(changed, owned_metadata)
                )

    def test_owned_zombie_requires_correlated_command_name(self) -> None:
        observed = _process(
            17,
            name="/tmp/benchmark_child",
            args="/tmp/benchmark_child --flag",
            user="test-user",
            ppid=10,
            start_time_ticks=1234,
        )
        owned_metadata = {process_metadata_identity(observed)}
        zombie = _process(
            17,
            name="[No data]",
            args="[benchmark_child] <defunct>",
            user="test-user",
            ppid=10,
            start_time_ticks=None,
        )
        self.assertTrue(
            is_previously_observed_owned_process(zombie, owned_metadata)
        )
        for field, value in (
            ("pid", 18),
            ("ppid", 11),
            ("user", "other"),
            ("args", "[other_process] <defunct>"),
            ("args", "benchmark_child <defunct>"),
        ):
            with self.subTest(field=field, value=value):
                self.assertFalse(
                    is_previously_observed_owned_process(
                        dict(zombie, **{field: value}), owned_metadata
                    )
                )
        self.assertFalse(
            is_previously_observed_owned_process(
                dict(zombie, start_time_ticks=5678), owned_metadata
            )
        )

    def test_owned_zombie_accepts_exact_retained_nvml_name(self) -> None:
        observed = _process(
            17,
            name="/tmp/model_server_dense",
            args="/tmp/model_server_dense",
            user="test-user",
            ppid=10,
            start_time_ticks=1234,
        )
        owned_metadata = {process_metadata_identity(observed)}
        zombie = _process(
            17,
            name="/tmp/model_server_dense",
            args="[model_server_de] <defunct>",
            user="test-user",
            ppid=10,
            start_time_ticks=None,
        )
        self.assertTrue(
            is_previously_observed_owned_process(zombie, owned_metadata)
        )
        self.assertFalse(
            is_previously_observed_owned_process(
                dict(zombie, process_name="/tmp/other"), owned_metadata
            )
        )

    def test_identity_free_nvml_teardown_row_is_deferred_for_owned_pid(self) -> None:
        teardown = _process(
            17,
            name="[No data]",
            args=None,
            user=None,
            ppid=None,
            start_time_ticks=None,
        )
        owned = {(17, 1234)}
        self.assertTrue(is_unresolved_owned_teardown_process(teardown, owned))
        self.assertFalse(
            is_unresolved_owned_teardown_process(dict(teardown, pid=18), owned)
        )
        for field, value in (
            ("start_time_ticks", 5678),
            ("ppid", 10),
            ("user", "test-user"),
            ("process_name", "/tmp/new_process"),
            ("args", "/tmp/new_process"),
        ):
            with self.subTest(field=field):
                self.assertFalse(
                    is_unresolved_owned_teardown_process(
                        dict(teardown, **{field: value}), owned
                    )
                )

    def test_identity_free_teardown_accepts_retained_owned_nvml_name(self) -> None:
        observed = _process(
            17,
            name="/tmp/model_server_dense",
            args="/tmp/model_server_dense",
            user="test-user",
            ppid=10,
            start_time_ticks=1234,
        )
        owned = {(17, 1234)}
        owned_metadata = {process_metadata_identity(observed)}
        teardown = _process(
            17,
            name="/tmp/model_server_dense",
            args=None,
            user=None,
            ppid=None,
            start_time_ticks=None,
        )
        self.assertTrue(
            is_unresolved_owned_teardown_process(
                teardown, owned, owned_metadata
            )
        )
        self.assertFalse(
            is_unresolved_owned_teardown_process(
                dict(teardown, process_name="/tmp/other"),
                owned,
                owned_metadata,
            )
        )

    @patch("scripts.common.gpu_monitor.run_nvidia_smi")
    def test_malformed_compute_process_row_fails_closed(self, query) -> None:
        query.return_value = ["GPU-test, not-a-pid, python, 10"]
        with self.assertRaisesRegex(RuntimeError, "malformed.*PID"):
            compute_processes("GPU-test")

    def test_incomplete_observation_statuses_fail_closed(self) -> None:
        self.assertEqual(
            telemetry_status(
                0,
                sampling_failed=True,
                samples_present=True,
                post_processes_present=False,
                contention=False,
            ),
            ("invalid_sampling_error", INVALID_EXIT),
        )
        self.assertEqual(
            telemetry_status(
                0,
                sampling_failed=False,
                samples_present=True,
                post_processes_present=True,
                contention=False,
            ),
            ("invalid_post_processes", INVALID_EXIT),
        )
        self.assertEqual(
            telemetry_status(
                0,
                sampling_failed=False,
                samples_present=True,
                post_processes_present=False,
                contention=True,
            ),
            ("invalid_contended", INVALID_EXIT),
        )
        self.assertEqual(
            telemetry_status(
                0,
                sampling_failed=False,
                samples_present=False,
                post_processes_present=False,
                contention=False,
            ),
            ("invalid_no_samples", INVALID_EXIT),
        )


class PerTrialTelemetryTests(unittest.TestCase):
    def test_monitored_trial_executes_binary_as_direct_child(self) -> None:
        command = monitored_trial_command(
            Path("/tmp/engine"),
            gpu_index=3,
            telemetry_output=Path("/tmp/trial.json"),
        )
        self.assertEqual(command[-2:], ["--", "/tmp/engine"])
        self.assertEqual(Path(command[1]).resolve(),
                         Path(__file__).resolve().parents[2] / "scripts/common/gpu_monitor.py")
        self.assertTrue(Path(command[1]).is_file())
        self.assertEqual(command[command.index("--gpu-index") + 1], "3")
        self.assertEqual(command[command.index("--output") + 1], "/tmp/trial.json")

    def test_complete_telemetry_is_summarized_and_hashed(self) -> None:
        record = {
            "status": "complete",
            "gpu": {"index": "3", "name": "test"},
            "metric_summary": {"sample_count": 4},
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trial.json"
            data = json.dumps(record).encode()
            path.write_bytes(data)
            summary = load_complete_telemetry(path)
        self.assertEqual(summary["status"], "complete")
        self.assertEqual(summary["metric_summary"]["sample_count"], 4)
        self.assertEqual(summary["sha256"], hashlib.sha256(data).hexdigest())

    def test_rejected_telemetry_cannot_enter_benchmark_result(self) -> None:
        record = {
            "status": "invalid_contended",
            "gpu": {},
            "metric_summary": {"sample_count": 1},
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trial.json"
            path.write_text(json.dumps(record))
            with self.assertRaisesRegex(RuntimeError, "invalid_contended"):
                load_complete_telemetry(path)
