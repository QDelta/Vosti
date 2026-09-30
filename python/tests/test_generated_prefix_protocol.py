import unittest
import contextlib
import io
import json
from pathlib import Path
import tempfile
from unittest.mock import patch

import numpy as np

from scripts.determinism_tests.generated_prefix import check_witness, expected_cached_tokens, run_check
from scripts.determinism_tests.protocol import EXECUTION_CONFIGS
from scripts.determinism_tests.vosti_worker import _run_invocation


class GeneratedPrefixProtocolTests(unittest.TestCase):
    def test_worker_lowers_distinct_donor_and_followup_shapes(self):
        # Exercise the real worker, not just the orchestration's FakeRunner.
        # Stop at the native process boundary after checking both TSV inputs.
        def native_boundary(binary, *, env, stdout, stderr):
            donor = Path(env["VOSTI_BENCH_WARMUP_INPUT"]).read_text().splitlines()
            followup = Path(env["VOSTI_BENCH_INPUT"]).read_text().splitlines()
            self.assertEqual(len(donor), 1)
            self.assertEqual(len(followup), 1)
            self.assertEqual(donor[0].split("\t"), ["128", "1,2,3"])
            self.assertEqual(followup[0].split("\t"), ["2", "1,2,3,4,5"])
            primer = Path(env["VOSTI_BENCH_GRAPH_PRIMER_INPUT"]).read_text().splitlines()
            self.assertEqual(primer, ["4\t6,7,8,9", "4\t10,11,12,13"])
            raise RuntimeError("native boundary reached")

        with tempfile.TemporaryDirectory() as temporary, patch(
            "scripts.determinism_tests.vosti_worker._run_binary_with_identifiable_teardown",
            side_effect=native_boundary,
        ):
            with self.assertRaisesRegex(RuntimeError, "native boundary reached"):
                _run_invocation(
                    arm={"model_path": "/unused", "engine": {
                        "max_num_batched_tokens": 4096, "max_num_seqs": 8, "num_blocks": 64,
                        "graph_primer": {"prompts": [[6, 7, 8, 9], [10, 11, 12, 13]],
                                         "max_tokens": 4}}},
                    binary=Path("/unused"), deployment_bundle=Path("/unused"),
                    invocation_dir=Path(temporary) / "invocation",
                    artifact_dir=Path(temporary) / "rows", call_index=1, vocab_size=100,
                    warmup_call={"prompts": [[1, 2, 3]], "max_tokens": 128},
                    measured_call={"prompts": [[1, 2, 3, 4, 5]], "max_tokens": 2},
                )

    def test_last_sample_is_excluded_before_page_rounding(self):
        self.assertEqual(expected_cached_tokens(63, 1), 0)
        self.assertEqual(expected_cached_tokens(63, 2), 64)
        self.assertEqual(expected_cached_tokens(1024, 128), 1088)
        self.assertEqual(expected_cached_tokens(1024, 129), 1152)

    def witness(self, **changes):
        args = dict(baseline_tokens=[1] * 128, repeated_tokens=[1] * 128,
                    prompt_length=1024, cold_cached=0, warm_cached=1088,
                    warmup_graph={"cover_replay_count": 1})
        args.update(changes)
        return check_witness(**args)

    def test_valid_generated_page_reuse(self):
        self.assertTrue(all(self.witness().values()))

    def test_prompt_only_reuse_is_not_qualification(self):
        self.assertFalse(self.witness(warm_cached=1024)["generated_pages_reused"])

    def test_unexecuted_page_is_not_accepted(self):
        self.assertFalse(self.witness(warm_cached=1152)["exact_materialized_page_count"])

    def test_drift_cold_hits_and_missing_graph_evidence_rejected(self):
        self.assertFalse(self.witness(repeated_tokens=[2] * 128)["donor_generation_repeated_exactly"])
        self.assertFalse(self.witness(cold_cached=64)["cold_cache_empty"])
        self.assertFalse(self.witness(warmup_graph=None)["donor_padded_cover_replayed"])

    def test_orchestration_uses_a_paired_engine_lifetime_and_first_followup_logits(self):
        class FakeRunner:
            def __init__(self, output):
                self.output = output
                self.arm_records = []
                self.configs = []

            def run_arm(self, name, arm):
                self.configs.append(arm)
                self.arm_records.append({"name": name})
                directory = self.output / name
                (directory / "rows").mkdir(parents=True)
                calls = []
                for index, call in enumerate(arm["calls"]):
                    artifact = f"call-{index}.npy"
                    np.save(directory / "rows" / artifact, np.zeros(8, dtype=np.float32))
                    calls.append({"requests": [{
                        "output_token_ids": [1] * call["max_tokens"],
                        "num_cached_tokens": 1088 if index == 1 else 0,
                        "last_row": {"artifact": artifact},
                    }]})
                return directory, {"calls": calls, "backend_evidence": {"invocations": [{
                    "graph_warmup_stats": {"cover_replay_count": 1},
                }]}}

        execution = next(e for e in EXECUTION_CONFIGS if e.key == "vosti-padded-graph")
        with tempfile.TemporaryDirectory() as temporary:
            runner = FakeRunner(Path(temporary))
            with patch("scripts.determinism_tests.generated_prefix.deterministic_prompt",
                       side_effect=lambda path, length, seed: [2] * length), contextlib.redirect_stdout(io.StringIO()):
                run_check(runner, {"model": {"key": "fixture"}, "model_path": "/unused", "seed": 123},
                          execution, hardware="h200")
            self.assertEqual([len(a["calls"]) for a in runner.configs], [1, 1, 2])
            cold, warm = runner.configs[1]["calls"][0], runner.configs[2]["calls"][1]
            self.assertEqual(cold, warm)
            self.assertEqual(len(cold["prompts"][0]), 1217)
            self.assertEqual(cold["record_generated_position"], 0)
            self.assertEqual(cold["max_tokens"], 2)
            self.assertTrue(all(a["engine"]["max_model_len"] == 2048 for a in runner.configs))
            primers = [a["engine"]["graph_primer"] for a in runner.configs]
            self.assertTrue(all(p == primers[0] for p in primers))
            self.assertEqual([len(p) for p in primers[0]["prompts"]], [1280, 1280])
            self.assertEqual(primers[0]["max_tokens"], 4)
            summary = json.loads((runner.output / "summary.json").read_text())
            self.assertEqual(summary["status"], "pass")
            self.assertEqual(len(summary["comparisons"]), 2)


if __name__ == "__main__":
    unittest.main()
