#!/usr/bin/env python3
"""Run a benchmark while recording GPU telemetry and process contention.

The selected GPU must be idle before launch.  While the child process runs,
GPU metrics and compute-process ancestry are sampled once per second.  If a
process outside the benchmark's process tree appears, the benchmark process
group is terminated and the record is marked invalid.
"""

from __future__ import annotations

import argparse
import getpass
import json
import os
from pathlib import Path
import signal
import socket
import statistics
import subprocess
import sys
import threading
import time
from typing import Any


INVALID_EXIT = 86
GPU_FIELDS = (
    "timestamp,index,uuid,pstate,temperature.gpu,utilization.gpu,"
    "utilization.memory,memory.used,clocks.sm,clocks.mem,power.draw"
)
NVIDIA_SMI_TIMEOUT_S = 10.0


def wait_for_identifiable_exit(child, gpu_uuid, *, quiescence_s=0.0, timeout_s=120.0):
    """Keep the exited direct child in /proc until NVML and peer queries settle.

    Reaping first loses the PID/start-time binding while NVIDIA may still list
    its allocation. WNOWAIT prevents PID reuse and preserves peer ancestry.
    Cohorts additionally leave one maximum query interval for in-flight peer
    observations; this is teardown cost, not measured serving performance.
    """
    os.waitid(os.P_PID, child.pid, os.WEXITED | os.WNOWAIT)
    try:
        deadline = time.monotonic() + timeout_s
        while any(row['pid'] == child.pid for row in compute_processes(gpu_uuid)):
            if time.monotonic() >= deadline:
                raise RuntimeError('exited GPU child did not disappear from NVML')
            time.sleep(0.1)
        if quiescence_s:
            time.sleep(quiescence_s)
    finally:
        returncode = child.wait()
    return returncode


class CohortPeers:
    """Explicit correctness-only peer lineage; never enabled by default."""

    def __init__(self, identity: str | None):
        self.root = None
        self.observed = set()
        if identity is not None:
            fields = identity.split(':')
            if len(fields) != 2 or any(not value.isdecimal() for value in fields):
                raise ValueError('cohort identity must be PID:START_TICKS')
            self.root = tuple(map(int, fields))
            if self.root[0] <= 1 or not self.live() or not is_descendant(os.getpid(), self.root[0]):
                raise RuntimeError('cohort root must be the live, identity-bound campaign ancestor')

    def live(self):
        return self.root is not None and process_start_time_ticks(self.root[0]) == self.root[1]

    def accepts(self, process, own_session=None):
        if not self.live() or process.get('session_id') == own_session:
            return False
        start = process.get('start_time_ticks')
        if not isinstance(start, int):
            return False
        identity = (process['pid'], start)
        if identity in self.observed:
            return True
        if is_descendant(process['pid'], self.root[0]):
            self.observed.add(identity)
            return True
        return False


def run_nvidia_smi(arguments: list[str]) -> list[str]:
    completed = subprocess.run(
        ["nvidia-smi", *arguments],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=NVIDIA_SMI_TIMEOUT_S,
    )
    return [line.strip() for line in completed.stdout.splitlines() if line.strip()]


def gpu_identity(index: int) -> dict[str, str]:
    rows = run_nvidia_smi(
        [
            f"--id={index}",
            "--query-gpu=index,uuid,name,driver_version,memory.total",
            "--format=csv,noheader,nounits",
        ]
    )
    if len(rows) != 1:
        raise RuntimeError(f"expected one GPU row for index {index}, got {rows!r}")
    fields = [field.strip() for field in rows[0].split(",")]
    return dict(zip(("index", "uuid", "name", "driver_version", "memory_total_mib"), fields))


def compute_processes(gpu_uuid: str) -> list[dict[str, Any]]:
    rows = run_nvidia_smi(
        [
            "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ]
    )
    processes = []
    for row in rows:
        fields = [field.strip() for field in row.split(",", 3)]
        if len(fields) != 4:
            raise RuntimeError(f"malformed NVIDIA compute-process row: {row!r}")
        if fields[0] != gpu_uuid:
            continue
        try:
            pid = int(fields[1])
        except ValueError as error:
            raise RuntimeError(
                f"malformed NVIDIA compute-process PID: {fields[1]!r}"
            ) from error
        user = None
        ppid = None
        session_id = None
        args = None
        try:
            completed = subprocess.run(
                ["ps", "-o", "user=,ppid=,sid=,args=", "-p", str(pid)],
                check=False,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
            )
            process_row = completed.stdout.strip()
            if process_row:
                user, raw_ppid, raw_sid, args = process_row.split(None, 3)
                ppid = int(raw_ppid)
                session_id = int(raw_sid)
        except (OSError, ValueError):
            pass
        processes.append(
            {
                "pid": pid,
                "ppid": ppid,
                "session_id": session_id,
                "start_time_ticks": process_start_time_ticks(pid),
                "user": user,
                "process_name": fields[2],
                "used_memory_mib": fields[3],
                "args": args,
            }
        )
    return processes


def process_start_time_ticks(pid: int) -> int | None:
    """Return Linux's boot-relative process identity component for ``pid``."""
    try:
        stat = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
        fields = stat[stat.rfind(")") + 2 :].split()
        # fields starts at proc(5)'s field 3 (state); starttime is field 22.
        return int(fields[19])
    except (FileNotFoundError, PermissionError, ValueError, IndexError):
        return None


def is_descendant(pid: int, ancestor: int) -> bool:
    seen = set()
    current = pid
    while current > 1 and current not in seen:
        if current == ancestor:
            return True
        seen.add(current)
        try:
            stat = Path(f"/proc/{current}/stat").read_text(encoding="utf-8")
            # The command name is parenthesized and can contain spaces.
            current = int(stat[stat.rfind(")") + 2 :].split()[1])
        except (FileNotFoundError, PermissionError, ValueError, IndexError):
            return False
    return False


def is_owned_process_session(process: dict[str, Any], session_id: int) -> bool:
    """Recognize a process in the fresh session created for this trial.

    Parent-chain traversal can race with a short-lived launcher such as Cargo:
    a GPU child may remain as a zombie after its parent disappears.  Session
    membership is kernel-maintained lineage evidence and cannot be joined by
    an unrelated process, so it remains valid across that teardown race.
    """

    return process.get("session_id") == session_id


def is_exact_owned_process(
    process: dict[str, Any], owned_identities: set[tuple[int, int]]
) -> bool:
    """Recognize a child only while its full Linux identity is observable.

    A live ``/proc`` identity must match both PID and boot-relative start time.
    Once ``/proc`` has vanished, an NVML teardown row retains only the PID and
    is observationally indistinguishable from sufficiently rapid PID reuse.
    Such a row must therefore fail closed.  Process names and zombie markers
    are never treated as identity evidence.
    """
    start_time = process.get("start_time_ticks")
    return isinstance(start_time, int) and (
        process["pid"], start_time
    ) in owned_identities


def process_metadata_identity(process: dict[str, Any]) -> tuple[Any, ...]:
    """Return the stable metadata available from both ``ps`` and NVML."""

    return (
        process.get("pid"),
        process.get("ppid"),
        process.get("user"),
        process.get("process_name"),
        process.get("args"),
    )


def is_previously_observed_owned_process(
    process: dict[str, Any], owned_metadata: set[tuple[Any, ...]]
) -> bool:
    """Recognize a disappearing owned process across the NVML teardown race.

    NVIDIA's process row can outlive ``/proc`` briefly.  Accept that row only
    when the process was previously observed as a live descendant and all
    metadata still reported by ``ps`` and NVML matches.  The caller records
    every use of this fallback separately.
    """

    if (
        process.get("start_time_ticks") is not None
        or process.get("ppid") is None
        or process.get("user") is None
        or process.get("args") is None
    ):
        return False
    if process_metadata_identity(process) in owned_metadata:
        return True
    # ``ps`` can expose the descendant as a zombie in the same sampling pass
    # in which reading its /proc identity fails.  NVML may replace its process
    # name with "[No data]" or may retain the exact prior executable name.
    # Require the explicit zombie markers and correlate PID, parent PID, user,
    # NVML name, and Linux's 15-character command name with a previously
    # observed live descendant.
    if not (
        str(process.get("args")).startswith("[")
        and str(process.get("args")).endswith("] <defunct>")
    ):
        return False
    zombie_args = str(process.get("args"))
    zombie_comm = zombie_args[1 : -len("] <defunct>")]
    pid = process.get("pid")
    ppid = process.get("ppid")
    user = process.get("user")
    return any(
        owned_pid == pid
        and owned_ppid == ppid
        and owned_user == user
        and process.get("process_name") in {"[No data]", owned_name}
        and zombie_comm
        in {
            Path(str(owned_name)).name[:15],
            Path(str(owned_args).split(None, 1)[0]).name[:15],
        }
        for owned_pid, owned_ppid, owned_user, owned_name, owned_args in owned_metadata
    )


def is_unresolved_owned_teardown_process(
    process: dict[str, Any],
    owned_identities: set[tuple[int, int]],
    owned_metadata: set[tuple[Any, ...]] | None = None,
) -> bool:
    """Defer an identity-free NVML row for an already proved owned PID.

    A telemetry query can begin while a retained child is still observable,
    then resolve its compute-process row after the harness has reaped that
    child.  NVML reports only the PID and either ``[No data]`` or the exact
    executable name retained from its earlier live row in this narrow race.
    This is not accepted as a live exact identity: it is recorded separately
    and the run still fails closed if any GPU process remains in the final
    post-run sample.  Unknown PIDs and rows with a new name or conflicting
    live metadata remain foreign immediately.
    """

    pid = process.get("pid")
    retained_names = {
        name
        for owned_pid, _ppid, _user, name, _args in (owned_metadata or set())
        if owned_pid == pid
    }
    return (
        pid in {owned_pid for owned_pid, _ in owned_identities}
        and process.get("start_time_ticks") is None
        and process.get("ppid") is None
        and process.get("user") is None
        and process.get("process_name") in {"[No data]", *retained_names}
        and process.get("args") is None
    )


def parse_metric_row(row: str) -> dict[str, Any]:
    names = (
        "timestamp",
        "index",
        "uuid",
        "pstate",
        "temperature_c",
        "gpu_utilization_pct",
        "memory_utilization_pct",
        "memory_used_mib",
        "sm_clock_mhz",
        "memory_clock_mhz",
        "power_draw_w",
    )
    values = [value.strip() for value in row.split(",")]
    if len(values) != len(names):
        raise RuntimeError(f"malformed NVIDIA GPU metric row: {row!r}")
    parsed: dict[str, Any] = dict(zip(names, values))
    for name in names[4:]:
        try:
            parsed[name] = float(parsed[name])
        except (KeyError, ValueError):
            parsed[name] = None
    return parsed


def summarize_metrics(samples: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {"sample_count": len(samples)}
    for field in (
        "temperature_c",
        "gpu_utilization_pct",
        "memory_utilization_pct",
        "memory_used_mib",
        "sm_clock_mhz",
        "memory_clock_mhz",
        "power_draw_w",
    ):
        values = [sample[field] for sample in samples if sample.get(field) is not None]
        if values:
            summary[field] = {
                "min": min(values),
                "median": statistics.median(values),
                "max": max(values),
            }
    return summary


def telemetry_status(
    returncode: int,
    *,
    sampling_failed: bool,
    samples_present: bool,
    post_processes_present: bool,
    contention: bool,
) -> tuple[str, int]:
    """Classify a run, giving every incomplete observation fatal precedence."""

    if sampling_failed:
        return "invalid_sampling_error", INVALID_EXIT
    if not samples_present:
        return "invalid_no_samples", INVALID_EXIT
    if post_processes_present:
        return "invalid_post_processes", INVALID_EXIT
    if contention:
        return "invalid_contended", INVALID_EXIT
    if returncode == 0:
        return "complete", 0
    return "child_failed", returncode


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu-index", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--interval", type=float, default=1.0)
    parser.add_argument('--cohort-root', help='Correctness tests only: permit peers descended from this PID:START_TICKS')
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("a command is required after --")
    if args.interval <= 0:
        parser.error("--interval must be positive")

    peers = CohortPeers(args.cohort_root)
    identity = gpu_identity(args.gpu_index)
    preexisting = compute_processes(identity["uuid"])
    record: dict[str, Any] = {
        "hostname": socket.gethostname(),
        "runner_user": getpass.getuser(),
        "gpu": identity,
        "command": command,
        "interval_s": args.interval,
        "started_unix_s": time.time(),
        "preexisting_processes": preexisting,
        "samples": [],
        "process_observations": [],
        "foreign_processes": [],
        "concurrency_policy": 'exclusive' if peers.root is None else 'owned_correctness_cohort',
        "cohort_root": peers.root,
        "peer_processes": [],
    }
    args.output = args.output.expanduser().resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if any(not peers.accepts(process) for process in preexisting):
        record["status"] = "invalid_preexisting_process"
        record["exit_code"] = INVALID_EXIT
        args.output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
        print(f"GPU {args.gpu_index} is not idle: {preexisting}", file=sys.stderr)
        raise SystemExit(INVALID_EXIT)

    child = subprocess.Popen(command, start_new_session=True)
    stopped = threading.Event()
    contention = threading.Event()
    sampling_failed = threading.Event()
    observed: set[tuple[int, str | None, str]] = set()
    owned_identities: set[tuple[int, int]] = set()
    owned_metadata: set[tuple[Any, ...]] = set()
    child_start_time = process_start_time_ticks(child.pid)
    if child_start_time is None:
        child.terminate()
        child.wait()
        raise RuntimeError("cannot establish the benchmark child process identity")
    owned_identities.add((child.pid, child_start_time))

    def sample() -> None:
        while not stopped.is_set():
            sampled_at = time.time()
            try:
                if peers.root is not None and not peers.live():
                    raise RuntimeError('correctness cohort root identity disappeared')
                rows = run_nvidia_smi(
                    [
                        f"--id={args.gpu_index}",
                        f"--query-gpu={GPU_FIELDS}",
                        "--format=csv,noheader,nounits",
                    ]
                )
                if len(rows) != 1:
                    raise RuntimeError(
                        f"expected one GPU metric row, got {rows!r}"
                    )
                metric = parse_metric_row(rows[0])
                metric["sampled_unix_s"] = sampled_at
                record["samples"].append(metric)
                for process in compute_processes(identity["uuid"]):
                    key = (process["pid"], process["user"], process["process_name"])
                    if key not in observed:
                        observed.add(key)
                        observation = {"sampled_unix_s": sampled_at, **process}
                        record["process_observations"].append(observation)
                    descendant = is_descendant(process["pid"], child.pid)
                    owned_session = is_owned_process_session(process, child.pid)
                    if descendant or owned_session:
                        start_time = process.get("start_time_ticks")
                        if isinstance(start_time, int):
                            owned_identities.add((process["pid"], start_time))
                            owned_metadata.add(process_metadata_identity(process))
                    exact_owned = is_exact_owned_process(
                        process, owned_identities
                    )
                    stale_owned = is_previously_observed_owned_process(
                        process, owned_metadata
                    )
                    if stale_owned:
                        stale = {"sampled_unix_s": sampled_at, **process}
                        if stale not in record.setdefault(
                            "stale_owned_processes", []
                        ):
                            record["stale_owned_processes"].append(stale)
                    unresolved_owned = is_unresolved_owned_teardown_process(
                        process, owned_identities, owned_metadata
                    )
                    if unresolved_owned:
                        unresolved = {"sampled_unix_s": sampled_at, **process}
                        if unresolved not in record.setdefault(
                            "unresolved_owned_teardown_processes", []
                        ):
                            record["unresolved_owned_teardown_processes"].append(
                                unresolved
                            )
                    if (
                        not descendant
                        and not owned_session
                        and not exact_owned
                        and not stale_owned
                        and not unresolved_owned
                    ):
                        if peers.accepts(process, child.pid):
                            record['peer_processes'].append({'sampled_unix_s': sampled_at, **process})
                            continue
                        if key not in {
                            (entry["pid"], entry["user"], entry["process_name"])
                            for entry in record["foreign_processes"]
                        }:
                            record["foreign_processes"].append(
                                {"sampled_unix_s": sampled_at, **process}
                            )
                        contention.set()
                        try:
                            os.killpg(child.pid, signal.SIGTERM)
                        except ProcessLookupError:
                            pass
            except (
                OSError,
                RuntimeError,
                subprocess.CalledProcessError,
                subprocess.TimeoutExpired,
            ) as error:
                record.setdefault("sampling_errors", []).append(repr(error))
                sampling_failed.set()
                try:
                    os.killpg(child.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
            stopped.wait(args.interval)

    sampler = threading.Thread(target=sample, name="gpu-telemetry", daemon=True)
    sampler.start()
    try:
        returncode = wait_for_identifiable_exit(child, identity['uuid'],
            quiescence_s=NVIDIA_SMI_TIMEOUT_S + 0.25 if peers.root is not None else 0.0)
    except (OSError, RuntimeError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
        record.setdefault('sampling_errors', []).append(f'identity-preserving teardown: {error!r}')
        sampling_failed.set()
        returncode = child.wait()
    stopped.set()
    sampler.join(timeout=NVIDIA_SMI_TIMEOUT_S + args.interval + 1.0)
    if sampler.is_alive():
        record.setdefault("sampling_errors", []).append(
            "telemetry sampler did not stop after the nvidia-smi timeout"
        )
        sampling_failed.set()
        sampler.join()
    record["finished_unix_s"] = time.time()
    record["child_returncode"] = returncode
    try:
        record["post_processes"] = compute_processes(identity["uuid"])
    except (
        OSError,
        RuntimeError,
        subprocess.CalledProcessError,
        subprocess.TimeoutExpired,
    ) as error:
        record["post_processes"] = []
        record.setdefault("sampling_errors", []).append(repr(error))
        sampling_failed.set()
    post_foreign = [
        process
        for process in record["post_processes"]
        if not is_owned_process_session(process, child.pid)
        and not peers.accepts(process, child.pid)
        and not is_exact_owned_process(process, owned_identities)
        and not is_unresolved_owned_teardown_process(
            process, owned_identities, owned_metadata
        )
    ]
    if post_foreign:
        record["foreign_processes"].extend(
            {"sampled_unix_s": record["finished_unix_s"], **process}
            for process in post_foreign
        )
        contention.set()
    post_processes_present = any(not peers.accepts(process, child.pid) for process in record['post_processes'])
    record["metric_summary"] = summarize_metrics(record["samples"])
    record["status"], record["exit_code"] = telemetry_status(
        returncode,
        sampling_failed=sampling_failed.is_set(),
        samples_present=bool(record["samples"]),
        post_processes_present=post_processes_present,
        contention=contention.is_set(),
    )
    args.output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(
        f"GPU_TELEMETRY status={record['status']} samples={len(record['samples'])} "
        f"foreign={len(record['foreign_processes'])}",
        flush=True,
    )
    raise SystemExit(record["exit_code"])


if __name__ == "__main__":
    main()
