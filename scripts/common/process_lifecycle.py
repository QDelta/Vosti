"""Process-lifecycle helpers for fail-closed GPU telemetry."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import time
from typing import Callable


NVML_RELEASE_TIMEOUT_S = 15.0
NVML_RELEASE_POLL_S = 0.05
NVML_INFLIGHT_SETTLE_S = 10.25


def detached(command, log_path, *, cwd, env):
    """The child owns no terminal/pipe descriptors, even after its launcher exits."""
    with Path(log_path).open('x') as log:
        return subprocess.Popen(command, cwd=cwd, env=env,
            stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True, close_fds=True)


def compute_process_pids() -> set[int]:
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=10.0,
    )
    pids: set[int] = set()
    for line in completed.stdout.splitlines():
        value = line.strip()
        if not value:
            continue
        try:
            pids.add(int(value))
        except ValueError as error:
            raise RuntimeError(
                f"malformed NVIDIA compute-process PID: {value!r}"
            ) from error
    return pids


def wait_for_child_exit_without_reaping(
    pid: int,
    *,
    timeout_s: float = NVML_RELEASE_TIMEOUT_S,
    poll_s: float = NVML_RELEASE_POLL_S,
) -> None:
    """Wait for a direct child while retaining its observable /proc identity."""

    deadline = time.monotonic() + timeout_s
    while True:
        try:
            result = os.waitid(
                os.P_PID,
                pid,
                os.WEXITED | os.WNOWAIT | os.WNOHANG,
            )
        except ChildProcessError as error:
            raise RuntimeError(
                f"GPU child PID {pid} was reaped before telemetry release"
            ) from error
        if result is not None:
            return
        if time.monotonic() >= deadline:
            raise RuntimeError(f"GPU child PID {pid} did not exit after shutdown")
        time.sleep(poll_s)


def wait_for_nvml_release(
    pid: int,
    *,
    compute_pids: Callable[[], set[int]] = compute_process_pids,
    timeout_s: float = NVML_RELEASE_TIMEOUT_S,
    poll_s: float = NVML_RELEASE_POLL_S,
    settle_s: float = NVML_INFLIGHT_SETTLE_S,
    owner: str = "GPU",
) -> None:
    """Keep an exited child identifiable until NVML has stably dropped its PID."""

    deadline = time.monotonic() + timeout_s
    while True:
        if pid not in compute_pids():
            # A concurrent monitor may already have captured the old NVML row
            # and still be resolving /proc metadata. Retain the zombie longer
            # than that monitor's nvidia-smi timeout, then confirm monotonic
            # release before making the PID reusable.
            time.sleep(settle_s)
            if pid not in compute_pids():
                return
        if time.monotonic() >= deadline:
            raise RuntimeError(f"NVML retained exited {owner} child PID {pid}")
        time.sleep(poll_s)
