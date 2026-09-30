"""Shared fail-closed helpers for benchmark GPU telemetry records."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]


def monitored_trial_command(
    binary: Path,
    *,
    gpu_index: int,
    telemetry_output: Path,
) -> list[str]:
    """Run a GPU-owning binary as the telemetry monitor's direct child."""

    return [
        sys.executable,
        str(ROOT / "scripts/common/gpu_monitor.py"),
        "--gpu-index",
        str(gpu_index),
        "--output",
        str(telemetry_output),
        "--",
        str(binary),
    ]


def load_complete_telemetry(path: Path, *, allow_cohort: bool = False) -> dict:
    """Load one complete record and bind its summary to the exact bytes."""

    data = path.read_bytes()
    record = json.loads(data)
    if record.get("status") != "complete":
        raise RuntimeError(
            f"GPU telemetry rejected {path}: status={record.get('status')!r}"
        )
    if record.get('concurrency_policy', 'exclusive') != 'exclusive' and not allow_cohort:
        raise RuntimeError('shared-GPU correctness telemetry cannot validate an exclusive performance run')
    return {
        "gpu": record["gpu"],
        "metric_summary": record["metric_summary"],
        "path": str(path),
        "sha256": hashlib.sha256(data).hexdigest(),
        "status": record["status"],
    }


__all__ = ["load_complete_telemetry", "monitored_trial_command"]
