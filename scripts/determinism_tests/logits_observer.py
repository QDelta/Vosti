"""Intrusive, test-only capture of full next-token logit rows.

This module belongs to the determinism test harness, not to the serving or
kernel packages.  Test-only import hooks call it at an engine's sampling
boundary.  Rows are copied to CPU and stored losslessly as float32 ``.npy``
files with a JSONL index.  This is correctness instrumentation, not a
performance measurement path.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import threading
from typing import Any

import numpy as np
import torch


_LOCK = threading.Lock()
_CALL_SEQUENCE = 0


def _next_call_sequence() -> int:
    global _CALL_SEQUENCE
    with _LOCK:
        sequence = _CALL_SEQUENCE
        _CALL_SEQUENCE += 1
    return sequence


def _write_digest(row_bytes: bytes, batch_index: int) -> None:
    """Write the optional Vosti two-column digest instrumentation."""

    path = os.environ.get("VOSTI_LOGITS_DIGEST")
    if not path:
        return
    digest = hashlib.sha256(row_bytes).hexdigest()
    with _LOCK, open(path, "a", encoding="utf-8") as handle:
        handle.write(f"{batch_index} {digest}\n")


def observe_logits_rows(
    rows: torch.Tensor,
    *,
    source: str,
    metadata: dict[str, Any] | None = None,
    batch_indices: list[int] | None = None,
    row_metadata: list[dict[str, Any]] | None = None,
) -> None:
    """Record every vocabulary row in one sampler call, when enabled."""

    if rows.dim() != 2:
        raise RuntimeError("logits observer requires a rank-2 tensor")
    call_sequence = _next_call_sequence()
    observer_dir_raw = os.environ.get("VOSTI_LOGITS_OBSERVER_DIR")
    observer_dir = Path(observer_dir_raw).resolve() if observer_dir_raw else None
    if observer_dir is not None:
        observer_dir.mkdir(parents=True, exist_ok=True)

    source_dtype = str(rows.dtype)
    batch_size = int(rows.shape[0])
    if batch_indices is None:
        batch_indices = list(range(batch_size))
    if len(batch_indices) != batch_size:
        raise RuntimeError("logits observer batch-index count does not match rows")
    if row_metadata is not None and len(row_metadata) != batch_size:
        raise RuntimeError("logits observer row-metadata count does not match rows")
    for row_offset, batch_index in enumerate(batch_indices):
        array = (
            rows[row_offset]
            .detach()
            .to(dtype=torch.float32, device="cpu")
            .contiguous()
            .numpy()
        )
        row_bytes = array.tobytes(order="C")
        _write_digest(row_bytes, batch_index)
        if observer_dir is None:
            continue

        pid = os.getpid()
        filename = f"row-{pid}-{call_sequence:06d}-{batch_index:04d}.npy"
        artifact = observer_dir / filename
        with artifact.open("xb") as handle:
            np.save(handle, array, allow_pickle=False)
        record: dict[str, Any] = {
            "schema_version": 1,
            "pid": pid,
            "call_sequence": call_sequence,
            "batch_index": batch_index,
            "row_offset": row_offset,
            "batch_size": batch_size,
            "source": source,
            "source_dtype": source_dtype,
            "observed_dtype": str(array.dtype),
            "shape": list(array.shape),
            "numel": int(array.size),
            "byte_length": len(row_bytes),
            "sha256": hashlib.sha256(row_bytes).hexdigest(),
            "finite": bool(np.isfinite(array).all()),
            "argmax": int(np.argmax(array)) if array.size else None,
            "artifact": filename,
        }
        phase = os.environ.get("VOSTI_LOGITS_OBSERVER_PHASE")
        if phase:
            record["phase"] = phase
        engine_step = os.environ.get("VOSTI_LOGITS_OBSERVER_STEP")
        if engine_step:
            record["engine_step"] = engine_step
        combined_metadata = dict(metadata or {})
        if row_metadata is not None:
            combined_metadata.update(row_metadata[row_offset])
        if combined_metadata:
            record["metadata"] = combined_metadata
        index = observer_dir / f"observer-{pid}.jsonl"
        encoded = json.dumps(record, sort_keys=True, separators=(",", ":"))
        with _LOCK, index.open("a", encoding="utf-8") as handle:
            handle.write(encoded + "\n")


def observe_logits_row(
    row: torch.Tensor,
    batch_index: int,
    *,
    source: str,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Record one selected row while preserving its external batch index."""

    if row.dim() != 1:
        raise RuntimeError("logits observer requires a rank-1 selected row")
    observe_logits_rows(
        row.unsqueeze(0),
        source=source,
        metadata=metadata,
        batch_indices=[int(batch_index)],
    )


__all__ = ["observe_logits_row", "observe_logits_rows"]
