"""Intrusive, test-only capture of model K/V rows at cache scatter.

The serving path does not import this module.  The reference harness installs
an import hook through ``sitecustomize`` and wraps the checked Engine's KV
scatter entry point.  Every record contains both the rows supplied to the
scatter and the rows read back from their physical cache slots.  Arrays are
copied to CPU as float32 solely to make BF16 values portable through NumPy.
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


def _float32_array(tensor: torch.Tensor) -> np.ndarray:
    return (
        tensor.detach()
        .to(dtype=torch.float32, device="cpu")
        .contiguous()
        .numpy()
    )


def _write_array(directory: Path, stem: str, array: np.ndarray) -> dict[str, Any]:
    filename = f"{stem}.npy"
    path = directory / filename
    with path.open("xb") as handle:
        np.save(handle, array, allow_pickle=False)
    raw = array.tobytes(order="C")
    return {
        "artifact": filename,
        "dtype": str(array.dtype),
        "shape": list(array.shape),
        "numel": int(array.size),
        "byte_length": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "finite": bool(np.isfinite(array).all()),
    }


def observe_kv_store(
    k: torch.Tensor,
    v: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    *,
    source: str,
) -> None:
    """Record active K/V inputs and their post-scatter physical cache rows."""

    directory_raw = os.environ.get("VOSTI_KV_OBSERVER_DIR")
    if not directory_raw:
        raise RuntimeError("KV observer has no VOSTI_KV_OBSERVER_DIR")
    directory = Path(directory_raw).resolve()
    directory.mkdir(parents=True, exist_ok=True)

    if k.shape != v.shape or k.dim() != 3:
        raise RuntimeError("KV observer requires matching [rows, heads, dim] tensors")
    if k_cache.shape != v_cache.shape or k_cache.dim() != 4:
        raise RuntimeError("KV observer requires matching paged rank-4 caches")
    if slot_mapping.dim() != 1 or int(slot_mapping.numel()) != int(k.shape[0]):
        raise RuntimeError("KV observer requires one slot per K/V row")

    slots = slot_mapping.detach().to(dtype=torch.int64, device="cpu")
    active_mask = slots >= 0
    active_slots = slots[active_mask]
    if active_slots.numel() == 0:
        return
    if int(active_slots.max().item()) >= int(k_cache.shape[0] * k_cache.shape[1]):
        raise RuntimeError("KV observer saw an out-of-range active cache slot")
    if int(torch.unique(active_slots).numel()) != int(active_slots.numel()):
        raise RuntimeError("KV observer requires injective active slots")

    device_mask = active_mask.to(device=k.device)
    device_slots = active_slots.to(device=k_cache.device)
    input_k = _float32_array(k[device_mask])
    input_v = _float32_array(v[device_mask])
    flat_tail = tuple(int(dimension) for dimension in k_cache.shape[2:])
    stored_k = _float32_array(k_cache.reshape(-1, *flat_tail)[device_slots])
    stored_v = _float32_array(v_cache.reshape(-1, *flat_tail)[device_slots])
    slot_array = active_slots.contiguous().numpy()

    sequence = _next_call_sequence()
    pid = os.getpid()
    prefix = f"kv-{pid}-{sequence:06d}"
    record = {
        "schema_version": 1,
        "pid": pid,
        "call_sequence": sequence,
        "source": source,
        "source_dtype": str(k.dtype),
        "input_k": _write_array(directory, f"{prefix}-input-k", input_k),
        "input_v": _write_array(directory, f"{prefix}-input-v", input_v),
        "stored_k": _write_array(directory, f"{prefix}-stored-k", stored_k),
        "stored_v": _write_array(directory, f"{prefix}-stored-v", stored_v),
        "slots": _write_array(directory, f"{prefix}-slots", slot_array),
    }
    index = directory / f"observer-{pid}.jsonl"
    encoded = json.dumps(record, sort_keys=True, separators=(",", ":"))
    with _LOCK, index.open("a", encoding="utf-8") as handle:
        handle.write(encoded + "\n")


def load_kv_observer_records(directory: Path) -> list[dict[str, Any]]:
    """Load one process's ordered KV records from a completed observation."""

    indexes = sorted(directory.glob("observer-*.jsonl"))
    records = []
    for index in indexes:
        for line in index.read_text(encoding="utf-8").splitlines():
            records.append(json.loads(line))
    records.sort(key=lambda record: (int(record["pid"]), int(record["call_sequence"])))
    return records


__all__ = ["load_kv_observer_records", "observe_kv_store"]
