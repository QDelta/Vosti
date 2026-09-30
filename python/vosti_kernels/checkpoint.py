"""Shared fail-closed safetensor inspection and materialization mechanics.

Families supply exact expected names/shapes and the language-model prefix.
This module neither selects kernels nor creates proof permissions.
"""

# @kernel-bridge-begin vosti_kernels::checkpoint_io
import json
from pathlib import Path

import torch
from safetensors import safe_open


def weight_map(model_path: Path, *, label: str) -> dict[str, str]:
    index = model_path / "model.safetensors.index.json"
    single = model_path / "model.safetensors"
    if index.is_file():
        payload = json.loads(index.read_text())
        mapping = payload.get("weight_map") if isinstance(payload, dict) else None
        if not isinstance(mapping, dict) or not mapping or any(
            not isinstance(k, str) or not isinstance(v, str) for k, v in mapping.items()
        ):
            raise ValueError(f"{label} safetensors index has invalid weight_map")
        return mapping
    if single.is_file():
        with safe_open(single, framework="pt", device="cpu") as handle:
            return {key: single.name for key in handle.keys()}
    raise FileNotFoundError(f"{label} checkpoint has no safetensors weights")


def resolve_shards(model_path: Path, mapping: dict[str, str], *, label: str) -> dict[str, Path]:
    root = model_path.resolve()
    result = {}
    for name in set(mapping.values()):
        path = (root / name).resolve()
        if path.parent != root or not path.is_file():
            raise ValueError(f"{label} checkpoint shard is invalid: {name}")
        result[name] = path
    return result

def validate_layout(
    model_path: Path, expected: dict[str, tuple[int, ...]], *, text_prefix: str, label: str,
) -> None:
    mapping = weight_map(model_path, label=label)
    actual = {key for key in mapping if key.startswith(text_prefix)}
    if actual != set(expected):
        raise ValueError(f"{label} text checkpoint key mismatch: "
                         f"missing={sorted(set(expected)-actual)}, extra={sorted(actual-set(expected))}")
    shards = resolve_shards(model_path, mapping, label=label)
    grouped: dict[str, list[str]] = {}
    for key in expected:
        grouped.setdefault(mapping[key], []).append(key)
    for name, keys in grouped.items():
        with safe_open(shards[name], framework="pt", device="cpu") as handle:
            for key in keys:
                view = handle.get_slice(key)
                shape = tuple(view.get_shape())
                if shape != expected[key]:
                    raise ValueError(f"{label} weight {key} has shape {shape}, expected {expected[key]}")
                if view.get_dtype() != "BF16":
                    raise ValueError(f"{label} weight {key} must be BF16, got {view.get_dtype()}")


def load_tensors(
    model_path: Path, keys: set[str], *, device: str | torch.device, label: str,
) -> dict[str, torch.Tensor]:
    mapping = weight_map(model_path, label=label)
    shards = resolve_shards(model_path, mapping, label=label)
    grouped: dict[str, list[str]] = {}
    for key in keys:
        grouped.setdefault(mapping[key], []).append(key)
    result = {}
    for name, shard_keys in grouped.items():
        with safe_open(shards[name], framework="pt", device=str(device)) as handle:
            for key in shard_keys:
                result[key] = handle.get_tensor(key).contiguous()
    return result
# @kernel-bridge-end vosti_kernels::checkpoint_io
