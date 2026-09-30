#!/usr/bin/env python3
"""CUDA falsifier for the shared qualified KV-scatter effect."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import os
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT, ROOT / "python"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from vosti_kernels import kernels
from vosti_kernels import physical
from scripts.common.model_paths import checkpoint_path

FAMILIES = {
    "qwen3": {
        "architecture": "qwen3",
        "default_model": checkpoint_path("Qwen3-0.6B"),
        "loader_module": "vosti_kernels.model_families.qwen3.loader",
        "runtime_module": "vosti_kernels.model_families.qwen3.runtime",
    },
    "gemma3": {
        "architecture": "gemma3_text",
        "default_model": checkpoint_path("gemma-3-4b-it"),
        "loader_module": "vosti_kernels.model_families.gemma3.loader",
        "runtime_module": "vosti_kernels.model_families.gemma3.runtime",
    },
    "llama3": {
        "architecture": "llama3",
        "default_model": checkpoint_path("Llama-3.1-8B"),
        "loader_module": "vosti_kernels.model_families.llama3.loader",
        "runtime_module": "vosti_kernels.model_families.llama3.runtime",
    },
    "gemma4": {
        "architecture": "gemma4_text",
        "default_model": checkpoint_path("gemma-4-31b-it"),
        "loader_module": "vosti_kernels.model_families.gemma4.loader",
        "runtime_module": "vosti_kernels.model_families.gemma4.runtime",
    },
}

NUM_PAGES = 8


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _normalize_config(config: object) -> dict[str, Any]:
    if isinstance(config, dict):
        return dict(config)
    as_runtime_dict = getattr(config, "as_runtime_dict", None)
    if callable(as_runtime_dict):
        resolved = as_runtime_dict()
        if isinstance(resolved, dict):
            return dict(resolved)
    raise TypeError("model checkpoint inspector returned no runtime config")


def _validate_runtime_report(report: object, architecture: str) -> None:
    if not isinstance(report, dict):
        raise RuntimeError("qualified runtime report is not a dictionary")
    if report.get("architecture") != architecture:
        raise RuntimeError("qualified runtime reported the wrong architecture")
    if report.get("engine_reachable") is not False:
        raise RuntimeError("raw primitive runtime unexpectedly claims Engine authority")
    if report.get("backend_qualified") is not True:
        raise RuntimeError("KV-store runtime is not backend-qualified")
    if not _is_sha256(report.get("deployment_sha256")):
        raise RuntimeError("qualified runtime report has no deployment identity")
    qualification = report.get("qualification")
    if not isinstance(qualification, dict):
        raise RuntimeError("qualified runtime report has no qualification record")
    for field in (
        "candidate_sha256",
        "qualification_report_sha256",
        "deployment_sha256",
    ):
        if not _is_sha256(qualification.get(field)):
            raise RuntimeError(f"qualification record has no valid {field}")

    origins = report.get("module_origins")
    digests = report.get("module_source_sha256")
    if not isinstance(origins, dict) or not origins:
        raise RuntimeError("qualified runtime report has no module origins")
    if not isinstance(digests, dict) or set(digests) != set(origins):
        raise RuntimeError("qualified runtime module origins and digests disagree")
    for name, origin in origins.items():
        if not isinstance(name, str) or not isinstance(origin, str) or not origin:
            raise RuntimeError("qualified runtime report has an invalid module origin")
        if not _is_sha256(digests[name]):
            raise RuntimeError(f"qualified runtime omits the source digest for {name}")


def _load_bound_runtime(family_name: str, device: torch.device):
    family = FAMILIES[family_name]
    bundle_path = os.environ.get("VOSTI_DEPLOYMENT_BUNDLE")
    if not bundle_path:
        raise RuntimeError("KV-store qualification requires VOSTI_DEPLOYMENT_BUNDLE")
    framework_root = os.environ.get("VOSTI_FRAMEWORK_ROOT")
    if not framework_root:
        raise RuntimeError("KV-store qualification requires VOSTI_FRAMEWORK_ROOT")
    model_path = Path(
        os.environ.get("MODEL_PATH", str(family["default_model"]))
    ).expanduser().resolve()
    raw_config = (model_path / "config.json").read_bytes()

    loader = importlib.import_module(str(family["loader_module"]))
    resolved_config = _normalize_config(loader.inspect_text_checkpoint(model_path))
    runtime_module = importlib.import_module(str(family["runtime_module"]))
    runtime = runtime_module.load_qualified_runtime(
        resolved_config,
        deployment_bundle=bundle_path,
        model_config_sha256=hashlib.sha256(raw_config).hexdigest(),
        framework_root=framework_root,
        device=str(device),
        dtype=torch.bfloat16,
    )
    _validate_runtime_report(runtime.report(), str(family["architecture"]))
    return runtime


def _runtime_geometry(runtime, device: torch.device) -> tuple[int, int]:
    config = runtime.runtime_config()
    num_kv_heads = int(config["num_kv_heads"])
    head_dim = int(config["head_dim"])
    if num_kv_heads <= 0 or head_dim <= 0:
        raise RuntimeError("qualified runtime has invalid KV geometry")
    if config["dtype"] is not torch.bfloat16:
        raise RuntimeError("qualified runtime did not bind bfloat16")
    if torch.device(config["device"]) != device:
        raise RuntimeError("qualified runtime did not bind the selected CUDA device")
    runtime.static_launch_config(
        "store_kv_cache", {"kvd": num_kv_heads * head_dim}
    )
    return num_kv_heads, head_dim


def _check_case(
    slots: torch.Tensor,
    runtime,
    *,
    num_kv_heads: int,
    head_dim: int,
) -> None:
    rows = int(slots.numel())
    device = slots.device
    page_size = physical.PAGE_SIZE
    k = torch.randn(
        (rows, num_kv_heads, head_dim), device=device, dtype=torch.bfloat16
    )
    v = torch.randn_like(k)
    old_k = torch.randn(
        (NUM_PAGES, page_size, num_kv_heads, head_dim),
        device=device,
        dtype=torch.bfloat16,
    )
    old_v = torch.randn_like(old_k)
    actual_k, actual_v = old_k.clone(), old_v.clone()
    expected_k, expected_v = old_k.clone(), old_v.clone()
    flat_width = num_kv_heads * head_dim
    indices = slots.to(torch.int64)
    expected_k.view(-1, flat_width).index_copy_(0, indices, k.view(rows, -1))
    expected_v.view(-1, flat_width).index_copy_(0, indices, v.view(rows, -1))
    k_pointer, v_pointer = actual_k.data_ptr(), actual_v.data_ptr()

    kernels.store_kv_cache(k, v, actual_k, actual_v, slots, runtime)
    torch.cuda.synchronize(device)

    assert actual_k.data_ptr() == k_pointer
    assert actual_v.data_ptr() == v_pointer
    assert torch.equal(actual_k, expected_k)
    assert torch.equal(actual_v, expected_v)

    written = torch.zeros(NUM_PAGES * page_size, device=device, dtype=torch.bool)
    written[indices] = True
    assert torch.equal(
        actual_k.view(-1, flat_width)[~written],
        old_k.view(-1, flat_width)[~written],
    )
    assert torch.equal(
        actual_v.view(-1, flat_width)[~written],
        old_v.view(-1, flat_width)[~written],
    )

    if rows <= 4:
        sequential_k, sequential_v = old_k.clone(), old_v.clone()
        for index in range(rows):
            kernels.store_kv_cache(
                k[index : index + 1],
                v[index : index + 1],
                sequential_k,
                sequential_v,
                slots[index : index + 1],
                runtime,
            )
        torch.cuda.synchronize(device)
        assert torch.equal(actual_k, sequential_k)
        assert torch.equal(actual_v, sequential_v)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("family", choices=tuple(FAMILIES))
    args = parser.parse_args()

    torch.manual_seed(20260816)
    device = torch.device(os.environ.get("CUDA_DEVICE", "cuda:0"))
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError(f"KV-store qualification requires CUDA, got {device}")
    torch.cuda.set_device(device)
    runtime = _load_bound_runtime(args.family, device)
    num_kv_heads, head_dim = _runtime_geometry(runtime, device)

    num_slots = NUM_PAGES * physical.PAGE_SIZE
    cases = [
        torch.tensor([num_slots - 1], device=device, dtype=torch.int32),
        torch.tensor(
            [0, physical.PAGE_SIZE - 1, physical.PAGE_SIZE, num_slots - 1],
            device=device,
            dtype=torch.int32,
        ),
    ]
    for rows in (17, 65, 257):
        cases.append(
            torch.randperm(num_slots, device=device, dtype=torch.int32)[
                :rows
            ].contiguous()
        )
    for slots in cases:
        _check_case(
            slots,
            runtime,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
        )

    print(
        f"{args.family} KV-store exact-effect check passed: written rows copied "
        "bit-for-bit, unwritten rows preserved, cache identities stable, and "
        "batched/singleton stores agree"
    )


if __name__ == "__main__":
    main()
