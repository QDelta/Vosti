"""Qwen3 adapter for the shared qualified dense runtime capability."""

from __future__ import annotations

import os
from typing import Any

import torch

from ...dense_runtime import (
    DenseRuntimeFamily,
    QualifiedRuntime as _QualifiedRuntime,
    Runtime as _Runtime,
)
from . import deployment as _deployment
from .profile import (
    SCOPE_SCHEMA,
    SCOPE_STATUS,
    model_config,
    model_profile_for_config,
    model_profile_for_name,
    scope,
)


class Runtime(_Runtime):
    """Qwen3-typed adapter over the shared runtime implementation."""


# @kernel-bridge-begin vosti_kernels::qwen3_runtime_capability
_FAMILY = DenseRuntimeFamily(
    architecture="qwen3",
    label="Qwen3",
    family_module="qwen3",
    scope_schema=SCOPE_SCHEMA,
    scope_status=SCOPE_STATUS,
    scope=scope(),
    model_config=model_config,
    runtime_model_config=model_config,
    model_profile_for_config=model_profile_for_config,
    model_profile_for_name=model_profile_for_name,
    deployment=_deployment,
    runtime_type=Runtime,
)

PAGE_SIZE = _FAMILY.page_size
VERIFIED_SCOPE = _FAMILY.scope
VERIFIED_MODEL_SHAPES = _FAMILY.verified_model_shapes
VERIFIED_RUNTIME = _FAMILY.verified_runtime
ENGINE_KERNEL_CONTRACTS = _FAMILY.kernel_contracts
CONTRACTS_BY_WRAPPER = _FAMILY.contracts_by_wrapper
ENGINE_KERNEL_MODULES = _FAMILY.kernel_modules
ENGINE_MODULE_SOURCES = _FAMILY.module_sources


class QualifiedRuntime(_QualifiedRuntime):
    """Backend-qualified Qwen3 capability reserved for the Rust boundary."""

    def __init__(self, runtime: Runtime) -> None:
        super().__init__(runtime, family=_FAMILY)


def config_for_profile(profile_name: str | None = None) -> dict[str, Any]:
    return _FAMILY.config_for_profile(profile_name)


def config_from_bundle(
    deployment_bundle: str | os.PathLike[str],
) -> dict[str, Any]:
    return _FAMILY.config_from_bundle(deployment_bundle)


def load_runtime(config: dict[str, Any], **kwargs) -> Runtime:
    return _FAMILY.load_runtime(config, **kwargs)


def load_qualified_runtime(config: dict[str, Any], **kwargs) -> QualifiedRuntime:
    return QualifiedRuntime(load_runtime(config, **kwargs))


def runtime_for_tests(
    *,
    hidden_size: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    intermediate_size: int | None = None,
    rms_norm_eps: float,
    rope_theta: float = 1000000.0,
    device: str | None = None,
    dtype: torch.dtype = torch.float32,
) -> Runtime:
    return _FAMILY.runtime_for_tests(
        {
            "hidden_size": int(hidden_size),
            "num_heads": int(num_heads),
            "num_kv_heads": int(num_kv_heads),
            "head_dim": int(head_dim),
            "intermediate_size": (
                int(intermediate_size)
                if intermediate_size is not None
                else None
            ),
            "rms_norm_eps": float(rms_norm_eps),
            "rope_theta": float(rope_theta),
            "rope_scaling_kind": "none",
            "device": device or "cpu",
            "dtype": dtype,
        }
    )
# @kernel-bridge-end vosti_kernels::qwen3_runtime_capability


__all__ = [
    "PAGE_SIZE",
    "VERIFIED_SCOPE",
    "VERIFIED_MODEL_SHAPES",
    "VERIFIED_RUNTIME",
    "ENGINE_KERNEL_CONTRACTS",
    "CONTRACTS_BY_WRAPPER",
    "ENGINE_KERNEL_MODULES",
    "ENGINE_MODULE_SOURCES",
    "QualifiedRuntime",
    "Runtime",
    "config_for_profile",
    "config_from_bundle",
    "load_qualified_runtime",
    "load_runtime",
    "runtime_for_tests",
]
