"""Explicit capability protocol for architecture-neutral primitives.

Rust retains one qualified family runtime inside ``ModelRuntime`` and passes
that exact object to every shared primitive which needs kernel configuration.
There is deliberately no process-global binding or implicit family lookup.
Passing ``None`` cannot select a deployed kernel. Some primitives retain
CPU-only test fixtures; attention instead requires a verified binding and
rejects missing bindings without selecting a reference implementation.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import json
from types import MappingProxyType
from typing import Any, Protocol


@dataclass(frozen=True)
class PackedAttentionMetadata:
    """Architecture-neutral packed attention launch metadata."""

    positions: Any
    kv_caches: Any
    block_table: Any
    slot_mapping: Any
    cu_seqlens_q: Any
    cu_seqlens_k: Any
    max_seqlen_q: int
    max_seqlen_k: int


# @kernel-bridge-begin vosti_kernels::primitive_runtime_binding
def runtime_binding_identity(
    qualification: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Read the live binding identity without constructing a diagnostic report.

    Do not cache this projection: qualified capabilities compare it with the
    identity pinned at binding, including on subsequent primitive calls.
    """

    return {
        "engine_reachable": False,
        "backend_qualified": qualification is not None,
        "qualification": (
            {
                key: qualification[key]
                for key in (
                    "candidate_sha256",
                    "qualification_report_sha256",
                    "deployment_sha256",
                )
            }
            if qualification is not None
            else None
        ),
    }


class QualifiedPrimitiveRuntime:
    """Shared live-identity guard; Rust separately owns engine admission."""

    def __init__(self, runtime):
        report = runtime.report()
        if report.get("engine_reachable") is not False:
            raise ValueError("unqualified runtime must not claim engine authority")
        if report.get("backend_qualified") is not True:
            raise ValueError("qualified capability requires a sealed backend bundle")
        qualification = report.get("qualification")
        if not isinstance(qualification, dict):
            raise ValueError("qualified capability has no qualification identity")
        self._runtime = runtime
        self._qualification = MappingProxyType(json.loads(json.dumps(qualification)))

    def _checked_runtime(self):
        identity = self._runtime.binding_identity()
        if (identity.get("engine_reachable") is not False
                or identity.get("backend_qualified") is not True
                or identity.get("qualification") != self._qualification):
            raise RuntimeError("qualified runtime identity changed after binding")
        return self._runtime

    def report(self):
        return self._checked_runtime().report()

    def model_config(self):
        return self._checked_runtime().model_config()

    def runtime_config(self):
        return self._checked_runtime().runtime_config()

    def verified_for(self, tensor):
        return self._checked_runtime().verified_for(tensor)

    def static_launch_config(self, wrapper, key):
        return self._checked_runtime().static_launch_config(wrapper, key)

    def kernel_entrypoint(self, kernels, wrapper):
        return self._checked_runtime().kernel_entrypoint(kernels, wrapper)


class PrimitiveRuntime(Protocol):
    """Minimal explicit capability consumed by shared primitive bodies."""

    def runtime_config(self) -> Mapping[str, Any]: ...

    def verified_for(self, tensor: Any) -> Any | None: ...

    def static_launch_config(
        self, wrapper: str, key: dict[str, int]
    ) -> Mapping[str, Any]: ...

    def kernel_entrypoint(self, kernels: Any, wrapper: str) -> Any: ...


class RotaryRuntime(PrimitiveRuntime, Protocol):
    """Primitive capability with a sealed, runtime-owned RoPE table."""

    def rotary_tables(
        self, positions: Any, dtype: Any
    ) -> tuple[Any, Any]: ...


def runtime_config_or_none(
    runtime: PrimitiveRuntime | None,
) -> Mapping[str, Any] | None:
    if runtime is None:
        return None
    config = runtime.runtime_config()
    if not isinstance(config, Mapping):
        raise TypeError("primitive runtime configuration must be a mapping")
    return config


def runtime_config(runtime: PrimitiveRuntime) -> Mapping[str, Any]:
    config = runtime_config_or_none(runtime)
    if config is None:
        raise RuntimeError("an explicit family runtime is required")
    return config


def verified_for(runtime: PrimitiveRuntime | None, tensor: Any) -> Any | None:
    return None if runtime is None else runtime.verified_for(tensor)


def static_launch_config(
    runtime: PrimitiveRuntime,
    wrapper: str, key: dict[str, int]
) -> Mapping[str, Any]:
    return runtime.static_launch_config(wrapper, key)


def kernel_entrypoint(
    runtime: PrimitiveRuntime, kernels: Any, wrapper: str,
) -> Any:
    return runtime.kernel_entrypoint(kernels, wrapper)


def rotary_tables(
    runtime: RotaryRuntime, positions: Any, dtype: Any
) -> tuple[Any, Any]:
    return runtime.rotary_tables(positions, dtype)
# @kernel-bridge-end vosti_kernels::primitive_runtime_binding


__all__ = [
    "QualifiedPrimitiveRuntime",
    "PrimitiveRuntime",
    "RotaryRuntime",
    "PackedAttentionMetadata",
    "kernel_entrypoint",
    "rotary_tables",
    "runtime_binding_identity",
    "runtime_config",
    "runtime_config_or_none",
    "static_launch_config",
    "verified_for",
]
