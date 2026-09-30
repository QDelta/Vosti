"""Text-only Gemma 3 primitive runtime qualification.

This module is deliberately not exported from :mod:`vosti_kernels`. Its raw
runtime is not Engine-admissible; only the sealed qualified wrapper is
consumed by the Rust runtime capability:

* exact model profiles, each with its own closed static launch inventory;
* source and git identity attestation for every kernel it can call; and
* a fixed primitive surface consumed by the checked Rust composition.

The kernel certificates establish their stated decomposition properties, not
end-to-end numerical correctness.  Exact Python/Torch/CUDA primitive effects
remain conditional on backend qualification. End-to-end differential checks
use the common external model-reference harness. Memory-efficient sliding-
window KV eviction remains explicitly deferred.
"""

from __future__ import annotations

import os
from types import MappingProxyType
from typing import Any

import torch

from ... import rotary as ROTARY
from ...static_runtime import StaticPrimitiveRuntime, QualifiedPrimitiveRuntime, load_family_runtime
from ... import primitive_runtime as PRIMITIVE_RUNTIME
from .loader import (
    FULL_ATTENTION,
    SLIDING_ATTENTION,
)
from .profile import (
    model_config,
    model_profile_for_config,
    model_profile_for_name,
    scope,
)


# @kernel-bridge-begin vosti_kernels::gemma3_runtime_capability
_SCOPE = scope()


def config_for_profile(profile_name: str | None = None) -> dict[str, Any]:
    """Return one exact loader config from the profile registry.

    The omitted name remains valid while the registry has one profile.  Once a
    second profile is declared, callers must select explicitly or supply a
    resolved config to the runtime loader.
    """

    return model_config(model_profile_for_name(profile_name))


def _validate_model_config(config: dict[str, Any]) -> dict[str, Any]:
    return model_profile_for_config(config)


def config_from_bundle(
    deployment_bundle: str | os.PathLike[str],
) -> dict[str, Any]:
    """Recover the exact declared config selected by a sealed bundle."""

    from .deployment import load_bundle

    bundle = load_bundle(deployment_bundle)
    deployment_model = bundle["deployment"]["model"]
    config = deployment_model["resolved_config"]
    profile = _validate_model_config(config)
    if deployment_model["catalog_name"] != profile["model"]["name"]:
        raise ValueError("Gemma 3 bundle model name differs from its exact config")
    return model_config(profile)


class Runtime(StaticPrimitiveRuntime):
    """Source-attested static implementation of the primitive surface."""

    def __init__(self, modules, origins, digests, kernel_root, qualification, profile, device, dtype):
        super().__init__(scope=_SCOPE, config=model_config(profile), modules=modules,
            origins=origins, digests=digests, kernel_root=kernel_root,
            qualification=qualification, profile=profile, device=device, dtype=dtype)
        self._rope_table_cache = (
            MappingProxyType(
                {
                    attention_kind: ROTARY.PrecomputedTables(
                        *self._compute_rope_tables(
                            torch.arange(
                                int(self._config["max_position_embeddings"]),
                                dtype=torch.int64,
                                device=torch.device(self._device),
                            ),
                            attention_kind=attention_kind,
                            dtype=self._dtype,
                        )
                    )
                    for attention_kind in (SLIDING_ATTENTION, FULL_ATTENTION)
                }
            )
            if self._qualification is not None
            else None
        )


    def scaled_embed(self, input_ids: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        return self._modules["scaled_embedding"].scaled_embedding(
            input_ids,
            weight,
            launch_config=self._config_for("token_embedding"),
        )

    def rms_norm(
        self, x: torch.Tensor, weight: torch.Tensor, *, site: str
    ) -> torch.Tensor:
        return self._modules["gemma_rmsnorm"].gemma_rmsnorm(
            x,
            weight,
            float(self._config["rms_norm_eps"]),
            launch_config=self._config_for(site),
        )

    def qk_norm(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        q_weight: torch.Tensor,
        k_weight: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self._modules["gemma_qk_norm"].gemma_qk_norm(
            q,
            k,
            q_weight,
            k_weight,
            int(self._config["num_attention_heads"]),
            int(self._config["num_key_value_heads"]),
            float(self._config["rms_norm_eps"]),
            q_launch_config=self._config_for("q_norm"),
            k_launch_config=self._config_for("k_norm"),
        )

    def _compute_rope_tables(
        self,
        positions: torch.Tensor,
        *,
        attention_kind: str,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if positions.dim() != 1:
            raise ValueError("Gemma 3 packed positions must be rank-1")
        if attention_kind == SLIDING_ATTENTION:
            theta = float(self._config["local_rope_theta"])
            factor = 1.0
        elif attention_kind == FULL_ATTENTION:
            theta = float(self._config["global_rope_theta"])
            factor = float(self._config["global_rope_factor"])
        else:
            raise ValueError("Gemma 3 RoPE received an unsupported attention kind")
        head_dim = int(self._config["head_dim"])
        exponents = torch.arange(
            0,
            head_dim,
            2,
            dtype=torch.float32,
            device=positions.device,
        ) / head_dim
        inv_freq = (1.0 / torch.pow(theta, exponents)) / factor
        frequencies = positions.float()[:, None] * inv_freq[None, :]
        return frequencies.cos().to(dtype), frequencies.sin().to(dtype)

    def _rope_tables(
        self,
        positions: torch.Tensor,
        *,
        attention_kind: str,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self._rope_table_cache is None:
            return self._compute_rope_tables(
                positions,
                attention_kind=attention_kind,
                dtype=dtype,
            )
        if dtype != self._dtype:
            raise ValueError("Gemma 3 RoPE input dtype differs from sealed runtime")
        try:
            tables = self._rope_table_cache[attention_kind]
        except KeyError as error:
            raise ValueError(
                "Gemma 3 RoPE received an unsupported attention kind"
            ) from error
        return tables.select(positions)

    def rotary_embed(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        positions: torch.Tensor,
        *,
        attention_kind: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        rows = q.shape[0]
        q_heads = int(self._config["num_attention_heads"])
        kv_heads = int(self._config["num_key_value_heads"])
        head_dim = int(self._config["head_dim"])
        if tuple(q.shape) != (rows, q_heads * head_dim) or tuple(k.shape) != (
            rows,
            kv_heads * head_dim,
        ):
            raise ValueError("Gemma 3 rotary inputs disagree with head geometry")
        cos, sin = self._rope_tables(
            positions, attention_kind=attention_kind, dtype=q.dtype
        )
        rope = self._modules["rope"].rope
        q_out = rope(
            q.reshape(rows * q_heads, head_dim),
            cos.repeat_interleave(q_heads, dim=0),
            sin.repeat_interleave(q_heads, dim=0),
            launch_config=self._config_for("rotary_embed_q"),
        ).reshape(rows, q_heads, head_dim)
        k_out = rope(
            k.reshape(rows * kv_heads, head_dim),
            cos.repeat_interleave(kv_heads, dim=0),
            sin.repeat_interleave(kv_heads, dim=0),
            launch_config=self._config_for("rotary_embed_k"),
        ).reshape(rows, kv_heads, head_dim)
        return q_out, k_out

    def paged_attention(
        self,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        step: PRIMITIVE_RUNTIME.PackedAttentionMetadata,
        *,
        attention_kind: str,
        value_checks: bool = True,
    ) -> torch.Tensor:
        common = {
            "softmax_scale": float(self._config["query_pre_attn_scalar"])
            ** -0.5,
            "block_table": step.block_table,
            "value_checks": value_checks,
        }
        if attention_kind == SLIDING_ATTENTION:
            return self._modules["fattn_paged_swa"].fattn_varlen_paged_swa(
                q,
                k_cache,
                v_cache,
                step.cu_seqlens_q,
                step.cu_seqlens_k,
                step.max_seqlen_q,
                step.max_seqlen_k,
                window_size=int(self._config["sliding_window"]),
                launch_config=self._config_for("sliding_attention"),
                **common,
            )
        if attention_kind == FULL_ATTENTION:
            return self._modules[
                "fattn_paged"
            ].fattn_varlen_paged_fwd_block_ptr(
                q,
                k_cache,
                v_cache,
                step.cu_seqlens_q,
                step.cu_seqlens_k,
                step.max_seqlen_q,
                step.max_seqlen_k,
                launch_config=self._config_for("full_attention"),
                **common,
            )
        raise ValueError("Gemma 3 attention received an unsupported kind")

    def gelu_tanh_mul(
        self, gate: torch.Tensor, up: torch.Tensor
    ) -> torch.Tensor:
        return self._modules["gelu_tanh_mul"].gelu_tanh_mul(
            gate,
            up,
            launch_config=self._config_for("mlp_activation"),
        )

    def add(
        self, x: torch.Tensor, y: torch.Tensor, *, site: str
    ) -> torch.Tensor:
        return self._modules["add"].add(
            x,
            y,
            launch_config=self._config_for(site),
        )

class QualifiedRuntime(QualifiedPrimitiveRuntime):
    """Backend-qualified capability reserved for the Rust Engine boundary.

    The unqualified runtime remains useful without a deployment bundle for
    source-attestation tests. Rust admission must instead consume this
    narrower wrapper: it can only be constructed from a runtime whose sealed
    report is backend-qualified. The raw report must still deny independent
    engine authority; Rust supplies the separate closed admission capability.
    """


    # The following four methods are the complete architecture-neutral
    # capability consumed by vosti_kernels.kernels. They delegate to the
    # already-attested runtime and perform no runtime selection.


    # Family-specific element-wise and attention operations stay on the
    # family capability. Rust composes these calls with the shared linear,
    # reshape, cache-store, and row-selection primitives.
    def scaled_embed(
        self, input_ids: torch.Tensor, weight: torch.Tensor
    ) -> torch.Tensor:
        return self._checked_runtime().scaled_embed(input_ids, weight)

    def rms_norm(
        self, x: torch.Tensor, weight: torch.Tensor, site: str
    ) -> torch.Tensor:
        return self._checked_runtime().rms_norm(x, weight, site=site)

    def qk_norm(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        q_weight: torch.Tensor,
        k_weight: torch.Tensor,
        attention_kind: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if attention_kind not in (SLIDING_ATTENTION, FULL_ATTENTION):
            raise ValueError("unsupported Gemma 3 attention kind")
        return self._checked_runtime().qk_norm(q, k, q_weight, k_weight)

    def rotary_embed(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        positions: torch.Tensor,
        attention_kind: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self._checked_runtime().rotary_embed(
            q, k, positions, attention_kind=attention_kind
        )

    def paged_attention(
        self,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        block_table: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        max_seqlen_q: int,
        max_seqlen_k: int,
        attention_kind: str,
    ) -> torch.Tensor:
        step = PRIMITIVE_RUNTIME.PackedAttentionMetadata(
            positions=None,
            kv_caches=(),
            block_table=block_table,
            slot_mapping=None,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
        )
        return self._checked_runtime().paged_attention(
            q,
            k_cache,
            v_cache,
            step,
            attention_kind=attention_kind,
            # Rust's launch-readiness proof and the attested metadata
            # materializers discharge these device-value checks. Avoiding
            # their synchronizing `.item()` calls is also required during
            # CUDA stream capture. The direct Python primitive entry point
            # retains its value-checking default.
            value_checks=False,
        )

    def add(
        self, x: torch.Tensor, y: torch.Tensor, site: str
    ) -> torch.Tensor:
        return self._checked_runtime().add(x, y, site=site)

    def gelu_tanh_mul(
        self, gate: torch.Tensor, up: torch.Tensor
    ) -> torch.Tensor:
        return self._checked_runtime().gelu_tanh_mul(gate, up)

def load_runtime(
    config: dict[str, Any],
    *,
    kernel_root: str | os.PathLike[str] | None = None,
    deployment_bundle: dict[str, Any] | str | os.PathLike[str] | None = None,
    framework_root: str | os.PathLike[str] | None = None,
    model_config_sha256: str | None = None,
    device: str | None = None,
    dtype: torch.dtype = torch.bfloat16,
    environment: dict[str, Any] | None = None,
) -> Runtime:
    """Load the isolated source-attested runtime for the exact model."""

    profile = _validate_model_config(config)
    from . import deployment

    return load_family_runtime(Runtime, family_module="gemma3", scope=_SCOPE, profile=profile, config=config,
        deployment=deployment, kernel_root=kernel_root, deployment_bundle=deployment_bundle,
        framework_root=framework_root, model_config_sha256=model_config_sha256,
        device=device, dtype=dtype, environment=environment)


def load_qualified_runtime(
    config: dict[str, Any],
    *,
    deployment_bundle: dict[str, Any] | str | os.PathLike[str],
    model_config_sha256: str,
    kernel_root: str | os.PathLike[str] | None = None,
    framework_root: str | os.PathLike[str] | None = None,
    device: str | None = None,
    dtype: torch.dtype = torch.bfloat16,
    environment: dict[str, Any] | None = None,
) -> QualifiedRuntime:
    """Load the backend-qualified capability consumed by the Rust bridge."""

    runtime = load_runtime(
        config,
        kernel_root=kernel_root,
        deployment_bundle=deployment_bundle,
        framework_root=framework_root,
        model_config_sha256=model_config_sha256,
        device=device,
        dtype=dtype,
        environment=environment,
    )
    return QualifiedRuntime(runtime)
# @kernel-bridge-end vosti_kernels::gemma3_runtime_capability


__all__ = [
    "QualifiedRuntime",
    "Runtime",
    "config_for_profile",
    "config_from_bundle",
    "load_qualified_runtime",
    "load_runtime",
]
