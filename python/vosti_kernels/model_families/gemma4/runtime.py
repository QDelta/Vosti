"""Gemma-4 dense text primitives on the common sealed runtime capability.

Rust composition and admission are separate; this module has no full forward
and never independently declares the family engine-reachable.
"""

from types import MappingProxyType

import torch

from ... import rotary, primitive_runtime
from ...static_runtime import StaticPrimitiveRuntime, QualifiedPrimitiveRuntime, load_family_runtime
from . import profile, deployment
from .loader import config_from_runtime_dict, FULL_ATTENTION, SLIDING_ATTENTION


# @kernel-bridge-begin vosti_kernels::gemma4_runtime_capability
class Runtime(StaticPrimitiveRuntime):
    def __init__(self, modules, origins, digests, kernel_root, qualification, selected_profile, device, dtype):
        config = profile.model_config(selected_profile)
        super().__init__(scope=profile.scope(), config=config, modules=modules, origins=origins,
            digests=digests, kernel_root=kernel_root, qualification=qualification,
            profile=selected_profile, device=device, dtype=dtype)
        resolved = config_from_runtime_dict(config)
        self._geometry = MappingProxyType({kind: resolved.attention_geometry(i)
                                          for i, kind in enumerate(resolved.layer_types)})
        self._rope_table_cache = None
        self._value_norm_weights = None
        if qualification is not None:
            self._rope_table_cache = MappingProxyType({kind: rotary.PrecomputedTables(
                *self._compute_rope_tables(torch.arange(resolved.max_position_embeddings,
                    dtype=torch.int64, device=device), attention_kind=kind, dtype=dtype))
                for kind in self._geometry})
            self._value_norm_weights = MappingProxyType({kind: torch.ones(geometry.head_dim,
                dtype=dtype, device=device) for kind, geometry in self._geometry.items()})

    def _kind(self, attention_kind):
        try:
            geometry = self._geometry[attention_kind]
        except KeyError as error:
            raise ValueError("Gemma 4 received an unsupported attention kind") from error
        return geometry, "local" if attention_kind == SLIDING_ATTENTION else "global"

    def scaled_embed(self, input_ids, weight):
        return self._modules["scaled_embedding"].scaled_embedding(input_ids, weight,
            launch_config=self._config_for("token_embedding"))

    def rms_norm(self, x, weight, *, site):
        return self._modules["rmsnorm"].rmsnorm(x, weight, self._config["rms_norm_eps"],
            launch_config=self._config_for(site))

    def qk_norm(self, q, k, q_weight, k_weight, *, attention_kind):
        geometry, prefix = self._kind(attention_kind)
        norm = self._modules["qk_norm"].head_rms_norm
        return (norm(q, q_weight, geometry.query_heads, self._config["rms_norm_eps"],
                     launch_config=self._config_for(f"{prefix}.q_norm")),
                norm(k, k_weight, geometry.kv_heads, self._config["rms_norm_eps"],
                     launch_config=self._config_for(f"{prefix}.k_norm")))

    def value_norm(self, v, *, attention_kind):
        geometry, prefix = self._kind(attention_kind)
        if v.ndim != 2 or v.shape[1] != geometry.kv_width:
            raise ValueError("Gemma 4 value rows disagree with static layer geometry")
        # Qualified serving always uses preallocated immutable all-ones weights.
        # The unqualified direct-kernel test surface still uses the same kernel.
        weight = (self._value_norm_weights[attention_kind] if self._value_norm_weights is not None
                  else torch.ones(geometry.head_dim, dtype=v.dtype, device=v.device))
        out = self._modules["qk_norm"].head_rms_norm(v, weight, geometry.kv_heads,
            self._config["rms_norm_eps"], launch_config=self._config_for(f"{prefix}.v_norm"))
        return out.reshape(v.shape[0], geometry.kv_heads, geometry.head_dim)

    def _compute_rope_tables(self, positions, *, attention_kind, dtype):
        geometry, prefix = self._kind(attention_kind)
        scaling = None if attention_kind == SLIDING_ATTENTION else {
            "rope_type": "proportional", "factor": self._config["global_rope_factor"],
            "partial_rotary_factor": self._config["global_partial_rotary_factor"]}
        return rotary.tables(positions, head_dim=geometry.head_dim,
            theta=self._config[f"{prefix}_rope_theta"], scaling=scaling, dtype=dtype)

    def _rope_tables(self, positions, *, attention_kind, dtype):
        self._kind(attention_kind)
        if self._rope_table_cache is None:
            return self._compute_rope_tables(positions, attention_kind=attention_kind, dtype=dtype)
        if dtype != self._dtype:
            raise ValueError("Gemma 4 RoPE input dtype differs from sealed runtime")
        return self._rope_table_cache[attention_kind].select(positions)

    def rotary_embed(self, q, k, positions, *, attention_kind):
        geometry, prefix = self._kind(attention_kind)
        rows = q.shape[0]
        if q.shape != (rows, geometry.query_width) or k.shape != (rows, geometry.kv_width):
            raise ValueError("Gemma 4 rotary inputs disagree with static layer geometry")
        cos, sin = self._rope_tables(positions, attention_kind=attention_kind, dtype=q.dtype)
        rope = self._modules["rope"].rope
        results = []
        for value, heads, site in ((q, geometry.query_heads, "rotary_embed_q"),
                                   (k, geometry.kv_heads, "rotary_embed_k")):
            results.append(rope(value.reshape(rows * heads, geometry.head_dim),
                cos.repeat_interleave(heads, dim=0), sin.repeat_interleave(heads, dim=0),
                launch_config=self._config_for(f"{prefix}.{site}")).reshape(rows, heads, geometry.head_dim))
        return tuple(results)

    def paged_attention(self, q, k_cache, v_cache, step, *, attention_kind, value_checks=True):
        self._kind(attention_kind)
        args = (q, k_cache, v_cache, step.cu_seqlens_q, step.cu_seqlens_k,
                step.max_seqlen_q, step.max_seqlen_k)
        common = dict(softmax_scale=1.0, block_table=step.block_table, value_checks=value_checks)
        if attention_kind == FULL_ATTENTION:
            return self._modules["fattn_paged"].fattn_varlen_paged_fwd_block_ptr(*args,
                launch_config=self._config_for("full_attention"), **common)
        return self._modules["fattn_paged_swa"].fattn_varlen_paged_swa(*args,
            window_size=self._config["sliding_window"],
            launch_config=self._config_for("sliding_attention"), **common)

    def add(self, x, y, *, site):
        return self._modules["add"].add(x, y, launch_config=self._config_for(site))

    def gelu_tanh_mul(self, gate, up):
        return self._modules["gelu_tanh_mul"].gelu_tanh_mul(gate, up,
            launch_config=self._config_for("mlp_activation"))

    def scale(self, x, scalar):
        return self._modules["scale"].scale(x, scalar, launch_config=self._config_for("layer_output_scale"))

    def softcap(self, logits):
        cap = self._config["final_logit_softcapping"]
        if cap is None:
            return logits.clone().contiguous()
        return self._modules["softcap"].softcap(logits, cap,
            launch_config=self._config_for("final_logits_softcap"))


class QualifiedRuntime(QualifiedPrimitiveRuntime):
    def scaled_embed(self, input_ids, weight):
        return self._checked_runtime().scaled_embed(input_ids, weight)

    def rms_norm(self, x, weight, site):
        return self._checked_runtime().rms_norm(x, weight, site=site)

    def qk_norm(self, q, k, q_weight, k_weight, attention_kind):
        return self._checked_runtime().qk_norm(q, k, q_weight, k_weight, attention_kind=attention_kind)

    def value_norm(self, v, attention_kind):
        return self._checked_runtime().value_norm(v, attention_kind=attention_kind)

    def rotary_embed(self, q, k, positions, attention_kind):
        return self._checked_runtime().rotary_embed(q, k, positions, attention_kind=attention_kind)

    def paged_attention(self, q, k_cache, v_cache, block_table, cu_seqlens_q,
                        cu_seqlens_k, max_seqlen_q, max_seqlen_k, attention_kind):
        step = primitive_runtime.PackedAttentionMetadata(positions=None, kv_caches=(),
            block_table=block_table, slot_mapping=None, cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k, max_seqlen_q=max_seqlen_q, max_seqlen_k=max_seqlen_k)
        # Only the checked Rust caller may discharge device-value readiness here.
        return self._checked_runtime().paged_attention(q, k_cache, v_cache, step,
            attention_kind=attention_kind, value_checks=False)

    def add(self, x, y, site):
        return self._checked_runtime().add(x, y, site=site)

    def gelu_tanh_mul(self, gate, up):
        return self._checked_runtime().gelu_tanh_mul(gate, up)

    def scale(self, x, scalar):
        return self._checked_runtime().scale(x, scalar)

    def softcap(self, logits):
        return self._checked_runtime().softcap(logits)


def config_for_profile(profile_name=None):
    return profile.model_config(profile.model_profile_for_name(profile_name))


def config_from_bundle(bundle_path):
    bundle = deployment.load_bundle(bundle_path)
    model = bundle["deployment"]["model"]
    selected = profile.model_profile_for_config(model["resolved_config"])
    if selected["model"]["name"] != model["catalog_name"]:
        raise ValueError("Gemma 4 bundle model name differs from its exact config")
    return profile.model_config(selected)


def load_runtime(config, *, kernel_root=None, deployment_bundle=None, framework_root=None,
                 model_config_sha256=None, device=None, dtype=torch.bfloat16, environment=None):
    selected = profile.model_profile_for_config(config)
    return load_family_runtime(Runtime, family_module="gemma4", scope=profile.scope(), profile=selected, config=config,
        deployment=deployment, kernel_root=kernel_root, deployment_bundle=deployment_bundle,
        framework_root=framework_root, model_config_sha256=model_config_sha256,
        device=device, dtype=dtype, environment=environment)


def load_qualified_runtime(config, *, deployment_bundle, model_config_sha256, **kwargs):
    return QualifiedRuntime(load_runtime(config, deployment_bundle=deployment_bundle,
        model_config_sha256=model_config_sha256, **kwargs))
# @kernel-bridge-end vosti_kernels::gemma4_runtime_capability
