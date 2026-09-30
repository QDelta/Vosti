"""Fail-closed Gemma-4 dense text configuration and checkpoint loading.

Loading is not admission to the verified serving engine. The family runtime,
checked composition and deployment qualification must independently admit it.
"""

from dataclasses import asdict, dataclass, fields
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import torch
from transformers import AutoConfig

from ...attention_geometry import AttentionGeometry
from ... import checkpoint
from ...physical import four_norm_gated_layer_shapes
from .physical import LAYER_WEIGHT_ROLES

# @kernel-bridge-begin vosti_kernels::gemma4_configuration_contract
SLIDING_ATTENTION = "sliding_attention"
FULL_ATTENTION = "full_attention"
TEXT_PREFIX = "model.language_model."


@dataclass(frozen=True)
class Gemma4TextConfig:
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    num_global_key_value_heads: int
    head_dim: int
    global_head_dim: int
    max_position_embeddings: int
    rms_norm_eps: float
    sliding_window: int
    local_rope_theta: float
    global_rope_theta: float
    global_rope_factor: float
    global_partial_rotary_factor: float
    attention_k_eq_v: bool
    final_logit_softcapping: float | None
    layer_types: tuple[str, ...]

    def as_runtime_dict(self) -> dict[str, Any]:
        return asdict(self)

    def attention_geometry(self, layer: int) -> AttentionGeometry:
        if type(layer) is not int or not 0 <= layer < self.num_hidden_layers:
            raise ValueError("attention layer index is outside model geometry")
        local = self.layer_types[layer] == SLIDING_ATTENTION
        return AttentionGeometry(self.num_attention_heads,
            self.num_key_value_heads if local else self.num_global_key_value_heads,
            self.head_dim if local else self.global_head_dim)

    def shared_kv_projection(self, layer: int) -> bool:
        self.attention_geometry(layer)  # Validate the model-static index.
        return self.attention_k_eq_v and self.layer_types[layer] == FULL_ATTENTION

    def rope_scaling(self, layer: int) -> dict[str, Any] | None:
        self.attention_geometry(layer)
        if self.layer_types[layer] == SLIDING_ATTENTION:
            return None
        return {"rope_type": "proportional", "factor": self.global_rope_factor,
                "partial_rotary_factor": self.global_partial_rotary_factor}


def _value(config: Any, key: str, default: Any = None) -> Any:
    return config.get(key, default) if isinstance(config, dict) else getattr(config, key, default)


def _positive_int(config: Any, key: str) -> int:
    value = _value(config, key)
    if type(value) is not int or value <= 0:
        raise ValueError(f"Gemma 4 requires positive integer {key}")
    return value


def _positive_float(config: Any, key: str) -> float:
    value = _value(config, key)
    if (isinstance(value, bool) or not isinstance(value, (float, int))
            or not math.isfinite(float(value)) or value <= 0):
        raise ValueError(f"Gemma 4 requires finite positive {key}")
    return float(value)


def parse_text_config(config: Any) -> Gemma4TextConfig:
    text = _value(config, "text_config") or config
    if _value(text, "model_type") not in {"gemma4_text", "gemma4_unified_text"}:
        raise ValueError("checkpoint is not a Gemma 4 text model")
    # Both checkpoint formats must satisfy the same dense text feature checks.
    for field in ("num_kv_shared_layers", "hidden_size_per_layer_input"):
        if _value(text, field, 0) != 0:
            raise ValueError(f"Gemma 4 dense loader does not support {field}")
    for field in ("enable_moe_block", "use_double_wide_mlp", "attention_bias"):
        if _value(text, field, False) is not False:
            raise ValueError(f"Gemma 4 dense loader does not support {field}")
    if _value(text, "hidden_activation") != "gelu_pytorch_tanh":
        raise ValueError("Gemma 4 requires gelu_pytorch_tanh")
    if _value(text, "tie_word_embeddings") is not True:
        raise ValueError("Gemma 4 requires tied embeddings")
    if _value(text, "attention_dropout", 0.0) != 0.0:
        raise ValueError("Gemma 4 requires zero attention dropout")
    if _value(text, "use_bidirectional_attention", False) not in (False, "vision"):
        raise ValueError("Gemma 4 text serving requires causal attention")
    if _value(text, "attn_logit_softcapping") is not None:
        raise ValueError("Gemma 4 does not support attention-score softcapping")
    shared = _value(text, "attention_k_eq_v")
    if type(shared) is not bool:
        raise ValueError("Gemma 4 requires boolean attention_k_eq_v")
    values = {field: _positive_int(text, field) for field in (
        "vocab_size", "hidden_size", "intermediate_size", "num_hidden_layers",
        "num_attention_heads", "num_key_value_heads", "num_global_key_value_heads",
        "head_dim", "global_head_dim", "max_position_embeddings", "sliding_window")}
    kinds = tuple(_value(text, "layer_types", ()))
    if len(kinds) != values["num_hidden_layers"] or any(
        kind not in (SLIDING_ATTENTION, FULL_ATTENTION) for kind in kinds
    ):
        raise ValueError("Gemma 4 attention schedule must cover every layer with supported kinds")
    for heads, dim in (("num_key_value_heads", "head_dim"),
                       ("num_global_key_value_heads", "global_head_dim")):
        AttentionGeometry(values["num_attention_heads"], values[heads], values[dim])
    ropes = _value(text, "rope_parameters")
    if not isinstance(ropes, dict) or set(ropes) != {SLIDING_ATTENTION, FULL_ATTENTION}:
        raise ValueError("Gemma 4 requires local/global RoPE policies")
    local, global_ = ropes[SLIDING_ATTENTION], ropes[FULL_ATTENTION]
    if not isinstance(local, dict) or local.get("rope_type") != "default" or set(local) != {"rope_type", "rope_theta"}:
        raise ValueError("Gemma 4 sliding layers require default RoPE")
    if not isinstance(global_, dict) or global_.get("rope_type") != "proportional" or not (
        {"rope_type", "rope_theta", "partial_rotary_factor"} <= set(global_)
        <= {"rope_type", "rope_theta", "partial_rotary_factor", "factor"}
    ):
        raise ValueError("Gemma 4 global layers require proportional RoPE")
    proportion = global_["partial_rotary_factor"]
    if (isinstance(proportion, bool) or not isinstance(proportion, (float, int))
            or not math.isfinite(float(proportion)) or not 0 <= proportion <= 1):
        raise ValueError("Gemma 4 requires a rotary fraction in [0, 1]")
    cap = _value(text, "final_logit_softcapping")
    return Gemma4TextConfig(**values, rms_norm_eps=_positive_float(text, "rms_norm_eps"),
        local_rope_theta=_positive_float(local, "rope_theta"),
        global_rope_theta=_positive_float(global_, "rope_theta"),
        global_rope_factor=_positive_float({"factor": global_.get("factor", 1.0)}, "factor"),
        global_partial_rotary_factor=float(proportion), attention_k_eq_v=shared,
        final_logit_softcapping=None if cap is None else _positive_float(text, "final_logit_softcapping"),
        layer_types=kinds)


def config_from_runtime_dict(config: dict[str, Any]) -> Gemma4TextConfig:
    """Validate the exact flattened family schema before interpreting weights."""
    if not isinstance(config, dict) or set(config) != {field.name for field in fields(Gemma4TextConfig)}:
        raise ValueError("Gemma 4 runtime config differs from its closed schema")
    raw = {key: value for key, value in config.items() if key not in {
        "local_rope_theta", "global_rope_theta", "global_rope_factor", "global_partial_rotary_factor"}}
    raw.update(model_type="gemma4_text", hidden_activation="gelu_pytorch_tanh",
        tie_word_embeddings=True, attention_bias=False, attention_dropout=0.0,
        hidden_size_per_layer_input=0, num_kv_shared_layers=0, enable_moe_block=False,
        use_double_wide_mlp=False, use_bidirectional_attention=False,
        rope_parameters={SLIDING_ATTENTION: {"rope_type": "default", "rope_theta": config["local_rope_theta"]},
            FULL_ATTENTION: {"rope_type": "proportional", "rope_theta": config["global_rope_theta"],
                "factor": config["global_rope_factor"], "partial_rotary_factor": config["global_partial_rotary_factor"]}})
    return parse_text_config(raw)


# @kernel-bridge-end vosti_kernels::gemma4_configuration_contract


# @kernel-bridge-begin vosti_kernels::gemma4_checkpoint_loader
def layer_checkpoint_keys(layer: int, config: Gemma4TextConfig) -> dict[str, str | tuple[str, str]]:
    prefix = f"{TEXT_PREFIX}layers.{layer}"
    keys = {role: f"{prefix}.self_attn.{role}.weight"
            for role in ("q_proj", "k_proj", "v_proj", "q_norm", "k_norm", "o_proj")}
    if config.shared_kv_projection(layer):
        keys["v_proj"] = keys["k_proj"]
    keys.update({role: f"{prefix}.{name}.weight" for role, name in (
        ("input_norm", "input_layernorm"), ("post_attn_norm", "post_attention_layernorm"),
        ("pre_feedforward_norm", "pre_feedforward_layernorm"),
        ("post_feedforward_norm", "post_feedforward_layernorm"))})
    keys["gate_up_proj"] = (f"{prefix}.mlp.gate_proj.weight", f"{prefix}.mlp.up_proj.weight")
    keys["down_proj"] = f"{prefix}.mlp.down_proj.weight"
    keys["layer_scalar"] = f"{prefix}.layer_scalar"
    return keys


def expected_checkpoint_shapes(config: Gemma4TextConfig) -> dict[str, tuple[int, ...]]:
    h, ff = config.hidden_size, config.intermediate_size
    expected = {f"{TEXT_PREFIX}embed_tokens.weight": (config.vocab_size, h),
                f"{TEXT_PREFIX}norm.weight": (h,)}
    for i in range(config.num_hidden_layers):
        geometry = config.attention_geometry(i)
        shapes = four_norm_gated_layer_shapes(h, ff, geometry, layer_scale=True)
        for role, key in layer_checkpoint_keys(i, config).items():
            if isinstance(key, tuple):
                for part in key:
                    expected[part] = (ff, h)
            else:
                expected[key] = shapes[role]
    return expected


def validate_checkpoint_layout(model_path: str | Path, config: Gemma4TextConfig) -> None:
    checkpoint.validate_layout(Path(model_path), expected_checkpoint_shapes(config),
                               text_prefix=TEXT_PREFIX, label="Gemma 4")


def inspect_text_checkpoint(model_path: str | Path) -> Gemma4TextConfig:
    root = Path(model_path).expanduser().resolve()
    raw = json.loads((root / "config.json").read_text())
    # Unified checkpoints supply the complete text configuration, but their
    # multimodal AutoConfig class is absent in some supported Transformers
    # versions. Parse only that declared text schema; do not register a global
    # Transformers alias or modify the checkpoint on disk.
    if raw.get("model_type") in {"gemma4_unified", "gemma4_unified_text"}:
        config = parse_text_config(raw)
    else:
        config = parse_text_config(AutoConfig.from_pretrained(root, local_files_only=True))
    validate_checkpoint_layout(root, config)
    return config


def load_text_weights(model_path: str | Path, *, device: str | torch.device = "cpu") -> dict[str, Any]:
    root = Path(model_path).expanduser().resolve()
    config = inspect_text_checkpoint(root)
    raw = checkpoint.load_tensors(root, set(expected_checkpoint_shapes(config)), device=device, label="Gemma 4")
    embed = raw[f"{TEXT_PREFIX}embed_tokens.weight"]
    layers = []
    for i in range(config.num_hidden_layers):
        keys = layer_checkpoint_keys(i, config)
        tensors = []
        for role in LAYER_WEIGHT_ROLES:
            key = keys[role]
            if isinstance(key, tuple):
                # Release separate gate/up allocations after packing each layer.
                tensor = torch.cat((raw.pop(key[0]), raw.pop(key[1])), dim=0).contiguous()
            else:
                tensor = raw[key]
            tensors.append(tensor)
        layers.append(tuple(tensors))
    loaded = dict(architecture="gemma4_text", config=config.as_runtime_dict(),
        model_config_sha256=hashlib.sha256((root / "config.json").read_bytes()).hexdigest(),
        embed_weight=embed, layers=layers, attention_kinds=list(config.layer_types),
        final_norm=raw[f"{TEXT_PREFIX}norm.weight"], lm_head=embed)
    from .physical import validate_model_weights_runtime_contract

    validate_model_weights_runtime_contract(loaded["embed_weight"], loaded["layers"],
        loaded["attention_kinds"], loaded["final_norm"], loaded["lm_head"],
        loaded["config"], config.num_hidden_layers)
    return loaded
# @kernel-bridge-end vosti_kernels::gemma4_checkpoint_loader
