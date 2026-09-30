"""Fail-closed text-only Llama 3 checkpoint inspection and loading."""

# @kernel-bridge-begin vosti_kernels::llama3_checkpoint_loader
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import math
from pathlib import Path
from typing import Any

import torch
from transformers import AutoConfig

from ...attention_geometry import AttentionGeometry
from ... import checkpoint
from .physical import LLAMA3_LAYER_WEIGHT_ROLES


@dataclass(frozen=True)
class Llama3TextConfig:
    model_type: str
    transformers_architecture: str
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    max_position_embeddings: int
    rms_norm_eps: float
    rope_theta: float
    rope_factor: float
    rope_low_frequency_factor: float
    rope_high_frequency_factor: float
    rope_original_max_position_embeddings: int
    tie_word_embeddings: bool
    hidden_act: str
    attention_bias: bool
    attention_dropout: float
    mlp_bias: bool
    pretraining_tp: int
    attention_kind: str
    rope_scaling_kind: str

    def as_runtime_dict(self) -> dict[str, Any]:
        return asdict(self)

    def attention_geometry(self, layer: int) -> AttentionGeometry:
        if type(layer) is not int or not 0 <= layer < self.num_hidden_layers:
            raise ValueError("attention layer index is outside model geometry")
        return AttentionGeometry(self.num_attention_heads, self.num_key_value_heads, self.head_dim)


def _value(config: Any, name: str, default: Any = None) -> Any:
    if isinstance(config, dict):
        return config.get(name, default)
    return getattr(config, name, default)


def _positive_int(config: Any, name: str) -> int:
    value = _value(config, name)
    if type(value) is not int or value <= 0:
        raise ValueError(f"Llama 3 config requires positive {name}")
    return value


def _positive_float(config: Any, name: str) -> float:
    value = _value(config, name)
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or value <= 0
    ):
        raise ValueError(f"Llama 3 config requires finite positive {name}")
    return float(value)


def parse_text_config(config: Any) -> Llama3TextConfig:
    """Resolve exactly the supported dense Llama 3 composition."""

    if _value(config, "model_type") != "llama":
        raise ValueError("checkpoint is not a Llama model")
    architectures = _value(config, "architectures", ["LlamaForCausalLM"])
    if architectures != ["LlamaForCausalLM"]:
        raise ValueError("Llama 3 config requires LlamaForCausalLM")

    values = {
        name: _positive_int(config, name)
        for name in (
            "vocab_size",
            "hidden_size",
            "intermediate_size",
            "num_hidden_layers",
            "num_attention_heads",
            "num_key_value_heads",
            "head_dim",
            "max_position_embeddings",
        )
    }
    if values["num_attention_heads"] % values["num_key_value_heads"]:
        raise ValueError("Llama 3 query heads must be divisible by KV heads")
    if values["head_dim"] % 2:
        raise ValueError("Llama 3 head dimension must be even")
    if _value(config, "hidden_act") != "silu":
        raise ValueError("Llama 3 loader supports only SiLU")
    if _value(config, "attention_bias") is not False:
        raise ValueError("Llama 3 loader requires bias-free attention")
    if _value(config, "mlp_bias", False) is not False:
        raise ValueError("Llama 3 loader requires bias-free MLP projections")
    if float(_value(config, "attention_dropout", 0.0)) != 0.0:
        raise ValueError("Llama 3 loader requires zero attention dropout")
    tie_word_embeddings = _value(config, "tie_word_embeddings")
    if type(tie_word_embeddings) is not bool:
        raise ValueError("Llama 3 loader requires boolean tie_word_embeddings")
    if _value(config, "pretraining_tp", 1) != 1:
        raise ValueError("Llama 3 loader requires pretraining_tp=1")

    scaling = _value(config, "rope_scaling")
    if not isinstance(scaling, dict) or scaling.get("rope_type") != "llama3":
        raise ValueError("Llama 3 loader requires llama3 RoPE scaling")
    required_scaling = {
        "factor",
        "low_freq_factor",
        "high_freq_factor",
        "original_max_position_embeddings",
        "rope_type",
    }
    # Current Transformers normalizes the top-level theta into this mapping.
    # Accept only that one redundant key and require exact agreement below.
    if set(scaling) not in (required_scaling, required_scaling | {"rope_theta"}):
        raise ValueError("Llama 3 RoPE scaling differs from its closed schema")
    original = scaling["original_max_position_embeddings"]
    if type(original) is not int or original <= 0:
        raise ValueError("Llama 3 RoPE original context must be positive")

    def scaling_float(name: str) -> float:
        value = scaling[name]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or value <= 0
        ):
            raise ValueError(f"Llama 3 RoPE requires finite positive {name}")
        return float(value)

    top_level_theta = _value(config, "rope_theta")
    if top_level_theta is None:
        rope_theta = scaling_float("rope_theta")
    else:
        rope_theta = _positive_float(config, "rope_theta")
    if "rope_theta" in scaling and float(scaling["rope_theta"]).hex() != rope_theta.hex():
        raise ValueError("Llama 3 RoPE theta representations disagree")

    return Llama3TextConfig(
        model_type="llama",
        transformers_architecture="LlamaForCausalLM",
        **values,
        rms_norm_eps=_positive_float(config, "rms_norm_eps"),
        rope_theta=rope_theta,
        rope_factor=scaling_float("factor"),
        rope_low_frequency_factor=scaling_float("low_freq_factor"),
        rope_high_frequency_factor=scaling_float("high_freq_factor"),
        rope_original_max_position_embeddings=original,
        tie_word_embeddings=tie_word_embeddings,
        hidden_act="silu",
        attention_bias=False,
        attention_dropout=0.0,
        mlp_bias=False,
        pretraining_tp=1,
        attention_kind="full_attention",
        rope_scaling_kind="llama3",
    )


def _layer_checkpoint_keys(layer: int) -> dict[str, str | tuple[str, str]]:
    prefix = f"model.layers.{layer}"
    return {
        "input_norm": f"{prefix}.input_layernorm.weight",
        "q_proj": f"{prefix}.self_attn.q_proj.weight",
        "k_proj": f"{prefix}.self_attn.k_proj.weight",
        "v_proj": f"{prefix}.self_attn.v_proj.weight",
        "o_proj": f"{prefix}.self_attn.o_proj.weight",
        "post_attn_norm": f"{prefix}.post_attention_layernorm.weight",
        "gate_up_proj": (
            f"{prefix}.mlp.gate_proj.weight",
            f"{prefix}.mlp.up_proj.weight",
        ),
        "down_proj": f"{prefix}.mlp.down_proj.weight",
    }


def expected_checkpoint_shapes(config: Llama3TextConfig) -> dict[str, tuple[int, ...]]:
    hidden = config.hidden_size
    intermediate = config.intermediate_size
    q_width = config.num_attention_heads * config.head_dim
    kv_width = config.num_key_value_heads * config.head_dim
    expected = {
        "model.embed_tokens.weight": (config.vocab_size, hidden),
        "model.norm.weight": (hidden,),
    }
    if not config.tie_word_embeddings:
        expected["lm_head.weight"] = (config.vocab_size, hidden)
    role_shapes = {
        "input_norm": (hidden,),
        "q_proj": (q_width, hidden),
        "k_proj": (kv_width, hidden),
        "v_proj": (kv_width, hidden),
        "o_proj": (hidden, q_width),
        "post_attn_norm": (hidden,),
        "down_proj": (hidden, intermediate),
    }
    for layer in range(config.num_hidden_layers):
        for role, key in _layer_checkpoint_keys(layer).items():
            if role == "gate_up_proj":
                gate_key, up_key = key
                expected[gate_key] = (intermediate, hidden)
                expected[up_key] = (intermediate, hidden)
            else:
                assert isinstance(key, str)
                expected[key] = role_shapes[role]
    return expected


def validate_checkpoint_layout(model_path: str | Path, config: Llama3TextConfig) -> None:
    """Validate exact text keys, shapes and BF16 dtype without materializing data."""
    checkpoint.validate_layout(Path(model_path), expected_checkpoint_shapes(config),
                               text_prefix='', label='Llama 3')


def inspect_text_checkpoint(model_path: str | Path) -> Llama3TextConfig:
    resolved = Path(model_path).expanduser().resolve()
    config = parse_text_config(
        AutoConfig.from_pretrained(resolved, local_files_only=True)
    )
    validate_checkpoint_layout(resolved, config)
    # Inspection is the production admission boundary, not a family-wide
    # parser. Reject otherwise valid Llama variants unless their complete
    # normalized configuration is present in the sealed profile registry.
    from .profile import model_profile_for_config

    model_profile_for_config(config.as_runtime_dict())
    return config


def _load_raw_tensors(
    model_path: Path, keys: set[str], *, device: str | torch.device,
) -> dict[str, torch.Tensor]:
    return checkpoint.load_tensors(model_path, keys, device=device, label='Llama 3')


def load_text_weights(
    model_path: str | Path,
    *,
    device: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Load the exact Llama roles; perform no runtime qualification."""

    resolved = Path(model_path).expanduser().resolve()
    config = inspect_text_checkpoint(resolved)
    expected = expected_checkpoint_shapes(config)
    raw = _load_raw_tensors(resolved, set(expected), device=device)
    layers = []
    for layer_index in range(config.num_hidden_layers):
        keys = _layer_checkpoint_keys(layer_index)
        values = []
        for role in LLAMA3_LAYER_WEIGHT_ROLES:
            key = keys[role]
            if role == "gate_up_proj":
                gate_key, up_key = key
                value = torch.cat((raw[gate_key], raw[up_key]), dim=0).contiguous()
            else:
                assert isinstance(key, str)
                value = raw[key]
            values.append(value)
        layers.append(tuple(values))
    embed_weight = raw["model.embed_tokens.weight"]
    loaded = {
        "architecture": "llama3",
        "config": config.as_runtime_dict(),
        "model_config_sha256": hashlib.sha256(
            (resolved / "config.json").read_bytes()
        ).hexdigest(),
        "embed_weight": embed_weight,
        "layers": layers,
        "attention_kinds": ["full_attention"] * config.num_hidden_layers,
        "final_norm": raw["model.norm.weight"],
        "lm_head": (
            embed_weight
            if config.tie_word_embeddings
            else raw["lm_head.weight"]
        ),
    }
    from .physical import validate_model_weights_runtime_contract

    validate_model_weights_runtime_contract(
        loaded["embed_weight"],
        loaded["layers"],
        loaded["final_norm"],
        loaded["lm_head"],
        loaded["config"],
        config.num_hidden_layers,
    )
    return loaded
# @kernel-bridge-end vosti_kernels::llama3_checkpoint_loader


__all__ = [
    "Llama3TextConfig",
    "expected_checkpoint_shapes",
    "inspect_text_checkpoint",
    "load_text_weights",
    "parse_text_config",
    "validate_checkpoint_layout",
]
