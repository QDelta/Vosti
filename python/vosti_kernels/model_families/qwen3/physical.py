"""Qwen3-owned physical weight contracts."""

from __future__ import annotations

from collections.abc import Mapping
import math
from typing import Any

import torch

from ... import physical


_LAYER_WEIGHT_SCHEMA = (
    ("input_norm", 1),
    ("q_proj", 2),
    ("k_proj", 2),
    ("v_proj", 2),
    ("q_norm", 1),
    ("k_norm", 1),
    ("o_proj", 2),
    ("post_attn_norm", 1),
    ("gate_up_proj", 2),
    ("down_proj", 2),
)
QWEN3_LAYER_WEIGHT_ROLES = tuple(role for role, _rank in _LAYER_WEIGHT_SCHEMA)
_LAYER_WEIGHT_RANKS = dict(_LAYER_WEIGHT_SCHEMA)

_FIXED_NUMERICAL_ASSUMPTIONS = {
    "rms_norm_eps": 1e-6,
    "rope_theta": 1_000_000.0,
}


# @kernel-bridge-begin vosti_kernels::qwen3_physical_model_contract
def validate_model_weights_permission_contract(
    embed_weight: torch.Tensor,
    layers: list,
    final_norm: torch.Tensor,
    lm_head: torch.Tensor,
    expected_num_layers: int,
) -> None:
    """Check Qwen's config-independent immutable weight premises."""

    if expected_num_layers < 0:
        raise RuntimeError("Qwen3 model weights require a nonnegative layer count")
    if not isinstance(layers, list) or len(layers) != expected_num_layers:
        raise RuntimeError("Qwen3 model weights returned the wrong number of layers")
    named_tensors: list[tuple[str, torch.Tensor]] = [
        ("embed_weight", embed_weight),
        ("final_norm", final_norm),
        ("lm_head", lm_head),
    ]
    named_tensors.extend(physical.named_layer_tensors(
        layers, QWEN3_LAYER_WEIGHT_ROLES, label="Qwen3 model weights"
    ))
    dtypes, devices = physical.validate_named_tensors(
        named_tensors, label="Qwen3 model weights"
    )
    if len(dtypes) != 1 or len(devices) != 1:
        raise RuntimeError("Qwen3 model weights require one dtype and device")
    if embed_weight.dim() != 2 or lm_head.dim() != 2:
        raise RuntimeError("Qwen3 embedding and LM head must be rank-2")
    if tuple(embed_weight.shape) != tuple(lm_head.shape):
        raise RuntimeError("Qwen3 embedding and LM head shapes must match")
    if final_norm.dim() not in (1, 2):
        raise RuntimeError("Qwen3 final norm must be rank-1 or rank-2")
    for layer_index, layer in enumerate(layers):
        for role, tensor in zip(QWEN3_LAYER_WEIGHT_ROLES, layer):
            expected_rank = _LAYER_WEIGHT_RANKS[role]
            if tensor.dim() != expected_rank:
                raise RuntimeError(
                    f"Qwen3 model weights layer {layer_index} {role} must be "
                    f"rank-{expected_rank}"
                )
        gate_up = layer[QWEN3_LAYER_WEIGHT_ROLES.index("gate_up_proj")]
        if gate_up.shape[0] % 2 != 0:
            raise RuntimeError(
                f"Qwen3 model weights layer {layer_index} gate_up_proj "
                "requires an even output width"
            )


def validate_model_weights_runtime_contract(
    embed_weight: torch.Tensor,
    layers: list,
    final_norm: torch.Tensor,
    lm_head: torch.Tensor,
    config: Mapping[str, Any],
    expected_num_layers: int,
) -> None:
    """Bind the permission-checked facade to one qualified Qwen config."""

    validate_model_weights_permission_contract(
        embed_weight, layers, final_norm, lm_head, expected_num_layers
    )
    strict_integer_fields = (
        "vocab_size",
        "hidden_size",
        "intermediate_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "head_dim",
        "max_position_embeddings",
    )
    has_strict_config = all(field in config for field in strict_integer_fields)
    if has_strict_config:
        for field in strict_integer_fields:
            if type(config[field]) is not int or config[field] <= 0:
                raise RuntimeError(
                    f"Qwen3 model weights require positive {field}"
                )
        if config["num_hidden_layers"] != expected_num_layers:
            raise RuntimeError("Qwen3 model weights disagree on the layer count")
        for field in ("rms_norm_eps", "rope_theta"):
            value = config.get(field)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or value <= 0
            ):
                raise RuntimeError(
                    f"Qwen3 model weights require finite positive {field}"
                )
        if type(config.get("tie_word_embeddings")) is not bool:
            raise RuntimeError(
                "Qwen3 model weights require boolean tie_word_embeddings"
            )
        if config["tie_word_embeddings"] and lm_head is not embed_weight:
            raise RuntimeError(
                "Qwen3 model weights require the configured tied embedding"
            )
    for field, expected in _FIXED_NUMERICAL_ASSUMPTIONS.items():
        if field in config and float(config[field]).hex() != expected.hex():
            raise RuntimeError(
                f"Qwen3 model weights require exact {field}={expected!r}"
            )
    hidden = int(config["hidden_size"])
    intermediate = int(config["intermediate_size"])
    num_heads = int(
        config["num_attention_heads"]
        if "num_attention_heads" in config
        else config["num_heads"]
    )
    num_kv_heads = int(
        config["num_key_value_heads"]
        if "num_key_value_heads" in config
        else config["num_kv_heads"]
    )
    q_width = num_heads * int(config["head_dim"])
    kv_width = num_kv_heads * int(config["head_dim"])
    vocab = int(config["vocab_size"]) if has_strict_config else int(
        embed_weight.shape[0]
    )
    expected_global_shapes = {
        "embed_weight": (vocab, hidden),
        "final_norm": (hidden,),
        "lm_head": (vocab, hidden),
    }
    expected_layer_shapes = {
        "input_norm": (hidden,),
        "q_proj": (q_width, hidden),
        "k_proj": (kv_width, hidden),
        "v_proj": (kv_width, hidden),
        "q_norm": (int(config["head_dim"]),),
        "k_norm": (int(config["head_dim"]),),
        "o_proj": (hidden, q_width),
        "post_attn_norm": (hidden,),
        "gate_up_proj": (2 * intermediate, hidden),
        "down_proj": (hidden, intermediate),
    }
    actual_global = {
        "embed_weight": embed_weight,
        "final_norm": final_norm,
        "lm_head": lm_head,
    }
    physical.validate_tensor_shapes(
        actual_global.items(), expected_global_shapes, label="Qwen3 model weights"
    )
    for layer_index, layer in enumerate(layers):
        physical.validate_tensor_shapes(
            zip(QWEN3_LAYER_WEIGHT_ROLES, layer), expected_layer_shapes,
            label=f"Qwen3 model weights layer {layer_index}",
        )
    expected_dtype = config["dtype"]
    expected_device = torch.device(config["device"])
    tensors = [tensor for _, tensor in actual_global.items()]
    tensors.extend(tensor for layer in layers for tensor in layer)
    if {tensor.dtype for tensor in tensors} != {expected_dtype} or {
        tensor.device for tensor in tensors
    } != {expected_device}:
        raise RuntimeError(
            "Qwen3 model weights dtype/device disagree with configured runtime"
        )


# @kernel-bridge-end vosti_kernels::qwen3_physical_model_contract


__all__ = [
    "QWEN3_LAYER_WEIGHT_ROLES",
    "validate_model_weights_permission_contract",
    "validate_model_weights_runtime_contract",
]
