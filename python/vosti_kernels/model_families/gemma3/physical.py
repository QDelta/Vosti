"""Text-only Gemma3 physical weight and KV-cache contracts."""

from __future__ import annotations

import math
from typing import Any

import torch

from ... import physical


GEMMA3_LAYER_WEIGHT_ROLES = (
    "input_norm",
    "q_proj",
    "k_proj",
    "v_proj",
    "q_norm",
    "k_norm",
    "o_proj",
    "post_attn_norm",
    "pre_feedforward_norm",
    "gate_up_proj",
    "down_proj",
    "post_feedforward_norm",
)

_FIXED_NUMERICAL_ASSUMPTIONS = {
    "rms_norm_eps": 1e-6,
    "local_rope_theta": 10_000.0,
    "global_rope_theta": 1_000_000.0,
    "global_rope_factor": 8.0,
}


# @kernel-bridge-begin vosti_kernels::gemma3_physical_model_contract
def validate_model_weights_permission_contract(
    embed_weight: torch.Tensor,
    layers: list,
    attention_kinds: list,
    final_norm: torch.Tensor,
    lm_head: torch.Tensor,
    hidden_size: int,
    sliding_window: int,
    expected_num_layers: int,
) -> None:
    """Check Gemma's config-independent immutable weight premises."""

    roles = GEMMA3_LAYER_WEIGHT_ROLES
    if expected_num_layers < 0:
        raise RuntimeError("Gemma 3 model weights require a nonnegative layer count")
    if type(hidden_size) is not int or hidden_size <= 0:
        raise RuntimeError("Gemma 3 model weights require a positive hidden size")
    if type(sliding_window) is not int or sliding_window <= 0:
        raise RuntimeError("Gemma 3 model weights require a positive sliding window")
    if not isinstance(layers, list) or len(layers) != expected_num_layers:
        raise RuntimeError("Gemma 3 model weights returned the wrong number of layers")
    if not isinstance(attention_kinds, list) or len(
        attention_kinds
    ) != expected_num_layers or any(
        kind not in ("sliding_attention", "full_attention")
        for kind in attention_kinds
    ):
        raise RuntimeError("Gemma 3 model weights contain an unsupported attention kind")
    if lm_head is not embed_weight:
        raise RuntimeError("Gemma 3 model weights require a tied embedding and LM head")

    named_tensors: list[tuple[str, torch.Tensor]] = [
        ("embed_weight", embed_weight),
        ("final_norm", final_norm),
        ("lm_head", lm_head),
    ]
    named_tensors.extend(physical.named_layer_tensors(
        layers, roles, label="Gemma 3 model weights"
    ))
    dtypes, devices = physical.validate_named_tensors(
        named_tensors, label="Gemma 3 model weights"
    )
    if dtypes != {torch.bfloat16} or len(devices) != 1:
        raise RuntimeError("Gemma 3 model weights require one BF16 dtype/device")
    if embed_weight.dim() != 2 or min(embed_weight.shape) <= 0:
        raise RuntimeError("Gemma 3 model weights require a nonempty rank-2 embedding")
    vocab, hidden = map(int, embed_weight.shape)
    if hidden != hidden_size:
        raise RuntimeError("Gemma 3 model weights disagree on the hidden size")
    if tuple(lm_head.shape) != (vocab, hidden):
        raise RuntimeError("Gemma 3 model weights tied LM head has the wrong shape")
    if tuple(final_norm.shape) != (hidden,):
        raise RuntimeError(
            f"Gemma 3 model weights final_norm has shape {tuple(final_norm.shape)}, "
            f"expected {(hidden,)}"
        )

    role_index = {role: index for index, role in enumerate(roles)}
    hidden_norm_roles = (
        "input_norm",
        "post_attn_norm",
        "pre_feedforward_norm",
        "post_feedforward_norm",
    )
    for layer_index, layer in enumerate(layers):
        for role in hidden_norm_roles:
            tensor = layer[role_index[role]]
            if tuple(tensor.shape) != (hidden,):
                raise RuntimeError(
                    f"Gemma 3 model weights layer {layer_index} {role} has shape "
                    f"{tuple(tensor.shape)}, expected {(hidden,)}"
                )
        q_proj = layer[role_index["q_proj"]]
        k_proj = layer[role_index["k_proj"]]
        v_proj = layer[role_index["v_proj"]]
        q_norm = layer[role_index["q_norm"]]
        k_norm = layer[role_index["k_norm"]]
        o_proj = layer[role_index["o_proj"]]
        gate_up = layer[role_index["gate_up_proj"]]
        down_proj = layer[role_index["down_proj"]]
        matrices = (q_proj, k_proj, v_proj, o_proj, gate_up, down_proj)
        if any(tensor.dim() != 2 for tensor in matrices):
            raise RuntimeError(
                f"Gemma 3 model weights layer {layer_index} projections must be rank-2"
            )
        if q_norm.dim() != 1 or k_norm.dim() != 1 or q_norm.numel() <= 0:
            raise RuntimeError(
                f"Gemma 3 model weights layer {layer_index} Q/K norms must be nonempty rank-1"
            )
        head_dim = int(q_norm.numel())
        q_width = int(q_proj.shape[0])
        kv_width = int(k_proj.shape[0])
        gate_width = int(gate_up.shape[0])
        if (
            tuple(q_proj.shape)[1:] != (hidden,)
            or tuple(k_proj.shape)[1:] != (hidden,)
            or tuple(v_proj.shape) != (kv_width, hidden)
            or tuple(k_norm.shape) != (head_dim,)
            or q_width <= 0
            or kv_width <= 0
            or q_width % head_dim != 0
            or kv_width % head_dim != 0
            or tuple(o_proj.shape) != (hidden, q_width)
            or gate_width <= 0
            or gate_width % 2 != 0
            or tuple(gate_up.shape)[1:] != (hidden,)
            or tuple(down_proj.shape) != (hidden, gate_width // 2)
        ):
            raise RuntimeError(
                f"Gemma 3 model weights layer {layer_index} projection shapes are inconsistent"
            )


def validate_model_weights_runtime_contract(
    embed_weight: torch.Tensor,
    layers: list,
    attention_kinds: list,
    final_norm: torch.Tensor,
    lm_head: torch.Tensor,
    config: dict[str, Any],
    expected_num_layers: int,
) -> None:
    """Bind the permission-checked facade to one resolved Gemma config."""

    if not isinstance(config, dict):
        raise RuntimeError("Gemma 3 model weights require a config dictionary")
    integer_fields = (
        "vocab_size",
        "hidden_size",
        "intermediate_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "head_dim",
        "max_position_embeddings",
        "sliding_window",
    )
    for field in integer_fields:
        if type(config.get(field)) is not int or config[field] <= 0:
            raise RuntimeError(f"Gemma 3 model weights require positive {field}")
    if expected_num_layers < 0 or config["num_hidden_layers"] != expected_num_layers:
        raise RuntimeError("Gemma 3 model weights disagree on the layer count")
    for field in (
        "rms_norm_eps",
        "query_pre_attn_scalar",
        "local_rope_theta",
        "global_rope_theta",
        "global_rope_factor",
    ):
        value = config.get(field)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or value <= 0
        ):
            raise RuntimeError(
                f"Gemma 3 model weights require finite positive {field}"
            )
        if (
            field in _FIXED_NUMERICAL_ASSUMPTIONS
            and float(value).hex()
            != _FIXED_NUMERICAL_ASSUMPTIONS[field].hex()
        ):
            raise RuntimeError(
                f"Gemma 3 model weights require exact {field}="
                f"{_FIXED_NUMERICAL_ASSUMPTIONS[field]!r}"
            )
    validate_model_weights_permission_contract(
        embed_weight,
        layers,
        attention_kinds,
        final_norm,
        lm_head,
        config["hidden_size"],
        config["sliding_window"],
        expected_num_layers,
    )
    if attention_kinds != list(config.get("layer_types", ())):
        raise RuntimeError("Gemma 3 attention kinds disagree with the text config")

    hidden = config["hidden_size"]
    intermediate = config["intermediate_size"]
    q_width = config["num_attention_heads"] * config["head_dim"]
    kv_width = config["num_key_value_heads"] * config["head_dim"]
    expected_global_shapes = {
        "embed_weight": (config["vocab_size"], hidden),
        "final_norm": (hidden,),
        "lm_head": (config["vocab_size"], hidden),
    }
    expected_layer_shapes = {
        "input_norm": (hidden,),
        "q_proj": (q_width, hidden),
        "k_proj": (kv_width, hidden),
        "v_proj": (kv_width, hidden),
        "q_norm": (config["head_dim"],),
        "k_norm": (config["head_dim"],),
        "o_proj": (hidden, q_width),
        "post_attn_norm": (hidden,),
        "pre_feedforward_norm": (hidden,),
        "gate_up_proj": (2 * intermediate, hidden),
        "down_proj": (hidden, intermediate),
        "post_feedforward_norm": (hidden,),
    }
    actual_global = {
        "embed_weight": embed_weight,
        "final_norm": final_norm,
        "lm_head": lm_head,
    }
    physical.validate_tensor_shapes(
        actual_global.items(), expected_global_shapes, label="Gemma 3 model weights"
    )
    for layer_index, layer in enumerate(layers):
        physical.validate_tensor_shapes(
            zip(GEMMA3_LAYER_WEIGHT_ROLES, layer), expected_layer_shapes,
            label=f"Gemma 3 model weights layer {layer_index}",
        )


# @kernel-bridge-end vosti_kernels::gemma3_physical_model_contract


__all__ = [
    "GEMMA3_LAYER_WEIGHT_ROLES",
    "validate_model_weights_permission_contract",
    "validate_model_weights_runtime_contract",
]
