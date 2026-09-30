"""Text-only Llama 3 physical weight contracts."""

from __future__ import annotations

import math
from typing import Any

import torch

from ... import physical


LLAMA3_LAYER_WEIGHT_ROLES = (
    "input_norm",
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "post_attn_norm",
    "gate_up_proj",
    "down_proj",
)

_FIXED_NUMERICAL_ASSUMPTIONS = {
    "rms_norm_eps": 1e-5,
    "rope_theta": 500_000.0,
    "rope_low_frequency_factor": 1.0,
    "rope_high_frequency_factor": 4.0,
    "rope_original_max_position_embeddings": 8192,
}
_FIXED_COMPOSITION_ASSUMPTIONS = {
    "model_type": "llama",
    "transformers_architecture": "LlamaForCausalLM",
    "hidden_act": "silu",
    "attention_bias": False,
    "attention_dropout": 0.0,
    "mlp_bias": False,
    "pretraining_tp": 1,
    "attention_kind": "full_attention",
    "rope_scaling_kind": "llama3",
}
_QUALIFIED_ROPE_FACTORS = {8.0.hex(), 32.0.hex()}


# @kernel-bridge-begin vosti_kernels::llama3_physical_model_contract
def validate_model_weights_permission_contract(
    embed_weight: torch.Tensor,
    layers: list,
    final_norm: torch.Tensor,
    lm_head: torch.Tensor,
    hidden_size: int,
    head_dim: int,
    expected_num_layers: int,
) -> None:
    """Check config-independent immutable Llama weight premises."""

    if expected_num_layers < 0:
        raise RuntimeError("Llama 3 model weights require a nonnegative layer count")
    if type(hidden_size) is not int or hidden_size <= 0:
        raise RuntimeError("Llama 3 model weights require a positive hidden size")
    if type(head_dim) is not int or head_dim <= 0 or head_dim % 2:
        raise RuntimeError("Llama 3 model weights require a positive even head dimension")
    if not isinstance(layers, list) or len(layers) != expected_num_layers:
        raise RuntimeError("Llama 3 model weights returned the wrong number of layers")
    named_tensors: list[tuple[str, torch.Tensor]] = [
        ("embed_weight", embed_weight),
        ("final_norm", final_norm),
        ("lm_head", lm_head),
    ]
    named_tensors.extend(physical.named_layer_tensors(
        layers, LLAMA3_LAYER_WEIGHT_ROLES, label="Llama 3 model weights"
    ))
    dtypes, devices = physical.validate_named_tensors(
        named_tensors, label="Llama 3 model weights"
    )
    if dtypes != {torch.bfloat16} or len(devices) != 1:
        raise RuntimeError("Llama 3 model weights require one BF16 dtype/device")

    # Model weights are immutable. The embedding and LM-head roles may be the
    # same tensor for a tied checkpoint; all other role aliasing is rejected.
    owners: dict[int, str] = {}
    for name, tensor in named_tensors:
        if tensor.numel() == 0:
            raise RuntimeError(f"Llama 3 model weights {name} is empty")
        pointer = tensor.untyped_storage().data_ptr()
        if pointer in owners:
            if {owners[pointer], name} == {"embed_weight", "lm_head"}:
                continue
            raise RuntimeError(
                f"Llama 3 model weights {owners[pointer]} and {name} share storage"
            )
        owners[pointer] = name

    if tuple(embed_weight.shape)[1:] != (hidden_size,):
        raise RuntimeError("Llama 3 embedding disagrees on the hidden size")
    vocab_size = int(embed_weight.shape[0])
    if tuple(lm_head.shape) != (vocab_size, hidden_size):
        raise RuntimeError("Llama 3 LM head has the wrong shape")
    if tuple(final_norm.shape) != (hidden_size,):
        raise RuntimeError("Llama 3 final norm has the wrong shape")

    role_index = {
        role: index for index, role in enumerate(LLAMA3_LAYER_WEIGHT_ROLES)
    }
    for layer_index, layer in enumerate(layers):
        for role in ("input_norm", "post_attn_norm"):
            if tuple(layer[role_index[role]].shape) != (hidden_size,):
                raise RuntimeError(
                    f"Llama 3 model weights layer {layer_index} {role} has the wrong shape"
                )
        q_proj = layer[role_index["q_proj"]]
        k_proj = layer[role_index["k_proj"]]
        v_proj = layer[role_index["v_proj"]]
        o_proj = layer[role_index["o_proj"]]
        gate_up = layer[role_index["gate_up_proj"]]
        down_proj = layer[role_index["down_proj"]]
        if any(
            tensor.dim() != 2
            for tensor in (q_proj, k_proj, v_proj, o_proj, gate_up, down_proj)
        ):
            raise RuntimeError(
                f"Llama 3 model weights layer {layer_index} projections must be rank-2"
            )
        q_width = int(q_proj.shape[0])
        kv_width = int(k_proj.shape[0])
        gate_width = int(gate_up.shape[0])
        if (
            tuple(q_proj.shape)[1:] != (hidden_size,)
            or tuple(k_proj.shape)[1:] != (hidden_size,)
            or tuple(v_proj.shape) != (kv_width, hidden_size)
            or q_width % head_dim
            or kv_width % head_dim
            or tuple(o_proj.shape) != (hidden_size, q_width)
            or gate_width <= 0
            or gate_width % 2
            or tuple(gate_up.shape)[1:] != (hidden_size,)
            or tuple(down_proj.shape) != (hidden_size, gate_width // 2)
        ):
            raise RuntimeError(
                f"Llama 3 model weights layer {layer_index} projection shapes are inconsistent"
            )


def validate_model_weights_runtime_contract(
    embed_weight: torch.Tensor,
    layers: list,
    final_norm: torch.Tensor,
    lm_head: torch.Tensor,
    config: dict[str, Any],
    expected_num_layers: int,
) -> None:
    """Bind the permission-checked facade to one resolved Llama config."""

    if not isinstance(config, dict):
        raise RuntimeError("Llama 3 model weights require a config dictionary")
    integer_fields = (
        "vocab_size",
        "hidden_size",
        "intermediate_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "head_dim",
        "max_position_embeddings",
        "rope_original_max_position_embeddings",
    )
    for field in integer_fields:
        if type(config.get(field)) is not int or config[field] <= 0:
            raise RuntimeError(f"Llama 3 model weights require positive {field}")
    if config["num_hidden_layers"] != expected_num_layers:
        raise RuntimeError("Llama 3 model weights disagree on the layer count")
    if config["num_attention_heads"] % config["num_key_value_heads"]:
        raise RuntimeError("Llama 3 model weights have invalid head geometry")
    for field in (
        "rms_norm_eps",
        "rope_theta",
        "rope_factor",
        "rope_low_frequency_factor",
        "rope_high_frequency_factor",
    ):
        value = config.get(field)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or value <= 0
        ):
            raise RuntimeError(
                f"Llama 3 model weights require finite positive {field}"
            )
    for field, expected in _FIXED_NUMERICAL_ASSUMPTIONS.items():
        value = config[field]
        if isinstance(expected, float):
            matches = float(value).hex() == expected.hex()
        else:
            matches = value == expected
        if not matches:
            raise RuntimeError(
                f"Llama 3 model weights require exact {field}={expected!r}"
            )
    if float(config["rope_factor"]).hex() not in _QUALIFIED_ROPE_FACTORS:
        raise RuntimeError("Llama 3 model weights require qualified rope_factor")
    for field, expected in _FIXED_COMPOSITION_ASSUMPTIONS.items():
        value = config.get(field)
        if value != expected or type(value) is not type(expected):
            raise RuntimeError(
                f"Llama 3 model weights require exact {field}={expected!r}"
            )
    tied = config.get("tie_word_embeddings")
    if type(tied) is not bool:
        raise RuntimeError(
            "Llama 3 model weights require boolean tie_word_embeddings"
        )
    if tied and lm_head is not embed_weight:
        raise RuntimeError(
            "Llama 3 model weights require the configured tied embedding"
        )
    if (
        not tied
        and embed_weight.numel() > 0
        and lm_head.numel() > 0
        and embed_weight.untyped_storage().data_ptr()
            == lm_head.untyped_storage().data_ptr()
    ):
        raise RuntimeError(
            "Llama 3 model weights require a distinct configured LM head"
        )

    validate_model_weights_permission_contract(
        embed_weight,
        layers,
        final_norm,
        lm_head,
        config["hidden_size"],
        config["head_dim"],
        expected_num_layers,
    )
    hidden = config["hidden_size"]
    intermediate = config["intermediate_size"]
    q_width = config["num_attention_heads"] * config["head_dim"]
    kv_width = config["num_key_value_heads"] * config["head_dim"]
    expected_global = {
        "embed_weight": (config["vocab_size"], hidden),
        "final_norm": (hidden,),
        "lm_head": (config["vocab_size"], hidden),
    }
    physical.validate_tensor_shapes((
        ("embed_weight", embed_weight),
        ("final_norm", final_norm),
        ("lm_head", lm_head),
    ), expected_global, label="Llama 3 model weights")
    expected_layer = {
        "input_norm": (hidden,),
        "q_proj": (q_width, hidden),
        "k_proj": (kv_width, hidden),
        "v_proj": (kv_width, hidden),
        "o_proj": (hidden, q_width),
        "post_attn_norm": (hidden,),
        "gate_up_proj": (2 * intermediate, hidden),
        "down_proj": (hidden, intermediate),
    }
    for layer_index, layer in enumerate(layers):
        physical.validate_tensor_shapes(
            zip(LLAMA3_LAYER_WEIGHT_ROLES, layer), expected_layer,
            label=f"Llama 3 model weights layer {layer_index}",
        )
# @kernel-bridge-end vosti_kernels::llama3_physical_model_contract


__all__ = [
    "LLAMA3_LAYER_WEIGHT_ROLES",
    "validate_model_weights_permission_contract",
    "validate_model_weights_runtime_contract",
]
