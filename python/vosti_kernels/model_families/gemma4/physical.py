"""Gemma-4 dense text weight bindings and heterogeneous full-retention KV caches."""

from typing import Any

import torch

from ... import physical
from ...attention_geometry import AttentionGeometry

# @kernel-bridge-begin vosti_kernels::gemma4_physical_model_contract
LAYER_WEIGHT_ROLES = (
    "input_norm", "q_proj", "k_proj", "v_proj", "q_norm", "k_norm", "o_proj",
    "post_attn_norm", "pre_feedforward_norm", "gate_up_proj", "down_proj",
    "post_feedforward_norm", "layer_scalar",
)


def validate_model_weights_permission_contract(
    embed_weight: torch.Tensor, layers: list, attention_kinds: list,
    final_norm: torch.Tensor, lm_head: torch.Tensor, hidden_size: int,
    expected_num_layers: int,
) -> None:
    """Check immutable roles without relying on one admitted model profile."""
    label = "Gemma 4 model weights"
    if type(expected_num_layers) is not int or expected_num_layers <= 0:
        raise RuntimeError(f"{label} require positive layers")
    if not isinstance(layers, list) or len(layers) != expected_num_layers:
        raise RuntimeError(f"{label} have the wrong number of layers")
    if (not isinstance(attention_kinds, list) or len(attention_kinds) != expected_num_layers
            or any(kind not in ("sliding_attention", "full_attention") for kind in attention_kinds)):
        raise RuntimeError(f"{label} have an invalid attention schedule")
    if lm_head is not embed_weight:
        raise RuntimeError(f"{label} require a tied embedding and LM head")
    named = [("embedding", embed_weight), ("final_norm", final_norm)]
    named.extend(physical.named_layer_tensors(
        layers, LAYER_WEIGHT_ROLES, label="Gemma 4 model weights"
    ))
    dtypes, devices = physical.validate_named_tensors(named, label=label)
    if dtypes != {torch.bfloat16} or len(devices) != 1:
        raise RuntimeError(f"{label} require one BF16 dtype/device")
    if (type(hidden_size) is not int or hidden_size <= 0 or embed_weight.ndim != 2
            or embed_weight.shape[0] <= 0 or embed_weight.shape[1] != hidden_size
            or final_norm.shape != (hidden_size,)):
        raise RuntimeError(f"{label} have invalid embedding/final-norm geometry")
    for i, layer in enumerate(layers):
        roles = dict(zip(LAYER_WEIGHT_ROLES, layer))
        if roles["q_norm"].ndim != 1 or roles["q_norm"].numel() == 0 or any(
            roles[name].ndim != 2 for name in ("q_proj", "k_proj", "down_proj")
        ):
            raise RuntimeError(f"{label} layer {i} has invalid projection/norm ranks")
        head_dim = roles["q_norm"].numel()
        q_width, kv_width = roles["q_proj"].shape[0], roles["k_proj"].shape[0]
        if q_width % head_dim or kv_width % head_dim:
            raise RuntimeError(f"{label} layer {i} has nonintegral head geometry")
        try:
            geometry = AttentionGeometry(q_width // head_dim, kv_width // head_dim, head_dim)
            expected = physical.four_norm_gated_layer_shapes(
                hidden_size, roles["down_proj"].shape[1], geometry, layer_scale=True)
        except ValueError as error:
            raise RuntimeError(f"{label} layer {i}: {error}") from error
        physical.validate_tensor_shapes(
            ((role, roles[role]) for role in expected), expected, label=f"{label} layer {i}"
        )


def validate_model_weights_runtime_contract(
    embed_weight: torch.Tensor, layers: list, attention_kinds: list,
    final_norm: torch.Tensor, lm_head: torch.Tensor, config: dict[str, Any],
    expected_num_layers: int,
) -> None:
    from .loader import config_from_runtime_dict

    try:
        resolved = config_from_runtime_dict(config)
    except (TypeError, ValueError) as error:
        raise RuntimeError(f"Gemma 4 model weights have invalid configuration: {error}") from error
    validate_model_weights_permission_contract(embed_weight, layers, attention_kinds,
        final_norm, lm_head, resolved.hidden_size, expected_num_layers)
    if resolved.num_hidden_layers != expected_num_layers or tuple(attention_kinds) != resolved.layer_types:
        raise RuntimeError("Gemma 4 model weights disagree with their layer schedule")
    if tuple(embed_weight.shape) != (resolved.vocab_size, resolved.hidden_size):
        raise RuntimeError("Gemma 4 model weights disagree with their vocabulary/hidden geometry")
    for i, layer in enumerate(layers):
        roles = dict(zip(LAYER_WEIGHT_ROLES, layer))
        shapes = physical.four_norm_gated_layer_shapes(resolved.hidden_size, resolved.intermediate_size,
            resolved.attention_geometry(i), layer_scale=True)
        physical.validate_tensor_shapes(
            ((role, roles[role]) for role in shapes), shapes,
            label=f"Gemma 4 model weights layer {i} configured geometry",
        )
        if resolved.shared_kv_projection(i) and roles["v_proj"] is not roles["k_proj"]:
            raise RuntimeError(f"Gemma 4 layer {i} must share its raw K/V projection weight")


# @kernel-bridge-end vosti_kernels::gemma4_physical_model_contract


def init_model_kv_caches(
    embed_weight: torch.Tensor, layers: list, config: dict[str, Any], token_capacity: int,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    from .loader import config_from_runtime_dict

    resolved = config_from_runtime_dict(config)
    if not isinstance(layers, list) or len(layers) != resolved.num_hidden_layers or any(
        not isinstance(layer, tuple) or len(layer) != len(LAYER_WEIGHT_ROLES) for layer in layers
    ):
        raise RuntimeError("Gemma 4 KV allocation has invalid layer roles")
    index = LAYER_WEIGHT_ROLES.index("k_proj")
    projections = tuple(layer[index] for layer in layers)
    dims = tuple(resolved.attention_geometry(i).head_dim for i in range(resolved.num_hidden_layers))
    # Bind to configured KV-head counts, not merely any divisible projection width.
    for i, projection in enumerate(projections):
        expected = (resolved.attention_geometry(i).kv_width, resolved.hidden_size)
        if tuple(projection.shape) != expected:
            raise RuntimeError(f"Gemma 4 KV layer {i} differs from configured geometry")
    return physical.init_layer_model_kv_caches(embed_weight, projections, dims, token_capacity)
