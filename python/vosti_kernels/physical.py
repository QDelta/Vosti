"""Architecture-neutral physical tensor and KV-cache helpers.

Model families own their weight roles and derive their cache geometry.  This
module only implements the common, source-attested checks and allocation
mechanics used after that geometry has been selected.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

import torch
# Read the fixed ABI without preloading the runtime's selectable triton_kernels
# package. Both namespaces refer to the same constants source in this checkout.
from kernels.triton_kernels.constants import PAGE_SIZE

from .attention_geometry import AttentionGeometry


# @kernel-bridge-begin vosti_kernels::physical_tensor_contracts
def blocks_needed(token_capacity: int) -> int:
    if token_capacity < 0:
        raise ValueError("KV-cache capacity must be nonnegative")
    if token_capacity == 0:
        return 0
    return (token_capacity - 1) // PAGE_SIZE + 1


def validate_named_tensors(
    named_tensors: Iterable[tuple[str, torch.Tensor]],
    *,
    label: str,
) -> tuple[set[torch.dtype], set[torch.device]]:
    """Validate common immutable-weight premises and return dtype/devices."""

    tensors = list(named_tensors)
    for name, tensor in tensors:
        if not isinstance(tensor, torch.Tensor):
            raise RuntimeError(f"{label} {name} is not a tensor")
        if not tensor.is_contiguous():
            raise RuntimeError(f"{label} {name} is not contiguous")
        if tensor.requires_grad:
            raise RuntimeError(f"{label} {name} still requires gradients")
    return (
        {tensor.dtype for _, tensor in tensors},
        {tensor.device for _, tensor in tensors},
    )


def named_layer_tensors(
    layers: list,
    roles: tuple[str, ...],
    *,
    label: str,
) -> list[tuple[str, torch.Tensor]]:
    """Check exact layer tuples before assigning family-owned role names.

    The caller validates layer count; dtype, storage and geometry policies are
    separate checks. Never silently truncate a malformed tuple with zip.
    """
    named = []
    for index, layer in enumerate(layers):
        if not isinstance(layer, tuple) or len(layer) != len(roles):
            raise RuntimeError(f"{label} layer {index} is not an exact role tuple")
        named.extend((f"layer {index} {role}", tensor) for role, tensor in zip(roles, layer))
    return named


def validate_tensor_shapes(
    named_tensors: Iterable[tuple[str, torch.Tensor]],
    expected_shapes: Mapping[str, tuple[int, ...]],
    *,
    label: str,
) -> None:
    """Check already permission-validated tensors against caller-owned geometry."""
    for role, tensor in named_tensors:
        expected = expected_shapes[role]
        if tuple(tensor.shape) != expected:
            raise RuntimeError(
                f"{label} {role} has shape {tuple(tensor.shape)}, expected {expected}"
            )


def four_norm_gated_layer_shapes(
    hidden: int, intermediate: int, geometry: AttentionGeometry, *, layer_scale: bool = False,
) -> dict[str, tuple[int, ...]]:
    """Weight roles shared by four-norm, gated dense decoder compositions."""
    if any(type(value) is not int or value <= 0 for value in (hidden, intermediate)):
        raise ValueError("dense layer widths must be positive integers")
    shapes = {role: (hidden,) for role in (
        "input_norm", "post_attn_norm", "pre_feedforward_norm", "post_feedforward_norm")}
    shapes.update(q_proj=(geometry.query_width, hidden), k_proj=(geometry.kv_width, hidden),
        v_proj=(geometry.kv_width, hidden), q_norm=(geometry.head_dim,),
        k_norm=(geometry.head_dim,), o_proj=(hidden, geometry.query_width),
        gate_up_proj=(2 * intermediate, hidden), down_proj=(hidden, intermediate))
    if layer_scale:
        shapes["layer_scalar"] = (1,)
    return shapes


def _cache_tail_shapes(
    num_layers: int,
    tail_shape: tuple[int, ...] | None,
    layer_tail_shapes: tuple[tuple[int, ...], ...] | None,
) -> tuple[tuple[int, ...], ...]:
    """Resolve either uniform or per-layer geometry, never both."""

    if type(num_layers) is not int or num_layers < 0:
        raise ValueError("KV-cache requires a nonnegative integer layer count")
    if (tail_shape is None) == (layer_tail_shapes is None):
        raise ValueError("KV-cache requires exactly one uniform or per-layer shape")
    shapes = (tail_shape,) * num_layers if layer_tail_shapes is None else layer_tail_shapes
    if not isinstance(shapes, tuple) or len(shapes) != num_layers:
        raise ValueError("KV-cache geometry must cover every layer")
    # Validate the uniform shape even for an empty collection.
    for shape in (shapes if tail_shape is None else (tail_shape,)):
        if not isinstance(shape, tuple) or any(
            type(dimension) is not int or dimension <= 0 for dimension in shape
        ):
            raise ValueError("KV-cache tail dimensions must be positive integers")
    return shapes


def allocate_kv_cache_collection(
    *,
    num_layers: int,
    token_capacity: int,
    tail_shape: tuple[int, ...] | None = None,
    layer_tail_shapes: tuple[tuple[int, ...], ...] | None = None,
    dtype: torch.dtype,
    device: str | torch.device,
    zero_initialize: bool = False,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Allocate one fresh, non-aliased K/V pair for every decoder layer."""

    if num_layers < 0 or token_capacity < 0:
        raise ValueError("KV-cache allocation requires nonnegative sizes")
    tails = _cache_tail_shapes(num_layers, tail_shape, layer_tail_shapes)
    factory = torch.zeros if zero_initialize else torch.empty
    return [
        (
            factory(shape, dtype=dtype, device=device).contiguous(),
            factory(shape, dtype=dtype, device=device).contiguous(),
        )
        for tail in tails
        for shape in [(blocks_needed(token_capacity), PAGE_SIZE, *tail)]
    ]


def validate_kv_cache_collection(
    caches: list,
    *,
    num_layers: int,
    token_capacity: int,
    tail_shape: tuple[int, ...] | None = None,
    layer_tail_shapes: tuple[tuple[int, ...], ...] | None = None,
    dtype: torch.dtype,
    device: str | torch.device,
    label: str,
) -> None:
    """Check shape, placement, contiguity, and allocation independence."""

    if num_layers < 0 or token_capacity < 0:
        raise RuntimeError(f"{label} requires nonnegative sizes")
    try:
        tails = _cache_tail_shapes(num_layers, tail_shape, layer_tail_shapes)
    except ValueError as error:
        raise RuntimeError(f"{label}: {error}") from error
    expected_device = torch.device(device)
    if not isinstance(caches, list) or len(caches) != num_layers:
        raise RuntimeError(f"{label} returned the wrong number of layers")

    tensors: list[tuple[str, torch.Tensor]] = []
    for layer, pair in enumerate(caches):
        expected_shape = (blocks_needed(token_capacity), PAGE_SIZE, *tails[layer])
        if not isinstance(pair, tuple) or len(pair) != 2:
            raise RuntimeError(f"{label} layer {layer} is not an exact K/V pair")
        for role, tensor in zip(("k", "v"), pair):
            name = f"layer {layer} {role}"
            if not isinstance(tensor, torch.Tensor):
                raise RuntimeError(f"{label} {name} is not a tensor")
            if tuple(tensor.shape) != expected_shape:
                raise RuntimeError(
                    f"{label} {name} has shape {tuple(tensor.shape)}, "
                    f"expected {expected_shape}"
                )
            if tensor.dtype != dtype or tensor.device != expected_device:
                raise RuntimeError(
                    f"{label} {name} has dtype/device "
                    f"{tensor.dtype}/{tensor.device}, expected {dtype}/{expected_device}"
                )
            if not tensor.is_contiguous():
                raise RuntimeError(f"{label} {name} is not contiguous")
            tensors.append((name, tensor))

    object_owners: dict[int, str] = {}
    storage_owners: dict[int, str] = {}
    for name, tensor in tensors:
        object_identity = id(tensor)
        if object_identity in object_owners:
            raise RuntimeError(
                f"{label} returned the same tensor object for "
                f"{object_owners[object_identity]} and {name}"
            )
        object_owners[object_identity] = name
        if tensor.numel() == 0:
            continue
        storage_identity = tensor.untyped_storage().data_ptr()
        if storage_identity in storage_owners:
            raise RuntimeError(
                f"{label} returned shared storage for "
                f"{storage_owners[storage_identity]} and {name}"
            )
        storage_owners[storage_identity] = name


def model_kv_cache_geometry(
    embed_weight: torch.Tensor,
    k_proj: torch.Tensor,
    head_dim: int,
) -> tuple[tuple[int, int], torch.dtype, torch.device]:
    """Project paged-KV geometry from weights and sealed decoder geometry."""

    dtypes, devices = validate_named_tensors(
        (
            ("embed_weight", embed_weight),
            ("k_proj", k_proj),
        ),
        label="model KV-cache geometry",
    )
    if len(dtypes) != 1 or len(devices) != 1:
        raise RuntimeError("model KV-cache geometry requires one dtype and device")
    if embed_weight.dim() != 2 or k_proj.dim() != 2:
        raise RuntimeError("model KV-cache geometry tensors have invalid ranks")
    if type(head_dim) is not int or head_dim <= 0:
        raise RuntimeError("model KV-cache head dimension must be positive")
    hidden = int(embed_weight.shape[1])
    kv_width = int(k_proj.shape[0])
    if (
        hidden <= 0
        or head_dim <= 0
        or kv_width <= 0
        or tuple(k_proj.shape)[1:] != (hidden,)
        or kv_width % head_dim != 0
    ):
        raise RuntimeError("model KV-cache geometry tensors are inconsistent")
    return (kv_width // head_dim, head_dim), dtypes.pop(), devices.pop()


def validate_model_kv_caches(
    caches: list,
    embed_weight: torch.Tensor,
    k_proj: torch.Tensor,
    head_dim: int,
    num_layers: int,
    token_capacity: int,
) -> None:
    if num_layers <= 0:
        raise RuntimeError("model KV-cache allocation requires positive layers")
    tail_shape, dtype, device = model_kv_cache_geometry(
        embed_weight, k_proj, head_dim
    )
    validate_kv_cache_collection(
        caches,
        num_layers=num_layers,
        token_capacity=token_capacity,
        tail_shape=tail_shape,
        dtype=dtype,
        device=device,
        label="model KV-cache allocator",
    )


def init_model_kv_caches(
    embed_weight: torch.Tensor,
    k_proj: torch.Tensor,
    head_dim: int,
    num_layers: int,
    token_capacity: int,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Allocate full causal storage from architecture-neutral weight roles."""

    if num_layers <= 0:
        raise ValueError("model KV-cache allocation requires positive layers")
    tail_shape, dtype, device = model_kv_cache_geometry(
        embed_weight, k_proj, head_dim
    )
    caches = allocate_kv_cache_collection(
        num_layers=num_layers,
        token_capacity=token_capacity,
        tail_shape=tail_shape,
        dtype=dtype,
        device=device,
    )
    validate_model_kv_caches(
        caches, embed_weight, k_proj, head_dim, num_layers, token_capacity
    )
    return caches


def layer_kv_cache_geometry(
    embed_weight: torch.Tensor,
    layer_k_projections: tuple[torch.Tensor, ...],
    head_dims: tuple[int, ...],
) -> tuple[tuple[tuple[int, int], ...], torch.dtype, torch.device]:
    """Derive every layer's static cache shape from its immutable projection."""
    if not isinstance(layer_k_projections, tuple) or not layer_k_projections or (
        not isinstance(head_dims, tuple) or len(layer_k_projections) != len(head_dims)
    ):
        raise ValueError("KV-cache geometry must cover every positive model layer")
    projected = [model_kv_cache_geometry(embed_weight, weight, dim)
                 for weight, dim in zip(layer_k_projections, head_dims)]
    # Each projection was checked against the same embedding dtype/device.
    return tuple(entry[0] for entry in projected), projected[0][1], projected[0][2]


def validate_layer_model_kv_caches(
    caches: list,
    embed_weight: torch.Tensor,
    layer_k_projections: tuple[torch.Tensor, ...],
    head_dims: tuple[int, ...],
    token_capacity: int,
) -> None:
    tails, dtype, device = layer_kv_cache_geometry(embed_weight, layer_k_projections, head_dims)
    validate_kv_cache_collection(caches, num_layers=len(tails), token_capacity=token_capacity,
        layer_tail_shapes=tails, dtype=dtype, device=device, label="model KV-cache allocator")


def init_layer_model_kv_caches(
    embed_weight: torch.Tensor,
    layer_k_projections: tuple[torch.Tensor, ...],
    head_dims: tuple[int, ...],
    token_capacity: int,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    tails, dtype, device = layer_kv_cache_geometry(embed_weight, layer_k_projections, head_dims)
    caches = allocate_kv_cache_collection(num_layers=len(tails), token_capacity=token_capacity,
        layer_tail_shapes=tails, dtype=dtype, device=device)
    validate_layer_model_kv_caches(caches, embed_weight, layer_k_projections, head_dims, token_capacity)
    return caches
# @kernel-bridge-end vosti_kernels::physical_tensor_contracts


__all__ = [
    "PAGE_SIZE",
    "allocate_kv_cache_collection",
    "blocks_needed",
    "four_norm_gated_layer_shapes",
    "init_layer_model_kv_caches",
    "init_model_kv_caches",
    "model_kv_cache_geometry",
    "named_layer_tensors",
    "layer_kv_cache_geometry",
    "validate_kv_cache_collection",
    "validate_model_kv_caches",
    "validate_layer_model_kv_caches",
    "validate_named_tensors",
    "validate_tensor_shapes",
]
