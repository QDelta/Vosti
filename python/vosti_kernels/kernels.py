"""Architecture-neutral primitive implementations for the trusted boundary.

The default path keeps small CPU-friendly tensors for unit tests.  A model
family runtime supplies the qualified implementation and immutable launch
plan explicitly; the primitive bodies remain shared and do not own family
geometry.
"""

from __future__ import annotations

import os

import torch

from . import primitive_runtime as PRIMITIVE_RUNTIME
from . import physical as PHYSICAL
from . import rotary as ROTARY

# All currently admitted paged-cache backends use 64-token pages. This shared
# geometry is intentionally independent of any one model-family runtime.
_BLOCK_SIZE = PHYSICAL.PAGE_SIZE
_INT32_MAX = (1 << 31) - 1

# @kernel-bridge-begin vosti_kernels::nonnegative_int32_guard
def _require_nonnegative_int32_values(values, name: str) -> None:
    """Reject index data that would change value when materialized as int32.

    Rust supplies ``u64`` indices, while the deployed CUDA kernels consume
    signed int32 tensors (and the KV scatter casts loaded slots to int32).
    Checking the Python integers before ``torch.tensor(..., dtype=int32)`` is
    essential: checking the tensor afterwards would only see wrapped values.
    """
    for value in values:
        index = int(value)
        if index < 0 or index > _INT32_MAX:
            raise OverflowError(
                f"{name} value {index} is outside the deployed int32 index range"
            )
# @kernel-bridge-end vosti_kernels::nonnegative_int32_guard


def _require_nonnegative_int32_tensor(tensor: torch.Tensor, name: str) -> None:
    """Validate a public tensor argument before a narrowing int32 conversion."""
    if tensor.numel() == 0:
        return
    minimum = int(tensor.min().item())
    maximum = int(tensor.max().item())
    if minimum < 0 or maximum > _INT32_MAX:
        raise OverflowError(
            f"{name} values [{minimum}, {maximum}] are outside the deployed "
            "int32 index range"
        )


# @kernel-bridge-begin vosti_kernels::linear
def linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    runtime: PRIMITIVE_RUNTIME.PrimitiveRuntime | None = None,
) -> torch.Tensor:
    """y = x @ weight.T

    Mirrors ``torch.nn.functional.linear`` with no bias.  Output is freshly
    allocated; inputs are not aliased.
    """
    if x.dim() != 2 or weight.dim() != 2:
        raise ValueError(f"linear expects 2D inputs, got x={x.shape} w={weight.shape}")
    if x.shape[1] != weight.shape[1]:
        raise ValueError(f"linear shape mismatch: x={x.shape} w={weight.shape}")
    if x.dtype != weight.dtype:
        raise ValueError(f"linear dtype mismatch: x={x.dtype} w={weight.dtype}")
    if x.device != weight.device:
        raise ValueError(f"linear device mismatch: x={x.device} w={weight.device}")
    vk = PRIMITIVE_RUNTIME.verified_for(runtime, x)
    if vk is not None:
        launch_config = PRIMITIVE_RUNTIME.static_launch_config(
            runtime, "linear",
            {"n": int(weight.shape[0]), "k": int(weight.shape[1])},
        )
        return PRIMITIVE_RUNTIME.kernel_entrypoint(runtime, vk, "linear")(
            x, weight.t(), launch_config=launch_config
        ).contiguous()
    return torch.matmul(x, weight.t()).contiguous()
# @kernel-bridge-end vosti_kernels::linear


# @kernel-bridge-begin vosti_kernels::qkv_linear
def qkv_linear(
    x: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    v_weight: torch.Tensor,
    runtime: PRIMITIVE_RUNTIME.PrimitiveRuntime | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute the bias-free Q, K, and V projections in one kernel launch."""

    weights = (q_weight, k_weight, v_weight)
    if x.dim() != 2 or any(weight.dim() != 2 for weight in weights):
        raise ValueError("qkv_linear expects four rank-2 tensors")
    if any(weight.shape[1] != x.shape[1] for weight in weights):
        raise ValueError("qkv_linear input and weight reductions differ")
    if x.shape[1] <= 0:
        raise ValueError("qkv_linear requires positive reduction width")
    if k_weight.shape[0] != v_weight.shape[0]:
        raise ValueError("qkv_linear K and V widths differ")
    if q_weight.shape[0] < k_weight.shape[0] or k_weight.shape[0] <= 0:
        raise ValueError("qkv_linear requires Q width >= KV width > 0")
    if any(weight.dtype != x.dtype for weight in weights):
        raise ValueError("qkv_linear input and weight dtypes differ")
    if any(weight.device != x.device for weight in weights):
        raise ValueError("qkv_linear input and weight devices differ")

    vk = PRIMITIVE_RUNTIME.verified_for(runtime, x)
    if vk is not None:
        key = {
            "q_width": int(q_weight.shape[0]),
            "kv_width": int(k_weight.shape[0]),
            "k": int(x.shape[1]),
        }
        launch_config = PRIMITIVE_RUNTIME.static_launch_config(
            runtime, "qkv_linear", key
        )
        outputs = PRIMITIVE_RUNTIME.kernel_entrypoint(
            runtime, vk, "qkv_linear"
        )(
            x,
            q_weight.t(),
            k_weight.t(),
            v_weight.t(),
            launch_config=launch_config,
        )
        return tuple(output.contiguous() for output in outputs)

    return tuple(
        torch.matmul(x, weight.t()).contiguous() for weight in weights
    )
# @kernel-bridge-end vosti_kernels::qkv_linear


def from_flat(rows: int, cols: int, data: list[float]) -> torch.Tensor:
    """Build a fresh, contiguous (rows, cols) f32 tensor from a flat list."""
    if len(data) != rows * cols:
        raise ValueError(f"from_flat: |data|={len(data)} != rows*cols={rows*cols}")
    return torch.tensor(data, dtype=torch.float32).reshape(rows, cols).contiguous()


# ---------------------------------------------------------------------------
# KV cache initialization.
#
# Allocates `num_layers` (k_cache, v_cache) pairs, each of shape
# (num_pages, page_size, num_kv_heads, head_dim). The CPU-only unit-test
# fallback deliberately collapses the last two axes.
#
# Soundness obligation (see docs/architecture.md): every returned tensor must
# be a freshly allocated, non-aliased tensor.  We allocate each tensor with a
# separate `torch.empty`/`torch.zeros` call and never use an aliasing view.
# The validator below checks the erased physical premises that are observable
# at runtime; the Python-object-to-Verus-ghost identity binding remains trusted.
# ---------------------------------------------------------------------------

def _blocks_needed_for(token_capacity: int) -> int:
    return PHYSICAL.blocks_needed(token_capacity)


# @kernel-bridge-begin vosti_kernels::init_kv_caches_runtime_contract
def _configured_kv_cache_contract(
    runtime: PRIMITIVE_RUNTIME.PrimitiveRuntime | None = None,
) -> tuple[tuple[int, ...], torch.dtype, str | torch.device, bool]:
    config = PRIMITIVE_RUNTIME.runtime_config_or_none(runtime)
    if config is None:
        return (), torch.float32, "cpu", True
    return (
        (int(config["num_kv_heads"]), int(config["head_dim"])),
        config["dtype"],
        config["device"],
        False,
    )


def _validate_init_kv_caches_runtime_contract(
    caches: list,
    num_layers: int,
    token_capacity: int,
    runtime: PRIMITIVE_RUNTIME.PrimitiveRuntime | None = None,
) -> None:
    """Check the common cache-collection permission premises."""

    tail_shape, dtype, device, _ = _configured_kv_cache_contract(runtime)
    PHYSICAL.validate_kv_cache_collection(
        caches,
        num_layers=num_layers,
        token_capacity=token_capacity,
        tail_shape=tail_shape,
        dtype=dtype,
        device=device,
        label="init_kv_caches",
    )
# @kernel-bridge-end vosti_kernels::init_kv_caches_runtime_contract


# @kernel-bridge-begin vosti_kernels::init_kv_caches
def init_kv_caches(
    num_layers: int,
    token_capacity: int,
    runtime: PRIMITIVE_RUNTIME.PrimitiveRuntime | None = None,
) -> list:
    """Allocate caches from an explicit capability or the CPU fallback."""

    tail_shape, dtype, device, zero_initialize = _configured_kv_cache_contract(
        runtime
    )
    caches = PHYSICAL.allocate_kv_cache_collection(
        num_layers=num_layers,
        token_capacity=token_capacity,
        tail_shape=tail_shape,
        dtype=dtype,
        device=device,
        zero_initialize=zero_initialize,
    )
    _validate_init_kv_caches_runtime_contract(
        caches, num_layers, token_capacity, runtime
    )
    return caches
# @kernel-bridge-end vosti_kernels::init_kv_caches


# ---------------------------------------------------------------------------
# Forward wrappers. The unconfigured/test-only CPU path remains intentionally
# small. A production family capability dispatches only through its attested
# contract catalog and immutable sealed launch plan.
# ---------------------------------------------------------------------------


# @kernel-bridge-begin vosti_kernels::embed
def embed(
    input_ids: torch.Tensor,
    weight: torch.Tensor,
    runtime: PRIMITIVE_RUNTIME.PrimitiveRuntime | None = None,
) -> torch.Tensor:
    if input_ids.dim() != 1 or weight.dim() != 2:
        raise ValueError(
            f"embed expects ids[rows] and weight[vocab, hidden], got "
            f"ids={input_ids.shape} weight={weight.shape}"
        )
    if input_ids.device != weight.device:
        raise ValueError(
            f"embed device mismatch: ids={input_ids.device} weight={weight.device}"
        )
    if PRIMITIVE_RUNTIME.runtime_config_or_none(runtime) is None:
        return torch.nn.functional.embedding(input_ids, weight).contiguous()
    cfg = PRIMITIVE_RUNTIME.runtime_config(runtime)
    if weight.shape[0] == 0 or weight.shape[1] != cfg["hidden_size"] or weight.dtype != cfg["dtype"]:
        raise ValueError(
            "embed weight disagrees with configured model metadata: "
            f"weight={tuple(weight.shape)}/{weight.dtype}, "
            f"hidden={cfg['hidden_size']}/dtype={cfg['dtype']}"
        )
    vk = PRIMITIVE_RUNTIME.verified_for(runtime, input_ids)
    if vk is not None:
        launch_config = PRIMITIVE_RUNTIME.static_launch_config(
            runtime, "embed", {"width": int(weight.shape[1])}
        )
        return PRIMITIVE_RUNTIME.kernel_entrypoint(runtime, vk, "embed")(
            input_ids, weight, launch_config=launch_config
        ).contiguous()
    return torch.nn.functional.embedding(input_ids, weight).contiguous()
# @kernel-bridge-end vosti_kernels::embed


# @kernel-bridge-begin vosti_kernels::rms_norm
def rms_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    runtime: PRIMITIVE_RUNTIME.PrimitiveRuntime | None = None,
) -> torch.Tensor:
    """RMSNorm over the last axis."""
    eps = (
        PRIMITIVE_RUNTIME.runtime_config(runtime)["rms_norm_eps"]
        if PRIMITIVE_RUNTIME.runtime_config_or_none(runtime) is not None
        else 1e-6
    )
    vk = PRIMITIVE_RUNTIME.verified_for(runtime, x)
    if vk is not None:
        if x.dim() != 2:
            raise RuntimeError("configured rms_norm requires a rank-2 tensor")
        if weight.numel() != x.shape[1]:
            raise RuntimeError("configured rms_norm weight width differs from input")
        launch_config = PRIMITIVE_RUNTIME.static_launch_config(
            runtime, "rms_norm", {"width": int(x.shape[1])}
        )
        return PRIMITIVE_RUNTIME.kernel_entrypoint(runtime, vk, "rms_norm")(
            x, weight.reshape(-1), eps, launch_config=launch_config
        ).contiguous()
    var = x.float().pow(2).mean(dim=-1, keepdim=True)
    x_norm = x * torch.rsqrt(var + eps)
    out = (x_norm * weight).to(x.dtype).contiguous()
    return out
# @kernel-bridge-end vosti_kernels::rms_norm


# @kernel-bridge-begin vosti_kernels::add_rms_norm
def add_rms_norm(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    runtime: PRIMITIVE_RUNTIME.PrimitiveRuntime | None = None,
):
    """Returns (normed, new_residual)."""
    vk = PRIMITIVE_RUNTIME.verified_for(runtime, x)
    if vk is not None:
        if x.dim() != 2:
            raise RuntimeError("configured add_rms_norm requires rank-2 tensors")
        if residual.shape != x.shape or weight.numel() != x.shape[1]:
            raise RuntimeError("configured add_rms_norm input, residual and weight shapes differ")
        eps = (
            PRIMITIVE_RUNTIME.runtime_config(runtime)["rms_norm_eps"]
            if PRIMITIVE_RUNTIME.runtime_config_or_none(runtime) is not None
            else 1e-6
        )
        launch_config = PRIMITIVE_RUNTIME.static_launch_config(
            runtime, "add_rms_norm", {"width": int(x.shape[1])}
        )
        normed, new_res = PRIMITIVE_RUNTIME.kernel_entrypoint(
            runtime, vk, "add_rms_norm"
        )(
            x,
            residual,
            weight.reshape(-1),
            eps,
            launch_config=launch_config,
        )
        return (normed.contiguous(), new_res.contiguous())
    new_res = (x + residual).contiguous()
    normed = rms_norm(new_res, weight, runtime)
    return (normed, new_res)
# @kernel-bridge-end vosti_kernels::add_rms_norm


# @kernel-bridge-begin vosti_kernels::qk_norm
def qk_norm(
    q: torch.Tensor,
    k: torch.Tensor,
    q_norm_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    runtime: PRIMITIVE_RUNTIME.PrimitiveRuntime | None = None,
):
    """Returns (nq, nk)."""
    if PRIMITIVE_RUNTIME.runtime_config_or_none(runtime) is not None:
        cfg = PRIMITIVE_RUNTIME.runtime_config(runtime)
        vk = PRIMITIVE_RUNTIME.verified_for(runtime, q)
        if vk is not None:
            eps = cfg["rms_norm_eps"]
            norm = PRIMITIVE_RUNTIME.kernel_entrypoint(
                runtime, vk, "qk_norm"
            )
            q_launch_config = PRIMITIVE_RUNTIME.static_launch_config(
                runtime, "qk_norm",
                {"heads": cfg["num_heads"], "head_dim": cfg["head_dim"]},
            )
            nq = norm(
                q.reshape(q.shape[0], -1), q_norm_weight.reshape(-1),
                cfg["num_heads"], eps, launch_config=q_launch_config,
            ).reshape(-1, cfg["num_heads"], cfg["head_dim"])
            k_launch_config = PRIMITIVE_RUNTIME.static_launch_config(
                runtime, "qk_norm",
                {"heads": cfg["num_kv_heads"], "head_dim": cfg["head_dim"]},
            )
            nk = norm(
                k.reshape(k.shape[0], -1), k_norm_weight.reshape(-1),
                cfg["num_kv_heads"], eps, launch_config=k_launch_config,
            ).reshape(-1, cfg["num_kv_heads"], cfg["head_dim"])
            return (nq.contiguous(), nk.contiguous())
        q = q.reshape(-1, cfg["num_heads"], cfg["head_dim"])
        k = k.reshape(-1, cfg["num_kv_heads"], cfg["head_dim"])
    nq = rms_norm(q, q_norm_weight, runtime)
    nk = rms_norm(k, k_norm_weight, runtime)
    return (nq, nk)
# @kernel-bridge-end vosti_kernels::qk_norm


# @kernel-bridge-begin vosti_kernels::view_as_kv
def view_as_kv(
    v: torch.Tensor,
    runtime: PRIMITIVE_RUNTIME.PrimitiveRuntime | None = None,
) -> torch.Tensor:
    """Return a reshaped tensor with storage independent from ``v``.

    Fresh storage is part of the trusted Verus permission contract.  A bare
    ``reshape(...).contiguous()`` is insufficient because ``contiguous()`` may
    return the reshape view unchanged when its layout is already contiguous.
    """
    if PRIMITIVE_RUNTIME.runtime_config_or_none(runtime) is not None:
        cfg = PRIMITIVE_RUNTIME.runtime_config(runtime)
        return v.reshape(
            -1, cfg["num_kv_heads"], cfg["head_dim"]
        ).clone().contiguous()
    return v.clone().contiguous()
# @kernel-bridge-end vosti_kernels::view_as_kv


# @kernel-bridge-begin vosti_kernels::row_layout_adapters
def merge_attention_heads(x: torch.Tensor) -> torch.Tensor:
    """Flatten per-token attention heads into a fresh dense row tensor."""

    if x.dim() != 3:
        raise ValueError("merge_attention_heads requires a rank-3 tensor")
    return x.reshape(x.shape[0], -1).clone().contiguous()


def split_last_axis_halves(
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Copy the two equal halves of an even-width row tensor."""

    if x.dim() != 2 or x.shape[1] % 2 != 0:
        raise ValueError(
            "split_last_axis_halves requires an even-width rank-2 tensor"
        )
    middle = x.shape[1] // 2
    return (
        x[:, :middle].clone().contiguous(),
        x[:, middle:].clone().contiguous(),
    )
# @kernel-bridge-end vosti_kernels::row_layout_adapters


# @kernel-bridge-begin vosti_kernels::rotary_embed
def rotary_embed(
    positions: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    runtime: PRIMITIVE_RUNTIME.RotaryRuntime | None = None,
):
    if PRIMITIVE_RUNTIME.runtime_config_or_none(runtime) is None:
        return (q.clone().contiguous(), k.clone().contiguous())
    cfg = PRIMITIVE_RUNTIME.runtime_config(runtime)
    vk = PRIMITIVE_RUNTIME.verified_for(runtime, q)
    if vk is not None:
        # Structurally certified rope kernel (half-rotation form matches
        # transformers rotate_half in executable differential tests).
        # Table selection stays wrapper glue: cos/sin rows are pure functions
        # of each token's position and the sealed model policy.
        hd = cfg["head_dim"]
        cos_t, sin_t = PRIMITIVE_RUNTIME.rotary_tables(
            runtime, positions, q.dtype
        )

        def rot(x: torch.Tensor, heads: int) -> torch.Tensor:
            m = x.shape[0]
            xf = x.reshape(m * heads, hd)
            cos_r = cos_t.repeat_interleave(heads, dim=0)
            sin_r = sin_t.repeat_interleave(heads, dim=0)
            launch_config = PRIMITIVE_RUNTIME.static_launch_config(
                runtime, "rotary_embed", {"width": hd}
            )
            return PRIMITIVE_RUNTIME.kernel_entrypoint(
                runtime, vk, "rotary_embed"
            )(
                xf, cos_r, sin_r, launch_config=launch_config
            ).reshape(m, heads, hd)

        nq = rot(q.reshape(q.shape[0], cfg["num_heads"], hd), cfg["num_heads"])
        nk = rot(k.reshape(k.shape[0], cfg["num_kv_heads"], hd), cfg["num_kv_heads"])
        return (nq.contiguous(), nk.contiguous())
    cos_half, sin_half = ROTARY.tables(
        positions,
        head_dim=cfg["head_dim"],
        theta=cfg["rope_theta"],
        scaling=ROTARY.runtime_scaling(cfg),
        dtype=q.dtype,
    )
    cos = torch.cat((cos_half, cos_half), dim=-1).unsqueeze(0)
    sin = torch.cat((sin_half, sin_half), dim=-1).unsqueeze(0)
    q_t = q.transpose(0, 1).unsqueeze(0)
    k_t = k.transpose(0, 1).unsqueeze(0)
    def rotate_half(x: torch.Tensor) -> torch.Tensor:
        half = x.shape[-1] // 2
        return torch.cat((-x[..., half:], x[..., :half]), dim=-1)

    q_t = q_t * cos.unsqueeze(1) + rotate_half(q_t) * sin.unsqueeze(1)
    k_t = k_t * cos.unsqueeze(1) + rotate_half(k_t) * sin.unsqueeze(1)
    return q_t.squeeze(0).transpose(0, 1).contiguous(), k_t.squeeze(0).transpose(0, 1).contiguous()
# @kernel-bridge-end vosti_kernels::rotary_embed


# @kernel-bridge-begin vosti_kernels::silu_and_mul
def silu_and_mul(
    x: torch.Tensor,
    runtime: PRIMITIVE_RUNTIME.PrimitiveRuntime | None = None,
) -> torch.Tensor:
    """Splits last axis in half: y = silu(x[..., :half]) * x[..., half:].
    """
    half = x.shape[-1] // 2
    if half == 0 and PRIMITIVE_RUNTIME.runtime_config_or_none(runtime) is None:
        return torch.zeros_like(x).contiguous()
    a = x[..., :half]
    b = x[..., half:]
    vk = PRIMITIVE_RUNTIME.verified_for(runtime, x)
    if vk is not None:
        if x.dim() != 2 or half <= 0 or x.shape[-1] != 2 * half:
            raise RuntimeError("configured silu_and_mul requires rank-2 even-width rows")
        launch_config = PRIMITIVE_RUNTIME.static_launch_config(
            runtime, "silu_and_mul", {"width": int(half)}
        )
        return PRIMITIVE_RUNTIME.kernel_entrypoint(
            runtime, vk, "silu_and_mul"
        )(
            a, b, launch_config=launch_config
        ).contiguous()
    return (torch.nn.functional.silu(a) * b).contiguous()
# @kernel-bridge-end vosti_kernels::silu_and_mul


# Tensor materializers: exact trusted conversions from exec-side lists. The
# source-attested Verus bridge records the corresponding ghost representations;
# it does not turn Python/Torch allocation semantics into a formal proof.

# @kernel-bridge-begin vosti_kernels::step_plan_materializers
def _index_tensor_options(
    *,
    device_anchor: torch.Tensor | None = None,
    force_int64: bool = False,
    runtime: PRIMITIVE_RUNTIME.PrimitiveRuntime | None = None,
):
    if device_anchor is not None:
        if not isinstance(device_anchor, torch.Tensor) or not device_anchor.is_cuda:
            raise ValueError("step-plan device anchor must be a CUDA tensor")
        device = device_anchor.device
        dtype = torch.int64 if force_int64 else torch.int32
        return dtype, device
    config = PRIMITIVE_RUNTIME.runtime_config_or_none(runtime)
    device = config["device"] if config is not None else None
    dtype = (
        torch.int64
        if force_int64 or config is None
        else torch.int32
    )
    return dtype, device


def _flat_index_tensor(
    values: list[int],
    name: str,
    *,
    device_anchor: torch.Tensor | None = None,
    force_int64: bool = False,
    runtime: PRIMITIVE_RUNTIME.PrimitiveRuntime | None = None,
) -> torch.Tensor:
    dtype, device = _index_tensor_options(
        device_anchor=device_anchor,
        force_int64=force_int64,
        runtime=runtime,
    )
    if dtype == torch.int32:
        _require_nonnegative_int32_values(values, name)
    return torch.tensor(values, dtype=dtype, device=device).contiguous()


def token_tensor(
    tokens: list[int],
    device_anchor: torch.Tensor | None = None,
    runtime: PRIMITIVE_RUNTIME.PrimitiveRuntime | None = None,
) -> torch.Tensor:
    return _flat_index_tensor(
        tokens, "input_ids", device_anchor=device_anchor, runtime=runtime
    )


def position_tensor(
    positions: list[int],
    device_anchor: torch.Tensor | None = None,
    runtime: PRIMITIVE_RUNTIME.PrimitiveRuntime | None = None,
) -> torch.Tensor:
    return _flat_index_tensor(
        positions,
        "positions",
        device_anchor=device_anchor,
        force_int64=True,
        runtime=runtime,
    )


def slot_tensor(
    slot_mapping: list[int],
    device_anchor: torch.Tensor | None = None,
    runtime: PRIMITIVE_RUNTIME.PrimitiveRuntime | None = None,
) -> torch.Tensor:
    return _flat_index_tensor(
        slot_mapping, "slot_mapping",
        device_anchor=device_anchor, runtime=runtime,
    )


def block_tables_tensor(
    block_ids: list[list[int]],
    device_anchor: torch.Tensor | None = None,
    runtime: PRIMITIVE_RUNTIME.PrimitiveRuntime | None = None,
) -> torch.Tensor:
    dtype, device = _index_tensor_options(
        device_anchor=device_anchor, runtime=runtime
    )
    if dtype == torch.int32:
        for row in block_ids:
            _require_nonnegative_int32_values(row, "block_table")
    if len(block_ids) == 0:
        return torch.zeros((0, 0), dtype=dtype, device=device).contiguous()
    max_len = max(len(row) for row in block_ids)
    # The verified paged-attention annotation requires every rectangular table
    # entry to be a valid nonnegative page id, including padding it never reads.
    # Page 0 is a safe sentinel whenever the engine has a nonempty cache.
    out = torch.zeros((len(block_ids), max_len), dtype=dtype, device=device)
    for i, row in enumerate(block_ids):
        if row:
            out[i, : len(row)] = torch.tensor(row, dtype=dtype, device=device)
    return out.contiguous()


def seq_lens_tensor(
    lengths: list[int],
    device_anchor: torch.Tensor | None = None,
    runtime: PRIMITIVE_RUNTIME.PrimitiveRuntime | None = None,
) -> torch.Tensor:
    return _flat_index_tensor(
        lengths, "cu_seqlens", device_anchor=device_anchor, runtime=runtime
    )
# @kernel-bridge-end vosti_kernels::step_plan_materializers


# Mutating: store_kv_cache.  Writes (k, v) rows into k_cache, v_cache at
# distinct nonnegative physical slots from slot_mapping.  Slot -1 is the sole
# admitted no-write sentinel used by covering CUDA-graph pad rows.  Exact
# scatter behavior is a reviewed TCB item; this guard pins its complete
# physical representation.

# @kernel-bridge-begin vosti_kernels::store_kv_cache_runtime_contract
def _store_kv_cache_runtime_contract(
    k: torch.Tensor,
    v: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    runtime: PRIMITIVE_RUNTIME.PrimitiveRuntime | None = None,
) -> None:
    if k.dim() != 3 or v.dim() != 3:
        raise RuntimeError("verified store_kv_cache requires 3D K/V rows")
    if k_cache.dim() != 4 or v_cache.dim() != 4:
        raise RuntimeError("verified store_kv_cache requires 4D paged K/V caches")
    if k.shape != v.shape:
        raise RuntimeError("verified store_kv_cache requires matching K/V row shapes")
    if k_cache.shape != v_cache.shape:
        raise RuntimeError("verified store_kv_cache requires matching K/V cache shapes")
    if int(k_cache.shape[0]) <= 0 or int(k_cache.shape[1]) != _BLOCK_SIZE:
        raise RuntimeError(
            f"verified store_kv_cache requires nonempty {_BLOCK_SIZE}-token pages"
        )
    if tuple(k.shape[1:]) != tuple(k_cache.shape[2:]):
        raise RuntimeError(
            "verified store_kv_cache requires source rows to match cache rows"
        )

    data_tensors = (k, v, k_cache, v_cache)
    if len({tensor.dtype for tensor in data_tensors}) != 1:
        raise RuntimeError("verified store_kv_cache forbids implicit dtype conversion")
    if len({tensor.device for tensor in (*data_tensors, slot_mapping)}) != 1:
        raise RuntimeError("verified store_kv_cache requires one tensor device")
    if any(not tensor.is_contiguous() for tensor in (*data_tensors, slot_mapping)):
        raise RuntimeError("verified store_kv_cache requires contiguous tensors")
    if slot_mapping.dim() != 1 or slot_mapping.dtype != torch.int32:
        raise RuntimeError("verified store_kv_cache requires a 1D int32 slot mapping")

    if PRIMITIVE_RUNTIME.runtime_config_or_none(runtime) is not None:
        cfg = PRIMITIVE_RUNTIME.runtime_config(runtime)
        expected_tail = (cfg["num_kv_heads"], cfg["head_dim"])
        if tuple(k.shape[1:]) != expected_tail:
            raise RuntimeError("verified store_kv_cache K/V geometry is outside scope")
        if k.dtype != cfg["dtype"] or k.device != torch.device(cfg["device"]):
            raise RuntimeError("verified store_kv_cache dtype/device is outside scope")

    storage_owners: dict[int, str] = {}
    for name, tensor in (
        ("k", k), ("v", v), ("k_cache", k_cache),
        ("v_cache", v_cache), ("slot_mapping", slot_mapping),
    ):
        pointer = tensor.untyped_storage().data_ptr()
        if pointer in storage_owners:
            raise RuntimeError(
                "verified store_kv_cache requires independent storage for "
                f"{storage_owners[pointer]} and {name}"
            )
        storage_owners[pointer] = name

    slots = [int(slot) for slot in slot_mapping.detach().cpu().tolist()]
    if len(slots) != int(k.shape[0]):
        raise RuntimeError("verified store_kv_cache requires one slot per input row")
    num_slots = int(k_cache.shape[0]) * _BLOCK_SIZE
    if any(slot < -1 or slot >= num_slots for slot in slots):
        raise RuntimeError(
            "verified store_kv_cache requires every slot to be -1 or in cache bounds"
        )
    active_slots = [slot for slot in slots if slot >= 0]
    if len(set(active_slots)) != len(active_slots):
        raise RuntimeError(
            "verified store_kv_cache requires injective nonnegative slots"
        )
# @kernel-bridge-end vosti_kernels::store_kv_cache_runtime_contract


# @kernel-bridge-begin vosti_kernels::store_kv_cache
def store_kv_cache(
    k: torch.Tensor,
    v: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    runtime: PRIMITIVE_RUNTIME.PrimitiveRuntime | None = None,
) -> None:
    if k.dim() == 0:
        raise RuntimeError("store_kv_cache requires a row axis")
    if k.shape[0] == 0:
        return
    vk = PRIMITIVE_RUNTIME.verified_for(runtime, k)
    if vk is not None:
        if k_cache.dim() != 4:
            raise RuntimeError("configured store_kv_cache requires a rank-4 cache")
        kvd = int(k.reshape(k.shape[0], -1).shape[1])
        launch_config = PRIMITIVE_RUNTIME.static_launch_config(
            runtime, "store_kv_cache", {"kvd": kvd}
        )
        _store_kv_cache_runtime_contract(
            k, v, k_cache, v_cache, slot_mapping, runtime
        )
        PRIMITIVE_RUNTIME.kernel_entrypoint(
            runtime, vk, "store_kv_cache"
        )(
            k,
            v,
            k_cache,
            v_cache,
            slot_mapping,
            launch_config=launch_config,
        )
        return
    if k_cache.dim() == 4:
        page_size = int(k_cache.shape[1])
        for i, s_raw in enumerate(slot_mapping.tolist()):
            s = int(s_raw)
            page = s // page_size
            offset = s % page_size
            if 0 <= page < int(k_cache.shape[0]) and 0 <= offset < page_size:
                k_cache[page, offset].copy_(k[i])
                v_cache[page, offset].copy_(v[i])
        return
    if k_cache.dim() < 2:
        raise RuntimeError("store_kv_cache requires a paged cache")
    page_size = int(k_cache.shape[1])
    slots = slot_mapping.tolist()
    # Match store_kv_cache_repr's total fallback semantics: later input rows
    # overwrite earlier rows when a caller supplies a repeated slot. Verified
    # engine launches require injective slots, so that case never reaches the
    # attested Triton path.
    for i, s_raw in enumerate(slots):
        s = int(s_raw)
        page = s // page_size
        offset = s % page_size
        if 0 <= page < int(k_cache.shape[0]) and 0 <= offset < page_size:
            # Reduce to scalar to fit the (num_pages, page_size) toy shape.
            k_cache[page, offset] = k[i].mean().to(k_cache.dtype)
            v_cache[page, offset] = v[i].mean().to(v_cache.dtype)
# @kernel-bridge-end vosti_kernels::store_kv_cache


# @kernel-bridge-begin vosti_kernels::store_kv_cache_from_verified_caller
def store_kv_cache_from_verified_caller(
    k: torch.Tensor,
    v: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    runtime: PRIMITIVE_RUNTIME.PrimitiveRuntime | None = None,
) -> None:
    """Launch KV scatter for the verified engine without device-value reads.

    The Rust caller proves ``store_kv_cache_launch_ready``.  The source-pinned
    step-plan materializer supplies the exact int32 slot tensor, while the
    cache allocator and preceding model kernels establish the physical tensor
    roles.  Re-reading slots on the host here would only re-check those
    producer invariants once per layer and would synchronize CUDA.

    Callers without that proof/provenance must use :func:`store_kv_cache`.
    """
    if k.shape[0] == 0:
        return
    vk = PRIMITIVE_RUNTIME.verified_for(runtime, k)
    if vk is not None:
        if k_cache.dim() != 4:
            raise RuntimeError("configured store_kv_cache requires a rank-4 cache")
        kvd = int(k.reshape(k.shape[0], -1).shape[1])
        launch_config = PRIMITIVE_RUNTIME.static_launch_config(
            runtime, "store_kv_cache", {"kvd": kvd}
        )
        PRIMITIVE_RUNTIME.kernel_entrypoint(
            runtime, vk, "store_kv_cache"
        )(
            k,
            v,
            k_cache,
            v_cache,
            slot_mapping,
            launch_config=launch_config,
        )
        return
    # CPU and toy-shape tests retain the checked public behavior. The configured
    # CUDA engine is fail closed and therefore cannot take this branch.
    store_kv_cache(k, v, k_cache, v_cache, slot_mapping, runtime)
# @kernel-bridge-end vosti_kernels::store_kv_cache_from_verified_caller


# @kernel-bridge-begin vosti_kernels::paged_attention_runtime_contract
def _validate_paged_attention_runtime_contract(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    block_table: torch.Tensor,
    runtime: PRIMITIVE_RUNTIME.PrimitiveRuntime,
) -> None:
    """Fail closed unless a launch realizes the imported logical contract.

    Verus intentionally erases physical head axes, dtype, device, and strides.
    This check is the executable half of that representation adapter: it binds
    the logical ragged rows/pages to the exact dense CUDA layout consumed by
    the pinned Triton kernel.  It does not discharge the separate finite-value
    numeric-domain premise.
    """

    cfg = PRIMITIVE_RUNTIME.runtime_config(runtime)
    if q.dim() != 3:
        raise RuntimeError("verified paged_attention requires q[total_q, heads, head_dim]")
    if k_cache.dim() != 4 or v_cache.dim() != 4:
        raise RuntimeError(
            "verified paged_attention requires rank-4 paged K/V caches"
        )
    if cu_seqlens_q.dim() != 1 or cu_seqlens_k.dim() != 1:
        raise RuntimeError("verified paged_attention requires rank-1 cumulative lengths")
    if block_table.dim() != 2:
        raise RuntimeError("verified paged_attention requires a rank-2 block table")

    expected_q_tail = (cfg["num_heads"], cfg["head_dim"])
    expected_cache_tail = (_BLOCK_SIZE, cfg["num_kv_heads"], cfg["head_dim"])
    if tuple(q.shape[1:]) != expected_q_tail:
        raise RuntimeError(
            "verified paged_attention query geometry disagrees with deployed scope"
        )
    if tuple(k_cache.shape[1:]) != expected_cache_tail:
        raise RuntimeError(
            "verified paged_attention key-cache geometry disagrees with deployed scope"
        )
    if tuple(v_cache.shape) != tuple(k_cache.shape):
        raise RuntimeError("verified paged_attention K/V cache shapes must match")
    if int(k_cache.shape[0]) <= 0:
        raise RuntimeError("verified paged_attention requires a nonempty cache pool")

    data_tensors = (q, k_cache, v_cache)
    if any(t.dtype != cfg["dtype"] for t in data_tensors):
        raise RuntimeError(
            "verified paged_attention Q/K/V dtype disagrees with deployed runtime"
        )
    metadata_tensors = (cu_seqlens_q, cu_seqlens_k, block_table)
    if any(t.dtype != torch.int32 for t in metadata_tensors):
        raise RuntimeError(
            "verified paged_attention metadata tensors must have int32 dtype"
        )

    if cu_seqlens_q.numel() != cu_seqlens_k.numel():
        raise RuntimeError("verified paged_attention cumulative-length sizes must match")
    batch = int(cu_seqlens_q.numel()) - 1
    if batch <= 0 or int(block_table.shape[0]) != batch:
        raise RuntimeError(
            "verified paged_attention block-table batch must match a nonempty launch"
        )
    table_width = int(block_table.shape[1])
    if table_width <= 0:
        raise RuntimeError("verified paged_attention requires nonempty block-table rows")

    cu_q = [int(value) for value in cu_seqlens_q.detach().cpu().tolist()]
    cu_k = [int(value) for value in cu_seqlens_k.detach().cpu().tolist()]
    if cu_q[0] != 0 or cu_k[0] != 0 or cu_q[-1] != int(q.shape[0]):
        raise RuntimeError(
            "verified paged_attention cumulative lengths must start at zero and close q"
        )
    q_lens = [right - left for left, right in zip(cu_q, cu_q[1:])]
    k_lens = [right - left for left, right in zip(cu_k, cu_k[1:])]
    if any(q_len <= 0 or k_len <= 0 for q_len, k_len in zip(q_lens, k_lens)):
        raise RuntimeError("verified paged_attention requires strictly increasing lengths")
    if any(q_len > k_len for q_len, k_len in zip(q_lens, k_lens)):
        raise RuntimeError("verified causal paged_attention requires q_len <= k_len")

    max_q = int(max_seqlen_q)
    max_k = int(max_seqlen_k)
    if not (0 < max_q <= _INT32_MAX and 0 < max_k <= _INT32_MAX):
        raise RuntimeError("verified paged_attention maxima must be positive int32 values")
    if max(q_lens) > max_q or max(k_lens) > max_k:
        raise RuntimeError("verified paged_attention maxima do not cover cumulative lengths")
    if any(_blocks_needed_for(k_len) > table_width for k_len in k_lens):
        raise RuntimeError("verified paged_attention block table does not cover every KV row")

    minimum_block = int(block_table.min().item())
    maximum_block = int(block_table.max().item())
    if minimum_block < 0 or maximum_block >= int(k_cache.shape[0]):
        raise RuntimeError(
            "verified paged_attention requires every padded block-table entry "
            "to name an in-bounds cache page"
        )
    if any(t.device != q.device for t in data_tensors):
        raise RuntimeError("verified paged_attention Q/K/V devices must match")
    if not q.is_cuda:
        raise RuntimeError("verified paged_attention requires CUDA Q/K/V tensors")
    if any(not t.is_contiguous() for t in data_tensors):
        raise RuntimeError("verified paged_attention requires dense contiguous Q/K/V tensors")
# @kernel-bridge-end vosti_kernels::paged_attention_runtime_contract


# @kernel-bridge-begin vosti_kernels::paged_attention
def paged_attention(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    block_table: torch.Tensor,
    runtime: PRIMITIVE_RUNTIME.PrimitiveRuntime | None = None,
) -> torch.Tensor:
    """Checked paged attention; there is no unverified implementation fallback."""
    cfg = PRIMITIVE_RUNTIME.runtime_config(runtime)
    vk = PRIMITIVE_RUNTIME.verified_for(runtime, q)
    if k_cache.dim() != 4:
        raise RuntimeError("configured paged_attention requires rank-4 caches")
    # This wrapper is public and can be called with tensors not produced by
    # the guarded materializers above.  Validate before every narrowing cast.
    _require_nonnegative_int32_tensor(cu_seqlens_q, "cu_seqlens_q")
    _require_nonnegative_int32_tensor(cu_seqlens_k, "cu_seqlens_k")
    _require_nonnegative_int32_tensor(block_table, "block_table")
    if vk is None:
        raise RuntimeError("paged_attention requires a verified kernel binding")
    launch_config = PRIMITIVE_RUNTIME.static_launch_config(
        runtime, "paged_attention", {"head_dim": int(q.shape[-1])},
    )
    _validate_paged_attention_runtime_contract(
        q, k_cache, v_cache, cu_seqlens_q, cu_seqlens_k,
        max_seqlen_q, max_seqlen_k, block_table, runtime,
    )
    out = PRIMITIVE_RUNTIME.kernel_entrypoint(runtime, vk, "paged_attention")(
        q, k_cache, v_cache,
        cu_seqlens_q.to(dtype=torch.int32),
        cu_seqlens_k.to(dtype=torch.int32),
        int(max_seqlen_q), int(max_seqlen_k),
        softmax_scale=cfg["head_dim"] ** -0.5,
        block_table=block_table.to(dtype=torch.int32),
        launch_config=launch_config,
    )
    return out.reshape(out.shape[0], -1).contiguous()
# @kernel-bridge-end vosti_kernels::paged_attention


# @kernel-bridge-begin vosti_kernels::paged_attention_from_verified_caller
def paged_attention_from_verified_caller(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    block_table: torch.Tensor,
    runtime: PRIMITIVE_RUNTIME.PrimitiveRuntime | None = None,
) -> torch.Tensor:
    """Launch paged attention from proved, engine-produced metadata.

    ``paged_attention_launch_ready`` proves cumulative-length endpoints,
    strictness, maxima, page coverage, and logical page bounds.  The attested
    materializers establish the int32 CUDA representation and valid page-zero
    padding.  This path therefore neither copies metadata to the host nor
    invokes ``.item()`` on it.  Unverified callers use :func:`paged_attention`.
    """
    cfg = PRIMITIVE_RUNTIME.runtime_config(runtime)
    vk = PRIMITIVE_RUNTIME.verified_for(runtime, q)
    if vk is not None:
        if k_cache.dim() != 4:
            raise RuntimeError("configured paged_attention requires rank-4 caches")
        launch_config = PRIMITIVE_RUNTIME.static_launch_config(
            runtime, "paged_attention",
            {"head_dim": int(q.shape[-1])},
        )
        out = PRIMITIVE_RUNTIME.kernel_entrypoint(
            runtime, vk, "paged_attention"
        )(
            q,
            k_cache,
            v_cache,
            cu_seqlens_q,
            cu_seqlens_k,
            int(max_seqlen_q),
            int(max_seqlen_k),
            softmax_scale=cfg["head_dim"] ** -0.5,
            block_table=block_table,
            value_checks=False,
            launch_config=launch_config,
        )
        return out.reshape(out.shape[0], -1).contiguous()

    raise RuntimeError("paged_attention requires a verified kernel binding")
# @kernel-bridge-end vosti_kernels::paged_attention_from_verified_caller


# @kernel-bridge-begin vosti_kernels::select_sample_logits
def _maybe_digest_logits(row: torch.Tensor, index: int) -> None:
    """Differential batch-invariance instrumentation: when
    VOSTI_LOGITS_DIGEST names a file, append the sha256 of this selected
    logits row (exact bf16→f32 bytes) tagged with its batch index.  See
    architecture-specific batch-invariance falsifiers."""
    path = os.environ.get("VOSTI_LOGITS_DIGEST")
    if not path:
        return
    import hashlib
    data = row.detach().to(torch.float32).cpu().numpy().tobytes()
    with open(path, "a") as f:
        f.write(f"{index} {hashlib.sha256(data).hexdigest()}\n")


def select_sample_logits(logits: torch.Tensor, cu_seqlens_q: torch.Tensor,
                          index: int) -> torch.Tensor:
    """Extract the last query-token logit row for batch element `index`."""
    if logits.dim() != 2 or cu_seqlens_q.dim() != 1:
        raise RuntimeError(
            "select_sample_logits requires 2D logits and 1D cu_seqlens_q"
        )
    if index < 0 or index + 1 >= int(cu_seqlens_q.numel()):
        raise RuntimeError("select_sample_logits index is outside cu_seqlens_q")
    start = int(cu_seqlens_q[index].item())
    end = int(cu_seqlens_q[index + 1].item())
    if not (0 <= start < end <= int(logits.shape[0])):
        raise RuntimeError("select_sample_logits requires a nonempty in-bounds segment")

    # Indexing alone returns an aliased view.  Clone the exact semantic row so
    # the Rust external-body wrapper can soundly return a fresh TensorPerm.
    row = logits[end - 1].clone().contiguous()
    _maybe_digest_logits(row, index)
    return row
# @kernel-bridge-end vosti_kernels::select_sample_logits


# @kernel-bridge-begin vosti_kernels::select_rows_for_sampling
def select_rows_for_sampling(
    hidden: torch.Tensor, cu_seqlens_q: torch.Tensor
) -> torch.Tensor:
    """Copy every request's last hidden row before the LM-head projection.

    The verified caller supplies a strictly increasing cumulative layout whose
    final endpoint is ``hidden.shape[0]``.  ``index_select`` performs the exact
    row copy in one device operation and allocates storage independent from the
    full hidden-state tensor.  Keeping this adapter separate makes the exact
    gather semantics explicit in the runtime TCB.
    """
    if hidden.dim() != 2 or cu_seqlens_q.dim() != 1:
        raise RuntimeError(
            "select_rows_for_sampling requires 2D hidden states and 1D cu_seqlens_q"
        )
    indices = (cu_seqlens_q[1:].to(dtype=torch.int64) - 1).to(hidden.device)
    return hidden.index_select(0, indices).contiguous()
# @kernel-bridge-end vosti_kernels::select_rows_for_sampling


# @kernel-bridge-begin vosti_kernels::sample
def sample(logits: torch.Tensor) -> int:
    """Deterministic argmax sample."""
    if logits.numel() == 0:
        return 0
    return int(torch.argmax(logits).item())
# @kernel-bridge-end vosti_kernels::sample


# @kernel-bridge-begin vosti_kernels::sample_tokens_rows
def sample_tokens_rows(rows: torch.Tensor) -> list[int]:
    """Apply the deterministic argmax sampler to every row at once."""
    if rows.dim() != 2:
        raise RuntimeError("sample_tokens_rows requires a 2D logits tensor")
    batch = int(rows.shape[0])
    if batch == 0:
        return []
    if (
        os.environ.get("VOSTI_LOGITS_DIGEST")
        or os.environ.get("VOSTI_KERNELS_LOGITS_OBSERVER")
    ):
        for index in range(batch):
            _maybe_digest_logits(rows[index], index)
    if int(rows.shape[1]) == 0:
        return [0] * batch
    return [int(token) for token in torch.argmax(rows, dim=1).cpu().tolist()]
# @kernel-bridge-end vosti_kernels::sample_tokens_rows
