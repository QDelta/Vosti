import torch
import triton
import triton.language as tl

from triton_kernels.runtime_contracts import require_dense_launch_tensors
from triton_kernels.constants import PAGE_SIZE


CONFIGS = (
    {"BLOCK_M": 16, "BLOCK_N": 64, "num_warps": 2, "num_stages": 2},
    {"BLOCK_M": 16, "BLOCK_N": 64, "num_warps": 4, "num_stages": 2},
    {"BLOCK_M": 32, "BLOCK_N": 64, "num_warps": 4, "num_stages": 2},
    {"BLOCK_M": 64, "BLOCK_N": 64, "num_warps": 4, "num_stages": 2},
    {"BLOCK_M": 16, "BLOCK_N": 128, "num_warps": 4, "num_stages": 2},
    {"BLOCK_M": 32, "BLOCK_N": 128, "num_warps": 4, "num_stages": 2},
    {"BLOCK_M": 64, "BLOCK_N": 128, "num_warps": 4, "num_stages": 2},
)

# Offline performance policy over verified candidates.  Unknown head
# dimensions use CONFIGS[0], the required fallback.
DEFAULT_CONFIG_INDEX = 0
CONFIG_INDEX_BY_HEAD_DIM = {
    128: 1,
    256: 1,
}


def select_config(head_dim: int) -> dict:
    """Return one complete static paged-attention launch config."""

    index = CONFIG_INDEX_BY_HEAD_DIM.get(
        int(head_dim), DEFAULT_CONFIG_INDEX
    )
    candidate = CONFIGS[index]
    if PAGE_SIZE % candidate["BLOCK_N"] != 0:
        raise AssertionError("selected paged-attention tile does not divide a page")
    return {**candidate, "D_HEAD": int(head_dim)}

# @params(
#   scalar(B, int),
#   scalar(max_seqlen_q, int),
#   tensor(q, float, shape(Tq, H, D_HEAD), strides(stride_qt, stride_qh, stride_qd)),
#   tensor(k_cache, float, shape(NUM_PAGES, PAGE_BLOCK_SIZE, Hkv, D_HEAD), strides(stride_kcb, stride_kcs, stride_kch, stride_kcd)),
#   tensor(v_cache, float, shape(NUM_PAGES, PAGE_BLOCK_SIZE, Hkv, D_HEAD), strides(stride_vcb, stride_vcs, stride_vch, stride_vcd)),
#   tensor(block_table, int32, shape(B, MAX_NUM_PAGES), strides(stride_btb, stride_bts)),
#   tensor(o, float, shape(Tq, H, D_HEAD), strides(stride_ot, stride_oh, stride_od)),
#   tensor(lse, float, shape(H, Tq), strides(stride_lseh, stride_lset)),
#   tensor(cu_seqlens_q, int32, shape(add(B, 1))),
#   tensor(cu_seqlens_k, int32, shape(add(B, 1))),
#   scalar(scale_log2, float),
# )
# @grid(B, H, cdiv(max_seqlen_q, BLOCK_M))
# @verif(batch_invariance,
#   same(H, Hkv, D_HEAD, scale_log2),
#   pre(
#     right(B) == 1,
#     left(B) > 0,
#     left(Tq) > 0, right(Tq) > 0,
#     left(NUM_PAGES) > 0, right(NUM_PAGES) > 0,
#     H > 0, Hkv > 0,
#     left(MAX_NUM_PAGES) > 0, right(MAX_NUM_PAGES) > 0,
#     left(max_seqlen_q) > 0, right(max_seqlen_q) > 0,
#     x >= 0, x < left(B),
#     Hkv <= H,
#     H % Hkv == 0,
#     left(cu_seqlens_q)[0] == 0, left(cu_seqlens_k)[0] == 0,
#     right(cu_seqlens_q)[0] == 0, right(cu_seqlens_k)[0] == 0,
#     left(cu_seqlens_q)[left(B)] == left(Tq),
#     left(cu_seqlens_k)[left(B)] == left(Tk),
#     right(cu_seqlens_q)[1] == right(Tq),
#     right(cu_seqlens_k)[1] == right(Tk),
#     forall(i, j, implies(and(i >= 0, i < j, j <= left(B)), and(left(cu_seqlens_q)[i] < left(cu_seqlens_q)[j], left(cu_seqlens_k)[i] < left(cu_seqlens_k)[j]))),
#     right(cu_seqlens_q)[0] < right(cu_seqlens_q)[1],
#     right(cu_seqlens_k)[0] < right(cu_seqlens_k)[1],
#     left(cu_seqlens_q)[x+1] - left(cu_seqlens_q)[x] <= left(max_seqlen_q),
#     right(cu_seqlens_q)[1] - right(cu_seqlens_q)[0] <= right(max_seqlen_q),
#     left(cu_seqlens_q)[x+1] - left(cu_seqlens_q)[x] == right(cu_seqlens_q)[1] - right(cu_seqlens_q)[0],
#     left(cu_seqlens_k)[x+1] - left(cu_seqlens_k)[x] == right(cu_seqlens_k)[1] - right(cu_seqlens_k)[0],
#     cdiv(left(cu_seqlens_k)[x+1] - left(cu_seqlens_k)[x], PAGE_BLOCK_SIZE) <= left(MAX_NUM_PAGES),
#     cdiv(right(cu_seqlens_k)[1] - right(cu_seqlens_k)[0], PAGE_BLOCK_SIZE) <= right(MAX_NUM_PAGES),
#     forall(bi_idx, si_idx, implies(and(bi_idx >= 0, bi_idx < left(B), si_idx >= 0, si_idx < left(MAX_NUM_PAGES)), and(left(block_table)[bi_idx, si_idx] >= 0, left(block_table)[bi_idx, si_idx] < left(NUM_PAGES)))),
#     forall(bi_idx, si_idx, implies(and(bi_idx >= 0, bi_idx < 1, si_idx >= 0, si_idx < right(MAX_NUM_PAGES)), and(right(block_table)[bi_idx, si_idx] >= 0, right(block_table)[bi_idx, si_idx] < right(NUM_PAGES)))),
#     left(q)[left(cu_seqlens_q)[x]:left(cu_seqlens_q)[x+1], 0:H, 0:D_HEAD] == right(q)[right(cu_seqlens_q)[0]:right(cu_seqlens_q)[1], 0:H, 0:D_HEAD],
#     forall(cache_page, implies(and(cache_page >= 0, cache_page < cdiv(left(cu_seqlens_k)[x+1] - left(cu_seqlens_k)[x], PAGE_BLOCK_SIZE)), left(k_cache)[left(block_table)[x, cache_page]:left(block_table)[x, cache_page]+1, 0:PAGE_BLOCK_SIZE, 0:Hkv, 0:D_HEAD] == right(k_cache)[right(block_table)[0, cache_page]:right(block_table)[0, cache_page]+1, 0:PAGE_BLOCK_SIZE, 0:Hkv, 0:D_HEAD])),
#     forall(cache_page, implies(and(cache_page >= 0, cache_page < cdiv(left(cu_seqlens_k)[x+1] - left(cu_seqlens_k)[x], PAGE_BLOCK_SIZE)), left(v_cache)[left(block_table)[x, cache_page]:left(block_table)[x, cache_page]+1, 0:PAGE_BLOCK_SIZE, 0:Hkv, 0:D_HEAD] == right(v_cache)[right(block_table)[0, cache_page]:right(block_table)[0, cache_page]+1, 0:PAGE_BLOCK_SIZE, 0:Hkv, 0:D_HEAD])),
#   ),
#   post(
#     left(o)[left(cu_seqlens_q)[x]:left(cu_seqlens_q)[x+1], 0:H, 0:D_HEAD] == right(o)[right(cu_seqlens_q)[0]:right(cu_seqlens_q)[1], 0:H, 0:D_HEAD],
#     left(lse)[0:H, left(cu_seqlens_q)[x]:left(cu_seqlens_q)[x+1]] == right(lse)[0:H, right(cu_seqlens_q)[0]:right(cu_seqlens_q)[1]]
#   ),
#   singleton(bi left=x right=0),
# )
# @verif(selected_row_prefix_equivalence,
#   same(H, Hkv, D_HEAD, scale_log2),
#   pre(
#     left(B) == 1, right(B) == 1,
#     left(Tq) > 0, right(Tq) > 0,
#     left(Tk) > 0, right(Tk) > 0,
#     left(Tq) <= left(Tk), right(Tq) <= right(Tk),
#     left(Tq) <= left(max_seqlen_q),
#     right(Tq) <= right(max_seqlen_q),
#     H > 0, Hkv > 0,
#     left(NUM_PAGES) > 0, right(NUM_PAGES) > 0,
#     left(MAX_NUM_PAGES) > 0, right(MAX_NUM_PAGES) > 0,
#     cdiv(left(Tk), PAGE_BLOCK_SIZE) <= left(MAX_NUM_PAGES),
#     cdiv(right(Tk), PAGE_BLOCK_SIZE) <= right(MAX_NUM_PAGES),
#     forall(si_idx, implies(and(si_idx >= 0, si_idx < cdiv(left(Tk), PAGE_BLOCK_SIZE)), and(left(block_table)[0, si_idx] >= 0, left(block_table)[0, si_idx] < left(NUM_PAGES)))),
#     forall(si_idx, implies(and(si_idx >= 0, si_idx < cdiv(right(Tk), PAGE_BLOCK_SIZE)), and(right(block_table)[0, si_idx] >= 0, right(block_table)[0, si_idx] < right(NUM_PAGES)))),
#     selected_left_row >= 0, selected_left_row < left(Tq),
#     selected_right_row >= 0, selected_right_row < right(Tq),
#     left(Tk) - left(Tq) + selected_left_row == right(Tk) - right(Tq) + selected_right_row,
#     left(cu_seqlens_q)[0] == 0, right(cu_seqlens_q)[0] == 0,
#     left(cu_seqlens_k)[0] == 0, right(cu_seqlens_k)[0] == 0,
#     left(cu_seqlens_q)[1] == left(Tq), right(cu_seqlens_q)[1] == right(Tq),
#     left(cu_seqlens_k)[1] == left(Tk), right(cu_seqlens_k)[1] == right(Tk),
#     left(q)[selected_left_row:selected_left_row+1, 0:H, 0:D_HEAD] == right(q)[selected_right_row:selected_right_row+1, 0:H, 0:D_HEAD],
#     forall(prefix_tile, implies(and(prefix_tile >= 0, prefix_tile < cdiv(left(Tk) - left(Tq) + selected_left_row + 1, BLOCK_N)), left(k_cache)[left(block_table)[0, prefix_tile * BLOCK_N // PAGE_BLOCK_SIZE]:left(block_table)[0, prefix_tile * BLOCK_N // PAGE_BLOCK_SIZE]+1, 0:min(PAGE_BLOCK_SIZE, left(Tk) - left(Tq) + selected_left_row + 1 - (prefix_tile * BLOCK_N // PAGE_BLOCK_SIZE) * PAGE_BLOCK_SIZE), 0:Hkv, 0:D_HEAD] == right(k_cache)[right(block_table)[0, prefix_tile * BLOCK_N // PAGE_BLOCK_SIZE]:right(block_table)[0, prefix_tile * BLOCK_N // PAGE_BLOCK_SIZE]+1, 0:min(PAGE_BLOCK_SIZE, right(Tk) - right(Tq) + selected_right_row + 1 - (prefix_tile * BLOCK_N // PAGE_BLOCK_SIZE) * PAGE_BLOCK_SIZE), 0:Hkv, 0:D_HEAD])),
#     forall(prefix_tile, implies(and(prefix_tile >= 0, prefix_tile < cdiv(left(Tk) - left(Tq) + selected_left_row + 1, BLOCK_N)), left(v_cache)[left(block_table)[0, prefix_tile * BLOCK_N // PAGE_BLOCK_SIZE]:left(block_table)[0, prefix_tile * BLOCK_N // PAGE_BLOCK_SIZE]+1, 0:min(PAGE_BLOCK_SIZE, left(Tk) - left(Tq) + selected_left_row + 1 - (prefix_tile * BLOCK_N // PAGE_BLOCK_SIZE) * PAGE_BLOCK_SIZE), 0:Hkv, 0:D_HEAD] == right(v_cache)[right(block_table)[0, prefix_tile * BLOCK_N // PAGE_BLOCK_SIZE]:right(block_table)[0, prefix_tile * BLOCK_N // PAGE_BLOCK_SIZE]+1, 0:min(PAGE_BLOCK_SIZE, right(Tk) - right(Tq) + selected_right_row + 1 - (prefix_tile * BLOCK_N // PAGE_BLOCK_SIZE) * PAGE_BLOCK_SIZE), 0:Hkv, 0:D_HEAD])),
#   ),
#   post(
#     left(o)[selected_left_row:selected_left_row+1, 0:H, 0:D_HEAD] == right(o)[selected_right_row:selected_right_row+1, 0:H, 0:D_HEAD]
#   ),
#   singleton(bi left=0 right=0),
# )
# Serving supplies a complete config selected and qualified offline.
@triton.jit
def fattn_varlen_paged_fwd_block_ptr_kernel(
    q,  # [total_q, H, D]
    k_cache,  # [num_blocks, page_block_size, Hkv, D]
    v_cache,  # [num_blocks, page_block_size, Hkv, D]
    block_table,  # [B, max_num_blocks_per_seq]
    cu_seqlens_q,  # [B + 1]
    cu_seqlens_k,  # [B + 1]
    o,  # [total_q, H, D]
    lse,  # [H, total_q]
    H,
    Hkv,
    stride_qt,
    stride_qh,
    stride_qd,
    stride_kcb,
    stride_kcs,
    stride_kch,
    stride_kcd,
    stride_vcb,
    stride_vcs,
    stride_vch,
    stride_vcd,
    stride_btb,
    stride_bts,
    stride_ot,
    stride_oh,
    stride_od,
    stride_lseh,
    stride_lset,
    scale_log2,
    D_HEAD: tl.constexpr,
    PAGE_BLOCK_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    bi = tl.program_id(axis=0)
    hi = tl.program_id(axis=1)
    qi = tl.program_id(axis=2) * BLOCK_M

    q_start = tl.load(cu_seqlens_q + bi)
    q_end = tl.load(cu_seqlens_q + bi + 1)
    k_start = tl.load(cu_seqlens_k + bi)
    k_end = tl.load(cu_seqlens_k + bi + 1)

    q_len = q_end - q_start
    k_len = k_end - k_start
    if qi >= q_len:
        return

    hkvi = hi * Hkv // H
    q_shift = k_len - q_len
    q_block_end = min(q_len, qi + BLOCK_M)
    causal_k_end = max(0, min(k_len, q_shift + q_block_end))

    q_indices = qi + tl.arange(0, BLOCK_M)
    q_mask = q_indices < q_len
    q_block_ptr = tl.make_block_ptr(
        base=q + q_start * stride_qt + hi * stride_qh,
        shape=(q_len, D_HEAD),
        strides=(stride_qt, stride_qd),
        offsets=(qi, 0),
        block_shape=(BLOCK_M, D_HEAD),
        order=(0, 1),
    )
    q_block = tl.load(q_block_ptr, boundary_check=(0,), padding_option="zero")

    acc = tl.zeros((BLOCK_M, D_HEAD), dtype=tl.float32)
    logsum = tl.zeros((BLOCK_M,), dtype=tl.float32)
    scores_max = tl.full((BLOCK_M,), float("-inf"), dtype=tl.float32)

    # Model-static head geometry only; never query length or live batch size.
    for ki in tl.range(tl.cdiv(causal_k_end, BLOCK_N), loop_unroll_factor=2 if D_HEAD == 128 else 1):
        k_tile_start = ki * BLOCK_N
        k_indices = k_tile_start + tl.arange(0, BLOCK_N)
        k_mask = k_indices < k_len

        page_slot = k_tile_start // PAGE_BLOCK_SIZE
        page_offset = k_tile_start % PAGE_BLOCK_SIZE
        page_id = tl.load(block_table + bi * stride_btb + page_slot * stride_bts)
        page_len = tl.minimum(PAGE_BLOCK_SIZE, k_len - page_slot * PAGE_BLOCK_SIZE)

        k_block_ptr = tl.make_block_ptr(
            base=k_cache + page_id * stride_kcb + hkvi * stride_kch,
            shape=(page_len, D_HEAD),
            strides=(stride_kcs, stride_kcd),
            offsets=(page_offset, 0),
            block_shape=(BLOCK_N, D_HEAD),
            order=(0, 1),
        )
        k_block = tl.load(k_block_ptr, boundary_check=(0,), padding_option="zero")

        qk = tl.dot(q_block, tl.trans(k_block))

        attn_mask = q_mask[:, None] & k_mask[None, :]
        attn_mask &= k_indices[None, :] <= (q_indices[:, None] + q_shift)
        # Make an all-false query row an explicit state no-op.  Relying on
        # algebraic cancellation is not IEEE-safe: 0*NaN and inf-inf can make
        # a causally invisible tile change the online-softmax state. Triton
        # promotes the bool max-reduction to int32.
        row_has_any = tl.max(attn_mask, axis=1) > 0

        scores = qk * scale_log2
        masked_scores = tl.where(attn_mask, scores, float("-inf"))
        next_max = tl.maximum(scores_max, tl.max(masked_scores, axis=1))
        alpha = tl.exp2(scores_max - next_max)
        p = tl.where(attn_mask, tl.exp2(scores - next_max[:, None]), 0.0)

        next_acc = acc * alpha[:, None]
        next_logsum = logsum * alpha + tl.sum(p, axis=1)

        v_block_ptr = tl.make_block_ptr(
            base=v_cache + page_id * stride_vcb + hkvi * stride_vch,
            shape=(page_len, D_HEAD),
            strides=(stride_vcs, stride_vcd),
            offsets=(page_offset, 0),
            block_shape=(BLOCK_N, D_HEAD),
            order=(0, 1),
        )
        v_block = tl.load(v_block_ptr, boundary_check=(0,), padding_option="zero")
        next_acc = tl.dot(p.to(v_cache.dtype.element_ty), v_block, next_acc)

        acc = tl.where(row_has_any[:, None], next_acc, acc)
        logsum = tl.where(row_has_any, next_logsum, logsum)
        scores_max = tl.where(row_has_any, next_max, scores_max)

    out = acc * tl.where(logsum > 0.0, 1.0 / logsum, 0.0)[:, None]

    o_block_ptr = tl.make_block_ptr(
        base=o + q_start * stride_ot + hi * stride_oh,
        shape=(q_len, D_HEAD),
        strides=(stride_ot, stride_od),
        offsets=(qi, 0),
        block_shape=(BLOCK_M, D_HEAD),
        order=(0, 1),
    )
    tl.store(o_block_ptr, out.to(o.dtype.element_ty), boundary_check=(0,))

    lse_vals = tl.where(
        logsum > 0.0,
        (tl.log2(logsum) + scores_max) * 0.6931471805599453,
        float("-inf"),
    )
    lse_block_ptr = tl.make_block_ptr(
        base=lse + hi * stride_lseh + q_start * stride_lset,
        shape=(q_len,),
        strides=(stride_lset,),
        offsets=(qi,),
        block_shape=(BLOCK_M,),
        order=(0,),
    )
    tl.store(lse_block_ptr, lse_vals, boundary_check=(0,))




# @kernel-bridge-begin fattn_paged::physical_launch_adapter
def _maybe_contiguous_last_dim(x: torch.Tensor) -> torch.Tensor:
    return x.contiguous() if x.stride(-1) != 1 else x


def _prepare_inputs(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    dropout_p: float,
    softmax_scale: float | None,
    softcap: float,
    alibi_slopes: torch.Tensor | None,
    block_table: torch.Tensor | None,
    value_checks: bool = True,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    int,
    int,
    int,
    int,
    int,
    float,
]:
    if block_table is None:
        raise ValueError("block_table is required")
    if dropout_p != 0.0:
        raise NotImplementedError("dropout is not supported")
    if softcap != 0.0:
        raise NotImplementedError("softcap is not supported")
    if alibi_slopes is not None:
        raise NotImplementedError("alibi slopes are not supported")

    if q.ndim != 3:
        raise ValueError("q must have shape [total_q, nheads, headdim]")
    if k_cache.ndim != 4 or v_cache.ndim != 4:
        raise ValueError(
            "k_cache and v_cache must have shape [num_blocks, page_block_size, nheads_k, headdim]"
        )
    if cu_seqlens_q.ndim != 1 or cu_seqlens_k.ndim != 1:
        raise ValueError("cu_seqlens_q and cu_seqlens_k must be rank-1")
    if block_table.ndim != 2:
        raise ValueError(
            "block_table must have shape [batch_size, max_num_blocks_per_seq]"
        )
    if not (q.is_cuda and k_cache.is_cuda and v_cache.is_cuda):
        raise ValueError("q, k_cache, and v_cache must be CUDA tensors")

    total_q, H, D = q.shape
    _, page_block_size, Hkv, Dk = k_cache.shape
    _, _, Hkv_v, Dv = v_cache.shape
    if q.dtype != k_cache.dtype or q.dtype != v_cache.dtype:
        raise ValueError("q, k_cache, and v_cache must have the same dtype")
    if D != Dk or D != Dv:
        raise ValueError("q, k_cache, and v_cache must share the same headdim")
    if Hkv != Hkv_v:
        raise ValueError("k_cache and v_cache must have the same number of KV heads")
    if H % Hkv != 0:
        raise ValueError(
            "the number of query heads must be divisible by the number of KV heads"
        )

    q = _maybe_contiguous_last_dim(q)
    k_cache = _maybe_contiguous_last_dim(k_cache)
    v_cache = _maybe_contiguous_last_dim(v_cache)
    cu_seqlens_q = cu_seqlens_q.to(device=q.device, dtype=torch.int32).contiguous()
    cu_seqlens_k = cu_seqlens_k.to(device=q.device, dtype=torch.int32).contiguous()
    block_table = block_table.to(device=q.device, dtype=torch.int32).contiguous()

    if cu_seqlens_q.numel() != cu_seqlens_k.numel():
        raise ValueError("cu_seqlens_q and cu_seqlens_k must have the same length")
    if cu_seqlens_q.numel() < 2:
        raise ValueError("empty batch")
    B = cu_seqlens_q.numel() - 1
    if block_table.shape[0] != B:
        raise ValueError("block_table batch dimension must match cu_seqlens")

    if value_checks:
        # These reads synchronize CUDA with the host.  The verified engine
        # disables them only after proving the endpoints and per-row maxima
        # through paged_attention_launch_ready; unverified callers retain the
        # checked default.
        if int(cu_seqlens_q[0].item()) != 0 or int(cu_seqlens_k[0].item()) != 0:
            raise ValueError("cu_seqlens must start at 0")
        if int(cu_seqlens_q[-1].item()) != total_q:
            raise ValueError("cu_seqlens_q[-1] must equal q.shape[0]")
        q_lens = cu_seqlens_q[1:] - cu_seqlens_q[:-1]
        k_lens = cu_seqlens_k[1:] - cu_seqlens_k[:-1]
        if int(q_lens.max().item()) > max_seqlen_q:
            raise ValueError("max_seqlen_q is smaller than the longest query sequence")
        if int(k_lens.max().item()) > max_seqlen_k:
            raise ValueError("max_seqlen_k is smaller than the longest KV sequence")
    if max_seqlen_k > block_table.shape[1] * page_block_size:
        raise ValueError("block_table does not cover max_seqlen_k")

    if softmax_scale is None:
        softmax_scale = D ** (-0.5)

    return (
        q,
        k_cache,
        v_cache,
        cu_seqlens_q,
        cu_seqlens_k,
        block_table,
        B,
        H,
        Hkv,
        D,
        total_q,
        softmax_scale,
    )


def _run_kernel(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    dropout_p: float = 0.0,
    softmax_scale: float | None = None,
    softcap: float = 0.0,
    alibi_slopes: torch.Tensor | None = None,
    deterministic: bool = False,
    return_attn_probs: bool = False,
    block_table: torch.Tensor | None = None,
    value_checks: bool = True,
    launch_config: dict | None = None,
):
    del deterministic
    (
        q,
        k_cache,
        v_cache,
        cu_seqlens_q,
        cu_seqlens_k,
        block_table,
        B,
        H,
        Hkv,
        D,
        total_q,
        softmax_scale,
    ) = _prepare_inputs(
        q,
        k_cache,
        v_cache,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        dropout_p,
        softmax_scale,
        softcap,
        alibi_slopes,
        block_table,
        value_checks=value_checks,
    )

    # Both writable tensors are fresh, distinct, and row-major.  The logical
    # disjointness certificate therefore describes distinct physical cells.
    o = torch.empty_like(q, memory_format=torch.contiguous_format)
    softmax_lse = torch.empty((H, total_q), dtype=torch.float32, device=q.device)
    require_dense_launch_tensors(
        readonly={
            "q": q,
            "k_cache": k_cache,
            "v_cache": v_cache,
            "block_table": block_table,
            "cu_seqlens_q": cu_seqlens_q,
            "cu_seqlens_k": cu_seqlens_k,
        },
        writable={"o": o, "lse": softmax_lse},
    )
    if total_q == 0 or max_seqlen_q == 0:
        if return_attn_probs:
            return o, softmax_lse, torch.empty(0, dtype=q.dtype, device=q.device)
        return o

    if launch_config is None:
        raise ValueError("paged attention requires a sealed launch config")
    cfg = launch_config
    if (
        cfg["D_HEAD"] != D
        or PAGE_SIZE != k_cache.shape[1]
    ):
        raise ValueError("paged attention inputs differ from the sealed config")
    grid = (B, H, triton.cdiv(max_seqlen_q, cfg["BLOCK_M"]))

    LOG2_E = 1.4426950408889634
    softmax_scale *= LOG2_E

    fattn_varlen_paged_fwd_block_ptr_kernel[grid](
        q,
        k_cache,
        v_cache,
        block_table,
        cu_seqlens_q,
        cu_seqlens_k,
        o,
        softmax_lse,
        H,
        Hkv,
        *q.stride(),
        *k_cache.stride(),
        *v_cache.stride(),
        *block_table.stride(),
        *o.stride(),
        *softmax_lse.stride(),
        scale_log2=softmax_scale,
        D_HEAD=cfg["D_HEAD"],  # pyright: ignore[reportArgumentType]
        PAGE_BLOCK_SIZE=PAGE_SIZE,
        BLOCK_M=cfg["BLOCK_M"],  # pyright: ignore[reportArgumentType]
        BLOCK_N=cfg["BLOCK_N"],  # pyright: ignore[reportArgumentType]
        num_warps=cfg["num_warps"],
        num_stages=cfg["num_stages"],
    )

    if return_attn_probs:
        return o, softmax_lse, torch.empty(0, dtype=q.dtype, device=q.device)
    return o


def fattn_varlen_paged_fwd_block_ptr(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    dropout_p: float = 0.0,
    softmax_scale: float | None = None,
    softcap: float = 0.0,
    alibi_slopes: torch.Tensor | None = None,
    deterministic: bool = False,
    return_attn_probs: bool = False,
    block_table: torch.Tensor | None = None,
    value_checks: bool = True,
    launch_config: dict | None = None,
):
    return _run_kernel(
        q,
        k_cache,
        v_cache,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        dropout_p=dropout_p,
        softmax_scale=softmax_scale,
        softcap=softcap,
        alibi_slopes=alibi_slopes,
        deterministic=deterministic,
        return_attn_probs=return_attn_probs,
        block_table=block_table,
        value_checks=value_checks,
        launch_config=launch_config,
    )
# @kernel-bridge-end fattn_paged::physical_launch_adapter


def _build_cu_seqlens(lengths: torch.Tensor) -> torch.Tensor:
    return torch.cat(
        [
            torch.zeros(1, dtype=torch.int32, device=lengths.device),
            lengths.to(dtype=torch.int32).cumsum(dim=0, dtype=torch.int32),
        ]
    )


def _pack_paged_kv(
    k_packed: torch.Tensor,
    v_packed: torch.Tensor,
    lengths: torch.Tensor,
    page_block_size: int,
    extra_blocks: int = 3,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    device = k_packed.device
    batch = lengths.numel()
    _, Hkv, D = k_packed.shape
    blocks_per_seq = (
        lengths.to(dtype=torch.int32) + page_block_size - 1
    ) // page_block_size
    used_blocks = int(blocks_per_seq.sum().item())
    num_blocks = used_blocks + extra_blocks
    max_blocks = int(blocks_per_seq.max().item())

    block_table = torch.randint(
        0,
        num_blocks,
        (batch, max_blocks),
        dtype=torch.int32,
        device=device,
    )
    k_cache = torch.randn(
        (num_blocks, page_block_size, Hkv, D), dtype=k_packed.dtype, device=device
    )
    v_cache = torch.randn(
        (num_blocks, page_block_size, Hkv, D), dtype=v_packed.dtype, device=device
    )

    page_ids = torch.randperm(num_blocks, device=device, dtype=torch.int64)[
        :used_blocks
    ]
    cu_k = _build_cu_seqlens(lengths)
    page_cursor = 0
    for seq_idx, num_blocks_seq in enumerate(blocks_per_seq.tolist()):
        seq_start = int(cu_k[seq_idx].item())
        seq_len = int(lengths[seq_idx].item())
        for logical_block in range(num_blocks_seq):
            page_id = int(page_ids[page_cursor].item())
            page_cursor += 1
            block_table[seq_idx, logical_block] = page_id
            block_start = seq_start + logical_block * page_block_size
            block_len = min(page_block_size, seq_len - logical_block * page_block_size)
            k_cache[page_id, :block_len] = k_packed[
                block_start : block_start + block_len
            ]
            v_cache[page_id, :block_len] = v_packed[
                block_start : block_start + block_len
            ]

    return k_cache, v_cache, block_table


def _poison_unused_inputs(
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_table: torch.Tensor,
    lengths: torch.Tensor,
    page_block_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    k_mut = k_cache.clone()
    v_mut = v_cache.clone()
    block_table_mut = block_table.clone()
    num_blocks = k_cache.shape[0]
    used_pages = torch.zeros(num_blocks, dtype=torch.bool, device=k_cache.device)
    blocks_per_seq = (
        lengths.to(dtype=torch.int32) + page_block_size - 1
    ) // page_block_size

    for seq_idx, num_blocks_seq in enumerate(blocks_per_seq.tolist()):
        if num_blocks_seq == 0:
            continue
        page_ids = block_table[seq_idx, :num_blocks_seq].to(dtype=torch.int64)
        used_pages[page_ids] = True

        seq_len = int(lengths[seq_idx].item())
        tail = seq_len % page_block_size
        if tail != 0:
            page_id = int(page_ids[-1].item())
            k_mut[page_id, tail:] = (
                torch.randn_like(k_mut[page_id, tail:]) * 37.0 + 111.0
            )
            v_mut[page_id, tail:] = (
                torch.randn_like(v_mut[page_id, tail:]) * 41.0 - 73.0
            )

        if num_blocks_seq < block_table.shape[1]:
            block_table_mut[seq_idx, num_blocks_seq:] = torch.randint(
                0,
                num_blocks,
                (block_table.shape[1] - num_blocks_seq,),
                dtype=torch.int32,
                device=block_table.device,
            )

    unused_pages = ~used_pages
    if unused_pages.any():
        k_mut[unused_pages] = torch.randn_like(k_mut[unused_pages]) * 53.0 + 19.0
        v_mut[unused_pages] = torch.randn_like(v_mut[unused_pages]) * 47.0 - 29.0

    return k_mut, v_mut, block_table_mut


def _valid_rows(
    q_lens: torch.Tensor,
    k_lens: torch.Tensor,
    *,
    device: torch.device,
) -> torch.Tensor:
    masks = []
    for q_len_t, k_len_t in zip(q_lens.tolist(), k_lens.tolist()):
        q_len = int(q_len_t)
        k_len = int(k_len_t)
        if k_len == 0:
            masks.append(torch.zeros(q_len, dtype=torch.bool, device=device))
            continue
        q_idx = torch.arange(q_len, dtype=torch.int32, device=device)
        masks.append(q_idx + (k_len - q_len) >= 0)
    return torch.cat(masks, dim=0)


def _max_abs_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    both_neginf = torch.isneginf(a) & torch.isneginf(b)
    diff = (a - b).abs()
    diff = torch.where(both_neginf, torch.zeros_like(diff), diff)
    diff = torch.nan_to_num(
        diff, nan=float("inf"), posinf=float("inf"), neginf=float("inf")
    )
    return diff.max().item()


def _max_abs_diff_masked(a: torch.Tensor, b: torch.Tensor, mask: torch.Tensor) -> float:
    if not mask.any():
        return 0.0
    return _max_abs_diff(a[mask], b[mask])


def _check_invalid_rows(
    out: torch.Tensor, lse: torch.Tensor, invalid_rows: torch.Tensor
) -> tuple[int, float, bool]:
    if not invalid_rows.any():
        return 0, 0.0, True
    return (
        int(invalid_rows.sum().item()),
        out[invalid_rows].abs().max().item(),
        torch.isneginf(lse[:, invalid_rows]).all().item(),
    )


def _sdpa_reference(
    q, k_cache, v_cache, cu_q, cu_k, max_seqlen_q, max_seqlen_k,
    *, block_table, return_attn_probs=False, softmax_scale=None,
):
    """Test-only paged oracle: gather logical KV, then use Torch's math SDPA.

    Unequal Q/K lengths require bottom-right causal alignment. SDPA's plain
    ``is_causal=True`` uses top-left alignment, so supply the explicit mask.
    The optional logsumexp reference is computed separately because SDPA's
    public interface returns only the output. No serving path calls this.
    """
    del max_seqlen_q, max_seqlen_k
    from torch.nn.attention import SDPBackend, sdpa_kernel

    scale = q.shape[-1] ** -0.5 if softmax_scale is None else softmax_scale
    q_offsets, k_offsets = cu_q.tolist(), cu_k.tolist()
    outputs, normalizers = [], []
    for batch, (start, end) in enumerate(zip(q_offsets, q_offsets[1:])):
        q_len = end - start
        k_len = k_offsets[batch + 1] - k_offsets[batch]
        qi = q[start:end].transpose(0, 1).float()
        if k_len == 0:
            outputs.append(torch.zeros_like(q[start:end]))
            if return_attn_probs:
                normalizers.append(torch.full(qi.shape[:2], -torch.inf, device=q.device))
            continue
        positions = torch.arange(k_len, device=q.device)
        pages = block_table[batch, positions // k_cache.shape[1]].long()
        offsets = positions % k_cache.shape[1]
        ki = k_cache[pages, offsets].transpose(0, 1).float()
        vi = v_cache[pages, offsets].transpose(0, 1).float()
        repeats = qi.shape[0] // ki.shape[0]
        ki = ki.repeat_interleave(repeats, dim=0)
        vi = vi.repeat_interleave(repeats, dim=0)
        query_positions = torch.arange(q_len, device=q.device) + k_len - q_len
        visible = positions[None, :] <= query_positions[:, None]
        with sdpa_kernel(SDPBackend.MATH):
            output = torch.nn.functional.scaled_dot_product_attention(
                qi, ki, vi, attn_mask=visible, dropout_p=0.0, scale=scale,
            )
        outputs.append(output.transpose(0, 1).to(q.dtype))
        if return_attn_probs:
            scores = (qi @ ki.transpose(-1, -2)) * scale
            normalizers.append(torch.logsumexp(scores.masked_fill(~visible, -torch.inf), -1))
    output = torch.cat(outputs, dim=0)
    if return_attn_probs:
        return output, torch.cat(normalizers, dim=1), torch.empty(0, device=q.device)
    return output


def _run_single_case(
    name: str,
    attention_ref,
    *,
    seed: int,
    dtype: torch.dtype,
    H: int,
    Hkv: int,
    D: int,
    page_block_size: int,
    q_lens_list: list[int],
    k_lens_list: list[int],
) -> dict[str, float | int | str]:
    torch.manual_seed(seed)
    device = "cuda"
    q_lens = torch.tensor(q_lens_list, dtype=torch.int32, device=device)
    k_lens = torch.tensor(k_lens_list, dtype=torch.int32, device=device)
    cu_q = _build_cu_seqlens(q_lens)
    cu_k = _build_cu_seqlens(k_lens)

    q = torch.randn((int(cu_q[-1].item()), H, D), device=device, dtype=dtype)
    k_packed = torch.randn((int(cu_k[-1].item()), Hkv, D), device=device, dtype=dtype)
    v_packed = torch.randn((int(cu_k[-1].item()), Hkv, D), device=device, dtype=dtype)
    k_cache, v_cache, block_table = _pack_paged_kv(
        k_packed, v_packed, k_lens, page_block_size
    )

    max_seqlen_q = int(q_lens.max().item())
    max_seqlen_k = int(k_lens.max().item())
    valid_rows = _valid_rows(q_lens, k_lens, device=q.device)
    valid_lse_mask = valid_rows[None, :].expand(H, valid_rows.numel())
    invalid_rows = ~valid_rows
    if page_block_size != PAGE_SIZE:
        raise ValueError(f"test page size must be compiled PAGE_SIZE={PAGE_SIZE}")
    launch_config = select_config(D)

    out_triton, lse_triton, _ = fattn_varlen_paged_fwd_block_ptr(
        q,
        k_cache,
        v_cache,
        cu_q,
        cu_k,
        max_seqlen_q,
        max_seqlen_k,
        return_attn_probs=True,
        block_table=block_table,
        launch_config=launch_config,
    )
    out_ref, lse_ref, _ = attention_ref(
        q,
        k_cache,
        v_cache,
        cu_q,
        cu_k,
        max_seqlen_q,
        max_seqlen_k,
        return_attn_probs=True,
        block_table=block_table,
    )

    reference_out_diff = _max_abs_diff(out_triton, out_ref)
    reference_lse_diff = _max_abs_diff_masked(lse_triton, lse_ref, valid_lse_mask)

    k_cache_mut, v_cache_mut, block_table_mut = _poison_unused_inputs(
        k_cache,
        v_cache,
        block_table,
        k_lens,
        page_block_size,
    )
    out_triton_mut, lse_triton_mut, _ = fattn_varlen_paged_fwd_block_ptr(
        q,
        k_cache_mut,
        v_cache_mut,
        cu_q,
        cu_k,
        max_seqlen_q,
        max_seqlen_k,
        return_attn_probs=True,
        block_table=block_table_mut,
        launch_config=launch_config,
    )
    out_ref_mut, lse_ref_mut, _ = attention_ref(
        q,
        k_cache_mut,
        v_cache_mut,
        cu_q,
        cu_k,
        max_seqlen_q,
        max_seqlen_k,
        return_attn_probs=True,
        block_table=block_table_mut,
    )

    triton_invariance = _max_abs_diff(out_triton_mut, out_triton)
    reference_invariance = _max_abs_diff(out_ref_mut, out_ref)
    triton_lse_invariance = _max_abs_diff_masked(
        lse_triton_mut, lse_triton, valid_lse_mask
    )
    reference_lse_invariance = _max_abs_diff_masked(
        lse_ref_mut, lse_ref, valid_lse_mask
    )

    invalid_count, invalid_abs, invalid_lse = _check_invalid_rows(
        out_triton, lse_triton, invalid_rows
    )

    return {
        "name": name,
        "seed": seed,
        "reference_out_diff": reference_out_diff,
        "reference_lse_diff": reference_lse_diff,
        "triton_invariance": triton_invariance,
        "reference_invariance": reference_invariance,
        "triton_lse_invariance": triton_lse_invariance,
        "reference_lse_invariance": reference_lse_invariance,
        "invalid_rows": invalid_count,
        "invalid_abs": invalid_abs,
        "invalid_lse": int(invalid_lse),
    }


def compare_kernels() -> None:
    if not torch.cuda.is_available():
        print("CUDA is not available; skipping comparison.")
        return

    cases = [
        {
            "name": "causal_page_boundary",
            "dtype": torch.bfloat16,
            "H": 8,
            "Hkv": 2,
            "D": 64,
            "page_block_size": PAGE_SIZE,
            "q_lens_list": [63, 64, 65, 129],
            "k_lens_list": [255, 256, 257, 321],
        },
        {
            "name": "causal_zero_rows",
            "dtype": torch.bfloat16,
            "H": 6,
            "Hkv": 2,
            "D": 64,
            "page_block_size": PAGE_SIZE,
            "q_lens_list": [257, 65, 33, 9],
            "k_lens_list": [1, 64, 32, 1],
        },
        {
            "name": "causal_long_prefix",
            "dtype": torch.bfloat16,
            "H": 8,
            "Hkv": 2,
            "D": 64,
            "page_block_size": PAGE_SIZE,
            "q_lens_list": [65, 127, 129, 257],
            "k_lens_list": [511, 512, 513, 700],
        },
    ]

    reference_out_tol = 1e-2
    reference_lse_tol = 1e-5
    results: list[dict[str, float | int | str]] = []

    for seed in (0, 1):
        for case in cases:
            result = _run_single_case(
                case["name"],
                _sdpa_reference,
                seed=seed,
                dtype=case["dtype"],
                H=case["H"],
                Hkv=case["Hkv"],
                D=case["D"],
                page_block_size=case["page_block_size"],
                q_lens_list=case["q_lens_list"],
                k_lens_list=case["k_lens_list"],
            )
            print(
                f"{result['name']} seed={result['seed']} "
                f"sdpa_diff={result['reference_out_diff']:.6f}/{result['reference_lse_diff']:.6f} "
                f"inv_triton={result['triton_invariance']:.6f}/{result['triton_lse_invariance']:.6f} "
                f"inv_reference={result['reference_invariance']:.6f}/{result['reference_lse_invariance']:.6f} "
                f"invalid={result['invalid_rows']} "
                f"invalid_abs={result['invalid_abs']:.6f} "
                f"invalid_lse={result['invalid_lse']}"
            )

            if result["reference_out_diff"] > reference_out_tol:
                raise AssertionError(
                    f"SDPA output mismatch in case {result['name']} seed={seed}"
                )
            if result["reference_lse_diff"] > reference_lse_tol:
                raise AssertionError(
                    f"reference lse mismatch in case {result['name']} seed={seed}"
                )
            if result["triton_invariance"] != 0.0:
                raise AssertionError(
                    f"masked KV data affected Triton kernel in case {result['name']} seed={seed}"
                )
            if result["reference_invariance"] != 0.0:
                raise AssertionError(
                    f"masked KV data affected SDPA output in case {result['name']} seed={seed}"
                )
            if (
                result["triton_lse_invariance"] != 0.0
                or result["reference_lse_invariance"] != 0.0
            ):
                raise AssertionError(
                    f"masked KV data affected lse on valid rows in case {result['name']} seed={seed}"
                )
            if result["invalid_rows"] > 0:
                if result["invalid_abs"] != 0.0:
                    raise AssertionError(
                        f"fully masked rows produced nonzero output in case {result['name']} seed={seed}"
                    )
                if result["invalid_lse"] != 1:
                    raise AssertionError(
                        f"fully masked rows did not get -inf lse in case {result['name']} seed={seed}"
                    )
            results.append(result)

    max_reference_out = max(
        float(result["reference_out_diff"]) for result in results
    )
    max_reference_lse = max(
        float(result["reference_lse_diff"]) for result in results
    )
    print(
        f"checked {len(results)} cases; "
        f"max_sdpa_out_diff={max_reference_out:.6f}, "
        f"max_reference_lse_diff={max_reference_lse:.6f}"
    )


if __name__ == "__main__":
    compare_kernels()
