import torch
import triton
import triton.language as tl

from triton_kernels.fattn_paged import CONFIGS, PAGE_SIZE, _prepare_inputs
from triton_kernels.runtime_contracts import require_dense_launch_tensors


# Sliding attention shares the tile catalog, not the full-attention tuning
# policy: its bounded key traversal has a different performance tradeoff.
DEFAULT_CONFIG_INDEX = 0
CONFIG_INDEX_BY_HEAD_DIM = {256: 1}


def select_config(head_dim: int, window_size: int) -> dict:
    """Return one complete static sliding-attention launch config."""

    if int(window_size) <= 0:
        raise ValueError("sliding-window attention requires a positive window")
    index = CONFIG_INDEX_BY_HEAD_DIM.get(int(head_dim), DEFAULT_CONFIG_INDEX)
    candidate = CONFIGS[index]
    if PAGE_SIZE % candidate["BLOCK_N"] != 0:
        raise AssertionError("selected sliding-attention tile does not divide a page")
    return {**candidate, "D_HEAD": int(head_dim)}


# This sliding-attention kernel retains the complete paged KV prefix but skips
# tiles strictly before every query row's window. Its decomposition contract
# still conservatively assumes equality of every causal-prefix
# KV tile. A future eviction-aware certificate must strengthen this contract to
# mention only tiles inside the window before storage can be reclaimed.
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
#   scalar(window_size, int),
# )
# @grid(B, H, cdiv(max_seqlen_q, BLOCK_M))
# @verif(batch_invariance,
#   same(H, Hkv, D_HEAD, scale_log2, window_size),
#   pre(
#     right(B) == 1,
#     left(B) > 0,
#     left(Tq) > 0, right(Tq) > 0,
#     left(NUM_PAGES) > 0, right(NUM_PAGES) > 0,
#     H > 0, Hkv > 0, window_size > 0,
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
#   same(H, Hkv, D_HEAD, scale_log2, window_size),
#   pre(
#     left(B) == 1, right(B) == 1,
#     left(Tq) > 0, right(Tq) > 0,
#     left(Tk) > 0, right(Tk) > 0,
#     left(Tq) <= left(Tk), right(Tq) <= right(Tk),
#     left(Tq) <= left(max_seqlen_q),
#     right(Tq) <= right(max_seqlen_q),
#     H > 0, Hkv > 0, window_size > 0,
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
@triton.jit
def fattn_varlen_paged_swa_kernel(
    q,
    k_cache,
    v_cache,
    block_table,
    cu_seqlens_q,
    cu_seqlens_k,
    o,
    lse,
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
    window_size,
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
    # Keep absolute tile boundaries and the order of every potentially visible
    # tile. The first query row has the earliest window in this query block.
    window_k_start = max(0, q_shift + qi - window_size + 1)

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

    for ki in range(  # pyright: ignore[reportUnreachable]
        window_k_start // BLOCK_N, tl.cdiv(causal_k_end, BLOCK_N)
    ):
        k_tile_start = ki * BLOCK_N
        k_indices = k_tile_start + tl.arange(0, BLOCK_N)
        k_mask = k_indices < k_len

        page_slot = k_tile_start // PAGE_BLOCK_SIZE
        page_offset = k_tile_start % PAGE_BLOCK_SIZE
        page_id = tl.load(
            block_table
            + bi * stride_btb
            + page_slot * stride_bts
        )
        page_len = tl.minimum(
            PAGE_BLOCK_SIZE, k_len - page_slot * PAGE_BLOCK_SIZE
        )

        k_block_ptr = tl.make_block_ptr(
            base=k_cache + page_id * stride_kcb + hkvi * stride_kch,
            shape=(page_len, D_HEAD),
            strides=(stride_kcs, stride_kcd),
            offsets=(page_offset, 0),
            block_shape=(BLOCK_N, D_HEAD),
            order=(0, 1),
        )
        k_block = tl.load(
            k_block_ptr, boundary_check=(0,), padding_option="zero"
        )
        qk = tl.dot(q_block, tl.trans(k_block))

        attn_mask = q_mask[:, None] & k_mask[None, :]
        attn_mask &= k_indices[None, :] <= (
            q_indices[:, None] + q_shift
        )
        attn_mask &= k_indices[None, :] > (
            q_indices[:, None] + q_shift - window_size
        )
        row_has_any = tl.max(attn_mask, axis=1) > 0

        scores = qk * scale_log2
        masked_scores = tl.where(attn_mask, scores, float("-inf"))
        next_max = tl.maximum(scores_max, tl.max(masked_scores, axis=1))
        alpha = tl.exp2(scores_max - next_max)
        p = tl.where(
            attn_mask,
            tl.exp2(scores - next_max[:, None]),
            0.0,
        )

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
        v_block = tl.load(
            v_block_ptr, boundary_check=(0,), padding_option="zero"
        )
        next_acc = tl.dot(
            p.to(v_cache.dtype.element_ty),
            v_block,
            next_acc,
        )

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
    tl.store(
        o_block_ptr, out.to(o.dtype.element_ty), boundary_check=(0,)
    )

    lse_values = tl.where(
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
    tl.store(lse_block_ptr, lse_values, boundary_check=(0,))


def fattn_varlen_paged_swa(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    *,
    window_size: int,
    softmax_scale: float,
    block_table: torch.Tensor,
    value_checks: bool = True,
    launch_config: dict,
) -> torch.Tensor:
    if window_size <= 0:
        raise ValueError("sliding-window attention requires a positive window")
    (
        q,
        k_cache,
        v_cache,
        cu_seqlens_q,
        cu_seqlens_k,
        block_table,
        batch_size,
        num_heads,
        num_kv_heads,
        head_dim,
        total_q,
        resolved_scale,
    ) = _prepare_inputs(
        q,
        k_cache,
        v_cache,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        0.0,
        softmax_scale,
        0.0,
        None,
        block_table,
        value_checks=value_checks,
    )
    if launch_config["D_HEAD"] != head_dim:
        raise ValueError("attention head dimension differs from launch metadata")
    if PAGE_SIZE != k_cache.shape[1]:
        raise ValueError("attention page size differs from launch metadata")

    output = torch.empty_like(q, memory_format=torch.contiguous_format)
    softmax_lse = torch.empty(
        (num_heads, total_q), dtype=torch.float32, device=q.device
    )
    require_dense_launch_tensors(
        readonly={
            "q": q,
            "k_cache": k_cache,
            "v_cache": v_cache,
            "block_table": block_table,
            "cu_seqlens_q": cu_seqlens_q,
            "cu_seqlens_k": cu_seqlens_k,
        },
        writable={"o": output, "lse": softmax_lse},
    )
    if total_q == 0 or max_seqlen_q == 0:
        return output

    grid = (
        batch_size,
        num_heads,
        triton.cdiv(max_seqlen_q, launch_config["BLOCK_M"]),
    )
    scale_log2 = resolved_scale * 1.4426950408889634
    fattn_varlen_paged_swa_kernel[grid](
        q,
        k_cache,
        v_cache,
        block_table,
        cu_seqlens_q,
        cu_seqlens_k,
        output,
        softmax_lse,
        num_heads,
        num_kv_heads,
        *q.stride(),
        *k_cache.stride(),
        *v_cache.stride(),
        *block_table.stride(),
        *output.stride(),
        *softmax_lse.stride(),
        scale_log2,
        window_size,
        D_HEAD=launch_config["D_HEAD"],
        PAGE_BLOCK_SIZE=PAGE_SIZE,
        BLOCK_M=launch_config["BLOCK_M"],
        BLOCK_N=launch_config["BLOCK_N"],
        num_warps=launch_config["num_warps"],
        num_stages=launch_config["num_stages"],
    )
    return output
