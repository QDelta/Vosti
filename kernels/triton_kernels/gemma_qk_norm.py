import torch
import triton
import triton.language as tl

from triton_kernels.runtime_contracts import require_writable_tensors


def select_config(heads: int, head_dim: int) -> dict:
    heads, head_dim = int(heads), int(head_dim)
    if heads <= 0 or head_dim <= 0:
        raise ValueError("Gemma Q/K-normalization geometry must be positive")
    return {"H": heads, "D": head_dim, "BLOCK_M": 1,
            "num_warps": 4, "num_stages": 3}


# Gemma per-head Q/K RMSNorm uses the architecture-wide (1 + weight)
# convention. Q and K invoke this kernel separately with H=8 and H=4 for the
# 4B model while sharing D=256.
# @params(
#   tensor(x, float, shape(M, mul(H, D)), strides(stride_xm, stride_xd)),
#   tensor(w, float, shape(D), strides(stride_w)),
#   tensor(o, float, shape(M, mul(H, D)), strides(stride_om, stride_od)),
#   scalar(eps, float),
# )
# @grid(cdiv(M, BLOCK_M))
# @verif(batch_invariance,
#   same(H, D, eps),
#   pre(
#     right(M) == 1,
#     b >= 0, b < left(M),
#     left(x)[b:b+1, 0:mul(H, D)] == right(x)[0:1, 0:mul(H, D)],
#     left(w)[0:D] == right(w)[0:D],
#   ),
#   post(
#     left(o)[b:b+1, 0:mul(H, D)] == right(o)[0:1, 0:mul(H, D)]
#   ),
# )
@triton.jit
def gemma_head_rms_norm_kernel(
    x,
    w,
    o,
    M,
    stride_xm,
    stride_xd,
    stride_w,
    stride_om,
    stride_od,
    eps,
    H: tl.constexpr,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    row = tl.program_id(axis=0) * BLOCK_M

    w_block_ptr = tl.make_block_ptr(
        w,
        shape=(D,),
        strides=(stride_w,),
        offsets=(0,),
        block_shape=(D,),
        order=(0,),
    )
    w_block = tl.load(
        w_block_ptr, boundary_check=(0,), padding_option="zero"
    ).to(tl.float32)

    for r in range(BLOCK_M):  # pyright: ignore[reportUnreachable]
        cur_row = row + r  # pyright: ignore[reportUnreachable]
        for h in range(H):
            x_block_ptr = tl.make_block_ptr(
                x,
                shape=(M, H * D),
                strides=(stride_xm, stride_xd),
                offsets=(cur_row, h * D),
                block_shape=(1, D),
                order=(0, 1),
            )
            x_block = tl.load(
                x_block_ptr, boundary_check=(0, 1), padding_option="zero"
            ).to(tl.float32)

            variance = tl.sum(x_block * x_block, axis=1) / D
            normalized = x_block * tl.math.rsqrt(variance + eps)[:, None]
            result = normalized * (1.0 + w_block[None, :])

            o_block_ptr = tl.make_block_ptr(
                o,
                shape=(M, H * D),
                strides=(stride_om, stride_od),
                offsets=(cur_row, h * D),
                block_shape=(1, D),
                order=(0, 1),
            )
            tl.store(
                o_block_ptr,
                result.to(o.dtype.element_ty),
                boundary_check=(0, 1),
            )


def gemma_head_rms_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    num_heads: int,
    eps: float,
    *,
    launch_config: dict,
) -> torch.Tensor:
    M, hidden = x.shape
    D = hidden // num_heads
    assert hidden == num_heads * D, "Head geometry mismatch"
    assert weight.shape == (D,), "Weight shape mismatch"
    assert launch_config["H"] == num_heads, "Head launch metadata mismatch"
    assert launch_config["D"] == D, "Head-dimension launch metadata mismatch"

    o = torch.empty_like(x, memory_format=torch.contiguous_format)
    require_writable_tensors(o=o)

    def grid(meta):
        return (triton.cdiv(M, meta["BLOCK_M"]),)

    gemma_head_rms_norm_kernel[grid](
        x,
        weight,
        o,
        M,
        *x.stride(),
        *weight.stride(),
        *o.stride(),
        eps,
        H=launch_config["H"],
        D=launch_config["D"],
        BLOCK_M=launch_config["BLOCK_M"],
        num_warps=launch_config["num_warps"],
        num_stages=launch_config["num_stages"],
    )
    return o


def gemma_qk_norm(
    q: torch.Tensor,
    k: torch.Tensor,
    q_norm_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    num_heads: int,
    num_kv_heads: int,
    eps: float,
    *,
    q_launch_config: dict,
    k_launch_config: dict,
) -> tuple[torch.Tensor, torch.Tensor]:
    nq = gemma_head_rms_norm(
        q.reshape(q.shape[0], -1),
        q_norm_weight,
        num_heads,
        eps,
        launch_config=q_launch_config,
    )
    nk = gemma_head_rms_norm(
        k.reshape(k.shape[0], -1),
        k_norm_weight,
        num_kv_heads,
        eps,
        launch_config=k_launch_config,
    )
    return nq, nk


def torch_gemma_head_rms_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    num_heads: int,
    eps: float,
) -> torch.Tensor:
    rows, hidden = x.shape
    head_dim = hidden // num_heads
    x_heads = x.float().reshape(rows, num_heads, head_dim)
    normalized = x_heads * torch.rsqrt(
        x_heads.square().mean(dim=-1, keepdim=True) + eps
    )
    return (normalized * (1.0 + weight.float())).reshape(rows, hidden).to(x.dtype)
