import torch
import triton
import triton.language as tl

from triton_kernels.runtime_contracts import require_writable_tensors


def select_config(heads: int, head_dim: int) -> dict:
    heads, head_dim = int(heads), int(head_dim)
    if heads <= 0 or head_dim <= 0:
        raise ValueError("Q/K-normalization geometry must be positive")
    return {"H": heads, "D": head_dim, "BLOCK_M": 1,
            "num_warps": 4, "num_stages": 3}


# Per-head RMS norm (Qwen3-style q/k norm): each token row of x holds H
# heads of D features; every head slice is RMS-normalized independently
# with the shared (D,) weight.  The q/k pair is normalized by launching
# this kernel twice (once with the q weight, once with the k weight).
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
def head_rms_norm_kernel(
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
    w_block = tl.load(w_block_ptr, boundary_check=(0,), padding_option="zero").to(tl.float32)

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

            x_sq = x_block * x_block
            var = tl.sum(x_sq, axis=1) / D
            rrms = tl.math.rsqrt(var + eps)
            result = x_block * rrms[:, None] * w_block[None, :]

            o_block_ptr = tl.make_block_ptr(
                o,
                shape=(M, H * D),
                strides=(stride_om, stride_od),
                offsets=(cur_row, h * D),
                block_shape=(1, D),
                order=(0, 1),
            )
            tl.store(o_block_ptr, result.to(o.dtype.element_ty), boundary_check=(0, 1))


def head_rms_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    num_heads: int,
    eps: float = 1e-6,
    *,
    launch_config: dict,
) -> torch.Tensor:
    """RMS-normalize each head slice of x: (M, H*D) with weight (D,)."""
    M, HD = x.shape
    D = HD // num_heads
    assert weight.shape == (D,), "Weight shape mismatch"

    o = torch.empty_like(x, memory_format=torch.contiguous_format)
    require_writable_tensors(o=o)

    def grid(meta):
        return (triton.cdiv(M, meta["BLOCK_M"]),)

    head_rms_norm_kernel[grid](
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


def qk_norm(
    q: torch.Tensor,
    k: torch.Tensor,
    q_norm_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    num_heads: int,
    num_kv_heads: int,
    eps: float = 1e-6,
    *,
    q_launch_config: dict,
    k_launch_config: dict,
):
    """Qwen3 q/k norm: per-head RMS norm of q and k with separate weights."""
    nq = head_rms_norm(
        q.reshape(q.shape[0], -1),
        q_norm_weight,
        num_heads,
        eps,
        launch_config=q_launch_config,
    )
    nk = head_rms_norm(
        k.reshape(k.shape[0], -1),
        k_norm_weight,
        num_kv_heads,
        eps,
        launch_config=k_launch_config,
    )
    return nq, nk


if __name__ == "__main__":
    torch.manual_seed(0)
    device = "cuda"
    dtype = torch.bfloat16
    M, H, D = 64, 4, 128
    x = torch.randn((M, H * D), device=device, dtype=dtype)
    w = torch.randn((D,), device=device, dtype=dtype)
    eps = 1e-6

    o_triton = head_rms_norm(
        x,
        w,
        H,
        eps,
        launch_config={
            "H": H,
            "D": D,
            "BLOCK_M": 1,
            "num_warps": 4,
            "num_stages": 3,
        },
    )

    xh = x.float().reshape(M, H, D)
    var = xh.pow(2).mean(dim=-1, keepdim=True)
    o_ref = (xh * torch.rsqrt(var + eps) * w.float()).reshape(M, H * D).to(dtype)

    print(f"Max error: {(o_ref - o_triton).abs().max().item()}")
