import torch
import triton
import triton.language as tl

from triton_kernels.runtime_contracts import require_writable_tensors


def select_config(width: int) -> dict:
    width = int(width)
    if width <= 0:
        raise ValueError("RMSNorm width must be positive")
    return {"BLOCK_M": 1, "BLOCK_N": 1 << (width - 1).bit_length(),
            "num_warps": 8 if width in (3072, 4096) else 4,
            "num_stages": 3}


# @params(
#   tensor(x, float, shape(M, N), strides(stride_xm, stride_xn)),
#   tensor(w, float, shape(N), strides(stride_w)),
#   tensor(o, float, shape(M, N), strides(stride_om, stride_on)),
#   scalar(eps, float),
# )
# @grid(cdiv(M, BLOCK_M), cdiv(N, BLOCK_N))
# @verif(batch_invariance,
#   same(N, eps),
#   pre(
#     right(M) == 1,
#     b >= 0, b < left(M),
#     left(x)[b:b+1, 0:N] == right(x)[0:1, 0:N],
#     left(w)[0:N] == right(w)[0:N],
#   ),
#   post(
#     left(o)[b:b+1, 0:N] == right(o)[0:1, 0:N]
#   ),
# )
@triton.jit
def rmsnorm_kernel(
    x,
    w,
    o,
    M,
    N,
    stride_xm,
    stride_xn,
    stride_w,
    stride_om,
    stride_on,
    eps,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(axis=0) * BLOCK_M
    col = tl.program_id(axis=1) * BLOCK_N

    x_block_ptr = tl.make_block_ptr(
        x,
        shape=(M, N),
        strides=(stride_xm, stride_xn),
        offsets=(row, col),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(0, 1),
    )
    x_block = tl.load(x_block_ptr, boundary_check=(0, 1), padding_option="zero").to(tl.float32)

    # Compute variance: mean(x^2)
    x_sq = x_block * x_block
    var = tl.sum(x_sq, axis=1) / N
    rrms = tl.math.rsqrt(var + eps)

    # Normalize
    x_normed = x_block * rrms[:, None]

    # Scale by weight
    w_block_ptr = tl.make_block_ptr(
        w,
        shape=(N,),
        strides=(stride_w,),
        offsets=(col,),
        block_shape=(BLOCK_N,),
        order=(0,),
    )
    w_block = tl.load(w_block_ptr, boundary_check=(0,), padding_option="zero").to(tl.float32)
    result = x_normed * w_block[None, :]

    o_block_ptr = tl.make_block_ptr(
        o,
        shape=(M, N),
        strides=(stride_om, stride_on),
        offsets=(row, col),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(0, 1),
    )
    tl.store(o_block_ptr, result.to(o.dtype.element_ty), boundary_check=(0, 1))


def rmsnorm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    *,
    launch_config: dict,
) -> torch.Tensor:
    M, N = x.shape
    assert weight.shape == (N,), "Weight shape mismatch"

    # Logical write disjointness lifts to physical addresses because writable
    # outputs use a fresh, injective row-major layout.
    o = torch.empty_like(x, memory_format=torch.contiguous_format)
    require_writable_tensors(o=o)

    def grid(meta):
        return (
            triton.cdiv(M, meta["BLOCK_M"]),
            triton.cdiv(N, meta["BLOCK_N"]),
        )

    rmsnorm_kernel[grid](
        x,
        weight,
        o,
        M,
        N,
        *x.stride(),
        *weight.stride(),
        *o.stride(),
        eps,
        BLOCK_M=launch_config["BLOCK_M"],
        BLOCK_N=launch_config["BLOCK_N"],
        num_warps=launch_config["num_warps"],
        num_stages=launch_config["num_stages"],
    )

    return o


if __name__ == "__main__":
    M = 128
    N = 256
    device = "cuda"
    dtype = torch.bfloat16
    x = torch.randn((M, N), device=device, dtype=dtype)
    weight = torch.randn((N,), device=device, dtype=dtype)
    eps = 1e-6

    o_triton = rmsnorm(
        x,
        weight,
        eps,
        launch_config={
            "BLOCK_M": 1,
            "BLOCK_N": 256,
            "num_warps": 4,
            "num_stages": 3,
        },
    )

    # Reference
    x_f = x.float()
    var = x_f.pow(2).mean(dim=-1, keepdim=True)
    x_normed = x_f * torch.rsqrt(var + eps)
    o_ref = (x_normed * weight.float()).to(dtype)

    print(f"Max error: {(o_ref - o_triton).abs().max().item()}")
