import torch
import triton
import triton.language as tl

from triton_kernels.runtime_contracts import require_writable_tensors


def select_config(width: int) -> dict:
    width = int(width)
    if width <= 0:
        raise ValueError("residual-RMSNorm width must be positive")
    return {"BLOCK_M": 1, "BLOCK_N": 1 << (width - 1).bit_length(),
            "num_warps": 8 if width in (3072, 4096) else 4,
            "num_stages": 3}


# @params(
#   tensor(x, float, shape(M, N), strides(stride_xm, stride_xn)),
#   tensor(residual, float, shape(M, N), strides(stride_rm, stride_rn)),
#   tensor(w, float, shape(N), strides(stride_w)),
#   tensor(o, float, shape(M, N), strides(stride_om, stride_on)),
#   tensor(residual_out, float, shape(M, N), strides(stride_rom, stride_ron)),
#   scalar(eps, float),
# )
# @grid(cdiv(M, BLOCK_M), cdiv(N, BLOCK_N))
# @verif(batch_invariance,
#   same(N, eps),
#   pre(
#     right(M) == 1,
#     b >= 0, b < left(M),
#     left(x)[b:b+1, 0:N] == right(x)[0:1, 0:N],
#     left(residual)[b:b+1, 0:N] == right(residual)[0:1, 0:N],
#     left(w)[0:N] == right(w)[0:N],
#   ),
#   post(
#     left(o)[b:b+1, 0:N] == right(o)[0:1, 0:N],
#     left(residual_out)[b:b+1, 0:N] == right(residual_out)[0:1, 0:N]
#   ),
# )
@triton.jit
def rmsnorm_residual_kernel(
    x,
    residual,
    w,
    o,
    residual_out,
    M,
    N,
    stride_xm,
    stride_xn,
    stride_rm,
    stride_rn,
    stride_w,
    stride_om,
    stride_on,
    stride_rom,
    stride_ron,
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

    r_block_ptr = tl.make_block_ptr(
        residual,
        shape=(M, N),
        strides=(stride_rm, stride_rn),
        offsets=(row, col),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(0, 1),
    )
    r_block = tl.load(r_block_ptr, boundary_check=(0, 1), padding_option="zero").to(tl.float32)

    # Add residual
    hidden = x_block + r_block

    # Store updated residual
    ro_block_ptr = tl.make_block_ptr(
        residual_out,
        shape=(M, N),
        strides=(stride_rom, stride_ron),
        offsets=(row, col),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(0, 1),
    )
    tl.store(ro_block_ptr, hidden.to(residual_out.dtype.element_ty), boundary_check=(0, 1))

    # RMSNorm
    x_sq = hidden * hidden
    var = tl.sum(x_sq, axis=1) / N
    rrms = tl.math.rsqrt(var + eps)
    x_normed = hidden * rrms[:, None]

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


def rmsnorm_residual(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    *,
    launch_config: dict,
) -> tuple[torch.Tensor, torch.Tensor]:
    M, N = x.shape
    assert residual.shape == (M, N), "Residual shape mismatch"
    assert weight.shape == (N,), "Weight shape mismatch"

    # Separate fresh contiguous allocations discharge writable layout/no-alias.
    o = torch.empty_like(x, memory_format=torch.contiguous_format)
    residual_out = torch.empty_like(x, memory_format=torch.contiguous_format)
    require_writable_tensors(o=o, residual_out=residual_out)

    def grid(meta):
        return (
            triton.cdiv(M, meta["BLOCK_M"]),
            triton.cdiv(N, meta["BLOCK_N"]),
        )

    rmsnorm_residual_kernel[grid](
        x,
        residual,
        weight,
        o,
        residual_out,
        M,
        N,
        *x.stride(),
        *residual.stride(),
        *weight.stride(),
        *o.stride(),
        *residual_out.stride(),
        eps,
        BLOCK_M=launch_config["BLOCK_M"],
        BLOCK_N=launch_config["BLOCK_N"],
        num_warps=launch_config["num_warps"],
        num_stages=launch_config["num_stages"],
    )

    return o, residual_out


if __name__ == "__main__":
    M = 128
    N = 256
    device = "cuda"
    dtype = torch.bfloat16
    x = torch.randn((M, N), device=device, dtype=dtype)
    residual = torch.randn((M, N), device=device, dtype=dtype)
    weight = torch.randn((N,), device=device, dtype=dtype)
    eps = 1e-6

    o_triton, r_triton = rmsnorm_residual(
        x,
        residual,
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
    hidden = x.float() + residual.float()
    r_ref = hidden.to(dtype)
    var = hidden.pow(2).mean(dim=-1, keepdim=True)
    x_normed = hidden * torch.rsqrt(var + eps)
    o_ref = (x_normed * weight.float()).to(dtype)

    print(f"Output max error: {(o_ref - o_triton).abs().max().item()}")
    print(f"Residual max error: {(r_ref - r_triton).abs().max().item()}")
