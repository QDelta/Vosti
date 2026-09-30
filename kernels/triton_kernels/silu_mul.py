import torch
import triton
import triton.language as tl

from triton_kernels.runtime_contracts import require_writable_tensors


def select_config(width: int) -> dict:
    width = int(width)
    if width <= 0:
        raise ValueError("SiLU-multiply width must be positive")
    # Keep each program's elementwise working set bounded. Wider rows are
    # covered by additional independent column programs rather than one
    # oversized power-of-two tile.
    return {"BLOCK_M": 1, "BLOCK_N": 1024,
            "num_warps": 8, "num_stages": 1}


# @params(
#   tensor(x, float, shape(M, N), strides(stride_xm, stride_xn)),
#   tensor(y, float, shape(M, N), strides(stride_ym, stride_yn)),
#   tensor(o, float, shape(M, N), strides(stride_om, stride_on)),
# )
# @grid(cdiv(M, BLOCK_M), cdiv(N, BLOCK_N))
# @verif(batch_invariance,
#   same(N),
#   pre(
#     right(M) == 1,
#     b >= 0, b < left(M),
#     left(x)[b:b+1, 0:N] == right(x)[0:1, 0:N],
#     left(y)[b:b+1, 0:N] == right(y)[0:1, 0:N],
#   ),
#   post(
#     left(o)[b:b+1, 0:N] == right(o)[0:1, 0:N]
#   ),
# )
@triton.jit
def silu_mul_kernel(
    x,
    y,
    o,
    M,
    N,
    stride_xm,
    stride_xn,
    stride_ym,
    stride_yn,
    stride_om,
    stride_on,
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

    y_block_ptr = tl.make_block_ptr(
        y,
        shape=(M, N),
        strides=(stride_ym, stride_yn),
        offsets=(row, col),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(0, 1),
    )
    y_block = tl.load(y_block_ptr, boundary_check=(0, 1), padding_option="zero").to(tl.float32)

    # silu(x) * y = x * sigmoid(x) * y
    result = x_block * tl.sigmoid(x_block) * y_block

    o_block_ptr = tl.make_block_ptr(
        o,
        shape=(M, N),
        strides=(stride_om, stride_on),
        offsets=(row, col),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(0, 1),
    )
    tl.store(o_block_ptr, result.to(o.dtype.element_ty), boundary_check=(0, 1))


def silu_mul(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    launch_config: dict,
) -> torch.Tensor:
    M, N = x.shape
    assert y.shape == (M, N), "Shape mismatch"

    o = torch.empty((M, N), device=x.device, dtype=x.dtype)
    require_writable_tensors(o=o)

    def grid(meta):
        return (
            triton.cdiv(M, meta["BLOCK_M"]),
            triton.cdiv(N, meta["BLOCK_N"]),
        )

    silu_mul_kernel[grid](
        x,
        y,
        o,
        M,
        N,
        *x.stride(),
        *y.stride(),
        *o.stride(),
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
    y = torch.randn((M, N), device=device, dtype=dtype)

    o_triton = silu_mul(
        x,
        y,
        launch_config={
            "BLOCK_M": 1,
            "BLOCK_N": 256,
            "num_warps": 4,
            "num_stages": 3,
        },
    )

    # Reference
    o_ref = (torch.nn.functional.silu(x.float()) * y.float()).to(dtype)

    print(f"Max error: {(o_ref - o_triton).abs().max().item()}")
