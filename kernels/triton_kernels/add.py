import torch
import triton
import triton.language as tl

from triton_kernels.runtime_contracts import require_writable_tensors


def select_config(width: int) -> dict:
    if int(width) <= 0:
        raise ValueError("add width must be positive")
    return {"BLOCK_M": 1, "BLOCK_N": 4096,
            "num_warps": 4, "num_stages": 3}


# Gemma applies post-sublayer normalization before this residual addition.
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
#     left(M) > 0, N > 0,
#     b >= 0, b < left(M),
#     left(x)[b:b+1, 0:N] == right(x)[0:1, 0:N],
#     left(y)[b:b+1, 0:N] == right(y)[0:1, 0:N],
#   ),
#   post(
#     left(o)[b:b+1, 0:N] == right(o)[0:1, 0:N]
#   ),
# )
@triton.jit
def add_kernel(
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
    x_block = tl.load(
        x_block_ptr, boundary_check=(0, 1), padding_option="zero"
    )

    y_block_ptr = tl.make_block_ptr(
        y,
        shape=(M, N),
        strides=(stride_ym, stride_yn),
        offsets=(row, col),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(0, 1),
    )
    y_block = tl.load(
        y_block_ptr, boundary_check=(0, 1), padding_option="zero"
    )

    o_block_ptr = tl.make_block_ptr(
        o,
        shape=(M, N),
        strides=(stride_om, stride_on),
        offsets=(row, col),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(0, 1),
    )
    tl.store(
        o_block_ptr,
        (x_block + y_block).to(o.dtype.element_ty),
        boundary_check=(0, 1),
    )


def add(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    launch_config: dict,
) -> torch.Tensor:
    M, N = x.shape
    if N <= 0:
        raise ValueError("add requires positive row width")
    assert y.shape == (M, N), "Shape mismatch"

    o = torch.empty_like(x, memory_format=torch.contiguous_format)
    require_writable_tensors(o=o)

    def grid(meta):
        return (
            triton.cdiv(M, meta["BLOCK_M"]),
            triton.cdiv(N, meta["BLOCK_N"]),
        )

    add_kernel[grid](
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
