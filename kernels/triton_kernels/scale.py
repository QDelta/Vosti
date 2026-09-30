"""Row-invariant multiplication by an immutable scalar tensor."""

import torch
import triton
import triton.language as tl

from triton_kernels.runtime_contracts import require_writable_tensors


def select_config(width: int) -> dict:
    if type(width) is not int or width <= 0:
        raise ValueError("scale width must be positive")
    return {"BLOCK_M": 1, "BLOCK_N": 1024, "num_warps": 4, "num_stages": 1}


# @params(
#   tensor(x, float, shape(M, N), strides(stride_xm, stride_xn)),
#   tensor(s, float, shape(1), strides(stride_s)),
#   tensor(o, float, shape(M, N), strides(stride_om, stride_on)),
# )
# @grid(cdiv(M, BLOCK_M), cdiv(N, BLOCK_N))
# @verif(batch_invariance,
#   same(N),
#   pre(
#     right(M) == 1,
#     b >= 0, b < left(M),
#     left(x)[b:b+1, 0:N] == right(x)[0:1, 0:N],
#     left(s)[0:1] == right(s)[0:1],
#   ),
#   post(left(o)[b:b+1, 0:N] == right(o)[0:1, 0:N]),
# )
@triton.jit
def scale_kernel(x, s, o, M, N, stride_xm, stride_xn, stride_s,
                 stride_om, stride_on, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    row = tl.program_id(0) * BLOCK_M
    col = tl.program_id(1) * BLOCK_N
    xp = tl.make_block_ptr(x, shape=(M, N), strides=(stride_xm, stride_xn),
        offsets=(row, col), block_shape=(BLOCK_M, BLOCK_N), order=(0, 1))
    sp = tl.make_block_ptr(s, shape=(1,), strides=(stride_s,), offsets=(0,),
        block_shape=(1,), order=(0,))
    op = tl.make_block_ptr(o, shape=(M, N), strides=(stride_om, stride_on),
        offsets=(row, col), block_shape=(BLOCK_M, BLOCK_N), order=(0, 1))
    xv = tl.load(xp, boundary_check=(0, 1), padding_option="zero").to(tl.float32)
    sv = tl.load(sp, boundary_check=(0,), padding_option="zero").to(tl.float32)
    coefficients = tl.broadcast_to(sv[None, :], (BLOCK_M, BLOCK_N))
    tl.store(op, (xv * coefficients).to(o.dtype.element_ty), boundary_check=(0, 1))


def scale(x: torch.Tensor, scalar: torch.Tensor, *, launch_config: dict) -> torch.Tensor:
    if x.ndim != 2 or scalar.shape != (1,) or scalar.device != x.device:
        raise ValueError("scale requires a matrix and colocated one-element weight")
    o = torch.empty_like(x, memory_format=torch.contiguous_format)
    require_writable_tensors(o=o)
    m, n = x.shape
    scale_kernel[lambda cfg: (triton.cdiv(m, cfg["BLOCK_M"]), triton.cdiv(n, cfg["BLOCK_N"]))](
        x, scalar, o, m, n, *x.stride(), scalar.stride(0), *o.stride(), **launch_config)
    return o
