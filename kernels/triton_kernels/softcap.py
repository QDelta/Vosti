"""Row-invariant logit softcapping with a deployment-fixed positive cap."""

import math

import torch
import triton
import triton.language as tl

from triton_kernels.runtime_contracts import require_writable_tensors


def select_config(width: int) -> dict:
    if type(width) is not int or width <= 0:
        raise ValueError("softcap width must be positive")
    return {"BLOCK_M": 1, "BLOCK_N": 1024, "num_warps": 4, "num_stages": 1}


# @params(
#   tensor(x, float, shape(M, N), strides(stride_xm, stride_xn)),
#   tensor(o, float, shape(M, N), strides(stride_om, stride_on)),
#   scalar(cap, float),
# )
# @grid(cdiv(M, BLOCK_M), cdiv(N, BLOCK_N))
# @verif(batch_invariance,
#   same(N, cap),
#   pre(
#     right(M) == 1,
#     b >= 0, b < left(M),
#     left(x)[b:b+1, 0:N] == right(x)[0:1, 0:N],
#   ),
#   post(left(o)[b:b+1, 0:N] == right(o)[0:1, 0:N]),
# )
@triton.jit
def softcap_kernel(x, o, cap, M, N, stride_xm, stride_xn, stride_om, stride_on,
                   BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    row = tl.program_id(0) * BLOCK_M
    col = tl.program_id(1) * BLOCK_N
    xp = tl.make_block_ptr(x, shape=(M, N), strides=(stride_xm, stride_xn),
        offsets=(row, col), block_shape=(BLOCK_M, BLOCK_N), order=(0, 1))
    op = tl.make_block_ptr(o, shape=(M, N), strides=(stride_om, stride_on),
        offsets=(row, col), block_shape=(BLOCK_M, BLOCK_N), order=(0, 1))
    xv = tl.load(xp, boundary_check=(0, 1), padding_option="zero").to(tl.float32)
    # Same already-supported elementwise vocabulary as the gated GELU kernel.
    result = cap * (2.0 * tl.sigmoid(2.0 * (xv / cap)) - 1.0)
    tl.store(op, result.to(o.dtype.element_ty), boundary_check=(0, 1))


def softcap(x: torch.Tensor, cap: float, *, launch_config: dict) -> torch.Tensor:
    if x.ndim != 2 or isinstance(cap, bool) or not math.isfinite(cap) or cap <= 0:
        raise ValueError("softcap requires a matrix and finite positive cap")
    o = torch.empty_like(x, memory_format=torch.contiguous_format)
    require_writable_tensors(o=o)
    m, n = x.shape
    softcap_kernel[lambda cfg: (triton.cdiv(m, cfg["BLOCK_M"]), triton.cdiv(n, cfg["BLOCK_N"]))](
        x, o, cap, m, n, *x.stride(), *o.stride(), **launch_config)
    return o
