import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from triton_kernels.runtime_contracts import require_writable_tensors


def select_config(width: int) -> dict:
    if int(width) <= 0:
        raise ValueError("GELU-multiply width must be positive")
    # Match the shared activation tiling policy: bounded independent column
    # programs avoid oversized per-row working sets during decode.
    return {"BLOCK_M": 1, "BLOCK_N": 1024,
            "num_warps": 8, "num_stages": 1}


# Fused Gemma MLP activation: gelu_tanh(gate) * up.
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
def gelu_tanh_mul_kernel(
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
    ).to(tl.float32)

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

    inner = 0.7978845608028654 * (
        x_block + 0.044715 * x_block * x_block * x_block
    )
    # tanh(z) = 2 * sigmoid(2*z) - 1 keeps this kernel in the verifier's
    # already-qualified elementwise vocabulary.
    tanh_inner = 2.0 * tl.sigmoid(2.0 * inner) - 1.0
    gelu = (0.5 * x_block * (1.0 + tanh_inner)).to(x.dtype.element_ty)
    result = gelu * y_block

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
        result.to(o.dtype.element_ty),
        boundary_check=(0, 1),
    )


def gelu_tanh_mul(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    launch_config: dict,
) -> torch.Tensor:
    M, N = x.shape
    assert y.shape == (M, N), "Shape mismatch"

    o = torch.empty_like(x, memory_format=torch.contiguous_format)
    require_writable_tensors(o=o)

    def grid(meta):
        return (
            triton.cdiv(M, meta["BLOCK_M"]),
            triton.cdiv(N, meta["BLOCK_N"]),
        )

    gelu_tanh_mul_kernel[grid](
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


def torch_gelu_tanh_mul(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Literal unfused Gemma MLP activation reference."""

    return F.gelu(x, approximate="tanh") * y
