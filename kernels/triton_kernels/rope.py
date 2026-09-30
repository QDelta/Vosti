import torch
import triton
import triton.language as tl

from triton_kernels.runtime_contracts import require_writable_tensors


def select_config(width: int) -> dict:
    width = int(width)
    if width <= 0 or width % 2:
        raise ValueError("rotary width must be positive and even")
    return {"D": width, "HD": width // 2, "BLOCK_M": 1,
            "num_warps": 4, "num_stages": 3}


# @params(
#   tensor(x, float, shape(M, D), strides(stride_xm, stride_xd)),
#   tensor(cos_table, float, shape(M, HD), strides(stride_cm, stride_cd)),
#   tensor(sin_table, float, shape(M, HD), strides(stride_sm, stride_sd)),
#   tensor(o, float, shape(M, D), strides(stride_om, stride_od)),
# )
# @grid(cdiv(M, BLOCK_M))
# @verif(batch_invariance,
#   same(D, HD),
#   pre(
#     right(M) == 1,
#     b >= 0, b < left(M),
#     left(x)[b:b+1, 0:D] == right(x)[0:1, 0:D],
#     left(cos_table)[b:b+1, 0:HD] == right(cos_table)[0:1, 0:HD],
#     left(sin_table)[b:b+1, 0:HD] == right(sin_table)[0:1, 0:HD],
#   ),
#   post(
#     left(o)[b:b+1, 0:D] == right(o)[0:1, 0:D]
#   ),
# )
@triton.jit
def rope_kernel(
    x,
    cos_table,
    sin_table,
    o,
    M,
    D: tl.constexpr,
    HD: tl.constexpr,
    stride_xm,
    stride_xd,
    stride_cm,
    stride_cd,
    stride_sm,
    stride_sd,
    stride_om,
    stride_od,
    BLOCK_M: tl.constexpr,
):
    row = tl.program_id(axis=0) * BLOCK_M

    # Load cos and sin
    cos_block_ptr = tl.make_block_ptr(
        cos_table,
        shape=(M, HD),
        strides=(stride_cm, stride_cd),
        offsets=(row, 0),
        block_shape=(BLOCK_M, HD),
        order=(0, 1),
    )
    cos_block = tl.load(cos_block_ptr, boundary_check=(0,), padding_option="zero").to(tl.float32)

    sin_block_ptr = tl.make_block_ptr(
        sin_table,
        shape=(M, HD),
        strides=(stride_sm, stride_sd),
        offsets=(row, 0),
        block_shape=(BLOCK_M, HD),
        order=(0, 1),
    )
    sin_block = tl.load(sin_block_ptr, boundary_check=(0,), padding_option="zero").to(tl.float32)

    # Load x as two halves: x1 = x[:, :HD], x2 = x[:, HD:]
    x1_block_ptr = tl.make_block_ptr(
        x,
        shape=(M, D),
        strides=(stride_xm, stride_xd),
        offsets=(row, 0),
        block_shape=(BLOCK_M, HD),
        order=(0, 1),
    )
    x1 = tl.load(x1_block_ptr, boundary_check=(0,), padding_option="zero").to(tl.float32)

    x2_block_ptr = tl.make_block_ptr(
        x,
        shape=(M, D),
        strides=(stride_xm, stride_xd),
        offsets=(row, HD),
        block_shape=(BLOCK_M, HD),
        order=(0, 1),
    )
    x2 = tl.load(x2_block_ptr, boundary_check=(0,), padding_option="zero").to(tl.float32)

    # Apply rotation
    y1 = x1 * cos_block - x2 * sin_block
    y2 = x2 * cos_block + x1 * sin_block

    # Store first half
    o1_block_ptr = tl.make_block_ptr(
        o,
        shape=(M, D),
        strides=(stride_om, stride_od),
        offsets=(row, 0),
        block_shape=(BLOCK_M, HD),
        order=(0, 1),
    )
    tl.store(o1_block_ptr, y1.to(o.dtype.element_ty), boundary_check=(0,))

    # Store second half
    o2_block_ptr = tl.make_block_ptr(
        o,
        shape=(M, D),
        strides=(stride_om, stride_od),
        offsets=(row, HD),
        block_shape=(BLOCK_M, HD),
        order=(0, 1),
    )
    tl.store(o2_block_ptr, y2.to(o.dtype.element_ty), boundary_check=(0,))


def rope(
    x: torch.Tensor,
    cos_table: torch.Tensor,
    sin_table: torch.Tensor,
    *,
    launch_config: dict,
) -> torch.Tensor:
    M, D = x.shape
    HD = D // 2
    assert cos_table.shape == (M, HD), "cos_table shape mismatch"
    assert sin_table.shape == (M, HD), "sin_table shape mismatch"

    o = torch.empty_like(x, memory_format=torch.contiguous_format)
    require_writable_tensors(o=o)

    def grid(meta):
        return (triton.cdiv(M, meta["BLOCK_M"]),)

    rope_kernel[grid](
        x,
        cos_table,
        sin_table,
        o,
        M,
        launch_config["D"],
        launch_config["HD"],
        *x.stride(),
        *cos_table.stride(),
        *sin_table.stride(),
        *o.stride(),
        BLOCK_M=launch_config["BLOCK_M"],
        num_warps=launch_config["num_warps"],
        num_stages=launch_config["num_stages"],
    )

    return o


if __name__ == "__main__":
    M = 128
    D = 128
    device = "cuda"
    dtype = torch.bfloat16
    x = torch.randn((M, D), device=device, dtype=dtype)
    cos_table = torch.randn((M, D // 2), device=device, dtype=dtype)
    sin_table = torch.randn((M, D // 2), device=device, dtype=dtype)

    o_triton = rope(
        x,
        cos_table,
        sin_table,
        launch_config={
            "D": D,
            "HD": D // 2,
            "BLOCK_M": 1,
            "num_warps": 4,
            "num_stages": 3,
        },
    )

    # Reference
    x_f = x.float()
    x1, x2 = x_f[:, : D // 2], x_f[:, D // 2 :]
    cos_f = cos_table.float()
    sin_f = sin_table.float()
    y1 = x1 * cos_f - x2 * sin_f
    y2 = x2 * cos_f + x1 * sin_f
    o_ref = torch.cat([y1, y2], dim=-1).to(dtype)

    print(f"Max error: {(o_ref - o_triton).abs().max().item()}")
