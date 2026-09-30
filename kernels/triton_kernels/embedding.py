import torch
import triton
import triton.language as tl

from triton_kernels.runtime_contracts import (
    require_int32_tensors,
    require_writable_tensors,
)


def select_config(width: int) -> dict:
    width = int(width)
    if width <= 0:
        raise ValueError("embedding width must be positive")
    return {"D": width, "BLOCK_M": 1,
            "BLOCK_D": 1 << (width - 1).bit_length(),
            "num_warps": 4, "num_stages": 3}


# @params(
#   tensor(ids, int32, shape(M), strides(stride_i)),
#   tensor(weight, float, shape(V, D), strides(stride_wv, stride_wd)),
#   tensor(o, float, shape(M, D), strides(stride_om, stride_od)),
# )
# @grid(cdiv(M, BLOCK_M), cdiv(D, BLOCK_D))
# @verif(batch_invariance,
#   same(V, D, weight),
#   pre(
#     right(M) == 1,
#     x >= 0, x < left(M),
#     left(ids)[x:x+1] == right(ids)[0:1],
#     left(ids)[x] == right(ids)[0],
#   ),
#   post(
#     left(o)[x:x+1, 0:D] == right(o)[0:1, 0:D]
#   ),
# )
@triton.jit
def embedding_kernel(
    ids,
    weight,
    o,
    M,
    V,
    D,
    stride_i,
    stride_wv,
    stride_wd,
    stride_om,
    stride_od,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    row = tl.program_id(axis=0) * BLOCK_M
    col = tl.program_id(axis=1) * BLOCK_D

    # We process one row at a time since each row may access a different embedding
    for r in range(BLOCK_M):  # pyright: ignore[reportUnreachable]
        cur_row = row + r  # pyright: ignore[reportUnreachable]
        # Load token id for this row via scalar pointer (block-ptr offsets
        # must be 32-bit)
        tid = tl.load(ids + cur_row * stride_i).to(tl.int32)

        # Load embedding row
        w_block_ptr = tl.make_block_ptr(
            weight,
            shape=(V, D),
            strides=(stride_wv, stride_wd),
            offsets=(tid, col),
            block_shape=(1, BLOCK_D),
            order=(0, 1),
        )
        emb = tl.load(w_block_ptr, boundary_check=(0, 1), padding_option="zero")

        # Store to output
        o_block_ptr = tl.make_block_ptr(
            o,
            shape=(M, D),
            strides=(stride_om, stride_od),
            offsets=(cur_row, col),
            block_shape=(1, BLOCK_D),
            order=(0, 1),
        )
        tl.store(o_block_ptr, emb, boundary_check=(0, 1))


def embedding(
    ids: torch.Tensor,
    weight: torch.Tensor,
    *,
    launch_config: dict,
) -> torch.Tensor:
    require_int32_tensors(ids=ids)
    M = ids.shape[0]
    V, D = weight.shape
    assert launch_config["D"] == D, "Hidden-size launch metadata mismatch"

    o = torch.empty((M, D), device=ids.device, dtype=weight.dtype)
    require_writable_tensors(o=o)

    def grid(meta):
        return (
            triton.cdiv(M, meta["BLOCK_M"]),
            triton.cdiv(D, meta["BLOCK_D"]),
        )

    embedding_kernel[grid](
        ids,
        weight,
        o,
        M,
        V,
        D,
        *ids.stride(),
        *weight.stride(),
        *o.stride(),
        BLOCK_M=launch_config["BLOCK_M"],
        BLOCK_D=launch_config["BLOCK_D"],
        num_warps=launch_config["num_warps"],
        num_stages=launch_config["num_stages"],
    )

    return o


if __name__ == "__main__":
    M = 128
    V = 32000
    D = 128
    device = "cuda"
    dtype = torch.bfloat16

    ids = torch.randint(0, V, (M,), device=device, dtype=torch.int32)
    weight = torch.randn((V, D), device=device, dtype=dtype)

    o_triton = embedding(
        ids,
        weight,
        launch_config={
            "D": D,
            "BLOCK_M": 1,
            "BLOCK_D": triton.next_power_of_2(D),
            "num_warps": 4,
            "num_stages": 3,
        },
    )
    o_ref = torch.nn.functional.embedding(ids, weight)

    print(f"Max error: {(o_ref - o_triton).abs().max().item()}")
