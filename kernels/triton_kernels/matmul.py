import torch
import triton
import triton.language as tl

from triton_kernels.runtime_contracts import require_writable_tensors


CONFIGS = (
    {"BLOCK_M": 16, "BLOCK_N": 64, "BLOCK_K": 64, "num_warps": 2, "num_stages": 4},
    {"BLOCK_M": 16, "BLOCK_N": 128, "BLOCK_K": 64, "num_warps": 4, "num_stages": 4},
    {"BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_K": 64, "num_warps": 2, "num_stages": 4},
    {"BLOCK_M": 32, "BLOCK_N": 128, "BLOCK_K": 64, "num_warps": 4, "num_stages": 4},
    {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 64, "num_warps": 4, "num_stages": 4},
    {"BLOCK_M": 64, "BLOCK_N": 256, "BLOCK_K": 64, "num_warps": 8, "num_stages": 3},
    {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64, "num_warps": 8, "num_stages": 4},
    {"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 64, "num_warps": 8, "num_stages": 2},
    {"BLOCK_M": 64, "BLOCK_N": 256, "BLOCK_K": 64, "num_warps": 8, "num_stages": 4},
    {"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 64, "num_warps": 8, "num_stages": 4},
    {"BLOCK_M": 16, "BLOCK_N": 16, "BLOCK_K": 128, "num_warps": 2, "num_stages": 3},
    {"BLOCK_M": 16, "BLOCK_N": 64, "BLOCK_K": 128, "num_warps": 2, "num_stages": 3},
    {"BLOCK_M": 16, "BLOCK_N": 32, "BLOCK_K": 256, "num_warps": 4, "num_stages": 3},
    {"BLOCK_M": 16, "BLOCK_N": 64, "BLOCK_K": 256, "num_warps": 4, "num_stages": 3},
    {"BLOCK_M": 32, "BLOCK_N": 32, "BLOCK_K": 256, "num_warps": 4, "num_stages": 3},
    {"BLOCK_M": 64, "BLOCK_N": 32, "BLOCK_K": 256, "num_warps": 4, "num_stages": 3},
    {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 256, "num_warps": 4, "num_stages": 3},
    {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 128, "num_warps": 4, "num_stages": 3},
    {"BLOCK_M": 32, "BLOCK_N": 32, "BLOCK_K": 512, "num_warps": 4, "num_stages": 3},
)

# Offline performance policy over verified candidates. Unknown geometries use
# CONFIGS[0], the required fallback.
DEFAULT_CONFIG_INDEX = 0
CONFIG_INDEX_BY_NK = {
    (6144, 1024): 6,
    (4096, 4096): 14,
    (1024, 4096): 10,
    (24576, 4096): 4,
    (28672, 4096): 9,
    (4096, 12288): 3,
    (4096, 14336): 15,
    (128256, 4096): 11,
    (151936, 4096): 3,
    (3072, 3072): 14,
    (1024, 3072): 10,
    (16384, 3072): 4,
    (3072, 8192): 14,
    (128256, 3072): 11,
    (2048, 2560): 10,
    (1024, 2560): 10,
    (2560, 2048): 18,
    (20480, 2560): 4,
    (2560, 10240): 18,
    (262208, 2560): 13,
    (4096, 3840): 10,
    (2048, 3840): 10,
    (3840, 4096): 14,
    (30720, 3840): 9,
    (3840, 15360): 14,
    (262208, 3840): 13,
    (4096, 5376): 10,
    (2048, 5376): 10,
    (5376, 4096): 16,
    (43008, 5376): 17,
    (5376, 21504): 13,
    (262208, 5376): 13,
}


def select_config(n: int, k: int) -> dict:
    """Return one complete static launch config for a model matrix shape."""

    index = CONFIG_INDEX_BY_NK.get((int(n), int(k)), DEFAULT_CONFIG_INDEX)
    return dict(CONFIGS[index])

# @params(
#   tensor(a, float, shape(M, K), strides(stride_am, stride_ak)),
#   tensor(b, float, shape(K, N), strides(stride_bk, stride_bn)),
#   tensor(c, float, shape(M, N), strides(stride_cm, stride_cn)),
# )
# @grid(cdiv(M, BLOCK_M), cdiv(N, BLOCK_N))
# @verif(batch_invariance,
#   same(N, K),
#   pre(
#     right(M) == 1,
#     x >= 0, x < left(M),
#     left(a)[x:x+1, 0:K] == right(a)[0:1, 0:K],
#     left(b)[0:K, 0:N] == right(b)[0:K, 0:N],
#   ),
#   post(
#     left(c)[x:x+1, 0:N] == right(c)[0:1, 0:N]
#   ),
# )
# Serving supplies a complete config selected and qualified offline.
@triton.jit
def matmul_kernel(
    a,
    b,
    c,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    i = tl.program_id(axis=0)
    j = tl.program_id(axis=1)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(tl.cdiv(K, BLOCK_K)):  # pyright: ignore[reportUnreachable]
        block_ptr_a = tl.make_block_ptr(  # pyright: ignore[reportUnreachable]
            a,
            shape=(M, K),
            strides=(stride_am, stride_ak),
            offsets=(i * BLOCK_M, k * BLOCK_K),
            block_shape=(BLOCK_M, BLOCK_K),
            order=(0, 1),
        )
        block_ptr_b = tl.make_block_ptr(
            b,
            shape=(K, N),
            strides=(stride_bk, stride_bn),
            offsets=(k * BLOCK_K, j * BLOCK_N),
            block_shape=(BLOCK_K, BLOCK_N),
            order=(0, 1),
        )

        block_a = tl.load(block_ptr_a, boundary_check=(0, 1), padding_option="zero")
        block_b = tl.load(block_ptr_b, boundary_check=(0, 1), padding_option="zero")

        acc = tl.dot(block_a, block_b, acc)

    block_ptr_c = tl.make_block_ptr(
        c,
        shape=(M, N),
        strides=(stride_cm, stride_cn),
        offsets=(i * BLOCK_M, j * BLOCK_N),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(0, 1),
    )
    tl.store(block_ptr_c, acc.to(c.dtype.element_ty), boundary_check=(0, 1))


def matmul(
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    launch_config: dict,
) -> torch.Tensor:
    M, K = a.shape
    _K, N = b.shape
    dtype = a.dtype
    assert K == _K, "Incompatible dimensions"
    assert dtype == b.dtype, "Incompatible dtypes"

    c = torch.empty((M, N), device=a.device, dtype=dtype)
    require_writable_tensors(c=c)

    cfg = launch_config
    grid = (
        triton.cdiv(M, cfg["BLOCK_M"]),
        triton.cdiv(N, cfg["BLOCK_N"]),
    )

    matmul_kernel[grid](
        a,
        b,
        c,
        M,
        N,
        K,
        *a.stride(),  # pyright: ignore[reportArgumentType]
        *b.stride(),
        *c.stride(),
        BLOCK_M=cfg["BLOCK_M"],  # pyright: ignore[reportArgumentType]
        BLOCK_N=cfg["BLOCK_N"],  # pyright: ignore[reportArgumentType]
        BLOCK_K=cfg["BLOCK_K"],  # pyright: ignore[reportArgumentType]
        num_warps=cfg["num_warps"],
        num_stages=cfg["num_stages"],
    )

    return c


if __name__ == "__main__":
    M = 1024
    N = 1024
    K = 1024
    device = "cuda"
    dtype = torch.bfloat16
    a = torch.randn((M, K), device=device, dtype=dtype)
    b = torch.randn((K, N), device=device, dtype=dtype)
    c_triton = matmul(a, b, launch_config=select_config(N, K))
    c_torch = a @ b
    print((c_torch - c_triton).square().max().item())
