"""Single-launch Q/K/V projection with separate weights and fresh outputs."""

import torch
import triton
import triton.language as tl

from triton_kernels.runtime_contracts import require_writable_tensors


CONFIGS = (
    {"BLOCK_M": 16, "BLOCK_N": 64, "BLOCK_K": 64,
     "num_warps": 2, "num_stages": 4},
    {"BLOCK_M": 16, "BLOCK_N": 32, "BLOCK_K": 128,
     "num_warps": 2, "num_stages": 3},
    {"BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_K": 256,
     "num_warps": 4, "num_stages": 3},
    {"BLOCK_M": 16, "BLOCK_N": 32, "BLOCK_K": 256,
     "num_warps": 4, "num_stages": 3},
)

# Offline performance policy over verified candidates. Unknown geometries use
# CONFIGS[0], the required fallback. The key contains only model geometry.
DEFAULT_CONFIG_INDEX = 0
CONFIG_INDEX_BY_QKV_GEOMETRY = {
    (2048, 1024, 1024): 1,
    (4096, 1024, 4096): 2,
    (2048, 1024, 2560): 3,
    (4096, 2048, 3840): 2,
    (4096, 2048, 5376): 2,
    (3072, 1024, 3072): 2,
    (8192, 1024, 8192): 1,
}


def select_config(q_width: int, kv_width: int, k: int) -> dict:
    """Return one complete static config from model-only QKV geometry."""

    geometry = (int(q_width), int(kv_width), int(k))
    index = CONFIG_INDEX_BY_QKV_GEOMETRY.get(geometry, DEFAULT_CONFIG_INDEX)
    return dict(CONFIGS[index])


# @params(
#   tensor(a, float, shape(M, K), strides(stride_am, stride_ak)),
#   tensor(wq, float, shape(K, QN), strides(stride_wqk, stride_wqn)),
#   tensor(wk, float, shape(K, KVN), strides(stride_wkk, stride_wkn)),
#   tensor(wv, float, shape(K, KVN), strides(stride_wvk, stride_wvn)),
#   tensor(oq, float, shape(M, QN), strides(stride_oqm, stride_oqn)),
#   tensor(ok, float, shape(M, KVN), strides(stride_okm, stride_okn)),
#   tensor(ov, float, shape(M, KVN), strides(stride_ovm, stride_ovn)),
# )
# Cover the three differently sized outputs without launching Q-width tiles
# for the narrower K and V projections.
# @grid(cdiv(M, BLOCK_M), add(cdiv(QN, BLOCK_N), mul(2, cdiv(KVN, BLOCK_N))))
# @verif(batch_invariance,
#   same(QN, KVN, K),
#   pre(
#     right(M) == 1,
#     QN >= KVN, KVN > 0, K > 0,
#     x >= 0, x < left(M),
#     left(a)[x:x+1, 0:K] == right(a)[0:1, 0:K],
#     left(wq)[0:K, 0:QN] == right(wq)[0:K, 0:QN],
#     left(wk)[0:K, 0:KVN] == right(wk)[0:K, 0:KVN],
#     left(wv)[0:K, 0:KVN] == right(wv)[0:K, 0:KVN],
#   ),
#   post(
#     left(oq)[x:x+1, 0:QN] == right(oq)[0:1, 0:QN],
#     left(ok)[x:x+1, 0:KVN] == right(ok)[0:1, 0:KVN],
#     left(ov)[x:x+1, 0:KVN] == right(ov)[0:1, 0:KVN]
#   ),
# )
@triton.jit
def qkv_matmul_kernel(
    a,
    wq,
    wk,
    wv,
    oq,
    ok,
    ov,
    M,
    QN,
    KVN,
    K,
    stride_am,
    stride_ak,
    stride_wqk,
    stride_wqn,
    stride_wkk,
    stride_wkn,
    stride_wvk,
    stride_wvn,
    stride_oqm,
    stride_oqn,
    stride_okm,
    stride_okn,
    stride_ovm,
    stride_ovn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    i = tl.program_id(axis=0)
    projection_tile = tl.program_id(axis=1)
    q_tiles = tl.cdiv(QN, BLOCK_N)
    kv_tiles = tl.cdiv(KVN, BLOCK_N)

    if projection_tile < q_tiles:
        q_j = projection_tile
        q_acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for q_k in range(tl.cdiv(K, BLOCK_K)):  # pyright: ignore[reportUnreachable]
            q_a_ptr = tl.make_block_ptr(  # pyright: ignore[reportUnreachable]
                a,
                shape=(M, K),
                strides=(stride_am, stride_ak),
                offsets=(i * BLOCK_M, q_k * BLOCK_K),
                block_shape=(BLOCK_M, BLOCK_K),
                order=(0, 1),
            )
            q_w_ptr = tl.make_block_ptr(
                wq,
                shape=(K, QN),
                strides=(stride_wqk, stride_wqn),
                offsets=(q_k * BLOCK_K, q_j * BLOCK_N),
                block_shape=(BLOCK_K, BLOCK_N),
                order=(0, 1),
            )
            q_a = tl.load(q_a_ptr, boundary_check=(0, 1), padding_option="zero")
            q_w = tl.load(q_w_ptr, boundary_check=(0, 1), padding_option="zero")
            q_acc = tl.dot(q_a, q_w, q_acc)
        q_o_ptr = tl.make_block_ptr(
            oq,
            shape=(M, QN),
            strides=(stride_oqm, stride_oqn),
            offsets=(i * BLOCK_M, q_j * BLOCK_N),
            block_shape=(BLOCK_M, BLOCK_N),
            order=(0, 1),
        )
        tl.store(q_o_ptr, q_acc.to(oq.dtype.element_ty), boundary_check=(0, 1))
    elif projection_tile < q_tiles + kv_tiles:
        k_j = projection_tile - q_tiles
        k_acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k_k in range(tl.cdiv(K, BLOCK_K)):  # pyright: ignore[reportUnreachable]
            k_a_ptr = tl.make_block_ptr(  # pyright: ignore[reportUnreachable]
                a,
                shape=(M, K),
                strides=(stride_am, stride_ak),
                offsets=(i * BLOCK_M, k_k * BLOCK_K),
                block_shape=(BLOCK_M, BLOCK_K),
                order=(0, 1),
            )
            k_w_ptr = tl.make_block_ptr(
                wk,
                shape=(K, KVN),
                strides=(stride_wkk, stride_wkn),
                offsets=(k_k * BLOCK_K, k_j * BLOCK_N),
                block_shape=(BLOCK_K, BLOCK_N),
                order=(0, 1),
            )
            k_a = tl.load(
                k_a_ptr, boundary_check=(0, 1), padding_option="zero"
            )
            k_w = tl.load(
                k_w_ptr, boundary_check=(0, 1), padding_option="zero"
            )
            k_acc = tl.dot(k_a, k_w, k_acc)
        k_o_ptr = tl.make_block_ptr(
            ok,
            shape=(M, KVN),
            strides=(stride_okm, stride_okn),
            offsets=(i * BLOCK_M, k_j * BLOCK_N),
            block_shape=(BLOCK_M, BLOCK_N),
            order=(0, 1),
        )
        tl.store(k_o_ptr, k_acc.to(ok.dtype.element_ty), boundary_check=(0, 1))
    else:
        v_j = projection_tile - q_tiles - kv_tiles
        v_acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for v_k in range(tl.cdiv(K, BLOCK_K)):  # pyright: ignore[reportUnreachable]
            v_a_ptr = tl.make_block_ptr(  # pyright: ignore[reportUnreachable]
                a,
                shape=(M, K),
                strides=(stride_am, stride_ak),
                offsets=(i * BLOCK_M, v_k * BLOCK_K),
                block_shape=(BLOCK_M, BLOCK_K),
                order=(0, 1),
            )
            v_w_ptr = tl.make_block_ptr(
                wv,
                shape=(K, KVN),
                strides=(stride_wvk, stride_wvn),
                offsets=(v_k * BLOCK_K, v_j * BLOCK_N),
                block_shape=(BLOCK_K, BLOCK_N),
                order=(0, 1),
            )
            v_a = tl.load(
                v_a_ptr, boundary_check=(0, 1), padding_option="zero"
            )
            v_w = tl.load(
                v_w_ptr, boundary_check=(0, 1), padding_option="zero"
            )
            v_acc = tl.dot(v_a, v_w, v_acc)
        v_o_ptr = tl.make_block_ptr(
            ov,
            shape=(M, KVN),
            strides=(stride_ovm, stride_ovn),
            offsets=(i * BLOCK_M, v_j * BLOCK_N),
            block_shape=(BLOCK_M, BLOCK_N),
            order=(0, 1),
        )
        tl.store(v_o_ptr, v_acc.to(ov.dtype.element_ty), boundary_check=(0, 1))


def qkv_matmul(
    a: torch.Tensor,
    wq: torch.Tensor,
    wk: torch.Tensor,
    wv: torch.Tensor,
    *,
    launch_config: dict,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute three projections in one launch without aliasing outputs."""

    if a.dim() != 2 or any(weight.dim() != 2 for weight in (wq, wk, wv)):
        raise ValueError("qkv_matmul expects four rank-2 tensors")
    m, k = a.shape
    if wq.shape[0] != k or wk.shape[0] != k or wv.shape[0] != k:
        raise ValueError("qkv_matmul input and weight reductions differ")
    q_width = int(wq.shape[1])
    kv_width = int(wk.shape[1])
    if wv.shape[1] != kv_width or q_width < kv_width or kv_width <= 0:
        raise ValueError("qkv_matmul has unsupported Q/K/V widths")
    if any(weight.dtype != a.dtype for weight in (wq, wk, wv)):
        raise ValueError("qkv_matmul input and weight dtypes differ")
    if any(weight.device != a.device for weight in (wq, wk, wv)):
        raise ValueError("qkv_matmul input and weight devices differ")

    oq = torch.empty((m, q_width), device=a.device, dtype=a.dtype)
    ok = torch.empty((m, kv_width), device=a.device, dtype=a.dtype)
    ov = torch.empty((m, kv_width), device=a.device, dtype=a.dtype)
    require_writable_tensors(oq=oq, ok=ok, ov=ov)

    config = launch_config
    q_tiles = triton.cdiv(q_width, config["BLOCK_N"])
    kv_tiles = triton.cdiv(kv_width, config["BLOCK_N"])
    grid = (triton.cdiv(m, config["BLOCK_M"]), q_tiles + 2 * kv_tiles)
    qkv_matmul_kernel[grid](
        a,
        wq,
        wk,
        wv,
        oq,
        ok,
        ov,
        m,
        q_width,
        kv_width,
        k,
        *a.stride(),  # pyright: ignore[reportArgumentType]
        *wq.stride(),
        *wk.stride(),
        *wv.stride(),
        *oq.stride(),
        *ok.stride(),
        *ov.stride(),
        BLOCK_M=config["BLOCK_M"],  # pyright: ignore[reportArgumentType]
        BLOCK_N=config["BLOCK_N"],  # pyright: ignore[reportArgumentType]
        BLOCK_K=config["BLOCK_K"],  # pyright: ignore[reportArgumentType]
        num_warps=config["num_warps"],
        num_stages=config["num_stages"],
    )
    return oq, ok, ov


__all__ = [
    "CONFIGS",
    "qkv_matmul",
    "qkv_matmul_kernel",
    "select_config",
]
