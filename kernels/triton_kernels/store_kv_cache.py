import torch
import triton
import triton.language as tl

from triton_kernels.runtime_contracts import (
    require_int32_tensors,
    require_writable_tensors,
)


def select_config(kvd: int) -> dict:
    kvd = int(kvd)
    if kvd <= 0:
        raise ValueError("KV-store width must be positive")
    return {"KVD": kvd, "BLOCK_M": 1,
            "num_warps": 4, "num_stages": 3}


# Paged KV-cache scatter: row i of x is written to cache slot
# slot_mapping[i].  The cache is addressed flat: slot s is row s of a
# (NUM_SLOTS, KVD) view of the (num_pages, page_size, Hkv, D) cache, with
# s = page * page_size + offset and KVD = Hkv * D.  This matches vosti's
# slot semantics (`block_table_slot` / `cache_at`) exactly.  The K/V pair
# is stored by launching this kernel twice.
#
# Row equivalence: given duplicate-free in-bounds slots, the cache region
# written for token b in a full batch is identical to the region written by
# a singleton run on token b's row and slot.  This is the kernel-level
# non-interference fact behind vosti's `store_kv_cache` isolation lemmas.
# @params(
#   tensor(x, float, shape(M, KVD), strides(stride_xm, stride_xd)),
#   tensor(slot_mapping, int32, shape(M), strides(stride_s)),
#   tensor(cache, float, shape(NUM_SLOTS, KVD), strides(stride_cs, stride_cd)),
# )
# @grid(cdiv(M, BLOCK_M))
# @verif(batch_invariance,
#   same(KVD),
#   pre(
#     right(M) == 1,
#     left(M) > 0, KVD > 0,
#     left(NUM_SLOTS) > 0, right(NUM_SLOTS) > 0,
#     b >= 0, b < left(M),
#     forall(i, implies(and(i >= 0, i < left(M)), and(left(slot_mapping)[i] >= sub(0, 1), left(slot_mapping)[i] < left(NUM_SLOTS)))),
#     right(slot_mapping)[0] >= 0, right(slot_mapping)[0] < right(NUM_SLOTS),
#     left(slot_mapping)[b] >= 0,
#     forall(i, j, implies(and(i >= 0, i < left(M), j >= 0, j < left(M), left(slot_mapping)[i] >= 0, left(slot_mapping)[i] == left(slot_mapping)[j]), i == j)),
#     left(x)[b:b+1, 0:KVD] == right(x)[0:1, 0:KVD],
#   ),
#   post(
#     left(cache)[left(slot_mapping)[b]:left(slot_mapping)[b]+1, 0:KVD] == right(cache)[right(slot_mapping)[0]:right(slot_mapping)[0]+1, 0:KVD]
#   ),
# )
# Exact copy and framing are separate from the cross-execution row theorem.
# Matching physical dtypes makes the explicit cast and implicit store bitwise
# identities; injective nonnegative destinations rule out concurrent writers.
# @verif(exact_effect,
#   pre(
#     M >= 0, NUM_SLOTS > 0, KVD > 0,
#     dtype(x) == dtype(cache),
#     forall(i, implies(and(i >= 0, i < M),
#       and(before(slot_mapping)[i] >= sub(0, 1), before(slot_mapping)[i] < NUM_SLOTS))),
#     forall(i, j, implies(and(i >= 0, i < M, j >= 0, j < M,
#       before(slot_mapping)[i] >= 0, before(slot_mapping)[i] == before(slot_mapping)[j]), i == j)),
#   ),
#   post(
#     forall(i, implies(and(i >= 0, i < M, before(slot_mapping)[i] >= 0),
#       after(cache)[before(slot_mapping)[i]:before(slot_mapping)[i]+1, 0:KVD] == before(x)[i:i+1, 0:KVD])),
#     forall(s, implies(and(s >= 0, s < NUM_SLOTS,
#       forall(i, implies(and(i >= 0, i < M), not(before(slot_mapping)[i] == s)))),
#       after(cache)[s:s+1, 0:KVD] == before(cache)[s:s+1, 0:KVD])),
#   ),
# )
# @kernel-bridge-begin store_kv_cache::store_cache_kernel
@triton.jit
def store_cache_kernel(
    x,
    slot_mapping,
    cache,
    M,
    NUM_SLOTS,
    stride_xm,
    stride_xd,
    stride_s,
    stride_cs,
    stride_cd,
    KVD: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    row = tl.program_id(axis=0) * BLOCK_M

    # One token row at a time: each row goes to its own data-dependent slot.
    for r in range(BLOCK_M):  # pyright: ignore[reportUnreachable]
        cur_row = row + r  # pyright: ignore[reportUnreachable]
        # Load this row's target slot via scalar pointer (block-ptr offsets
        # must be 32-bit)
        s = tl.load(slot_mapping + cur_row * stride_s).to(tl.int32)

        x_block_ptr = tl.make_block_ptr(
            x,
            shape=(M, KVD),
            strides=(stride_xm, stride_xd),
            offsets=(cur_row, 0),
            block_shape=(1, KVD),
            order=(0, 1),
        )
        x_row = tl.load(x_block_ptr, boundary_check=(0, 1), padding_option="zero")

        cache_ptr = tl.make_block_ptr(
            cache,
            shape=(NUM_SLOTS, KVD),
            strides=(stride_cs, stride_cd),
            offsets=(s, 0),
            block_shape=(1, KVD),
            order=(0, 1),
        )
        # slot -1 is the one admitted no-write sentinel.  The block pointer's
        # row boundary check suppresses the complete store for that row.  It
        # lets a covering CUDA graph execute inert padding rows without a
        # scratch page or any mutation outside the real step's write set.
        tl.store(cache_ptr, x_row.to(cache.dtype.element_ty), boundary_check=(0, 1))
# @kernel-bridge-end store_kv_cache::store_cache_kernel


# @kernel-bridge-begin store_kv_cache::store_kv_cache
def store_kv_cache(
    k: torch.Tensor,
    v: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    *,
    launch_config: dict,
) -> None:
    """Scatter K/V token rows into the paged caches at slot_mapping.

    k, v: (M, Hkv, D) or (M, KVD).  k_cache, v_cache: (num_pages, page_size,
    Hkv, D) or any contiguous shape flat-viewable as (NUM_SLOTS, KVD).
    """
    require_int32_tensors(slot_mapping=slot_mapping)
    M = k.shape[0]
    if M == 0:
        return

    def grid(meta):
        return (triton.cdiv(M, meta["BLOCK_M"]),)

    for src, cache in ((k, k_cache), (v, v_cache)):
        require_writable_tensors(cache=cache)
        sf = src.reshape(M, -1)
        kvd = sf.shape[1]
        cf = cache.view(-1, kvd)
        store_cache_kernel[grid](
            sf,
            slot_mapping,
            cf,
            M,
            cf.shape[0],
            *sf.stride(),
            *slot_mapping.stride(),
            *cf.stride(),
            KVD=launch_config["KVD"],
            BLOCK_M=launch_config["BLOCK_M"],
            num_warps=launch_config["num_warps"],
            num_stages=launch_config["num_stages"],
        )
# @kernel-bridge-end store_kv_cache::store_kv_cache


if __name__ == "__main__":
    torch.manual_seed(0)
    device = "cuda"
    dtype = torch.bfloat16
    M, Hkv, D = 7, 2, 64
    num_pages, page_size = 4, 16
    k = torch.randn((M, Hkv, D), device=device, dtype=dtype)
    v = torch.randn((M, Hkv, D), device=device, dtype=dtype)
    k_cache = torch.zeros((num_pages, page_size, Hkv, D), device=device, dtype=dtype)
    v_cache = torch.zeros((num_pages, page_size, Hkv, D), device=device, dtype=dtype)
    slots = torch.tensor([3, 17, 18, 19, 40, 41, 63], device=device, dtype=torch.int32)

    store_kv_cache(
        k,
        v,
        k_cache,
        v_cache,
        slots,
        launch_config={
            "KVD": Hkv * D,
            "BLOCK_M": 1,
            "num_warps": 4,
            "num_stages": 3,
        },
    )

    kc = k_cache.view(-1, Hkv * D)
    vc = v_cache.view(-1, Hkv * D)
    err = 0.0
    for i, s in enumerate(slots.tolist()):
        err = max(err, (kc[s] - k.reshape(M, -1)[i]).abs().max().item())
        err = max(err, (vc[s] - v.reshape(M, -1)[i]).abs().max().item())
    print(f"Max error: {err}")
