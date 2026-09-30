//! Architecture-neutral physical-layout predicates shared by model families.
//!
//! Family adapters discharge compatibility with family-local predicates;
//! generic dispatchers depend only on this canonical definition.

use crate::{types::*, proof::model::types::*, proof::tensor::types::*};
use vstd::prelude::*;

verus! {

// Common batched request-projection domain exported by Engine.  It is
// intentionally stronger than either current family's local theorem at only
// one point: cache capacity covers every cell in the key row's final page.
// That strength lets full-attention and sliding-window adapters share the same
// physical premise while retaining the complete causal KV allocation.
pub open spec fn request_projection_common_domain(
    wr: ModelWeightsRepr,
    architecture_repr: ModelWeightsArchitectureRepr,
    input_ids: IntTensor1D,
    positions: IntTensor1D,
    pre_kv: Seq<(KVCacheLayerRepr, KVCacheLayerRepr)>,
    slots: Seq<int>,
    cu_q: Seq<int>,
    cu_k: Seq<int>,
    max_q: nat,
    max_k: nat,
    block_table: Seq<Seq<BlockId>>,
    row: nat,
) -> bool {
    &&& crate::boundary::tensor_runtime::paged_attention_numeric_domain()
    &&& crate::proof::model::architecture::request_projection_configuration_ready(
        wr,
        architecture_repr,
    )
    &&& pre_kv.len() >= wr.layers.len()
    &&& input_ids.len() == positions.len()
    &&& slots.len() == input_ids.len()
    &&& cu_q.len() == cu_k.len()
    &&& cu_k.len() == block_table.len() + 1
    &&& row < block_table.len()
    &&& cu_q[0] == 0
    &&& cu_k[0] == 0
    &&& cu_q[block_table.len() as int] == input_ids.len() as int
    &&& forall|j: int| 0 <= j < block_table.len() as int ==>
        cu_q[j] < #[trigger] cu_q[j + 1]
    &&& forall|j: int| 0 <= j < block_table.len() as int ==>
        cu_k[j] < #[trigger] cu_k[j + 1]
    &&& 0 <= cu_q[row as int] < cu_q[row as int + 1]
        <= input_ids.len() as int
    &&& 0 <= cu_k[row as int] < cu_k[row as int + 1]
    &&& max_q > 0
    &&& cu_q[row as int + 1] - cu_q[row as int] <= max_q as int
    &&& cu_k[row as int + 1] - cu_k[row as int] <= max_k as int
    &&& cu_q[row as int + 1] - cu_q[row as int]
        <= cu_k[row as int + 1] - cu_k[row as int]
    &&& crate::proof::tensor::geometry::blocks_needed_for(
        (cu_k[row as int + 1] - cu_k[row as int]) as nat,
    ) <= block_table[row as int].len()
    &&& forall|m: int, layer: int|
        #![trigger slots.subrange(0, cu_q[row as int])[m]
            / (crate::types::BLOCK_SIZE_SPEC as int),
            block_table[row as int][layer]]
        0 <= m < cu_q[row as int]
            && 0 <= layer < block_table[row as int].len() ==>
            slots.subrange(0, cu_q[row as int])[m]
                / (crate::types::BLOCK_SIZE_SPEC as int)
                != block_table[row as int][layer] as int
    &&& forall|m: int, layer: int|
        #![trigger slots.subrange(
            cu_q[row as int + 1], slots.len() as int,
        )[m] / (crate::types::BLOCK_SIZE_SPEC as int),
            block_table[row as int][layer]]
        0 <= m < slots.len() - cu_q[row as int + 1]
            && 0 <= layer < block_table[row as int].len() ==>
            slots.subrange(
                cu_q[row as int + 1], slots.len() as int,
            )[m] / (crate::types::BLOCK_SIZE_SPEC as int)
                != block_table[row as int][layer] as int
    &&& forall|layer: int, pos: nat|
        #![trigger pre_kv[layer].0,
            crate::proof::tensor::geometry::block_table_slot(
                block_table[row as int], pos,
            )]
        0 <= layer < wr.layers.len()
            && pos < crate::proof::tensor::geometry::blocks_needed_for(
                (cu_k[row as int + 1] - cu_k[row as int]) as nat,
            ) * crate::types::BLOCK_SIZE_SPEC ==>
            crate::proof::tensor::geometry::slot_in_cache(
                pre_kv[layer].0,
                crate::proof::tensor::geometry::block_table_slot(
                    block_table[row as int], pos,
                ),
            )
            && crate::proof::tensor::geometry::slot_in_cache(
                pre_kv[layer].1,
                crate::proof::tensor::geometry::block_table_slot(
                    block_table[row as int], pos,
                ),
            )
    &&& forall|layer: int, pos: nat|
        #![trigger pre_kv[layer].0,
            crate::proof::tensor::geometry::block_table_slot(
                block_table[row as int], pos,
            )]
        0 <= layer < wr.layers.len()
            && pos < (cu_k[row as int + 1] - cu_k[row as int]) as nat ==>
            crate::proof::tensor::geometry::slot_in_cache(
                pre_kv[layer].0,
                crate::proof::tensor::geometry::block_table_slot(
                    block_table[row as int], pos,
                ),
            )
            && crate::proof::tensor::geometry::slot_in_cache(
                pre_kv[layer].1,
                crate::proof::tensor::geometry::block_table_slot(
                    block_table[row as int], pos,
                ),
            )
    &&& forall|layer: int| 0 <= layer < wr.layers.len() ==>
        #[trigger] crate::boundary::tensor_runtime::paged_attention_launch_ready(
            input_ids.len(), pre_kv[layer].0, pre_kv[layer].1,
            cu_q, cu_k, max_q, max_k, block_table,
        )
}

pub open spec fn fresh_writes_miss_cached_prefix(
    slot_a: Seq<int>,
    bt_row_a: Seq<BlockId>,
    slot_b: Seq<int>,
    bt_row_b: Seq<BlockId>,
    prefix_len: int,
) -> bool {
    forall|pos: nat| #![trigger crate::proof::tensor::geometry::block_table_slot(bt_row_a, pos)]
        pos < prefix_len ==>
            !slot_a.contains(
                crate::proof::tensor::geometry::block_table_slot(bt_row_a, pos) as int,
            )
            && !slot_b.contains(
                crate::proof::tensor::geometry::block_table_slot(bt_row_b, pos) as int,
            )
}

pub proof fn fresh_writes_miss_cached_prefix_at(
    slot_a: Seq<int>,
    bt_row_a: Seq<BlockId>,
    slot_b: Seq<int>,
    bt_row_b: Seq<BlockId>,
    prefix_len: int,
    pos: nat,
)
    requires
        fresh_writes_miss_cached_prefix(
            slot_a, bt_row_a, slot_b, bt_row_b, prefix_len,
        ),
        pos < prefix_len,
        (pos as int) / (crate::types::BLOCK_SIZE_SPEC as int) < bt_row_a.len(),
        (pos as int) / (crate::types::BLOCK_SIZE_SPEC as int) < bt_row_b.len(),
    ensures
        !slot_a.contains(
            crate::proof::tensor::geometry::block_table_slot(bt_row_a, pos) as int,
        ),
        !slot_b.contains(
            crate::proof::tensor::geometry::block_table_slot(bt_row_b, pos) as int,
        ),
{
    reveal(fresh_writes_miss_cached_prefix);
    assert(!slot_a.contains(
        crate::proof::tensor::geometry::block_table_slot(bt_row_a, pos) as int,
    ));
    assert(!slot_b.contains(
        crate::proof::tensor::geometry::block_table_slot(bt_row_b, pos) as int,
    ));
}

// A duplicate-free page table maps distinct logical positions to distinct
// physical slots.  This address fact is shared by every model family and by
// scheduler proofs that establish scatter uniqueness.
pub proof fn lemma_block_table_slot_injective(
    block_table: Seq<BlockId>,
    left: nat,
    right: nat,
)
    requires
        block_table.no_duplicates(),
        left != right,
        (left as int) / (crate::types::BLOCK_SIZE_SPEC as int)
            < block_table.len(),
        (right as int) / (crate::types::BLOCK_SIZE_SPEC as int)
            < block_table.len(),
    ensures
        crate::proof::tensor::geometry::block_table_slot(block_table, left)
            != crate::proof::tensor::geometry::block_table_slot(block_table, right),
{
    let block_size = crate::types::BLOCK_SIZE_SPEC as int;
    let left_page = (left as int) / block_size;
    let right_page = (right as int) / block_size;
    vstd::arithmetic::div_mod::lemma_fundamental_div_mod(
        left as int, block_size,
    );
    vstd::arithmetic::div_mod::lemma_fundamental_div_mod(
        right as int, block_size,
    );
    vstd::arithmetic::div_mod::lemma_mod_bound(
        left as int, block_size,
    );
    vstd::arithmetic::div_mod::lemma_mod_bound(
        right as int, block_size,
    );
    if left_page == right_page {
        assert((left as int) % block_size
            != (right as int) % block_size);
        assert(crate::proof::tensor::geometry::block_table_slot(block_table, left) as int
            == block_table[left_page] * block_size
                + (left as int) % block_size);
        assert(crate::proof::tensor::geometry::block_table_slot(block_table, right) as int
            == block_table[right_page] * block_size
                + (right as int) % block_size);
    } else {
        assert(block_table[left_page] != block_table[right_page]) by {
            reveal(Seq::no_duplicates);
        }
        crate::proof::tensor::geometry::block_table_slot_block(block_table, left);
        crate::proof::tensor::geometry::block_table_slot_block(block_table, right);
        assert((crate::proof::tensor::geometry::block_table_slot(block_table, left) as int)
            / block_size == block_table[left_page] as int);
        assert((crate::proof::tensor::geometry::block_table_slot(block_table, right) as int)
            / block_size == block_table[right_page] as int);
    }
}

// Copy a logical prefix from a paged cache into the same logical positions of
// a private contiguous cache, retaining the destination's page geometry and
// every cell outside the prefix.  This is scheduler/model neutral: a cached
// admission uses it to construct the base of an independent singleton
// forward, while an already-running request gets an extensionally identical
// copy of its coherent prefix.
pub open spec fn relocated_prefix_base(
    destination: KVCacheLayerRepr,
    source: KVCacheLayerRepr,
    source_bt_row: Seq<BlockId>,
    prefix_len: nat,
) -> KVCacheLayerRepr {
    Seq::new(destination.len(), |page: int| {
        Seq::new(destination[page].len(), |offset: int| {
            let pos = page * (crate::types::BLOCK_SIZE_SPEC as int) + offset;
            if 0 <= pos < prefix_len as int {
                crate::proof::tensor::geometry::cache_at(
                    source,
                    crate::proof::tensor::geometry::block_table_slot(
                        source_bt_row, pos as nat,
                    ),
                )
            } else {
                destination[page][offset]
            }
        })
    })
}

pub proof fn lemma_relocated_prefix_base_shape(
    destination: KVCacheLayerRepr,
    source: KVCacheLayerRepr,
    source_bt_row: Seq<BlockId>,
    prefix_len: nat,
)
    ensures
        relocated_prefix_base(
            destination, source, source_bt_row, prefix_len,
        ).len() == destination.len(),
        forall|page: int| 0 <= page < destination.len() ==>
            (#[trigger] relocated_prefix_base(
                destination, source, source_bt_row, prefix_len,
            )[page]).len() == destination[page].len(),
{
    reveal(relocated_prefix_base);
}

pub proof fn lemma_relocated_prefix_base_at(
    destination: KVCacheLayerRepr,
    source: KVCacheLayerRepr,
    source_bt_row: Seq<BlockId>,
    prefix_len: nat,
    pos: nat,
)
    requires crate::proof::tensor::geometry::slot_in_cache(destination, pos),
    ensures
        crate::proof::tensor::geometry::slot_in_cache(
            relocated_prefix_base(
                destination, source, source_bt_row, prefix_len,
            ),
            pos,
        ),
        pos < prefix_len ==>
            crate::proof::tensor::geometry::cache_at(
                relocated_prefix_base(
                    destination, source, source_bt_row, prefix_len,
                ),
                pos,
            ) == crate::proof::tensor::geometry::cache_at(
                source,
                crate::proof::tensor::geometry::block_table_slot(source_bt_row, pos),
            ),
        pos >= prefix_len ==>
            crate::proof::tensor::geometry::cache_at(
                relocated_prefix_base(
                    destination, source, source_bt_row, prefix_len,
                ),
                pos,
            ) == crate::proof::tensor::geometry::cache_at(destination, pos),
{
    let block_size = crate::types::BLOCK_SIZE_SPEC as int;
    let page = pos as int / block_size;
    let offset = pos as int % block_size;
    vstd::arithmetic::div_mod::lemma_fundamental_div_mod(
        pos as int, block_size,
    );
    vstd::arithmetic::div_mod::lemma_mod_bound(pos as int, block_size);
    assert(page * block_size + offset == pos as int);
    let base = relocated_prefix_base(
        destination, source, source_bt_row, prefix_len,
    );
    assert(base[page].len() == destination[page].len());
    assert(base[page][offset] ==
        if 0 <= page * block_size + offset < prefix_len as int {
            crate::proof::tensor::geometry::cache_at(
                source,
                crate::proof::tensor::geometry::block_table_slot(
                    source_bt_row,
                    (page * block_size + offset) as nat,
                ),
            )
        } else {
            destination[page][offset]
        });
}

// Architecture-neutral equality of two physical cache pairs when addressed
// through their respective logical block-table rows.  The contract covers the
// complete causal prefix for both full and sliding attention; it deliberately
// makes no window-only dependence claim.
pub open spec fn cache_pair_logical_prefix_equal(
    cache_a: (KVCacheLayerRepr, KVCacheLayerRepr),
    bt_row_a: Seq<BlockId>,
    cache_b: (KVCacheLayerRepr, KVCacheLayerRepr),
    bt_row_b: Seq<BlockId>,
    prefix_len: nat,
) -> bool {
    crate::proof::tensor::geometry::blocks_needed_for(prefix_len) <= bt_row_a.len()
    && crate::proof::tensor::geometry::blocks_needed_for(prefix_len) <= bt_row_b.len()
    && forall|pos: nat|
        #![trigger crate::proof::tensor::geometry::block_table_slot(bt_row_a, pos)]
        pos < prefix_len ==> {
            let slot_a = crate::proof::tensor::geometry::block_table_slot(bt_row_a, pos);
            let slot_b = crate::proof::tensor::geometry::block_table_slot(bt_row_b, pos);
            &&& crate::proof::tensor::geometry::slot_in_cache(cache_a.0, slot_a)
            &&& crate::proof::tensor::geometry::slot_in_cache(cache_a.1, slot_a)
            &&& crate::proof::tensor::geometry::slot_in_cache(cache_b.0, slot_b)
            &&& crate::proof::tensor::geometry::slot_in_cache(cache_b.1, slot_b)
            &&& crate::proof::tensor::geometry::cache_at(cache_a.0, slot_a)
                == crate::proof::tensor::geometry::cache_at(cache_b.0, slot_b)
            &&& crate::proof::tensor::geometry::cache_at(cache_a.1, slot_a)
                == crate::proof::tensor::geometry::cache_at(cache_b.1, slot_b)
        }
}

pub open spec fn cache_sequence_logical_prefix_equal(
    caches_a: Seq<(KVCacheLayerRepr, KVCacheLayerRepr)>,
    bt_row_a: Seq<BlockId>,
    caches_b: Seq<(KVCacheLayerRepr, KVCacheLayerRepr)>,
    bt_row_b: Seq<BlockId>,
    num_layers: nat,
    prefix_len: nat,
) -> bool {
    num_layers <= caches_a.len()
    && num_layers <= caches_b.len()
    && forall|layer: int| 0 <= layer < num_layers ==>
        #[trigger] cache_pair_logical_prefix_equal(
            caches_a[layer], bt_row_a,
            caches_b[layer], bt_row_b,
            prefix_len,
        )
}

pub proof fn lemma_cache_pair_logical_prefix_equal_transitive(
    cache_a: (KVCacheLayerRepr, KVCacheLayerRepr),
    bt_row_a: Seq<BlockId>,
    cache_b: (KVCacheLayerRepr, KVCacheLayerRepr),
    bt_row_b: Seq<BlockId>,
    cache_c: (KVCacheLayerRepr, KVCacheLayerRepr),
    bt_row_c: Seq<BlockId>,
    prefix_len: nat,
)
    requires
        cache_pair_logical_prefix_equal(
            cache_a, bt_row_a, cache_b, bt_row_b, prefix_len,
        ),
        cache_pair_logical_prefix_equal(
            cache_b, bt_row_b, cache_c, bt_row_c, prefix_len,
        ),
    ensures
        cache_pair_logical_prefix_equal(
            cache_a, bt_row_a, cache_c, bt_row_c, prefix_len,
        ),
{
    reveal(cache_pair_logical_prefix_equal);
    assert forall|pos: nat|
        #![trigger crate::proof::tensor::geometry::block_table_slot(bt_row_a, pos)]
        pos < prefix_len implies {
            let slot_a = crate::proof::tensor::geometry::block_table_slot(bt_row_a, pos);
            let slot_c = crate::proof::tensor::geometry::block_table_slot(bt_row_c, pos);
            &&& crate::proof::tensor::geometry::slot_in_cache(cache_a.0, slot_a)
            &&& crate::proof::tensor::geometry::slot_in_cache(cache_a.1, slot_a)
            &&& crate::proof::tensor::geometry::slot_in_cache(cache_c.0, slot_c)
            &&& crate::proof::tensor::geometry::slot_in_cache(cache_c.1, slot_c)
            &&& crate::proof::tensor::geometry::cache_at(cache_a.0, slot_a)
                == crate::proof::tensor::geometry::cache_at(cache_c.0, slot_c)
            &&& crate::proof::tensor::geometry::cache_at(cache_a.1, slot_a)
                == crate::proof::tensor::geometry::cache_at(cache_c.1, slot_c)
        }
    by {}
}

pub proof fn lemma_cache_sequence_logical_prefix_equal_transitive(
    caches_a: Seq<(KVCacheLayerRepr, KVCacheLayerRepr)>,
    bt_row_a: Seq<BlockId>,
    caches_b: Seq<(KVCacheLayerRepr, KVCacheLayerRepr)>,
    bt_row_b: Seq<BlockId>,
    caches_c: Seq<(KVCacheLayerRepr, KVCacheLayerRepr)>,
    bt_row_c: Seq<BlockId>,
    num_layers: nat,
    prefix_len: nat,
)
    requires
        cache_sequence_logical_prefix_equal(
            caches_a, bt_row_a, caches_b, bt_row_b,
            num_layers, prefix_len,
        ),
        cache_sequence_logical_prefix_equal(
            caches_b, bt_row_b, caches_c, bt_row_c,
            num_layers, prefix_len,
        ),
    ensures
        cache_sequence_logical_prefix_equal(
            caches_a, bt_row_a, caches_c, bt_row_c,
            num_layers, prefix_len,
        ),
{
    reveal(cache_sequence_logical_prefix_equal);
    assert forall|layer: int| 0 <= layer < num_layers implies
        #[trigger] cache_pair_logical_prefix_equal(
            caches_a[layer], bt_row_a,
            caches_c[layer], bt_row_c,
            prefix_len,
        )
    by {
        lemma_cache_pair_logical_prefix_equal_transitive(
            caches_a[layer], bt_row_a,
            caches_b[layer], bt_row_b,
            caches_c[layer], bt_row_c,
            prefix_len,
        );
    }
}

// Shared physical-relocation lemma for one layer store.  Family proofs supply
// only equality of the K/V rows being written; all address, prefix framing,
// and read-after-write reasoning lives here.
pub proof fn lemma_relocated_store_logical_prefix_equal(
    kr: Tensor2D,
    vr: Tensor2D,
    old_k_a: KVCacheLayerRepr,
    old_v_a: KVCacheLayerRepr,
    slots_a: Seq<int>,
    bt_row_a: Seq<BlockId>,
    old_k_b: KVCacheLayerRepr,
    old_v_b: KVCacheLayerRepr,
    slots_b: Seq<int>,
    bt_row_b: Seq<BlockId>,
    q_len: nat,
    k_len: nat,
)
    requires
        kr.len() == vr.len(),
        vr.len() == slots_a.len(),
        slots_a.len() == q_len,
        slots_b.len() == q_len,
        q_len <= k_len,
        crate::proof::tensor::geometry::blocks_needed_for(k_len) <= bt_row_a.len(),
        crate::proof::tensor::geometry::blocks_needed_for(k_len) <= bt_row_b.len(),
        forall|j: int| 0 <= j < q_len as int ==>
            crate::proof::tensor::geometry::block_table_slot(
                bt_row_a, (k_len - q_len + j as nat) as nat,
            ) == #[trigger] slots_a[j] as nat
            && crate::proof::tensor::geometry::block_table_slot(
                bt_row_b, (k_len - q_len + j as nat) as nat,
            ) == slots_b[j] as nat,
        forall|j: int| 0 <= j < q_len as int ==>
            #[trigger] slots_a[j] >= 0 && slots_b[j] >= 0,
        forall|j: int, m: int| #![trigger slots_a[m], slots_a[j]]
            0 <= j < q_len as int && j < m < q_len as int ==>
                slots_a[m] != slots_a[j],
        forall|j: int, m: int| #![trigger slots_b[m], slots_b[j]]
            0 <= j < q_len as int && j < m < q_len as int ==>
                slots_b[m] != slots_b[j],
        fresh_writes_miss_cached_prefix(
            slots_a, bt_row_a, slots_b, bt_row_b,
            (k_len - q_len) as int,
        ),
        forall|j: int| 0 <= j < q_len as int ==>
            crate::proof::tensor::geometry::slot_in_cache(
                old_k_a, #[trigger] slots_a[j] as nat,
            )
            && crate::proof::tensor::geometry::slot_in_cache(old_v_a, slots_a[j] as nat)
            && crate::proof::tensor::geometry::slot_in_cache(
                old_k_b, #[trigger] slots_b[j] as nat,
            )
            && crate::proof::tensor::geometry::slot_in_cache(old_v_b, slots_b[j] as nat),
        forall|pos: nat|
            #![trigger crate::proof::tensor::geometry::block_table_slot(bt_row_a, pos)]
            pos < k_len - q_len ==> {
                let slot_a = crate::proof::tensor::geometry::block_table_slot(bt_row_a, pos);
                let slot_b = crate::proof::tensor::geometry::block_table_slot(bt_row_b, pos);
                &&& crate::proof::tensor::geometry::slot_in_cache(old_k_a, slot_a)
                &&& crate::proof::tensor::geometry::slot_in_cache(old_v_a, slot_a)
                &&& crate::proof::tensor::geometry::slot_in_cache(old_k_b, slot_b)
                &&& crate::proof::tensor::geometry::slot_in_cache(old_v_b, slot_b)
                &&& crate::proof::tensor::geometry::cache_at(old_k_a, slot_a)
                    == crate::proof::tensor::geometry::cache_at(old_k_b, slot_b)
                &&& crate::proof::tensor::geometry::cache_at(old_v_a, slot_a)
                    == crate::proof::tensor::geometry::cache_at(old_v_b, slot_b)
            },
    ensures
        cache_pair_logical_prefix_equal(
            crate::boundary::tensor_runtime::store_kv_cache_repr(
                kr, vr, old_k_a, old_v_a, slots_a,
            ),
            bt_row_a,
            crate::boundary::tensor_runtime::store_kv_cache_repr(
                kr, vr, old_k_b, old_v_b, slots_b,
            ),
            bt_row_b,
            k_len,
        ),
{
    let post_a = crate::boundary::tensor_runtime::store_kv_cache_repr(
        kr, vr, old_k_a, old_v_a, slots_a,
    );
    let post_b = crate::boundary::tensor_runtime::store_kv_cache_repr(
        kr, vr, old_k_b, old_v_b, slots_b,
    );
    assert forall|pos: nat|
        #![trigger crate::proof::tensor::geometry::block_table_slot(bt_row_a, pos)]
        pos < k_len implies {
            let slot_a = crate::proof::tensor::geometry::block_table_slot(bt_row_a, pos);
            let slot_b = crate::proof::tensor::geometry::block_table_slot(bt_row_b, pos);
            &&& crate::proof::tensor::geometry::slot_in_cache(post_a.0, slot_a)
            &&& crate::proof::tensor::geometry::slot_in_cache(post_a.1, slot_a)
            &&& crate::proof::tensor::geometry::slot_in_cache(post_b.0, slot_b)
            &&& crate::proof::tensor::geometry::slot_in_cache(post_b.1, slot_b)
            &&& crate::proof::tensor::geometry::cache_at(post_a.0, slot_a)
                == crate::proof::tensor::geometry::cache_at(post_b.0, slot_b)
            &&& crate::proof::tensor::geometry::cache_at(post_a.1, slot_a)
                == crate::proof::tensor::geometry::cache_at(post_b.1, slot_b)
        }
    by {
        crate::proof::tensor::geometry::lemma_blocks_needed_covers_pos(pos, k_len);
        if pos < k_len - q_len {
            fresh_writes_miss_cached_prefix_at(
                slots_a, bt_row_a, slots_b, bt_row_b,
                (k_len - q_len) as int, pos,
            );
            crate::boundary::tensor_runtime::prefix_positions_relocation_agree(
                kr, vr,
                old_k_a, old_v_a, slots_a, bt_row_a,
                old_k_b, old_v_b, slots_b, bt_row_b,
                pos,
            );
        } else {
            let j = (pos - (k_len - q_len)) as int;
            assert((k_len - q_len + j as nat) as nat == pos);
            crate::boundary::tensor_runtime::fresh_positions_relocation_agree(
                kr, vr,
                old_k_a, old_v_a, slots_a, bt_row_a,
                old_k_b, old_v_b, slots_b, bt_row_b,
                q_len, k_len, j,
            );
        }
    }
}

} // verus!
