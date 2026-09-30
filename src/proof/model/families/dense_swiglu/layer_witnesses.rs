//! Dense-SwiGLU discharge of the common per-layer KV-store witness contract.
//!
//! This module owns no family policy. It applies the shared recursive proofs
//! to the exact dense-SwiGLU configuration carried by the admitted model.

use crate::proof::model::dense_swiglu::cache_semantics as CACHE;
#[cfg(verus_only)]
use crate::proof::model::dense_swiglu::layer_store_witnesses as LAYER_IMPL;
#[cfg(verus_only)]
use crate::proof::model::dense_swiglu::relational as RELATIONAL;
use crate::{types::*, proof::model::types::*, proof::tensor::types::*};
use crate::boundary::tensor_runtime as RT;
use vstd::prelude::*;

verus! {

pub open spec fn layer_kv_rows(
    wr: ModelWeightsRepr,
    family: DenseSwiGluDecoderConfigRepr,
    input_ids: IntTensor1D,
    positions: IntTensor1D,
    pre_kv: Seq<(KVCacheLayerRepr, KVCacheLayerRepr)>,
    slots: Seq<int>,
    cu_q: Seq<int>,
    cu_k: Seq<int>,
    max_q: nat,
    max_k: nat,
    block_table: Seq<Seq<BlockId>>,
    layer: nat,
) -> (Tensor2D, Tensor2D) {
    LAYER_IMPL::model_layer_kv_rows(
        dense_swiglu_forward_config_repr(family),
        wr, input_ids, positions, pre_kv, slots,
        cu_q, cu_k, max_q, max_k, block_table, layer,
    )
}

pub proof fn lemma_forward_layer_store(
    wr: ModelWeightsRepr,
    family: DenseSwiGluDecoderConfigRepr,
    input_ids: IntTensor1D,
    positions: IntTensor1D,
    pre_kv: Seq<(KVCacheLayerRepr, KVCacheLayerRepr)>,
    slots: Seq<int>,
    cu_q: Seq<int>,
    cu_k: Seq<int>,
    max_q: nat,
    max_k: nat,
    block_table: Seq<Seq<BlockId>>,
    layer: nat,
)
    requires
        layer < wr.layers.len(),
        pre_kv.len() >= wr.layers.len(),
    ensures ({
        let rows = layer_kv_rows(
            wr, family, input_ids, positions, pre_kv, slots,
            cu_q, cu_k, max_q, max_k, block_table, layer,
        );
        super::forward_kv_reprs(
            wr, family, input_ids, positions, pre_kv, slots,
            cu_q, cu_k, max_q, max_k, block_table,
        )[layer as int] == RT::store_kv_cache_repr(
            rows.0, rows.1,
            pre_kv[layer as int].0, pre_kv[layer as int].1, slots,
        )
    }),
{
    LAYER_IMPL::lemma_model_forward_layer_store(
        dense_swiglu_forward_config_repr(family),
        wr, input_ids, positions, pre_kv, slots,
        cu_q, cu_k, max_q, max_k, block_table, layer,
    );
}

pub proof fn lemma_layer_kv_rows_shape(
    wr: ModelWeightsRepr,
    family: DenseSwiGluDecoderConfigRepr,
    input_ids: IntTensor1D,
    positions: IntTensor1D,
    pre_kv: Seq<(KVCacheLayerRepr, KVCacheLayerRepr)>,
    slots: Seq<int>,
    cu_q: Seq<int>,
    cu_k: Seq<int>,
    max_q: nat,
    max_k: nat,
    block_table: Seq<Seq<BlockId>>,
    layer: nat,
)
    requires
        layer < wr.layers.len(),
        pre_kv.len() >= wr.layers.len(),
        input_ids.len() == positions.len(),
        slots.len() == input_ids.len(),
    ensures ({
        let rows = layer_kv_rows(
            wr, family, input_ids, positions, pre_kv, slots,
            cu_q, cu_k, max_q, max_k, block_table, layer,
        );
        rows.0.len() == input_ids.len()
            && rows.1.len() == input_ids.len()
    }),
{
    LAYER_IMPL::lemma_model_layer_kv_rows_shape(
        dense_swiglu_forward_config_repr(family),
        wr, input_ids, positions, pre_kv, slots,
        cu_q, cu_k, max_q, max_k, block_table, layer,
    );
}

// Whole-model KV relocation derived from the shared physical-store lemma and
// dense-SwiGLU equality of the rows presented to each layer store.
pub proof fn lemma_forward_relocation(
    wr: ModelWeightsRepr,
    family: DenseSwiGluDecoderConfigRepr,
    input_ids: IntTensor1D,
    positions: IntTensor1D,
    caches_a: Seq<(KVCacheLayerRepr, KVCacheLayerRepr)>,
    slots_a: Seq<int>,
    bt_row_a: Seq<BlockId>,
    caches_b: Seq<(KVCacheLayerRepr, KVCacheLayerRepr)>,
    slots_b: Seq<int>,
    bt_row_b: Seq<BlockId>,
    q_len: nat,
    k_len: nat,
)
    requires
        crate::boundary::tensor_runtime::paged_attention_numeric_domain(),
        wr.layers.len() > 0,
        caches_a.len() >= wr.layers.len(),
        caches_b.len() >= wr.layers.len(),
        input_ids.len() == positions.len(),
        slots_a.len() == input_ids.len(),
        slots_b.len() == input_ids.len(),
        input_ids.len() == q_len,
        q_len > 0,
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
        crate::proof::model::family_layout::fresh_writes_miss_cached_prefix(
            slots_a, bt_row_a, slots_b, bt_row_b,
            (k_len - q_len) as int,
        ),
        forall|layer: int, j: int| #![trigger caches_a[layer].0, slots_a[j]]
            0 <= layer < wr.layers.len() && 0 <= j < q_len as int ==>
                crate::proof::tensor::geometry::slot_in_cache(
                    caches_a[layer].0, slots_a[j] as nat,
                )
                && crate::proof::tensor::geometry::slot_in_cache(
                    caches_a[layer].1, slots_a[j] as nat,
                )
                && crate::proof::tensor::geometry::slot_in_cache(
                    caches_b[layer].0, slots_b[j] as nat,
                )
                && crate::proof::tensor::geometry::slot_in_cache(
                    caches_b[layer].1, slots_b[j] as nat,
                ),
        forall|layer: int, pos: nat|
            #![trigger caches_a[layer].0,
                crate::proof::tensor::geometry::block_table_slot(bt_row_a, pos)]
            0 <= layer < wr.layers.len() && pos < k_len - q_len ==> {
                let slot_a = crate::proof::tensor::geometry::block_table_slot(bt_row_a, pos);
                let slot_b = crate::proof::tensor::geometry::block_table_slot(bt_row_b, pos);
                &&& crate::proof::tensor::geometry::slot_in_cache(caches_a[layer].0, slot_a)
                &&& crate::proof::tensor::geometry::slot_in_cache(caches_a[layer].1, slot_a)
                &&& crate::proof::tensor::geometry::slot_in_cache(caches_b[layer].0, slot_b)
                &&& crate::proof::tensor::geometry::slot_in_cache(caches_b[layer].1, slot_b)
                &&& crate::proof::tensor::geometry::cache_at(caches_a[layer].0, slot_a)
                    == crate::proof::tensor::geometry::cache_at(caches_b[layer].0, slot_b)
                &&& crate::proof::tensor::geometry::cache_at(caches_a[layer].1, slot_a)
                    == crate::proof::tensor::geometry::cache_at(caches_b[layer].1, slot_b)
            },
    ensures
        super::forward_logits_repr(
            wr, family, input_ids, positions,
            caches_a, slots_a,
            seq![0int, q_len as int], seq![0int, k_len as int],
            q_len, k_len, seq![bt_row_a],
        ) == super::forward_logits_repr(
            wr, family, input_ids, positions,
            caches_b, slots_b,
            seq![0int, q_len as int], seq![0int, k_len as int],
            q_len, k_len, seq![bt_row_b],
        ),
        crate::proof::model::family_layout::cache_sequence_logical_prefix_equal(
            super::forward_kv_reprs(
                wr, family, input_ids, positions, caches_a, slots_a,
                seq![0int, q_len as int], seq![0int, k_len as int],
                q_len, k_len, seq![bt_row_a],
            ),
            bt_row_a,
            super::forward_kv_reprs(
                wr, family, input_ids, positions, caches_b, slots_b,
                seq![0int, q_len as int], seq![0int, k_len as int],
                q_len, k_len, seq![bt_row_b],
            ),
            bt_row_b,
            wr.layers.len(),
            k_len,
        ),
{
    reveal(crate::proof::model::family_layout::fresh_writes_miss_cached_prefix);
    reveal(RELATIONAL::fresh_writes_miss_cached_prefix);
    RELATIONAL::model_forward_relocation(dense_swiglu_forward_config_repr(family),
        wr, input_ids, positions,
        caches_a, slots_a, bt_row_a,
        caches_b, slots_b, bt_row_b,
        q_len, k_len,
    );
    let cu_q = seq![0int, q_len as int];
    let cu_k = seq![0int, k_len as int];
    let bt_a = seq![bt_row_a];
    let bt_b = seq![bt_row_b];
    let post_a = super::forward_kv_reprs(
        wr, family, input_ids, positions, caches_a, slots_a,
        cu_q, cu_k, q_len, k_len, bt_a,
    );
    let post_b = super::forward_kv_reprs(
        wr, family, input_ids, positions, caches_b, slots_b,
        cu_q, cu_k, q_len, k_len, bt_b,
    );
    CACHE::lemma_model_forward_kv_reprs_len(dense_swiglu_forward_config_repr(family),
        wr, input_ids, positions, caches_a, slots_a,
        cu_q, cu_k, q_len, k_len, bt_a,
    );
    CACHE::lemma_model_forward_kv_reprs_len(dense_swiglu_forward_config_repr(family),
        wr, input_ids, positions, caches_b, slots_b,
        cu_q, cu_k, q_len, k_len, bt_b,
    );
    assert forall|layer: int| 0 <= layer < wr.layers.len() implies
        #[trigger] crate::proof::model::family_layout::cache_pair_logical_prefix_equal(
            post_a[layer], bt_row_a, post_b[layer], bt_row_b, k_len,
        )
    by {
        let rows_a = layer_kv_rows(
            wr, family, input_ids, positions, caches_a, slots_a,
            cu_q, cu_k, q_len, k_len, bt_a, layer as nat,
        );
        let rows_b = layer_kv_rows(
            wr, family, input_ids, positions, caches_b, slots_b,
            cu_q, cu_k, q_len, k_len, bt_b, layer as nat,
        );
        lemma_layer_kv_rows_shape(
            wr, family, input_ids, positions, caches_a, slots_a,
            cu_q, cu_k, q_len, k_len, bt_a, layer as nat,
        );
        reveal(crate::proof::model::family_layout::fresh_writes_miss_cached_prefix);
        LAYER_IMPL::lemma_model_layer_kv_rows_relocation(
            dense_swiglu_forward_config_repr(family),
            wr, input_ids, positions,
            caches_a, slots_a, bt_row_a,
            caches_b, slots_b, bt_row_b,
            q_len, k_len, layer as nat,
        );
        assert(rows_a == rows_b);
        lemma_forward_layer_store(
            wr, family, input_ids, positions, caches_a, slots_a,
            cu_q, cu_k, q_len, k_len, bt_a, layer as nat,
        );
        lemma_forward_layer_store(
            wr, family, input_ids, positions, caches_b, slots_b,
            cu_q, cu_k, q_len, k_len, bt_b, layer as nat,
        );
        assert forall|j: int| 0 <= j < q_len as int implies
            crate::proof::tensor::geometry::block_table_slot(
                bt_row_a, (k_len - q_len + j as nat) as nat,
            ) == #[trigger] slots_a[j] as nat
            && crate::proof::tensor::geometry::block_table_slot(
                bt_row_b, (k_len - q_len + j as nat) as nat,
            ) == slots_b[j] as nat by {}
        crate::proof::model::family_layout::lemma_relocated_store_logical_prefix_equal(
            rows_a.0, rows_a.1,
            caches_a[layer].0, caches_a[layer].1,
            slots_a, bt_row_a,
            caches_b[layer].0, caches_b[layer].1,
            slots_b, bt_row_b,
            q_len, k_len,
        );
    }
    reveal(crate::proof::model::family_layout::cache_sequence_logical_prefix_equal);
}

// Family discharge of the common pre-store row-isolation contract used by
// architecture-neutral graph covering and cache projection.
pub proof fn lemma_layer_kv_rows_request_isolation(
    wr: ModelWeightsRepr,
    family: DenseSwiGluDecoderConfigRepr,
    input_ids: IntTensor1D,
    positions: IntTensor1D,
    pre_kv: Seq<(KVCacheLayerRepr, KVCacheLayerRepr)>,
    slots: Seq<int>,
    cu_q: Seq<int>,
    cu_k: Seq<int>,
    max_q: nat,
    max_k: nat,
    block_table: Seq<Seq<BlockId>>,
    i: nat,
    layer: nat,
)
    requires
        super::request_projection_ready(
            wr, family, input_ids, positions, pre_kv, slots,
            cu_q, cu_k, max_q, max_k, block_table, i,
        ),
        layer < wr.layers.len(),
    ensures ({
        let lo = cu_q[i as int];
        let hi = cu_q[i as int + 1];
        let q_len = (hi - lo) as nat;
        let k_len = (cu_k[i as int + 1] - cu_k[i as int]) as nat;
        let full = layer_kv_rows(
            wr, family, input_ids, positions, pre_kv, slots,
            cu_q, cu_k, max_q, max_k, block_table, layer,
        );
        let single = layer_kv_rows(
            wr, family, input_ids.subrange(lo, hi),
            positions.subrange(lo, hi), pre_kv, slots.subrange(lo, hi),
            seq![0int, q_len as int], seq![0int, k_len as int],
            q_len, k_len, seq![block_table[i as int]], layer,
        );
        &&& full.0.subrange(lo, hi) == single.0
        &&& full.1.subrange(lo, hi) == single.1
    }),
{
    reveal(super::request_projection_ready);
    LAYER_IMPL::lemma_model_layer_kv_rows_request_isolation(
        dense_swiglu_forward_config_repr(family),
        wr, input_ids, positions, pre_kv, slots,
        cu_q, cu_k, max_q, max_k, block_table,
        slots.subrange(cu_q[i as int], cu_q[i as int + 1]), i, layer,
    );
}

// Batched-to-singleton KV projection for one request.  The family proof shows
// that the selected K/V rows are equal; the generic scatter-store theorem
// frames writes belonging to all other requests.
pub proof fn lemma_kv_request_isolation(
    wr: ModelWeightsRepr,
    family: DenseSwiGluDecoderConfigRepr,
    input_ids: IntTensor1D,
    positions: IntTensor1D,
    pre_kv: Seq<(KVCacheLayerRepr, KVCacheLayerRepr)>,
    slots: Seq<int>,
    cu_q: Seq<int>,
    cu_k: Seq<int>,
    max_q: nat,
    max_k: nat,
    block_table: Seq<Seq<BlockId>>,
    i: nat,
)
    requires
        super::request_projection_ready(
            wr, family, input_ids, positions, pre_kv, slots,
            cu_q, cu_k, max_q, max_k, block_table, i,
        ),
    ensures ({
        let lo = cu_q[i as int];
        let hi = cu_q[i as int + 1];
        let q_len = (hi - lo) as nat;
        let k_len = (cu_k[i as int + 1] - cu_k[i as int]) as nat;
        crate::proof::model::family_layout::cache_sequence_logical_prefix_equal(
            super::forward_kv_reprs(
                wr, family, input_ids, positions, pre_kv, slots,
                cu_q, cu_k, max_q, max_k, block_table,
            ),
            block_table[i as int],
            super::forward_kv_reprs(
                wr, family,
                input_ids.subrange(lo, hi),
                positions.subrange(lo, hi),
                pre_kv,
                slots.subrange(lo, hi),
                seq![0int, q_len as int],
                seq![0int, k_len as int],
                q_len,
                k_len,
                seq![block_table[i as int]],
            ),
            block_table[i as int],
            wr.layers.len(),
            k_len,
        )
    }),
{
    reveal(super::request_projection_ready);
    let lo = cu_q[i as int];
    let hi = cu_q[i as int + 1];
    let q_len = (hi - lo) as nat;
    let k_len = (cu_k[i as int + 1] - cu_k[i as int]) as nat;
    let own_slots = slots.subrange(lo, hi);
    let single_cu_q = seq![0int, q_len as int];
    let single_cu_k = seq![0int, k_len as int];
    let single_bt = seq![block_table[i as int]];
    let full_post = super::forward_kv_reprs(
        wr, family, input_ids, positions, pre_kv, slots,
        cu_q, cu_k, max_q, max_k, block_table,
    );
    let single_post = super::forward_kv_reprs(
        wr, family, input_ids.subrange(lo, hi),
        positions.subrange(lo, hi), pre_kv, own_slots,
        single_cu_q, single_cu_k, q_len, k_len, single_bt,
    );
    CACHE::lemma_model_forward_kv_reprs_len(dense_swiglu_forward_config_repr(family),
        wr, input_ids, positions, pre_kv, slots,
        cu_q, cu_k, max_q, max_k, block_table,
    );
    CACHE::lemma_model_forward_kv_reprs_len(dense_swiglu_forward_config_repr(family),
        wr, input_ids.subrange(lo, hi),
        positions.subrange(lo, hi), pre_kv, own_slots,
        single_cu_q, single_cu_k, q_len, k_len, single_bt,
    );
    assert forall|layer: int| 0 <= layer < wr.layers.len() implies
        #[trigger] crate::proof::model::family_layout::cache_pair_logical_prefix_equal(
            full_post[layer], block_table[i as int],
            single_post[layer], block_table[i as int], k_len,
        )
    by {
        let full_rows = layer_kv_rows(
            wr, family, input_ids, positions, pre_kv, slots,
            cu_q, cu_k, max_q, max_k, block_table, layer as nat,
        );
        let single_rows = layer_kv_rows(
            wr, family, input_ids.subrange(lo, hi),
            positions.subrange(lo, hi), pre_kv, own_slots,
            single_cu_q, single_cu_k, q_len, k_len, single_bt,
            layer as nat,
        );
        lemma_layer_kv_rows_shape(
            wr, family, input_ids, positions, pre_kv, slots,
            cu_q, cu_k, max_q, max_k, block_table, layer as nat,
        );
        lemma_layer_kv_rows_request_isolation(
            wr, family, input_ids, positions, pre_kv, slots,
            cu_q, cu_k, max_q, max_k, block_table, i, layer as nat,
        );
        assert(full_rows.0.subrange(lo, hi) == single_rows.0);
        assert(full_rows.1.subrange(lo, hi) == single_rows.1);
        crate::proof::tensor::seq_flatten::lemma_seq_split3(full_rows.0, lo, hi);
        crate::proof::tensor::seq_flatten::lemma_seq_split3(full_rows.1, lo, hi);
        crate::proof::tensor::seq_flatten::lemma_seq_split3(slots, lo, hi);
        crate::boundary::tensor_runtime::store_agrees_at_all_block_pos(
            full_rows.0.subrange(0, lo),
            full_rows.1.subrange(0, lo),
            full_rows.0.subrange(lo, hi),
            full_rows.1.subrange(lo, hi),
            full_rows.0.subrange(hi, full_rows.0.len() as int),
            full_rows.1.subrange(hi, full_rows.1.len() as int),
            pre_kv[layer].0,
            pre_kv[layer].1,
            slots.subrange(0, lo),
            own_slots,
            slots.subrange(hi, slots.len() as int),
            block_table[i as int],
            k_len,
        );
        lemma_forward_layer_store(
            wr, family, input_ids, positions, pre_kv, slots,
            cu_q, cu_k, max_q, max_k, block_table, layer as nat,
        );
        lemma_forward_layer_store(
            wr, family, input_ids.subrange(lo, hi),
            positions.subrange(lo, hi), pre_kv, own_slots,
            single_cu_q, single_cu_k, q_len, k_len, single_bt,
            layer as nat,
        );
        let full_store = crate::boundary::tensor_runtime::store_kv_cache_repr(
            full_rows.0, full_rows.1,
            pre_kv[layer].0, pre_kv[layer].1, slots,
        );
        let own_store = crate::boundary::tensor_runtime::store_kv_cache_repr(
            full_rows.0.subrange(lo, hi),
            full_rows.1.subrange(lo, hi),
            pre_kv[layer].0, pre_kv[layer].1, own_slots,
        );
        let single_store = crate::boundary::tensor_runtime::store_kv_cache_repr(
            single_rows.0, single_rows.1,
            pre_kv[layer].0, pre_kv[layer].1, own_slots,
        );
        assert(full_post[layer] == full_store);
        assert(single_post[layer] == single_store);
        assert(own_store == single_store);
        reveal(crate::proof::model::family_layout::cache_pair_logical_prefix_equal);
        assert forall|pos: nat|
            #![trigger crate::proof::tensor::geometry::block_table_slot(
                block_table[i as int], pos,
            )]
            pos < k_len implies {
                let slot = crate::proof::tensor::geometry::block_table_slot(
                    block_table[i as int], pos,
                );
                &&& crate::proof::tensor::geometry::slot_in_cache(full_post[layer].0, slot)
                &&& crate::proof::tensor::geometry::slot_in_cache(full_post[layer].1, slot)
                &&& crate::proof::tensor::geometry::slot_in_cache(single_post[layer].0, slot)
                &&& crate::proof::tensor::geometry::slot_in_cache(single_post[layer].1, slot)
                &&& crate::proof::tensor::geometry::cache_at(full_post[layer].0, slot)
                    == crate::proof::tensor::geometry::cache_at(single_post[layer].0, slot)
                &&& crate::proof::tensor::geometry::cache_at(full_post[layer].1, slot)
                    == crate::proof::tensor::geometry::cache_at(single_post[layer].1, slot)
            }
        by {
            let slot = crate::proof::tensor::geometry::block_table_slot(
                block_table[i as int], pos,
            );
            crate::boundary::tensor_runtime::store_kv_cache_repr_preserves_slot_in_cache(
                full_rows.0.subrange(lo, hi),
                full_rows.1.subrange(lo, hi),
                pre_kv[layer].0,
                pre_kv[layer].1,
                own_slots,
                slot,
            );
        }
    }
    reveal(crate::proof::model::family_layout::cache_sequence_logical_prefix_equal);
}

} // verus!
