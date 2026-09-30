//! Shared four-norm decoder KV-store witnesses and relational fold proofs.
//!
//! K/V rows are the actual post-normalization/RoPE values presented to the
//! scatter. Per-layer geometry and row policies are immutable inputs; full
//! and sliding attention retain the same conservative cache contract.

#[cfg(verus_only)]
use crate::proof::model::four_norm_gated::batch_invariance as BI;
#[cfg(verus_only)]
use crate::proof::model::four_norm_gated::relocation as RELOCATION;
#[cfg(verus_only)]
use crate::proof::model::four_norm_gated::model as MODEL;
#[cfg(verus_only)]
use crate::boundary::backend_certificates::support as GAC;
use crate::{types::*, proof::model::types::*, proof::tensor::types::*};
use crate::boundary::tensor_runtime as RT;
use vstd::prelude::*;

verus! {

pub open spec fn chain_layer_kv_rows(
    common_layers: Seq<LayerWeightsRepr>,
    extension_layers: Seq<FourNormGatedLayerExtensionRepr>,
    hidden: Tensor2D,
    positions: IntTensor1D,
    caches: Seq<(KVCacheLayerRepr, KVCacheLayerRepr)>,
    slots: Seq<int>,
    cu_q: Seq<int>,
    cu_k: Seq<int>,
    max_q: nat,
    max_k: nat,
    block_table: Seq<Seq<BlockId>>,
    start: nat,
    layer: nat,
) -> (Tensor2D, Tensor2D)
    recommends
        start <= layer < common_layers.len(),
        extension_layers.len() == common_layers.len(),
        caches.len() >= common_layers.len(),
    decreases layer - start,
{
    if layer <= start {
        let pre = MODEL::attention_pre_store_repr(
            common_layers[start as int],
            extension_layers[start as int],
            hidden,
            positions,
        );
        (pre.1, pre.2)
    } else {
        let step = MODEL::decoder_layer_step_repr(
            common_layers[start as int],
            extension_layers[start as int],
            hidden,
            positions,
            caches[start as int].0,
            caches[start as int].1,
            slots,
            cu_q,
            cu_k,
            max_q,
            max_k,
            block_table,
        );
        chain_layer_kv_rows(
            common_layers,
            extension_layers,
            step.0,
            positions,
            caches.update(start as int, step.1),
            slots,
            cu_q,
            cu_k,
            max_q,
            max_k,
            block_table,
            (start + 1) as nat,
            layer,
        )
    }
}

pub open spec fn layer_kv_rows(
    wr: ModelWeightsRepr,
    family: FourNormGatedDecoderConfigRepr,
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
) -> (Tensor2D, Tensor2D)
    recommends
        family.layers.len() == wr.layers.len(),
        layer < wr.layers.len(),
        pre_kv.len() >= wr.layers.len(),
{
    chain_layer_kv_rows(
        wr.layers,
        family.layers,
        MODEL::scaled_embed_repr(
            input_ids, wr.embed_weight, family.geometry.hidden_size,
        ),
        positions,
        pre_kv,
        slots,
        cu_q,
        cu_k,
        max_q,
        max_k,
        block_table,
        0,
        layer,
    )
}

proof fn lemma_chain_layer_store(
    common_layers: Seq<LayerWeightsRepr>,
    extension_layers: Seq<FourNormGatedLayerExtensionRepr>,
    hidden: Tensor2D,
    positions: IntTensor1D,
    caches: Seq<(KVCacheLayerRepr, KVCacheLayerRepr)>,
    slots: Seq<int>,
    cu_q: Seq<int>,
    cu_k: Seq<int>,
    max_q: nat,
    max_k: nat,
    block_table: Seq<Seq<BlockId>>,
    start: nat,
    layer: nat,
)
    requires
        start <= layer < common_layers.len(),
        extension_layers.len() == common_layers.len(),
        caches.len() >= common_layers.len(),
        hidden.len() == positions.len(),
        hidden.len() == slots.len(),
    ensures ({
        let rows = chain_layer_kv_rows(
            common_layers, extension_layers, hidden, positions,
            caches, slots, cu_q, cu_k, max_q, max_k, block_table,
            start, layer,
        );
        MODEL::layer_chain_repr(
            common_layers, extension_layers, hidden, positions,
            caches, slots, cu_q, cu_k, max_q, max_k, block_table, start,
        ).1[layer as int] == RT::store_kv_cache_repr(
            rows.0, rows.1,
            caches[layer as int].0, caches[layer as int].1, slots,
        )
    }),
    decreases layer - start,
{
    let current = start as int;
    let step = MODEL::decoder_layer_step_repr(
        common_layers[current], extension_layers[current], hidden,
        positions, caches[current].0, caches[current].1, slots,
        cu_q, cu_k, max_q, max_k, block_table,
    );
    let next_caches = caches.update(current, step.1);
    MODEL::lemma_decoder_layer_step_repr_shape(
        common_layers[current], extension_layers[current], hidden,
        positions, caches[current].0, caches[current].1, slots,
        cu_q, cu_k, max_q, max_k, block_table,
    );
    if layer <= start {
        assert(layer == start);
        MODEL::lemma_layer_chain_cache_before_start_unchanged(
            common_layers,
            extension_layers,
            step.0,
            positions,
            next_caches,
            slots,
            cu_q,
            cu_k,
            max_q,
            max_k,
            block_table,
            (start + 1) as nat,
            layer as int,
        );
        let pre = MODEL::attention_pre_store_repr(
            common_layers[current], extension_layers[current],
            hidden, positions,
        );
        assert(step.1 == RT::store_kv_cache_repr(
            pre.1, pre.2,
            caches[current].0, caches[current].1, slots,
        ));
    } else {
        assert((start + 1) as nat <= layer);
        lemma_chain_layer_store(
            common_layers,
            extension_layers,
            step.0,
            positions,
            next_caches,
            slots,
            cu_q,
            cu_k,
            max_q,
            max_k,
            block_table,
            (start + 1) as nat,
            layer,
        );
        assert(next_caches[layer as int] == caches[layer as int]);
    }
}

proof fn lemma_chain_layer_kv_rows_shape(
    common_layers: Seq<LayerWeightsRepr>,
    extension_layers: Seq<FourNormGatedLayerExtensionRepr>,
    hidden: Tensor2D,
    positions: IntTensor1D,
    caches: Seq<(KVCacheLayerRepr, KVCacheLayerRepr)>,
    slots: Seq<int>,
    cu_q: Seq<int>,
    cu_k: Seq<int>,
    max_q: nat,
    max_k: nat,
    block_table: Seq<Seq<BlockId>>,
    start: nat,
    layer: nat,
)
    requires
        start <= layer < common_layers.len(),
        extension_layers.len() == common_layers.len(),
        caches.len() >= common_layers.len(),
        hidden.len() == positions.len(),
        hidden.len() == slots.len(),
    ensures ({
        let rows = chain_layer_kv_rows(
            common_layers, extension_layers, hidden, positions,
            caches, slots, cu_q, cu_k, max_q, max_k, block_table,
            start, layer,
        );
        rows.0.len() == hidden.len() && rows.1.len() == hidden.len()
    }),
    decreases layer - start,
{
    let current = start as int;
    if layer <= start {
        MODEL::lemma_attention_pre_store_repr_shape(
            common_layers[current], extension_layers[current],
            hidden, positions,
        );
    } else {
        let step = MODEL::decoder_layer_step_repr(
            common_layers[current], extension_layers[current], hidden,
            positions, caches[current].0, caches[current].1, slots,
            cu_q, cu_k, max_q, max_k, block_table,
        );
        MODEL::lemma_decoder_layer_step_repr_shape(
            common_layers[current], extension_layers[current], hidden,
            positions, caches[current].0, caches[current].1, slots,
            cu_q, cu_k, max_q, max_k, block_table,
        );
        lemma_chain_layer_kv_rows_shape(
            common_layers,
            extension_layers,
            step.0,
            positions,
            caches.update(current, step.1),
            slots,
            cu_q,
            cu_k,
            max_q,
            max_k,
            block_table,
            (start + 1) as nat,
            layer,
        );
    }
}

pub proof fn lemma_forward_layer_store(
    wr: ModelWeightsRepr,
    family: FourNormGatedDecoderConfigRepr,
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
        family.layers.len() == wr.layers.len(),
        layer < wr.layers.len(),
        pre_kv.len() >= wr.layers.len(),
        input_ids.len() == positions.len(),
        slots.len() == input_ids.len(),
    ensures ({
        let rows = layer_kv_rows(
            wr, family, input_ids, positions, pre_kv, slots,
            cu_q, cu_k, max_q, max_k, block_table, layer,
        );
        MODEL::model_forward_kv_reprs(
            wr, family, input_ids, positions, pre_kv, slots,
            cu_q, cu_k, max_q, max_k, block_table,
        )[layer as int] == RT::store_kv_cache_repr(
            rows.0, rows.1,
            pre_kv[layer as int].0, pre_kv[layer as int].1, slots,
        )
    }),
{
    let hidden = MODEL::scaled_embed_repr(
        input_ids, wr.embed_weight, family.geometry.hidden_size,
    );
    MODEL::lemma_scaled_embed_repr_shape(
        input_ids, wr.embed_weight, family.geometry.hidden_size,
    );
    lemma_chain_layer_store(
        wr.layers, family.layers, hidden, positions, pre_kv, slots,
        cu_q, cu_k, max_q, max_k, block_table, 0, layer,
    );
    reveal(MODEL::model_forward_kv_reprs);
    reveal(MODEL::model_forward_hidden_and_kv_reprs);
}

pub proof fn lemma_layer_kv_rows_shape(
    wr: ModelWeightsRepr,
    family: FourNormGatedDecoderConfigRepr,
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
        family.layers.len() == wr.layers.len(),
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
    let hidden = MODEL::scaled_embed_repr(
        input_ids, wr.embed_weight, family.geometry.hidden_size,
    );
    MODEL::lemma_scaled_embed_repr_shape(
        input_ids, wr.embed_weight, family.geometry.hidden_size,
    );
    lemma_chain_layer_kv_rows_shape(
        wr.layers, family.layers, hidden, positions, pre_kv, slots,
        cu_q, cu_k, max_q, max_k, block_table, 0, layer,
    );
}

// Whole-model KV relocation is the family-local discharge of the shared
// physical-layout contract.  The underlying four-norm proof treats full and
// sliding attention uniformly through each layer's attention configuration.
pub proof fn lemma_forward_relocation(
    wr: ModelWeightsRepr,
    family: FourNormGatedDecoderConfigRepr,
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
        family.layers.len() == wr.layers.len(),
        forall|layer: int| 0 <= layer < family.layers.len() ==>
            layer_attention_config_valid(
                #[trigger] family.layers[layer].attention,
            ),
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
        MODEL::model_forward_logits_repr(
            wr, family, input_ids, positions,
            caches_a, slots_a,
            seq![0int, q_len as int], seq![0int, k_len as int],
            q_len, k_len, seq![bt_row_a],
        ) == MODEL::model_forward_logits_repr(
            wr, family, input_ids, positions,
            caches_b, slots_b,
            seq![0int, q_len as int], seq![0int, k_len as int],
            q_len, k_len, seq![bt_row_b],
        ),
        crate::proof::model::family_layout::cache_sequence_logical_prefix_equal(
            MODEL::model_forward_kv_reprs(
                wr, family, input_ids, positions, caches_a, slots_a,
                seq![0int, q_len as int], seq![0int, k_len as int],
                q_len, k_len, seq![bt_row_a],
            ),
            bt_row_a,
            MODEL::model_forward_kv_reprs(
                wr, family, input_ids, positions, caches_b, slots_b,
                seq![0int, q_len as int], seq![0int, k_len as int],
                q_len, k_len, seq![bt_row_b],
            ),
            bt_row_b,
            wr.layers.len(),
            k_len,
        ),
{
    // Forward the supplied cache-bounds facts without expanding their
    // division/remainder arithmetic in this structural adapter.
    hide(crate::proof::tensor::geometry::slot_in_cache);
    // Instantiate the input slot correspondence explicitly at this module
    // boundary; do not rely on unrelated declarations to trigger the premise.
    assert forall|j: int| 0 <= j < q_len as int implies
        crate::proof::tensor::geometry::block_table_slot(
            bt_row_a, (k_len - q_len + j as nat) as nat,
        ) == #[trigger] slots_a[j] as nat
        && crate::proof::tensor::geometry::block_table_slot(
            bt_row_b, (k_len - q_len + j as nat) as nat,
        ) == slots_b[j] as nat by {
        assert(slots_a[j] >= 0);
        assert(crate::proof::tensor::geometry::block_table_slot(
            bt_row_a, (k_len - q_len + j as nat) as nat,
        ) == slots_a[j] as nat);
    }
    assert forall|layer: int, j: int| #![trigger caches_a[layer].0, slots_a[j]]
        0 <= layer < wr.layers.len() && 0 <= j < q_len as int implies
            crate::proof::tensor::geometry::slot_in_cache(caches_a[layer].0, slots_a[j] as nat)
            && crate::proof::tensor::geometry::slot_in_cache(caches_a[layer].1, slots_a[j] as nat)
            && crate::proof::tensor::geometry::slot_in_cache(caches_b[layer].0, slots_b[j] as nat)
            && crate::proof::tensor::geometry::slot_in_cache(caches_b[layer].1, slots_b[j] as nat) by {
        assert(crate::proof::tensor::geometry::slot_in_cache(caches_a[layer].0, slots_a[j] as nat));
        assert(crate::proof::tensor::geometry::slot_in_cache(caches_a[layer].1, slots_a[j] as nat));
        assert(crate::proof::tensor::geometry::slot_in_cache(caches_b[layer].0, slots_b[j] as nat));
        assert(crate::proof::tensor::geometry::slot_in_cache(caches_b[layer].1, slots_b[j] as nat));
    }
    RELOCATION::model_forward_relocation_from_layout(
        wr, family, input_ids, positions,
        caches_a, slots_a, bt_row_a,
        caches_b, slots_b, bt_row_b,
        q_len, k_len,
    );
}

#[verifier::spinoff_prover]
proof fn lemma_chain_layer_kv_rows_request_isolation(
    common_layers: Seq<LayerWeightsRepr>,
    extension_layers: Seq<FourNormGatedLayerExtensionRepr>,
    full_hidden: Tensor2D,
    single_hidden: Tensor2D,
    positions: IntTensor1D,
    full_caches: Seq<(KVCacheLayerRepr, KVCacheLayerRepr)>,
    single_caches: Seq<(KVCacheLayerRepr, KVCacheLayerRepr)>,
    slots: Seq<int>,
    cu_q: Seq<int>,
    cu_k: Seq<int>,
    max_q: nat,
    max_k: nat,
    block_table: Seq<Seq<BlockId>>,
    i: nat,
    start: nat,
    layer: nat,
)
    requires
        start <= layer < common_layers.len(),
        extension_layers.len() == common_layers.len(),
        full_caches.len() >= common_layers.len(),
        single_caches.len() >= common_layers.len(),
        full_hidden.len() == positions.len(),
        full_hidden.len() == slots.len(),
        i < block_table.len(),
        0 <= cu_q[i as int] < cu_q[i as int + 1]
            <= full_hidden.len() as int,
        single_hidden == full_hidden.subrange(
            cu_q[i as int], cu_q[i as int + 1],
        ),
        forall|j: int| start <= j < common_layers.len() ==>
            #[trigger] single_caches[j] == full_caches[j],
        BI::layer_chain_attention_launch_ready(
            common_layers, extension_layers, full_hidden, positions,
            full_caches, slots, cu_q, cu_k, max_q, max_k,
            block_table, start,
        ),
        forall|m: int, l: int|
            #![trigger slots.subrange(0, cu_q[i as int])[m]
                / (crate::types::BLOCK_SIZE_SPEC as int),
                block_table[i as int][l]]
            0 <= m < cu_q[i as int]
                && 0 <= l < block_table[i as int].len() ==>
                slots.subrange(0, cu_q[i as int])[m]
                    / (crate::types::BLOCK_SIZE_SPEC as int)
                    != block_table[i as int][l] as int,
        forall|m: int, l: int|
            #![trigger slots.subrange(
                cu_q[i as int + 1], slots.len() as int,
            )[m] / (crate::types::BLOCK_SIZE_SPEC as int),
                block_table[i as int][l]]
            0 <= m < slots.len() - cu_q[i as int + 1]
                && 0 <= l < block_table[i as int].len() ==>
                slots.subrange(
                    cu_q[i as int + 1], slots.len() as int,
                )[m] / (crate::types::BLOCK_SIZE_SPEC as int)
                    != block_table[i as int][l] as int,
        forall|ell: int, pos: nat|
            #![trigger full_caches[ell].0,
                crate::proof::tensor::geometry::block_table_slot(block_table[i as int], pos)]
            start <= ell < common_layers.len()
                && pos < crate::proof::tensor::geometry::blocks_needed_for(
                    (cu_k[i as int + 1] - cu_k[i as int]) as nat,
                ) * crate::types::BLOCK_SIZE_SPEC ==>
                crate::proof::tensor::geometry::slot_in_cache(
                    full_caches[ell].0,
                    crate::proof::tensor::geometry::block_table_slot(block_table[i as int], pos),
                )
                && crate::proof::tensor::geometry::slot_in_cache(
                    full_caches[ell].1,
                    crate::proof::tensor::geometry::block_table_slot(block_table[i as int], pos),
                ),
    ensures ({
        let lo = cu_q[i as int];
        let hi = cu_q[i as int + 1];
        let q_len = (hi - lo) as nat;
        let k_len = (cu_k[i as int + 1] - cu_k[i as int]) as nat;
        let full = chain_layer_kv_rows(
            common_layers, extension_layers, full_hidden, positions,
            full_caches, slots, cu_q, cu_k, max_q, max_k,
            block_table, start, layer,
        );
        let single = chain_layer_kv_rows(
            common_layers, extension_layers, single_hidden,
            positions.subrange(lo, hi), single_caches,
            slots.subrange(lo, hi), seq![0int, q_len as int],
            seq![0int, k_len as int], q_len, k_len,
            seq![block_table[i as int]], start, layer,
        );
        &&& full.0.subrange(lo, hi) == single.0
        &&& full.1.subrange(lo, hi) == single.1
    }),
    decreases layer - start,
{
    let lo = cu_q[i as int];
    let hi = cu_q[i as int + 1];
    let q_len = (hi - lo) as nat;
    let k_len = (cu_k[i as int + 1] - cu_k[i as int]) as nat;
    let current = start as int;
    if layer <= start {
        assert(single_caches[current] == full_caches[current]);
        BI::attention_pre_store_subrange_invariance(
            common_layers[current], extension_layers[current],
            full_hidden, positions, lo, hi,
        );
    } else {
        reveal(BI::layer_chain_attention_launch_ready);
        assert(single_caches[current] == full_caches[current]);
        BI::decoder_layer_request_isolation_from_layout(
            common_layers[current], extension_layers[current],
            full_hidden, positions,
            full_caches[current].0, full_caches[current].1,
            slots, cu_q, cu_k, max_q, max_k, block_table, i,
        );
        let full_step = MODEL::decoder_layer_step_repr(
            common_layers[current], extension_layers[current],
            full_hidden, positions,
            full_caches[current].0, full_caches[current].1,
            slots, cu_q, cu_k, max_q, max_k, block_table,
        );
        let single_step = MODEL::decoder_layer_step_repr(
            common_layers[current], extension_layers[current],
            single_hidden, positions.subrange(lo, hi),
            single_caches[current].0, single_caches[current].1,
            slots.subrange(lo, hi), seq![0int, q_len as int],
            seq![0int, k_len as int], q_len, k_len,
            seq![block_table[i as int]],
        );
        MODEL::lemma_decoder_layer_step_repr_shape(
            common_layers[current], extension_layers[current],
            full_hidden, positions,
            full_caches[current].0, full_caches[current].1,
            slots, cu_q, cu_k, max_q, max_k, block_table,
        );
        let next_full = full_caches.update(current, full_step.1);
        let next_single = single_caches.update(current, single_step.1);
        assert(single_step.0 == full_step.0.subrange(lo, hi));
        assert forall|j: int| (start + 1) as int <= j < common_layers.len()
            implies #[trigger] next_single[j] == next_full[j]
        by {
            assert(j != current);
        }
        assert forall|ell: int, pos: nat|
            #![trigger next_full[ell].0,
                crate::proof::tensor::geometry::block_table_slot(block_table[i as int], pos)]
            (start + 1) as int <= ell < common_layers.len()
                && pos < crate::proof::tensor::geometry::blocks_needed_for(k_len)
                    * crate::types::BLOCK_SIZE_SPEC
            implies
                crate::proof::tensor::geometry::slot_in_cache(
                    next_full[ell].0,
                    crate::proof::tensor::geometry::block_table_slot(block_table[i as int], pos),
                )
                && crate::proof::tensor::geometry::slot_in_cache(
                    next_full[ell].1,
                    crate::proof::tensor::geometry::block_table_slot(block_table[i as int], pos),
                )
        by {
            assert(ell != current);
        }
        lemma_chain_layer_kv_rows_request_isolation(
            common_layers, extension_layers,
            full_step.0, single_step.0, positions,
            next_full, next_single, slots,
            cu_q, cu_k, max_q, max_k, block_table, i,
            (start + 1) as nat, layer,
        );
    }
}

// Family discharge of the common pre-store row-isolation contract. Both
// four-norm attention variants share the same row-local K/V projection; attention
// configuration is used only while advancing through preceding layers.
pub proof fn lemma_layer_kv_rows_request_isolation(
    wr: ModelWeightsRepr,
    family: FourNormGatedDecoderConfigRepr,
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
        BI::request_projection_ready(
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
    reveal(BI::request_projection_ready);
    let hidden = MODEL::scaled_embed_repr(
        input_ids, wr.embed_weight, family.geometry.hidden_size,
    );
    let lo = cu_q[i as int];
    let hi = cu_q[i as int + 1];
    MODEL::lemma_scaled_embed_repr_shape(
        input_ids, wr.embed_weight, family.geometry.hidden_size,
    );
    BI::scaled_embed_subrange_invariance(
        input_ids, wr.embed_weight, family.geometry.hidden_size, lo, hi,
    );
    lemma_chain_layer_kv_rows_request_isolation(
        wr.layers, family.layers, hidden,
        MODEL::scaled_embed_repr(
            input_ids.subrange(lo, hi), wr.embed_weight,
            family.geometry.hidden_size,
        ),
        positions, pre_kv, pre_kv, slots,
        cu_q, cu_k, max_q, max_k, block_table, i, 0, layer,
    );
}

// Batched-to-singleton KV projection for one request.  The four-norm decoder's full/SWA
// attention certificates establish whole-page equality; this adapter lowers
// that stronger fact to the common complete-causal-prefix cache predicate.
pub proof fn lemma_kv_request_isolation(
    wr: ModelWeightsRepr,
    family: FourNormGatedDecoderConfigRepr,
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
        BI::request_projection_ready(
            wr, family, input_ids, positions, pre_kv, slots,
            cu_q, cu_k, max_q, max_k, block_table, i,
        ),
    ensures ({
        let lo = cu_q[i as int];
        let hi = cu_q[i as int + 1];
        let q_len = (hi - lo) as nat;
        let k_len = (cu_k[i as int + 1] - cu_k[i as int]) as nat;
        crate::proof::model::family_layout::cache_sequence_logical_prefix_equal(
            MODEL::model_forward_kv_reprs(
                wr, family, input_ids, positions, pre_kv, slots,
                cu_q, cu_k, max_q, max_k, block_table,
            ),
            block_table[i as int],
            MODEL::model_forward_kv_reprs(
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
    reveal(BI::request_projection_ready);
    let lo = cu_q[i as int];
    let hi = cu_q[i as int + 1];
    let q_len = (hi - lo) as nat;
    let k_len = (cu_k[i as int + 1] - cu_k[i as int]) as nat;
    let own_slots = slots.subrange(lo, hi);
    let single_cu_q = seq![0int, q_len as int];
    let single_cu_k = seq![0int, k_len as int];
    let single_bt = seq![block_table[i as int]];
    let full_post = MODEL::model_forward_kv_reprs(
        wr, family, input_ids, positions, pre_kv, slots,
        cu_q, cu_k, max_q, max_k, block_table,
    );
    let single_post = MODEL::model_forward_kv_reprs(
        wr, family, input_ids.subrange(lo, hi),
        positions.subrange(lo, hi), pre_kv, own_slots,
        single_cu_q, single_cu_k, q_len, k_len, single_bt,
    );
    BI::model_forward_kv_request_isolation_from_layout(
        wr, family, input_ids, positions, pre_kv, slots,
        cu_q, cu_k, max_q, max_k, block_table, i,
    );
    MODEL::lemma_model_forward_kv_reprs_len(
        wr, family, input_ids, positions, pre_kv, slots,
        cu_q, cu_k, max_q, max_k, block_table,
    );
    MODEL::lemma_model_forward_kv_reprs_len(
        wr, family, input_ids.subrange(lo, hi),
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
        lemma_layer_kv_rows_shape(
            wr, family, input_ids.subrange(lo, hi),
            positions.subrange(lo, hi), pre_kv, own_slots,
            single_cu_q, single_cu_k, q_len, k_len, single_bt,
            layer as nat,
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
        assert(full_post[layer] == RT::store_kv_cache_repr(
            full_rows.0, full_rows.1,
            pre_kv[layer].0, pre_kv[layer].1, slots,
        ));
        assert(single_post[layer] == RT::store_kv_cache_repr(
            single_rows.0, single_rows.1,
            pre_kv[layer].0, pre_kv[layer].1, own_slots,
        ));
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
            crate::proof::tensor::geometry::lemma_blocks_needed_covers_pos(pos, k_len);
            assert(pos < crate::proof::tensor::geometry::blocks_needed_for(k_len)
                * crate::types::BLOCK_SIZE_SPEC) by {};
            let slot = crate::proof::tensor::geometry::block_table_slot(
                block_table[i as int], pos,
            );
            assert(crate::proof::tensor::geometry::slot_in_cache(pre_kv[layer].0, slot));
            assert(crate::proof::tensor::geometry::slot_in_cache(pre_kv[layer].1, slot));
            RT::store_kv_cache_repr_preserves_slot_in_cache(
                full_rows.0, full_rows.1,
                pre_kv[layer].0, pre_kv[layer].1, slots, slot,
            );
            RT::store_kv_cache_repr_preserves_slot_in_cache(
                single_rows.0, single_rows.1,
                pre_kv[layer].0, pre_kv[layer].1, own_slots, slot,
            );
            assert(BI::paged_attention_selected_cache_pages_equal(
                full_post[layer].0,
                full_post[layer].1,
                block_table[i as int],
                single_post[layer].0,
                single_post[layer].1,
                block_table[i as int],
                k_len,
                family.layers[layer].attention,
            ));
            reveal(BI::paged_attention_selected_cache_pages_equal);
            match family.layers[layer].attention {
                AttentionConfigRepr::Full => {
                    reveal(GAC::paged_attention_selected_cache_pages_equal);
                },
                AttentionConfigRepr::SlidingWindow(_) => {
                    reveal(GAC::paged_attention_selected_cache_pages_equal);
                },
            }
        }
    }
    reveal(crate::proof::model::family_layout::cache_sequence_logical_prefix_equal);
}

} // verus!
