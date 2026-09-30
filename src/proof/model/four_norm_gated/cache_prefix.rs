//! Prefix stability and continuation for the shared four-norm decoder fold.
//!
//! These inductions depend on row-local policies and conservative causal
//! attention contracts, not a concrete model tag or cache-eviction policy.

#[cfg(verus_only)]
use crate::proof::model::four_norm_gated::batch_invariance as BI;
#[cfg(verus_only)]
use crate::proof::model::four_norm_gated::model as MODEL;
#[cfg(verus_only)]
use crate::proof::model::cache as CACHE_LAWS;
#[cfg(verus_only)]
use crate::proof::reference::request_machine as MACHINE;
use crate::{types::*, proof::model::types::*, proof::tensor::types::*};
#[cfg(verus_only)]
use crate::boundary::tensor_runtime as RT;
use vstd::prelude::*;

verus! {

// A single four-norm block preserves cold-prefill prefix rows and stores identical
// K/V values on that prefix.  All non-attention work is row-local; the sole
// causal step is `paged_attention_singleton_prefix_invariance`.
pub proof fn lemma_decoder_layer_full_prefill_prefix_stable(
    common: LayerWeightsRepr,
    extension: FourNormGatedLayerExtensionRepr,
    full_hidden: Tensor2D,
    prefix_hidden: Tensor2D,
    full_base_k: KVCacheLayerRepr,
    full_base_v: KVCacheLayerRepr,
    prefix_base_k: KVCacheLayerRepr,
    prefix_base_v: KVCacheLayerRepr,
    full_len: nat,
    prefix_len: nat,
)
    requires
        RT::paged_attention_numeric_domain(),
        layer_attention_config_valid(extension.attention),
        0 < prefix_len <= full_len,
        crate::proof::tensor::geometry::blocks_needed_for(full_len) <= u64::MAX as nat,
        full_hidden.len() == full_len,
        prefix_hidden.len() == prefix_len,
        full_hidden.subrange(0, prefix_len as int) == prefix_hidden,
        forall|pos: nat| #![auto]
            pos < full_len ==>
                crate::proof::tensor::geometry::slot_in_cache(full_base_k, pos)
                && crate::proof::tensor::geometry::slot_in_cache(full_base_v, pos),
        forall|pos: nat| #![auto]
            pos < prefix_len ==>
                crate::proof::tensor::geometry::slot_in_cache(prefix_base_k, pos)
                && crate::proof::tensor::geometry::slot_in_cache(prefix_base_v, pos),
    ensures ({
        let full_positions = crate::proof::tensor::geometry::positions_from(0, full_len);
        let prefix_positions = crate::proof::tensor::geometry::positions_from(0, prefix_len);
        let full_slots = MACHINE::slots_from(0, full_len);
        let prefix_slots = MACHINE::slots_from(0, prefix_len);
        let full_layer = MODEL::decoder_layer_step_repr(
            common, extension, full_hidden, full_positions,
            full_base_k, full_base_v, full_slots,
            MACHINE::seq_lens_for_single(full_len),
            MACHINE::seq_lens_for_single(full_len),
            full_len, full_len, MACHINE::singleton_block_rows(full_len),
        );
        let prefix_layer = MODEL::decoder_layer_step_repr(
            common, extension, prefix_hidden, prefix_positions,
            prefix_base_k, prefix_base_v, prefix_slots,
            MACHINE::seq_lens_for_single(prefix_len),
            MACHINE::seq_lens_for_single(prefix_len),
            prefix_len, prefix_len, MACHINE::singleton_block_rows(prefix_len),
        );
        &&& full_layer.0.subrange(0, prefix_len as int) == prefix_layer.0
        &&& forall|pos: nat|
            #![trigger crate::proof::tensor::geometry::cache_at(full_layer.1.0, pos)]
            #![trigger crate::proof::tensor::geometry::cache_at(prefix_layer.1.0, pos)]
            pos < prefix_len ==> {
                &&& crate::proof::tensor::geometry::slot_in_cache(full_layer.1.0, pos)
                &&& crate::proof::tensor::geometry::slot_in_cache(full_layer.1.1, pos)
                &&& crate::proof::tensor::geometry::slot_in_cache(prefix_layer.1.0, pos)
                &&& crate::proof::tensor::geometry::slot_in_cache(prefix_layer.1.1, pos)
                &&& crate::proof::tensor::geometry::cache_at(full_layer.1.0, pos)
                    == crate::proof::tensor::geometry::cache_at(prefix_layer.1.0, pos)
                &&& crate::proof::tensor::geometry::cache_at(full_layer.1.1, pos)
                    == crate::proof::tensor::geometry::cache_at(prefix_layer.1.1, pos)
            }
    }),
{
    broadcast use {
        MODEL::lemma_attention_pre_store_repr_shape,
        MODEL::lemma_paged_attention_repr_shape,
    };
    let full_positions = crate::proof::tensor::geometry::positions_from(0, full_len);
    let prefix_positions = crate::proof::tensor::geometry::positions_from(0, prefix_len);
    let full_slots = MACHINE::slots_from(0, full_len);
    let prefix_slots = MACHINE::slots_from(0, prefix_len);
    assert(full_positions.subrange(0, prefix_len as int)
        =~= prefix_positions);

    let full_pre = MODEL::attention_pre_store_repr(
        common, extension, full_hidden, full_positions,
    );
    let prefix_pre = MODEL::attention_pre_store_repr(
        common, extension, prefix_hidden, prefix_positions,
    );
    MODEL::lemma_attention_pre_store_repr_shape(
        common, extension, full_hidden, full_positions,
    );
    MODEL::lemma_attention_pre_store_repr_shape(
        common, extension, prefix_hidden, prefix_positions,
    );
    assert(full_pre.0.len() == full_len);
    assert(full_pre.1.len() == full_len);
    assert(full_pre.2.len() == full_len);
    assert(prefix_pre.0.len() == prefix_len);
    assert(prefix_pre.1.len() == prefix_len);
    assert(prefix_pre.2.len() == prefix_len);
    assert(full_slots.len() == full_len);
    assert(prefix_slots.len() == prefix_len);
    BI::attention_pre_store_subrange_invariance(
        common, extension, full_hidden, full_positions,
        0, prefix_len as int,
    );
    assert(prefix_pre.0 == full_pre.0.subrange(0, prefix_len as int));
    assert(prefix_pre.1 == full_pre.1.subrange(0, prefix_len as int));
    assert(prefix_pre.2 == full_pre.2.subrange(0, prefix_len as int));

    let full_cache = RT::store_kv_cache_repr(
        full_pre.1, full_pre.2,
        full_base_k, full_base_v, full_slots,
    );
    let prefix_cache = RT::store_kv_cache_repr(
        prefix_pre.1, prefix_pre.2,
        prefix_base_k, prefix_base_v, prefix_slots,
    );
    assert forall|pos: nat| #![auto]
        pos < prefix_len implies {
            &&& crate::proof::tensor::geometry::slot_in_cache(full_cache.0, pos)
            &&& crate::proof::tensor::geometry::slot_in_cache(full_cache.1, pos)
            &&& crate::proof::tensor::geometry::slot_in_cache(prefix_cache.0, pos)
            &&& crate::proof::tensor::geometry::slot_in_cache(prefix_cache.1, pos)
            &&& crate::proof::tensor::geometry::cache_at(full_cache.0, pos)
                == crate::proof::tensor::geometry::cache_at(prefix_cache.0, pos)
            &&& crate::proof::tensor::geometry::cache_at(full_cache.1, pos)
                == crate::proof::tensor::geometry::cache_at(prefix_cache.1, pos)
        }
    by {
        assert(full_slots[pos as int] == pos as int);
        assert(prefix_slots[pos as int] == pos as int);
        assert forall|m: int| pos < m < full_slots.len()
            implies full_slots[m] != full_slots[pos as int] by {}
        assert forall|m: int| pos < m < prefix_slots.len()
            implies prefix_slots[m] != prefix_slots[pos as int] by {}
        RT::store_kv_cache_repr_reads_own_write(
            full_pre.1, full_pre.2,
            full_base_k, full_base_v, full_slots, pos as int,
        );
        RT::store_kv_cache_repr_reads_own_write(
            prefix_pre.1, prefix_pre.2,
            prefix_base_k, prefix_base_v, prefix_slots, pos as int,
        );
        assert(full_pre.1[pos as int] == prefix_pre.1[pos as int]);
        assert(full_pre.2[pos as int] == prefix_pre.2[pos as int]);
    }

    let full_bt = MACHINE::singleton_block_rows(full_len)[0];
    let prefix_bt = MACHINE::singleton_block_rows(prefix_len)[0];
    let full_pages = crate::proof::tensor::geometry::blocks_needed_for(full_len);
    let prefix_pages = crate::proof::tensor::geometry::blocks_needed_for(prefix_len);
    assert(full_bt =~= MACHINE::contiguous_block_ids(full_pages));
    assert(prefix_bt =~= MACHINE::contiguous_block_ids(prefix_pages));
    crate::proof::tensor::geometry::lemma_blocks_needed_monotone(prefix_len, full_len);
    assert(prefix_pages <= u64::MAX as nat);
    assert forall|pos: nat|
        #![trigger crate::proof::tensor::geometry::block_table_slot(full_bt, pos)]
        #![trigger crate::proof::tensor::geometry::block_table_slot(prefix_bt, pos)]
        pos < prefix_len implies {
            let full_slot = crate::proof::tensor::geometry::block_table_slot(full_bt, pos);
            let prefix_slot = crate::proof::tensor::geometry::block_table_slot(prefix_bt, pos);
            &&& crate::proof::tensor::geometry::slot_in_cache(full_cache.0, full_slot)
            &&& crate::proof::tensor::geometry::slot_in_cache(full_cache.1, full_slot)
            &&& crate::proof::tensor::geometry::slot_in_cache(prefix_cache.0, prefix_slot)
            &&& crate::proof::tensor::geometry::slot_in_cache(prefix_cache.1, prefix_slot)
            &&& crate::proof::tensor::geometry::cache_at(full_cache.0, full_slot)
                == crate::proof::tensor::geometry::cache_at(prefix_cache.0, prefix_slot)
            &&& crate::proof::tensor::geometry::cache_at(full_cache.1, full_slot)
                == crate::proof::tensor::geometry::cache_at(prefix_cache.1, prefix_slot)
        }
    by {
        crate::proof::tensor::geometry::lemma_blocks_needed_covers_pos(pos, full_len);
        crate::proof::tensor::geometry::lemma_blocks_needed_covers_pos(pos, prefix_len);
        MACHINE::lemma_contiguous_block_table_slot(full_pages, pos);
        MACHINE::lemma_contiguous_block_table_slot(prefix_pages, pos);
    }

    let full_attended = MODEL::paged_attention_repr(
        full_pre.0, full_cache.0, full_cache.1,
        MACHINE::seq_lens_for_single(full_len),
        MACHINE::seq_lens_for_single(full_len),
        full_len, full_len, MACHINE::singleton_block_rows(full_len),
        extension.attention, crate::boundary::dense_layer_primitives::layer_attention_parameters_repr(common, extension.attention_scale),
    );
    let prefix_attended = MODEL::paged_attention_repr(
        prefix_pre.0, prefix_cache.0, prefix_cache.1,
        MACHINE::seq_lens_for_single(prefix_len),
        MACHINE::seq_lens_for_single(prefix_len),
        prefix_len, prefix_len, MACHINE::singleton_block_rows(prefix_len),
        extension.attention, crate::boundary::dense_layer_primitives::layer_attention_parameters_repr(common, extension.attention_scale),
    );
    BI::paged_attention_singleton_prefix_invariance(
        full_pre.0, prefix_pre.0,
        full_cache.0, full_cache.1,
        prefix_cache.0, prefix_cache.1,
        full_bt, prefix_bt, full_len, prefix_len,
        extension.attention, crate::boundary::dense_layer_primitives::layer_attention_parameters_repr(common, extension.attention_scale),
    );
    assert(full_attended.subrange(0, prefix_len as int)
        == prefix_attended);

    BI::post_attention_and_mlp_subrange_invariance(
        common, extension, full_hidden, full_attended,
        0, prefix_len as int,
    );
    let full_post = MODEL::post_attention_and_mlp_repr(
        common, extension, full_hidden, full_attended,
    );
    let prefix_post = MODEL::post_attention_and_mlp_repr(
        common, extension, prefix_hidden, prefix_attended,
    );
    assert(full_post.subrange(0, prefix_len as int) == prefix_post);

    let full_layer = MODEL::decoder_layer_step_repr(
        common, extension, full_hidden, full_positions,
        full_base_k, full_base_v, full_slots,
        MACHINE::seq_lens_for_single(full_len),
        MACHINE::seq_lens_for_single(full_len),
        full_len, full_len, MACHINE::singleton_block_rows(full_len),
    );
    let prefix_layer = MODEL::decoder_layer_step_repr(
        common, extension, prefix_hidden, prefix_positions,
        prefix_base_k, prefix_base_v, prefix_slots,
        MACHINE::seq_lens_for_single(prefix_len),
        MACHINE::seq_lens_for_single(prefix_len),
        prefix_len, prefix_len, MACHINE::singleton_block_rows(prefix_len),
    );
    reveal(MODEL::decoder_layer_step_repr);
    assert(full_layer.0 == full_post && full_layer.1 == full_cache);
    assert(prefix_layer.0 == prefix_post && prefix_layer.1 == prefix_cache);
}

pub proof fn lemma_layer_chain_full_prefill_prefix_stable(
    common_layers: Seq<LayerWeightsRepr>,
    extension_layers: Seq<FourNormGatedLayerExtensionRepr>,
    full_hidden: Tensor2D,
    prefix_hidden: Tensor2D,
    full_caches: Seq<(KVCacheLayerRepr, KVCacheLayerRepr)>,
    prefix_caches: Seq<(KVCacheLayerRepr, KVCacheLayerRepr)>,
    full_len: nat,
    prefix_len: nat,
    start: nat,
)
    requires
        RT::paged_attention_numeric_domain(),
        start <= common_layers.len(),
        extension_layers.len() == common_layers.len(),
        forall|layer: int| start <= layer < extension_layers.len() ==>
            layer_attention_config_valid(
                #[trigger] extension_layers[layer].attention,
            ),
        0 < prefix_len <= full_len,
        crate::proof::tensor::geometry::blocks_needed_for(full_len) <= u64::MAX as nat,
        full_caches.len() >= common_layers.len(),
        prefix_caches.len() >= common_layers.len(),
        full_hidden.len() == full_len,
        prefix_hidden.len() == prefix_len,
        full_hidden.subrange(0, prefix_len as int) == prefix_hidden,
        CACHE_LAWS::cache_sequence_has_positions_in_range(
            full_caches, start, common_layers.len(), full_len,
        ),
        CACHE_LAWS::cache_sequence_has_positions_in_range(
            prefix_caches, start, common_layers.len(), prefix_len,
        ),
    ensures ({
        let full_positions = crate::proof::tensor::geometry::positions_from(0, full_len);
        let prefix_positions = crate::proof::tensor::geometry::positions_from(0, prefix_len);
        let full_slots = MACHINE::slots_from(0, full_len);
        let prefix_slots = MACHINE::slots_from(0, prefix_len);
        let full_out = MODEL::layer_chain_repr(
            common_layers, extension_layers, full_hidden, full_positions,
            full_caches, full_slots,
            MACHINE::seq_lens_for_single(full_len),
            MACHINE::seq_lens_for_single(full_len),
            full_len, full_len, MACHINE::singleton_block_rows(full_len), start,
        );
        let prefix_out = MODEL::layer_chain_repr(
            common_layers, extension_layers, prefix_hidden, prefix_positions,
            prefix_caches, prefix_slots,
            MACHINE::seq_lens_for_single(prefix_len),
            MACHINE::seq_lens_for_single(prefix_len),
            prefix_len, prefix_len, MACHINE::singleton_block_rows(prefix_len), start,
        );
        &&& full_out.0.subrange(0, prefix_len as int) == prefix_out.0
        &&& CACHE_LAWS::cache_sequences_agree_on_prefix_in_range(
            full_out.1, prefix_out.1,
            start, common_layers.len(), prefix_len,
        )
    }),
    decreases (common_layers.len() - start) as nat,
{
    let full_positions = crate::proof::tensor::geometry::positions_from(0, full_len);
    let prefix_positions = crate::proof::tensor::geometry::positions_from(0, prefix_len);
    let full_slots = MACHINE::slots_from(0, full_len);
    let prefix_slots = MACHINE::slots_from(0, prefix_len);
    MODEL::lemma_layer_chain_repr_shape(
        common_layers, extension_layers, full_hidden, full_positions,
        full_caches, full_slots,
        MACHINE::seq_lens_for_single(full_len),
        MACHINE::seq_lens_for_single(full_len),
        full_len, full_len, MACHINE::singleton_block_rows(full_len), start,
    );
    MODEL::lemma_layer_chain_repr_shape(
        common_layers, extension_layers, prefix_hidden, prefix_positions,
        prefix_caches, prefix_slots,
        MACHINE::seq_lens_for_single(prefix_len),
        MACHINE::seq_lens_for_single(prefix_len),
        prefix_len, prefix_len, MACHINE::singleton_block_rows(prefix_len), start,
    );
    if start >= common_layers.len() {
        assert(CACHE_LAWS::cache_sequences_agree_on_prefix_in_range(
            full_caches, prefix_caches,
            start, common_layers.len(), prefix_len,
        )) by {
            reveal(CACHE_LAWS::cache_sequences_agree_on_prefix_in_range);
        }
    } else {
        let layer = start as int;
        let full_step = MODEL::decoder_layer_step_repr(
            common_layers[layer], extension_layers[layer],
            full_hidden, full_positions,
            full_caches[layer].0, full_caches[layer].1, full_slots,
            MACHINE::seq_lens_for_single(full_len),
            MACHINE::seq_lens_for_single(full_len),
            full_len, full_len, MACHINE::singleton_block_rows(full_len),
        );
        let prefix_step = MODEL::decoder_layer_step_repr(
            common_layers[layer], extension_layers[layer],
            prefix_hidden, prefix_positions,
            prefix_caches[layer].0, prefix_caches[layer].1, prefix_slots,
            MACHINE::seq_lens_for_single(prefix_len),
            MACHINE::seq_lens_for_single(prefix_len),
            prefix_len, prefix_len, MACHINE::singleton_block_rows(prefix_len),
        );
        lemma_decoder_layer_full_prefill_prefix_stable(
            common_layers[layer], extension_layers[layer],
            full_hidden, prefix_hidden,
            full_caches[layer].0, full_caches[layer].1,
            prefix_caches[layer].0, prefix_caches[layer].1,
            full_len, prefix_len,
        );
        MODEL::lemma_decoder_layer_step_repr_shape(
            common_layers[layer], extension_layers[layer],
            full_hidden, full_positions,
            full_caches[layer].0, full_caches[layer].1, full_slots,
            MACHINE::seq_lens_for_single(full_len),
            MACHINE::seq_lens_for_single(full_len),
            full_len, full_len, MACHINE::singleton_block_rows(full_len),
        );
        MODEL::lemma_decoder_layer_step_repr_shape(
            common_layers[layer], extension_layers[layer],
            prefix_hidden, prefix_positions,
            prefix_caches[layer].0, prefix_caches[layer].1, prefix_slots,
            MACHINE::seq_lens_for_single(prefix_len),
            MACHINE::seq_lens_for_single(prefix_len),
            prefix_len, prefix_len, MACHINE::singleton_block_rows(prefix_len),
        );
        assert(full_step.0.subrange(0, prefix_len as int) == prefix_step.0);
        let full_next_caches = full_caches.update(layer, full_step.1);
        let prefix_next_caches = prefix_caches.update(layer, prefix_step.1);
        assert(CACHE_LAWS::cache_sequence_has_positions_in_range(
            full_next_caches, (start + 1) as nat,
            common_layers.len(), full_len,
        )) by {
            reveal(CACHE_LAWS::cache_sequence_has_positions_in_range);
            assert forall|later: int, pos: nat| #![auto]
                start + 1 <= later < common_layers.len()
                    && pos < full_len implies
                    crate::proof::tensor::geometry::slot_in_cache(
                        full_next_caches[later].0, pos,
                    )
                    && crate::proof::tensor::geometry::slot_in_cache(
                        full_next_caches[later].1, pos,
                    )
            by {
                assert(later != layer);
                assert(full_next_caches[later] == full_caches[later]);
            }
        }
        assert(CACHE_LAWS::cache_sequence_has_positions_in_range(
            prefix_next_caches, (start + 1) as nat,
            common_layers.len(), prefix_len,
        )) by {
            reveal(CACHE_LAWS::cache_sequence_has_positions_in_range);
            assert forall|later: int, pos: nat| #![auto]
                start + 1 <= later < common_layers.len()
                    && pos < prefix_len implies
                    crate::proof::tensor::geometry::slot_in_cache(
                        prefix_next_caches[later].0, pos,
                    )
                    && crate::proof::tensor::geometry::slot_in_cache(
                        prefix_next_caches[later].1, pos,
                    )
            by {
                assert(later != layer);
                assert(prefix_next_caches[later] == prefix_caches[later]);
            }
        }
        lemma_layer_chain_full_prefill_prefix_stable(
            common_layers, extension_layers,
            full_step.0, prefix_step.0,
            full_next_caches, prefix_next_caches,
            full_len, prefix_len, (start + 1) as nat,
        );

        let full_tail = MODEL::layer_chain_repr(
            common_layers, extension_layers, full_step.0, full_positions,
            full_next_caches, full_slots,
            MACHINE::seq_lens_for_single(full_len),
            MACHINE::seq_lens_for_single(full_len),
            full_len, full_len, MACHINE::singleton_block_rows(full_len),
            (start + 1) as nat,
        );
        let prefix_tail = MODEL::layer_chain_repr(
            common_layers, extension_layers, prefix_step.0, prefix_positions,
            prefix_next_caches, prefix_slots,
            MACHINE::seq_lens_for_single(prefix_len),
            MACHINE::seq_lens_for_single(prefix_len),
            prefix_len, prefix_len, MACHINE::singleton_block_rows(prefix_len),
            (start + 1) as nat,
        );
        MODEL::lemma_layer_chain_cache_before_start_unchanged(
            common_layers, extension_layers, full_step.0, full_positions,
            full_next_caches, full_slots,
            MACHINE::seq_lens_for_single(full_len),
            MACHINE::seq_lens_for_single(full_len),
            full_len, full_len, MACHINE::singleton_block_rows(full_len),
            (start + 1) as nat, layer,
        );
        MODEL::lemma_layer_chain_cache_before_start_unchanged(
            common_layers, extension_layers, prefix_step.0, prefix_positions,
            prefix_next_caches, prefix_slots,
            MACHINE::seq_lens_for_single(prefix_len),
            MACHINE::seq_lens_for_single(prefix_len),
            prefix_len, prefix_len, MACHINE::singleton_block_rows(prefix_len),
            (start + 1) as nat, layer,
        );
        assert(full_tail.1[layer] == full_step.1);
        assert(prefix_tail.1[layer] == prefix_step.1);
        assert forall|later: int, pos: nat| #![auto]
            start <= later < common_layers.len() && pos < prefix_len implies {
                &&& crate::proof::tensor::geometry::slot_in_cache(full_tail.1[later].0, pos)
                &&& crate::proof::tensor::geometry::slot_in_cache(full_tail.1[later].1, pos)
                &&& crate::proof::tensor::geometry::slot_in_cache(prefix_tail.1[later].0, pos)
                &&& crate::proof::tensor::geometry::slot_in_cache(prefix_tail.1[later].1, pos)
                &&& crate::proof::tensor::geometry::cache_at(full_tail.1[later].0, pos)
                    == crate::proof::tensor::geometry::cache_at(prefix_tail.1[later].0, pos)
                &&& crate::proof::tensor::geometry::cache_at(full_tail.1[later].1, pos)
                    == crate::proof::tensor::geometry::cache_at(prefix_tail.1[later].1, pos)
            }
        by {
            if later == layer {
                assert(crate::proof::tensor::geometry::cache_at(full_step.1.0, pos)
                    == crate::proof::tensor::geometry::cache_at(prefix_step.1.0, pos));
                assert(crate::proof::tensor::geometry::cache_at(full_step.1.1, pos)
                    == crate::proof::tensor::geometry::cache_at(prefix_step.1.1, pos));
            } else {
                assert(start + 1 <= later);
                assert(CACHE_LAWS::cache_sequences_agree_on_prefix_in_range(
                    full_tail.1, prefix_tail.1,
                    (start + 1) as nat, common_layers.len(), prefix_len,
                ));
            }
        }
        assert(CACHE_LAWS::cache_sequences_agree_on_prefix_in_range(
            full_tail.1, prefix_tail.1,
            start, common_layers.len(), prefix_len,
        )) by {
            reveal(CACHE_LAWS::cache_sequences_agree_on_prefix_in_range);
        }
        let full_out = MODEL::layer_chain_repr(
            common_layers, extension_layers, full_hidden, full_positions,
            full_caches, full_slots,
            MACHINE::seq_lens_for_single(full_len),
            MACHINE::seq_lens_for_single(full_len),
            full_len, full_len, MACHINE::singleton_block_rows(full_len), start,
        );
        let prefix_out = MODEL::layer_chain_repr(
            common_layers, extension_layers, prefix_hidden, prefix_positions,
            prefix_caches, prefix_slots,
            MACHINE::seq_lens_for_single(prefix_len),
            MACHINE::seq_lens_for_single(prefix_len),
            prefix_len, prefix_len, MACHINE::singleton_block_rows(prefix_len), start,
        );
        reveal(MODEL::layer_chain_repr);
        assert(full_out == full_tail);
        assert(prefix_out == prefix_tail);
    }
}


pub proof fn lemma_decoder_layer_canonical_prefix_continuation(
    common: LayerWeightsRepr,
    extension: FourNormGatedLayerExtensionRepr,
    full_hidden: Tensor2D,
    suffix_hidden: Tensor2D,
    full_base_k: KVCacheLayerRepr,
    full_base_v: KVCacheLayerRepr,
    suffix_base_k: KVCacheLayerRepr,
    suffix_base_v: KVCacheLayerRepr,
    full_len: nat,
    prefix_len: nat,
)
    requires
        RT::paged_attention_numeric_domain(),
        layer_attention_config_valid(extension.attention),
        0 < prefix_len < full_len,
        crate::proof::tensor::geometry::blocks_needed_for(full_len) <= u64::MAX as nat,
        full_hidden.len() == full_len,
        suffix_hidden.len() == full_len - prefix_len,
        full_hidden.subrange(prefix_len as int, full_len as int)
            == suffix_hidden,
        forall|pos: nat| #![auto]
            pos < full_len ==>
                crate::proof::tensor::geometry::slot_in_cache(full_base_k, pos)
                && crate::proof::tensor::geometry::slot_in_cache(full_base_v, pos)
                && crate::proof::tensor::geometry::slot_in_cache(suffix_base_k, pos)
                && crate::proof::tensor::geometry::slot_in_cache(suffix_base_v, pos),
        forall|pos: nat|
            #![trigger crate::proof::tensor::geometry::cache_at(suffix_base_k, pos)]
            #![trigger crate::proof::tensor::geometry::cache_at(suffix_base_v, pos)]
            pos < prefix_len ==> {
                let full_positions = crate::proof::tensor::geometry::positions_from(0, full_len);
                let full_slots = MACHINE::slots_from(0, full_len);
                let full_step = MODEL::decoder_layer_step_repr(
                    common, extension, full_hidden, full_positions,
                    full_base_k, full_base_v, full_slots,
                    MACHINE::seq_lens_for_single(full_len),
                    MACHINE::seq_lens_for_single(full_len),
                    full_len, full_len,
                    MACHINE::singleton_block_rows(full_len),
                );
                &&& crate::proof::tensor::geometry::cache_at(suffix_base_k, pos)
                    == crate::proof::tensor::geometry::cache_at(full_step.1.0, pos)
                &&& crate::proof::tensor::geometry::cache_at(suffix_base_v, pos)
                    == crate::proof::tensor::geometry::cache_at(full_step.1.1, pos)
            },
    ensures ({
        let suffix_len = (full_len - prefix_len) as nat;
        let full_step = MODEL::decoder_layer_step_repr(
            common, extension, full_hidden,
            crate::proof::tensor::geometry::positions_from(0, full_len),
            full_base_k, full_base_v, MACHINE::slots_from(0, full_len),
            MACHINE::seq_lens_for_single(full_len),
            MACHINE::seq_lens_for_single(full_len),
            full_len, full_len, MACHINE::singleton_block_rows(full_len),
        );
        let suffix_step = MODEL::decoder_layer_step_repr(
            common, extension, suffix_hidden,
            crate::proof::tensor::geometry::positions_from(prefix_len, suffix_len),
            suffix_base_k, suffix_base_v,
            MACHINE::slots_from(prefix_len, suffix_len),
            MACHINE::seq_lens_for_single(suffix_len),
            MACHINE::seq_lens_for_single(full_len),
            suffix_len, full_len, MACHINE::singleton_block_rows(full_len),
        );
        &&& full_step.0.subrange(prefix_len as int, full_len as int)
            == suffix_step.0
        &&& forall|pos: nat| #![auto]
            pos < full_len ==> {
                &&& crate::proof::tensor::geometry::slot_in_cache(full_step.1.0, pos)
                &&& crate::proof::tensor::geometry::slot_in_cache(full_step.1.1, pos)
                &&& crate::proof::tensor::geometry::slot_in_cache(suffix_step.1.0, pos)
                &&& crate::proof::tensor::geometry::slot_in_cache(suffix_step.1.1, pos)
                &&& crate::proof::tensor::geometry::cache_at(full_step.1.0, pos)
                    == crate::proof::tensor::geometry::cache_at(suffix_step.1.0, pos)
                &&& crate::proof::tensor::geometry::cache_at(full_step.1.1, pos)
                    == crate::proof::tensor::geometry::cache_at(suffix_step.1.1, pos)
            }
    }),
{
    broadcast use {
        MODEL::lemma_attention_pre_store_repr_shape,
        MODEL::lemma_paged_attention_repr_shape,
    };
    let suffix_len = (full_len - prefix_len) as nat;
    let full_positions = crate::proof::tensor::geometry::positions_from(0, full_len);
    let suffix_positions = crate::proof::tensor::geometry::positions_from(
        prefix_len, suffix_len,
    );
    let full_slots = MACHINE::slots_from(0, full_len);
    let suffix_slots = MACHINE::slots_from(prefix_len, suffix_len);
    assert(full_positions.subrange(prefix_len as int, full_len as int)
        =~= suffix_positions);

    let full_pre = MODEL::attention_pre_store_repr(
        common, extension, full_hidden, full_positions,
    );
    let suffix_pre = MODEL::attention_pre_store_repr(
        common, extension, suffix_hidden, suffix_positions,
    );
    MODEL::lemma_attention_pre_store_repr_shape(
        common, extension, full_hidden, full_positions,
    );
    MODEL::lemma_attention_pre_store_repr_shape(
        common, extension, suffix_hidden, suffix_positions,
    );
    BI::attention_pre_store_subrange_invariance(
        common, extension, full_hidden, full_positions,
        prefix_len as int, full_len as int,
    );
    assert(suffix_pre.0
        == full_pre.0.subrange(prefix_len as int, full_len as int));
    assert(suffix_pre.1
        == full_pre.1.subrange(prefix_len as int, full_len as int));
    assert(suffix_pre.2
        == full_pre.2.subrange(prefix_len as int, full_len as int));
    assert(full_slots.len() == full_len);
    assert(suffix_slots.len() == suffix_len);

    let full_cache = RT::store_kv_cache_repr(
        full_pre.1, full_pre.2,
        full_base_k, full_base_v, full_slots,
    );
    let suffix_cache = RT::store_kv_cache_repr(
        suffix_pre.1, suffix_pre.2,
        suffix_base_k, suffix_base_v, suffix_slots,
    );
    assert forall|pos: nat| #![auto]
        pos < full_len implies {
            &&& crate::proof::tensor::geometry::slot_in_cache(full_cache.0, pos)
            &&& crate::proof::tensor::geometry::slot_in_cache(full_cache.1, pos)
            &&& crate::proof::tensor::geometry::slot_in_cache(suffix_cache.0, pos)
            &&& crate::proof::tensor::geometry::slot_in_cache(suffix_cache.1, pos)
            &&& crate::proof::tensor::geometry::cache_at(full_cache.0, pos)
                == crate::proof::tensor::geometry::cache_at(suffix_cache.0, pos)
            &&& crate::proof::tensor::geometry::cache_at(full_cache.1, pos)
                == crate::proof::tensor::geometry::cache_at(suffix_cache.1, pos)
        }
    by {
        if pos < prefix_len {
            assert(!suffix_slots.contains(pos as int)) by {
                if suffix_slots.contains(pos as int) {
                    let j = suffix_slots.index_of(pos as int);
                    assert(suffix_slots[j] == prefix_len as int + j);
                }
            }
            RT::store_kv_cache_repr_preserves_unwritten_slots(
                suffix_pre.1, suffix_pre.2,
                suffix_base_k, suffix_base_v, suffix_slots, pos,
            );
            RT::store_kv_cache_repr_preserves_slot_in_cache(
                full_pre.1, full_pre.2,
                full_base_k, full_base_v, full_slots, pos,
            );
            let full_step = MODEL::decoder_layer_step_repr(
                common, extension, full_hidden, full_positions,
                full_base_k, full_base_v, full_slots,
                MACHINE::seq_lens_for_single(full_len),
                MACHINE::seq_lens_for_single(full_len),
                full_len, full_len,
                MACHINE::singleton_block_rows(full_len),
            );
            reveal(MODEL::decoder_layer_step_repr);
            assert(full_step.1 == full_cache);
        } else {
            let j = (pos - prefix_len) as int;
            assert(0 <= j < suffix_len);
            assert(full_slots[pos as int] == pos as int);
            assert(suffix_slots[j] == pos as int);
            assert forall|m: int| pos < m < full_slots.len()
                implies full_slots[m] != full_slots[pos as int] by {}
            assert forall|m: int| j < m < suffix_slots.len()
                implies suffix_slots[m] != suffix_slots[j] by {}
            RT::store_kv_cache_repr_reads_own_write(
                full_pre.1, full_pre.2,
                full_base_k, full_base_v, full_slots, pos as int,
            );
            RT::store_kv_cache_repr_reads_own_write(
                suffix_pre.1, suffix_pre.2,
                suffix_base_k, suffix_base_v, suffix_slots, j,
            );
            assert(full_pre.1[pos as int] == suffix_pre.1[j]);
            assert(full_pre.2[pos as int] == suffix_pre.2[j]);
        }
    }

    let bt = MACHINE::singleton_block_rows(full_len)[0];
    let pages = crate::proof::tensor::geometry::blocks_needed_for(full_len);
    assert(bt =~= MACHINE::contiguous_block_ids(pages));
    assert forall|pos: nat| #![auto]
        pos < full_len implies
            crate::proof::tensor::geometry::block_table_slot(bt, pos) == pos
    by {
        crate::proof::tensor::geometry::lemma_blocks_needed_covers_pos(pos, full_len);
        MACHINE::lemma_contiguous_block_table_slot(pages, pos);
    }
    let full_attended = MODEL::paged_attention_repr(
        full_pre.0, full_cache.0, full_cache.1,
        MACHINE::seq_lens_for_single(full_len),
        MACHINE::seq_lens_for_single(full_len),
        full_len, full_len, MACHINE::singleton_block_rows(full_len),
        extension.attention, crate::boundary::dense_layer_primitives::layer_attention_parameters_repr(common, extension.attention_scale),
    );
    let suffix_attended = MODEL::paged_attention_repr(
        suffix_pre.0, suffix_cache.0, suffix_cache.1,
        MACHINE::seq_lens_for_single(suffix_len),
        MACHINE::seq_lens_for_single(full_len),
        suffix_len, full_len, MACHINE::singleton_block_rows(full_len),
        extension.attention, crate::boundary::dense_layer_primitives::layer_attention_parameters_repr(common, extension.attention_scale),
    );
    assert forall|pos: nat|
        #![trigger crate::proof::tensor::geometry::block_table_slot(bt, pos)]
        pos < full_len ==> {
            let slot = crate::proof::tensor::geometry::block_table_slot(bt, pos);
            &&& crate::proof::tensor::geometry::slot_in_cache(full_cache.0, slot)
            &&& crate::proof::tensor::geometry::slot_in_cache(full_cache.1, slot)
            &&& crate::proof::tensor::geometry::slot_in_cache(suffix_cache.0, slot)
            &&& crate::proof::tensor::geometry::slot_in_cache(suffix_cache.1, slot)
            &&& crate::proof::tensor::geometry::cache_at(full_cache.0, slot)
                == crate::proof::tensor::geometry::cache_at(suffix_cache.0, slot)
            &&& crate::proof::tensor::geometry::cache_at(full_cache.1, slot)
                == crate::proof::tensor::geometry::cache_at(suffix_cache.1, slot)
        }
    by {}
    BI::paged_attention_singleton_suffix_invariance(
        full_pre.0, suffix_pre.0,
        full_cache.0, full_cache.1,
        suffix_cache.0, suffix_cache.1,
        bt, bt, full_len, prefix_len,
        extension.attention, crate::boundary::dense_layer_primitives::layer_attention_parameters_repr(common, extension.attention_scale),
    );
    assert(full_attended.subrange(prefix_len as int, full_len as int)
        == suffix_attended);

    BI::post_attention_and_mlp_subrange_invariance(
        common, extension, full_hidden, full_attended,
        prefix_len as int, full_len as int,
    );
    let full_post = MODEL::post_attention_and_mlp_repr(
        common, extension, full_hidden, full_attended,
    );
    let suffix_post = MODEL::post_attention_and_mlp_repr(
        common, extension, suffix_hidden, suffix_attended,
    );
    assert(full_post.subrange(prefix_len as int, full_len as int)
        == suffix_post);
    reveal(MODEL::decoder_layer_step_repr);
}

pub proof fn lemma_layer_chain_canonical_prefix_continuation(
    common_layers: Seq<LayerWeightsRepr>,
    extension_layers: Seq<FourNormGatedLayerExtensionRepr>,
    full_hidden: Tensor2D,
    suffix_hidden: Tensor2D,
    full_caches: Seq<(KVCacheLayerRepr, KVCacheLayerRepr)>,
    suffix_caches: Seq<(KVCacheLayerRepr, KVCacheLayerRepr)>,
    full_len: nat,
    prefix_len: nat,
    start: nat,
)
    requires
        RT::paged_attention_numeric_domain(),
        start <= common_layers.len(),
        extension_layers.len() == common_layers.len(),
        forall|layer: int| start <= layer < extension_layers.len() ==>
            layer_attention_config_valid(
                #[trigger] extension_layers[layer].attention,
            ),
        0 < prefix_len < full_len,
        crate::proof::tensor::geometry::blocks_needed_for(full_len) <= u64::MAX as nat,
        full_caches.len() >= common_layers.len(),
        suffix_caches.len() >= common_layers.len(),
        full_hidden.len() == full_len,
        suffix_hidden.len() == full_len - prefix_len,
        full_hidden.subrange(prefix_len as int, full_len as int)
            == suffix_hidden,
        CACHE_LAWS::cache_sequence_has_positions_in_range(
            full_caches, start, common_layers.len(), full_len,
        ),
        CACHE_LAWS::cache_sequence_has_positions_in_range(
            suffix_caches, start, common_layers.len(), full_len,
        ),
        ({
            let full_out = MODEL::layer_chain_repr(
                common_layers, extension_layers, full_hidden,
                crate::proof::tensor::geometry::positions_from(0, full_len),
                full_caches, MACHINE::slots_from(0, full_len),
                MACHINE::seq_lens_for_single(full_len),
                MACHINE::seq_lens_for_single(full_len),
                full_len, full_len,
                MACHINE::singleton_block_rows(full_len), start,
            );
            CACHE_LAWS::cache_sequence_has_positions_in_range(
                full_out.1, start, common_layers.len(), full_len,
            )
        }),
        forall|layer: int, pos: nat|
            #![trigger crate::proof::tensor::geometry::cache_at(suffix_caches[layer].0, pos)]
            #![trigger crate::proof::tensor::geometry::cache_at(suffix_caches[layer].1, pos)]
            start <= layer < common_layers.len() && pos < prefix_len ==> {
                let full_out = MODEL::layer_chain_repr(
                    common_layers, extension_layers, full_hidden,
                    crate::proof::tensor::geometry::positions_from(0, full_len),
                    full_caches, MACHINE::slots_from(0, full_len),
                    MACHINE::seq_lens_for_single(full_len),
                    MACHINE::seq_lens_for_single(full_len),
                    full_len, full_len,
                    MACHINE::singleton_block_rows(full_len), start,
                );
                &&& crate::proof::tensor::geometry::cache_at(suffix_caches[layer].0, pos)
                    == crate::proof::tensor::geometry::cache_at(full_out.1[layer].0, pos)
                &&& crate::proof::tensor::geometry::cache_at(suffix_caches[layer].1, pos)
                    == crate::proof::tensor::geometry::cache_at(full_out.1[layer].1, pos)
            },
    ensures ({
        let suffix_len = (full_len - prefix_len) as nat;
        let full_out = MODEL::layer_chain_repr(
            common_layers, extension_layers, full_hidden,
            crate::proof::tensor::geometry::positions_from(0, full_len),
            full_caches, MACHINE::slots_from(0, full_len),
            MACHINE::seq_lens_for_single(full_len),
            MACHINE::seq_lens_for_single(full_len),
            full_len, full_len, MACHINE::singleton_block_rows(full_len), start,
        );
        let suffix_out = MODEL::layer_chain_repr(
            common_layers, extension_layers, suffix_hidden,
            crate::proof::tensor::geometry::positions_from(prefix_len, suffix_len),
            suffix_caches, MACHINE::slots_from(prefix_len, suffix_len),
            MACHINE::seq_lens_for_single(suffix_len),
            MACHINE::seq_lens_for_single(full_len),
            suffix_len, full_len, MACHINE::singleton_block_rows(full_len), start,
        );
        &&& full_out.0.subrange(prefix_len as int, full_len as int)
            == suffix_out.0
        &&& CACHE_LAWS::cache_sequences_agree_on_prefix_in_range(
            full_out.1, suffix_out.1,
            start, common_layers.len(), full_len,
        )
    }),
    decreases (common_layers.len() - start) as nat,
{
    let suffix_len = (full_len - prefix_len) as nat;
    let full_positions = crate::proof::tensor::geometry::positions_from(0, full_len);
    let suffix_positions = crate::proof::tensor::geometry::positions_from(
        prefix_len, suffix_len,
    );
    let full_slots = MACHINE::slots_from(0, full_len);
    let suffix_slots = MACHINE::slots_from(prefix_len, suffix_len);
    let full_out = MODEL::layer_chain_repr(
        common_layers, extension_layers, full_hidden, full_positions,
        full_caches, full_slots,
        MACHINE::seq_lens_for_single(full_len),
        MACHINE::seq_lens_for_single(full_len),
        full_len, full_len, MACHINE::singleton_block_rows(full_len), start,
    );
    let suffix_out = MODEL::layer_chain_repr(
        common_layers, extension_layers, suffix_hidden, suffix_positions,
        suffix_caches, suffix_slots,
        MACHINE::seq_lens_for_single(suffix_len),
        MACHINE::seq_lens_for_single(full_len),
        suffix_len, full_len, MACHINE::singleton_block_rows(full_len), start,
    );
    MODEL::lemma_layer_chain_repr_shape(
        common_layers, extension_layers, full_hidden, full_positions,
        full_caches, full_slots,
        MACHINE::seq_lens_for_single(full_len),
        MACHINE::seq_lens_for_single(full_len),
        full_len, full_len, MACHINE::singleton_block_rows(full_len), start,
    );
    MODEL::lemma_layer_chain_repr_shape(
        common_layers, extension_layers, suffix_hidden, suffix_positions,
        suffix_caches, suffix_slots,
        MACHINE::seq_lens_for_single(suffix_len),
        MACHINE::seq_lens_for_single(full_len),
        suffix_len, full_len, MACHINE::singleton_block_rows(full_len), start,
    );
    if start >= common_layers.len() {
        assert(CACHE_LAWS::cache_sequences_agree_on_prefix_in_range(
            full_out.1, suffix_out.1,
            start, common_layers.len(), full_len,
        )) by {
            reveal(CACHE_LAWS::cache_sequences_agree_on_prefix_in_range);
        }
    } else {
        let layer = start as int;
        let full_step = MODEL::decoder_layer_step_repr(
            common_layers[layer], extension_layers[layer],
            full_hidden, full_positions,
            full_caches[layer].0, full_caches[layer].1, full_slots,
            MACHINE::seq_lens_for_single(full_len),
            MACHINE::seq_lens_for_single(full_len),
            full_len, full_len, MACHINE::singleton_block_rows(full_len),
        );
        let suffix_step = MODEL::decoder_layer_step_repr(
            common_layers[layer], extension_layers[layer],
            suffix_hidden, suffix_positions,
            suffix_caches[layer].0, suffix_caches[layer].1, suffix_slots,
            MACHINE::seq_lens_for_single(suffix_len),
            MACHINE::seq_lens_for_single(full_len),
            suffix_len, full_len, MACHINE::singleton_block_rows(full_len),
        );
        MODEL::lemma_decoder_layer_step_repr_shape(
            common_layers[layer], extension_layers[layer],
            full_hidden, full_positions,
            full_caches[layer].0, full_caches[layer].1, full_slots,
            MACHINE::seq_lens_for_single(full_len),
            MACHINE::seq_lens_for_single(full_len),
            full_len, full_len, MACHINE::singleton_block_rows(full_len),
        );
        MODEL::lemma_decoder_layer_step_repr_shape(
            common_layers[layer], extension_layers[layer],
            suffix_hidden, suffix_positions,
            suffix_caches[layer].0, suffix_caches[layer].1, suffix_slots,
            MACHINE::seq_lens_for_single(suffix_len),
            MACHINE::seq_lens_for_single(full_len),
            suffix_len, full_len, MACHINE::singleton_block_rows(full_len),
        );
        let full_next_caches = full_caches.update(layer, full_step.1);
        let suffix_next_caches = suffix_caches.update(layer, suffix_step.1);
        let full_tail = MODEL::layer_chain_repr(
            common_layers, extension_layers, full_step.0, full_positions,
            full_next_caches, full_slots,
            MACHINE::seq_lens_for_single(full_len),
            MACHINE::seq_lens_for_single(full_len),
            full_len, full_len, MACHINE::singleton_block_rows(full_len),
            (start + 1) as nat,
        );
        let suffix_tail = MODEL::layer_chain_repr(
            common_layers, extension_layers, suffix_step.0, suffix_positions,
            suffix_next_caches, suffix_slots,
            MACHINE::seq_lens_for_single(suffix_len),
            MACHINE::seq_lens_for_single(full_len),
            suffix_len, full_len, MACHINE::singleton_block_rows(full_len),
            (start + 1) as nat,
        );
        reveal(MODEL::layer_chain_repr);
        assert(full_out == full_tail);
        assert(suffix_out == suffix_tail);
        MODEL::lemma_layer_chain_cache_before_start_unchanged(
            common_layers, extension_layers, full_step.0, full_positions,
            full_next_caches, full_slots,
            MACHINE::seq_lens_for_single(full_len),
            MACHINE::seq_lens_for_single(full_len),
            full_len, full_len, MACHINE::singleton_block_rows(full_len),
            (start + 1) as nat, layer,
        );
        assert(full_out.1[layer] == full_step.1);
        assert forall|pos: nat|
            #![trigger crate::proof::tensor::geometry::cache_at(suffix_caches[layer].0, pos)]
            #![trigger crate::proof::tensor::geometry::cache_at(suffix_caches[layer].1, pos)]
            pos < prefix_len implies {
                &&& crate::proof::tensor::geometry::cache_at(suffix_caches[layer].0, pos)
                    == crate::proof::tensor::geometry::cache_at(full_step.1.0, pos)
                &&& crate::proof::tensor::geometry::cache_at(suffix_caches[layer].1, pos)
                    == crate::proof::tensor::geometry::cache_at(full_step.1.1, pos)
            }
        by {}
        lemma_decoder_layer_canonical_prefix_continuation(
            common_layers[layer], extension_layers[layer],
            full_hidden, suffix_hidden,
            full_caches[layer].0, full_caches[layer].1,
            suffix_caches[layer].0, suffix_caches[layer].1,
            full_len, prefix_len,
        );
        assert(full_step.0.subrange(prefix_len as int, full_len as int)
            == suffix_step.0);
        assert(CACHE_LAWS::cache_sequence_has_positions_in_range(
            full_next_caches, (start + 1) as nat,
            common_layers.len(), full_len,
        )) by {
            reveal(CACHE_LAWS::cache_sequence_has_positions_in_range);
            assert forall|later: int, pos: nat| #![auto]
                start + 1 <= later < common_layers.len()
                    && pos < full_len implies
                    crate::proof::tensor::geometry::slot_in_cache(
                        full_next_caches[later].0, pos,
                    )
                    && crate::proof::tensor::geometry::slot_in_cache(
                        full_next_caches[later].1, pos,
                    )
            by {
                assert(later != layer);
                assert(full_next_caches[later] == full_caches[later]);
            }
        }
        assert(CACHE_LAWS::cache_sequence_has_positions_in_range(
            suffix_next_caches, (start + 1) as nat,
            common_layers.len(), full_len,
        )) by {
            reveal(CACHE_LAWS::cache_sequence_has_positions_in_range);
            assert forall|later: int, pos: nat| #![auto]
                start + 1 <= later < common_layers.len()
                    && pos < full_len implies
                    crate::proof::tensor::geometry::slot_in_cache(
                        suffix_next_caches[later].0, pos,
                    )
                    && crate::proof::tensor::geometry::slot_in_cache(
                        suffix_next_caches[later].1, pos,
                    )
            by {
                assert(later != layer);
                assert(suffix_next_caches[later] == suffix_caches[later]);
            }
        }
        assert(CACHE_LAWS::cache_sequence_has_positions_in_range(
            full_tail.1, (start + 1) as nat,
            common_layers.len(), full_len,
        ));
        assert forall|later: int, pos: nat|
            #![trigger crate::proof::tensor::geometry::cache_at(suffix_next_caches[later].0, pos)]
            #![trigger crate::proof::tensor::geometry::cache_at(suffix_next_caches[later].1, pos)]
            start + 1 <= later < common_layers.len()
                && pos < prefix_len implies {
                &&& crate::proof::tensor::geometry::cache_at(
                    suffix_next_caches[later].0, pos,
                ) == crate::proof::tensor::geometry::cache_at(full_tail.1[later].0, pos)
                &&& crate::proof::tensor::geometry::cache_at(
                    suffix_next_caches[later].1, pos,
                ) == crate::proof::tensor::geometry::cache_at(full_tail.1[later].1, pos)
            }
        by {
            assert(start + 1 <= later);
            assert(layer == start);
            assert(later != layer);
            assert(suffix_next_caches[later] == suffix_caches[later]);
            assert(full_tail == full_out);
        }
        lemma_layer_chain_canonical_prefix_continuation(
            common_layers, extension_layers,
            full_step.0, suffix_step.0,
            full_next_caches, suffix_next_caches,
            full_len, prefix_len, (start + 1) as nat,
        );
        MODEL::lemma_layer_chain_cache_before_start_unchanged(
            common_layers, extension_layers, suffix_step.0, suffix_positions,
            suffix_next_caches, suffix_slots,
            MACHINE::seq_lens_for_single(suffix_len),
            MACHINE::seq_lens_for_single(full_len),
            suffix_len, full_len, MACHINE::singleton_block_rows(full_len),
            (start + 1) as nat, layer,
        );
        assert(suffix_out.1[layer] == suffix_step.1);
        assert forall|later: int, pos: nat| #![auto]
            start <= later < common_layers.len() && pos < full_len implies {
                &&& crate::proof::tensor::geometry::slot_in_cache(full_out.1[later].0, pos)
                &&& crate::proof::tensor::geometry::slot_in_cache(full_out.1[later].1, pos)
                &&& crate::proof::tensor::geometry::slot_in_cache(suffix_out.1[later].0, pos)
                &&& crate::proof::tensor::geometry::slot_in_cache(suffix_out.1[later].1, pos)
                &&& crate::proof::tensor::geometry::cache_at(full_out.1[later].0, pos)
                    == crate::proof::tensor::geometry::cache_at(suffix_out.1[later].0, pos)
                &&& crate::proof::tensor::geometry::cache_at(full_out.1[later].1, pos)
                    == crate::proof::tensor::geometry::cache_at(suffix_out.1[later].1, pos)
            }
        by {
            if later == layer {
                assert(crate::proof::tensor::geometry::cache_at(full_step.1.0, pos)
                    == crate::proof::tensor::geometry::cache_at(suffix_step.1.0, pos));
                assert(crate::proof::tensor::geometry::cache_at(full_step.1.1, pos)
                    == crate::proof::tensor::geometry::cache_at(suffix_step.1.1, pos));
            } else {
                assert(start + 1 <= later);
                assert(CACHE_LAWS::cache_sequences_agree_on_prefix_in_range(
                    full_tail.1, suffix_tail.1,
                    (start + 1) as nat, common_layers.len(), full_len,
                ));
            }
        }
        assert(CACHE_LAWS::cache_sequences_agree_on_prefix_in_range(
            full_out.1, suffix_out.1,
            start, common_layers.len(), full_len,
        )) by {
            reveal(CACHE_LAWS::cache_sequences_agree_on_prefix_in_range);
        }
    }
}

} // verus!
