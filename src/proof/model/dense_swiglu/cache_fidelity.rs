//! Semantic KV-cache fidelity for the neutral dense SwiGLU decoder.
//
// Two equivalent formulations describe prefix-cache fidelity:
//
//   A. `option_a_prefix_stability`: two cold reference forwards whose token
//      histories share a prefix produce the same stored K/V on that prefix.
//   B. `option_b_canonicality`: every K/V cell produced by a cold reference
//      forward equals a canonical per-position value, defined by forwarding
//      exactly the token prefix ending at that position.
//
// Both hold for arbitrary decoder depth. B gives a pointwise invariant for
// live requests, shared pages, and the engine's paged cache. Continuation
// theorems show that a suffix-only forward from a canonical prefix matches
// the same suffix of cold full prefill. These proofs use row-local kernel
// specs; this module adds no axiom or external body.

#[cfg(verus_only)]
use crate::{types::{BLOCK_SIZE_SPEC}, proof::tensor::geometry::{block_table_slot, blocks_needed_for, cache_at, positions_from, slot_in_cache}};
#[cfg(verus_only)]
use crate::proof::model::dense_swiglu::batch_invariance as BI;
#[cfg(verus_only)]
use crate::proof::model::layer_properties as DENSE_PROPS;
#[cfg(verus_only)]
use crate::proof::model::dense_swiglu::cache_semantics as CC;
#[cfg(verus_only)]
use crate::proof::model::dense_swiglu::layer_store_witnesses as PW;
#[cfg(verus_only)]
use crate::proof::model::dense_swiglu::relational as RELATIONAL;
#[cfg(verus_only)]
use crate::proof::model::dense_swiglu::semantics as BD;
use crate::{types::*, proof::model::types::*, proof::tensor::types::*};
#[cfg(verus_only)]
use crate::boundary::tensor_runtime as RT;
use vstd::prelude::*;

verus! {

// Full-prefill, empty-cache reference result, including every layer's K/V
// cache after the stores.  This is the cache analogue of
// `BD::reference_logits_last_row`.
pub open spec fn cold_reference_cache_reprs(
    config: DenseSwiGluForwardConfigRepr,
    wr: ModelWeightsRepr,
    tokens: IntTensor1D,
) -> Seq<(KVCacheLayerRepr, KVCacheLayerRepr)> {
    let n = tokens.len();
    CC::model_forward_kv_reprs(config,
        wr,
        tokens,
        positions_from(0, n),
        BD::synthetic_cache_reprs(n, wr.layers.len()),
        BD::slots_from(0, n),
        BD::seq_lens_for_single(n),
        BD::seq_lens_for_single(n),
        n,
        n,
        BD::singleton_block_rows(n),
    )
}

pub open spec fn cache_pair_has_position(
    caches: Seq<(KVCacheLayerRepr, KVCacheLayerRepr)>,
    layer: int,
    pos: nat,
) -> bool {
    0 <= layer < caches.len()
    && slot_in_cache(caches[layer].0, pos)
    && slot_in_cache(caches[layer].1, pos)
}

pub open spec fn cache_pair_at(
    caches: Seq<(KVCacheLayerRepr, KVCacheLayerRepr)>,
    layer: int,
    pos: nat,
) -> (Tensor1D, Tensor1D)
    recommends cache_pair_has_position(caches, layer, pos),
{
    (cache_at(caches[layer].0, pos), cache_at(caches[layer].1, pos))
}

pub open spec fn histories_share_prefix(
    left: IntTensor1D,
    right: IntTensor1D,
    upto: nat,
) -> bool {
    upto <= left.len()
    && upto <= right.len()
    && left.subrange(0, upto as int) == right.subrange(0, upto as int)
}

// The singleton reference block table stores logical block indices as `u64`.
// Without this explicit bound an unbounded spec-level sequence can make the
// `i as u64` construction wrap, so the reference geometry is not contiguous.
// Executable histories are far below this bound, but the theorem must say so.
pub open spec fn reference_history_supported(tokens: IntTensor1D) -> bool {
    blocks_needed_for(tokens.len()) <= u64::MAX as nat
}

// Runtime machines may carry one newly emitted token in `history` that has not
// yet been processed into KV (`cached_tokens == history.len() - 1`).  Fidelity
// only reads canonical prefixes through `cached_tokens`, so representability
// must be stated at that boundary rather than for the extra history token.
pub open spec fn cached_prefix_supported(cached_tokens: nat) -> bool {
    blocks_needed_for(cached_tokens) <= u64::MAX as nat
}

// Option A, specialized to one pair of histories and one prefix length.
pub open spec fn full_reference_caches_agree_on_prefix(
    config: DenseSwiGluForwardConfigRepr,
    wr: ModelWeightsRepr,
    left: IntTensor1D,
    right: IntTensor1D,
    upto: nat,
) -> bool {
    let lc = cold_reference_cache_reprs(config, wr, left);
    let rc = cold_reference_cache_reprs(config, wr, right);
    forall|layer: int, pos: nat|
        #![trigger cache_pair_at(lc, layer, pos), cache_pair_at(rc, layer, pos)]
        0 <= layer < wr.layers.len() && pos < upto ==> {
            &&& cache_pair_has_position(lc, layer, pos)
            &&& cache_pair_has_position(rc, layer, pos)
            &&& cache_pair_at(lc, layer, pos) == cache_pair_at(rc, layer, pos)
        }
}

// The global theorem required by Option A.  It is a predicate, not an axiom:
// callers do not get it unless a proof supplies it.
pub open spec fn option_a_prefix_stability(
    config: DenseSwiGluForwardConfigRepr,
    wr: ModelWeightsRepr,
) -> bool {
    forall|left: IntTensor1D, right: IntTensor1D, upto: nat|
        histories_share_prefix(left, right, upto)
        && reference_history_supported(left)
        && reference_history_supported(right) ==>
            #[trigger] full_reference_caches_agree_on_prefix(config, wr, left, right, upto)
}

// Option B's canonical value at `(layer, pos)`.  Only the causally relevant
// token prefix `[0, pos + 1)` is forwarded, so equal token prefixes select the
// same value definitionally.  The nontrivial theorem is that a longer forward
// stores this value at `pos`.
pub open spec fn canonical_kv_at(
    config: DenseSwiGluForwardConfigRepr,
    wr: ModelWeightsRepr,
    tokens: IntTensor1D,
    layer: int,
    pos: nat,
) -> (Tensor1D, Tensor1D)
    recommends 0 <= layer < wr.layers.len(), pos < tokens.len(),
{
    let prefix = tokens.subrange(0, pos as int + 1);
    cache_pair_at(cold_reference_cache_reprs(config, wr, prefix), layer, pos)
}

pub open spec fn canonical_kv_defined(
    config: DenseSwiGluForwardConfigRepr,
    wr: ModelWeightsRepr,
    tokens: IntTensor1D,
    layer: int,
    pos: nat,
) -> bool {
    &&& 0 <= layer < wr.layers.len()
    &&& pos < tokens.len()
    &&& cache_pair_has_position(
        cold_reference_cache_reprs(config, wr, tokens.subrange(0, pos as int + 1)),
        layer,
        pos,
    )
}

pub open spec fn reference_cache_is_canonical_at(
    config: DenseSwiGluForwardConfigRepr,
    wr: ModelWeightsRepr,
    tokens: IntTensor1D,
    layer: int,
    pos: nat,
) -> bool {
    let full = cold_reference_cache_reprs(config, wr, tokens);
    &&& canonical_kv_defined(config, wr, tokens, layer, pos)
    &&& cache_pair_has_position(full, layer, pos)
    &&& cache_pair_at(full, layer, pos) == canonical_kv_at(config, wr, tokens, layer, pos)
}

// The global theorem required by Option B.  As with Option A, this remains an
// explicit proof obligation rather than being assumed.
pub open spec fn option_b_canonicality(
    config: DenseSwiGluForwardConfigRepr,
    wr: ModelWeightsRepr,
) -> bool {
    forall|tokens: IntTensor1D, layer: int, pos: nat|
        reference_history_supported(tokens)
        && 0 <= layer < wr.layers.len() && pos < tokens.len() ==>
            #[trigger] reference_cache_is_canonical_at(config, wr, tokens, layer, pos)
}

// Canonical K/V depends only on the token prefix ending at `pos`.  Unlike the
// decoder-chain causal theorem, this is definitional and fully proved here.
pub proof fn lemma_canonical_kv_respects_shared_prefix(
    config: DenseSwiGluForwardConfigRepr,
    wr: ModelWeightsRepr,
    left: IntTensor1D,
    right: IntTensor1D,
    upto: nat,
    layer: int,
    pos: nat,
)
    requires
        histories_share_prefix(left, right, upto),
        0 <= layer < wr.layers.len(),
        pos < upto,
        canonical_kv_defined(config, wr, left, layer, pos),
        canonical_kv_defined(config, wr, right, layer, pos),
    ensures
        canonical_kv_at(config, wr, left, layer, pos)
            == canonical_kv_at(config, wr, right, layer, pos),
{
    let n = pos as int + 1;
    assert(left.subrange(0, n) =~= right.subrange(0, n)) by {
        assert forall|i: int| 0 <= i < n implies left[i] == right[i] by {
            assert(left.subrange(0, upto as int)[i]
                == right.subrange(0, upto as int)[i]);
        }
    }
}

// Pointwise form of B => A.  This is the useful proof rule for prefix sharing:
// establish canonicality independently for both requests, then equality follows
// from their shared tokens without a pairwise cache invariant.
pub proof fn lemma_prefix_stability_gives_canonicality_at(
    config: DenseSwiGluForwardConfigRepr,
    wr: ModelWeightsRepr,
    tokens: IntTensor1D,
    layer: int,
    pos: nat,
)
    requires
        option_a_prefix_stability(config, wr),
        reference_history_supported(tokens),
        0 <= layer < wr.layers.len(),
        pos < tokens.len(),
    ensures
        reference_cache_is_canonical_at(config, wr, tokens, layer, pos),
{
    let n = (pos + 1) as nat;
    let prefix = tokens.subrange(0, n as int);
    assert(prefix.len() == n);
    assert(prefix.subrange(0, n as int) =~= prefix);
    assert(tokens.subrange(0, n as int) == prefix);
    assert(histories_share_prefix(tokens, prefix, n));
    crate::proof::tensor::geometry::lemma_blocks_needed_monotone(prefix.len(), tokens.len());
    assert(reference_history_supported(prefix));
    assert(full_reference_caches_agree_on_prefix(config, wr, tokens, prefix, n));
    let full = cold_reference_cache_reprs(config, wr, tokens);
    let short = cold_reference_cache_reprs(config, wr, prefix);
    // Mention both triggered reads to instantiate the pointwise agreement.
    assert(cache_pair_at(full, layer, pos) == cache_pair_at(short, layer, pos));
    assert(cache_pair_has_position(full, layer, pos));
    assert(cache_pair_has_position(short, layer, pos));
    assert(canonical_kv_defined(config, wr, tokens, layer, pos));
    assert(canonical_kv_at(config, wr, tokens, layer, pos)
        == cache_pair_at(short, layer, pos));
}

// Layer 0's K/V is canonical for every history and position. Its K/V path is
// row-wise (embed -> rms_norm -> projections -> rotary/store), so the proof uses
// the established subrange lemmas and the store's read-own-write theorem.  The
// attention recurrence only becomes necessary when proving layer >= 1.
pub proof fn lemma_decoder_core_full_prefill_prefix_stable(
    config: DenseSwiGluForwardConfigRepr,
    wr: LayerWeightsRepr,
    full_norm: Tensor2D,
    prefix_norm: Tensor2D,
    full_residual: Tensor2D,
    prefix_residual: Tensor2D,
    full_base_k: KVCacheLayerRepr,
    full_base_v: KVCacheLayerRepr,
    prefix_base_k: KVCacheLayerRepr,
    prefix_base_v: KVCacheLayerRepr,
    full_len: nat,
    prefix_len: nat,
)
    requires
        RT::paged_attention_numeric_domain(),
        0 < prefix_len <= full_len,
        blocks_needed_for(full_len) <= u64::MAX as nat,
        full_norm.len() == full_len,
        prefix_norm.len() == prefix_len,
        full_residual.len() == full_len,
        prefix_residual.len() == prefix_len,
        full_norm.subrange(0, prefix_len as int) == prefix_norm,
        full_residual.subrange(0, prefix_len as int) == prefix_residual,
        forall|pos: nat| #![auto] pos < full_len ==>
            slot_in_cache(full_base_k, pos) && slot_in_cache(full_base_v, pos),
        forall|pos: nat| #![auto] pos < prefix_len ==>
            slot_in_cache(prefix_base_k, pos) && slot_in_cache(prefix_base_v, pos),
    ensures ({
        let full_positions = positions_from(0, full_len);
        let prefix_positions = positions_from(0, prefix_len);
        let full_slots = BD::slots_from(0, full_len);
        let prefix_slots = BD::slots_from(0, prefix_len);
        let full_kv = BD::layer_kv_update_repr(config,
            wr, full_norm, full_positions, full_base_k, full_base_v, full_slots,
        );
        let prefix_kv = BD::layer_kv_update_repr(config,
            wr, prefix_norm, prefix_positions,
            prefix_base_k, prefix_base_v, prefix_slots,
        );
        let full_out = BD::decoder_core_output_repr(config,
            wr, full_norm, full_residual, full_positions,
            full_base_k, full_base_v, full_slots,
            BD::seq_lens_for_single(full_len), BD::seq_lens_for_single(full_len),
            full_len, full_len, BD::singleton_block_rows(full_len),
        );
        let prefix_out = BD::decoder_core_output_repr(config,
            wr, prefix_norm, prefix_residual, prefix_positions,
            prefix_base_k, prefix_base_v, prefix_slots,
            BD::seq_lens_for_single(prefix_len), BD::seq_lens_for_single(prefix_len),
            prefix_len, prefix_len, BD::singleton_block_rows(prefix_len),
        );
        &&& forall|pos: nat| #![trigger cache_at(full_kv.0, pos), cache_at(prefix_kv.0, pos)]
            pos < prefix_len ==> {
                &&& slot_in_cache(full_kv.0, pos)
                &&& slot_in_cache(full_kv.1, pos)
                &&& slot_in_cache(prefix_kv.0, pos)
                &&& slot_in_cache(prefix_kv.1, pos)
                &&& cache_at(full_kv.0, pos) == cache_at(prefix_kv.0, pos)
                &&& cache_at(full_kv.1, pos) == cache_at(prefix_kv.1, pos)
            }
        &&& full_out.0.subrange(0, prefix_len as int) == prefix_out.0
        &&& full_out.1.subrange(0, prefix_len as int) == prefix_out.1
    }),
{
    broadcast use {
        BD::lemma_pre_attention_repr_shape,
        BD::lemma_post_attention_repr_shape,
        RT::lemma_qkv_linear_repr_shape,
        RT::lemma_view_as_kv_repr_shape,
        RT::lemma_paged_attention_repr_shape,
    };
    let full_positions = positions_from(0, full_len);
    let prefix_positions = positions_from(0, prefix_len);
    let full_slots = BD::slots_from(0, full_len);
    let prefix_slots = BD::slots_from(0, prefix_len);
    assert(full_positions.subrange(0, prefix_len as int) =~= prefix_positions);

    let full_pre = BD::pre_attention_repr(config, wr, full_norm, full_positions);
    let prefix_pre = BD::pre_attention_repr(config, wr, prefix_norm, prefix_positions);
    BD::lemma_pre_attention_repr_shape(
        config, wr, full_norm, full_positions,
    );
    BD::lemma_pre_attention_repr_shape(
        config, wr, prefix_norm, prefix_positions,
    );
    RELATIONAL::pre_attention_subrange_invariance(config,
        wr, full_norm, full_positions, 0, prefix_len as int,
    );
    assert(full_pre.0.len() == full_len);
    assert(full_pre.1.len() == full_len);
    assert(prefix_pre.0.len() == prefix_len);
    assert(prefix_pre.1.len() == prefix_len);
    assert(prefix_pre.0 == full_pre.0.subrange(0, prefix_len as int));
    assert(prefix_pre.1 == full_pre.1.subrange(0, prefix_len as int));

    let full_v_linear = RT::qkv_linear_repr(
        full_norm, wr.q_proj, wr.k_proj, wr.v_proj,
    ).2;
    let prefix_v_linear = RT::qkv_linear_repr(
        prefix_norm, wr.q_proj, wr.k_proj, wr.v_proj,
    ).2;
    DENSE_PROPS::qkv_linear_subrange_invariance(
        full_norm, wr.q_proj, wr.k_proj, wr.v_proj, 0, prefix_len as int,
    );
    assert(prefix_v_linear == full_v_linear.subrange(0, prefix_len as int));
    let full_v = RT::view_as_kv_repr(full_v_linear);
    let prefix_v = RT::view_as_kv_repr(prefix_v_linear);
    BI::view_as_kv_subrange_invariance(full_v_linear, 0, prefix_len as int);
    assert(prefix_v == full_v.subrange(0, prefix_len as int));

    let full_kv = RT::store_kv_cache_repr(
        full_pre.1, full_v, full_base_k, full_base_v, full_slots,
    );
    let prefix_kv = RT::store_kv_cache_repr(
        prefix_pre.1, prefix_v, prefix_base_k, prefix_base_v, prefix_slots,
    );
    assert(full_slots.len() == full_len);
    assert(prefix_slots.len() == prefix_len);
    assert forall|pos: nat| #![auto]
        pos < prefix_len implies {
            &&& slot_in_cache(full_kv.0, pos)
            &&& slot_in_cache(full_kv.1, pos)
            &&& slot_in_cache(prefix_kv.0, pos)
            &&& slot_in_cache(prefix_kv.1, pos)
            &&& cache_at(full_kv.0, pos) == cache_at(prefix_kv.0, pos)
            &&& cache_at(full_kv.1, pos) == cache_at(prefix_kv.1, pos)
        }
    by {
        assert(full_slots[pos as int] == pos as int);
        assert(prefix_slots[pos as int] == pos as int);
        assert forall|m: int| pos < m < full_slots.len()
            implies full_slots[m] != full_slots[pos as int] by {
        }
        assert forall|m: int| pos < m < prefix_slots.len()
            implies prefix_slots[m] != prefix_slots[pos as int] by {
        }
        RT::store_kv_cache_repr_reads_own_write(
            full_pre.1, full_v, full_base_k, full_base_v, full_slots, pos as int,
        );
        RT::store_kv_cache_repr_reads_own_write(
            prefix_pre.1, prefix_v, prefix_base_k, prefix_base_v,
            prefix_slots, pos as int,
        );
        assert(full_pre.1[pos as int] == prefix_pre.1[pos as int]);
        assert(full_v[pos as int] == prefix_v[pos as int]);
    }

    let full_bt = BD::singleton_block_rows(full_len)[0];
    let prefix_bt = BD::singleton_block_rows(prefix_len)[0];
    let full_pages = blocks_needed_for(full_len);
    let prefix_pages = blocks_needed_for(prefix_len);
    assert(full_bt =~= crate::proof::reference::request_machine::contiguous_block_ids(full_pages));
    assert(prefix_bt =~= crate::proof::reference::request_machine::contiguous_block_ids(prefix_pages));
    crate::proof::tensor::geometry::lemma_blocks_needed_monotone(prefix_len, full_len);
    assert(prefix_pages <= u64::MAX as nat);
    assert forall|pos: nat| #![auto] pos < prefix_len implies
        (pos as int) / (BLOCK_SIZE_SPEC as int) < full_bt.len()
        && (pos as int) / (BLOCK_SIZE_SPEC as int) < prefix_bt.len()
        && slot_in_cache(full_kv.0, block_table_slot(full_bt, pos))
        && slot_in_cache(full_kv.1, block_table_slot(full_bt, pos))
        && slot_in_cache(prefix_kv.0, block_table_slot(prefix_bt, pos))
        && slot_in_cache(prefix_kv.1, block_table_slot(prefix_bt, pos))
        && cache_at(full_kv.0, block_table_slot(full_bt, pos))
            == cache_at(prefix_kv.0, block_table_slot(prefix_bt, pos))
        && cache_at(full_kv.1, block_table_slot(full_bt, pos))
            == cache_at(prefix_kv.1, block_table_slot(prefix_bt, pos))
    by {
        crate::proof::tensor::geometry::lemma_blocks_needed_covers_pos(pos, full_len);
        crate::proof::tensor::geometry::lemma_blocks_needed_covers_pos(pos, prefix_len);
        crate::proof::reference::request_machine::lemma_contiguous_block_table_slot(full_pages, pos);
        crate::proof::reference::request_machine::lemma_contiguous_block_table_slot(prefix_pages, pos);
        assert(block_table_slot(full_bt, pos) == pos);
        assert(block_table_slot(prefix_bt, pos) == pos);
        assert(slot_in_cache(full_kv.0, pos));
        assert(slot_in_cache(full_kv.1, pos));
        assert(slot_in_cache(prefix_kv.0, pos));
        assert(slot_in_cache(prefix_kv.1, pos));
        assert(cache_at(full_kv.0, pos) == cache_at(prefix_kv.0, pos));
    }

    let full_attn = BI::paged_attention_singleton_repr(
        full_pre.0, full_kv.0, full_kv.1, full_len, full_len, full_bt,
        crate::boundary::dense_layer_primitives::layer_attention_parameters_repr(wr, AttentionScaleRepr::InverseSqrtHeadDim),
    );
    let prefix_attn = BI::paged_attention_singleton_repr(
        prefix_pre.0, prefix_kv.0, prefix_kv.1,
        prefix_len, prefix_len, prefix_bt,
        crate::boundary::dense_layer_primitives::layer_attention_parameters_repr(wr, AttentionScaleRepr::InverseSqrtHeadDim),
    );
    BI::paged_attention_singleton_prefix_invariance(
        full_pre.0, prefix_pre.0,
        full_kv.0, full_kv.1, prefix_kv.0, prefix_kv.1,
        full_bt, prefix_bt, full_len, prefix_len,
        crate::boundary::dense_layer_primitives::layer_attention_parameters_repr(wr, AttentionScaleRepr::InverseSqrtHeadDim),
    );
    assert(full_attn.subrange(0, prefix_len as int) == prefix_attn);

    let full_post = BD::post_attention_repr(config, wr, full_attn, full_residual);
    let prefix_post = BD::post_attention_repr(config, wr, prefix_attn, prefix_residual);
    RELATIONAL::post_attention_subrange_invariance(config,
        wr, full_attn, full_residual, 0, prefix_len as int,
    );
    assert(full_post.0.subrange(0, prefix_len as int) == prefix_post.0);
    assert(full_post.1.subrange(0, prefix_len as int) == prefix_post.1);

    let full_layer_kv = BD::layer_kv_update_repr(config,
        wr, full_norm, full_positions, full_base_k, full_base_v, full_slots,
    );
    let prefix_layer_kv = BD::layer_kv_update_repr(config,
        wr, prefix_norm, prefix_positions,
        prefix_base_k, prefix_base_v, prefix_slots,
    );
    assert(full_layer_kv == full_kv);
    assert(prefix_layer_kv == prefix_kv);
    assert forall|pos: nat|
        #![trigger cache_at(full_layer_kv.0, pos), cache_at(prefix_layer_kv.0, pos)]
        pos < prefix_len implies {
            &&& slot_in_cache(full_layer_kv.0, pos)
            &&& slot_in_cache(full_layer_kv.1, pos)
            &&& slot_in_cache(prefix_layer_kv.0, pos)
            &&& slot_in_cache(prefix_layer_kv.1, pos)
            &&& cache_at(full_layer_kv.0, pos) == cache_at(prefix_layer_kv.0, pos)
            &&& cache_at(full_layer_kv.1, pos) == cache_at(prefix_layer_kv.1, pos)
        }
    by {
        assert(cache_at(full_kv.0, pos) == cache_at(prefix_kv.0, pos));
    }
    let full_out = BD::decoder_core_output_repr(config,
        wr, full_norm, full_residual, full_positions,
        full_base_k, full_base_v, full_slots,
        BD::seq_lens_for_single(full_len), BD::seq_lens_for_single(full_len),
        full_len, full_len, BD::singleton_block_rows(full_len),
    );
    let prefix_out = BD::decoder_core_output_repr(config,
        wr, prefix_norm, prefix_residual, prefix_positions,
        prefix_base_k, prefix_base_v, prefix_slots,
        BD::seq_lens_for_single(prefix_len), BD::seq_lens_for_single(prefix_len),
        prefix_len, prefix_len, BD::singleton_block_rows(prefix_len),
    );
    assert(full_out == full_post);
    assert(prefix_out == prefix_post);
    assert(full_out.0.subrange(0, prefix_len as int) == prefix_out.0);
    assert(full_out.1.subrange(0, prefix_len as int) == prefix_out.1);
}

// Semantic continuation step for a nonempty cached prefix.  The `full` side
// computes every token from scratch.  The `suffix` side submits only
// `[prefix_len, full_len)`, writes those rows at their logical slots, and reads
// the earlier K/V through the same contiguous singleton block table.  If its
// base cache already agrees with the full layer result on the cached prefix,
// then both the complete post-store cache and the suffix hidden/residual rows
// agree.  This is the layer-local theorem needed to justify partial prefill;
// the prefix-cache origin proof supplies its cache premise at the engine level.
pub proof fn lemma_decoder_core_canonical_prefix_continuation(
    config: DenseSwiGluForwardConfigRepr,
    wr: LayerWeightsRepr,
    full_norm: Tensor2D,
    suffix_norm: Tensor2D,
    full_residual: Tensor2D,
    suffix_residual: Tensor2D,
    full_base_k: KVCacheLayerRepr,
    full_base_v: KVCacheLayerRepr,
    suffix_base_k: KVCacheLayerRepr,
    suffix_base_v: KVCacheLayerRepr,
    full_len: nat,
    prefix_len: nat,
)
    requires
        RT::paged_attention_numeric_domain(),
        0 < prefix_len < full_len,
        blocks_needed_for(full_len) <= u64::MAX as nat,
        full_norm.len() == full_len,
        suffix_norm.len() == full_len - prefix_len,
        full_residual.len() == full_len,
        suffix_residual.len() == full_len - prefix_len,
        full_norm.subrange(prefix_len as int, full_len as int) == suffix_norm,
        full_residual.subrange(prefix_len as int, full_len as int) == suffix_residual,
        forall|pos: nat| #![auto] pos < full_len ==>
            slot_in_cache(full_base_k, pos)
            && slot_in_cache(full_base_v, pos)
            && slot_in_cache(suffix_base_k, pos)
            && slot_in_cache(suffix_base_v, pos),
        forall|pos: nat|
            #![trigger cache_at(suffix_base_k, pos), cache_at(suffix_base_v, pos)]
            pos < prefix_len ==> {
            let full_kv = BD::layer_kv_update_repr(config,
                wr, full_norm, positions_from(0, full_len),
                full_base_k, full_base_v, BD::slots_from(0, full_len),
            );
            &&& cache_at(suffix_base_k, pos) == cache_at(full_kv.0, pos)
            &&& cache_at(suffix_base_v, pos) == cache_at(full_kv.1, pos)
        },
    ensures ({
        let suffix_len = (full_len - prefix_len) as nat;
        let full_positions = positions_from(0, full_len);
        let suffix_positions = positions_from(prefix_len, suffix_len);
        let full_slots = BD::slots_from(0, full_len);
        let suffix_slots = BD::slots_from(prefix_len, suffix_len);
        let full_kv = BD::layer_kv_update_repr(config,
            wr, full_norm, full_positions,
            full_base_k, full_base_v, full_slots,
        );
        let suffix_kv = BD::layer_kv_update_repr(config,
            wr, suffix_norm, suffix_positions,
            suffix_base_k, suffix_base_v, suffix_slots,
        );
        let full_out = BD::decoder_core_output_repr(config,
            wr, full_norm, full_residual, full_positions,
            full_base_k, full_base_v, full_slots,
            BD::seq_lens_for_single(full_len), BD::seq_lens_for_single(full_len),
            full_len, full_len, BD::singleton_block_rows(full_len),
        );
        let suffix_out = BD::decoder_core_output_repr(config,
            wr, suffix_norm, suffix_residual, suffix_positions,
            suffix_base_k, suffix_base_v, suffix_slots,
            BD::seq_lens_for_single(suffix_len), BD::seq_lens_for_single(full_len),
            suffix_len, full_len, BD::singleton_block_rows(full_len),
        );
        &&& forall|pos: nat|
            #![trigger cache_at(full_kv.0, pos), cache_at(suffix_kv.0, pos)]
            pos < full_len ==> {
                &&& slot_in_cache(full_kv.0, pos)
                &&& slot_in_cache(full_kv.1, pos)
                &&& slot_in_cache(suffix_kv.0, pos)
                &&& slot_in_cache(suffix_kv.1, pos)
                &&& cache_at(full_kv.0, pos) == cache_at(suffix_kv.0, pos)
                &&& cache_at(full_kv.1, pos) == cache_at(suffix_kv.1, pos)
            }
        &&& full_out.0.len() == full_len
        &&& full_out.1.len() == full_len
        &&& suffix_out.0.len() == suffix_len
        &&& suffix_out.1.len() == suffix_len
        &&& full_out.0.subrange(prefix_len as int, full_len as int) == suffix_out.0
        &&& full_out.1.subrange(prefix_len as int, full_len as int) == suffix_out.1
    }),
{
    broadcast use {
        BD::lemma_pre_attention_repr_shape,
        BD::lemma_post_attention_repr_shape,
        RT::lemma_qkv_linear_repr_shape,
        RT::lemma_view_as_kv_repr_shape,
        RT::lemma_paged_attention_repr_shape,
    };
    let suffix_len = (full_len - prefix_len) as nat;
    let full_positions = positions_from(0, full_len);
    let suffix_positions = positions_from(prefix_len, suffix_len);
    let full_slots = BD::slots_from(0, full_len);
    let suffix_slots = BD::slots_from(prefix_len, suffix_len);
    assert(full_positions.subrange(prefix_len as int, full_len as int)
        =~= suffix_positions);

    let full_pre = BD::pre_attention_repr(config, wr, full_norm, full_positions);
    let suffix_pre = BD::pre_attention_repr(config, wr, suffix_norm, suffix_positions);
    BD::lemma_pre_attention_repr_shape(
        config, wr, full_norm, full_positions,
    );
    BD::lemma_pre_attention_repr_shape(
        config, wr, suffix_norm, suffix_positions,
    );
    RELATIONAL::pre_attention_subrange_invariance(config,
        wr, full_norm, full_positions, prefix_len as int, full_len as int,
    );
    assert(suffix_pre.0 == full_pre.0.subrange(prefix_len as int, full_len as int));
    assert(suffix_pre.1 == full_pre.1.subrange(prefix_len as int, full_len as int));

    let full_v_linear = RT::qkv_linear_repr(
        full_norm, wr.q_proj, wr.k_proj, wr.v_proj,
    ).2;
    let suffix_v_linear = RT::qkv_linear_repr(
        suffix_norm, wr.q_proj, wr.k_proj, wr.v_proj,
    ).2;
    DENSE_PROPS::qkv_linear_subrange_invariance(
        full_norm, wr.q_proj, wr.k_proj, wr.v_proj,
        prefix_len as int, full_len as int,
    );
    assert(suffix_v_linear
        == full_v_linear.subrange(prefix_len as int, full_len as int));
    let full_v = RT::view_as_kv_repr(full_v_linear);
    let suffix_v = RT::view_as_kv_repr(suffix_v_linear);
    BI::view_as_kv_subrange_invariance(
        full_v_linear, prefix_len as int, full_len as int,
    );
    assert(suffix_v == full_v.subrange(prefix_len as int, full_len as int));

    let full_kv = RT::store_kv_cache_repr(
        full_pre.1, full_v, full_base_k, full_base_v, full_slots,
    );
    let suffix_kv = RT::store_kv_cache_repr(
        suffix_pre.1, suffix_v, suffix_base_k, suffix_base_v, suffix_slots,
    );
    assert forall|pos: nat| #![auto] pos < full_len implies {
        &&& slot_in_cache(full_kv.0, pos)
        &&& slot_in_cache(full_kv.1, pos)
        &&& slot_in_cache(suffix_kv.0, pos)
        &&& slot_in_cache(suffix_kv.1, pos)
        &&& cache_at(full_kv.0, pos) == cache_at(suffix_kv.0, pos)
        &&& cache_at(full_kv.1, pos) == cache_at(suffix_kv.1, pos)
    } by {
        if pos < prefix_len {
            assert(!suffix_slots.contains(pos as int)) by {
                if suffix_slots.contains(pos as int) {
                    let j = suffix_slots.index_of(pos as int);
                    assert(suffix_slots[j] == prefix_len as int + j);
                }
            }
            RT::store_kv_cache_repr_preserves_unwritten_slots(
                suffix_pre.1, suffix_v, suffix_base_k, suffix_base_v,
                suffix_slots, pos,
            );
            RT::store_kv_cache_repr_preserves_slot_in_cache(
                full_pre.1, full_v, full_base_k, full_base_v, full_slots, pos,
            );
            assert(cache_at(suffix_base_k, pos) == cache_at(full_kv.0, pos));
            assert(cache_at(suffix_base_v, pos) == cache_at(full_kv.1, pos));
        } else {
            let j = (pos - prefix_len) as int;
            assert(0 <= j < suffix_len);
            assert(full_slots[pos as int] == pos as int);
            assert(suffix_slots[j] == pos as int);
            assert forall|m: int| pos < m < full_slots.len()
                implies full_slots[m] != full_slots[pos as int] by {
            }
            assert forall|m: int| j < m < suffix_slots.len()
                implies suffix_slots[m] != suffix_slots[j] by {
            }
            RT::store_kv_cache_repr_reads_own_write(
                full_pre.1, full_v, full_base_k, full_base_v,
                full_slots, pos as int,
            );
            RT::store_kv_cache_repr_reads_own_write(
                suffix_pre.1, suffix_v, suffix_base_k, suffix_base_v,
                suffix_slots, j,
            );
            assert(full_pre.1[pos as int] == suffix_pre.1[j]);
            assert(full_v[pos as int] == suffix_v[j]);
        }
    }

    let bt = BD::singleton_block_rows(full_len)[0];
    let pages = blocks_needed_for(full_len);
    assert(bt =~= crate::proof::reference::request_machine::contiguous_block_ids(pages));
    assert forall|pos: nat| #![auto] pos < full_len implies
        (pos as int) / (BLOCK_SIZE_SPEC as int) < bt.len()
        && block_table_slot(bt, pos) == pos
    by {
        crate::proof::tensor::geometry::lemma_blocks_needed_covers_pos(pos, full_len);
        crate::proof::reference::request_machine::lemma_contiguous_block_table_slot(pages, pos);
    }

    let full_attn = BI::paged_attention_singleton_repr(
        full_pre.0, full_kv.0, full_kv.1, full_len, full_len, bt,
        crate::boundary::dense_layer_primitives::layer_attention_parameters_repr(wr, AttentionScaleRepr::InverseSqrtHeadDim),
    );
    let suffix_attn = BI::paged_attention_singleton_repr(
        suffix_pre.0, suffix_kv.0, suffix_kv.1, suffix_len, full_len, bt,
        crate::boundary::dense_layer_primitives::layer_attention_parameters_repr(wr, AttentionScaleRepr::InverseSqrtHeadDim),
    );
    assert forall|j: int| 0 <= j < suffix_len implies
        full_attn[prefix_len as int + j] == suffix_attn[j]
    by {
        let jf = (prefix_len as int + j) as nat;
        let js = j as nat;
        assert(full_pre.0[jf as int] == suffix_pre.0[js as int]);
        assert(full_len as int - full_len as int + jf as int
            == full_len as int - suffix_len as int + js as int);
        assert forall|pos: nat| #![auto]
            pos as int <= full_len as int - full_len as int + jf as int implies
                (pos as int) / (BLOCK_SIZE_SPEC as int) < bt.len()
                && slot_in_cache(full_kv.0, block_table_slot(bt, pos))
                && slot_in_cache(full_kv.1, block_table_slot(bt, pos))
                && slot_in_cache(suffix_kv.0, block_table_slot(bt, pos))
                && slot_in_cache(suffix_kv.1, block_table_slot(bt, pos))
                && cache_at(full_kv.0, block_table_slot(bt, pos))
                    == cache_at(suffix_kv.0, block_table_slot(bt, pos))
                && cache_at(full_kv.1, block_table_slot(bt, pos))
                    == cache_at(suffix_kv.1, block_table_slot(bt, pos))
        by {
            assert(pos < full_len);
            assert(block_table_slot(bt, pos) == pos);
        }
        BI::paged_attention_singleton_equivalence(
            full_pre.0, suffix_pre.0,
            full_kv.0, full_kv.1, suffix_kv.0, suffix_kv.1,
            full_len, full_len, suffix_len, full_len,
            bt, bt, jf, js,
            crate::boundary::dense_layer_primitives::layer_attention_parameters_repr(wr, AttentionScaleRepr::InverseSqrtHeadDim),
        );
    }
    assert(full_attn.subrange(prefix_len as int, full_len as int) =~= suffix_attn);

    let full_post = BD::post_attention_repr(config, wr, full_attn, full_residual);
    let suffix_post = BD::post_attention_repr(config, wr, suffix_attn, suffix_residual);
    RELATIONAL::post_attention_subrange_invariance(config,
        wr, full_attn, full_residual, prefix_len as int, full_len as int,
    );
    assert(full_post.0.subrange(prefix_len as int, full_len as int)
        == suffix_post.0);
    assert(full_post.1.subrange(prefix_len as int, full_len as int)
        == suffix_post.1);
}

pub open spec fn cache_sequence_has_positions_in_range(
    caches: Seq<(KVCacheLayerRepr, KVCacheLayerRepr)>,
    start: nat,
    end: nat,
    token_count: nat,
) -> bool {
    end <= caches.len()
    && forall|layer: int, pos: nat| #![auto]
        start <= layer < end && pos < token_count ==>
            slot_in_cache(caches[layer].0, pos)
            && slot_in_cache(caches[layer].1, pos)
}

pub open spec fn cache_sequences_agree_on_prefix_in_range(
    full: Seq<(KVCacheLayerRepr, KVCacheLayerRepr)>,
    prefix: Seq<(KVCacheLayerRepr, KVCacheLayerRepr)>,
    start: nat,
    end: nat,
    prefix_len: nat,
) -> bool {
    end <= full.len()
    && end <= prefix.len()
    && forall|layer: int, pos: nat| #![auto]
        start <= layer < end && pos < prefix_len ==> {
            &&& slot_in_cache(full[layer].0, pos)
            &&& slot_in_cache(full[layer].1, pos)
            &&& slot_in_cache(prefix[layer].0, pos)
            &&& slot_in_cache(prefix[layer].1, pos)
            &&& cache_at(full[layer].0, pos) == cache_at(prefix[layer].0, pos)
            &&& cache_at(full[layer].1, pos) == cache_at(prefix[layer].1, pos)
        }
}

// Lift the single-layer causal-prefix step through the recursive decoder chain.
// The result simultaneously carries hidden-state prefix equality (for the next
// layer) and K/V prefix equality (the semantic cache result for this layer).
pub proof fn lemma_layer_chain_full_prefill_prefix_stable(
    config: DenseSwiGluForwardConfigRepr,
    layers: Seq<LayerWeightsRepr>,
    full_hidden: Tensor2D,
    prefix_hidden: Tensor2D,
    full_residual: Tensor2D,
    prefix_residual: Tensor2D,
    full_kvs: Seq<(KVCacheLayerRepr, KVCacheLayerRepr)>,
    prefix_kvs: Seq<(KVCacheLayerRepr, KVCacheLayerRepr)>,
    full_len: nat,
    prefix_len: nat,
    start: nat,
)
    requires
        RT::paged_attention_numeric_domain(),
        start <= layers.len(),
        0 < prefix_len <= full_len,
        blocks_needed_for(full_len) <= u64::MAX as nat,
        full_kvs.len() >= layers.len(),
        prefix_kvs.len() >= layers.len(),
        full_hidden.len() == full_len,
        prefix_hidden.len() == prefix_len,
        full_residual.len() == full_len,
        prefix_residual.len() == prefix_len,
        full_hidden.subrange(0, prefix_len as int) == prefix_hidden,
        full_residual.subrange(0, prefix_len as int) == prefix_residual,
        cache_sequence_has_positions_in_range(
            full_kvs, start, layers.len(), full_len,
        ),
        cache_sequence_has_positions_in_range(
            prefix_kvs, start, layers.len(), prefix_len,
        ),
    ensures ({
        let full_positions = positions_from(0, full_len);
        let prefix_positions = positions_from(0, prefix_len);
        let full_slots = BD::slots_from(0, full_len);
        let prefix_slots = BD::slots_from(0, prefix_len);
        let full_out = BD::layer_chain_repr(config,
            layers, full_hidden, full_residual, full_positions,
            full_kvs, full_slots,
            BD::seq_lens_for_single(full_len), BD::seq_lens_for_single(full_len),
            full_len, full_len, BD::singleton_block_rows(full_len), start,
        );
        let prefix_out = BD::layer_chain_repr(config,
            layers, prefix_hidden, prefix_residual, prefix_positions,
            prefix_kvs, prefix_slots,
            BD::seq_lens_for_single(prefix_len), BD::seq_lens_for_single(prefix_len),
            prefix_len, prefix_len, BD::singleton_block_rows(prefix_len), start,
        );
        let full_post_kvs = CC::layer_chain_kv_reprs(config,
            layers, full_hidden, full_residual, full_positions,
            full_kvs, full_slots,
            BD::seq_lens_for_single(full_len), BD::seq_lens_for_single(full_len),
            full_len, full_len, BD::singleton_block_rows(full_len), start,
        );
        let prefix_post_kvs = CC::layer_chain_kv_reprs(config,
            layers, prefix_hidden, prefix_residual, prefix_positions,
            prefix_kvs, prefix_slots,
            BD::seq_lens_for_single(prefix_len), BD::seq_lens_for_single(prefix_len),
            prefix_len, prefix_len, BD::singleton_block_rows(prefix_len), start,
        );
        &&& full_out.0.subrange(0, prefix_len as int) == prefix_out.0
        &&& full_out.1.subrange(0, prefix_len as int) == prefix_out.1
        &&& cache_sequences_agree_on_prefix_in_range(
            full_post_kvs, prefix_post_kvs, start, layers.len(), prefix_len,
        )
    }),
    decreases (layers.len() - start) as nat,
{
    let full_positions = positions_from(0, full_len);
    let prefix_positions = positions_from(0, prefix_len);
    let full_slots = BD::slots_from(0, full_len);
    let prefix_slots = BD::slots_from(0, prefix_len);
    BD::lemma_layer_chain_repr_shape(config,
        layers, full_hidden, full_residual, full_positions,
        full_kvs, full_slots,
        BD::seq_lens_for_single(full_len), BD::seq_lens_for_single(full_len),
        full_len, full_len, BD::singleton_block_rows(full_len), start,
    );
    BD::lemma_layer_chain_repr_shape(config,
        layers, prefix_hidden, prefix_residual, prefix_positions,
        prefix_kvs, prefix_slots,
        BD::seq_lens_for_single(prefix_len), BD::seq_lens_for_single(prefix_len),
        prefix_len, prefix_len, BD::singleton_block_rows(prefix_len), start,
    );
    if start >= layers.len() {
        assert(full_hidden.subrange(0, prefix_len as int) == prefix_hidden);
        assert(full_residual.subrange(0, prefix_len as int) == prefix_residual);
        assert(cache_sequences_agree_on_prefix_in_range(
            full_kvs, prefix_kvs, start, layers.len(), prefix_len,
        ));
    } else {
        broadcast use {
            RT::lemma_add_rms_norm_repr_shape,
        };
        let layer = start as int;
        let layer_wr = layers[layer];
        let full_pre = BD::add_rms_norm_repr(config,
            full_hidden, full_residual, layer_wr.input_norm,
        );
        let prefix_pre = BD::add_rms_norm_repr(config,
            prefix_hidden, prefix_residual, layer_wr.input_norm,
        );
        BI::add_rms_norm_subrange_invariance(
            full_hidden, full_residual, layer_wr.input_norm,
            config.rms_norm_epsilon, 0, prefix_len as int,
        );
        assert(full_pre.0.subrange(0, prefix_len as int) == prefix_pre.0);
        assert(full_pre.1.subrange(0, prefix_len as int) == prefix_pre.1);
        assert(full_pre.0.len() == full_len && full_pre.1.len() == full_len);
        assert(prefix_pre.0.len() == prefix_len && prefix_pre.1.len() == prefix_len);

        lemma_decoder_core_full_prefill_prefix_stable(config,
            layer_wr,
            full_pre.0, prefix_pre.0, full_pre.1, prefix_pre.1,
            full_kvs[layer].0, full_kvs[layer].1,
            prefix_kvs[layer].0, prefix_kvs[layer].1,
            full_len, prefix_len,
        );

        let full_updated = CC::decoder_layer_kv_update_repr(config,
            layer_wr, full_hidden, full_residual, full_positions,
            full_kvs[layer].0, full_kvs[layer].1, full_slots,
        );
        let prefix_updated = CC::decoder_layer_kv_update_repr(config,
            layer_wr, prefix_hidden, prefix_residual, prefix_positions,
            prefix_kvs[layer].0, prefix_kvs[layer].1, prefix_slots,
        );
        let full_next = BD::decoder_layer_output_repr(config,
            layer_wr, full_hidden, full_residual, full_positions,
            full_kvs[layer].0, full_kvs[layer].1, full_slots,
            BD::seq_lens_for_single(full_len), BD::seq_lens_for_single(full_len),
            full_len, full_len, BD::singleton_block_rows(full_len),
        );
        let prefix_next = BD::decoder_layer_output_repr(config,
            layer_wr, prefix_hidden, prefix_residual, prefix_positions,
            prefix_kvs[layer].0, prefix_kvs[layer].1, prefix_slots,
            BD::seq_lens_for_single(prefix_len), BD::seq_lens_for_single(prefix_len),
            prefix_len, prefix_len, BD::singleton_block_rows(prefix_len),
        );
        BD::lemma_decoder_layer_output_repr_shape(
            config, layer_wr,
            full_hidden, full_residual, full_positions,
            full_kvs[layer].0, full_kvs[layer].1, full_slots,
            BD::seq_lens_for_single(full_len), BD::seq_lens_for_single(full_len),
            full_len, full_len, BD::singleton_block_rows(full_len),
        );
        BD::lemma_decoder_layer_output_repr_shape(
            config, layer_wr,
            prefix_hidden, prefix_residual, prefix_positions,
            prefix_kvs[layer].0, prefix_kvs[layer].1, prefix_slots,
            BD::seq_lens_for_single(prefix_len), BD::seq_lens_for_single(prefix_len),
            prefix_len, prefix_len, BD::singleton_block_rows(prefix_len),
        );
        assert(full_updated == BD::layer_kv_update_repr(config,
            layer_wr, full_pre.0, full_positions,
            full_kvs[layer].0, full_kvs[layer].1, full_slots,
        ));
        assert(prefix_updated == BD::layer_kv_update_repr(config,
            layer_wr, prefix_pre.0, prefix_positions,
            prefix_kvs[layer].0, prefix_kvs[layer].1, prefix_slots,
        ));
        assert(full_next.0.subrange(0, prefix_len as int) == prefix_next.0);
        assert(full_next.1.subrange(0, prefix_len as int) == prefix_next.1);
        assert(full_next.0.len() == full_len && full_next.1.len() == full_len);
        assert(prefix_next.0.len() == prefix_len && prefix_next.1.len() == prefix_len);

        let full_kvs2 = full_kvs.update(layer, full_updated);
        let prefix_kvs2 = prefix_kvs.update(layer, prefix_updated);
        assert(cache_sequence_has_positions_in_range(
            full_kvs2, (start + 1) as nat, layers.len(), full_len,
        )) by {
            assert forall|ell: int, pos: nat| #![auto]
                start + 1 <= ell < layers.len() && pos < full_len implies
                    slot_in_cache(full_kvs2[ell].0, pos)
                    && slot_in_cache(full_kvs2[ell].1, pos)
            by {
                assert(ell != layer);
                assert(full_kvs2[ell] == full_kvs[ell]);
            }
        }
        assert(cache_sequence_has_positions_in_range(
            prefix_kvs2, (start + 1) as nat, layers.len(), prefix_len,
        )) by {
            assert forall|ell: int, pos: nat| #![auto]
                start + 1 <= ell < layers.len() && pos < prefix_len implies
                    slot_in_cache(prefix_kvs2[ell].0, pos)
                    && slot_in_cache(prefix_kvs2[ell].1, pos)
            by {
                assert(ell != layer);
                assert(prefix_kvs2[ell] == prefix_kvs[ell]);
            }
        }
        lemma_layer_chain_full_prefill_prefix_stable(config,
            layers,
            full_next.0, prefix_next.0, full_next.1, prefix_next.1,
            full_kvs2, prefix_kvs2, full_len, prefix_len,
            (start + 1) as nat,
        );

        let full_post_kvs = CC::layer_chain_kv_reprs(config,
            layers, full_hidden, full_residual, full_positions,
            full_kvs, full_slots,
            BD::seq_lens_for_single(full_len), BD::seq_lens_for_single(full_len),
            full_len, full_len, BD::singleton_block_rows(full_len), start,
        );
        let prefix_post_kvs = CC::layer_chain_kv_reprs(config,
            layers, prefix_hidden, prefix_residual, prefix_positions,
            prefix_kvs, prefix_slots,
            BD::seq_lens_for_single(prefix_len), BD::seq_lens_for_single(prefix_len),
            prefix_len, prefix_len, BD::singleton_block_rows(prefix_len), start,
        );
        let full_tail_kvs = CC::layer_chain_kv_reprs(config,
            layers, full_next.0, full_next.1, full_positions,
            full_kvs2, full_slots,
            BD::seq_lens_for_single(full_len), BD::seq_lens_for_single(full_len),
            full_len, full_len, BD::singleton_block_rows(full_len),
            (start + 1) as nat,
        );
        let prefix_tail_kvs = CC::layer_chain_kv_reprs(config,
            layers, prefix_next.0, prefix_next.1, prefix_positions,
            prefix_kvs2, prefix_slots,
            BD::seq_lens_for_single(prefix_len), BD::seq_lens_for_single(prefix_len),
            prefix_len, prefix_len, BD::singleton_block_rows(prefix_len),
            (start + 1) as nat,
        );
        assert(full_post_kvs == full_tail_kvs);
        assert(prefix_post_kvs == prefix_tail_kvs);
        CC::lemma_layer_chain_kv_reprs_preserves_before_start(config,
            layers, full_next.0, full_next.1, full_positions,
            full_kvs2, full_slots,
            BD::seq_lens_for_single(full_len), BD::seq_lens_for_single(full_len),
            full_len, full_len, BD::singleton_block_rows(full_len),
            (start + 1) as nat, layer,
        );
        CC::lemma_layer_chain_kv_reprs_preserves_before_start(config,
            layers, prefix_next.0, prefix_next.1, prefix_positions,
            prefix_kvs2, prefix_slots,
            BD::seq_lens_for_single(prefix_len), BD::seq_lens_for_single(prefix_len),
            prefix_len, prefix_len, BD::singleton_block_rows(prefix_len),
            (start + 1) as nat, layer,
        );
        assert(full_post_kvs[layer] == full_updated);
        assert(prefix_post_kvs[layer] == prefix_updated);
        CC::lemma_layer_chain_kv_reprs_len(config,
            layers, full_hidden, full_residual, full_positions,
            full_kvs, full_slots,
            BD::seq_lens_for_single(full_len), BD::seq_lens_for_single(full_len),
            full_len, full_len, BD::singleton_block_rows(full_len), start,
        );
        CC::lemma_layer_chain_kv_reprs_len(config,
            layers, prefix_hidden, prefix_residual, prefix_positions,
            prefix_kvs, prefix_slots,
            BD::seq_lens_for_single(prefix_len), BD::seq_lens_for_single(prefix_len),
            prefix_len, prefix_len, BD::singleton_block_rows(prefix_len), start,
        );
        assert(full_post_kvs.len() == full_kvs.len());
        assert(prefix_post_kvs.len() == prefix_kvs.len());
        assert(layers.len() <= full_post_kvs.len());
        assert(layers.len() <= prefix_post_kvs.len());
        assert forall|ell: int, pos: nat| #![auto]
            start <= ell < layers.len() && pos < prefix_len implies {
                &&& slot_in_cache(full_post_kvs[ell].0, pos)
                &&& slot_in_cache(full_post_kvs[ell].1, pos)
                &&& slot_in_cache(prefix_post_kvs[ell].0, pos)
                &&& slot_in_cache(prefix_post_kvs[ell].1, pos)
                &&& cache_at(full_post_kvs[ell].0, pos)
                    == cache_at(prefix_post_kvs[ell].0, pos)
                &&& cache_at(full_post_kvs[ell].1, pos)
                    == cache_at(prefix_post_kvs[ell].1, pos)
            }
        by {
            if ell == layer {
                assert(cache_at(full_updated.0, pos) == cache_at(prefix_updated.0, pos));
                assert(slot_in_cache(full_updated.0, pos));
                assert(slot_in_cache(full_updated.1, pos));
                assert(slot_in_cache(prefix_updated.0, pos));
                assert(slot_in_cache(prefix_updated.1, pos));
                assert(cache_at(full_updated.1, pos) == cache_at(prefix_updated.1, pos));
            } else {
                assert(start + 1 <= ell);
                assert(cache_at(full_tail_kvs[ell].0, pos)
                    == cache_at(prefix_tail_kvs[ell].0, pos));
                assert(slot_in_cache(full_tail_kvs[ell].0, pos));
                assert(slot_in_cache(full_tail_kvs[ell].1, pos));
                assert(slot_in_cache(prefix_tail_kvs[ell].0, pos));
                assert(slot_in_cache(prefix_tail_kvs[ell].1, pos));
                assert(cache_at(full_tail_kvs[ell].1, pos)
                    == cache_at(prefix_tail_kvs[ell].1, pos));
            }
        }
        assert(cache_sequences_agree_on_prefix_in_range(
            full_post_kvs, prefix_post_kvs, start, layers.len(), prefix_len,
        ));

        let full_out = BD::layer_chain_repr(config,
            layers, full_hidden, full_residual, full_positions,
            full_kvs, full_slots,
            BD::seq_lens_for_single(full_len), BD::seq_lens_for_single(full_len),
            full_len, full_len, BD::singleton_block_rows(full_len), start,
        );
        let prefix_out = BD::layer_chain_repr(config,
            layers, prefix_hidden, prefix_residual, prefix_positions,
            prefix_kvs, prefix_slots,
            BD::seq_lens_for_single(prefix_len), BD::seq_lens_for_single(prefix_len),
            prefix_len, prefix_len, BD::singleton_block_rows(prefix_len), start,
        );
        let full_tail_out = BD::layer_chain_repr(config,
            layers, full_next.0, full_next.1, full_positions,
            full_kvs2, full_slots,
            BD::seq_lens_for_single(full_len), BD::seq_lens_for_single(full_len),
            full_len, full_len, BD::singleton_block_rows(full_len),
            (start + 1) as nat,
        );
        let prefix_tail_out = BD::layer_chain_repr(config,
            layers, prefix_next.0, prefix_next.1, prefix_positions,
            prefix_kvs2, prefix_slots,
            BD::seq_lens_for_single(prefix_len), BD::seq_lens_for_single(prefix_len),
            prefix_len, prefix_len, BD::singleton_block_rows(prefix_len),
            (start + 1) as nat,
        );
        let full_tail_out_base = BD::layer_chain_repr(config,
            layers, full_next.0, full_next.1, full_positions,
            full_kvs, full_slots,
            BD::seq_lens_for_single(full_len), BD::seq_lens_for_single(full_len),
            full_len, full_len, BD::singleton_block_rows(full_len),
            (start + 1) as nat,
        );
        let prefix_tail_out_base = BD::layer_chain_repr(config,
            layers, prefix_next.0, prefix_next.1, prefix_positions,
            prefix_kvs, prefix_slots,
            BD::seq_lens_for_single(prefix_len), BD::seq_lens_for_single(prefix_len),
            prefix_len, prefix_len, BD::singleton_block_rows(prefix_len),
            (start + 1) as nat,
        );
        assert forall|ell: int| start + 1 <= ell < layers.len() implies
            #[trigger] full_kvs[ell] == full_kvs2[ell] by {
            assert(ell != layer);
        }
        assert forall|ell: int| start + 1 <= ell < layers.len() implies
            #[trigger] prefix_kvs[ell] == prefix_kvs2[ell] by {
            assert(ell != layer);
        }
        RELATIONAL::lemma_layer_chain_repr_ignores_before_start(config,
            layers, full_next.0, full_next.1, full_positions,
            full_kvs, full_kvs2, full_slots,
            BD::seq_lens_for_single(full_len), BD::seq_lens_for_single(full_len),
            full_len, full_len, BD::singleton_block_rows(full_len),
            (start + 1) as nat,
        );
        RELATIONAL::lemma_layer_chain_repr_ignores_before_start(config,
            layers, prefix_next.0, prefix_next.1, prefix_positions,
            prefix_kvs, prefix_kvs2, prefix_slots,
            BD::seq_lens_for_single(prefix_len), BD::seq_lens_for_single(prefix_len),
            prefix_len, prefix_len, BD::singleton_block_rows(prefix_len),
            (start + 1) as nat,
        );
        assert(full_out == full_tail_out_base);
        assert(prefix_out == prefix_tail_out_base);
        assert(full_tail_out_base == full_tail_out);
        assert(prefix_tail_out_base == prefix_tail_out);
        assert(full_tail_out.0.subrange(0, prefix_len as int) == prefix_tail_out.0);
        assert(full_tail_out.1.subrange(0, prefix_len as int) == prefix_tail_out.1);
        assert(full_out.0.subrange(0, prefix_len as int) == prefix_out.0);
        assert(full_out.1.subrange(0, prefix_len as int) == prefix_out.1);
    }
}

// Lift canonical-prefix continuation through all subsequent decoder layers.
// Each suffix-side layer starts from the corresponding full-run post-store K/V
// on `[0, prefix_len)`.  The one-layer theorem establishes equality on the
// whole logical history and preserves suffix hidden/residual equality for the
// recursive layer.
pub proof fn lemma_layer_chain_canonical_prefix_continuation(
    config: DenseSwiGluForwardConfigRepr,
    layers: Seq<LayerWeightsRepr>,
    full_hidden: Tensor2D,
    suffix_hidden: Tensor2D,
    full_residual: Tensor2D,
    suffix_residual: Tensor2D,
    full_kvs: Seq<(KVCacheLayerRepr, KVCacheLayerRepr)>,
    suffix_kvs: Seq<(KVCacheLayerRepr, KVCacheLayerRepr)>,
    full_len: nat,
    prefix_len: nat,
    start: nat,
)
    requires
        RT::paged_attention_numeric_domain(),
        start <= layers.len(),
        0 < prefix_len < full_len,
        blocks_needed_for(full_len) <= u64::MAX as nat,
        full_kvs.len() >= layers.len(),
        suffix_kvs.len() >= layers.len(),
        full_hidden.len() == full_len,
        suffix_hidden.len() == full_len - prefix_len,
        full_residual.len() == full_len,
        suffix_residual.len() == full_len - prefix_len,
        full_hidden.subrange(prefix_len as int, full_len as int) == suffix_hidden,
        full_residual.subrange(prefix_len as int, full_len as int) == suffix_residual,
        cache_sequence_has_positions_in_range(
            full_kvs, start, layers.len(), full_len,
        ),
        cache_sequence_has_positions_in_range(
            suffix_kvs, start, layers.len(), full_len,
        ),
        ({
            let full_post = CC::layer_chain_kv_reprs(config,
                layers, full_hidden, full_residual,
                positions_from(0, full_len), full_kvs,
                BD::slots_from(0, full_len),
                BD::seq_lens_for_single(full_len),
                BD::seq_lens_for_single(full_len),
                full_len, full_len, BD::singleton_block_rows(full_len), start,
            );
            cache_sequence_has_positions_in_range(
                full_post, start, layers.len(), full_len,
            )
        }),
        forall|layer: int, pos: nat|
            #![trigger cache_at(suffix_kvs[layer].0, pos)]
            start <= layer < layers.len() && pos < prefix_len ==> {
                let full_post = CC::layer_chain_kv_reprs(config,
                    layers, full_hidden, full_residual,
                    positions_from(0, full_len), full_kvs,
                    BD::slots_from(0, full_len),
                    BD::seq_lens_for_single(full_len),
                    BD::seq_lens_for_single(full_len),
                    full_len, full_len, BD::singleton_block_rows(full_len), start,
                );
                &&& 0 <= layer < full_post.len()
                &&& slot_in_cache(full_post[layer].0, pos)
                &&& slot_in_cache(full_post[layer].1, pos)
                &&& cache_at(full_post[layer].0, pos)
                    == cache_at(suffix_kvs[layer].0, pos)
                &&& cache_at(suffix_kvs[layer].1, pos)
                    == cache_at(full_post[layer].1, pos)
            },
    ensures ({
        let suffix_len = (full_len - prefix_len) as nat;
        let full_positions = positions_from(0, full_len);
        let suffix_positions = positions_from(prefix_len, suffix_len);
        let full_slots = BD::slots_from(0, full_len);
        let suffix_slots = BD::slots_from(prefix_len, suffix_len);
        let full_out = BD::layer_chain_repr(config,
            layers, full_hidden, full_residual, full_positions,
            full_kvs, full_slots,
            BD::seq_lens_for_single(full_len), BD::seq_lens_for_single(full_len),
            full_len, full_len, BD::singleton_block_rows(full_len), start,
        );
        let suffix_out = BD::layer_chain_repr(config,
            layers, suffix_hidden, suffix_residual, suffix_positions,
            suffix_kvs, suffix_slots,
            BD::seq_lens_for_single(suffix_len), BD::seq_lens_for_single(full_len),
            suffix_len, full_len, BD::singleton_block_rows(full_len), start,
        );
        let full_post = CC::layer_chain_kv_reprs(config,
            layers, full_hidden, full_residual, full_positions,
            full_kvs, full_slots,
            BD::seq_lens_for_single(full_len), BD::seq_lens_for_single(full_len),
            full_len, full_len, BD::singleton_block_rows(full_len), start,
        );
        let suffix_post = CC::layer_chain_kv_reprs(config,
            layers, suffix_hidden, suffix_residual, suffix_positions,
            suffix_kvs, suffix_slots,
            BD::seq_lens_for_single(suffix_len), BD::seq_lens_for_single(full_len),
            suffix_len, full_len, BD::singleton_block_rows(full_len), start,
        );
        &&& full_out.0.len() == full_len
        &&& full_out.1.len() == full_len
        &&& suffix_out.0.len() == suffix_len
        &&& suffix_out.1.len() == suffix_len
        &&& full_out.0.subrange(prefix_len as int, full_len as int) == suffix_out.0
        &&& full_out.1.subrange(prefix_len as int, full_len as int) == suffix_out.1
        &&& cache_sequences_agree_on_prefix_in_range(
            full_post, suffix_post, start, layers.len(), full_len,
        )
    }),
    decreases (layers.len() - start) as nat,
{
    let suffix_len = (full_len - prefix_len) as nat;
    let full_positions = positions_from(0, full_len);
    let suffix_positions = positions_from(prefix_len, suffix_len);
    let full_slots = BD::slots_from(0, full_len);
    let suffix_slots = BD::slots_from(prefix_len, suffix_len);
    BD::lemma_layer_chain_repr_shape(config,
        layers, full_hidden, full_residual, full_positions,
        full_kvs, full_slots,
        BD::seq_lens_for_single(full_len), BD::seq_lens_for_single(full_len),
        full_len, full_len, BD::singleton_block_rows(full_len), start,
    );
    BD::lemma_layer_chain_repr_shape(config,
        layers, suffix_hidden, suffix_residual, suffix_positions,
        suffix_kvs, suffix_slots,
        BD::seq_lens_for_single(suffix_len), BD::seq_lens_for_single(full_len),
        suffix_len, full_len, BD::singleton_block_rows(full_len), start,
    );
    if start >= layers.len() {
        assert(cache_sequences_agree_on_prefix_in_range(
            full_kvs, suffix_kvs, start, layers.len(), full_len,
        ));
    } else {
        broadcast use {
            RT::lemma_add_rms_norm_repr_shape,
        };
        let layer = start as int;
        let layer_wr = layers[layer];
        let full_pre = BD::add_rms_norm_repr(config,
            full_hidden, full_residual, layer_wr.input_norm,
        );
        let suffix_pre = BD::add_rms_norm_repr(config,
            suffix_hidden, suffix_residual, layer_wr.input_norm,
        );
        BI::add_rms_norm_subrange_invariance(
            full_hidden, full_residual, layer_wr.input_norm,
            config.rms_norm_epsilon, prefix_len as int, full_len as int,
        );
        assert(full_pre.0.subrange(prefix_len as int, full_len as int)
            == suffix_pre.0);
        assert(full_pre.1.subrange(prefix_len as int, full_len as int)
            == suffix_pre.1);

        let full_updated = CC::decoder_layer_kv_update_repr(config,
            layer_wr, full_hidden, full_residual, full_positions,
            full_kvs[layer].0, full_kvs[layer].1, full_slots,
        );
        let suffix_updated = CC::decoder_layer_kv_update_repr(config,
            layer_wr, suffix_hidden, suffix_residual, suffix_positions,
            suffix_kvs[layer].0, suffix_kvs[layer].1, suffix_slots,
        );
        let full_next = BD::decoder_layer_output_repr(config,
            layer_wr, full_hidden, full_residual, full_positions,
            full_kvs[layer].0, full_kvs[layer].1, full_slots,
            BD::seq_lens_for_single(full_len), BD::seq_lens_for_single(full_len),
            full_len, full_len, BD::singleton_block_rows(full_len),
        );
        let suffix_next = BD::decoder_layer_output_repr(config,
            layer_wr, suffix_hidden, suffix_residual, suffix_positions,
            suffix_kvs[layer].0, suffix_kvs[layer].1, suffix_slots,
            BD::seq_lens_for_single(suffix_len), BD::seq_lens_for_single(full_len),
            suffix_len, full_len, BD::singleton_block_rows(full_len),
        );
        BD::lemma_decoder_layer_output_repr_shape(
            config, layer_wr,
            full_hidden, full_residual, full_positions,
            full_kvs[layer].0, full_kvs[layer].1, full_slots,
            BD::seq_lens_for_single(full_len), BD::seq_lens_for_single(full_len),
            full_len, full_len, BD::singleton_block_rows(full_len),
        );
        BD::lemma_decoder_layer_output_repr_shape(
            config, layer_wr,
            suffix_hidden, suffix_residual, suffix_positions,
            suffix_kvs[layer].0, suffix_kvs[layer].1, suffix_slots,
            BD::seq_lens_for_single(suffix_len), BD::seq_lens_for_single(full_len),
            suffix_len, full_len, BD::singleton_block_rows(full_len),
        );
        let full_kvs2 = full_kvs.update(layer, full_updated);
        let suffix_kvs2 = suffix_kvs.update(layer, suffix_updated);
        let full_post = CC::layer_chain_kv_reprs(config,
            layers, full_hidden, full_residual, full_positions,
            full_kvs, full_slots,
            BD::seq_lens_for_single(full_len), BD::seq_lens_for_single(full_len),
            full_len, full_len, BD::singleton_block_rows(full_len), start,
        );
        let full_tail = CC::layer_chain_kv_reprs(config,
            layers, full_next.0, full_next.1, full_positions,
            full_kvs2, full_slots,
            BD::seq_lens_for_single(full_len), BD::seq_lens_for_single(full_len),
            full_len, full_len, BD::singleton_block_rows(full_len),
            (start + 1) as nat,
        );
        assert(full_post == full_tail);
        CC::lemma_layer_chain_kv_reprs_len(config,
            layers, full_hidden, full_residual, full_positions,
            full_kvs, full_slots,
            BD::seq_lens_for_single(full_len), BD::seq_lens_for_single(full_len),
            full_len, full_len, BD::singleton_block_rows(full_len), start,
        );
        assert(full_post.len() == full_kvs.len());
        assert(0 <= layer < full_post.len());
        CC::lemma_layer_chain_kv_reprs_preserves_before_start(config,
            layers, full_next.0, full_next.1, full_positions,
            full_kvs2, full_slots,
            BD::seq_lens_for_single(full_len), BD::seq_lens_for_single(full_len),
            full_len, full_len, BD::singleton_block_rows(full_len),
            (start + 1) as nat, layer,
        );
        assert(full_post[layer] == full_updated);
        assert forall|pos: nat|
            #![trigger cache_at(suffix_kvs[layer].0, pos),
                cache_at(suffix_kvs[layer].1, pos)]
            pos < prefix_len implies {
                &&& cache_at(suffix_kvs[layer].0, pos)
                    == cache_at(full_updated.0, pos)
                &&& cache_at(suffix_kvs[layer].1, pos)
                    == cache_at(full_updated.1, pos)
            }
        by {
            assert(start <= layer < layers.len());
            assert(pos < prefix_len);
            assert(0 <= layer < full_post.len());
            assert(
                cache_at(suffix_kvs[layer].0, pos)
                    == cache_at(full_post[layer].0, pos)
                && cache_at(suffix_kvs[layer].1, pos)
                    == cache_at(full_post[layer].1, pos)
            );
        }
        lemma_decoder_core_canonical_prefix_continuation(config,
            layer_wr,
            full_pre.0, suffix_pre.0, full_pre.1, suffix_pre.1,
            full_kvs[layer].0, full_kvs[layer].1,
            suffix_kvs[layer].0, suffix_kvs[layer].1,
            full_len, prefix_len,
        );
        assert(full_updated == BD::layer_kv_update_repr(config,
            layer_wr, full_pre.0, full_positions,
            full_kvs[layer].0, full_kvs[layer].1, full_slots,
        ));
        assert(suffix_updated == BD::layer_kv_update_repr(config,
            layer_wr, suffix_pre.0, suffix_positions,
            suffix_kvs[layer].0, suffix_kvs[layer].1, suffix_slots,
        ));
        assert(full_next.0.subrange(prefix_len as int, full_len as int)
            == suffix_next.0);
        assert(full_next.1.subrange(prefix_len as int, full_len as int)
            == suffix_next.1);

        assert(cache_sequence_has_positions_in_range(
            full_kvs2, (start + 1) as nat, layers.len(), full_len,
        )) by {
            assert forall|ell: int, pos: nat| #![auto]
                start + 1 <= ell < layers.len() && pos < full_len implies
                    slot_in_cache(full_kvs2[ell].0, pos)
                    && slot_in_cache(full_kvs2[ell].1, pos)
            by {
                assert(ell != layer);
                assert(full_kvs2[ell] == full_kvs[ell]);
            }
        }
        assert(cache_sequence_has_positions_in_range(
            suffix_kvs2, (start + 1) as nat, layers.len(), full_len,
        )) by {
            assert forall|ell: int, pos: nat| #![auto]
                start + 1 <= ell < layers.len() && pos < full_len implies
                    slot_in_cache(suffix_kvs2[ell].0, pos)
                    && slot_in_cache(suffix_kvs2[ell].1, pos)
            by {
                assert(ell != layer);
                assert(suffix_kvs2[ell] == suffix_kvs[ell]);
            }
        }
        assert forall|ell: int, pos: nat|
            #![trigger cache_at(suffix_kvs2[ell].0, pos),
                cache_at(suffix_kvs2[ell].1, pos)]
            start + 1 <= ell < layers.len() && pos < prefix_len implies {
                let tail = CC::layer_chain_kv_reprs(config,
                    layers, full_next.0, full_next.1, full_positions,
                    full_kvs2, full_slots,
                    BD::seq_lens_for_single(full_len),
                    BD::seq_lens_for_single(full_len),
                    full_len, full_len, BD::singleton_block_rows(full_len),
                    (start + 1) as nat,
                );
                &&& cache_at(suffix_kvs2[ell].0, pos)
                    == cache_at(tail[ell].0, pos)
                &&& cache_at(suffix_kvs2[ell].1, pos)
                    == cache_at(tail[ell].1, pos)
            }
        by {
            assert(ell != layer);
            assert(suffix_kvs2[ell] == suffix_kvs[ell]);
            assert(cache_at(suffix_kvs[ell].0, pos)
                == cache_at(full_post[ell].0, pos));
            assert(cache_at(suffix_kvs[ell].1, pos)
                == cache_at(full_post[ell].1, pos));
        }
        lemma_layer_chain_canonical_prefix_continuation(config,
            layers,
            full_next.0, suffix_next.0, full_next.1, suffix_next.1,
            full_kvs2, suffix_kvs2, full_len, prefix_len,
            (start + 1) as nat,
        );

        let suffix_post = CC::layer_chain_kv_reprs(config,
            layers, suffix_hidden, suffix_residual, suffix_positions,
            suffix_kvs, suffix_slots,
            BD::seq_lens_for_single(suffix_len), BD::seq_lens_for_single(full_len),
            suffix_len, full_len, BD::singleton_block_rows(full_len), start,
        );
        let suffix_tail = CC::layer_chain_kv_reprs(config,
            layers, suffix_next.0, suffix_next.1, suffix_positions,
            suffix_kvs2, suffix_slots,
            BD::seq_lens_for_single(suffix_len), BD::seq_lens_for_single(full_len),
            suffix_len, full_len, BD::singleton_block_rows(full_len),
            (start + 1) as nat,
        );
        assert(suffix_post == suffix_tail);
        CC::lemma_layer_chain_kv_reprs_preserves_before_start(config,
            layers, suffix_next.0, suffix_next.1, suffix_positions,
            suffix_kvs2, suffix_slots,
            BD::seq_lens_for_single(suffix_len), BD::seq_lens_for_single(full_len),
            suffix_len, full_len, BD::singleton_block_rows(full_len),
            (start + 1) as nat, layer,
        );
        assert(suffix_post[layer] == suffix_updated);
        CC::lemma_layer_chain_kv_reprs_len(config,
            layers, full_hidden, full_residual, full_positions,
            full_kvs, full_slots,
            BD::seq_lens_for_single(full_len), BD::seq_lens_for_single(full_len),
            full_len, full_len, BD::singleton_block_rows(full_len), start,
        );
        CC::lemma_layer_chain_kv_reprs_len(config,
            layers, suffix_hidden, suffix_residual, suffix_positions,
            suffix_kvs, suffix_slots,
            BD::seq_lens_for_single(suffix_len), BD::seq_lens_for_single(full_len),
            suffix_len, full_len, BD::singleton_block_rows(full_len), start,
        );
        assert forall|ell: int, pos: nat| #![auto]
            start <= ell < layers.len() && pos < full_len implies {
                &&& slot_in_cache(full_post[ell].0, pos)
                &&& slot_in_cache(full_post[ell].1, pos)
                &&& slot_in_cache(suffix_post[ell].0, pos)
                &&& slot_in_cache(suffix_post[ell].1, pos)
                &&& cache_at(full_post[ell].0, pos)
                    == cache_at(suffix_post[ell].0, pos)
                &&& cache_at(full_post[ell].1, pos)
                    == cache_at(suffix_post[ell].1, pos)
            }
        by {
            if ell == layer {
                assert(cache_at(full_updated.0, pos)
                    == cache_at(suffix_updated.0, pos));
                assert(cache_at(full_updated.1, pos)
                    == cache_at(suffix_updated.1, pos));
            } else {
                assert(start + 1 <= ell);
                assert(cache_at(full_tail[ell].0, pos)
                    == cache_at(suffix_tail[ell].0, pos));
                assert(cache_at(full_tail[ell].1, pos)
                    == cache_at(suffix_tail[ell].1, pos));
            }
        }
        assert(cache_sequences_agree_on_prefix_in_range(
            full_post, suffix_post, start, layers.len(), full_len,
        ));

        let full_out = BD::layer_chain_repr(config,
            layers, full_hidden, full_residual, full_positions,
            full_kvs, full_slots,
            BD::seq_lens_for_single(full_len), BD::seq_lens_for_single(full_len),
            full_len, full_len, BD::singleton_block_rows(full_len), start,
        );
        let suffix_out = BD::layer_chain_repr(config,
            layers, suffix_hidden, suffix_residual, suffix_positions,
            suffix_kvs, suffix_slots,
            BD::seq_lens_for_single(suffix_len), BD::seq_lens_for_single(full_len),
            suffix_len, full_len, BD::singleton_block_rows(full_len), start,
        );
        let full_tail_out = BD::layer_chain_repr(config,
            layers, full_next.0, full_next.1, full_positions,
            full_kvs2, full_slots,
            BD::seq_lens_for_single(full_len), BD::seq_lens_for_single(full_len),
            full_len, full_len, BD::singleton_block_rows(full_len),
            (start + 1) as nat,
        );
        let suffix_tail_out = BD::layer_chain_repr(config,
            layers, suffix_next.0, suffix_next.1, suffix_positions,
            suffix_kvs2, suffix_slots,
            BD::seq_lens_for_single(suffix_len), BD::seq_lens_for_single(full_len),
            suffix_len, full_len, BD::singleton_block_rows(full_len),
            (start + 1) as nat,
        );
        let full_tail_out_base = BD::layer_chain_repr(config,
            layers, full_next.0, full_next.1, full_positions,
            full_kvs, full_slots,
            BD::seq_lens_for_single(full_len), BD::seq_lens_for_single(full_len),
            full_len, full_len, BD::singleton_block_rows(full_len),
            (start + 1) as nat,
        );
        let suffix_tail_out_base = BD::layer_chain_repr(config,
            layers, suffix_next.0, suffix_next.1, suffix_positions,
            suffix_kvs, suffix_slots,
            BD::seq_lens_for_single(suffix_len), BD::seq_lens_for_single(full_len),
            suffix_len, full_len, BD::singleton_block_rows(full_len),
            (start + 1) as nat,
        );
        assert forall|ell: int| start + 1 <= ell < layers.len() implies
            #[trigger] full_kvs[ell] == full_kvs2[ell] by {
            assert(ell != layer);
        }
        assert forall|ell: int| start + 1 <= ell < layers.len() implies
            #[trigger] suffix_kvs[ell] == suffix_kvs2[ell] by {
            assert(ell != layer);
        }
        RELATIONAL::lemma_layer_chain_repr_ignores_before_start(config,
            layers, full_next.0, full_next.1, full_positions,
            full_kvs, full_kvs2, full_slots,
            BD::seq_lens_for_single(full_len), BD::seq_lens_for_single(full_len),
            full_len, full_len, BD::singleton_block_rows(full_len),
            (start + 1) as nat,
        );
        RELATIONAL::lemma_layer_chain_repr_ignores_before_start(config,
            layers, suffix_next.0, suffix_next.1, suffix_positions,
            suffix_kvs, suffix_kvs2, suffix_slots,
            BD::seq_lens_for_single(suffix_len), BD::seq_lens_for_single(full_len),
            suffix_len, full_len, BD::singleton_block_rows(full_len),
            (start + 1) as nat,
        );
        assert(full_out == full_tail_out_base);
        assert(suffix_out == suffix_tail_out_base);
        assert(full_tail_out_base == full_tail_out);
        assert(suffix_tail_out_base == suffix_tail_out);
        assert(full_out.0.subrange(prefix_len as int, full_len as int)
            == suffix_out.0);
        assert(full_out.1.subrange(prefix_len as int, full_len as int)
            == suffix_out.1);
    }
}

// A complete cold-reference run agrees with the run truncated to one of its
// nonempty token prefixes, at every layer and every position in that prefix.
pub proof fn lemma_cold_reference_agrees_with_own_prefix(
    config: DenseSwiGluForwardConfigRepr,
    wr: ModelWeightsRepr,
    tokens: IntTensor1D,
    prefix_len: nat,
)
    requires
        RT::paged_attention_numeric_domain(),
        0 < prefix_len <= tokens.len(),
        reference_history_supported(tokens),
    ensures
        full_reference_caches_agree_on_prefix(config,
            wr, tokens, tokens.subrange(0, prefix_len as int), prefix_len,
        ),
{
    let full_len = tokens.len();
    let prefix_tokens = tokens.subrange(0, prefix_len as int);
    let full_cache = cold_reference_cache_reprs(config, wr, tokens);
    let prefix_cache = cold_reference_cache_reprs(config, wr, prefix_tokens);
    if wr.layers.len() == 0 {
        assert forall|layer: int, pos: nat|
            0 <= layer < wr.layers.len() && pos < prefix_len implies {
                &&& cache_pair_has_position(full_cache, layer, pos)
                &&& cache_pair_has_position(prefix_cache, layer, pos)
                &&& cache_pair_at(full_cache, layer, pos)
                    == cache_pair_at(prefix_cache, layer, pos)
            }
        by {
        }
    } else {
        broadcast use {
            RT::lemma_embed_repr_shape,
            RT::lemma_rms_norm_repr_shape,
        };
        let full_positions = positions_from(0, full_len);
        let prefix_positions = positions_from(0, prefix_len);
        let full_slots = BD::slots_from(0, full_len);
        let prefix_slots = BD::slots_from(0, prefix_len);
        let full_base = BD::synthetic_cache_reprs(full_len, wr.layers.len());
        let prefix_base = BD::synthetic_cache_reprs(prefix_len, wr.layers.len());
        let wr0 = wr.layers[0];

        let full_embed = RT::embed_repr(tokens, wr.embed_weight);
        let prefix_embed = RT::embed_repr(prefix_tokens, wr.embed_weight);
        BI::embed_subrange_invariance(tokens, wr.embed_weight, 0, prefix_len as int);
        assert(full_embed.subrange(0, prefix_len as int) == prefix_embed);
        let full_norm = BD::rms_norm_repr(config, full_embed, wr0.input_norm);
        let prefix_norm = BD::rms_norm_repr(config, prefix_embed, wr0.input_norm);
        BI::rms_norm_subrange_invariance(
            full_embed, wr0.input_norm, config.rms_norm_epsilon,
            0, prefix_len as int,
        );
        assert(full_norm.subrange(0, prefix_len as int) == prefix_norm);

        assert forall|pos: nat| #![auto] pos < full_len implies
            slot_in_cache(full_base[0].0, pos)
            && slot_in_cache(full_base[0].1, pos)
        by {
            BD::lemma_synthetic_cache_slot_in_cache(
                full_len, wr.layers.len(), 0, pos,
            );
        }
        assert forall|pos: nat| #![auto] pos < prefix_len implies
            slot_in_cache(prefix_base[0].0, pos)
            && slot_in_cache(prefix_base[0].1, pos)
        by {
            BD::lemma_synthetic_cache_slot_in_cache(
                prefix_len, wr.layers.len(), 0, pos,
            );
        }
        lemma_decoder_core_full_prefill_prefix_stable(config,
            wr0, full_norm, prefix_norm, full_embed, prefix_embed,
            full_base[0].0, full_base[0].1,
            prefix_base[0].0, prefix_base[0].1,
            full_len, prefix_len,
        );

        let full_first_update = CC::first_decoder_layer_kv_update_repr(config,
            wr0, full_embed, full_positions,
            full_base[0].0, full_base[0].1, full_slots,
        );
        let prefix_first_update = CC::first_decoder_layer_kv_update_repr(config,
            wr0, prefix_embed, prefix_positions,
            prefix_base[0].0, prefix_base[0].1, prefix_slots,
        );
        let full_first_out = BD::first_decoder_layer_output_repr(config,
            wr0, full_embed, full_positions,
            full_base[0].0, full_base[0].1, full_slots,
            BD::seq_lens_for_single(full_len), BD::seq_lens_for_single(full_len),
            full_len, full_len, BD::singleton_block_rows(full_len),
        );
        let prefix_first_out = BD::first_decoder_layer_output_repr(config,
            wr0, prefix_embed, prefix_positions,
            prefix_base[0].0, prefix_base[0].1, prefix_slots,
            BD::seq_lens_for_single(prefix_len), BD::seq_lens_for_single(prefix_len),
            prefix_len, prefix_len, BD::singleton_block_rows(prefix_len),
        );
        BD::lemma_first_decoder_layer_output_repr_shape(
            config, wr0,
            full_embed, full_positions,
            full_base[0].0, full_base[0].1, full_slots,
            BD::seq_lens_for_single(full_len), BD::seq_lens_for_single(full_len),
            full_len, full_len, BD::singleton_block_rows(full_len),
        );
        BD::lemma_first_decoder_layer_output_repr_shape(
            config, wr0,
            prefix_embed, prefix_positions,
            prefix_base[0].0, prefix_base[0].1, prefix_slots,
            BD::seq_lens_for_single(prefix_len), BD::seq_lens_for_single(prefix_len),
            prefix_len, prefix_len, BD::singleton_block_rows(prefix_len),
        );
        assert(full_first_update == BD::layer_kv_update_repr(config,
            wr0, full_norm, full_positions,
            full_base[0].0, full_base[0].1, full_slots,
        ));
        assert(prefix_first_update == BD::layer_kv_update_repr(config,
            wr0, prefix_norm, prefix_positions,
            prefix_base[0].0, prefix_base[0].1, prefix_slots,
        ));
        assert(full_first_out.0.subrange(0, prefix_len as int)
            == prefix_first_out.0);
        assert(full_first_out.1.subrange(0, prefix_len as int)
            == prefix_first_out.1);
        assert(full_first_out.0.len() == full_len);
        assert(full_first_out.1.len() == full_len);
        assert(prefix_first_out.0.len() == prefix_len);
        assert(prefix_first_out.1.len() == prefix_len);

        let full_kv1 = full_base.update(0, full_first_update);
        let prefix_kv1 = prefix_base.update(0, prefix_first_update);
        assert(cache_sequence_has_positions_in_range(
            full_kv1, 1, wr.layers.len(), full_len,
        )) by {
            assert forall|layer: int, pos: nat| #![auto]
                1 <= layer < wr.layers.len() && pos < full_len implies
                    slot_in_cache(full_kv1[layer].0, pos)
                    && slot_in_cache(full_kv1[layer].1, pos)
            by {
                assert(layer != 0);
                assert(full_kv1[layer] == full_base[layer]);
                BD::lemma_synthetic_cache_slot_in_cache(
                    full_len, wr.layers.len(), layer, pos,
                );
            }
        }
        assert(cache_sequence_has_positions_in_range(
            prefix_kv1, 1, wr.layers.len(), prefix_len,
        )) by {
            assert forall|layer: int, pos: nat| #![auto]
                1 <= layer < wr.layers.len() && pos < prefix_len implies
                    slot_in_cache(prefix_kv1[layer].0, pos)
                    && slot_in_cache(prefix_kv1[layer].1, pos)
            by {
                assert(layer != 0);
                assert(prefix_kv1[layer] == prefix_base[layer]);
                BD::lemma_synthetic_cache_slot_in_cache(
                    prefix_len, wr.layers.len(), layer, pos,
                );
            }
        }
        lemma_layer_chain_full_prefill_prefix_stable(config,
            wr.layers,
            full_first_out.0, prefix_first_out.0,
            full_first_out.1, prefix_first_out.1,
            full_kv1, prefix_kv1, full_len, prefix_len, 1,
        );

        let full_tail = CC::layer_chain_kv_reprs(config,
            wr.layers, full_first_out.0, full_first_out.1, full_positions,
            full_kv1, full_slots,
            BD::seq_lens_for_single(full_len), BD::seq_lens_for_single(full_len),
            full_len, full_len, BD::singleton_block_rows(full_len), 1,
        );
        let prefix_tail = CC::layer_chain_kv_reprs(config,
            wr.layers, prefix_first_out.0, prefix_first_out.1, prefix_positions,
            prefix_kv1, prefix_slots,
            BD::seq_lens_for_single(prefix_len), BD::seq_lens_for_single(prefix_len),
            prefix_len, prefix_len, BD::singleton_block_rows(prefix_len), 1,
        );
        assert(full_cache == full_tail);
        assert(prefix_cache == prefix_tail);
        CC::lemma_layer_chain_kv_reprs_preserves_before_start(config,
            wr.layers, full_first_out.0, full_first_out.1, full_positions,
            full_kv1, full_slots,
            BD::seq_lens_for_single(full_len), BD::seq_lens_for_single(full_len),
            full_len, full_len, BD::singleton_block_rows(full_len), 1, 0,
        );
        CC::lemma_layer_chain_kv_reprs_preserves_before_start(config,
            wr.layers, prefix_first_out.0, prefix_first_out.1, prefix_positions,
            prefix_kv1, prefix_slots,
            BD::seq_lens_for_single(prefix_len), BD::seq_lens_for_single(prefix_len),
            prefix_len, prefix_len, BD::singleton_block_rows(prefix_len), 1, 0,
        );
        CC::lemma_layer_chain_kv_reprs_len(config,
            wr.layers, full_first_out.0, full_first_out.1, full_positions,
            full_kv1, full_slots,
            BD::seq_lens_for_single(full_len), BD::seq_lens_for_single(full_len),
            full_len, full_len, BD::singleton_block_rows(full_len), 1,
        );
        CC::lemma_layer_chain_kv_reprs_len(config,
            wr.layers, prefix_first_out.0, prefix_first_out.1, prefix_positions,
            prefix_kv1, prefix_slots,
            BD::seq_lens_for_single(prefix_len), BD::seq_lens_for_single(prefix_len),
            prefix_len, prefix_len, BD::singleton_block_rows(prefix_len), 1,
        );
        assert(full_cache.len() == wr.layers.len());
        assert(prefix_cache.len() == wr.layers.len());
        assert(full_cache[0] == full_first_update);
        assert(prefix_cache[0] == prefix_first_update);
        assert forall|layer: int, pos: nat|
            0 <= layer < wr.layers.len() && pos < prefix_len implies {
                &&& cache_pair_has_position(full_cache, layer, pos)
                &&& cache_pair_has_position(prefix_cache, layer, pos)
                &&& cache_pair_at(full_cache, layer, pos)
                    == cache_pair_at(prefix_cache, layer, pos)
            }
        by {
            if layer == 0 {
                assert(cache_at(full_first_update.0, pos)
                    == cache_at(prefix_first_update.0, pos));
                assert(slot_in_cache(full_first_update.0, pos));
                assert(slot_in_cache(full_first_update.1, pos));
                assert(slot_in_cache(prefix_first_update.0, pos));
                assert(slot_in_cache(prefix_first_update.1, pos));
                assert(cache_at(full_first_update.1, pos)
                    == cache_at(prefix_first_update.1, pos));
            } else {
                assert(1 <= layer);
                assert(cache_at(full_tail[layer].0, pos)
                    == cache_at(prefix_tail[layer].0, pos));
                assert(slot_in_cache(full_tail[layer].0, pos));
                assert(slot_in_cache(full_tail[layer].1, pos));
                assert(slot_in_cache(prefix_tail[layer].0, pos));
                assert(slot_in_cache(prefix_tail[layer].1, pos));
                assert(cache_at(full_tail[layer].1, pos)
                    == cache_at(prefix_tail[layer].1, pos));
            }
        }
    }
}

// Option A for arbitrary model depth.  Reduce both histories to their common
// prefix and compose the two own-prefix results above.
pub proof fn lemma_all_layers_option_a_prefix_stability(
    config: DenseSwiGluForwardConfigRepr,
    wr: ModelWeightsRepr,
)
    requires RT::paged_attention_numeric_domain(),
    ensures option_a_prefix_stability(config, wr),
{
    assert forall|left: IntTensor1D, right: IntTensor1D, upto: nat|
        histories_share_prefix(left, right, upto)
        && reference_history_supported(left)
        && reference_history_supported(right) implies
            full_reference_caches_agree_on_prefix(config, wr, left, right, upto)
    by {
        if upto == 0 {
            let lc = cold_reference_cache_reprs(config, wr, left);
            let rc = cold_reference_cache_reprs(config, wr, right);
            assert forall|layer: int, pos: nat|
                0 <= layer < wr.layers.len() && pos < upto implies {
                    &&& cache_pair_has_position(lc, layer, pos)
                    &&& cache_pair_has_position(rc, layer, pos)
                    &&& cache_pair_at(lc, layer, pos) == cache_pair_at(rc, layer, pos)
                }
            by {
            }
        } else {
            let common = left.subrange(0, upto as int);
            assert(common == right.subrange(0, upto as int));
            lemma_cold_reference_agrees_with_own_prefix(config, wr, left, upto);
            lemma_cold_reference_agrees_with_own_prefix(config, wr, right, upto);
            let lc = cold_reference_cache_reprs(config, wr, left);
            let rc = cold_reference_cache_reprs(config, wr, right);
            let cc = cold_reference_cache_reprs(config, wr, common);
            assert forall|layer: int, pos: nat|
                0 <= layer < wr.layers.len() && pos < upto implies {
                    &&& cache_pair_has_position(lc, layer, pos)
                    &&& cache_pair_has_position(rc, layer, pos)
                    &&& cache_pair_at(lc, layer, pos) == cache_pair_at(rc, layer, pos)
                }
            by {
                assert(cache_pair_at(lc, layer, pos) == cache_pair_at(cc, layer, pos));
                assert(cache_pair_at(rc, layer, pos) == cache_pair_at(cc, layer, pos));
                assert(cache_pair_has_position(lc, layer, pos));
                assert(cache_pair_has_position(rc, layer, pos));
            }
        }
    }
}

// Canonical per-position K/V for arbitrary model depth, derived from the
// full-cache theorem.
pub proof fn lemma_all_layers_option_b_canonicality(
    config: DenseSwiGluForwardConfigRepr,
    wr: ModelWeightsRepr,
)
    requires RT::paged_attention_numeric_domain(),
    ensures option_b_canonicality(config, wr),
{
    lemma_all_layers_option_a_prefix_stability(config, wr);
    assert forall|tokens: IntTensor1D, layer: int, pos: nat|
        reference_history_supported(tokens)
        && 0 <= layer < wr.layers.len() && pos < tokens.len() implies
            reference_cache_is_canonical_at(config, wr, tokens, layer, pos)
    by {
        lemma_prefix_stability_gives_canonicality_at(config, wr, tokens, layer, pos);
    }
}

// A cold full-prefill cache is therefore a concrete witness of the selected
// pointwise invariant.  This is the establishment lemma needed by the fresh
// prefill branch once its machine cache is identified with the reference store.
pub proof fn lemma_cold_reference_cache_matches_canonical(
    config: DenseSwiGluForwardConfigRepr,
    wr: ModelWeightsRepr,
    tokens: IntTensor1D,
)
    requires
        RT::paged_attention_numeric_domain(),
        reference_history_supported(tokens),
    ensures cache_reprs_match_canonical(config,
        cold_reference_cache_reprs(config, wr, tokens), wr, tokens, tokens.len(),
    ),
{
    lemma_all_layers_option_b_canonicality(config, wr);
    let caches = cold_reference_cache_reprs(config, wr, tokens);
    let n = tokens.len();
    reveal(reference_history_supported);
    reveal(cached_prefix_supported);
    assert(cached_prefix_supported(n));
    let base = BD::synthetic_cache_reprs(n, wr.layers.len());
    CC::lemma_model_forward_kv_reprs_len(config,
        wr, tokens, positions_from(0, n), base, BD::slots_from(0, n),
        BD::seq_lens_for_single(n), BD::seq_lens_for_single(n),
        n, n, BD::singleton_block_rows(n),
    );
    assert(caches.len() == wr.layers.len());
    assert forall|layer: int, pos: nat|
        0 <= layer < wr.layers.len() && pos < n implies {
            &&& cache_pair_has_position(caches, layer, pos)
            &&& canonical_kv_defined(config, wr, tokens, layer, pos)
            &&& cache_pair_at(caches, layer, pos)
                == canonical_kv_at(config, wr, tokens, layer, pos)
        }
    by {
        assert(reference_cache_is_canonical_at(config, wr, tokens, layer, pos));
    }
    reveal(cache_reprs_match_canonical);
}

// Compact singleton execution of only the uncached suffix, initialized with
// the canonical cold cache for the complete token history.  Using the complete
// cache as the base is harmless: suffix slots are overwritten before attention
// reads them, while prefix slots supply exactly the canonical cached values.
pub open spec fn canonical_prefix_continuation_cache_reprs(
    config: DenseSwiGluForwardConfigRepr,
    wr: ModelWeightsRepr,
    tokens: IntTensor1D,
    prefix_len: nat,
) -> Seq<(KVCacheLayerRepr, KVCacheLayerRepr)>
    recommends prefix_len < tokens.len(),
{
    let n = tokens.len();
    let q = (n - prefix_len) as nat;
    CC::model_forward_kv_reprs(config,
        wr,
        tokens.subrange(prefix_len as int, n as int),
        positions_from(prefix_len, q),
        cold_reference_cache_reprs(config, wr, tokens),
        BD::slots_from(prefix_len, q),
        BD::seq_lens_for_single(q),
        BD::seq_lens_for_single(n),
        q,
        n,
        BD::singleton_block_rows(n),
    )
}

pub open spec fn canonical_prefix_continuation_logits_repr(
    config: DenseSwiGluForwardConfigRepr,
    wr: ModelWeightsRepr,
    tokens: IntTensor1D,
    prefix_len: nat,
) -> Tensor2D
    recommends prefix_len < tokens.len(),
{
    let n = tokens.len();
    let q = (n - prefix_len) as nat;
    BD::model_forward_logits_repr(config,
        wr,
        tokens.subrange(prefix_len as int, n as int),
        positions_from(prefix_len, q),
        cold_reference_cache_reprs(config, wr, tokens),
        BD::slots_from(prefix_len, q),
        BD::seq_lens_for_single(q),
        BD::seq_lens_for_single(n),
        q,
        n,
        BD::singleton_block_rows(n),
    )
}

// Model-level cached-partial semantic theorem for compact canonical geometry.
// A suffix-only forward from a nonempty canonical prefix produces exactly the
// cold full-prefill suffix logits and leaves every logical K/V cell equal to
// the cold full-prefill cache.  The proof uses only the existing row-local
// kernel specifications; no new kernel axiom is introduced.
pub proof fn lemma_canonical_prefix_continuation_matches_cold_prefill(
    config: DenseSwiGluForwardConfigRepr,
    wr: ModelWeightsRepr,
    tokens: IntTensor1D,
    prefix_len: nat,
)
    requires
        RT::paged_attention_numeric_domain(),
        0 < prefix_len < tokens.len(),
        reference_history_supported(tokens),
    ensures ({
        let n = tokens.len();
        let cold_base = BD::synthetic_cache_reprs(n, wr.layers.len());
        let cold_logits = BD::model_forward_logits_repr(config,
            wr, tokens, positions_from(0, n), cold_base,
            BD::slots_from(0, n),
            BD::seq_lens_for_single(n), BD::seq_lens_for_single(n),
            n, n, BD::singleton_block_rows(n),
        );
        let cold_cache = cold_reference_cache_reprs(config, wr, tokens);
        let continuation_cache =
            canonical_prefix_continuation_cache_reprs(config, wr, tokens, prefix_len);
        let continuation_logits =
            canonical_prefix_continuation_logits_repr(config, wr, tokens, prefix_len);
        &&& cold_logits.len() == n
        &&& continuation_logits.len() == n - prefix_len
        &&& cold_logits.subrange(prefix_len as int, n as int)
            == continuation_logits
        &&& cache_sequences_agree_on_prefix_in_range(
            cold_cache, continuation_cache, 0, wr.layers.len(), n,
        )
    }),
{
    reveal(BD::model_forward_logits_repr);
    let n = tokens.len();
    let q = (n - prefix_len) as nat;
    let suffix_tokens = tokens.subrange(prefix_len as int, n as int);
    let full_positions = positions_from(0, n);
    let suffix_positions = positions_from(prefix_len, q);
    let full_slots = BD::slots_from(0, n);
    let suffix_slots = BD::slots_from(prefix_len, q);
    let full_cu = BD::seq_lens_for_single(n);
    let suffix_cu_q = BD::seq_lens_for_single(q);
    let bt = BD::singleton_block_rows(n);
    let full_base = BD::synthetic_cache_reprs(n, wr.layers.len());
    let cold_cache = cold_reference_cache_reprs(config, wr, tokens);
    let continuation_cache =
        canonical_prefix_continuation_cache_reprs(config, wr, tokens, prefix_len);
    let cold_logits = BD::model_forward_logits_repr(config,
        wr, tokens, full_positions, full_base, full_slots,
        full_cu, full_cu, n, n, bt,
    );
    lemma_cold_reference_cache_matches_canonical(config, wr, tokens);
    assert(cold_cache.len() == wr.layers.len());
    let continuation_logits = BD::model_forward_logits_repr(config,
        wr, suffix_tokens, suffix_positions, cold_cache, suffix_slots,
        suffix_cu_q, full_cu, q, n, bt,
    );
    CC::lemma_model_forward_kv_reprs_len(config,
        wr, suffix_tokens, suffix_positions, cold_cache, suffix_slots,
        suffix_cu_q, full_cu, q, n, bt,
    );
    assert(continuation_cache.len() == cold_cache.len());
    assert forall|layer: int, pos: nat| #![auto]
        0 <= layer < wr.layers.len() && pos < n implies
            slot_in_cache(cold_cache[layer].0, pos)
            && slot_in_cache(cold_cache[layer].1, pos)
    by {
        assert(cache_pair_has_position(cold_cache, layer, pos));
    }

    if wr.layers.len() == 0 {
        broadcast use {
            RT::lemma_embed_repr_shape,
            RT::lemma_rms_norm_repr_shape,
            RT::lemma_linear_repr_shape,
        };
        let full_embed = RT::embed_repr(tokens, wr.embed_weight);
        let suffix_embed = RT::embed_repr(suffix_tokens, wr.embed_weight);
        BI::embed_subrange_invariance(
            tokens, wr.embed_weight, prefix_len as int, n as int,
        );
        let full_norm = BD::rms_norm_repr(config, full_embed, wr.final_norm);
        let suffix_norm = BD::rms_norm_repr(config, suffix_embed, wr.final_norm);
        BI::rms_norm_subrange_invariance(
            full_embed, wr.final_norm, config.rms_norm_epsilon,
            prefix_len as int, n as int,
        );
        BI::linear_subrange_invariance(
            full_norm, wr.lm_head, prefix_len as int, n as int,
        );
        assert(cold_logits.len() == n);
        assert(continuation_logits.len() == q);
        assert(cold_logits.subrange(prefix_len as int, n as int)
            == continuation_logits);
        assert(cache_sequences_agree_on_prefix_in_range(
            cold_cache, continuation_cache, 0, wr.layers.len(), n,
        ));
    } else {
        broadcast use {
            RT::lemma_embed_repr_shape,
            RT::lemma_rms_norm_repr_shape,
            RT::lemma_add_rms_norm_repr_shape,
            RT::lemma_linear_repr_shape,
        };
        let wr0 = wr.layers[0];
        let full_embed = RT::embed_repr(tokens, wr.embed_weight);
        let suffix_embed = RT::embed_repr(suffix_tokens, wr.embed_weight);
        BI::embed_subrange_invariance(
            tokens, wr.embed_weight, prefix_len as int, n as int,
        );
        assert(full_embed.subrange(prefix_len as int, n as int) == suffix_embed);
        let full_norm = BD::rms_norm_repr(config, full_embed, wr0.input_norm);
        let suffix_norm = BD::rms_norm_repr(config, suffix_embed, wr0.input_norm);
        BI::rms_norm_subrange_invariance(
            full_embed, wr0.input_norm, config.rms_norm_epsilon,
            prefix_len as int, n as int,
        );
        assert(full_norm.subrange(prefix_len as int, n as int) == suffix_norm);

        assert forall|pos: nat| #![auto] pos < n implies
            slot_in_cache(full_base[0].0, pos)
            && slot_in_cache(full_base[0].1, pos)
        by {
            BD::lemma_synthetic_cache_slot_in_cache(
                n, wr.layers.len(), 0, pos,
            );
        }
        let full_first_update = CC::first_decoder_layer_kv_update_repr(config,
            wr0, full_embed, full_positions,
            full_base[0].0, full_base[0].1, full_slots,
        );
        CC::lemma_model_forward_kv_reprs_layer_zero(config,
            wr, tokens, full_positions, full_base, full_slots,
            full_cu, full_cu, n, n, bt,
        );
        assert(cold_cache[0] == full_first_update);
        assert forall|pos: nat|
            #![trigger cache_at(cold_cache[0].0, pos),
                cache_at(cold_cache[0].1, pos)]
            pos < prefix_len implies {
                &&& cache_at(cold_cache[0].0, pos)
                    == cache_at(full_first_update.0, pos)
                &&& cache_at(cold_cache[0].1, pos)
                    == cache_at(full_first_update.1, pos)
            }
        by {
        }
        lemma_decoder_core_canonical_prefix_continuation(config,
            wr0,
            full_norm, suffix_norm, full_embed, suffix_embed,
            full_base[0].0, full_base[0].1,
            cold_cache[0].0, cold_cache[0].1,
            n, prefix_len,
        );

        let suffix_first_update = CC::first_decoder_layer_kv_update_repr(config,
            wr0, suffix_embed, suffix_positions,
            cold_cache[0].0, cold_cache[0].1, suffix_slots,
        );
        let suffix_pre_rows = BD::pre_attention_repr(config,
            wr0, suffix_norm, suffix_positions,
        );
        let suffix_v_rows = RT::view_as_kv_repr(
            RT::qkv_linear_repr(
                suffix_norm, wr0.q_proj, wr0.k_proj, wr0.v_proj,
            ).2,
        );
        assert(suffix_norm.len() == q);
        assert(suffix_positions.len() == q);
        BD::lemma_pre_attention_repr_shape(config,
            wr0, suffix_norm, suffix_positions,
        );
        RT::lemma_qkv_linear_repr_shape(
            suffix_norm, wr0.q_proj, wr0.k_proj, wr0.v_proj,
        );
        RT::lemma_view_as_kv_repr_shape(
            RT::qkv_linear_repr(
                suffix_norm, wr0.q_proj, wr0.k_proj, wr0.v_proj,
            ).2,
        );
        assert(suffix_pre_rows.1.len() == q);
        assert(suffix_v_rows.len() == q);
        assert(suffix_slots.len() == q);
        assert(suffix_first_update == RT::store_kv_cache_repr(
            suffix_pre_rows.1, suffix_v_rows,
            cold_cache[0].0, cold_cache[0].1, suffix_slots,
        ));
        assert forall|pos: nat| #![auto] pos < n implies
            slot_in_cache(suffix_first_update.0, pos)
            && slot_in_cache(suffix_first_update.1, pos)
        by {
            RT::store_kv_cache_repr_preserves_slot_in_cache(
                suffix_pre_rows.1, suffix_v_rows,
                cold_cache[0].0, cold_cache[0].1, suffix_slots, pos,
            );
        }
        let full_first_out = BD::first_decoder_layer_output_repr(config,
            wr0, full_embed, full_positions,
            full_base[0].0, full_base[0].1, full_slots,
            full_cu, full_cu, n, n, bt,
        );
        let suffix_first_out = BD::first_decoder_layer_output_repr(config,
            wr0, suffix_embed, suffix_positions,
            cold_cache[0].0, cold_cache[0].1, suffix_slots,
            suffix_cu_q, full_cu, q, n, bt,
        );
        BD::lemma_first_decoder_layer_output_repr_shape(
            config, wr0,
            full_embed, full_positions,
            full_base[0].0, full_base[0].1, full_slots,
            full_cu, full_cu, n, n, bt,
        );
        BD::lemma_first_decoder_layer_output_repr_shape(
            config, wr0,
            suffix_embed, suffix_positions,
            cold_cache[0].0, cold_cache[0].1, suffix_slots,
            suffix_cu_q, full_cu, q, n, bt,
        );
        assert(full_first_update == BD::layer_kv_update_repr(config,
            wr0, full_norm, full_positions,
            full_base[0].0, full_base[0].1, full_slots,
        ));
        assert(suffix_first_update == BD::layer_kv_update_repr(config,
            wr0, suffix_norm, suffix_positions,
            cold_cache[0].0, cold_cache[0].1, suffix_slots,
        ));
        assert(full_first_out.0.len() == n);
        assert(full_first_out.1.len() == n);
        assert(suffix_first_out.0.len() == q);
        assert(suffix_first_out.1.len() == q);
        assert(full_first_out.0.subrange(prefix_len as int, n as int)
            == suffix_first_out.0);
        assert(full_first_out.1.subrange(prefix_len as int, n as int)
            == suffix_first_out.1);
        let full_kv1 = full_base.update(0, full_first_update);
        let suffix_kv1 = cold_cache.update(0, suffix_first_update);

        assert(cache_sequence_has_positions_in_range(
            full_kv1, 1, wr.layers.len(), n,
        )) by {
            assert forall|layer: int, pos: nat| #![auto]
                1 <= layer < wr.layers.len() && pos < n implies
                    slot_in_cache(full_kv1[layer].0, pos)
                    && slot_in_cache(full_kv1[layer].1, pos)
            by {
                assert(layer != 0);
                assert(full_kv1[layer] == full_base[layer]);
                BD::lemma_synthetic_cache_slot_in_cache(
                    n, wr.layers.len(), layer, pos,
                );
            }
        }
        assert(cache_sequence_has_positions_in_range(
            suffix_kv1, 1, wr.layers.len(), n,
        )) by {
            assert forall|layer: int, pos: nat| #![auto]
                1 <= layer < wr.layers.len() && pos < n implies
                    slot_in_cache(suffix_kv1[layer].0, pos)
                    && slot_in_cache(suffix_kv1[layer].1, pos)
            by {
                assert(layer != 0);
                assert(suffix_kv1[layer] == cold_cache[layer]);
            }
        }
        let full_tail = CC::layer_chain_kv_reprs(config,
            wr.layers, full_first_out.0, full_first_out.1, full_positions,
            full_kv1, full_slots, full_cu, full_cu, n, n, bt, 1,
        );
        assert(full_tail == cold_cache);
        assert(cache_sequence_has_positions_in_range(
            full_tail, 1, wr.layers.len(), n,
        )) by {
            assert forall|layer: int, pos: nat| #![auto]
                1 <= layer < wr.layers.len() && pos < n implies
                    slot_in_cache(full_tail[layer].0, pos)
                    && slot_in_cache(full_tail[layer].1, pos)
            by {
                assert(full_tail[layer] == cold_cache[layer]);
            }
        }
        assert forall|layer: int, pos: nat|
            #![trigger cache_at(suffix_kv1[layer].0, pos)]
            1 <= layer < wr.layers.len() && pos < prefix_len implies {
                &&& 0 <= layer < full_tail.len()
                &&& slot_in_cache(full_tail[layer].0, pos)
                &&& slot_in_cache(full_tail[layer].1, pos)
                &&& cache_at(full_tail[layer].0, pos)
                    == cache_at(suffix_kv1[layer].0, pos)
                &&& cache_at(suffix_kv1[layer].1, pos)
                    == cache_at(full_tail[layer].1, pos)
            }
        by {
            assert(layer != 0);
            assert(suffix_kv1[layer] == cold_cache[layer]);
            assert(full_tail[layer] == cold_cache[layer]);
        }
        lemma_layer_chain_canonical_prefix_continuation(config,
            wr.layers,
            full_first_out.0, suffix_first_out.0,
            full_first_out.1, suffix_first_out.1,
            full_kv1, suffix_kv1, n, prefix_len, 1,
        );

        let suffix_tail = CC::layer_chain_kv_reprs(config,
            wr.layers, suffix_first_out.0, suffix_first_out.1, suffix_positions,
            suffix_kv1, suffix_slots, suffix_cu_q, full_cu, q, n, bt, 1,
        );
        assert(continuation_cache == suffix_tail);
        CC::lemma_layer_chain_kv_reprs_preserves_before_start(config,
            wr.layers, suffix_first_out.0, suffix_first_out.1, suffix_positions,
            suffix_kv1, suffix_slots, suffix_cu_q, full_cu, q, n, bt, 1, 0,
        );
        CC::lemma_layer_chain_kv_reprs_len(config,
            wr.layers, suffix_first_out.0, suffix_first_out.1, suffix_positions,
            suffix_kv1, suffix_slots, suffix_cu_q, full_cu, q, n, bt, 1,
        );
        assert(suffix_tail.len() == suffix_kv1.len());
        assert(continuation_cache.len() == cold_cache.len());
        assert(continuation_cache[0] == suffix_first_update);
        assert forall|layer: int, pos: nat| #![auto]
            0 <= layer < wr.layers.len() && pos < n implies {
                &&& slot_in_cache(cold_cache[layer].0, pos)
                &&& slot_in_cache(cold_cache[layer].1, pos)
                &&& slot_in_cache(continuation_cache[layer].0, pos)
                &&& slot_in_cache(continuation_cache[layer].1, pos)
                &&& cache_at(cold_cache[layer].0, pos)
                    == cache_at(continuation_cache[layer].0, pos)
                &&& cache_at(cold_cache[layer].1, pos)
                    == cache_at(continuation_cache[layer].1, pos)
            }
        by {
            if layer == 0 {
                assert(slot_in_cache(full_first_update.0, pos));
                assert(slot_in_cache(full_first_update.1, pos));
                assert(slot_in_cache(suffix_first_update.0, pos));
                assert(slot_in_cache(suffix_first_update.1, pos));
                assert(cache_at(full_first_update.0, pos)
                    == cache_at(suffix_first_update.0, pos));
                assert(cache_at(full_first_update.1, pos)
                    == cache_at(suffix_first_update.1, pos));
            } else {
                assert(1 <= layer);
                assert(slot_in_cache(full_tail[layer].0, pos));
                assert(slot_in_cache(full_tail[layer].1, pos));
                assert(slot_in_cache(suffix_tail[layer].0, pos));
                assert(slot_in_cache(suffix_tail[layer].1, pos));
                assert(cache_at(full_tail[layer].0, pos)
                    == cache_at(suffix_tail[layer].0, pos));
                assert(cache_at(full_tail[layer].1, pos)
                    == cache_at(suffix_tail[layer].1, pos));
            }
        }
        assert(cache_sequences_agree_on_prefix_in_range(
            cold_cache, continuation_cache, 0, wr.layers.len(), n,
        ));

        let full_chain = BD::layer_chain_repr(config,
            wr.layers, full_first_out.0, full_first_out.1, full_positions,
            full_kv1, full_slots, full_cu, full_cu, n, n, bt, 1,
        );
        let suffix_chain = BD::layer_chain_repr(config,
            wr.layers, suffix_first_out.0, suffix_first_out.1, suffix_positions,
            suffix_kv1, suffix_slots, suffix_cu_q, full_cu, q, n, bt, 1,
        );
        assert(full_chain.0.len() == n);
        assert(full_chain.1.len() == n);
        assert(suffix_chain.0.len() == q);
        assert(suffix_chain.1.len() == q);
        assert(full_chain.0.subrange(prefix_len as int, n as int)
            == suffix_chain.0);
        assert(full_chain.1.subrange(prefix_len as int, n as int)
            == suffix_chain.1);
        let full_final = BD::add_rms_norm_repr(config,
            full_chain.0, full_chain.1, wr.final_norm,
        );
        let suffix_final = BD::add_rms_norm_repr(config,
            suffix_chain.0, suffix_chain.1, wr.final_norm,
        );
        BI::add_rms_norm_subrange_invariance(
            full_chain.0, full_chain.1, wr.final_norm,
            config.rms_norm_epsilon, prefix_len as int, n as int,
        );
        BI::linear_subrange_invariance(
            full_final.0, wr.lm_head, prefix_len as int, n as int,
        );
        let full_chain_base = BD::layer_chain_repr(config,
            wr.layers, full_first_out.0, full_first_out.1, full_positions,
            full_base, full_slots, full_cu, full_cu, n, n, bt, 1,
        );
        let suffix_chain_base = BD::layer_chain_repr(config,
            wr.layers, suffix_first_out.0, suffix_first_out.1, suffix_positions,
            cold_cache, suffix_slots, suffix_cu_q, full_cu, q, n, bt, 1,
        );
        assert forall|layer: int| 1 <= layer < wr.layers.len() implies
            #[trigger] full_base[layer] == full_kv1[layer] by {
            assert(layer != 0);
        }
        assert forall|layer: int| 1 <= layer < wr.layers.len() implies
            #[trigger] cold_cache[layer] == suffix_kv1[layer] by {
            assert(layer != 0);
        }
        RELATIONAL::lemma_layer_chain_repr_ignores_before_start(config,
            wr.layers, full_first_out.0, full_first_out.1, full_positions,
            full_base, full_kv1, full_slots,
            full_cu, full_cu, n, n, bt, 1,
        );
        RELATIONAL::lemma_layer_chain_repr_ignores_before_start(config,
            wr.layers, suffix_first_out.0, suffix_first_out.1, suffix_positions,
            cold_cache, suffix_kv1, suffix_slots,
            suffix_cu_q, full_cu, q, n, bt, 1,
        );
        assert(full_chain_base == full_chain);
        assert(suffix_chain_base == suffix_chain);
        assert(cold_logits == RT::linear_repr(full_final.0, wr.lm_head));
        assert(continuation_logits == RT::linear_repr(suffix_final.0, wr.lm_head));
        assert(cold_logits.len() == n);
        assert(continuation_logits.len() == q);
        assert(cold_logits.subrange(prefix_len as int, n as int)
            == continuation_logits);
    }
}

// Replace the capstone's cold-cache base with any contiguous cache whose
// logical prefix is canonical and whose complete history range is writable.
// Geometry relocation makes the suffix K/V rows and logits independent of all
// other base cells; unwritten-prefix and read-own-write lemmas then transfer the
// post-store cache pointwise.  This is the reusable semantic bridge expected by
// the actual `prefix_filled_base` construction.
pub proof fn lemma_canonical_prefix_base_continuation_matches_cold_prefill(
    config: DenseSwiGluForwardConfigRepr,
    wr: ModelWeightsRepr,
    tokens: IntTensor1D,
    prefix_len: nat,
    base: Seq<(KVCacheLayerRepr, KVCacheLayerRepr)>,
)
    requires
        RT::paged_attention_numeric_domain(),
        wr.layers.len() > 0,
        0 < prefix_len < tokens.len(),
        reference_history_supported(tokens),
        base.len() >= wr.layers.len(),
        forall|layer: int, pos: nat| #![auto]
            0 <= layer < wr.layers.len() && pos < tokens.len() ==>
                slot_in_cache(base[layer].0, pos)
                && slot_in_cache(base[layer].1, pos),
        forall|layer: int, pos: nat|
            #![trigger cache_pair_at(base, layer, pos)]
            0 <= layer < wr.layers.len() && pos < prefix_len ==> {
                &&& cache_pair_has_position(base, layer, pos)
                &&& canonical_kv_defined(config, wr, tokens, layer, pos)
                &&& cache_pair_at(base, layer, pos)
                    == canonical_kv_at(config, wr, tokens, layer, pos)
            },
    ensures ({
        let n = tokens.len();
        let q = (n - prefix_len) as nat;
        let suffix = tokens.subrange(prefix_len as int, n as int);
        let positions = positions_from(prefix_len, q);
        let slots = BD::slots_from(prefix_len, q);
        let cu_q = BD::seq_lens_for_single(q);
        let cu_k = BD::seq_lens_for_single(n);
        let bt = BD::singleton_block_rows(n);
        let cold_base = BD::synthetic_cache_reprs(n, wr.layers.len());
        let cold_logits = BD::model_forward_logits_repr(config,
            wr, tokens, positions_from(0, n), cold_base,
            BD::slots_from(0, n), cu_k, cu_k, n, n, bt,
        );
        let continuation_logits = BD::model_forward_logits_repr(config,
            wr, suffix, positions, base, slots, cu_q, cu_k, q, n, bt,
        );
        let continuation_cache = CC::model_forward_kv_reprs(config,
            wr, suffix, positions, base, slots, cu_q, cu_k, q, n, bt,
        );
        let cold_cache = cold_reference_cache_reprs(config, wr, tokens);
        &&& cold_logits.len() == n
        &&& continuation_logits.len() == q
        &&& cold_logits.subrange(prefix_len as int, n as int)
            == continuation_logits
        &&& cache_sequences_agree_on_prefix_in_range(
            cold_cache, continuation_cache, 0, wr.layers.len(), n,
        )
    }),
{
    reveal(BD::model_forward_logits_repr);
    let n = tokens.len();
    let q = (n - prefix_len) as nat;
    let suffix = tokens.subrange(prefix_len as int, n as int);
    let positions = positions_from(prefix_len, q);
    let slots = BD::slots_from(prefix_len, q);
    let cu_q = BD::seq_lens_for_single(q);
    let cu_k = BD::seq_lens_for_single(n);
    let bt_rows = BD::singleton_block_rows(n);
    let bt = bt_rows[0];
    let pages = blocks_needed_for(n);
    let cold_cache = cold_reference_cache_reprs(config, wr, tokens);
    let canonical_cache = canonical_prefix_continuation_cache_reprs(config,
        wr, tokens, prefix_len,
    );
    let continuation_cache = CC::model_forward_kv_reprs(config,
        wr, suffix, positions, base, slots, cu_q, cu_k, q, n, bt_rows,
    );
    let cold_logits = BD::model_forward_logits_repr(config,
        wr, tokens, positions_from(0, n),
        BD::synthetic_cache_reprs(n, wr.layers.len()),
        BD::slots_from(0, n), cu_k, cu_k, n, n, bt_rows,
    );
    let continuation_logits = BD::model_forward_logits_repr(config,
        wr, suffix, positions, base, slots, cu_q, cu_k, q, n, bt_rows,
    );
    let canonical_logits = canonical_prefix_continuation_logits_repr(config,
        wr, tokens, prefix_len,
    );

    lemma_cold_reference_cache_matches_canonical(config, wr, tokens);
    lemma_canonical_prefix_continuation_matches_cold_prefill(config,
        wr, tokens, prefix_len,
    );
    assert(cold_cache.len() == wr.layers.len());
    assert(bt =~= crate::proof::reference::request_machine::contiguous_block_ids(pages));
    assert(blocks_needed_for(n) <= bt.len());
    assert(suffix.len() == q);
    assert(positions.len() == q);
    assert(slots.len() == q);

    assert forall|j: int| #![trigger slots[j]]
        0 <= j < q as int implies
            block_table_slot(bt, (n - q + j as nat) as nat) == slots[j] as nat
            && slots[j] >= 0
    by {
        let pos = (prefix_len as int + j) as nat;
        assert(n - q == prefix_len);
        assert(slots[j] == prefix_len as int + j);
        crate::proof::tensor::geometry::lemma_blocks_needed_covers_pos(pos, n);
        crate::proof::reference::request_machine::lemma_contiguous_block_table_slot(pages, pos);
    }
    assert forall|j: int, m: int| #![trigger slots[m], slots[j]]
        0 <= j < q as int && j < m < q as int implies slots[m] != slots[j]
    by {
    }
    assert forall|pos: nat| #![trigger block_table_slot(bt, pos)]
        pos < n - q implies
            !slots.contains(block_table_slot(bt, pos) as int)
    by {
        assert(n - q == prefix_len);
        crate::proof::tensor::geometry::lemma_blocks_needed_covers_pos(pos, n);
        crate::proof::reference::request_machine::lemma_contiguous_block_table_slot(pages, pos);
        assert(block_table_slot(bt, pos) == pos);
        if slots.contains(pos as int) {
            let j = slots.index_of(pos as int);
            assert(slots[j] == prefix_len as int + j);
        }
    }
    assert forall|layer: int, j: int|
        #![trigger base[layer].0, slots[j]]
        0 <= layer < wr.layers.len() && 0 <= j < q as int implies
            slot_in_cache(base[layer].0, slots[j] as nat)
            && slot_in_cache(base[layer].1, slots[j] as nat)
            && slot_in_cache(cold_cache[layer].0, slots[j] as nat)
            && slot_in_cache(cold_cache[layer].1, slots[j] as nat)
    by {
        assert(0 <= slots[j] < n);
        assert(cache_pair_has_position(cold_cache, layer, slots[j] as nat));
    }
    assert forall|layer: int, pos: nat|
        #![trigger base[layer].0, block_table_slot(bt, pos)]
        0 <= layer < wr.layers.len() && pos < n - q implies
            slot_in_cache(base[layer].0, block_table_slot(bt, pos))
            && slot_in_cache(base[layer].1, block_table_slot(bt, pos))
            && slot_in_cache(cold_cache[layer].0, block_table_slot(bt, pos))
            && slot_in_cache(cold_cache[layer].1, block_table_slot(bt, pos))
            && cache_at(base[layer].0, block_table_slot(bt, pos))
                == cache_at(cold_cache[layer].0, block_table_slot(bt, pos))
            && cache_at(base[layer].1, block_table_slot(bt, pos))
                == cache_at(cold_cache[layer].1, block_table_slot(bt, pos))
    by {
        assert(n - q == prefix_len);
        crate::proof::tensor::geometry::lemma_blocks_needed_covers_pos(pos, n);
        crate::proof::reference::request_machine::lemma_contiguous_block_table_slot(pages, pos);
        assert(block_table_slot(bt, pos) == pos);
        assert(cache_pair_has_position(base, layer, pos));
        assert(cache_pair_at(base, layer, pos)
            == canonical_kv_at(config, wr, tokens, layer, pos));
        assert(cache_pair_at(cold_cache, layer, pos)
            == canonical_kv_at(config, wr, tokens, layer, pos));
    }

    RELATIONAL::model_forward_relocation(config,
        wr, suffix, positions,
        base, slots, bt,
        cold_cache, slots, bt,
        q, n,
    );
    assert(continuation_logits == canonical_logits);
    assert(cold_logits.len() == n);
    assert(cold_logits.subrange(prefix_len as int, n as int)
        == canonical_logits);

    CC::lemma_model_forward_kv_reprs_len(config,
        wr, suffix, positions, base, slots, cu_q, cu_k, q, n, bt_rows,
    );
    CC::lemma_model_forward_kv_reprs_len(config,
        wr, suffix, positions, cold_cache, slots, cu_q, cu_k, q, n, bt_rows,
    );
    assert(continuation_cache.len() == base.len());
    assert(canonical_cache.len() == cold_cache.len());
    assert forall|layer: int, pos: nat| #![auto]
        0 <= layer < wr.layers.len() && pos < n implies {
            &&& slot_in_cache(continuation_cache[layer].0, pos)
            &&& slot_in_cache(continuation_cache[layer].1, pos)
            &&& slot_in_cache(canonical_cache[layer].0, pos)
            &&& slot_in_cache(canonical_cache[layer].1, pos)
            &&& cache_at(continuation_cache[layer].0, pos)
                == cache_at(canonical_cache[layer].0, pos)
            &&& cache_at(continuation_cache[layer].1, pos)
                == cache_at(canonical_cache[layer].1, pos)
        }
    by {
        PW::lemma_model_layer_kv_rows_relocation(
            config,
            wr, suffix, positions,
            base, slots, bt,
            cold_cache, slots, bt,
            q, n, layer as nat,
        );
        PW::lemma_model_forward_layer_store(
            config,
            wr, suffix, positions, base, slots, cu_q, cu_k, q, n, bt_rows,
            layer as nat,
        );
        PW::lemma_model_forward_layer_store(
            config,
            wr, suffix, positions, cold_cache, slots, cu_q, cu_k, q, n, bt_rows,
            layer as nat,
        );
        let rows = PW::model_layer_kv_rows(
            config,
            wr, suffix, positions, base, slots, cu_q, cu_k, q, n, bt_rows,
            layer as nat,
        );
        let canonical_rows = PW::model_layer_kv_rows(
            config,
            wr, suffix, positions, cold_cache, slots, cu_q, cu_k, q, n, bt_rows,
            layer as nat,
        );
        assert(rows == canonical_rows);
        PW::lemma_model_layer_kv_rows_shape(
            config,
            wr, suffix, positions, base, slots, cu_q, cu_k, q, n, bt_rows,
            layer as nat,
        );
        if pos < prefix_len {
            assert(!slots.contains(pos as int));
            assert(cache_pair_at(base, layer, pos)
                == canonical_kv_at(config, wr, tokens, layer, pos));
            assert(cache_pair_at(cold_cache, layer, pos)
                == canonical_kv_at(config, wr, tokens, layer, pos));
            RT::store_kv_cache_repr_preserves_unwritten_slots(
                rows.0, rows.1, base[layer].0, base[layer].1, slots, pos,
            );
            RT::store_kv_cache_repr_preserves_unwritten_slots(
                canonical_rows.0, canonical_rows.1,
                cold_cache[layer].0, cold_cache[layer].1, slots, pos,
            );
            assert(cache_at(base[layer].0, pos)
                == cache_at(cold_cache[layer].0, pos));
            assert(cache_at(base[layer].1, pos)
                == cache_at(cold_cache[layer].1, pos));
        } else {
            let j = (pos - prefix_len) as int;
            assert(0 <= j < q);
            assert(slots[j] == pos as int);
            assert forall|m: int| j < m < slots.len()
                implies slots[m] != slots[j] by {
            }
            RT::store_kv_cache_repr_reads_own_write(
                rows.0, rows.1, base[layer].0, base[layer].1, slots, j,
            );
            RT::store_kv_cache_repr_reads_own_write(
                canonical_rows.0, canonical_rows.1,
                cold_cache[layer].0, cold_cache[layer].1, slots, j,
            );
        }
    }
    assert forall|layer: int, pos: nat| #![auto]
        0 <= layer < wr.layers.len() && pos < n implies {
            &&& slot_in_cache(cold_cache[layer].0, pos)
            &&& slot_in_cache(cold_cache[layer].1, pos)
            &&& slot_in_cache(continuation_cache[layer].0, pos)
            &&& slot_in_cache(continuation_cache[layer].1, pos)
            &&& cache_at(cold_cache[layer].0, pos)
                == cache_at(continuation_cache[layer].0, pos)
            &&& cache_at(cold_cache[layer].1, pos)
                == cache_at(continuation_cache[layer].1, pos)
        }
    by {
        assert(cache_at(canonical_cache[layer].0, pos)
            == cache_at(cold_cache[layer].0, pos));
        assert(cache_at(canonical_cache[layer].1, pos)
            == cache_at(cold_cache[layer].1, pos));
        assert(slot_in_cache(canonical_cache[layer].0, pos));
        assert(slot_in_cache(canonical_cache[layer].1, pos));
    }
}

// Pointwise cache fidelity used by the selected (Option B) runtime invariant.
// It is O(layers * cached positions) per request, rather than pairwise in the
// number of requests that happen to share a prefix.  The u64 bound covers only
// the positions represented by this cache; `tokens` may include the next
// emitted token, which does not require a block until the following step.
pub open spec fn cache_reprs_match_canonical(
    config: DenseSwiGluForwardConfigRepr,
    caches: Seq<(KVCacheLayerRepr, KVCacheLayerRepr)>,
    wr: ModelWeightsRepr,
    tokens: IntTensor1D,
    cached_tokens: nat,
) -> bool {
    &&& caches.len() == wr.layers.len()
    &&& cached_tokens <= tokens.len()
    &&& cached_prefix_supported(cached_tokens)
    &&& forall|layer: int, pos: nat|
        #![trigger cache_pair_at(caches, layer, pos)]
        #![trigger cache_pair_has_position(caches, layer, pos)]
        0 <= layer < wr.layers.len() && pos < cached_tokens ==> {
            &&& cache_pair_has_position(caches, layer, pos)
            &&& canonical_kv_defined(config, wr, tokens, layer, pos)
            &&& cache_pair_at(caches, layer, pos) == canonical_kv_at(config, wr, tokens, layer, pos)
    }
}

} // verus!
