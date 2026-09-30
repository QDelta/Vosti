//! Architecture-neutral cache-fidelity vocabulary and structural laws.
//!
//! A family-specific proof is responsible only for showing that its cold
//! forward is canonical and that continuation from an arbitrary canonical
//! prefix agrees with a cold full-history forward.  Everything below those
//! two semantic capstones is shared by every dense causal decoder family.

#[cfg(verus_only)]
use crate::proof::reference::independent_batch_model::ibm_semantic_model;
use crate::proof::reference::independent_batch_model::IndependentBatchModel;
use crate::proof::reference::request_machine::RequestMachine;
#[cfg(verus_only)]
use crate::proof::reference::request_machine::{
    seq_lens_for_single, singleton_block_rows, slots_from, synthetic_cache_reprs, token_seq_to_int,
};
#[cfg(verus_only)]
use crate::exec::request_state::history;
use crate::{types::*, proof::model::types::*, proof::tensor::types::*};
use vstd::prelude::*;

verus! {

pub open spec fn reference_history_supported(tokens: IntTensor1D) -> bool {
    crate::proof::tensor::geometry::blocks_needed_for(tokens.len()) <= u64::MAX as nat
}

pub open spec fn cached_prefix_supported(cached_tokens: nat) -> bool {
    crate::proof::tensor::geometry::blocks_needed_for(cached_tokens) <= u64::MAX as nat
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

pub open spec fn cache_pair_has_position(
    caches: Seq<(KVCacheLayerRepr, KVCacheLayerRepr)>,
    layer: int,
    pos: nat,
) -> bool {
    0 <= layer < caches.len()
    && crate::proof::tensor::geometry::slot_in_cache(caches[layer].0, pos)
    && crate::proof::tensor::geometry::slot_in_cache(caches[layer].1, pos)
}

pub open spec fn cache_pair_at(
    caches: Seq<(KVCacheLayerRepr, KVCacheLayerRepr)>,
    layer: int,
    pos: nat,
) -> (Tensor1D, Tensor1D)
    recommends cache_pair_has_position(caches, layer, pos),
{
    (
        crate::proof::tensor::geometry::cache_at(caches[layer].0, pos),
        crate::proof::tensor::geometry::cache_at(caches[layer].1, pos),
    )
}

// Architecture-neutral physical page geometry for a private cache sequence.
// Model forwards may update cells but must preserve this allocation shape.
pub open spec fn cache_sequence_page_shape(
    caches: Seq<(KVCacheLayerRepr, KVCacheLayerRepr)>,
    num_pages: nat,
) -> bool {
    forall|layer: int| 0 <= layer < caches.len() ==> {
        &&& (#[trigger] caches[layer]).0.len() == num_pages
        &&& caches[layer].1.len() == num_pages
        &&& (forall|page: int| 0 <= page < num_pages ==>
            (#[trigger] caches[layer].0[page]).len()
                == crate::types::BLOCK_SIZE_SPEC as int)
        &&& (forall|page: int| 0 <= page < num_pages ==>
            (#[trigger] caches[layer].1[page]).len()
                == crate::types::BLOCK_SIZE_SPEC as int)
    }
}

pub open spec fn cold_reference_cache_reprs(
    model: SemanticModelRepr,
    tokens: IntTensor1D,
) -> Seq<(KVCacheLayerRepr, KVCacheLayerRepr)> {
    let n = tokens.len();
    crate::proof::model::architecture::model_forward_kv_reprs(
        model.weights,
        model.architecture,
        tokens,
        crate::proof::tensor::geometry::positions_from(0, n),
        synthetic_cache_reprs(n, model.weights.layers.len()),
        slots_from(0, n),
        seq_lens_for_single(n),
        seq_lens_for_single(n),
        n,
        n,
        singleton_block_rows(n),
    )
}

pub open spec fn cold_reference_logits_repr(
    model: SemanticModelRepr,
    tokens: IntTensor1D,
) -> Tensor2D {
    let n = tokens.len();
    crate::proof::model::architecture::model_forward_logits_repr(
        model.weights,
        model.architecture,
        tokens,
        crate::proof::tensor::geometry::positions_from(0, n),
        synthetic_cache_reprs(n, model.weights.layers.len()),
        slots_from(0, n),
        seq_lens_for_single(n),
        seq_lens_for_single(n),
        n,
        n,
        singleton_block_rows(n),
    )
}

pub open spec fn canonical_kv_at(
    model: SemanticModelRepr,
    tokens: IntTensor1D,
    layer: int,
    pos: nat,
) -> (Tensor1D, Tensor1D)
    recommends
        0 <= layer < model.weights.layers.len(),
        pos < tokens.len(),
{
    let prefix = tokens.subrange(0, pos as int + 1);
    cache_pair_at(cold_reference_cache_reprs(model, prefix), layer, pos)
}

pub open spec fn canonical_kv_defined(
    model: SemanticModelRepr,
    tokens: IntTensor1D,
    layer: int,
    pos: nat,
) -> bool {
    &&& 0 <= layer < model.weights.layers.len()
    &&& pos < tokens.len()
    &&& cache_pair_has_position(
        cold_reference_cache_reprs(
            model, tokens.subrange(0, pos as int + 1),
        ),
        layer,
        pos,
    )
}

pub open spec fn cache_reprs_match_canonical(
    caches: Seq<(KVCacheLayerRepr, KVCacheLayerRepr)>,
    model: SemanticModelRepr,
    tokens: IntTensor1D,
    cached_tokens: nat,
) -> bool {
    &&& caches.len() == model.weights.layers.len()
    &&& cached_tokens <= tokens.len()
    &&& cached_prefix_supported(cached_tokens)
    &&& forall|layer: int, pos: nat|
        #![trigger cache_pair_at(caches, layer, pos)]
        #![trigger cache_pair_has_position(caches, layer, pos)]
        0 <= layer < model.weights.layers.len() && pos < cached_tokens ==> {
            &&& cache_pair_has_position(caches, layer, pos)
            &&& canonical_kv_defined(model, tokens, layer, pos)
            &&& cache_pair_at(caches, layer, pos)
                == canonical_kv_at(model, tokens, layer, pos)
        }
}

pub proof fn lemma_canonical_kv_respects_shared_prefix(
    model: SemanticModelRepr,
    left: IntTensor1D,
    right: IntTensor1D,
    upto: nat,
    layer: int,
    pos: nat,
)
    requires
        histories_share_prefix(left, right, upto),
        0 <= layer < model.weights.layers.len(),
        pos < upto,
        canonical_kv_defined(model, left, layer, pos),
        canonical_kv_defined(model, right, layer, pos),
    ensures
        canonical_kv_at(model, left, layer, pos)
            == canonical_kv_at(model, right, layer, pos),
{
    let end = pos as int + 1;
    assert(left.subrange(0, end) =~= right.subrange(0, end)) by {
        assert forall|i: int| 0 <= i < end implies left[i] == right[i] by {
            assert(left.subrange(0, upto as int)[i]
                == right.subrange(0, upto as int)[i]);
        }
    }
}

pub proof fn lemma_cache_fidelity_respects_shared_history(
    caches: Seq<(KVCacheLayerRepr, KVCacheLayerRepr)>,
    model: SemanticModelRepr,
    left: IntTensor1D,
    right: IntTensor1D,
    cached_tokens: nat,
)
    requires
        cache_reprs_match_canonical(caches, model, left, cached_tokens),
        histories_share_prefix(left, right, cached_tokens),
    ensures
        cache_reprs_match_canonical(caches, model, right, cached_tokens),
{
    reveal(cache_reprs_match_canonical);
    reveal(histories_share_prefix);
    assert forall|layer: int, pos: nat|
        0 <= layer < model.weights.layers.len() && pos < cached_tokens implies {
            &&& cache_pair_has_position(caches, layer, pos)
            &&& canonical_kv_defined(model, right, layer, pos)
            &&& cache_pair_at(caches, layer, pos)
                == canonical_kv_at(model, right, layer, pos)
        }
    by {
        assert(cache_pair_has_position(caches, layer, pos));
        assert(canonical_kv_defined(model, left, layer, pos));
        let end = pos as int + 1;
        assert(left.subrange(0, end) =~= right.subrange(0, end)) by {
            assert forall|i: int| 0 <= i < end implies left[i] == right[i] by {
                assert(left.subrange(0, cached_tokens as int)[i]
                    == right.subrange(0, cached_tokens as int)[i]);
            }
        }
        assert(canonical_kv_defined(model, right, layer, pos));
        lemma_canonical_kv_respects_shared_prefix(
            model, left, right, cached_tokens, layer, pos,
        );
    }
}

pub open spec fn cache_sequences_agree_on_prefix_in_range(
    full: Seq<(KVCacheLayerRepr, KVCacheLayerRepr)>,
    other: Seq<(KVCacheLayerRepr, KVCacheLayerRepr)>,
    start: nat,
    end: nat,
    prefix_len: nat,
) -> bool {
    end <= full.len()
    && end <= other.len()
    && forall|layer: int, pos: nat| #![auto]
        start <= layer < end && pos < prefix_len ==> {
            &&& crate::proof::tensor::geometry::slot_in_cache(full[layer].0, pos)
            &&& crate::proof::tensor::geometry::slot_in_cache(full[layer].1, pos)
            &&& crate::proof::tensor::geometry::slot_in_cache(other[layer].0, pos)
            &&& crate::proof::tensor::geometry::slot_in_cache(other[layer].1, pos)
            &&& crate::proof::tensor::geometry::cache_at(full[layer].0, pos)
                == crate::proof::tensor::geometry::cache_at(other[layer].0, pos)
            &&& crate::proof::tensor::geometry::cache_at(full[layer].1, pos)
                == crate::proof::tensor::geometry::cache_at(other[layer].1, pos)
    }
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
            crate::proof::tensor::geometry::slot_in_cache(caches[layer].0, pos)
            && crate::proof::tensor::geometry::slot_in_cache(caches[layer].1, pos)
}

// Transport canonical cache fidelity across pointwise equality of the logical
// contiguous prefix.  This is the common final step for both a nonempty-prefix
// continuation and a cold (zero-prefix) relocation.
pub proof fn lemma_cache_fidelity_from_prefix_agreement(
    canonical: Seq<(KVCacheLayerRepr, KVCacheLayerRepr)>,
    other: Seq<(KVCacheLayerRepr, KVCacheLayerRepr)>,
    model: SemanticModelRepr,
    tokens: IntTensor1D,
    cached_tokens: nat,
)
    requires
        cache_reprs_match_canonical(
            canonical, model, tokens, cached_tokens,
        ),
        cache_sequences_agree_on_prefix_in_range(
            canonical,
            other,
            0,
            model.weights.layers.len(),
            cached_tokens,
        ),
        other.len() == model.weights.layers.len(),
    ensures
        cache_reprs_match_canonical(
            other, model, tokens, cached_tokens,
        ),
{
    reveal(cache_reprs_match_canonical);
    reveal(cache_sequences_agree_on_prefix_in_range);
    assert forall|layer: int, pos: nat|
        #![trigger cache_pair_at(other, layer, pos)]
        #![trigger cache_pair_has_position(other, layer, pos)]
        0 <= layer < model.weights.layers.len() && pos < cached_tokens
        implies {
            &&& cache_pair_has_position(other, layer, pos)
            &&& canonical_kv_defined(model, tokens, layer, pos)
            &&& cache_pair_at(other, layer, pos)
                == canonical_kv_at(model, tokens, layer, pos)
        }
    by {
        assert(cache_pair_has_position(canonical, layer, pos));
        assert(cache_pair_has_position(other, layer, pos));
        assert(cache_pair_at(other, layer, pos)
            == cache_pair_at(canonical, layer, pos));
    }
}

pub open spec fn prefix_continuation_cache_reprs(
    model: SemanticModelRepr,
    tokens: IntTensor1D,
    prefix_len: nat,
    base: Seq<(KVCacheLayerRepr, KVCacheLayerRepr)>,
) -> Seq<(KVCacheLayerRepr, KVCacheLayerRepr)>
    recommends prefix_len < tokens.len(),
{
    let n = tokens.len();
    let q = (n - prefix_len) as nat;
    crate::proof::model::architecture::model_forward_kv_reprs(
        model.weights,
        model.architecture,
        tokens.subrange(prefix_len as int, n as int),
        crate::proof::tensor::geometry::positions_from(prefix_len, q),
        base,
        slots_from(prefix_len, q),
        seq_lens_for_single(q),
        seq_lens_for_single(n),
        q,
        n,
        singleton_block_rows(n),
    )
}

pub open spec fn prefix_continuation_logits_repr(
    model: SemanticModelRepr,
    tokens: IntTensor1D,
    prefix_len: nat,
    base: Seq<(KVCacheLayerRepr, KVCacheLayerRepr)>,
) -> Tensor2D
    recommends prefix_len < tokens.len(),
{
    let n = tokens.len();
    let q = (n - prefix_len) as nat;
    crate::proof::model::architecture::model_forward_logits_repr(
        model.weights,
        model.architecture,
        tokens.subrange(prefix_len as int, n as int),
        crate::proof::tensor::geometry::positions_from(prefix_len, q),
        base,
        slots_from(prefix_len, q),
        seq_lens_for_single(q),
        seq_lens_for_single(n),
        q,
        n,
        singleton_block_rows(n),
    )
}

pub open spec fn canonical_prefix_continuation_cache_reprs(
    model: SemanticModelRepr,
    tokens: IntTensor1D,
    prefix_len: nat,
) -> Seq<(KVCacheLayerRepr, KVCacheLayerRepr)>
    recommends prefix_len < tokens.len(),
{
    prefix_continuation_cache_reprs(
        model,
        tokens,
        prefix_len,
        cold_reference_cache_reprs(model, tokens),
    )
}

pub open spec fn canonical_prefix_continuation_logits_repr(
    model: SemanticModelRepr,
    tokens: IntTensor1D,
    prefix_len: nat,
) -> Tensor2D
    recommends prefix_len < tokens.len(),
{
    prefix_continuation_logits_repr(
        model,
        tokens,
        prefix_len,
        cold_reference_cache_reprs(model, tokens),
    )
}

// Family proof obligation 1: every cold full-history cache stores the
// canonical per-position values selected by the history prefix.
pub open spec fn cold_reference_cache_canonical(model: SemanticModelRepr) -> bool {
    forall|tokens: IntTensor1D|
        crate::boundary::tensor_runtime::paged_attention_numeric_domain()
        && reference_history_supported(tokens) ==>
            #[trigger] cache_reprs_match_canonical(
                cold_reference_cache_reprs(model, tokens),
                model,
                tokens,
                tokens.len(),
            )
}

pub open spec fn prefix_continuation_ready(
    model: SemanticModelRepr,
    tokens: IntTensor1D,
    prefix_len: nat,
    base: Seq<(KVCacheLayerRepr, KVCacheLayerRepr)>,
) -> bool {
    &&& crate::boundary::tensor_runtime::paged_attention_numeric_domain()
    &&& model.weights.layers.len() > 0
    &&& 0 < prefix_len < tokens.len()
    &&& reference_history_supported(tokens)
    &&& base.len() >= model.weights.layers.len()
    &&& forall|layer: int, pos: nat| #![auto]
        0 <= layer < model.weights.layers.len() && pos < tokens.len() ==>
            crate::proof::tensor::geometry::slot_in_cache(base[layer].0, pos)
            && crate::proof::tensor::geometry::slot_in_cache(base[layer].1, pos)
    &&& forall|layer: int, pos: nat|
        #![trigger cache_pair_at(base, layer, pos)]
        #![trigger cache_pair_has_position(base, layer, pos)]
        0 <= layer < model.weights.layers.len() && pos < prefix_len ==> {
            &&& cache_pair_has_position(base, layer, pos)
            &&& canonical_kv_defined(model, tokens, layer, pos)
            &&& cache_pair_at(base, layer, pos)
                == canonical_kv_at(model, tokens, layer, pos)
        }
}

pub open spec fn prefix_continuation_matches_cold(
    model: SemanticModelRepr,
    tokens: IntTensor1D,
    prefix_len: nat,
    base: Seq<(KVCacheLayerRepr, KVCacheLayerRepr)>,
) -> bool {
    let n = tokens.len();
    let cold_logits = cold_reference_logits_repr(model, tokens);
    let continuation_logits = prefix_continuation_logits_repr(
        model, tokens, prefix_len, base,
    );
    let continuation_cache = prefix_continuation_cache_reprs(
        model, tokens, prefix_len, base,
    );
    let cold_cache = cold_reference_cache_reprs(model, tokens);
    &&& cold_logits.len() == n
    &&& continuation_logits.len() == n - prefix_len
    &&& cold_logits.subrange(prefix_len as int, n as int)
        == continuation_logits
    &&& cache_sequences_agree_on_prefix_in_range(
        cold_cache,
        continuation_cache,
        0,
        model.weights.layers.len(),
        n,
    )
}

pub open spec fn canonical_prefix_continuation_case(
    model: SemanticModelRepr,
    tokens: IntTensor1D,
    prefix_len: nat,
    base: Seq<(KVCacheLayerRepr, KVCacheLayerRepr)>,
) -> bool {
    prefix_continuation_ready(model, tokens, prefix_len, base)
        ==> prefix_continuation_matches_cold(
            model, tokens, prefix_len, base,
        )
}

// Family proof obligation 2: continuing from any writable cache whose logical
// prefix is canonical yields the cold full-history suffix logits and cache.
// This full-causal statement intentionally makes no window-only eviction claim
// for SWA.
pub open spec fn canonical_prefix_continuation_correct(
    model: SemanticModelRepr,
) -> bool {
    forall|
        tokens: IntTensor1D,
        prefix_len: nat,
        base: Seq<(KVCacheLayerRepr, KVCacheLayerRepr)>,
    |
        #[trigger] canonical_prefix_continuation_case(
            model, tokens, prefix_len, base,
        )
}

pub proof fn lemma_canonical_prefix_continuation_correct_intro(
    model: SemanticModelRepr,
)
    requires
        forall|
            tokens: IntTensor1D,
            prefix_len: nat,
            base: Seq<(KVCacheLayerRepr, KVCacheLayerRepr)>,
        |
            #[trigger] canonical_prefix_continuation_case(
                model, tokens, prefix_len, base,
            ),
    ensures canonical_prefix_continuation_correct(model),
{
    reveal(canonical_prefix_continuation_correct);
}

pub open spec fn cache_refinement_laws(model: SemanticModelRepr) -> bool {
    semantic_model_repr_valid(model)
    && cold_reference_cache_canonical(model)
    && canonical_prefix_continuation_correct(model)
}

// A singleton forward over any writable base whose logical prefix is
// canonical produces a cache canonical for the complete processed history.
// For a nonempty prefix this is exactly the family continuation law.  For a
// zero prefix, the shared relocation theorem identifies the arbitrary base
// with the neutral cold-reference base; no family-specific proof is needed.
pub open spec fn canonical_prefix_forward_ready(
    model: SemanticModelRepr,
    tokens: IntTensor1D,
    prefix_len: nat,
    base: Seq<(KVCacheLayerRepr, KVCacheLayerRepr)>,
) -> bool {
    &&& crate::boundary::tensor_runtime::paged_attention_numeric_domain()
    &&& cache_refinement_laws(model)
    &&& crate::proof::model::architecture::request_projection_configuration_ready(
        model.weights,
        model.architecture,
    )
    &&& model.weights.layers.len() > 0
    &&& prefix_len < tokens.len()
    &&& reference_history_supported(tokens)
    &&& base.len() == model.weights.layers.len()
    &&& forall|layer: int, pos: nat| #![auto]
        0 <= layer < model.weights.layers.len() && pos < tokens.len() ==>
            crate::proof::tensor::geometry::slot_in_cache(base[layer].0, pos)
            && crate::proof::tensor::geometry::slot_in_cache(base[layer].1, pos)
    &&& forall|layer: int, pos: nat|
        #![trigger cache_pair_at(base, layer, pos)]
        #![trigger cache_pair_has_position(base, layer, pos)]
        0 <= layer < model.weights.layers.len() && pos < prefix_len ==> {
            &&& cache_pair_has_position(base, layer, pos)
            &&& canonical_kv_defined(model, tokens, layer, pos)
            &&& cache_pair_at(base, layer, pos)
                == canonical_kv_at(model, tokens, layer, pos)
        }
}

#[verifier::spinoff_prover]
pub proof fn lemma_canonical_prefix_forward_cache_fidelity(
    model: SemanticModelRepr,
    tokens: IntTensor1D,
    prefix_len: nat,
    base: Seq<(KVCacheLayerRepr, KVCacheLayerRepr)>,
)
    requires
        canonical_prefix_forward_ready(
            model, tokens, prefix_len, base,
        ),
    ensures
        cache_reprs_match_canonical(
            prefix_continuation_cache_reprs(
                model, tokens, prefix_len, base,
            ),
            model,
            tokens,
            tokens.len(),
        ),
        prefix_continuation_matches_cold(
            model, tokens, prefix_len, base,
        ),
{
    reveal(canonical_prefix_forward_ready);
    let n = tokens.len();
    let q = (n - prefix_len) as nat;
    let positions = crate::proof::tensor::geometry::positions_from(prefix_len, q);
    let slots = slots_from(prefix_len, q);
    let cu_q = seq_lens_for_single(q);
    let cu_k = seq_lens_for_single(n);
    let bt_rows = singleton_block_rows(n);
    let bt = bt_rows[0];
    let post = prefix_continuation_cache_reprs(
        model, tokens, prefix_len, base,
    );
    reveal(cache_refinement_laws);
    assert(semantic_model_repr_valid(model));
    assert(q > 0);
    crate::proof::model::architecture::lemma_model_forward_kv_reprs_len(
        model.weights,
        model.architecture,
        tokens.subrange(prefix_len as int, n as int),
        positions,
        base,
        slots,
        cu_q,
        cu_k,
        q,
        n,
        bt_rows,
    );
    assert(post.len() == model.weights.layers.len());
    assert(cache_reprs_match_canonical(
        cold_reference_cache_reprs(model, tokens),
        model,
        tokens,
        n,
    ));
    if prefix_len > 0 {
        assert(prefix_continuation_ready(
            model, tokens, prefix_len, base,
        )) by {
            reveal(prefix_continuation_ready);
        }
        assert(canonical_prefix_continuation_case(
            model, tokens, prefix_len, base,
        ));
        reveal(canonical_prefix_continuation_case);
        assert(prefix_continuation_matches_cold(
            model, tokens, prefix_len, base,
        ));
        reveal(prefix_continuation_matches_cold);
        lemma_cache_fidelity_from_prefix_agreement(
            cold_reference_cache_reprs(model, tokens),
            post,
            model,
            tokens,
            n,
        );
    } else {
        assert(prefix_len == 0);
        assert(q == n);
        assert(tokens.subrange(0, n as int) =~= tokens);
        let cold_base = synthetic_cache_reprs(
            n, model.weights.layers.len(),
        );
        let cold = cold_reference_cache_reprs(model, tokens);
        assert(bt_rows.len() == 1);
        assert(bt_rows =~= seq![bt]);
        assert(crate::proof::model::relocation::model_forward_singleton_relocation_ready(
            model.weights,
            model.architecture,
            tokens,
            crate::proof::tensor::geometry::positions_from(0, n),
            base,
            slots_from(0, n),
            bt,
            cold_base,
            slots_from(0, n),
            bt,
            n,
            n,
        )) by {
            reveal(crate::proof::model::relocation::model_forward_singleton_relocation_ready);
            assert(bt_rows.len() == 1);
            assert(bt.len() == crate::proof::tensor::geometry::blocks_needed_for(n));
            assert(crate::proof::tensor::geometry::blocks_needed_for(n) <= u64::MAX as nat);
            assert forall|j: int| 0 <= j < n as int implies
                crate::proof::tensor::geometry::block_table_slot(bt, j as nat)
                    == #[trigger] slots_from(0, n)[j] as nat
            by {
                crate::proof::tensor::geometry::lemma_blocks_needed_covers_pos(j as nat, n);
                crate::proof::reference::request_machine::lemma_contiguous_block_table_slot(
                    crate::proof::tensor::geometry::blocks_needed_for(n), j as nat,
                );
            }
            assert forall|j: int, m: int|
                #![trigger slots_from(0, n)[m], slots_from(0, n)[j]]
                0 <= j < n as int && j < m < n as int implies
                    slots_from(0, n)[m] != slots_from(0, n)[j]
            by {}
            assert(crate::proof::model::family_layout::fresh_writes_miss_cached_prefix(
                slots_from(0, n), bt, slots_from(0, n), bt, 0,
            )) by {
                reveal(crate::proof::model::family_layout::fresh_writes_miss_cached_prefix);
            }
            assert forall|layer: int, j: int|
                #![trigger base[layer].0, slots_from(0, n)[j]]
                0 <= layer < model.weights.layers.len()
                    && 0 <= j < n as int implies
                    crate::proof::tensor::geometry::slot_in_cache(
                        base[layer].0, slots_from(0, n)[j] as nat,
                    )
                    && crate::proof::tensor::geometry::slot_in_cache(
                        base[layer].1, slots_from(0, n)[j] as nat,
                    )
                    && crate::proof::tensor::geometry::slot_in_cache(
                        cold_base[layer].0,
                        slots_from(0, n)[j] as nat,
                    )
                    && crate::proof::tensor::geometry::slot_in_cache(
                        cold_base[layer].1,
                        slots_from(0, n)[j] as nat,
                    )
            by {
                crate::proof::reference::request_machine::lemma_synthetic_cache_slot_in_cache(
                    n, model.weights.layers.len(), layer, j as nat,
                );
            }
        }
        crate::proof::model::relocation::lemma_model_forward_singleton_relocation(
            model.weights,
            model.architecture,
            tokens,
            crate::proof::tensor::geometry::positions_from(0, n),
            base,
            slots_from(0, n),
            bt,
            cold_base,
            slots_from(0, n),
            bt,
            n,
            n,
        );
        crate::proof::model::architecture::lemma_model_forward_logits_repr_shape(
            model.weights,
            model.architecture,
            tokens,
            crate::proof::tensor::geometry::positions_from(0, n),
            cold_base,
            slots_from(0, n),
            seq_lens_for_single(n),
            seq_lens_for_single(n),
            n,
            n,
            bt_rows,
        );
        crate::proof::model::architecture::lemma_model_forward_logits_repr_shape(
            model.weights,
            model.architecture,
            tokens,
            crate::proof::tensor::geometry::positions_from(0, n),
            base,
            slots_from(0, n),
            seq_lens_for_single(n),
            seq_lens_for_single(n),
            n,
            n,
            bt_rows,
        );
        reveal(prefix_continuation_logits_repr);
        reveal(cold_reference_logits_repr);
        assert(prefix_continuation_logits_repr(
            model, tokens, 0, base,
        ) == cold_reference_logits_repr(model, tokens));
        reveal(prefix_continuation_cache_reprs);
        reveal(cold_reference_cache_reprs);
        assert(post == crate::proof::model::architecture::model_forward_kv_reprs(
            model.weights,
            model.architecture,
            tokens,
            crate::proof::tensor::geometry::positions_from(0, n),
            base,
            slots_from(0, n),
            seq_lens_for_single(n),
            seq_lens_for_single(n),
            n,
            n,
            seq![bt],
        ));
        assert(cold == crate::proof::model::architecture::model_forward_kv_reprs(
            model.weights,
            model.architecture,
            tokens,
            crate::proof::tensor::geometry::positions_from(0, n),
            cold_base,
            slots_from(0, n),
            seq_lens_for_single(n),
            seq_lens_for_single(n),
            n,
            n,
            seq![bt],
        ));
        assert(crate::proof::model::family_layout::cache_sequence_logical_prefix_equal(
            post,
            bt,
            cold,
            bt,
            model.weights.layers.len(),
            n,
        ));
        assert(cache_sequences_agree_on_prefix_in_range(
            cold,
            post,
            0,
            model.weights.layers.len(),
            n,
        )) by {
            reveal(crate::proof::model::family_layout::cache_sequence_logical_prefix_equal);
            reveal(crate::proof::model::family_layout::cache_pair_logical_prefix_equal);
            reveal(cache_sequences_agree_on_prefix_in_range);
            assert forall|layer: int, pos: nat| #![auto]
                0 <= layer < model.weights.layers.len() && pos < n implies {
                    &&& crate::proof::tensor::geometry::slot_in_cache(cold[layer].0, pos)
                    &&& crate::proof::tensor::geometry::slot_in_cache(cold[layer].1, pos)
                    &&& crate::proof::tensor::geometry::slot_in_cache(post[layer].0, pos)
                    &&& crate::proof::tensor::geometry::slot_in_cache(post[layer].1, pos)
                    &&& crate::proof::tensor::geometry::cache_at(cold[layer].0, pos)
                        == crate::proof::tensor::geometry::cache_at(post[layer].0, pos)
                    &&& crate::proof::tensor::geometry::cache_at(cold[layer].1, pos)
                        == crate::proof::tensor::geometry::cache_at(post[layer].1, pos)
                }
            by {
                crate::proof::tensor::geometry::lemma_blocks_needed_covers_pos(pos, n);
                crate::proof::reference::request_machine::lemma_contiguous_block_table_slot(
                    crate::proof::tensor::geometry::blocks_needed_for(n), pos,
                );
                assert(crate::proof::model::family_layout::cache_pair_logical_prefix_equal(
                    post[layer], bt, cold[layer], bt, n,
                ));
            }
        }
        lemma_cache_fidelity_from_prefix_agreement(
            cold, post, model, tokens, n,
        );
        assert(prefix_continuation_matches_cold(
            model, tokens, 0, base,
        )) by {
            reveal(prefix_continuation_matches_cold);
            assert(cold_reference_logits_repr(model, tokens).subrange(
                0, n as int,
            ) =~= cold_reference_logits_repr(model, tokens));
        }
    }
}

pub open spec fn machine_cache_fidelity(
    machine: RequestMachine,
    model: SemanticModelRepr,
) -> bool {
    cache_reprs_match_canonical(
        machine.kv_cache_reprs,
        model,
        token_seq_to_int(history(machine.request_state)),
        machine.kv_tokens,
    )
}

pub open spec fn ibm_cache_fidelity(ibm: &IndependentBatchModel) -> bool {
    forall|rid: RequestId|
        ibm.machines.contains_key(rid) ==>
            #[trigger] machine_cache_fidelity(
                ibm.machines[rid], ibm_semantic_model(*ibm),
            )
}

// Architecture-neutral semantic contract for one surviving machine.  It is
// deliberately composed from two reusable neutral properties: the post-cache
// is canonical for the new cached history, and it agrees with the old cache on
// the old logical prefix.  Concrete architectures do not appear here.
pub open spec fn machine_cache_extension(
    old: RequestMachine,
    new: RequestMachine,
    model: SemanticModelRepr,
) -> bool {
    let old_tokens = token_seq_to_int(history(old.request_state));
    let new_tokens = token_seq_to_int(history(new.request_state));
    &&& crate::exec::request_state::valid_request_state(old.request_state)
    &&& crate::exec::request_state::valid_request_state(new.request_state)
    &&& old.kv_tokens <= new.kv_tokens
    &&& new.kv_tokens <= new_tokens.len()
    &&& histories_share_prefix(old_tokens, new_tokens, old.kv_tokens)
    &&& cache_reprs_match_canonical(
        new.kv_cache_reprs, model, new_tokens, new.kv_tokens,
    )
    &&& cache_sequences_agree_on_prefix_in_range(
        old.kv_cache_reprs,
        new.kv_cache_reprs,
        0,
        model.weights.layers.len(),
        old.kv_tokens,
    )
}

// Constructor for the architecture-neutral extension contract.  Concrete
// engine proofs discharge these pointwise premises without unfolding the
// complete semantic predicate in their already-large solver context.
pub proof fn lemma_machine_cache_extension_intro(
    old: RequestMachine,
    new: RequestMachine,
    model: SemanticModelRepr,
)
    requires
        crate::exec::request_state::valid_request_state(old.request_state),
        crate::exec::request_state::valid_request_state(new.request_state),
        old.kv_tokens <= new.kv_tokens,
        new.kv_tokens <= token_seq_to_int(
            history(new.request_state),
        ).len(),
        histories_share_prefix(
            token_seq_to_int(history(old.request_state)),
            token_seq_to_int(history(new.request_state)),
            old.kv_tokens,
        ),
        cache_reprs_match_canonical(
            new.kv_cache_reprs,
            model,
            token_seq_to_int(history(new.request_state)),
            new.kv_tokens,
        ),
        cache_sequences_agree_on_prefix_in_range(
            old.kv_cache_reprs,
            new.kv_cache_reprs,
            0,
            model.weights.layers.len(),
            old.kv_tokens,
        ),
    ensures machine_cache_extension(old, new, model),
{
    let old_tokens = token_seq_to_int(history(old.request_state));
    let new_tokens = token_seq_to_int(history(new.request_state));
    reveal(machine_cache_extension);
    assert(old.kv_tokens <= new.kv_tokens);
    assert(new.kv_tokens <= new_tokens.len());
    assert(histories_share_prefix(
        old_tokens, new_tokens, old.kv_tokens,
    ));
}

pub proof fn lemma_machine_cache_extension_preserves_fidelity(
    old: RequestMachine,
    new: RequestMachine,
    model: SemanticModelRepr,
)
    requires
        machine_cache_fidelity(old, model),
        machine_cache_extension(old, new, model),
    ensures machine_cache_fidelity(new, model),
{
    reveal(machine_cache_extension);
}

pub proof fn lemma_machine_cache_extension_reflexive(
    machine: RequestMachine,
    model: SemanticModelRepr,
)
    requires
        crate::exec::request_state::valid_request_state(machine.request_state),
        machine_cache_fidelity(machine, model),
    ensures machine_cache_extension(machine, machine, model),
{
    let tokens = token_seq_to_int(history(machine.request_state));
    assert(histories_share_prefix(tokens, tokens, machine.kv_tokens)) by {
        assert(tokens.subrange(0, machine.kv_tokens as int)
            =~= tokens.subrange(0, machine.kv_tokens as int));
    }
    assert(cache_sequences_agree_on_prefix_in_range(
        machine.kv_cache_reprs,
        machine.kv_cache_reprs,
        0,
        model.weights.layers.len(),
        machine.kv_tokens,
    )) by {
        reveal(cache_sequences_agree_on_prefix_in_range);
        reveal(machine_cache_fidelity);
        reveal(cache_reprs_match_canonical);
        assert forall|layer: int, pos: nat| #![auto]
            0 <= layer < model.weights.layers.len()
                && pos < machine.kv_tokens implies {
                &&& crate::proof::tensor::geometry::slot_in_cache(
                    machine.kv_cache_reprs[layer].0, pos,
                )
                &&& crate::proof::tensor::geometry::slot_in_cache(
                    machine.kv_cache_reprs[layer].1, pos,
                )
                &&& crate::proof::tensor::geometry::slot_in_cache(
                    machine.kv_cache_reprs[layer].0, pos,
                )
                &&& crate::proof::tensor::geometry::slot_in_cache(
                    machine.kv_cache_reprs[layer].1, pos,
                )
                &&& crate::proof::tensor::geometry::cache_at(
                    machine.kv_cache_reprs[layer].0, pos,
                ) == crate::proof::tensor::geometry::cache_at(
                    machine.kv_cache_reprs[layer].0, pos,
                )
                &&& crate::proof::tensor::geometry::cache_at(
                    machine.kv_cache_reprs[layer].1, pos,
                ) == crate::proof::tensor::geometry::cache_at(
                    machine.kv_cache_reprs[layer].1, pos,
                )
            }
        by {
            assert(cache_pair_has_position(
                machine.kv_cache_reprs, layer, pos,
            ));
        }
    }
    reveal(machine_cache_extension);
}

// IBM-level lifting is phrased over complete semantic-model identity, not a
// family payload or weights-only equality.  A model architecture therefore
// cannot change silently while cache fidelity is transported across a step.
pub open spec fn ibm_cache_extension(
    old: IndependentBatchModel,
    new: IndependentBatchModel,
) -> bool {
    &&& ibm_semantic_model(old) == ibm_semantic_model(new)
    &&& forall|rid: RequestId|
        #![trigger new.machines.contains_key(rid)]
        new.machines.contains_key(rid) ==> {
            &&& old.machines.contains_key(rid)
            &&& machine_cache_extension(
                old.machines[rid],
                new.machines[rid],
                ibm_semantic_model(new),
            )
        }
}

pub proof fn lemma_ibm_cache_extension_preserves_fidelity(
    old: IndependentBatchModel,
    new: IndependentBatchModel,
)
    requires
        ibm_cache_fidelity(&old),
        ibm_cache_extension(old, new),
    ensures ibm_cache_fidelity(&new),
{
    assert forall|rid: RequestId|
        new.machines.contains_key(rid) implies
            #[trigger] machine_cache_fidelity(
                new.machines[rid], ibm_semantic_model(new),
            )
    by {
        assert(old.machines.contains_key(rid));
        assert(machine_cache_fidelity(
            old.machines[rid], ibm_semantic_model(old),
        ));
        assert(ibm_semantic_model(old) == ibm_semantic_model(new));
        lemma_machine_cache_extension_preserves_fidelity(
            old.machines[rid],
            new.machines[rid],
            ibm_semantic_model(new),
        );
    }
}

} // verus!
