//! Four-norm gated decoder discharge of architecture-neutral cache laws.
//!
//! Full and sliding-window layers share this proof.  Sliding-window attention
//! is treated as consuming the complete causal cache; no window-only KV
//! dependency or eviction theorem is used here.

#[cfg(verus_only)]
use crate::proof::model::four_norm_gated::batch_invariance as BI;
#[cfg(verus_only)]
use crate::proof::model::four_norm_gated::model as MODEL;
#[cfg(verus_only)]
use crate::proof::model::architecture as ARCH;
#[cfg(verus_only)]
use crate::proof::model::cache as CACHE_LAWS;
#[cfg(verus_only)]
use crate::proof::reference::request_machine as MACHINE;
use crate::{types::*, proof::model::types::*, proof::tensor::types::*};
#[cfg(verus_only)]
use crate::boundary::tensor_runtime as RT;
use vstd::prelude::*;

#[cfg(verus_only)]
use crate::proof::model::four_norm_gated::cache_prefix::{
    lemma_layer_chain_full_prefill_prefix_stable,
    lemma_layer_chain_canonical_prefix_continuation,
};

verus! {

pub open spec fn semantic_model(
    wr: ModelWeightsRepr, architecture: ModelWeightsArchitectureRepr,
) -> SemanticModelRepr {
    SemanticModelRepr { weights: wr, architecture }
}

broadcast use ARCH::lemma_four_norm_forward_composition;


pub proof fn lemma_cold_reference_agrees_with_own_prefix(
    wr: ModelWeightsRepr,
    family: FourNormGatedDecoderConfigRepr,
    architecture: ModelWeightsArchitectureRepr,
    tokens: IntTensor1D,
    prefix_len: nat,
)
    requires
        ARCH::four_norm_decoder_config(architecture) == Some(family),
        RT::paged_attention_numeric_domain(),
        family.layers.len() == wr.layers.len(),
        forall|layer: int| 0 <= layer < family.layers.len() ==>
            layer_attention_config_valid(
                #[trigger] family.layers[layer].attention,
            ),
        0 < prefix_len <= tokens.len(),
        CACHE_LAWS::reference_history_supported(tokens),
    ensures
        CACHE_LAWS::cache_sequences_agree_on_prefix_in_range(
            CACHE_LAWS::cold_reference_cache_reprs(
                semantic_model(wr, architecture), tokens,
            ),
            CACHE_LAWS::cold_reference_cache_reprs(
                semantic_model(wr, architecture),
                tokens.subrange(0, prefix_len as int),
            ),
            0,
            wr.layers.len(),
            prefix_len,
        ),
{
    let n = tokens.len();
    let prefix = tokens.subrange(0, prefix_len as int);
    let full_hidden = MODEL::scaled_embed_repr(
        tokens, wr.embed_weight, family.geometry.hidden_size,
    );
    let prefix_hidden = MODEL::scaled_embed_repr(
        prefix, wr.embed_weight, family.geometry.hidden_size,
    );
    BI::scaled_embed_subrange_invariance(
        tokens, wr.embed_weight, family.geometry.hidden_size,
        0, prefix_len as int,
    );
    MODEL::lemma_scaled_embed_repr_shape(
        tokens, wr.embed_weight, family.geometry.hidden_size,
    );
    MODEL::lemma_scaled_embed_repr_shape(
        prefix, wr.embed_weight, family.geometry.hidden_size,
    );
    assert(full_hidden.subrange(0, prefix_len as int) == prefix_hidden);

    let full_base = MACHINE::synthetic_cache_reprs(n, wr.layers.len());
    let prefix_base = MACHINE::synthetic_cache_reprs(
        prefix_len, wr.layers.len(),
    );
    assert(CACHE_LAWS::cache_sequence_has_positions_in_range(
        full_base, 0, wr.layers.len(), n,
    )) by {
        reveal(CACHE_LAWS::cache_sequence_has_positions_in_range);
        assert(full_base.len() == wr.layers.len());
        assert forall|layer: int, pos: nat| #![auto]
            0 <= layer < wr.layers.len() && pos < n implies
                crate::proof::tensor::geometry::slot_in_cache(full_base[layer].0, pos)
                && crate::proof::tensor::geometry::slot_in_cache(full_base[layer].1, pos)
        by {
            MACHINE::lemma_synthetic_cache_slot_in_cache(
                n, wr.layers.len(), layer, pos,
            );
        }
    }
    assert(CACHE_LAWS::cache_sequence_has_positions_in_range(
        prefix_base, 0, wr.layers.len(), prefix_len,
    )) by {
        reveal(CACHE_LAWS::cache_sequence_has_positions_in_range);
        assert(prefix_base.len() == wr.layers.len());
        assert forall|layer: int, pos: nat| #![auto]
            0 <= layer < wr.layers.len() && pos < prefix_len implies
                crate::proof::tensor::geometry::slot_in_cache(prefix_base[layer].0, pos)
                && crate::proof::tensor::geometry::slot_in_cache(prefix_base[layer].1, pos)
        by {
            MACHINE::lemma_synthetic_cache_slot_in_cache(
                prefix_len, wr.layers.len(), layer, pos,
            );
        }
    }
    lemma_layer_chain_full_prefill_prefix_stable(
        wr.layers, family.layers,
        full_hidden, prefix_hidden,
        full_base, prefix_base,
        n, prefix_len, 0,
    );
    reveal(CACHE_LAWS::cold_reference_cache_reprs);
    reveal(ARCH::model_forward_kv_reprs);
    reveal(MODEL::model_forward_kv_reprs);
    reveal(MODEL::model_forward_hidden_and_kv_reprs);
}

pub proof fn lemma_cold_reference_cache_canonical(
    wr: ModelWeightsRepr,
    family: FourNormGatedDecoderConfigRepr,
    architecture: ModelWeightsArchitectureRepr,
)
    requires
        ARCH::four_norm_decoder_config(architecture) == Some(family),
        family.layers.len() == wr.layers.len(),
        forall|layer: int| 0 <= layer < family.layers.len() ==>
            layer_attention_config_valid(
                #[trigger] family.layers[layer].attention,
            ),
    ensures
        CACHE_LAWS::cold_reference_cache_canonical(
            semantic_model(wr, architecture),
        ),
{
    assert forall|tokens: IntTensor1D|
        RT::paged_attention_numeric_domain()
        && CACHE_LAWS::reference_history_supported(tokens) implies
            #[trigger] CACHE_LAWS::cache_reprs_match_canonical(
                CACHE_LAWS::cold_reference_cache_reprs(
                    semantic_model(wr, architecture), tokens,
                ),
                semantic_model(wr, architecture),
                tokens,
                tokens.len(),
            )
    by {
        let model = semantic_model(wr, architecture);
        let caches = CACHE_LAWS::cold_reference_cache_reprs(model, tokens);
        let n = tokens.len();
        MODEL::lemma_model_forward_kv_reprs_len(
            wr, family, tokens,
            crate::proof::tensor::geometry::positions_from(0, n),
            MACHINE::synthetic_cache_reprs(n, wr.layers.len()),
            MACHINE::slots_from(0, n),
            MACHINE::seq_lens_for_single(n),
            MACHINE::seq_lens_for_single(n),
            n, n, MACHINE::singleton_block_rows(n),
        );
        assert(caches.len() == wr.layers.len()) by {
            reveal(CACHE_LAWS::cold_reference_cache_reprs);
            reveal(ARCH::model_forward_kv_reprs);
        }
        assert forall|layer: int, pos: nat|
            0 <= layer < wr.layers.len() && pos < n implies {
                &&& CACHE_LAWS::cache_pair_has_position(caches, layer, pos)
                &&& CACHE_LAWS::canonical_kv_defined(
                    model, tokens, layer, pos,
                )
                &&& CACHE_LAWS::cache_pair_at(caches, layer, pos)
                    == CACHE_LAWS::canonical_kv_at(
                        model, tokens, layer, pos,
                    )
            }
        by {
            let own_len = (pos + 1) as nat;
            let own = tokens.subrange(0, own_len as int);
            lemma_cold_reference_agrees_with_own_prefix(
                wr, family, architecture, tokens, own_len,
            );
            let own_caches = CACHE_LAWS::cold_reference_cache_reprs(
                model, own,
            );
            assert(CACHE_LAWS::cache_sequences_agree_on_prefix_in_range(
                caches, own_caches, 0, wr.layers.len(), own_len,
            ));
            assert(crate::proof::tensor::geometry::slot_in_cache(caches[layer].0, pos));
            assert(crate::proof::tensor::geometry::slot_in_cache(caches[layer].1, pos));
            assert(crate::proof::tensor::geometry::slot_in_cache(own_caches[layer].0, pos));
            assert(crate::proof::tensor::geometry::slot_in_cache(own_caches[layer].1, pos));
            reveal(CACHE_LAWS::canonical_kv_defined);
            reveal(CACHE_LAWS::canonical_kv_at);
            reveal(CACHE_LAWS::cache_pair_has_position);
            reveal(CACHE_LAWS::cache_pair_at);
        }
        reveal(CACHE_LAWS::cache_reprs_match_canonical);
    }
}

proof fn lemma_canonical_prefix_continuation_correct(
    wr: ModelWeightsRepr,
    family: FourNormGatedDecoderConfigRepr,
    architecture: ModelWeightsArchitectureRepr,
)
    requires
        ARCH::four_norm_decoder_config(architecture) == Some(family),
        family.layers.len() == wr.layers.len(),
        forall|layer: int| 0 <= layer < family.layers.len() ==>
            layer_attention_config_valid(
                #[trigger] family.layers[layer].attention,
            ),
    ensures
        CACHE_LAWS::canonical_prefix_continuation_correct(
            semantic_model(wr, architecture),
        ),
{
    let model = semantic_model(wr, architecture);
    lemma_cold_reference_cache_canonical(wr, family, architecture);
    assert forall|
        tokens: IntTensor1D,
        prefix_len: nat,
        base: Seq<(KVCacheLayerRepr, KVCacheLayerRepr)>,
    |
        #[trigger] CACHE_LAWS::canonical_prefix_continuation_case(
            model, tokens, prefix_len, base,
        )
    by {
        reveal(CACHE_LAWS::canonical_prefix_continuation_case);
        if CACHE_LAWS::prefix_continuation_ready(
            model, tokens, prefix_len, base,
        ) {
            reveal(CACHE_LAWS::prefix_continuation_ready);
            let n = tokens.len();
            let q = (n - prefix_len) as nat;
            let suffix_tokens = tokens.subrange(prefix_len as int, n as int);
            let full_hidden = MODEL::scaled_embed_repr(
                tokens, wr.embed_weight, family.geometry.hidden_size,
            );
            let suffix_hidden = MODEL::scaled_embed_repr(
                suffix_tokens, wr.embed_weight, family.geometry.hidden_size,
            );
            BI::scaled_embed_subrange_invariance(
                tokens, wr.embed_weight, family.geometry.hidden_size,
                prefix_len as int, n as int,
            );
            MODEL::lemma_scaled_embed_repr_shape(
                tokens, wr.embed_weight, family.geometry.hidden_size,
            );
            MODEL::lemma_scaled_embed_repr_shape(
                suffix_tokens, wr.embed_weight, family.geometry.hidden_size,
            );
            assert(full_hidden.subrange(prefix_len as int, n as int)
                == suffix_hidden);

            let full_base = MACHINE::synthetic_cache_reprs(
                n, wr.layers.len(),
            );
            assert(CACHE_LAWS::cache_sequence_has_positions_in_range(
                full_base, 0, wr.layers.len(), n,
            )) by {
                reveal(CACHE_LAWS::cache_sequence_has_positions_in_range);
                assert forall|layer: int, pos: nat| #![auto]
                    0 <= layer < wr.layers.len() && pos < n implies
                        crate::proof::tensor::geometry::slot_in_cache(
                            full_base[layer].0, pos,
                        )
                        && crate::proof::tensor::geometry::slot_in_cache(
                            full_base[layer].1, pos,
                        )
                by {
                    MACHINE::lemma_synthetic_cache_slot_in_cache(
                        n, wr.layers.len(), layer, pos,
                    );
                }
            }
            assert(CACHE_LAWS::cache_sequence_has_positions_in_range(
                base, 0, wr.layers.len(), n,
            )) by {
                reveal(CACHE_LAWS::cache_sequence_has_positions_in_range);
            }

            let full_chain = MODEL::layer_chain_repr(
                wr.layers, family.layers, full_hidden,
                crate::proof::tensor::geometry::positions_from(0, n),
                full_base, MACHINE::slots_from(0, n),
                MACHINE::seq_lens_for_single(n),
                MACHINE::seq_lens_for_single(n),
                n, n, MACHINE::singleton_block_rows(n), 0,
            );
            let suffix_chain = MODEL::layer_chain_repr(
                wr.layers, family.layers, suffix_hidden,
                crate::proof::tensor::geometry::positions_from(prefix_len, q),
                base, MACHINE::slots_from(prefix_len, q),
                MACHINE::seq_lens_for_single(q),
                MACHINE::seq_lens_for_single(n),
                q, n, MACHINE::singleton_block_rows(n), 0,
            );
            let cold_cache = CACHE_LAWS::cold_reference_cache_reprs(
                model, tokens,
            );
            assert(cold_cache == full_chain.1) by {
                reveal(CACHE_LAWS::cold_reference_cache_reprs);
                reveal(ARCH::model_forward_kv_reprs);
                reveal(MODEL::model_forward_kv_reprs);
                reveal(MODEL::model_forward_hidden_and_kv_reprs);
            }
            assert(CACHE_LAWS::cache_reprs_match_canonical(
                cold_cache, model, tokens, n,
            ));
            assert(CACHE_LAWS::cache_sequence_has_positions_in_range(
                full_chain.1, 0, wr.layers.len(), n,
            )) by {
                reveal(CACHE_LAWS::cache_sequence_has_positions_in_range);
                reveal(CACHE_LAWS::cache_reprs_match_canonical);
                assert forall|layer: int, pos: nat| #![auto]
                    0 <= layer < wr.layers.len() && pos < n implies
                        crate::proof::tensor::geometry::slot_in_cache(
                            full_chain.1[layer].0, pos,
                        )
                        && crate::proof::tensor::geometry::slot_in_cache(
                            full_chain.1[layer].1, pos,
                        )
                by {
                    assert(CACHE_LAWS::cache_pair_has_position(
                        cold_cache, layer, pos,
                    ));
                    assert(cold_cache == full_chain.1);
                    reveal(CACHE_LAWS::cache_pair_has_position);
                }
            }
            assert forall|layer: int, pos: nat|
                #![trigger crate::proof::tensor::geometry::cache_at(base[layer].0, pos)]
                #![trigger crate::proof::tensor::geometry::cache_at(base[layer].1, pos)]
                0 <= layer < wr.layers.len() && pos < prefix_len implies {
                    &&& crate::proof::tensor::geometry::cache_at(base[layer].0, pos)
                        == crate::proof::tensor::geometry::cache_at(
                            full_chain.1[layer].0, pos,
                        )
                    &&& crate::proof::tensor::geometry::cache_at(base[layer].1, pos)
                        == crate::proof::tensor::geometry::cache_at(
                            full_chain.1[layer].1, pos,
                        )
                }
            by {
                assert(CACHE_LAWS::cache_pair_at(base, layer, pos)
                    == CACHE_LAWS::canonical_kv_at(
                        model, tokens, layer, pos,
                    ));
                assert(CACHE_LAWS::cache_pair_at(cold_cache, layer, pos)
                    == CACHE_LAWS::canonical_kv_at(
                        model, tokens, layer, pos,
                    ));
                assert(cold_cache == full_chain.1);
                reveal(CACHE_LAWS::cache_pair_at);
            }
            MODEL::lemma_layer_chain_repr_shape(
                wr.layers, family.layers, full_hidden,
                crate::proof::tensor::geometry::positions_from(0, n),
                full_base, MACHINE::slots_from(0, n),
                MACHINE::seq_lens_for_single(n),
                MACHINE::seq_lens_for_single(n),
                n, n, MACHINE::singleton_block_rows(n), 0,
            );
            lemma_layer_chain_canonical_prefix_continuation(
                wr.layers, family.layers,
                full_hidden, suffix_hidden,
                full_base, base,
                n, prefix_len, 0,
            );
            assert(full_chain.0.subrange(prefix_len as int, n as int)
                == suffix_chain.0);
            assert(CACHE_LAWS::cache_sequences_agree_on_prefix_in_range(
                full_chain.1, suffix_chain.1,
                0, wr.layers.len(), n,
            ));
            crate::proof::model::four_norm_gated::layers::final_logits_subrange_invariance(
                full_chain.0, wr.final_norm, wr.lm_head,
                family.final_norm_policy, family.final_logit_softcap,
                prefix_len as int, n as int,
            );
            MODEL::lemma_model_forward_logits_repr_shape(
                wr, family, tokens,
                crate::proof::tensor::geometry::positions_from(0, n),
                full_base, MACHINE::slots_from(0, n),
                MACHINE::seq_lens_for_single(n),
                MACHINE::seq_lens_for_single(n),
                n, n, MACHINE::singleton_block_rows(n),
            );
            MODEL::lemma_model_forward_logits_repr_shape(
                wr, family, suffix_tokens,
                crate::proof::tensor::geometry::positions_from(prefix_len, q),
                base, MACHINE::slots_from(prefix_len, q),
                MACHINE::seq_lens_for_single(q),
                MACHINE::seq_lens_for_single(n),
                q, n, MACHINE::singleton_block_rows(n),
            );
            reveal(CACHE_LAWS::cold_reference_logits_repr);
            reveal(CACHE_LAWS::prefix_continuation_logits_repr);
            reveal(CACHE_LAWS::prefix_continuation_cache_reprs);
            reveal(ARCH::model_forward_logits_repr);
            reveal(ARCH::model_forward_kv_reprs);
            reveal(MODEL::model_forward_logits_repr);
            reveal(MODEL::model_forward_kv_reprs);
            reveal(MODEL::model_forward_hidden_and_kv_reprs);
            reveal(CACHE_LAWS::prefix_continuation_matches_cold);
        }
    }
    CACHE_LAWS::lemma_canonical_prefix_continuation_correct_intro(model);
}

pub proof fn lemma_cache_refinement_laws(
    wr: ModelWeightsRepr,
    family: FourNormGatedDecoderConfigRepr,
    architecture: ModelWeightsArchitectureRepr,
)
    requires
        ARCH::four_norm_decoder_config(architecture) == Some(family),
        semantic_model_repr_valid(semantic_model(wr, architecture)),
        family.layers.len() == wr.layers.len(),
        forall|layer: int| 0 <= layer < family.layers.len() ==>
            layer_attention_config_valid(#[trigger] family.layers[layer].attention),
    ensures
        CACHE_LAWS::cache_refinement_laws(semantic_model(wr, architecture)),
{
    lemma_cold_reference_cache_canonical(wr, family, architecture);
    lemma_canonical_prefix_continuation_correct(wr, family, architecture);
    reveal(CACHE_LAWS::cache_refinement_laws);
}

} // verus!
