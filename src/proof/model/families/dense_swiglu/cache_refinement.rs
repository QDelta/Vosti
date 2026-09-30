//! Dense-SwiGLU discharge of the architecture-neutral cache-refinement laws.
//!
//! This module instantiates the shared recursive cache proof for any admitted
//! architecture whose public payload selects this dense-SwiGLU composition.
//! It remains isolated so unrelated Engine VCs do not import its quantified
//! bridge lemmas.

#[cfg(verus_only)]
use super::{
    architecture_uses_config, forward_kv_reprs, forward_logits_repr,
    semantic_model as family_semantic_model,
};
#[cfg(verus_only)]
use crate::proof::model::dense_swiglu::cache_fidelity as CACHE_IMPL;
#[cfg(verus_only)]
use crate::proof::model::dense_swiglu::semantics as SEMANTICS;
#[cfg(verus_only)]
use crate::proof::model::architecture as ARCH;
#[cfg(verus_only)]
use crate::proof::model::cache as CACHE_LAWS;
use crate::{types::*, proof::model::types::*, proof::tensor::types::*};
use vstd::prelude::*;

verus! {

// Local projection to the architecture-neutral semantic model.
spec fn semantic_model(
    wr: ModelWeightsRepr,
    architecture: ModelWeightsArchitectureRepr,
) -> SemanticModelRepr {
    family_semantic_model(wr, architecture)
}


proof fn lemma_cold_reference_cache_reprs_agree(
    wr: ModelWeightsRepr,
    architecture: ModelWeightsArchitectureRepr,
    family: DenseSwiGluDecoderConfigRepr,
    tokens: IntTensor1D,
)
    requires architecture_uses_config(architecture, family),
    ensures
        CACHE_LAWS::cold_reference_cache_reprs(
            semantic_model(wr, architecture), tokens,
        )
            == CACHE_IMPL::cold_reference_cache_reprs(
                dense_swiglu_forward_config_repr(family), wr, tokens,
            ),
{
    reveal(CACHE_LAWS::cold_reference_cache_reprs);
    reveal(CACHE_IMPL::cold_reference_cache_reprs);
    reveal(ARCH::model_forward_kv_reprs);
    reveal(forward_kv_reprs);
    reveal(architecture_uses_config);
    match architecture {
        ModelWeightsArchitectureRepr::Qwen3(_) => {},
        ModelWeightsArchitectureRepr::Llama3(_) => {},
        ModelWeightsArchitectureRepr::Gemma3Text(_) => { assert(false); },
        ModelWeightsArchitectureRepr::Gemma4Text(_) => { assert(false); },
    }
}

proof fn lemma_cold_reference_logits_repr_agree(
    wr: ModelWeightsRepr,
    architecture: ModelWeightsArchitectureRepr,
    family: DenseSwiGluDecoderConfigRepr,
    tokens: IntTensor1D,
)
    requires architecture_uses_config(architecture, family),
    ensures
        CACHE_LAWS::cold_reference_logits_repr(
            semantic_model(wr, architecture), tokens,
        )
            == SEMANTICS::model_forward_logits_repr(
                dense_swiglu_forward_config_repr(family),
                wr,
                tokens,
                crate::proof::tensor::geometry::positions_from(0, tokens.len()),
                SEMANTICS::synthetic_cache_reprs(
                    tokens.len(), wr.layers.len(),
                ),
                SEMANTICS::slots_from(0, tokens.len()),
                SEMANTICS::seq_lens_for_single(tokens.len()),
                SEMANTICS::seq_lens_for_single(tokens.len()),
                tokens.len(),
                tokens.len(),
                SEMANTICS::singleton_block_rows(tokens.len()),
            ),
{
    reveal(CACHE_LAWS::cold_reference_logits_repr);
    reveal(ARCH::model_forward_logits_repr);
    reveal(forward_logits_repr);
    reveal(architecture_uses_config);
    match architecture {
        ModelWeightsArchitectureRepr::Qwen3(_) => {},
        ModelWeightsArchitectureRepr::Llama3(_) => {},
        ModelWeightsArchitectureRepr::Gemma3Text(_) => { assert(false); },
        ModelWeightsArchitectureRepr::Gemma4Text(_) => { assert(false); },
    }
}

proof fn lemma_canonical_kv_vocabulary_agrees(
    wr: ModelWeightsRepr,
    architecture: ModelWeightsArchitectureRepr,
    family: DenseSwiGluDecoderConfigRepr,
    tokens: IntTensor1D,
    layer: int,
    pos: nat,
)
    requires
        architecture_uses_config(architecture, family),
        0 <= layer < wr.layers.len(),
        pos < tokens.len(),
    ensures
        CACHE_LAWS::canonical_kv_defined(
            semantic_model(wr, architecture), tokens, layer, pos,
        ) == CACHE_IMPL::canonical_kv_defined(
            dense_swiglu_forward_config_repr(family),
            wr, tokens, layer, pos,
        ),
        CACHE_LAWS::canonical_kv_at(
            semantic_model(wr, architecture), tokens, layer, pos,
        ) == CACHE_IMPL::canonical_kv_at(
            dense_swiglu_forward_config_repr(family),
            wr, tokens, layer, pos,
        ),
{
    let prefix = tokens.subrange(0, pos as int + 1);
    lemma_cold_reference_cache_reprs_agree(
        wr, architecture, family, prefix,
    );
    reveal(CACHE_LAWS::canonical_kv_defined);
    reveal(CACHE_IMPL::canonical_kv_defined);
    reveal(CACHE_LAWS::canonical_kv_at);
    reveal(CACHE_IMPL::canonical_kv_at);
    reveal(CACHE_LAWS::cache_pair_has_position);
    reveal(CACHE_IMPL::cache_pair_has_position);
    reveal(CACHE_LAWS::cache_pair_at);
    reveal(CACHE_IMPL::cache_pair_at);
    reveal(CACHE_LAWS::cached_prefix_supported);
    reveal(CACHE_IMPL::cached_prefix_supported);
    reveal(semantic_model);
}

proof fn lemma_cache_reprs_match_canonical_agree(
    caches: Seq<(KVCacheLayerRepr, KVCacheLayerRepr)>,
    wr: ModelWeightsRepr,
    architecture: ModelWeightsArchitectureRepr,
    family: DenseSwiGluDecoderConfigRepr,
    tokens: IntTensor1D,
    cached_tokens: nat,
)
    requires architecture_uses_config(architecture, family),
    ensures
        CACHE_LAWS::cache_reprs_match_canonical(
            caches, semantic_model(wr, architecture), tokens, cached_tokens,
        ) == CACHE_IMPL::cache_reprs_match_canonical(
            dense_swiglu_forward_config_repr(family),
            caches, wr, tokens, cached_tokens,
        ),
{
    reveal(semantic_model);
    reveal(CACHE_LAWS::cache_reprs_match_canonical);
    reveal(CACHE_IMPL::cache_reprs_match_canonical);
    reveal(CACHE_LAWS::cached_prefix_supported);
    reveal(CACHE_IMPL::cached_prefix_supported);
    reveal(CACHE_LAWS::cache_pair_has_position);
    reveal(CACHE_IMPL::cache_pair_has_position);
    reveal(CACHE_LAWS::cache_pair_at);
    reveal(CACHE_IMPL::cache_pair_at);
    if CACHE_LAWS::cache_reprs_match_canonical(
        caches, semantic_model(wr, architecture), tokens, cached_tokens,
    ) {
        assert forall|layer: int, pos: nat|
            0 <= layer < wr.layers.len() && pos < cached_tokens implies {
                &&& CACHE_IMPL::cache_pair_has_position(
                    caches, layer, pos,
                )
                &&& CACHE_IMPL::canonical_kv_defined(
                    dense_swiglu_forward_config_repr(family),
                    wr, tokens, layer, pos,
                )
                &&& CACHE_IMPL::cache_pair_at(caches, layer, pos)
                    == CACHE_IMPL::canonical_kv_at(
                        dense_swiglu_forward_config_repr(family),
                        wr, tokens, layer, pos,
                    )
            }
        by {
            assert(pos < tokens.len());
            assert(CACHE_LAWS::cache_pair_has_position(
                caches, layer, pos,
            ));
            assert(CACHE_LAWS::canonical_kv_defined(
                semantic_model(wr, architecture), tokens, layer, pos,
            ));
            assert(CACHE_LAWS::cache_pair_at(caches, layer, pos)
                == CACHE_LAWS::canonical_kv_at(
                    semantic_model(wr, architecture), tokens, layer, pos,
                ));
            lemma_canonical_kv_vocabulary_agrees(
                wr, architecture, family, tokens, layer, pos,
            );
        }
        assert(CACHE_IMPL::cache_reprs_match_canonical(
            dense_swiglu_forward_config_repr(family),
            caches, wr, tokens, cached_tokens,
        ));
    }
    if CACHE_IMPL::cache_reprs_match_canonical(
        dense_swiglu_forward_config_repr(family),
        caches, wr, tokens, cached_tokens,
    ) {
        assert forall|layer: int, pos: nat|
            0 <= layer < wr.layers.len() && pos < cached_tokens implies {
                &&& CACHE_LAWS::cache_pair_has_position(
                    caches, layer, pos,
                )
                &&& CACHE_LAWS::canonical_kv_defined(
                    semantic_model(wr, architecture), tokens, layer, pos,
                )
                &&& CACHE_LAWS::cache_pair_at(caches, layer, pos)
                    == CACHE_LAWS::canonical_kv_at(
                        semantic_model(wr, architecture), tokens, layer, pos,
                    )
            }
        by {
            assert(pos < tokens.len());
            assert(CACHE_IMPL::cache_pair_has_position(
                caches, layer, pos,
            ));
            assert(CACHE_IMPL::canonical_kv_defined(
                dense_swiglu_forward_config_repr(family),
                wr, tokens, layer, pos,
            ));
            assert(CACHE_IMPL::cache_pair_at(caches, layer, pos)
                == CACHE_IMPL::canonical_kv_at(
                    dense_swiglu_forward_config_repr(family),
                    wr, tokens, layer, pos,
                ));
            lemma_canonical_kv_vocabulary_agrees(
                wr, architecture, family, tokens, layer, pos,
            );
        }
        assert(CACHE_LAWS::cache_reprs_match_canonical(
            caches, semantic_model(wr, architecture), tokens, cached_tokens,
        ));
    }
}

proof fn lemma_cold_reference_cache_canonical(
    wr: ModelWeightsRepr,
    architecture: ModelWeightsArchitectureRepr,
    family: DenseSwiGluDecoderConfigRepr,
)
    requires architecture_uses_config(architecture, family),
    ensures
        CACHE_LAWS::cold_reference_cache_canonical(
            semantic_model(wr, architecture),
        ),
{
    assert forall|tokens: IntTensor1D|
        crate::boundary::tensor_runtime::paged_attention_numeric_domain()
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
        CACHE_IMPL::lemma_cold_reference_cache_matches_canonical(
            dense_swiglu_forward_config_repr(family),
            wr, tokens,
        );
        lemma_cold_reference_cache_reprs_agree(
            wr, architecture, family, tokens,
        );
        lemma_cache_reprs_match_canonical_agree(
            CACHE_LAWS::cold_reference_cache_reprs(
                semantic_model(wr, architecture), tokens,
            ),
            wr,
            architecture,
            family,
            tokens,
            tokens.len(),
        );
    }
}


proof fn lemma_canonical_prefix_continuation_correct(
    wr: ModelWeightsRepr,
    architecture: ModelWeightsArchitectureRepr,
    family: DenseSwiGluDecoderConfigRepr,
)
    requires architecture_uses_config(architecture, family),
    ensures
        CACHE_LAWS::canonical_prefix_continuation_correct(
            semantic_model(wr, architecture),
        ),
{
    assert forall|
        tokens: IntTensor1D,
        prefix_len: nat,
        base: Seq<(KVCacheLayerRepr, KVCacheLayerRepr)>,
    |
        #[trigger] CACHE_LAWS::canonical_prefix_continuation_case(
            semantic_model(wr, architecture), tokens, prefix_len, base,
        )
    by {
        reveal(CACHE_LAWS::canonical_prefix_continuation_case);
        if CACHE_LAWS::prefix_continuation_ready(
            semantic_model(wr, architecture), tokens, prefix_len, base,
        ) {
            reveal(CACHE_LAWS::prefix_continuation_ready);
            let n = tokens.len();
            let q = (n - prefix_len) as nat;
            assert(crate::boundary::tensor_runtime::paged_attention_numeric_domain());
            assert forall|layer: int, pos: nat|
                #![trigger CACHE_IMPL::cache_pair_at(base, layer, pos)]
                #![trigger CACHE_IMPL::cache_pair_has_position(base, layer, pos)]
                0 <= layer < wr.layers.len() && pos < prefix_len implies {
                    &&& CACHE_IMPL::cache_pair_has_position(base, layer, pos)
                    &&& CACHE_IMPL::canonical_kv_defined(
                        dense_swiglu_forward_config_repr(family),
                        wr, tokens, layer, pos,
                    )
                    &&& CACHE_IMPL::cache_pair_at(base, layer, pos)
                        == CACHE_IMPL::canonical_kv_at(
                            dense_swiglu_forward_config_repr(family),
                            wr, tokens, layer, pos,
                        )
                }
            by {
                assert(CACHE_LAWS::cache_pair_has_position(
                    base, layer, pos,
                ));
                assert(CACHE_LAWS::canonical_kv_defined(
                    semantic_model(wr, architecture), tokens, layer, pos,
                ));
                assert(CACHE_LAWS::cache_pair_at(base, layer, pos)
                    == CACHE_LAWS::canonical_kv_at(
                        semantic_model(wr, architecture), tokens, layer, pos,
                    ));
                lemma_canonical_kv_vocabulary_agrees(
                    wr, architecture, family, tokens, layer, pos,
                );
                reveal(CACHE_LAWS::cache_pair_has_position);
                reveal(CACHE_IMPL::cache_pair_has_position);
                reveal(CACHE_LAWS::cache_pair_at);
                reveal(CACHE_IMPL::cache_pair_at);
            }
            CACHE_IMPL::lemma_canonical_prefix_base_continuation_matches_cold_prefill(
                dense_swiglu_forward_config_repr(family),
                wr, tokens, prefix_len, base,
            );
            lemma_cold_reference_cache_reprs_agree(
                wr, architecture, family, tokens,
            );
            lemma_cold_reference_logits_repr_agree(
                wr, architecture, family, tokens,
            );
            reveal(CACHE_LAWS::prefix_continuation_logits_repr);
            reveal(CACHE_LAWS::prefix_continuation_cache_reprs);
            reveal(ARCH::model_forward_logits_repr);
            reveal(ARCH::model_forward_kv_reprs);
            reveal(forward_logits_repr);
            reveal(forward_kv_reprs);
            reveal(CACHE_LAWS::cache_sequences_agree_on_prefix_in_range);
            reveal(CACHE_IMPL::cache_sequences_agree_on_prefix_in_range);
            reveal(CACHE_LAWS::prefix_continuation_matches_cold);
        }
    }
    CACHE_LAWS::lemma_canonical_prefix_continuation_correct_intro(
        semantic_model(wr, architecture),
    );
}

pub proof fn lemma_cache_refinement_laws(
    wr: ModelWeightsRepr,
    architecture: ModelWeightsArchitectureRepr,
    family: DenseSwiGluDecoderConfigRepr,
)
    requires
        architecture_uses_config(architecture, family),
        semantic_model_repr_valid(family_semantic_model(wr, architecture)),
    ensures
        CACHE_LAWS::cache_refinement_laws(
            family_semantic_model(wr, architecture),
        ),
{
    lemma_cold_reference_cache_canonical(wr, architecture, family);
    lemma_canonical_prefix_continuation_correct(wr, architecture, family);
    reveal(CACHE_LAWS::cache_refinement_laws);
}

} // verus!
