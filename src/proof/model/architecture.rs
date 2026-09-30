//! Architecture-neutral dispatch for whole-model semantic capstones.
//!
//! Closed architecture payloads select shared composition capstones. Family
//! configuration checks stay at this boundary; the shared proofs do not copy
//! a decoder induction or assume a concrete model tag.

#[cfg(verus_only)]
use crate::proof::model::families::dense_swiglu as DENSE;
#[cfg(verus_only)]
use crate::proof::model::families::four_norm_gated as FOUR;
#[cfg(verus_only)]
use crate::proof::model::four_norm_gated::layer_witnesses as WITNESS;
use crate::{types::*, proof::model::types::*, proof::tensor::types::*};
use vstd::prelude::*;

verus! {

// The architecture adapter selects a composition; the cache-refinement proof
// consumes only that composition, never a concrete family tag.
pub open spec fn four_norm_decoder_config(
    architecture: ModelWeightsArchitectureRepr,
) -> Option<FourNormGatedDecoderConfigRepr> {
    match architecture {
        ModelWeightsArchitectureRepr::Gemma3Text(config) => Some(config),
        ModelWeightsArchitectureRepr::Gemma4Text(config) => Some(config.decoder),
        ModelWeightsArchitectureRepr::Qwen3(_) => None,
        ModelWeightsArchitectureRepr::Llama3(_) => None,
    }
}

pub broadcast proof fn lemma_four_norm_forward_composition(
    wr: ModelWeightsRepr,
    architecture: ModelWeightsArchitectureRepr,
    input_ids: IntTensor1D,
    positions: IntTensor1D,
    pre_kv: Seq<(KVCacheLayerRepr, KVCacheLayerRepr)>,
    slots: Seq<int>, cu_q: Seq<int>, cu_k: Seq<int>,
    max_q: nat, max_k: nat, block_table: Seq<Seq<BlockId>>,
)
    ensures
        four_norm_decoder_config(architecture).is_some() ==> {
            let config = four_norm_decoder_config(architecture).unwrap();
            &&& #[trigger] model_forward_logits_repr(
                wr, architecture, input_ids, positions, pre_kv, slots,
                cu_q, cu_k, max_q, max_k, block_table)
                == crate::proof::model::four_norm_gated::model::model_forward_logits_repr(
                    wr, config, input_ids, positions, pre_kv, slots,
                    cu_q, cu_k, max_q, max_k, block_table)
            &&& #[trigger] model_forward_kv_reprs(
                wr, architecture, input_ids, positions, pre_kv, slots,
                cu_q, cu_k, max_q, max_k, block_table)
                == crate::proof::model::four_norm_gated::model::model_forward_kv_reprs(
                    wr, config, input_ids, positions, pre_kv, slots,
                    cu_q, cu_k, max_q, max_k, block_table)
        },
{
    match architecture {
        ModelWeightsArchitectureRepr::Gemma3Text(_) => {
            reveal(FOUR::forward_logits_repr);
            reveal(FOUR::forward_kv_reprs);
        },
        ModelWeightsArchitectureRepr::Gemma4Text(_) => {
            reveal(FOUR::forward_logits_repr);
            reveal(FOUR::forward_kv_reprs);
        },
        ModelWeightsArchitectureRepr::Qwen3(_) => {},
        ModelWeightsArchitectureRepr::Llama3(_) => {},
    }
}

pub open spec fn model_forward_logits_repr(
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
) -> Tensor2D {
    match architecture_repr {
        ModelWeightsArchitectureRepr::Qwen3(family) =>
            DENSE::forward_logits_repr(
                wr, family,
                input_ids, positions, pre_kv, slots,
                cu_q, cu_k, max_q, max_k, block_table,
            ),
        ModelWeightsArchitectureRepr::Llama3(family) =>
            DENSE::forward_logits_repr(
                wr, family,
                input_ids, positions, pre_kv, slots,
                cu_q, cu_k, max_q, max_k, block_table,
            ),
        ModelWeightsArchitectureRepr::Gemma3Text(family) =>
            FOUR::forward_logits_repr(
                wr, family, input_ids, positions, pre_kv, slots,
                cu_q, cu_k, max_q, max_k, block_table,
            ),
        ModelWeightsArchitectureRepr::Gemma4Text(family) =>
            FOUR::forward_logits_repr(
                wr, family.decoder, input_ids, positions, pre_kv, slots,
                cu_q, cu_k, max_q, max_k, block_table,
            ),
    }
}

pub open spec fn model_forward_kv_reprs(
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
) -> Seq<(KVCacheLayerRepr, KVCacheLayerRepr)> {
    match architecture_repr {
        ModelWeightsArchitectureRepr::Qwen3(family) =>
            DENSE::forward_kv_reprs(
                wr, family,
                input_ids, positions, pre_kv, slots,
                cu_q, cu_k, max_q, max_k, block_table,
            ),
        ModelWeightsArchitectureRepr::Llama3(family) =>
            DENSE::forward_kv_reprs(
                wr, family,
                input_ids, positions, pre_kv, slots,
                cu_q, cu_k, max_q, max_k, block_table,
            ),
        ModelWeightsArchitectureRepr::Gemma3Text(family) =>
            FOUR::forward_kv_reprs(
                wr, family, input_ids, positions, pre_kv, slots,
                cu_q, cu_k, max_q, max_k, block_table,
            ),
        ModelWeightsArchitectureRepr::Gemma4Text(family) =>
            FOUR::forward_kv_reprs(
                wr, family.decoder, input_ids, positions, pre_kv, slots,
                cu_q, cu_k, max_q, max_k, block_table,
            ),
    }
}

// Per-layer K/V rows presented to the scatter store.  Scheduler/cache proofs
// consume this architecture-neutral witness instead of reconstructing a
// family's decoder fold.
pub open spec fn model_layer_kv_rows(
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
    layer: nat,
) -> (Tensor2D, Tensor2D) {
    match architecture_repr {
        ModelWeightsArchitectureRepr::Qwen3(family) =>
            DENSE::layer_kv_rows(
                wr, family, input_ids, positions, pre_kv, slots,
                cu_q, cu_k, max_q, max_k, block_table, layer,
            ),
        ModelWeightsArchitectureRepr::Llama3(family) =>
            DENSE::layer_kv_rows(
                wr, family, input_ids, positions, pre_kv, slots,
                cu_q, cu_k, max_q, max_k, block_table, layer,
            ),
        ModelWeightsArchitectureRepr::Gemma3Text(family) =>
            WITNESS::layer_kv_rows(
                wr, family, input_ids, positions, pre_kv, slots,
                cu_q, cu_k, max_q, max_k, block_table, layer,
            ),
        ModelWeightsArchitectureRepr::Gemma4Text(family) =>
            WITNESS::layer_kv_rows(
                wr, family.decoder, input_ids, positions, pre_kv, slots,
                cu_q, cu_k, max_q, max_k, block_table, layer,
            ),
    }
}

pub proof fn lemma_model_forward_layer_store(
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
    layer: nat,
)
    requires
        model_weights_architecture_repr_valid(wr, architecture_repr),
        layer < wr.layers.len(),
        pre_kv.len() == wr.layers.len(),
        input_ids.len() == positions.len(),
        slots.len() == input_ids.len(),
    ensures ({
        let rows = model_layer_kv_rows(
            wr, architecture_repr, input_ids, positions, pre_kv, slots,
            cu_q, cu_k, max_q, max_k, block_table, layer,
        );
        model_forward_kv_reprs(
            wr, architecture_repr, input_ids, positions, pre_kv, slots,
            cu_q, cu_k, max_q, max_k, block_table,
        )[layer as int] == crate::boundary::tensor_runtime::store_kv_cache_repr(
            rows.0, rows.1,
            pre_kv[layer as int].0, pre_kv[layer as int].1, slots,
        )
    }),
{
    match architecture_repr {
        ModelWeightsArchitectureRepr::Qwen3(family) => {
            DENSE::lemma_forward_layer_store(
                wr, family, input_ids, positions, pre_kv, slots,
                cu_q, cu_k, max_q, max_k, block_table, layer,
            );
        },
        ModelWeightsArchitectureRepr::Llama3(family) => {
            DENSE::lemma_forward_layer_store(
                wr, family, input_ids, positions, pre_kv, slots,
                cu_q, cu_k, max_q, max_k, block_table, layer,
            );
        },
        ModelWeightsArchitectureRepr::Gemma3Text(family) => {
            WITNESS::lemma_forward_layer_store(
                wr, family, input_ids, positions, pre_kv, slots,
                cu_q, cu_k, max_q, max_k, block_table, layer,
            );
        },
        ModelWeightsArchitectureRepr::Gemma4Text(family) => {
            WITNESS::lemma_forward_layer_store(
                wr, family.decoder, input_ids, positions, pre_kv, slots,
                cu_q, cu_k, max_q, max_k, block_table, layer,
            );
        },
    }
}

pub proof fn lemma_model_layer_kv_rows_shape(
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
    layer: nat,
)
    requires
        model_weights_architecture_repr_valid(wr, architecture_repr),
        layer < wr.layers.len(),
        pre_kv.len() == wr.layers.len(),
        input_ids.len() == positions.len(),
        slots.len() == input_ids.len(),
    ensures ({
        let rows = model_layer_kv_rows(
            wr, architecture_repr, input_ids, positions, pre_kv, slots,
            cu_q, cu_k, max_q, max_k, block_table, layer,
        );
        rows.0.len() == input_ids.len()
            && rows.1.len() == input_ids.len()
    }),
{
    match architecture_repr {
        ModelWeightsArchitectureRepr::Qwen3(family) => {
            DENSE::lemma_layer_kv_rows_shape(
                wr, family, input_ids, positions, pre_kv, slots,
                cu_q, cu_k, max_q, max_k, block_table, layer,
            );
        },
        ModelWeightsArchitectureRepr::Llama3(family) => {
            DENSE::lemma_layer_kv_rows_shape(
                wr, family, input_ids, positions, pre_kv, slots,
                cu_q, cu_k, max_q, max_k, block_table, layer,
            );
        },
        ModelWeightsArchitectureRepr::Gemma3Text(family) => {
            WITNESS::lemma_layer_kv_rows_shape(
                wr, family, input_ids, positions, pre_kv, slots,
                cu_q, cu_k, max_q, max_k, block_table, layer,
            );
        },
        ModelWeightsArchitectureRepr::Gemma4Text(family) => {
            WITNESS::lemma_layer_kv_rows_shape(
                wr, family.decoder, input_ids, positions, pre_kv, slots,
                cu_q, cu_k, max_q, max_k, block_table, layer,
            );
        },
    }
}

// Architecture-neutral request projection for the K/V rows presented to a
// layer's scatter store. This is the sole family-dispatch hook needed by the
// generic padded-decode cover proof.
pub proof fn lemma_model_layer_kv_rows_request_isolation(
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
    i: nat,
    layer: nat,
)
    requires
        model_forward_request_projection_domain(
            wr, architecture_repr, input_ids, positions, pre_kv, slots,
            cu_q, cu_k, max_q, max_k, block_table, i,
        ),
        layer < wr.layers.len(),
    ensures ({
        let lo = cu_q[i as int];
        let hi = cu_q[i as int + 1];
        let q_len = (hi - lo) as nat;
        let k_len = (cu_k[i as int + 1] - cu_k[i as int]) as nat;
        let full = model_layer_kv_rows(
            wr, architecture_repr, input_ids, positions, pre_kv, slots,
            cu_q, cu_k, max_q, max_k, block_table, layer,
        );
        let single = model_layer_kv_rows(
            wr, architecture_repr,
            input_ids.subrange(lo, hi), positions.subrange(lo, hi),
            pre_kv, slots.subrange(lo, hi),
            seq![0int, q_len as int], seq![0int, k_len as int],
            q_len, k_len, seq![block_table[i as int]], layer,
        );
        &&& full.0.subrange(lo, hi) == single.0
        &&& full.1.subrange(lo, hi) == single.1
    }),
{
    reveal(model_forward_request_projection_domain);
    match architecture_repr {
        ModelWeightsArchitectureRepr::Qwen3(family) => {
            DENSE::lemma_layer_kv_rows_request_isolation(
                wr, family, input_ids, positions, pre_kv, slots,
                cu_q, cu_k, max_q, max_k, block_table, i, layer,
            );
        },
        ModelWeightsArchitectureRepr::Llama3(family) => {
            DENSE::lemma_layer_kv_rows_request_isolation(
                wr, family, input_ids, positions, pre_kv, slots,
                cu_q, cu_k, max_q, max_k, block_table, i, layer,
            );
        },
        ModelWeightsArchitectureRepr::Gemma3Text(family) => {
            WITNESS::lemma_layer_kv_rows_request_isolation(
                wr, family, input_ids, positions, pre_kv, slots,
                cu_q, cu_k, max_q, max_k, block_table, i, layer,
            );
        },
        ModelWeightsArchitectureRepr::Gemma4Text(family) => {
            WITNESS::lemma_layer_kv_rows_request_isolation(
                wr, family.decoder, input_ids, positions, pre_kv, slots,
                cu_q, cu_k, max_q, max_k, block_table, i, layer,
            );
        },
    }
}

// A model forward can mutate a cache layer only at the common scatter-slot
// sequence.  This frame lemma is architecture-neutral because each family has
// already lowered its layer fold to `model_layer_kv_rows` plus the shared
// scatter store above.
pub proof fn lemma_model_forward_kv_preserves_unwritten_slot(
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
    layer: nat,
    slot: nat,
)
    requires
        model_weights_architecture_repr_valid(wr, architecture_repr),
        layer < wr.layers.len(),
        pre_kv.len() == wr.layers.len(),
        input_ids.len() == positions.len(),
        slots.len() == input_ids.len(),
        !slots.contains(slot as int),
        crate::proof::tensor::geometry::slot_in_cache(pre_kv[layer as int].0, slot),
        crate::proof::tensor::geometry::slot_in_cache(pre_kv[layer as int].1, slot),
    ensures ({
        let post = model_forward_kv_reprs(
            wr, architecture_repr, input_ids, positions, pre_kv, slots,
            cu_q, cu_k, max_q, max_k, block_table,
        )[layer as int];
        &&& crate::proof::tensor::geometry::slot_in_cache(post.0, slot)
        &&& crate::proof::tensor::geometry::slot_in_cache(post.1, slot)
        &&& crate::proof::tensor::geometry::cache_at(post.0, slot)
            == crate::proof::tensor::geometry::cache_at(pre_kv[layer as int].0, slot)
        &&& crate::proof::tensor::geometry::cache_at(post.1, slot)
            == crate::proof::tensor::geometry::cache_at(pre_kv[layer as int].1, slot)
    }),
{
    let rows = model_layer_kv_rows(
        wr, architecture_repr, input_ids, positions, pre_kv, slots,
        cu_q, cu_k, max_q, max_k, block_table, layer,
    );
    lemma_model_layer_kv_rows_shape(
        wr, architecture_repr, input_ids, positions, pre_kv, slots,
        cu_q, cu_k, max_q, max_k, block_table, layer,
    );
    lemma_model_forward_layer_store(
        wr, architecture_repr, input_ids, positions, pre_kv, slots,
        cu_q, cu_k, max_q, max_k, block_table, layer,
    );
    crate::boundary::tensor_runtime::store_kv_cache_repr_preserves_unwritten_slots(
        rows.0,
        rows.1,
        pre_kv[layer as int].0,
        pre_kv[layer as int].1,
        slots,
        slot,
    );
}

// Canonical cold full-history observation for one semantic model.  This is
// the only reference-logit function consumed above the family boundary.
pub open spec fn reference_logits_last_row(
    model: SemanticModelRepr,
    history: IntTensor1D,
) -> Tensor1D
    recommends
        semantic_model_repr_valid(model),
        model.weights.layers.len() > 0,
        history.len() > 0,
{
    match model.architecture {
        ModelWeightsArchitectureRepr::Qwen3(family) =>
            DENSE::reference_logits_last_row(
                model.weights, family, history,
            ),
        ModelWeightsArchitectureRepr::Llama3(family) =>
            DENSE::reference_logits_last_row(
                model.weights, family, history,
            ),
        ModelWeightsArchitectureRepr::Gemma3Text(family) =>
            FOUR::reference_logits_last_row(
                model.weights, family, history,
            ),
        ModelWeightsArchitectureRepr::Gemma4Text(family) =>
            FOUR::reference_logits_last_row(
                model.weights, family.decoder, history,
            ),
    }
}

// Closed-dispatch reduction used by architecture-neutral observable proofs:
// the public cold observation is the last row of the ordinary singleton cold
// forward.  No family constructor escapes this module.
pub proof fn lemma_reference_logits_last_row_is_forward_last(
    model: SemanticModelRepr,
    tokens: IntTensor1D,
)
    requires
        semantic_model_repr_valid(model),
        model.weights.layers.len() > 0,
        tokens.len() > 0,
    ensures
        reference_logits_last_row(model, tokens)
            == model_forward_logits_repr(
                model.weights,
                model.architecture,
                tokens,
                crate::proof::tensor::geometry::positions_from(0, tokens.len()),
                crate::proof::reference::request_machine::synthetic_cache_reprs(
                    tokens.len(), model.weights.layers.len(),
                ),
                crate::proof::reference::request_machine::slots_from(0, tokens.len()),
                crate::proof::reference::request_machine::seq_lens_for_single(tokens.len()),
                crate::proof::reference::request_machine::seq_lens_for_single(tokens.len()),
                tokens.len(),
                tokens.len(),
                crate::proof::reference::request_machine::singleton_block_rows(tokens.len()),
            )[tokens.len() as int - 1],
{
    reveal(reference_logits_last_row);
    reveal(model_forward_logits_repr);
    match model.architecture {
        ModelWeightsArchitectureRepr::Qwen3(family) => {
            DENSE::lemma_reference_logits_last_row_is_forward_last(
                model.weights, family, tokens,
            );
        },
        ModelWeightsArchitectureRepr::Llama3(family) => {
            DENSE::lemma_reference_logits_last_row_is_forward_last(
                model.weights, family, tokens,
            );
        },
        ModelWeightsArchitectureRepr::Gemma3Text(family) => {
            FOUR::lemma_reference_logits_last_row_is_forward_last(
                model.weights, family, tokens,
            );
        },
        ModelWeightsArchitectureRepr::Gemma4Text(family) => {
            FOUR::lemma_reference_logits_last_row_is_forward_last(
                model.weights, family.decoder, tokens,
            );
        },
    }
}

// Formal support boundary for the cache-canonicality refinement.  Every
// family exposes the same capability; a family is enabled here only after its
// cache and continuation laws are checked.  Forward execution support alone
// does not imply refinement support.
pub open spec fn cache_refinement_supported(model: SemanticModelRepr) -> bool {
    match model.architecture {
        ModelWeightsArchitectureRepr::Qwen3(family) =>
            DENSE::cache_refinement_supported(
                model.weights, model.architecture, family,
            ),
        ModelWeightsArchitectureRepr::Llama3(family) =>
            DENSE::cache_refinement_supported(
                model.weights, model.architecture, family,
            ),
        ModelWeightsArchitectureRepr::Gemma3Text(family) =>
            gemma3_config_valid(family)
            && FOUR::cache_refinement_supported(model.weights, family),
        ModelWeightsArchitectureRepr::Gemma4Text(family) =>
            gemma4_config_valid(family)
            && FOUR::cache_refinement_supported(model.weights, family.decoder),
    }
}

// Architecture-specific configuration facts needed by the otherwise common
// request-projection and relocation domains.  Keeping this closed dispatch at
// the model boundary prevents physical-layout proofs from naming families.
pub open spec fn request_projection_configuration_ready(
    wr: ModelWeightsRepr,
    architecture_repr: ModelWeightsArchitectureRepr,
) -> bool {
    match architecture_repr {
        ModelWeightsArchitectureRepr::Qwen3(family) =>
            family.geometry.num_layers == wr.layers.len()
                && wr.layers.len() > 0,
        ModelWeightsArchitectureRepr::Llama3(family) =>
            family.geometry.num_layers == wr.layers.len()
                && wr.layers.len() > 0,
        ModelWeightsArchitectureRepr::Gemma3Text(family) =>
            family.layers.len() == wr.layers.len()
                && forall|layer: int| 0 <= layer < family.layers.len() ==>
                    layer_attention_config_valid(
                        #[trigger] family.layers[layer].attention,
                    ),
        ModelWeightsArchitectureRepr::Gemma4Text(family) =>
            family.decoder.layers.len() == wr.layers.len()
                && forall|layer: int| 0 <= layer < family.decoder.layers.len() ==>
                    layer_attention_config_valid(
                        #[trigger] family.decoder.layers[layer].attention,
                    ),
    }
}

pub proof fn lemma_cache_refinement_support_implies_projection_configuration_ready(
    model: SemanticModelRepr,
)
    requires
        semantic_model_repr_valid(model),
        cache_refinement_supported(model),
        model.weights.layers.len() > 0,
    ensures
        request_projection_configuration_ready(
            model.weights,
            model.architecture,
        ),
{
    reveal(cache_refinement_supported);
    reveal(request_projection_configuration_ready);
    match model.architecture {
        ModelWeightsArchitectureRepr::Qwen3(_) => {},
        ModelWeightsArchitectureRepr::Llama3(_) => {},
        ModelWeightsArchitectureRepr::Gemma3Text(_) => {},
        ModelWeightsArchitectureRepr::Gemma4Text(_) => {},
    }
}

// Every enabled family discharges the same architecture-neutral cache laws.
// Adding another decoder family extends only this closed dispatch and its
// family-local adapter proof.
pub proof fn lemma_cache_refinement_laws(model: SemanticModelRepr)
    requires
        semantic_model_repr_valid(model),
        cache_refinement_supported(model),
    ensures crate::proof::model::cache::cache_refinement_laws(model),
{
    reveal(cache_refinement_supported);
    match model.architecture {
        ModelWeightsArchitectureRepr::Qwen3(family) => {
            assert(model == DENSE::semantic_model(
                model.weights, model.architecture,
            ));
            DENSE::lemma_cache_refinement_laws(
                model.weights, model.architecture, family,
            );
        },
        ModelWeightsArchitectureRepr::Llama3(family) => {
            assert(model == DENSE::semantic_model(
                model.weights, model.architecture,
            ));
            DENSE::lemma_cache_refinement_laws(
                model.weights, model.architecture, family,
            );
        },
        ModelWeightsArchitectureRepr::Gemma3Text(family) => {
            FOUR::lemma_cache_refinement_laws(
                model.weights, family, model.architecture);
        },
        ModelWeightsArchitectureRepr::Gemma4Text(family) => {
            FOUR::lemma_cache_refinement_laws(
                model.weights, family.decoder, model.architecture);
        },
    }
}

// One closed dispatch point for the architecture-specific projection domain.
// This fact deliberately does not constrain `wr.architecture`: family reference
// lemmas quantify over the compact common weight record before it is paired
// with an executable architecture capability.
pub open spec fn model_forward_request_projection_domain(
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
    i: nat,
) -> bool {
    match architecture_repr {
        ModelWeightsArchitectureRepr::Qwen3(family) => DENSE::request_projection_ready(
            wr, family,
            input_ids, positions, pre_kv, slots,
            cu_q, cu_k, max_q, max_k, block_table, i,
        ),
        ModelWeightsArchitectureRepr::Llama3(family) => DENSE::request_projection_ready(
            wr, family,
            input_ids, positions, pre_kv, slots,
            cu_q, cu_k, max_q, max_k, block_table, i,
        ),
        ModelWeightsArchitectureRepr::Gemma3Text(family) =>
            FOUR::request_projection_ready(
                wr, family, input_ids, positions, pre_kv, slots,
                cu_q, cu_k, max_q, max_k, block_table, i,
            ),
        ModelWeightsArchitectureRepr::Gemma4Text(family) =>
            FOUR::request_projection_ready(
                wr, family.decoder, input_ids, positions, pre_kv, slots,
                cu_q, cu_k, max_q, max_k, block_table, i,
            ),
    }
}

pub proof fn lemma_model_forward_request_projection_domain_from_common(
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
)
    requires
        crate::proof::model::family_layout::request_projection_common_domain(
            wr,
            architecture_repr,
            input_ids,
            positions,
            pre_kv,
            slots,
            cu_q,
            cu_k,
            max_q,
            max_k,
            block_table,
            row,
        ),
    ensures
        model_forward_request_projection_domain(
            wr,
            architecture_repr,
            input_ids,
            positions,
            pre_kv,
            slots,
            cu_q,
            cu_k,
            max_q,
            max_k,
            block_table,
            row,
        ),
{
    reveal(model_forward_request_projection_domain);
    match architecture_repr {
        ModelWeightsArchitectureRepr::Qwen3(family) => {
            DENSE::lemma_request_projection_ready_from_common_domain(
                wr, architecture_repr, family, input_ids, positions, pre_kv, slots,
                cu_q, cu_k, max_q, max_k, block_table, row,
            );
        },
        ModelWeightsArchitectureRepr::Llama3(family) => {
            DENSE::lemma_request_projection_ready_from_common_domain(
                wr, architecture_repr, family, input_ids, positions, pre_kv, slots,
                cu_q, cu_k, max_q, max_k, block_table, row,
            );
        },
        ModelWeightsArchitectureRepr::Gemma3Text(family) => {
            FOUR::lemma_request_projection_ready_from_common_domain(
                wr, family, architecture_repr, input_ids, positions, pre_kv, slots,
                cu_q, cu_k, max_q, max_k, block_table, row,
            );
        },
        ModelWeightsArchitectureRepr::Gemma4Text(family) => {
            FOUR::lemma_request_projection_ready_from_common_domain(
                wr, family.decoder, architecture_repr, input_ids, positions, pre_kv, slots,
                cu_q, cu_k, max_q, max_k, block_table, row,
            );
        },
    }
}

// Architecture-neutral logits projection.  The model forward remains opaque
// to consumers: this proof performs the sole architecture split, then invokes
// the already-checked family-specific capstone.
pub proof fn lemma_model_forward_logits_request_isolation(
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
    i: nat,
)
    requires model_forward_request_projection_domain(
        wr, architecture_repr, input_ids, positions, pre_kv, slots,
        cu_q, cu_k, max_q, max_k, block_table, i,
    ),
    ensures
        model_forward_logits_repr(
            wr, architecture_repr, input_ids, positions, pre_kv, slots,
            cu_q, cu_k, max_q, max_k, block_table,
        ).subrange(cu_q[i as int], cu_q[i as int + 1])
        == model_forward_logits_repr(
            wr, architecture_repr,
            input_ids.subrange(cu_q[i as int], cu_q[i as int + 1]),
            positions.subrange(cu_q[i as int], cu_q[i as int + 1]),
            pre_kv, slots.subrange(cu_q[i as int], cu_q[i as int + 1]),
            seq![0int, cu_q[i as int + 1] - cu_q[i as int]],
            seq![0int, cu_k[i as int + 1] - cu_k[i as int]],
            (cu_q[i as int + 1] - cu_q[i as int]) as nat,
            (cu_k[i as int + 1] - cu_k[i as int]) as nat,
            seq![block_table[i as int]],
        ),
{
    reveal(model_forward_request_projection_domain);
    match architecture_repr {
        ModelWeightsArchitectureRepr::Qwen3(family) => {
            DENSE::lemma_logits_request_isolation(
                wr, family,
                input_ids, positions, pre_kv, slots,
                cu_q, cu_k, max_q, max_k, block_table, i,
            );
        },
        ModelWeightsArchitectureRepr::Llama3(family) => {
            DENSE::lemma_logits_request_isolation(
                wr, family,
                input_ids, positions, pre_kv, slots,
                cu_q, cu_k, max_q, max_k, block_table, i,
            );
        },
        ModelWeightsArchitectureRepr::Gemma3Text(family) => {
            FOUR::lemma_logits_request_isolation(
                wr, family, input_ids, positions, pre_kv, slots,
                cu_q, cu_k, max_q, max_k, block_table, i,
            );
        },
        ModelWeightsArchitectureRepr::Gemma4Text(family) => {
            FOUR::lemma_logits_request_isolation(
                wr, family.decoder, input_ids, positions, pre_kv, slots,
                cu_q, cu_k, max_q, max_k, block_table, i,
            );
        },
    }
}

// Architecture-neutral cache projection for one request.  Both family
// adapters lower their internal attention/cache facts to the same logical
// causal-prefix predicate, so downstream refinement never inspects a decoder
// layer or an attention kind.
pub proof fn lemma_model_forward_kv_request_isolation(
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
    i: nat,
)
    requires model_forward_request_projection_domain(
        wr, architecture_repr, input_ids, positions, pre_kv, slots,
        cu_q, cu_k, max_q, max_k, block_table, i,
    ),
    ensures ({
        let lo = cu_q[i as int];
        let hi = cu_q[i as int + 1];
        let q_len = (hi - lo) as nat;
        let k_len = (cu_k[i as int + 1] - cu_k[i as int]) as nat;
        crate::proof::model::family_layout::cache_sequence_logical_prefix_equal(
            model_forward_kv_reprs(
                wr, architecture_repr,
                input_ids, positions, pre_kv, slots,
                cu_q, cu_k, max_q, max_k, block_table,
            ),
            block_table[i as int],
            model_forward_kv_reprs(
                wr, architecture_repr,
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
    reveal(model_forward_request_projection_domain);
    match architecture_repr {
        ModelWeightsArchitectureRepr::Qwen3(family) => {
            DENSE::lemma_kv_request_isolation(
                wr, family, input_ids, positions, pre_kv, slots,
                cu_q, cu_k, max_q, max_k, block_table, i,
            );
        },
        ModelWeightsArchitectureRepr::Llama3(family) => {
            DENSE::lemma_kv_request_isolation(
                wr, family, input_ids, positions, pre_kv, slots,
                cu_q, cu_k, max_q, max_k, block_table, i,
            );
        },
        ModelWeightsArchitectureRepr::Gemma3Text(family) => {
            WITNESS::lemma_kv_request_isolation(
                wr, family, input_ids, positions, pre_kv, slots,
                cu_q, cu_k, max_q, max_k, block_table, i,
            );
        },
        ModelWeightsArchitectureRepr::Gemma4Text(family) => {
            WITNESS::lemma_kv_request_isolation(
                wr, family.decoder, input_ids, positions, pre_kv, slots,
                cu_q, cu_k, max_q, max_k, block_table, i,
            );
        },
    }
}

pub proof fn lemma_model_forward_logits_repr_shape(
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
)
    requires
        model_weights_architecture_repr_valid(wr, architecture_repr),
        input_ids.len() == positions.len(),
        input_ids.len() == slots.len(),
        pre_kv.len() == wr.layers.len(),
    ensures model_forward_logits_repr(
        wr, architecture_repr, input_ids, positions, pre_kv, slots,
        cu_q, cu_k, max_q, max_k, block_table,
    ).len() == input_ids.len(),
{
    match architecture_repr {
        ModelWeightsArchitectureRepr::Qwen3(family) => {
            DENSE::lemma_logits_repr_shape(
                wr, family,
                input_ids, positions, pre_kv, slots,
                cu_q, cu_k, max_q, max_k, block_table,
            );
        },
        ModelWeightsArchitectureRepr::Llama3(family) => {
            DENSE::lemma_logits_repr_shape(
                wr, family,
                input_ids, positions, pre_kv, slots,
                cu_q, cu_k, max_q, max_k, block_table,
            );
        },
        ModelWeightsArchitectureRepr::Gemma3Text(family) => {
            FOUR::lemma_logits_repr_shape(
                wr, family, input_ids, positions, pre_kv, slots,
                cu_q, cu_k, max_q, max_k, block_table,
            );
        },
        ModelWeightsArchitectureRepr::Gemma4Text(family) => {
            FOUR::lemma_logits_repr_shape(
                wr, family.decoder, input_ids, positions, pre_kv, slots,
                cu_q, cu_k, max_q, max_k, block_table,
            );
        },
    }
}

pub proof fn lemma_model_forward_kv_reprs_len(
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
)
    requires
        model_weights_architecture_repr_valid(wr, architecture_repr),
        input_ids.len() == positions.len(),
        input_ids.len() == slots.len(),
        pre_kv.len() >= wr.layers.len(),
    ensures model_forward_kv_reprs(
        wr, architecture_repr, input_ids, positions, pre_kv, slots,
        cu_q, cu_k, max_q, max_k, block_table,
    ).len() == pre_kv.len(),
{
    match architecture_repr {
        ModelWeightsArchitectureRepr::Qwen3(family) => {
            DENSE::lemma_kv_reprs_len(
                wr, family,
                input_ids, positions, pre_kv, slots,
                cu_q, cu_k, max_q, max_k, block_table,
            );
        },
        ModelWeightsArchitectureRepr::Llama3(family) => {
            DENSE::lemma_kv_reprs_len(
                wr, family,
                input_ids, positions, pre_kv, slots,
                cu_q, cu_k, max_q, max_k, block_table,
            );
        },
        ModelWeightsArchitectureRepr::Gemma3Text(family) => {
            FOUR::lemma_kv_reprs_len(
                wr, family, input_ids, positions, pre_kv, slots,
                cu_q, cu_k, max_q, max_k, block_table,
            );
        },
        ModelWeightsArchitectureRepr::Gemma4Text(family) => {
            FOUR::lemma_kv_reprs_len(
                wr, family.decoder, input_ids, positions, pre_kv, slots,
                cu_q, cu_k, max_q, max_k, block_table,
            );
        },
    }
}

pub proof fn lemma_model_forward_kv_reprs_empty(
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
)
    requires
        model_weights_architecture_repr_valid(wr, architecture_repr),
        pre_kv.len() >= wr.layers.len(),
        input_ids.len() == 0,
        positions.len() == 0,
        slots.len() == 0,
    ensures
        model_forward_kv_reprs(
            wr, architecture_repr, input_ids, positions, pre_kv, slots,
            cu_q, cu_k, max_q, max_k, block_table,
        ) == pre_kv,
{
    match architecture_repr {
        ModelWeightsArchitectureRepr::Qwen3(family) => {
            DENSE::lemma_kv_reprs_empty(
                wr, family,
                input_ids, positions, pre_kv, slots,
                cu_q, cu_k, max_q, max_k, block_table,
            );
        },
        ModelWeightsArchitectureRepr::Llama3(family) => {
            DENSE::lemma_kv_reprs_empty(
                wr, family,
                input_ids, positions, pre_kv, slots,
                cu_q, cu_k, max_q, max_k, block_table,
            );
        },
        ModelWeightsArchitectureRepr::Gemma3Text(family) => {
            FOUR::lemma_kv_reprs_empty(
                wr, family, input_ids, positions, pre_kv, slots,
                cu_q, cu_k, max_q, max_k, block_table,
            );
        },
        ModelWeightsArchitectureRepr::Gemma4Text(family) => {
            FOUR::lemma_kv_reprs_empty(
                wr, family.decoder, input_ids, positions, pre_kv, slots,
                cu_q, cu_k, max_q, max_k, block_table,
            );
        },
    }
}

pub proof fn lemma_model_forward_cache_shape_preserved(
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
    num_pages: nat,
)
    requires
        model_weights_architecture_repr_valid(wr, architecture_repr),
        pre_kv.len() == wr.layers.len(),
        input_ids.len() == positions.len(),
        input_ids.len() == slots.len(),
        crate::proof::model::cache::cache_sequence_page_shape(pre_kv, num_pages),
    ensures
        crate::proof::model::cache::cache_sequence_page_shape(
            model_forward_kv_reprs(
                wr, architecture_repr, input_ids, positions, pre_kv, slots,
                cu_q, cu_k, max_q, max_k, block_table,
            ),
            num_pages,
        ),
{
    match architecture_repr {
        ModelWeightsArchitectureRepr::Qwen3(family) => {
            DENSE::lemma_forward_cache_shape_preserved(
                wr, family, input_ids, positions, pre_kv, slots,
                cu_q, cu_k, max_q, max_k, block_table, num_pages,
            );
        },
        ModelWeightsArchitectureRepr::Llama3(family) => {
            DENSE::lemma_forward_cache_shape_preserved(
                wr, family, input_ids, positions, pre_kv, slots,
                cu_q, cu_k, max_q, max_k, block_table, num_pages,
            );
        },
        ModelWeightsArchitectureRepr::Gemma3Text(family) => {
            FOUR::lemma_forward_cache_shape_preserved(
                wr, family, input_ids, positions, pre_kv, slots,
                cu_q, cu_k, max_q, max_k, block_table, num_pages,
            );
        },
        ModelWeightsArchitectureRepr::Gemma4Text(family) => {
            FOUR::lemma_forward_cache_shape_preserved(
                wr, family.decoder, input_ids, positions, pre_kv, slots,
                cu_q, cu_k, max_q, max_k, block_table, num_pages,
            );
        },
    }
}

// Focused reduction lemmas let existing Qwen proofs move to the dispatched
// API without unfolding either architecture's internal layer chain.
pub proof fn lemma_qwen3_forward_dispatch(
    wr: ModelWeightsRepr,
    qwen: Qwen3ModelWeightsExtensionRepr,
    input_ids: IntTensor1D,
    positions: IntTensor1D,
    pre_kv: Seq<(KVCacheLayerRepr, KVCacheLayerRepr)>,
    slots: Seq<int>,
    cu_q: Seq<int>,
    cu_k: Seq<int>,
    max_q: nat,
    max_k: nat,
    block_table: Seq<Seq<BlockId>>,
)
    ensures
        model_forward_logits_repr(
            wr, ModelWeightsArchitectureRepr::Qwen3(qwen),
            input_ids, positions, pre_kv, slots,
            cu_q, cu_k, max_q, max_k, block_table,
        ) == DENSE::forward_logits_repr(
            wr, qwen,
            input_ids, positions, pre_kv, slots,
            cu_q, cu_k, max_q, max_k, block_table,
        ),
        model_forward_kv_reprs(
            wr, ModelWeightsArchitectureRepr::Qwen3(qwen),
            input_ids, positions, pre_kv, slots,
            cu_q, cu_k, max_q, max_k, block_table,
        ) == DENSE::forward_kv_reprs(
            wr, qwen,
            input_ids, positions, pre_kv, slots,
            cu_q, cu_k, max_q, max_k, block_table,
        ),
{
}

pub proof fn lemma_llama3_forward_dispatch(
    wr: ModelWeightsRepr,
    llama: Llama3ModelWeightsExtensionRepr,
    input_ids: IntTensor1D,
    positions: IntTensor1D,
    pre_kv: Seq<(KVCacheLayerRepr, KVCacheLayerRepr)>,
    slots: Seq<int>,
    cu_q: Seq<int>,
    cu_k: Seq<int>,
    max_q: nat,
    max_k: nat,
    block_table: Seq<Seq<BlockId>>,
)
    ensures
        model_forward_logits_repr(
            wr, ModelWeightsArchitectureRepr::Llama3(llama),
            input_ids, positions, pre_kv, slots,
            cu_q, cu_k, max_q, max_k, block_table,
        ) == DENSE::forward_logits_repr(
            wr, llama,
            input_ids, positions, pre_kv, slots,
            cu_q, cu_k, max_q, max_k, block_table,
        ),
        model_forward_kv_reprs(
            wr, ModelWeightsArchitectureRepr::Llama3(llama),
            input_ids, positions, pre_kv, slots,
            cu_q, cu_k, max_q, max_k, block_table,
        ) == DENSE::forward_kv_reprs(
            wr, llama,
            input_ids, positions, pre_kv, slots,
            cu_q, cu_k, max_q, max_k, block_table,
        ),
{
}

pub proof fn lemma_gemma3_forward_dispatch(
    wr: ModelWeightsRepr,
    gemma: Gemma3ModelWeightsExtensionRepr,
    input_ids: IntTensor1D,
    positions: IntTensor1D,
    pre_kv: Seq<(KVCacheLayerRepr, KVCacheLayerRepr)>,
    slots: Seq<int>,
    cu_q: Seq<int>,
    cu_k: Seq<int>,
    max_q: nat,
    max_k: nat,
    block_table: Seq<Seq<BlockId>>,
)
    ensures
        model_forward_logits_repr(
            wr, ModelWeightsArchitectureRepr::Gemma3Text(gemma),
            input_ids, positions, pre_kv, slots,
            cu_q, cu_k, max_q, max_k, block_table,
        ) == FOUR::forward_logits_repr(
            wr, gemma, input_ids, positions, pre_kv, slots,
            cu_q, cu_k, max_q, max_k, block_table,
        ),
        model_forward_kv_reprs(
            wr, ModelWeightsArchitectureRepr::Gemma3Text(gemma),
            input_ids, positions, pre_kv, slots,
            cu_q, cu_k, max_q, max_k, block_table,
        ) == FOUR::forward_kv_reprs(
            wr, gemma, input_ids, positions, pre_kv, slots,
            cu_q, cu_k, max_q, max_k, block_table,
        ),
{
}

} // verus!
