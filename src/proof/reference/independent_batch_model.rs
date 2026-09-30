// Abstract batch model: a collection of `RequestMachine`s keyed by
// `RequestId`.  Each machine steps independently; the abstract model's
// "step" maps over the selected ids. Mirrors the earlier Dafny independent
// batch model, recast as a value type.

use crate::model_config::ModelConfig;
use crate::proof::reference::request_machine::*;
use crate::exec::request_state::*;
use crate::{types::*, proof::model::types::*, proof::tensor::types::*};
use vstd::prelude::*;

verus! {


// ---------------------------------------------------------------------------
// IndependentBatchModel — the value-typed abstract batch.  No heap;
// state-passing across step.
// ---------------------------------------------------------------------------

pub struct IndependentBatchModel {
    pub model_config: ModelConfig,
    pub wr: ModelWeightsRepr,
    // Closed family payload paired with `wr`. Keeping the fields separate
    // prevents a concrete family representation from leaking into common
    // cache proofs; `ibm_semantic_model` is the canonical semantic identity.
    pub architecture_repr: ModelWeightsArchitectureRepr,
    pub machines: Map<RequestId, RequestMachine>,
}

pub open spec fn ibm_semantic_model(ibm: IndependentBatchModel) -> SemanticModelRepr {
    SemanticModelRepr {
        weights: ibm.wr,
        architecture: ibm.architecture_repr,
    }
}

pub open spec fn ibm_semantic_models_equal(
    left: IndependentBatchModel,
    right: IndependentBatchModel,
) -> bool {
    ibm_semantic_model(left) == ibm_semantic_model(right)
}

// Kept closed so architecture identity remains available at the semantic
// boundary without making every scheduler/refinement proof unfold an enum
// equality it does not use.
pub closed spec fn ibm_model_architecture_valid(ibm: IndependentBatchModel) -> bool {
    ibm.wr.architecture == ibm.model_config.architecture
    && semantic_model_repr_valid(ibm_semantic_model(ibm))
}

pub proof fn lemma_ibm_model_architecture_valid(ibm: IndependentBatchModel)
    requires
        ibm.wr.architecture == ibm.model_config.architecture,
        model_weights_architecture_repr_valid(ibm.wr, ibm.architecture_repr),
    ensures ibm_model_architecture_valid(ibm),
{
}

pub proof fn lemma_ibm_model_architecture_valid_preserved(
    old_ibm: IndependentBatchModel,
    new_ibm: IndependentBatchModel,
)
    requires
        ibm_model_architecture_valid(old_ibm),
        new_ibm.model_config == old_ibm.model_config,
        new_ibm.wr == old_ibm.wr,
        new_ibm.architecture_repr == old_ibm.architecture_repr,
    ensures
        ibm_model_architecture_valid(new_ibm),
{
}

pub proof fn lemma_ibm_valid_semantic_model(ibm: IndependentBatchModel)
    requires ibm_valid(ibm),
    ensures semantic_model_repr_valid(ibm_semantic_model(ibm)),
{
    reveal(ibm_model_architecture_valid);
}

// IBM consumers obtain the same cache laws for every enabled architecture;
// no family semantics or case split crosses this model boundary.
pub proof fn lemma_ibm_cache_refinement_laws(ibm: IndependentBatchModel)
    requires
        ibm_valid(ibm),
        crate::proof::model::architecture::cache_refinement_supported(
            ibm_semantic_model(ibm),
        ),
    ensures
        crate::proof::model::cache::cache_refinement_laws(
            ibm_semantic_model(ibm),
        ),
{
    lemma_ibm_valid_semantic_model(ibm);
    crate::proof::model::architecture::lemma_cache_refinement_laws(
        ibm_semantic_model(ibm),
    );
}

pub open spec fn ibm_valid(ibm: IndependentBatchModel) -> bool {
    ibm_model_architecture_valid(ibm)
    && ibm.wr.layers.len() == ibm.model_config.num_layers as nat
    && (forall|rid: RequestId|
        #[trigger] ibm.machines.contains_key(rid)
        ==> request_machine_alive(ibm.machines[rid], ibm.model_config)
            && ibm.machines[rid].request_state.request_id == rid)
}

// The complete abstract IBM step: unselected machines unchanged; each selected
// machine either finishes (removed) or transitions via `machine_step_transition_full`
// using its sampled `(next_sampler_state, token)`.  `samples` carries the
// deterministic `sample_from_repr` result per request, so the post request-state
// is fully pinned (incl. `sampler_state`) — letting `refinement_step` keep the
// engine and abstract sides in lockstep (`shared_rid_coherence`) by construction.
pub open spec fn ibm_step(
    old_ibm: IndependentBatchModel,
    new_ibm: IndependentBatchModel,
    selected_ids: Set<RequestId>,
    samples: Map<RequestId, (SamplerState, TokenId)>,
) -> bool {
    new_ibm.model_config == old_ibm.model_config
    && new_ibm.wr == old_ibm.wr
    && new_ibm.architecture_repr == old_ibm.architecture_repr
    && selected_ids.subset_of(old_ibm.machines.dom())
    // No new requests are admitted by a step.
    && new_ibm.machines.dom().subset_of(old_ibm.machines.dom())
    // Unselected machines unchanged.
    && (forall|rid: RequestId|
        #[trigger] old_ibm.machines.contains_key(rid) && !selected_ids.contains(rid)
        ==> new_ibm.machines.contains_key(rid)
            && new_ibm.machines[rid] == old_ibm.machines[rid])
    // Selected machines transition (full) or are removed when finished.
    && (forall|rid: RequestId| #[trigger] selected_ids.contains(rid) ==> {
        let pre = old_ibm.machines[rid].request_state;
        &&& can_step(pre)
        &&& samples.contains_key(rid)
        &&& (if should_finish_after_append(pre, samples[rid].1) {
                !new_ibm.machines.contains_key(rid)
            } else {
                new_ibm.machines.contains_key(rid)
                && machine_step_transition_full(pre, new_ibm.machines[rid].request_state,
                    samples[rid].0, samples[rid].1)
                // A stepped machine remains alive (its KV cache stays well-formed
                // and its kv counters advance) — the abstract step's contract.
                && request_machine_alive(new_ibm.machines[rid], old_ibm.model_config)
            })
    })
}

} // verus!
