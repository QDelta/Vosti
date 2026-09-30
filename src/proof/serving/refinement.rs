//! Shared checked initialization, compute and admission refinement.
//!
//! These invariants and certificate witnesses support both trace proofs. None
//! is a premise of the explicit execution specification's transition relation.

use crate::exec::cache_scheduler;
use crate::exec::engine::{self, Engine, StepReprs};
use crate::proof::reference::independent_batch_model::{self as IBM, IndependentBatchModel};
use crate::proof::serving::transitions as E;
#[cfg(verus_only)]
use E::{serving_init_relation, serving_step_relation};
use crate::proof::engine::refinement as R;
use crate::exec::request_state::{RequestState, SamplerState};
use crate::{types::{RequestId, TokenId}, proof::model::types::{SemanticModelRepr}};
use crate::boundary::tensor_runtime as RT;
use vstd::prelude::*;

verus! {

/// Stable engine state for a fixed semantic model. The existential IBM is a
/// proof witness, not part of the observable serving specification.
/// These predicates stay solver-opaque by default, but checked interpretation
/// proofs may reveal them explicitly to construct hidden witnesses.
#[verifier::opaque]
pub open spec fn serving_inv(
    engine: &Engine,
    model: SemanticModelRepr,
) -> bool {
    exists|ibm: IndependentBatchModel|
        R::architecture_persistent_semantic_runtime_inv(engine, &ibm)
        && crate::proof::reference::independent_batch_model::ibm_semantic_model(ibm) == model
}

/// One stable semantic state under a fixed model and qualified kernel plan.
pub open spec fn admissible(e: Engine, model: SemanticModelRepr, plan: RT::KernelPlanId) -> bool {
    serving_inv(&e, model) && E::semantic_model(e) == model
        && RT::model_runtime_kernel_plan_id(&e.runtime) == plan
}

/// Every emitting engine row has the singleton full-history meaning for the
/// fixed served model `wr`. Non-final chunked-prefill rows are KV-only.
pub open spec fn step_logits_match_reference(
    model: SemanticModelRepr,
    old_e: Engine,
    reprs: StepReprs,
) -> bool {
    crate::exec::engine::step_semantic_model(old_e, reprs) == model
    && R::architecture_step_logits_match_reference(old_e, reprs)
}

// @kernel-bridge-begin proof::serving::refinement::lemma_init_establishes_serving_inv
#[verifier::spinoff_prover]
pub proof fn lemma_init_establishes_serving_inv(
    engine: Engine,
    model: SemanticModelRepr,
    config: cache_scheduler::SchedulerConfig,
    num_blocks: u64,
    requests: Seq<RequestState>,
)
    requires
        serving_init_relation(&engine, model, config, num_blocks, requests),
    ensures
        serving_inv(&engine, model),
{
    reveal(serving_init_relation);
    reveal(crate::exec::engine::engine_init_request_relation);
    assert forall|rid: RequestId|
        #[trigger] engine.cs.live_requests@.contains_key(rid) implies {
            &&& crate::exec::request_state::valid_request_state(
                engine.cs.live_requests@[rid],
            )
            &&& engine.cs.live_requests@[rid].request_id == rid
        }
    by {
        let k = choose|k: int| 0 <= k < requests.len()
            && requests[k].request_id == rid;
        assert(crate::exec::request_state::request_state_view_eq(
            engine.cs.live_requests@[rid], requests[k],
        ));
        crate::exec::request_state::lemma_request_lifecycle_view_eq_fields(
            engine.cs.live_requests@[rid], requests[k],
        );
        assert(crate::exec::request_state::can_step(requests[k]));
    }
    R::architecture_persistent_semantic_refinement_initialized(&engine);
    let ibm = R::initialized_ibm(&engine);
    assert(R::architecture_persistent_semantic_runtime_inv(&engine, &ibm));
    assert(ibm.wr == RT::model_weights_repr_of(&engine.weights_perms@));
    assert(crate::proof::reference::independent_batch_model::ibm_semantic_model(ibm) == model);
    reveal(serving_inv);
    assert(exists|witness: IndependentBatchModel|
        R::architecture_persistent_semantic_runtime_inv(&engine, &witness)
        && crate::proof::reference::independent_batch_model::ibm_semantic_model(witness) == model);
}
// @kernel-bridge-end proof::serving::refinement::lemma_init_establishes_serving_inv

// @kernel-bridge-begin proof::serving::refinement::lemma_step_logits_match_reference
#[verifier::spinoff_prover]
pub proof fn lemma_step_logits_match_reference(
    old_e: Engine,
    new_e: Engine,
    model: SemanticModelRepr,
    emitted: Map<RequestId, TokenId>,
    samples: Map<RequestId, (SamplerState, TokenId)>,
    reprs: StepReprs,
)
    requires
        RT::paged_attention_numeric_domain(),
        serving_inv(&old_e, model),
        serving_step_relation(old_e, new_e, emitted, samples, reprs),
    ensures
        serving_inv(&new_e, model),
        step_logits_match_reference(model, old_e, reprs),
{
    reveal(serving_inv);
    reveal(serving_step_relation);
    let old_ibm = choose|ibm: IndependentBatchModel|
        R::architecture_persistent_semantic_runtime_inv(&old_e, &ibm)
        && crate::proof::reference::independent_batch_model::ibm_semantic_model(ibm) == model;
    let new_ibm = crate::proof::engine::abstract_step::architecture_stepped_ibm(
        old_e, new_e, old_ibm, emitted, reprs,
    );
    assert(R::architecture_persistent_semantic_runtime_inv(
        &old_e, &old_ibm,
    ));
    reveal(R::architecture_persistent_semantic_runtime_inv);
    assert(reprs.wr.layers.len() == old_e.model_config.num_layers as nat);
    assert(reprs.wr.layers.len() > 0);
    R::architecture_persistent_semantic_refinement_observable_step(
        old_e, new_e, old_ibm, emitted, samples, reprs,
    );
    R::lemma_architecture_phase_aligned_preserved(
        old_e, new_e, old_ibm, emitted, samples, reprs,
    );
    R::lemma_architecture_mach_cache_shape_preserved(
        old_e, new_e, old_ibm, emitted, samples, reprs,
    );
    engine::lemma_architecture_live_request_step_ready_preserved(
        old_e, new_e, emitted, samples, reprs,
    );
    assert(R::architecture_persistent_semantic_runtime_inv(
        &new_e, &new_ibm,
    ));
    assert(crate::proof::reference::independent_batch_model::ibm_semantic_model(new_ibm) == model);
    assert(exists|witness: IndependentBatchModel|
        R::architecture_persistent_semantic_runtime_inv(&new_e, &witness)
        && crate::proof::reference::independent_batch_model::ibm_semantic_model(witness) == model);
    assert(R::architecture_step_logits_match_reference(old_e, reprs));
    assert(crate::exec::engine::architecture_engine_step_relation(
        old_e, new_e, emitted, samples, reprs,
    ));
    assert(crate::exec::engine::engine_step_semantic_identity(
        old_e, new_e,
    ));
    reveal(crate::exec::engine::engine_step_semantic_identity);
    assert(crate::exec::engine::step_semantic_model(old_e, reprs) == model);
    assert(step_logits_match_reference(model, old_e, reprs));
}
// @kernel-bridge-end proof::serving::refinement::lemma_step_logits_match_reference

pub open spec fn erase_event(event: R::DynamicServingEvent) -> E::Event {
    match event {
        R::DynamicServingEvent::Admission(step) => E::Event {
            before: step.pre_engine, after: step.post_engine, action: E::Action::Admit(step.request),
        },
        R::DynamicServingEvent::Compute(step) => E::Event {
            before: step.pre_engine, after: step.post_engine,
            action: E::Action::Compute { emitted: step.emitted, samples: step.samples, reprs: step.reprs },
        },
    }
}

pub proof fn lemma_admissible(e: Engine, ibm: IndependentBatchModel)
    requires R::architecture_persistent_semantic_runtime_inv(&e, &ibm),
    ensures admissible(e, E::semantic_model(e), RT::model_runtime_kernel_plan_id(&e.runtime)),
{
    assert(IBM::ibm_semantic_model(ibm) == E::semantic_model(e));
    reveal(serving_inv);
    assert(exists|w: IndependentBatchModel|
        R::architecture_persistent_semantic_runtime_inv(&e, &w)
        && IBM::ibm_semantic_model(w) == E::semantic_model(e));
}

/// This is the semantic work: correctness facts are conclusions of existing
/// checked refinement lemmas, not premises of transitions::event_valid.
#[verifier::spinoff_prover]
pub proof fn lemma_certify_event(event: E::Event, ibm: IndependentBatchModel)
    -> (certificate: R::DynamicServingEvent)
    requires
        RT::paged_attention_numeric_domain(),
        E::event_valid(event),
        R::architecture_persistent_semantic_runtime_inv(&event.before, &ibm),
    ensures
        erase_event(certificate) == event,
        R::dynamic_event_pre_ibm(certificate) == ibm,
        R::dynamic_event_agreement(certificate),
        R::architecture_persistent_semantic_runtime_inv(
            &event.after, &R::dynamic_event_post_ibm(certificate)),
{
    let old_e = event.before;
    let new_e = event.after;
    match event.action {
        E::Action::Admit(request) => {
            reveal(E::admission);
            R::architecture_persistent_semantic_refinement_admission(old_e, new_e, ibm, request);
            let next = R::admitted_ibm(ibm, &new_e, request.request_id);
            R::DynamicServingEvent::Admission(R::AdmissionTraceStep {
                pre_engine: old_e, post_engine: new_e, pre_ibm: ibm, post_ibm: next, request,
            })
        },
        E::Action::Compute { emitted, samples, reprs } => {
            reveal(serving_step_relation);
            let next = crate::proof::engine::abstract_step::architecture_stepped_ibm(
                old_e, new_e, ibm, emitted, reprs);
            assert(reprs.wr.layers.len() == old_e.model_config.num_layers as nat);
            assert(reprs.wr.layers.len() > 0);
            R::architecture_persistent_semantic_refinement_observable_step(
                old_e, new_e, ibm, emitted, samples, reprs);
            R::lemma_architecture_phase_aligned_preserved(old_e, new_e, ibm, emitted, samples, reprs);
            R::lemma_architecture_mach_cache_shape_preserved(old_e, new_e, ibm, emitted, samples, reprs);
            engine::lemma_architecture_live_request_step_ready_preserved(old_e, new_e, emitted, samples, reprs);
            assert(R::architecture_persistent_semantic_runtime_inv(&new_e, &next));
            let step = R::ObservableTraceStep {
                pre_engine: old_e, post_engine: new_e, pre_ibm: ibm, post_ibm: next,
                emitted, samples, reprs,
            };
            R::lemma_observable_trace_step_agreement_intro(step);
            R::DynamicServingEvent::Compute(step)
        },
    }
}

/// Successful compute/admission keeps the domain closed. In particular a
/// newly admitted request can start a request trace at the resulting state.
pub proof fn lemma_event_preserves_admissibility(event: E::Event)
    requires
        RT::paged_attention_numeric_domain(),
        serving_inv(&event.before, E::semantic_model(event.before)),
        E::event_valid(event),
    ensures admissible(event.after, E::semantic_model(event.after),
        RT::model_runtime_kernel_plan_id(&event.after.runtime)),
{
    reveal(serving_inv);
    let ibm = choose|ibm: IndependentBatchModel|
        R::architecture_persistent_semantic_runtime_inv(&event.before, &ibm)
        && IBM::ibm_semantic_model(ibm) == E::semantic_model(event.before);
    let certificate = lemma_certify_event(event, ibm);
    lemma_admissible(event.after, R::dynamic_event_post_ibm(certificate));
}

} // verus!
