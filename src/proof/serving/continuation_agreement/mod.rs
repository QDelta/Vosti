//! Supplementary agreement of request continuations from invariant-satisfying states.
//!
//! This retains the earlier guarantee, including partially generated histories,
//! without a second system-level determinism predicate or satisfaction theorem.

pub mod vocabulary;
pub mod interpretation;
pub mod certified;

use crate::exec::engine::Engine;
use crate::proof::reference::independent_batch_model::{self as IBM, IndependentBatchModel};
use super::transitions as E;
use self::{interpretation as V, vocabulary as S};
use crate::proof::serving::refinement as P;
#[cfg(verus_only)]
use P::{erase_event, lemma_admissible, lemma_certify_event};
use crate::proof::serving::continuation_agreement::certified as C;
use crate::proof::engine::refinement as R;
use crate::{types::{RequestId}, proof::model::types::{SemanticModelRepr}};
use crate::boundary::tensor_runtime as RT;
use vstd::prelude::*;

verus! {

pub proof fn lemma_input(e: Engine, ibm: IndependentBatchModel, rid: RequestId)
    requires R::architecture_persistent_semantic_runtime_inv(&e, &ibm),
    ensures C::state_input(C::CertifiedState { engine: e, ibm }, rid) == V::input(e, rid),
{
    assert(e.cs.live_requests@.contains_key(rid) == ibm.machines.contains_key(rid));
    if ibm.machines.contains_key(rid) {
        assert(crate::exec::request_state::request_state_view_eq(
            e.cs.live_requests@[rid], ibm.machines[rid].request_state));
    }
}

#[verifier::spinoff_prover]
pub proof fn lemma_certify_trace(trace: V::EngineTrace, ibm: IndependentBatchModel,
    model: SemanticModelRepr, plan: RT::KernelPlanId)
    -> (certificate: C::CertifiedTrace)
    requires
        RT::paged_attention_numeric_domain(),
        S::legal_trace(V::engine_trace_interface(model, plan), trace),
        R::architecture_persistent_semantic_runtime_inv(&trace.initial, &ibm),
    ensures
        S::legal_trace(C::certified_trace_interface(), certificate),
        certificate.initial == (C::CertifiedState { engine: trace.initial, ibm }),
        certificate.final_state.engine == trace.final_state,
        R::architecture_persistent_semantic_runtime_inv(
            &certificate.final_state.engine, &certificate.final_state.ibm),
        certificate.events.len() == trace.events.len(),
        forall|i: int| 0 <= i < trace.events.len() ==>
            erase_event(#[trigger] certificate.events[i]) == trace.events[i],
    decreases trace.events.len(),
{
    // Trace assembly only transports checked step facts. Keep their large
    // semantic bodies out of sequence/index reasoning.
    hide(E::event_valid);
    hide(R::dynamic_event_agreement);
    hide(R::architecture_persistent_semantic_runtime_inv);
    let initial = C::CertifiedState { engine: trace.initial, ibm };
    if trace.events.len() == 0 {
        S::Trace { initial, final_state: initial, events: Seq::empty() }
    } else {
        let first = trace.events[0];
        let head = lemma_certify_event(first, ibm);
        let next_ibm = R::dynamic_event_post_ibm(head);
        lemma_admissible(first.after, next_ibm);
        let tail = S::Trace {
            initial: first.after, final_state: trace.final_state,
            events: trace.events.subrange(1, trace.events.len() as int),
        };
        assert forall|i: int| 0 <= i < tail.events.len() implies
            E::event_valid(#[trigger] tail.events[i]) by {
            assert(tail.events[i] == trace.events[i + 1]);
        }
        assert forall|i: int| 0 <= i && i + 1 < tail.events.len() implies
            (#[trigger] tail.events[i]).after == (#[trigger] tail.events[i + 1]).before by {
            assert(tail.events[i] == trace.events[i + 1]);
            assert(tail.events[i + 1] == trace.events[i + 2]);
        }
        // Certification transports the same structural transitions. Each
        // recursive suffix uses its own initial scope; cross-trace model/plan
        // equality is checked at the original request roots, as before.
        assert(S::legal_trace(V::engine_trace_interface(E::semantic_model(first.after),
            RT::model_runtime_kernel_plan_id(&first.after.runtime)), tail));
        let rest = lemma_certify_trace(tail, next_ibm, E::semantic_model(first.after),
            RT::model_runtime_kernel_plan_id(&first.after.runtime));
        let events = seq![head] + rest.events;
        assert forall|i: int| 0 <= i < events.len() implies
            R::dynamic_event_agreement(#[trigger] events[i]) by {
            if i > 0 { assert(events[i] == rest.events[i - 1]); }
        }
        assert forall|i: int| 0 <= i && i + 1 < events.len() implies
            (C::certified_trace_interface().after)(#[trigger] events[i])
                == (C::certified_trace_interface().before)(#[trigger] events[i + 1]) by {
            if i > 0 {
                assert(events[i] == rest.events[i - 1]);
                assert(events[i + 1] == rest.events[i]);
            }
        }
        assert forall|i: int| 0 <= i < trace.events.len() implies
            erase_event(#[trigger] events[i]) == trace.events[i] by {
            if i > 0 {
                assert(events[i] == rest.events[i - 1]);
                assert(tail.events[i - 1] == trace.events[i]);
            }
        }
        S::Trace { initial, final_state: rest.final_state, events }
    }
}

pub proof fn lemma_observations(events: Seq<E::Event>, certificates: Seq<R::DynamicServingEvent>, rid: RequestId)
    requires
        events.len() == certificates.len(),
        forall|i: int| 0 <= i < events.len() ==>
            erase_event(#[trigger] certificates[i]) == events[i],
    ensures
        S::observations(|e, r| V::output(e, r), events, rid)
            == S::observations(C::certified_trace_interface().output, certificates, rid),
    decreases events.len(),
{
    if events.len() > 0 {
        let tail = events.subrange(1, events.len() as int);
        let cert_tail = certificates.subrange(1, certificates.len() as int);
        assert forall|i: int| 0 <= i < tail.len() implies
            erase_event(#[trigger] cert_tail[i]) == tail[i] by {
            assert(cert_tail[i] == certificates[i + 1]);
            assert(tail[i] == events[i + 1]);
        }
        lemma_observations(tail, cert_tail, rid);
        assert(V::output(events[0], rid) == C::event_output(certificates[0], rid));
    }
}

#[verifier::spinoff_prover]
pub proof fn lemma_request_prefix_agreement(
    left: V::EngineTrace, right: V::EngineTrace, left_id: RequestId, right_id: RequestId,
    model: SemanticModelRepr, plan: RT::KernelPlanId,
)
    requires RT::paged_attention_numeric_domain(),
    ensures S::request_consistent(V::engine_trace_interface(model, plan), left, right, left_id, right_id),
{
    if S::legal_trace(V::engine_trace_interface(model, plan), left) && S::legal_trace(V::engine_trace_interface(model, plan), right)
        && V::input(left.initial, left_id).is_some()
        && V::input(left.initial, left_id) == V::input(right.initial, right_id) {
        reveal(P::serving_inv);
        let a = choose|ibm: IndependentBatchModel|
            R::architecture_persistent_semantic_runtime_inv(&left.initial, &ibm)
            && IBM::ibm_semantic_model(ibm) == E::semantic_model(left.initial);
        let b = choose|ibm: IndependentBatchModel|
            R::architecture_persistent_semantic_runtime_inv(&right.initial, &ibm)
            && IBM::ibm_semantic_model(ibm) == E::semantic_model(right.initial);
        let lc = lemma_certify_trace(left, a, model, plan);
        let rc = lemma_certify_trace(right, b, model, plan);
        lemma_input(left.initial, a, left_id);
        lemma_input(right.initial, b, right_id);
        lemma_observations(left.events, lc.events, left_id);
        lemma_observations(right.events, rc.events, right_id);
        C::lemma_request_consistent(lc, rc, left_id, right_id);
    }
}

/// Existing checked initialization establishes this interpretation's domain.
pub proof fn lemma_initialization_admissible(
    e: Engine, model: SemanticModelRepr, config: crate::exec::cache_scheduler::SchedulerConfig,
    num_blocks: u64, requests: Seq<crate::exec::request_state::RequestState>,
)
    requires E::serving_init_relation(&e, model, config, num_blocks, requests),
    ensures (V::engine_trace_interface(E::semantic_model(e), RT::model_runtime_kernel_plan_id(&e.runtime)).admissible)(e),
{
    P::lemma_init_establishes_serving_inv(e, model, config, num_blocks, requests);
    reveal(E::serving_init_relation);
    assert(E::semantic_model(e) == model);
}

} // verus!
