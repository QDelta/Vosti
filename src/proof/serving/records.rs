//! Checked conversion from executable contracts to request/logit/token records.
//! Whole-execution composition lives in `trace` and `consistency`.

use crate::exec::engine::{self, Engine, StepReprs};
use super::{transitions as E, interpretation as I};
use crate::spec as X;
use crate::proof::serving::refinement as P;
use crate::exec::request_state::{self as RS, RequestState, SamplerState};
use crate::{types::{RequestId, TokenId}, proof::model::types::{SemanticModelRepr}};
use crate::boundary::tensor_runtime as RT;
use vstd::prelude::*;

verus! {

/// Initialization establishes the init relation and
/// records the input arguments; it does not choose a smaller request domain.
pub proof fn lemma_initialization_record(
    e: Engine, model: SemanticModelRepr, config: crate::exec::cache_scheduler::SchedulerConfig,
    num_blocks: u64, requests: Seq<RequestState>,
)
    requires E::serving_init_relation(&e, model, config, num_blocks, requests),
    ensures
        I::initialized(e, I::initial_inputs(e), model, RT::model_runtime_kernel_plan_id(&e.runtime)),
        I::initial_inputs(e).dom() == e.cs.live_requests@.dom(),
        forall|k: int| 0 <= k < requests.len() ==> {
            let request = #[trigger] requests[k];
            &&& I::initial_inputs(e).contains_key(request.request_id)
            &&& I::initial_inputs(e)[request.request_id] == I::request_input(request)
        },
{
    assert(exists|c: crate::exec::cache_scheduler::SchedulerConfig, n: u64, rs: Seq<RequestState>|
        #[trigger] E::serving_init_relation(&e, model, c, n, rs));
    reveal(E::serving_init_relation);
    reveal(engine::engine_init_request_relation);
    assert forall|k: int| 0 <= k < requests.len() implies {
        let request = #[trigger] requests[k];
        &&& I::initial_inputs(e).contains_key(request.request_id)
        &&& I::initial_inputs(e)[request.request_id] == I::request_input(request)
    } by {
        let request = requests[k];
        assert(RS::request_state_view_eq(e.cs.live_requests@[request.request_id], request));
    }
}

/// Every valid API effect has a record transition.
pub proof fn lemma_record_transition(event: E::Event)
    requires E::event_valid(event), event.after.runtime == event.before.runtime,
    ensures I::transition(event.before, I::record(event), event.after),
{
    assert(event == (E::Event { before: event.before, after: event.after, action: event.action }));
}

pub proof fn lemma_admission_record(before: Engine, after: Engine, request: RequestState)
    requires E::admission(before, after, request),
    ensures ({
        let event = E::Event { before, after, action: E::Action::Admit(request) };
        let step = I::record(event);
        &&& I::transition(before, step, after)
        &&& step.admitted.dom() == Set::empty().insert(request.request_id)
        &&& step.admitted[request.request_id] == I::request_input(request)
        &&& step.outputs == Map::<RequestId, X::OutputRecord>::empty()
    }),
{
    reveal(E::admission);
    lemma_record_transition(E::Event { before, after, action: E::Action::Admit(request) });
}

/// Exact emitted domain and token values, independent of the reference model.
pub proof fn lemma_compute_record(
    before: Engine, after: Engine, emitted: Map<RequestId, TokenId>,
    samples: Map<RequestId, (SamplerState, TokenId)>, reprs: StepReprs,
)
    requires
        E::serving_step_relation(before, after, emitted, samples, reprs),
        after.runtime == before.runtime,
    ensures ({
        let event = E::Event { before, after, action: E::Action::Compute { emitted, samples, reprs } };
        let step = I::record(event);
        &&& I::transition(before, step, after)
        &&& step.admitted == Map::<RequestId, X::RequestInput>::empty()
        &&& step.outputs.dom() == emitted.dom()
        &&& forall|rid: RequestId| emitted.contains_key(rid) ==>
            #[trigger] step.outputs[rid].token == emitted[rid]
    }),
{
    lemma_record_transition(E::Event { before, after, action: E::Action::Compute { emitted, samples, reprs } });
}

/// An emitted record contains exactly the row supplied to the sampler. No
/// reference-forward equality is assumed or used to construct the record.
#[verifier::spinoff_prover]
pub proof fn lemma_compute_record_row(
    before: Engine, after: Engine, emitted: Map<RequestId, TokenId>,
    samples: Map<RequestId, (SamplerState, TokenId)>, reprs: StepReprs, rid: RequestId,
)
    requires
        E::serving_step_relation(before, after, emitted, samples, reprs),
        emitted.contains_key(rid),
    ensures
        0 <= I::emitting_row(reprs, rid) < reprs.scheduled.len(),
        reprs.scheduled[I::emitting_row(reprs, rid)] == rid,
        reprs.sample_mask[I::emitting_row(reprs, rid)],
        before.cs.live_requests@.contains_key(rid),
        samples.contains_key(rid),
        RT::sample_from_repr(I::compute_outputs(before, emitted, reprs)[rid].logits,
            before.cs.live_requests@[rid].sampler_state) == samples[rid],
        I::compute_outputs(before, emitted, reprs)[rid].token == samples[rid].1,
        forall|k: int| 0 <= k < reprs.scheduled.len()
            && #[trigger] reprs.scheduled[k] == rid ==> k == I::emitting_row(reprs, rid),
{
    reveal(E::serving_step_relation);
    assert(engine::reprs_emits(reprs, rid));
    let k = I::emitting_row(reprs, rid);
    assert(0 <= k < reprs.scheduled.len());
    assert(engine::engine_samples_match_logits(before, samples, reprs,
        engine::architecture_step_logits_repr(before, reprs)));
    assert(reprs.scheduled.no_duplicates());
}

/// One-step refinement supplies logit equality for the whole-execution theorem.
/// Record construction is independent of this equality.
#[verifier::spinoff_prover]
pub proof fn lemma_compute_record_reference(
    before: Engine, after: Engine, model: SemanticModelRepr,
    emitted: Map<RequestId, TokenId>, samples: Map<RequestId, (SamplerState, TokenId)>,
    reprs: StepReprs, rid: RequestId,
)
    requires
        RT::paged_attention_numeric_domain(),
        P::serving_inv(&before, model),
        E::serving_step_relation(before, after, emitted, samples, reprs),
        emitted.contains_key(rid),
    ensures
        I::compute_outputs(before, emitted, reprs)[rid].logits
            == crate::proof::model::architecture::reference_logits_last_row(model,
                crate::proof::reference::request_machine::token_seq_to_int(RS::history(before.cs.live_requests@[rid]))),
{
    lemma_compute_record_row(before, after, emitted, samples, reprs, rid);
    P::lemma_step_logits_match_reference(before, after, model, emitted, samples, reprs);
    let k = I::emitting_row(reprs, rid);
    assert(crate::proof::engine::refinement::architecture_step_logits_match_reference(before, reprs));
    assert(engine::step_semantic_model(before, reprs) == model);
}

/// Local elimination of structural compute details. Separating it keeps the
/// large forward and scheduler predicates out of sequence induction queries.
#[verifier::spinoff_prover]
pub proof fn lemma_compute_effect(
    event: E::Event, model: SemanticModelRepr, rid: u64,
)
    requires
        RT::paged_attention_numeric_domain(), E::event_valid(event),
        event.action is Compute, P::serving_inv(&event.before, model),
    ensures
        !I::record(event).admitted.contains_key(rid),
        event.after.cs.live_requests@.contains_key(rid) ==> event.before.cs.live_requests@.contains_key(rid),
        I::record(event).outputs.contains_key(rid) ==> {
            let request = event.before.cs.live_requests@[rid];
            let output = I::record(event).outputs[rid];
            &&& event.before.cs.live_requests@.contains_key(rid)
            &&& output.logits == crate::proof::model::architecture::reference_logits_last_row(model,
                crate::proof::reference::request_machine::token_seq_to_int(RS::history(request)))
            &&& output.token == RT::sample_from_repr(output.logits, request.sampler_state).1
        },
        event.after.cs.live_requests@.contains_key(rid) ==> {
            let before = event.before.cs.live_requests@[rid];
            let after = event.after.cs.live_requests@[rid];
            let output = I::record(event).outputs[rid];
            &&& after.prompt_tokens@ == before.prompt_tokens@
            &&& if I::record(event).outputs.contains_key(rid) {
                after.generated_tokens@ == before.generated_tokens@.push(output.token)
                    && after.sampler_state == RT::sample_from_repr(output.logits, before.sampler_state).0
            } else { after.generated_tokens@ == before.generated_tokens@ && after.sampler_state == before.sampler_state }
        },
{
    match event.action {
        E::Action::Compute { emitted, samples, reprs } => {
            if emitted.contains_key(rid) {
                lemma_compute_record_reference(event.before, event.after, model, emitted, samples, reprs, rid);
                lemma_compute_record_row(event.before, event.after, emitted, samples, reprs, rid);
            }
            reveal(E::serving_step_relation);
        },
        _ => {},
    }
}

} // verus!
