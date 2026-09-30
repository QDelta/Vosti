//! Certified witnesses for the internal continuation prefix-agreement proof.
//!
//! These witnesses are absent from the public spec and concrete API effects.

use crate::exec::engine::Engine;
use crate::proof::reference::independent_batch_model::{self as IBM, IndependentBatchModel};
use crate::proof::serving::continuation_agreement::vocabulary as S;
use S::{Observation, SamplingInput};
use crate::proof::engine::refinement as R;
use crate::proof::engine::refinement::{DynamicRequestTrace, ObservableServingTrace};
#[cfg(verus_only)]
use crate::proof::engine::refinement::{arrival_request_view_eq, dynamic_request_trace,
    dynamic_trace_samples_for, observable_serving_trace, trace_samples_for};
use crate::exec::request_state::SamplerState;
use crate::{types::{RequestId, TokenId}, proof::model::types::{SemanticModelRepr}};
use crate::boundary::tensor_runtime as RT;
use vstd::prelude::*;

verus! {

pub ghost struct CertifiedState {
    pub engine: Engine,
    pub ibm: IndependentBatchModel,
}

pub type CertifiedTrace = S::Trace<CertifiedState, R::DynamicServingEvent>;
pub type CertifiedTraceInterface = S::TraceInterface<CertifiedState, R::DynamicServingEvent>;

pub open spec fn state_input(state: CertifiedState, rid: RequestId) -> Option<SamplingInput> {
    if state.ibm.machines.contains_key(rid) {
        let request = state.ibm.machines[rid].request_state;
        Some(SamplingInput {
            prompt: request.prompt_tokens@,
            generated: request.generated_tokens@,
            sampler: request.sampler_state,
        })
    } else {
        None
    }
}

/// Exactly the sample/token pairs emitted by compute; admissions emit nothing.
pub open spec fn event_output(event: R::DynamicServingEvent, rid: RequestId)
    -> Option<Observation>
{
    match event {
        R::DynamicServingEvent::Admission(_) => None,
        R::DynamicServingEvent::Compute(step) => {
            if step.emitted.contains_key(rid) { Some(step.samples[rid]) } else { None }
        },
    }
}

// Internal unscoped witness carrier, not a claimed deterministic deployment.
// Its pairwise consistency lemma explicitly requires equal model/plan roots.
pub open spec fn certified_trace_interface() -> CertifiedTraceInterface {
    S::TraceInterface {
        admissible: |s: CertifiedState|
            R::architecture_persistent_semantic_runtime_inv(&s.engine, &s.ibm),
        before: |e: R::DynamicServingEvent| CertifiedState {
            engine: R::dynamic_event_pre_engine(e), ibm: R::dynamic_event_pre_ibm(e),
        },
        after: |e: R::DynamicServingEvent| CertifiedState {
            engine: R::dynamic_event_post_engine(e), ibm: R::dynamic_event_post_ibm(e),
        },
        step: |e: R::DynamicServingEvent| R::dynamic_event_agreement(e),
        input: |s: CertifiedState, rid: RequestId| state_input(s, rid),
        output: |e: R::DynamicServingEvent, rid: RequestId| event_output(e, rid),
    }
}

/// Representation-only bridge: no new semantic or runtime assumption.
pub proof fn lemma_observations(events: Seq<R::DynamicServingEvent>, rid: RequestId)
    ensures
        S::observations(certified_trace_interface().output, events, rid)
            == R::dynamic_trace_samples_for(events, rid),
    decreases events.len(),
{
    if events.len() > 0 {
        lemma_observations(events.subrange(1, events.len() as int), rid);
    }
}

pub proof fn lemma_legal_chain(trace: CertifiedTrace)
    requires S::legal_trace(certified_trace_interface(), trace),
    ensures R::dynamic_trace_chain(trace.events),
{
    assert forall|i: int| 0 <= i < trace.events.len() implies
        #[trigger] R::dynamic_event_agreement(trace.events[i]) by {}
    assert forall|i: int| 0 <= i && i + 1 < trace.events.len() implies
        #[trigger] R::dynamic_event_post_engine(trace.events[i])
            == R::dynamic_event_pre_engine(trace.events[i + 1])
        && R::dynamic_event_post_ibm(trace.events[i])
            == R::dynamic_event_pre_ibm(trace.events[i + 1]) by {}
}

/// The canonical reference computation is proof machinery, not public-spec
/// vocabulary. Admission of the same ID is excluded by the checked ledger law.
#[verifier::spinoff_prover]
pub proof fn lemma_reference_run(trace: CertifiedTrace, rid: RequestId)
    requires
        S::legal_trace(certified_trace_interface(), trace),
        state_input(trace.initial, rid).is_some(),
    ensures ({
        let input = state_input(trace.initial, rid).unwrap();
        R::reference_sample_run(IBM::ibm_semantic_model(trace.initial.ibm), input.prompt, input.generated, input.sampler,
            S::observations(certified_trace_interface().output, trace.events, rid))
    }),
{
    lemma_legal_chain(trace);
    lemma_observations(trace.events, rid);
    if trace.events.len() > 0 {
        assert(R::dynamic_event_pre_engine(trace.events[0]) == trace.initial.engine);
        assert(R::dynamic_event_pre_ibm(trace.events[0]) == trace.initial.ibm);
        assert(trace.initial.engine.cs.live_requests@.contains_key(rid));
        assert(trace.initial.engine.cs.accepted_requests@.contains_key(rid));
        R::lemma_dynamic_chain_cannot_readmit(trace.events, rid);
        R::lemma_dynamic_events_request_sample_run(trace.events, rid);
    }
}

#[verifier::spinoff_prover]
pub proof fn lemma_request_consistent(
    left: CertifiedTrace, right: CertifiedTrace, left_id: RequestId, right_id: RequestId,
)
    requires
        IBM::ibm_semantic_model(left.initial.ibm) == IBM::ibm_semantic_model(right.initial.ibm),
        RT::model_runtime_kernel_plan_id(&left.initial.engine.runtime)
            == RT::model_runtime_kernel_plan_id(&right.initial.engine.runtime),
    ensures S::request_consistent(certified_trace_interface(), left, right, left_id, right_id),
{
    let system = certified_trace_interface();
    if S::legal_trace(system, left) && S::legal_trace(system, right)
        && state_input(left.initial, left_id).is_some()
        && state_input(left.initial, left_id) == state_input(right.initial, right_id) {
        lemma_reference_run(left, left_id);
        lemma_reference_run(right, right_id);
        let input = state_input(left.initial, left_id).unwrap();
        let a = S::observations(system.output, left.events, left_id);
        let b = S::observations(system.output, right.events, right_id);
        let upto = if a.len() <= b.len() { a.len() } else { b.len() };
        R::lemma_reference_sample_runs_agree(
            IBM::ibm_semantic_model(left.initial.ibm), input.prompt, input.generated, input.sampler, a, b, upto,
        );
        assert(S::common_prefix_equal(a, b));
    }
}

// These internal corollaries use certified traces. The public satisfaction
// theorem instantiates the engine-independent property.

// @kernel-bridge-begin proof::serving::continuation_agreement::certified::lemma_observable_serving_traces_request_prefix_equal
pub proof fn lemma_observable_serving_traces_request_prefix_equal(
    left: ObservableServingTrace,
    right: ObservableServingTrace,
    rid: RequestId,
    upto: nat,
)
    requires
        observable_serving_trace(left, left.steps.len()),
        observable_serving_trace(right, right.steps.len()),
        left.initial_ibm.machines.contains_key(rid),
        right.initial_ibm.machines.contains_key(rid),
        crate::proof::reference::independent_batch_model::ibm_semantic_models_equal(
            left.initial_ibm, right.initial_ibm,
        ),
        crate::boundary::tensor_runtime::model_runtime_kernel_plan_id(
            &left.initial_engine.runtime,
        ) == crate::boundary::tensor_runtime::model_runtime_kernel_plan_id(
            &right.initial_engine.runtime,
        ),
        crate::exec::request_state::request_sampling_view_eq(
            left.initial_ibm.machines[rid].request_state,
            right.initial_ibm.machines[rid].request_state,
        ),
        upto <= trace_samples_for(left.steps, rid).len(),
        upto <= trace_samples_for(right.steps, rid).len(),
    ensures
        forall|i: int|
            #![trigger trace_samples_for(left.steps, rid)[i]]
            0 <= i < upto ==> {
                let left_samples = trace_samples_for(left.steps, rid);
                let right_samples = trace_samples_for(right.steps, rid);
                &&& left_samples[i] == right_samples[i]
                &&& left_samples[i].1 == right_samples[i].1
            },
{
    let left_initial = left.initial_ibm.machines[rid].request_state;
    let right_initial = right.initial_ibm.machines[rid].request_state;
    let left_samples = trace_samples_for(left.steps, rid);
    let right_samples = trace_samples_for(right.steps, rid);
    R::lemma_observable_serving_trace_request_sample_run(left, rid);
    R::lemma_observable_serving_trace_request_sample_run(right, rid);
    assert(left_initial.prompt_tokens@ == right_initial.prompt_tokens@);
    assert(left_initial.generated_tokens@ == right_initial.generated_tokens@);
    assert(left_initial.sampler_state == right_initial.sampler_state);
    assert(crate::proof::reference::independent_batch_model::ibm_semantic_model(left.initial_ibm)
        == crate::proof::reference::independent_batch_model::ibm_semantic_model(right.initial_ibm));
    assert(R::reference_sample_run(
        crate::proof::reference::independent_batch_model::ibm_semantic_model(left.initial_ibm),
        left_initial.prompt_tokens@,
        left_initial.generated_tokens@,
        left_initial.sampler_state,
        right_samples,
    ));
    R::lemma_reference_sample_runs_agree(
        crate::proof::reference::independent_batch_model::ibm_semantic_model(left.initial_ibm),
        left_initial.prompt_tokens@,
        left_initial.generated_tokens@,
        left_initial.sampler_state,
        left_samples,
        right_samples,
        upto,
    );
}
// @kernel-bridge-end proof::serving::continuation_agreement::certified::lemma_observable_serving_traces_request_prefix_equal

// @kernel-bridge-begin proof::serving::continuation_agreement::certified::lemma_dynamic_request_traces_prefix_equal
pub proof fn lemma_dynamic_request_traces_prefix_equal(
    left: DynamicRequestTrace,
    right: DynamicRequestTrace,
    upto: nat,
)
    requires
        dynamic_request_trace(left),
        dynamic_request_trace(right),
        crate::proof::reference::independent_batch_model::ibm_semantic_models_equal(
            left.arrival.pre_ibm, right.arrival.pre_ibm,
        ),
        crate::boundary::tensor_runtime::model_runtime_kernel_plan_id(
            &left.arrival.pre_engine.runtime,
        ) == crate::boundary::tensor_runtime::model_runtime_kernel_plan_id(
            &right.arrival.pre_engine.runtime,
        ),
        arrival_request_view_eq(left.arrival.request, right.arrival.request),
        upto <= dynamic_trace_samples_for(
            left.events, left.arrival.request.request_id,
        ).len(),
        upto <= dynamic_trace_samples_for(
            right.events, right.arrival.request.request_id,
        ).len(),
    ensures
        forall|i: int|
            #![trigger dynamic_trace_samples_for(
                left.events, left.arrival.request.request_id,
            )[i]]
            0 <= i < upto ==> {
                let left_samples = dynamic_trace_samples_for(
                    left.events, left.arrival.request.request_id,
                );
                let right_samples = dynamic_trace_samples_for(
                    right.events, right.arrival.request.request_id,
                );
                &&& left_samples[i] == right_samples[i]
                &&& left_samples[i].1 == right_samples[i].1
            },
{
    let left_request = left.arrival.request;
    let right_request = right.arrival.request;
    let left_samples = dynamic_trace_samples_for(
        left.events, left_request.request_id,
    );
    let right_samples = dynamic_trace_samples_for(
        right.events, right_request.request_id,
    );
    R::lemma_dynamic_request_trace_sample_run(left);
    R::lemma_dynamic_request_trace_sample_run(right);
    assert(left_request.prompt_tokens@ == right_request.prompt_tokens@);
    assert(left_request.generated_tokens@ == right_request.generated_tokens@);
    assert(left_request.sampler_state == right_request.sampler_state);
    assert(crate::proof::reference::independent_batch_model::ibm_semantic_model(
        left.arrival.pre_ibm,
    ) == crate::proof::reference::independent_batch_model::ibm_semantic_model(
        right.arrival.pre_ibm,
    ));
    assert(R::reference_sample_run(
        crate::proof::reference::independent_batch_model::ibm_semantic_model(
            left.arrival.pre_ibm,
        ),
        left_request.prompt_tokens@,
        left_request.generated_tokens@,
        left_request.sampler_state,
        right_samples,
    ));
    R::lemma_reference_sample_runs_agree(
        crate::proof::reference::independent_batch_model::ibm_semantic_model(
            left.arrival.pre_ibm,
        ),
        left_request.prompt_tokens@,
        left_request.generated_tokens@,
        left_request.sampler_state,
        left_samples,
        right_samples,
        upto,
    );
}
// @kernel-bridge-end proof::serving::continuation_agreement::certified::lemma_dynamic_request_traces_prefix_equal

} // verus!
