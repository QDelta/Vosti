//! Internal reference-run argument for the explicit logits/token trace property.
//! Reference runs are proof witnesses only, never execution-legality premises.

use crate::exec::engine::Engine;
use super::{transitions as E, interpretation as I};
use crate::spec as X;
use crate::proof::serving::refinement as P;
use super::{records as B, trace as T};
use crate::exec::request_state as RS;
use crate::{proof::model::types::{SemanticModelRepr}};
use crate::boundary::tensor_runtime as RT;
use vstd::prelude::*;

verus! {

pub ghost struct ReferenceRun {
    pub outputs: Seq<X::OutputRecord>,
    pub generated: Seq<u64>,
    pub sampler: RS::SamplerState,
}

#[verifier::opaque]
pub open spec fn reference_run(model: SemanticModelRepr, input: X::RequestInput, n: nat) -> ReferenceRun
    decreases n,
{
    if n == 0 {
        ReferenceRun { outputs: Seq::empty(), generated: Seq::empty(), sampler: input.initial_sampler_state }
    } else {
        let previous = reference_run(model, input, (n - 1) as nat);
        let logits = crate::proof::model::architecture::reference_logits_last_row(model,
            crate::proof::reference::request_machine::token_seq_to_int(input.prompt + previous.generated));
        let sample = RT::sample_from_repr(logits, previous.sampler);
        ReferenceRun {
            outputs: previous.outputs.push(X::OutputRecord { logits, token: sample.1 }),
            generated: previous.generated.push(sample.1), sampler: sample.0,
        }
    }
}

pub proof fn lemma_reference_run_length(model: SemanticModelRepr, input: X::RequestInput, n: nat)
    ensures reference_run(model, input, n).outputs.len() == n,
    decreases n,
{
    reveal(reference_run);
    if n > 0 { lemma_reference_run_length(model, input, (n - 1) as nat); }
}

pub proof fn lemma_reference_run_zero(model: SemanticModelRepr, input: X::RequestInput)
    ensures
        reference_run(model, input, 0).outputs == Seq::<X::OutputRecord>::empty(),
        reference_run(model, input, 0).generated == Seq::<u64>::empty(),
        reference_run(model, input, 0).sampler == input.initial_sampler_state,
{ reveal(reference_run); }

pub proof fn lemma_reference_run_extend(
    model: SemanticModelRepr, input: X::RequestInput, n: nat, output: X::OutputRecord,
)
    requires
        output.logits == crate::proof::model::architecture::reference_logits_last_row(model,
            crate::proof::reference::request_machine::token_seq_to_int(input.prompt + reference_run(model, input, n).generated)),
        output.token == RT::sample_from_repr(output.logits, reference_run(model, input, n).sampler).1,
    ensures
        reference_run(model, input, n + 1).outputs == reference_run(model, input, n).outputs.push(output),
        reference_run(model, input, n + 1).generated == reference_run(model, input, n).generated.push(output.token),
        reference_run(model, input, n + 1).sampler == RT::sample_from_repr(output.logits, reference_run(model, input, n).sampler).0,
{
    hide(crate::proof::model::architecture::reference_logits_last_row);
    reveal(reference_run);
}

pub proof fn lemma_reference_run_prefix(model: SemanticModelRepr, input: X::RequestInput, a: nat, b: nat)
    requires a <= b,
    ensures X::is_prefix(reference_run(model, input, a).outputs, reference_run(model, input, b).outputs),
    decreases b,
{
    lemma_reference_run_length(model, input, a);
    lemma_reference_run_length(model, input, b);
    if a < b {
        lemma_reference_run_prefix(model, input, a, (b - 1) as nat);
        reveal(reference_run);
        assert forall|i: int| #![trigger reference_run(model, input, a).outputs[i], reference_run(model, input, b).outputs[i]]
            0 <= i < a implies reference_run(model, input, a).outputs[i] == reference_run(model, input, b).outputs[i] by {
            assert(reference_run(model, input, b).outputs[i] == reference_run(model, input, (b - 1) as nat).outputs[i]);
        }
    }
}

/// Sequence-only lemma: the public view preserves a final emission exactly.
pub proof fn lemma_view_push(steps: Seq<X::Step>, step: X::Step, rid: u64)
    ensures X::view(steps.push(step), rid) ==
        if step.outputs.contains_key(rid) { X::view(steps, rid).push(step.outputs[rid]) }
        else { X::view(steps, rid) },
    decreases steps.len(),
{
    reveal_with_fuel(X::view, 2);
    if steps.len() > 0 {
        let tail = steps.subrange(1, steps.len() as int);
        lemma_view_push(tail, step, rid);
        assert(steps.push(step).subrange(1, steps.len() as int + 1) =~= tail.push(step));
        if steps[0].outputs.contains_key(rid) && step.outputs.contains_key(rid) {
            assert((seq![steps[0].outputs[rid]] + X::view(tail, rid)).push(step.outputs[rid]) =~=
                seq![steps[0].outputs[rid]] + X::view(tail, rid).push(step.outputs[rid]));
        }
    } else {
        assert(steps =~= Seq::<X::Step>::empty());
        assert(steps.push(step) =~= seq![step]);
        assert(X::view(steps.push(step), rid) =~=
            if step.outputs.contains_key(rid) { seq![step.outputs[rid]] } else { Seq::empty() });
    }
}

pub open spec fn request_prefix(
    initial: Engine, events: Seq<E::Event>, model: SemanticModelRepr, rid: u64, n: int,
) -> bool {
    let prefix = T::records(events).take(n);
    let inputs = X::requests_after(I::initial_inputs(initial), prefix);
    let outputs = X::view(prefix, rid);
    let state = T::state_at(initial, events, n);
    if inputs.contains_key(rid) {
        let expected = reference_run(model, inputs[rid], outputs.len());
        &&& outputs == expected.outputs
        &&& (state.cs.live_requests@.contains_key(rid) ==> {
            let request = state.cs.live_requests@[rid];
            &&& request.prompt_tokens@ == inputs[rid].prompt
            &&& request.generated_tokens@ == expected.generated
            &&& request.sampler_state == expected.sampler
        })
    } else {
        outputs.len() == 0 && !state.cs.live_requests@.contains_key(rid)
    }
}

#[verifier::spinoff_prover]
pub proof fn lemma_request_prefix(
    initial: Engine, events: Seq<E::Event>, model: SemanticModelRepr,
    plan: RT::KernelPlanId, rid: u64, n: int,
)
    requires
        RT::paged_attention_numeric_domain(),
        I::initialized(initial, I::initial_inputs(initial), model, plan),
        T::chained(initial, events), 0 <= n <= events.len(),
    ensures request_prefix(initial, events, model, rid, n),
    decreases n,
{
    hide(E::event_valid);
    hide(I::record);
    hide(crate::proof::model::architecture::reference_logits_last_row);
    T::lemma_initial_execution_facts(initial, model, plan);
    if n == 0 {
        lemma_reference_run_zero(model, I::initial_inputs(initial)[rid]);
        if initial.cs.live_requests@.contains_key(rid) {
            assert(initial.cs.live_requests@[rid].generated_tokens@ =~= Seq::<u64>::empty());
        }
        assert(request_prefix(initial, events, model, rid, n));
    } else {
        lemma_request_prefix(initial, events, model, plan, rid, n - 1);
        T::lemma_prefix_execution_facts(initial, events, n - 1);
        T::lemma_prefix_invariant(initial, events, model, plan, n - 1);
        let event = events[n - 1];
        T::lemma_event_execution_facts(event);
        let prefix = T::records(events).take(n - 1);
        let step = I::record(event);
        let inputs = X::requests_after(I::initial_inputs(initial), prefix);
        let outputs = X::view(prefix, rid);
        assert(T::records(events).take(n) =~= prefix.push(step));
        assert(prefix.push(step).drop_last() =~= prefix);
        assert(X::requests_after(I::initial_inputs(initial), T::records(events).take(n))
            == inputs.union_prefer_right(step.admitted));
        lemma_view_push(prefix, step, rid);
        match event.action {
            E::Action::Admit(request) => {
                reveal(E::event_valid);
                reveal(E::admission);
                reveal(I::record);
                if request.request_id == rid {
                    lemma_reference_run_zero(model, I::request_input(request));
                    assert(!inputs.contains_key(rid));
                    assert(request.generated_tokens@ =~= Seq::<u64>::empty());
                    assert(outputs =~= Seq::<X::OutputRecord>::empty());
                }
                assert(request_prefix(initial, events, model, rid, n));
            },
            E::Action::Compute { .. } => {
                B::lemma_compute_effect(event, model, rid);
                if step.outputs.contains_key(rid) {
                    assert(inputs.contains_key(rid));
                    let previous = reference_run(model, inputs[rid], outputs.len());
                    assert(outputs == previous.outputs);
                    lemma_reference_run_extend(model, inputs[rid], outputs.len(), step.outputs[rid]);
                    assert(reference_run(model, inputs[rid], outputs.len() + 1).outputs == outputs.push(step.outputs[rid]));
                }
                assert(request_prefix(initial, events, model, rid, n));
            },
        }
    }
}

#[verifier::spinoff_prover]
pub proof fn lemma_execution_view(
    execution: X::Execution<Engine>, model: SemanticModelRepr, plan: RT::KernelPlanId, rid: u64,
)
    requires
        RT::paged_attention_numeric_domain(),
        X::legal_execution(I::system(model, plan), execution),
        X::requests_after(execution.initial_requests, execution.steps).contains_key(rid),
    ensures
        X::view(execution.steps, rid) == reference_run(model,
            X::requests_after(execution.initial_requests, execution.steps)[rid],
            X::view(execution.steps, rid).len()).outputs,
{
    hide(T::witness_events);
    hide(E::event_valid);
    hide(I::record);
    hide(I::transition);
    hide(X::records_well_formed);
    T::lemma_reconstruct_execution(execution, model, plan);
    let events = T::witness_events(execution);
    lemma_request_prefix(execution.states[0], events, model, plan, rid, events.len() as int);
    assert(T::records(events).take(events.len() as int) =~= execution.steps);
}

pub proof fn lemma_request_consistent(
    left: X::Execution<Engine>, right: X::Execution<Engine>,
    model: SemanticModelRepr, plan: RT::KernelPlanId, left_id: u64, right_id: u64,
)
    requires RT::paged_attention_numeric_domain(),
    ensures X::request_consistent(I::system(model, plan), left, right, left_id, right_id),
{
    let a = X::requests_after(left.initial_requests, left.steps);
    let b = X::requests_after(right.initial_requests, right.steps);
    if X::legal_execution(I::system(model, plan), left) && X::legal_execution(I::system(model, plan), right)
        && a.contains_key(left_id) && b.contains_key(right_id) && a[left_id] == b[right_id] {
        lemma_execution_view(left, model, plan, left_id);
        lemma_execution_view(right, model, plan, right_id);
        let m = X::view(left.steps, left_id).len();
        let n = X::view(right.steps, right_id).len();
        if m <= n { lemma_reference_run_prefix(model, a[left_id], m, n); }
        else { lemma_reference_run_prefix(model, a[left_id], n, m); }
    }
}

} // verus!
