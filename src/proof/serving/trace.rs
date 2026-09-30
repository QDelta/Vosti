//! Lossless whole-execution assembly and reconstruction at the engine
//! API boundary. Registry well-formedness is proved from the accepted-ID ledger,
//! not imposed as an extra restriction on actual executions.

use crate::exec::cache_scheduler as CS;
use crate::exec::engine::{self, Engine};
use super::{transitions as E, interpretation as I};
use crate::spec as X;
use crate::proof::serving::refinement as P;
use crate::{proof::model::types::{SemanticModelRepr}};
use crate::boundary::tensor_runtime as RT;
use vstd::prelude::*;

verus! {

pub open spec fn state_at(initial: Engine, events: Seq<E::Event>, i: int) -> Engine {
    if i == 0 { initial } else { events[i - 1].after }
}

// Trigger on the state position, not events[i]: the latter also introduces
// events[i - 1] through state_at and can recursively instantiate predecessors.
pub open spec fn chained(initial: Engine, events: Seq<E::Event>) -> bool {
    forall|i: int| 0 <= i < events.len() ==> {
        &&& E::event_valid(events[i])
        &&& events[i].before == #[trigger] state_at(initial, events, i)
        &&& events[i].after.runtime == events[i].before.runtime
    }
}

pub open spec fn records(events: Seq<E::Event>) -> Seq<X::Step> {
    Seq::new(events.len(), |i: int| I::record(events[i]))
}

pub open spec fn execution_of(initial: Engine, events: Seq<E::Event>) -> X::Execution<Engine> {
    X::Execution {
        initial_requests: I::initial_inputs(initial),
        states: Seq::new(events.len() + 1, |i: int| state_at(initial, events, i)),
        steps: records(events),
    }
}

/// Choose only the structural API witness already present in the transition.
/// lemma_witness_event preserves endpoints; lemma_reconstruct_execution preserves
/// the output records and input registry. No emissions are filtered.
pub open spec fn witness_events(execution: X::Execution<Engine>) -> Seq<E::Event> {
    Seq::new(execution.steps.len(), |i: int| E::Event {
        before: execution.states[i], after: execution.states[i + 1],
        action: choose|action: E::Action|
            #[trigger] E::event_valid(E::Event {
                before: execution.states[i], after: execution.states[i + 1], action })
            && execution.steps[i] == I::record(E::Event {
                before: execution.states[i], after: execution.states[i + 1], action }),
    })
}

#[verifier::spinoff_prover]
pub proof fn lemma_initial_execution_facts(initial: Engine, model: SemanticModelRepr, plan: RT::KernelPlanId)
    requires I::initialized(initial, I::initial_inputs(initial), model, plan),
    ensures
        CS::cs_valid(&initial.cs),
        I::initial_inputs(initial).dom() == initial.cs.accepted_requests@.dom(),
        P::admissible(initial, model, plan),
        forall|rid: u64| initial.cs.live_requests@.contains_key(rid) ==>
            (#[trigger] initial.cs.live_requests@[rid]).generated_tokens@.len() == 0,
{
    let (config, num_blocks, requests) = choose|config: CS::SchedulerConfig, num_blocks: u64,
        requests: Seq<crate::exec::request_state::RequestState>|
        #[trigger] E::serving_init_relation(&initial, model, config, num_blocks, requests);
    P::lemma_init_establishes_serving_inv(initial, model, config, num_blocks, requests);
    reveal(E::serving_init_relation);
    reveal(engine::engine_init_request_relation);
    assert forall|rid: u64| initial.cs.live_requests@.contains_key(rid) implies
        (#[trigger] initial.cs.live_requests@[rid]).generated_tokens@.len() == 0 by {
        let k = choose|k: int| 0 <= k < requests.len() && requests[k].request_id == rid;
        assert(crate::exec::request_state::request_state_view_eq(initial.cs.live_requests@[rid], requests[k]));
    }
}

/// These are structural consequences of the init/admit/step contracts.
/// No semantic refinement or numeric-domain assumption is needed for the ledger.
#[verifier::spinoff_prover]
pub proof fn lemma_event_execution_facts(event: E::Event)
    requires
        E::event_valid(event), CS::cs_valid(&event.before.cs),
        event.after.runtime == event.before.runtime,
    ensures
        CS::cs_valid(&event.after.cs),
        I::record(event).admitted.dom().disjoint(event.before.cs.accepted_requests@.dom()),
        I::record(event).outputs.dom().subset_of(event.before.cs.accepted_requests@.dom()),
        event.after.cs.accepted_requests@.dom()
            == event.before.cs.accepted_requests@.dom().union(I::record(event).admitted.dom()),
        E::semantic_model(event.after) == E::semantic_model(event.before),
        RT::model_runtime_kernel_plan_id(&event.after.runtime)
            == RT::model_runtime_kernel_plan_id(&event.before.runtime),
{
    match event.action {
        E::Action::Admit(_) => { reveal(E::admission); },
        E::Action::Compute { .. } => {
            reveal(E::serving_step_relation);
            reveal(engine::engine_step_semantic_identity);
        },
    }
}

/// Induction over every actual API call. Completed requests remain in the
/// accepted-ID ledger even after leaving the live map.
pub proof fn lemma_prefix_execution_facts(initial: Engine, events: Seq<E::Event>, n: int)
    requires
        chained(initial, events), CS::cs_valid(&initial.cs),
        I::initial_inputs(initial).dom() == initial.cs.accepted_requests@.dom(),
        0 <= n <= events.len(),
    ensures
        CS::cs_valid(&state_at(initial, events, n).cs),
        X::requests_after(I::initial_inputs(initial), records(events).take(n)).dom()
            == state_at(initial, events, n).cs.accepted_requests@.dom(),
        E::semantic_model(state_at(initial, events, n)) == E::semantic_model(initial),
        RT::model_runtime_kernel_plan_id(&state_at(initial, events, n).runtime)
            == RT::model_runtime_kernel_plan_id(&initial.runtime),
    decreases n,
{
    if n > 0 {
        lemma_prefix_execution_facts(initial, events, n - 1);
        lemma_event_execution_facts(events[n - 1]);
        let prefix = records(events).take(n);
        assert(prefix.drop_last() =~= records(events).take(n - 1));
        assert(prefix.last() == I::record(events[n - 1]));
    }
}

/// No well-formed-record premise: actual executions establish it themselves.
pub proof fn lemma_record_execution(
    initial: Engine, events: Seq<E::Event>, model: SemanticModelRepr, plan: RT::KernelPlanId,
)
    requires
        I::initialized(initial, I::initial_inputs(initial), model, plan),
        chained(initial, events),
    ensures X::legal_execution(I::system(model, plan), execution_of(initial, events)),
{
    lemma_initial_execution_facts(initial, model, plan);
    let execution = execution_of(initial, events);
    assert forall|i: int| 0 <= i < events.len() implies {
        let seen = X::requests_after(execution.initial_requests, execution.steps.take(i));
        let step = #[trigger] execution.steps[i];
        &&& step.admitted.dom().disjoint(seen.dom())
        &&& step.outputs.dom().subset_of(seen.dom().union(step.admitted.dom()))
    } by {
        lemma_prefix_execution_facts(initial, events, i);
        lemma_event_execution_facts(events[i]);
    }
    assert forall|i: int| 0 <= i < execution.steps.len() implies
        (I::system(model, plan).step)(execution.states[i], #[trigger] execution.steps[i], execution.states[i + 1]) by {
        assert(events[i].before == state_at(initial, events, i));
        crate::proof::serving::records::lemma_record_transition(events[i]);
    }
}

/// Conversely, every legal abstract execution has the same states and records
/// as a chain of structural engine API effects. No reference-output premise.
pub proof fn lemma_witness_event(execution: X::Execution<Engine>, i: int)
    requires
        0 <= i < execution.steps.len(),
        I::transition(execution.states[i], execution.steps[i], execution.states[i + 1]),
    ensures
        E::event_valid(witness_events(execution)[i]),
        I::record(witness_events(execution)[i]) == execution.steps[i],
        witness_events(execution)[i].before == execution.states[i],
        witness_events(execution)[i].after == execution.states[i + 1],
        witness_events(execution)[i].after.runtime == witness_events(execution)[i].before.runtime,
{
    hide(E::event_valid);
    hide(I::record);
}

#[verifier::spinoff_prover]
pub proof fn lemma_reconstruct_chain(execution: X::Execution<Engine>)
    requires
        execution.states.len() == execution.steps.len() + 1,
        forall|i: int| 0 <= i < execution.steps.len() ==>
            I::transition(execution.states[i], #[trigger] execution.steps[i], execution.states[i + 1]),
    ensures
        chained(execution.states[0], witness_events(execution)),
        records(witness_events(execution)) == execution.steps,
{
    hide(E::event_valid);
    hide(I::record);
    hide(witness_events);
    hide(I::transition);
    let events = witness_events(execution);
    assert(events.len() == execution.steps.len()) by { reveal(witness_events); }
    assert forall|i: int| 0 <= i < events.len() implies {
        &&& E::event_valid(#[trigger] events[i])
        &&& I::record(events[i]) == execution.steps[i]
        &&& events[i].before == execution.states[i]
        &&& events[i].after == execution.states[i + 1]
        &&& events[i].after.runtime == events[i].before.runtime
    } by {
        lemma_witness_event(execution, i);
    }
    assert(chained(execution.states[0], events)) by {
        assert forall|i: int| 0 <= i < events.len() implies {
            &&& E::event_valid(events[i])
            &&& events[i].before == #[trigger] state_at(execution.states[0], events, i)
            &&& events[i].after.runtime == events[i].before.runtime
        } by { if i > 0 { assert(events[i - 1].after == execution.states[i]); } }
    }
    assert(records(events) =~= execution.steps) by {
        assert forall|i: int| 0 <= i < execution.steps.len() implies
            records(events)[i] == #[trigger] execution.steps[i] by {}
    }
}

pub proof fn lemma_reconstruct_execution(
    execution: X::Execution<Engine>, model: SemanticModelRepr, plan: RT::KernelPlanId,
)
    requires X::legal_execution(I::system(model, plan), execution),
    ensures
        chained(execution.states[0], witness_events(execution)),
        records(witness_events(execution)) == execution.steps,
        I::initial_inputs(execution.states[0]) == execution.initial_requests,
{
    hide(E::event_valid);
    hide(I::record);
    hide(witness_events);
    hide(I::transition);
    hide(chained);
    hide(records);
    hide(X::records_well_formed);
    lemma_reconstruct_chain(execution);
}

/// The checked initial invariant is preserved along every reconstructed call,
/// with the original model and kernel plan, including admission-only steps.
pub proof fn lemma_prefix_invariant(
    initial: Engine, events: Seq<E::Event>, model: SemanticModelRepr,
    plan: RT::KernelPlanId, n: int,
)
    requires
        RT::paged_attention_numeric_domain(),
        I::initialized(initial, I::initial_inputs(initial), model, plan),
        chained(initial, events), 0 <= n <= events.len(),
    ensures P::admissible(state_at(initial, events, n), model, plan),
    decreases n,
{
    lemma_initial_execution_facts(initial, model, plan);
    if n > 0 {
        lemma_prefix_invariant(initial, events, model, plan, n - 1);
        lemma_prefix_execution_facts(initial, events, n);
        P::lemma_event_preserves_admissibility(events[n - 1]);
    }
}

} // verus!
