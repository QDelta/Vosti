//! Internal vocabulary for pairwise continuation prefix agreement.
//!
//! Arbitrary stable starts may include generated history. This is supporting
//! proof machinery, not a second public definition of serving determinism.

pub use crate::boundary::sampler::SamplerState;
pub use crate::types::{RequestId, TokenId};
use vstd::prelude::*;

verus! {

/// Request identity excludes IDs and lifecycle controls, which may differ.
pub ghost struct SamplingInput {
    pub prompt: Seq<TokenId>,
    pub generated: Seq<TokenId>,
    pub sampler: SamplerState,
}

/// The observed sampler state and selected token, not raw logits.
pub type Observation = (SamplerState, TokenId);

pub ghost struct Trace<State, Event> {
    pub initial: State,
    pub final_state: State,
    pub events: Seq<Event>,
}

/// Events carry their endpoints through these projections. `step` describes
/// legal transitions, including any chosen invariant/certificate obligations.
/// A request is observed starting at a state where `input` returns Some.
/// Dynamic admission is an ordinary event; start its request trace after it.
#[verifier::reject_recursive_types(State)]
#[verifier::reject_recursive_types(Event)]
pub ghost struct TraceInterface<State, Event> {
    pub admissible: spec_fn(State) -> bool,
    pub before: spec_fn(Event) -> State,
    pub after: spec_fn(Event) -> State,
    pub step: spec_fn(Event) -> bool,
    pub input: spec_fn(State, RequestId) -> Option<SamplingInput>,
    pub output: spec_fn(Event, RequestId) -> Option<Observation>,
}

pub open spec fn legal_trace<S, E>(
    system: TraceInterface<S, E>, trace: Trace<S, E>,
) -> bool {
    &&& (system.admissible)(trace.initial)
    &&& forall|i: int| 0 <= i < trace.events.len() ==>
        (system.step)(#[trigger] trace.events[i])
    &&& forall|i: int| 0 <= i && i + 1 < trace.events.len() ==>
        (system.after)(#[trigger] trace.events[i])
            == (system.before)(#[trigger] trace.events[i + 1])
    &&& if trace.events.len() == 0 {
        trace.initial == trace.final_state
    } else {
        (system.before)(trace.events[0]) == trace.initial
        && (system.after)(trace.events[trace.events.len() - 1]) == trace.final_state
    }
}

/// Erase other requests, admission-only events, and non-emitting compute steps.
pub open spec fn observations<E>(
    output: spec_fn(E, RequestId) -> Option<Observation>, events: Seq<E>, rid: RequestId,
) -> Seq<Observation>
    decreases events.len(),
{
    if events.len() == 0 {
        Seq::empty()
    } else {
        let rest = observations(output, events.subrange(1, events.len() as int), rid);
        match output(events[0], rid) {
            Some(value) => seq![value] + rest,
            None => rest,
        }
    }
}

pub open spec fn common_prefix_equal(left: Seq<Observation>, right: Seq<Observation>) -> bool {
    forall|i: int| #![trigger left[i], right[i]]
        0 <= i < left.len() && i < right.len() ==> left[i] == right[i]
}

/// IDs, batching, timing, and output lengths may differ between executions.
/// Only equal *present* request inputs impose an observational obligation.
pub open spec fn request_consistent<S, E>(
    system: TraceInterface<S, E>,
    left: Trace<S, E>, right: Trace<S, E>, left_id: RequestId, right_id: RequestId,
) -> bool {
    (legal_trace(system, left) && legal_trace(system, right)
        && (system.input)(left.initial, left_id).is_some()
        && (system.input)(left.initial, left_id) == (system.input)(right.initial, right_id))
    ==> common_prefix_equal(
        observations(system.output, left.events, left_id),
        observations(system.output, right.events, right_id),
    )
}

} // verus!
