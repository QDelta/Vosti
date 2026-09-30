//! Explicit whole-execution determinism specification, starting at initialization.
//!
//! Only the state representation is abstract. Requests and emitted records are
//! data, not configurable projections. `Map` has finite domain in vstd.
//! Logits use an opaque scalar representation. This specification adds no
//! byte-encoding axiom or numerical interpretation at the runtime boundary.
//! The concrete engine satisfaction theorem lives in `proof`; execution
//! records and reference-run witnesses are connected by checked proofs.

pub use crate::boundary::sampler::SamplerState;
pub use crate::boundary::scalar::Scalar;
pub use crate::types::{RequestId, TokenId};
use vstd::prelude::*;

verus! {

pub type Logits = Seq<Scalar>;

pub ghost struct RequestInput {
    pub prompt: Seq<TokenId>,
    pub initial_sampler_state: SamplerState,
}

pub ghost struct OutputRecord {
    pub logits: Logits,
    pub token: TokenId,
}

pub ghost struct Step {
    pub admitted: Map<RequestId, RequestInput>,
    pub outputs: Map<RequestId, OutputRecord>,
}

pub ghost struct Execution<State> {
    pub initial_requests: Map<RequestId, RequestInput>,
    pub states: Seq<State>,
    pub steps: Seq<Step>,
}

/// One fixed inference configuration per instantiation. The implementation
/// supplies execution legality, not the definition of observable equality.
#[verifier::reject_recursive_types(State)]
pub ghost struct System<State> {
    pub init: spec_fn(State, Map<RequestId, RequestInput>) -> bool,
    pub step: spec_fn(State, Step, State) -> bool,
}

/// Persistent input registry; completed requests are not forgotten.
pub open spec fn requests_after(
    initial: Map<RequestId, RequestInput>, steps: Seq<Step>,
) -> Map<RequestId, RequestInput>
    decreases steps.len(),
{
    if steps.len() == 0 { initial }
    else {
        requests_after(initial, steps.drop_last()).union_prefer_right(steps.last().admitted)
    }
}

pub open spec fn records_well_formed(
    initial: Map<RequestId, RequestInput>, steps: Seq<Step>,
) -> bool {
    forall|i: int| 0 <= i < steps.len() ==> {
        let seen = requests_after(initial, steps.take(i));
        let step = #[trigger] steps[i];
        &&& step.admitted.dom().disjoint(seen.dom())
        &&& step.outputs.dom().subset_of(seen.dom().union(step.admitted.dom()))
    }
}

pub open spec fn legal_execution<S>(system: System<S>, execution: Execution<S>) -> bool {
    &&& execution.states.len() == execution.steps.len() + 1
    &&& (system.init)(execution.states[0], execution.initial_requests)
    &&& records_well_formed(execution.initial_requests, execution.steps)
    &&& forall|i: int| 0 <= i < execution.steps.len() ==>
        (system.step)(execution.states[i], #[trigger] execution.steps[i], execution.states[i + 1])
}

/// The same fixed view for every implementation: retain emitted records for
/// this ID, in step order. Neither admissions nor other requests are outputs.
pub open spec fn view(steps: Seq<Step>, rid: RequestId) -> Seq<OutputRecord>
    decreases steps.len(),
{
    if steps.len() == 0 { Seq::empty() }
    else {
        let tail = view(steps.subrange(1, steps.len() as int), rid);
        if steps[0].outputs.contains_key(rid) { seq![steps[0].outputs[rid]] + tail }
        else { tail }
    }
}

pub open spec fn is_prefix(left: Seq<OutputRecord>, right: Seq<OutputRecord>) -> bool {
    left.len() <= right.len()
        && forall|i: int| #![trigger left[i], right[i]]
            0 <= i < left.len() ==> left[i] == right[i]
}

pub open spec fn prefix_comparable(left: Seq<OutputRecord>, right: Seq<OutputRecord>) -> bool {
    is_prefix(left, right) || is_prefix(right, left)
}

pub open spec fn request_consistent<S>(
    system: System<S>, left: Execution<S>, right: Execution<S>,
    left_id: RequestId, right_id: RequestId,
) -> bool {
    let a = requests_after(left.initial_requests, left.steps);
    let b = requests_after(right.initial_requests, right.steps);
    (legal_execution(system, left) && legal_execution(system, right)
        && a.contains_key(left_id) && b.contains_key(right_id)
        && a[left_id] == b[right_id])
    ==> prefix_comparable(view(left.steps, left_id), view(right.steps, right_id))
}

/// Safety only: this requires neither scheduling progress nor completion,
/// and does not state functional correctness of the model.
pub open spec fn deterministic<S>(system: System<S>) -> bool {
    forall|left: Execution<S>, right: Execution<S>, left_id: RequestId, right_id: RequestId|
        #[trigger] request_consistent(system, left, right, left_id, right_id)
}

} // verus!
