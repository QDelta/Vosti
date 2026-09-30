//! Concrete API interpretation for arbitrary stable-start request continuations.
//!
//! Shared effects come from serving::transitions; semantic admissibility is
//! confined to this supplementary proof's starting-state condition.

use crate::exec::engine::Engine;
use crate::proof::serving::continuation_agreement::vocabulary as S;
use crate::proof::serving::transitions::{Action, Event};
#[cfg(verus_only)]
use crate::proof::serving::transitions::event_valid;
use crate::proof::serving::refinement as P;
use S::{Observation, SamplingInput};
use crate::{types::{RequestId}, proof::model::types::{SemanticModelRepr}};
use crate::boundary::tensor_runtime as RT;
use vstd::prelude::*;

verus! {

pub type EngineTrace = S::Trace<Engine, Event>;
pub type EngineTraceInterface = S::TraceInterface<Engine, Event>;

pub open spec fn input(e: Engine, rid: RequestId) -> Option<SamplingInput> {
    if e.cs.live_requests@.contains_key(rid) {
        let request = e.cs.live_requests@[rid];
        Some(SamplingInput {
            prompt: request.prompt_tokens@,
            generated: request.generated_tokens@,
            sampler: request.sampler_state,
        })
    } else { None }
}

pub open spec fn output(event: Event, rid: RequestId) -> Option<Observation> {
    match event.action {
        Action::Admit(_) => None,
        Action::Compute { emitted, samples, .. } => {
            if emitted.contains_key(rid) { Some(samples[rid]) } else { None }
        },
    }
}

/// One interpretation per exact semantic model and qualified kernel plan.
/// These are bound once for both traces, not hidden inside request equality.
pub open spec fn engine_trace_interface(model: SemanticModelRepr, plan: RT::KernelPlanId) -> EngineTraceInterface {
    S::TraceInterface {
        admissible: |e: Engine| P::admissible(e, model, plan),
        before: |event: Event| event.before,
        after: |event: Event| event.after,
        step: |event: Event| event_valid(event),
        input: |e: Engine, rid: RequestId| input(e, rid),
        output: |event: Event, rid: RequestId| output(event, rid),
    }
}

} // verus!
