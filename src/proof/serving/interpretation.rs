//! Record the actual engine API effects for the explicit execution spec.
//!
//! Initialization/admission come from checked contracts. Compute
//! records use the batched forward representation actually passed to sampling,
//! not the cold reference forward. Reference equality is proved separately.

use crate::exec::engine::{self, Engine, StepReprs};
use super::transitions;
use crate::spec;
use crate::exec::request_state::RequestState;
use crate::{types::{RequestId, TokenId}, proof::model::types::{SemanticModelRepr}};
use crate::boundary::tensor_runtime as RT;
use vstd::prelude::*;

verus! {

pub open spec fn request_input(request: RequestState) -> spec::RequestInput {
    spec::RequestInput {
        prompt: request.prompt_tokens@,
        initial_sampler_state: request.sampler_state,
    }
}

pub open spec fn initial_inputs(engine: Engine) -> Map<RequestId, spec::RequestInput> {
    engine.cs.live_requests@.map_values(|request: RequestState| request_input(request))
}

/// Used only on the emitted domain. The record proof establishes existence
/// and uniqueness from the executed plan's row/mask contract.
pub open spec fn emitting_row(reprs: StepReprs, rid: RequestId) -> int {
    choose|i: int| 0 <= i < reprs.scheduled.len()
        && reprs.scheduled[i] == rid && #[trigger] reprs.sample_mask[i]
}

pub open spec fn compute_outputs(
    before: Engine, emitted: Map<RequestId, TokenId>, reprs: StepReprs,
) -> Map<RequestId, spec::OutputRecord> {
    Map::new(emitted.dom(), |rid: RequestId| spec::OutputRecord {
        logits: RT::select_sample_logits_repr(
            engine::architecture_step_logits_repr(before, reprs),
            reprs.cu_q, emitting_row(reprs, rid) as nat,
        ),
        token: emitted[rid],
    })
}

pub open spec fn record(event: transitions::Event) -> spec::Step {
    match event.action {
        transitions::Action::Admit(request) => spec::Step {
            admitted: Map::empty().insert(request.request_id, request_input(request)),
            outputs: Map::empty(),
        },
        transitions::Action::Compute { emitted, reprs, .. } => spec::Step {
            admitted: Map::empty(),
            outputs: compute_outputs(event.before, emitted, reprs),
        },
    }
}

pub open spec fn initialized(
    engine: Engine, inputs: Map<RequestId, spec::RequestInput>,
    model: SemanticModelRepr, plan: RT::KernelPlanId,
) -> bool {
    &&& RT::model_runtime_kernel_plan_id(&engine.runtime) == plan
    &&& inputs == initial_inputs(engine)
    &&& exists|config: crate::exec::cache_scheduler::SchedulerConfig, num_blocks: u64,
            requests: Seq<RequestState>|
        #[trigger] transitions::serving_init_relation(&engine, model, config, num_blocks, requests)
}

/// Witnesses are structural API effects, never certificates of agreement with
/// reference outputs. The record equality prevents dropping/replacing outputs.
/// Runtime identity follows from the Engine::step/add_request postconditions.
pub open spec fn transition(before: Engine, step: spec::Step, after: Engine) -> bool {
    after.runtime == before.runtime && exists|action: transitions::Action|
        #[trigger] transitions::event_valid(transitions::Event { before, after, action })
        && step == record(transitions::Event { before, after, action })
}

pub open spec fn system(model: SemanticModelRepr, plan: RT::KernelPlanId) -> spec::System<Engine> {
    spec::System {
        init: |e: Engine, inputs| initialized(e, inputs, model, plan),
        step: |before: Engine, step: spec::Step, after: Engine| transition(before, step, after),
    }
}

} // verus!
