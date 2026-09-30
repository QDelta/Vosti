//! Shared concrete engine API effects, independent of either trace specification.
//!
//! These definitions describe checked executable contracts. Semantic invariants
//! and their proofs live in refinement.

use crate::exec::cache_scheduler;
use crate::exec::cache_scheduler as CS;
use crate::exec::engine::{self, Engine, StepReprs};
#[cfg(verus_only)]
use crate::exec::request_state::initial_request_batch_ready;
use crate::exec::request_state::{self as RS, RequestState, SamplerState};
use crate::{types::{RequestId, TokenId}, proof::model::types::{SemanticModelRepr}};
use crate::boundary::tensor_runtime as RT;
use vstd::prelude::*;

verus! {

pub ghost enum Action {
    Compute {
        emitted: Map<RequestId, TokenId>,
        samples: Map<RequestId, (SamplerState, TokenId)>,
        reprs: StepReprs,
    },
    Admit(RequestState),
}

pub ghost struct Event {
    pub before: Engine,
    pub after: Engine,
    pub action: Action,
}

pub open spec fn semantic_model(e: Engine) -> SemanticModelRepr {
    SemanticModelRepr {
        weights: RT::model_weights_repr_of(&e.weights_perms@),
        architecture: RT::model_weights_architecture_repr_of(&e.weights_perms@),
    }
}

/// Engine-visible result of initializing a deployment and a fresh request
/// collection. Concrete tensor and permission setup is checked by the actual
/// `Engine::init` contract; this relation records the stable state it produces.
#[verifier::opaque]
pub open spec fn serving_init_relation(
    engine: &Engine,
    model: SemanticModelRepr,
    config: cache_scheduler::SchedulerConfig,
    num_blocks: u64,
    requests: Seq<RequestState>,
) -> bool {
    crate::exec::engine::engine_init_request_relation(
        engine, config, num_blocks, requests,
    )
    && initial_request_batch_ready(requests)
    && cache_scheduler::cs_valid(&engine.cs)
    && cache_scheduler::free_queue_valid(&engine.cs)
    && crate::exec::engine::eng_execution_perms_ok(engine)
    && crate::exec::engine::eng_cache_shape_ok(engine)
    && engine.model_config.num_layers > 0
    && engine.cs.num_blocks <= u64::MAX / crate::types::BLOCK_SIZE
    && RT::model_weights_repr_of(&engine.weights_perms@) == model.weights
    && RT::model_weights_architecture_repr_of(&engine.weights_perms@)
        == model.architecture
    && cache_scheduler::live_request_step_ready(&engine.cs)
    && cache_scheduler::residency_running_aligned(&engine.cs)
    && cache_scheduler::persistent_provenance_closed(&engine.cs)
}

/// Stable-boundary compute transition. The relation adds the checked
/// post-state safety facts returned by the executable engine to its detailed
/// semantic step relation, so the hidden invariant is inductive.
#[verifier::opaque]
pub open spec fn serving_step_relation(
    old_e: Engine,
    new_e: Engine,
    emitted: Map<RequestId, TokenId>,
    samples: Map<RequestId, (SamplerState, TokenId)>,
    reprs: StepReprs,
) -> bool {
    crate::exec::engine::architecture_engine_step_relation(
        old_e, new_e, emitted, samples, reprs,
    )
    && crate::exec::engine::engine_step_semantic_identity(old_e, new_e)
    && cache_scheduler::free_queue_valid(&new_e.cs)
    && crate::exec::engine::eng_execution_perms_ok(&new_e)
    && crate::exec::engine::eng_cache_shape_ok(&new_e)
    && cache_scheduler::residency_history_aligned(&new_e.cs)
    && cache_scheduler::slot_mapping_aligned(&new_e.cs)
    && cache_scheduler::tail_write_exclusive(&new_e.cs)
    && cache_scheduler::residency_running_aligned(&new_e.cs)
    && cache_scheduler::persistent_provenance_closed(&new_e.cs)
}

/// Preconditions/effects of an accepted Engine::add_request at a stable boundary.
/// This states no independent-model step or reference-output agreement.
#[verifier::opaque]
pub open spec fn admission(before: Engine, after: Engine, request: RequestState) -> bool {
    &&& engine::engine_admission_relation(&before, &after, request)
    &&& !before.cs.accepted_requests@.contains_key(request.request_id)
    &&& request.generated_tokens@.len() == 0
    &&& RS::can_step(request)
    &&& RS::request_history_capacity_safe(request)
    &&& CS::cs_valid(&after.cs)
    &&& CS::free_queue_valid(&after.cs)
    &&& engine::eng_execution_perms_ok(&after)
    &&& engine::eng_cache_shape_ok(&after)
    &&& CS::live_request_step_ready(&after.cs)
    &&& CS::residency_history_aligned(&after.cs)
    &&& CS::slot_mapping_aligned(&after.cs)
    &&& CS::tail_write_exclusive(&after.cs)
    &&& CS::residency_running_aligned(&after.cs)
    &&& CS::persistent_provenance_closed(&after.cs)
}

pub open spec fn event_valid(event: Event) -> bool {
    match event.action {
        Action::Admit(request) => admission(event.before, event.after, request),
        Action::Compute { emitted, samples, reprs } =>
            serving_step_relation(event.before, event.after, emitted, samples, reprs),
    }
}

} // verus!
