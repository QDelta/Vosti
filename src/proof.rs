//! The engine satisfies the public determinism definition in `spec`.
//!
//! The contract names the definition and concrete interpretation; the body
//! delegates to the execution-consistency proof. Supporting modules follow.

use serving::{interpretation, consistency};
use crate::exec::engine::Engine;
use crate::{types::{RequestId}, proof::model::types::{SemanticModelRepr}};
use crate::{spec, boundary::tensor_runtime};
use vstd::prelude::*;

verus! {

// @kernel-bridge-begin proof::lemma_serving_deterministic
pub proof fn lemma_serving_deterministic(
    model: SemanticModelRepr, plan: tensor_runtime::KernelPlanId,
)
    requires tensor_runtime::paged_attention_numeric_domain(),
    ensures spec::deterministic(interpretation::system(model, plan)),
{
    assert forall|left: spec::Execution<Engine>, right: spec::Execution<Engine>,
        left_id: RequestId, right_id: RequestId|
        #[trigger] spec::request_consistent(interpretation::system(model, plan),
            left, right, left_id, right_id) by {
        consistency::lemma_request_consistent(left, right, model, plan, left_id, right_id);
    }
}
// @kernel-bridge-end proof::lemma_serving_deterministic

} // verus!

// Concrete trace interpretation and composition.
pub mod serving;
// Supporting developments, grouped by responsibility.
pub mod engine;
pub mod model;
pub mod cache;
pub mod scheduler;
pub mod tensor;

pub mod reference;
