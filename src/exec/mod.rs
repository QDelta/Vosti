//! Verified executable implementation.
//!
//! Mutation, scheduling, model-forward orchestration, and the serving engine
//! live here. External contracts used by this code are owned by `boundary`;
//! refinement and semantic arguments are owned by `proof`.

pub mod cache_scheduler;
pub mod engine;
pub mod model;
pub mod model_families;
pub(crate) mod model_kv_loop;
// Keep the reusable dense layer implementation after the top-level engine:
// Engine consumes only the family-forward contract, and including these
// internal declarations in its solver environment needlessly destabilizes the
// large `step` query.
pub(crate) mod dense_swiglu_decoder;
pub(crate) mod dense_swiglu_model;
pub(crate) mod four_norm_gated_decoder;
pub(crate) mod four_norm_gated_model;
pub mod step_observation;

pub mod request_state;
