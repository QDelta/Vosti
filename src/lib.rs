//! Vosti: verified deterministic LLM serving.
//!
//! Engine identities live in `types`, model configuration in `model_config`,
//! and checked execution in `exec`,
//! external contracts in `boundary`, the public definition in `spec`, and
//! reference models and supporting arguments in `proof`.

// Verus keeps ghost imports/bindings that ordinary Rust erases. Do not suppress
// dead_code: unused private executable code should remain visible.
#![allow(unused_imports)]
#![allow(unused_variables)]
#![allow(unused_assignments)]

pub mod boundary;
pub mod exec;
pub mod model_config;
pub mod proof;
pub mod spec;
pub mod types;
