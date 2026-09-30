//! Concrete serving interpretation and checked execution composition.
//!
//! The satisfaction theorem is in the parent `proof` module.

pub mod interpretation;
pub mod transitions;
pub mod refinement;
pub mod records;
pub mod trace;
pub mod consistency;
// Supplementary arbitrary-start agreement, not a second determinism definition.
pub(crate) mod continuation_agreement;
