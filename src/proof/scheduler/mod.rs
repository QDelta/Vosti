//! Cache-scheduler specifications, invariants, and preservation proofs.
//!
//! The executable scheduler re-exports this module's public contracts so the
//! established `cache_scheduler::*` facade remains stable.

use crate::proof::tensor::geometry::*;
use crate::exec::cache_scheduler::availability_queue::*;
#[cfg(verus_only)]
use crate::exec::cache_scheduler::step_plan_shape_ok;
use crate::exec::cache_scheduler::types::*;
use crate::exec::request_state::*;
use crate::{types::*, proof::model::types::*, proof::tensor::types::*};
use crate::boundary::tensor_runtime as RT;
use vstd::hash_map::HashMapWithView;
use vstd::prelude::*;
#[cfg(verus_only)]
use vstd::std_specs::hash::obeys_key_model;
use vstd::{assert_seqs_equal, assert_sets_equal};

mod invariants;
pub use invariants::*;
mod transition_invariants;
pub use transition_invariants::*;
mod planning_invariants;
pub use planning_invariants::*;
mod publication;
pub use publication::*;
mod executed_publication;
pub use executed_publication::*;
