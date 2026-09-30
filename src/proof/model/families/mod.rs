//! Proof adapters organized by layer composition rather than model name.
//!
//! Architectures that share a forward graph share one proof composition here.
//! The generic dispatcher binds a closed architecture payload to that
//! composition; genuinely different graphs retain separate implementations.

use crate::exec::engine::*;
use crate::{types::*, proof::model::types::*, proof::tensor::types::*};
use crate::boundary::tensor_runtime as RT;
use vstd::prelude::*;

verus! {

// Canonical context supplied by Engine to every family-local forward-readiness
// proof. Architecture-specific configuration facts are intentionally absent:
// each family derives them from the common execution capability inside its
// adapter.
pub open spec fn engine_forward_context(
    old_e: Engine,
    reprs: StepReprs,
    wp: &RT::ModelWeightsPerms,
    pre_kv: Seq<(KVCacheLayerRepr, KVCacheLayerRepr)>,
) -> bool {
    &&& RT::model_execution_valid(&old_e.weights, &old_e.runtime, wp)
    &&& step_reprs_wf(old_e, reprs)
    &&& reprs.wr == RT::model_weights_repr_of(wp)
    &&& reprs.input_ids.len() > 0
    &&& pre_kv.len() == wp.num_layers()
    &&& forall|i: int| 0 <= i < wp.num_layers() as int ==>
        #[trigger] RT::store_kv_cache_launch_ready(
            reprs.input_ids.len(), pre_kv[i].0, pre_kv[i].1, reprs.slots,
        )
    &&& forall|i: int| 0 <= i < wp.num_layers() as int ==>
        #[trigger] RT::paged_attention_launch_ready(
            reprs.input_ids.len(), pre_kv[i].0, pre_kv[i].1,
            reprs.cu_q, reprs.cu_k, reprs.max_q, reprs.max_k, reprs.bt,
        )
}

} // verus!

pub mod dense_swiglu;
pub mod four_norm_gated;
