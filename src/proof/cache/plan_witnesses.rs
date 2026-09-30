//! Architecture-neutral cache-frame witnesses for scheduler plans.
//!
//! Model-family row semantics live behind `model_architecture`. This module
//! records only the common consequence needed for requests that survive a
//! step without being emitted: their private machine is unchanged and every
//! represented engine-cache cell still agrees with it.

use crate::exec::engine::{Engine, StepReprs};
use crate::proof::reference::independent_batch_model::IndependentBatchModel;
use crate::{types::*, proof::model::types::*, proof::tensor::types::*};
use vstd::prelude::*;

verus! {

#[verifier::opaque]
pub open spec fn unscheduled_no_touch(
    old_e: Engine,
    new_e: Engine,
    old_ibm: IndependentBatchModel,
    new_ibm: IndependentBatchModel,
    reprs: StepReprs,
    num_layers: nat,
) -> bool {
    forall|rid: RequestId, layer: int, pos: nat|
        #![trigger new_e.kv_caches_repr@[layer],
            crate::proof::tensor::geometry::block_table_slot(
                crate::proof::engine::refinement::residency_block_table(&new_e.cs, rid), pos)]
        new_e.cs.live_requests@.contains_key(rid)
        && new_ibm.machines.contains_key(rid)
        && !crate::exec::engine::reprs_emits(reprs, rid)
        && 0 <= layer < num_layers as int
        && pos < new_ibm.machines[rid].kv_tokens
        ==> {
            let bt = crate::proof::engine::refinement::residency_block_table(&new_e.cs, rid);
            let slot = crate::proof::tensor::geometry::block_table_slot(bt, pos);
            &&& new_ibm.machines[rid] == old_ibm.machines[rid]
            &&& crate::proof::tensor::geometry::slot_in_cache(
                new_e.kv_caches_repr@[layer].0, slot,
            )
            &&& crate::proof::tensor::geometry::slot_in_cache(
                new_e.kv_caches_repr@[layer].1, slot,
            )
            &&& crate::proof::tensor::geometry::cache_at(
                new_e.kv_caches_repr@[layer].0, slot,
            ) == crate::proof::tensor::geometry::cache_at(
                old_ibm.machines[rid].kv_cache_reprs[layer].0, pos,
            )
            &&& crate::proof::tensor::geometry::cache_at(
                new_e.kv_caches_repr@[layer].1, slot,
            ) == crate::proof::tensor::geometry::cache_at(
                old_ibm.machines[rid].kv_cache_reprs[layer].1, pos,
            )
        }
}

pub proof fn lemma_unscheduled_coherence_at(
    old_e: Engine,
    new_e: Engine,
    old_ibm: IndependentBatchModel,
    new_ibm: IndependentBatchModel,
    reprs: StepReprs,
    num_layers: nat,
    rid: RequestId,
    layer: int,
    pos: nat,
)
    requires
        unscheduled_no_touch(
            old_e, new_e, old_ibm, new_ibm, reprs, num_layers,
        ),
        new_e.cs.live_requests@.contains_key(rid),
        new_ibm.machines.contains_key(rid),
        !crate::exec::engine::reprs_emits(reprs, rid),
        0 <= layer < num_layers as int,
        pos < new_ibm.machines[rid].kv_tokens,
    ensures ({
        let bt = crate::proof::engine::refinement::residency_block_table(&new_e.cs, rid);
        let slot = crate::proof::tensor::geometry::block_table_slot(bt, pos);
        &&& new_ibm.machines[rid] == old_ibm.machines[rid]
        &&& crate::proof::tensor::geometry::slot_in_cache(
            new_e.kv_caches_repr@[layer].0, slot,
        )
        &&& crate::proof::tensor::geometry::slot_in_cache(
            new_e.kv_caches_repr@[layer].1, slot,
        )
        &&& crate::proof::tensor::geometry::cache_at(
            new_e.kv_caches_repr@[layer].0, slot,
        ) == crate::proof::tensor::geometry::cache_at(
            new_ibm.machines[rid].kv_cache_reprs[layer].0, pos,
        )
        &&& crate::proof::tensor::geometry::cache_at(
            new_e.kv_caches_repr@[layer].1, slot,
        ) == crate::proof::tensor::geometry::cache_at(
            new_ibm.machines[rid].kv_cache_reprs[layer].1, pos,
        )
    }),
{
    reveal(unscheduled_no_touch);
    let bt = crate::proof::engine::refinement::residency_block_table(&new_e.cs, rid);
    let slot = crate::proof::tensor::geometry::block_table_slot(bt, pos);
    let probe = new_e.kv_caches_repr@[layer];
    assert(new_ibm.machines[rid] == old_ibm.machines[rid]
        && crate::proof::tensor::geometry::slot_in_cache(
            new_e.kv_caches_repr@[layer].0, slot,
        )
        && crate::proof::tensor::geometry::slot_in_cache(
            new_e.kv_caches_repr@[layer].1, slot,
        )
        && crate::proof::tensor::geometry::cache_at(
            new_e.kv_caches_repr@[layer].0, slot,
        ) == crate::proof::tensor::geometry::cache_at(
            old_ibm.machines[rid].kv_cache_reprs[layer].0, pos,
        )
        && crate::proof::tensor::geometry::cache_at(
            new_e.kv_caches_repr@[layer].1, slot,
        ) == crate::proof::tensor::geometry::cache_at(
            old_ibm.machines[rid].kv_cache_reprs[layer].1, pos,
        ));
}

} // verus!
