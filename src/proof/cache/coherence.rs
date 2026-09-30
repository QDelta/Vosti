//! Architecture-neutral KV-coherence discharge for an Engine step.
//!
//! Emitted requests use the closed model-dispatch projection and relocation
//! contract. Unemitted survivors use the common cache-frame witness. No
//! concrete decoder fold is imported here.

#[cfg(verus_only)]
use crate::{types::{BLOCK_SIZE_SPEC}, proof::tensor::geometry::{block_table_slot, blocks_needed_for, cache_at, slot_in_cache}};
use crate::exec::engine::Engine;
use crate::proof::reference::independent_batch_model::IndependentBatchModel;
#[cfg(verus_only)]
use crate::proof::engine::refinement::{engine_kv_coherent, engine_kv_coherent_at, residency_block_table};
#[cfg(verus_only)]
use crate::proof::reference::request_machine::slots_from;
use crate::{types::*, proof::model::types::*, proof::tensor::types::*};
use crate::boundary::tensor_runtime as RT;
use vstd::prelude::*;

verus! {

use crate::exec::engine::StepReprs;
#[cfg(verus_only)]
use crate::proof::cache::plan_witnesses::unscheduled_no_touch;

// Cache-shape bridge for the architecture-neutral coherence proof. A post-step
// residency block is in range of the unchanged physical
// page pool, so the same logical position is addressable in the pre-forward
// engine cache at every valid layer.
pub proof fn lemma_residency_slot_in_pre_cache(
    old_e: Engine,
    new_e: Engine,
    rid: RequestId,
    layer: int,
    pos: nat,
)
    requires
        crate::exec::engine::eng_cache_shape_ok(&old_e),
        0 <= layer < old_e.kv_caches_repr@.len(),
        crate::exec::cache_scheduler::cs_valid(&new_e.cs),
        new_e.cs.request_residency@.contains_key(rid),
        new_e.cs.num_blocks == old_e.cs.num_blocks,
        (pos as int) / (BLOCK_SIZE_SPEC as int)
            < residency_block_table(&new_e.cs, rid).len(),
    ensures
        slot_in_cache(
            old_e.kv_caches_repr@[layer].0,
            block_table_slot(residency_block_table(&new_e.cs, rid), pos),
        ),
        slot_in_cache(
            old_e.kv_caches_repr@[layer].1,
            block_table_slot(residency_block_table(&new_e.cs, rid), pos),
        ),
{
    let block_size = BLOCK_SIZE_SPEC as int;
    let bt = residency_block_table(&new_e.cs, rid);
    let block_index = (pos as int) / block_size;
    assert(new_e.cs.request_residency@[rid].block_ids@[block_index]
        == bt[block_index]);
    assert(new_e.cs.blocks@.contains_key(bt[block_index])) by {
        assert(crate::exec::cache_scheduler::residency_blocks_in_range(
            &new_e.cs,
        ));
    }
    assert(bt[block_index] < new_e.cs.num_blocks) by {
        assert(crate::exec::cache_scheduler::blocks_dom_in_range(&new_e.cs));
    }
    crate::proof::tensor::geometry::block_table_slot_block(bt, pos);
    let slot = block_table_slot(bt, pos);
    assert((slot as int) / block_size == bt[block_index] as int);
    assert((slot as int) / block_size < old_e.cs.num_blocks as int);
    vstd::arithmetic::div_mod::lemma_fundamental_div_mod(
        slot as int, block_size,
    );
    vstd::arithmetic::div_mod::lemma_mod_bound(slot as int, block_size);
    assert(old_e.kv_caches_repr@[layer].0.len()
        == old_e.cs.num_blocks as nat);
    assert(old_e.kv_caches_repr@[layer].0[
        (slot as int) / block_size
    ].len() == block_size);
    assert(old_e.kv_caches_repr@[layer].1[
        (slot as int) / block_size
    ].len() == block_size);
}

// The stable plan row bounds every referenced block by the engine's physical
// page count.  Together with the cache-shape companion, this makes every
// logical position below the row's key length addressable before the forward.
pub proof fn lemma_plan_slot_in_pre_cache(
    old_e: Engine,
    reprs: StepReprs,
    row: int,
    layer: int,
    pos: nat,
)
    requires
        crate::exec::engine::eng_cache_shape_ok(&old_e),
        crate::exec::engine::step_reprs_wf(old_e, reprs),
        crate::exec::engine::reprs_forward_layout_ok(old_e, reprs),
        crate::exec::engine::reprs_forward_layout_at(old_e, reprs, row),
        0 <= row < reprs.scheduled.len(),
        0 <= layer < reprs.wr.layers.len(),
        old_e.kv_caches_repr@.len() >= reprs.wr.layers.len(),
        pos < (reprs.cu_k[row + 1] - reprs.cu_k[row]) as nat,
    ensures
        slot_in_cache(
            old_e.kv_caches_repr@[layer].0,
            block_table_slot(reprs.bt[row], pos),
        ),
        slot_in_cache(
            old_e.kv_caches_repr@[layer].1,
            block_table_slot(reprs.bt[row], pos),
        ),
{
    let block_size = BLOCK_SIZE_SPEC as int;
    let key_len = reprs.cu_k[row + 1] - reprs.cu_k[row];
    let block_index = (pos as int) / block_size;
    reveal(crate::exec::engine::eng_cache_shape_ok);
    reveal(crate::exec::engine::step_reprs_wf);
    reveal(crate::exec::engine::reprs_forward_layout_at);
    crate::proof::tensor::geometry::lemma_blocks_needed_covers_pos(pos, key_len as nat);
    assert(0 <= block_index < reprs.bt[row].len());
    assert(reprs.bt[row][block_index] < old_e.cs.num_blocks);
    crate::proof::tensor::geometry::block_table_slot_block(reprs.bt[row], pos);
    let slot = block_table_slot(reprs.bt[row], pos);
    assert((slot as int) / block_size
        == reprs.bt[row][block_index] as int);
    vstd::arithmetic::div_mod::lemma_fundamental_div_mod(
        slot as int, block_size,
    );
    vstd::arithmetic::div_mod::lemma_mod_bound(slot as int, block_size);
    assert(old_e.kv_caches_repr@[layer].0.len()
        == old_e.cs.num_blocks as nat);
    assert(old_e.kv_caches_repr@[layer].1.len()
        == old_e.cs.num_blocks as nat);
    assert(old_e.kv_caches_repr@[layer].0[
        (slot as int) / block_size
    ].len() == block_size);
    assert(old_e.kv_caches_repr@[layer].1[
        (slot as int) / block_size
    ].len() == block_size);
}

// Full-page variant consumed by the sliding-window projection adapter. Staged SWA and
// full attention share the same allocated key pages today, so readiness covers
// every cell of the final page even though logical coherence is stated only
// through the exact causal `k_len`.
pub proof fn lemma_plan_page_slot_in_pre_cache(
    old_e: Engine,
    reprs: StepReprs,
    row: int,
    layer: int,
    pos: nat,
)
    requires
        crate::exec::engine::eng_cache_shape_ok(&old_e),
        crate::exec::engine::step_reprs_wf(old_e, reprs),
        crate::exec::engine::reprs_forward_layout_ok(old_e, reprs),
        crate::exec::engine::reprs_forward_layout_at(old_e, reprs, row),
        0 <= row < reprs.scheduled.len(),
        0 <= layer < reprs.wr.layers.len(),
        old_e.kv_caches_repr@.len() >= reprs.wr.layers.len(),
        pos < crate::proof::tensor::geometry::blocks_needed_for(
            (reprs.cu_k[row + 1] - reprs.cu_k[row]) as nat,
        ) * BLOCK_SIZE_SPEC,
    ensures
        slot_in_cache(
            old_e.kv_caches_repr@[layer].0,
            block_table_slot(reprs.bt[row], pos),
        ),
        slot_in_cache(
            old_e.kv_caches_repr@[layer].1,
            block_table_slot(reprs.bt[row], pos),
        ),
{
    let block_size = BLOCK_SIZE_SPEC as int;
    let key_len = (reprs.cu_k[row + 1] - reprs.cu_k[row]) as nat;
    let pages = blocks_needed_for(key_len);
    let block_index = (pos as int) / block_size;
    reveal(crate::exec::engine::eng_cache_shape_ok);
    reveal(crate::exec::engine::step_reprs_wf);
    reveal(crate::exec::engine::reprs_forward_layout_at);
    vstd::arithmetic::div_mod::lemma_div_is_ordered(
        pos as int,
        (pages * BLOCK_SIZE_SPEC) as int,
        block_size,
    );
    assert(0 <= block_index < pages as int);
    assert(pages <= reprs.bt[row].len());
    assert(0 <= block_index < reprs.bt[row].len());
    assert(reprs.bt[row][block_index] < old_e.cs.num_blocks);
    crate::proof::tensor::geometry::block_table_slot_block(reprs.bt[row], pos);
    let slot = block_table_slot(reprs.bt[row], pos);
    assert((slot as int) / block_size
        == reprs.bt[row][block_index] as int);
    vstd::arithmetic::div_mod::lemma_fundamental_div_mod(
        slot as int, block_size,
    );
    vstd::arithmetic::div_mod::lemma_mod_bound(slot as int, block_size);
    assert(old_e.kv_caches_repr@[layer].0.len()
        == old_e.cs.num_blocks as nat);
    assert(old_e.kv_caches_repr@[layer].1.len()
        == old_e.cs.num_blocks as nat);
    assert(old_e.kv_caches_repr@[layer].0[
        (slot as int) / block_size
    ].len() == block_size);
    assert(old_e.kv_caches_repr@[layer].1[
        (slot as int) / block_size
    ].len() == block_size);
}

// Recover the common paged-attention launch domain from stable scheduler rows
// and the engine cache-shape companion.  This is not a model-family fact;
// family projection adapters consume it through the common domain.
pub proof fn lemma_reprs_paged_attention_launch_ready(
    old_e: Engine,
    reprs: StepReprs,
)
    requires
        crate::exec::engine::eng_cache_shape_ok(&old_e),
        crate::exec::engine::step_reprs_wf(old_e, reprs),
        crate::exec::engine::reprs_forward_layout_ok(old_e, reprs),
        reprs.scheduled.len() > 0,
        old_e.kv_caches_repr@.len() >= reprs.wr.layers.len(),
    ensures
        forall|layer: int| 0 <= layer < reprs.wr.layers.len() ==>
            #[trigger] RT::paged_attention_launch_ready(
                reprs.input_ids.len(),
                old_e.kv_caches_repr@[layer].0,
                old_e.kv_caches_repr@[layer].1,
                reprs.cu_q,
                reprs.cu_k,
                reprs.max_q,
                reprs.max_k,
                reprs.bt,
            ),
{
    reveal(crate::exec::engine::step_reprs_wf);
    reveal(RT::paged_attention_metadata_ready);
    let num_rows = reprs.scheduled.len() as int;
    let first: int = 0;
    assert(num_rows > 0);
    assert(0 <= first < num_rows);
    assert forall|row: int| 0 <= row < reprs.scheduled.len() as int
        implies reprs.cu_q[row] < #[trigger] reprs.cu_q[row + 1]
    by {}
    assert forall|row: int| 0 <= row < reprs.scheduled.len() as int
        implies reprs.cu_k[row] < #[trigger] reprs.cu_k[row + 1]
    by {}
    assert(reprs.cu_q[first] < reprs.cu_q[first + 1]);
    assert(reprs.cu_k[first] < reprs.cu_k[first + 1]);
    crate::proof::tensor::geometry::lemma_cu_mono(reprs.cu_q, num_rows, 1, num_rows);
    assert(reprs.input_ids.len() > 0);
    assert(reprs.max_q > 0);
    assert(reprs.max_k > 0);

    assert(crate::exec::engine::reprs_forward_layout_at(
        old_e, reprs, first,
    ));
    reveal(crate::exec::engine::reprs_forward_layout_at);
    let first_k_len = reprs.cu_k[first + 1] - reprs.cu_k[first];
    assert(first_k_len > 0);
    assert(blocks_needed_for(first_k_len as nat)
        <= reprs.bt[first].len());
    reveal(blocks_needed_for);
    assert(reprs.bt[first].len() > 0);
    assert(reprs.bt[first][first] < old_e.cs.num_blocks);
    assert(old_e.cs.num_blocks > 0);

    assert forall|row: int| 0 <= row < reprs.bt.len() implies {
        let q_len = reprs.cu_q[row + 1] - reprs.cu_q[row];
        let k_len = reprs.cu_k[row + 1] - reprs.cu_k[row];
        &&& reprs.cu_q[row] < #[trigger] reprs.cu_q[row + 1]
        &&& reprs.cu_k[row] < #[trigger] reprs.cu_k[row + 1]
        &&& q_len <= reprs.max_q as int
        &&& k_len <= reprs.max_k as int
        &&& q_len <= k_len
        &&& blocks_needed_for(k_len as nat) <= reprs.bt[row].len()
        &&& forall|block: int| 0 <= block < reprs.bt[row].len() ==>
            #[trigger] reprs.bt[row][block] < old_e.cs.num_blocks as nat
    } by {
        assert(crate::exec::engine::reprs_forward_layout_at(
            old_e, reprs, row,
        ));
    }
    assert(RT::paged_attention_metadata_ready(
        reprs.input_ids.len(),
        old_e.cs.num_blocks as nat,
        reprs.cu_q,
        reprs.cu_k,
        reprs.max_q,
        reprs.max_k,
        reprs.bt,
    ));

    reveal(crate::exec::engine::eng_cache_shape_ok);
    assert forall|layer: int| 0 <= layer < reprs.wr.layers.len()
        implies #[trigger] RT::paged_attention_launch_ready(
            reprs.input_ids.len(),
            old_e.kv_caches_repr@[layer].0,
            old_e.kv_caches_repr@[layer].1,
            reprs.cu_q,
            reprs.cu_k,
            reprs.max_q,
            reprs.max_k,
            reprs.bt,
        )
    by {
        reveal(RT::paged_cache_geometry);
        assert(RT::paged_cache_geometry(
            old_e.kv_caches_repr@[layer].0,
            old_e.kv_caches_repr@[layer].1,
        ));
        RT::lemma_paged_attention_launch_ready_from_parts(
            reprs.input_ids.len(),
            old_e.kv_caches_repr@[layer].0,
            old_e.kv_caches_repr@[layer].1,
            reprs.cu_q,
            reprs.cu_k,
            reprs.max_q,
            reprs.max_k,
            reprs.bt,
        );
    }
}

// Locate an arbitrary plan slot's row and apply the scheduler's cross-request
// page-disjointness export.  This is model-neutral and is shared by both cache
// discharge paths.
pub proof fn lemma_other_request_rows_miss_residency(
    old_e: Engine,
    new_e: Engine,
    reprs: StepReprs,
    rid: RequestId,
    q: int,
    block_index: int,
)
    requires
        crate::exec::engine::step_reprs_wf(old_e, reprs),
        crate::exec::engine::reprs_rows_disjoint(new_e, reprs),
        0 <= q < reprs.slots.len(),
        forall|k: int| 0 <= k < reprs.scheduled.len()
            && #[trigger] reprs.scheduled[k] == rid
            ==> !(reprs.cu_q[k] <= q < reprs.cu_q[k + 1]),
        new_e.cs.running@.contains(rid),
        new_e.cs.live_requests@.contains_key(rid),
        new_e.cs.request_residency@.contains_key(rid),
        0 <= block_index < new_e.cs.request_residency@[rid].block_ids@.len(),
        block_index < blocks_needed_for(
            (crate::exec::request_state::history(
                new_e.cs.live_requests@[rid]
            ).len() - 1) as nat,
        ),
    ensures
        reprs.slots[q] / (BLOCK_SIZE_SPEC as int)
            != new_e.cs.request_residency@[rid].block_ids@[block_index] as int,
{
    let num_rows = reprs.scheduled.len() as int;
    assert(reprs.cu_q[num_rows] == reprs.slots.len() as int);
    assert(num_rows >= 1) by {
        if num_rows == 0 {
            assert(reprs.cu_q[0] == 0);
        }
    }
    let row = crate::proof::tensor::geometry::lemma_cu_locate(
        reprs.cu_q, num_rows, q,
    );
    assert(reprs.scheduled[row] != rid);
    assert(crate::exec::engine::reprs_rows_disjoint_at(
        new_e, reprs, row,
    ));
    reveal(crate::exec::engine::reprs_rows_disjoint_at);
}

// Stable plan-table form of cross-row separation.  Unlike the post-residency
// helper above, this remains applicable when the request owning `row` finishes
// during commit and its residency is removed.
pub proof fn lemma_other_request_rows_miss_plan_row(
    old_e: Engine,
    reprs: StepReprs,
    row: int,
    q: int,
    block_index: int,
)
    requires
        crate::exec::engine::step_reprs_wf(old_e, reprs),
        crate::exec::engine::reprs_other_writes_miss_plan_rows(reprs),
        0 <= row < reprs.scheduled.len(),
        0 <= q < reprs.slots.len(),
        !(reprs.cu_q[row] <= q < reprs.cu_q[row + 1]),
        0 <= block_index < reprs.bt[row].len(),
    ensures
        reprs.slots[q] / (BLOCK_SIZE_SPEC as int)
            != reprs.bt[row][block_index] as int,
{
    let num_rows = reprs.scheduled.len() as int;
    assert(reprs.cu_q[num_rows] == reprs.slots.len() as int);
    assert(num_rows >= 1) by {
        if num_rows == 0 {
            assert(reprs.cu_q[0] == 0);
        }
    }
    let other_row = crate::proof::tensor::geometry::lemma_cu_locate(
        reprs.cu_q, num_rows, q,
    );
    assert(0 <= other_row < reprs.scheduled.len());
    assert(other_row != row);
    reveal(crate::exec::engine::reprs_other_writes_miss_plan_rows);
}


// Whole-step contract: every shared survivor satisfies the per-layer contract
// at every layer.

// Scheduler/layout facts needed to turn the architecture-dispatched
// singleton relocation theorem into post-step engine/private-cache
// coherence.  This predicate is model-opaque: it names only the common
// relocation contract, the request's post-forward key length, and the stable
// post-commit block-table view.
pub open spec fn architecture_emitted_cache_alignment(
    old_e: Engine,
    new_e: Engine,
    old_ibm: IndependentBatchModel,
    emitted: Map<RequestId, TokenId>,
    reprs: StepReprs,
) -> bool {
    let new_ibm = crate::proof::engine::abstract_step::architecture_stepped_ibm(
        old_e, new_e, old_ibm, emitted, reprs,
    );
    forall|rid: RequestId|
        #![trigger new_e.cs.live_requests@.contains_key(rid)]
        new_e.cs.live_requests@.contains_key(rid)
            && new_ibm.machines.contains_key(rid)
            && crate::exec::engine::reprs_emits(reprs, rid) ==> {
            let i = crate::proof::engine::abstract_step::architecture_scheduled_index_of(
                reprs, rid,
            );
            let k_len = crate::proof::engine::abstract_step::architecture_machine_key_len(
                reprs, rid,
            );
            let post_bt = residency_block_table(&new_e.cs, rid);
            &&& crate::proof::engine::abstract_step::architecture_machine_relocation_ready(
                old_e, old_ibm, reprs, rid,
            )
            &&& new_ibm.machines[rid].kv_tokens == k_len
            &&& forall|pos: nat|
                #![trigger block_table_slot(post_bt, pos)]
                pos < k_len ==> block_table_slot(post_bt, pos)
                    == block_table_slot(reprs.bt[i], pos)
        }
}

// The request-count and post-residency portions of the alignment bundle are
// consequences of the common scheduler/commit relation.  The caller supplies
// only the forward relocation readiness for each emitted survivor; deriving
// that physical launch/capacity fact is deliberately kept as a separate
// framework adapter.
#[verifier::spinoff_prover]
pub proof fn derive_architecture_emitted_cache_alignment_from_forward_ready(
    old_e: Engine,
    new_e: Engine,
    old_ibm: IndependentBatchModel,
    emitted: Map<RequestId, TokenId>,
    samples: Map<RequestId, (crate::exec::request_state::SamplerState, TokenId)>,
    reprs: StepReprs,
)
    requires
        crate::proof::engine::refinement::architecture_semantic_inv(&old_e, &old_ibm),
        crate::exec::engine::architecture_engine_step_relation(
            old_e, new_e, emitted, samples, reprs,
        ),
        forall|rid: RequestId|
            #![trigger new_e.cs.live_requests@.contains_key(rid)]
            new_e.cs.live_requests@.contains_key(rid)
                && crate::exec::engine::reprs_emits(reprs, rid) ==>
                crate::proof::engine::abstract_step::architecture_machine_relocation_ready(
                    old_e, old_ibm, reprs, rid,
                ),
    ensures
        architecture_emitted_cache_alignment(
            old_e, new_e, old_ibm, emitted, reprs,
        ),
{
    let new_ibm = crate::proof::engine::abstract_step::architecture_stepped_ibm(
        old_e, new_e, old_ibm, emitted, reprs,
    );
    reveal(crate::proof::engine::refinement::architecture_semantic_inv);
    reveal(crate::exec::engine::architecture_engine_step_relation);
    reveal(architecture_emitted_cache_alignment);
    assert forall|rid: RequestId|
        #![trigger new_e.cs.live_requests@.contains_key(rid)]
        new_e.cs.live_requests@.contains_key(rid)
            && new_ibm.machines.contains_key(rid)
            && crate::exec::engine::reprs_emits(reprs, rid)
        implies {
            let i = crate::proof::engine::abstract_step::architecture_scheduled_index_of(
                reprs, rid,
            );
            let k_len = crate::proof::engine::abstract_step::architecture_machine_key_len(
                reprs, rid,
            );
            let post_bt = residency_block_table(&new_e.cs, rid);
            &&& crate::proof::engine::abstract_step::architecture_machine_relocation_ready(
                old_e, old_ibm, reprs, rid,
            )
            &&& new_ibm.machines[rid].kv_tokens == k_len
            &&& forall|pos: nat|
                #![trigger block_table_slot(post_bt, pos)]
                pos < k_len ==> block_table_slot(post_bt, pos)
                    == block_table_slot(reprs.bt[i], pos)
        }
    by {
        assert(old_e.cs.live_requests@.contains_key(rid));
        assert(old_ibm.machines.contains_key(rid));
        assert(emitted.contains_key(rid));
        assert(reprs.scheduled.contains(rid));
        let i = crate::proof::engine::abstract_step::architecture_scheduled_index_of(
            reprs, rid,
        );
        assert(exists|index: int| 0 <= index < reprs.scheduled.len()
            && reprs.scheduled[index] == rid);
        assert(0 <= i < reprs.scheduled.len()
            && reprs.scheduled[i] == rid);
        let emit_i = choose|index: int|
            0 <= index < reprs.scheduled.len()
                && reprs.scheduled[index] == rid
                && reprs.sample_mask[index];
        assert(0 <= emit_i < reprs.scheduled.len()
            && reprs.scheduled[emit_i] == rid
            && reprs.sample_mask[emit_i]);
        assert(i == emit_i) by {
            assert(reprs.scheduled.no_duplicates());
        }
        assert(reprs.sample_mask[i]);
        assert(crate::exec::engine::reprs_forward_layout_at(old_e, reprs, i));
        let q_len = crate::proof::engine::abstract_step::architecture_machine_query_len(
            reprs, rid,
        );
        let k_len = crate::proof::engine::abstract_step::architecture_machine_key_len(
            reprs, rid,
        );
        let old_state = old_e.cs.live_requests@[rid];
        let new_state = new_e.cs.live_requests@[rid];
        let old_machine_state = old_ibm.machines[rid].request_state;
        assert(crate::exec::request_state::request_state_view_eq(
            old_state, old_machine_state,
        ));
        assert(crate::proof::reference::request_machine::machine_step_transition_full(
            old_state, new_state, samples[rid].0, samples[rid].1,
        ));
        assert(crate::exec::request_state::history(new_state).len()
            == crate::exec::request_state::history(old_state).len() + 1);
        assert(new_ibm.machines[rid]
            == crate::proof::engine::abstract_step::architecture_stepped_machine(
                old_e, new_e, old_ibm, emitted, reprs, rid,
            ));
        assert(new_ibm.machines[rid].kv_tokens
            == (crate::exec::request_state::history(new_state).len() - 1) as nat);
        if old_e.cs.running@.contains(rid) {
            reveal(crate::exec::engine::reprs_forward_layout_at);
            assert(k_len == crate::exec::request_state::history(old_state).len());
            assert(new_ibm.machines[rid].kv_tokens == k_len);
            assert(old_e.cs.request_residency@.contains_key(rid));
            assert(!crate::exec::engine::reprs_parks(reprs, rid)) by {
                if crate::exec::engine::reprs_parks(reprs, rid) {
                    let parked_i = choose|index: int|
                        0 <= index < reprs.scheduled.len()
                            && reprs.scheduled[index] == rid
                            && !reprs.sample_mask[index];
                    assert(0 <= parked_i < reprs.scheduled.len()
                        && reprs.scheduled[parked_i] == rid
                        && !reprs.sample_mask[parked_i]);
                    assert(parked_i == i) by {
                        assert(reprs.scheduled.no_duplicates());
                    }
                }
            }
            assert(!(emitted.contains_key(rid)
                && crate::exec::request_state::should_finish_after_append(
                    old_state, emitted[rid],
                )));
            assert(new_e.cs.running@.contains(rid));
            assert(new_e.cs.request_residency@.contains_key(rid));
            let old_bt = old_e.cs.request_residency@[rid].block_ids@;
            let post_bt = new_e.cs.request_residency@[rid].block_ids@;
            assert(reprs.bt[i] == old_bt);
            assert(post_bt.len() >= old_bt.len());
            assert(post_bt.subrange(0, old_bt.len() as int) == old_bt);
            assert forall|pos: nat|
                #![trigger block_table_slot(post_bt, pos)]
                pos < k_len implies block_table_slot(post_bt, pos)
                    == block_table_slot(reprs.bt[i], pos)
            by {
                crate::proof::tensor::geometry::lemma_blocks_needed_covers_pos(pos, k_len);
                let page = (pos / crate::types::BLOCK_SIZE_SPEC) as int;
                assert(0 <= page < old_bt.len());
                assert(post_bt[page]
                    == post_bt.subrange(0, old_bt.len() as int)[page]);
                assert(post_bt[page] == old_bt[page]);
            }
        } else {
            assert(old_e.cs.waiting@.contains(rid)) by {
                assert(reprs.scheduled.contains(rid));
            }
            assert(old_e.cs.live_requests@[rid].generated_tokens@.len() == 0) by {
                assert(crate::exec::cache_scheduler::waiting_unstarted(&old_e.cs));
            }
            assert(crate::exec::engine::reprs_sample_policy(old_e, reprs));
            crate::exec::engine::lemma_waiting_sample_row_is_final(
                old_e, reprs, i,
            );
            assert(k_len == old_e.cs.live_requests@[rid].prompt_tokens@.len());
            assert(crate::exec::request_state::history(old_state).len()
                == old_e.cs.live_requests@[rid].prompt_tokens@.len());
            assert(new_ibm.machines[rid].kv_tokens == k_len);
            assert(crate::exec::engine::reprs_prefill_residencies(
                old_e, new_e, reprs,
            ));
            reveal(crate::exec::engine::reprs_prefill_residencies);
            assert(crate::exec::engine::reprs_prefill_residency_at(
                old_e, new_e, reprs, i,
            ));
            reveal(crate::exec::engine::reprs_prefill_residency_at);
            assert(new_e.cs.request_residency@.contains_key(rid));
            let post_bt = new_e.cs.request_residency@[rid].block_ids@;
            assert(reprs.bt[i].len() <= post_bt.len());
            assert(post_bt.subrange(0, reprs.bt[i].len() as int)
                == reprs.bt[i]);
            assert forall|pos: nat|
                #![trigger block_table_slot(post_bt, pos)]
                pos < k_len implies block_table_slot(post_bt, pos)
                    == block_table_slot(reprs.bt[i], pos)
            by {
                crate::proof::tensor::geometry::lemma_blocks_needed_covers_pos(pos, k_len);
                let page = (pos / crate::types::BLOCK_SIZE_SPEC) as int;
                assert(0 <= page < reprs.bt[i].len());
                assert(post_bt[page]
                    == post_bt.subrange(0, reprs.bt[i].len() as int)[page]);
                assert(post_bt[page] == reprs.bt[i][page]);
            }
        }
    }
}

// Discharge the physical singleton-relocation contract from common scheduler
// row geometry and cache shapes.  The only forward-semantic input is the
// architecture-neutral request-projection domain; all remaining obligations
// are address arithmetic or definitional properties of the copied private
// prefix.
#[verifier::spinoff_prover]
#[verifier::rlimit(300)]
pub proof fn derive_architecture_forward_relocation_ready_from_projection(
    old_e: Engine,
    old_ibm: IndependentBatchModel,
    reprs: StepReprs,
    rid: RequestId,
)
    requires
        RT::paged_attention_numeric_domain(),
        crate::proof::engine::refinement::architecture_semantic_inv(&old_e, &old_ibm),
        crate::exec::engine::eng_cache_shape_ok(&old_e),
        crate::proof::engine::refinement::mach_cache_shape_ok(
            &old_ibm, old_e.cs.num_blocks as nat,
        ),
        old_e.model_config.num_layers > 0,
        crate::exec::engine::step_reprs_wf(old_e, reprs),
        crate::exec::engine::step_reprs_block_tables_bounded(old_e, reprs),
        crate::exec::engine::reprs_forward_layout_ok(old_e, reprs),
        reprs.scheduled.contains(rid),
        crate::proof::model::architecture::model_forward_request_projection_domain(
            reprs.wr,
            old_ibm.architecture_repr,
            reprs.input_ids,
            reprs.positions,
            old_e.kv_caches_repr@,
            reprs.slots,
            reprs.cu_q,
            reprs.cu_k,
            reprs.max_q,
            reprs.max_k,
            reprs.bt,
            crate::proof::engine::abstract_step::architecture_scheduled_index_of(
                reprs, rid,
            ) as nat,
        ),
    ensures
        crate::proof::engine::abstract_step::architecture_machine_relocation_ready(
            old_e, old_ibm, reprs, rid,
        ),
{
    reveal(crate::proof::engine::refinement::architecture_semantic_inv);
    reveal(crate::proof::engine::abstract_step::architecture_machine_relocation_ready);
    reveal(crate::proof::model::relocation::model_forward_engine_to_machine_ready);
    reveal(crate::proof::model::relocation::model_forward_singleton_relocation_ready);
    let row = crate::proof::engine::abstract_step::architecture_scheduled_index_of(
        reprs, rid,
    );
    assert(exists|i: int| 0 <= i < reprs.scheduled.len()
        && reprs.scheduled[i] == rid);
    assert(0 <= row < reprs.scheduled.len()
        && reprs.scheduled[row] == rid);
    assert(crate::exec::engine::reprs_forward_layout_at(old_e, reprs, row));
    let lo = reprs.cu_q[row];
    let hi = reprs.cu_q[row + 1];
    let q_len = crate::proof::engine::abstract_step::architecture_machine_query_len(
        reprs, rid,
    );
    let k_len = crate::proof::engine::abstract_step::architecture_machine_key_len(
        reprs, rid,
    );
    let prefix_len = crate::proof::engine::abstract_step::architecture_machine_prefix_len(
        reprs, rid,
    );
    let request_slots = reprs.slots.subrange(lo, hi);
    let machine_slots = slots_from(prefix_len, q_len);
    let machine_bt = crate::proof::reference::request_machine::singleton_block_rows(k_len);
    let machine_base = crate::proof::engine::abstract_step::architecture_machine_cache_base(
        old_e, old_ibm, reprs, rid,
    );
    reveal(crate::exec::engine::step_reprs_wf);
    assert(0 <= lo < hi <= reprs.input_ids.len() as int);
    assert(q_len == (hi - lo) as nat);
    assert(hi - lo == q_len as int);
    assert(reprs.cu_k[row + 1] - reprs.cu_k[row]
        == k_len as int);
    assert(q_len > 0);
    assert(q_len <= k_len);
    assert(prefix_len == k_len - q_len);
    assert(reprs.input_ids.subrange(lo, hi).len() == q_len);
    assert(reprs.positions.subrange(lo, hi).len() == q_len);
    assert(request_slots.len() == q_len);
    assert(machine_slots.len() == q_len);
    assert(old_e.kv_caches_repr@.len() >= reprs.wr.layers.len());
    assert(machine_base.len() == reprs.wr.layers.len());
    assert(crate::proof::tensor::geometry::blocks_needed_for(k_len) <= reprs.bt[row].len()) by {
        reveal(crate::exec::engine::reprs_forward_layout_at);
    }
    assert(machine_bt.len() == 1);
    assert(machine_bt[0].len() == crate::proof::tensor::geometry::blocks_needed_for(k_len));
    assert(crate::proof::tensor::geometry::blocks_needed_for(k_len)
        <= old_e.cs.num_blocks as nat) by {
        assert(crate::exec::engine::step_reprs_block_tables_bounded(old_e, reprs));
    }
    assert(crate::proof::tensor::geometry::blocks_needed_for(k_len) <= u64::MAX as nat);
    crate::proof::reference::independent_batch_model::lemma_ibm_valid_semantic_model(
        old_ibm,
    );
    assert(reprs.wr == old_ibm.wr);
    assert(crate::proof::model::types::model_weights_architecture_repr_valid(
        reprs.wr, old_ibm.architecture_repr,
    ));
    assert(reprs.wr.layers.len()
        == old_e.model_config.num_layers as nat);
    assert(reprs.wr.layers.len() > 0);
    crate::proof::model::architecture::lemma_cache_refinement_support_implies_projection_configuration_ready(
        crate::proof::reference::independent_batch_model::ibm_semantic_model(old_ibm),
    );
    assert forall|j: int| 0 <= j < q_len as int implies {
        let logical_pos = (prefix_len + j as nat) as nat;
        &&& block_table_slot(reprs.bt[row], logical_pos)
            == #[trigger] request_slots[j] as nat
        &&& block_table_slot(machine_bt[0], logical_pos)
            == machine_slots[j] as nat
        &&& request_slots[j] >= 0
        &&& machine_slots[j] >= 0
    } by {
        let query_index = lo + j;
        let logical_pos = (prefix_len + j as nat) as nat;
        assert(lo <= query_index < hi);
        crate::exec::engine::lemma_reprs_forward_layout_slot_at(
            old_e, reprs, row, query_index,
        );
        let layout_pos = reprs.cu_k[row + 1] - reprs.cu_k[row]
            - (hi - lo) + query_index - lo;
        assert(layout_pos == prefix_len as int + j);
        assert(logical_pos as int == layout_pos);
        assert(request_slots[j] == reprs.slots[query_index]);
        assert(logical_pos as int == prefix_len as int + j);
        assert(request_slots[j] as nat
            == block_table_slot(reprs.bt[row], logical_pos));
        assert(machine_slots[j] == prefix_len as int + j);
        assert(machine_slots[j] >= 0);
        crate::proof::tensor::geometry::lemma_blocks_needed_covers_pos(
            logical_pos, k_len,
        );
        crate::proof::reference::request_machine::lemma_contiguous_block_table_slot(
            crate::proof::tensor::geometry::blocks_needed_for(k_len), logical_pos,
        );
        assert(machine_bt[0]
            == crate::proof::reference::request_machine::contiguous_block_ids(
                crate::proof::tensor::geometry::blocks_needed_for(k_len),
            ));
    }
    assert forall|j: int, m: int|
        #![trigger request_slots[m], request_slots[j]]
        0 <= j < q_len as int && j < m < q_len as int implies
            request_slots[m] != request_slots[j]
    by {
        let left = (prefix_len + j as nat) as nat;
        let right = (prefix_len + m as nat) as nat;
        assert(left < k_len && right < k_len);
        crate::proof::tensor::geometry::lemma_blocks_needed_covers_pos(left, k_len);
        crate::proof::tensor::geometry::lemma_blocks_needed_covers_pos(right, k_len);
        reveal(crate::exec::engine::reprs_forward_layout_at);
        crate::proof::model::family_layout::lemma_block_table_slot_injective(
            reprs.bt[row], left, right,
        );
    }
    assert forall|j: int, m: int|
        #![trigger machine_slots[m], machine_slots[j]]
        0 <= j < q_len as int && j < m < q_len as int implies
            machine_slots[m] != machine_slots[j]
    by {
        assert(machine_slots[j] == prefix_len as int + j);
        assert(machine_slots[m] == prefix_len as int + m);
    }
    assert(crate::proof::model::family_layout::fresh_writes_miss_cached_prefix(
        request_slots,
        reprs.bt[row],
        machine_slots,
        machine_bt[0],
        prefix_len as int,
    )) by {
        reveal(crate::proof::model::family_layout::fresh_writes_miss_cached_prefix);
        assert forall|pos: nat|
            #![trigger block_table_slot(reprs.bt[row], pos)]
            pos < prefix_len implies
                !request_slots.contains(
                    block_table_slot(reprs.bt[row], pos) as int,
                )
                && !machine_slots.contains(
                    block_table_slot(machine_bt[0], pos) as int,
                )
        by {
            crate::proof::tensor::geometry::lemma_blocks_needed_covers_pos(pos, k_len);
            crate::proof::reference::request_machine::lemma_contiguous_block_table_slot(
                crate::proof::tensor::geometry::blocks_needed_for(k_len), pos,
            );
            if request_slots.contains(
                block_table_slot(reprs.bt[row], pos) as int,
            ) {
                let j = request_slots.index_of(
                    block_table_slot(reprs.bt[row], pos) as int,
                );
                let written = (prefix_len + j as nat) as nat;
                assert(0 <= j < q_len as int);
                assert(written < k_len);
                crate::proof::tensor::geometry::lemma_blocks_needed_covers_pos(written, k_len);
                reveal(crate::exec::engine::reprs_forward_layout_at);
                crate::proof::model::family_layout::lemma_block_table_slot_injective(
                    reprs.bt[row], pos, written,
                );
            }
            if machine_slots.contains(pos as int) {
                let j = machine_slots.index_of(pos as int);
                assert(0 <= j < q_len as int);
                assert(machine_slots[j] == prefix_len as int + j);
                assert(pos as int >= prefix_len as int);
            }
        }
    }
    assert forall|layer: int, pos: nat|
        #![trigger old_e.kv_caches_repr@[layer].0,
            block_table_slot(reprs.bt[row], pos)]
        0 <= layer < reprs.wr.layers.len() && pos < k_len implies {
            let engine_slot = block_table_slot(reprs.bt[row], pos);
            &&& slot_in_cache(old_e.kv_caches_repr@[layer].0, engine_slot)
            &&& slot_in_cache(old_e.kv_caches_repr@[layer].1, engine_slot)
        }
    by {
        lemma_plan_slot_in_pre_cache(old_e, reprs, row, layer, pos);
    }
    assert forall|layer: int, pos: nat|
        #![trigger slot_in_cache(
            old_ibm.machines[rid].kv_cache_reprs[layer].0, pos,
        )]
        0 <= layer < reprs.wr.layers.len() && pos < k_len implies
            slot_in_cache(
                old_ibm.machines[rid].kv_cache_reprs[layer].0, pos,
            )
            && slot_in_cache(
                old_ibm.machines[rid].kv_cache_reprs[layer].1, pos,
            )
    by {
        assert(old_ibm.machines.contains_key(rid));
        assert(old_ibm.machines[rid].kv_cache_reprs.len()
            == old_ibm.model_config.num_layers as nat);
        assert(old_ibm.machines[rid].kv_cache_reprs[layer].0.len()
            == old_e.cs.num_blocks as nat);
        assert(old_ibm.machines[rid].kv_cache_reprs[layer].1.len()
            == old_e.cs.num_blocks as nat);
        crate::proof::tensor::geometry::lemma_blocks_needed_covers_pos(pos, k_len);
        let page = (pos as int) / (BLOCK_SIZE_SPEC as int);
        let offset = (pos as int) % (BLOCK_SIZE_SPEC as int);
        vstd::arithmetic::div_mod::lemma_mod_bound(
            pos as int, BLOCK_SIZE_SPEC as int,
        );
        assert(0 <= page < old_e.cs.num_blocks as int);
        assert(old_ibm.machines[rid].kv_cache_reprs[layer].0[page].len()
            == BLOCK_SIZE_SPEC as int);
        assert(old_ibm.machines[rid].kv_cache_reprs[layer].1[page].len()
            == BLOCK_SIZE_SPEC as int);
    }
    assert forall|layer: int, j: int|
        #![trigger old_e.kv_caches_repr@[layer].0, request_slots[j]]
        0 <= layer < reprs.wr.layers.len() && 0 <= j < q_len as int implies
            slot_in_cache(
                old_e.kv_caches_repr@[layer].0, request_slots[j] as nat,
            )
            && slot_in_cache(
                old_e.kv_caches_repr@[layer].1, request_slots[j] as nat,
            )
            && slot_in_cache(machine_base[layer].0, machine_slots[j] as nat)
            && slot_in_cache(machine_base[layer].1, machine_slots[j] as nat)
    by {
        let pos = (prefix_len + j as nat) as nat;
        assert(pos < k_len);
        assert(request_slots[j] as nat
            == block_table_slot(reprs.bt[row], pos));
        assert(machine_slots[j] as nat == pos);
        assert(slot_in_cache(
            old_ibm.machines[rid].kv_cache_reprs[layer].0, pos,
        ));
        assert(slot_in_cache(
            old_ibm.machines[rid].kv_cache_reprs[layer].1, pos,
        ));
        crate::proof::model::family_layout::lemma_relocated_prefix_base_at(
            old_ibm.machines[rid].kv_cache_reprs[layer].0,
            old_e.kv_caches_repr@[layer].0,
            reprs.bt[row],
            prefix_len,
            pos,
        );
        crate::proof::model::family_layout::lemma_relocated_prefix_base_at(
            old_ibm.machines[rid].kv_cache_reprs[layer].1,
            old_e.kv_caches_repr@[layer].1,
            reprs.bt[row],
            prefix_len,
            pos,
        );
    }
    assert forall|layer: int, pos: nat|
        #![trigger old_e.kv_caches_repr@[layer].0,
            block_table_slot(reprs.bt[row], pos)]
        0 <= layer < reprs.wr.layers.len() && pos < prefix_len implies {
            let engine_slot = block_table_slot(reprs.bt[row], pos);
            let machine_slot = block_table_slot(machine_bt[0], pos);
            &&& slot_in_cache(old_e.kv_caches_repr@[layer].0, engine_slot)
            &&& slot_in_cache(old_e.kv_caches_repr@[layer].1, engine_slot)
            &&& slot_in_cache(machine_base[layer].0, machine_slot)
            &&& slot_in_cache(machine_base[layer].1, machine_slot)
            &&& cache_at(old_e.kv_caches_repr@[layer].0, engine_slot)
                == cache_at(machine_base[layer].0, machine_slot)
            &&& cache_at(old_e.kv_caches_repr@[layer].1, engine_slot)
                == cache_at(machine_base[layer].1, machine_slot)
        }
    by {
        crate::proof::tensor::geometry::lemma_blocks_needed_covers_pos(pos, k_len);
        crate::proof::reference::request_machine::lemma_contiguous_block_table_slot(
            crate::proof::tensor::geometry::blocks_needed_for(k_len), pos,
        );
        assert(slot_in_cache(
            old_ibm.machines[rid].kv_cache_reprs[layer].0, pos,
        ));
        assert(slot_in_cache(
            old_ibm.machines[rid].kv_cache_reprs[layer].1, pos,
        ));
        crate::proof::model::family_layout::lemma_relocated_prefix_base_at(
            old_ibm.machines[rid].kv_cache_reprs[layer].0,
            old_e.kv_caches_repr@[layer].0,
            reprs.bt[row],
            prefix_len,
            pos,
        );
        crate::proof::model::family_layout::lemma_relocated_prefix_base_at(
            old_ibm.machines[rid].kv_cache_reprs[layer].1,
            old_e.kv_caches_repr@[layer].1,
            reprs.bt[row],
            prefix_len,
            pos,
        );
    }
}

// Assemble the family-neutral projection domain from scheduler metadata,
// plan-row separation, and engine cache shape.  Family adapters consume this
// one predicate symmetrically; no family theorem is imported here.
#[verifier::spinoff_prover]
#[verifier::rlimit(200)]
pub proof fn derive_architecture_request_projection_common_domain(
    old_e: Engine,
    old_ibm: IndependentBatchModel,
    reprs: StepReprs,
    rid: RequestId,
)
    requires
        RT::paged_attention_numeric_domain(),
        crate::proof::engine::refinement::architecture_semantic_inv(&old_e, &old_ibm),
        crate::exec::engine::eng_cache_shape_ok(&old_e),
        old_e.model_config.num_layers > 0,
        crate::exec::engine::step_reprs_wf(old_e, reprs),
        crate::exec::engine::reprs_forward_layout_ok(old_e, reprs),
        crate::exec::engine::reprs_other_writes_miss_plan_rows(reprs),
        reprs.scheduled.contains(rid),
    ensures
        crate::proof::model::family_layout::request_projection_common_domain(
            reprs.wr,
            old_ibm.architecture_repr,
            reprs.input_ids,
            reprs.positions,
            old_e.kv_caches_repr@,
            reprs.slots,
            reprs.cu_q,
            reprs.cu_k,
            reprs.max_q,
            reprs.max_k,
            reprs.bt,
            crate::proof::engine::abstract_step::architecture_scheduled_index_of(
                reprs, rid,
            ) as nat,
        ),
{
    let row = crate::proof::engine::abstract_step::architecture_scheduled_index_of(
        reprs, rid,
    );
    assert(exists|i: int| 0 <= i < reprs.scheduled.len()
        && reprs.scheduled[i] == rid);
    assert(0 <= row < reprs.scheduled.len()
        && reprs.scheduled[row] == rid);
    assert(crate::exec::engine::reprs_forward_layout_at(old_e, reprs, row));
    reveal(crate::proof::engine::refinement::architecture_semantic_inv);
    reveal(crate::exec::engine::step_reprs_wf);
    reveal(crate::exec::engine::reprs_forward_layout_at);
    reveal(crate::proof::model::family_layout::request_projection_common_domain);
    crate::proof::reference::independent_batch_model::lemma_ibm_valid_semantic_model(
        old_ibm,
    );
    assert(reprs.wr == old_ibm.wr);
    assert(old_e.kv_caches_repr@.len() >= reprs.wr.layers.len());
    assert(reprs.wr.layers.len()
        == old_e.model_config.num_layers as nat);
    assert(reprs.wr.layers.len() > 0);
    crate::proof::model::architecture::lemma_cache_refinement_support_implies_projection_configuration_ready(
        crate::proof::reference::independent_batch_model::ibm_semantic_model(old_ibm),
    );
    let lo = reprs.cu_q[row];
    let hi = reprs.cu_q[row + 1];
    let k_len = (reprs.cu_k[row + 1] - reprs.cu_k[row]) as nat;
    crate::proof::tensor::geometry::lemma_cu_mono(
        reprs.cu_q, reprs.scheduled.len() as int, 0, row,
    );
    crate::proof::tensor::geometry::lemma_cu_mono(
        reprs.cu_k, reprs.scheduled.len() as int, 0, row,
    );
    assert(0 <= lo < hi <= reprs.input_ids.len() as int);
    assert(0 <= reprs.cu_k[row] < reprs.cu_k[row + 1]);
    assert(hi - lo <= reprs.max_q as int);
    assert(reprs.cu_k[row + 1] - reprs.cu_k[row]
        <= reprs.max_k as int);
    assert(hi - lo
        <= reprs.cu_k[row + 1] - reprs.cu_k[row]);
    assert(reprs.max_q > 0);
    assert(blocks_needed_for(k_len) <= reprs.bt[row].len());
    assert forall|m: int, block: int|
        #![trigger reprs.slots.subrange(0, lo)[m]
            / (BLOCK_SIZE_SPEC as int), reprs.bt[row][block]]
        0 <= m < lo && 0 <= block < reprs.bt[row].len() implies
            reprs.slots.subrange(0, lo)[m]
                / (BLOCK_SIZE_SPEC as int)
                != reprs.bt[row][block] as int
    by {
        assert(reprs.slots.subrange(0, lo)[m] == reprs.slots[m]);
        lemma_other_request_rows_miss_plan_row(
            old_e, reprs, row, m, block,
        );
    }
    assert forall|m: int, block: int|
        #![trigger reprs.slots.subrange(
            hi, reprs.slots.len() as int,
        )[m] / (BLOCK_SIZE_SPEC as int), reprs.bt[row][block]]
        0 <= m < reprs.slots.len() - hi
            && 0 <= block < reprs.bt[row].len() implies
            reprs.slots.subrange(
                hi, reprs.slots.len() as int,
            )[m] / (BLOCK_SIZE_SPEC as int)
                != reprs.bt[row][block] as int
    by {
        let q = hi + m;
        assert(reprs.slots.subrange(
            hi, reprs.slots.len() as int,
        )[m] == reprs.slots[q]);
        lemma_other_request_rows_miss_plan_row(
            old_e, reprs, row, q, block,
        );
    }
    assert forall|layer: int, pos: nat|
        #![trigger old_e.kv_caches_repr@[layer].0,
            block_table_slot(reprs.bt[row], pos)]
        0 <= layer < reprs.wr.layers.len()
            && pos < blocks_needed_for(k_len) * BLOCK_SIZE_SPEC implies
            slot_in_cache(
                old_e.kv_caches_repr@[layer].0,
                block_table_slot(reprs.bt[row], pos),
            )
            && slot_in_cache(
                old_e.kv_caches_repr@[layer].1,
                block_table_slot(reprs.bt[row], pos),
            )
    by {
        lemma_plan_page_slot_in_pre_cache(
            old_e, reprs, row, layer, pos,
        );
    }
    assert forall|layer: int, pos: nat|
        #![trigger old_e.kv_caches_repr@[layer].0,
            block_table_slot(reprs.bt[row], pos)]
        0 <= layer < reprs.wr.layers.len() && pos < k_len implies
            slot_in_cache(
                old_e.kv_caches_repr@[layer].0,
                block_table_slot(reprs.bt[row], pos),
            )
            && slot_in_cache(
                old_e.kv_caches_repr@[layer].1,
                block_table_slot(reprs.bt[row], pos),
            )
    by {
        lemma_plan_slot_in_pre_cache(
            old_e, reprs, row, layer, pos,
        );
    }
    lemma_reprs_paged_attention_launch_ready(old_e, reprs);
}

pub proof fn derive_architecture_emitted_forward_ready(
    old_e: Engine,
    new_e: Engine,
    old_ibm: IndependentBatchModel,
    emitted: Map<RequestId, TokenId>,
    samples: Map<RequestId, (crate::exec::request_state::SamplerState, TokenId)>,
    reprs: StepReprs,
)
    requires
        RT::paged_attention_numeric_domain(),
        crate::proof::engine::refinement::architecture_semantic_inv(&old_e, &old_ibm),
        crate::exec::engine::eng_cache_shape_ok(&old_e),
        crate::proof::engine::refinement::mach_cache_shape_ok(
            &old_ibm, old_e.cs.num_blocks as nat,
        ),
        old_e.model_config.num_layers > 0,
        crate::exec::engine::architecture_engine_step_relation(
            old_e, new_e, emitted, samples, reprs,
        ),
    ensures
        forall|rid: RequestId|
            #![trigger new_e.cs.live_requests@.contains_key(rid)]
            new_e.cs.live_requests@.contains_key(rid)
                && crate::exec::engine::reprs_emits(reprs, rid) ==>
                crate::proof::engine::abstract_step::architecture_machine_relocation_ready(
                    old_e, old_ibm, reprs, rid,
                ),
{
    reveal(crate::exec::engine::architecture_engine_step_relation);
    assert forall|rid: RequestId|
        #![trigger new_e.cs.live_requests@.contains_key(rid)]
        new_e.cs.live_requests@.contains_key(rid)
            && crate::exec::engine::reprs_emits(reprs, rid)
        implies crate::proof::engine::abstract_step::architecture_machine_relocation_ready(
            old_e, old_ibm, reprs, rid,
        )
    by {
        assert(reprs.scheduled.contains(rid));
        derive_architecture_request_projection_common_domain(
            old_e, old_ibm, reprs, rid,
        );
        let row = crate::proof::engine::abstract_step::architecture_scheduled_index_of(
            reprs, rid,
        );
        crate::proof::model::architecture::lemma_model_forward_request_projection_domain_from_common(
            reprs.wr,
            old_ibm.architecture_repr,
            reprs.input_ids,
            reprs.positions,
            old_e.kv_caches_repr@,
            reprs.slots,
            reprs.cu_q,
            reprs.cu_k,
            reprs.max_q,
            reprs.max_k,
            reprs.bt,
            row as nat,
        );
        derive_architecture_forward_relocation_ready_from_projection(
            old_e, old_ibm, reprs, rid,
        );
    }
}

// Complete architecture-neutral post-step coherence proof.  This composes
// the shared scheduler projection domain, physical relocation, post-commit
// row alignment, and unwritten-slot frame; it is independent of the selected
// full-attention or full-cache sliding-window layer composition.
pub proof fn derive_architecture_engine_kv_coherent(
    old_e: Engine,
    new_e: Engine,
    old_ibm: IndependentBatchModel,
    emitted: Map<RequestId, TokenId>,
    samples: Map<RequestId, (crate::exec::request_state::SamplerState, TokenId)>,
    reprs: StepReprs,
)
    requires
        RT::paged_attention_numeric_domain(),
        crate::proof::engine::refinement::architecture_semantic_inv(&old_e, &old_ibm),
        crate::proof::engine::refinement::phase_aligned(&old_e, &old_ibm),
        crate::exec::engine::eng_cache_shape_ok(&old_e),
        crate::proof::engine::refinement::mach_cache_shape_ok(
            &old_ibm, old_e.cs.num_blocks as nat,
        ),
        crate::exec::cache_scheduler::residency_history_aligned(&new_e.cs),
        old_e.model_config.num_layers > 0,
        crate::exec::engine::architecture_engine_step_relation(
            old_e, new_e, emitted, samples, reprs,
        ),
        crate::exec::engine::engine_step_semantic_identity(old_e, new_e),
    ensures
        engine_kv_coherent(
            &new_e,
            &crate::proof::engine::abstract_step::architecture_stepped_ibm(
                old_e, new_e, old_ibm, emitted, reprs,
            ),
        ),
{
    derive_architecture_emitted_forward_ready(
        old_e, new_e, old_ibm, emitted, samples, reprs,
    );
    derive_architecture_emitted_cache_alignment_from_forward_ready(
        old_e, new_e, old_ibm, emitted, samples, reprs,
    );
    derive_architecture_unscheduled_no_touch(
        old_e, new_e, old_ibm, emitted, samples, reprs,
    );
    discharge_architecture_engine_kv_coherent(
        old_e, new_e, old_ibm, emitted, samples, reprs,
    );
}

// Derive the unemitted-survivor frame directly from the common per-layer
// scatter witness.  The request is unchanged, its residency is unchanged, and
// cross-request page separation keeps every plan slot away from its cached
// logical prefix.  The final cache-frame step dispatches only through
// `model_architecture`.
#[verifier::spinoff_prover]
#[verifier::rlimit(200)]
pub proof fn derive_architecture_unscheduled_no_touch(
    old_e: Engine,
    new_e: Engine,
    old_ibm: IndependentBatchModel,
    emitted: Map<RequestId, TokenId>,
    samples: Map<RequestId, (crate::exec::request_state::SamplerState, TokenId)>,
    reprs: StepReprs,
)
    requires
        crate::proof::engine::refinement::architecture_semantic_inv(&old_e, &old_ibm),
        crate::proof::engine::refinement::phase_aligned(&old_e, &old_ibm),
        crate::exec::engine::eng_cache_shape_ok(&old_e),
        crate::exec::cache_scheduler::residency_history_aligned(&new_e.cs),
        crate::exec::engine::architecture_engine_step_relation(
            old_e, new_e, emitted, samples, reprs,
        ),
    ensures
        unscheduled_no_touch(
            old_e,
            new_e,
            old_ibm,
            crate::proof::engine::abstract_step::architecture_stepped_ibm(
                old_e, new_e, old_ibm, emitted, reprs,
            ),
            reprs,
            new_e.model_config.num_layers as nat,
        ),
{
    let new_ibm = crate::proof::engine::abstract_step::architecture_stepped_ibm(
        old_e, new_e, old_ibm, emitted, reprs,
    );
    let num_layers = new_e.model_config.num_layers as nat;
    reveal(crate::proof::engine::refinement::architecture_semantic_inv);
    reveal(crate::exec::engine::architecture_engine_step_relation);
    reveal(unscheduled_no_touch);
    crate::proof::reference::independent_batch_model::lemma_ibm_valid_semantic_model(
        old_ibm,
    );
    assert(reprs.wr == old_ibm.wr);
    assert(crate::proof::model::types::model_weights_architecture_repr_valid(
        reprs.wr, old_ibm.architecture_repr,
    ));
    assert forall|rid: RequestId, layer: int, pos: nat|
        #![trigger new_e.kv_caches_repr@[layer],
            block_table_slot(residency_block_table(&new_e.cs, rid), pos)]
        new_e.cs.live_requests@.contains_key(rid)
            && new_ibm.machines.contains_key(rid)
            && !crate::exec::engine::reprs_emits(reprs, rid)
            && 0 <= layer < num_layers as int
            && pos < new_ibm.machines[rid].kv_tokens
        implies {
            let bt = residency_block_table(&new_e.cs, rid);
            let slot = block_table_slot(bt, pos);
            &&& new_ibm.machines[rid] == old_ibm.machines[rid]
            &&& slot_in_cache(new_e.kv_caches_repr@[layer].0, slot)
            &&& slot_in_cache(new_e.kv_caches_repr@[layer].1, slot)
            &&& cache_at(new_e.kv_caches_repr@[layer].0, slot)
                == cache_at(
                    old_ibm.machines[rid].kv_cache_reprs[layer].0, pos,
                )
            &&& cache_at(new_e.kv_caches_repr@[layer].1, slot)
                == cache_at(
                    old_ibm.machines[rid].kv_cache_reprs[layer].1, pos,
                )
        }
    by {
        assert(old_e.cs.live_requests@.contains_key(rid));
        assert(old_ibm.machines.contains_key(rid));
        assert(!emitted.contains_key(rid));
        assert(new_ibm.machines[rid]
            == crate::proof::engine::abstract_step::architecture_stepped_machine(
                old_e, new_e, old_ibm, emitted, reprs, rid,
            ));
        assert(new_ibm.machines[rid] == old_ibm.machines[rid]);
        assert(pos < old_ibm.machines[rid].kv_tokens);
        assert(crate::proof::reference::request_machine::request_machine_alive(
            old_ibm.machines[rid], old_ibm.model_config,
        ));
        assert(old_ibm.machines[rid].kv_initialized) by {
            if !old_ibm.machines[rid].kv_initialized {
                assert(old_ibm.machines[rid].kv_tokens == 0);
            }
        }
        assert(old_e.cs.running@.contains(rid));
        assert(!reprs.scheduled.contains(rid)) by {
            if reprs.scheduled.contains(rid) {
                let row = choose|i: int| 0 <= i < reprs.scheduled.len()
                    && reprs.scheduled[i] == rid;
                assert(0 <= row < reprs.scheduled.len()
                    && reprs.scheduled[row] == rid);
                crate::exec::engine::lemma_old_running_row_emits(
                    old_e, reprs, row,
                );
            }
        }
        assert(!crate::exec::engine::reprs_parks(reprs, rid)) by {
            if crate::exec::engine::reprs_parks(reprs, rid) {
                assert(reprs.scheduled.contains(rid));
            }
        }
        assert(new_e.cs.running@.contains(rid));
        assert(new_e.cs.live_requests@[rid]
            == old_e.cs.live_requests@[rid]);
        assert(new_e.cs.request_residency@.contains_key(rid));
        assert(new_e.cs.request_residency@[rid]
            == old_e.cs.request_residency@[rid]);
        let bt = residency_block_table(&new_e.cs, rid);
        let block_size = BLOCK_SIZE_SPEC as int;
        let block_index = (pos as int) / block_size;
        let new_state = new_e.cs.live_requests@[rid];
        let machine_state = old_ibm.machines[rid].request_state;
        assert(crate::exec::request_state::request_state_view_eq(
            old_e.cs.live_requests@[rid], machine_state,
        ));
        assert(crate::exec::request_state::history(new_state).len()
            == crate::exec::request_state::history(machine_state).len());
        assert(old_ibm.machines[rid].kv_tokens + 1
            == crate::exec::request_state::history(machine_state).len());
        let ids = new_e.cs.request_residency@[rid].block_ids@;
        let tail = new_e.cs.blocks@[ids[ids.len() - 1]].tokens@.len() as int;
        assert(crate::exec::request_state::history(new_state).len() as int
            == (ids.len() - 1) * block_size + tail);
        assert(tail >= 1);
        assert(tail <= block_size) by {
            assert(crate::exec::cache_scheduler::block_token_bound(&new_e.cs));
            assert(new_e.cs.blocks@.contains_key(ids[ids.len() - 1]));
        }
        crate::exec::cache_scheduler::lemma_aligned_blocks_needed(
            crate::exec::request_state::history(new_state).len() as int,
            ids.len() as int,
            tail,
        );
        crate::proof::tensor::geometry::lemma_blocks_needed_covers_pos(
            pos, old_ibm.machines[rid].kv_tokens,
        );
        assert(0 <= block_index < bt.len());
        let slot = block_table_slot(bt, pos);
        assert(!reprs.slots.contains(slot as int)) by {
            if reprs.slots.contains(slot as int) {
                let q = choose|q: int| 0 <= q < reprs.slots.len()
                    && reprs.slots[q] == slot as int;
                assert forall|row: int| 0 <= row < reprs.scheduled.len()
                    && #[trigger] reprs.scheduled[row] == rid implies false
                by {
                    assert(reprs.scheduled.contains(rid));
                }
                assert(block_index < blocks_needed_for(
                    (crate::exec::request_state::history(new_state).len() - 1) as nat,
                ));
                lemma_other_request_rows_miss_residency(
                    old_e, new_e, reprs, rid, q, block_index,
                );
                crate::proof::tensor::geometry::block_table_slot_block(bt, pos);
                assert(reprs.slots[q] / block_size == bt[block_index] as int);
            }
        }
        assert(cache_at(old_e.kv_caches_repr@[layer].0, slot)
            == cache_at(
                old_ibm.machines[rid].kv_cache_reprs[layer].0, pos,
            )
            && cache_at(old_e.kv_caches_repr@[layer].1, slot)
                == cache_at(
                    old_ibm.machines[rid].kv_cache_reprs[layer].1, pos,
                )) by {
            assert(bt == residency_block_table(&old_e.cs, rid));
            assert(crate::proof::engine::refinement::engine_kv_coherent(&old_e, &old_ibm));
        }
        assert(0 <= layer < old_e.kv_caches_repr@.len());
        lemma_residency_slot_in_pre_cache(
            old_e, new_e, rid, layer, pos,
        );
        assert(0 <= layer < reprs.wr.layers.len());
        crate::proof::model::architecture::lemma_model_forward_kv_preserves_unwritten_slot(
            reprs.wr,
            old_ibm.architecture_repr,
            reprs.input_ids,
            reprs.positions,
            old_e.kv_caches_repr@,
            reprs.slots,
            reprs.cu_q,
            reprs.cu_k,
            reprs.max_q,
            reprs.max_k,
            reprs.bt,
            layer as nat,
            slot,
        );
        assert(new_e.kv_caches_repr@
            == crate::exec::engine::architecture_engine_post_kv_of(
                old_e, reprs, old_e.kv_caches_repr@,
            ));
        assert(old_ibm.architecture_repr
            == RT::model_weights_architecture_repr_of(
                &old_e.weights_perms@,
            ));
        reveal(crate::exec::engine::architecture_engine_post_kv_of);
    }
}

// Architecture-neutral coherence discharge.  Emitting requests are covered
// directly by the whole-model relocation theorem, which already establishes
// equality for every layer and every logical position through `k_len`.
// Unemitted survivors reuse the common no-touch bundle.  No family layer fold
// or family cache semantics is visible here.
#[verifier::spinoff_prover]
pub proof fn discharge_architecture_engine_kv_coherent(
    old_e: Engine,
    new_e: Engine,
    old_ibm: IndependentBatchModel,
    emitted: Map<RequestId, TokenId>,
    samples: Map<RequestId, (crate::exec::request_state::SamplerState, TokenId)>,
    reprs: StepReprs,
)
    requires
        crate::proof::engine::refinement::architecture_semantic_inv(&old_e, &old_ibm),
        crate::exec::engine::architecture_engine_step_relation(
            old_e, new_e, emitted, samples, reprs,
        ),
        crate::exec::engine::engine_step_semantic_identity(old_e, new_e),
        architecture_emitted_cache_alignment(
            old_e, new_e, old_ibm, emitted, reprs,
        ),
        unscheduled_no_touch(
            old_e,
            new_e,
            old_ibm,
            crate::proof::engine::abstract_step::architecture_stepped_ibm(
                old_e, new_e, old_ibm, emitted, reprs,
            ),
            reprs,
            new_e.model_config.num_layers as nat,
        ),
    ensures
        engine_kv_coherent(
            &new_e,
            &crate::proof::engine::abstract_step::architecture_stepped_ibm(
                old_e, new_e, old_ibm, emitted, reprs,
            ),
        ),
{
    let new_ibm = crate::proof::engine::abstract_step::architecture_stepped_ibm(
        old_e, new_e, old_ibm, emitted, reprs,
    );
    let num_layers = new_e.model_config.num_layers as nat;
    reveal(crate::proof::engine::refinement::architecture_semantic_inv);
    reveal(crate::exec::engine::architecture_engine_step_relation);
    reveal(crate::exec::engine::engine_step_semantic_identity);
    reveal(architecture_emitted_cache_alignment);
    assert(new_e.kv_caches_repr@.len() == num_layers);
    assert(engine_kv_coherent_at(
        new_e.kv_caches_repr@, &new_e.cs, &new_ibm, num_layers,
    )) by {
        assert forall|rid: RequestId, layer: int, pos: nat|
            #![trigger new_e.kv_caches_repr@[layer],
                block_table_slot(residency_block_table(&new_e.cs, rid), pos)]
            new_e.cs.live_requests@.contains_key(rid)
                && new_ibm.machines.contains_key(rid)
                && 0 <= layer < num_layers as int
                && pos < new_ibm.machines[rid].kv_tokens
            implies {
                let bt = residency_block_table(&new_e.cs, rid);
                cache_at(
                    new_e.kv_caches_repr@[layer].0,
                    block_table_slot(bt, pos),
                ) == cache_at(
                    new_ibm.machines[rid].kv_cache_reprs[layer].0,
                    pos,
                )
                && cache_at(
                    new_e.kv_caches_repr@[layer].1,
                    block_table_slot(bt, pos),
                ) == cache_at(
                    new_ibm.machines[rid].kv_cache_reprs[layer].1,
                    pos,
                )
            }
        by {
            if crate::exec::engine::reprs_emits(reprs, rid) {
                assert(reprs.scheduled.contains(rid));
                let i = crate::proof::engine::abstract_step::architecture_scheduled_index_of(
                    reprs, rid,
                );
                let k_len = crate::proof::engine::abstract_step::architecture_machine_key_len(
                    reprs, rid,
                );
                let machine_bt = crate::proof::reference::request_machine::singleton_block_rows(
                    k_len,
                );
                crate::proof::engine::abstract_step::lemma_architecture_machine_forward_matches_engine(
                    old_e, old_ibm, reprs, rid,
                );
                assert(new_e.kv_caches_repr@
                    == crate::exec::engine::architecture_engine_post_kv_of(
                        old_e, reprs, old_e.kv_caches_repr@,
                    ));
                assert(new_ibm.machines[rid]
                    == crate::proof::engine::abstract_step::architecture_stepped_machine(
                        old_e, new_e, old_ibm, emitted, reprs, rid,
                    ));
                assert(emitted.contains_key(rid));
                assert(new_ibm.machines[rid].kv_cache_reprs
                    == crate::proof::engine::abstract_step::architecture_machine_cache_after(
                        old_e, old_ibm, reprs, rid,
                    ));
                assert(pos < k_len);
                assert(crate::proof::model::family_layout::cache_pair_logical_prefix_equal(
                    new_e.kv_caches_repr@[layer],
                    reprs.bt[i],
                    new_ibm.machines[rid].kv_cache_reprs[layer],
                    machine_bt[0],
                    k_len,
                ));
                let plan_slot = block_table_slot(reprs.bt[i], pos);
                let post_slot = block_table_slot(
                    residency_block_table(&new_e.cs, rid), pos,
                );
                assert(post_slot == plan_slot);
                crate::proof::tensor::geometry::lemma_blocks_needed_covers_pos(pos, k_len);
                assert(crate::proof::tensor::geometry::blocks_needed_for(k_len)
                    <= u64::MAX as nat);
                crate::proof::reference::request_machine::lemma_contiguous_block_table_slot(
                    crate::proof::tensor::geometry::blocks_needed_for(k_len), pos,
                );
                assert(block_table_slot(machine_bt[0], pos) == pos);
                reveal(crate::proof::model::family_layout::cache_pair_logical_prefix_equal);
            } else {
                crate::proof::cache::plan_witnesses::lemma_unscheduled_coherence_at(
                    old_e, new_e, old_ibm, new_ibm, reprs,
                    num_layers, rid, layer, pos,
                );
            }
        }
    }
}

} // verus!
