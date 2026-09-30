//! Architecture-neutral launch-layout facts derived from scheduler geometry.
//!
//! This module deliberately contains no model-family semantics. It proves the
//! scatter-store metadata domain shared by every eager forward.

#[cfg(verus_only)]
use crate::{types::{BLOCK_SIZE_SPEC}, proof::tensor::geometry::{block_table_slot}};
use crate::exec::engine::{Engine, StepReprs};
use crate::exec::request_state::*;
use crate::boundary::tensor_runtime as RT;
use vstd::prelude::*;

verus! {

// A materialized plan slot is never overwritten by a later scatter entry.
// Within its own row this follows from the duplicate-free block table and
// distinct logical positions; across rows it follows from stable page
// separation.  This deliberately avoids post-commit residency state.
proof fn lemma_plan_slot_not_overwritten_later(
    old_e: Engine,
    reprs: StepReprs,
    k: int,
    q: int,
)
    requires
        crate::exec::engine::step_reprs_wf(old_e, reprs),
        crate::exec::engine::reprs_forward_layout_at(old_e, reprs, k),
        crate::exec::engine::reprs_write_pages_disjoint(reprs),
        0 <= k < reprs.scheduled.len(),
        reprs.cu_q[k] <= q < reprs.cu_q[k + 1],
    ensures
        forall|m: int| q < m < reprs.slots.len() ==>
            reprs.slots[m] != reprs.slots[q],
{
    let nrows = reprs.scheduled.len() as int;
    let s0 = reprs.cu_q[k];
    let s1 = reprs.cu_q[k + 1];
    let rid = reprs.scheduled[k];
    reveal(crate::exec::engine::reprs_forward_layout_at);
    assert forall|m: int| q < m < reprs.slots.len() implies
        reprs.slots[m] != reprs.slots[q]
    by {
        reveal(crate::exec::engine::reprs_forward_layout_at);
        if m < s1 {
            let a = if old_e.cs.running@.contains(rid) {
                (crate::exec::request_state::history(
                    old_e.cs.live_requests@[rid],
                ).len() - 1) as nat
            } else {
                let kd = reprs.cu_k[k + 1] - reprs.cu_k[k];
                let c = kd - (s1 - s0);
                (c + q - s0) as nat
            };
            let b = if old_e.cs.running@.contains(rid) {
                (crate::exec::request_state::history(
                    old_e.cs.live_requests@[rid],
                ).len() - 1) as nat
            } else {
                let kd = reprs.cu_k[k + 1] - reprs.cu_k[k];
                let c = kd - (s1 - s0);
                (c + m - s0) as nat
            };
            if old_e.cs.running@.contains(rid) {
                assert(s1 == s0 + 1);
                assert(false);
            } else {
                let kd = reprs.cu_k[k + 1] - reprs.cu_k[k];
                let c = kd - (s1 - s0);
                assert(a == (c + q - s0) as nat);
                assert(b == (c + m - s0) as nat);
                assert(reprs.input_ids[q]
                    == old_e.cs.live_requests@[rid].prompt_tokens@[
                        c + q - s0
                    ] as int);
                assert(reprs.input_ids[m]
                    == old_e.cs.live_requests@[rid].prompt_tokens@[
                        c + m - s0
                    ] as int);
                assert(reprs.slots[q] >= 0);
                assert(reprs.slots[m] >= 0);
                assert(a != b);
                assert((a as int) / (BLOCK_SIZE_SPEC as int)
                    < reprs.bt[k].len());
                assert((b as int) / (BLOCK_SIZE_SPEC as int)
                    < reprs.bt[k].len());
                crate::proof::model::family_layout::lemma_block_table_slot_injective(
                    reprs.bt[k], a, b,
                );
                assert(reprs.slots[q] as nat
                    == block_table_slot(
                        reprs.bt[k], (c + q - s0) as nat,
                    ));
                assert(reprs.slots[m] as nat
                    == block_table_slot(
                        reprs.bt[k], (c + m - s0) as nat,
                    ));
                assert(reprs.slots[q] as nat
                    == block_table_slot(reprs.bt[k], a));
                assert(reprs.slots[m] as nat
                    == block_table_slot(reprs.bt[k], b));
            }
        } else {
            let row = crate::proof::tensor::geometry::lemma_cu_locate(reprs.cu_q, nrows, m);
            assert(row != k);
            reveal(crate::exec::engine::reprs_write_pages_disjoint);
            assert(reprs.slots[q] / (BLOCK_SIZE_SPEC as int)
                != reprs.slots[m] / (BLOCK_SIZE_SPEC as int));
        }
    }
}

// A plan scatter slot is inside the fixed physical cache pool.  This is the
// unary launch fact consumed by the framework-to-store-kernel bridge; it is
// intentionally independent of any numerical cache contents.
proof fn lemma_reprs_store_slot_in_bounds(
    old_e: Engine,
    reprs: StepReprs,
    k: int,
    q: int,
)
    requires
        crate::exec::engine::step_reprs_wf(old_e, reprs),
        crate::exec::engine::reprs_forward_layout_at(old_e, reprs, k),
        old_e.cs.num_blocks > 0,
        0 <= k < reprs.scheduled.len(),
        reprs.cu_q[k] <= q < reprs.cu_q[k + 1],
    ensures
        0 <= reprs.slots[q],
        reprs.slots[q] < old_e.cs.num_blocks as int
            * (BLOCK_SIZE_SPEC as int),
{
    let bs = BLOCK_SIZE_SPEC as int;
    let rid = reprs.scheduled[k];
    let s0 = reprs.cu_q[k];
    let s1 = reprs.cu_q[k + 1];
    let kd = reprs.cu_k[k + 1] - reprs.cu_k[k];
    reveal(crate::exec::engine::reprs_forward_layout_at);

    let pos: nat;
    if old_e.cs.running@.contains(rid) {
        let h = history(old_e.cs.live_requests@[rid]);
        assert(s1 == s0 + 1);
        assert(q == s0);
        assert(kd == h.len() as int);
        assert(h.len() > 0);
        pos = (h.len() - 1) as nat;
        assert(pos < kd as nat);
        assert(reprs.slots[q] >= 0);
        assert(reprs.slots[q] as nat
            == block_table_slot(reprs.bt[k], pos));
    } else {
        let c = kd - (s1 - s0);
        let p = c + q - s0;
        assert(0 <= p < kd);
        // Trigger the per-query payload clause whose primary trigger is the
        // input row; slot bounds are another conjunct of that same clause.
        assert(reprs.input_ids[q]
            == old_e.cs.live_requests@[rid].prompt_tokens@[p] as int);
        pos = p as nat;
        assert(pos < kd as nat);
        assert(reprs.slots[q] >= 0);
        assert(reprs.slots[q] as nat
            == block_table_slot(reprs.bt[k], pos));
    }

    crate::proof::tensor::geometry::lemma_blocks_needed_covers_pos(pos, kd as nat);
    assert(pos / BLOCK_SIZE_SPEC < reprs.bt[k].len());
    crate::proof::tensor::geometry::block_table_slot_block(reprs.bt[k], pos);
    let slot = reprs.slots[q];
    assert(slot / bs
        == reprs.bt[k][(pos as int) / bs] as int);
    assert(slot / bs < old_e.cs.num_blocks as int);
    vstd::arithmetic::div_mod::lemma_fundamental_div_mod(slot, bs);
    vstd::arithmetic::div_mod::lemma_mod_bound(slot, bs);
    assert(bs == 64);
    assert(slot == (slot / bs) * bs + slot % bs);
    assert(0 <= slot % bs < bs);
    assert(slot < old_e.cs.num_blocks as int * bs);
}

// Exact plan-level discharge of store_kv_cache.py's quantified unary
// metadata preconditions: nonempty rows, in-bounds slots, and injectivity.
// Cross-row uniqueness uses page separation; same-row uniqueness uses the
// duplicate-free block table through lemma_plan_slot_not_overwritten_later.
pub proof fn lemma_reprs_store_kv_cache_metadata_ready(
    old_e: Engine,
    reprs: StepReprs,
)
    requires
        crate::exec::engine::step_reprs_wf(old_e, reprs),
        crate::exec::engine::reprs_forward_layout_ok(old_e, reprs),
        crate::exec::engine::reprs_write_pages_disjoint(reprs),
        reprs.scheduled.len() > 0,
        old_e.cs.num_blocks > 0,
    ensures
        RT::store_kv_cache_metadata_ready(
            reprs.input_ids.len(), old_e.cs.num_blocks as nat, reprs.slots,
        ),
{
    reveal(crate::exec::engine::step_reprs_wf);
    reveal(RT::store_kv_cache_metadata_ready);
    let nrows = reprs.scheduled.len() as int;
    assert(0 < nrows);
    assert(0 <= 0 < nrows);
    assert(reprs.cu_q[0] < reprs.cu_q[0int + 1]);
    assert(reprs.cu_q[0] < reprs.cu_q[1]);
    crate::proof::tensor::geometry::lemma_cu_mono(reprs.cu_q, nrows, 1, nrows);
    assert(reprs.input_ids.len() > 0);

    assert forall|q: int| 0 <= q < reprs.slots.len() implies {
        let slot = #[trigger] reprs.slots[q];
        &&& 0 <= slot
        &&& slot < old_e.cs.num_blocks as int
            * (BLOCK_SIZE_SPEC as int)
    } by {
        let k = crate::proof::tensor::geometry::lemma_cu_locate(
            reprs.cu_q, nrows, q,
        );
        assert(crate::exec::engine::reprs_forward_layout_at(old_e, reprs, k));
        lemma_reprs_store_slot_in_bounds(old_e, reprs, k, q);
    }

    assert(reprs.slots.no_duplicates()) by {
        reveal(Seq::no_duplicates);
        assert forall|q: int, m: int|
            0 <= q < reprs.slots.len()
            && 0 <= m < reprs.slots.len()
            && q != m
            implies reprs.slots[q] != reprs.slots[m]
        by {
            if q < m {
                let k = crate::proof::tensor::geometry::lemma_cu_locate(
                    reprs.cu_q, nrows, q,
                );
                assert(crate::exec::engine::reprs_forward_layout_at(
                    old_e, reprs, k,
                ));
                lemma_plan_slot_not_overwritten_later(old_e, reprs, k, q);
            } else {
                let k = crate::proof::tensor::geometry::lemma_cu_locate(
                    reprs.cu_q, nrows, m,
                );
                assert(crate::exec::engine::reprs_forward_layout_at(
                    old_e, reprs, k,
                ));
                lemma_plan_slot_not_overwritten_later(old_e, reprs, k, m);
            }
        }
    }
}

} // verus!
