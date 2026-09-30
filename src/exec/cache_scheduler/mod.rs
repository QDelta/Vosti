// Concrete cache-scheduler facade and step-plan materialization.
//
// Scheduler representation, queue mechanics, stable-boundary invariants,
// eviction policy, and executable transitions live in the focused child
// modules below. Public types, specs, and lemmas are re-exported here so the
// engine and refinement layers retain one stable `cache_scheduler::*` surface.
//
// Request bootstrapping, allocation/reuse, mixed prefill/decode planning,
// append/deallocate, and commit are fully verified real exec bodies.  See
// `README.md` for feature scope (dynamic arrivals are stable-boundary
// events; there is still no preemption, and pressure reclamation is
// driven by vLLM-style intrusive availability queues rather than LRU).

use crate::proof::tensor::geometry::*;
use crate::exec::request_state::*;
use crate::{types::*, proof::model::types::*, proof::tensor::types::*};
use crate::boundary::tensor_runtime as RT;
use vstd::hash_map::HashMapWithView;
use vstd::prelude::*;
#[cfg(verus_only)]
use vstd::std_specs::hash::obeys_key_model;
use vstd::{assert_seqs_equal, assert_sets_equal};

pub(crate) mod availability_queue;
pub use availability_queue::*;
pub(crate) mod types;
pub use crate::proof::scheduler::*;
pub use types::*;
mod admission;
mod append;
mod commit;
mod decode_publication;
mod eviction_policy;
mod planning;
pub use planning::select_prefill_chunk_len;
mod prefix_cache;
mod reclamation;
mod request_lifecycle;

verus! {
pub open spec fn step_plan_perms_valid(plan: &StepPlan, perms: StepPlanPerms) -> bool {
    RT::int_tensor_repr_1d(perms.input_ids, plan.input_ids, plan.input_ids_repr@)
    && RT::int_tensor_repr_1d(perms.positions, plan.positions, plan.positions_repr@)
    && RT::block_table_repr(perms.block_table, plan.block_table, plan.block_table_repr@)
    && RT::int_tensor_repr_1d(perms.slot_mapping, plan.slot_mapping, plan.slot_mapping_repr@)
    && RT::int_tensor_repr_1d(perms.cu_seqlens_q, plan.cu_seqlens_q, plan.cu_seqlens_q_repr@)
    && RT::int_tensor_repr_1d(perms.cu_seqlens_k, plan.cu_seqlens_k, plan.cu_seqlens_k_repr@)
}

pub open spec fn step_plan_commit_ready(cs: &CacheScheduler, plan: &StepPlan) -> bool {
    cs.num_blocks <= u64::MAX / BLOCK_SIZE
    && plan.scheduled_ids@.no_duplicates()
    && (forall|i: int|
        #![trigger plan.scheduled_ids@[i]]
        0 <= i < plan.scheduled_ids@.len()
        ==> !cs.waiting@.contains(plan.scheduled_ids@[i])
            && (cs.live_requests@.contains_key(plan.scheduled_ids@[i]) ==> can_step(
                cs.live_requests@[plan.scheduled_ids@[i]],
            ) && cs.live_requests@[plan.scheduled_ids@[i]].generated_tokens@.len()
                < usize::MAX as int))
}

pub open spec fn decode_plan_ready(cs: &CacheScheduler) -> bool {
    cs.num_blocks <= u64::MAX / BLOCK_SIZE
    && (forall|rid: RequestId|
        #[trigger] cs.running@.contains(rid)
        ==> cs.live_requests@.contains_key(rid)
            && can_step(cs.live_requests@[rid])
            && cs.live_requests@[rid].generated_tokens@.len() < usize::MAX as int
            && history(cs.live_requests@[rid]).len() <= usize::MAX as int
            && history(cs.live_requests@[rid]).len() <= u64::MAX as int
            && cs.request_residency@.contains_key(rid)
            && cs.request_residency@[rid].slot_mapping@.len() > 0)
}

// Prefill-side readiness facts not in `cs_valid` (mirrors
// `decode_plan_ready`): waiting requests, when live, are steppable with
// executable prompt/generated-token capacities.
pub open spec fn prefill_plan_ready(cs: &CacheScheduler) -> bool {
    forall|rid: RequestId|
        #[trigger] cs.waiting@.contains(rid) && cs.live_requests@.contains_key(rid)
        ==> can_step(cs.live_requests@[rid])
            && cs.live_requests@[rid].generated_tokens@.len() < usize::MAX as int
            && cs.live_requests@[rid].prompt_tokens@.len() <= u64::MAX as int
            && cs.live_requests@[rid].prompt_tokens@.len() <= usize::MAX as int
}

// Inductive scheduler-readiness invariant.  Unlike the queue-specific
// `prefill_plan_ready` / `decode_plan_ready` predicates, this talks only about
// live request state and is therefore preserved uniformly when a step moves a
// request from waiting to running or removes a finished request.
pub open spec fn live_request_step_ready(cs: &CacheScheduler) -> bool {
    forall|rid: RequestId|
        #[trigger] cs.live_requests@.contains_key(rid)
        ==> can_step(cs.live_requests@[rid])
            && request_history_capacity_safe(cs.live_requests@[rid])
}

// The stable live-request budget discharges both mode-specific arithmetic
// interfaces consumed by `plan`.  Queue coverage, residency existence, and a
// nonempty decode slot mapping come from the already-preserved scheduler
// companions rather than being duplicated in `live_request_step_ready`.
pub proof fn lemma_live_request_step_ready_implies_plan_ready(
    cs: &CacheScheduler,
)
    requires
        cs_valid(cs),
        cs.num_blocks <= u64::MAX / BLOCK_SIZE,
        live_request_step_ready(cs),
        residency_history_aligned(cs),
        slot_mapping_aligned(cs),
    ensures
        prefill_plan_ready(cs),
        decode_plan_ready(cs),
{
    assert(prefill_plan_ready(cs)) by {
        assert forall|rid: RequestId|
            #[trigger] cs.waiting@.contains(rid)
                && cs.live_requests@.contains_key(rid)
            implies can_step(cs.live_requests@[rid])
                && cs.live_requests@[rid].generated_tokens@.len()
                    < usize::MAX as int
                && cs.live_requests@[rid].prompt_tokens@.len()
                    <= u64::MAX as int
                && cs.live_requests@[rid].prompt_tokens@.len()
                    <= usize::MAX as int
        by {
            let state = cs.live_requests@[rid];
            assert(can_step(state));
            assert(request_history_capacity_safe(state));
            assert(valid_request_state(state));
            assert(!is_finished(state));
            assert(state.generated_tokens@.len() <= state.max_tokens as int);
            assert(state.generated_tokens@.len() != state.max_tokens as int);
            assert(state.generated_tokens@.len() < state.max_tokens as int);
            assert(state.max_tokens as int <= usize::MAX as int);
            assert(state.generated_tokens@.len() < usize::MAX as int);
            assert(state.prompt_tokens@.len()
                <= state.prompt_tokens@.len() + state.max_tokens as int);
        }
    }
    assert(decode_plan_ready(cs)) by {
        assert forall|rid: RequestId|
            #[trigger] cs.running@.contains(rid)
            implies cs.live_requests@.contains_key(rid)
                && can_step(cs.live_requests@[rid])
                && cs.live_requests@[rid].generated_tokens@.len()
                    < usize::MAX as int
                && history(cs.live_requests@[rid]).len()
                    <= usize::MAX as int
                && history(cs.live_requests@[rid]).len()
                    <= u64::MAX as int
                && cs.request_residency@.contains_key(rid)
                && cs.request_residency@[rid].slot_mapping@.len() > 0
        by {
            assert(live_covers_queue(cs));
            assert(cs.live_requests@.contains_key(rid));
            let state = cs.live_requests@[rid];
            assert(can_step(state));
            assert(request_history_capacity_safe(state));
            assert(valid_request_state(state));
            assert(!is_finished(state));
            assert(state.generated_tokens@.len() <= state.max_tokens as int);
            assert(state.generated_tokens@.len() != state.max_tokens as int);
            assert(state.generated_tokens@.len() < state.max_tokens as int);
            assert(state.max_tokens as int <= usize::MAX as int);
            assert(state.generated_tokens@.len() < usize::MAX as int);
            assert(history(state).len()
                == state.prompt_tokens@.len()
                    + state.generated_tokens@.len());
            assert(history(state).len()
                <= state.prompt_tokens@.len() + state.max_tokens as int);
            assert(running_has_residency(cs));
            assert(cs.request_residency@.contains_key(rid));
            assert(slot_mapping_aligned(cs));
            assert(cs.request_residency@[rid].slot_mapping@.len() >= 1);
        }
    }
}

// Plan tensor-shape facts (2026-08-05): query-token bookkeeping ties
// cu_seqlens_q to the input rows.  Consumed by the engine's verified
// `step_core` (previously supplied on trust by the external_body `step`).
pub open spec fn step_plan_shape_ok(plan: &StepPlan) -> bool {
    plan.sample_mask@.len() == plan.scheduled_ids@.len()
    && plan.input_ids_repr@.len() == plan.positions_repr@.len()
    && plan.slot_mapping_repr@.len() == plan.input_ids_repr@.len()
    && plan.cu_seqlens_q_repr@.len() == plan.scheduled_ids@.len() + 1
    && plan.cu_seqlens_k_repr@.len() == plan.scheduled_ids@.len() + 1
    && plan.block_table_repr@.len() == plan.scheduled_ids@.len()
    && plan.cu_seqlens_q_repr@[0] == 0
    && plan.cu_seqlens_k_repr@[0] == 0
    && (forall|j: int| 0 <= j < plan.scheduled_ids@.len() as int ==>
        plan.cu_seqlens_q_repr@[j] < #[trigger] plan.cu_seqlens_q_repr@[j + 1])
    && (forall|j: int| 0 <= j < plan.scheduled_ids@.len() as int ==>
        plan.cu_seqlens_k_repr@[j] < #[trigger] plan.cu_seqlens_k_repr@[j + 1])
    && (forall|j: int| #![trigger plan.cu_seqlens_q_repr@[j + 1]]
        0 <= j < plan.scheduled_ids@.len() as int ==> {
        let q_len = plan.cu_seqlens_q_repr@[j + 1]
            - plan.cu_seqlens_q_repr@[j];
        let k_len = plan.cu_seqlens_k_repr@[j + 1]
            - plan.cu_seqlens_k_repr@[j];
        &&& q_len <= plan.max_seqlen_q as int
        &&& k_len <= plan.max_seqlen_k as int
        &&& q_len <= k_len
    })
    && plan.cu_seqlens_q_repr@[plan.scheduled_ids@.len() as int]
        == plan.input_ids_repr@.len() as int
}

pub open spec fn step_plan_emits(plan: &StepPlan, rid: RequestId) -> bool {
    exists|i: int| #![trigger plan.scheduled_ids@[i], plan.sample_mask@[i]]
        0 <= i < plan.scheduled_ids@.len()
            && plan.scheduled_ids@[i] == rid
            && plan.sample_mask@[i]
}

pub open spec fn step_plan_parks_before(
    plan: &StepPlan,
    rid: RequestId,
    upto: int,
) -> bool {
    exists|i: int| #![trigger plan.scheduled_ids@[i], plan.sample_mask@[i]]
        0 <= i < upto
            && plan.scheduled_ids@[i] == rid
            && !plan.sample_mask@[i]
}

pub open spec fn step_plan_parks(plan: &StepPlan, rid: RequestId) -> bool {
    step_plan_parks_before(plan, rid, plan.scheduled_ids@.len() as int)
}

// Initial mask for the decode prefix. Admissions append their own finality
// decision as planning selects full or KV-only prefill chunks.
pub fn all_true_sample_mask(len: usize) -> (mask: Vec<bool>)
    ensures
        mask@.len() == len,
        forall|i: int| 0 <= i < mask@.len() ==> #[trigger] mask@[i],
{
    let mut mask = Vec::<bool>::new();
    while mask.len() < len
        invariant
            mask@.len() <= len,
            forall|i: int| 0 <= i < mask@.len() ==> #[trigger] mask@[i],
        decreases len - mask.len(),
    {
        mask.push(true);
    }
    mask
}

// Executable prefix copy used when publishing only the complete pages whose
// K/V values were materialized by a chunked-prefill row.
pub fn token_prefix(tokens: &Vec<TokenId>, end: usize) -> (out: Vec<TokenId>)
    requires
        end as int <= tokens@.len(),
    ensures
        out@ == tokens@.subrange(0, end as int),
{
    let mut out = Vec::new();
    let mut i: usize = 0;
    while i < end
        invariant
            i <= end,
            end as int <= tokens@.len(),
            out@ == tokens@.subrange(0, i as int),
        decreases end - i,
    {
        out.push(tokens[i]);
        i += 1;
    }
    out
}

pub fn remove_request_id_from_queue(queue: &mut Vec<RequestId>, rid: RequestId)
    requires
        old(queue)@.no_duplicates(),
    ensures
        !final(queue)@.contains(rid),
        final(queue)@.no_duplicates(),
        forall|r: RequestId| #[trigger] final(queue)@.contains(r) ==> old(queue)@.contains(r),
        forall|r: RequestId|
            r != rid && #[trigger] old(queue)@.contains(r) ==> final(queue)@.contains(r),
{
    let mut i: usize = 0;
    while i < queue.len()
        invariant
            i <= queue@.len(),
            queue@ == old(queue)@,
            queue@.no_duplicates(),
            forall|k: int| 0 <= k < i as int ==> queue@[k] != rid,
        decreases queue@.len() - i
    {
        if queue[i] == rid {
            let ghost old_queue = queue@;
            queue.remove(i);
            proof {
                old_queue.remove_ensures(i as int);
            }
            assert(!queue@.contains(rid)) by {
                if queue@.contains(rid) {
                    let idx = queue@.index_of(rid);
                    assert(0 <= idx < queue@.len());
                    if idx < i as int {
                        assert(queue@[idx] == old_queue[idx]);
                        assert(old_queue[idx] == rid);
                        assert(old_queue[i as int] == rid);
                        assert(idx != i as int);
                        assert(old_queue[idx] != old_queue[i as int]);
                    } else {
                        assert(queue@[idx] == old_queue[idx + 1]);
                        assert(old_queue[idx + 1] == rid);
                        assert(old_queue[i as int] == rid);
                        assert(idx + 1 != i as int);
                        assert(old_queue[idx + 1] != old_queue[i as int]);
                    }
                }
            }
            assert(queue@.no_duplicates());
            assert forall|r: RequestId|
                #[trigger] queue@.contains(r) implies old(queue)@.contains(r)
            by {
                let idx = queue@.index_of(r);
                assert(0 <= idx < queue@.len());
                if idx < i as int {
                    assert(queue@[idx] == old_queue[idx]);
                } else {
                    assert(queue@[idx] == old_queue[idx + 1]);
                }
            }
            assert forall|r: RequestId|
                r != rid && #[trigger] old(queue)@.contains(r) implies queue@.contains(r)
            by {
                if !queue@.contains(r) {
                    let idx = old_queue.index_of(r);
                    assert(0 <= idx < old_queue.len());
                    if idx < i as int {
                        assert(queue@[idx] == old_queue[idx]);
                    } else if idx > i as int {
                        assert(queue@[idx - 1] == old_queue[idx]);
                    } else {
                        assert(old_queue[i as int] == rid);
                    }
                }
            }
            return;
        }
        assert(queue@[i as int] != rid);
        i = i + 1;
        assert forall|k: int| 0 <= k < i as int implies queue@[k] != rid by {
            if 0 <= k < i as int {
                if k < i as int - 1 {
                } else {
                    assert(k == i as int - 1);
                }
            }
        }
    }
    assert(queue@.no_duplicates());
    assert forall|r: RequestId| #[trigger] queue@.contains(r) implies old(queue)@.contains(r) by {
    }
    assert forall|r: RequestId|
        r != rid && #[trigger] old(queue)@.contains(r) implies queue@.contains(r)
    by {
    }
    assert(!queue@.contains(rid)) by {
        if queue@.contains(rid) {
            let idx = queue@.index_of(rid);
            assert(0 <= idx < queue@.len());
            assert(idx < i as int);
            assert(queue@[idx] != rid);
        }
    }
}

pub fn record_emitted_token(
    out: &mut EmittedTokens,
    emitted: &SampleResults,
    rid: RequestId,
    token: TokenId,
)
    requires
        old(out)@.dom().subset_of(emitted@.dom()),
        forall|r: RequestId|
            #[trigger] old(out)@.contains_key(r)
            ==> old(out)@[r] == emitted@[r].token,
        emitted@.contains_key(rid),
        token == emitted@[rid].token,
    ensures
        final(out)@.dom().subset_of(emitted@.dom()),
        forall|r: RequestId|
            #[trigger] final(out)@.contains_key(r) ==> final(out)@[r] == emitted@[r].token,
        final(out)@.contains_key(rid),
        final(out)@[rid] == token,
        final(out)@ == old(out)@.insert(rid, token),
{
    let ghost out_before = out@;
    out.insert(rid, token);
    proof {
        vstd::map::lemma_map_insert_domain(out_before, rid, token);
    }
    assert(out@.dom() == out_before.dom().insert(rid));
    assert(out@.dom().subset_of(emitted@.dom())) by {
        assert forall|r: RequestId|
            #[trigger] out@.dom().contains(r)
            implies emitted@.dom().contains(r)
        by {
            if r == rid {
                assert(emitted@.contains_key(rid));
            } else {
                assert(out_before.dom().contains(r));
                assert(out_before.dom().subset_of(emitted@.dom()));
            }
        }
    }
    assert forall|r: RequestId|
        #[trigger] out@.contains_key(r) implies out@[r] == emitted@[r].token
    by {
        if r == rid {
        } else {
            assert(out_before.contains_key(r));
            assert(out@[r] == out_before[r]);
        }
    }
}

pub fn block_rows_for_scheduled(
    cs: &CacheScheduler,
    scheduled_ids: &Vec<RequestId>,
) -> (out: Vec<Vec<u64>>)
    requires
        cs_valid(cs),
        forall|i: int|
            #![trigger scheduled_ids@[i]]
            0 <= i < scheduled_ids@.len()
            ==> cs.request_residency@.contains_key(scheduled_ids@[i]),
    ensures
        out@.len() == scheduled_ids@.len(),
        forall|i: int|
            #![trigger out@[i]]
            0 <= i < out@.len()
            ==> out@[i]@ == cs.request_residency@[scheduled_ids@[i]].block_ids@,
{
    let mut out: Vec<Vec<u64>> = Vec::new();
    let mut i: usize = 0;
    while i < scheduled_ids.len()
        invariant
            i <= scheduled_ids@.len(),
            out@.len() == i as int,
            cs_valid(cs),
            forall|j: int|
                #![trigger scheduled_ids@[j]]
                0 <= j < scheduled_ids@.len()
                ==> cs.request_residency@.contains_key(scheduled_ids@[j]),
            forall|j: int|
                #![trigger out@[j]]
                0 <= j < out@.len()
                ==> out@[j]@ == cs.request_residency@[scheduled_ids@[j]].block_ids@,
        decreases scheduled_ids@.len() - i
    {
        let rid = scheduled_ids[i];
        assert(cs.request_residency@.contains_key(rid));
        let residency_opt = cs.request_residency.get(&rid);
        match residency_opt {
            Some(residency) => {
                let row = residency.block_ids.clone();
                assert(row@ == residency.block_ids@);
                assert(row@ == cs.request_residency@[rid].block_ids@);
                out.push(row);
                assert(out@[i as int]@ == cs.request_residency@[scheduled_ids@[i as int]].block_ids@);
            },
            None => {
                assert(false);
            },
        }
        i = i + 1;
        assert forall|j: int|
            #![trigger out@[j]]
            0 <= j < out@.len()
            implies out@[j]@ == cs.request_residency@[scheduled_ids@[j]].block_ids@
        by {
            if j == i as int - 1 {
            } else {
                assert(0 <= j < i as int - 1);
            }
        }
    }
    out
}

// Total logical key length represented by a decode schedule.  This is kept as
// a named recursive function (instead of an inline fold) so the executable
// planner can prove that cumulative `cu_seqlens_k` construction does not
// saturate.  Every summand is a sequence length and is therefore nonnegative.
pub open spec fn scheduled_history_len_at(
    cs: &CacheScheduler,
    rid: RequestId,
) -> int {
    if cs.live_requests@.contains_key(rid) {
        cs.live_requests@[rid].prompt_tokens@.len() as int
            + cs.live_requests@[rid].generated_tokens@.len() as int
    } else {
        0
    }
}

pub open spec fn scheduled_history_total(
    cs: &CacheScheduler,
    ids: Seq<RequestId>,
) -> int
    decreases ids.len(),
{
    if ids.len() == 0 {
        0
    } else {
        scheduled_history_len_at(cs, ids[0])
            + scheduled_history_total(cs, ids.subrange(1, ids.len() as int))
    }
}

pub proof fn lemma_scheduled_history_total_nonnegative(
    cs: &CacheScheduler,
    ids: Seq<RequestId>,
)
    ensures scheduled_history_total(cs, ids) >= 0,
    decreases ids.len(),
{
    if ids.len() > 0 {
        let tail = ids.subrange(1, ids.len() as int);
        lemma_scheduled_history_total_nonnegative(
            cs,
            tail,
        );
    }
}

pub proof fn lemma_scheduled_history_total_snoc(
    cs: &CacheScheduler,
    ids: Seq<RequestId>,
    rid: RequestId,
)
    ensures
        scheduled_history_total(cs, ids.push(rid))
            == scheduled_history_total(cs, ids)
                + scheduled_history_len_at(cs, rid),
    decreases ids.len(),
{
    if ids.len() == 0 {
        assert(ids.push(rid).len() == 1);
        assert(ids =~= Seq::<RequestId>::empty());
        assert(ids.push(rid)[0] == rid);
        assert(ids.push(rid).subrange(1, 1) =~= Seq::<RequestId>::empty());
        assert(scheduled_history_total(cs, Seq::<RequestId>::empty()) == 0);
        assert(scheduled_history_total(cs, ids.push(rid))
            == scheduled_history_len_at(cs, rid)
                + scheduled_history_total(cs, Seq::<RequestId>::empty()));
        assert(scheduled_history_total(cs, ids.push(rid))
            == scheduled_history_len_at(cs, rid));
        assert(scheduled_history_total(cs, ids) == 0);
    } else {
        let tail = ids.subrange(1, ids.len() as int);
        assert(ids.push(rid)[0] == ids[0]);
        assert(ids.push(rid).subrange(1, ids.len() as int + 1)
            =~= tail.push(rid));
        lemma_scheduled_history_total_snoc(cs, tail, rid);
        assert(scheduled_history_total(cs, ids.push(rid))
            == scheduled_history_len_at(cs, ids[0])
                + scheduled_history_total(cs, tail.push(rid)));
        assert(scheduled_history_total(cs, ids)
            == scheduled_history_len_at(cs, ids[0])
                + scheduled_history_total(cs, tail));
    }
}

pub open spec fn decode_input_row_at(
    cs: &CacheScheduler,
    scheduled_ids: Seq<RequestId>,
    input_values: Seq<u64>,
    position_values: Seq<u64>,
    cu_k_values: Seq<u64>,
    max_seqlen_k: usize,
    j: int,
) -> bool {
    let rid = scheduled_ids[j];
    let h = history(cs.live_requests@[rid]);
    &&& h.len() > 0
    &&& input_values[j] == h[h.len() - 1]
    &&& position_values[j] as int == h.len() as int - 1
    &&& cu_k_values[j + 1] as int == cu_k_values[j] as int + h.len() as int
    &&& h.len() <= max_seqlen_k as int
}

pub fn decode_inputs_for_scheduled(
    cs: &CacheScheduler,
    scheduled_ids: &Vec<RequestId>,
) -> (out: (Vec<u64>, Vec<u64>, Vec<u64>, Vec<u64>, usize))
    requires
        cs_valid(cs),
        scheduled_ids@.len() <= usize::MAX as int,
        scheduled_history_total(cs, scheduled_ids@) <= u64::MAX as int,
        forall|i: int|
            #![trigger scheduled_ids@[i]]
            0 <= i < scheduled_ids@.len()
            ==> cs.live_requests@.contains_key(scheduled_ids@[i])
                && valid_request_state(cs.live_requests@[scheduled_ids@[i]])
                && history(cs.live_requests@[scheduled_ids@[i]]).len() <= usize::MAX as int
                && history(cs.live_requests@[scheduled_ids@[i]]).len() <= u64::MAX as int,
    ensures ({
        let (input_values, position_values, cu_q_values, cu_k_values, max_seqlen_k) = out;
        input_values@.len() == scheduled_ids@.len()
        && position_values@.len() == scheduled_ids@.len()
        && cu_q_values@.len() == scheduled_ids@.len() + 1
        && cu_k_values@.len() == scheduled_ids@.len() + 1
        && cu_q_values@[0] == 0
        && cu_k_values@[0] == 0
        && (forall|j: int| 0 <= j < cu_q_values@.len()
            ==> #[trigger] cu_q_values@[j] == j)
        && (forall|j: int| 0 <= j < scheduled_ids@.len() ==>
            #[trigger] decode_input_row_at(
                cs, scheduled_ids@, input_values@, position_values@,
                cu_k_values@, max_seqlen_k, j,
            ))
    }),
{
    let mut input_values: Vec<u64> = Vec::new();
    let mut position_values: Vec<u64> = Vec::new();
    let mut cu_q_values: Vec<u64> = Vec::new();
    let mut cu_k_values: Vec<u64> = Vec::new();
    cu_q_values.push(0);
    cu_k_values.push(0);
    let mut max_seqlen_k: usize = 0;
    let mut i: usize = 0;
    proof {
        assert(scheduled_ids@.subrange(0, scheduled_ids@.len() as int)
            =~= scheduled_ids@);
    }
    while i < scheduled_ids.len()
        invariant
            i <= scheduled_ids@.len(),
            input_values@.len() == i as int,
            position_values@.len() == i as int,
            cu_q_values@.len() == i as int + 1,
            cu_k_values@.len() == i as int + 1,
            cu_q_values@[0] == 0,
            cu_k_values@[0] == 0,
            scheduled_history_total(cs, scheduled_ids@) <= u64::MAX as int,
            cu_k_values@[i as int] as int
                + scheduled_history_total(
                    cs,
                    scheduled_ids@.subrange(i as int, scheduled_ids@.len() as int),
                )
                == scheduled_history_total(cs, scheduled_ids@),
            forall|j: int| 0 <= j < cu_q_values@.len()
                ==> #[trigger] cu_q_values@[j] == j,
            forall|j: int| 0 <= j < i as int ==>
                #[trigger] decode_input_row_at(
                    cs, scheduled_ids@, input_values@, position_values@,
                    cu_k_values@, max_seqlen_k, j,
                ),
            cs_valid(cs),
            scheduled_ids@.len() <= usize::MAX as int,
            forall|j: int|
                #![trigger scheduled_ids@[j]]
                0 <= j < scheduled_ids@.len()
                ==> cs.live_requests@.contains_key(scheduled_ids@[j])
                    && valid_request_state(cs.live_requests@[scheduled_ids@[j]])
                    && history(cs.live_requests@[scheduled_ids@[j]]).len() <= usize::MAX as int
                    && history(cs.live_requests@[scheduled_ids@[j]]).len() <= u64::MAX as int,
        decreases scheduled_ids@.len() - i
    {
        let rid = scheduled_ids[i];
        assert(cs.live_requests@.contains_key(rid));
        let ghost state_view = cs.live_requests@[rid];
        let ghost inputs_pre = input_values@;
        let ghost positions_pre = position_values@;
        let ghost cu_k_pre = cu_k_values@;
        let state_opt = cs.live_requests.get(&rid);
        let old_max_seqlen_k = max_seqlen_k;
        match state_opt {
            Some(state_ref) => {
                assert(*state_ref == cs.live_requests@[rid]);
                assert(valid_request_state(*state_ref));
                let token = last_history_token(state_ref);
                let hlen = history_len(state_ref);
                assert(hlen > 0);
                let pos = hlen - 1;
                assert(pos as int == history(*state_ref).len() - 1);
                input_values.push(token);
                position_values.push(pos as u64);
                let next_q = (i + 1) as u64;
                cu_q_values.push(next_q);
                let prev_k = cu_k_values[i];
                proof {
                    let remaining = scheduled_ids@.subrange(
                        i as int, scheduled_ids@.len() as int,
                    );
                    assert(remaining.len() > 0);
                    assert(remaining[0] == rid);
                    let tail = remaining.subrange(1, remaining.len() as int);
                    assert(tail =~= scheduled_ids@.subrange(
                        i as int + 1, scheduled_ids@.len() as int,
                    ));
                    lemma_scheduled_history_total_nonnegative(cs, tail);
                    assert(scheduled_history_len_at(cs, rid) == hlen as int) by {
                        assert(history(state_view).len()
                            == state_view.prompt_tokens@.len()
                                + state_view.generated_tokens@.len());
                    }
                    assert(prev_k as int + hlen as int <= u64::MAX as int);
                }
                let next_k = prev_k + hlen as u64;
                cu_k_values.push(next_k);
                if hlen > max_seqlen_k {
                    max_seqlen_k = hlen;
                }
                assert(old_max_seqlen_k <= max_seqlen_k);
                assert(history(state_view).len() > 0);
                assert(input_values@[i as int]
                    == history(state_view)[history(state_view).len() - 1]);
                assert(position_values@[i as int] as int
                    == history(state_view).len() as int - 1);
                assert(cu_k_values@[i as int + 1] as int
                    == cu_k_values@[i as int] as int
                        + history(state_view).len() as int);
                assert(history(state_view).len() <= max_seqlen_k as int);
                assert(decode_input_row_at(
                    cs, scheduled_ids@, input_values@, position_values@,
                    cu_k_values@, max_seqlen_k, i as int,
                ));
            },
            None => {
                assert(false);
            },
        }
        i = i + 1;
        proof {
            assert forall|j: int| 0 <= j < i as int implies
                #[trigger] decode_input_row_at(
                    cs, scheduled_ids@, input_values@, position_values@,
                    cu_k_values@, max_seqlen_k, j,
                ) by {
                if j < i as int - 1 {
                    assert(decode_input_row_at(
                        cs, scheduled_ids@, inputs_pre, positions_pre,
                        cu_k_pre, old_max_seqlen_k, j,
                    ));
                    assert(input_values@[j] == inputs_pre[j]);
                    assert(position_values@[j] == positions_pre[j]);
                    assert(cu_k_values@[j] == cu_k_pre[j]);
                    assert(cu_k_values@[j + 1] == cu_k_pre[j + 1]);
                } else {
                    assert(j == i as int - 1);
                    assert(scheduled_ids@[j] == scheduled_ids@[i as int - 1]);
                    assert(state_view == cs.live_requests@[scheduled_ids@[j]]);
                    assert(decode_input_row_at(
                        cs, scheduled_ids@, input_values@, position_values@,
                        cu_k_values@, max_seqlen_k, j,
                    ));
                }
            }
        }
    }
    (input_values, position_values, cu_q_values, cu_k_values, max_seqlen_k)
}

pub fn decode_slots_for_scheduled(
    cs: &CacheScheduler,
    scheduled_ids: &Vec<RequestId>,
) -> (out: Vec<u64>)
    requires
        cs_valid(cs),
        forall|i: int|
            #![trigger scheduled_ids@[i]]
            0 <= i < scheduled_ids@.len()
            ==> cs.request_residency@.contains_key(scheduled_ids@[i])
                && cs.request_residency@[scheduled_ids@[i]].slot_mapping@.len() > 0,
    ensures
        out@.len() == scheduled_ids@.len(),
        // Each decode row carries the residency's LAST mapped slot.
        forall|j: int|
            #![trigger out@[j]]
            0 <= j < scheduled_ids@.len()
            ==> out@[j] == cs.request_residency@[scheduled_ids@[j]].slot_mapping@[
                    cs.request_residency@[scheduled_ids@[j]].slot_mapping@.len() - 1],
{
    let mut out: Vec<u64> = Vec::new();
    let mut i: usize = 0;
    while i < scheduled_ids.len()
        invariant
            i <= scheduled_ids@.len(),
            out@.len() == i as int,
            cs_valid(cs),
            forall|j: int|
                #![trigger scheduled_ids@[j]]
                0 <= j < scheduled_ids@.len()
                ==> cs.request_residency@.contains_key(scheduled_ids@[j])
                    && cs.request_residency@[scheduled_ids@[j]].slot_mapping@.len() > 0,
            forall|j: int|
                #![trigger out@[j]]
                0 <= j < i as int
                ==> out@[j] == cs.request_residency@[scheduled_ids@[j]].slot_mapping@[
                        cs.request_residency@[scheduled_ids@[j]].slot_mapping@.len() - 1],
        decreases scheduled_ids@.len() - i
    {
        let rid = scheduled_ids[i];
        assert(cs.request_residency@.contains_key(rid));
        let residency_opt = cs.request_residency.get(&rid);
        match residency_opt {
            Some(residency) => {
                assert(residency.slot_mapping@.len() > 0);
                let slot_index = residency.slot_mapping.len() - 1;
                let slot = residency.slot_mapping[slot_index];
                out.push(slot);
            },
            None => {
                assert(false);
            },
        }
        i = i + 1;
    }
    out
}

pub fn materialize_step_plan(
    scheduled_ids: Vec<RequestId>,
    sample_mask: Vec<bool>,
    mode: StepMode,
    input_values: Vec<u64>,
    position_values: Vec<u64>,
    block_rows: Vec<Vec<u64>>,
    slot_values: Vec<u64>,
    cu_q_values: Vec<u64>,
    cu_k_values: Vec<u64>,
    max_seqlen_q: usize,
    max_seqlen_k: usize,
    device_anchor: Option<&RT::Tensor>,
) -> (out: (StepPlan, Tracked<StepPlanPerms>))
    requires
        sample_mask@.len() == scheduled_ids@.len(),
        input_values@.len() == position_values@.len(),
        slot_values@.len() == input_values@.len(),
        block_rows@.len() == scheduled_ids@.len(),
        cu_q_values@.len() == scheduled_ids@.len() + 1,
        cu_k_values@.len() == scheduled_ids@.len() + 1,
        cu_q_values@[0] == 0,
        cu_k_values@[0] == 0,
        forall|j: int| 0 <= j < scheduled_ids@.len() ==>
            cu_q_values@[j] < #[trigger] cu_q_values@[j + 1],
        forall|j: int| 0 <= j < scheduled_ids@.len() ==>
            cu_k_values@[j] < #[trigger] cu_k_values@[j + 1],
        forall|j: int| #![trigger cu_q_values@[j + 1]]
            0 <= j < scheduled_ids@.len() ==> {
            let q_len = cu_q_values@[j + 1] as int - cu_q_values@[j] as int;
            let k_len = cu_k_values@[j + 1] as int - cu_k_values@[j] as int;
            &&& q_len <= max_seqlen_q as int
            &&& k_len <= max_seqlen_k as int
            &&& q_len <= k_len
        },
        cu_q_values@[scheduled_ids@.len() as int] as int == input_values@.len(),
    ensures
        step_plan_perms_valid(&out.0, out.1@),
        step_plan_shape_ok(&out.0),
        out.0.scheduled_ids@ == scheduled_ids@,
        out.0.sample_mask@ == sample_mask@,
        out.0.mode == mode,
        out.0.input_ids_repr@ == RT::u64_seq_to_int_repr(input_values@),
        out.0.positions_repr@ == RT::u64_seq_to_int_repr(position_values@),
        out.0.block_table_repr@ == RT::nested_u64_seq_to_block_repr(
            Seq::new(block_rows@.len(), |i: int| block_rows@[i]@),
        ),
        out.0.slot_mapping_repr@ == RT::u64_seq_to_int_repr(slot_values@),
        out.0.cu_seqlens_q_repr@ == RT::u64_seq_to_int_repr(cu_q_values@),
        out.0.cu_seqlens_k_repr@ == RT::u64_seq_to_int_repr(cu_k_values@),
        out.0.max_seqlen_q == max_seqlen_q,
        out.0.max_seqlen_k == max_seqlen_k,
{
    let input_ids_repr: Ghost<IntTensor1D> = Ghost(RT::u64_seq_to_int_repr(input_values@));
    let positions_repr: Ghost<IntTensor1D> = Ghost(RT::u64_seq_to_int_repr(position_values@));
    let slot_mapping_repr: Ghost<Seq<int>> = Ghost(RT::u64_seq_to_int_repr(slot_values@));
    let block_table_repr: Ghost<Seq<Seq<BlockId>>> = Ghost(
        RT::nested_u64_seq_to_block_repr(
            Seq::new(block_rows@.len(), |i: int| block_rows@[i]@),
        ),
    );
    let cu_seqlens_q_repr: Ghost<Seq<int>> = Ghost(RT::u64_seq_to_int_repr(cu_q_values@));
    let cu_seqlens_k_repr: Ghost<Seq<int>> = Ghost(RT::u64_seq_to_int_repr(cu_k_values@));

    let empty_scope: Ghost<Set<RT::TensorId>> = Ghost(Set::<RT::TensorId>::empty());
    let (input_ids, input_ids_perm) = RT::token_tensor(input_values, device_anchor, empty_scope);
    let scope1: Ghost<Set<RT::TensorId>> = Ghost(empty_scope@.insert(input_ids.id()));
    let (positions, positions_perm) = RT::position_tensor(position_values, device_anchor, scope1);
    let scope2: Ghost<Set<RT::TensorId>> = Ghost(scope1@.insert(positions.id()));
    let (slot_mapping, slot_mapping_perm) = RT::slot_tensor(slot_values, device_anchor, scope2);
    let scope3: Ghost<Set<RT::TensorId>> = Ghost(scope2@.insert(slot_mapping.id()));
    let (block_table, block_table_perm) = RT::block_tables_tensor(block_rows, device_anchor, scope3);
    let scope4: Ghost<Set<RT::TensorId>> = Ghost(scope3@.insert(block_table.id()));
    let (cu_seqlens_q, cu_seqlens_q_perm) = RT::seq_lens_tensor(cu_q_values, device_anchor, scope4);
    let scope5: Ghost<Set<RT::TensorId>> = Ghost(scope4@.insert(cu_seqlens_q.id()));
    let (cu_seqlens_k, cu_seqlens_k_perm) = RT::seq_lens_tensor(cu_k_values, device_anchor, scope5);

    let plan = StepPlan {
        scheduled_ids,
        sample_mask,
        mode,
        input_ids,
        positions,
        block_table,
        slot_mapping,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        input_ids_repr,
        positions_repr,
        cu_seqlens_q_repr,
        cu_seqlens_k_repr,
        block_table_repr,
        slot_mapping_repr,
    };
    let tracked perms = StepPlanPerms {
        input_ids: input_ids_perm.get(),
        positions: positions_perm.get(),
        block_table: block_table_perm.get(),
        slot_mapping: slot_mapping_perm.get(),
        cu_seqlens_q: cu_seqlens_q_perm.get(),
        cu_seqlens_k: cu_seqlens_k_perm.get(),
    };
    assert(step_plan_perms_valid(&plan, perms));
    (plan, Tracked(perms))
}



// Rolling content hash over `tokens[start..end)` (FNV-1a over u64 tokens,
// chained through `prev`).  Correctness never depends on this function's
// distribution: reuse checks the candidate's exact physical predecessor
// chain as well as its page tokens.  The hash is only an index, so no spec is
// needed beyond exec determinism.
pub fn chain_hash_span(prev: u64, tokens: &Vec<TokenId>, start: usize, end: usize) -> u64
    requires
        start <= end,
        end as int <= tokens@.len(),
{
    let mut h: u64 = prev ^ 0xcbf29ce484222325u64;
    let mut i: usize = start;
    while i < end
        invariant
            start <= i <= end,
            end as int <= tokens@.len(),
        decreases end - i,
    {
        let mixed: u64 = h ^ tokens[i];
        let widened: u128 = (mixed as u128) * 0x100000001b3u128;
        h = (widened % 0x1_0000_0000_0000_0000u128) as u64;
        i += 1;
    }
    h
}

// ---------------------------------------------------------------------------
// Methods.
// ---------------------------------------------------------------------------


} // verus!
