// Verified planning, headroom, and forward-layout invariants.

use super::*;

verus! {
// Headroom for one commit-time append: if `rid`'s tail block is exactly
// full, a fresh block must be available.  Without it the executable
// `append_token` silently no-ops while the machine history advances — the
// pool-exhaustion corner the alignment audit surfaced (stale slot_mapping
// would then overwrite an old KV slot).
pub open spec fn append_headroom(cs: &CacheScheduler, rid: RequestId) -> bool {
    (cs.request_residency@.contains_key(rid)
        && cs.request_residency@[rid].block_ids@.len() > 0
        && cs.blocks@.contains_key(cs.request_residency@[rid].block_ids@[
            cs.request_residency@[rid].block_ids@.len() - 1])
        && cs.blocks@[cs.request_residency@[rid].block_ids@[
            cs.request_residency@[rid].block_ids@.len() - 1]].tokens@.len()
            == BLOCK_SIZE_SPEC as int)
    ==> cs.free_blocks > 0
}

// Aggregate commit headroom: one fresh block per scheduled request whose
// history is block-aligned (⟺ full tail, via the alignment companion).
pub open spec fn full_tail_debt(
    live: Map<RequestId, RequestState>,
    ids: Seq<RequestId>,
) -> int
    decreases ids.len(),
{
    if ids.len() == 0 {
        0
    } else {
        (if live.contains_key(ids[0])
            && history(live[ids[0]]).len() as int % (BLOCK_SIZE_SPEC as int) == 0 {
            1int
        } else {
            0int
        }) + full_tail_debt(live, ids.subrange(1, ids.len() as int))
    }
}

pub open spec fn commit_headroom(cs: &CacheScheduler, ids: Seq<RequestId>) -> bool {
    full_tail_debt(cs.live_requests@, ids) <= cs.free_blocks as int
}

// Loop-carried form of the slot-segment fact, over the
// raw value vectors while the plan is being built.  The per-row body is a
// named predicate so the quantifier folds/unfolds through one application;
// nested quantifiers resist definitional folding.
pub open spec fn plan_seg_at(
    pre: &CacheScheduler,
    post: &CacheScheduler,
    sched: Seq<RequestId>,
    cu_q: Seq<u64>,
    cu_k: Seq<u64>,
    slots: Seq<u64>,
    k: int,
) -> bool {
    let srid = sched[k];
    &&& (cu_q[k + 1] as int) <= slots.len()
    &&& post.request_residency@.contains_key(srid)
    &&& post.live_requests@.contains_key(srid)
    &&& (if pre.running@.contains(srid) {
        cu_q[k + 1] as int == cu_q[k] as int + 1
        && slots[cu_q[k] as int] as int
            == crate::proof::tensor::geometry::block_table_slot(
                post.request_residency@[srid].block_ids@,
                (history(post.live_requests@[srid]).len() - 1) as nat,
            ) as int
    } else {
        let c = post.request_residency@[srid].cached_prefix_blocks as int
            * (BLOCK_SIZE_SPEC as int);
        let end = cu_k[k + 1] as int - cu_k[k] as int;
        cu_q[k + 1] as int == cu_q[k] as int + (end - c)
        && 0 <= c < end
        && (forall|q: int|
            cu_q[k] as int <= q < cu_q[k + 1] as int
            ==> #[trigger] slots[q] as int
                == crate::proof::tensor::geometry::block_table_slot(
                    post.request_residency@[srid].block_ids@,
                    (c + q - cu_q[k] as int) as nat) as int)
    })
}

pub open spec fn plan_seg_inv(
    pre: &CacheScheduler,
    post: &CacheScheduler,
    sched: Seq<RequestId>,
    cu_q: Seq<u64>,
    cu_k: Seq<u64>,
    slots: Seq<u64>,
) -> bool {
    forall|k: int| 0 <= k < sched.len()
        ==> #[trigger] plan_seg_at(pre, post, sched, cu_q, cu_k, slots, k)
}

// Full forward-row payload while the executable plan is still represented by
// raw u64 vectors.  Unlike `plan_seg_at` (which only characterizes writes),
// this also pins the query tokens/positions, cumulative key length, and block
// table row to the scheduled request.  These are precisely the facts needed
// to instantiate the request-isolation and prefix-fidelity theorems.
pub open spec fn plan_data_at_intro_ready(
    pre: &CacheScheduler,
    post: &CacheScheduler,
    sched: Seq<RequestId>,
    inputs: Seq<u64>,
    positions: Seq<u64>,
    cu_q: Seq<u64>,
    cu_k: Seq<u64>,
    block_rows: Seq<Vec<u64>>,
    k: int,
) -> bool {
    let rid = sched[k];
    let s0 = cu_q[k] as int;
    let s1 = cu_q[k + 1] as int;
    let h = history(post.live_requests@[rid]);
    &&& post.live_requests@.contains_key(rid)
    &&& valid_request_state(post.live_requests@[rid])
    &&& post.request_residency@.contains_key(rid)
    &&& block_rows[k]@ == post.request_residency@[rid].block_ids@
    &&& 0 <= s0
    &&& s1 <= inputs.len()
    &&& positions.len() == inputs.len()
    &&& (if pre.running@.contains(rid) {
        s1 == s0 + 1
        && h.len() > 0
        && inputs[s0] == h[h.len() - 1]
        && positions[s0] as int == h.len() as int - 1
        && cu_k[k + 1] as int == cu_k[k] as int + h.len() as int
    } else {
        let c = post.request_residency@[rid].cached_prefix_blocks as int
            * (BLOCK_SIZE_SPEC as int);
        let n = post.live_requests@[rid].prompt_tokens@.len() as int;
        let end = cu_k[k + 1] as int - cu_k[k] as int;
        s1 == s0 + (end - c)
        && 0 <= c < end
        && end <= n
        && (forall|q: int| s0 <= q < s1 ==> {
            let p = c + q - s0;
            &&& #[trigger] inputs[q]
                == post.live_requests@[rid].prompt_tokens@[p]
            &&& positions[q] as int == p
        })
    })
}

pub open spec fn plan_data_at(
    pre: &CacheScheduler,
    post: &CacheScheduler,
    sched: Seq<RequestId>,
    inputs: Seq<u64>,
    positions: Seq<u64>,
    cu_q: Seq<u64>,
    cu_k: Seq<u64>,
    block_rows: Seq<Vec<u64>>,
    k: int,
) -> bool {
    plan_data_at_intro_ready(
        pre, post, sched, inputs, positions, cu_q, cu_k, block_rows, k,
    )
}

// Introduction rule kept separate from the solver-heavy admission loop.  Its
// premise is exactly the open predicate body; isolating the definitional fold
// prevents unrelated cache-domain quantifiers from obscuring this otherwise
// propositional step.
pub proof fn lemma_plan_data_at_intro(
    pre: &CacheScheduler,
    post: &CacheScheduler,
    sched: Seq<RequestId>,
    inputs: Seq<u64>,
    positions: Seq<u64>,
    cu_q: Seq<u64>,
    cu_k: Seq<u64>,
    block_rows: Seq<Vec<u64>>,
    k: int,
)
    requires
        plan_data_at_intro_ready(
            pre, post, sched, inputs, positions, cu_q, cu_k, block_rows, k,
        ),
    ensures
        plan_data_at(
            pre, post, sched, inputs, positions, cu_q, cu_k, block_rows, k,
        ),
{
}

pub open spec fn plan_data_inv(
    pre: &CacheScheduler,
    post: &CacheScheduler,
    sched: Seq<RequestId>,
    inputs: Seq<u64>,
    positions: Seq<u64>,
    cu_q: Seq<u64>,
    cu_k: Seq<u64>,
    block_rows: Seq<Vec<u64>>,
) -> bool {
    forall|k: int| 0 <= k < sched.len() ==>
        #[trigger] plan_data_at(
            pre, post, sched, inputs, positions, cu_q, cu_k, block_rows, k,
        )
}

// Row-local effect policy while plan tensors are still raw u64 vectors. Keep
// the quantifier opaque so the solver-heavy admission loop carries one stable
// fact instead of repeatedly instantiating sampling/finality arithmetic.
pub open spec fn raw_plan_sample_policy_at(
    pre: &CacheScheduler,
    sched: Seq<RequestId>,
    cu_k: Seq<u64>,
    sample_mask: Seq<bool>,
    k: int,
) -> bool {
    let rid = sched[k];
    let end = cu_k[k + 1] as int - cu_k[k] as int;
    &&& pre.live_requests@.contains_key(rid)
    &&& if pre.running@.contains(rid) {
        sample_mask[k]
    } else {
        pre.waiting@.contains(rid)
        && (sample_mask[k]
            <==> end == pre.live_requests@[rid].prompt_tokens@.len())
    }
}

#[verifier::opaque]
pub open spec fn raw_plan_sample_policy(
    pre: &CacheScheduler,
    sched: Seq<RequestId>,
    cu_k: Seq<u64>,
    sample_mask: Seq<bool>,
) -> bool {
    forall|k: int| 0 <= k < sched.len()
        ==> #[trigger] raw_plan_sample_policy_at(
            pre, sched, cu_k, sample_mask, k,
        )
}

pub open spec fn plan_sample_policy_at(
    pre: &CacheScheduler,
    plan: &StepPlan,
    k: int,
) -> bool {
    let rid = plan.scheduled_ids@[k];
    let end = plan.cu_seqlens_k_repr@[k + 1]
        - plan.cu_seqlens_k_repr@[k];
    &&& pre.live_requests@.contains_key(rid)
    &&& if pre.running@.contains(rid) {
        plan.sample_mask@[k]
    } else {
        pre.waiting@.contains(rid)
        && (plan.sample_mask@[k]
            <==> end == pre.live_requests@[rid].prompt_tokens@.len())
    }
}

#[verifier::opaque]
pub open spec fn plan_sample_policy(
    pre: &CacheScheduler,
    plan: &StepPlan,
) -> bool {
    plan.sample_mask@.len() == plan.scheduled_ids@.len()
    && forall|k: int| 0 <= k < plan.scheduled_ids@.len()
        ==> #[trigger] plan_sample_policy_at(pre, plan, k)
}

pub proof fn lemma_raw_plan_sample_policy_decode(
    pre: &CacheScheduler,
    sched: Seq<RequestId>,
    cu_k: Seq<u64>,
    sample_mask: Seq<bool>,
)
    requires
        cu_k.len() == sched.len() + 1,
        sample_mask.len() == sched.len(),
        forall|k: int| 0 <= k < sched.len()
            ==> pre.running@.contains(#[trigger] sched[k]),
        live_covers_queue(pre),
        forall|k: int| 0 <= k < sample_mask.len()
            ==> #[trigger] sample_mask[k],
    ensures
        raw_plan_sample_policy(pre, sched, cu_k, sample_mask),
{
    reveal(raw_plan_sample_policy);
    reveal(live_covers_queue);
    assert forall|k: int| 0 <= k < sched.len() implies
        #[trigger] raw_plan_sample_policy_at(
            pre, sched, cu_k, sample_mask, k,
        )
    by {
        assert(pre.running@.contains(sched[k]));
        assert(pre.live_requests@.contains_key(sched[k]));
        reveal(raw_plan_sample_policy_at);
    }
}

pub proof fn lemma_raw_plan_sample_policy_push_admission(
    pre: &CacheScheduler,
    sched: Seq<RequestId>,
    cu_k: Seq<u64>,
    sample_mask: Seq<bool>,
    rid: RequestId,
    end: u64,
    next_k: u64,
    sample: bool,
)
    requires
        cu_k.len() == sched.len() + 1,
        sample_mask.len() == sched.len(),
        raw_plan_sample_policy(pre, sched, cu_k, sample_mask),
        pre.waiting@.contains(rid),
        !pre.running@.contains(rid),
        pre.live_requests@.contains_key(rid),
        0 < end as int <= pre.live_requests@[rid].prompt_tokens@.len(),
        next_k as int
            == cu_k[cu_k.len() - 1] as int + end as int,
        sample <==> end as int
            == pre.live_requests@[rid].prompt_tokens@.len(),
    ensures
        raw_plan_sample_policy(
            pre,
            sched.push(rid),
            cu_k.push(next_k),
            sample_mask.push(sample),
        ),
{
    reveal(raw_plan_sample_policy);
    assert(cu_k.len() > 0);
    assert(cu_k.push(next_k).len() == sched.push(rid).len() + 1);
    assert(sample_mask.push(sample).len() == sched.push(rid).len());
    assert forall|k: int| 0 <= k < sched.push(rid).len() implies
        #[trigger] raw_plan_sample_policy_at(
            pre,
            sched.push(rid),
            cu_k.push(next_k),
            sample_mask.push(sample),
            k,
        )
    by {
        if k < sched.len() {
            assert(raw_plan_sample_policy_at(
                pre, sched, cu_k, sample_mask, k,
            ));
            assert(sched.push(rid)[k] == sched[k]);
            assert(sample_mask.push(sample)[k] == sample_mask[k]);
            assert(cu_k.push(next_k)[k] == cu_k[k]);
            assert(cu_k.push(next_k)[k + 1] == cu_k[k + 1]);
            reveal(raw_plan_sample_policy_at);
        } else {
            assert(k == sched.len());
            assert(sched.push(rid)[k] == rid);
            assert(sample_mask.push(sample)[k] == sample);
            assert(cu_k.push(next_k)[k] == cu_k[cu_k.len() - 1]);
            assert(cu_k.push(next_k)[k + 1] == next_k);
            assert(cu_k.push(next_k)[k + 1] as int
                - cu_k.push(next_k)[k] as int == end as int);
            assert(pre.live_requests@.contains_key(rid));
            reveal(raw_plan_sample_policy_at);
        }
    }
    assert(raw_plan_sample_policy(
        pre,
        sched.push(rid),
        cu_k.push(next_k),
        sample_mask.push(sample),
    ));
}

pub proof fn lemma_plan_sample_policy_from_raw(
    pre: &CacheScheduler,
    plan: &StepPlan,
    sched: Seq<RequestId>,
    cu_k: Seq<u64>,
    sample_mask: Seq<bool>,
)
    requires
        cu_k.len() == sched.len() + 1,
        sample_mask.len() == sched.len(),
        raw_plan_sample_policy(pre, sched, cu_k, sample_mask),
        plan.scheduled_ids@ == sched,
        plan.sample_mask@ == sample_mask,
        plan.cu_seqlens_k_repr@ == RT::u64_seq_to_int_repr(cu_k),
    ensures
        plan_sample_policy(pre, plan),
{
    reveal(raw_plan_sample_policy);
    reveal(plan_sample_policy);
    assert(plan.sample_mask@.len() == plan.scheduled_ids@.len());
    assert forall|k: int| 0 <= k < plan.scheduled_ids@.len() implies
        #[trigger] plan_sample_policy_at(pre, plan, k)
    by {
        assert(raw_plan_sample_policy_at(
            pre, sched, cu_k, sample_mask, k,
        ));
        assert(plan.scheduled_ids@[k] == sched[k]);
        assert(plan.sample_mask@[k] == sample_mask[k]);
        assert(RT::u64_seq_to_int_repr(cu_k)[k] == cu_k[k] as int);
        assert(RT::u64_seq_to_int_repr(cu_k)[k + 1]
            == cu_k[k + 1] as int);
        reveal(raw_plan_sample_policy_at);
        reveal(plan_sample_policy_at);
    }
    assert(plan_sample_policy(pre, plan));
}

// `plan_data_inv` reads only the post-state live and residency maps.
pub proof fn lemma_plan_data_inv_frame(
    pre: &CacheScheduler,
    a: &CacheScheduler,
    b: &CacheScheduler,
    sched: Seq<RequestId>,
    inputs: Seq<u64>,
    positions: Seq<u64>,
    cu_q: Seq<u64>,
    cu_k: Seq<u64>,
    block_rows: Seq<Vec<u64>>,
)
    requires
        plan_data_inv(
            pre, a, sched, inputs, positions, cu_q, cu_k, block_rows,
        ),
        b.request_residency@ == a.request_residency@,
        b.live_requests@ == a.live_requests@,
    ensures
        plan_data_inv(
            pre, b, sched, inputs, positions, cu_q, cu_k, block_rows,
        ),
{
    assert forall|k: int| 0 <= k < sched.len() implies
        #[trigger] plan_data_at(
            pre, b, sched, inputs, positions, cu_q, cu_k, block_rows, k,
        )
    by {
        assert(plan_data_at(
            pre, a, sched, inputs, positions, cu_q, cu_k, block_rows, k,
        ));
        let rid = sched[k];
        if !pre.running@.contains(rid) {
            let c = b.request_residency@[rid].cached_prefix_blocks as int
                * (BLOCK_SIZE_SPEC as int);
            let s0 = cu_q[k] as int;
            let s1 = cu_q[k + 1] as int;
            assert forall|q: int| s0 <= q < s1 implies {
                let p = c + q - s0;
                &&& #[trigger] inputs[q]
                    == b.live_requests@[rid].prompt_tokens@[p]
                &&& positions[q] as int == p
            } by {}
        }
    }
}

// plan_seg_inv only reads `post`'s residency and live maps.
pub proof fn lemma_plan_seg_inv_frame(
    pre: &CacheScheduler,
    a: &CacheScheduler,
    b: &CacheScheduler,
    sched: Seq<RequestId>,
    cu_q: Seq<u64>,
    cu_k: Seq<u64>,
    slots: Seq<u64>,
)
    requires
        plan_seg_inv(pre, a, sched, cu_q, cu_k, slots),
        b.request_residency@ == a.request_residency@,
        b.live_requests@ == a.live_requests@,
    ensures
        plan_seg_inv(pre, b, sched, cu_q, cu_k, slots),
{
    assert forall|k: int| 0 <= k < sched.len()
        implies #[trigger] plan_seg_at(pre, b, sched, cu_q, cu_k, slots, k)
    by {
        assert(plan_seg_at(pre, a, sched, cu_q, cu_k, slots, k));
        let srid = sched[k];
        if !pre.running@.contains(srid) {
            let c = b.request_residency@[srid].cached_prefix_blocks as int
                * (BLOCK_SIZE_SPEC as int);
            assert forall|q: int|
                cu_q[k] as int <= q < cu_q[k + 1] as int
                implies #[trigger] slots[q] as int
                    == crate::proof::tensor::geometry::block_table_slot(
                        b.request_residency@[srid].block_ids@,
                        (c + q - cu_q[k] as int) as nat) as int
            by {
                assert(slots[q] as int
                    == crate::proof::tensor::geometry::block_table_slot(
                        a.request_residency@[srid].block_ids@,
                        (c + q - cu_q[k] as int) as nat) as int);
            }
        }
    }
}

// The plan's per-row slot segments, read off the plan-exit
// scheduler state. Decode rows (pre-plan running) carry exactly one slot:
// the block-table slot of the request's last history position; admitted rows
// carry the uncached suffix pointwise at its global block-table positions.
#[verifier::opaque]
pub open spec fn plan_slot_segments_at(
    pre: &CacheScheduler,
    post: &CacheScheduler,
    plan: &StepPlan,
    k: int,
) -> bool {
    let srid = plan.scheduled_ids@[k];
    let s0 = plan.cu_seqlens_q_repr@[k];
    let s1 = plan.cu_seqlens_q_repr@[k + 1];
    &&& post.request_residency@.contains_key(srid)
    &&& post.live_requests@.contains_key(srid)
    &&& plan.block_table_repr@[k]
        == post.request_residency@[srid].block_ids@
    &&& 0 <= s0
    &&& s1 <= plan.slot_mapping_repr@.len() as int
    &&& (if pre.running@.contains(srid) {
        s1 == s0 + 1
        && plan.slot_mapping_repr@[s0]
            == crate::proof::tensor::geometry::block_table_slot(
                post.request_residency@[srid].block_ids@,
                (history(post.live_requests@[srid]).len() - 1) as nat,
            ) as int
    } else {
        let c = post.request_residency@[srid].cached_prefix_blocks as int
            * (BLOCK_SIZE_SPEC as int);
        let end = plan.cu_seqlens_k_repr@[k + 1]
            - plan.cu_seqlens_k_repr@[k];
        s1 == s0 + (end - c)
        && 0 <= c < end
        && (forall|q: int| s0 <= q < s1
            ==> #[trigger] plan.slot_mapping_repr@[q]
                == crate::proof::tensor::geometry::block_table_slot(
                    post.request_residency@[srid].block_ids@,
                    (c + q - s0) as nat) as int)
    })
}

pub open spec fn plan_slot_segments_ok(
    pre: &CacheScheduler,
    post: &CacheScheduler,
    plan: &StepPlan,
) -> bool {
    forall|k: int| 0 <= k < plan.scheduled_ids@.len()
        ==> #[trigger] plan_slot_segments_at(pre, post, plan, k)
}

// Every physical page written by one plan row is exclusively owned at the
// post-plan boundary.  This packages the two scheduler cases behind one
// stable fact: decode writes its refcount-one tail, while an admission writes
// only refcount-one pages at or after its cached-prefix boundary.
pub proof fn lemma_plan_slot_page_exclusive(
    pre: &CacheScheduler,
    post: &CacheScheduler,
    plan: &StepPlan,
    k: int,
    q: int,
)
    requires
        cs_valid(post),
        residency_history_aligned(post),
        pre_commit_tails_exclusive(post),
        super::step_plan_shape_ok(plan),
        post.live_requests@ == pre.live_requests@,
        0 <= k < plan.scheduled_ids@.len(),
        post.running@.contains(plan.scheduled_ids@[k]),
        plan_slot_segments_at(pre, post, plan, k),
        plan_forward_layout_at(pre, plan, k),
        !pre.running@.contains(plan.scheduled_ids@[k]) ==>
            admitted_pages_exclusive_post(post, plan.scheduled_ids@, k),
        plan.cu_seqlens_q_repr@[k] <= q
            < plan.cu_seqlens_q_repr@[k + 1],
    ensures ({
        let rid = plan.scheduled_ids@[k];
        let slot = plan.slot_mapping_repr@[q];
        let bid = (slot / (BLOCK_SIZE_SPEC as int)) as BlockId;
        &&& slot >= 0
        &&& post.request_residency@[rid].block_ids@.contains(bid)
        &&& post.blocks@.contains_key(bid)
        &&& post.blocks@[bid].refcount == 1
    }),
{
    let rid = plan.scheduled_ids@[k];
    let s0 = plan.cu_seqlens_q_repr@[k];
    let s1 = plan.cu_seqlens_q_repr@[k + 1];
    let ids = post.request_residency@[rid].block_ids@;
    let slot = plan.slot_mapping_repr@[q];
    let bs = BLOCK_SIZE_SPEC as int;
    reveal(plan_slot_segments_at);
    reveal(plan_forward_layout_at);
    if pre.running@.contains(rid) {
        assert(s1 == s0 + 1);
        assert(q == s0);
        let hist = history(post.live_requests@[rid]).len() as int;
        let tail_bid = ids[ids.len() - 1];
        let tail = post.blocks@[tail_bid].tokens@.len() as int;
        assert(hist == (ids.len() - 1) * bs + tail);
        assert(1 <= tail <= bs) by {
            assert(block_token_bound(post));
        }
        assert((hist - 1) / bs == ids.len() - 1) by {
            vstd::arithmetic::div_mod::lemma_fundamental_div_mod_converse_div(
                hist - 1, bs, ids.len() - 1, tail - 1,
            );
        }
        crate::proof::tensor::geometry::block_table_slot_block(ids, (hist - 1) as nat);
        assert(slot == crate::proof::tensor::geometry::block_table_slot(
            ids, (hist - 1) as nat,
        ) as int);
        assert(slot / bs == tail_bid as int);
        assert((slot / bs) as BlockId == tail_bid);
        assert(pre_commit_tails_exclusive(post));
    } else {
        let c_blocks = post.request_residency@[rid]
            .cached_prefix_blocks as int;
        let c = c_blocks * bs;
        let end = plan.cu_seqlens_k_repr@[k + 1]
            - plan.cu_seqlens_k_repr@[k];
        let pos = c + q - s0;
        assert(s1 == s0 + (end - c));
        assert(0 <= pos < end);
        assert(plan.block_table_repr@[k] == ids);
        assert(blocks_needed_for(end as nat)
            <= plan.block_table_repr@[k].len());
        crate::proof::tensor::geometry::lemma_blocks_needed_covers_pos(pos as nat, end as nat);
        assert(0 <= pos / bs < ids.len());
        assert(c_blocks <= pos / bs) by {
            vstd::arithmetic::div_mod::lemma_div_is_ordered(c, pos, bs);
            assert(c / bs == c_blocks) by (nonlinear_arith)
                requires bs == 64, c == c_blocks * 64,
            {}
        }
        crate::proof::tensor::geometry::block_table_slot_block(ids, pos as nat);
        assert(slot == crate::proof::tensor::geometry::block_table_slot(ids, pos as nat) as int);
        assert(slot / bs == ids[pos / bs] as int);
        assert((slot / bs) as BlockId == ids[pos / bs]);
        assert(ids.contains(ids[pos / bs]));
        assert(admitted_pages_exclusive_post(post, plan.scheduled_ids@, k));
        assert(post.blocks@[ids[pos / bs]].refcount == 1);
    }
}

// Stable, materialized forward-row contract.  It intentionally refers only
// to the pre-plan request state plus the plan tensors, so it can be transported
// across commit (which appends one sampled token and may remove finished
// requests).  For admitted rows, `c = k_len - q_len` recovers the prefix that
// precedes this row. `k_len` may be a proper prompt prefix for a KV-only chunk.
pub open spec fn plan_forward_layout_at(
    pre: &CacheScheduler,
    plan: &StepPlan,
    k: int,
) -> bool {
    let rid = plan.scheduled_ids@[k];
    let s0 = plan.cu_seqlens_q_repr@[k];
    let s1 = plan.cu_seqlens_q_repr@[k + 1];
    let kd = plan.cu_seqlens_k_repr@[k + 1]
        - plan.cu_seqlens_k_repr@[k];
    &&& pre.live_requests@.contains_key(rid)
    &&& valid_request_state(pre.live_requests@[rid])
    &&& 0 <= s0 < s1
    &&& s1 <= plan.input_ids_repr@.len() as int
    &&& crate::proof::tensor::geometry::blocks_needed_for(kd as nat)
        <= plan.block_table_repr@[k].len()
    &&& (if pre.running@.contains(rid) {
        let h = history(pre.live_requests@[rid]);
        pre.request_residency@.contains_key(rid)
        && plan.block_table_repr@[k]
            == pre.request_residency@[rid].block_ids@
        && plan.block_table_repr@[k].len()
            == crate::proof::tensor::geometry::blocks_needed_for(h.len())
        && s1 == s0 + 1
        && kd == h.len() as int
        && plan.input_ids_repr@[s0] == h[h.len() - 1] as int
        && plan.positions_repr@[s0] == h.len() as int - 1
    } else {
        let n = pre.live_requests@[rid].prompt_tokens@.len() as int;
        let c = kd - (s1 - s0);
        pre.waiting@.contains(rid)
        && plan.block_table_repr@[k].len()
            == crate::proof::tensor::geometry::blocks_needed_for(n as nat)
        && 0 <= c < kd
        && kd <= n
        && (forall|q: int| s0 <= q < s1 ==> {
            let p = c + q - s0;
            &&& #[trigger] plan.input_ids_repr@[q]
                == pre.live_requests@[rid].prompt_tokens@[p] as int
            &&& plan.positions_repr@[q] == p
        })
    })
}

pub open spec fn plan_forward_layout_ok(
    pre: &CacheScheduler,
    plan: &StepPlan,
) -> bool {
    forall|k: int| 0 <= k < plan.scheduled_ids@.len() ==>
        #[trigger] plan_forward_layout_at(pre, plan, k)
}

// Transient physical coverage of one planned row. The scheduler may reserve
// the complete prompt residency even when this step computes only a proper
// prefix; the row therefore needs coverage through `end`, not equality
// between `end` and the physical residency extent.
pub open spec fn plan_residency_extent_at(
    pre: &CacheScheduler,
    post: &CacheScheduler,
    plan: &StepPlan,
    k: int,
) -> bool {
    let rid = plan.scheduled_ids@[k];
    let end = plan.cu_seqlens_k_repr@[k + 1]
        - plan.cu_seqlens_k_repr@[k];
    let ids = post.request_residency@[rid].block_ids@;
    let tail = ids[ids.len() - 1];
    let tokens = if pre.running@.contains(rid) {
        history(pre.live_requests@[rid])
    } else {
        pre.live_requests@[rid].prompt_tokens@
    };
    &&& post.request_residency@.contains_key(rid)
    &&& plan.block_table_repr@[k] == ids
    &&& ids.len() >= 1
    &&& ids.len() == blocks_needed_for(tokens.len())
    &&& post.blocks@.contains_key(tail)
    &&& 0 < end
    &&& blocks_needed_for(end as nat) <= ids.len()
    &&& 1 <= post.blocks@[tail].tokens@.len()
    &&& post.blocks@[tail].tokens@.len() <= BLOCK_SIZE_SPEC as int
    &&& end <= tokens.len()
    &&& token_placement_prefix(post.blocks@, ids, tokens, end)
}

#[verifier::opaque]
pub open spec fn plan_residency_extents(
    pre: &CacheScheduler,
    post: &CacheScheduler,
    plan: &StepPlan,
) -> bool {
    forall|k: int| 0 <= k < plan.scheduled_ids@.len() ==>
        #[trigger] plan_residency_extent_at(pre, post, plan, k)
}

// Discharge the unary metadata/index portion of the checked paged-attention
// annotation from the scheduler's materialized plan.  Physical head geometry
// is handled separately by the engine cache-shape invariant; rectangular
// padding is performed and guarded by the trusted Python materializer.
pub proof fn lemma_plan_paged_attention_metadata_ready(
    pre: &CacheScheduler,
    post: &CacheScheduler,
    plan: &StepPlan,
)
    requires
        plan.scheduled_ids@.len() > 0,
        super::step_plan_shape_ok(plan),
        plan_forward_layout_ok(pre, plan),
        cs_valid(post),
        post.num_blocks == pre.num_blocks,
        forall|r: RequestId| #[trigger] plan.scheduled_ids@.contains(r)
            ==> post.running@.contains(r),
        plan_slot_segments_ok(pre, post, plan),
    ensures
        pre.num_blocks > 0,
        RT::paged_attention_metadata_ready(
            plan.input_ids_repr@.len(), pre.num_blocks as nat,
            plan.cu_seqlens_q_repr@, plan.cu_seqlens_k_repr@,
            plan.max_seqlen_q as nat, plan.max_seqlen_k as nat,
            plan.block_table_repr@,
        ),
{
    reveal(super::step_plan_shape_ok);
    reveal(RT::paged_attention_metadata_ready);
    let b = plan.scheduled_ids@.len() as int;
    assert(b > 0);
    assert(0 < plan.scheduled_ids@.len());
    assert(0 <= 0 < plan.scheduled_ids@.len() as int);
    assert(plan.block_table_repr@.len() == plan.scheduled_ids@.len());
    assert forall|j: int| 0 <= j < plan.scheduled_ids@.len() as int
        implies plan.cu_seqlens_q_repr@[j]
            < #[trigger] plan.cu_seqlens_q_repr@[j + 1]
    by {}
    assert forall|j: int| 0 <= j < plan.scheduled_ids@.len() as int
        implies plan.cu_seqlens_k_repr@[j]
            < #[trigger] plan.cu_seqlens_k_repr@[j + 1]
    by {}
    let j0: int = 0;
    assert(plan.cu_seqlens_q_repr@[j0]
        < plan.cu_seqlens_q_repr@[j0 + 1]);
    assert(plan.cu_seqlens_k_repr@[j0]
        < plan.cu_seqlens_k_repr@[j0 + 1]);
    assert(plan.input_ids_repr@.len() > 0) by {
        crate::proof::tensor::geometry::lemma_cu_mono(
            plan.cu_seqlens_q_repr@, b, 1, b,
        );
    }
    assert(plan.max_seqlen_q > 0) by {
        let q_len = plan.cu_seqlens_q_repr@[1]
            - plan.cu_seqlens_q_repr@[0];
        assert(q_len > 0);
        assert(q_len <= plan.max_seqlen_q as int);
    }
    assert(plan.max_seqlen_k > 0) by {
        let k_len = plan.cu_seqlens_k_repr@[1]
            - plan.cu_seqlens_k_repr@[0];
        assert(k_len > 0);
        assert(k_len <= plan.max_seqlen_k as int);
    }
    assert(pre.num_blocks > 0) by {
        assert(plan_forward_layout_at(pre, plan, j0));
        assert(plan_slot_segments_at(pre, post, plan, j0));
        reveal(plan_forward_layout_at);
        reveal(plan_slot_segments_at);
        let k_len = plan.cu_seqlens_k_repr@[1]
            - plan.cu_seqlens_k_repr@[0];
        assert(k_len > 0);
        assert(crate::proof::tensor::geometry::blocks_needed_for(k_len as nat)
            <= plan.block_table_repr@[0].len());
        reveal(crate::proof::tensor::geometry::blocks_needed_for);
        assert(plan.block_table_repr@[0].len() > 0);
        let l0: int = 0;
        let rid = plan.scheduled_ids@[j0];
        assert(plan.scheduled_ids@.contains(rid));
        assert(post.running@.contains(rid));
        assert(plan.block_table_repr@[j0]
            == post.request_residency@[rid].block_ids@);
        let bid = plan.block_table_repr@[j0][l0];
        assert(post.request_residency@[rid].block_ids@.contains(bid));
        assert(post.blocks@.contains_key(bid)) by {
            assert(residency_blocks_in_range(post));
        }
        assert(bid < post.num_blocks) by {
            assert(blocks_dom_in_range(post));
        }
    }
    assert forall|j: int| 0 <= j < plan.block_table_repr@.len() as int
        implies {
            let q_len = plan.cu_seqlens_q_repr@[j + 1]
                - plan.cu_seqlens_q_repr@[j];
            let k_len = plan.cu_seqlens_k_repr@[j + 1]
                - plan.cu_seqlens_k_repr@[j];
            &&& plan.cu_seqlens_q_repr@[j]
                < #[trigger] plan.cu_seqlens_q_repr@[j + 1]
            &&& plan.cu_seqlens_k_repr@[j]
                < #[trigger] plan.cu_seqlens_k_repr@[j + 1]
            &&& q_len <= plan.max_seqlen_q as int
            &&& k_len <= plan.max_seqlen_k as int
            &&& q_len <= k_len
            &&& crate::proof::tensor::geometry::blocks_needed_for(k_len as nat)
                <= plan.block_table_repr@[j].len()
            &&& (forall|l: int| 0 <= l < plan.block_table_repr@[j].len() ==>
                #[trigger] plan.block_table_repr@[j][l] < pre.num_blocks as nat)
        }
    by {
        assert(j < plan.scheduled_ids@.len());
        assert(plan_forward_layout_at(pre, plan, j));
        assert(plan_slot_segments_at(pre, post, plan, j));
        reveal(plan_slot_segments_at);
        let rid = plan.scheduled_ids@[j];
        assert(plan.scheduled_ids@.contains(rid));
        assert(post.running@.contains(rid));
        assert(plan.block_table_repr@[j]
            == post.request_residency@[rid].block_ids@);
        assert forall|l: int| 0 <= l < plan.block_table_repr@[j].len()
            implies #[trigger] plan.block_table_repr@[j][l]
                < pre.num_blocks as nat
        by {
            let bid = plan.block_table_repr@[j][l];
            assert(post.request_residency@[rid].block_ids@.contains(bid));
            assert(post.blocks@.contains_key(bid)) by {
                assert(residency_blocks_in_range(post));
            }
            assert(bid < post.num_blocks) by {
                assert(blocks_dom_in_range(post));
            }
        }
    }
}

// Stable plan-facing form of the cached-admission origin. The number of cached
// pages is recovered from the omitted query prefix, so this contract refers
// only to the pre-plan state and materialized plan tensors and survives commit.
pub open spec fn plan_cached_prefix_origin_at(
    pre: &CacheScheduler,
    plan: &StepPlan,
    k: int,
) -> bool {
    let rid = plan.scheduled_ids@[k];
    if pre.running@.contains(rid) {
        true
    } else {
        let q = plan.cu_seqlens_q_repr@[k + 1] - plan.cu_seqlens_q_repr@[k];
        let kd = plan.cu_seqlens_k_repr@[k + 1]
            - plan.cu_seqlens_k_repr@[k];
        let c_tokens = kd - q;
        let c = c_tokens / (BLOCK_SIZE_SPEC as int);
        let ids = plan.block_table_repr@[k];
        &&& pre.live_requests@.contains_key(rid)
        &&& 0 <= c_tokens
        &&& c_tokens % (BLOCK_SIZE_SPEC as int) == 0
        &&& 0 <= c <= ids.len()
        &&& registered_prefix_chain(pre.blocks@, ids.subrange(0, c))
        &&& token_placement_prefix(
            pre.blocks@, ids, pre.live_requests@[rid].prompt_tokens@,
            c_tokens,
        )
        &&& forall|l: int| #![trigger ids[l]] 0 <= l < c ==> {
            let bid = ids[l];
            &&& pre.blocks@.contains_key(bid)
            &&& pre.hash_to_block@.contains_key(pre.blocks@[bid].hash_value)
            &&& pre.hash_to_block@[pre.blocks@[bid].hash_value] == bid
        }
    }
}

#[verifier::opaque]
pub open spec fn plan_cached_prefix_origins(
    pre: &CacheScheduler,
    plan: &StepPlan,
) -> bool {
    forall|k: int| 0 <= k < plan.scheduled_ids@.len() ==>
        #[trigger] plan_cached_prefix_origin_at(pre, plan, k)
}

// Materialization bridge for the loop-carried admission-origin invariant.  In
// the stable plan contract the cached-page count is reconstructed from the
// omitted query prefix; `plan_data_inv` proves that this is exactly the
// scheduler residency's `cached_prefix_blocks * BLOCK_SIZE`.
pub proof fn lemma_plan_cached_prefix_origins_from_raw(
    pre: &CacheScheduler,
    post: &CacheScheduler,
    plan: &StepPlan,
    sched: Seq<RequestId>,
    inputs: Seq<u64>,
    positions: Seq<u64>,
    cu_q: Seq<u64>,
    cu_k: Seq<u64>,
    block_rows: Seq<Vec<BlockId>>,
)
    requires
        pre.live_requests@ == post.live_requests@,
        plan_data_inv(
            pre, post, sched, inputs, positions, cu_q, cu_k, block_rows,
        ),
        admitted_prefixes_from_pre(pre, post, sched, block_rows),
        plan.scheduled_ids@ == sched,
        forall|k: int| 0 <= k < sched.len()
            ==> plan.block_table_repr@[k] == #[trigger] block_rows[k]@,
        forall|k: int| 0 <= k <= sched.len()
            ==> plan.cu_seqlens_q_repr@[k] == #[trigger] cu_q[k] as int,
        forall|k: int| 0 <= k <= sched.len()
            ==> plan.cu_seqlens_k_repr@[k] == #[trigger] cu_k[k] as int,
    ensures
        plan_cached_prefix_origins(pre, plan),
{
    reveal(plan_cached_prefix_origins);
    reveal(admitted_prefixes_from_pre);
    assert forall|k: int| 0 <= k < plan.scheduled_ids@.len() implies
        #[trigger] plan_cached_prefix_origin_at(pre, plan, k)
    by {
        let rid = sched[k];
        if !pre.running@.contains(rid) {
            assert(plan_data_at(
                pre, post, sched, inputs, positions, cu_q, cu_k,
                block_rows, k,
            ));
            assert(admitted_prefix_from_pre_at(
                pre, post, sched, block_rows, k,
            ));
            let cpb = post.request_residency@[rid].cached_prefix_blocks as int;
            let bs = BLOCK_SIZE_SPEC as int;
            let q = plan.cu_seqlens_q_repr@[k + 1]
                - plan.cu_seqlens_q_repr@[k];
            let kd = plan.cu_seqlens_k_repr@[k + 1]
                - plan.cu_seqlens_k_repr@[k];
            let c_tokens = kd - q;
            let n = pre.live_requests@[rid].prompt_tokens@.len() as int;
            assert(post.live_requests@[rid].prompt_tokens@.len() as int == n);
            assert(q == kd - cpb * bs);
            assert(0 <= cpb * bs < kd <= n);
            assert(c_tokens == cpb * bs);
            assert(0 <= cpb);
            vstd::arithmetic::div_mod::lemma_fundamental_div_mod_converse_div(
                c_tokens, bs, cpb, 0,
            );
            vstd::arithmetic::div_mod::lemma_fundamental_div_mod_converse_mod(
                c_tokens, bs, cpb, 0,
            );
            assert(c_tokens / bs == cpb);
            assert(c_tokens % bs == 0);
            assert(plan.block_table_repr@[k] == block_rows[k]@);
            assert(registered_prefix_chain(
                pre.blocks@,
                plan.block_table_repr@[k].subrange(0, c_tokens / bs),
            ));
            assert forall|l: int|
                #![trigger plan.block_table_repr@[k][l]]
                0 <= l < c_tokens / bs implies {
                    let bid = plan.block_table_repr@[k][l];
                    &&& pre.blocks@.contains_key(bid)
                    &&& pre.hash_to_block@.contains_key(
                        pre.blocks@[bid].hash_value)
                    &&& pre.hash_to_block@[pre.blocks@[bid].hash_value] == bid
                } by {
                assert(plan.block_table_repr@[k][l] == block_rows[k]@[l]);
            }
        }
    }
}

// Debt decomposes at any processed prefix position.
pub proof fn lemma_debt_head(
    live: Map<RequestId, RequestState>,
    ids: Seq<RequestId>,
    i: int,
)
    requires 0 <= i < ids.len(),
    ensures
        full_tail_debt(live, ids.subrange(i, ids.len() as int))
            == (if live.contains_key(ids[i])
                && history(live[ids[i]]).len() as int % (BLOCK_SIZE_SPEC as int) == 0 {
                1int
            } else {
                0int
            }) + full_tail_debt(live, ids.subrange(i + 1, ids.len() as int)),
        full_tail_debt(live, ids.subrange(i + 1, ids.len() as int)) >= 0,
{
    let n = ids.len() as int;
    let sfx = ids.subrange(i, n);
    assert(sfx[0] == ids[i]);
    assert(sfx.subrange(1, sfx.len() as int) =~= ids.subrange(i + 1, n));
    lemma_debt_nonneg(live, ids.subrange(i + 1, n));
}

pub proof fn lemma_debt_nonneg(live: Map<RequestId, RequestState>, ids: Seq<RequestId>)
    ensures full_tail_debt(live, ids) >= 0,
    decreases ids.len(),
{
    if ids.len() > 0 {
        lemma_debt_nonneg(live, ids.subrange(1, ids.len() as int));
    }
}

// Debt only reads the listed requests' live entries: any state change that
// leaves those entries untouched preserves the debt.
pub proof fn lemma_debt_frame(
    live1: Map<RequestId, RequestState>,
    live2: Map<RequestId, RequestState>,
    ids: Seq<RequestId>,
)
    requires
        forall|j: int| 0 <= j < ids.len()
            ==> (live2.contains_key(#[trigger] ids[j]) <==> live1.contains_key(ids[j]))
                && (live1.contains_key(ids[j]) ==> live2[ids[j]] == live1[ids[j]]),
    ensures
        full_tail_debt(live2, ids) == full_tail_debt(live1, ids),
    decreases ids.len(),
{
    if ids.len() > 0 {
        let tail = ids.subrange(1, ids.len() as int);
        assert forall|j: int| 0 <= j < tail.len()
            implies (live2.contains_key(#[trigger] tail[j]) <==> live1.contains_key(tail[j]))
                && (live1.contains_key(tail[j]) ==> live2[tail[j]] == live1[tail[j]])
        by {
            assert(tail[j] == ids[j + 1]);
        }
        lemma_debt_frame(live1, live2, tail);
    }
}

// Debt extends on the right by the new request's head.
pub proof fn lemma_debt_snoc(
    live: Map<RequestId, RequestState>,
    ids: Seq<RequestId>,
    x: RequestId,
)
    ensures
        full_tail_debt(live, ids.push(x)) == full_tail_debt(live, ids)
            + (if live.contains_key(x)
                && history(live[x]).len() as int % (BLOCK_SIZE_SPEC as int) == 0 {
                1int
            } else {
                0int
            }),
    decreases ids.len(),
{
    let bs = BLOCK_SIZE_SPEC as int;
    let head_x: int = if live.contains_key(x)
        && history(live[x]).len() as int % bs == 0 { 1 } else { 0 };
    if ids.len() == 0 {
        assert(ids.push(x).len() == 1);
        assert(ids.push(x)[0] == x);
        assert(ids.push(x).subrange(1, 1) =~= Seq::<RequestId>::empty());
        assert(full_tail_debt(live, Seq::<RequestId>::empty()) == 0);
        assert(full_tail_debt(live, ids.push(x)) == head_x);
        assert(full_tail_debt(live, ids) == 0);
    } else {
        let head_0: int = if live.contains_key(ids[0])
            && history(live[ids[0]]).len() as int % bs == 0 { 1 } else { 0 };
        let sub = ids.subrange(1, ids.len() as int);
        assert(ids.push(x)[0] == ids[0]);
        assert(ids.push(x).len() as int == ids.len() as int + 1);
        assert(ids.push(x).subrange(1, ids.len() as int + 1) =~= sub.push(x));
        lemma_debt_snoc(live, sub, x);
        assert(full_tail_debt(live, ids.push(x))
            == head_0 + full_tail_debt(live, sub.push(x)));
        assert(full_tail_debt(live, ids) == head_0 + full_tail_debt(live, sub));
    }
}


} // verus!
