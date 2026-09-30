//! Small publication lemmas kept separate from the scheduler's large loops.

use super::*;
#[cfg(verus_only)]
use crate::exec::cache_scheduler::{decode_plan_ready, prefill_plan_ready};

verus! {

// Metadata publication does not change request state, physical token contents,
// refcounts, or scheduler geometry. Queue validity is carried separately.
pub open spec fn publication_layout_frame(a: &CacheScheduler, b: &CacheScheduler) -> bool {
    &&& b.config == a.config
    &&& b.num_blocks == a.num_blocks
    &&& b.free_blocks == a.free_blocks
    &&& b.running@ == a.running@
    &&& b.waiting@ == a.waiting@
    &&& b.live_requests@ == a.live_requests@
    &&& b.accepted_requests@ == a.accepted_requests@
    &&& b.request_residency@ == a.request_residency@
    &&& b.blocks@.dom() == a.blocks@.dom()
    &&& forall|bid: BlockId| #[trigger] a.blocks@.contains_key(bid)
        ==> b.blocks@[bid].tokens@ == a.blocks@[bid].tokens@
            && b.blocks@[bid].refcount == a.blocks@[bid].refcount
}

pub open spec fn published_decode_row(a: &CacheScheduler, b: &CacheScheduler, rid: RequestId) -> bool {
    let ids = a.request_residency@[rid].block_ids@;
    let tokens = history(a.live_requests@[rid]);
    &&& a.running@.contains(rid)
    &&& a.live_requests@.contains_key(rid)
    &&& a.live_requests@[rid].generated_tokens@.len() > 0
    &&& a.request_residency@.contains_key(rid)
    &&& ids.len() > 0
    &&& tokens.len() == ids.len() * (BLOCK_SIZE_SPEC as int)
    &&& tokens.len() <= usize::MAX as int
    &&& a.blocks@.contains_key(ids[ids.len() - 1])
    &&& a.blocks@[ids[ids.len() - 1]].prefix_depth == 0
    &&& a.blocks@[ids[ids.len() - 1]].refcount == 1
    &&& registered_prefix_chain(b.blocks@, ids)
    &&& token_placement_prefix(b.blocks@, ids, tokens, tokens.len() as int)
}

#[verifier::opaque]
pub open spec fn decode_publication_effect(
    a: &CacheScheduler, b: &CacheScheduler, published: Set<RequestId>,
) -> bool {
    &&& publication_layout_frame(a, b)
    &&& positive_provenance_metadata_frame(a, b)
    &&& forall|bid: BlockId| #[trigger] a.blocks@.contains_key(bid)
        && a.blocks@[bid].tokens@.len() < BLOCK_SIZE_SPEC as int
        ==> b.blocks@[bid] == a.blocks@[bid]
    &&& forall|rid: RequestId| #[trigger] published.contains(rid)
        ==> published_decode_row(a, b, rid)
    &&& forall|bid: BlockId| #[trigger] a.blocks@.contains_key(bid)
        && !(exists|rid: RequestId| published.contains(rid)
            && a.request_residency@[rid].block_ids@[
                a.request_residency@[rid].block_ids@.len() - 1] == bid)
        ==> b.blocks@[bid] == a.blocks@[bid]
}

#[verifier::spinoff_prover]
pub proof fn lemma_publication_layout_preserved(a: &CacheScheduler, b: &CacheScheduler)
    requires
        publication_layout_frame(a, b),
        residency_history_aligned(a),
        slot_mapping_aligned(a),
        residency_running_aligned(a),
    ensures
        residency_history_aligned(b),
        slot_mapping_aligned(b),
        residency_running_aligned(b),
{
    lemma_residency_running_aligned_frame(a, b);
    lemma_slot_mapping_aligned_frame(a, b);
    assert forall|rid: RequestId| #[trigger] b.running@.contains(rid)
        && b.live_requests@.contains_key(rid)
        implies {
            let ids = b.request_residency@[rid].block_ids@;
            let tokens = history(b.live_requests@[rid]);
            &&& b.request_residency@.contains_key(rid)
            &&& ids.len() >= 1
            &&& b.blocks@.contains_key(ids[ids.len() - 1])
            &&& tokens.len() == (ids.len() - 1) * (BLOCK_SIZE_SPEC as int)
                + b.blocks@[ids[ids.len() - 1]].tokens@.len()
            &&& b.blocks@[ids[ids.len() - 1]].tokens@.len() >= 1
            &&& token_placement_prefix(b.blocks@, ids, tokens, tokens.len() as int)
            &&& (b.blocks@[ids[ids.len() - 1]].refcount == 1
                || b.blocks@[ids[ids.len() - 1]].tokens@.len() == BLOCK_SIZE_SPEC as int)
        }
    by {
        let ids = a.request_residency@[rid].block_ids@;
        let tokens = history(a.live_requests@[rid]);
        lemma_token_placement_prefix_transfer(a.blocks@, b.blocks@, ids, tokens, tokens.len() as int);
    }
}

pub proof fn lemma_decode_publication_effect_refl(a: &CacheScheduler)
    ensures decode_publication_effect(a, a, Set::empty()),
{
    reveal(decode_publication_effect);
    lemma_positive_provenance_metadata_frame_refl(a);
}

// A newly published tail was private. Publication therefore cannot change a
// page held by another request, including unscheduled and prefill requests.
#[verifier::spinoff_prover]
pub proof fn lemma_decode_publication_bystander_frame(
    a: &CacheScheduler, b: &CacheScheduler, published: Set<RequestId>, rid: RequestId,
)
    requires
        cs_valid(a),
        decode_publication_effect(a, b, published),
        a.request_residency@.contains_key(rid),
        !published.contains(rid),
    ensures
        forall|j: int| 0 <= j < a.request_residency@[rid].block_ids@.len()
            ==> b.blocks@[a.request_residency@[rid].block_ids@[j]]
                == a.blocks@[a.request_residency@[rid].block_ids@[j]],
{
    reveal(decode_publication_effect);
    assert forall|j: int| 0 <= j < a.request_residency@[rid].block_ids@.len()
        implies b.blocks@[a.request_residency@[rid].block_ids@[j]]
            == a.blocks@[a.request_residency@[rid].block_ids@[j]]
    by {
        let bid = a.request_residency@[rid].block_ids@[j];
        assert(a.request_residency@[rid].block_ids@.contains(bid));
        if exists|r: RequestId| published.contains(r)
            && a.request_residency@[r].block_ids@[
                a.request_residency@[r].block_ids@.len() - 1] == bid {
            let r = choose|r: RequestId| published.contains(r)
                && a.request_residency@[r].block_ids@[
                    a.request_residency@[r].block_ids@.len() - 1] == bid;
            assert(a.request_residency@[r].block_ids@.contains(bid));
            lemma_two_holders(residency_holders_of(a, bid), rid, r);
            assert(a.blocks@[bid].refcount >= 2);
        }
    }
}

#[verifier::spinoff_prover]
pub proof fn lemma_plan_decode_publication_ready(pre: &CacheScheduler, post: &CacheScheduler)
    requires
        cs_valid(pre),
        cs_valid(post),
        prefill_plan_ready(pre),
        pre.running@.len() > 0 ==> decode_plan_ready(pre),
        post.num_blocks == pre.num_blocks,
        post.num_blocks <= u64::MAX / BLOCK_SIZE,
        post.live_requests@ == pre.live_requests@,
        residency_history_aligned(post),
        slot_mapping_aligned(post),
        forall|rid: RequestId| #[trigger] post.running@.contains(rid)
            ==> pre.running@.contains(rid) || pre.waiting@.contains(rid),
    ensures decode_plan_ready(post),
{
    assert forall|rid: RequestId| #[trigger] post.running@.contains(rid)
        implies post.live_requests@.contains_key(rid)
            && can_step(post.live_requests@[rid])
            && post.live_requests@[rid].generated_tokens@.len() < usize::MAX as int
            && history(post.live_requests@[rid]).len() <= usize::MAX as int
            && history(post.live_requests@[rid]).len() <= u64::MAX as int
            && post.request_residency@.contains_key(rid)
            && post.request_residency@[rid].slot_mapping@.len() > 0
    by {
        if pre.running@.contains(rid) {
            assert(pre.running@.len() > 0);
            assert(decode_plan_ready(pre));
        } else {
            assert(pre.waiting@.contains(rid));
            assert(pre.live_requests@[rid].generated_tokens@.len() == 0);
        }
    }
}

#[verifier::spinoff_prover]
pub proof fn lemma_decode_publication_plan_frame(
    pre: &CacheScheduler, before: &CacheScheduler, after: &CacheScheduler,
    published: Set<RequestId>, plan: &StepPlan,
)
    requires
        decode_publication_effect(before, after, published),
        plan_slot_segments_ok(pre, before, plan),
        plan_residency_extents(pre, before, plan),
        forall|k: int| 0 <= k < plan.scheduled_ids@.len()
            && !pre.running@.contains(plan.scheduled_ids@[k])
            ==> #[trigger] admitted_pages_exclusive_post(before, plan.scheduled_ids@, k),
    ensures
        plan_slot_segments_ok(pre, after, plan),
        plan_residency_extents(pre, after, plan),
        forall|k: int| 0 <= k < plan.scheduled_ids@.len()
            && !pre.running@.contains(plan.scheduled_ids@[k])
            ==> #[trigger] admitted_pages_exclusive_post(after, plan.scheduled_ids@, k),
{
    reveal(decode_publication_effect);
    reveal(plan_residency_extents);
    assert forall|k: int| 0 <= k < plan.scheduled_ids@.len()
        implies #[trigger] plan_residency_extent_at(pre, after, plan, k)
    by {
        assert(plan_residency_extent_at(pre, before, plan, k));
        let rid = plan.scheduled_ids@[k];
        let tokens = if pre.running@.contains(rid) {
            history(pre.live_requests@[rid])
        } else { pre.live_requests@[rid].prompt_tokens@ };
        lemma_token_placement_prefix_transfer(
            before.blocks@, after.blocks@, before.request_residency@[rid].block_ids@,
            tokens, plan.cu_seqlens_k_repr@[k + 1] - plan.cu_seqlens_k_repr@[k],
        );
    }
    assert forall|k: int| 0 <= k < plan.scheduled_ids@.len()
        implies #[trigger] plan_slot_segments_at(pre, after, plan, k)
    by {
        assert(plan_slot_segments_at(pre, before, plan, k));
        reveal(plan_slot_segments_at);
    }
    assert forall|k: int| 0 <= k < plan.scheduled_ids@.len()
        && !pre.running@.contains(plan.scheduled_ids@[k])
        implies #[trigger] admitted_pages_exclusive_post(after, plan.scheduled_ids@, k)
    by {
        assert(admitted_pages_exclusive_post(before, plan.scheduled_ids@, k));
        let ids = before.request_residency@[plan.scheduled_ids@[k]].block_ids@;
        let tail = ids[ids.len() - 1];
        if before.blocks@[tail].tokens@.len() < BLOCK_SIZE_SPEC as int {
            assert(before.blocks@[tail] == after.blocks@[tail]);
        }
    }
}

#[verifier::spinoff_prover]
pub proof fn lemma_decode_publication_effect_transitive(
    a: &CacheScheduler, b: &CacheScheduler, c: &CacheScheduler,
    first: Set<RequestId>, second: Set<RequestId>,
)
    requires
        decode_publication_effect(a, b, first),
        decode_publication_effect(b, c, second),
    ensures
        decode_publication_effect(a, c, first.union(second)),
{
    reveal(decode_publication_effect);
    lemma_positive_provenance_metadata_frame_transitive(a, b, c);
    assert(publication_layout_frame(a, c));
    assert forall|rid: RequestId| #[trigger] first.union(second).contains(rid)
        implies published_decode_row(a, c, rid)
    by {
        if first.contains(rid) {
            let ids = a.request_residency@[rid].block_ids@;
            let tokens = history(a.live_requests@[rid]);
            assert forall|j: int| 0 <= j < ids.len() implies {
                let bid = #[trigger] ids[j];
                &&& c.blocks@.contains_key(bid)
                &&& c.blocks@[bid].prefix_depth == b.blocks@[bid].prefix_depth
                &&& c.blocks@[bid].parent_block == b.blocks@[bid].parent_block
            }
            by {
                reveal(registered_prefix_chain);
                assert(b.blocks@[ids[j]].prefix_depth as int == j + 1);
            }
            lemma_registered_prefix_chain_transfer(b.blocks@, c.blocks@, ids);
            lemma_token_placement_prefix_transfer(b.blocks@, c.blocks@, ids, tokens, tokens.len() as int);
        } else {
            assert(second.contains(rid));
            let ids = b.request_residency@[rid].block_ids@;
            let tail = ids[ids.len() - 1];
            // Earlier publication may change only earlier published tails,
            // each of which has positive depth, unlike this private tail.
            assert(a.blocks@[tail] == b.blocks@[tail]) by {
                assert(!(exists|r: RequestId| first.contains(r)
                    && a.request_residency@[r].block_ids@[
                        a.request_residency@[r].block_ids@.len() - 1] == tail)) by {
                    if exists|r: RequestId| first.contains(r)
                        && a.request_residency@[r].block_ids@[
                            a.request_residency@[r].block_ids@.len() - 1] == tail {
                        let r = choose|r: RequestId| first.contains(r)
                            && a.request_residency@[r].block_ids@[
                                a.request_residency@[r].block_ids@.len() - 1] == tail;
                        let chain = a.request_residency@[r].block_ids@;
                        reveal(registered_prefix_chain);
                        assert(b.blocks@[chain[chain.len() - 1]].prefix_depth as int == chain.len());
                    }
                }
            }
        }
    }
    assert forall|bid: BlockId| #[trigger] a.blocks@.contains_key(bid)
        && !(exists|rid: RequestId| first.union(second).contains(rid)
            && a.request_residency@[rid].block_ids@[
                a.request_residency@[rid].block_ids@.len() - 1] == bid)
        implies c.blocks@[bid] == a.blocks@[bid]
    by {
        assert(!(exists|rid: RequestId| first.contains(rid)
            && a.request_residency@[rid].block_ids@[
                a.request_residency@[rid].block_ids@.len() - 1] == bid)) by {
            if exists|rid: RequestId| first.contains(rid)
                && a.request_residency@[rid].block_ids@[
                    a.request_residency@[rid].block_ids@.len() - 1] == bid {
                let rid = choose|rid: RequestId| first.contains(rid)
                    && a.request_residency@[rid].block_ids@[
                        a.request_residency@[rid].block_ids@.len() - 1] == bid;
                assert(first.union(second).contains(rid));
                assert(exists|r: RequestId| first.union(second).contains(r)
                    && a.request_residency@[r].block_ids@[
                        a.request_residency@[r].block_ids@.len() - 1] == bid) by {
                    assert(first.union(second).contains(rid));
                }
            }
        }
        assert(!(exists|rid: RequestId| second.contains(rid)
            && b.request_residency@[rid].block_ids@[
                b.request_residency@[rid].block_ids@.len() - 1] == bid)) by {
            if exists|rid: RequestId| second.contains(rid)
                && b.request_residency@[rid].block_ids@[
                    b.request_residency@[rid].block_ids@.len() - 1] == bid {
                let rid = choose|rid: RequestId| second.contains(rid)
                    && b.request_residency@[rid].block_ids@[
                        b.request_residency@[rid].block_ids@.len() - 1] == bid;
                assert(first.union(second).contains(rid));
                assert(exists|r: RequestId| first.union(second).contains(r)
                    && a.request_residency@[r].block_ids@[
                        a.request_residency@[r].block_ids@.len() - 1] == bid) by {
                    assert(first.union(second).contains(rid));
                }
            }
        }
    }
}

// A positive page certifies its entire physical ancestry. The executable
// publisher can inspect just its immediate parent; the existing persistent
// provenance invariant supplies the rest without a runtime prefix scan.
#[verifier::spinoff_prover]
pub proof fn lemma_positive_residency_prefix_registered(
    cs: &CacheScheduler,
    rid: RequestId,
    last: int,
)
    requires
        cs_valid(cs),
        persistent_provenance_closed(cs),
        cs.request_residency@.contains_key(rid),
        0 <= last < cs.request_residency@[rid].block_ids@.len(),
        cs.blocks@.contains_key(cs.request_residency@[rid].block_ids@[last]),
        cs.blocks@[cs.request_residency@[rid].block_ids@[last]].prefix_depth > 0,
    ensures
        registered_prefix_chain(
            cs.blocks@, cs.request_residency@[rid].block_ids@.subrange(0, last + 1),
        ),
    decreases last,
{
    let ids = cs.request_residency@[rid].block_ids@;
    let bid = ids[last];
    reveal(registered_provenance_aligned);
    reveal(persistent_provenance_closed);
    assert(cs.blocks@[bid].prefix_depth as int == last + 1);
    if last > 0 {
        let parent = ids[last - 1];
        assert(cs.blocks@[bid].parent_block == Some(parent));
        assert(cs.blocks@.contains_key(parent));
        assert(cs.blocks@[parent].prefix_depth as int == last);
        lemma_positive_residency_prefix_registered(cs, rid, last - 1);
    }
    reveal(registered_prefix_chain);
    assert forall|j: int| #![trigger ids.subrange(0, last + 1)[j]]
        0 <= j < last + 1 implies {
            let page = ids.subrange(0, last + 1)[j];
            &&& cs.blocks@.contains_key(page)
            &&& cs.blocks@[page].prefix_depth as int == j + 1
            &&& cs.blocks@[page].parent_block
                == if j == 0 { None } else { Some(ids.subrange(0, last + 1)[j - 1]) }
        }
    by {
        assert(ids.subrange(0, last + 1)[j] == ids[j]);
        if j < last {
            assert(ids.subrange(0, last)[j] == ids[j]);
        }
    }
}

}
