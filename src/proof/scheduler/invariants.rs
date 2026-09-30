// Stable-boundary scheduler correctness model and preservation library.
//
// This module contains the mutually dependent stable-boundary validity,
// queue-topology, and persistent-provenance specifications and proofs.
// Transition contracts/preservation and planning/layout invariants live in
// their own acyclic layers; eviction choice is isolated in `eviction_policy`.

use crate::proof::tensor::geometry::*;
use crate::exec::cache_scheduler::availability_queue::*;
use crate::exec::cache_scheduler::types::{BlockEntry, CacheScheduler, StepPlan};
use crate::exec::request_state::*;
use crate::{types::*, proof::model::types::*, proof::tensor::types::*};
use crate::boundary::tensor_runtime as RT;
use vstd::assert_sets_equal;
use vstd::prelude::*;

verus! {
// Exact stable-boundary effect of admitting one fresh request.  Admission is
// deliberately metadata-only: the request becomes live and is appended to the
// waiting queue, while every residency/cache field is left untouched.  KV
// allocation and prefix matching remain responsibilities of the next `plan`.
pub open spec fn scheduler_admission_relation(
    before: &CacheScheduler,
    after: &CacheScheduler,
    request: RequestState,
) -> bool {
    let rid = request.request_id;
    after.config == before.config
    && after.num_blocks == before.num_blocks
    && after.free_blocks == before.free_blocks
    && after.free_queue.order@ == before.free_queue.order@
    && after.cached_queue.head == before.cached_queue.head
    && after.blocks@ == before.blocks@
    && after.running@ == before.running@
    && after.waiting@ == before.waiting@.push(rid)
    && after.request_residency@ == before.request_residency@
    && after.live_requests@ == before.live_requests@.insert(rid, request)
    && after.accepted_requests@
        == before.accepted_requests@.insert(rid, true)
    && after.hash_to_block@ == before.hash_to_block@
}

// ---------------------------------------------------------------------------
// Invariant cluster.  Each predicate reads the spec view of the
// HashMapWithView field via `@`.
// ---------------------------------------------------------------------------

// Every rid in running ∪ waiting appears in `live_requests`.
pub open spec fn live_covers_queue(cs: &CacheScheduler) -> bool {
    forall|rid: RequestId|
        cs.running@.contains(rid) || cs.waiting@.contains(rid)
        ==> #[trigger] cs.live_requests@.contains_key(rid)
}

// Every currently live request has been accepted exactly once.  The converse
// is intentionally false after completion: the ledger retains tombstones.
pub open spec fn live_requests_accepted(cs: &CacheScheduler) -> bool {
    forall|rid: RequestId| #[trigger] cs.live_requests@.contains_key(rid)
        ==> cs.accepted_requests@.contains_key(rid)
            && cs.accepted_requests@[rid]
}

// Running and waiting queues are disjoint.
pub open spec fn queue_disjoint(cs: &CacheScheduler) -> bool {
    forall|rid: RequestId|
        cs.running@.contains(rid) ==> !#[trigger] cs.waiting@.contains(rid)
}

// Running queue has no duplicates.
pub open spec fn running_unique(cs: &CacheScheduler) -> bool {
    cs.running@.no_duplicates()
}

// Waiting queue has no duplicates.
pub open spec fn waiting_unique(cs: &CacheScheduler) -> bool {
    cs.waiting@.no_duplicates()
}

// Every running rid has a residency entry.
pub open spec fn running_has_residency(cs: &CacheScheduler) -> bool {
    forall|rid: RequestId|
        #[trigger] cs.running@.contains(rid)
        ==> cs.request_residency@.contains_key(rid)
}

// Waiting requests have not generated tokens: only running requests accumulate
// them. This establishes the generated-empty clause in admitted-prefill geometry.
pub open spec fn waiting_unstarted(cs: &CacheScheduler) -> bool {
    forall|rid: RequestId|
        #[trigger] cs.waiting@.contains(rid) && cs.live_requests@.contains_key(rid)
        ==> cs.live_requests@[rid].generated_tokens@.len() == 0
}

// Every waiting rid has no residency entry.
pub open spec fn waiting_has_no_residency(cs: &CacheScheduler) -> bool {
    forall|rid: RequestId|
        #[trigger] cs.waiting@.contains(rid)
        ==> !cs.request_residency@.contains_key(rid)
}

// Every residency only references existing blocks.  Vec view via `@`.
pub open spec fn residency_blocks_in_range(cs: &CacheScheduler) -> bool {
    forall|rid: RequestId, k: int|
        #![trigger cs.request_residency@[rid].block_ids@[k]]
        cs.request_residency@.contains_key(rid)
        && 0 <= k < cs.request_residency@[rid].block_ids@.len()
        ==> cs.blocks@.contains_key(cs.request_residency@[rid].block_ids@[k])
}

// Every residency belongs to a live request.  This rules out orphaned
// refcount holders.  Reusable zero-ref pages deliberately have no residency;
// their semantic validity is tracked by `registered_cache_fidelity` instead.
pub open spec fn residency_has_live_request(cs: &CacheScheduler) -> bool {
    forall|rid: RequestId|
        #[trigger] cs.request_residency@.contains_key(rid)
        ==> cs.live_requests@.contains_key(rid)
}

// Every block's stored token list does not exceed BLOCK_SIZE_SPEC.
pub open spec fn block_token_bound(cs: &CacheScheduler) -> bool {
    forall|bid: BlockId|
        #[trigger] cs.blocks@.contains_key(bid)
        ==> cs.blocks@[bid].tokens@.len() <= BLOCK_SIZE_SPEC as int
}

// Hash-to-block points to existing full blocks.
pub open spec fn hash_to_block_in_range(cs: &CacheScheduler) -> bool {
    forall|h: u64|
        #[trigger] cs.hash_to_block@.contains_key(h)
        ==> cs.blocks@.contains_key(cs.hash_to_block@[h])
            && cs.blocks@[cs.hash_to_block@[h]].tokens@.len() == BLOCK_SIZE_SPEC as int
}

// Per-residency: block_ids has no duplicates.  Each residency entry
// represents the (block_index → BlockId) table for a request, and
// distinct logical block-indices map to distinct physical blocks.
pub open spec fn residency_block_ids_unique(cs: &CacheScheduler) -> bool {
    forall|rid: RequestId|
        #[trigger] cs.request_residency@.contains_key(rid)
        ==> cs.request_residency@[rid].block_ids@.no_duplicates()
}

// The set of residencies that hold a given block. Used by
// `refcount_valid`; every `Set` is finite in the current `vstd` model.
pub open spec fn residency_holders_of(cs: &CacheScheduler, bid: BlockId) -> Set<RequestId> {
    cs.request_residency@.dom().filter(
        |rid: RequestId| cs.request_residency@[rid].block_ids@.contains(bid)
    )
}

// Refcount on a block equals the number of residencies holding it.
// This is the bridging invariant that lets `deallocate` argue, when
// `refcount` hits zero, that no other request still references the
// block — so removing it preserves `residency_blocks_in_range`.
pub open spec fn refcount_valid(cs: &CacheScheduler) -> bool {
    forall|bid: BlockId|
        #[trigger] cs.blocks@.contains_key(bid)
        ==> cs.blocks@[bid].refcount as int ==
            residency_holders_of(cs, bid).len() as int
}

// Every registry key maps to a block carrying that hash and registered
// provenance.  This is only a forward consistency property: multiple blocks
// may carry the same hash after a collision, while the registry selects at
// most one of them.  Deallocation therefore checks that the registry still
// targets the page being freed before removing the key.
pub open spec fn hash_to_block_consistent(cs: &CacheScheduler) -> bool {
    forall|h: u64|
        #[trigger] cs.hash_to_block@.contains_key(h)
        ==> cs.blocks@[cs.hash_to_block@[h]].hash_value == h
            && cs.blocks@[cs.hash_to_block@[h]].prefix_depth > 0
}

// Any page carrying registered prefix provenance occupies that exact logical
// depth in every residency that holds it, with the recorded physical parent
// immediately before it.  Thus a surviving holder is also a valid donor of
// the entire physical context chain, including after the original request is
// deallocated.
pub open spec fn registered_provenance_aligned(cs: &CacheScheduler) -> bool {
    forall|rid: RequestId, j: int|
        #![trigger cs.blocks@[cs.request_residency@[rid].block_ids@[j]].prefix_depth]
        cs.request_residency@.contains_key(rid)
        && 0 <= j < cs.request_residency@[rid].block_ids@.len()
        && cs.blocks@.contains_key(cs.request_residency@[rid].block_ids@[j])
        && cs.blocks@[cs.request_residency@[rid].block_ids@[j]].prefix_depth > 0
        ==> {
            let ids = cs.request_residency@[rid].block_ids@;
            let bid = ids[j];
            &&& cs.blocks@[bid].prefix_depth as int == j + 1
            &&& cs.blocks@[bid].parent_block
                == if j == 0 { None } else { Some(ids[j - 1]) }
        }
}

pub proof fn lemma_provenance_parent_refcount_ge_child(
    cs: &CacheScheduler,
    child: BlockId,
    parent: BlockId,
)
    requires
        refcount_valid(cs),
        registered_provenance_aligned(cs),
        residency_block_ids_unique(cs),
        cs.blocks@.contains_key(child),
        cs.blocks@.contains_key(parent),
        cs.blocks@[child].prefix_depth > 0,
        cs.blocks@[child].parent_block == Some(parent),
    ensures
        cs.blocks@[child].refcount <= cs.blocks@[parent].refcount,
{
    let child_holders = residency_holders_of(cs, child);
    let parent_holders = residency_holders_of(cs, parent);
    assert(child_holders.subset_of(parent_holders)) by {
        assert forall|rid: RequestId| child_holders.contains(rid)
            implies parent_holders.contains(rid) by {
            assert(cs.request_residency@.contains_key(rid));
            let held = cs.request_residency@[rid].block_ids@;
            assert(held.contains(child));
            let j = held.index_of(child);
            assert(0 <= j < held.len());
            assert(held[j] == child);
            assert(cs.blocks@[held[j]].prefix_depth > 0);
            assert(registered_provenance_aligned(cs));
            assert(cs.blocks@[child].parent_block
                == if j == 0 { None } else { Some(held[j - 1]) });
            assert(j > 0);
            assert(held[j - 1] == parent);
            assert(held.contains(parent));
        }
    }
    vstd::set_lib::lemma_len_subset(child_holders, parent_holders);
    assert(cs.blocks@[child].refcount as int == child_holders.len() as int);
    assert(cs.blocks@[parent].refcount as int == parent_holders.len() as int);
}

// Every resident registered page has a resident physical parent at the prior
// prefix depth.  Leaf-only eviction preserves this forest closure, eliminating
// the physical-id ABA case that arbitrary parent-first eviction introduced.
#[verifier::opaque]
pub open spec fn persistent_provenance_closed(cs: &CacheScheduler) -> bool {
    forall|bid: BlockId|
        #[trigger] cs.blocks@[bid].prefix_depth > 0
        && cs.blocks@.contains_key(bid)
        && cs.blocks@[bid].prefix_depth > 0
        ==> {
            let entry = cs.blocks@[bid];
            &&& entry.tokens@.len() == BLOCK_SIZE_SPEC as int
            &&& (entry.prefix_depth == 1 ==>
                entry.parent_block is None)
            &&& (entry.prefix_depth > 1 ==> {
                let parent = entry.parent_block.unwrap();
                &&& entry.parent_block is Some
                &&& cs.blocks@.contains_key(parent)
                &&& cs.blocks@[parent].prefix_depth + 1
                    == entry.prefix_depth
            })
        }
}

// Stable metadata frame for every page that already carries registered
// provenance.  Generated-token appends may mutate a provenance-free tail and
// deallocation may reclaim provenance-free pages, but neither transition may
// alter or remove a published full-prefix page.  Keeping this as an opaque
// boundary fact avoids exposing the quantified forest to the commit loop.
#[verifier::opaque]
pub open spec fn positive_provenance_metadata_frame(
    before: &CacheScheduler,
    after: &CacheScheduler,
) -> bool {
    forall|bid: BlockId|
        #[trigger] before.blocks@[bid].prefix_depth > 0
        && before.blocks@.contains_key(bid)
        && before.blocks@[bid].prefix_depth > 0
        ==> after.blocks@.contains_key(bid)
            && after.blocks@[bid].tokens@ == before.blocks@[bid].tokens@
            && after.blocks@[bid].hash_value == before.blocks@[bid].hash_value
            && after.blocks@[bid].prefix_depth == before.blocks@[bid].prefix_depth
            && after.blocks@[bid].parent_block == before.blocks@[bid].parent_block
}

// One-way provenance origin used while planning.  Reclamation may remove old
// zero-ref leaves, so the old-to-new frame above is intentionally too strong
// for the plan phase.  What planning does guarantee is the converse: every
// positive-depth page that still exists before publication is an unchanged
// page from the plan-entry state.  Fresh allocation contributes only
// provenance-free (depth-zero) pages; publication is the sole transition that
// turns those pages into new positive-depth certificates.
#[verifier::opaque]
pub open spec fn positive_provenance_origin(
    before: &CacheScheduler,
    after: &CacheScheduler,
) -> bool {
    forall|bid: BlockId|
        #[trigger] after.blocks@[bid].prefix_depth > 0
        && after.blocks@.contains_key(bid)
        && after.blocks@[bid].prefix_depth > 0
        ==> before.blocks@.contains_key(bid)
            && after.blocks@[bid].tokens@ == before.blocks@[bid].tokens@
            && after.blocks@[bid].hash_value == before.blocks@[bid].hash_value
            && after.blocks@[bid].prefix_depth == before.blocks@[bid].prefix_depth
            && after.blocks@[bid].parent_block == before.blocks@[bid].parent_block
}

pub open spec fn positive_page_unchanged_from_pre(
    before: &CacheScheduler,
    after: &CacheScheduler,
    bid: BlockId,
) -> bool {
    before.blocks@.contains_key(bid)
    && after.blocks@.contains_key(bid)
    && after.blocks@[bid].prefix_depth > 0
    && after.blocks@[bid].tokens@ == before.blocks@[bid].tokens@
    && after.blocks@[bid].hash_value == before.blocks@[bid].hash_value
    && after.blocks@[bid].prefix_depth == before.blocks@[bid].prefix_depth
    && after.blocks@[bid].parent_block == before.blocks@[bid].parent_block
}

pub proof fn lemma_positive_provenance_origin_refl(cs: &CacheScheduler)
    ensures
        positive_provenance_origin(cs, cs),
{
    reveal(positive_provenance_origin);
}

pub proof fn lemma_positive_provenance_origin_blocks_eq(
    before: &CacheScheduler,
    after: &CacheScheduler,
)
    requires
        after.blocks@ == before.blocks@,
    ensures
        positive_provenance_origin(before, after),
{
    reveal(positive_provenance_origin);
}

pub proof fn lemma_positive_provenance_origin_from_surviving_blocks(
    before: &CacheScheduler,
    after: &CacheScheduler,
)
    requires
        forall|bid: BlockId| #[trigger] after.blocks@.contains_key(bid)
            ==> before.blocks@.contains_key(bid)
                && after.blocks@[bid] == before.blocks@[bid],
    ensures
        positive_provenance_origin(before, after),
{
    reveal(positive_provenance_origin);
}

pub proof fn lemma_positive_provenance_origin_transitive(
    first: &CacheScheduler,
    middle: &CacheScheduler,
    last: &CacheScheduler,
)
    requires
        positive_provenance_origin(first, middle),
        positive_provenance_origin(middle, last),
    ensures
        positive_provenance_origin(first, last),
{
    reveal(positive_provenance_origin);
}

// Convenient bridge for commit-time transitions.  The metadata frame says
// that every old positive page survives unchanged; if a transition also
// establishes that each post positive page already existed, the two facts
// together give the converse origin relation used by the engine step.
pub proof fn lemma_positive_provenance_origin_from_metadata_frame(
    before: &CacheScheduler,
    after: &CacheScheduler,
)
    requires
        positive_provenance_metadata_frame(before, after),
        forall|bid: BlockId|
            #[trigger] after.blocks@[bid].prefix_depth > 0
            && after.blocks@.contains_key(bid)
            && after.blocks@[bid].prefix_depth > 0
            ==> before.blocks@.contains_key(bid)
                && before.blocks@[bid].prefix_depth > 0,
    ensures
        positive_provenance_origin(before, after),
{
    reveal(positive_provenance_metadata_frame);
    reveal(positive_provenance_origin);
}

pub proof fn lemma_positive_provenance_metadata_frame_refl(
    cs: &CacheScheduler,
)
    ensures
        positive_provenance_metadata_frame(cs, cs),
{
    reveal(positive_provenance_metadata_frame);
}

pub proof fn lemma_positive_provenance_metadata_frame_blocks_eq(
    before: &CacheScheduler,
    after: &CacheScheduler,
)
    requires
        after.blocks@ == before.blocks@,
    ensures
        positive_provenance_metadata_frame(before, after),
{
    reveal(positive_provenance_metadata_frame);
}

pub proof fn lemma_positive_provenance_metadata_frame_transitive(
    first: &CacheScheduler,
    middle: &CacheScheduler,
    last: &CacheScheduler,
)
    requires
        positive_provenance_metadata_frame(first, middle),
        positive_provenance_metadata_frame(middle, last),
    ensures
        positive_provenance_metadata_frame(first, last),
{
    reveal(positive_provenance_metadata_frame);
}

pub proof fn lemma_physical_parent_closure_from_metadata_frame(
    before: &CacheScheduler,
    after: &CacheScheduler,
)
    requires
        persistent_provenance_closed(before),
        positive_provenance_metadata_frame(before, after),
        positive_provenance_origin(before, after),
    ensures
        forall|bid: BlockId| #[trigger] after.blocks@.contains_key(bid)
            && after.blocks@[bid].prefix_depth > 1
            ==> after.blocks@[bid].parent_block is Some
                && after.blocks@.contains_key(
                    after.blocks@[bid].parent_block.unwrap(),
                )
                && after.blocks@[
                    after.blocks@[bid].parent_block.unwrap()
                ].prefix_depth + 1 == after.blocks@[bid].prefix_depth,
{
    reveal(persistent_provenance_closed);
    reveal(positive_provenance_metadata_frame);
    reveal(positive_provenance_origin);
    assert forall|bid: BlockId| #[trigger] after.blocks@.contains_key(bid)
        && after.blocks@[bid].prefix_depth > 1
        implies after.blocks@[bid].parent_block is Some
            && after.blocks@.contains_key(
                after.blocks@[bid].parent_block.unwrap(),
            )
            && after.blocks@[
                after.blocks@[bid].parent_block.unwrap()
            ].prefix_depth + 1 == after.blocks@[bid].prefix_depth
    by {
        assert(before.blocks@.contains_key(bid));
        assert(before.blocks@[bid].prefix_depth
            == after.blocks@[bid].prefix_depth);
        assert(before.blocks@[bid].parent_block
            == after.blocks@[bid].parent_block);
        assert(before.blocks@[bid].parent_block is Some);
        let parent = before.blocks@[bid].parent_block.unwrap();
        assert(before.blocks@.contains_key(parent));
        assert(before.blocks@[parent].prefix_depth + 1
            == before.blocks@[bid].prefix_depth);
        assert(before.blocks@[parent].prefix_depth > 0);
        assert(after.blocks@.contains_key(parent));
        assert(after.blocks@[parent].prefix_depth
            == before.blocks@[parent].prefix_depth);
    }
}

// Generic preservation rule for transitions that add only provenance-free
// pages and otherwise leave every provenance-bearing page's local certificate
// verbatim.  Callers explicitly establish physical parent closure after any
// transition that can remove a resident page.
pub proof fn lemma_persistent_provenance_closed_frame(
    before: &CacheScheduler,
    after: &CacheScheduler,
)
    requires
        persistent_provenance_closed(before),
        forall|bid: BlockId|
            #[trigger] before.blocks@[bid].prefix_depth > 0
            && before.blocks@.contains_key(bid)
            && before.blocks@[bid].prefix_depth > 0
            && after.blocks@.contains_key(bid)
            ==> after.blocks@[bid].tokens@ == before.blocks@[bid].tokens@
                && after.blocks@[bid].prefix_depth
                    == before.blocks@[bid].prefix_depth
                && after.blocks@[bid].parent_block
                    == before.blocks@[bid].parent_block,
        forall|bid: BlockId| #[trigger] after.blocks@.contains_key(bid)
            && after.blocks@[bid].prefix_depth > 0
            ==> before.blocks@.contains_key(bid)
                && after.blocks@[bid].tokens@ == before.blocks@[bid].tokens@
                && after.blocks@[bid].prefix_depth
                    == before.blocks@[bid].prefix_depth
                && after.blocks@[bid].parent_block
                    == before.blocks@[bid].parent_block,
        forall|bid: BlockId| #[trigger] after.blocks@.contains_key(bid)
            && after.blocks@[bid].prefix_depth > 1
            ==> after.blocks@[bid].parent_block is Some
                && after.blocks@.contains_key(
                    after.blocks@[bid].parent_block.unwrap(),
                )
                && after.blocks@[
                    after.blocks@[bid].parent_block.unwrap()
                ].prefix_depth + 1 == after.blocks@[bid].prefix_depth,
    ensures
        persistent_provenance_closed(after),
{
    reveal(persistent_provenance_closed);
    assert forall|bid: BlockId|
        #[trigger] after.blocks@[bid].prefix_depth > 0
        && after.blocks@.contains_key(bid)
        && after.blocks@[bid].prefix_depth > 0
        implies {
            let entry = after.blocks@[bid];
            &&& entry.tokens@.len() == BLOCK_SIZE_SPEC as int
            &&& (entry.prefix_depth == 1 ==>
                entry.parent_block is None)
            &&& (entry.prefix_depth > 1 ==> {
                let parent = entry.parent_block.unwrap();
                &&& entry.parent_block is Some
                &&& after.blocks@.contains_key(parent)
                &&& after.blocks@[parent].prefix_depth + 1
                    == entry.prefix_depth
            })
        }
    by {
        assert(before.blocks@.contains_key(bid));
        assert(before.blocks@[bid].prefix_depth > 0);
    }
}

pub proof fn lemma_persistent_provenance_closed_blocks_eq(
    before: &CacheScheduler,
    after: &CacheScheduler,
)
    requires
        persistent_provenance_closed(before),
        after.blocks@ == before.blocks@,
        after.num_blocks == before.num_blocks,
    ensures
        persistent_provenance_closed(after),
{
    reveal(persistent_provenance_closed);
}

// A zero-reference registered page with no resident provenance child.  The
// active cached queue exposes only such pages at its head.
pub open spec fn zero_ref_provenance_leaf(
    cs: &CacheScheduler,
    bid: BlockId,
) -> bool {
    cs.blocks@.contains_key(bid)
    && cs.blocks@[bid].refcount == 0
    && cs.blocks@[bid].prefix_depth > 0
    && (forall|child: BlockId| cs.blocks@.contains_key(child)
        && cs.blocks@[child].prefix_depth > 0
        ==> #[trigger] cs.blocks@[child].parent_block != Some(bid))
}

// Resident registered page with no holders.  Queue membership uses this
// extensional predicate; topological order separately restricts eviction to a
// leaf.
pub open spec fn zero_ref_cached_page(
    cs: &CacheScheduler,
    bid: BlockId,
) -> bool {
    cs.blocks@.contains_key(bid)
    && cs.blocks@[bid].refcount == 0
    && cs.blocks@[bid].prefix_depth > 0
}

// The two availability queues are homogeneous and exact: `free_queue` contains
// precisely the physically vacant ids, while `cached_queue` contains precisely
// the resident zero-reference provenance pages.
pub open spec fn free_queue_membership_valid(cs: &CacheScheduler) -> bool {
    let vacant = cs.free_queue.order@;
    let cached = cs.cached_queue.order@;
    &&& forall|bid: BlockId| #[trigger] vacant.contains(bid) <==>
        bid < cs.num_blocks && !cs.blocks@.contains_key(bid)
    &&& forall|bid: BlockId| #[trigger] cached.contains(bid) <==>
        zero_ref_cached_page(cs, bid)
}

// If a resident parent is cached, every resident provenance child is also
// cached and precedes it. Consequently the cached queue head is a leaf
// of the physical provenance forest.  This is a correctness eligibility rule;
// ordering among eligible leaves is still replacement policy.
pub open spec fn free_queue_topological(cs: &CacheScheduler) -> bool {
    let q = cs.cached_queue.order@;
    forall|child: BlockId| cs.blocks@.contains_key(child)
        && cs.blocks@[child].prefix_depth > 0
        && cs.blocks@[child].parent_block is Some
        && q.contains(cs.blocks@[child].parent_block.unwrap())
        ==> #[trigger] q.contains(child)
            && q.index_of(child)
                < q.index_of(cs.blocks@[child].parent_block.unwrap())
}

// `free_blocks` keeps its existing meaning and is exactly the vacant-queue
// length.  The two queues are disjoint and together never exceed the physical
// block pool.
pub open spec fn free_queue_partitioned(cs: &CacheScheduler) -> bool {
    let vacant = cs.free_queue.order@;
    let cached = cs.cached_queue.order@;
    &&& cs.free_blocks as int == vacant.len()
    &&& vacant.to_set().disjoint(cached.to_set())
    &&& vacant.len() + cached.len() <= cs.num_blocks as int
}

pub open spec fn free_queue_valid(cs: &CacheScheduler) -> bool {
    free_queue_shape(&cs.free_queue)
    && free_queue_shape(&cs.cached_queue)
    && free_queue_membership_valid(cs)
    && free_queue_partitioned(cs)
    && free_queue_topological(cs)
}

#[verifier::opaque]
pub open spec fn free_queue_valid_token(cs: &CacheScheduler) -> bool {
    free_queue_valid(cs)
}

pub proof fn lemma_free_queue_valid_to_token(cs: &CacheScheduler)
    requires free_queue_valid(cs),
    ensures free_queue_valid_token(cs),
{
    reveal(free_queue_valid_token);
}

pub proof fn lemma_free_queue_token_to_valid(cs: &CacheScheduler)
    requires free_queue_valid_token(cs),
    ensures free_queue_valid(cs),
{
    reveal(free_queue_valid_token);
}

#[verifier::spinoff_prover]
#[verifier::rlimit(20)]
pub proof fn lemma_free_queue_valid_token_frame(
    before: &CacheScheduler,
    after: &CacheScheduler,
)
    requires
        free_queue_valid_token(before),
        after.num_blocks == before.num_blocks,
        after.free_blocks == before.free_blocks,
        after.blocks@ == before.blocks@,
        after.free_queue == before.free_queue,
        after.cached_queue == before.cached_queue,
    ensures
        free_queue_valid_token(after),
{
    reveal(free_queue_valid_token);
}

pub proof fn lemma_free_queue_valid_token_shape_token(cs: &CacheScheduler)
    requires free_queue_valid_token(cs),
    ensures free_queue_shape_token(&cs.free_queue),
{
    reveal(free_queue_valid_token);
    reveal(free_queue_shape_token);
}

pub proof fn lemma_free_queue_valid_token_cached_shape_token(cs: &CacheScheduler)
    requires free_queue_valid_token(cs),
    ensures free_queue_shape_token(&cs.cached_queue),
{
    reveal(free_queue_valid_token);
    reveal(free_queue_shape_token);
}


#[verifier::spinoff_prover]
#[verifier::rlimit(30)]
pub proof fn lemma_reconciled_vacant_count(
    before: &CacheScheduler,
    after: &CacheScheduler,
    ids: Seq<BlockId>,
    vacant: Seq<BlockId>,
)
    requires
        block_count_valid(before),
        block_count_valid(after),
        after.num_blocks == before.num_blocks,
        vacant.no_duplicates(),
        forall|b: BlockId| vacant.contains(b)
            ==> ids.contains(b) && !after.blocks@.contains_key(b),
        forall|j: int| 0 <= j < ids.len() ==> {
            let b = #[trigger] ids[j];
            before.blocks@.contains_key(b)
        },
        forall|j: int| 0 <= j < ids.len()
            && !after.blocks@.contains_key(#[trigger] ids[j])
            ==> vacant.contains(ids[j]),
        forall|b: BlockId| #[trigger] after.blocks@.contains_key(b)
            ==> before.blocks@.contains_key(b),
        forall|b: BlockId| before.blocks@.contains_key(b)
            && !after.blocks@.contains_key(b)
            ==> ids.contains(b),
    ensures
        after.free_blocks as int
            == before.free_blocks as int + vacant.len(),
{
    let pre_dom = before.blocks@.dom();
    let post_dom = after.blocks@.dom();
    let vacant_set = vacant.to_set();
    assert_sets_equal!(post_dom == pre_dom.difference(vacant_set), b: BlockId => {
        if post_dom.contains(b) {
            assert(pre_dom.contains(b));
            assert(!vacant_set.contains(b)) by {
                if vacant_set.contains(b) {
                    assert(vacant.contains(b));
                    assert(!after.blocks@.contains_key(b));
                }
            }
        }
        if pre_dom.difference(vacant_set).contains(b) {
            if !post_dom.contains(b) {
                assert(ids.contains(b));
                let j = ids.index_of(b);
                assert(0 <= j < ids.len());
                assert(ids[j] == b);
                assert(vacant.contains(b));
                assert(vacant_set.contains(b));
            }
        }
    });
    vacant.unique_seq_to_set();
    assert(vacant_set.subset_of(pre_dom)) by {
        assert forall|b: BlockId| vacant_set.contains(b)
            implies pre_dom.contains(b) by {
            assert(vacant.contains(b));
            let j = ids.index_of(b);
            assert(0 <= j < ids.len());
            assert(ids[j] == b);
        }
    }
    assert(pre_dom.intersect(vacant_set) == vacant_set) by {
        assert_sets_equal!(pre_dom.intersect(vacant_set) == vacant_set,
            b: BlockId => {});
    }
    vstd::set_lib::lemma_set_difference_len(pre_dom, vacant_set);
    assert(post_dom.len() + vacant_set.len() == pre_dom.len());
}







pub proof fn lemma_cached_queue_head_is_leaf(cs: &CacheScheduler)
    requires
        free_queue_valid(cs),
        cs.cached_queue.head is Some,
    ensures
        zero_ref_provenance_leaf(cs, cs.cached_queue.head.unwrap()),
{
    let q = cs.cached_queue.order@;
    let bid = cs.cached_queue.head.unwrap();
    assert(q.len() > 0);
    assert(q[0] == bid);
    assert(q.contains(bid));
    assert(zero_ref_cached_page(cs, bid));
    assert forall|child: BlockId| cs.blocks@.contains_key(child)
        && cs.blocks@[child].prefix_depth > 0
        implies #[trigger] cs.blocks@[child].parent_block != Some(bid)
    by {
        if cs.blocks@[child].parent_block == Some(bid) {
            assert(free_queue_topological(cs));
            assert(q.contains(child));
            assert(q.index_of(child) < q.index_of(bid));
            assert(q.index_of(bid) == 0) by {
                assert(q.no_duplicates());
            }
            assert(q.index_of(child) >= 0);
        }
    }
}

pub proof fn lemma_free_queue_valid_frame(
    before: &CacheScheduler,
    after: &CacheScheduler,
)
    requires
        free_queue_valid(before),
        after.num_blocks == before.num_blocks,
        after.free_blocks == before.free_blocks,
        after.blocks@ == before.blocks@,
        free_queue_shape(&after.free_queue),
        free_queue_shape(&after.cached_queue),
        after.free_queue.order@ == before.free_queue.order@,
        after.cached_queue.order@ == before.cached_queue.order@,
    ensures
        free_queue_valid(after),
{
    assert(free_queue_membership_valid(after));
    assert(free_queue_partitioned(after));
    assert(free_queue_topological(after));
}

// Establish the extensional membership effect without cache-map or refcount
// facts in scope. Callers can then reason about one page at a time.
pub proof fn lemma_queue_remove_membership<A>(
    before: Seq<A>,
    after: Seq<A>,
    removed: A,
)
    requires
        before.no_duplicates(),
        before.contains(removed),
        after == before.remove(before.index_of(removed)),
    ensures
        forall|x: A| #[trigger] after.contains(x)
            <==> before.contains(x) && x != removed,
{
    let ri = before.index_of(removed);
    assert(0 <= ri < before.len());
    assert(before[ri] == removed);
    before.remove_ensures(ri);
    assert forall|x: A| #[trigger] after.contains(x)
        <==> before.contains(x) && x != removed by {
        if after.contains(x) {
            let ai = after.index_of(x);
            let bi = if ai < ri { ai } else { ai + 1 };
            assert(0 <= ai < after.len());
            assert(after[ai] == x);
            assert(before[bi] == x);
            assert(bi != ri);
        }
        if before.contains(x) && x != removed {
            let bi = before.index_of(x);
            let ai = if bi < ri { bi } else { bi - 1 };
            assert(0 <= bi < before.len());
            assert(before[bi] == x);
            assert(bi != ri);
            assert(0 <= ai < after.len());
            assert(after[ai] == x);
        }
    }
}

pub proof fn lemma_queue_remove_preserves_order(
    before: Seq<BlockId>,
    after: Seq<BlockId>,
    removed: BlockId,
    earlier: BlockId,
    later: BlockId,
)
    requires
        before.no_duplicates(),
        before.contains(removed),
        before.contains(earlier),
        before.contains(later),
        earlier != removed,
        later != removed,
        before.index_of(earlier) < before.index_of(later),
        after == before.subrange(0, before.index_of(removed))
            + before.subrange(
                before.index_of(removed) + 1,
                before.len() as int,
            ),
    ensures
        after.contains(earlier),
        after.contains(later),
        after.index_of(earlier) < after.index_of(later),
{
    let ri = before.index_of(removed);
    let ei = before.index_of(earlier);
    let li = before.index_of(later);
    let new_ei = if ei < ri { ei } else { ei - 1 };
    let new_li = if li < ri { li } else { li - 1 };
    assert(ei != ri) by {
        if ei == ri {
            assert(before[ei] == earlier);
            assert(before[ri] == removed);
        }
    }
    assert(li != ri) by {
        if li == ri {
            assert(before[li] == later);
            assert(before[ri] == removed);
        }
    }
    assert(0 <= new_ei < after.len());
    assert(0 <= new_li < after.len());
    assert(after[new_ei] == earlier) by {
        if ei < ri {
            assert(after[new_ei] == before[ei]);
        } else {
            assert(after[new_ei] == before[new_ei + 1]);
        }
    }
    assert(after[new_li] == later) by {
        if li < ri {
            assert(after[new_li] == before[li]);
        } else {
            assert(after[new_li] == before[new_li + 1]);
        }
    }
    assert(after.no_duplicates()) by {
        vstd::seq_lib::lemma_no_dup_in_concat(
            before.subrange(0, ri),
            before.subrange(ri + 1, before.len() as int),
        );
    }
    assert(after.index_of(earlier) == new_ei) by {
        assert(after.contains(earlier));
        assert(after[after.index_of(earlier)] == earlier);
        if after.index_of(earlier) != new_ei {
            assert(after.no_duplicates());
        }
    }
    assert(after.index_of(later) == new_li) by {
        assert(after.contains(later));
        assert(after[after.index_of(later)] == later);
        if after.index_of(later) != new_li {
            assert(after.no_duplicates());
        }
    }
    assert(new_ei < new_li) by {
        if ei < ri && ri < li {
            assert(new_ei == ei);
            assert(new_li == li - 1);
        }
    }
}


// Two-queue allocation consumes vacant ids without touching cached pages.
// Newly installed entries are active, unregistered pages, so they cannot
// become members of (or add edges to) the cached provenance order.
#[verifier::spinoff_prover]
#[verifier::rlimit(100)]
pub proof fn lemma_two_queue_after_vacant_prefix_allocation(
    before: &CacheScheduler,
    after: &CacheScheduler,
    taken: int,
)
    requires
        free_queue_valid(before),
        0 <= taken <= before.free_blocks as int,
        after.num_blocks == before.num_blocks,
        after.free_blocks as int + taken == before.free_blocks as int,
        free_queue_shape(&after.free_queue),
        free_queue_shape(&after.cached_queue),
        after.free_queue.order@ == before.free_queue.order@.subrange(
            taken, before.free_queue.order@.len() as int,
        ),
        after.cached_queue.order@ == before.cached_queue.order@,
        forall|bid: BlockId| #[trigger] before.blocks@.contains_key(bid)
            ==> after.blocks@.contains_key(bid)
                && after.blocks@[bid] == before.blocks@[bid],
        forall|bid: BlockId| #[trigger] after.blocks@.contains_key(bid)
            ==> before.blocks@.contains_key(bid)
                || before.free_queue.order@.subrange(0, taken).contains(bid),
        forall|bid: BlockId|
            #[trigger] before.free_queue.order@.subrange(0, taken).contains(bid)
            ==> !before.blocks@.contains_key(bid)
                && after.blocks@.contains_key(bid)
                && after.blocks@[bid].refcount > 0
                && after.blocks@[bid].prefix_depth == 0,
    ensures
        free_queue_valid(after),
{
    let bv = before.free_queue.order@;
    let av = after.free_queue.order@;
    let cached = before.cached_queue.order@;
    let allocated = bv.subrange(0, taken);
    assert(taken <= bv.len());

    assert(free_queue_membership_valid(after)) by {
        assert forall|bid: BlockId| #[trigger] av.contains(bid) <==>
            bid < after.num_blocks && !after.blocks@.contains_key(bid)
        by {
            if av.contains(bid) {
                vstd::seq_lib::lemma_seq_subrange_elements(
                    bv, taken, bv.len() as int, bid,
                );
                assert(bv.contains(bid));
                assert(bid < before.num_blocks);
                assert(!before.blocks@.contains_key(bid));
                if after.blocks@.contains_key(bid) {
                    assert(allocated.contains(bid));
                    let ai = allocated.index_of(bid);
                    let vi = av.index_of(bid);
                    assert(0 <= ai < allocated.len());
                    assert(0 <= vi < av.len());
                    assert(bv[ai] == bid);
                    assert(bv[vi + taken] == bid);
                    assert(ai != vi + taken);
                    assert(bv.no_duplicates());
                }
            }
            if bid < after.num_blocks && !after.blocks@.contains_key(bid) {
                assert(!before.blocks@.contains_key(bid));
                assert(bv.contains(bid));
                assert(!allocated.contains(bid)) by {
                    if allocated.contains(bid) {
                        assert(after.blocks@.contains_key(bid));
                    }
                }
                let i = bv.index_of(bid);
                assert(0 <= i < bv.len());
                assert(i >= taken) by {
                    if i < taken {
                        vstd::seq_lib::lemma_seq_subrange_elements(
                            bv, 0, taken, bid,
                        );
                    }
                }
                vstd::seq_lib::lemma_seq_subrange_elements(
                    bv, taken, bv.len() as int, bid,
                );
            }
        }
        assert forall|bid: BlockId| #[trigger] cached.contains(bid) <==>
            zero_ref_cached_page(after, bid)
        by {
            if cached.contains(bid) {
                assert(zero_ref_cached_page(before, bid));
                assert(after.blocks@[bid] == before.blocks@[bid]);
            }
            if zero_ref_cached_page(after, bid) {
                if before.blocks@.contains_key(bid) {
                    assert(after.blocks@[bid] == before.blocks@[bid]);
                    assert(zero_ref_cached_page(before, bid));
                } else {
                    assert(allocated.contains(bid));
                    assert(after.blocks@[bid].refcount > 0);
                }
            }
        }
    }

    assert(free_queue_partitioned(after)) by {
        assert(after.free_blocks as int == av.len()) by {
            assert(before.free_blocks as int == bv.len());
            assert(av.len() == bv.len() - taken);
        }
        assert(av.to_set().disjoint(cached.to_set())) by {
            assert forall|bid: BlockId| av.to_set().contains(bid)
                implies !cached.to_set().contains(bid) by {
                assert(bv.to_set().contains(bid));
                assert(bv.to_set().disjoint(cached.to_set()));
            }
        }
        assert(av.len() + cached.len() <= after.num_blocks as int) by {
            assert(bv.len() + cached.len() <= before.num_blocks as int);
            assert(av.len() <= bv.len());
        }
    }

    assert(free_queue_topological(after)) by {
        assert forall|child: BlockId| after.blocks@.contains_key(child)
            && after.blocks@[child].prefix_depth > 0
            && after.blocks@[child].parent_block is Some
            && cached.contains(after.blocks@[child].parent_block.unwrap())
            implies #[trigger] cached.contains(child)
                && cached.index_of(child)
                    < cached.index_of(after.blocks@[child].parent_block.unwrap())
        by {
            assert(before.blocks@.contains_key(child)) by {
                if !before.blocks@.contains_key(child) {
                    assert(allocated.contains(child));
                    assert(after.blocks@[child].prefix_depth == 0);
                }
            }
            assert(after.blocks@[child] == before.blocks@[child]);
            assert(free_queue_topological(before));
        }
    }
}


pub proof fn lemma_two_queue_valid_publish_active_page(
    before: &CacheScheduler,
    after: &CacheScheduler,
    bid: BlockId,
)
    requires
        free_queue_valid(before),
        after.num_blocks == before.num_blocks,
        after.free_blocks == before.free_blocks,
        free_queue_shape(&after.free_queue),
        free_queue_shape(&after.cached_queue),
        after.free_queue.order@ == before.free_queue.order@,
        after.cached_queue.order@ == before.cached_queue.order@,
        after.blocks@.dom() == before.blocks@.dom(),
        before.blocks@.contains_key(bid),
        after.blocks@.contains_key(bid),
        after.blocks@[bid].refcount == before.blocks@[bid].refcount,
        after.blocks@[bid].refcount > 0,
        after.blocks@[bid].prefix_depth > 0
            && after.blocks@[bid].parent_block is Some ==>
            !after.cached_queue.order@.contains(
                after.blocks@[bid].parent_block.unwrap(),
            ),
        forall|other: BlockId| other != bid
            && #[trigger] before.blocks@.contains_key(other)
            ==> after.blocks@[other] == before.blocks@[other],
    ensures
        free_queue_valid(after),
{
    let vacant = after.free_queue.order@;
    let cached = after.cached_queue.order@;

    assert(!vacant.contains(bid));
    assert(!cached.contains(bid));
    assert(free_queue_membership_valid(after)) by {
        assert forall|b: BlockId| #[trigger] vacant.contains(b) <==>
            b < after.num_blocks && !after.blocks@.contains_key(b)
        by {
            if b == bid {
                assert(after.blocks@.contains_key(bid));
                assert(!before.free_queue.order@.contains(bid));
            } else {
                assert(after.blocks@.contains_key(b)
                    <==> before.blocks@.contains_key(b));
            }
        }
        assert forall|b: BlockId| #[trigger] cached.contains(b) <==>
            zero_ref_cached_page(after, b)
        by {
            if b == bid {
                assert(after.blocks@[bid].refcount > 0);
            } else if before.blocks@.contains_key(b) {
                assert(after.blocks@[b] == before.blocks@[b]);
            }
        }
    }
    assert(free_queue_partitioned(after));
    assert(free_queue_topological(after)) by {
        assert forall|child: BlockId| after.blocks@.contains_key(child)
            && after.blocks@[child].prefix_depth > 0
            && after.blocks@[child].parent_block is Some
            && cached.contains(after.blocks@[child].parent_block.unwrap())
            implies #[trigger] cached.contains(child)
                && cached.index_of(child)
                    < cached.index_of(after.blocks@[child].parent_block.unwrap())
        by {
            let parent = after.blocks@[child].parent_block.unwrap();
            if child == bid {
                assert(!cached.contains(parent));
            } else {
                assert(after.blocks@[child] == before.blocks@[child]);
                assert(before.blocks@.contains_key(child));
                assert(before.blocks@[child].prefix_depth > 0);
                assert(before.blocks@[child].parent_block == Some(parent));
                assert(before.cached_queue.order@.contains(parent));
                assert(free_queue_topological(before));
                assert(before.cached_queue.order@.contains(child));
                assert(before.cached_queue.order@.index_of(child)
                    < before.cached_queue.order@.index_of(parent));
            }
        }
    }
}

#[verifier::spinoff_prover]
#[verifier::rlimit(100)]
pub proof fn lemma_two_queue_valid_token_publish_active_page(
    before: &CacheScheduler,
    after: &CacheScheduler,
    bid: BlockId,
)
    requires
        free_queue_valid_token(before),
        after.num_blocks == before.num_blocks,
        after.free_blocks == before.free_blocks,
        after.free_queue == before.free_queue,
        after.cached_queue == before.cached_queue,
        after.blocks@.dom() == before.blocks@.dom(),
        before.blocks@.contains_key(bid),
        after.blocks@.contains_key(bid),
        after.blocks@[bid].refcount == before.blocks@[bid].refcount,
        after.blocks@[bid].refcount > 0,
        after.blocks@[bid].prefix_depth > 0
            && after.blocks@[bid].parent_block is Some ==>
            !after.cached_queue.order@.contains(
                after.blocks@[bid].parent_block.unwrap(),
            ),
        forall|other: BlockId| other != bid
            && #[trigger] before.blocks@.contains_key(other)
            ==> after.blocks@[other] == before.blocks@[other],
    ensures
        free_queue_valid_token(after),
{
    lemma_free_queue_token_to_valid(before);
    assert(free_queue_shape(&after.free_queue));
    assert(free_queue_shape(&after.cached_queue));
    lemma_two_queue_valid_publish_active_page(before, after, bid);
    lemma_free_queue_valid_to_token(after);
}


#[verifier::spinoff_prover]
#[verifier::rlimit(150)]
pub proof fn lemma_two_queue_after_block_activation(
    before: &CacheScheduler,
    after: &CacheScheduler,
    bid: BlockId,
)
    requires
        free_queue_valid(before),
        before.blocks@.contains_key(bid),
        bid < before.num_blocks,
        before.blocks@[bid].refcount < u64::MAX,
        after.num_blocks == before.num_blocks,
        after.free_blocks == before.free_blocks,
        free_queue_shape(&after.free_queue),
        free_queue_shape(&after.cached_queue),
        after.free_queue.order@ == before.free_queue.order@,
        after.blocks@.dom() == before.blocks@.dom(),
        after.blocks@[bid].tokens@ == before.blocks@[bid].tokens@,
        after.blocks@[bid].refcount as int
            == before.blocks@[bid].refcount as int + 1,
        after.blocks@[bid].hash_value == before.blocks@[bid].hash_value,
        after.blocks@[bid].prefix_depth == before.blocks@[bid].prefix_depth,
        after.blocks@[bid].parent_block == before.blocks@[bid].parent_block,
        forall|other: BlockId| other != bid
            && #[trigger] before.blocks@.contains_key(other)
            ==> after.blocks@[other] == before.blocks@[other],
        before.blocks@[bid].refcount == 0 ==> {
            let q = before.cached_queue.order@;
            let idx = q.index_of(bid);
            &&& q.contains(bid)
            &&& after.cached_queue.order@ == q.subrange(0, idx)
                    + q.subrange(idx + 1, q.len() as int)
            &&& before.blocks@[bid].parent_block is Some ==>
                !q.contains(before.blocks@[bid].parent_block.unwrap())
        },
        before.blocks@[bid].refcount > 0 ==>
            after.cached_queue.order@ == before.cached_queue.order@,
    ensures
        free_queue_valid(after),
{
    let vacant = before.free_queue.order@;
    let bq = before.cached_queue.order@;
    let aq = after.cached_queue.order@;
    let old_rc = before.blocks@[bid].refcount;

    if old_rc == 0 {
        lemma_queue_remove_membership(bq, aq, bid);
    }

    assert(free_queue_membership_valid(after)) by {
        assert forall|b: BlockId| #[trigger] vacant.contains(b) <==>
            b < after.num_blocks && !after.blocks@.contains_key(b)
        by {
            assert(after.blocks@.contains_key(b)
                <==> before.blocks@.contains_key(b));
        }
        assert forall|b: BlockId| #[trigger] aq.contains(b) <==>
            zero_ref_cached_page(after, b)
        by {
            if b == bid {
                assert(!aq.contains(bid));
                assert(after.blocks@[bid].refcount > 0);
            } else if before.blocks@.contains_key(b) {
                assert(after.blocks@[b] == before.blocks@[b]);
            }
        }
    }

    assert(free_queue_partitioned(after)) by {
        assert(after.free_blocks as int == vacant.len());
        assert(vacant.to_set().disjoint(aq.to_set())) by {
            assert forall|b: BlockId| vacant.to_set().contains(b)
                implies !aq.to_set().contains(b) by {
                assert(!bq.to_set().contains(b));
                if aq.contains(b) {
                    assert(bq.contains(b));
                }
            }
        }
        assert(vacant.len() + aq.len() <= after.num_blocks as int) by {
            if old_rc == 0 {
                assert(aq.len() + 1 == bq.len());
            } else {
                assert(aq == bq);
            }
            assert(vacant.len() + bq.len() <= before.num_blocks as int);
        }
    }

    assert(free_queue_topological(after)) by {
        assert forall|child: BlockId| after.blocks@.contains_key(child)
            && after.blocks@[child].prefix_depth > 0
            && after.blocks@[child].parent_block is Some
            && aq.contains(after.blocks@[child].parent_block.unwrap())
            implies #[trigger] aq.contains(child)
                && aq.index_of(child)
                    < aq.index_of(after.blocks@[child].parent_block.unwrap())
        by {
            let parent = after.blocks@[child].parent_block.unwrap();
            if old_rc == 0 {
                assert(child != bid) by {
                    if child == bid {
                        assert(before.blocks@[bid].parent_block == Some(parent));
                        assert(!bq.contains(parent));
                        assert(aq.contains(parent) ==> bq.contains(parent));
                    }
                }
                assert(after.blocks@[child] == before.blocks@[child]);
                assert(parent != bid) by {
                    if parent == bid {
                        assert(!aq.contains(bid));
                    }
                }
                assert(bq.contains(parent)) by {
                    if !bq.contains(parent) {
                        assert(!aq.contains(parent));
                    }
                }
                assert(free_queue_topological(before));
                assert(bq.contains(child));
                assert(bq.index_of(child) < bq.index_of(parent));
                assert(child != bid);
                lemma_queue_remove_preserves_order(
                    bq, aq, bid, child, parent,
                );
            } else {
                assert(aq == bq);
                assert(child != bid) by {
                    if child == bid {
                        assert(!bq.contains(bid));
                    }
                }
                assert(after.blocks@[child] == before.blocks@[child]);
                assert(free_queue_topological(before));
            }
        }
    }
}


#[verifier::spinoff_prover]
#[verifier::rlimit(150)]
pub proof fn lemma_two_queue_after_cached_head_eviction(
    before: &CacheScheduler,
    after: &CacheScheduler,
    bid: BlockId,
)
    requires
        free_queue_valid(before),
        before.cached_queue.head == Some(bid),
        bid < before.num_blocks,
        after.num_blocks == before.num_blocks,
        after.free_blocks as int == before.free_blocks as int + 1,
        free_queue_shape(&after.free_queue),
        free_queue_shape(&after.cached_queue),
        after.free_queue.order@ == Seq::<BlockId>::empty().push(bid)
            + before.free_queue.order@,
        after.cached_queue.order@ == before.cached_queue.order@.subrange(
            1, before.cached_queue.order@.len() as int,
        ),
        after.blocks@.dom() == before.blocks@.dom().remove(bid),
        forall|other: BlockId| other != bid
            && #[trigger] before.blocks@.contains_key(other)
            ==> after.blocks@.contains_key(other)
                && after.blocks@[other] == before.blocks@[other],
    ensures
        free_queue_valid(after),
{
    let bv = before.free_queue.order@;
    let av = after.free_queue.order@;
    let bq = before.cached_queue.order@;
    let aq = after.cached_queue.order@;
    assert(bq.len() > 0);
    assert(bq[0] == bid);
    assert(bq.contains(bid));
    assert(zero_ref_cached_page(before, bid));
    assert(!after.blocks@.contains_key(bid));

    assert(free_queue_membership_valid(after)) by {
        assert forall|b: BlockId| #[trigger] av.contains(b) <==>
            b < after.num_blocks && !after.blocks@.contains_key(b)
        by {
            if b == bid {
                assert(bid < before.num_blocks);
                assert(av[0] == bid);
            } else {
                assert(before.blocks@.contains_key(b)
                    <==> after.blocks@.contains_key(b));
                if av.contains(b) {
                    assert(bv.contains(b)) by {
                        if !bv.contains(b) {
                            assert(b == bid);
                        }
                    }
                }
                if bv.contains(b) {
                    vstd::seq_lib::lemma_seq_concat_contains_all_elements(
                        Seq::<BlockId>::empty().push(bid), bv, b,
                    );
                }
            }
        }
        assert forall|b: BlockId| #[trigger] aq.contains(b) <==>
            zero_ref_cached_page(after, b)
        by {
            if aq.contains(b) {
                vstd::seq_lib::lemma_seq_subrange_elements(
                    bq, 1, bq.len() as int, b,
                );
                assert(bq.contains(b));
                assert(b != bid) by {
                    if b == bid {
                        assert(bq.no_duplicates());
                    }
                }
                assert(after.blocks@[b] == before.blocks@[b]);
            }
            if zero_ref_cached_page(after, b) {
                assert(b != bid);
                assert(after.blocks@[b] == before.blocks@[b]);
                assert(bq.contains(b));
                let i = bq.index_of(b);
                assert(i != 0) by {
                    if i == 0 {
                        assert(bq[i] == b);
                        assert(bq[0] == bid);
                    }
                }
                assert(i >= 1);
                vstd::seq_lib::lemma_seq_subrange_elements(
                    bq, 1, bq.len() as int, b,
                );
            }
        }
    }

    assert(free_queue_partitioned(after)) by {
        assert(after.free_blocks as int == av.len()) by {
            assert(before.free_blocks as int == bv.len());
            assert(av.len() == bv.len() + 1);
        }
        assert(av.to_set().disjoint(aq.to_set())) by {
            assert forall|b: BlockId| av.to_set().contains(b)
                implies !aq.to_set().contains(b) by {
                if b == bid {
                    assert(!aq.contains(bid)) by {
                        if aq.contains(bid) {
                            assert(bq.no_duplicates());
                        }
                    }
                } else {
                    assert(bv.contains(b));
                    if aq.contains(b) {
                        assert(bq.contains(b));
                        assert(bv.to_set().disjoint(bq.to_set()));
                    }
                }
            }
        }
        assert(av.len() + aq.len() <= after.num_blocks as int) by {
            assert(av.len() == bv.len() + 1);
            assert(aq.len() + 1 == bq.len());
            assert(bv.len() + bq.len() <= before.num_blocks as int);
        }
    }

    assert(free_queue_topological(after)) by {
        assert forall|child: BlockId| after.blocks@.contains_key(child)
            && after.blocks@[child].prefix_depth > 0
            && after.blocks@[child].parent_block is Some
            && aq.contains(after.blocks@[child].parent_block.unwrap())
            implies #[trigger] aq.contains(child)
                && aq.index_of(child)
                    < aq.index_of(after.blocks@[child].parent_block.unwrap())
        by {
            let parent = after.blocks@[child].parent_block.unwrap();
            assert(child != bid);
            assert(parent != bid) by {
                if parent == bid {
                    assert(!aq.contains(bid)) by {
                        if aq.contains(bid) {
                            assert(bq.no_duplicates());
                        }
                    }
                }
            }
            assert(after.blocks@[child] == before.blocks@[child]);
            assert(bq.contains(parent)) by {
                vstd::seq_lib::lemma_seq_subrange_elements(
                    bq, 1, bq.len() as int, parent,
                );
            }
            assert(free_queue_topological(before));
            assert(bq.contains(child));
            assert(bq.index_of(child) < bq.index_of(parent));
            assert(bq.index_of(bid) == 0) by {
                assert(bq.no_duplicates());
            }
            assert(aq == bq.subrange(0, bq.index_of(bid))
                + bq.subrange(bq.index_of(bid) + 1, bq.len() as int));
            lemma_queue_remove_preserves_order(
                bq, aq, bid, child, parent,
            );
        }
    }
}

// Stable-boundary queue reconciliation for request release.  `vacant` is the
// set of request pages that ceased to be resident; `cached` is the reverse
// request-chain sequence of pages whose refcount newly reached zero.
#[verifier::spinoff_prover]
#[verifier::rlimit(200)]
pub proof fn lemma_two_queue_after_request_release(
    before: &CacheScheduler,
    after: &CacheScheduler,
    vacant: Seq<BlockId>,
    cached: Seq<BlockId>,
)
    requires
        cs_valid(before),
        cs_valid(after),
        free_queue_valid(before),
        after.num_blocks == before.num_blocks,
        free_queue_shape(&after.free_queue),
        free_queue_shape(&after.cached_queue),
        after.free_queue.order@ == vacant + before.free_queue.order@,
        after.cached_queue.order@ == before.cached_queue.order@ + cached,
        after.free_blocks as int
            == before.free_blocks as int + vacant.len(),
        forall|b: BlockId| #[trigger] vacant.contains(b)
            ==> before.blocks@.contains_key(b)
                && !after.blocks@.contains_key(b),
        forall|b: BlockId| #[trigger] before.blocks@.contains_key(b)
            && !after.blocks@.contains_key(b)
            ==> vacant.contains(b),
        forall|b: BlockId| #[trigger] after.blocks@.contains_key(b)
            ==> before.blocks@.contains_key(b)
                && after.blocks@[b].prefix_depth
                    == before.blocks@[b].prefix_depth
                && after.blocks@[b].parent_block
                    == before.blocks@[b].parent_block,
        forall|b: BlockId| #[trigger] zero_ref_cached_page(before, b)
            ==> after.blocks@.contains_key(b)
                && after.blocks@[b] == before.blocks@[b],
        forall|b: BlockId| #[trigger] cached.contains(b)
            ==> zero_ref_cached_page(after, b)
                && !zero_ref_cached_page(before, b),
        forall|b: BlockId| #[trigger] zero_ref_cached_page(after, b)
            && !zero_ref_cached_page(before, b)
            ==> cached.contains(b),
        forall|child: BlockId| after.blocks@.contains_key(child)
            && after.blocks@[child].prefix_depth > 0
            && after.blocks@[child].parent_block is Some
            && cached.contains(child)
            && cached.contains(after.blocks@[child].parent_block.unwrap())
            ==> #[trigger] cached.index_of(child)
                < cached.index_of(after.blocks@[child].parent_block.unwrap()),
    ensures
        free_queue_valid(after),
{
    let bv = before.free_queue.order@;
    let bq = before.cached_queue.order@;
    let av = after.free_queue.order@;
    let aq = after.cached_queue.order@;

    assert(free_queue_membership_valid(after)) by {
        assert forall|b: BlockId| #[trigger] av.contains(b) <==>
            b < after.num_blocks && !after.blocks@.contains_key(b)
        by {
            if av.contains(b) {
                if vacant.contains(b) {
                    assert(before.blocks@.contains_key(b));
                    assert(b < before.num_blocks) by {
                        assert(blocks_dom_in_range(before));
                    }
                } else {
                    assert(bv.contains(b));
                    assert(!before.blocks@.contains_key(b));
                    assert(!after.blocks@.contains_key(b)) by {
                        if after.blocks@.contains_key(b) {
                            assert(before.blocks@.contains_key(b));
                        }
                    }
                    assert(b < before.num_blocks);
                }
            }
            if b < after.num_blocks && !after.blocks@.contains_key(b) {
                if before.blocks@.contains_key(b) {
                    assert(vacant.contains(b));
                    vstd::seq_lib::lemma_seq_concat_contains_all_elements(
                        vacant, bv, b,
                    );
                } else {
                    assert(bv.contains(b));
                    vstd::seq_lib::lemma_seq_concat_contains_all_elements(
                        vacant, bv, b,
                    );
                }
            }
        }
        assert forall|b: BlockId| #[trigger] aq.contains(b) <==>
            zero_ref_cached_page(after, b)
        by {
            if aq.contains(b) {
                if bq.contains(b) {
                    assert(zero_ref_cached_page(before, b));
                    assert(after.blocks@[b] == before.blocks@[b]);
                } else {
                    assert(cached.contains(b));
                }
            }
            if zero_ref_cached_page(after, b) {
                if zero_ref_cached_page(before, b) {
                    assert(bq.contains(b));
                    vstd::seq_lib::lemma_seq_concat_contains_all_elements(
                        bq, cached, b,
                    );
                } else {
                    assert(cached.contains(b));
                    vstd::seq_lib::lemma_seq_concat_contains_all_elements(
                        bq, cached, b,
                    );
                }
            }
        }
    }

    assert(free_queue_partitioned(after)) by {
        assert(after.free_blocks as int == av.len()) by {
            assert(before.free_blocks as int == bv.len());
            assert(av.len() == vacant.len() + bv.len());
        }
        assert(av.to_set().disjoint(aq.to_set())) by {
            assert forall|b: BlockId| av.to_set().contains(b)
                implies !aq.to_set().contains(b) by {
                assert(!after.blocks@.contains_key(b));
                if aq.contains(b) {
                    assert(zero_ref_cached_page(after, b));
                }
            }
        }
        assert(aq.to_set().subset_of(after.blocks@.dom())) by {
            assert forall|b: BlockId| aq.to_set().contains(b)
                implies after.blocks@.dom().contains(b) by {
                assert(zero_ref_cached_page(after, b));
            }
        }
        aq.unique_seq_to_set();
        vstd::set_lib::lemma_len_subset(
            aq.to_set(), after.blocks@.dom(),
        );
        assert(av.len() + aq.len() <= after.num_blocks as int) by {
            assert(block_count_valid(after));
            assert(after.free_blocks as int + after.blocks@.dom().len() as int
                == after.num_blocks as int);
        }
    }

    assert(free_queue_topological(after)) by {
        assert forall|child: BlockId| after.blocks@.contains_key(child)
            && after.blocks@[child].prefix_depth > 0
            && after.blocks@[child].parent_block is Some
            && aq.contains(after.blocks@[child].parent_block.unwrap())
            implies #[trigger] aq.contains(child)
                && aq.index_of(child)
                    < aq.index_of(after.blocks@[child].parent_block.unwrap())
        by {
            let parent = after.blocks@[child].parent_block.unwrap();
            assert(before.blocks@.contains_key(child));
            assert(before.blocks@[child].parent_block == Some(parent));
            assert(after.blocks@.contains_key(parent)) by {
                assert(zero_ref_cached_page(after, parent));
            }
            assert(before.blocks@.contains_key(parent));

            if bq.contains(parent) {
                assert(zero_ref_cached_page(before, parent));
                lemma_provenance_parent_refcount_ge_child(
                    before, child, parent,
                );
                assert(before.blocks@[child].refcount == 0);
                assert(zero_ref_cached_page(before, child));
                assert(bq.contains(child));
                assert(free_queue_topological(before));
                assert(bq.index_of(child) < bq.index_of(parent));
                let ci = bq.index_of(child);
                let pi = bq.index_of(parent);
                assert(aq[ci] == child);
                assert(aq[pi] == parent);
                assert(aq.index_of(child) == ci) by {
                    assert(aq.no_duplicates());
                }
                assert(aq.index_of(parent) == pi) by {
                    assert(aq.no_duplicates());
                }
            } else {
                assert(cached.contains(parent));
                lemma_provenance_parent_refcount_ge_child(
                    after, child, parent,
                );
                assert(after.blocks@[parent].refcount == 0);
                assert(after.blocks@[child].refcount == 0);
                assert(zero_ref_cached_page(after, child));
                assert(aq.contains(child));
                if bq.contains(child) {
                    let ci = bq.index_of(child);
                    let pi = bq.len() + cached.index_of(parent);
                    assert(aq[ci] == child);
                    assert(aq[pi] == parent);
                    assert(aq.index_of(child) == ci) by {
                        assert(aq.no_duplicates());
                    }
                    assert(aq.index_of(parent) == pi) by {
                        assert(aq.no_duplicates());
                    }
                    assert(ci < pi);
                } else {
                    assert(cached.contains(child));
                    assert(cached.index_of(child) < cached.index_of(parent));
                    let ci = bq.len() + cached.index_of(child);
                    let pi = bq.len() + cached.index_of(parent);
                    assert(aq[ci] == child);
                    assert(aq[pi] == parent);
                    assert(aq.index_of(child) == ci) by {
                        assert(aq.no_duplicates());
                    }
                    assert(aq.index_of(parent) == pi) by {
                        assert(aq.no_duplicates());
                    }
                }
            }
        }
    }
}

// Stable success relation exported by the executable eviction primitive.  It
// is deliberately phrased as a frame, so the engine-level semantic proof can
// show that removing a candidate cannot invalidate any surviving registered
// prefix certificate.
pub open spec fn cached_leaf_eviction_frame(
    pre: &CacheScheduler,
    post: &CacheScheduler,
    bid: BlockId,
) -> bool {
    &&& zero_ref_provenance_leaf(pre, bid)
    &&& post.config == pre.config
    &&& post.num_blocks == pre.num_blocks
    &&& post.running@ == pre.running@
    &&& post.waiting@ == pre.waiting@
    &&& post.live_requests@ == pre.live_requests@
    &&& post.request_residency@ == pre.request_residency@
    &&& post.blocks@.dom() == pre.blocks@.dom().remove(bid)
    &&& post.free_blocks as int == pre.free_blocks as int + 1
    &&& forall|other: BlockId|
        other != bid && #[trigger] pre.blocks@.contains_key(other)
        ==> post.blocks@.contains_key(other)
            && post.blocks@[other] == pre.blocks@[other]
    &&& forall|h: u64| #[trigger] post.hash_to_block@.contains_key(h)
        ==> pre.hash_to_block@.contains_key(h)
            && post.hash_to_block@[h] == pre.hash_to_block@[h]
            && pre.hash_to_block@[h] != bid
    &&& forall|h: u64| #[trigger] pre.hash_to_block@.contains_key(h)
        && pre.hash_to_block@[h] != bid
        ==> post.hash_to_block@.contains_key(h)
            && post.hash_to_block@[h] == pre.hash_to_block@[h]
}

pub open spec fn blocks_dom_in_range(cs: &CacheScheduler) -> bool {
    forall|bid: BlockId|
        #[trigger] cs.blocks@.contains_key(bid) ==> bid < cs.num_blocks
}

// The counting invariant: `free_blocks + |blocks.dom()| == num_blocks`.
// Together with exact vacant-queue membership, this proves that a sufficient
// `free_blocks` count supplies enough ids for allocation and that returning a
// block to the vacant queue keeps the physical pool balanced.
pub open spec fn block_count_valid(cs: &CacheScheduler) -> bool {
    cs.free_blocks as int + cs.blocks@.dom().len() as int == cs.num_blocks as int
}


// Exact observable shape of a successful no-dedup prefill allocation.
// This is deliberately separate from `cs_valid`: scheduler callers need
// concrete facts about the new residency and slot mapping, while `cs_valid`
// is the global preservation invariant.
pub open spec fn allocate_prefill_success(
    pre: &CacheScheduler,
    post: &CacheScheduler,
    rid: RequestId,
    prompt_tokens: Seq<TokenId>,
) -> bool {
    let n = prompt_tokens.len() as nat;
    let blocks_needed = blocks_needed_for(n);
    post.request_residency@.contains_key(rid)
    && post.request_residency@[rid].cached_prefix_blocks == 0
    && post.request_residency@[rid].block_ids@.len() == blocks_needed as int
    && post.request_residency@[rid].block_ids@.no_duplicates()
    && post.request_residency@[rid].slot_mapping@.len() == prompt_tokens.len()
    && post.free_blocks as int == pre.free_blocks as int - blocks_needed as int
    && (forall|k: int|
        #![trigger post.request_residency@[rid].block_ids@[k]]
        0 <= k < post.request_residency@[rid].block_ids@.len()
        ==> !pre.blocks@.contains_key(post.request_residency@[rid].block_ids@[k])
            && post.blocks@.contains_key(post.request_residency@[rid].block_ids@[k])
            && post.blocks@[post.request_residency@[rid].block_ids@[k]].refcount == 1
            && post.blocks@[post.request_residency@[rid].block_ids@[k]].hash_value == 0)
    && (forall|i: int|
        #![trigger post.request_residency@[rid].slot_mapping@[i]]
        0 <= i < prompt_tokens.len()
        ==> post.request_residency@[rid].slot_mapping@[i] as int
            == block_table_slot(post.request_residency@[rid].block_ids@, i as nat) as int)
    && (forall|i: int|
        #![trigger prompt_tokens[i]]
        0 <= i < prompt_tokens.len()
        ==> {
            let bid = post.request_residency@[rid].block_ids@[i / BLOCK_SIZE_SPEC as int];
            let off = i % BLOCK_SIZE_SPEC as int;
            post.blocks@[bid].tokens@[off] == prompt_tokens[i]
        })
}

pub open spec fn append_token_common_frame(
    pre: &CacheScheduler,
    post: &CacheScheduler,
    rid: RequestId,
) -> bool {
    post.config == pre.config
    && post.num_blocks == pre.num_blocks
    && post.live_requests@ == pre.live_requests@
    && post.running@ == pre.running@
    && post.waiting@ == pre.waiting@
    && post.hash_to_block@ == pre.hash_to_block@
    && (forall|other: RequestId|
        other != rid && #[trigger] pre.request_residency@.contains_key(other)
        ==> post.request_residency@.contains_key(other)
            && post.request_residency@[other] == pre.request_residency@[other])
}

pub open spec fn append_token_tail_append_success(
    pre: &CacheScheduler,
    post: &CacheScheduler,
    rid: RequestId,
    token: TokenId,
) -> bool {
    if pre.request_residency@.contains_key(rid)
        && pre.request_residency@[rid].block_ids@.len() > 0 {
        let old_residency = pre.request_residency@[rid];
        let last_index = old_residency.block_ids@.len() - 1;
        let last_bid = old_residency.block_ids@[last_index];
        if pre.blocks@.contains_key(last_bid)
            && pre.blocks@[last_bid].tokens@.len() < BLOCK_SIZE_SPEC as int {
            let old_tail = pre.blocks@[last_bid];
            let last_slot = last_bid * BLOCK_SIZE + old_tail.tokens@.len() as u64;
            append_token_common_frame(pre, post, rid)
            && post.free_blocks == pre.free_blocks
            && post.blocks@.dom() == pre.blocks@.dom()
            && post.request_residency@.contains_key(rid)
            && post.request_residency@[rid].block_ids@ == old_residency.block_ids@
            && post.request_residency@[rid].cached_prefix_blocks == old_residency.cached_prefix_blocks
            && post.request_residency@[rid].slot_mapping@.len() == 1
            && post.request_residency@[rid].slot_mapping@[0] == last_slot
            && post.blocks@[last_bid].tokens@ == old_tail.tokens@.push(token)
            && post.blocks@[last_bid].refcount == old_tail.refcount
            && post.blocks@[last_bid].hash_value == old_tail.hash_value
            && post.blocks@[last_bid].prefix_depth == old_tail.prefix_depth
            && post.blocks@[last_bid].parent_block == old_tail.parent_block
            && (forall|bid: BlockId|
                bid != last_bid && #[trigger] pre.blocks@.contains_key(bid)
                ==> post.blocks@.contains_key(bid) && post.blocks@[bid] == pre.blocks@[bid])
        } else {
            false
        }
    } else {
        false
    }
}

pub open spec fn append_token_new_tail_success(
    pre: &CacheScheduler,
    post: &CacheScheduler,
    rid: RequestId,
    token: TokenId,
) -> bool {
    if pre.request_residency@.contains_key(rid)
        && pre.request_residency@[rid].block_ids@.len() > 0 {
        let old_residency = pre.request_residency@[rid];
        let last_index = old_residency.block_ids@.len() - 1;
        let last_bid = old_residency.block_ids@[last_index];
        if pre.blocks@.contains_key(last_bid)
            && pre.blocks@[last_bid].tokens@.len() == BLOCK_SIZE_SPEC as int
            && pre.free_blocks > 0 {
            append_token_common_frame(pre, post, rid)
            && post.free_blocks as int == pre.free_blocks as int - 1
            && post.request_residency@.contains_key(rid)
            && post.request_residency@[rid].block_ids@.len() == old_residency.block_ids@.len() + 1
            && post.request_residency@[rid].block_ids@.subrange(
                0,
                old_residency.block_ids@.len() as int,
            ) == old_residency.block_ids@
            && post.request_residency@[rid].cached_prefix_blocks == old_residency.cached_prefix_blocks
            && post.request_residency@[rid].slot_mapping@.len() == 1
            && {
                let new_bid = post.request_residency@[rid].block_ids@[
                    old_residency.block_ids@.len() as int
                ];
                !pre.blocks@.contains_key(new_bid)
                && post.blocks@.contains_key(new_bid)
                && new_bid < post.num_blocks
                && post.request_residency@[rid].slot_mapping@[0] == new_bid * BLOCK_SIZE
                && post.blocks@[new_bid].tokens@ == seq![token]
                && post.blocks@[new_bid].refcount == 1
                && post.blocks@[new_bid].hash_value == 0
                && post.blocks@[new_bid].prefix_depth == 0
                && post.blocks@[new_bid].parent_block is None
                && post.blocks@.dom() == pre.blocks@.dom().insert(new_bid)
                && (forall|bid: BlockId|
                    bid != new_bid && #[trigger] pre.blocks@.contains_key(bid)
                    ==> post.blocks@.contains_key(bid) && post.blocks@[bid] == pre.blocks@[bid])
            }
        } else {
            false
        }
    } else {
        false
    }
}

pub open spec fn token_placement_prefix(
    blocks: Map<BlockId, BlockEntry>,
    block_ids: Seq<BlockId>,
    prompt_tokens: Seq<TokenId>,
    limit: int,
) -> bool {
    forall|p: int|
        #![trigger prompt_tokens[p]]
        #![trigger token_placement_at(blocks, block_ids, prompt_tokens, p)]
        0 <= p < limit
        ==> token_placement_at(blocks, block_ids, prompt_tokens, p)
}

pub open spec fn token_placement_at(
    blocks: Map<BlockId, BlockEntry>,
    block_ids: Seq<BlockId>,
    prompt_tokens: Seq<TokenId>,
    p: int,
) -> bool {
    let j = p / BLOCK_SIZE_SPEC as int;
    let off = p % BLOCK_SIZE_SPEC as int;
    j < block_ids.len()
    && blocks.contains_key(block_ids[j])
    && off < blocks[block_ids[j]].tokens@.len()
    && blocks[block_ids[j]].tokens@[off] == prompt_tokens[p]
}

// Registered prompt pages form an exact physical predecessor chain.  The
// one-based depth prevents the same chain from being accepted at a shifted
// logical position; the parent edge prevents hash collisions from splicing a
// page whose K/V was computed under a different prefix.
pub open spec fn registered_prefix_chain(
    blocks: Map<BlockId, BlockEntry>,
    block_ids: Seq<BlockId>,
) -> bool {
    forall|j: int| #![trigger block_ids[j]]
        0 <= j < block_ids.len() ==> {
        let bid = block_ids[j];
        &&& blocks.contains_key(bid)
        &&& blocks[bid].prefix_depth as int == j + 1
        &&& blocks[bid].parent_block
            == if j == 0 { None } else { Some(block_ids[j - 1]) }
    }
}

// Registered ancestry is prefix-closed.  This elementary slicing lemma keeps
// subrange index normalization out of the larger engine-step proof.
pub proof fn lemma_registered_prefix_chain_prefix(
    blocks: Map<BlockId, BlockEntry>,
    ids: Seq<BlockId>,
    n: int,
)
    requires
        registered_prefix_chain(blocks, ids),
        0 <= n <= ids.len(),
    ensures
        registered_prefix_chain(blocks, ids.subrange(0, n)),
{
    reveal(registered_prefix_chain);
    assert forall|j: int|
        #![trigger blocks[ids.subrange(0, n)[j]].prefix_depth]
        0 <= j < ids.subrange(0, n).len() implies {
            let bid = ids.subrange(0, n)[j];
            &&& blocks.contains_key(bid)
            &&& blocks[bid].prefix_depth as int == j + 1
            &&& blocks[bid].parent_block
                == if j == 0 { None }
                    else { Some(ids.subrange(0, n)[j - 1]) }
    }
    by {
        assert(registered_prefix_chain(blocks, ids));
        assert(ids.subrange(0, n)[j] == ids[j]);
        assert(blocks[ids[j]].prefix_depth as int == j + 1);
        assert(blocks.contains_key(ids[j]));
        if j > 0 {
            assert(ids.subrange(0, n)[j - 1] == ids[j - 1]);
        }
    }
}

pub proof fn lemma_registered_prefix_chain_transfer(
    before: Map<BlockId, BlockEntry>,
    after: Map<BlockId, BlockEntry>,
    ids: Seq<BlockId>,
)
    requires
        registered_prefix_chain(before, ids),
        forall|j: int| 0 <= j < ids.len() ==> {
            let bid = #[trigger] ids[j];
            &&& after.contains_key(bid)
            &&& after[bid].prefix_depth == before[bid].prefix_depth
            &&& after[bid].parent_block == before[bid].parent_block
        },
    ensures
        registered_prefix_chain(after, ids),
{
    assert forall|j: int| #![trigger after[ids[j]].prefix_depth]
        0 <= j < ids.len() implies {
            let bid = ids[j];
            &&& after.contains_key(bid)
            &&& after[bid].prefix_depth as int == j + 1
            &&& after[bid].parent_block
                == if j == 0 { None } else { Some(ids[j - 1]) }
        }
    by {
    }
}

// A holder of the last page in a registered chain necessarily holds the
// entire same physical chain at the same logical indices.  The proof walks
// the immutable parent links backwards.  This is the scheduler-side bridge
// from a registry target to a donor's complete token context.
pub proof fn lemma_registered_chain_matches_holder_prefix(
    cs: &CacheScheduler,
    chain: Seq<BlockId>,
    donor: RequestId,
    k: int,
)
    requires
        cs_valid(cs),
        registered_prefix_chain(cs.blocks@, chain),
        cs.request_residency@.contains_key(donor),
        0 <= k < chain.len(),
        cs.request_residency@[donor].block_ids@.contains(chain[k]),
    ensures ({
        let donor_ids = cs.request_residency@[donor].block_ids@;
        &&& k < donor_ids.len()
        &&& donor_ids.subrange(0, k + 1) == chain.subrange(0, k + 1)
    }),
    decreases k,
{
    let donor_ids = cs.request_residency@[donor].block_ids@;
    let bid = chain[k];
    let q = donor_ids.index_of(bid);
    assert(0 <= q < donor_ids.len());
    assert(donor_ids[q] == bid);
    assert(cs.blocks@.contains_key(bid));
    assert(cs.blocks@[bid].prefix_depth as int == k + 1);
    assert(registered_provenance_aligned(cs));
    assert(cs.blocks@[donor_ids[q]].prefix_depth as int == q + 1);
    assert(q == k);
    assert(donor_ids[k] == chain[k]);
    if k > 0 {
        assert(cs.blocks@[bid].parent_block == Some(chain[k - 1]));
        assert(cs.blocks@[bid].parent_block == Some(donor_ids[k - 1]));
        assert(donor_ids[k - 1] == chain[k - 1]);
        assert(donor_ids.contains(chain[k - 1]));
        lemma_registered_chain_matches_holder_prefix(cs, chain, donor, k - 1);
        assert(donor_ids.subrange(0, k) == chain.subrange(0, k));
    }
    assert(donor_ids.subrange(0, k + 1) =~= chain.subrange(0, k + 1)) by {
        assert forall|j: int| 0 <= j < k + 1 implies
            #[trigger] donor_ids.subrange(0, k + 1)[j]
                == chain.subrange(0, k + 1)[j]
        by {
            if j < k {
                assert(donor_ids.subrange(0, k)[j] == chain.subrange(0, k)[j]);
            } else {
                assert(j == k);
            }
        }
    }
}

// Two registered chains that reach the same physical page must have the same
// logical depth and the same complete physical ancestry through that page.
// This is the residency-independent counterpart of
// `lemma_registered_chain_matches_holder_prefix`, used after an admission has
// finished and only its stable materialized plan row remains.
pub proof fn lemma_registered_chains_match_through_target(
    blocks: Map<BlockId, BlockEntry>,
    left: Seq<BlockId>,
    right: Seq<BlockId>,
    left_index: int,
    right_index: int,
)
    requires
        registered_prefix_chain(blocks, left),
        registered_prefix_chain(blocks, right),
        0 <= left_index < left.len(),
        0 <= right_index < right.len(),
        left[left_index] == right[right_index],
    ensures
        left_index == right_index,
        left.subrange(0, left_index + 1)
            == right.subrange(0, right_index + 1),
    decreases left_index,
{
    let bid = left[left_index];
    assert(blocks[bid].prefix_depth as int == left_index + 1);
    assert(blocks[bid].prefix_depth as int == right_index + 1);
    assert(left_index == right_index);
    if left_index > 0 {
        assert(blocks[bid].parent_block == Some(left[left_index - 1]));
        assert(blocks[bid].parent_block == Some(right[right_index - 1]));
        assert(left[left_index - 1] == right[right_index - 1]);
        lemma_registered_chains_match_through_target(
            blocks, left, right, left_index - 1, right_index - 1,
        );
        assert(left.subrange(0, left_index)
            == right.subrange(0, right_index));
    }
    assert(left.subrange(0, left_index + 1)
        =~= right.subrange(0, right_index + 1)) by {
        assert forall|j: int| 0 <= j < left_index + 1 implies
            #[trigger] left.subrange(0, left_index + 1)[j]
                == right.subrange(0, right_index + 1)[j]
        by {
            if j < left_index {
                assert(left.subrange(0, left_index)[j]
                    == right.subrange(0, right_index)[j]);
            } else {
                assert(j == left_index);
            }
        }
    }
}

pub proof fn lemma_token_placement_prefix_at(
    blocks: Map<BlockId, BlockEntry>,
    block_ids: Seq<BlockId>,
    prompt_tokens: Seq<TokenId>,
    limit: int,
    p: int,
)
    requires
        token_placement_prefix(blocks, block_ids, prompt_tokens, limit),
        0 <= p < limit,
        limit <= prompt_tokens.len(),
    ensures
        token_placement_at(blocks, block_ids, prompt_tokens, p),
{
    reveal(token_placement_prefix);
    reveal(token_placement_at);
    assert(0 <= p < prompt_tokens.len());
    assert(prompt_tokens[p] == prompt_tokens[p]);
    assert(token_placement_at(blocks, block_ids, prompt_tokens, p));
}

pub proof fn lemma_token_placement_prefix_shrink(
    blocks: Map<BlockId, BlockEntry>,
    block_ids: Seq<BlockId>,
    tokens: Seq<TokenId>,
    full: int,
    short: int,
)
    requires
        token_placement_prefix(blocks, block_ids, tokens, full),
        0 <= short <= full,
        full <= tokens.len(),
    ensures
        token_placement_prefix(blocks, block_ids, tokens, short),
{
    reveal(token_placement_prefix);
    assert forall|p: int| 0 <= p < short implies
        #[trigger] token_placement_at(blocks, block_ids, tokens, p)
    by {
        lemma_token_placement_prefix_at(
            blocks, block_ids, tokens, full, p,
        );
    }
}

// Exact token placement is insensitive to scheduler-side metadata changes
// (refcounts and hash stamps) when every referenced block keeps its tokens.
// This is the framing rule used when a new admission mutates shared-prefix
// metadata while preserving older running requests' logical histories.
pub proof fn lemma_token_placement_prefix_transfer(
    pre_blocks: Map<BlockId, BlockEntry>,
    post_blocks: Map<BlockId, BlockEntry>,
    block_ids: Seq<BlockId>,
    tokens: Seq<TokenId>,
    limit: int,
)
    requires
        token_placement_prefix(pre_blocks, block_ids, tokens, limit),
        limit <= tokens.len(),
        forall|bid: BlockId| #[trigger] pre_blocks.contains_key(bid)
            ==> post_blocks.contains_key(bid)
                && post_blocks[bid].tokens@ == pre_blocks[bid].tokens@,
    ensures
        token_placement_prefix(post_blocks, block_ids, tokens, limit),
{
    assert forall|p: int|
        #![trigger token_placement_at(post_blocks, block_ids, tokens, p)]
        0 <= p < limit
        implies token_placement_at(post_blocks, block_ids, tokens, p)
    by {
        lemma_token_placement_prefix_at(pre_blocks, block_ids, tokens, limit, p);
        reveal(token_placement_at);
        let j = p / BLOCK_SIZE_SPEC as int;
        assert(pre_blocks.contains_key(block_ids[j]));
        assert(post_blocks.contains_key(block_ids[j]));
        assert(post_blocks[block_ids[j]].tokens@ == pre_blocks[block_ids[j]].tokens@);
    }
}

// A narrower framing rule when unrelated old blocks may have been freed: only
// the blocks named by this logical row need to survive token-identically.
pub proof fn lemma_token_placement_prefix_transfer_for_ids(
    pre_blocks: Map<BlockId, BlockEntry>,
    post_blocks: Map<BlockId, BlockEntry>,
    block_ids: Seq<BlockId>,
    tokens: Seq<TokenId>,
    limit: int,
)
    requires
        token_placement_prefix(pre_blocks, block_ids, tokens, limit),
        limit <= tokens.len(),
        forall|j: int| 0 <= j < block_ids.len()
            && #[trigger] pre_blocks.contains_key(block_ids[j])
            ==> post_blocks.contains_key(block_ids[j])
                && post_blocks[block_ids[j]].tokens@ == pre_blocks[block_ids[j]].tokens@,
    ensures
        token_placement_prefix(post_blocks, block_ids, tokens, limit),
{
    assert forall|p: int|
        #![trigger token_placement_at(post_blocks, block_ids, tokens, p)]
        0 <= p < limit
        implies token_placement_at(post_blocks, block_ids, tokens, p)
    by {
        lemma_token_placement_prefix_at(pre_blocks, block_ids, tokens, limit, p);
        reveal(token_placement_at);
        let j = p / BLOCK_SIZE_SPEC as int;
        assert(0 <= j < block_ids.len());
        assert(pre_blocks.contains_key(block_ids[j]));
        assert(post_blocks.contains_key(block_ids[j]));
        assert(post_blocks[block_ids[j]].tokens@ == pre_blocks[block_ids[j]].tokens@);
    }
}

// Extend exact placement when the logical history grows into a partial tail.
pub proof fn lemma_token_placement_prefix_tail_append(
    pre_blocks: Map<BlockId, BlockEntry>,
    post_blocks: Map<BlockId, BlockEntry>,
    block_ids: Seq<BlockId>,
    tokens: Seq<TokenId>,
    token: TokenId,
)
    requires
        block_ids.len() >= 1,
        pre_blocks.contains_key(block_ids[block_ids.len() - 1]),
        tokens.len() == (block_ids.len() - 1) * (BLOCK_SIZE_SPEC as int)
            + pre_blocks[block_ids[block_ids.len() - 1]].tokens@.len(),
        1 <= pre_blocks[block_ids[block_ids.len() - 1]].tokens@.len()
            < BLOCK_SIZE_SPEC as int,
        token_placement_prefix(pre_blocks, block_ids, tokens, tokens.len() as int),
        post_blocks.contains_key(block_ids[block_ids.len() - 1]),
        post_blocks[block_ids[block_ids.len() - 1]].tokens@
            == pre_blocks[block_ids[block_ids.len() - 1]].tokens@.push(token),
        forall|bid: BlockId| bid != block_ids[block_ids.len() - 1]
            && #[trigger] pre_blocks.contains_key(bid)
            ==> post_blocks.contains_key(bid) && post_blocks[bid] == pre_blocks[bid],
    ensures
        token_placement_prefix(
            post_blocks,
            block_ids,
            tokens.push(token),
            tokens.len() as int + 1,
        ),
{
    let bs = BLOCK_SIZE_SPEC as int;
    let last = block_ids[block_ids.len() - 1];
    let tail = pre_blocks[last].tokens@.len() as int;
    assert forall|p: int|
        #![trigger token_placement_at(post_blocks, block_ids, tokens.push(token), p)]
        0 <= p < tokens.len() as int + 1
        implies token_placement_at(post_blocks, block_ids, tokens.push(token), p)
    by {
        let j = p / bs;
        let off = p % bs;
        reveal(token_placement_at);
        if p < tokens.len() as int {
            lemma_token_placement_prefix_at(
                pre_blocks, block_ids, tokens, tokens.len() as int, p,
            );
            assert(token_placement_at(pre_blocks, block_ids, tokens, p));
            assert(tokens.push(token)[p] == tokens[p]);
            if block_ids[j] == last {
                assert(post_blocks.contains_key(last));
                assert(off < pre_blocks[last].tokens@.len());
                assert(post_blocks[last].tokens@[off]
                    == pre_blocks[last].tokens@[off]);
            } else {
                assert(post_blocks.contains_key(block_ids[j]));
                assert(post_blocks[block_ids[j]] == pre_blocks[block_ids[j]]);
            }
        } else {
            assert(p == tokens.len() as int);
            vstd::arithmetic::div_mod::lemma_fundamental_div_mod_converse_div(
                p, bs, block_ids.len() - 1, tail,
            );
            vstd::arithmetic::div_mod::lemma_fundamental_div_mod_converse_mod(
                p, bs, block_ids.len() - 1, tail,
            );
            assert(j == block_ids.len() - 1);
            assert(off == tail);
            assert(j < block_ids.len());
            assert(block_ids[j] == last);
            assert(post_blocks.contains_key(last));
            assert(off < post_blocks[last].tokens@.len());
            assert(post_blocks[last].tokens@[off] == token);
            assert(tokens.push(token)[p] == token);
        }
    }
}

// Extend exact placement when a full tail forces allocation of a fresh page.
pub proof fn lemma_token_placement_prefix_new_tail(
    pre_blocks: Map<BlockId, BlockEntry>,
    post_blocks: Map<BlockId, BlockEntry>,
    pre_ids: Seq<BlockId>,
    post_ids: Seq<BlockId>,
    tokens: Seq<TokenId>,
    token: TokenId,
    new_bid: BlockId,
)
    requires
        pre_ids.len() >= 1,
        tokens.len() == pre_ids.len() * (BLOCK_SIZE_SPEC as int),
        token_placement_prefix(pre_blocks, pre_ids, tokens, tokens.len() as int),
        post_ids == pre_ids.push(new_bid),
        post_blocks.contains_key(new_bid),
        post_blocks[new_bid].tokens@ == seq![token],
        forall|bid: BlockId| #[trigger] pre_blocks.contains_key(bid)
            ==> post_blocks.contains_key(bid) && post_blocks[bid] == pre_blocks[bid],
    ensures
        token_placement_prefix(
            post_blocks,
            post_ids,
            tokens.push(token),
            tokens.len() as int + 1,
        ),
{
    let bs = BLOCK_SIZE_SPEC as int;
    assert forall|p: int|
        #![trigger token_placement_at(post_blocks, post_ids, tokens.push(token), p)]
        0 <= p < tokens.len() as int + 1
        implies token_placement_at(post_blocks, post_ids, tokens.push(token), p)
    by {
        let j = p / bs;
        let off = p % bs;
        if p < tokens.len() as int {
            lemma_token_placement_prefix_at(
                pre_blocks, pre_ids, tokens, tokens.len() as int, p,
            );
            assert(token_placement_at(pre_blocks, pre_ids, tokens, p));
            assert(j < pre_ids.len());
            assert(post_ids[j] == pre_ids[j]);
            assert(post_blocks[pre_ids[j]] == pre_blocks[pre_ids[j]]);
        } else {
            assert(p == tokens.len() as int);
            vstd::arithmetic::div_mod::lemma_fundamental_div_mod_converse_div(
                p, bs, pre_ids.len() as int, 0,
            );
            vstd::arithmetic::div_mod::lemma_fundamental_div_mod_converse_mod(
                p, bs, pre_ids.len() as int, 0,
            );
            assert(j == pre_ids.len());
            assert(off == 0);
            assert(post_ids[j] == new_bid);
            assert(post_blocks[new_bid].tokens@[0] == token);
            assert(tokens.push(token)[p] == token);
        }
    }
}

// Aggregated invariant.  `init`, `plan`, and `commit` preserve this.
// Registry hygiene: hash 0 is reserved as the
// "unregistered" stamp carried by freshly allocated and append-created
// blocks, so the registry never uses it as a key.  `publish_full_prefix_pages`
// skips a computed chain hash of 0 (conservative: that block just never
// becomes reusable).  Gives "hash_value == 0 ⇒ not a registry target"
// through `hash_to_block_consistent`.
pub open spec fn hash_to_block_no_zero(cs: &CacheScheduler) -> bool {
    !cs.hash_to_block@.contains_key(0u64)
}

pub open spec fn cs_valid(cs: &CacheScheduler) -> bool {
    live_covers_queue(cs)
    && live_requests_accepted(cs)
    && queue_disjoint(cs)
    && running_unique(cs)
    && waiting_unique(cs)
    && running_has_residency(cs)
    && waiting_has_no_residency(cs)
    && waiting_unstarted(cs)
    && residency_blocks_in_range(cs)
    && residency_has_live_request(cs)
    && block_token_bound(cs)
    && hash_to_block_in_range(cs)
    && residency_block_ids_unique(cs)
    && refcount_valid(cs)
    && hash_to_block_consistent(cs)
    && registered_provenance_aligned(cs)
    && blocks_dom_in_range(cs)
    && block_count_valid(cs)
    && hash_to_block_no_zero(cs)
}

// Queue-link maintenance does not change any field observed by `cs_valid`.
// Keep that representation-only transport behind a small proof boundary so
// queue-reconciliation loops do not repeatedly instantiate the scheduler's
// semantic invariants.
#[verifier::spinoff_prover]
pub proof fn lemma_cs_valid_semantic_frame(
    before: &CacheScheduler,
    after: &CacheScheduler,
)
    requires
        cs_valid(before),
        after.num_blocks == before.num_blocks,
        after.free_blocks == before.free_blocks,
        after.blocks@ == before.blocks@,
        after.running@ == before.running@,
        after.waiting@ == before.waiting@,
        after.request_residency@ == before.request_residency@,
        after.live_requests@ == before.live_requests@,
        after.accepted_requests@ == before.accepted_requests@,
        after.hash_to_block@ == before.hash_to_block@,
    ensures
        cs_valid(after),
{
}

// Sanity proof: the empty-scheduler shape satisfies cs_valid.
pub proof fn lemma_empty_cs_valid(cs: &CacheScheduler)
    requires
        cs.running@.len() == 0,
        cs.waiting@.len() == 0,
        cs.live_requests@.dom().is_empty(),
        cs.request_residency@.dom().is_empty(),
        cs.blocks@.dom().is_empty(),
        cs.hash_to_block@.dom().is_empty(),
        cs.free_blocks == cs.num_blocks,
    ensures cs_valid(cs),
{
    assert(forall|rid: RequestId| !cs.running@.contains(rid));
    assert(forall|rid: RequestId| !cs.waiting@.contains(rid));
    assert(waiting_unstarted(cs));
    assert(forall|rid: RequestId| !cs.request_residency@.contains_key(rid));
    assert(forall|bid: BlockId| !cs.blocks@.contains_key(bid));
    assert(forall|h: u64| !cs.hash_to_block@.contains_key(h));
    // refcount_valid / residency_block_ids_unique / hash_to_block_consistent
    // / blocks_dom_in_range discharge vacuously from the empty domains.
    // block_count_valid follows from the empty block domain.
    vstd::set_lib::lemma_set_is_empty_len0(cs.blocks@.dom());
    vstd::set_lib::lemma_set_is_empty_len0(cs.request_residency@.dom());
    assert(cs.blocks@.dom().len() == 0);
}

} // verus!
