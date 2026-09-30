// Cache-eviction policy boundary.
//
// Correctness is expressed by `zero_ref_provenance_leaf`: only a resident,
// zero-reference provenance leaf may be recycled.  The current O(1) policy
// selects the intrusive cached-queue head.  The scheduler's topological queue
// invariant proves that this policy choice is always correctness-eligible.

use super::types::CacheScheduler;
use crate::{types::{BlockId}};
use vstd::prelude::*;

verus! {

pub open spec fn cached_head_policy_selects(
    cs: &CacheScheduler,
    bid: BlockId,
) -> bool {
    cs.cached_queue.head == Some(bid)
}

pub proof fn lemma_cached_head_policy_selects_eligible(
    cs: &CacheScheduler,
    bid: BlockId,
)
    requires
        super::free_queue_valid(cs),
        cached_head_policy_selects(cs, bid),
    ensures
        super::zero_ref_provenance_leaf(cs, bid),
{
    super::lemma_cached_queue_head_is_leaf(cs);
}

pub fn select_cached_eviction_candidate(
    cs: &CacheScheduler,
) -> (out: Option<BlockId>)
    requires
        super::free_queue_valid(cs),
    ensures
        out == cs.cached_queue.head,
        match out {
            Some(bid) => cached_head_policy_selects(cs, bid)
                && super::zero_ref_provenance_leaf(cs, bid),
            None => forall|bid: BlockId| !super::zero_ref_cached_page(cs, bid),
        },
{
    match cs.cached_queue.head {
        Some(bid) => {
            proof {
                lemma_cached_head_policy_selects_eligible(cs, bid);
            }
            Some(bid)
        },
        None => {
            assert(cs.cached_queue.order@.len() == 0);
            assert forall|bid: BlockId| !super::zero_ref_cached_page(cs, bid) by {
                if super::zero_ref_cached_page(cs, bid) {
                    assert(super::free_queue_membership_valid(cs));
                    assert(cs.cached_queue.order@.contains(bid));
                }
            }
            None
        },
    }
}

} // verus!
