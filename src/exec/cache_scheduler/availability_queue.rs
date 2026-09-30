// Generic intrusive availability queue used by the cache scheduler.
//
// This module owns only representation and queue-manipulation correctness.
// Cache membership, provenance-leaf eligibility, and replacement policy remain
// scheduler-level concerns.

use crate::proof::tensor::geometry::*;
use crate::{types::{BlockId}};
use vstd::hash_map::HashMapWithView;
use vstd::prelude::*;
#[cfg(verus_only)]
use vstd::std_specs::hash::obeys_key_model;

verus! {
// Runtime availability-queue link. Each of the vacant and cached queues is
// intrusive by physical block ID and owns links only for its own members.
// Keeping links outside `BlockEntry` lets vacant IDs participate and keeps
// allocator policy out of semantic block equality.
pub struct FreeBlockLink {
    pub prev: Option<BlockId>,
    pub next: Option<BlockId>,
}

impl Clone for FreeBlockLink {
    fn clone(&self) -> (out: Self)
        ensures
            out.prev == self.prev,
            out.next == self.next,
    {
        FreeBlockLink {
            prev: self.prev,
            next: self.next,
        }
    }
}

// vLLM-style intrusive availability queue. `order` is ghost-only: it is the exact
// mathematical sequence represented by the executable head/tail/link fields.
// CacheScheduler later strengthens this representation invariant with queue
// membership and, for the cached instance, physical-prefix topology facts.
pub struct FreeBlockQueue {
    pub head: Option<BlockId>,
    pub tail: Option<BlockId>,
    pub len: u64,
    pub links: HashMapWithView<BlockId, FreeBlockLink>,
    pub order: Ghost<Seq<BlockId>>,
}

pub open spec fn free_queue_shape(q: &FreeBlockQueue) -> bool {
    let s = q.order@;
    &&& q.len as int == s.len()
    &&& s.no_duplicates()
    &&& q.links@.dom() == s.to_set()
    &&& (s.len() == 0 ==> q.head is None && q.tail is None)
    &&& (s.len() > 0 ==> q.head == Some(s[0]) && q.tail == Some(s[s.len() - 1]))
    &&& forall|i: int| 0 <= i < s.len() ==> {
        let bid = #[trigger] s[i];
        &&& q.links@.contains_key(bid)
        &&& q.links@[bid].prev == if i == 0 { None } else { Some(s[i - 1]) }
        &&& q.links@[bid].next == if i + 1 == s.len() { None } else { Some(s[i + 1]) }
    }
}

// Opaque loop token for the intrusive representation.  Carrying the expanded
// shape through a scheduler loop exposes `Seq::no_duplicates` and every link
// equation to each iteration query; this token keeps those facts sealed until
// a queue primitive needs them.
#[verifier::opaque]
pub open spec fn free_queue_shape_token(q: &FreeBlockQueue) -> bool {
    free_queue_shape(q)
}

pub proof fn lemma_free_queue_shape_to_token(q: &FreeBlockQueue)
    requires free_queue_shape(q),
    ensures free_queue_shape_token(q),
{
    reveal(free_queue_shape_token);
}

pub proof fn lemma_free_queue_token_to_shape(q: &FreeBlockQueue)
    requires free_queue_shape_token(q),
    ensures free_queue_shape(q),
{
    reveal(free_queue_shape_token);
}

pub proof fn lemma_free_queue_shape_token_frame(
    before: &FreeBlockQueue,
    after: &FreeBlockQueue,
)
    requires
        free_queue_shape_token(before),
        after.head == before.head,
        after.tail == before.tail,
        after.len == before.len,
        after.links@ == before.links@,
        after.order@ == before.order@,
    ensures
        free_queue_shape_token(after),
{
    reveal(free_queue_shape_token);
}

pub proof fn lemma_free_queue_token_has_room_for_missing(
    q: &FreeBlockQueue,
    num_blocks: u64,
    bid: BlockId,
)
    requires
        free_queue_shape_token(q),
        q.len <= num_blocks,
        bid < num_blocks,
        !q.order@.contains(bid),
        forall|b: BlockId| #[trigger] q.order@.contains(b)
            ==> b < num_blocks,
    ensures
        q.len < num_blocks,
        q.len < u64::MAX,
{
    reveal(free_queue_shape_token);
    let ids = q.order@.to_set();
    let range = Set::<BlockId>::range(0, num_blocks);
    q.order@.unique_seq_to_set();
    vstd::set_lib::range_set_properties(0u64, num_blocks);
    assert(ids.insert(bid).subset_of(range)) by {
        assert forall|b: BlockId| ids.insert(bid).contains(b)
            implies range.contains(b) by {
            if b == bid {
            } else {
                assert(ids.contains(b));
                assert(q.order@.contains(b));
                assert(b < num_blocks);
            }
        }
    }
    assert(!ids.contains(bid));
    vstd::set::lemma_set_insert_len(ids, bid);
    vstd::set_lib::lemma_len_subset(ids.insert(bid), range);
    assert(ids.insert(bid).len() == ids.len() + 1);
    assert(range.len() == num_blocks as nat);
    assert(q.len as int + 1 <= num_blocks as int);
}

impl FreeBlockQueue {
    pub fn empty() -> (out: FreeBlockQueue)
        requires
            obeys_key_model::<u64>(),
        ensures
            free_queue_shape(&out),
            out.order@.len() == 0,
    {
        let out = FreeBlockQueue {
            head: None,
            tail: None,
            len: 0,
            links: HashMapWithView::<BlockId, FreeBlockLink>::new(),
            order: Ghost(Seq::<BlockId>::empty()),
        };
        assert(out.links@.dom().is_empty());
        assert(out.order@.to_set().is_empty());
        assert(out.links@.dom() == out.order@.to_set());
        assert(free_queue_shape(&out));
        out
    }

    pub fn append(&mut self, bid: BlockId)
        requires
            free_queue_shape(old(self)),
            !old(self).order@.contains(bid),
            old(self).len < u64::MAX,
        ensures
            free_queue_shape(final(self)),
            final(self).order@ == old(self).order@.push(bid),
    {
        let ghost old_order = old(self).order@;
        let ghost old_links = old(self).links@;
        let old_tail = self.tail;
        match old_tail {
            Some(tail_bid) => {
                assert(old_order.len() > 0);
                assert(tail_bid == old_order[old_order.len() - 1]);
                let tail_link = match self.links.get(&tail_bid) {
                    Some(link_ref) => link_ref.clone(),
                    None => {
                        proof { assert(false); }
                        FreeBlockLink { prev: None, next: None }
                    },
                };
                assert(tail_link.next is None);
                self.links.insert(tail_bid, FreeBlockLink {
                    prev: tail_link.prev,
                    next: Some(bid),
                });
            },
            None => {
                assert(old_order.len() == 0);
                self.head = Some(bid);
            },
        }
        let ghost links_before_new = self.links@;
        self.links.insert(bid, FreeBlockLink {
            prev: old_tail,
            next: None,
        });
        proof {
            vstd::map::lemma_map_insert_domain(links_before_new, bid, self.links@[bid]);
        }
        self.tail = Some(bid);
        self.len = self.len + 1;
        self.order = Ghost(old_order.push(bid));

        assert(self.links@.dom() == old_links.dom().insert(bid)) by {
            assert(!old_links.contains_key(bid));
        }
        proof { old_order.lemma_push_to_set_commute(bid); }
        assert(self.order@.to_set() == old_order.to_set().insert(bid));
        assert(self.order@.no_duplicates());
        assert forall|i: int| 0 <= i < self.order@.len() implies {
            let at = #[trigger] self.order@[i];
            &&& self.links@.contains_key(at)
            &&& self.links@[at].prev
                == if i == 0 { None } else { Some(self.order@[i - 1]) }
            &&& self.links@[at].next
                == if i + 1 == self.order@.len() { None } else { Some(self.order@[i + 1]) }
        } by {
            if i == old_order.len() {
                assert(self.order@[i] == bid);
            } else {
                assert(0 <= i < old_order.len());
                assert(self.order@[i] == old_order[i]);
                if i + 1 == old_order.len() {
                    assert(old_order[i] == old_order[old_order.len() - 1]);
                    assert(old_tail == Some(old_order[i]));
                } else {
                    assert(i + 1 < old_order.len());
                    assert(self.order@[i + 1] == old_order[i + 1]);
                }
                if i > 0 {
                    assert(self.order@[i - 1] == old_order[i - 1]);
                }
            }
        }
        assert(free_queue_shape(self));
    }

    pub fn prepend(&mut self, bid: BlockId)
        requires
            free_queue_shape(old(self)),
            !old(self).order@.contains(bid),
            old(self).len < u64::MAX,
        ensures
            free_queue_shape(final(self)),
            final(self).order@ == Seq::<BlockId>::empty().push(bid) + old(self).order@,
    {
        let ghost old_order = old(self).order@;
        let ghost old_links = old(self).links@;
        let old_head = self.head;
        match old_head {
            Some(head_bid) => {
                assert(old_order.len() > 0);
                assert(head_bid == old_order[0]);
                let head_link = match self.links.get(&head_bid) {
                    Some(link_ref) => link_ref.clone(),
                    None => {
                        proof { assert(false); }
                        FreeBlockLink { prev: None, next: None }
                    },
                };
                assert(head_link.prev is None);
                self.links.insert(head_bid, FreeBlockLink {
                    prev: Some(bid),
                    next: head_link.next,
                });
            },
            None => {
                assert(old_order.len() == 0);
                self.tail = Some(bid);
            },
        }
        let ghost links_before_new = self.links@;
        self.links.insert(bid, FreeBlockLink {
            prev: None,
            next: old_head,
        });
        proof {
            vstd::map::lemma_map_insert_domain(links_before_new, bid, self.links@[bid]);
        }
        self.head = Some(bid);
        self.len = self.len + 1;
        let ghost singleton = Seq::<BlockId>::empty().push(bid);
        self.order = Ghost(singleton + old_order);

        assert(self.links@.dom() == old_links.dom().insert(bid)) by {
            assert(!old_links.contains_key(bid));
        }
        proof {
            Seq::<BlockId>::empty().lemma_push_to_set_commute(bid);
            vstd::seq_lib::seq_to_set_distributes_over_add(singleton, old_order);
            assert forall|i: int, j: int|
                0 <= i < singleton.len() && 0 <= j < old_order.len()
                implies singleton[i] != old_order[j]
            by {
                assert(singleton[i] == bid);
                if singleton[i] == old_order[j] {
                    assert(old_order.contains(bid));
                }
            }
            vstd::seq_lib::lemma_no_dup_in_concat(singleton, old_order);
        }
        assert(self.order@.to_set() == old_order.to_set().insert(bid));
        assert(self.order@.no_duplicates());
        assert forall|i: int| 0 <= i < self.order@.len() implies {
            let at = #[trigger] self.order@[i];
            &&& self.links@.contains_key(at)
            &&& self.links@[at].prev
                == if i == 0 { None } else { Some(self.order@[i - 1]) }
            &&& self.links@[at].next
                == if i + 1 == self.order@.len() { None } else { Some(self.order@[i + 1]) }
        } by {
            if i == 0 {
                assert(self.order@[i] == bid);
                if old_order.len() > 0 {
                    assert(self.order@[1] == old_order[0]);
                }
            } else {
                let j = i - 1;
                assert(0 <= j < old_order.len());
                assert(self.order@[i] == old_order[j]);
                if j == 0 {
                    assert(old_head == Some(old_order[j]));
                    assert(self.order@[i - 1] == bid);
                } else {
                    assert(self.order@[i - 1] == old_order[j - 1]);
                }
                if j + 1 == old_order.len() {
                    assert(i + 1 == self.order@.len());
                } else {
                    assert(self.order@[i + 1] == old_order[j + 1]);
                }
            }
        }
        assert(free_queue_shape(self));
    }

    // Loop-facing wrappers keep the expanded link/no-duplicate invariant out
    // of the caller's SMT context.
    pub fn append_tokenized(&mut self, bid: BlockId)
        requires
            free_queue_shape_token(old(self)),
            !old(self).order@.contains(bid),
            old(self).len < u64::MAX,
        ensures
            free_queue_shape_token(final(self)),
            final(self).order@ == old(self).order@.push(bid),
            final(self).len as int == old(self).len as int + 1,
    {
        proof { lemma_free_queue_token_to_shape(old(self)); }
        self.append(bid);
        proof { lemma_free_queue_shape_to_token(self); }
    }

    pub fn prepend_tokenized(&mut self, bid: BlockId)
        requires
            free_queue_shape_token(old(self)),
            !old(self).order@.contains(bid),
            old(self).len < u64::MAX,
        ensures
            free_queue_shape_token(final(self)),
            final(self).order@ == Seq::<BlockId>::empty().push(bid) + old(self).order@,
            final(self).len as int == old(self).len as int + 1,
    {
        proof { lemma_free_queue_token_to_shape(old(self)); }
        self.prepend(bid);
        proof { lemma_free_queue_shape_to_token(self); }
    }

    pub fn pop_front(&mut self) -> (out: Option<BlockId>)
        requires
            free_queue_shape(old(self)),
        ensures
            free_queue_shape(final(self)),
            match out {
                None => old(self).order@.len() == 0
                    && final(self).order@ == old(self).order@,
                Some(bid) => old(self).order@.len() > 0
                    && bid == old(self).order@[0]
                    && final(self).order@ == old(self).order@.subrange(
                        1, old(self).order@.len() as int,
                    ),
            },
    {
        let ghost old_order = old(self).order@;
        let ghost old_links = old(self).links@;
        let head_bid = match self.head {
            Some(bid) => bid,
            None => {
                assert(old_order.len() == 0);
                return None;
            },
        };
        assert(old_order.len() > 0);
        assert(head_bid == old_order[0]);
        let head_link = match self.links.get(&head_bid) {
            Some(link_ref) => link_ref.clone(),
            None => {
                proof { assert(false); }
                return None;
            },
        };
        assert(head_link.prev is None);
        let next_bid = head_link.next;
        match next_bid {
            Some(next) => {
                assert(old_order.len() > 1);
                assert(next == old_order[1]);
                let next_link = match self.links.get(&next) {
                    Some(link_ref) => link_ref.clone(),
                    None => {
                        proof { assert(false); }
                        return None;
                    },
                };
                assert(next_link.prev == Some(head_bid));
                self.links.insert(next, FreeBlockLink {
                    prev: None,
                    next: next_link.next,
                });
                self.head = Some(next);
            },
            None => {
                assert(old_order.len() == 1) by {
                    if old_order.len() > 1 {
                        assert(old(self).links@[head_bid].next == Some(old_order[1]));
                    }
                }
                self.head = None;
                self.tail = None;
            },
        }
        let ghost links_before_remove = self.links@;
        self.links.remove(&head_bid);
        proof {
            vstd::map::lemma_map_remove_domain(links_before_remove, head_bid);
        }
        self.len = self.len - 1;
        self.order = Ghost(old_order.subrange(1, old_order.len() as int));

        assert(self.links@.dom() == old_links.dom().remove(head_bid)) by {
            assert(links_before_remove.dom() == old_links.dom());
        }
        assert(self.order@.to_set() =~= old_order.to_set().remove(head_bid)) by {
            assert forall|x: BlockId| self.order@.to_set().contains(x)
                <==> old_order.to_set().remove(head_bid).contains(x) by {
                if self.order@.to_set().contains(x) {
                    let i = self.order@.index_of(x);
                    assert(0 <= i < self.order@.len());
                    assert(self.order@[i] == x);
                    assert(old_order[i + 1] == x);
                    assert(old_order.to_set().contains(x));
                    assert(x != head_bid) by {
                        if x == head_bid {
                            assert(old_order[0] == old_order[i + 1]);
                            assert(old_order.no_duplicates());
                        }
                    }
                }
                if old_order.to_set().remove(head_bid).contains(x) {
                    assert(old_order.to_set().contains(x));
                    assert(x != head_bid);
                    let j = old_order.index_of(x);
                    assert(0 <= j < old_order.len());
                    assert(old_order[j] == x);
                    assert(j != 0) by {
                        if j == 0 {
                            assert(x == head_bid);
                        }
                    }
                    assert(1 <= j);
                    assert(self.order@[j - 1] == old_order[j]);
                    assert(self.order@.to_set().contains(x));
                }
            }
        }
        assert(self.order@.no_duplicates());
        assert forall|i: int| 0 <= i < self.order@.len() implies {
            let at = #[trigger] self.order@[i];
            &&& self.links@.contains_key(at)
            &&& self.links@[at].prev
                == if i == 0 { None } else { Some(self.order@[i - 1]) }
            &&& self.links@[at].next
                == if i + 1 == self.order@.len() { None } else { Some(self.order@[i + 1]) }
        } by {
            let j = i + 1;
            assert(self.order@[i] == old_order[j]);
            assert(old_order[j] != head_bid);
            if i == 0 {
                assert(next_bid == Some(old_order[j]));
            } else {
                assert(self.order@[i - 1] == old_order[j - 1]);
                assert(old_order[j] != old_order[1]);
            }
            if i + 1 == self.order@.len() {
                assert(j + 1 == old_order.len());
            } else {
                assert(j + 1 < old_order.len());
                assert(self.order@[i + 1] == old_order[j + 1]);
            }
        }
        assert(free_queue_shape(self));
        Some(head_bid)
    }

    #[verifier::spinoff_prover]
    #[verifier::rlimit(200)]
    pub fn remove(&mut self, bid: BlockId) -> (removed: bool)
        requires
            free_queue_shape(old(self)),
        ensures
            free_queue_shape(final(self)),
            removed <==> old(self).order@.contains(bid),
            removed ==> final(self).order@ == old(self).order@.subrange(
                    0, old(self).order@.index_of(bid),
                ) + old(self).order@.subrange(
                    old(self).order@.index_of(bid) + 1,
                    old(self).order@.len() as int,
                ),
            !removed ==> final(self).order@ == old(self).order@,
    {
        let ghost old_order = old(self).order@;
        let ghost old_links = old(self).links@;
        if !self.links.contains_key(&bid) {
            assert(!old_order.to_set().contains(bid));
            assert(!old_order.contains(bid));
            return false;
        }
        assert(old_order.contains(bid));
        let ghost idx = old_order.index_of(bid);
        assert(0 <= idx < old_order.len());
        assert(old_order[idx] == bid);

        if self.head == Some(bid) {
            assert(idx == 0) by {
                assert(self.head == Some(old_order[0]));
                assert(old_order.no_duplicates());
            }
            let popped = self.pop_front();
            assert(popped == Some(bid));
            assert(old_order.subrange(0, idx) =~= Seq::<BlockId>::empty());
            assert(Seq::<BlockId>::empty() + old_order.subrange(
                idx + 1, old_order.len() as int,
            ) =~= old_order.subrange(1, old_order.len() as int));
            return true;
        }

        assert(idx > 0);
        let link = match self.links.get(&bid) {
            Some(link_ref) => link_ref.clone(),
            None => {
                proof { assert(false); }
                return false;
            },
        };
        let prev_bid = match link.prev {
            Some(prev) => prev,
            None => {
                proof { assert(false); }
                return false;
            },
        };
        assert(prev_bid == old_order[idx - 1]);
        let prev_link = match self.links.get(&prev_bid) {
            Some(link_ref) => link_ref.clone(),
            None => {
                proof { assert(false); }
                return false;
            },
        };
        assert(prev_link.next == Some(bid));
        self.links.insert(prev_bid, FreeBlockLink {
            prev: prev_link.prev,
            next: link.next,
        });

        match link.next {
            Some(next_bid) => {
                assert(idx + 1 < old_order.len());
                assert(next_bid == old_order[idx + 1]);
                assert(next_bid != prev_bid) by {
                    assert(old_order.no_duplicates());
                }
                let next_link = match self.links.get(&next_bid) {
                    Some(link_ref) => link_ref.clone(),
                    None => {
                        proof { assert(false); }
                        return false;
                    },
                };
                assert(next_link.prev == Some(bid));
                self.links.insert(next_bid, FreeBlockLink {
                    prev: Some(prev_bid),
                    next: next_link.next,
                });
            },
            None => {
                assert(idx + 1 == old_order.len());
                self.tail = Some(prev_bid);
            },
        }

        let ghost links_before_remove = self.links@;
        self.links.remove(&bid);
        proof {
            vstd::map::lemma_map_remove_domain(links_before_remove, bid);
        }
        self.len = self.len - 1;
        let ghost left = old_order.subrange(0, idx);
        let ghost right = old_order.subrange(idx + 1, old_order.len() as int);
        self.order = Ghost(left + right);

        assert(self.links@.dom() == old_links.dom().remove(bid)) by {
            assert(links_before_remove.dom() == old_links.dom());
        }
        assert(self.order@.to_set() =~= old_order.to_set().remove(bid)) by {
            assert forall|x: BlockId| self.order@.to_set().contains(x)
                <==> old_order.to_set().remove(bid).contains(x) by {
                if self.order@.to_set().contains(x) {
                    let i = self.order@.index_of(x);
                    assert(0 <= i < self.order@.len());
                    if i < left.len() {
                        assert(self.order@[i] == left[i]);
                        assert(left[i] == old_order[i]);
                        assert(old_order.to_set().contains(x));
                        assert(x != bid) by {
                            if x == bid {
                                assert(old_order[i] == old_order[idx]);
                                assert(old_order.no_duplicates());
                            }
                        }
                    } else {
                        let j = i - left.len();
                        assert(0 <= j < right.len());
                        assert(self.order@[i] == right[j]);
                        assert(right[j] == old_order[idx + 1 + j]);
                        assert(old_order.to_set().contains(x));
                        assert(x != bid) by {
                            if x == bid {
                                assert(old_order[idx + 1 + j] == old_order[idx]);
                                assert(old_order.no_duplicates());
                            }
                        }
                    }
                }
                if old_order.to_set().remove(bid).contains(x) {
                    let j = old_order.index_of(x);
                    assert(0 <= j < old_order.len());
                    assert(old_order[j] == x);
                    assert(x != bid);
                    assert(j != idx) by {
                        if j == idx {
                            assert(x == bid);
                        }
                    }
                    if j < idx {
                        assert(left[j] == x);
                        assert(self.order@[j] == x);
                    } else {
                        assert(idx < j);
                        let i = j - 1;
                        assert(right[j - idx - 1] == x);
                        assert(self.order@[i] == x);
                    }
                    assert(self.order@.to_set().contains(x));
                }
            }
        }
        assert(left.no_duplicates());
        assert(right.no_duplicates());
        assert forall|i: int, j: int|
            0 <= i < left.len() && 0 <= j < right.len()
            implies left[i] != right[j]
        by {
            assert(left[i] == old_order[i]);
            assert(right[j] == old_order[idx + 1 + j]);
            assert(i < idx);
            assert(idx < idx + 1 + j);
            assert(old_order.no_duplicates());
        }
        proof {
            vstd::seq_lib::lemma_no_dup_in_concat(left, right);
        }
        assert(self.order@.no_duplicates());

        assert forall|i: int| 0 <= i < self.order@.len() implies {
            let at = #[trigger] self.order@[i];
            &&& self.links@.contains_key(at)
            &&& self.links@[at].prev
                == if i == 0 { None } else { Some(self.order@[i - 1]) }
            &&& self.links@[at].next
                == if i + 1 == self.order@.len() { None } else { Some(self.order@[i + 1]) }
        } by {
            if i < idx {
                assert(self.order@[i] == old_order[i]);
                assert(old_order[i] != bid);
                if i + 1 == idx {
                    assert(old_order[i] == prev_bid);
                    match link.next {
                        Some(next_bid) => {
                            assert(self.order@[i + 1] == old_order[idx + 1]);
                            assert(next_bid == self.order@[i + 1]);
                        },
                        None => {
                            assert(i + 1 == self.order@.len());
                        },
                    }
                } else {
                    assert(i + 1 < idx || idx < i + 1);
                    assert(i + 1 < idx);
                    assert(self.order@[i + 1] == old_order[i + 1]);
                }
                if i > 0 {
                    assert(self.order@[i - 1] == old_order[i - 1]);
                }
            } else {
                let j = i + 1;
                assert(idx < j < old_order.len());
                assert(self.order@[i] == old_order[j]);
                assert(old_order[j] != bid);
                if i == idx {
                    assert(old_order[j] == link.next.unwrap());
                    assert(self.order@[i - 1] == old_order[idx - 1]);
                    assert(self.order@[i - 1] == prev_bid);
                } else {
                    assert(self.order@[i - 1] == old_order[j - 1]);
                }
                if i + 1 == self.order@.len() {
                    assert(j + 1 == old_order.len());
                } else {
                    assert(self.order@[i + 1] == old_order[j + 1]);
                }
            }
        }
        assert(free_queue_shape(self));
        true
    }

    pub fn init_all(num_blocks: u64) -> (out: FreeBlockQueue)
        requires
            obeys_key_model::<u64>(),
        ensures
            free_queue_shape(&out),
            out.order@.len() == num_blocks as int,
            forall|i: int| 0 <= i < num_blocks as int
                ==> #[trigger] out.order@[i] == i as BlockId,
    {
        let mut out = Self::empty();
        let mut bid: BlockId = 0;
        while bid < num_blocks
            invariant
                bid <= num_blocks,
                free_queue_shape(&out),
                out.order@.len() == bid as int,
                forall|i: int| 0 <= i < bid as int
                    ==> #[trigger] out.order@[i] == i as BlockId,
            decreases num_blocks - bid
        {
            assert(!out.order@.contains(bid)) by {
                if out.order@.contains(bid) {
                    let i = out.order@.index_of(bid);
                    assert(0 <= i < out.order@.len());
                    assert(out.order@[i] == bid);
                    assert(out.order@[i] == i as BlockId);
                    assert(i < bid as int);
                    assert(false);
                }
            }
            out.append(bid);
            bid = bid + 1;
        }
        out
    }
}


} // verus!
