// Verified cached-leaf eviction and request deallocation transitions.

use super::eviction_policy;
use super::*;

verus! {
impl CacheScheduler {
    // Reclaim the first cached page in the intrusive queue.  The topological
    // queue invariant makes this page a provenance leaf; the particular leaf
    // chosen among eligible leaves remains replacement policy.
    #[verifier::spinoff_prover]
    #[verifier::rlimit(300)]
    pub fn evict_cached_leaf(&mut self, bid: BlockId) -> (evicted: bool)
        requires
            cs_valid(old(self)),
            free_queue_valid(old(self)),
        ensures
            cs_valid(final(self)),
            free_queue_valid(final(self)),
            final(self).config == old(self).config,
            final(self).num_blocks == old(self).num_blocks,
            final(self).running@ == old(self).running@,
            final(self).waiting@ == old(self).waiting@,
            final(self).live_requests@ == old(self).live_requests@,
            final(self).accepted_requests@ == old(self).accepted_requests@,
            final(self).request_residency@ == old(self).request_residency@,
            evicted <==> eviction_policy::cached_head_policy_selects(old(self), bid),
            evicted ==> !final(self).blocks@.contains_key(bid),
            evicted ==> final(self).blocks@.dom() == old(self).blocks@.dom().remove(bid),
            evicted ==> final(self).free_blocks as int == old(self).free_blocks as int + 1,
            evicted ==> cached_leaf_eviction_frame(old(self), final(self), bid),
            forall|other: BlockId|
                other != bid && #[trigger] old(self).blocks@.contains_key(other)
                ==> final(self).blocks@.contains_key(other)
                    && final(self).blocks@[other] == old(self).blocks@[other],
            evicted ==> (forall|h: u64|
                #[trigger] final(self).hash_to_block@.contains_key(h)
                ==> old(self).hash_to_block@.contains_key(h)
                    && final(self).hash_to_block@[h] == old(self).hash_to_block@[h]
                    && old(self).hash_to_block@[h] != bid),
            forall|h: u64| #[trigger] old(self).hash_to_block@.contains_key(h)
                && old(self).hash_to_block@[h] != bid
                ==> final(self).hash_to_block@.contains_key(h)
                    && final(self).hash_to_block@[h] == old(self).hash_to_block@[h],
            !evicted ==> final(self).blocks@ == old(self).blocks@
                && final(self).hash_to_block@ == old(self).hash_to_block@
                && final(self).free_blocks == old(self).free_blocks
                && final(self).free_queue.order@ == old(self).free_queue.order@
                && final(self).cached_queue.head == old(self).cached_queue.head,
            persistent_provenance_closed(old(self))
                ==> persistent_provenance_closed(final(self)),
    {
        if self.cached_queue.head != Some(bid) {
            return false;
        }
        let entry = match self.blocks.get(&bid) {
            Some(entry_ref) => entry_ref.clone(),
            None => {
                proof { assert(false); }
                return false;
            },
        };
        assert(self.blocks@.contains_key(bid));
        assert(entry.tokens@ == self.blocks@[bid].tokens@);
        assert(entry.refcount == self.blocks@[bid].refcount);
        assert(entry.hash_value == self.blocks@[bid].hash_value);
        assert(entry.prefix_depth == self.blocks@[bid].prefix_depth);
        assert(entry.parent_block == self.blocks@[bid].parent_block);
        assert(entry.refcount == 0);
        assert(entry.prefix_depth > 0);
        assert(zero_ref_cached_page(old(self), bid));
        proof {
            lemma_cached_queue_head_is_leaf(old(self));
        }
        assert(zero_ref_provenance_leaf(old(self), bid));
        assert(bid < old(self).num_blocks) by {
            assert(blocks_dom_in_range(old(self)));
        }

        let ghost old_blocks = old(self).blocks@;
        let ghost old_registry = old(self).hash_to_block@;
        let ghost holders = residency_holders_of(old(self), bid);
        proof {
            assert(refcount_valid(old(self)));
            assert(holders.len() == 0);
            vstd::set_lib::lemma_set_is_empty_len0(holders);
            assert(holders.is_empty());
        }
        assert forall|rid: RequestId| !holders.contains(rid) by {}

        let registry_targets_bid = match self.hash_to_block.get(&entry.hash_value) {
            Some(mapped) => *mapped == bid,
            None => false,
        };
        assert(registry_targets_bid ==
            (old_registry.contains_key(entry.hash_value)
                && old_registry[entry.hash_value] == bid));

        let popped = self.cached_queue.pop_front();
        assert(popped == Some(bid));

        self.blocks.remove(&bid);
        proof {
            vstd::map::lemma_map_remove_domain(old_blocks, bid);
            vstd::set::lemma_set_remove_len(old_blocks.dom(), bid);
        }
        assert(self.blocks@.dom() == old_blocks.dom().remove(bid));
        assert(old_blocks.dom().len() == self.blocks@.dom().len() + 1);
        assert(old(self).free_blocks < u64::MAX) by {
            assert(block_count_valid(old(self)));
            assert(old_blocks.dom().len() >= 1);
        }
        self.free_blocks = self.free_blocks.saturating_add(1);
        assert(self.free_blocks as int == old(self).free_blocks as int + 1);
        if registry_targets_bid {
            self.hash_to_block.remove(&entry.hash_value);
        }
        let ghost vacant_before_prepend = self.free_queue.order@;
        self.free_queue.prepend(bid);
        proof {
            let singleton = Seq::<BlockId>::empty().push(bid);
            assert(self.free_queue.order@ == singleton + vacant_before_prepend);
        }

        assert forall|h: u64| #[trigger] self.hash_to_block@.contains_key(h)
            implies old_registry.contains_key(h)
                && self.hash_to_block@[h] == old_registry[h]
                && old_registry[h] != bid
        by {
            if registry_targets_bid {
                assert(h != entry.hash_value);
            } else if old_registry[h] == bid {
                assert(hash_to_block_consistent(old(self)));
                assert(old_blocks[old_registry[h]].hash_value == h);
                assert(old_blocks[bid].hash_value == entry.hash_value);
                assert(h == entry.hash_value);
                assert(registry_targets_bid);
            }
        }
        assert forall|h: u64| #[trigger] old_registry.contains_key(h)
            && old_registry[h] != bid
            implies self.hash_to_block@.contains_key(h)
                && self.hash_to_block@[h] == old_registry[h]
        by {
            if registry_targets_bid {
                assert(h != entry.hash_value);
            }
        }
        assert forall|other: BlockId|
            other != bid && #[trigger] old_blocks.contains_key(other)
            implies self.blocks@.contains_key(other)
                && self.blocks@[other] == old_blocks[other]
        by {}

        assert(live_covers_queue(self));
        assert(self.live_requests@ == old(self).live_requests@);
        assert(self.accepted_requests@ == old(self).accepted_requests@);
        assert(live_requests_accepted(self)) by {
            assert forall|rid: RequestId|
                #[trigger] self.live_requests@.contains_key(rid)
                implies self.accepted_requests@.contains_key(rid)
                    && self.accepted_requests@[rid]
            by {
                assert(old(self).live_requests@.contains_key(rid));
                assert(live_requests_accepted(old(self)));
            }
        }
        assert(queue_disjoint(self));
        assert(running_unique(self));
        assert(waiting_unique(self));
        assert(running_has_residency(self));
        assert(waiting_has_no_residency(self));
        assert(waiting_unstarted(self));
        assert(residency_has_live_request(self));
        assert(residency_block_ids_unique(self));
        assert(residency_blocks_in_range(self)) by {
            assert forall|rid: RequestId, j: int|
                #![trigger self.request_residency@[rid].block_ids@[j]]
                self.request_residency@.contains_key(rid)
                && 0 <= j < self.request_residency@[rid].block_ids@.len()
                implies self.blocks@.contains_key(
                    self.request_residency@[rid].block_ids@[j])
            by {
                let held = self.request_residency@[rid].block_ids@[j];
                assert(old(self).blocks@.contains_key(held)) by {
                    assert(residency_blocks_in_range(old(self)));
                }
                if held == bid {
                    assert(holders.contains(rid));
                    assert(false);
                }
            }
        }
        assert(block_token_bound(self)) by {
            assert forall|b: BlockId| #[trigger] self.blocks@.contains_key(b)
                implies self.blocks@[b].tokens@.len() <= BLOCK_SIZE_SPEC as int
            by {
                assert(b != bid);
                assert(self.blocks@[b] == old_blocks[b]);
                assert(block_token_bound(old(self)));
            }
        }
        assert(blocks_dom_in_range(self)) by {
            assert forall|b: BlockId| #[trigger] self.blocks@.contains_key(b)
                implies b < self.num_blocks
            by {
                assert(old_blocks.contains_key(b));
                assert(blocks_dom_in_range(old(self)));
            }
        }
        assert(block_count_valid(self)) by {
            assert(block_count_valid(old(self)));
        }
        assert(hash_to_block_in_range(self)) by {
            assert forall|h: u64| #[trigger] self.hash_to_block@.contains_key(h)
                implies self.blocks@.contains_key(self.hash_to_block@[h])
                    && self.blocks@[self.hash_to_block@[h]].tokens@.len()
                        == BLOCK_SIZE_SPEC as int
            by {
                assert(old_registry.contains_key(h));
                assert(self.hash_to_block@[h] == old_registry[h]);
                assert(old_registry[h] != bid);
                assert(hash_to_block_in_range(old(self)));
                assert(self.blocks@[old_registry[h]] == old_blocks[old_registry[h]]);
            }
        }
        assert(hash_to_block_consistent(self)) by {
            assert forall|h: u64| #[trigger] self.hash_to_block@.contains_key(h)
                implies self.blocks@[self.hash_to_block@[h]].hash_value == h
                    && self.blocks@[self.hash_to_block@[h]].prefix_depth > 0
            by {
                assert(old_registry.contains_key(h));
                assert(self.hash_to_block@[h] == old_registry[h]);
                assert(old_registry[h] != bid);
                assert(hash_to_block_consistent(old(self)));
                assert(self.blocks@[old_registry[h]] == old_blocks[old_registry[h]]);
            }
        }
        assert(hash_to_block_no_zero(self)) by {
            if self.hash_to_block@.contains_key(0u64) {
                assert(old_registry.contains_key(0u64));
                assert(hash_to_block_no_zero(old(self)));
            }
        }
        assert(refcount_valid(self)) by {
            assert forall|b: BlockId| #[trigger] self.blocks@.contains_key(b)
                implies self.blocks@[b].refcount as int
                    == residency_holders_of(self, b).len() as int
            by {
                assert(b != bid);
                assert(self.blocks@[b] == old_blocks[b]);
                assert(residency_holders_of(self, b)
                    == residency_holders_of(old(self), b));
                assert(refcount_valid(old(self)));
            }
        }
        assert(registered_provenance_aligned(self)) by {
            assert forall|rid: RequestId, j: int|
                #![trigger self.blocks@[self.request_residency@[rid]
                    .block_ids@[j]].prefix_depth]
                self.request_residency@.contains_key(rid)
                && 0 <= j < self.request_residency@[rid].block_ids@.len()
                && self.blocks@.contains_key(self.request_residency@[rid].block_ids@[j])
                && self.blocks@[self.request_residency@[rid]
                    .block_ids@[j]].prefix_depth > 0
                implies {
                    let ids = self.request_residency@[rid].block_ids@;
                    let held = ids[j];
                    &&& self.blocks@[held].prefix_depth as int == j + 1
                    &&& self.blocks@[held].parent_block
                        == if j == 0 { None } else { Some(ids[j - 1]) }
                }
            by {
                let held = self.request_residency@[rid].block_ids@[j];
                assert(held != bid);
                assert(self.blocks@[held] == old_blocks[held]);
                assert(registered_provenance_aligned(old(self)));
                if j > 0 {
                    let parent = self.request_residency@[rid].block_ids@[j - 1];
                    assert(parent != bid) by {
                        if parent == bid {
                            assert(old(self).request_residency@[rid].block_ids@
                                .contains(bid));
                            assert(holders.contains(rid));
                            assert(false);
                        }
                    }
                    assert(self.blocks@[parent] == old_blocks[parent]);
                }
            }
        }
        proof {
            if persistent_provenance_closed(old(self)) {
                assert forall|child: BlockId|
                    #[trigger] self.blocks@.contains_key(child)
                    && self.blocks@[child].prefix_depth > 1
                    implies self.blocks@[child].parent_block is Some
                        && self.blocks@.contains_key(
                            self.blocks@[child].parent_block.unwrap(),
                        )
                        && self.blocks@[
                            self.blocks@[child].parent_block.unwrap()
                        ].prefix_depth + 1
                            == self.blocks@[child].prefix_depth
                by {
                    assert(child != bid);
                    assert(self.blocks@[child] == old_blocks[child]);
                    reveal(persistent_provenance_closed);
                    assert(old_blocks[child].parent_block is Some);
                    assert(self.blocks@[child].parent_block is Some);
                    let parent = self.blocks@[child].parent_block.unwrap();
                    assert(old_blocks.contains_key(parent));
                    assert(old_blocks[parent].prefix_depth + 1
                        == old_blocks[child].prefix_depth);
                    assert(parent != bid) by {
                        if parent == bid {
                            assert(zero_ref_provenance_leaf(old(self), bid));
                            assert(old_blocks[child].parent_block == Some(bid));
                        }
                    }
                    assert(self.blocks@[parent] == old_blocks[parent]);
                }
                lemma_persistent_provenance_closed_frame(old(self), self);
            }
        }
        proof {
            lemma_two_queue_after_cached_head_eviction(
                old(self), self, bid,
            );
        }
        assert(cs_valid(self));
        assert(cached_leaf_eviction_frame(old(self), self, bid));
        true
    }

    // Intrusive-queue eviction consumes the first cached provenance leaf.
    // Topological queue order determines eligibility; ordering among eligible
    // leaves remains replacement policy only.
    #[verifier::spinoff_prover]
    #[verifier::rlimit(100)]
    pub fn evict_one_cached_leaf(&mut self) -> (out: Option<BlockId>)
        requires
            cs_valid(old(self)),
            free_queue_valid(old(self)),
        ensures
            cs_valid(final(self)),
            free_queue_valid(final(self)),
            final(self).config == old(self).config,
            final(self).num_blocks == old(self).num_blocks,
            final(self).running@ == old(self).running@,
            final(self).waiting@ == old(self).waiting@,
            final(self).live_requests@ == old(self).live_requests@,
            final(self).accepted_requests@ == old(self).accepted_requests@,
            final(self).request_residency@ == old(self).request_residency@,
            final(self).free_blocks >= old(self).free_blocks,
            forall|bid: BlockId| #[trigger] final(self).blocks@.contains_key(bid)
                ==> old(self).blocks@.contains_key(bid)
                    && final(self).blocks@[bid] == old(self).blocks@[bid],
            forall|bid: BlockId| #[trigger] old(self).blocks@.contains_key(bid)
                && old(self).blocks@[bid].refcount > 0
                ==> final(self).blocks@.contains_key(bid)
                    && final(self).blocks@[bid] == old(self).blocks@[bid],
            forall|h: u64| #[trigger] final(self).hash_to_block@.contains_key(h)
                ==> old(self).hash_to_block@.contains_key(h)
                    && final(self).hash_to_block@[h] == old(self).hash_to_block@[h],
            match out {
                Some(bid) => cached_leaf_eviction_frame(old(self), final(self), bid),
                None => final(self).blocks@ == old(self).blocks@
                    && final(self).hash_to_block@ == old(self).hash_to_block@
                    && final(self).free_blocks == old(self).free_blocks
                    && final(self).cached_queue.head is None
                    && forall|bid: BlockId|
                        !zero_ref_cached_page(final(self), bid),
            },
            persistent_provenance_closed(old(self))
                ==> persistent_provenance_closed(final(self)),
    {
        let bid = match eviction_policy::select_cached_eviction_candidate(self) {
            Some(candidate) => candidate,
            None => return None,
        };
        let evicted = self.evict_cached_leaf(bid);
        assert(evicted);
        Some(bid)
    }

    #[verifier::spinoff_prover]
    #[verifier::rlimit(200)]
    pub fn reclaim_cached_leaves_until(
        &mut self,
        required_free: u64,
    ) -> (reclaimed: u64)
        requires
            cs_valid(old(self)),
            free_queue_valid(old(self)),
        ensures
            cs_valid(final(self)),
            free_queue_valid(final(self)),
            final(self).config == old(self).config,
            final(self).num_blocks == old(self).num_blocks,
            final(self).running@ == old(self).running@,
            final(self).waiting@ == old(self).waiting@,
            final(self).live_requests@ == old(self).live_requests@,
            final(self).accepted_requests@ == old(self).accepted_requests@,
            final(self).request_residency@ == old(self).request_residency@,
            final(self).free_blocks as int
                == old(self).free_blocks as int + reclaimed as int,
            reclaimed <= old(self).num_blocks,
            required_free <= final(self).free_blocks
                || forall|bid: BlockId|
                    !zero_ref_cached_page(final(self), bid),
            forall|bid: BlockId| #[trigger] final(self).blocks@.contains_key(bid)
                ==> old(self).blocks@.contains_key(bid)
                    && final(self).blocks@[bid] == old(self).blocks@[bid],
            forall|bid: BlockId| #[trigger] old(self).blocks@.contains_key(bid)
                && old(self).blocks@[bid].refcount > 0
                ==> final(self).blocks@.contains_key(bid)
                    && final(self).blocks@[bid] == old(self).blocks@[bid],
            forall|h: u64| #[trigger] final(self).hash_to_block@.contains_key(h)
                ==> old(self).hash_to_block@.contains_key(h)
                    && final(self).hash_to_block@[h] == old(self).hash_to_block@[h],
            persistent_provenance_closed(old(self))
                ==> persistent_provenance_closed(final(self)),
            positive_provenance_origin(old(self), final(self)),
    {
        let mut count: u64 = 0;
        while self.free_blocks < required_free && count < self.num_blocks
            invariant
                count <= self.num_blocks,
                cs_valid(self),
                free_queue_valid(self),
                self.config == old(self).config,
                self.num_blocks == old(self).num_blocks,
                self.running@ == old(self).running@,
                self.waiting@ == old(self).waiting@,
                self.live_requests@ == old(self).live_requests@,
                self.accepted_requests@ == old(self).accepted_requests@,
                self.request_residency@ == old(self).request_residency@,
                self.free_blocks as int
                    == old(self).free_blocks as int + count as int,
                forall|bid: BlockId| #[trigger] self.blocks@.contains_key(bid)
                    ==> old(self).blocks@.contains_key(bid)
                        && self.blocks@[bid] == old(self).blocks@[bid],
                forall|bid: BlockId| #[trigger] old(self).blocks@.contains_key(bid)
                    && old(self).blocks@[bid].refcount > 0
                    ==> self.blocks@.contains_key(bid)
                        && self.blocks@[bid] == old(self).blocks@[bid],
                forall|h: u64| #[trigger] self.hash_to_block@.contains_key(h)
                    ==> old(self).hash_to_block@.contains_key(h)
                        && self.hash_to_block@[h] == old(self).hash_to_block@[h],
                persistent_provenance_closed(old(self))
                    ==> persistent_provenance_closed(self),
            decreases self.num_blocks - count
        {
            let ghost before = *self;
            let evicted = self.evict_one_cached_leaf();
            match evicted {
                Some(victim) => {
                    assert(self.free_blocks as int
                        == before.free_blocks as int + 1) by {
                        assert(cached_leaf_eviction_frame(
                            &before, self, victim,
                        ));
                    }
                    assert forall|bid: BlockId|
                        #[trigger] self.blocks@.contains_key(bid)
                        implies old(self).blocks@.contains_key(bid)
                            && self.blocks@[bid] == old(self).blocks@[bid]
                    by {
                        assert(before.blocks@.contains_key(bid));
                        assert(self.blocks@[bid] == before.blocks@[bid]);
                    }
                    assert forall|bid: BlockId|
                        #[trigger] old(self).blocks@.contains_key(bid)
                        && old(self).blocks@[bid].refcount > 0
                        implies self.blocks@.contains_key(bid)
                            && self.blocks@[bid] == old(self).blocks@[bid]
                    by {
                        assert(before.blocks@.contains_key(bid));
                        assert(before.blocks@[bid] == old(self).blocks@[bid]);
                        assert(before.blocks@[bid].refcount > 0);
                    }
                    assert forall|h: u64|
                        #[trigger] self.hash_to_block@.contains_key(h)
                        implies old(self).hash_to_block@.contains_key(h)
                            && self.hash_to_block@[h]
                                == old(self).hash_to_block@[h]
                    by {
                        assert(before.hash_to_block@.contains_key(h));
                        assert(self.hash_to_block@[h]
                            == before.hash_to_block@[h]);
                    }
                    count = count + 1;
                },
                None => {
                    proof {
                        lemma_positive_provenance_origin_from_surviving_blocks(
                            old(self), self,
                        );
                    }
                    return count;
                },
            }
        }
        if self.free_blocks < required_free {
            assert(count == self.num_blocks);
            assert(self.free_blocks <= self.num_blocks) by {
                assert(block_count_valid(self));
            }
            assert(old(self).free_blocks == 0);
            assert(self.free_blocks == self.num_blocks);
            assert(self.blocks@.dom().len() == 0) by {
                assert(block_count_valid(self));
            }
            assert forall|bid: BlockId|
                !zero_ref_cached_page(self, bid)
            by {
                if zero_ref_cached_page(self, bid) {
                    assert(self.blocks@.contains_key(bid));
                    assert(false);
                }
            }
        }
        proof {
            lemma_positive_provenance_origin_from_surviving_blocks(
                old(self), self,
            );
        }
        count
    }

    #[verifier::spinoff_prover]
    #[verifier::rlimit(10)]
    fn append_deallocated_cached_queue(
        &mut self,
        ids: &Vec<BlockId>,
        Ghost(before): Ghost<CacheScheduler>,
        Ghost(released_vacant): Ghost<Seq<BlockId>>,
    )
        requires
            cs_valid(&before),
            persistent_provenance_closed(&before),
            free_queue_valid_token(&before),
            before.free_queue.len <= before.num_blocks,
            forall|b: BlockId| #[trigger] before.free_queue.order@.contains(b)
                ==> b < before.num_blocks
                    && (!before.blocks@.contains_key(b)
                        || before.blocks@[b].refcount == 0),
            cs_valid(old(self)),
            persistent_provenance_closed(old(self)),
            old(self).num_blocks == before.num_blocks,
            free_queue_shape_token(&old(self).free_queue),
            old(self).free_queue.order@
                == released_vacant + before.free_queue.order@,
            old(self).cached_queue.head == before.cached_queue.head,
            old(self).cached_queue.tail == before.cached_queue.tail,
            old(self).cached_queue.len == before.cached_queue.len,
            old(self).cached_queue.links@ == before.cached_queue.links@,
            old(self).cached_queue.order@ == before.cached_queue.order@,
            old(self).free_queue.len <= old(self).num_blocks,
            forall|b: BlockId| #[trigger] old(self).free_queue.order@.contains(b)
                ==> b < old(self).num_blocks,
            forall|b: BlockId| #[trigger] released_vacant.contains(b)
                ==> ids@.contains(b) && !old(self).blocks@.contains_key(b),
            released_vacant.no_duplicates(),
            forall|j: int| 0 <= j < ids@.len()
                && !old(self).blocks@.contains_key(#[trigger] ids@[j])
                ==> released_vacant.contains(ids@[j]),
            ids@.no_duplicates(),
            forall|j: int| 0 <= j < ids@.len() ==> {
                let bid = #[trigger] ids@[j];
                &&& before.blocks@.contains_key(bid)
                &&& before.blocks@[bid].refcount > 0
                &&& bid < before.num_blocks
            },
            forall|bid: BlockId| #[trigger] old(self).blocks@.contains_key(bid)
                ==> before.blocks@.contains_key(bid),
            forall|bid: BlockId| #[trigger] old(self).blocks@.contains_key(bid)
                ==> old(self).blocks@[bid].prefix_depth
                    == before.blocks@[bid].prefix_depth
                    && old(self).blocks@[bid].parent_block
                        == before.blocks@[bid].parent_block,
            forall|bid: BlockId| #[trigger] before.blocks@.contains_key(bid)
                && !old(self).blocks@.contains_key(bid)
                ==> ids@.contains(bid)
                    && before.blocks@[bid].prefix_depth == 0,
            forall|bid: BlockId| #[trigger] before.blocks@.contains_key(bid)
                && !ids@.contains(bid)
                ==> old(self).blocks@.contains_key(bid)
                    && old(self).blocks@[bid] == before.blocks@[bid],
            forall|j: int| 0 <= j < ids@.len()
                && old(self).blocks@.contains_key(#[trigger] ids@[j])
                && old(self).blocks@[ids@[j]].refcount == 0
                ==> old(self).blocks@[ids@[j]].prefix_depth > 0,
            forall|j: int| 0 <= j < ids@.len()
                && before.blocks@[#[trigger] ids@[j]].prefix_depth > 0
                ==> before.blocks@[ids@[j]].prefix_depth as int == j + 1
                    && before.blocks@[ids@[j]].parent_block
                        == if j == 0 { None } else { Some(ids@[j - 1]) },
        ensures
            cs_valid(final(self)),
            persistent_provenance_closed(final(self)),
            free_queue_valid(final(self)),
            final(self).config == old(self).config,
            final(self).num_blocks == old(self).num_blocks,
            final(self).free_blocks == old(self).free_blocks,
            final(self).blocks@ == old(self).blocks@,
            final(self).running@ == old(self).running@,
            final(self).waiting@ == old(self).waiting@,
            final(self).request_residency@ == old(self).request_residency@,
            final(self).live_requests@ == old(self).live_requests@,
            final(self).accepted_requests@ == old(self).accepted_requests@,
            final(self).hash_to_block@ == old(self).hash_to_block@,
    {
        let ghost semantic_post = *self;
        let ghost queue_before_release = before.free_queue.order@;
        let ghost cached_before_release = before.cached_queue.order@;
        let ghost queue_after_vacant = self.free_queue.order@;
        proof {
            lemma_free_queue_token_to_valid(&before);
            lemma_free_queue_valid_token_cached_shape_token(&before);
            lemma_free_queue_shape_token_frame(
                &before.cached_queue, &self.cached_queue,
            );
        }
        assert(self.cached_queue.len <= self.num_blocks) by {
            assert(before.cached_queue.order@.len()
                <= before.num_blocks as int);
        }
        assert forall|b: BlockId|
            #[trigger] self.cached_queue.order@.contains(b)
            implies b < self.num_blocks
        by {
            assert(before.cached_queue.order@.contains(b));
            assert(zero_ref_cached_page(&before, b));
            assert(blocks_dom_in_range(&before));
        }
        // Pass two appends reusable pages in tail-to-root order.
        let mut released_cached: Ghost<Seq<BlockId>> =
            Ghost(Seq::<BlockId>::empty());
        let mut remaining: usize = ids.len();
        while remaining > 0
            invariant
                remaining <= ids@.len(),
                cs_valid(&before),
                free_queue_valid(&before),
                free_queue_shape_token(&self.free_queue),
                free_queue_shape_token(&self.cached_queue),
                self.free_queue.order@ == queue_after_vacant,
                self.cached_queue.order@
                    == cached_before_release + released_cached@,
                cached_before_release == before.cached_queue.order@,
                released_cached@.no_duplicates(),
                forall|b: BlockId| #[trigger] released_cached@.contains(b)
                    ==> ids@.contains(b)
                        && self.blocks@.contains_key(b)
                        && self.blocks@[b].refcount == 0
                        && self.blocks@[b].prefix_depth > 0,
                forall|j: int| remaining as int <= j && j < ids@.len()
                    && self.blocks@.contains_key(#[trigger] ids@[j])
                    && self.blocks@[ids@[j]].refcount == 0
                    ==> released_cached@.contains(ids@[j]),
                forall|j: int| 0 <= j < remaining as int
                    ==> !released_cached@.contains(#[trigger] ids@[j]),
                forall|i: int, j: int| 0 <= i < j < ids@.len()
                    && released_cached@.contains(#[trigger] ids@[i])
                    && released_cached@.contains(#[trigger] ids@[j])
                    ==> released_cached@.index_of(ids@[j])
                        < released_cached@.index_of(ids@[i]),
                self.free_queue.len <= self.num_blocks,
                self.cached_queue.len <= self.num_blocks,
                forall|b: BlockId| #[trigger] self.free_queue.order@.contains(b)
                    ==> b < self.num_blocks,
                forall|b: BlockId| #[trigger] self.cached_queue.order@.contains(b)
                    ==> b < self.num_blocks,
                ids@.no_duplicates(),
                queue_after_vacant == released_vacant + queue_before_release,
                forall|b: BlockId| #[trigger] released_vacant.contains(b)
                    ==> ids@.contains(b) && !self.blocks@.contains_key(b),
                queue_before_release == before.free_queue.order@,
                before.free_queue.len <= before.num_blocks,
                forall|b: BlockId| #[trigger] queue_before_release.contains(b)
                    ==> b < before.num_blocks
                        && (!before.blocks@.contains_key(b)
                            || before.blocks@[b].refcount == 0),
                self.num_blocks == semantic_post.num_blocks,
                self.config == semantic_post.config,
                self.free_blocks == semantic_post.free_blocks,
                self.blocks@ == semantic_post.blocks@,
                self.running@ == semantic_post.running@,
                self.waiting@ == semantic_post.waiting@,
                self.request_residency@ == semantic_post.request_residency@,
                self.live_requests@ == semantic_post.live_requests@,
                self.accepted_requests@ == semantic_post.accepted_requests@,
                self.hash_to_block@ == semantic_post.hash_to_block@,
                semantic_post.num_blocks == before.num_blocks,
                forall|b: BlockId| #[trigger] self.blocks@.contains_key(b)
                    ==> before.blocks@.contains_key(b)
                        && self.blocks@[b].prefix_depth
                            == before.blocks@[b].prefix_depth
                        && self.blocks@[b].parent_block
                            == before.blocks@[b].parent_block,
                forall|j: int| 0 <= j < ids@.len() ==> {
                    let b = #[trigger] ids@[j];
                    &&& before.blocks@.contains_key(b)
                    &&& before.blocks@[b].refcount > 0
                    &&& b < before.num_blocks
                },
                forall|j: int| 0 <= j < ids@.len()
                    && self.blocks@.contains_key(#[trigger] ids@[j])
                    && self.blocks@[ids@[j]].refcount == 0
                    ==> self.blocks@[ids@[j]].prefix_depth > 0,
                forall|j: int| 0 <= j < ids@.len()
                    && before.blocks@[#[trigger] ids@[j]].prefix_depth > 0
                    ==> before.blocks@[ids@[j]].prefix_depth as int == j + 1
                        && before.blocks@[ids@[j]].parent_block
                            == if j == 0 { None } else { Some(ids@[j - 1]) },
            decreases remaining
        {
            let ghost cached_before = released_cached@;
            let ghost queue_before_iter = self.cached_queue.order@;
            let remaining_before = remaining;
            remaining -= 1;
            assert(remaining_before == remaining + 1);
            let bid = ids[remaining];
            assert(bid == ids@[remaining as int]);
            let became_cached = match self.blocks.get(&bid) {
                Some(e) => e.refcount == 0 && e.prefix_depth > 0,
                None => false,
            };
            assert(became_cached == (self.blocks@.contains_key(bid)
                && self.blocks@[bid].refcount == 0
                && self.blocks@[bid].prefix_depth > 0));
            if became_cached {
                assert(self.blocks@.contains_key(bid));
                assert(self.blocks@[bid].refcount == 0);
                assert(self.blocks@[bid].prefix_depth > 0);
                assert(!self.cached_queue.order@.contains(bid)) by {
                    if self.cached_queue.order@.contains(bid) {
                        assert(queue_before_iter.contains(bid));
                        assert(!released_cached@.contains(bid));
                        assert(cached_before_release.contains(bid));
                        assert(before.cached_queue.order@.contains(bid));
                        assert(before.blocks@.contains_key(bid));
                        assert(before.blocks@[bid].refcount > 0);
                        assert(zero_ref_cached_page(&before, bid));
                        assert(before.blocks@[bid].refcount == 0);
                    }
                }
                proof {
                    lemma_free_queue_token_has_room_for_missing(
                        &self.cached_queue, self.num_blocks, bid,
                    );
                }
                self.cached_queue.append_tokenized(bid);
                released_cached = Ghost(cached_before.push(bid));
                assert(released_cached@.no_duplicates());
                assert(released_cached@[released_cached@.len() - 1] == bid);
                assert(released_cached@.contains(bid));
                assert(self.cached_queue.order@
                    == cached_before_release + released_cached@) by {
                    assert_seqs_equal!(
                        self.cached_queue.order@
                            == cached_before_release + released_cached@
                    );
                }
                assert forall|b: BlockId| #[trigger] released_cached@.contains(b)
                    implies ids@.contains(b)
                        && self.blocks@.contains_key(b)
                        && self.blocks@[b].refcount == 0
                        && self.blocks@[b].prefix_depth > 0
                by {
                    if b == bid {
                        assert(ids@.contains(bid));
                    } else {
                        assert(cached_before.contains(b));
                    }
                }
                assert forall|b: BlockId|
                    #[trigger] self.cached_queue.order@.contains(b)
                    implies b < self.num_blocks
                by {
                    if b == bid {
                        assert(before.blocks@.contains_key(bid));
                        assert(bid < before.num_blocks);
                    } else {
                        assert(queue_before_iter.contains(b));
                    }
                }
                assert(self.cached_queue.len <= self.num_blocks);
            }
            assert forall|j: int| remaining as int <= j && j < ids@.len()
                && self.blocks@.contains_key(#[trigger] ids@[j])
                && self.blocks@[ids@[j]].refcount == 0
                implies released_cached@.contains(ids@[j])
            by {
                if j == remaining as int {
                    assert(ids@[j] == bid);
                    assert(self.blocks@[bid].prefix_depth > 0);
                    assert(became_cached);
                    assert(released_cached@.contains(bid));
                    assert(released_cached@.contains(ids@[j]));
                } else {
                    assert(remaining_before as int <= j);
                    assert(cached_before.contains(ids@[j]));
                    if became_cached {
                        assert(released_cached@ == cached_before.push(bid));
                        let ci = cached_before.index_of(ids@[j]);
                        assert(0 <= ci < cached_before.len());
                        assert(cached_before[ci] == ids@[j]);
                        assert(released_cached@[ci] == ids@[j]);
                    } else {
                        assert(released_cached@ == cached_before);
                    }
                    assert(released_cached@.contains(ids@[j]));
                }
            }
            assert forall|j: int| 0 <= j < remaining as int
                implies !released_cached@.contains(#[trigger] ids@[j])
            by {
                assert(j < remaining_before as int);
                if released_cached@.contains(ids@[j])
                    && !cached_before.contains(ids@[j])
                {
                    assert(ids@[j] == bid);
                    assert(ids@[remaining as int] == bid);
                    assert(ids@.no_duplicates());
                    assert(j != remaining as int);
                }
            }
            assert forall|i: int, j: int| 0 <= i < j < ids@.len()
                && released_cached@.contains(#[trigger] ids@[i])
                && released_cached@.contains(#[trigger] ids@[j])
                implies released_cached@.index_of(ids@[j])
                    < released_cached@.index_of(ids@[i])
            by {
                if became_cached && ids@[i] == bid {
                    assert(i == remaining as int) by {
                        assert(ids@[remaining as int] == bid);
                        assert(ids@.no_duplicates());
                    }
                    assert(j > remaining as int);
                    assert(cached_before.contains(ids@[j]));
                    assert(released_cached@.index_of(ids@[i])
                        == cached_before.len() as int) by {
                        assert(released_cached@[cached_before.len() as int] == bid);
                        assert(released_cached@.no_duplicates());
                    }
                    assert(released_cached@.index_of(ids@[j])
                        < cached_before.len() as int) by {
                        let ji = cached_before.index_of(ids@[j]);
                        assert(0 <= ji < cached_before.len());
                        assert(released_cached@[ji] == ids@[j]);
                        assert(released_cached@.no_duplicates());
                    }
                } else {
                    assert(cached_before.contains(ids@[i]));
                    assert(cached_before.contains(ids@[j]));
                    assert(cached_before.index_of(ids@[j])
                        < cached_before.index_of(ids@[i]));
                    let ii = cached_before.index_of(ids@[i]);
                    let ji = cached_before.index_of(ids@[j]);
                    assert(released_cached@[ii] == ids@[i]);
                    assert(released_cached@[ji] == ids@[j]);
                    assert(released_cached@.no_duplicates());
                }
            }
        }
        assert(cs_valid(self)) by {
            lemma_cs_valid_semantic_frame(&semantic_post, self);
        }
        assert(self.free_queue.order@
            == released_vacant + queue_before_release);
        assert(self.cached_queue.order@
            == cached_before_release + released_cached@);
        assert forall|j: int| 0 <= j < ids@.len()
            && self.blocks@.contains_key(#[trigger] ids@[j])
            && self.blocks@[ids@[j]].refcount == 0
            implies released_cached@.contains(ids@[j]) by {}
        assert(free_queue_valid(self)) by {
            lemma_free_queue_token_to_valid(&before);
            lemma_reconciled_vacant_count(
                &before, self, ids@, released_vacant,
            );
            assert forall|b: BlockId| #[trigger] released_vacant.contains(b)
                implies before.blocks@.contains_key(b)
                    && !self.blocks@.contains_key(b)
            by {
                assert(ids@.contains(b));
                let j = ids@.index_of(b);
                assert(0 <= j < ids@.len());
                assert(ids@[j] == b);
            }
            assert forall|b: BlockId| #[trigger] before.blocks@.contains_key(b)
                && !self.blocks@.contains_key(b)
                implies released_vacant.contains(b)
            by {
                assert(ids@.contains(b));
                let j = ids@.index_of(b);
                assert(0 <= j < ids@.len());
                assert(ids@[j] == b);
            }
            assert forall|b: BlockId| #[trigger] self.blocks@.contains_key(b)
                implies before.blocks@.contains_key(b)
                    && self.blocks@[b].prefix_depth
                        == before.blocks@[b].prefix_depth
                    && self.blocks@[b].parent_block
                        == before.blocks@[b].parent_block
            by {}
            assert forall|b: BlockId| #[trigger] zero_ref_cached_page(&before, b)
                implies self.blocks@.contains_key(b)
                    && self.blocks@[b] == before.blocks@[b]
            by {
                assert(!ids@.contains(b)) by {
                    if ids@.contains(b) {
                        let j = ids@.index_of(b);
                        assert(0 <= j < ids@.len());
                        assert(ids@[j] == b);
                        assert(before.blocks@[b].refcount > 0);
                    }
                }
            }
            assert forall|b: BlockId| #[trigger] released_cached@.contains(b)
                implies zero_ref_cached_page(self, b)
                    && !zero_ref_cached_page(&before, b)
            by {
                assert(ids@.contains(b));
                let j = ids@.index_of(b);
                assert(0 <= j < ids@.len());
                assert(ids@[j] == b);
                assert(before.blocks@[b].refcount > 0);
            }
            assert forall|b: BlockId| #[trigger] zero_ref_cached_page(self, b)
                && !zero_ref_cached_page(&before, b)
                implies released_cached@.contains(b)
            by {
                assert(ids@.contains(b)) by {
                    if !ids@.contains(b) {
                        assert(self.blocks@[b] == before.blocks@[b]);
                        assert(zero_ref_cached_page(&before, b));
                    }
                }
                let j = ids@.index_of(b);
                assert(0 <= j < ids@.len());
                assert(ids@[j] == b);
            }
            assert forall|child: BlockId| self.blocks@.contains_key(child)
                && self.blocks@[child].prefix_depth > 0
                && self.blocks@[child].parent_block is Some
                && released_cached@.contains(child)
                && released_cached@.contains(
                    self.blocks@[child].parent_block.unwrap(),
                )
                implies #[trigger] released_cached@.index_of(child)
                    < released_cached@.index_of(
                        self.blocks@[child].parent_block.unwrap(),
                    )
            by {
                let parent = self.blocks@[child].parent_block.unwrap();
                assert(ids@.contains(child));
                assert(ids@.contains(parent));
                let ci = ids@.index_of(child);
                let pi = ids@.index_of(parent);
                assert(0 <= ci < ids@.len());
                assert(0 <= pi < ids@.len());
                assert(ids@[ci] == child);
                assert(ids@[pi] == parent);
                assert(before.blocks@[child].prefix_depth > 0);
                assert(before.blocks@[child].parent_block == Some(parent));
                assert(ci > 0);
                assert(before.blocks@[child].parent_block
                    == Some(ids@[ci - 1]));
                assert(pi == ci - 1) by {
                    assert(ids@[pi] == ids@[ci - 1]);
                    assert(ids@.no_duplicates());
                }
                assert(released_cached@.index_of(ids@[ci])
                    < released_cached@.index_of(ids@[pi]));
            }
            lemma_free_queue_token_to_shape(&self.free_queue);
            lemma_free_queue_token_to_shape(&self.cached_queue);
            lemma_two_queue_after_request_release(
                &before, self, released_vacant, released_cached@,
            );
        }
        assert(persistent_provenance_closed(self)) by {
            lemma_persistent_provenance_closed_blocks_eq(
                &semantic_post, self,
            );
        }

    }


    // Representation half of request release.  The semantic deallocation
    // transition first updates refcounts/residency and proves `cs_valid`;
    // this separately verified pass then exposes newly available pages in the
    // intrusive queue. Splitting the transitions keeps the semantic
    // proof independent of queue policy and makes the executable pass linear
    // only in the completed request's block table.
    #[verifier::spinoff_prover]
    #[verifier::rlimit(100)]
    fn reconcile_deallocated_queue(
        &mut self,
        ids: &Vec<BlockId>,
        Ghost(before): Ghost<CacheScheduler>,
    )
        requires
            cs_valid(&before),
            persistent_provenance_closed(&before),
            free_queue_valid_token(&before),
            before.free_queue.len <= before.num_blocks,
            forall|b: BlockId| #[trigger] before.free_queue.order@.contains(b)
                ==> b < before.num_blocks
                    && (!before.blocks@.contains_key(b)
                        || before.blocks@[b].refcount == 0),
            cs_valid(old(self)),
            persistent_provenance_closed(old(self)),
            old(self).num_blocks == before.num_blocks,
            old(self).free_queue.head == before.free_queue.head,
            old(self).free_queue.tail == before.free_queue.tail,
            old(self).free_queue.len == before.free_queue.len,
            old(self).free_queue.links@ == before.free_queue.links@,
            old(self).free_queue.order@ == before.free_queue.order@,
            old(self).cached_queue.head == before.cached_queue.head,
            old(self).cached_queue.tail == before.cached_queue.tail,
            old(self).cached_queue.len == before.cached_queue.len,
            old(self).cached_queue.links@ == before.cached_queue.links@,
            old(self).cached_queue.order@ == before.cached_queue.order@,
            ids@.no_duplicates(),
            forall|j: int| 0 <= j < ids@.len() ==> {
                let bid = #[trigger] ids@[j];
                &&& before.blocks@.contains_key(bid)
                &&& before.blocks@[bid].refcount > 0
                &&& bid < before.num_blocks
                &&& (old(self).blocks@.contains_key(bid) ==> {
                    &&& old(self).blocks@[bid].tokens@ == before.blocks@[bid].tokens@
                    &&& old(self).blocks@[bid].refcount as int + 1
                        == before.blocks@[bid].refcount as int
                    &&& old(self).blocks@[bid].hash_value == before.blocks@[bid].hash_value
                    &&& old(self).blocks@[bid].prefix_depth == before.blocks@[bid].prefix_depth
                    &&& old(self).blocks@[bid].parent_block == before.blocks@[bid].parent_block
                })
            },
            forall|bid: BlockId| #[trigger] old(self).blocks@.contains_key(bid)
                ==> before.blocks@.contains_key(bid),
            forall|bid: BlockId| #[trigger] before.blocks@.contains_key(bid)
                && !old(self).blocks@.contains_key(bid)
                ==> ids@.contains(bid)
                    && before.blocks@[bid].prefix_depth == 0,
            forall|bid: BlockId| #[trigger] before.blocks@.contains_key(bid)
                && !ids@.contains(bid)
                ==> old(self).blocks@.contains_key(bid)
                    && old(self).blocks@[bid] == before.blocks@[bid],
            forall|j: int| 0 <= j < ids@.len()
                && old(self).blocks@.contains_key(#[trigger] ids@[j])
                && old(self).blocks@[ids@[j]].refcount == 0
                ==> old(self).blocks@[ids@[j]].prefix_depth > 0,
            forall|j: int| 0 <= j < ids@.len()
                && before.blocks@[#[trigger] ids@[j]].prefix_depth > 0
                ==> before.blocks@[ids@[j]].prefix_depth as int == j + 1
                    && before.blocks@[ids@[j]].parent_block
                        == if j == 0 { None } else { Some(ids@[j - 1]) },
        ensures
            cs_valid(final(self)),
            persistent_provenance_closed(final(self)),
            free_queue_valid(final(self)),
            final(self).config == old(self).config,
            final(self).num_blocks == old(self).num_blocks,
            final(self).free_blocks == old(self).free_blocks,
            final(self).blocks@ == old(self).blocks@,
            final(self).running@ == old(self).running@,
            final(self).waiting@ == old(self).waiting@,
            final(self).request_residency@ == old(self).request_residency@,
            final(self).live_requests@ == old(self).live_requests@,
            final(self).accepted_requests@ == old(self).accepted_requests@,
            final(self).hash_to_block@ == old(self).hash_to_block@,
    {
        let ghost semantic_post = *self;
        let ghost queue_before_release = self.free_queue.order@;
        assert(queue_before_release == before.free_queue.order@);
        proof {
            lemma_free_queue_valid_token_shape_token(&before);
            lemma_free_queue_shape_token_frame(
                &before.free_queue, &self.free_queue,
            );
            lemma_free_queue_valid_token_cached_shape_token(&before);
            lemma_free_queue_shape_token_frame(
                &before.cached_queue, &self.cached_queue,
            );
        }
        let mut released_vacant: Ghost<Seq<BlockId>> =
            Ghost(Seq::<BlockId>::empty());
        // Pass one grows only the vacant prefix.
        let mut remaining: usize = ids.len();
        while remaining > 0
            invariant
                remaining <= ids@.len(),
                free_queue_shape_token(&self.free_queue),
                free_queue_shape_token(&self.cached_queue),
                self.free_queue.order@ == released_vacant@ + queue_before_release,
                released_vacant@.no_duplicates(),
                forall|b: BlockId| #[trigger] released_vacant@.contains(b)
                    ==> ids@.contains(b)
                        && !self.blocks@.contains_key(b),
                forall|j: int| remaining as int <= j && j < ids@.len()
                    && !self.blocks@.contains_key(#[trigger] ids@[j])
                    ==> released_vacant@.contains(ids@[j]),
                forall|j: int| 0 <= j < remaining as int
                    ==> !released_vacant@.contains(#[trigger] ids@[j]),
                self.cached_queue.head == before.cached_queue.head,
                self.cached_queue.tail == before.cached_queue.tail,
                self.cached_queue.len == before.cached_queue.len,
                self.cached_queue.links@ == before.cached_queue.links@,
                self.cached_queue.order@ == before.cached_queue.order@,
                self.free_queue.len <= self.num_blocks,
                forall|b: BlockId| #[trigger] self.free_queue.order@.contains(b)
                    ==> b < self.num_blocks,
                ids@.no_duplicates(),
                queue_before_release == before.free_queue.order@,
                before.free_queue.len <= before.num_blocks,
                forall|b: BlockId| #[trigger] queue_before_release.contains(b)
                    ==> b < before.num_blocks
                        && (!before.blocks@.contains_key(b)
                            || before.blocks@[b].refcount == 0),
                self.num_blocks == semantic_post.num_blocks,
                self.config == semantic_post.config,
                self.free_blocks == semantic_post.free_blocks,
                self.blocks@ == semantic_post.blocks@,
                self.running@ == semantic_post.running@,
                self.waiting@ == semantic_post.waiting@,
                self.request_residency@ == semantic_post.request_residency@,
                self.live_requests@ == semantic_post.live_requests@,
                self.accepted_requests@ == semantic_post.accepted_requests@,
                self.hash_to_block@ == semantic_post.hash_to_block@,
                semantic_post.num_blocks == before.num_blocks,
                forall|j: int| 0 <= j < ids@.len() ==> {
                    let b = #[trigger] ids@[j];
                    &&& before.blocks@.contains_key(b)
                    &&& before.blocks@[b].refcount > 0
                    &&& b < before.num_blocks
                },
                forall|j: int| 0 <= j < ids@.len()
                    && self.blocks@.contains_key(#[trigger] ids@[j])
                    && self.blocks@[ids@[j]].refcount == 0
                    ==> self.blocks@[ids@[j]].prefix_depth > 0,
                forall|j: int| 0 <= j < ids@.len()
                    && before.blocks@[#[trigger] ids@[j]].prefix_depth > 0
                    ==> before.blocks@[ids@[j]].prefix_depth as int == j + 1
                        && before.blocks@[ids@[j]].parent_block
                            == if j == 0 { None } else { Some(ids@[j - 1]) },
            decreases remaining
        {
            let ghost vacant_before = released_vacant@;
            let ghost queue_before_iter = self.free_queue.order@;
            let remaining_before = remaining;
            remaining -= 1;
            assert(remaining_before == remaining + 1);
            let bid = ids[remaining];
            assert(bid == ids@[remaining as int]);
            let became_vacant = match self.blocks.get(&bid) {
                Some(_) => false,
                None => true,
            };
            assert(became_vacant == !self.blocks@.contains_key(bid));
            if became_vacant {
                assert(!self.free_queue.order@.contains(bid)) by {
                    if self.free_queue.order@.contains(bid) {
                        assert(released_vacant@.contains(bid)
                            || queue_before_release.contains(bid));
                        assert(!released_vacant@.contains(bid));
                        if queue_before_release.contains(bid) {
                            assert(before.blocks@.contains_key(bid));
                            assert(before.blocks@[bid].refcount > 0);
                            assert(before.blocks@[bid].refcount == 0);
                        }
                    }
                }
                let ghost singleton = Seq::<BlockId>::empty().push(bid);
                proof {
                    lemma_free_queue_token_has_room_for_missing(
                        &self.free_queue, self.num_blocks, bid,
                    );
                }
                self.free_queue.prepend_tokenized(bid);
                released_vacant = Ghost(singleton + vacant_before);
                assert(released_vacant@.no_duplicates());
                assert(released_vacant@[0] == bid);
                assert(released_vacant@.contains(bid));
                proof {
                    assert_seqs_equal!(
                        self.free_queue.order@
                            == released_vacant@ + queue_before_release
                    );
                }
                assert forall|b: BlockId| #[trigger] released_vacant@.contains(b)
                    implies ids@.contains(b)
                        && !self.blocks@.contains_key(b)
                by {
                    if b == bid {
                        assert(ids@.contains(bid));
                    } else {
                        assert(vacant_before.contains(b));
                    }
                }
                assert forall|b: BlockId|
                    #[trigger] self.free_queue.order@.contains(b)
                    implies b < self.num_blocks
                by {
                    if b == bid {
                        assert(before.blocks@.contains_key(bid));
                        assert(bid < before.num_blocks);
                    } else {
                        assert(queue_before_iter.contains(b));
                    }
                }
                assert(self.free_queue.len <= self.num_blocks);
            } else {
                assert(self.blocks@.contains_key(bid));
            }
            assert forall|j: int| remaining as int <= j && j < ids@.len()
                && !self.blocks@.contains_key(#[trigger] ids@[j])
                implies released_vacant@.contains(ids@[j])
            by {
                if j == remaining as int {
                    assert(ids@[j] == bid);
                    assert(became_vacant);
                    assert(released_vacant@.contains(bid));
                    assert(released_vacant@.contains(ids@[j]));
                } else {
                    assert(remaining_before as int <= j);
                    assert(vacant_before.contains(ids@[j]));
                    if became_vacant {
                        assert(released_vacant@
                            == Seq::<BlockId>::empty().push(bid) + vacant_before);
                        let vi = vacant_before.index_of(ids@[j]);
                        assert(0 <= vi < vacant_before.len());
                        assert(vacant_before[vi] == ids@[j]);
                        assert(released_vacant@[vi + 1] == ids@[j]);
                    } else {
                        assert(released_vacant@ == vacant_before);
                    }
                    assert(released_vacant@.contains(ids@[j]));
                }
            }
            assert forall|j: int| 0 <= j < remaining as int
                implies !released_vacant@.contains(#[trigger] ids@[j])
            by {
                assert(j < remaining_before as int);
                if released_vacant@.contains(ids@[j])
                    && !vacant_before.contains(ids@[j])
                {
                    assert(ids@[j] == bid);
                    assert(ids@[remaining as int] == bid);
                    assert(ids@.no_duplicates());
                    assert(j != remaining as int);
                }
            }
        }

        let ghost queue_after_vacant = self.free_queue.order@;
        assert(queue_after_vacant == released_vacant@ + queue_before_release);
        assert forall|b: BlockId| #[trigger] released_vacant@.contains(b)
            implies ids@.contains(b) && !self.blocks@.contains_key(b) by {}
        assert forall|j: int| 0 <= j < ids@.len()
            && !self.blocks@.contains_key(#[trigger] ids@[j])
            implies released_vacant@.contains(ids@[j]) by {}

        assert forall|bid: BlockId| #[trigger] self.blocks@.contains_key(bid)
            implies before.blocks@.contains_key(bid)
                && self.blocks@[bid].prefix_depth
                    == before.blocks@[bid].prefix_depth
                && self.blocks@[bid].parent_block
                    == before.blocks@[bid].parent_block
        by {
            if ids@.contains(bid) {
                let j = ids@.index_of(bid);
                assert(0 <= j < ids@.len());
                assert(ids@[j] == bid);
            } else {
                assert(self.blocks@[bid] == before.blocks@[bid]);
            }
        }
        assert(persistent_provenance_closed(self)) by {
            lemma_persistent_provenance_closed_blocks_eq(
                &semantic_post, self,
            );
        }

        self.append_deallocated_cached_queue(
            ids, Ghost(before), Ghost(released_vacant@),
        );
    }

    // Allocator: release path.  Drop `rid`'s residency and decrement its
    // block refcounts.  Unregistered tails whose refcount reaches zero return
    // immediately to the vacant pool.  Registered/provenance-bearing pages
    // remain resident at refcount zero for cross-batch reuse; a separate
    // intrusive-queue eviction transition will reclaim them under pressure.
    //
    // Precondition: rid must already have been removed from `running` and
    // `waiting` (otherwise `running_has_residency` / queue invariants
    // would break — this is the contract from the engine side: finished
    // requests are dequeued before deallocation).
    //
    // Proof side is localized: the executable body snapshots `block_ids`
    // before the refcount/free/hash loop, removes residency at the end, and
    // proves `cs_valid` preservation.  Registered pages are explicitly
    // preserved even when this was their final residency holder.
    // Use a fresh prover instance: this proof is sensitive to solver context,
    // including unrelated planning definitions.
    #[verifier::spinoff_prover]
    #[verifier::rlimit(200)]
    pub fn deallocate(&mut self, rid: RequestId)
        requires
            cs_valid(old(self)),
            free_queue_valid(old(self)),
            persistent_provenance_closed(old(self)),
            !old(self).running@.contains(rid),
            !old(self).waiting@.contains(rid),
        ensures
            cs_valid(final(self)),
            free_queue_valid(final(self)),
            final(self).config == old(self).config,
            final(self).num_blocks == old(self).num_blocks,
            final(self).running@ == old(self).running@,
            final(self).waiting@ == old(self).waiting@,
            final(self).live_requests@ == old(self).live_requests@,
            final(self).accepted_requests@ == old(self).accepted_requests@,
            !final(self).request_residency@.contains_key(rid),
            forall|other: RequestId|
                other != rid && #[trigger] old(self).request_residency@.contains_key(other)
                ==> final(self).request_residency@.contains_key(other)
                    && final(self).request_residency@[other] == old(self).request_residency@[other],
            forall|r: RequestId| #[trigger] final(self).request_residency@.contains_key(r)
                ==> old(self).request_residency@.contains_key(r),
            // Alignment transport: surviving blocks keep their
            // tokens; blocks outside the deallocated residency are verbatim;
            // freeing only grows the pool.
            forall|b: BlockId| #[trigger] final(self).blocks@.contains_key(b)
                ==> old(self).blocks@.contains_key(b)
                    && final(self).blocks@[b].tokens@ == old(self).blocks@[b].tokens@
                    && final(self).blocks@[b].prefix_depth == old(self).blocks@[b].prefix_depth
                    && final(self).blocks@[b].parent_block == old(self).blocks@[b].parent_block,
            forall|b: BlockId| #[trigger] final(self).blocks@.contains_key(b)
                && (old(self).request_residency@.contains_key(rid)
                    ==> !old(self).request_residency@[rid].block_ids@.contains(b))
                ==> final(self).blocks@[b] == old(self).blocks@[b],
            forall|b: BlockId|
                #[trigger] old(self).blocks@[b].prefix_depth > 0
                && old(self).blocks@.contains_key(b)
                && old(self).blocks@[b].prefix_depth > 0
                ==> final(self).blocks@.contains_key(b)
                    && final(self).blocks@[b].tokens@ == old(self).blocks@[b].tokens@
                    && final(self).blocks@[b].hash_value == old(self).blocks@[b].hash_value
                    && final(self).blocks@[b].prefix_depth == old(self).blocks@[b].prefix_depth
                    && final(self).blocks@[b].parent_block == old(self).blocks@[b].parent_block,
            persistent_provenance_closed(old(self))
                ==> persistent_provenance_closed(final(self)),
            positive_provenance_metadata_frame(old(self), final(self)),
            positive_provenance_origin(old(self), final(self)),
            registry_entries_from_pre(
                old(self), final(self), Seq::<u64>::empty(),
            ),
            final(self).free_blocks >= old(self).free_blocks,
    {
        let ghost dealloc_pre = *self;
        let residency_opt: Option<RequestResidency> = match self.request_residency.get(&rid) {
            Some(r) => {
                let c = r.clone();
                assert(c.block_ids@ == r.block_ids@);
                assert(c.cached_prefix_blocks == r.cached_prefix_blocks);
                assert(c.slot_mapping@ == r.slot_mapping@);
                Some(c)
            },
            None => None,
        };
        let release_ids: Vec<BlockId> = match &residency_opt {
            Some(r) => {
                let ids = r.block_ids.clone();
                assert(ids@ == r.block_ids@);
                ids
            },
            None => Vec::<BlockId>::new(),
        };
        let ghost dealloc_block_ids =
            if old(self).request_residency@.contains_key(rid) {
                old(self).request_residency@[rid].block_ids@
            } else {
                Seq::<BlockId>::empty()
            };
        assert(release_ids@ == dealloc_block_ids) by {
            if old(self).request_residency@.contains_key(rid) {
                assert(residency_opt is Some);
            } else {
                assert(residency_opt is None);
            }
        }
        assert(release_ids@.no_duplicates()) by {
            if old(self).request_residency@.contains_key(rid) {
                assert(residency_block_ids_unique(old(self)));
            }
        }
        if let Some(residency) = residency_opt {
            assert(old(self).request_residency@.contains_key(rid));
            assert(old(self).request_residency@.dom().contains(rid));
            assert(refcount_valid(old(self)));
            assert(residency.block_ids@ == old(self).request_residency@[rid].block_ids@);
            assert(residency.block_ids@ == dealloc_block_ids);
            assert(residency.block_ids@.no_duplicates()) by {
                assert(residency_block_ids_unique(old(self)));
            }
            assert forall|j: int|
                #![auto]
                0 <= j < residency.block_ids@.len()
                implies old(self).blocks@.contains_key(residency.block_ids@[j])
            by {
                assert(residency_blocks_in_range(old(self)));
            }
            let n = residency.block_ids.len();
            let mut k: usize = 0;
            while k < n
                invariant
                    k <= n,
                    n == residency.block_ids@.len(),
                    old(self).request_residency@.contains_key(rid),
                    old(self).request_residency@.dom().contains(rid),
                    refcount_valid(old(self)),
                    residency.block_ids@ == old(self).request_residency@[rid].block_ids@,
                    residency.block_ids@ == dealloc_block_ids,
                    residency.block_ids@.no_duplicates(),
                    self.config == old(self).config,
                    self.num_blocks == old(self).num_blocks,
                    self.running@ == old(self).running@,
                    self.waiting@ == old(self).waiting@,
                    self.live_requests@ == old(self).live_requests@,
                    self.accepted_requests@ == old(self).accepted_requests@,
                    self.request_residency@ == old(self).request_residency@,
                    self.free_queue.head == old(self).free_queue.head,
                    self.free_queue.tail == old(self).free_queue.tail,
                    self.free_queue.len == old(self).free_queue.len,
                    self.free_queue.links@ == old(self).free_queue.links@,
                    self.free_queue.order@ == old(self).free_queue.order@,
                    self.cached_queue.head == old(self).cached_queue.head,
                    self.cached_queue.tail == old(self).cached_queue.tail,
                    self.cached_queue.len == old(self).cached_queue.len,
                    self.cached_queue.links@ == old(self).cached_queue.links@,
                    self.cached_queue.order@ == old(self).cached_queue.order@,
                    block_token_bound(self),
                    hash_to_block_in_range(self),
                    hash_to_block_consistent(self),
                forall|h: u64| #[trigger] self.hash_to_block@.contains_key(h)
                    ==> old(self).hash_to_block@.contains_key(h)
                        && self.hash_to_block@[h] == old(self).hash_to_block@[h],
                blocks_dom_in_range(self),
                block_count_valid(self),
                forall|j: int|
                    #![trigger residency.block_ids@[j]]
                    0 <= j < residency.block_ids@.len()
                    ==> old(self).blocks@.contains_key(residency.block_ids@[j]),
                forall|j: int|
                    #![trigger self.blocks@.contains_key(residency.block_ids@[j])]
                    k as int <= j < residency.block_ids@.len()
                    ==> self.blocks@.contains_key(residency.block_ids@[j])
                        && self.blocks@[residency.block_ids@[j]]
                            == old(self).blocks@[residency.block_ids@[j]],
                forall|j: int, other: RequestId|
                    #![trigger old(self).request_residency@[other].block_ids@.contains(residency.block_ids@[j])]
                    0 <= j < k as int
                    && !self.blocks@.contains_key(residency.block_ids@[j])
                    && other != rid
                    && old(self).request_residency@.contains_key(other)
                    ==> !old(self).request_residency@[other].block_ids@.contains(residency.block_ids@[j]),
                forall|b: BlockId|
                    #![trigger self.blocks@.contains_key(b)]
                    self.blocks@.contains_key(b)
                    ==> old(self).blocks@.contains_key(b),
                forall|b: BlockId|
                    #![trigger self.blocks@.contains_key(b)]
                    old(self).blocks@.contains_key(b)
                    && !self.blocks@.contains_key(b)
                    ==> residency.block_ids@.contains(b),
                forall|b: BlockId|
                    #![trigger self.blocks@.contains_key(b)]
                    old(self).blocks@.contains_key(b)
                    && !self.blocks@.contains_key(b)
                    ==> old(self).blocks@[b].prefix_depth == 0,
                forall|b: BlockId|
                    #![trigger self.blocks@.contains_key(b)]
                    old(self).blocks@.contains_key(b)
                    && self.blocks@.contains_key(b)
                    && !residency.block_ids@.contains(b)
                    ==> self.blocks@[b] == old(self).blocks@[b],
                forall|j: int|
                    #![trigger self.blocks@.contains_key(residency.block_ids@[j])]
                    0 <= j < k as int
                    && self.blocks@.contains_key(residency.block_ids@[j])
                    ==> self.blocks@[residency.block_ids@[j]].tokens@
                            == old(self).blocks@[residency.block_ids@[j]].tokens@
                        && self.blocks@[residency.block_ids@[j]].hash_value
                            == old(self).blocks@[residency.block_ids@[j]].hash_value
                        && self.blocks@[residency.block_ids@[j]].prefix_depth
                            == old(self).blocks@[residency.block_ids@[j]].prefix_depth
                        && self.blocks@[residency.block_ids@[j]].parent_block
                            == old(self).blocks@[residency.block_ids@[j]].parent_block
                        && self.blocks@[residency.block_ids@[j]].refcount as int + 1
                            == old(self).blocks@[residency.block_ids@[j]].refcount as int,
                forall|j: int| 0 <= j < k as int
                    && self.blocks@.contains_key(#[trigger] residency.block_ids@[j])
                    && self.blocks@[residency.block_ids@[j]].refcount == 0
                    ==> self.blocks@[residency.block_ids@[j]].prefix_depth > 0,
                decreases n - k
            {
                let bid = residency.block_ids[k];
                let ghost blocks_at_iteration = self.blocks@;
                assert(bid == residency.block_ids@[k as int]);
                assert(self.blocks@.contains_key(bid));
                assert(self.blocks@[bid] == old(self).blocks@[bid]);
                // Clone the entry out so we can modify and reinsert without
                // a borrow conflict on self.blocks.
                let entry_clone: Option<BlockEntry> = match self.blocks.get(&bid) {
                    Some(e) => {
                        let c = e.clone();
                        assert(c.tokens@ == e.tokens@);
                        assert(c.refcount == e.refcount);
                        assert(c.hash_value == e.hash_value);
                        Some(c)
                    },
                    None => None,
                };
                if let Some(e) = entry_clone {
                    assert(self.blocks@.contains_key(bid));
                    assert(e.tokens@ == self.blocks@[bid].tokens@);
                    assert(e.refcount == self.blocks@[bid].refcount);
                    assert(e.hash_value == self.blocks@[bid].hash_value);
                    assert(e.tokens@.len() <= BLOCK_SIZE_SPEC as int);
                    assert(bid < self.num_blocks);
                    assert(e.refcount == old(self).blocks@[bid].refcount);
                    assert(old(self).request_residency@[rid].block_ids@.contains(bid));
                    let ghost old_holders = residency_holders_of(old(self), bid);
                    assert(old(self).request_residency@.dom().contains(rid));
                    assert(old(self).request_residency@[rid].block_ids@.contains(bid));
                    assert(old_holders.contains(rid));
                    assert(refcount_valid(old(self)));
                    assert(old(self).blocks@[bid].refcount as int == old_holders.len() as int);
                    assert(e.refcount as int == old_holders.len() as int);
                    let new_refcount = e.refcount.saturating_sub(1);
                    // Unregistered tails become vacant immediately.  A page
                    // with prefix provenance remains resident at refcount 0:
                    // its exact chain and KV contents may be reused by a
                    // later batch and are reclaimed only by eviction.
                    if new_refcount == 0 && e.prefix_depth == 0 {
                        assert(e.refcount == 1);
                        assert(old_holders.len() == 1);
                        proof { Set::lemma_is_singleton(old_holders); }
                        assert(old_holders.is_singleton());
                        assert forall|other: RequestId|
                            #![auto]
                            other != rid
                            && old(self).request_residency@.contains_key(other)
                            implies !old(self).request_residency@[other].block_ids@.contains(bid)
                        by {
                            if old(self).request_residency@[other].block_ids@.contains(bid) {
                                assert(old_holders.contains(other));
                                assert(other == rid);
                            }
                        }
                        let ghost blocks_before_remove = self.blocks@;
                        let ghost hash_to_block_before_remove = self.hash_to_block@;
                        let free_blocks_before_remove = self.free_blocks;
                        // Hash stamps are not unique across blocks.  Only
                        // clear the registry key when it actually targets
                        // the page being freed; a collision may make it
                        // point at another live page.
                        let registry_targets_bid = match self.hash_to_block.get(&e.hash_value) {
                            Some(mapped) => *mapped == bid,
                            None => false,
                        };
                        assert(registry_targets_bid ==
                            (hash_to_block_before_remove.contains_key(e.hash_value)
                                && hash_to_block_before_remove[e.hash_value] == bid));
                        assert forall|b: BlockId|
                            #![trigger blocks_before_remove.contains_key(b)]
                            old(self).blocks@.contains_key(b)
                            && !blocks_before_remove.contains_key(b)
                            implies residency.block_ids@.contains(b)
                        by {
                        }
                        assert(block_count_valid(self));
                        assert(blocks_before_remove.dom().contains(bid));
                        assert forall|h: u64|
                            #![auto]
                            hash_to_block_before_remove.contains_key(h)
                            implies blocks_before_remove.contains_key(hash_to_block_before_remove[h])
                                && blocks_before_remove[hash_to_block_before_remove[h]].tokens@.len()
                                    == BLOCK_SIZE_SPEC as int
                        by {
                            assert(hash_to_block_in_range(self));
                        }
                        assert forall|h: u64|
                            #![auto]
                            hash_to_block_before_remove.contains_key(h)
                            implies blocks_before_remove[hash_to_block_before_remove[h]].hash_value == h
                        by {
                            assert(hash_to_block_consistent(self));
                        }
                        self.blocks.remove(&bid);
                        proof {
                            vstd::map::lemma_map_remove_domain(blocks_before_remove, bid);
                            vstd::set::lemma_set_remove_len(blocks_before_remove.dom(), bid);
                        }
                        assert(self.blocks@.dom() == blocks_before_remove.dom().remove(bid));
                        assert(blocks_before_remove.dom().len() == self.blocks@.dom().len() + 1);
                        assert(block_token_bound(self)) by {
                            assert forall|b2: BlockId|
                                #![auto]
                                self.blocks@.contains_key(b2)
                                implies self.blocks@[b2].tokens@.len() <= BLOCK_SIZE_SPEC as int
                            by {
                                assert(b2 != bid);
                                assert(blocks_before_remove.contains_key(b2));
                                assert(self.blocks@[b2] == blocks_before_remove[b2]);
                            }
                        }
                        assert(blocks_dom_in_range(self)) by {
                            assert forall|b2: BlockId|
                                #![auto]
                                self.blocks@.contains_key(b2)
                                implies b2 < self.num_blocks
                            by {
                                assert(b2 != bid);
                                assert(blocks_before_remove.contains_key(b2));
                            }
                        }
                        assert(free_blocks_before_remove < u64::MAX);
                        self.free_blocks = self.free_blocks.saturating_add(1);
                        assert(self.free_blocks as int == free_blocks_before_remove as int + 1);
                        assert(block_count_valid(self)) by {
                            assert(self.blocks@.dom().len() + 1 == blocks_before_remove.dom().len());
                        }
                        assert(self.hash_to_block@ == hash_to_block_before_remove);
                        if registry_targets_bid {
                            self.hash_to_block.remove(&e.hash_value);
                        }
                        assert forall|h: u64|
                            #![auto]
                            self.hash_to_block@.contains_key(h)
                            implies hash_to_block_before_remove.contains_key(h)
                                && self.hash_to_block@[h] == hash_to_block_before_remove[h]
                                && hash_to_block_before_remove[h] != bid
                        by {
                            if registry_targets_bid {
                                assert(h != e.hash_value);
                            } else if hash_to_block_before_remove[h] == bid {
                                assert(blocks_before_remove[bid].hash_value == h);
                                assert(blocks_before_remove[bid].hash_value == e.hash_value);
                                assert(h == e.hash_value);
                                assert(registry_targets_bid);
                            }
                        }
                        assert(hash_to_block_in_range(self)) by {
                            assert forall|h: u64|
                                #![auto]
                                self.hash_to_block@.contains_key(h)
                                implies self.blocks@.contains_key(self.hash_to_block@[h])
                                    && self.blocks@[self.hash_to_block@[h]].tokens@.len()
                                        == BLOCK_SIZE_SPEC as int
                            by {
                                assert(hash_to_block_before_remove.contains_key(h));
                                assert(self.hash_to_block@[h] == hash_to_block_before_remove[h]);
                                assert(hash_to_block_before_remove[h] != bid);
                                assert(self.blocks@.contains_key(hash_to_block_before_remove[h]));
                                assert(self.blocks@[hash_to_block_before_remove[h]]
                                    == blocks_before_remove[hash_to_block_before_remove[h]]);
                            }
                        }
                        assert(hash_to_block_consistent(self)) by {
                            assert forall|h: u64|
                                #![auto]
                                self.hash_to_block@.contains_key(h)
                                implies self.blocks@[self.hash_to_block@[h]].hash_value == h
                                    && self.blocks@[self.hash_to_block@[h]].prefix_depth > 0
                            by {
                                assert(hash_to_block_before_remove.contains_key(h));
                                assert(self.hash_to_block@[h] == hash_to_block_before_remove[h]);
                                assert(hash_to_block_before_remove[h] != bid);
                                assert(self.blocks@[hash_to_block_before_remove[h]]
                                    == blocks_before_remove[hash_to_block_before_remove[h]]);
                            }
                        }
                        assert forall|j: int, other: RequestId|
                            #![trigger old(self).request_residency@[other].block_ids@.contains(residency.block_ids@[j])]
                            0 <= j < k as int + 1
                            && !self.blocks@.contains_key(residency.block_ids@[j])
                            && other != rid
                            && old(self).request_residency@.contains_key(other)
                            implies !old(self).request_residency@[other].block_ids@.contains(residency.block_ids@[j])
                        by {
                            if j == k as int {
                                assert(residency.block_ids@[j] == bid);
                            } else {
                                assert(0 <= j < k as int);
                            }
                        }
                        assert forall|b: BlockId|
                            #![trigger self.blocks@.contains_key(b)]
                            self.blocks@.contains_key(b)
                            implies old(self).blocks@.contains_key(b)
                        by {
                            assert(b != bid);
                            assert(blocks_before_remove.contains_key(b));
                        }
                        assert forall|b: BlockId|
                            #![trigger self.blocks@.contains_key(b)]
                            old(self).blocks@.contains_key(b)
                            && !self.blocks@.contains_key(b)
                            implies residency.block_ids@.contains(b)
                        by {
                            if b == bid {
                                assert(residency.block_ids@[k as int] == bid);
                            } else {
                                assert(!blocks_before_remove.contains_key(b));
                            }
                        }
                        assert forall|b: BlockId|
                            #![trigger self.blocks@.contains_key(b)]
                            old(self).blocks@.contains_key(b)
                            && self.blocks@.contains_key(b)
                            && !residency.block_ids@.contains(b)
                            implies self.blocks@[b] == old(self).blocks@[b]
                        by {
                            assert(b != bid);
                            assert(blocks_before_remove.contains_key(b));
                            assert(self.blocks@[b] == blocks_before_remove[b]);
                        }
                        assert forall|j: int|
                            #![trigger self.blocks@.contains_key(residency.block_ids@[j])]
                            0 <= j < k as int + 1
                            && self.blocks@.contains_key(residency.block_ids@[j])
                            implies self.blocks@[residency.block_ids@[j]].tokens@
                                    == old(self).blocks@[residency.block_ids@[j]].tokens@
                                && self.blocks@[residency.block_ids@[j]].hash_value
                                    == old(self).blocks@[residency.block_ids@[j]].hash_value
                                && self.blocks@[residency.block_ids@[j]].prefix_depth
                                    == old(self).blocks@[residency.block_ids@[j]].prefix_depth
                                && self.blocks@[residency.block_ids@[j]].parent_block
                                    == old(self).blocks@[residency.block_ids@[j]].parent_block
                                && self.blocks@[residency.block_ids@[j]].refcount as int + 1
                                    == old(self).blocks@[residency.block_ids@[j]].refcount as int
                        by {
                            if j == k as int {
                                assert(residency.block_ids@[j] == bid);
                                assert(!self.blocks@.contains_key(bid));
                            } else {
                                assert(0 <= j < k as int);
                                assert(residency.block_ids@[j] != bid);
                                assert(blocks_before_remove.contains_key(residency.block_ids@[j]));
                                assert(self.blocks@[residency.block_ids@[j]]
                                    == blocks_before_remove[residency.block_ids@[j]]);
                            }
                        }
                    } else {
                        let ghost blocks_before_insert = self.blocks@;
                        assert forall|b: BlockId|
                            #![trigger blocks_before_insert.contains_key(b)]
                            old(self).blocks@.contains_key(b)
                            && !blocks_before_insert.contains_key(b)
                            implies residency.block_ids@.contains(b)
                        by {
                        }
                        assert(block_count_valid(self));
                        assert(blocks_before_insert.dom().contains(bid));
                        self.blocks.insert(bid, BlockEntry {
                            tokens: e.tokens,
                            refcount: new_refcount,
                            hash_value: e.hash_value,
                            prefix_depth: e.prefix_depth,
                            parent_block: e.parent_block,
                        });
                        proof {
                            vstd::map::lemma_map_insert_domain(blocks_before_insert, bid, self.blocks@[bid]);
                            vstd::set::lemma_set_insert_len(blocks_before_insert.dom(), bid);
                        }
                        assert(self.blocks@.dom() == blocks_before_insert.dom().insert(bid));
                        assert(self.blocks@.dom().len() == blocks_before_insert.dom().len());
                        assert(block_token_bound(self)) by {
                            assert forall|b2: BlockId|
                                #![auto]
                                self.blocks@.contains_key(b2)
                                implies self.blocks@[b2].tokens@.len() <= BLOCK_SIZE_SPEC as int
                            by {
                                if b2 == bid {
                                } else {
                                    assert(blocks_before_insert.contains_key(b2));
                                    assert(self.blocks@[b2] == blocks_before_insert[b2]);
                                }
                            }
                        }
                        assert(blocks_dom_in_range(self)) by {
                            assert forall|b2: BlockId|
                                #![auto]
                                self.blocks@.contains_key(b2)
                                implies b2 < self.num_blocks
                            by {
                                if b2 == bid {
                                } else {
                                    assert(blocks_before_insert.contains_key(b2));
                                }
                            }
                        }
                        assert(block_count_valid(self));
                        assert forall|b: BlockId|
                            #![trigger self.blocks@.contains_key(b)]
                            self.blocks@.contains_key(b)
                            implies old(self).blocks@.contains_key(b)
                        by {
                            if b == bid {
                            } else {
                                assert(blocks_before_insert.contains_key(b));
                            }
                        }
                        assert(new_refcount as int + 1 == e.refcount as int);
                        assert forall|b: BlockId|
                            #![trigger self.blocks@.contains_key(b)]
                            old(self).blocks@.contains_key(b)
                            && !self.blocks@.contains_key(b)
                            implies residency.block_ids@.contains(b)
                        by {
                            assert(!blocks_before_insert.contains_key(b));
                        }
                        assert forall|b: BlockId|
                            #![trigger self.blocks@.contains_key(b)]
                            old(self).blocks@.contains_key(b)
                            && self.blocks@.contains_key(b)
                            && !residency.block_ids@.contains(b)
                            implies self.blocks@[b] == old(self).blocks@[b]
                        by {
                            assert(b != bid);
                            assert(blocks_before_insert.contains_key(b));
                            assert(self.blocks@[b] == blocks_before_insert[b]);
                        }
                        assert forall|j: int|
                            #![trigger self.blocks@.contains_key(residency.block_ids@[j])]
                            0 <= j < k as int + 1
                            && self.blocks@.contains_key(residency.block_ids@[j])
                            implies self.blocks@[residency.block_ids@[j]].tokens@
                                    == old(self).blocks@[residency.block_ids@[j]].tokens@
                                && self.blocks@[residency.block_ids@[j]].hash_value
                                    == old(self).blocks@[residency.block_ids@[j]].hash_value
                                && self.blocks@[residency.block_ids@[j]].prefix_depth
                                    == old(self).blocks@[residency.block_ids@[j]].prefix_depth
                                && self.blocks@[residency.block_ids@[j]].parent_block
                                    == old(self).blocks@[residency.block_ids@[j]].parent_block
                                && self.blocks@[residency.block_ids@[j]].refcount as int + 1
                                    == old(self).blocks@[residency.block_ids@[j]].refcount as int
                        by {
                            if j == k as int {
                                assert(residency.block_ids@[j] == bid);
                                assert(self.blocks@[bid].refcount == new_refcount);
                            } else {
                                assert(0 <= j < k as int);
                                assert(residency.block_ids@[j] != bid);
                                assert(blocks_before_insert.contains_key(residency.block_ids@[j]));
                                assert(self.blocks@[residency.block_ids@[j]]
                                    == blocks_before_insert[residency.block_ids@[j]]);
                            }
                        }
                    }
                    assert forall|j: int|
                        #![trigger self.blocks@.contains_key(residency.block_ids@[j])]
                        k as int + 1 <= j < residency.block_ids@.len()
                        implies self.blocks@.contains_key(residency.block_ids@[j])
                            && self.blocks@[residency.block_ids@[j]]
                                == old(self).blocks@[residency.block_ids@[j]]
                    by {
                        assert(j != k as int);
                        assert(residency.block_ids@[j] != bid);
                        assert(blocks_at_iteration.contains_key(
                            residency.block_ids@[j]));
                        assert(blocks_at_iteration[residency.block_ids@[j]]
                            == old(self).blocks@[residency.block_ids@[j]]);
                        assert(self.blocks@[residency.block_ids@[j]]
                            == blocks_at_iteration[residency.block_ids@[j]]);
                    }
                    assert forall|j: int| 0 <= j < k as int + 1
                        && self.blocks@.contains_key(#[trigger] residency.block_ids@[j])
                        && self.blocks@[residency.block_ids@[j]].refcount == 0
                        implies self.blocks@[residency.block_ids@[j]].prefix_depth > 0
                    by {
                        if j == k as int {
                            assert(residency.block_ids@[j] == bid);
                            assert(self.blocks@[bid].prefix_depth == e.prefix_depth);
                            if e.prefix_depth == 0 {
                                assert(!self.blocks@.contains_key(bid));
                            }
                        } else {
                            assert(0 <= j < k as int);
                        }
                    }
                } else {
                    assert(false);
                }
                let ghost k_before = k;
                assert(bid == residency.block_ids@[k_before as int]);
                assert forall|j: int|
                    #![trigger self.blocks@.contains_key(residency.block_ids@[j])]
                    k_before as int + 1 <= j < residency.block_ids@.len()
                    implies self.blocks@.contains_key(residency.block_ids@[j])
                        && self.blocks@[residency.block_ids@[j]]
                            == old(self).blocks@[residency.block_ids@[j]]
                by {
                    assert((k_before as int) < j);
                    assert(residency.block_ids@[j] != bid) by {
                        if residency.block_ids@[j] == bid {
                            assert(residency.block_ids@[j]
                                == residency.block_ids@[k_before as int]);
                            assert(false);
                        }
                    }
                }
                k += 1;
                assert(k == k_before + 1);
                assert(k as int == k_before as int + 1);
                assert forall|j: int|
                    #![trigger self.blocks@.contains_key(residency.block_ids@[j])]
                    k as int <= j < residency.block_ids@.len()
                    implies self.blocks@.contains_key(residency.block_ids@[j])
                        && self.blocks@[residency.block_ids@[j]]
                            == old(self).blocks@[residency.block_ids@[j]]
                by {
                    assert(k_before as int + 1 <= j);
                    assert((k_before as int) < j);
                    assert(residency.block_ids@[j] != bid) by {
                        if residency.block_ids@[j] == bid {
                            assert(residency.block_ids@[j]
                                == residency.block_ids@[k_before as int]);
                            assert(false);
                        }
                    }
                    assert(self.blocks@.contains_key(residency.block_ids@[j]));
                    assert(self.blocks@[residency.block_ids@[j]]
                        == old(self).blocks@[residency.block_ids@[j]]);
                }
                assert forall|j: int|
                    #![trigger self.blocks@.contains_key(residency.block_ids@[j])]
                    0 <= j < k as int
                    && self.blocks@.contains_key(residency.block_ids@[j])
                    implies self.blocks@[residency.block_ids@[j]].tokens@
                            == old(self).blocks@[residency.block_ids@[j]].tokens@
                        && self.blocks@[residency.block_ids@[j]].hash_value
                            == old(self).blocks@[residency.block_ids@[j]].hash_value
                        && self.blocks@[residency.block_ids@[j]].prefix_depth
                            == old(self).blocks@[residency.block_ids@[j]].prefix_depth
                        && self.blocks@[residency.block_ids@[j]].parent_block
                            == old(self).blocks@[residency.block_ids@[j]].parent_block
                        && self.blocks@[residency.block_ids@[j]].refcount as int + 1
                            == old(self).blocks@[residency.block_ids@[j]].refcount as int
                by {
                    assert(j <= k as int - 1);
                }
            }
            assert(block_token_bound(self));
            assert(hash_to_block_in_range(self));
            assert(hash_to_block_consistent(self));
            assert(blocks_dom_in_range(self));
            assert(block_count_valid(self));
            assert forall|b: BlockId|
                #![trigger self.blocks@.contains_key(b)]
                self.blocks@.contains_key(b)
                implies old(self).blocks@.contains_key(b)
            by {
                assert(residency.block_ids@ == dealloc_block_ids);
            }
            assert forall|b: BlockId|
                #![trigger self.blocks@.contains_key(b)]
                old(self).blocks@.contains_key(b)
                && !self.blocks@.contains_key(b)
                implies dealloc_block_ids.contains(b)
            by {
                assert(residency.block_ids@ == dealloc_block_ids);
            }
            assert forall|j: int, other: RequestId|
                #![trigger old(self).request_residency@[other].block_ids@.contains(dealloc_block_ids[j])]
                0 <= j < dealloc_block_ids.len()
                && !self.blocks@.contains_key(dealloc_block_ids[j])
                && other != rid
                && old(self).request_residency@.contains_key(other)
                implies !old(self).request_residency@[other].block_ids@.contains(dealloc_block_ids[j])
            by {
                assert(residency.block_ids@ == dealloc_block_ids);
            }
            assert forall|b: BlockId|
                #![trigger self.blocks@.contains_key(b)]
                old(self).blocks@.contains_key(b)
                && self.blocks@.contains_key(b)
                && !dealloc_block_ids.contains(b)
                implies self.blocks@[b] == old(self).blocks@[b]
            by {
                assert(residency.block_ids@ == dealloc_block_ids);
            }
            assert forall|j: int|
                #![trigger self.blocks@.contains_key(dealloc_block_ids[j])]
                0 <= j < dealloc_block_ids.len()
                && self.blocks@.contains_key(dealloc_block_ids[j])
                implies self.blocks@[dealloc_block_ids[j]].refcount as int + 1
                    == old(self).blocks@[dealloc_block_ids[j]].refcount as int
            by {
                assert(residency.block_ids@ == dealloc_block_ids);
            }
            assert forall|j: int|
                #![trigger self.blocks@.contains_key(dealloc_block_ids[j])]
                0 <= j < dealloc_block_ids.len()
                && self.blocks@.contains_key(dealloc_block_ids[j])
                implies self.blocks@[dealloc_block_ids[j]].tokens@
                        == old(self).blocks@[dealloc_block_ids[j]].tokens@
                    && self.blocks@[dealloc_block_ids[j]].hash_value
                        == old(self).blocks@[dealloc_block_ids[j]].hash_value
                    && self.blocks@[dealloc_block_ids[j]].prefix_depth
                        == old(self).blocks@[dealloc_block_ids[j]].prefix_depth
                    && self.blocks@[dealloc_block_ids[j]].parent_block
                        == old(self).blocks@[dealloc_block_ids[j]].parent_block
            by {
                assert(residency.block_ids@ == dealloc_block_ids);
            }
            let ghost request_residency_before_remove = self.request_residency@;
            let removed = self.request_residency.remove(&rid);
            assert(removed is Some);
            proof {
                vstd::map::lemma_map_remove_domain(request_residency_before_remove, rid);
            }
            assert(self.request_residency@.dom()
                == request_residency_before_remove.dom().remove(rid));
            assert(request_residency_before_remove == old(self).request_residency@);
            assert(self.request_residency@.dom()
                == old(self).request_residency@.dom().remove(rid));

            assert(release_ids@ == residency.block_ids@);
        } else {
            assert(!old(self).request_residency@.contains_key(rid));
            assert(cs_valid(self));
            assert(self.blocks@ == old(self).blocks@);
            assert(self.hash_to_block@ == old(self).hash_to_block@);
            proof {
                lemma_registry_entries_from_pre_refl(old(self));
                lemma_positive_provenance_metadata_frame_blocks_eq(
                    old(self), self,
                );
                lemma_positive_provenance_origin_blocks_eq(old(self), self);
                assert(registry_entries_from_pre(
                    old(self), self, Seq::<u64>::empty(),
                )) by {
                    reveal(registry_entries_from_pre);
                }
            }
            return;
        }
        assert(self.config == old(self).config);
        assert(!self.request_residency@.contains_key(rid));
        assert(block_token_bound(self));
        assert(hash_to_block_in_range(self));
        assert(hash_to_block_consistent(self));
        assert(blocks_dom_in_range(self));
        assert(block_count_valid(self));
        assert forall|other: RequestId|
            #![auto]
            other != rid && old(self).request_residency@.contains_key(other)
            implies self.request_residency@.contains_key(other)
                && self.request_residency@[other] == old(self).request_residency@[other]
        by {
        }
        assert(live_covers_queue(self));
        assert(queue_disjoint(self));
        assert(running_unique(self));
        assert(waiting_unique(self));
        assert(running_has_residency(self)) by {
            assert forall|r: RequestId|
                #![auto]
                self.running@.contains(r)
                implies self.request_residency@.contains_key(r)
            by {
                assert(old(self).running@.contains(r));
                assert(r != rid);
                assert(running_has_residency(old(self)));
            }
        }
        assert(waiting_has_no_residency(self)) by {
            assert forall|w: RequestId|
                #![auto]
                self.waiting@.contains(w)
                implies !self.request_residency@.contains_key(w)
            by {
                assert(old(self).waiting@.contains(w));
                assert(w != rid);
                assert(waiting_has_no_residency(old(self)));
            }
        }
        assert(residency_block_ids_unique(self)) by {
            assert forall|r: RequestId|
                #![auto]
                self.request_residency@.contains_key(r)
                implies self.request_residency@[r].block_ids@.no_duplicates()
            by {
                assert(r != rid);
                assert(old(self).request_residency@.contains_key(r));
                assert(self.request_residency@[r] == old(self).request_residency@[r]);
                assert(residency_block_ids_unique(old(self)));
            }
        }
        assert(residency_blocks_in_range(self)) by {
            assert forall|r: RequestId, idx: int|
                #![trigger self.request_residency@[r].block_ids@[idx]]
                self.request_residency@.contains_key(r)
                && 0 <= idx < self.request_residency@[r].block_ids@.len()
                implies self.blocks@.contains_key(self.request_residency@[r].block_ids@[idx])
            by {
                assert(r != rid);
                assert(old(self).request_residency@.contains_key(r));
                assert(self.request_residency@[r] == old(self).request_residency@[r]);
                let b = self.request_residency@[r].block_ids@[idx];
                assert(old(self).request_residency@[r].block_ids@[idx] == b);
                assert(residency_blocks_in_range(old(self)));
                assert(old(self).blocks@.contains_key(b));
                if !self.blocks@.contains_key(b) {
                    assert(dealloc_block_ids.contains(b));
                    let j = dealloc_block_ids.index_of(b);
                    assert(0 <= j < dealloc_block_ids.len());
                    assert(dealloc_block_ids[j] == b);
                    assert(old(self).request_residency@[r].block_ids@.contains(dealloc_block_ids[j]));
                    assert(!old(self).request_residency@[r].block_ids@.contains(dealloc_block_ids[j]));
                }
            }
        }
        assert(refcount_valid(self)) by {
            assert forall|bid: BlockId|
                #[trigger] self.blocks@.contains_key(bid)
                implies self.blocks@[bid].refcount as int
                    == residency_holders_of(self, bid).len() as int
            by {
                assert(old(self).blocks@.contains_key(bid)) by {
                    if !old(self).blocks@.contains_key(bid) {
                        assert(!self.blocks@.contains_key(bid));
                    }
                }
                let old_holders = residency_holders_of(old(self), bid);
                let final_holders = residency_holders_of(self, bid);
                assert_sets_equal!(final_holders == old_holders.remove(rid), r: RequestId => {
                    if final_holders.contains(r) {
                        assert(self.request_residency@.contains_key(r));
                        assert(r != rid);
                        assert(old(self).request_residency@.contains_key(r));
                        assert(self.request_residency@[r] == old(self).request_residency@[r]);
                    }
                    if old_holders.remove(rid).contains(r) {
                        assert(r != rid);
                        assert(old_holders.contains(r));
                        assert(old(self).request_residency@.contains_key(r));
                        assert(self.request_residency@.contains_key(r));
                        assert(self.request_residency@[r] == old(self).request_residency@[r]);
                    }
                });
                vstd::set::lemma_set_remove_len(old_holders, rid);
                assert(refcount_valid(old(self)));
                assert(old(self).blocks@[bid].refcount as int == old_holders.len() as int);
                if dealloc_block_ids.contains(bid) {
                    let j = dealloc_block_ids.index_of(bid);
                    assert(0 <= j < dealloc_block_ids.len());
                    assert(dealloc_block_ids[j] == bid);
                    assert(old_holders.contains(rid)) by {
                        assert(old(self).request_residency@.contains_key(rid));
                        assert(old(self).request_residency@[rid].block_ids@ == dealloc_block_ids);
                    }
                    assert(self.blocks@[bid].refcount as int + 1
                        == old(self).blocks@[bid].refcount as int);
                    assert(old_holders.len() as int == final_holders.len() as int + 1);
                } else {
                    assert(!old_holders.contains(rid)) by {
                        if old_holders.contains(rid) {
                            assert(old(self).request_residency@[rid].block_ids@.contains(bid));
                            assert(old(self).request_residency@[rid].block_ids@ == dealloc_block_ids);
                        }
                    }
                    assert(self.blocks@[bid] == old(self).blocks@[bid]);
                    assert(old_holders.len() == final_holders.len());
                }
            }
        }
        // Alignment-transport exports: survivor tokens, untouched-block
        // verbatim, pool growth.
        assert forall|b: BlockId| #[trigger] self.blocks@.contains_key(b)
            implies old(self).blocks@.contains_key(b)
                && self.blocks@[b].tokens@ == old(self).blocks@[b].tokens@
                && self.blocks@[b].prefix_depth == old(self).blocks@[b].prefix_depth
                && self.blocks@[b].parent_block == old(self).blocks@[b].parent_block
        by {
            if dealloc_block_ids.contains(b) {
                let j = dealloc_block_ids.index_of(b);
                assert(self.blocks@.contains_key(dealloc_block_ids[j]));
            }
        }
        assert forall|b: BlockId| #[trigger] self.blocks@.contains_key(b)
            && (old(self).request_residency@.contains_key(rid)
                ==> !old(self).request_residency@[rid].block_ids@.contains(b))
            implies self.blocks@[b] == old(self).blocks@[b]
        by {
            assert(!dealloc_block_ids.contains(b));
        }
        assert forall|b: BlockId|
            #[trigger] old(self).blocks@[b].prefix_depth > 0
            && old(self).blocks@.contains_key(b)
            && old(self).blocks@[b].prefix_depth > 0
            implies self.blocks@.contains_key(b)
                && self.blocks@[b].tokens@ == old(self).blocks@[b].tokens@
                && self.blocks@[b].hash_value == old(self).blocks@[b].hash_value
                && self.blocks@[b].prefix_depth == old(self).blocks@[b].prefix_depth
                && self.blocks@[b].parent_block == old(self).blocks@[b].parent_block
        by {
            if !self.blocks@.contains_key(b) {
                assert(old(self).blocks@[b].prefix_depth == 0);
                assert(false);
            }
            if dealloc_block_ids.contains(b) {
                let j = dealloc_block_ids.index_of(b);
                assert(0 <= j < dealloc_block_ids.len());
                assert(dealloc_block_ids[j] == b);
                assert(self.blocks@[b].tokens@
                    == old(self).blocks@[b].tokens@);
                assert(self.blocks@[b].hash_value
                    == old(self).blocks@[b].hash_value);
                assert(self.blocks@[b].prefix_depth
                    == old(self).blocks@[b].prefix_depth);
                assert(self.blocks@[b].parent_block
                    == old(self).blocks@[b].parent_block);
            }
        }
        assert(positive_provenance_metadata_frame(old(self), self)) by {
            reveal(positive_provenance_metadata_frame);
        }
        proof {
            lemma_positive_provenance_origin_from_metadata_frame(
                old(self), self,
            );
        }
        assert(persistent_provenance_closed(self)) by {
            assert forall|child: BlockId|
                #[trigger] self.blocks@.contains_key(child)
                && self.blocks@[child].prefix_depth > 1
                implies self.blocks@[child].parent_block is Some
                    && self.blocks@.contains_key(
                        self.blocks@[child].parent_block.unwrap(),
                    )
                    && self.blocks@[
                        self.blocks@[child].parent_block.unwrap()
                    ].prefix_depth + 1
                        == self.blocks@[child].prefix_depth
            by {
                assert(old(self).blocks@.contains_key(child));
                assert(self.blocks@[child].prefix_depth
                    == old(self).blocks@[child].prefix_depth);
                assert(self.blocks@[child].parent_block
                    == old(self).blocks@[child].parent_block);
                let parent = self.blocks@[child].parent_block.unwrap();
                reveal(persistent_provenance_closed);
                assert(old(self).blocks@.contains_key(parent));
                assert(old(self).blocks@[parent].prefix_depth + 1
                    == old(self).blocks@[child].prefix_depth);
                assert(old(self).blocks@[parent].prefix_depth > 0);
                assert(positive_provenance_metadata_frame(old(self), self));
                assert(self.blocks@.contains_key(parent));
                assert(self.blocks@[parent].prefix_depth
                    == old(self).blocks@[parent].prefix_depth);
            }
            lemma_physical_parent_closure_from_metadata_frame(old(self), self);
            lemma_persistent_provenance_closed_frame(old(self), self);
        }
        assert(self.free_blocks >= old(self).free_blocks) by {
            assert(block_count_valid(old(self)));
            assert(block_count_valid(self));
            assert(self.blocks@.dom().subset_of(old(self).blocks@.dom())) by {
                assert forall|b: BlockId| self.blocks@.dom().contains(b)
                    implies old(self).blocks@.dom().contains(b) by {
                    assert(self.blocks@.contains_key(b));
                }
            }
            vstd::set_lib::lemma_len_subset(self.blocks@.dom(), old(self).blocks@.dom());
        }
        assert(hash_to_block_no_zero(self)) by {
            if self.hash_to_block@.contains_key(0u64) {
                assert(old(self).hash_to_block@.contains_key(0u64));
            }
        }
        assert(registered_provenance_aligned(self)) by {
            assert forall|r: RequestId, j: int|
                #![trigger self.blocks@[self.request_residency@[r]
                    .block_ids@[j]].prefix_depth]
                self.request_residency@.contains_key(r)
                && 0 <= j < self.request_residency@[r].block_ids@.len()
                && self.blocks@.contains_key(self.request_residency@[r].block_ids@[j])
                && self.blocks@[self.request_residency@[r]
                    .block_ids@[j]].prefix_depth > 0
                implies {
                    let ids = self.request_residency@[r].block_ids@;
                    let bid = ids[j];
                    &&& self.blocks@[bid].prefix_depth as int == j + 1
                    &&& self.blocks@[bid].parent_block
                        == if j == 0 { None } else { Some(ids[j - 1]) }
                }
            by {
                assert(r != rid);
                assert(old(self).request_residency@.contains_key(r));
                assert(self.request_residency@[r]
                    == old(self).request_residency@[r]);
                let bid = self.request_residency@[r].block_ids@[j];
                assert(old(self).blocks@.contains_key(bid));
                assert(self.blocks@[bid].prefix_depth
                    == old(self).blocks@[bid].prefix_depth);
                assert(self.blocks@[bid].parent_block
                    == old(self).blocks@[bid].parent_block);
                assert(registered_provenance_aligned(old(self)));
            }
        }
        assert(registry_entries_from_pre(
            old(self), self, Seq::<u64>::empty(),
        )) by {
            reveal(registry_entries_from_pre);
            assert forall|h: u64|
                #[trigger] self.hash_to_block@.contains_key(h)
                && !Seq::<u64>::empty().contains(h)
                implies {
                    let bid = self.hash_to_block@[h];
                    &&& old(self).hash_to_block@.contains_key(h)
                    &&& old(self).hash_to_block@[h] == bid
                    &&& old(self).blocks@.contains_key(bid)
                    &&& self.blocks@.contains_key(bid)
                    &&& self.blocks@[bid].tokens@
                        == old(self).blocks@[bid].tokens@
                    &&& self.blocks@[bid].hash_value
                        == old(self).blocks@[bid].hash_value
                    &&& self.blocks@[bid].prefix_depth
                        == old(self).blocks@[bid].prefix_depth
                    &&& self.blocks@[bid].parent_block
                        == old(self).blocks@[bid].parent_block
                }
            by {
                let bid = self.hash_to_block@[h];
                assert(old(self).hash_to_block@.contains_key(h));
                assert(self.hash_to_block@[h]
                    == old(self).hash_to_block@[h]);
                assert(hash_to_block_consistent(old(self)));
                assert(old(self).blocks@[bid].prefix_depth > 0);
            }
        }
        assert(cs_valid(self));
        assert forall|j: int| 0 <= j < release_ids@.len()
            && self.blocks@.contains_key(#[trigger] release_ids@[j])
            && self.blocks@[release_ids@[j]].refcount == 0
            implies self.blocks@[release_ids@[j]].prefix_depth > 0
        by {
            if old(self).request_residency@.contains_key(rid) {
                assert(residency_opt is Some);
            } else {
                assert(release_ids@.len() == 0);
            }
        }
        assert forall|j: int| 0 <= j < release_ids@.len() implies {
            let bid = #[trigger] release_ids@[j];
            &&& dealloc_pre.blocks@.contains_key(bid)
            &&& dealloc_pre.blocks@[bid].refcount > 0
            &&& bid < dealloc_pre.num_blocks
            &&& (self.blocks@.contains_key(bid) ==> {
                &&& self.blocks@[bid].tokens@ == dealloc_pre.blocks@[bid].tokens@
                &&& self.blocks@[bid].refcount as int + 1
                    == dealloc_pre.blocks@[bid].refcount as int
                &&& self.blocks@[bid].hash_value == dealloc_pre.blocks@[bid].hash_value
                &&& self.blocks@[bid].prefix_depth == dealloc_pre.blocks@[bid].prefix_depth
                &&& self.blocks@[bid].parent_block == dealloc_pre.blocks@[bid].parent_block
            })
        }
        by {
            assert(dealloc_pre.blocks@ == old(self).blocks@);
            assert(dealloc_pre.num_blocks == old(self).num_blocks);
            if old(self).request_residency@.contains_key(rid) {
                assert(residency_opt is Some);
                assert(release_ids@ == old(self).request_residency@[rid].block_ids@);
                assert(residency_blocks_in_range(old(self)));
                assert(refcount_valid(old(self)));
                let bid = release_ids@[j];
                assert(old(self).blocks@.contains_key(bid));
                assert(old(self).blocks@[bid].refcount > 0) by {
                    let holders = residency_holders_of(old(self), bid);
                    assert(holders.contains(rid));
                    assert(holders.len() > 0);
                }
                if self.blocks@.contains_key(bid) {
                    assert(self.blocks@[bid].refcount as int + 1
                        == old(self).blocks@[bid].refcount as int);
                    assert(self.blocks@[bid].tokens@
                        == old(self).blocks@[bid].tokens@);
                    assert(self.blocks@[bid].hash_value
                        == old(self).blocks@[bid].hash_value);
                    assert(self.blocks@[bid].prefix_depth
                        == old(self).blocks@[bid].prefix_depth);
                    assert(self.blocks@[bid].parent_block
                        == old(self).blocks@[bid].parent_block);
                }
            } else {
                assert(release_ids@.len() == 0);
            }
        }
        assert forall|j: int| 0 <= j < release_ids@.len()
            && dealloc_pre.blocks@[#[trigger] release_ids@[j]].prefix_depth > 0
            implies dealloc_pre.blocks@[release_ids@[j]].prefix_depth as int
                    == j + 1
                && dealloc_pre.blocks@[release_ids@[j]].parent_block
                    == if j == 0 {
                        None
                    } else {
                        Some(release_ids@[j - 1])
                    }
        by {
            assert(dealloc_pre.blocks@ == old(self).blocks@);
            if old(self).request_residency@.contains_key(rid) {
                assert(release_ids@
                    == old(self).request_residency@[rid].block_ids@);
                assert(registered_provenance_aligned(old(self)));
            } else {
                assert(release_ids@.len() == 0);
            }
        }
        let ghost semantic_post = *self;
        proof { lemma_free_queue_valid_to_token(&dealloc_pre); }
        self.reconcile_deallocated_queue(&release_ids, Ghost(dealloc_pre));
        assert(positive_provenance_metadata_frame(old(self), self)) by {
            lemma_positive_provenance_metadata_frame_blocks_eq(
                &semantic_post, self,
            );
            lemma_positive_provenance_metadata_frame_transitive(
                old(self), &semantic_post, self,
            );
        }
        proof {
            lemma_positive_provenance_origin_from_metadata_frame(
                old(self), self,
            );
        }
        assert(registry_entries_from_pre(
            old(self), self, Seq::<u64>::empty(),
        )) by {
            reveal(registry_entries_from_pre);
            assert forall|h: u64|
                #[trigger] self.hash_to_block@.contains_key(h)
                && !Seq::<u64>::empty().contains(h)
                implies {
                    let bid = self.hash_to_block@[h];
                    &&& old(self).hash_to_block@.contains_key(h)
                    &&& old(self).hash_to_block@[h] == bid
                    &&& old(self).blocks@.contains_key(bid)
                    &&& self.blocks@.contains_key(bid)
                    &&& self.blocks@[bid].tokens@
                        == old(self).blocks@[bid].tokens@
                    &&& self.blocks@[bid].hash_value
                        == old(self).blocks@[bid].hash_value
                    &&& self.blocks@[bid].prefix_depth
                        == old(self).blocks@[bid].prefix_depth
                    &&& self.blocks@[bid].parent_block
                        == old(self).blocks@[bid].parent_block
                }
            by {
                let bid = self.hash_to_block@[h];
                assert(semantic_post.hash_to_block@.contains_key(h));
                assert(self.hash_to_block@[h]
                    == semantic_post.hash_to_block@[h]);
                assert(registry_entries_from_pre(
                    old(self), &semantic_post, Seq::<u64>::empty(),
                ));
            }
        }
    }
}

} // verus!
