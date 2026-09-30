// Verified token append and fresh-tail allocation transitions.

use super::*;

verus! {
impl CacheScheduler {
    pub fn append_token_tail_append(&mut self, rid: RequestId, token: TokenId)
        requires
            cs_valid(old(self)),
            free_queue_valid(old(self)),
            persistent_provenance_closed(old(self)),
            old(self).request_residency@.contains_key(rid),
            old(self).request_residency@[rid].block_ids@.len() > 0,
            ({
                let old_residency = old(self).request_residency@[rid];
                let last_bid = old_residency.block_ids@[old_residency.block_ids@.len() - 1];
                old(self).blocks@.contains_key(last_bid)
                    && old(self).blocks@[last_bid].tokens@.len() < BLOCK_SIZE_SPEC as int
            }),
            old(self).num_blocks <= u64::MAX / BLOCK_SIZE,
        ensures
            cs_valid(final(self)),
            free_queue_valid(final(self)),
            persistent_provenance_closed(final(self)),
            positive_provenance_metadata_frame(old(self), final(self)),
            positive_provenance_origin(old(self), final(self)),
            append_token_tail_append_success(old(self), final(self), rid, token),
            registry_entries_from_pre(
                old(self), final(self), Seq::<u64>::empty(),
            ),
            final(self).accepted_requests@ == old(self).accepted_requests@,
            final(self).request_residency@.dom() == old(self).request_residency@.dom(),
    {
        let residency = match self.request_residency.get(&rid) {
            Some(residency_ref) => residency_ref.clone(),
            None => {
                proof { assert(false); }
                return;
            },
        };
        let last_index = residency.block_ids.len() - 1;
        let last_bid = residency.block_ids[last_index];
        let entry = match self.blocks.get(&last_bid) {
            Some(entry_ref) => entry_ref.clone(),
            None => {
                proof { assert(false); }
                return;
            },
        };
        let mut next_tokens = entry.tokens.clone();
        let offset = next_tokens.len();
        assert(offset as int == entry.tokens@.len());
        assert(offset < BLOCK_SIZE as usize);
        next_tokens.push(token);
        assert(next_tokens@ == entry.tokens@.push(token));

        assert(last_bid < self.num_blocks) by {
            assert(residency_blocks_in_range(self));
            assert(self.request_residency@.contains_key(rid));
            assert(self.request_residency@[rid].block_ids@ == residency.block_ids@);
            assert(residency.block_ids@[last_index as int] == last_bid);
            assert(self.blocks@.contains_key(last_bid));
            assert(blocks_dom_in_range(self));
        }
        assert(last_bid < u64::MAX / BLOCK_SIZE);
        let slot = last_bid * BLOCK_SIZE + offset as u64;

        let ghost blocks_before_update = self.blocks@;
        self.blocks.insert(last_bid, BlockEntry {
            tokens: next_tokens,
            refcount: entry.refcount,
            hash_value: entry.hash_value,
            prefix_depth: entry.prefix_depth,
            parent_block: entry.parent_block,
        });
        proof {
            vstd::map::lemma_map_insert_domain(
                blocks_before_update, last_bid, self.blocks@[last_bid]);
        }
        assert(self.blocks@.dom() == blocks_before_update.dom());
        assert(self.blocks@[last_bid].tokens@ == entry.tokens@.push(token));
        assert(self.blocks@[last_bid].refcount == entry.refcount);
        assert(self.blocks@[last_bid].hash_value == entry.hash_value);
        assert(self.blocks@[last_bid].prefix_depth == entry.prefix_depth);
        assert(self.blocks@[last_bid].parent_block == entry.parent_block);
        assert forall|bid: BlockId|
            bid != last_bid && #[trigger] blocks_before_update.contains_key(bid)
            implies self.blocks@.contains_key(bid) && self.blocks@[bid] == blocks_before_update[bid]
        by {
        }

        let mut slot_mapping: Vec<SlotId> = Vec::new();
        slot_mapping.push(slot);
        assert(slot_mapping@.len() == 1);
        assert(slot_mapping@[0] == slot);

        let mut next_residency = residency.clone();
        next_residency.slot_mapping = slot_mapping;
        let ghost residency_before_update = self.request_residency@;
        self.request_residency.insert(rid, next_residency);
        proof {
            vstd::map::lemma_map_insert_domain(
                residency_before_update, rid, self.request_residency@[rid]);
        }
        assert(self.request_residency@.dom() == residency_before_update.dom());
        assert(self.request_residency@[rid].block_ids@ == residency.block_ids@);
        assert(self.request_residency@[rid].cached_prefix_blocks == residency.cached_prefix_blocks);
        assert(self.request_residency@[rid].slot_mapping@.len() == 1);
        assert(self.request_residency@[rid].slot_mapping@[0] == slot);

        assert(live_covers_queue(self));
        assert(queue_disjoint(self));
        assert(running_unique(self));
        assert(waiting_unique(self));
        assert(running_has_residency(self)) by {
            assert forall|r: RequestId|
                #[trigger] self.running@.contains(r)
                implies self.request_residency@.contains_key(r)
            by {
                assert(running_has_residency(old(self)));
                assert(old(self).request_residency@.contains_key(r));
            }
        }
        assert(waiting_has_no_residency(self)) by {
            assert forall|r: RequestId|
                #[trigger] self.waiting@.contains(r)
                implies !self.request_residency@.contains_key(r)
            by {
                assert(waiting_has_no_residency(old(self)));
                assert(!old(self).request_residency@.contains_key(r));
            }
        }
        assert(residency_blocks_in_range(self)) by {
            assert forall|r: RequestId, k: int|
                #![trigger self.request_residency@[r].block_ids@[k]]
                self.request_residency@.contains_key(r)
                && 0 <= k < self.request_residency@[r].block_ids@.len()
                implies self.blocks@.contains_key(self.request_residency@[r].block_ids@[k])
            by {
                let bid = self.request_residency@[r].block_ids@[k];
                assert(old(self).request_residency@.contains_key(r));
                if r == rid {
                    assert(self.request_residency@[r].block_ids@ == old(self).request_residency@[r].block_ids@);
                } else {
                    assert(self.request_residency@[r] == old(self).request_residency@[r]);
                }
                assert(residency_blocks_in_range(old(self)));
                assert(old(self).blocks@.contains_key(bid));
            }
        }
        assert(block_token_bound(self)) by {
            assert forall|bid: BlockId|
                #[trigger] self.blocks@.contains_key(bid)
                implies self.blocks@[bid].tokens@.len() <= BLOCK_SIZE_SPEC as int
            by {
                assert(old(self).blocks@.contains_key(bid));
                if bid == last_bid {
                    assert(self.blocks@[bid].tokens@.len() == entry.tokens@.len() + 1);
                    assert(entry.tokens@.len() < BLOCK_SIZE_SPEC as int);
                } else {
                    assert(self.blocks@[bid] == old(self).blocks@[bid]);
                    assert(block_token_bound(old(self)));
                }
            }
        }
        assert(hash_to_block_in_range(self)) by {
            assert forall|h: u64|
                #[trigger] self.hash_to_block@.contains_key(h)
                implies self.blocks@.contains_key(self.hash_to_block@[h])
                    && self.blocks@[self.hash_to_block@[h]].tokens@.len()
                        == BLOCK_SIZE_SPEC as int
            by {
                let bid = self.hash_to_block@[h];
                assert(hash_to_block_in_range(old(self)));
                assert(old(self).blocks@.contains_key(bid));
                if bid == last_bid {
                    assert(old(self).blocks@[bid].tokens@.len() == BLOCK_SIZE_SPEC as int);
                    assert(entry.tokens@.len() < BLOCK_SIZE_SPEC as int);
                    assert(false);
                } else {
                    assert(self.blocks@[bid] == old(self).blocks@[bid]);
                }
            }
        }
        assert(residency_block_ids_unique(self)) by {
            assert forall|r: RequestId|
                #[trigger] self.request_residency@.contains_key(r)
                implies self.request_residency@[r].block_ids@.no_duplicates()
            by {
                assert(old(self).request_residency@.contains_key(r));
                if r == rid {
                    assert(self.request_residency@[r].block_ids@ == old(self).request_residency@[r].block_ids@);
                } else {
                    assert(self.request_residency@[r] == old(self).request_residency@[r]);
                }
                assert(residency_block_ids_unique(old(self)));
            }
        }
        assert(refcount_valid(self)) by {
            assert forall|bid: BlockId|
                #[trigger] self.blocks@.contains_key(bid)
                implies self.blocks@[bid].refcount as int
                    == residency_holders_of(self, bid).len() as int
            by {
                assert(old(self).blocks@.contains_key(bid));
                let holders = residency_holders_of(self, bid);
                let old_holders = residency_holders_of(old(self), bid);
                assert_sets_equal!(holders == old_holders, r: RequestId => {
                    if holders.contains(r) {
                        assert(self.request_residency@.contains_key(r));
                        assert(old(self).request_residency@.contains_key(r));
                        if r == rid {
                            assert(self.request_residency@[r].block_ids@ == old(self).request_residency@[r].block_ids@);
                        } else {
                            assert(self.request_residency@[r] == old(self).request_residency@[r]);
                        }
                    }
                    if old_holders.contains(r) {
                        assert(old(self).request_residency@.contains_key(r));
                        assert(self.request_residency@.contains_key(r));
                        if r == rid {
                            assert(self.request_residency@[r].block_ids@ == old(self).request_residency@[r].block_ids@);
                        } else {
                            assert(self.request_residency@[r] == old(self).request_residency@[r]);
                        }
                    }
                });
                assert(refcount_valid(old(self)));
                if bid == last_bid {
                    assert(self.blocks@[bid].refcount == entry.refcount);
                    assert(entry.refcount == old(self).blocks@[bid].refcount);
                } else {
                    assert(self.blocks@[bid] == old(self).blocks@[bid]);
                }
            }
        }
        assert(hash_to_block_consistent(self)) by {
            assert forall|h: u64|
                #[trigger] self.hash_to_block@.contains_key(h)
                implies self.blocks@[self.hash_to_block@[h]].hash_value == h
                    && self.blocks@[self.hash_to_block@[h]].prefix_depth > 0
            by {
                let bid = self.hash_to_block@[h];
                assert(hash_to_block_consistent(old(self)));
                if bid == last_bid {
                    assert(self.blocks@[bid].hash_value == entry.hash_value);
                    assert(entry.hash_value == old(self).blocks@[bid].hash_value);
                } else {
                    assert(self.blocks@[bid] == old(self).blocks@[bid]);
                }
            }
        }
        assert(blocks_dom_in_range(self)) by {
            assert forall|bid: BlockId|
                #[trigger] self.blocks@.contains_key(bid)
                implies bid < self.num_blocks
            by {
                assert(old(self).blocks@.contains_key(bid));
                assert(blocks_dom_in_range(old(self)));
            }
        }
        assert(block_count_valid(self)) by {
            assert(block_count_valid(old(self)));
            assert(self.blocks@.dom().len() == old(self).blocks@.dom().len());
            assert(self.free_blocks == old(self).free_blocks);
        }
        assert(cs_valid(self));

        proof {
            let old_ids = old(self).request_residency@[rid].block_ids@;
            let old_last = old_ids[old_ids.len() - 1];
            assert(old_last == last_bid);
            assert(old(self).blocks@[last_bid].prefix_depth == 0) by {
                if old(self).blocks@[last_bid].prefix_depth > 0 {
                    reveal(persistent_provenance_closed);
                    assert(old(self).blocks@[last_bid].tokens@.len()
                        == BLOCK_SIZE_SPEC as int);
                    assert(entry.tokens@.len() < BLOCK_SIZE_SPEC as int);
                }
            }
            assert(self.blocks@[last_bid].refcount > 0) by {
                let holders = residency_holders_of(self, last_bid);
                assert(holders.contains(rid));
                assert(refcount_valid(self));
                if holders.len() == 0 {
                    vstd::set_lib::lemma_set_is_empty_len0(holders);
                }
            }
            assert(self.blocks@[last_bid].prefix_depth > 0
                && self.blocks@[last_bid].parent_block is Some
                ==> !self.free_queue.order@.contains(
                    self.blocks@[last_bid].parent_block.unwrap(),
                )) by {
                assert(self.blocks@[last_bid].prefix_depth == 0);
            }
            lemma_two_queue_valid_publish_active_page(
                old(self), self, last_bid,
            );
            assert forall|bid: BlockId|
                #[trigger] self.blocks@.contains_key(bid)
                && self.blocks@[bid].prefix_depth > 0
                implies old(self).blocks@.contains_key(bid)
                    && self.blocks@[bid].tokens@ == old(self).blocks@[bid].tokens@
                    && self.blocks@[bid].prefix_depth
                        == old(self).blocks@[bid].prefix_depth
                    && self.blocks@[bid].parent_block
                        == old(self).blocks@[bid].parent_block
            by {
                assert(bid != last_bid);
                assert(self.blocks@[bid] == old(self).blocks@[bid]);
            }
            assert forall|bid: BlockId|
                #[trigger] old(self).blocks@.contains_key(bid)
                && old(self).blocks@[bid].prefix_depth > 0
                implies self.blocks@.contains_key(bid)
                    && self.blocks@[bid].prefix_depth
                        == old(self).blocks@[bid].prefix_depth
            by {
                assert(bid != last_bid);
                assert(self.blocks@[bid] == old(self).blocks@[bid]);
            }
            assert(positive_provenance_metadata_frame(old(self), self)) by {
                reveal(positive_provenance_metadata_frame);
                assert forall|bid: BlockId|
                    #[trigger] old(self).blocks@[bid].prefix_depth > 0
                    && old(self).blocks@.contains_key(bid)
                    && old(self).blocks@[bid].prefix_depth > 0
                    implies self.blocks@.contains_key(bid)
                        && self.blocks@[bid] == old(self).blocks@[bid]
                by {
                    assert(bid != last_bid);
                }
            }
            lemma_positive_provenance_origin_from_metadata_frame(
                old(self), self,
            );
            lemma_physical_parent_closure_from_metadata_frame(old(self), self);
            lemma_persistent_provenance_closed_frame(old(self), self);
        }

        assert(append_token_common_frame(old(self), self, rid)) by {
            assert forall|other: RequestId|
                other != rid && #[trigger] old(self).request_residency@.contains_key(other)
                implies self.request_residency@.contains_key(other)
                    && self.request_residency@[other] == old(self).request_residency@[other]
            by {
            }
        }
        assert(append_token_tail_append_success(old(self), self, rid, token));
        proof {
            lemma_registry_entries_from_pre_tail_append(
                old(self), self, rid, token,
            );
        }
    }

    pub fn append_token_new_tail(&mut self, rid: RequestId, token: TokenId, new_bid: BlockId)
        requires
            cs_valid(old(self)),
            free_queue_valid(old(self)),
            persistent_provenance_closed(old(self)),
            old(self).request_residency@.contains_key(rid),
            old(self).request_residency@[rid].block_ids@.len() > 0,
            ({
                let old_residency = old(self).request_residency@[rid];
                let last_bid = old_residency.block_ids@[old_residency.block_ids@.len() - 1];
                old(self).blocks@.contains_key(last_bid)
                    && old(self).blocks@[last_bid].tokens@.len() == BLOCK_SIZE_SPEC as int
            }),
            !old(self).blocks@.contains_key(new_bid),
            new_bid < old(self).num_blocks,
            old(self).free_blocks > 0,
            old(self).free_queue.head == Some(new_bid),
            old(self).num_blocks <= u64::MAX / BLOCK_SIZE,
        ensures
            cs_valid(final(self)),
            free_queue_valid(final(self)),
            persistent_provenance_closed(final(self)),
            positive_provenance_metadata_frame(old(self), final(self)),
            positive_provenance_origin(old(self), final(self)),
            append_token_new_tail_success(old(self), final(self), rid, token),
            registry_entries_from_pre(
                old(self), final(self), Seq::<u64>::empty(),
            ),
            final(self).accepted_requests@ == old(self).accepted_requests@,
            final(self).request_residency@.dom() == old(self).request_residency@.dom(),
    {
        let popped = self.free_queue.pop_front();
        assert(popped == Some(new_bid));
        let ghost queue_after_pop = self.free_queue.order@;
        assert(queue_after_pop == old(self).free_queue.order@.subrange(
            1, old(self).free_queue.order@.len() as int,
        ));
        let residency = match self.request_residency.get(&rid) {
            Some(residency_ref) => residency_ref.clone(),
            None => {
                proof { assert(false); }
                return;
            },
        };
        let old_len = residency.block_ids.len();
        let last_index = old_len - 1;
        let last_bid = residency.block_ids[last_index];
        assert(self.blocks@.contains_key(last_bid));
        assert(self.blocks@[last_bid].tokens@.len() == BLOCK_SIZE_SPEC as int);

        let mut block_tokens: Vec<TokenId> = Vec::new();
        block_tokens.push(token);
        assert(block_tokens@ == seq![token]);

        let ghost blocks_before_insert = self.blocks@;
        assert(blocks_before_insert == old(self).blocks@);
        self.blocks.insert(new_bid, BlockEntry {
            tokens: block_tokens,
            refcount: 1,
            hash_value: 0,
            prefix_depth: 0,
            parent_block: None,
        });
        proof {
            vstd::map::lemma_map_insert_domain(
                blocks_before_insert, new_bid, self.blocks@[new_bid]);
            vstd::set::lemma_set_insert_len(blocks_before_insert.dom(), new_bid);
        }
        assert(self.blocks@.dom() == blocks_before_insert.dom().insert(new_bid));
        assert(self.blocks@.dom().len() == blocks_before_insert.dom().len() + 1);
        assert(self.blocks@[new_bid].tokens@ == seq![token]);
        assert(self.blocks@[new_bid].refcount == 1);
        assert(self.blocks@[new_bid].hash_value == 0);
        assert(self.blocks@[new_bid].prefix_depth == 0);
        assert(self.blocks@[new_bid].parent_block is None);
        assert forall|bid: BlockId|
            bid != new_bid && #[trigger] blocks_before_insert.contains_key(bid)
            implies self.blocks@.contains_key(bid) && self.blocks@[bid] == blocks_before_insert[bid]
        by {
        }

        self.free_blocks = self.free_blocks - 1;
        assert(self.free_blocks as int == old(self).free_blocks as int - 1);

        let slot = new_bid * BLOCK_SIZE;
        let mut slot_mapping: Vec<SlotId> = Vec::new();
        slot_mapping.push(slot);
        assert(slot_mapping@.len() == 1);
        assert(slot_mapping@[0] == slot);

        let mut next_block_ids = residency.block_ids.clone();
        next_block_ids.push(new_bid);
        assert(next_block_ids@.len() == residency.block_ids@.len() + 1);
        assert(next_block_ids@.subrange(0, residency.block_ids@.len() as int) == residency.block_ids@);
        assert(next_block_ids@[residency.block_ids@.len() as int] == new_bid);

        let next_residency = RequestResidency {
            block_ids: next_block_ids,
            cached_prefix_blocks: residency.cached_prefix_blocks,
            slot_mapping,
        };
        let ghost residency_before_update = self.request_residency@;
        self.request_residency.insert(rid, next_residency);
        proof {
            vstd::map::lemma_map_insert_domain(
                residency_before_update, rid, self.request_residency@[rid]);
        }
        assert(self.request_residency@.dom() == residency_before_update.dom());
        assert(self.request_residency@[rid].block_ids@.len() == residency.block_ids@.len() + 1);
        assert(self.request_residency@[rid].block_ids@.subrange(
            0,
            residency.block_ids@.len() as int,
        ) == residency.block_ids@);
        assert(self.request_residency@[rid].block_ids@[residency.block_ids@.len() as int] == new_bid);
        assert(self.request_residency@[rid].cached_prefix_blocks == residency.cached_prefix_blocks);
        assert(self.request_residency@[rid].slot_mapping@.len() == 1);
        assert(self.request_residency@[rid].slot_mapping@[0] == slot);

        assert(live_covers_queue(self));
        assert(queue_disjoint(self));
        assert(running_unique(self));
        assert(waiting_unique(self));
        assert(running_has_residency(self)) by {
            assert forall|r: RequestId|
                #[trigger] self.running@.contains(r)
                implies self.request_residency@.contains_key(r)
            by {
                assert(running_has_residency(old(self)));
                assert(old(self).request_residency@.contains_key(r));
            }
        }
        assert(waiting_has_no_residency(self)) by {
            assert forall|r: RequestId|
                #[trigger] self.waiting@.contains(r)
                implies !self.request_residency@.contains_key(r)
            by {
                assert(waiting_has_no_residency(old(self)));
                assert(!old(self).request_residency@.contains_key(r));
            }
        }
        assert(residency_blocks_in_range(self)) by {
            assert forall|r: RequestId, k: int|
                #![trigger self.request_residency@[r].block_ids@[k]]
                self.request_residency@.contains_key(r)
                && 0 <= k < self.request_residency@[r].block_ids@.len()
                implies self.blocks@.contains_key(self.request_residency@[r].block_ids@[k])
            by {
                let bid = self.request_residency@[r].block_ids@[k];
                assert(old(self).request_residency@.contains_key(r));
                if r == rid {
                    if k < residency.block_ids@.len() {
                        assert(bid == residency.block_ids@[k]);
                        assert(old(self).request_residency@[rid].block_ids@ == residency.block_ids@);
                        assert(residency_blocks_in_range(old(self)));
                        assert(old(self).blocks@.contains_key(bid));
                    } else {
                        assert(k == residency.block_ids@.len());
                        assert(bid == new_bid);
                    }
                } else {
                    assert(self.request_residency@[r] == old(self).request_residency@[r]);
                    assert(residency_blocks_in_range(old(self)));
                    assert(old(self).blocks@.contains_key(bid));
                }
            }
        }
        assert(block_token_bound(self)) by {
            assert forall|bid: BlockId|
                #[trigger] self.blocks@.contains_key(bid)
                implies self.blocks@[bid].tokens@.len() <= BLOCK_SIZE_SPEC as int
            by {
                if bid == new_bid {
                    assert(self.blocks@[bid].tokens@.len() == 1);
                    assert(1 <= BLOCK_SIZE_SPEC as int);
                } else {
                    assert(old(self).blocks@.contains_key(bid));
                    assert(self.blocks@[bid] == old(self).blocks@[bid]);
                    assert(block_token_bound(old(self)));
                }
            }
        }
        assert(hash_to_block_in_range(self)) by {
            assert forall|h: u64|
                #[trigger] self.hash_to_block@.contains_key(h)
                implies self.blocks@.contains_key(self.hash_to_block@[h])
                    && self.blocks@[self.hash_to_block@[h]].tokens@.len()
                        == BLOCK_SIZE_SPEC as int
            by {
                let bid = self.hash_to_block@[h];
                assert(hash_to_block_in_range(old(self)));
                assert(old(self).blocks@.contains_key(bid));
                assert(bid != new_bid);
                assert(self.blocks@[bid] == old(self).blocks@[bid]);
            }
        }
        assert(residency_block_ids_unique(self)) by {
            assert forall|r: RequestId|
                #[trigger] self.request_residency@.contains_key(r)
                implies self.request_residency@[r].block_ids@.no_duplicates()
            by {
                assert(old(self).request_residency@.contains_key(r));
                if r == rid {
                    assert(old(self).request_residency@[rid].block_ids@ == residency.block_ids@);
                    assert(residency_block_ids_unique(old(self)));
                    assert(!residency.block_ids@.contains(new_bid)) by {
                        if residency.block_ids@.contains(new_bid) {
                            let idx = residency.block_ids@.index_of(new_bid);
                            assert(residency.block_ids@[idx] == new_bid);
                            assert(residency_blocks_in_range(old(self)));
                            assert(old(self).blocks@.contains_key(new_bid));
                        }
                    }
                    assert(self.request_residency@[r].block_ids@ == residency.block_ids@.push(new_bid));
                } else {
                    assert(self.request_residency@[r] == old(self).request_residency@[r]);
                    assert(residency_block_ids_unique(old(self)));
                }
            }
        }
        assert(refcount_valid(self)) by {
            assert forall|bid: BlockId|
                #[trigger] self.blocks@.contains_key(bid)
                implies self.blocks@[bid].refcount as int
                    == residency_holders_of(self, bid).len() as int
            by {
                let holders = residency_holders_of(self, bid);
                if bid == new_bid {
                    let singleton = Set::<RequestId>::empty().insert(rid);
                    assert_sets_equal!(holders == singleton, r: RequestId => {
                        if holders.contains(r) {
                            assert(self.request_residency@.contains_key(r));
                            if r == rid {
                                assert(self.request_residency@[rid].block_ids@.contains(new_bid));
                            } else {
                                assert(self.request_residency@[r] == old(self).request_residency@[r]);
                                assert(old(self).request_residency@.contains_key(r));
                                assert(residency_blocks_in_range(old(self)));
                                let idx = old(self).request_residency@[r].block_ids@.index_of(new_bid);
                                assert(old(self).request_residency@[r].block_ids@[idx] == new_bid);
                                assert(old(self).blocks@.contains_key(new_bid));
                            }
                        }
                        if singleton.contains(r) {
                            if r == rid {
                                assert(self.request_residency@[rid].block_ids@.contains(new_bid));
                            } else {
                                vstd::set::lemma_set_empty(r);
                                vstd::set::lemma_set_insert_different(Set::<RequestId>::empty(), r, rid);
                            }
                        }
                    });
                    vstd::set_lib::lemma_set_is_empty_len0(Set::<RequestId>::empty());
                    vstd::set::lemma_set_insert_len(Set::<RequestId>::empty(), rid);
                    assert(holders.len() == 1);
                    assert(self.blocks@[bid].refcount == 1);
                } else {
                    assert(old(self).blocks@.contains_key(bid));
                    let old_holders = residency_holders_of(old(self), bid);
                    assert_sets_equal!(holders == old_holders, r: RequestId => {
                        if holders.contains(r) {
                            assert(self.request_residency@.contains_key(r));
                            assert(old(self).request_residency@.contains_key(r));
                            if r == rid {
                                assert(self.request_residency@[r].block_ids@ == old(self).request_residency@[r].block_ids@.push(new_bid));
                                assert(bid != new_bid);
                            } else {
                                assert(self.request_residency@[r] == old(self).request_residency@[r]);
                            }
                        }
                        if old_holders.contains(r) {
                            assert(old(self).request_residency@.contains_key(r));
                            assert(self.request_residency@.contains_key(r));
                            if r == rid {
                                assert(self.request_residency@[r].block_ids@ == old(self).request_residency@[r].block_ids@.push(new_bid));
                                assert(bid != new_bid);
                            } else {
                                assert(self.request_residency@[r] == old(self).request_residency@[r]);
                            }
                        }
                    });
                    assert(refcount_valid(old(self)));
                    assert(self.blocks@[bid] == old(self).blocks@[bid]);
                }
            }
        }
        assert(hash_to_block_consistent(self)) by {
            assert forall|h: u64|
                #[trigger] self.hash_to_block@.contains_key(h)
                implies self.blocks@[self.hash_to_block@[h]].hash_value == h
                    && self.blocks@[self.hash_to_block@[h]].prefix_depth > 0
            by {
                let bid = self.hash_to_block@[h];
                assert(hash_to_block_consistent(old(self)));
                assert(old(self).blocks@.contains_key(bid));
                assert(bid != new_bid);
                assert(self.blocks@[bid] == old(self).blocks@[bid]);
            }
        }
        assert(blocks_dom_in_range(self)) by {
            assert forall|bid: BlockId|
                #[trigger] self.blocks@.contains_key(bid)
                implies bid < self.num_blocks
            by {
                if bid == new_bid {
                } else {
                    assert(old(self).blocks@.contains_key(bid));
                    assert(blocks_dom_in_range(old(self)));
                }
            }
        }
        assert(block_count_valid(self)) by {
            assert(block_count_valid(old(self)));
            assert(self.blocks@.dom().len() == old(self).blocks@.dom().len() + 1);
            assert(self.free_blocks as int == old(self).free_blocks as int - 1);
        }
        proof {
            assert(self.free_queue.order@ == queue_after_pop);
            assert(old(self).free_queue.order@.subrange(0, 1)
                =~= Seq::<BlockId>::empty().push(new_bid));
            assert forall|bid: BlockId|
                #[trigger] self.blocks@.contains_key(bid)
                implies old(self).blocks@.contains_key(bid)
                    || old(self).free_queue.order@.subrange(0, 1).contains(bid)
            by {
                if !old(self).blocks@.contains_key(bid) {
                    assert(bid == new_bid);
                    assert(old(self).free_queue.order@.subrange(0, 1)[0]
                        == new_bid);
                    assert(old(self).free_queue.order@.subrange(
                        0, 1,
                    ).contains(bid));
                }
            }
            assert forall|bid: BlockId|
                #[trigger] old(self).free_queue.order@.subrange(0, 1).contains(bid)
                implies !old(self).blocks@.contains_key(bid)
                    && self.blocks@.contains_key(bid)
                    && self.blocks@[bid].refcount > 0
                    && self.blocks@[bid].prefix_depth == 0
            by {
                assert(bid == new_bid);
            }
            lemma_two_queue_after_vacant_prefix_allocation(
                old(self), self, 1,
            );
        }
        assert(cs_valid(self));

        proof {
            assert forall|bid: BlockId|
                #[trigger] self.blocks@.contains_key(bid)
                && self.blocks@[bid].prefix_depth > 0
                implies old(self).blocks@.contains_key(bid)
                    && self.blocks@[bid].tokens@ == old(self).blocks@[bid].tokens@
                    && self.blocks@[bid].prefix_depth
                        == old(self).blocks@[bid].prefix_depth
                    && self.blocks@[bid].parent_block
                        == old(self).blocks@[bid].parent_block
            by {
                assert(bid != new_bid);
                assert(self.blocks@[bid] == old(self).blocks@[bid]);
            }
            assert forall|bid: BlockId|
                #[trigger] old(self).blocks@.contains_key(bid)
                && old(self).blocks@[bid].prefix_depth > 0
                implies self.blocks@.contains_key(bid)
                    && self.blocks@[bid].prefix_depth
                        == old(self).blocks@[bid].prefix_depth
            by {
                assert(bid != new_bid);
                assert(self.blocks@[bid] == old(self).blocks@[bid]);
            }
            assert(positive_provenance_metadata_frame(old(self), self)) by {
                reveal(positive_provenance_metadata_frame);
                assert forall|bid: BlockId|
                    #[trigger] old(self).blocks@[bid].prefix_depth > 0
                    && old(self).blocks@.contains_key(bid)
                    && old(self).blocks@[bid].prefix_depth > 0
                    implies self.blocks@.contains_key(bid)
                        && self.blocks@[bid] == old(self).blocks@[bid]
                by {
                    assert(bid != new_bid);
                }
            }
            lemma_positive_provenance_origin_from_metadata_frame(
                old(self), self,
            );
            lemma_physical_parent_closure_from_metadata_frame(old(self), self);
            lemma_persistent_provenance_closed_frame(old(self), self);
        }

        assert(append_token_common_frame(old(self), self, rid)) by {
            assert forall|other: RequestId|
                other != rid && #[trigger] old(self).request_residency@.contains_key(other)
                implies self.request_residency@.contains_key(other)
                    && self.request_residency@[other] == old(self).request_residency@[other]
            by {
            }
        }
        assert(append_token_new_tail_success(old(self), self, rid, token));
        proof {
            lemma_registry_entries_from_pre_new_tail(
                old(self), self, rid, token,
            );
        }
    }

    pub fn append_token_dispatch_with_candidate(
        &mut self,
        rid: RequestId,
        token: TokenId,
        candidate: BlockId,
    )
        requires
            cs_valid(old(self)),
            free_queue_valid(old(self)),
            persistent_provenance_closed(old(self)),
            old(self).request_residency@.contains_key(rid),
            old(self).num_blocks <= u64::MAX / BLOCK_SIZE,
            ({
                if old(self).request_residency@[rid].block_ids@.len() > 0 {
                    let old_residency = old(self).request_residency@[rid];
                    let last_bid = old_residency.block_ids@[
                        old_residency.block_ids@.len() - 1
                    ];
                    if old(self).blocks@.contains_key(last_bid)
                        && old(self).blocks@[last_bid].tokens@.len() == BLOCK_SIZE_SPEC as int
                        && old(self).free_blocks > 0 {
                        !old(self).blocks@.contains_key(candidate)
                            && candidate < old(self).num_blocks
                            && old(self).free_queue.head == Some(candidate)
                    } else {
                        true
                    }
                } else {
                    true
                }
            }),
        ensures
            cs_valid(final(self)),
            free_queue_valid(final(self)),
            persistent_provenance_closed(final(self)),
            positive_provenance_metadata_frame(old(self), final(self)),
            positive_provenance_origin(old(self), final(self)),
            append_token_common_frame(old(self), final(self), rid),
            registry_entries_from_pre(
                old(self), final(self), Seq::<u64>::empty(),
            ),
            final(self).accepted_requests@ == old(self).accepted_requests@,
            // Blocks outside rid's residency are never touched (including
            // the no-op corners) — bystander alignment transport.
            forall|b: BlockId| #[trigger] old(self).blocks@.contains_key(b)
                && (old(self).request_residency@.contains_key(rid)
                    ==> !old(self).request_residency@[rid].block_ids@.contains(b))
                ==> final(self).blocks@.contains_key(b)
                    && final(self).blocks@[b] == old(self).blocks@[b],
            final(self).request_residency@.dom() == old(self).request_residency@.dom(),
            ({
                if old(self).request_residency@.contains_key(rid)
                    && old(self).request_residency@[rid].block_ids@.len() > 0 {
                    let old_residency = old(self).request_residency@[rid];
                    let last_bid = old_residency.block_ids@[
                        old_residency.block_ids@.len() - 1
                    ];
                    if old(self).blocks@.contains_key(last_bid)
                        && old(self).blocks@[last_bid].tokens@.len() < BLOCK_SIZE_SPEC as int {
                        append_token_tail_append_success(old(self), final(self), rid, token)
                    } else {
                        true
                    }
                } else {
                    true
                }
            }),
            ({
                if old(self).request_residency@.contains_key(rid)
                    && old(self).request_residency@[rid].block_ids@.len() > 0 {
                    let old_residency = old(self).request_residency@[rid];
                    let last_bid = old_residency.block_ids@[
                        old_residency.block_ids@.len() - 1
                    ];
                    if old(self).blocks@.contains_key(last_bid)
                        && old(self).blocks@[last_bid].tokens@.len() == BLOCK_SIZE_SPEC as int
                        && old(self).free_blocks > 0 {
                        append_token_new_tail_success(old(self), final(self), rid, token)
                    } else {
                        true
                    }
                } else {
                    true
                }
            }),
    {
        let residency = match self.request_residency.get(&rid) {
            Some(residency_ref) => residency_ref.clone(),
            None => {
                proof { assert(false); }
                return;
            },
        };
        if residency.block_ids.len() == 0 {
            assert(append_token_common_frame(old(self), self, rid)) by {
                assert forall|other: RequestId|
                    other != rid && #[trigger] old(self).request_residency@.contains_key(other)
                    implies self.request_residency@.contains_key(other)
                        && self.request_residency@[other] == old(self).request_residency@[other]
                by {
                }
            }
            assert(cs_valid(self));
            proof {
                lemma_registry_entries_from_pre_refl(old(self));
                lemma_positive_provenance_metadata_frame_refl(old(self));
                lemma_positive_provenance_origin_refl(old(self));
            }
            return;
        }

        let last_index = residency.block_ids.len() - 1;
        let last_id = residency.block_ids[last_index];
        assert(self.blocks@.contains_key(last_id)) by {
            assert(residency_blocks_in_range(self));
            assert(self.request_residency@[rid].block_ids@ == residency.block_ids@);
        }
        let entry = match self.blocks.get(&last_id) {
            Some(entry_ref) => entry_ref.clone(),
            None => {
                proof { assert(false); }
                return;
            },
        };
        if entry.tokens.len() < BLOCK_SIZE as usize {
            self.append_token_tail_append(rid, token);
            return;
        }
        if self.free_blocks > 0 {
            assert(entry.tokens@.len() == BLOCK_SIZE_SPEC as int) by {
                assert(block_token_bound(self));
                assert(entry.tokens@.len() >= BLOCK_SIZE_SPEC as int);
            }
            self.append_token_new_tail(rid, token, candidate);
            return;
        }

        assert(append_token_common_frame(old(self), self, rid)) by {
            assert forall|other: RequestId|
                other != rid && #[trigger] old(self).request_residency@.contains_key(other)
                implies self.request_residency@.contains_key(other)
                    && self.request_residency@[other] == old(self).request_residency@[other]
            by {
            }
        }
        assert(cs_valid(self));
        proof {
            lemma_registry_entries_from_pre_refl(old(self));
            lemma_positive_provenance_metadata_frame_refl(old(self));
            lemma_positive_provenance_origin_refl(old(self));
        }
    }

    pub fn find_free_block(&self) -> (out: Option<BlockId>)
        requires
            cs_valid(self),
            free_queue_valid(self),
        ensures
            match out {
                Some(bid) => bid < self.num_blocks
                    && !self.blocks@.contains_key(bid)
                    && self.free_queue.head == Some(bid),
                None => self.free_blocks == 0,
            },
    {
        if self.free_blocks == 0 {
            return None;
        }
        assert(self.free_queue.order@.len() > 0);
        let bid = match self.free_queue.head {
            Some(candidate) => candidate,
            None => {
                proof { assert(false); }
                return None;
            },
        };
        assert(bid == self.free_queue.order@[0]);
        assert(!self.blocks@.contains_key(bid)) by {
            assert(free_queue_partitioned(self));
        }
        assert(bid < self.num_blocks) by {
            assert(free_queue_membership_valid(self));
        }
        Some(bid)
    }

    // Append one generated token to a resident request's cache metadata.
    //
    // Executable shape mirrors Dafny `AppendToken`: append into the current
    // tail block when it has capacity, otherwise allocate a fresh tail block,
    // then set decode slot_mapping to the singleton physical slot for the
    // newly appended token.
    pub fn append_token(&mut self, rid: RequestId, token: TokenId)
        requires
            cs_valid(old(self)),
            free_queue_valid(old(self)),
            persistent_provenance_closed(old(self)),
            old(self).request_residency@.contains_key(rid),
            old(self).num_blocks <= u64::MAX / BLOCK_SIZE,
        ensures
            cs_valid(final(self)),
            free_queue_valid(final(self)),
            persistent_provenance_closed(final(self)),
            positive_provenance_metadata_frame(old(self), final(self)),
            positive_provenance_origin(old(self), final(self)),
            append_token_common_frame(old(self), final(self), rid),
            registry_entries_from_pre(
                old(self), final(self), Seq::<u64>::empty(),
            ),
            final(self).accepted_requests@ == old(self).accepted_requests@,
            // Blocks outside rid's residency are never touched (including
            // the no-op corners) — bystander alignment transport.
            forall|b: BlockId| #[trigger] old(self).blocks@.contains_key(b)
                && (old(self).request_residency@.contains_key(rid)
                    ==> !old(self).request_residency@[rid].block_ids@.contains(b))
                ==> final(self).blocks@.contains_key(b)
                    && final(self).blocks@[b] == old(self).blocks@[b],
            final(self).request_residency@.dom() == old(self).request_residency@.dom(),
            ({
                if old(self).request_residency@.contains_key(rid)
                    && old(self).request_residency@[rid].block_ids@.len() > 0 {
                    let old_residency = old(self).request_residency@[rid];
                    let last_bid = old_residency.block_ids@[
                        old_residency.block_ids@.len() - 1
                    ];
                    if old(self).blocks@.contains_key(last_bid)
                        && old(self).blocks@[last_bid].tokens@.len() < BLOCK_SIZE_SPEC as int {
                        append_token_tail_append_success(old(self), final(self), rid, token)
                    } else {
                        true
                    }
                } else {
                    true
                }
            }),
            ({
                if old(self).request_residency@.contains_key(rid)
                    && old(self).request_residency@[rid].block_ids@.len() > 0 {
                    let old_residency = old(self).request_residency@[rid];
                    let last_bid = old_residency.block_ids@[
                        old_residency.block_ids@.len() - 1
                    ];
                    if old(self).blocks@.contains_key(last_bid)
                        && old(self).blocks@[last_bid].tokens@.len() == BLOCK_SIZE_SPEC as int
                        && old(self).free_blocks > 0 {
                        append_token_new_tail_success(old(self), final(self), rid, token)
                    } else {
                        true
                    }
                } else {
                    true
                }
            }),
    {
        let residency = match self.request_residency.get(&rid) {
            Some(residency_ref) => residency_ref.clone(),
            None => {
                proof { assert(false); }
                return;
            },
        };
        if residency.block_ids.len() == 0 {
            self.append_token_dispatch_with_candidate(rid, token, 0);
            return;
        }

        let last_index = residency.block_ids.len() - 1;
        let last_id = residency.block_ids[last_index];
        assert(self.blocks@.contains_key(last_id)) by {
            assert(residency_blocks_in_range(self));
            assert(self.request_residency@[rid].block_ids@ == residency.block_ids@);
        }
        let entry = match self.blocks.get(&last_id) {
            Some(entry_ref) => entry_ref.clone(),
            None => {
                proof { assert(false); }
                return;
            },
        };
        if entry.tokens.len() < BLOCK_SIZE as usize {
            assert(entry.tokens@.len() < BLOCK_SIZE_SPEC as int);
            self.append_token_dispatch_with_candidate(rid, token, 0);
            return;
        }

        assert(entry.tokens@.len() == BLOCK_SIZE_SPEC as int) by {
            assert(entry.tokens@.len() >= BLOCK_SIZE_SPEC as int);
            assert(block_token_bound(self));
        }
        let candidate = if self.free_blocks > 0 {
            match self.find_free_block() {
                Some(bid) => bid,
                None => {
                    proof { assert(false); }
                    0
                },
            }
        } else {
            0
        };
        self.append_token_dispatch_with_candidate(rid, token, candidate);
    }
}

} // verus!
