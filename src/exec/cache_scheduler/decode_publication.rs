// Completed-tail publication. The Engine must call this only after forward
// has materialized the supplied history; metadata fullness alone is not KV
// readiness. Keep the one-page transition separate from plan/commit proofs.

use super::*;

verus! {

impl CacheScheduler {
    #[verifier::spinoff_prover]
    pub fn publish_decode_prefixes(&mut self, scheduled: &Vec<RequestId>)
        -> (published_out: Ghost<Set<RequestId>>)
        requires
            cs_valid(old(self)),
            free_queue_valid(old(self)),
            persistent_provenance_closed(old(self)),
            decode_plan_ready(old(self)),
            residency_history_aligned(old(self)),
            slot_mapping_aligned(old(self)),
            residency_running_aligned(old(self)),
            forall|i: int| 0 <= i < scheduled@.len()
                ==> old(self).running@.contains(#[trigger] scheduled@[i]),
        ensures
            cs_valid(final(self)),
            free_queue_valid(final(self)),
            persistent_provenance_closed(final(self)),
            decode_plan_ready(final(self)),
            residency_history_aligned(final(self)),
            slot_mapping_aligned(final(self)),
            residency_running_aligned(final(self)),
            decode_publication_effect(old(self), final(self), published_out@),
            publication_layout_frame(old(self), final(self)),
            published_out@.subset_of(scheduled@.to_set()),
            pre_commit_tails_exclusive(old(self)) ==> pre_commit_tails_exclusive(final(self)),
    {
        hide(free_queue_valid);
        reveal(decode_publication_effect);
        let ghost mut published = Set::<RequestId>::empty();
        proof { lemma_decode_publication_effect_refl(old(self)); }
        let mut i: usize = 0;
        while i < scheduled.len()
            invariant
                i <= scheduled@.len(),
                cs_valid(self),
                free_queue_valid(self),
                persistent_provenance_closed(self),
                decode_plan_ready(self),
                residency_history_aligned(self),
                slot_mapping_aligned(self),
                residency_running_aligned(self),
                decode_publication_effect(old(self), self, published),
                publication_layout_frame(old(self), self),
                published.subset_of(scheduled@.to_set()),
                forall|k: int| 0 <= k < scheduled@.len()
                    ==> old(self).running@.contains(#[trigger] scheduled@[k]),
            decreases scheduled.len() - i,
        {
            let rid = scheduled[i];
            let ghost before = *self;
            let did_publish = self.try_publish_decode_tail(rid);
            proof {
                reveal(decode_publication_effect);
                let next = if did_publish { Set::empty().insert(rid) } else { Set::empty() };
                lemma_decode_publication_effect_transitive(old(self), &before, self, published, next);
                published = published.union(next);
                assert(scheduled@.contains(rid));
            }
            i += 1;
        }
        proof {
            if pre_commit_tails_exclusive(old(self)) {
                assert forall|rid: RequestId| #[trigger] self.running@.contains(rid)
                    && self.live_requests@.contains_key(rid) implies {
                        let ids = self.request_residency@[rid].block_ids@;
                        &&& self.request_residency@.contains_key(rid)
                        &&& ids.len() >= 1
                        &&& self.blocks@.contains_key(ids[ids.len() - 1])
                        &&& self.blocks@[ids[ids.len() - 1]].refcount == 1
                        &&& (self.blocks@[ids[ids.len() - 1]].tokens@.len() < BLOCK_SIZE_SPEC as int
                            ==> self.blocks@[ids[ids.len() - 1]].hash_value == 0)
                    }
                by {}
            }
        }
        Ghost(published)
    }

    // Called for scheduled rows only, after their forward. There is no work on
    // prefill rows or between page boundaries. The last sampled token is not
    // appended until commit, so `history` here names only executed positions.
    #[verifier::spinoff_prover]
    pub fn try_publish_decode_tail(&mut self, rid: RequestId) -> (published: bool)
        requires
            cs_valid(old(self)),
            free_queue_valid(old(self)),
            persistent_provenance_closed(old(self)),
            decode_plan_ready(old(self)),
            residency_history_aligned(old(self)),
            slot_mapping_aligned(old(self)),
            residency_running_aligned(old(self)),
            old(self).running@.contains(rid),
        ensures
            cs_valid(final(self)),
            free_queue_valid(final(self)),
            persistent_provenance_closed(final(self)),
            decode_plan_ready(final(self)),
            residency_history_aligned(final(self)),
            slot_mapping_aligned(final(self)),
            residency_running_aligned(final(self)),
            decode_publication_effect(old(self), final(self),
                if published { Set::empty().insert(rid) } else { Set::empty() }),
    {
        hide(free_queue_valid);
        reveal(decode_publication_effect);
        proof { lemma_decode_publication_effect_refl(old(self)); }
        let state = match self.live_requests.get(&rid) {
            Some(state) => {
                let len = history_len(state);
                if state.generated_tokens.len() == 0 || len % (BLOCK_SIZE as usize) != 0 {
                    return false;
                }
                state.clone()
            },
            None => { proof { assert(false); } return false; },
        };
        let len = history_len(&state);
        if state.generated_tokens.len() == 0 || len % (BLOCK_SIZE as usize) != 0 {
            return false;
        }
        let ids = match self.request_residency.get(&rid) {
            Some(residency) => residency.block_ids.clone(),
            None => { proof { assert(false); } return false; },
        };
        let tail = ids[ids.len() - 1];
        let entry = match self.blocks.get(&tail) {
            Some(entry) => entry.clone(),
            None => { proof { assert(false); } return false; },
        };
        if entry.prefix_depth != 0 || entry.hash_value != 0
            || entry.refcount != 1 || entry.parent_block.is_some() {
            return false;
        }
        if ids.len() > 1 {
            let parent = ids[ids.len() - 2];
            match self.blocks.get(&parent) {
                Some(parent_entry) => {
                    if parent_entry.prefix_depth == 0 { return false; }
                },
                None => { proof { assert(false); } return false; },
            }
        }
        let mut tokens = state.prompt_tokens.clone();
        let mut i: usize = 0;
        while i < state.generated_tokens.len()
            invariant
                i <= state.generated_tokens@.len(),
                tokens@ == state.prompt_tokens@ + state.generated_tokens@.subrange(0, i as int),
                state.prompt_tokens@.len() + state.generated_tokens@.len() <= usize::MAX as int,
            decreases state.generated_tokens.len() - i,
        {
            tokens.push(state.generated_tokens[i]);
            i += 1;
            assert(tokens@ == state.prompt_tokens@ + state.generated_tokens@.subrange(0, i as int));
        }
        proof {
            assert(tokens@ == history(state));
            let n = len as int;
            let pages = ids@.len() as int;
            let tail_len = entry.tokens@.len() as int;
            assert(n == pages * (BLOCK_SIZE_SPEC as int)) by (nonlinear_arith)
                requires n == (pages - 1) * 64 + tail_len,
                    1 <= tail_len <= 64, n % 64 == 0, BLOCK_SIZE_SPEC as int == 64,
            {}
        }
        let _registered = self.publish_completed_tail(rid, &tokens);
        proof {
            assert(publication_layout_frame(old(self), self));
            lemma_publication_layout_preserved(old(self), self);
            lemma_token_placement_prefix_transfer(old(self).blocks@, self.blocks@,
                ids@, tokens@, tokens@.len() as int);
            assert(published_decode_row(old(self), self, rid));
            assert(entry.tokens@.len() == BLOCK_SIZE_SPEC as int);
            let chosen = Set::<RequestId>::empty().insert(rid);
            assert forall|bid: BlockId| #[trigger] old(self).blocks@.contains_key(bid)
                && !(exists|r: RequestId| chosen.contains(r)
                    && old(self).request_residency@[r].block_ids@[
                        old(self).request_residency@[r].block_ids@.len() - 1] == bid)
                implies self.blocks@[bid] == old(self).blocks@[bid]
            by {
                if bid == tail {
                    assert(exists|r: RequestId| chosen.contains(r)
                        && old(self).request_residency@[r].block_ids@[
                            old(self).request_residency@[r].block_ids@.len() - 1] == bid) by {
                        assert(chosen.contains(rid));
                    }
                }
            }
        }
        true
    }

    #[verifier::spinoff_prover]
    pub fn publish_completed_tail(
        &mut self,
        rid: RequestId,
        executed_tokens: &Vec<TokenId>,
    ) -> (registered: Vec<u64>)
        requires
            cs_valid(old(self)),
            free_queue_valid(old(self)),
            persistent_provenance_closed(old(self)),
            old(self).request_residency@.contains_key(rid),
            old(self).request_residency@[rid].block_ids@.len() > 0,
            executed_tokens@.len() == old(self).request_residency@[rid].block_ids@.len()
                * (BLOCK_SIZE_SPEC as int),
            token_placement_prefix(
                old(self).blocks@, old(self).request_residency@[rid].block_ids@,
                executed_tokens@, executed_tokens@.len() as int,
            ),
            ({
                let ids = old(self).request_residency@[rid].block_ids@;
                let tail = ids[ids.len() - 1];
                &&& old(self).blocks@.contains_key(tail)
                &&& old(self).blocks@[tail].refcount == 1
                &&& old(self).blocks@[tail].prefix_depth == 0
                &&& old(self).blocks@[tail].hash_value == 0
                &&& old(self).blocks@[tail].parent_block is None
                &&& (ids.len() > 1 ==> {
                    let parent = ids[ids.len() - 2];
                    old(self).blocks@.contains_key(parent)
                        && old(self).blocks@[parent].prefix_depth > 0
                })
            }),
        ensures
            cs_valid(final(self)),
            free_queue_valid(final(self)),
            persistent_provenance_closed(final(self)),
            final(self).config == old(self).config,
            final(self).num_blocks == old(self).num_blocks,
            final(self).free_blocks == old(self).free_blocks,
            final(self).running@ == old(self).running@,
            final(self).waiting@ == old(self).waiting@,
            final(self).live_requests@ == old(self).live_requests@,
            final(self).accepted_requests@ == old(self).accepted_requests@,
            final(self).request_residency@ == old(self).request_residency@,
            final(self).blocks@.dom() == old(self).blocks@.dom(),
            forall|bid: BlockId| #[trigger] old(self).blocks@.contains_key(bid)
                ==> final(self).blocks@[bid].tokens@ == old(self).blocks@[bid].tokens@
                    && final(self).blocks@[bid].refcount == old(self).blocks@[bid].refcount,
            forall|bid: BlockId|
                #[trigger] old(self).blocks@.contains_key(bid)
                && bid != old(self).request_residency@[rid].block_ids@[
                    old(self).request_residency@[rid].block_ids@.len() - 1]
                ==> final(self).blocks@[bid] == old(self).blocks@[bid],
            registered_prefix_chain(
                final(self).blocks@, old(self).request_residency@[rid].block_ids@,
            ),
            registered@.len() == 1,
            registration_registry_frame(
                old(self), final(self), rid, executed_tokens@.len() as int,
                old(self).request_residency@[rid].block_ids@.len() as int - 1,
                registered@,
            ),
            registration_positive_page_origin(
                old(self), final(self), rid, executed_tokens@.len() as int,
                old(self).request_residency@[rid].block_ids@.len() as int - 1,
            ),
            positive_provenance_metadata_frame(old(self), final(self)),
    {
        hide(free_queue_valid);
        let _prefix_len = executed_tokens.len();
        let ids = match self.request_residency.get(&rid) {
            Some(residency) => residency.block_ids.clone(),
            None => { proof { assert(false); } Vec::new() },
        };
        let last = ids.len() - 1;
        proof {
            assert(ids@ == old(self).request_residency@[rid].block_ids@);
            let n = executed_tokens@.len() as int;
            let pages = ids@.len() as int;
            assert(n / (BLOCK_SIZE_SPEC as int) == pages) by (nonlinear_arith)
                requires n == pages * (BLOCK_SIZE_SPEC as int), BLOCK_SIZE_SPEC as int == 64,
            {}
            assert(blocks_needed_for(n as nat) == pages) by (nonlinear_arith)
                requires n == pages * 64, pages > 0,
                    blocks_needed_for(n as nat) == (n + 63) / 64,
            {}
            if last > 0 {
                lemma_positive_residency_prefix_registered(old(self), rid, last as int - 1);
            } else {
                reveal(registered_prefix_chain);
            }
            assert(ids@.subrange(last as int, ids@.len() as int) == seq![ids@[last as int]]);
            assert forall|h: u64| #[trigger] old(self).hash_to_block@.contains_key(h)
                implies !ids@.subrange(last as int, ids@.len() as int)
                    .contains(old(self).hash_to_block@[h])
            by {
                assert(hash_to_block_no_zero(old(self)));
                assert(hash_to_block_consistent(old(self)));
            }
        }
        let registered = self.publish_full_prefix_pages(rid, executed_tokens, last);
        proof {
            lemma_positive_residency_prefix_registered(self, rid, last as int);
            assert(ids@.subrange(0, ids@.len() as int) == ids@);
        }
        registered
    }
}

}
