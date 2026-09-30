// Verified live-request state and single-request commit transitions.

use super::*;

verus! {
impl CacheScheduler {
    pub fn remove_live_request_after_dequeue(&mut self, rid: RequestId)
        requires
            cs_valid(old(self)),
            free_queue_valid(old(self)),
            !old(self).running@.contains(rid),
            !old(self).waiting@.contains(rid),
            !old(self).request_residency@.contains_key(rid),
        ensures
            cs_valid(final(self)),
            free_queue_valid(final(self)),
            final(self).config == old(self).config,
            final(self).num_blocks == old(self).num_blocks,
            final(self).free_blocks == old(self).free_blocks,
            final(self).blocks@ == old(self).blocks@,
            final(self).running@ == old(self).running@,
            final(self).waiting@ == old(self).waiting@,
            final(self).request_residency@ == old(self).request_residency@,
            final(self).hash_to_block@ == old(self).hash_to_block@,
            final(self).accepted_requests@ == old(self).accepted_requests@,
            !final(self).live_requests@.contains_key(rid),
            forall|r: RequestId|
                #[trigger] final(self).live_requests@.contains_key(r)
                ==> old(self).live_requests@.contains_key(r),
            forall|other: RequestId|
                other != rid && #[trigger] old(self).live_requests@.contains_key(other)
                ==> final(self).live_requests@.contains_key(other)
                    && final(self).live_requests@[other] == old(self).live_requests@[other],
    {
        let ghost live_before_remove = self.live_requests@;
        self.live_requests.remove(&rid);
        proof {
            vstd::map::lemma_map_remove_domain(live_before_remove, rid);
        }
        assert(self.live_requests@ == live_before_remove.remove(rid));
        assert(!self.live_requests@.contains_key(rid));

        assert(live_covers_queue(self)) by {
            assert forall|r: RequestId|
                self.running@.contains(r) || self.waiting@.contains(r)
                implies #[trigger] self.live_requests@.contains_key(r)
            by {
                assert(old(self).running@.contains(r) || old(self).waiting@.contains(r));
                assert(r != rid);
                assert(live_covers_queue(old(self)));
                assert(old(self).live_requests@.contains_key(r));
            }
        }
        assert(queue_disjoint(self));
        assert(running_unique(self));
        assert(waiting_unique(self));
        assert(running_has_residency(self));
        assert(waiting_has_no_residency(self));
        assert(waiting_unstarted(self)) by {
            assert forall|w: RequestId|
                #[trigger] self.waiting@.contains(w) && self.live_requests@.contains_key(w)
                implies self.live_requests@[w].generated_tokens@.len() == 0
            by {
                assert(w != rid);
                assert(old(self).waiting@.contains(w));
                assert(self.live_requests@[w] == old(self).live_requests@[w]);
            }
        }
        assert(residency_blocks_in_range(self));
        assert(block_token_bound(self));
        assert(hash_to_block_in_range(self));
        assert(residency_block_ids_unique(self));
        assert(refcount_valid(self));
        assert(hash_to_block_consistent(self));
        assert(blocks_dom_in_range(self));
        assert(block_count_valid(self));
        assert(cs_valid(self));

        assert forall|other: RequestId|
            other != rid && #[trigger] old(self).live_requests@.contains_key(other)
            implies self.live_requests@.contains_key(other)
                && self.live_requests@[other] == old(self).live_requests@[other]
        by {
        }
        assert forall|r: RequestId|
            #[trigger] self.live_requests@.contains_key(r)
            implies old(self).live_requests@.contains_key(r)
        by {
        }
    }

    pub fn update_live_request(&mut self, rid: RequestId, next_state: RequestState)
        requires
            cs_valid(old(self)),
            free_queue_valid(old(self)),
            old(self).live_requests@.contains_key(rid),
            !old(self).waiting@.contains(rid),
        ensures
            cs_valid(final(self)),
            free_queue_valid(final(self)),
            final(self).config == old(self).config,
            final(self).num_blocks == old(self).num_blocks,
            final(self).free_blocks == old(self).free_blocks,
            final(self).blocks@ == old(self).blocks@,
            final(self).running@ == old(self).running@,
            final(self).waiting@ == old(self).waiting@,
            final(self).request_residency@ == old(self).request_residency@,
            final(self).hash_to_block@ == old(self).hash_to_block@,
            final(self).accepted_requests@ == old(self).accepted_requests@,
            final(self).live_requests@.contains_key(rid),
            final(self).live_requests@[rid] == next_state,
            forall|r: RequestId|
                #[trigger] final(self).live_requests@.contains_key(r)
                ==> old(self).live_requests@.contains_key(r),
            forall|other: RequestId|
                other != rid && #[trigger] old(self).live_requests@.contains_key(other)
                ==> final(self).live_requests@.contains_key(other)
                    && final(self).live_requests@[other] == old(self).live_requests@[other],
    {
        let ghost live_before_update = self.live_requests@;
        self.live_requests.insert(rid, next_state);
        proof {
            vstd::map::lemma_map_insert_domain(
                live_before_update,
                rid,
                self.live_requests@[rid],
            );
        }

        assert(live_covers_queue(self)) by {
            assert forall|r: RequestId|
                self.running@.contains(r) || self.waiting@.contains(r)
                implies #[trigger] self.live_requests@.contains_key(r)
            by {
                assert(old(self).running@.contains(r) || old(self).waiting@.contains(r));
                assert(live_covers_queue(old(self)));
                assert(old(self).live_requests@.contains_key(r));
            }
        }
        assert(queue_disjoint(self));
        assert(running_unique(self));
        assert(waiting_unique(self));
        assert(running_has_residency(self));
        assert(waiting_has_no_residency(self));
        assert(waiting_unstarted(self)) by {
            assert forall|w: RequestId|
                #[trigger] self.waiting@.contains(w) && self.live_requests@.contains_key(w)
                implies self.live_requests@[w].generated_tokens@.len() == 0
            by {
                assert(w != rid);
                assert(old(self).waiting@.contains(w));
                assert(self.live_requests@[w] == old(self).live_requests@[w]);
            }
        }
        assert(residency_blocks_in_range(self));
        assert(block_token_bound(self));
        assert(hash_to_block_in_range(self));
        assert(residency_block_ids_unique(self));
        assert(refcount_valid(self));
        assert(hash_to_block_consistent(self));
        assert(blocks_dom_in_range(self));
        assert(block_count_valid(self));
        assert(cs_valid(self));

        assert forall|other: RequestId|
            other != rid && #[trigger] old(self).live_requests@.contains_key(other)
            implies self.live_requests@.contains_key(other)
                && self.live_requests@[other] == old(self).live_requests@[other]
        by {
        }
        assert forall|r: RequestId|
            #[trigger] self.live_requests@.contains_key(r)
            implies old(self).live_requests@.contains_key(r)
        by {
            if r == rid {
            }
        }
    }

    // Finalize a cache-only prefill row.  The request state is deliberately
    // unchanged: its computed full prompt pages remain registered in the
    // persistent prefix cache after residency refcounts drop, while the live
    // request returns to `waiting` and may be admitted again from that prefix.
    // An unregistered partial tail is reclaimed and will be recomputed.
    #[verifier::spinoff_prover]
    #[verifier::rlimit(200)]
    pub fn commit_kv_only_prefill(&mut self, rid: RequestId)
        requires
            cs_valid(old(self)),
            free_queue_valid(old(self)),
            persistent_provenance_closed(old(self)),
            residency_history_aligned(old(self)),
            slot_mapping_aligned(old(self)),
            pre_commit_tails_exclusive(old(self)),
            residency_running_aligned(old(self)),
            old(self).running@.contains(rid),
            old(self).live_requests@.contains_key(rid),
            old(self).live_requests@[rid].generated_tokens@.len() == 0,
        ensures
            cs_valid(final(self)),
            free_queue_valid(final(self)),
            persistent_provenance_closed(final(self)),
            positive_provenance_metadata_frame(old(self), final(self)),
            positive_provenance_origin(old(self), final(self)),
            registry_entries_from_pre(old(self), final(self), Seq::<u64>::empty()),
            residency_history_aligned(final(self)),
            slot_mapping_aligned(final(self)),
            pre_commit_tails_exclusive(final(self)),
            residency_running_aligned(final(self)),
            final(self).config == old(self).config,
            final(self).num_blocks == old(self).num_blocks,
            final(self).live_requests@ == old(self).live_requests@,
            final(self).accepted_requests@ == old(self).accepted_requests@,
            !final(self).running@.contains(rid),
            final(self).waiting@ == old(self).waiting@.push(rid),
            !final(self).request_residency@.contains_key(rid),
            forall|r: RequestId| #[trigger] final(self).running@.contains(r)
                ==> old(self).running@.contains(r),
            forall|r: RequestId| r != rid && #[trigger] old(self).running@.contains(r)
                ==> final(self).running@.contains(r),
            forall|other: RequestId| other != rid
                && #[trigger] old(self).request_residency@.contains_key(other)
                ==> final(self).request_residency@.contains_key(other)
                    && final(self).request_residency@[other]
                        == old(self).request_residency@[other],
            forall|r: RequestId| #[trigger] final(self).request_residency@.contains_key(r)
                ==> old(self).request_residency@.contains_key(r),
            forall|b: BlockId| #[trigger] final(self).blocks@.contains_key(b)
                ==> old(self).blocks@.contains_key(b)
                    && final(self).blocks@[b].tokens@ == old(self).blocks@[b].tokens@
                    && final(self).blocks@[b].prefix_depth
                        == old(self).blocks@[b].prefix_depth
                    && final(self).blocks@[b].parent_block
                        == old(self).blocks@[b].parent_block,
            forall|r: RequestId| r != rid
                && #[trigger] old(self).running@.contains(r)
                && old(self).live_requests@.contains_key(r)
                ==> {
                    let ids = old(self).request_residency@[r].block_ids@;
                    let tail = ids[ids.len() - 1];
                    final(self).blocks@.contains_key(tail)
                        && final(self).blocks@[tail] == old(self).blocks@[tail]
                },
            final(self).free_blocks >= old(self).free_blocks,
    {
        let ghost before = *self;
        remove_request_id_from_queue(&mut self.running, rid);
        let ghost pre_deallocate = *self;

        proof {
            assert(!pre_deallocate.running@.contains(rid));
            assert(!pre_deallocate.waiting@.contains(rid)) by {
                assert(queue_disjoint(&before));
            }
            assert(live_covers_queue(&pre_deallocate)) by {
                assert forall|r: RequestId|
                    pre_deallocate.running@.contains(r)
                        || pre_deallocate.waiting@.contains(r)
                    implies #[trigger] pre_deallocate.live_requests@.contains_key(r)
                by {
                    assert(before.running@.contains(r) || before.waiting@.contains(r));
                }
            }
            assert(queue_disjoint(&pre_deallocate)) by {
                assert forall|r: RequestId|
                    pre_deallocate.running@.contains(r)
                    implies !#[trigger] pre_deallocate.waiting@.contains(r)
                by {
                    assert(before.running@.contains(r));
                }
            }
            assert(running_unique(&pre_deallocate));
            assert(waiting_unique(&pre_deallocate));
            assert(running_has_residency(&pre_deallocate)) by {
                assert forall|r: RequestId|
                    #[trigger] pre_deallocate.running@.contains(r)
                    implies pre_deallocate.request_residency@.contains_key(r)
                by {
                    assert(before.running@.contains(r));
                }
            }
            assert(waiting_has_no_residency(&pre_deallocate));
            assert(waiting_unstarted(&pre_deallocate));
            assert(cs_valid(&pre_deallocate));

            lemma_persistent_provenance_closed_blocks_eq(
                &before, &pre_deallocate,
            );
            lemma_registry_entries_from_pre_blocks_eq(
                &before, &pre_deallocate,
            );
            lemma_positive_provenance_metadata_frame_blocks_eq(
                &before, &pre_deallocate,
            );
            lemma_positive_provenance_origin_blocks_eq(
                &before, &pre_deallocate,
            );
        }

        self.deallocate(rid);
        let ghost post_deallocate = *self;

        proof {
            lemma_registry_entries_from_pre_transitive(
                &before, &pre_deallocate, &post_deallocate,
                Seq::<u64>::empty(),
            );
            lemma_positive_provenance_metadata_frame_transitive(
                &before, &pre_deallocate, &post_deallocate,
            );
            lemma_positive_provenance_origin_transitive(
                &before, &pre_deallocate, &post_deallocate,
            );
        }

        self.waiting.push(rid);

        proof {
            assert(self.waiting@ == before.waiting@.push(rid));
            assert(live_covers_queue(self)) by {
                assert forall|r: RequestId|
                    self.running@.contains(r) || self.waiting@.contains(r)
                    implies #[trigger] self.live_requests@.contains_key(r)
                by {
                    if r == rid {
                    } else {
                        if self.waiting@.contains(r) {
                            let k = choose|k: int| 0 <= k < self.waiting@.len()
                                && self.waiting@[k] == r;
                            if k < before.waiting@.len() {
                                assert(before.waiting@[k] == r);
                                assert(before.waiting@.contains(r));
                            } else {
                                assert(r == rid);
                            }
                        }
                        assert(before.running@.contains(r)
                            || before.waiting@.contains(r));
                    }
                }
            }
            assert(queue_disjoint(self)) by {
                assert forall|r: RequestId|
                    self.running@.contains(r)
                    implies !#[trigger] self.waiting@.contains(r)
                by {
                    assert(r != rid);
                    assert(before.running@.contains(r));
                    assert(!before.waiting@.contains(r));
                }
            }
            assert(running_unique(self));
            assert(waiting_unique(self)) by {
                assert(before.waiting@.no_duplicates());
                assert(!before.waiting@.contains(rid));
                reveal(Seq::no_duplicates);
            }
            assert(running_has_residency(self));
            assert(waiting_has_no_residency(self)) by {
                assert forall|r: RequestId|
                    #[trigger] self.waiting@.contains(r)
                    implies !self.request_residency@.contains_key(r)
                by {
                    if r == rid {
                    } else {
                        let k = choose|k: int| 0 <= k < self.waiting@.len()
                            && self.waiting@[k] == r;
                        if k < before.waiting@.len() {
                            assert(before.waiting@[k] == r);
                            assert(before.waiting@.contains(r));
                        } else {
                            assert(r == rid);
                        }
                    }
                }
            }
            assert(waiting_unstarted(self)) by {
                assert forall|r: RequestId|
                    #[trigger] self.waiting@.contains(r)
                        && self.live_requests@.contains_key(r)
                    implies self.live_requests@[r].generated_tokens@.len() == 0
                by {
                    if r == rid {
                    } else {
                        let k = choose|k: int| 0 <= k < self.waiting@.len()
                            && self.waiting@[k] == r;
                        if k < before.waiting@.len() {
                            assert(before.waiting@[k] == r);
                            assert(before.waiting@.contains(r));
                        } else {
                            assert(r == rid);
                        }
                    }
                }
            }
            assert(cs_valid(self));

            assert forall|r: RequestId|
                #[trigger] self.running@.contains(r)
                    && self.live_requests@.contains_key(r)
                implies {
                    let ids = self.request_residency@[r].block_ids@;
                    let tail = ids[ids.len() - 1];
                    self.blocks@[tail] == before.blocks@[tail]
                }
            by {
                assert(r != rid);
                assert(before.running@.contains(r));
                assert(before.live_requests@.contains_key(r));
                assert(self.request_residency@[r]
                    == before.request_residency@[r]);
                let ids = before.request_residency@[r].block_ids@;
                assert(ids.len() >= 1);
                let tail = ids[ids.len() - 1];
                assert(before.blocks@[tail].refcount == 1);
                if before.request_residency@[rid].block_ids@.contains(tail) {
                    assert(refcount_valid(&before));
                    let holders = residency_holders_of(&before, tail);
                    assert(holders.contains(r));
                    assert(holders.contains(rid));
                    lemma_two_holders(holders, r, rid);
                    assert(before.blocks@[tail].refcount as int >= 2);
                    assert(false);
                }
                assert(post_deallocate.blocks@[tail]
                    == pre_deallocate.blocks@[tail]);
            }

            lemma_stable_companions_after_park(&before, self, rid);

            lemma_persistent_provenance_closed_blocks_eq(
                &post_deallocate, self,
            );
            lemma_registry_entries_from_pre_blocks_eq(
                &post_deallocate, self,
            );
            lemma_registry_entries_from_pre_transitive(
                &before, &post_deallocate, self,
                Seq::<u64>::empty(),
            );
            lemma_positive_provenance_metadata_frame_blocks_eq(
                &post_deallocate, self,
            );
            lemma_positive_provenance_metadata_frame_transitive(
                &before, &post_deallocate, self,
            );
            lemma_positive_provenance_origin_blocks_eq(
                &post_deallocate, self,
            );
            lemma_positive_provenance_origin_transitive(
                &before, &post_deallocate, self,
            );
        }
    }

    // Keep this transition in its own solver context: it composes several
    // allocator/provenance frames and is otherwise sensitive to unrelated
    // growth in the crate's proof environment.
    #[verifier::spinoff_prover]
    #[verifier::rlimit(200)]
    pub fn commit_sample_for_request(
        &mut self,
        out: &mut EmittedTokens,
        emitted: &SampleResults,
        rid: RequestId,
        sample: &SampleResult,
    )
        requires
            cs_valid(old(self)),
            free_queue_valid(old(self)),
            persistent_provenance_closed(old(self)),
            residency_history_aligned(old(self)),
            old(self).num_blocks <= u64::MAX / BLOCK_SIZE,
            !old(self).waiting@.contains(rid),
            old(out)@.dom().subset_of(emitted@.dom()),
            forall|r: RequestId|
                #[trigger] old(out)@.contains_key(r)
                ==> old(out)@[r] == emitted@[r].token,
            emitted@.contains_key(rid),
            sample.token == emitted@[rid].token,
            old(self).live_requests@.contains_key(rid) ==> can_step(
                old(self).live_requests@[rid],
            ) && old(self).live_requests@[rid].generated_tokens@.len() < usize::MAX as int,
            // Capacity + queue-shape side conditions for the
            // alignment companion (see `append_headroom`).
            append_headroom(old(self), rid),
            slot_mapping_aligned(old(self)),
            pre_commit_tails_exclusive(old(self)),
            residency_running_aligned(old(self)),
            old(self).request_residency@.contains_key(rid)
                ==> old(self).running@.contains(rid),
        ensures
            cs_valid(final(self)),
            free_queue_valid(final(self)),
            persistent_provenance_closed(final(self)),
            positive_provenance_metadata_frame(old(self), final(self)),
            positive_provenance_origin(old(self), final(self)),
            registry_entries_from_pre(
                old(self), final(self), Seq::<u64>::empty(),
            ),
            final(self).num_blocks == old(self).num_blocks,
            final(self).waiting@ == old(self).waiting@,
            final(self).accepted_requests@ == old(self).accepted_requests@,
            forall|other: RequestId|
                other != rid && #[trigger] old(self).live_requests@.contains_key(other)
                ==> final(self).live_requests@.contains_key(other)
                    && final(self).live_requests@[other] == old(self).live_requests@[other],
            forall|r: RequestId|
                #[trigger] final(self).live_requests@.contains_key(r)
                ==> old(self).live_requests@.contains_key(r),
            final(out)@.dom().subset_of(emitted@.dom()),
            forall|r: RequestId|
                #[trigger] final(out)@.contains_key(r) ==> final(out)@[r] == emitted@[r].token,
            final(out)@.contains_key(rid),
            final(out)@[rid] == sample.token,
            final(out)@ == old(out)@.insert(rid, sample.token),
            // Per-rid evolution (2026-08-05): rid's own state takes the FULL
            // machine transition (append `sample` via the machine step), or is
            // removed if it finished; non-live rid leaves the map unchanged.
            old(self).live_requests@.contains_key(rid) ==> ({
                let pre = old(self).live_requests@[rid];
                if should_finish_after_append(pre, sample.token) {
                    !final(self).live_requests@.contains_key(rid)
                } else {
                    final(self).live_requests@.contains_key(rid)
                    && crate::proof::reference::request_machine::machine_step_transition_full(
                        pre, final(self).live_requests@[rid],
                        sample.sampler_state, sample.token)
                }
            }),
            // Running-queue evolution — membership survives the
            // commit exactly when the request did not finish.
            forall|r: RequestId| #[trigger] final(self).running@.contains(r)
                <==> (old(self).running@.contains(r)
                    && !(r == rid && old(self).live_requests@.contains_key(rid)
                        && should_finish_after_append(
                            old(self).live_requests@[rid], sample.token))),
            residency_history_aligned(final(self)),
            slot_mapping_aligned(final(self)),
            // Tail exclusivity — rid's post tail is a fresh (or
            // still-exclusive) hash-0 block; bystander tails are untouched.
            pre_commit_tails_exclusive(final(self)),
            old(self).live_requests@.contains_key(rid)
                && final(self).running@.contains(rid)
                ==> {
                    let ids = final(self).request_residency@[rid].block_ids@;
                    &&& final(self).request_residency@.contains_key(rid)
                    &&& ids.len() >= 1
                    &&& final(self).blocks@.contains_key(ids[ids.len() - 1])
                    &&& final(self).blocks@[ids[ids.len() - 1]].refcount == 1
                    &&& final(self).blocks@[ids[ids.len() - 1]].prefix_depth == 0
                    &&& final(self).blocks@[ids[ids.len() - 1]].hash_value == 0
                },
            forall|r: RequestId| r != rid
                && #[trigger] old(self).running@.contains(r)
                && old(self).live_requests@.contains_key(r)
                ==> {
                    let t = old(self).request_residency@[r].block_ids@[
                        old(self).request_residency@[r].block_ids@.len() - 1];
                    final(self).blocks@.contains_key(t)
                        && final(self).blocks@[t].tokens@ == old(self).blocks@[t].tokens@
                        && final(self).blocks@[t].refcount == old(self).blocks@[t].refcount
                        && final(self).blocks@[t].prefix_depth
                            == old(self).blocks@[t].prefix_depth
                        && final(self).blocks@[t].hash_value
                            == old(self).blocks@[t].hash_value
                },
            // Residency evolution: bystanders keep their
            // residency verbatim; the domain only shrinks; a surviving
            // committed request's block table EXTENDS its old one.
            forall|other: RequestId|
                other != rid && #[trigger] old(self).request_residency@.contains_key(other)
                ==> final(self).request_residency@.contains_key(other)
                    && final(self).request_residency@[other]
                        == old(self).request_residency@[other],
            forall|r: RequestId| #[trigger] final(self).request_residency@.contains_key(r)
                ==> old(self).request_residency@.contains_key(r),
            old(self).request_residency@.contains_key(rid)
                && final(self).request_residency@.contains_key(rid)
                ==> final(self).request_residency@[rid].block_ids@.len()
                        >= old(self).request_residency@[rid].block_ids@.len()
                    && final(self).request_residency@[rid].block_ids@.subrange(0,
                        old(self).request_residency@[rid].block_ids@.len() as int)
                        == old(self).request_residency@[rid].block_ids@
                    && final(self).request_residency@[rid].cached_prefix_blocks
                        == old(self).request_residency@[rid].cached_prefix_blocks,
            // Free-pool consumption is bounded by the request's own debt:
            // one block iff its history was block-aligned (full tail).
            final(self).free_blocks as int >= old(self).free_blocks as int
                - (if old(self).live_requests@.contains_key(rid)
                    && old(self).running@.contains(rid)
                    && history(old(self).live_requests@[rid]).len() as int
                        % (BLOCK_SIZE_SPEC as int) == 0 { 1int } else { 0int }),
            !old(self).live_requests@.contains_key(rid)
                ==> final(self).live_requests@ == old(self).live_requests@,
    {
        // Append/deallocation and the explicit queue-frame lemmas preserve
        // availability topology. The request transition only consumes that
        // contract, not the linked-queue representation.
        hide(free_queue_valid);
        record_emitted_token(out, emitted, rid, sample.token);

        let state_opt = self.live_requests.get(&rid);
        match state_opt {
            Some(state_ref) => {
                assert(old(self).live_requests@.contains_key(rid));
                assert(*state_ref == old(self).live_requests@[rid]);
                assert(can_step(*state_ref));
                let state = state_ref.clone();
                assert(can_step(state));
                proof {
                    lemma_eos_tokens_from_policy_eq(
                        state,
                        old(self).live_requests@[rid],
                    );
                    lemma_same_eos_tokens_contains(
                        eos_tokens(state),
                        eos_tokens(old(self).live_requests@[rid]),
                        sample.token,
                    );
                    assert(should_finish_after_append(state, sample.token)
                        == should_finish_after_append(
                            old(self).live_requests@[rid], sample.token,
                        ));
                }
                let has_residency = self.request_residency.contains_key(&rid);
                let finished = should_finish_after_append_exec(&state, sample.token);
                let next_state = append_generated_token(&state, sample.sampler_state, sample.token);
                // Package the request transition before entering the much
                // larger cache/residency proof. Lifecycle policy is copied
                // exactly but remains opaque to the outer solver context.
                assert(crate::proof::reference::request_machine::machine_step_transition_full(
                    state, next_state, sample.sampler_state, sample.token,
                )) by {
                    lemma_request_lifecycle_view_eq_from_fields(
                        next_state, state,
                    );
                }
                proof {
                    lemma_request_lifecycle_view_eq_from_fields(
                        state,
                        old(self).live_requests@[rid],
                    );
                    lemma_request_lifecycle_view_eq_transitive(
                        next_state,
                        state,
                        old(self).live_requests@[rid],
                    );
                    assert(crate::proof::reference::request_machine::machine_step_transition_full(
                        old(self).live_requests@[rid], next_state,
                        sample.sampler_state, sample.token,
                    ));
                }
                if has_residency {
                    self.append_token(rid, sample.token);
                }
                let ghost post_append = *self;
                proof {
                    if !has_residency {
                        assert(post_append.blocks@ == old(self).blocks@);
                        assert(post_append.hash_to_block@
                            == old(self).hash_to_block@);
                        lemma_persistent_provenance_closed_blocks_eq(
                            old(self), &post_append,
                        );
                        lemma_registry_entries_from_pre_blocks_eq(
                            old(self), &post_append,
                        );
                        lemma_positive_provenance_metadata_frame_blocks_eq(
                            old(self), &post_append,
                        );
                        lemma_positive_provenance_origin_blocks_eq(
                            old(self), &post_append,
                        );
                    }
                    assert(registry_entries_from_pre(
                        old(self), &post_append, Seq::<u64>::empty(),
                    ));
                    assert(persistent_provenance_closed(&post_append));
                    assert(positive_provenance_metadata_frame(
                        old(self), &post_append,
                    ));
                    assert(positive_provenance_origin(
                        old(self), &post_append,
                    ));
                }
                self.update_live_request(rid, next_state);
                let ghost post_update = *self;
                proof {
                    assert(post_update.blocks@ == post_append.blocks@);
                    assert(post_update.hash_to_block@
                        == post_append.hash_to_block@);
                    lemma_persistent_provenance_closed_blocks_eq(
                        &post_append, &post_update,
                    );
                    lemma_registry_entries_from_pre_blocks_eq(
                        &post_append, &post_update,
                    );
                    lemma_registry_entries_from_pre_transitive(
                        old(self), &post_append, &post_update,
                        Seq::<u64>::empty(),
                    );
                    lemma_positive_provenance_metadata_frame_blocks_eq(
                        &post_append, &post_update,
                    );
                    lemma_positive_provenance_metadata_frame_transitive(
                        old(self), &post_append, &post_update,
                    );
                    lemma_positive_provenance_origin_blocks_eq(
                        &post_append, &post_update,
                    );
                    lemma_positive_provenance_origin_transitive(
                        old(self), &post_append, &post_update,
                    );
                }
                if finished {
                    remove_request_id_from_queue(&mut self.running, rid);
                    assert(!self.running@.contains(rid));
                    assert(!self.waiting@.contains(rid));
                    let ghost pre_deallocate = *self;
                    proof {
                        assert(pre_deallocate.blocks@ == post_update.blocks@);
                        assert(pre_deallocate.hash_to_block@
                            == post_update.hash_to_block@);
                        lemma_persistent_provenance_closed_blocks_eq(
                            &post_update, &pre_deallocate,
                        );
                        lemma_registry_entries_from_pre_blocks_eq(
                            &post_update, &pre_deallocate,
                        );
                        lemma_registry_entries_from_pre_transitive(
                            old(self), &post_update, &pre_deallocate,
                            Seq::<u64>::empty(),
                        );
                        lemma_positive_provenance_metadata_frame_blocks_eq(
                            &post_update, &pre_deallocate,
                        );
                        lemma_positive_provenance_metadata_frame_transitive(
                            old(self), &post_update, &pre_deallocate,
                        );
                        lemma_positive_provenance_origin_blocks_eq(
                            &post_update, &pre_deallocate,
                        );
                        lemma_positive_provenance_origin_transitive(
                            old(self), &post_update, &pre_deallocate,
                        );
                        lemma_free_queue_valid_to_token(&post_update);
                        lemma_free_queue_valid_token_frame(
                            &post_update, &pre_deallocate,
                        );
                        lemma_free_queue_token_to_valid(&pre_deallocate);
                    }
                    self.deallocate(rid);
                    let ghost post_deallocate = *self;
                    proof {
                        assert(persistent_provenance_closed(&post_deallocate));
                        lemma_registry_entries_from_pre_transitive(
                            old(self), &pre_deallocate, &post_deallocate,
                            Seq::<u64>::empty(),
                        );
                        lemma_positive_provenance_metadata_frame_transitive(
                            old(self), &pre_deallocate, &post_deallocate,
                        );
                        lemma_positive_provenance_origin_transitive(
                            old(self), &pre_deallocate, &post_deallocate,
                        );
                    }
                    assert(!self.request_residency@.contains_key(rid));
                    self.remove_live_request_after_dequeue(rid);
                    proof {
                        assert(self.blocks@ == post_deallocate.blocks@);
                        assert(self.hash_to_block@
                            == post_deallocate.hash_to_block@);
                        lemma_persistent_provenance_closed_blocks_eq(
                            &post_deallocate, self,
                        );
                        lemma_registry_entries_from_pre_blocks_eq(
                            &post_deallocate, self,
                        );
                        lemma_registry_entries_from_pre_transitive(
                            old(self), &post_deallocate, self,
                            Seq::<u64>::empty(),
                        );
                        lemma_positive_provenance_metadata_frame_blocks_eq(
                            &post_deallocate, self,
                        );
                        lemma_positive_provenance_metadata_frame_transitive(
                            old(self), &post_deallocate, self,
                        );
                        lemma_positive_provenance_origin_blocks_eq(
                            &post_deallocate, self,
                        );
                        lemma_positive_provenance_origin_transitive(
                            old(self), &post_deallocate, self,
                        );
                    }
                } else {
                    proof {
                        assert(self.blocks@ == post_update.blocks@);
                        assert(self.hash_to_block@
                            == post_update.hash_to_block@);
                        lemma_registry_entries_from_pre_blocks_eq(
                            &post_update, self,
                        );
                        lemma_registry_entries_from_pre_transitive(
                            old(self), &post_update, self,
                            Seq::<u64>::empty(),
                        );
                        lemma_positive_provenance_metadata_frame_blocks_eq(
                            &post_update, self,
                        );
                        lemma_positive_provenance_metadata_frame_transitive(
                            old(self), &post_update, self,
                        );
                        lemma_positive_provenance_origin_blocks_eq(
                            &post_update, self,
                        );
                        lemma_positive_provenance_origin_transitive(
                            old(self), &post_update, self,
                        );
                    }
                }
                proof {
                    let bs = BLOCK_SIZE_SPEC as int;
                    let h0 = history(old(self).live_requests@[rid]).len() as int;
                    assert(history(next_state).len() as int == h0 + 1);
                    // ---- Alignment for every surviving running request. ----
                    assert forall|r: RequestId|
                        #[trigger] self.running@.contains(r)
                        && self.live_requests@.contains_key(r)
                        implies {
                            let hist = history(self.live_requests@[r]).len() as int;
                            let idsr = self.request_residency@[r].block_ids@;
                            &&& self.request_residency@.contains_key(r)
                            &&& idsr.len() >= 1
                            &&& self.blocks@.contains_key(idsr[idsr.len() - 1])
                            &&& hist == (idsr.len() - 1) * bs
                                + self.blocks@[idsr[idsr.len() - 1]].tokens@.len()
                            &&& self.blocks@[idsr[idsr.len() - 1]].tokens@.len() >= 1
                            &&& token_placement_prefix(
                                self.blocks@,
                                idsr,
                                history(self.live_requests@[r]),
                                hist,
                            )
                            &&& (self.blocks@[idsr[idsr.len() - 1]].refcount == 1
                                || self.blocks@[idsr[idsr.len() - 1]].tokens@.len() == bs)
                        }
                    by {
                        if r == rid {
                            assert(!finished);
                            assert(old(self).running@.contains(rid));
                            let ids0 = old(self).request_residency@[rid].block_ids@;
                            assert(old(self).request_residency@.contains_key(rid));
                            assert(has_residency);
                            let last0 = ids0[ids0.len() - 1];
                            let tail0 = old(self).blocks@[last0].tokens@.len() as int;
                            let hist0 = history(old(self).live_requests@[rid]);
                            assert(ids0.len() >= 1);
                            assert(old(self).blocks@.contains_key(last0));
                            assert(h0 == (ids0.len() - 1) * bs + tail0);
                            assert(token_placement_prefix(
                                old(self).blocks@, ids0, hist0, h0,
                            ));
                            assert(history(next_state) == hist0.push(sample.token));
                            assert(tail0 >= 1);
                            assert(tail0 <= bs) by {
                                assert(block_token_bound(old(self)));
                            }
                            assert(self.live_requests@[rid] == next_state);
                            assert(self.blocks@ == post_update.blocks@);
                            assert(self.request_residency@ == post_update.request_residency@);
                            assert(post_update.blocks@ == post_append.blocks@);
                            assert(post_update.request_residency@
                                == post_append.request_residency@);
                            if tail0 < bs {
                                assert(append_token_tail_append_success(
                                    old(self), &post_append, rid, sample.token));
                                assert(post_append.request_residency@[rid].block_ids@
                                    == ids0);
                                assert(post_append.blocks@[last0].tokens@
                                    == old(self).blocks@[last0].tokens@.push(sample.token));
                                assert(old(self).blocks@[last0].refcount == 1);
                                assert(post_append.blocks@[last0].refcount == 1);
                                lemma_token_placement_prefix_tail_append(
                                    old(self).blocks@,
                                    post_append.blocks@,
                                    ids0,
                                    hist0,
                                    sample.token,
                                );
                            } else {
                                assert(old(self).free_blocks > 0);
                                assert(append_token_new_tail_success(
                                    old(self), &post_append, rid, sample.token));
                                let ids1 = post_append.request_residency@[rid].block_ids@;
                                let nb = ids1[ids0.len() as int];
                                assert(ids1.len() == ids0.len() + 1);
                                assert(ids1[ids1.len() - 1] == nb);
                                assert(post_append.blocks@[nb].tokens@
                                    == seq![sample.token]);
                                assert(post_append.blocks@[nb].tokens@.len() == 1);
                                assert(post_append.blocks@[nb].refcount == 1);
                                assert(h0 == (ids0.len() - 1) * bs + bs);
                                assert((ids0.len() - 1) * bs + bs == ids0.len() * bs);
                                assert(h0 + 1 == (ids1.len() - 1) * bs + 1);
                                assert(ids1 =~= ids0.push(nb));
                                lemma_token_placement_prefix_new_tail(
                                    old(self).blocks@,
                                    post_append.blocks@,
                                    ids0,
                                    ids1,
                                    hist0,
                                    sample.token,
                                    nb,
                                );
                            }
                            assert(token_placement_prefix(
                                post_append.blocks@,
                                post_append.request_residency@[rid].block_ids@,
                                history(next_state),
                                h0 + 1,
                            ));
                            assert(self.live_requests@[rid] == next_state);
                            assert(self.blocks@ == post_append.blocks@);
                            assert(self.request_residency@[rid]
                                == post_append.request_residency@[rid]);
                            assert(token_placement_prefix(
                                self.blocks@,
                                self.request_residency@[rid].block_ids@,
                                history(self.live_requests@[rid]),
                                h0 + 1,
                            ));
                        } else {
                            // Bystander: frames + tail-block transport.
                            assert(old(self).running@.contains(r));
                            assert(old(self).live_requests@.contains_key(r));
                            assert(self.live_requests@[r] == old(self).live_requests@[r]);
                            let ids0 = old(self).request_residency@[r].block_ids@;
                            assert(old(self).request_residency@.contains_key(r));
                            let lastr = ids0[ids0.len() - 1];
                            let tail0 = old(self).blocks@[lastr].tokens@.len() as int;
                            assert(ids0.len() >= 1);
                            assert(old(self).blocks@.contains_key(lastr));
                            assert(tail0 >= 1);
                            assert(old(self).blocks@[lastr].refcount == 1 || tail0 == bs);
                            assert(ids0.contains(lastr));
                            assert(self.request_residency@.contains_key(r)
                                && self.request_residency@[r]
                                    == old(self).request_residency@[r]);
                            // Phase A: the append preserves r's tail block.
                            assert(post_append.blocks@.contains_key(lastr)
                                && post_append.blocks@[lastr] == old(self).blocks@[lastr])
                            by {
                                if has_residency {
                                    let ridsr = old(self).request_residency@[rid].block_ids@;
                                    if ridsr.contains(lastr) {
                                        // Shared block ⇒ two holders ⇒ full.
                                        assert(refcount_valid(old(self)));
                                        let holders = residency_holders_of(old(self), lastr);
                                        assert(holders.contains(r));
                                        assert(holders.contains(rid));
                                        lemma_two_holders(holders, r, rid);
                                        assert(old(self).blocks@[lastr].refcount as int >= 2);
                                        assert(tail0 == bs);
                                        assert(ridsr.len() >= 1);
                                        let rlast = ridsr[ridsr.len() - 1];
                                        assert(old(self).blocks@.contains_key(rlast)) by {
                                            assert(residency_blocks_in_range(old(self)));
                                        }
                                        let rtail =
                                            old(self).blocks@[rlast].tokens@.len() as int;
                                        assert(rtail <= bs) by {
                                            assert(block_token_bound(old(self)));
                                        }
                                        if rtail < bs {
                                            assert(append_token_tail_append_success(
                                                old(self), &post_append, rid, sample.token));
                                            assert(lastr != rlast);
                                        } else {
                                            assert(old(self).free_blocks > 0);
                                            assert(append_token_new_tail_success(
                                                old(self), &post_append, rid, sample.token));
                                        }
                                    }
                                }
                            }
                            // Phase B frames blocks.
                            assert(post_update.blocks@ == post_append.blocks@);
                            // Phase C: deallocation of a finished rid.
                            if finished {
                                assert(self.blocks@.contains_key(lastr)) by {
                                    assert(residency_blocks_in_range(self));
                                    assert(self.request_residency@[r].block_ids@ == ids0);
                                    assert(self.request_residency@[r].block_ids@[
                                        ids0.len() - 1] == lastr);
                                }
                                assert(self.blocks@[lastr].tokens@
                                    == post_update.blocks@[lastr].tokens@);
                                if tail0 < bs {
                                    assert(post_update.blocks@[lastr].refcount == 1);
                                    if post_update.request_residency@.contains_key(rid)
                                        && post_update.request_residency@[rid]
                                            .block_ids@.contains(lastr) {
                                        assert(refcount_valid(&post_update));
                                        let holders2 =
                                            residency_holders_of(&post_update, lastr);
                                        assert(post_update.request_residency@[r]
                                            .block_ids@.contains(lastr));
                                        assert(holders2.contains(r));
                                        assert(holders2.contains(rid));
                                        lemma_two_holders(holders2, r, rid);
                                        assert(false);
                                    }
                                    assert(self.blocks@[lastr]
                                        == post_update.blocks@[lastr]);
                                }
                            } else {
                                assert(self.blocks@ == post_update.blocks@);
                            }
                            // Every block in r's row survives token-identically.
                            // The only append write is rid's exclusive tail; a
                            // finished rid's deallocation may drop unrelated
                            // pages but preserves all surviving page contents.
                            assert forall|j: int| 0 <= j < ids0.len()
                                && #[trigger] old(self).blocks@.contains_key(ids0[j])
                                implies self.blocks@.contains_key(ids0[j])
                                    && self.blocks@[ids0[j]].tokens@
                                        == old(self).blocks@[ids0[j]].tokens@
                            by {
                                let bid = ids0[j];
                                assert(ids0.contains(bid));
                                assert(post_append.blocks@.contains_key(bid)
                                    && post_append.blocks@[bid]
                                        == old(self).blocks@[bid])
                                by {
                                    if has_residency {
                                        let ridsr = old(self)
                                            .request_residency@[rid].block_ids@;
                                        assert(ridsr.len() >= 1);
                                        let rlast = ridsr[ridsr.len() - 1];
                                        if bid == rlast {
                                            assert(refcount_valid(old(self)));
                                            let holders = residency_holders_of(old(self), bid);
                                            assert(holders.contains(r));
                                            assert(holders.contains(rid));
                                            lemma_two_holders(holders, r, rid);
                                            assert(old(self).blocks@[bid].refcount as int >= 2);
                                            assert(old(self).running@.contains(rid));
                                            assert(pre_commit_tails_exclusive(old(self)));
                                            assert(old(self).blocks@[rlast].refcount == 1);
                                            assert(false);
                                        }
                                        let rtail = old(self).blocks@[rlast].tokens@.len() as int;
                                        assert(rtail <= bs) by {
                                            assert(block_token_bound(old(self)));
                                        }
                                        if rtail < bs {
                                            assert(append_token_tail_append_success(
                                                old(self), &post_append, rid, sample.token));
                                        } else {
                                            assert(old(self).free_blocks > 0);
                                            assert(append_token_new_tail_success(
                                                old(self), &post_append, rid, sample.token));
                                        }
                                    } else {
                                        assert(post_append.blocks@ == old(self).blocks@);
                                    }
                                }
                                assert(post_update.blocks@ == post_append.blocks@);
                                if finished {
                                    assert(self.request_residency@.contains_key(r));
                                    assert(self.request_residency@[r].block_ids@ == ids0);
                                    assert(self.blocks@.contains_key(bid)) by {
                                        assert(residency_blocks_in_range(self));
                                    }
                                    assert(self.blocks@[bid].tokens@
                                        == post_update.blocks@[bid].tokens@);
                                } else {
                                    assert(self.blocks@ == post_update.blocks@);
                                }
                            }
                            let hist0r = history(old(self).live_requests@[r]);
                            assert(token_placement_prefix(
                                old(self).blocks@,
                                ids0,
                                hist0r,
                                hist0r.len() as int,
                            ));
                            lemma_token_placement_prefix_transfer_for_ids(
                                old(self).blocks@,
                                self.blocks@,
                                ids0,
                                hist0r,
                                hist0r.len() as int,
                            );
                            assert(history(self.live_requests@[r]) == hist0r);
                        }
                    }
                    // ---- Free-pool consumption bound. ----
                    assert(self.free_blocks as int >= old(self).free_blocks as int
                        - (if old(self).live_requests@.contains_key(rid)
                            && old(self).running@.contains(rid)
                            && h0 % bs == 0 { 1int } else { 0int }))
                    by {
                        if has_residency {
                            assert(old(self).running@.contains(rid));
                            assert(old(self).live_requests@.contains_key(rid));
                            let ids0 = old(self).request_residency@[rid].block_ids@;
                            let last0 = ids0[ids0.len() - 1];
                            let tail0 = old(self).blocks@[last0].tokens@.len() as int;
                            assert(ids0.len() >= 1);
                            assert(h0 == (ids0.len() - 1) * bs + tail0);
                            assert(tail0 >= 1);
                            assert(tail0 <= bs) by {
                                assert(block_token_bound(old(self)));
                            }
                            lemma_tail_mod(h0, ids0.len() as int, tail0);
                            if tail0 == bs {
                                assert(old(self).free_blocks > 0);
                                assert(append_token_new_tail_success(
                                    old(self), &post_append, rid, sample.token));
                                assert(post_append.free_blocks as int
                                    == old(self).free_blocks as int - 1);
                            } else {
                                assert(append_token_tail_append_success(
                                    old(self), &post_append, rid, sample.token));
                                assert(post_append.free_blocks == old(self).free_blocks);
                                assert(!(h0 % bs == 0));
                            }
                        } else {
                            assert(post_append.free_blocks == old(self).free_blocks);
                        }
                        assert(post_update.free_blocks == post_append.free_blocks);
                        if finished {
                            assert(self.free_blocks >= post_update.free_blocks);
                        } else {
                            assert(self.free_blocks == post_update.free_blocks);
                        }
                    }
                    // ---- Slot-mapping alignment: the append refreshed
                    // rid's mapping to its new last position; bystanders
                    // are framed. ----
                    assert forall|r: RequestId|
                        #[trigger] self.running@.contains(r)
                        && self.live_requests@.contains_key(r)
                        implies {
                            let hist = history(self.live_requests@[r]).len() as int;
                            let idsr = self.request_residency@[r].block_ids@;
                            let smr = self.request_residency@[r].slot_mapping@;
                            &&& self.request_residency@.contains_key(r)
                            &&& smr.len() >= 1
                            &&& hist >= 1
                            &&& smr[smr.len() - 1] as int
                                == crate::proof::tensor::geometry::block_table_slot(idsr,
                                    (hist - 1) as nat) as int
                        }
                    by {
                        if r == rid {
                            assert(!finished);
                            assert(old(self).running@.contains(rid));
                            let ids0 = old(self).request_residency@[rid].block_ids@;
                            assert(old(self).request_residency@.contains_key(rid));
                            assert(has_residency);
                            let last0 = ids0[ids0.len() - 1];
                            let tail0 = old(self).blocks@[last0].tokens@.len() as int;
                            assert(ids0.len() >= 1);
                            assert(h0 == (ids0.len() - 1) * bs + tail0);
                            assert(tail0 >= 1);
                            assert(tail0 <= bs) by {
                                assert(block_token_bound(old(self)));
                            }
                            assert(self.live_requests@[rid] == next_state);
                            assert(history(self.live_requests@[rid]).len() as int
                                == h0 + 1);
                            assert(self.request_residency@[rid]
                                == post_append.request_residency@[rid]);
                            if tail0 < bs {
                                assert(append_token_tail_append_success(
                                    old(self), &post_append, rid, sample.token));
                                let smr = self.request_residency@[rid].slot_mapping@;
                                let idsr = self.request_residency@[rid].block_ids@;
                                assert(idsr == ids0);
                                assert(smr.len() == 1);
                                assert(smr[0] == last0 * BLOCK_SIZE
                                    + old(self).blocks@[last0].tokens@.len() as u64);
                                // position h0 sits in the (unchanged) tail page
                                vstd::arithmetic::div_mod::
                                    lemma_fundamental_div_mod_converse_div(
                                        h0, bs, ids0.len() as int - 1, tail0);
                                vstd::arithmetic::div_mod::
                                    lemma_fundamental_div_mod_converse_mod(
                                        h0, bs, ids0.len() as int - 1, tail0);
                                assert(h0 / bs == ids0.len() as int - 1);
                                assert(h0 % bs == tail0);
                                assert(crate::proof::tensor::geometry::block_table_slot(idsr,
                                    h0 as nat) as int
                                    == idsr[ids0.len() - 1] as int * bs + tail0);
                            } else {
                                assert(old(self).free_blocks > 0);
                                assert(append_token_new_tail_success(
                                    old(self), &post_append, rid, sample.token));
                                let smr = self.request_residency@[rid].slot_mapping@;
                                let idsr = self.request_residency@[rid].block_ids@;
                                let nb2 = idsr[ids0.len() as int];
                                assert(smr.len() == 1);
                                assert(smr[0] == nb2 * BLOCK_SIZE);
                                assert(h0 == ids0.len() as int * bs);
                                vstd::arithmetic::div_mod::
                                    lemma_fundamental_div_mod_converse_div(
                                        h0, bs, ids0.len() as int, 0);
                                vstd::arithmetic::div_mod::
                                    lemma_fundamental_div_mod_converse_mod(
                                        h0, bs, ids0.len() as int, 0);
                                assert(h0 / bs == ids0.len() as int);
                                assert(h0 % bs == 0);
                                assert(crate::proof::tensor::geometry::block_table_slot(idsr,
                                    h0 as nat) as int
                                    == idsr[ids0.len() as int] as int * bs);
                            }
                        } else {
                            assert(old(self).running@.contains(r));
                            assert(old(self).live_requests@.contains_key(r));
                            assert(self.live_requests@[r]
                                == old(self).live_requests@[r]);
                            assert(old(self).request_residency@.contains_key(r));
                            assert(self.request_residency@.contains_key(r)
                                && self.request_residency@[r]
                                    == old(self).request_residency@[r]);
                        }
                    }
                    // ---- Tail exclusivity. ----
                    assert forall|r: RequestId| r != rid
                        && #[trigger] old(self).running@.contains(r)
                        && old(self).live_requests@.contains_key(r)
                        implies {
                            let t = old(self).request_residency@[r].block_ids@[
                                old(self).request_residency@[r].block_ids@.len() - 1];
                            self.blocks@.contains_key(t)
                                && self.blocks@[t].tokens@
                                    == old(self).blocks@[t].tokens@
                                && self.blocks@[t].refcount
                                    == old(self).blocks@[t].refcount
                                && self.blocks@[t].prefix_depth
                                    == old(self).blocks@[t].prefix_depth
                                && self.blocks@[t].hash_value
                                    == old(self).blocks@[t].hash_value
                        }
                    by {
                        let idsr = old(self).request_residency@[r].block_ids@;
                        let t = idsr[idsr.len() - 1];
                        assert(old(self).request_residency@.contains_key(r));
                        assert(idsr.len() >= 1);
                        assert(old(self).blocks@.contains_key(t));
                        assert(old(self).blocks@[t].refcount == 1);
                        // Phase A: rid's append never touches r's tail — the
                        // tail is exclusively held by r.
                        assert(post_append.blocks@.contains_key(t)
                            && post_append.blocks@[t] == old(self).blocks@[t])
                        by {
                            if has_residency {
                                let ridsr = old(self)
                                    .request_residency@[rid].block_ids@;
                                if ridsr.contains(t) {
                                    assert(refcount_valid(old(self)));
                                    let holders = residency_holders_of(old(self), t);
                                    assert(idsr[idsr.len() - 1] == t);
                                    assert(idsr.contains(t));
                                    assert(holders.contains(r));
                                    assert(holders.contains(rid));
                                    lemma_two_holders(holders, r, rid);
                                    assert(false);
                                }
                            }
                        }
                        assert(post_update.blocks@ == post_append.blocks@);
                        if finished {
                            // r's tail is not among rid's deallocated blocks
                            // (sole holder), so it survives verbatim.
                            if post_update.request_residency@.contains_key(rid)
                                && post_update.request_residency@[rid]
                                    .block_ids@.contains(t) {
                                assert(refcount_valid(&post_update));
                                let holders2 = residency_holders_of(&post_update, t);
                                assert(post_update.request_residency@.contains_key(r)
                                    && post_update.request_residency@[r]
                                        == old(self).request_residency@[r]);
                                assert(post_update.request_residency@[r]
                                    .block_ids@.contains(t));
                                assert(holders2.contains(r));
                                assert(holders2.contains(rid));
                                lemma_two_holders(holders2, r, rid);
                                assert(post_update.blocks@[t].refcount as int >= 2);
                                assert(false);
                            }
                            assert(self.blocks@.contains_key(t)) by {
                                assert(residency_blocks_in_range(self));
                                assert(self.request_residency@.contains_key(r));
                                assert(self.request_residency@[r].block_ids@[
                                    idsr.len() - 1] == t);
                            }
                            assert(self.blocks@[t] == post_update.blocks@[t]);
                        } else {
                            assert(self.blocks@ == post_update.blocks@);
                        }
                    }
                    // rid's own post tail: fresh new_tail or preserved
                    // hash-0 partial tail.
                    if !finished && has_residency {
                        assert(old(self).running@.contains(rid));
                        let ids0 = old(self).request_residency@[rid].block_ids@;
                        let last0 = ids0[ids0.len() - 1];
                        let tail0 = old(self).blocks@[last0].tokens@.len() as int;
                        assert(tail0 <= bs) by {
                            assert(block_token_bound(old(self)));
                        }
                        if tail0 < bs {
                            assert(append_token_tail_append_success(
                                old(self), &post_append, rid, sample.token));
                            assert(old(self).blocks@[last0].prefix_depth == 0) by {
                                if old(self).blocks@[last0].prefix_depth > 0 {
                                    reveal(persistent_provenance_closed);
                                    assert(old(self).blocks@[last0].tokens@.len() == bs);
                                }
                            }
                            assert(post_append.blocks@[last0].prefix_depth == 0);
                            assert(old(self).blocks@[last0].hash_value == 0);
                            assert(old(self).blocks@[last0].refcount == 1);
                        } else {
                            assert(old(self).free_blocks > 0);
                            assert(append_token_new_tail_success(
                                old(self), &post_append, rid, sample.token));
                            let ids1 = post_append.request_residency@[rid].block_ids@;
                            let nb = ids1[ids0.len() as int];
                            assert(post_append.blocks@[nb].prefix_depth == 0);
                        }
                    }
                    // pce(self): assembled from the two facts above.
                    assert forall|r: RequestId|
                        #[trigger] self.running@.contains(r)
                        && self.live_requests@.contains_key(r)
                        implies {
                            let ids = self.request_residency@[r].block_ids@;
                            &&& self.request_residency@.contains_key(r)
                            &&& ids.len() >= 1
                            &&& self.blocks@.contains_key(ids[ids.len() - 1])
                            &&& self.blocks@[ids[ids.len() - 1]].refcount == 1
                            &&& (self.blocks@[ids[ids.len() - 1]].tokens@.len() < bs
                                ==> self.blocks@[ids[ids.len() - 1]].hash_value == 0)
                        }
                    by {
                        if r == rid {
                            assert(!finished);
                        } else {
                            assert(old(self).running@.contains(r));
                            assert(old(self).live_requests@.contains_key(r));
                            assert(self.request_residency@[r]
                                == old(self).request_residency@[r]);
                        }
                    }
                    // ---- Residency extension for a surviving rid. ----
                    if !finished && has_residency {
                        assert(old(self).running@.contains(rid));
                        assert(old(self).live_requests@.contains_key(rid));
                        let ids0 = old(self).request_residency@[rid].block_ids@;
                        let last0 = ids0[ids0.len() - 1];
                        let tail0 = old(self).blocks@[last0].tokens@.len() as int;
                        assert(ids0.len() >= 1);
                        assert(tail0 <= bs) by {
                            assert(block_token_bound(old(self)));
                        }
                        assert(self.request_residency@[rid]
                            == post_append.request_residency@[rid]);
                        if tail0 < bs {
                            assert(append_token_tail_append_success(
                                old(self), &post_append, rid, sample.token));
                            assert(self.request_residency@[rid].block_ids@ == ids0);
                            assert(ids0.subrange(0, ids0.len() as int) =~= ids0);
                        } else {
                            assert(old(self).free_blocks > 0);
                            assert(append_token_new_tail_success(
                                old(self), &post_append, rid, sample.token));
                            assert(self.request_residency@[rid].block_ids@
                                .subrange(0, ids0.len() as int) == ids0);
                        }
                    }
                }
            },
            None => {
                proof {
                    assert(self.blocks@ == old(self).blocks@);
                    assert(self.hash_to_block@ == old(self).hash_to_block@);
                    lemma_persistent_provenance_closed_blocks_eq(
                        old(self), self,
                    );
                    lemma_registry_entries_from_pre_blocks_eq(
                        old(self), self,
                    );
                    lemma_positive_provenance_metadata_frame_blocks_eq(
                        old(self), self,
                    );
                    lemma_positive_provenance_origin_blocks_eq(old(self), self);
                }
            },
        }

        assert(out@.dom().subset_of(emitted@.dom()));
        assert forall|r: RequestId|
            #[trigger] out@.contains_key(r) implies out@[r] == emitted@[r].token
        by {
        }
        assert(out@.contains_key(rid));
        assert(out@[rid] == sample.token);
    }
}

} // verus!
