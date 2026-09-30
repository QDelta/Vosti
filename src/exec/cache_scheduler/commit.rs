// Verified whole-step commit transition.

use super::*;

verus! {
impl CacheScheduler {
    // Commit a step: append emitted tokens to live requests, free / preempt
    // blocks, update queues.
    #[verifier::spinoff_prover]
    #[verifier::rlimit(300)]
    pub fn commit(
        &mut self,
        plan: &StepPlan,
        emitted: &SampleResults,
    ) -> (out: EmittedTokens)
        requires
            cs_valid(old(self)),
            free_queue_valid(old(self)),
            persistent_provenance_closed(old(self)),
            step_plan_commit_ready(old(self), plan),
            plan.sample_mask@.len() == plan.scheduled_ids@.len(),
            residency_history_aligned(old(self)),
            slot_mapping_aligned(old(self)),
            pre_commit_tails_exclusive(old(self)),
            residency_running_aligned(old(self)),
            // Unscheduled running requests still carry the
            // boundary provenance-free tail stamp (the plan never touches
            // them)...
            forall|r: RequestId| #[trigger] old(self).running@.contains(r)
                && old(self).live_requests@.contains_key(r)
                && !plan.scheduled_ids@.contains(r)
                ==> old(self).blocks@[old(self).request_residency@[r].block_ids@[
                        old(self).request_residency@[r].block_ids@.len() - 1]]
                    .hash_value == 0
                    && old(self).blocks@[old(self).request_residency@[r].block_ids@[
                        old(self).request_residency@[r].block_ids@.len() - 1]]
                        .prefix_depth == 0,
            // The sample map follows the row policy exactly.  KV-only rows
            // must still be in their pre-generation phase: parking keeps the
            // live RequestState unchanged and represents progress in the
            // persistent prefix cache.
            forall|k: int| #![trigger plan.scheduled_ids@[k], plan.sample_mask@[k]]
                0 <= k < plan.scheduled_ids@.len()
                ==> (emitted@.contains_key(plan.scheduled_ids@[k])
                        <==> plan.sample_mask@[k]),
            forall|k: int| #![trigger plan.scheduled_ids@[k], plan.sample_mask@[k]]
                0 <= k < plan.scheduled_ids@.len()
                    && !plan.sample_mask@[k]
                ==> old(self).live_requests@.contains_key(plan.scheduled_ids@[k])
                    && old(self).live_requests@[plan.scheduled_ids@[k]]
                        .generated_tokens@.len() == 0,
            // Capacity side condition: enough free blocks for every
            // scheduled request whose tail is exactly full (see
            // `commit_headroom`).
            commit_headroom(old(self), plan.scheduled_ids@),
            // Scheduled requests are running at commit entry (plan ensures
            // running ⊇ scheduled).
            forall|k: int|
                #![trigger plan.scheduled_ids@[k]]
                0 <= k < plan.scheduled_ids@.len()
                ==> old(self).running@.contains(plan.scheduled_ids@[k]),
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
            final(self).accepted_requests@ == old(self).accepted_requests@,
            forall|r: RequestId| #[trigger] final(self).waiting@.contains(r)
                <==> (old(self).waiting@.contains(r)
                    || step_plan_parks(plan, r)),
            out@.dom().subset_of(emitted@.dom()),
            forall|rid: RequestId|
                #[trigger] out@.contains_key(rid)
                ==> out@.contains_key(rid) && out@[rid] == emitted@[rid].token,
            // commit never admits new requests; `live_requests` only shrinks
            // (finished requests are removed).  Needed by `refinement_step` to
            // preserve `shared_rid_keyset`.  Already maintained by the loop.
            forall|r: RequestId|
                #[trigger] final(self).live_requests@.contains_key(r)
                ==> old(self).live_requests@.contains_key(r),
            // Full step characterization (2026-08-05) — the facts
            // `engine_step_relation` previously attributed to commit on trust:
            // out's domain is drawn from the scheduled ids; every surviving
            // request either took the FULL machine transition (if committed)
            // or is untouched; and an old live request survives iff it did not
            // finish by appending its committed token.
            forall|r: RequestId| #[trigger] out@.contains_key(r) ==>
                step_plan_emits(plan, r),
            forall|r: RequestId| #[trigger] final(self).live_requests@.contains_key(r) ==>
                if out@.contains_key(r) {
                    crate::proof::reference::request_machine::machine_step_transition_full(
                        old(self).live_requests@[r], final(self).live_requests@[r],
                        emitted@[r].sampler_state, emitted@[r].token)
                } else {
                    final(self).live_requests@[r] == old(self).live_requests@[r]
                },
            forall|r: RequestId| #[trigger] old(self).live_requests@.contains_key(r) ==>
                (final(self).live_requests@.contains_key(r) <==>
                    !(out@.contains_key(r)
                      && should_finish_after_append(old(self).live_requests@[r], out@[r]))),
            forall|k: int|
                #![trigger plan.scheduled_ids@[k], plan.sample_mask@[k]]
                0 <= k < plan.scheduled_ids@.len()
                && plan.sample_mask@[k]
                ==> out@.contains_key(plan.scheduled_ids@[k]),
            // Running-queue evolution — membership survives the
            // whole commit exactly when the request did not finish.
            forall|r: RequestId| #[trigger] final(self).running@.contains(r)
                <==> (old(self).running@.contains(r)
                    && !step_plan_parks(plan, r)
                    && !(out@.contains_key(r)
                        && should_finish_after_append(
                            old(self).live_requests@[r], out@[r]))),
            residency_history_aligned(final(self)),
            slot_mapping_aligned(final(self)),
            tail_write_exclusive(final(self)),
            residency_running_aligned(final(self)),
            // Residency evolution: uncommitted requests
            // keep their residency verbatim; committed survivors EXTEND
            // their block tables; the domain only shrinks.
            forall|r: RequestId|
                !out@.contains_key(r)
                && !step_plan_parks(plan, r)
                && #[trigger] old(self).request_residency@.contains_key(r)
                ==> final(self).request_residency@.contains_key(r)
                    && final(self).request_residency@[r]
                        == old(self).request_residency@[r],
            forall|r: RequestId|
                #[trigger] out@.contains_key(r)
                && old(self).request_residency@.contains_key(r)
                && final(self).request_residency@.contains_key(r)
                ==> final(self).request_residency@[r].block_ids@.len()
                        >= old(self).request_residency@[r].block_ids@.len()
                    && final(self).request_residency@[r].block_ids@.subrange(0,
                        old(self).request_residency@[r].block_ids@.len() as int)
                        == old(self).request_residency@[r].block_ids@
                    && final(self).request_residency@[r].cached_prefix_blocks
                        == old(self).request_residency@[r].cached_prefix_blocks,
            forall|r: RequestId| #[trigger] final(self).request_residency@.contains_key(r)
                ==> old(self).request_residency@.contains_key(r),
    {
        // Queue topology is established and preserved by the called scheduler
        // transitions. Commit composes that contract without inspecting links
        // or ordering; keep those quantifiers out of its request-history loop.
        hide(free_queue_valid);
        let mut out = HashMapWithView::<u64, TokenId>::new();
        proof {
            assert(plan.scheduled_ids@.subrange(0, plan.scheduled_ids@.len() as int)
                =~= plan.scheduled_ids@);
            lemma_registry_entries_from_pre_refl(old(self));
            lemma_positive_provenance_metadata_frame_refl(old(self));
            lemma_positive_provenance_origin_refl(old(self));
        }
        let mut i: usize = 0;
        while i < plan.scheduled_ids.len()
            invariant
                i <= plan.scheduled_ids@.len(),
                cs_valid(self),
                free_queue_valid(self),
                persistent_provenance_closed(self),
                positive_provenance_metadata_frame(old(self), self),
                positive_provenance_origin(old(self), self),
                registry_entries_from_pre(
                    old(self), self, Seq::<u64>::empty(),
                ),
                self.num_blocks == old(self).num_blocks,
                forall|r: RequestId| #[trigger] self.waiting@.contains(r)
                    <==> (old(self).waiting@.contains(r)
                        || step_plan_parks_before(plan, r, i as int)),
                self.accepted_requests@ == old(self).accepted_requests@,
                step_plan_commit_ready(old(self), plan),
                plan.sample_mask@.len() == plan.scheduled_ids@.len(),
                residency_history_aligned(self),
                slot_mapping_aligned(self),
                pre_commit_tails_exclusive(self),
                pre_commit_tails_exclusive(old(self)),
                residency_running_aligned(self),
                forall|r: RequestId| #[trigger] old(self).running@.contains(r)
                    && old(self).live_requests@.contains_key(r)
                    && !plan.scheduled_ids@.contains(r)
                    ==> {
                        let t = old(self).request_residency@[r]
                            .block_ids@[old(self).request_residency@[r]
                                .block_ids@.len() - 1];
                        old(self).blocks@[t].hash_value == 0
                            && old(self).blocks@[t].prefix_depth == 0
                    },
                forall|k2: int|
                    #![trigger plan.scheduled_ids@[k2], plan.sample_mask@[k2]]
                    0 <= k2 < plan.scheduled_ids@.len()
                    ==> (emitted@.contains_key(plan.scheduled_ids@[k2])
                            <==> plan.sample_mask@[k2]),
                forall|k2: int|
                    #![trigger plan.scheduled_ids@[k2], plan.sample_mask@[k2]]
                    0 <= k2 < plan.scheduled_ids@.len()
                        && !plan.sample_mask@[k2]
                    ==> old(self).live_requests@
                            .contains_key(plan.scheduled_ids@[k2])
                        && old(self).live_requests@[plan.scheduled_ids@[k2]]
                            .generated_tokens@.len() == 0,
                // Committed tails are boundary-good; uncommitted
                // tails keep their pre-commit hash.
                forall|r: RequestId| #[trigger] out@.contains_key(r)
                    && self.running@.contains(r)
                    && self.live_requests@.contains_key(r)
                    ==> {
                        let ids = self.request_residency@[r].block_ids@;
                        &&& self.request_residency@.contains_key(r)
                        &&& ids.len() >= 1
                        &&& self.blocks@.contains_key(ids[ids.len() - 1])
                        &&& self.blocks@[ids[ids.len() - 1]].refcount == 1
                        &&& self.blocks@[ids[ids.len() - 1]].prefix_depth == 0
                        &&& self.blocks@[ids[ids.len() - 1]].hash_value == 0
                    },
                forall|r: RequestId| #[trigger] self.running@.contains(r)
                    && self.live_requests@.contains_key(r)
                    && !out@.contains_key(r)
                    ==> {
                        let t = old(self).request_residency@[r].block_ids@[
                            old(self).request_residency@[r].block_ids@.len() - 1];
                        self.request_residency@[r]
                                == old(self).request_residency@[r]
                            && self.blocks@.contains_key(t)
                            && self.blocks@[t].prefix_depth
                                == old(self).blocks@[t].prefix_depth
                            && self.blocks@[t].hash_value
                                == old(self).blocks@[t].hash_value
                    },
                forall|r: RequestId|
                    !out@.contains_key(r)
                    && !step_plan_parks_before(plan, r, i as int)
                    && #[trigger] old(self).request_residency@.contains_key(r)
                    ==> self.request_residency@.contains_key(r)
                        && self.request_residency@[r]
                            == old(self).request_residency@[r],
                forall|r: RequestId|
                    #[trigger] out@.contains_key(r)
                    && old(self).request_residency@.contains_key(r)
                    && self.request_residency@.contains_key(r)
                    ==> self.request_residency@[r].block_ids@.len()
                            >= old(self).request_residency@[r].block_ids@.len()
                        && self.request_residency@[r].block_ids@.subrange(0,
                            old(self).request_residency@[r].block_ids@.len() as int)
                            == old(self).request_residency@[r].block_ids@
                        && self.request_residency@[r].cached_prefix_blocks
                            == old(self).request_residency@[r].cached_prefix_blocks,
                forall|r: RequestId| #[trigger] self.request_residency@.contains_key(r)
                    ==> old(self).request_residency@.contains_key(r),
                full_tail_debt(self.live_requests@,
                    plan.scheduled_ids@.subrange(i as int,
                        plan.scheduled_ids@.len() as int))
                    <= self.free_blocks as int,
                forall|k: int|
                    #![trigger plan.scheduled_ids@[k]]
                    0 <= k < plan.scheduled_ids@.len()
                    ==> old(self).running@.contains(plan.scheduled_ids@[k]),
                out@.dom().subset_of(emitted@.dom()),
                forall|r: RequestId|
                    #[trigger] out@.contains_key(r) ==> out@[r] == emitted@[r].token,
                forall|r: RequestId|
                    #[trigger] self.live_requests@.contains_key(r)
                    ==> old(self).live_requests@.contains_key(r),
                forall|j: int|
                    #![trigger plan.scheduled_ids@[j]]
                    i <= j < plan.scheduled_ids@.len()
                    && old(self).live_requests@.contains_key(plan.scheduled_ids@[j])
                    ==> self.live_requests@.contains_key(plan.scheduled_ids@[j])
                        && self.live_requests@[plan.scheduled_ids@[j]]
                            == old(self).live_requests@[plan.scheduled_ids@[j]],
                forall|r: RequestId| #[trigger] out@.contains_key(r) ==>
                    exists|k: int| #![trigger plan.scheduled_ids@[k], plan.sample_mask@[k]]
                        0 <= k < i as int
                            && plan.scheduled_ids@[k] == r
                            && plan.sample_mask@[k],
                forall|r: RequestId| #[trigger] self.live_requests@.contains_key(r) ==>
                    if out@.contains_key(r) {
                        crate::proof::reference::request_machine::machine_step_transition_full(
                            old(self).live_requests@[r], self.live_requests@[r],
                            emitted@[r].sampler_state, emitted@[r].token)
                    } else {
                        self.live_requests@[r] == old(self).live_requests@[r]
                    },
                forall|r: RequestId| #[trigger] old(self).live_requests@.contains_key(r) ==>
                    (self.live_requests@.contains_key(r) <==>
                        !(out@.contains_key(r)
                          && should_finish_after_append(
                              old(self).live_requests@[r], out@[r]))),
                forall|r: RequestId| #[trigger] self.running@.contains(r)
                    <==> (old(self).running@.contains(r)
                        && !step_plan_parks_before(plan, r, i as int)
                        && !(out@.contains_key(r)
                            && should_finish_after_append(
                                old(self).live_requests@[r], out@[r]))),
                forall|k: int|
                    #![trigger plan.scheduled_ids@[k], plan.sample_mask@[k]]
                    0 <= k < i as int
                    && plan.sample_mask@[k]
                    ==> out@.contains_key(plan.scheduled_ids@[k]),
            decreases plan.scheduled_ids@.len() - i
        {
            let rid = plan.scheduled_ids[i];
            let sample_opt = emitted.get(&rid);
            match sample_opt {
                Some(sample) => {
                    assert(emitted@.contains_key(rid));
                    assert(plan.sample_mask@[i as int]);
                    assert(!self.waiting@.contains(rid)) by {
                        if step_plan_parks_before(plan, rid, i as int) {
                            let k = choose|k: int|
                                0 <= k < i as int
                                    && plan.scheduled_ids@[k] == rid
                                    && !plan.sample_mask@[k];
                            assert(plan.scheduled_ids@[k]
                                == plan.scheduled_ids@[i as int]);
                            assert(false);
                        }
                        assert(!old(self).waiting@.contains(rid));
                    }
                    assert(self.num_blocks <= u64::MAX / BLOCK_SIZE);
                    assert(old(self).live_requests@.contains_key(rid) ==> can_step(
                        old(self).live_requests@[rid],
                    ) && old(self).live_requests@[rid].generated_tokens@.len()
                        < usize::MAX as int);
                    assert(self.live_requests@.contains_key(rid) ==> can_step(
                        self.live_requests@[rid],
                    ) && self.live_requests@[rid].generated_tokens@.len() < usize::MAX as int)
                    by {
                        if self.live_requests@.contains_key(rid) {
                            assert(old(self).live_requests@.contains_key(rid));
                            assert(self.live_requests@[rid] == old(self).live_requests@[rid]);
                        }
                    }
                    let ghost pre_iter = *self;
                    let ghost pre_out = out@;
                    proof {
                        // rid was not committed yet: out's domain is drawn from
                        // the processed prefix, which excludes index i
                        // (no_duplicates).
                        if out@.contains_key(rid) {
                            let k = choose|k: int| 0 <= k < i as int
                                && plan.scheduled_ids@[k] == rid;
                            assert(plan.scheduled_ids@[k] == plan.scheduled_ids@[i as int]);
                            assert(false);
                        }
                        // Hence rid's live state is still the commit-entry state.
                        if pre_iter.live_requests@.contains_key(rid) {
                            assert(pre_iter.live_requests@[rid]
                                == old(self).live_requests@[rid]);
                        }
                        // rid is still running (and hence live) at this
                        // iteration: it was running at entry and cannot have
                        // finished (not yet committed).
                        assert(plan.scheduled_ids@[i as int] == rid);
                        assert(old(self).running@.contains(rid));
                        assert(self.running@.contains(rid));
                        assert(self.live_requests@.contains_key(rid)) by {
                            assert(live_covers_queue(self));
                        }
                        assert(self.request_residency@.contains_key(rid)
                            ==> self.running@.contains(rid));
                        // Per-request headroom from the aggregate debt.
                        assert(append_headroom(self, rid)) by {
                            let idsr = self.request_residency@[rid].block_ids@;
                            let bsz = BLOCK_SIZE_SPEC as int;
                            if self.request_residency@.contains_key(rid)
                                && idsr.len() > 0
                                && self.blocks@.contains_key(idsr[idsr.len() - 1])
                                && self.blocks@[idsr[idsr.len() - 1]].tokens@.len()
                                    == bsz {
                                let hist =
                                    history(self.live_requests@[rid]).len() as int;
                                assert(hist == (idsr.len() - 1) * bsz
                                    + self.blocks@[idsr[idsr.len() - 1]].tokens@.len());
                                lemma_tail_mod(hist, idsr.len() as int, bsz);
                                assert(hist % bsz == 0);
                                lemma_debt_head(self.live_requests@,
                                    plan.scheduled_ids@, i as int);
                                assert(full_tail_debt(self.live_requests@,
                                    plan.scheduled_ids@.subrange(i as int,
                                        plan.scheduled_ids@.len() as int)) >= 1);
                            }
                        }
                    }
                    self.commit_sample_for_request(&mut out, emitted, rid, sample);
                    proof {
                        assert(registry_entries_from_pre(
                            &pre_iter, self, Seq::<u64>::empty(),
                        ));
                        lemma_registry_entries_from_pre_transitive(
                            old(self), &pre_iter, self,
                            Seq::<u64>::empty(),
                        );
                        lemma_positive_provenance_metadata_frame_transitive(
                            old(self), &pre_iter, self,
                        );
                        lemma_positive_provenance_origin_transitive(
                            old(self), &pre_iter, self,
                        );
                        assert forall|r: RequestId| r != rid
                            && #[trigger] pre_iter.running@.contains(r)
                            implies self.running@.contains(r)
                        by {}
                        assert(self.request_residency@.contains_key(rid)
                            ==> self.running@.contains(rid)) by {
                            if self.request_residency@.contains_key(rid) {
                                assert(self.live_requests@.contains_key(rid)) by {
                                    assert(residency_has_live_request(self));
                                }
                                assert(pre_iter.live_requests@.contains_key(rid)) by {
                                    assert(live_covers_queue(&pre_iter));
                                }
                                assert(!should_finish_after_append(
                                    pre_iter.live_requests@[rid], sample.token));
                            }
                        }
                        lemma_residency_running_aligned_commit_one(
                            &pre_iter, self, rid,
                        );
                        assert(*sample == emitted@[rid]);
                        // Debt bookkeeping: this iteration consumed at most
                        // its own head; the remaining suffix is framed.
                        let ghost sfx = plan.scheduled_ids@.subrange(i as int + 1,
                            plan.scheduled_ids@.len() as int);
                        lemma_debt_head(pre_iter.live_requests@,
                            plan.scheduled_ids@, i as int);
                        assert forall|j: int| 0 <= j < sfx.len()
                            implies (self.live_requests@.contains_key(#[trigger] sfx[j])
                                    <==> pre_iter.live_requests@.contains_key(sfx[j]))
                                && (pre_iter.live_requests@.contains_key(sfx[j])
                                    ==> self.live_requests@[sfx[j]]
                                        == pre_iter.live_requests@[sfx[j]])
                        by {
                            assert(sfx[j] == plan.scheduled_ids@[i as int + 1 + j]);
                            assert(sfx[j] != rid);
                        }
                        lemma_debt_frame(pre_iter.live_requests@,
                            self.live_requests@, sfx);
                        assert(self.free_blocks as int
                            >= pre_iter.free_blocks as int
                            - (if pre_iter.live_requests@.contains_key(rid)
                                && pre_iter.running@.contains(rid)
                                && history(pre_iter.live_requests@[rid]).len() as int
                                    % (BLOCK_SIZE_SPEC as int) == 0 {
                                1int
                            } else {
                                0int
                            }));
                        assert(full_tail_debt(self.live_requests@, sfx)
                            <= self.free_blocks as int);
                        // Re-establish the processed-side invariants with the
                        // per-rid ensures, splitting r == rid vs frame.
                        assert forall|r: RequestId| #[trigger] out@.contains_key(r)
                            implies exists|k: int|
                                0 <= k < i as int + 1
                                    && plan.scheduled_ids@[k] == r
                                    && #[trigger] plan.sample_mask@[k] by {
                            if r == rid {
                                assert(plan.scheduled_ids@[i as int] == r);
                                assert(plan.sample_mask@[i as int]);
                            } else {
                                assert(pre_out.contains_key(r));
                            }
                        }
                        assert forall|r: RequestId|
                            #[trigger] self.live_requests@.contains_key(r)
                            implies (if out@.contains_key(r) {
                                crate::proof::reference::request_machine::machine_step_transition_full(
                                    old(self).live_requests@[r], self.live_requests@[r],
                                    emitted@[r].sampler_state, emitted@[r].token)
                            } else {
                                self.live_requests@[r] == old(self).live_requests@[r]
                            }) by {
                            if r != rid {
                                assert(pre_iter.live_requests@.contains_key(r));
                                assert(self.live_requests@[r]
                                    == pre_iter.live_requests@[r]);
                                assert(out@.contains_key(r) == pre_out.contains_key(r));
                            }
                        }
                        assert forall|r: RequestId|
                            #[trigger] old(self).live_requests@.contains_key(r)
                            implies (self.live_requests@.contains_key(r) <==>
                                !(out@.contains_key(r)
                                  && should_finish_after_append(
                                      old(self).live_requests@[r], out@[r]))) by {
                            if r != rid {
                                assert(out@.contains_key(r) == pre_out.contains_key(r));
                                if pre_out.contains_key(r) {
                                    assert(out@[r] == pre_out[r]);
                                }
                                assert(self.live_requests@.contains_key(r)
                                    == pre_iter.live_requests@.contains_key(r));
                            }
                        }
                        assert forall|r: RequestId| #[trigger] self.running@.contains(r)
                            implies (old(self).running@.contains(r)
                                && !(out@.contains_key(r)
                                    && should_finish_after_append(
                                        old(self).live_requests@[r], out@[r]))) by {
                            if r != rid {
                                assert(self.running@.contains(r)
                                    == pre_iter.running@.contains(r));
                                assert(out@.contains_key(r) == pre_out.contains_key(r));
                                if pre_out.contains_key(r) {
                                    assert(out@[r] == pre_out[r]);
                                }
                            } else {
                                assert(pre_iter.running@.contains(rid)
                                    == old(self).running@.contains(rid));
                                assert(out@[rid] == emitted@[rid].token);
                                if pre_iter.live_requests@.contains_key(rid) {
                                    assert(pre_iter.live_requests@[rid]
                                        == old(self).live_requests@[rid]);
                                }
                            }
                        }
                        assert forall|r: RequestId|
                            old(self).running@.contains(r)
                                && !step_plan_parks_before(
                                    plan, r, i as int + 1,
                                )
                                && !((#[trigger] out@.contains_key(r))
                                    && should_finish_after_append(
                                        old(self).live_requests@[r], out@[r]))
                            implies self.running@.contains(r) by {
                            if r != rid {
                                assert(pre_iter.running@.contains(r)
                                    == self.running@.contains(r));
                                assert(out@.contains_key(r) == pre_out.contains_key(r));
                                if pre_out.contains_key(r) {
                                    assert(out@[r] == pre_out[r]);
                                }
                            } else {
                                assert(out@.contains_key(rid));
                                assert(out@[rid] == emitted@[rid].token);
                                if pre_iter.live_requests@.contains_key(rid) {
                                    assert(pre_iter.live_requests@[rid]
                                        == old(self).live_requests@[rid]);
                                }
                            }
                        }
                    }
                },
                None => {
                    assert(!emitted@.contains_key(rid));
                    assert(!plan.sample_mask@[i as int]);
                    let ghost pre_iter = *self;
                    proof {
                        if out@.contains_key(rid) {
                            let k = choose|k: int|
                                0 <= k < i as int
                                    && plan.scheduled_ids@[k] == rid
                                    && plan.sample_mask@[k];
                            assert(plan.scheduled_ids@[k]
                                == plan.scheduled_ids@[i as int]);
                            assert(false);
                        }
                        if step_plan_parks_before(plan, rid, i as int) {
                            let k = choose|k: int|
                                0 <= k < i as int
                                    && plan.scheduled_ids@[k] == rid
                                    && !plan.sample_mask@[k];
                            assert(plan.scheduled_ids@[k]
                                == plan.scheduled_ids@[i as int]);
                            assert(false);
                        }
                        assert(old(self).running@.contains(rid));
                        assert(self.running@.contains(rid));
                        assert(self.live_requests@.contains_key(rid)) by {
                            assert(live_covers_queue(self));
                        }
                        assert(self.live_requests@[rid]
                            == old(self).live_requests@[rid]);
                        assert(self.live_requests@[rid]
                            .generated_tokens@.len() == 0);
                    }

                    self.commit_kv_only_prefill(rid);

                    proof {
                        assert(self.waiting@
                            == pre_iter.waiting@.push(rid));
                        assert(self.waiting@[pre_iter.waiting@.len() as int] == rid);
                        assert(registry_entries_from_pre(
                            &pre_iter, self, Seq::<u64>::empty(),
                        ));
                        lemma_registry_entries_from_pre_transitive(
                            old(self), &pre_iter, self,
                            Seq::<u64>::empty(),
                        );
                        lemma_positive_provenance_metadata_frame_transitive(
                            old(self), &pre_iter, self,
                        );
                        lemma_positive_provenance_origin_transitive(
                            old(self), &pre_iter, self,
                        );

                        // The parked row changes no live state and releases
                        // rather than consumes capacity, so dropping the debt
                        // head leaves enough room for every later row.
                        lemma_debt_head(self.live_requests@,
                            plan.scheduled_ids@, i as int);
                        let ghost sfx = plan.scheduled_ids@.subrange(
                            i as int + 1,
                            plan.scheduled_ids@.len() as int,
                        );
                        assert(self.live_requests@ == pre_iter.live_requests@);
                        assert(full_tail_debt(self.live_requests@, sfx)
                            <= pre_iter.free_blocks as int);
                        assert(pre_iter.free_blocks <= self.free_blocks);
                        assert(full_tail_debt(self.live_requests@, sfx)
                            <= self.free_blocks as int);

                        assert forall|r: RequestId|
                            #[trigger] self.waiting@.contains(r)
                            implies old(self).waiting@.contains(r)
                                || step_plan_parks_before(
                                    plan, r, i as int + 1,
                                )
                        by {
                            if pre_iter.waiting@.contains(r) {
                            } else {
                                assert(r == rid);
                                assert(plan.scheduled_ids@[i as int] == r);
                            }
                        }
                        assert forall|r: RequestId|
                            old(self).waiting@.contains(r)
                                || step_plan_parks_before(
                                    plan, r, i as int + 1,
                                )
                            implies #[trigger] self.waiting@.contains(r)
                        by {
                            if old(self).waiting@.contains(r) {
                                assert(pre_iter.waiting@.contains(r));
                                let q = pre_iter.waiting@.index_of(r);
                                assert(self.waiting@[q] == r);
                            } else if step_plan_parks_before(
                                plan, r, i as int + 1,
                            ) && !step_plan_parks_before(
                                plan, r, i as int,
                            ) {
                                let k = choose|k: int|
                                    0 <= k < i as int + 1
                                        && plan.scheduled_ids@[k] == r
                                        && !plan.sample_mask@[k];
                                assert(k == i as int) by {
                                    if k < i as int {
                                        assert(step_plan_parks_before(
                                            plan, r, i as int,
                                        ));
                                    }
                                }
                                assert(r == rid);
                                assert(self.waiting@.contains(rid));
                            } else {
                                assert(pre_iter.waiting@.contains(r));
                                let q = pre_iter.waiting@.index_of(r);
                                assert(self.waiting@[q] == r);
                            }
                        }

                        assert forall|r: RequestId|
                            #[trigger] self.running@.contains(r)
                            implies old(self).running@.contains(r)
                                && !step_plan_parks_before(
                                    plan, r, i as int + 1,
                                )
                                && !(out@.contains_key(r)
                                    && should_finish_after_append(
                                        old(self).live_requests@[r], out@[r]))
                        by {
                            assert(r != rid);
                            assert(pre_iter.running@.contains(r));
                            if step_plan_parks_before(
                                plan, r, i as int + 1,
                            ) && !step_plan_parks_before(
                                plan, r, i as int,
                            ) {
                                let k = choose|k: int|
                                    0 <= k < i as int + 1
                                        && plan.scheduled_ids@[k] == r
                                        && !plan.sample_mask@[k];
                                assert(k == i as int) by {
                                    if k < i as int {
                                        assert(step_plan_parks_before(
                                            plan, r, i as int,
                                        ));
                                    }
                                }
                                assert(r == rid);
                            }
                        }
                        assert forall|r: RequestId|
                            old(self).running@.contains(r)
                                && !step_plan_parks_before(
                                    plan, r, i as int + 1,
                                )
                                && !(out@.contains_key(r)
                                    && should_finish_after_append(
                                        old(self).live_requests@[r], out@[r]))
                            implies #[trigger] self.running@.contains(r)
                        by {
                            assert(r != rid) by {
                                if r == rid {
                                    assert(step_plan_parks_before(
                                        plan, r, i as int + 1,
                                    ));
                                }
                            }
                            assert(pre_iter.running@.contains(r));
                        }

                        assert forall|r: RequestId|
                            !out@.contains_key(r)
                                && !step_plan_parks_before(
                                    plan, r, i as int + 1,
                                )
                                && #[trigger] old(self).request_residency@
                                    .contains_key(r)
                            implies self.request_residency@.contains_key(r)
                                && self.request_residency@[r]
                                    == old(self).request_residency@[r]
                        by {
                            assert(r != rid) by {
                                if r == rid {
                                    assert(step_plan_parks_before(
                                        plan, r, i as int + 1,
                                    ));
                                }
                            }
                            assert(pre_iter.request_residency@.contains_key(r));
                            assert(pre_iter.request_residency@[r]
                                == old(self).request_residency@[r]);
                        }

                        assert forall|r: RequestId|
                            #[trigger] out@.contains_key(r)
                                && self.running@.contains(r)
                                && self.live_requests@.contains_key(r)
                            implies {
                                let ids = self.request_residency@[r].block_ids@;
                                &&& self.request_residency@.contains_key(r)
                                &&& ids.len() >= 1
                                &&& self.blocks@.contains_key(ids[ids.len() - 1])
                                &&& self.blocks@[ids[ids.len() - 1]].refcount == 1
                                &&& self.blocks@[ids[ids.len() - 1]].prefix_depth == 0
                                &&& self.blocks@[ids[ids.len() - 1]].hash_value == 0
                            }
                        by {
                            assert(r != rid);
                            assert(pre_iter.running@.contains(r));
                            assert(pre_iter.live_requests@.contains_key(r));
                            let ids = pre_iter.request_residency@[r].block_ids@;
                            let tail = ids[ids.len() - 1];
                            assert(self.blocks@[tail] == pre_iter.blocks@[tail]);
                        }
                        assert forall|r: RequestId|
                            #[trigger] self.running@.contains(r)
                                && self.live_requests@.contains_key(r)
                                && !out@.contains_key(r)
                            implies {
                                let t = old(self).request_residency@[r]
                                    .block_ids@[old(self).request_residency@[r]
                                        .block_ids@.len() - 1];
                                self.request_residency@[r]
                                        == old(self).request_residency@[r]
                                    && self.blocks@.contains_key(t)
                                    && self.blocks@[t].prefix_depth
                                        == old(self).blocks@[t].prefix_depth
                                    && self.blocks@[t].hash_value
                                        == old(self).blocks@[t].hash_value
                            }
                        by {
                            assert(r != rid);
                            assert(pre_iter.running@.contains(r));
                            assert(pre_iter.live_requests@.contains_key(r));
                            assert(pre_iter.request_residency@[r]
                                == old(self).request_residency@[r]);
                            let ids = pre_iter.request_residency@[r].block_ids@;
                            let tail = ids[ids.len() - 1];
                            assert(self.blocks@[tail] == pre_iter.blocks@[tail]);
                        }
                    }
                },
            }
            i = i + 1;
            assert forall|j: int|
                #![trigger plan.scheduled_ids@[j]]
                i <= j < plan.scheduled_ids@.len()
                    && old(self).live_requests@.contains_key(plan.scheduled_ids@[j])
                implies self.live_requests@.contains_key(plan.scheduled_ids@[j])
                    && self.live_requests@[plan.scheduled_ids@[j]]
                        == old(self).live_requests@[plan.scheduled_ids@[j]]
            by {
                let prev = i as int - 1;
                assert(0 <= prev < plan.scheduled_ids@.len());
                assert(plan.scheduled_ids@[prev] != plan.scheduled_ids@[j]) by {
                    assert(plan.scheduled_ids@.no_duplicates());
                }
            }
        }
        proof {
            assert(tail_write_exclusive(self)) by {
                assert forall|r: RequestId|
                    #[trigger] self.running@.contains(r)
                        && self.live_requests@.contains_key(r)
                    implies {
                        let ids = self.request_residency@[r].block_ids@;
                        &&& self.request_residency@.contains_key(r)
                        &&& ids.len() >= 1
                        &&& self.blocks@.contains_key(ids[ids.len() - 1])
                        &&& self.blocks@[ids[ids.len() - 1]].refcount == 1
                        &&& self.blocks@[ids[ids.len() - 1]].prefix_depth == 0
                        &&& self.blocks@[ids[ids.len() - 1]].hash_value == 0
                    }
                by {
                    if out@.contains_key(r) {
                    } else {
                        assert(!plan.scheduled_ids@.contains(r)) by {
                            if plan.scheduled_ids@.contains(r) {
                                let k = choose|k: int|
                                    0 <= k < plan.scheduled_ids@.len()
                                        && plan.scheduled_ids@[k] == r;
                                if plan.sample_mask@[k] {
                                    assert(out@.contains_key(r));
                                } else {
                                    assert(step_plan_parks(plan, r));
                                    assert(!self.running@.contains(r));
                                }
                            }
                        }
                        let ids = self.request_residency@[r].block_ids@;
                        let old_ids = old(self).request_residency@[r].block_ids@;
                        assert(ids == old_ids);
                        let tail = ids[ids.len() - 1];
                        assert(self.blocks@[tail].prefix_depth
                            == old(self).blocks@[tail].prefix_depth);
                        assert(self.blocks@[tail].hash_value
                            == old(self).blocks@[tail].hash_value);
                    }
                }
            }
        }
        out
    }
}

} // verus!
