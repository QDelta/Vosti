// Verified demand calculation, scheduling, and plan transitions.

use super::*;

verus! {
// Select the number of prompt tokens computed by one prefill row.  A final
// chunk may end at an arbitrary token; every non-final chunk is page-aligned
// so it can be represented by complete reusable KV pages.  Returning zero is
// an ordinary scheduling decision when the remaining batch budget cannot fit
// one page.
pub fn select_prefill_chunk_len(
    remaining: usize,
    token_budget: usize,
) -> (out: usize)
    requires
        remaining > 0,
    ensures
        out <= remaining,
        out <= token_budget,
        (out == remaining) <==> remaining <= token_budget,
        out < remaining ==> out % (BLOCK_SIZE as usize) == 0,
        token_budget >= BLOCK_SIZE as usize ==> out > 0,
{
    if remaining <= token_budget {
        remaining
    } else {
        let pages = token_budget / (BLOCK_SIZE as usize);
        let chunk = pages * (BLOCK_SIZE as usize);
        proof {
            assert(pages as int == token_budget as int / (BLOCK_SIZE_SPEC as int));
            assert(chunk as int == pages as int * (BLOCK_SIZE_SPEC as int));
            vstd::arithmetic::div_mod::lemma_remainder_lower(
                token_budget as int,
                BLOCK_SIZE_SPEC as int,
            );
            assert(BLOCK_SIZE as usize as int == BLOCK_SIZE_SPEC as int);
            assert(chunk as int <= token_budget as int);
            vstd::arithmetic::div_mod::lemma_mod_multiples_basic(
                pages as int,
                BLOCK_SIZE_SPEC as int,
            );
            assert(chunk < remaining);
            if token_budget >= BLOCK_SIZE as usize {
                assert(pages >= 1);
            }
        }
        chunk
    }
}

impl CacheScheduler {
    // Exact physical allocation demand after `matched` full prefix pages are
    // reused.  The matcher's strict bound leaves at least one suffix token.
    fn fresh_prefill_blocks(
        prompt_tokens: &Vec<TokenId>,
        matched: &Vec<BlockId>,
    ) -> (fresh: u64)
        requires
            prompt_tokens@.len() <= u64::MAX as int,
            prompt_tokens@.len() == 0 ==> matched@.len() == 0,
            prompt_tokens@.len() > 0 ==> matched@.len() as int
                * (BLOCK_SIZE_SPEC as int) < prompt_tokens@.len() as int,
        ensures
            fresh as int + matched@.len() as int
                == blocks_needed_for(prompt_tokens@.len() as nat) as int,
    {
        let n = prompt_tokens.len() as u64;
        if n == 0 {
            assert(blocks_needed_for(0nat) == 0);
            return 0;
        }
        let total = (n - 1) / BLOCK_SIZE + 1;
        proof {
            assert(total as int
                == blocks_needed_for(prompt_tokens@.len() as nat) as int);
            lemma_blocks_needed_split(
                prompt_tokens@.len() as nat, matched@.len() as nat,
            );
            assert(matched@.len() <= total as int);
            assert(matched@.len() <= u64::MAX as int);
        }
        total - matched.len() as u64
    }

    // Isolate the cache-pressure policy from the admission loop. The first
    // match sizes reclamation from the exact fresh suffix; the returned match
    // is obtained after reclamation and is therefore the only one admission
    // may attach. The frame contract is exactly what planning's persistent
    // scheduler invariants need to transport across zero-ref leaf eviction.
    #[verifier::spinoff_prover]
    fn prepare_prefill_match_under_pressure(
        &mut self,
        prompt_tokens: &Vec<TokenId>,
        excluded: &Vec<u64>,
        query_used: u64,
        key_used: u64,
        commit_debt: u64,
    ) -> (out: Vec<BlockId>)
        requires
            cs_valid(old(self)),
            free_queue_valid(old(self)),
            persistent_provenance_closed(old(self)),
            prompt_tokens@.len() > 0,
            prompt_tokens@.len() <= u64::MAX as int,
            prompt_tokens@.len() <= usize::MAX as int,
        ensures
            cs_valid(final(self)),
            free_queue_valid(final(self)),
            persistent_provenance_closed(final(self)),
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
            positive_provenance_origin(old(self), final(self)),
            out@.no_duplicates(),
            out@.len() as int * (BLOCK_SIZE_SPEC as int)
                < prompt_tokens@.len() as int,
            forall|j: int| 0 <= j < out@.len()
                ==> final(self).blocks@.contains_key(#[trigger] out@[j])
                    && final(self).blocks@[out@[j]].tokens@.len()
                        == BLOCK_SIZE_SPEC as int,
            forall|j: int| 0 <= j < out@.len()
                ==> (#[trigger] final(self).blocks@[out@[j]].hash_value) != 0
                    && !excluded@.contains(final(self).blocks@[out@[j]].hash_value),
            forall|j: int| 0 <= j < out@.len()
                ==> final(self).hash_to_block@.contains_key(
                        #[trigger] final(self).blocks@[out@[j]].hash_value)
                    && final(self).hash_to_block@[final(self).blocks@[out@[j]].hash_value]
                        == out@[j],
            registered_prefix_chain(final(self).blocks@, out@),
            token_placement_prefix(
                final(self).blocks@, out@, prompt_tokens@,
                out@.len() as int * (BLOCK_SIZE_SPEC as int),
            ),
    {
        let tentative = self.match_cached_prefix(prompt_tokens, excluded);
        let start = tentative.len() * (BLOCK_SIZE as usize);
        proof {
            assert(start < prompt_tokens.len());
        }
        let remaining = prompt_tokens.len() - start;
        let used = query_used as usize;
        let available = if used >= self.config.max_num_batched_tokens {
            0
        } else {
            self.config.max_num_batched_tokens - used
        };
        let chunk = select_prefill_chunk_len(remaining, available);
        let end = start + chunk;
        let fits_q = chunk as u64 <= u64::MAX - query_used;
        let fits_k = end as u64 <= u64::MAX - key_used;
        if chunk > 0 && fits_q && fits_k {
            let inc: u64 = if prompt_tokens.len() % (BLOCK_SIZE as usize) == 0 {
                1
            } else {
                0
            };
            let fresh = Self::fresh_prefill_blocks(prompt_tokens, &tentative);
            let required = fresh.saturating_add(commit_debt).saturating_add(inc);
            let _reclaimed = self.reclaim_cached_leaves_until(required);
        } else {
            proof {
                lemma_positive_provenance_origin_refl(self);
            }
        }
        self.match_cached_prefix(prompt_tokens, excluded)
    }

    // Count the commit-time pages needed by the longest decode prefix that is
    // otherwise schedulable (sequence cap and cumulative-key arithmetic),
    // deliberately ignoring the current free-list size. `plan` uses this as
    // its pressure-reclamation target before calling the capacity-aware
    // selector below.
    pub fn decode_headroom_demand(&self) -> (debt: u64)
        requires
            cs_valid(self),
            self.num_blocks <= u64::MAX / BLOCK_SIZE,
            self.running@.len() > 0 ==> decode_plan_ready(self),
    {
        let mut debt: u64 = 0;
        let mut total_k: u64 = 0;
        let mut i: usize = 0;
        while i < self.running.len() && i < self.config.max_num_seqs
            invariant
                cs_valid(self),
                self.running@.len() > 0 ==> decode_plan_ready(self),
                self.num_blocks <= u64::MAX / BLOCK_SIZE,
                i <= self.running@.len(),
                i <= self.config.max_num_seqs,
            decreases self.running@.len() - i,
        {
            let rid = self.running[i];
            assert(self.running@.contains(rid));
            let state = match self.live_requests.get(&rid) {
                Some(s) => s,
                None => {
                    proof {
                        assert(decode_plan_ready(self));
                        assert(false);
                    }
                    return debt;
                },
            };
            proof {
                assert(*state == self.live_requests@[rid]);
                assert(decode_plan_ready(self));
                assert(history(self.live_requests@[rid]).len()
                    == state.prompt_tokens@.len()
                        + state.generated_tokens@.len());
                assert(history(self.live_requests@[rid]).len()
                    <= usize::MAX as int);
            }
            let hist = state.prompt_tokens.len() + state.generated_tokens.len();
            let hist_u64 = hist as u64;
            if hist_u64 > u64::MAX - total_k {
                return debt;
            }
            if hist % (BLOCK_SIZE as usize) == 0 {
                debt = debt.saturating_add(1);
            }
            total_k = total_k + hist_u64;
            i = i + 1;
        }
        debt
    }

    // Select the longest running prefix within `max_num_seqs` whose commit-time
    // block debt fits in the free pool. Each request with a block-aligned history
    // (a full tail) needs one fresh block. The scan also stops before cumulative
    // `cu_k` would overflow u64, establishing the bounds needed by later addition
    // and the strict partition theorem. Pool exhaustion reduces the schedule so
    // every selected request has space for its commit-time append.
    pub fn select_decode_schedule(&self) -> (out: (Vec<RequestId>, u64))
        requires
            cs_valid(self),
            self.num_blocks <= u64::MAX / BLOCK_SIZE,
            self.running@.len() > 0 ==> decode_plan_ready(self),
        ensures
            out.0@ == self.running@.subrange(0, out.0@.len() as int),
            out.0@.len() <= self.running@.len(),
            out.0@.len() <= self.config.max_num_seqs,
            out.1 as int == full_tail_debt(self.live_requests@, out.0@),
            out.1 <= self.free_blocks,
            scheduled_history_total(self, out.0@) <= u64::MAX as int,
    {
        let mut out: Vec<RequestId> = Vec::new();
        let mut debt: u64 = 0;
        let mut total_k: u64 = 0;
        let mut i: usize = 0;
        proof {
            assert(self.running@.subrange(0, 0) =~= Seq::<RequestId>::empty());
        }
        assert(self.free_blocks <= self.num_blocks) by {
            assert(block_count_valid(self));
            assert(self.blocks@.dom().len() >= 0);
        }
        while i < self.running.len()
            invariant
                cs_valid(self),
                self.running@.len() > 0 ==> decode_plan_ready(self),
                self.free_blocks <= self.num_blocks,
                self.num_blocks <= u64::MAX / BLOCK_SIZE,
                i <= self.running@.len(),
                out@ == self.running@.subrange(0, i as int),
                out@.len() == i,
                out@.len() <= self.config.max_num_seqs,
                debt as int == full_tail_debt(self.live_requests@, out@),
                debt <= self.free_blocks,
                total_k as int == scheduled_history_total(self, out@),
            decreases self.running@.len() - i,
        {
            if out.len() >= self.config.max_num_seqs {
                return (out, debt);
            }
            let rid = self.running[i];
            proof {
                assert(self.running@.contains(rid));
                assert(decode_plan_ready(self));
            }
            let state = match self.live_requests.get(&rid) {
                Some(s) => s,
                None => {
                    proof { assert(false); }
                    return (out, debt);
                },
            };
            proof {
                assert(*state == self.live_requests@[rid]);
                assert(history(self.live_requests@[rid]).len()
                    == state.prompt_tokens@.len() + state.generated_tokens@.len());
                assert(history(self.live_requests@[rid]).len() <= usize::MAX as int);
            }
            let hist: usize = state.prompt_tokens.len() + state.generated_tokens.len();
            let hist_u64 = hist as u64;
            if hist_u64 > u64::MAX - total_k {
                return (out, debt);
            }
            let full = hist % (BLOCK_SIZE as usize) == 0;
            let inc: u64 = if full { 1 } else { 0 };
            if debt + inc > self.free_blocks {
                return (out, debt);
            }
            let ghost out_pre = out@;
            out.push(rid);
            debt = debt + inc;
            total_k = total_k + hist_u64;
            proof {
                lemma_debt_snoc(self.live_requests@, out_pre, rid);
                lemma_scheduled_history_total_snoc(self, out_pre, rid);
                assert(hist as int == history(self.live_requests@[rid]).len() as int);
                assert(scheduled_history_len_at(self, rid) == hist as int) by {
                    assert(history(self.live_requests@[rid]).len()
                        == self.live_requests@[rid].prompt_tokens@.len()
                            + self.live_requests@[rid].generated_tokens@.len());
                }
                assert((hist as int % (BLOCK_SIZE_SPEC as int) == 0) == full) by {
                    assert(BLOCK_SIZE_SPEC as int == BLOCK_SIZE as usize as int);
                }
                assert(out@ =~= self.running@.subrange(0, i as int + 1));
            }
            i = i + 1;
        }
        (out, debt)
    }

    // Plan the next step. One verified unified path: the running prefix
    // contributes decode rows (q_len 1, k_len = history), then waiting
    // requests are admitted into the same cu-partitioned batch as final or
    // KV-only prefill chunks. Cached prefix tokens count toward k_len but not
    // the query budget; non-final fresh suffixes are page-aligned. Admission
    // stops when the remaining query budget cannot fit a valid next chunk.
    // Each candidate is popped from `waiting` BEFORE allocating
    // (re-queued on budget/allocation failure), so `allocate_prefill`'s
    // queue-absence preconditions hold; on success the rid moves to
    // `running`.  Downstream (kernels, witnesses, coherence discharge) is
    // already per-request via the cu partition and `pw_is_decode`, so
    // mixed plans need no proof changes outside this function.
    #[verifier::spinoff_prover]
    #[verifier::rlimit(400)]
    pub fn plan(
        &mut self,
        device_anchor: Option<&RT::Tensor>,
    ) -> (out: (StepPlan, Tracked<StepPlanPerms>))
        requires
            cs_valid(old(self)),
            free_queue_valid(old(self)),
            old(self).num_blocks <= u64::MAX / BLOCK_SIZE,
            prefill_plan_ready(old(self)),
            old(self).running@.len() > 0 ==> decode_plan_ready(old(self)),
            residency_history_aligned(old(self)),
            slot_mapping_aligned(old(self)),
            tail_write_exclusive(old(self)),
            residency_running_aligned(old(self)),
            persistent_provenance_closed(old(self)),
        ensures
            cs_valid(final(self)),
            free_queue_valid(final(self)),
            step_plan_perms_valid(&out.0, out.1@),
            step_plan_commit_ready(final(self), &out.0),
            step_plan_shape_ok(&out.0),
            final(self).live_requests@ == old(self).live_requests@,
            final(self).accepted_requests@ == old(self).accepted_requests@,
            final(self).num_blocks == old(self).num_blocks,
            // Every scheduled ID came from a pre-plan queue: the running prefix
            // or waiting admissions. The geometry proof uses this origin fact.
            forall|i: int|
                #![trigger out.0.scheduled_ids@[i]]
                0 <= i < out.0.scheduled_ids@.len()
                ==> old(self).running@.contains(out.0.scheduled_ids@[i])
                    || old(self).waiting@.contains(out.0.scheduled_ids@[i]),
            // The post-plan running queue is exactly the old one
            // plus this step's schedule.
            forall|r: RequestId| #[trigger] final(self).running@.contains(r)
                <==> (old(self).running@.contains(r)
                    || out.0.scheduled_ids@.contains(r)),
            residency_history_aligned(final(self)),
            slot_mapping_aligned(final(self)),
            plan_slot_segments_ok(old(self), final(self), &out.0),
            plan_forward_layout_ok(old(self), &out.0),
            plan_sample_policy(old(self), &out.0),
            plan_residency_extents(old(self), final(self), &out.0),
            plan_cached_prefix_origins(old(self), &out.0),
            positive_chains_from_pre_or_admission_rows(
                old(self), final(self), out.0.scheduled_ids@,
                out.0.block_table_repr@, out.0.cu_seqlens_k_repr@,
            ),
            published_admission_row_prefixes(
                old(self), final(self), out.0.scheduled_ids@,
                out.0.block_table_repr@, out.0.cu_seqlens_k_repr@,
            ),
            // Write-page exclusivity at plan exit.
            pre_commit_tails_exclusive(final(self)),
            residency_running_aligned(final(self)),
            persistent_provenance_closed(final(self)),
            forall|r: RequestId| #[trigger] old(self).running@.contains(r)
                && old(self).live_requests@.contains_key(r)
                ==> {
                    let t = old(self).request_residency@[r].block_ids@[
                        old(self).request_residency@[r].block_ids@.len() - 1];
                    final(self).blocks@.contains_key(t)
                        && final(self).blocks@[t].tokens@
                            == old(self).blocks@[t].tokens@
                        && final(self).blocks@[t].refcount
                            == old(self).blocks@[t].refcount
                        && final(self).blocks@[t].prefix_depth
                            == old(self).blocks@[t].prefix_depth
                        && final(self).blocks@[t].hash_value
                            == old(self).blocks@[t].hash_value
                },
            forall|k: int| 0 <= k < out.0.scheduled_ids@.len()
                && !old(self).running@.contains(out.0.scheduled_ids@[k])
                ==> #[trigger] admitted_pages_exclusive_post(final(self),
                    out.0.scheduled_ids@, k),
            // The schedule is capped so every
            // scheduled full-tail request can take a fresh block at commit.
            commit_headroom(final(self), out.0.scheduled_ids@),
            // Plan never touches a running request's
            // residency (admissions allocate only for waiting requests).
            forall|r: RequestId| #[trigger] old(self).running@.contains(r)
                ==> old(self).request_residency@.contains_key(r)
                    && final(self).request_residency@.contains_key(r)
                    && final(self).request_residency@[r]
                        == old(self).request_residency@[r],
    {
        hide(free_queue_valid);
        let ghost plan_pre = *self;
        // Decode capacity is decided before the admission loop, so reclaim
        // its required commit headroom here. Without this pre-pass a running
        // request with a full tail could be omitted solely because reusable
        // zero-ref prefix pages occupied the physical pool.
        let decode_required_free = self.decode_headroom_demand();
        let _decode_reclaimed = self.reclaim_cached_leaves_until(
            decode_required_free,
        );
        let ghost post_decode_reclaim = *self;
        proof {
            assert(positive_provenance_origin(
                &plan_pre, &post_decode_reclaim,
            ));
            lemma_residency_history_aligned_positive_frame(
                &plan_pre, &post_decode_reclaim,
            );
            lemma_slot_mapping_aligned_frame(
                &plan_pre, &post_decode_reclaim,
            );
            lemma_residency_running_aligned_frame(
                &plan_pre, &post_decode_reclaim,
            );
            lemma_registry_entries_from_pre_refl(&plan_pre);
            lemma_registry_entries_from_pre_reclaim(
                old(self), &plan_pre, &post_decode_reclaim,
                Seq::<u64>::empty(),
            );
            assert(decode_plan_ready(self)) by {
                assert(self.live_requests@ == old(self).live_requests@);
                assert(self.request_residency@ == old(self).request_residency@);
                assert(self.running@ == old(self).running@);
            }
            assert forall|r: RequestId|
                #[trigger] old(self).running@.contains(r)
                && old(self).live_requests@.contains_key(r)
                implies {
                    let t = old(self).request_residency@[r]
                        .block_ids@[old(self).request_residency@[r]
                            .block_ids@.len() - 1];
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
                let t = old(self).request_residency@[r]
                    .block_ids@[old(self).request_residency@[r]
                        .block_ids@.len() - 1];
                assert(old(self).blocks@[t].refcount == 1) by {
                    assert(tail_write_exclusive(old(self)));
                }
                assert(self.blocks@[t] == old(self).blocks@[t]);
            }
        }
        // ---- Decode base: the running prefix, one query row each,
        // capped by commit-time block headroom (capacity by construction:
        // never schedule a request whose committed token could not get a
        // fresh tail block). ----
        let (mut scheduled_ids, mut commit_debt) = self.select_decode_schedule();
        assert(scheduled_ids@.no_duplicates()) by {
            assert(running_unique(self));
            assert(scheduled_ids@ ==
                self.running@.subrange(0, scheduled_ids@.len() as int));
        }
        assert forall|i: int|
            #![trigger scheduled_ids@[i]]
            0 <= i < scheduled_ids@.len()
            implies self.live_requests@.contains_key(scheduled_ids@[i])
                && can_step(self.live_requests@[scheduled_ids@[i]])
                && valid_request_state(self.live_requests@[scheduled_ids@[i]])
                && self.live_requests@[scheduled_ids@[i]].generated_tokens@.len()
                    < usize::MAX as int
                && history(self.live_requests@[scheduled_ids@[i]]).len() <= usize::MAX as int
                && history(self.live_requests@[scheduled_ids@[i]]).len() <= u64::MAX as int
                && self.request_residency@.contains_key(scheduled_ids@[i])
                && self.request_residency@[scheduled_ids@[i]].slot_mapping@.len() > 0
                && self.running@.contains(scheduled_ids@[i])
        by {
            assert(scheduled_ids@ == self.running@.subrange(
                0, scheduled_ids@.len() as int));
            assert(scheduled_ids@[i] == self.running@.subrange(
                0, scheduled_ids@.len() as int)[i]);
            assert(scheduled_ids@[i] == self.running@[i]);
            assert(self.running@.contains(scheduled_ids@[i]));
            assert(decode_plan_ready(old(self)));
            assert(can_step(self.live_requests@[scheduled_ids@[i]]));
        }
        let (mut input_values, mut position_values, mut cu_q_values, mut cu_k_values, mut max_seqlen_k) =
            decode_inputs_for_scheduled(self, &scheduled_ids);
        let mut block_rows = block_rows_for_scheduled(self, &scheduled_ids);
        let mut slot_values = decode_slots_for_scheduled(self, &scheduled_ids);
        let mut max_seqlen_q: usize = if scheduled_ids.len() == 0 { 0 } else { 1 };
        let decode_count = scheduled_ids.len();
        let mut sample_mask = all_true_sample_mask(decode_count);
        proof {
            // Segment shape at loop entry: every base row is a decode row
            // with cu [k, k+1] and the residency's last mapped slot, which
            // the slot-mapping companion pins to the last history position.
            assert forall|k: int| 0 <= k < scheduled_ids@.len()
                implies #[trigger] plan_seg_at(old(self), self, scheduled_ids@,
                    cu_q_values@, cu_k_values@, slot_values@, k)
            by {
                let srid = scheduled_ids@[k];
                assert(scheduled_ids@ == self.running@.subrange(
                    0, scheduled_ids@.len() as int));
                assert(srid == self.running@[k]);
                assert(self.running@.contains(srid));
                assert(old(self).running@.contains(srid));
                assert(decode_plan_ready(old(self)));
                assert(cu_q_values@[k] == k && cu_q_values@[k + 1] == k + 1);
                assert(slot_values@[k]
                    == self.request_residency@[srid].slot_mapping@[
                        self.request_residency@[srid].slot_mapping@.len() - 1]);
            }
            assert(plan_seg_inv(old(self), self, scheduled_ids@, cu_q_values@,
                cu_k_values@, slot_values@));
            assert forall|j: int| 0 <= j < scheduled_ids@.len() implies
                cu_k_values@[j] < #[trigger] cu_k_values@[j + 1]
            by {
                assert(decode_input_row_at(
                    self, scheduled_ids@, input_values@, position_values@,
                    cu_k_values@, max_seqlen_k, j,
                ));
            }
            assert forall|j: int| #![trigger cu_q_values@[j + 1]]
                0 <= j < scheduled_ids@.len() implies {
                let q_len = cu_q_values@[j + 1] as int
                    - cu_q_values@[j] as int;
                let k_len = cu_k_values@[j + 1] as int
                    - cu_k_values@[j] as int;
                &&& q_len <= max_seqlen_q as int
                &&& k_len <= max_seqlen_k as int
                &&& q_len <= k_len
            } by {
                assert(cu_q_values@[j] == j);
                assert(cu_q_values@[j + 1] == j + 1);
                assert(decode_input_row_at(
                    self, scheduled_ids@, input_values@, position_values@,
                    cu_k_values@, max_seqlen_k, j,
                ));
                assert(scheduled_ids@.len() > 0);
            }
            assert forall|k: int| 0 <= k < scheduled_ids@.len() implies
                #[trigger] plan_data_at(
                    old(self), self, scheduled_ids@, input_values@,
                    position_values@, cu_q_values@, cu_k_values@,
                    block_rows@, k,
                )
            by {
                let rid = scheduled_ids@[k];
                assert(old(self).running@.contains(rid));
                assert(self.live_requests@ == old(self).live_requests@);
                assert(self.request_residency@ == old(self).request_residency@);
                assert(cu_q_values@[k] == k);
                assert(cu_q_values@[k + 1] == k + 1);
                assert(decode_input_row_at(
                    self, scheduled_ids@, input_values@, position_values@,
                    cu_k_values@, max_seqlen_k, k,
                ));
                assert(block_rows@[k]@
                    == self.request_residency@[rid].block_ids@);
            }
            assert(plan_data_inv(
                old(self), self, scheduled_ids@, input_values@,
                position_values@, cu_q_values@, cu_k_values@, block_rows@,
            ));
            lemma_raw_plan_sample_policy_decode(
                old(self), scheduled_ids@, cu_k_values@, sample_mask@,
            );
        }
        // The current-step exclusion set remains empty: newly admitted hashes
        // are published only after this loop, so every candidate visible here
        // was already reusable at plan entry.
        let step_hashes: Vec<u64> = Vec::new();
        proof {
            assert(step_hashes@ =~= Seq::<u64>::empty());
            assert(registry_entries_from_pre(
                old(self), self, step_hashes@,
            ));
            lemma_admitted_prefixes_from_pre_running(
                old(self), self, scheduled_ids@, block_rows@,
            );
        }

        // ---- Admission loop: waiting -> running, extending the same batch. ----
        proof { lemma_free_queue_valid_to_token(self); }
        while self.waiting.len() > 0 && scheduled_ids.len() < self.config.max_num_seqs
            invariant
                cs_valid(self),
                free_queue_valid_token(self),
                self.config == old(self).config,
                self.num_blocks == old(self).num_blocks,
                self.num_blocks <= u64::MAX / BLOCK_SIZE,
                self.live_requests@ == old(self).live_requests@,
                self.accepted_requests@ == old(self).accepted_requests@,
                prefill_plan_ready(old(self)),
                scheduled_ids@.no_duplicates(),
                forall|r: RequestId| #[trigger] scheduled_ids@.contains(r)
                    ==> self.running@.contains(r),
                forall|r: RequestId| #[trigger] self.running@.contains(r)
                    ==> old(self).running@.contains(r) || scheduled_ids@.contains(r),
                forall|r: RequestId| #[trigger] old(self).running@.contains(r)
                    ==> self.running@.contains(r),
                forall|r: RequestId| #[trigger] old(self).running@.contains(r)
                    ==> old(self).request_residency@.contains_key(r)
                        && self.request_residency@.contains_key(r)
                        && self.request_residency@[r]
                            == old(self).request_residency@[r],
                residency_history_aligned(self),
                slot_mapping_aligned(self),
                residency_running_aligned(self),
                forall|r: RequestId| #[trigger] self.waiting@.contains(r)
                    ==> old(self).waiting@.contains(r),
                // Capacity by construction: the batch's commit-time block
                // debt (one fresh block per full-tail request) stays within
                // the free pool.
                commit_debt as int
                    == full_tail_debt(self.live_requests@, scheduled_ids@),
                commit_debt <= self.free_blocks,
                // Per-row slot segments.
                plan_seg_inv(old(self), self, scheduled_ids@, cu_q_values@,
                    cu_k_values@, slot_values@),
                // Full per-row forward payload: tokens, positions, key
                // lengths, and block-table rows.
                plan_data_inv(
                    old(self), self, scheduled_ids@, input_values@,
                    position_values@, cu_q_values@, cu_k_values@,
                    block_rows@,
                ),
                raw_plan_sample_policy(
                    old(self), scheduled_ids@, cu_k_values@, sample_mask@,
                ),
                sample_mask@.len() == scheduled_ids@.len(),
                registry_entries_from_pre(old(self), self, step_hashes@),
                positive_provenance_origin(old(self), self),
                admitted_prefixes_from_pre(
                    old(self), self, scheduled_ids@, block_rows@,
                ),
                persistent_provenance_closed(self),
                // Function-requires restated for the isolated loop body.
                tail_write_exclusive(old(self)),
                // Decode-base rows come from the pre-plan running queue.
                decode_count as int <= scheduled_ids@.len(),
                forall|k: int| 0 <= k < decode_count as int
                    ==> old(self).running@.contains(#[trigger] scheduled_ids@[k]),
                // Pre-running tails untouched (hash 0 means the
                // prefix matcher can never bump them)...
                forall|r: RequestId| #[trigger] old(self).running@.contains(r)
                    && old(self).live_requests@.contains_key(r)
                    ==> {
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
                    },
                // ...and admitted rows' suffix pages stay exclusively owned
                // (their nonzero hashes sit in the exclusion list).
                forall|k: int| decode_count as int <= k < scheduled_ids@.len()
                    ==> #[trigger] admitted_pages_exclusive_at(self,
                        scheduled_ids@, step_hashes@, k),
                forall|k: int| 0 <= k < scheduled_ids@.len() ==> {
                    let srid = #[trigger] scheduled_ids@[k];
                    self.live_requests@.contains_key(srid) ==> can_step(
                        self.live_requests@[srid],
                    ) && self.live_requests@[srid].generated_tokens@.len()
                        < usize::MAX as int
                },
                forall|k: int| 0 <= k < scheduled_ids@.len() ==> {
                    let srid = #[trigger] scheduled_ids@[k];
                    old(self).running@.contains(srid) || old(self).waiting@.contains(srid)
                },
                forall|k: int| decode_count as int <= k < scheduled_ids@.len()
                    ==> old(self).waiting@.contains(#[trigger] scheduled_ids@[k]),
                // Data-vector bookkeeping (step_plan_shape_ok).
                input_values@.len() == position_values@.len(),
                slot_values@.len() == input_values@.len(),
                block_rows@.len() == scheduled_ids@.len(),
                cu_q_values@.len() == scheduled_ids@.len() + 1,
                cu_k_values@.len() == cu_q_values@.len(),
                cu_q_values@[0] == 0,
                cu_k_values@[0] == 0,
                forall|j: int| 0 <= j < scheduled_ids@.len() as int ==>
                    cu_q_values@[j] < #[trigger] cu_q_values@[j + 1],
                forall|j: int| 0 <= j < scheduled_ids@.len() as int ==>
                    cu_k_values@[j] < #[trigger] cu_k_values@[j + 1],
                forall|j: int| #![trigger cu_q_values@[j + 1]]
                    0 <= j < scheduled_ids@.len() as int ==> {
                    let q_len = cu_q_values@[j + 1] as int
                        - cu_q_values@[j] as int;
                    let k_len = cu_k_values@[j + 1] as int
                        - cu_k_values@[j] as int;
                    &&& q_len <= max_seqlen_q as int
                    &&& k_len <= max_seqlen_k as int
                    &&& q_len <= k_len
                },
                cu_q_values@[scheduled_ids@.len() as int] as int
                    == input_values@.len() as int,
            decreases self.waiting@.len(),
        {
            proof { lemma_free_queue_token_to_valid(self); }
            // Match once to size pressure from the exact fresh suffix, reclaim
            // from the intrusive queue, then rematch.  Only the second match is
            // attached, so eviction order can change hit rate but never the
            // validity of the reuse certificate.
            let pressure_rid = self.waiting[0];
            let mut pressure_match: Vec<BlockId> = Vec::new();
            let pressure_state_opt = self.live_requests.get(&pressure_rid);
            match pressure_state_opt {
                Some(pressure_state_ref) => {
                    let pressure_tokens = pressure_state_ref.prompt_tokens.clone();
                    let pressure_cu_last = cu_q_values[cu_q_values.len() - 1];
                    let pressure_k_last = cu_k_values[cu_k_values.len() - 1];
                    proof {
                        assert(self.waiting@[0] == pressure_rid);
                        assert(self.waiting@.contains(pressure_rid));
                        assert(old(self).waiting@.contains(pressure_rid));
                        assert(self.live_requests@.contains_key(pressure_rid));
                        assert(old(self).live_requests@.contains_key(pressure_rid));
                        assert(can_step(old(self).live_requests@[pressure_rid]));
                        assert(valid_request_state(
                            old(self).live_requests@[pressure_rid]));
                        assert(pressure_tokens@
                            == old(self).live_requests@[pressure_rid]
                                .prompt_tokens@);
                        assert(pressure_tokens@.len() > 0);
                        assert(pressure_tokens@.len() <= u64::MAX as int);
                    }
                    let ghost pressure_pre = *self;
                    pressure_match = self.prepare_prefill_match_under_pressure(
                        &pressure_tokens,
                        &step_hashes,
                        pressure_cu_last,
                        pressure_k_last,
                        commit_debt,
                    );
                    proof {
                            // Restore every non-`cs_valid` loop companion.
                            // Reclamation preserves all resident/admitted
                            // pages because each has positive refcount.
                            lemma_residency_history_aligned_positive_frame(
                                &pressure_pre, self,
                            );
                            lemma_slot_mapping_aligned_frame(&pressure_pre, self);
                            lemma_residency_running_aligned_frame(&pressure_pre, self);
                            lemma_plan_seg_inv_frame(
                                old(self), &pressure_pre, self,
                                scheduled_ids@, cu_q_values@, cu_k_values@,
                                slot_values@,
                            );
                            lemma_plan_data_inv_frame(
                                old(self), &pressure_pre, self,
                                scheduled_ids@, input_values@, position_values@,
                                cu_q_values@, cu_k_values@, block_rows@,
                            );
                            lemma_registry_entries_from_pre_reclaim(
                                old(self), &pressure_pre, self, step_hashes@,
                            );
                            lemma_positive_provenance_origin_transitive(
                                old(self), &pressure_pre, self,
                            );
                            lemma_admitted_current_chains_positive_frame(
                                old(self), &pressure_pre, self,
                                scheduled_ids@, block_rows@,
                            );
                            lemma_admitted_prefixes_from_pre_frame(
                                old(self), &pressure_pre, self,
                                scheduled_ids@, block_rows@,
                            );
                            assert forall|r: RequestId|
                                #[trigger] old(self).running@.contains(r)
                                && old(self).live_requests@.contains_key(r)
                                implies {
                                    let t = old(self).request_residency@[r]
                                        .block_ids@[old(self).request_residency@[r]
                                            .block_ids@.len() - 1];
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
                                let t = old(self).request_residency@[r]
                                    .block_ids@[old(self).request_residency@[r]
                                        .block_ids@.len() - 1];
                                assert(pressure_pre.blocks@.contains_key(t));
                                assert(pressure_pre.blocks@[t].refcount
                                    == old(self).blocks@[t].refcount);
                                assert(old(self).blocks@[t].refcount == 1) by {
                                    assert(tail_write_exclusive(old(self)));
                                }
                                assert(pressure_pre.blocks@[t].refcount > 0);
                                assert(self.blocks@[t] == pressure_pre.blocks@[t]);
                            }
                            assert forall|k: int|
                                decode_count as int <= k < scheduled_ids@.len()
                                implies #[trigger] admitted_pages_exclusive_at(
                                    self, scheduled_ids@, step_hashes@, k,
                                )
                            by {
                                lemma_admitted_pages_positive_frame(
                                    &pressure_pre, self, scheduled_ids@,
                                    step_hashes@, k,
                                );
                            }
                    }
                },
                None => {
                    proof {
                        assert(self.waiting@.contains(pressure_rid));
                        assert(live_covers_queue(self));
                        assert(false);
                    }
                },
            }
            let ghost loop_pre = *self;
            let ghost sched_pre = scheduled_ids@;
            let ghost input_vals_iter_pre = input_values@;
            let ghost position_vals_iter_pre = position_values@;
            let ghost slot_vals_iter_pre = slot_values@;
            let ghost cu_q_iter_pre = cu_q_values@;
            let ghost cu_k_vals_iter_pre = cu_k_values@;
            let ghost sample_mask_iter_pre = sample_mask@;
            let ghost block_rows_iter_pre = block_rows@;
            let ghost sh_iter_pre = step_hashes@;
            let rid = self.waiting.remove(0);
            proof {
                assert(loop_pre.waiting@.contains(rid));
                assert(!loop_pre.request_residency@.contains_key(rid)) by {
                    assert(waiting_has_no_residency(&loop_pre));
                }
                assert(!loop_pre.running@.contains(rid)) by {
                    assert(queue_disjoint(&loop_pre));
                }
                assert(self.waiting@ == loop_pre.waiting@.subrange(
                    1, loop_pre.waiting@.len() as int));
                assert(!self.waiting@.contains(rid)) by {
                    assert(waiting_unique(&loop_pre));
                    if self.waiting@.contains(rid) {
                        let idx = choose|idx: int| 0 <= idx < self.waiting@.len()
                            && self.waiting@[idx] == rid;
                        assert(loop_pre.waiting@[idx + 1] == rid);
                        assert(loop_pre.waiting@[0] == rid);
                        assert(false);
                    }
                }
                assert forall|r: RequestId| #[trigger] self.waiting@.contains(r)
                    implies loop_pre.waiting@.contains(r) by {
                    let idx = choose|idx: int| 0 <= idx < self.waiting@.len()
                        && self.waiting@[idx] == r;
                    assert(loop_pre.waiting@[idx + 1] == r);
                }
                assert(waiting_unique(self)) by {
                    assert forall|a: int, b: int|
                        0 <= a < self.waiting@.len() && 0 <= b < self.waiting@.len()
                        && a != b
                        implies self.waiting@[a] != self.waiting@[b] by {
                        assert(self.waiting@[a] == loop_pre.waiting@[a + 1]);
                        assert(self.waiting@[b] == loop_pre.waiting@[b + 1]);
                    }
                    assert(self.waiting@.no_duplicates()) by {
                        reveal(Seq::no_duplicates);
                    }
                }
                assert(cs_valid(self));
                lemma_free_queue_valid_to_token(&loop_pre);
                lemma_free_queue_valid_token_frame(&loop_pre, self);
                lemma_free_queue_token_to_valid(self);
                // rid cannot already be scheduled: scheduled ids sit in the
                // running queue, and rid was still waiting (queue_disjoint).
                assert(!sched_pre.contains(rid)) by {
                    if sched_pre.contains(rid) {
                        assert(loop_pre.running@.contains(rid));
                    }
                }
            }
            let state_opt = self.live_requests.get(&rid);
            match state_opt {
                Some(state_ref) => {
                    let state = state_ref.clone();
                    proof {
                        assert(old(self).waiting@.contains(rid));
                        assert(old(self).live_requests@.contains_key(rid));
                        assert(state.prompt_tokens@
                            == old(self).live_requests@[rid].prompt_tokens@);
                        assert(can_step(old(self).live_requests@[rid]));
                        assert(state.prompt_tokens@.len() > 0);
                        assert(state.prompt_tokens@.len() <= u64::MAX as int);
                        assert(state.prompt_tokens@.len() <= usize::MAX as int);
                        assert(!self.request_residency@.contains_key(rid));
                        assert(!self.running@.contains(rid));
                    }
                    // Token budget applies to this row's query chunk. Cached
                    // prefix tokens are key context and do not consume query
                    // budget; a non-final chunk is page-aligned.
                    let n_tokens = state.prompt_tokens.len();
                    let cu_last = cu_q_values[cu_q_values.len() - 1];
                    let k_last = cu_k_values[cu_k_values.len() - 1];
                    let matched_start = pressure_match.len()
                        * (BLOCK_SIZE as usize);
                    proof {
                        assert(state.prompt_tokens@.len() > 0);
                        assert(pressure_match@.len() as int
                            * (BLOCK_SIZE_SPEC as int)
                            < state.prompt_tokens@.len() as int);
                        assert(matched_start < n_tokens);
                    }
                    let remaining = n_tokens - matched_start;
                    let used = cu_last as usize;
                    let available = if used >= self.config.max_num_batched_tokens {
                        0
                    } else {
                        self.config.max_num_batched_tokens - used
                    };
                    let chunk_len = select_prefill_chunk_len(remaining, available);
                    let row_end = matched_start + chunk_len;
                    let final_chunk = row_end == n_tokens;
                    let fits_budget = chunk_len > 0;
                    // Capacity by construction: the protected pressure pass
                    // targeted exactly the unmatched suffix plus the batch's
                    // commit debt and this request's own commit-time head.
                    let inc: u64 = if n_tokens % (BLOCK_SIZE as usize) == 0 {
                        1
                    } else {
                        0
                    };
                    let prior_commit_debt = commit_debt;
                    proof {
                        assert(self.free_blocks <= self.num_blocks) by {
                            assert(block_count_valid(self));
                            assert(self.blocks@.dom().len() >= 0);
                        }
                        assert(commit_debt as int + inc as int <= u64::MAX as int);
                        assert(n_tokens as int <= u64::MAX as int);
                    }
                    let fresh_for_prompt = Self::fresh_prefill_blocks(
                        &state.prompt_tokens, &pressure_match,
                    );
                    let matched_count = pressure_match.len() as u64;
                    let headroom_ok = fresh_for_prompt <= self.free_blocks
                        && prior_commit_debt + inc
                            <= self.free_blocks - fresh_for_prompt;
                    let fits_q = chunk_len as u64 <= u64::MAX - cu_last;
                    let fits_k = row_end as u64 <= u64::MAX - k_last;
                    let fits = fits_budget && headroom_ok && fits_q && fits_k;
                    if !fits {
                        self.waiting.insert(0, rid);
                        proof {
                            assert(self.waiting@ == loop_pre.waiting@);
                            assert(self.request_residency@ == loop_pre.request_residency@);
                            assert(self.running@ == loop_pre.running@);
                            assert(self.blocks@ == loop_pre.blocks@);
                            assert(self.free_blocks == loop_pre.free_blocks);
                            assert(commit_debt <= loop_pre.free_blocks);
                            lemma_persistent_provenance_closed_blocks_eq(
                                &loop_pre, self,
                            );
                            lemma_residency_running_aligned_frame(&loop_pre, self);
                            assert(cs_valid(self));
                            // Segment invariant survives: this path leaves
                            // the residency and live maps untouched.
                            assert(plan_seg_inv(old(self), &loop_pre,
                                scheduled_ids@, cu_q_values@, cu_k_values@,
                                slot_values@));
                            lemma_plan_seg_inv_frame(old(self), &loop_pre, self,
                                scheduled_ids@, cu_q_values@, cu_k_values@,
                                slot_values@);
                            assert(plan_data_inv(
                                old(self), &loop_pre, scheduled_ids@,
                                input_values@, position_values@, cu_q_values@,
                                cu_k_values@, block_rows@,
                            ));
                            lemma_plan_data_inv_frame(
                                old(self), &loop_pre, self, scheduled_ids@,
                                input_values@, position_values@, cu_q_values@,
                                cu_k_values@, block_rows@,
                            );
                            // Write-page exclusivity is preserved: blocks are untouched.
                            assert(self.blocks@ == loop_pre.blocks@);
                            assert forall|r: RequestId|
                                #[trigger] old(self).running@.contains(r)
                                && old(self).live_requests@.contains_key(r)
                                implies {
                                    let t = old(self).request_residency@[r]
                                        .block_ids@[old(self).request_residency@[r]
                                            .block_ids@.len() - 1];
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
                                let t = old(self).request_residency@[r]
                                    .block_ids@[old(self).request_residency@[r]
                                        .block_ids@.len() - 1];
                                assert(loop_pre.blocks@.contains_key(t));
                            }
                            assert forall|k: int|
                                decode_count as int <= k < scheduled_ids@.len()
                                implies #[trigger] admitted_pages_exclusive_at(
                                    self, scheduled_ids@, step_hashes@, k)
                            by {
                                assert(admitted_pages_exclusive_at(&loop_pre,
                                    scheduled_ids@, step_hashes@, k));
                                lemma_admitted_pages_frame(&loop_pre, self,
                                    scheduled_ids@, step_hashes@, k);
                            }
                            lemma_registry_entries_from_pre_allocate(
                                old(self), &loop_pre, self, step_hashes@,
                            );
                            lemma_positive_provenance_origin_blocks_eq(
                                &loop_pre, self,
                            );
                            lemma_positive_provenance_origin_transitive(
                                old(self), &loop_pre, self,
                            );
                            lemma_admitted_current_chains_positive_frame(
                                old(self), &loop_pre, self,
                                scheduled_ids@, block_rows@,
                            );
                            lemma_admitted_prefixes_from_pre_frame(
                                old(self), &loop_pre, self,
                                scheduled_ids@, block_rows@,
                            );
                        }
                        proof {
                            lemma_free_queue_valid_token_frame(&loop_pre, self);
                        }
                        break;
                    }
                    proof {
                        // Query and key cumulative ends are independently
                        // bounded: the selector accounts only the fresh query
                        // chunk, while `fits_k` guards the row's key end.
                        if sched_pre.len() == 0 {
                            assert(cu_last == cu_q_values@[0]);
                        }
                        assert(cu_last as int + chunk_len as int
                            <= u64::MAX as int);
                        assert(k_last as int + row_end as int
                            <= u64::MAX as int);
                    }
                    let ghost pre_alloc = *self;
                    proof {
                        assert(pre_alloc.blocks@ == loop_pre.blocks@);
                        lemma_persistent_provenance_closed_blocks_eq(
                            &loop_pre, &pre_alloc,
                        );
                        lemma_registry_entries_from_pre_allocate(
                            old(self), &loop_pre, &pre_alloc, sh_iter_pre,
                        );
                        lemma_positive_provenance_origin_blocks_eq(
                            &loop_pre, &pre_alloc,
                        );
                        lemma_positive_provenance_origin_transitive(
                            old(self), &loop_pre, &pre_alloc,
                        );
                    }
                    let admitted = self.allocate_prefill_from_match(
                        rid, &state.prompt_tokens, &step_hashes, pressure_match,
                    );
                    let ghost post_reuse = *self;
                    if admitted {
                        let cpb: u64 = match self.request_residency.get(&rid) {
                            Some(r) => r.cached_prefix_blocks,
                            None => {
                                proof { assert(false); }
                                0
                            },
                        };
                        let ghost n_g = state.prompt_tokens@.len() as int;
                        let ghost c_g = cpb as int;
                        proof {
                            assert(allocate_prefill_reuse_success(&pre_alloc, self, rid,
                                state.prompt_tokens@, c_g));
                            assert(cpb == matched_count);
                            lemma_registry_entries_from_pre_allocate(
                                old(self), &pre_alloc, &post_reuse, sh_iter_pre,
                            );
                            lemma_positive_provenance_origin_transitive(
                                old(self), &pre_alloc, &post_reuse,
                            );
                            lemma_reused_prefix_from_pre(
                                old(self), &pre_alloc, &post_reuse,
                                sh_iter_pre, rid, state.prompt_tokens@, c_g,
                            );
                            assert(c_g * (BLOCK_SIZE_SPEC as int) < n_g);
                            // cpb fits in usize and c*BS fits too.
                            assert(c_g <= c_g * (BLOCK_SIZE_SPEC as int)) by (nonlinear_arith)
                                requires 0 <= c_g, BLOCK_SIZE_SPEC as int == 64,
                            {}
                        }
                        let c_us: usize = cpb as usize;
                        let sfx_start: usize = c_us * (BLOCK_SIZE as usize);
                        proof {
                            assert(sfx_start as int == c_g * (BLOCK_SIZE_SPEC as int));
                            assert(self.request_residency@.contains_key(rid));
                            assert(self.request_residency@[rid].slot_mapping@.len()
                                == n_g - c_g * (BLOCK_SIZE_SPEC as int));
                            // register's block-count requires, via the split.
                            if c_g > 0 {
                                lemma_blocks_needed_split(state.prompt_tokens@.len() as nat,
                                    cpb as nat);
                            }
                            assert(self.request_residency@[rid].block_ids@.len() as int
                                == blocks_needed_for(state.prompt_tokens@.len() as nat) as int);
                            // The FRESH tail is untargeted by the hash registry
                            // (its targets predate allocation; reused prefix
                            // blocks are of course registered).
                            assert(self.hash_to_block@ == pre_alloc.hash_to_block@);
                            assert forall|h: u64|
                                #[trigger] self.hash_to_block@.contains_key(h)
                                implies !self.request_residency@[rid].block_ids@.subrange(
                                    c_us as int,
                                    self.request_residency@[rid].block_ids@.len() as int,
                                ).contains(self.hash_to_block@[h])
                            by {
                                assert(pre_alloc.hash_to_block@.contains_key(h));
                                assert(hash_to_block_in_range(&pre_alloc));
                                assert(pre_alloc.blocks@.contains_key(
                                    pre_alloc.hash_to_block@[h]));
                                let ids_g = self.request_residency@[rid].block_ids@;
                                if ids_g.subrange(c_us as int, ids_g.len() as int)
                                    .contains(self.hash_to_block@[h]) {
                                    let idx = ids_g.subrange(c_us as int, ids_g.len() as int)
                                        .index_of(self.hash_to_block@[h]);
                                    assert(ids_g[c_us as int + idx]
                                        == self.hash_to_block@[h]);
                                    assert(!pre_alloc.blocks@.contains_key(
                                        ids_g[c_us as int + idx]));
                                }
                            }
                        }
                        // Registration is deferred until every admission has
                        // been decided, so this forward cannot consume K/V
                        // that it has not materialized yet.
                        let ghost pre_push = *self;
                        self.running.push(rid);
                        scheduled_ids.push(rid);
                        commit_debt = prior_commit_debt + inc;
                        proof {
                            // cs_valid after the running push: the queue-facing
                            // conjuncts get rid's facts from the pop + allocation.
                            assert(self.running@ == pre_push.running@.push(rid));
                            assert(!pre_push.running@.contains(rid));
                            assert(running_unique(self)) by {
                                assert(pre_push.running@.no_duplicates());
                                reveal(Seq::no_duplicates);
                            }
                            assert(running_has_residency(self)) by {
                                assert forall|r: RequestId|
                                    #[trigger] self.running@.contains(r)
                                    implies self.request_residency@.contains_key(r) by {
                                    let idx = choose|idx: int|
                                        0 <= idx < self.running@.len()
                                        && self.running@[idx] == r;
                                    if idx < self.running@.len() - 1 {
                                        assert(pre_push.running@[idx] == r);
                                        assert(pre_push.running@.contains(r));
                                    } else {
                                        assert(r == rid);
                                    }
                                }
                            }
                            assert(queue_disjoint(self)) by {
                                assert forall|r: RequestId|
                                    #[trigger] self.running@.contains(r)
                                    implies !self.waiting@.contains(r) by {
                                    let idx = choose|idx: int|
                                        0 <= idx < self.running@.len()
                                        && self.running@[idx] == r;
                                    if idx < self.running@.len() - 1 {
                                        assert(pre_push.running@[idx] == r);
                                        assert(pre_push.running@.contains(r));
                                    } else {
                                        assert(r == rid);
                                        assert(!self.waiting@.contains(rid));
                                    }
                                }
                            }
                            assert(live_covers_queue(self)) by {
                                assert forall|r: RequestId|
                                    self.running@.contains(r) || self.waiting@.contains(r)
                                    implies #[trigger] self.live_requests@.contains_key(r) by {
                                    if self.waiting@.contains(r) {
                                        assert(pre_push.waiting@.contains(r));
                                    } else {
                                        let idx = choose|idx: int|
                                            0 <= idx < self.running@.len()
                                            && self.running@[idx] == r;
                                        if idx < self.running@.len() - 1 {
                                            assert(pre_push.running@[idx] == r);
                                            assert(pre_push.running@.contains(r));
                                        } else {
                                            assert(r == rid);
                                        }
                                    }
                                }
                            }
                            assert(cs_valid(self));
                            assert(pre_push.request_residency@.dom()
                                == pre_alloc.request_residency@.dom().insert(rid));
                            assert(pre_alloc.request_residency@
                                == loop_pre.request_residency@);
                            assert(pre_push.running@ == loop_pre.running@);
                            lemma_residency_running_aligned_admit(
                                &loop_pre, self, rid,
                            );
                            lemma_registry_entries_from_pre_allocate(
                                old(self), &post_reuse, self, step_hashes@,
                            );
                            lemma_positive_provenance_origin_blocks_eq(
                                &post_reuse, self,
                            );
                            lemma_positive_provenance_origin_transitive(
                                old(self), &post_reuse, self,
                            );
                            // Extend the scheduled-set invariants.
                            assert(scheduled_ids@ == sched_pre.push(rid));
                            assert(scheduled_ids@.no_duplicates()) by {
                                reveal(Seq::no_duplicates);
                            }
                            assert forall|r: RequestId| #[trigger] scheduled_ids@.contains(r)
                                implies self.running@.contains(r) by {
                                let idx = choose|idx: int|
                                    0 <= idx < scheduled_ids@.len()
                                    && scheduled_ids@[idx] == r;
                                if idx < scheduled_ids@.len() - 1 {
                                    assert(sched_pre[idx] == r);
                                    assert(sched_pre.contains(r));
                                    assert(pre_push.running@.contains(r));
                                    let ridx = choose|ridx: int|
                                        0 <= ridx < pre_push.running@.len()
                                        && pre_push.running@[ridx] == r;
                                    assert(self.running@[ridx] == r);
                                } else {
                                    assert(r == rid);
                                    assert(self.running@[self.running@.len() - 1] == rid);
                                }
                            }
                            assert forall|r: RequestId| #[trigger] self.running@.contains(r)
                                implies old(self).running@.contains(r)
                                    || scheduled_ids@.contains(r) by {
                                let ridx = choose|ridx: int|
                                    0 <= ridx < self.running@.len()
                                    && self.running@[ridx] == r;
                                if ridx < self.running@.len() - 1 {
                                    assert(pre_push.running@[ridx] == r);
                                    assert(pre_push.running@.contains(r));
                                    if !old(self).running@.contains(r) {
                                        assert(sched_pre.contains(r));
                                        let sidx = choose|sidx: int|
                                            0 <= sidx < sched_pre.len()
                                            && sched_pre[sidx] == r;
                                        assert(scheduled_ids@[sidx] == r);
                                    }
                                } else {
                                    assert(r == rid);
                                    assert(scheduled_ids@[scheduled_ids@.len() - 1] == rid);
                                }
                            }
                            assert forall|r: RequestId|
                                #[trigger] old(self).running@.contains(r)
                                implies self.running@.contains(r) by {
                                assert(pre_push.running@.contains(r));
                                let ridx = choose|ridx: int|
                                    0 <= ridx < pre_push.running@.len()
                                    && pre_push.running@[ridx] == r;
                                assert(self.running@[ridx] == r);
                            }
                        }
                        let ghost inputs_head = input_values@.len();
                        let ghost cu_k_iter_pre = cu_k_values@;
                        let ghost max_q_iter_pre = max_seqlen_q;
                        let ghost max_k_iter_pre = max_seqlen_k;
                        let mut j: usize = sfx_start;
                        while j < row_end
                            invariant
                                sfx_start <= j,
                                j <= row_end,
                                row_end as int <= state.prompt_tokens@.len(),
                                inputs_head == input_vals_iter_pre.len(),
                                inputs_head == position_vals_iter_pre.len(),
                                input_values@.len()
                                    == inputs_head + (j as int - sfx_start as int),
                                position_values@.len() == input_values@.len(),
                                forall|q: int| 0 <= q < inputs_head as int ==>
                                    #[trigger] input_values@[q]
                                        == input_vals_iter_pre[q],
                                forall|q: int| 0 <= q < inputs_head as int ==>
                                    #[trigger] position_values@[q]
                                        == position_vals_iter_pre[q],
                                forall|q: int|
                                    inputs_head as int <= q < input_values@.len()
                                    ==> {
                                        let p = sfx_start as int
                                            + q - inputs_head as int;
                                        &&& 0 <= p < state.prompt_tokens@.len()
                                        &&& #[trigger] input_values@[q]
                                            == state.prompt_tokens@[p]
                                        &&& position_values@[q] as int == p
                                    },
                            decreases row_end - j,
                        {
                            input_values.push(state.prompt_tokens[j]);
                            position_values.push(j as u64);
                            j = j + 1;
                        }
                        let sfx_len: usize = row_end - sfx_start;
                        proof {
                            assert(sfx_len as int
                                == row_end as int
                                    - c_g * (BLOCK_SIZE_SPEC as int));
                            assert(sfx_len >= 1);
                            assert(cu_last as int + sfx_len as int <= u64::MAX as int);
                        }
                        let next_cu = cu_last + sfx_len as u64;
                        cu_q_values.push(next_cu);
                        let next_k = k_last + row_end as u64;
                        cu_k_values.push(next_k);
                        if sfx_len > max_seqlen_q {
                            max_seqlen_q = sfx_len;
                        }
                        if row_end > max_seqlen_k {
                            max_seqlen_k = row_end;
                        }
                        let ghost slots_head = slot_values@.len();
                        let residency_opt = self.request_residency.get(&rid);
                        match residency_opt {
                            Some(residency) => {
                                proof {
                                    assert(residency.slot_mapping@.len()
                                        == n_g - c_g * (BLOCK_SIZE_SPEC as int));
                                    assert(sfx_len as int
                                        <= residency.slot_mapping@.len());
                                }
                                block_rows.push(residency.block_ids.clone());
                                proof {
                                    assert(slot_values@ == slot_vals_iter_pre);
                                    assert(slots_head == slot_vals_iter_pre.len());
                                }
                                let mut k2: usize = 0;
                                while k2 < sfx_len
                                    invariant
                                        k2 <= sfx_len,
                                        sfx_len as int <= residency.slot_mapping@.len(),
                                        slot_values@.len() == slots_head + k2 as int,
                                        forall|q: int| 0 <= q < slots_head
                                            ==> #[trigger] slot_values@[q]
                                                == slot_vals_iter_pre[q],
                                        forall|q: int|
                                            slots_head as int <= q < slot_values@.len()
                                            ==> #[trigger] slot_values@[q]
                                                == residency.slot_mapping@[
                                                    q - slots_head as int],
                                    decreases sfx_len - k2,
                                {
                                    slot_values.push(residency.slot_mapping[k2]);
                                    k2 = k2 + 1;
                                }
                                proof {
                                    // Segment invariant restored: previous
                                    // rows framed; the new row is the
                                    // admitted suffix at global positions.
                                    assert(slots_head == cu_last as int);
                                    assert(residency.slot_mapping@
                                        == self.request_residency@[rid].slot_mapping@);
                                    assert(residency.block_ids@
                                        == self.request_residency@[rid].block_ids@);
                                    assert(!old(self).running@.contains(rid)) by {
                                        assert(loop_pre.waiting@[0] == rid);
                                        assert(loop_pre.waiting@.contains(rid));
                                        assert(cs_valid(&loop_pre));
                                        assert(queue_disjoint(&loop_pre));
                                        assert(!loop_pre.running@.contains(rid));
                                        if old(self).running@.contains(rid) {
                                            assert(loop_pre.running@.contains(rid));
                                        }
                                    }
                                    assert forall|k: int|
                                        0 <= k < scheduled_ids@.len()
                                        implies #[trigger] plan_seg_at(old(self),
                                            self, scheduled_ids@, cu_q_values@,
                                            cu_k_values@, slot_values@, k)
                                    by {
                                        let srid = scheduled_ids@[k];
                                        if k < scheduled_ids@.len() - 1 {
                                            assert(srid == sched_pre[k]);
                                            assert(plan_seg_at(old(self), &loop_pre,
                                                sched_pre, cu_q_iter_pre,
                                                cu_k_vals_iter_pre,
                                                slot_vals_iter_pre, k));
                                            assert(sched_pre.contains(srid));
                                            assert(loop_pre.running@.contains(srid));
                                            assert(srid != rid) by {
                                                assert(!loop_pre.running@.contains(rid));
                                            }
                                            assert(self.request_residency@.contains_key(srid)
                                                && self.request_residency@[srid]
                                                    == loop_pre.request_residency@[srid]);
                                            assert(self.live_requests@[srid]
                                                == loop_pre.live_requests@[srid]);
                                            assert(cu_q_values@[k] == cu_q_iter_pre[k]
                                                && cu_q_values@[k + 1]
                                                    == cu_q_iter_pre[k + 1]);
                                            assert(cu_q_iter_pre[k + 1] as int
                                                <= slot_vals_iter_pre.len());
                                            assert forall|q: int|
                                                cu_q_values@[k] as int <= q
                                                    < cu_q_values@[k + 1] as int
                                                implies #[trigger] slot_values@[q]
                                                    == slot_vals_iter_pre[q]
                                            by {
                                                assert(q < slots_head as int);
                                            }
                                        } else {
                                            assert(srid == rid);
                                            assert(cu_q_values@[k] == cu_last);
                                            assert(cu_q_values@[k + 1] == next_cu);
                                            assert(next_cu as int
                                                == cu_last as int + sfx_len as int);
                                            assert(self.request_residency@[rid]
                                                .cached_prefix_blocks as int == c_g);
                                            assert(self.live_requests@[rid]
                                                .prompt_tokens@.len() as int == n_g);
                                            assert forall|q: int|
                                                cu_q_values@[k] as int <= q
                                                    < cu_q_values@[k + 1] as int
                                                implies #[trigger] slot_values@[q] as int
                                                    == crate::proof::tensor::geometry::block_table_slot(
                                                        self.request_residency@[rid]
                                                            .block_ids@,
                                                        (c_g * (BLOCK_SIZE_SPEC as int)
                                                            + q - cu_q_values@[k] as int)
                                                            as nat) as int
                                            by {
                                                let i2 = q - slots_head as int;
                                                assert(slot_values@[q]
                                                    == residency.slot_mapping@[i2]);
                                                assert(residency.slot_mapping@[i2] as int
                                                    == crate::proof::tensor::geometry::block_table_slot(
                                                        self.request_residency@[rid]
                                                            .block_ids@,
                                                        (c_g * (BLOCK_SIZE_SPEC as int)
                                                            + i2) as nat) as int);
                                            }
                                        }
                                    }
                                    assert(plan_seg_inv(old(self), self,
                                        scheduled_ids@, cu_q_values@,
                                        cu_k_values@, slot_values@));
                                    assert forall|k: int|
                                        0 <= k < scheduled_ids@.len()
                                        implies #[trigger] plan_data_at(
                                            old(self), self, scheduled_ids@,
                                            input_values@, position_values@,
                                            cu_q_values@, cu_k_values@,
                                            block_rows@, k,
                                        )
                                    by {
                                        let srid = scheduled_ids@[k];
                                        if k < scheduled_ids@.len() - 1 {
                                            assert(srid == sched_pre[k]);
                                            assert(plan_data_at(
                                                old(self), &loop_pre, sched_pre,
                                                input_vals_iter_pre,
                                                position_vals_iter_pre,
                                                cu_q_iter_pre,
                                                cu_k_vals_iter_pre,
                                                block_rows_iter_pre, k,
                                            ));
                                            assert(srid != rid) by {
                                                assert(sched_pre.contains(srid));
                                                assert(loop_pre.running@.contains(srid));
                                                assert(!loop_pre.running@.contains(rid));
                                            }
                                            assert(self.live_requests@[srid]
                                                == loop_pre.live_requests@[srid]);
                                            assert(self.request_residency@[srid]
                                                == loop_pre.request_residency@[srid]);
                                            assert(self.live_requests@
                                                .contains_key(srid));
                                            assert(valid_request_state(
                                                self.live_requests@[srid]));
                                            assert(self.request_residency@
                                                .contains_key(srid));
                                            assert(input_values@[cu_q_values@[k] as int]
                                                == input_vals_iter_pre[
                                                    cu_q_iter_pre[k] as int]);
                                            assert(position_values@[
                                                cu_q_values@[k] as int]
                                                == position_vals_iter_pre[
                                                    cu_q_iter_pre[k] as int]);
                                            assert(cu_q_values@[k]
                                                == cu_q_iter_pre[k]);
                                            assert(cu_q_values@[k + 1]
                                                == cu_q_iter_pre[k + 1]);
                                            assert(cu_k_values@[k]
                                                == cu_k_vals_iter_pre[k]);
                                            assert(cu_k_values@[k + 1]
                                                == cu_k_vals_iter_pre[k + 1]);
                                            assert(block_rows@[k]@
                                                == block_rows_iter_pre[k]@);
                                            let row_s0 = cu_q_values@[k] as int;
                                            let row_s1 = cu_q_values@[k + 1] as int;
                                            let row_h = history(self.live_requests@[srid]);
                                            assert(row_s0 >= 0);
                                            assert(row_s1 <= input_vals_iter_pre.len());
                                            assert(row_s1 <= input_values@.len());
                                            assert(position_values@.len()
                                                == input_values@.len());
                                            assert(block_rows@[k]@
                                                == self.request_residency@[srid]
                                                    .block_ids@);
                                            if !old(self).running@.contains(srid) {
                                                let s0 = row_s0;
                                                let s1 = row_s1;
                                                let row_c = self.request_residency@[srid]
                                                    .cached_prefix_blocks as int
                                                    * (BLOCK_SIZE_SPEC as int);
                                                let row_n = self.live_requests@[srid]
                                                    .prompt_tokens@.len() as int;
                                                let row_end = cu_k_values@[k + 1] as int
                                                    - cu_k_values@[k] as int;
                                                assert(s1 == s0 + (row_end - row_c));
                                                assert(0 <= row_c < row_end);
                                                assert(row_end <= row_n);
                                                assert forall|q: int|
                                                    s0 <= q < s1 implies {
                                                    let c = self.request_residency@[srid]
                                                        .cached_prefix_blocks as int
                                                        * (BLOCK_SIZE_SPEC as int);
                                                    let p = c + q - s0;
                                                    &&& #[trigger] input_values@[q]
                                                        == self.live_requests@[srid]
                                                            .prompt_tokens@[p]
                                                    &&& position_values@[q] as int == p
                                                } by {
                                                    assert(0 <= q);
                                                    assert(q < inputs_head as int);
                                                    assert(inputs_head as int
                                                        == input_vals_iter_pre.len());
                                                    assert(position_vals_iter_pre.len()
                                                        == input_vals_iter_pre.len());
                                                    assert(q < input_vals_iter_pre.len());
                                                    assert(q < position_vals_iter_pre.len());
                                                    assert(input_values@[q]
                                                        == input_vals_iter_pre[q]);
                                                    assert(position_values@[q]
                                                        == position_vals_iter_pre[q]);
                                                }
                                                assert(scheduled_ids@[k] == srid);
                                                assert(!old(self).running@
                                                    .contains(scheduled_ids@[k]));
                                                assert(plan_data_at_intro_ready(
                                                    old(self), self, scheduled_ids@,
                                                    input_values@, position_values@,
                                                    cu_q_values@, cu_k_values@,
                                                    block_rows@, k,
                                                ));
                                                lemma_plan_data_at_intro(
                                                    old(self), self, scheduled_ids@,
                                                    input_values@, position_values@,
                                                    cu_q_values@, cu_k_values@,
                                                    block_rows@, k,
                                                );
                                            } else {
                                                assert(row_h
                                                    == history(loop_pre.live_requests@[srid]));
                                                assert(row_s1 == row_s0 + 1);
                                                assert(row_h.len() > 0);
                                                assert(input_values@[row_s0]
                                                    == row_h[row_h.len() - 1]);
                                                assert(position_values@[row_s0] as int
                                                    == row_h.len() as int - 1);
                                                assert(cu_k_values@[k + 1] as int
                                                    == cu_k_values@[k] as int
                                                        + row_h.len() as int);
                                                assert(row_s1 == row_s0 + 1
                                                    && row_h.len() > 0
                                                    && input_values@[row_s0]
                                                        == row_h[row_h.len() - 1]
                                                    && position_values@[row_s0] as int
                                                        == row_h.len() as int - 1
                                                    && cu_k_values@[k + 1] as int
                                                        == cu_k_values@[k] as int
                                                            + row_h.len() as int);
                                                assert(plan_data_at_intro_ready(
                                                    old(self), self, scheduled_ids@,
                                                    input_values@, position_values@,
                                                    cu_q_values@, cu_k_values@,
                                                    block_rows@, k,
                                                ));
                                                lemma_plan_data_at_intro(
                                                    old(self), self, scheduled_ids@,
                                                    input_values@, position_values@,
                                                    cu_q_values@, cu_k_values@,
                                                    block_rows@, k,
                                                );
                                            }
                                            assert(plan_data_at(
                                                old(self), self, scheduled_ids@,
                                                input_values@, position_values@,
                                                cu_q_values@, cu_k_values@,
                                                block_rows@, k,
                                            ));
                                        } else {
                                            assert(srid == rid);
                                            assert(!old(self).running@.contains(rid));
                                            assert(self.live_requests@[rid]
                                                .prompt_tokens@
                                                == state.prompt_tokens@);
                                            assert(self.request_residency@[rid]
                                                .cached_prefix_blocks as int == c_g);
                                            assert(cu_q_values@[k] == cu_last);
                                            assert(cu_q_values@[k + 1] == next_cu);
                                            assert(cu_k_values@[k] == k_last);
                                            assert(cu_k_values@[k + 1] == next_k);
                                            assert(block_rows@[k]@
                                                == self.request_residency@[rid].block_ids@);
                                            assert(inputs_head as int == cu_last as int);
                                            assert(cu_q_values@[k] as int
                                                == inputs_head as int);
                                            assert(cu_q_values@[k + 1] as int
                                                == input_values@.len() as int);
                                            assert forall|q: int|
                                                cu_q_values@[k] as int <= q
                                                    < cu_q_values@[k + 1] as int
                                                implies {
                                                let p = c_g * (BLOCK_SIZE_SPEC as int)
                                                    + q - cu_q_values@[k] as int;
                                                &&& #[trigger] input_values@[q]
                                                    == self.live_requests@[rid]
                                                        .prompt_tokens@[p]
                                                &&& position_values@[q] as int == p
                                            } by {
                                                assert(sfx_start as int
                                                    == c_g * (BLOCK_SIZE_SPEC as int));
                                                assert(inputs_head as int <= q
                                                    < input_values@.len() as int);
                                                assert(input_values@[q]
                                                    == state.prompt_tokens@[
                                                        sfx_start as int + q
                                                            - inputs_head as int]);
                                                assert(position_values@[q] as int
                                                    == sfx_start as int + q
                                                        - inputs_head as int);
                                                assert(sfx_start as int + q
                                                    - inputs_head as int
                                                    == c_g
                                                        * (BLOCK_SIZE_SPEC as int)
                                                        + q
                                                        - cu_q_values@[k] as int);
                                            }
                                            assert(valid_request_state(
                                                self.live_requests@[rid]));
                                            assert(self.live_requests@
                                                .contains_key(rid));
                                            assert(self.request_residency@
                                                .contains_key(rid));
                                            assert(self.live_requests@[rid]
                                                .prompt_tokens@.len() as int == n_g);
                                            assert(n_g
                                                - c_g * (BLOCK_SIZE_SPEC as int) >= 1);
                                            assert(cu_last as int >= 0);
                                            assert(next_cu as int
                                                == input_values@.len() as int);
                                            assert(position_values@.len()
                                                == input_values@.len());
                                            assert(next_cu as int
                                                == cu_last as int + row_end as int
                                                    - c_g * (BLOCK_SIZE_SPEC as int));
                                            assert(next_k as int
                                                == k_last as int + row_end as int);
                                            let c_actual = self.request_residency@[rid]
                                                .cached_prefix_blocks as int
                                                * (BLOCK_SIZE_SPEC as int);
                                            assert(c_actual
                                                == c_g * (BLOCK_SIZE_SPEC as int));
                                            assert forall|q: int|
                                                cu_q_values@[k] as int <= q
                                                    < cu_q_values@[k + 1] as int
                                                implies {
                                                let p = c_actual + q
                                                    - cu_q_values@[k] as int;
                                                &&& #[trigger] input_values@[q]
                                                    == self.live_requests@[rid]
                                                        .prompt_tokens@[p]
                                                &&& position_values@[q] as int == p
                                            } by {
                                                assert(c_actual
                                                    == c_g * (BLOCK_SIZE_SPEC as int));
                                                assert(input_values@[q]
                                                    == self.live_requests@[rid]
                                                        .prompt_tokens@[
                                                            c_g * (BLOCK_SIZE_SPEC as int)
                                                                + q
                                                                - cu_q_values@[k] as int
                                                        ]);
                                                assert(position_values@[q] as int
                                                    == c_g * (BLOCK_SIZE_SPEC as int)
                                                        + q
                                                        - cu_q_values@[k] as int);
                                            }
                                            let raw_s0 = cu_q_values@[k] as int;
                                            let raw_s1 = cu_q_values@[k + 1] as int;
                                            let raw_n = self.live_requests@[rid]
                                                .prompt_tokens@.len() as int;
                                            let raw_c = self.request_residency@[rid]
                                                .cached_prefix_blocks as int
                                                * (BLOCK_SIZE_SPEC as int);
                                            let raw_end = row_end as int;
                                            assert(block_rows@[k]@
                                                == self.request_residency@[rid].block_ids@);
                                            assert(0 <= raw_s0);
                                            assert(raw_s1 <= input_values@.len());
                                            assert(position_values@.len()
                                                == input_values@.len());
                                            assert(raw_s1 == raw_s0 + (raw_end - raw_c));
                                            assert(0 <= raw_c < raw_end);
                                            assert(raw_end <= raw_n);
                                            assert(cu_k_values@[k + 1] as int
                                                == cu_k_values@[k] as int + raw_end);
                                            assert forall|q: int|
                                                raw_s0 <= q < raw_s1 implies {
                                                let p = raw_c + q - raw_s0;
                                                &&& #[trigger] input_values@[q]
                                                    == self.live_requests@[rid]
                                                        .prompt_tokens@[p]
                                                    &&& position_values@[q] as int == p
                                            } by {
                                                assert(raw_s0
                                                    == cu_q_values@[k] as int);
                                                assert(raw_s1
                                                    == cu_q_values@[k + 1] as int);
                                                assert(raw_c == c_actual);
                                                assert(raw_c + q - raw_s0
                                                    == c_actual + q
                                                        - cu_q_values@[k] as int);
                                                assert(input_values@[q]
                                                    == self.live_requests@[rid]
                                                        .prompt_tokens@[
                                                            c_actual + q
                                                                - cu_q_values@[k] as int
                                                        ]);
                                                assert(position_values@[q] as int
                                                    == c_actual + q
                                                        - cu_q_values@[k] as int);
                                            }
                                            assert(self.live_requests@.contains_key(rid));
                                            assert(valid_request_state(
                                                self.live_requests@[rid]));
                                            assert(self.request_residency@
                                                .contains_key(rid));
                                            assert(!old(self).running@.contains(rid));
                                            assert(raw_s0
                                                == cu_q_values@[k] as int);
                                            assert(raw_s1
                                                == cu_q_values@[k + 1] as int);
                                            assert(raw_n
                                                == self.live_requests@[rid]
                                                    .prompt_tokens@.len() as int);
                                            assert(raw_c
                                                == self.request_residency@[rid]
                                                    .cached_prefix_blocks as int
                                                    * (BLOCK_SIZE_SPEC as int));
                                            assert(scheduled_ids@[k] == rid);
                                            assert(!old(self).running@
                                                .contains(scheduled_ids@[k]));
                                            assert(plan_data_at_intro_ready(
                                                old(self), self, scheduled_ids@,
                                                input_values@, position_values@,
                                                cu_q_values@, cu_k_values@,
                                                block_rows@, k,
                                            ));
                                            lemma_plan_data_at_intro(
                                                old(self), self, scheduled_ids@,
                                                input_values@, position_values@,
                                                cu_q_values@, cu_k_values@,
                                                block_rows@, k,
                                            );
                                        }
                                    }
                                    assert(plan_data_inv(
                                        old(self), self, scheduled_ids@,
                                        input_values@, position_values@,
                                        cu_q_values@, cu_k_values@,
                                        block_rows@,
                                    ));
                                    // Preserve write-page exclusivity.
                                    assert(pre_alloc.blocks@ == loop_pre.blocks@);
                                    // (M1) pre-running tails: hash 0 at entry
                                    // means the matcher never bumped them and
                                    // register never touched them.
                                    assert forall|r: RequestId|
                                        #[trigger] old(self).running@.contains(r)
                                        && old(self).live_requests@.contains_key(r)
                                        implies {
                                            let t = old(self).request_residency@[r]
                                                .block_ids@[old(self)
                                                    .request_residency@[r]
                                                    .block_ids@.len() - 1];
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
                                        let t = old(self).request_residency@[r]
                                            .block_ids@[old(self).request_residency@[r]
                                                .block_ids@.len() - 1];
                                        assert(loop_pre.blocks@.contains_key(t)
                                            && loop_pre.blocks@[t].tokens@
                                                == old(self).blocks@[t].tokens@
                                            && loop_pre.blocks@[t].refcount
                                                == old(self).blocks@[t].refcount
                                            && loop_pre.blocks@[t].prefix_depth
                                                == old(self).blocks@[t].prefix_depth
                                            && loop_pre.blocks@[t].hash_value
                                                == old(self).blocks@[t].hash_value);
                                        assert(old(self).blocks@[t].hash_value == 0);
                                        assert(old(self).blocks@[t].refcount == 1);
                                        assert(pre_alloc.blocks@.contains_key(t));
                                        assert(post_reuse.blocks@.contains_key(t)
                                            && post_reuse.blocks@[t].tokens@
                                                == pre_alloc.blocks@[t].tokens@
                                            && post_reuse.blocks@[t].hash_value
                                                == pre_alloc.blocks@[t].hash_value);
                                        assert(post_reuse.blocks@[t].refcount
                                            == pre_alloc.blocks@[t].refcount) by {
                                            assert(pre_alloc.blocks@[t].hash_value == 0);
                                        }
                                        // t is outside rid's stitched residency.
                                        let rids_new = post_reuse
                                            .request_residency@[rid].block_ids@;
                                        assert(!rids_new.contains(t)) by {
                                            if rids_new.contains(t) {
                                                let q = rids_new.index_of(t);
                                                if q < c_g {
                                                    assert(post_reuse.blocks@[
                                                        rids_new[q]].refcount as int
                                                        == pre_alloc.blocks@[
                                                            rids_new[q]].refcount
                                                                as int + 1);
                                                } else {
                                                    assert(!pre_alloc.blocks@
                                                        .contains_key(rids_new[q]));
                                                }
                                            }
                                        }
                                        assert(self.blocks@[t]
                                            == post_reuse.blocks@[t]);
                                    }
                                    // (M2) admitted rows' suffix pages.
                                    assert forall|k9: int|
                                        decode_count as int <= k9
                                            < scheduled_ids@.len()
                                        implies #[trigger] admitted_pages_exclusive_at(
                                            self, scheduled_ids@, step_hashes@, k9)
                                    by {
                                        let bsz9 = BLOCK_SIZE_SPEC as int;
                                        let srid = scheduled_ids@[k9];
                                        if k9 < scheduled_ids@.len() - 1 {
                                            // Prior admitted row: framed.
                                            assert(srid == sched_pre[k9]);
                                            assert(admitted_pages_exclusive_at(
                                                &loop_pre, sched_pre, sh_iter_pre, k9));
                                            assert(sched_pre.contains(srid));
                                            assert(loop_pre.running@.contains(srid));
                                            assert(srid != rid) by {
                                                assert(!loop_pre.running@.contains(rid));
                                            }
                                            assert(self.request_residency@[srid]
                                                == loop_pre.request_residency@[srid]);
                                            let ids9 = loop_pre
                                                .request_residency@[srid].block_ids@;
                                            let cpb9 = loop_pre
                                                .request_residency@[srid]
                                                .cached_prefix_blocks as int;
                                            assert forall|l: int| cpb9 <= l < ids9.len()
                                                implies self.blocks@.contains_key(
                                                    #[trigger] ids9[l])
                                                    && self.blocks@[ids9[l]].refcount == 1
                                                    && (self.blocks@[ids9[l]].hash_value
                                                            == 0
                                                        || step_hashes@.contains(
                                                            self.blocks@[ids9[l]]
                                                                .hash_value))
                                                    && (!step_hashes@.contains(
                                                            self.blocks@[ids9[l]].hash_value)
                                                        ==> self.blocks@[ids9[l]]
                                                                .prefix_depth == 0
                                                            && self.blocks@[ids9[l]]
                                                                .parent_block
                                                                == Option::<BlockId>::None)
                                            by {
                                                let b9 = ids9[l];
                                                assert(loop_pre.blocks@.contains_key(b9)
                                                    && loop_pre.blocks@[b9].refcount == 1
                                                    && (loop_pre.blocks@[b9].hash_value
                                                            == 0
                                                        || sh_iter_pre.contains(
                                                            loop_pre.blocks@[b9]
                                                                .hash_value))
                                                    && (!sh_iter_pre.contains(
                                                            loop_pre.blocks@[b9].hash_value)
                                                        ==> loop_pre.blocks@[b9]
                                                                .prefix_depth == 0
                                                            && loop_pre.blocks@[b9]
                                                                .parent_block
                                                                == Option::<BlockId>::None));
                                                assert(pre_alloc.blocks@.contains_key(b9));
                                                // No bump: hash 0 or excluded.
                                                assert(post_reuse.blocks@.contains_key(b9)
                                                    && post_reuse.blocks@[b9].tokens@
                                                        == pre_alloc.blocks@[b9].tokens@
                                                    && post_reuse.blocks@[b9].hash_value
                                                        == pre_alloc.blocks@[b9]
                                                            .hash_value);
                                                assert(post_reuse.blocks@[b9].refcount
                                                    == pre_alloc.blocks@[b9].refcount);
                                                // Outside rid's stitched residency.
                                                let rids_new9 = post_reuse
                                                    .request_residency@[rid].block_ids@;
                                                assert(!rids_new9.contains(b9)) by {
                                                    if rids_new9.contains(b9) {
                                                        let q9 = rids_new9.index_of(b9);
                                                        if q9 < c_g {
                                                            assert(post_reuse.blocks@[
                                                                rids_new9[q9]].refcount
                                                                    as int
                                                                == pre_alloc.blocks@[
                                                                    rids_new9[q9]]
                                                                    .refcount as int + 1);
                                                        } else {
                                                            assert(!pre_alloc.blocks@
                                                                .contains_key(
                                                                    rids_new9[q9]));
                                                        }
                                                    }
                                                }
                                                assert(self.blocks@[b9]
                                                    == post_reuse.blocks@[b9]);
                                                // Exclusion membership survives
                                                // the append.
                                                if sh_iter_pre.contains(
                                                    loop_pre.blocks@[b9].hash_value) {
                                                    let w = sh_iter_pre.index_of(
                                                        loop_pre.blocks@[b9].hash_value);
                                                    assert(step_hashes@[w]
                                                        == sh_iter_pre[w]);
                                                }
                                            }
                                        } else {
                                            // The freshly admitted row.
                                            assert(srid == rid);
                                            let ids9 = self
                                                .request_residency@[rid].block_ids@;
                                            assert(self.request_residency@[rid]
                                                .cached_prefix_blocks as int == c_g);
                                            assert(c_g < ids9.len());
                                            assert forall|l: int| c_g <= l < ids9.len()
                                                implies self.blocks@.contains_key(
                                                    #[trigger] ids9[l])
                                                    && self.blocks@[ids9[l]].refcount == 1
                                                    && (self.blocks@[ids9[l]].hash_value
                                                            == 0
                                                        || step_hashes@.contains(
                                                            self.blocks@[ids9[l]]
                                                                .hash_value))
                                                    && (!step_hashes@.contains(
                                                            self.blocks@[ids9[l]].hash_value)
                                                        ==> self.blocks@[ids9[l]]
                                                                .prefix_depth == 0
                                                            && self.blocks@[ids9[l]]
                                                                .parent_block
                                                                == Option::<BlockId>::None)
                                            by {
                                                // Fresh at allocation...
                                                assert(post_reuse.blocks@.contains_key(
                                                    ids9[l])
                                                    && post_reuse.blocks@[ids9[l]]
                                                        .refcount == 1
                                                    && post_reuse.blocks@[ids9[l]]
                                                        .hash_value == 0);
                                                assert(self.blocks@[ids9[l]]
                                                    .hash_value
                                                    == post_reuse.blocks@[ids9[l]]
                                                        .hash_value);
                                                assert(self.blocks@[ids9[l]].refcount
                                                    == post_reuse.blocks@[ids9[l]]
                                                        .refcount);
                                            }
                                            // Partial tail keeps hash 0.
                                            let lt = ids9.len() - 1;
                                            if self.blocks@[ids9[lt]].tokens@.len()
                                                < bsz9 {
                                                assert(self.blocks@[ids9[lt]]
                                                    .tokens@.len()
                                                    == n_g - (ids9.len() - 1) * bsz9);
                                                assert(self.blocks@[ids9[lt]]
                                                    .hash_value
                                                    == post_reuse.blocks@[ids9[lt]]
                                                        .hash_value);
                                            }
                                        }
                                    }
                                }
                            },
                            None => {
                                proof { assert(false); }
                            },
                        }
                        sample_mask.push(final_chunk);
                        proof {
                            lemma_raw_plan_sample_policy_push_admission(
                                old(self), sched_pre, cu_k_vals_iter_pre,
                                sample_mask_iter_pre, rid, row_end as u64,
                                next_k, final_chunk,
                            );
                            // Re-establish the whole-batch shape invariants.
                            assert(cu_q_values@[sched_pre.len() as int] as int
                                == inputs_head as int);
                            assert forall|j2: int| 0 <= j2 < scheduled_ids@.len() as int implies
                                cu_q_values@[j2] < #[trigger] cu_q_values@[j2 + 1] by {
                                if j2 < scheduled_ids@.len() as int - 1 {
                                } else {
                                    assert(cu_q_values@[j2] == cu_last);
                                    assert(cu_q_values@[j2 + 1] == next_cu);
                                    assert(sfx_len as int > 0);
                                }
                            }
                            assert(cu_q_values@[scheduled_ids@.len() as int] as int
                                == input_values@.len() as int);
                            assert forall|j2: int|
                                0 <= j2 < scheduled_ids@.len() as int implies
                                cu_k_values@[j2] < #[trigger] cu_k_values@[j2 + 1]
                            by {
                                if j2 < scheduled_ids@.len() as int - 1 {
                                    assert(cu_k_values@[j2] == cu_k_iter_pre[j2]);
                                    assert(cu_k_values@[j2 + 1]
                                        == cu_k_iter_pre[j2 + 1]);
                                } else {
                                    assert(cu_k_values@[j2] == k_last);
                                    assert(cu_k_values@[j2 + 1] == next_k);
                                    assert(row_end > 0);
                                }
                            }
                            assert forall|j2: int| #![trigger cu_q_values@[j2 + 1]]
                                0 <= j2 < scheduled_ids@.len() as int implies {
                                let q_len = cu_q_values@[j2 + 1] as int
                                    - cu_q_values@[j2] as int;
                                let k_len = cu_k_values@[j2 + 1] as int
                                    - cu_k_values@[j2] as int;
                                &&& q_len <= max_seqlen_q as int
                                &&& k_len <= max_seqlen_k as int
                                &&& q_len <= k_len
                            } by {
                                if j2 < scheduled_ids@.len() as int - 1 {
                                    assert(cu_q_values@[j2] == cu_q_iter_pre[j2]);
                                    assert(cu_q_values@[j2 + 1]
                                        == cu_q_iter_pre[j2 + 1]);
                                    assert(cu_k_values@[j2] == cu_k_iter_pre[j2]);
                                    assert(cu_k_values@[j2 + 1]
                                        == cu_k_iter_pre[j2 + 1]);
                                    assert(max_q_iter_pre <= max_seqlen_q);
                                    assert(max_k_iter_pre <= max_seqlen_k);
                                } else {
                                    assert(cu_q_values@[j2] == cu_last);
                                    assert(cu_q_values@[j2 + 1] == next_cu);
                                    assert(cu_k_values@[j2] == k_last);
                                    assert(cu_k_values@[j2 + 1] == next_k);
                                    assert(sfx_len <= row_end);
                                }
                            }
                            // Per-scheduled commit facts: old prefix + the new
                            // rid (steppable via prefill_plan_ready(old)).
                            assert forall|k: int| 0 <= k < scheduled_ids@.len() implies {
                                let srid = #[trigger] scheduled_ids@[k];
                                self.live_requests@.contains_key(srid) ==> can_step(
                                    self.live_requests@[srid],
                                ) && self.live_requests@[srid].generated_tokens@.len()
                                    < usize::MAX as int
                            } by {
                                if k < scheduled_ids@.len() - 1 {
                                    assert(scheduled_ids@[k] == sched_pre[k]);
                                } else {
                                    assert(scheduled_ids@[k] == rid);
                                }
                            }
                            assert forall|k: int|
                                decode_count as int <= k < scheduled_ids@.len()
                                implies old(self).waiting@.contains(
                                    #[trigger] scheduled_ids@[k])
                            by {
                                if k < scheduled_ids@.len() - 1 {
                                    assert(scheduled_ids@[k] == sched_pre[k]);
                                } else {
                                    assert(scheduled_ids@[k] == rid);
                                    assert(old(self).waiting@.contains(rid));
                                }
                            }
                            assert forall|k: int| 0 <= k < scheduled_ids@.len() implies {
                                let srid = #[trigger] scheduled_ids@[k];
                                old(self).running@.contains(srid)
                                    || old(self).waiting@.contains(srid)
                            } by {
                                if k < scheduled_ids@.len() - 1 {
                                    assert(scheduled_ids@[k] == sched_pre[k]);
                                } else {
                                    assert(scheduled_ids@[k] == rid);
                                    assert(old(self).waiting@.contains(rid));
                                }
                            }
                            // Residency/history alignment survives the
                            // admission: the new rid is aligned by
                            // construction (fresh non-empty tail, refcount
                            // 1); bystanders are framed (tokens preserved,
                            // refcounts move only on full blocks).
                            assert forall|bid: BlockId|
                                #[trigger] loop_pre.blocks@.contains_key(bid)
                                implies self.blocks@.contains_key(bid)
                                    && self.blocks@[bid].tokens@
                                        == loop_pre.blocks@[bid].tokens@
                            by {
                                assert(pre_alloc.blocks@.contains_key(bid));
                                assert(post_reuse.blocks@.contains_key(bid)
                                    && post_reuse.blocks@[bid].tokens@
                                        == pre_alloc.blocks@[bid].tokens@);
                                assert(self.blocks@.contains_key(bid)
                                    && self.blocks@[bid].tokens@
                                        == post_reuse.blocks@[bid].tokens@);
                            }
                            assert forall|r: RequestId|
                                #[trigger] self.running@.contains(r)
                                && self.live_requests@.contains_key(r)
                                implies {
                                    let hist = history(self.live_requests@[r]).len() as int;
                                    let idsr = self.request_residency@[r].block_ids@;
                                    let bsz = BLOCK_SIZE_SPEC as int;
                                    &&& self.request_residency@.contains_key(r)
                                    &&& idsr.len() >= 1
                                    &&& self.blocks@.contains_key(idsr[idsr.len() - 1])
                                    &&& hist == (idsr.len() - 1) * bsz
                                        + self.blocks@[idsr[idsr.len() - 1]].tokens@.len()
                                    &&& self.blocks@[idsr[idsr.len() - 1]].tokens@.len() >= 1
                                    &&& token_placement_prefix(
                                        self.blocks@,
                                        idsr,
                                        history(self.live_requests@[r]),
                                        hist,
                                    )
                                    &&& (self.blocks@[idsr[idsr.len() - 1]].refcount == 1
                                        || self.blocks@[idsr[idsr.len() - 1]].tokens@.len()
                                            == bsz)
                                }
                            by {
                                let bsz = BLOCK_SIZE_SPEC as int;
                                if r == rid {
                                    let idsr = self.request_residency@[rid].block_ids@;
                                    let lastb = idsr[idsr.len() - 1];
                                    // Tail is a FRESH suffix block: suffix
                                    // needs at least one block.
                                    assert(idsr.len() == c_g
                                        + blocks_needed_for((n_g - c_g
                                            * (BLOCK_SIZE_SPEC as int)) as nat) as int);
                                    assert(blocks_needed_for((n_g - c_g
                                        * (BLOCK_SIZE_SPEC as int)) as nat) >= 1);
                                    assert(c_g <= idsr.len() - 1);
                                    assert(self.blocks@.contains_key(lastb)
                                        && self.blocks@[lastb].tokens@.len()
                                            == n_g - (idsr.len() - 1) * bsz
                                        && self.blocks@[lastb].tokens@.len() >= 1
                                        && self.blocks@[lastb].refcount == 1);
                                    // An admitted waiting request is
                                    // unstarted: history == prompt.
                                    assert(waiting_unstarted(&loop_pre));
                                    assert(loop_pre.waiting@[0] == rid);
                                    assert(loop_pre.waiting@.contains(rid));
                                    assert(self.live_requests@[rid].generated_tokens@.len()
                                        == 0);
                                    assert(history(self.live_requests@[rid])
                                        == self.live_requests@[rid].prompt_tokens@);
                                    assert(history(self.live_requests@[rid]).len() as int
                                        == n_g);
                                    assert(self.request_residency@[rid]
                                        == post_reuse.request_residency@[rid]);
                                    assert(token_placement_prefix(
                                        post_reuse.blocks@,
                                        post_reuse.request_residency@[rid].block_ids@,
                                        state.prompt_tokens@,
                                        n_g,
                                    ));
                                    assert forall|bid: BlockId|
                                        #[trigger] post_reuse.blocks@.contains_key(bid)
                                        implies self.blocks@.contains_key(bid)
                                            && self.blocks@[bid].tokens@
                                                == post_reuse.blocks@[bid].tokens@
                                    by {}
                                    lemma_token_placement_prefix_transfer(
                                        post_reuse.blocks@,
                                        self.blocks@,
                                        idsr,
                                        state.prompt_tokens@,
                                        n_g,
                                    );
                                    assert(state.prompt_tokens@
                                        == self.live_requests@[rid].prompt_tokens@);
                                    assert(token_placement_prefix(
                                        self.blocks@,
                                        idsr,
                                        history(self.live_requests@[rid]),
                                        n_g,
                                    ));
                                } else {
                                    assert(pre_push.running@.contains(r));
                                    assert(loop_pre.running@.contains(r));
                                    assert(loop_pre.live_requests@.contains_key(r));
                                    let idsr = loop_pre.request_residency@[r].block_ids@;
                                    let lastb = idsr[idsr.len() - 1];
                                    assert(loop_pre.request_residency@.contains_key(r));
                                    assert(self.request_residency@[r]
                                        == loop_pre.request_residency@[r]);
                                    assert(loop_pre.blocks@.contains_key(lastb));
                                    assert(pre_alloc.blocks@.contains_key(lastb));
                                    assert(self.blocks@.contains_key(lastb)
                                        && self.blocks@[lastb].tokens@
                                            == loop_pre.blocks@[lastb].tokens@);
                                    assert(self.blocks@[lastb].refcount
                                            == loop_pre.blocks@[lastb].refcount
                                        || self.blocks@[lastb].tokens@.len() == bsz);
                                    assert(self.live_requests@[r]
                                        == loop_pre.live_requests@[r]);
                                    assert(residency_history_aligned(&loop_pre));
                                    assert(token_placement_prefix(
                                        loop_pre.blocks@,
                                        idsr,
                                        history(loop_pre.live_requests@[r]),
                                        history(loop_pre.live_requests@[r]).len() as int,
                                    ));
                                    lemma_token_placement_prefix_transfer(
                                        loop_pre.blocks@,
                                        self.blocks@,
                                        idsr,
                                        history(loop_pre.live_requests@[r]),
                                        history(loop_pre.live_requests@[r]).len() as int,
                                    );
                                }
                            }
                            assert(residency_history_aligned(self));
                            // Slot-mapping alignment: the admitted rid's
                            // suffix mapping ends at prompt position n-1;
                            // bystanders are framed.
                            assert forall|r: RequestId|
                                #[trigger] self.running@.contains(r)
                                && self.live_requests@.contains_key(r)
                                implies {
                                    let hist =
                                        history(self.live_requests@[r]).len() as int;
                                    let idsr =
                                        self.request_residency@[r].block_ids@;
                                    let smr =
                                        self.request_residency@[r].slot_mapping@;
                                    &&& self.request_residency@.contains_key(r)
                                    &&& smr.len() >= 1
                                    &&& hist >= 1
                                    &&& smr[smr.len() - 1] as int
                                        == crate::proof::tensor::geometry::block_table_slot(idsr,
                                            (hist - 1) as nat) as int
                                }
                            by {
                                if r == rid {
                                    let idsr =
                                        self.request_residency@[rid].block_ids@;
                                    let smr =
                                        self.request_residency@[rid].slot_mapping@;
                                    let bsz2 = BLOCK_SIZE_SPEC as int;
                                    assert(smr.len() == n_g - c_g * bsz2);
                                    assert(smr.len() >= 1);
                                    assert(history(self.live_requests@[rid]).len()
                                        as int == n_g);
                                    let li = smr.len() - 1;
                                    assert(smr[li] as int
                                        == crate::proof::tensor::geometry::block_table_slot(idsr,
                                            (c_g * bsz2 + li) as nat) as int);
                                    assert(c_g * bsz2 + li == n_g - 1);
                                } else {
                                    assert(pre_push.running@.contains(r));
                                    assert(loop_pre.running@.contains(r));
                                    assert(loop_pre.live_requests@.contains_key(r));
                                    assert(self.request_residency@[r]
                                        == loop_pre.request_residency@[r]);
                                    assert(self.live_requests@[r]
                                        == loop_pre.live_requests@[r]);
                                }
                            }
                            assert(slot_mapping_aligned(self));
                            // Commit-debt bookkeeping: the admitted request
                            // adds its own head; the allocation consumed exactly
                            // `fresh_for_prompt`, the unmatched suffix demand.
                            lemma_debt_snoc(self.live_requests@, sched_pre, rid);
                            assert(self.live_requests@.contains_key(rid));
                            assert(self.live_requests@[rid]
                                .generated_tokens@.len() == 0) by {
                                assert(waiting_unstarted(&loop_pre));
                                assert(loop_pre.waiting@[0] == rid);
                                assert(loop_pre.waiting@.contains(rid));
                            }
                            assert(history(self.live_requests@[rid]).len() as int
                                == n_g);
                            assert(n_g > 0);
                            assert((n_g % (BLOCK_SIZE_SPEC as int) == 0)
                                == (inc == 1));
                            assert(full_tail_debt(self.live_requests@,
                                scheduled_ids@)
                                == full_tail_debt(self.live_requests@, sched_pre)
                                    + inc as int);
                            lemma_blocks_needed_split(n_g as nat, c_g as nat);
                            assert(fresh_for_prompt as int + c_g
                                == blocks_needed_for(n_g as nat) as int);
                            assert(self.request_residency@[rid]
                                .block_ids@.len() as int - c_g
                                == blocks_needed_for((n_g - c_g
                                    * (BLOCK_SIZE_SPEC as int)) as nat) as int);
                            assert(self.request_residency@[rid]
                                .block_ids@.len() as int - c_g
                                == fresh_for_prompt as int);
                            assert(self.free_blocks
                                == pre_alloc.free_blocks - fresh_for_prompt);
                            assert(prior_commit_debt + inc
                                <= pre_alloc.free_blocks - fresh_for_prompt);
                            assert(commit_debt <= self.free_blocks);
                            let ghost admitted_row = block_rows@[block_rows@.len() - 1];
                            assert(admitted_row@
                                == self.request_residency@[rid].block_ids@);
                            assert(block_rows@ == block_rows_iter_pre.push(admitted_row));
                            assert(self.blocks@ == post_reuse.blocks@);
                            lemma_persistent_provenance_closed_blocks_eq(
                                &post_reuse, self,
                            );
                            assert(registered_prefix_chain(
                                self.blocks@, admitted_row@.subrange(0, c_g),
                            )) by {
                                assert(allocate_prefill_reuse_success(
                                    &pre_alloc, &post_reuse, rid,
                                    state.prompt_tokens@, c_g,
                                ));
                            }
                            assert forall|k: int| 0 <= k < sched_pre.len()
                                && !old(self).running@.contains(
                                    #[trigger] sched_pre[k],
                                ) implies {
                                    let old_rid = sched_pre[k];
                                    let old_ids = self.request_residency@[old_rid]
                                        .block_ids@;
                                    let old_c = self.request_residency@[old_rid]
                                        .cached_prefix_blocks as int;
                                    registered_prefix_chain(
                                        self.blocks@,
                                        old_ids.subrange(0, old_c),
                                    )
                                }
                            by {
                                let old_rid = sched_pre[k];
                                assert(old_rid != rid) by {
                                    assert(sched_pre.contains(old_rid));
                                    assert(!sched_pre.contains(rid));
                                }
                                lemma_admitted_current_chain_at(
                                    old(self), &loop_pre, sched_pre,
                                    block_rows_iter_pre, k,
                                );
                                let old_ids = loop_pre.request_residency@[old_rid]
                                    .block_ids@;
                                let old_c = loop_pre.request_residency@[old_rid]
                                    .cached_prefix_blocks as int;
                                assert(loop_pre.request_residency@
                                    .contains_key(old_rid));
                                assert(0 <= old_c <= old_ids.len());
                                assert(pre_alloc.request_residency@
                                    == loop_pre.request_residency@);
                                assert(pre_alloc.request_residency@
                                    .contains_key(old_rid));
                                assert(post_reuse.request_residency@
                                    .contains_key(old_rid));
                                assert(post_reuse.request_residency@[old_rid]
                                    == pre_alloc.request_residency@[old_rid]);
                                assert(pre_push.request_residency@
                                    == post_reuse.request_residency@);
                                assert(self.request_residency@
                                    == pre_push.request_residency@);
                                assert(self.request_residency@[old_rid]
                                    == loop_pre.request_residency@[old_rid]);
                                assert forall|j: int|
                                    0 <= j < old_ids.subrange(0, old_c).len()
                                    implies {
                                        let b = #[trigger]
                                            old_ids.subrange(0, old_c)[j];
                                        &&& self.blocks@.contains_key(b)
                                        &&& self.blocks@[b].prefix_depth
                                            == loop_pre.blocks@[b].prefix_depth
                                        &&& self.blocks@[b].parent_block
                                            == loop_pre.blocks@[b].parent_block
                                }
                                by {
                                    let b = old_ids.subrange(0, old_c)[j];
                                    assert(0 <= j < old_c);
                                    assert(b == old_ids[j]);
                                    assert(loop_pre.request_residency@[old_rid]
                                        .block_ids@.contains(b));
                                    assert(pre_alloc.blocks@ == loop_pre.blocks@);
                                    assert(pre_alloc.blocks@.contains_key(b)) by {
                                        assert(residency_blocks_in_range(&loop_pre));
                                    }
                                    assert(post_reuse.blocks@.contains_key(b));
                                    assert(post_reuse.blocks@[b].prefix_depth
                                        == pre_alloc.blocks@[b].prefix_depth);
                                    assert(post_reuse.blocks@[b].parent_block
                                        == pre_alloc.blocks@[b].parent_block);
                                    assert(self.blocks@ == post_reuse.blocks@);
                                }
                                lemma_registered_prefix_chain_transfer(
                                    loop_pre.blocks@, self.blocks@,
                                    old_ids.subrange(0, old_c),
                                );
                            }
                            lemma_admitted_prefixes_from_pre_admit(
                                old(self), &loop_pre, &post_reuse, self,
                                sched_pre, block_rows_iter_pre, rid,
                                admitted_row, c_g,
                            );
                        }
                        proof {
                            lemma_free_queue_valid_to_token(&post_reuse);
                            lemma_free_queue_valid_token_frame(&post_reuse, self);
                        }
                    } else {
                        self.waiting.insert(0, rid);
                        proof {
                            assert(self.waiting@ == loop_pre.waiting@);
                            assert(self.blocks@ == loop_pre.blocks@);
                            assert(self.request_residency@ == loop_pre.request_residency@);
                            assert(self.hash_to_block@ == loop_pre.hash_to_block@);
                            assert(self.running@ == loop_pre.running@);
                            assert(self.live_requests@ == loop_pre.live_requests@);
                            assert(self.free_blocks == loop_pre.free_blocks);
                            assert(self.num_blocks == loop_pre.num_blocks);
                            lemma_persistent_provenance_closed_blocks_eq(
                                &loop_pre, self,
                            );
                            lemma_residency_running_aligned_frame(&loop_pre, self);
                            assert(cs_valid(&loop_pre));
                            assert(cs_valid(self));
                            lemma_registry_entries_from_pre_allocate(
                                old(self), &loop_pre, self, step_hashes@,
                            );
                            lemma_positive_provenance_origin_blocks_eq(
                                &loop_pre, self,
                            );
                            lemma_positive_provenance_origin_transitive(
                                old(self), &loop_pre, self,
                            );
                            lemma_admitted_current_chains_positive_frame(
                                old(self), &loop_pre, self,
                                scheduled_ids@, block_rows@,
                            );
                            lemma_admitted_prefixes_from_pre_frame(
                                old(self), &loop_pre, self,
                                scheduled_ids@, block_rows@,
                            );
                            // Segment invariant survives: this path leaves
                            // the residency and live maps untouched.
                            assert(plan_seg_inv(old(self), &loop_pre,
                                scheduled_ids@, cu_q_values@, cu_k_values@,
                                slot_values@));
                            lemma_plan_seg_inv_frame(old(self), &loop_pre, self,
                                scheduled_ids@, cu_q_values@, cu_k_values@,
                                slot_values@);
                            assert(plan_data_inv(
                                old(self), &loop_pre, scheduled_ids@,
                                input_values@, position_values@, cu_q_values@,
                                cu_k_values@, block_rows@,
                            ));
                            lemma_plan_data_inv_frame(
                                old(self), &loop_pre, self, scheduled_ids@,
                                input_values@, position_values@, cu_q_values@,
                                cu_k_values@, block_rows@,
                            );
                            // Write-page exclusivity is preserved: blocks are untouched.
                            assert(self.blocks@ == loop_pre.blocks@);
                            assert forall|r: RequestId|
                                #[trigger] old(self).running@.contains(r)
                                && old(self).live_requests@.contains_key(r)
                                implies {
                                    let t = old(self).request_residency@[r]
                                        .block_ids@[old(self).request_residency@[r]
                                            .block_ids@.len() - 1];
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
                                let t = old(self).request_residency@[r]
                                    .block_ids@[old(self).request_residency@[r]
                                        .block_ids@.len() - 1];
                                assert(loop_pre.blocks@.contains_key(t));
                            }
                            assert forall|k: int|
                                decode_count as int <= k < scheduled_ids@.len()
                                implies #[trigger] admitted_pages_exclusive_at(
                                    self, scheduled_ids@, step_hashes@, k)
                            by {
                                assert(admitted_pages_exclusive_at(&loop_pre,
                                    scheduled_ids@, step_hashes@, k));
                                lemma_admitted_pages_frame(&loop_pre, self,
                                    scheduled_ids@, step_hashes@, k);
                            }
                            lemma_registry_entries_from_pre_allocate(
                                old(self), &loop_pre, self, step_hashes@,
                            );
                            lemma_positive_provenance_origin_blocks_eq(
                                &loop_pre, self,
                            );
                            lemma_positive_provenance_origin_transitive(
                                old(self), &loop_pre, self,
                            );
                            lemma_admitted_current_chains_positive_frame(
                                old(self), &loop_pre, self,
                                scheduled_ids@, block_rows@,
                            );
                            lemma_admitted_prefixes_from_pre_frame(
                                old(self), &loop_pre, self,
                                scheduled_ids@, block_rows@,
                            );
                        }
                        proof {
                            lemma_free_queue_valid_to_token(&post_reuse);
                            lemma_free_queue_valid_token_frame(&post_reuse, self);
                        }
                        break;
                    }
                },
                None => {
                    proof {
                        assert(loop_pre.waiting@.contains(rid));
                        assert(live_covers_queue(&loop_pre));
                        assert(loop_pre.live_requests@.contains_key(rid));
                        assert(self.live_requests@ == loop_pre.live_requests@);
                        assert(false);
                    }
                },
            }
        }
        proof {
            lemma_free_queue_token_to_valid(self);
            assert(step_hashes@ =~= Seq::<u64>::empty());
            assert forall|k: int| decode_count as int <= k < scheduled_ids@.len()
                implies #[trigger] admitted_registration_ready_at(
                    self, scheduled_ids@, cu_k_values@, k)
            by {
                let rid = scheduled_ids@[k];
                lemma_admitted_current_chain_at(
                    old(self), self, scheduled_ids@, block_rows@, k,
                );
                assert(old(self).waiting@.contains(rid));
                assert(scheduled_ids@.contains(rid));
                assert(self.running@.contains(rid));
                assert(self.live_requests@.contains_key(rid));
                assert(self.request_residency@.contains_key(rid));
                assert(admitted_pages_exclusive_at(
                    self, scheduled_ids@, Seq::<u64>::empty(), k,
                ));
                assert(plan_data_at(
                    old(self), self, scheduled_ids@, input_values@,
                    position_values@, cu_q_values@, cu_k_values@,
                    block_rows@, k,
                ));
                assert(waiting_unstarted(old(self)));
                assert(prefill_plan_ready(old(self)));
                assert(self.live_requests@ == old(self).live_requests@);
                assert(self.live_requests@[rid].generated_tokens@.len() == 0);
                assert(history(self.live_requests@[rid])
                    == self.live_requests@[rid].prompt_tokens@);
                let ids = self.request_residency@[rid].block_ids@;
                let n = self.live_requests@[rid].prompt_tokens@.len() as int;
                let tail_id = ids[ids.len() - 1];
                let tail = self.blocks@[tail_id].tokens@.len() as int;
                assert(residency_history_aligned(self));
                assert(token_placement_prefix(
                    self.blocks@, ids,
                    self.live_requests@[rid].prompt_tokens@, n,
                ));
                assert(n == (ids.len() - 1) * (BLOCK_SIZE_SPEC as int) + tail);
                assert(1 <= tail <= BLOCK_SIZE_SPEC as int);
                lemma_aligned_blocks_needed(n, ids.len() as int, tail);
                assert(ids.len() == blocks_needed_for(n as nat) as int);
                let c = self.request_residency@[rid]
                    .cached_prefix_blocks as int;
                assert(c * (BLOCK_SIZE_SPEC as int) < n);
                assert forall|h: u64| #[trigger] self.hash_to_block@.contains_key(h)
                    implies !ids.subrange(c, ids.len() as int)
                        .contains(self.hash_to_block@[h])
                by {
                    if ids.subrange(c, ids.len() as int)
                        .contains(self.hash_to_block@[h]) {
                        let q = ids.subrange(c, ids.len() as int)
                            .index_of(self.hash_to_block@[h]);
                        let b = ids[c + q];
                        assert(self.hash_to_block@[h] == b);
                        assert(self.blocks@[b].hash_value == 0);
                        assert(hash_to_block_consistent(self));
                        assert(self.blocks@[self.hash_to_block@[h]].hash_value == h);
                        assert(hash_to_block_no_zero(self));
                        assert(h != 0);
                    }
                }
            }
        }
        let ghost pre_publish = *self;
        proof {
            assert(registry_entries_from_pre(
                old(self), &pre_publish, Seq::<u64>::empty(),
            ));
        }
        self.publish_admitted_prefixes(
            &scheduled_ids, &cu_k_values, decode_count, Ghost(plan_pre),
        );
        proof {
            lemma_plan_seg_inv_frame(
                old(self), &pre_publish, self, scheduled_ids@,
                cu_q_values@, cu_k_values@, slot_values@,
            );
            lemma_plan_data_inv_frame(
                old(self), &pre_publish, self, scheduled_ids@,
                input_values@, position_values@, cu_q_values@,
                cu_k_values@, block_rows@,
            );
        }
        // ---- Materialize one mixed cu-partitioned plan. ----
        let mode = if scheduled_ids.len() == decode_count {
            StepMode::Decode
        } else if decode_count == 0 {
            StepMode::Prefill
        } else {
            StepMode::Mixed
        };
        proof {
            // Write-page exclusivity at plan exit.
            let bsz9 = BLOCK_SIZE_SPEC as int;
            // Every running request's tail is exclusively owned; partial
            // tails keep the hash-0 stamp.
            assert forall|r: RequestId|
                #[trigger] self.running@.contains(r)
                && self.live_requests@.contains_key(r)
                implies {
                    let ids = self.request_residency@[r].block_ids@;
                    &&& self.request_residency@.contains_key(r)
                    &&& ids.len() >= 1
                    &&& self.blocks@.contains_key(ids[ids.len() - 1])
                    &&& self.blocks@[ids[ids.len() - 1]].refcount == 1
                    &&& (self.blocks@[ids[ids.len() - 1]].tokens@.len() < bsz9
                        ==> self.blocks@[ids[ids.len() - 1]].hash_value == 0)
                }
            by {
                if old(self).running@.contains(r) {
                    assert(old(self).live_requests@.contains_key(r));
                    assert(self.request_residency@[r]
                        == old(self).request_residency@[r]);
                    let t = old(self).request_residency@[r].block_ids@[
                        old(self).request_residency@[r].block_ids@.len() - 1];
                    assert(old(self).blocks@[t].refcount == 1);
                    assert(old(self).blocks@[t].hash_value == 0);
                    assert(self.blocks@.contains_key(t)
                        && self.blocks@[t].refcount == old(self).blocks@[t].refcount
                        && self.blocks@[t].prefix_depth
                            == old(self).blocks@[t].prefix_depth
                        && self.blocks@[t].hash_value
                            == old(self).blocks@[t].hash_value);
                } else {
                    assert(scheduled_ids@.contains(r));
                    let k = scheduled_ids@.index_of(r);
                    assert(scheduled_ids@[k] == r);
                    assert(k >= decode_count as int) by {
                        if k < decode_count as int {
                            assert(old(self).running@.contains(scheduled_ids@[k]));
                        }
                    }
                    assert(admitted_pages_exclusive_post(
                        self, scheduled_ids@, k));
                    let ids = self.request_residency@[r].block_ids@;
                    let cpb = self.request_residency@[r].cached_prefix_blocks as int;
                    assert(cpb <= ids.len() - 1);
                    assert(self.blocks@.contains_key(ids[ids.len() - 1])
                        && self.blocks@[ids[ids.len() - 1]].refcount == 1);
                }
            }
            assert(pre_commit_tails_exclusive(self));
            // Admitted rows expose the step-facing pages predicate.
            assert forall|k: int| 0 <= k < scheduled_ids@.len()
                && !old(self).running@.contains(scheduled_ids@[k])
                implies #[trigger] admitted_pages_exclusive_post(self,
                    scheduled_ids@, k)
            by {
                assert(k >= decode_count as int) by {
                    if k < decode_count as int {
                        assert(old(self).running@.contains(scheduled_ids@[k]));
                    }
                }
                assert(admitted_pages_exclusive_post(
                    self, scheduled_ids@, k));
                let srid = scheduled_ids@[k];
                let ids = self.request_residency@[srid].block_ids@;
                let cpb = self.request_residency@[srid].cached_prefix_blocks as int;
                assert forall|l: int| cpb <= l < ids.len()
                    implies self.blocks@.contains_key(#[trigger] ids[l])
                        && self.blocks@[ids[l]].refcount == 1
                by {}
            }
        }
        let ghost sched_final = scheduled_ids@;
        let ghost inputs_final = input_values@;
        let ghost positions_final = position_values@;
        let ghost cu_final = cu_q_values@;
        let ghost cu_k_final = cu_k_values@;
        let ghost slots_final = slot_values@;
        let ghost block_rows_final = block_rows@;
        let ghost sample_mask_final = sample_mask@;
        proof {
            assert(admitted_prefixes_from_pre(
                old(self), &pre_publish, sched_final, block_rows_final,
            ));
            assert forall|k: int| 0 <= k < sched_final.len() implies
                self.running@.contains(#[trigger] sched_final[k])
            by {
                assert(scheduled_ids@.contains(scheduled_ids@[k]));
            }
        }
        let out = materialize_step_plan(
            scheduled_ids,
            sample_mask,
            mode,
            input_values,
            position_values,
            block_rows,
            slot_values,
            cu_q_values,
            cu_k_values,
            max_seqlen_q,
            max_seqlen_k,
            device_anchor,
        );
        proof {
            lemma_plan_sample_policy_from_raw(
                old(self), &out.0, sched_final, cu_k_final,
                sample_mask_final,
            );
            assert forall|k: int| 0 <= k < out.0.scheduled_ids@.len()
                implies out.0.block_table_repr@[k]
                    == #[trigger] self.request_residency@[
                        out.0.scheduled_ids@[k]
                    ].block_ids@
            by {
                assert(out.0.scheduled_ids@ == sched_final);
                assert(out.0.block_table_repr@[k] == block_rows_final[k]@);
                assert(plan_data_at(
                    old(self), self, sched_final, inputs_final,
                    positions_final, cu_final, cu_k_final,
                    block_rows_final, k,
                ));
            }
            assert forall|k: int|
                decode_count as int <= k < out.0.scheduled_ids@.len()
                implies !old(self).running@.contains(
                        #[trigger] out.0.scheduled_ids@[k])
                    && old(self).live_requests@.contains_key(
                        out.0.scheduled_ids@[k])
            by {
                assert(old(self).waiting@.contains(
                    out.0.scheduled_ids@[k]));
                assert(queue_disjoint(old(self)));
                assert(live_covers_queue(old(self)));
            }
            assert(out.0.cu_seqlens_k_repr@.len()
                == out.0.scheduled_ids@.len() + 1);
            assert forall|k: int|
                decode_count as int <= k < out.0.scheduled_ids@.len()
                implies 0 < out.0.cu_seqlens_k_repr@[k + 1]
                        - out.0.cu_seqlens_k_repr@[k]
                    <= self.live_requests@[#[trigger] out.0.scheduled_ids@[k]]
                        .prompt_tokens@.len()
            by {
                assert(out.0.scheduled_ids@ == sched_final);
                assert(out.0.cu_seqlens_k_repr@[k]
                    == cu_k_final[k] as int);
                assert(out.0.cu_seqlens_k_repr@[k + 1]
                    == cu_k_final[k + 1] as int);
                assert(plan_data_at(
                    old(self), self, sched_final, inputs_final,
                    positions_final, cu_final, cu_k_final,
                    block_rows_final, k,
                ));
            }
            lemma_positive_chains_from_publication(
                old(self), &pre_publish, self,
                out.0.scheduled_ids@, out.0.block_table_repr@,
                out.0.cu_seqlens_k_repr@,
                decode_count as int,
            );
            assert forall|k: int| 0 <= k < out.0.scheduled_ids@.len()
                && !old(self).running@.contains(
                    #[trigger] out.0.scheduled_ids@[k])
                implies decode_count as int <= k
            by {
                if k < decode_count as int {
                    assert(old(self).running@.contains(
                        out.0.scheduled_ids@[k]));
                }
            }
            assert forall|k: int|
                decode_count as int <= k < out.0.scheduled_ids@.len()
                implies self.request_residency@.contains_key(
                    #[trigger] out.0.scheduled_ids@[k])
                    && self.live_requests@.contains_key(
                        out.0.scheduled_ids@[k])
                    && {
                        let rid = out.0.scheduled_ids@[k];
                        let ids = self.request_residency@[rid].block_ids@;
                        let prompt = self.live_requests@[rid].prompt_tokens@;
                        let end = out.0.cu_seqlens_k_repr@[k + 1]
                            - out.0.cu_seqlens_k_repr@[k];
                        let full = end / (BLOCK_SIZE_SPEC as int);
                        &&& 0 <= full <= ids.len()
                        &&& 0 <= end <= prompt.len()
                        &&& prompt.len() <= u64::MAX as int
                        &&& blocks_needed_for(end as nat) <= u64::MAX as nat
                        &&& registered_prefix_chain(
                            self.blocks@, ids.subrange(0, full),
                        )
                        &&& token_placement_prefix(
                            self.blocks@, ids, prompt,
                            full * (BLOCK_SIZE_SPEC as int),
                        )
                        &&& forall|l: int| full <= l < ids.len()
                            && #[trigger] self.blocks@.contains_key(ids[l])
                            ==> self.blocks@[ids[l]].prefix_depth == 0
                    }
            by {
                assert(admitted_pages_exclusive_post(
                    self, out.0.scheduled_ids@, k,
                ));
                assert(out.0.scheduled_ids@ == sched_final);
                assert(plan_data_at(
                    old(self), self, sched_final, inputs_final,
                    positions_final, cu_final, cu_k_final,
                    block_rows_final, k,
                ));
                let rid = out.0.scheduled_ids@[k];
                let ids = self.request_residency@[rid].block_ids@;
                let prompt = self.live_requests@[rid].prompt_tokens@;
                let end = out.0.cu_seqlens_k_repr@[k + 1]
                    - out.0.cu_seqlens_k_repr@[k];
                let full = end / (BLOCK_SIZE_SPEC as int);
                assert(admitted_registration_ready_at(
                    &pre_publish, out.0.scheduled_ids@, cu_k_final, k,
                ));
                assert(self.request_residency@ == pre_publish.request_residency@);
                assert(self.live_requests@ == pre_publish.live_requests@);
                assert(self.running@.contains(rid));
                assert(self.live_requests@[rid].generated_tokens@.len() == 0);
                assert(cu_k_final[k] as int
                    == out.0.cu_seqlens_k_repr@[k]);
                assert(cu_k_final[k + 1] as int
                    == out.0.cu_seqlens_k_repr@[k + 1]);
            }
            lemma_published_admission_row_prefixes_from_publication(
                old(self), self, out.0.scheduled_ids@,
                out.0.block_table_repr@, out.0.cu_seqlens_k_repr@,
                decode_count as int,
            );
            assert forall|k: int| 0 <= k < sched_final.len()
                && !old(self).running@.contains(#[trigger] sched_final[k])
                implies {
                    let rid = sched_final[k];
                    let ids = self.request_residency@[rid].block_ids@;
                    let c = self.request_residency@[rid]
                        .cached_prefix_blocks as int;
                    registered_prefix_chain(
                        self.blocks@, ids.subrange(0, c),
                    )
                }
            by {
                assert(k >= decode_count as int) by {
                    if k < decode_count as int {
                        assert(old(self).running@.contains(sched_final[k]));
                    }
                }
            }
            lemma_admitted_prefixes_from_pre_frame(
                old(self), &pre_publish, self,
                sched_final, block_rows_final,
            );
            lemma_plan_cached_prefix_origins_from_raw(
                old(self), self, &out.0, sched_final, inputs_final,
                positions_final, cu_final, cu_k_final, block_rows_final,
            );
            // Bridge the loop-carried value-level segments to the plan reprs
            // (pointwise u64 -> int).
            assert forall|k: int| 0 <= k < out.0.scheduled_ids@.len()
                implies #[trigger] plan_slot_segments_at(old(self), self, &out.0, k)
            by {
                reveal(plan_slot_segments_at);
                assert(plan_seg_at(old(self), self, sched_final, cu_final,
                    cu_k_final, slots_final, k));
                assert(plan_data_at(
                    old(self), self, sched_final, inputs_final,
                    positions_final, cu_final, cu_k_final,
                    block_rows_final, k,
                ));
                assert(out.0.scheduled_ids@ == sched_final);
                let srid = sched_final[k];
                assert(out.0.block_table_repr@[k] == block_rows_final[k]@);
                assert(block_rows_final[k]@
                    == self.request_residency@[srid].block_ids@);
                assert(out.0.cu_seqlens_q_repr@[k] == cu_final[k] as int);
                assert(out.0.cu_seqlens_q_repr@[k + 1] == cu_final[k + 1] as int);
                let s0 = out.0.cu_seqlens_q_repr@[k];
                let s1 = out.0.cu_seqlens_q_repr@[k + 1];
                if !old(self).running@.contains(srid) {
                    let c = self.request_residency@[srid].cached_prefix_blocks as int
                        * (BLOCK_SIZE_SPEC as int);
                    assert forall|q: int| s0 <= q < s1
                        implies #[trigger] out.0.slot_mapping_repr@[q]
                            == crate::proof::tensor::geometry::block_table_slot(
                                self.request_residency@[srid].block_ids@,
                                (c + q - s0) as nat) as int
                    by {
                        assert(out.0.slot_mapping_repr@[q] == slots_final[q] as int);
                        assert(slots_final[q] as int
                            == crate::proof::tensor::geometry::block_table_slot(
                                self.request_residency@[srid].block_ids@,
                                (c + q - cu_final[k] as int) as nat) as int);
                    }
                } else {
                    assert(out.0.slot_mapping_repr@[s0] == slots_final[
                        cu_final[k] as int] as int);
                }
            }
        }
        proof {
            // Bridge the complete raw forward-row payload to the materialized
            // tensor reprs.  Coverage follows from the scheduler's exact
            // residency/history alignment, not from a kernel assumption.
            assert forall|k: int| 0 <= k < out.0.scheduled_ids@.len()
                implies #[trigger] plan_forward_layout_at(old(self), &out.0, k)
            by {
                assert(out.0.scheduled_ids@ == sched_final);
                assert(plan_data_at(
                    old(self), self, sched_final, inputs_final,
                    positions_final, cu_final, cu_k_final,
                    block_rows_final, k,
                ));
                let rid = sched_final[k];
                let s0 = out.0.cu_seqlens_q_repr@[k];
                let s1 = out.0.cu_seqlens_q_repr@[k + 1];
                let kd = out.0.cu_seqlens_k_repr@[k + 1]
                    - out.0.cu_seqlens_k_repr@[k];
                assert(self.running@.contains(rid));
                assert(self.live_requests@.contains_key(rid));
                assert(self.request_residency@.contains_key(rid));
                let ids = self.request_residency@[rid].block_ids@;
                let hist = history(self.live_requests@[rid]).len() as int;
                assert(ids.len() >= 1);
                let tail_id = ids[ids.len() - 1];
                let tail = self.blocks@[tail_id].tokens@.len() as int;
                assert(self.blocks@.contains_key(tail_id));
                assert(hist == (ids.len() - 1) * (BLOCK_SIZE_SPEC as int) + tail);
                assert(1 <= tail <= BLOCK_SIZE_SPEC as int);
                lemma_aligned_blocks_needed(hist, ids.len() as int, tail);
                assert(out.0.block_table_repr@[k] == block_rows_final[k]@);
                assert(block_rows_final[k]@ == ids);
                assert(out.0.input_ids_repr@[s0]
                    == inputs_final[cu_final[k] as int] as int);
                assert(out.0.positions_repr@[s0]
                    == positions_final[cu_final[k] as int] as int);
                assert(old(self).live_requests@.contains_key(rid));
                assert(valid_request_state(old(self).live_requests@[rid]));
                assert(0 <= s0 < s1);
                assert(s1 <= out.0.input_ids_repr@.len() as int);
                if old(self).running@.contains(rid) {
                    assert(self.live_requests@[rid]
                        == old(self).live_requests@[rid]);
                    assert(self.request_residency@[rid]
                        == old(self).request_residency@[rid]);
                    assert(kd == hist);
                    assert(crate::proof::tensor::geometry::blocks_needed_for(kd as nat)
                        == ids.len());
                    let old_h = history(old(self).live_requests@[rid]);
                    assert(out.0.block_table_repr@[k]
                        == old(self).request_residency@[rid].block_ids@);
                    assert(s1 == s0 + 1);
                    assert(kd == old_h.len() as int);
                    assert(out.0.input_ids_repr@[s0]
                        == old_h[old_h.len() - 1] as int);
                    assert(out.0.positions_repr@[s0]
                        == old_h.len() as int - 1);
                    assert(plan_forward_layout_at(old(self), &out.0, k));
                } else {
                    assert(old(self).waiting@.contains(rid));
                    assert(old(self).live_requests@[rid]
                        .generated_tokens@.len() == 0) by {
                        assert(waiting_unstarted(old(self)));
                    }
                    assert(self.live_requests@[rid]
                        == old(self).live_requests@[rid]);
                    let n = old(self).live_requests@[rid]
                        .prompt_tokens@.len() as int;
                    let c0 = self.request_residency@[rid]
                        .cached_prefix_blocks as int * (BLOCK_SIZE_SPEC as int);
                    assert(hist == n);
                    assert(ids.len() == blocks_needed_for(n as nat));
                    assert(0 <= c0 < kd);
                    assert(kd <= n);
                    assert(s1 - s0 == kd - c0);
                    assert(kd - (s1 - s0) == c0);
                    crate::proof::tensor::geometry::lemma_blocks_needed_monotone(
                        kd as nat, n as nat,
                    );
                    assert(crate::proof::tensor::geometry::blocks_needed_for(kd as nat)
                        <= ids.len());
                    assert forall|q: int| s0 <= q < s1 implies {
                        let p = kd - (s1 - s0) + q - s0;
                        &&& #[trigger] out.0.input_ids_repr@[q]
                            == old(self).live_requests@[rid]
                                .prompt_tokens@[p] as int
                        &&& out.0.positions_repr@[q] == p
                    } by {
                        assert(out.0.input_ids_repr@[q]
                            == inputs_final[q] as int);
                        assert(out.0.positions_repr@[q]
                            == positions_final[q] as int);
                    }
                    assert(plan_forward_layout_at(old(self), &out.0, k));
                }
            }
            reveal(plan_residency_extents);
            assert forall|k: int| 0 <= k < out.0.scheduled_ids@.len()
                implies #[trigger] plan_residency_extent_at(
                    old(self), self, &out.0, k,
                )
            by {
                reveal(plan_residency_extent_at);
                assert(out.0.scheduled_ids@ == sched_final);
                assert(plan_data_at(
                    old(self), self, sched_final, inputs_final,
                    positions_final, cu_final, cu_k_final,
                    block_rows_final, k,
                ));
                assert(plan_slot_segments_at(old(self), self, &out.0, k));
                reveal(plan_slot_segments_at);
                let rid = sched_final[k];
                let end = out.0.cu_seqlens_k_repr@[k + 1]
                    - out.0.cu_seqlens_k_repr@[k];
                let ids = self.request_residency@[rid].block_ids@;
                let hist = history(self.live_requests@[rid]);
                assert(self.running@.contains(rid));
                assert(residency_history_aligned(self));
                assert(ids.len() >= 1);
                let tail = ids[ids.len() - 1];
                assert(self.blocks@.contains_key(tail));
                assert(hist.len() as int
                    == (ids.len() - 1) * (BLOCK_SIZE_SPEC as int)
                        + self.blocks@[tail].tokens@.len());
                assert(1 <= self.blocks@[tail].tokens@.len());
                assert(self.blocks@[tail].tokens@.len()
                    <= BLOCK_SIZE_SPEC as int);
                lemma_aligned_blocks_needed(
                    hist.len() as int,
                    ids.len() as int,
                    self.blocks@[tail].tokens@.len() as int,
                );
                assert(ids.len() == blocks_needed_for(hist.len()));
                assert(out.0.block_table_repr@[k] == ids);
                if old(self).running@.contains(rid) {
                    assert(self.live_requests@[rid]
                        == old(self).live_requests@[rid]);
                    assert(end == hist.len() as int);
                    assert(token_placement_prefix(
                        self.blocks@, ids,
                        history(old(self).live_requests@[rid]), end,
                    ));
                } else {
                    assert(old(self).waiting@.contains(rid));
                    assert(old(self).live_requests@[rid]
                        .generated_tokens@.len() == 0) by {
                        assert(waiting_unstarted(old(self)));
                    }
                    assert(self.live_requests@[rid]
                        == old(self).live_requests@[rid]);
                    assert(hist == old(self).live_requests@[rid].prompt_tokens@);
                    assert(0 < end <= hist.len() as int);
                    crate::proof::tensor::geometry::lemma_blocks_needed_monotone(
                        end as nat, hist.len(),
                    );
                    assert(blocks_needed_for(end as nat) <= ids.len());
                    lemma_token_placement_prefix_shrink(
                        self.blocks@,
                        ids,
                        old(self).live_requests@[rid].prompt_tokens@,
                        hist.len() as int,
                        end,
                    );
                }
            }
        }
        proof {
            assert(step_plan_commit_ready(self, &out.0)) by {
                assert forall|i: int|
                    #![trigger out.0.scheduled_ids@[i]]
                    0 <= i < out.0.scheduled_ids@.len()
                    implies !self.waiting@.contains(out.0.scheduled_ids@[i])
                        && (self.live_requests@.contains_key(out.0.scheduled_ids@[i])
                            ==> can_step(self.live_requests@[out.0.scheduled_ids@[i]])
                                && self.live_requests@[out.0.scheduled_ids@[i]]
                                    .generated_tokens@.len() < usize::MAX as int) by {
                    let srid = out.0.scheduled_ids@[i];
                    assert(srid == scheduled_ids@[i]);
                    assert(scheduled_ids@.contains(srid));
                    assert(self.running@.contains(srid));
                    assert(queue_disjoint(self));
                }
            }
        }
        out
    }
}

} // verus!
