// Verified scheduler initialization and request admission transitions.

use super::*;

verus! {
impl CacheScheduler {
    // Initialize an empty scheduler with `num_blocks` available pages.
    // Real exec body — uses HashMapWithView::new() and Vec::new() for the
    // empty starting shape.  cs_valid follows from `lemma_empty_cs_valid`.
    pub fn init(config: SchedulerConfig, num_blocks: u64) -> (out: CacheScheduler)
        requires
            obeys_key_model::<u64>(),
            num_blocks <= u64::MAX / BLOCK_SIZE,
        ensures
            cs_valid(&out),
            free_queue_valid(&out),
            out.config == config,
            out.num_blocks == num_blocks,
            out.free_blocks == num_blocks,
            free_queue_shape(&out.free_queue),
            free_queue_shape(&out.cached_queue),
            free_queue_valid(&out),
            out.free_queue.order@.len() == num_blocks as int,
            out.cached_queue.order@.len() == 0,
            forall|i: int| 0 <= i < num_blocks as int
                ==> #[trigger] out.free_queue.order@[i] == i as BlockId,
            out.running@.len() == 0,
            out.waiting@.len() == 0,
            out.live_requests@.dom().is_empty(),
            out.accepted_requests@.dom().is_empty(),
            out.request_residency@.dom().is_empty(),
            out.blocks@.dom().is_empty(),
            out.hash_to_block@.dom().is_empty(),
            residency_running_aligned(&out),
            persistent_provenance_closed(&out),
    {
        let free_queue = FreeBlockQueue::init_all(num_blocks);
        let cached_queue = FreeBlockQueue::empty();
        let cs = CacheScheduler {
            config,
            num_blocks,
            free_blocks: num_blocks,
            free_queue,
            cached_queue,
            blocks: HashMapWithView::<u64, BlockEntry>::new(),
            running: Vec::<RequestId>::new(),
            waiting: Vec::<RequestId>::new(),
            request_residency: HashMapWithView::<u64, RequestResidency>::new(),
            live_requests: HashMapWithView::<u64, RequestState>::new(),
            accepted_requests: HashMapWithView::<u64, bool>::new(),
            hash_to_block: HashMapWithView::<u64, u64>::new(),
        };
        proof {
            lemma_empty_cs_valid(&cs);
            assert forall|bid: BlockId| cs.free_queue.order@.contains(bid)
                <==> bid < num_blocks by {
                if cs.free_queue.order@.contains(bid) {
                    let i = cs.free_queue.order@.index_of(bid);
                    assert(0 <= i < cs.free_queue.order@.len());
                    assert(cs.free_queue.order@[i] == bid);
                    assert(cs.free_queue.order@[i] == i as BlockId);
                    assert(i < num_blocks as int);
                    assert(bid < num_blocks);
                }
                if bid < num_blocks {
                    assert((bid as int) < num_blocks as int);
                    assert(cs.free_queue.order@[bid as int] == bid);
                    assert(cs.free_queue.order@.contains(bid));
                }
            }
            assert(free_queue_membership_valid(&cs)) by {
                assert forall|bid: BlockId|
                    #[trigger] cs.free_queue.order@.contains(bid)
                    <==> bid < cs.num_blocks
                        && !cs.blocks@.contains_key(bid)
                by {
                    assert(!cs.blocks@.contains_key(bid));
                }
                assert forall|bid: BlockId|
                    #[trigger] cs.cached_queue.order@.contains(bid)
                    <==> zero_ref_cached_page(&cs, bid)
                by {
                }
            }
            assert(free_queue_topological(&cs)) by {
                assert forall|child: BlockId|
                    cs.blocks@.contains_key(child)
                    && cs.blocks@[child].prefix_depth > 0
                    && cs.blocks@[child].parent_block is Some
                    && cs.cached_queue.order@.contains(
                        cs.blocks@[child].parent_block.unwrap(),
                    )
                    implies #[trigger] cs.cached_queue.order@.contains(child)
                        && cs.cached_queue.order@.index_of(child)
                            < cs.cached_queue.order@.index_of(
                                cs.blocks@[child].parent_block.unwrap(),
                            )
                by {
                    assert(false);
                }
            }
            assert(free_queue_partitioned(&cs)) by {
                assert(cs.free_blocks as int == cs.free_queue.order@.len());
                assert(cs.cached_queue.order@.len() == 0);
                assert(cs.free_queue.order@.to_set().disjoint(
                    cs.cached_queue.order@.to_set(),
                ));
            }
            assert(free_queue_valid(&cs));
            reveal(residency_running_aligned);
            assert(residency_running_aligned(&cs));
            reveal(persistent_provenance_closed);
            assert(persistent_provenance_closed(&cs));
        }
        cs
    }

    // Build the initial waiting population by iterating the same verified
    // stable-boundary admission transition used for dynamic arrivals. This
    // keeps accepted-id, live-map, and waiting-queue mutation on one path while
    // exporting the exact batch initialization relation used by refinement.
    pub fn init_with_requests(
        config: SchedulerConfig,
        num_blocks: u64,
        requests: Vec<RequestState>,
    ) -> (out: CacheScheduler)
        requires
            obeys_key_model::<u64>(),
            num_blocks <= u64::MAX / BLOCK_SIZE,
            forall|a: int, b: int|
                #![trigger requests@[a].request_id, requests@[b].request_id]
                0 <= a < b < requests@.len() ==>
                    requests@[a].request_id != requests@[b].request_id,
            forall|k: int| 0 <= k < requests@.len() ==> {
                let request = #[trigger] requests@[k];
                &&& request.generated_tokens@.len() == 0
                &&& can_step(request)
                &&& request_history_capacity_safe(request)
            },
        ensures
            cs_valid(&out),
            free_queue_valid(&out),
            out.config == config,
            out.num_blocks == num_blocks,
            out.free_blocks == num_blocks,
            out.running@.len() == 0,
            out.waiting@.len() == requests@.len(),
            out.request_residency@.dom().is_empty(),
            out.blocks@.dom().is_empty(),
            out.hash_to_block@.dom().is_empty(),
            forall|k: int| 0 <= k < requests@.len() ==>
                #[trigger] out.waiting@[k] == requests@[k].request_id,
            forall|rid: RequestId| #[trigger] out.live_requests@.contains_key(rid) ==>
                exists|k: int| 0 <= k < requests@.len()
                    && requests@[k].request_id == rid,
            forall|k: int| 0 <= k < requests@.len() ==>
                out.live_requests@.contains_key(#[trigger] requests@[k].request_id),
            out.accepted_requests@.dom() == out.live_requests@.dom(),
            forall|rid: RequestId|
                #[trigger] out.accepted_requests@.contains_key(rid)
                ==> out.accepted_requests@[rid],
            forall|k: int| 0 <= k < requests@.len() ==>
                #[trigger] request_state_view_eq(
                    out.live_requests@[requests@[k].request_id], requests@[k],
                ),
            residency_running_aligned(&out),
            persistent_provenance_closed(&out),
            live_request_step_ready(&out),
    {
        let mut cs = CacheScheduler::init(config, num_blocks);
        proof {
            assert(residency_history_aligned(&cs));
            assert(slot_mapping_aligned(&cs));
            assert(tail_write_exclusive(&cs));
            assert(live_request_step_ready(&cs));
        }
        let mut i: usize = 0;
        while i < requests.len()
            invariant
                obeys_key_model::<u64>(),
                i <= requests@.len(),
                forall|a: int, b: int|
                    #![trigger requests@[a].request_id, requests@[b].request_id]
                    0 <= a < b < requests@.len() ==>
                        requests@[a].request_id != requests@[b].request_id,
                forall|k: int| 0 <= k < requests@.len() ==> {
                    let request = #[trigger] requests@[k];
                    &&& request.generated_tokens@.len() == 0
                    &&& can_step(request)
                    &&& request_history_capacity_safe(request)
                },
                cs_valid(&cs),
                residency_history_aligned(&cs),
                slot_mapping_aligned(&cs),
                tail_write_exclusive(&cs),
                residency_running_aligned(&cs),
                persistent_provenance_closed(&cs),
                live_request_step_ready(&cs),
                cs.config == config,
                cs.num_blocks == num_blocks,
                cs.free_blocks == num_blocks,
                free_queue_valid(&cs),
                cs.running@.len() == 0,
                cs.blocks@.dom().is_empty(),
                cs.request_residency@.dom().is_empty(),
                cs.hash_to_block@.dom().is_empty(),
                cs.waiting@.len() == i as int,
                forall|k: int| 0 <= k < i as int ==>
                    #[trigger] cs.waiting@[k] == requests@[k].request_id,
                forall|rid: RequestId| #[trigger] cs.live_requests@.contains_key(rid) ==>
                    exists|k: int| 0 <= k < i as int && requests@[k].request_id == rid,
                forall|k: int| 0 <= k < i as int ==>
                    cs.live_requests@.contains_key(#[trigger] requests@[k].request_id),
                cs.accepted_requests@.dom() == cs.live_requests@.dom(),
                forall|rid: RequestId|
                    #[trigger] cs.accepted_requests@.contains_key(rid)
                    ==> cs.accepted_requests@[rid],
                forall|k: int| 0 <= k < i as int ==>
                    #[trigger] request_state_view_eq(
                        cs.live_requests@[requests@[k].request_id], requests@[k],
                    ),
            decreases requests@.len() - i,
        {
            let request = requests[i].clone();
            let rid = request.request_id;
            proof {
                lemma_request_lifecycle_view_eq_from_fields(
                    request,
                    requests@[i as int],
                );
                assert(request_state_view_eq(
                    request,
                    requests@[i as int],
                ));
                assert(request.generated_tokens@.len() == 0);
                assert(can_step(request));
                assert(request_history_capacity_safe(request));
                assert(!cs.accepted_requests@.contains_key(rid)) by {
                    if cs.accepted_requests@.contains_key(rid) {
                        assert(cs.accepted_requests@.dom().contains(rid));
                        assert(cs.live_requests@.dom().contains(rid));
                        assert(cs.live_requests@.contains_key(rid));
                        let k = choose|k: int| 0 <= k < i as int
                            && requests@[k].request_id == rid;
                        assert(requests@[k].request_id
                            == requests@[i as int].request_id);
                        assert(false);
                    }
                }
            }
            let ghost before = cs;
            cs.add_request(request);
            proof {
                assert(scheduler_admission_relation(&before, &cs, request));
                assert(cs.waiting@ == before.waiting@.push(rid));
                assert(cs.live_requests@
                    == before.live_requests@.insert(rid, request));
                assert(cs.accepted_requests@
                    == before.accepted_requests@.insert(rid, true));

                assert forall|k: int| 0 <= k < i as int + 1 implies
                    #[trigger] cs.waiting@[k] == requests@[k].request_id by {
                    if k < i as int {
                        assert(cs.waiting@[k] == before.waiting@[k]);
                    } else {
                        assert(k == i as int);
                        assert(cs.waiting@[k] == rid);
                    }
                }
                assert forall|r: RequestId|
                    #[trigger] cs.live_requests@.contains_key(r) implies
                    exists|k: int| 0 <= k < i as int + 1
                        && requests@[k].request_id == r by {
                    if r == rid {
                        assert(requests@[i as int].request_id == r);
                    } else {
                        assert(before.live_requests@.contains_key(r));
                    }
                }
                assert forall|k: int| 0 <= k < i as int + 1 implies
                    cs.live_requests@.contains_key(
                        #[trigger] requests@[k].request_id,
                    ) by {
                    if k < i as int {
                        assert(before.live_requests@.contains_key(
                            requests@[k].request_id,
                        ));
                    } else {
                        assert(k == i as int);
                        assert(requests@[k].request_id == rid);
                    }
                }
                assert(cs.accepted_requests@.dom()
                    =~= cs.live_requests@.dom()) by {
                    assert forall|r: RequestId|
                        cs.accepted_requests@.dom().contains(r)
                            <==> cs.live_requests@.dom().contains(r) by {
                        if r != rid {
                            assert(before.accepted_requests@.dom().contains(r)
                                <==> before.live_requests@.dom().contains(r));
                        }
                    }
                }
                assert forall|r: RequestId|
                    #[trigger] cs.accepted_requests@.contains_key(r)
                    implies cs.accepted_requests@[r] by {
                    if r != rid {
                        assert(before.accepted_requests@.contains_key(r));
                    }
                }
                assert forall|k: int| 0 <= k < i as int implies
                    #[trigger] request_state_view_eq(
                        cs.live_requests@[requests@[k].request_id], requests@[k],
                    ) by {
                    assert(requests@[k].request_id != requests@[i as int].request_id);
                }
                assert(request_state_view_eq(
                    cs.live_requests@[requests@[i as int].request_id],
                    requests@[i as int],
                ));
            }
            i = i + 1;
        }
        cs
    }

    // Admit one fresh request at a stable engine boundary. Startup admission
    // iterates this same transition from an empty scheduler; online admission
    // exercises its populated-cache preservation guarantees.
    #[verifier::rlimit(5)]
    pub fn add_request(&mut self, request: RequestState)
        requires
            obeys_key_model::<u64>(),
            cs_valid(old(self)),
            free_queue_valid(old(self)),
            residency_history_aligned(old(self)),
            slot_mapping_aligned(old(self)),
            tail_write_exclusive(old(self)),
            residency_running_aligned(old(self)),
            persistent_provenance_closed(old(self)),
            live_request_step_ready(old(self)),
            !old(self).accepted_requests@.contains_key(request.request_id),
            request.generated_tokens@.len() == 0,
            can_step(request),
            request_history_capacity_safe(request),
        ensures
            scheduler_admission_relation(old(self), final(self), request),
            cs_valid(final(self)),
            free_queue_valid(final(self)),
            residency_history_aligned(final(self)),
            slot_mapping_aligned(final(self)),
            tail_write_exclusive(final(self)),
            residency_running_aligned(final(self)),
            persistent_provenance_closed(final(self)),
            live_request_step_ready(final(self)),
    {
        let rid = request.request_id;
        let ghost before = *self;

        proof {
            assert(!before.live_requests@.contains_key(rid)) by {
                if before.live_requests@.contains_key(rid) {
                    assert(live_requests_accepted(&before));
                    assert(before.accepted_requests@.contains_key(rid));
                }
            }
            assert(!before.running@.contains(rid)) by {
                if before.running@.contains(rid) {
                    assert(live_covers_queue(&before));
                    assert(before.live_requests@.contains_key(rid));
                }
            }
            assert(!before.waiting@.contains(rid)) by {
                if before.waiting@.contains(rid) {
                    assert(live_covers_queue(&before));
                    assert(before.live_requests@.contains_key(rid));
                }
            }
            assert(!before.request_residency@.contains_key(rid)) by {
                if before.request_residency@.contains_key(rid) {
                    assert(residency_has_live_request(&before));
                    assert(before.live_requests@.contains_key(rid));
                }
            }
        }

        self.accepted_requests.insert(rid, true);
        self.live_requests.insert(rid, request);
        proof {
            vstd::map::lemma_map_insert_domain(
                before.live_requests@,
                rid,
                self.live_requests@[rid],
            );
        }
        self.waiting.push(rid);

        proof {
            assert(scheduler_admission_relation(&before, self, request));
            lemma_free_queue_valid_frame(&before, self);

            assert(live_covers_queue(self)) by {
                assert forall|r: RequestId|
                    self.running@.contains(r) || self.waiting@.contains(r)
                    implies #[trigger] self.live_requests@.contains_key(r)
                by {
                    if r == rid {
                        assert(self.live_requests@.contains_key(rid));
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
                        assert(before.live_requests@.contains_key(r));
                    }
                }
            }
            assert(live_requests_accepted(self)) by {
                assert forall|r: RequestId|
                    #[trigger] self.live_requests@.contains_key(r)
                    implies self.accepted_requests@.contains_key(r)
                        && self.accepted_requests@[r]
                by {
                    if r == rid {
                        assert(self.accepted_requests@[rid]);
                    } else {
                        assert(before.live_requests@.contains_key(r));
                        assert(live_requests_accepted(&before));
                    }
                }
            }
            assert(queue_disjoint(self)) by {
                assert forall|r: RequestId|
                    self.running@.contains(r)
                    implies !#[trigger] self.waiting@.contains(r)
                by {
                    assert(before.running@.contains(r));
                    assert(r != rid);
                    assert(!before.waiting@.contains(r));
                }
            }
            assert(running_unique(self));
            assert(waiting_unique(self)) by {
                assert(before.waiting@.no_duplicates());
                reveal(Seq::no_duplicates);
            }
            assert(running_has_residency(self));
            assert(waiting_has_no_residency(self)) by {
                assert forall|r: RequestId|
                    #[trigger] self.waiting@.contains(r)
                    implies !self.request_residency@.contains_key(r)
                by {
                    if r == rid {
                        assert(!before.request_residency@.contains_key(rid));
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
                        assert(self.live_requests@[r] == request);
                    } else {
                        let k = choose|k: int| 0 <= k < self.waiting@.len()
                            && self.waiting@[k] == r;
                        if k < before.waiting@.len() {
                            assert(before.waiting@[k] == r);
                            assert(before.waiting@.contains(r));
                            assert(self.live_requests@[r]
                                == before.live_requests@[r]);
                        } else {
                            assert(r == rid);
                        }
                    }
                }
            }
            assert(residency_has_live_request(self)) by {
                assert forall|r: RequestId|
                    #[trigger] self.request_residency@.contains_key(r)
                    implies self.live_requests@.contains_key(r)
                by {
                    assert(before.request_residency@.contains_key(r));
                    assert(before.live_requests@.contains_key(r));
                    assert(r != rid);
                }
            }
            assert(residency_blocks_in_range(self));
            assert(block_token_bound(self));
            assert(hash_to_block_in_range(self));
            assert(residency_block_ids_unique(self));
            assert(refcount_valid(self));
            assert(hash_to_block_consistent(self));
            assert(registered_provenance_aligned(self));
            assert(blocks_dom_in_range(self));
            assert(block_count_valid(self));
            assert(hash_to_block_no_zero(self));
            assert(cs_valid(self));

            assert(residency_history_aligned(self)) by {
                assert forall|r: RequestId|
                    #[trigger] self.running@.contains(r)
                        && self.live_requests@.contains_key(r)
                    implies {
                        let hist = history(self.live_requests@[r]).len() as int;
                        let ids = self.request_residency@[r].block_ids@;
                        let bs = BLOCK_SIZE_SPEC as int;
                        &&& self.request_residency@.contains_key(r)
                        &&& ids.len() >= 1
                        &&& self.blocks@.contains_key(ids[ids.len() - 1])
                        &&& hist == (ids.len() - 1) * bs
                            + self.blocks@[ids[ids.len() - 1]].tokens@.len()
                        &&& self.blocks@[ids[ids.len() - 1]].tokens@.len() >= 1
                        &&& token_placement_prefix(
                            self.blocks@, ids, history(self.live_requests@[r]), hist,
                        )
                        &&& (self.blocks@[ids[ids.len() - 1]].refcount == 1
                            || self.blocks@[ids[ids.len() - 1]].tokens@.len()
                                == BLOCK_SIZE_SPEC as int)
                    }
                by {
                    assert(before.running@.contains(r));
                    assert(r != rid);
                    assert(self.live_requests@[r] == before.live_requests@[r]);
                }
            }
            assert(slot_mapping_aligned(self)) by {
                assert forall|r: RequestId|
                    #[trigger] self.running@.contains(r)
                        && self.live_requests@.contains_key(r)
                    implies {
                        let hist = history(self.live_requests@[r]).len() as int;
                        let ids = self.request_residency@[r].block_ids@;
                        let sm = self.request_residency@[r].slot_mapping@;
                        &&& self.request_residency@.contains_key(r)
                        &&& sm.len() >= 1
                        &&& hist >= 1
                        &&& sm[sm.len() - 1] as int
                            == crate::proof::tensor::geometry::block_table_slot(
                                ids, (hist - 1) as nat,
                            ) as int
                    }
                by {
                    assert(before.running@.contains(r));
                    assert(r != rid);
                    assert(self.live_requests@[r] == before.live_requests@[r]);
                }
            }
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
                    assert(before.running@.contains(r));
                    assert(r != rid);
                    assert(self.live_requests@[r] == before.live_requests@[r]);
                }
            }
            lemma_residency_running_aligned_frame(&before, self);
            lemma_persistent_provenance_closed_blocks_eq(&before, self);

            assert(live_request_step_ready(self)) by {
                assert forall|r: RequestId|
                    #[trigger] self.live_requests@.contains_key(r)
                    implies can_step(self.live_requests@[r])
                        && request_history_capacity_safe(self.live_requests@[r])
                by {
                    if r == rid {
                        assert(self.live_requests@[r] == request);
                    } else {
                        assert(before.live_requests@.contains_key(r));
                        assert(self.live_requests@[r] == before.live_requests@[r]);
                    }
                }
            }
        }
    }

    // Allocator: prefill path.  Reserve a residency for `rid` covering its
    // prompt-token prefix.  Two-phase body:
    //   Phase 1: pop `blocks_needed` vacant ids from the intrusive queue.
    //            The precheck makes this infallible and removes the old
    //            `0..num_blocks` allocator scan.
    //   Phase 2: if Phase 1 found enough, populate `BlockEntry`s,
    //            build `slot_mapping`, insert `RequestResidency`,
    //            decrement `free_blocks`.
    // This primitive is the fully fresh allocation path.  The planner normally
    // enters through `allocate_prefill_with_reuse`, which delegates here for
    // the uncached suffix (or the whole prompt when no prefix matches).
    //
    // Proof side is now fully local (no method-level `external_body` and no
    // local admits): cs_valid, slot-address facts, token placement, block
    // cardinality, and refcount preservation are discharged.
    #[verifier::spinoff_prover]
    #[verifier::rlimit(200)]
    pub fn allocate_prefill(
        &mut self,
        rid: RequestId,
        prompt_tokens: &Vec<TokenId>,
    ) -> (admitted: bool)
        requires
            cs_valid(old(self)),
            free_queue_valid(old(self)),
            !old(self).request_residency@.contains_key(rid),
            !old(self).running@.contains(rid),
            !old(self).waiting@.contains(rid),
            old(self).live_requests@.contains_key(rid),
            old(self).num_blocks <= u64::MAX / BLOCK_SIZE,
            prompt_tokens@.len() <= u64::MAX as int,
            prompt_tokens@.len() <= usize::MAX as int,
        ensures
            cs_valid(final(self)),
            free_queue_valid(final(self)),
            final(self).config == old(self).config,
            final(self).num_blocks == old(self).num_blocks,
            final(self).live_requests@ == old(self).live_requests@,
            final(self).accepted_requests@ == old(self).accepted_requests@,
            final(self).running@ == old(self).running@,
            final(self).waiting@ == old(self).waiting@,
            admitted ==> final(self).request_residency@.contains_key(rid),
            admitted ==> final(self).request_residency@.dom()
                == old(self).request_residency@.dom().insert(rid),
            admitted ==> prompt_tokens@.len() > 0,
            admitted ==> allocate_prefill_success(old(self), final(self), rid, prompt_tokens@),
            // Prefix caching (2026-08-06): allocation never touches the hash
            // registry, the freshly written blocks carry the prompt
            // (exported so `publish_full_prefix_pages` can be called next), and
            // pre-existing blocks are preserved verbatim (exported so the
            // reuse composition can keep the matched prefix's facts).
            final(self).hash_to_block@ == old(self).hash_to_block@,
            admitted ==> token_placement_prefix(final(self).blocks@,
                final(self).request_residency@[rid].block_ids@, prompt_tokens@,
                prompt_tokens@.len() as int),
            // Exact per-block token counts for residency/history alignment:
            // full blocks hold BLOCK_SIZE tokens; the tail holds the remainder.
            admitted ==> (forall|j: int|
                #![trigger final(self).request_residency@[rid].block_ids@[j]]
                0 <= j < final(self).request_residency@[rid].block_ids@.len()
                ==> final(self).blocks@[final(self).request_residency@[rid].block_ids@[j]].tokens@.len()
                    == (if (j + 1) * (BLOCK_SIZE_SPEC as int) <= prompt_tokens@.len() as int {
                        BLOCK_SIZE_SPEC as int
                    } else {
                        prompt_tokens@.len() as int - j * (BLOCK_SIZE_SPEC as int)
                    })),
            admitted ==> (forall|j: int|
                #![trigger final(self).blocks@[final(self).request_residency@[rid].block_ids@[j]].prefix_depth]
                0 <= j < final(self).request_residency@[rid].block_ids@.len()
                ==> final(self).blocks@[final(self).request_residency@[rid].block_ids@[j]].prefix_depth == 0
                    && final(self).blocks@[final(self).request_residency@[rid].block_ids@[j]].parent_block
                        == Option::<BlockId>::None),
            forall|bid: BlockId| #[trigger] old(self).blocks@.contains_key(bid)
                ==> final(self).blocks@.contains_key(bid)
                    && final(self).blocks@[bid] == old(self).blocks@[bid],
            !admitted ==> final(self).request_residency@ == old(self).request_residency@
                          && final(self).blocks@ == old(self).blocks@
                          && final(self).hash_to_block@ == old(self).hash_to_block@
                          && final(self).free_blocks == old(self).free_blocks
                          && final(self).free_queue.order@ == old(self).free_queue.order@
                          && final(self).cached_queue.head == old(self).cached_queue.head,
            forall|other: RequestId|
                other != rid && #[trigger] old(self).request_residency@.contains_key(other)
                ==> final(self).request_residency@.contains_key(other)
                    && final(self).request_residency@[other] == old(self).request_residency@[other],
            persistent_provenance_closed(old(self))
                ==> persistent_provenance_closed(final(self)),
            positive_provenance_origin(old(self), final(self)),
    {
        let n: u64 = prompt_tokens.len() as u64;
        assert((prompt_tokens@).len() == n as int);
        if n == 0 {
            proof {
                lemma_positive_provenance_origin_blocks_eq(old(self), self);
            }
            return false;
        }
        let blocks_needed: u64 = (n - 1) / BLOCK_SIZE + 1;
        assert(blocks_needed as int == blocks_needed_for(n as nat) as int);
        if blocks_needed > self.free_blocks {
            proof {
                lemma_positive_provenance_origin_blocks_eq(old(self), self);
            }
            return false;
        }
        assert(self.free_blocks <= self.num_blocks) by {
            assert(block_count_valid(self));
            assert(self.blocks@.dom().len() >= 0);
        }

        // Phase 1: reserve the vacant queue prefix.  `free_blocks` is not
        // changed until all corresponding BlockEntry values have been
        // installed, so the stable queue invariant is re-established at the
        // end of phase 2.
        let mut new_block_ids: Vec<BlockId> = Vec::new();
        proof {
            assert(new_block_ids@ =~= old(self).free_queue.order@.subrange(0, 0));
            assert(old(self).free_queue.order@.subrange(
                0, old(self).free_queue.order@.len() as int,
            ) =~= old(self).free_queue.order@);
        }
        while (new_block_ids.len() as u64) < blocks_needed
            invariant
                self.config == old(self).config,
                self.num_blocks == old(self).num_blocks,
                self.free_blocks == old(self).free_blocks,
                free_queue_valid(old(self)),
                self.live_requests@ == old(self).live_requests@,
                self.accepted_requests@ == old(self).accepted_requests@,
                self.running@ == old(self).running@,
                self.waiting@ == old(self).waiting@,
                self.request_residency@ == old(self).request_residency@,
                self.blocks@ == old(self).blocks@,
                self.hash_to_block@ == old(self).hash_to_block@,
                self.cached_queue.head == old(self).cached_queue.head,
                self.cached_queue.tail == old(self).cached_queue.tail,
                self.cached_queue.len == old(self).cached_queue.len,
                self.cached_queue.links@ == old(self).cached_queue.links@,
                self.cached_queue.order@ == old(self).cached_queue.order@,
                free_queue_shape(&self.free_queue),
                blocks_needed <= old(self).free_blocks,
                old(self).free_blocks as int
                    <= old(self).free_queue.order@.len(),
                new_block_ids@.len() <= blocks_needed as int,
                new_block_ids@.len() <= u64::MAX as int,
                new_block_ids@ == old(self).free_queue.order@.subrange(
                    0, new_block_ids@.len() as int,
                ),
                self.free_queue.order@ == old(self).free_queue.order@.subrange(
                    new_block_ids@.len() as int,
                    old(self).free_queue.order@.len() as int,
                ),
                new_block_ids@.no_duplicates(),
                forall|j: int|
                    #![trigger new_block_ids@[j]]
                    0 <= j < new_block_ids@.len()
                    ==> new_block_ids@[j] < self.num_blocks
                        && !self.blocks@.contains_key(new_block_ids@[j]),
            decreases blocks_needed - new_block_ids.len() as u64
        {
            let ghost taken = new_block_ids@.len() as int;
            assert((new_block_ids.len() as u64) as int == taken);
            assert((new_block_ids.len() as u64) < blocks_needed);
            assert(taken < blocks_needed as int);
            assert(taken < old(self).free_blocks as int);
            assert(taken < old(self).free_queue.order@.len());
            assert(self.free_queue.order@.len() > 0);
            let ghost queue_before_pop = self.free_queue.order@;
            let popped = self.free_queue.pop_front();
            let bid = match popped {
                Some(candidate) => candidate,
                None => {
                    proof { assert(false); }
                    return false;
                },
            };
            assert(bid == old(self).free_queue.order@[taken]);
            proof {
                old(self).free_queue.order@.lemma_slice_of_slice(
                    taken,
                    old(self).free_queue.order@.len() as int,
                    1,
                    old(self).free_queue.order@.len() as int - taken,
                );
                assert(self.free_queue.order@ =~=
                    old(self).free_queue.order@.subrange(
                        taken + 1,
                        old(self).free_queue.order@.len() as int,
                    ));
            }
            assert(!old(self).blocks@.contains_key(bid)) by {
                assert(free_queue_partitioned(old(self)));
            }
            proof {
                assert(!new_block_ids@.contains(bid)) by {
                    if new_block_ids@.contains(bid) {
                        let j = new_block_ids@.index_of(bid);
                        assert(old(self).free_queue.order@[j]
                            == old(self).free_queue.order@[taken]);
                        assert(old(self).free_queue.order@.no_duplicates());
                    }
                }
            }
            new_block_ids.push(bid);
            proof {
                assert_seqs_equal!(new_block_ids@
                    == old(self).free_queue.order@.subrange(0, taken + 1), i => {
                    if i == taken {
                        assert(new_block_ids@[i] == bid);
                    } else {
                        assert(0 <= i < taken);
                    }
                });
            }
        }
        assert(new_block_ids@.len() == blocks_needed as int);

        // Phase 2: populate blocks.
        let mut k: usize = 0;
        let mut start: u64 = 0;
        while k < new_block_ids.len()
            invariant
                k <= new_block_ids.len(),
                start <= n,
                (start as int) == (if (k as int) < new_block_ids@.len() {
                    k as int * BLOCK_SIZE_SPEC as int
                } else {
                    n as int
                }),
                (prompt_tokens@).len() == n as int,
                n as int <= usize::MAX as int,
                blocks_needed as int == blocks_needed_for(n as nat) as int,
                self.config == old(self).config,
                self.num_blocks == old(self).num_blocks,
                self.free_blocks == old(self).free_blocks,
                self.cached_queue.head == old(self).cached_queue.head,
                self.cached_queue.tail == old(self).cached_queue.tail,
                self.cached_queue.len == old(self).cached_queue.len,
                self.cached_queue.links@ == old(self).cached_queue.links@,
                self.cached_queue.order@ == old(self).cached_queue.order@,
                free_queue_shape(&self.free_queue),
                self.free_queue.order@ == old(self).free_queue.order@.subrange(
                    blocks_needed as int,
                    old(self).free_queue.order@.len() as int,
                ),
                self.live_requests@ == old(self).live_requests@,
                self.accepted_requests@ == old(self).accepted_requests@,
                self.running@ == old(self).running@,
                self.waiting@ == old(self).waiting@,
                self.request_residency@ == old(self).request_residency@,
                self.hash_to_block@ == old(self).hash_to_block@,
                self.blocks@.dom().len() == old(self).blocks@.dom().len() + k as int,
                forall|bid: BlockId|
                    #[trigger] old(self).blocks@.contains_key(bid)
                    ==> self.blocks@.contains_key(bid)
                        && self.blocks@[bid] == old(self).blocks@[bid],
                forall|bid: BlockId|
                    #[trigger] self.blocks@.contains_key(bid)
                    ==> old(self).blocks@.contains_key(bid)
                        || new_block_ids@.contains(bid),
                old(self).free_blocks <= old(self).num_blocks,
                blocks_needed <= old(self).free_blocks,
                old(self).free_blocks as int
                    <= old(self).free_queue.order@.len(),
                new_block_ids@.len() == blocks_needed as int,
                new_block_ids@ == old(self).free_queue.order@.subrange(
                    0, blocks_needed as int,
                ),
                new_block_ids@.len() <= u64::MAX as int,
                new_block_ids@.no_duplicates(),
                forall|j: int|
                    #![trigger new_block_ids@[j]]
                    0 <= j < new_block_ids@.len()
                    ==> new_block_ids@[j] < self.num_blocks
                        && !old(self).blocks@.contains_key(new_block_ids@[j]),
                forall|j: int|
                    #![trigger self.blocks@.contains_key(new_block_ids@[j])]
                    0 <= j < k as int
                    ==> self.blocks@.contains_key(new_block_ids@[j])
                        && self.blocks@[new_block_ids@[j]].refcount == 1
                        && self.blocks@[new_block_ids@[j]].hash_value == 0
                        && self.blocks@[new_block_ids@[j]].prefix_depth == 0
                        && self.blocks@[new_block_ids@[j]].parent_block
                            == Option::<BlockId>::None
                        && self.blocks@[new_block_ids@[j]].tokens@.len()
                            <= BLOCK_SIZE_SPEC as int
                        && self.blocks@[new_block_ids@[j]].tokens@.len()
                            == (if (j + 1) * (BLOCK_SIZE_SPEC as int) <= n as int {
                                BLOCK_SIZE_SPEC as int
                            } else {
                                n as int - j * (BLOCK_SIZE_SPEC as int)
                            }),
                token_placement_prefix(self.blocks@, new_block_ids@, prompt_tokens@, start as int),
                forall|j: int|
                    #![trigger self.blocks@.contains_key(new_block_ids@[j])]
                    k as int <= j < new_block_ids@.len()
                    ==> !self.blocks@.contains_key(new_block_ids@[j]),
            decreases new_block_ids.len() - k
        {
            let bid = new_block_ids[k];
            assert((k as int) < blocks_needed as int);
            let remaining: u64 = n - start;
            let end_capped: u64 = if remaining < BLOCK_SIZE { n } else { start + BLOCK_SIZE };
            let mut block_tokens: Vec<TokenId> = Vec::new();
            let mut t: u64 = start;
            assert(start <= end_capped);
            assert(end_capped as int - start as int <= BLOCK_SIZE_SPEC as int);
            assert((prompt_tokens@).len() == n as int);
            while t < end_capped
                invariant
                    start <= t <= end_capped,
                    end_capped <= n,
                    end_capped as int - start as int <= BLOCK_SIZE_SPEC as int,
                    (prompt_tokens@).len() == n as int,
                    n as int <= usize::MAX as int,
                    block_tokens@.len() == t as int - start as int,
                    block_tokens@.len() <= BLOCK_SIZE_SPEC as int,
                    forall|o: int|
                        #![trigger block_tokens@[o]]
                        0 <= o < block_tokens@.len()
                        ==> block_tokens@[o] == prompt_tokens@[start as int + o],
                decreases end_capped - t
            {
                assert(t < n);
                assert((t as int) < (prompt_tokens@).len());
                assert(t as int <= usize::MAX as int);
                assert((t as usize) as int == t as int);
                block_tokens.push(prompt_tokens[t as usize]);
                assert forall|o: int|
                    0 <= o < block_tokens@.len()
                    implies block_tokens@[o] == prompt_tokens@[start as int + o]
                by {
                    if o == t as int - start as int {
                        assert(start as int + o == t as int);
                    } else {
                        assert(0 <= o < t as int - start as int);
                    }
                }
                t += 1;
            }
            assert(block_tokens@.len() <= BLOCK_SIZE_SPEC as int);
            assert(!self.blocks@.contains_key(bid));
            let ghost blocks_before_insert = self.blocks@;
            let ghost iter_start = start as int;
            let ghost iter_end = end_capped as int;
            let ghost iter_k = k as int;
            assert(bid == new_block_ids@[k as int]);
            assert(token_placement_prefix(blocks_before_insert, new_block_ids@, prompt_tokens@, iter_start));
            assert(blocks_before_insert.dom().len() == old(self).blocks@.dom().len() + iter_k);
            self.blocks.insert(bid, BlockEntry {
                tokens: block_tokens,
                refcount: 1,
                hash_value: 0,
                prefix_depth: 0,
                parent_block: None,
            });
            assert(self.blocks@.contains_key(bid));
            proof {
                vstd::map::lemma_map_insert_domain(blocks_before_insert, bid, self.blocks@[bid]);
                vstd::set::lemma_set_insert_len(blocks_before_insert.dom(), bid);
            }
            assert(self.blocks@.dom() == blocks_before_insert.dom().insert(bid));
            assert(!blocks_before_insert.dom().contains(bid));
            assert(self.blocks@.dom().len() == blocks_before_insert.dom().len() + 1);
            assert(self.blocks@[bid].refcount == 1);
            assert(self.blocks@[bid].hash_value == 0);
            assert(self.blocks@[bid].prefix_depth == 0);
            assert(self.blocks@[bid].parent_block == Option::<BlockId>::None);
            assert(self.blocks@[bid].tokens@.len() <= BLOCK_SIZE_SPEC as int);
            assert forall|o: int|
                0 <= o < self.blocks@[bid].tokens@.len()
                implies self.blocks@[bid].tokens@[o] == prompt_tokens@[start as int + o]
            by {
            }
            assert forall|x: BlockId|
                #![auto]
                self.blocks@.contains_key(x)
                implies old(self).blocks@.contains_key(x)
                    || new_block_ids@.contains(x)
            by {
                if x == bid {
                    assert(new_block_ids@.contains(bid));
                }
            }
            assert forall|j: int|
                #![auto]
                0 <= j < k as int + 1
                implies self.blocks@.contains_key(new_block_ids@[j])
                    && self.blocks@[new_block_ids@[j]].refcount == 1
                    && self.blocks@[new_block_ids@[j]].hash_value == 0
                    && self.blocks@[new_block_ids@[j]].tokens@.len()
                        <= BLOCK_SIZE_SPEC as int
            by {
                if j == k as int {
                    assert(new_block_ids@[j] == bid);
                } else {
                    assert(0 <= j < k as int);
                    assert(new_block_ids@[j] != bid);
                    assert(self.blocks@.contains_key(new_block_ids@[j]));
                }
            }
            assert forall|j: int|
                #![auto]
                k as int + 1 <= j < new_block_ids@.len()
                implies !self.blocks@.contains_key(new_block_ids@[j])
            by {
                assert((k as int) < j);
                assert(new_block_ids@[j] != bid);
            }
            k += 1;
            start = end_capped;
            assert(self.blocks@.dom().len() == old(self).blocks@.dom().len() + k as int);
            assert(token_placement_prefix(self.blocks@, new_block_ids@, prompt_tokens@, start as int)) by {
                assert forall|p: int|
                    #![trigger token_placement_at(self.blocks@, new_block_ids@, prompt_tokens@, p)]
                    0 <= p < start as int
                    implies token_placement_at(self.blocks@, new_block_ids@, prompt_tokens@, p)
                by {
                    if p < iter_start {
                        assert(token_placement_prefix(blocks_before_insert, new_block_ids@, prompt_tokens@, iter_start));
                        assert(0 <= p < iter_start);
                        assert(iter_start <= prompt_tokens@.len());
                        lemma_token_placement_prefix_at(
                            blocks_before_insert, new_block_ids@, prompt_tokens@, iter_start, p);
                        let j = p / BLOCK_SIZE_SPEC as int;
                        let off = p % BLOCK_SIZE_SPEC as int;
                        assert(j < iter_k);
                        assert(new_block_ids@[j] != bid);
                        assert(j < new_block_ids@.len());
                        assert(blocks_before_insert.contains_key(new_block_ids@[j]));
                        assert(off < blocks_before_insert[new_block_ids@[j]].tokens@.len());
                        assert(blocks_before_insert[new_block_ids@[j]].tokens@[off] == prompt_tokens@[p]);
                        assert(self.blocks@.contains_key(new_block_ids@[j]));
                        assert(self.blocks@[new_block_ids@[j]] == blocks_before_insert[new_block_ids@[j]]);
                        assert(off < self.blocks@[new_block_ids@[j]].tokens@.len());
                        assert(self.blocks@[new_block_ids@[j]].tokens@[off] == prompt_tokens@[p]);
                        assert(token_placement_at(self.blocks@, new_block_ids@, prompt_tokens@, p));
                    } else {
                        let j = p / BLOCK_SIZE_SPEC as int;
                        let off = p % BLOCK_SIZE_SPEC as int;
                        assert(iter_start <= p < iter_end);
                        assert(iter_start == iter_k * BLOCK_SIZE_SPEC as int);
                        assert(p < iter_start + BLOCK_SIZE_SPEC as int);
                        assert(j == iter_k);
                        assert(iter_k < new_block_ids@.len());
                        assert(new_block_ids@[j] == bid);
                        assert(self.blocks@.contains_key(new_block_ids@[j]));
                        assert(off == p - iter_start);
                        assert(0 <= off);
                        assert(off < self.blocks@[bid].tokens@.len());
                        assert(self.blocks@[bid].tokens@[off] == prompt_tokens@[iter_start + off]);
                        assert(iter_start + off == p);
                        assert(token_placement_at(self.blocks@, new_block_ids@, prompt_tokens@, p));
                    }
                }
            }
        }
        assert(k == new_block_ids.len());
        assert forall|j: int|
            #![auto]
            0 <= j < new_block_ids@.len()
            implies !old(self).blocks@.contains_key(new_block_ids@[j])
                && self.blocks@.contains_key(new_block_ids@[j])
                && self.blocks@[new_block_ids@[j]].refcount == 1
                && self.blocks@[new_block_ids@[j]].hash_value == 0
                && self.blocks@[new_block_ids@[j]].tokens@.len() <= BLOCK_SIZE_SPEC as int
        by {
        }
        assert(start == n);
        assert(token_placement_prefix(self.blocks@, new_block_ids@, prompt_tokens@, n as int)) by {
        }
        assert forall|bid: BlockId|
            #![auto]
            self.blocks@.contains_key(bid)
            implies old(self).blocks@.contains_key(bid)
                || new_block_ids@.contains(bid)
        by {
        }
        assert(blocks_needed <= self.free_blocks);
        self.free_blocks = self.free_blocks - blocks_needed;
        assert(token_placement_prefix(self.blocks@, new_block_ids@, prompt_tokens@, n as int)) by {
        }

        // Build slot_mapping: token i lives at block_ids[i / BLOCK_SIZE] * BLOCK_SIZE + (i % BLOCK_SIZE).
        let mut slot_mapping: Vec<SlotId> = Vec::new();
        let mut i: u64 = 0;
        while i < n
            invariant
                i <= n,
                slot_mapping@.len() == i as int,
                forall|a: int|
                    #![trigger slot_mapping@[a]]
                    0 <= a < slot_mapping@.len()
                    ==> slot_mapping@[a] as int
                        == block_table_slot(new_block_ids@, a as nat) as int,
                self.config == old(self).config,
                self.live_requests@ == old(self).live_requests@,
                self.accepted_requests@ == old(self).accepted_requests@,
                self.running@ == old(self).running@,
                self.waiting@ == old(self).waiting@,
                self.hash_to_block@ == old(self).hash_to_block@,
                self.cached_queue.head == old(self).cached_queue.head,
                self.cached_queue.tail == old(self).cached_queue.tail,
                self.cached_queue.len == old(self).cached_queue.len,
                self.cached_queue.links@ == old(self).cached_queue.links@,
                self.cached_queue.order@ == old(self).cached_queue.order@,
                free_queue_shape(&self.free_queue),
                self.free_queue.order@ == old(self).free_queue.order@.subrange(
                    blocks_needed as int,
                    old(self).free_queue.order@.len() as int,
                ),
                self.request_residency@ == old(self).request_residency@,
                self.free_blocks as int == old(self).free_blocks as int - blocks_needed as int,
                self.blocks@.dom().len()
                    == old(self).blocks@.dom().len() + blocks_needed as int,
                self.num_blocks <= u64::MAX / BLOCK_SIZE,
                self.num_blocks == old(self).num_blocks,
                n == (prompt_tokens@).len(),
                n as int <= usize::MAX as int,
                blocks_needed as int == blocks_needed_for(n as nat) as int,
                new_block_ids@.len() == blocks_needed as int,
                forall|j: int|
                    #![trigger new_block_ids@[j]]
                    0 <= j < new_block_ids@.len()
                    ==> new_block_ids@[j] < self.num_blocks,
                forall|bid: BlockId|
                    #[trigger] old(self).blocks@.contains_key(bid)
                    ==> self.blocks@.contains_key(bid)
                        && self.blocks@[bid] == old(self).blocks@[bid],
                forall|bid: BlockId|
                    #[trigger] self.blocks@.contains_key(bid)
                    ==> old(self).blocks@.contains_key(bid)
                        || new_block_ids@.contains(bid),
                forall|j: int|
                    #![trigger self.blocks@.contains_key(new_block_ids@[j])]
                    0 <= j < new_block_ids@.len()
                    ==> !old(self).blocks@.contains_key(new_block_ids@[j])
                        && self.blocks@.contains_key(new_block_ids@[j])
                        && self.blocks@[new_block_ids@[j]].refcount == 1
                        && self.blocks@[new_block_ids@[j]].hash_value == 0
                        && self.blocks@[new_block_ids@[j]].prefix_depth == 0
                        && self.blocks@[new_block_ids@[j]].parent_block is None
                        && self.blocks@[new_block_ids@[j]].tokens@.len()
                            <= BLOCK_SIZE_SPEC as int
                        && self.blocks@[new_block_ids@[j]].tokens@.len()
                            == (if (j + 1) * (BLOCK_SIZE_SPEC as int) <= n as int {
                                BLOCK_SIZE_SPEC as int
                            } else {
                                n as int - j * (BLOCK_SIZE_SPEC as int)
                            }),
                token_placement_prefix(self.blocks@, new_block_ids@, prompt_tokens@, n as int),
            decreases n - i
        {
            proof { lemma_blocks_needed_covers_pos(i as nat, n as nat); }
            assert((i as nat) / BLOCK_SIZE_SPEC < blocks_needed_for(n as nat));
            assert((i as int) / (BLOCK_SIZE_SPEC as int) < blocks_needed as int);
            assert(((i / BLOCK_SIZE) as int) < (new_block_ids@).len());
            assert(i as int <= usize::MAX as int);
            assert(((i as int) / (BLOCK_SIZE_SPEC as int)) <= usize::MAX as int);
            let bidx: usize = (i / BLOCK_SIZE) as usize;
            let bid: BlockId = new_block_ids[bidx];
            assert(bid < self.num_blocks);
            assert(bid <= u64::MAX / BLOCK_SIZE);
            let slot: SlotId = bid * BLOCK_SIZE + (i % BLOCK_SIZE);
            assert(bid == new_block_ids@[bidx as int]);
            assert(bidx as int == (i as int) / (BLOCK_SIZE_SPEC as int));
            assert(slot as int == block_table_slot(new_block_ids@, i as nat) as int);
            slot_mapping.push(slot);
            assert forall|a: int|
                0 <= a < slot_mapping@.len()
                implies slot_mapping@[a] as int
                    == block_table_slot(new_block_ids@, a as nat) as int
            by {
                if a == i as int {
                } else {
                    assert(0 <= a < i as int);
                }
            }
            assert forall|j: int|
                #![auto]
                0 <= j < new_block_ids@.len()
                implies !old(self).blocks@.contains_key(new_block_ids@[j])
                    && self.blocks@.contains_key(new_block_ids@[j])
                    && self.blocks@[new_block_ids@[j]].refcount == 1
                    && self.blocks@[new_block_ids@[j]].hash_value == 0
                    && self.blocks@[new_block_ids@[j]].tokens@.len()
                        <= BLOCK_SIZE_SPEC as int
            by {
            }
            assert(token_placement_prefix(self.blocks@, new_block_ids@, prompt_tokens@, n as int)) by {
            }
            i += 1;
            assert(token_placement_prefix(self.blocks@, new_block_ids@, prompt_tokens@, n as int)) by {
            }
        }

        let ghost block_ids_view = new_block_ids@;
        let ghost slot_mapping_view = slot_mapping@;
        let ghost request_residency_before_insert = self.request_residency@;
        self.request_residency.insert(rid, RequestResidency {
            block_ids: new_block_ids,
            cached_prefix_blocks: 0,
            slot_mapping,
        });
        proof {
            vstd::map::lemma_map_insert_domain(
                request_residency_before_insert, rid, self.request_residency@[rid]);
        }

        assert(self.request_residency@[rid].block_ids@ == block_ids_view);
        assert(self.request_residency@[rid].slot_mapping@ == slot_mapping_view);
        assert(self.request_residency@.contains_key(rid));
        assert(self.request_residency@[rid].cached_prefix_blocks == 0);
        assert(self.request_residency@[rid].block_ids@.len() == blocks_needed as int);
        assert(self.request_residency@[rid].block_ids@.no_duplicates());
        assert(self.request_residency@[rid].slot_mapping@.len() == prompt_tokens@.len());
        assert(self.free_blocks as int == old(self).free_blocks as int - blocks_needed as int);
        assert(self.blocks@.dom().len()
            == old(self).blocks@.dom().len() + blocks_needed as int);
        assert forall|i: int|
            0 <= i < prompt_tokens.len()
            implies self.request_residency@[rid].slot_mapping@[i] as int
                == block_table_slot(self.request_residency@[rid].block_ids@, i as nat) as int
        by {
            assert(self.request_residency@[rid].slot_mapping@[i] == slot_mapping_view[i]);
            assert(self.request_residency@[rid].block_ids@ == block_ids_view);
            assert(block_ids_view == new_block_ids@);
        }
        assert forall|k: int|
            #![auto]
            0 <= k < self.request_residency@[rid].block_ids@.len()
            implies !old(self).blocks@.contains_key(self.request_residency@[rid].block_ids@[k])
                && self.blocks@.contains_key(self.request_residency@[rid].block_ids@[k])
                && self.blocks@[self.request_residency@[rid].block_ids@[k]].refcount == 1
                && self.blocks@[self.request_residency@[rid].block_ids@[k]].hash_value == 0
        by {
            assert(self.request_residency@[rid].block_ids@[k] == block_ids_view[k]);
        }
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
        assert(waiting_has_no_residency(self)) by {
            assert forall|w: RequestId|
                #![auto]
                self.waiting@.contains(w)
                implies !self.request_residency@.contains_key(w)
            by {
                assert(old(self).waiting@.contains(w));
                assert(w != rid);
            }
        }
        assert(residency_blocks_in_range(self));
        assert(block_token_bound(self)) by {
            assert forall|bid: BlockId|
                #![auto]
                self.blocks@.contains_key(bid)
                implies self.blocks@[bid].tokens@.len() <= BLOCK_SIZE_SPEC as int
            by {
                if old(self).blocks@.contains_key(bid) {
                    assert(block_token_bound(old(self)));
                } else {
                    assert(new_block_ids@.contains(bid));
                    let idx = new_block_ids@.index_of(bid);
                    assert(0 <= idx < new_block_ids@.len());
                    assert(new_block_ids@[idx] == bid);
                }
            }
        }
        assert(hash_to_block_in_range(self));
        assert(residency_block_ids_unique(self));
        assert(refcount_valid(self)) by {
            assert forall|bid: BlockId|
                #[trigger] self.blocks@.contains_key(bid)
                implies self.blocks@[bid].refcount as int
                    == residency_holders_of(self, bid).len() as int
            by {
                let holders = residency_holders_of(self, bid);
                if old(self).blocks@.contains_key(bid) {
                    let old_holders = residency_holders_of(old(self), bid);
                    assert(!block_ids_view.contains(bid)) by {
                        if block_ids_view.contains(bid) {
                            let idx = block_ids_view.index_of(bid);
                            assert(block_ids_view[idx] == bid);
                            assert(!old(self).blocks@.contains_key(block_ids_view[idx]));
                        }
                    }
                    assert_sets_equal!(holders == old_holders, r: RequestId => {
                        if holders.contains(r) {
                            assert(self.request_residency@.contains_key(r));
                            if r == rid {
                                assert(self.request_residency@[rid].block_ids@ == block_ids_view);
                                assert(!block_ids_view.contains(bid));
                            } else {
                                assert(request_residency_before_insert == old(self).request_residency@);
                                assert(request_residency_before_insert.contains_key(r));
                                assert(self.request_residency@[r] == request_residency_before_insert[r]);
                            }
                        }
                        if old_holders.contains(r) {
                            assert(old(self).request_residency@.contains_key(r));
                            assert(r != rid);
                            assert(request_residency_before_insert == old(self).request_residency@);
                            assert(self.request_residency@.contains_key(r));
                            assert(self.request_residency@[r] == request_residency_before_insert[r]);
                        }
                    });
                    assert(refcount_valid(old(self)));
                    assert(self.blocks@[bid] == old(self).blocks@[bid]);
                    assert(self.blocks@[bid].refcount as int == old(self).blocks@[bid].refcount as int);
                } else {
                    let singleton = Set::<RequestId>::empty().insert(rid);
                    assert(new_block_ids@.contains(bid));
                    assert_sets_equal!(holders == singleton, r: RequestId => {
                        if holders.contains(r) {
                            assert(self.request_residency@.contains_key(r));
                            if r == rid {
                            } else {
                                assert(request_residency_before_insert == old(self).request_residency@);
                                assert(request_residency_before_insert.contains_key(r));
                                assert(self.request_residency@[r] == request_residency_before_insert[r]);
                                assert(old(self).request_residency@.contains_key(r));
                                assert(old(self).request_residency@[r].block_ids@.contains(bid));
                                assert(residency_blocks_in_range(old(self)));
                                let idx = old(self).request_residency@[r].block_ids@.index_of(bid);
                                assert(old(self).request_residency@[r].block_ids@[idx] == bid);
                                assert(old(self).blocks@.contains_key(bid));
                            }
                        }
                        if singleton.contains(r) {
                            if r == rid {
                                assert(self.request_residency@[rid].block_ids@ == block_ids_view);
                                assert(block_ids_view.contains(bid));
                            } else {
                                vstd::set::lemma_set_empty(r);
                                vstd::set::lemma_set_insert_different(Set::<RequestId>::empty(), r, rid);
                            }
                        }
                    });
                    vstd::set_lib::lemma_set_is_empty_len0(Set::<RequestId>::empty());
                    assert(Set::<RequestId>::empty().is_empty());
                    assert(Set::<RequestId>::empty().len() == 0);
                    vstd::set::lemma_set_insert_len(Set::<RequestId>::empty(), rid);
                    assert(singleton.len() == 1);
                    assert(holders.len() == 1);
                    assert(self.blocks@[bid].refcount == 1);
                }
            }
        }
        assert(hash_to_block_consistent(self));
        assert(registered_provenance_aligned(self)) by {
            assert forall|r: RequestId, j: int|
                #![trigger self.blocks@[self.request_residency@[r].block_ids@[j]].prefix_depth]
                self.request_residency@.contains_key(r)
                && 0 <= j < self.request_residency@[r].block_ids@.len()
                && self.blocks@.contains_key(self.request_residency@[r].block_ids@[j])
                && self.blocks@[self.request_residency@[r].block_ids@[j]].prefix_depth > 0
                implies {
                    let ids = self.request_residency@[r].block_ids@;
                    let bid = ids[j];
                    &&& self.blocks@[bid].prefix_depth as int == j + 1
                    &&& self.blocks@[bid].parent_block
                        == if j == 0 { None } else { Some(ids[j - 1]) }
                }
            by {
                if r == rid {
                    assert(self.request_residency@[rid].block_ids@ == new_block_ids@);
                    assert(self.blocks@[new_block_ids@[j]].prefix_depth == 0);
                } else {
                    assert(old(self).request_residency@.contains_key(r));
                    assert(self.request_residency@[r]
                        == old(self).request_residency@[r]);
                    let bid = self.request_residency@[r].block_ids@[j];
                    assert(old(self).blocks@.contains_key(bid)) by {
                        assert(residency_blocks_in_range(old(self)));
                    }
                    assert(self.blocks@[bid] == old(self).blocks@[bid]);
                    assert(registered_provenance_aligned(old(self)));
                }
            }
        }
        assert(blocks_dom_in_range(self));
        assert(block_count_valid(self)) by {
            assert(block_count_valid(old(self)));
            assert(self.blocks@.dom().len()
                == old(self).blocks@.dom().len() + blocks_needed as int);
            assert(self.free_blocks as int
                == old(self).free_blocks as int - blocks_needed as int);
        }
        proof {
            assert(old(self).free_queue.order@.subrange(
                0, blocks_needed as int,
            ) == new_block_ids@);
            assert forall|bid: BlockId|
                #[trigger] old(self).free_queue.order@.subrange(
                    0, blocks_needed as int,
                ).contains(bid)
                implies !old(self).blocks@.contains_key(bid)
                    && self.blocks@.contains_key(bid)
                    && self.blocks@[bid].refcount > 0
                    && self.blocks@[bid].prefix_depth == 0
            by {
                assert(new_block_ids@.contains(bid));
                let j = new_block_ids@.index_of(bid);
                assert(0 <= j < new_block_ids@.len());
                assert(new_block_ids@[j] == bid);
            }
            assert(free_queue_shape(&self.cached_queue)) by {
                assert(free_queue_valid(old(self)));
            }
            lemma_two_queue_after_vacant_prefix_allocation(
                old(self), self, blocks_needed as int,
            );
        }
        assert(cs_valid(self));
        assert forall|i: int|
            #![trigger prompt_tokens[i]]
            0 <= i < prompt_tokens.len()
            implies {
                let bid = self.request_residency@[rid].block_ids@[i / BLOCK_SIZE_SPEC as int];
                let off = i % BLOCK_SIZE_SPEC as int;
                self.blocks@[bid].tokens@[off] == prompt_tokens[i]
            }
        by {
            assert(i < n as int);
            assert(self.request_residency@[rid].block_ids@ == block_ids_view);
            assert(block_ids_view == new_block_ids@);
        }
        assert(allocate_prefill_success(old(self), self, rid, prompt_tokens@));
        assert(token_placement_prefix(self.blocks@, self.request_residency@[rid].block_ids@,
            prompt_tokens@, prompt_tokens@.len() as int)) by {
            assert(self.request_residency@[rid].block_ids@ == new_block_ids@);
            assert(token_placement_prefix(self.blocks@, new_block_ids@, prompt_tokens@, n as int));
        }
        assert forall|j: int|
            #![trigger self.request_residency@[rid].block_ids@[j]]
            0 <= j < self.request_residency@[rid].block_ids@.len()
            implies self.blocks@[self.request_residency@[rid].block_ids@[j]].tokens@.len()
                == (if (j + 1) * (BLOCK_SIZE_SPEC as int) <= prompt_tokens@.len() as int {
                    BLOCK_SIZE_SPEC as int
                } else {
                    prompt_tokens@.len() as int - j * (BLOCK_SIZE_SPEC as int)
                })
        by {
            assert(self.request_residency@[rid].block_ids@ == new_block_ids@);
            assert(self.blocks@.contains_key(new_block_ids@[j]));
        }
        proof {
          reveal(positive_provenance_origin);
          assert forall|bid: BlockId|
              #[trigger] self.blocks@[bid].prefix_depth > 0
              && self.blocks@.contains_key(bid)
              && self.blocks@[bid].prefix_depth > 0
              implies old(self).blocks@.contains_key(bid)
                  && self.blocks@[bid].tokens@ == old(self).blocks@[bid].tokens@
                  && self.blocks@[bid].hash_value == old(self).blocks@[bid].hash_value
                  && self.blocks@[bid].prefix_depth == old(self).blocks@[bid].prefix_depth
                  && self.blocks@[bid].parent_block == old(self).blocks@[bid].parent_block
          by {
              if !old(self).blocks@.contains_key(bid) {
                  assert(new_block_ids@.contains(bid));
                  let j = new_block_ids@.index_of(bid);
                  assert(self.blocks@[new_block_ids@[j]].prefix_depth == 0);
                  assert(false);
              }
          }
          if persistent_provenance_closed(old(self)) {
            assert(positive_provenance_metadata_frame(old(self), self)) by {
                reveal(positive_provenance_metadata_frame);
                assert forall|bid: BlockId|
                    #[trigger] old(self).blocks@[bid].prefix_depth > 0
                    && old(self).blocks@.contains_key(bid)
                    && old(self).blocks@[bid].prefix_depth > 0
                    implies self.blocks@.contains_key(bid)
                        && self.blocks@[bid] == old(self).blocks@[bid]
                by {
                }
            }
            assert forall|bid: BlockId|
                #[trigger] self.blocks@.contains_key(bid)
                && self.blocks@[bid].prefix_depth > 0
                implies old(self).blocks@.contains_key(bid)
                    && self.blocks@[bid].tokens@
                        == old(self).blocks@[bid].tokens@
                    && self.blocks@[bid].prefix_depth
                        == old(self).blocks@[bid].prefix_depth
                    && self.blocks@[bid].parent_block
                        == old(self).blocks@[bid].parent_block
            by {
                if !old(self).blocks@.contains_key(bid) {
                    assert(new_block_ids@.contains(bid));
                    let j = new_block_ids@.index_of(bid);
                    assert(self.blocks@[new_block_ids@[j]].prefix_depth == 0);
                    assert(false);
                }
            }
            lemma_physical_parent_closure_from_metadata_frame(old(self), self);
            lemma_persistent_provenance_closed_frame(old(self), self);
          }
        }
        true
    }
}

} // verus!
