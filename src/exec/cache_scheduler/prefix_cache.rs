// Verified prefix publication, matching, and reuse transitions.

use super::*;

verus! {
pub proof fn lemma_no_duplicates_distinct(
    s: Seq<BlockId>,
    i: int,
    j: int,
)
    requires
        s.no_duplicates(),
        0 <= i < s.len(),
        0 <= j < s.len(),
        i != j,
    ensures
        s[i] != s[j],
{
    reveal(Seq::no_duplicates);
}

pub proof fn lemma_stitched_block_ids_no_duplicates(
    prefix: Seq<BlockId>,
    fresh: Seq<BlockId>,
    existing: Set<BlockId>,
)
    requires
        prefix.no_duplicates(),
        fresh.no_duplicates(),
        forall|i: int| 0 <= i < prefix.len()
            ==> existing.contains(#[trigger] prefix[i]),
        forall|i: int| 0 <= i < fresh.len()
            ==> !existing.contains(#[trigger] fresh[i]),
    ensures
        (prefix + fresh).no_duplicates(),
{
    assert forall|i: int, j: int|
        0 <= i < prefix.len() && 0 <= j < fresh.len()
        implies prefix[i] != fresh[j]
    by {
        if prefix[i] == fresh[j] {
            assert(existing.contains(prefix[i]));
            assert(!existing.contains(fresh[j]));
        }
    }
    vstd::seq_lib::lemma_no_dup_in_concat(prefix, fresh);
}

pub proof fn lemma_free_queue_token_positive_block_not_cached(
    cs: &CacheScheduler,
    bid: BlockId,
)
    requires
        free_queue_valid_token(cs),
        cs.blocks@.contains_key(bid),
        cs.blocks@[bid].refcount > 0,
    ensures
        !cs.cached_queue.order@.contains(bid),
{
    lemma_free_queue_token_to_valid(cs);
    if cs.cached_queue.order@.contains(bid) {
        assert(free_queue_membership_valid(cs));
    }
}

impl CacheScheduler {
    // Extend an already published physical prefix with newly completed pages.
    // `skip_blocks` is publication progress, independent of initial cache hits.
    // Only full, exclusively owned, provenance-free pages are stamped; first
    // writer wins on a hash collision. Exact token/parent-chain checks, not the
    // hash, justify reuse. The Engine separately proves the published KV values
    // are materialized and valid before another request can consume them.
    #[verifier::spinoff_prover]
    #[verifier::rlimit(120)]
    pub fn publish_full_prefix_pages(
        &mut self,
        rid: RequestId,
        prompt_tokens: &Vec<TokenId>,
        skip_blocks: usize,
    ) -> (registered: Vec<u64>)
        requires
            cs_valid(old(self)),
            free_queue_valid(old(self)),
            old(self).request_residency@.contains_key(rid),
            blocks_needed_for(prompt_tokens@.len() as nat)
                <= old(self).request_residency@[rid].block_ids@.len(),
            token_placement_prefix(old(self).blocks@,
                old(self).request_residency@[rid].block_ids@, prompt_tokens@,
                prompt_tokens@.len() as int),
            skip_blocks as int <= prompt_tokens@.len() as int / (BLOCK_SIZE_SPEC as int),
            registered_prefix_chain(
                old(self).blocks@,
                old(self).request_residency@[rid].block_ids@.subrange(
                    0, skip_blocks as int,
                ),
            ),
            // Blocks past the published prefix are not yet indexed: no
            // existing hash entry targets them.  (The cached prefix's blocks
            // ARE registered — that is how they were found — so they are
            // skipped below.)
            forall|h: u64| #[trigger] old(self).hash_to_block@.contains_key(h)
                ==> !old(self).request_residency@[rid].block_ids@.subrange(
                        skip_blocks as int,
                        old(self).request_residency@[rid].block_ids@.len() as int,
                    ).contains(old(self).hash_to_block@[h]),
            forall|j: int| skip_blocks as int <= j
                && j < prompt_tokens@.len() as int / (BLOCK_SIZE_SPEC as int)
                ==> old(self).blocks@[#[trigger] old(self).request_residency@[rid]
                        .block_ids@[j]].refcount == 1
                    && old(self).blocks@[old(self).request_residency@[rid]
                        .block_ids@[j]].prefix_depth == 0
                    && old(self).blocks@[old(self).request_residency@[rid]
                        .block_ids@[j]].parent_block == Option::<BlockId>::None,
            prompt_tokens@.len() <= usize::MAX as int,
        ensures
            cs_valid(final(self)),
            free_queue_valid(final(self)),
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
                ==> final(self).blocks@.contains_key(bid)
                    && final(self).blocks@[bid].tokens@ == old(self).blocks@[bid].tokens@
                    && final(self).blocks@[bid].refcount == old(self).blocks@[bid].refcount,
            forall|bid: BlockId|
                #[trigger] old(self).blocks@.contains_key(bid)
                && !old(self).request_residency@[rid].block_ids@.contains(bid)
                ==> final(self).blocks@[bid] == old(self).blocks@[bid],
            forall|bid: BlockId|
                #[trigger] old(self).blocks@.contains_key(bid)
                && !old(self).request_residency@[rid].block_ids@.subrange(
                    skip_blocks as int,
                    prompt_tokens@.len() as int / (BLOCK_SIZE_SPEC as int),
                ).contains(bid)
                ==> final(self).blocks@[bid] == old(self).blocks@[bid],
            // The STAMPED hashes are returned verbatim, so the
            // caller's same-step exclusion list is correct by construction.
            registered@.len() as int
                == prompt_tokens@.len() as int / (BLOCK_SIZE_SPEC as int)
                    - skip_blocks as int,
            forall|j: int|
                skip_blocks as int <= j
                    < prompt_tokens@.len() as int / (BLOCK_SIZE_SPEC as int)
                ==> final(self).blocks@[#[trigger] old(self).request_residency@[rid]
                        .block_ids@[j]].hash_value
                    == registered@[j - skip_blocks as int],
            forall|j: int|
                skip_blocks as int <= j
                    < prompt_tokens@.len() as int / (BLOCK_SIZE_SPEC as int)
                ==> final(self).blocks@[#[trigger] old(self).request_residency@[rid]
                        .block_ids@[j]].prefix_depth as int == j + 1
                    && final(self).blocks@[old(self).request_residency@[rid]
                        .block_ids@[j]].parent_block
                        == if j == 0 { None } else {
                            Some(old(self).request_residency@[rid].block_ids@[j - 1])
                        },
            // Unstamped residency blocks (past the full range) keep their
            // hash — in particular a partial suffix tail stays hash 0.
            forall|j: int|
                prompt_tokens@.len() as int / (BLOCK_SIZE_SPEC as int) <= j
                    < old(self).request_residency@[rid].block_ids@.len()
                ==> final(self).blocks@[#[trigger] old(self).request_residency@[rid]
                        .block_ids@[j]].hash_value
                    == old(self).blocks@[old(self).request_residency@[rid]
                        .block_ids@[j]].hash_value,
            forall|j: int|
                prompt_tokens@.len() as int / (BLOCK_SIZE_SPEC as int) <= j
                    < old(self).request_residency@[rid].block_ids@.len()
                    ==> final(self).blocks@[#[trigger] old(self).request_residency@[rid]
                            .block_ids@[j]].prefix_depth
                        == old(self).blocks@[old(self).request_residency@[rid]
                            .block_ids@[j]].prefix_depth
                    && final(self).blocks@[old(self).request_residency@[rid]
                            .block_ids@[j]].parent_block
                        == old(self).blocks@[old(self).request_residency@[rid]
                            .block_ids@[j]].parent_block,
            forall|h: u64| #[trigger] old(self).hash_to_block@.contains_key(h)
                ==> final(self).hash_to_block@.contains_key(h)
                    && final(self).hash_to_block@[h] == old(self).hash_to_block@[h],
            forall|h: u64| #[trigger] final(self).hash_to_block@.contains_key(h)
                ==> old(self).hash_to_block@.contains_key(h)
                    || old(self).request_residency@[rid].block_ids@.subrange(
                        skip_blocks as int,
                        prompt_tokens@.len() as int / (BLOCK_SIZE_SPEC as int),
                    ).contains(final(self).hash_to_block@[h]),
            registration_registry_frame(
                old(self), final(self), rid, prompt_tokens@.len() as int,
                skip_blocks as int, registered@,
            ),
            registration_positive_page_origin(
                old(self), final(self), rid, prompt_tokens@.len() as int,
                skip_blocks as int,
            ),
            positive_provenance_metadata_frame(old(self), final(self)),
            persistent_provenance_closed(old(self))
                ==> persistent_provenance_closed(final(self)),
    {
        hide(free_queue_valid);
        // Consume placement through lemma_token_placement_prefix_at; avoid
        // carrying its token-index quantifier through the publication loop.
        hide(token_placement_prefix);
        proof { lemma_free_queue_valid_to_token(old(self)); }
        let ghost n_spec = prompt_tokens@.len() as int;
        let n: u64 = prompt_tokens.len() as u64;
        let full_blocks: u64 = n / BLOCK_SIZE;
        let block_ids: Vec<BlockId> = match self.request_residency.get(&rid) {
            Some(r) => r.block_ids.clone(),
            None => Vec::new(),
        };
        assert(block_ids@ == old(self).request_residency@[rid].block_ids@);
        // Every full block already holds exactly BLOCK_SIZE tokens (from the
        // exported token placement), and full blocks fit in the residency.
        assert forall|j: int| 0 <= j < full_blocks as int
            implies block_ids@.len() > j
                && self.blocks@.contains_key(#[trigger] block_ids@[j])
                && self.blocks@[block_ids@[j]].tokens@.len() == BLOCK_SIZE_SPEC as int
        by {
            assert(block_ids@[j] == block_ids@[j]);
            let bs = BLOCK_SIZE_SPEC as int;
            assert((j + 1) * bs <= n as int) by (nonlinear_arith)
                requires j < full_blocks as int, full_blocks as int == n as int / 64,
                    bs == 64, n >= 0,
            {}
            let p = j * bs + (bs - 1);
            assert(0 <= p < n as int) by (nonlinear_arith)
                requires p == j * bs + (bs - 1), (j + 1) * bs <= n as int, 0 <= j,
                    bs == 64,
            {}
            vstd::arithmetic::div_mod::lemma_fundamental_div_mod_converse_div(
                p, bs, j, bs - 1);
            assert(p / bs == j);
            crate::proof::tensor::geometry::lemma_blocks_needed_covers_pos(p as nat, n as nat);
            assert(j < blocks_needed_for(n as nat) as int);
            assert(block_ids@.len() > j);
            lemma_token_placement_prefix_at(self.blocks@, block_ids@, prompt_tokens@,
                n_spec, p);
            assert(token_placement_at(self.blocks@, block_ids@, prompt_tokens@, p));
            assert(p % bs == bs - 1) by (nonlinear_arith)
                requires p == j * bs + (bs - 1), 0 <= j, bs == 64,
            {}
            assert(self.blocks@.contains_key(block_ids@[j]));
            assert(self.blocks@[block_ids@[j]].tokens@.len() > bs - 1);
            assert(block_token_bound(self));
            assert(self.blocks@[block_ids@[j]].tokens@.len() <= bs);
        }
        let ghost ids = block_ids@;
        assert(full_blocks as int <= ids.len()) by {
            if full_blocks as int > 0 {
                let jl = full_blocks as int - 1;
                assert(block_ids@[jl] == block_ids@[jl]);
                assert(block_ids@.len() > jl);
            }
        }
        assert(full_blocks_sized(self, ids, full_blocks as int));
        // Continue from the stored parent hash instead of rehashing the entire
        // prefix on each decode page boundary. Hash collisions may cause misses
        // but cannot bypass the exact physical provenance check during reuse.
        let mut ph: u64 = if skip_blocks == 0 {
            0
        } else {
            let parent = block_ids[skip_blocks - 1];
            match self.blocks.get(&parent) {
                Some(entry) => entry.hash_value,
                None => { proof { assert(false); } 0 },
            }
        };
        let mut registered: Vec<u64> = Vec::new();
        let mut k: usize = skip_blocks;
        while (k as u64) < full_blocks
            invariant
                registered@.len() as int == k as int - skip_blocks as int,
                forall|j: int| skip_blocks as int <= j < k as int
                    ==> self.blocks@[#[trigger] ids[j]].hash_value
                        == registered@[j - skip_blocks as int],
                forall|j: int| skip_blocks as int <= j < k as int
                    ==> self.blocks@[#[trigger] ids[j]].prefix_depth as int == j + 1
                        && self.blocks@[ids[j]].parent_block
                            == if j == 0 { None } else { Some(ids[j - 1]) },
                forall|j: int| k as int <= j < ids.len()
                    ==> self.blocks@[#[trigger] ids[j]].hash_value
                        == old(self).blocks@[ids[j]].hash_value,
                forall|j: int| k as int <= j < ids.len()
                    ==> self.blocks@[#[trigger] ids[j]].prefix_depth
                            == old(self).blocks@[ids[j]].prefix_depth
                        && self.blocks@[ids[j]].parent_block
                            == old(self).blocks@[ids[j]].parent_block,
                skip_blocks as int <= k as int,
                cs_valid(self),
                free_queue_valid_token(self),
                n as int == prompt_tokens@.len(),
                n as int <= usize::MAX as int,
                full_blocks == n / BLOCK_SIZE,
                k as int <= full_blocks as int,
                ids == old(self).request_residency@[rid].block_ids@,
                ids == block_ids@,
                ids.no_duplicates(),
                self.config == old(self).config,
                self.num_blocks == old(self).num_blocks,
                self.free_blocks == old(self).free_blocks,
                self.running@ == old(self).running@,
                self.waiting@ == old(self).waiting@,
                self.live_requests@ == old(self).live_requests@,
                self.accepted_requests@ == old(self).accepted_requests@,
                self.request_residency@ == old(self).request_residency@,
                self.request_residency@.contains_key(rid),
                self.blocks@.dom() == old(self).blocks@.dom(),
                forall|bid: BlockId| #[trigger] old(self).blocks@.contains_key(bid)
                    ==> self.blocks@.contains_key(bid)
                        && self.blocks@[bid].tokens@ == old(self).blocks@[bid].tokens@
                        && self.blocks@[bid].refcount == old(self).blocks@[bid].refcount,
                forall|bid: BlockId|
                    #[trigger] old(self).blocks@.contains_key(bid) && !ids.contains(bid)
                    ==> self.blocks@[bid] == old(self).blocks@[bid],
                forall|bid: BlockId|
                    #[trigger] old(self).blocks@.contains_key(bid)
                    && !ids.subrange(skip_blocks as int, k as int).contains(bid)
                    ==> self.blocks@[bid] == old(self).blocks@[bid],
                forall|h: u64| #[trigger] old(self).hash_to_block@.contains_key(h)
                    ==> self.hash_to_block@.contains_key(h)
                        && self.hash_to_block@[h] == old(self).hash_to_block@[h],
                forall|h: u64| #[trigger] old(self).hash_to_block@.contains_key(h)
                    ==> old(self).blocks@.contains_key(old(self).hash_to_block@[h])
                        && self.blocks@.contains_key(old(self).hash_to_block@[h])
                        && self.blocks@[old(self).hash_to_block@[h]].tokens@
                            == old(self).blocks@[old(self).hash_to_block@[h]].tokens@
                        && self.blocks@[old(self).hash_to_block@[h]].hash_value
                            == old(self).blocks@[old(self).hash_to_block@[h]].hash_value
                        && self.blocks@[old(self).hash_to_block@[h]].prefix_depth
                            == old(self).blocks@[old(self).hash_to_block@[h]].prefix_depth
                        && self.blocks@[old(self).hash_to_block@[h]].parent_block
                            == old(self).blocks@[old(self).hash_to_block@[h]].parent_block,
                forall|h: u64| #[trigger] old(self).hash_to_block@.contains_key(h)
                    ==> !ids.subrange(skip_blocks as int, ids.len() as int)
                        .contains(old(self).hash_to_block@[h]),
                forall|j: int| skip_blocks as int <= j
                    && j < prompt_tokens@.len() as int / (BLOCK_SIZE_SPEC as int)
                    ==> old(self).blocks@[#[trigger] old(self).request_residency@[rid]
                            .block_ids@[j]].refcount == 1
                        && old(self).blocks@[old(self).request_residency@[rid]
                            .block_ids@[j]].prefix_depth == 0
                        && old(self).blocks@[old(self).request_residency@[rid]
                            .block_ids@[j]].parent_block == Option::<BlockId>::None,
                forall|h: u64| #[trigger] self.hash_to_block@.contains_key(h)
                    ==> old(self).hash_to_block@.contains_key(h)
                        || ids.subrange(skip_blocks as int, k as int)
                            .contains(self.hash_to_block@[h]),
                full_blocks_sized(self, ids, full_blocks as int),
            decreases full_blocks as int - k as int,
        {
            proof {
                assert((k as int) < full_blocks as int);
                assert(self.blocks@.contains_key(ids[k as int]));
                assert(ids.len() > k as int);
                // start/end arithmetic stays in usize: (k+1)*64 <= n.
                assert((k as int + 1) * (BLOCK_SIZE_SPEC as int) <= n as int
                        && (k as int) * (BLOCK_SIZE_SPEC as int) + (BLOCK_SIZE_SPEC as int)
                            <= n as int)
                    by (nonlinear_arith)
                    requires (k as int) < full_blocks as int,
                        full_blocks as int == n as int / 64,
                        BLOCK_SIZE_SPEC as int == 64, n >= 0,
                {}
            }
            let bid = block_ids[k];
            let start: usize = k * (BLOCK_SIZE as usize);
            let end: usize = start + (BLOCK_SIZE as usize);
            ph = chain_hash_span(ph, prompt_tokens, start, end);
            let ghost pre_iter = *self;
            // Stamp the chain hash on the (full) block.
            let (toks, rc) = match self.blocks.get(&bid) {
                Some(e) => (e.tokens.clone(), e.refcount),
                None => {
                    proof { assert(false); }
                    (Vec::new(), 0)
                },
            };
            proof {
                assert(toks@ == pre_iter.blocks@[bid].tokens@);
                assert(rc == pre_iter.blocks@[bid].refcount);
            }
            let parent_block = if k == 0 { None } else { Some(block_ids[k - 1]) };
            self.blocks.insert(bid, BlockEntry {
                tokens: toks,
                refcount: rc,
                hash_value: ph,
                prefix_depth: k as u64 + 1,
                parent_block,
            });
            registered.push(ph);
            proof {
                assert(self.blocks@[bid].tokens@ == pre_iter.blocks@[bid].tokens@);
                vstd::map::lemma_map_insert_domain(pre_iter.blocks@, bid, self.blocks@[bid]);
                assert(self.blocks@.dom() =~= pre_iter.blocks@.dom());
                assert(self.blocks@[bid].refcount > 0) by {
                    assert(pre_iter.blocks@[bid].refcount == 1);
                }
                assert(self.blocks@[bid].parent_block is Some ==>
                    self.blocks@.contains_key(
                        self.blocks@[bid].parent_block.unwrap(),
                    )
                    && self.blocks@[
                        self.blocks@[bid].parent_block.unwrap()
                    ].refcount > 0
                ) by {
                    if self.blocks@[bid].parent_block is Some {
                        assert(k > 0);
                        let parent = block_ids[k - 1];
                        assert(self.blocks@[bid].parent_block == Some(parent));
                        assert(pre_iter.request_residency@.contains_key(rid));
                        assert(pre_iter.request_residency@[rid].block_ids@ == ids);
                        assert(ids[k as int - 1] == parent);
                        assert(pre_iter.blocks@.contains_key(parent));
                        let holders = residency_holders_of(&pre_iter, parent);
                        assert(holders.contains(rid));
                        assert(refcount_valid(&pre_iter));
                        assert(pre_iter.blocks@[parent].refcount as int
                            == holders.len() as int);
                        assert(holders.len() > 0) by {
                            if holders.len() == 0 {
                                vstd::set_lib::lemma_set_is_empty_len0(holders);
                            }
                        }
                        assert(self.blocks@[parent]
                            == pre_iter.blocks@[parent]) by {
                            assert(parent != bid) by {
                                assert(ids.no_duplicates());
                            }
                        }
                    }
                }
                assert(self.blocks@[bid].prefix_depth > 0
                    && self.blocks@[bid].parent_block is Some ==>
                    !self.cached_queue.order@.contains(
                        self.blocks@[bid].parent_block.unwrap(),
                    )) by {
                    if self.blocks@[bid].parent_block is Some {
                        let parent = self.blocks@[bid].parent_block.unwrap();
                        assert(k > 0);
                        assert(parent == block_ids@[k as int - 1]);
                        lemma_no_duplicates_distinct(ids, k as int - 1, k as int);
                        assert(parent != bid);
                        assert(self.blocks@[parent] == pre_iter.blocks@[parent]);
                        assert(pre_iter.blocks@[parent].refcount > 0);
                        lemma_free_queue_token_positive_block_not_cached(
                            &pre_iter, parent,
                        );
                        assert(self.cached_queue.order@
                            == pre_iter.cached_queue.order@);
                    }
                }
                assert forall|other: BlockId| other != bid
                    && #[trigger] pre_iter.blocks@.contains_key(other)
                    implies self.blocks@[other] == pre_iter.blocks@[other]
                by {
                }
                lemma_two_queue_valid_token_publish_active_page(
                    &pre_iter, self, bid,
                );
                // No existing hash entry targets `bid` (old entries by the
                // freshness requires; new entries target earlier prefix ids,
                // all distinct from ids[k]).
                assert forall|h: u64| #[trigger] self.hash_to_block@.contains_key(h)
                    implies self.hash_to_block@[h] != bid
                by {
                    if old(self).hash_to_block@.contains_key(h) {
                        assert(!ids.subrange(skip_blocks as int, ids.len() as int)
                            .contains(old(self).hash_to_block@[h]));
                        assert(ids.subrange(skip_blocks as int, ids.len() as int)
                            [k as int - skip_blocks as int] == bid);
                        assert(ids.subrange(skip_blocks as int, ids.len() as int)
                            .contains(bid));
                    } else {
                        assert(ids.subrange(skip_blocks as int, k as int)
                            .contains(self.hash_to_block@[h]));
                        let j = choose|j: int|
                            0 <= j < k as int - skip_blocks as int
                            && ids.subrange(skip_blocks as int, k as int)[j]
                                == self.hash_to_block@[h];
                        assert(ids[skip_blocks as int + j] == self.hash_to_block@[h]);
                        assert(ids[skip_blocks as int + j] != ids[k as int]);
                    }
                }
                assert(hash_to_block_in_range(self)) by {
                    assert forall|h: u64| #[trigger] self.hash_to_block@.contains_key(h)
                        implies self.blocks@.contains_key(self.hash_to_block@[h])
                            && self.blocks@[self.hash_to_block@[h]].tokens@.len()
                                == BLOCK_SIZE_SPEC as int
                    by {
                        assert(pre_iter.hash_to_block@.contains_key(h));
                        assert(self.hash_to_block@[h] != bid);
                        assert(self.blocks@[self.hash_to_block@[h]]
                            == pre_iter.blocks@[self.hash_to_block@[h]]);
                        assert(hash_to_block_in_range(&pre_iter));
                    }
                }
                assert(hash_to_block_consistent(self)) by {
                    assert forall|h: u64| #[trigger] self.hash_to_block@.contains_key(h)
                        implies self.blocks@[self.hash_to_block@[h]].hash_value == h
                            && self.blocks@[self.hash_to_block@[h]].prefix_depth > 0
                    by {
                        assert(self.hash_to_block@[h] != bid);
                        assert(self.blocks@[self.hash_to_block@[h]]
                            == pre_iter.blocks@[self.hash_to_block@[h]]);
                        assert(hash_to_block_consistent(&pre_iter));
                    }
                }
                lemma_refcount_valid_transfer(&pre_iter, self);
                assert(block_token_bound(self)) by {
                    assert forall|b2: BlockId| #[trigger] self.blocks@.contains_key(b2)
                        implies self.blocks@[b2].tokens@.len() <= BLOCK_SIZE_SPEC as int
                    by {
                        if b2 != bid {
                            assert(pre_iter.blocks@.contains_key(b2));
                            assert(self.blocks@[b2] == pre_iter.blocks@[b2]);
                        } else {
                            assert(self.blocks@[bid].tokens@
                                == pre_iter.blocks@[bid].tokens@);
                        }
                    }
                }
                assert(residency_blocks_in_range(self));
                assert(registered_provenance_aligned(self)) by {
                    assert forall|r: RequestId, q: int|
                        #![trigger self.blocks@[self.request_residency@[r]
                            .block_ids@[q]].prefix_depth]
                        self.request_residency@.contains_key(r)
                        && 0 <= q < self.request_residency@[r].block_ids@.len()
                        && self.blocks@.contains_key(
                            self.request_residency@[r].block_ids@[q])
                        && self.blocks@[self.request_residency@[r]
                            .block_ids@[q]].prefix_depth > 0
                        implies {
                            let rids = self.request_residency@[r].block_ids@;
                            let rb = rids[q];
                            &&& self.blocks@[rb].prefix_depth as int == q + 1
                            &&& self.blocks@[rb].parent_block
                                == if q == 0 { None } else { Some(rids[q - 1]) }
                        }
                    by {
                        let rids = self.request_residency@[r].block_ids@;
                        let rb = rids[q];
                        if rb == bid {
                            assert(self.request_residency@[rid].block_ids@ == ids);
                            assert(ids[k as int] == bid);
                            assert(skip_blocks as int <= k as int);
                            assert((k as int) < (full_blocks as int));
                            assert((full_blocks as int)
                                == prompt_tokens@.len() as int / (BLOCK_SIZE_SPEC as int));
                            assert((k as int) < prompt_tokens@.len() as int
                                / (BLOCK_SIZE_SPEC as int));
                            assert(ids == old(self).request_residency@[rid].block_ids@);
                            assert(old(self).blocks@[old(self).request_residency@[rid]
                                .block_ids@[k as int]].refcount == 1);
                            assert(old(self).blocks@[ids[k as int]].refcount == 1);
                            assert(self.blocks@[bid].refcount
                                == old(self).blocks@[bid].refcount);
                            assert(self.blocks@[bid].refcount == 1);
                            assert(refcount_valid(self));
                            let holders = residency_holders_of(self, bid);
                            assert(holders.contains(r));
                            assert(holders.contains(rid));
                            if r != rid {
                                lemma_two_holders(holders, r, rid);
                                assert(self.blocks@[bid].refcount as int >= 2);
                                assert(false);
                            }
                            assert(r == rid);
                            assert(q == k as int) by {
                                if q != k as int {
                                    assert(ids.no_duplicates());
                                    reveal(Seq::no_duplicates);
                                }
                            }
                            assert(self.blocks@[bid].prefix_depth as int == q + 1);
                            if q == 0 {
                                assert(self.blocks@[bid].parent_block
                                    == Option::<BlockId>::None);
                            } else {
                                assert(self.blocks@[bid].parent_block == Some(ids[q - 1]));
                            }
                        } else {
                            assert(self.blocks@[rb] == pre_iter.blocks@[rb]);
                            assert(self.request_residency@ == pre_iter.request_residency@);
                            assert(registered_provenance_aligned(&pre_iter));
                        }
                    }
                }
                assert(blocks_dom_in_range(self)) by {
                    assert forall|b2: BlockId| #[trigger] self.blocks@.contains_key(b2)
                        implies b2 < self.num_blocks
                    by {
                        assert(pre_iter.blocks@.contains_key(b2));
                        assert(blocks_dom_in_range(&pre_iter));
                    }
                }
                assert(cs_valid(self));
            }
            let ghost pre_reg = *self;
            // Register the hash if this content chain is new — and nonzero
            // (0 is the reserved "unregistered" stamp; skipping keeps the
            // registry-hygiene invariant, at worst losing reuse for a
            // one-in-2^64 content chain).
            let absent = match self.hash_to_block.get(&ph) {
                Some(_) => false,
                None => true,
            };
            if ph != 0 && absent {
                self.hash_to_block.insert(ph, bid);
                proof {
                    assert(self.hash_to_block@ == pre_reg.hash_to_block@.insert(ph, bid));
                    assert(hash_to_block_in_range(self)) by {
                        assert forall|h: u64| #[trigger] self.hash_to_block@.contains_key(h)
                            implies self.blocks@.contains_key(self.hash_to_block@[h])
                                && self.blocks@[self.hash_to_block@[h]].tokens@.len()
                                    == BLOCK_SIZE_SPEC as int
                        by {
                            if h == ph {
                                assert(self.hash_to_block@[h] == bid);
                                assert(self.blocks@[bid].tokens@
                                    == pre_iter.blocks@[bid].tokens@);
                                assert(pre_iter.blocks@[bid].tokens@.len()
                                    == BLOCK_SIZE_SPEC as int) by {
                                    assert(pre_iter.blocks@.contains_key(ids[k as int]));
                                    assert(pre_iter.blocks@[ids[k as int]].tokens@.len()
                                        == BLOCK_SIZE_SPEC as int);
                                }
                            } else {
                                assert(pre_reg.hash_to_block@.contains_key(h));
                                assert(self.hash_to_block@[h] == pre_reg.hash_to_block@[h]);
                                assert(hash_to_block_in_range(&pre_reg));
                            }
                        }
                    }
                    assert(hash_to_block_consistent(self)) by {
                        assert forall|h: u64| #[trigger] self.hash_to_block@.contains_key(h)
                            implies self.blocks@[self.hash_to_block@[h]].hash_value == h
                                && self.blocks@[self.hash_to_block@[h]].prefix_depth > 0
                        by {
                            if h == ph {
                                assert(self.hash_to_block@[h] == bid);
                                assert(self.blocks@[bid].hash_value == ph);
                            } else {
                                assert(pre_reg.hash_to_block@.contains_key(h));
                                assert(self.hash_to_block@[h] == pre_reg.hash_to_block@[h]);
                                assert(hash_to_block_consistent(&pre_reg));
                            }
                        }
                    }
                    assert(cs_valid(self));
                }
            }
            proof {
                // Re-establish the hash-map bookkeeping invariants.
                assert forall|h: u64| #[trigger] old(self).hash_to_block@.contains_key(h)
                    implies self.hash_to_block@.contains_key(h)
                        && self.hash_to_block@[h] == old(self).hash_to_block@[h]
                by {
                    assert(pre_reg.hash_to_block@.contains_key(h));
                    if absent {
                        assert(h != ph);
                    }
                }
                assert forall|h: u64| #[trigger] self.hash_to_block@.contains_key(h)
                    implies old(self).hash_to_block@.contains_key(h)
                        || ids.subrange(skip_blocks as int, k as int + 1)
                            .contains(self.hash_to_block@[h])
                by {
                    if absent && h == ph {
                        assert(self.hash_to_block@[h] == bid);
                        assert(ids.subrange(skip_blocks as int, k as int + 1)
                            [k as int - skip_blocks as int] == bid);
                    } else {
                        assert(pre_reg.hash_to_block@.contains_key(h));
                        assert(self.hash_to_block@[h] == pre_reg.hash_to_block@[h]);
                        if !old(self).hash_to_block@.contains_key(h) {
                            assert(ids.subrange(skip_blocks as int, k as int).contains(
                                pre_reg.hash_to_block@[h]));
                            let j = choose|j: int|
                                0 <= j < k as int - skip_blocks as int
                                && ids.subrange(skip_blocks as int, k as int)[j]
                                    == pre_reg.hash_to_block@[h];
                            assert(ids.subrange(skip_blocks as int, k as int + 1)[j]
                                == pre_reg.hash_to_block@[h]);
                        }
                    }
                }
                // Full blocks keep their exact token counts across the
                // hash stamp (tokens preserved; other ids untouched).
                assert forall|b2: BlockId| #[trigger] pre_iter.blocks@.contains_key(b2)
                    implies self.blocks@.contains_key(b2)
                        && self.blocks@[b2].tokens@ == pre_iter.blocks@[b2].tokens@
                by {
                    if b2 == bid {
                        assert(self.blocks@[bid].tokens@ == pre_iter.blocks@[bid].tokens@);
                    } else {
                        assert(self.blocks@[b2] == pre_iter.blocks@[b2]);
                        assert(self.blocks@.dom().contains(b2));
                    }
                }
                assert(full_blocks_sized(&pre_iter, ids, full_blocks as int));
                lemma_full_blocks_sized_transfer(&pre_iter, self, ids, full_blocks as int);
                assert forall|b2: BlockId|
                    #[trigger] old(self).blocks@.contains_key(b2)
                    && !ids.subrange(skip_blocks as int, k as int + 1)
                        .contains(b2)
                    implies self.blocks@[b2] == old(self).blocks@[b2]
                by {
                    assert(ids.subrange(skip_blocks as int, k as int + 1)
                        [k as int - skip_blocks as int] == bid);
                    assert(b2 != bid);
                    assert(!ids.subrange(skip_blocks as int, k as int)
                        .contains(b2));
                    assert(pre_iter.blocks@[b2] == old(self).blocks@[b2]);
                    assert(self.blocks@[b2] == pre_iter.blocks@[b2]);
                }
                lemma_free_queue_valid_token_frame(&pre_reg, self);
            }
            k += 1;
        }
        proof {
            reveal(registration_registry_frame);
        }
        proof {
            reveal(registration_positive_page_origin);
            let stamped = ids.subrange(
                skip_blocks as int, full_blocks as int,
            );
            assert forall|bid: BlockId|
                #[trigger] self.blocks@[bid].prefix_depth > 0
                && self.blocks@.contains_key(bid)
                && self.blocks@[bid].prefix_depth > 0
                implies positive_page_unchanged_from_pre(
                    old(self), self, bid,
                ) || stamped.contains(bid)
            by {
                if !stamped.contains(bid) {
                    assert(old(self).blocks@.contains_key(bid)) by {
                        assert(self.blocks@.dom() == old(self).blocks@.dom());
                    }
                    assert(self.blocks@[bid] == old(self).blocks@[bid]);
                }
            }
        }
        proof {
            reveal(positive_provenance_metadata_frame);
            let stamped = ids.subrange(
                skip_blocks as int, full_blocks as int,
            );
            assert forall|bid: BlockId|
                #[trigger] old(self).blocks@[bid].prefix_depth > 0
                && old(self).blocks@.contains_key(bid)
                && old(self).blocks@[bid].prefix_depth > 0
                implies self.blocks@.contains_key(bid)
                    && self.blocks@[bid].tokens@
                        == old(self).blocks@[bid].tokens@
                    && self.blocks@[bid].hash_value
                        == old(self).blocks@[bid].hash_value
                    && self.blocks@[bid].prefix_depth
                        == old(self).blocks@[bid].prefix_depth
                    && self.blocks@[bid].parent_block
                        == old(self).blocks@[bid].parent_block
            by {
                assert(!stamped.contains(bid)) by {
                    if stamped.contains(bid) {
                        let q = stamped.index_of(bid);
                        let j = skip_blocks as int + q;
                        assert(ids[j] == bid);
                        assert(old(self).blocks@[ids[j]].prefix_depth == 0);
                    }
                }
                assert(self.blocks@[bid] == old(self).blocks@[bid]);
            }
        }
        proof {
            if persistent_provenance_closed(old(self)) {
                reveal(persistent_provenance_closed);
                let stamped = ids.subrange(
                    skip_blocks as int, full_blocks as int,
                );
                assert forall|bid: BlockId|
                    #[trigger] self.blocks@[bid].prefix_depth > 0
                    && self.blocks@.contains_key(bid)
                    && self.blocks@[bid].prefix_depth > 0
                    implies {
                        let entry = self.blocks@[bid];
                        &&& entry.tokens@.len() == BLOCK_SIZE_SPEC as int
                        &&& (entry.prefix_depth == 1 ==>
                            entry.parent_block is None)
                        &&& (entry.prefix_depth > 1 ==> {
                            let parent = entry.parent_block.unwrap();
                            &&& entry.parent_block is Some
                            &&& self.blocks@.contains_key(parent)
                            &&& self.blocks@[parent].prefix_depth + 1
                                == entry.prefix_depth
                        })
                    }
                by {
                    assert(old(self).blocks@.contains_key(bid)) by {
                        assert(self.blocks@.dom() == old(self).blocks@.dom());
                    }
                    if stamped.contains(bid) {
                        let q = stamped.index_of(bid);
                        let j = skip_blocks as int + q;
                        assert(skip_blocks as int <= j < full_blocks as int);
                        assert(ids[j] == bid);
                        assert(self.blocks@[bid].tokens@.len()
                            == BLOCK_SIZE_SPEC as int) by {
                            assert(full_blocks_sized(
                                self, ids, full_blocks as int,
                            ));
                        }
                        assert(self.blocks@[bid].prefix_depth as int == j + 1);
                        if j == 0 {
                            assert(self.blocks@[bid].parent_block is None);
                        } else {
                            let parent = ids[j - 1];
                            assert(self.blocks@[bid].parent_block == Some(parent));
                            assert(self.blocks@.contains_key(parent)) by {
                                assert(full_blocks_sized(
                                    self, ids, full_blocks as int,
                                ));
                            }
                            if skip_blocks as int <= j - 1 {
                                assert(self.blocks@[parent].prefix_depth as int == j);
                            } else {
                                assert(j == skip_blocks as int);
                                assert(registered_prefix_chain(
                                    old(self).blocks@,
                                    ids.subrange(0, skip_blocks as int),
                                ));
                                assert(ids.subrange(0, skip_blocks as int)[j - 1]
                                    == ids[j - 1]);
                                assert(old(self).blocks@[parent].prefix_depth as int == j);
                                assert(!stamped.contains(parent)) by {
                                    if stamped.contains(parent) {
                                        let qp = stamped.index_of(parent);
                                        assert(ids[skip_blocks as int + qp] == parent);
                                        assert(ids[j - 1] == parent);
                                        reveal(Seq::no_duplicates);
                                    }
                                }
                                assert(self.blocks@[parent]
                                    == old(self).blocks@[parent]);
                            }
                        }
                    } else {
                        assert(self.blocks@[bid] == old(self).blocks@[bid]);
                        if self.blocks@[bid].prefix_depth > 1 {
                            let parent = self.blocks@[bid].parent_block.unwrap();
                            assert(old(self).blocks@[bid].parent_block
                                == Some(parent));
                            assert(old(self).blocks@.contains_key(parent));
                            assert(old(self).blocks@[parent].prefix_depth + 1
                                == old(self).blocks@[bid].prefix_depth);
                            assert(old(self).blocks@[parent].prefix_depth > 0);
                            assert(!stamped.contains(parent)) by {
                                if stamped.contains(parent) {
                                    let qp = stamped.index_of(parent);
                                    let jp = skip_blocks as int + qp;
                                    assert(ids[jp] == parent);
                                    assert(old(self).blocks@[ids[jp]]
                                        .prefix_depth == 0);
                                }
                            }
                            assert(self.blocks@[parent]
                                == old(self).blocks@[parent]);
                        }
                    }
                }
                assert(persistent_provenance_closed(self));
            }
        }
        proof { lemma_free_queue_token_to_valid(self); }
        registered
    }

    // Publish prompt hashes only after the caller has completed all reuse
    // decisions for this forward.  This preserves cross-step prefix caching
    // while making same-forward reuse impossible by lifecycle, rather than
    // by an exclusion list threaded through the admission loop.
    #[verifier::spinoff_prover]
    #[verifier::rlimit(300)]
    pub fn publish_admitted_prefixes(
        &mut self,
        scheduled: &Vec<RequestId>,
        cu_k: &Vec<u64>,
        decode_count: usize,
        Ghost(plan_pre): Ghost<CacheScheduler>,
    )
        requires
            cs_valid(&plan_pre),
            tail_write_exclusive(&plan_pre),
            cs_valid(old(self)),
            free_queue_valid(old(self)),
            persistent_provenance_closed(old(self)),
            registry_entries_from_pre(
                &plan_pre, old(self), Seq::<u64>::empty(),
            ),
            positive_provenance_origin(&plan_pre, old(self)),
            decode_count <= scheduled@.len(),
            cu_k@.len() == scheduled@.len() + 1,
            scheduled@.no_duplicates(),
            old(self).live_requests@ == plan_pre.live_requests@,
            forall|k: int| decode_count as int <= k < scheduled@.len()
                ==> plan_pre.waiting@.contains(#[trigger] scheduled@[k]),
            forall|k: int| decode_count as int <= k < scheduled@.len()
                ==> #[trigger] admitted_registration_ready_at(
                    old(self), scheduled@, cu_k@, k),
            forall|r: RequestId| #[trigger] plan_pre.running@.contains(r)
                ==> plan_pre.request_residency@.contains_key(r)
                    && old(self).request_residency@.contains_key(r)
                    && old(self).request_residency@[r]
                        == plan_pre.request_residency@[r],
            forall|r: RequestId| #[trigger] plan_pre.running@.contains(r)
                && plan_pre.live_requests@.contains_key(r)
                ==> {
                    let t = plan_pre.request_residency@[r].block_ids@[
                        plan_pre.request_residency@[r].block_ids@.len() - 1];
                    old(self).blocks@.contains_key(t)
                        && old(self).blocks@[t].tokens@
                            == plan_pre.blocks@[t].tokens@
                        && old(self).blocks@[t].refcount
                            == plan_pre.blocks@[t].refcount
                        && old(self).blocks@[t].prefix_depth
                            == plan_pre.blocks@[t].prefix_depth
                        && old(self).blocks@[t].hash_value
                            == plan_pre.blocks@[t].hash_value
                },
            residency_history_aligned(old(self)),
            slot_mapping_aligned(old(self)),
            residency_running_aligned(old(self)),
        ensures
            cs_valid(final(self)),
            free_queue_valid(final(self)),
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
                ==> final(self).blocks@.contains_key(bid)
                    && final(self).blocks@[bid].tokens@ == old(self).blocks@[bid].tokens@
                    && final(self).blocks@[bid].refcount == old(self).blocks@[bid].refcount,
            residency_history_aligned(final(self)),
            slot_mapping_aligned(final(self)),
            residency_running_aligned(final(self)),
            persistent_provenance_closed(final(self)),
            positive_pages_from_publication_base_or_processed(
                old(self), final(self), scheduled@,
                RT::u64_seq_to_int_repr(cu_k@), decode_count as int,
                scheduled@.len() as int,
            ),
            positive_provenance_metadata_frame(old(self), final(self)),
            forall|r: RequestId| #[trigger] plan_pre.running@.contains(r)
                && plan_pre.live_requests@.contains_key(r)
                ==> {
                    let t = plan_pre.request_residency@[r].block_ids@[
                        plan_pre.request_residency@[r].block_ids@.len() - 1];
                    final(self).blocks@.contains_key(t)
                        && final(self).blocks@[t].tokens@ == plan_pre.blocks@[t].tokens@
                        && final(self).blocks@[t].refcount == plan_pre.blocks@[t].refcount
                        && final(self).blocks@[t].prefix_depth
                            == plan_pre.blocks@[t].prefix_depth
                        && final(self).blocks@[t].hash_value == plan_pre.blocks@[t].hash_value
                },
            forall|k: int| decode_count as int <= k < scheduled@.len()
                ==> #[trigger] admitted_pages_exclusive_post(final(self), scheduled@, k),
            forall|k: int| decode_count as int <= k < scheduled@.len()
                ==> {
                    let rid = #[trigger] scheduled@[k];
                    let ids = final(self).request_residency@[rid].block_ids@;
                    let c = final(self).request_residency@[rid]
                        .cached_prefix_blocks as int;
                    registered_prefix_chain(
                        final(self).blocks@, ids.subrange(0, c),
                    )
                },
            forall|k: int| decode_count as int <= k < scheduled@.len()
                ==> {
                    let rid = #[trigger] scheduled@[k];
                    let ids = final(self).request_residency@[rid].block_ids@;
                    let prompt = final(self).live_requests@[rid].prompt_tokens@;
                    let end = cu_k@[k + 1] as int - cu_k@[k] as int;
                    let full = end / (BLOCK_SIZE_SPEC as int);
                    &&& 0 <= full <= ids.len()
                    &&& 0 <= end <= prompt.len()
                    &&& prompt.len() <= u64::MAX as int
                    &&& blocks_needed_for(end as nat) <= u64::MAX as nat
                    &&& registered_prefix_chain(
                        final(self).blocks@, ids.subrange(0, full),
                    )
                    &&& token_placement_prefix(
                        final(self).blocks@, ids, prompt,
                        full * (BLOCK_SIZE_SPEC as int),
                    )
                    &&& forall|l: int| full <= l < ids.len()
                        && #[trigger] final(self).blocks@.contains_key(ids[l])
                        ==> final(self).blocks@[ids[l]].prefix_depth == 0
                },
    {
        // Each per-request publication preserves queue validity. This outer
        // loop composes those effects without inspecting availability queues.
        hide(free_queue_valid);
        proof {
            lemma_publication_positive_pages_base(
                old(self), scheduled@, RT::u64_seq_to_int_repr(cu_k@),
                decode_count as int,
            );
            lemma_positive_provenance_metadata_frame_refl(old(self));
            assert forall|k: int| decode_count as int <= k < scheduled@.len()
                implies old(self).request_residency@.contains_key(
                    #[trigger] scheduled@[k])
            by {
                assert(admitted_registration_ready_at(
                    old(self), scheduled@, cu_k@, k,
                ));
            }
            assert forall|k: int| decode_count as int <= k < scheduled@.len()
                implies {
                    let chain_rid = #[trigger] scheduled@[k];
                    let chain_ids = old(self).request_residency@[chain_rid]
                        .block_ids@;
                    let chain_c = old(self).request_residency@[chain_rid]
                        .cached_prefix_blocks as int;
                    &&& 0 <= chain_c <= chain_ids.len()
                    &&& registered_prefix_chain(
                        old(self).blocks@, chain_ids.subrange(0, chain_c),
                    )
                }
            by {
                assert(admitted_registration_ready_at(
                    old(self), scheduled@, cu_k@, k,
                ));
                assert(admitted_pages_exclusive_at(
                    old(self), scheduled@, Seq::<u64>::empty(), k,
                ));
            }
            assert forall|k: int| decode_count as int <= k < scheduled@.len()
                implies #[trigger] admitted_pages_exclusive_post(
                    self, scheduled@, k)
            by {
                assert(admitted_registration_ready_at(
                    self, scheduled@, cu_k@, k,
                ));
                assert(admitted_pages_exclusive_at(
                    self, scheduled@, Seq::<u64>::empty(), k,
                ));
            }
        }
        let mut i = decode_count;
        while i < scheduled.len()
            invariant
                decode_count <= i,
                i <= scheduled@.len(),
                cs_valid(&plan_pre),
                tail_write_exclusive(&plan_pre),
                cs_valid(old(self)),
                cs_valid(self),
                free_queue_valid(self),
                persistent_provenance_closed(self),
                decode_count <= scheduled@.len(),
                cu_k@.len() == scheduled@.len() + 1,
                scheduled@.no_duplicates(),
                self.config == old(self).config,
                self.num_blocks == old(self).num_blocks,
                self.free_blocks == old(self).free_blocks,
                self.running@ == old(self).running@,
                self.waiting@ == old(self).waiting@,
                self.live_requests@ == old(self).live_requests@,
                self.accepted_requests@ == old(self).accepted_requests@,
                self.live_requests@ == plan_pre.live_requests@,
                self.request_residency@ == old(self).request_residency@,
                forall|k: int| decode_count as int <= k < scheduled@.len()
                    ==> self.request_residency@.contains_key(
                        #[trigger] scheduled@[k]),
                forall|r: RequestId| #[trigger] plan_pre.running@.contains(r)
                    ==> plan_pre.request_residency@.contains_key(r)
                        && self.request_residency@.contains_key(r)
                        && self.request_residency@[r]
                            == plan_pre.request_residency@[r],
                self.blocks@.dom() == old(self).blocks@.dom(),
                forall|bid: BlockId|
                    #[trigger] old(self).blocks@.contains_key(bid)
                    ==> self.blocks@.contains_key(bid)
                        && self.blocks@[bid].tokens@ == old(self).blocks@[bid].tokens@
                        && self.blocks@[bid].refcount == old(self).blocks@[bid].refcount,
                forall|k: int| decode_count as int <= k < scheduled@.len()
                    ==> plan_pre.waiting@.contains(#[trigger] scheduled@[k]),
                forall|k: int| i as int <= k < scheduled@.len()
                    ==> #[trigger] admitted_registration_ready_at(
                        self, scheduled@, cu_k@, k),
                forall|k: int| decode_count as int <= k < scheduled@.len()
                    ==> #[trigger] admitted_pages_exclusive_post(
                        self, scheduled@, k),
                forall|k: int| decode_count as int <= k < scheduled@.len()
                    ==> {
                        let chain_rid = #[trigger] scheduled@[k];
                        let chain_ids = self.request_residency@[chain_rid]
                            .block_ids@;
                        let chain_c = self.request_residency@[chain_rid]
                            .cached_prefix_blocks as int;
                        &&& 0 <= chain_c <= chain_ids.len()
                        &&& registered_prefix_chain(
                            self.blocks@, chain_ids.subrange(0, chain_c),
                        )
                    },
                forall|k: int| decode_count as int <= k < i as int
                    ==> {
                        let chain_rid = #[trigger] scheduled@[k];
                        let chain_ids = self.request_residency@[chain_rid]
                            .block_ids@;
                        let chain_prompt = self.live_requests@[chain_rid]
                            .prompt_tokens@;
                        let chain_end = cu_k@[k + 1] as int
                            - cu_k@[k] as int;
                        let chain_full = chain_end / (BLOCK_SIZE_SPEC as int);
                        &&& 0 <= chain_full <= chain_ids.len()
                        &&& 0 <= chain_end <= chain_prompt.len()
                        &&& chain_prompt.len() <= u64::MAX as int
                        &&& blocks_needed_for(chain_end as nat)
                            <= u64::MAX as nat
                        &&& registered_prefix_chain(
                            self.blocks@,
                            chain_ids.subrange(0, chain_full),
                        )
                        &&& token_placement_prefix(
                            self.blocks@, chain_ids, chain_prompt,
                            chain_full * (BLOCK_SIZE_SPEC as int),
                        )
                        &&& forall|l: int| chain_full <= l < chain_ids.len()
                            && #[trigger] self.blocks@.contains_key(chain_ids[l])
                            ==> self.blocks@[chain_ids[l]].prefix_depth == 0
                    },
                forall|r: RequestId| #[trigger] plan_pre.running@.contains(r)
                    && plan_pre.live_requests@.contains_key(r)
                    ==> {
                        let t = plan_pre.request_residency@[r].block_ids@[
                            plan_pre.request_residency@[r].block_ids@.len() - 1];
                        self.blocks@.contains_key(t)
                            && self.blocks@[t].tokens@ == plan_pre.blocks@[t].tokens@
                            && self.blocks@[t].refcount == plan_pre.blocks@[t].refcount
                            && self.blocks@[t].prefix_depth
                                == plan_pre.blocks@[t].prefix_depth
                            && self.blocks@[t].hash_value == plan_pre.blocks@[t].hash_value
                    },
                residency_history_aligned(self),
                slot_mapping_aligned(self),
                residency_running_aligned(self),
                positive_pages_from_publication_base_or_processed(
                    old(self), self, scheduled@,
                    RT::u64_seq_to_int_repr(cu_k@), decode_count as int,
                    i as int,
                ),
                positive_provenance_metadata_frame(old(self), self),
            decreases scheduled@.len() - i,
        {
            let rid = scheduled[i];
            proof {
                assert(admitted_registration_ready_at(
                    self, scheduled@, cu_k@, i as int,
                ));
                assert(self.live_requests@.contains_key(rid));
                assert(self.request_residency@.contains_key(rid));
            }
            let state = match self.live_requests.get(&rid) {
                Some(s) => s.clone(),
                None => {
                    proof { assert(false); }
                    RequestState::from_parts(
                        rid,
                        Vec::new(),
                        Vec::new(),
                        SamplerState::empty(),
                        0,
                        EosTokenSet::singleton(0),
                        false,
                    )
                },
            };
            let cpb = match self.request_residency.get(&rid) {
                Some(r) => r.cached_prefix_blocks,
                None => {
                    proof { assert(false); }
                    0
                },
            };
            let row_end_u64 = cu_k[i + 1] - cu_k[i];
            let row_end = row_end_u64 as usize;
            proof {
                let c = cpb as int;
                let n = state.prompt_tokens@.len() as int;
                assert(row_end as int == row_end_u64 as int);
                assert(c * (BLOCK_SIZE_SPEC as int) < row_end as int <= n);
                assert(c <= c * (BLOCK_SIZE_SPEC as int)) by (nonlinear_arith)
                    requires 0 <= c, BLOCK_SIZE_SPEC as int == 64,
                {}
                assert(c < (usize::MAX as int));
            }
            let c_us = cpb as usize;
            proof {
                assert(c_us as int == cpb as int);
                assert(state.prompt_tokens@
                    == self.live_requests@[rid].prompt_tokens@);
                let ids = self.request_residency@[rid].block_ids@;
                let c = cpb as int;
                let end = row_end as int;
                assert forall|j: int| c <= j
                    && j < end / (BLOCK_SIZE_SPEC as int)
                    implies self.blocks@[#[trigger] ids[j]].refcount == 1
                        && self.blocks@[ids[j]].prefix_depth == 0
                        && self.blocks@[ids[j]].parent_block
                            == Option::<BlockId>::None
                by {
                    assert(admitted_pages_exclusive_at(
                        self, scheduled@, Seq::<u64>::empty(), i as int,
                    ));
                }
            }
            let publish_tokens = token_prefix(&state.prompt_tokens, row_end);
            let ghost pre_reg = *self;
            proof {
                let ids = self.request_residency@[rid].block_ids@;
                let end = row_end as int;
                let n = state.prompt_tokens@.len() as int;
                assert(publish_tokens@.len() == end);
                crate::proof::tensor::geometry::lemma_blocks_needed_monotone(
                    end as nat, n as nat,
                );
                assert(blocks_needed_for(publish_tokens@.len()) <= ids.len());
                assert(token_placement_prefix(
                    self.blocks@, ids, publish_tokens@, end,
                )) by {
                    reveal(token_placement_prefix);
                    assert forall|p: int| 0 <= p < end implies
                        #[trigger] token_placement_at(
                            self.blocks@, ids, publish_tokens@, p,
                        )
                    by {
                        lemma_token_placement_prefix_at(
                            self.blocks@, ids, state.prompt_tokens@, n, p,
                        );
                        reveal(token_placement_at);
                        assert(publish_tokens@[p] == state.prompt_tokens@[p]);
                    }
                }
            }
            let _registered = self.publish_full_prefix_pages(
                rid, &publish_tokens, c_us,
            );
            proof {
                assert(state.prompt_tokens@
                    == plan_pre.live_requests@[rid].prompt_tokens@);
                lemma_publication_positive_pages_register(
                    old(self), &pre_reg, self, scheduled@,
                    RT::u64_seq_to_int_repr(cu_k@),
                    decode_count as int, i as int, rid,
                    row_end as int, c_us as int,
                );
                lemma_positive_provenance_metadata_frame_transitive(
                    old(self), &pre_reg, self,
                );
                assert(self.running@ == pre_reg.running@);
                assert(self.waiting@ == pre_reg.waiting@);
                assert(self.live_requests@ == pre_reg.live_requests@);
                assert(self.request_residency@ == pre_reg.request_residency@);
                assert forall|k: int|
                    decode_count as int <= k < scheduled@.len()
                    implies {
                        let chain_rid = #[trigger] scheduled@[k];
                        let chain_ids = self.request_residency@[chain_rid]
                            .block_ids@;
                        let chain_c = self.request_residency@[chain_rid]
                            .cached_prefix_blocks as int;
                        &&& 0 <= chain_c <= chain_ids.len()
                        &&& registered_prefix_chain(
                            self.blocks@, chain_ids.subrange(0, chain_c),
                        )
                    }
                by {
                    let other = scheduled@[k];
                    assert(pre_reg.request_residency@.contains_key(other));
                    let other_ids = pre_reg.request_residency@[other].block_ids@;
                    let other_c = pre_reg.request_residency@[other]
                        .cached_prefix_blocks as int;
                    let rid_ids = pre_reg.request_residency@[rid].block_ids@;
                    let rid_c = pre_reg.request_residency@[rid]
                        .cached_prefix_blocks as int;
                    let rid_full = row_end as int
                        / (BLOCK_SIZE_SPEC as int);
                    assert(admitted_pages_exclusive_post(
                        &pre_reg, scheduled@, k,
                    ));
                    assert(0 <= other_c < other_ids.len());
                    assert(registered_prefix_chain(
                        pre_reg.blocks@,
                        other_ids.subrange(0, other_c),
                    ));
                    assert forall|j: int|
                        0 <= j < other_ids.subrange(0, other_c).len()
                        implies {
                            let b = #[trigger]
                                other_ids.subrange(0, other_c)[j];
                            &&& self.blocks@.contains_key(b)
                            &&& self.blocks@[b].prefix_depth
                                == pre_reg.blocks@[b].prefix_depth
                            &&& self.blocks@[b].parent_block
                                == pre_reg.blocks@[b].parent_block
                    }
                    by {
                        let b = other_ids.subrange(0, other_c)[j];
                        assert(0 <= j < other_c);
                        assert(b == other_ids[j]);
                        assert(pre_reg.request_residency@[other]
                            .block_ids@.contains(b));
                        assert(!rid_ids.subrange(rid_c, rid_full).contains(b)) by {
                            if rid_ids.subrange(rid_c, rid_full).contains(b) {
                                assert(pre_reg.request_residency@[rid]
                                    .block_ids@.contains(b));
                                if other == rid {
                                    let q = rid_ids.subrange(rid_c, rid_full)
                                        .index_of(b);
                                    assert(rid_ids[rid_c + q] == b);
                                    assert(rid_ids[j] == b);
                                    assert(j < rid_c);
                                    assert(rid_c <= rid_c + q);
                                    assert(rid_ids.no_duplicates());
                                    reveal(Seq::no_duplicates);
                                } else {
                                    assert(pre_reg.blocks@[b].refcount == 1) by {
                                        let q = rid_ids.subrange(rid_c, rid_full)
                                            .index_of(b);
                                        assert(rid_ids[rid_c + q] == b);
                                    }
                                    lemma_refcount_one_excludes_other_holder(
                                        &pre_reg, b, rid, other,
                                    );
                                    assert(pre_reg.request_residency@
                                        .contains_key(other));
                                    assert(pre_reg.request_residency@[other]
                                        .block_ids@.contains(b));
                                    assert(false);
                                }
                            }
                        }
                        assert(self.blocks@[b] == pre_reg.blocks@[b]);
                    }
                    lemma_registered_prefix_chain_transfer(
                        pre_reg.blocks@, self.blocks@,
                        other_ids.subrange(0, other_c),
                    );
                }
                assert forall|k: int| decode_count as int <= k < (i + 1) as int
                    implies {
                        let chain_rid = #[trigger] scheduled@[k];
                        let chain_ids = self.request_residency@[chain_rid]
                            .block_ids@;
                        let chain_prompt = self.live_requests@[chain_rid]
                            .prompt_tokens@;
                        let chain_end = cu_k@[k + 1] as int
                            - cu_k@[k] as int;
                        let chain_full = chain_end / (BLOCK_SIZE_SPEC as int);
                        &&& 0 <= chain_full <= chain_ids.len()
                        &&& 0 <= chain_end <= chain_prompt.len()
                        &&& chain_prompt.len() <= u64::MAX as int
                        &&& blocks_needed_for(chain_end as nat)
                            <= u64::MAX as nat
                        &&& registered_prefix_chain(
                            self.blocks@,
                            chain_ids.subrange(0, chain_full),
                        )
                        &&& token_placement_prefix(
                            self.blocks@, chain_ids, chain_prompt,
                            chain_full * (BLOCK_SIZE_SPEC as int),
                        )
                        &&& forall|l: int| chain_full <= l < chain_ids.len()
                            && #[trigger] self.blocks@.contains_key(chain_ids[l])
                            ==> self.blocks@[chain_ids[l]].prefix_depth == 0
                    }
                by {
                    let other = scheduled@[k];
                    let other_ids = pre_reg.request_residency@[other].block_ids@;
                    let other_prompt = pre_reg.live_requests@[other]
                        .prompt_tokens@;
                    let other_end = cu_k@[k + 1] as int - cu_k@[k] as int;
                    let other_full = other_end / (BLOCK_SIZE_SPEC as int);
                    let rid_ids = pre_reg.request_residency@[rid].block_ids@;
                    let rid_c = pre_reg.request_residency@[rid]
                        .cached_prefix_blocks as int;
                    let rid_full = row_end as int
                        / (BLOCK_SIZE_SPEC as int);
                    assert forall|bid: BlockId|
                        #[trigger] pre_reg.blocks@.contains_key(bid)
                        implies self.blocks@.contains_key(bid)
                            && self.blocks@[bid].tokens@
                                == pre_reg.blocks@[bid].tokens@
                    by {
                    }
                    if k == i as int {
                        assert(other == rid);
                        assert(other_ids == rid_ids);
                        assert(other_end == row_end as int);
                        assert(token_placement_prefix(
                            pre_reg.blocks@, other_ids, other_prompt,
                            other_full * (BLOCK_SIZE_SPEC as int),
                        ));
                        lemma_token_placement_prefix_transfer(
                            pre_reg.blocks@, self.blocks@, other_ids,
                            other_prompt,
                            other_full * (BLOCK_SIZE_SPEC as int),
                        );
                        assert(registered_prefix_chain(
                            self.blocks@,
                            other_ids.subrange(0, other_full),
                        )) by {
                            assert forall|j: int|
                                #![trigger self.blocks@[
                                    other_ids.subrange(0, other_full)[j]
                                ].prefix_depth]
                                0 <= j < other_ids.subrange(0, other_full).len()
                                implies {
                                    let b = other_ids.subrange(0, other_full)[j];
                                    &&& self.blocks@.contains_key(b)
                                    &&& self.blocks@[b].prefix_depth as int == j + 1
                                    &&& self.blocks@[b].parent_block
                                        == if j == 0 {
                                            None
                                        } else {
                                            Some(other_ids.subrange(
                                                0, other_full,
                                            )[j - 1])
                                        }
                                }
                            by {
                                assert(other_ids.subrange(0, other_full)[j]
                                    == other_ids[j]);
                                assert(self.request_residency@.contains_key(rid));
                                assert(self.request_residency@[rid].block_ids@
                                    == other_ids);
                                assert(self.blocks@.contains_key(other_ids[j])) by {
                                    assert(residency_blocks_in_range(self));
                                }
                                if j < rid_c {
                                    assert(registered_prefix_chain(
                                        self.blocks@,
                                        other_ids.subrange(0, rid_c),
                                    ));
                                    assert(other_ids.subrange(0, rid_c)[j]
                                        == other_ids[j]);
                                    assert(self.blocks@[other_ids[j]].prefix_depth
                                        as int == j + 1);
                                    if j > 0 {
                                        assert(other_ids.subrange(0, rid_c)[j - 1]
                                            == other_ids[j - 1]);
                                        assert(self.blocks@[other_ids[j]].parent_block
                                            == Some(other_ids[j - 1]));
                                    } else {
                                        assert(self.blocks@[other_ids[j]].parent_block
                                            == Option::<BlockId>::None);
                                    }
                                } else {
                                    assert(self.blocks@[other_ids[j]].prefix_depth
                                        as int == j + 1);
                                    if j > 0 {
                                        assert(other_ids.subrange(0, other_full)[j - 1]
                                            == other_ids[j - 1]);
                                        assert(self.blocks@[other_ids[j]].parent_block
                                            == Some(other_ids[j - 1]));
                                    } else {
                                        assert(self.blocks@[other_ids[j]].parent_block
                                            == Option::<BlockId>::None);
                                    }
                                }
                            }
                        }
                    } else {
                        assert(k < i as int);
                        assert(other != rid) by {
                            reveal(Seq::no_duplicates);
                        }
                        assert(registered_prefix_chain(
                            pre_reg.blocks@,
                            other_ids.subrange(0, other_full),
                        ));
                        assert forall|j: int|
                            0 <= j < other_ids.subrange(0, other_full).len()
                            implies {
                                let b = #[trigger]
                                    other_ids.subrange(0, other_full)[j];
                                &&& self.blocks@.contains_key(b)
                                &&& self.blocks@[b].prefix_depth
                                    == pre_reg.blocks@[b].prefix_depth
                                &&& self.blocks@[b].parent_block
                                    == pre_reg.blocks@[b].parent_block
                            }
                        by {
                            let b = other_ids[j];
                            assert(pre_reg.request_residency@[other]
                                .block_ids@.contains(b));
                            assert(!rid_ids.subrange(rid_c, rid_full).contains(b)) by {
                                if rid_ids.subrange(rid_c, rid_full).contains(b) {
                                    let z = rid_ids.subrange(rid_c, rid_full)
                                        .index_of(b);
                                    assert(rid_ids[rid_c + z] == b);
                                    assert(pre_reg.blocks@[b].refcount == 1);
                                    lemma_refcount_one_excludes_other_holder(
                                        &pre_reg, b, rid, other,
                                    );
                                    assert(false);
                                }
                            }
                            assert(self.blocks@[b] == pre_reg.blocks@[b]);
                        }
                        lemma_registered_prefix_chain_transfer(
                            pre_reg.blocks@, self.blocks@,
                            other_ids.subrange(0, other_full),
                        );
                        assert(token_placement_prefix(
                            pre_reg.blocks@, other_ids, other_prompt,
                            other_full * (BLOCK_SIZE_SPEC as int),
                        ));
                        lemma_token_placement_prefix_transfer(
                            pre_reg.blocks@, self.blocks@, other_ids,
                            other_prompt,
                            other_full * (BLOCK_SIZE_SPEC as int),
                        );
                    }
                    assert forall|l: int| other_full <= l < other_ids.len()
                        && #[trigger] self.blocks@.contains_key(other_ids[l])
                        implies self.blocks@[other_ids[l]].prefix_depth == 0
                    by {
                        let b = other_ids[l];
                        if k == i as int {
                            assert(other == rid);
                            assert(other_full == rid_full);
                            assert(pre_reg.blocks@[b].prefix_depth == 0) by {
                                assert(l >= rid_c);
                                assert(admitted_pages_exclusive_at(
                                    &pre_reg, scheduled@,
                                    Seq::<u64>::empty(), i as int,
                                ));
                            }
                            assert(self.blocks@[b] == pre_reg.blocks@[b]);
                        } else {
                            assert(other != rid);
                            assert(pre_reg.blocks@[b].prefix_depth == 0);
                            assert(!rid_ids.subrange(rid_c, rid_full).contains(b)) by {
                                if rid_ids.subrange(rid_c, rid_full).contains(b) {
                                    assert(pre_reg.request_residency@[rid]
                                        .block_ids@.contains(b));
                                    assert(pre_reg.request_residency@[other]
                                        .block_ids@.contains(b));
                                    assert(pre_reg.blocks@[b].refcount == 1);
                                    lemma_refcount_one_excludes_other_holder(
                                        &pre_reg, b, rid, other,
                                    );
                                    assert(false);
                                }
                            }
                            assert(self.blocks@[b] == pre_reg.blocks@[b]);
                        }
                    }
                }
                assert(residency_history_aligned(self)) by {
                    assert forall|r: RequestId|
                        #[trigger] self.running@.contains(r)
                        && self.live_requests@.contains_key(r)
                        implies {
                            let hist = history(self.live_requests@[r]).len() as int;
                            let ids = self.request_residency@[r].block_ids@;
                            let bsz = BLOCK_SIZE_SPEC as int;
                            &&& self.request_residency@.contains_key(r)
                            &&& ids.len() >= 1
                            &&& self.blocks@.contains_key(ids[ids.len() - 1])
                            &&& hist == (ids.len() - 1) * bsz
                                + self.blocks@[ids[ids.len() - 1]].tokens@.len()
                            &&& self.blocks@[ids[ids.len() - 1]].tokens@.len() >= 1
                            &&& token_placement_prefix(
                                self.blocks@, ids, history(self.live_requests@[r]), hist,
                            )
                            &&& (self.blocks@[ids[ids.len() - 1]].refcount == 1
                                || self.blocks@[ids[ids.len() - 1]].tokens@.len()
                                    == bsz)
                        }
                    by {
                        assert(residency_history_aligned(&pre_reg));
                        let ids = self.request_residency@[r].block_ids@;
                        let hist = history(self.live_requests@[r]).len() as int;
                        assert forall|bid: BlockId|
                            #[trigger] pre_reg.blocks@.contains_key(bid)
                            implies self.blocks@.contains_key(bid)
                                && self.blocks@[bid].tokens@
                                    == pre_reg.blocks@[bid].tokens@
                        by {}
                        lemma_token_placement_prefix_transfer(
                            pre_reg.blocks@, self.blocks@, ids,
                            history(self.live_requests@[r]), hist,
                        );
                    }
                }
                assert(slot_mapping_aligned(self));
                lemma_residency_running_aligned_frame(&pre_reg, self);
                assert forall|r: RequestId|
                    #[trigger] plan_pre.running@.contains(r)
                    && plan_pre.live_requests@.contains_key(r)
                    implies {
                        let t = plan_pre.request_residency@[r].block_ids@[
                            plan_pre.request_residency@[r].block_ids@.len() - 1];
                        self.blocks@.contains_key(t)
                            && self.blocks@[t].tokens@ == plan_pre.blocks@[t].tokens@
                            && self.blocks@[t].refcount == plan_pre.blocks@[t].refcount
                            && self.blocks@[t].prefix_depth
                                == plan_pre.blocks@[t].prefix_depth
                            && self.blocks@[t].hash_value == plan_pre.blocks@[t].hash_value
                    }
                by {
                    let t = plan_pre.request_residency@[r].block_ids@[
                        plan_pre.request_residency@[r].block_ids@.len() - 1];
                    assert(plan_pre.request_residency@.contains_key(r));
                    assert(pre_reg.request_residency@[r]
                        == plan_pre.request_residency@[r]);
                    assert(pre_reg.request_residency@[r].block_ids@.contains(t));
                    assert(plan_pre.blocks@[t].refcount == 1);
                    assert(pre_reg.blocks@[t].refcount == 1);
                    assert(r != rid) by {
                        assert(plan_pre.waiting@.contains(rid));
                        assert(queue_disjoint(&plan_pre));
                    }
                    lemma_refcount_one_excludes_other_holder(
                        &pre_reg, t, r, rid,
                    );
                    assert(!pre_reg.request_residency@[rid].block_ids@.contains(t));
                    assert(self.blocks@[t] == pre_reg.blocks@[t]);
                }
                assert forall|k: int| decode_count as int <= k < scheduled@.len()
                    implies #[trigger] admitted_pages_exclusive_post(
                        self, scheduled@, k)
                by {
                    let other = scheduled@[k];
                    assert(admitted_pages_exclusive_post(
                        &pre_reg, scheduled@, k,
                    ));
                    let ids = pre_reg.request_residency@[other].block_ids@;
                    let c = pre_reg.request_residency@[other]
                        .cached_prefix_blocks as int;
                    assert forall|l: int| c <= l < ids.len()
                        implies self.blocks@.contains_key(#[trigger] ids[l])
                            && self.blocks@[ids[l]].refcount == 1
                    by {
                        assert(pre_reg.blocks@[ids[l]].refcount == 1);
                    }
                    let tail = ids[ids.len() - 1];
                    if pre_reg.blocks@[tail].tokens@.len()
                        < BLOCK_SIZE_SPEC as int {
                        if other == rid {
                            let n = state.prompt_tokens@.len() as int;
                            let full = row_end as int
                                / (BLOCK_SIZE_SPEC as int);
                            let prompt_full = n / (BLOCK_SIZE_SPEC as int);
                            assert(full <= prompt_full) by {
                                vstd::arithmetic::div_mod::lemma_div_is_ordered(
                                    row_end as int, n,
                                    BLOCK_SIZE_SPEC as int,
                                );
                            }
                            assert(ids.len() - 1 >= prompt_full) by {
                                if ids.len() - 1 < prompt_full {
                                    assert(false) by (nonlinear_arith)
                                        requires ids.len() - 1 < prompt_full,
                                            prompt_full == n / 64,
                                            pre_reg.blocks@[tail].tokens@.len() < 64,
                                            n == (ids.len() - 1) * 64
                                                + pre_reg.blocks@[tail].tokens@.len(),
                                    {}
                                }
                            }
                            assert(ids.len() - 1 >= full) by {
                            }
                            assert(self.blocks@[tail] == pre_reg.blocks@[tail]);
                        } else {
                            assert(other != rid);
                            assert(pre_reg.blocks@[tail].refcount == 1);
                            lemma_refcount_one_excludes_other_holder(
                                &pre_reg, tail, other, rid,
                            );
                            assert(!pre_reg.request_residency@[rid].block_ids@
                                .contains(tail));
                            assert(self.blocks@[tail] == pre_reg.blocks@[tail]);
                        }
                    }
                }
                assert forall|k: int| (i + 1) as int <= k < scheduled@.len()
                    implies #[trigger] admitted_registration_ready_at(
                        self, scheduled@, cu_k@, k)
                by {
                    let other = scheduled@[k];
                    assert(admitted_registration_ready_at(
                        &pre_reg, scheduled@, cu_k@, k,
                    ));
                    assert(other != rid) by {
                        assert(scheduled@.no_duplicates());
                        reveal(Seq::no_duplicates);
                    }
                    let ids = pre_reg.request_residency@[other].block_ids@;
                    let c = pre_reg.request_residency@[other]
                        .cached_prefix_blocks as int;
                    assert forall|l: int| c <= l < ids.len()
                        implies self.blocks@.contains_key(#[trigger] ids[l])
                            && self.blocks@[ids[l]] == pre_reg.blocks@[ids[l]]
                    by {
                        assert(admitted_pages_exclusive_at(
                            &pre_reg, scheduled@, Seq::<u64>::empty(), k,
                        ));
                        assert(pre_reg.blocks@[ids[l]].refcount == 1);
                        lemma_refcount_one_excludes_other_holder(
                            &pre_reg, ids[l], other, rid,
                        );
                        assert(!pre_reg.request_residency@[rid].block_ids@
                            .contains(ids[l]));
                    }
                    assert(admitted_pages_exclusive_at(
                        self, scheduled@, Seq::<u64>::empty(), k,
                    ));
                    assert forall|h: u64|
                        #[trigger] self.hash_to_block@.contains_key(h)
                        implies !ids.subrange(c, ids.len() as int)
                            .contains(self.hash_to_block@[h])
                    by {
                        if ids.subrange(c, ids.len() as int)
                            .contains(self.hash_to_block@[h]) {
                            let b = self.hash_to_block@[h];
                            assert(pre_reg.hash_to_block@.contains_key(h)
                                || pre_reg.request_residency@[rid].block_ids@
                                    .subrange(
                                        pre_reg.request_residency@[rid]
                                            .cached_prefix_blocks as int,
                                        row_end as int
                                            / (BLOCK_SIZE_SPEC as int),
                                    ).contains(b));
                            if pre_reg.hash_to_block@.contains_key(h) {
                                assert(pre_reg.hash_to_block@[h] == b);
                            } else {
                                assert(pre_reg.request_residency@[rid].block_ids@
                                    .contains(b));
                                let q = ids.subrange(c, ids.len() as int)
                                    .index_of(b);
                                let l = c + q;
                                assert(ids[l] == b);
                                assert(pre_reg.blocks@[b].refcount == 1);
                                lemma_refcount_one_excludes_other_holder(
                                    &pre_reg, b, other, rid,
                                );
                                assert(false);
                            }
                        }
                    }
                    assert(token_placement_prefix(
                        self.blocks@, ids,
                        self.live_requests@[other].prompt_tokens@,
                        self.live_requests@[other].prompt_tokens@.len() as int,
                    )) by {
                        lemma_token_placement_prefix_transfer(
                            pre_reg.blocks@, self.blocks@, ids,
                            pre_reg.live_requests@[other].prompt_tokens@,
                            pre_reg.live_requests@[other].prompt_tokens@.len() as int,
                        );
                    }
                }
            }
            i += 1;
        }
    }

    // Public composition used outside the pressure-aware planner: discover one
    // valid cached prefix and immediately allocate from that certified match.
    #[verifier::spinoff_prover]
    #[verifier::rlimit(200)]
    fn activate_cached_block(&mut self, bid: BlockId)
        requires
            free_queue_valid_token(old(self)),
            old(self).blocks@.contains_key(bid),
            bid < old(self).num_blocks,
            old(self).blocks@[bid].refcount < u64::MAX,
            old(self).blocks@[bid].prefix_depth > 0,
            old(self).blocks@[bid].refcount == 0
                && old(self).blocks@[bid].parent_block is Some ==>
                !old(self).cached_queue.order@.contains(
                    old(self).blocks@[bid].parent_block.unwrap(),
                ),
        ensures
            free_queue_valid_token(final(self)),
            final(self).config == old(self).config,
            final(self).num_blocks == old(self).num_blocks,
            final(self).free_blocks == old(self).free_blocks,
            final(self).running@ == old(self).running@,
            final(self).waiting@ == old(self).waiting@,
            final(self).live_requests@ == old(self).live_requests@,
            final(self).accepted_requests@ == old(self).accepted_requests@,
            final(self).request_residency@ == old(self).request_residency@,
            final(self).hash_to_block@ == old(self).hash_to_block@,
            final(self).blocks@.dom() == old(self).blocks@.dom(),
            final(self).blocks@[bid].tokens@ == old(self).blocks@[bid].tokens@,
            final(self).blocks@[bid].refcount as int
                == old(self).blocks@[bid].refcount as int + 1,
            final(self).blocks@[bid].hash_value == old(self).blocks@[bid].hash_value,
            final(self).blocks@[bid].prefix_depth == old(self).blocks@[bid].prefix_depth,
            final(self).blocks@[bid].parent_block == old(self).blocks@[bid].parent_block,
            forall|other: BlockId| other != bid
                && #[trigger] old(self).blocks@.contains_key(other)
                ==> final(self).blocks@[other] == old(self).blocks@[other],
            final(self).free_queue.order@ == old(self).free_queue.order@,
            old(self).blocks@[bid].refcount == 0 ==>
                !final(self).cached_queue.order@.contains(bid),
            old(self).blocks@[bid].refcount > 0 ==>
                final(self).cached_queue.order@ == old(self).cached_queue.order@,
    {
        proof { lemma_free_queue_token_to_valid(old(self)); }
        let ghost before = *self;
        let entry = match self.blocks.get(&bid) {
            Some(e) => e.clone(),
            None => {
                proof { assert(false); }
                return;
            },
        };
        assert(entry.refcount == before.blocks@[bid].refcount);
        if entry.refcount == 0 {
            assert(zero_ref_cached_page(&before, bid));
            assert(before.cached_queue.order@.contains(bid));
            let removed = self.cached_queue.remove(bid);
            assert(removed);
            assert(!self.cached_queue.order@.contains(bid)) by {
                if self.cached_queue.order@.contains(bid) {
                    assert(self.cached_queue.order@.no_duplicates());
                }
            }
        } else {
            assert(!before.cached_queue.order@.contains(bid)) by {
                if before.cached_queue.order@.contains(bid) {
                    assert(free_queue_membership_valid(&before));
                }
            }
        }
        let ghost blocks_before_insert = self.blocks@;
        self.blocks.insert(bid, BlockEntry {
            tokens: entry.tokens,
            refcount: entry.refcount + 1,
            hash_value: entry.hash_value,
            prefix_depth: entry.prefix_depth,
            parent_block: entry.parent_block,
        });
        proof {
            vstd::map::lemma_map_insert_domain(
                blocks_before_insert, bid, self.blocks@[bid],
            );
            assert(self.blocks@.dom() =~= blocks_before_insert.dom());
            assert(blocks_before_insert.dom() == before.blocks@.dom());
            assert forall|other: BlockId| other != bid
                && #[trigger] before.blocks@.contains_key(other)
                implies self.blocks@[other] == before.blocks@[other]
            by {
                assert(blocks_before_insert.contains_key(other));
                assert(self.blocks@[other] == blocks_before_insert[other]);
            }
            lemma_two_queue_after_block_activation(&before, self, bid);
            lemma_free_queue_valid_to_token(self);
        }
    }

    #[verifier::spinoff_prover]
    #[verifier::rlimit(10)]
    pub fn allocate_prefill_with_reuse(
        &mut self,
        rid: RequestId,
        prompt_tokens: &Vec<TokenId>,
        excluded: &Vec<u64>,
    ) -> (admitted: bool)
        requires
            cs_valid(old(self)),
            free_queue_valid(old(self)),
            persistent_provenance_closed(old(self)),
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
            final(self).hash_to_block@ == old(self).hash_to_block@,
            admitted ==> final(self).request_residency@.contains_key(rid),
            admitted ==> final(self).request_residency@.dom()
                == old(self).request_residency@.dom().insert(rid),
            admitted ==> prompt_tokens@.len() > 0,
            admitted ==> allocate_prefill_reuse_success(old(self), final(self), rid,
                prompt_tokens@, final(self).request_residency@[rid].cached_prefix_blocks as int),
            admitted ==> reused_prefix_excludes(
                old(self), final(self), rid, excluded@,
                final(self).request_residency@[rid].cached_prefix_blocks as int,
            ),
            !admitted ==> final(self).request_residency@ == old(self).request_residency@
                          && final(self).blocks@ == old(self).blocks@
                          && final(self).free_blocks == old(self).free_blocks,
            forall|bid: BlockId| #[trigger] old(self).blocks@.contains_key(bid)
                ==> final(self).blocks@.contains_key(bid)
                    && final(self).blocks@[bid].tokens@ == old(self).blocks@[bid].tokens@,
            forall|bid: BlockId| #[trigger] old(self).blocks@.contains_key(bid)
                ==> (final(self).blocks@[bid].refcount == old(self).blocks@[bid].refcount
                    || final(self).blocks@[bid].tokens@.len() == BLOCK_SIZE_SPEC as int),
            forall|bid: BlockId| #[trigger] old(self).blocks@.contains_key(bid)
                ==> final(self).blocks@[bid].hash_value == old(self).blocks@[bid].hash_value,
            forall|bid: BlockId| #[trigger] old(self).blocks@.contains_key(bid)
                ==> final(self).blocks@[bid].prefix_depth == old(self).blocks@[bid].prefix_depth
                    && final(self).blocks@[bid].parent_block == old(self).blocks@[bid].parent_block,
            forall|bid: BlockId| #[trigger] old(self).blocks@.contains_key(bid)
                ==> (final(self).blocks@[bid].refcount == old(self).blocks@[bid].refcount
                    || (old(self).blocks@[bid].hash_value != 0
                        && !excluded@.contains(old(self).blocks@[bid].hash_value))),
            forall|other: RequestId|
                other != rid && #[trigger] old(self).request_residency@.contains_key(other)
                ==> final(self).request_residency@.contains_key(other)
                    && final(self).request_residency@[other] == old(self).request_residency@[other],
            persistent_provenance_closed(old(self))
                ==> persistent_provenance_closed(final(self)),
            positive_provenance_origin(old(self), final(self)),
    {
        let matched = self.match_cached_prefix(prompt_tokens, excluded);
        self.allocate_prefill_from_match(rid, prompt_tokens, excluded, matched)
    }

    #[verifier::spinoff_prover]
    #[verifier::rlimit(200)]
    fn activate_cached_blocks(
        &mut self,
        matched: &Vec<BlockId>,
    )
        requires
            free_queue_valid_token(old(self)),
            matched@.no_duplicates(),
            registered_prefix_chain(old(self).blocks@, matched@),
            forall|k: int| 0 <= k < matched@.len()
                ==> old(self).blocks@.contains_key(#[trigger] matched@[k])
                    && matched@[k] < old(self).num_blocks
                    && old(self).blocks@[matched@[k]].refcount < u64::MAX,
        ensures
            free_queue_valid_token(final(self)),
            final(self).config == old(self).config,
            final(self).num_blocks == old(self).num_blocks,
            final(self).free_blocks == old(self).free_blocks,
            final(self).running@ == old(self).running@,
            final(self).waiting@ == old(self).waiting@,
            final(self).live_requests@ == old(self).live_requests@,
            final(self).accepted_requests@ == old(self).accepted_requests@,
            final(self).request_residency@ == old(self).request_residency@,
            final(self).hash_to_block@ == old(self).hash_to_block@,
            final(self).blocks@.dom() == old(self).blocks@.dom(),
            forall|k: int| 0 <= k < matched@.len()
                ==> final(self).blocks@.contains_key(#[trigger] matched@[k])
                    && final(self).blocks@[matched@[k]].tokens@
                        == old(self).blocks@[matched@[k]].tokens@
                    && final(self).blocks@[matched@[k]].hash_value
                        == old(self).blocks@[matched@[k]].hash_value
                    && final(self).blocks@[matched@[k]].prefix_depth
                        == old(self).blocks@[matched@[k]].prefix_depth
                    && final(self).blocks@[matched@[k]].parent_block
                        == old(self).blocks@[matched@[k]].parent_block
                    && final(self).blocks@[matched@[k]].refcount as int
                        == old(self).blocks@[matched@[k]].refcount as int + 1,
            forall|bid: BlockId|
                #[trigger] old(self).blocks@.contains_key(bid)
                && !matched@.contains(bid)
                ==> final(self).blocks@.contains_key(bid)
                    && final(self).blocks@[bid] == old(self).blocks@[bid],
    {
        hide(Seq::no_duplicates);
        let ghost before = *self;
        let mut i: usize = 0;
        while i < matched.len()
            invariant
                i <= matched@.len(),
                matched@.no_duplicates(),
                registered_prefix_chain(before.blocks@, matched@),
                free_queue_valid_token(self),
                forall|k: int| 0 <= k < matched@.len()
                    ==> before.blocks@.contains_key(#[trigger] matched@[k])
                        && matched@[k] < before.num_blocks
                        && before.blocks@[matched@[k]].refcount < u64::MAX,
                self.blocks@.dom() == before.blocks@.dom(),
                forall|k: int| 0 <= k < i as int
                    ==> self.blocks@.contains_key(#[trigger] matched@[k])
                        && self.blocks@[matched@[k]].tokens@
                            == before.blocks@[matched@[k]].tokens@
                        && self.blocks@[matched@[k]].hash_value
                            == before.blocks@[matched@[k]].hash_value
                        && self.blocks@[matched@[k]].prefix_depth
                            == before.blocks@[matched@[k]].prefix_depth
                        && self.blocks@[matched@[k]].parent_block
                            == before.blocks@[matched@[k]].parent_block
                        && self.blocks@[matched@[k]].refcount as int
                            == before.blocks@[matched@[k]].refcount as int + 1,
                forall|bid: BlockId|
                    #[trigger] before.blocks@.contains_key(bid)
                    && !matched@.subrange(0, i as int).contains(bid)
                    ==> self.blocks@.contains_key(bid)
                        && self.blocks@[bid] == before.blocks@[bid],
                self.config == before.config,
                self.num_blocks == before.num_blocks,
                self.free_blocks == before.free_blocks,
                self.running@ == before.running@,
                self.waiting@ == before.waiting@,
                self.live_requests@ == before.live_requests@,
                self.accepted_requests@ == before.accepted_requests@,
                self.request_residency@ == before.request_residency@,
                self.hash_to_block@ == before.hash_to_block@,
            decreases matched@.len() - i,
        {
            let bid = matched[i];
            proof {
                assert(!matched@.subrange(0, i as int).contains(bid)) by {
                    if matched@.subrange(0, i as int).contains(bid) {
                        let q = choose|q: int| 0 <= q < i as int
                            && matched@.subrange(0, i as int)[q] == bid;
                        assert(matched@[q] == matched@[i as int]);
                        lemma_no_duplicates_distinct(matched@, q, i as int);
                    }
                }
                assert(self.blocks@[bid] == before.blocks@[bid]);
                if self.blocks@[bid].parent_block is Some {
                    assert(i > 0) by {
                        assert(registered_prefix_chain(before.blocks@, matched@));
                    }
                    let parent_bid = matched@[i as int - 1];
                    assert(self.blocks@[bid].parent_block == Some(parent_bid)) by {
                        assert(registered_prefix_chain(before.blocks@, matched@));
                    }
                    assert(self.blocks@.contains_key(parent_bid));
                    assert(self.blocks@[parent_bid].refcount as int
                        == before.blocks@[parent_bid].refcount as int + 1);
                    assert(self.blocks@[parent_bid].refcount > 0);
                    lemma_free_queue_token_positive_block_not_cached(self, parent_bid);
                }
            }
            let ghost before_activation = *self;
            self.activate_cached_block(bid);
            proof {
                assert(self.blocks@.dom() =~= before_activation.blocks@.dom());
                assert forall|k: int| 0 <= k < i as int + 1
                    implies self.blocks@.contains_key(#[trigger] matched@[k])
                        && self.blocks@[matched@[k]].tokens@
                            == before.blocks@[matched@[k]].tokens@
                        && self.blocks@[matched@[k]].hash_value
                            == before.blocks@[matched@[k]].hash_value
                        && self.blocks@[matched@[k]].prefix_depth
                            == before.blocks@[matched@[k]].prefix_depth
                        && self.blocks@[matched@[k]].parent_block
                            == before.blocks@[matched@[k]].parent_block
                        && self.blocks@[matched@[k]].refcount as int
                            == before.blocks@[matched@[k]].refcount as int + 1
                by {
                    if k != i as int {
                        lemma_no_duplicates_distinct(matched@, k, i as int);
                        assert(self.blocks@[matched@[k]]
                            == before_activation.blocks@[matched@[k]]);
                    }
                }
                assert forall|other: BlockId|
                    #[trigger] before.blocks@.contains_key(other)
                    && !matched@.subrange(0, i as int + 1).contains(other)
                    implies self.blocks@.contains_key(other)
                        && self.blocks@[other] == before.blocks@[other]
                by {
                    assert(other != bid) by {
                        assert(matched@.subrange(0, i as int + 1)[i as int] == bid);
                    }
                    assert(!matched@.subrange(0, i as int).contains(other)) by {
                        if matched@.subrange(0, i as int).contains(other) {
                            let q = choose|q: int| 0 <= q < i as int
                                && matched@.subrange(0, i as int)[q] == other;
                            assert(matched@.subrange(0, i as int + 1)[q] == other);
                        }
                    }
                    assert(self.blocks@[other] == before_activation.blocks@[other]);
                }
            }
            i += 1;
        }
        proof {
            assert(matched@.subrange(0, matched@.len() as int) =~= matched@);
        }
    }

    fn matched_refcounts_have_headroom(
        &self,
        matched: &Vec<BlockId>,
    ) -> (out: bool)
        requires
            forall|k: int| 0 <= k < matched@.len()
                ==> self.blocks@.contains_key(#[trigger] matched@[k]),
        ensures
            out ==> forall|k: int| 0 <= k < matched@.len()
                ==> (#[trigger] self.blocks@[matched@[k]].refcount) < u64::MAX,
    {
        let c: usize = matched.len();
        let mut bi: usize = 0;
        while bi < c
            invariant
                bi <= c,
                c == matched@.len(),
                forall|k: int| 0 <= k < matched@.len()
                    ==> self.blocks@.contains_key(#[trigger] matched@[k]),
                forall|k: int| 0 <= k < bi as int
                    ==> (#[trigger] self.blocks@[matched@[k]].refcount) < u64::MAX,
            decreases c - bi,
        {
            let bid = matched[bi];
            let rc = match self.blocks.get(&bid) {
                Some(e) => e.refcount,
                None => {
                    proof { assert(false); }
                    0
                },
            };
            if rc == u64::MAX {
                return false;
            }
            bi += 1;
        }
        true
    }

    // Allocate from a verified cached-prefix match. Only the suffix goes through
    // `allocate_prefill` (the physical slots coincide because offsets agree mod
    // BLOCK_SIZE and the fresh blocks sit at table indices c..), then stitch
    // the residency (matched ++ fresh, cached_prefix_blocks = c) and bump
    // the matched blocks' refcounts. This is the allocation path used by
    // `plan`; it completes all reuse decisions before publishing any
    // newly admitted prompt hashes, so same-forward K/V cannot be reused.
    #[verifier::spinoff_prover]
    #[verifier::rlimit(300)]
    pub fn allocate_prefill_from_match(
        &mut self,
        rid: RequestId,
        prompt_tokens: &Vec<TokenId>,
        excluded: &Vec<u64>,
        matched: Vec<BlockId>,
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
            matched@.no_duplicates(),
            prompt_tokens@.len() == 0 ==> matched@.len() == 0,
            prompt_tokens@.len() > 0 ==> matched@.len() as int
                * (BLOCK_SIZE_SPEC as int) < prompt_tokens@.len() as int,
            forall|j: int| 0 <= j < matched@.len()
                ==> old(self).blocks@.contains_key(#[trigger] matched@[j])
                    && old(self).blocks@[matched@[j]].tokens@.len()
                        == BLOCK_SIZE_SPEC as int,
            forall|j: int| 0 <= j < matched@.len()
                ==> (#[trigger] old(self).blocks@[matched@[j]].hash_value) != 0
                    && !excluded@.contains(old(self).blocks@[matched@[j]].hash_value),
            forall|j: int| 0 <= j < matched@.len()
                ==> old(self).hash_to_block@.contains_key(
                        #[trigger] old(self).blocks@[matched@[j]].hash_value)
                    && old(self).hash_to_block@[
                        old(self).blocks@[matched@[j]].hash_value] == matched@[j],
            registered_prefix_chain(old(self).blocks@, matched@),
            token_placement_prefix(old(self).blocks@, matched@, prompt_tokens@,
                matched@.len() as int * (BLOCK_SIZE_SPEC as int)),
        ensures
            cs_valid(final(self)),
            free_queue_valid(final(self)),
            final(self).config == old(self).config,
            final(self).num_blocks == old(self).num_blocks,
            final(self).live_requests@ == old(self).live_requests@,
            final(self).accepted_requests@ == old(self).accepted_requests@,
            final(self).running@ == old(self).running@,
            final(self).waiting@ == old(self).waiting@,
            final(self).hash_to_block@ == old(self).hash_to_block@,
            admitted ==> final(self).request_residency@.contains_key(rid),
            admitted ==> final(self).request_residency@[rid].cached_prefix_blocks as int
                == matched@.len(),
            admitted ==> final(self).request_residency@.dom()
                == old(self).request_residency@.dom().insert(rid),
            admitted ==> prompt_tokens@.len() > 0,
            admitted ==> allocate_prefill_reuse_success(old(self), final(self), rid,
                prompt_tokens@, final(self).request_residency@[rid].cached_prefix_blocks as int),
            admitted ==> reused_prefix_excludes(
                old(self), final(self), rid, excluded@,
                final(self).request_residency@[rid].cached_prefix_blocks as int,
            ),
            !admitted ==> final(self).request_residency@ == old(self).request_residency@
                          && final(self).blocks@ == old(self).blocks@
                          && final(self).free_blocks == old(self).free_blocks,
            // Pre-existing blocks keep their TOKENS (reuse bumps refcounts
            // and allocation only adds fresh blocks).
            forall|bid: BlockId| #[trigger] old(self).blocks@.contains_key(bid)
                ==> final(self).blocks@.contains_key(bid)
                    && final(self).blocks@[bid].tokens@ == old(self).blocks@[bid].tokens@,
            // Refcounts of pre-existing blocks change only on FULL blocks
            // (the matched prefix): feeds the tail-exclusivity clause of
            // `residency_history_aligned`.
            forall|bid: BlockId| #[trigger] old(self).blocks@.contains_key(bid)
                ==> (final(self).blocks@[bid].refcount == old(self).blocks@[bid].refcount
                    || final(self).blocks@[bid].tokens@.len() == BLOCK_SIZE_SPEC as int),
            // Pre-existing blocks keep their HASH verbatim, and a
            // refcount bump only hits registry targets whose stored hash is
            // nonzero and outside the exclusion list (the matched prefix).
            forall|bid: BlockId| #[trigger] old(self).blocks@.contains_key(bid)
                ==> final(self).blocks@[bid].hash_value == old(self).blocks@[bid].hash_value,
            forall|bid: BlockId| #[trigger] old(self).blocks@.contains_key(bid)
                ==> final(self).blocks@[bid].prefix_depth == old(self).blocks@[bid].prefix_depth
                    && final(self).blocks@[bid].parent_block == old(self).blocks@[bid].parent_block,
            forall|bid: BlockId| #[trigger] old(self).blocks@.contains_key(bid)
                ==> (final(self).blocks@[bid].refcount == old(self).blocks@[bid].refcount
                    || (old(self).blocks@[bid].hash_value != 0
                        && !excluded@.contains(old(self).blocks@[bid].hash_value))),
            forall|other: RequestId|
                other != rid && #[trigger] old(self).request_residency@.contains_key(other)
                ==> final(self).request_residency@.contains_key(other)
                    && final(self).request_residency@[other] == old(self).request_residency@[other],
            persistent_provenance_closed(old(self))
                ==> persistent_provenance_closed(final(self)),
            positive_provenance_origin(old(self), final(self)),
    {
        hide(Seq::no_duplicates);
        hide(free_queue_valid);
        proof { lemma_free_queue_valid_to_token(old(self)); }
        let c: usize = matched.len();
        let ghost bs = BLOCK_SIZE_SPEC as int;
        if c == 0 {
            let admitted = self.allocate_prefill(rid, prompt_tokens);
            proof {
                if admitted {
                    // Bridge the fresh-only success shape to the reuse shape
                    // at c == 0.
                    let n = prompt_tokens@.len() as int;
                    let ids = self.request_residency@[rid].block_ids@;
                    assert(self.request_residency@[rid].cached_prefix_blocks as int == 0);
                    assert(ids.len() == blocks_needed_for(n as nat) as int);
                    assert forall|i: int|
                        #![trigger self.request_residency@[rid].slot_mapping@[i]]
                        0 <= i < n - 0 * bs
                        implies self.request_residency@[rid].slot_mapping@[i] as int
                            == block_table_slot(ids, (0 * bs + i) as nat) as int
                    by {
                        assert(self.request_residency@[rid].slot_mapping@[i] as int
                            == block_table_slot(ids, i as nat) as int);
                    }
                    assert forall|k: int| #![trigger ids[k]] 0 <= k < ids.len()
                        implies !old(self).blocks@.contains_key(ids[k])
                            && self.blocks@.contains_key(ids[k])
                            && self.blocks@[ids[k]].refcount == 1
                            && self.blocks@[ids[k]].hash_value == 0
                            && self.blocks@[ids[k]].prefix_depth == 0
                            && self.blocks@[ids[k]].parent_block == Option::<BlockId>::None
                    by {
                        assert(allocate_prefill_success(old(self), self, rid,
                            prompt_tokens@));
                        assert(!old(self).blocks@.contains_key(ids[k]));
                        assert(self.blocks@.contains_key(ids[k]));
                        assert(self.blocks@[ids[k]].refcount == 1);
                        assert(self.blocks@[ids[k]].hash_value == 0);
                        assert(self.blocks@[self.request_residency@[rid]
                            .block_ids@[k]].prefix_depth == 0);
                        assert(self.blocks@[self.request_residency@[rid]
                            .block_ids@[k]].parent_block == Option::<BlockId>::None);
                    }
                    assert(ids.subrange(0, 0) =~= Seq::<BlockId>::empty());
                    assert(registered_prefix_chain(self.blocks@, ids.subrange(0, 0)));
                    assert forall|k: int| #![trigger ids[k]] 0 <= k < 0
                        implies old(self).hash_to_block@.contains_key(
                                old(self).blocks@[ids[k]].hash_value)
                            && old(self).hash_to_block@[
                                old(self).blocks@[ids[k]].hash_value] == ids[k]
                    by {}
                    assert(allocate_prefill_reuse_success(old(self), self, rid,
                        prompt_tokens@, 0));
                    reveal(reused_prefix_excludes);
                    assert(reused_prefix_excludes(
                        old(self), self, rid, excluded@, 0,
                    ));
                }
            }
            return admitted;
        }
        // Refcount headroom: refuse the (absurd) saturated case up front,
        // before any mutation.
        if !self.matched_refcounts_have_headroom(&matched) {
            proof {
                lemma_positive_provenance_origin_blocks_eq(old(self), self);
            }
            return false;
        }
        // Suffix slice: positions [c*BLOCK_SIZE, n).
        let n: usize = prompt_tokens.len();
        proof {
            assert(c as int * bs < n as int);
        }
        let sstart: usize = c * (BLOCK_SIZE as usize);
        let mut suffix: Vec<TokenId> = Vec::new();
        let mut t: usize = sstart;
        while t < n
            invariant
                sstart <= t <= n,
                n as int == prompt_tokens@.len(),
                suffix@.len() == t as int - sstart as int,
                forall|i: int| 0 <= i < suffix@.len()
                    ==> #[trigger] suffix@[i] == prompt_tokens@[sstart as int + i],
            decreases n - t,
        {
            suffix.push(prompt_tokens[t]);
            t += 1;
        }
        proof {
            assert(suffix@.len() == n as int - c as int * bs);
            assert(suffix@.len() > 0);
        }
        let ghost pre_state = *self;
        let admitted = self.allocate_prefill(rid, &suffix);
        if !admitted {
            return false;
        }
        let ghost post_alloc = *self;
        // Stitch the residency: matched prefix ++ fresh suffix blocks, with
        // the suffix slot_mapping kept verbatim (identical physical slots).
        let (fresh_ids, slots) = match self.request_residency.get(&rid) {
            Some(r) => (r.block_ids.clone(), r.slot_mapping.clone()),
            None => {
                proof { assert(false); }
                (Vec::new(), Vec::new())
            },
        };
        let matched_snapshot = matched.clone();
        let mut full_ids = matched;
        let mut fi: usize = 0;
        while fi < fresh_ids.len()
            invariant
                fi <= fresh_ids@.len(),
                full_ids@ == matched_snapshot@ + fresh_ids@.subrange(0, fi as int),
                matched_snapshot@.len() == c as int,
            decreases fresh_ids@.len() - fi,
        {
            full_ids.push(fresh_ids[fi]);
            proof {
                assert(full_ids@ =~= matched_snapshot@ + fresh_ids@.subrange(0, fi as int + 1));
            }
            fi += 1;
        }
        proof {
            assert(fresh_ids@.subrange(0, fresh_ids@.len() as int) =~= fresh_ids@);
            assert(full_ids@ == matched_snapshot@ + fresh_ids@);
        }
        let ghost resid_before = self.request_residency@;
        self.request_residency.insert(rid, RequestResidency {
            block_ids: full_ids,
            cached_prefix_blocks: c as u64,
            slot_mapping: slots,
        });
        proof {
            vstd::map::lemma_map_insert_domain(resid_before, rid, self.request_residency@[rid]);
            assert(self.request_residency@.dom() =~= resid_before.dom());
            lemma_free_queue_valid_to_token(&post_alloc);
            lemma_free_queue_valid_token_frame(&post_alloc, self);
        }
        // Bump the matched blocks in an isolated transition so the caller's
        // semantic reconstruction does not carry the loop's quantified state.
        let ghost bump_pre = *self;
        proof {
            assert(bump_pre.blocks@ == post_alloc.blocks@);
            assert(bump_pre.num_blocks == post_alloc.num_blocks);
            assert(registered_prefix_chain(
                bump_pre.blocks@, matched_snapshot@,
            )) by {
                assert forall|j: int|
                    #![trigger bump_pre.blocks@[matched_snapshot@[j]].prefix_depth]
                    0 <= j < matched_snapshot@.len() implies {
                    let bid = matched_snapshot@[j];
                    &&& bump_pre.blocks@.contains_key(bid)
                    &&& bump_pre.blocks@[bid].prefix_depth as int == j + 1
                    &&& bump_pre.blocks@[bid].parent_block
                        == if j == 0 { None } else { Some(matched_snapshot@[j - 1]) }
                } by {
                    assert(registered_prefix_chain(
                        pre_state.blocks@, matched_snapshot@,
                    ));
                    assert(post_alloc.blocks@[matched_snapshot@[j]]
                        == pre_state.blocks@[matched_snapshot@[j]]);
                }
            }
            assert forall|k: int| 0 <= k < matched_snapshot@.len()
                implies bump_pre.blocks@.contains_key(#[trigger] matched_snapshot@[k])
                    && matched_snapshot@[k] < bump_pre.num_blocks
                    && bump_pre.blocks@[matched_snapshot@[k]].refcount < u64::MAX
            by {
                assert(pre_state.blocks@.contains_key(matched_snapshot@[k]));
                assert(post_alloc.blocks@[matched_snapshot@[k]]
                    == pre_state.blocks@[matched_snapshot@[k]]);
                assert(blocks_dom_in_range(&post_alloc));
            }
        }
        self.activate_cached_blocks(&matched_snapshot);
        proof {
            assert(self.blocks@.dom() == post_alloc.blocks@.dom());
            assert forall|k: int| 0 <= k < matched_snapshot@.len()
                implies self.blocks@.contains_key(#[trigger] matched_snapshot@[k])
                    && self.blocks@[matched_snapshot@[k]].tokens@
                        == pre_state.blocks@[matched_snapshot@[k]].tokens@
                    && self.blocks@[matched_snapshot@[k]].hash_value
                        == pre_state.blocks@[matched_snapshot@[k]].hash_value
                    && self.blocks@[matched_snapshot@[k]].prefix_depth
                        == pre_state.blocks@[matched_snapshot@[k]].prefix_depth
                    && self.blocks@[matched_snapshot@[k]].parent_block
                        == pre_state.blocks@[matched_snapshot@[k]].parent_block
                    && self.blocks@[matched_snapshot@[k]].refcount as int
                        == pre_state.blocks@[matched_snapshot@[k]].refcount as int + 1
            by {
                assert(bump_pre.blocks@[matched_snapshot@[k]]
                    == post_alloc.blocks@[matched_snapshot@[k]]);
                assert(post_alloc.blocks@[matched_snapshot@[k]]
                    == pre_state.blocks@[matched_snapshot@[k]]);
            }
            assert forall|bid: BlockId|
                #[trigger] post_alloc.blocks@.contains_key(bid)
                && !matched_snapshot@.contains(bid)
                implies self.blocks@.contains_key(bid)
                    && self.blocks@[bid] == post_alloc.blocks@[bid]
            by {
                assert(bump_pre.blocks@.contains_key(bid));
                assert(self.blocks@[bid] == bump_pre.blocks@[bid]);
            }
            assert(self.config == post_alloc.config);
            assert(self.num_blocks == post_alloc.num_blocks);
            assert(self.free_blocks == post_alloc.free_blocks);
            assert(self.running@ == post_alloc.running@);
            assert(self.waiting@ == post_alloc.waiting@);
            assert(self.live_requests@ == post_alloc.live_requests@);
            assert(self.accepted_requests@ == post_alloc.accepted_requests@);
            assert(self.hash_to_block@ == post_alloc.hash_to_block@);
            assert(self.request_residency@ == bump_pre.request_residency@);
            assert(matched_snapshot@.subrange(
                0, matched_snapshot@.len() as int,
            ) =~= matched_snapshot@);
        }
        proof {
            let n_i = n as int;
            let c_i = c as int;
            let ids = self.request_residency@[rid].block_ids@;
            assert(pre_state == *old(self));
            assert(ids == matched_snapshot@ + fresh_ids@);
            assert(resid_before == post_alloc.request_residency@);
            assert(fresh_ids@ == post_alloc.request_residency@[rid].block_ids@);
            assert(slots@ == post_alloc.request_residency@[rid].slot_mapping@);
            let suffix_len = n_i - c_i * bs;
            assert(suffix@.len() == suffix_len);
            assert(allocate_prefill_success(&pre_state, &post_alloc, rid, suffix@));
            assert(fresh_ids@.len() == blocks_needed_for(suffix@.len() as nat) as int);
            assert(ids.len() == c_i + fresh_ids@.len());

            // ---- Per-id facts: matched prefix and fresh suffix. ----
            assert forall|k: int| 0 <= k < c_i implies
                pre_state.blocks@.contains_key(#[trigger] ids[k])
                && self.blocks@.contains_key(ids[k])
                && self.blocks@[ids[k]].tokens@ == pre_state.blocks@[ids[k]].tokens@
                && self.blocks@[ids[k]].tokens@.len() == bs
                && self.blocks@[ids[k]].hash_value == pre_state.blocks@[ids[k]].hash_value
                && self.blocks@[ids[k]].prefix_depth
                    == pre_state.blocks@[ids[k]].prefix_depth
                && self.blocks@[ids[k]].parent_block
                    == pre_state.blocks@[ids[k]].parent_block
                && self.blocks@[ids[k]].refcount as int
                    == pre_state.blocks@[ids[k]].refcount as int + 1
            by {
                assert(ids[k] == matched_snapshot@[k]);
            }
            assert forall|k: int| c_i <= k < ids.len() implies
                !pre_state.blocks@.contains_key(#[trigger] ids[k])
                && post_alloc.blocks@.contains_key(ids[k])
                && post_alloc.blocks@[ids[k]].refcount == 1
                && post_alloc.blocks@[ids[k]].hash_value == 0
                && post_alloc.blocks@[ids[k]].prefix_depth == 0
                && post_alloc.blocks@[ids[k]].parent_block == Option::<BlockId>::None
                && !matched_snapshot@.contains(ids[k])
                && self.blocks@.contains_key(ids[k])
                && self.blocks@[ids[k]] == post_alloc.blocks@[ids[k]]
            by {
                assert(ids[k] == fresh_ids@[k - c_i]);
                assert(!pre_state.blocks@.contains_key(fresh_ids@[k - c_i]));
                assert(post_alloc.blocks@.contains_key(fresh_ids@[k - c_i]));
                assert(post_alloc.blocks@[fresh_ids@[k - c_i]].refcount == 1);
                assert(post_alloc.blocks@[fresh_ids@[k - c_i]].prefix_depth == 0);
                assert(post_alloc.blocks@[fresh_ids@[k - c_i]].parent_block
                    == Option::<BlockId>::None);
                assert(!matched_snapshot@.contains(ids[k])) by {
                    if matched_snapshot@.contains(ids[k]) {
                        let q = matched_snapshot@.index_of(ids[k]);
                        assert(pre_state.blocks@.contains_key(matched_snapshot@[q]));
                    }
                }
            }

            // ---- Refcount evolution on pre-existing blocks: matched
            // (full) blocks bumped, everything else verbatim. ----
            assert forall|bid: BlockId| #[trigger] old(self).blocks@.contains_key(bid)
                implies (self.blocks@[bid].refcount == old(self).blocks@[bid].refcount
                    || self.blocks@[bid].tokens@.len() == BLOCK_SIZE_SPEC as int)
            by {
                if matched_snapshot@.contains(bid) {
                    let q = matched_snapshot@.index_of(bid);
                    assert(ids[q] == matched_snapshot@[q]);
                    assert(self.blocks@[ids[q]].tokens@.len() == bs);
                } else {
                    assert(!matched_snapshot@.subrange(0, c_i).contains(bid));
                    assert(self.blocks@[bid] == post_alloc.blocks@[bid]);
                    assert(post_alloc.blocks@[bid] == pre_state.blocks@[bid]);
                }
            }
            // ---- Hash preservation + bump characterization. ----
            assert forall|bid: BlockId| #[trigger] old(self).blocks@.contains_key(bid)
                implies self.blocks@[bid].hash_value
                    == old(self).blocks@[bid].hash_value
            by {
                if matched_snapshot@.contains(bid) {
                    let q = matched_snapshot@.index_of(bid);
                    assert(ids[q] == matched_snapshot@[q]);
                    assert(self.blocks@[matched_snapshot@[q]].hash_value
                        == pre_state.blocks@[matched_snapshot@[q]].hash_value);
                } else {
                    assert(!matched_snapshot@.subrange(0, c_i).contains(bid));
                    assert(self.blocks@[bid] == post_alloc.blocks@[bid]);
                    assert(post_alloc.blocks@[bid] == pre_state.blocks@[bid]);
                }
            }
            assert forall|bid: BlockId| #[trigger] old(self).blocks@.contains_key(bid)
                implies self.blocks@[bid].prefix_depth
                        == old(self).blocks@[bid].prefix_depth
                    && self.blocks@[bid].parent_block
                        == old(self).blocks@[bid].parent_block
            by {
                if matched_snapshot@.contains(bid) {
                    let q = matched_snapshot@.index_of(bid);
                    assert(ids[q] == matched_snapshot@[q]);
                } else {
                    assert(!matched_snapshot@.subrange(0, c_i).contains(bid));
                    assert(self.blocks@[bid] == post_alloc.blocks@[bid]);
                    assert(post_alloc.blocks@[bid] == pre_state.blocks@[bid]);
                }
            }
            assert forall|bid: BlockId| #[trigger] old(self).blocks@.contains_key(bid)
                implies (self.blocks@[bid].refcount == old(self).blocks@[bid].refcount
                    || (old(self).blocks@[bid].hash_value != 0
                        && !excluded@.contains(old(self).blocks@[bid].hash_value)))
            by {
                if matched_snapshot@.contains(bid) {
                    let q = matched_snapshot@.index_of(bid);
                    assert(matched_snapshot@[q] == bid);
                    assert(pre_state.blocks@[bid].hash_value != 0
                        && !excluded@.contains(pre_state.blocks@[bid].hash_value));
                } else {
                    assert(!matched_snapshot@.subrange(0, c_i).contains(bid));
                    assert(self.blocks@[bid] == post_alloc.blocks@[bid]);
                    assert(post_alloc.blocks@[bid] == pre_state.blocks@[bid]);
                }
            }

            // ---- No duplicates across the stitched table. ----
            lemma_stitched_block_ids_no_duplicates(
                matched_snapshot@, fresh_ids@, pre_state.blocks@.dom(),
            );
            assert(ids.no_duplicates());

            // ---- Suffix slots sit at the GLOBAL block-table positions. ----
            assert forall|i: int| #![trigger self.request_residency@[rid].slot_mapping@[i]]
                0 <= i < n_i - c_i * bs
                implies self.request_residency@[rid].slot_mapping@[i] as int
                    == block_table_slot(ids, (c_i * bs + i) as nat) as int
            by {
                assert(self.request_residency@[rid].slot_mapping@[i] == slots@[i]);
                assert(slots@[i] as int
                    == block_table_slot(fresh_ids@, i as nat) as int);
                let q = i / bs;
                let r = i % bs;
                vstd::arithmetic::div_mod::lemma_fundamental_div_mod(i, bs);
                vstd::arithmetic::div_mod::lemma_mod_bound(i, bs);
                crate::proof::tensor::geometry::lemma_blocks_needed_covers_pos(i as nat, suffix_len as nat);
                assert(q < fresh_ids@.len());
                vstd::arithmetic::div_mod::lemma_hoist_over_denominator(i, c_i,
                    BLOCK_SIZE_SPEC);
                assert((c_i * bs + i) / bs == c_i + q);
                vstd::arithmetic::div_mod::lemma_fundamental_div_mod(c_i * bs + i, bs);
                assert((c_i * bs + i) % bs == r);
                assert(ids[c_i + q] == fresh_ids@[q]);
            }

            // ---- Whole-prompt placement: cached prefix + fresh suffix. ----
            assert forall|p: int|
                #![trigger token_placement_at(self.blocks@, ids, prompt_tokens@, p)]
                0 <= p < n_i
                implies token_placement_at(self.blocks@, ids, prompt_tokens@, p)
            by {
                if p < c_i * bs {
                    lemma_token_placement_prefix_at(pre_state.blocks@, matched_snapshot@,
                        prompt_tokens@, c_i * bs, p);
                    let j = p / bs;
                    let off = p % bs;
                    assert(0 <= j < c_i) by (nonlinear_arith)
                        requires 0 <= p, p < c_i * bs, bs == 64, j == p / 64,
                    {}
                    assert(ids[j] == matched_snapshot@[j]);
                    assert(self.blocks@[ids[j]].tokens@
                        == pre_state.blocks@[ids[j]].tokens@);
                } else {
                    let p2 = p - c_i * bs;
                    assert(0 <= p2 < suffix_len);
                    lemma_token_placement_prefix_at(post_alloc.blocks@, fresh_ids@,
                        suffix@, suffix_len, p2);
                    let q = p2 / bs;
                    let off = p2 % bs;
                    vstd::arithmetic::div_mod::lemma_fundamental_div_mod(p2, bs);
                    vstd::arithmetic::div_mod::lemma_mod_bound(p2, bs);
                    crate::proof::tensor::geometry::lemma_blocks_needed_covers_pos(p2 as nat, suffix_len as nat);
                    assert(q < fresh_ids@.len());
                    vstd::arithmetic::div_mod::lemma_hoist_over_denominator(p2, c_i,
                        BLOCK_SIZE_SPEC);
                    assert(p / bs == c_i + q) by {
                        assert(p == c_i * bs + p2);
                        assert((c_i * bs + p2) / bs == c_i + q);
                    }
                    vstd::arithmetic::div_mod::lemma_fundamental_div_mod(p, bs);
                    assert(p % bs == off);
                    assert(ids[c_i + q] == fresh_ids@[q]);
                    assert(suffix@[p2] == prompt_tokens@[sstart as int + p2]);
                    assert(sstart as int + p2 == p);
                    assert(self.blocks@[fresh_ids@[q]] == post_alloc.blocks@[fresh_ids@[q]]) by {
                        let kk = c_i + q;
                        assert(ids[kk] == fresh_ids@[q]);
                        assert(self.blocks@[ids[kk]] == post_alloc.blocks@[ids[kk]]);
                    }
                }
            }
            assert(token_placement_prefix(self.blocks@, ids, prompt_tokens@, n_i));

            // ---- Collision-safe provenance of the reused prefix. ----
            assert(ids.subrange(0, c_i) =~= matched_snapshot@);
            assert(registered_prefix_chain(self.blocks@, matched_snapshot@)) by {
                assert forall|j: int|
                    #![trigger self.blocks@[matched_snapshot@[j]].prefix_depth]
                    0 <= j < matched_snapshot@.len() implies {
                    let bid = matched_snapshot@[j];
                    &&& self.blocks@.contains_key(bid)
                    &&& self.blocks@[bid].prefix_depth as int == j + 1
                    &&& self.blocks@[bid].parent_block
                        == if j == 0 { None } else { Some(matched_snapshot@[j - 1]) }
                } by {
                    assert(registered_prefix_chain(pre_state.blocks@, matched_snapshot@));
                    assert(ids[j] == matched_snapshot@[j]);
                    assert(self.blocks@[ids[j]].prefix_depth
                        == pre_state.blocks@[matched_snapshot@[j]].prefix_depth);
                    assert(self.blocks@[ids[j]].parent_block
                        == pre_state.blocks@[matched_snapshot@[j]].parent_block);
                }
            }
            assert(registered_prefix_chain(self.blocks@, ids.subrange(0, c_i)));
            assert forall|k: int| #![trigger ids[k]] 0 <= k < c_i
                implies old(self).hash_to_block@.contains_key(
                        old(self).blocks@[ids[k]].hash_value)
                    && old(self).hash_to_block@[
                        old(self).blocks@[ids[k]].hash_value] == ids[k]
            by {
                assert(ids[k] == matched_snapshot@[k]);
                assert(pre_state.blocks@[ids[k]]
                    == old(self).blocks@[ids[k]]);
                assert(pre_state.hash_to_block@.contains_key(
                    pre_state.blocks@[matched_snapshot@[k]].hash_value));
                assert(pre_state.hash_to_block@[
                    pre_state.blocks@[matched_snapshot@[k]].hash_value]
                    == matched_snapshot@[k]);
            }

            // ---- Residency-domain bookkeeping. ----
            assert(self.request_residency@.dom() == post_alloc.request_residency@.dom());

            // ---- Refcount validity WITH SHARING: matched blocks gained
            // exactly the one new holder `rid`. ----
            assert(refcount_valid(self)) by {
                assert forall|bid: BlockId| #[trigger] self.blocks@.contains_key(bid)
                    implies self.blocks@[bid].refcount as int
                        == residency_holders_of(self, bid).len() as int
                by {
                    let hf = residency_holders_of(self, bid);
                    let hp = residency_holders_of(&post_alloc, bid);
                    assert(post_alloc.blocks@.dom().contains(bid));
                    assert(post_alloc.blocks@.contains_key(bid));
                    assert(refcount_valid(&post_alloc));
                    assert(post_alloc.blocks@[bid].refcount as int == hp.len() as int);
                    if matched_snapshot@.contains(bid) {
                        let k = matched_snapshot@.index_of(bid);
                        assert(ids[k] == bid);
                        assert(pre_state.blocks@.contains_key(bid));
                        assert(post_alloc.blocks@[bid] == pre_state.blocks@[bid]);
                        assert(self.blocks@[bid].refcount as int
                            == post_alloc.blocks@[bid].refcount as int + 1);
                        assert(!fresh_ids@.contains(bid)) by {
                            if fresh_ids@.contains(bid) {
                                let fx = fresh_ids@.index_of(bid);
                                assert(ids[c_i + fx] == bid);
                                assert(!pre_state.blocks@.contains_key(ids[c_i + fx]));
                            }
                        }
                        assert(!hp.contains(rid)) by {
                            if hp.contains(rid) {
                                assert(post_alloc.request_residency@[rid].block_ids@
                                    .contains(bid));
                            }
                        }
                        assert_sets_equal!(hf == hp.insert(rid), r: RequestId => {
                            if hf.contains(r) {
                                assert(self.request_residency@.contains_key(r));
                                if r == rid {
                                } else {
                                    assert(resid_before.contains_key(r));
                                    assert(self.request_residency@[r] == resid_before[r]);
                                    assert(post_alloc.request_residency@.contains_key(r));
                                }
                            }
                            if hp.insert(rid).contains(r) {
                                if r == rid {
                                    assert(self.request_residency@.contains_key(rid));
                                    assert(self.request_residency@[rid].block_ids@[k] == bid);
                                    assert(self.request_residency@[rid].block_ids@
                                        .contains(bid));
                                } else {
                                    assert(hp.contains(r));
                                    assert(post_alloc.request_residency@.contains_key(r));
                                    assert(resid_before.contains_key(r));
                                    assert(self.request_residency@.contains_key(r));
                                    assert(self.request_residency@[r] == resid_before[r]);
                                }
                            }
                        });
                        vstd::set::lemma_set_insert_len(hp, rid);
                        assert(hf.len() == hp.len() + 1);
                    } else {
                        assert(self.blocks@[bid] == post_alloc.blocks@[bid]);
                        assert_sets_equal!(hf == hp, r: RequestId => {
                            if hf.contains(r) {
                                if r == rid {
                                    assert(self.request_residency@[rid].block_ids@
                                        .contains(bid));
                                    let ix = self.request_residency@[rid].block_ids@
                                        .index_of(bid);
                                    assert(ids[ix] == bid);
                                    assert(ix >= c_i) by {
                                        if ix < c_i {
                                            assert(ids[ix] == matched_snapshot@[ix]);
                                            assert(matched_snapshot@.contains(bid));
                                        }
                                    }
                                    assert(fresh_ids@[ix - c_i] == bid);
                                    assert(post_alloc.request_residency@[rid].block_ids@
                                        .contains(bid));
                                    assert(post_alloc.request_residency@.contains_key(rid));
                                } else {
                                    assert(resid_before.contains_key(r));
                                    assert(self.request_residency@[r] == resid_before[r]);
                                    assert(post_alloc.request_residency@.contains_key(r));
                                }
                            }
                            if hp.contains(r) {
                                if r == rid {
                                    assert(post_alloc.request_residency@[rid].block_ids@
                                        .contains(bid));
                                    let fx = fresh_ids@.index_of(bid);
                                    assert(ids[c_i + fx] == bid);
                                    assert(self.request_residency@[rid].block_ids@
                                        .contains(bid));
                                    assert(self.request_residency@.contains_key(rid));
                                } else {
                                    assert(resid_before.contains_key(r));
                                    assert(self.request_residency@.contains_key(r));
                                    assert(self.request_residency@[r] == resid_before[r]);
                                }
                            }
                        });
                    }
                }
            }

            // ---- Remaining cs_valid conjuncts. ----
            assert(live_covers_queue(self));
            assert(live_requests_accepted(self)) by {
                assert(live_requests_accepted(&post_alloc));
            }
            assert(queue_disjoint(self));
            assert(running_unique(self));
            assert(waiting_unique(self));
            assert(running_has_residency(self)) by {
                assert forall|r: RequestId| #[trigger] self.running@.contains(r)
                    implies self.request_residency@.contains_key(r)
                by {
                    assert(post_alloc.running@.contains(r));
                    assert(post_alloc.request_residency@.contains_key(r));
                }
            }
            assert(waiting_has_no_residency(self)) by {
                assert forall|w: RequestId| #[trigger] self.waiting@.contains(w)
                    implies !self.request_residency@.contains_key(w)
                by {
                    assert(post_alloc.waiting@.contains(w));
                    assert(!post_alloc.request_residency@.contains_key(w));
                }
            }
            assert(waiting_unstarted(self)) by {
                assert(waiting_unstarted(&post_alloc));
            }
            assert(residency_has_live_request(self)) by {
                assert forall|r: RequestId|
                    #[trigger] self.request_residency@.contains_key(r)
                    implies self.live_requests@.contains_key(r)
                by {
                    assert(resid_before.dom()
                        == post_alloc.request_residency@.dom());
                    assert(post_alloc.request_residency@.contains_key(r));
                    assert(residency_has_live_request(&post_alloc));
                }
            }
            assert(residency_blocks_in_range(self)) by {
                assert forall|r: RequestId, k: int|
                    #![trigger self.request_residency@[r].block_ids@[k]]
                    self.request_residency@.contains_key(r)
                    && 0 <= k < self.request_residency@[r].block_ids@.len()
                    implies self.blocks@.contains_key(
                        self.request_residency@[r].block_ids@[k])
                by {
                    if r == rid {
                        assert(self.request_residency@[rid].block_ids@[k] == ids[k]);
                    } else {
                        assert(resid_before.contains_key(r));
                        assert(self.request_residency@[r] == resid_before[r]);
                        assert(post_alloc.request_residency@.contains_key(r));
                        assert(post_alloc.blocks@.contains_key(
                            post_alloc.request_residency@[r].block_ids@[k]));
                        assert(self.blocks@.dom().contains(
                            self.request_residency@[r].block_ids@[k]));
                    }
                }
            }
            assert(block_token_bound(self)) by {
                assert forall|bid: BlockId| #[trigger] self.blocks@.contains_key(bid)
                    implies self.blocks@[bid].tokens@.len() <= BLOCK_SIZE_SPEC as int
                by {
                    assert(post_alloc.blocks@.dom().contains(bid));
                    assert(post_alloc.blocks@.contains_key(bid));
                    if matched_snapshot@.contains(bid) {
                        let k = matched_snapshot@.index_of(bid);
                        assert(ids[k] == bid);
                        assert(self.blocks@[ids[k]].tokens@.len() == bs);
                    } else {
                        assert(self.blocks@[bid] == post_alloc.blocks@[bid]);
                        assert(block_token_bound(&post_alloc));
                    }
                }
            }
            assert(hash_to_block_in_range(self)) by {
                assert forall|h: u64| #[trigger] self.hash_to_block@.contains_key(h)
                    implies self.blocks@.contains_key(self.hash_to_block@[h])
                        && self.blocks@[self.hash_to_block@[h]].tokens@.len()
                            == BLOCK_SIZE_SPEC as int
                by {
                    assert(post_alloc.hash_to_block@.contains_key(h));
                    assert(hash_to_block_in_range(&post_alloc));
                    let tgt = self.hash_to_block@[h];
                    assert(post_alloc.blocks@.contains_key(tgt));
                    assert(self.blocks@.dom().contains(tgt));
                    if matched_snapshot@.contains(tgt) {
                        let k = matched_snapshot@.index_of(tgt);
                        assert(ids[k] == tgt);
                        assert(self.blocks@[ids[k]].tokens@.len() == bs);
                    } else {
                        assert(self.blocks@[tgt] == post_alloc.blocks@[tgt]);
                    }
                }
            }
            assert(hash_to_block_consistent(self)) by {
                assert forall|h: u64| #[trigger] self.hash_to_block@.contains_key(h)
                    implies self.blocks@[self.hash_to_block@[h]].hash_value == h
                        && self.blocks@[self.hash_to_block@[h]].prefix_depth > 0
                by {
                    assert(post_alloc.hash_to_block@.contains_key(h));
                    assert(hash_to_block_consistent(&post_alloc));
                    let tgt = self.hash_to_block@[h];
                    if matched_snapshot@.contains(tgt) {
                        let k = matched_snapshot@.index_of(tgt);
                        assert(ids[k] == tgt);
                        assert(pre_state.blocks@.contains_key(tgt));
                        assert(post_alloc.blocks@[tgt] == pre_state.blocks@[tgt]);
                        assert(self.blocks@[ids[k]].hash_value
                            == pre_state.blocks@[ids[k]].hash_value);
                    } else {
                        assert(hash_to_block_in_range(&post_alloc));
                        assert(post_alloc.blocks@.contains_key(tgt));
                        assert(self.blocks@[tgt] == post_alloc.blocks@[tgt]);
                    }
                }
            }
            assert(registered_provenance_aligned(self)) by {
                assert forall|r: RequestId, j: int|
                    #![trigger self.blocks@[self.request_residency@[r]
                        .block_ids@[j]].prefix_depth]
                    self.request_residency@.contains_key(r)
                    && 0 <= j < self.request_residency@[r].block_ids@.len()
                    && self.blocks@.contains_key(
                        self.request_residency@[r].block_ids@[j])
                    && self.blocks@[self.request_residency@[r]
                        .block_ids@[j]].prefix_depth > 0
                    implies {
                        let rids = self.request_residency@[r].block_ids@;
                        let rb = rids[j];
                        &&& self.blocks@[rb].prefix_depth as int == j + 1
                        &&& self.blocks@[rb].parent_block
                            == if j == 0 { None } else { Some(rids[j - 1]) }
                    }
                by {
                    let rids = self.request_residency@[r].block_ids@;
                    let rb = rids[j];
                    if r == rid {
                        assert(rids == ids);
                        if j < c_i {
                            assert(ids[j] == matched_snapshot@[j]);
                            assert(self.blocks@[ids[j]].prefix_depth
                                == pre_state.blocks@[matched_snapshot@[j]].prefix_depth);
                            assert(self.blocks@[ids[j]].parent_block
                                == pre_state.blocks@[matched_snapshot@[j]].parent_block);
                            assert(registered_prefix_chain(
                                pre_state.blocks@, matched_snapshot@));
                            if j > 0 {
                                assert(ids[j - 1] == matched_snapshot@[j - 1]);
                            }
                        } else {
                            assert(self.blocks@[ids[j]].prefix_depth == 0);
                        }
                    } else {
                        assert(post_alloc.request_residency@.dom()
                            == pre_state.request_residency@.dom().insert(rid));
                        assert(post_alloc.request_residency@.contains_key(r));
                        assert(r != rid);
                        assert(pre_state.request_residency@.contains_key(r));
                        assert(self.request_residency@[r]
                            == pre_state.request_residency@[r]);
                        assert(pre_state.blocks@.contains_key(rb)) by {
                            assert(residency_blocks_in_range(&pre_state));
                        }
                        assert(self.blocks@[rb].prefix_depth
                            == pre_state.blocks@[rb].prefix_depth);
                        assert(self.blocks@[rb].parent_block
                            == pre_state.blocks@[rb].parent_block);
                        assert(registered_provenance_aligned(&pre_state));
                    }
                }
            }
            assert(residency_block_ids_unique(self)) by {
                assert forall|r: RequestId| #[trigger] self.request_residency@.contains_key(r)
                    implies self.request_residency@[r].block_ids@.no_duplicates()
                by {
                    if r == rid {
                    } else {
                        assert(resid_before.contains_key(r));
                        assert(self.request_residency@[r] == resid_before[r]);
                        assert(post_alloc.request_residency@.contains_key(r));
                        assert(residency_block_ids_unique(&post_alloc));
                    }
                }
            }
            assert(blocks_dom_in_range(self)) by {
                assert forall|bid: BlockId| #[trigger] self.blocks@.contains_key(bid)
                    implies bid < self.num_blocks
                by {
                    assert(post_alloc.blocks@.dom().contains(bid));
                    assert(post_alloc.blocks@.contains_key(bid));
                    assert(blocks_dom_in_range(&post_alloc));
                }
            }
            assert(block_count_valid(self)) by {
                assert(block_count_valid(&post_alloc));
                assert(self.blocks@.dom().len() == post_alloc.blocks@.dom().len());
            }
            assert(hash_to_block_no_zero(self)) by {
                assert(hash_to_block_no_zero(&post_alloc));
            }
            assert(cs_valid(self));

            // ---- The generalized success shape. ----
            assert(self.request_residency@[rid].cached_prefix_blocks as int == c_i);
            assert(self.request_residency@.dom()
                == pre_state.request_residency@.dom().insert(rid));
            assert(self.free_blocks as int
                == pre_state.free_blocks as int - (ids.len() - c_i)) by {
                assert(post_alloc.free_blocks as int
                    == pre_state.free_blocks as int
                        - blocks_needed_for(suffix@.len() as nat) as int);
            }
            assert(prompt_tokens@.len() > 0) by {
                assert(c_i >= 1);
                assert(c_i * bs < n_i);
                assert(bs <= c_i * bs) by (nonlinear_arith)
                    requires 1 <= c_i, bs == 64,
                {}
            }
            assert(allocate_prefill_reuse_success(old(self), self, rid,
                prompt_tokens@, c_i));
            assert forall|k: int| #![trigger ids[k]] 0 <= k < c_i
                implies !excluded@.contains(old(self).blocks@[ids[k]].hash_value)
            by {
                assert(ids[k] == matched_snapshot@[k]);
                assert(pre_state.blocks@[ids[k]].hash_value
                    == pre_state.blocks@[matched_snapshot@[k]].hash_value);
            }
            reveal(reused_prefix_excludes);
            assert(reused_prefix_excludes(
                old(self), self, rid, excluded@, c_i,
            ));
            assert forall|other: RequestId|
                other != rid && #[trigger] old(self).request_residency@.contains_key(other)
                implies self.request_residency@.contains_key(other)
                    && self.request_residency@[other] == old(self).request_residency@[other]
            by {
                assert(post_alloc.request_residency@.contains_key(other));
                assert(post_alloc.request_residency@[other]
                    == pre_state.request_residency@[other]);
                assert(resid_before.contains_key(other));
                assert(self.request_residency@[other] == resid_before[other]);
            }
            assert(positive_provenance_origin(&post_alloc, self)) by {
                reveal(positive_provenance_origin);
                assert forall|bid: BlockId|
                    #[trigger] self.blocks@[bid].prefix_depth > 0
                    && self.blocks@.contains_key(bid)
                    && self.blocks@[bid].prefix_depth > 0
                    implies post_alloc.blocks@.contains_key(bid)
                        && self.blocks@[bid].tokens@ == post_alloc.blocks@[bid].tokens@
                        && self.blocks@[bid].hash_value == post_alloc.blocks@[bid].hash_value
                        && self.blocks@[bid].prefix_depth == post_alloc.blocks@[bid].prefix_depth
                        && self.blocks@[bid].parent_block == post_alloc.blocks@[bid].parent_block
                by {
                    assert(post_alloc.blocks@.contains_key(bid));
                    if matched_snapshot@.contains(bid) {
                        let k = matched_snapshot@.index_of(bid);
                        assert(ids[k] == bid);
                        assert(post_alloc.blocks@[bid] == pre_state.blocks@[bid]);
                    } else {
                        assert(self.blocks@[bid] == post_alloc.blocks@[bid]);
                    }
                }
            }
            lemma_positive_provenance_origin_transitive(
                old(self), &post_alloc, self,
            );
            if persistent_provenance_closed(old(self)) {
                assert(persistent_provenance_closed(&pre_state));
                assert(persistent_provenance_closed(&post_alloc));
                assert forall|bid: BlockId|
                    #[trigger] self.blocks@.contains_key(bid)
                    && self.blocks@[bid].prefix_depth > 0
                    implies post_alloc.blocks@.contains_key(bid)
                        && self.blocks@[bid].tokens@
                            == post_alloc.blocks@[bid].tokens@
                        && self.blocks@[bid].hash_value
                            == post_alloc.blocks@[bid].hash_value
                        && self.blocks@[bid].prefix_depth
                            == post_alloc.blocks@[bid].prefix_depth
                        && self.blocks@[bid].parent_block
                            == post_alloc.blocks@[bid].parent_block
                by {
                    assert(post_alloc.blocks@.contains_key(bid));
                    if matched_snapshot@.contains(bid) {
                        let k = matched_snapshot@.index_of(bid);
                        assert(ids[k] == bid);
                        assert(post_alloc.blocks@[bid] == pre_state.blocks@[bid]);
                    } else {
                        assert(self.blocks@[bid] == post_alloc.blocks@[bid]);
                    }
                }
                assert forall|bid: BlockId|
                    #[trigger] post_alloc.blocks@.contains_key(bid)
                    && post_alloc.blocks@[bid].prefix_depth > 0
                    implies self.blocks@.contains_key(bid)
                        && self.blocks@[bid].tokens@
                            == post_alloc.blocks@[bid].tokens@
                        && self.blocks@[bid].hash_value
                            == post_alloc.blocks@[bid].hash_value
                        && self.blocks@[bid].prefix_depth
                            == post_alloc.blocks@[bid].prefix_depth
                        && self.blocks@[bid].parent_block
                            == post_alloc.blocks@[bid].parent_block
                by {
                    if matched_snapshot@.contains(bid) {
                        let k = matched_snapshot@.index_of(bid);
                        assert(ids[k] == bid);
                    } else {
                        assert(self.blocks@[bid] == post_alloc.blocks@[bid]);
                    }
                }
                assert(positive_provenance_metadata_frame(
                    &post_alloc, self,
                )) by {
                    reveal(positive_provenance_metadata_frame);
                }
                lemma_physical_parent_closure_from_metadata_frame(
                    &post_alloc, self,
                );
                lemma_persistent_provenance_closed_frame(&post_alloc, self);
            }
        }
        proof { lemma_free_queue_token_to_valid(self); }
        true
    }

    // Scan for reusable pages. Walk the
    // prompt's full blocks left to right, chain-hash, look up a registered
    // candidate, check its exact physical prefix provenance, and token-compare
    // it; collect matching block ids until the
    // first miss.  Guards: never take a candidate twice (keeps the future
    // residency's block_ids duplicate-free with no hash-injectivity
    // argument), and always leave at least one prompt position uncached so
    // a partial prefill still has a query row to sample from. This scan is read-only;
    // the subsequent allocation bumps refcounts and allocates the suffix.
    // The parent/depth check plus token comparison makes reuse collision-safe:
    // the hash is only an index, while accepted pages must be the exact chain
    // under which their K/V was materialized.
    #[verifier::spinoff_prover]
    #[verifier::rlimit(150)]
    pub fn match_cached_prefix(
        &self,
        prompt_tokens: &Vec<TokenId>,
        excluded: &Vec<u64>,
    ) -> (out: Vec<BlockId>)
        requires
            cs_valid(self),
            persistent_provenance_closed(self),
            prompt_tokens@.len() <= usize::MAX as int,
        ensures
            out@.no_duplicates(),
            prompt_tokens@.len() == 0 ==> out@.len() == 0,
            prompt_tokens@.len() > 0
                ==> out@.len() as int * (BLOCK_SIZE_SPEC as int) < prompt_tokens@.len() as int,
            forall|j: int| 0 <= j < out@.len()
                ==> self.blocks@.contains_key(#[trigger] out@[j])
                    && self.blocks@[out@[j]].tokens@.len() == BLOCK_SIZE_SPEC as int,
            // Matched blocks are registry targets — their stored
            // hash is nonzero (registry hygiene) and NOT in the caller's
            // exclusion list (the same-step guard).
            forall|j: int| 0 <= j < out@.len()
                ==> (#[trigger] self.blocks@[out@[j]].hash_value) != 0
                    && !excluded@.contains(self.blocks@[out@[j]].hash_value),
            forall|j: int| 0 <= j < out@.len()
                ==> self.hash_to_block@.contains_key(
                        #[trigger] self.blocks@[out@[j]].hash_value)
                    && self.hash_to_block@[self.blocks@[out@[j]].hash_value] == out@[j],
            registered_prefix_chain(self.blocks@, out@),
            token_placement_prefix(self.blocks@, out@, prompt_tokens@,
                out@.len() as int * (BLOCK_SIZE_SPEC as int)),
    {
        let n: u64 = prompt_tokens.len() as u64;
        let c_max: u64 = if n == 0 { 0 } else { (n - 1) / BLOCK_SIZE };
        let mut out: Vec<BlockId> = Vec::new();
        let mut ph: u64 = 0;
        let mut j: usize = 0;
        proof {
            assert(token_placement_prefix(self.blocks@, out@, prompt_tokens@, 0));
            assert(registered_prefix_chain(self.blocks@, out@));
        }
        while (j as u64) < c_max
            invariant
                cs_valid(self),
                n as int == prompt_tokens@.len(),
                n as int <= usize::MAX as int,
                c_max as int == (if n == 0 { 0int } else { (n as int - 1) / 64 }),
                j as int == out@.len(),
                j as int <= c_max as int,
                out@.no_duplicates(),
                forall|q: int| 0 <= q < out@.len()
                    ==> self.blocks@.contains_key(#[trigger] out@[q])
                        && self.blocks@[out@[q]].tokens@.len() == BLOCK_SIZE_SPEC as int,
                forall|q: int| 0 <= q < out@.len()
                    ==> (#[trigger] self.blocks@[out@[q]].hash_value) != 0
                        && !excluded@.contains(self.blocks@[out@[q]].hash_value),
                forall|q: int| 0 <= q < out@.len()
                    ==> self.hash_to_block@.contains_key(
                            #[trigger] self.blocks@[out@[q]].hash_value)
                        && self.hash_to_block@[self.blocks@[out@[q]].hash_value] == out@[q],
                registered_prefix_chain(self.blocks@, out@),
                token_placement_prefix(self.blocks@, out@, prompt_tokens@,
                    out@.len() as int * (BLOCK_SIZE_SPEC as int)),
            decreases c_max as int - j as int,
        {
            proof {
                // (j+1)*64 <= n: j < c_max == (n-1)/64.
                assert((j as int + 1) * 64 <= n as int) by (nonlinear_arith)
                    requires (j as int) < c_max as int,
                        c_max as int == (n as int - 1) / 64, n >= 1,
                {}
            }
            let start: usize = j * (BLOCK_SIZE as usize);
            let end: usize = start + (BLOCK_SIZE as usize);
            ph = chain_hash_span(ph, prompt_tokens, start, end);
            // SAME-STEP guard: hashes registered by THIS plan invocation are
            // excluded — their blocks' KV is not materialized until the
            // step's forward runs, so reusing them would be wrong.  (Sound
            // to over-exclude; the ensures are unaffected.)
            let mut excluded_hit = false;
            let mut ei: usize = 0;
            while ei < excluded.len()
                invariant
                    ei <= excluded@.len(),
                    excluded_hit == (exists|t: int|
                        0 <= t < ei as int && excluded@[t] == ph),
                decreases excluded@.len() - ei,
            {
                if excluded[ei] == ph {
                    excluded_hit = true;
                    proof { assert(excluded@[ei as int] == ph); }
                }
                ei += 1;
            }
            proof {
                if !excluded_hit {
                    assert(!excluded@.contains(ph));
                }
            }
            if excluded_hit {
                break;
            }
            let cand: BlockId = match self.hash_to_block.get(&ph) {
                Some(b) => *b,
                None => {
                    break;
                },
            };
            proof {
                assert(self.hash_to_block@.contains_key(ph));
                assert(hash_to_block_in_range(self));
                assert(self.blocks@.contains_key(cand));
                assert(self.blocks@[cand].tokens@.len() == BLOCK_SIZE_SPEC as int);
                assert((j as u64) < u64::MAX);
            }
            let (cand_depth, cand_parent) = match self.blocks.get(&cand) {
                Some(e) => (e.prefix_depth, e.parent_block),
                None => {
                    proof { assert(false); }
                    (0, None)
                },
            };
            let expected_parent = if j == 0 { None } else { Some(out[j - 1]) };
            if cand_depth != j as u64 + 1
                || cand_parent != expected_parent
            {
                break;
            }
            proof {
                assert(cand_depth as int == j as int + 1);
                assert(cand_parent
                    == if j as int == 0 { None } else { Some(out@[j as int - 1]) });
            }
            // Duplicate guard: never reuse the same physical block twice.
            let mut dup = false;
            let mut q: usize = 0;
            while q < out.len()
                invariant
                    q <= out@.len(),
                    dup == (exists|t: int| 0 <= t < q as int && out@[t] == cand),
                decreases out@.len() - q,
            {
                if out[q] == cand {
                    dup = true;
                    proof {
                        assert(out@[q as int] == cand);
                    }
                }
                q += 1;
            }
            if dup {
                break;
            }
            // TOKEN COMPARE: the candidate must hold exactly this prompt block.
            let cand_tokens_len = match self.blocks.get(&cand) {
                Some(e) => e.tokens.len(),
                None => {
                    proof { assert(false); }
                    0
                },
            };
            proof {
                assert(cand_tokens_len as int == BLOCK_SIZE_SPEC as int);
            }
            let mut all_match = true;
            let mut o: usize = 0;
            while o < BLOCK_SIZE as usize
                invariant
                    o <= BLOCK_SIZE_SPEC as nat,
                    n as int == prompt_tokens@.len(),
                    (start as int) == (j as int) * 64,
                    (j as int + 1) * 64 <= n as int,
                    self.blocks@.contains_key(cand),
                    self.blocks@[cand].tokens@.len() == BLOCK_SIZE_SPEC as int,
                    all_match ==> forall|t: int| 0 <= t < o as int
                        ==> self.blocks@[cand].tokens@[t]
                            == #[trigger] prompt_tokens@[start as int + t],
                decreases BLOCK_SIZE_SPEC as int - o as int,
            {
                let block_tok = match self.blocks.get(&cand) {
                    Some(e) => e.tokens[o],
                    None => {
                        proof { assert(false); }
                        0
                    },
                };
                if block_tok != prompt_tokens[start + o] {
                    all_match = false;
                }
                proof {
                    if all_match {
                        assert forall|t: int| 0 <= t < o as int + 1
                            implies self.blocks@[cand].tokens@[t]
                                == #[trigger] prompt_tokens@[start as int + t]
                        by {
                            if t < o as int {
                            } else {
                                assert(t == o as int);
                            }
                        }
                    }
                }
                o += 1;
            }
            if !all_match {
                break;
            }
            let ghost out_pre = out@;
            out.push(cand);
            proof {
                assert(out@ == out_pre.push(cand));
                assert(!out_pre.contains(cand)) by {
                    if out_pre.contains(cand) {
                        let t = out_pre.index_of(cand);
                        assert(0 <= t < out_pre.len() && out_pre[t] == cand);
                    }
                }
                assert(out@.no_duplicates()) by {
                    reveal(Seq::no_duplicates);
                }
                assert(self.hash_to_block@[ph] == cand);
                assert(hash_to_block_consistent(self));
                assert(self.blocks@[cand].hash_value == ph);
                assert forall|t: int| 0 <= t < out@.len()
                    implies self.hash_to_block@.contains_key(
                            #[trigger] self.blocks@[out@[t]].hash_value)
                        && self.hash_to_block@[self.blocks@[out@[t]].hash_value] == out@[t]
                by {
                    if t < out_pre.len() {
                        assert(out@[t] == out_pre[t]);
                    } else {
                        assert(t == out_pre.len());
                        assert(out@[t] == cand);
                    }
                }
                assert(registered_prefix_chain(self.blocks@, out@)) by {
                    assert forall|t: int|
                        #![trigger self.blocks@[out@[t]].prefix_depth]
                        0 <= t < out@.len() implies {
                        let bid = out@[t];
                        &&& self.blocks@.contains_key(bid)
                        &&& self.blocks@[bid].prefix_depth as int == t + 1
                        &&& self.blocks@[bid].parent_block
                            == if t == 0 { None } else { Some(out@[t - 1]) }
                    } by {
                        if t < out_pre.len() {
                            assert(registered_prefix_chain(self.blocks@, out_pre));
                            assert(out@[t] == out_pre[t]);
                            assert(self.blocks@[out_pre[t]].prefix_depth as int == t + 1);
                            assert(self.blocks@[out_pre[t]].parent_block
                                == if t == 0 { None } else { Some(out_pre[t - 1]) });
                            if t > 0 {
                                assert(out@[t - 1] == out_pre[t - 1]);
                            }
                        } else {
                            assert(t == out_pre.len());
                            assert(t == j as int);
                            assert(out@[t] == cand);
                            assert(cand_depth == self.blocks@[cand].prefix_depth);
                            assert(cand_parent == self.blocks@[cand].parent_block);
                            assert(self.blocks@[cand].prefix_depth as int == t + 1);
                            if t == 0 {
                                assert(expected_parent == Option::<BlockId>::None);
                            } else {
                                assert(expected_parent == Some(out_pre[t - 1]));
                                assert(out@[t - 1] == out_pre[t - 1]);
                            }
                        }
                    }
                }
                // Extend placement to the newly matched block's range.
                assert forall|p: int|
                    #![trigger token_placement_at(self.blocks@, out@, prompt_tokens@, p)]
                    0 <= p < out@.len() as int * (BLOCK_SIZE_SPEC as int)
                    implies token_placement_at(self.blocks@, out@, prompt_tokens@, p)
                by {
                    let bs = BLOCK_SIZE_SPEC as int;
                    if p < out_pre.len() as int * bs {
                        lemma_token_placement_prefix_at(self.blocks@, out_pre,
                            prompt_tokens@, out_pre.len() as int * bs, p);
                        assert(0 <= p / bs < out_pre.len() as int) by (nonlinear_arith)
                            requires 0 <= p, p < out_pre.len() as int * bs, bs == 64,
                        {}
                        assert(out@[p / bs] == out_pre[p / bs]);
                    } else {
                        assert(p / bs == j as int) by (nonlinear_arith)
                            requires out_pre.len() as int * bs <= p,
                                p < (out_pre.len() as int + 1) * bs,
                                out_pre.len() as int == j as int, bs == 64, 0 <= p,
                        {}
                        assert(out@[p / bs] == cand);
                        let off = p % bs;
                        assert(p == j as int * bs + off) by (nonlinear_arith)
                            requires p / bs == j as int, off == p % bs, bs == 64, 0 <= p,
                        {}
                        assert(0 <= off < bs) by {
                            vstd::arithmetic::div_mod::lemma_mod_bound(p, bs);
                        }
                        assert(self.blocks@[cand].tokens@[off]
                            == prompt_tokens@[start as int + off]);
                        assert(start as int + off == p);
                    }
                }
            }
            j += 1;
        }
        proof {
            if prompt_tokens@.len() > 0 {
                // out.len() <= c_max == (n-1)/64 implies out.len()*64 <= n-1 < n.
                assert(out@.len() as int * 64 < n as int) by (nonlinear_arith)
                    requires out@.len() as int <= (n as int - 1) / 64, n >= 1,
                {}
            }
        }
        out
    }
}

} // verus!
