// Verified scheduler transition contracts and preservation lemmas.

use super::*;

verus! {
// The first `fb` residency blocks each hold exactly BLOCK_SIZE tokens.
// The length bound sits outside the quantifier. If the trigger-free arithmetic
// conjunct is placed inside the consequent, Z3 can select a disjunct without
// materializing the `ids[j]` trigger term, preventing reliable folding.
pub open spec fn full_blocks_sized(cs: &CacheScheduler, ids: Seq<BlockId>, fb: int) -> bool {
    fb <= ids.len()
    && (forall|j: int| 0 <= j < fb
        ==> cs.blocks@.contains_key(#[trigger] ids[j])
            && cs.blocks@[ids[j]].tokens@.len() == BLOCK_SIZE_SPEC as int)
}

pub proof fn lemma_full_blocks_sized_transfer(
    a: &CacheScheduler, b: &CacheScheduler, ids: Seq<BlockId>, fb: int,
)
    requires
        full_blocks_sized(a, ids, fb),
        forall|bid: BlockId| #[trigger] a.blocks@.contains_key(bid)
            ==> b.blocks@.contains_key(bid)
                && b.blocks@[bid].tokens@ == a.blocks@[bid].tokens@,
    ensures
        full_blocks_sized(b, ids, fb),
{
    assert forall|j: int| 0 <= j < fb
        implies b.blocks@.contains_key(#[trigger] ids[j])
            && b.blocks@[ids[j]].tokens@.len() == BLOCK_SIZE_SPEC as int
    by {
        assert(a.blocks@.contains_key(ids[j]));
        assert(a.blocks@[ids[j]].tokens@.len() == BLOCK_SIZE_SPEC as int);
    }
}

// Block budget splits across a cached prefix: `c` full cached blocks plus
// the blocks for the remaining suffix cover the whole prompt.
pub proof fn lemma_blocks_needed_split(n: nat, c: nat)
    requires c * BLOCK_SIZE_SPEC < n,
    ensures
        blocks_needed_for(n) == c + blocks_needed_for((n - c * BLOCK_SIZE_SPEC) as nat),
{
    let bs = BLOCK_SIZE_SPEC as int;
    let r = n as int - c as int * bs - 1;
    assert(r >= 0);
    vstd::arithmetic::div_mod::lemma_hoist_over_denominator(r, c as int, bs as nat);
    assert(r / bs + c as int == (r + c as int * bs) / bs);
    assert(r + c as int * bs == n as int - 1);
}

// Success shape for a reuse-aware prefill allocation (prefix caching step
// 2b): the residency's block table is the MATCHED cached prefix followed by
// fresh suffix blocks; `slot_mapping` covers only the uncached suffix
// positions (at their global block-table slots); matched blocks gained one
// holder; and the whole prompt is placed — the cached prefix by token
// equality (the scan compared it), the suffix by the fresh writes.
pub open spec fn allocate_prefill_reuse_success(
    pre: &CacheScheduler,
    post: &CacheScheduler,
    rid: RequestId,
    prompt_tokens: Seq<TokenId>,
    c: int,
) -> bool {
    let n = prompt_tokens.len() as int;
    let bs = BLOCK_SIZE_SPEC as int;
    let ids = post.request_residency@[rid].block_ids@;
    &&& post.request_residency@.contains_key(rid)
    &&& 0 <= c
    &&& c * bs < n
    &&& post.request_residency@[rid].cached_prefix_blocks as int == c
    &&& ids.len() == c + blocks_needed_for((n - c * bs) as nat) as int
    &&& ids.no_duplicates()
    // Every reused page is registered for exactly this physical predecessor
    // chain.  This is the collision-safety bridge needed by cache-fidelity
    // proofs; the rolling hash alone is not trusted.
    &&& registered_prefix_chain(post.blocks@, ids.subrange(0, c))
    // Each reused page was an actual registry target in the pre-state.  This
    // reverse lookup is what lets the semantic proof recover a pre-existing
    // live holder; a nonzero page stamp alone would not suffice.
    &&& (forall|k: int| #![trigger ids[k]] 0 <= k < c
        ==> pre.hash_to_block@.contains_key(pre.blocks@[ids[k]].hash_value)
            && pre.hash_to_block@[pre.blocks@[ids[k]].hash_value] == ids[k])
    &&& post.request_residency@[rid].slot_mapping@.len() == n - c * bs
    &&& (forall|i: int| #![trigger post.request_residency@[rid].slot_mapping@[i]]
        0 <= i < n - c * bs
        ==> post.request_residency@[rid].slot_mapping@[i] as int
            == block_table_slot(ids, (c * bs + i) as nat) as int)
    &&& (forall|k: int| #![trigger ids[k]] 0 <= k < c
        ==> pre.blocks@.contains_key(ids[k])
            && post.blocks@.contains_key(ids[k])
            && post.blocks@[ids[k]].tokens@ == pre.blocks@[ids[k]].tokens@
            && post.blocks@[ids[k]].tokens@.len() == bs
            && post.blocks@[ids[k]].refcount as int
                == pre.blocks@[ids[k]].refcount as int + 1)
    &&& (forall|k: int| #![trigger ids[k]] c <= k < ids.len()
        ==> !pre.blocks@.contains_key(ids[k])
            && post.blocks@.contains_key(ids[k])
            && post.blocks@[ids[k]].refcount == 1
            && post.blocks@[ids[k]].hash_value == 0
            && post.blocks@[ids[k]].prefix_depth == 0
            && post.blocks@[ids[k]].parent_block == Option::<BlockId>::None)
    &&& token_placement_prefix(post.blocks@, ids, prompt_tokens, n)
    &&& post.free_blocks as int == pre.free_blocks as int - (ids.len() - c)
    // The tail block holds exactly the remainder — feeds the
    // residency/history alignment companion.
    &&& ids.len() >= 1
    &&& post.blocks@.contains_key(ids[ids.len() - 1])
    &&& post.blocks@[ids[ids.len() - 1]].tokens@.len() == n - (ids.len() - 1) * bs
    &&& post.blocks@[ids[ids.len() - 1]].tokens@.len() >= 1
}

// Same-step cache-safety fact exported opaquely: callers that do not reason
// about prefix origin should not instantiate this per-page quantifier merely
// because they invoked the allocator.
#[verifier::opaque]
pub open spec fn reused_prefix_excludes(
    pre: &CacheScheduler,
    post: &CacheScheduler,
    rid: RequestId,
    excluded: Seq<u64>,
    c: int,
) -> bool {
    forall|k: int| #![trigger post.request_residency@[rid].block_ids@[k]]
        0 <= k < c ==> !excluded.contains(pre.blocks@[
            post.request_residency@[rid].block_ids@[k]].hash_value)
}

// A running request's residency MIRRORS its history — the block
// table holds exactly `history` tokens: full blocks plus a non-empty tail
// with the remainder.  Carried as a COMPANION predicate (threaded through
// plan/commit/step, not folded into cs_valid: commit's intermediate states
// between the cache append and the live-map update legitimately break the
// equality).  It pins both block-table coverage and exact token contents; the
// latter is required to transport canonical K/V through a reused prefix page.
pub open spec fn residency_history_aligned(cs: &CacheScheduler) -> bool {
    forall|rid: RequestId|
        #[trigger] cs.running@.contains(rid) && cs.live_requests@.contains_key(rid)
        ==> {
            let hist = history(cs.live_requests@[rid]).len() as int;
            let ids = cs.request_residency@[rid].block_ids@;
            let bs = BLOCK_SIZE_SPEC as int;
            &&& cs.request_residency@.contains_key(rid)
            &&& ids.len() >= 1
            &&& cs.blocks@.contains_key(ids[ids.len() - 1])
            &&& hist == (ids.len() - 1) * bs
                + cs.blocks@[ids[ids.len() - 1]].tokens@.len()
            &&& cs.blocks@[ids[ids.len() - 1]].tokens@.len() >= 1
            &&& token_placement_prefix(
                cs.blocks@,
                ids,
                history(cs.live_requests@[rid]),
                hist,
            )
            // Tail exclusivity: a partially-filled tail is never shared
            // (prefix reuse only ever bumps refcounts of FULL registered
            // blocks).  Commit needs this so one request's tail-append
            // cannot silently grow a bystander's tail.
            &&& (cs.blocks@[ids[ids.len() - 1]].refcount == 1
                || cs.blocks@[ids[ids.len() - 1]].tokens@.len() == BLOCK_SIZE_SPEC as int)
        }
}

// The decode handoff: a running request's LAST mapped
// slot is exactly the block-table slot of its last history position — the
// token appended by the latest commit, or the prompt's last position right
// after allocation.  Threaded as a companion like
// `residency_history_aligned`; gives the decode row's slot identity (D1)
// by construction.
pub open spec fn slot_mapping_aligned(cs: &CacheScheduler) -> bool {
    forall|rid: RequestId|
        #[trigger] cs.running@.contains(rid) && cs.live_requests@.contains_key(rid)
        ==> {
            let hist = history(cs.live_requests@[rid]).len() as int;
            let ids = cs.request_residency@[rid].block_ids@;
            let sm = cs.request_residency@[rid].slot_mapping@;
            &&& cs.request_residency@.contains_key(rid)
            &&& sm.len() >= 1
            &&& hist >= 1
            &&& sm[sm.len() - 1] as int
                == crate::proof::tensor::geometry::block_table_slot(ids, (hist - 1) as nat) as int
        }
}

// Step-boundary phase companion: every resident request is running.  This is
// intentionally not part of `cs_valid`, because admission temporarily creates
// a residency between allocation and the subsequent running-queue push.
// At stable engine boundaries it turns a registry holder into an initialized
// donor machine through `refinement::phase_aligned`.
#[verifier::opaque]
pub open spec fn residency_running_aligned(cs: &CacheScheduler) -> bool {
    forall|rid: RequestId| #[trigger] cs.request_residency@.contains_key(rid)
        ==> cs.running@.contains(rid)
}

// Re-establish the opaque stable-boundary companion after an operation that
// only changes unrelated scheduler fields (for example, popping and then
// restoring the waiting-queue head).
pub proof fn lemma_residency_running_aligned_frame(
    before: &CacheScheduler,
    after: &CacheScheduler,
)
    requires
        residency_running_aligned(before),
        after.request_residency@ == before.request_residency@,
        after.running@ == before.running@,
    ensures
        residency_running_aligned(after),
{
    reveal(residency_running_aligned);
}

// Every page named by a residency has at least that request as a holder.
// This small bridge lets reclamation clients use the executable contract
// that preserves all positive-refcount pages, without exposing the holder
// cardinality definition throughout their larger proofs.
pub proof fn lemma_resident_block_has_positive_refcount(
    cs: &CacheScheduler,
    rid: RequestId,
    bid: BlockId,
)
    requires
        cs_valid(cs),
        cs.request_residency@.contains_key(rid),
        cs.request_residency@[rid].block_ids@.contains(bid),
    ensures
        cs.blocks@.contains_key(bid),
        cs.blocks@[bid].refcount > 0,
{
    assert(residency_blocks_in_range(cs));
    let j = cs.request_residency@[rid].block_ids@.index_of(bid);
    assert(cs.request_residency@[rid].block_ids@[j] == bid);
    let holders = residency_holders_of(cs, bid);
    assert(holders.contains(rid));
    assert(holders.len() > 0);
    assert(refcount_valid(cs));
    assert(cs.blocks@[bid].refcount as int == holders.len() as int);
}

pub proof fn lemma_registered_residency_prefix_positive_frame(
    before: &CacheScheduler,
    after: &CacheScheduler,
    rid: RequestId,
)
    requires
        cs_valid(before),
        before.request_residency@.contains_key(rid),
        after.request_residency@.contains_key(rid),
        after.request_residency@[rid] == before.request_residency@[rid],
        0 <= before.request_residency@[rid].cached_prefix_blocks as int
            <= before.request_residency@[rid].block_ids@.len(),
        ({
            let ids = before.request_residency@[rid].block_ids@;
            let c = before.request_residency@[rid].cached_prefix_blocks as int;
            registered_prefix_chain(before.blocks@, ids.subrange(0, c))
        }),
        forall|bid: BlockId| #[trigger] before.blocks@.contains_key(bid)
            && before.blocks@[bid].refcount > 0
            ==> after.blocks@.contains_key(bid)
                && after.blocks@[bid] == before.blocks@[bid],
    ensures
        ({
            let ids = after.request_residency@[rid].block_ids@;
            let c = after.request_residency@[rid].cached_prefix_blocks as int;
            registered_prefix_chain(after.blocks@, ids.subrange(0, c))
        }),
{
    let ids = before.request_residency@[rid].block_ids@;
    let c = before.request_residency@[rid].cached_prefix_blocks as int;
    assert forall|j: int| 0 <= j < ids.subrange(0, c).len() implies {
        let bid = #[trigger] ids.subrange(0, c)[j];
        &&& after.blocks@.contains_key(bid)
        &&& after.blocks@[bid].prefix_depth
            == before.blocks@[bid].prefix_depth
        &&& after.blocks@[bid].parent_block
            == before.blocks@[bid].parent_block
    }
    by {
        assert(ids.subrange(0, c)[j] == ids[j]);
        let bid = ids[j];
        assert(ids.contains(bid));
        lemma_resident_block_has_positive_refcount(before, rid, bid);
        assert(after.blocks@[bid] == before.blocks@[bid]);
    }
    lemma_registered_prefix_chain_transfer(
        before.blocks@, after.blocks@, ids.subrange(0, c),
    );
}

// Reclamation changes neither request ownership nor any resident page: all
// resident pages have positive refcount, and the pressure primitive preserves
// such pages verbatim.  Consequently the history/KV placement companion is
// stable even though unrelated zero-ref cache pages may disappear.
pub proof fn lemma_residency_history_aligned_positive_frame(
    before: &CacheScheduler,
    after: &CacheScheduler,
)
    requires
        cs_valid(before),
        residency_history_aligned(before),
        after.request_residency@ == before.request_residency@,
        after.live_requests@ == before.live_requests@,
        after.running@ == before.running@,
        forall|bid: BlockId| #[trigger] before.blocks@.contains_key(bid)
            && before.blocks@[bid].refcount > 0
            ==> after.blocks@.contains_key(bid)
                && after.blocks@[bid] == before.blocks@[bid],
    ensures
        residency_history_aligned(after),
{
    assert forall|rid: RequestId|
        #[trigger] after.running@.contains(rid)
            && after.live_requests@.contains_key(rid)
        implies {
            let hist = history(after.live_requests@[rid]).len() as int;
            let ids = after.request_residency@[rid].block_ids@;
            let bs = BLOCK_SIZE_SPEC as int;
            &&& after.request_residency@.contains_key(rid)
            &&& ids.len() >= 1
            &&& after.blocks@.contains_key(ids[ids.len() - 1])
            &&& hist == (ids.len() - 1) * bs
                + after.blocks@[ids[ids.len() - 1]].tokens@.len()
            &&& after.blocks@[ids[ids.len() - 1]].tokens@.len() >= 1
            &&& token_placement_prefix(
                after.blocks@, ids,
                history(after.live_requests@[rid]), hist,
            )
            &&& (after.blocks@[ids[ids.len() - 1]].refcount == 1
                || after.blocks@[ids[ids.len() - 1]].tokens@.len()
                    == BLOCK_SIZE_SPEC as int)
        }
    by {
        assert(before.running@.contains(rid));
        assert(before.live_requests@.contains_key(rid));
        assert(before.request_residency@.contains_key(rid));
        let ids = before.request_residency@[rid].block_ids@;
        let hist = history(before.live_requests@[rid]).len() as int;
        assert(ids.len() >= 1);
        let tail = ids[ids.len() - 1];
        lemma_resident_block_has_positive_refcount(before, rid, tail);
        assert(after.blocks@[tail] == before.blocks@[tail]);
        assert(token_placement_prefix(
            after.blocks@, ids,
            history(after.live_requests@[rid]), hist,
        )) by {
            assert forall|p: int|
                #![trigger history(after.live_requests@[rid])[p]]
                #![trigger token_placement_at(
                    after.blocks@, ids,
                    history(after.live_requests@[rid]), p,
                )]
                0 <= p < hist
                implies token_placement_at(
                    after.blocks@, ids,
                    history(after.live_requests@[rid]), p,
                )
            by {
                let j = p / (BLOCK_SIZE_SPEC as int);
                let bid = ids[j];
                assert(token_placement_at(
                    before.blocks@, ids,
                    history(before.live_requests@[rid]), p,
                ));
                assert(ids.contains(bid));
                lemma_resident_block_has_positive_refcount(before, rid, bid);
                assert(after.blocks@[bid] == before.blocks@[bid]);
            }
        }
    }
}

pub proof fn lemma_slot_mapping_aligned_frame(
    before: &CacheScheduler,
    after: &CacheScheduler,
)
    requires
        slot_mapping_aligned(before),
        after.request_residency@ == before.request_residency@,
        after.live_requests@ == before.live_requests@,
        after.running@ == before.running@,
    ensures
        slot_mapping_aligned(after),
{
}

// Admission is the one stable-boundary extension: allocation inserts exactly
// `rid` into the residency domain and the following queue update appends the
// same id to `running`.  Keeping the reveal inside this small lemma prevents
// the quantified definition from polluting the solver-sensitive plan proof.
pub proof fn lemma_residency_running_aligned_admit(
    before: &CacheScheduler,
    after: &CacheScheduler,
    rid: RequestId,
)
    requires
        residency_running_aligned(before),
        after.request_residency@.dom()
            == before.request_residency@.dom().insert(rid),
        after.running@ == before.running@.push(rid),
    ensures
        residency_running_aligned(after),
{
    reveal(residency_running_aligned);
    assert forall|r: RequestId|
        #[trigger] after.request_residency@.contains_key(r)
        implies after.running@.contains(r)
    by {
        if r == rid {
            assert(after.running@[after.running@.len() - 1] == rid);
        } else {
            assert(before.request_residency@.contains_key(r));
            assert(before.running@.contains(r));
            let q = before.running@.index_of(r);
            assert(after.running@[q] == r);
        }
    }
}

// Commit never creates residency.  Bystander running membership is framed;
// the caller separately shows that `rid` is still running whenever its
// residency survives (a finished request loses both).
pub proof fn lemma_residency_running_aligned_commit_one(
    before: &CacheScheduler,
    after: &CacheScheduler,
    rid: RequestId,
)
    requires
        residency_running_aligned(before),
        forall|r: RequestId|
            #[trigger] after.request_residency@.contains_key(r)
            ==> before.request_residency@.contains_key(r),
        forall|r: RequestId| r != rid && #[trigger] before.running@.contains(r)
            ==> after.running@.contains(r),
        after.request_residency@.contains_key(rid)
            ==> after.running@.contains(rid),
    ensures
        residency_running_aligned(after),
{
    reveal(residency_running_aligned);
    assert forall|r: RequestId|
        #[trigger] after.request_residency@.contains_key(r)
        implies after.running@.contains(r)
    by {
        assert(before.request_residency@.contains_key(r));
        assert(before.running@.contains(r));
    }
}

// Stable-boundary transport for parking one running request: `rid` loses its
// residency and queue membership, while every bystander keeps both.  The
// executable transition supplies the token frame and the stronger exact-tail
// frame.  Keeping the quantified reconstruction here avoids duplicating the
// companion-invariant proofs in cache-only commit and future preemption paths.
pub proof fn lemma_stable_companions_after_park(
    before: &CacheScheduler,
    after: &CacheScheduler,
    rid: RequestId,
)
    requires
        cs_valid(after),
        residency_history_aligned(before),
        slot_mapping_aligned(before),
        pre_commit_tails_exclusive(before),
        residency_running_aligned(before),
        after.live_requests@ == before.live_requests@,
        !after.running@.contains(rid),
        !after.request_residency@.contains_key(rid),
        forall|r: RequestId| #[trigger] after.running@.contains(r)
            ==> before.running@.contains(r),
        forall|r: RequestId| r != rid && #[trigger] before.running@.contains(r)
            ==> after.running@.contains(r),
        forall|r: RequestId| #[trigger] after.request_residency@.contains_key(r)
            ==> before.request_residency@.contains_key(r),
        forall|r: RequestId| r != rid
            && #[trigger] before.request_residency@.contains_key(r)
            ==> after.request_residency@.contains_key(r)
                && after.request_residency@[r] == before.request_residency@[r],
        forall|b: BlockId| #[trigger] after.blocks@.contains_key(b)
            ==> before.blocks@.contains_key(b)
                && after.blocks@[b].tokens@ == before.blocks@[b].tokens@,
        forall|r: RequestId| #[trigger] after.running@.contains(r)
            && after.live_requests@.contains_key(r)
            ==> {
                let ids = after.request_residency@[r].block_ids@;
                let tail = ids[ids.len() - 1];
                after.blocks@[tail] == before.blocks@[tail]
            },
    ensures
        residency_history_aligned(after),
        slot_mapping_aligned(after),
        pre_commit_tails_exclusive(after),
        residency_running_aligned(after),
{
    reveal(residency_running_aligned);
    assert forall|r: RequestId|
        #[trigger] after.request_residency@.contains_key(r)
        implies after.running@.contains(r)
    by {
        assert(before.request_residency@.contains_key(r));
        assert(before.running@.contains(r));
        assert(r != rid);
    }

    assert(slot_mapping_aligned(after)) by {
        assert forall|r: RequestId|
            #[trigger] after.running@.contains(r)
                && after.live_requests@.contains_key(r)
            implies {
                let hist = history(after.live_requests@[r]).len() as int;
                let ids = after.request_residency@[r].block_ids@;
                let sm = after.request_residency@[r].slot_mapping@;
                &&& after.request_residency@.contains_key(r)
                &&& sm.len() >= 1
                &&& hist >= 1
                &&& sm[sm.len() - 1] as int
                    == crate::proof::tensor::geometry::block_table_slot(ids, (hist - 1) as nat) as int
            }
        by {
            assert(r != rid);
            assert(before.running@.contains(r));
            assert(before.live_requests@.contains_key(r));
            assert(after.request_residency@[r] == before.request_residency@[r]);
        }
    }

    assert(residency_history_aligned(after)) by {
        assert forall|r: RequestId|
            #[trigger] after.running@.contains(r)
                && after.live_requests@.contains_key(r)
            implies {
                let hist = history(after.live_requests@[r]).len() as int;
                let ids = after.request_residency@[r].block_ids@;
                let bs = BLOCK_SIZE_SPEC as int;
                &&& after.request_residency@.contains_key(r)
                &&& ids.len() >= 1
                &&& after.blocks@.contains_key(ids[ids.len() - 1])
                &&& hist == (ids.len() - 1) * bs
                    + after.blocks@[ids[ids.len() - 1]].tokens@.len()
                &&& after.blocks@[ids[ids.len() - 1]].tokens@.len() >= 1
                &&& token_placement_prefix(
                    after.blocks@,
                    ids,
                    history(after.live_requests@[r]),
                    hist,
                )
                &&& (after.blocks@[ids[ids.len() - 1]].refcount == 1
                    || after.blocks@[ids[ids.len() - 1]].tokens@.len() == bs)
            }
        by {
            assert(r != rid);
            assert(before.running@.contains(r));
            assert(before.live_requests@.contains_key(r));
            assert(after.request_residency@[r] == before.request_residency@[r]);
            let ids = before.request_residency@[r].block_ids@;
            let hist = history(before.live_requests@[r]);
            assert(ids.len() >= 1);
            let tail = ids[ids.len() - 1];
            assert(after.blocks@[tail] == before.blocks@[tail]);
            assert forall|j: int| 0 <= j < ids.len()
                implies after.blocks@.contains_key(#[trigger] ids[j])
                    && after.blocks@[ids[j]].tokens@
                        == before.blocks@[ids[j]].tokens@
            by {
                assert(after.request_residency@.contains_key(r));
                assert(residency_blocks_in_range(after));
            }
            lemma_token_placement_prefix_transfer_for_ids(
                before.blocks@,
                after.blocks@,
                ids,
                hist,
                hist.len() as int,
            );
        }
    }

    assert(pre_commit_tails_exclusive(after)) by {
        assert forall|r: RequestId|
            #[trigger] after.running@.contains(r)
                && after.live_requests@.contains_key(r)
            implies {
                let ids = after.request_residency@[r].block_ids@;
                &&& after.request_residency@.contains_key(r)
                &&& ids.len() >= 1
                &&& after.blocks@.contains_key(ids[ids.len() - 1])
                &&& after.blocks@[ids[ids.len() - 1]].refcount == 1
                &&& (after.blocks@[ids[ids.len() - 1]].tokens@.len()
                        < BLOCK_SIZE_SPEC as int
                    ==> after.blocks@[ids[ids.len() - 1]].hash_value == 0)
            }
        by {
            assert(r != rid);
            assert(before.running@.contains(r));
            assert(before.live_requests@.contains_key(r));
            assert(after.request_residency@[r] == before.request_residency@[r]);
            let ids = after.request_residency@[r].block_ids@;
            let tail = ids[ids.len() - 1];
            assert(after.blocks@[tail] == before.blocks@[tail]);
        }
    }
}

// Boundary companion: at step boundaries every running request's
// tail block is exclusively owned (refcount 1) and provenance-free
// (prefix-depth and hash are both zero —
// append-created and partially-filled blocks are never stamped, and a full
// prompt tail moves off the tail position at the request's first commit).
// With the registry hygiene (`hash_to_block_no_zero`) hash 0 also means
// "not a registry target", so a plan's prefix matching can never bump it.
pub open spec fn tail_write_exclusive(cs: &CacheScheduler) -> bool {
    forall|rid: RequestId|
        #[trigger] cs.running@.contains(rid) && cs.live_requests@.contains_key(rid)
        ==> {
            let ids = cs.request_residency@[rid].block_ids@;
            &&& cs.request_residency@.contains_key(rid)
            &&& ids.len() >= 1
            &&& cs.blocks@.contains_key(ids[ids.len() - 1])
            &&& cs.blocks@[ids[ids.len() - 1]].refcount == 1
            &&& cs.blocks@[ids[ids.len() - 1]].prefix_depth == 0
            &&& cs.blocks@[ids[ids.len() - 1]].hash_value == 0
        }
}

// Mid-step (post-plan) weakening: every running tail is still refcount 1,
// and PARTIAL tails still carry hash 0 (a freshly admitted FULL prompt
// tail is registered, hence hash != 0, until its first commit replaces it).
pub open spec fn pre_commit_tails_exclusive(cs: &CacheScheduler) -> bool {
    forall|rid: RequestId|
        #[trigger] cs.running@.contains(rid) && cs.live_requests@.contains_key(rid)
        ==> {
            let ids = cs.request_residency@[rid].block_ids@;
            &&& cs.request_residency@.contains_key(rid)
            &&& ids.len() >= 1
            &&& cs.blocks@.contains_key(ids[ids.len() - 1])
            &&& cs.blocks@[ids[ids.len() - 1]].refcount == 1
            &&& (cs.blocks@[ids[ids.len() - 1]].tokens@.len()
                    < BLOCK_SIZE_SPEC as int
                ==> cs.blocks@[ids[ids.len() - 1]].hash_value == 0)
        }
}

// Per-row page exclusivity of an admitted (non-pre-running) request at plan
// exit: every SUFFIX page (the row's write targets) has refcount 1, and
// each one's hash is either 0 (unregistered) or recorded in the same-step
// exclusion list, so later admissions in the same plan cannot bump it.
pub open spec fn admitted_pages_exclusive_at(
    post: &CacheScheduler,
    sched: Seq<RequestId>,
    step_hashes: Seq<u64>,
    k: int,
) -> bool {
    let srid = sched[k];
    let ids = post.request_residency@[srid].block_ids@;
    let cpb = post.request_residency@[srid].cached_prefix_blocks as int;
    &&& post.request_residency@.contains_key(srid)
    &&& cpb < ids.len()
    &&& (forall|l: int| cpb <= l < ids.len()
        ==> post.blocks@.contains_key(#[trigger] ids[l])
            && post.blocks@[ids[l]].refcount == 1
            && (post.blocks@[ids[l]].hash_value == 0
                || step_hashes.contains(post.blocks@[ids[l]].hash_value))
            && (!step_hashes.contains(post.blocks@[ids[l]].hash_value)
                ==> post.blocks@[ids[l]].prefix_depth == 0
                    && post.blocks@[ids[l]].parent_block
                        == Option::<BlockId>::None))
    &&& ids.len() >= 1
    &&& (post.blocks@[ids[ids.len() - 1]].tokens@.len() < BLOCK_SIZE_SPEC as int
        ==> post.blocks@[ids[ids.len() - 1]].hash_value == 0)
}

// The pages predicate reads only residency + blocks.
pub proof fn lemma_admitted_pages_frame(
    a: &CacheScheduler,
    b: &CacheScheduler,
    sched: Seq<RequestId>,
    sh: Seq<u64>,
    k: int,
)
    requires
        admitted_pages_exclusive_at(a, sched, sh, k),
        b.request_residency@ == a.request_residency@,
        b.blocks@ == a.blocks@,
    ensures
        admitted_pages_exclusive_at(b, sched, sh, k),
{
    let srid = sched[k];
    let ids = b.request_residency@[srid].block_ids@;
    let cpb = b.request_residency@[srid].cached_prefix_blocks as int;
    assert forall|l: int| cpb <= l < ids.len()
        implies b.blocks@.contains_key(#[trigger] ids[l])
            && b.blocks@[ids[l]].refcount == 1
            && (b.blocks@[ids[l]].hash_value == 0
                || sh.contains(b.blocks@[ids[l]].hash_value))
            && (!sh.contains(b.blocks@[ids[l]].hash_value)
                ==> b.blocks@[ids[l]].prefix_depth == 0
                    && b.blocks@[ids[l]].parent_block
                        == Option::<BlockId>::None)
    by {
        assert(a.blocks@.contains_key(ids[l]));
    }
}

// Pressure reclamation can delete unrelated zero-ref pages.  The admitted
// suffix pages are all refcount one by definition, so the reclaimer's
// positive-page frame is sufficient to carry their exclusivity invariant.
pub proof fn lemma_admitted_pages_positive_frame(
    a: &CacheScheduler,
    b: &CacheScheduler,
    sched: Seq<RequestId>,
    sh: Seq<u64>,
    k: int,
)
    requires
        admitted_pages_exclusive_at(a, sched, sh, k),
        b.request_residency@ == a.request_residency@,
        forall|bid: BlockId| #[trigger] a.blocks@.contains_key(bid)
            && a.blocks@[bid].refcount > 0
            ==> b.blocks@.contains_key(bid)
                && b.blocks@[bid] == a.blocks@[bid],
    ensures
        admitted_pages_exclusive_at(b, sched, sh, k),
{
    let srid = sched[k];
    let ids = a.request_residency@[srid].block_ids@;
    let cpb = a.request_residency@[srid].cached_prefix_blocks as int;
    assert forall|l: int| cpb <= l < ids.len()
        implies b.blocks@.contains_key(#[trigger] ids[l])
            && b.blocks@[ids[l]].refcount == 1
            && (b.blocks@[ids[l]].hash_value == 0
                || sh.contains(b.blocks@[ids[l]].hash_value))
            && (!sh.contains(b.blocks@[ids[l]].hash_value)
                ==> b.blocks@[ids[l]].prefix_depth == 0
                    && b.blocks@[ids[l]].parent_block
                        == Option::<BlockId>::None)
    by {
        assert(a.blocks@[ids[l]].refcount == 1);
        assert(b.blocks@[ids[l]] == a.blocks@[ids[l]]);
    }
    let tail = ids[ids.len() - 1];
    assert(cpb <= ids.len() - 1);
    assert(a.blocks@[tail].refcount == 1);
    assert(b.blocks@[tail] == a.blocks@[tail]);
}

// Step-facing weakening of `admitted_pages_exclusive_at` (no exclusion-list
// clause — that one is plan-internal maintenance machinery).
pub open spec fn admitted_pages_exclusive_post(
    post: &CacheScheduler,
    sched: Seq<RequestId>,
    k: int,
) -> bool {
    let srid = sched[k];
    let ids = post.request_residency@[srid].block_ids@;
    let cpb = post.request_residency@[srid].cached_prefix_blocks as int;
    &&& post.request_residency@.contains_key(srid)
    &&& cpb < ids.len()
    &&& (forall|l: int| cpb <= l < ids.len()
        ==> post.blocks@.contains_key(#[trigger] ids[l])
            && post.blocks@[ids[l]].refcount == 1)
    &&& ids.len() >= 1
    &&& (post.blocks@[ids[ids.len() - 1]].tokens@.len() < BLOCK_SIZE_SPEC as int
        ==> post.blocks@[ids[ids.len() - 1]].hash_value == 0)
}

pub open spec fn admitted_registration_ready_at(
    cs: &CacheScheduler,
    scheduled: Seq<RequestId>,
    cu_k: Seq<u64>,
    k: int,
) -> bool {
    let rid = scheduled[k];
    let ids = cs.request_residency@[rid].block_ids@;
    let c = cs.request_residency@[rid].cached_prefix_blocks as int;
    let tokens = cs.live_requests@[rid].prompt_tokens@;
    let n = tokens.len() as int;
    let end = cu_k[k + 1] as int - cu_k[k] as int;
    &&& cs.running@.contains(rid)
    &&& cs.live_requests@.contains_key(rid)
    &&& valid_request_state(cs.live_requests@[rid])
    &&& cs.live_requests@[rid].generated_tokens@.len() == 0
    &&& cs.request_residency@.contains_key(rid)
    &&& ids.len() == blocks_needed_for(n as nat) as int
    &&& token_placement_prefix(cs.blocks@, ids, tokens, n)
    &&& 0 <= c
    &&& c * (BLOCK_SIZE_SPEC as int) < end <= n
    &&& blocks_needed_for(end as nat) <= u64::MAX as nat
    &&& registered_prefix_chain(cs.blocks@, ids.subrange(0, c))
    &&& n <= u64::MAX as int
    &&& n <= usize::MAX as int
    &&& admitted_pages_exclusive_at(
        cs, scheduled, Seq::<u64>::empty(), k,
    )
    &&& forall|h: u64| #[trigger] cs.hash_to_block@.contains_key(h)
        ==> !ids.subrange(c, ids.len() as int)
            .contains(cs.hash_to_block@[h])
}

pub proof fn lemma_refcount_one_excludes_other_holder(
    cs: &CacheScheduler,
    bid: BlockId,
    owner: RequestId,
    other: RequestId,
)
    requires
        cs_valid(cs),
        cs.request_residency@.contains_key(owner),
        cs.request_residency@[owner].block_ids@.contains(bid),
        cs.blocks@.contains_key(bid),
        cs.blocks@[bid].refcount == 1,
        other != owner,
    ensures
        !cs.request_residency@.contains_key(other)
            || !cs.request_residency@[other].block_ids@.contains(bid),
{
    if cs.request_residency@.contains_key(other)
        && cs.request_residency@[other].block_ids@.contains(bid) {
        let holders = residency_holders_of(cs, bid);
        assert(holders.contains(owner));
        assert(holders.contains(other));
        lemma_two_holders(holders, owner, other);
        assert(refcount_valid(cs));
        assert(cs.blocks@[bid].refcount as int == holders.len() as int);
        assert(false);
    }
}

// Entries still visible to matching (not in the same-step exclusion set) come
// from the pre-plan registry, with the immutable content/provenance fields
// unchanged. Refcounts may increase as later requests reuse a page. This is
// the direction needed for sound reuse; it does not claim every pre entry is
// still present.
#[verifier::opaque]
pub open spec fn registry_entries_from_pre(
    pre: &CacheScheduler,
    post: &CacheScheduler,
    excluded: Seq<u64>,
) -> bool {
    forall|h: u64|
        #[trigger] post.hash_to_block@.contains_key(h)
        && !excluded.contains(h) ==> {
            let bid = post.hash_to_block@[h];
            &&& pre.hash_to_block@.contains_key(h)
            &&& pre.hash_to_block@[h] == bid
            &&& pre.blocks@.contains_key(bid)
            &&& post.blocks@.contains_key(bid)
            &&& post.blocks@[bid].tokens@ == pre.blocks@[bid].tokens@
            &&& post.blocks@[bid].hash_value == pre.blocks@[bid].hash_value
            &&& post.blocks@[bid].prefix_depth == pre.blocks@[bid].prefix_depth
            &&& post.blocks@[bid].parent_block == pre.blocks@[bid].parent_block
        }
}

pub proof fn lemma_registry_entries_from_pre_transfer(
    pre: &CacheScheduler,
    before: &CacheScheduler,
    after: &CacheScheduler,
    excluded_before: Seq<u64>,
    excluded_after: Seq<u64>,
)
    requires
        registry_entries_from_pre(pre, before, excluded_before),
        forall|h: u64| excluded_before.contains(h) ==> excluded_after.contains(h),
        forall|h: u64| #[trigger] after.hash_to_block@.contains_key(h)
            && !excluded_after.contains(h) ==> {
                let bid = after.hash_to_block@[h];
                &&& before.hash_to_block@.contains_key(h)
                &&& before.hash_to_block@[h] == bid
                &&& before.blocks@.contains_key(bid)
                &&& after.blocks@.contains_key(bid)
                &&& after.blocks@[bid].tokens@ == before.blocks@[bid].tokens@
                &&& after.blocks@[bid].hash_value == before.blocks@[bid].hash_value
                &&& after.blocks@[bid].prefix_depth == before.blocks@[bid].prefix_depth
                &&& after.blocks@[bid].parent_block == before.blocks@[bid].parent_block
            },
    ensures
        registry_entries_from_pre(pre, after, excluded_after),
{
    reveal(registry_entries_from_pre);
}

pub proof fn lemma_registry_entries_from_pre_refl(cs: &CacheScheduler)
    requires
        hash_to_block_in_range(cs),
    ensures
        registry_entries_from_pre(cs, cs, Seq::<u64>::empty()),
{
    reveal(registry_entries_from_pre);
    assert forall|h: u64| #[trigger] cs.hash_to_block@.contains_key(h)
        && !Seq::<u64>::empty().contains(h) implies {
            let bid = cs.hash_to_block@[h];
            &&& cs.hash_to_block@.contains_key(h)
            &&& cs.hash_to_block@[h] == bid
            &&& cs.blocks@.contains_key(bid)
            &&& cs.blocks@.contains_key(bid)
            &&& cs.blocks@[bid].tokens@ == cs.blocks@[bid].tokens@
            &&& cs.blocks@[bid].hash_value == cs.blocks@[bid].hash_value
            &&& cs.blocks@[bid].prefix_depth == cs.blocks@[bid].prefix_depth
            &&& cs.blocks@[bid].parent_block == cs.blocks@[bid].parent_block
        } by {
        assert(hash_to_block_in_range(cs));
    }
}

// Registry-origin frames compose.  This is the scheduler-side analogue of a
// semantic frame rule: an entry surviving both transitions has the same target
// and immutable page metadata all the way back to the first state.
pub proof fn lemma_registry_entries_from_pre_transitive(
    pre: &CacheScheduler,
    mid: &CacheScheduler,
    post: &CacheScheduler,
    excluded: Seq<u64>,
)
    requires
        registry_entries_from_pre(pre, mid, excluded),
        registry_entries_from_pre(mid, post, excluded),
    ensures
        registry_entries_from_pre(pre, post, excluded),
{
    reveal(registry_entries_from_pre);
}

// Transitions that only change queues, live-request state, or other scheduler
// bookkeeping have a reflexive registry-origin frame.
pub proof fn lemma_registry_entries_from_pre_blocks_eq(
    before: &CacheScheduler,
    after: &CacheScheduler,
)
    requires
        hash_to_block_in_range(before),
        after.blocks@ == before.blocks@,
        after.hash_to_block@ == before.hash_to_block@,
    ensures
        registry_entries_from_pre(
            before, after, Seq::<u64>::empty(),
        ),
{
    reveal(registry_entries_from_pre);
}

// Publication-local origin.  Unlike the plan-entry predicate above, this uses
// the state immediately before deferred registration as its base and therefore
// distinguishes pages that physically survived planning from pages whose IDs
// were allocated for this forward.  Queue phase is intentionally absent: the
// caller converts the processed residency witnesses to stable admission rows.
#[verifier::opaque]
pub open spec fn positive_pages_from_publication_base_or_processed(
    base: &CacheScheduler,
    post: &CacheScheduler,
    scheduled: Seq<RequestId>,
    cu_k: Seq<int>,
    start: int,
    processed: int,
) -> bool {
    cu_k.len() == scheduled.len() + 1
    && forall|bid: BlockId|
        #[trigger] post.blocks@[bid].prefix_depth > 0
        && post.blocks@.contains_key(bid)
        && post.blocks@[bid].prefix_depth > 0
        ==> (base.blocks@.contains_key(bid)
                && base.blocks@[bid].prefix_depth > 0)
            || exists|k: int, l: int|
                0 <= start <= k < processed
                && k < scheduled.len()
                && post.live_requests@.contains_key(scheduled[k])
                && post.request_residency@.contains_key(scheduled[k])
                && 0 <= cu_k[k + 1] - cu_k[k]
                    <= post.live_requests@[scheduled[k]].prompt_tokens@.len()
                && 0 <= l < (cu_k[k + 1] - cu_k[k])
                    / (BLOCK_SIZE_SPEC as int)
                && #[trigger] post.request_residency@[scheduled[k]]
                    .block_ids@[l] == bid
}

// Chain-level engine boundary.  For any registered physical chain, the target
// is classified as either a full page below a named admission row's planned
// `k_len`, or as an
// inherited page whose complete ancestry is unchanged.  The cases are made
// disjoint on purpose: block IDs may be evicted and reused, and an admitted
// page can coincidentally end with metadata equal to the old occupant.  A
// downstream cache-value proof must still treat that physical lifetime as an
// admission, not frame the old K/V merely because the metadata compares equal.
// This also rules out the bad abstraction in which an old child certificate is
// paired with a reallocated parent block ID.
#[verifier::opaque]
pub open spec fn positive_chains_from_pre_or_admission_rows(
    pre: &CacheScheduler,
    post: &CacheScheduler,
    scheduled: Seq<RequestId>,
    block_rows: Seq<Seq<BlockId>>,
    cu_k: Seq<int>,
) -> bool {
    cu_k.len() == scheduled.len() + 1
    && forall|chain: Seq<BlockId>, j: int|
        #![trigger registered_prefix_chain(post.blocks@, chain), chain[j]]
        registered_prefix_chain(post.blocks@, chain)
        && 0 <= j < chain.len()
        ==> (exists|k: int, l: int|
                0 <= k < scheduled.len()
                && k < block_rows.len()
                && !pre.running@.contains(scheduled[k])
                && pre.live_requests@.contains_key(scheduled[k])
                && 0 <= cu_k[k + 1] - cu_k[k]
                    <= pre.live_requests@[scheduled[k]].prompt_tokens@.len()
                && 0 <= l < (cu_k[k + 1] - cu_k[k])
                    / (BLOCK_SIZE_SPEC as int)
                && #[trigger] block_rows[k][l] == chain[j])
            || ((forall|l: int| 0 <= l <= j ==>
                    #[trigger] positive_page_unchanged_from_pre(
                        pre, post, chain[l],
                    ))
                && !(exists|k: int, l: int|
                    0 <= k < scheduled.len()
                    && k < block_rows.len()
                    && !pre.running@.contains(scheduled[k])
                    && pre.live_requests@.contains_key(scheduled[k])
                    && 0 <= cu_k[k + 1] - cu_k[k]
                        <= pre.live_requests@[scheduled[k]].prompt_tokens@.len()
                    && 0 <= l < (cu_k[k + 1] - cu_k[k])
                        / (BLOCK_SIZE_SPEC as int)
                    && #[trigger] block_rows[k][l] == chain[j]))
}

// Stable semantic certificate for the complete pages published below each
// admission row's planned `k_len`.  Those pages carry the registered chain and
// exact prompt tokens.  Any trailing partial page is provenance-free when
// it remains allocated; commit may append the generated token there, but may
// not turn it into a reusable prefix page.  The latter negative fact is needed
// to rule out confusing an admission write with an inherited positive page.
#[verifier::opaque]
pub open spec fn published_admission_row_prefixes(
    pre: &CacheScheduler,
    post: &CacheScheduler,
    scheduled: Seq<RequestId>,
    block_rows: Seq<Seq<BlockId>>,
    cu_k: Seq<int>,
) -> bool {
    block_rows.len() == scheduled.len()
    && cu_k.len() == scheduled.len() + 1
    && forall|k: int| 0 <= k < scheduled.len()
        && !pre.running@.contains(#[trigger] scheduled[k])
        ==> pre.live_requests@.contains_key(scheduled[k]) && {
            let prompt = pre.live_requests@[scheduled[k]].prompt_tokens@;
            let end = cu_k[k + 1] - cu_k[k];
            let full = end / (BLOCK_SIZE_SPEC as int);
            let ids = block_rows[k];
            &&& 0 <= full <= ids.len()
            &&& 0 <= end <= prompt.len()
            &&& prompt.len() <= u64::MAX as int
            &&& blocks_needed_for(end as nat) <= u64::MAX as nat
            &&& registered_prefix_chain(
                post.blocks@, ids.subrange(0, full),
            )
            &&& token_placement_prefix(
                post.blocks@, ids, prompt,
                full * (BLOCK_SIZE_SPEC as int),
            )
            &&& forall|l: int| full <= l < ids.len()
                && #[trigger] post.blocks@.contains_key(ids[l])
                ==> post.blocks@[ids[l]].prefix_depth == 0
        }
}

pub proof fn lemma_published_admission_row_prefixes_frame(
    pre: &CacheScheduler,
    before: &CacheScheduler,
    after: &CacheScheduler,
    scheduled: Seq<RequestId>,
    block_rows: Seq<Seq<BlockId>>,
    cu_k: Seq<int>,
)
    requires
        published_admission_row_prefixes(
            pre, before, scheduled, block_rows, cu_k,
        ),
        positive_provenance_metadata_frame(before, after),
        positive_provenance_origin(before, after),
    ensures
        published_admission_row_prefixes(
            pre, after, scheduled, block_rows, cu_k,
        ),
{
    reveal(published_admission_row_prefixes);
    reveal(positive_provenance_metadata_frame);
    reveal(positive_provenance_origin);
    assert forall|k: int| 0 <= k < scheduled.len()
        && !pre.running@.contains(#[trigger] scheduled[k])
        implies pre.live_requests@.contains_key(scheduled[k]) && {
            let prompt = pre.live_requests@[scheduled[k]].prompt_tokens@;
            let end = cu_k[k + 1] - cu_k[k];
            let full = end / (BLOCK_SIZE_SPEC as int);
            let ids = block_rows[k];
            &&& 0 <= full <= ids.len()
            &&& 0 <= end <= prompt.len()
            &&& prompt.len() <= u64::MAX as int
            &&& blocks_needed_for(end as nat) <= u64::MAX as nat
            &&& registered_prefix_chain(
                after.blocks@, ids.subrange(0, full),
            )
            &&& token_placement_prefix(
                after.blocks@, ids, prompt,
                full * (BLOCK_SIZE_SPEC as int),
            )
            &&& forall|l: int| full <= l < ids.len()
                && #[trigger] after.blocks@.contains_key(ids[l])
                ==> after.blocks@[ids[l]].prefix_depth == 0
        }
    by {
        let prompt = pre.live_requests@[scheduled[k]].prompt_tokens@;
        let end = cu_k[k + 1] - cu_k[k];
        let full = end / (BLOCK_SIZE_SPEC as int);
        let ids = block_rows[k];
        let chain = ids.subrange(0, full);
        assert(registered_prefix_chain(before.blocks@, chain));
        assert forall|j: int| 0 <= j < chain.len() implies {
            let bid = #[trigger] chain[j];
            &&& after.blocks@.contains_key(bid)
            &&& after.blocks@[bid].prefix_depth
                == before.blocks@[bid].prefix_depth
            &&& after.blocks@[bid].parent_block
                == before.blocks@[bid].parent_block
        }
        by {
            assert(chain[j] == ids[j]);
            assert(before.blocks@[chain[j]].prefix_depth as int == j + 1);
            assert(before.blocks@.contains_key(chain[j]));
            assert(before.blocks@[chain[j]].prefix_depth > 0);
        }
        lemma_registered_prefix_chain_transfer(
            before.blocks@, after.blocks@, chain,
        );
        let limit = full * (BLOCK_SIZE_SPEC as int);
        assert(limit <= prompt.len()) by {
            vstd::arithmetic::div_mod::lemma_fundamental_div_mod(
                end, BLOCK_SIZE_SPEC as int,
            );
        }
        assert(token_placement_prefix(after.blocks@, ids, prompt, limit)) by {
            assert forall|p: int|
                #![trigger token_placement_at(after.blocks@, ids, prompt, p)]
                0 <= p < limit
                implies token_placement_at(after.blocks@, ids, prompt, p)
            by {
                lemma_token_placement_prefix_at(
                    before.blocks@, ids, prompt, limit, p,
                );
                reveal(token_placement_at);
                let j = p / (BLOCK_SIZE_SPEC as int);
                assert(0 <= j < full) by {
                    vstd::arithmetic::div_mod::lemma_div_is_ordered(
                        p, limit, BLOCK_SIZE_SPEC as int,
                    );
                }
                assert(chain[j] == ids[j]);
                assert(before.blocks@[ids[j]].prefix_depth as int == j + 1);
                assert(before.blocks@[ids[j]].prefix_depth > 0);
            }
        }
        assert forall|l: int| full <= l < ids.len()
            && #[trigger] after.blocks@.contains_key(ids[l])
            implies after.blocks@[ids[l]].prefix_depth == 0
        by {
            if after.blocks@[ids[l]].prefix_depth > 0 {
                reveal(positive_provenance_origin);
                assert(before.blocks@.contains_key(ids[l]));
                assert(after.blocks@[ids[l]].prefix_depth
                    == before.blocks@[ids[l]].prefix_depth);
                assert(before.blocks@[ids[l]].prefix_depth == 0);
            }
        }
    }
}

// Convert the publication loop's transient residency facts into the stable
// materialized-row predicate.  Spinning this proof off keeps the large plan
// verifier from mixing token-placement quantifiers with unrelated layout
// obligations later in the function.
#[verifier::spinoff_prover]
pub proof fn lemma_published_admission_row_prefixes_from_publication(
    pre: &CacheScheduler,
    post: &CacheScheduler,
    scheduled: Seq<RequestId>,
    block_rows: Seq<Seq<BlockId>>,
    cu_k: Seq<int>,
    decode_count: int,
)
    requires
        0 <= decode_count <= scheduled.len(),
        block_rows.len() == scheduled.len(),
        cu_k.len() == scheduled.len() + 1,
        post.live_requests@ == pre.live_requests@,
        forall|k: int| 0 <= k < scheduled.len()
            ==> block_rows[k]
                == #[trigger] post.request_residency@[scheduled[k]].block_ids@,
        forall|k: int| 0 <= k < scheduled.len()
            && !pre.running@.contains(#[trigger] scheduled[k])
            ==> decode_count <= k,
        forall|k: int| decode_count <= k < scheduled.len()
            ==> post.request_residency@.contains_key(#[trigger] scheduled[k])
                && post.live_requests@.contains_key(scheduled[k])
                && {
                let rid = #[trigger] scheduled[k];
                let ids = post.request_residency@[rid].block_ids@;
                let prompt = post.live_requests@[rid].prompt_tokens@;
                let end = cu_k[k + 1] - cu_k[k];
                let full = end / (BLOCK_SIZE_SPEC as int);
                &&& 0 <= full <= ids.len()
                &&& 0 <= end <= prompt.len()
                &&& prompt.len() <= u64::MAX as int
                &&& blocks_needed_for(end as nat) <= u64::MAX as nat
                &&& registered_prefix_chain(
                    post.blocks@, ids.subrange(0, full),
                )
                &&& token_placement_prefix(
                    post.blocks@, ids, prompt,
                    full * (BLOCK_SIZE_SPEC as int),
                )
                &&& forall|l: int| full <= l < ids.len()
                    && #[trigger] post.blocks@.contains_key(ids[l])
                    ==> post.blocks@[ids[l]].prefix_depth == 0
            },
    ensures
        published_admission_row_prefixes(
            pre, post, scheduled, block_rows, cu_k,
        ),
{
    reveal(published_admission_row_prefixes);
    assert forall|k: int| 0 <= k < scheduled.len()
        && !pre.running@.contains(#[trigger] scheduled[k])
        implies pre.live_requests@.contains_key(scheduled[k]) && {
            let prompt = pre.live_requests@[scheduled[k]].prompt_tokens@;
            let end = cu_k[k + 1] - cu_k[k];
            let full = end / (BLOCK_SIZE_SPEC as int);
            let ids = block_rows[k];
            &&& 0 <= full <= ids.len()
            &&& 0 <= end <= prompt.len()
            &&& prompt.len() <= u64::MAX as int
            &&& blocks_needed_for(end as nat) <= u64::MAX as nat
            &&& registered_prefix_chain(
                post.blocks@, ids.subrange(0, full),
            )
            &&& token_placement_prefix(
                post.blocks@, ids, prompt,
                full * (BLOCK_SIZE_SPEC as int),
            )
            &&& forall|l: int| full <= l < ids.len()
                && #[trigger] post.blocks@.contains_key(ids[l])
                ==> post.blocks@[ids[l]].prefix_depth == 0
        }
    by {
        let rid = scheduled[k];
        let prompt = pre.live_requests@[rid].prompt_tokens@;
        let end = cu_k[k + 1] - cu_k[k];
        let full = end / (BLOCK_SIZE_SPEC as int);
        let ids = block_rows[k];
        assert(decode_count <= k);
        assert(post.live_requests@.contains_key(rid));
        assert(pre.live_requests@.contains_key(rid));
        assert(post.request_residency@.contains_key(rid));
        assert(ids == post.request_residency@[rid].block_ids@);
        assert(0 <= full <= ids.len());
        assert(registered_prefix_chain(
            post.blocks@, ids.subrange(0, full),
        ));
        assert(full * (BLOCK_SIZE_SPEC as int) <= end) by {
            vstd::arithmetic::div_mod::lemma_fundamental_div_mod(
                end, BLOCK_SIZE_SPEC as int,
            );
        }
        assert(token_placement_prefix(
            post.blocks@, ids, prompt,
            full * (BLOCK_SIZE_SPEC as int),
        )) by {
            reveal(token_placement_prefix);
            assert forall|p: int|
                #![trigger token_placement_at(
                    post.blocks@, ids, prompt, p)]
                0 <= p < full * (BLOCK_SIZE_SPEC as int)
                implies token_placement_at(
                    post.blocks@, ids, prompt, p,
                )
            by {
                assert(p < end);
            }
        }
    }
}

pub proof fn lemma_publication_positive_pages_base(
    base: &CacheScheduler,
    scheduled: Seq<RequestId>,
    cu_k: Seq<int>,
    start: int,
)
    requires
        cu_k.len() == scheduled.len() + 1,
    ensures
        positive_pages_from_publication_base_or_processed(
            base, base, scheduled, cu_k, start, start,
        ),
{
    reveal(positive_pages_from_publication_base_or_processed);
}

#[verifier::spinoff_prover]
pub proof fn lemma_publication_positive_pages_register(
    base: &CacheScheduler,
    before: &CacheScheduler,
    after: &CacheScheduler,
    scheduled: Seq<RequestId>,
    cu_k: Seq<int>,
    start: int,
    processed: int,
    rid: RequestId,
    prompt_len: int,
    skip: int,
)
    requires
        positive_pages_from_publication_base_or_processed(
            base, before, scheduled, cu_k, start, processed,
        ),
        cu_k.len() == scheduled.len() + 1,
        0 <= start <= processed < scheduled.len(),
        scheduled[processed] == rid,
        before.live_requests@ == after.live_requests@,
        before.request_residency@ == after.request_residency@,
        before.live_requests@.contains_key(rid),
        before.request_residency@.contains_key(rid),
        0 <= prompt_len
            <= before.live_requests@[rid].prompt_tokens@.len(),
        prompt_len == cu_k[processed + 1] - cu_k[processed],
        skip == before.request_residency@[rid].cached_prefix_blocks as int,
        registration_positive_page_origin(
            before, after, rid, prompt_len, skip,
        ),
    ensures
        positive_pages_from_publication_base_or_processed(
            base, after, scheduled, cu_k, start, processed + 1,
        ),
{
    reveal(positive_pages_from_publication_base_or_processed);
    reveal(registration_positive_page_origin);
    assert forall|bid: BlockId|
        #[trigger] after.blocks@[bid].prefix_depth > 0
        && after.blocks@.contains_key(bid)
        && after.blocks@[bid].prefix_depth > 0
        implies (base.blocks@.contains_key(bid)
                && base.blocks@[bid].prefix_depth > 0)
            || exists|k: int, l: int|
                0 <= start <= k < processed + 1
                && k < scheduled.len()
                && after.live_requests@.contains_key(scheduled[k])
                && after.request_residency@.contains_key(scheduled[k])
                && 0 <= l < after.live_requests@[scheduled[k]]
                    .prompt_tokens@.len() as int / (BLOCK_SIZE_SPEC as int)
                && #[trigger] after.request_residency@[scheduled[k]]
                    .block_ids@[l] == bid
    by {
        let ids = before.request_residency@[rid].block_ids@;
        let full = prompt_len / (BLOCK_SIZE_SPEC as int);
        if positive_page_unchanged_from_pre(before, after, bid) {
            assert(before.blocks@.contains_key(bid));
            assert(before.blocks@[bid].prefix_depth > 0);
            if !(base.blocks@.contains_key(bid)
                    && base.blocks@[bid].prefix_depth > 0) {
                assert(exists|k: int, l: int|
                    0 <= start <= k < processed
                    && k < scheduled.len()
                    && before.live_requests@.contains_key(scheduled[k])
                    && before.request_residency@.contains_key(scheduled[k])
                    && 0 <= l < before.live_requests@[scheduled[k]]
                        .prompt_tokens@.len() as int / (BLOCK_SIZE_SPEC as int)
                    && before.request_residency@[scheduled[k]]
                        .block_ids@[l] == bid);
            }
        } else {
            assert(ids.subrange(skip, full).contains(bid));
            let q = ids.subrange(skip, full).index_of(bid);
            let l = skip + q;
            assert(skip <= l < full);
            assert(ids[l] == bid);
            assert(after.request_residency@[rid].block_ids@[l] == bid);
        }
    }
}

pub proof fn lemma_publication_inherited_chain_prefix(
    base: &CacheScheduler,
    post: &CacheScheduler,
    chain: Seq<BlockId>,
    j: int,
)
    requires
        persistent_provenance_closed(base),
        persistent_provenance_closed(post),
        positive_provenance_metadata_frame(base, post),
        post.blocks@.dom() == base.blocks@.dom(),
        registered_prefix_chain(post.blocks@, chain),
        0 <= j < chain.len(),
        base.blocks@.contains_key(chain[j]),
        base.blocks@[chain[j]].prefix_depth > 0,
    ensures
        forall|l: int| 0 <= l <= j ==>
            #[trigger] positive_page_unchanged_from_pre(
                base, post, chain[l],
            ),
    decreases j,
{
    reveal(positive_provenance_metadata_frame);
    reveal(persistent_provenance_closed);
    reveal(registered_prefix_chain);
    let bid = chain[j];
    assert(post.blocks@[bid].prefix_depth as int == j + 1);
    assert(post.blocks@.contains_key(bid));
    assert(post.blocks@[bid].tokens@ == base.blocks@[bid].tokens@);
    assert(post.blocks@[bid].hash_value == base.blocks@[bid].hash_value);
    assert(post.blocks@[bid].prefix_depth == base.blocks@[bid].prefix_depth);
    assert(post.blocks@[bid].parent_block == base.blocks@[bid].parent_block);
    assert(positive_page_unchanged_from_pre(base, post, bid));
    if j > 0 {
        assert(base.blocks@[bid].prefix_depth > 1);
        assert(post.blocks@[bid].parent_block == Some(chain[j - 1]));
        assert(base.blocks@[bid].parent_block == Some(chain[j - 1]));
        assert(post.blocks@.contains_key(chain[j - 1]));
        assert(base.blocks@.contains_key(chain[j - 1]));
        assert(base.blocks@[chain[j - 1]].prefix_depth > 0);
        lemma_publication_inherited_chain_prefix(
            base, post, chain, j - 1,
        );
    }
    assert forall|l: int| 0 <= l <= j implies
        #[trigger] positive_page_unchanged_from_pre(
            base, post, chain[l],
        )
    by {
        if l == j {
        } else {
            assert(l <= j - 1);
        }
    }
}

#[verifier::spinoff_prover]
#[verifier::rlimit(300)]
pub proof fn lemma_positive_chains_from_publication(
    pre: &CacheScheduler,
    base: &CacheScheduler,
    post: &CacheScheduler,
    scheduled: Seq<RequestId>,
    block_rows: Seq<Seq<BlockId>>,
    cu_k: Seq<int>,
    start: int,
)
    requires
        persistent_provenance_closed(base),
        persistent_provenance_closed(post),
        positive_provenance_origin(pre, base),
        positive_provenance_metadata_frame(base, post),
        post.blocks@.dom() == base.blocks@.dom(),
        positive_pages_from_publication_base_or_processed(
            base, post, scheduled, cu_k, start, scheduled.len() as int,
        ),
        0 <= start <= scheduled.len(),
        block_rows.len() == scheduled.len(),
        cu_k.len() == scheduled.len() + 1,
        post.live_requests@ == pre.live_requests@,
        forall|k: int| start <= k < scheduled.len()
            ==> !pre.running@.contains(#[trigger] scheduled[k])
                && pre.live_requests@.contains_key(scheduled[k]),
        forall|k: int| start <= k < scheduled.len()
            ==> post.request_residency@.contains_key(#[trigger] scheduled[k])
                && block_rows[k]
                    == post.request_residency@[scheduled[k]].block_ids@,
        forall|k: int| start <= k < scheduled.len()
            ==> 0 <= cu_k[k + 1] - cu_k[k]
                <= post.live_requests@[#[trigger] scheduled[k]]
                    .prompt_tokens@.len(),
    ensures
        positive_chains_from_pre_or_admission_rows(
            pre, post, scheduled, block_rows, cu_k,
        ),
{
    reveal(positive_chains_from_pre_or_admission_rows);
    reveal(positive_pages_from_publication_base_or_processed);
    reveal(positive_provenance_origin);
    reveal(positive_provenance_metadata_frame);
    assert forall|chain: Seq<BlockId>, j: int|
        #![trigger registered_prefix_chain(post.blocks@, chain), chain[j]]
        registered_prefix_chain(post.blocks@, chain)
        && 0 <= j < chain.len()
        implies (forall|l: int| 0 <= l <= j ==>
                #[trigger] positive_page_unchanged_from_pre(
                    pre, post, chain[l],
                ))
            || exists|k: int, l: int|
                0 <= k < scheduled.len()
                && k < block_rows.len()
                && !pre.running@.contains(scheduled[k])
                && pre.live_requests@.contains_key(scheduled[k])
                && 0 <= cu_k[k + 1] - cu_k[k]
                    <= pre.live_requests@[scheduled[k]].prompt_tokens@.len()
                && 0 <= l < (cu_k[k + 1] - cu_k[k])
                    / (BLOCK_SIZE_SPEC as int)
                && #[trigger] block_rows[k][l] == chain[j]
    by {
        let bid = chain[j];
        reveal(registered_prefix_chain);
        assert(post.blocks@[bid].prefix_depth as int == j + 1);
        assert(post.blocks@.contains_key(bid));
        assert(post.blocks@[bid].prefix_depth > 0);
        if base.blocks@.contains_key(bid)
            && base.blocks@[bid].prefix_depth > 0 {
            lemma_publication_inherited_chain_prefix(
                base, post, chain, j,
            );
            assert forall|l: int| 0 <= l <= j implies
                #[trigger] positive_page_unchanged_from_pre(
                    pre, post, chain[l],
                )
            by {
                assert(positive_page_unchanged_from_pre(
                    base, post, chain[l],
                ));
                assert(base.blocks@.contains_key(chain[l]));
                assert(base.blocks@[chain[l]].prefix_depth > 0);
                assert(pre.blocks@.contains_key(chain[l]));
            }
        } else {
            assert(exists|k: int, l: int|
                0 <= start <= k < scheduled.len()
                && post.live_requests@.contains_key(scheduled[k])
                && post.request_residency@.contains_key(scheduled[k])
                && 0 <= cu_k[k + 1] - cu_k[k]
                    <= post.live_requests@[scheduled[k]].prompt_tokens@.len()
                && 0 <= l < (cu_k[k + 1] - cu_k[k])
                    / (BLOCK_SIZE_SPEC as int)
                && post.request_residency@[scheduled[k]].block_ids@[l]
                    == bid);
        }
    }
}


// Appending into a partial tail cannot rewrite a registry target: every
// registry target is a full page, whereas the unique page changed by this
// transition was partial in the pre-state.
pub proof fn lemma_registry_entries_from_pre_tail_append(
    pre: &CacheScheduler,
    post: &CacheScheduler,
    rid: RequestId,
    token: TokenId,
)
    requires
        cs_valid(pre),
        append_token_tail_append_success(pre, post, rid, token),
    ensures
        registry_entries_from_pre(pre, post, Seq::<u64>::empty()),
{
    reveal(append_token_tail_append_success);
    reveal(append_token_common_frame);
    reveal(registry_entries_from_pre);
    let residency = pre.request_residency@[rid];
    let last_bid = residency.block_ids@[residency.block_ids@.len() - 1];
    assert forall|h: u64| #[trigger] post.hash_to_block@.contains_key(h)
        && !Seq::<u64>::empty().contains(h) implies {
            let bid = post.hash_to_block@[h];
            &&& pre.hash_to_block@.contains_key(h)
            &&& pre.hash_to_block@[h] == bid
            &&& pre.blocks@.contains_key(bid)
            &&& post.blocks@.contains_key(bid)
            &&& post.blocks@[bid].tokens@ == pre.blocks@[bid].tokens@
            &&& post.blocks@[bid].hash_value == pre.blocks@[bid].hash_value
            &&& post.blocks@[bid].prefix_depth == pre.blocks@[bid].prefix_depth
            &&& post.blocks@[bid].parent_block == pre.blocks@[bid].parent_block
        }
    by {
        let bid = post.hash_to_block@[h];
        assert(pre.hash_to_block@.contains_key(h));
        assert(pre.hash_to_block@[h] == bid);
        assert(pre.blocks@.contains_key(bid));
        assert(bid != last_bid) by {
            if bid == last_bid {
                assert(hash_to_block_in_range(pre));
                assert(pre.blocks@[bid].tokens@.len() == BLOCK_SIZE_SPEC as int);
                assert(pre.blocks@[last_bid].tokens@.len() < BLOCK_SIZE_SPEC as int);
            }
        }
        assert(post.blocks@.contains_key(bid));
        assert(post.blocks@[bid] == pre.blocks@[bid]);
    }
}

// Allocating a fresh partial tail also leaves every old registry target
// verbatim; the new page is not in the unchanged hash registry.
pub proof fn lemma_registry_entries_from_pre_new_tail(
    pre: &CacheScheduler,
    post: &CacheScheduler,
    rid: RequestId,
    token: TokenId,
)
    requires
        cs_valid(pre),
        append_token_new_tail_success(pre, post, rid, token),
    ensures
        registry_entries_from_pre(pre, post, Seq::<u64>::empty()),
{
    reveal(append_token_new_tail_success);
    reveal(append_token_common_frame);
    reveal(registry_entries_from_pre);
    let residency = pre.request_residency@[rid];
    let new_bid = post.request_residency@[rid].block_ids@[
        residency.block_ids@.len() as int
    ];
    assert forall|h: u64| #[trigger] post.hash_to_block@.contains_key(h)
        && !Seq::<u64>::empty().contains(h) implies {
            let bid = post.hash_to_block@[h];
            &&& pre.hash_to_block@.contains_key(h)
            &&& pre.hash_to_block@[h] == bid
            &&& pre.blocks@.contains_key(bid)
            &&& post.blocks@.contains_key(bid)
            &&& post.blocks@[bid].tokens@ == pre.blocks@[bid].tokens@
            &&& post.blocks@[bid].hash_value == pre.blocks@[bid].hash_value
            &&& post.blocks@[bid].prefix_depth == pre.blocks@[bid].prefix_depth
            &&& post.blocks@[bid].parent_block == pre.blocks@[bid].parent_block
        }
    by {
        let bid = post.hash_to_block@[h];
        assert(pre.hash_to_block@.contains_key(h));
        assert(pre.hash_to_block@[h] == bid);
        assert(pre.blocks@.contains_key(bid));
        assert(bid != new_bid);
        assert(post.blocks@.contains_key(bid));
        assert(post.blocks@[bid] == pre.blocks@[bid]);
    }
}

// A successful reuse scan may bump refcounts and add fresh suffix pages, but
// it does not rewrite the registry or any pre-existing page's immutable
// content/provenance fields.  Therefore the loop-carried registry origin is
// unchanged across allocation.
pub proof fn lemma_registry_entries_from_pre_allocate(
    pre: &CacheScheduler,
    before: &CacheScheduler,
    after: &CacheScheduler,
    excluded: Seq<u64>,
)
    requires
        registry_entries_from_pre(pre, before, excluded),
        after.hash_to_block@ == before.hash_to_block@,
        forall|bid: BlockId| #[trigger] before.blocks@.contains_key(bid)
            ==> after.blocks@.contains_key(bid)
                && after.blocks@[bid].tokens@ == before.blocks@[bid].tokens@
                && after.blocks@[bid].hash_value == before.blocks@[bid].hash_value
                && after.blocks@[bid].prefix_depth == before.blocks@[bid].prefix_depth
                && after.blocks@[bid].parent_block == before.blocks@[bid].parent_block,
    ensures
        registry_entries_from_pre(pre, after, excluded),
{
    reveal(registry_entries_from_pre);
    assert forall|h: u64| #[trigger] after.hash_to_block@.contains_key(h)
        && !excluded.contains(h) implies {
            let bid = after.hash_to_block@[h];
            &&& before.hash_to_block@.contains_key(h)
            &&& before.hash_to_block@[h] == bid
            &&& before.blocks@.contains_key(bid)
            &&& after.blocks@.contains_key(bid)
            &&& after.blocks@[bid].tokens@ == before.blocks@[bid].tokens@
            &&& after.blocks@[bid].hash_value == before.blocks@[bid].hash_value
            &&& after.blocks@[bid].prefix_depth == before.blocks@[bid].prefix_depth
            &&& after.blocks@[bid].parent_block == before.blocks@[bid].parent_block
        } by {
        assert(before.hash_to_block@.contains_key(h));
        let bid = after.hash_to_block@[h];
        assert(before.hash_to_block@[h] == bid);
        assert(before.blocks@.contains_key(bid));
        assert(after.blocks@.contains_key(bid));
        assert(after.blocks@[bid].tokens@ == before.blocks@[bid].tokens@);
        assert(after.blocks@[bid].hash_value == before.blocks@[bid].hash_value);
        assert(after.blocks@[bid].prefix_depth == before.blocks@[bid].prefix_depth);
        assert(after.blocks@[bid].parent_block == before.blocks@[bid].parent_block);
    }
    lemma_registry_entries_from_pre_transfer(
        pre, before, after, excluded, excluded,
    );
}

// Pressure reclamation may delete zero-ref registry entries and their pages,
// but every surviving entry still names the same, verbatim page.  Since the
// loop invariant is deliberately one-directional (post entries originate in
// the plan-entry registry), deletion composes without adding an exclusion.
pub proof fn lemma_registry_entries_from_pre_reclaim(
    pre: &CacheScheduler,
    before: &CacheScheduler,
    after: &CacheScheduler,
    excluded: Seq<u64>,
)
    requires
        registry_entries_from_pre(pre, before, excluded),
        hash_to_block_in_range(after),
        forall|h: u64| #[trigger] after.hash_to_block@.contains_key(h)
            ==> before.hash_to_block@.contains_key(h)
                && after.hash_to_block@[h] == before.hash_to_block@[h],
        forall|bid: BlockId| #[trigger] after.blocks@.contains_key(bid)
            ==> before.blocks@.contains_key(bid)
                && after.blocks@[bid] == before.blocks@[bid],
    ensures
        registry_entries_from_pre(pre, after, excluded),
{
    assert forall|h: u64| #[trigger] after.hash_to_block@.contains_key(h)
        && !excluded.contains(h) implies {
            let bid = after.hash_to_block@[h];
            &&& before.hash_to_block@.contains_key(h)
            &&& before.hash_to_block@[h] == bid
            &&& before.blocks@.contains_key(bid)
            &&& after.blocks@.contains_key(bid)
            &&& after.blocks@[bid].tokens@ == before.blocks@[bid].tokens@
            &&& after.blocks@[bid].hash_value == before.blocks@[bid].hash_value
            &&& after.blocks@[bid].prefix_depth == before.blocks@[bid].prefix_depth
            &&& after.blocks@[bid].parent_block == before.blocks@[bid].parent_block
        }
    by {
        let bid = after.hash_to_block@[h];
        assert(before.hash_to_block@[h] == bid);
        assert(hash_to_block_in_range(after));
        assert(after.blocks@.contains_key(bid));
        assert(after.blocks@[bid] == before.blocks@[bid]);
    }
    lemma_registry_entries_from_pre_transfer(
        pre, before, after, excluded, excluded,
    );
}

#[verifier::opaque]
pub open spec fn registration_registry_frame(
    before: &CacheScheduler,
    after: &CacheScheduler,
    rid: RequestId,
    prompt_len: int,
    skip: int,
    registered: Seq<u64>,
) -> bool {
    let ids = before.request_residency@[rid].block_ids@;
    let full = prompt_len / (BLOCK_SIZE_SPEC as int);
    &&& 0 <= skip <= full <= ids.len()
    &&& registered.len() == full - skip
    &&& forall|j: int| skip <= j < full
        ==> after.blocks@.contains_key(#[trigger] ids[j])
            && after.blocks@[ids[j]].hash_value == registered[j - skip]
    &&& forall|h: u64| #[trigger] before.hash_to_block@.contains_key(h)
        ==> after.hash_to_block@.contains_key(h)
            && after.hash_to_block@[h] == before.hash_to_block@[h]
            && before.blocks@.contains_key(before.hash_to_block@[h])
            && after.blocks@.contains_key(before.hash_to_block@[h])
            && after.blocks@[before.hash_to_block@[h]].tokens@
                == before.blocks@[before.hash_to_block@[h]].tokens@
            && after.blocks@[before.hash_to_block@[h]].hash_value
                == before.blocks@[before.hash_to_block@[h]].hash_value
            && after.blocks@[before.hash_to_block@[h]].prefix_depth
                == before.blocks@[before.hash_to_block@[h]].prefix_depth
            && after.blocks@[before.hash_to_block@[h]].parent_block
                == before.blocks@[before.hash_to_block@[h]].parent_block
    &&& forall|h: u64| #[trigger] after.hash_to_block@.contains_key(h)
        ==> before.hash_to_block@.contains_key(h)
            || ids.subrange(skip, full).contains(after.hash_to_block@[h])
}

#[verifier::opaque]
pub open spec fn registration_positive_page_origin(
    before: &CacheScheduler,
    after: &CacheScheduler,
    rid: RequestId,
    prompt_len: int,
    skip: int,
) -> bool {
    let ids = before.request_residency@[rid].block_ids@;
    let full = prompt_len / (BLOCK_SIZE_SPEC as int);
    &&& 0 <= skip <= full <= ids.len()
    &&& forall|bid: BlockId|
        #[trigger] after.blocks@[bid].prefix_depth > 0
        && after.blocks@.contains_key(bid)
        && after.blocks@[bid].prefix_depth > 0
        ==> positive_page_unchanged_from_pre(before, after, bid)
            || ids.subrange(skip, full).contains(bid)
}


#[verifier::opaque]
pub open spec fn reused_prefix_origin(
    pre: &CacheScheduler,
    ids: Seq<BlockId>,
    prompt_tokens: Seq<TokenId>,
    c: int,
) -> bool {
    &&& 0 <= c <= ids.len()
    &&& registered_prefix_chain(pre.blocks@, ids.subrange(0, c))
    &&& token_placement_prefix(
        pre.blocks@, ids, prompt_tokens,
        c * (BLOCK_SIZE_SPEC as int),
    )
    &&& forall|k: int| #![trigger ids[k]] 0 <= k < c ==> {
        let bid = ids[k];
        &&& pre.blocks@.contains_key(bid)
        &&& pre.hash_to_block@.contains_key(pre.blocks@[bid].hash_value)
        &&& pre.hash_to_block@[pre.blocks@[bid].hash_value] == bid
    }
}

// Convert the iteration-local allocation result into a pre-plan fact.  The
// same-step exclusion premise is essential: without it a later request could
// appear to reuse a page registered earlier in this very forward batch.  The
// result is opaque so its prefix-depth quantifier cannot cross-contaminate the
// large `plan` loop query.
pub proof fn lemma_reused_prefix_from_pre(
    pre: &CacheScheduler,
    before: &CacheScheduler,
    after: &CacheScheduler,
    excluded: Seq<u64>,
    rid: RequestId,
    prompt_tokens: Seq<TokenId>,
    c: int,
)
    requires
        registry_entries_from_pre(pre, before, excluded),
        allocate_prefill_reuse_success(before, after, rid, prompt_tokens, c),
        reused_prefix_excludes(before, after, rid, excluded, c),
        forall|bid: BlockId| #[trigger] before.blocks@.contains_key(bid)
            ==> after.blocks@.contains_key(bid)
                && after.blocks@[bid].hash_value == before.blocks@[bid].hash_value
                && after.blocks@[bid].prefix_depth == before.blocks@[bid].prefix_depth
                && after.blocks@[bid].parent_block == before.blocks@[bid].parent_block,
    ensures
        reused_prefix_origin(
            pre, after.request_residency@[rid].block_ids@, prompt_tokens, c,
        ),
{
    let ids = after.request_residency@[rid].block_ids@;
    reveal(registry_entries_from_pre);
    reveal(reused_prefix_origin);
    reveal(reused_prefix_excludes);
    assert(0 <= c <= ids.len());
    assert forall|k: int| #![trigger ids[k]] 0 <= k < c implies {
        let bid = ids[k];
        &&& pre.blocks@.contains_key(bid)
        &&& pre.hash_to_block@.contains_key(pre.blocks@[bid].hash_value)
        &&& pre.hash_to_block@[pre.blocks@[bid].hash_value] == bid
    } by {
        let bid = ids[k];
        let h = before.blocks@[bid].hash_value;
        assert(before.hash_to_block@.contains_key(h));
        assert(before.hash_to_block@[h] == bid);
        assert(!excluded.contains(h));
        assert(pre.hash_to_block@.contains_key(h));
        assert(pre.hash_to_block@[h] == bid);
        assert(pre.blocks@[bid].hash_value == h);
    }
    assert(registered_prefix_chain(pre.blocks@, ids.subrange(0, c))) by {
        assert forall|j: int| #![trigger pre.blocks@[ids.subrange(0, c)[j]].prefix_depth]
            0 <= j < ids.subrange(0, c).len() implies {
                let bid = ids.subrange(0, c)[j];
                &&& pre.blocks@.contains_key(bid)
                &&& pre.blocks@[bid].prefix_depth as int == j + 1
                &&& pre.blocks@[bid].parent_block
                    == if j == 0 { None } else {
                        Some(ids.subrange(0, c)[j - 1])
                    }
            } by {
            assert(ids.subrange(0, c)[j] == ids[j]);
            let bid = ids[j];
            let h = before.blocks@[bid].hash_value;
            assert(before.hash_to_block@.contains_key(h));
            assert(!excluded.contains(h));
            assert(pre.blocks@[bid].prefix_depth == before.blocks@[bid].prefix_depth);
            assert(pre.blocks@[bid].parent_block == before.blocks@[bid].parent_block);
            assert(after.blocks@[bid].prefix_depth == before.blocks@[bid].prefix_depth);
            assert(after.blocks@[bid].parent_block == before.blocks@[bid].parent_block);
            assert(registered_prefix_chain(after.blocks@, ids.subrange(0, c)));
            assert(after.blocks@[bid].prefix_depth as int == j + 1);
            if j > 0 {
                assert(ids.subrange(0, c)[j - 1] == ids[j - 1]);
            }
        }
    }
    assert(token_placement_prefix(
        pre.blocks@, ids, prompt_tokens,
        c * (BLOCK_SIZE_SPEC as int),
    )) by {
        assert forall|p: int| 0 <= p < c * (BLOCK_SIZE_SPEC as int)
            implies #[trigger] token_placement_at(
                pre.blocks@, ids, prompt_tokens, p,
            )
        by {
            assert(token_placement_prefix(
                after.blocks@, ids, prompt_tokens,
                prompt_tokens.len() as int,
            ));
            lemma_token_placement_prefix_at(
                after.blocks@, ids, prompt_tokens,
                prompt_tokens.len() as int, p,
            );
            let j = p / (BLOCK_SIZE_SPEC as int);
            assert(0 <= j < c) by {
                vstd::arithmetic::div_mod::lemma_div_is_ordered(
                    p, c * (BLOCK_SIZE_SPEC as int),
                    BLOCK_SIZE_SPEC as int,
                );
            }
            let bid = ids[j];
            let h = before.blocks@[bid].hash_value;
            assert(before.hash_to_block@.contains_key(h));
            assert(!excluded.contains(h));
            assert(pre.blocks@[bid].tokens@ == before.blocks@[bid].tokens@);
            assert(after.blocks@[bid].tokens@ == before.blocks@[bid].tokens@);
            reveal(token_placement_at);
        }
    }
}

// Loop-carried origin of a cached admission. Every reused page was already a
// registry target before this plan began, and its exact registered chain is
// read in that pre-state. This is stronger than merely being registered at
// plan exit, where pages produced by this same batch have also been stamped.
pub open spec fn admitted_prefix_from_pre_at(
    pre: &CacheScheduler,
    post: &CacheScheduler,
    sched: Seq<RequestId>,
    block_rows: Seq<Vec<BlockId>>,
    k: int,
) -> bool {
    let rid = sched[k];
    if pre.running@.contains(rid) {
        true
    } else {
        let ids = block_rows[k]@;
        let c = post.request_residency@[rid].cached_prefix_blocks as int;
        &&& post.request_residency@.contains_key(rid)
        &&& block_rows[k]@ == post.request_residency@[rid].block_ids@
        &&& 0 <= c <= ids.len()
        &&& registered_prefix_chain(pre.blocks@, ids.subrange(0, c))
        &&& registered_prefix_chain(post.blocks@, ids.subrange(0, c))
        &&& token_placement_prefix(
            pre.blocks@, ids, pre.live_requests@[rid].prompt_tokens@,
            c * (BLOCK_SIZE_SPEC as int),
        )
        &&& forall|l: int| #![trigger ids[l]] 0 <= l < c ==> {
            let bid = ids[l];
            &&& pre.blocks@.contains_key(bid)
            &&& pre.hash_to_block@.contains_key(pre.blocks@[bid].hash_value)
            &&& pre.hash_to_block@[pre.blocks@[bid].hash_value] == bid
        }
    }
}

#[verifier::opaque]
pub open spec fn admitted_prefixes_from_pre(
    pre: &CacheScheduler,
    post: &CacheScheduler,
    sched: Seq<RequestId>,
    block_rows: Seq<Vec<BlockId>>,
) -> bool {
    forall|k: int| 0 <= k < sched.len() ==>
        #[trigger] admitted_prefix_from_pre_at(pre, post, sched, block_rows, k)
}

pub proof fn lemma_admitted_prefixes_from_pre_running(
    pre: &CacheScheduler,
    post: &CacheScheduler,
    sched: Seq<RequestId>,
    block_rows: Seq<Vec<BlockId>>,
)
    requires
        forall|k: int| 0 <= k < sched.len()
            ==> pre.running@.contains(#[trigger] sched[k]),
    ensures
        admitted_prefixes_from_pre(pre, post, sched, block_rows),
{
    reveal(admitted_prefixes_from_pre);
    assert forall|k: int| 0 <= k < sched.len() implies
        #[trigger] admitted_prefix_from_pre_at(pre, post, sched, block_rows, k)
    by {
        assert(pre.running@.contains(sched[k]));
    }
}

pub proof fn lemma_admitted_prefixes_from_pre_frame(
    pre: &CacheScheduler,
    before: &CacheScheduler,
    after: &CacheScheduler,
    sched: Seq<RequestId>,
    block_rows: Seq<Vec<BlockId>>,
)
    requires
        admitted_prefixes_from_pre(pre, before, sched, block_rows),
        after.request_residency@ == before.request_residency@,
        forall|k: int| 0 <= k < sched.len()
            && !pre.running@.contains(#[trigger] sched[k]) ==> {
                let rid = sched[k];
                let ids = after.request_residency@[rid].block_ids@;
                let c = after.request_residency@[rid].cached_prefix_blocks as int;
                registered_prefix_chain(after.blocks@, ids.subrange(0, c))
            },
    ensures
        admitted_prefixes_from_pre(pre, after, sched, block_rows),
{
    reveal(admitted_prefixes_from_pre);
    assert forall|k: int| 0 <= k < sched.len() implies
        #[trigger] admitted_prefix_from_pre_at(
            pre, after, sched, block_rows, k,
        )
    by {
        assert(admitted_prefix_from_pre_at(
            pre, before, sched, block_rows, k,
        ));
    }
}

pub proof fn lemma_admitted_current_chains_positive_frame(
    pre: &CacheScheduler,
    before: &CacheScheduler,
    after: &CacheScheduler,
    sched: Seq<RequestId>,
    block_rows: Seq<Vec<BlockId>>,
)
    requires
        admitted_prefixes_from_pre(pre, before, sched, block_rows),
        cs_valid(before),
        after.request_residency@ == before.request_residency@,
        forall|bid: BlockId| #[trigger] before.blocks@.contains_key(bid)
            && before.blocks@[bid].refcount > 0
            ==> after.blocks@.contains_key(bid)
                && after.blocks@[bid] == before.blocks@[bid],
    ensures
        forall|k: int| 0 <= k < sched.len()
            && !pre.running@.contains(#[trigger] sched[k]) ==> {
                let rid = sched[k];
                let ids = after.request_residency@[rid].block_ids@;
                let c = after.request_residency@[rid].cached_prefix_blocks as int;
                registered_prefix_chain(after.blocks@, ids.subrange(0, c))
            },
{
    reveal(admitted_prefixes_from_pre);
    assert forall|k: int| 0 <= k < sched.len()
        && !pre.running@.contains(#[trigger] sched[k]) implies {
            let rid = sched[k];
            let ids = after.request_residency@[rid].block_ids@;
            let c = after.request_residency@[rid].cached_prefix_blocks as int;
            registered_prefix_chain(after.blocks@, ids.subrange(0, c))
        }
    by {
        let rid = sched[k];
        assert(admitted_prefix_from_pre_at(
            pre, before, sched, block_rows, k,
        ));
        assert(0 <= before.request_residency@[rid]
            .cached_prefix_blocks as int
            <= before.request_residency@[rid].block_ids@.len());
        lemma_registered_residency_prefix_positive_frame(before, after, rid);
    }
}

pub proof fn lemma_admitted_current_chain_at(
    pre: &CacheScheduler,
    post: &CacheScheduler,
    sched: Seq<RequestId>,
    block_rows: Seq<Vec<BlockId>>,
    k: int,
)
    requires
        admitted_prefixes_from_pre(pre, post, sched, block_rows),
        0 <= k < sched.len(),
        !pre.running@.contains(sched[k]),
    ensures
        ({
            let rid = sched[k];
            let ids = post.request_residency@[rid].block_ids@;
            let c = post.request_residency@[rid].cached_prefix_blocks as int;
            &&& post.request_residency@.contains_key(rid)
            &&& 0 <= c <= ids.len()
            &&& registered_prefix_chain(post.blocks@, ids.subrange(0, c))
        }),
{
    reveal(admitted_prefixes_from_pre);
    assert(admitted_prefix_from_pre_at(
        pre, post, sched, block_rows, k,
    ));
}

pub proof fn lemma_admitted_prefixes_from_pre_push(
    pre: &CacheScheduler,
    before: &CacheScheduler,
    after: &CacheScheduler,
    sched_before: Seq<RequestId>,
    rows_before: Seq<Vec<BlockId>>,
    rid: RequestId,
    row: Vec<BlockId>,
)
    requires
        admitted_prefixes_from_pre(
            pre, before, sched_before, rows_before,
        ),
        sched_before.len() == rows_before.len(),
        forall|r: RequestId| #[trigger] sched_before.contains(r)
            ==> after.request_residency@.contains_key(r)
                && before.request_residency@.contains_key(r)
                && after.request_residency@[r] == before.request_residency@[r],
        forall|k: int| 0 <= k < sched_before.len()
            && !pre.running@.contains(#[trigger] sched_before[k]) ==> {
                let old_rid = sched_before[k];
                let old_ids = after.request_residency@[old_rid].block_ids@;
                let old_c = after.request_residency@[old_rid]
                    .cached_prefix_blocks as int;
                registered_prefix_chain(
                    after.blocks@, old_ids.subrange(0, old_c),
                )
            },
        admitted_prefix_from_pre_at(
            pre, after, sched_before.push(rid), rows_before.push(row),
            sched_before.len() as int,
        ),
    ensures
        admitted_prefixes_from_pre(
            pre, after, sched_before.push(rid), rows_before.push(row),
        ),
{
    reveal(admitted_prefixes_from_pre);
    assert forall|k: int| 0 <= k < sched_before.push(rid).len() implies
        #[trigger] admitted_prefix_from_pre_at(
            pre, after, sched_before.push(rid), rows_before.push(row), k,
        )
    by {
        if k < sched_before.len() {
            assert(admitted_prefix_from_pre_at(
                pre, before, sched_before, rows_before, k,
            ));
            let old_rid = sched_before[k];
            assert(sched_before.contains(old_rid));
        } else {
            assert(k == sched_before.len());
        }
    }
}

// Package the successful-admission origin step so the scheduler's large plan
// proof does not repeatedly expand the registered-chain and registry-target
// quantifiers. `source` is the post-reuse/pre-registration state: registration
// and queue/vector updates leave the admitted residency unchanged.
pub proof fn lemma_admitted_prefixes_from_pre_admit(
    pre: &CacheScheduler,
    before: &CacheScheduler,
    source: &CacheScheduler,
    after: &CacheScheduler,
    sched_before: Seq<RequestId>,
    rows_before: Seq<Vec<BlockId>>,
    rid: RequestId,
    row: Vec<BlockId>,
    c: int,
)
    requires
        admitted_prefixes_from_pre(
            pre, before, sched_before, rows_before,
        ),
        sched_before.len() == rows_before.len(),
        !pre.running@.contains(rid),
        source.request_residency@.contains_key(rid),
        source.request_residency@[rid].cached_prefix_blocks as int == c,
        row@ == source.request_residency@[rid].block_ids@,
        0 <= c <= row@.len(),
        after.request_residency@.contains_key(rid),
        after.request_residency@[rid] == source.request_residency@[rid],
        reused_prefix_origin(
            pre, row@, pre.live_requests@[rid].prompt_tokens@, c,
        ),
        registered_prefix_chain(after.blocks@, row@.subrange(0, c)),
        forall|r: RequestId| #[trigger] sched_before.contains(r)
            ==> after.request_residency@.contains_key(r)
                && before.request_residency@.contains_key(r)
                && after.request_residency@[r] == before.request_residency@[r],
        forall|k: int| 0 <= k < sched_before.len()
            && !pre.running@.contains(#[trigger] sched_before[k]) ==> {
                let old_rid = sched_before[k];
                let old_ids = after.request_residency@[old_rid].block_ids@;
                let old_c = after.request_residency@[old_rid]
                    .cached_prefix_blocks as int;
                registered_prefix_chain(
                    after.blocks@, old_ids.subrange(0, old_c),
                )
            },
    ensures
        admitted_prefixes_from_pre(
            pre, after, sched_before.push(rid), rows_before.push(row),
        ),
{
    reveal(reused_prefix_origin);
    assert(admitted_prefix_from_pre_at(
        pre, after, sched_before.push(rid), rows_before.push(row),
        sched_before.len() as int,
    )) by {
        assert(sched_before.push(rid)[sched_before.len() as int] == rid);
        assert(rows_before.push(row)[rows_before.len() as int]@ == row@);
        assert(rows_before.len() == sched_before.len());
        assert(rows_before.push(row)[sched_before.len() as int]@ == row@);
        assert(after.request_residency@[rid].cached_prefix_blocks as int == c);
        assert(after.request_residency@[rid].block_ids@ == row@);
        assert(registered_prefix_chain(after.blocks@, row@.subrange(0, c)));
        assert(token_placement_prefix(
            pre.blocks@, row@, pre.live_requests@[rid].prompt_tokens@,
            c * (BLOCK_SIZE_SPEC as int),
        ));
        assert forall|l: int| #![trigger row@[l]] 0 <= l < c implies {
            let bid = row@[l];
            &&& pre.blocks@.contains_key(bid)
            &&& pre.hash_to_block@.contains_key(pre.blocks@[bid].hash_value)
            &&& pre.hash_to_block@[pre.blocks@[bid].hash_value] == bid
        } by {
        }
    }
    lemma_admitted_prefixes_from_pre_push(
        pre, before, after, sched_before, rows_before, rid, row,
    );
}

// A set containing two distinct elements has at least two elements
// (used to contradict "refcount == 1" for a block held by two requests).
pub proof fn lemma_two_holders(s: Set<RequestId>, a: RequestId, b: RequestId)
    requires
        s.contains(a),
        s.contains(b),
        a != b,
    ensures
        s.len() >= 2,
{
    let two = Set::<RequestId>::empty().insert(a).insert(b);
    assert(two.subset_of(s));
    assert(two.len() == 2) by {
        assert(Set::<RequestId>::empty().insert(a).len() == 1);
        assert(!Set::<RequestId>::empty().insert(a).contains(b));
    }
    vstd::set_lib::lemma_len_subset(two, s);
}

// History alignment turns "tail exactly full" into a pure history fact.
pub proof fn lemma_tail_mod(hist: int, len: int, tail: int)
    requires
        len >= 1,
        1 <= tail <= BLOCK_SIZE_SPEC as int,
        hist == (len - 1) * (BLOCK_SIZE_SPEC as int) + tail,
    ensures
        (hist % (BLOCK_SIZE_SPEC as int) == 0) <==> (tail == BLOCK_SIZE_SPEC as int),
{
    let bs = BLOCK_SIZE_SPEC as int;
    if tail == bs {
        assert(hist == len * bs);
        vstd::arithmetic::div_mod::lemma_mod_multiples_basic(len, bs);
    } else {
        vstd::arithmetic::div_mod::lemma_fundamental_div_mod_converse_mod(
            hist, bs, len - 1, tail);
        assert(hist % bs == tail);
    }
}

// Distinct resident block ids all live in the (bounded) blocks map, so a
// residency's block table can never exceed the pool size.
pub proof fn lemma_residency_len_bounded(cs: &CacheScheduler, rid: RequestId)
    requires
        residency_block_ids_unique(cs),
        residency_blocks_in_range(cs),
        block_count_valid(cs),
        cs.request_residency@.contains_key(rid),
    ensures
        cs.request_residency@[rid].block_ids@.len() <= cs.num_blocks,
{
    let ids = cs.request_residency@[rid].block_ids@;
    assert(ids.no_duplicates());
    ids.unique_seq_to_set();
    assert(ids.to_set().subset_of(cs.blocks@.dom())) by {
        assert forall|b: BlockId| ids.to_set().contains(b)
            implies cs.blocks@.dom().contains(b) by {
            assert(ids.contains(b));
            let j = choose|j: int| 0 <= j < ids.len() && ids[j] == b;
            assert(cs.blocks@.contains_key(ids[j]));
        }
    }
    vstd::set_lib::lemma_len_subset(ids.to_set(), cs.blocks@.dom());
}

// The alignment pins the block budget exactly.
pub proof fn lemma_aligned_blocks_needed(hist: int, len: int, tail: int)
    requires
        len >= 1,
        1 <= tail <= BLOCK_SIZE_SPEC as int,
        hist == (len - 1) * (BLOCK_SIZE_SPEC as int) + tail,
    ensures
        blocks_needed_for(hist as nat) as int == len,
{
    let bs = BLOCK_SIZE_SPEC as int;
    assert(hist >= 1);
    assert(hist - 1 == (len - 1) * bs + (tail - 1));
    vstd::arithmetic::div_mod::lemma_fundamental_div_mod_converse_div(
        hist - 1, bs, len - 1, tail - 1);
    assert((hist - 1) / bs == len - 1);
}

// Refcount validity transfers across block updates that keep the residency
// map, the block domain, and every refcount (prefix-cache registration
// rewrites hash/provenance metadata but not ownership).
pub proof fn lemma_refcount_valid_transfer(a: &CacheScheduler, b: &CacheScheduler)
    requires
        refcount_valid(a),
        b.request_residency@ == a.request_residency@,
        b.blocks@.dom() == a.blocks@.dom(),
        forall|bid: BlockId| #[trigger] a.blocks@.contains_key(bid)
            ==> b.blocks@[bid].refcount == a.blocks@[bid].refcount,
    ensures
        refcount_valid(b),
{
    assert forall|bid: BlockId|
        #[trigger] b.blocks@.contains_key(bid)
        implies b.blocks@[bid].refcount as int
            == residency_holders_of(b, bid).len() as int
    by {
        assert(a.blocks@.dom().contains(bid));
        assert(a.blocks@.contains_key(bid));
        assert_sets_equal!(residency_holders_of(b, bid) == residency_holders_of(a, bid),
            r: RequestId => {
            if residency_holders_of(b, bid).contains(r) {
                assert(a.request_residency@.contains_key(r));
                assert(a.request_residency@[r] == b.request_residency@[r]);
            }
            if residency_holders_of(a, bid).contains(r) {
                assert(b.request_residency@.contains_key(r));
                assert(b.request_residency@[r] == a.request_residency@[r]);
            }
        });
    }
}

} // verus!
