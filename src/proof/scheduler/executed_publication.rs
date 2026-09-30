//! Forward-executed page origins, independent of prefill/decode and model family.

use super::*;

verus! {

pub open spec fn admission_page_origin(
    pre: &CacheScheduler, scheduled: Seq<RequestId>, rows: Seq<Seq<BlockId>>,
    cu_k: Seq<int>, bid: BlockId,
) -> bool {
    exists|k: int, l: int|
        0 <= k < scheduled.len() && k < rows.len()
        && !pre.running@.contains(scheduled[k])
        && pre.live_requests@.contains_key(scheduled[k])
        && 0 <= cu_k[k + 1] - cu_k[k]
            <= pre.live_requests@[scheduled[k]].prompt_tokens@.len()
        && 0 <= l < (cu_k[k + 1] - cu_k[k]) / (BLOCK_SIZE_SPEC as int)
        && #[trigger] rows[k][l] == bid
}

// This certificate survives request completion: it refers to the executed
// row and its physical pages, not the request's post-commit residency.
pub open spec fn executed_decode_row(
    pre: &CacheScheduler, post: &CacheScheduler, scheduled: Seq<RequestId>,
    rows: Seq<Seq<BlockId>>, cu_k: Seq<int>, k: int,
) -> bool {
    let rid = scheduled[k];
    let tokens = history(pre.live_requests@[rid]);
    let ids = rows[k];
    &&& 0 <= k < scheduled.len()
    &&& k < rows.len()
    &&& pre.running@.contains(rid)
    &&& pre.live_requests@.contains_key(rid)
    &&& ids.len() > 0
    &&& tokens.len() <= u64::MAX as int
    &&& ids.len() <= u64::MAX as int
    &&& cu_k[k + 1] - cu_k[k] == tokens.len()
    &&& tokens.len() == ids.len() * (BLOCK_SIZE_SPEC as int)
    &&& registered_prefix_chain(post.blocks@, ids)
    &&& token_placement_prefix(post.blocks@, ids, tokens, tokens.len() as int)
}

#[verifier::opaque]
pub open spec fn decode_page_origin(
    pre: &CacheScheduler, post: &CacheScheduler, scheduled: Seq<RequestId>,
    rows: Seq<Seq<BlockId>>, cu_k: Seq<int>, bid: BlockId,
) -> bool {
    exists|k: int| executed_decode_row(pre, post, scheduled, rows, cu_k, k)
        && #[trigger] rows[k][rows[k].len() - 1] == bid
}

#[verifier::opaque]
pub open spec fn positive_chains_from_pre_or_executed_rows(
    pre: &CacheScheduler, post: &CacheScheduler, scheduled: Seq<RequestId>,
    rows: Seq<Seq<BlockId>>, cu_k: Seq<int>,
) -> bool {
    cu_k.len() == scheduled.len() + 1
    && forall|chain: Seq<BlockId>, j: int|
        #![trigger registered_prefix_chain(post.blocks@, chain), chain[j]]
        registered_prefix_chain(post.blocks@, chain) && 0 <= j < chain.len()
        ==> admission_page_origin(pre, scheduled, rows, cu_k, chain[j])
            || decode_page_origin(pre, post, scheduled, rows, cu_k, chain[j])
            || (forall|l: int| 0 <= l <= j ==>
                #[trigger] positive_page_unchanged_from_pre(pre, post, chain[l]))
}

// The publisher's physical effect is connected to the forward's rows here;
// tensor materialization itself is discharged by the engine semantic proof.
#[verifier::spinoff_prover]
pub proof fn lemma_decode_publication_row_origins(
    pre: &CacheScheduler, before: &CacheScheduler, after: &CacheScheduler,
    published: Set<RequestId>, scheduled: Seq<RequestId>, rows: Seq<Seq<BlockId>>,
    cu_k: Seq<int>,
)
    requires
        decode_publication_effect(before, after, published),
        published.subset_of(scheduled.to_set()),
        before.live_requests@ == pre.live_requests@,
        before.num_blocks <= u64::MAX,
        rows.len() == scheduled.len(),
        cu_k.len() == scheduled.len() + 1,
        forall|k: int| 0 <= k < scheduled.len() ==> {
            let rid = #[trigger] scheduled[k];
            &&& before.request_residency@.contains_key(rid)
            &&& rows[k] == before.request_residency@[rid].block_ids@
            &&& rows[k].len() <= before.num_blocks
            &&& (pre.running@.contains(rid)
                ==> cu_k[k + 1] - cu_k[k] == history(pre.live_requests@[rid]).len())
            &&& (!pre.running@.contains(rid)
                ==> pre.live_requests@[rid].generated_tokens@.len() == 0)
        },
    ensures
        forall|rid: RequestId| #[trigger] published.contains(rid)
            ==> decode_page_origin(pre, after, scheduled, rows, cu_k,
                before.request_residency@[rid].block_ids@[
                    before.request_residency@[rid].block_ids@.len() - 1]),
{
    reveal(decode_publication_effect);
    reveal(decode_page_origin);
    assert forall|rid: RequestId| #[trigger] published.contains(rid)
        implies decode_page_origin(pre, after, scheduled, rows, cu_k,
            before.request_residency@[rid].block_ids@[
                before.request_residency@[rid].block_ids@.len() - 1])
    by {
        assert(scheduled.contains(rid));
        let k = scheduled.index_of(rid);
        assert(scheduled[k] == rid);
        assert(pre.running@.contains(rid));
        assert(executed_decode_row(pre, after, scheduled, rows, cu_k, k));
    }
}

// Used by the no-forward/empty-schedule path: admission-only origins are a
// special case of the executed-row relation, without any decode publication.
pub proof fn lemma_admission_origins_are_executed_rows(
    pre: &CacheScheduler, post: &CacheScheduler, scheduled: Seq<RequestId>,
    rows: Seq<Seq<BlockId>>, cu_k: Seq<int>,
)
    requires positive_chains_from_pre_or_admission_rows(pre, post, scheduled, rows, cu_k),
    ensures positive_chains_from_pre_or_executed_rows(pre, post, scheduled, rows, cu_k),
{
    reveal(positive_chains_from_pre_or_admission_rows);
    reveal(positive_chains_from_pre_or_executed_rows);
}

#[verifier::spinoff_prover]
pub proof fn lemma_decode_publication_chain_origins(
    pre: &CacheScheduler, before: &CacheScheduler, after: &CacheScheduler,
    published: Set<RequestId>, scheduled: Seq<RequestId>, rows: Seq<Seq<BlockId>>,
    cu_k: Seq<int>,
)
    requires
        persistent_provenance_closed(before),
        persistent_provenance_closed(after),
        decode_publication_effect(before, after, published),
        positive_chains_from_pre_or_admission_rows(pre, before, scheduled, rows, cu_k),
        forall|rid: RequestId| #[trigger] published.contains(rid)
            ==> decode_page_origin(pre, after, scheduled, rows, cu_k,
                before.request_residency@[rid].block_ids@[
                    before.request_residency@[rid].block_ids@.len() - 1]),
    ensures
        positive_chains_from_pre_or_executed_rows(pre, after, scheduled, rows, cu_k),
{
    reveal(decode_publication_effect);
    reveal(positive_chains_from_pre_or_executed_rows);
    reveal(positive_chains_from_pre_or_admission_rows);
    assert forall|chain: Seq<BlockId>, j: int|
        #![trigger registered_prefix_chain(after.blocks@, chain), chain[j]]
        registered_prefix_chain(after.blocks@, chain) && 0 <= j < chain.len()
        implies admission_page_origin(pre, scheduled, rows, cu_k, chain[j])
            || decode_page_origin(pre, after, scheduled, rows, cu_k, chain[j])
            || (forall|l: int| 0 <= l <= j ==>
                #[trigger] positive_page_unchanged_from_pre(pre, after, chain[l]))
    by {
        let bid = chain[j];
        if exists|rid: RequestId| published.contains(rid)
            && before.request_residency@[rid].block_ids@[
                before.request_residency@[rid].block_ids@.len() - 1] == bid {
            let rid = choose|rid: RequestId| published.contains(rid)
                && before.request_residency@[rid].block_ids@[
                    before.request_residency@[rid].block_ids@.len() - 1] == bid;
        } else {
            reveal(registered_prefix_chain);
            assert(after.blocks@.contains_key(bid));
            assert(before.blocks@[bid] == after.blocks@[bid]);
            lemma_publication_inherited_chain_prefix(before, after, chain, j);
            let prefix = chain.subrange(0, j + 1);
            assert(registered_prefix_chain(before.blocks@, prefix)) by {
                assert forall|l: int| 0 <= l < prefix.len() implies {
                    let page = #[trigger] prefix[l];
                    &&& before.blocks@.contains_key(page)
                    &&& before.blocks@[page].prefix_depth as int == l + 1
                    &&& before.blocks@[page].parent_block
                        == if l == 0 { None } else { Some(prefix[l - 1]) }
                } by {
                    assert(prefix[l] == chain[l]);
                    assert(positive_page_unchanged_from_pre(before, after, chain[l]));
                    assert(after.blocks@[chain[l]].prefix_depth as int == l + 1);
                    if l > 0 { assert(prefix[l - 1] == chain[l - 1]); }
                }
            }
            assert(prefix[j] == bid);
            if !admission_page_origin(pre, scheduled, rows, cu_k, bid) {
                assert(forall|l: int| 0 <= l <= j ==>
                    #[trigger] positive_page_unchanged_from_pre(pre, before, prefix[l]));
                assert forall|l: int| 0 <= l <= j implies
                    #[trigger] positive_page_unchanged_from_pre(pre, after, chain[l])
                by {
                    assert(prefix[l] == chain[l]);
                    assert(positive_page_unchanged_from_pre(pre, before, prefix[l]));
                    assert(positive_page_unchanged_from_pre(before, after, chain[l]));
                }
            }
        }
    }
}

#[verifier::spinoff_prover]
pub proof fn lemma_decode_page_origin_frame(
    pre: &CacheScheduler, before: &CacheScheduler, after: &CacheScheduler,
    scheduled: Seq<RequestId>, rows: Seq<Seq<BlockId>>, cu_k: Seq<int>, bid: BlockId,
)
    requires
        positive_provenance_metadata_frame(before, after),
        decode_page_origin(pre, before, scheduled, rows, cu_k, bid),
    ensures
        decode_page_origin(pre, after, scheduled, rows, cu_k, bid),
{
    reveal(decode_page_origin);
    let k = choose|k: int| executed_decode_row(pre, before, scheduled, rows, cu_k, k)
        && rows[k][rows[k].len() - 1] == bid;
    let ids = rows[k];
    let tokens = history(pre.live_requests@[scheduled[k]]);
    reveal(positive_provenance_metadata_frame);
    assert forall|j: int| 0 <= j < ids.len() implies {
        let page = #[trigger] ids[j];
        &&& after.blocks@.contains_key(page)
        &&& after.blocks@[page].prefix_depth == before.blocks@[page].prefix_depth
        &&& after.blocks@[page].parent_block == before.blocks@[page].parent_block
        &&& after.blocks@[page].tokens@ == before.blocks@[page].tokens@
    } by {
        reveal(registered_prefix_chain);
        assert(before.blocks@[ids[j]].prefix_depth as int == j + 1);
    }
    lemma_registered_prefix_chain_transfer(before.blocks@, after.blocks@, ids);
    lemma_token_placement_prefix_transfer_for_ids(
        before.blocks@, after.blocks@, ids, tokens, tokens.len() as int,
    );
    assert(executed_decode_row(pre, after, scheduled, rows, cu_k, k));
}

#[verifier::spinoff_prover]
pub proof fn lemma_positive_chains_executed_rows_frame(
    pre: &CacheScheduler, before: &CacheScheduler, after: &CacheScheduler,
    scheduled: Seq<RequestId>, rows: Seq<Seq<BlockId>>, cu_k: Seq<int>,
)
    requires
        positive_chains_from_pre_or_executed_rows(pre, before, scheduled, rows, cu_k),
        positive_provenance_metadata_frame(before, after),
        positive_provenance_origin(before, after),
    ensures
        positive_chains_from_pre_or_executed_rows(pre, after, scheduled, rows, cu_k),
{
    reveal(positive_chains_from_pre_or_executed_rows);
    reveal(positive_provenance_origin);
    assert forall|chain: Seq<BlockId>, j: int|
        #![trigger registered_prefix_chain(after.blocks@, chain), chain[j]]
        registered_prefix_chain(after.blocks@, chain) && 0 <= j < chain.len()
        implies admission_page_origin(pre, scheduled, rows, cu_k, chain[j])
            || decode_page_origin(pre, after, scheduled, rows, cu_k, chain[j])
            || (forall|l: int| 0 <= l <= j ==>
                #[trigger] positive_page_unchanged_from_pre(pre, after, chain[l]))
    by {
        reveal(registered_prefix_chain);
        assert forall|l: int| 0 <= l < chain.len() implies {
            let bid = #[trigger] chain[l];
            &&& before.blocks@.contains_key(bid)
            &&& before.blocks@[bid].prefix_depth as int == l + 1
            &&& before.blocks@[bid].parent_block
                == if l == 0 { None } else { Some(chain[l - 1]) }
        } by {
            assert(after.blocks@[chain[l]].prefix_depth as int == l + 1);
        }
        assert(registered_prefix_chain(before.blocks@, chain));
        if decode_page_origin(pre, before, scheduled, rows, cu_k, chain[j]) {
            lemma_decode_page_origin_frame(pre, before, after, scheduled, rows, cu_k, chain[j]);
        } else if !admission_page_origin(pre, scheduled, rows, cu_k, chain[j]) {
            assert forall|l: int| 0 <= l <= j implies
                #[trigger] positive_page_unchanged_from_pre(pre, after, chain[l])
            by {
                assert(positive_page_unchanged_from_pre(pre, before, chain[l]));
                assert(after.blocks@[chain[l]].prefix_depth > 0);
            }
        }
    }
}

#[verifier::spinoff_prover]
pub proof fn lemma_decode_publication_plan_origins(
    pre: &CacheScheduler, before: &CacheScheduler, after: &CacheScheduler,
    published: Set<RequestId>, plan: &StepPlan,
)
    requires
        cs_valid(pre),
        cs_valid(before),
        persistent_provenance_closed(before),
        persistent_provenance_closed(after),
        before.live_requests@ == pre.live_requests@,
        before.num_blocks <= u64::MAX,
        decode_publication_effect(before, after, published),
        published.subset_of(plan.scheduled_ids@.to_set()),
        plan.block_table_repr@.len() == plan.scheduled_ids@.len(),
        plan.cu_seqlens_k_repr@.len() == plan.scheduled_ids@.len() + 1,
        plan_slot_segments_ok(pre, before, plan),
        plan_forward_layout_ok(pre, plan),
        positive_chains_from_pre_or_admission_rows(pre, before,
            plan.scheduled_ids@, plan.block_table_repr@, plan.cu_seqlens_k_repr@),
        published_admission_row_prefixes(pre, before,
            plan.scheduled_ids@, plan.block_table_repr@, plan.cu_seqlens_k_repr@),
    ensures
        positive_chains_from_pre_or_executed_rows(pre, after,
            plan.scheduled_ids@, plan.block_table_repr@, plan.cu_seqlens_k_repr@),
        published_admission_row_prefixes(pre, after,
            plan.scheduled_ids@, plan.block_table_repr@, plan.cu_seqlens_k_repr@),
{
    let scheduled = plan.scheduled_ids@;
    let rows = plan.block_table_repr@;
    let cu_k = plan.cu_seqlens_k_repr@;
    assert forall|k: int| 0 <= k < scheduled.len() implies {
        let rid = #[trigger] scheduled[k];
        &&& before.request_residency@.contains_key(rid)
        &&& rows[k] == before.request_residency@[rid].block_ids@
        &&& rows[k].len() <= before.num_blocks
        &&& (pre.running@.contains(rid)
            ==> cu_k[k + 1] - cu_k[k] == history(pre.live_requests@[rid]).len())
        &&& (!pre.running@.contains(rid)
            ==> pre.live_requests@[rid].generated_tokens@.len() == 0)
    } by {
        assert(plan_slot_segments_at(pre, before, plan, k));
        reveal(plan_slot_segments_at);
        assert(plan_forward_layout_at(pre, plan, k));
        lemma_residency_len_bounded(before, scheduled[k]);
    }
    lemma_decode_publication_row_origins(pre, before, after, published, scheduled, rows, cu_k);
    lemma_decode_publication_chain_origins(pre, before, after, published, scheduled, rows, cu_k);
    lemma_admission_prefixes_decode_publication_frame(pre, before, after, published, scheduled, rows, cu_k);
}

// Decode publication cannot touch an admission's private suffix. Its already
// registered prefix is protected by the ordinary positive-metadata frame.
#[verifier::spinoff_prover]
pub proof fn lemma_admission_prefixes_decode_publication_frame(
    pre: &CacheScheduler, before: &CacheScheduler, after: &CacheScheduler,
    published: Set<RequestId>, scheduled: Seq<RequestId>, rows: Seq<Seq<BlockId>>,
    cu_k: Seq<int>,
)
    requires
        cs_valid(before),
        before.live_requests@ == pre.live_requests@,
        decode_publication_effect(before, after, published),
        published_admission_row_prefixes(pre, before, scheduled, rows, cu_k),
        forall|k: int| 0 <= k < scheduled.len()
            && !pre.running@.contains(#[trigger] scheduled[k]) ==> {
                &&& pre.live_requests@[scheduled[k]].generated_tokens@.len() == 0
                &&& before.request_residency@.contains_key(scheduled[k])
                &&& rows[k] == before.request_residency@[scheduled[k]].block_ids@
            },
    ensures
        published_admission_row_prefixes(pre, after, scheduled, rows, cu_k),
{
    reveal(decode_publication_effect);
    reveal(published_admission_row_prefixes);
    assert forall|k: int| 0 <= k < scheduled.len()
        && !pre.running@.contains(#[trigger] scheduled[k])
        implies pre.live_requests@.contains_key(scheduled[k]) && {
            let prompt = pre.live_requests@[scheduled[k]].prompt_tokens@;
            let end = cu_k[k + 1] - cu_k[k];
            let full = end / (BLOCK_SIZE_SPEC as int);
            let ids = rows[k];
            &&& 0 <= full <= ids.len()
            &&& 0 <= end <= prompt.len()
            &&& prompt.len() <= u64::MAX as int
            &&& blocks_needed_for(end as nat) <= u64::MAX as nat
            &&& registered_prefix_chain(after.blocks@, ids.subrange(0, full))
            &&& token_placement_prefix(after.blocks@, ids, prompt, full * (BLOCK_SIZE_SPEC as int))
            &&& forall|l: int| full <= l < ids.len()
                && #[trigger] after.blocks@.contains_key(ids[l])
                ==> after.blocks@[ids[l]].prefix_depth == 0
        }
    by {
        let rid = scheduled[k];
        let ids = rows[k];
        let prompt = pre.live_requests@[rid].prompt_tokens@;
        let full = (cu_k[k + 1] - cu_k[k]) / (BLOCK_SIZE_SPEC as int);
        assert(!published.contains(rid));
        lemma_decode_publication_bystander_frame(before, after, published, rid);
        assert forall|j: int| 0 <= j < ids.subrange(0, full).len() implies {
            let bid = #[trigger] ids.subrange(0, full)[j];
            &&& after.blocks@.contains_key(bid)
            &&& after.blocks@[bid].prefix_depth == before.blocks@[bid].prefix_depth
            &&& after.blocks@[bid].parent_block == before.blocks@[bid].parent_block
        } by { assert(ids.subrange(0, full)[j] == ids[j]); }
        lemma_registered_prefix_chain_transfer(before.blocks@, after.blocks@, ids.subrange(0, full));
        lemma_token_placement_prefix_transfer(before.blocks@, after.blocks@, ids, prompt,
            full * (BLOCK_SIZE_SPEC as int));
    }
}

}
