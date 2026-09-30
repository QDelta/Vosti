// Prefix caching steps 1+2a: hash registration + the reuse scan.
// Admit request A, register its full blocks, then scan
// with prompts sharing / not sharing A's first block.

use vosti_verus::exec::cache_scheduler::{chain_hash_span, BlockEntry, CacheScheduler, SchedulerConfig};
use vosti_verus::types::BLOCK_SIZE;
use vosti_verus::exec::request_state::{EosTokenSet, RequestState, SamplerState};

const PAGE: usize = BLOCK_SIZE as usize;

fn prompt(len: usize, seed: u64) -> Vec<u64> {
    (0..len as u64).map(|i| seed.wrapping_add(i)).collect()
}

#[test]
fn publication_can_extend_beyond_initial_cache_hits() {
    let mut cs = CacheScheduler::init(
        SchedulerConfig { max_num_seqs: 4, max_num_batched_tokens: 4096 }, 16,
    );
    let mut tokens = prompt(PAGE + 1, 42_000);
    assert!(cs.allocate_prefill(7, &tokens));
    cs.publish_full_prefix_pages(7, &tokens, 0);
    for token in prompt(PAGE - 1, 77_000) {
        cs.append_token(7, token);
        tokens.push(token);
    }
    // This is a scheduler-metadata test, not evidence of GPU KV materialization.
    // The eventual Engine hook must separately establish that these tokens
    // have actually been processed before invoking the publisher.
    let hashes = cs.publish_completed_tail(7, &tokens);
    assert_eq!(hashes.len(), 1);
    assert_eq!(cs.request_residency.get(&7).unwrap().cached_prefix_blocks, 0);
    let first = chain_hash_span(0, &tokens, 0, PAGE);
    assert_eq!(hashes[0], chain_hash_span(first, &tokens, PAGE, 2 * PAGE));
    let ids = cs.request_residency.get(&7).unwrap().block_ids.clone();
    assert_eq!(cs.blocks.get(&ids[1]).unwrap().parent_block, Some(ids[0]));
    assert_eq!(cs.blocks.get(&ids[1]).unwrap().prefix_depth, 2);
    // A fully published prefix is an idempotent no-op, including exact pages.
    assert!(cs.publish_full_prefix_pages(7, &tokens, 2).is_empty());
    cs.deallocate(7);
    tokens.push(123);
    assert_eq!(cs.match_cached_prefix(&tokens, &Vec::new()), ids);
}

#[test]
fn extension_preserves_a_shared_parent_and_initial_reuse_count() {
    let mut cs = CacheScheduler::init(
        SchedulerConfig { max_num_seqs: 4, max_num_batched_tokens: 4096 }, 16,
    );
    let donor = prompt(PAGE + 1, 1_000);
    assert!(cs.allocate_prefill(7, &donor));
    cs.publish_full_prefix_pages(7, &donor, 0);
    let mut tokens = donor[..PAGE].to_vec();
    tokens.extend(prompt(PAGE + 1, 7_000));
    assert!(cs.allocate_prefill_with_reuse(8, &tokens, &Vec::new()));
    cs.publish_full_prefix_pages(8, &tokens, 1);
    let parent = cs.request_residency.get(&8).unwrap().block_ids[0];
    assert_eq!(cs.blocks.get(&parent).unwrap().refcount, 2);
    for token in prompt(PAGE - 1, 17_000) {
        cs.append_token(8, token);
        tokens.push(token);
    }
    let hashes = cs.publish_completed_tail(8, &tokens);
    assert_eq!(hashes.len(), 1);
    assert_eq!(cs.request_residency.get(&8).unwrap().cached_prefix_blocks, 1);
    assert_eq!(cs.blocks.get(&parent).unwrap().refcount, 2);
    let mut hash = 0;
    for page in 0..3 {
        hash = chain_hash_span(hash, &tokens, page * PAGE, (page + 1) * PAGE);
    }
    assert_eq!(hashes[0], hash);
    let ids = cs.request_residency.get(&8).unwrap().block_ids.clone();
    cs.deallocate(8);
    tokens.push(789);
    assert_eq!(cs.match_cached_prefix(&tokens, &Vec::new()), ids);
}

#[test]
fn completed_first_page_can_be_published_without_a_parent() {
    let mut cs = CacheScheduler::init(
        SchedulerConfig { max_num_seqs: 4, max_num_batched_tokens: 4096 }, 8,
    );
    let mut tokens = prompt(PAGE - 1, 12_000);
    assert!(cs.allocate_prefill(7, &tokens));
    assert!(cs.publish_full_prefix_pages(7, &tokens, 0).is_empty());
    cs.append_token(7, 88_000);
    tokens.push(88_000);
    // Model the metadata at the next forward's completion, not the instant
    // this token was sampled. The Engine integration must enforce that timing.
    assert_eq!(cs.publish_completed_tail(7, &tokens).len(), 1);
    let first = cs.request_residency.get(&7).unwrap().block_ids[0];
    assert_eq!(cs.blocks.get(&first).unwrap().parent_block, None);
    cs.append_token(7, 88_001);
    let tail = cs.request_residency.get(&7).unwrap().block_ids[1];
    assert_eq!(cs.blocks.get(&tail).unwrap().prefix_depth, 0);
    assert_eq!(cs.blocks.get(&tail).unwrap().hash_value, 0);
    cs.deallocate(7);
    tokens.push(99_000);
    assert_eq!(cs.match_cached_prefix(&tokens, &Vec::new()), vec![first]);
}

#[test]
fn scan_matches_registered_shared_prefix() {
    let config = SchedulerConfig {
        max_num_seqs: 4,
        max_num_batched_tokens: 4096,
    };
    let mut cs = CacheScheduler::init(config, 16);

    // Admit A with one full block plus a 45-token partial tail.
    let a = prompt(PAGE + 45, 1000);
    let admitted = cs.allocate_prefill(7, &a);
    assert!(admitted);
    cs.publish_full_prefix_pages(7, &a, 0);

    // B shares A's entire first block, then diverges: the scan must reuse
    // exactly that one block.
    let mut b = a[..PAGE].to_vec();
    b.extend(prompt(100, 999_000));
    let matched = cs.match_cached_prefix(&b, &Vec::new());
    assert_eq!(matched.len(), 1);

    // C diverges inside block 0: no reuse.
    let c = prompt(PAGE + 45, 2000);
    let matched_c = cs.match_cached_prefix(&c, &Vec::new());
    assert_eq!(matched_c.len(), 0);

    // A itself: cap keeps at least one position uncached; with PAGE+45 tokens
    // the single full block is still reusable.
    let matched_a = cs.match_cached_prefix(&a, &Vec::new());
    assert_eq!(matched_a.len(), 1);

    // Exactly PAGE tokens: the only full block IS the whole prompt, so the
    // leave-one-position guard forbids reuse.
    let d = a[..PAGE].to_vec();
    let matched_d = cs.match_cached_prefix(&d, &Vec::new());
    assert_eq!(matched_d.len(), 0);
}

#[test]
fn reuse_allocation_shares_prefix_block() {
    let config = SchedulerConfig {
        max_num_seqs: 4,
        max_num_batched_tokens: 4096,
    };
    let mut cs = CacheScheduler::init(config, 16);

    // Donor A: one full block plus a 44-token tail, registered.
    let a = prompt(PAGE + 44, 5000);
    assert!(cs.allocate_prefill(7, &a));
    cs.publish_full_prefix_pages(7, &a, 0);
    let free_after_a = cs.free_blocks;

    // B shares A's first block then diverges: reuse-aware allocation must
    // take the shared block (refcount 2), allocate only the suffix, and
    // record cached_prefix_blocks = 1 with a suffix-only slot_mapping.
    let mut b = a[..PAGE].to_vec();
    b.extend(prompt(44, 777_000));
    assert!(cs.allocate_prefill_with_reuse(8, &b, &Vec::new()));

    let res_b = cs.request_residency.get(&8).unwrap();
    assert_eq!(res_b.cached_prefix_blocks, 1);
    assert_eq!(res_b.block_ids.len(), 2); // 1 shared + 1 fresh suffix page
    assert_eq!(res_b.slot_mapping.len(), 44); // suffix positions only
    let shared_bid = res_b.block_ids[0];
    let res_a = cs.request_residency.get(&7).unwrap();
    assert_eq!(shared_bid, res_a.block_ids[0]);
    assert_eq!(cs.blocks.get(&shared_bid).unwrap().refcount, 2);
    // Only the fresh suffix block was drawn from the free pool.
    assert_eq!(cs.free_blocks, free_after_a - 1);
    // Suffix slots live in the fresh block at global offsets mod BLOCK_SIZE.
    let fresh_bid = res_b.block_ids[1];
    assert_eq!(res_b.slot_mapping[0], fresh_bid * BLOCK_SIZE);

    // C diverges in block 0: falls back to a fully fresh allocation.
    let cprompt = prompt(PAGE + 44, 9000);
    assert!(cs.allocate_prefill_with_reuse(9, &cprompt, &Vec::new()));
    let res_c = cs.request_residency.get(&9).unwrap();
    assert_eq!(res_c.cached_prefix_blocks, 0);
    assert_eq!(res_c.slot_mapping.len(), PAGE + 44);
}

#[test]
fn registered_prefix_survives_last_holder_and_is_reused() {
    let config = SchedulerConfig {
        max_num_seqs: 4,
        max_num_batched_tokens: 4096,
    };
    let mut cs = CacheScheduler::init(config, 16);

    let a = prompt(PAGE + 44, 42_000);
    assert!(cs.allocate_prefill(7, &a));
    cs.publish_full_prefix_pages(7, &a, 0);
    let prefix_bid = cs.request_residency.get(&7).unwrap().block_ids[0];
    let tail_bid = cs.request_residency.get(&7).unwrap().block_ids[1];
    let free_before_release = cs.free_blocks;

    cs.deallocate(7);

    // The full registered page becomes a zero-ref resident cache entry.  The
    // unregistered partial tail becomes vacant immediately.
    assert!(!cs.request_residency.contains_key(&7));
    assert_eq!(cs.blocks.get(&prefix_bid).unwrap().refcount, 0);
    assert!(!cs.blocks.contains_key(&tail_bid));
    assert_eq!(cs.free_blocks, free_before_release + 1);

    let mut b = a[..PAGE].to_vec();
    b.extend(prompt(44, 700_000));
    let matched = cs.match_cached_prefix(&b, &Vec::new());
    assert_eq!(matched, vec![prefix_bid]);
    assert!(cs.allocate_prefill_with_reuse(8, &b, &Vec::new()));
    assert_eq!(
        cs.request_residency.get(&8).unwrap().block_ids[0],
        prefix_bid,
    );
    assert_eq!(cs.blocks.get(&prefix_bid).unwrap().refcount, 1);
}

#[test]
fn deallocation_enqueues_cached_chain_tail_first() {
    let config = SchedulerConfig {
        max_num_seqs: 4,
        max_num_batched_tokens: 4096,
    };
    let mut cs = CacheScheduler::init(config, 3);

    // Two registered full pages plus an unregistered tail fill the pool.
    let tokens = prompt(2 * PAGE + 44, 88_000);
    assert!(cs.allocate_prefill(7, &tokens));
    cs.publish_full_prefix_pages(7, &tokens, 0);
    let ids = cs.request_residency.get(&7).unwrap().block_ids.clone();
    let parent = ids[0];
    let child = ids[1];
    let parent_hash = cs.blocks.get(&parent).unwrap().hash_value;
    let child_hash = cs.blocks.get(&child).unwrap().hash_value;

    cs.deallocate(7);
    assert_eq!(cs.free_blocks, 1); // only the unregistered tail was freed
    assert_eq!(cs.blocks.get(&parent).unwrap().refcount, 0);
    assert_eq!(cs.blocks.get(&child).unwrap().refcount, 0);

    // Reverse release preserves cached-queue topology: the child is the first
    // eligible victim and points to its parent as the next victim.
    assert_eq!(cs.cached_queue.head, Some(child));
    assert_eq!(
        cs.cached_queue.links.get(&child).unwrap().next,
        Some(parent)
    );

    assert_eq!(cs.evict_one_cached_leaf(), Some(child));
    assert!(!cs.blocks.contains_key(&child));
    assert!(!cs.hash_to_block.contains_key(&child_hash));
    assert_eq!(cs.free_blocks, 2);

    assert_eq!(cs.cached_queue.head, Some(parent));
    assert_eq!(cs.evict_one_cached_leaf(), Some(parent));
    assert!(!cs.blocks.contains_key(&parent));
    assert!(!cs.hash_to_block.contains_key(&parent_hash));
    assert_eq!(cs.free_blocks, 3);
}

#[test]
fn pressure_reclamation_repeats_until_target_or_exhaustion() {
    let config = SchedulerConfig {
        max_num_seqs: 4,
        max_num_batched_tokens: 4096,
    };
    let mut cs = CacheScheduler::init(config, 4);

    let tokens = prompt(2 * PAGE + 44, 123_000);
    assert!(cs.allocate_prefill(7, &tokens));
    cs.publish_full_prefix_pages(7, &tokens, 0);
    cs.deallocate(7);

    // Two registered zero-ref pages remain; the partial tail was freed.
    assert_eq!(cs.blocks.len(), 2);
    assert_eq!(cs.free_blocks, 2);

    // The cached suffix already stores child before parent, so reclamation is
    // O(1) per victim and naturally consumes the chain from its tail.
    assert_eq!(cs.reclaim_cached_leaves_until(4), 2);
    assert_eq!(cs.free_blocks, 4);
    assert!(cs.blocks.is_empty());
    assert!(cs.hash_to_block.is_empty());

    // Exhaustion is stable: no candidate means no state change.
    assert_eq!(cs.reclaim_cached_leaves_until(5), 0);
    assert_eq!(cs.free_blocks, 4);
}

#[test]
fn plan_protects_match_while_reclaiming_exact_suffix_demand() {
    let config = SchedulerConfig {
        max_num_seqs: 1,
        max_num_batched_tokens: 4096,
    };
    let mut cs = CacheScheduler::init(config, 3);

    // Retain a two-page zero-ref chain from an earlier batch. The partial
    // tail returns immediately, leaving only one free block.
    let a = prompt(2 * PAGE + 44, 321_000);
    assert!(cs.allocate_prefill(7, &a));
    cs.publish_full_prefix_pages(7, &a, 0);
    let old_ids = cs.request_residency.get(&7).unwrap().block_ids.clone();
    let parent = old_ids[0];
    let old_child_hash = cs.blocks.get(&old_ids[1]).unwrap().hash_value;
    cs.deallocate(7);
    assert_eq!(cs.free_blocks, 1);

    // B shares A's first page but diverges in the second. Exact sizing needs
    // only two fresh suffix pages: queue reclamation removes A's child before
    // reaching the matched root. A whole-prompt target would
    // have peeled the root too and lost the hit.
    let mut b = a[..PAGE].to_vec();
    b.extend(prompt(PAGE + 44, 999_000));
    cs.live_requests.insert(
        8,
        RequestState::from_parts(
            8,
            b,
            Vec::new(),
            SamplerState::empty(),
            4,
            EosTokenSet::singleton(0),
            true,
        ),
    );
    cs.accepted_requests.insert(8, true);
    cs.waiting.push(8);

    let (plan, _perms) = cs.plan(None);
    assert_eq!(plan.scheduled_ids, vec![8]);
    let residency = cs.request_residency.get(&8).unwrap();
    assert_eq!(residency.cached_prefix_blocks, 1);
    assert_eq!(residency.block_ids[0], parent);
    assert_eq!(cs.blocks.get(&parent).unwrap().refcount, 1);
    assert!(!cs.hash_to_block.contains_key(&old_child_hash));
    // 1 free + 1 reclaimed - 2 fresh suffix pages.
    assert_eq!(cs.free_blocks, 0);
}

#[test]
fn plan_keeps_long_cached_prefix_when_only_one_suffix_page_is_needed() {
    let config = SchedulerConfig {
        max_num_seqs: 1,
        max_num_batched_tokens: 4096,
    };
    let mut cs = CacheScheduler::init(config, 4);

    // Three reusable pages plus a partial tail fill the pool. Deallocation
    // retains the registered chain and returns only the partial page.
    let tokens = prompt(3 * PAGE + 44, 654_000);
    assert!(cs.allocate_prefill(7, &tokens));
    cs.publish_full_prefix_pages(7, &tokens, 0);
    let cached_ids = cs.request_residency.get(&7).unwrap().block_ids[..3].to_vec();
    cs.deallocate(7);
    assert_eq!(cs.free_blocks, 1);

    cs.live_requests.insert(
        8,
        RequestState::from_parts(
            8,
            tokens,
            Vec::new(),
            SamplerState::empty(),
            4,
            EosTokenSet::singleton(0),
            true,
        ),
    );
    cs.accepted_requests.insert(8, true);
    cs.waiting.push(8);

    let (plan, _perms) = cs.plan(None);
    assert_eq!(plan.scheduled_ids, vec![8]);
    let residency = cs.request_residency.get(&8).unwrap();
    assert_eq!(residency.cached_prefix_blocks, 3);
    assert_eq!(&residency.block_ids[..3], cached_ids.as_slice());
    assert!(cached_ids
        .iter()
        .all(|bid| cs.blocks.get(bid).unwrap().refcount == 1));
    assert_eq!(cs.blocks.len(), 4);
    assert_eq!(cs.free_blocks, 0);
}

#[test]
fn plan_reclaims_decode_commit_headroom_before_selection() {
    let config = SchedulerConfig {
        max_num_seqs: 1,
        max_num_batched_tokens: 4096,
    };
    let mut cs = CacheScheduler::init(config, 3);

    // Two retained pages plus one free physical slot.
    let cached = prompt(2 * PAGE + 44, 555_000);
    assert!(cs.allocate_prefill(7, &cached));
    cs.publish_full_prefix_pages(7, &cached, 0);
    let cached_ids = cs.request_residency.get(&7).unwrap().block_ids.clone();
    let parent = cached_ids[0];
    let child_hash = cs.blocks.get(&cached_ids[1]).unwrap().hash_value;
    cs.deallocate(7);
    assert_eq!(cs.free_blocks, 1);

    // Consume the last free page with a running request whose full-page
    // tail needs one fresh page at commit. Decode selection would return an
    // empty schedule without the pre-selection reclamation pass.
    let running_prompt = prompt(PAGE, 777_000);
    cs.live_requests.insert(
        8,
        RequestState::from_parts(
            8,
            running_prompt.clone(),
            Vec::new(),
            SamplerState::empty(),
            4,
            EosTokenSet::singleton(0),
            true,
        ),
    );
    cs.accepted_requests.insert(8, true);
    assert!(cs.allocate_prefill(8, &running_prompt));
    cs.running.push(8);
    assert_eq!(cs.free_blocks, 0);

    let (plan, _perms) = cs.plan(None);
    assert_eq!(plan.scheduled_ids, vec![8]);
    assert_eq!(cs.free_blocks, 1);
    assert_eq!(cs.blocks.get(&parent).unwrap().refcount, 0);
    assert!(!cs.hash_to_block.contains_key(&child_hash));
}

#[test]
fn registration_records_exact_physical_prefix_chain() {
    let config = SchedulerConfig {
        max_num_seqs: 4,
        max_num_batched_tokens: 4096,
    };
    let mut cs = CacheScheduler::init(config, 16);

    // Two full pages plus a tail are enough to observe both the root and a
    // predecessor edge in the registered provenance chain.
    let tokens = prompt(2 * PAGE + 44, 12_000);
    assert!(cs.allocate_prefill(7, &tokens));
    cs.publish_full_prefix_pages(7, &tokens, 0);

    let ids = &cs.request_residency.get(&7).unwrap().block_ids;
    let first = cs.blocks.get(&ids[0]).unwrap();
    let second = cs.blocks.get(&ids[1]).unwrap();
    assert_eq!(first.prefix_depth, 1);
    assert_eq!(first.parent_block, None);
    assert_eq!(second.prefix_depth, 2);
    assert_eq!(second.parent_block, Some(ids[0]));
}

#[test]
fn scan_rejects_same_page_tokens_from_a_different_prefix_chain() {
    let config = SchedulerConfig {
        max_num_seqs: 4,
        max_num_batched_tokens: 4096,
    };
    let mut cs = CacheScheduler::init(config, 16);

    // A and X have identical second pages but different first pages.  Their
    // second-page K/V therefore belongs to different attention contexts.
    let a = prompt(2 * PAGE + 44, 20_000);
    let mut x = prompt(PAGE, 900_000);
    x.extend_from_slice(&a[PAGE..2 * PAGE]);
    x.extend(prompt(44, 950_000));
    assert_eq!(x.len(), 2 * PAGE + 44);

    assert!(cs.allocate_prefill(7, &a));
    cs.publish_full_prefix_pages(7, &a, 0);
    assert!(cs.allocate_prefill(8, &x));
    cs.publish_full_prefix_pages(8, &x, 0);

    let a_ids = cs.request_residency.get(&7).unwrap().block_ids.clone();
    let x_ids = cs.request_residency.get(&8).unwrap().block_ids.clone();
    assert_eq!(
        cs.blocks.get(&a_ids[1]).unwrap().tokens,
        cs.blocks.get(&x_ids[1]).unwrap().tokens,
    );
    assert_ne!(a_ids[0], x_ids[0]);

    // Model a rolling-hash collision: route A's second-prefix hash to X's
    // byte-identical second page and stamp it with the colliding value.  This
    // remains a structurally meaningful collision state; the crucial
    // difference is X's physical parent page.
    let a_h0 = chain_hash_span(0, &a, 0, PAGE);
    let a_h1 = chain_hash_span(a_h0, &a, PAGE, 2 * PAGE);
    let x_h0 = chain_hash_span(0, &x, 0, PAGE);
    let x_h1 = chain_hash_span(x_h0, &x, PAGE, 2 * PAGE);
    assert_ne!(a_h1, 0);
    assert_ne!(x_h1, 0);
    assert_ne!(a_h1, x_h1);

    let x_second = cs.blocks.get(&x_ids[1]).unwrap();
    let collision_entry = BlockEntry {
        tokens: x_second.tokens.clone(),
        refcount: x_second.refcount,
        hash_value: a_h1,
        prefix_depth: x_second.prefix_depth,
        parent_block: x_second.parent_block,
    };
    cs.blocks.insert(x_ids[1], collision_entry);
    cs.hash_to_block.remove(&x_h1);
    cs.hash_to_block.insert(a_h1, x_ids[1]);

    // Block 0 is accepted.  Block 1 has matching tokens and hash, but its
    // recorded parent is X[0], so the scan must stop instead of reusing K/V
    // materialized under the wrong context.
    let matched = cs.match_cached_prefix(&a, &Vec::new());
    assert_eq!(matched, vec![a_ids[0]]);

    // Releasing A must not evict the registry entry that belongs to X's
    // still-live colliding second page.
    cs.deallocate(7);
    assert_eq!(cs.hash_to_block.get(&a_h1), Some(&x_ids[1]));
}

#[test]
fn resident_child_prevents_parent_first_eviction_and_id_reuse() {
    let config = SchedulerConfig {
        max_num_seqs: 4,
        max_num_batched_tokens: 4096,
    };
    let mut cs = CacheScheduler::init(config, 4);
    let tokens = prompt(2 * PAGE + 1, 450_000);
    assert!(cs.allocate_prefill(7, &tokens));
    cs.publish_full_prefix_pages(7, &tokens, 0);
    let ids = cs.request_residency.get(&7).unwrap().block_ids.clone();
    let parent = ids[0];
    let child = ids[1];
    cs.deallocate(7);

    // The executable primitive accepts only the topological queue head.  A
    // resident child therefore prevents its parent from being recycled.
    assert_eq!(cs.cached_queue.head, Some(child));
    assert!(!cs.evict_cached_leaf(parent));
    assert!(cs.blocks.contains_key(&parent));
    assert!(cs.blocks.contains_key(&child));

    // Once the child is gone, the parent becomes the head and may be evicted.
    assert!(cs.evict_cached_leaf(child));
    assert_eq!(cs.cached_queue.head, Some(parent));
    assert!(cs.evict_cached_leaf(parent));
    assert!(!cs.blocks.contains_key(&child));

    // The parent id can be reused only after no resident child can carry an
    // edge to its prior lifetime.
    let replacement = vec![999_999];
    assert!(cs.allocate_prefill(8, &replacement));
    assert_eq!(cs.request_residency.get(&8).unwrap().block_ids[0], parent);
}

#[test]
fn successful_prefix_reuse_removes_the_matched_chain_from_the_free_queue() {
    let config = SchedulerConfig {
        max_num_seqs: 4,
        max_num_batched_tokens: 4096,
    };
    let mut cs = CacheScheduler::init(config, 8);
    let donor = prompt(2 * PAGE + 1, 500_000);
    assert!(cs.allocate_prefill(7, &donor));
    cs.publish_full_prefix_pages(7, &donor, 0);
    let cached = cs.request_residency.get(&7).unwrap().block_ids[..2].to_vec();
    cs.deallocate(7);
    for bid in &cached {
        assert!(cs.cached_queue.links.contains_key(bid));
    }

    let mut arrival = donor[..2 * PAGE].to_vec();
    arrival.push(900_000);
    assert!(cs.allocate_prefill_with_reuse(8, &arrival, &Vec::new()));
    assert_eq!(
        cs.request_residency.get(&8).unwrap().cached_prefix_blocks,
        2,
    );
    for bid in cached {
        assert!(!cs.cached_queue.links.contains_key(&bid));
        assert_eq!(cs.blocks.get(&bid).unwrap().refcount, 1);
    }
}
