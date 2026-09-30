use vosti_verus::exec::cache_scheduler::StepMode;
use vosti_verus::exec::cache_scheduler::{BlockEntry, CacheScheduler, RequestResidency, SchedulerConfig};
use vosti_verus::types::BLOCK_SIZE;
use vosti_verus::exec::request_state::{EosTokenSet, RequestState, SamplerState};


#[test]
fn plan_returns_bounded_decode_plan() {
    let config = SchedulerConfig {
        max_num_seqs: 4,
        max_num_batched_tokens: 16,
    };
    let mut scheduler = CacheScheduler::init(config, 8);
    let mut prompt_tokens = Vec::new();
    prompt_tokens.push(11);
    let mut generated_tokens = Vec::new();
    generated_tokens.push(42);
    scheduler.live_requests.insert(
        7,
        RequestState::from_parts(
            7,
            prompt_tokens,
            generated_tokens,
            SamplerState::empty(),
            4,
            EosTokenSet::singleton(99),
            true,
        ),
    );
    scheduler.accepted_requests.insert(7, true);

    let mut block_ids = Vec::new();
    block_ids.push(0);
    let mut slot_mapping = Vec::new();
    slot_mapping.push(1);
    scheduler.request_residency.insert(
        7,
        RequestResidency {
            block_ids,
            cached_prefix_blocks: 0,
            slot_mapping,
        },
    );
    let mut block_tokens = Vec::new();
    block_tokens.push(11);
    block_tokens.push(42);
    scheduler.blocks.insert(
        0,
        BlockEntry {
            tokens: block_tokens,
            refcount: 1,
            hash_value: 0,
            prefix_depth: 0,
            parent_block: None,
        },
    );
    assert!(scheduler.free_queue.remove(0));
    scheduler.free_blocks = 7;
    scheduler.running.push(7);

    let (plan, _perms) = scheduler.plan(None);

    match plan.mode {
        StepMode::Decode => {}
        StepMode::Prefill | StepMode::Mixed => panic!("expected decode plan"),
    }
    assert_eq!(plan.scheduled_ids, vec![7]);
    assert_eq!(plan.max_seqlen_q, 1);
    assert_eq!(plan.max_seqlen_k, 2);
}

// Continuous batching: one running decode request + waiting prompts produce a
// single MIXED cu-partitioned plan (decode row q_len 1, final short-prefill
// rows q_len == prompt len) whose admissions respect max_num_batched_tokens.
#[test]
fn plan_mixes_running_decode_with_admitted_prefills() {
    let config = SchedulerConfig {
        max_num_seqs: 4,
        // Budget: decode row (1) + first prompt (2) fit; the second prompt
        // (3 tokens) would exceed 5 and must stay waiting.
        max_num_batched_tokens: 5,
    };
    let mut scheduler = CacheScheduler::init(config, 8);

    // Running request 7: prompt [11], generated [42] — one decode row.
    let mut prompt_tokens = Vec::new();
    prompt_tokens.push(11);
    let mut generated_tokens = Vec::new();
    generated_tokens.push(42);
    scheduler.live_requests.insert(
        7,
        RequestState::from_parts(
            7,
            prompt_tokens,
            generated_tokens,
            SamplerState::empty(),
            4,
            EosTokenSet::singleton(99),
            true,
        ),
    );
    scheduler.accepted_requests.insert(7, true);
    let mut block_ids = Vec::new();
    block_ids.push(0);
    let mut slot_mapping = Vec::new();
    slot_mapping.push(1);
    scheduler.request_residency.insert(
        7,
        RequestResidency {
            block_ids,
            cached_prefix_blocks: 0,
            slot_mapping,
        },
    );
    let mut block_tokens = Vec::new();
    block_tokens.push(11);
    block_tokens.push(42);
    scheduler.blocks.insert(
        0,
        BlockEntry {
            tokens: block_tokens,
            refcount: 1,
            hash_value: 0,
            prefix_depth: 0,
            parent_block: None,
        },
    );
    assert!(scheduler.free_queue.remove(0));
    scheduler.free_blocks = 7;
    scheduler.running.push(7);

    // Waiting request 8 (2-token prompt, fits) and 9 (3-token prompt, over budget).
    for (rid, len) in [(8u64, 2u64), (9u64, 3u64)] {
        let mut prompt = Vec::new();
        for t in 0..len {
            prompt.push(100 + t);
        }
        scheduler.live_requests.insert(
            rid,
            RequestState::from_parts(
                rid,
                prompt,
                Vec::new(),
                SamplerState::empty(),
                4,
                EosTokenSet::singleton(99),
                true,
            ),
        );
        scheduler.accepted_requests.insert(rid, true);
        scheduler.waiting.push(rid);
    }

    let (plan, _perms) = scheduler.plan(None);

    match plan.mode {
        StepMode::Mixed => {}
        StepMode::Decode | StepMode::Prefill => panic!("expected mixed plan"),
    }
    // Decode row for 7, admitted prefill row for 8; 9 stays waiting.
    assert_eq!(plan.scheduled_ids, vec![7, 8]);
    assert_eq!(scheduler.running, vec![7, 8]);
    assert_eq!(scheduler.waiting, vec![9]);
    assert_eq!(plan.max_seqlen_q, 2);
    assert_eq!(plan.max_seqlen_k, 2);
}

#[test]
fn plan_chunks_long_prefill_at_page_boundary_without_sampling() {
    let config = SchedulerConfig {
        max_num_seqs: 1,
        max_num_batched_tokens: BLOCK_SIZE as usize,
    };
    let mut scheduler = CacheScheduler::init(config, 16);
    let prompt: Vec<u64> = (0..(2 * BLOCK_SIZE)).map(|i| i % 8).collect();
    scheduler.live_requests.insert(
        7,
        RequestState::from_parts(
            7,
            prompt,
            Vec::new(),
            SamplerState::empty(),
            1,
            EosTokenSet::singleton(999),
            true,
        ),
    );
    scheduler.accepted_requests.insert(7, true);
    scheduler.waiting.push(7);

    let (plan, _perms) = scheduler.plan(None);

    assert_eq!(plan.scheduled_ids, vec![7]);
    assert_eq!(plan.sample_mask, vec![false]);
    assert_eq!(plan.max_seqlen_q, BLOCK_SIZE as usize);
    assert_eq!(plan.max_seqlen_k, BLOCK_SIZE as usize);
}

#[test]
fn plan_defers_prefix_publication_until_all_admissions_are_decided() {
    let config = SchedulerConfig {
        max_num_seqs: 2,
        max_num_batched_tokens: 1024,
    };
    let mut scheduler = CacheScheduler::init(config, 16);
    let shared: Vec<u64> = (0..BLOCK_SIZE + 44).map(|i| 10_000 + i).collect();

    for rid in [1u64, 2u64] {
        scheduler.live_requests.insert(
            rid,
            RequestState::from_parts(
                rid,
                shared.clone(),
                Vec::new(),
                SamplerState::empty(),
                4,
                EosTokenSet::singleton(99),
                true,
            ),
        );
        scheduler.accepted_requests.insert(rid, true);
        scheduler.waiting.push(rid);
    }

    let (plan, _perms) = scheduler.plan(None);
    assert_eq!(plan.scheduled_ids, vec![1, 2]);

    // Both match decisions happened against the registry at plan entry, which
    // was empty. The second request therefore cannot consume the first
    // request's not-yet-materialized K/V in the same forward.
    assert_eq!(
        scheduler
            .request_residency
            .get(&1)
            .unwrap()
            .cached_prefix_blocks,
        0,
    );
    assert_eq!(
        scheduler
            .request_residency
            .get(&2)
            .unwrap()
            .cached_prefix_blocks,
        0,
    );
    assert_ne!(
        scheduler.request_residency.get(&1).unwrap().block_ids[0],
        scheduler.request_residency.get(&2).unwrap().block_ids[0],
    );

    // Publication occurs after admission completes, so a subsequent matcher
    // can reuse the full first page.
    assert_eq!(scheduler.match_cached_prefix(&shared, &Vec::new()).len(), 1);
}
