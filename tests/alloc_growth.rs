// Regression probe: appends into a non-full tail must not allocate a block.
use vosti_verus::exec::cache_scheduler::{CacheScheduler, SchedulerConfig};

#[test]
fn appends_do_not_leak_blocks() {
    let config = SchedulerConfig {
        max_num_seqs: 4,
        max_num_batched_tokens: 4096,
    };
    let mut scheduler = CacheScheduler::init(config, 64);
    let block_size = vosti_verus::types::BLOCK_SIZE as usize;
    let prompt_len = 300usize;
    let prompt: Vec<u64> = (0..prompt_len as u64).collect();

    assert!(scheduler.allocate_prefill(7, &prompt));
    let expected_blocks = (prompt_len - 1) / block_size + 1;
    let blocks_before = scheduler.request_residency.get(&7).unwrap().block_ids.len();
    let free_before = scheduler.free_blocks;
    assert_eq!(blocks_before, expected_blocks);

    let tail_room = expected_blocks * block_size - prompt_len;
    for offset in 0..tail_room as u64 {
        scheduler.append_token(7, 1000 + offset);
    }

    let blocks_after = scheduler.request_residency.get(&7).unwrap().block_ids.len();
    assert_eq!(blocks_after, expected_blocks);
    assert_eq!(scheduler.free_blocks, free_before);
}
