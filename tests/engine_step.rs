// End-to-end executable engine check:
//   CacheScheduler.InitWithRequests -> Plan(prefill) -> ModelForward
//   -> Sample -> Commit.
//
// The model is intentionally zero-layer so this test exercises the real
// top-level data path without needing layer KV-cache kernels.

#[path = "support/model_families/qwen3.rs"]
mod qwen3_fixture;

use qwen3_fixture::zero_layer_engine;
use vosti_verus::boundary::model_families::qwen3::deployment::init_staged_runtime_for_tests;
use vosti_verus::types::BLOCK_SIZE;
use vosti_verus::exec::request_state::{EosTokenSet, RequestState, SamplerState};
use vosti_verus::boundary::tensor_runtime::{model_runtime_admitted_by_runtime_gate, ModelRuntime};

#[test]
fn staged_scheduler_fixture_cannot_admit_a_nonzero_layer_model() {
    let runtime = ModelRuntime::Qwen3(init_staged_runtime_for_tests("qwen3-0.6b"));
    assert_eq!(
        model_runtime_admitted_by_runtime_gate(&runtime, 0),
        cfg!(debug_assertions),
    );
    assert!(!model_runtime_admitted_by_runtime_gate(&runtime, 1));
}

fn run_engine_step() -> (usize, bool, u64, usize, usize) {
    let mut prompt_tokens: Vec<u64> = Vec::new();
    prompt_tokens.push(1);
    prompt_tokens.push(2);
    prompt_tokens.push(3);

    let request = RequestState::from_parts(
        7,
        prompt_tokens,
        Vec::<u64>::new(),
        SamplerState::empty(),
        1,
        EosTokenSet::singleton(999),
        true,
    );
    let mut requests: Vec<RequestState> = Vec::new();
    requests.push(request);

    let mut engine = zero_layer_engine(requests, 1024, 16);

    let (emitted, _samples, _reprs) = engine.step(None);
    let token = match emitted.get(&7) {
        Some(t) => *t,
        None => 9999,
    };

    (
        emitted.len(),
        emitted.contains_key(&7),
        token,
        engine.cs.running.len(),
        engine.cs.waiting.len(),
    )
}

fn run_two_step_decode_slot_inspection(
) -> (usize, u64, usize, usize, usize, usize, u64, usize, usize) {
    let mut prompt_tokens: Vec<u64> = Vec::new();
    prompt_tokens.push(1);
    prompt_tokens.push(2);
    prompt_tokens.push(3);

    let request = RequestState::from_parts(
        7,
        prompt_tokens,
        Vec::<u64>::new(),
        SamplerState::empty(),
        3,
        EosTokenSet::singleton(999),
        true,
    );
    let mut requests: Vec<RequestState> = Vec::new();
    requests.push(request);

    let mut engine = zero_layer_engine(requests, 1024, 16);

    let (first, _s1, _r1) = engine.step(None);
    let first_emitted_len = first.len();
    let mut slot_len: usize = 9999;
    let mut slot_value: u64 = 9999;
    let mut block_token_len: usize = 9999;
    match engine.cs.request_residency.get(&7) {
        Some(residency) => {
            slot_len = residency.slot_mapping.len();
            if slot_len > 0 {
                slot_value = residency.slot_mapping[0];
            }
            if residency.block_ids.len() > 0 {
                let bid = residency.block_ids[residency.block_ids.len() - 1];
                match engine.cs.blocks.get(&bid) {
                    Some(block) => {
                        block_token_len = block.tokens.len();
                    }
                    None => {}
                }
            }
        }
        None => {}
    }
    let running_after_first = engine.cs.running.len();

    let (second, _s2, _r2) = engine.step(None);
    let mut slot_len_after_second: usize = 9999;
    let mut slot_value_after_second: u64 = 9999;
    let mut block_token_len_after_second: usize = 9999;
    match engine.cs.request_residency.get(&7) {
        Some(residency) => {
            slot_len_after_second = residency.slot_mapping.len();
            if slot_len_after_second > 0 {
                slot_value_after_second = residency.slot_mapping[0];
            }
            if residency.block_ids.len() > 0 {
                let bid = residency.block_ids[residency.block_ids.len() - 1];
                match engine.cs.blocks.get(&bid) {
                    Some(block) => {
                        block_token_len_after_second = block.tokens.len();
                    }
                    None => {}
                }
            }
        }
        None => {}
    }
    (
        slot_len,
        slot_value,
        block_token_len,
        running_after_first,
        first_emitted_len,
        second.len(),
        slot_value_after_second,
        slot_len_after_second,
        block_token_len_after_second,
    )
}

fn run_full_tail_append_inspection() -> (usize, u64, usize, usize, u64, usize) {
    let mut prompt_tokens: Vec<u64> = Vec::new();
    let mut p: usize = 0;
    while p < BLOCK_SIZE as usize {
        prompt_tokens.push((p % 8) as u64);
        p = p + 1;
    }

    let request = RequestState::from_parts(
        7,
        prompt_tokens,
        Vec::<u64>::new(),
        SamplerState::empty(),
        2,
        EosTokenSet::singleton(999),
        true,
    );
    let mut requests: Vec<RequestState> = Vec::new();
    requests.push(request);

    let mut engine = zero_layer_engine(requests, 4096, 512);

    let (first, _s1, _r1) = engine.step(None);
    let first_emitted_len = first.len();
    let mut slot_len: usize = 9999;
    let mut slot_value: u64 = 9999;
    let mut block_ids_len: usize = 9999;
    let mut tail_block_len: usize = 9999;
    let mut tail_block_id: u64 = 9999;
    match engine.cs.request_residency.get(&7) {
        Some(residency) => {
            slot_len = residency.slot_mapping.len();
            if slot_len > 0 {
                slot_value = residency.slot_mapping[0];
            }
            block_ids_len = residency.block_ids.len();
            if residency.block_ids.len() > 0 {
                tail_block_id = residency.block_ids[residency.block_ids.len() - 1];
                match engine.cs.blocks.get(&tail_block_id) {
                    Some(block) => {
                        tail_block_len = block.tokens.len();
                    }
                    None => {}
                }
            }
        }
        None => {}
    }
    (
        slot_len,
        slot_value,
        block_ids_len,
        tail_block_len,
        tail_block_id,
        first_emitted_len,
    )
}

#[test]
fn engine_prefill_step_emits_and_finishes_request() {
    let (emitted_len, has_request, token, running_len, waiting_len) = run_engine_step();
    assert_eq!(emitted_len, 1);
    assert!(has_request);
    assert_eq!(token, 0);
    assert_eq!(running_len, 0);
    assert_eq!(waiting_len, 0);
}

#[test]
fn chunked_prefill_parks_then_reuses_prefix_and_emits() {
    let prompt_tokens: Vec<u64> = (0..(3 * BLOCK_SIZE)).map(|i| i % 8).collect();
    let request = RequestState::from_parts(
        7,
        prompt_tokens,
        Vec::<u64>::new(),
        SamplerState::empty(),
        1,
        EosTokenSet::singleton(999),
        true,
    );
    let mut engine = zero_layer_engine(vec![request], 4096, BLOCK_SIZE as usize);

    let (first, _first_samples, _first_reprs) = engine.step(None);
    assert!(first.is_empty());
    assert!(engine.cs.running.is_empty());
    assert_eq!(engine.cs.waiting, vec![7]);
    assert!(!engine.cs.request_residency.contains_key(&7));
    assert!(!engine.cs.hash_to_block.is_empty());

    let (second, _second_samples, _second_reprs) = engine.step(None);
    assert!(second.is_empty());
    assert!(engine.cs.running.is_empty());
    assert_eq!(engine.cs.waiting, vec![7]);
    assert!(!engine.cs.request_residency.contains_key(&7));

    let (third, _third_samples, _third_reprs) = engine.step(None);
    assert_eq!(third.get(&7), Some(&0));
    assert!(!engine.cs.live_requests.contains_key(&7));
    assert!(engine.cs.running.is_empty());
    assert!(engine.cs.waiting.is_empty());
}

#[test]
fn mixed_step_parks_chunked_row_and_emits_final_row() {
    let chunked = RequestState::from_parts(
        7,
        (0..(2 * BLOCK_SIZE)).map(|i| i % 8).collect(),
        Vec::<u64>::new(),
        SamplerState::empty(),
        1,
        EosTokenSet::singleton(999),
        true,
    );
    let final_row = RequestState::from_parts(
        8,
        vec![1],
        Vec::<u64>::new(),
        SamplerState::empty(),
        1,
        EosTokenSet::singleton(999),
        true,
    );
    let mut engine = zero_layer_engine(vec![chunked, final_row], 4096, BLOCK_SIZE as usize + 1);

    let (emitted, _samples, _reprs) = engine.step(None);

    assert_eq!(emitted.len(), 1);
    assert_eq!(emitted.get(&8), Some(&0));
    assert!(engine.cs.live_requests.contains_key(&7));
    assert!(!engine.cs.live_requests.contains_key(&8));
    assert_eq!(engine.cs.waiting, vec![7]);
    assert!(engine.cs.running.is_empty());
    assert!(!engine.cs.request_residency.contains_key(&7));
}

#[test]
fn commit_sets_singleton_decode_slot_for_generated_token() {
    let (
        slot_len,
        slot_value,
        block_token_len,
        running_after_first,
        first_emitted_len,
        second_emitted_len,
        slot_value_after_second,
        slot_len_after_second,
        block_token_len_after_second,
    ) = run_two_step_decode_slot_inspection();
    assert_eq!(first_emitted_len, 1);
    assert_eq!(slot_len, 1);
    assert_eq!(slot_value, 3);
    assert_eq!(block_token_len, 4);
    assert_eq!(running_after_first, 1);
    assert_eq!(second_emitted_len, 1);
    assert_eq!(slot_len_after_second, 1);
    assert_eq!(slot_value_after_second, 4);
    assert_eq!(block_token_len_after_second, 5);
}

#[test]
fn commit_allocates_new_tail_block_when_decode_tail_is_full() {
    let (slot_len, slot_value, block_ids_len, tail_block_len, tail_block_id, first_emitted_len) =
        run_full_tail_append_inspection();
    assert_eq!(first_emitted_len, 1);
    assert_eq!(slot_len, 1);
    assert_eq!(slot_value, BLOCK_SIZE);
    assert_eq!(block_ids_len, 2);
    assert_eq!(tail_block_len, 1);
    assert_eq!(tail_block_id, 1);
}

#[test]
#[should_panic(expected = "Engine::init requires fresh requests")]
fn init_rejects_request_with_generated_history() {
    let request = RequestState::from_parts(
        7,
        vec![1],
        vec![2],
        SamplerState::empty(),
        2,
        EosTokenSet::singleton(999),
        true,
    );
    let _engine = zero_layer_engine(vec![request], 1024, 16);
}

#[test]
#[should_panic(expected = "Engine::init requires a positive generation budget")]
fn init_rejects_zero_generation_budget() {
    let request = RequestState::from_parts(
        7,
        vec![1],
        Vec::<u64>::new(),
        SamplerState::empty(),
        0,
        EosTokenSet::singleton(999),
        true,
    );
    let _engine = zero_layer_engine(vec![request], 1024, 16);
}
