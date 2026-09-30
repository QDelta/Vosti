#[path = "../examples/support/engine_benchmark.rs"]
#[allow(dead_code)]
mod engine_benchmark;
#[path = "../examples/support/engine_serving.rs"]
#[allow(dead_code, unused_imports)]
mod engine_serving;
#[path = "../examples/support/engine_setup.rs"]
#[allow(dead_code)]
mod engine_setup;
#[path = "../examples/support/output_tokens.rs"]
#[allow(dead_code)]
mod output_tokens;
#[path = "support/model_families/qwen3.rs"]
mod qwen3_fixture;

use vosti_verus::exec::engine::Engine;
use vosti_verus::exec::request_state::{EosTokenSet, RequestState, SamplerState};

fn request(id: u64, tokens: usize, value: u64, output: usize) -> RequestState {
    RequestState::from_parts(
        id,
        vec![value; tokens],
        Vec::new(),
        SamplerState::empty(),
        output,
        EosTokenSet::singleton(999),
        true,
    )
}

fn observations(engine: &Engine) -> Vec<(u64, Option<u64>)> {
    engine
        .last_step_prefix_reuse
        .iter()
        .map(|row| (row.request_id, row.cached_prefix_blocks))
        .collect()
}

// Keep embedded Python execution on one test thread/lifetime.
#[test]
fn actual_first_plan_survives_chunk_parking_completion_and_mixed_batches() {
    let mut cold = qwen3_fixture::zero_layer_engine(vec![request(7, 128, 1, 2)], 1024, 64);
    cold.step(None);
    assert!(!cold.cs.request_residency.contains_key(&7));
    assert_eq!(observations(&cold), vec![(7, Some(0))]);
    cold.step(None);
    assert_eq!(observations(&cold), vec![(7, Some(1))]);
    cold.step(None);
    assert!(cold.cs.live_requests.is_empty());
    assert_eq!(observations(&cold), vec![(7, Some(1))]);
    cold.step(None);
    assert!(observations(&cold).is_empty());

    // A finished donor contributes one page. The followup still needs two
    // chunks; its first hit is one page, not its later two-page self-hit.
    let mut warm = qwen3_fixture::zero_layer_engine(vec![request(10, 65, 1, 1)], 1024, 64);
    warm.step(None);
    warm.step(None);
    assert!(warm.cs.live_requests.is_empty());
    warm.add_request(request(11, 192, 1, 1));
    warm.step(None);
    assert_eq!(observations(&warm), vec![(11, Some(1))]);
    assert!(!warm.cs.request_residency.contains_key(&11));
    warm.step(None);
    assert_eq!(observations(&warm), vec![(11, Some(2))]);
    assert!(warm.cs.live_requests.is_empty());

    let mut mixed = qwen3_fixture::zero_layer_engine(vec![request(20, 17, 1, 3)], 1024, 64);
    mixed.step(None);
    mixed.add_request(request(21, 17, 2, 1));
    mixed.step(None);
    assert_eq!(observations(&mixed), vec![(20, Some(0)), (21, Some(0))]);
    assert!(!mixed.cs.request_residency.contains_key(&21));

    let mut delayed = qwen3_fixture::zero_layer_engine(
        vec![request(30, 17, 1, 2), request(31, 17, 2, 1)],
        1024,
        64,
    );
    delayed.cs.config.max_num_seqs = 1;
    delayed.step(None);
    assert_eq!(observations(&delayed), vec![(30, Some(0))]);
    delayed.step(None);
    assert_eq!(observations(&delayed), vec![(30, Some(0))]);
    delayed.step(None);
    assert_eq!(observations(&delayed), vec![(31, Some(0))]);

    // The native benchmark must count the first chunk even if the request
    // completes in its final prefill step and never retains residency.
    let mut native = qwen3_fixture::zero_layer_engine(Vec::new(), 1024, 64);
    let prepared = vec![engine_benchmark::PreparedRequest {
        prompt_tokens: vec![1; 128],
        max_tokens: 1,
        arrival_step: 0,
    }];
    let result =
        engine_benchmark::run_prepared_requests(&mut native, None, &prepared, 40, &[999]).unwrap();
    assert_eq!(result.admitted_requests, 1);
    assert_eq!(result.requests_with_reuse, 0);
    assert_eq!(result.reused_prefix_blocks, 0);
    let result =
        engine_benchmark::run_prepared_requests(&mut native, None, &prepared, 41, &[999]).unwrap();
    assert_eq!(result.admitted_requests, 1);
    assert_eq!(result.requests_with_reuse, 1);
    assert_eq!(result.reused_prefix_blocks, 1);
}
