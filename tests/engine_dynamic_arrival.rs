#[path = "support/model_families/qwen3.rs"]
mod qwen3_fixture;

use qwen3_fixture::zero_layer_engine;
use vosti_verus::exec::request_state::{
    AdmissionStatus, EosTokenSet, NewRequest, RequestState, SamplerState,
};

// Zero-layer end-to-end runtime check: request 7 is already decoding when
// request 8 arrives.  The next engine step must form a mixed batch containing
// both the old decode row and the newly admitted prefill row.
fn run_dynamic_arrival() -> (
    bool,
    bool,
    usize,
    usize,
    usize,
    bool,
    usize,
    bool,
    bool,
    usize,
    usize,
    bool,
) {
    let mut initial_prompt: Vec<u64> = Vec::new();
    initial_prompt.push(1);
    initial_prompt.push(2);
    initial_prompt.push(3);
    let initial = RequestState::from_parts(
        7,
        initial_prompt,
        Vec::<u64>::new(),
        SamplerState::empty(),
        3,
        EosTokenSet::singleton(999),
        true,
    );
    let mut requests = Vec::<RequestState>::new();
    requests.push(initial);

    let mut engine = zero_layer_engine(requests, 1024, 16);

    let (first, _samples1, _reprs1) = engine.step(None);

    let mut new_prompt = Vec::<u64>::new();
    new_prompt.push(4);
    new_prompt.push(5);
    let status = engine.try_add_request(NewRequest {
        request_id: 8,
        prompt_tokens: new_prompt,
        max_tokens: 1,
        eos_token_ids: vec![999],
        ignore_eos: true,
    });
    let accepted = status == AdmissionStatus::Accepted;
    let running_after_arrival = engine.cs.running.len();
    let waiting_after_arrival = engine.cs.waiting.len();
    let live_after_arrival = engine.cs.live_requests.len();

    let mut duplicate_prompt = Vec::<u64>::new();
    duplicate_prompt.push(6);
    let duplicate = engine.try_add_request(NewRequest {
        request_id: 8,
        prompt_tokens: duplicate_prompt,
        max_tokens: 1,
        eos_token_ids: vec![999],
        ignore_eos: true,
    });
    let duplicate_rejected = duplicate == AdmissionStatus::DuplicateRequestId;
    let waiting_after_duplicate = engine.cs.waiting.len();

    let (second, _samples2, _reprs2) = engine.step(None);
    let completed_duplicate = engine.try_add_request(NewRequest {
        request_id: 8,
        prompt_tokens: vec![7],
        max_tokens: 1,
        eos_token_ids: vec![999],
        ignore_eos: true,
    });
    (
        first.contains_key(&7),
        accepted,
        running_after_arrival,
        waiting_after_arrival,
        live_after_arrival,
        duplicate_rejected,
        waiting_after_duplicate,
        second.contains_key(&7),
        second.contains_key(&8),
        second.len(),
        engine.cs.live_requests.len(),
        completed_duplicate == AdmissionStatus::DuplicateRequestId,
    )
}

#[test]
fn arrival_between_steps_joins_next_mixed_batch() {
    let (
        initial_emitted,
        accepted,
        running_after_arrival,
        waiting_after_arrival,
        live_after_arrival,
        duplicate_rejected,
        waiting_after_duplicate,
        old_emitted_again,
        new_emitted,
        second_emitted_len,
        live_after_second,
        completed_duplicate_rejected,
    ) = run_dynamic_arrival();

    assert!(initial_emitted);
    assert!(accepted);
    assert_eq!(running_after_arrival, 1);
    assert_eq!(waiting_after_arrival, 1);
    assert_eq!(live_after_arrival, 2);
    assert!(duplicate_rejected);
    assert_eq!(waiting_after_duplicate, 1);
    assert!(old_emitted_again);
    assert!(new_emitted);
    assert_eq!(second_emitted_len, 2);
    assert_eq!(live_after_second, 1);
    assert!(completed_duplicate_rejected);
}

#[test]
fn admission_enforces_the_bounded_nonempty_eos_policy() {
    let mut engine = zero_layer_engine(Vec::new(), 1024, 16);

    let empty = engine.try_add_request(NewRequest {
        request_id: 10,
        prompt_tokens: vec![1],
        max_tokens: 1,
        eos_token_ids: vec![],
        ignore_eos: false,
    });
    assert_eq!(empty, AdmissionStatus::EmptyEosTokenIds);

    let too_many = engine.try_add_request(NewRequest {
        request_id: 11,
        prompt_tokens: vec![1],
        max_tokens: 1,
        eos_token_ids: vec![1, 2, 3, 4],
        ignore_eos: false,
    });
    assert_eq!(too_many, AdmissionStatus::TooManyEosTokenIds);

    let bounded = engine.try_add_request(NewRequest {
        request_id: 12,
        prompt_tokens: vec![1],
        max_tokens: 1,
        eos_token_ids: vec![1, 2, 3],
        ignore_eos: false,
    });
    assert_eq!(bounded, AdmissionStatus::Accepted);
    assert_eq!(engine.cs.live_requests.len(), 1);
}
