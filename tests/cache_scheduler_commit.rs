use vstd::hash_map::HashMapWithView;
use vosti_verus::exec::cache_scheduler::{
    record_emitted_token, CacheScheduler, EmittedTokens, SampleResult, SampleResults,
    SchedulerConfig,
};
use vosti_verus::exec::request_state::{EosTokenSet, RequestState, SamplerState};

#[test]
fn record_emitted_token_projects_sample_token() {
    let mut emitted = SampleResults::new();
    emitted.insert(
        7,
        SampleResult {
            sampler_state: SamplerState::empty(),
            token: 42,
        },
    );

    let mut out: EmittedTokens = HashMapWithView::new();
    record_emitted_token(&mut out, &emitted, 7, 42);

    assert_eq!(out.get(&7), Some(&42));
}

#[test]
fn commit_sample_for_request_records_and_updates_live_state() {
    let config = SchedulerConfig {
        max_num_seqs: 4,
        max_num_batched_tokens: 16,
    };
    let mut scheduler = CacheScheduler::init(config, 8);
    let mut prompt_tokens = Vec::new();
    prompt_tokens.push(11);
    scheduler.live_requests.insert(
        7,
        RequestState::from_parts(
            7,
            prompt_tokens,
            Vec::new(),
            SamplerState::empty(),
            2,
            EosTokenSet::singleton(99),
            true,
        ),
    );
    scheduler.accepted_requests.insert(7, true);

    let sample = SampleResult {
        sampler_state: SamplerState::empty(),
        token: 42,
    };
    let mut emitted = SampleResults::new();
    emitted.insert(7, sample);
    let mut out: EmittedTokens = HashMapWithView::new();
    let stored_sample = emitted.get(&7).expect("sample is present");

    scheduler.commit_sample_for_request(&mut out, &emitted, 7, stored_sample);

    assert_eq!(out.get(&7), Some(&42));
    let updated = scheduler
        .live_requests
        .get(&7)
        .expect("request remains live");
    assert_eq!(updated.generated_tokens, vec![42]);
}
