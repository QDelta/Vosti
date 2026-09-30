use vosti_verus::exec::cache_scheduler::{remove_request_id_from_queue, CacheScheduler, SchedulerConfig};
use vosti_verus::exec::request_state::{EosTokenSet, RequestState, SamplerState};

#[test]
fn remove_request_id_from_queue_removes_target_and_preserves_others() {
    let mut queue = Vec::new();
    queue.push(10);
    queue.push(20);
    queue.push(30);

    remove_request_id_from_queue(&mut queue, 20);

    assert_eq!(queue, vec![10, 30]);
}

#[test]
fn remove_request_id_from_queue_is_noop_when_absent() {
    let mut queue = Vec::new();
    queue.push(10);
    queue.push(30);

    remove_request_id_from_queue(&mut queue, 20);

    assert_eq!(queue, vec![10, 30]);
}

#[test]
fn remove_live_request_after_dequeue_removes_unqueued_request() {
    let config = SchedulerConfig {
        max_num_seqs: 4,
        max_num_batched_tokens: 16,
    };
    let mut scheduler = CacheScheduler::init(config, 8);
    let mut prompt_tokens = Vec::new();
    prompt_tokens.push(11);

    scheduler.live_requests.insert(
        20,
        RequestState::from_parts(
            20,
            prompt_tokens,
            Vec::new(),
            SamplerState::empty(),
            1,
            EosTokenSet::singleton(99),
            true,
        ),
    );
    scheduler.accepted_requests.insert(20, true);

    scheduler.remove_live_request_after_dequeue(20);

    assert!(!scheduler.live_requests.contains_key(&20));
}

#[test]
fn update_live_request_replaces_state() {
    let config = SchedulerConfig {
        max_num_seqs: 4,
        max_num_batched_tokens: 16,
    };
    let mut scheduler = CacheScheduler::init(config, 8);
    let mut prompt_tokens = Vec::new();
    prompt_tokens.push(11);

    scheduler.live_requests.insert(
        20,
        RequestState::from_parts(
            20,
            prompt_tokens.clone(),
            Vec::new(),
            SamplerState::empty(),
            2,
            EosTokenSet::singleton(99),
            true,
        ),
    );
    scheduler.accepted_requests.insert(20, true);

    let mut generated_tokens = Vec::new();
    generated_tokens.push(42);
    scheduler.update_live_request(
        20,
        RequestState::from_parts(
            20,
            prompt_tokens,
            generated_tokens,
            SamplerState::empty(),
            2,
            EosTokenSet::singleton(99),
            true,
        ),
    );

    let updated = scheduler
        .live_requests
        .get(&20)
        .expect("request still live");
    assert_eq!(updated.generated_tokens, vec![42]);
}
