use super::*;
use vosti_verus::exec::cache_scheduler::CacheScheduler;
use vosti_verus::exec::cache_scheduler::SchedulerConfig;
use vosti_verus::exec::step_observation::record_planned_prefix_reuse;

fn active_request(prompt_tokens: usize) -> ActiveRequest {
    let (events, _) = tokio_mpsc::unbounded_channel();
    ActiveRequest {
        prompt_tokens,
        cached_prompt_tokens: None,
        cache_observed: false,
        max_tokens: 128,
        emitted_tokens: 0,
        ignore_eos: true,
        events,
    }
}

fn scheduler() -> CacheScheduler {
    CacheScheduler::init(
        SchedulerConfig {
            max_num_seqs: 4,
            max_num_batched_tokens: 4096,
        },
        16,
    )
}

#[test]
fn reused_prefix_is_not_total_resident_kv_and_is_recorded_once() {
    let mut cs = scheduler();
    let page = BLOCK_SIZE as usize;
    let donor: Vec<u64> = (0..BLOCK_SIZE + 44).collect();
    assert!(cs.allocate_prefill(7, &donor));
    cs.publish_full_prefix_pages(7, &donor, 0);
    let mut prompt = donor[..page].to_vec();
    prompt.extend(10_000..10_044);
    assert!(cs.allocate_prefill_with_reuse(8, &prompt, &Vec::new()));
    assert_eq!(cs.request_residency.get(&8).unwrap().block_ids.len(), 2);
    let mut observations = Vec::new();
    record_planned_prefix_reuse(&cs, &vec![8], &mut observations);
    let mut active = HashMap::from([(8, active_request(prompt.len()))]);
    observe_cached_prompt_tokens(&observations, &mut active);
    assert_eq!(active[&8].cached_prompt_tokens, Some(page));
    // Later residency changes/removal must not overwrite the initial count.
    cs.request_residency.remove(&8);
    record_planned_prefix_reuse(&cs, &vec![8], &mut observations);
    observe_cached_prompt_tokens(&observations, &mut active);
    assert_eq!(active[&8].cached_prompt_tokens, Some(page));
}

#[test]
fn waiting_is_unknown_and_observed_cold_allocation_is_zero() {
    let mut cs = scheduler();
    let prompt = vec![1; BLOCK_SIZE as usize + 1];
    assert!(cs.allocate_prefill(7, &prompt));
    let mut active = HashMap::from([(7, active_request(prompt.len()))]);
    observe_cached_prompt_tokens(&[], &mut active);
    assert_eq!(active[&7].cached_prompt_tokens, None);
    let mut observations = Vec::new();
    record_planned_prefix_reuse(&cs, &vec![7], &mut observations);
    observe_cached_prompt_tokens(&observations, &mut active);
    assert_eq!(active[&7].cached_prompt_tokens, Some(0));
}

#[test]
fn first_step_completion_uses_snapshot_despite_removed_residency() {
    let mut cs = scheduler();
    assert!(cs.allocate_prefill(7, &vec![1; 128]));
    let mut observations = Vec::new();
    record_planned_prefix_reuse(&cs, &vec![7], &mut observations);
    cs.request_residency.remove(&7);
    let mut active = HashMap::from([(7, active_request(128))]);
    observe_cached_prompt_tokens(&observations, &mut active);
    assert_eq!(active[&7].cached_prompt_tokens, Some(0));
}

#[test]
fn invalid_counts_are_not_published() {
    let mut cs = scheduler();
    assert!(cs.allocate_prefill(7, &vec![1; 128]));
    cs.running.push(7);
    for blocks in [u64::MAX, 128 / BLOCK_SIZE + 1] {
        let mut active = HashMap::from([(7, active_request(128))]);
        let mut residency = cs.request_residency.remove(&7).unwrap();
        residency.cached_prefix_blocks = blocks;
        cs.request_residency.insert(7, residency);
        let mut observations = Vec::new();
        record_planned_prefix_reuse(&cs, &vec![7], &mut observations);
        observe_cached_prompt_tokens(&observations, &mut active);
        assert_eq!(active[&7].cached_prompt_tokens, None);
        // An invalid first observation must not turn into a later self-hit.
        observations[0].cached_prefix_blocks = Some(1);
        observe_cached_prompt_tokens(&observations, &mut active);
        assert_eq!(active[&7].cached_prompt_tokens, None);
    }
}

#[test]
fn cold_chunk_observation_is_not_overwritten_by_self_reuse() {
    let mut active = HashMap::from([(7, active_request(128))]);
    observe_cached_prompt_tokens(
        &[PlannedPrefixReuse { request_id: 7, cached_prefix_blocks: Some(0) }],
        &mut active,
    );
    observe_cached_prompt_tokens(
        &[PlannedPrefixReuse { request_id: 7, cached_prefix_blocks: Some(1) }],
        &mut active,
    );
    assert_eq!(active[&7].cached_prompt_tokens, Some(0));
}

#[test]
fn usage_distinguishes_unknown_from_zero() {
    let unknown = usage_json(128, 3, None);
    assert_eq!(
        unknown,
        json!({"prompt_tokens": 128, "completion_tokens": 3, "total_tokens": 131})
    );
    for cached in [0, 64] {
        assert_eq!(
            usage_json(128, 3, Some(cached))["prompt_tokens_details"]["cached_tokens"],
            cached
        );
    }
}

// Exercise response serialization without loading Python, a model, or a GPU.
#[tokio::test]
async fn cache_usage_reaches_both_response_kinds_streaming_and_nonstreaming() {
    let (sender, receiver) = std_mpsc::channel();
    let tokenizer_thread = thread::spawn(move || {
        while let Ok(command) = receiver.recv() {
            match command {
                TokenizerCommand::Decode { reply, .. } => {
                    let _ = reply.send(Ok("x".into()));
                }
                _ => panic!("unexpected tokenizer operation"),
            }
        }
    });
    let (commands, _) = std_mpsc::sync_channel(1);
    let state = AppState {
        commands,
        tokenizer: TokenizerClient { sender },
        next_request_id: Arc::new(AtomicU64::new(1)),
        architecture: "test".into(),
        model: "test".into(),
        deployment_sha256: "test".into(),
        max_model_len: 2048,
    };
    for kind in [ResponseKind::Completion, ResponseKind::Chat] {
        for cached in [None, Some(0), Some(64)] {
            for (streaming, include_usage) in [(false, false), (true, true), (true, false)] {
                let (events, received) = tokio_mpsc::unbounded_channel();
                events.send(GenerationEvent::Token(1)).unwrap();
                events
                    .send(GenerationEvent::Finished {
                        reason: "length",
                        tokens: 1,
                        cached_prompt_tokens: cached,
                    })
                    .unwrap();
                events.send(GenerationEvent::Done).unwrap();
                drop(events);
                let prepared = PreparedGeneration {
                    request_id: 1,
                    prompt_tokens: 128,
                    max_tokens: 1,
                    include_usage,
                    events: received,
                };
                let response = if streaming {
                    streaming_response(state.clone(), kind, prepared)
                } else {
                    nonstreaming_response(state.clone(), kind, prepared).await
                };
                let bytes = axum::body::to_bytes(response.into_body(), 65536)
                    .await
                    .unwrap();
                let body = String::from_utf8(bytes.to_vec()).unwrap();
                if streaming {
                    let chunks: Vec<Value> = body
                        .lines()
                        .filter_map(|line| line.strip_prefix("data: "))
                        .filter(|data| *data != "[DONE]")
                        .map(|data| serde_json::from_str(data).unwrap())
                        .collect();
                    let finish = chunks.last().unwrap();
                    assert_eq!(
                        finish["usage"],
                        if include_usage {
                            usage_json(128, 1, cached)
                        } else {
                            Value::Null
                        }
                    );
                } else {
                    let response: Value = serde_json::from_str(&body).unwrap();
                    assert_eq!(response["usage"], usage_json(128, 1, cached));
                }
            }
        }
    }
    drop(state);
    tokenizer_thread.join().unwrap();
}
