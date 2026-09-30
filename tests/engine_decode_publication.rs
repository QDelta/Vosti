// Real engine-path coverage of generated-prefix publication timing.
#[path = "support/model_families/qwen3.rs"]
mod qwen3_fixture;

use qwen3_fixture::zero_layer_engine;
use vosti_verus::types::BLOCK_SIZE;
use vosti_verus::exec::request_state::{EosTokenSet, RequestState, SamplerState};

// These tests use the real Engine::step publication hook. The zero-layer
// fixture checks scheduling/publication timing, not GPU KV numeric fidelity.
fn decode_publication_waits_for_execution_and_survives_completion() {
    use vosti_verus::exec::request_state::{AdmissionStatus, NewRequest};
    let prompt = vec![1; BLOCK_SIZE as usize - 1];
    let request = RequestState::from_parts(
        7,
        prompt.clone(),
        vec![],
        SamplerState::empty(),
        2,
        EosTokenSet::singleton(999),
        true,
    );
    let mut engine = zero_layer_engine(vec![request], 1024, 256);
    let (first, _, _) = engine.step(None);
    let mut history = prompt;
    history.push(*first.get(&7).unwrap());
    let page = engine.cs.request_residency.get(&7).unwrap().block_ids[0];
    // Commit filled the token metadata, but the sample has not run forward.
    assert_eq!(
        engine.cs.blocks.get(&page).unwrap().tokens.len(),
        BLOCK_SIZE as usize
    );
    assert_eq!(engine.cs.blocks.get(&page).unwrap().prefix_depth, 0);
    let mut query = history.clone();
    query.push(3);
    assert!(engine.cs.match_cached_prefix(&query, &vec![]).is_empty());

    let (second, _, _) = engine.step(None);
    history.push(*second.get(&7).unwrap());
    assert!(!engine.cs.live_requests.contains_key(&7));
    let entry = engine.cs.blocks.get(&page).unwrap();
    assert_eq!(entry.prefix_depth, 1);
    assert_eq!(entry.refcount, 0);
    assert_eq!(entry.tokens, history[..BLOCK_SIZE as usize]);
    // The second sample is not included in the published page.
    assert_eq!(engine.cs.match_cached_prefix(&history, &vec![]), vec![page]);
    assert_eq!(
        engine.try_add_request(NewRequest {
            request_id: 8,
            prompt_tokens: history,
            max_tokens: 2,
            eos_token_ids: vec![999],
            ignore_eos: true,
        }),
        AdmissionStatus::Accepted
    );
    let (next, _, _) = engine.step(None);
    assert!(next.contains_key(&8));
    let reused = engine.cs.request_residency.get(&8).unwrap();
    assert_eq!(reused.cached_prefix_blocks, 1);
    assert_eq!(reused.block_ids[0], page);
}

fn decode_publication_reuses_live_donor_and_keeps_new_tail_private() {
    use vosti_verus::exec::request_state::{AdmissionStatus, NewRequest};
    let prompt = vec![2; BLOCK_SIZE as usize - 1];
    let request = RequestState::from_parts(
        7,
        prompt.clone(),
        vec![],
        SamplerState::empty(),
        5,
        EosTokenSet::singleton(999),
        true,
    );
    let mut engine = zero_layer_engine(vec![request], 1024, 256);
    let (first, _, _) = engine.step(None);
    let mut prefix = prompt;
    prefix.push(*first.get(&7).unwrap());
    engine.step(None);
    let ids = engine
        .cs
        .request_residency
        .get(&7)
        .unwrap()
        .block_ids
        .clone();
    assert_eq!(ids.len(), 2);
    assert_eq!(engine.cs.blocks.get(&ids[0]).unwrap().prefix_depth, 1);
    assert_eq!(engine.cs.blocks.get(&ids[1]).unwrap().prefix_depth, 0);
    assert_eq!(engine.cs.blocks.get(&ids[1]).unwrap().tokens.len(), 1);
    prefix.push(4);
    assert_eq!(
        engine.try_add_request(NewRequest {
            request_id: 8,
            prompt_tokens: prefix,
            max_tokens: 2,
            eos_token_ids: vec![999],
            ignore_eos: true,
        }),
        AdmissionStatus::Accepted
    );
    let (mixed, _, _) = engine.step(None);
    assert!(mixed.contains_key(&7) && mixed.contains_key(&8));
    let recipient = engine.cs.request_residency.get(&8).unwrap();
    assert_eq!(recipient.cached_prefix_blocks, 1);
    assert_eq!(recipient.block_ids[0], ids[0]);
    assert_eq!(engine.cs.blocks.get(&ids[0]).unwrap().refcount, 2);
    assert_ne!(recipient.block_ids[1], ids[1]);
}

fn decode_publication_extends_multiple_pages_without_publishing_final_sample() {
    let page_size = BLOCK_SIZE as usize;
    // On either side of a boundary, count materialized tokens rather than
    // sampled tokens. The final output never has KV in this request.
    for output_count in [1, page_size + 1, page_size + 2] {
        let prompt = vec![3; page_size - 1];
        let request = RequestState::from_parts(
            7,
            prompt.clone(),
            vec![],
            SamplerState::empty(),
            output_count,
            EosTokenSet::singleton(999),
            true,
        );
        let mut engine = zero_layer_engine(vec![request], 1024, 256);
        let mut history = prompt;
        for _ in 0..output_count {
            let (emitted, _, _) = engine.step(None);
            history.push(*emitted.get(&7).unwrap());
        }
        assert!(!engine.cs.live_requests.contains_key(&7));
        let expected_pages = (history.len() - 1) / page_size;
        // Add a query token so matching's leave-one-uncached guard does not
        // mask accidental publication of the last, unexecuted sample.
        let mut next_prompt = history.clone();
        next_prompt.push(5);
        let matched = engine.cs.match_cached_prefix(&next_prompt, &vec![]);
        assert_eq!(matched.len(), expected_pages, "output count {output_count}");
        for (j, &bid) in matched.iter().enumerate() {
            let entry = engine.cs.blocks.get(&bid).unwrap();
            assert_eq!(entry.refcount, 0);
            assert_eq!(entry.prefix_depth, j as u64 + 1);
            assert_eq!(
                entry.parent_block,
                if j == 0 { None } else { Some(matched[j - 1]) }
            );
            assert_eq!(entry.tokens, history[j * page_size..(j + 1) * page_size]);
        }
    }
}

fn simultaneous_decode_publications_keep_distinct_request_chains() {
    let requests = [7, 8]
        .into_iter()
        .map(|rid| {
            RequestState::from_parts(
                rid,
                vec![rid - 6; BLOCK_SIZE as usize - 1],
                vec![],
                SamplerState::empty(),
                2,
                EosTokenSet::singleton(999),
                true,
            )
        })
        .collect();
    let mut engine = zero_layer_engine(requests, 1024, 256);
    engine.step(None);
    let pages: Vec<_> = [7, 8]
        .into_iter()
        .map(|rid| engine.cs.request_residency.get(&rid).unwrap().block_ids[0])
        .collect();
    assert_ne!(pages[0], pages[1]);
    engine.step(None);
    for (i, bid) in pages.into_iter().enumerate() {
        let entry = engine.cs.blocks.get(&bid).unwrap();
        assert_eq!(entry.prefix_depth, 1);
        assert_eq!(entry.refcount, 0);
        let mut query = vec![i as u64 + 1; BLOCK_SIZE as usize - 1];
        query.extend([0, 4]);
        assert_eq!(engine.cs.match_cached_prefix(&query, &vec![]), vec![bid]);
    }
}

// Keep the long-lived embedded-Python scenarios sequential in one integration
// test binary; they do not need concurrent Rust test-worker threads.
#[test]
fn generated_prefix_publication_engine_scenarios() {
    decode_publication_waits_for_execution_and_survives_completion();
    decode_publication_reuses_live_donor_and_keeps_new_tail_private();
    decode_publication_extends_multiple_pages_without_publishing_final_sample();
    simultaneous_decode_publications_keep_distinct_request_chains();
}
