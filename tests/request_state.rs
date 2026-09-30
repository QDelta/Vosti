use vosti_verus::exec::request_state::{
    should_finish_after_append_exec, EosTokenSet, RequestState, SamplerState,
};

fn request_with_generated(
    generated: &[u64],
    max_tokens: usize,
    eos_token_ids: &[u64],
    ignore_eos: bool,
) -> RequestState {
    let mut prompt_tokens = Vec::new();
    prompt_tokens.push(11);

    let mut generated_tokens = Vec::new();
    for token in generated {
        generated_tokens.push(*token);
    }
    let mut eos_tokens = Vec::new();
    for token in eos_token_ids {
        eos_tokens.push(*token);
    }
    let eos_token_set = EosTokenSet::from_nonempty_bounded(&eos_tokens);

    RequestState::from_parts(
        7,
        prompt_tokens,
        generated_tokens,
        SamplerState::empty(),
        max_tokens,
        eos_token_set,
        ignore_eos,
    )
}

#[test]
fn should_finish_after_append_covers_max_tokens_multiple_eos_and_continue() {
    let max_case = request_with_generated(&[1], 2, &[99, 100], true);
    assert!(should_finish_after_append_exec(&max_case, 42));

    let eos_case = request_with_generated(&[], 4, &[99, 100, 101], false);
    assert!(should_finish_after_append_exec(&eos_case, 99));
    assert!(should_finish_after_append_exec(&eos_case, 100));
    assert!(should_finish_after_append_exec(&eos_case, 101));

    let continuing_case = request_with_generated(&[1], 4, &[99, 100], false);
    assert!(!should_finish_after_append_exec(&continuing_case, 42));
}
