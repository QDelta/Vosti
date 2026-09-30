// Architecture-neutral serving configuration for executable Engine examples.
//
// A model-family example should only choose family defaults and provide a
// qualified weight/runtime tuple. Environment parsing, fresh-request
// construction, cache sizing, scheduler setup, and graph-overlay setup use
// this shared protocol.

use vosti_verus::model_config::ModelConfig;
use pyo3::types::PyAnyMethods;
use vstd::prelude::Tracked;
use vosti_verus::exec::request_state::{EosTokenSet, RequestState, SamplerState, MAX_EOS_TOKEN_IDS};

use vosti_verus::boundary::tensor_runtime as RT;

pub use crate::engine_setup::{
    checkpoint_eos_token_ids, deployment_environment, env_u64, env_usize,
    initialize_qualified_engine,
};

pub fn env_present(name: &str) -> bool {
    std::env::var_os(name).is_some()
}

pub fn fresh_request(
    request_id: u64,
    prompt_tokens: Vec<u64>,
    max_tokens: usize,
    eos_token_ids: Vec<u64>,
    ignore_eos: bool,
) -> Result<RequestState, String> {
    if prompt_tokens.is_empty() {
        return Err("request prompt must be nonempty".to_string());
    }
    if max_tokens == 0 {
        return Err("request generation budget must be positive".to_string());
    }
    if eos_token_ids.is_empty() {
        return Err("request EOS token-id list must be nonempty".to_string());
    }
    if eos_token_ids.len() > MAX_EOS_TOKEN_IDS {
        return Err(format!(
            "request has {} EOS token ids; at most {MAX_EOS_TOKEN_IDS} are supported",
            eos_token_ids.len(),
        ));
    }
    let history_capacity = prompt_tokens
        .len()
        .checked_add(max_tokens)
        .ok_or_else(|| "request history capacity exceeds usize".to_string())?;
    u64::try_from(history_capacity)
        .map_err(|_| "request history capacity exceeds u64".to_string())?;
    let eos_token_set = EosTokenSet::from_nonempty_bounded(&eos_token_ids);
    Ok(RequestState::from_parts(
        request_id,
        prompt_tokens,
        Vec::new(),
        SamplerState::empty(),
        max_tokens,
        eos_token_set,
        ignore_eos,
    ))
}

pub fn set_python_environment(name: &str, value: &str) {
    std::env::set_var(name, value);
    pyo3::Python::with_gil(|py| {
        let os = py.import_bound("os").expect("failed to import Python os");
        os.getattr("environ")
            .expect("Python os has no environ mapping")
            .set_item(name, value)
            .expect("failed to update Python os.environ");
    });
}

fn prompt_token_ids(default_prompt_ids: &str) -> Vec<u64> {
    let value =
        std::env::var("VOSTI_PROMPT_IDS").unwrap_or_else(|_| default_prompt_ids.to_string());
    let tokens = value
        .split(',')
        .map(str::trim)
        .filter(|part| !part.is_empty())
        .map(|part| {
            part.parse::<u64>()
                .unwrap_or_else(|_| panic!("VOSTI_PROMPT_IDS contains non-integer {part:?}"))
        })
        .collect::<Vec<_>>();
    assert!(!tokens.is_empty(), "VOSTI_PROMPT_IDS must be nonempty");
    tokens
}

pub fn run_qualified_model(
    architecture: &str,
    default_prompt_ids: &str,
    eos_token_ids: Vec<u64>,
    weights: RT::ModelWeights,
    runtime: RT::ModelRuntime,
    weights_perms: Tracked<RT::ModelWeightsPerms>,
    model_config: ModelConfig,
) {
    let num_blocks = env_u64("VOSTI_NUM_BLOCKS", 32);

    let benchmark_input = std::env::var_os("VOSTI_BENCH_INPUT");
    let multi_calls = std::env::var_os("VOSTI_BENCH_CALLS");
    let benchmark_requests = benchmark_input.as_ref().map(|path| {
        crate::engine_benchmark::read_prepared_requests(std::path::Path::new(path))
            .unwrap_or_else(|error| panic!("invalid VOSTI_BENCH_INPUT: {error}"))
    });
    let prompt_tokens = prompt_token_ids(default_prompt_ids);
    let max_tokens = env_usize("VOSTI_MAX_TOKENS", 8);
    assert!(max_tokens > 0, "VOSTI_MAX_TOKENS must be positive");
    let requests = if multi_calls.is_some() {
        Vec::new()
    } else {
        benchmark_requests
            .as_ref()
            .map(|_| Vec::new())
            .unwrap_or_else(|| {
                vec![fresh_request(
                    0,
                    prompt_tokens.clone(),
                    max_tokens,
                    eos_token_ids.clone(),
                    env_present("VOSTI_IGNORE_EOS"),
                )
                .unwrap_or_else(|error| panic!("invalid {architecture} request: {error}"))]
            })
    };
    let benchmark_batch = benchmark_requests.as_ref().map(Vec::len).unwrap_or(1);
    let (mut engine, graph_overlay) = initialize_qualified_engine(
        architecture,
        benchmark_batch,
        num_blocks,
        weights,
        runtime,
        weights_perms,
        model_config,
        requests,
    );

    if let Some(path) = multi_calls {
        #[cfg(feature = "openai-server")]
        crate::engine_benchmark::calls::run(
            &mut engine,
            graph_overlay.as_ref(),
            std::path::Path::new(&path),
            &eos_token_ids,
        )
        .unwrap_or_else(|error| panic!("native multi-call driver failed: {error}"));
        #[cfg(not(feature = "openai-server"))]
        panic!(
            "native multi-call input {:?} requires the openai-server feature",
            path
        );
        return;
    }

    if let Some(prepared) = benchmark_requests {
        let warmup_input = std::env::var("VOSTI_BENCH_WARMUP_INPUT")
            .expect("VOSTI_BENCH_WARMUP_INPUT must name a disjoint prepared workload");
        let warmup =
            crate::engine_benchmark::read_prepared_requests(std::path::Path::new(&warmup_input))
                .unwrap_or_else(|error| panic!("invalid VOSTI_BENCH_WARMUP_INPUT: {error}"));
        // Optional graph primer is a separate, observable workload in the same
        // Engine. It does not replace the donor or measured request histories.
        let request_id_base = if let Some(path) = std::env::var_os("VOSTI_BENCH_GRAPH_PRIMER_INPUT")
        {
            let primer =
                crate::engine_benchmark::read_prepared_requests(std::path::Path::new(&path))
                    .unwrap_or_else(|error| panic!("invalid graph primer: {error}"));
            set_python_environment("VOSTI_LOGITS_OBSERVER_PHASE", "primer");
            crate::engine_benchmark::run_prepared_requests(
                &mut engine,
                graph_overlay.as_ref(),
                &primer,
                0,
                &eos_token_ids,
            )
            .unwrap_or_else(|error| panic!("graph primer failed: {error}"));
            set_python_environment("VOSTI_LOGITS_OBSERVER_PHASE", "warmup");
            u64::try_from(primer.len()).expect("graph primer request count exceeds u64")
        } else {
            0
        };
        let result = crate::engine_benchmark::run_prepared_pair(
            &mut engine,
            graph_overlay.as_ref(),
            request_id_base,
            &warmup,
            &prepared,
            &eos_token_ids,
            || set_python_environment("VOSTI_LOGITS_OBSERVER_PHASE", "measured"),
        )
        .unwrap_or_else(|error| panic!("{architecture} benchmark failed: {error}"));
        let output_path = std::env::var("VOSTI_OUTPUT_TOKENS")
            .expect("VOSTI_OUTPUT_TOKENS must name the benchmark output record");
        crate::output_tokens::write_outputs(
            std::path::Path::new(&output_path),
            &result.measured.outputs,
        )
        .unwrap_or_else(|error| panic!("failed to record {architecture} outputs: {error}"));
        result.report(warmup.len(), prepared.len());
        return;
    }

    let max_steps = prompt_tokens
        .len()
        .checked_add(max_tokens)
        .and_then(|value| value.checked_add(2))
        .expect("model step limit overflow");
    let mut output = Vec::new();
    for _ in 0..max_steps {
        if engine.cs.live_requests.is_empty() {
            break;
        }
        let (emitted, _samples, _reprs) = engine.step(graph_overlay.as_ref());
        if let Some(token) = emitted.get(&0) {
            output.push(*token);
        }
    }
    assert!(
        engine.cs.live_requests.is_empty(),
        "{architecture} request did not finish within the conservative step limit",
    );
    if let Ok(path) = std::env::var("VOSTI_OUTPUT_TOKENS") {
        crate::output_tokens::write_outputs(std::path::Path::new(&path), &[output.clone()])
            .unwrap_or_else(|error| panic!("failed to write {architecture} outputs: {error}"));
    }
    println!("MODEL_OUTPUT_TOKEN_IDS={output:?}");
    if let Some(overlay) = graph_overlay.as_ref() {
        println!("GRAPH_STATS {}", RT::cuda_graph_overlay_stats_json(overlay));
    }
}
