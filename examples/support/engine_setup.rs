// Architecture-neutral construction of one qualified, long-lived Engine.

use vosti_verus::model_config::ModelConfig;
use pyo3::types::PyAnyMethods;
use vstd::prelude::Tracked;
use vosti_verus::exec::cache_scheduler::SchedulerConfig;
use vosti_verus::types::BLOCK_SIZE;
use vosti_verus::exec::engine::Engine;
use vosti_verus::exec::request_state::RequestState;

use vosti_verus::boundary::tensor_runtime as RT;

pub fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .map(|value| {
            value
                .parse::<usize>()
                .unwrap_or_else(|_| panic!("{name} must be a nonnegative integer"))
        })
        .unwrap_or(default)
}

pub fn env_u64(name: &str, default: u64) -> u64 {
    std::env::var(name)
        .ok()
        .map(|value| {
            value
                .parse::<u64>()
                .unwrap_or_else(|_| panic!("{name} must be a nonnegative integer"))
        })
        .unwrap_or(default)
}

pub fn env_flag(name: &str, default: bool) -> bool {
    match std::env::var(name)
        .unwrap_or_else(|_| if default { "1" } else { "0" }.to_string())
        .as_str()
    {
        "0" => false,
        "1" => true,
        value => panic!("{name} must be exactly 0 or 1, got {value:?}"),
    }
}

fn token_capacity(num_blocks: u64) -> usize {
    assert!(
        num_blocks <= u64::MAX / BLOCK_SIZE,
        "VOSTI_NUM_BLOCKS exceeds the Engine block-capacity bound",
    );
    usize::try_from(num_blocks)
        .expect("VOSTI_NUM_BLOCKS exceeds usize")
        .checked_mul(BLOCK_SIZE as usize)
        .expect("model KV token capacity overflow")
}

fn scheduler_config(default_max_num_seqs: usize) -> SchedulerConfig {
    let max_num_seqs = env_usize("VOSTI_MAX_SEQS", default_max_num_seqs);
    let max_num_batched_tokens = env_usize("VOSTI_MAX_BATCHED_TOKENS", 4096);
    assert!(max_num_seqs > 0, "VOSTI_MAX_SEQS must be positive");
    assert!(
        max_num_batched_tokens > 0,
        "VOSTI_MAX_BATCHED_TOKENS must be positive",
    );
    SchedulerConfig {
        max_num_seqs,
        max_num_batched_tokens,
    }
}

fn qualified_runtime_marker(runtime: &RT::ModelRuntime, architecture: &str) {
    assert!(
        RT::model_runtime_reports_backend_qualified(runtime),
        "{architecture} runtime lost backend qualification before Engine admission",
    );
    println!("MODEL_RUNTIME architecture={architecture} backend_qualified=true");
}

fn graph_overlay() -> Option<RT::CudaGraphOverlay> {
    env_flag("VOSTI_CUDA_GRAPH", false).then(RT::init_cuda_graph_overlay)
}

pub fn initialize_qualified_engine(
    architecture: &str,
    default_max_num_seqs: usize,
    num_blocks: u64,
    weights: RT::ModelWeights,
    runtime: RT::ModelRuntime,
    weights_perms: Tracked<RT::ModelWeightsPerms>,
    model_config: ModelConfig,
    requests: Vec<RequestState>,
) -> (Engine, Option<RT::CudaGraphOverlay>) {
    qualified_runtime_marker(&runtime, architecture);
    let token_capacity = token_capacity(num_blocks);
    let (kv_caches, kv_perms) = RT::init_model_kv_caches(&weights, token_capacity);
    let engine = Engine::init(
        scheduler_config(default_max_num_seqs),
        num_blocks,
        kv_caches,
        kv_perms,
        weights,
        runtime,
        weights_perms,
        model_config,
        requests,
    );
    (engine, graph_overlay())
}

pub fn deployment_environment() -> (String, String, String) {
    let model_path = std::env::var("MODEL_PATH")
        .expect("MODEL_PATH must name a local model checkpoint directory");
    assert!(!model_path.trim().is_empty(), "MODEL_PATH must not be empty");
    let deployment_bundle = std::env::var("VOSTI_DEPLOYMENT_BUNDLE")
        .expect("VOSTI_DEPLOYMENT_BUNDLE must name a sealed deployment bundle");
    let device = std::env::var("CUDA_DEVICE").unwrap_or_else(|_| "cuda:0".to_string());
    (model_path, deployment_bundle, device)
}

// Resolve checkpoint generation policy outside the Engine, then pass it
// explicitly into every request.
pub fn checkpoint_eos_token_ids(model_path: &str) -> Vec<u64> {
    pyo3::Python::with_gil(|py| {
        let module = py
            .import_bound("vosti_kernels.serving_workload")
            .expect("failed to import tokenizer workload support");
        let ids = module
            .getattr("load_generation_eos_token_ids")
            .expect("tokenizer workload support lacks EOS metadata loader")
            .call1((model_path,))
            .expect("failed to resolve checkpoint EOS token ids")
            .extract::<Vec<u64>>()
            .expect("checkpoint EOS token ids are not a u64 list");
        assert!(!ids.is_empty(), "checkpoint EOS token ids must be nonempty");
        ids
    })
}
