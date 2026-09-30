// Text-only Gemma 4 OpenAI-compatible serving through the dispatched Engine.

use vosti_verus::boundary::model_families::gemma4::deployment as GEMMA_DEPLOYMENT;

#[path = "support/engine_setup.rs"]
mod engine_setup;
#[path = "support/openai_server.rs"]
mod openai_server;

fn main() {
    let (model_path, deployment_bundle, device) =
        engine_setup::deployment_environment();
    let eos_token_ids = engine_setup::checkpoint_eos_token_ids(&model_path);
    let checkpoint = GEMMA_DEPLOYMENT::load_checkpoint(&model_path, &device);
    let (weights, runtime, weights_perms, model_config) =
        GEMMA_DEPLOYMENT::qualify_checkpoint(checkpoint, &deployment_bundle);
    // 32K aggregate retained KV is 27.5 GiB for the 31B text checkpoint.
    // The smaller-family default of 4096 pages would exceed one H200.
    let num_blocks = engine_setup::env_u64("VOSTI_NUM_BLOCKS", 512);
    let (engine, graph_overlay) = engine_setup::initialize_qualified_engine(
        "gemma4_text", 64, num_blocks,
        weights, runtime, weights_perms, model_config, Vec::new(),
    );
    openai_server::run_openai_server(
        "gemma4_text", &model_path, &deployment_bundle, eos_token_ids,
        engine, graph_overlay,
    )
    .unwrap_or_else(|error| panic!("Gemma 4 OpenAI server failed: {error}"));
}
