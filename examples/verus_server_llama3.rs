// Llama 3 OpenAI-compatible serving through the architecture-dispatched Engine.

use vosti_verus::boundary::model_families::llama3::deployment as LLAMA_DEPLOYMENT;

#[path = "support/engine_setup.rs"]
mod engine_setup;
#[path = "support/openai_server.rs"]
mod openai_server;

fn main() {
    let (model_path, deployment_bundle, device) =
        engine_setup::deployment_environment();
    let eos_token_ids = engine_setup::checkpoint_eos_token_ids(&model_path);
    let checkpoint = LLAMA_DEPLOYMENT::load_checkpoint(&model_path, &device);
    let (weights, runtime, weights_perms, model_config) =
        LLAMA_DEPLOYMENT::qualify_checkpoint(checkpoint, &deployment_bundle);
    let num_blocks = engine_setup::env_u64("VOSTI_NUM_BLOCKS", 4096);
    let (engine, graph_overlay) = engine_setup::initialize_qualified_engine(
        "llama3",
        64,
        num_blocks,
        weights,
        runtime,
        weights_perms,
        model_config,
        Vec::new(),
    );
    openai_server::run_openai_server(
        "llama3",
        &model_path,
        &deployment_bundle,
        eos_token_ids,
        engine,
        graph_overlay,
    )
    .unwrap_or_else(|error| panic!("Llama 3 OpenAI server failed: {error}"));
}
