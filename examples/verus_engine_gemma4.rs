// Text-only Gemma 4 inference through the architecture-dispatched Engine.

use vosti_verus::boundary::model_families::gemma4::deployment as GEMMA_DEPLOYMENT;

#[path = "support/engine_benchmark.rs"]
mod engine_benchmark;
#[path = "support/engine_serving.rs"]
mod engine_serving;
#[path = "support/engine_setup.rs"]
mod engine_setup;
#[path = "support/output_tokens.rs"]
mod output_tokens;

fn main() {
    let (model_path, deployment_bundle, device) =
        engine_serving::deployment_environment();
    let eos_token_ids = engine_serving::checkpoint_eos_token_ids(&model_path);
    let checkpoint = GEMMA_DEPLOYMENT::load_checkpoint(&model_path, &device);
    let (weights, runtime, weights_perms, model_config) =
        GEMMA_DEPLOYMENT::qualify_checkpoint(checkpoint, &deployment_bundle);
    engine_serving::run_qualified_model(
        "gemma4_text", "2,3", eos_token_ids,
        weights, runtime, weights_perms, model_config,
    );
}
