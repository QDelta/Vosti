//! Qwen3 zero-layer fixture for architecture-neutral Engine integration tests.
//!
//! The closed runtime/weight sums require a concrete variant, so this fixture
//! uses Qwen3 solely to construct the debug-only staged capability. With zero
//! layers, no family layer composition or kernel executes; the tests exercise
//! scheduler, admission, sample, and commit mechanics only.

use vosti_verus::model_config::{DenseGeometry, FloatParameterBits, ModelArchitecture, ModelConfig};
use vosti_verus::boundary::model_families::qwen3::config::{QWEN3_RMS_NORM_EPSILON_F64_BITS, QWEN3_ROPE_THETA_F64_BITS, Qwen3Config};
use vstd::prelude::*;
use vosti_verus::boundary::model_families::qwen3::deployment::init_staged_runtime_for_tests;
use vosti_verus::exec::cache_scheduler::SchedulerConfig;
use vosti_verus::exec::engine::Engine;
use vosti_verus::exec::request_state::RequestState;
use vosti_verus::{proof::tensor::types::{Tensor2D}};
use vosti_verus::boundary::tensor_runtime::{
    from_pylist_2d, init_kv_caches, ModelRuntime, ModelWeights, Qwen3LayerWeights,
    Qwen3ModelWeights, TensorId,
};

pub fn zero_layer_engine(
    requests: Vec<RequestState>,
    kv_capacity: usize,
    max_num_batched_tokens: usize,
) -> Engine {
    let final_norm_data = vec![1.0; 64];
    let lm_head_data = vec![0.0; 8 * 64];
    let empty_repr: Ghost<Tensor2D> = Ghost::assume_new();
    let empty_scope: Ghost<Set<TensorId>> = Ghost::assume_new();
    let (final_norm, _final_norm_perm) =
        from_pylist_2d(1, 64, final_norm_data.as_slice(), empty_repr, empty_scope);
    let (lm_head, _lm_head_perm) =
        from_pylist_2d(8, 64, lm_head_data.as_slice(), empty_repr, empty_scope);
    let (embed_weight, _embed_weight_perm) =
        from_pylist_2d(8, 64, lm_head_data.as_slice(), empty_repr, empty_scope);

    let weights = ModelWeights::Qwen3(Qwen3ModelWeights {
        embed_weight,
        layers: Vec::<Qwen3LayerWeights>::new(),
        final_norm,
        lm_head,
        config: Qwen3Config {
            geometry: DenseGeometry {
                vocab_size: 8,
                hidden_size: 64,
                intermediate_size: 64,
                num_layers: 0,
                num_attention_heads: 1,
                num_key_value_heads: 1,
                head_dim: 64,
                max_position_embeddings: 16,
            },
            rms_norm_epsilon: FloatParameterBits {
                bits: QWEN3_RMS_NORM_EPSILON_F64_BITS,
            },
            rope_theta: FloatParameterBits {
                bits: QWEN3_ROPE_THETA_F64_BITS,
            },
            tie_word_embeddings: false,
        },
    });
    // This debug-only zero-layer fixture is intentionally outside qualified
    // model admission. Do not ask the production permission binder to certify
    // an impossible profile; ordinary Rust tests need only the erased token.
    let weights_perms = Tracked::assume_new();
    let model_config = ModelConfig {
        architecture: ModelArchitecture::Qwen3,
        num_layers: 0,
    };
    let (kv_caches, kv_perms) = init_kv_caches(0, kv_capacity);
    let config = SchedulerConfig {
        max_num_seqs: 4,
        max_num_batched_tokens,
    };
    Engine::init(
        config,
        16,
        kv_caches,
        kv_perms,
        weights,
        ModelRuntime::Qwen3(init_staged_runtime_for_tests("qwen3-0.6b")),
        weights_perms,
        model_config,
        requests,
    )
}
