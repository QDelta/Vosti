use vosti_verus::model_config::{DenseGeometry, FloatParameterBits};
use vosti_verus::boundary::model_families::gemma3::config::{GEMMA3_GLOBAL_ROPE_FACTOR_F64_BITS, GEMMA3_GLOBAL_ROPE_THETA_F64_BITS, GEMMA3_LOCAL_ROPE_THETA_F64_BITS, GEMMA3_RMS_NORM_EPSILON_F64_BITS, Gemma3Config};
use vosti_verus::boundary::model_families::llama3::config::{LLAMA3_RMS_NORM_EPSILON_F64_BITS, LLAMA3_ROPE_FACTOR_8_F64_BITS, LLAMA3_ROPE_HIGH_FREQUENCY_FACTOR_F64_BITS, LLAMA3_ROPE_LOW_FREQUENCY_FACTOR_F64_BITS, LLAMA3_ROPE_ORIGINAL_MAX_POSITION_EMBEDDINGS, LLAMA3_ROPE_THETA_F64_BITS, Llama3Config};
use vosti_verus::boundary::model_families::qwen3::config::{QWEN3_RMS_NORM_EPSILON_F64_BITS, QWEN3_ROPE_THETA_F64_BITS, Qwen3Config};
use pyo3::prelude::*;

use vosti_verus::boundary::tensor_runtime::{
    model_step_plan_device_anchor, model_weights_device_anchor, Gemma3ModelWeights,
    Llama3ModelWeights, ModelWeights, Qwen3ModelWeights, Tensor,
};

fn tensor(py: Python<'_>) -> Tensor {
    let inner = py
        .import_bound("torch")
        .and_then(|torch| torch.getattr("zeros"))
        .and_then(|zeros| zeros.call1(((1,),)))
        .map(|value| value.unbind())
        .expect("construct CPU tensor");
    Tensor { inner }
}

#[test]
fn every_model_family_anchors_step_metadata_to_its_embedding_device() {
    Python::with_gil(|py| {
        let qwen = ModelWeights::Qwen3(Qwen3ModelWeights {
            embed_weight: tensor(py),
            layers: Vec::new(),
            final_norm: tensor(py),
            lm_head: tensor(py),
            config: Qwen3Config {
                geometry: DenseGeometry {
                    vocab_size: 1,
                    hidden_size: 1,
                    intermediate_size: 1,
                    num_layers: 0,
                    num_attention_heads: 1,
                    num_key_value_heads: 1,
                    head_dim: 2,
                    max_position_embeddings: 1,
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
        let qwen_anchor = model_weights_device_anchor(&qwen);
        match &qwen {
            ModelWeights::Qwen3(weights) => {
                assert!(std::ptr::eq(qwen_anchor, &weights.embed_weight));
            }
            _ => unreachable!(),
        }
        assert!(model_step_plan_device_anchor(&qwen).is_none());

        let llama = ModelWeights::Llama3(Llama3ModelWeights {
            embed_weight: tensor(py),
            layers: Vec::new(),
            final_norm: tensor(py),
            lm_head: tensor(py),
            config: Llama3Config {
                geometry: DenseGeometry {
                    vocab_size: 1,
                    hidden_size: 1,
                    intermediate_size: 1,
                    num_layers: 0,
                    num_attention_heads: 1,
                    num_key_value_heads: 1,
                    head_dim: 2,
                    max_position_embeddings: 1,
                },
                rms_norm_epsilon: FloatParameterBits {
                    bits: LLAMA3_RMS_NORM_EPSILON_F64_BITS,
                },
                rope_theta: FloatParameterBits {
                    bits: LLAMA3_ROPE_THETA_F64_BITS,
                },
                rope_factor: FloatParameterBits {
                    bits: LLAMA3_ROPE_FACTOR_8_F64_BITS,
                },
                rope_low_frequency_factor: FloatParameterBits {
                    bits: LLAMA3_ROPE_LOW_FREQUENCY_FACTOR_F64_BITS,
                },
                rope_high_frequency_factor: FloatParameterBits {
                    bits: LLAMA3_ROPE_HIGH_FREQUENCY_FACTOR_F64_BITS,
                },
                rope_original_max_position_embeddings: LLAMA3_ROPE_ORIGINAL_MAX_POSITION_EMBEDDINGS,
                tie_word_embeddings: false,
            },
        });
        let llama_anchor = model_weights_device_anchor(&llama);
        match &llama {
            ModelWeights::Llama3(weights) => {
                assert!(std::ptr::eq(llama_anchor, &weights.embed_weight));
            }
            _ => unreachable!(),
        }
        assert!(model_step_plan_device_anchor(&llama).is_none());

        let gemma = ModelWeights::Gemma3Text(Gemma3ModelWeights {
            embed_weight: tensor(py),
            layers: Vec::new(),
            final_norm: tensor(py),
            lm_head: tensor(py),
            config: Gemma3Config {
                geometry: DenseGeometry {
                    vocab_size: 1,
                    hidden_size: 1,
                    intermediate_size: 1,
                    num_layers: 0,
                    num_attention_heads: 1,
                    num_key_value_heads: 1,
                    head_dim: 2,
                    max_position_embeddings: 1,
                },
                rms_norm_epsilon: FloatParameterBits {
                    bits: GEMMA3_RMS_NORM_EPSILON_F64_BITS,
                },
                query_pre_attention_scalar: FloatParameterBits {
                    bits: 2.0f64.to_bits(),
                },
                sliding_window: 16,
                local_rope_theta: FloatParameterBits {
                    bits: GEMMA3_LOCAL_ROPE_THETA_F64_BITS,
                },
                global_rope_theta: FloatParameterBits {
                    bits: GEMMA3_GLOBAL_ROPE_THETA_F64_BITS,
                },
                global_rope_factor: FloatParameterBits {
                    bits: GEMMA3_GLOBAL_ROPE_FACTOR_F64_BITS,
                },
            },
        });
        let gemma_anchor = model_weights_device_anchor(&gemma);
        match &gemma {
            ModelWeights::Gemma3Text(weights) => {
                assert!(std::ptr::eq(gemma_anchor, &weights.embed_weight));
            }
            _ => unreachable!(),
        }
        assert!(model_step_plan_device_anchor(&gemma).is_none());
    });
}
