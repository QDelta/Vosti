use vosti_verus::model_config::{AttentionKind, DenseGeometry, FloatParameterBits};
use vosti_verus::boundary::model_families::gemma3::config::{GEMMA3_GLOBAL_ROPE_FACTOR_F64_BITS, GEMMA3_GLOBAL_ROPE_THETA_F64_BITS, GEMMA3_LOCAL_ROPE_THETA_F64_BITS, GEMMA3_RMS_NORM_EPSILON_F64_BITS, Gemma3Config};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use vosti_verus::boundary::model_families::gemma3::deployment::{
    init_staged_runtime_for_tests, reports_backend_qualified,
};

use vosti_verus::boundary::tensor_runtime::{
    bind_model_weights_perms, init_model_kv_caches, model_runtime_reports_backend_qualified,
    Gemma3LayerWeights, Gemma3ModelWeights, ModelRuntime, ModelWeights, Tensor,
};

fn zeros(py: Python<'_>, shape: impl IntoPy<PyObject>) -> PyResult<Py<PyAny>> {
    let torch = py.import_bound("torch")?;
    let kwargs = PyDict::new_bound(py);
    kwargs.set_item("dtype", torch.getattr("bfloat16")?)?;
    Ok(torch
        .getattr("zeros")?
        .call((shape.into_py(py),), Some(&kwargs))?
        .unbind())
}

fn tensor(inner: Py<PyAny>) -> Tensor {
    Tensor { inner }
}

fn tiny_gemma_weights(tied_head: bool) -> ModelWeights {
    Python::with_gil(|py| -> PyResult<ModelWeights> {
        let embed_object = zeros(py, (7, 4))?;
        let embed_weight = tensor(embed_object.clone_ref(py));
        let lm_head = if tied_head {
            tensor(embed_object)
        } else {
            tensor(zeros(py, (7, 4))?)
        };
        let layer = Gemma3LayerWeights {
            input_norm: tensor(zeros(py, (4,))?),
            q_proj: tensor(zeros(py, (4, 4))?),
            k_proj: tensor(zeros(py, (2, 4))?),
            v_proj: tensor(zeros(py, (2, 4))?),
            q_norm: tensor(zeros(py, (2,))?),
            k_norm: tensor(zeros(py, (2,))?),
            o_proj: tensor(zeros(py, (4, 4))?),
            post_attn_norm: tensor(zeros(py, (4,))?),
            pre_feedforward_norm: tensor(zeros(py, (4,))?),
            gate_up_proj: tensor(zeros(py, (12, 4))?),
            down_proj: tensor(zeros(py, (4, 6))?),
            post_feedforward_norm: tensor(zeros(py, (4,))?),
            attention_kind: AttentionKind::SlidingWindow,
            layer_scale: None,
        };
        Ok(ModelWeights::Gemma3Text(Gemma3ModelWeights {
            embed_weight,
            layers: vec![layer],
            final_norm: tensor(zeros(py, (4,))?),
            lm_head,
            config: Gemma3Config {
                geometry: DenseGeometry {
                    vocab_size: 7,
                    hidden_size: 4,
                    intermediate_size: 6,
                    num_layers: 1,
                    num_attention_heads: 2,
                    num_key_value_heads: 1,
                    head_dim: 2,
                    max_position_embeddings: 32,
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
        }))
    })
    .expect("construct tiny Gemma 3 weights")
}

#[test]
fn gemma3_staged_runtime_capability_is_not_backend_qualified() {
    let runtime = init_staged_runtime_for_tests("gemma-3-4b-it-text");
    assert!(!reports_backend_qualified(&runtime));
    assert!(!model_runtime_reports_backend_qualified(
        &ModelRuntime::Gemma3Text(runtime)
    ));
}

#[test]
fn gemma3_permission_binder_accepts_exact_twelve_role_facade() {
    let weights = tiny_gemma_weights(true);
    let _perms = bind_model_weights_perms(&weights, 1);
}

#[test]
#[should_panic(expected = "Gemma 3 layers must not contain an output scale tensor")]
fn gemma3_permission_binder_rejects_an_output_scale_role() {
    let mut weights = tiny_gemma_weights(true);
    if let ModelWeights::Gemma3Text(gemma) = &mut weights {
        gemma.layers[0].layer_scale = Some(Python::with_gil(|py| {
            tensor(zeros(py, (1,)).expect("construct unexpected scalar role"))
        }));
    }
    let _perms = bind_model_weights_perms(&weights, 1);
}

#[test]
fn gemma3_model_cache_allocator_uses_weight_geometry() {
    let weights = tiny_gemma_weights(true);
    let (caches, _perms) = init_model_kv_caches(&weights, 65);
    assert_eq!(caches.len(), 1);
    Python::with_gil(|py| -> PyResult<()> {
        for tensor in [&caches[0].0, &caches[0].1] {
            let bound = tensor.inner.bind(py);
            let shape: Vec<usize> = bound.getattr("shape")?.extract()?;
            assert_eq!(shape, vec![2, 64, 1, 2]);
            assert_eq!(bound.getattr("dtype")?.str()?.to_str()?, "torch.bfloat16");
        }
        Ok(())
    })
    .expect("inspect Gemma 3 cache allocation");
}

#[test]
#[should_panic(expected = "python Gemma 3 model-weight permission contract failed")]
fn gemma3_permission_binder_rejects_untied_head() {
    let weights = tiny_gemma_weights(false);
    let _perms = bind_model_weights_perms(&weights, 1);
}

#[test]
#[should_panic(expected = "python Gemma 3 model-weight permission contract failed")]
fn gemma3_permission_binder_rejects_mismatched_hidden_size() {
    let mut weights = tiny_gemma_weights(true);
    if let ModelWeights::Gemma3Text(gemma) = &mut weights {
        gemma.config.geometry.hidden_size = 5;
    }
    let _perms = bind_model_weights_perms(&weights, 1);
}

#[test]
#[should_panic(expected = "python Gemma 3 model-weight permission contract failed")]
fn gemma3_permission_binder_rejects_nonpositive_window() {
    let mut weights = tiny_gemma_weights(true);
    if let ModelWeights::Gemma3Text(gemma) = &mut weights {
        gemma.config.sliding_window = 0;
    }
    let _perms = bind_model_weights_perms(&weights, 1);
}
