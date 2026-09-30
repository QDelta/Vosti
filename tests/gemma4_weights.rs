//! CPU-only weight binding checks; these do not admit an engine/runtime.

use vosti_verus::model_config::{AttentionKind, DenseGeometry, FloatParameterBits};
use vosti_verus::boundary::model_families::gemma4::config::Gemma4Config;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use vosti_verus::boundary::model_families::gemma4::weights::{
    bind_model_weights_perms, Gemma4LayerWeights, Gemma4ModelWeights,
};

use vosti_verus::boundary::tensor_runtime::{self as RT, ModelWeights, Tensor};

fn parameter(value: f64) -> FloatParameterBits {
    FloatParameterBits { bits: value.to_bits() }
}

fn zeros(py: Python<'_>, shape: impl IntoPy<PyObject>) -> Tensor {
    let torch = py.import_bound("torch").unwrap();
    let kwargs = PyDict::new_bound(py);
    kwargs.set_item("dtype", torch.getattr("bfloat16").unwrap()).unwrap();
    Tensor { inner: torch.getattr("zeros").unwrap()
        .call((shape.into_py(py),), Some(&kwargs)).unwrap().unbind() }
}

fn fixture() -> Gemma4ModelWeights {
    Python::with_gil(|py| {
        let embed_weight = zeros(py, (7, 4));
        let lm_head = Tensor { inner: embed_weight.inner.clone_ref(py) };
        let mut layers = Vec::new();
        for (attention_kind, head_dim, kv_heads) in [
            (AttentionKind::SlidingWindow, 2, 2), (AttentionKind::Full, 4, 1),
        ] {
            let k_proj = zeros(py, (kv_heads * head_dim, 4));
            let v_proj = match attention_kind {
                AttentionKind::Full => Tensor { inner: k_proj.inner.clone_ref(py) },
                AttentionKind::SlidingWindow => zeros(py, (kv_heads * head_dim, 4)),
            };
            layers.push(Gemma4LayerWeights {
                input_norm: zeros(py, (4,)), q_proj: zeros(py, (2 * head_dim, 4)),
                k_proj, v_proj, q_norm: zeros(py, (head_dim,)), k_norm: zeros(py, (head_dim,)),
                o_proj: zeros(py, (4, 2 * head_dim)), post_attn_norm: zeros(py, (4,)),
                pre_feedforward_norm: zeros(py, (4,)), gate_up_proj: zeros(py, (12, 4)),
                down_proj: zeros(py, (4, 6)), post_feedforward_norm: zeros(py, (4,)),
                attention_kind, layer_scale: Some(zeros(py, (1,))),
            });
        }
        Gemma4ModelWeights {
            embed_weight, lm_head, final_norm: zeros(py, (4,)), layers,
            config: Gemma4Config {
                geometry: DenseGeometry { vocab_size: 7, hidden_size: 4, intermediate_size: 6,
                    num_layers: 2, num_attention_heads: 2, num_key_value_heads: 2, head_dim: 2,
                    max_position_embeddings: 32 },
                num_global_key_value_heads: 1, global_head_dim: 4, rms_norm_epsilon: parameter(1e-6),
                sliding_window: 16, local_rope_theta: parameter(10000.0),
                global_rope_theta: parameter(1000000.0), global_rope_factor: parameter(1.0),
                global_partial_rotary_factor: parameter(0.5), attention_k_eq_v: true,
                final_logit_softcap: Some(parameter(30.0)),
            },
        }
    })
}

#[test]
fn binds_heterogeneous_geometry_and_raw_global_kv_alias() {
    let weights = fixture();
    let _perms = bind_model_weights_perms(&weights, 2);
}

#[test]
fn common_weight_facade_binds_and_preserves_device_anchor() {
    let weights = ModelWeights::Gemma4Text(fixture());
    let _perms = RT::bind_model_weights_perms(&weights, 2);
    let ModelWeights::Gemma4Text(gemma) = &weights else { unreachable!() };
    assert!(std::ptr::eq(RT::model_weights_device_anchor(&weights), &gemma.embed_weight));
    assert!(RT::model_step_plan_device_anchor(&weights).is_none());
}

#[test]
fn common_allocator_preserves_each_layers_kv_geometry() {
    let weights = ModelWeights::Gemma4Text(fixture());
    let (caches, _perms) = RT::init_model_kv_caches(&weights, 16);
    assert_eq!(caches.len(), 2);
    Python::with_gil(|py| {
        let page_size: usize = py.import_bound("vosti_kernels.physical").unwrap()
            .getattr("PAGE_SIZE").unwrap().extract().unwrap();
        for ((k, v), (heads, width)) in caches.iter().zip([(2, 2), (1, 4)]) {
            let expected = (16usize.div_ceil(page_size), page_size, heads, width);
            assert_eq!(k.inner.bind(py).getattr("shape").unwrap()
                .extract::<(usize, usize, usize, usize)>().unwrap(), expected);
            assert_eq!(v.inner.bind(py).getattr("shape").unwrap()
                .extract::<(usize, usize, usize, usize)>().unwrap(), expected);
            assert!(!k.inner.bind(py).is(v.inner.bind(py)));
        }
    });
}

#[test]
#[should_panic(expected = "must share its raw K/V projection")]
fn common_weight_facade_does_not_bypass_family_validation() {
    let mut weights = fixture();
    weights.layers[1].v_proj = Python::with_gil(|py| zeros(py, (4, 4)));
    let _perms = RT::bind_model_weights_perms(&ModelWeights::Gemma4Text(weights), 2);
}

#[test]
fn accepts_absent_optional_softcap() {
    let mut weights = fixture();
    weights.config.final_logit_softcap = None;
    let _perms = bind_model_weights_perms(&weights, 2);
}

#[test]
fn accepts_independent_global_kv_when_configured() {
    let mut weights = fixture();
    weights.config.attention_k_eq_v = false;
    weights.layers[1].v_proj = Python::with_gil(|py| zeros(py, (4, 4)));
    let _perms = bind_model_weights_perms(&weights, 2);
}

#[test]
#[should_panic(expected = "output-scalar role differs")]
fn rejects_missing_scalar_role() {
    let mut weights = fixture();
    weights.layers[0].layer_scale = None;
    let _perms = bind_model_weights_perms(&weights, 2);
}

#[test]
#[should_panic(expected = "layer_scalar has shape")]
fn rejects_nonscalar_role() {
    let mut weights = fixture();
    weights.layers[0].layer_scale = Some(Python::with_gil(|py| zeros(py, (2,))));
    let _perms = bind_model_weights_perms(&weights, 2);
}

#[test]
#[should_panic(expected = "must share its raw K/V projection")]
fn rejects_unshared_global_projection_when_required() {
    let mut weights = fixture();
    weights.layers[1].v_proj = Python::with_gil(|py| zeros(py, (4, 4)));
    let _perms = bind_model_weights_perms(&weights, 2);
}

#[test]
#[should_panic(expected = "configured geometry q_proj has shape")]
fn rejects_global_width_mismatch() {
    let mut weights = fixture();
    weights.config.global_head_dim = 8;
    let _perms = bind_model_weights_perms(&weights, 2);
}

#[test]
#[should_panic(expected = "tied embedding")]
fn rejects_untied_head() {
    let mut weights = fixture();
    weights.lm_head = Python::with_gil(|py| zeros(py, (7, 4)));
    let _perms = bind_model_weights_perms(&weights, 2);
}

#[test]
#[should_panic(expected = "rms_norm_eps")]
fn rejects_nonpositive_norm_epsilon() {
    let mut weights = fixture();
    weights.config.rms_norm_epsilon = parameter(0.0);
    let _perms = bind_model_weights_perms(&weights, 2);
}

#[test]
#[should_panic(expected = "rotary fraction in (0, 1]")]
fn rejects_zero_rotary_fraction_outside_rust_parameter_domain() {
    let mut weights = fixture();
    weights.config.global_partial_rotary_factor = parameter(0.0);
    let _perms = bind_model_weights_perms(&weights, 2);
}
