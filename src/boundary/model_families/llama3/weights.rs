//! Llama 3 physical weight roles and their semantic projection.

use crate::model_config::ModelArchitecture;
use crate::boundary::model_families::llama3::config::Llama3Config;
use crate::boundary::dense_swiglu_decoder as DENSE;
use crate::{types::*, proof::model::types::*, proof::tensor::types::*};
use crate::boundary::tensor_runtime as RT;
use crate::proof::tensor::shape as TS;
use vstd::prelude::*;

pub use crate::boundary::dense_swiglu_decoder::{
    DenseSwiGluLayerWeights as Llama3LayerWeights,
    DenseSwiGluLayerWeightsPerms as Llama3LayerWeightsPerms,
};

#[cfg(not(verus_only))]
use pyo3::prelude::*;

verus! {

// @kernel-bridge-begin boundary::model_families::llama3::model_weights_contract

pub open spec fn layer_weights_valid(
    weights: &Llama3LayerWeights,
    perms: &Llama3LayerWeightsPerms,
) -> bool {
    DENSE::layer_weights_valid(weights, perms)
    && match (&weights.qk_norm, &perms.qk_norm) {
        (DENSE::DenseQkNormWeights::Disabled,
         DENSE::DenseQkNormWeightsPerms::Disabled) => true,
        _ => false,
    }
}

pub open spec fn layer_weights_common_repr_of(
    model_perms: &RT::ModelWeightsPerms,
    perms: &Llama3LayerWeightsPerms,
) -> LayerWeightsRepr {
    DENSE::layer_weights_repr_of(perms, model_perms.llama3_config().geometry.head_dim)
}

pub proof fn lemma_layer_qk_norm_is_disabled(
    weights: &Llama3LayerWeights,
    perms: &Llama3LayerWeightsPerms,
)
    requires layer_weights_valid(weights, perms),
    ensures
        qk_norm_weights_kind(DENSE::qk_norm_weights_repr_of(&perms.qk_norm))
            == QkNormKind::Disabled,
{
    reveal(layer_weights_valid);
    reveal(DENSE::qk_norm_weights_repr_of);
    reveal(qk_norm_weights_kind);
    match (&weights.qk_norm, &perms.qk_norm) {
        (DENSE::DenseQkNormWeights::Disabled,
         DENSE::DenseQkNormWeightsPerms::Disabled) => {},
        _ => { assert(false); },
    }
}

pub open spec fn common_layers_repr_of(
    perms: &RT::ModelWeightsPerms,
) -> Seq<LayerWeightsRepr> {
    Seq::new(perms.num_layers(), |i: int|
        layer_weights_common_repr_of(perms, &perms.llama3_layer(i)))
}

pub struct Llama3ModelWeights {
    pub embed_weight: RT::Tensor,
    pub layers: Vec<Llama3LayerWeights>,
    pub final_norm: RT::Tensor,
    pub lm_head: RT::Tensor,
    pub config: Llama3Config,
}

pub open spec fn model_weights_bound(
    weights: &Llama3ModelWeights,
    perms: &RT::ModelWeightsPerms,
) -> bool {
    let config = llama3_config_repr(weights.config);
    let geometry = config.geometry;
    let q_width = geometry.num_attention_heads * geometry.head_dim;
    let kv_width = geometry.num_key_value_heads * geometry.head_dim;
    perms.architecture() == ModelArchitecture::Llama3
    && perms.llama3_config() == config
    && llama3_config_valid(config)
    && weights.embed_weight.id() == perms.embed_weight_id()
    && weights.layers.len() == perms.num_layers()
    && weights.layers.len() == geometry.num_layers
    && weights.final_norm.id() == perms.final_norm_id()
    && weights.lm_head.id() == perms.lm_head_id()
    && TS::tensor2d_shape(
        perms.embed_weight_repr(), geometry.vocab_size, geometry.hidden_size,
    )
    && perms.final_norm_repr().len() == geometry.hidden_size
    && TS::tensor2d_shape(
        perms.lm_head_repr(), geometry.vocab_size, geometry.hidden_size,
    )
    && forall|i: int| 0 <= i < weights.layers.len() as int ==>
        #[trigger] layer_weights_valid(&weights.layers[i], &perms.llama3_layer(i))
        && perms.llama3_layer(i).input_norm.repr_1d().len()
            == geometry.hidden_size
        && TS::tensor2d_shape(
            perms.llama3_layer(i).q_proj.repr_2d(),
            q_width, geometry.hidden_size,
        )
        && TS::tensor2d_shape(
            perms.llama3_layer(i).k_proj.repr_2d(),
            kv_width, geometry.hidden_size,
        )
        && TS::tensor2d_shape(
            perms.llama3_layer(i).v_proj.repr_2d(),
            kv_width, geometry.hidden_size,
        )
        && TS::tensor2d_shape(
            perms.llama3_layer(i).o_proj.repr_2d(),
            geometry.hidden_size, q_width,
        )
        && perms.llama3_layer(i).post_attn_norm.repr_1d().len()
            == geometry.hidden_size
        && TS::tensor2d_shape(
            perms.llama3_layer(i).gate_up_proj.repr_2d(),
            2 * geometry.intermediate_size, geometry.hidden_size,
        )
        && TS::tensor2d_shape(
            perms.llama3_layer(i).down_proj.repr_2d(),
            geometry.hidden_size, geometry.intermediate_size,
        )
}

pub proof fn lemma_layer_attention_geometry_matches_config(
    weights: &Llama3ModelWeights,
    perms: &RT::ModelWeightsPerms,
    layer: int,
)
    requires
        model_weights_bound(weights, perms),
        0 <= layer < weights.layers.len() as int,
    ensures
        crate::boundary::dense_layer_primitives::layer_attention_geometry_repr(
            layer_weights_common_repr_of(perms, &perms.llama3_layer(layer)),
        ) == attention_geometry_repr(llama3_config_repr(weights.config).geometry),
{
    reveal(model_weights_bound);
    let config = llama3_config_repr(weights.config);
    lemma_llama3_config_valid_implies_dense_config_valid(config);
    lemma_dense_swiglu_decoder_config_valid_implies_geometry_valid(config);
    assert(layer_weights_valid(
        &weights.layers[layer], &perms.llama3_layer(layer),
    ));
    assert(config.geometry.head_dim > 0);
    reveal(layer_weights_common_repr_of);
    crate::boundary::dense_layer_primitives::lemma_layer_attention_geometry_from_shapes(
        layer_weights_common_repr_of(perms, &perms.llama3_layer(layer)),
        config.geometry,
    );
    reveal(crate::boundary::dense_layer_primitives::layer_attention_geometry_repr);
}

pub proof fn lemma_physical_deployment_config_valid(
    weights: &Llama3ModelWeights,
    perms: &RT::ModelWeightsPerms,
)
    requires model_weights_bound(weights, perms),
    ensures
        model_deployment_config_valid(
            RT::llama3_physical_deployment_config_repr(weights),
        ),
{
    reveal(model_weights_bound);
    reveal(RT::llama3_physical_deployment_config_repr);
    reveal(model_deployment_config_valid);
}

#[verifier::external_body]
pub fn bind_model_weights_perms(
    weights: &Llama3ModelWeights,
    expected_num_layers: usize,
) -> (out: Tracked<RT::ModelWeightsPerms>)
    requires weights.layers.len() == expected_num_layers,
    ensures model_weights_bound(weights, &out@),
{
    #[cfg(not(verus_only))]
    {
        pyo3::Python::with_gil(|py| -> pyo3::PyResult<()> {
            let layer_tuples: Vec<pyo3::Bound<pyo3::types::PyTuple>> =
                weights.layers.iter().map(|layer| {
                    assert!(
                        matches!(&layer.qk_norm, DENSE::DenseQkNormWeights::Disabled),
                        "Llama 3 layer cannot enable Q/K normalization",
                    );
                    pyo3::types::PyTuple::new_bound(py, [
                        layer.input_norm.inner.bind(py),
                        layer.q_proj.inner.bind(py),
                        layer.k_proj.inner.bind(py),
                        layer.v_proj.inner.bind(py),
                        layer.o_proj.inner.bind(py),
                        layer.post_attn_norm.inner.bind(py),
                        layer.gate_up_proj.inner.bind(py),
                        layer.down_proj.inner.bind(py),
                    ])
                }).collect();
            let layers = pyo3::types::PyList::new_bound(py, &layer_tuples);
            let config = pyo3::types::PyDict::new_bound(py);
            let geometry = weights.config.geometry;
            config.set_item("vocab_size", geometry.vocab_size)?;
            config.set_item("hidden_size", geometry.hidden_size)?;
            config.set_item("intermediate_size", geometry.intermediate_size)?;
            config.set_item("num_hidden_layers", geometry.num_layers)?;
            config.set_item("num_attention_heads", geometry.num_attention_heads)?;
            config.set_item("num_key_value_heads", geometry.num_key_value_heads)?;
            config.set_item("head_dim", geometry.head_dim)?;
            config.set_item("max_position_embeddings", geometry.max_position_embeddings)?;
            config.set_item("rms_norm_eps", f64::from_bits(weights.config.rms_norm_epsilon.bits))?;
            config.set_item("rope_theta", f64::from_bits(weights.config.rope_theta.bits))?;
            config.set_item("rope_factor", f64::from_bits(weights.config.rope_factor.bits))?;
            config.set_item(
                "rope_low_frequency_factor",
                f64::from_bits(weights.config.rope_low_frequency_factor.bits),
            )?;
            config.set_item(
                "rope_high_frequency_factor",
                f64::from_bits(weights.config.rope_high_frequency_factor.bits),
            )?;
            config.set_item(
                "rope_original_max_position_embeddings",
                weights.config.rope_original_max_position_embeddings,
            )?;
            config.set_item("tie_word_embeddings", weights.config.tie_word_embeddings)?;
            config.set_item("model_type", "llama")?;
            config.set_item("transformers_architecture", "LlamaForCausalLM")?;
            config.set_item("hidden_act", "silu")?;
            config.set_item("attention_bias", false)?;
            config.set_item("attention_dropout", 0.0f64)?;
            config.set_item("mlp_bias", false)?;
            config.set_item("pretraining_tp", 1usize)?;
            config.set_item("attention_kind", "full_attention")?;
            config.set_item("rope_scaling_kind", "llama3")?;
            config.set_item("dtype", weights.embed_weight.inner.bind(py).getattr("dtype")?)?;
            config.set_item("device", weights.embed_weight.inner.bind(py).getattr("device")?)?;
            let implementation = py.import_bound(
                "vosti_kernels.model_families.llama3.physical",
            )?;
            implementation
                .getattr("validate_model_weights_runtime_contract")?
                .call1((
                    weights.embed_weight.inner.bind(py),
                    layers,
                    weights.final_norm.inner.bind(py),
                    weights.lm_head.inner.bind(py),
                    config,
                    expected_num_layers,
                ))?;
            Ok(())
        }).expect("python Llama 3 model-weight permission contract failed");
        Tracked::assume_new()
    }
    #[cfg(verus_only)]
    {
        unreachable!()
    }
}

// @kernel-bridge-end boundary::model_families::llama3::model_weights_contract

} // verus!
