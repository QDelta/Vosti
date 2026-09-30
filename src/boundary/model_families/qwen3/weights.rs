//! Qwen3-owned physical weight roles and their semantic projection.

use crate::model_config::ModelArchitecture;
use crate::boundary::model_families::qwen3::config::Qwen3Config;
use crate::{types::*, proof::model::types::*, proof::tensor::types::*};
use crate::boundary::tensor_runtime as RT;
use crate::proof::tensor::shape as TS;
use vstd::prelude::*;

use crate::boundary::dense_swiglu_decoder as DENSE;
pub use crate::boundary::dense_swiglu_decoder::{
    DenseSwiGluLayerWeights as Qwen3LayerWeights,
    DenseSwiGluLayerWeightsPerms as Qwen3LayerWeightsPerms,
};

#[cfg(not(verus_only))]
use pyo3::prelude::*;

verus! {

// @kernel-bridge-begin boundary::model_families::qwen3::model_weights_contract

pub open spec fn layer_weights_valid(
    weights: &Qwen3LayerWeights,
    perms: &Qwen3LayerWeightsPerms,
) -> bool {
    DENSE::layer_weights_valid(weights, perms)
    && match (&weights.qk_norm, &perms.qk_norm) {
        (
            DENSE::DenseQkNormWeights::RmsNorm { .. },
            DENSE::DenseQkNormWeightsPerms::RmsNorm { .. },
        ) => true,
        _ => false,
    }
}

pub open spec fn layer_weights_common_repr_of(
    perms: &Qwen3LayerWeightsPerms,
) -> LayerWeightsRepr {
    let qk_norm = DENSE::qk_norm_weights_repr_of(&perms.qk_norm);
    DENSE::layer_weights_repr_of(perms, rms_q_norm_weight(qk_norm).len())
}

pub proof fn lemma_layer_qk_norm_is_rms(
    weights: &Qwen3LayerWeights,
    perms: &Qwen3LayerWeightsPerms,
)
    requires layer_weights_valid(weights, perms),
    ensures
        qk_norm_weights_kind(layer_weights_common_repr_of(perms).qk_norm)
            == QkNormKind::RmsNorm,
{
    reveal(layer_weights_valid);
    reveal(layer_weights_common_repr_of);
    reveal(DENSE::layer_weights_repr_of);
    reveal(DENSE::qk_norm_weights_repr_of);
    reveal(qk_norm_weights_kind);
    match (&weights.qk_norm, &perms.qk_norm) {
        (
            DENSE::DenseQkNormWeights::RmsNorm { .. },
            DENSE::DenseQkNormWeightsPerms::RmsNorm { .. },
        ) => {},
        _ => { assert(false); },
    }
}

pub open spec fn common_layers_repr_of(
    perms: &RT::ModelWeightsPerms,
) -> Seq<LayerWeightsRepr> {
    Seq::new(perms.num_layers(), |i: int|
        layer_weights_common_repr_of(&perms.qwen3_layer(i)))
}

pub struct Qwen3ModelWeights {
    pub embed_weight: RT::Tensor,
    pub layers: Vec<Qwen3LayerWeights>,
    pub final_norm: RT::Tensor,
    pub lm_head: RT::Tensor,
    pub config: Qwen3Config,
}

pub open spec fn model_weights_bound(
    weights: &Qwen3ModelWeights,
    perms: &RT::ModelWeightsPerms,
) -> bool {
    let config = qwen3_config_repr(weights.config);
    let geometry = config.geometry;
    let q_width = geometry.num_attention_heads * geometry.head_dim;
    let kv_width = geometry.num_key_value_heads * geometry.head_dim;
    perms.architecture() == ModelArchitecture::Qwen3
    && perms.qwen3_config() == config
    && qwen3_config_valid(config)
    && weights.embed_weight.id() == perms.embed_weight_id()
    && weights.layers.len() == perms.num_layers()
    && weights.layers.len() == geometry.num_layers
    && weights.final_norm.id() == perms.final_norm_id()
    && weights.lm_head.id() == perms.lm_head_id()
    && (!config.composition.tie_word_embeddings
        || weights.embed_weight.id() == weights.lm_head.id())
    && TS::tensor2d_shape(
        perms.embed_weight_repr(), geometry.vocab_size, geometry.hidden_size,
    )
    && perms.final_norm_repr().len() == geometry.hidden_size
    && TS::tensor2d_shape(
        perms.lm_head_repr(), geometry.vocab_size, geometry.hidden_size,
    )
    && (forall|i: int| 0 <= i < weights.layers.len() as int ==>
        #[trigger] layer_weights_valid(&weights.layers[i], &perms.qwen3_layer(i))
        && perms.qwen3_layer(i).input_norm.repr_1d().len()
            == geometry.hidden_size
        && TS::tensor2d_shape(
            perms.qwen3_layer(i).q_proj.repr_2d(),
            q_width, geometry.hidden_size,
        )
        && TS::tensor2d_shape(
            perms.qwen3_layer(i).k_proj.repr_2d(),
            kv_width, geometry.hidden_size,
        )
        && TS::tensor2d_shape(
            perms.qwen3_layer(i).v_proj.repr_2d(),
            kv_width, geometry.hidden_size,
        )
        && rms_q_norm_weight(
            layer_weights_common_repr_of(&perms.qwen3_layer(i)).qk_norm,
        ).len()
            == geometry.head_dim
        && rms_k_norm_weight(
            layer_weights_common_repr_of(&perms.qwen3_layer(i)).qk_norm,
        ).len()
            == geometry.head_dim
        && TS::tensor2d_shape(
            perms.qwen3_layer(i).o_proj.repr_2d(),
            geometry.hidden_size, q_width,
        )
        && perms.qwen3_layer(i).post_attn_norm.repr_1d().len()
            == geometry.hidden_size
        && TS::tensor2d_shape(
            perms.qwen3_layer(i).gate_up_proj.repr_2d(),
            2 * geometry.intermediate_size, geometry.hidden_size,
        )
        && TS::tensor2d_shape(
            perms.qwen3_layer(i).down_proj.repr_2d(),
            geometry.hidden_size, geometry.intermediate_size,
        ))
}

pub proof fn lemma_layer_attention_geometry_matches_config(
    weights: &Qwen3ModelWeights,
    perms: &RT::ModelWeightsPerms,
    layer: int,
)
    requires
        model_weights_bound(weights, perms),
        0 <= layer < weights.layers.len() as int,
    ensures
        crate::boundary::dense_layer_primitives::layer_attention_geometry_repr(
            layer_weights_common_repr_of(&perms.qwen3_layer(layer)),
        ) == attention_geometry_repr(qwen3_config_repr(weights.config).geometry),
{
    reveal(model_weights_bound);
    let config = qwen3_config_repr(weights.config);
    lemma_qwen3_config_valid_implies_dense_config_valid(config);
    lemma_dense_swiglu_decoder_config_valid_implies_geometry_valid(config);
    assert(layer_weights_valid(
        &weights.layers[layer], &perms.qwen3_layer(layer),
    ));
    assert(config.geometry.head_dim > 0);
    assert(TS::tensor2d_shape(
        perms.qwen3_layer(layer).q_proj.repr_2d(),
        config.geometry.num_attention_heads * config.geometry.head_dim,
        config.geometry.hidden_size,
    ));
    assert(TS::tensor2d_shape(
        perms.qwen3_layer(layer).k_proj.repr_2d(),
        config.geometry.num_key_value_heads * config.geometry.head_dim,
        config.geometry.hidden_size,
    ));
    reveal(layer_weights_common_repr_of);
    crate::boundary::dense_layer_primitives::lemma_layer_attention_geometry_from_shapes(
        layer_weights_common_repr_of(&perms.qwen3_layer(layer)),
        config.geometry,
    );
    reveal(crate::boundary::dense_layer_primitives::layer_attention_geometry_repr);
}

pub proof fn lemma_physical_deployment_config_valid(
    weights: &Qwen3ModelWeights,
    perms: &RT::ModelWeightsPerms,
)
    requires model_weights_bound(weights, perms),
    ensures
        model_deployment_config_valid(
            RT::qwen3_physical_deployment_config_repr(weights),
        ),
{
    reveal(model_weights_bound);
    reveal(RT::qwen3_physical_deployment_config_repr);
    reveal(model_deployment_config_valid);
}

#[verifier::external_body]
pub fn bind_model_weights_perms(
    weights: &Qwen3ModelWeights,
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
                    let (q_norm, k_norm) = match &layer.qk_norm {
                        DENSE::DenseQkNormWeights::RmsNorm {
                            q_weight,
                            k_weight,
                        } => (q_weight, k_weight),
                        DENSE::DenseQkNormWeights::Disabled => {
                            panic!("Qwen3 layer cannot disable Q/K normalization")
                        },
                    };
                    pyo3::types::PyTuple::new_bound(py, [
                        layer.input_norm.inner.bind(py),
                        layer.q_proj.inner.bind(py),
                        layer.k_proj.inner.bind(py),
                        layer.v_proj.inner.bind(py),
                        q_norm.inner.bind(py),
                        k_norm.inner.bind(py),
                        layer.o_proj.inner.bind(py),
                        layer.post_attn_norm.inner.bind(py),
                        layer.gate_up_proj.inner.bind(py),
                        layer.down_proj.inner.bind(py),
                    ])
                }).collect();
            let layers = pyo3::types::PyList::new_bound(py, &layer_tuples);
            let config = pyo3::types::PyDict::new_bound(py);
            config.set_item("vocab_size", weights.config.geometry.vocab_size)?;
            config.set_item("hidden_size", weights.config.geometry.hidden_size)?;
            config.set_item(
                "intermediate_size", weights.config.geometry.intermediate_size,
            )?;
            config.set_item("num_hidden_layers", weights.config.geometry.num_layers)?;
            config.set_item(
                "num_attention_heads", weights.config.geometry.num_attention_heads,
            )?;
            config.set_item(
                "num_key_value_heads", weights.config.geometry.num_key_value_heads,
            )?;
            config.set_item("head_dim", weights.config.geometry.head_dim)?;
            config.set_item(
                "max_position_embeddings",
                weights.config.geometry.max_position_embeddings,
            )?;
            config.set_item(
                "rms_norm_eps", f64::from_bits(weights.config.rms_norm_epsilon.bits),
            )?;
            config.set_item(
                "rope_theta", f64::from_bits(weights.config.rope_theta.bits),
            )?;
            config.set_item(
                "tie_word_embeddings", weights.config.tie_word_embeddings,
            )?;
            config.set_item(
                "dtype", weights.embed_weight.inner.bind(py).getattr("dtype")?,
            )?;
            config.set_item(
                "device", weights.embed_weight.inner.bind(py).getattr("device")?,
            )?;
            let implementation = py.import_bound("vosti_kernels.model_families.qwen3.physical")?;
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
        }).expect("python Qwen3 model-weight permission contract failed");
        Tracked::assume_new()
    }
    #[cfg(verus_only)]
    {
        unreachable!()
    }
}

// @kernel-bridge-end boundary::model_families::qwen3::model_weights_contract

} // verus!
