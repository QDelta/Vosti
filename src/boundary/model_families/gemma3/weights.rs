//! Gemma3 text-owned physical weight roles and their semantic projection.

use crate::model_config::{AttentionKind, ModelArchitecture};
use crate::boundary::model_families::gemma3::config::Gemma3Config;
use crate::{types::*, proof::model::types::*, proof::tensor::types::*};
use crate::boundary::tensor_runtime as RT;
use crate::proof::tensor::shape as TS;
use vstd::prelude::*;

#[cfg(not(verus_only))]
use pyo3::prelude::*;

verus! {

// @kernel-bridge-begin boundary::model_families::gemma3::model_weights_contract

pub type Gemma3LayerWeights = crate::boundary::four_norm_gated_weights::FourNormGatedLayerWeights;
pub type Gemma3LayerWeightsPerms = crate::boundary::four_norm_gated_weights::FourNormGatedLayerWeightsPerms;

pub open spec fn layer_weights_valid(
    weights: &Gemma3LayerWeights, perms: &Gemma3LayerWeightsPerms,
) -> bool {
    crate::boundary::four_norm_gated_weights::layer_weights_valid(weights, perms)
    && weights.layer_scale.is_none()
    && perms.layer_scale.is_none()
}

pub open spec fn layer_weights_common_repr_of(
    perms: &Gemma3LayerWeightsPerms,
) -> LayerWeightsRepr {
    crate::boundary::four_norm_gated_weights::layer_weights_common_repr_of(perms)
}

pub open spec fn common_layers_repr_of(
    perms: &RT::ModelWeightsPerms,
) -> Seq<LayerWeightsRepr> {
    Seq::new(perms.num_layers(), |i: int|
        layer_weights_common_repr_of(&perms.gemma3_layer(i)))
}

pub open spec fn layer_weights_extension_repr_of(
    perms: &Gemma3LayerWeightsPerms,
    attention_kind: AttentionKind,
    attention_window: nat,
    attention_scale: AttentionScaleRepr,
) -> Gemma3LayerWeightsExtensionRepr {
    Gemma3LayerWeightsExtensionRepr {
        pre_feedforward_norm: perms.pre_feedforward_norm.repr_1d(),
        post_feedforward_norm: perms.post_feedforward_norm.repr_1d(),
        attention: match attention_kind {
            AttentionKind::SlidingWindow =>
                AttentionConfigRepr::SlidingWindow(attention_window),
            AttentionKind::Full => AttentionConfigRepr::Full,
        },
        attention_scale,
        row_parameters: gemma3_row_parameters_repr(attention_kind),
    }
}

pub type Gemma3ModelWeights =
    crate::boundary::four_norm_gated_weights::FourNormGatedModelWeights<Gemma3Config>;

pub open spec fn model_weights_bound(
    weights: &Gemma3ModelWeights,
    perms: &RT::ModelWeightsPerms,
) -> bool {
    let layers = Seq::new(perms.num_layers(), |i: int|
        layer_weights_extension_repr_of(
            &perms.gemma3_layer(i), perms.gemma3_attention_kind(i),
            weights.config.sliding_window as nat,
            gemma3_attention_scale_repr(weights.config),
        )
    );
    let config = gemma3_config_repr(weights.config, layers);
    let geometry = config.geometry;
    perms.architecture() == ModelArchitecture::Gemma3Text
    && perms.gemma3_config() == config
    && gemma3_config_valid(config)
    && weights.embed_weight.id() == perms.embed_weight_id()
    && weights.layers.len() == perms.num_layers()
    && weights.final_norm.id() == perms.final_norm_id()
    && weights.lm_head.id() == perms.lm_head_id()
    && weights.embed_weight.id() == weights.lm_head.id()
    && weights.config.geometry.hidden_size == perms.gemma3_hidden_size()
    && weights.config.sliding_window == perms.gemma3_sliding_window()
    && TS::tensor2d_shape(
        perms.embed_weight_repr(), geometry.vocab_size, geometry.hidden_size,
    )
    && perms.final_norm_repr().len() == geometry.hidden_size
    && TS::tensor2d_shape(
        perms.lm_head_repr(), geometry.vocab_size, geometry.hidden_size,
    )
    && (forall|i: int| 0 <= i < weights.layers.len() as int ==> {
        &&& #[trigger] layer_weights_valid(
            &weights.layers[i], &perms.gemma3_layer(i),
        )
        &&& weights.layers[i].attention_kind == perms.gemma3_attention_kind(i)
        &&& crate::boundary::four_norm_gated_weights::layer_shapes_valid(
            &perms.gemma3_layer(i), geometry.hidden_size, geometry.intermediate_size,
            attention_geometry_repr(geometry),
        )
    })
}

pub proof fn lemma_layer_attention_geometry_matches_config(
    weights: &Gemma3ModelWeights,
    perms: &RT::ModelWeightsPerms,
    layer: int,
)
    requires
        model_weights_bound(weights, perms),
        0 <= layer < weights.layers.len() as int,
    ensures
        crate::boundary::dense_layer_primitives::layer_attention_geometry_repr(
            layer_weights_common_repr_of(&perms.gemma3_layer(layer)),
        ) == attention_geometry_repr(
            dense_geometry_repr(weights.config.geometry),
        ),
{
    reveal(model_weights_bound);
    let geometry = dense_geometry_repr(weights.config.geometry);
    assert(layer_weights_valid(
        &weights.layers[layer], &perms.gemma3_layer(layer),
    ));
    assert(geometry.head_dim > 0);
    assert(TS::tensor2d_shape(
        perms.gemma3_layer(layer).q_proj.repr_2d(),
        geometry.num_attention_heads * geometry.head_dim,
        geometry.hidden_size,
    ));
    assert(TS::tensor2d_shape(
        perms.gemma3_layer(layer).k_proj.repr_2d(),
        geometry.num_key_value_heads * geometry.head_dim,
        geometry.hidden_size,
    ));
    reveal(layer_weights_common_repr_of);
    crate::boundary::dense_layer_primitives::lemma_layer_attention_geometry_from_shapes(
        layer_weights_common_repr_of(&perms.gemma3_layer(layer)),
        geometry,
    );
}

pub proof fn lemma_physical_deployment_config_valid(
    weights: &Gemma3ModelWeights,
    perms: &RT::ModelWeightsPerms,
)
    requires model_weights_bound(weights, perms),
    ensures
        model_deployment_config_valid(
            RT::gemma3_physical_deployment_config_repr(weights),
        ),
{
    reveal(model_weights_bound);
    let layers = Seq::new(perms.num_layers(), |i: int|
        layer_weights_extension_repr_of(
            &perms.gemma3_layer(i), perms.gemma3_attention_kind(i),
            weights.config.sliding_window as nat,
            gemma3_attention_scale_repr(weights.config),
        )
    );
    let config = gemma3_config_repr(weights.config, layers);
    assert(gemma3_config_valid(config));
    lemma_gemma3_config_valid_implies_deployment_projection(config);
    reveal(RT::gemma3_physical_deployment_config_repr);
    reveal(RT::gemma3_physical_layer_attention_kinds);
    reveal(gemma3_deployment_config_repr);
    reveal(layer_weights_extension_repr_of);
    assert forall|i: int| 0 <= i < weights.layers.len() as int implies
        #[trigger] weights.layers[i].attention_kind
            == perms.gemma3_attention_kind(i) by {
        assert(layer_weights_valid(
            &weights.layers[i], &perms.gemma3_layer(i),
        ));
    }
    assert(Seq::new(weights.layers.len() as nat, |i: int|
        gemma3_layer_attention_config(
            weights.config, weights.layers[i].attention_kind,
        )) =~= layers.map_values(
            |layer: Gemma3LayerWeightsExtensionRepr| layer.attention,
        ));
    reveal(model_deployment_config_valid);
    reveal(gemma3_deployment_config_valid);
}

// Bind the exact Gemma facade without independently enabling execution. The
// Python guard checks physical roles, tied-head identity, dtype/device/layout,
// and internally consistent shapes before minting the ghost representation.
#[verifier::external_body]
pub fn bind_model_weights_perms(
    weights: &Gemma3ModelWeights,
    expected_num_layers: usize,
) -> (out: Tracked<RT::ModelWeightsPerms>)
    requires weights.layers.len() == expected_num_layers,
    ensures model_weights_bound(weights, &out@),
{
    #[cfg(not(verus_only))]
    {
        pyo3::Python::with_gil(|py| -> pyo3::PyResult<()> {
            if weights.layers.iter().any(|layer| layer.layer_scale.is_some()) {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "Gemma 3 layers must not contain an output scale tensor",
                ));
            }
            let layer_tuples: Vec<pyo3::Bound<pyo3::types::PyTuple>> = weights
                .layers
                .iter()
                .map(|layer| crate::boundary::four_norm_gated_weights::python_layer_roles(
                    py, layer, false,
                ))
                .collect::<pyo3::PyResult<_>>()?;
            let attention_kinds: Vec<&str> = weights
                .layers
                .iter()
                .map(|layer| match &layer.attention_kind {
                    AttentionKind::SlidingWindow => "sliding_attention",
                    AttentionKind::Full => "full_attention",
                })
                .collect();
            let layers = pyo3::types::PyList::new_bound(py, &layer_tuples);
            let attention_kinds = pyo3::types::PyList::new_bound(py, &attention_kinds);
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
            config.set_item("sliding_window", weights.config.sliding_window)?;
            config.set_item(
                "rms_norm_eps", f64::from_bits(weights.config.rms_norm_epsilon.bits),
            )?;
            config.set_item(
                "query_pre_attn_scalar",
                f64::from_bits(weights.config.query_pre_attention_scalar.bits),
            )?;
            config.set_item(
                "local_rope_theta",
                f64::from_bits(weights.config.local_rope_theta.bits),
            )?;
            config.set_item(
                "global_rope_theta",
                f64::from_bits(weights.config.global_rope_theta.bits),
            )?;
            config.set_item(
                "global_rope_factor",
                f64::from_bits(weights.config.global_rope_factor.bits),
            )?;
            config.set_item("layer_types", &attention_kinds)?;
            let implementation = py.import_bound("vosti_kernels.model_families.gemma3.physical")?;
            implementation
                .getattr("validate_model_weights_runtime_contract")?
                .call1((
                    weights.embed_weight.inner.bind(py),
                    layers,
                    attention_kinds,
                    weights.final_norm.inner.bind(py),
                    weights.lm_head.inner.bind(py),
                    config,
                    expected_num_layers,
                ))?;
            Ok(())
        })
        .expect("python Gemma 3 model-weight permission contract failed");
        Tracked::assume_new()
    }
    #[cfg(verus_only)]
    {
        unreachable!()
    }
}

// @kernel-bridge-end boundary::model_families::gemma3::model_weights_contract

} // verus!
