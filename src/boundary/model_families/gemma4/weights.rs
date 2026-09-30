//! Exact dense Gemma-4 weight roles and their shared decoder projection.
//!
//! Binding weights is not engine admission: no runtime/kernel capability is
//! produced here. The checked forward and deployment gate must consume both.

use crate::model_config::{AttentionKind, ModelArchitecture};
use crate::boundary::model_families::gemma4::config::Gemma4Config;
use crate::boundary::four_norm_gated_weights as SHARED;
use crate::{types::*, proof::model::types::*, proof::tensor::types::*};
use crate::boundary::tensor_runtime as RT;
use crate::proof::tensor::shape as TS;
use vstd::prelude::*;

#[cfg(not(verus_only))]
use pyo3::prelude::*;

verus! {

// @kernel-bridge-begin boundary::model_families::gemma4::model_weights_contract

pub type Gemma4LayerWeights = SHARED::FourNormGatedLayerWeights;
pub type Gemma4LayerWeightsPerms = SHARED::FourNormGatedLayerWeightsPerms;
pub type Gemma4ModelWeights = SHARED::FourNormGatedModelWeights<Gemma4Config>;
pub type Gemma4ModelWeightsPerms = RT::ModelWeightsPerms;

pub open spec fn layer_weights_valid(
    weights: &Gemma4LayerWeights, perms: &Gemma4LayerWeightsPerms,
) -> bool {
    SHARED::layer_weights_valid(weights, perms)
    && weights.layer_scale.is_some()
    && perms.layer_scale.is_some()
}

pub open spec fn layer_weights_common_repr_of(
    perms: &Gemma4LayerWeightsPerms,
) -> LayerWeightsRepr {
    SHARED::layer_weights_common_repr_of(perms)
}

pub open spec fn physical_attention_kinds(weights: &Gemma4ModelWeights) -> Seq<AttentionKind> {
    Seq::new(weights.layers.len() as nat, |i: int| weights.layers[i].attention_kind)
}

pub open spec fn layer_extension_repr_of(
    config: Gemma4DeploymentConfigRepr, kind: AttentionKind,
    perms: &SHARED::FourNormGatedLayerWeightsPerms,
) -> FourNormGatedLayerExtensionRepr {
    let scalar = match SHARED::layer_scale_repr(perms) {
        Some(value) => value,
        None => Seq::empty(), // Excluded by model_weights_bound/config_valid.
    };
    FourNormGatedLayerExtensionRepr {
        pre_feedforward_norm: perms.pre_feedforward_norm.repr_1d(),
        post_feedforward_norm: perms.post_feedforward_norm.repr_1d(),
        attention: match kind {
            AttentionKind::Full => AttentionConfigRepr::Full,
            AttentionKind::SlidingWindow => AttentionConfigRepr::SlidingWindow(config.sliding_window),
        },
        attention_scale: unit_attention_scale_repr(),
        row_parameters: gemma4_row_parameters_repr(config, kind, scalar),
    }
}

pub open spec fn config_repr_of(
    weights: &Gemma4ModelWeights, perms: &SHARED::FourNormGatedModelWeightsPerms,
) -> Gemma4ModelWeightsExtensionRepr {
    let deployment = gemma4_deployment_config_repr(weights.config, physical_attention_kinds(weights));
    gemma4_config_repr(weights.config, Seq::new(weights.layers.len() as nat, |i: int|
        layer_extension_repr_of(deployment, weights.layers[i].attention_kind, &perms.layers[i])))
}

pub open spec fn common_layers_repr_of(
    perms: &RT::ModelWeightsPerms,
) -> Seq<LayerWeightsRepr> {
    Seq::new(perms.num_layers(), |i: int|
        layer_weights_common_repr_of(&perms.four_norm_gated_weights().layers[i]))
}

pub open spec fn physical_weights_bound(
    weights: &Gemma4ModelWeights, perms: &SHARED::FourNormGatedModelWeightsPerms,
) -> bool {
    let deployment = gemma4_deployment_config_repr(weights.config, physical_attention_kinds(weights));
    let geometry = deployment.geometry;
    SHARED::model_roles_bound(weights, perms)
    && gemma4_config_valid(config_repr_of(weights, perms))
    && weights.layers.len() == geometry.num_layers
    && weights.embed_weight.id() == weights.lm_head.id()
    && TS::tensor2d_shape(perms.embed_weight.repr_2d(), geometry.vocab_size, geometry.hidden_size)
    && perms.final_norm.repr_1d().len() == geometry.hidden_size
    && perms.lm_head.repr_2d() == perms.embed_weight.repr_2d()
    && forall|i: int| #![trigger perms.layers[i]] 0 <= i < weights.layers.len() ==> {
        let layer = &weights.layers[i];
        let lp = &perms.layers[i];
        &&& SHARED::layer_shapes_valid(lp, geometry.hidden_size,
            geometry.intermediate_size, gemma4_layer_attention_geometry(deployment, layer.attention_kind))
        &&& layer.layer_scale.is_some()
        &&& lp.layer_scale.is_some()
        &&& (deployment.attention_k_eq_v && layer.attention_kind == AttentionKind::Full ==>
            layer.k_proj.id() == layer.v_proj.id()
            && lp.k_proj.repr_2d() == lp.v_proj.repr_2d())
    }
}

pub proof fn lemma_physical_deployment_config_valid(
    weights: &Gemma4ModelWeights, perms: &SHARED::FourNormGatedModelWeightsPerms,
)
    requires physical_weights_bound(weights, perms),
    ensures gemma4_deployment_config_valid(
        gemma4_deployment_config_repr(weights.config, physical_attention_kinds(weights))),
{
    let config = config_repr_of(weights, perms);
    lemma_gemma4_config_valid_implies_deployment_projection(config);
    let physical = gemma4_deployment_config_repr(weights.config, physical_attention_kinds(weights));
    assert(gemma4_deployment_projection(config).layer_attention =~= physical.layer_attention);
    assert(gemma4_deployment_projection(config) == physical);
}

// Bind the shared physical record to the same outer permission facade used by
// other families. These equations record identity, not a forward assumption.
pub open spec fn model_weights_bound(
    weights: &Gemma4ModelWeights, perms: &RT::ModelWeightsPerms,
) -> bool {
    let physical = perms.four_norm_gated_weights();
    &&& perms.architecture() == ModelArchitecture::Gemma4Text
    &&& perms.num_layers() == weights.layers.len()
    &&& physical_weights_bound(weights, &physical)
    &&& perms.gemma4_config() == config_repr_of(weights, &physical)
    &&& perms.embed_weight_id() == physical.embed_weight.id()
    &&& perms.embed_weight_repr() == physical.embed_weight.repr_2d()
    &&& perms.final_norm_id() == physical.final_norm.id()
    &&& perms.final_norm_repr() == physical.final_norm.repr_1d()
    &&& perms.lm_head_id() == physical.lm_head.id()
    &&& perms.lm_head_repr() == physical.lm_head.repr_2d()
}

#[verifier::external_body]
pub fn bind_model_weights_perms(
    weights: &Gemma4ModelWeights, expected_num_layers: usize,
) -> (out: Tracked<Gemma4ModelWeightsPerms>)
    requires weights.layers.len() == expected_num_layers,
    ensures model_weights_bound(weights, &out@),
{
    #[cfg(not(verus_only))]
    {
        Python::with_gil(|py| -> PyResult<()> {
            let layers = weights.layers.iter().map(|layer|
                SHARED::python_layer_roles(py, layer, true)
            ).collect::<PyResult<Vec<_>>>()?;
            let kinds = weights.layers.iter().map(|layer| match layer.attention_kind {
                AttentionKind::SlidingWindow => "sliding_attention",
                AttentionKind::Full => "full_attention",
            }).collect::<Vec<_>>();
            let config = runtime_config_dict(py, weights.config, &kinds)?;
            py.import_bound("vosti_kernels.model_families.gemma4.physical")?
                .getattr("validate_model_weights_runtime_contract")?.call1((
                    weights.embed_weight.inner.bind(py),
                    pyo3::types::PyList::new_bound(py, layers),
                    pyo3::types::PyList::new_bound(py, &kinds),
                    weights.final_norm.inner.bind(py), weights.lm_head.inner.bind(py),
                    config, expected_num_layers,
                ))?;
            Ok(())
        }).expect("python Gemma 4 model-weight permission contract failed");
        Tracked::assume_new()
    }
    #[cfg(verus_only)]
    { unreachable!() }
}

// @kernel-bridge-end boundary::model_families::gemma4::model_weights_contract

} // verus!

// @kernel-bridge-begin boundary::model_families::gemma4::config_dictionary
#[cfg(not(verus_only))]
pub(crate) fn runtime_config_dict<'py>(
    py: Python<'py>, config: Gemma4Config, kinds: &[&str],
) -> PyResult<pyo3::Bound<'py, pyo3::types::PyDict>> {
    let fraction = f64::from_bits(config.global_partial_rotary_factor.bits);
    // The rotary fraction must be finite and in (0, 1].
    if !fraction.is_finite() || fraction <= 0.0 || fraction > 1.0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Gemma 4 Rust binding requires rotary fraction in (0, 1]"));
    }
    let out = pyo3::types::PyDict::new_bound(py);
    for (field, value) in [
        ("vocab_size", config.geometry.vocab_size),
        ("hidden_size", config.geometry.hidden_size),
        ("intermediate_size", config.geometry.intermediate_size),
        ("num_hidden_layers", config.geometry.num_layers),
        ("num_attention_heads", config.geometry.num_attention_heads),
        ("num_key_value_heads", config.geometry.num_key_value_heads),
        ("num_global_key_value_heads", config.num_global_key_value_heads),
        ("head_dim", config.geometry.head_dim), ("global_head_dim", config.global_head_dim),
        ("max_position_embeddings", config.geometry.max_position_embeddings),
        ("sliding_window", config.sliding_window),
    ] { out.set_item(field, value)?; }
    for (field, value) in [
        ("rms_norm_eps", config.rms_norm_epsilon.bits),
        ("local_rope_theta", config.local_rope_theta.bits),
        ("global_rope_theta", config.global_rope_theta.bits),
        ("global_rope_factor", config.global_rope_factor.bits),
        ("global_partial_rotary_factor", config.global_partial_rotary_factor.bits),
    ] { out.set_item(field, f64::from_bits(value))?; }
    out.set_item("attention_k_eq_v", config.attention_k_eq_v)?;
    out.set_item("final_logit_softcapping", config.final_logit_softcap.map(|v| f64::from_bits(v.bits)))?;
    out.set_item("layer_types", kinds)?;
    Ok(out)
}
// @kernel-bridge-end boundary::model_families::gemma4::config_dictionary
