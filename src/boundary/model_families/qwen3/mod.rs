//! Qwen3 projection of the common tensor-runtime boundary.

pub mod deployment;
pub mod primitives;
pub mod config;
pub mod weights;

use crate::model_config::ModelArchitecture;
#[cfg(verus_only)]
pub use primitives::forward_config_repr;
pub use weights::{Qwen3LayerWeights, Qwen3LayerWeightsPerms, Qwen3ModelWeights};

use crate::{types::*, proof::model::types::*, proof::tensor::types::*};
use crate::boundary::tensor_runtime as RT;
use vstd::prelude::*;

verus! {

pub proof fn lemma_config_valid_implies_forward_config_repr(
    config: Qwen3ModelWeightsExtensionRepr,
)
    requires qwen3_config_valid(config),
    ensures dense_swiglu_forward_config_repr(config) == forward_config_repr(),
{
    lemma_qwen3_config_valid_implies_rms_norm_epsilon_identity(config);
    lemma_qwen3_config_valid_implies_rotary_identity(config);
    reveal(dense_swiglu_forward_config_repr);
    reveal(forward_config_repr);
}

pub closed spec fn weights_extension_repr_of(
    perms: &RT::ModelWeightsPerms,
) -> Qwen3ModelWeightsExtensionRepr {
    perms.qwen3_config()
}

pub proof fn lemma_weights_extension_repr(
    perms: &RT::ModelWeightsPerms,
)
    ensures
        weights_extension_repr_of(perms) == perms.qwen3_config(),
{
    reveal(weights_extension_repr_of);
}

pub proof fn lemma_bound_architecture_repr_valid(
    weights: &Qwen3ModelWeights,
    perms: &RT::ModelWeightsPerms,
)
    requires weights::model_weights_bound(weights, perms),
    ensures
        model_weights_architecture_repr_valid(
            RT::model_weights_repr_of(perms),
            RT::model_weights_architecture_repr_of(perms),
        ),
{
    reveal(weights::model_weights_bound);
    lemma_common_layers_repr(perms);
    lemma_architecture_repr(perms);
    lemma_weights_extension_repr(perms);
    let family = weights_extension_repr_of(perms);
    assert(qwen3_config_valid(family));
    lemma_qwen3_config_valid_implies_qk_rms_norm(family);
    assert(family.composition.qk_norm == QkNormKind::RmsNorm);
    assert forall|i: int| 0 <= i < perms.num_layers() implies
        #[trigger] layer_qk_norm_matches_composition(
            RT::model_weights_repr_of(perms).layers[i],
            family.composition,
        ) by {
        assert(0 <= i < weights.layers.len() as int);
        assert(weights::layer_weights_valid(
            &weights.layers[i], &perms.qwen3_layer(i),
        ));
        weights::lemma_layer_qk_norm_is_rms(
            &weights.layers[i], &perms.qwen3_layer(i),
        );
        assert(RT::model_weights_repr_of(perms).layers[i]
            == weights::layer_weights_common_repr_of(&perms.qwen3_layer(i)));
        reveal(layer_qk_norm_matches_composition);
    }
    reveal(model_weights_architecture_repr_valid);
}

pub proof fn lemma_common_layers_repr(perms: &RT::ModelWeightsPerms)
    requires perms.architecture() == ModelArchitecture::Qwen3,
    ensures
        RT::model_weights_repr_of(perms).layers
            == weights::common_layers_repr_of(perms),
        RT::model_weights_repr_of(perms).layers.len() == perms.num_layers(),
        forall|i: int| 0 <= i < perms.num_layers() ==>
            #[trigger] RT::model_weights_repr_of(perms).layers[i]
                == weights::layer_weights_common_repr_of(&perms.qwen3_layer(i)),
{
    RT::lemma_model_weights_common_layers_repr_projection(perms);
}

pub proof fn lemma_architecture_repr(perms: &RT::ModelWeightsPerms)
    requires perms.architecture() == ModelArchitecture::Qwen3,
    ensures
        RT::model_weights_architecture_repr_of(perms)
            == ModelWeightsArchitectureRepr::Qwen3(
                weights_extension_repr_of(perms),
            ),
{
    RT::lemma_model_weights_architecture_repr_projection(perms);
}

pub proof fn lemma_architecture_repr_implies_tag(
    perms: &RT::ModelWeightsPerms,
    family: Qwen3ModelWeightsExtensionRepr,
)
    requires
        RT::model_weights_architecture_repr_of(perms)
            == ModelWeightsArchitectureRepr::Qwen3(family),
    ensures perms.architecture() == ModelArchitecture::Qwen3,
{
    RT::lemma_model_weights_architecture_repr_projection(perms);
    match perms.architecture() {
        ModelArchitecture::Qwen3 => {},
        ModelArchitecture::Llama3 => { assert(false); },
        ModelArchitecture::Gemma3Text => { assert(false); },
        ModelArchitecture::Gemma4Text => { assert(false); },
    }
}

pub open spec fn configuration_ready(perms: &RT::ModelWeightsPerms) -> bool {
    perms.architecture() == ModelArchitecture::Qwen3
    && qwen3_config_valid(weights_extension_repr_of(perms))
}

pub proof fn lemma_execution_valid_implies_configuration_ready(
    weights: &RT::ModelWeights,
    runtime: &RT::ModelRuntime,
    perms: &RT::ModelWeightsPerms,
)
    requires
        RT::model_execution_valid(weights, runtime, perms),
        perms.architecture() == ModelArchitecture::Qwen3,
    ensures configuration_ready(perms),
{
    reveal(RT::model_execution_valid);
    reveal(RT::qwen3_model_weights_bound);
    reveal(configuration_ready);
}

} // verus!
