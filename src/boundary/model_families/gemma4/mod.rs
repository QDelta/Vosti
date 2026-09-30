//! Dense text-only Gemma-4 weight binding. Engine admission is separate.

pub mod config;
pub mod weights;
pub mod deployment;

use crate::model_config::ModelArchitecture;
pub use weights::{Gemma4LayerWeights, Gemma4LayerWeightsPerms, Gemma4ModelWeights};

use crate::{types::*, proof::model::types::*, proof::tensor::types::*};
use crate::boundary::tensor_runtime as RT;
use vstd::prelude::*;

verus! {

pub open spec fn configuration_ready(perms: &RT::ModelWeightsPerms) -> bool {
    perms.architecture() == ModelArchitecture::Gemma4Text
    && gemma4_config_valid(weights_extension_repr_of(perms))
}

pub proof fn lemma_execution_valid_implies_configuration_ready(
    weights: &RT::ModelWeights, runtime: &RT::ModelRuntime, perms: &RT::ModelWeightsPerms,
)
    requires RT::model_execution_valid(weights, runtime, perms),
        perms.architecture() == ModelArchitecture::Gemma4Text,
    ensures configuration_ready(perms),
{
    reveal(RT::model_execution_valid);
    lemma_weights_extension_repr(perms);
    match (weights, runtime) {
        (RT::ModelWeights::Gemma4Text(gemma), RT::ModelRuntime::Gemma4Text(_)) => {
            assert(weights::model_weights_bound(gemma, perms));
        },
        _ => { assert(false); },
    }
}

pub proof fn lemma_extension_attention_configs_valid(perms: &RT::ModelWeightsPerms)
    requires gemma4_config_valid(weights_extension_repr_of(perms)),
        weights_extension_repr_of(perms).decoder.layers.len() == perms.num_layers(),
    ensures forall|i: int| 0 <= i < perms.num_layers() ==>
        layer_attention_config_valid(
            #[trigger] weights_extension_repr_of(perms).decoder.layers[i].attention),
{
    lemma_weights_extension_repr(perms);
    let config = weights_extension_repr_of(perms);
    assert forall|i: int| 0 <= i < perms.num_layers() implies
        layer_attention_config_valid(#[trigger] config.decoder.layers[i].attention) by {
        assert(gemma4_deployment_projection(config).layer_attention[i]
            == config.decoder.layers[i].attention);
    }
}

pub closed spec fn weights_extension_repr_of(
    perms: &RT::ModelWeightsPerms,
) -> Gemma4ModelWeightsExtensionRepr {
    perms.gemma4_config()
}

pub proof fn lemma_weights_extension_repr(
    perms: &RT::ModelWeightsPerms,
)
    ensures weights_extension_repr_of(perms) == perms.gemma4_config(),
{
    reveal(weights_extension_repr_of);
}

pub proof fn lemma_common_layers_repr(perms: &RT::ModelWeightsPerms)
    requires perms.architecture() == ModelArchitecture::Gemma4Text,
    ensures
        RT::model_weights_repr_of(perms).layers
            == weights::common_layers_repr_of(perms),
        RT::model_weights_repr_of(perms).layers.len() == perms.num_layers(),
        forall|i: int| 0 <= i < perms.num_layers() ==>
            #[trigger] RT::model_weights_repr_of(perms).layers[i]
                == weights::layer_weights_common_repr_of(
                    &perms.four_norm_gated_weights().layers[i]),
{
    RT::lemma_model_weights_common_layers_repr_projection(perms);
}

// Physical binding is sufficient for a coherent model representation. Runtime
// qualification is intentionally not a premise or a conclusion of this lemma.
pub proof fn lemma_bound_architecture_repr_valid(
    weights: &weights::Gemma4ModelWeights, perms: &RT::ModelWeightsPerms,
)
    requires weights::model_weights_bound(weights, perms),
    ensures
        model_weights_architecture_repr_valid(
            RT::model_weights_repr_of(perms), RT::model_weights_architecture_repr_of(perms)),
        RT::model_weights_repr_of(perms).layers.len() == perms.num_layers(),
        model_weights_deployment_config_repr(RT::model_weights_architecture_repr_of(perms))
            == RT::physical_model_deployment_config_repr(&RT::ModelWeights::Gemma4Text(*weights)),
{
    lemma_architecture_repr(perms);
    lemma_weights_extension_repr(perms);
    RT::lemma_model_weights_common_layers_repr_projection(perms);
    let physical = perms.four_norm_gated_weights();
    weights::lemma_physical_deployment_config_valid(weights, &physical);
    let family = weights_extension_repr_of(perms);
    assert forall|i: int| 0 <= i < perms.num_layers() implies
        #[trigger] layer_qk_norm_matches_composition(
            RT::model_weights_repr_of(perms).layers[i], family.decoder.composition) by {
        assert(RT::model_weights_repr_of(perms).layers[i]
            == crate::boundary::four_norm_gated_weights::layer_weights_common_repr_of(
                &physical.layers[i]));
    }
    let deployment = gemma4_deployment_config_repr(weights.config,
        weights::physical_attention_kinds(weights));
    assert(gemma4_deployment_projection(family).layer_attention =~= deployment.layer_attention);
    assert(gemma4_deployment_projection(family) == deployment);
}

pub proof fn lemma_architecture_repr(perms: &RT::ModelWeightsPerms)
    requires perms.architecture() == ModelArchitecture::Gemma4Text,
    ensures RT::model_weights_architecture_repr_of(perms)
        == ModelWeightsArchitectureRepr::Gemma4Text(weights_extension_repr_of(perms)),
{
    RT::lemma_model_weights_architecture_repr_projection(perms);
}

pub proof fn lemma_architecture_repr_implies_tag(
    perms: &RT::ModelWeightsPerms, family: Gemma4ModelWeightsExtensionRepr,
)
    requires RT::model_weights_architecture_repr_of(perms)
        == ModelWeightsArchitectureRepr::Gemma4Text(family),
    ensures perms.architecture() == ModelArchitecture::Gemma4Text,
{
    RT::lemma_model_weights_architecture_repr_projection(perms);
    match perms.architecture() {
        ModelArchitecture::Qwen3 => { assert(false); },
        ModelArchitecture::Llama3 => { assert(false); },
        ModelArchitecture::Gemma3Text => { assert(false); },
        ModelArchitecture::Gemma4Text => {},
    }
}

} // verus!
