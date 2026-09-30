//! Gemma3 text projection of the tensor-runtime and role-binding boundary.

pub mod deployment;
pub mod config;
pub mod weights;

use crate::model_config::ModelArchitecture;
pub use weights::{Gemma3LayerWeights, Gemma3LayerWeightsPerms, Gemma3ModelWeights};

use crate::{types::*, proof::model::types::*, proof::tensor::types::*};
use crate::boundary::tensor_runtime as RT;
use vstd::prelude::*;

verus! {

pub closed spec fn weights_extension_repr_of(
    perms: &RT::ModelWeightsPerms,
) -> Gemma3ModelWeightsExtensionRepr {
    perms.gemma3_config()
}

pub proof fn lemma_weights_extension_repr(
    perms: &RT::ModelWeightsPerms,
)
    ensures
        weights_extension_repr_of(perms) == perms.gemma3_config(),
{
    reveal(weights_extension_repr_of);
}

pub proof fn lemma_common_layers_repr(perms: &RT::ModelWeightsPerms)
    requires perms.architecture() == ModelArchitecture::Gemma3Text,
    ensures
        RT::model_weights_repr_of(perms).layers
            == weights::common_layers_repr_of(perms),
        RT::model_weights_repr_of(perms).layers.len() == perms.num_layers(),
        forall|i: int| 0 <= i < perms.num_layers() ==>
            #[trigger] RT::model_weights_repr_of(perms).layers[i]
                == weights::layer_weights_common_repr_of(&perms.gemma3_layer(i)),
{
    RT::lemma_model_weights_common_layers_repr_projection(perms);
}

pub proof fn lemma_extension_attention_configs_valid(
    perms: &RT::ModelWeightsPerms,
)
    requires
        gemma3_config_valid(weights_extension_repr_of(perms)),
        weights_extension_repr_of(perms).layers.len() == perms.num_layers(),
    ensures
        forall|i: int| 0 <= i < perms.num_layers() ==>
            layer_attention_config_valid(
                #[trigger] weights_extension_repr_of(perms).layers[i].attention,
            ),
{
    reveal(weights_extension_repr_of);
    reveal(gemma3_config_valid);
    reveal(gemma3_family_config_valid);
    assert forall|i: int| 0 <= i < perms.num_layers() implies
        layer_attention_config_valid(
            #[trigger] perms.gemma3_config().layers[i].attention,
        ) by {
        assert(perms.gemma3_config().layers.map_values(
            |layer: Gemma3LayerWeightsExtensionRepr| layer.attention,
        )[i] == perms.gemma3_config().layers[i].attention);
    }
}

pub proof fn lemma_architecture_repr(perms: &RT::ModelWeightsPerms)
    requires perms.architecture() == ModelArchitecture::Gemma3Text,
    ensures
        RT::model_weights_architecture_repr_of(perms)
            == ModelWeightsArchitectureRepr::Gemma3Text(
                weights_extension_repr_of(perms),
            ),
{
    RT::lemma_model_weights_architecture_repr_projection(perms);
}

pub proof fn lemma_architecture_repr_implies_tag(
    perms: &RT::ModelWeightsPerms,
    family: Gemma3ModelWeightsExtensionRepr,
)
    requires
        RT::model_weights_architecture_repr_of(perms)
            == ModelWeightsArchitectureRepr::Gemma3Text(family),
    ensures perms.architecture() == ModelArchitecture::Gemma3Text,
{
    RT::lemma_model_weights_architecture_repr_projection(perms);
    match perms.architecture() {
        ModelArchitecture::Qwen3 => { assert(false); },
        ModelArchitecture::Llama3 => { assert(false); },
        ModelArchitecture::Gemma3Text => {},
        ModelArchitecture::Gemma4Text => { assert(false); },
    }
}

pub open spec fn configuration_ready(perms: &RT::ModelWeightsPerms) -> bool {
    perms.architecture() == ModelArchitecture::Gemma3Text
        && gemma3_config_valid(weights_extension_repr_of(perms))
}

pub proof fn lemma_execution_valid_implies_configuration_ready(
    weights: &RT::ModelWeights,
    runtime: &RT::ModelRuntime,
    perms: &RT::ModelWeightsPerms,
)
    requires
        RT::model_execution_valid(weights, runtime, perms),
        perms.architecture() == ModelArchitecture::Gemma3Text,
    ensures configuration_ready(perms),
{
    reveal(RT::model_execution_valid);
    match (weights, runtime) {
        (
            RT::ModelWeights::Gemma3Text(gemma),
            RT::ModelRuntime::Gemma3Text(_),
        ) => {
            assert(RT::gemma3_model_weights_bound(gemma, perms));
            reveal(RT::gemma3_model_weights_bound);
        },
        _ => { assert(false); },
    }
    reveal(configuration_ready);
}

} // verus!
