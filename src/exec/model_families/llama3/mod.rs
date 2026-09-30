//! Checked Llama3 model-forward composition.

use crate::model_config::ModelArchitecture;
use crate::boundary::dense_layer_primitives as DLP;
use crate::boundary::dense_swiglu_decoder as DENSE;
use crate::boundary::model_families::llama3 as LLAMA_BOUNDARY;
use crate::exec::dense_swiglu_model as DENSE_MODEL;
use crate::{types::*, proof::model::types::*, proof::tensor::types::*};
use crate::boundary::tensor_runtime as RT;
use crate::proof::tensor::shape as TS;
use vstd::prelude::*;

verus! {

#[verifier::spinoff_prover]
pub(crate) proof fn lemma_model_execution_ready(
    runtime: &RT::ModelFamilyRuntime,
    llama: &RT::Llama3ModelWeights,
    wp: &RT::ModelWeightsPerms,
)
    requires
        RT::llama3_runtime_matches_weights(runtime, llama),
        RT::llama3_model_weights_bound(llama, wp),
    ensures DENSE_MODEL::model_execution_ready(
        runtime,
        &llama.embed_weight,
        &llama.layers,
        &llama.final_norm,
        &llama.lm_head,
        wp,
        llama3_config_repr(llama.config).geometry.head_dim,
        LLAMA_BOUNDARY::forward_config_repr(llama3_config_repr(llama.config)),
    ),
{
    reveal(DENSE_MODEL::model_execution_ready);
    reveal(RT::llama3_runtime_matches_weights);
    reveal(RT::llama3_physical_deployment_config_repr);
    reveal(RT::llama3_model_weights_bound);
    reveal(LLAMA_BOUNDARY::weights::model_weights_bound);
    reveal(RT::ModelWeightsPerms::dense_swiglu_layer);
    reveal(LLAMA_BOUNDARY::weights::layer_weights_common_repr_of);
    reveal(LLAMA_BOUNDARY::forward_config_repr);
    reveal(dense_swiglu_forward_config_repr);
    let family = llama3_config_repr(llama.config);
    assert(llama3_config_valid(family));
    lemma_llama3_config_valid_implies_rms_norm_epsilon_identity(family);
    assert(RT::family_runtime_deployment_config_repr(runtime)
        == Some(ModelDeploymentConfigRepr::Llama3(family)));
    RT::lemma_llama3_runtime_rms_norm_matches(runtime);
    assert(RT::dense_swiglu_runtime_rms_norm_matches(
        runtime, LLAMA_BOUNDARY::forward_config_repr(llama3_config_repr(llama.config)).rms_norm_epsilon));
    RT::lemma_llama3_runtime_qk_norm_matches(runtime);
    LLAMA_BOUNDARY::lemma_common_layers_repr(wp);
    assert(wp.architecture() == ModelArchitecture::Llama3);
    assert(RT::family_runtime_execution_valid(runtime)) by {
        reveal(RT::llama3_runtime_configuration_valid);
    }
    assert(llama.layers.len() == wp.num_layers());
    assert(llama.embed_weight.id() == wp.embed_weight_id());
    assert(llama.final_norm.id() == wp.final_norm_id());
    assert(llama.lm_head.id() == wp.lm_head_id());
    assert(TS::rectangular(wp.embed_weight_repr())) by {
        reveal(TS::tensor2d_shape);
    }
    assert(TS::rectangular(wp.lm_head_repr())) by {
        reveal(TS::tensor2d_shape);
    }
    assert(RT::model_weights_repr_of(wp).layers =~= Seq::new(
        wp.num_layers(),
        |i: int| DENSE::layer_weights_repr_of(
            &wp.dense_swiglu_layer(i),
            llama3_config_repr(llama.config).geometry.head_dim,
        ),
    )) by {
        assert forall|i: int| 0 <= i < wp.num_layers() implies
            #[trigger] RT::model_weights_repr_of(wp).layers[i]
                == DENSE::layer_weights_repr_of(
                    &wp.dense_swiglu_layer(i),
                    llama3_config_repr(llama.config).geometry.head_dim,
        ) by {
            assert(0 <= i < llama.layers.len() as int);
        }
    }
    assert forall|i: int| 0 <= i < llama.layers.len() as int implies
        #[trigger] DENSE_MODEL::layer_execution_ready(
            runtime, &llama.layers[i], &wp.dense_swiglu_layer(i),
            llama3_config_repr(llama.config).geometry.head_dim,
            LLAMA_BOUNDARY::forward_config_repr(llama3_config_repr(llama.config))) by {
        assert(LLAMA_BOUNDARY::weights::layer_weights_valid(
            &llama.layers[i], &wp.llama3_layer(i),
        ));
        LLAMA_BOUNDARY::weights::lemma_layer_qk_norm_is_disabled(
            &llama.layers[i], &wp.llama3_layer(i),
        );
        LLAMA_BOUNDARY::weights::lemma_layer_attention_geometry_matches_config(
            llama, wp, i,
        );
        assert(DENSE::layer_weights_repr_of(
            &wp.dense_swiglu_layer(i),
            llama3_config_repr(llama.config).geometry.head_dim,
        ) == LLAMA_BOUNDARY::weights::layer_weights_common_repr_of(
            wp, &wp.llama3_layer(i),
        ));
        assert(DLP::layer_attention_geometry_repr(
            DENSE::layer_weights_repr_of(
                &wp.dense_swiglu_layer(i), family.geometry.head_dim,
            ),
        ) == attention_geometry_repr(family.geometry));
        RT::lemma_llama3_runtime_rotary_matches(
            runtime, family,
        );
    }
    assert(DENSE_MODEL::model_execution_ready(
        runtime,
        &llama.embed_weight,
        &llama.layers,
        &llama.final_norm,
        &llama.lm_head,
        wp,
        llama3_config_repr(llama.config).geometry.head_dim,
        LLAMA_BOUNDARY::forward_config_repr(llama3_config_repr(llama.config)),
    ));
}

} // verus!
