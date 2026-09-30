//! Checked Qwen3 model-forward composition.

use crate::model_config::ModelArchitecture;
use crate::boundary::dense_layer_primitives as DLP;
use crate::boundary::dense_swiglu_decoder as DENSE;
use crate::boundary::model_families::qwen3 as QWEN_BOUNDARY;
use crate::exec::dense_swiglu_model as DENSE_MODEL;
use crate::{types::*, proof::model::types::*, proof::tensor::types::*};
use crate::boundary::tensor_runtime as RT;
use crate::proof::tensor::shape as TS;
use vstd::prelude::*;

verus! {

#[verifier::spinoff_prover]
pub(crate) proof fn lemma_model_execution_ready(
    runtime: &RT::ModelFamilyRuntime,
    qwen: &RT::Qwen3ModelWeights,
    wp: &RT::ModelWeightsPerms,
)
    requires
        RT::qwen3_runtime_matches_weights(runtime, qwen),
        RT::qwen3_model_weights_bound(qwen, wp),
    ensures DENSE_MODEL::model_execution_ready(
        runtime,
        &qwen.embed_weight,
        &qwen.layers,
        &qwen.final_norm,
        &qwen.lm_head,
        wp,
        qwen3_config_repr(qwen.config).geometry.head_dim,
        QWEN_BOUNDARY::forward_config_repr(),
    ),
{
    reveal(DENSE_MODEL::model_execution_ready);
    reveal(RT::qwen3_runtime_matches_weights);
    reveal(RT::qwen3_physical_deployment_config_repr);
    reveal(RT::qwen3_model_weights_bound);
    reveal(QWEN_BOUNDARY::weights::model_weights_bound);
    reveal(RT::ModelWeightsPerms::dense_swiglu_layer);
    reveal(QWEN_BOUNDARY::weights::layer_weights_common_repr_of);
    reveal(QWEN_BOUNDARY::forward_config_repr);
    RT::lemma_qwen3_runtime_rms_norm_matches(runtime);
    assert(RT::dense_swiglu_runtime_rms_norm_matches(
        runtime, QWEN_BOUNDARY::forward_config_repr().rms_norm_epsilon));
    RT::lemma_qwen3_runtime_qk_norm_matches(runtime);
    QWEN_BOUNDARY::lemma_common_layers_repr(wp);
    assert(wp.architecture() == ModelArchitecture::Qwen3);
    assert(RT::family_runtime_execution_valid(runtime)) by {
        reveal(RT::qwen3_runtime_configuration_valid);
    }
    assert(qwen.layers.len() == wp.num_layers());
    assert(qwen.embed_weight.id() == wp.embed_weight_id());
    assert(qwen.final_norm.id() == wp.final_norm_id());
    assert(qwen.lm_head.id() == wp.lm_head_id());
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
            qwen3_config_repr(qwen.config).geometry.head_dim,
        ),
    )) by {
        assert forall|i: int| 0 <= i < wp.num_layers() implies
            #[trigger] RT::model_weights_repr_of(wp).layers[i]
                == DENSE::layer_weights_repr_of(
                    &wp.dense_swiglu_layer(i),
                    qwen3_config_repr(qwen.config).geometry.head_dim,
        ) by {
            assert(0 <= i < qwen.layers.len() as int);
            assert(QWEN_BOUNDARY::weights::layer_weights_valid(
                &qwen.layers[i], &wp.qwen3_layer(i),
            ));
            QWEN_BOUNDARY::weights::lemma_layer_qk_norm_is_rms(
                &qwen.layers[i], &wp.qwen3_layer(i),
            );
            assert(rms_q_norm_weight(DENSE::qk_norm_weights_repr_of(
                &wp.qwen3_layer(i).qk_norm,
            )).len() == qwen3_config_repr(qwen.config).geometry.head_dim);
        }
    }
    assert forall|i: int| 0 <= i < qwen.layers.len() as int implies
        #[trigger] DENSE_MODEL::layer_execution_ready(
            runtime, &qwen.layers[i], &wp.dense_swiglu_layer(i),
            qwen3_config_repr(qwen.config).geometry.head_dim,
            QWEN_BOUNDARY::forward_config_repr()) by {
        assert(QWEN_BOUNDARY::weights::layer_weights_valid(
            &qwen.layers[i], &wp.qwen3_layer(i),
        ));
        QWEN_BOUNDARY::weights::lemma_layer_qk_norm_is_rms(
            &qwen.layers[i], &wp.qwen3_layer(i),
        );
        QWEN_BOUNDARY::weights::lemma_layer_attention_geometry_matches_config(
            qwen, wp, i,
        );
        assert(DENSE::layer_weights_repr_of(
            &wp.dense_swiglu_layer(i),
            qwen3_config_repr(qwen.config).geometry.head_dim,
        ) == QWEN_BOUNDARY::weights::layer_weights_common_repr_of(
            &wp.qwen3_layer(i),
        ));
        reveal(RT::qwen3_runtime_attention_geometry_matches);
        RT::lemma_qwen3_runtime_rotary_matches(
            runtime,
            DLP::layer_attention_geometry_repr(
                QWEN_BOUNDARY::weights::layer_weights_common_repr_of(
                    &wp.qwen3_layer(i),
                ),
            ),
        );
    }
    assert(DENSE_MODEL::model_execution_ready(
        runtime,
        &qwen.embed_weight,
        &qwen.layers,
        &qwen.final_norm,
        &qwen.lm_head,
        wp,
        qwen3_config_repr(qwen.config).geometry.head_dim,
        QWEN_BOUNDARY::forward_config_repr(),
    ));
}

} // verus!
