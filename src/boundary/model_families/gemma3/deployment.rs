//! Text-only Gemma3 checkpoint loading and checked deployment assembly.

use crate::model_config::{AttentionKind, DenseGeometry, FloatParameterBits, ModelArchitecture, ModelConfig};
use crate::boundary::model_families::gemma3::config::Gemma3Config;
use super::weights::Gemma3ModelWeights;
use crate::boundary::model_deployment as COMMON_DEPLOYMENT;
use crate::{types::*, proof::model::types::*, proof::tensor::types::*};
use crate::boundary::tensor_runtime as RT;
use vstd::prelude::*;

verus! {

// @kernel-bridge-begin boundary::model_families::gemma3::runtime_capability
#[verifier::external_body]
fn init_runtime_raw(
    staged_profile_name: Option<&str>,
    deployment_bundle: Option<&str>,
    model_config_sha256: Option<&str>,
) -> (out: RT::RuntimeCapabilityHandle) {
    #[cfg(not(verus_only))]
    {
        COMMON_DEPLOYMENT::init_runtime_capability_raw(
            "vosti_kernels.model_families.gemma3.runtime",
            staged_profile_name,
            deployment_bundle,
            model_config_sha256,
        )
        .expect("initialize source-attested Gemma 3 runtime capability")
    }
    #[cfg(verus_only)]
    {
        unreachable!()
    }
}

#[verifier::external_body]
fn validate_runtime(
    handle: &RT::RuntimeCapabilityHandle,
    config: Gemma3Config,
    layer_attention_kinds: &Vec<AttentionKind>,
) -> (out: String) {
    #[cfg(not(verus_only))]
    {
        let deployment_sha256 = COMMON_DEPLOYMENT::validate_runtime_raw(
            handle,
            "gemma3_text",
        )
        .expect("validate qualified Gemma 3 runtime");
        let layer_types = layer_attention_kinds.iter().map(|kind| match kind {
            AttentionKind::SlidingWindow => "sliding_attention".to_string(),
            AttentionKind::Full => "full_attention".to_string(),
        }).collect::<Vec<_>>();
        COMMON_DEPLOYMENT::validate_runtime_model_config_raw(
            handle,
            &[
                ("vocab_size", config.geometry.vocab_size),
                ("hidden_size", config.geometry.hidden_size),
                ("intermediate_size", config.geometry.intermediate_size),
                ("num_hidden_layers", config.geometry.num_layers),
                ("num_attention_heads", config.geometry.num_attention_heads),
                ("num_key_value_heads", config.geometry.num_key_value_heads),
                ("head_dim", config.geometry.head_dim),
                ("max_position_embeddings", config.geometry.max_position_embeddings),
                ("sliding_window", config.sliding_window),
            ],
            &[
                ("rms_norm_eps", config.rms_norm_epsilon.bits),
                ("query_pre_attn_scalar", config.query_pre_attention_scalar.bits),
                ("local_rope_theta", config.local_rope_theta.bits),
                ("global_rope_theta", config.global_rope_theta.bits),
                ("global_rope_factor", config.global_rope_factor.bits),
            ],
            &[],
            &[],
            &[],
            Some(("layer_types", &layer_types)),
        ).expect("Gemma 3 runtime config differs from checkpoint config");
        deployment_sha256
    }
    #[cfg(verus_only)]
    {
        unreachable!()
    }
}

// Explicitly unqualified capability for zero-layer/runtime plumbing tests. It
// cannot satisfy the neutral execution-validity gate.
pub fn init_staged_runtime_for_tests(profile_name: &str) -> (out: RT::ModelFamilyRuntime)
    ensures
        RT::family_runtime_kernel_plan_architecture(&out)
            == ModelArchitecture::Gemma3Text,
        RT::family_runtime_kernel_plan_qualification(&out)
            == RT::KernelPlanQualification::Staged,
{
    let handle = init_runtime_raw(Some(profile_name), None, None);
    let kernel_plan = COMMON_DEPLOYMENT::staged_kernel_plan(
        ModelArchitecture::Gemma3Text,
    );
    let out = RT::ModelFamilyRuntime {
        handle,
        kernel_plan,
        model_config: RT::RuntimeModelConfig::Staged,
    };
    proof {
        RT::lemma_family_runtime_kernel_plan_projection(&out);
        RT::lemma_family_runtime_deployment_config_projection(&out);
    }
    out
}

pub fn init_qualified_runtime(
    deployment_bundle: &str,
    model_config_sha256: &str,
    config: Gemma3Config,
    layer_attention_kinds: Vec<AttentionKind>,
) -> (out: RT::ModelFamilyRuntime)
    ensures
        RT::family_runtime_kernel_plan_architecture(&out)
            == ModelArchitecture::Gemma3Text,
        RT::family_runtime_kernel_plan_qualification(&out)
            == RT::KernelPlanQualification::BackendQualified,
        RT::family_runtime_deployment_config_repr(&out)
            == Some(ModelDeploymentConfigRepr::Gemma3Text(
                gemma3_deployment_config_repr(config, layer_attention_kinds@),
            )),
{
    let handle = init_runtime_raw(
        None, Some(deployment_bundle), Some(model_config_sha256),
    );
    let deployment_sha256 = validate_runtime(
        &handle, config, &layer_attention_kinds,
    );
    let kernel_plan = COMMON_DEPLOYMENT::backend_qualified_kernel_plan(
        ModelArchitecture::Gemma3Text,
        &deployment_sha256,
    );
    let out = RT::ModelFamilyRuntime {
        handle,
        kernel_plan,
        model_config: RT::RuntimeModelConfig::Gemma3Text(
            config, layer_attention_kinds,
        ),
    };
    proof {
        RT::lemma_family_runtime_kernel_plan_projection(&out);
        RT::lemma_family_runtime_deployment_config_projection(&out);
    }
    out
}

#[verifier::external_body]
pub fn reports_backend_qualified(runtime: &RT::ModelFamilyRuntime) -> (out: bool)
    ensures
        out == (RT::family_runtime_kernel_plan_qualification(runtime)
            == RT::KernelPlanQualification::BackendQualified),
{
    #[cfg(not(verus_only))]
    {
        COMMON_DEPLOYMENT::reports_backend_qualified_raw(
            &runtime.handle,
            &runtime.kernel_plan,
            "gemma3_text",
        )
        .expect("read Gemma 3 runtime qualification report")
    }
    #[cfg(verus_only)]
    {
        unreachable!()
    }
}
// @kernel-bridge-end boundary::model_families::gemma3::runtime_capability

pub struct Gemma3TextCheckpoint {
    weights: RT::ModelWeights,
    config: Gemma3Config,
    layer_attention_kinds: Vec<AttentionKind>,
    model_config_sha256: String,
    num_layers: usize,
}

pub closed spec fn checkpoint_valid(checkpoint: &Gemma3TextCheckpoint) -> bool {
    COMMON_DEPLOYMENT::checkpoint_contents_valid(
        &checkpoint.weights,
        checkpoint.num_layers,
        ModelArchitecture::Gemma3Text,
    )
    && match &checkpoint.weights {
        RT::ModelWeights::Gemma3Text(gemma) =>
            gemma.config == checkpoint.config
            && checkpoint.layer_attention_kinds@
                == RT::physical_model_layer_attention_kinds(
                    &checkpoint.weights,
                ),
        _ => false,
    }
}

// @kernel-bridge-begin boundary::model_families::gemma3::checkpoint_loader
// Load the text-only checkpoint through the exact Python role schema. This is
// a deployment boundary, not a numerical theorem: the Python loader validates
// the checkpoint key/shape/dtype inventory and this body preserves its object
// identities while materializing the closed Rust facade.
#[verifier::external_body]
pub fn load_checkpoint(
    model_path: &str,
    device: &str,
) -> (out: Gemma3TextCheckpoint)
    ensures checkpoint_valid(&out),
{
    #[cfg(not(verus_only))]
    {
        let loaded = COMMON_DEPLOYMENT::load_text_checkpoint_raw(
            "vosti_kernels.model_families.gemma3.loader",
            "gemma3_text",
            12,
            &[
                "vocab_size",
                "hidden_size",
                "intermediate_size",
                "num_hidden_layers",
                "num_attention_heads",
                "num_key_value_heads",
                "head_dim",
                "max_position_embeddings",
                "sliding_window",
            ],
            &[
                "rms_norm_eps",
                "query_pre_attn_scalar",
                "local_rope_theta",
                "global_rope_theta",
                "global_rope_factor",
            ],
            &[],
            &[],
            model_path,
            device,
        )
        .expect("load text-only Gemma 3 checkpoint");
        let layers = loaded
            .layers
            .into_iter()
            .zip(loaded.attention_kinds)
            .map(|(roles, attention_kind)|
                crate::boundary::four_norm_gated_weights::from_checkpoint_roles(
                    roles, &attention_kind, false,
                ))
            .collect::<pyo3::PyResult<Vec<_>>>()
            .expect("interpret Gemma 3 checkpoint roles");
        let num_layers = layers.len();
        let geometry = DenseGeometry {
            vocab_size: loaded.config_usize["vocab_size"],
            hidden_size: loaded.config_usize["hidden_size"],
            intermediate_size: loaded.config_usize["intermediate_size"],
            num_layers: loaded.config_usize["num_hidden_layers"],
            num_attention_heads: loaded.config_usize["num_attention_heads"],
            num_key_value_heads: loaded.config_usize["num_key_value_heads"],
            head_dim: loaded.config_usize["head_dim"],
            max_position_embeddings: loaded.config_usize["max_position_embeddings"],
        };
        let config = Gemma3Config {
            geometry,
            rms_norm_epsilon: FloatParameterBits {
                bits: loaded.config_f64_bits["rms_norm_eps"],
            },
            query_pre_attention_scalar: FloatParameterBits {
                bits: loaded.config_f64_bits["query_pre_attn_scalar"],
            },
            sliding_window: loaded.config_usize["sliding_window"],
            local_rope_theta: FloatParameterBits {
                bits: loaded.config_f64_bits["local_rope_theta"],
            },
            global_rope_theta: FloatParameterBits {
                bits: loaded.config_f64_bits["global_rope_theta"],
            },
            global_rope_factor: FloatParameterBits {
                bits: loaded.config_f64_bits["global_rope_factor"],
            },
        };
        let layer_attention_kinds = layers.iter()
            .map(|layer| layer.attention_kind)
            .collect::<Vec<_>>();
        Gemma3TextCheckpoint {
            weights: RT::ModelWeights::Gemma3Text(Gemma3ModelWeights {
                embed_weight: loaded.embed_weight,
                layers,
                final_norm: loaded.final_norm,
                lm_head: loaded.lm_head,
                config,
            }),
            config,
            layer_attention_kinds,
            model_config_sha256: loaded.model_config_sha256,
            num_layers,
        }
    }
    #[cfg(verus_only)]
    {
        unreachable!()
    }
}
// @kernel-bridge-end boundary::model_families::gemma3::checkpoint_loader

pub fn qualify_checkpoint(
    checkpoint: Gemma3TextCheckpoint,
    deployment_bundle: &str,
) -> (out: (
    RT::ModelWeights,
    RT::ModelRuntime,
    Tracked<RT::ModelWeightsPerms>,
    ModelConfig,
))
    requires checkpoint_valid(&checkpoint),
    ensures
        RT::model_execution_valid(&out.0, &out.1, &out.2@),
        RT::model_weights_num_layers(&out.0) == out.3.num_layers,
        RT::model_weights_repr_of(&out.2@).architecture
            == out.3.architecture,
        out.3.architecture == ModelArchitecture::Gemma3Text,
{
    let num_layers = checkpoint.num_layers;
    let config = checkpoint.config;
    let layer_attention_kinds = checkpoint.layer_attention_kinds;
    let ghost layer_attention_repr = layer_attention_kinds@;
    proof {
        reveal(checkpoint_valid);
        assert(layer_attention_repr
            == RT::physical_model_layer_attention_kinds(&checkpoint.weights));
    }
    let gemma_runtime = init_qualified_runtime(
        deployment_bundle, &checkpoint.model_config_sha256,
        config, layer_attention_kinds,
    );
    let weights = checkpoint.weights;
    let runtime = RT::ModelRuntime::Gemma3Text(gemma_runtime);
    proof {
        reveal(checkpoint_valid);
        assert(RT::model_weights_num_layers(&weights) == num_layers);
        assert(RT::model_weights_architecture(&weights)
            == ModelArchitecture::Gemma3Text);
        RT::lemma_model_runtime_architecture_projection(&runtime);
        RT::lemma_model_runtime_kernel_plan_architecture_projection(&runtime);
        RT::lemma_model_runtime_kernel_plan_qualification_projection(&runtime);
        RT::lemma_model_runtime_deployment_config_projection(&runtime);
        reveal(RT::physical_model_deployment_config_repr);
        assert(layer_attention_repr
            == RT::physical_model_layer_attention_kinds(&weights));
    }
    COMMON_DEPLOYMENT::assemble_qualified_model(
        weights,
        runtime,
        num_layers,
        ModelArchitecture::Gemma3Text,
    )
}

} // verus!
