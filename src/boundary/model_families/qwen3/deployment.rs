//! Qwen3 checkpoint identity and checked deployment assembly.

use crate::model_config::{DenseGeometry, FloatParameterBits, ModelArchitecture, ModelConfig};
use crate::boundary::model_families::qwen3::config::Qwen3Config;
use crate::boundary::model_deployment as COMMON_DEPLOYMENT;
use crate::{types::*, proof::model::types::*, proof::tensor::types::*};
use crate::boundary::tensor_runtime as RT;
use vstd::prelude::*;

verus! {

// @kernel-bridge-begin boundary::model_families::qwen3::runtime_capability

#[verifier::external_body]
fn init_runtime_raw(
    staged_profile_name: Option<&str>,
    deployment_bundle: Option<&str>,
    model_config_sha256: Option<&str>,
) -> (out: RT::RuntimeCapabilityHandle)
{
    #[cfg(not(verus_only))]
    {
        COMMON_DEPLOYMENT::init_runtime_capability_raw(
            "vosti_kernels.model_families.qwen3.runtime",
            staged_profile_name,
            deployment_bundle,
            model_config_sha256,
        )
        .expect("initialize source-attested Qwen runtime capability")
    }
    #[cfg(verus_only)]
    {
        unreachable!()
    }
}

#[verifier::external_body]
fn validate_runtime(
    handle: &RT::RuntimeCapabilityHandle,
    config: Qwen3Config,
) -> (out: String) {
    #[cfg(not(verus_only))]
    {
        let deployment_sha256 =
            COMMON_DEPLOYMENT::validate_runtime_raw(handle, "qwen3")
                .expect("validate qualified Qwen runtime");
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
            ],
            &[
                ("rms_norm_eps", config.rms_norm_epsilon.bits),
                ("rope_theta", config.rope_theta.bits),
                ("attention_dropout", 0.0f64.to_bits()),
            ],
            &[
                ("tie_word_embeddings", config.tie_word_embeddings),
                ("attention_bias", false),
                ("use_sliding_window", false),
            ],
            &[("model_type", "qwen3"), ("hidden_act", "silu")],
            &["sliding_window", "rope_scaling"],
            None,
        ).expect("Qwen runtime config differs from checkpoint config");
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
            == ModelArchitecture::Qwen3,
        RT::family_runtime_kernel_plan_qualification(&out)
            == RT::KernelPlanQualification::Staged,
{
    let handle = init_runtime_raw(Some(profile_name), None, None);
    let kernel_plan = COMMON_DEPLOYMENT::staged_kernel_plan(
        ModelArchitecture::Qwen3,
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
    config: Qwen3Config,
) -> (out: RT::ModelFamilyRuntime)
    ensures
        RT::family_runtime_kernel_plan_architecture(&out)
            == ModelArchitecture::Qwen3,
        RT::family_runtime_kernel_plan_qualification(&out)
            == RT::KernelPlanQualification::BackendQualified,
        RT::family_runtime_deployment_config_repr(&out)
            == Some(ModelDeploymentConfigRepr::Qwen3(qwen3_config_repr(config))),
{
    let handle = init_runtime_raw(
        None, Some(deployment_bundle), Some(model_config_sha256),
    );
    let deployment_sha256 = validate_runtime(&handle, config);
    let kernel_plan = COMMON_DEPLOYMENT::backend_qualified_kernel_plan(
        ModelArchitecture::Qwen3,
        &deployment_sha256,
    );
    let out = RT::ModelFamilyRuntime {
        handle,
        kernel_plan,
        model_config: RT::RuntimeModelConfig::Qwen3(config),
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
            "qwen3",
        )
        .expect("read Qwen kernel-plan qualification report")
    }
    #[cfg(verus_only)]
    {
        unreachable!()
    }
}
// @kernel-bridge-end boundary::model_families::qwen3::runtime_capability

// The Python Qwen loader remains the trusted checkpoint adapter. This value
// narrows its model-bearing output to the same family-local deployment protocol
// used by Gemma before anything is admitted to Engine.
pub struct Qwen3Checkpoint {
    weights: RT::ModelWeights,
    config: Qwen3Config,
    model_config_sha256: String,
    num_layers: usize,
}

pub closed spec fn checkpoint_valid(checkpoint: &Qwen3Checkpoint) -> bool {
    COMMON_DEPLOYMENT::checkpoint_contents_valid(
        &checkpoint.weights,
        checkpoint.num_layers,
        ModelArchitecture::Qwen3,
    )
    && match &checkpoint.weights {
        RT::ModelWeights::Qwen3(qwen) => qwen.config == checkpoint.config,
        _ => false,
    }
}

// @kernel-bridge-begin boundary::model_families::qwen3::checkpoint_loader
// Preserve the exact Python role order while materializing the closed Qwen
// weight facade. Runtime qualification remains a separate second phase.
#[verifier::external_body]
pub fn load_checkpoint(
    model_path: &str,
    device: &str,
) -> (out: Qwen3Checkpoint)
    ensures checkpoint_valid(&out),
{
    #[cfg(not(verus_only))]
    {
        let loaded = COMMON_DEPLOYMENT::load_text_checkpoint_raw(
            "vosti_kernels.model_families.qwen3.loader",
            "qwen3",
            10,
            &[
                "vocab_size",
                "hidden_size",
                "intermediate_size",
                "num_hidden_layers",
                "num_attention_heads",
                "num_key_value_heads",
                "head_dim",
                "max_position_embeddings",
            ],
            &["rms_norm_eps", "rope_theta"],
            &[],
            &["tie_word_embeddings"],
            model_path,
            device,
        )
        .expect("load text-only Qwen checkpoint");
        assert!(
            loaded.attention_kinds.iter().all(|kind| kind == "full_attention"),
            "Qwen checkpoint loader returned non-full attention",
        );
        let layers = loaded
            .layers
            .into_iter()
            .map(|roles| {
                let [
                    input_norm,
                    q_proj,
                    k_proj,
                    v_proj,
                    q_norm,
                    k_norm,
                    o_proj,
                    post_attn_norm,
                    gate_up_proj,
                    down_proj,
                ]: [RT::Tensor; 10] = roles.try_into().unwrap_or_else(|_| unreachable!());
                RT::Qwen3LayerWeights {
                    input_norm,
                    q_proj,
                    k_proj,
                    v_proj,
                    qk_norm: crate::boundary::dense_swiglu_decoder::DenseQkNormWeights::RmsNorm {
                        q_weight: q_norm,
                        k_weight: k_norm,
                    },
                    o_proj,
                    post_attn_norm,
                    gate_up_proj,
                    down_proj,
                }
            })
            .collect::<Vec<_>>();
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
        let config = Qwen3Config {
            geometry,
            rms_norm_epsilon: FloatParameterBits {
                bits: loaded.config_f64_bits["rms_norm_eps"],
            },
            rope_theta: FloatParameterBits {
                bits: loaded.config_f64_bits["rope_theta"],
            },
            tie_word_embeddings: loaded.config_bool["tie_word_embeddings"],
        };
        Qwen3Checkpoint {
            weights: RT::ModelWeights::Qwen3(RT::Qwen3ModelWeights {
                embed_weight: loaded.embed_weight,
                layers,
                final_norm: loaded.final_norm,
                lm_head: loaded.lm_head,
                config,
            }),
            config,
            model_config_sha256: loaded.model_config_sha256,
            num_layers,
        }
    }
    #[cfg(verus_only)]
    {
        unreachable!()
    }
}
// @kernel-bridge-end boundary::model_families::qwen3::checkpoint_loader

pub fn qualify_checkpoint(
    checkpoint: Qwen3Checkpoint,
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
        out.3.architecture == ModelArchitecture::Qwen3,
{
    let num_layers = checkpoint.num_layers;
    let config = checkpoint.config;
    let weights = checkpoint.weights;
    let qwen_runtime = init_qualified_runtime(
        deployment_bundle, &checkpoint.model_config_sha256, config,
    );
    let runtime = RT::ModelRuntime::Qwen3(qwen_runtime);
    proof {
        reveal(checkpoint_valid);
        assert(RT::model_weights_num_layers(&weights) == num_layers);
        assert(RT::model_weights_architecture(&weights)
            == ModelArchitecture::Qwen3);
        RT::lemma_model_runtime_architecture_projection(&runtime);
        RT::lemma_model_runtime_kernel_plan_architecture_projection(&runtime);
        RT::lemma_model_runtime_kernel_plan_qualification_projection(&runtime);
        RT::lemma_model_runtime_deployment_config_projection(&runtime);
        reveal(RT::physical_model_deployment_config_repr);
    }
    COMMON_DEPLOYMENT::assemble_qualified_model(
        weights,
        runtime,
        num_layers,
        ModelArchitecture::Qwen3,
    )
}

} // verus!
