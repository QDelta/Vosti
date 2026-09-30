//! Llama 3 checkpoint identity and checked deployment assembly.

use crate::model_config::{DenseGeometry, FloatParameterBits, ModelArchitecture, ModelConfig};
use crate::boundary::model_families::llama3::config::Llama3Config;
use crate::boundary::model_deployment as COMMON_DEPLOYMENT;
use crate::{types::*, proof::model::types::*, proof::tensor::types::*};
use crate::boundary::tensor_runtime as RT;
use vstd::prelude::*;

verus! {

// @kernel-bridge-begin boundary::model_families::llama3::runtime_capability

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
            "vosti_kernels.model_families.llama3.runtime",
            staged_profile_name,
            deployment_bundle,
            model_config_sha256,
        )
        .expect("initialize source-attested Llama 3 runtime capability")
    }
    #[cfg(verus_only)]
    {
        unreachable!()
    }
}

#[verifier::external_body]
fn validate_runtime(
    handle: &RT::RuntimeCapabilityHandle,
    config: Llama3Config,
) -> (out: String) {
    #[cfg(not(verus_only))]
    {
        let deployment_sha256 =
            COMMON_DEPLOYMENT::validate_runtime_raw(handle, "llama3")
                .expect("validate qualified Llama 3 runtime");
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
                (
                    "rope_original_max_position_embeddings",
                    config.rope_original_max_position_embeddings,
                ),
                ("pretraining_tp", 1),
            ],
            &[
                ("rms_norm_eps", config.rms_norm_epsilon.bits),
                ("rope_theta", config.rope_theta.bits),
                ("rope_factor", config.rope_factor.bits),
                (
                    "rope_low_frequency_factor",
                    config.rope_low_frequency_factor.bits,
                ),
                (
                    "rope_high_frequency_factor",
                    config.rope_high_frequency_factor.bits,
                ),
                ("attention_dropout", 0.0f64.to_bits()),
            ],
            &[
                ("tie_word_embeddings", config.tie_word_embeddings),
                ("attention_bias", false),
                ("mlp_bias", false),
            ],
            &[
                ("model_type", "llama"),
                ("transformers_architecture", "LlamaForCausalLM"),
                ("hidden_act", "silu"),
                ("attention_kind", "full_attention"),
                ("rope_scaling_kind", "llama3"),
            ],
            &[],
            None,
        ).expect("Llama 3 runtime config differs from checkpoint config");
        deployment_sha256
    }
    #[cfg(verus_only)]
    {
        unreachable!()
    }
}

pub fn init_staged_runtime_for_tests(
    profile_name: &str,
) -> (out: RT::ModelFamilyRuntime)
    ensures
        RT::family_runtime_kernel_plan_architecture(&out)
            == ModelArchitecture::Llama3,
        RT::family_runtime_kernel_plan_qualification(&out)
            == RT::KernelPlanQualification::Staged,
{
    let handle = init_runtime_raw(Some(profile_name), None, None);
    let kernel_plan = COMMON_DEPLOYMENT::staged_kernel_plan(
        ModelArchitecture::Llama3,
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
    config: Llama3Config,
) -> (out: RT::ModelFamilyRuntime)
    ensures
        RT::family_runtime_kernel_plan_architecture(&out)
            == ModelArchitecture::Llama3,
        RT::family_runtime_kernel_plan_qualification(&out)
            == RT::KernelPlanQualification::BackendQualified,
        RT::family_runtime_deployment_config_repr(&out)
            == Some(ModelDeploymentConfigRepr::Llama3(llama3_config_repr(config))),
{
    let handle = init_runtime_raw(
        None, Some(deployment_bundle), Some(model_config_sha256),
    );
    let deployment_sha256 = validate_runtime(&handle, config);
    let kernel_plan = COMMON_DEPLOYMENT::backend_qualified_kernel_plan(
        ModelArchitecture::Llama3,
        &deployment_sha256,
    );
    let out = RT::ModelFamilyRuntime {
        handle,
        kernel_plan,
        model_config: RT::RuntimeModelConfig::Llama3(config),
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
            "llama3",
        )
        .expect("read Llama 3 kernel-plan qualification report")
    }
    #[cfg(verus_only)]
    {
        unreachable!()
    }
}

// @kernel-bridge-end boundary::model_families::llama3::runtime_capability

pub struct Llama3Checkpoint {
    weights: RT::ModelWeights,
    config: Llama3Config,
    model_config_sha256: String,
    num_layers: usize,
}

pub closed spec fn checkpoint_valid(checkpoint: &Llama3Checkpoint) -> bool {
    COMMON_DEPLOYMENT::checkpoint_contents_valid(
        &checkpoint.weights,
        checkpoint.num_layers,
        ModelArchitecture::Llama3,
    )
    && match &checkpoint.weights {
        RT::ModelWeights::Llama3(llama) => llama.config == checkpoint.config,
        _ => false,
    }
}

// @kernel-bridge-begin boundary::model_families::llama3::checkpoint_loader

#[verifier::external_body]
pub fn load_checkpoint(
    model_path: &str,
    device: &str,
) -> (out: Llama3Checkpoint)
    ensures checkpoint_valid(&out),
{
    #[cfg(not(verus_only))]
    {
        let loaded = COMMON_DEPLOYMENT::load_text_checkpoint_raw(
            "vosti_kernels.model_families.llama3.loader",
            "llama3",
            8,
            &[
                "vocab_size",
                "hidden_size",
                "intermediate_size",
                "num_hidden_layers",
                "num_attention_heads",
                "num_key_value_heads",
                "head_dim",
                "max_position_embeddings",
                "rope_original_max_position_embeddings",
            ],
            &[
                "rms_norm_eps",
                "rope_theta",
                "rope_factor",
                "rope_low_frequency_factor",
                "rope_high_frequency_factor",
            ],
            &[],
            &["tie_word_embeddings"],
            model_path,
            device,
        )
        .expect("load text-only Llama 3 checkpoint");
        assert!(
            loaded.attention_kinds.iter().all(|kind| kind == "full_attention"),
            "Llama 3 checkpoint loader returned non-full attention",
        );
        let layers = loaded.layers.into_iter().map(|roles| {
            let [
                input_norm,
                q_proj,
                k_proj,
                v_proj,
                o_proj,
                post_attn_norm,
                gate_up_proj,
                down_proj,
            ]: [RT::Tensor; 8] = roles.try_into().unwrap_or_else(|_| unreachable!());
            RT::Llama3LayerWeights {
                input_norm,
                q_proj,
                k_proj,
                v_proj,
                qk_norm: crate::boundary::dense_swiglu_decoder::DenseQkNormWeights::Disabled,
                o_proj,
                post_attn_norm,
                gate_up_proj,
                down_proj,
            }
        }).collect::<Vec<_>>();
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
        let config = Llama3Config {
            geometry,
            rms_norm_epsilon: FloatParameterBits {
                bits: loaded.config_f64_bits["rms_norm_eps"],
            },
            rope_theta: FloatParameterBits {
                bits: loaded.config_f64_bits["rope_theta"],
            },
            rope_factor: FloatParameterBits {
                bits: loaded.config_f64_bits["rope_factor"],
            },
            rope_low_frequency_factor: FloatParameterBits {
                bits: loaded.config_f64_bits["rope_low_frequency_factor"],
            },
            rope_high_frequency_factor: FloatParameterBits {
                bits: loaded.config_f64_bits["rope_high_frequency_factor"],
            },
            rope_original_max_position_embeddings:
                loaded.config_usize["rope_original_max_position_embeddings"],
            tie_word_embeddings: loaded.config_bool["tie_word_embeddings"],
        };
        Llama3Checkpoint {
            weights: RT::ModelWeights::Llama3(RT::Llama3ModelWeights {
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

// @kernel-bridge-end boundary::model_families::llama3::checkpoint_loader

pub fn qualify_checkpoint(
    checkpoint: Llama3Checkpoint,
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
        RT::model_weights_repr_of(&out.2@).architecture == out.3.architecture,
        out.3.architecture == ModelArchitecture::Llama3,
{
    let num_layers = checkpoint.num_layers;
    let config = checkpoint.config;
    let weights = checkpoint.weights;
    let llama_runtime = init_qualified_runtime(
        deployment_bundle, &checkpoint.model_config_sha256, config,
    );
    let runtime = RT::ModelRuntime::Llama3(llama_runtime);
    proof {
        reveal(checkpoint_valid);
        assert(RT::model_weights_num_layers(&weights) == num_layers);
        assert(RT::model_weights_architecture(&weights)
            == ModelArchitecture::Llama3);
        RT::lemma_model_runtime_architecture_projection(&runtime);
        RT::lemma_model_runtime_kernel_plan_architecture_projection(&runtime);
        RT::lemma_model_runtime_kernel_plan_qualification_projection(&runtime);
        RT::lemma_model_runtime_deployment_config_projection(&runtime);
        reveal(RT::physical_model_deployment_config_repr);
    }
    COMMON_DEPLOYMENT::assemble_qualified_model(
        weights, runtime, num_layers, ModelArchitecture::Llama3,
    )
}

} // verus!
