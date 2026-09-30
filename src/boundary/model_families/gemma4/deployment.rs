//! Dense text checkpoint loading and checked qualified-runtime assembly.

use crate::model_config::{AttentionKind, DenseGeometry, FloatParameterBits, ModelArchitecture, ModelConfig};
use crate::boundary::model_families::gemma4::config::{Gemma4Config, gemma4_attention_geometry};
use super::weights::{self, Gemma4ModelWeights, Gemma4ModelWeightsPerms};
use crate::boundary::model_deployment as COMMON;
use crate::{types::*, proof::model::types::*, proof::tensor::types::*};
use crate::boundary::tensor_runtime as RT;
use vstd::prelude::*;

verus! {

// @kernel-bridge-begin boundary::model_families::gemma4::runtime_capability
#[verifier::external_body]
fn init_runtime_raw(
    staged_profile_name: Option<&str>, deployment_bundle: Option<&str>,
    model_config_sha256: Option<&str>,
) -> (out: RT::RuntimeCapabilityHandle) {
    #[cfg(not(verus_only))]
    {
        COMMON::init_runtime_capability_raw(
            "vosti_kernels.model_families.gemma4.runtime", staged_profile_name,
            deployment_bundle, model_config_sha256,
        ).expect("initialize source-attested Gemma 4 runtime capability")
    }
    #[cfg(verus_only)]
    { unreachable!() }
}

#[verifier::external_body]
fn validate_runtime(
    handle: &RT::RuntimeCapabilityHandle, config: Gemma4Config,
    layer_attention_kinds: &Vec<AttentionKind>,
) -> (out: String) {
    #[cfg(not(verus_only))]
    {
        let deployment_sha256 = COMMON::validate_runtime_raw(handle, "gemma4_text")
            .expect("validate qualified Gemma 4 runtime");
        let layer_types = layer_attention_kinds.iter().map(|kind| match kind {
            AttentionKind::SlidingWindow => "sliding_attention".to_string(),
            AttentionKind::Full => "full_attention".to_string(),
        }).collect::<Vec<_>>();
        let mut floats = vec![
            ("rms_norm_eps", config.rms_norm_epsilon.bits),
            ("local_rope_theta", config.local_rope_theta.bits),
            ("global_rope_theta", config.global_rope_theta.bits),
            ("global_rope_factor", config.global_rope_factor.bits),
            ("global_partial_rotary_factor", config.global_partial_rotary_factor.bits),
        ];
        let mut absent = Vec::new();
        match config.final_logit_softcap {
            Some(value) => floats.push(("final_logit_softcapping", value.bits)),
            None => absent.push("final_logit_softcapping"),
        }
        COMMON::validate_runtime_model_config_raw(handle, &[
            ("vocab_size", config.geometry.vocab_size),
            ("hidden_size", config.geometry.hidden_size),
            ("intermediate_size", config.geometry.intermediate_size),
            ("num_hidden_layers", config.geometry.num_layers),
            ("num_attention_heads", config.geometry.num_attention_heads),
            ("num_key_value_heads", config.geometry.num_key_value_heads),
            ("num_global_key_value_heads", config.num_global_key_value_heads),
            ("head_dim", config.geometry.head_dim),
            ("global_head_dim", config.global_head_dim),
            ("max_position_embeddings", config.geometry.max_position_embeddings),
            ("sliding_window", config.sliding_window),
        ], &floats, &[("attention_k_eq_v", config.attention_k_eq_v)], &[], &absent,
            Some(("layer_types", &layer_types)))
            .expect("Gemma 4 runtime config differs from checkpoint config");
        deployment_sha256
    }
    #[cfg(verus_only)]
    { unreachable!() }
}

pub fn init_staged_runtime_for_tests(profile_name: &str) -> (out: RT::ModelFamilyRuntime)
    ensures
        RT::family_runtime_kernel_plan_architecture(&out) == ModelArchitecture::Gemma4Text,
        RT::family_runtime_kernel_plan_qualification(&out) == RT::KernelPlanQualification::Staged,
{
    let handle = init_runtime_raw(Some(profile_name), None, None);
    let kernel_plan = COMMON::staged_kernel_plan(ModelArchitecture::Gemma4Text);
    let out = RT::ModelFamilyRuntime {
        handle, kernel_plan, model_config: RT::RuntimeModelConfig::Staged,
    };
    proof { RT::lemma_family_runtime_kernel_plan_projection(&out); }
    out
}

// The shared Python loader verifies the sealed deployment and checkpoint hash;
// the host validator then binds every retained parameter to the Rust config.
// This constructs a primitive capability, not an Engine/full-forward theorem.
pub fn init_qualified_runtime(
    deployment_bundle: &str, model_config_sha256: &str, config: Gemma4Config,
    layer_attention_kinds: Vec<AttentionKind>,
) -> (out: RT::ModelFamilyRuntime)
    ensures
        RT::family_runtime_kernel_plan_architecture(&out) == ModelArchitecture::Gemma4Text,
        RT::family_runtime_kernel_plan_qualification(&out) == RT::KernelPlanQualification::BackendQualified,
        RT::family_runtime_deployment_config_repr(&out) == Some(
            ModelDeploymentConfigRepr::Gemma4Text(
                gemma4_deployment_config_repr(config, layer_attention_kinds@))),
{
    let handle = init_runtime_raw(None, Some(deployment_bundle), Some(model_config_sha256));
    let deployment_sha256 = validate_runtime(&handle, config, &layer_attention_kinds);
    let kernel_plan = COMMON::backend_qualified_kernel_plan(
        ModelArchitecture::Gemma4Text, &deployment_sha256);
    let out = RT::ModelFamilyRuntime {
        handle, kernel_plan,
        model_config: RT::RuntimeModelConfig::Gemma4Text(config, layer_attention_kinds),
    };
    proof {
        RT::lemma_family_runtime_kernel_plan_projection(&out);
        RT::lemma_family_runtime_deployment_config_projection(&out);
    }
    out
}

#[verifier::external_body]
pub fn reports_backend_qualified(runtime: &RT::ModelFamilyRuntime) -> (out: bool)
    ensures out == (RT::family_runtime_kernel_plan_qualification(runtime)
        == RT::KernelPlanQualification::BackendQualified),
{
    #[cfg(not(verus_only))]
    {
        COMMON::reports_backend_qualified_raw(&runtime.handle, &runtime.kernel_plan, "gemma4_text")
            .expect("read Gemma 4 runtime qualification report")
    }
    #[cfg(verus_only)]
    { unreachable!() }
}
// @kernel-bridge-end boundary::model_families::gemma4::runtime_capability

pub struct Gemma4TextCheckpoint {
    pub weights: Gemma4ModelWeights,
    pub model_config_sha256: String,
}

pub closed spec fn checkpoint_valid(checkpoint: &Gemma4TextCheckpoint) -> bool {
    COMMON::checkpoint_contents_valid(
        &RT::ModelWeights::Gemma4Text(checkpoint.weights),
        checkpoint.weights.config.geometry.num_layers,
        ModelArchitecture::Gemma4Text,
    )
}

// @kernel-bridge-begin boundary::model_families::gemma4::checkpoint_loader
// The Python loader validates keys/shapes/dtypes and preserves raw K/V sharing.
// This trusted host adapter preserves all roles and exact parameter bits; it
// does not assert a model-forward result or qualify a runtime capability.
#[verifier::external_body]
pub fn load_checkpoint(model_path: &str, device: &str) -> (out: Gemma4TextCheckpoint)
    ensures checkpoint_valid(&out),
{
    #[cfg(not(verus_only))]
    {
        let loaded = COMMON::load_text_checkpoint_raw(
            "vosti_kernels.model_families.gemma4.loader", "gemma4_text", 13,
            &["vocab_size", "hidden_size", "intermediate_size", "num_hidden_layers",
              "num_attention_heads", "num_key_value_heads", "num_global_key_value_heads",
              "head_dim", "global_head_dim", "max_position_embeddings", "sliding_window"],
            &["rms_norm_eps", "local_rope_theta", "global_rope_theta", "global_rope_factor",
              "global_partial_rotary_factor"],
            &["final_logit_softcapping"], &["attention_k_eq_v"], model_path, device,
        ).expect("load text-only Gemma 4 checkpoint");
        let layers = loaded.layers.into_iter().zip(loaded.attention_kinds)
            .map(|(roles, kind)| crate::boundary::four_norm_gated_weights::from_checkpoint_roles(
                roles, &kind, true,
            )).collect::<pyo3::PyResult<Vec<_>>>().expect("interpret Gemma 4 checkpoint roles");
        let config = Gemma4Config {
            geometry: DenseGeometry {
                vocab_size: loaded.config_usize["vocab_size"],
                hidden_size: loaded.config_usize["hidden_size"],
                intermediate_size: loaded.config_usize["intermediate_size"],
                num_layers: loaded.config_usize["num_hidden_layers"],
                num_attention_heads: loaded.config_usize["num_attention_heads"],
                num_key_value_heads: loaded.config_usize["num_key_value_heads"],
                head_dim: loaded.config_usize["head_dim"],
                max_position_embeddings: loaded.config_usize["max_position_embeddings"],
            },
            num_global_key_value_heads: loaded.config_usize["num_global_key_value_heads"],
            global_head_dim: loaded.config_usize["global_head_dim"],
            sliding_window: loaded.config_usize["sliding_window"],
            rms_norm_epsilon: FloatParameterBits { bits: loaded.config_f64_bits["rms_norm_eps"] },
            local_rope_theta: FloatParameterBits { bits: loaded.config_f64_bits["local_rope_theta"] },
            global_rope_theta: FloatParameterBits { bits: loaded.config_f64_bits["global_rope_theta"] },
            global_rope_factor: FloatParameterBits { bits: loaded.config_f64_bits["global_rope_factor"] },
            global_partial_rotary_factor: FloatParameterBits {
                bits: loaded.config_f64_bits["global_partial_rotary_factor"],
            },
            attention_k_eq_v: loaded.config_bool["attention_k_eq_v"],
            final_logit_softcap: loaded.config_optional_f64_bits["final_logit_softcapping"]
                .map(|bits| FloatParameterBits { bits }),
        };
        assert!(!layers.is_empty() && layers.len() == config.geometry.num_layers,
            "Gemma 4 checkpoint layer count differs from configuration");
        Gemma4TextCheckpoint {
            weights: Gemma4ModelWeights {
                embed_weight: loaded.embed_weight, layers, final_norm: loaded.final_norm,
                lm_head: loaded.lm_head, config,
            },
            model_config_sha256: loaded.model_config_sha256,
        }
    }
    #[cfg(verus_only)]
    { unreachable!() }
}
// @kernel-bridge-end boundary::model_families::gemma4::checkpoint_loader

// Checked composition of the loading and immutable-weight permission boundary.
// A later qualified deployment must additionally bind its kernels/runtime/KV.
pub fn bind_checkpoint(checkpoint: Gemma4TextCheckpoint)
    -> (out: (Gemma4ModelWeights, Tracked<Gemma4ModelWeightsPerms>, String))
    requires checkpoint_valid(&checkpoint),
    ensures
        weights::model_weights_bound(&out.0, &out.1@),
        out.0.config == checkpoint.weights.config,
        out.0.layers.len() == checkpoint.weights.layers.len(),
        out.2 == checkpoint.model_config_sha256,
{
    proof { reveal(checkpoint_valid); }
    let num_layers = checkpoint.weights.layers.len();
    let perms = weights::bind_model_weights_perms(&checkpoint.weights, num_layers);
    (checkpoint.weights, perms, checkpoint.model_config_sha256)
}

// Use the same checked final assembly as the existing model families. The
// bundle loader checks the deployment/checkpoint identity before permissions
// and the executable model configuration are admitted together.
pub fn qualify_checkpoint(checkpoint: Gemma4TextCheckpoint, deployment_bundle: &str)
    -> (out: (RT::ModelWeights, RT::ModelRuntime, Tracked<RT::ModelWeightsPerms>, ModelConfig))
    requires checkpoint_valid(&checkpoint),
    ensures
        RT::model_execution_valid(&out.0, &out.1, &out.2@),
        RT::model_weights_num_layers(&out.0) == out.3.num_layers,
        RT::model_weights_repr_of(&out.2@).architecture == out.3.architecture,
        out.3.architecture == ModelArchitecture::Gemma4Text,
{
    proof { reveal(checkpoint_valid); }
    let config = checkpoint.weights.config;
    let n = checkpoint.weights.layers.len();
    let mut kinds: Vec<AttentionKind> = Vec::new();
    let mut i = 0usize;
    while i < n
        invariant
            i <= n, n == checkpoint.weights.layers.len(), kinds.len() == i,
            forall|j: int| 0 <= j < i ==> #[trigger] kinds[j]
                == checkpoint.weights.layers[j].attention_kind,
        decreases n - i,
    {
        kinds.push(checkpoint.weights.layers[i].attention_kind);
        i += 1;
    }
    let ghost attention_kinds = kinds@;
    proof { assert(attention_kinds =~= weights::physical_attention_kinds(&checkpoint.weights)); }
    let family_runtime = init_qualified_runtime(deployment_bundle,
        &checkpoint.model_config_sha256, config, kinds);
    let weights = RT::ModelWeights::Gemma4Text(checkpoint.weights);
    let runtime = RT::ModelRuntime::Gemma4Text(family_runtime);
    proof {
        RT::lemma_model_runtime_architecture_projection(&runtime);
        RT::lemma_model_runtime_kernel_plan_architecture_projection(&runtime);
        RT::lemma_model_runtime_kernel_plan_qualification_projection(&runtime);
        RT::lemma_model_runtime_deployment_config_projection(&runtime);
    }
    COMMON::assemble_qualified_model(weights, runtime, n, ModelArchitecture::Gemma4Text)
}

// Full-retention caches use the existing page/ownership contract. Head counts
// and widths remain layer-local physical geometry; SWA eviction is not added.
pub fn init_model_kv_caches(weights: &Gemma4ModelWeights, token_capacity: usize)
    -> (out: (Vec<(RT::Tensor, RT::Tensor)>, Tracked<RT::KVCachePerms>))
    requires weights.layers.len() > 0,
    ensures ({ let (caches, perms) = out;
        &&& caches.len() == weights.layers.len()
        &&& perms@.len() == weights.layers.len()
        &&& RT::kv_perms_ids_distinct(perms@)
        &&& RT::kv_perms_initial_shape(perms@, token_capacity as nat)
        &&& perms@.extracted() == Set::<int>::empty()
        &&& RT::kv_cache_tensor_ids_match(caches@, perms@, weights.layers.len() as nat)
    }),
{
    let mut projections: Vec<&RT::Tensor> = Vec::new();
    let mut dimensions: Vec<usize> = Vec::new();
    let mut i = 0usize;
    while i < weights.layers.len()
        invariant
            i <= weights.layers.len(), projections.len() == i, dimensions.len() == i,
        decreases weights.layers.len() - i,
    {
        let layer = &weights.layers[i];
        let geometry = gemma4_attention_geometry(weights.config, layer.attention_kind);
        projections.push(&layer.k_proj);
        dimensions.push(geometry.head_dim);
        i += 1;
    }
    RT::init_layer_model_kv_caches(&weights.embed_weight, &projections, &dimensions, token_capacity)
}

} // verus!
