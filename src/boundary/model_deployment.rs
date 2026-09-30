//! Architecture-neutral checked assembly of a qualified model deployment.
//!
//! Family modules materialize checkpoint weights and a backend-qualified
//! runtime capability.  This module performs the shared permission binding
//! and proves the exact tuple admitted by `Engine`.

use crate::model_config::{ModelArchitecture, ModelConfig};
use crate::{types::*, proof::model::types::*, proof::tensor::types::*};
use crate::boundary::tensor_runtime as RT;
use vstd::prelude::*;

#[cfg(not(verus_only))]
use pyo3::prelude::*;

// @kernel-bridge-begin boundary::model_deployment::text_checkpoint_loader
/// Architecture-neutral host representation returned by every text loader.
/// Family adapters only interpret the ordered layer roles and metadata.
#[cfg(not(verus_only))]
pub(crate) struct LoadedTextCheckpoint {
    pub(crate) config_usize: std::collections::BTreeMap<String, usize>,
    pub(crate) config_f64_bits: std::collections::BTreeMap<String, u64>,
    pub(crate) config_optional_f64_bits: std::collections::BTreeMap<String, Option<u64>>,
    pub(crate) config_bool: std::collections::BTreeMap<String, bool>,
    pub(crate) model_config_sha256: String,
    pub(crate) embed_weight: RT::Tensor,
    pub(crate) layers: Vec<Vec<RT::Tensor>>,
    pub(crate) attention_kinds: Vec<String>,
    pub(crate) final_norm: RT::Tensor,
    pub(crate) lm_head: RT::Tensor,
}

/// Call a family loader and validate the common text-checkpoint protocol.
/// Numerical and family-specific role validation remains in the Python loader;
/// the caller interprets the exact ordered roles after this structural check.
#[cfg(not(verus_only))]
pub(crate) fn load_text_checkpoint_raw(
    loader_module: &str,
    expected_architecture: &str,
    expected_layer_roles: usize,
    config_usize_fields: &[&str],
    config_f64_fields: &[&str],
    config_optional_f64_fields: &[&str],
    config_bool_fields: &[&str],
    model_path: &str,
    device: &str,
) -> pyo3::PyResult<LoadedTextCheckpoint> {
    pyo3::Python::with_gil(|py| {
        let module = py.import_bound(loader_module)?;
        let kwargs = pyo3::types::PyDict::new_bound(py);
        kwargs.set_item("device", device)?;
        let loaded = module
            .getattr("load_text_weights")?
            .call((model_path,), Some(&kwargs))?
            .downcast_into::<pyo3::types::PyDict>()?;

        let architecture: String = loaded
            .get_item("architecture")?
            .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err("architecture"))?
            .extract()?;
        if architecture != expected_architecture {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "checkpoint loader returned {architecture:?}, expected {expected_architecture:?}",
            )));
        }

        let layers_py = loaded
            .get_item("layers")?
            .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err("layers"))?
            .downcast_into::<pyo3::types::PyList>()?;
        let attention_kinds: Vec<String> = loaded
            .get_item("attention_kinds")?
            .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err("attention_kinds"))?
            .extract()?;
        if layers_py.is_empty() || attention_kinds.len() != layers_py.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "checkpoint loader returned inconsistent layers",
            ));
        }
        let mut layers = Vec::with_capacity(layers_py.len());
        for item in layers_py.iter() {
            let tuple = item.downcast_into::<pyo3::types::PyTuple>()?;
            if tuple.len() != expected_layer_roles {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "checkpoint loader returned {} roles, expected {expected_layer_roles}",
                    tuple.len(),
                )));
            }
            layers.push(
                tuple
                    .iter()
                    .map(|role| RT::Tensor {
                        inner: role.unbind(),
                    })
                    .collect(),
            );
        }

        let config = loaded
            .get_item("config")?
            .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err("config"))?
            .downcast_into::<pyo3::types::PyDict>()?;
        let mut config_usize = std::collections::BTreeMap::new();
        for &field in config_usize_fields {
            let value = config
                .get_item(field)?
                .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err(field.to_string()))?
                .extract()?;
            config_usize.insert(field.to_string(), value);
        }
        let mut config_f64_bits = std::collections::BTreeMap::new();
        for &field in config_f64_fields {
            let value: f64 = config
                .get_item(field)?
                .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err(field.to_string()))?
                .extract()?;
            if !value.is_finite() || value <= 0.0 {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "checkpoint config field {field:?} must be finite and positive",
                )));
            }
            config_f64_bits.insert(field.to_string(), value.to_bits());
        }
        let mut config_optional_f64_bits = std::collections::BTreeMap::new();
        for &field in config_optional_f64_fields {
            // The key is mandatory; only its value may be None. This preserves
            // an explicitly disabled operator without inventing a parameter.
            let value: Option<f64> = config
                .get_item(field)?
                .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err(field.to_string()))?
                .extract()?;
            if value.is_some_and(|v| !v.is_finite() || v <= 0.0) {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "checkpoint config field {field:?} must be None or finite and positive",
                )));
            }
            config_optional_f64_bits.insert(field.to_string(), value.map(f64::to_bits));
        }
        let mut config_bool = std::collections::BTreeMap::new();
        for &field in config_bool_fields {
            let value = config
                .get_item(field)?
                .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err(field.to_string()))?
                .extract()?;
            config_bool.insert(field.to_string(), value);
        }

        let tensor = |name: &str| -> pyo3::PyResult<RT::Tensor> {
            Ok(RT::Tensor {
                inner: loaded
                    .get_item(name)?
                    .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err(name.to_string()))?
                    .unbind(),
            })
        };
        Ok(LoadedTextCheckpoint {
            config_usize,
            config_f64_bits,
            config_optional_f64_bits,
            config_bool,
            model_config_sha256: loaded
                .get_item("model_config_sha256")?
                .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err("model_config_sha256"))?
                .extract()?,
            embed_weight: tensor("embed_weight")?,
            layers,
            attention_kinds,
            final_norm: tensor("final_norm")?,
            lm_head: tensor("lm_head")?,
        })
    })
}
// @kernel-bridge-end boundary::model_deployment::text_checkpoint_loader

// @kernel-bridge-begin boundary::model_deployment::runtime_capability_host
/// Initialize a staged or backend-qualified Python runtime through the common
/// family runtime protocol. Concrete Rust runtime wrappers stay family-owned.
#[cfg(not(verus_only))]
pub(crate) fn init_runtime_capability_raw(
    runtime_module: &str,
    staged_profile_name: Option<&str>,
    deployment_bundle: Option<&str>,
    model_config_sha256: Option<&str>,
) -> pyo3::PyResult<RT::RuntimeCapabilityHandle> {
    pyo3::Python::with_gil(|py| {
        let module = py.import_bound(runtime_module)?;
        let config = match (staged_profile_name, deployment_bundle) {
            (None, Some(bundle)) => module.getattr("config_from_bundle")?.call1((bundle,))?,
            (Some(profile_name), None) => module
                .getattr("config_for_profile")?
                .call1((profile_name,))?,
            _ => {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "staged profile and deployment bundle must be exclusive",
                ))
            }
        };
        let inner = match (staged_profile_name, deployment_bundle, model_config_sha256) {
            (Some(_), None, None) => module.getattr("load_runtime")?.call1((config,))?.unbind(),
            (None, Some(bundle), Some(config_sha256)) => {
                let kwargs = pyo3::types::PyDict::new_bound(py);
                kwargs.set_item("deployment_bundle", bundle)?;
                kwargs.set_item("model_config_sha256", config_sha256)?;
                module
                    .getattr("load_qualified_runtime")?
                    .call((config,), Some(&kwargs))?
                    .unbind()
            }
            _ => {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "runtime binding arguments are not paired",
                ))
            }
        };
        Ok(RT::RuntimeCapabilityHandle { inner })
    })
}

#[cfg(not(verus_only))]
pub(crate) fn validate_runtime_raw(
    handle: &RT::RuntimeCapabilityHandle,
    expected_architecture: &str,
) -> pyo3::PyResult<String> {
    pyo3::Python::with_gil(|py| {
        let report = handle.inner.bind(py).call_method0("report")?;
        let architecture: String = report.get_item("architecture")?.extract()?;
        let backend_qualified: bool = report.get_item("backend_qualified")?.extract()?;
        let deployment_sha256: Option<String> = report.get_item("deployment_sha256")?.extract()?;
        if architecture != expected_architecture || !backend_qualified {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "runtime reported a different or unqualified architecture",
            ));
        }
        let deployment_sha256 = deployment_sha256.ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("qualified runtime has no deployment identity")
        })?;
        Ok(deployment_sha256)
    })
}

/// Check that the Python capability retained by Rust reports the same exact
/// model configuration that Rust records in `ModelFamilyRuntime`. Numeric
/// floating-point fields compare by IEEE-754 bits, not approximate equality.
#[cfg(not(verus_only))]
pub(crate) fn validate_runtime_model_config_raw(
    handle: &RT::RuntimeCapabilityHandle,
    expected_usize: &[(&str, usize)],
    expected_f64_bits: &[(&str, u64)],
    expected_bool: &[(&str, bool)],
    expected_string: &[(&str, &str)],
    expected_none: &[&str],
    expected_string_sequence: Option<(&str, &[String])>,
) -> pyo3::PyResult<()> {
    pyo3::Python::with_gil(|py| {
        let config = handle.inner.bind(py).call_method0("model_config")?;
        for &(field, expected) in expected_usize {
            let actual: usize = config.get_item(field)?.extract()?;
            if actual != expected {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "runtime model config field {field:?} is {actual}, expected {expected}",
                )));
            }
        }
        for &(field, expected_bits) in expected_f64_bits {
            let actual: f64 = config.get_item(field)?.extract()?;
            if actual.to_bits() != expected_bits {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "runtime model config field {field:?} has different float bits",
                )));
            }
        }
        for &(field, expected) in expected_bool {
            let actual: bool = config.get_item(field)?.extract()?;
            if actual != expected {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "runtime model config field {field:?} is {actual}, expected {expected}",
                )));
            }
        }
        for &(field, expected) in expected_string {
            let actual: String = config.get_item(field)?.extract()?;
            if actual != expected {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "runtime model config field {field:?} is {actual:?}, expected {expected:?}",
                )));
            }
        }
        for &field in expected_none {
            if !config.get_item(field)?.is_none() {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "runtime model config field {field:?} must be null",
                )));
            }
        }
        if let Some((field, expected)) = expected_string_sequence {
            let actual: Vec<String> = config.get_item(field)?.extract()?;
            if actual != expected {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "runtime model config field {field:?} has a different sequence",
                )));
            }
        }
        Ok(())
    })
}

#[cfg(not(verus_only))]
pub(crate) fn reports_backend_qualified_raw(
    handle: &RT::RuntimeCapabilityHandle,
    kernel_plan: &RT::QualifiedKernelPlan,
    expected_architecture: &str,
) -> pyo3::PyResult<bool> {
    pyo3::Python::with_gil(|py| {
        let report = handle.inner.bind(py).call_method0("report")?;
        let architecture: String = report.get_item("architecture")?.extract()?;
        let backend_qualified: bool = report.get_item("backend_qualified")?.extract()?;
        let deployment_sha256: Option<String> = report.get_item("deployment_sha256")?.extract()?;
        let expected = match &kernel_plan.qualification {
            RT::KernelPlanQualification::Staged => false,
            RT::KernelPlanQualification::BackendQualified => true,
        };
        let expected_sha256 = if kernel_plan.deployment_sha256.is_empty() {
            None
        } else {
            Some(kernel_plan.deployment_sha256.as_str())
        };
        if architecture != expected_architecture
            || deployment_sha256.as_deref() != expected_sha256
            || backend_qualified != expected
        {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "retained runtime disagrees with its admitted kernel plan",
            ));
        }
        Ok(backend_qualified)
    })
}
// @kernel-bridge-end boundary::model_deployment::runtime_capability_host

verus! {

pub open spec fn checkpoint_contents_valid(
    weights: &RT::ModelWeights,
    num_layers: usize,
    architecture: ModelArchitecture,
) -> bool {
    RT::model_weights_architecture(weights) == architecture
    && RT::model_weights_num_layers(weights) == num_layers
    && num_layers > 0
}

pub(crate) fn staged_kernel_plan(
    architecture: ModelArchitecture,
) -> (out: RT::QualifiedKernelPlan)
    ensures
        out.architecture == architecture,
        out.qualification == RT::KernelPlanQualification::Staged,
{
    RT::QualifiedKernelPlan {
        architecture,
        qualification: RT::KernelPlanQualification::Staged,
        plan_id: RT::KernelPlanId { word0: 0, word1: 0, word2: 0, word3: 0 },
        #[cfg(not(verus_only))]
        deployment_sha256: String::new(),
    }
}

pub(crate) fn backend_qualified_kernel_plan(
    architecture: ModelArchitecture,
    deployment_sha256: &str,
) -> (out: RT::QualifiedKernelPlan)
    ensures
        out.architecture == architecture,
        out.qualification == RT::KernelPlanQualification::BackendQualified,
{
    RT::QualifiedKernelPlan {
        architecture,
        qualification: RT::KernelPlanQualification::BackendQualified,
        plan_id: RT::deployment_sha256_plan_id(deployment_sha256),
        #[cfg(not(verus_only))]
        deployment_sha256: deployment_sha256.to_string(),
    }
}

pub fn assemble_qualified_model(
    weights: RT::ModelWeights,
    runtime: RT::ModelRuntime,
    num_layers: usize,
    architecture: ModelArchitecture,
) -> (out: (
    RT::ModelWeights,
    RT::ModelRuntime,
    Tracked<RT::ModelWeightsPerms>,
    ModelConfig,
))
    requires
        checkpoint_contents_valid(&weights, num_layers, architecture),
        RT::model_runtime_architecture(&runtime) == architecture,
        RT::model_runtime_kernel_plan_architecture(&runtime) == architecture,
        RT::model_runtime_kernel_plan_qualification(&runtime)
            == RT::KernelPlanQualification::BackendQualified,
        RT::model_runtime_deployment_config_repr(&runtime)
            == Some(RT::physical_model_deployment_config_repr(&weights)),
    ensures
        RT::model_execution_valid(&out.0, &out.1, &out.2@),
        RT::model_weights_num_layers(&out.0) == out.3.num_layers,
        RT::model_weights_repr_of(&out.2@).architecture
            == out.3.architecture,
        out.3.architecture == architecture,
{
    proof {
        reveal(checkpoint_contents_valid);
    }
    let weights_perms = RT::bind_model_weights_perms(&weights, num_layers);
    let model_config = ModelConfig { architecture, num_layers };
    proof {
        RT::lemma_runtime_execution_valid_from_bound_configuration(
            &weights, &runtime, &weights_perms@,
        );
        reveal(RT::model_execution_valid);
        reveal(RT::model_weights_bound);
        reveal(RT::physical_model_deployment_config_repr);
        RT::lemma_model_runtime_architecture_projection(&runtime);
        RT::lemma_model_runtime_kernel_plan_architecture_projection(&runtime);
        RT::lemma_model_runtime_kernel_plan_qualification_projection(&runtime);
        match (&weights, &runtime) {
            (RT::ModelWeights::Qwen3(_), RT::ModelRuntime::Qwen3(_)) => {
                reveal(RT::qwen3_model_weights_bound);
            },
            (RT::ModelWeights::Llama3(_), RT::ModelRuntime::Llama3(_)) => {
                reveal(RT::llama3_model_weights_bound);
            },
            (
                RT::ModelWeights::Gemma3Text(_),
                RT::ModelRuntime::Gemma3Text(_),
            ) => {
                reveal(RT::gemma3_model_weights_bound);
            },
            (RT::ModelWeights::Gemma4Text(_), RT::ModelRuntime::Gemma4Text(_)) => {},
            _ => {
                assert(false);
            },
        }
    }
    (weights, runtime, weights_perms, model_config)
}

} // verus!
