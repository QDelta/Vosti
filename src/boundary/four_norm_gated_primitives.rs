//! Shared exact-effect primitives for four-norm gated decoders.
//! Family runtime admission supplies the immutable numerical policy. No
//! full-forward effect or cross-kernel numerical equivalence is assumed here.

use crate::model_config::{AttentionKind, FloatParameterBits};
use crate::{types::*, proof::model::types::*, proof::tensor::types::*};
use crate::boundary::tensor_runtime as RT;
#[cfg(verus_only)]
use crate::proof::model::four_norm_gated::model as MODEL;
#[cfg(verus_only)]
use crate::proof::model::four_norm_gated::layers as ROWS;
#[cfg(verus_only)]
use crate::boundary::scalar::{float_parameter_scalar_repr, positive_float_parameter_valid};
use vstd::prelude::*;
#[cfg(verus_only)]
use crate::boundary::attention_operator as ATTN;
#[cfg(verus_only)]
use crate::boundary::normalization_operator as NORM;
#[cfg(verus_only)]
use crate::boundary::pointwise_operator as PW;
#[cfg(verus_only)]
use crate::boundary::embedding_operator as EMBED;
#[cfg(verus_only)]
use crate::boundary::head_normalization_operator as HEAD;
#[cfg(verus_only)]
use crate::boundary::backend_certificates::{attention as RAW_ATTENTION, support as KERNEL_SUPPORT};

#[cfg(not(verus_only))]
use pyo3::prelude::*;

// @kernel-bridge-begin boundary::four_norm_gated_primitives::runtime
#[cfg(not(verus_only))]
pub(crate) fn attention_kind_name(kind: AttentionKind) -> &'static str {
    match kind {
        AttentionKind::SlidingWindow => "sliding_attention",
        AttentionKind::Full => "full_attention",
    }
}

#[cfg(not(verus_only))]
fn rms_norm_site_name(site: RmsNormSite) -> &'static str {
    match site {
        RmsNormSite::Input => "input_norm",
        RmsNormSite::PostAttention => "post_attention_norm",
        RmsNormSite::PreFeedforward => "pre_feedforward_norm",
        RmsNormSite::PostFeedforward => "post_feedforward_norm",
        RmsNormSite::Final => "final_norm",
    }
}

#[cfg(not(verus_only))]
fn add_site_name(site: AddSite) -> &'static str {
    match site {
        AddSite::AttentionResidual => "attention_residual_add",
        AddSite::FeedforwardResidual => "feedforward_residual_add",
    }
}


verus! {

#[derive(Clone, Copy)]
pub enum RmsNormSite {
    Input,
    PostAttention,
    PreFeedforward,
    PostFeedforward,
    Final,
}

#[derive(Clone, Copy)]
pub enum AddSite {
    AttentionResidual,
    FeedforwardResidual,
}


// One model-static numerical policy drives every four-norm primitive. This
// ghost record describes operator identity, never launch tiles or scheduling.
pub struct RuntimePolicyRepr {
    pub hidden_size: nat,
    pub local_geometry: AttentionGeometryRepr,
    pub global_geometry: AttentionGeometryRepr,
    pub norm: NormPolicyRepr,
    pub qk_norm: NormPolicyRepr,
    pub local_rotary: RotaryConfigRepr,
    pub global_rotary: RotaryConfigRepr,
    pub attention_scale: AttentionScaleRepr,
    pub sliding_window: nat,
    pub value_norm_epsilon: Option<FloatParameterBits>,
    pub layer_scale: bool,
    pub logits_softcap: Option<FloatParameterBits>,
}

pub open spec fn runtime_policy(runtime: &RT::ModelFamilyRuntime) -> Option<RuntimePolicyRepr> {
    match RT::family_runtime_deployment_config_repr(runtime) {
        Some(ModelDeploymentConfigRepr::Gemma3Text(config)) => Some(RuntimePolicyRepr {
            hidden_size: config.geometry.hidden_size,
            local_geometry: attention_geometry_repr(config.geometry),
            global_geometry: attention_geometry_repr(config.geometry),
            norm: NormPolicyRepr::UnitOffset,
            qk_norm: NormPolicyRepr::UnitOffset,
            local_rotary: config.local_rotary,
            global_rotary: config.global_rotary,
            attention_scale: config.attention_scale,
            sliding_window: config.sliding_window,
            value_norm_epsilon: None,
            layer_scale: false,
            logits_softcap: None,
        }),
        Some(ModelDeploymentConfigRepr::Gemma4Text(config)) => Some(RuntimePolicyRepr {
                hidden_size: config.geometry.hidden_size,
                local_geometry: attention_geometry_repr(config.geometry),
                global_geometry: config.global_attention_geometry,
                norm: NormPolicyRepr::Direct(config.rms_norm.epsilon),
                qk_norm: NormPolicyRepr::Direct(config.rms_norm.epsilon),
                local_rotary: config.local_rotary,
                global_rotary: config.global_rotary,
                attention_scale: unit_attention_scale_repr(),
                sliding_window: config.sliding_window,
                value_norm_epsilon: Some(config.rms_norm.epsilon),
                layer_scale: true,
                logits_softcap: config.final_logit_softcap,
        }),
        _ => None,
    }
}

// A projected policy alone is not admission: an exact backend-qualified plan
// is still required by configuration_valid and the caller's engine gate.
pub open spec fn runtime_norm_policy(runtime: &RT::ModelFamilyRuntime) -> Option<NormPolicyRepr> {
    match runtime_policy(runtime) {
        Some(policy) => Some(policy.norm),
        None => None,
    }
}

pub open spec fn configuration_valid(runtime: &RT::ModelFamilyRuntime) -> bool {
    RT::family_runtime_execution_valid(runtime) && runtime_norm_policy(runtime).is_some()
}

pub open spec fn hidden_size_matches(runtime: &RT::ModelFamilyRuntime, hidden_size: nat) -> bool {
    configuration_valid(runtime) && match runtime_policy(runtime) {
        Some(policy) => policy.hidden_size == hidden_size,
        None => false,
    }
}

pub open spec fn norm_repr(input: Tensor2D, weight: Tensor1D, policy: NormPolicyRepr) -> Tensor2D {
    Seq::new(input.len(), |i: int| ROWS::norm_row(input[i], weight, policy))
}

pub open spec fn norm_raw_output(input: Tensor2D, weight: Tensor1D, policy: NormPolicyRepr) -> Tensor2D {
    match policy {
        NormPolicyRepr::UnitOffset => NORM::offset_raw_output(input, weight,
            float_parameter_scalar_repr(unit_offset_norm_epsilon_repr())),
        NormPolicyRepr::Direct(epsilon) => NORM::rms_raw_output(input, weight,
            float_parameter_scalar_repr(epsilon)),
    }
}

pub proof fn checked_norm_binding(input: Tensor2D, weight: Tensor1D, policy: NormPolicyRepr)
    requires NORM::layout(input, weight),
    ensures norm_raw_output(input, weight, policy) == norm_repr(input, weight, policy),
{
    match policy {
        NormPolicyRepr::UnitOffset => {
            let eps = float_parameter_scalar_repr(unit_offset_norm_epsilon_repr());
            NORM::checked_offset_binding(input, weight, eps);
            reveal(RT::offset_rms_norm_kernel_cell_repr);
            assert forall|r: int| 0 <= r < input.len() implies
                (#[trigger] norm_repr(input, weight, policy)[r]) == NORM::offset_output(input, weight, eps)[r] by {
                assert(ROWS::norm_row(input[r], weight, policy) =~= NORM::offset_row_output(input[r], weight, eps));
            };
            assert(norm_repr(input, weight, policy) =~= NORM::offset_output(input, weight, eps));
        },
        NormPolicyRepr::Direct(epsilon) => {
            let eps = float_parameter_scalar_repr(epsilon);
            NORM::checked_rms_binding(input, weight, eps);
            reveal(RT::rms_norm_kernel_cell_repr);
            assert forall|r: int| 0 <= r < input.len() implies
                (#[trigger] norm_repr(input, weight, policy)[r]) == NORM::rms_output(input, weight, eps)[r] by {
                assert(ROWS::norm_row(input[r], weight, policy) =~= NORM::rms_row_output(input[r], weight, eps));
            };
            assert(norm_repr(input, weight, policy) =~= NORM::rms_output(input, weight, eps));
        },
    }
}

pub fn scaled_embed(
    runtime: &RT::ModelFamilyRuntime,
    input_ids: &RT::Tensor,
    weight: &RT::Tensor,
    Tracked(ip): Tracked<&RT::TensorPerm>,
    Tracked(wp): Tracked<&RT::TensorPerm>,
    hidden_size: usize,
    Ghost(input_ids_repr): Ghost<IntTensor1D>,
    Ghost(weight_repr): Ghost<Tensor2D>,
    Ghost(scope): Ghost<Set<RT::TensorId>>,
) -> (out: (RT::Tensor, Tracked<RT::TensorPerm>))
    requires
        hidden_size_matches(runtime, hidden_size as nat),
        RT::int_tensor_repr_1d(*ip, *input_ids, input_ids_repr),
        RT::tensor_repr_2d(*wp, *weight, weight_repr),
        hidden_size > 0,
    ensures ({ let (tensor, perm) = out;
        &&& tensor.id() != input_ids.id()
        &&& tensor.id() != weight.id()
        &&& !scope.contains(tensor.id())
        &&& RT::tensor_repr_2d(
            perm@, tensor,
            MODEL::scaled_embed_repr(
                input_ids_repr, weight_repr, hidden_size as nat,
            ),
        )
    }),
{
    let out = scaled_embed_raw(runtime, input_ids, weight, Tracked(ip), Tracked(wp), hidden_size,
        Ghost(input_ids_repr), Ghost(weight_repr), Ghost(scope));
    proof {
        let width = hidden_size as nat;
        EMBED::checked_scaled_binding(input_ids_repr, weight_repr, width);
        reveal(RT::scaled_embed_kernel_cell_repr);
        reveal(MODEL::scaled_embed_repr);
        assert forall|row: int| 0 <= row < input_ids_repr.len() implies
            (#[trigger] MODEL::scaled_embed_repr(input_ids_repr, weight_repr, width)[row])
                == EMBED::scaled_output(input_ids_repr, weight_repr, width)[row] by {
            EMBED::scaled_row_shape(input_ids_repr[row], weight_repr, width);
            assert(MODEL::scaled_embed_repr(input_ids_repr, weight_repr, width)[row]
                =~= EMBED::scaled_row_output(input_ids_repr[row], weight_repr, width));
        };
        assert(MODEL::scaled_embed_repr(input_ids_repr, weight_repr, width)
            =~= EMBED::scaled_output(input_ids_repr, weight_repr, width));
    }
    out
}

#[verifier::external_body]
fn scaled_embed_raw(
    runtime: &RT::ModelFamilyRuntime,
    input_ids: &RT::Tensor,
    weight: &RT::Tensor,
    Tracked(ip): Tracked<&RT::TensorPerm>,
    Tracked(wp): Tracked<&RT::TensorPerm>,
    hidden_size: usize,
    Ghost(input_ids_repr): Ghost<IntTensor1D>,
    Ghost(weight_repr): Ghost<Tensor2D>,
    Ghost(scope): Ghost<Set<RT::TensorId>>,
) -> (out: (RT::Tensor, Tracked<RT::TensorPerm>))
    requires
        hidden_size_matches(runtime, hidden_size as nat),
        RT::int_tensor_repr_1d(*ip, *input_ids, input_ids_repr),
        RT::tensor_repr_2d(*wp, *weight, weight_repr),
        hidden_size > 0,
    ensures ({ let (tensor, perm) = out;
        &&& tensor.id() != input_ids.id()
        &&& tensor.id() != weight.id()
        &&& !scope.contains(tensor.id())
        &&& EMBED::scaled_layout(weight_repr, hidden_size as nat)
        &&& RT::tensor_repr_2d(
            perm@, tensor,
            EMBED::scaled_raw_output(
                input_ids_repr, weight_repr, hidden_size as nat,
            ),
        )
    }),
{
    #[cfg(not(verus_only))]
    {
        let inner = pyo3::Python::with_gil(
            |py| -> pyo3::PyResult<pyo3::Py<pyo3::PyAny>> {
                Ok(runtime.handle.inner.bind(py).call_method1(
                    "scaled_embed",
                    (input_ids.inner.bind(py), weight.inner.bind(py)),
                )?.unbind())
            },
        )
        .expect("qualified four-norm scaled embedding failed");
        (RT::Tensor { inner }, Tracked::assume_new())
    }
    #[cfg(verus_only)]
    {
        unreachable!()
    }
}

pub fn rms_norm(
    runtime: &RT::ModelFamilyRuntime,
    input: &RT::Tensor,
    weight: &RT::Tensor,
    site: RmsNormSite,
    Ghost(policy): Ghost<NormPolicyRepr>,
    Tracked(ip): Tracked<&RT::TensorPerm>,
    Tracked(wp): Tracked<&RT::TensorPerm>,
    Ghost(input_repr): Ghost<Tensor2D>,
    Ghost(weight_repr): Ghost<Tensor1D>,
    Ghost(scope): Ghost<Set<RT::TensorId>>,
) -> (out: (RT::Tensor, Tracked<RT::TensorPerm>))
    requires
        configuration_valid(runtime),
        runtime_norm_policy(runtime) == Some(policy),
        RT::tensor_repr_2d(*ip, *input, input_repr),
        RT::tensor_repr_1d(*wp, *weight, weight_repr),
    ensures ({ let (tensor, perm) = out;
        &&& tensor.id() != input.id()
        &&& tensor.id() != weight.id()
        &&& !scope.contains(tensor.id())
        &&& RT::tensor_repr_2d(
            perm@, tensor, norm_repr(input_repr, weight_repr, policy),
        )
    }),
{
    let out = rms_norm_raw(runtime, input, weight, site, Ghost(policy),
        Tracked(ip), Tracked(wp), Ghost(input_repr), Ghost(weight_repr), Ghost(scope));
    proof {
        checked_norm_binding(input_repr, weight_repr, policy);
    }
    out
}

// The successful Python call supplies actual layout and the policy-selected
// raw execution. Its row projection follows from the generated kernel law.
#[verifier::external_body]
fn rms_norm_raw(
    runtime: &RT::ModelFamilyRuntime,
    input: &RT::Tensor,
    weight: &RT::Tensor,
    site: RmsNormSite,
    Ghost(policy): Ghost<NormPolicyRepr>,
    Tracked(ip): Tracked<&RT::TensorPerm>,
    Tracked(wp): Tracked<&RT::TensorPerm>,
    Ghost(input_repr): Ghost<Tensor2D>,
    Ghost(weight_repr): Ghost<Tensor1D>,
    Ghost(scope): Ghost<Set<RT::TensorId>>,
) -> (out: (RT::Tensor, Tracked<RT::TensorPerm>))
    requires
        configuration_valid(runtime),
        runtime_norm_policy(runtime) == Some(policy),
        RT::tensor_repr_2d(*ip, *input, input_repr),
        RT::tensor_repr_1d(*wp, *weight, weight_repr),
    ensures ({ let (tensor, perm) = out;
        &&& tensor.id() != input.id()
        &&& tensor.id() != weight.id()
        &&& !scope.contains(tensor.id())
        &&& NORM::layout(input_repr, weight_repr)
        &&& RT::tensor_repr_2d(
            perm@, tensor, norm_raw_output(input_repr, weight_repr, policy),
        )
    }),
{
    #[cfg(not(verus_only))]
    {
        let inner = pyo3::Python::with_gil(
            |py| -> pyo3::PyResult<pyo3::Py<pyo3::PyAny>> {
                Ok(runtime.handle.inner.bind(py).call_method1(
                    "rms_norm",
                    (
                        input.inner.bind(py),
                        weight.inner.bind(py),
                        rms_norm_site_name(site),
                    ),
                )?.unbind())
            },
        )
        .expect("qualified four-norm RMSNorm failed");
        (RT::Tensor { inner }, Tracked::assume_new())
    }
    #[cfg(verus_only)]
    {
        unreachable!()
    }
}

pub fn add(
    runtime: &RT::ModelFamilyRuntime,
    left: &RT::Tensor,
    right: &RT::Tensor,
    site: AddSite,
    Tracked(lp): Tracked<&RT::TensorPerm>,
    Tracked(rp): Tracked<&RT::TensorPerm>,
    Ghost(left_repr): Ghost<Tensor2D>,
    Ghost(right_repr): Ghost<Tensor2D>,
    Ghost(scope): Ghost<Set<RT::TensorId>>,
) -> (out: (RT::Tensor, Tracked<RT::TensorPerm>))
    requires
        configuration_valid(runtime),
        RT::tensor_repr_2d(*lp, *left, left_repr),
        RT::tensor_repr_2d(*rp, *right, right_repr),
        left_repr.len() == right_repr.len(),
    ensures ({ let (tensor, perm) = out;
        &&& tensor.id() != left.id()
        &&& tensor.id() != right.id()
        &&& !scope.contains(tensor.id())
        &&& RT::tensor_repr_2d(
            perm@, tensor, MODEL::add_repr(left_repr, right_repr),
        )
    }),
{
    let out = add_raw(runtime, left, right, site, Tracked(lp), Tracked(rp), Ghost(left_repr), Ghost(right_repr), Ghost(scope));
    proof {
        PW::checked_add_binding(left_repr, right_repr);
        reveal(RT::add_kernel_cell_repr);
        reveal(MODEL::add_repr);
        assert forall|r: int| 0 <= r < left_repr.len() implies
            (#[trigger] MODEL::add_repr(left_repr, right_repr)[r]) == PW::add_output(left_repr, right_repr)[r] by {
            assert(MODEL::add_repr(left_repr, right_repr)[r] =~= PW::add_row_output(left_repr[r], right_repr[r]));
        };
        assert(PW::add_output(left_repr, right_repr) =~= MODEL::add_repr(left_repr, right_repr));
    }
    out
}

#[verifier::external_body]
fn add_raw(
    runtime: &RT::ModelFamilyRuntime,
    left: &RT::Tensor,
    right: &RT::Tensor,
    site: AddSite,
    Tracked(lp): Tracked<&RT::TensorPerm>,
    Tracked(rp): Tracked<&RT::TensorPerm>,
    Ghost(left_repr): Ghost<Tensor2D>,
    Ghost(right_repr): Ghost<Tensor2D>,
    Ghost(scope): Ghost<Set<RT::TensorId>>,
) -> (out: (RT::Tensor, Tracked<RT::TensorPerm>))
    requires
        configuration_valid(runtime),
        RT::tensor_repr_2d(*lp, *left, left_repr),
        RT::tensor_repr_2d(*rp, *right, right_repr),
        left_repr.len() == right_repr.len(),
    ensures ({ let (tensor, perm) = out;
        &&& tensor.id() != left.id()
        &&& tensor.id() != right.id()
        &&& !scope.contains(tensor.id())
        &&& PW::binary_layout(left_repr, right_repr)
        &&& (left_repr.len() > 0 ==> PW::width(left_repr) > 0)
        &&& RT::tensor_repr_2d(
            perm@, tensor, PW::add_raw_output(left_repr, right_repr),
        )
    }),
{
    #[cfg(not(verus_only))]
    {
        let inner = pyo3::Python::with_gil(
            |py| -> pyo3::PyResult<pyo3::Py<pyo3::PyAny>> {
                Ok(runtime.handle.inner.bind(py).call_method1(
                    "add",
                    (
                        left.inner.bind(py),
                        right.inner.bind(py),
                        add_site_name(site),
                    ),
                )?.unbind())
            },
        )
        .expect("qualified four-norm residual add failed");
        (RT::Tensor { inner }, Tracked::assume_new())
    }
    #[cfg(verus_only)]
    {
        unreachable!()
    }
}

pub fn gelu_tanh_mul(
    runtime: &RT::ModelFamilyRuntime,
    gate: &RT::Tensor,
    up: &RT::Tensor,
    Tracked(gp): Tracked<&RT::TensorPerm>,
    Tracked(up_perm): Tracked<&RT::TensorPerm>,
    Ghost(gate_repr): Ghost<Tensor2D>,
    Ghost(up_repr): Ghost<Tensor2D>,
    Ghost(scope): Ghost<Set<RT::TensorId>>,
) -> (out: (RT::Tensor, Tracked<RT::TensorPerm>))
    requires
        configuration_valid(runtime),
        RT::tensor_repr_2d(*gp, *gate, gate_repr),
        RT::tensor_repr_2d(*up_perm, *up, up_repr),
        gate_repr.len() == up_repr.len(),
    ensures ({ let (tensor, perm) = out;
        &&& tensor.id() != gate.id()
        &&& tensor.id() != up.id()
        &&& !scope.contains(tensor.id())
        &&& RT::tensor_repr_2d(
            perm@, tensor, MODEL::gelu_tanh_mul_repr(gate_repr, up_repr),
        )
    }),
{
    let out = gelu_tanh_mul_raw(runtime, gate, up, Tracked(gp), Tracked(up_perm), Ghost(gate_repr), Ghost(up_repr), Ghost(scope));
    proof {
        PW::checked_gelu_tanh_mul_binding(gate_repr, up_repr);
        reveal(RT::gelu_tanh_mul_kernel_cell_repr);
        reveal(MODEL::gelu_tanh_mul_repr);
        assert forall|r: int| 0 <= r < gate_repr.len() implies
            (#[trigger] MODEL::gelu_tanh_mul_repr(gate_repr, up_repr)[r]) == PW::gelu_tanh_mul_output(gate_repr, up_repr)[r] by {
            assert(MODEL::gelu_tanh_mul_repr(gate_repr, up_repr)[r] =~= PW::gelu_tanh_mul_row_output(gate_repr[r], up_repr[r]));
        };
        assert(PW::gelu_tanh_mul_output(gate_repr, up_repr) =~= MODEL::gelu_tanh_mul_repr(gate_repr, up_repr));
    }
    out
}

#[verifier::external_body]
fn gelu_tanh_mul_raw(
    runtime: &RT::ModelFamilyRuntime,
    gate: &RT::Tensor,
    up: &RT::Tensor,
    Tracked(gp): Tracked<&RT::TensorPerm>,
    Tracked(up_perm): Tracked<&RT::TensorPerm>,
    Ghost(gate_repr): Ghost<Tensor2D>,
    Ghost(up_repr): Ghost<Tensor2D>,
    Ghost(scope): Ghost<Set<RT::TensorId>>,
) -> (out: (RT::Tensor, Tracked<RT::TensorPerm>))
    requires
        configuration_valid(runtime),
        RT::tensor_repr_2d(*gp, *gate, gate_repr),
        RT::tensor_repr_2d(*up_perm, *up, up_repr),
        gate_repr.len() == up_repr.len(),
    ensures ({ let (tensor, perm) = out;
        &&& tensor.id() != gate.id()
        &&& tensor.id() != up.id()
        &&& !scope.contains(tensor.id())
        &&& PW::binary_layout(gate_repr, up_repr)
        &&& RT::tensor_repr_2d(
            perm@, tensor, PW::gelu_tanh_mul_raw_output(gate_repr, up_repr),
        )
    }),
{
    #[cfg(not(verus_only))]
    {
        let inner = pyo3::Python::with_gil(
            |py| -> pyo3::PyResult<pyo3::Py<pyo3::PyAny>> {
                Ok(runtime.handle.inner.bind(py).call_method1(
                    "gelu_tanh_mul",
                    (gate.inner.bind(py), up.inner.bind(py)),
                )?.unbind())
            },
        )
        .expect("qualified four-norm GELU tanh multiply failed");
        (RT::Tensor { inner }, Tracked::assume_new())
    }
    #[cfg(verus_only)]
    {
        unreachable!()
    }
}

pub open spec fn runtime_qk_norm_policy(runtime: &RT::ModelFamilyRuntime, kind: AttentionKind)
    -> Option<NormPolicyRepr>
{
    match runtime_policy(runtime) {
        Some(policy) => Some(policy.qk_norm),
        None => None,
    }
}

pub open spec fn qk_norm_repr(
    q: Tensor2D, k: Tensor2D, q_weight: Tensor1D, k_weight: Tensor1D, policy: NormPolicyRepr,
) -> (Tensor2D, Tensor2D) {
    (Seq::new(q.len(), |i: int| ROWS::head_norm_row(q[i], q_weight, policy)),
     Seq::new(k.len(), |i: int| ROWS::head_norm_row(k[i], k_weight, policy)))
}

pub open spec fn head_norm_layout(input: Tensor2D, weight: Tensor1D, policy: NormPolicyRepr) -> bool {
    HEAD::layout(input, weight, match policy { NormPolicyRepr::UnitOffset => true, _ => false })
}

pub open spec fn head_norm_raw_output(input: Tensor2D, weight: Tensor1D, policy: NormPolicyRepr) -> Tensor2D {
    match policy {
        NormPolicyRepr::UnitOffset => HEAD::raw_output(input, weight,
            float_parameter_scalar_repr(unit_offset_norm_epsilon_repr()), true),
        NormPolicyRepr::Direct(eps) => HEAD::raw_output(input, weight,
            float_parameter_scalar_repr(eps), false),
    }
}

pub proof fn checked_head_norm_binding(input: Tensor2D, weight: Tensor1D, policy: NormPolicyRepr)
    requires head_norm_layout(input, weight, policy),
    ensures head_norm_raw_output(input, weight, policy)
        == Seq::new(input.len(), |r: int| ROWS::head_norm_row(input[r], weight, policy)),
{
    match policy {
        NormPolicyRepr::UnitOffset => {
            let eps = float_parameter_scalar_repr(unit_offset_norm_epsilon_repr());
            HEAD::checked_binding(input, weight, eps, true);
            reveal(crate::boundary::dense_layer_primitives::qk_norm_row_repr);
        },
        NormPolicyRepr::Direct(eps) => {
            RT::checked_head_norm_binding(input, weight, float_parameter_scalar_repr(eps));
            assert forall|r: int| 0 <= r < input.len() implies
                (#[trigger] RT::head_rms_norm_kernel_repr(input, weight, float_parameter_scalar_repr(eps))[r])
                == ROWS::head_norm_row(input[r], weight, policy) by {
                assert(input[r].len() == HEAD::width(input));
            };
        },
    }
    assert(head_norm_raw_output(input, weight, policy)
        =~= Seq::new(input.len(), |r: int| ROWS::head_norm_row(input[r], weight, policy)));
}

pub fn qk_norm(
    runtime: &RT::ModelFamilyRuntime,
    q: &RT::Tensor,
    k: &RT::Tensor,
    q_weight: &RT::Tensor,
    k_weight: &RT::Tensor,
    attention_kind: AttentionKind,
    Ghost(policy): Ghost<NormPolicyRepr>,
    Tracked(qp): Tracked<&RT::TensorPerm>,
    Tracked(kp): Tracked<&RT::TensorPerm>,
    Tracked(qwp): Tracked<&RT::TensorPerm>,
    Tracked(kwp): Tracked<&RT::TensorPerm>,
    Ghost(q_repr): Ghost<Tensor2D>,
    Ghost(k_repr): Ghost<Tensor2D>,
    Ghost(q_weight_repr): Ghost<Tensor1D>,
    Ghost(k_weight_repr): Ghost<Tensor1D>,
    Ghost(scope): Ghost<Set<RT::TensorId>>,
) -> (out: ((RT::Tensor, RT::Tensor),
            (Tracked<RT::TensorPerm>, Tracked<RT::TensorPerm>)))
    requires
        configuration_valid(runtime),
        runtime_qk_norm_policy(runtime, attention_kind) == Some(policy),
        RT::tensor_repr_2d(*qp, *q, q_repr),
        RT::tensor_repr_2d(*kp, *k, k_repr),
        RT::tensor_repr_1d(*qwp, *q_weight, q_weight_repr),
        RT::tensor_repr_1d(*kwp, *k_weight, k_weight_repr),
        q_repr.len() == k_repr.len(),
    ensures ({ let ((nq, nk), (nqp, nkp)) = out;
        let normalized = qk_norm_repr(
            q_repr, k_repr, q_weight_repr, k_weight_repr, policy,
        );
        &&& nq.id() != nk.id()
        &&& !scope.contains(nq.id())
        &&& !scope.contains(nk.id())
        &&& RT::tensor_repr_2d(nqp@, nq, normalized.0)
        &&& RT::tensor_repr_2d(nkp@, nk, normalized.1)
    }),
{
    let out = qk_norm_raw(runtime, q, k, q_weight, k_weight, attention_kind, Ghost(policy),
        Tracked(qp), Tracked(kp), Tracked(qwp), Tracked(kwp),
        Ghost(q_repr), Ghost(k_repr), Ghost(q_weight_repr), Ghost(k_weight_repr), Ghost(scope));
    proof {
        checked_head_norm_binding(q_repr, q_weight_repr, policy);
        checked_head_norm_binding(k_repr, k_weight_repr, policy);
    }
    out
}

#[verifier::external_body]
fn qk_norm_raw(
    runtime: &RT::ModelFamilyRuntime,
    q: &RT::Tensor,
    k: &RT::Tensor,
    q_weight: &RT::Tensor,
    k_weight: &RT::Tensor,
    attention_kind: AttentionKind,
    Ghost(policy): Ghost<NormPolicyRepr>,
    Tracked(qp): Tracked<&RT::TensorPerm>,
    Tracked(kp): Tracked<&RT::TensorPerm>,
    Tracked(qwp): Tracked<&RT::TensorPerm>,
    Tracked(kwp): Tracked<&RT::TensorPerm>,
    Ghost(q_repr): Ghost<Tensor2D>,
    Ghost(k_repr): Ghost<Tensor2D>,
    Ghost(q_weight_repr): Ghost<Tensor1D>,
    Ghost(k_weight_repr): Ghost<Tensor1D>,
    Ghost(scope): Ghost<Set<RT::TensorId>>,
) -> (out: ((RT::Tensor, RT::Tensor),
            (Tracked<RT::TensorPerm>, Tracked<RT::TensorPerm>)))
    requires
        configuration_valid(runtime),
        runtime_qk_norm_policy(runtime, attention_kind) == Some(policy),
        RT::tensor_repr_2d(*qp, *q, q_repr),
        RT::tensor_repr_2d(*kp, *k, k_repr),
        RT::tensor_repr_1d(*qwp, *q_weight, q_weight_repr),
        RT::tensor_repr_1d(*kwp, *k_weight, k_weight_repr),
        q_repr.len() == k_repr.len(),
    ensures ({ let ((nq, nk), (nqp, nkp)) = out;
        let normalized = (head_norm_raw_output(q_repr, q_weight_repr, policy),
            head_norm_raw_output(k_repr, k_weight_repr, policy));
        &&& head_norm_layout(q_repr, q_weight_repr, policy)
        &&& head_norm_layout(k_repr, k_weight_repr, policy)
        &&& nq.id() != nk.id()
        &&& !scope.contains(nq.id())
        &&& !scope.contains(nk.id())
        &&& RT::tensor_repr_2d(nqp@, nq, normalized.0)
        &&& RT::tensor_repr_2d(nkp@, nk, normalized.1)
    }),
{
    #[cfg(not(verus_only))]
    {
        let (nq, nk) = pyo3::Python::with_gil(
            |py| -> pyo3::PyResult<(
                pyo3::Py<pyo3::PyAny>,
                pyo3::Py<pyo3::PyAny>,
            )> {
                let result = runtime.handle.inner.bind(py).call_method1(
                    "qk_norm",
                    (
                        q.inner.bind(py),
                        k.inner.bind(py),
                        q_weight.inner.bind(py),
                        k_weight.inner.bind(py),
                        attention_kind_name(attention_kind),
                    ),
                )?;
                let tuple = result.downcast::<pyo3::types::PyTuple>()?;
                Ok((tuple.get_item(0)?.unbind(), tuple.get_item(1)?.unbind()))
            },
        )
        .expect("qualified four-norm Q/K norm failed");
        (
            (RT::Tensor { inner: nq }, RT::Tensor { inner: nk }),
            (Tracked::assume_new(), Tracked::assume_new()),
        )
    }
    #[cfg(verus_only)]
    {
        unreachable!()
    }
}


pub open spec fn runtime_rotary_config(runtime: &RT::ModelFamilyRuntime, kind: AttentionKind)
    -> Option<RotaryConfigRepr>
{
    match runtime_policy(runtime) {
        Some(policy) => Some(match kind {
            AttentionKind::SlidingWindow => policy.local_rotary,
            AttentionKind::Full => policy.global_rotary,
        }),
        _ => None,
    }
}

pub open spec fn rotary_embed_repr(
    positions: IntTensor1D, q: Tensor2D, k: Tensor2D, geometry: AttentionGeometryRepr, rotary: RotaryConfigRepr,
) -> (Tensor2D, Tensor2D) {
    (Seq::new(q.len(), |i: int| crate::boundary::dense_layer_primitives::rotary_row_with_config_repr(
        positions[i], q[i], geometry.num_attention_heads, geometry.head_dim, rotary)),
     Seq::new(k.len(), |i: int| crate::boundary::dense_layer_primitives::rotary_row_with_config_repr(
        positions[i], k[i], geometry.num_key_value_heads, geometry.head_dim, rotary)))
}

pub fn rotary_embed(
    runtime: &RT::ModelFamilyRuntime,
    q: &RT::Tensor,
    k: &RT::Tensor,
    positions: &RT::Tensor,
    attention_kind: AttentionKind,
    Ghost(geometry): Ghost<AttentionGeometryRepr>,
    Ghost(rotary): Ghost<RotaryConfigRepr>,
    Tracked(qp): Tracked<&RT::TensorPerm>,
    Tracked(kp): Tracked<&RT::TensorPerm>,
    Tracked(pp): Tracked<&RT::TensorPerm>,
    Ghost(q_repr): Ghost<Tensor2D>,
    Ghost(k_repr): Ghost<Tensor2D>,
    Ghost(positions_repr): Ghost<IntTensor1D>,
    Ghost(scope): Ghost<Set<RT::TensorId>>,
) -> (out: ((RT::Tensor, RT::Tensor),
            (Tracked<RT::TensorPerm>, Tracked<RT::TensorPerm>)))
    requires
        configuration_valid(runtime),
        runtime_rotary_config(runtime, attention_kind) == Some(rotary),
        runtime_attention_geometry(runtime, attention_kind) == Some(geometry),
        RT::tensor_repr_2d(*qp, *q, q_repr),
        RT::tensor_repr_2d(*kp, *k, k_repr),
        RT::int_tensor_repr_1d(*pp, *positions, positions_repr),
        positions_repr.len() == q_repr.len(),
        q_repr.len() == k_repr.len(),
    ensures ({ let ((rq, rk), (rqp, rkp)) = out;
        let rotated = rotary_embed_repr(
            positions_repr, q_repr, k_repr, geometry, rotary,
        );
        &&& rq.id() != rk.id()
        &&& !scope.contains(rq.id())
        &&& !scope.contains(rk.id())
        &&& RT::tensor_repr_2d(rqp@, rq, rotated.0)
        &&& RT::tensor_repr_2d(rkp@, rk, rotated.1)
    }),
{
    let out = rotary_embed_raw(runtime, q, k, positions, attention_kind, Ghost(geometry), Ghost(rotary),
        Tracked(qp), Tracked(kp), Tracked(pp), Ghost(q_repr), Ghost(k_repr), Ghost(positions_repr), Ghost(scope));
    proof {
        crate::boundary::dense_layer_primitives::checked_rotary_rows_binding(positions_repr, q_repr, geometry.num_attention_heads, geometry.head_dim, rotary);
        crate::boundary::dense_layer_primitives::checked_rotary_rows_binding(positions_repr, k_repr, geometry.num_key_value_heads, geometry.head_dim, rotary);
    }
    out
}

#[verifier::external_body]
fn rotary_embed_raw(
    runtime: &RT::ModelFamilyRuntime,
    q: &RT::Tensor,
    k: &RT::Tensor,
    positions: &RT::Tensor,
    attention_kind: AttentionKind,
    Ghost(geometry): Ghost<AttentionGeometryRepr>,
    Ghost(rotary): Ghost<RotaryConfigRepr>,
    Tracked(qp): Tracked<&RT::TensorPerm>,
    Tracked(kp): Tracked<&RT::TensorPerm>,
    Tracked(pp): Tracked<&RT::TensorPerm>,
    Ghost(q_repr): Ghost<Tensor2D>,
    Ghost(k_repr): Ghost<Tensor2D>,
    Ghost(positions_repr): Ghost<IntTensor1D>,
    Ghost(scope): Ghost<Set<RT::TensorId>>,
) -> (out: ((RT::Tensor, RT::Tensor),
            (Tracked<RT::TensorPerm>, Tracked<RT::TensorPerm>)))
    requires
        configuration_valid(runtime),
        runtime_rotary_config(runtime, attention_kind) == Some(rotary),
        runtime_attention_geometry(runtime, attention_kind) == Some(geometry),
        RT::tensor_repr_2d(*qp, *q, q_repr),
        RT::tensor_repr_2d(*kp, *k, k_repr),
        RT::int_tensor_repr_1d(*pp, *positions, positions_repr),
        positions_repr.len() == q_repr.len(),
        q_repr.len() == k_repr.len(),
    ensures ({ let ((rq, rk), (rqp, rkp)) = out;
        let rotated = (
            RT::rotary_component_raw_output(positions_repr, q_repr, geometry.num_attention_heads, geometry.head_dim, rotary),
            RT::rotary_component_raw_output(positions_repr, k_repr, geometry.num_key_value_heads, geometry.head_dim, rotary));
        &&& RT::rotary_component_layout(positions_repr, q_repr, geometry.num_attention_heads, geometry.head_dim, rotary)
        &&& RT::rotary_component_layout(positions_repr, k_repr, geometry.num_key_value_heads, geometry.head_dim, rotary)
        &&& rq.id() != rk.id()
        &&& !scope.contains(rq.id())
        &&& !scope.contains(rk.id())
        &&& RT::tensor_repr_2d(rqp@, rq, rotated.0)
        &&& RT::tensor_repr_2d(rkp@, rk, rotated.1)
    }),
{
    #[cfg(not(verus_only))]
    {
        let (rq, rk) = pyo3::Python::with_gil(
            |py| -> pyo3::PyResult<(
                pyo3::Py<pyo3::PyAny>,
                pyo3::Py<pyo3::PyAny>,
            )> {
                let result = runtime.handle.inner.bind(py).call_method1(
                    "rotary_embed",
                    (
                        q.inner.bind(py),
                        k.inner.bind(py),
                        positions.inner.bind(py),
                        attention_kind_name(attention_kind),
                    ),
                )?;
                let tuple = result.downcast::<pyo3::types::PyTuple>()?;
                Ok((tuple.get_item(0)?.unbind(), tuple.get_item(1)?.unbind()))
            },
        )
        .expect("qualified four-norm rotary embedding failed");
        (
            (RT::Tensor { inner: rq }, RT::Tensor { inner: rk }),
            (Tracked::assume_new(), Tracked::assume_new()),
        )
    }
    #[cfg(verus_only)]
    {
        unreachable!()
    }
}

pub open spec fn attention_config(
    kind: AttentionKind,
    sliding_window: nat,
) -> AttentionConfigRepr {
    match kind {
        AttentionKind::SlidingWindow =>
            AttentionConfigRepr::SlidingWindow(sliding_window),
        AttentionKind::Full => AttentionConfigRepr::Full,
    }
}

pub open spec fn runtime_attention_parameters_match(
    runtime: &RT::ModelFamilyRuntime, kind: AttentionKind, geometry: AttentionGeometryRepr,
    sliding_window: nat, attention_scale: AttentionScaleRepr,
) -> bool {
    configuration_valid(runtime) && match runtime_policy(runtime) {
        Some(policy) =>
            runtime_attention_geometry(runtime, kind) == Some(geometry)
            && policy.sliding_window == sliding_window && policy.attention_scale == attention_scale,
        None => false,
    }
}

pub open spec fn runtime_attention_geometry(runtime: &RT::ModelFamilyRuntime, kind: AttentionKind)
    -> Option<AttentionGeometryRepr>
{
    match runtime_policy(runtime) {
        Some(policy) => Some(match kind {
            AttentionKind::SlidingWindow => policy.local_geometry,
            AttentionKind::Full => policy.global_geometry,
        }),
        None => None,
    }
}

// Both four-norm families obtain policy through the closed deployment identity.
pub proof fn lemma_gemma4_deployment_policy_is_exact(
    runtime: &RT::ModelFamilyRuntime, config: Gemma4DeploymentConfigRepr,
)
    requires RT::family_runtime_deployment_config_repr(runtime)
        == Some(ModelDeploymentConfigRepr::Gemma4Text(config)),
    ensures
        runtime_policy(runtime).is_some(),
        runtime_norm_policy(runtime) == Some(NormPolicyRepr::Direct(config.rms_norm.epsilon)),
        runtime_attention_geometry(runtime, AttentionKind::SlidingWindow)
            == Some(attention_geometry_repr(config.geometry)),
        runtime_attention_geometry(runtime, AttentionKind::Full) == Some(config.global_attention_geometry),
        runtime_rotary_config(runtime, AttentionKind::Full) == Some(config.global_rotary),
        runtime_policy(runtime).unwrap().value_norm_epsilon == Some(config.rms_norm.epsilon),
        runtime_policy(runtime).unwrap().layer_scale,
        runtime_policy(runtime).unwrap().logits_softcap == config.final_logit_softcap,
{
}

pub open spec fn value_norm_repr(input: Tensor2D, head_dim: nat, epsilon: FloatParameterBits)
    -> Tensor2D
{
    Seq::new(input.len(), |i: int| ROWS::value_row(input[i], head_dim, Some(epsilon)))
}

pub open spec fn value_norm_weight(head_dim: nat) -> Tensor1D {
    Seq::new(head_dim, |_col: int| float_parameter_scalar_repr(
        FloatParameterBits { bits: 4_607_182_418_800_017_408 }))
}

pub fn value_norm(
    runtime: &RT::ModelFamilyRuntime,
    input: &RT::Tensor,
    attention_kind: AttentionKind,
    Tracked(ip): Tracked<&RT::TensorPerm>,
    Ghost(input_repr): Ghost<Tensor2D>,
    Ghost(geometry): Ghost<AttentionGeometryRepr>,
    Ghost(epsilon): Ghost<FloatParameterBits>,
    Ghost(scope): Ghost<Set<RT::TensorId>>,
) -> (out: (RT::Tensor, Tracked<RT::TensorPerm>))
    requires
        configuration_valid(runtime),
        runtime_policy(runtime).unwrap().value_norm_epsilon == Some(epsilon),
        runtime_attention_geometry(runtime, attention_kind) == Some(geometry),
        RT::tensor_repr_2d(*ip, *input, input_repr),
        crate::proof::tensor::shape::tensor2d_shape(input_repr, input_repr.len(),
            geometry.num_key_value_heads * geometry.head_dim),
    ensures ({ let (tensor, perm) = out;
        &&& tensor.id() != input.id()
        &&& !scope.contains(tensor.id())
        &&& RT::tensor_repr_2d(perm@, tensor, value_norm_repr(input_repr, geometry.head_dim, epsilon))
    }),
{
    let out = value_norm_raw(runtime, input, attention_kind, Tracked(ip), Ghost(input_repr),
        Ghost(geometry), Ghost(epsilon), Ghost(scope));
    proof {
        let weight = value_norm_weight(geometry.head_dim);
        let eps = float_parameter_scalar_repr(epsilon);
        RT::checked_head_norm_binding(input_repr, weight, eps);
        reveal(RT::view_as_kv_repr);
        assert forall|r: int| 0 <= r < input_repr.len() implies
            (#[trigger] RT::view_as_kv_repr(HEAD::raw_output(input_repr, weight, eps, false))[r])
                == value_norm_repr(input_repr, geometry.head_dim, epsilon)[r] by {
            assert(input_repr[r].len() == HEAD::width(input_repr));
        };
        assert(RT::view_as_kv_repr(HEAD::raw_output(input_repr, weight, eps, false))
            =~= value_norm_repr(input_repr, geometry.head_dim, epsilon));
    }
    out
}

#[verifier::external_body]
fn value_norm_raw(
    runtime: &RT::ModelFamilyRuntime,
    input: &RT::Tensor,
    attention_kind: AttentionKind,
    Tracked(ip): Tracked<&RT::TensorPerm>,
    Ghost(input_repr): Ghost<Tensor2D>,
    Ghost(geometry): Ghost<AttentionGeometryRepr>,
    Ghost(epsilon): Ghost<FloatParameterBits>,
    Ghost(scope): Ghost<Set<RT::TensorId>>,
) -> (out: (RT::Tensor, Tracked<RT::TensorPerm>))
    requires
        configuration_valid(runtime),
        runtime_policy(runtime).unwrap().value_norm_epsilon == Some(epsilon),
        runtime_attention_geometry(runtime, attention_kind) == Some(geometry),
        RT::tensor_repr_2d(*ip, *input, input_repr),
        crate::proof::tensor::shape::tensor2d_shape(input_repr, input_repr.len(),
            geometry.num_key_value_heads * geometry.head_dim),
    ensures ({ let (tensor, perm) = out;
        &&& tensor.id() != input.id()
        &&& !scope.contains(tensor.id())
        &&& HEAD::layout(input_repr, value_norm_weight(geometry.head_dim), false)
        &&& RT::tensor_repr_2d(perm@, tensor, RT::view_as_kv_repr(HEAD::raw_output(
            input_repr, value_norm_weight(geometry.head_dim), float_parameter_scalar_repr(epsilon), false)))
    }),
{
    #[cfg(not(verus_only))]
    {
        let inner = pyo3::Python::with_gil(|py| -> pyo3::PyResult<pyo3::Py<pyo3::PyAny>> {
            Ok(runtime.handle.inner.bind(py).call_method1("value_norm",
                (input.inner.bind(py), attention_kind_name(attention_kind)))?.unbind())
        }).expect("qualified four-norm value normalization failed");
        (RT::Tensor { inner }, Tracked::assume_new())
    }
    #[cfg(verus_only)]
    { unreachable!() }
}

pub open spec fn scale_repr(input: Tensor2D, scalar: Tensor1D) -> Tensor2D {
    Seq::new(input.len(), |i: int| ROWS::scale_row(input[i], Some(scalar)))
}

pub fn scale(
    runtime: &RT::ModelFamilyRuntime,
    input: &RT::Tensor,
    scalar: &RT::Tensor,
    Tracked(ip): Tracked<&RT::TensorPerm>,
    Tracked(sp): Tracked<&RT::TensorPerm>,
    Ghost(input_repr): Ghost<Tensor2D>,
    Ghost(scalar_repr): Ghost<Tensor1D>,
    Ghost(scope): Ghost<Set<RT::TensorId>>,
) -> (out: (RT::Tensor, Tracked<RT::TensorPerm>))
    requires
        configuration_valid(runtime),
        runtime_policy(runtime).unwrap().layer_scale,
        RT::tensor_repr_2d(*ip, *input, input_repr),
        RT::tensor_repr_1d(*sp, *scalar, scalar_repr),
        scalar_repr.len() == 1,
    ensures ({ let (tensor, perm) = out;
        &&& tensor.id() != input.id()
        &&& tensor.id() != scalar.id()
        &&& !scope.contains(tensor.id())
        &&& RT::tensor_repr_2d(perm@, tensor, scale_repr(input_repr, scalar_repr))
    }),
{
    let out = scale_raw(runtime, input, scalar, Tracked(ip), Tracked(sp), Ghost(input_repr), Ghost(scalar_repr), Ghost(scope));
    proof {
        PW::checked_scale_binding(input_repr, scalar_repr);
        reveal(RT::scale_kernel_cell_repr);
        assert forall|r: int| 0 <= r < input_repr.len() implies
            (#[trigger] scale_repr(input_repr, scalar_repr)[r]) == PW::scale_output(input_repr, scalar_repr)[r] by {
            assert(scale_repr(input_repr, scalar_repr)[r] =~= PW::scale_row_output(input_repr[r], scalar_repr));
        };
        assert(PW::scale_output(input_repr, scalar_repr) =~= scale_repr(input_repr, scalar_repr));
    }
    out
}

#[verifier::external_body]
fn scale_raw(
    runtime: &RT::ModelFamilyRuntime,
    input: &RT::Tensor,
    scalar: &RT::Tensor,
    Tracked(ip): Tracked<&RT::TensorPerm>,
    Tracked(sp): Tracked<&RT::TensorPerm>,
    Ghost(input_repr): Ghost<Tensor2D>,
    Ghost(scalar_repr): Ghost<Tensor1D>,
    Ghost(scope): Ghost<Set<RT::TensorId>>,
) -> (out: (RT::Tensor, Tracked<RT::TensorPerm>))
    requires
        configuration_valid(runtime),
        runtime_policy(runtime).unwrap().layer_scale,
        RT::tensor_repr_2d(*ip, *input, input_repr),
        RT::tensor_repr_1d(*sp, *scalar, scalar_repr),
        scalar_repr.len() == 1,
    ensures ({ let (tensor, perm) = out;
        &&& tensor.id() != input.id()
        &&& tensor.id() != scalar.id()
        &&& !scope.contains(tensor.id())
        &&& PW::layout(input_repr)
        &&& RT::tensor_repr_2d(perm@, tensor, PW::scale_raw_output(input_repr, scalar_repr))
    }),
{
    #[cfg(not(verus_only))]
    {
        let inner = pyo3::Python::with_gil(|py| -> pyo3::PyResult<pyo3::Py<pyo3::PyAny>> {
            Ok(runtime.handle.inner.bind(py).call_method1("scale",
                (input.inner.bind(py), scalar.inner.bind(py)))?.unbind())
        }).expect("qualified four-norm layer scaling failed");
        (RT::Tensor { inner }, Tracked::assume_new())
    }
    #[cfg(verus_only)]
    { unreachable!() }
}

pub fn softcap(
    runtime: &RT::ModelFamilyRuntime,
    input: &RT::Tensor,
    Tracked(ip): Tracked<&RT::TensorPerm>,
    Ghost(input_repr): Ghost<Tensor2D>,
    Ghost(cap): Ghost<FloatParameterBits>,
    Ghost(scope): Ghost<Set<RT::TensorId>>,
) -> (out: (RT::Tensor, Tracked<RT::TensorPerm>))
    requires
        configuration_valid(runtime),
        runtime_policy(runtime).unwrap().logits_softcap == Some(cap),
        RT::tensor_repr_2d(*ip, *input, input_repr),
        positive_float_parameter_valid(cap),
    ensures ({ let (tensor, perm) = out;
        &&& tensor.id() != input.id()
        &&& !scope.contains(tensor.id())
        &&& RT::tensor_repr_2d(perm@, tensor, ROWS::logits_softcap_repr(input_repr, Some(cap)))
    }),
{
    let out = softcap_raw(runtime, input, Tracked(ip), Ghost(input_repr), Ghost(cap), Ghost(scope));
    proof {
        PW::checked_softcap_binding(input_repr, float_parameter_scalar_repr(cap));
        reveal(RT::softcap_kernel_cell_repr);
        assert forall|r: int| 0 <= r < input_repr.len() implies
            (#[trigger] ROWS::logits_softcap_repr(input_repr, Some(cap))[r]) == PW::softcap_output(input_repr, float_parameter_scalar_repr(cap))[r] by {
            assert(ROWS::logits_softcap_repr(input_repr, Some(cap))[r] =~= PW::softcap_row_output(input_repr[r], float_parameter_scalar_repr(cap)));
        };
        assert(PW::softcap_output(input_repr, float_parameter_scalar_repr(cap)) =~= ROWS::logits_softcap_repr(input_repr, Some(cap)));
    }
    out
}

#[verifier::external_body]
fn softcap_raw(
    runtime: &RT::ModelFamilyRuntime,
    input: &RT::Tensor,
    Tracked(ip): Tracked<&RT::TensorPerm>,
    Ghost(input_repr): Ghost<Tensor2D>,
    Ghost(cap): Ghost<FloatParameterBits>,
    Ghost(scope): Ghost<Set<RT::TensorId>>,
) -> (out: (RT::Tensor, Tracked<RT::TensorPerm>))
    requires
        configuration_valid(runtime),
        runtime_policy(runtime).unwrap().logits_softcap == Some(cap),
        RT::tensor_repr_2d(*ip, *input, input_repr),
        positive_float_parameter_valid(cap),
    ensures ({ let (tensor, perm) = out;
        &&& tensor.id() != input.id()
        &&& !scope.contains(tensor.id())
        &&& PW::layout(input_repr)
        &&& RT::tensor_repr_2d(perm@, tensor, PW::softcap_raw_output(input_repr, float_parameter_scalar_repr(cap)))
    }),
{
    #[cfg(not(verus_only))]
    {
        let inner = pyo3::Python::with_gil(|py| -> pyo3::PyResult<pyo3::Py<pyo3::PyAny>> {
            Ok(runtime.handle.inner.bind(py).call_method1("softcap", (input.inner.bind(py),))?.unbind())
        }).expect("qualified four-norm logit softcap failed");
        (RT::Tensor { inner }, Tracked::assume_new())
    }
    #[cfg(verus_only)]
    { unreachable!() }
}

pub fn paged_attention(
    runtime: &RT::ModelFamilyRuntime,
    q: &RT::Tensor,
    k_cache: &RT::Tensor,
    v_cache: &RT::Tensor,
    block_table: &RT::Tensor,
    cu_seqlens_q: &RT::Tensor,
    cu_seqlens_k: &RT::Tensor,
    max_seqlen_q: usize,
    max_seqlen_k: usize,
    attention_kind: AttentionKind,
    sliding_window: usize,
    Tracked(qp): Tracked<&RT::TensorPerm>,
    Tracked(kcp): Tracked<&RT::TensorPerm>,
    Tracked(vcp): Tracked<&RT::TensorPerm>,
    Tracked(btp): Tracked<&RT::TensorPerm>,
    Tracked(cuqp): Tracked<&RT::TensorPerm>,
    Tracked(cukp): Tracked<&RT::TensorPerm>,
    Ghost(q_repr): Ghost<Tensor2D>,
    Ghost(k_cache_repr): Ghost<KVCacheLayerRepr>,
    Ghost(v_cache_repr): Ghost<KVCacheLayerRepr>,
    Ghost(bt_repr): Ghost<Seq<Seq<BlockId>>>,
    Ghost(cu_q_repr): Ghost<Seq<int>>,
    Ghost(cu_k_repr): Ghost<Seq<int>>,
    Ghost(geometry): Ghost<AttentionGeometryRepr>,
    Ghost(attention_scale): Ghost<AttentionScaleRepr>,
    Ghost(scope): Ghost<Set<RT::TensorId>>,
) -> (out: (RT::Tensor, Tracked<RT::TensorPerm>))
    requires
        RT::paged_attention_numeric_domain(),
        runtime_attention_parameters_match(
            runtime, attention_kind,
            geometry,
            sliding_window as nat,
            attention_scale,
        ),
        RT::tensor_repr_2d(*qp, *q, q_repr),
        RT::kv_cache_tensor_repr(*kcp, *k_cache, k_cache_repr),
        RT::kv_cache_tensor_repr(*vcp, *v_cache, v_cache_repr),
        RT::block_table_repr(*btp, *block_table, bt_repr),
        RT::int_tensor_repr_1d(*cuqp, *cu_seqlens_q, cu_q_repr),
        RT::int_tensor_repr_1d(*cukp, *cu_seqlens_k, cu_k_repr),
        sliding_window > 0,
        RT::paged_attention_launch_ready(
            q_repr.len(), k_cache_repr, v_cache_repr,
            cu_q_repr, cu_k_repr,
            max_seqlen_q as nat, max_seqlen_k as nat, bt_repr,
        ),
    ensures ({ let (tensor, perm) = out;
        &&& tensor.id() != q.id()
        &&& !scope.contains(tensor.id())
        &&& RT::tensor_repr_2d(
            perm@, tensor,
            MODEL::paged_attention_repr(
                q_repr, k_cache_repr, v_cache_repr,
                cu_q_repr, cu_k_repr,
                max_seqlen_q as nat, max_seqlen_k as nat, bt_repr,
                attention_config(attention_kind, sliding_window as nat),
                AttentionParametersRepr { geometry, scale: attention_scale },
            ),
        )
    }),
{
    let out = paged_attention_raw(runtime, q, k_cache, v_cache, block_table,
        cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, attention_kind, sliding_window,
        Tracked(qp), Tracked(kcp), Tracked(vcp), Tracked(btp), Tracked(cuqp), Tracked(cukp),
        Ghost(q_repr), Ghost(k_cache_repr), Ghost(v_cache_repr), Ghost(bt_repr), Ghost(cu_q_repr), Ghost(cu_k_repr),
        Ghost(geometry), Ghost(attention_scale), Ghost(scope));
    proof {
        ATTN::checked_runtime_binding(q_repr, k_cache_repr, v_cache_repr, cu_q_repr, cu_k_repr, bt_repr,
            max_seqlen_q as nat, max_seqlen_k as nat, attention_kind, AttentionParametersRepr { geometry, scale: attention_scale }, ATTN::effective_window(attention_kind, sliding_window as nat));
        reveal(MODEL::paged_attention_repr);
        reveal(crate::boundary::dense_layer_primitives::full_paged_attention_repr);
        reveal(crate::boundary::dense_layer_primitives::sliding_window_paged_attention_repr);
    }
    out
}

// Trusted physical boundary only: qualified raw execution, concrete geometry,
// allocation/representation, and the exact generated numeric predicates under
// the explicit deployed finiteness assumption. No mapped-output equality or
// batching/causal relational law is assumed here.
#[verifier::external_body]
fn paged_attention_raw(
    runtime: &RT::ModelFamilyRuntime,
    q: &RT::Tensor,
    k_cache: &RT::Tensor,
    v_cache: &RT::Tensor,
    block_table: &RT::Tensor,
    cu_seqlens_q: &RT::Tensor,
    cu_seqlens_k: &RT::Tensor,
    max_seqlen_q: usize,
    max_seqlen_k: usize,
    attention_kind: AttentionKind,
    sliding_window: usize,
    Tracked(qp): Tracked<&RT::TensorPerm>,
    Tracked(kcp): Tracked<&RT::TensorPerm>,
    Tracked(vcp): Tracked<&RT::TensorPerm>,
    Tracked(btp): Tracked<&RT::TensorPerm>,
    Tracked(cuqp): Tracked<&RT::TensorPerm>,
    Tracked(cukp): Tracked<&RT::TensorPerm>,
    Ghost(q_repr): Ghost<Tensor2D>,
    Ghost(k_cache_repr): Ghost<KVCacheLayerRepr>,
    Ghost(v_cache_repr): Ghost<KVCacheLayerRepr>,
    Ghost(bt_repr): Ghost<Seq<Seq<BlockId>>>,
    Ghost(cu_q_repr): Ghost<Seq<int>>,
    Ghost(cu_k_repr): Ghost<Seq<int>>,
    Ghost(geometry): Ghost<AttentionGeometryRepr>,
    Ghost(attention_scale): Ghost<AttentionScaleRepr>,
    Ghost(scope): Ghost<Set<RT::TensorId>>,
) -> (out: (RT::Tensor, Tracked<RT::TensorPerm>))
    requires
        RT::paged_attention_numeric_domain(),
        runtime_attention_parameters_match(
            runtime, attention_kind,
            geometry,
            sliding_window as nat,
            attention_scale,
        ),
        RT::tensor_repr_2d(*qp, *q, q_repr),
        RT::kv_cache_tensor_repr(*kcp, *k_cache, k_cache_repr),
        RT::kv_cache_tensor_repr(*vcp, *v_cache, v_cache_repr),
        RT::block_table_repr(*btp, *block_table, bt_repr),
        RT::int_tensor_repr_1d(*cuqp, *cu_seqlens_q, cu_q_repr),
        RT::int_tensor_repr_1d(*cukp, *cu_seqlens_k, cu_k_repr),
        sliding_window > 0,
        RT::paged_attention_launch_ready(
            q_repr.len(), k_cache_repr, v_cache_repr,
            cu_q_repr, cu_k_repr,
            max_seqlen_q as nat, max_seqlen_k as nat, bt_repr,
        ),
    ensures
        RAW_ATTENTION::binding_valid(attention_kind, geometry, ATTN::effective_window(attention_kind, sliding_window as nat)),
        RAW_ATTENTION::layout_ready(q_repr, k_cache_repr, v_cache_repr, bt_repr, cu_q_repr, cu_k_repr, max_seqlen_q as nat, max_seqlen_k as nat, geometry),
        RAW_ATTENTION::numeric_requirements(q_repr, k_cache_repr, v_cache_repr, bt_repr, cu_q_repr, cu_k_repr, attention_kind, geometry, ATTN::scale_log2(AttentionParametersRepr { geometry, scale: attention_scale }), ATTN::effective_window(attention_kind, sliding_window as nat), KERNEL_SUPPORT::generated_kernel_allocation_cell()),
        ({ let (tensor, perm) = out;
            &&& tensor.id() != q.id()
            &&& !scope.contains(tensor.id())
            &&& RT::tensor_repr_2d(perm@, tensor,
                RAW_ATTENTION::raw_output(q_repr, k_cache_repr, v_cache_repr, bt_repr, cu_q_repr, cu_k_repr, max_seqlen_q as nat,
                    attention_kind, geometry, ATTN::scale_log2(AttentionParametersRepr { geometry, scale: attention_scale }), ATTN::effective_window(attention_kind, sliding_window as nat), KERNEL_SUPPORT::generated_kernel_allocation_cell()).unwrap())
        }),

{
    #[cfg(not(verus_only))]
    {
        let inner = pyo3::Python::with_gil(
            |py| -> pyo3::PyResult<pyo3::Py<pyo3::PyAny>> {
                Ok(runtime.handle.inner.bind(py).call_method1(
                    "paged_attention",
                    (
                        q.inner.bind(py),
                        k_cache.inner.bind(py),
                        v_cache.inner.bind(py),
                        block_table.inner.bind(py),
                        cu_seqlens_q.inner.bind(py),
                        cu_seqlens_k.inner.bind(py),
                        max_seqlen_q,
                        max_seqlen_k,
                        attention_kind_name(attention_kind),
                    ),
                )?.unbind())
            },
        )
        .expect("qualified four-norm paged attention failed");
        (RT::Tensor { inner }, Tracked::assume_new())
    }
    #[cfg(verus_only)]
    {
        unreachable!()
    }
}

} // verus!
// @kernel-bridge-end boundary::four_norm_gated_primitives::runtime
