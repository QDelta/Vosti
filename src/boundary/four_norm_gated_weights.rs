//! Neutral physical roles for four-norm gated decoder layers.
//!
//! These records do not admit a family or create tensor permissions. Family
//! loaders and checked runtime binding must establish the concrete contract.

use crate::model_config::AttentionKind;
use crate::{types::*, proof::model::types::*, proof::tensor::types::*};
use crate::boundary::tensor_runtime as RT;
use crate::proof::tensor::shape as TS;
use vstd::prelude::*;

verus! {

// @kernel-bridge-begin boundary::four_norm_gated_weights::layer_contract

// Common read-only roles of a four-norm gated decoder. An optional scalar
// is a real tensor role only in models whose layer graph uses it.
pub struct FourNormGatedLayerWeights {
    pub input_norm: RT::Tensor,
    pub q_proj: RT::Tensor,
    pub k_proj: RT::Tensor,
    pub v_proj: RT::Tensor,
    pub q_norm: RT::Tensor,
    pub k_norm: RT::Tensor,
    pub o_proj: RT::Tensor,
    pub post_attn_norm: RT::Tensor,
    pub pre_feedforward_norm: RT::Tensor,
    pub gate_up_proj: RT::Tensor,
    pub down_proj: RT::Tensor,
    pub post_feedforward_norm: RT::Tensor,
    pub attention_kind: AttentionKind,
    pub layer_scale: Option<RT::Tensor>,
}

pub struct FourNormGatedModelWeights<C> {
    pub embed_weight: RT::Tensor,
    pub layers: Vec<FourNormGatedLayerWeights>,
    pub final_norm: RT::Tensor,
    pub lm_head: RT::Tensor,
    pub config: C,
}

// Explicit immutable permissions need no new opaque borrowing axioms. The
// family binder validates the tensors before creating this tracked record.
pub tracked struct FourNormGatedModelWeightsPerms {
    pub tracked embed_weight: RT::TensorPerm,
    pub tracked layers: Map<int, FourNormGatedLayerWeightsPerms>,
    pub tracked final_norm: RT::TensorPerm,
    pub tracked lm_head: RT::TensorPerm,
}

impl FourNormGatedModelWeightsPerms {
    pub proof fn borrowed_layer_permissions<'a>(tracked &'a self, count: int)
        -> (tracked refs: Map<int, &'a FourNormGatedLayerWeightsPerms>)
        requires
            count >= 0,
            forall|i: int| 0 <= i < count ==> #[trigger] self.layers.dom().contains(i),
        ensures
            forall|i: int| #[trigger] refs.dom().contains(i) <==> 0 <= i < count,
            forall|i: int| 0 <= i < count ==> *#[trigger] refs[i] == self.layers[i],
        decreases count,
    {
        if count == 0 {
            Map::tracked_empty()
        } else {
            let tracked mut refs = self.borrowed_layer_permissions(count - 1);
            let tracked layer = self.tracked_borrow_layer(count - 1);
            refs.tracked_insert(count - 1, layer);
            refs
        }
    }

    pub proof fn tracked_borrow_layer<'a>(tracked &'a self, i: int)
        -> (tracked out: &'a FourNormGatedLayerWeightsPerms)
        requires self.layers.dom().contains(i),
        ensures *out == self.layers[i],
    {
        self.layers.tracked_borrow(i)
    }
}

pub tracked struct FourNormGatedLayerWeightsPerms {
    pub tracked input_norm: RT::TensorPerm,
    pub tracked q_proj: RT::TensorPerm,
    pub tracked k_proj: RT::TensorPerm,
    pub tracked v_proj: RT::TensorPerm,
    pub tracked q_norm: RT::TensorPerm,
    pub tracked k_norm: RT::TensorPerm,
    pub tracked o_proj: RT::TensorPerm,
    pub tracked post_attn_norm: RT::TensorPerm,
    pub tracked pre_feedforward_norm: RT::TensorPerm,
    pub tracked gate_up_proj: RT::TensorPerm,
    pub tracked down_proj: RT::TensorPerm,
    pub tracked post_feedforward_norm: RT::TensorPerm,
    pub tracked layer_scale: Option<RT::TensorPerm>,
}

pub open spec fn layer_weights_valid(
    weights: &FourNormGatedLayerWeights,
    perms: &FourNormGatedLayerWeightsPerms,
) -> bool {
    weights.input_norm.id() == perms.input_norm.id()
    && weights.q_proj.id() == perms.q_proj.id()
    && weights.k_proj.id() == perms.k_proj.id()
    && weights.v_proj.id() == perms.v_proj.id()
    && weights.q_norm.id() == perms.q_norm.id()
    && weights.k_norm.id() == perms.k_norm.id()
    && weights.o_proj.id() == perms.o_proj.id()
    && weights.post_attn_norm.id() == perms.post_attn_norm.id()
    && weights.pre_feedforward_norm.id() == perms.pre_feedforward_norm.id()
    && weights.gate_up_proj.id() == perms.gate_up_proj.id()
    && weights.down_proj.id() == perms.down_proj.id()
    && weights.post_feedforward_norm.id() == perms.post_feedforward_norm.id()
    && TS::rectangular(perms.q_proj.repr_2d())
    && TS::rectangular(perms.k_proj.repr_2d())
    && TS::rectangular(perms.v_proj.repr_2d())
    && TS::rectangular(perms.o_proj.repr_2d())
    && TS::rectangular(perms.gate_up_proj.repr_2d())
    && perms.gate_up_proj.repr_2d().len() % 2 == 0
    && TS::rectangular(perms.down_proj.repr_2d())
    && match (&weights.layer_scale, &perms.layer_scale) {
        (Some(weight), Some(perm)) => weight.id() == perm.id()
            && perm.repr_1d().len() == 1,
        (None, None) => true,
        _ => false,
    }
}

pub open spec fn layer_weights_common_repr_of(
    perms: &FourNormGatedLayerWeightsPerms,
) -> LayerWeightsRepr {
    LayerWeightsRepr {
        input_norm: perms.input_norm.repr_1d(),
        q_proj: perms.q_proj.repr_2d(),
        k_proj: perms.k_proj.repr_2d(),
        v_proj: perms.v_proj.repr_2d(),
        head_dim: perms.q_norm.repr_1d().len(),
        qk_norm: QkNormWeightsRepr::RmsNorm {
            q_weight: perms.q_norm.repr_1d(),
            k_weight: perms.k_norm.repr_1d(),
        },
        o_proj: perms.o_proj.repr_2d(),
        post_attn_norm: perms.post_attn_norm.repr_1d(),
        gate_up_proj: perms.gate_up_proj.repr_2d(),
        down_proj: perms.down_proj.repr_2d(),
    }
}

pub open spec fn layer_scale_repr(perms: &FourNormGatedLayerWeightsPerms)
    -> Option<Tensor1D>
{
    match &perms.layer_scale {
        Some(value) => Some(value.repr_1d()),
        None => None,
    }
}

pub open spec fn layer_shapes_valid(
    perms: &FourNormGatedLayerWeightsPerms,
    hidden_size: nat, intermediate_size: nat, attention: AttentionGeometryRepr,
) -> bool {
    let q_width = attention.num_attention_heads * attention.head_dim;
    let kv_width = attention.num_key_value_heads * attention.head_dim;
    perms.input_norm.repr_1d().len() == hidden_size
    && TS::tensor2d_shape(perms.q_proj.repr_2d(), q_width, hidden_size)
    && TS::tensor2d_shape(perms.k_proj.repr_2d(), kv_width, hidden_size)
    && TS::tensor2d_shape(perms.v_proj.repr_2d(), kv_width, hidden_size)
    && perms.q_norm.repr_1d().len() == attention.head_dim
    && perms.k_norm.repr_1d().len() == attention.head_dim
    && TS::tensor2d_shape(perms.o_proj.repr_2d(), hidden_size, q_width)
    && perms.post_attn_norm.repr_1d().len() == hidden_size
    && perms.pre_feedforward_norm.repr_1d().len() == hidden_size
    && TS::tensor2d_shape(perms.gate_up_proj.repr_2d(), 2 * intermediate_size, hidden_size)
    && TS::tensor2d_shape(perms.down_proj.repr_2d(), hidden_size, intermediate_size)
    && perms.post_feedforward_norm.repr_1d().len() == hidden_size
}

pub open spec fn model_roles_bound<C>(
    weights: &FourNormGatedModelWeights<C>, perms: &FourNormGatedModelWeightsPerms,
) -> bool {
    weights.embed_weight.id() == perms.embed_weight.id()
    && weights.final_norm.id() == perms.final_norm.id()
    && weights.lm_head.id() == perms.lm_head.id()
    && (forall|i: int| #[trigger] perms.layers.dom().contains(i)
        <==> 0 <= i < weights.layers.len())
    && forall|i: int| 0 <= i < weights.layers.len() ==>
        #[trigger] layer_weights_valid(&weights.layers[i], &perms.layers[i])
}

// @kernel-bridge-end boundary::four_norm_gated_weights::layer_contract

} // verus!

// @kernel-bridge-begin boundary::four_norm_gated_weights::python_roles
#[cfg(not(verus_only))]
pub(crate) fn python_layer_roles<'py>(
    py: pyo3::Python<'py>, layer: &FourNormGatedLayerWeights, require_scale: bool,
) -> pyo3::PyResult<pyo3::Bound<'py, pyo3::types::PyTuple>> {
    let mut roles = vec![
        layer.input_norm.inner.bind(py), layer.q_proj.inner.bind(py),
        layer.k_proj.inner.bind(py), layer.v_proj.inner.bind(py),
        layer.q_norm.inner.bind(py), layer.k_norm.inner.bind(py),
        layer.o_proj.inner.bind(py), layer.post_attn_norm.inner.bind(py),
        layer.pre_feedforward_norm.inner.bind(py), layer.gate_up_proj.inner.bind(py),
        layer.down_proj.inner.bind(py), layer.post_feedforward_norm.inner.bind(py),
    ];
    match (&layer.layer_scale, require_scale) {
        (Some(scale), true) => roles.push(scale.inner.bind(py)),
        (None, false) => {},
        _ => return Err(pyo3::exceptions::PyValueError::new_err(
            "four-norm layer output-scalar role differs from family policy")),
    }
    Ok(pyo3::types::PyTuple::new_bound(py, roles))
}
// @kernel-bridge-end boundary::four_norm_gated_weights::python_roles

// @kernel-bridge-begin boundary::four_norm_gated_weights::checkpoint_roles
#[cfg(not(verus_only))]
pub(crate) fn from_checkpoint_roles(
    mut roles: Vec<RT::Tensor>, attention_kind: &str, require_scale: bool,
) -> pyo3::PyResult<FourNormGatedLayerWeights> {
    let expected = if require_scale { 13 } else { 12 };
    if roles.len() != expected {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "four-norm checkpoint layer requires {expected} roles, got {}", roles.len())));
    }
    let attention_kind = match attention_kind {
        "sliding_attention" => AttentionKind::SlidingWindow,
        "full_attention" => AttentionKind::Full,
        kind => return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "unsupported four-norm attention kind: {kind}"))),
    };
    let layer_scale = if require_scale { roles.pop() } else { None };
    let [input_norm, q_proj, k_proj, v_proj, q_norm, k_norm, o_proj,
        post_attn_norm, pre_feedforward_norm, gate_up_proj, down_proj, post_feedforward_norm]:
        [RT::Tensor; 12] = roles.try_into().unwrap_or_else(|_| unreachable!());
    Ok(FourNormGatedLayerWeights {
        input_norm, q_proj, k_proj, v_proj, q_norm, k_norm, o_proj,
        post_attn_norm, pre_feedforward_norm, gate_up_proj, down_proj,
        post_feedforward_norm, attention_kind, layer_scale,
    })
}
// @kernel-bridge-end boundary::four_norm_gated_weights::checkpoint_roles
