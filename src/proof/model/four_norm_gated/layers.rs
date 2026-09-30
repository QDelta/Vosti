//! Shared row-local stages of a four-norm, gated-GELU decoder.
//!
//! Attention and KV mutation deliberately remain outside these stages. Each
//! stage composes existing opaque operator cells; no numerical equivalence or
//! full-forward axiom is introduced. The same row projection lemmas cover
//! direct/offset norms, optional V normalization and optional layer scaling.

#[cfg(verus_only)]
use crate::model_config::FloatParameterBits;
use crate::boundary::dense_layer_primitives as LAYERS;
use crate::{types::*, proof::model::types::*, proof::tensor::types::*};
#[cfg(verus_only)]
use crate::boundary::tensor_runtime as RT;
#[cfg(verus_only)]
use crate::boundary::scalar::{float_parameter_scalar_repr, positive_float_parameter_valid};
use vstd::prelude::*;

verus! {

pub open spec fn norm_row(
    row: Tensor1D, weight: Tensor1D, policy: NormPolicyRepr,
) -> Tensor1D {
    match policy {
        NormPolicyRepr::UnitOffset => Seq::new(weight.len(), |col: int|
            RT::offset_rms_norm_kernel_cell_repr(row, weight, col)),
        NormPolicyRepr::Direct(epsilon) => RT::rms_norm_kernel_row_repr(
            row, weight, float_parameter_scalar_repr(epsilon)),
    }
}

pub open spec fn head_norm_row(
    row: Tensor1D, weight: Tensor1D, policy: NormPolicyRepr,
) -> Tensor1D {
    match policy {
        NormPolicyRepr::UnitOffset => LAYERS::qk_norm_row_repr(row, weight),
        NormPolicyRepr::Direct(epsilon) => RT::head_rms_norm_kernel_row_repr(
            row, weight, float_parameter_scalar_repr(epsilon), row.len()),
    }
}

pub open spec fn value_row(
    row: Tensor1D, head_dim: nat, epsilon: Option<FloatParameterBits>,
) -> Tensor1D {
    let normalized = match epsilon {
        None => row,
        Some(eps) => {
            let one = float_parameter_scalar_repr(
                FloatParameterBits { bits: 4_607_182_418_800_017_408 });
            head_norm_row(row, Seq::new(head_dim, |_col: int| one), NormPolicyRepr::Direct(eps))
        },
    };
    RT::view_as_kv_row_repr(normalized)
}

pub open spec fn add_row(left: Tensor1D, right: Tensor1D) -> Tensor1D {
    Seq::new(left.len(), |col: int| RT::add_kernel_cell_repr(left, right, col))
}

pub open spec fn scale_row(row: Tensor1D, weight: Option<Tensor1D>) -> Tensor1D {
    match weight {
        None => row,
        Some(scalar) => Seq::new(row.len(), |col: int|
            RT::scale_kernel_cell_repr(row, scalar, col)),
    }
}

// This is the row-wise decomposition of the QKV operator, not an assertion
// that the fused projection equals three different matmul operators. Shared
// K/V projection weights also do not imply equality of normalized cache rows.
#[verifier::opaque]
pub open spec fn attention_pre_store_row(
    layer: FourNormGatedLayerRepr, hidden: Tensor1D, position: int,
) -> (Tensor1D, Tensor1D, Tensor1D) {
    let common = layer.common;
    let input = norm_row(hidden, common.input_norm, layer.norm);
    let q = Seq::new(common.q_proj.len(), |col: int|
        RT::qkv_q_kernel_cell_repr(input, common.q_proj, common.k_proj, common.v_proj, col));
    let k = Seq::new(common.k_proj.len(), |col: int|
        RT::qkv_k_kernel_cell_repr(input, common.q_proj, common.k_proj, common.v_proj, col));
    let v = Seq::new(common.k_proj.len(), |col: int|
        RT::qkv_v_kernel_cell_repr(input, common.q_proj, common.k_proj, common.v_proj, col));
    let nq = head_norm_row(q, rms_q_norm_weight(common.qk_norm), layer.qk_norm);
    let nk = head_norm_row(k, rms_k_norm_weight(common.qk_norm), layer.qk_norm);
    (
        LAYERS::rotary_row_with_config_repr(position, nq,
            LAYERS::layer_attention_geometry_repr(common).num_attention_heads, common.head_dim, layer.rotary),
        LAYERS::rotary_row_with_config_repr(position, nk,
            LAYERS::layer_attention_geometry_repr(common).num_key_value_heads, common.head_dim, layer.rotary),
        value_row(v, common.head_dim, layer.value_norm_epsilon),
    )
}

pub open spec fn attention_pre_store_repr(
    layer: FourNormGatedLayerRepr, hidden: Tensor2D, positions: IntTensor1D,
) -> (Tensor2D, Tensor2D, Tensor2D)
    recommends hidden.len() == positions.len(),
{
    (
        Seq::new(hidden.len(), |i: int| attention_pre_store_row(layer, hidden[i], positions[i]).0),
        Seq::new(hidden.len(), |i: int| attention_pre_store_row(layer, hidden[i], positions[i]).1),
        Seq::new(hidden.len(), |i: int| attention_pre_store_row(layer, hidden[i], positions[i]).2),
    )
}

#[verifier::opaque]
pub open spec fn post_attention_and_mlp_row(
    layer: FourNormGatedLayerRepr, residual: Tensor1D, attended: Tensor1D,
) -> Tensor1D {
    let common = layer.common;
    let merged = LAYERS::merge_attention_heads_row_repr(attended);
    let projected = RT::linear_kernel_row_repr(merged, common.o_proj);
    let after_attention = add_row(residual, norm_row(projected, common.post_attn_norm, layer.norm));
    let ff_input = norm_row(after_attention, layer.pre_feedforward_norm, layer.norm);
    let gate_up = RT::linear_kernel_row_repr(ff_input, common.gate_up_proj);
    let half = (gate_up.len() / 2) as int;
    let gate = gate_up.subrange(0, half);
    let up = gate_up.subrange(half, 2 * half);
    let activated = Seq::new(gate.len(), |col: int|
        RT::gelu_tanh_mul_kernel_cell_repr(gate, up, col));
    let down = RT::linear_kernel_row_repr(activated, common.down_proj);
    let post_ff = norm_row(down, layer.post_feedforward_norm, layer.norm);
    scale_row(add_row(after_attention, post_ff), layer.layer_scale)
}

pub open spec fn post_attention_and_mlp_repr(
    layer: FourNormGatedLayerRepr, residual: Tensor2D, attended: Tensor2D,
) -> Tensor2D
    recommends residual.len() == attended.len(),
{
    Seq::new(residual.len(), |i: int| post_attention_and_mlp_row(layer, residual[i], attended[i]))
}

pub open spec fn logits_softcap_repr(
    logits: Tensor2D, cap: Option<FloatParameterBits>,
) -> Tensor2D {
    match cap {
        None => logits,
        Some(value) => Seq::new(logits.len(), |i: int|
            Seq::new(logits[i].len(), |col: int|
                RT::softcap_kernel_cell_repr(logits[i], float_parameter_scalar_repr(value), col))),
    }
}

pub open spec fn final_logits_repr(
    hidden: Tensor2D, weight: Tensor1D, head: Tensor2D,
    norm: NormPolicyRepr, cap: Option<FloatParameterBits>,
) -> Tensor2D {
    logits_softcap_repr(Seq::new(hidden.len(), |i: int|
        RT::linear_kernel_row_repr(norm_row(hidden[i], weight, norm), head)), cap)
}

pub broadcast proof fn lemma_final_logits_shape(
    hidden: Tensor2D, weight: Tensor1D, head: Tensor2D,
    norm: NormPolicyRepr, cap: Option<FloatParameterBits>,
)
    ensures (#[trigger] final_logits_repr(hidden, weight, head, norm, cap)).len() == hidden.len(),
{}

pub proof fn final_logits_subrange_invariance(
    hidden: Tensor2D, weight: Tensor1D, head: Tensor2D,
    norm: NormPolicyRepr, cap: Option<FloatParameterBits>, a: int, b: int,
)
    requires 0 <= a <= b <= hidden.len(),
    ensures final_logits_repr(hidden.subrange(a, b), weight, head, norm, cap)
        == final_logits_repr(hidden, weight, head, norm, cap).subrange(a, b),
{
    let projected = Seq::new(hidden.len(), |i: int|
        RT::linear_kernel_row_repr(norm_row(hidden[i], weight, norm), head));
    assert(Seq::new(hidden.subrange(a, b).len(), |i: int|
        RT::linear_kernel_row_repr(norm_row(hidden.subrange(a, b)[i], weight, norm), head))
        =~= projected.subrange(a, b));
    logits_softcap_subrange_invariance(projected, cap, a, b);
}

pub broadcast proof fn lemma_attention_pre_store_repr_shape(
    layer: FourNormGatedLayerRepr, hidden: Tensor2D, positions: IntTensor1D,
)
    ensures
        (#[trigger] attention_pre_store_repr(layer, hidden, positions)).0.len() == hidden.len(),
        attention_pre_store_repr(layer, hidden, positions).1.len() == hidden.len(),
        attention_pre_store_repr(layer, hidden, positions).2.len() == hidden.len(),
{}

pub proof fn attention_pre_store_subrange_invariance(
    layer: FourNormGatedLayerRepr, hidden: Tensor2D, positions: IntTensor1D, a: int, b: int,
)
    requires hidden.len() == positions.len(), 0 <= a <= b <= hidden.len(),
    ensures
        attention_pre_store_repr(layer, hidden.subrange(a, b), positions.subrange(a, b)).0
            == attention_pre_store_repr(layer, hidden, positions).0.subrange(a, b),
        attention_pre_store_repr(layer, hidden.subrange(a, b), positions.subrange(a, b)).1
            == attention_pre_store_repr(layer, hidden, positions).1.subrange(a, b),
        attention_pre_store_repr(layer, hidden.subrange(a, b), positions.subrange(a, b)).2
            == attention_pre_store_repr(layer, hidden, positions).2.subrange(a, b),
{
    let left = attention_pre_store_repr(layer, hidden.subrange(a, b), positions.subrange(a, b));
    let right = attention_pre_store_repr(layer, hidden, positions);
    assert(left.0 =~= right.0.subrange(a, b));
    assert(left.1 =~= right.1.subrange(a, b));
    assert(left.2 =~= right.2.subrange(a, b));
}

pub proof fn post_attention_and_mlp_subrange_invariance(
    layer: FourNormGatedLayerRepr, residual: Tensor2D, attended: Tensor2D, a: int, b: int,
)
    requires residual.len() == attended.len(), 0 <= a <= b <= residual.len(),
    ensures post_attention_and_mlp_repr(layer, residual.subrange(a, b), attended.subrange(a, b))
        == post_attention_and_mlp_repr(layer, residual, attended).subrange(a, b),
{
    assert(post_attention_and_mlp_repr(layer, residual.subrange(a, b), attended.subrange(a, b))
        =~= post_attention_and_mlp_repr(layer, residual, attended).subrange(a, b));
}

pub proof fn logits_softcap_subrange_invariance(
    logits: Tensor2D, cap: Option<FloatParameterBits>, a: int, b: int,
)
    requires 0 <= a <= b <= logits.len(),
    ensures logits_softcap_repr(logits.subrange(a, b), cap)
        == logits_softcap_repr(logits, cap).subrange(a, b),
{
    assert(logits_softcap_repr(logits.subrange(a, b), cap)
        =~= logits_softcap_repr(logits, cap).subrange(a, b));
}

} // verus!
