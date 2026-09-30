//! Architecture-neutral properties of reusable dense-layer operations.
//!
//! These lemmas depend only on the semantic operation, not on any model-family
//! layer ordering. Composition proofs should depend here instead of reaching
//! into another architecture's proof development.

use crate::{types::*, proof::model::types::*, proof::tensor::types::*};
use crate::boundary::tensor_runtime as RT;
use vstd::prelude::*;

verus! {

pub proof fn linear_subrange_invariance(
    input: Tensor2D,
    weight: Tensor2D,
    start: int,
    end: int,
)
    requires 0 <= start <= end <= input.len(),
    ensures
        RT::linear_repr(input.subrange(start, end), weight)
            == RT::linear_repr(input, weight).subrange(start, end),
{
    reveal(RT::linear_repr);
    assert(RT::linear_repr(input.subrange(start, end), weight)
        =~= RT::linear_repr(input, weight).subrange(start, end));
}

pub proof fn qkv_linear_subrange_invariance(
    input: Tensor2D,
    q_weight: Tensor2D,
    k_weight: Tensor2D,
    v_weight: Tensor2D,
    start: int,
    end: int,
)
    requires 0 <= start <= end <= input.len(),
    ensures
        RT::qkv_linear_repr(
            input.subrange(start, end), q_weight, k_weight, v_weight,
        ).0 == RT::qkv_linear_repr(
            input, q_weight, k_weight, v_weight,
        ).0.subrange(start, end),
        RT::qkv_linear_repr(
            input.subrange(start, end), q_weight, k_weight, v_weight,
        ).1 == RT::qkv_linear_repr(
            input, q_weight, k_weight, v_weight,
        ).1.subrange(start, end),
        RT::qkv_linear_repr(
            input.subrange(start, end), q_weight, k_weight, v_weight,
        ).2 == RT::qkv_linear_repr(
            input, q_weight, k_weight, v_weight,
        ).2.subrange(start, end),
{
    reveal(RT::qkv_linear_repr);
    assert(RT::qkv_linear_repr(
        input.subrange(start, end), q_weight, k_weight, v_weight,
    ).0 =~= RT::qkv_linear_repr(
        input, q_weight, k_weight, v_weight,
    ).0.subrange(start, end));
    assert(RT::qkv_linear_repr(
        input.subrange(start, end), q_weight, k_weight, v_weight,
    ).1 =~= RT::qkv_linear_repr(
        input, q_weight, k_weight, v_weight,
    ).1.subrange(start, end));
    assert(RT::qkv_linear_repr(
        input.subrange(start, end), q_weight, k_weight, v_weight,
    ).2 =~= RT::qkv_linear_repr(
        input, q_weight, k_weight, v_weight,
    ).2.subrange(start, end));
}

pub proof fn view_as_kv_subrange_invariance(
    input: Tensor2D,
    start: int,
    end: int,
)
    requires 0 <= start <= end <= input.len(),
    ensures
        RT::view_as_kv_repr(input.subrange(start, end))
            == RT::view_as_kv_repr(input).subrange(start, end),
{
    reveal(RT::view_as_kv_repr);
    assert(RT::view_as_kv_repr(input.subrange(start, end))
        =~= RT::view_as_kv_repr(input).subrange(start, end));
}

} // verus!
