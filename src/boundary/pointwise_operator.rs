//! Checked row bindings for pointwise operators. The raw operator symbols remain
//! distinct: matching relational laws do not imply matching numerical functions.
use vstd::prelude::*;
use crate::{boundary::scalar::{Scalar}, proof::tensor::types::{Tensor1D, Tensor2D}};
use crate::proof::tensor::shape as TS;
use crate::boundary::backend_certificates::{add as ADD, silu_mul as SILU,
    gelu_tanh_mul as GELU, scale as SCALE, softcap as SOFTCAP};

verus! {
pub open spec fn width(input: Tensor2D) -> nat {
    if input.len() > 0 { input[0].len() } else { 0 }
}

pub open spec fn layout(input: Tensor2D) -> bool {
    TS::tensor2d_shape(input, input.len(), width(input))
}

pub open spec fn binary_layout(input: Tensor2D, other: Tensor2D) -> bool {
    layout(input) && TS::tensor2d_shape(other, input.len(), width(input))
}

pub open spec fn add_raw_output(input: Tensor2D, other: Tensor2D) -> Tensor2D {
    ADD::row_projection_repr(input, other, width(input))
}

pub open spec fn add_row_output(row: Tensor1D, other: Tensor1D) -> Tensor1D {
    add_raw_output(seq![row], seq![other])[0]
}

pub open spec fn add_output(input: Tensor2D, other: Tensor2D) -> Tensor2D {
    Seq::new(input.len(), |r: int| add_row_output(input[r], other[r]))
}

pub proof fn checked_add_binding(input: Tensor2D, other: Tensor2D)
    requires binary_layout(input, other), input.len() > 0 ==> width(input) > 0,
    ensures add_raw_output(input, other) == add_output(input, other),
{
    assert forall|row: nat| row < input.len() implies
        #[trigger] ADD::row_projection_domain(input, other, input.len(), width(input), row) by {
        assert forall|r: int, c: int| 0 <= r < 1 && 0 <= c < width(input) implies
            #[trigger] input[row as int + r][c] == seq![input[row as int]][r][c]
                && #[trigger] other[row as int + r][c] == seq![other[row as int]][r][c] by {
            assert(r == 0);
        };
    };
    ADD::row_projection_launch_equivalence(input, other, input.len(), width(input));
    assert forall|r: int| 0 <= r < input.len() implies
        #[trigger] add_raw_output(input, other)[r] == add_row_output(input[r], other[r]) by {
        assert(input[r].len() == width(input));
    };
    assert(add_raw_output(input, other) =~= add_output(input, other));
}

pub open spec fn silu_mul_raw_output(input: Tensor2D, other: Tensor2D) -> Tensor2D {
    SILU::row_projection_repr(input, other, width(input))
}

pub open spec fn silu_mul_row_output(row: Tensor1D, other: Tensor1D) -> Tensor1D {
    silu_mul_raw_output(seq![row], seq![other])[0]
}

pub open spec fn silu_mul_output(input: Tensor2D, other: Tensor2D) -> Tensor2D {
    Seq::new(input.len(), |r: int| silu_mul_row_output(input[r], other[r]))
}

pub proof fn checked_silu_mul_binding(input: Tensor2D, other: Tensor2D)
    requires binary_layout(input, other),
    ensures silu_mul_raw_output(input, other) == silu_mul_output(input, other),
{
    assert forall|row: nat| row < input.len() implies
        #[trigger] SILU::row_projection_domain(input, other, input.len(), width(input), row) by {
        assert forall|r: int, c: int| 0 <= r < 1 && 0 <= c < width(input) implies
            #[trigger] input[row as int + r][c] == seq![input[row as int]][r][c]
                && #[trigger] other[row as int + r][c] == seq![other[row as int]][r][c] by {
            assert(r == 0);
        };
    };
    SILU::row_projection_launch_equivalence(input, other, input.len(), width(input));
    assert forall|r: int| 0 <= r < input.len() implies
        #[trigger] silu_mul_raw_output(input, other)[r] == silu_mul_row_output(input[r], other[r]) by {
        assert(input[r].len() == width(input));
    };
    assert(silu_mul_raw_output(input, other) =~= silu_mul_output(input, other));
}

pub open spec fn gelu_tanh_mul_raw_output(input: Tensor2D, other: Tensor2D) -> Tensor2D {
    GELU::row_projection_repr(input, other, width(input))
}

pub open spec fn gelu_tanh_mul_row_output(row: Tensor1D, other: Tensor1D) -> Tensor1D {
    gelu_tanh_mul_raw_output(seq![row], seq![other])[0]
}

pub open spec fn gelu_tanh_mul_output(input: Tensor2D, other: Tensor2D) -> Tensor2D {
    Seq::new(input.len(), |r: int| gelu_tanh_mul_row_output(input[r], other[r]))
}

pub proof fn checked_gelu_tanh_mul_binding(input: Tensor2D, other: Tensor2D)
    requires binary_layout(input, other),
    ensures gelu_tanh_mul_raw_output(input, other) == gelu_tanh_mul_output(input, other),
{
    assert forall|row: nat| row < input.len() implies
        #[trigger] GELU::row_projection_domain(input, other, input.len(), width(input), row) by {
        assert forall|r: int, c: int| 0 <= r < 1 && 0 <= c < width(input) implies
            #[trigger] input[row as int + r][c] == seq![input[row as int]][r][c]
                && #[trigger] other[row as int + r][c] == seq![other[row as int]][r][c] by {
            assert(r == 0);
        };
    };
    GELU::row_projection_launch_equivalence(input, other, input.len(), width(input));
    assert forall|r: int| 0 <= r < input.len() implies
        #[trigger] gelu_tanh_mul_raw_output(input, other)[r] == gelu_tanh_mul_row_output(input[r], other[r]) by {
        assert(input[r].len() == width(input));
    };
    assert(gelu_tanh_mul_raw_output(input, other) =~= gelu_tanh_mul_output(input, other));
}

pub open spec fn scale_raw_output(input: Tensor2D, other: Tensor1D) -> Tensor2D {
    SCALE::row_projection_repr(input, other, width(input))
}

pub open spec fn scale_row_output(row: Tensor1D, other: Tensor1D) -> Tensor1D {
    scale_raw_output(seq![row], other)[0]
}

pub open spec fn scale_output(input: Tensor2D, other: Tensor1D) -> Tensor2D {
    Seq::new(input.len(), |r: int| scale_row_output(input[r], other))
}

pub proof fn checked_scale_binding(input: Tensor2D, other: Tensor1D)
    requires layout(input), other.len() == 1,
    ensures scale_raw_output(input, other) == scale_output(input, other),
{
    assert forall|row: nat| row < input.len() implies
        #[trigger] SCALE::row_projection_domain(input, other, input.len(), width(input), row) by {
        assert forall|r: int, c: int| 0 <= r < 1 && 0 <= c < width(input) implies
            #[trigger] input[row as int + r][c] == seq![input[row as int]][r][c] by {
            assert(r == 0);
        };
    };
    SCALE::row_projection_launch_equivalence(input, other, input.len(), width(input));
    assert forall|r: int| 0 <= r < input.len() implies
        #[trigger] scale_raw_output(input, other)[r] == scale_row_output(input[r], other) by {
        assert(input[r].len() == width(input));
    };
    assert(scale_raw_output(input, other) =~= scale_output(input, other));
}

pub open spec fn softcap_raw_output(input: Tensor2D, other: Scalar) -> Tensor2D {
    SOFTCAP::row_projection_repr(input, other, width(input))
}

pub open spec fn softcap_row_output(row: Tensor1D, other: Scalar) -> Tensor1D {
    softcap_raw_output(seq![row], other)[0]
}

pub open spec fn softcap_output(input: Tensor2D, other: Scalar) -> Tensor2D {
    Seq::new(input.len(), |r: int| softcap_row_output(input[r], other))
}

pub proof fn checked_softcap_binding(input: Tensor2D, other: Scalar)
    requires layout(input),
    ensures softcap_raw_output(input, other) == softcap_output(input, other),
{
    assert forall|row: nat| row < input.len() implies
        #[trigger] SOFTCAP::row_projection_domain(input, other, input.len(), width(input), row) by {
        assert forall|r: int, c: int| 0 <= r < 1 && 0 <= c < width(input) implies
            #[trigger] input[row as int + r][c] == seq![input[row as int]][r][c] by {
            assert(r == 0);
        };
    };
    SOFTCAP::row_projection_launch_equivalence(input, other, input.len(), width(input));
    assert forall|r: int| 0 <= r < input.len() implies
        #[trigger] softcap_raw_output(input, other)[r] == softcap_row_output(input[r], other) by {
        assert(input[r].len() == width(input));
    };
    assert(softcap_raw_output(input, other) =~= softcap_output(input, other));
}

} // verus!
