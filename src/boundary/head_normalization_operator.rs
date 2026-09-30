//! Checked per-head normalization bindings. `offset` selects distinct
//! annotated operators, not alternative launch configurations.
use vstd::prelude::*;
use crate::{boundary::scalar::{Scalar}, proof::tensor::types::{Tensor1D, Tensor2D}};
use crate::proof::tensor::shape as TS;
use crate::boundary::backend_certificates::{head_rms_norm as DIRECT, offset_head_rms_norm as OFFSET};

verus! {

pub open spec fn width(input: Tensor2D) -> nat {
    if input.len() > 0 { input[0].len() } else { 0 }
}

pub open spec fn heads(row_width: nat, weight: Tensor1D) -> nat {
    if weight.len() > 0 { row_width / weight.len() } else { 0 }
}

pub open spec fn geometry_valid(d: nat, h: nat, offset: bool) -> bool {
    if offset { OFFSET::geometry_valid(d, h) } else { DIRECT::geometry_valid(d, h) }
}

pub open spec fn layout(input: Tensor2D, weight: Tensor1D, offset: bool) -> bool {
    input.len() == 0 || (
        weight.len() > 0
        && TS::tensor2d_shape(input, input.len(), width(input))
        && width(input) == heads(width(input), weight) * weight.len()
        && geometry_valid(weight.len(), heads(width(input), weight), offset)
    )
}

pub open spec fn raw_output(input: Tensor2D, weight: Tensor1D, eps: Scalar, offset: bool) -> Tensor2D {
    // No rows means no output cells, regardless of the deployed head geometry.
    if input.len() == 0 { Seq::empty() }
    else if offset {
        OFFSET::raw_output(input, weight, eps, weight.len(), heads(width(input), weight)).unwrap()
    } else {
        DIRECT::raw_output(input, weight, eps, weight.len(), heads(width(input), weight)).unwrap()
    }
}

pub open spec fn row_output(row: Tensor1D, weight: Tensor1D, eps: Scalar, offset: bool) -> Tensor1D {
    raw_output(seq![row], weight, eps, offset)[0]
}

pub open spec fn output(input: Tensor2D, weight: Tensor1D, eps: Scalar, offset: bool) -> Tensor2D {
    Seq::new(input.len(), |r: int| row_output(input[r], weight, eps, offset))
}

pub proof fn checked_binding(input: Tensor2D, weight: Tensor1D, eps: Scalar, offset: bool)
    requires layout(input, weight, offset),
    ensures raw_output(input, weight, eps, offset) == output(input, weight, eps, offset),
{
    if input.len() > 0 {
        let d = weight.len();
        let h = heads(width(input), weight);
        assert forall|row: nat, r: int, c: int| row < input.len() && 0 <= r < 1 && 0 <= c < h * d implies
            #[trigger] input[row as int + r][c] == seq![input[row as int]][r][c] by {
            assert(r == 0);
        };
        if offset {
            assert(OFFSET::launch_valid(input, weight, eps, d, h));
            OFFSET::checked_launch_equivalence(input, weight, eps, d, h);
        } else {
            assert(DIRECT::launch_valid(input, weight, eps, d, h));
            DIRECT::checked_launch_equivalence(input, weight, eps, d, h);
        }
        assert forall|r: int| 0 <= r < input.len() implies
            (#[trigger] raw_output(input, weight, eps, offset)[r]) == output(input, weight, eps, offset)[r] by {
            assert(input[r].len() == width(input));
        };
    }
    assert(raw_output(input, weight, eps, offset) =~= output(input, weight, eps, offset));
}

pub proof fn row_shape(input: Tensor2D, weight: Tensor1D, eps: Scalar, offset: bool, r: int)
    requires layout(input, weight, offset), 0 <= r < input.len(),
    ensures row_output(input[r], weight, eps, offset).len() == input[r].len(),
{
    assert(input[r].len() == width(input));
}

} // verus!
