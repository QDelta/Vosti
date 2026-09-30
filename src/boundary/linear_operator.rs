//! Checked engine-weight layout and row semantics for the generated raw kernel.
//! No matmul arithmetic or cross-configuration output equality is assumed.

use vstd::prelude::*;
use crate::{boundary::scalar::{Scalar}, proof::tensor::types::{Tensor1D, Tensor2D}};
use crate::proof::tensor::shape as TS;
use crate::boundary::backend_certificates::linear as RAW;
#[cfg(verus_only)]
use crate::proof::tensor::layout::transposed_weights;

verus! {

#[verifier::opaque]
pub open spec fn cell(row: Tensor1D, weights: Tensor2D, col: int) -> Scalar {
    RAW::row_projection_repr(seq![row], transposed_weights(weights, row.len()),
        row.len(), weights.len())[0][col]
}

pub open spec fn row_output(row: Tensor1D, weights: Tensor2D) -> Tensor1D {
    Seq::new(weights.len(), |col: int| cell(row, weights, col))
}

pub open spec fn output(input: Tensor2D, weights: Tensor2D) -> Tensor2D {
    Seq::new(input.len(), |r: int| row_output(input[r], weights))
}

pub open spec fn layout(input: Tensor2D, weights: Tensor2D, width: nat) -> bool {
    TS::tensor2d_shape(input, input.len(), width)
        && TS::tensor2d_shape(weights, weights.len(), width)
}

pub open spec fn raw_output(input: Tensor2D, weights: Tensor2D, width: nat) -> Tensor2D {
    RAW::row_projection_repr(input, transposed_weights(weights, width), width, weights.len())
}

pub proof fn checked_runtime_binding(input: Tensor2D, weights: Tensor2D, width: nat)
    requires layout(input, weights, width),
    ensures raw_output(input, weights, width) == output(input, weights),
{
    let transposed = transposed_weights(weights, width);
    assert(TS::tensor2d_shape(transposed, width, weights.len()));
    assert forall|row: nat| row < input.len() implies
        #[trigger] RAW::row_projection_domain(input, transposed, input.len(), width, weights.len(), row) by {
        reveal(RAW::row_projection_domain);
        assert forall|r: int| 0 <= r < input.len() implies
            (#[trigger] input[r]).len() == width by {};
        assert forall|r: int, c: int| 0 <= r < 1 && 0 <= c < width implies
            #[trigger] input[row as int + r][c] == seq![input[row as int]][r][c] by {
            assert(r == 0);
        };
    };
    RAW::row_projection_launch_equivalence(input, transposed, input.len(), width, weights.len());
    assert forall|r: int| 0 <= r < input.len() implies
        (#[trigger] raw_output(input, weights, width)[r]) == row_output(input[r], weights) by {
        assert(input[r].len() == width);
        let single = RAW::row_projection_repr(seq![input[r]], transposed, width, weights.len());
        assert(single[0].len() == weights.len());
        assert forall|c: int| 0 <= c < weights.len() implies
            (#[trigger] single[0][c]) == row_output(input[r], weights)[c] by {
            reveal(cell);
        };
        assert(single[0] =~= row_output(input[r], weights));
    };
    assert(raw_output(input, weights, width) =~= output(input, weights));
}

} // verus!
