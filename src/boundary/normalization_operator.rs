//! Checked row adapters for qualified normalization kernels.
//! Scalar parameters stay explicit; no numerical equivalence between fused
//! residual normalization and separate addition/normalization is assumed.

use vstd::prelude::*;
use crate::{boundary::scalar::{Scalar}, proof::tensor::types::{Tensor1D, Tensor2D}};
use crate::proof::tensor::shape as TS;
use crate::boundary::backend_certificates::{rms_norm as RMS, residual_rms_norm as RES};
use crate::boundary::backend_certificates::offset_rms_norm as OFFSET;

verus! {

pub open spec fn layout(input: Tensor2D, weight: Tensor1D) -> bool {
    TS::tensor2d_shape(input, input.len(), weight.len())
}

pub open spec fn rms_raw_output(input: Tensor2D, weight: Tensor1D, eps: Scalar) -> Tensor2D {
    RMS::row_projection_repr(input, weight, eps, weight.len())
}

pub open spec fn rms_row_output(row: Tensor1D, weight: Tensor1D, eps: Scalar) -> Tensor1D {
    rms_raw_output(seq![row], weight, eps)[0]
}

pub open spec fn rms_output(input: Tensor2D, weight: Tensor1D, eps: Scalar) -> Tensor2D {
    Seq::new(input.len(), |r: int| rms_row_output(input[r], weight, eps))
}

pub proof fn checked_rms_binding(input: Tensor2D, weight: Tensor1D, eps: Scalar)
    requires layout(input, weight),
    ensures rms_raw_output(input, weight, eps) == rms_output(input, weight, eps),
{
    assert forall|row: nat| row < input.len() implies
        #[trigger] RMS::row_projection_domain(input, weight, eps, input.len(), weight.len(), row) by {
        assert forall|r: int, c: int| 0 <= r < 1 && 0 <= c < weight.len() implies
            #[trigger] input[row as int + r][c] == seq![input[row as int]][r][c] by {
            assert(r == 0);
        };
    };
    RMS::row_projection_launch_equivalence(input, weight, eps, input.len(), weight.len());
    assert(rms_raw_output(input, weight, eps) =~= rms_output(input, weight, eps));
}

pub open spec fn residual_layout(input: Tensor2D, residual: Tensor2D, weight: Tensor1D) -> bool {
    layout(input, weight) && TS::tensor2d_shape(residual, input.len(), weight.len())
}

pub open spec fn offset_raw_output(input: Tensor2D, weight: Tensor1D, eps: Scalar) -> Tensor2D {
    OFFSET::row_projection_repr(input, weight, eps, weight.len())
}

pub open spec fn offset_row_output(row: Tensor1D, weight: Tensor1D, eps: Scalar) -> Tensor1D {
    offset_raw_output(seq![row], weight, eps)[0]
}

pub open spec fn offset_output(input: Tensor2D, weight: Tensor1D, eps: Scalar) -> Tensor2D {
    Seq::new(input.len(), |r: int| offset_row_output(input[r], weight, eps))
}

pub proof fn checked_offset_binding(input: Tensor2D, weight: Tensor1D, eps: Scalar)
    requires layout(input, weight),
    ensures offset_raw_output(input, weight, eps) == offset_output(input, weight, eps),
{
    assert forall|row: nat| row < input.len() implies
        #[trigger] OFFSET::row_projection_domain(input, weight, eps, input.len(), weight.len(), row) by {
        assert forall|r: int, c: int| 0 <= r < 1 && 0 <= c < weight.len() implies
            #[trigger] input[row as int + r][c] == seq![input[row as int]][r][c] by {
            assert(r == 0);
        };
    };
    OFFSET::row_projection_launch_equivalence(input, weight, eps, input.len(), weight.len());
    assert(offset_raw_output(input, weight, eps) =~= offset_output(input, weight, eps));
}

pub open spec fn residual_raw_output(
    input: Tensor2D, residual: Tensor2D, weight: Tensor1D, eps: Scalar,
) -> (Tensor2D, Tensor2D) {
    (
        RES::row_projection_o_repr(residual, input, weight, eps, weight.len()),
        RES::row_projection_residual_out_repr(residual, input, weight, eps, weight.len()),
    )
}

pub open spec fn residual_row_output(
    row: Tensor1D, residual: Tensor1D, weight: Tensor1D, eps: Scalar,
) -> (Tensor1D, Tensor1D) {
    let raw = residual_raw_output(seq![row], seq![residual], weight, eps);
    (raw.0[0], raw.1[0])
}

pub open spec fn residual_output(
    input: Tensor2D, residual: Tensor2D, weight: Tensor1D, eps: Scalar,
) -> (Tensor2D, Tensor2D) {
    (
        Seq::new(input.len(), |r: int| residual_row_output(input[r], residual[r], weight, eps).0),
        Seq::new(input.len(), |r: int| residual_row_output(input[r], residual[r], weight, eps).1),
    )
}

pub proof fn checked_residual_binding(
    input: Tensor2D, residual: Tensor2D, weight: Tensor1D, eps: Scalar,
)
    requires residual_layout(input, residual, weight),
    ensures residual_raw_output(input, residual, weight, eps) == residual_output(input, residual, weight, eps),
{
    assert forall|row: nat| row < input.len() implies
        #[trigger] RES::row_projection_domain(residual, input, weight, eps, input.len(), weight.len(), row) by {
        assert forall|r: int, c: int| 0 <= r < 1 && 0 <= c < weight.len() implies
            #[trigger] input[row as int + r][c] == seq![input[row as int]][r][c]
                && #[trigger] residual[row as int + r][c] == seq![residual[row as int]][r][c] by {
            assert(r == 0);
        };
    };
    RES::row_projection_launch_equivalence(residual, input, weight, eps, input.len(), weight.len());
    let raw = residual_raw_output(input, residual, weight, eps);
    let mapped = residual_output(input, residual, weight, eps);
    assert(raw.0 =~= mapped.0);
    assert(raw.1 =~= mapped.1);
}

} // verus!
