//! Checked layout and row projection of one fused, fixed QKV implementation.
//! This does not equate fused QKV with three independent linear operations.

use vstd::prelude::*;
use crate::{proof::tensor::types::{Tensor1D, Tensor2D}};
use crate::proof::tensor::shape as TS;
#[cfg(verus_only)]
use crate::proof::tensor::layout::transposed_weights;
use crate::boundary::backend_certificates::qkv as RAW;

verus! {

pub open spec fn layout(
    input: Tensor2D, qw: Tensor2D, kw: Tensor2D, vw: Tensor2D, width: nat,
) -> bool {
    TS::tensor2d_shape(input, input.len(), width)
        && TS::tensor2d_shape(qw, qw.len(), width)
        && TS::tensor2d_shape(kw, kw.len(), width)
        && TS::tensor2d_shape(vw, kw.len(), width)
        && qw.len() >= kw.len() && kw.len() > 0 && width > 0
}

pub open spec fn raw_output(
    input: Tensor2D, qw: Tensor2D, kw: Tensor2D, vw: Tensor2D, width: nat,
) -> (Tensor2D, Tensor2D, Tensor2D) {
    let kt = transposed_weights(kw, width);
    let qt = transposed_weights(qw, width);
    let vt = transposed_weights(vw, width);
    (
        RAW::row_projection_oq_repr(input, kt, qt, vt, width, kw.len(), qw.len()),
        RAW::row_projection_ok_repr(input, kt, qt, vt, width, kw.len(), qw.len()),
        RAW::row_projection_ov_repr(input, kt, qt, vt, width, kw.len(), qw.len()),
    )
}

pub open spec fn row_output(
    row: Tensor1D, qw: Tensor2D, kw: Tensor2D, vw: Tensor2D,
) -> (Tensor1D, Tensor1D, Tensor1D) {
    let raw = raw_output(seq![row], qw, kw, vw, row.len());
    (raw.0[0], raw.1[0], raw.2[0])
}

pub open spec fn output(
    input: Tensor2D, qw: Tensor2D, kw: Tensor2D, vw: Tensor2D,
) -> (Tensor2D, Tensor2D, Tensor2D) {
    (
        Seq::new(input.len(), |r: int| row_output(input[r], qw, kw, vw).0),
        Seq::new(input.len(), |r: int| row_output(input[r], qw, kw, vw).1),
        Seq::new(input.len(), |r: int| row_output(input[r], qw, kw, vw).2),
    )
}

pub proof fn checked_runtime_binding(
    input: Tensor2D, qw: Tensor2D, kw: Tensor2D, vw: Tensor2D, width: nat,
)
    requires layout(input, qw, kw, vw, width),
    ensures raw_output(input, qw, kw, vw, width) == output(input, qw, kw, vw),
{
    let kt = transposed_weights(kw, width);
    let qt = transposed_weights(qw, width);
    let vt = transposed_weights(vw, width);
    assert(TS::tensor2d_shape(kt, width, kw.len()));
    assert(TS::tensor2d_shape(qt, width, qw.len()));
    assert(TS::tensor2d_shape(vt, width, kw.len()));
    assert forall|row: nat| row < input.len() implies
        #[trigger] RAW::row_projection_domain(input, kt, qt, vt, input.len(), width, kw.len(), qw.len(), row) by {
        reveal(RAW::row_projection_domain);
        assert forall|r: int| 0 <= r < input.len() implies
            (#[trigger] input[r]).len() == width by {};
        assert forall|r: int, c: int| 0 <= r < 1 && 0 <= c < width implies
            #[trigger] input[row as int + r][c] == seq![input[row as int]][r][c] by {
            assert(r == 0);
        };
    };
    RAW::row_projection_launch_equivalence(input, kt, qt, vt, input.len(), width, kw.len(), qw.len());
    let raw = raw_output(input, qw, kw, vw, width);
    let mapped = output(input, qw, kw, vw);
    assert forall|r: int| 0 <= r < input.len() implies
        (#[trigger] raw.0[r], #[trigger] raw.1[r], #[trigger] raw.2[r])
            == (mapped.0[r], mapped.1[r], mapped.2[r]) by {
        assert(input[r].len() == width);
    };
    assert(raw.0 =~= mapped.0);
    assert(raw.1 =~= mapped.1);
    assert(raw.2 =~= mapped.2);
}

} // verus!
