//! Checked binding of the annotated rotation kernel. Table construction and
//! token/head reshapes remain separate from this raw tensor operator.
use vstd::prelude::*;
use crate::{proof::tensor::types::{Tensor1D, Tensor2D}};
use crate::proof::tensor::shape as TS;
use crate::boundary::backend_certificates::rotary as RAW;

verus! {

pub open spec fn width(input: Tensor2D) -> nat {
    if input.len() > 0 { input[0].len() } else { 0 }
}

pub open spec fn layout(input: Tensor2D, cos: Tensor2D, sin: Tensor2D) -> bool {
    cos.len() == input.len() && sin.len() == input.len()
    && (input.len() == 0 || (
        TS::tensor2d_shape(input, input.len(), width(input))
        && TS::tensor2d_shape(cos, input.len(), width(cos))
        && TS::tensor2d_shape(sin, input.len(), width(cos))
        && RAW::geometry_valid(width(input), width(cos))
    ))
}

pub open spec fn raw_output(input: Tensor2D, cos: Tensor2D, sin: Tensor2D) -> Tensor2D {
    if input.len() == 0 { Seq::empty() }
    else { RAW::raw_output(cos, sin, input, width(input), width(cos)).unwrap() }
}

pub open spec fn row_output(row: Tensor1D, cos: Tensor1D, sin: Tensor1D) -> Tensor1D {
    raw_output(seq![row], seq![cos], seq![sin])[0]
}

pub open spec fn output(input: Tensor2D, cos: Tensor2D, sin: Tensor2D) -> Tensor2D {
    Seq::new(input.len(), |r: int| row_output(input[r], cos[r], sin[r]))
}

pub proof fn checked_binding(input: Tensor2D, cos: Tensor2D, sin: Tensor2D)
    requires layout(input, cos, sin),
    ensures raw_output(input, cos, sin) == output(input, cos, sin),
{
    if input.len() > 0 {
        assert forall|row: nat, r: int, c: int| row < input.len() && 0 <= r < 1 && 0 <= c < width(input) implies
            #[trigger] input[row as int + r][c] == seq![input[row as int]][r][c] by { assert(r == 0); };
        assert forall|row: nat, r: int, c: int| row < input.len() && 0 <= r < 1 && 0 <= c < width(cos) implies
            #[trigger] cos[row as int + r][c] == seq![cos[row as int]][r][c] by { assert(r == 0); };
        assert forall|row: nat, r: int, c: int| row < input.len() && 0 <= r < 1 && 0 <= c < width(cos) implies
            #[trigger] sin[row as int + r][c] == seq![sin[row as int]][r][c] by { assert(r == 0); };
        RAW::checked_launch_equivalence(cos, sin, input, width(input), width(cos));
        assert forall|r: int| 0 <= r < input.len() implies
            (#[trigger] raw_output(input, cos, sin)[r]) == output(input, cos, sin)[r] by {
            assert(input[r].len() == width(input));
            assert(cos[r].len() == width(cos));
        };
    }
    assert(raw_output(input, cos, sin) =~= output(input, cos, sin));
}

pub proof fn row_shape(input: Tensor2D, cos: Tensor2D, sin: Tensor2D, r: int)
    requires layout(input, cos, sin), 0 <= r < input.len(),
    ensures row_output(input[r], cos[r], sin[r]).len() == input[r].len(),
{
    assert(input[r].len() == width(input));
    assert(cos[r].len() == width(cos));
}

} // verus!
