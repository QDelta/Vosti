//! Checked lookup bindings to generated geometry catalogs. No numerical
//! equivalence between plain and scaled lookup is assumed.
use vstd::prelude::*;
use crate::{boundary::scalar::{Scalar}, proof::tensor::types::{Tensor1D, Tensor2D, IntTensor1D}};
use crate::proof::tensor::shape as TS;
use crate::boundary::backend_certificates::{embedding as PLAIN, scaled_embedding as SCALED};

verus! {

pub open spec fn weight_width(weight: Tensor2D) -> nat {
    if weight.len() > 0 { weight[0].len() } else { 0 }
}

// Host scalar construction for ONE deployed dtype: sqrt(width), rounded to
// the weight dtype before passing the scalar to Triton. Arithmetic stays
// opaque; this is not a kernel output or a fused/unfused equality assumption.
pub uninterp spec fn scale_parameter(width: nat) -> Scalar;

pub open spec fn layout(weight: Tensor2D) -> bool {
    weight.len() > 0 && TS::tensor2d_shape(weight, weight.len(), weight_width(weight))
        && PLAIN::geometry_valid(weight_width(weight), weight.len())
}

pub open spec fn scaled_layout(weight: Tensor2D, width: nat) -> bool {
    TS::tensor2d_shape(weight, weight.len(), width)
        && SCALED::geometry_valid(width, weight.len())
}

pub open spec fn plain_raw_output(ids: IntTensor1D, weight: Tensor2D) -> Tensor2D {
    PLAIN::raw_output(ids, weight, weight_width(weight), weight.len()).unwrap()
}

pub open spec fn plain_row_output(token: int, weight: Tensor2D) -> Tensor1D {
    plain_raw_output(seq![token], weight)[0]
}

pub open spec fn plain_output(ids: IntTensor1D, weight: Tensor2D) -> Tensor2D {
    Seq::new(ids.len(), |row: int| plain_row_output(ids[row], weight))
}

pub proof fn checked_plain_binding(ids: IntTensor1D, weight: Tensor2D)
    requires layout(weight),
    ensures plain_raw_output(ids, weight) == plain_output(ids, weight),
{
    assert forall|row: nat, r: int| row < ids.len() && 0 <= r < 1 implies
        #[trigger] ids[row as int + r] == seq![ids[row as int]][r] by {
        assert(r == 0);
    };
    assert(PLAIN::launch_valid(ids, weight, weight_width(weight), weight.len()));
    PLAIN::checked_launch_equivalence(ids, weight, weight_width(weight), weight.len());
    assert(plain_raw_output(ids, weight) =~= plain_output(ids, weight));
}

pub proof fn plain_row_shape(token: int, weight: Tensor2D)
    requires layout(weight),
    ensures plain_row_output(token, weight).len() == weight_width(weight),
{
    assert(PLAIN::raw_output(seq![token], weight, weight_width(weight), weight.len()).is_some());
}

pub open spec fn scaled_raw_output(ids: IntTensor1D, weight: Tensor2D, width: nat) -> Tensor2D {
    SCALED::raw_output(ids, weight, scale_parameter(width), width, weight.len()).unwrap()
}

pub open spec fn scaled_row_output(token: int, weight: Tensor2D, width: nat) -> Tensor1D {
    scaled_raw_output(seq![token], weight, width)[0]
}

pub open spec fn scaled_output(ids: IntTensor1D, weight: Tensor2D, width: nat) -> Tensor2D {
    Seq::new(ids.len(), |row: int| scaled_row_output(ids[row], weight, width))
}

pub proof fn checked_scaled_binding(ids: IntTensor1D, weight: Tensor2D, width: nat)
    requires scaled_layout(weight, width),
    ensures scaled_raw_output(ids, weight, width) == scaled_output(ids, weight, width),
{
    assert forall|row: nat, r: int| row < ids.len() && 0 <= r < 1 implies
        #[trigger] ids[row as int + r] == seq![ids[row as int]][r] by {
        assert(r == 0);
    };
    assert(SCALED::launch_valid(ids, weight, scale_parameter(width), width, weight.len()));
    SCALED::checked_launch_equivalence(ids, weight, scale_parameter(width), width, weight.len());
    assert(scaled_raw_output(ids, weight, width) =~= scaled_output(ids, weight, width));
}

pub proof fn scaled_row_shape(token: int, weight: Tensor2D, width: nat)
    requires scaled_layout(weight, width),
    ensures scaled_row_output(token, weight, width).len() == width,
{
    assert(SCALED::raw_output(seq![token], weight, scale_parameter(width), width, weight.len()).is_some());
}

} // verus!
