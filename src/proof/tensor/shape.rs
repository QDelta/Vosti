// Rectangular shape vocabulary for dense semantic tensors.
//
// `Tensor2D` remains a nested `Seq` because the engine proof relies heavily on
// sequence indexing, concatenation, and subranges.  Verus has no refinement
// type that would make a wrapper intrinsically rectangular, so shape is an
// explicit predicate carried by representation and semantic contracts.

use crate::{proof::tensor::types::{Tensor2D, Tensor3D, Tensor4D}};
use vstd::prelude::*;

verus! {

pub open spec fn tensor2d_shape(
    tensor: Tensor2D,
    rows: nat,
    cols: nat,
) -> bool {
    tensor.len() == rows
    && forall|i: int| 0 <= i < rows ==>
        (#[trigger] tensor[i]).len() == cols
}

pub open spec fn rectangular(tensor: Tensor2D) -> bool {
    exists|cols: nat| tensor2d_shape(tensor, tensor.len(), cols)
}

pub open spec fn tensor3d_shape(tensor: Tensor3D, x: nat, y: nat, z: nat) -> bool {
    tensor.len() == x
    && forall|i: int| 0 <= i < x ==>
        tensor2d_shape(#[trigger] tensor[i], y, z)
}

pub open spec fn tensor4d_shape(tensor: Tensor4D, x: nat, y: nat, z: nat, w: nat) -> bool {
    tensor.len() == x
    && forall|i: int| 0 <= i < x ==>
        tensor3d_shape(#[trigger] tensor[i], y, z, w)
}

// Exact domain needed by the executable gate split.  This deliberately says
// nothing about a particular model width: any rectangular matrix with an even
// last-axis extent is admissible.
pub open spec fn tensor2d_even_width(tensor: Tensor2D) -> bool {
    exists|cols: nat|
        tensor2d_shape(tensor, tensor.len(), cols) && cols % 2 == 0
}

pub proof fn lemma_tensor2d_shape_even_width(
    tensor: Tensor2D,
    rows: nat,
    cols: nat,
)
    requires
        tensor2d_shape(tensor, rows, cols),
        cols % 2 == 0,
    ensures
        tensor2d_even_width(tensor),
{
    assert(tensor.len() == rows);
    assert(tensor2d_shape(tensor, tensor.len(), cols));
}

pub proof fn lemma_tensor2d_shape_row(
    tensor: Tensor2D,
    rows: nat,
    cols: nat,
    i: int,
)
    requires
        tensor2d_shape(tensor, rows, cols),
        0 <= i < rows,
    ensures
        tensor[i].len() == cols,
{
}

pub proof fn lemma_tensor2d_shape_subrange(
    tensor: Tensor2D,
    rows: nat,
    cols: nat,
    lo: int,
    hi: int,
)
    requires
        tensor2d_shape(tensor, rows, cols),
        0 <= lo <= hi <= rows,
    ensures
        tensor2d_shape(tensor.subrange(lo, hi), (hi - lo) as nat, cols),
{
    assert(tensor.subrange(lo, hi).len() == (hi - lo) as nat);
    assert forall|j: int| 0 <= j < hi - lo implies
        (#[trigger] tensor.subrange(lo, hi)[j]).len() == cols by {
        assert(0 <= lo + j < rows);
        assert(tensor.subrange(lo, hi)[j] == tensor[lo + j]);
    }
}

} // verus!
