//! Checked last-axis reshapes shared by kernel representation adapters.
//!
//! No numerical kernel semantics or execution-equivalence axiom is introduced.
//! Exact flattened widths are explicit premises; no truncation/padding rule is
//! used to bypass missing model geometry. Generated attention adapters consume
//! the checked shape, row-equality, and regional-equality lemmas below.

use crate::{boundary::scalar::{Scalar}, proof::tensor::types::{Tensor1D, Tensor2D, Tensor3D, Tensor4D}};
use crate::proof::tensor::shape as TS;
use vstd::prelude::*;

verus! {

// Engine linear weights use [output, reduction]; kernels use its transposed
// view [reduction, output]. The explicit width also describes empty weights.
pub open spec fn transposed_weights(weights: Tensor2D, width: nat) -> Tensor2D {
    Seq::new(width, |k: int| Seq::new(weights.len(), |n: int| weights[n][k]))
}

pub open spec fn split_vector(row: Tensor1D, groups: nat, width: nat) -> Tensor2D {
    Seq::new(groups, |g: int| row.subrange(g * width as int, (g + 1) * width as int))
}

pub open spec fn split_last_axis(rows: Tensor2D, groups: nat, width: nat) -> Tensor3D {
    Seq::new(rows.len(), |r: int| split_vector(rows[r], groups, width))
}

pub open spec fn split_last_axis_3d(tensor: Tensor3D, groups: nat, width: nat) -> Tensor4D {
    Seq::new(tensor.len(), |i: int| split_last_axis(tensor[i], groups, width))
}

pub open spec fn merge_last_axis(tensor: Tensor3D) -> Tensor2D {
    Seq::new(tensor.len(), |r: int| tensor[r].flatten())
}

pub proof fn lemma_split_vector_shape(row: Tensor1D, groups: nat, width: nat)
    requires row.len() == groups * width,
    ensures TS::tensor2d_shape(split_vector(row, groups, width), groups, width),
{
    assert forall|g: int| 0 <= g < groups implies
        (#[trigger] split_vector(row, groups, width)[g]).len() == width by {
        assert(0 <= g * width as int <= (g + 1) * width as int <= row.len())
            by (nonlinear_arith)
            requires 0 <= g < groups, row.len() == groups * width, width >= 0,
        {}
        assert((g + 1) * width as int - g * width as int == width) by (nonlinear_arith);
    }
}

pub proof fn lemma_split_last_axis_shape(
    rows: Tensor2D, count: nat, groups: nat, width: nat,
)
    requires TS::tensor2d_shape(rows, count, groups * width),
    ensures TS::tensor3d_shape(split_last_axis(rows, groups, width), count, groups, width),
{
    assert forall|r: int| 0 <= r < count implies
        TS::tensor2d_shape(#[trigger] split_last_axis(rows, groups, width)[r], groups, width) by {
        lemma_split_vector_shape(rows[r], groups, width);
    }
}

pub proof fn lemma_split_last_axis_3d_shape(
    tensor: Tensor3D, outer: nat, rows: nat, groups: nat, width: nat,
)
    requires TS::tensor3d_shape(tensor, outer, rows, groups * width),
    ensures TS::tensor4d_shape(split_last_axis_3d(tensor, groups, width), outer, rows, groups, width),
{
    assert forall|i: int| 0 <= i < outer implies
        TS::tensor3d_shape(#[trigger] split_last_axis_3d(tensor, groups, width)[i], rows, groups, width) by {
        lemma_split_last_axis_shape(tensor[i], rows, groups, width);
    }
}

pub proof fn lemma_split_last_axis_subrange(
    rows: Tensor2D, groups: nat, width: nat, lo: int, hi: int,
)
    requires 0 <= lo <= hi <= rows.len(),
    ensures
        split_last_axis(rows.subrange(lo, hi), groups, width)
            == split_last_axis(rows, groups, width).subrange(lo, hi),
{
    assert(split_last_axis(rows.subrange(lo, hi), groups, width)
        =~= split_last_axis(rows, groups, width).subrange(lo, hi));
}

pub proof fn lemma_split_last_axis_row_equality(
    a: Tensor2D, b: Tensor2D, ia: int, ib: int, groups: nat, width: nat,
)
    requires 0 <= ia < a.len(), 0 <= ib < b.len(), a[ia] == b[ib],
    ensures
        split_last_axis(a, groups, width)[ia] == split_last_axis(b, groups, width)[ib],
{
}

pub proof fn lemma_merge_last_axis_shape(
    tensor: Tensor3D, rows: nat, groups: nat, width: nat,
)
    requires TS::tensor3d_shape(tensor, rows, groups, width),
    ensures TS::tensor2d_shape(merge_last_axis(tensor), rows, groups * width),
{
    assert forall|r: int| 0 <= r < rows implies
        (#[trigger] merge_last_axis(tensor)[r]).len() == groups * width by {
        assert(TS::tensor2d_shape(tensor[r], groups, width));
        crate::proof::tensor::seq_flatten::lemma_fixed_width_flatten_len(tensor[r], width);
    }
}

pub proof fn lemma_merge_last_axis_row_equality(
    a: Tensor3D, b: Tensor3D, ia: int, ib: int,
)
    requires 0 <= ia < a.len(), 0 <= ib < b.len(), a[ia] == b[ib],
    ensures merge_last_axis(a)[ia] == merge_last_axis(b)[ib],
{
}

// Raw regional contracts express cellwise equality. Rectangularity is needed
// before that equality can be promoted to equality of entire engine rows.
pub proof fn lemma_merge_last_axis_region_equality(
    a: Tensor3D, b: Tensor3D, lo_a: int, lo_b: int,
    count: nat, groups: nat, width: nat,
)
    requires
        TS::tensor3d_shape(a, a.len(), groups, width),
        TS::tensor3d_shape(b, b.len(), groups, width),
        0 <= lo_a <= lo_a + count <= a.len(),
        0 <= lo_b <= lo_b + count <= b.len(),
        forall|r: int, g: int, c: int|
            0 <= r < count && 0 <= g < groups && 0 <= c < width ==>
                (#[trigger] a[lo_a + r][g][c]) == b[lo_b + r][g][c],
    ensures
        merge_last_axis(a).subrange(lo_a, lo_a + count)
            == merge_last_axis(b).subrange(lo_b, lo_b + count),
{
    assert forall|r: int| 0 <= r < count implies
        (#[trigger] merge_last_axis(a)[lo_a + r]) == merge_last_axis(b)[lo_b + r] by {
        assert(TS::tensor2d_shape(a[lo_a + r], groups, width));
        assert(TS::tensor2d_shape(b[lo_b + r], groups, width));
        assert forall|g: int| 0 <= g < groups implies
            (#[trigger] a[lo_a + r][g]) == b[lo_b + r][g] by {
            assert forall|c: int| 0 <= c < width implies
                (#[trigger] a[lo_a + r][g][c]) == b[lo_b + r][g][c] by {}
            assert(a[lo_a + r][g] =~= b[lo_b + r][g]);
        }
        assert(a[lo_a + r] =~= b[lo_b + r]);
        lemma_merge_last_axis_row_equality(a, b, lo_a + r, lo_b + r);
    }
    let left = merge_last_axis(a).subrange(lo_a, lo_a + count);
    let right = merge_last_axis(b).subrange(lo_b, lo_b + count);
    assert forall|r: int| 0 <= r < count implies (#[trigger] left[r]) == right[r] by {
        assert(left[r] == merge_last_axis(a)[lo_a + r]);
        assert(merge_last_axis(a)[lo_a + r] == merge_last_axis(b)[lo_b + r]);
        assert(right[r] == merge_last_axis(b)[lo_b + r]);
    }
    assert(left =~= right);
}

// The caller supplies an allocation token. No new uninterpreted function or
// equality between distinct allocations is assumed here.
pub open spec fn tensor2d_allocation(rows: nat, cols: nat, fill: Scalar) -> Tensor2D {
    Seq::new(rows, |_r: int| Seq::new(cols, |_c: int| fill))
}

pub proof fn lemma_tensor2d_allocation_shape(rows: nat, cols: nat, fill: Scalar)
    ensures TS::tensor2d_shape(tensor2d_allocation(rows, cols, fill), rows, cols),
{
}

pub open spec fn tensor3d_allocation(
    rows: nat, groups: nat, width: nat, fill: Scalar,
) -> Tensor3D {
    Seq::new(rows, |_r: int| tensor2d_allocation(groups, width, fill))
}

pub proof fn lemma_tensor3d_allocation_shape(rows: nat, groups: nat, width: nat, fill: Scalar)
    ensures TS::tensor3d_shape(tensor3d_allocation(rows, groups, width, fill), rows, groups, width),
{
}

} // verus!
