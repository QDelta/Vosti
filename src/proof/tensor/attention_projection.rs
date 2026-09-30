//! Shared request/row projection, parameterized by one fixed row operation.
//! The generated adapter binds that operation to canonical raw execution.
//! These definitions do not assert a numerical attention law.

use crate::{types::{BlockId}, proof::tensor::types::{KVCacheLayerRepr, Tensor1D, Tensor2D}};
#[cfg(verus_only)]
use crate::proof::tensor::geometry::{block_table_slot, cache_at};
use crate::proof::tensor::shape as TS;
use vstd::prelude::*;

verus! {

pub type RowOperation = spec_fn(Tensor1D, Tensor2D, Tensor2D) -> Tensor1D;

pub open spec fn logical_prefix(cache: KVCacheLayerRepr, row: Seq<BlockId>, count: nat)
    -> Tensor2D
{
    Seq::new(count, |pos: int| cache_at(cache, block_table_slot(row, pos as nat)))
}

pub open spec fn offsets_valid(cu: Seq<int>, rows: nat, tokens: nat) -> bool {
    &&& cu.len() == rows + 1
    &&& cu[0] == 0
    &&& cu[rows as int] == tokens
    // Match an existing adjacent pair; matching only cu[r] manufactures a
    // successor term that can repeatedly trigger this same quantifier.
    &&& forall|r: int| 0 <= r < rows ==> (#[trigger] cu[r]) < (#[trigger] cu[r + 1])
}

pub open spec fn owns(cu: Seq<int>, rows: nat, request: int, token: int) -> bool {
    0 <= request < rows && cu[request] <= token < cu[request + 1]
}

// Total outside the supported metadata domain as well. The checked lemmas
// establish existence/uniqueness whenever the engine may submit this launch.
#[verifier::opaque]
pub open spec fn owner(cu: Seq<int>, rows: nat, token: int) -> int {
    if exists|request: int| owns(cu, rows, request, token) {
        choose|request: int| owns(cu, rows, request, token)
    } else { 0 }
}

pub proof fn lemma_owner(cu: Seq<int>, rows: nat, request: int, token: int)
    requires
        cu.len() == rows + 1,
        forall|r: int| 0 <= r < rows ==> (#[trigger] cu[r]) < cu[r + 1],
        owns(cu, rows, request, token),
    ensures owner(cu, rows, token) == request,
{
    reveal(owner);
    assert(exists|r: int| owns(cu, rows, r, token));
    let selected = owner(cu, rows, token);
    if selected < request {
        crate::proof::tensor::geometry::lemma_cu_mono(cu, rows as int, selected + 1, request);
    } else if request < selected {
        crate::proof::tensor::geometry::lemma_cu_mono(cu, rows as int, request + 1, selected);
    }
}

pub proof fn lemma_owner_exists(cu: Seq<int>, rows: nat, token: int)
    requires
        cu.len() == rows + 1,
        0 < rows, cu[0] <= token < cu[rows as int],
    ensures owns(cu, rows, owner(cu, rows, token), token),
    decreases rows,
{
    reveal(owner);
    if cu[rows as int - 1] <= token {
        assert(owns(cu, rows, rows as int - 1, token));
    } else {
        assert(rows > 1);
        lemma_owner_exists(cu.drop_last(), (rows - 1) as nat, token);
        let r = owner(cu.drop_last(), (rows - 1) as nat, token);
        assert(cu.drop_last()[r] == cu[r]);
        assert(cu.drop_last()[r + 1] == cu[r + 1]);
        assert(owns(cu, rows, r, token));
    }
}

pub open spec fn row_output(
    op: RowOperation, width: nat, q: Tensor1D, k: Tensor2D, v: Tensor2D,
) -> Tensor1D {
    Seq::new(width, |c: int| op(q, k, v)[c])
}

pub proof fn lemma_row_output_exact(op: RowOperation, width: nat, q: Tensor1D, k: Tensor2D, v: Tensor2D)
    requires op(q, k, v).len() == width,
    ensures row_output(op, width, q, k, v) == op(q, k, v),
{
    assert(row_output(op, width, q, k, v) =~= op(q, k, v));
}

#[verifier::opaque]
pub open spec fn launch_output(
    op: RowOperation, width: nat, q: Tensor2D,
    k: KVCacheLayerRepr, v: KVCacheLayerRepr,
    cu_q: Seq<int>, cu_k: Seq<int>, table: Seq<Seq<BlockId>>,
) -> Tensor2D {
    Seq::new(q.len(), |token: int| {
        let request = owner(cu_q, table.len(), token);
        let count = (cu_k[request + 1] - cu_k[request]
            - (cu_q[request + 1] - cu_q[request]) + token - cu_q[request] + 1) as nat;
        row_output(op, width, q[token], logical_prefix(k, table[request], count),
            logical_prefix(v, table[request], count))
    })
}

pub proof fn lemma_launch_shape(
    op: RowOperation, width: nat, q: Tensor2D, k: KVCacheLayerRepr, v: KVCacheLayerRepr,
    cu_q: Seq<int>, cu_k: Seq<int>, table: Seq<Seq<BlockId>>,
)
    ensures TS::tensor2d_shape(launch_output(op, width, q, k, v, cu_q, cu_k, table), q.len(), width),
{ reveal(launch_output); }

pub proof fn lemma_selected_row(
    op: RowOperation, width: nat, q: Tensor2D, k: KVCacheLayerRepr, v: KVCacheLayerRepr,
    cu_q: Seq<int>, cu_k: Seq<int>, table: Seq<Seq<BlockId>>, request: int, token: int,
)
    requires
        offsets_valid(cu_q, table.len(), q.len()),
        owns(cu_q, table.len(), request, token),
        0 <= token < q.len(),
    ensures
        launch_output(op, width, q, k, v, cu_q, cu_k, table)[token]
        == row_output(op, width, q[token],
            logical_prefix(k, table[request], (cu_k[request + 1] - cu_k[request]
                - cu_q[request + 1] + token + 1) as nat),
            logical_prefix(v, table[request], (cu_k[request + 1] - cu_k[request]
                - cu_q[request + 1] + token + 1) as nat)),
{
    reveal(launch_output);
    lemma_owner(cu_q, table.len(), request, token);
}

pub proof fn lemma_batch_projection(
    op: RowOperation, width: nat, q: Tensor2D, k: KVCacheLayerRepr, v: KVCacheLayerRepr,
    cu_q: Seq<int>, cu_k: Seq<int>, table: Seq<Seq<BlockId>>, request: int,
)
    requires
        offsets_valid(cu_q, table.len(), q.len()),
        0 <= request < table.len(),
        0 <= cu_q[request] < cu_q[request + 1] <= q.len(),
    ensures
        launch_output(op, width, q, k, v, cu_q, cu_k, table)
            .subrange(cu_q[request], cu_q[request + 1])
        == launch_output(op, width, q.subrange(cu_q[request], cu_q[request + 1]), k, v,
            seq![0int, cu_q[request + 1] - cu_q[request]],
            seq![0int, cu_k[request + 1] - cu_k[request]], seq![table[request]]),
{
    let start = cu_q[request];
    let end = cu_q[request + 1];
    let selected = q.subrange(start, end);
    let qs = seq![0int, end - start];
    let ks = seq![0int, cu_k[request + 1] - cu_k[request]];
    let one = seq![table[request]];
    let left = launch_output(op, width, q, k, v, cu_q, cu_k, table).subrange(start, end);
    let right = launch_output(op, width, selected, k, v, qs, ks, one);
    lemma_launch_shape(op, width, q, k, v, cu_q, cu_k, table);
    lemma_launch_shape(op, width, selected, k, v, qs, ks, one);
    assert forall|r: int| 0 <= r < end - start implies (#[trigger] left[r]) == right[r] by {
        lemma_selected_row(op, width, q, k, v, cu_q, cu_k, table, request, start + r);
        lemma_selected_row(op, width, selected, k, v, qs, ks, one, 0, r);
        assert(selected[r] == q[start + r]);
    }
    assert(left =~= right);
}

pub proof fn lemma_prefix_equality(
    a: KVCacheLayerRepr, row_a: Seq<BlockId>, b: KVCacheLayerRepr, row_b: Seq<BlockId>, count: nat,
)
    requires forall|pos: nat| pos < count ==>
        (#[trigger] cache_at(a, block_table_slot(row_a, pos))) == cache_at(b, block_table_slot(row_b, pos)),
    ensures logical_prefix(a, row_a, count) == logical_prefix(b, row_b, count),
{
    assert(logical_prefix(a, row_a, count) =~= logical_prefix(b, row_b, count));
}

pub proof fn lemma_singleton_equivalence(
    op: RowOperation, width: nat,
    qa: Tensor2D, ka: KVCacheLayerRepr, va: KVCacheLayerRepr, ra: Seq<BlockId>, kla: nat, ja: int,
    qb: Tensor2D, kb: KVCacheLayerRepr, vb: KVCacheLayerRepr, rb: Seq<BlockId>, klb: nat, jb: int,
)
    requires
        0 <= ja < qa.len() <= kla, 0 <= jb < qb.len() <= klb,
        qa[ja] == qb[jb], kla - qa.len() + ja == klb - qb.len() + jb,
        forall|pos: nat| pos <= kla - qa.len() + ja ==>
            (#[trigger] cache_at(ka, block_table_slot(ra, pos))) == cache_at(kb, block_table_slot(rb, pos)),
        forall|pos: nat| pos <= kla - qa.len() + ja ==>
            (#[trigger] cache_at(va, block_table_slot(ra, pos))) == cache_at(vb, block_table_slot(rb, pos)),
    ensures
        launch_output(op, width, qa, ka, va, seq![0int, qa.len() as int], seq![0int, kla as int], seq![ra])[ja]
        == launch_output(op, width, qb, kb, vb, seq![0int, qb.len() as int], seq![0int, klb as int], seq![rb])[jb],
{
    lemma_selected_row(op, width, qa, ka, va, seq![0int, qa.len() as int], seq![0int, kla as int], seq![ra], 0, ja);
    lemma_selected_row(op, width, qb, kb, vb, seq![0int, qb.len() as int], seq![0int, klb as int], seq![rb], 0, jb);
    let count = (kla - qa.len() + ja + 1) as nat;
    lemma_prefix_equality(ka, ra, kb, rb, count);
    lemma_prefix_equality(va, ra, vb, rb, count);
}

} // verus!
