//! One generated attention operator, shared by every model composition.
//!
//! The mapped semantics uses the generated raw canonical row operation. Its
//! correspondence to a raw launch is a checked proof under the complete layout
//! and numeric premises, not an additional assumed output equivalence.

use crate::model_config::AttentionKind;
use vstd::prelude::*;
use crate::{types::*, proof::model::types::*, proof::tensor::types::*};
#[cfg(verus_only)]
use crate::boundary::backend_certificates::{attention as RAW, support as SUP};
#[cfg(verus_only)]
use crate::proof::tensor::{attention_projection as AP, paged as PL, shape as TS};

verus! {

// The host computes its fixed scalar argument from the model's scale policy
// (softmax scale times log2(e)). Scalar is opaque; no floating-point arithmetic
// identity or equivalence to query pre-scaling is asserted here.
pub uninterp spec fn scale_log2(parameters: AttentionParametersRepr) -> Scalar;

pub open spec fn effective_window(kind: AttentionKind, window: nat) -> nat {
    match kind { AttentionKind::Full => 0, AttentionKind::SlidingWindow => window }
}

#[verifier::opaque]
pub open spec fn row_operation(
    kind: AttentionKind, parameters: AttentionParametersRepr, window: nat,
) -> AP::RowOperation {
    match RAW::row_operation(kind, parameters.geometry, scale_log2(parameters), window,
        SUP::generated_kernel_allocation_cell()) {
        Some(op) => op,
        // Totalize the pure specification outside the admitted domain. This
        // branch is not a serving fallback: checked_runtime_binding requires
        // an admitted geometry and proves that a raw output exists.
        None => |q: Tensor1D, k: Tensor2D, v: Tensor2D| Seq::empty(),
    }
}

#[verifier::opaque]
pub open spec fn output(
    q: Tensor2D, k: KVCacheLayerRepr, v: KVCacheLayerRepr,
    cu_q: Seq<int>, cu_k: Seq<int>, table: Seq<Seq<BlockId>>,
    kind: AttentionKind, parameters: AttentionParametersRepr, window: nat,
) -> Tensor2D {
    AP::launch_output(row_operation(kind, parameters, window),
        parameters.geometry.num_attention_heads * parameters.geometry.head_dim,
        q, k, v, cu_q, cu_k, table)
}

pub proof fn lemma_shape(
    q: Tensor2D, k: KVCacheLayerRepr, v: KVCacheLayerRepr,
    cu_q: Seq<int>, cu_k: Seq<int>, table: Seq<Seq<BlockId>>,
    kind: AttentionKind, parameters: AttentionParametersRepr, window: nat,
)
    ensures
        output(q, k, v, cu_q, cu_k, table, kind, parameters, window).len() == q.len(),
        TS::rectangular(output(q, k, v, cu_q, cu_k, table, kind, parameters, window)),
{
    reveal(output);
    AP::lemma_launch_shape(row_operation(kind, parameters, window),
        parameters.geometry.num_attention_heads * parameters.geometry.head_dim,
        q, k, v, cu_q, cu_k, table);
}

pub proof fn lemma_batch_projection(
    q: Tensor2D, k: KVCacheLayerRepr, v: KVCacheLayerRepr,
    cu_q: Seq<int>, cu_k: Seq<int>, table: Seq<Seq<BlockId>>, max_q: nat, max_k: nat,
    kind: AttentionKind, parameters: AttentionParametersRepr, window: nat, request: int,
)
    requires
        SUP::paged_attention_metadata_ready(q.len(), k.len(), cu_q, cu_k, max_q, max_k, table),
        0 <= request < table.len(),
    ensures
        output(q, k, v, cu_q, cu_k, table, kind, parameters, window)
            .subrange(cu_q[request], cu_q[request + 1])
        == output(q.subrange(cu_q[request], cu_q[request + 1]), k, v,
            seq![0int, cu_q[request + 1] - cu_q[request]],
            seq![0int, cu_k[request + 1] - cu_k[request]], seq![table[request]],
            kind, parameters, window),
{
    reveal(output);
    PL::lemma_metadata_strict_offsets(q.len(), k.len(), cu_q, cu_k, max_q, max_k, table);
    crate::proof::tensor::geometry::lemma_cu_int_bounds(cu_q, table.len() as int);
    AP::lemma_batch_projection(row_operation(kind, parameters, window),
        parameters.geometry.num_attention_heads * parameters.geometry.head_dim,
        q, k, v, cu_q, cu_k, table, request);
}

pub proof fn lemma_singleton_equivalence(
    qa: Tensor2D, ka: KVCacheLayerRepr, va: KVCacheLayerRepr, ra: Seq<BlockId>, kla: nat, ja: int,
    qb: Tensor2D, kb: KVCacheLayerRepr, vb: KVCacheLayerRepr, rb: Seq<BlockId>, klb: nat, jb: int,
    kind: AttentionKind, parameters: AttentionParametersRepr, window: nat,
)
    requires
        0 <= ja < qa.len() <= kla, 0 <= jb < qb.len() <= klb,
        qa[ja] == qb[jb], kla - qa.len() + ja == klb - qb.len() + jb,
        forall|pos: nat| pos <= kla - qa.len() + ja ==>
            (#[trigger] crate::proof::tensor::geometry::cache_at(ka, crate::proof::tensor::geometry::block_table_slot(ra, pos)))
            == crate::proof::tensor::geometry::cache_at(kb, crate::proof::tensor::geometry::block_table_slot(rb, pos)),
        forall|pos: nat| pos <= kla - qa.len() + ja ==>
            (#[trigger] crate::proof::tensor::geometry::cache_at(va, crate::proof::tensor::geometry::block_table_slot(ra, pos)))
            == crate::proof::tensor::geometry::cache_at(vb, crate::proof::tensor::geometry::block_table_slot(rb, pos)),
    ensures
        output(qa, ka, va, seq![0int, qa.len() as int], seq![0int, kla as int], seq![ra],
            kind, parameters, window)[ja]
        == output(qb, kb, vb, seq![0int, qb.len() as int], seq![0int, klb as int], seq![rb],
            kind, parameters, window)[jb],
{
    reveal(output);
    AP::lemma_singleton_equivalence(row_operation(kind, parameters, window),
        parameters.geometry.num_attention_heads * parameters.geometry.head_dim,
        qa, ka, va, ra, kla, ja, qb, kb, vb, rb, klb, jb);
}

#[verifier::spinoff_prover]
pub proof fn lemma_batch_projection_relocated(
    q: Tensor2D, k: KVCacheLayerRepr, v: KVCacheLayerRepr,
    cu_q: Seq<int>, cu_k: Seq<int>, table: Seq<Seq<BlockId>>, max_q: nat, max_k: nat,
    selected_k: KVCacheLayerRepr, selected_v: KVCacheLayerRepr, selected_row: Seq<BlockId>,
    kind: AttentionKind, parameters: AttentionParametersRepr, window: nat, request: int,
)
    requires
        SUP::paged_attention_metadata_ready(q.len(), k.len(), cu_q, cu_k, max_q, max_k, table),
        0 <= request < table.len(),
        forall|pos: nat| pos < cu_k[request + 1] - cu_k[request] ==>
            (#[trigger] crate::proof::tensor::geometry::cache_at(k, crate::proof::tensor::geometry::block_table_slot(table[request], pos)))
            == crate::proof::tensor::geometry::cache_at(selected_k, crate::proof::tensor::geometry::block_table_slot(selected_row, pos)),
        forall|pos: nat| pos < cu_k[request + 1] - cu_k[request] ==>
            (#[trigger] crate::proof::tensor::geometry::cache_at(v, crate::proof::tensor::geometry::block_table_slot(table[request], pos)))
            == crate::proof::tensor::geometry::cache_at(selected_v, crate::proof::tensor::geometry::block_table_slot(selected_row, pos)),
    ensures
        output(q, k, v, cu_q, cu_k, table, kind, parameters, window)
            .subrange(cu_q[request], cu_q[request + 1])
        == output(q.subrange(cu_q[request], cu_q[request + 1]), selected_k, selected_v,
            seq![0int, cu_q[request + 1] - cu_q[request]],
            seq![0int, cu_k[request + 1] - cu_k[request]], seq![selected_row], kind, parameters, window),
{
    PL::lemma_metadata_strict_offsets(q.len(), k.len(), cu_q, cu_k, max_q, max_k, table);
    crate::proof::tensor::geometry::lemma_cu_int_bounds(cu_q, table.len() as int);
    lemma_batch_projection(q, k, v, cu_q, cu_k, table, max_q, max_k, kind, parameters, window, request);
    let start = cu_q[request];
    let end = cu_q[request + 1];
    let qs = q.subrange(start, end);
    let kl = (cu_k[request + 1] - cu_k[request]) as nat;
    let qs_cu = seq![0int, end - start];
    let ks_cu = seq![0int, kl as int];
    let original = output(qs, k, v, qs_cu, ks_cu, seq![table[request]], kind, parameters, window);
    let relocated = output(qs, selected_k, selected_v, qs_cu, ks_cu, seq![selected_row], kind, parameters, window);
    lemma_shape(qs, k, v, qs_cu, ks_cu, seq![table[request]], kind, parameters, window);
    lemma_shape(qs, selected_k, selected_v, qs_cu, ks_cu, seq![selected_row], kind, parameters, window);
    assert(qs.len() <= kl);
    assert forall|j: int| 0 <= j < qs.len() implies (#[trigger] original[j]) == relocated[j] by {
        lemma_singleton_equivalence(qs, k, v, table[request], kl, j,
            qs, selected_k, selected_v, selected_row, kl, j, kind, parameters, window);
    }
    assert(original =~= relocated);
}

#[verifier::spinoff_prover]
pub proof fn checked_runtime_binding(
    q: Tensor2D, k: KVCacheLayerRepr, v: KVCacheLayerRepr,
    cu_q: Seq<int>, cu_k: Seq<int>, table: Seq<Seq<BlockId>>, max_q: nat, max_k: nat,
    kind: AttentionKind, parameters: AttentionParametersRepr, window: nat,
)
    requires
        RAW::binding_valid(kind, parameters.geometry, window),
        RAW::layout_ready(q, k, v, table, cu_q, cu_k, max_q, max_k, parameters.geometry),
        RAW::numeric_requirements(q, k, v, table, cu_q, cu_k, kind, parameters.geometry,
            scale_log2(parameters), window, SUP::generated_kernel_allocation_cell()),
    ensures
        RAW::raw_output(q, k, v, table, cu_q, cu_k, max_q, kind, parameters.geometry,
            scale_log2(parameters), window, SUP::generated_kernel_allocation_cell())
        == Some(output(q, k, v, cu_q, cu_k, table, kind, parameters, window)),
{
    reveal(output);
    reveal(row_operation);
    RAW::checked_launch_equivalence(q, k, v, table, cu_q, cu_k, max_q, max_k,
        kind, parameters.geometry, scale_log2(parameters), window, SUP::generated_kernel_allocation_cell());
}

}
