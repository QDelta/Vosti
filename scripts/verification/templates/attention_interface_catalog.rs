// Geometry dispatch is shared across architectures. Each abstract raw module
// denotes one fixed deployed implementation, not equality between tile choices.
use crate::model_config::AttentionKind;
use vstd::prelude::*;
use crate::{types::*, proof::model::types::*, proof::tensor::types::*};
#[cfg(verus_only)]
use crate::proof::tensor::geometry as common;
#[cfg(verus_only)]
use crate::boundary::backend_certificates::support as SUP;
#[cfg(verus_only)]
use crate::proof::tensor::{attention_projection as AP, shape as TS};

verus! {
pub open spec fn binding_valid(kind: AttentionKind, geometry: AttentionGeometryRepr, window: nat) -> bool {
    __VALID__
}

pub open spec fn row_operation(
    kind: AttentionKind, geometry: AttentionGeometryRepr, scale: Scalar, window: nat, fill: Scalar,
) -> Option<AP::RowOperation> {
    __OPERATION__
}

pub open spec fn raw_output(
    q: Tensor2D, k: KVCacheLayerRepr, v: KVCacheLayerRepr,
    table: Seq<Seq<BlockId>>, cu_q: Seq<int>, cu_k: Seq<int>, max_q: nat,
    kind: AttentionKind, geometry: AttentionGeometryRepr, scale: Scalar, window: nat, fill: Scalar,
) -> Option<Tensor2D> {
    __RAW__
}

pub open spec fn numeric_requirements(
    q: Tensor2D, k: KVCacheLayerRepr, v: KVCacheLayerRepr,
    table: Seq<Seq<BlockId>>, cu_q: Seq<int>, cu_k: Seq<int>,
    kind: AttentionKind, geometry: AttentionGeometryRepr, scale: Scalar, window: nat, fill: Scalar,
) -> bool {
    __NUMERIC__
}

pub open spec fn layout_ready(
    q: Tensor2D, k: KVCacheLayerRepr, v: KVCacheLayerRepr,
    table: Seq<Seq<BlockId>>, cu_q: Seq<int>, cu_k: Seq<int>, max_q: nat, max_k: nat,
    geometry: AttentionGeometryRepr,
) -> bool {
    SUP::paged_attention_metadata_ready(q.len(), k.len(), cu_q, cu_k, max_q, max_k, table)
    && TS::tensor2d_shape(q, q.len(), geometry.num_attention_heads * geometry.head_dim)
    && TS::tensor3d_shape(k, k.len(), 64, geometry.num_key_value_heads * geometry.head_dim)
    && TS::tensor3d_shape(v, k.len(), 64, geometry.num_key_value_heads * geometry.head_dim)
    && common::blocks_needed_for(max_k) <= u64::MAX
}

pub open spec fn model_output(
    q: Tensor2D, k: KVCacheLayerRepr, v: KVCacheLayerRepr,
    table: Seq<Seq<BlockId>>, cu_q: Seq<int>, cu_k: Seq<int>,
    kind: AttentionKind, geometry: AttentionGeometryRepr, scale: Scalar, window: nat, fill: Scalar,
) -> Option<Tensor2D> {
    match row_operation(kind, geometry, scale, window, fill) {
        Some(op) => Some(AP::launch_output(op, geometry.num_attention_heads * geometry.head_dim,
            q, k, v, cu_q, cu_k, table)),
        None => None,
    }
}

#[verifier::spinoff_prover]
pub proof fn checked_launch_equivalence(
    q: Tensor2D, k: KVCacheLayerRepr, v: KVCacheLayerRepr,
    table: Seq<Seq<BlockId>>, cu_q: Seq<int>, cu_k: Seq<int>, max_q: nat, max_k: nat,
    kind: AttentionKind, geometry: AttentionGeometryRepr, scale: Scalar, window: nat, fill: Scalar,
)
    requires
        binding_valid(kind, geometry, window),
        layout_ready(q, k, v, table, cu_q, cu_k, max_q, max_k, geometry),
        numeric_requirements(q, k, v, table, cu_q, cu_k, kind, geometry, scale, window, fill),
    ensures
        raw_output(q, k, v, table, cu_q, cu_k, max_q, kind, geometry, scale, window, fill)
        == model_output(q, k, v, table, cu_q, cu_k, kind, geometry, scale, window, fill),
        model_output(q, k, v, table, cu_q, cu_k, kind, geometry, scale, window, fill).is_some(),
{
    __PROOF__
}
}
