// Bind one fixed raw operator to model geometry. Scale is the actual raw
// kernel argument, not a query-rescaling operation or an arithmetic axiom.
verus! {
pub open spec fn model_binding_valid(
    geometry: crate::proof::model::types::AttentionGeometryRepr, window: nat,
) -> bool {
    geometry.head_dim == __D__
    && geometry.num_attention_heads > 0
    && geometry.num_key_value_heads > 0
    && geometry.num_key_value_heads <= geometry.num_attention_heads
    && geometry.num_attention_heads % geometry.num_key_value_heads == 0
    __WINDOW_VALIDITY__
}

pub open spec fn bound_row_operation(
    geometry: crate::proof::model::types::AttentionGeometryRepr, scale: Scalar, window: nat, fill: Scalar,
) -> Option<AP::RowOperation> {
    if model_binding_valid(geometry, window) {
        Some(canonical_row_operation(geometry.num_attention_heads,
            geometry.num_key_value_heads, scale, window, fill))
    } else { None }
}

pub open spec fn bound_model_output(
    q: Tensor2D, k: KVCacheLayerRepr, v: KVCacheLayerRepr,
    table: Seq<Seq<BlockId>>, cu_q: Seq<int>, cu_k: Seq<int>,
    geometry: crate::proof::model::types::AttentionGeometryRepr, scale: Scalar, window: nat, fill: Scalar,
) -> Option<Tensor2D> {
    match bound_row_operation(geometry, scale, window, fill) {
        Some(op) => Some(AP::launch_output(op, geometry.num_attention_heads * geometry.head_dim,
            q, k, v, cu_q, cu_k, table)),
        None => None,
    }
}

pub proof fn checked_bound_launch_equivalence(
    q: Tensor2D, k: KVCacheLayerRepr, v: KVCacheLayerRepr,
    table: Seq<Seq<BlockId>>, cu_q: Seq<int>, cu_k: Seq<int>, max_q: nat, max_k: nat,
    geometry: crate::proof::model::types::AttentionGeometryRepr, scale: Scalar, window: nat, fill: Scalar,
)
    requires
        model_binding_valid(geometry, window),
        SUP::paged_attention_metadata_ready(q.len(), k.len(), cu_q, cu_k, max_q, max_k, table),
        TS::tensor2d_shape(q, q.len(), geometry.num_attention_heads * geometry.head_dim),
        TS::tensor3d_shape(k, k.len(), 64, geometry.num_key_value_heads * geometry.head_dim),
        TS::tensor3d_shape(v, k.len(), 64, geometry.num_key_value_heads * geometry.head_dim),
        common::blocks_needed_for(max_k) <= u64::MAX,
        launch_runtime_assumptions(q, k, v, table, cu_q, cu_k,
            geometry.num_attention_heads, geometry.num_key_value_heads, scale, window, fill),
    ensures
        bound_model_output(q, k, v, table, cu_q, cu_k, geometry, scale, window, fill)
        == Some(adapter_output(q, k, v, table, cu_q, cu_k, max_q,
            geometry.num_attention_heads, geometry.num_key_value_heads, scale, window, fill)),
{
    checked_launch_equivalence(q, k, v, table, cu_q, cu_k, max_q, max_k,
        geometry.num_attention_heads, geometry.num_key_value_heads, scale, window, fill);
}
}
