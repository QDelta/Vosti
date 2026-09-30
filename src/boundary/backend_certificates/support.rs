//! Architecture-neutral vocabulary shared by generated certificate catalogs.
//!
//! This module contains no imported theorem. It only centralizes the logical
//! allocation token and the normalized paged-attention launch domain so that
//! neither family catalog depends on the other.

#[cfg(verus_only)]
use crate::{types::{BLOCK_SIZE_SPEC}, proof::tensor::geometry::{block_table_slot, blocks_needed_for, cache_at}};
use crate::{types::*, proof::model::types::*, proof::tensor::types::*};
use vstd::prelude::*;

verus! {

// Kernel contracts do not relate pre-launch output contents. Every generated
// execute function replaces all logical output cells.
pub uninterp spec fn generated_kernel_allocation_cell() -> Scalar;

pub open spec fn paged_attention_metadata_header_ready(
    query_rows: nat,
    num_pages: nat,
    cu_q_repr: Seq<int>,
    cu_k_repr: Seq<int>,
    max_seqlen_q: nat,
    max_seqlen_k: nat,
    bt_repr: Seq<Seq<BlockId>>,
) -> bool {
    &&& query_rows > 0
    &&& num_pages > 0
    &&& bt_repr.len() > 0
    &&& cu_q_repr.len() == bt_repr.len() + 1
    &&& cu_k_repr.len() == bt_repr.len() + 1
    &&& cu_q_repr[0] == 0
    &&& cu_k_repr[0] == 0
    &&& cu_q_repr[bt_repr.len() as int] == query_rows as int
    &&& max_seqlen_q > 0
    &&& max_seqlen_k > 0
}

pub open spec fn paged_attention_metadata_rows_ready(
    num_pages: nat,
    cu_q_repr: Seq<int>,
    cu_k_repr: Seq<int>,
    max_seqlen_q: nat,
    max_seqlen_k: nat,
    bt_repr: Seq<Seq<BlockId>>,
) -> bool {
    forall|j: int| #![trigger cu_q_repr[j + 1]]
        0 <= j < bt_repr.len() as int ==> {
        let q_len = cu_q_repr[j + 1] - cu_q_repr[j];
        let k_len = cu_k_repr[j + 1] - cu_k_repr[j];
        &&& cu_q_repr[j] < cu_q_repr[j + 1]
        &&& cu_k_repr[j] < cu_k_repr[j + 1]
        &&& q_len <= max_seqlen_q as int
        &&& k_len <= max_seqlen_k as int
        &&& q_len <= k_len
        &&& blocks_needed_for(k_len as nat) <= bt_repr[j].len()
        &&& (forall|l: int| 0 <= l < bt_repr[j].len() ==>
            #[trigger] bt_repr[j][l] < num_pages)
    }
}

pub open spec fn paged_attention_metadata_ready(
    query_rows: nat,
    num_pages: nat,
    cu_q_repr: Seq<int>,
    cu_k_repr: Seq<int>,
    max_seqlen_q: nat,
    max_seqlen_k: nat,
    bt_repr: Seq<Seq<BlockId>>,
) -> bool {
    paged_attention_metadata_header_ready(
        query_rows, num_pages, cu_q_repr, cu_k_repr,
        max_seqlen_q, max_seqlen_k, bt_repr,
    ) && paged_attention_metadata_rows_ready(
        num_pages, cu_q_repr, cu_k_repr,
        max_seqlen_q, max_seqlen_k, bt_repr,
    )
}

pub open spec fn paged_cache_geometry(
    k_cache_repr: KVCacheLayerRepr,
    v_cache_repr: KVCacheLayerRepr,
) -> bool {
    k_cache_repr.len() > 0
    && v_cache_repr.len() == k_cache_repr.len()
    && (forall|p: int| 0 <= p < k_cache_repr.len() ==>
        (#[trigger] k_cache_repr[p]).len() == BLOCK_SIZE_SPEC as int)
    && (forall|p: int| 0 <= p < v_cache_repr.len() ==>
        (#[trigger] v_cache_repr[p]).len() == BLOCK_SIZE_SPEC as int)
}

pub open spec fn paged_attention_launch_ready(
    query_rows: nat,
    k_cache_repr: KVCacheLayerRepr,
    v_cache_repr: KVCacheLayerRepr,
    cu_q_repr: Seq<int>,
    cu_k_repr: Seq<int>,
    max_seqlen_q: nat,
    max_seqlen_k: nat,
    bt_repr: Seq<Seq<BlockId>>,
) -> bool {
    paged_cache_geometry(k_cache_repr, v_cache_repr)
    && paged_attention_metadata_ready(
        query_rows, k_cache_repr.len(), cu_q_repr, cu_k_repr,
        max_seqlen_q, max_seqlen_k, bt_repr,
    )
}

pub open spec fn swa_paged_attention_launch_ready(
    query_rows: nat,
    k_cache_repr: KVCacheLayerRepr, v_cache_repr: KVCacheLayerRepr,
    cu_q_repr: Seq<int>, cu_k_repr: Seq<int>,
    max_seqlen_q: nat, max_seqlen_k: nat, bt_repr: Seq<Seq<BlockId>>, window_size: nat,
) -> bool {
    paged_attention_launch_ready(query_rows, k_cache_repr, v_cache_repr,
        cu_q_repr, cu_k_repr, max_seqlen_q, max_seqlen_k, bt_repr)
    && window_size > 0
}

// Strong whole-page relation retained for existing cache-layout consumers.
// The generated raw binding separately proves selected causal-prefix equality;
// no attention arithmetic or relational output axiom is introduced here.
pub open spec fn paged_attention_selected_cache_pages_equal(
    k_cache_a: KVCacheLayerRepr, v_cache_a: KVCacheLayerRepr, bt_row_a: Seq<BlockId>,
    k_cache_b: KVCacheLayerRepr, v_cache_b: KVCacheLayerRepr, bt_row_b: Seq<BlockId>, k_len: nat,
) -> bool {
    &&& blocks_needed_for(k_len) <= bt_row_a.len()
    &&& blocks_needed_for(k_len) <= bt_row_b.len()
    &&& forall|pos: nat| #![trigger block_table_slot(bt_row_a, pos)]
        pos < blocks_needed_for(k_len) * BLOCK_SIZE_SPEC ==> {
            let slot_a = block_table_slot(bt_row_a, pos);
            let slot_b = block_table_slot(bt_row_b, pos);
            &&& cache_at(k_cache_a, slot_a) == cache_at(k_cache_b, slot_b)
            &&& cache_at(v_cache_a, slot_a) == cache_at(v_cache_b, slot_b)
        }
}

} // verus!
