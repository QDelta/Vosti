// Checked consumer of a generated raw kernel contract. Generated and qualified
// by the attention adapter renderer; no framework-level axiom is added here.
use crate::proof::tensor::layout as TL;
use crate::proof::tensor::shape as TS;
use crate::proof::tensor::attention_projection as AP;
use crate::proof::tensor::paged as PL;
use crate::boundary::backend_certificates::support as SUP;
use crate::proof::tensor::geometry as common;
use crate::{boundary::scalar::{Scalar}, proof::tensor::types::{Tensor2D, KVCacheLayerRepr}, types::{BlockId}};

verus! {
pub open spec fn adapter_side(
    q: Tensor2D, k: KVCacheLayerRepr, v: KVCacheLayerRepr,
    table: Seq<Seq<BlockId>>, cu_q: Seq<int>, cu_k: Seq<int>, max_q: nat,
    h: nat, hkv: nat, scale: Scalar, window: nat, fill: Scalar,
) -> __SIDE__ {
    __SIDE__ {
        __PORT_QUERY__: TL::split_last_axis(q, h, __D__),
        __PORT_KEYS__: TL::split_last_axis_3d(k, hkv, __D__),
        __PORT_VALUES__: TL::split_last_axis_3d(v, hkv, __D__),
        __PORT_PAGE_TABLE__: PL::rectangular_page_table(table, PL::page_table_width(table)),
        __PORT_QUERY_OFFSETS__: cu_q, __PORT_KEY_OFFSETS__: cu_k,
        __PORT_OUTPUT__: TL::tensor3d_allocation(q.len(), h, __D__, fill),
        __PORT_LOGSUMEXP__: TL::tensor2d_allocation(h, q.len(), fill),
        __PORT_SCALE__: scale, __PORT_MAX_QUERY__: max_q as int,
        __EXTENT_BATCH__: table.len() as int, __EXTENT_QUERY_HEADS__: h as int, __EXTENT_KV_HEADS__: hkv as int,
        __EXTENT_TABLE_WIDTH__: PL::page_table_width(table) as int,
        __EXTENT_PAGES__: k.len() as int, __PORT_TOTAL_KEYS__: cu_k[table.len() as int], __EXTENT_QUERY_TOKENS__: q.len() as int,
        __WINDOW_FIELD__
    }
}

#[verifier::opaque]
pub open spec fn adapter_output(
    q: Tensor2D, k: KVCacheLayerRepr, v: KVCacheLayerRepr,
    table: Seq<Seq<BlockId>>, cu_q: Seq<int>, cu_k: Seq<int>, max_q: nat,
    h: nat, hkv: nat, scale: Scalar, window: nat, fill: Scalar,
) -> Tensor2D {
    TL::merge_last_axis(__EXEC__(adapter_side(q, k, v, table, cu_q, cu_k, max_q,
        h, hkv, scale, window, fill)).__PORT_OUTPUT__)
}

pub proof fn lemma_adapter_output_shape(
    q: Tensor2D, k: KVCacheLayerRepr, v: KVCacheLayerRepr,
    table: Seq<Seq<BlockId>>, cu_q: Seq<int>, cu_k: Seq<int>, max_q: nat,
    h: nat, hkv: nat, scale: Scalar, window: nat, fill: Scalar,
)
    ensures TS::tensor2d_shape(adapter_output(q, k, v, table, cu_q, cu_k, max_q,
        h, hkv, scale, window, fill), q.len(), h * __D__),
{
    reveal(adapter_output);
    let result = __EXEC__(adapter_side(q, k, v, table, cu_q, cu_k, max_q,
        h, hkv, scale, window, fill));
    TL::lemma_tensor3d_allocation_shape(q.len(), h, __D__, fill);
    assert(TS::tensor3d_shape(result.__PORT_OUTPUT__, q.len(), h, __D__));
    TL::lemma_merge_last_axis_shape(result.__PORT_OUTPUT__, q.len(), h, __D__);
}

pub proof fn checked_batch_projection(
    q: Tensor2D, k: KVCacheLayerRepr, v: KVCacheLayerRepr,
    table: Seq<Seq<BlockId>>, cu_q: Seq<int>, cu_k: Seq<int>, max_q: nat, max_k: nat,
    h: nat, hkv: nat, scale: Scalar, window: nat, fill: Scalar, i: int,
    selected_k: KVCacheLayerRepr, selected_v: KVCacheLayerRepr,
    selected_row: Seq<BlockId>,
)
    requires
        SUP::paged_attention_metadata_ready(q.len(), k.len(), cu_q, cu_k, max_q, max_k, table),
        TS::tensor2d_shape(q, q.len(), h * __D__),
        TS::tensor3d_shape(k, k.len(), 64, hkv * __D__),
        TS::tensor3d_shape(v, k.len(), 64, hkv * __D__),
        h > 0, hkv > 0, hkv <= h, h % hkv == 0,
        __WINDOW_REQUIREMENT__
        0 <= i < table.len(),
        selected_k.len() > 0,
        TS::tensor3d_shape(selected_k, selected_k.len(), 64, hkv * __D__),
        TS::tensor3d_shape(selected_v, selected_k.len(), 64, hkv * __D__),
        PL::page_table_ids_valid(seq![selected_row], selected_k.len()),
        common::blocks_needed_for((cu_k[i + 1] - cu_k[i]) as nat) <= selected_row.len(),
        forall|pos: nat| pos < common::blocks_needed_for((cu_k[i + 1] - cu_k[i]) as nat) * 64 ==>
            (#[trigger] common::cache_at(k, common::block_table_slot(table[i], pos)))
                == common::cache_at(selected_k, common::block_table_slot(selected_row, pos)),
        forall|pos: nat| pos < common::blocks_needed_for((cu_k[i + 1] - cu_k[i]) as nat) * 64 ==>
            (#[trigger] common::cache_at(v, common::block_table_slot(table[i], pos)))
                == common::cache_at(selected_v, common::block_table_slot(selected_row, pos)),
    ensures
        adapter_output(q, k, v, table, cu_q, cu_k, max_q, h, hkv, scale, window, fill)
                .subrange(cu_q[i], cu_q[i + 1])
        == adapter_output(
            q.subrange(cu_q[i], cu_q[i + 1]), selected_k, selected_v, seq![selected_row],
            seq![0int, cu_q[i + 1] - cu_q[i]], seq![0int, cu_k[i + 1] - cu_k[i]],
            (cu_q[i + 1] - cu_q[i]) as nat, h, hkv, scale, window, fill),
{
    reveal(adapter_output);
    PL::lemma_metadata_page_table(q.len(), k.len(), cu_q, cu_k, max_q, max_k, table);
    PL::lemma_metadata_strict_offsets(q.len(), k.len(), cu_q, cu_k, max_q, max_k, table);
    common::lemma_cu_int_bounds(cu_q, table.len() as int);
    let start = cu_q[i];
    let end = cu_q[i + 1];
    let qlen = (end - start) as nat;
    let klen = (cu_k[i + 1] - cu_k[i]) as nat;
    let singleton = seq![selected_row];
    assert(0 <= start < end <= q.len());
    assert(klen > 0);
    assert(common::blocks_needed_for(klen) <= table[i].len());
    PL::lemma_canonical_page_table(singleton, selected_k.len());
    TS::lemma_tensor2d_shape_subrange(q, q.len(), h * __D__, start, end);
    TL::lemma_split_last_axis_shape(q, q.len(), h, __D__);
    TL::lemma_split_last_axis_shape(q.subrange(start, end), qlen, h, __D__);
    TL::lemma_split_last_axis_subrange(q, h, __D__, start, end);
    TL::lemma_split_last_axis_3d_shape(k, k.len(), 64, hkv, __D__);
    TL::lemma_split_last_axis_3d_shape(v, k.len(), 64, hkv, __D__);
    TL::lemma_split_last_axis_3d_shape(selected_k, selected_k.len(), 64, hkv, __D__);
    TL::lemma_split_last_axis_3d_shape(selected_v, selected_k.len(), 64, hkv, __D__);
    TL::lemma_tensor3d_allocation_shape(q.len(), h, __D__, fill);
    TL::lemma_tensor2d_allocation_shape(h, q.len(), fill);
    TL::lemma_tensor3d_allocation_shape(qlen, h, __D__, fill);
    TL::lemma_tensor2d_allocation_shape(h, qlen, fill);
    let left = adapter_side(q, k, v, table, cu_q, cu_k, max_q, h, hkv, scale, window, fill);
    let right = adapter_side(q.subrange(start, end), selected_k, selected_v, singleton,
        seq![0int, qlen as int], seq![0int, klen as int], qlen, h, hkv, scale, window, fill);
    let pages = common::blocks_needed_for(klen);
    // Logical cache contents agree; physical page IDs and pool sizes need not.
    assert forall|page: int| 0 <= page < pages implies
        (#[trigger] left.__PORT_KEYS__[left.__PORT_PAGE_TABLE__[i][page]])
            == right.__PORT_KEYS__[right.__PORT_PAGE_TABLE__[0][page]] by {
        PL::lemma_rectangular_cache_page_equality(k, table[i], PL::page_table_width(table),
            selected_k, selected_row, PL::page_table_width(singleton), pages, page as nat);
    }
    assert forall|page: int| 0 <= page < pages implies
        (#[trigger] left.__PORT_VALUES__[left.__PORT_PAGE_TABLE__[i][page]])
            == right.__PORT_VALUES__[right.__PORT_PAGE_TABLE__[0][page]] by {
        PL::lemma_rectangular_cache_page_equality(v, table[i], PL::page_table_width(table),
            selected_v, selected_row, PL::page_table_width(singleton), pages, page as nat);
    }
    let free = __FREE__ { __PORT_BATCH_SELECTOR__: i };
    assert(__PRE__(left, right, free));
    __CERT__(left, right, free);
    let a = __EXEC__(left);
    let b = __EXEC__(right);
    assert(__POST__(a, b, free));
    assert(TS::tensor3d_shape(a.__PORT_OUTPUT__, q.len(), h, __D__));
    assert(TS::tensor3d_shape(b.__PORT_OUTPUT__, qlen, h, __D__));
    assert forall|r: int, g: int, c: int|
        0 <= r < qlen && 0 <= g < h && 0 <= c < __D__ implies
            (#[trigger] a.__PORT_OUTPUT__[start + r][g][c]) == b.__PORT_OUTPUT__[r][g][c] by {
        assert(a.__PORT_OUTPUT__[a.__PORT_QUERY_OFFSETS__[i] + r][0 + g][0 + c]
            == b.__PORT_OUTPUT__[b.__PORT_QUERY_OFFSETS__[0] + r][0 + g][0 + c]);
    }
    TL::lemma_merge_last_axis_region_equality(a.__PORT_OUTPUT__, b.__PORT_OUTPUT__, start, 0, qlen, h, __D__);
    assert(TL::merge_last_axis(b.__PORT_OUTPUT__).subrange(0, qlen as int) =~= TL::merge_last_axis(b.__PORT_OUTPUT__));
}
}
