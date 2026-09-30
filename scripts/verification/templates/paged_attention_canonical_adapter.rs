// A row-local semantic representative, constructed from the same raw execution
// model. Checked equivalence below connects it to a real singleton launch.
// No extra uninterpreted attention operation or equivalence axiom is introduced.
verus! {
pub open spec fn logical_prefix(cache: KVCacheLayerRepr, row: Seq<BlockId>, count: nat)
    -> Tensor2D
{
    AP::logical_prefix(cache, row, count)
}

pub open spec fn packed_prefix(prefix: Tensor2D, padding: crate::proof::tensor::types::Tensor1D)
    -> KVCacheLayerRepr
{
    Seq::new(common::blocks_needed_for(prefix.len()), |page: int|
        Seq::new(64, |offset: int| if page * 64 + offset < prefix.len() {
            prefix[page * 64 + offset]
        } else { padding }))
}

pub open spec fn packed_table(count: nat) -> Seq<BlockId> {
    Seq::new(common::blocks_needed_for(count), |page: int| page as BlockId)
}

pub proof fn lemma_packed_prefix(prefix: Tensor2D, padding: crate::proof::tensor::types::Tensor1D, width: nat)
    requires
        TS::tensor2d_shape(prefix, prefix.len(), width), padding.len() == width,
        0 < prefix.len(), common::blocks_needed_for(prefix.len()) <= u64::MAX,
    ensures
        TS::tensor3d_shape(packed_prefix(prefix, padding),
            common::blocks_needed_for(prefix.len()), 64, width),
        PL::page_table_ids_valid(seq![packed_table(prefix.len())],
            common::blocks_needed_for(prefix.len())),
        forall|pos: nat| pos < prefix.len() ==>
            (#[trigger] common::cache_at(packed_prefix(prefix, padding),
                common::block_table_slot(packed_table(prefix.len()), pos))) == prefix[pos as int],
{
    let cache = packed_prefix(prefix, padding);
    let row = packed_table(prefix.len());
    assert forall|page: int| 0 <= page < cache.len() implies
        TS::tensor2d_shape(#[trigger] cache[page], 64, width) by {
        assert forall|offset: int| 0 <= offset < 64 implies
            (#[trigger] cache[page][offset]).len() == width by {}
    }
    assert forall|pos: nat| pos < prefix.len() implies
        (#[trigger] common::cache_at(cache, common::block_table_slot(row, pos))) == prefix[pos as int] by {
        let page = pos / 64;
        assert(page < row.len());
        assert(row[page as int] == page);
        assert(common::block_table_slot(row, pos) == pos);
        assert(pos == page * 64 + pos % 64);
    }
}

pub open spec fn canonical_row_side(
    q: crate::proof::tensor::types::Tensor1D, k: Tensor2D, v: Tensor2D,
    h: nat, hkv: nat, scale: Scalar, window: nat, fill: Scalar,
) -> __SIDE__ {
    let padding = Seq::new(hkv * __D__, |col: int| fill);
    adapter_side(seq![q], packed_prefix(k, padding), packed_prefix(v, padding),
        seq![packed_table(k.len())], seq![0int, 1int], seq![0int, k.len() as int],
        1, h, hkv, scale, window, fill)
}

#[verifier::opaque]
pub open spec fn canonical_row_output(
    q: crate::proof::tensor::types::Tensor1D, k: Tensor2D, v: Tensor2D,
    h: nat, hkv: nat, scale: Scalar, window: nat, fill: Scalar,
) -> crate::proof::tensor::types::Tensor1D {
    TL::merge_last_axis(__EXEC__(canonical_row_side(q, k, v, h, hkv, scale, window, fill)).__PORT_OUTPUT__)[0]
}

pub open spec fn canonical_row_operation(
    h: nat, hkv: nat, scale: Scalar, window: nat, fill: Scalar,
) -> AP::RowOperation {
    |q: crate::proof::tensor::types::Tensor1D, k: Tensor2D, v: Tensor2D|
        canonical_row_output(q, k, v, h, hkv, scale, window, fill)
}

pub proof fn lemma_canonical_row_shape(
    q: crate::proof::tensor::types::Tensor1D, k: Tensor2D, v: Tensor2D,
    h: nat, hkv: nat, scale: Scalar, window: nat, fill: Scalar,
)
    ensures canonical_row_output(q, k, v, h, hkv, scale, window, fill).len() == h * __D__,
{
    reveal(canonical_row_output);
    let result = __EXEC__(canonical_row_side(q, k, v, h, hkv, scale, window, fill));
    TL::lemma_tensor3d_allocation_shape(1, h, __D__, fill);
    assert(TS::tensor3d_shape(result.__PORT_OUTPUT__, 1, h, __D__));
    TL::lemma_merge_last_axis_shape(result.__PORT_OUTPUT__, 1, h, __D__);
}

pub proof fn checked_canonical_selected_projection(
    q: Tensor2D, k: KVCacheLayerRepr, v: KVCacheLayerRepr, row: Seq<BlockId>, kl: nat,
    j: int, h: nat, hkv: nat, scale: Scalar, window: nat, fill: Scalar,
)
    requires
        TS::tensor2d_shape(q, q.len(), h * __D__),
        TS::tensor3d_shape(k, k.len(), 64, hkv * __D__),
        TS::tensor3d_shape(v, k.len(), 64, hkv * __D__),
        k.len() > 0, PL::page_table_ids_valid(seq![row], k.len()),
        common::blocks_needed_for(kl) <= row.len(),
        common::blocks_needed_for(kl) <= u64::MAX,
        0 <= j < q.len() <= kl,
        h > 0, hkv > 0, hkv <= h, h % hkv == 0,
        __WINDOW_REQUIREMENT__
        selected_runtime_assumptions(
            adapter_side(q, k, v, seq![row], seq![0int, q.len() as int], seq![0int, kl as int],
                q.len(), h, hkv, scale, window, fill),
            canonical_row_side(q[j], logical_prefix(k, row, (kl - q.len() + j + 1) as nat),
                logical_prefix(v, row, (kl - q.len() + j + 1) as nat),
                h, hkv, scale, window, fill),
            __SELECTED_FREE__ { __PORT_LEFT_SELECTOR__: j, __PORT_RIGHT_SELECTOR__: 0 }),
    ensures
        adapter_output(q, k, v, seq![row],
            seq![0int, q.len() as int], seq![0int, kl as int], q.len(),
            h, hkv, scale, window, fill)[j]
        == canonical_row_output(q[j], logical_prefix(k, row, (kl - q.len() + j + 1) as nat),
            logical_prefix(v, row, (kl - q.len() + j + 1) as nat), h, hkv, scale, window, fill),
{
    reveal(adapter_output);
    reveal(canonical_row_output);
    let count = (kl - q.len() + j + 1) as nat;
    let kp = logical_prefix(k, row, count);
    let vp = logical_prefix(v, row, count);
    let padding = Seq::new(hkv * __D__, |col: int| fill);
    assert(0 < count <= kl);
    assert(common::blocks_needed_for(count) <= common::blocks_needed_for(kl));
    assert forall|pos: int| 0 <= pos < count implies
        (#[trigger] kp[pos]).len() == hkv * __D__ && vp[pos].len() == hkv * __D__ by {
        let page = pos / 64;
        assert(page < common::blocks_needed_for(kl));
        assert(row[page] < k.len());
        PL::lemma_rectangular_cache_lookup(k, row, row.len(), page as nat, (pos % 64) as nat);
        PL::lemma_rectangular_cache_lookup(v, row, row.len(), page as nat, (pos % 64) as nat);
    }
    lemma_packed_prefix(kp, padding, hkv * __D__);
    lemma_packed_prefix(vp, padding, hkv * __D__);
    checked_selected_row_projection(q, k, v, row, kl,
        seq![q[j]], packed_prefix(kp, padding), packed_prefix(vp, padding), packed_table(count), count,
        j, 0, h, hkv, scale, window, fill);
}
}
