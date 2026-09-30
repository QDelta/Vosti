// Exact analyzer predicates, not a second interpretation of their labels.
verus! {
pub open spec fn selected_runtime_assumptions(
    left: __SIDE__, right: __SIDE__, free: __SELECTED_FREE__,
) -> bool {
    __SELECTED_CONDITIONS__
}

pub proof fn checked_selected_row_projection(
    qa: Tensor2D, ka: KVCacheLayerRepr, va: KVCacheLayerRepr, row_a: Seq<BlockId>, kl_a: nat,
    qb: Tensor2D, kb: KVCacheLayerRepr, vb: KVCacheLayerRepr, row_b: Seq<BlockId>, kl_b: nat,
    ja: int, jb: int, h: nat, hkv: nat, scale: Scalar, window: nat, fill: Scalar,
)
    requires
        TS::tensor2d_shape(qa, qa.len(), h * __D__),
        TS::tensor2d_shape(qb, qb.len(), h * __D__),
        TS::tensor3d_shape(ka, ka.len(), 64, hkv * __D__),
        TS::tensor3d_shape(va, ka.len(), 64, hkv * __D__),
        TS::tensor3d_shape(kb, kb.len(), 64, hkv * __D__),
        TS::tensor3d_shape(vb, kb.len(), 64, hkv * __D__),
        ka.len() > 0, kb.len() > 0,
        PL::page_table_ids_valid(seq![row_a], ka.len()),
        PL::page_table_ids_valid(seq![row_b], kb.len()),
        common::blocks_needed_for(kl_a) <= row_a.len(),
        common::blocks_needed_for(kl_b) <= row_b.len(),
        0 <= ja < qa.len() <= kl_a, 0 <= jb < qb.len() <= kl_b,
        kl_a - qa.len() + ja == kl_b - qb.len() + jb,
        qa[ja] == qb[jb],
        h > 0, hkv > 0, hkv <= h, h % hkv == 0,
        __WINDOW_REQUIREMENT__
        forall|pos: nat| pos <= kl_a - qa.len() + ja ==>
            (#[trigger] common::cache_at(ka, common::block_table_slot(row_a, pos)))
                == common::cache_at(kb, common::block_table_slot(row_b, pos)),
        forall|pos: nat| pos <= kl_a - qa.len() + ja ==>
            (#[trigger] common::cache_at(va, common::block_table_slot(row_a, pos)))
                == common::cache_at(vb, common::block_table_slot(row_b, pos)),
        selected_runtime_assumptions(
            adapter_side(qa, ka, va, seq![row_a], seq![0int, qa.len() as int],
                seq![0int, kl_a as int], qa.len(), h, hkv, scale, window, fill),
            adapter_side(qb, kb, vb, seq![row_b], seq![0int, qb.len() as int],
                seq![0int, kl_b as int], qb.len(), h, hkv, scale, window, fill),
            __SELECTED_FREE__ { __PORT_LEFT_SELECTOR__: ja, __PORT_RIGHT_SELECTOR__: jb }),
    ensures
        adapter_output(qa, ka, va, seq![row_a],
            seq![0int, qa.len() as int], seq![0int, kl_a as int], qa.len(),
            h, hkv, scale, window, fill)[ja]
        == adapter_output(qb, kb, vb, seq![row_b],
            seq![0int, qb.len() as int], seq![0int, kl_b as int], qb.len(),
            h, hkv, scale, window, fill)[jb],
{
    reveal(adapter_output);
    PL::lemma_canonical_page_table(seq![row_a], ka.len());
    PL::lemma_canonical_page_table(seq![row_b], kb.len());
    TL::lemma_split_last_axis_shape(qa, qa.len(), h, __D__);
    TL::lemma_split_last_axis_shape(qb, qb.len(), h, __D__);
    TL::lemma_split_last_axis_row_equality(qa, qb, ja, jb, h, __D__);
    TL::lemma_split_last_axis_3d_shape(ka, ka.len(), 64, hkv, __D__);
    TL::lemma_split_last_axis_3d_shape(va, ka.len(), 64, hkv, __D__);
    TL::lemma_split_last_axis_3d_shape(kb, kb.len(), 64, hkv, __D__);
    TL::lemma_split_last_axis_3d_shape(vb, kb.len(), 64, hkv, __D__);
    TL::lemma_tensor3d_allocation_shape(qa.len(), h, __D__, fill);
    TL::lemma_tensor3d_allocation_shape(qb.len(), h, __D__, fill);
    TL::lemma_tensor2d_allocation_shape(h, qa.len(), fill);
    TL::lemma_tensor2d_allocation_shape(h, qb.len(), fill);
    let left = adapter_side(qa, ka, va, seq![row_a], seq![0int, qa.len() as int],
        seq![0int, kl_a as int], qa.len(), h, hkv, scale, window, fill);
    let right = adapter_side(qb, kb, vb, seq![row_b], seq![0int, qb.len() as int],
        seq![0int, kl_b as int], qb.len(), h, hkv, scale, window, fill);
    let prefix = kl_a - qa.len() + ja + 1;
    assert forall|page: int, offset: int|
        0 <= page && 0 <= offset < 64 && page * 64 + offset < prefix implies
        (#[trigger] left.__PORT_KEYS__[left.__PORT_PAGE_TABLE__[0][page]][offset])
            == right.__PORT_KEYS__[right.__PORT_PAGE_TABLE__[0][page]][offset] by {
        let pos = (page * 64 + offset) as nat;
        assert(pos / 64 == page);
        assert(page < common::blocks_needed_for(kl_a));
        assert(page < common::blocks_needed_for(kl_b));
        PL::lemma_rectangular_cache_lookup(ka, row_a, PL::page_table_width(seq![row_a]),
            page as nat, offset as nat);
        PL::lemma_rectangular_cache_lookup(kb, row_b, PL::page_table_width(seq![row_b]),
            page as nat, offset as nat);
    }
    assert forall|page: int, offset: int|
        0 <= page && 0 <= offset < 64 && page * 64 + offset < prefix implies
        (#[trigger] left.__PORT_VALUES__[left.__PORT_PAGE_TABLE__[0][page]][offset])
            == right.__PORT_VALUES__[right.__PORT_PAGE_TABLE__[0][page]][offset] by {
        let pos = (page * 64 + offset) as nat;
        assert(pos / 64 == page);
        assert(page < common::blocks_needed_for(kl_a));
        assert(page < common::blocks_needed_for(kl_b));
        PL::lemma_rectangular_cache_lookup(va, row_a, PL::page_table_width(seq![row_a]),
            page as nat, offset as nat);
        PL::lemma_rectangular_cache_lookup(vb, row_b, PL::page_table_width(seq![row_b]),
            page as nat, offset as nat);
    }
    let free = __SELECTED_FREE__ { __PORT_LEFT_SELECTOR__: ja, __PORT_RIGHT_SELECTOR__: jb };
    assert(__SELECTED_PRE__(left, right, free));
    __SELECTED_CERT__(left, right, free);
    let a = __EXEC__(left);
    let b = __EXEC__(right);
    assert(__SELECTED_POST__(a, b, free));
    assert(TS::tensor3d_shape(a.__PORT_OUTPUT__, qa.len(), h, __D__));
    assert(TS::tensor3d_shape(b.__PORT_OUTPUT__, qb.len(), h, __D__));
    assert forall|r: int, g: int, c: int|
        0 <= r < 1 && 0 <= g < h && 0 <= c < __D__ implies
            (#[trigger] a.__PORT_OUTPUT__[ja + r][g][c]) == b.__PORT_OUTPUT__[jb + r][g][c] by {
        assert(a.__PORT_OUTPUT__[free.__PORT_LEFT_SELECTOR__ + r][0 + g][0 + c]
            == b.__PORT_OUTPUT__[free.__PORT_RIGHT_SELECTOR__ + r][0 + g][0 + c]);
    }
    TL::lemma_merge_last_axis_region_equality(a.__PORT_OUTPUT__, b.__PORT_OUTPUT__, ja, jb, 1, h, __D__);
    assert(TL::merge_last_axis(a.__PORT_OUTPUT__).subrange(ja, ja + 1)[0]
        == TL::merge_last_axis(b.__PORT_OUTPUT__).subrange(jb, jb + 1)[0]);
}
}
