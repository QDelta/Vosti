// Concrete positive caller with no cache/geometry assumptions. Scalar values
// remain opaque and arbitrary. Two nonuniform pages are relocated into a
// three-page pool; the selected request uses 65 tokens across two full pages.
verus! {
proof fn concrete_relocation(a: Scalar, b: Scalar) {
    let first = Seq::new(64, |slot: int| Seq::new(__D__, |col: int| a));
    let second = Seq::new(64, |slot: int| Seq::new(__D__, |col: int|
        if slot == 0 { a } else { b }));
    let k = seq![first, second];
    let selected_k = seq![first, first, second];
    let table = seq![seq![0u64], seq![1u64, 0u64]];
    let selected_row = seq![2u64, 0u64];
    let q = Seq::new(2, |r: int| Seq::new(__D__, |col: int| a));
    let cu_q = seq![0int, 1int, 2int];
    let cu_k = seq![0int, 64int, 129int];
    assert(SUP::paged_attention_metadata_ready(2, 2, cu_q, cu_k, 1, 65, table));
    assert(TS::tensor2d_shape(q, 2, __D__));
    assert(TS::tensor3d_shape(k, 2, 64, __D__));
    assert(TS::tensor3d_shape(selected_k, 3, 64, __D__));
    assert(PL::page_table_ids_valid(seq![selected_row], 3));
    assert forall|pos: nat| pos < 128 implies
        (#[trigger] common::cache_at(k, common::block_table_slot(table[1], pos)))
            == common::cache_at(selected_k, common::block_table_slot(selected_row, pos)) by {
        if pos < 64 {
            assert(pos / 64 == 0);
            PL::lemma_rectangular_cache_lookup(k, table[1], 2, 0, pos);
            PL::lemma_rectangular_cache_lookup(selected_k, selected_row, 2, 0, pos);
        } else {
            assert(pos / 64 == 1);
            assert(pos == 64 + pos % 64);
            PL::lemma_rectangular_cache_lookup(k, table[1], 2, 1, pos % 64);
            PL::lemma_rectangular_cache_lookup(selected_k, selected_row, 2, 1, pos % 64);
        }
    }
    checked_batch_projection(q, k, k, table, cu_q, cu_k, 1, 65,
        1, 1, a, 32, b, 1, selected_k, selected_k, selected_row);
    // Shared pool/IDs require no second proof of attention composition.
    checked_batch_projection(q, k, k, table, cu_q, cu_k, 1, 65,
        1, 1, a, 32, b, 1, k, k, table[1]);
}
}
