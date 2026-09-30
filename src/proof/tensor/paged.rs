//! Checked logical-to-rectangular page-table materialization.
//!
//! Shared by full and sliding-window attention, independently of model family.
//! Padding is a representation operation, not an attention equivalence axiom.
//! These lemmas are groundwork for the raw attention adapter; they do not yet
//! replace the production attention imports.

use crate::{types::{BlockId}, proof::tensor::types::{IntTensor2D, KVCacheLayerRepr}};
#[cfg(verus_only)]
use crate::{types::{BLOCK_SIZE_SPEC}, proof::tensor::geometry::{block_table_slot, cache_at}};
use crate::boundary::backend_certificates::support as SUP;
use vstd::prelude::*;

verus! {

pub open spec fn page_table_fits(table: Seq<Seq<BlockId>>, width: nat) -> bool {
    forall|row: int| 0 <= row < table.len() ==>
        (#[trigger] table[row]).len() <= width
}

pub open spec fn page_table_ids_valid(table: Seq<Seq<BlockId>>, pages: nat) -> bool {
    forall|row: int, col: int|
        0 <= row < table.len() && 0 <= col < table[row].len() ==>
            (#[trigger] table[row][col]) < pages
}

// The raw tensor type requires a nonempty rectangular column dimension, even
// for an empty logical row. This width is ghost representation metadata, not
// a runtime kernel configuration or a tuning decision.
pub open spec fn page_table_width(table: Seq<Seq<BlockId>>) -> nat
    decreases table.len(),
{
    if table.len() == 0 { 1 }
    else {
        let prefix_width = page_table_width(table.drop_last());
        if table.last().len() > prefix_width { table.last().len() }
        else { prefix_width }
    }
}

pub proof fn lemma_page_table_width(table: Seq<Seq<BlockId>>)
    ensures
        page_table_width(table) > 0,
        page_table_fits(table, page_table_width(table)),
    decreases table.len(),
{
    if table.len() > 0 {
        lemma_page_table_width(table.drop_last());
        assert forall|row: int| 0 <= row < table.len() implies
            (#[trigger] table[row]).len() <= page_table_width(table) by {
            if row < table.len() - 1 {
                assert(table.drop_last()[row] == table[row]);
            } else {
                assert(table[row] == table.last());
            }
        }
    }
}

// Zero is a valid physical page exactly when the cache pool is nonempty.
// Padding entries need not be read by attention, but must satisfy the raw
// annotation's all-column page-table validity premise nevertheless.
pub open spec fn rectangular_page_table(
    table: Seq<Seq<BlockId>>, width: nat,
) -> IntTensor2D {
    Seq::new(table.len(), |row: int| Seq::new(width, |col: int|
        if col < table[row].len() { table[row][col] as int } else { 0int }))
}

pub open spec fn rectangular_page_table_domain(
    table: IntTensor2D, rows: nat, width: nat, pages: nat,
) -> bool {
    &&& table.len() == rows
    &&& forall|row: int| 0 <= row < rows ==>
        (#[trigger] table[row]).len() == width
    &&& forall|row: int, col: int| 0 <= row < rows && 0 <= col < width ==>
        0 <= (#[trigger] table[row][col]) < pages
}

pub proof fn lemma_rectangular_page_table_domain(
    table: Seq<Seq<BlockId>>, width: nat, pages: nat,
)
    requires
        pages > 0,
        page_table_ids_valid(table, pages),
    ensures
        rectangular_page_table_domain(
            rectangular_page_table(table, width), table.len(), width, pages),
{
    let padded = rectangular_page_table(table, width);
    assert forall|row: int, col: int| 0 <= row < table.len() && 0 <= col < width implies
        0 <= (#[trigger] padded[row][col]) < pages by {
        if col < table[row].len() {
            assert(padded[row][col] == table[row][col] as int);
        } else {
            assert(padded[row][col] == 0);
        }
    }
}

pub proof fn lemma_rectangular_page_table_preserves_entries(
    table: Seq<Seq<BlockId>>, width: nat,
)
    requires page_table_fits(table, width),
    ensures
        forall|row: int, col: int|
            0 <= row < table.len() && 0 <= col < table[row].len() ==>
                (#[trigger] rectangular_page_table(table, width)[row][col])
                    == table[row][col] as int,
{
    assert forall|row: int, col: int|
        0 <= row < table.len() && 0 <= col < table[row].len() implies
            (#[trigger] rectangular_page_table(table, width)[row][col])
                == table[row][col] as int by {
        assert(table[row].len() <= width);
    }
}

pub proof fn lemma_canonical_page_table(
    table: Seq<Seq<BlockId>>, pages: nat,
)
    requires pages > 0, page_table_ids_valid(table, pages),
    ensures
        page_table_width(table) > 0,
        page_table_fits(table, page_table_width(table)),
        rectangular_page_table_domain(
            rectangular_page_table(table, page_table_width(table)),
            table.len(), page_table_width(table), pages),
        forall|row: int, col: int|
            0 <= row < table.len() && 0 <= col < table[row].len() ==>
                (#[trigger] rectangular_page_table(table, page_table_width(table))[row][col])
                    == table[row][col] as int,
{
    lemma_page_table_width(table);
    lemma_rectangular_page_table_domain(table, page_table_width(table), pages);
    lemma_rectangular_page_table_preserves_entries(table, page_table_width(table));
}

// Discharge the raw rectangular page-table domain from the actual engine
// metadata predicate, including all padded columns, not just live KV pages.
pub proof fn lemma_metadata_page_table(
    query_rows: nat, pages: nat,
    cu_q: Seq<int>, cu_k: Seq<int>, max_q: nat, max_k: nat,
    table: Seq<Seq<BlockId>>,
)
    requires
        SUP::paged_attention_metadata_ready(
            query_rows, pages, cu_q, cu_k, max_q, max_k, table),
    ensures
        page_table_ids_valid(table, pages),
        page_table_width(table) > 0,
        page_table_fits(table, page_table_width(table)),
        rectangular_page_table_domain(
            rectangular_page_table(table, page_table_width(table)),
            table.len(), page_table_width(table), pages),
        forall|row: int, col: int|
            0 <= row < table.len() && 0 <= col < table[row].len() ==>
                (#[trigger] rectangular_page_table(table, page_table_width(table))[row][col])
                    == table[row][col] as int,
{
    assert forall|row: int, col: int|
        0 <= row < table.len() && 0 <= col < table[row].len() implies
            (#[trigger] table[row][col]) < pages by {
        assert(cu_q[row] < cu_q[row + 1]);
    }
    lemma_canonical_page_table(table, pages);
}

// The raw batch annotation quantifies all ordered offset pairs, while the
// engine's metadata invariant supplies consecutive strict inequalities.
pub proof fn lemma_strict_offset_pairs(cu: Seq<int>, rows: nat)
    requires
        cu.len() >= rows + 1,
        forall|row: int| 0 <= row < rows ==> cu[row] < (#[trigger] cu[row + 1]),
    ensures
        forall|a: int, b: int| 0 <= a < b <= rows ==>
            (#[trigger] cu[a]) < (#[trigger] cu[b]),
{
    assert forall|a: int, b: int| 0 <= a < b <= rows implies
        (#[trigger] cu[a]) < (#[trigger] cu[b]) by {
        crate::proof::tensor::geometry::lemma_cu_mono(cu, rows as int, a + 1, b);
        assert(cu[a] < cu[a + 1]);
    }
}

pub proof fn lemma_metadata_strict_offsets(
    query_rows: nat, pages: nat,
    cu_q: Seq<int>, cu_k: Seq<int>, max_q: nat, max_k: nat,
    table: Seq<Seq<BlockId>>,
)
    requires
        SUP::paged_attention_metadata_ready(
            query_rows, pages, cu_q, cu_k, max_q, max_k, table),
    ensures
        forall|a: int, b: int| 0 <= a < b <= table.len() ==>
            (#[trigger] cu_q[a]) < (#[trigger] cu_q[b]),
        forall|a: int, b: int| 0 <= a < b <= table.len() ==>
            (#[trigger] cu_k[a]) < (#[trigger] cu_k[b]),
{
    assert forall|row: int| 0 <= row < table.len() implies
        cu_k[row] < (#[trigger] cu_k[row + 1]) by {
        assert(cu_q[row] < cu_q[row + 1]);
    }
    lemma_strict_offset_pairs(cu_q, table.len());
    lemma_strict_offset_pairs(cu_k, table.len());
}

pub proof fn lemma_rectangular_cache_lookup(
    cache: KVCacheLayerRepr, row: Seq<BlockId>, width: nat,
    page: nat, offset: nat,
)
    requires
        page < row.len(), page < width,
        row[page as int] < cache.len(),
        cache[row[page as int] as int].len() == BLOCK_SIZE_SPEC,
        offset < BLOCK_SIZE_SPEC,
    ensures
        cache_at(cache, block_table_slot(row, page * BLOCK_SIZE_SPEC + offset))
            == cache[rectangular_page_table(seq![row], width)[0][page as int]][offset as int],
{
    let pos = page * BLOCK_SIZE_SPEC + offset;
    let physical_page = row[page as int] as int;
    assert(pos / BLOCK_SIZE_SPEC == page);
    assert(pos % BLOCK_SIZE_SPEC == offset);
    let slot = block_table_slot(row, pos);
    assert(slot == physical_page * BLOCK_SIZE_SPEC + offset);
    assert(slot / BLOCK_SIZE_SPEC == physical_page);
    assert(slot % BLOCK_SIZE_SPEC == offset);
}

// A complete logical-page equality becomes a physical-page regional equality
// after rectangular materialization. Include the final page's masked slots:
// this is deliberately not a window-only or token-prefix weakening.
pub proof fn lemma_rectangular_cache_page_equality(
    cache_a: KVCacheLayerRepr, row_a: Seq<BlockId>, width_a: nat,
    cache_b: KVCacheLayerRepr, row_b: Seq<BlockId>, width_b: nat,
    pages: nat, page: nat,
)
    requires
        page < pages,
        pages <= row_a.len(), pages <= width_a,
        pages <= row_b.len(), pages <= width_b,
        row_a[page as int] < cache_a.len(),
        row_b[page as int] < cache_b.len(),
        cache_a[row_a[page as int] as int].len() == BLOCK_SIZE_SPEC,
        cache_b[row_b[page as int] as int].len() == BLOCK_SIZE_SPEC,
        forall|pos: nat| pos < pages * BLOCK_SIZE_SPEC ==>
            (#[trigger] cache_at(cache_a, block_table_slot(row_a, pos)))
                == cache_at(cache_b, block_table_slot(row_b, pos)),
    ensures
        cache_a[rectangular_page_table(seq![row_a], width_a)[0][page as int]]
            == cache_b[rectangular_page_table(seq![row_b], width_b)[0][page as int]],
{
    let a = cache_a[rectangular_page_table(seq![row_a], width_a)[0][page as int]];
    let b = cache_b[rectangular_page_table(seq![row_b], width_b)[0][page as int]];
    assert(a.len() == BLOCK_SIZE_SPEC && b.len() == BLOCK_SIZE_SPEC);
    assert forall|offset: int| 0 <= offset < BLOCK_SIZE_SPEC implies
        (#[trigger] a[offset]) == b[offset] by {
        let pos = page * BLOCK_SIZE_SPEC + offset as nat;
        assert(pos < pages * BLOCK_SIZE_SPEC);
        assert(cache_at(cache_a, block_table_slot(row_a, pos))
            == cache_at(cache_b, block_table_slot(row_b, pos)));
        lemma_rectangular_cache_lookup(cache_a, row_a, width_a, page, offset as nat);
        lemma_rectangular_cache_lookup(cache_b, row_b, width_b, page, offset as nat);
    }
    assert(a =~= b);
}

} // verus!
