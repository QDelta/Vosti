// Architecture-independent page geometry and sequence arithmetic.
//
// Opacity discipline: helpers that hide non-linear arithmetic (BlocksNeededFor,
// block_table_slot, slot_in_cache, cache_at) are `closed spec fn` so Z3 doesn't
// auto-unfold them outside focused proof blocks (`reveal_with_fuel` or
// `proof { reveal(...); }`).

use crate::types::BlockId;
#[cfg(verus_only)]
use crate::types::BLOCK_SIZE_SPEC;
use crate::proof::tensor::types::{KVCacheLayerRepr, Scalar, Tensor1D};
#[cfg(verus_only)]
use vstd::arithmetic::div_mod::lemma_fundamental_div_mod_converse_div;
use vstd::prelude::*;

verus! {

// Singleton geometry shared by the request machine and model compositions.
pub open spec fn slots_from(start: nat, count: nat) -> Seq<int> {
    Seq::new(count, |i: int| (start as int + i) as int)
}

pub open spec fn contiguous_block_ids(block_count: nat) -> Seq<BlockId> {
    Seq::new(block_count, |i: int| i as u64)
}

pub open spec fn singleton_block_rows(context_len: nat) -> Seq<Seq<BlockId>> {
    seq![contiguous_block_ids(blocks_needed_for(context_len))]
}

pub open spec fn seq_lens_for_single(length: nat) -> Seq<int> {
    seq![0int, length as int]
}

// Family-neutral empty cache geometry for a cold singleton forward.  The
// values stored in these cells are irrelevant because the full-history input
// overwrites every logical position before attention reads it; only the page
// shape is used by the reference construction.
pub open spec fn synthetic_cache_reprs(
    context_len: nat,
    num_layers: nat,
) -> Seq<(KVCacheLayerRepr, KVCacheLayerRepr)> {
    let pages = blocks_needed_for(context_len);
    let zero_layer: KVCacheLayerRepr = Seq::new(pages, |_p: int|
        Seq::new(BLOCK_SIZE_SPEC, |_o: int|
            Seq::<Scalar>::empty()));
    Seq::new(num_layers, |_i: int| (zero_layer, zero_layer))
}

// Number of physical blocks needed to hold `token_count` tokens.
//
// Open so other modules (e.g. batch_invariance) can reason about the body
// without a `reveal()` boundary.  At this codebase size the auto-unfolding
// cost is fine; if Z3 starts choking we can switch to `#[verifier::opaque]`
// + targeted `reveal` later.
pub open spec fn blocks_needed_for(token_count: nat) -> nat {
    if token_count == 0 { 0 }
    else { ((token_count - 1) / BLOCK_SIZE_SPEC as int + 1) as nat }
}

// If a position is within a sequence, the block holding it is within the
// block budget.  Single non-linear-arithmetic fact, exposed as a lemma.
pub proof fn lemma_blocks_needed_covers_pos(pos: nat, token_count: nat)
    requires pos < token_count,
    ensures pos / BLOCK_SIZE_SPEC < blocks_needed_for(token_count),
{
}

// Consecutive-monotone cu sequences are (non-strictly) monotone.
pub proof fn lemma_cu_mono(cu: Seq<int>, nrows: int, a: int, b: int)
    requires
        cu.len() >= nrows + 1,
        forall|j: int| 0 <= j < nrows ==> cu[j] < #[trigger] cu[j + 1],
        0 <= a <= b <= nrows,
    ensures
        cu[a] <= cu[b],
    decreases b - a,
{
    if a < b {
        lemma_cu_mono(cu, nrows, a + 1, b);
        assert(cu[a] < cu[a + 1]);
    }
}

// Every in-range flat index lands in exactly one cu row.
pub proof fn lemma_cu_locate(cu: Seq<int>, nrows: int, q: int) -> (k: int)
    requires
        nrows >= 1,
        cu.len() >= nrows + 1,
        cu[0] == 0,
        forall|j: int| 0 <= j < nrows ==> cu[j] < #[trigger] cu[j + 1],
        0 <= q < cu[nrows],
    ensures
        0 <= k < nrows,
        cu[k] <= q < cu[k + 1],
    decreases nrows,
{
    if q >= cu[nrows - 1] {
        nrows - 1
    } else {
        if nrows == 1 {
            assert(cu[0] <= q);
            0
        } else {
            let k = lemma_cu_locate(cu, nrows - 1, q);
            k
        }
    }
}

// The block budget is monotone in the token count.
pub proof fn lemma_blocks_needed_monotone(a: nat, b: nat)
    requires a <= b,
    ensures blocks_needed_for(a) <= blocks_needed_for(b),
{
    if a > 0 {
        vstd::arithmetic::div_mod::lemma_div_is_ordered(
            a as int - 1, b as int - 1, BLOCK_SIZE_SPEC as int);
    }
}

// A whole number of cache pages rounds to exactly that many pages.  The
// ragged attention ContractIR compares complete processed pages, so Gemma's
// cache-relocation proof uses this identity with
// `page_count = blocks_needed_for(k_len)`.
pub proof fn lemma_blocks_needed_for_full_pages(page_count: nat)
    ensures
        blocks_needed_for(page_count * BLOCK_SIZE_SPEC) == page_count,
{
    if page_count > 0 {
        vstd::arithmetic::div_mod::lemma_fundamental_div_mod_converse_div(
            (page_count * BLOCK_SIZE_SPEC - 1) as int,
            BLOCK_SIZE_SPEC as int,
            (page_count - 1) as int,
            BLOCK_SIZE_SPEC as int - 1,
        );
    }
}

// `[start, start+1, ..., start+count-1]`.  Returns Seq<int> so it lines up
// with `IntTensor1D = Seq<int>` used by the kernel reprs.  Element values
// are all non-negative — int instead of nat is a typing convenience, not a
// signedness change.
pub open spec fn positions_from(start: nat, count: nat) -> Seq<int> {
    Seq::new(count, |i: int| start as int + i)
}

// Block-table slot: physical slot index for a logical position within a row.
pub open spec fn block_table_slot(bt_row: Seq<BlockId>, pos: nat) -> nat
    recommends pos / BLOCK_SIZE_SPEC < bt_row.len(),
{
    (bt_row[pos as int / BLOCK_SIZE_SPEC as int] * BLOCK_SIZE_SPEC + (pos % BLOCK_SIZE_SPEC)) as nat
}

// Two-level cache indexing.  Bounds predicate then value accessor.
pub open spec fn slot_in_cache(cache: KVCacheLayerRepr, slot: nat) -> bool {
    slot / BLOCK_SIZE_SPEC < cache.len()
        && slot % BLOCK_SIZE_SPEC < cache[slot as int / BLOCK_SIZE_SPEC as int].len()
}

pub open spec fn cache_at(cache: KVCacheLayerRepr, slot: nat) -> Tensor1D
    recommends slot_in_cache(cache, slot),
{
    cache[slot as int / BLOCK_SIZE_SPEC as int][slot as int % BLOCK_SIZE_SPEC as int]
}

// Cumulative-sum upper bound: if cu[0..n] strictly increases, every prefix
// element is bounded by cu[n].  Used by decode-path proofs to thread the
// final-index bound through intermediate elements.
pub proof fn lemma_cu_int_bounds(cu: Seq<int>, n: int)
    requires
        0 <= n < cu.len(),
        cu[0] == 0,
        forall|j: int| 0 <= j < n ==> cu[j] < #[trigger] cu[j + 1],
    ensures
        forall|a: int| 0 <= a <= n ==> 0 <= #[trigger] cu[a] <= cu[n],
    decreases n,
{
    if n == 0 {
    } else {
        lemma_cu_int_bounds(cu, n - 1);
        assert(cu[(n - 1) + 1] == cu[n]);
        assert(cu[n - 1] < cu[(n - 1) + 1]);
        assert forall|a: int| 0 <= a <= n implies
            0 <= #[trigger] cu[a] <= cu[n]
        by {
            if a <= n - 1 {
            }
        }
    }
}

// A physical slot lives in the block named by its block-table entry: the slot
// `bt_row[pos/BLOCK_SIZE] * BLOCK_SIZE + pos%BLOCK_SIZE` divided by BLOCK_SIZE is
// exactly `bt_row[pos/BLOCK_SIZE]`.  Bridges slot arithmetic to block identity.
pub proof fn block_table_slot_block(bt_row: Seq<BlockId>, pos: nat)
    requires
        (pos as int) / (BLOCK_SIZE_SPEC as int) < bt_row.len(),
    ensures
        (block_table_slot(bt_row, pos) as int) / (BLOCK_SIZE_SPEC as int)
            == bt_row[(pos as int) / (BLOCK_SIZE_SPEC as int)] as int,
{
    let bs = BLOCK_SIZE_SPEC as int;
    let b = bt_row[(pos as int) / bs] as int;
    let r = (pos as int) % bs;
    assert(bs == 64);
    assert(0 <= r < bs);
    assert(block_table_slot(bt_row, pos) as int == b * bs + r);
    lemma_fundamental_div_mod_converse_div(b * bs + r, bs, b, r);
}

// Block disjointness ⇒ slot disjointness: if every entry of `slot_b` lies in a
// block different from the block holding slot `block_table_slot(bt_row, pos)`,
// then `slot_b` does not contain that slot.  This is how the scheduler's
// per-request block uniqueness discharges the KV-cache-agreement hypothesis:
// other requests' writes land in disjoint blocks, hence miss this request's
// read slots.
pub proof fn slots_miss_disjoint_block(slot_b: Seq<int>, bt_row: Seq<BlockId>, pos: nat)
    requires
        (pos as int) / (BLOCK_SIZE_SPEC as int) < bt_row.len(),
        forall|m: int| 0 <= m < slot_b.len() ==>
            #[trigger] slot_b[m] / (BLOCK_SIZE_SPEC as int)
                != bt_row[(pos as int) / (BLOCK_SIZE_SPEC as int)] as int,
    ensures
        !slot_b.contains(block_table_slot(bt_row, pos) as int),
{
    let s = block_table_slot(bt_row, pos) as int;
    block_table_slot_block(bt_row, pos);
    if slot_b.contains(s) {
        let m = slot_b.index_of(s);
        assert(slot_b[m] == s);
        assert(slot_b[m] / (BLOCK_SIZE_SPEC as int) == s / (BLOCK_SIZE_SPEC as int));
    }
}

pub proof fn lemma_cumsum_upper_bound(cu: Seq<nat>, n: nat)
    requires
        n < cu.len(),
        forall|j: int| 0 <= j < n as int ==> cu[j] < #[trigger] cu[j + 1],
    ensures
        forall|a: int| 0 <= a <= n as int ==> #[trigger] cu[a] <= cu[n as int],
    decreases n,
{
    if n == 0 {
    } else {
        lemma_cumsum_upper_bound(cu, (n - 1) as nat);
        // Trigger the strict-increase hypothesis at j = (n - 1) as int.
        let j = (n - 1) as int;
        assert(0 <= j < n as int);
        assert(cu[j + 1] > cu[j]);
        assert(j + 1 == n as int);
    }
}

} // verus!
