// Pure, reusable `Seq<Seq<A>>` partition lemmas.
//
// A flattened sequence of variable-length segments, indexed by a prefix-sum
// offset sequence `off` (with `off[0] == 0` and `off[j+1] == off[j] + segs[j].len()`),
// recovers each original segment as a subrange:
//
//     segs.flatten().subrange(off[i], off[i + 1]) == segs[i]
//
// This is the structural core of paged-attention *batch invariance*: when the
// per-request output segments are concatenated, request `i`'s output occupies
// exactly rows `[off[i], off[i+1])`, so it depends only on request `i`'s own
// rows.  Kept here as standalone `Seq` facts (no tensor/perm types) so they can
// be reused and verified in isolation.

use vstd::prelude::*;

verus! {

// Offsets are nondecreasing under the segment-length recurrence.
pub proof fn lemma_offsets_monotone<A>(segs: Seq<Seq<A>>, off: Seq<int>, a: int, b: int)
    requires
        off.len() == segs.len() + 1,
        forall|j: int| 0 <= j < segs.len() ==> #[trigger] off[j + 1] == off[j] + segs[j].len(),
        0 <= a <= b <= segs.len(),
    ensures
        off[a] <= off[b],
    decreases b - a,
{
    if a < b {
        lemma_offsets_monotone(segs, off, a, b - 1);
        assert(off[(b - 1) + 1] == off[b - 1] + segs[b - 1].len());
    }
}

// The length of the flattened sequence equals the final offset.
pub proof fn lemma_flatten_len_eq_off<A>(segs: Seq<Seq<A>>, off: Seq<int>)
    requires
        off.len() == segs.len() + 1,
        off[0] == 0,
        forall|j: int| 0 <= j < segs.len() ==> #[trigger] off[j + 1] == off[j] + segs[j].len(),
    ensures
        segs.flatten().len() == off[segs.len() as int],
    decreases segs.len(),
{
    if segs.len() == 0 {
    } else {
        let p = segs.first();
        let segs2 = segs.drop_first();
        let off2 = Seq::new(segs2.len() + 1, |j: int| off[j + 1] - p.len());
        assert(off[0int + 1] == off[0] + segs[0].len());
        assert(off2[0] == 0);
        assert(segs2.len() == segs.len() - 1);
        assert forall|j: int| 0 <= j < segs2.len() implies
            #[trigger] off2[j + 1] == off2[j] + segs2[j].len() by {
            assert(0 <= j + 1 < segs.len());
            assert(segs2[j] == segs[j + 1]);
            assert(off[(j + 1) + 1] == off[j + 1] + segs[j + 1].len());
        }
        lemma_flatten_len_eq_off(segs2, off2);
        assert(segs.flatten() == p.add(segs2.flatten()));
        assert(off2[segs2.len() as int] == off[segs.len() as int] - p.len());
    }
}

// Each segment is recovered as a subrange of the flattened sequence.
pub proof fn lemma_flatten_subrange_at_offsets<A>(segs: Seq<Seq<A>>, off: Seq<int>, i: int)
    requires
        off.len() == segs.len() + 1,
        off[0] == 0,
        forall|j: int| 0 <= j < segs.len() ==> #[trigger] off[j + 1] == off[j] + segs[j].len(),
        0 <= i < segs.len(),
    ensures
        segs.flatten().subrange(off[i], off[i + 1]) == segs[i],
    decreases segs.len(),
{
    let p = segs.first();
    let q = segs.drop_first().flatten();
    assert(off[0int + 1] == off[0] + segs[0].len());
    assert(segs.flatten() == p.add(q));
    if i == 0 {
        assert(p.add(q).subrange(off[0], off[1]) =~= segs[0]);
    } else {
        let segs2 = segs.drop_first();
        let off2 = Seq::new(segs2.len() + 1, |j: int| off[j + 1] - p.len());
        assert(off2[0] == 0);
        assert(segs2.len() == segs.len() - 1);
        assert forall|j: int| 0 <= j < segs2.len() implies
            #[trigger] off2[j + 1] == off2[j] + segs2[j].len() by {
            assert(0 <= j + 1 < segs.len());
            assert(segs2[j] == segs[j + 1]);
            assert(off[(j + 1) + 1] == off[j + 1] + segs[j + 1].len());
        }

        // Bounds: off[i] >= p.len(), and off[i+1] <= flatten length.
        lemma_offsets_monotone(segs, off, 1, i);
        lemma_offsets_monotone(segs, off, i + 1, segs.len() as int);
        lemma_flatten_len_eq_off(segs, off);
        assert(off[i] >= p.len());
        assert(segs.flatten().len() == p.len() + q.len());
        assert(off[i + 1] <= segs.flatten().len());

        // Inductive hypothesis on the tail.
        lemma_flatten_subrange_at_offsets(segs2, off2, i - 1);
        assert(off2[i - 1] == off[i] - p.len());
        assert(off2[i] == off[i + 1] - p.len());
        assert(segs2[i - 1] == segs[i]);

        // Shift the subrange past the first segment.
        assert(p.add(q).subrange(off[i], off[i + 1])
            =~= q.subrange(off[i] - p.len(), off[i + 1] - p.len()));
    }
}

// Fixed-width specialization used by token-to-head adapters.  Keeping the
// arithmetic here avoids duplicating offset proofs in every grouped launch.
pub proof fn lemma_fixed_width_flatten_group<A>(
    groups: Seq<Seq<A>>, width: nat, i: int,
)
    requires
        0 <= i < groups.len(),
        forall|j: int| 0 <= j < groups.len() ==>
            #[trigger] groups[j].len() == width,
    ensures
        groups.flatten().subrange(
            i * width as int, (i + 1) * width as int,
        ) == groups[i],
{
    let offsets = Seq::new(groups.len() + 1, |j: int| j * width as int);
    assert(offsets.len() == groups.len() + 1);
    assert(offsets[0] == 0);
    assert forall|j: int| 0 <= j < groups.len() implies
        #[trigger] offsets[j + 1] == offsets[j] + groups[j].len()
    by {
        assert(groups[j].len() == width);
        assert((j + 1) * width as int == j * width as int + width as int)
            by (nonlinear_arith)
    }
    lemma_fixed_width_flatten_len(groups, width);
    let lo = i * width as int;
    let hi = (i + 1) * width as int;
    assert(0 <= i);
    assert(i + 1 <= groups.len());
    assert(0 <= width as int);
    assert(0 <= lo) by (nonlinear_arith)
        requires 0 <= i, 0 <= width as int, lo == i * width as int,
    {}
    assert(lo <= hi) by (nonlinear_arith)
        requires 0 <= width as int,
            lo == i * width as int,
            hi == (i + 1) * width as int,
    {}
    assert(hi <= groups.flatten().len() as int) by (nonlinear_arith)
        requires i + 1 <= groups.len(),
            0 <= width as int,
            hi == (i + 1) * width as int,
            groups.flatten().len() as int
                == groups.len() as int * width as int,
    {}
    assert(0 <= lo <= hi <= groups.flatten().len());
    lemma_flatten_subrange_at_offsets(groups, offsets, i);
}

pub proof fn lemma_fixed_width_flatten_len<A>(groups: Seq<Seq<A>>, width: nat)
    requires
        forall|j: int| 0 <= j < groups.len() ==>
            #[trigger] groups[j].len() == width,
    ensures
        groups.flatten().len() as int
            == groups.len() as int * width as int,
{
    let offsets = Seq::new(groups.len() + 1, |j: int| j * width as int);
    assert(offsets.len() == groups.len() + 1);
    assert(offsets[0] == 0);
    assert forall|j: int| 0 <= j < groups.len() implies
        #[trigger] offsets[j + 1] == offsets[j] + groups[j].len()
    by {
        assert(groups[j].len() == width);
        assert((j + 1) * width as int == j * width as int + width as int)
            by (nonlinear_arith)
    }
    lemma_flatten_len_eq_off(groups, offsets);
    assert(offsets[groups.len() as int]
        == groups.len() as int * width as int);
}

// A sequence splits three ways around any two cut points.  Used to decompose a
// store's K/V/slot into `before ++ own ++ after` (request-major layout) so the
// disjointness lemmas apply.
pub proof fn lemma_seq_split3<A>(x: Seq<A>, a: int, b: int)
    requires 0 <= a <= b <= x.len(),
    ensures x == x.subrange(0, a) + x.subrange(a, b) + x.subrange(b, x.len() as int),
{
    assert(x =~= x.subrange(0, a) + x.subrange(a, b) + x.subrange(b, x.len() as int));
}

} // verus!
