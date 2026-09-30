//! Checked bridge from annotated flat-row scatter to the paged cache model.
use vstd::prelude::*;
use crate::{proof::tensor::types::{Tensor2D, IntTensor1D, KVCacheLayerRepr}};
#[cfg(verus_only)]
use crate::{types::{BLOCK_SIZE_SPEC}, proof::tensor::geometry::{slot_in_cache}};
use crate::boundary::backend_certificates::kv_store as RAW;
use crate::boundary::tensor_runtime as RT;

verus! {

pub open spec fn flatten(cache: KVCacheLayerRepr) -> Tensor2D {
    Seq::new(cache.len() * BLOCK_SIZE_SPEC, |s: int|
        cache[s / BLOCK_SIZE_SPEC as int][s % BLOCK_SIZE_SPEC as int])
}

pub open spec fn restore(rows: Tensor2D, pages: nat) -> KVCacheLayerRepr {
    Seq::new(pages, |p: int| Seq::new(BLOCK_SIZE_SPEC, |o: int|
        rows[p * BLOCK_SIZE_SPEC as int + o]))
}

pub open spec fn domain(rows: Tensor2D, cache: KVCacheLayerRepr, slots: IntTensor1D, width: nat) -> bool {
    RAW::domain(rows, flatten(cache), slots, width)
}

pub open spec fn output(rows: Tensor2D, cache: KVCacheLayerRepr, slots: IntTensor1D, width: nat) -> KVCacheLayerRepr {
    restore(RAW::output(rows, flatten(cache), slots, width), cache.len())
}

pub proof fn checked_binding(
    kr: Tensor2D, vr: Tensor2D, old_k: KVCacheLayerRepr, old_v: KVCacheLayerRepr,
    slots: IntTensor1D, width: nat,
)
    requires
        RT::store_kv_cache_launch_ready(kr.len(), old_k, old_v, slots),
        kr.len() == vr.len(), vr.len() == slots.len(),
        domain(kr, old_k, slots, width), domain(vr, old_v, slots, width),
    ensures
        output(kr, old_k, slots, width) == RT::store_kv_cache_repr(kr, vr, old_k, old_v, slots).0,
        output(vr, old_v, slots, width) == RT::store_kv_cache_repr(kr, vr, old_k, old_v, slots).1,
{
    reveal(RT::store_kv_cache_launch_ready);
    reveal(RT::store_kv_cache_metadata_ready);
    reveal(RT::paged_cache_geometry);
    RT::lemma_store_kv_cache_repr_lengths(kr, vr, old_k, old_v, slots);
    let expected = RT::store_kv_cache_repr(kr, vr, old_k, old_v, slots);
    let kflat = flatten(old_k);
    let vflat = flatten(old_v);
    let kout = RAW::output(kr, kflat, slots, width);
    let vout = RAW::output(vr, vflat, slots, width);
    let result_k = restore(kout, old_k.len());
    let result_v = restore(vout, old_v.len());
    RAW::checked_shape(kr, kflat, slots, width);
    RAW::checked_shape(vr, vflat, slots, width);
    let bs = BLOCK_SIZE_SPEC as int;
    assert(bs > 0);
    assert forall|p: int| #![trigger result_k[p]] #![trigger result_v[p]]
        0 <= p < old_k.len() implies
        result_k[p] == expected.0[p] && result_v[p] == expected.1[p] by {
        assert(expected.0.len() == old_k.len() && expected.1.len() == old_v.len());
        assert(expected.0[p].len() == old_k[p].len());
        assert(expected.1[p].len() == old_v[p].len());
        assert(result_k[p].len() == bs && expected.0[p].len() == bs);
        assert(result_v[p].len() == bs && expected.1[p].len() == bs);
        assert forall|o: int| #![trigger result_k[p][o]] #![trigger result_v[p][o]]
            0 <= o < bs implies
            result_k[p][o] == expected.0[p][o] && result_v[p][o] == expected.1[p][o] by {
            let s = (p * bs + o) as nat;
            vstd::arithmetic::div_mod::lemma_div_multiples_vanish_fancy(p, o, bs);
            vstd::arithmetic::div_mod::lemma_mod_multiples_vanish(p, o, bs);
            vstd::arithmetic::div_mod::lemma_small_mod(o as nat, bs as nat);
            assert(s as int / bs == p && s as int % bs == o);
            assert(slot_in_cache(old_k, s) && slot_in_cache(old_v, s));
            if slots.contains(s as int) {
                let i = slots.index_of(s as int);
                assert forall|m: int| i < m < slots.len() implies slots[m] != slots[i] by {}
                RT::store_kv_cache_repr_reads_own_write(kr, vr, old_k, old_v, slots, i);
                RAW::checked_copy(kr, kflat, slots, width, i);
                RAW::checked_copy(vr, vflat, slots, width, i);
            } else {
                RT::store_kv_cache_repr_preserves_unwritten_slots(kr, vr, old_k, old_v, slots, s);
                assert forall|i: int| 0 <= i < slots.len() implies (#[trigger] slots[i]) != s as int by {}
                RAW::checked_frame(kr, kflat, slots, width, s as int);
                RAW::checked_frame(vr, vflat, slots, width, s as int);
            }
        }
        assert(result_k[p] =~= expected.0[p]);
        assert(result_v[p] =~= expected.1[p]);
    }
    assert(result_k =~= expected.0);
    assert(result_v =~= expected.1);
}

} // verus!
