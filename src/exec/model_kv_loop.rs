//! Architecture-neutral invariant for in-place model KV layer folds.

use crate::{types::*, proof::model::types::*, proof::tensor::types::*};
use crate::boundary::tensor_runtime as RT;
use vstd::prelude::*;

verus! {

/// Remaining KV-cache layers retain their pre-loop representations while a
/// family forward mutates the current prefix of the layer vector.
pub open spec fn suffix_matches(
    perms: &RT::KVCachePerms,
    expected: Seq<(KVCacheLayerRepr, KVCacheLayerRepr)>,
    start: int,
    end: int,
) -> bool {
    forall|layer: int| start <= layer < end ==> {
        &&& #[trigger] perms.k_repr(layer) == expected[layer].0
        &&& perms.v_repr(layer) == expected[layer].1
    }
}

pub proof fn suffix_matches_at(
    perms: &RT::KVCachePerms,
    expected: Seq<(KVCacheLayerRepr, KVCacheLayerRepr)>,
    start: int,
    end: int,
    layer: int,
)
    requires
        suffix_matches(perms, expected, start, end),
        start <= layer < end,
        0 <= layer < expected.len(),
    ensures
        perms.k_repr(layer) == expected[layer].0,
        perms.v_repr(layer) == expected[layer].1,
{
    reveal(suffix_matches);
    assert(perms.k_repr(layer) == expected[layer].0);
    assert(perms.v_repr(layer) == expected[layer].1);
}

} // verus!
