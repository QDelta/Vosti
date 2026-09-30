//! Shared engine identities and compiled page constants.

use vstd::prelude::*;

verus! {

// Fixed implementation constraint, NOT a configurable page size. Python's
// counterpart is PAGE_SIZE in kernels/triton_kernels/constants.py. Changing
// both values is insufficient: review cache layouts, scheduler arithmetic/proofs,
// and kernel specializations; regenerate certificates and deployment evidence,
// then reverify and retest.
#[verifier::inline]
pub const BLOCK_SIZE: u64 = 64;

// Inline the derived value so page arithmetic retains literal-constant simplification.
#[verifier::inline]
pub spec const BLOCK_SIZE_SPEC: nat = BLOCK_SIZE as nat;

// Identity types.  `u64` so they're real at exec (a `Vec<RequestId>`
// stores actual ids, a `HashMapWithView<BlockId, _>` is constructible).
// In spec contexts u64 lifts to int the same way nat does, so most
// arithmetic is unchanged; the difference is that operations preserve
// nonnegativity by type construction rather than by predicate.
pub type RequestId = u64;

pub type TokenId = u64;

pub type BlockId = u64;

pub type SlotId = u64;

} // verus!
