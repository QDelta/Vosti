//! Pure tensor and KV-cache representations used by specifications and proofs.

pub use crate::boundary::scalar::Scalar;
use vstd::prelude::*;

verus! {

// Semantic tensor type aliases — pure ghost values used by repr predicates.
pub type Tensor1D = Seq<Scalar>;

pub type Tensor2D = Seq<Tensor1D>;

pub type Tensor3D = Seq<Tensor2D>;

pub type Tensor4D = Seq<Tensor3D>;

pub type IntTensor1D = Seq<int>;

pub type IntTensor2D = Seq<IntTensor1D>;

// KV cache per layer: (num_pages, BLOCK_SIZE, D).
//   cache[page_id][offset] gives a Tensor1D (one K or V vector).
pub type KVCacheLayerRepr = Seq<Tensor2D>;

} // verus!
