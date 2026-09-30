//! Opaque deterministic sampler state and its trusted runtime initializer.

use vstd::prelude::*;

verus! {

#[verifier::external_body]
#[verifier::ext_equal]
#[derive(Clone, Copy)]
pub struct SamplerState {
    _private: (),
}

impl SamplerState {
    #[verifier::external_body]
    pub fn empty() -> (out: SamplerState) {
        SamplerState { _private: () }
    }
}

} // verus!
