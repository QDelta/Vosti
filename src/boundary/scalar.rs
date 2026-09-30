//! Opaque scalar identity used by the semantic tensor model.

use crate::model_config::FloatParameterBits;
use vstd::prelude::*;

verus! {

// Kernel values are intentionally opaque: the proof relates executions and
// cache placement without formalizing floating-point arithmetic.
#[verifier::external_body]
#[verifier::ext_equal]
pub struct Scalar {
    _private: (),
}

// Float-valued checkpoint parameters remain exactly identified by their
// binary64 bits in the pure model. Their numerical interpretation is part of
// this explicit trusted scalar boundary, not an unconstrained family atom.
pub uninterp spec fn positive_float_parameter_valid(
    parameter: FloatParameterBits,
) -> bool;

pub uninterp spec fn float_parameter_scalar_repr(
    parameter: FloatParameterBits,
) -> Scalar;

} // verus!
