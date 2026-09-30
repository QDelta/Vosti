//! Llama 3 numerical specialization of the neutral dense-SwiGLU primitives.

use crate::{types::*, proof::model::types::*, proof::tensor::types::*};
use vstd::prelude::*;

verus! {

pub open spec fn forward_config_repr(
    config: Llama3ModelWeightsExtensionRepr,
) -> DenseSwiGluForwardConfigRepr {
    dense_swiglu_forward_config_repr(config)
}

} // verus!
