//! Qwen3 numerical specialization of the neutral dense-SwiGLU primitives.

use crate::{types::*, proof::model::types::*, proof::tensor::types::*};
use crate::boundary::tensor_runtime as RT;
use vstd::prelude::*;

verus! {

pub open spec fn forward_config_repr() -> DenseSwiGluForwardConfigRepr {
    DenseSwiGluForwardConfigRepr {
        rms_norm_epsilon: qwen3_rms_norm_epsilon_repr(),
        rotary: RT::qwen3_rotary_config_repr(),
    }
}

} // verus!
