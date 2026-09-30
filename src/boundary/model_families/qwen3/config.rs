//! Qwen3 checkpoint configuration and its fixed numerical identities.

use crate::model_config::{DenseGeometry, FloatParameterBits};
use vstd::prelude::*;

verus! {

#[derive(Clone, Copy)]
pub struct Qwen3Config {
    pub geometry: DenseGeometry,
    pub rms_norm_epsilon: FloatParameterBits,
    pub rope_theta: FloatParameterBits,
    pub tie_word_embeddings: bool,
}

pub const QWEN3_RMS_NORM_EPSILON_F64_BITS: u64 = 4_517_329_193_108_106_637; // 1e-6

pub const QWEN3_ROPE_THETA_F64_BITS: u64 = 4_696_837_146_684_686_336; // 1e6

} // verus!
