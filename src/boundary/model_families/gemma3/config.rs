//! Gemma3 checkpoint configuration and its fixed numerical identities.

use crate::model_config::{DenseGeometry, FloatParameterBits};
use vstd::prelude::*;

verus! {

#[derive(Clone, Copy)]
pub struct Gemma3Config {
    pub geometry: DenseGeometry,
    pub rms_norm_epsilon: FloatParameterBits,
    pub query_pre_attention_scalar: FloatParameterBits,
    pub sliding_window: usize,
    pub local_rope_theta: FloatParameterBits,
    pub global_rope_theta: FloatParameterBits,
    pub global_rope_factor: FloatParameterBits,
}

pub const GEMMA3_RMS_NORM_EPSILON_F64_BITS: u64 = 4_517_329_193_108_106_637; // 1e-6

pub const GEMMA3_LOCAL_ROPE_THETA_F64_BITS: u64 = 4_666_723_172_467_343_360; // 1e4

pub const GEMMA3_GLOBAL_ROPE_THETA_F64_BITS: u64 = 4_696_837_146_684_686_336; // 1e6

pub const GEMMA3_GLOBAL_ROPE_FACTOR_F64_BITS: u64 = 4_620_693_217_682_128_896; // 8

} // verus!
