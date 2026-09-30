//! Llama3 checkpoint configuration and its fixed numerical identities.

use crate::model_config::{DenseGeometry, FloatParameterBits};
use vstd::prelude::*;

verus! {

#[derive(Clone, Copy)]
pub struct Llama3Config {
    pub geometry: DenseGeometry,
    pub rms_norm_epsilon: FloatParameterBits,
    pub rope_theta: FloatParameterBits,
    pub rope_factor: FloatParameterBits,
    pub rope_low_frequency_factor: FloatParameterBits,
    pub rope_high_frequency_factor: FloatParameterBits,
    pub rope_original_max_position_embeddings: usize,
    pub tie_word_embeddings: bool,
}

pub const LLAMA3_RMS_NORM_EPSILON_F64_BITS: u64 = 4_532_020_583_610_935_537; // 1e-5

pub const LLAMA3_ROPE_THETA_F64_BITS: u64 = 4_692_333_547_057_315_840; // 500000

pub const LLAMA3_ROPE_FACTOR_8_F64_BITS: u64 = 4_620_693_217_682_128_896; // 8

pub const LLAMA3_ROPE_FACTOR_32_F64_BITS: u64 = 4_629_707_216_379_358_208; // 32

pub const LLAMA3_ROPE_LOW_FREQUENCY_FACTOR_F64_BITS: u64 = 4_607_182_418_800_017_408; // 1

pub const LLAMA3_ROPE_HIGH_FREQUENCY_FACTOR_F64_BITS: u64 = 4_616_189_618_054_758_400; // 4

pub const LLAMA3_ROPE_ORIGINAL_MAX_POSITION_EMBEDDINGS: usize = 8192;

} // verus!
