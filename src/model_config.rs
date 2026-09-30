//! Shared executable model-configuration schema.
//!
//! Family-specific checkpoint records live beside their family adapters.
//! This module defines data only; semantic projections live in `proof`.

use vstd::prelude::*;

verus! {

// Closed set of supported model architectures.
pub enum ModelArchitecture {
    Qwen3,
    Llama3,
    Gemma3Text,
    Gemma4Text,
}

// Architecture-neutral attention policy shared by dense decoder families.
// A family decides which policy each layer uses; the attention layer and its
// proofs do not otherwise depend on the family name.
#[derive(Clone, Copy)]
pub enum AttentionKind {
    SlidingWindow,
    Full,
}

// Executable checkpoint configuration retained beside the physical weights.
// The loader constructs these values only after selecting an exact supported
// profile.  Float parameters use their runtime bits so the executable and
// proof projections have one lossless identity.
#[derive(Clone, Copy)]
pub struct DenseGeometry {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub max_position_embeddings: usize,
}

/// Exact IEEE-754 binary64 identity; positivity and finiteness are validated
/// separately. The bits themselves carry no validity constraint.
#[derive(Clone, Copy)]
pub struct FloatParameterBits {
    pub bits: u64,
}

#[derive(Clone, Copy)]
pub struct AttentionGeometry {
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
}

pub struct ModelConfig {
    pub architecture: ModelArchitecture,
    pub num_layers: usize,
}

} // verus!
