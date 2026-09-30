//! Gemma4 checkpoint configuration and its fixed numerical identities.

use crate::model_config::{AttentionGeometry, AttentionKind, DenseGeometry, FloatParameterBits};
#[cfg(verus_only)]
use crate::proof::model::types as repr;
use vstd::prelude::*;

verus! {

#[derive(Clone, Copy)]
pub struct Gemma4Config {
    // The common geometry carries the local attention shape. Global layers
    // have their own KV head count and width; neither is runtime-selected.
    pub geometry: DenseGeometry,
    pub num_global_key_value_heads: usize,
    pub global_head_dim: usize,
    pub rms_norm_epsilon: FloatParameterBits,
    pub sliding_window: usize,
    pub local_rope_theta: FloatParameterBits,
    pub global_rope_theta: FloatParameterBits,
    pub global_rope_factor: FloatParameterBits,
    pub global_partial_rotary_factor: FloatParameterBits,
    pub attention_k_eq_v: bool,
    pub final_logit_softcap: Option<FloatParameterBits>,
}

pub fn gemma4_attention_geometry(config: Gemma4Config, kind: AttentionKind)
    -> (out: AttentionGeometry)
    ensures repr::physical_attention_geometry_repr(out) == repr::gemma4_layer_attention_geometry(
        repr::gemma4_deployment_config_repr(config, Seq::empty()), kind),
{
    match kind {
        AttentionKind::Full => AttentionGeometry {
            num_attention_heads: config.geometry.num_attention_heads,
            num_key_value_heads: config.num_global_key_value_heads,
            head_dim: config.global_head_dim,
        },
        AttentionKind::SlidingWindow => AttentionGeometry {
            num_attention_heads: config.geometry.num_attention_heads,
            num_key_value_heads: config.geometry.num_key_value_heads,
            head_dim: config.geometry.head_dim,
        },
    }
}

} // verus!
