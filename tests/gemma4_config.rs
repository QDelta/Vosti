//! Host-only regression for the typed heterogeneous attention configuration.

use vosti_verus::model_config::{AttentionKind, DenseGeometry, FloatParameterBits};
use vosti_verus::boundary::model_families::gemma4::config::{Gemma4Config, gemma4_attention_geometry};

fn parameter(value: f64) -> FloatParameterBits {
    FloatParameterBits { bits: value.to_bits() }
}

fn config() -> Gemma4Config {
    Gemma4Config {
        geometry: DenseGeometry {
            vocab_size: 262144,
            hidden_size: 5376,
            intermediate_size: 21504,
            num_layers: 60,
            num_attention_heads: 32,
            num_key_value_heads: 16,
            head_dim: 256,
            max_position_embeddings: 262144,
        },
        num_global_key_value_heads: 4,
        global_head_dim: 512,
        rms_norm_epsilon: parameter(1e-6),
        sliding_window: 1024,
        local_rope_theta: parameter(10000.0),
        global_rope_theta: parameter(1000000.0),
        global_rope_factor: parameter(1.0),
        global_partial_rotary_factor: parameter(0.25),
        attention_k_eq_v: true,
        final_logit_softcap: Some(parameter(30.0)),
    }
}

#[test]
fn local_and_global_layers_preserve_their_own_geometry() {
    let config = config();
    let local = gemma4_attention_geometry(config, AttentionKind::SlidingWindow);
    let global = gemma4_attention_geometry(config, AttentionKind::Full);
    assert_eq!((local.num_attention_heads, local.num_key_value_heads, local.head_dim), (32, 16, 256));
    assert_eq!((global.num_attention_heads, global.num_key_value_heads, global.head_dim), (32, 4, 512));
    assert_eq!(config.global_partial_rotary_factor.bits, 0.25f64.to_bits());
    assert_eq!(config.final_logit_softcap.unwrap().bits, 30.0f64.to_bits());
}

#[test]
fn geometry_is_derived_from_config_not_a_profile_width_switch() {
    let mut config = config();
    config.geometry.num_attention_heads = 12;
    config.geometry.num_key_value_heads = 6;
    config.geometry.head_dim = 64;
    config.num_global_key_value_heads = 3;
    config.global_head_dim = 128;
    let local = gemma4_attention_geometry(config, AttentionKind::SlidingWindow);
    let global = gemma4_attention_geometry(config, AttentionKind::Full);
    assert_eq!((local.num_attention_heads, local.num_key_value_heads, local.head_dim), (12, 6, 64));
    assert_eq!((global.num_attention_heads, global.num_key_value_heads, global.head_dim), (12, 3, 128));
}
