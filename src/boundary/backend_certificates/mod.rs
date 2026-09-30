//! Generated backend proof certificates, separate from public model adapters.
//!
//! `support` is architecture-neutral vocabulary shared by generated interfaces.
//! Raw imported evidence is grouped by operator geometry, not model family;
//! exact static implementations remain separately qualified in their manifest.

pub mod attention;
pub mod linear;
pub mod qkv;
pub mod rms_norm;
pub mod residual_rms_norm;
pub mod offset_rms_norm;
pub mod add;
pub mod silu_mul;
pub mod gelu_tanh_mul;
pub mod scale;
pub mod softcap;
pub mod embedding;
pub mod scaled_embedding;
pub mod head_rms_norm;
pub mod offset_head_rms_norm;
pub mod rotary;
pub mod kv_store;
pub mod support;
