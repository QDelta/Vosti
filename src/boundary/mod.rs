//! Audited trust boundary and checked adapters for external execution.
//!
//! Every Verus `external_body`/uninterpreted declaration in the crate is
//! defined below this directory. Checked lemmas and wrappers may live beside
//! the declaration they constrain, but proof claims never originate here.

pub mod backend_certificates;
pub mod attention_operator;
pub mod linear_operator;
pub mod qkv_operator;
pub mod normalization_operator;
pub mod pointwise_operator;
pub mod embedding_operator;
pub mod head_normalization_operator;
pub mod rotary_operator;
pub mod kv_store_operator;
pub mod dense_layer_primitives;
pub mod dense_swiglu_decoder;
pub mod four_norm_gated_weights;
pub mod four_norm_gated_primitives;
pub mod model_deployment;
pub mod model_families;
pub mod model_forward_graph;
pub mod sampler;
pub mod scalar;
pub mod tensor_runtime;
