//! Family-neutral semantics for a pre-norm, full-attention SwiGLU decoder.
//!
//! Family adapters validate their closed composition and pass the exact RMS
//! epsilon and RoPE policy through `DenseSwiGluForwardConfigRepr`. Optional
//! Q/K normalization remains an honest part of each layer's weight record.

use crate::boundary::dense_layer_primitives as DLP;
#[cfg(verus_only)]
pub use crate::proof::tensor::geometry::{
    slots_from, singleton_block_rows,
    seq_lens_for_single, synthetic_cache_reprs,
};
use crate::{types::*, proof::model::types::*, proof::tensor::types::*};
#[cfg(verus_only)]
use crate::boundary::tensor_runtime as RT;
use vstd::prelude::*;

verus! {

pub open spec fn rms_norm_repr(
    config: DenseSwiGluForwardConfigRepr,
    input: Tensor2D,
    weight: Tensor1D,
) -> Tensor2D {
    RT::rms_norm_repr(input, weight, config.rms_norm_epsilon)
}

pub open spec fn add_rms_norm_repr(
    config: DenseSwiGluForwardConfigRepr,
    input: Tensor2D,
    residual: Tensor2D,
    weight: Tensor1D,
) -> (Tensor2D, Tensor2D) {
    RT::add_rms_norm_repr(input, residual, weight, config.rms_norm_epsilon)
}

pub open spec fn apply_qk_norm_repr(
    config: DenseSwiGluForwardConfigRepr,
    query: Tensor2D,
    key: Tensor2D,
    weights: QkNormWeightsRepr,
) -> (Tensor2D, Tensor2D) {
    RT::apply_qk_norm_repr(query, key, weights, config.rms_norm_epsilon)
}

pub open spec fn pre_attention_repr(
    config: DenseSwiGluForwardConfigRepr,
    wr: LayerWeightsRepr,
    normed_repr: Tensor2D,
    positions_repr: IntTensor1D,
) -> (Tensor2D, Tensor2D)
    recommends normed_repr.len() == positions_repr.len(),
{
    let qkv = RT::qkv_linear_repr(
        normed_repr, wr.q_proj, wr.k_proj, wr.v_proj,
    );
    let qr = qkv.0;
    let kr = qkv.1;
    let (nq, nk) = apply_qk_norm_repr(config, qr, kr, wr.qk_norm);
    DLP::rotary_embed_repr(
        DLP::layer_attention_geometry_repr(wr),
        config.rotary,
        positions_repr,
        nq,
        nk,
    )
}

pub open spec fn post_attention_repr(
    config: DenseSwiGluForwardConfigRepr,
    wr: LayerWeightsRepr,
    attn_repr: Tensor2D,
    residual_repr: Tensor2D,
) -> (Tensor2D, Tensor2D)
    recommends attn_repr.len() == residual_repr.len(),
{
    let projected = RT::linear_repr(attn_repr, wr.o_proj);
    let post = add_rms_norm_repr(config, projected, residual_repr, wr.post_attn_norm);
    let gate_up = RT::linear_repr(post.0, wr.gate_up_proj);
    let mlp = RT::silu_and_mul_repr(gate_up);
    let next_hidden = RT::linear_repr(mlp, wr.down_proj);
    (next_hidden, post.1)
}

pub open spec fn layer_kv_update_repr(
    config: DenseSwiGluForwardConfigRepr,
    wr: LayerWeightsRepr,
    normed_repr: Tensor2D,
    positions_repr: IntTensor1D,
    k_cache_repr: KVCacheLayerRepr,
    v_cache_repr: KVCacheLayerRepr,
    slot_repr: Seq<int>,
) -> (KVCacheLayerRepr, KVCacheLayerRepr)
    recommends
        normed_repr.len() == positions_repr.len(),
        slot_repr.len() == normed_repr.len(),
{
    let pre = pre_attention_repr(config, wr, normed_repr, positions_repr);
    let vr = RT::qkv_linear_repr(
        normed_repr, wr.q_proj, wr.k_proj, wr.v_proj,
    ).2;
    let vvr = RT::view_as_kv_repr(vr);
    RT::store_kv_cache_repr(pre.1, vvr, k_cache_repr, v_cache_repr, slot_repr)
}

pub open spec fn decoder_core_attention_launch_ready(
    config: DenseSwiGluForwardConfigRepr,
    wr: LayerWeightsRepr,
    normed_repr: Tensor2D,
    positions_repr: IntTensor1D,
    k_cache_repr: KVCacheLayerRepr,
    v_cache_repr: KVCacheLayerRepr,
    slot_repr: Seq<int>,
    cu_q_repr: Seq<int>,
    cu_k_repr: Seq<int>,
    max_seqlen_q: nat,
    max_seqlen_k: nat,
    bt_repr: Seq<Seq<BlockId>>,
) -> bool
    recommends
        normed_repr.len() == positions_repr.len(),
        slot_repr.len() == normed_repr.len(),
{
    let caches = layer_kv_update_repr(
        config,
        wr,
        normed_repr,
        positions_repr,
        k_cache_repr,
        v_cache_repr,
        slot_repr,
    );
    RT::paged_attention_launch_ready(
        normed_repr.len(),
        caches.0,
        caches.1,
        cu_q_repr,
        cu_k_repr,
        max_seqlen_q,
        max_seqlen_k,
        bt_repr,
    )
}

pub open spec fn decoder_layer_attention_launch_ready(
    config: DenseSwiGluForwardConfigRepr,
    wr: LayerWeightsRepr,
    hidden_repr: Tensor2D,
    residual_repr: Tensor2D,
    positions_repr: IntTensor1D,
    k_cache_repr: KVCacheLayerRepr,
    v_cache_repr: KVCacheLayerRepr,
    slot_repr: Seq<int>,
    cu_q_repr: Seq<int>,
    cu_k_repr: Seq<int>,
    max_seqlen_q: nat,
    max_seqlen_k: nat,
    bt_repr: Seq<Seq<BlockId>>,
) -> bool
    recommends
        hidden_repr.len() == residual_repr.len(),
        hidden_repr.len() == positions_repr.len(),
        slot_repr.len() == hidden_repr.len(),
{
    decoder_core_attention_launch_ready(
        config,
        wr,
        RT::add_rms_norm_repr(
            hidden_repr,
            residual_repr,
            wr.input_norm,
            config.rms_norm_epsilon,
        ).0,
        positions_repr,
        k_cache_repr,
        v_cache_repr,
        slot_repr,
        cu_q_repr,
        cu_k_repr,
        max_seqlen_q,
        max_seqlen_k,
        bt_repr,
    )
}

pub open spec fn first_decoder_layer_attention_launch_ready(
    config: DenseSwiGluForwardConfigRepr,
    wr: LayerWeightsRepr,
    hidden_repr: Tensor2D,
    positions_repr: IntTensor1D,
    k_cache_repr: KVCacheLayerRepr,
    v_cache_repr: KVCacheLayerRepr,
    slot_repr: Seq<int>,
    cu_q_repr: Seq<int>,
    cu_k_repr: Seq<int>,
    max_seqlen_q: nat,
    max_seqlen_k: nat,
    bt_repr: Seq<Seq<BlockId>>,
) -> bool
    recommends
        hidden_repr.len() == positions_repr.len(),
        slot_repr.len() == hidden_repr.len(),
{
    decoder_core_attention_launch_ready(
        config,
        wr,
        RT::rms_norm_repr(
            hidden_repr,
            wr.input_norm,
            config.rms_norm_epsilon,
        ),
        positions_repr,
        k_cache_repr,
        v_cache_repr,
        slot_repr,
        cu_q_repr,
        cu_k_repr,
        max_seqlen_q,
        max_seqlen_k,
        bt_repr,
    )
}

pub open spec fn decoder_core_output_repr(
    config: DenseSwiGluForwardConfigRepr,
    wr: LayerWeightsRepr,
    normed_repr: Tensor2D,
    residual_repr: Tensor2D,
    positions_repr: IntTensor1D,
    k_cache_repr: KVCacheLayerRepr,
    v_cache_repr: KVCacheLayerRepr,
    slot_repr: Seq<int>,
    cu_q_repr: Seq<int>,
    cu_k_repr: Seq<int>,
    max_seqlen_q: nat,
    max_seqlen_k: nat,
    bt_repr: Seq<Seq<BlockId>>,
) -> (Tensor2D, Tensor2D)
    recommends
        normed_repr.len() == residual_repr.len(),
        normed_repr.len() == positions_repr.len(),
        slot_repr.len() == normed_repr.len(),
{
    let pre = pre_attention_repr(config, wr, normed_repr, positions_repr);
    let vr = RT::qkv_linear_repr(
        normed_repr, wr.q_proj, wr.k_proj, wr.v_proj,
    ).2;
    let vvr = RT::view_as_kv_repr(vr);
    let new_caches = RT::store_kv_cache_repr(
        pre.1, vvr, k_cache_repr, v_cache_repr, slot_repr,
    );
    let attn = RT::paged_attention_repr(
        pre.0,
        new_caches.0,
        new_caches.1,
        cu_q_repr,
        cu_k_repr,
        max_seqlen_q,
        max_seqlen_k,
        bt_repr,
        crate::boundary::dense_layer_primitives::layer_attention_parameters_repr(wr, AttentionScaleRepr::InverseSqrtHeadDim),
    );
    post_attention_repr(config, wr, attn, residual_repr)
}

pub open spec fn first_decoder_layer_output_repr(
    config: DenseSwiGluForwardConfigRepr,
    wr: LayerWeightsRepr,
    hidden_repr: Tensor2D,
    positions_repr: IntTensor1D,
    k_cache_repr: KVCacheLayerRepr,
    v_cache_repr: KVCacheLayerRepr,
    slot_repr: Seq<int>,
    cu_q_repr: Seq<int>,
    cu_k_repr: Seq<int>,
    max_seqlen_q: nat,
    max_seqlen_k: nat,
    bt_repr: Seq<Seq<BlockId>>,
) -> (Tensor2D, Tensor2D)
    recommends
        hidden_repr.len() == positions_repr.len(),
        slot_repr.len() == hidden_repr.len(),
{
    let normed = RT::rms_norm_repr(
        hidden_repr,
        wr.input_norm,
        config.rms_norm_epsilon,
    );
    decoder_core_output_repr(
        config,
        wr,
        normed,
        hidden_repr,
        positions_repr,
        k_cache_repr,
        v_cache_repr,
        slot_repr,
        cu_q_repr,
        cu_k_repr,
        max_seqlen_q,
        max_seqlen_k,
        bt_repr,
    )
}

pub open spec fn decoder_layer_output_repr(
    config: DenseSwiGluForwardConfigRepr,
    wr: LayerWeightsRepr,
    hidden_repr: Tensor2D,
    residual_repr: Tensor2D,
    positions_repr: IntTensor1D,
    k_cache_repr: KVCacheLayerRepr,
    v_cache_repr: KVCacheLayerRepr,
    slot_repr: Seq<int>,
    cu_q_repr: Seq<int>,
    cu_k_repr: Seq<int>,
    max_seqlen_q: nat,
    max_seqlen_k: nat,
    bt_repr: Seq<Seq<BlockId>>,
) -> (Tensor2D, Tensor2D)
    recommends
        hidden_repr.len() == residual_repr.len(),
        hidden_repr.len() == positions_repr.len(),
        slot_repr.len() == hidden_repr.len(),
{
    let pre = RT::add_rms_norm_repr(
        hidden_repr,
        residual_repr,
        wr.input_norm,
        config.rms_norm_epsilon,
    );
    decoder_core_output_repr(
        config,
        wr,
        pre.0,
        pre.1,
        positions_repr,
        k_cache_repr,
        v_cache_repr,
        slot_repr,
        cu_q_repr,
        cu_k_repr,
        max_seqlen_q,
        max_seqlen_k,
        bt_repr,
    )
}

pub open spec fn layer_chain_repr(
    config: DenseSwiGluForwardConfigRepr,
    layers: Seq<LayerWeightsRepr>,
    hidden_repr: Tensor2D,
    residual_repr: Tensor2D,
    positions_repr: IntTensor1D,
    kv_cache_reprs: Seq<(KVCacheLayerRepr, KVCacheLayerRepr)>,
    slot_repr: Seq<int>,
    cu_q_repr: Seq<int>,
    cu_k_repr: Seq<int>,
    max_seqlen_q: nat,
    max_seqlen_k: nat,
    bt_repr: Seq<Seq<BlockId>>,
    start: nat,
) -> (Tensor2D, Tensor2D)
    recommends
        start <= layers.len(),
        hidden_repr.len() == residual_repr.len(),
        hidden_repr.len() == positions_repr.len(),
        slot_repr.len() == hidden_repr.len(),
        kv_cache_reprs.len() >= layers.len(),
    decreases (layers.len() - start) as nat,
{
    if start >= layers.len() {
        (hidden_repr, residual_repr)
    } else {
        let next = decoder_layer_output_repr(
            config,
            layers[start as int],
            hidden_repr,
            residual_repr,
            positions_repr,
            kv_cache_reprs[start as int].0,
            kv_cache_reprs[start as int].1,
            slot_repr,
            cu_q_repr,
            cu_k_repr,
            max_seqlen_q,
            max_seqlen_k,
            bt_repr,
        );
        layer_chain_repr(
            config,
            layers,
            next.0,
            next.1,
            positions_repr,
            kv_cache_reprs,
            slot_repr,
            cu_q_repr,
            cu_k_repr,
            max_seqlen_q,
            max_seqlen_k,
            bt_repr,
            (start + 1) as nat,
        )
    }
}

pub open spec fn model_forward_logits_repr(
    config: DenseSwiGluForwardConfigRepr,
    wr: ModelWeightsRepr,
    input_ids_repr: IntTensor1D,
    positions_repr: IntTensor1D,
    kv_cache_reprs: Seq<(KVCacheLayerRepr, KVCacheLayerRepr)>,
    slot_repr: Seq<int>,
    cu_q_repr: Seq<int>,
    cu_k_repr: Seq<int>,
    max_seqlen_q: nat,
    max_seqlen_k: nat,
    bt_repr: Seq<Seq<BlockId>>,
) -> Tensor2D
    recommends
        input_ids_repr.len() == positions_repr.len(),
        slot_repr.len() == input_ids_repr.len(),
        kv_cache_reprs.len() >= wr.layers.len(),
{
    let embed = RT::embed_repr(input_ids_repr, wr.embed_weight);
    if wr.layers.len() == 0 {
        let final_normed = RT::rms_norm_repr(
            embed,
            wr.final_norm,
            config.rms_norm_epsilon,
        );
        RT::linear_repr(final_normed, wr.lm_head)
    } else {
        let first = first_decoder_layer_output_repr(
            config,
            wr.layers[0],
            embed,
            positions_repr,
            kv_cache_reprs[0].0,
            kv_cache_reprs[0].1,
            slot_repr,
            cu_q_repr,
            cu_k_repr,
            max_seqlen_q,
            max_seqlen_k,
            bt_repr,
        );
        let chain = layer_chain_repr(
            config,
            wr.layers,
            first.0,
            first.1,
            positions_repr,
            kv_cache_reprs,
            slot_repr,
            cu_q_repr,
            cu_k_repr,
            max_seqlen_q,
            max_seqlen_k,
            bt_repr,
            1,
        );
        let final_pair = RT::add_rms_norm_repr(
            chain.0,
            chain.1,
            wr.final_norm,
            config.rms_norm_epsilon,
        );
        RT::linear_repr(final_pair.0, wr.lm_head)
    }
}


pub open spec fn reference_logits_last_row(
    config: DenseSwiGluForwardConfigRepr,
    wr: ModelWeightsRepr,
    history: IntTensor1D,
) -> Tensor1D
    recommends wr.layers.len() > 0, history.len() > 0,
{
    let n = history.len();
    let logits = model_forward_logits_repr(
        config,
        wr,
        history,
        crate::proof::tensor::geometry::positions_from(0, n),
        synthetic_cache_reprs(n, wr.layers.len()),
        slots_from(0, n),
        seq_lens_for_single(n),
        seq_lens_for_single(n),
        n,
        n,
        singleton_block_rows(n),
    );
    logits[n as int - 1]
}

pub broadcast proof fn lemma_pre_attention_repr_shape(
    config: DenseSwiGluForwardConfigRepr,
    wr: LayerWeightsRepr,
    normed_repr: Tensor2D,
    positions_repr: IntTensor1D,
)
    requires normed_repr.len() == positions_repr.len(),
    ensures
        (#[trigger] pre_attention_repr(
            config, wr, normed_repr, positions_repr,
        )).0.len() == normed_repr.len(),
        pre_attention_repr(config, wr, normed_repr, positions_repr).1.len()
            == normed_repr.len(),
{
    let qkv = RT::qkv_linear_repr(
        normed_repr, wr.q_proj, wr.k_proj, wr.v_proj,
    );
    let qr = qkv.0;
    let kr = qkv.1;
    RT::lemma_qkv_linear_repr_shape(
        normed_repr, wr.q_proj, wr.k_proj, wr.v_proj,
    );
    RT::lemma_apply_qk_norm_repr_shape(
        qr,
        kr,
        wr.qk_norm,
        config.rms_norm_epsilon,
    );
    let normalized = RT::apply_qk_norm_repr(
        qr,
        kr,
        wr.qk_norm,
        config.rms_norm_epsilon,
    );
    DLP::lemma_rotary_embed_repr_shape(
        DLP::layer_attention_geometry_repr(wr),
        config.rotary,
        positions_repr,
        normalized.0,
        normalized.1,
    );
}

pub broadcast proof fn lemma_post_attention_repr_shape(
    config: DenseSwiGluForwardConfigRepr,
    wr: LayerWeightsRepr,
    attn_repr: Tensor2D,
    residual_repr: Tensor2D,
)
    requires attn_repr.len() == residual_repr.len(),
    ensures
        (#[trigger] post_attention_repr(
            config, wr, attn_repr, residual_repr,
        )).0.len() == attn_repr.len(),
        post_attention_repr(config, wr, attn_repr, residual_repr).1.len()
            == attn_repr.len(),
{
    broadcast use RT::lemma_linear_repr_shape;
    broadcast use RT::lemma_add_rms_norm_repr_shape;
    let projected = RT::linear_repr(attn_repr, wr.o_proj);
    let post = RT::add_rms_norm_repr(
        projected,
        residual_repr,
        wr.post_attn_norm,
        config.rms_norm_epsilon,
    );
    let gate_up = RT::linear_repr(post.0, wr.gate_up_proj);
    RT::lemma_linear_repr_shape(attn_repr, wr.o_proj);
    RT::lemma_add_rms_norm_repr_shape(
        projected,
        residual_repr,
        wr.post_attn_norm,
        config.rms_norm_epsilon,
    );
    RT::lemma_linear_repr_shape(post.0, wr.gate_up_proj);
    RT::lemma_silu_and_mul_repr_shape(gate_up);
    RT::lemma_linear_repr_shape(
        RT::silu_and_mul_repr(gate_up),
        wr.down_proj,
    );
}

pub broadcast proof fn lemma_decoder_core_output_repr_shape(
    config: DenseSwiGluForwardConfigRepr,
    wr: LayerWeightsRepr,
    normed_repr: Tensor2D,
    residual_repr: Tensor2D,
    positions_repr: IntTensor1D,
    k_cache_repr: KVCacheLayerRepr,
    v_cache_repr: KVCacheLayerRepr,
    slot_repr: Seq<int>,
    cu_q_repr: Seq<int>,
    cu_k_repr: Seq<int>,
    max_seqlen_q: nat,
    max_seqlen_k: nat,
    bt_repr: Seq<Seq<BlockId>>,
)
    requires
        normed_repr.len() == residual_repr.len(),
        normed_repr.len() == positions_repr.len(),
        slot_repr.len() == normed_repr.len(),
    ensures
        (#[trigger] decoder_core_output_repr(
            config,
            wr,
            normed_repr,
            residual_repr,
            positions_repr,
            k_cache_repr,
            v_cache_repr,
            slot_repr,
            cu_q_repr,
            cu_k_repr,
            max_seqlen_q,
            max_seqlen_k,
            bt_repr,
        )).0.len() == normed_repr.len(),
        decoder_core_output_repr(
            config,
            wr,
            normed_repr,
            residual_repr,
            positions_repr,
            k_cache_repr,
            v_cache_repr,
            slot_repr,
            cu_q_repr,
            cu_k_repr,
            max_seqlen_q,
            max_seqlen_k,
            bt_repr,
        ).1.len() == normed_repr.len(),
{
    lemma_pre_attention_repr_shape(config, wr, normed_repr, positions_repr);
    let pre = pre_attention_repr(config, wr, normed_repr, positions_repr);
    let vr = RT::qkv_linear_repr(
        normed_repr, wr.q_proj, wr.k_proj, wr.v_proj,
    ).2;
    RT::lemma_qkv_linear_repr_shape(
        normed_repr, wr.q_proj, wr.k_proj, wr.v_proj,
    );
    let vvr = RT::view_as_kv_repr(vr);
    RT::lemma_view_as_kv_repr_shape(vr);
    let new_caches = RT::store_kv_cache_repr(
        pre.1, vvr, k_cache_repr, v_cache_repr, slot_repr,
    );
    let attn = RT::paged_attention_repr(
        pre.0,
        new_caches.0,
        new_caches.1,
        cu_q_repr,
        cu_k_repr,
        max_seqlen_q,
        max_seqlen_k,
        bt_repr,
        crate::boundary::dense_layer_primitives::layer_attention_parameters_repr(wr, AttentionScaleRepr::InverseSqrtHeadDim),
    );
    RT::lemma_paged_attention_repr_shape(
        pre.0,
        new_caches.0,
        new_caches.1,
        cu_q_repr,
        cu_k_repr,
        max_seqlen_q,
        max_seqlen_k,
        bt_repr,
        crate::boundary::dense_layer_primitives::layer_attention_parameters_repr(wr, AttentionScaleRepr::InverseSqrtHeadDim),
    );
    lemma_post_attention_repr_shape(config, wr, attn, residual_repr);
}

pub broadcast proof fn lemma_first_decoder_layer_output_repr_shape(
    config: DenseSwiGluForwardConfigRepr,
    wr: LayerWeightsRepr,
    hidden_repr: Tensor2D,
    positions_repr: IntTensor1D,
    k_cache_repr: KVCacheLayerRepr,
    v_cache_repr: KVCacheLayerRepr,
    slot_repr: Seq<int>,
    cu_q_repr: Seq<int>,
    cu_k_repr: Seq<int>,
    max_seqlen_q: nat,
    max_seqlen_k: nat,
    bt_repr: Seq<Seq<BlockId>>,
)
    requires
        hidden_repr.len() == positions_repr.len(),
        slot_repr.len() == hidden_repr.len(),
    ensures
        (#[trigger] first_decoder_layer_output_repr(
            config,
            wr,
            hidden_repr,
            positions_repr,
            k_cache_repr,
            v_cache_repr,
            slot_repr,
            cu_q_repr,
            cu_k_repr,
            max_seqlen_q,
            max_seqlen_k,
            bt_repr,
        )).0.len() == hidden_repr.len(),
        first_decoder_layer_output_repr(
            config,
            wr,
            hidden_repr,
            positions_repr,
            k_cache_repr,
            v_cache_repr,
            slot_repr,
            cu_q_repr,
            cu_k_repr,
            max_seqlen_q,
            max_seqlen_k,
            bt_repr,
        ).1.len() == hidden_repr.len(),
{
    let normed = RT::rms_norm_repr(
        hidden_repr,
        wr.input_norm,
        config.rms_norm_epsilon,
    );
    RT::lemma_rms_norm_repr_len(
        hidden_repr,
        wr.input_norm,
        config.rms_norm_epsilon,
    );
    lemma_decoder_core_output_repr_shape(
        config,
        wr,
        normed,
        hidden_repr,
        positions_repr,
        k_cache_repr,
        v_cache_repr,
        slot_repr,
        cu_q_repr,
        cu_k_repr,
        max_seqlen_q,
        max_seqlen_k,
        bt_repr,
    );
}

pub broadcast proof fn lemma_decoder_layer_output_repr_shape(
    config: DenseSwiGluForwardConfigRepr,
    wr: LayerWeightsRepr,
    hidden_repr: Tensor2D,
    residual_repr: Tensor2D,
    positions_repr: IntTensor1D,
    k_cache_repr: KVCacheLayerRepr,
    v_cache_repr: KVCacheLayerRepr,
    slot_repr: Seq<int>,
    cu_q_repr: Seq<int>,
    cu_k_repr: Seq<int>,
    max_seqlen_q: nat,
    max_seqlen_k: nat,
    bt_repr: Seq<Seq<BlockId>>,
)
    requires
        hidden_repr.len() == residual_repr.len(),
        hidden_repr.len() == positions_repr.len(),
        slot_repr.len() == hidden_repr.len(),
    ensures
        (#[trigger] decoder_layer_output_repr(
            config,
            wr,
            hidden_repr,
            residual_repr,
            positions_repr,
            k_cache_repr,
            v_cache_repr,
            slot_repr,
            cu_q_repr,
            cu_k_repr,
            max_seqlen_q,
            max_seqlen_k,
            bt_repr,
        )).0.len() == hidden_repr.len(),
        decoder_layer_output_repr(
            config,
            wr,
            hidden_repr,
            residual_repr,
            positions_repr,
            k_cache_repr,
            v_cache_repr,
            slot_repr,
            cu_q_repr,
            cu_k_repr,
            max_seqlen_q,
            max_seqlen_k,
            bt_repr,
        ).1.len() == hidden_repr.len(),
{
    let pre = RT::add_rms_norm_repr(
        hidden_repr,
        residual_repr,
        wr.input_norm,
        config.rms_norm_epsilon,
    );
    RT::lemma_add_rms_norm_repr_len(
        hidden_repr,
        residual_repr,
        wr.input_norm,
        config.rms_norm_epsilon,
    );
    lemma_decoder_core_output_repr_shape(
        config,
        wr,
        pre.0,
        pre.1,
        positions_repr,
        k_cache_repr,
        v_cache_repr,
        slot_repr,
        cu_q_repr,
        cu_k_repr,
        max_seqlen_q,
        max_seqlen_k,
        bt_repr,
    );
}

pub proof fn lemma_layer_chain_repr_shape(
    config: DenseSwiGluForwardConfigRepr,
    layers: Seq<LayerWeightsRepr>,
    hidden_repr: Tensor2D,
    residual_repr: Tensor2D,
    positions_repr: IntTensor1D,
    kv_cache_reprs: Seq<(KVCacheLayerRepr, KVCacheLayerRepr)>,
    slot_repr: Seq<int>,
    cu_q_repr: Seq<int>,
    cu_k_repr: Seq<int>,
    max_seqlen_q: nat,
    max_seqlen_k: nat,
    bt_repr: Seq<Seq<BlockId>>,
    start: nat,
)
    requires
        start <= layers.len(),
        kv_cache_reprs.len() >= layers.len(),
        hidden_repr.len() == residual_repr.len(),
        hidden_repr.len() == positions_repr.len(),
        slot_repr.len() == hidden_repr.len(),
    ensures
        layer_chain_repr(
            config,
            layers,
            hidden_repr,
            residual_repr,
            positions_repr,
            kv_cache_reprs,
            slot_repr,
            cu_q_repr,
            cu_k_repr,
            max_seqlen_q,
            max_seqlen_k,
            bt_repr,
            start,
        ).0.len() == hidden_repr.len(),
        layer_chain_repr(
            config,
            layers,
            hidden_repr,
            residual_repr,
            positions_repr,
            kv_cache_reprs,
            slot_repr,
            cu_q_repr,
            cu_k_repr,
            max_seqlen_q,
            max_seqlen_k,
            bt_repr,
            start,
        ).1.len() == hidden_repr.len(),
    decreases (layers.len() - start) as nat,
{
    if start < layers.len() {
        let next = decoder_layer_output_repr(
            config,
            layers[start as int],
            hidden_repr,
            residual_repr,
            positions_repr,
            kv_cache_reprs[start as int].0,
            kv_cache_reprs[start as int].1,
            slot_repr,
            cu_q_repr,
            cu_k_repr,
            max_seqlen_q,
            max_seqlen_k,
            bt_repr,
        );
        lemma_decoder_layer_output_repr_shape(
            config,
            layers[start as int],
            hidden_repr,
            residual_repr,
            positions_repr,
            kv_cache_reprs[start as int].0,
            kv_cache_reprs[start as int].1,
            slot_repr,
            cu_q_repr,
            cu_k_repr,
            max_seqlen_q,
            max_seqlen_k,
            bt_repr,
        );
        lemma_layer_chain_repr_shape(
            config,
            layers,
            next.0,
            next.1,
            positions_repr,
            kv_cache_reprs,
            slot_repr,
            cu_q_repr,
            cu_k_repr,
            max_seqlen_q,
            max_seqlen_k,
            bt_repr,
            (start + 1) as nat,
        );
    }
}

pub proof fn lemma_model_forward_logits_repr_shape(
    config: DenseSwiGluForwardConfigRepr,
    wr: ModelWeightsRepr,
    input_ids_repr: IntTensor1D,
    positions_repr: IntTensor1D,
    kv_cache_reprs: Seq<(KVCacheLayerRepr, KVCacheLayerRepr)>,
    slot_repr: Seq<int>,
    cu_q_repr: Seq<int>,
    cu_k_repr: Seq<int>,
    max_seqlen_q: nat,
    max_seqlen_k: nat,
    bt_repr: Seq<Seq<BlockId>>,
)
    requires
        input_ids_repr.len() == positions_repr.len(),
        slot_repr.len() == input_ids_repr.len(),
        kv_cache_reprs.len() >= wr.layers.len(),
    ensures model_forward_logits_repr(
        config,
        wr,
        input_ids_repr,
        positions_repr,
        kv_cache_reprs,
        slot_repr,
        cu_q_repr,
        cu_k_repr,
        max_seqlen_q,
        max_seqlen_k,
        bt_repr,
    ).len() == input_ids_repr.len(),
{
    broadcast use {
        RT::lemma_embed_repr_shape,
        RT::lemma_rms_norm_repr_shape,
        RT::lemma_add_rms_norm_repr_shape,
        RT::lemma_linear_repr_shape,
    };
    let embed = RT::embed_repr(input_ids_repr, wr.embed_weight);
    if wr.layers.len() > 0 {
        let first = first_decoder_layer_output_repr(
            config,
            wr.layers[0],
            embed,
            positions_repr,
            kv_cache_reprs[0].0,
            kv_cache_reprs[0].1,
            slot_repr,
            cu_q_repr,
            cu_k_repr,
            max_seqlen_q,
            max_seqlen_k,
            bt_repr,
        );
        lemma_first_decoder_layer_output_repr_shape(
            config,
            wr.layers[0],
            embed,
            positions_repr,
            kv_cache_reprs[0].0,
            kv_cache_reprs[0].1,
            slot_repr,
            cu_q_repr,
            cu_k_repr,
            max_seqlen_q,
            max_seqlen_k,
            bt_repr,
        );
        lemma_layer_chain_repr_shape(
            config,
            wr.layers,
            first.0,
            first.1,
            positions_repr,
            kv_cache_reprs,
            slot_repr,
            cu_q_repr,
            cu_k_repr,
            max_seqlen_q,
            max_seqlen_k,
            bt_repr,
            1,
        );
    }
}

pub proof fn lemma_synthetic_cache_slot_in_cache(
    context_len: nat,
    num_layers: nat,
    ell: int,
    j: nat,
)
    requires
        0 <= ell < num_layers as int,
        j < context_len,
    ensures
        crate::proof::tensor::geometry::slot_in_cache(
            synthetic_cache_reprs(context_len, num_layers)[ell].0,
            j,
        ),
        crate::proof::tensor::geometry::slot_in_cache(
            synthetic_cache_reprs(context_len, num_layers)[ell].1,
            j,
        ),
{
    crate::proof::tensor::geometry::lemma_blocks_needed_covers_pos(j, context_len);
}

} // verus!
