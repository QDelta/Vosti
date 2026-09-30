//! Family-neutral post-store KV-cache semantics for dense SwiGLU decoders.
//!
// Layer 3, Step A: chain-level post-store KV-cache reprs.
//
// `model_forward` runs the decoder layers in sequence; each layer's
// paged-attention writes that layer's K/V into the cache (exposed at the leaf by
// `decoder_core_forward`'s `ensures` as `DS::layer_kv_update_repr`). But the
// `model_forward` loop invariant only tracks *unprocessed* layers as unchanged
// and discards the *processed* layers' post-store reprs, so `model_forward`'s
// `ensures` constrains only the logits.
//
// These spec functions characterize the FULL post-store per-layer cache after
// the chain runs — `layer_chain_kv_reprs` mirrors `DS::layer_chain_repr`'s fold,
// but instead of only advancing `(hidden, residual)` it also records each layer's
// `layer_kv_update_repr` (the store over that layer's chain-folded normed input).
// A later step wires these into `model_forward`'s loop invariant + `ensures`.

use crate::proof::model::dense_swiglu::semantics as DS;
use crate::{types::*, proof::model::types::*, proof::tensor::types::*};
use crate::boundary::tensor_runtime as RT;
use vstd::prelude::*;

verus! {

// Per-layer post-store cache for a SUBSEQUENT layer (input via `add_rms_norm`,
// matching `decoder_layer_forward` / `decoder_layer_output_repr`).
pub open spec fn decoder_layer_kv_update_repr(
    config: DenseSwiGluForwardConfigRepr,
    wr: LayerWeightsRepr,
    hidden_repr: Tensor2D,
    residual_repr: Tensor2D,
    positions_repr: IntTensor1D,
    k_cache_repr: KVCacheLayerRepr,
    v_cache_repr: KVCacheLayerRepr,
    slot_repr: Seq<int>,
) -> (KVCacheLayerRepr, KVCacheLayerRepr) {
    let normed = RT::add_rms_norm_repr(
        hidden_repr, residual_repr, wr.input_norm, config.rms_norm_epsilon,
    ).0;
    DS::layer_kv_update_repr(
        config, wr, normed, positions_repr, k_cache_repr, v_cache_repr, slot_repr,
    )
}

// Per-layer post-store cache for the FIRST layer (input via `rms_norm`, matching
// `first_decoder_layer_forward` / `first_decoder_layer_output_repr`).
pub open spec fn first_decoder_layer_kv_update_repr(
    config: DenseSwiGluForwardConfigRepr,
    wr: LayerWeightsRepr,
    hidden_repr: Tensor2D,
    positions_repr: IntTensor1D,
    k_cache_repr: KVCacheLayerRepr,
    v_cache_repr: KVCacheLayerRepr,
    slot_repr: Seq<int>,
) -> (KVCacheLayerRepr, KVCacheLayerRepr) {
    let normed = RT::rms_norm_repr(
        hidden_repr, wr.input_norm, config.rms_norm_epsilon,
    );
    DS::layer_kv_update_repr(
        config, wr, normed, positions_repr, k_cache_repr, v_cache_repr, slot_repr,
    )
}

// Post-store cache reprs after running SUBSEQUENT layers `[start, len)` of the
// chain, given the `(hidden, residual)` entering layer `start`.  Mirrors
// `DS::layer_chain_repr`'s fold: layer `start` updates `kv[start]` with its
// `decoder_layer_kv_update_repr`, then the hidden/residual advance via
// `DS::decoder_layer_output_repr` and the recursion continues. Layers outside
// `[start, len)` keep their incoming repr.
pub open spec fn layer_chain_kv_reprs(
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
) -> Seq<(KVCacheLayerRepr, KVCacheLayerRepr)>
    recommends
        start <= layers.len(),
        kv_cache_reprs.len() >= layers.len(),
    decreases (layers.len() - start) as nat,
{
    if start >= layers.len() {
        kv_cache_reprs
    } else {
        let updated = decoder_layer_kv_update_repr(config, layers[start as int],
            hidden_repr, residual_repr, positions_repr,
            kv_cache_reprs[start as int].0, kv_cache_reprs[start as int].1, slot_repr);
        let kv2 = kv_cache_reprs.update(start as int, updated);
        let next = DS::decoder_layer_output_repr(config, layers[start as int],
            hidden_repr, residual_repr, positions_repr,
            kv_cache_reprs[start as int].0, kv_cache_reprs[start as int].1, slot_repr,
            cu_q_repr, cu_k_repr, max_seqlen_q, max_seqlen_k, bt_repr);
        layer_chain_kv_reprs(config, layers, next.0, next.1, positions_repr,
            kv2, slot_repr, cu_q_repr, cu_k_repr, max_seqlen_q, max_seqlen_k, bt_repr,
            (start + 1) as nat)
    }
}

// The FULL post-store per-layer cache after `model_forward` runs: the first
// layer (rms_norm input) updates index 0, then the subsequent-layer chain
// (`layer_chain_kv_reprs` from layer 1) updates the rest.  Mirrors
// `model_forward`'s structure (embed → first layer → loop).  `num_layers == 0`
// leaves the cache unchanged.
pub open spec fn model_forward_kv_reprs(
    config: DenseSwiGluForwardConfigRepr,
    wr: ModelWeightsRepr,
    input_ids_repr: IntTensor1D,
    positions_repr: IntTensor1D,
    pre_kv_reprs: Seq<(KVCacheLayerRepr, KVCacheLayerRepr)>,
    slot_repr: Seq<int>,
    cu_q_repr: Seq<int>,
    cu_k_repr: Seq<int>,
    max_seqlen_q: nat,
    max_seqlen_k: nat,
    bt_repr: Seq<Seq<BlockId>>,
) -> Seq<(KVCacheLayerRepr, KVCacheLayerRepr)> {
    let layers = wr.layers;
    if layers.len() == 0 {
        pre_kv_reprs
    } else {
        let embed = RT::embed_repr(input_ids_repr, wr.embed_weight);
        let first_update = first_decoder_layer_kv_update_repr(config, layers[0],
            embed, positions_repr, pre_kv_reprs[0].0, pre_kv_reprs[0].1, slot_repr);
        let kv1 = pre_kv_reprs.update(0, first_update);
        let first_out = DS::first_decoder_layer_output_repr(config, layers[0],
            embed, positions_repr, pre_kv_reprs[0].0, pre_kv_reprs[0].1, slot_repr,
            cu_q_repr, cu_k_repr, max_seqlen_q, max_seqlen_k, bt_repr);
        layer_chain_kv_reprs(config, layers, first_out.0, first_out.1, positions_repr,
            kv1, slot_repr, cu_q_repr, cu_k_repr, max_seqlen_q, max_seqlen_k, bt_repr, 1)
    }
}

// The chain fold preserves the cache-seq length (each step is an `update`).
pub proof fn lemma_layer_chain_kv_reprs_len(
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
    requires start <= layers.len(), kv_cache_reprs.len() >= layers.len(),
    ensures
        layer_chain_kv_reprs(config, layers, hidden_repr, residual_repr, positions_repr,
            kv_cache_reprs, slot_repr, cu_q_repr, cu_k_repr, max_seqlen_q, max_seqlen_k,
            bt_repr, start).len() == kv_cache_reprs.len(),
    decreases (layers.len() - start) as nat,
{
    if start >= layers.len() {
    } else {
        let updated = decoder_layer_kv_update_repr(config, layers[start as int],
            hidden_repr, residual_repr, positions_repr,
            kv_cache_reprs[start as int].0, kv_cache_reprs[start as int].1, slot_repr);
        let kv2 = kv_cache_reprs.update(start as int, updated);
        let next = DS::decoder_layer_output_repr(config, layers[start as int],
            hidden_repr, residual_repr, positions_repr,
            kv_cache_reprs[start as int].0, kv_cache_reprs[start as int].1, slot_repr,
            cu_q_repr, cu_k_repr, max_seqlen_q, max_seqlen_k, bt_repr);
        lemma_layer_chain_kv_reprs_len(config, layers, next.0, next.1, positions_repr,
            kv2, slot_repr, cu_q_repr, cu_k_repr, max_seqlen_q, max_seqlen_k, bt_repr,
            (start + 1) as nat);
    }
}

// The fold starting at `start` only updates indices `start..layers.len()`.
// Earlier layer caches are framing state and remain unchanged.  Besides being
// useful for cache fidelity, this makes explicit a property that was previously
// only implicit in the recursive `Seq::update` definition.
pub proof fn lemma_layer_chain_kv_reprs_preserves_before_start(
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
    layer: int,
)
    requires
        start <= layers.len(),
        kv_cache_reprs.len() >= layers.len(),
        0 <= layer < start,
    ensures
        layer_chain_kv_reprs(config, layers, hidden_repr, residual_repr, positions_repr,
            kv_cache_reprs, slot_repr, cu_q_repr, cu_k_repr, max_seqlen_q,
            max_seqlen_k, bt_repr, start)[layer] == kv_cache_reprs[layer],
    decreases (layers.len() - start) as nat,
{
    if start >= layers.len() {
    } else {
        let updated = decoder_layer_kv_update_repr(config, layers[start as int],
            hidden_repr, residual_repr, positions_repr,
            kv_cache_reprs[start as int].0, kv_cache_reprs[start as int].1, slot_repr);
        let kv2 = kv_cache_reprs.update(start as int, updated);
        let next = DS::decoder_layer_output_repr(config, layers[start as int],
            hidden_repr, residual_repr, positions_repr,
            kv_cache_reprs[start as int].0, kv_cache_reprs[start as int].1, slot_repr,
            cu_q_repr, cu_k_repr, max_seqlen_q, max_seqlen_k, bt_repr);
        assert(kv2[layer] == kv_cache_reprs[layer]);
        lemma_layer_chain_kv_reprs_preserves_before_start(config,
            layers, next.0, next.1, positions_repr, kv2, slot_repr,
            cu_q_repr, cu_k_repr, max_seqlen_q, max_seqlen_k, bt_repr,
            (start + 1) as nat, layer);
    }
}

// In particular, the model's final cache at layer 0 is exactly the first
// layer's store; later decoder-layer iterations cannot overwrite it.
pub proof fn lemma_model_forward_kv_reprs_layer_zero(
    config: DenseSwiGluForwardConfigRepr,
    wr: ModelWeightsRepr,
    input_ids_repr: IntTensor1D,
    positions_repr: IntTensor1D,
    pre_kv_reprs: Seq<(KVCacheLayerRepr, KVCacheLayerRepr)>,
    slot_repr: Seq<int>,
    cu_q_repr: Seq<int>,
    cu_k_repr: Seq<int>,
    max_seqlen_q: nat,
    max_seqlen_k: nat,
    bt_repr: Seq<Seq<BlockId>>,
)
    requires
        wr.layers.len() > 0,
        pre_kv_reprs.len() >= wr.layers.len(),
    ensures
        model_forward_kv_reprs(config, wr, input_ids_repr, positions_repr, pre_kv_reprs,
            slot_repr, cu_q_repr, cu_k_repr, max_seqlen_q, max_seqlen_k, bt_repr)[0]
        == first_decoder_layer_kv_update_repr(config, wr.layers[0],
            RT::embed_repr(input_ids_repr, wr.embed_weight), positions_repr,
            pre_kv_reprs[0].0, pre_kv_reprs[0].1, slot_repr),
{
    let embed = RT::embed_repr(input_ids_repr, wr.embed_weight);
    let first_update = first_decoder_layer_kv_update_repr(config, wr.layers[0],
        embed, positions_repr, pre_kv_reprs[0].0, pre_kv_reprs[0].1, slot_repr);
    let kv1 = pre_kv_reprs.update(0, first_update);
    let first_out = DS::first_decoder_layer_output_repr(config, wr.layers[0],
        embed, positions_repr, pre_kv_reprs[0].0, pre_kv_reprs[0].1, slot_repr,
        cu_q_repr, cu_k_repr, max_seqlen_q, max_seqlen_k, bt_repr);
    lemma_layer_chain_kv_reprs_preserves_before_start(config,
        wr.layers, first_out.0, first_out.1, positions_repr, kv1, slot_repr,
        cu_q_repr, cu_k_repr, max_seqlen_q, max_seqlen_k, bt_repr, 1, 0);
    assert(kv1[0] == first_update);
}


// ---------------------------------------------------------------------------
// Empty-plan identity (2026-08-06): a step over ZERO input tokens leaves every
// layer's cache untouched.  Needed by the verified `engine.step` for its
// empty-schedule early return (the plan's shape facts give input/positions/
// slots all empty there).  Pure shape reasoning: the stored row block is
// empty, and storing zero rows is the identity.
// ---------------------------------------------------------------------------

// Length preservation at the model level (wrapper over the chain lemma).
pub proof fn lemma_model_forward_kv_reprs_len(
    config: DenseSwiGluForwardConfigRepr,
    wr: ModelWeightsRepr,
    input_ids_repr: IntTensor1D,
    positions_repr: IntTensor1D,
    pre_kv_reprs: Seq<(KVCacheLayerRepr, KVCacheLayerRepr)>,
    slot_repr: Seq<int>,
    cu_q_repr: Seq<int>,
    cu_k_repr: Seq<int>,
    max_seqlen_q: nat,
    max_seqlen_k: nat,
    bt_repr: Seq<Seq<BlockId>>,
)
    requires pre_kv_reprs.len() >= wr.layers.len(),
    ensures
        model_forward_kv_reprs(config, wr, input_ids_repr, positions_repr, pre_kv_reprs,
            slot_repr, cu_q_repr, cu_k_repr, max_seqlen_q, max_seqlen_k, bt_repr)
            .len() == pre_kv_reprs.len(),
{
    if wr.layers.len() == 0 {
    } else {
        let embed = RT::embed_repr(input_ids_repr, wr.embed_weight);
        let first_update = first_decoder_layer_kv_update_repr(config, wr.layers[0],
            embed, positions_repr, pre_kv_reprs[0].0, pre_kv_reprs[0].1, slot_repr);
        let kv1 = pre_kv_reprs.update(0, first_update);
        let first_out = DS::first_decoder_layer_output_repr(config, wr.layers[0],
            embed, positions_repr, pre_kv_reprs[0].0, pre_kv_reprs[0].1, slot_repr,
            cu_q_repr, cu_k_repr, max_seqlen_q, max_seqlen_k, bt_repr);
        lemma_layer_chain_kv_reprs_len(config, wr.layers, first_out.0, first_out.1,
            positions_repr, kv1, slot_repr, cu_q_repr, cu_k_repr,
            max_seqlen_q, max_seqlen_k, bt_repr, 1);
    }
}

pub proof fn lemma_layer_chain_kv_reprs_empty(
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
        hidden_repr.len() == 0,
        residual_repr.len() == 0,
        positions_repr.len() == 0,
        slot_repr.len() == 0,
    ensures
        layer_chain_kv_reprs(config, layers, hidden_repr, residual_repr, positions_repr,
            kv_cache_reprs, slot_repr, cu_q_repr, cu_k_repr,
            max_seqlen_q, max_seqlen_k, bt_repr, start) == kv_cache_reprs,
    decreases layers.len() - start,
{
    if start >= layers.len() {
    } else {
        broadcast use RT::lemma_add_rms_norm_repr_shape;
        let wr = layers[start as int];
        let kv = kv_cache_reprs[start as int];
        let normed = RT::add_rms_norm_repr(
            hidden_repr, residual_repr, wr.input_norm, config.rms_norm_epsilon,
        ).0;
        RT::lemma_add_rms_norm_repr_shape(
            hidden_repr, residual_repr, wr.input_norm, config.rms_norm_epsilon,
        );
        assert(normed.len() == 0);
        let pre = DS::pre_attention_repr(config, wr, normed, positions_repr);
        DS::lemma_pre_attention_repr_shape(config, wr, normed, positions_repr);
        assert(pre.1.len() == 0);
        RT::lemma_store_kv_cache_repr_empty_rows(
            pre.1,
            RT::view_as_kv_repr(RT::qkv_linear_repr(
                normed, wr.q_proj, wr.k_proj, wr.v_proj,
            ).2),
            kv.0, kv.1, slot_repr);
        assert(decoder_layer_kv_update_repr(config, wr, hidden_repr, residual_repr,
            positions_repr, kv.0, kv.1, slot_repr) == kv);
        let kv2 = kv_cache_reprs.update(start as int,
            decoder_layer_kv_update_repr(config, wr, hidden_repr, residual_repr,
                positions_repr, kv.0, kv.1, slot_repr));
        assert(kv2 =~= kv_cache_reprs);
        let next = DS::decoder_layer_output_repr(config, wr, hidden_repr, residual_repr,
            positions_repr, kv.0, kv.1, slot_repr,
            cu_q_repr, cu_k_repr, max_seqlen_q, max_seqlen_k, bt_repr);
        DS::lemma_decoder_layer_output_repr_shape(
            config, wr, hidden_repr, residual_repr, positions_repr,
            kv.0, kv.1, slot_repr, cu_q_repr, cu_k_repr,
            max_seqlen_q, max_seqlen_k, bt_repr,
        );
        assert(next.0.len() == 0 && next.1.len() == 0);
        lemma_layer_chain_kv_reprs_empty(config, layers, next.0, next.1, positions_repr,
            kv2, slot_repr, cu_q_repr, cu_k_repr,
            max_seqlen_q, max_seqlen_k, bt_repr, (start + 1) as nat);
    }
}

pub proof fn lemma_model_forward_kv_reprs_empty(
    config: DenseSwiGluForwardConfigRepr,
    wr: ModelWeightsRepr,
    input_ids_repr: IntTensor1D,
    positions_repr: IntTensor1D,
    pre_kv_reprs: Seq<(KVCacheLayerRepr, KVCacheLayerRepr)>,
    slot_repr: Seq<int>,
    cu_q_repr: Seq<int>,
    cu_k_repr: Seq<int>,
    max_seqlen_q: nat,
    max_seqlen_k: nat,
    bt_repr: Seq<Seq<BlockId>>,
)
    requires
        pre_kv_reprs.len() >= wr.layers.len(),
        input_ids_repr.len() == 0,
        positions_repr.len() == 0,
        slot_repr.len() == 0,
    ensures
        model_forward_kv_reprs(config, wr, input_ids_repr, positions_repr, pre_kv_reprs,
            slot_repr, cu_q_repr, cu_k_repr, max_seqlen_q, max_seqlen_k, bt_repr)
            == pre_kv_reprs,
{
    if wr.layers.len() == 0 {
    } else {
        broadcast use {
            RT::lemma_embed_repr_shape,
            RT::lemma_rms_norm_repr_shape,
        };
        let embed = RT::embed_repr(input_ids_repr, wr.embed_weight);
        assert(embed.len() == 0);
        let wr0 = wr.layers[0];
        let normed = RT::rms_norm_repr(
            embed, wr0.input_norm, config.rms_norm_epsilon,
        );
        RT::lemma_rms_norm_repr_shape(
            embed, wr0.input_norm, config.rms_norm_epsilon,
        );
        assert(normed.len() == 0);
        let pre = DS::pre_attention_repr(config, wr0, normed, positions_repr);
        DS::lemma_pre_attention_repr_shape(config, wr0, normed, positions_repr);
        assert(pre.1.len() == 0);
        RT::lemma_store_kv_cache_repr_empty_rows(
            pre.1,
            RT::view_as_kv_repr(RT::qkv_linear_repr(
                normed, wr0.q_proj, wr0.k_proj, wr0.v_proj,
            ).2),
            pre_kv_reprs[0].0, pre_kv_reprs[0].1, slot_repr);
        assert(first_decoder_layer_kv_update_repr(config, wr0, embed, positions_repr,
            pre_kv_reprs[0].0, pre_kv_reprs[0].1, slot_repr) == pre_kv_reprs[0]);
        let kv1 = pre_kv_reprs.update(0,
            first_decoder_layer_kv_update_repr(config, wr0, embed, positions_repr,
                pre_kv_reprs[0].0, pre_kv_reprs[0].1, slot_repr));
        assert(kv1 =~= pre_kv_reprs);
        let first_out = DS::first_decoder_layer_output_repr(config, wr0,
            embed, positions_repr, pre_kv_reprs[0].0, pre_kv_reprs[0].1, slot_repr,
            cu_q_repr, cu_k_repr, max_seqlen_q, max_seqlen_k, bt_repr);
        DS::lemma_first_decoder_layer_output_repr_shape(
            config, wr0, embed, positions_repr,
            pre_kv_reprs[0].0, pre_kv_reprs[0].1, slot_repr,
            cu_q_repr, cu_k_repr, max_seqlen_q, max_seqlen_k, bt_repr,
        );
        assert(first_out.0.len() == 0 && first_out.1.len() == 0);
        lemma_layer_chain_kv_reprs_empty(config, wr.layers, first_out.0, first_out.1,
            positions_repr, kv1, slot_repr, cu_q_repr, cu_k_repr,
            max_seqlen_q, max_seqlen_k, bt_repr, 1);
    }
}

} // verus!
