//! Shared four-norm gated decoder semantics and structural fold lemmas.
//!
//! Model policies are immutable inputs; per-layer geometry lives in common
//! layer weights. Leaf operations remain opaque relational kernel operators.
//! This module does not admit a model profile or import attention certificates.

use crate::model_config::AttentionKind;
#[cfg(verus_only)]
use crate::boundary::dense_layer_primitives as LAYERS;
#[cfg(verus_only)]
use crate::proof::model::four_norm_gated::layers as FOUR_NORM;
use crate::{types::*, proof::model::types::*, proof::tensor::types::*};
#[cfg(verus_only)]
use crate::boundary::tensor_runtime as RT;
use vstd::prelude::*;

verus! {

// ---------------------------------------------------------------------------
// Shared primitive composition.
// ---------------------------------------------------------------------------

#[verifier::opaque]
pub open spec fn scaled_embed_repr(
    input_ids: IntTensor1D,
    weight: Tensor2D,
    hidden_size: nat,
) -> Tensor2D {
    Seq::new(input_ids.len(), |i: int|
        Seq::new(hidden_size, |col: int|
            RT::scaled_embed_kernel_cell_repr(
                input_ids[i], weight, hidden_size, col,
            )))
}

pub broadcast proof fn lemma_scaled_embed_repr_shape(
    input_ids: IntTensor1D,
    weight: Tensor2D,
    hidden_size: nat,
)
    ensures
        #[trigger] scaled_embed_repr(input_ids, weight, hidden_size).len()
            == input_ids.len(),
{
    reveal(scaled_embed_repr);
}

// Both variants receive the complete causal cache.  SlidingWindow changes the
// attention semantic value, but this signature intentionally does not assert
// that old cache entries outside the window are irrelevant.
pub open spec fn paged_attention_repr(
    q: Tensor2D,
    k_cache: KVCacheLayerRepr,
    v_cache: KVCacheLayerRepr,
    cu_q: Seq<int>,
    cu_k: Seq<int>,
    max_q: nat,
    max_k: nat,
    block_table: Seq<Seq<BlockId>>,
    attention: AttentionConfigRepr,
    parameters: AttentionParametersRepr,
) -> Tensor2D {
    match attention {
        AttentionConfigRepr::Full => LAYERS::full_paged_attention_repr(
            q, k_cache, v_cache, cu_q, cu_k, max_q, max_k, block_table,
            parameters,
        ),
        AttentionConfigRepr::SlidingWindow(attention_window) =>
            LAYERS::sliding_window_paged_attention_repr(
                q, k_cache, v_cache, cu_q, cu_k, max_q, max_k,
                block_table, attention_window,
                parameters,
            ),
    }
}

pub open spec fn merge_attention_heads_repr(input: Tensor2D) -> Tensor2D {
    LAYERS::merge_attention_heads_repr(input)
}

#[verifier::opaque]
pub open spec fn gelu_tanh_mul_repr(
    gate: Tensor2D,
    up: Tensor2D,
) -> Tensor2D {
    Seq::new(gate.len(), |i: int| {
        let up_row = if i < up.len() { up[i] } else { Seq::empty() };
        Seq::new(gate[i].len(), |col: int|
            RT::gelu_tanh_mul_kernel_cell_repr(gate[i], up_row, col))
    })
}

#[verifier::opaque]
pub open spec fn add_repr(left: Tensor2D, right: Tensor2D) -> Tensor2D {
    Seq::new(left.len(), |i: int| {
        let right_row = if i < right.len() { right[i] } else { Seq::empty() };
        Seq::new(left[i].len(), |col: int|
            RT::add_kernel_cell_repr(left[i], right_row, col))
    })
}

pub broadcast proof fn lemma_gelu_tanh_mul_repr_shape(
    gate: Tensor2D,
    up: Tensor2D,
)
    ensures #[trigger] gelu_tanh_mul_repr(gate, up).len() == gate.len(),
{
    reveal(gelu_tanh_mul_repr);
}

pub broadcast proof fn lemma_add_repr_shape(
    left: Tensor2D,
    right: Tensor2D,
)
    ensures #[trigger] add_repr(left, right).len() == left.len(),
{
    reveal(add_repr);
}

// Assemble the immutable row-local policy for one layer.
pub open spec fn four_norm_layer_repr(
    common: LayerWeightsRepr, extension: FourNormGatedLayerExtensionRepr,
) -> FourNormGatedLayerRepr {
    FourNormGatedLayerRepr {
        common,
        pre_feedforward_norm: extension.pre_feedforward_norm,
        post_feedforward_norm: extension.post_feedforward_norm,
        norm: extension.row_parameters.norm,
        qk_norm: extension.row_parameters.qk_norm,
        rotary: extension.row_parameters.rotary,
        value_norm_epsilon: extension.row_parameters.value_norm_epsilon,
        layer_scale: extension.row_parameters.layer_scale,
    }
}

pub open spec fn attention_pre_store_repr(
    common: LayerWeightsRepr, extension: FourNormGatedLayerExtensionRepr,
    hidden: Tensor2D, positions: IntTensor1D,
) -> (Tensor2D, Tensor2D, Tensor2D)
    recommends hidden.len() == positions.len(),
{
    FOUR_NORM::attention_pre_store_repr(four_norm_layer_repr(common, extension), hidden, positions)
}

pub open spec fn post_attention_and_mlp_repr(
    common: LayerWeightsRepr, extension: FourNormGatedLayerExtensionRepr,
    residual: Tensor2D, attended: Tensor2D,
) -> Tensor2D
    recommends residual.len() == attended.len(),
{
    FOUR_NORM::post_attention_and_mlp_repr(four_norm_layer_repr(common, extension), residual, attended)
}

pub broadcast proof fn lemma_attention_pre_store_repr_shape(
    common: LayerWeightsRepr,
    extension: FourNormGatedLayerExtensionRepr,
    hidden: Tensor2D,
    positions: IntTensor1D,
)
    requires hidden.len() == positions.len(),
    ensures
        #[trigger] attention_pre_store_repr(
            common, extension, hidden, positions,
        ).0.len() == hidden.len(),
        attention_pre_store_repr(
            common, extension, hidden, positions,
        ).1.len() == hidden.len(),
        attention_pre_store_repr(
            common, extension, hidden, positions,
        ).2.len() == hidden.len(),
{
    FOUR_NORM::lemma_attention_pre_store_repr_shape(
        four_norm_layer_repr(common, extension), hidden, positions,
    );
}

pub broadcast proof fn lemma_paged_attention_repr_shape(
    q: Tensor2D,
    k_cache: KVCacheLayerRepr,
    v_cache: KVCacheLayerRepr,
    cu_q: Seq<int>,
    cu_k: Seq<int>,
    max_q: nat,
    max_k: nat,
    block_table: Seq<Seq<BlockId>>,
    attention: AttentionConfigRepr,
    parameters: AttentionParametersRepr,
)
    ensures
        #[trigger] paged_attention_repr(
            q, k_cache, v_cache, cu_q, cu_k, max_q, max_k,
            block_table, attention, parameters,
        ).len() == q.len(),
{

    reveal(paged_attention_repr);
    reveal(LAYERS::full_paged_attention_repr);
    reveal(LAYERS::sliding_window_paged_attention_repr);
    match attention {
        AttentionConfigRepr::Full => {
            crate::boundary::attention_operator::lemma_shape(q, k_cache, v_cache,
                cu_q, cu_k, block_table, AttentionKind::Full, parameters, 0);
        },
        AttentionConfigRepr::SlidingWindow(window) => {
            crate::boundary::attention_operator::lemma_shape(q, k_cache, v_cache,
                cu_q, cu_k, block_table, AttentionKind::SlidingWindow, parameters, window);
        },
    }
}

// ---------------------------------------------------------------------------
// Exact decoder composition.
// ---------------------------------------------------------------------------

// One four-norm block consumes and returns a single hidden tensor.  Its cache pair
// is the post-RoPE K and projected V scattered into the full incoming cache.
pub open spec fn decoder_layer_step_repr(
    common: LayerWeightsRepr,
    extension: FourNormGatedLayerExtensionRepr,
    hidden: Tensor2D,
    positions: IntTensor1D,
    k_cache: KVCacheLayerRepr,
    v_cache: KVCacheLayerRepr,
    slots: Seq<int>,
    cu_q: Seq<int>,
    cu_k: Seq<int>,
    max_q: nat,
    max_k: nat,
    block_table: Seq<Seq<BlockId>>,
) -> (Tensor2D, (KVCacheLayerRepr, KVCacheLayerRepr))
    recommends
        hidden.len() == positions.len(),
        hidden.len() == slots.len(),
{
    let attention_residual = hidden;
    let pre = attention_pre_store_repr(common, extension, hidden, positions);
    let cache = RT::store_kv_cache_repr(
        pre.1, pre.2, k_cache, v_cache, slots,
    );
    let attended = paged_attention_repr(
        pre.0, cache.0, cache.1, cu_q, cu_k, max_q, max_k,
        block_table, extension.attention, crate::boundary::dense_layer_primitives::layer_attention_parameters_repr(common, extension.attention_scale),
    );
    (
        post_attention_and_mlp_repr(
            common, extension, attention_residual, attended,
        ),
        cache,
    )
}

pub open spec fn layer_chain_repr(
    common_layers: Seq<LayerWeightsRepr>,
    extension_layers: Seq<FourNormGatedLayerExtensionRepr>,
    hidden: Tensor2D,
    positions: IntTensor1D,
    caches: Seq<(KVCacheLayerRepr, KVCacheLayerRepr)>,
    slots: Seq<int>,
    cu_q: Seq<int>,
    cu_k: Seq<int>,
    max_q: nat,
    max_k: nat,
    block_table: Seq<Seq<BlockId>>,
    start: nat,
) -> (Tensor2D, Seq<(KVCacheLayerRepr, KVCacheLayerRepr)>)
    recommends
        start <= common_layers.len(),
        extension_layers.len() == common_layers.len(),
        caches.len() >= common_layers.len(),
        hidden.len() == positions.len(),
        hidden.len() == slots.len(),
    decreases (common_layers.len() - start) as nat,
{
    if start >= common_layers.len() {
        (hidden, caches)
    } else {
        let layer = decoder_layer_step_repr(
            common_layers[start as int], extension_layers[start as int], hidden,
            positions, caches[start as int].0, caches[start as int].1, slots,
            cu_q, cu_k, max_q, max_k, block_table,
        );
        layer_chain_repr(
            common_layers, extension_layers, layer.0, positions,
            caches.update(start as int, layer.1), slots, cu_q, cu_k,
            max_q, max_k, block_table, (start + 1) as nat,
        )
    }
}

pub open spec fn model_forward_hidden_and_kv_reprs(
    wr: ModelWeightsRepr,
    config: FourNormGatedDecoderConfigRepr,
    input_ids: IntTensor1D,
    positions: IntTensor1D,
    pre_kv: Seq<(KVCacheLayerRepr, KVCacheLayerRepr)>,
    slots: Seq<int>,
    cu_q: Seq<int>,
    cu_k: Seq<int>,
    max_q: nat,
    max_k: nat,
    block_table: Seq<Seq<BlockId>>,
) -> (Tensor2D, Seq<(KVCacheLayerRepr, KVCacheLayerRepr)>)
    recommends
        config.layers.len() == wr.layers.len(),
        pre_kv.len() >= wr.layers.len(),
        input_ids.len() == positions.len(),
        input_ids.len() == slots.len(),
{
    layer_chain_repr(
        wr.layers, config.layers,
        scaled_embed_repr(
            input_ids, wr.embed_weight, config.geometry.hidden_size,
        ),
        positions, pre_kv, slots, cu_q, cu_k, max_q, max_k, block_table, 0,
    )
}

// This is the full query-row logits semantics.  The executable forward may
// select each sequence's final query row before the tied LM-head matmul; the
// generic engine contract observes the equivalent row-local projection.
pub open spec fn model_forward_logits_repr(
    wr: ModelWeightsRepr,
    config: FourNormGatedDecoderConfigRepr,
    input_ids: IntTensor1D,
    positions: IntTensor1D,
    pre_kv: Seq<(KVCacheLayerRepr, KVCacheLayerRepr)>,
    slots: Seq<int>,
    cu_q: Seq<int>,
    cu_k: Seq<int>,
    max_q: nat,
    max_k: nat,
    block_table: Seq<Seq<BlockId>>,
) -> Tensor2D {
    let out = model_forward_hidden_and_kv_reprs(
        wr, config, input_ids, positions, pre_kv, slots,
        cu_q, cu_k, max_q, max_k, block_table,
    );
    FOUR_NORM::final_logits_repr(out.0, wr.final_norm, wr.lm_head,
        config.final_norm_policy, config.final_logit_softcap)
}

pub open spec fn model_forward_kv_reprs(
    wr: ModelWeightsRepr,
    config: FourNormGatedDecoderConfigRepr,
    input_ids: IntTensor1D,
    positions: IntTensor1D,
    pre_kv: Seq<(KVCacheLayerRepr, KVCacheLayerRepr)>,
    slots: Seq<int>,
    cu_q: Seq<int>,
    cu_k: Seq<int>,
    max_q: nat,
    max_k: nat,
    block_table: Seq<Seq<BlockId>>,
) -> Seq<(KVCacheLayerRepr, KVCacheLayerRepr)> {
    model_forward_hidden_and_kv_reprs(
        wr, config, input_ids, positions, pre_kv, slots,
        cu_q, cu_k, max_q, max_k, block_table,
    ).1
}

// ---------------------------------------------------------------------------
// Architecture-level shape facts.  These follow from row-local allocations
// and Seq::update; no numerical kernel property is assumed.
// ---------------------------------------------------------------------------

pub open spec fn cache_sequence_page_shape(
    caches: Seq<(KVCacheLayerRepr, KVCacheLayerRepr)>,
    num_pages: nat,
) -> bool {
    forall|j: int| 0 <= j < caches.len() ==> {
        &&& (#[trigger] caches[j]).0.len() == num_pages
        &&& caches[j].1.len() == num_pages
        &&& (forall|p: int| 0 <= p < caches[j].0.len() ==>
            (#[trigger] caches[j].0[p]).len()
                == crate::types::BLOCK_SIZE_SPEC as int)
        &&& (forall|p: int| 0 <= p < caches[j].1.len() ==>
            (#[trigger] caches[j].1[p]).len()
                == crate::types::BLOCK_SIZE_SPEC as int)
    }
}

pub proof fn lemma_decoder_layer_step_repr_shape(
    common: LayerWeightsRepr,
    extension: FourNormGatedLayerExtensionRepr,
    hidden: Tensor2D,
    positions: IntTensor1D,
    k_cache: KVCacheLayerRepr,
    v_cache: KVCacheLayerRepr,
    slots: Seq<int>,
    cu_q: Seq<int>,
    cu_k: Seq<int>,
    max_q: nat,
    max_k: nat,
    block_table: Seq<Seq<BlockId>>,
)
    requires hidden.len() == positions.len(), hidden.len() == slots.len(),
    ensures decoder_layer_step_repr(
        common, extension, hidden, positions, k_cache, v_cache, slots,
        cu_q, cu_k, max_q, max_k, block_table,
    ).0.len() == hidden.len(),
{
    broadcast use RT::lemma_linear_repr_shape;
    broadcast use RT::lemma_view_as_kv_repr_shape;
    reveal(scaled_embed_repr);
    reveal(paged_attention_repr);
    reveal(merge_attention_heads_repr);
    reveal(LAYERS::merge_attention_heads_repr);
    reveal(gelu_tanh_mul_repr);
    reveal(add_repr);
}

pub proof fn lemma_decoder_layer_step_repr_empty(
    common: LayerWeightsRepr,
    extension: FourNormGatedLayerExtensionRepr,
    hidden: Tensor2D,
    positions: IntTensor1D,
    k_cache: KVCacheLayerRepr,
    v_cache: KVCacheLayerRepr,
    slots: Seq<int>,
    cu_q: Seq<int>,
    cu_k: Seq<int>,
    max_q: nat,
    max_k: nat,
    block_table: Seq<Seq<BlockId>>,
)
    requires
        hidden.len() == 0,
        positions.len() == 0,
        slots.len() == 0,
    ensures
        decoder_layer_step_repr(
            common, extension, hidden, positions, k_cache, v_cache, slots,
            cu_q, cu_k, max_q, max_k, block_table,
        ).0.len() == 0,
        decoder_layer_step_repr(
            common, extension, hidden, positions, k_cache, v_cache, slots,
            cu_q, cu_k, max_q, max_k, block_table,
        ).1 == (k_cache, v_cache),
{
    let pre = attention_pre_store_repr(common, extension, hidden, positions);
    lemma_attention_pre_store_repr_shape(common, extension, hidden, positions);
    assert(pre.1.len() == 0);
    RT::lemma_store_kv_cache_repr_empty_rows(
        pre.1, pre.2, k_cache, v_cache, slots,
    );
    lemma_decoder_layer_step_repr_shape(
        common, extension, hidden, positions, k_cache, v_cache, slots,
        cu_q, cu_k, max_q, max_k, block_table,
    );
}

pub proof fn lemma_layer_chain_repr_shape(
    common_layers: Seq<LayerWeightsRepr>,
    extension_layers: Seq<FourNormGatedLayerExtensionRepr>,
    hidden: Tensor2D,
    positions: IntTensor1D,
    caches: Seq<(KVCacheLayerRepr, KVCacheLayerRepr)>,
    slots: Seq<int>,
    cu_q: Seq<int>,
    cu_k: Seq<int>,
    max_q: nat,
    max_k: nat,
    block_table: Seq<Seq<BlockId>>,
    start: nat,
)
    requires
        start <= common_layers.len(),
        extension_layers.len() == common_layers.len(),
        caches.len() >= common_layers.len(),
        hidden.len() == positions.len(),
        hidden.len() == slots.len(),
    ensures
        layer_chain_repr(
            common_layers, extension_layers, hidden, positions, caches, slots,
            cu_q, cu_k, max_q, max_k, block_table, start,
        ).0.len() == hidden.len(),
        layer_chain_repr(
            common_layers, extension_layers, hidden, positions, caches, slots,
            cu_q, cu_k, max_q, max_k, block_table, start,
        ).1.len() == caches.len(),
    decreases (common_layers.len() - start) as nat,
{
    if start < common_layers.len() {
        let layer = decoder_layer_step_repr(
            common_layers[start as int], extension_layers[start as int], hidden,
            positions, caches[start as int].0, caches[start as int].1, slots,
            cu_q, cu_k, max_q, max_k, block_table,
        );
        lemma_decoder_layer_step_repr_shape(
            common_layers[start as int], extension_layers[start as int], hidden,
            positions, caches[start as int].0, caches[start as int].1, slots,
            cu_q, cu_k, max_q, max_k, block_table,
        );
        lemma_layer_chain_repr_shape(
            common_layers, extension_layers, layer.0, positions,
            caches.update(start as int, layer.1), slots, cu_q, cu_k,
            max_q, max_k, block_table, (start + 1) as nat,
        );
    }
}

pub proof fn lemma_layer_chain_repr_empty(
    common_layers: Seq<LayerWeightsRepr>,
    extension_layers: Seq<FourNormGatedLayerExtensionRepr>,
    hidden: Tensor2D,
    positions: IntTensor1D,
    caches: Seq<(KVCacheLayerRepr, KVCacheLayerRepr)>,
    slots: Seq<int>,
    cu_q: Seq<int>,
    cu_k: Seq<int>,
    max_q: nat,
    max_k: nat,
    block_table: Seq<Seq<BlockId>>,
    start: nat,
)
    requires
        start <= common_layers.len(),
        extension_layers.len() == common_layers.len(),
        caches.len() >= common_layers.len(),
        hidden.len() == 0,
        positions.len() == 0,
        slots.len() == 0,
    ensures
        layer_chain_repr(
            common_layers, extension_layers, hidden, positions, caches, slots,
            cu_q, cu_k, max_q, max_k, block_table, start,
        ).0.len() == 0,
        layer_chain_repr(
            common_layers, extension_layers, hidden, positions, caches, slots,
            cu_q, cu_k, max_q, max_k, block_table, start,
        ).1 == caches,
    decreases (common_layers.len() - start) as nat,
{
    if start < common_layers.len() {
        let layer = decoder_layer_step_repr(
            common_layers[start as int], extension_layers[start as int], hidden,
            positions, caches[start as int].0, caches[start as int].1, slots,
            cu_q, cu_k, max_q, max_k, block_table,
        );
        lemma_decoder_layer_step_repr_empty(
            common_layers[start as int], extension_layers[start as int], hidden,
            positions, caches[start as int].0, caches[start as int].1, slots,
            cu_q, cu_k, max_q, max_k, block_table,
        );
        assert(caches.update(start as int, layer.1) =~= caches);
        lemma_layer_chain_repr_empty(
            common_layers, extension_layers, layer.0, positions,
            caches.update(start as int, layer.1), slots, cu_q, cu_k,
            max_q, max_k, block_table, (start + 1) as nat,
        );
    }
}

pub proof fn lemma_layer_chain_cache_shape_preserved(
    common_layers: Seq<LayerWeightsRepr>,
    extension_layers: Seq<FourNormGatedLayerExtensionRepr>,
    hidden: Tensor2D,
    positions: IntTensor1D,
    caches: Seq<(KVCacheLayerRepr, KVCacheLayerRepr)>,
    slots: Seq<int>,
    cu_q: Seq<int>,
    cu_k: Seq<int>,
    max_q: nat,
    max_k: nat,
    block_table: Seq<Seq<BlockId>>,
    start: nat,
    num_pages: nat,
)
    requires
        start <= common_layers.len(),
        extension_layers.len() == common_layers.len(),
        caches.len() >= common_layers.len(),
        hidden.len() == positions.len(),
        hidden.len() == slots.len(),
        cache_sequence_page_shape(caches, num_pages),
    ensures
        cache_sequence_page_shape(
            layer_chain_repr(
                common_layers, extension_layers, hidden, positions, caches,
                slots, cu_q, cu_k, max_q, max_k, block_table, start,
            ).1,
            num_pages,
        ),
    decreases (common_layers.len() - start) as nat,
{
    if start < common_layers.len() {
        let current = start as int;
        let pre = attention_pre_store_repr(
            common_layers[current], extension_layers[current], hidden, positions,
        );
        let layer = decoder_layer_step_repr(
            common_layers[current], extension_layers[current], hidden,
            positions, caches[current].0, caches[current].1, slots,
            cu_q, cu_k, max_q, max_k, block_table,
        );
        RT::lemma_store_kv_cache_repr_shape_unconditional(
            pre.1, pre.2, caches[current].0, caches[current].1, slots,
        );
        let next_caches = caches.update(current, layer.1);
        assert(cache_sequence_page_shape(next_caches, num_pages)) by {
            reveal(cache_sequence_page_shape);
            assert forall|j: int| 0 <= j < next_caches.len() implies {
                &&& (#[trigger] next_caches[j]).0.len() == num_pages
                &&& next_caches[j].1.len() == num_pages
                &&& (forall|p: int| 0 <= p < next_caches[j].0.len() ==>
                    (#[trigger] next_caches[j].0[p]).len()
                        == crate::types::BLOCK_SIZE_SPEC as int)
                &&& (forall|p: int| 0 <= p < next_caches[j].1.len() ==>
                    (#[trigger] next_caches[j].1[p]).len()
                        == crate::types::BLOCK_SIZE_SPEC as int)
            } by {
                if j == current {
                    assert(layer.1 == RT::store_kv_cache_repr(
                        pre.1, pre.2, caches[current].0, caches[current].1, slots,
                    ));
                    assert forall|p: int| 0 <= p < next_caches[j].0.len()
                        implies (#[trigger] next_caches[j].0[p]).len()
                            == crate::types::BLOCK_SIZE_SPEC as int by {
                        assert(next_caches[j].0[p].len()
                            == caches[current].0[p].len());
                    }
                    assert forall|p: int| 0 <= p < next_caches[j].1.len()
                        implies (#[trigger] next_caches[j].1[p]).len()
                            == crate::types::BLOCK_SIZE_SPEC as int by {
                        assert(next_caches[j].1[p].len()
                            == caches[current].1[p].len());
                    }
                } else {
                    assert(next_caches[j] == caches[j]);
                }
            }
        }
        lemma_decoder_layer_step_repr_shape(
            common_layers[current], extension_layers[current], hidden,
            positions, caches[current].0, caches[current].1, slots,
            cu_q, cu_k, max_q, max_k, block_table,
        );
        lemma_layer_chain_cache_shape_preserved(
            common_layers, extension_layers, layer.0, positions,
            next_caches, slots, cu_q, cu_k, max_q, max_k, block_table,
            (start + 1) as nat, num_pages,
        );
    }
}

// A chain starting at `start` never revisits cache entries for earlier
// layers.  This is the cache-state counterpart of the hidden framing lemma
// below and lets relational proofs identify the final cache at the layer just
// processed with that layer's post-store cache.
pub proof fn lemma_layer_chain_cache_before_start_unchanged(
    common_layers: Seq<LayerWeightsRepr>,
    extension_layers: Seq<FourNormGatedLayerExtensionRepr>,
    hidden: Tensor2D,
    positions: IntTensor1D,
    caches: Seq<(KVCacheLayerRepr, KVCacheLayerRepr)>,
    slots: Seq<int>,
    cu_q: Seq<int>,
    cu_k: Seq<int>,
    max_q: nat,
    max_k: nat,
    block_table: Seq<Seq<BlockId>>,
    start: nat,
    layer: int,
)
    requires
        start <= common_layers.len(),
        extension_layers.len() == common_layers.len(),
        caches.len() >= common_layers.len(),
        hidden.len() == positions.len(),
        hidden.len() == slots.len(),
        0 <= layer < start,
    ensures
        layer_chain_repr(
            common_layers, extension_layers, hidden, positions, caches, slots,
            cu_q, cu_k, max_q, max_k, block_table, start,
        ).1[layer] == caches[layer],
    decreases (common_layers.len() - start) as nat,
{
    if start < common_layers.len() {
        let current = start as int;
        let step = decoder_layer_step_repr(
            common_layers[current], extension_layers[current], hidden,
            positions, caches[current].0, caches[current].1, slots,
            cu_q, cu_k, max_q, max_k, block_table,
        );
        let next_caches = caches.update(current, step.1);
        assert(next_caches[layer] == caches[layer]);
        lemma_decoder_layer_step_repr_shape(
            common_layers[current], extension_layers[current], hidden,
            positions, caches[current].0, caches[current].1, slots,
            cu_q, cu_k, max_q, max_k, block_table,
        );
        lemma_layer_chain_cache_before_start_unchanged(
            common_layers, extension_layers, step.0, positions,
            next_caches, slots, cu_q, cu_k, max_q, max_k, block_table,
            (start + 1) as nat, layer,
        );
    }
}

// The hidden-state result of a chain starting at `start` cannot observe cache
// entries for already-processed layers.  The returned cache sequences may
// still differ below `start`, so this framing lemma states only hidden-output
// equality.
pub proof fn lemma_layer_chain_hidden_ignores_before_start(
    common_layers: Seq<LayerWeightsRepr>,
    extension_layers: Seq<FourNormGatedLayerExtensionRepr>,
    hidden: Tensor2D,
    positions: IntTensor1D,
    caches_a: Seq<(KVCacheLayerRepr, KVCacheLayerRepr)>,
    caches_b: Seq<(KVCacheLayerRepr, KVCacheLayerRepr)>,
    slots: Seq<int>,
    cu_q: Seq<int>,
    cu_k: Seq<int>,
    max_q: nat,
    max_k: nat,
    block_table: Seq<Seq<BlockId>>,
    start: nat,
)
    requires
        start <= common_layers.len(),
        extension_layers.len() == common_layers.len(),
        caches_a.len() >= common_layers.len(),
        caches_b.len() >= common_layers.len(),
        hidden.len() == positions.len(),
        hidden.len() == slots.len(),
        forall|layer: int| start <= layer < common_layers.len() ==>
            #[trigger] caches_a[layer] == caches_b[layer],
    ensures
        layer_chain_repr(
            common_layers, extension_layers, hidden, positions,
            caches_a, slots, cu_q, cu_k, max_q, max_k, block_table, start,
        ).0 == layer_chain_repr(
            common_layers, extension_layers, hidden, positions,
            caches_b, slots, cu_q, cu_k, max_q, max_k, block_table, start,
        ).0,
    decreases (common_layers.len() - start) as nat,
{
    if start < common_layers.len() {
        let layer = start as int;
        assert(caches_a[layer] == caches_b[layer]);
        let step_a = decoder_layer_step_repr(
            common_layers[layer], extension_layers[layer], hidden, positions,
            caches_a[layer].0, caches_a[layer].1, slots,
            cu_q, cu_k, max_q, max_k, block_table,
        );
        let step_b = decoder_layer_step_repr(
            common_layers[layer], extension_layers[layer], hidden, positions,
            caches_b[layer].0, caches_b[layer].1, slots,
            cu_q, cu_k, max_q, max_k, block_table,
        );
        assert(step_a == step_b);
        assert forall|later: int|
            (start + 1) as int <= later < common_layers.len() implies
                #[trigger] caches_a.update(layer, step_a.1)[later]
                    == caches_b.update(layer, step_b.1)[later] by {}
        lemma_decoder_layer_step_repr_shape(
            common_layers[layer], extension_layers[layer], hidden, positions,
            caches_a[layer].0, caches_a[layer].1, slots,
            cu_q, cu_k, max_q, max_k, block_table,
        );
        lemma_layer_chain_hidden_ignores_before_start(
            common_layers, extension_layers, step_a.0, positions,
            caches_a.update(layer, step_a.1),
            caches_b.update(layer, step_b.1),
            slots, cu_q, cu_k, max_q, max_k, block_table,
            (start + 1) as nat,
        );
    }
}

pub proof fn lemma_model_forward_logits_repr_shape(
    wr: ModelWeightsRepr,
    config: FourNormGatedDecoderConfigRepr,
    input_ids: IntTensor1D,
    positions: IntTensor1D,
    pre_kv: Seq<(KVCacheLayerRepr, KVCacheLayerRepr)>,
    slots: Seq<int>,
    cu_q: Seq<int>,
    cu_k: Seq<int>,
    max_q: nat,
    max_k: nat,
    block_table: Seq<Seq<BlockId>>,
)
    requires
        config.layers.len() == wr.layers.len(),
        pre_kv.len() >= wr.layers.len(),
        input_ids.len() == positions.len(),
        input_ids.len() == slots.len(),
    ensures model_forward_logits_repr(
        wr, config, input_ids, positions, pre_kv, slots,
        cu_q, cu_k, max_q, max_k, block_table,
    ).len() == input_ids.len(),
{
    broadcast use FOUR_NORM::lemma_final_logits_shape;
    reveal(scaled_embed_repr);
    lemma_layer_chain_repr_shape(
        wr.layers, config.layers,
        scaled_embed_repr(
            input_ids, wr.embed_weight, config.geometry.hidden_size,
        ),
        positions, pre_kv, slots, cu_q, cu_k, max_q, max_k, block_table, 0,
    );
}

pub proof fn lemma_model_forward_kv_reprs_len(
    wr: ModelWeightsRepr,
    config: FourNormGatedDecoderConfigRepr,
    input_ids: IntTensor1D,
    positions: IntTensor1D,
    pre_kv: Seq<(KVCacheLayerRepr, KVCacheLayerRepr)>,
    slots: Seq<int>,
    cu_q: Seq<int>,
    cu_k: Seq<int>,
    max_q: nat,
    max_k: nat,
    block_table: Seq<Seq<BlockId>>,
)
    requires
        config.layers.len() == wr.layers.len(),
        pre_kv.len() >= wr.layers.len(),
        input_ids.len() == positions.len(),
        input_ids.len() == slots.len(),
    ensures model_forward_kv_reprs(
        wr, config, input_ids, positions, pre_kv, slots,
        cu_q, cu_k, max_q, max_k, block_table,
    ).len() == pre_kv.len(),
{
    reveal(scaled_embed_repr);
    lemma_layer_chain_repr_shape(
        wr.layers, config.layers,
        scaled_embed_repr(
            input_ids, wr.embed_weight, config.geometry.hidden_size,
        ),
        positions, pre_kv, slots, cu_q, cu_k, max_q, max_k, block_table, 0,
    );
}

pub proof fn lemma_model_forward_kv_reprs_empty(
    wr: ModelWeightsRepr,
    config: FourNormGatedDecoderConfigRepr,
    input_ids: IntTensor1D,
    positions: IntTensor1D,
    pre_kv: Seq<(KVCacheLayerRepr, KVCacheLayerRepr)>,
    slots: Seq<int>,
    cu_q: Seq<int>,
    cu_k: Seq<int>,
    max_q: nat,
    max_k: nat,
    block_table: Seq<Seq<BlockId>>,
)
    requires
        config.layers.len() == wr.layers.len(),
        pre_kv.len() >= wr.layers.len(),
        input_ids.len() == 0,
        positions.len() == 0,
        slots.len() == 0,
    ensures
        model_forward_kv_reprs(
            wr, config, input_ids, positions, pre_kv, slots,
            cu_q, cu_k, max_q, max_k, block_table,
        ) == pre_kv,
{
    let hidden = scaled_embed_repr(
        input_ids, wr.embed_weight, config.geometry.hidden_size,
    );
    reveal(scaled_embed_repr);
    assert(hidden.len() == 0);
    lemma_layer_chain_repr_empty(
        wr.layers, config.layers, hidden, positions, pre_kv, slots,
        cu_q, cu_k, max_q, max_k, block_table, 0,
    );
}

pub proof fn lemma_model_forward_cache_shape_preserved(
    wr: ModelWeightsRepr,
    config: FourNormGatedDecoderConfigRepr,
    input_ids: IntTensor1D,
    positions: IntTensor1D,
    pre_kv: Seq<(KVCacheLayerRepr, KVCacheLayerRepr)>,
    slots: Seq<int>,
    cu_q: Seq<int>,
    cu_k: Seq<int>,
    max_q: nat,
    max_k: nat,
    block_table: Seq<Seq<BlockId>>,
    num_pages: nat,
)
    requires
        config.layers.len() == wr.layers.len(),
        pre_kv.len() >= wr.layers.len(),
        input_ids.len() == positions.len(),
        input_ids.len() == slots.len(),
        cache_sequence_page_shape(pre_kv, num_pages),
    ensures
        cache_sequence_page_shape(
            model_forward_kv_reprs(
                wr, config, input_ids, positions, pre_kv, slots,
                cu_q, cu_k, max_q, max_k, block_table,
            ),
            num_pages,
        ),
{
    let hidden = scaled_embed_repr(
        input_ids, wr.embed_weight, config.geometry.hidden_size,
    );
    reveal(scaled_embed_repr);
    assert(hidden.len() == input_ids.len());
    lemma_layer_chain_cache_shape_preserved(
        wr.layers, config.layers, hidden, positions, pre_kv, slots,
        cu_q, cu_k, max_q, max_k, block_table, 0, num_pages,
    );
}

} // verus!
