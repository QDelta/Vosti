//! Per-layer KV-row induction for the neutral dense SwiGLU decoder.
//!
//! The recursion is parameterized by the immutable forward configuration so
//! every family using this decoder instantiates exactly the same proof body.

#[cfg(verus_only)]
use crate::proof::model::dense_swiglu::batch_invariance as BI;
use crate::proof::model::dense_swiglu::cache_semantics as CC;
use crate::proof::model::dense_swiglu::relational as RELATIONAL;
use crate::proof::model::dense_swiglu::semantics as BD;
use crate::{types::*, proof::model::types::*, proof::tensor::types::*};
use crate::boundary::tensor_runtime as RT;
use vstd::prelude::*;

verus! {

// The K/V rows layer `layer` stores, given the `(hidden, residual)` entering
// layer `start` and the pre-chain caches (attention at layers in
// `[start, layer)` reads `kv[idx]`, which the fold leaves untouched at each
// layer's own index).  Mirrors `CC::layer_chain_kv_reprs`'s hidden advance
// without tracking the cache updates (they never affect an index >= the
// current layer).
pub open spec fn chain_layer_kv_rows(
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
    layer: nat,
) -> (Tensor2D, Tensor2D)
    recommends
        start <= layer < layers.len(),
        kv_cache_reprs.len() >= layers.len(),
    decreases layer - start,
{
    if layer <= start {
        let wr = layers[start as int];
        let normed = BD::add_rms_norm_repr(config, hidden_repr, residual_repr, wr.input_norm).0;
        (BD::pre_attention_repr(config, wr, normed, positions_repr).1,
         RT::view_as_kv_repr(RT::qkv_linear_repr(
             normed, wr.q_proj, wr.k_proj, wr.v_proj,
         ).2))
    } else {
        let next = BD::decoder_layer_output_repr(config, layers[start as int],
            hidden_repr, residual_repr, positions_repr,
            kv_cache_reprs[start as int].0, kv_cache_reprs[start as int].1, slot_repr,
            cu_q_repr, cu_k_repr, max_seqlen_q, max_seqlen_k, bt_repr);
        chain_layer_kv_rows(config, layers, next.0, next.1, positions_repr,
            kv_cache_reprs, slot_repr, cu_q_repr, cu_k_repr,
            max_seqlen_q, max_seqlen_k, bt_repr, (start + 1) as nat, layer)
    }
}

// The K/V rows the whole forward stores at `layer` (first layer enters via
// `rms_norm` on the embedding; later layers via the chain from index 1).
pub open spec fn model_layer_kv_rows(
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
    layer: nat,
) -> (Tensor2D, Tensor2D)
    recommends
        layer < wr.layers.len(),
        pre_kv_reprs.len() >= wr.layers.len(),
{
    let embed = RT::embed_repr(input_ids_repr, wr.embed_weight);
    if layer == 0 {
        let wr0 = wr.layers[0];
        let normed = BD::rms_norm_repr(config, embed, wr0.input_norm);
        (BD::pre_attention_repr(config, wr0, normed, positions_repr).1,
         RT::view_as_kv_repr(RT::qkv_linear_repr(
             normed, wr0.q_proj, wr0.k_proj, wr0.v_proj,
         ).2))
    } else {
        let first_out = BD::first_decoder_layer_output_repr(config, wr.layers[0],
            embed, positions_repr, pre_kv_reprs[0].0, pre_kv_reprs[0].1, slot_repr,
            cu_q_repr, cu_k_repr, max_seqlen_q, max_seqlen_k, bt_repr);
        chain_layer_kv_rows(config, wr.layers, first_out.0, first_out.1, positions_repr,
            pre_kv_reprs, slot_repr, cu_q_repr, cu_k_repr,
            max_seqlen_q, max_seqlen_k, bt_repr, 1, layer)
    }
}

// Row-level counterpart of `BD::layer_chain_request_isolation`.  The existing
// capstone exposes only hidden/residual (and ultimately logits) equality.  KV
// fidelity additionally needs the K/V rows produced *at* an arbitrary target
// layer, so this induction stops immediately before that layer and applies the
// row-wise store-input theorem there.
pub proof fn lemma_chain_layer_kv_rows_request_isolation(
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
    slot_i: Seq<int>,
    i: nat,
    start: nat,
    layer: nat,
)
    requires
        RT::paged_attention_numeric_domain(),
        start <= layer < layers.len(),
        kv_cache_reprs.len() >= layers.len(),
        hidden_repr.len() == residual_repr.len(),
        hidden_repr.len() == positions_repr.len(),
        slot_repr.len() == hidden_repr.len(),
        cu_q_repr.len() == cu_k_repr.len(),
        cu_k_repr.len() == bt_repr.len() + 1,
        i < bt_repr.len(),
        cu_q_repr[0] == 0,
        cu_k_repr[0] == 0,
        cu_q_repr[bt_repr.len() as int] == hidden_repr.len() as int,
        forall|j: int| 0 <= j < bt_repr.len() as int ==>
            cu_q_repr[j] < #[trigger] cu_q_repr[j + 1],
        forall|j: int| 0 <= j < bt_repr.len() as int ==>
            cu_k_repr[j] < #[trigger] cu_k_repr[j + 1],
        0 <= cu_q_repr[i as int] < cu_q_repr[i as int + 1]
            <= hidden_repr.len() as int,
        0 <= cu_k_repr[i as int] < cu_k_repr[i as int + 1],
        max_seqlen_q > 0,
        cu_q_repr[i as int + 1] - cu_q_repr[i as int]
            <= max_seqlen_q as int,
        cu_k_repr[i as int + 1] - cu_k_repr[i as int]
            <= max_seqlen_k as int,
        cu_q_repr[i as int + 1] - cu_q_repr[i as int]
            <= cu_k_repr[i as int + 1] - cu_k_repr[i as int],
        crate::proof::tensor::geometry::blocks_needed_for(
            (cu_k_repr[i as int + 1] - cu_k_repr[i as int]) as nat,
        ) <= bt_repr[i as int].len(),
        slot_i == slot_repr.subrange(
            cu_q_repr[i as int], cu_q_repr[i as int + 1],
        ),
        forall|m: int, l: int|
            #![trigger slot_repr.subrange(0, cu_q_repr[i as int])[m]
                / (crate::types::BLOCK_SIZE_SPEC as int), bt_repr[i as int][l]]
            0 <= m < cu_q_repr[i as int]
                && 0 <= l < bt_repr[i as int].len() ==>
                slot_repr.subrange(0, cu_q_repr[i as int])[m]
                    / (crate::types::BLOCK_SIZE_SPEC as int)
                    != bt_repr[i as int][l] as int,
        forall|m: int, l: int|
            #![trigger slot_repr.subrange(
                cu_q_repr[i as int + 1], slot_repr.len() as int,
            )[m] / (crate::types::BLOCK_SIZE_SPEC as int), bt_repr[i as int][l]]
            0 <= m < slot_repr.len() - cu_q_repr[i as int + 1]
                && 0 <= l < bt_repr[i as int].len() ==>
                slot_repr.subrange(
                    cu_q_repr[i as int + 1], slot_repr.len() as int,
                )[m] / (crate::types::BLOCK_SIZE_SPEC as int)
                    != bt_repr[i as int][l] as int,
        forall|ell: int, pos: nat|
            #![trigger kv_cache_reprs[ell].0,
                crate::proof::tensor::geometry::block_table_slot(bt_repr[i as int], pos)]
            start <= ell <= layer as int
                && pos < (cu_k_repr[i as int + 1] - cu_k_repr[i as int]) as nat ==>
                crate::proof::tensor::geometry::slot_in_cache(
                    kv_cache_reprs[ell].0,
                    crate::proof::tensor::geometry::block_table_slot(bt_repr[i as int], pos),
                )
                && crate::proof::tensor::geometry::slot_in_cache(
                    kv_cache_reprs[ell].1,
                    crate::proof::tensor::geometry::block_table_slot(bt_repr[i as int], pos),
                ),
        forall|ell: int| start <= ell <= layer as int ==>
            #[trigger] RT::paged_attention_launch_ready(
                hidden_repr.len(), kv_cache_reprs[ell].0,
                kv_cache_reprs[ell].1, cu_q_repr, cu_k_repr,
                max_seqlen_q, max_seqlen_k, bt_repr,
            ),
    ensures ({
        let lo = cu_q_repr[i as int];
        let hi = cu_q_repr[i as int + 1];
        let qd = (hi - lo) as nat;
        let kd = (cu_k_repr[i as int + 1] - cu_k_repr[i as int]) as nat;
        let batched = chain_layer_kv_rows(config,
            layers, hidden_repr, residual_repr, positions_repr, kv_cache_reprs,
            slot_repr, cu_q_repr, cu_k_repr, max_seqlen_q, max_seqlen_k,
            bt_repr, start, layer,
        );
        let single = chain_layer_kv_rows(config,
            layers,
            hidden_repr.subrange(lo, hi),
            residual_repr.subrange(lo, hi),
            positions_repr.subrange(lo, hi),
            kv_cache_reprs,
            slot_i,
            seq![0int, qd as int],
            seq![0int, kd as int],
            qd,
            kd,
            seq![bt_repr[i as int]],
            start,
            layer,
        );
        &&& batched.0.subrange(lo, hi) == single.0
        &&& batched.1.subrange(lo, hi) == single.1
    }),
    decreases layer - start,
{
    broadcast use RT::lemma_add_rms_norm_repr_shape;
    let lo = cu_q_repr[i as int];
    let hi = cu_q_repr[i as int + 1];
    let qd = (hi - lo) as nat;
    let kd = (cu_k_repr[i as int + 1] - cu_k_repr[i as int]) as nat;
    let sing_cu_q = seq![0int, qd as int];
    let sing_cu_k = seq![0int, kd as int];
    let sing_bt = seq![bt_repr[i as int]];

    if layer <= start {
        let wr = layers[start as int];
        let normed = BD::add_rms_norm_repr(config,
            hidden_repr, residual_repr, wr.input_norm,
        ).0;
        BI::add_rms_norm_subrange_invariance(
            hidden_repr, residual_repr, wr.input_norm,
            config.rms_norm_epsilon, lo, hi,
        );
        RELATIONAL::kv_store_inputs_request_local(config,
            wr, normed, positions_repr, lo, hi,
        );
    } else {
        RELATIONAL::lemma_decoder_layer_attention_launch_ready_from_pre_store(config,
            layers[start as int], hidden_repr, residual_repr, positions_repr,
            kv_cache_reprs[start as int].0, kv_cache_reprs[start as int].1,
            slot_repr, cu_q_repr, cu_k_repr, max_seqlen_q, max_seqlen_k,
            bt_repr,
        );
        RELATIONAL::decoder_layer_isolation_from_layout(config,
            layers[start as int], hidden_repr, residual_repr, positions_repr,
            kv_cache_reprs[start as int].0, kv_cache_reprs[start as int].1,
            slot_repr, cu_q_repr, cu_k_repr, max_seqlen_q, max_seqlen_k,
            bt_repr, slot_i, i,
        );
        let next_b = BD::decoder_layer_output_repr(config,
            layers[start as int], hidden_repr, residual_repr, positions_repr,
            kv_cache_reprs[start as int].0, kv_cache_reprs[start as int].1,
            slot_repr, cu_q_repr, cu_k_repr, max_seqlen_q, max_seqlen_k,
            bt_repr,
        );
        let next_s = BD::decoder_layer_output_repr(config,
            layers[start as int],
            hidden_repr.subrange(lo, hi),
            residual_repr.subrange(lo, hi),
            positions_repr.subrange(lo, hi),
            kv_cache_reprs[start as int].0,
            kv_cache_reprs[start as int].1,
            slot_i,
            sing_cu_q,
            sing_cu_k,
            qd,
            kd,
            sing_bt,
        );
        BD::lemma_decoder_layer_output_repr_shape(config,
            layers[start as int], hidden_repr, residual_repr, positions_repr,
            kv_cache_reprs[start as int].0, kv_cache_reprs[start as int].1,
            slot_repr, cu_q_repr, cu_k_repr, max_seqlen_q, max_seqlen_k,
            bt_repr,
        );
        assert(next_b.0.subrange(lo, hi) == next_s.0);
        assert(next_b.1.subrange(lo, hi) == next_s.1);
        lemma_chain_layer_kv_rows_request_isolation(config,
            layers, next_b.0, next_b.1, positions_repr, kv_cache_reprs,
            slot_repr, cu_q_repr, cu_k_repr, max_seqlen_q, max_seqlen_k,
            bt_repr, slot_i, i, (start + 1) as nat, layer,
        );
    }
}

// Whole-model row isolation: request `i`'s slice of the batched K/V rows at an
// arbitrary layer equals the rows produced by a singleton run using that
// request's own paged geometry.  This is the cache-value analogue of
// `BD::model_forward_request_isolation` (which exposes only logits).
pub proof fn lemma_model_layer_kv_rows_request_isolation(
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
    slot_i: Seq<int>,
    i: nat,
    layer: nat,
)
    requires
        RT::paged_attention_numeric_domain(),
        layer < wr.layers.len(),
        wr.layers.len() > 0,
        kv_cache_reprs.len() >= wr.layers.len(),
        input_ids_repr.len() == positions_repr.len(),
        slot_repr.len() == input_ids_repr.len(),
        cu_q_repr.len() == cu_k_repr.len(),
        cu_k_repr.len() == bt_repr.len() + 1,
        i < bt_repr.len(),
        cu_q_repr[0] == 0,
        cu_k_repr[0] == 0,
        cu_q_repr[bt_repr.len() as int] == input_ids_repr.len() as int,
        forall|j: int| 0 <= j < bt_repr.len() as int ==>
            cu_q_repr[j] < #[trigger] cu_q_repr[j + 1],
        forall|j: int| 0 <= j < bt_repr.len() as int ==>
            cu_k_repr[j] < #[trigger] cu_k_repr[j + 1],
        0 <= cu_q_repr[i as int] < cu_q_repr[i as int + 1]
            <= input_ids_repr.len() as int,
        0 <= cu_k_repr[i as int] < cu_k_repr[i as int + 1],
        max_seqlen_q > 0,
        cu_q_repr[i as int + 1] - cu_q_repr[i as int]
            <= max_seqlen_q as int,
        cu_k_repr[i as int + 1] - cu_k_repr[i as int]
            <= max_seqlen_k as int,
        cu_q_repr[i as int + 1] - cu_q_repr[i as int]
            <= cu_k_repr[i as int + 1] - cu_k_repr[i as int],
        crate::proof::tensor::geometry::blocks_needed_for(
            (cu_k_repr[i as int + 1] - cu_k_repr[i as int]) as nat,
        ) <= bt_repr[i as int].len(),
        slot_i == slot_repr.subrange(
            cu_q_repr[i as int], cu_q_repr[i as int + 1],
        ),
        forall|m: int, l: int|
            #![trigger slot_repr.subrange(0, cu_q_repr[i as int])[m]
                / (crate::types::BLOCK_SIZE_SPEC as int), bt_repr[i as int][l]]
            0 <= m < cu_q_repr[i as int]
                && 0 <= l < bt_repr[i as int].len() ==>
                slot_repr.subrange(0, cu_q_repr[i as int])[m]
                    / (crate::types::BLOCK_SIZE_SPEC as int)
                    != bt_repr[i as int][l] as int,
        forall|m: int, l: int|
            #![trigger slot_repr.subrange(
                cu_q_repr[i as int + 1], slot_repr.len() as int,
            )[m] / (crate::types::BLOCK_SIZE_SPEC as int), bt_repr[i as int][l]]
            0 <= m < slot_repr.len() - cu_q_repr[i as int + 1]
                && 0 <= l < bt_repr[i as int].len() ==>
                slot_repr.subrange(
                    cu_q_repr[i as int + 1], slot_repr.len() as int,
                )[m] / (crate::types::BLOCK_SIZE_SPEC as int)
                    != bt_repr[i as int][l] as int,
        forall|ell: int, pos: nat|
            #![trigger kv_cache_reprs[ell].0,
                crate::proof::tensor::geometry::block_table_slot(bt_repr[i as int], pos)]
            0 <= ell < wr.layers.len()
                && pos < (cu_k_repr[i as int + 1] - cu_k_repr[i as int]) as nat ==>
                crate::proof::tensor::geometry::slot_in_cache(
                    kv_cache_reprs[ell].0,
                    crate::proof::tensor::geometry::block_table_slot(bt_repr[i as int], pos),
                )
                && crate::proof::tensor::geometry::slot_in_cache(
                    kv_cache_reprs[ell].1,
                    crate::proof::tensor::geometry::block_table_slot(bt_repr[i as int], pos),
                ),
        forall|ell: int| 0 <= ell <= layer as int ==>
            #[trigger] RT::paged_attention_launch_ready(
                input_ids_repr.len(), kv_cache_reprs[ell].0,
                kv_cache_reprs[ell].1, cu_q_repr, cu_k_repr,
                max_seqlen_q, max_seqlen_k, bt_repr,
            ),
    ensures ({
        let lo = cu_q_repr[i as int];
        let hi = cu_q_repr[i as int + 1];
        let qd = (hi - lo) as nat;
        let kd = (cu_k_repr[i as int + 1] - cu_k_repr[i as int]) as nat;
        let batched = model_layer_kv_rows(config,
            wr, input_ids_repr, positions_repr, kv_cache_reprs, slot_repr,
            cu_q_repr, cu_k_repr, max_seqlen_q, max_seqlen_k, bt_repr, layer,
        );
        let single = model_layer_kv_rows(config,
            wr,
            input_ids_repr.subrange(lo, hi),
            positions_repr.subrange(lo, hi),
            kv_cache_reprs,
            slot_i,
            seq![0int, qd as int],
            seq![0int, kd as int],
            qd,
            kd,
            seq![bt_repr[i as int]],
            layer,
        );
        &&& batched.0.subrange(lo, hi) == single.0
        &&& batched.1.subrange(lo, hi) == single.1
    }),
{
    broadcast use {
        RT::lemma_embed_repr_shape,
        RT::lemma_rms_norm_repr_shape,
    };
    let lo = cu_q_repr[i as int];
    let hi = cu_q_repr[i as int + 1];
    let qd = (hi - lo) as nat;
    let kd = (cu_k_repr[i as int + 1] - cu_k_repr[i as int]) as nat;
    let sing_cu_q = seq![0int, qd as int];
    let sing_cu_k = seq![0int, kd as int];
    let sing_bt = seq![bt_repr[i as int]];
    let embed = RT::embed_repr(input_ids_repr, wr.embed_weight);
    let embed_i = RT::embed_repr(input_ids_repr.subrange(lo, hi), wr.embed_weight);

    BI::embed_subrange_invariance(input_ids_repr, wr.embed_weight, lo, hi);
    assert(embed.subrange(lo, hi) == embed_i);
    if layer == 0 {
        let wr0 = wr.layers[0];
        let normed = BD::rms_norm_repr(config, embed, wr0.input_norm);
        BI::rms_norm_subrange_invariance(
            embed, wr0.input_norm, config.rms_norm_epsilon, lo, hi,
        );
        RELATIONAL::kv_store_inputs_request_local(config,
            wr0, normed, positions_repr, lo, hi,
        );
    } else {
        RELATIONAL::lemma_first_decoder_layer_attention_launch_ready_from_pre_store(config,
            wr.layers[0], embed, positions_repr,
            kv_cache_reprs[0].0, kv_cache_reprs[0].1, slot_repr,
            cu_q_repr, cu_k_repr, max_seqlen_q, max_seqlen_k, bt_repr,
        );
        RELATIONAL::first_decoder_layer_isolation_from_layout(config,
            wr.layers[0], embed, positions_repr,
            kv_cache_reprs[0].0, kv_cache_reprs[0].1,
            slot_repr, cu_q_repr, cu_k_repr, max_seqlen_q, max_seqlen_k,
            bt_repr, slot_i, i,
        );
        let first_b = BD::first_decoder_layer_output_repr(config,
            wr.layers[0], embed, positions_repr,
            kv_cache_reprs[0].0, kv_cache_reprs[0].1,
            slot_repr, cu_q_repr, cu_k_repr, max_seqlen_q, max_seqlen_k,
            bt_repr,
        );
        let first_s = BD::first_decoder_layer_output_repr(config,
            wr.layers[0], embed_i, positions_repr.subrange(lo, hi),
            kv_cache_reprs[0].0, kv_cache_reprs[0].1,
            slot_i, sing_cu_q, sing_cu_k, qd, kd, sing_bt,
        );
        BD::lemma_first_decoder_layer_output_repr_shape(config,
            wr.layers[0], embed, positions_repr,
            kv_cache_reprs[0].0, kv_cache_reprs[0].1,
            slot_repr, cu_q_repr, cu_k_repr, max_seqlen_q, max_seqlen_k,
            bt_repr,
        );
        assert(first_b.0.subrange(lo, hi) == first_s.0);
        assert(first_b.1.subrange(lo, hi) == first_s.1);
        lemma_chain_layer_kv_rows_request_isolation(config,
            wr.layers, first_b.0, first_b.1, positions_repr,
            kv_cache_reprs, slot_repr, cu_q_repr, cu_k_repr,
            max_seqlen_q, max_seqlen_k, bt_repr, slot_i, i, 1, layer,
        );
    }
}

// Row-level counterpart of `BD::layer_chain_relocation`: swapping physical KV
// geometry preserves not only the layer outputs but also the K/V rows produced
// at a later target layer.  At the target layer the rows are functions only of
// the equal hidden/residual inputs and positions; caches matter only while
// advancing through the preceding layers.
#[verifier::spinoff_prover]
pub proof fn lemma_chain_layer_kv_rows_relocation(
    config: DenseSwiGluForwardConfigRepr,
    layers: Seq<LayerWeightsRepr>,
    hidden_repr: Tensor2D,
    residual_repr: Tensor2D,
    positions_repr: IntTensor1D,
    kv_a: Seq<(KVCacheLayerRepr, KVCacheLayerRepr)>,
    slot_a: Seq<int>,
    bt_row_a: Seq<BlockId>,
    kv_b: Seq<(KVCacheLayerRepr, KVCacheLayerRepr)>,
    slot_b: Seq<int>,
    bt_row_b: Seq<BlockId>,
    q_len: nat,
    k_len: nat,
    start: nat,
    layer: nat,
)
    requires
        RT::paged_attention_numeric_domain(),
        start <= layer < layers.len(),
        kv_a.len() >= layers.len(),
        kv_b.len() >= layers.len(),
        hidden_repr.len() == residual_repr.len(),
        hidden_repr.len() == positions_repr.len(),
        slot_a.len() == hidden_repr.len(),
        slot_b.len() == hidden_repr.len(),
        hidden_repr.len() == q_len,
        q_len > 0,
        q_len <= k_len,
        crate::proof::tensor::geometry::blocks_needed_for(k_len) <= bt_row_a.len(),
        crate::proof::tensor::geometry::blocks_needed_for(k_len) <= bt_row_b.len(),
        forall|j: int| 0 <= j < q_len as int ==>
            crate::proof::tensor::geometry::block_table_slot(
                bt_row_a, (k_len - q_len + j as nat) as nat,
            ) == #[trigger] (slot_a[j]) as nat
            && crate::proof::tensor::geometry::block_table_slot(
                bt_row_b, (k_len - q_len + j as nat) as nat,
            ) == (slot_b[j]) as nat,
        forall|j: int| 0 <= j < q_len as int ==>
            #[trigger] (slot_a[j]) >= 0 && (slot_b[j]) >= 0,
        forall|j: int, m: int| #![trigger slot_a[m], slot_a[j]]
            0 <= j < q_len as int && j < m < q_len as int ==>
                slot_a[m] != slot_a[j],
        forall|j: int, m: int| #![trigger slot_b[m], slot_b[j]]
            0 <= j < q_len as int && j < m < q_len as int ==>
                slot_b[m] != slot_b[j],
        forall|pos: nat|
            #![trigger crate::proof::tensor::geometry::block_table_slot(bt_row_a, pos)]
            pos < k_len - q_len ==>
                !slot_a.contains(
                    crate::proof::tensor::geometry::block_table_slot(bt_row_a, pos) as int,
                )
                && !slot_b.contains(
                    crate::proof::tensor::geometry::block_table_slot(bt_row_b, pos) as int,
                ),
        forall|ell: int, j: int| #![trigger kv_a[ell].0, slot_a[j]]
            start <= ell <= layer as int && 0 <= j < q_len as int ==>
                crate::proof::tensor::geometry::slot_in_cache(kv_a[ell].0, (slot_a[j]) as nat)
                && crate::proof::tensor::geometry::slot_in_cache(kv_a[ell].1, (slot_a[j]) as nat)
                && crate::proof::tensor::geometry::slot_in_cache(kv_b[ell].0, (slot_b[j]) as nat)
                && crate::proof::tensor::geometry::slot_in_cache(kv_b[ell].1, (slot_b[j]) as nat),
        forall|ell: int, pos: nat|
            #![trigger kv_a[ell].0,
                crate::proof::tensor::geometry::block_table_slot(bt_row_a, pos)]
            start <= ell <= layer as int && pos < k_len - q_len ==>
                crate::proof::tensor::geometry::slot_in_cache(
                    kv_a[ell].0,
                    crate::proof::tensor::geometry::block_table_slot(bt_row_a, pos),
                )
                && crate::proof::tensor::geometry::slot_in_cache(
                    kv_a[ell].1,
                    crate::proof::tensor::geometry::block_table_slot(bt_row_a, pos),
                )
                && crate::proof::tensor::geometry::slot_in_cache(
                    kv_b[ell].0,
                    crate::proof::tensor::geometry::block_table_slot(bt_row_b, pos),
                )
                && crate::proof::tensor::geometry::slot_in_cache(
                    kv_b[ell].1,
                    crate::proof::tensor::geometry::block_table_slot(bt_row_b, pos),
                )
                && crate::proof::tensor::geometry::cache_at(
                    kv_a[ell].0,
                    crate::proof::tensor::geometry::block_table_slot(bt_row_a, pos),
                ) == crate::proof::tensor::geometry::cache_at(
                    kv_b[ell].0,
                    crate::proof::tensor::geometry::block_table_slot(bt_row_b, pos),
                )
                && crate::proof::tensor::geometry::cache_at(
                    kv_a[ell].1,
                    crate::proof::tensor::geometry::block_table_slot(bt_row_a, pos),
                ) == crate::proof::tensor::geometry::cache_at(
                    kv_b[ell].1,
                    crate::proof::tensor::geometry::block_table_slot(bt_row_b, pos),
                ),
    ensures
        chain_layer_kv_rows(config,
            layers, hidden_repr, residual_repr, positions_repr, kv_a, slot_a,
            seq![0int, q_len as int], seq![0int, k_len as int], q_len, k_len,
            seq![bt_row_a], start, layer,
        ) == chain_layer_kv_rows(config,
            layers, hidden_repr, residual_repr, positions_repr, kv_b, slot_b,
            seq![0int, q_len as int], seq![0int, k_len as int], q_len, k_len,
            seq![bt_row_b], start, layer,
        ),
    decreases layer - start,
{
    if layer <= start {
    } else {
        assert forall|j: int| 0 <= j < q_len as int implies
            crate::proof::tensor::geometry::block_table_slot(
                bt_row_a, (k_len - q_len + j as nat) as nat,
            ) == #[trigger] (slot_a[j]) as nat
            && crate::proof::tensor::geometry::block_table_slot(
                bt_row_b, (k_len - q_len + j as nat) as nat,
            ) == (slot_b[j]) as nat by {}
        assert(RELATIONAL::fresh_writes_miss_cached_prefix(
            slot_a, bt_row_a, slot_b, bt_row_b, k_len - q_len,
        )) by {
            reveal(RELATIONAL::fresh_writes_miss_cached_prefix);
        }
        RELATIONAL::decoder_layer_relocation_from_layout(config,
            layers[start as int], hidden_repr, residual_repr, positions_repr,
            kv_a[start as int].0, kv_a[start as int].1, slot_a, bt_row_a,
            kv_b[start as int].0, kv_b[start as int].1, slot_b, bt_row_b,
            q_len, k_len,
        );
        let next_a = BD::decoder_layer_output_repr(config,
            layers[start as int], hidden_repr, residual_repr, positions_repr,
            kv_a[start as int].0, kv_a[start as int].1, slot_a,
            seq![0int, q_len as int], seq![0int, k_len as int], q_len, k_len,
            seq![bt_row_a],
        );
        let next_b = BD::decoder_layer_output_repr(config,
            layers[start as int], hidden_repr, residual_repr, positions_repr,
            kv_b[start as int].0, kv_b[start as int].1, slot_b,
            seq![0int, q_len as int], seq![0int, k_len as int], q_len, k_len,
            seq![bt_row_b],
        );
        assert(next_a == next_b);
        BD::lemma_decoder_layer_output_repr_shape(config,
            layers[start as int], hidden_repr, residual_repr, positions_repr,
            kv_a[start as int].0, kv_a[start as int].1, slot_a,
            seq![0int, q_len as int], seq![0int, k_len as int], q_len, k_len,
            seq![bt_row_a],
        );
        assert forall|ell: int, j: int| #![trigger kv_a[ell].0, slot_a[j]]
            (start + 1) as int <= ell <= layer as int
                && 0 <= j < q_len as int implies
                crate::proof::tensor::geometry::slot_in_cache(kv_a[ell].0, (slot_a[j]) as nat)
                && crate::proof::tensor::geometry::slot_in_cache(kv_a[ell].1, (slot_a[j]) as nat)
                && crate::proof::tensor::geometry::slot_in_cache(kv_b[ell].0, (slot_b[j]) as nat)
                && crate::proof::tensor::geometry::slot_in_cache(kv_b[ell].1, (slot_b[j]) as nat)
        by {
            assert(start <= ell);
        }
        assert forall|ell: int, pos: nat|
            #![trigger kv_a[ell].0,
                crate::proof::tensor::geometry::block_table_slot(bt_row_a, pos)]
            (start + 1) as int <= ell <= layer as int
                && pos < k_len - q_len implies
                crate::proof::tensor::geometry::slot_in_cache(
                    kv_a[ell].0,
                    crate::proof::tensor::geometry::block_table_slot(bt_row_a, pos),
                )
                && crate::proof::tensor::geometry::slot_in_cache(
                    kv_a[ell].1,
                    crate::proof::tensor::geometry::block_table_slot(bt_row_a, pos),
                )
                && crate::proof::tensor::geometry::slot_in_cache(
                    kv_b[ell].0,
                    crate::proof::tensor::geometry::block_table_slot(bt_row_b, pos),
                )
                && crate::proof::tensor::geometry::slot_in_cache(
                    kv_b[ell].1,
                    crate::proof::tensor::geometry::block_table_slot(bt_row_b, pos),
                )
                && crate::proof::tensor::geometry::cache_at(
                    kv_a[ell].0,
                    crate::proof::tensor::geometry::block_table_slot(bt_row_a, pos),
                ) == crate::proof::tensor::geometry::cache_at(
                    kv_b[ell].0,
                    crate::proof::tensor::geometry::block_table_slot(bt_row_b, pos),
                )
                && crate::proof::tensor::geometry::cache_at(
                    kv_a[ell].1,
                    crate::proof::tensor::geometry::block_table_slot(bt_row_a, pos),
                ) == crate::proof::tensor::geometry::cache_at(
                    kv_b[ell].1,
                    crate::proof::tensor::geometry::block_table_slot(bt_row_b, pos),
                )
        by {
            assert(start <= ell);
        }
        assert forall|j: int| 0 <= j < q_len as int implies
            crate::proof::tensor::geometry::block_table_slot(
                bt_row_a, (k_len - q_len + j as nat) as nat,
            ) == #[trigger] (slot_a[j]) as nat
            && crate::proof::tensor::geometry::block_table_slot(
                bt_row_b, (k_len - q_len + j as nat) as nat,
            ) == (slot_b[j]) as nat by {}
        lemma_chain_layer_kv_rows_relocation(config,
            layers, next_a.0, next_a.1, positions_repr,
            kv_a, slot_a, bt_row_a, kv_b, slot_b, bt_row_b,
            q_len, k_len, (start + 1) as nat, layer,
        );
    }
}

// Whole-model row relocation, parallel to `BD::model_forward_relocation`.
pub proof fn lemma_model_layer_kv_rows_relocation(
    config: DenseSwiGluForwardConfigRepr,
    wr: ModelWeightsRepr,
    input_ids_repr: IntTensor1D,
    positions_repr: IntTensor1D,
    kv_a: Seq<(KVCacheLayerRepr, KVCacheLayerRepr)>,
    slot_a: Seq<int>,
    bt_row_a: Seq<BlockId>,
    kv_b: Seq<(KVCacheLayerRepr, KVCacheLayerRepr)>,
    slot_b: Seq<int>,
    bt_row_b: Seq<BlockId>,
    q_len: nat,
    k_len: nat,
    layer: nat,
)
    requires
        RT::paged_attention_numeric_domain(),
        layer < wr.layers.len(),
        wr.layers.len() > 0,
        kv_a.len() >= wr.layers.len(),
        kv_b.len() >= wr.layers.len(),
        input_ids_repr.len() == positions_repr.len(),
        slot_a.len() == input_ids_repr.len(),
        slot_b.len() == input_ids_repr.len(),
        input_ids_repr.len() == q_len,
        q_len > 0,
        q_len <= k_len,
        crate::proof::tensor::geometry::blocks_needed_for(k_len) <= bt_row_a.len(),
        crate::proof::tensor::geometry::blocks_needed_for(k_len) <= bt_row_b.len(),
        forall|j: int| 0 <= j < q_len as int ==>
            crate::proof::tensor::geometry::block_table_slot(
                bt_row_a, (k_len - q_len + j as nat) as nat,
            ) == #[trigger] (slot_a[j]) as nat
            && crate::proof::tensor::geometry::block_table_slot(
                bt_row_b, (k_len - q_len + j as nat) as nat,
            ) == (slot_b[j]) as nat,
        forall|j: int| 0 <= j < q_len as int ==>
            #[trigger] (slot_a[j]) >= 0 && (slot_b[j]) >= 0,
        forall|j: int, m: int| #![trigger slot_a[m], slot_a[j]]
            0 <= j < q_len as int && j < m < q_len as int ==>
                slot_a[m] != slot_a[j],
        forall|j: int, m: int| #![trigger slot_b[m], slot_b[j]]
            0 <= j < q_len as int && j < m < q_len as int ==>
                slot_b[m] != slot_b[j],
        forall|pos: nat|
            #![trigger crate::proof::tensor::geometry::block_table_slot(bt_row_a, pos)]
            pos < k_len - q_len ==>
                !slot_a.contains(
                    crate::proof::tensor::geometry::block_table_slot(bt_row_a, pos) as int,
                )
                && !slot_b.contains(
                    crate::proof::tensor::geometry::block_table_slot(bt_row_b, pos) as int,
                ),
        forall|ell: int, j: int| #![trigger kv_a[ell].0, slot_a[j]]
            0 <= ell < wr.layers.len() && 0 <= j < q_len as int ==>
                crate::proof::tensor::geometry::slot_in_cache(kv_a[ell].0, (slot_a[j]) as nat)
                && crate::proof::tensor::geometry::slot_in_cache(kv_a[ell].1, (slot_a[j]) as nat)
                && crate::proof::tensor::geometry::slot_in_cache(kv_b[ell].0, (slot_b[j]) as nat)
                && crate::proof::tensor::geometry::slot_in_cache(kv_b[ell].1, (slot_b[j]) as nat),
        forall|ell: int, pos: nat|
            #![trigger kv_a[ell].0,
                crate::proof::tensor::geometry::block_table_slot(bt_row_a, pos)]
            0 <= ell < wr.layers.len() && pos < k_len - q_len ==>
                crate::proof::tensor::geometry::slot_in_cache(
                    kv_a[ell].0,
                    crate::proof::tensor::geometry::block_table_slot(bt_row_a, pos),
                )
                && crate::proof::tensor::geometry::slot_in_cache(
                    kv_a[ell].1,
                    crate::proof::tensor::geometry::block_table_slot(bt_row_a, pos),
                )
                && crate::proof::tensor::geometry::slot_in_cache(
                    kv_b[ell].0,
                    crate::proof::tensor::geometry::block_table_slot(bt_row_b, pos),
                )
                && crate::proof::tensor::geometry::slot_in_cache(
                    kv_b[ell].1,
                    crate::proof::tensor::geometry::block_table_slot(bt_row_b, pos),
                )
                && crate::proof::tensor::geometry::cache_at(
                    kv_a[ell].0,
                    crate::proof::tensor::geometry::block_table_slot(bt_row_a, pos),
                ) == crate::proof::tensor::geometry::cache_at(
                    kv_b[ell].0,
                    crate::proof::tensor::geometry::block_table_slot(bt_row_b, pos),
                )
                && crate::proof::tensor::geometry::cache_at(
                    kv_a[ell].1,
                    crate::proof::tensor::geometry::block_table_slot(bt_row_a, pos),
                ) == crate::proof::tensor::geometry::cache_at(
                    kv_b[ell].1,
                    crate::proof::tensor::geometry::block_table_slot(bt_row_b, pos),
                ),
    ensures
        model_layer_kv_rows(config,
            wr, input_ids_repr, positions_repr, kv_a, slot_a,
            seq![0int, q_len as int], seq![0int, k_len as int], q_len, k_len,
            seq![bt_row_a], layer,
        ) == model_layer_kv_rows(config,
            wr, input_ids_repr, positions_repr, kv_b, slot_b,
            seq![0int, q_len as int], seq![0int, k_len as int], q_len, k_len,
            seq![bt_row_b], layer,
        ),
{
    broadcast use RT::lemma_embed_repr_shape;
    let embed = RT::embed_repr(input_ids_repr, wr.embed_weight);
    if layer == 0 {
    } else {
        RELATIONAL::first_decoder_layer_relocation_from_layout(config,
            wr.layers[0], embed, positions_repr,
            kv_a[0].0, kv_a[0].1, slot_a, bt_row_a,
            kv_b[0].0, kv_b[0].1, slot_b, bt_row_b, q_len, k_len,
        );
        let first_a = BD::first_decoder_layer_output_repr(config,
            wr.layers[0], embed, positions_repr,
            kv_a[0].0, kv_a[0].1, slot_a,
            seq![0int, q_len as int], seq![0int, k_len as int], q_len, k_len,
            seq![bt_row_a],
        );
        let first_b = BD::first_decoder_layer_output_repr(config,
            wr.layers[0], embed, positions_repr,
            kv_b[0].0, kv_b[0].1, slot_b,
            seq![0int, q_len as int], seq![0int, k_len as int], q_len, k_len,
            seq![bt_row_b],
        );
        assert(first_a == first_b);
        BD::lemma_first_decoder_layer_output_repr_shape(config,
            wr.layers[0], embed, positions_repr,
            kv_a[0].0, kv_a[0].1, slot_a,
            seq![0int, q_len as int], seq![0int, k_len as int], q_len, k_len,
            seq![bt_row_a],
        );
        lemma_chain_layer_kv_rows_relocation(config,
            wr.layers, first_a.0, first_a.1, positions_repr,
            kv_a, slot_a, bt_row_a, kv_b, slot_b, bt_row_b,
            q_len, k_len, 1, layer,
        );
    }
}

// The chain fold never touches cache indices outside `[start, layers.len())`.
pub proof fn lemma_layer_chain_kv_reprs_frame(
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
    j: int,
)
    requires
        start <= layers.len(),
        kv_cache_reprs.len() >= layers.len(),
        0 <= j < kv_cache_reprs.len(),
        j < start as int || j >= layers.len() as int,
    ensures
        CC::layer_chain_kv_reprs(config, layers, hidden_repr, residual_repr, positions_repr,
            kv_cache_reprs, slot_repr, cu_q_repr, cu_k_repr,
            max_seqlen_q, max_seqlen_k, bt_repr, start)[j] == kv_cache_reprs[j],
    decreases layers.len() - start,
{
    if start >= layers.len() {
    } else {
        let updated = CC::decoder_layer_kv_update_repr(config, layers[start as int],
            hidden_repr, residual_repr, positions_repr,
            kv_cache_reprs[start as int].0, kv_cache_reprs[start as int].1, slot_repr);
        let kv2 = kv_cache_reprs.update(start as int, updated);
        let next = BD::decoder_layer_output_repr(config, layers[start as int],
            hidden_repr, residual_repr, positions_repr,
            kv_cache_reprs[start as int].0, kv_cache_reprs[start as int].1, slot_repr,
            cu_q_repr, cu_k_repr, max_seqlen_q, max_seqlen_k, bt_repr);
        assert(kv2[j] == kv_cache_reprs[j]);
        lemma_layer_chain_kv_reprs_frame(config, layers, next.0, next.1, positions_repr,
            kv2, slot_repr, cu_q_repr, cu_k_repr,
            max_seqlen_q, max_seqlen_k, bt_repr, (start + 1) as nat, j);
    }
}

// `chain_layer_kv_rows` reads the cache only at indices `[start, layer)`:
// caches agreeing there produce the same rows.
pub proof fn lemma_chain_layer_kv_rows_agree(
    config: DenseSwiGluForwardConfigRepr,
    layers: Seq<LayerWeightsRepr>,
    hidden_repr: Tensor2D,
    residual_repr: Tensor2D,
    positions_repr: IntTensor1D,
    kv_a: Seq<(KVCacheLayerRepr, KVCacheLayerRepr)>,
    kv_b: Seq<(KVCacheLayerRepr, KVCacheLayerRepr)>,
    slot_repr: Seq<int>,
    cu_q_repr: Seq<int>,
    cu_k_repr: Seq<int>,
    max_seqlen_q: nat,
    max_seqlen_k: nat,
    bt_repr: Seq<Seq<BlockId>>,
    start: nat,
    layer: nat,
)
    requires
        start <= layer < layers.len(),
        kv_a.len() >= layers.len(),
        kv_b.len() >= layers.len(),
        forall|idx: int| start as int <= idx < layer as int ==> #[trigger] kv_a[idx] == kv_b[idx],
    ensures
        chain_layer_kv_rows(config, layers, hidden_repr, residual_repr, positions_repr,
            kv_a, slot_repr, cu_q_repr, cu_k_repr,
            max_seqlen_q, max_seqlen_k, bt_repr, start, layer)
        == chain_layer_kv_rows(config, layers, hidden_repr, residual_repr, positions_repr,
            kv_b, slot_repr, cu_q_repr, cu_k_repr,
            max_seqlen_q, max_seqlen_k, bt_repr, start, layer),
    decreases layer - start,
{
    if layer <= start {
    } else {
        assert(kv_a[start as int] == kv_b[start as int]);
        let next = BD::decoder_layer_output_repr(config, layers[start as int],
            hidden_repr, residual_repr, positions_repr,
            kv_a[start as int].0, kv_a[start as int].1, slot_repr,
            cu_q_repr, cu_k_repr, max_seqlen_q, max_seqlen_k, bt_repr);
        lemma_chain_layer_kv_rows_agree(config, layers, next.0, next.1, positions_repr,
            kv_a, kv_b, slot_repr, cu_q_repr, cu_k_repr,
            max_seqlen_q, max_seqlen_k, bt_repr, (start + 1) as nat, layer);
    }
}

// Chain-level store characterization: the fold's result at `layer` is one
// scatter store of `chain_layer_kv_rows` over the pre-chain cache at `layer`.
pub proof fn lemma_chain_layer_store(
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
    layer: nat,
)
    requires
        start <= layer < layers.len(),
        kv_cache_reprs.len() >= layers.len(),
    ensures ({
        let rows = chain_layer_kv_rows(config, layers, hidden_repr, residual_repr,
            positions_repr, kv_cache_reprs, slot_repr, cu_q_repr, cu_k_repr,
            max_seqlen_q, max_seqlen_k, bt_repr, start, layer);
        CC::layer_chain_kv_reprs(config, layers, hidden_repr, residual_repr, positions_repr,
            kv_cache_reprs, slot_repr, cu_q_repr, cu_k_repr,
            max_seqlen_q, max_seqlen_k, bt_repr, start)[layer as int]
            == RT::store_kv_cache_repr(rows.0, rows.1,
                kv_cache_reprs[layer as int].0, kv_cache_reprs[layer as int].1, slot_repr)
    }),
    decreases layer - start,
{
    let wr = layers[start as int];
    let updated = CC::decoder_layer_kv_update_repr(config, wr,
        hidden_repr, residual_repr, positions_repr,
        kv_cache_reprs[start as int].0, kv_cache_reprs[start as int].1, slot_repr);
    let kv2 = kv_cache_reprs.update(start as int, updated);
    let next = BD::decoder_layer_output_repr(config, wr,
        hidden_repr, residual_repr, positions_repr,
        kv_cache_reprs[start as int].0, kv_cache_reprs[start as int].1, slot_repr,
        cu_q_repr, cu_k_repr, max_seqlen_q, max_seqlen_k, bt_repr);
    if layer == start {
        // The fold sets index `start` to `updated` and never touches it again.
        lemma_layer_chain_kv_reprs_frame(config, layers, next.0, next.1, positions_repr,
            kv2, slot_repr, cu_q_repr, cu_k_repr,
            max_seqlen_q, max_seqlen_k, bt_repr, (start + 1) as nat, start as int);
        assert(kv2[start as int] == updated);
        // `decoder_layer_kv_update_repr` IS the store of the base-case rows.
        let normed = BD::add_rms_norm_repr(config, hidden_repr, residual_repr, wr.input_norm).0;
        assert(updated == RT::store_kv_cache_repr(
            BD::pre_attention_repr(config, wr, normed, positions_repr).1,
            RT::view_as_kv_repr(RT::qkv_linear_repr(
                normed, wr.q_proj, wr.k_proj, wr.v_proj,
            ).2),
            kv_cache_reprs[start as int].0, kv_cache_reprs[start as int].1, slot_repr));
    } else {
        // Recurse from start+1 over kv2, then transfer back to kv (they agree
        // at `layer` and on `[start+1, layer)`).
        lemma_chain_layer_store(config, layers, next.0, next.1, positions_repr,
            kv2, slot_repr, cu_q_repr, cu_k_repr,
            max_seqlen_q, max_seqlen_k, bt_repr, (start + 1) as nat, layer);
        assert(kv2[layer as int] == kv_cache_reprs[layer as int]);
        lemma_chain_layer_kv_rows_agree(config, layers, next.0, next.1, positions_repr,
            kv2, kv_cache_reprs, slot_repr, cu_q_repr, cu_k_repr,
            max_seqlen_q, max_seqlen_k, bt_repr, (start + 1) as nat, layer);
    }
}

// MODEL-LEVEL store characterization: the post-step cache at `layer` is one
// scatter store of `model_layer_kv_rows(config, layer)` over the PRE-step cache at
// `layer`, with the plan's slot vector.
pub proof fn lemma_model_forward_layer_store(
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
    layer: nat,
)
    requires
        layer < wr.layers.len(),
        pre_kv_reprs.len() >= wr.layers.len(),
    ensures ({
        let rows = model_layer_kv_rows(config, wr, input_ids_repr, positions_repr,
            pre_kv_reprs, slot_repr, cu_q_repr, cu_k_repr,
            max_seqlen_q, max_seqlen_k, bt_repr, layer);
        CC::model_forward_kv_reprs(config, wr, input_ids_repr, positions_repr, pre_kv_reprs,
            slot_repr, cu_q_repr, cu_k_repr, max_seqlen_q, max_seqlen_k, bt_repr)
            [layer as int]
            == RT::store_kv_cache_repr(rows.0, rows.1,
                pre_kv_reprs[layer as int].0, pre_kv_reprs[layer as int].1, slot_repr)
    }),
{
    let layers = wr.layers;
    let embed = RT::embed_repr(input_ids_repr, wr.embed_weight);
    let wr0 = layers[0];
    let first_update = CC::first_decoder_layer_kv_update_repr(config, wr0,
        embed, positions_repr, pre_kv_reprs[0].0, pre_kv_reprs[0].1, slot_repr);
    let kv1 = pre_kv_reprs.update(0, first_update);
    let first_out = BD::first_decoder_layer_output_repr(config, wr0,
        embed, positions_repr, pre_kv_reprs[0].0, pre_kv_reprs[0].1, slot_repr,
        cu_q_repr, cu_k_repr, max_seqlen_q, max_seqlen_k, bt_repr);
    if layer == 0 {
        // Chain from 1 leaves index 0 at `first_update`, which IS the store of
        // the layer-0 rows.
        lemma_layer_chain_kv_reprs_frame(config, layers, first_out.0, first_out.1,
            positions_repr, kv1, slot_repr, cu_q_repr, cu_k_repr,
            max_seqlen_q, max_seqlen_k, bt_repr, 1, 0);
        assert(kv1[0] == first_update);
        let normed = BD::rms_norm_repr(config, embed, wr0.input_norm);
        assert(first_update == RT::store_kv_cache_repr(
            BD::pre_attention_repr(config, wr0, normed, positions_repr).1,
            RT::view_as_kv_repr(RT::qkv_linear_repr(
                normed, wr0.q_proj, wr0.k_proj, wr0.v_proj,
            ).2),
            pre_kv_reprs[0].0, pre_kv_reprs[0].1, slot_repr));
    } else {
        lemma_chain_layer_store(config, layers, first_out.0, first_out.1, positions_repr,
            kv1, slot_repr, cu_q_repr, cu_k_repr,
            max_seqlen_q, max_seqlen_k, bt_repr, 1, layer);
        assert(kv1[layer as int] == pre_kv_reprs[layer as int]);
        lemma_chain_layer_kv_rows_agree(config, layers, first_out.0, first_out.1,
            positions_repr, kv1, pre_kv_reprs, slot_repr, cu_q_repr, cu_k_repr,
            max_seqlen_q, max_seqlen_k, bt_repr, 1, layer);
    }
}

// Row-count shape: the chain rows have one K row and one V row per input
// token (needed to split them along the cu_q partition).
pub proof fn lemma_chain_layer_kv_rows_shape(
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
    layer: nat,
)
    requires
        start <= layer < layers.len(),
        kv_cache_reprs.len() >= layers.len(),
        hidden_repr.len() == residual_repr.len(),
        hidden_repr.len() == positions_repr.len(),
        slot_repr.len() == hidden_repr.len(),
    ensures ({
        let rows = chain_layer_kv_rows(config, layers, hidden_repr, residual_repr,
            positions_repr, kv_cache_reprs, slot_repr, cu_q_repr, cu_k_repr,
            max_seqlen_q, max_seqlen_k, bt_repr, start, layer);
        rows.0.len() == hidden_repr.len() && rows.1.len() == hidden_repr.len()
    }),
    decreases layer - start,
{
    broadcast use {
        RT::lemma_add_rms_norm_repr_shape,
        RT::lemma_linear_repr_shape,
        RT::lemma_view_as_kv_repr_shape,
    };
    if layer <= start {
        let wr = layers[start as int];
        let normed = BD::add_rms_norm_repr(config, hidden_repr, residual_repr, wr.input_norm).0;
        BD::lemma_pre_attention_repr_shape(
            config, wr, normed, positions_repr,
        );
        assert(normed.len() == hidden_repr.len());
    } else {
        let next = BD::decoder_layer_output_repr(config, layers[start as int],
            hidden_repr, residual_repr, positions_repr,
            kv_cache_reprs[start as int].0, kv_cache_reprs[start as int].1, slot_repr,
            cu_q_repr, cu_k_repr, max_seqlen_q, max_seqlen_k, bt_repr);
        BD::lemma_decoder_layer_output_repr_shape(
            config, layers[start as int],
            hidden_repr, residual_repr, positions_repr,
            kv_cache_reprs[start as int].0, kv_cache_reprs[start as int].1,
            slot_repr, cu_q_repr, cu_k_repr, max_seqlen_q, max_seqlen_k, bt_repr,
        );
        assert(next.0.len() == hidden_repr.len());
        assert(next.1.len() == hidden_repr.len());
        lemma_chain_layer_kv_rows_shape(config, layers, next.0, next.1, positions_repr,
            kv_cache_reprs, slot_repr, cu_q_repr, cu_k_repr,
            max_seqlen_q, max_seqlen_k, bt_repr, (start + 1) as nat, layer);
    }
}

pub proof fn lemma_model_layer_kv_rows_shape(
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
    layer: nat,
)
    requires
        layer < wr.layers.len(),
        pre_kv_reprs.len() >= wr.layers.len(),
        input_ids_repr.len() == positions_repr.len(),
        slot_repr.len() == input_ids_repr.len(),
    ensures ({
        let rows = model_layer_kv_rows(config, wr, input_ids_repr, positions_repr,
            pre_kv_reprs, slot_repr, cu_q_repr, cu_k_repr,
            max_seqlen_q, max_seqlen_k, bt_repr, layer);
        rows.0.len() == input_ids_repr.len() && rows.1.len() == input_ids_repr.len()
    }),
{
    broadcast use {
        RT::lemma_embed_repr_shape,
        RT::lemma_rms_norm_repr_shape,
        RT::lemma_linear_repr_shape,
        RT::lemma_view_as_kv_repr_shape,
    };
    let embed = RT::embed_repr(input_ids_repr, wr.embed_weight);
    assert(embed.len() == input_ids_repr.len());
    if layer == 0 {
        let normed = BD::rms_norm_repr(config, embed, wr.layers[0].input_norm);
        BD::lemma_pre_attention_repr_shape(
            config, wr.layers[0], normed,
            positions_repr,
        );
        assert(normed.len() == input_ids_repr.len());
    } else {
        let first_out = BD::first_decoder_layer_output_repr(config, wr.layers[0],
            embed, positions_repr, pre_kv_reprs[0].0, pre_kv_reprs[0].1, slot_repr,
            cu_q_repr, cu_k_repr, max_seqlen_q, max_seqlen_k, bt_repr);
        BD::lemma_first_decoder_layer_output_repr_shape(
            config, wr.layers[0], embed,
            positions_repr, pre_kv_reprs[0].0, pre_kv_reprs[0].1, slot_repr,
            cu_q_repr, cu_k_repr, max_seqlen_q, max_seqlen_k, bt_repr,
        );
        assert(first_out.0.len() == input_ids_repr.len());
        lemma_chain_layer_kv_rows_shape(config, wr.layers, first_out.0, first_out.1,
            positions_repr, pre_kv_reprs, slot_repr, cu_q_repr, cu_k_repr,
            max_seqlen_q, max_seqlen_k, bt_repr, 1, layer);
    }
}

} // verus!
