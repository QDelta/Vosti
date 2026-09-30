//! Family-neutral request projection and cache-relocation proofs for the
//! full-attention dense SwiGLU decoder composition.

use crate::boundary::dense_layer_primitives as DLP;
#[cfg(verus_only)]
use crate::{types::{BLOCK_SIZE_SPEC}, proof::tensor::geometry::{block_table_slot, blocks_needed_for, cache_at, positions_from, slot_in_cache}};
#[cfg(verus_only)]
use crate::proof::model::dense_swiglu::batch_invariance as BI;
#[cfg(verus_only)]
use crate::proof::model::layer_properties as DENSE_PROPS;
use crate::proof::model::dense_swiglu::semantics as DS;
use crate::{types::*, proof::model::types::*, proof::tensor::types::*};
#[cfg(verus_only)]
use crate::boundary::tensor_runtime as RT;
use vstd::prelude::*;

verus! {

// Lift the pre-store launch domain exported by the engine to the post-store
// cache consumed by attention.  Store preserves page geometry, while the
// projected K/V rows have exactly one row per query/slot.
pub proof fn lemma_decoder_core_attention_launch_ready_from_pre_store(
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
)
    requires
        normed_repr.len() == positions_repr.len(),
        slot_repr.len() == normed_repr.len(),
        RT::paged_attention_launch_ready(
            normed_repr.len(), k_cache_repr, v_cache_repr,
            cu_q_repr, cu_k_repr, max_seqlen_q, max_seqlen_k, bt_repr,
        ),
    ensures
        DS::decoder_core_attention_launch_ready(config,
            wr, normed_repr, positions_repr, k_cache_repr, v_cache_repr,
            slot_repr, cu_q_repr, cu_k_repr, max_seqlen_q, max_seqlen_k,
            bt_repr,
        ),
{
    let pre = DS::pre_attention_repr(config, wr, normed_repr, positions_repr);
    let vr = RT::qkv_linear_repr(
        normed_repr, wr.q_proj, wr.k_proj, wr.v_proj,
    ).2;
    let vvr = RT::view_as_kv_repr(vr);
    DS::lemma_pre_attention_repr_shape(config, wr, normed_repr, positions_repr);
    RT::lemma_qkv_linear_repr_shape(
        normed_repr, wr.q_proj, wr.k_proj, wr.v_proj,
    );
    RT::lemma_view_as_kv_repr_shape(vr);
    RT::lemma_paged_attention_launch_ready_after_store(
        normed_repr.len(), pre.1, vvr, k_cache_repr, v_cache_repr,
        slot_repr, cu_q_repr, cu_k_repr, max_seqlen_q, max_seqlen_k,
        bt_repr,
    );
    reveal(DS::decoder_core_attention_launch_ready);
}

pub proof fn lemma_decoder_layer_attention_launch_ready_from_pre_store(
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
        RT::paged_attention_launch_ready(
            hidden_repr.len(), k_cache_repr, v_cache_repr,
            cu_q_repr, cu_k_repr, max_seqlen_q, max_seqlen_k, bt_repr,
        ),
    ensures
        DS::decoder_layer_attention_launch_ready(config,
            wr, hidden_repr, residual_repr, positions_repr,
            k_cache_repr, v_cache_repr, slot_repr,
            cu_q_repr, cu_k_repr, max_seqlen_q, max_seqlen_k, bt_repr,
        ),
{
    let pre = DS::add_rms_norm_repr(config, hidden_repr, residual_repr, wr.input_norm);
    RT::lemma_add_rms_norm_repr_len(
        hidden_repr, residual_repr, wr.input_norm,
        config.rms_norm_epsilon,
    );
    lemma_decoder_core_attention_launch_ready_from_pre_store(config,
        wr, pre.0, positions_repr, k_cache_repr, v_cache_repr, slot_repr,
        cu_q_repr, cu_k_repr, max_seqlen_q, max_seqlen_k, bt_repr,
    );
    reveal(DS::decoder_layer_attention_launch_ready);
}

pub proof fn lemma_first_decoder_layer_attention_launch_ready_from_pre_store(
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
        RT::paged_attention_launch_ready(
            hidden_repr.len(), k_cache_repr, v_cache_repr,
            cu_q_repr, cu_k_repr, max_seqlen_q, max_seqlen_k, bt_repr,
        ),
    ensures
        DS::first_decoder_layer_attention_launch_ready(config,
            wr, hidden_repr, positions_repr, k_cache_repr, v_cache_repr,
            slot_repr, cu_q_repr, cu_k_repr, max_seqlen_q, max_seqlen_k,
            bt_repr,
        ),
{
    let normed = DS::rms_norm_repr(config, hidden_repr, wr.input_norm);
    RT::lemma_rms_norm_repr_len(
        hidden_repr, wr.input_norm, config.rms_norm_epsilon,
    );
    lemma_decoder_core_attention_launch_ready_from_pre_store(config,
        wr, normed, positions_repr, k_cache_repr, v_cache_repr, slot_repr,
        cu_q_repr, cu_k_repr, max_seqlen_q, max_seqlen_k, bt_repr,
    );
    reveal(DS::first_decoder_layer_attention_launch_ready);
}

// DecoderCoreOutputRepr: (next_hidden, next_residual) from a full decoder
// layer (after input norm has already been applied; that's what `normed`
// is).  Composes pre-attn → store_kv → paged_attn → post-attn.
pub proof fn pre_attention_subrange_invariance(
    config: DenseSwiGluForwardConfigRepr,
    wr: LayerWeightsRepr,
    normed_repr: Tensor2D,
    positions_repr: IntTensor1D,
    a: int,
    b: int,
)
    requires
        normed_repr.len() == positions_repr.len(),
        0 <= a <= b <= normed_repr.len(),
    ensures
        DS::pre_attention_repr(config, wr, normed_repr.subrange(a, b), positions_repr.subrange(a, b)).0
            == DS::pre_attention_repr(config, wr, normed_repr, positions_repr).0.subrange(a, b),
        DS::pre_attention_repr(config, wr, normed_repr.subrange(a, b), positions_repr.subrange(a, b)).1
            == DS::pre_attention_repr(config, wr, normed_repr, positions_repr).1.subrange(a, b),
{
    broadcast use {
        RT::lemma_linear_repr_shape,
        RT::lemma_apply_qk_norm_repr_shape,
    };
    let qkv = RT::qkv_linear_repr(
        normed_repr, wr.q_proj, wr.k_proj, wr.v_proj,
    );
    let qr = qkv.0;
    let kr = qkv.1;
    let nqk = DS::apply_qk_norm_repr(config, qr, kr, wr.qk_norm);
    RT::lemma_qkv_linear_repr_shape(
        normed_repr, wr.q_proj, wr.k_proj, wr.v_proj,
    );
    DENSE_PROPS::qkv_linear_subrange_invariance(
        normed_repr, wr.q_proj, wr.k_proj, wr.v_proj, a, b,
    );
    BI::apply_qk_norm_subrange_invariance(
        qr, kr, wr.qk_norm, config.rms_norm_epsilon, a, b,
    );
    BI::rotary_embed_subrange_invariance(
        DLP::layer_attention_geometry_repr(wr),
        config.rotary,
        positions_repr, nqk.0, nqk.1, a, b,
    );
}

pub proof fn post_attention_subrange_invariance(
    config: DenseSwiGluForwardConfigRepr,
    wr: LayerWeightsRepr,
    attn_repr: Tensor2D,
    residual_repr: Tensor2D,
    a: int,
    b: int,
)
    requires
        attn_repr.len() == residual_repr.len(),
        0 <= a <= b <= attn_repr.len(),
    ensures
        DS::post_attention_repr(config, wr, attn_repr.subrange(a, b), residual_repr.subrange(a, b)).0
            == DS::post_attention_repr(config, wr, attn_repr, residual_repr).0.subrange(a, b),
        DS::post_attention_repr(config, wr, attn_repr.subrange(a, b), residual_repr.subrange(a, b)).1
            == DS::post_attention_repr(config, wr, attn_repr, residual_repr).1.subrange(a, b),
{
    broadcast use {
        RT::lemma_linear_repr_shape,
        RT::lemma_add_rms_norm_repr_shape,
        RT::lemma_silu_and_mul_repr_shape,
    };
    let projected = RT::linear_repr(attn_repr, wr.o_proj);
    let post = DS::add_rms_norm_repr(config, projected, residual_repr, wr.post_attn_norm);
    let gate_up = RT::linear_repr(post.0, wr.gate_up_proj);
    let mlp = RT::silu_and_mul_repr(gate_up);
    BI::linear_subrange_invariance(attn_repr, wr.o_proj, a, b);
    BI::add_rms_norm_subrange_invariance(
        projected, residual_repr, wr.post_attn_norm,
        config.rms_norm_epsilon, a, b,
    );
    BI::linear_subrange_invariance(post.0, wr.gate_up_proj, a, b);
    RT::lemma_linear_repr_shape(post.0, wr.gate_up_proj);
    BI::silu_and_mul_subrange_invariance(gate_up, a, b);
    BI::linear_subrange_invariance(mlp, wr.down_proj, a, b);
}

// ---------------------------------------------------------------------------
// Per-request decomposition of a full decoder layer.
//
// The decoder layer's output for request `i` (rows [cu_q[i], cu_q[i+1])) is
// exactly `post_attention` applied to request `i`'s paged-attention output and
// request `i`'s residual rows.  Request `i`'s attention reads the shared cache
// only through its own block-table row `bt[i]`.
//
// This composes `post_attention_subrange_invariance` + `paged_attention_batch_invariance`;
// it needs NO cache hypothesis.  It isolates the remaining cross-request
// dependency to "the cache contents read via bt[i]" — full batch invariance
// (independence from other requests) then follows once KV-cache agreement at
// those positions is established (the KV-equivalence pillar, discharged by the
// engine/refinement layer via `paged_attention_physical_relocation`).
// ---------------------------------------------------------------------------
pub proof fn decoder_core_request_decomposition(
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
    i: nat,
)
    requires
        normed_repr.len() == residual_repr.len(),
        normed_repr.len() == positions_repr.len(),
        DS::decoder_core_attention_launch_ready(config,
            wr, normed_repr, positions_repr,
            k_cache_repr, v_cache_repr, slot_repr,
            cu_q_repr, cu_k_repr, max_seqlen_q, max_seqlen_k, bt_repr,
        ),
        cu_q_repr.len() == cu_k_repr.len(),
        cu_k_repr.len() == bt_repr.len() + 1,
        i < bt_repr.len(),
        cu_q_repr[0] == 0,
        cu_k_repr[0] == 0,
        cu_q_repr[bt_repr.len() as int] == normed_repr.len() as int,
        forall|j: int| 0 <= j < bt_repr.len() as int ==>
            cu_q_repr[j] < #[trigger] cu_q_repr[j + 1],
        forall|j: int| 0 <= j < bt_repr.len() as int ==>
            cu_k_repr[j] < #[trigger] cu_k_repr[j + 1],
        0 <= cu_q_repr[i as int] < cu_q_repr[i as int + 1] <= normed_repr.len() as int,
        0 <= cu_k_repr[i as int] < cu_k_repr[i as int + 1],
        max_seqlen_q > 0,
        cu_q_repr[i as int + 1] - cu_q_repr[i as int] <= max_seqlen_q as int,
        cu_k_repr[i as int + 1] - cu_k_repr[i as int] <= max_seqlen_k as int,
        blocks_needed_for((cu_k_repr[i as int + 1] - cu_k_repr[i as int]) as nat)
            <= bt_repr[i as int].len(),
    ensures
        DS::decoder_core_output_repr(config, wr, normed_repr, residual_repr, positions_repr,
            k_cache_repr, v_cache_repr, slot_repr, cu_q_repr, cu_k_repr,
            max_seqlen_q, max_seqlen_k, bt_repr).0
            .subrange(cu_q_repr[i as int], cu_q_repr[i as int + 1])
        == DS::post_attention_repr(config, wr,
            RT::paged_attention_repr(
                DS::pre_attention_repr(config, wr, normed_repr, positions_repr).0
                    .subrange(cu_q_repr[i as int], cu_q_repr[i as int + 1]),
                DS::layer_kv_update_repr(config, wr, normed_repr, positions_repr,
                    k_cache_repr, v_cache_repr, slot_repr).0,
                DS::layer_kv_update_repr(config, wr, normed_repr, positions_repr,
                    k_cache_repr, v_cache_repr, slot_repr).1,
                seq![0int, cu_q_repr[i as int + 1] - cu_q_repr[i as int]],
                seq![0int, cu_k_repr[i as int + 1] - cu_k_repr[i as int]],
                (cu_q_repr[i as int + 1] - cu_q_repr[i as int]) as nat,
                (cu_k_repr[i as int + 1] - cu_k_repr[i as int]) as nat,
                seq![bt_repr[i as int]],
                crate::boundary::dense_layer_primitives::layer_attention_parameters_repr(wr, AttentionScaleRepr::InverseSqrtHeadDim),
                ),
            residual_repr.subrange(cu_q_repr[i as int], cu_q_repr[i as int + 1])).0,
        DS::decoder_core_output_repr(config, wr, normed_repr, residual_repr, positions_repr,
            k_cache_repr, v_cache_repr, slot_repr, cu_q_repr, cu_k_repr,
            max_seqlen_q, max_seqlen_k, bt_repr).1
            .subrange(cu_q_repr[i as int], cu_q_repr[i as int + 1])
        == DS::post_attention_repr(config, wr,
            RT::paged_attention_repr(
                DS::pre_attention_repr(config, wr, normed_repr, positions_repr).0
                    .subrange(cu_q_repr[i as int], cu_q_repr[i as int + 1]),
                DS::layer_kv_update_repr(config, wr, normed_repr, positions_repr,
                    k_cache_repr, v_cache_repr, slot_repr).0,
                DS::layer_kv_update_repr(config, wr, normed_repr, positions_repr,
                    k_cache_repr, v_cache_repr, slot_repr).1,
                seq![0int, cu_q_repr[i as int + 1] - cu_q_repr[i as int]],
                seq![0int, cu_k_repr[i as int + 1] - cu_k_repr[i as int]],
                (cu_q_repr[i as int + 1] - cu_q_repr[i as int]) as nat,
                (cu_k_repr[i as int + 1] - cu_k_repr[i as int]) as nat,
                seq![bt_repr[i as int]],
                crate::boundary::dense_layer_primitives::layer_attention_parameters_repr(wr, AttentionScaleRepr::InverseSqrtHeadDim),
                ),
            residual_repr.subrange(cu_q_repr[i as int], cu_q_repr[i as int + 1])).1,
{
    let pre = DS::pre_attention_repr(config, wr, normed_repr, positions_repr);
    let new_caches = DS::layer_kv_update_repr(config, wr, normed_repr, positions_repr,
        k_cache_repr, v_cache_repr, slot_repr);
    let attn = RT::paged_attention_repr(pre.0, new_caches.0, new_caches.1,
        cu_q_repr, cu_k_repr, max_seqlen_q, max_seqlen_k, bt_repr,
        crate::boundary::dense_layer_primitives::layer_attention_parameters_repr(wr, AttentionScaleRepr::InverseSqrtHeadDim),
        );

    DS::lemma_pre_attention_repr_shape(config, wr, normed_repr, positions_repr);
    RT::lemma_paged_attention_repr_shape(pre.0, new_caches.0, new_caches.1,
        cu_q_repr, cu_k_repr, max_seqlen_q, max_seqlen_k, bt_repr,
        crate::boundary::dense_layer_primitives::layer_attention_parameters_repr(wr, AttentionScaleRepr::InverseSqrtHeadDim),
        );
    // attn.len() == pre.0.len() == normed.len() == residual.len().

    // Slice the row-wise post-attention block down to request i's rows.
    post_attention_subrange_invariance(config, wr, attn, residual_repr,
        cu_q_repr[i as int], cu_q_repr[i as int + 1]);

    // Request i's attention slice is the singleton-batch run on request i.
    reveal(DS::decoder_core_attention_launch_ready);
    BI::paged_attention_batch_invariance(pre.0, new_caches.0, new_caches.1,
        cu_q_repr, cu_k_repr, max_seqlen_q, max_seqlen_k, bt_repr, i,
        crate::boundary::dense_layer_primitives::layer_attention_parameters_repr(wr, AttentionScaleRepr::InverseSqrtHeadDim),
        );
}

// ---------------------------------------------------------------------------
// Cross-request independence (the isolation property).
//
// Request `i`'s decoder-layer output is unchanged if the KV cache is replaced
// by ANY cache (`alt_*`) that agrees with the real_kv post-store cache at the
// positions request `i` reads through `bt[i]`.  Since other requests write only
// their own (disjoint) blocks, their writes never touch `bt[i]`'s positions —
// so request `i`'s output is independent of every other request's input.  The
// per-position cache agreement is the explicit hypothesis discharged by the
// engine/refinement layer (block-disjointness; the KV-equivalence pillar).
//
// Proof = `decoder_core_request_decomposition` (slice the layer to request i,
// over the real_kv cache) + `paged_attention_physical_relocation` (swap to the
// agreeing cache).
// ---------------------------------------------------------------------------
pub proof fn decoder_core_cache_independence(
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
    alt_k_cache: KVCacheLayerRepr,
    alt_v_cache: KVCacheLayerRepr,
    i: nat,
)
    requires
        RT::paged_attention_numeric_domain(),
        normed_repr.len() == residual_repr.len(),
        normed_repr.len() == positions_repr.len(),
        DS::decoder_core_attention_launch_ready(config,
            wr, normed_repr, positions_repr,
            k_cache_repr, v_cache_repr, slot_repr,
            cu_q_repr, cu_k_repr, max_seqlen_q, max_seqlen_k, bt_repr,
        ),
        cu_q_repr.len() == cu_k_repr.len(),
        cu_k_repr.len() == bt_repr.len() + 1,
        i < bt_repr.len(),
        cu_q_repr[0] == 0,
        cu_k_repr[0] == 0,
        cu_q_repr[bt_repr.len() as int] == normed_repr.len() as int,
        forall|j: int| 0 <= j < bt_repr.len() as int ==>
            cu_q_repr[j] < #[trigger] cu_q_repr[j + 1],
        forall|j: int| 0 <= j < bt_repr.len() as int ==>
            cu_k_repr[j] < #[trigger] cu_k_repr[j + 1],
        0 <= cu_q_repr[i as int] < cu_q_repr[i as int + 1] <= normed_repr.len() as int,
        0 <= cu_k_repr[i as int] < cu_k_repr[i as int + 1],
        max_seqlen_q > 0,
        cu_q_repr[i as int + 1] - cu_q_repr[i as int] <= max_seqlen_q as int,
        cu_k_repr[i as int + 1] - cu_k_repr[i as int] <= max_seqlen_k as int,
        // request i has at least as many keys as queries (causal / decode),
        cu_q_repr[i as int + 1] - cu_q_repr[i as int]
            <= cu_k_repr[i as int + 1] - cu_k_repr[i as int],
        blocks_needed_for((cu_k_repr[i as int + 1] - cu_k_repr[i as int]) as nat)
            <= bt_repr[i as int].len(),
        // The agreeing cache matches the real_kv post-store cache at request i's
        // read positions (the block-disjointness / KV-equivalence hypothesis).
        forall|pos: nat| #![trigger block_table_slot(bt_repr[i as int], pos)]
            pos < (cu_k_repr[i as int + 1] - cu_k_repr[i as int]) as nat ==> {
                let real_k = DS::layer_kv_update_repr(config, wr, normed_repr, positions_repr,
                    k_cache_repr, v_cache_repr, slot_repr).0;
                let real_v = DS::layer_kv_update_repr(config, wr, normed_repr, positions_repr,
                    k_cache_repr, v_cache_repr, slot_repr).1;
                let bt_row = bt_repr[i as int];
                (pos as int) / (BLOCK_SIZE_SPEC as int) < bt_row.len()
                && slot_in_cache(real_k, block_table_slot(bt_row, pos))
                && slot_in_cache(real_v, block_table_slot(bt_row, pos))
                && slot_in_cache(alt_k_cache, block_table_slot(bt_row, pos))
                && slot_in_cache(alt_v_cache, block_table_slot(bt_row, pos))
                && cache_at(real_k, block_table_slot(bt_row, pos))
                    == cache_at(alt_k_cache, block_table_slot(bt_row, pos))
                && cache_at(real_v, block_table_slot(bt_row, pos))
                    == cache_at(alt_v_cache, block_table_slot(bt_row, pos))
            },
    ensures
        DS::decoder_core_output_repr(config, wr, normed_repr, residual_repr, positions_repr,
            k_cache_repr, v_cache_repr, slot_repr, cu_q_repr, cu_k_repr,
            max_seqlen_q, max_seqlen_k, bt_repr).0
            .subrange(cu_q_repr[i as int], cu_q_repr[i as int + 1])
        == DS::post_attention_repr(config, wr,
            RT::paged_attention_repr(
                DS::pre_attention_repr(config, wr, normed_repr, positions_repr).0
                    .subrange(cu_q_repr[i as int], cu_q_repr[i as int + 1]),
                alt_k_cache, alt_v_cache,
                seq![0int, cu_q_repr[i as int + 1] - cu_q_repr[i as int]],
                seq![0int, cu_k_repr[i as int + 1] - cu_k_repr[i as int]],
                (cu_q_repr[i as int + 1] - cu_q_repr[i as int]) as nat,
                (cu_k_repr[i as int + 1] - cu_k_repr[i as int]) as nat,
                seq![bt_repr[i as int]],
                crate::boundary::dense_layer_primitives::layer_attention_parameters_repr(wr, AttentionScaleRepr::InverseSqrtHeadDim),
                ),
            residual_repr.subrange(cu_q_repr[i as int], cu_q_repr[i as int + 1])).0,
        DS::decoder_core_output_repr(config, wr, normed_repr, residual_repr, positions_repr,
            k_cache_repr, v_cache_repr, slot_repr, cu_q_repr, cu_k_repr,
            max_seqlen_q, max_seqlen_k, bt_repr).1
            .subrange(cu_q_repr[i as int], cu_q_repr[i as int + 1])
        == DS::post_attention_repr(config, wr,
            RT::paged_attention_repr(
                DS::pre_attention_repr(config, wr, normed_repr, positions_repr).0
                    .subrange(cu_q_repr[i as int], cu_q_repr[i as int + 1]),
                alt_k_cache, alt_v_cache,
                seq![0int, cu_q_repr[i as int + 1] - cu_q_repr[i as int]],
                seq![0int, cu_k_repr[i as int + 1] - cu_k_repr[i as int]],
                (cu_q_repr[i as int + 1] - cu_q_repr[i as int]) as nat,
                (cu_k_repr[i as int + 1] - cu_k_repr[i as int]) as nat,
                seq![bt_repr[i as int]],
                crate::boundary::dense_layer_primitives::layer_attention_parameters_repr(wr, AttentionScaleRepr::InverseSqrtHeadDim),
                ),
            residual_repr.subrange(cu_q_repr[i as int], cu_q_repr[i as int + 1])).1,
{
    let pre = DS::pre_attention_repr(config, wr, normed_repr, positions_repr);
    let new_caches = DS::layer_kv_update_repr(config, wr, normed_repr, positions_repr,
        k_cache_repr, v_cache_repr, slot_repr);
    let lo = cu_q_repr[i as int];
    let hi = cu_q_repr[i as int + 1];
    let q_len = (hi - lo) as nat;
    let k_len = (cu_k_repr[i as int + 1] - cu_k_repr[i as int]) as nat;

    DS::lemma_pre_attention_repr_shape(config, wr, normed_repr, positions_repr);
    assert(pre.0.subrange(lo, hi).len() == q_len);

    // Slice the layer to request i over the real_kv (post-store) cache.
    decoder_core_request_decomposition(config, wr, normed_repr, residual_repr, positions_repr,
        k_cache_repr, v_cache_repr, slot_repr, cu_q_repr, cu_k_repr,
        max_seqlen_q, max_seqlen_k, bt_repr, i);

    // Swap the real_kv cache for the agreeing cache: same paged-attention output.
    BI::paged_attention_physical_relocation(
        pre.0.subrange(lo, hi),
        new_caches.0, new_caches.1, alt_k_cache, alt_v_cache,
        q_len, k_len, bt_repr[i as int], bt_repr[i as int],
        crate::boundary::dense_layer_primitives::layer_attention_parameters_repr(wr, AttentionScaleRepr::InverseSqrtHeadDim),
        );
}


// Per-layer refinement unit: the batched decoder layer's output sliced to
// request `i` equals a SINGLE-REQUEST decoder-layer run on request `i`'s own
// inputs (its rows, positions, residual, its slot mapping `slot_i`, rebased
// cu_seqlens, and block-table row `bt[i]`) — i.e., exactly what a per-request
// machine computes.  `alt_*` is the single run's KV cache; the hypotheses are
// that it equals request `i`'s own store and agrees with the batched cache at
// `bt[i]`'s read positions (block-disjointness, dischargeable via
// `store_agrees_at_own_block`).  This is the statement the model-level
// refinement induction iterates over the layer chain.
pub proof fn decoder_core_request_isolation(
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
    alt_k_cache: KVCacheLayerRepr,
    alt_v_cache: KVCacheLayerRepr,
    slot_i: Seq<int>,
    i: nat,
)
    requires
        RT::paged_attention_numeric_domain(),
        normed_repr.len() == residual_repr.len(),
        normed_repr.len() == positions_repr.len(),
        DS::decoder_core_attention_launch_ready(config,
            wr, normed_repr, positions_repr,
            k_cache_repr, v_cache_repr, slot_repr,
            cu_q_repr, cu_k_repr, max_seqlen_q, max_seqlen_k, bt_repr,
        ),
        cu_q_repr.len() == cu_k_repr.len(),
        cu_k_repr.len() == bt_repr.len() + 1,
        i < bt_repr.len(),
        cu_q_repr[0] == 0,
        cu_k_repr[0] == 0,
        cu_q_repr[bt_repr.len() as int] == normed_repr.len() as int,
        forall|j: int| 0 <= j < bt_repr.len() as int ==>
            cu_q_repr[j] < #[trigger] cu_q_repr[j + 1],
        forall|j: int| 0 <= j < bt_repr.len() as int ==>
            cu_k_repr[j] < #[trigger] cu_k_repr[j + 1],
        0 <= cu_q_repr[i as int] < cu_q_repr[i as int + 1] <= normed_repr.len() as int,
        0 <= cu_k_repr[i as int] < cu_k_repr[i as int + 1],
        max_seqlen_q > 0,
        cu_q_repr[i as int + 1] - cu_q_repr[i as int] <= max_seqlen_q as int,
        cu_k_repr[i as int + 1] - cu_k_repr[i as int] <= max_seqlen_k as int,
        cu_q_repr[i as int + 1] - cu_q_repr[i as int]
            <= cu_k_repr[i as int + 1] - cu_k_repr[i as int],
        blocks_needed_for((cu_k_repr[i as int + 1] - cu_k_repr[i as int]) as nat)
            <= bt_repr[i as int].len(),
        // The single run's slot mapping matches its query length.
        slot_i.len() == cu_q_repr[i as int + 1] - cu_q_repr[i as int],
        // `alt_*` is exactly request i's own KV store.
        alt_k_cache == DS::layer_kv_update_repr(config, wr,
            normed_repr.subrange(cu_q_repr[i as int], cu_q_repr[i as int + 1]),
            positions_repr.subrange(cu_q_repr[i as int], cu_q_repr[i as int + 1]),
            k_cache_repr, v_cache_repr, slot_i).0,
        alt_v_cache == DS::layer_kv_update_repr(config, wr,
            normed_repr.subrange(cu_q_repr[i as int], cu_q_repr[i as int + 1]),
            positions_repr.subrange(cu_q_repr[i as int], cu_q_repr[i as int + 1]),
            k_cache_repr, v_cache_repr, slot_i).1,
        // request i's own store agrees with the batched store at its read slots.
        forall|pos: nat| #![trigger block_table_slot(bt_repr[i as int], pos)]
            pos < (cu_k_repr[i as int + 1] - cu_k_repr[i as int]) as nat ==> {
                let real_k = DS::layer_kv_update_repr(config, wr, normed_repr, positions_repr,
                    k_cache_repr, v_cache_repr, slot_repr).0;
                let real_v = DS::layer_kv_update_repr(config, wr, normed_repr, positions_repr,
                    k_cache_repr, v_cache_repr, slot_repr).1;
                let bt_row = bt_repr[i as int];
                (pos as int) / (BLOCK_SIZE_SPEC as int) < bt_row.len()
                && slot_in_cache(real_k, block_table_slot(bt_row, pos))
                && slot_in_cache(real_v, block_table_slot(bt_row, pos))
                && slot_in_cache(alt_k_cache, block_table_slot(bt_row, pos))
                && slot_in_cache(alt_v_cache, block_table_slot(bt_row, pos))
                && cache_at(real_k, block_table_slot(bt_row, pos))
                    == cache_at(alt_k_cache, block_table_slot(bt_row, pos))
                && cache_at(real_v, block_table_slot(bt_row, pos))
                    == cache_at(alt_v_cache, block_table_slot(bt_row, pos))
            },
    ensures
        DS::decoder_core_output_repr(config, wr, normed_repr, residual_repr, positions_repr,
            k_cache_repr, v_cache_repr, slot_repr, cu_q_repr, cu_k_repr,
            max_seqlen_q, max_seqlen_k, bt_repr).0
            .subrange(cu_q_repr[i as int], cu_q_repr[i as int + 1])
        == DS::decoder_core_output_repr(config, wr,
            normed_repr.subrange(cu_q_repr[i as int], cu_q_repr[i as int + 1]),
            residual_repr.subrange(cu_q_repr[i as int], cu_q_repr[i as int + 1]),
            positions_repr.subrange(cu_q_repr[i as int], cu_q_repr[i as int + 1]),
            k_cache_repr, v_cache_repr, slot_i,
            seq![0int, cu_q_repr[i as int + 1] - cu_q_repr[i as int]],
            seq![0int, cu_k_repr[i as int + 1] - cu_k_repr[i as int]],
            (cu_q_repr[i as int + 1] - cu_q_repr[i as int]) as nat,
            (cu_k_repr[i as int + 1] - cu_k_repr[i as int]) as nat,
            seq![bt_repr[i as int]]).0,
        DS::decoder_core_output_repr(config, wr, normed_repr, residual_repr, positions_repr,
            k_cache_repr, v_cache_repr, slot_repr, cu_q_repr, cu_k_repr,
            max_seqlen_q, max_seqlen_k, bt_repr).1
            .subrange(cu_q_repr[i as int], cu_q_repr[i as int + 1])
        == DS::decoder_core_output_repr(config, wr,
            normed_repr.subrange(cu_q_repr[i as int], cu_q_repr[i as int + 1]),
            residual_repr.subrange(cu_q_repr[i as int], cu_q_repr[i as int + 1]),
            positions_repr.subrange(cu_q_repr[i as int], cu_q_repr[i as int + 1]),
            k_cache_repr, v_cache_repr, slot_i,
            seq![0int, cu_q_repr[i as int + 1] - cu_q_repr[i as int]],
            seq![0int, cu_k_repr[i as int + 1] - cu_k_repr[i as int]],
            (cu_q_repr[i as int + 1] - cu_q_repr[i as int]) as nat,
            (cu_k_repr[i as int + 1] - cu_k_repr[i as int]) as nat,
            seq![bt_repr[i as int]]).1,
{
    let lo = cu_q_repr[i as int];
    let hi = cu_q_repr[i as int + 1];
    let qd = (hi - lo) as nat;
    let kd = (cu_k_repr[i as int + 1] - cu_k_repr[i as int]) as nat;
    let normed_i = normed_repr.subrange(lo, hi);
    let residual_i = residual_repr.subrange(lo, hi);
    let positions_i = positions_repr.subrange(lo, hi);
    let single = DS::decoder_core_output_repr(config, wr, normed_i, residual_i, positions_i,
        k_cache_repr, v_cache_repr, slot_i,
        seq![0int, qd as int], seq![0int, kd as int], qd, kd, seq![bt_repr[i as int]]);

    // Batched side: slice to request i, over the single run's (agreeing) cache.
    decoder_core_cache_independence(config, wr, normed_repr, residual_repr, positions_repr,
        k_cache_repr, v_cache_repr, slot_repr, cu_q_repr, cu_k_repr,
        max_seqlen_q, max_seqlen_k, bt_repr, alt_k_cache, alt_v_cache, i);

    // Unfold the singleton wrapper and tie its inputs to the selected slices.
    // No second ragged theorem application is needed for an already-singleton
    // launch.
    reveal(DS::decoder_core_output_repr);
    pre_attention_subrange_invariance(config, wr, normed_repr, positions_repr, lo, hi);
    DS::lemma_pre_attention_repr_shape(config, wr, normed_i, positions_i);
    DS::lemma_decoder_core_output_repr_shape(config, wr, normed_i, residual_i, positions_i,
        k_cache_repr, v_cache_repr, slot_i,
        seq![0int, qd as int], seq![0int, kd as int], qd, kd, seq![bt_repr[i as int]]);
    assert(single.0.subrange(0, qd as int) =~= single.0);
    assert(single.1.subrange(0, qd as int) =~= single.1);
    assert(DS::pre_attention_repr(config, wr, normed_i, positions_i).0.subrange(0, qd as int)
        =~= DS::pre_attention_repr(config, wr, normed_i, positions_i).0);
    assert(residual_i.subrange(0, qd as int) =~= residual_i);
}

// Per-layer refinement unit lifted to the `decoder_layer` wrapper (the unit the
// layer chain iterates).  Corollary of `decoder_core_request_isolation` with the
// normed input being `add_rms_norm(hidden, residual, input_norm).0`; the single
// run's normed/residual are identified with the batched ones' slices via
// `add_rms_norm_subrange_invariance`.
pub proof fn decoder_layer_request_isolation(
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
    alt_k_cache: KVCacheLayerRepr,
    alt_v_cache: KVCacheLayerRepr,
    slot_i: Seq<int>,
    i: nat,
)
    requires
        RT::paged_attention_numeric_domain(),
        hidden_repr.len() == residual_repr.len(),
        hidden_repr.len() == positions_repr.len(),
        DS::decoder_layer_attention_launch_ready(config,
            wr, hidden_repr, residual_repr, positions_repr,
            k_cache_repr, v_cache_repr, slot_repr,
            cu_q_repr, cu_k_repr, max_seqlen_q, max_seqlen_k, bt_repr,
        ),
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
        0 <= cu_q_repr[i as int] < cu_q_repr[i as int + 1] <= hidden_repr.len() as int,
        0 <= cu_k_repr[i as int] < cu_k_repr[i as int + 1],
        max_seqlen_q > 0,
        cu_q_repr[i as int + 1] - cu_q_repr[i as int] <= max_seqlen_q as int,
        cu_k_repr[i as int + 1] - cu_k_repr[i as int] <= max_seqlen_k as int,
        cu_q_repr[i as int + 1] - cu_q_repr[i as int]
            <= cu_k_repr[i as int + 1] - cu_k_repr[i as int],
        blocks_needed_for((cu_k_repr[i as int + 1] - cu_k_repr[i as int]) as nat)
            <= bt_repr[i as int].len(),
        slot_i.len() == cu_q_repr[i as int + 1] - cu_q_repr[i as int],
        // `alt_*` is request i's own KV store (from its own normed rows).
        alt_k_cache == DS::layer_kv_update_repr(config, wr,
            DS::add_rms_norm_repr(config,
                hidden_repr.subrange(cu_q_repr[i as int], cu_q_repr[i as int + 1]),
                residual_repr.subrange(cu_q_repr[i as int], cu_q_repr[i as int + 1]),
                wr.input_norm).0,
            positions_repr.subrange(cu_q_repr[i as int], cu_q_repr[i as int + 1]),
            k_cache_repr, v_cache_repr, slot_i).0,
        alt_v_cache == DS::layer_kv_update_repr(config, wr,
            DS::add_rms_norm_repr(config,
                hidden_repr.subrange(cu_q_repr[i as int], cu_q_repr[i as int + 1]),
                residual_repr.subrange(cu_q_repr[i as int], cu_q_repr[i as int + 1]),
                wr.input_norm).0,
            positions_repr.subrange(cu_q_repr[i as int], cu_q_repr[i as int + 1]),
            k_cache_repr, v_cache_repr, slot_i).1,
        // batched store agrees with request i's own store at its read slots.
        forall|pos: nat| #![trigger block_table_slot(bt_repr[i as int], pos)]
            pos < (cu_k_repr[i as int + 1] - cu_k_repr[i as int]) as nat ==> {
                let real_k = DS::layer_kv_update_repr(config, wr,
                    DS::add_rms_norm_repr(config, hidden_repr, residual_repr, wr.input_norm).0,
                    positions_repr, k_cache_repr, v_cache_repr, slot_repr).0;
                let real_v = DS::layer_kv_update_repr(config, wr,
                    DS::add_rms_norm_repr(config, hidden_repr, residual_repr, wr.input_norm).0,
                    positions_repr, k_cache_repr, v_cache_repr, slot_repr).1;
                let bt_row = bt_repr[i as int];
                (pos as int) / (BLOCK_SIZE_SPEC as int) < bt_row.len()
                && slot_in_cache(real_k, block_table_slot(bt_row, pos))
                && slot_in_cache(real_v, block_table_slot(bt_row, pos))
                && slot_in_cache(alt_k_cache, block_table_slot(bt_row, pos))
                && slot_in_cache(alt_v_cache, block_table_slot(bt_row, pos))
                && cache_at(real_k, block_table_slot(bt_row, pos))
                    == cache_at(alt_k_cache, block_table_slot(bt_row, pos))
                && cache_at(real_v, block_table_slot(bt_row, pos))
                    == cache_at(alt_v_cache, block_table_slot(bt_row, pos))
            },
    ensures
        DS::decoder_layer_output_repr(config, wr, hidden_repr, residual_repr, positions_repr,
            k_cache_repr, v_cache_repr, slot_repr, cu_q_repr, cu_k_repr,
            max_seqlen_q, max_seqlen_k, bt_repr).0
            .subrange(cu_q_repr[i as int], cu_q_repr[i as int + 1])
        == DS::decoder_layer_output_repr(config, wr,
            hidden_repr.subrange(cu_q_repr[i as int], cu_q_repr[i as int + 1]),
            residual_repr.subrange(cu_q_repr[i as int], cu_q_repr[i as int + 1]),
            positions_repr.subrange(cu_q_repr[i as int], cu_q_repr[i as int + 1]),
            k_cache_repr, v_cache_repr, slot_i,
            seq![0int, cu_q_repr[i as int + 1] - cu_q_repr[i as int]],
            seq![0int, cu_k_repr[i as int + 1] - cu_k_repr[i as int]],
            (cu_q_repr[i as int + 1] - cu_q_repr[i as int]) as nat,
            (cu_k_repr[i as int + 1] - cu_k_repr[i as int]) as nat,
            seq![bt_repr[i as int]]).0,
        DS::decoder_layer_output_repr(config, wr, hidden_repr, residual_repr, positions_repr,
            k_cache_repr, v_cache_repr, slot_repr, cu_q_repr, cu_k_repr,
            max_seqlen_q, max_seqlen_k, bt_repr).1
            .subrange(cu_q_repr[i as int], cu_q_repr[i as int + 1])
        == DS::decoder_layer_output_repr(config, wr,
            hidden_repr.subrange(cu_q_repr[i as int], cu_q_repr[i as int + 1]),
            residual_repr.subrange(cu_q_repr[i as int], cu_q_repr[i as int + 1]),
            positions_repr.subrange(cu_q_repr[i as int], cu_q_repr[i as int + 1]),
            k_cache_repr, v_cache_repr, slot_i,
            seq![0int, cu_q_repr[i as int + 1] - cu_q_repr[i as int]],
            seq![0int, cu_k_repr[i as int + 1] - cu_k_repr[i as int]],
            (cu_q_repr[i as int + 1] - cu_q_repr[i as int]) as nat,
            (cu_k_repr[i as int + 1] - cu_k_repr[i as int]) as nat,
            seq![bt_repr[i as int]]).1,
{
    broadcast use RT::lemma_add_rms_norm_repr_shape;
    let lo = cu_q_repr[i as int];
    let hi = cu_q_repr[i as int + 1];
    let pre = DS::add_rms_norm_repr(config, hidden_repr, residual_repr, wr.input_norm);
    // Identify the single run's normed/residual with the batched ones' slices.
    BI::add_rms_norm_subrange_invariance(
        hidden_repr, residual_repr, wr.input_norm,
        config.rms_norm_epsilon, lo, hi,
    );
    reveal(DS::decoder_layer_attention_launch_ready);
    decoder_core_request_isolation(config, wr, pre.0, pre.1, positions_repr,
        k_cache_repr, v_cache_repr, slot_repr, cu_q_repr, cu_k_repr,
        max_seqlen_q, max_seqlen_k, bt_repr, alt_k_cache, alt_v_cache, slot_i, i);
}

// First-layer refinement unit (input norm via rms_norm, residual == hidden).
// Corollary of `decoder_core_request_isolation` + `rms_norm_subrange_invariance`.
pub proof fn first_decoder_layer_request_isolation(
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
    alt_k_cache: KVCacheLayerRepr,
    alt_v_cache: KVCacheLayerRepr,
    slot_i: Seq<int>,
    i: nat,
)
    requires
        RT::paged_attention_numeric_domain(),
        hidden_repr.len() == positions_repr.len(),
        DS::first_decoder_layer_attention_launch_ready(config,
            wr, hidden_repr, positions_repr,
            k_cache_repr, v_cache_repr, slot_repr,
            cu_q_repr, cu_k_repr, max_seqlen_q, max_seqlen_k, bt_repr,
        ),
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
        0 <= cu_q_repr[i as int] < cu_q_repr[i as int + 1] <= hidden_repr.len() as int,
        0 <= cu_k_repr[i as int] < cu_k_repr[i as int + 1],
        max_seqlen_q > 0,
        cu_q_repr[i as int + 1] - cu_q_repr[i as int] <= max_seqlen_q as int,
        cu_k_repr[i as int + 1] - cu_k_repr[i as int] <= max_seqlen_k as int,
        cu_q_repr[i as int + 1] - cu_q_repr[i as int]
            <= cu_k_repr[i as int + 1] - cu_k_repr[i as int],
        blocks_needed_for((cu_k_repr[i as int + 1] - cu_k_repr[i as int]) as nat)
            <= bt_repr[i as int].len(),
        slot_i.len() == cu_q_repr[i as int + 1] - cu_q_repr[i as int],
        alt_k_cache == DS::layer_kv_update_repr(config, wr,
            DS::rms_norm_repr(config,
                hidden_repr.subrange(cu_q_repr[i as int], cu_q_repr[i as int + 1]),
                wr.input_norm),
            positions_repr.subrange(cu_q_repr[i as int], cu_q_repr[i as int + 1]),
            k_cache_repr, v_cache_repr, slot_i).0,
        alt_v_cache == DS::layer_kv_update_repr(config, wr,
            DS::rms_norm_repr(config,
                hidden_repr.subrange(cu_q_repr[i as int], cu_q_repr[i as int + 1]),
                wr.input_norm),
            positions_repr.subrange(cu_q_repr[i as int], cu_q_repr[i as int + 1]),
            k_cache_repr, v_cache_repr, slot_i).1,
        forall|pos: nat| #![trigger block_table_slot(bt_repr[i as int], pos)]
            pos < (cu_k_repr[i as int + 1] - cu_k_repr[i as int]) as nat ==> {
                let real_k = DS::layer_kv_update_repr(config, wr,
                    DS::rms_norm_repr(config, hidden_repr, wr.input_norm),
                    positions_repr, k_cache_repr, v_cache_repr, slot_repr).0;
                let real_v = DS::layer_kv_update_repr(config, wr,
                    DS::rms_norm_repr(config, hidden_repr, wr.input_norm),
                    positions_repr, k_cache_repr, v_cache_repr, slot_repr).1;
                let bt_row = bt_repr[i as int];
                (pos as int) / (BLOCK_SIZE_SPEC as int) < bt_row.len()
                && slot_in_cache(real_k, block_table_slot(bt_row, pos))
                && slot_in_cache(real_v, block_table_slot(bt_row, pos))
                && slot_in_cache(alt_k_cache, block_table_slot(bt_row, pos))
                && slot_in_cache(alt_v_cache, block_table_slot(bt_row, pos))
                && cache_at(real_k, block_table_slot(bt_row, pos))
                    == cache_at(alt_k_cache, block_table_slot(bt_row, pos))
                && cache_at(real_v, block_table_slot(bt_row, pos))
                    == cache_at(alt_v_cache, block_table_slot(bt_row, pos))
            },
    ensures
        DS::first_decoder_layer_output_repr(config, wr, hidden_repr, positions_repr,
            k_cache_repr, v_cache_repr, slot_repr, cu_q_repr, cu_k_repr,
            max_seqlen_q, max_seqlen_k, bt_repr).0
            .subrange(cu_q_repr[i as int], cu_q_repr[i as int + 1])
        == DS::first_decoder_layer_output_repr(config, wr,
            hidden_repr.subrange(cu_q_repr[i as int], cu_q_repr[i as int + 1]),
            positions_repr.subrange(cu_q_repr[i as int], cu_q_repr[i as int + 1]),
            k_cache_repr, v_cache_repr, slot_i,
            seq![0int, cu_q_repr[i as int + 1] - cu_q_repr[i as int]],
            seq![0int, cu_k_repr[i as int + 1] - cu_k_repr[i as int]],
            (cu_q_repr[i as int + 1] - cu_q_repr[i as int]) as nat,
            (cu_k_repr[i as int + 1] - cu_k_repr[i as int]) as nat,
            seq![bt_repr[i as int]]).0,
        DS::first_decoder_layer_output_repr(config, wr, hidden_repr, positions_repr,
            k_cache_repr, v_cache_repr, slot_repr, cu_q_repr, cu_k_repr,
            max_seqlen_q, max_seqlen_k, bt_repr).1
            .subrange(cu_q_repr[i as int], cu_q_repr[i as int + 1])
        == DS::first_decoder_layer_output_repr(config, wr,
            hidden_repr.subrange(cu_q_repr[i as int], cu_q_repr[i as int + 1]),
            positions_repr.subrange(cu_q_repr[i as int], cu_q_repr[i as int + 1]),
            k_cache_repr, v_cache_repr, slot_i,
            seq![0int, cu_q_repr[i as int + 1] - cu_q_repr[i as int]],
            seq![0int, cu_k_repr[i as int + 1] - cu_k_repr[i as int]],
            (cu_q_repr[i as int + 1] - cu_q_repr[i as int]) as nat,
            (cu_k_repr[i as int + 1] - cu_k_repr[i as int]) as nat,
            seq![bt_repr[i as int]]).1,
{
    broadcast use RT::lemma_rms_norm_repr_shape;
    let lo = cu_q_repr[i as int];
    let hi = cu_q_repr[i as int + 1];
    BI::rms_norm_subrange_invariance(
        hidden_repr, wr.input_norm, config.rms_norm_epsilon, lo, hi,
    );
    reveal(DS::first_decoder_layer_attention_launch_ready);
    decoder_core_request_isolation(config, wr, DS::rms_norm_repr(config, hidden_repr, wr.input_norm),
        hidden_repr, positions_repr,
        k_cache_repr, v_cache_repr, slot_repr, cu_q_repr, cu_k_repr,
        max_seqlen_q, max_seqlen_k, bt_repr, alt_k_cache, alt_v_cache, slot_i, i);
}

// `first_decoder_layer` (layer-0) wrapper isolation from block disjointness.
// First-layer analog of `decoder_layer_isolation_from_layout`: input norm via
// `rms_norm`, residual == hidden.  Discharges `first_decoder_layer_request_isolation`'s
// cache hypothesis via `cache_agreement_from_layout` on the rms-normed input.
pub proof fn first_decoder_layer_isolation_from_layout(
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
    slot_i: Seq<int>,
    i: nat,
)
    requires
        RT::paged_attention_numeric_domain(),
        hidden_repr.len() == positions_repr.len(),
        slot_repr.len() == hidden_repr.len(),
        DS::first_decoder_layer_attention_launch_ready(config,
            wr, hidden_repr, positions_repr,
            k_cache_repr, v_cache_repr, slot_repr,
            cu_q_repr, cu_k_repr, max_seqlen_q, max_seqlen_k, bt_repr,
        ),
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
        0 <= cu_q_repr[i as int] < cu_q_repr[i as int + 1] <= hidden_repr.len() as int,
        0 <= cu_k_repr[i as int] < cu_k_repr[i as int + 1],
        max_seqlen_q > 0,
        cu_q_repr[i as int + 1] - cu_q_repr[i as int] <= max_seqlen_q as int,
        cu_k_repr[i as int + 1] - cu_k_repr[i as int] <= max_seqlen_k as int,
        cu_q_repr[i as int + 1] - cu_q_repr[i as int]
            <= cu_k_repr[i as int + 1] - cu_k_repr[i as int],
        blocks_needed_for((cu_k_repr[i as int + 1] - cu_k_repr[i as int]) as nat)
            <= bt_repr[i as int].len(),
        slot_i == slot_repr.subrange(cu_q_repr[i as int], cu_q_repr[i as int + 1]),
        forall|m: int, l: int|
            #![trigger slot_repr.subrange(0, cu_q_repr[i as int])[m] / (BLOCK_SIZE_SPEC as int),
                bt_repr[i as int][l]]
            0 <= m < cu_q_repr[i as int] && 0 <= l < bt_repr[i as int].len() ==>
                slot_repr.subrange(0, cu_q_repr[i as int])[m] / (BLOCK_SIZE_SPEC as int)
                    != bt_repr[i as int][l] as int,
        forall|m: int, l: int|
            #![trigger slot_repr.subrange(cu_q_repr[i as int + 1], slot_repr.len() as int)[m]
                / (BLOCK_SIZE_SPEC as int), bt_repr[i as int][l]]
            0 <= m < slot_repr.len() - cu_q_repr[i as int + 1]
                && 0 <= l < bt_repr[i as int].len() ==>
                slot_repr.subrange(cu_q_repr[i as int + 1], slot_repr.len() as int)[m]
                    / (BLOCK_SIZE_SPEC as int) != bt_repr[i as int][l] as int,
        forall|pos: nat| pos < (cu_k_repr[i as int + 1] - cu_k_repr[i as int]) as nat ==>
            crate::proof::tensor::geometry::slot_in_cache(k_cache_repr,
                #[trigger] block_table_slot(bt_repr[i as int], pos))
            && crate::proof::tensor::geometry::slot_in_cache(v_cache_repr, block_table_slot(bt_repr[i as int], pos)),
    ensures
        DS::first_decoder_layer_output_repr(config, wr, hidden_repr, positions_repr,
            k_cache_repr, v_cache_repr, slot_repr, cu_q_repr, cu_k_repr,
            max_seqlen_q, max_seqlen_k, bt_repr).0
            .subrange(cu_q_repr[i as int], cu_q_repr[i as int + 1])
        == DS::first_decoder_layer_output_repr(config, wr,
            hidden_repr.subrange(cu_q_repr[i as int], cu_q_repr[i as int + 1]),
            positions_repr.subrange(cu_q_repr[i as int], cu_q_repr[i as int + 1]),
            k_cache_repr, v_cache_repr, slot_i,
            seq![0int, cu_q_repr[i as int + 1] - cu_q_repr[i as int]],
            seq![0int, cu_k_repr[i as int + 1] - cu_k_repr[i as int]],
            (cu_q_repr[i as int + 1] - cu_q_repr[i as int]) as nat,
            (cu_k_repr[i as int + 1] - cu_k_repr[i as int]) as nat,
            seq![bt_repr[i as int]]).0,
        DS::first_decoder_layer_output_repr(config, wr, hidden_repr, positions_repr,
            k_cache_repr, v_cache_repr, slot_repr, cu_q_repr, cu_k_repr,
            max_seqlen_q, max_seqlen_k, bt_repr).1
            .subrange(cu_q_repr[i as int], cu_q_repr[i as int + 1])
        == DS::first_decoder_layer_output_repr(config, wr,
            hidden_repr.subrange(cu_q_repr[i as int], cu_q_repr[i as int + 1]),
            positions_repr.subrange(cu_q_repr[i as int], cu_q_repr[i as int + 1]),
            k_cache_repr, v_cache_repr, slot_i,
            seq![0int, cu_q_repr[i as int + 1] - cu_q_repr[i as int]],
            seq![0int, cu_k_repr[i as int + 1] - cu_k_repr[i as int]],
            (cu_q_repr[i as int + 1] - cu_q_repr[i as int]) as nat,
            (cu_k_repr[i as int + 1] - cu_k_repr[i as int]) as nat,
            seq![bt_repr[i as int]]).1,
{
    broadcast use RT::lemma_rms_norm_repr_shape;
    let lo = cu_q_repr[i as int];
    let hi = cu_q_repr[i as int + 1];
    let kd = (cu_k_repr[i as int + 1] - cu_k_repr[i as int]) as nat;
    let normed = DS::rms_norm_repr(config, hidden_repr, wr.input_norm);
    let alt_k = DS::layer_kv_update_repr(config, wr,
        DS::rms_norm_repr(config, hidden_repr.subrange(lo, hi), wr.input_norm),
        positions_repr.subrange(lo, hi), k_cache_repr, v_cache_repr, slot_i).0;
    let alt_v = DS::layer_kv_update_repr(config, wr,
        DS::rms_norm_repr(config, hidden_repr.subrange(lo, hi), wr.input_norm),
        positions_repr.subrange(lo, hi), k_cache_repr, v_cache_repr, slot_i).1;
    // normed.subrange(lo,hi) == rms_norm(hidden.sub), so the single run's cache
    // (cache_agreement's alt) matches `alt_k`/`alt_v`.
    BI::rms_norm_subrange_invariance(
        hidden_repr, wr.input_norm, config.rms_norm_epsilon, lo, hi,
    );
    cache_agreement_from_layout(config, wr, normed, positions_repr, k_cache_repr, v_cache_repr,
        slot_repr, slot_i, bt_repr[i as int], lo, hi, kd);
    first_decoder_layer_request_isolation(config, wr, hidden_repr, positions_repr,
        k_cache_repr, v_cache_repr, slot_repr, cu_q_repr, cu_k_repr,
        max_seqlen_q, max_seqlen_k, bt_repr, alt_k, alt_v, slot_i, i);
}

// The KV-store inputs are request-local: a row segment's K (= pre_attention.1)
// and V (= view_as_kv(qkv(normed).v)) depend only on that segment of
// `normed`.  Building block for the eventual layer-chain fold, where per-layer
// store agreement needs that request i's own K/V rows match between the batched
// and single-request runs.  (The K side is `pre_attention_subrange_invariance.1`;
// this is the V side.)
pub proof fn kv_store_v_input_subrange(
    wr: LayerWeightsRepr,
    normed_repr: Tensor2D,
    a: int,
    b: int,
)
    requires 0 <= a <= b <= normed_repr.len(),
    ensures
        RT::view_as_kv_repr(RT::qkv_linear_repr(
            normed_repr.subrange(a, b), wr.q_proj, wr.k_proj, wr.v_proj,
        ).2) == RT::view_as_kv_repr(RT::qkv_linear_repr(
            normed_repr, wr.q_proj, wr.k_proj, wr.v_proj,
        ).2).subrange(a, b),
{
    broadcast use RT::lemma_qkv_linear_repr_shape;
    DENSE_PROPS::qkv_linear_subrange_invariance(
        normed_repr, wr.q_proj, wr.k_proj, wr.v_proj, a, b,
    );
    BI::view_as_kv_subrange_invariance(
        RT::qkv_linear_repr(normed_repr, wr.q_proj, wr.k_proj, wr.v_proj).2,
        a, b,
    );
}

// Both KV-store inputs are request-local, packaged in the `before ++ own ++ after`
// orientation the bridge uses: request i's own K (= pre_attention.1) and V slices
// equal the single-request run's K and V.  Combines `pre_attention_subrange_invariance`
// (K) and `kv_store_v_input_subrange` (V).
pub proof fn kv_store_inputs_request_local(
    config: DenseSwiGluForwardConfigRepr,
    wr: LayerWeightsRepr,
    normed_repr: Tensor2D,
    positions_repr: IntTensor1D,
    a: int,
    b: int,
)
    requires
        normed_repr.len() == positions_repr.len(),
        0 <= a <= b <= normed_repr.len(),
    ensures
        DS::pre_attention_repr(config, wr, normed_repr, positions_repr).1.subrange(a, b)
            == DS::pre_attention_repr(config, wr, normed_repr.subrange(a, b),
                positions_repr.subrange(a, b)).1,
        RT::view_as_kv_repr(RT::qkv_linear_repr(
            normed_repr, wr.q_proj, wr.k_proj, wr.v_proj,
        ).2).subrange(a, b) == RT::view_as_kv_repr(RT::qkv_linear_repr(
            normed_repr.subrange(a, b), wr.q_proj, wr.k_proj, wr.v_proj,
        ).2),
{
    pre_attention_subrange_invariance(config, wr, normed_repr, positions_repr, a, b);
    kv_store_v_input_subrange(wr, normed_repr, a, b);
}

// Standalone discharge of `decoder_core_request_isolation`'s cache-agreement
// hypothesis from the engine slot layout.  For a request-major layout — request
// i's own slots are `slot[lo..hi)` (`== slot_i`) and every other slot lies in a
// block disjoint from `bt_row` — the real_kv layer KV update agrees, at every read
// position, with the single-request layer KV update.  Extracted from the bridge
// so the term-matching is debuggable in isolation.
pub proof fn cache_agreement_from_layout(
    config: DenseSwiGluForwardConfigRepr,
    wr: LayerWeightsRepr,
    normed_repr: Tensor2D,
    positions_repr: IntTensor1D,
    k_cache_repr: KVCacheLayerRepr,
    v_cache_repr: KVCacheLayerRepr,
    slot_repr: Seq<int>,
    slot_i: Seq<int>,
    bt_row: Seq<BlockId>,
    lo: int,
    hi: int,
    kd: nat,
)
    requires
        normed_repr.len() == positions_repr.len(),
        slot_repr.len() == normed_repr.len(),
        0 <= lo <= hi <= normed_repr.len(),
        blocks_needed_for(kd) <= bt_row.len(),
        slot_i == slot_repr.subrange(lo, hi),
        forall|m: int, l: int|
            #![trigger slot_repr.subrange(0, lo)[m] / (BLOCK_SIZE_SPEC as int), bt_row[l]]
            0 <= m < lo && 0 <= l < bt_row.len() ==>
                slot_repr.subrange(0, lo)[m] / (BLOCK_SIZE_SPEC as int) != bt_row[l] as int,
        forall|m: int, l: int|
            #![trigger slot_repr.subrange(hi, slot_repr.len() as int)[m] / (BLOCK_SIZE_SPEC as int),
                bt_row[l]]
            0 <= m < slot_repr.len() - hi && 0 <= l < bt_row.len() ==>
                slot_repr.subrange(hi, slot_repr.len() as int)[m] / (BLOCK_SIZE_SPEC as int)
                    != bt_row[l] as int,
        forall|pos: nat| pos < kd ==>
            crate::proof::tensor::geometry::slot_in_cache(k_cache_repr, #[trigger] block_table_slot(bt_row, pos))
            && crate::proof::tensor::geometry::slot_in_cache(v_cache_repr, block_table_slot(bt_row, pos)),
    ensures
        forall|pos: nat| #![trigger block_table_slot(bt_row, pos)]
            pos < kd ==> {
                let real_kv = DS::layer_kv_update_repr(config, wr, normed_repr, positions_repr,
                    k_cache_repr, v_cache_repr, slot_repr);
                let alt_kv = DS::layer_kv_update_repr(config, wr, normed_repr.subrange(lo, hi),
                    positions_repr.subrange(lo, hi), k_cache_repr, v_cache_repr, slot_i);
                &&& (pos as int) / (BLOCK_SIZE_SPEC as int) < bt_row.len()
                &&& crate::proof::tensor::geometry::slot_in_cache(real_kv.0, block_table_slot(bt_row, pos))
                &&& crate::proof::tensor::geometry::slot_in_cache(real_kv.1, block_table_slot(bt_row, pos))
                &&& crate::proof::tensor::geometry::slot_in_cache(alt_kv.0, block_table_slot(bt_row, pos))
                &&& crate::proof::tensor::geometry::slot_in_cache(alt_kv.1, block_table_slot(bt_row, pos))
                &&& cache_at(real_kv.0, block_table_slot(bt_row, pos))
                        == cache_at(alt_kv.0, block_table_slot(bt_row, pos))
                &&& cache_at(real_kv.1, block_table_slot(bt_row, pos))
                        == cache_at(alt_kv.1, block_table_slot(bt_row, pos))
            },
{
    broadcast use {
        RT::lemma_qkv_linear_repr_shape,
        RT::lemma_view_as_kv_repr_shape,
    };
    let bs = BLOCK_SIZE_SPEC as int;
    let kk = DS::pre_attention_repr(config, wr, normed_repr, positions_repr).1;
    let vv = RT::view_as_kv_repr(RT::qkv_linear_repr(
        normed_repr, wr.q_proj, wr.k_proj, wr.v_proj,
    ).2);
    DS::lemma_pre_attention_repr_shape(config, wr, normed_repr, positions_repr);
    crate::proof::tensor::seq_flatten::lemma_seq_split3(kk, lo, hi);
    crate::proof::tensor::seq_flatten::lemma_seq_split3(vv, lo, hi);
    crate::proof::tensor::seq_flatten::lemma_seq_split3(slot_repr, lo, hi);
    kv_store_inputs_request_local(config, wr, normed_repr, positions_repr, lo, hi);

    // real_kv == store(kk, vv, kc, vc, slot) == store(split concats …) == full_store
    let real_kv = DS::layer_kv_update_repr(config, wr, normed_repr, positions_repr,
        k_cache_repr, v_cache_repr, slot_repr);
    let alt_kv = DS::layer_kv_update_repr(config, wr, normed_repr.subrange(lo, hi),
        positions_repr.subrange(lo, hi), k_cache_repr, v_cache_repr, slot_i);
    let full_store = RT::store_kv_cache_repr(
        kk.subrange(0, lo) + kk.subrange(lo, hi) + kk.subrange(hi, kk.len() as int),
        vv.subrange(0, lo) + vv.subrange(lo, hi) + vv.subrange(hi, vv.len() as int),
        k_cache_repr, v_cache_repr,
        slot_repr.subrange(0, lo) + slot_repr.subrange(lo, hi)
            + slot_repr.subrange(hi, slot_repr.len() as int));
    let own_store = RT::store_kv_cache_repr(kk.subrange(lo, hi), vv.subrange(lo, hi),
        k_cache_repr, v_cache_repr, slot_repr.subrange(lo, hi));
    assert(real_kv == full_store);
    assert(alt_kv == own_store);

    assert forall|pos: nat| #![trigger block_table_slot(bt_row, pos)]
        pos < kd implies {
            &&& crate::proof::tensor::geometry::slot_in_cache(real_kv.0, block_table_slot(bt_row, pos))
            &&& crate::proof::tensor::geometry::slot_in_cache(real_kv.1, block_table_slot(bt_row, pos))
            &&& crate::proof::tensor::geometry::slot_in_cache(alt_kv.0, block_table_slot(bt_row, pos))
            &&& crate::proof::tensor::geometry::slot_in_cache(alt_kv.1, block_table_slot(bt_row, pos))
            &&& cache_at(real_kv.0, block_table_slot(bt_row, pos))
                    == cache_at(alt_kv.0, block_table_slot(bt_row, pos))
            &&& cache_at(real_kv.1, block_table_slot(bt_row, pos))
                    == cache_at(alt_kv.1, block_table_slot(bt_row, pos))
        } by {
        crate::proof::tensor::geometry::lemma_blocks_needed_covers_pos(pos, kd);
        vstd::arithmetic::div_mod::lemma_fundamental_div_mod(pos as int, bs);
        assert((pos as int) / bs < bt_row.len());
        assert forall|m: int| 0 <= m < slot_repr.subrange(0, lo).len() implies
            #[trigger] slot_repr.subrange(0, lo)[m] / bs != bt_row[(pos as int) / bs] as int by {}
        assert forall|m: int|
            0 <= m < slot_repr.subrange(hi, slot_repr.len() as int).len() implies
            #[trigger] slot_repr.subrange(hi, slot_repr.len() as int)[m] / bs
                != bt_row[(pos as int) / bs] as int by {}
        RT::store_agrees_at_block_pos(
            kk.subrange(0, lo), vv.subrange(0, lo),
            kk.subrange(lo, hi), vv.subrange(lo, hi),
            kk.subrange(hi, kk.len() as int), vv.subrange(hi, vv.len() as int),
            k_cache_repr, v_cache_repr,
            slot_repr.subrange(0, lo), slot_repr.subrange(lo, hi),
            slot_repr.subrange(hi, slot_repr.len() as int),
            bt_row, pos);
        let s = block_table_slot(bt_row, pos);
        // `store_agrees` only certifies `slot_in_cache` for the FULL store, so
        // certify the OWN store (= alt) separately: the base cache has slot `s`
        // (old-cache hypothesis), and the store preserves cache shape.
        assert(crate::proof::tensor::geometry::slot_in_cache(k_cache_repr, s));
        assert(crate::proof::tensor::geometry::slot_in_cache(v_cache_repr, s));
        crate::boundary::tensor_runtime::lemma_store_kv_cache_repr_lengths(
            kk.subrange(lo, hi), vv.subrange(lo, hi),
            k_cache_repr, v_cache_repr, slot_repr.subrange(lo, hi));
    }
}


// `decoder_layer` (subsequent-layer) wrapper isolation from block disjointness:
// the unit the layer chain iterates.  Discharges `decoder_layer_request_isolation`'s
// cache hypothesis via `cache_agreement_from_layout` on the rms-normed input.
pub proof fn decoder_layer_isolation_from_layout(
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
    slot_i: Seq<int>,
    i: nat,
)
    requires
        RT::paged_attention_numeric_domain(),
        hidden_repr.len() == residual_repr.len(),
        hidden_repr.len() == positions_repr.len(),
        slot_repr.len() == hidden_repr.len(),
        DS::decoder_layer_attention_launch_ready(config,
            wr, hidden_repr, residual_repr, positions_repr,
            k_cache_repr, v_cache_repr, slot_repr,
            cu_q_repr, cu_k_repr, max_seqlen_q, max_seqlen_k, bt_repr,
        ),
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
        0 <= cu_q_repr[i as int] < cu_q_repr[i as int + 1] <= hidden_repr.len() as int,
        0 <= cu_k_repr[i as int] < cu_k_repr[i as int + 1],
        max_seqlen_q > 0,
        cu_q_repr[i as int + 1] - cu_q_repr[i as int] <= max_seqlen_q as int,
        cu_k_repr[i as int + 1] - cu_k_repr[i as int] <= max_seqlen_k as int,
        cu_q_repr[i as int + 1] - cu_q_repr[i as int]
            <= cu_k_repr[i as int + 1] - cu_k_repr[i as int],
        blocks_needed_for((cu_k_repr[i as int + 1] - cu_k_repr[i as int]) as nat)
            <= bt_repr[i as int].len(),
        slot_i == slot_repr.subrange(cu_q_repr[i as int], cu_q_repr[i as int + 1]),
        forall|m: int, l: int|
            #![trigger slot_repr.subrange(0, cu_q_repr[i as int])[m] / (BLOCK_SIZE_SPEC as int),
                bt_repr[i as int][l]]
            0 <= m < cu_q_repr[i as int] && 0 <= l < bt_repr[i as int].len() ==>
                slot_repr.subrange(0, cu_q_repr[i as int])[m] / (BLOCK_SIZE_SPEC as int)
                    != bt_repr[i as int][l] as int,
        forall|m: int, l: int|
            #![trigger slot_repr.subrange(cu_q_repr[i as int + 1], slot_repr.len() as int)[m]
                / (BLOCK_SIZE_SPEC as int), bt_repr[i as int][l]]
            0 <= m < slot_repr.len() - cu_q_repr[i as int + 1]
                && 0 <= l < bt_repr[i as int].len() ==>
                slot_repr.subrange(cu_q_repr[i as int + 1], slot_repr.len() as int)[m]
                    / (BLOCK_SIZE_SPEC as int) != bt_repr[i as int][l] as int,
        forall|pos: nat| pos < (cu_k_repr[i as int + 1] - cu_k_repr[i as int]) as nat ==>
            crate::proof::tensor::geometry::slot_in_cache(k_cache_repr,
                #[trigger] block_table_slot(bt_repr[i as int], pos))
            && crate::proof::tensor::geometry::slot_in_cache(v_cache_repr, block_table_slot(bt_repr[i as int], pos)),
    ensures
        DS::decoder_layer_output_repr(config, wr, hidden_repr, residual_repr, positions_repr,
            k_cache_repr, v_cache_repr, slot_repr, cu_q_repr, cu_k_repr,
            max_seqlen_q, max_seqlen_k, bt_repr).0
            .subrange(cu_q_repr[i as int], cu_q_repr[i as int + 1])
        == DS::decoder_layer_output_repr(config, wr,
            hidden_repr.subrange(cu_q_repr[i as int], cu_q_repr[i as int + 1]),
            residual_repr.subrange(cu_q_repr[i as int], cu_q_repr[i as int + 1]),
            positions_repr.subrange(cu_q_repr[i as int], cu_q_repr[i as int + 1]),
            k_cache_repr, v_cache_repr, slot_i,
            seq![0int, cu_q_repr[i as int + 1] - cu_q_repr[i as int]],
            seq![0int, cu_k_repr[i as int + 1] - cu_k_repr[i as int]],
            (cu_q_repr[i as int + 1] - cu_q_repr[i as int]) as nat,
            (cu_k_repr[i as int + 1] - cu_k_repr[i as int]) as nat,
            seq![bt_repr[i as int]]).0,
        DS::decoder_layer_output_repr(config, wr, hidden_repr, residual_repr, positions_repr,
            k_cache_repr, v_cache_repr, slot_repr, cu_q_repr, cu_k_repr,
            max_seqlen_q, max_seqlen_k, bt_repr).1
            .subrange(cu_q_repr[i as int], cu_q_repr[i as int + 1])
        == DS::decoder_layer_output_repr(config, wr,
            hidden_repr.subrange(cu_q_repr[i as int], cu_q_repr[i as int + 1]),
            residual_repr.subrange(cu_q_repr[i as int], cu_q_repr[i as int + 1]),
            positions_repr.subrange(cu_q_repr[i as int], cu_q_repr[i as int + 1]),
            k_cache_repr, v_cache_repr, slot_i,
            seq![0int, cu_q_repr[i as int + 1] - cu_q_repr[i as int]],
            seq![0int, cu_k_repr[i as int + 1] - cu_k_repr[i as int]],
            (cu_q_repr[i as int + 1] - cu_q_repr[i as int]) as nat,
            (cu_k_repr[i as int + 1] - cu_k_repr[i as int]) as nat,
            seq![bt_repr[i as int]]).1,
{
    broadcast use RT::lemma_add_rms_norm_repr_shape;
    let lo = cu_q_repr[i as int];
    let hi = cu_q_repr[i as int + 1];
    let kd = (cu_k_repr[i as int + 1] - cu_k_repr[i as int]) as nat;
    let normed = DS::add_rms_norm_repr(config, hidden_repr, residual_repr, wr.input_norm).0;
    let alt_k = DS::layer_kv_update_repr(config, wr,
        DS::add_rms_norm_repr(config, hidden_repr.subrange(lo, hi), residual_repr.subrange(lo, hi),
            wr.input_norm).0,
        positions_repr.subrange(lo, hi), k_cache_repr, v_cache_repr, slot_i).0;
    let alt_v = DS::layer_kv_update_repr(config, wr,
        DS::add_rms_norm_repr(config, hidden_repr.subrange(lo, hi), residual_repr.subrange(lo, hi),
            wr.input_norm).0,
        positions_repr.subrange(lo, hi), k_cache_repr, v_cache_repr, slot_i).1;
    // normed.subrange(lo,hi) == add_rms_norm(hidden.sub, residual.sub).0, so the
    // single run's cache (cache_agreement's alt) matches `alt_k`/`alt_v`.
    BI::add_rms_norm_subrange_invariance(
        hidden_repr, residual_repr, wr.input_norm,
        config.rms_norm_epsilon, lo, hi,
    );
    cache_agreement_from_layout(config, wr, normed, positions_repr, k_cache_repr, v_cache_repr,
        slot_repr, slot_i, bt_repr[i as int], lo, hi, kd);
    decoder_layer_request_isolation(config, wr, hidden_repr, residual_repr, positions_repr,
        k_cache_repr, v_cache_repr, slot_repr, cu_q_repr, cu_k_repr,
        max_seqlen_q, max_seqlen_k, bt_repr, alt_k, alt_v, slot_i, i);
}

// ---------------------------------------------------------------------------
// G3: whole-model isolation lift.  Induct the per-layer pair isolation
// (`decoder_layer_isolation_from_layout`) over `layer_chain_repr`: the batched
// chain's (hidden, residual) outputs sliced to request `i` equal the singleton
// chain run on request `i`'s inputs alone.  Both output components are carried
// because the chain feeds both forward.  Each layer's cache hypothesis (request
// `i`'s read slots are in-cache) is supplied per-layer over `kv_cache_reprs`;
// block-disjointness (layer-independent) discharges the cross-request
// non-interference at every layer.
// ---------------------------------------------------------------------------
pub proof fn layer_chain_request_isolation(
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
)
    requires
        RT::paged_attention_numeric_domain(),
        start <= layers.len(),
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
        0 <= cu_q_repr[i as int] < cu_q_repr[i as int + 1] <= hidden_repr.len() as int,
        0 <= cu_k_repr[i as int] < cu_k_repr[i as int + 1],
        max_seqlen_q > 0,
        cu_q_repr[i as int + 1] - cu_q_repr[i as int] <= max_seqlen_q as int,
        cu_k_repr[i as int + 1] - cu_k_repr[i as int] <= max_seqlen_k as int,
        cu_q_repr[i as int + 1] - cu_q_repr[i as int]
            <= cu_k_repr[i as int + 1] - cu_k_repr[i as int],
        blocks_needed_for((cu_k_repr[i as int + 1] - cu_k_repr[i as int]) as nat)
            <= bt_repr[i as int].len(),
        slot_i == slot_repr.subrange(cu_q_repr[i as int], cu_q_repr[i as int + 1]),
        forall|m: int, l: int|
            #![trigger slot_repr.subrange(0, cu_q_repr[i as int])[m] / (BLOCK_SIZE_SPEC as int),
                bt_repr[i as int][l]]
            0 <= m < cu_q_repr[i as int] && 0 <= l < bt_repr[i as int].len() ==>
                slot_repr.subrange(0, cu_q_repr[i as int])[m] / (BLOCK_SIZE_SPEC as int)
                    != bt_repr[i as int][l] as int,
        forall|m: int, l: int|
            #![trigger slot_repr.subrange(cu_q_repr[i as int + 1], slot_repr.len() as int)[m]
                / (BLOCK_SIZE_SPEC as int), bt_repr[i as int][l]]
            0 <= m < slot_repr.len() - cu_q_repr[i as int + 1]
                && 0 <= l < bt_repr[i as int].len() ==>
                slot_repr.subrange(cu_q_repr[i as int + 1], slot_repr.len() as int)[m]
                    / (BLOCK_SIZE_SPEC as int) != bt_repr[i as int][l] as int,
        // Per-layer cache coverage: request i's read slots are in-cache at every
        // remaining layer's K/V cache.
        forall|ell: int, pos: nat|
            #![trigger kv_cache_reprs[ell].0, block_table_slot(bt_repr[i as int], pos)]
            start <= ell < layers.len()
                && pos < (cu_k_repr[i as int + 1] - cu_k_repr[i as int]) as nat ==>
                crate::proof::tensor::geometry::slot_in_cache(kv_cache_reprs[ell].0,
                    block_table_slot(bt_repr[i as int], pos))
                && crate::proof::tensor::geometry::slot_in_cache(kv_cache_reprs[ell].1,
                    block_table_slot(bt_repr[i as int], pos)),
        forall|ell: int| start <= ell < layers.len() ==>
            #[trigger] RT::paged_attention_launch_ready(
                hidden_repr.len(), kv_cache_reprs[ell].0, kv_cache_reprs[ell].1,
                cu_q_repr, cu_k_repr, max_seqlen_q, max_seqlen_k, bt_repr,
            ),
    ensures
        DS::layer_chain_repr(config, layers, hidden_repr, residual_repr, positions_repr,
            kv_cache_reprs, slot_repr, cu_q_repr, cu_k_repr,
            max_seqlen_q, max_seqlen_k, bt_repr, start).0
            .subrange(cu_q_repr[i as int], cu_q_repr[i as int + 1])
        == DS::layer_chain_repr(config, layers,
            hidden_repr.subrange(cu_q_repr[i as int], cu_q_repr[i as int + 1]),
            residual_repr.subrange(cu_q_repr[i as int], cu_q_repr[i as int + 1]),
            positions_repr.subrange(cu_q_repr[i as int], cu_q_repr[i as int + 1]),
            kv_cache_reprs, slot_i,
            seq![0int, cu_q_repr[i as int + 1] - cu_q_repr[i as int]],
            seq![0int, cu_k_repr[i as int + 1] - cu_k_repr[i as int]],
            (cu_q_repr[i as int + 1] - cu_q_repr[i as int]) as nat,
            (cu_k_repr[i as int + 1] - cu_k_repr[i as int]) as nat,
            seq![bt_repr[i as int]], start).0,
        DS::layer_chain_repr(config, layers, hidden_repr, residual_repr, positions_repr,
            kv_cache_reprs, slot_repr, cu_q_repr, cu_k_repr,
            max_seqlen_q, max_seqlen_k, bt_repr, start).1
            .subrange(cu_q_repr[i as int], cu_q_repr[i as int + 1])
        == DS::layer_chain_repr(config, layers,
            hidden_repr.subrange(cu_q_repr[i as int], cu_q_repr[i as int + 1]),
            residual_repr.subrange(cu_q_repr[i as int], cu_q_repr[i as int + 1]),
            positions_repr.subrange(cu_q_repr[i as int], cu_q_repr[i as int + 1]),
            kv_cache_reprs, slot_i,
            seq![0int, cu_q_repr[i as int + 1] - cu_q_repr[i as int]],
            seq![0int, cu_k_repr[i as int + 1] - cu_k_repr[i as int]],
            (cu_q_repr[i as int + 1] - cu_q_repr[i as int]) as nat,
            (cu_k_repr[i as int + 1] - cu_k_repr[i as int]) as nat,
            seq![bt_repr[i as int]], start).1,
    decreases (layers.len() - start) as nat,
{
    let lo = cu_q_repr[i as int];
    let hi = cu_q_repr[i as int + 1];
    let qd = (hi - lo) as nat;
    let kd = (cu_k_repr[i as int + 1] - cu_k_repr[i as int]) as nat;
    let sing_cu_q = seq![0int, qd as int];
    let sing_cu_k = seq![0int, kd as int];
    let sing_bt = seq![bt_repr[i as int]];
    let hidden_i = hidden_repr.subrange(lo, hi);
    let residual_i = residual_repr.subrange(lo, hi);
    let positions_i = positions_repr.subrange(lo, hi);

    if start >= layers.len() {
        // Base case: both chains return their inputs.  LHS slices the batched
        // inputs; RHS already is the sliced inputs.
        assert(hidden_repr.subrange(lo, hi) == hidden_i);
        assert(residual_repr.subrange(lo, hi) == residual_i);
    } else {
        // Per-layer pair isolation for layer `start`.
        lemma_decoder_layer_attention_launch_ready_from_pre_store(config,
            layers[start as int], hidden_repr, residual_repr, positions_repr,
            kv_cache_reprs[start as int].0, kv_cache_reprs[start as int].1,
            slot_repr, cu_q_repr, cu_k_repr, max_seqlen_q, max_seqlen_k,
            bt_repr,
        );
        decoder_layer_isolation_from_layout(config, layers[start as int], hidden_repr, residual_repr,
            positions_repr, kv_cache_reprs[start as int].0, kv_cache_reprs[start as int].1,
            slot_repr, cu_q_repr, cu_k_repr, max_seqlen_q, max_seqlen_k, bt_repr, slot_i, i);

        let next_b = DS::decoder_layer_output_repr(config, layers[start as int], hidden_repr, residual_repr,
            positions_repr, kv_cache_reprs[start as int].0, kv_cache_reprs[start as int].1,
            slot_repr, cu_q_repr, cu_k_repr, max_seqlen_q, max_seqlen_k, bt_repr);
        let next_s = DS::decoder_layer_output_repr(config, layers[start as int], hidden_i, residual_i,
            positions_i, kv_cache_reprs[start as int].0, kv_cache_reprs[start as int].1,
            slot_i, sing_cu_q, sing_cu_k, qd, kd, sing_bt);

        // next_b output lengths are preserved (== hidden_repr.len()).
        DS::lemma_decoder_layer_output_repr_shape(config, layers[start as int], hidden_repr, residual_repr,
            positions_repr, kv_cache_reprs[start as int].0, kv_cache_reprs[start as int].1,
            slot_repr, cu_q_repr, cu_k_repr, max_seqlen_q, max_seqlen_k, bt_repr);

        // Inductive hypothesis on the tail (start+1) fed the batched layer output.
        layer_chain_request_isolation(config, layers, next_b.0, next_b.1, positions_repr,
            kv_cache_reprs, slot_repr, cu_q_repr, cu_k_repr, max_seqlen_q, max_seqlen_k,
            bt_repr, slot_i, i, (start + 1) as nat);

        // Isolation: the batched layer output sliced to request i is the singleton
        // layer output, so the IH's RHS chain input equals the singleton chain input.
        assert(next_b.0.subrange(lo, hi) == next_s.0);
        assert(next_b.1.subrange(lo, hi) == next_s.1);
    }
}

// A layer-chain fold starting at `start` never reads cache entries below
// `start`.  This framing lemma is useful when the companion cache-recording
// fold has already updated an earlier layer before recursing.
pub proof fn lemma_layer_chain_repr_ignores_before_start(
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
)
    requires
        start <= layers.len(),
        kv_a.len() >= layers.len(),
        kv_b.len() >= layers.len(),
        forall|layer: int| start <= layer < layers.len() ==>
            #[trigger] kv_a[layer] == kv_b[layer],
    ensures
        DS::layer_chain_repr(config, layers, hidden_repr, residual_repr, positions_repr,
            kv_a, slot_repr, cu_q_repr, cu_k_repr,
            max_seqlen_q, max_seqlen_k, bt_repr, start)
        == DS::layer_chain_repr(config, layers, hidden_repr, residual_repr, positions_repr,
            kv_b, slot_repr, cu_q_repr, cu_k_repr,
            max_seqlen_q, max_seqlen_k, bt_repr, start),
    decreases (layers.len() - start) as nat,
{
    if start >= layers.len() {
    } else {
        let layer = start as int;
        assert(kv_a[layer] == kv_b[layer]);
        let next_a = DS::decoder_layer_output_repr(config, layers[layer],
            hidden_repr, residual_repr, positions_repr,
            kv_a[layer].0, kv_a[layer].1, slot_repr,
            cu_q_repr, cu_k_repr, max_seqlen_q, max_seqlen_k, bt_repr);
        let next_b = DS::decoder_layer_output_repr(config, layers[layer],
            hidden_repr, residual_repr, positions_repr,
            kv_b[layer].0, kv_b[layer].1, slot_repr,
            cu_q_repr, cu_k_repr, max_seqlen_q, max_seqlen_k, bt_repr);
        assert(next_a == next_b);
        lemma_layer_chain_repr_ignores_before_start(config,
            layers, next_a.0, next_a.1, positions_repr,
            kv_a, kv_b, slot_repr, cu_q_repr, cu_k_repr,
            max_seqlen_q, max_seqlen_k, bt_repr, (start + 1) as nat,
        );
    }
}

// ---------------------------------------------------------------------------
// G3 capstone: whole-model isolation.  The batched `model_forward_logits_repr`
// sliced to request `i`'s query rows equals the single-request forward run on
// request `i`'s tokens/positions alone (its own block table `seq![bt[i]]`,
// contiguous slots `slot_i`).  Assembles: embed (row-wise) → first-layer pair
// isolation → layer-chain pair isolation → final-norm (row-wise) → lm_head
// (row-wise).  This is the engine⊑machine per-request forward equality (modulo
// cache geometry, supplied by the per-layer cache-coverage hypothesis).
// ---------------------------------------------------------------------------
#[verifier::spinoff_prover]
pub proof fn model_forward_request_isolation(
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
)
    requires
        RT::paged_attention_numeric_domain(),
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
        0 <= cu_q_repr[i as int] < cu_q_repr[i as int + 1] <= input_ids_repr.len() as int,
        0 <= cu_k_repr[i as int] < cu_k_repr[i as int + 1],
        max_seqlen_q > 0,
        cu_q_repr[i as int + 1] - cu_q_repr[i as int] <= max_seqlen_q as int,
        cu_k_repr[i as int + 1] - cu_k_repr[i as int] <= max_seqlen_k as int,
        cu_q_repr[i as int + 1] - cu_q_repr[i as int]
            <= cu_k_repr[i as int + 1] - cu_k_repr[i as int],
        blocks_needed_for((cu_k_repr[i as int + 1] - cu_k_repr[i as int]) as nat)
            <= bt_repr[i as int].len(),
        slot_i == slot_repr.subrange(cu_q_repr[i as int], cu_q_repr[i as int + 1]),
        forall|m: int, l: int|
            #![trigger slot_repr.subrange(0, cu_q_repr[i as int])[m] / (BLOCK_SIZE_SPEC as int),
                bt_repr[i as int][l]]
            0 <= m < cu_q_repr[i as int] && 0 <= l < bt_repr[i as int].len() ==>
                slot_repr.subrange(0, cu_q_repr[i as int])[m] / (BLOCK_SIZE_SPEC as int)
                    != bt_repr[i as int][l] as int,
        forall|m: int, l: int|
            #![trigger slot_repr.subrange(cu_q_repr[i as int + 1], slot_repr.len() as int)[m]
                / (BLOCK_SIZE_SPEC as int), bt_repr[i as int][l]]
            0 <= m < slot_repr.len() - cu_q_repr[i as int + 1]
                && 0 <= l < bt_repr[i as int].len() ==>
                slot_repr.subrange(cu_q_repr[i as int + 1], slot_repr.len() as int)[m]
                    / (BLOCK_SIZE_SPEC as int) != bt_repr[i as int][l] as int,
        forall|ell: int, pos: nat|
            #![trigger kv_cache_reprs[ell].0, block_table_slot(bt_repr[i as int], pos)]
            0 <= ell < wr.layers.len()
                && pos < (cu_k_repr[i as int + 1] - cu_k_repr[i as int]) as nat ==>
                crate::proof::tensor::geometry::slot_in_cache(kv_cache_reprs[ell].0,
                    block_table_slot(bt_repr[i as int], pos))
                && crate::proof::tensor::geometry::slot_in_cache(kv_cache_reprs[ell].1,
                    block_table_slot(bt_repr[i as int], pos)),
        forall|ell: int| 0 <= ell < wr.layers.len() ==>
            #[trigger] RT::paged_attention_launch_ready(
                input_ids_repr.len(), kv_cache_reprs[ell].0,
                kv_cache_reprs[ell].1, cu_q_repr, cu_k_repr,
                max_seqlen_q, max_seqlen_k, bt_repr,
            ),
    ensures
        DS::model_forward_logits_repr(config, wr, input_ids_repr, positions_repr, kv_cache_reprs,
            slot_repr, cu_q_repr, cu_k_repr, max_seqlen_q, max_seqlen_k, bt_repr)
            .subrange(cu_q_repr[i as int], cu_q_repr[i as int + 1])
        == DS::model_forward_logits_repr(config, wr,
            input_ids_repr.subrange(cu_q_repr[i as int], cu_q_repr[i as int + 1]),
            positions_repr.subrange(cu_q_repr[i as int], cu_q_repr[i as int + 1]),
            kv_cache_reprs, slot_i,
            seq![0int, cu_q_repr[i as int + 1] - cu_q_repr[i as int]],
            seq![0int, cu_k_repr[i as int + 1] - cu_k_repr[i as int]],
            (cu_q_repr[i as int + 1] - cu_q_repr[i as int]) as nat,
            (cu_k_repr[i as int + 1] - cu_k_repr[i as int]) as nat,
            seq![bt_repr[i as int]]),
{
    reveal(DS::model_forward_logits_repr);
    broadcast use {
        RT::lemma_add_rms_norm_repr_shape,
        RT::lemma_linear_repr_shape,
        RT::lemma_embed_repr_shape,
    };
    let lo = cu_q_repr[i as int];
    let hi = cu_q_repr[i as int + 1];
    let qd = (hi - lo) as nat;
    let kd = (cu_k_repr[i as int + 1] - cu_k_repr[i as int]) as nat;
    let sing_cu_q = seq![0int, qd as int];
    let sing_cu_k = seq![0int, kd as int];
    let sing_bt = seq![bt_repr[i as int]];

    // The imported embedding certificate composes from rows to this request slice.
    let embed = RT::embed_repr(input_ids_repr, wr.embed_weight);
    BI::embed_subrange_invariance(input_ids_repr, wr.embed_weight, lo, hi);
    assert(embed.subrange(lo, hi)
        == RT::embed_repr(input_ids_repr.subrange(lo, hi), wr.embed_weight));

    // First layer (layer 0) pair isolation over the embed input.
    lemma_first_decoder_layer_attention_launch_ready_from_pre_store(config,
        wr.layers[0], embed, positions_repr,
        kv_cache_reprs[0].0, kv_cache_reprs[0].1, slot_repr,
        cu_q_repr, cu_k_repr, max_seqlen_q, max_seqlen_k, bt_repr,
    );
    first_decoder_layer_isolation_from_layout(config, wr.layers[0], embed, positions_repr,
        kv_cache_reprs[0].0, kv_cache_reprs[0].1, slot_repr, cu_q_repr, cu_k_repr,
        max_seqlen_q, max_seqlen_k, bt_repr, slot_i, i);
    let first = DS::first_decoder_layer_output_repr(config, wr.layers[0], embed, positions_repr,
        kv_cache_reprs[0].0, kv_cache_reprs[0].1, slot_repr, cu_q_repr, cu_k_repr,
        max_seqlen_q, max_seqlen_k, bt_repr);
    DS::lemma_first_decoder_layer_output_repr_shape(config, wr.layers[0], embed, positions_repr,
        kv_cache_reprs[0].0, kv_cache_reprs[0].1, slot_repr, cu_q_repr, cu_k_repr,
        max_seqlen_q, max_seqlen_k, bt_repr);

    // Layer-chain (layers 1..n) pair isolation over the first layer's output.
    assert forall|ell: int, pos: nat|
        #![trigger kv_cache_reprs[ell].0, block_table_slot(bt_repr[i as int], pos)]
        1 <= ell < wr.layers.len() && pos < kd implies
            slot_in_cache(kv_cache_reprs[ell].0, block_table_slot(bt_repr[i as int], pos))
            && slot_in_cache(kv_cache_reprs[ell].1, block_table_slot(bt_repr[i as int], pos))
    by {
        assert(0 <= ell < wr.layers.len());
        assert(slot_in_cache(kv_cache_reprs[ell].0, block_table_slot(bt_repr[i as int], pos)));
        assert(slot_in_cache(kv_cache_reprs[ell].1, block_table_slot(bt_repr[i as int], pos)));
    }
    layer_chain_request_isolation(config, wr.layers, first.0, first.1, positions_repr,
        kv_cache_reprs, slot_repr, cu_q_repr, cu_k_repr, max_seqlen_q, max_seqlen_k,
        bt_repr, slot_i, i, 1);
    let chain = DS::layer_chain_repr(config, wr.layers, first.0, first.1, positions_repr,
        kv_cache_reprs, slot_repr, cu_q_repr, cu_k_repr, max_seqlen_q, max_seqlen_k,
        bt_repr, 1);
    DS::lemma_layer_chain_repr_shape(config, wr.layers, first.0, first.1, positions_repr,
        kv_cache_reprs, slot_repr, cu_q_repr, cu_k_repr, max_seqlen_q, max_seqlen_k,
        bt_repr, 1);

    // Final norm + lm_head are row-wise.
    BI::add_rms_norm_subrange_invariance(
        chain.0, chain.1, wr.final_norm,
        config.rms_norm_epsilon, lo, hi,
    );
    let final0 = DS::add_rms_norm_repr(config, chain.0, chain.1, wr.final_norm).0;
    BI::linear_subrange_invariance(final0, wr.lm_head, lo, hi);
}

// ===========================================================================
// G3′: geometry relocation.  A SINGLE-request decoder layer / chain / model is
// invariant under swapping the cache *geometry* (physical block table `bt_row`
// and slot mapping `slot`) for another, provided the post-store K/V the request
// reads agree at every position.  This is the relocation analog of the
// isolation lemmas above; it lifts the kernel fact
// `paged_attention_physical_relocation` to the model level.  The per-layer
// post-store-agreement hypothesis is what `engine_kv_coherent` (G6) discharges
// (engine paged geometry `A` ≡ machine contiguous geometry `B`).
// ===========================================================================

// Per-layer relocation unit (decoder core, input already normed).  Direct lift
// of `paged_attention_physical_relocation` to a decoder layer: identical inputs
// (normed/residual/positions), only the cache geometry differs.
pub proof fn decoder_core_relocation(
    config: DenseSwiGluForwardConfigRepr,
    wr: LayerWeightsRepr,
    normed_repr: Tensor2D,
    residual_repr: Tensor2D,
    positions_repr: IntTensor1D,
    k_cache_a: KVCacheLayerRepr,
    v_cache_a: KVCacheLayerRepr,
    slot_a: Seq<int>,
    bt_row_a: Seq<BlockId>,
    k_cache_b: KVCacheLayerRepr,
    v_cache_b: KVCacheLayerRepr,
    slot_b: Seq<int>,
    bt_row_b: Seq<BlockId>,
    q_len: nat,
    k_len: nat,
)
    requires
        RT::paged_attention_numeric_domain(),
        normed_repr.len() == residual_repr.len(),
        normed_repr.len() == positions_repr.len(),
        slot_a.len() == normed_repr.len(),
        slot_b.len() == normed_repr.len(),
        normed_repr.len() == q_len,
        q_len > 0,
        q_len <= k_len,
        blocks_needed_for(k_len) <= bt_row_a.len(),
        blocks_needed_for(k_len) <= bt_row_b.len(),
        // Post-store K/V the request reads agree across the two geometries.
        forall|pos: nat| #![trigger block_table_slot(bt_row_a, pos)]
            pos < k_len ==> {
                let post_a = DS::layer_kv_update_repr(config, wr, normed_repr, positions_repr,
                    k_cache_a, v_cache_a, slot_a);
                let post_b = DS::layer_kv_update_repr(config, wr, normed_repr, positions_repr,
                    k_cache_b, v_cache_b, slot_b);
                (pos as int) / (BLOCK_SIZE_SPEC as int) < bt_row_a.len()
                && (pos as int) / (BLOCK_SIZE_SPEC as int) < bt_row_b.len()
                && slot_in_cache(post_a.0, block_table_slot(bt_row_a, pos))
                && slot_in_cache(post_a.1, block_table_slot(bt_row_a, pos))
                && slot_in_cache(post_b.0, block_table_slot(bt_row_b, pos))
                && slot_in_cache(post_b.1, block_table_slot(bt_row_b, pos))
                && cache_at(post_a.0, block_table_slot(bt_row_a, pos))
                    == cache_at(post_b.0, block_table_slot(bt_row_b, pos))
                && cache_at(post_a.1, block_table_slot(bt_row_a, pos))
                    == cache_at(post_b.1, block_table_slot(bt_row_b, pos))
            },
    ensures
        DS::decoder_core_output_repr(config, wr, normed_repr, residual_repr, positions_repr,
            k_cache_a, v_cache_a, slot_a, seq![0int, q_len as int], seq![0int, k_len as int],
            q_len, k_len, seq![bt_row_a])
        == DS::decoder_core_output_repr(config, wr, normed_repr, residual_repr, positions_repr,
            k_cache_b, v_cache_b, slot_b, seq![0int, q_len as int], seq![0int, k_len as int],
            q_len, k_len, seq![bt_row_b]),
{
    let pre = DS::pre_attention_repr(config, wr, normed_repr, positions_repr);
    DS::lemma_pre_attention_repr_shape(config, wr, normed_repr, positions_repr);
    let post_a = DS::layer_kv_update_repr(config, wr, normed_repr, positions_repr,
        k_cache_a, v_cache_a, slot_a);
    let post_b = DS::layer_kv_update_repr(config, wr, normed_repr, positions_repr,
        k_cache_b, v_cache_b, slot_b);
    // Forward the post-store agreement hypothesis into the shape relocation wants.
    assert forall|pos: nat| pos < k_len implies
        (pos as int) / (BLOCK_SIZE_SPEC as int) < bt_row_a.len()
        && (pos as int) / (BLOCK_SIZE_SPEC as int) < bt_row_b.len()
        && slot_in_cache(post_a.0, block_table_slot(bt_row_a, pos))
        && slot_in_cache(post_a.1, block_table_slot(bt_row_a, pos))
        && slot_in_cache(post_b.0, block_table_slot(bt_row_b, pos))
        && slot_in_cache(post_b.1, block_table_slot(bt_row_b, pos))
        && cache_at(post_a.0, block_table_slot(bt_row_a, pos))
            == cache_at(post_b.0, block_table_slot(bt_row_b, pos))
        && cache_at(post_a.1, block_table_slot(bt_row_a, pos))
            == cache_at(post_b.1, block_table_slot(bt_row_b, pos))
    by {
        // Fire the requires forall's multi-trigger at this pos.
        let _ga = block_table_slot(bt_row_a, pos);
        let _gb = block_table_slot(bt_row_b, pos);
    }
    // Identical query rows + agreeing post-store caches ⇒ equal paged attention.
    BI::paged_attention_physical_relocation(pre.0, post_a.0, post_a.1, post_b.0, post_b.1,
        q_len, k_len, bt_row_a, bt_row_b,
        crate::boundary::dense_layer_primitives::layer_attention_parameters_repr(wr, AttentionScaleRepr::InverseSqrtHeadDim),
        );
    // post_attention is a function of (attn, residual): equal attn ⇒ equal output.
}

/// The slots written by the current query do not alias cache slots read from
/// the already-cached prefix, under either physical layout.  Naming this
/// repeated relocation obligation keeps wrapper theorem signatures aligned
/// and confines its quantifier trigger to one definition.
pub open spec fn fresh_writes_miss_cached_prefix(
    slot_a: Seq<int>,
    bt_row_a: Seq<BlockId>,
    slot_b: Seq<int>,
    bt_row_b: Seq<BlockId>,
    prefix_len: int,
) -> bool {
    forall|pos: nat| #![trigger crate::proof::tensor::geometry::block_table_slot(bt_row_a, pos)]
        pos < prefix_len ==>
            !slot_a.contains(crate::proof::tensor::geometry::block_table_slot(bt_row_a, pos) as int)
            && !slot_b.contains(crate::proof::tensor::geometry::block_table_slot(bt_row_b, pos) as int)
}

pub proof fn fresh_writes_miss_cached_prefix_at(
    slot_a: Seq<int>,
    bt_row_a: Seq<BlockId>,
    slot_b: Seq<int>,
    bt_row_b: Seq<BlockId>,
    prefix_len: int,
    pos: nat,
)
    requires
        fresh_writes_miss_cached_prefix(slot_a, bt_row_a, slot_b, bt_row_b, prefix_len),
        pos < prefix_len,
        (pos as int) / (BLOCK_SIZE_SPEC as int) < bt_row_a.len(),
        (pos as int) / (BLOCK_SIZE_SPEC as int) < bt_row_b.len(),
    ensures
        !slot_a.contains(crate::proof::tensor::geometry::block_table_slot(bt_row_a, pos) as int),
        !slot_b.contains(crate::proof::tensor::geometry::block_table_slot(bt_row_b, pos) as int),
{
    reveal(fresh_writes_miss_cached_prefix);
    assert(!slot_a.contains(crate::proof::tensor::geometry::block_table_slot(bt_row_a, pos) as int));
    assert(!slot_b.contains(crate::proof::tensor::geometry::block_table_slot(bt_row_b, pos) as int));
}

// Per-layer relocation FROM the engine/machine layout: derives
// `decoder_core_relocation`'s post-store-agreement hypothesis from
// *input-independent* layout facts — (F) fresh-position alignment
// `block_table_slot(bt, k−q+j) == slot[j]`, (F′) distinct fresh slots, (F″)
// fresh slots in-cache, (P) pre-store cache agreement on the cached prefix
// (`engine_kv_coherent`), and (P′) fresh writes miss the cached read slots.
// Selects fresh-vs-prefix per `pos < k` and assembles the full agreement.
pub proof fn decoder_core_relocation_from_layout(
    config: DenseSwiGluForwardConfigRepr,
    wr: LayerWeightsRepr,
    normed_repr: Tensor2D,
    residual_repr: Tensor2D,
    positions_repr: IntTensor1D,
    k_cache_a: KVCacheLayerRepr,
    v_cache_a: KVCacheLayerRepr,
    slot_a: Seq<int>,
    bt_row_a: Seq<BlockId>,
    k_cache_b: KVCacheLayerRepr,
    v_cache_b: KVCacheLayerRepr,
    slot_b: Seq<int>,
    bt_row_b: Seq<BlockId>,
    q_len: nat,
    k_len: nat,
)
    requires
        RT::paged_attention_numeric_domain(),
        normed_repr.len() == residual_repr.len(),
        normed_repr.len() == positions_repr.len(),
        slot_a.len() == normed_repr.len(),
        slot_b.len() == normed_repr.len(),
        normed_repr.len() == q_len,
        q_len > 0,
        q_len <= k_len,
        blocks_needed_for(k_len) <= bt_row_a.len(),
        blocks_needed_for(k_len) <= bt_row_b.len(),
        // (F) fresh-position alignment + nonnegativity.
        forall|j: int| 0 <= j < q_len as int ==>
            crate::proof::tensor::geometry::block_table_slot(bt_row_a, (k_len - q_len + j as nat) as nat)
                == #[trigger] (slot_a[j]) as nat
            && crate::proof::tensor::geometry::block_table_slot(bt_row_b, (k_len - q_len + j as nat) as nat)
                == (slot_b[j]) as nat,
        forall|j: int| 0 <= j < q_len as int ==>
            #[trigger] (slot_a[j]) >= 0 && (slot_b[j]) >= 0,
        // (F′) distinct fresh slots.
        forall|j: int, m: int| #![trigger slot_a[m], slot_a[j]]
            0 <= j < q_len as int && j < m < q_len as int ==> slot_a[m] != slot_a[j],
        forall|j: int, m: int| #![trigger slot_b[m], slot_b[j]]
            0 <= j < q_len as int && j < m < q_len as int ==> slot_b[m] != slot_b[j],
        // (F″) fresh slots in the pre-store caches.
        forall|j: int| 0 <= j < q_len as int ==>
            crate::proof::tensor::geometry::slot_in_cache(k_cache_a, #[trigger] (slot_a[j]) as nat)
            && crate::proof::tensor::geometry::slot_in_cache(v_cache_a, (slot_a[j]) as nat)
            && crate::proof::tensor::geometry::slot_in_cache(k_cache_b, #[trigger] (slot_b[j]) as nat)
            && crate::proof::tensor::geometry::slot_in_cache(v_cache_b, (slot_b[j]) as nat),
        // (P) pre-store cache agreement on the cached prefix (engine_kv_coherent).
        forall|pos: nat| #![trigger crate::proof::tensor::geometry::block_table_slot(bt_row_a, pos)]
            pos < k_len - q_len ==>
                crate::proof::tensor::geometry::slot_in_cache(k_cache_a, crate::proof::tensor::geometry::block_table_slot(bt_row_a, pos))
                && crate::proof::tensor::geometry::slot_in_cache(v_cache_a, crate::proof::tensor::geometry::block_table_slot(bt_row_a, pos))
                && crate::proof::tensor::geometry::slot_in_cache(k_cache_b, crate::proof::tensor::geometry::block_table_slot(bt_row_b, pos))
                && crate::proof::tensor::geometry::slot_in_cache(v_cache_b, crate::proof::tensor::geometry::block_table_slot(bt_row_b, pos))
                && crate::proof::tensor::geometry::cache_at(k_cache_a, crate::proof::tensor::geometry::block_table_slot(bt_row_a, pos))
                    == crate::proof::tensor::geometry::cache_at(k_cache_b, crate::proof::tensor::geometry::block_table_slot(bt_row_b, pos))
                && crate::proof::tensor::geometry::cache_at(v_cache_a, crate::proof::tensor::geometry::block_table_slot(bt_row_a, pos))
                    == crate::proof::tensor::geometry::cache_at(v_cache_b, crate::proof::tensor::geometry::block_table_slot(bt_row_b, pos)),
        // (P′) fresh writes miss the cached read slots.
        fresh_writes_miss_cached_prefix(
            slot_a, bt_row_a, slot_b, bt_row_b, k_len - q_len),
    ensures
        DS::decoder_core_output_repr(config, wr, normed_repr, residual_repr, positions_repr,
            k_cache_a, v_cache_a, slot_a, seq![0int, q_len as int], seq![0int, k_len as int],
            q_len, k_len, seq![bt_row_a])
        == DS::decoder_core_output_repr(config, wr, normed_repr, residual_repr, positions_repr,
            k_cache_b, v_cache_b, slot_b, seq![0int, q_len as int], seq![0int, k_len as int],
            q_len, k_len, seq![bt_row_b]),
{
    DS::lemma_pre_attention_repr_shape(config, wr, normed_repr, positions_repr);
    let pre = DS::pre_attention_repr(config, wr, normed_repr, positions_repr);
    let vr0 = RT::qkv_linear_repr(
        normed_repr, wr.q_proj, wr.k_proj, wr.v_proj,
    ).2;
    RT::lemma_qkv_linear_repr_shape(
        normed_repr, wr.q_proj, wr.k_proj, wr.v_proj,
    );
    let vvr = RT::view_as_kv_repr(vr0);
    RT::lemma_view_as_kv_repr_shape(vr0);
    // pre.1 (K) and vvr (V) are the stored rows; both have length q_len.
    assert(pre.1.len() == q_len);
    assert(vvr.len() == q_len);

    // Assemble decoder_core_relocation's per-position post-store agreement.
    assert forall|pos: nat| #![trigger block_table_slot(bt_row_a, pos)]
        pos < k_len implies {
            let post_a = DS::layer_kv_update_repr(config, wr, normed_repr, positions_repr,
                k_cache_a, v_cache_a, slot_a);
            let post_b = DS::layer_kv_update_repr(config, wr, normed_repr, positions_repr,
                k_cache_b, v_cache_b, slot_b);
            (pos as int) / (BLOCK_SIZE_SPEC as int) < bt_row_a.len()
            && (pos as int) / (BLOCK_SIZE_SPEC as int) < bt_row_b.len()
            && slot_in_cache(post_a.0, block_table_slot(bt_row_a, pos))
            && slot_in_cache(post_a.1, block_table_slot(bt_row_a, pos))
            && slot_in_cache(post_b.0, block_table_slot(bt_row_b, pos))
            && slot_in_cache(post_b.1, block_table_slot(bt_row_b, pos))
            && cache_at(post_a.0, block_table_slot(bt_row_a, pos))
                == cache_at(post_b.0, block_table_slot(bt_row_b, pos))
            && cache_at(post_a.1, block_table_slot(bt_row_a, pos))
                == cache_at(post_b.1, block_table_slot(bt_row_b, pos))
        } by {
        crate::proof::tensor::geometry::lemma_blocks_needed_covers_pos(pos, k_len);
        if pos < k_len - q_len {
            fresh_writes_miss_cached_prefix_at(
                slot_a, bt_row_a, slot_b, bt_row_b, k_len - q_len, pos);
            RT::prefix_positions_relocation_agree(pre.1, vvr,
                k_cache_a, v_cache_a, slot_a, bt_row_a,
                k_cache_b, v_cache_b, slot_b, bt_row_b, pos);
        } else {
            let j = (pos - (k_len - q_len)) as int;
            assert((k_len - q_len + j as nat) as nat == pos);
            RT::fresh_positions_relocation_agree(pre.1, vvr,
                k_cache_a, v_cache_a, slot_a, bt_row_a,
                k_cache_b, v_cache_b, slot_b, bt_row_b, q_len, k_len, j);
        }
    }
    decoder_core_relocation(config, wr, normed_repr, residual_repr, positions_repr,
        k_cache_a, v_cache_a, slot_a, bt_row_a,
        k_cache_b, v_cache_b, slot_b, bt_row_b, q_len, k_len);
}

// `decoder_layer` (subsequent-layer) relocation from layout: thin wrapper over
// `decoder_core_relocation_from_layout` with `normed = add_rms_norm(hidden,
// residual).0`.  The layout hypotheses are input-independent so they pass through.
#[verifier::spinoff_prover]
pub proof fn decoder_layer_relocation_from_layout(
    config: DenseSwiGluForwardConfigRepr,
    wr: LayerWeightsRepr,
    hidden_repr: Tensor2D,
    residual_repr: Tensor2D,
    positions_repr: IntTensor1D,
    k_cache_a: KVCacheLayerRepr,
    v_cache_a: KVCacheLayerRepr,
    slot_a: Seq<int>,
    bt_row_a: Seq<BlockId>,
    k_cache_b: KVCacheLayerRepr,
    v_cache_b: KVCacheLayerRepr,
    slot_b: Seq<int>,
    bt_row_b: Seq<BlockId>,
    q_len: nat,
    k_len: nat,
)
    requires
        RT::paged_attention_numeric_domain(),
        hidden_repr.len() == residual_repr.len(),
        hidden_repr.len() == positions_repr.len(),
        slot_a.len() == hidden_repr.len(),
        slot_b.len() == hidden_repr.len(),
        hidden_repr.len() == q_len,
        q_len > 0,
        q_len <= k_len,
        blocks_needed_for(k_len) <= bt_row_a.len(),
        blocks_needed_for(k_len) <= bt_row_b.len(),
        forall|j: int| 0 <= j < q_len as int ==>
            crate::proof::tensor::geometry::block_table_slot(bt_row_a, (k_len - q_len + j as nat) as nat)
                == #[trigger] (slot_a[j]) as nat
            && crate::proof::tensor::geometry::block_table_slot(bt_row_b, (k_len - q_len + j as nat) as nat)
                == (slot_b[j]) as nat,
        forall|j: int| 0 <= j < q_len as int ==>
            #[trigger] (slot_a[j]) >= 0 && (slot_b[j]) >= 0,
        forall|j: int, m: int| #![trigger slot_a[m], slot_a[j]]
            0 <= j < q_len as int && j < m < q_len as int ==> slot_a[m] != slot_a[j],
        forall|j: int, m: int| #![trigger slot_b[m], slot_b[j]]
            0 <= j < q_len as int && j < m < q_len as int ==> slot_b[m] != slot_b[j],
        forall|j: int| 0 <= j < q_len as int ==>
            crate::proof::tensor::geometry::slot_in_cache(k_cache_a, #[trigger] (slot_a[j]) as nat)
            && crate::proof::tensor::geometry::slot_in_cache(v_cache_a, (slot_a[j]) as nat)
            && crate::proof::tensor::geometry::slot_in_cache(k_cache_b, #[trigger] (slot_b[j]) as nat)
            && crate::proof::tensor::geometry::slot_in_cache(v_cache_b, (slot_b[j]) as nat),
        forall|pos: nat| #![trigger crate::proof::tensor::geometry::block_table_slot(bt_row_a, pos)]
            pos < k_len - q_len ==>
                crate::proof::tensor::geometry::slot_in_cache(k_cache_a, crate::proof::tensor::geometry::block_table_slot(bt_row_a, pos))
                && crate::proof::tensor::geometry::slot_in_cache(v_cache_a, crate::proof::tensor::geometry::block_table_slot(bt_row_a, pos))
                && crate::proof::tensor::geometry::slot_in_cache(k_cache_b, crate::proof::tensor::geometry::block_table_slot(bt_row_b, pos))
                && crate::proof::tensor::geometry::slot_in_cache(v_cache_b, crate::proof::tensor::geometry::block_table_slot(bt_row_b, pos))
                && crate::proof::tensor::geometry::cache_at(k_cache_a, crate::proof::tensor::geometry::block_table_slot(bt_row_a, pos))
                    == crate::proof::tensor::geometry::cache_at(k_cache_b, crate::proof::tensor::geometry::block_table_slot(bt_row_b, pos))
                && crate::proof::tensor::geometry::cache_at(v_cache_a, crate::proof::tensor::geometry::block_table_slot(bt_row_a, pos))
                    == crate::proof::tensor::geometry::cache_at(v_cache_b, crate::proof::tensor::geometry::block_table_slot(bt_row_b, pos)),
        fresh_writes_miss_cached_prefix(
            slot_a, bt_row_a, slot_b, bt_row_b, k_len - q_len),
    ensures
        DS::decoder_layer_output_repr(config, wr, hidden_repr, residual_repr, positions_repr,
            k_cache_a, v_cache_a, slot_a, seq![0int, q_len as int], seq![0int, k_len as int],
            q_len, k_len, seq![bt_row_a])
        == DS::decoder_layer_output_repr(config, wr, hidden_repr, residual_repr, positions_repr,
            k_cache_b, v_cache_b, slot_b, seq![0int, q_len as int], seq![0int, k_len as int],
            q_len, k_len, seq![bt_row_b]),
{
    broadcast use RT::lemma_add_rms_norm_repr_shape;
    let pre = DS::add_rms_norm_repr(config, hidden_repr, residual_repr, wr.input_norm);
    assert forall|j: int| 0 <= j < q_len as int implies
        crate::proof::tensor::geometry::block_table_slot(bt_row_a, (k_len - q_len + j as nat) as nat)
            == #[trigger] (slot_a[j]) as nat
        && crate::proof::tensor::geometry::block_table_slot(bt_row_b, (k_len - q_len + j as nat) as nat)
            == (slot_b[j]) as nat by {
        assert(crate::proof::tensor::geometry::block_table_slot(
            bt_row_a, (k_len - q_len + j as nat) as nat,
        ) == (slot_a[j]) as nat);
        assert(crate::proof::tensor::geometry::block_table_slot(
            bt_row_b, (k_len - q_len + j as nat) as nat,
        ) == (slot_b[j]) as nat);
    }
    decoder_core_relocation_from_layout(config, wr, pre.0, pre.1, positions_repr,
        k_cache_a, v_cache_a, slot_a, bt_row_a,
        k_cache_b, v_cache_b, slot_b, bt_row_b, q_len, k_len);
}

// `first_decoder_layer` (layer-0) relocation from layout: thin wrapper with
// `normed = rms_norm(hidden).0`, residual == hidden.
pub proof fn first_decoder_layer_relocation_from_layout(
    config: DenseSwiGluForwardConfigRepr,
    wr: LayerWeightsRepr,
    hidden_repr: Tensor2D,
    positions_repr: IntTensor1D,
    k_cache_a: KVCacheLayerRepr,
    v_cache_a: KVCacheLayerRepr,
    slot_a: Seq<int>,
    bt_row_a: Seq<BlockId>,
    k_cache_b: KVCacheLayerRepr,
    v_cache_b: KVCacheLayerRepr,
    slot_b: Seq<int>,
    bt_row_b: Seq<BlockId>,
    q_len: nat,
    k_len: nat,
)
    requires
        RT::paged_attention_numeric_domain(),
        hidden_repr.len() == positions_repr.len(),
        slot_a.len() == hidden_repr.len(),
        slot_b.len() == hidden_repr.len(),
        hidden_repr.len() == q_len,
        q_len > 0,
        q_len <= k_len,
        blocks_needed_for(k_len) <= bt_row_a.len(),
        blocks_needed_for(k_len) <= bt_row_b.len(),
        forall|j: int| 0 <= j < q_len as int ==>
            crate::proof::tensor::geometry::block_table_slot(bt_row_a, (k_len - q_len + j as nat) as nat)
                == #[trigger] (slot_a[j]) as nat
            && crate::proof::tensor::geometry::block_table_slot(bt_row_b, (k_len - q_len + j as nat) as nat)
                == (slot_b[j]) as nat,
        forall|j: int| 0 <= j < q_len as int ==>
            #[trigger] (slot_a[j]) >= 0 && (slot_b[j]) >= 0,
        forall|j: int, m: int| #![trigger slot_a[m], slot_a[j]]
            0 <= j < q_len as int && j < m < q_len as int ==> slot_a[m] != slot_a[j],
        forall|j: int, m: int| #![trigger slot_b[m], slot_b[j]]
            0 <= j < q_len as int && j < m < q_len as int ==> slot_b[m] != slot_b[j],
        forall|j: int| 0 <= j < q_len as int ==>
            crate::proof::tensor::geometry::slot_in_cache(k_cache_a, #[trigger] (slot_a[j]) as nat)
            && crate::proof::tensor::geometry::slot_in_cache(v_cache_a, (slot_a[j]) as nat)
            && crate::proof::tensor::geometry::slot_in_cache(k_cache_b, #[trigger] (slot_b[j]) as nat)
            && crate::proof::tensor::geometry::slot_in_cache(v_cache_b, (slot_b[j]) as nat),
        forall|pos: nat| #![trigger crate::proof::tensor::geometry::block_table_slot(bt_row_a, pos)]
            pos < k_len - q_len ==>
                crate::proof::tensor::geometry::slot_in_cache(k_cache_a, crate::proof::tensor::geometry::block_table_slot(bt_row_a, pos))
                && crate::proof::tensor::geometry::slot_in_cache(v_cache_a, crate::proof::tensor::geometry::block_table_slot(bt_row_a, pos))
                && crate::proof::tensor::geometry::slot_in_cache(k_cache_b, crate::proof::tensor::geometry::block_table_slot(bt_row_b, pos))
                && crate::proof::tensor::geometry::slot_in_cache(v_cache_b, crate::proof::tensor::geometry::block_table_slot(bt_row_b, pos))
                && crate::proof::tensor::geometry::cache_at(k_cache_a, crate::proof::tensor::geometry::block_table_slot(bt_row_a, pos))
                    == crate::proof::tensor::geometry::cache_at(k_cache_b, crate::proof::tensor::geometry::block_table_slot(bt_row_b, pos))
                && crate::proof::tensor::geometry::cache_at(v_cache_a, crate::proof::tensor::geometry::block_table_slot(bt_row_a, pos))
                    == crate::proof::tensor::geometry::cache_at(v_cache_b, crate::proof::tensor::geometry::block_table_slot(bt_row_b, pos)),
        fresh_writes_miss_cached_prefix(
            slot_a, bt_row_a, slot_b, bt_row_b, k_len - q_len),
    ensures
        DS::first_decoder_layer_output_repr(config, wr, hidden_repr, positions_repr,
            k_cache_a, v_cache_a, slot_a, seq![0int, q_len as int], seq![0int, k_len as int],
            q_len, k_len, seq![bt_row_a])
        == DS::first_decoder_layer_output_repr(config, wr, hidden_repr, positions_repr,
            k_cache_b, v_cache_b, slot_b, seq![0int, q_len as int], seq![0int, k_len as int],
            q_len, k_len, seq![bt_row_b]),
{
    broadcast use RT::lemma_rms_norm_repr_shape;
    let normed = DS::rms_norm_repr(config, hidden_repr, wr.input_norm);
    decoder_core_relocation_from_layout(config, wr, normed, hidden_repr, positions_repr,
        k_cache_a, v_cache_a, slot_a, bt_row_a,
        k_cache_b, v_cache_b, slot_b, bt_row_b, q_len, k_len);
}


// G3′ chain fold: the single-request layer chain is invariant under cache-geometry
// swap.  Both sides stay in lockstep (each layer's output is fully equal by
// `decoder_layer_relocation_from_layout`), so this is simpler than the isolation
// fold — no slicing, full equality.  Per-layer cache hypotheses (F″ fresh-in-cache,
// P prefix agreement) range over the remaining layers `[start, n)`; the geometry
// layout facts (F alignment, F′ distinct fresh, P′ fresh-miss-cached) are
// layer-independent.
#[verifier::spinoff_prover]
pub proof fn layer_chain_relocation(
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
)
    requires
        RT::paged_attention_numeric_domain(),
        start <= layers.len(),
        kv_a.len() >= layers.len(),
        kv_b.len() >= layers.len(),
        hidden_repr.len() == residual_repr.len(),
        hidden_repr.len() == positions_repr.len(),
        slot_a.len() == hidden_repr.len(),
        slot_b.len() == hidden_repr.len(),
        hidden_repr.len() == q_len,
        q_len > 0,
        q_len <= k_len,
        blocks_needed_for(k_len) <= bt_row_a.len(),
        blocks_needed_for(k_len) <= bt_row_b.len(),
        // (F) alignment, (F′) distinct fresh, (P′) fresh-miss-cached — layer-independent.
        forall|j: int| 0 <= j < q_len as int ==>
            crate::proof::tensor::geometry::block_table_slot(bt_row_a, (k_len - q_len + j as nat) as nat)
                == #[trigger] (slot_a[j]) as nat
            && crate::proof::tensor::geometry::block_table_slot(bt_row_b, (k_len - q_len + j as nat) as nat)
                == (slot_b[j]) as nat,
        forall|j: int| 0 <= j < q_len as int ==>
            #[trigger] (slot_a[j]) >= 0 && (slot_b[j]) >= 0,
        forall|j: int, m: int| #![trigger slot_a[m], slot_a[j]]
            0 <= j < q_len as int && j < m < q_len as int ==> slot_a[m] != slot_a[j],
        forall|j: int, m: int| #![trigger slot_b[m], slot_b[j]]
            0 <= j < q_len as int && j < m < q_len as int ==> slot_b[m] != slot_b[j],
        fresh_writes_miss_cached_prefix(
            slot_a, bt_row_a, slot_b, bt_row_b, k_len - q_len),
        // (F″) fresh slots in each remaining layer's pre-store caches.
        forall|ell: int, j: int| #![trigger kv_a[ell].0, slot_a[j]]
            start <= ell < layers.len() && 0 <= j < q_len as int ==>
                crate::proof::tensor::geometry::slot_in_cache(kv_a[ell].0, (slot_a[j]) as nat)
                && crate::proof::tensor::geometry::slot_in_cache(kv_a[ell].1, (slot_a[j]) as nat)
                && crate::proof::tensor::geometry::slot_in_cache(kv_b[ell].0, (slot_b[j]) as nat)
                && crate::proof::tensor::geometry::slot_in_cache(kv_b[ell].1, (slot_b[j]) as nat),
        // (P) prefix pre-store agreement at each remaining layer.
        forall|ell: int, pos: nat|
            #![trigger kv_a[ell].0, crate::proof::tensor::geometry::block_table_slot(bt_row_a, pos)]
            start <= ell < layers.len() && pos < k_len - q_len ==>
                crate::proof::tensor::geometry::slot_in_cache(kv_a[ell].0, crate::proof::tensor::geometry::block_table_slot(bt_row_a, pos))
                && crate::proof::tensor::geometry::slot_in_cache(kv_a[ell].1, crate::proof::tensor::geometry::block_table_slot(bt_row_a, pos))
                && crate::proof::tensor::geometry::slot_in_cache(kv_b[ell].0, crate::proof::tensor::geometry::block_table_slot(bt_row_b, pos))
                && crate::proof::tensor::geometry::slot_in_cache(kv_b[ell].1, crate::proof::tensor::geometry::block_table_slot(bt_row_b, pos))
                && crate::proof::tensor::geometry::cache_at(kv_a[ell].0, crate::proof::tensor::geometry::block_table_slot(bt_row_a, pos))
                    == crate::proof::tensor::geometry::cache_at(kv_b[ell].0, crate::proof::tensor::geometry::block_table_slot(bt_row_b, pos))
                && crate::proof::tensor::geometry::cache_at(kv_a[ell].1, crate::proof::tensor::geometry::block_table_slot(bt_row_a, pos))
                    == crate::proof::tensor::geometry::cache_at(kv_b[ell].1, crate::proof::tensor::geometry::block_table_slot(bt_row_b, pos)),
    ensures
        DS::layer_chain_repr(config, layers, hidden_repr, residual_repr, positions_repr,
            kv_a, slot_a, seq![0int, q_len as int], seq![0int, k_len as int],
            q_len, k_len, seq![bt_row_a], start)
        == DS::layer_chain_repr(config, layers, hidden_repr, residual_repr, positions_repr,
            kv_b, slot_b, seq![0int, q_len as int], seq![0int, k_len as int],
            q_len, k_len, seq![bt_row_b], start),
    decreases (layers.len() - start) as nat,
{
    if start >= layers.len() {
    } else {
        // Layer `start`: full equality of the two geometries' outputs.
        assert forall|j: int| 0 <= j < q_len as int implies
            crate::proof::tensor::geometry::block_table_slot(bt_row_a, (k_len - q_len + j as nat) as nat)
                == #[trigger] (slot_a[j]) as nat
            && crate::proof::tensor::geometry::block_table_slot(bt_row_b, (k_len - q_len + j as nat) as nat)
                == (slot_b[j]) as nat by {
            assert(crate::proof::tensor::geometry::block_table_slot(
                bt_row_a, (k_len - q_len + j as nat) as nat,
            ) == (slot_a[j]) as nat);
            assert(crate::proof::tensor::geometry::block_table_slot(
                bt_row_b, (k_len - q_len + j as nat) as nat,
            ) == (slot_b[j]) as nat);
        }
        decoder_layer_relocation_from_layout(config, layers[start as int], hidden_repr, residual_repr,
            positions_repr, kv_a[start as int].0, kv_a[start as int].1, slot_a, bt_row_a,
            kv_b[start as int].0, kv_b[start as int].1, slot_b, bt_row_b, q_len, k_len);
        let next = DS::decoder_layer_output_repr(config, layers[start as int], hidden_repr, residual_repr,
            positions_repr, kv_a[start as int].0, kv_a[start as int].1, slot_a,
            seq![0int, q_len as int], seq![0int, k_len as int], q_len, k_len, seq![bt_row_a]);
        DS::lemma_decoder_layer_output_repr_shape(config, layers[start as int], hidden_repr, residual_repr,
            positions_repr, kv_a[start as int].0, kv_a[start as int].1, slot_a,
            seq![0int, q_len as int], seq![0int, k_len as int], q_len, k_len, seq![bt_row_a]);
        // Tail folds over the (equal) next state.
        layer_chain_relocation(config, layers, next.0, next.1, positions_repr,
            kv_a, slot_a, bt_row_a, kv_b, slot_b, bt_row_b, q_len, k_len, (start + 1) as nat);
    }
}

// G3′ capstone: whole-model geometry relocation.  The single-request
// `model_forward_logits_repr` over engine paged geometry (`slot_a`, `bt_row_a`)
// equals the one over machine contiguous geometry (`slot_b`, `bt_row_b`), given
// per-layer post-store K/V agreement (supplied input-independently by the layout
// facts).  Assembles: embed (identical) → first-layer relocation → chain
// relocation → final-norm/lm_head (identical functions of equal inputs).
pub proof fn model_forward_relocation(
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
)
    requires
        RT::paged_attention_numeric_domain(),
        wr.layers.len() > 0,
        kv_a.len() >= wr.layers.len(),
        kv_b.len() >= wr.layers.len(),
        input_ids_repr.len() == positions_repr.len(),
        slot_a.len() == input_ids_repr.len(),
        slot_b.len() == input_ids_repr.len(),
        input_ids_repr.len() == q_len,
        q_len > 0,
        q_len <= k_len,
        blocks_needed_for(k_len) <= bt_row_a.len(),
        blocks_needed_for(k_len) <= bt_row_b.len(),
        forall|j: int| 0 <= j < q_len as int ==>
            crate::proof::tensor::geometry::block_table_slot(bt_row_a, (k_len - q_len + j as nat) as nat)
                == #[trigger] (slot_a[j]) as nat
            && crate::proof::tensor::geometry::block_table_slot(bt_row_b, (k_len - q_len + j as nat) as nat)
                == (slot_b[j]) as nat,
        forall|j: int| 0 <= j < q_len as int ==>
            #[trigger] (slot_a[j]) >= 0 && (slot_b[j]) >= 0,
        forall|j: int, m: int| #![trigger slot_a[m], slot_a[j]]
            0 <= j < q_len as int && j < m < q_len as int ==> slot_a[m] != slot_a[j],
        forall|j: int, m: int| #![trigger slot_b[m], slot_b[j]]
            0 <= j < q_len as int && j < m < q_len as int ==> slot_b[m] != slot_b[j],
        fresh_writes_miss_cached_prefix(
            slot_a, bt_row_a, slot_b, bt_row_b, k_len - q_len),
        forall|ell: int, j: int| #![trigger kv_a[ell].0, slot_a[j]]
            0 <= ell < wr.layers.len() && 0 <= j < q_len as int ==>
                crate::proof::tensor::geometry::slot_in_cache(kv_a[ell].0, (slot_a[j]) as nat)
                && crate::proof::tensor::geometry::slot_in_cache(kv_a[ell].1, (slot_a[j]) as nat)
                && crate::proof::tensor::geometry::slot_in_cache(kv_b[ell].0, (slot_b[j]) as nat)
                && crate::proof::tensor::geometry::slot_in_cache(kv_b[ell].1, (slot_b[j]) as nat),
        forall|ell: int, pos: nat|
            #![trigger kv_a[ell].0, crate::proof::tensor::geometry::block_table_slot(bt_row_a, pos)]
            0 <= ell < wr.layers.len() && pos < k_len - q_len ==>
                crate::proof::tensor::geometry::slot_in_cache(kv_a[ell].0, crate::proof::tensor::geometry::block_table_slot(bt_row_a, pos))
                && crate::proof::tensor::geometry::slot_in_cache(kv_a[ell].1, crate::proof::tensor::geometry::block_table_slot(bt_row_a, pos))
                && crate::proof::tensor::geometry::slot_in_cache(kv_b[ell].0, crate::proof::tensor::geometry::block_table_slot(bt_row_b, pos))
                && crate::proof::tensor::geometry::slot_in_cache(kv_b[ell].1, crate::proof::tensor::geometry::block_table_slot(bt_row_b, pos))
                && crate::proof::tensor::geometry::cache_at(kv_a[ell].0, crate::proof::tensor::geometry::block_table_slot(bt_row_a, pos))
                    == crate::proof::tensor::geometry::cache_at(kv_b[ell].0, crate::proof::tensor::geometry::block_table_slot(bt_row_b, pos))
                && crate::proof::tensor::geometry::cache_at(kv_a[ell].1, crate::proof::tensor::geometry::block_table_slot(bt_row_a, pos))
                    == crate::proof::tensor::geometry::cache_at(kv_b[ell].1, crate::proof::tensor::geometry::block_table_slot(bt_row_b, pos)),
    ensures
        DS::model_forward_logits_repr(config, wr, input_ids_repr, positions_repr, kv_a, slot_a,
            seq![0int, q_len as int], seq![0int, k_len as int], q_len, k_len, seq![bt_row_a])
        == DS::model_forward_logits_repr(config, wr, input_ids_repr, positions_repr, kv_b, slot_b,
            seq![0int, q_len as int], seq![0int, k_len as int], q_len, k_len, seq![bt_row_b]),
{
    reveal(DS::model_forward_logits_repr);
    broadcast use {
        RT::lemma_embed_repr_shape,
        RT::lemma_add_rms_norm_repr_shape,
    };
    let embed = RT::embed_repr(input_ids_repr, wr.embed_weight);
    // Layer 0: identical embed input, geometry swap.
    first_decoder_layer_relocation_from_layout(config, wr.layers[0], embed, positions_repr,
        kv_a[0].0, kv_a[0].1, slot_a, bt_row_a, kv_b[0].0, kv_b[0].1, slot_b, bt_row_b,
        q_len, k_len);
    let first = DS::first_decoder_layer_output_repr(config, wr.layers[0], embed, positions_repr,
        kv_a[0].0, kv_a[0].1, slot_a,
        seq![0int, q_len as int], seq![0int, k_len as int], q_len, k_len, seq![bt_row_a]);
    DS::lemma_first_decoder_layer_output_repr_shape(config, wr.layers[0], embed, positions_repr,
        kv_a[0].0, kv_a[0].1, slot_a,
        seq![0int, q_len as int], seq![0int, k_len as int], q_len, k_len, seq![bt_row_a]);
    // Layers 1..n: chain relocation over the (equal) first-layer output.
    assert forall|ell: int, j: int| #![trigger kv_a[ell].0, slot_a[j]]
        1 <= ell < wr.layers.len() && 0 <= j < q_len as int implies
            crate::proof::tensor::geometry::slot_in_cache(kv_a[ell].0, (slot_a[j]) as nat)
            && crate::proof::tensor::geometry::slot_in_cache(kv_a[ell].1, (slot_a[j]) as nat)
            && crate::proof::tensor::geometry::slot_in_cache(kv_b[ell].0, (slot_b[j]) as nat)
            && crate::proof::tensor::geometry::slot_in_cache(kv_b[ell].1, (slot_b[j]) as nat) by {}
    assert forall|ell: int, pos: nat|
        #![trigger kv_a[ell].0, crate::proof::tensor::geometry::block_table_slot(bt_row_a, pos)]
        1 <= ell < wr.layers.len() && pos < k_len - q_len implies
            crate::proof::tensor::geometry::slot_in_cache(
                kv_a[ell].0, crate::proof::tensor::geometry::block_table_slot(bt_row_a, pos),
            )
            && crate::proof::tensor::geometry::slot_in_cache(
                kv_a[ell].1, crate::proof::tensor::geometry::block_table_slot(bt_row_a, pos),
            )
            && crate::proof::tensor::geometry::slot_in_cache(
                kv_b[ell].0, crate::proof::tensor::geometry::block_table_slot(bt_row_b, pos),
            )
            && crate::proof::tensor::geometry::slot_in_cache(
                kv_b[ell].1, crate::proof::tensor::geometry::block_table_slot(bt_row_b, pos),
            )
            && crate::proof::tensor::geometry::cache_at(
                kv_a[ell].0, crate::proof::tensor::geometry::block_table_slot(bt_row_a, pos),
            ) == crate::proof::tensor::geometry::cache_at(
                kv_b[ell].0, crate::proof::tensor::geometry::block_table_slot(bt_row_b, pos),
            )
            && crate::proof::tensor::geometry::cache_at(
                kv_a[ell].1, crate::proof::tensor::geometry::block_table_slot(bt_row_a, pos),
            ) == crate::proof::tensor::geometry::cache_at(
                kv_b[ell].1, crate::proof::tensor::geometry::block_table_slot(bt_row_b, pos),
            ) by {}
    layer_chain_relocation(config, wr.layers, first.0, first.1, positions_repr,
        kv_a, slot_a, bt_row_a, kv_b, slot_b, bt_row_b, q_len, k_len, 1);
    // final-norm + lm_head are functions of the (equal) chain output.
}

} // verus!
