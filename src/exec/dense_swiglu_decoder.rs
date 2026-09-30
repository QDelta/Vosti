//! Checked family-neutral execution of one full-attention SwiGLU layer.
//!
//! Family adapters bind physical weights and a qualified runtime to an exact
//! closed configuration. This module owns only the common kernel composition
//! and proves it refines `proof::model::dense_swiglu::semantics`.

use crate::boundary::dense_layer_primitives as DLP;
use crate::boundary::dense_swiglu_decoder as WEIGHTS;
#[cfg(verus_only)]
use crate::proof::model::dense_swiglu::semantics as SEM;
use crate::{types::*, proof::model::types::*, proof::tensor::types::*};
use crate::boundary::tensor_runtime as RT;
use crate::proof::tensor::shape as TS;
use vstd::prelude::*;

verus! {

#[verifier::opaque]
pub open spec fn forward_ready(
    wp: &RT::ModelWeightsPerms,
    input_ids_repr: IntTensor1D,
    pre_kv_reprs: Seq<(KVCacheLayerRepr, KVCacheLayerRepr)>,
    slot_repr: Seq<int>,
    cu_q_repr: Seq<int>,
    cu_k_repr: Seq<int>,
    max_seqlen_q: nat,
    max_seqlen_k: nat,
    bt_repr: Seq<Seq<BlockId>>,
) -> bool {
    &&& forall|i: int| 0 <= i < wp.num_layers() as int ==>
        #[trigger] RT::store_kv_cache_launch_ready(
            input_ids_repr.len(), pre_kv_reprs[i].0,
            pre_kv_reprs[i].1, slot_repr,
        )
    &&& forall|i: int| 0 <= i < wp.num_layers() as int ==>
        #[trigger] RT::paged_attention_launch_ready(
            input_ids_repr.len(), pre_kv_reprs[i].0,
            pre_kv_reprs[i].1, cu_q_repr, cu_k_repr,
            max_seqlen_q, max_seqlen_k, bt_repr,
        )
}

pub fn decoder_core_forward(
    runtime: &RT::ModelFamilyRuntime,
    w: &WEIGHTS::DenseSwiGluLayerWeights,
    Tracked(wp): Tracked<&WEIGHTS::DenseSwiGluLayerWeightsPerms>,
    Ghost(head_dim): Ghost<nat>,
    Ghost(config): Ghost<DenseSwiGluForwardConfigRepr>,
    normed: &RT::Tensor,
    Tracked(np): Tracked<&RT::TensorPerm>,
    post_attn_residual: &RT::Tensor,
    Tracked(rp): Tracked<&RT::TensorPerm>,
    positions: &RT::Tensor,
    Tracked(pp): Tracked<&RT::TensorPerm>,
    k_cache: &RT::Tensor,
    Tracked(kc_perm): Tracked<&mut RT::TensorPerm>,
    v_cache: &RT::Tensor,
    Tracked(vc_perm): Tracked<&mut RT::TensorPerm>,
    block_table: &RT::Tensor,
    Tracked(bt_perm): Tracked<&RT::TensorPerm>,
    slot_mapping: &RT::Tensor,
    Tracked(sp): Tracked<&RT::TensorPerm>,
    cu_seqlens_q: &RT::Tensor,
    Tracked(cuq_perm): Tracked<&RT::TensorPerm>,
    cu_seqlens_k: &RT::Tensor,
    Tracked(cuk_perm): Tracked<&RT::TensorPerm>,
    max_seqlen_q: usize,
    max_seqlen_k: usize,
    Ghost(normed_repr): Ghost<Tensor2D>,
    Ghost(residual_repr): Ghost<Tensor2D>,
    Ghost(positions_repr): Ghost<IntTensor1D>,
    Ghost(cu_q_repr): Ghost<Seq<int>>,
    Ghost(cu_k_repr): Ghost<Seq<int>>,
    Ghost(bt_repr): Ghost<Seq<Seq<BlockId>>>,
    Ghost(k_cache_repr): Ghost<KVCacheLayerRepr>,
    Ghost(v_cache_repr): Ghost<KVCacheLayerRepr>,
    Ghost(slot_repr): Ghost<Seq<int>>,
    Ghost(scope): Ghost<Set<RT::TensorId>>,
) -> (out: (RT::Tensor, Tracked<RT::TensorPerm>, RT::Tensor, Tracked<RT::TensorPerm>))
    requires
        RT::paged_attention_numeric_domain(),
        RT::family_runtime_execution_valid(runtime),
        WEIGHTS::layer_weights_valid(w, wp),
        RT::dense_swiglu_runtime_rms_norm_matches(
            runtime, config.rms_norm_epsilon,
        ),
        RT::dense_swiglu_runtime_qk_norm_matches(
            runtime,
            qk_norm_weights_kind(WEIGHTS::qk_norm_weights_repr_of(&wp.qk_norm)),
        ),
        RT::dense_swiglu_runtime_rotary_matches(
            runtime,
            DLP::layer_attention_geometry_repr(
                WEIGHTS::layer_weights_repr_of(wp, head_dim),
            ),
            config.rotary,
        ),
        RT::tensor_repr_2d(*np, *normed, normed_repr),
        RT::tensor_repr_2d(*rp, *post_attn_residual, residual_repr),
        RT::int_tensor_repr_1d(*pp, *positions, positions_repr),
        RT::kv_cache_tensor_repr(*old(kc_perm), *k_cache, k_cache_repr),
        RT::kv_cache_tensor_repr(*old(vc_perm), *v_cache, v_cache_repr),
        RT::block_table_repr(*bt_perm, *block_table, bt_repr),
        RT::int_tensor_repr_1d(*sp, *slot_mapping, slot_repr),
        RT::int_tensor_repr_1d(*cuq_perm, *cu_seqlens_q, cu_q_repr),
        RT::int_tensor_repr_1d(*cuk_perm, *cu_seqlens_k, cu_k_repr),
        normed_repr.len() == residual_repr.len(),
        normed_repr.len() == positions_repr.len(),
        slot_repr.len() == normed_repr.len(),
        RT::store_kv_cache_launch_ready(
            normed_repr.len(), k_cache_repr, v_cache_repr, slot_repr,
        ),
        RT::paged_attention_launch_ready(
            normed_repr.len(),
            k_cache_repr,
            v_cache_repr,
            cu_q_repr,
            cu_k_repr,
            max_seqlen_q as nat,
            max_seqlen_k as nat,
            bt_repr,
        ),
    ensures ({
        let (next_hidden, nh_perm, next_residual, nr_perm) = out;
        let wr = WEIGHTS::layer_weights_repr_of(wp, head_dim);
        let core_out = SEM::decoder_core_output_repr(
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
            max_seqlen_q as nat,
            max_seqlen_k as nat,
            bt_repr,
        );
        let kv_out = SEM::layer_kv_update_repr(
            config,
            wr,
            normed_repr,
            positions_repr,
            k_cache_repr,
            v_cache_repr,
            slot_repr,
        );
        &&& next_hidden.id() != next_residual.id()
        &&& RT::tensor_repr_2d(nh_perm@, next_hidden, core_out.0)
        &&& RT::tensor_repr_2d(nr_perm@, next_residual, core_out.1)
        &&& RT::kv_cache_tensor_repr(*final(kc_perm), *k_cache, kv_out.0)
        &&& RT::kv_cache_tensor_repr(*final(vc_perm), *v_cache, kv_out.1)
        &&& final(kc_perm).id() == old(kc_perm).id()
        &&& final(vc_perm).id() == old(vc_perm).id()
    }),
{
    broadcast use {
        RT::lemma_linear_repr_shape,
        RT::lemma_view_as_kv_repr_shape,
        RT::lemma_apply_qk_norm_repr_shape,
        RT::lemma_rotary_embed_repr_shape,
        RT::lemma_paged_attention_repr_shape,
        RT::lemma_add_rms_norm_repr_shape,
        RT::lemma_silu_and_mul_repr_shape,
    };

    let ghost wr = WEIGHTS::layer_weights_repr_of(wp, head_dim);

    let ((q, k, v), (q_p, k_p, v_p)) = RT::qkv_linear(
        runtime,
        normed,
        &w.q_proj,
        &w.k_proj,
        &w.v_proj,
        Tracked(np),
        Tracked(&wp.q_proj),
        Tracked(&wp.k_proj),
        Tracked(&wp.v_proj),
        Ghost(normed_repr),
        Ghost(wp.q_proj.repr_2d()),
        Ghost(wp.k_proj.repr_2d()),
        Ghost(wp.v_proj.repr_2d()),
        Ghost(scope),
    );
    let ghost qkv_repr = RT::qkv_linear_repr(
        normed_repr,
        wp.q_proj.repr_2d(),
        wp.k_proj.repr_2d(),
        wp.v_proj.repr_2d(),
    );
    let ghost q_repr = qkv_repr.0;
    let ghost k_repr = qkv_repr.1;
    let ghost v_repr = qkv_repr.2;
    let scope3 = Ghost(scope.insert(q.id()).insert(k.id()).insert(v.id()));

    let (vv, vv_p) = RT::view_as_kv(
        runtime,
        &v,
        Tracked(v_p.borrow()),
        Ghost(v_repr),
        scope3,
    );
    let scope4 = Ghost(scope3@.insert(vv.id()));

    let (qk_tensors, qk_perms) = WEIGHTS::apply_qk_norm(
        runtime,
        &w.qk_norm,
        Tracked(&wp.qk_norm),
        q,
        k,
        q_p,
        k_p,
        Ghost(q_repr),
        Ghost(k_repr),
        Ghost(config.rms_norm_epsilon),
        scope4,
    );
    let nq_t = qk_tensors.0;
    let nk_t = qk_tensors.1;
    let nq_p = qk_perms.0;
    let nk_p = qk_perms.1;
    let scope5 = Ghost(scope4@.insert(nq_t.id()).insert(nk_t.id()));

    let ghost normalized = RT::apply_qk_norm_repr(
        q_repr,
        k_repr,
        wr.qk_norm,
        config.rms_norm_epsilon,
    );
    let (r_pair, r_perms) = DLP::rotary_embed(
        runtime,
        positions,
        &nq_t,
        &nk_t,
        Tracked(pp),
        Tracked(nq_p.borrow()),
        Tracked(nk_p.borrow()),
        Ghost(positions_repr),
        Ghost(normalized.0),
        Ghost(normalized.1),
        Ghost(DLP::layer_attention_geometry_repr(wr)),
        Ghost(config.rotary),
        scope5,
    );
    let rq_t = r_pair.0;
    let rk_t = r_pair.1;
    let rq_p = r_perms.0;
    let rk_p = r_perms.1;
    let scope6 = Ghost(scope5@.insert(rq_t.id()).insert(rk_t.id()));

    RT::store_kv_cache(
        runtime,
        &rk_t,
        &vv,
        k_cache,
        v_cache,
        slot_mapping,
        Tracked(rk_p.borrow()),
        Tracked(vv_p.borrow()),
        Tracked(kc_perm),
        Tracked(vc_perm),
        Tracked(sp),
        Ghost(SEM::pre_attention_repr(
            config, wr, normed_repr, positions_repr,
        ).1),
        Ghost(RT::view_as_kv_repr(v_repr)),
        Ghost(k_cache_repr),
        Ghost(v_cache_repr),
        Ghost(slot_repr),
    );

    proof {
        let pre = SEM::pre_attention_repr(
            config, wr, normed_repr, positions_repr,
        );
        let vv_repr = RT::view_as_kv_repr(v_repr);
        SEM::lemma_pre_attention_repr_shape(
            config, wr, normed_repr, positions_repr,
        );
        RT::lemma_qkv_linear_repr_shape(
            normed_repr, wp.q_proj.repr_2d(), wp.k_proj.repr_2d(),
            wp.v_proj.repr_2d(),
        );
        RT::lemma_view_as_kv_repr_shape(v_repr);
        RT::lemma_paged_attention_launch_ready_after_store(
            normed_repr.len(),
            pre.1,
            vv_repr,
            k_cache_repr,
            v_cache_repr,
            slot_repr,
            cu_q_repr,
            cu_k_repr,
            max_seqlen_q as nat,
            max_seqlen_k as nat,
            bt_repr,
        );
        assert(pre.0.len() == normed_repr.len());
    }

    let ghost pre = SEM::pre_attention_repr(
        config, wr, normed_repr, positions_repr,
    );
    let ghost stored = RT::store_kv_cache_repr(
        pre.1,
        RT::view_as_kv_repr(v_repr),
        k_cache_repr,
        v_cache_repr,
        slot_repr,
    );
    proof {
        reveal(RT::dense_swiglu_runtime_rotary_matches);
        reveal(RT::dense_swiglu_runtime_attention_matches);
        reveal(DLP::layer_attention_parameters_repr);
        assert(RT::dense_swiglu_runtime_attention_matches(runtime,
            DLP::layer_attention_parameters_repr(wr, AttentionScaleRepr::InverseSqrtHeadDim)));
    }
    let (attn, attn_p) = RT::paged_attention(
        runtime,
        &rq_t,
        k_cache,
        v_cache,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        block_table,
        Tracked(rq_p.borrow()),
        Tracked(kc_perm),
        Tracked(vc_perm),
        Tracked(cuq_perm),
        Tracked(cuk_perm),
        Tracked(bt_perm),
        Ghost(pre.0),
        Ghost(stored.0),
        Ghost(stored.1),
        Ghost(cu_q_repr),
        Ghost(cu_k_repr),
        Ghost(bt_repr),
        scope6,
        Ghost(crate::boundary::dense_layer_primitives::layer_attention_parameters_repr(wr, AttentionScaleRepr::InverseSqrtHeadDim)),
    );
    let scope7 = Ghost(scope6@.insert(attn.id()));

    let attn_repr_g: Ghost<Tensor2D> = Ghost(RT::paged_attention_repr(
        pre.0,
        stored.0,
        stored.1,
        cu_q_repr,
        cu_k_repr,
        max_seqlen_q as nat,
        max_seqlen_k as nat,
        bt_repr,
        crate::boundary::dense_layer_primitives::layer_attention_parameters_repr(wr, AttentionScaleRepr::InverseSqrtHeadDim),
    ));
    let (projected, projected_p) = RT::linear(
        runtime,
        &attn,
        &w.o_proj,
        Tracked(attn_p.borrow()),
        Tracked(&wp.o_proj),
        attn_repr_g,
        Ghost(wp.o_proj.repr_2d()),
        scope7,
    );
    let scope8 = Ghost(scope7@.insert(projected.id()));

    let (norm_pair, norm_perms) = RT::add_rms_norm(
        runtime,
        &projected,
        post_attn_residual,
        &w.post_attn_norm,
        Tracked(projected_p.borrow()),
        Tracked(rp),
        Tracked(&wp.post_attn_norm),
        Ghost(RT::linear_repr(attn_repr_g@, wp.o_proj.repr_2d())),
        Ghost(residual_repr),
        Ghost(wp.post_attn_norm.repr_1d()),
        Ghost(config.rms_norm_epsilon),
        scope8,
    );
    let normed2 = norm_pair.0;
    let next_residual = norm_pair.1;
    let normed2_p = norm_perms.0;
    let nr_p = norm_perms.1;
    let scope9 = Ghost(scope8@.insert(normed2.id()).insert(next_residual.id()));

    let normed2_repr_g: Ghost<Tensor2D> = Ghost(RT::add_rms_norm_repr(
        RT::linear_repr(attn_repr_g@, wp.o_proj.repr_2d()),
        residual_repr,
        wp.post_attn_norm.repr_1d(),
        config.rms_norm_epsilon,
    ).0);
    let (gate_up, gate_up_p) = RT::linear(
        runtime,
        &normed2,
        &w.gate_up_proj,
        Tracked(normed2_p.borrow()),
        Tracked(&wp.gate_up_proj),
        normed2_repr_g,
        Ghost(wp.gate_up_proj.repr_2d()),
        scope9,
    );
    let scope10 = Ghost(scope9@.insert(gate_up.id()));
    proof {
        RT::lemma_linear_repr_shape(
            normed2_repr_g@, wp.gate_up_proj.repr_2d(),
        );
        TS::lemma_tensor2d_shape_even_width(
            RT::linear_repr(normed2_repr_g@, wp.gate_up_proj.repr_2d()),
            normed2_repr_g@.len(),
            wp.gate_up_proj.repr_2d().len(),
        );
    }
    let (mlp, mlp_p) = RT::silu_and_mul(
        runtime,
        &gate_up,
        Tracked(gate_up_p.borrow()),
        Ghost(RT::linear_repr(
            normed2_repr_g@, wp.gate_up_proj.repr_2d(),
        )),
        scope10,
    );
    let scope11 = Ghost(scope10@.insert(mlp.id()));
    let (next_hidden, nh_p) = RT::linear(
        runtime,
        &mlp,
        &w.down_proj,
        Tracked(mlp_p.borrow()),
        Tracked(&wp.down_proj),
        Ghost(RT::silu_and_mul_repr(RT::linear_repr(
            normed2_repr_g@, wp.gate_up_proj.repr_2d(),
        ))),
        Ghost(wp.down_proj.repr_2d()),
        scope11,
    );

    (next_hidden, nh_p, next_residual, nr_p)
}

pub fn first_decoder_layer_forward(
    runtime: &RT::ModelFamilyRuntime,
    w: &WEIGHTS::DenseSwiGluLayerWeights,
    Tracked(wp): Tracked<&WEIGHTS::DenseSwiGluLayerWeightsPerms>,
    Ghost(head_dim): Ghost<nat>,
    Ghost(config): Ghost<DenseSwiGluForwardConfigRepr>,
    hidden: &RT::Tensor,
    Tracked(hp): Tracked<&RT::TensorPerm>,
    positions: &RT::Tensor,
    Tracked(pp): Tracked<&RT::TensorPerm>,
    k_cache: &RT::Tensor,
    Tracked(kc_perm): Tracked<&mut RT::TensorPerm>,
    v_cache: &RT::Tensor,
    Tracked(vc_perm): Tracked<&mut RT::TensorPerm>,
    block_table: &RT::Tensor,
    Tracked(bt_perm): Tracked<&RT::TensorPerm>,
    slot_mapping: &RT::Tensor,
    Tracked(sp): Tracked<&RT::TensorPerm>,
    cu_seqlens_q: &RT::Tensor,
    Tracked(cuq_perm): Tracked<&RT::TensorPerm>,
    cu_seqlens_k: &RT::Tensor,
    Tracked(cuk_perm): Tracked<&RT::TensorPerm>,
    max_seqlen_q: usize,
    max_seqlen_k: usize,
    Ghost(hidden_repr): Ghost<Tensor2D>,
    Ghost(positions_repr): Ghost<IntTensor1D>,
    Ghost(cu_q_repr): Ghost<Seq<int>>,
    Ghost(cu_k_repr): Ghost<Seq<int>>,
    Ghost(bt_repr): Ghost<Seq<Seq<BlockId>>>,
    Ghost(k_cache_repr): Ghost<KVCacheLayerRepr>,
    Ghost(v_cache_repr): Ghost<KVCacheLayerRepr>,
    Ghost(slot_repr): Ghost<Seq<int>>,
    Ghost(scope): Ghost<Set<RT::TensorId>>,
) -> (out: (RT::Tensor, Tracked<RT::TensorPerm>, RT::Tensor, Tracked<RT::TensorPerm>))
    requires
        RT::paged_attention_numeric_domain(),
        RT::family_runtime_execution_valid(runtime),
        WEIGHTS::layer_weights_valid(w, wp),
        RT::dense_swiglu_runtime_rms_norm_matches(
            runtime, config.rms_norm_epsilon,
        ),
        RT::dense_swiglu_runtime_qk_norm_matches(
            runtime,
            qk_norm_weights_kind(WEIGHTS::qk_norm_weights_repr_of(&wp.qk_norm)),
        ),
        RT::dense_swiglu_runtime_rotary_matches(
            runtime,
            DLP::layer_attention_geometry_repr(
                WEIGHTS::layer_weights_repr_of(wp, head_dim),
            ),
            config.rotary,
        ),
        RT::tensor_repr_2d(*hp, *hidden, hidden_repr),
        RT::int_tensor_repr_1d(*pp, *positions, positions_repr),
        RT::kv_cache_tensor_repr(*old(kc_perm), *k_cache, k_cache_repr),
        RT::kv_cache_tensor_repr(*old(vc_perm), *v_cache, v_cache_repr),
        RT::block_table_repr(*bt_perm, *block_table, bt_repr),
        RT::int_tensor_repr_1d(*sp, *slot_mapping, slot_repr),
        RT::int_tensor_repr_1d(*cuq_perm, *cu_seqlens_q, cu_q_repr),
        RT::int_tensor_repr_1d(*cuk_perm, *cu_seqlens_k, cu_k_repr),
        hidden_repr.len() == positions_repr.len(),
        slot_repr.len() == hidden_repr.len(),
        RT::store_kv_cache_launch_ready(
            hidden_repr.len(), k_cache_repr, v_cache_repr, slot_repr,
        ),
        RT::paged_attention_launch_ready(
            hidden_repr.len(),
            k_cache_repr,
            v_cache_repr,
            cu_q_repr,
            cu_k_repr,
            max_seqlen_q as nat,
            max_seqlen_k as nat,
            bt_repr,
        ),
    ensures ({
        let (next_hidden, nh_perm, next_residual, nr_perm) = out;
        let wr = WEIGHTS::layer_weights_repr_of(wp, head_dim);
        let layer_out = SEM::first_decoder_layer_output_repr(
            config,
            wr,
            hidden_repr,
            positions_repr,
            k_cache_repr,
            v_cache_repr,
            slot_repr,
            cu_q_repr,
            cu_k_repr,
            max_seqlen_q as nat,
            max_seqlen_k as nat,
            bt_repr,
        );
        let normed = RT::rms_norm_repr(
            hidden_repr, wr.input_norm, config.rms_norm_epsilon,
        );
        let kv_out = SEM::layer_kv_update_repr(
            config,
            wr,
            normed,
            positions_repr,
            k_cache_repr,
            v_cache_repr,
            slot_repr,
        );
        &&& next_hidden.id() != next_residual.id()
        &&& RT::tensor_repr_2d(nh_perm@, next_hidden, layer_out.0)
        &&& RT::tensor_repr_2d(nr_perm@, next_residual, layer_out.1)
        &&& final(kc_perm).id() == old(kc_perm).id()
        &&& final(vc_perm).id() == old(vc_perm).id()
        &&& RT::kv_cache_tensor_repr(*final(kc_perm), *k_cache, kv_out.0)
        &&& RT::kv_cache_tensor_repr(*final(vc_perm), *v_cache, kv_out.1)
    }),
{
    let (normed, normed_p) = RT::rms_norm(
        runtime,
        hidden,
        &w.input_norm,
        Tracked(hp),
        Tracked(&wp.input_norm),
        Ghost(hidden_repr),
        Ghost(wp.input_norm.repr_1d()),
        Ghost(config.rms_norm_epsilon),
        Ghost(scope),
    );
    proof {
        RT::lemma_rms_norm_repr_shape(
            hidden_repr,
            wp.input_norm.repr_1d(),
            config.rms_norm_epsilon,
        );
    }
    let scope_after_norm = Ghost(scope.insert(normed.id()));
    decoder_core_forward(
        runtime,
        w,
        Tracked(wp),
        Ghost(head_dim),
        Ghost(config),
        &normed,
        Tracked(normed_p.borrow()),
        hidden,
        Tracked(hp),
        positions,
        Tracked(pp),
        k_cache,
        Tracked(kc_perm),
        v_cache,
        Tracked(vc_perm),
        block_table,
        Tracked(bt_perm),
        slot_mapping,
        Tracked(sp),
        cu_seqlens_q,
        Tracked(cuq_perm),
        cu_seqlens_k,
        Tracked(cuk_perm),
        max_seqlen_q,
        max_seqlen_k,
        Ghost(RT::rms_norm_repr(
            hidden_repr,
            wp.input_norm.repr_1d(),
            config.rms_norm_epsilon,
        )),
        Ghost(hidden_repr),
        Ghost(positions_repr),
        Ghost(cu_q_repr),
        Ghost(cu_k_repr),
        Ghost(bt_repr),
        Ghost(k_cache_repr),
        Ghost(v_cache_repr),
        Ghost(slot_repr),
        scope_after_norm,
    )
}

pub fn decoder_layer_forward(
    runtime: &RT::ModelFamilyRuntime,
    w: &WEIGHTS::DenseSwiGluLayerWeights,
    Tracked(wp): Tracked<&WEIGHTS::DenseSwiGluLayerWeightsPerms>,
    Ghost(head_dim): Ghost<nat>,
    Ghost(config): Ghost<DenseSwiGluForwardConfigRepr>,
    hidden: &RT::Tensor,
    Tracked(hp): Tracked<&RT::TensorPerm>,
    residual: &RT::Tensor,
    Tracked(rp): Tracked<&RT::TensorPerm>,
    positions: &RT::Tensor,
    Tracked(pp): Tracked<&RT::TensorPerm>,
    k_cache: &RT::Tensor,
    Tracked(kc_perm): Tracked<&mut RT::TensorPerm>,
    v_cache: &RT::Tensor,
    Tracked(vc_perm): Tracked<&mut RT::TensorPerm>,
    block_table: &RT::Tensor,
    Tracked(bt_perm): Tracked<&RT::TensorPerm>,
    slot_mapping: &RT::Tensor,
    Tracked(sp): Tracked<&RT::TensorPerm>,
    cu_seqlens_q: &RT::Tensor,
    Tracked(cuq_perm): Tracked<&RT::TensorPerm>,
    cu_seqlens_k: &RT::Tensor,
    Tracked(cuk_perm): Tracked<&RT::TensorPerm>,
    max_seqlen_q: usize,
    max_seqlen_k: usize,
    Ghost(hidden_repr): Ghost<Tensor2D>,
    Ghost(residual_repr): Ghost<Tensor2D>,
    Ghost(positions_repr): Ghost<IntTensor1D>,
    Ghost(cu_q_repr): Ghost<Seq<int>>,
    Ghost(cu_k_repr): Ghost<Seq<int>>,
    Ghost(bt_repr): Ghost<Seq<Seq<BlockId>>>,
    Ghost(k_cache_repr): Ghost<KVCacheLayerRepr>,
    Ghost(v_cache_repr): Ghost<KVCacheLayerRepr>,
    Ghost(slot_repr): Ghost<Seq<int>>,
    Ghost(scope): Ghost<Set<RT::TensorId>>,
) -> (out: (RT::Tensor, Tracked<RT::TensorPerm>, RT::Tensor, Tracked<RT::TensorPerm>))
    requires
        RT::paged_attention_numeric_domain(),
        RT::family_runtime_execution_valid(runtime),
        WEIGHTS::layer_weights_valid(w, wp),
        RT::dense_swiglu_runtime_rms_norm_matches(
            runtime, config.rms_norm_epsilon,
        ),
        RT::dense_swiglu_runtime_qk_norm_matches(
            runtime,
            qk_norm_weights_kind(WEIGHTS::qk_norm_weights_repr_of(&wp.qk_norm)),
        ),
        RT::dense_swiglu_runtime_rotary_matches(
            runtime,
            DLP::layer_attention_geometry_repr(
                WEIGHTS::layer_weights_repr_of(wp, head_dim),
            ),
            config.rotary,
        ),
        RT::tensor_repr_2d(*hp, *hidden, hidden_repr),
        RT::tensor_repr_2d(*rp, *residual, residual_repr),
        RT::int_tensor_repr_1d(*pp, *positions, positions_repr),
        RT::kv_cache_tensor_repr(*old(kc_perm), *k_cache, k_cache_repr),
        RT::kv_cache_tensor_repr(*old(vc_perm), *v_cache, v_cache_repr),
        RT::block_table_repr(*bt_perm, *block_table, bt_repr),
        RT::int_tensor_repr_1d(*sp, *slot_mapping, slot_repr),
        RT::int_tensor_repr_1d(*cuq_perm, *cu_seqlens_q, cu_q_repr),
        RT::int_tensor_repr_1d(*cuk_perm, *cu_seqlens_k, cu_k_repr),
        hidden_repr.len() == residual_repr.len(),
        hidden_repr.len() == positions_repr.len(),
        slot_repr.len() == hidden_repr.len(),
        RT::store_kv_cache_launch_ready(
            hidden_repr.len(), k_cache_repr, v_cache_repr, slot_repr,
        ),
        RT::paged_attention_launch_ready(
            hidden_repr.len(),
            k_cache_repr,
            v_cache_repr,
            cu_q_repr,
            cu_k_repr,
            max_seqlen_q as nat,
            max_seqlen_k as nat,
            bt_repr,
        ),
    ensures ({
        let (next_hidden, nh_perm, next_residual, nr_perm) = out;
        let wr = WEIGHTS::layer_weights_repr_of(wp, head_dim);
        let layer_out = SEM::decoder_layer_output_repr(
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
            max_seqlen_q as nat,
            max_seqlen_k as nat,
            bt_repr,
        );
        let pre = RT::add_rms_norm_repr(
            hidden_repr,
            residual_repr,
            wr.input_norm,
            config.rms_norm_epsilon,
        );
        let kv_out = SEM::layer_kv_update_repr(
            config,
            wr,
            pre.0,
            positions_repr,
            k_cache_repr,
            v_cache_repr,
            slot_repr,
        );
        &&& next_hidden.id() != next_residual.id()
        &&& RT::tensor_repr_2d(nh_perm@, next_hidden, layer_out.0)
        &&& RT::tensor_repr_2d(nr_perm@, next_residual, layer_out.1)
        &&& final(kc_perm).id() == old(kc_perm).id()
        &&& final(vc_perm).id() == old(vc_perm).id()
        &&& RT::kv_cache_tensor_repr(*final(kc_perm), *k_cache, kv_out.0)
        &&& RT::kv_cache_tensor_repr(*final(vc_perm), *v_cache, kv_out.1)
    }),
{
    let (norm_pair, norm_perms) = RT::add_rms_norm(
        runtime,
        hidden,
        residual,
        &w.input_norm,
        Tracked(hp),
        Tracked(rp),
        Tracked(&wp.input_norm),
        Ghost(hidden_repr),
        Ghost(residual_repr),
        Ghost(wp.input_norm.repr_1d()),
        Ghost(config.rms_norm_epsilon),
        Ghost(scope),
    );
    let normed = norm_pair.0;
    let post_res = norm_pair.1;
    let normed_p = norm_perms.0;
    let post_res_p = norm_perms.1;
    proof {
        RT::lemma_add_rms_norm_repr_shape(
            hidden_repr,
            residual_repr,
            wp.input_norm.repr_1d(),
            config.rms_norm_epsilon,
        );
    }
    let scope_after_norm = Ghost(scope.insert(normed.id()).insert(post_res.id()));
    decoder_core_forward(
        runtime,
        w,
        Tracked(wp),
        Ghost(head_dim),
        Ghost(config),
        &normed,
        Tracked(normed_p.borrow()),
        &post_res,
        Tracked(post_res_p.borrow()),
        positions,
        Tracked(pp),
        k_cache,
        Tracked(kc_perm),
        v_cache,
        Tracked(vc_perm),
        block_table,
        Tracked(bt_perm),
        slot_mapping,
        Tracked(sp),
        cu_seqlens_q,
        Tracked(cuq_perm),
        cu_seqlens_k,
        Tracked(cuk_perm),
        max_seqlen_q,
        max_seqlen_k,
        Ghost(RT::add_rms_norm_repr(
            hidden_repr,
            residual_repr,
            wp.input_norm.repr_1d(),
            config.rms_norm_epsilon,
        ).0),
        Ghost(RT::add_rms_norm_repr(
            hidden_repr,
            residual_repr,
            wp.input_norm.repr_1d(),
            config.rms_norm_epsilon,
        ).1),
        Ghost(positions_repr),
        Ghost(cu_q_repr),
        Ghost(cu_k_repr),
        Ghost(bt_repr),
        Ghost(k_cache_repr),
        Ghost(v_cache_repr),
        Ghost(slot_repr),
        scope_after_norm,
    )
}

} // verus!
