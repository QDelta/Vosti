//! Checked row-local execution shared by four-norm gated decoders.
//! Attention/KV mutation remains in the layer caller. Optional layer-output
//! scaling and logit softcapping are composed from their primitive contracts.

use crate::model_config::{AttentionKind, FloatParameterBits};
use crate::boundary::four_norm_gated_primitives as P;
use crate::boundary::four_norm_gated_weights as W;
#[cfg(verus_only)]
use crate::proof::model::four_norm_gated::model as MODEL;
#[cfg(verus_only)]
use crate::proof::model::four_norm_gated::layers as ROWS;
use crate::{types::*, proof::model::types::*, proof::tensor::types::*};
use crate::boundary::tensor_runtime as RT;
use crate::proof::tensor::shape as TS;
#[cfg(verus_only)]
use crate::boundary::scalar::{float_parameter_scalar_repr, positive_float_parameter_valid};
use vstd::prelude::*;

verus! {

pub open spec fn value_cache_rows_repr(
    input: Tensor2D, head_dim: nat, epsilon: Option<FloatParameterBits>,
) -> Tensor2D {
    match epsilon {
        Some(eps) => P::value_norm_repr(input, head_dim, eps),
        None => RT::view_as_kv_repr(input),
    }
}

pub proof fn lemma_value_cache_rows_matches_layer_policy(
    input: Tensor2D, head_dim: nat, epsilon: Option<FloatParameterBits>,
)
    ensures value_cache_rows_repr(input, head_dim, epsilon)
        == Seq::new(input.len(), |i: int| ROWS::value_row(input[i], head_dim, epsilon)),
{
    reveal(RT::view_as_kv_repr);
    assert(value_cache_rows_repr(input, head_dim, epsilon)
        =~= Seq::new(input.len(), |i: int| ROWS::value_row(input[i], head_dim, epsilon)));
}

pub fn prepare_value_cache_rows(
    runtime: &RT::ModelFamilyRuntime,
    input: &RT::Tensor,
    kind: AttentionKind,
    epsilon: Option<FloatParameterBits>,
    Tracked(ip): Tracked<&RT::TensorPerm>,
    Ghost(input_repr): Ghost<Tensor2D>,
    Ghost(geometry): Ghost<AttentionGeometryRepr>,
    Ghost(scope): Ghost<Set<RT::TensorId>>,
) -> (out: (RT::Tensor, Tracked<RT::TensorPerm>))
    requires
        P::configuration_valid(runtime),
        P::runtime_attention_geometry(runtime, kind) == Some(geometry),
        P::runtime_policy(runtime).unwrap().value_norm_epsilon == epsilon,
        RT::tensor_repr_2d(*ip, *input, input_repr),
        epsilon.is_some() ==> TS::tensor2d_shape(input_repr, input_repr.len(),
            geometry.num_key_value_heads * geometry.head_dim),
        // The plain layout adapter currently uses the uniform/local runtime
        // geometry. Heterogeneous Gemma-4 always takes the normalized branch.
        epsilon.is_none() ==> geometry == P::runtime_policy(runtime).unwrap().local_geometry,
    ensures ({ let (tensor, perm) = out;
        &&& !scope.contains(tensor.id())
        &&& tensor.id() != input.id()
        &&& RT::tensor_repr_2d(perm@, tensor,
            value_cache_rows_repr(input_repr, geometry.head_dim, epsilon))
    }),
{
    match epsilon {
        Some(eps) => P::value_norm(runtime, input, kind, Tracked(ip), Ghost(input_repr),
            Ghost(geometry), Ghost(eps), Ghost(scope)),
        None => RT::view_as_kv(runtime, input, Tracked(ip), Ghost(input_repr), Ghost(scope)),
    }
}

pub fn post_attention_and_mlp(
    runtime: &RT::ModelFamilyRuntime,
    w: &W::FourNormGatedLayerWeights,
    Tracked(wp): Tracked<&W::FourNormGatedLayerWeightsPerms>,
    hidden: &RT::Tensor,
    Tracked(hp): Tracked<&RT::TensorPerm>,
    attended: &RT::Tensor,
    Tracked(ap): Tracked<&RT::TensorPerm>,
    Ghost(layer): Ghost<FourNormGatedLayerRepr>,
    Ghost(hidden_repr): Ghost<Tensor2D>,
    Ghost(attended_repr): Ghost<Tensor2D>,
    Ghost(scope): Ghost<Set<RT::TensorId>>,
) -> (out: (RT::Tensor, Tracked<RT::TensorPerm>))
    requires
        P::configuration_valid(runtime),
        P::runtime_norm_policy(runtime) == Some(layer.norm),
        W::layer_weights_valid(w, wp),
        layer.common == W::layer_weights_common_repr_of(wp),
        layer.pre_feedforward_norm == wp.pre_feedforward_norm.repr_1d(),
        layer.post_feedforward_norm == wp.post_feedforward_norm.repr_1d(),
        layer.layer_scale == W::layer_scale_repr(wp),
        layer.layer_scale.is_some() ==> P::runtime_policy(runtime).unwrap().layer_scale,
        RT::tensor_repr_2d(*hp, *hidden, hidden_repr),
        RT::tensor_repr_2d(*ap, *attended, attended_repr),
        hidden_repr.len() == attended_repr.len(),
    ensures ({ let (tensor, perm) = out;
        &&& !scope.contains(tensor.id())
        &&& RT::tensor_repr_2d(perm@, tensor,
            ROWS::post_attention_and_mlp_repr(layer, hidden_repr, attended_repr))
    }),
{
    broadcast use {
        RT::lemma_linear_repr_shape,
        RT::lemma_split_last_axis_half_repr_shape,
        MODEL::lemma_gelu_tanh_mul_repr_shape,
        MODEL::lemma_add_repr_shape,
        crate::boundary::dense_layer_primitives::lemma_merge_attention_heads_repr_shape,
    };
    let (merged, merged_p) = RT::merge_attention_heads(
        attended, Tracked(ap), Ghost(attended_repr), Ghost(scope),
    );
    let ghost merged_repr =
        crate::boundary::dense_layer_primitives::merge_attention_heads_repr(
            attended_repr,
        );
    let scope1 = Ghost(scope.insert(merged.id()));
    let (projected, projected_p) = RT::linear(
        runtime, &merged, &w.o_proj,
        Tracked(merged_p.borrow()), Tracked(&wp.o_proj),
        Ghost(merged_repr), Ghost(wp.o_proj.repr_2d()), scope1,
    );
    let ghost projected_repr = RT::linear_repr(merged_repr, wp.o_proj.repr_2d());
    let scope2 = Ghost(scope1@.insert(projected.id()));
    let (post_attention, post_attention_p) = P::rms_norm(
        runtime, &projected, &w.post_attn_norm,
        P::RmsNormSite::PostAttention, Ghost(layer.norm),
        Tracked(projected_p.borrow()), Tracked(&wp.post_attn_norm),
        Ghost(projected_repr), Ghost(wp.post_attn_norm.repr_1d()), scope2,
    );
    let ghost post_attention_repr = P::norm_repr(
        projected_repr, wp.post_attn_norm.repr_1d(), layer.norm,
    );
    let scope3 = Ghost(scope2@.insert(post_attention.id()));
    let (after_attention, after_attention_p) = P::add(
        runtime, hidden, &post_attention, P::AddSite::AttentionResidual,
        Tracked(hp), Tracked(post_attention_p.borrow()),
        Ghost(hidden_repr), Ghost(post_attention_repr), scope3,
    );
    let ghost after_attention_repr = MODEL::add_repr(
        hidden_repr, post_attention_repr,
    );
    let scope4 = Ghost(scope3@.insert(after_attention.id()));

    let (feedforward_input, feedforward_input_p) = P::rms_norm(
        runtime, &after_attention, &w.pre_feedforward_norm,
        P::RmsNormSite::PreFeedforward, Ghost(layer.norm),
        Tracked(after_attention_p.borrow()), Tracked(&wp.pre_feedforward_norm),
        Ghost(after_attention_repr),
        Ghost(wp.pre_feedforward_norm.repr_1d()), scope4,
    );
    let ghost feedforward_input_repr = P::norm_repr(
        after_attention_repr, wp.pre_feedforward_norm.repr_1d(), layer.norm,
    );
    let scope5 = Ghost(scope4@.insert(feedforward_input.id()));
    let (gate_up, gate_up_p) = RT::linear(
        runtime, &feedforward_input, &w.gate_up_proj,
        Tracked(feedforward_input_p.borrow()), Tracked(&wp.gate_up_proj),
        Ghost(feedforward_input_repr), Ghost(wp.gate_up_proj.repr_2d()), scope5,
    );
    let ghost gate_up_repr = RT::linear_repr(
        feedforward_input_repr, wp.gate_up_proj.repr_2d(),
    );
    proof {
        RT::lemma_linear_repr_shape(
            feedforward_input_repr, wp.gate_up_proj.repr_2d(),
        );
        TS::lemma_tensor2d_shape_even_width(
            gate_up_repr, feedforward_input_repr.len(),
            wp.gate_up_proj.repr_2d().len(),
        );
    }
    let scope6 = Ghost(scope5@.insert(gate_up.id()));
    let ((gate, up), (gate_p, up_p)) = RT::split_last_axis_halves(
        &gate_up, Tracked(gate_up_p.borrow()), Ghost(gate_up_repr), scope6,
    );
    let ghost gate_repr = RT::split_last_axis_half_repr(gate_up_repr, false);
    let ghost up_repr = RT::split_last_axis_half_repr(gate_up_repr, true);
    let scope7 = Ghost(scope6@.insert(gate.id()).insert(up.id()));
    let (activated, activated_p) = P::gelu_tanh_mul(
        runtime, &gate, &up,
        Tracked(gate_p.borrow()), Tracked(up_p.borrow()),
        Ghost(gate_repr), Ghost(up_repr), scope7,
    );
    let ghost activated_repr = MODEL::gelu_tanh_mul_repr(gate_repr, up_repr);
    let scope8 = Ghost(scope7@.insert(activated.id()));
    let (down, down_p) = RT::linear(
        runtime, &activated, &w.down_proj,
        Tracked(activated_p.borrow()), Tracked(&wp.down_proj),
        Ghost(activated_repr), Ghost(wp.down_proj.repr_2d()), scope8,
    );
    let ghost down_repr = RT::linear_repr(activated_repr, wp.down_proj.repr_2d());
    let scope9 = Ghost(scope8@.insert(down.id()));
    let (post_feedforward, post_feedforward_p) = P::rms_norm(
        runtime, &down, &w.post_feedforward_norm,
        P::RmsNormSite::PostFeedforward, Ghost(layer.norm),
        Tracked(down_p.borrow()), Tracked(&wp.post_feedforward_norm),
        Ghost(down_repr), Ghost(wp.post_feedforward_norm.repr_1d()), scope9,
    );
    let ghost post_feedforward_repr = P::norm_repr(
        down_repr, wp.post_feedforward_norm.repr_1d(), layer.norm,
    );
    let scope10 = Ghost(scope9@.insert(post_feedforward.id()));
    let (unscaled, unscaled_p) = P::add(
        runtime, &after_attention, &post_feedforward,
        P::AddSite::FeedforwardResidual,
        Tracked(after_attention_p.borrow()), Tracked(post_feedforward_p.borrow()),
        Ghost(after_attention_repr), Ghost(post_feedforward_repr), scope10,
    );
    let ghost unscaled_repr = MODEL::add_repr(after_attention_repr, post_feedforward_repr);
    let out = match &w.layer_scale {
        Some(scalar) => {
            let tracked scalar_p = match &wp.layer_scale {
                Some(perm) => perm,
                None => { assert(false); proof_from_false() },
            };
            P::scale(runtime, &unscaled, scalar, Tracked(unscaled_p.borrow()), Tracked(scalar_p),
                Ghost(unscaled_repr), Ghost(scalar_p.repr_1d()),
                Ghost(scope10@.insert(unscaled.id())))
        },
        None => (unscaled, unscaled_p),
    };
    proof {
        reveal(ROWS::post_attention_and_mlp_row);
        reveal(MODEL::add_repr);
        reveal(MODEL::gelu_tanh_mul_repr);
        reveal(RT::split_last_axis_half_repr);
        let result = Seq::new(unscaled_repr.len(), |i: int|
            ROWS::scale_row(unscaled_repr[i], layer.layer_scale));
        match &wp.layer_scale {
            Some(scalar) => { assert(result =~= P::scale_repr(unscaled_repr, scalar.repr_1d())); },
            None => { assert(result =~= unscaled_repr); },
        }
        assert(RT::tensor_repr_2d(out.1@, out.0, result));
        assert(result =~= ROWS::post_attention_and_mlp_repr(
            layer, hidden_repr, attended_repr));
    }
    out
}


pub fn project_last_logits(
    runtime: &RT::ModelFamilyRuntime,
    hidden: &RT::Tensor,
    norm_weight: &RT::Tensor,
    head: &RT::Tensor,
    cu_seqlens_q: &RT::Tensor,
    count: usize,
    softcap: Option<FloatParameterBits>,
    Tracked(hp): Tracked<&RT::TensorPerm>,
    Tracked(np): Tracked<&RT::TensorPerm>,
    Tracked(lp): Tracked<&RT::TensorPerm>,
    Tracked(cup): Tracked<&RT::TensorPerm>,
    Ghost(policy): Ghost<NormPolicyRepr>,
    Ghost(hidden_repr): Ghost<Tensor2D>,
    Ghost(norm_repr): Ghost<Tensor1D>,
    Ghost(head_repr): Ghost<Tensor2D>,
    Ghost(cu_repr): Ghost<Seq<int>>,
    Ghost(scope): Ghost<Set<RT::TensorId>>,
) -> (out: (RT::Tensor, Tracked<RT::TensorPerm>))
    requires
        P::configuration_valid(runtime),
        P::runtime_norm_policy(runtime) == Some(policy),
        P::runtime_policy(runtime).unwrap().logits_softcap == softcap,
        match softcap {
            Some(cap) => positive_float_parameter_valid(cap),
            None => true,
        },
        RT::tensor_repr_2d(*hp, *hidden, hidden_repr),
        RT::tensor_repr_1d(*np, *norm_weight, norm_repr),
        RT::tensor_repr_2d(*lp, *head, head_repr),
        RT::int_tensor_repr_1d(*cup, *cu_seqlens_q, cu_repr),
        cu_repr.len() == count as nat + 1,
        forall|j: int| 0 <= j < count as int ==>
            #[trigger] cu_repr[j + 1] > 0
                && cu_repr[j + 1] <= hidden_repr.len() as int,
    ensures ({ let (tensor, perm) = out;
        let full_logits = ROWS::final_logits_repr(hidden_repr, norm_repr, head_repr, policy,
            softcap);
        &&& !scope.contains(tensor.id())
        &&& RT::tensor_repr_2d(perm@, tensor, Seq::new(count as nat, |j: int|
            RT::select_sample_logits_repr(full_logits, cu_repr, j as nat)))
    }),
{
    let (normed, normed_p) = P::rms_norm(
        runtime, hidden, norm_weight, P::RmsNormSite::Final, Ghost(policy),
        Tracked(hp), Tracked(np), Ghost(hidden_repr), Ghost(norm_repr), Ghost(scope),
    );
    let ghost normed_repr = P::norm_repr(hidden_repr, norm_repr, policy);
    let scope_after_norm = Ghost(scope.insert(normed.id()));
    // Preserve last-row selection before the vocabulary projection; do not
    // materialize vocabulary logits for every prefill token.
    let (selected, selected_p) = RT::select_last_hidden_rows(
        &normed, cu_seqlens_q, count,
        Tracked(normed_p.borrow()), Tracked(cup),
        Ghost(normed_repr), Ghost(cu_repr), scope_after_norm,
    );
    let ghost selected_repr = Seq::new(count as nat, |j: int|
        normed_repr[cu_repr[j + 1] - 1]);
    let (uncapped, uncapped_p) = RT::linear(
        runtime, &selected, head,
        Tracked(selected_p.borrow()), Tracked(lp),
        Ghost(selected_repr), Ghost(head_repr),
        Ghost(scope_after_norm@.insert(selected.id())),
    );
    let ghost uncapped_repr = RT::linear_repr(selected_repr, head_repr);
    let out = match softcap {
        Some(cap) => P::softcap(runtime, &uncapped, Tracked(uncapped_p.borrow()),
            Ghost(uncapped_repr), Ghost(cap),
            Ghost(scope_after_norm@.insert(selected.id()).insert(uncapped.id()))),
        None => (uncapped, uncapped_p),
    };
    proof {
        reveal(RT::linear_repr);
        let full_logits = ROWS::final_logits_repr(
            hidden_repr, norm_repr, head_repr, policy, softcap);
        assert(ROWS::logits_softcap_repr(uncapped_repr, softcap) =~=
            Seq::new(count as nat, |j: int|
                RT::select_sample_logits_repr(full_logits, cu_repr, j as nat)));
    }
    out
}

// Exact immutable policy/role agreement for one shared decoder invocation.
// Physical head geometry is derived from the bound layer, never query length.
pub open spec fn layer_execution_policy_matches(
    runtime: &RT::ModelFamilyRuntime,
    w: &W::FourNormGatedLayerWeights,
    wp: &W::FourNormGatedLayerWeightsPerms,
    window: nat, value_epsilon: Option<FloatParameterBits>,
    extension: FourNormGatedLayerExtensionRepr,
) -> bool {
    let common = W::layer_weights_common_repr_of(wp);
    let geometry = crate::boundary::dense_layer_primitives::layer_attention_geometry_repr(common);
    let rows = extension.row_parameters;
    &&& W::layer_weights_valid(w, wp)
    &&& P::runtime_attention_parameters_match(runtime, w.attention_kind, geometry,
        window, extension.attention_scale)
    &&& P::runtime_norm_policy(runtime) == Some(rows.norm)
    &&& P::runtime_qk_norm_policy(runtime, w.attention_kind) == Some(rows.qk_norm)
    &&& P::runtime_rotary_config(runtime, w.attention_kind) == Some(rows.rotary)
    &&& extension.attention == P::attention_config(w.attention_kind, window)
    &&& extension.pre_feedforward_norm == wp.pre_feedforward_norm.repr_1d()
    &&& extension.post_feedforward_norm == wp.post_feedforward_norm.repr_1d()
    &&& rows.layer_scale == W::layer_scale_repr(wp)
    &&& (rows.layer_scale.is_some() ==> P::runtime_policy(runtime).unwrap().layer_scale)
    &&& rows.value_norm_epsilon == value_epsilon
    &&& rows.value_norm_epsilon == P::runtime_policy(runtime).unwrap().value_norm_epsilon
    &&& (value_epsilon.is_none() ==> geometry == P::runtime_policy(runtime).unwrap().local_geometry)
    &&& (value_epsilon.is_some() ==>
        common.k_proj.len() == geometry.num_key_value_heads * geometry.head_dim)
}

pub fn decoder_layer_forward(
    runtime: &RT::ModelFamilyRuntime,
    w: &W::FourNormGatedLayerWeights,
    Tracked(wp): Tracked<&W::FourNormGatedLayerWeightsPerms>,
    sliding_window: usize,
    value_norm_epsilon: Option<FloatParameterBits>,
    Ghost(extension): Ghost<FourNormGatedLayerExtensionRepr>,
    hidden: &RT::Tensor,
    Tracked(hp): Tracked<&RT::TensorPerm>,
    positions: &RT::Tensor,
    Tracked(pp): Tracked<&RT::TensorPerm>,
    k_cache: &RT::Tensor,
    Tracked(kcp): Tracked<&mut RT::TensorPerm>,
    v_cache: &RT::Tensor,
    Tracked(vcp): Tracked<&mut RT::TensorPerm>,
    block_table: &RT::Tensor,
    Tracked(btp): Tracked<&RT::TensorPerm>,
    slot_mapping: &RT::Tensor,
    Tracked(sp): Tracked<&RT::TensorPerm>,
    cu_seqlens_q: &RT::Tensor,
    Tracked(cuqp): Tracked<&RT::TensorPerm>,
    cu_seqlens_k: &RT::Tensor,
    Tracked(cukp): Tracked<&RT::TensorPerm>,
    max_seqlen_q: usize,
    max_seqlen_k: usize,
    Ghost(hidden_repr): Ghost<Tensor2D>,
    Ghost(positions_repr): Ghost<IntTensor1D>,
    Ghost(k_cache_repr): Ghost<KVCacheLayerRepr>,
    Ghost(v_cache_repr): Ghost<KVCacheLayerRepr>,
    Ghost(bt_repr): Ghost<Seq<Seq<BlockId>>>,
    Ghost(slot_repr): Ghost<Seq<int>>,
    Ghost(cu_q_repr): Ghost<Seq<int>>,
    Ghost(cu_k_repr): Ghost<Seq<int>>,
    Ghost(scope): Ghost<Set<RT::TensorId>>,
) -> (out: (RT::Tensor, Tracked<RT::TensorPerm>))
    requires
        RT::paged_attention_numeric_domain(),
        layer_execution_policy_matches(runtime, w, wp, sliding_window as nat,
            value_norm_epsilon, extension),
        sliding_window > 0,
        RT::tensor_repr_2d(*hp, *hidden, hidden_repr),
        RT::int_tensor_repr_1d(*pp, *positions, positions_repr),
        RT::kv_cache_tensor_repr(*old(kcp), *k_cache, k_cache_repr),
        RT::kv_cache_tensor_repr(*old(vcp), *v_cache, v_cache_repr),
        RT::block_table_repr(*btp, *block_table, bt_repr),
        RT::int_tensor_repr_1d(*sp, *slot_mapping, slot_repr),
        RT::int_tensor_repr_1d(*cuqp, *cu_seqlens_q, cu_q_repr),
        RT::int_tensor_repr_1d(*cukp, *cu_seqlens_k, cu_k_repr),
        hidden_repr.len() == positions_repr.len(),
        hidden_repr.len() == slot_repr.len(),
        RT::store_kv_cache_launch_ready(
            hidden_repr.len(), k_cache_repr, v_cache_repr, slot_repr,
        ),
        RT::paged_attention_launch_ready(
            hidden_repr.len(), k_cache_repr, v_cache_repr,
            cu_q_repr, cu_k_repr,
            max_seqlen_q as nat, max_seqlen_k as nat, bt_repr,
        ),
    ensures ({ let (next_hidden, next_perm) = out;
        let common = W::layer_weights_common_repr_of(wp);
        let step = MODEL::decoder_layer_step_repr(
            common, extension, hidden_repr, positions_repr,
            k_cache_repr, v_cache_repr, slot_repr,
            cu_q_repr, cu_k_repr,
            max_seqlen_q as nat, max_seqlen_k as nat, bt_repr,
        );
        &&& RT::tensor_repr_2d(next_perm@, next_hidden, step.0)
        &&& RT::kv_cache_tensor_repr(*final(kcp), *k_cache, step.1.0)
        &&& RT::kv_cache_tensor_repr(*final(vcp), *v_cache, step.1.1)
        &&& final(kcp).id() == old(kcp).id()
        &&& final(vcp).id() == old(vcp).id()
    }),
{
    broadcast use {
        RT::lemma_linear_repr_shape,
        RT::lemma_view_as_kv_repr_shape,
        RT::lemma_split_last_axis_half_repr_shape,
        MODEL::lemma_attention_pre_store_repr_shape,
        MODEL::lemma_paged_attention_repr_shape,
        MODEL::lemma_gelu_tanh_mul_repr_shape,
        MODEL::lemma_add_repr_shape,
        crate::boundary::dense_layer_primitives::lemma_merge_attention_heads_repr_shape,
    };

    let ghost common = W::layer_weights_common_repr_of(wp);
    let ghost attention_geometry =
        crate::boundary::dense_layer_primitives::layer_attention_geometry_repr(common);

    let (attention_input, attention_input_p) = P::rms_norm(
        runtime, hidden, &w.input_norm, P::RmsNormSite::Input, Ghost(extension.row_parameters.norm),
        Tracked(hp), Tracked(&wp.input_norm),
        Ghost(hidden_repr), Ghost(wp.input_norm.repr_1d()), Ghost(scope),
    );
    let ghost attention_input_repr = P::norm_repr(
        hidden_repr, wp.input_norm.repr_1d(), extension.row_parameters.norm,
    );
    let scope1 = Ghost(scope.insert(attention_input.id()));

    let ((q, k, v), (q_p, k_p, v_p)) = RT::qkv_linear(
        runtime, &attention_input, &w.q_proj, &w.k_proj, &w.v_proj,
        Tracked(attention_input_p.borrow()),
        Tracked(&wp.q_proj), Tracked(&wp.k_proj), Tracked(&wp.v_proj),
        Ghost(attention_input_repr),
        Ghost(wp.q_proj.repr_2d()), Ghost(wp.k_proj.repr_2d()),
        Ghost(wp.v_proj.repr_2d()), scope1,
    );
    let ghost qkv_repr = RT::qkv_linear_repr(
        attention_input_repr,
        wp.q_proj.repr_2d(),
        wp.k_proj.repr_2d(),
        wp.v_proj.repr_2d(),
    );
    let ghost q_repr = qkv_repr.0;
    let ghost k_repr = qkv_repr.1;
    let ghost v_repr = qkv_repr.2;
    let scope4 = Ghost(scope1@.insert(q.id()).insert(k.id()).insert(v.id()));

    let ((nq, nk), (nq_p, nk_p)) = P::qk_norm(
        runtime, &q, &k, &w.q_norm, &w.k_norm, w.attention_kind,
        Ghost(extension.row_parameters.qk_norm),
        Tracked(q_p.borrow()), Tracked(k_p.borrow()),
        Tracked(&wp.q_norm), Tracked(&wp.k_norm),
        Ghost(q_repr), Ghost(k_repr),
        Ghost(wp.q_norm.repr_1d()), Ghost(wp.k_norm.repr_1d()), scope4,
    );
    let ghost normalized = P::qk_norm_repr(
        q_repr, k_repr, wp.q_norm.repr_1d(), wp.k_norm.repr_1d(), extension.row_parameters.qk_norm,
    );
    let scope5 = Ghost(scope4@.insert(nq.id()).insert(nk.id()));
    let ((rq, rk), (rq_p, rk_p)) = P::rotary_embed(
        runtime, &nq, &nk, positions, w.attention_kind, Ghost(attention_geometry), Ghost(extension.row_parameters.rotary),
        Tracked(nq_p.borrow()), Tracked(nk_p.borrow()), Tracked(pp),
        Ghost(normalized.0), Ghost(normalized.1), Ghost(positions_repr), scope5,
    );
    let ghost rotated = P::rotary_embed_repr(
        positions_repr, normalized.0, normalized.1, attention_geometry, extension.row_parameters.rotary,
    );
    let scope6 = Ghost(scope5@.insert(rq.id()).insert(rk.id()));

    let (vv, vv_p) = prepare_value_cache_rows(
        runtime, &v, w.attention_kind, value_norm_epsilon,
        Tracked(v_p.borrow()), Ghost(v_repr), Ghost(attention_geometry), scope6,
    );
    let ghost vv_repr = value_cache_rows_repr(
        v_repr, common.head_dim, extension.row_parameters.value_norm_epsilon);
    let scope7 = Ghost(scope6@.insert(vv.id()));

    RT::store_kv_cache(
        runtime, &rk, &vv, k_cache, v_cache, slot_mapping,
        Tracked(rk_p.borrow()), Tracked(vv_p.borrow()),
        Tracked(kcp), Tracked(vcp), Tracked(sp),
        Ghost(rotated.1), Ghost(vv_repr),
        Ghost(k_cache_repr), Ghost(v_cache_repr), Ghost(slot_repr),
    );
    let ghost post_cache = RT::store_kv_cache_repr(
        rotated.1, vv_repr, k_cache_repr, v_cache_repr, slot_repr,
    );
    proof {
        MODEL::lemma_attention_pre_store_repr_shape(
            common, extension, hidden_repr, positions_repr,
        );
        RT::lemma_paged_attention_launch_ready_after_store(
            hidden_repr.len(), rotated.1, vv_repr,
            k_cache_repr, v_cache_repr, slot_repr,
            cu_q_repr, cu_k_repr,
            max_seqlen_q as nat, max_seqlen_k as nat, bt_repr,
        );
    }

    proof { reveal(crate::boundary::dense_layer_primitives::layer_attention_parameters_repr); }
    let (attended, attended_p) = P::paged_attention(
        runtime, &rq, k_cache, v_cache, block_table,
        cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
        w.attention_kind, sliding_window,
        Tracked(rq_p.borrow()), Tracked(kcp), Tracked(vcp),
        Tracked(btp), Tracked(cuqp), Tracked(cukp),
        Ghost(rotated.0), Ghost(post_cache.0), Ghost(post_cache.1),
        Ghost(bt_repr), Ghost(cu_q_repr), Ghost(cu_k_repr),
        Ghost(attention_geometry), Ghost(extension.attention_scale), scope7,
    );
    let ghost attended_repr = MODEL::paged_attention_repr(
        rotated.0, post_cache.0, post_cache.1,
        cu_q_repr, cu_k_repr,
        max_seqlen_q as nat, max_seqlen_k as nat, bt_repr,
        extension.attention,
        crate::boundary::dense_layer_primitives::layer_attention_parameters_repr(common, extension.attention_scale),
    );
    let scope8 = Ghost(scope7@.insert(attended.id()));
    let out = post_attention_and_mlp(
        runtime, w, Tracked(wp), hidden, Tracked(hp),
        &attended, Tracked(attended_p.borrow()),
        Ghost(MODEL::four_norm_layer_repr(common, extension)),
        Ghost(hidden_repr), Ghost(attended_repr), scope8,
    );
    proof {
        reveal(MODEL::decoder_layer_step_repr);
        reveal(ROWS::attention_pre_store_row);
        lemma_value_cache_rows_matches_layer_policy(
            v_repr, common.head_dim, extension.row_parameters.value_norm_epsilon);
        let pre = MODEL::attention_pre_store_repr(common, extension, hidden_repr, positions_repr);
        assert(rotated.0 =~= pre.0);
        assert(rotated.1 =~= pre.1);
        assert(vv_repr =~= pre.2);
    }
    out
}

} // verus!
