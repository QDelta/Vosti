//! Gemma-4 adapters for the shared checked four-norm decoder.
//! Physical weights and immutable layer policies bind to the shared full forward.

use crate::model_config::ModelArchitecture;
use crate::boundary::dense_layer_primitives as DLP;
use crate::boundary::four_norm_gated_primitives as P;
use crate::boundary::four_norm_gated_weights as W;
use crate::boundary::model_families::gemma4::weights as WEIGHTS;
use crate::exec::four_norm_gated_decoder as D;
use crate::exec::four_norm_gated_model as FOLD;
#[cfg(verus_only)]
use crate::proof::model::four_norm_gated::model as MODEL;
use crate::{types::*, proof::model::types::*, proof::tensor::types::*};
use crate::boundary::tensor_runtime as RT;
use vstd::prelude::*;

verus! {

pub proof fn lemma_engine_model_forward_ready(
    old_e: crate::exec::engine::Engine,
    reprs: crate::exec::engine::StepReprs,
    wp: &RT::ModelWeightsPerms,
    pre_kv: Seq<(KVCacheLayerRepr, KVCacheLayerRepr)>,
)
    requires
        wp.architecture() == ModelArchitecture::Gemma4Text,
        crate::proof::model::families::engine_forward_context(old_e, reprs, wp, pre_kv),
    ensures
        crate::exec::model::architecture_model_forward_ready(
            wp, reprs.input_ids, reprs.positions, pre_kv, reprs.slots,
            reprs.cu_q, reprs.cu_k, reprs.max_q, reprs.max_k, reprs.bt,
            reprs.scheduled.len(),
        ),
{
    reveal(crate::proof::model::families::engine_forward_context);
    crate::boundary::model_families::gemma4::lemma_execution_valid_implies_configuration_ready(
        &old_e.weights, &old_e.runtime, wp);
    crate::boundary::model_families::gemma4::lemma_weights_extension_repr(wp);
    crate::proof::tensor::geometry::lemma_cu_int_bounds(reprs.cu_q, reprs.scheduled.len() as int);
    assert forall|j: int| 0 <= j < reprs.scheduled.len() implies
        0 < #[trigger] reprs.cu_q[j + 1] <= reprs.input_ids.len() by {
        assert(0 <= reprs.cu_q[j]);
        assert(reprs.cu_q[j] < reprs.cu_q[j + 1]);
        assert(reprs.cu_q[j + 1] <= reprs.cu_q[reprs.scheduled.len() as int]);
    }
    reveal(forward_ready);
    reveal(crate::exec::model::architecture_model_forward_ready);
}


#[verifier::opaque]
pub open spec fn forward_ready(
    wp: &RT::ModelWeightsPerms, tokens: IntTensor1D, positions: IntTensor1D,
    pre: Seq<(KVCacheLayerRepr, KVCacheLayerRepr)>, slots: Seq<int>,
    cu_q: Seq<int>, cu_k: Seq<int>, max_q: nat, max_k: nat,
    bt: Seq<Seq<BlockId>>, count: nat,
) -> bool {
    wp.architecture() == ModelArchitecture::Gemma4Text
    && gemma4_config_valid(wp.gemma4_config())
    && pre.len() == wp.num_layers()
    && FOLD::inputs_ready(tokens, positions, pre, slots, cu_q, cu_k, max_q, max_k, bt, count)
}

// This adapter only borrows the bound weights and transports immutable policy
// and input facts. The complete tensor/KV execution is the shared composition.
#[verifier::spinoff_prover]
pub fn model_forward(
    runtime: &RT::ModelFamilyRuntime, weights: &WEIGHTS::Gemma4ModelWeights,
    Tracked(wp): Tracked<&RT::ModelWeightsPerms>,
    input_ids: &RT::Tensor, Tracked(ip): Tracked<&RT::TensorPerm>,
    positions: &RT::Tensor, Tracked(pp): Tracked<&RT::TensorPerm>,
    kv_caches: &Vec<(RT::Tensor, RT::Tensor)>,
    Tracked(kv_perms): Tracked<&mut RT::KVCachePerms>,
    block_table: &RT::Tensor, Tracked(btp): Tracked<&RT::TensorPerm>,
    slot_mapping: &RT::Tensor, Tracked(sp): Tracked<&RT::TensorPerm>,
    cu_seqlens_q: &RT::Tensor, Tracked(cuqp): Tracked<&RT::TensorPerm>,
    cu_seqlens_k: &RT::Tensor, Tracked(cukp): Tracked<&RT::TensorPerm>,
    max_q: usize, max_k: usize, count: usize,
    Ghost(tokens): Ghost<IntTensor1D>, Ghost(pos): Ghost<IntTensor1D>,
    Ghost(slots): Ghost<Seq<int>>, Ghost(cu_q): Ghost<Seq<int>>, Ghost(cu_k): Ghost<Seq<int>>,
    Ghost(bt): Ghost<Seq<Seq<BlockId>>>,
) -> (out: (RT::Tensor, Tracked<RT::TensorPerm>))
    requires
        RT::paged_attention_numeric_domain(),
        runtime_matches_weights(runtime, weights), WEIGHTS::model_weights_bound(weights, wp),
        RT::int_tensor_repr_1d(*ip, *input_ids, tokens),
        RT::int_tensor_repr_1d(*pp, *positions, pos),
        RT::int_tensor_repr_1d(*sp, *slot_mapping, slots),
        RT::int_tensor_repr_1d(*cuqp, *cu_seqlens_q, cu_q),
        RT::int_tensor_repr_1d(*cukp, *cu_seqlens_k, cu_k),
        RT::block_table_repr(*btp, *block_table, bt),
        kv_caches.len() == weights.layers.len(), old(kv_perms).len() == weights.layers.len(),
        RT::kv_perms_ids_distinct(*old(kv_perms)),
        old(kv_perms).extracted() == Set::<int>::empty(),
        forall|j: int| 0 <= j < weights.layers.len() ==>
            #[trigger] kv_caches[j].0.id() == old(kv_perms).k_id(j)
            && kv_caches[j].1.id() == old(kv_perms).v_id(j),
        forward_ready(wp, tokens, pos, Seq::new(weights.layers.len() as nat, |i: int|
            (old(kv_perms).k_repr(i), old(kv_perms).v_repr(i))),
            slots, cu_q, cu_k, max_q as nat, max_k as nat, bt, count as nat),
    ensures ({
        let pre = Seq::new(weights.layers.len() as nat, |i: int|
            (old(kv_perms).k_repr(i), old(kv_perms).v_repr(i)));
        let logits = MODEL::model_forward_logits_repr(RT::model_weights_repr_of(wp),
            wp.gemma4_config().decoder, tokens, pos, pre, slots, cu_q, cu_k,
            max_q as nat, max_k as nat, bt);
        let post = MODEL::model_forward_kv_reprs(RT::model_weights_repr_of(wp),
            wp.gemma4_config().decoder, tokens, pos, pre, slots, cu_q, cu_k,
            max_q as nat, max_k as nat, bt);
        &&& RT::tensor_repr_2d(out.1@, out.0, Seq::new(count as nat, |j: int|
            RT::select_sample_logits_repr(logits, cu_q, j as nat)))
        &&& final(kv_perms).len() == old(kv_perms).len()
        &&& final(kv_perms).extracted() == Set::<int>::empty()
        &&& RT::kv_perms_ids_distinct(*final(kv_perms))
        &&& forall|j: int| 0 <= j < weights.layers.len() ==> {
            &&& #[trigger] final(kv_perms).k_id(j) == old(kv_perms).k_id(j)
            &&& final(kv_perms).v_id(j) == old(kv_perms).v_id(j)
            &&& final(kv_perms).k_repr(j) == post[j].0
            &&& final(kv_perms).v_repr(j) == post[j].1
        }
    }),
{
    proof {
        reveal(forward_ready);
        crate::boundary::model_families::gemma4::lemma_bound_architecture_repr_valid(weights, wp);
        RT::lemma_model_weights_common_layers_repr_projection(wp);
    }
    #[cfg(not(verus_only))]
    if !crate::boundary::model_families::gemma4::deployment::reports_backend_qualified(runtime) {
        panic!("Gemma4Text forward requires a backend-qualified runtime");
    }
    let tracked physical = wp.tracked_borrow_four_norm_gated_weights();
    let tracked refs = physical.borrowed_layer_permissions(weights.layers.len() as int);
    let ghost wr = RT::model_weights_repr_of(wp);
    let ghost decoder = wp.gemma4_config().decoder;
    let ghost pre = Seq::new(weights.layers.len() as nat, |i: int|
        (kv_perms.k_repr(i), kv_perms.v_repr(i)));
    proof {
        assert(FOLD::inputs_ready(tokens, pos, pre, slots, cu_q, cu_k,
            max_q as nat, max_k as nat, bt, count as nat));
        assert forall|i: int| 0 <= i < weights.layers.len() implies
            #[trigger] RT::store_kv_cache_launch_ready(tokens.len(),
                kv_perms.k_repr(i), kv_perms.v_repr(i), slots) by {
            assert(pre[i] == (kv_perms.k_repr(i), kv_perms.v_repr(i)));
            assert(RT::store_kv_cache_launch_ready(tokens.len(), pre[i].0, pre[i].1, slots));
        }
        assert forall|i: int| 0 <= i < weights.layers.len() implies
            #[trigger] RT::paged_attention_launch_ready(tokens.len(),
                kv_perms.k_repr(i), kv_perms.v_repr(i), cu_q, cu_k,
                max_q as nat, max_k as nat, bt) by {
            assert(pre[i] == (kv_perms.k_repr(i), kv_perms.v_repr(i)));
            assert(RT::paged_attention_launch_ready(tokens.len(), pre[i].0, pre[i].1,
                cu_q, cu_k, max_q as nat, max_k as nat, bt));
        }
        assert forall|i: int| 0 <= i < weights.layers.len() implies
            #[trigger] D::layer_execution_policy_matches(runtime, &weights.layers[i], refs[i],
                weights.config.sliding_window as nat,
                Some(weights.config.rms_norm_epsilon),
                decoder.layers[i]) by {
            lemma_layer_execution_policy(runtime, weights, wp, i);
        }
        assert(FOLD::layers_ready(runtime, &weights.layers, &refs,
            weights.config.sliding_window as nat,
            Some(weights.config.rms_norm_epsilon),
            wr.layers, decoder.layers));
        WEIGHTS::lemma_physical_deployment_config_valid(weights, physical);
    }
    FOLD::model_forward(runtime, &weights.embed_weight, &weights.final_norm, &weights.lm_head,
        Tracked(&physical.embed_weight), Tracked(&physical.final_norm), Tracked(&physical.lm_head),
        &weights.layers, Tracked(&refs), weights.config.geometry.hidden_size,
        weights.config.sliding_window, Some(weights.config.rms_norm_epsilon), weights.config.final_logit_softcap,
        input_ids, Tracked(ip), positions, Tracked(pp), kv_caches, Tracked(kv_perms),
        block_table, Tracked(btp), slot_mapping, Tracked(sp),
        cu_seqlens_q, Tracked(cuqp), cu_seqlens_k, Tracked(cukp), max_q, max_k, count,
        Ghost(wr), Ghost(decoder), Ghost(tokens), Ghost(pos), Ghost(slots), Ghost(cu_q), Ghost(cu_k), Ghost(bt))
}

// Unlike the semantic tag alone, this premise binds a qualified primitive
// capability to the exact physical checkpoint geometry and numerical policy.
pub open spec fn runtime_matches_weights(
    runtime: &RT::ModelFamilyRuntime, weights: &WEIGHTS::Gemma4ModelWeights,
) -> bool {
    RT::family_runtime_execution_valid(runtime)
    && RT::family_runtime_deployment_config_repr(runtime) == Some(
        ModelDeploymentConfigRepr::Gemma4Text(gemma4_deployment_config_repr(
            weights.config, WEIGHTS::physical_attention_kinds(weights))))
}

pub proof fn lemma_layer_execution_policy(
    runtime: &RT::ModelFamilyRuntime, weights: &WEIGHTS::Gemma4ModelWeights,
    perms: &RT::ModelWeightsPerms, i: int,
)
    requires
        runtime_matches_weights(runtime, weights),
        WEIGHTS::model_weights_bound(weights, perms),
        0 <= i < weights.layers.len(),
    ensures D::layer_execution_policy_matches(
        runtime, &weights.layers[i], &perms.four_norm_gated_weights().layers[i],
        weights.config.sliding_window as nat,
        Some(weights.config.rms_norm_epsilon),
        perms.gemma4_config().decoder.layers[i]),
{
    let physical = perms.four_norm_gated_weights();
    let lp = &physical.layers[i];
    let kind = weights.layers[i].attention_kind;
    let config = gemma4_deployment_config_repr(weights.config,
        WEIGHTS::physical_attention_kinds(weights));
    WEIGHTS::lemma_physical_deployment_config_valid(weights, &physical);
    P::lemma_gemma4_deployment_policy_is_exact(runtime, config);
    let geometry = gemma4_layer_attention_geometry(config, kind);
    assert(W::layer_shapes_valid(lp, config.geometry.hidden_size,
        config.geometry.intermediate_size, geometry));
    assert(W::layer_weights_valid(&weights.layers[i], lp));
    let common = W::layer_weights_common_repr_of(lp);
    assert(geometry.head_dim > 0);
    vstd::arithmetic::div_mod::lemma_div_by_multiple(
        geometry.num_attention_heads as int, geometry.head_dim as int);
    vstd::arithmetic::div_mod::lemma_div_by_multiple(
        geometry.num_key_value_heads as int, geometry.head_dim as int);
    assert(DLP::layer_attention_geometry_repr(common) == geometry);
    assert(perms.gemma4_config().decoder.layers[i]
        == WEIGHTS::layer_extension_repr_of(config, kind, lp));
}

} // verus!
