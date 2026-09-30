//! Checked Gemma3 text model-forward composition.

pub(crate) mod readiness;
use crate::model_config::{ModelArchitecture, ModelConfig};
#[cfg(verus_only)]
pub use readiness::{forward_ready, lemma_engine_model_forward_ready};

use crate::boundary::model_families::gemma3 as GEMMA_BOUNDARY;
use crate::boundary::four_norm_gated_primitives as P;
use crate::boundary::four_norm_gated_weights as W;
use crate::exec::four_norm_gated_decoder as D;
use crate::exec::four_norm_gated_model as FOLD;
use crate::proof::model::architecture as MA;
#[cfg(verus_only)]
use crate::proof::model::families::four_norm_gated::semantics as G3;
use crate::{types::*, proof::model::types::*, proof::tensor::types::*};
use crate::boundary::tensor_runtime as RT;
use vstd::prelude::*;

verus! {

// Keep the qualified backend's concrete semantic body behind the same
// family-adapter reduction used by the direct Qwen implementation.
pub proof fn lemma_forward_result_matches_dispatch(
    wp: &RT::ModelWeightsPerms,
    wr: ModelWeightsRepr,
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
    requires wp.architecture() == ModelArchitecture::Gemma3Text,
    ensures
        MA::model_forward_logits_repr(
            wr, RT::model_weights_architecture_repr_of(wp),
            input_ids, positions, pre_kv, slots,
            cu_q, cu_k, max_q, max_k, block_table,
        ) == G3::model_forward_logits_repr(
            wr, GEMMA_BOUNDARY::weights_extension_repr_of(wp),
            input_ids, positions, pre_kv, slots,
            cu_q, cu_k, max_q, max_k, block_table,
        ),
        MA::model_forward_kv_reprs(
            wr, RT::model_weights_architecture_repr_of(wp),
            input_ids, positions, pre_kv, slots,
            cu_q, cu_k, max_q, max_k, block_table,
        ) == G3::model_forward_kv_reprs(
            wr, GEMMA_BOUNDARY::weights_extension_repr_of(wp),
            input_ids, positions, pre_kv, slots,
            cu_q, cu_k, max_q, max_k, block_table,
        ),
{
    GEMMA_BOUNDARY::lemma_architecture_repr(wp);
    MA::lemma_gemma3_forward_dispatch(
        wr, GEMMA_BOUNDARY::weights_extension_repr_of(wp),
        input_ids, positions, pre_kv, slots,
        cu_q, cu_k, max_q, max_k, block_table,
    );
    reveal(crate::proof::model::families::four_norm_gated::forward_logits_repr);
    reveal(crate::proof::model::families::four_norm_gated::forward_kv_reprs);
}

pub proof fn lemma_architecture_forward_ready_implies_launch_ready(
    wp: &RT::ModelWeightsPerms,
    input_ids_repr: IntTensor1D,
    positions_repr: IntTensor1D,
    pre_kv_reprs: Seq<(KVCacheLayerRepr, KVCacheLayerRepr)>,
    slot_repr: Seq<int>,
    cu_q_repr: Seq<int>,
    cu_k_repr: Seq<int>,
    max_seqlen_q: nat,
    max_seqlen_k: nat,
    bt_repr: Seq<Seq<BlockId>>,
    num_seqs: nat,
)
    requires
        wp.architecture() == ModelArchitecture::Gemma3Text,
        crate::exec::model::architecture_model_forward_ready(
            wp, input_ids_repr, positions_repr, pre_kv_reprs, slot_repr,
            cu_q_repr, cu_k_repr, max_seqlen_q, max_seqlen_k, bt_repr,
            num_seqs,
        ),
    ensures
        forall|i: int| 0 <= i < wp.num_layers() as int ==>
            #[trigger] RT::store_kv_cache_launch_ready(
                input_ids_repr.len(), pre_kv_reprs[i].0,
                pre_kv_reprs[i].1, slot_repr,
            ),
        forall|i: int| 0 <= i < wp.num_layers() as int ==>
            #[trigger] RT::paged_attention_launch_ready(
                input_ids_repr.len(), pre_kv_reprs[i].0,
                pre_kv_reprs[i].1, cu_q_repr, cu_k_repr,
                max_seqlen_q, max_seqlen_k, bt_repr,
            ),
{
    reveal(crate::exec::model::architecture_model_forward_ready);
    reveal(readiness::forward_ready);
}

/// Prove the fixed Gemma3 policy satisfies the shared layer contract.
pub proof fn lemma_layer_execution_policy(
    runtime: &RT::ModelFamilyRuntime,
    w: &RT::Gemma3LayerWeights,
    wp: &RT::Gemma3LayerWeightsPerms,
    sliding_window: nat,
    attention_scale: AttentionScaleRepr,
)
    requires
        RT::gemma3_runtime_attention_parameters_match(
            runtime,
            crate::boundary::dense_layer_primitives::layer_attention_geometry_repr(
                GEMMA_BOUNDARY::weights::layer_weights_common_repr_of(wp)),
            sliding_window, attention_scale),
        GEMMA_BOUNDARY::weights::layer_weights_valid(w, wp),
    ensures
        D::layer_execution_policy_matches(runtime, w, wp, sliding_window, None,
            GEMMA_BOUNDARY::weights::layer_weights_extension_repr_of(
                wp, w.attention_kind, sliding_window, attention_scale)),
{
    let config = match RT::family_runtime_deployment_config_repr(runtime) {
        Some(ModelDeploymentConfigRepr::Gemma3Text(config)) => config,
        _ => { assert(false); arbitrary() },
    };
    reveal(RT::family_runtime_configuration_valid);
    reveal(model_deployment_config_valid);
    reveal(gemma3_deployment_config_valid);
}

/// Borrow existing permissions into a proof-only, family-neutral collection.
/// No TensorPerm is copied and no runtime allocation or new axiom is needed.
pub(crate) proof fn borrowed_layer_permissions<'a>(
    tracked wp: &'a RT::ModelWeightsPerms,
    count: int,
) -> (tracked refs: Map<int, &'a W::FourNormGatedLayerWeightsPerms>)
    requires
        wp.architecture() == ModelArchitecture::Gemma3Text,
        0 <= count <= wp.num_layers(),
    ensures
        forall|i: int| #[trigger] refs.dom().contains(i) <==> 0 <= i < count,
        forall|i: int| 0 <= i < count ==> *#[trigger] refs[i] == wp.gemma3_layer(i),
    decreases count,
{
    if count == 0 {
        Map::tracked_empty()
    } else {
        let i = count - 1;
        let tracked mut refs = borrowed_layer_permissions(wp, i);
        let tracked layer = wp.tracked_borrow_gemma3_layer(i);
        refs.tracked_insert(i, layer);
        refs
    }
}

// Checked family policy/weight adapter. The dispatcher supplies a qualified
// runtime; tensor transitions are composed by the shared primitive, layer,
// multi-layer fold and sampled-projection functions below.
#[verifier::spinoff_prover]
pub(crate) fn model_forward(
    runtime: &RT::ModelFamilyRuntime,
    config: &ModelConfig,
    gemma: &RT::Gemma3ModelWeights,
    Tracked(wp): Tracked<&RT::ModelWeightsPerms>,
    input_ids: &RT::Tensor,
    Tracked(ip): Tracked<&RT::TensorPerm>,
    positions: &RT::Tensor,
    Tracked(pp): Tracked<&RT::TensorPerm>,
    kv_caches: &Vec<(RT::Tensor, RT::Tensor)>,
    Tracked(kv_perms): Tracked<&mut RT::KVCachePerms>,
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
    num_seqs: usize,
    Ghost(input_ids_repr): Ghost<IntTensor1D>,
    Ghost(positions_repr): Ghost<IntTensor1D>,
    Ghost(cu_q_repr): Ghost<Seq<int>>,
    Ghost(cu_k_repr): Ghost<Seq<int>>,
    Ghost(bt_repr): Ghost<Seq<Seq<BlockId>>>,
    Ghost(slot_repr): Ghost<Seq<int>>,
) -> (out: (RT::Tensor, Tracked<RT::TensorPerm>))
    requires
        RT::paged_attention_numeric_domain(),
        RT::gemma3_runtime_matches_weights(runtime, gemma),
        wp.architecture() == ModelArchitecture::Gemma3Text,
        RT::gemma3_model_weights_bound(gemma, wp),
        gemma.layers.len() == config.num_layers,
        RT::int_tensor_repr_1d(*ip, *input_ids, input_ids_repr),
        RT::int_tensor_repr_1d(*pp, *positions, positions_repr),
        RT::block_table_repr(*bt_perm, *block_table, bt_repr),
        RT::int_tensor_repr_1d(*sp, *slot_mapping, slot_repr),
        RT::int_tensor_repr_1d(*cuq_perm, *cu_seqlens_q, cu_q_repr),
        RT::int_tensor_repr_1d(*cuk_perm, *cu_seqlens_k, cu_k_repr),
        kv_caches.len() == config.num_layers as nat,
        old(kv_perms).len() == config.num_layers as nat,
        RT::kv_perms_ids_distinct(*old(kv_perms)),
        old(kv_perms).extracted() == Set::<int>::empty(),
        forall|i: int| 0 <= i < config.num_layers as int ==>
            #[trigger] kv_caches[i].0.id() == old(kv_perms).k_id(i)
            && kv_caches[i].1.id() == old(kv_perms).v_id(i),
        readiness::forward_ready(
            wp, input_ids_repr, positions_repr,
            Seq::new(config.num_layers as nat, |i: int|
                (old(kv_perms).k_repr(i), old(kv_perms).v_repr(i))),
            slot_repr, cu_q_repr, cu_k_repr,
            max_seqlen_q as nat, max_seqlen_k as nat, bt_repr,
            num_seqs as nat,
        ),
    ensures ({
        let pre_kv_reprs: Seq<(KVCacheLayerRepr, KVCacheLayerRepr)> =
            Seq::new(config.num_layers as nat, |i: int|
                (old(kv_perms).k_repr(i), old(kv_perms).v_repr(i)));
        let gemma_repr = GEMMA_BOUNDARY::weights_extension_repr_of(wp);
        let logits_repr = G3::model_forward_logits_repr(
            RT::model_weights_repr_of(wp), gemma_repr,
            input_ids_repr, positions_repr, pre_kv_reprs, slot_repr,
            cu_q_repr, cu_k_repr,
            max_seqlen_q as nat, max_seqlen_k as nat, bt_repr,
        );
        let post_kv_reprs = G3::model_forward_kv_reprs(
            RT::model_weights_repr_of(wp), gemma_repr,
            input_ids_repr, positions_repr, pre_kv_reprs, slot_repr,
            cu_q_repr, cu_k_repr,
            max_seqlen_q as nat, max_seqlen_k as nat, bt_repr,
        );
        let (logits, lp) = out;
        &&& RT::tensor_repr_2d(
            lp@, logits,
            Seq::new(num_seqs as nat, |i: int|
                RT::select_sample_logits_repr(logits_repr, cu_q_repr, i as nat)),
        )
        &&& final(kv_perms).len() == old(kv_perms).len()
        &&& final(kv_perms).extracted() == Set::<int>::empty()
        &&& RT::kv_perms_ids_distinct(*final(kv_perms))
        &&& forall|j: int| 0 <= j < config.num_layers as int ==> {
            &&& #[trigger] final(kv_perms).k_id(j) == old(kv_perms).k_id(j)
            &&& final(kv_perms).v_id(j) == old(kv_perms).v_id(j)
            &&& final(kv_perms).k_repr(j) == post_kv_reprs[j].0
            &&& final(kv_perms).v_repr(j) == post_kv_reprs[j].1
        }
    }),
{
    proof {
        reveal(RT::gemma3_model_weights_bound);
        reveal(GEMMA_BOUNDARY::weights::model_weights_bound);
        reveal(RT::gemma3_runtime_matches_weights);
        reveal(RT::gemma3_physical_deployment_config_repr);
        reveal(RT::gemma3_runtime_hidden_size_matches);
        GEMMA_BOUNDARY::lemma_common_layers_repr(wp);
        GEMMA_BOUNDARY::lemma_weights_extension_repr(wp);
        assert(gemma3_config_valid(
            GEMMA_BOUNDARY::weights_extension_repr_of(wp),
        ));
        assert(gemma.config.sliding_window > 0);
        assert(RT::gemma3_runtime_configuration_valid(runtime));
        assert(RT::gemma3_runtime_hidden_size_matches(
            runtime, gemma.config.geometry.hidden_size as nat,
        ));
    }
    reveal(readiness::forward_ready);
    reveal(readiness::packed_attention_rows_ready);
    #[cfg(not(verus_only))]
    if !GEMMA_BOUNDARY::deployment::reports_backend_qualified(runtime) {
        panic!("Gemma3Text forward requires a backend-qualified runtime");
    }

    broadcast use {
        RT::lemma_linear_repr_shape,
        G3::lemma_scaled_embed_repr_shape,
    };

    let n = gemma.layers.len();
    let ghost pre_kv_reprs: Seq<(KVCacheLayerRepr, KVCacheLayerRepr)> =
        Seq::new(n as nat, |j: int|
            (kv_perms.k_repr(j), kv_perms.v_repr(j)));
    let ghost wr = RT::model_weights_repr_of(wp);
    let ghost gemma_repr = GEMMA_BOUNDARY::weights_extension_repr_of(wp);
    let ghost common_layers = wr.layers;
    let ghost extension_layers = gemma_repr.layers;

    proof {
        assert(n == config.num_layers);
        assert(pre_kv_reprs.len() == n as nat);
        assert(pre_kv_reprs =~= Seq::new(config.num_layers as nat, |j: int|
            (old(kv_perms).k_repr(j), old(kv_perms).v_repr(j))));
        assert(readiness::forward_ready(
            wp, input_ids_repr, positions_repr, pre_kv_reprs, slot_repr,
            cu_q_repr, cu_k_repr,
            max_seqlen_q as nat, max_seqlen_k as nat, bt_repr,
            num_seqs as nat,
        ));
        assert(readiness::packed_attention_rows_ready(
            cu_q_repr, cu_k_repr, num_seqs as nat,
        ));
        assert forall|j: int| 0 <= j < n as int implies
            #[trigger] RT::store_kv_cache_launch_ready(
                input_ids_repr.len(), pre_kv_reprs[j].0,
                pre_kv_reprs[j].1, slot_repr,
            ) by {};
        assert forall|j: int| 0 <= j < n as int implies
            #[trigger] RT::paged_attention_launch_ready(
                input_ids_repr.len(), pre_kv_reprs[j].0,
                pre_kv_reprs[j].1, cu_q_repr, cu_k_repr,
                max_seqlen_q as nat, max_seqlen_k as nat, bt_repr,
            ) by {};
        GEMMA_BOUNDARY::lemma_weights_extension_repr(wp);
        assert(common_layers.len() == n as nat);
        assert(extension_layers.len() == n as nat);
        assert forall|j: int| 0 <= j < num_seqs as int implies
            cu_q_repr[j] < #[trigger] cu_q_repr[j + 1] by {
            readiness::packed_attention_rows_ready_at(
                cu_q_repr, cu_k_repr, num_seqs as nat, j,
            );
        };
    }


    let tracked layer_refs = borrowed_layer_permissions(wp, n as int);
    proof {
        assert forall|i: int| 0 <= i < n as int implies
            #[trigger] D::layer_execution_policy_matches(
                runtime, &gemma.layers[i], layer_refs[i],
                gemma.config.sliding_window as nat, None, extension_layers[i]) by {
            let layer_p = layer_refs[i];
            assert(*layer_p == wp.gemma3_layer(i));
            assert(GEMMA_BOUNDARY::weights::layer_weights_valid(
                &gemma.layers[i], &wp.gemma3_layer(i)));
            assert(gemma.layers[i].attention_kind == wp.gemma3_attention_kind(i));
            assert(gemma.config.sliding_window as nat == wp.gemma3_config().sliding_window);
            GEMMA_BOUNDARY::lemma_weights_extension_repr(wp);
            assert(extension_layers[i]
                == GEMMA_BOUNDARY::weights::layer_weights_extension_repr_of(
                    layer_p, gemma.layers[i].attention_kind,
                    gemma.config.sliding_window as nat,
                    wp.gemma3_config().attention_scale));
            GEMMA_BOUNDARY::weights::lemma_layer_attention_geometry_matches_config(
                gemma, wp, i);
            reveal(RT::gemma3_runtime_matches_weights);
            reveal(RT::gemma3_physical_deployment_config_repr);
            reveal(RT::gemma3_runtime_attention_parameters_match);
            lemma_layer_execution_policy(runtime, &gemma.layers[i], layer_p,
                gemma.config.sliding_window as nat, wp.gemma3_config().attention_scale);
        }
        assert(FOLD::layers_ready(runtime, &gemma.layers, &layer_refs,
            gemma.config.sliding_window as nat, None, common_layers, extension_layers));
    }
    proof {
        crate::proof::tensor::geometry::lemma_cu_int_bounds(cu_q_repr, num_seqs as int);
        assert forall|j: int| 0 <= j < gemma.layers.len() implies
            #[trigger] RT::store_kv_cache_launch_ready(input_ids_repr.len(),
                kv_perms.k_repr(j), kv_perms.v_repr(j), slot_repr) by {
            assert(pre_kv_reprs[j] == (kv_perms.k_repr(j), kv_perms.v_repr(j)));
            assert(RT::store_kv_cache_launch_ready(input_ids_repr.len(),
                pre_kv_reprs[j].0, pre_kv_reprs[j].1, slot_repr));
        }
        assert forall|j: int| 0 <= j < gemma.layers.len() implies
            #[trigger] RT::paged_attention_launch_ready(input_ids_repr.len(),
                kv_perms.k_repr(j), kv_perms.v_repr(j), cu_q_repr, cu_k_repr,
                max_seqlen_q as nat, max_seqlen_k as nat, bt_repr) by {
            assert(pre_kv_reprs[j] == (kv_perms.k_repr(j), kv_perms.v_repr(j)));
            assert(RT::paged_attention_launch_ready(input_ids_repr.len(),
                pre_kv_reprs[j].0, pre_kv_reprs[j].1, cu_q_repr, cu_k_repr,
                max_seqlen_q as nat, max_seqlen_k as nat, bt_repr));
        }
        assert forall|j: int| 0 <= j < num_seqs as int implies
            0 < #[trigger] cu_q_repr[j + 1] <= input_ids_repr.len() by {
            assert(0 <= cu_q_repr[j]);
            assert(cu_q_repr[j] < cu_q_repr[j + 1]);
            assert(cu_q_repr[j + 1] <= cu_q_repr[num_seqs as int]);
        }
    }
    let tracked embed_p = wp.tracked_borrow_embed_weight();
    let tracked final_norm_p = wp.tracked_borrow_final_norm();
    let tracked lm_head_p = wp.tracked_borrow_lm_head();
    FOLD::model_forward(
        runtime, &gemma.embed_weight, &gemma.final_norm, &gemma.lm_head,
        Tracked(embed_p), Tracked(final_norm_p), Tracked(lm_head_p),
        &gemma.layers, Tracked(&layer_refs), gemma.config.geometry.hidden_size,
        gemma.config.sliding_window, None, None,
        input_ids, Tracked(ip), positions, Tracked(pp), kv_caches, Tracked(kv_perms),
        block_table, Tracked(bt_perm), slot_mapping, Tracked(sp),
        cu_seqlens_q, Tracked(cuq_perm), cu_seqlens_k, Tracked(cuk_perm),
        max_seqlen_q, max_seqlen_k, num_seqs, Ghost(wr), Ghost(gemma_repr),
        Ghost(input_ids_repr), Ghost(positions_repr), Ghost(slot_repr),
        Ghost(cu_q_repr), Ghost(cu_k_repr), Ghost(bt_repr),
    )
}



} // verus!
