//! Family-neutral checked whole-model loop for dense SwiGLU decoders.

use crate::model_config::{ModelArchitecture, ModelConfig};
use crate::boundary::dense_swiglu_decoder as DENSE_WEIGHTS;
use crate::proof::model::dense_swiglu::batch_invariance as BI;
use crate::proof::model::dense_swiglu::cache_semantics as CC;
#[cfg(verus_only)]
use crate::proof::model::dense_swiglu::semantics as BD;
use crate::exec::dense_swiglu_decoder as DENSE_EXEC;
use crate::exec::model_kv_loop as KV_LOOP;
use crate::proof::model::architecture as MA;
use crate::{types::*, proof::model::types::*, proof::tensor::types::*};
use crate::boundary::tensor_runtime as RT;
use crate::proof::tensor::shape as TS;
use vstd::prelude::*;

verus! {

pub open spec fn layer_execution_ready(
    runtime: &RT::ModelFamilyRuntime,
    layer: &DENSE_WEIGHTS::DenseSwiGluLayerWeights,
    perms: &DENSE_WEIGHTS::DenseSwiGluLayerWeightsPerms,
    head_dim: nat,
    config: DenseSwiGluForwardConfigRepr,
) -> bool {
    &&& DENSE_WEIGHTS::layer_weights_valid(layer, perms)
    &&& RT::dense_swiglu_runtime_qk_norm_matches(runtime,
        qk_norm_weights_kind(DENSE_WEIGHTS::qk_norm_weights_repr_of(&perms.qk_norm)))
    &&& RT::dense_swiglu_runtime_rotary_matches(runtime,
        crate::boundary::dense_layer_primitives::layer_attention_geometry_repr(
            DENSE_WEIGHTS::layer_weights_repr_of(perms, head_dim)), config.rotary)
}

pub open spec fn model_execution_ready(
    runtime: &RT::ModelFamilyRuntime,
    embed_weight: &RT::Tensor,
    layers: &Vec<DENSE_WEIGHTS::DenseSwiGluLayerWeights>,
    final_norm: &RT::Tensor,
    lm_head: &RT::Tensor,
    wp: &RT::ModelWeightsPerms,
    head_dim: nat,
    forward_config: DenseSwiGluForwardConfigRepr,
) -> bool {
    &&& (wp.architecture() == ModelArchitecture::Qwen3
        || wp.architecture() == ModelArchitecture::Llama3)
    &&& RT::family_runtime_execution_valid(runtime)
    &&& layers.len() == wp.num_layers()
    &&& embed_weight.id() == wp.embed_weight_id()
    &&& final_norm.id() == wp.final_norm_id()
    &&& lm_head.id() == wp.lm_head_id()
    &&& TS::rectangular(wp.embed_weight_repr())
    &&& TS::rectangular(wp.lm_head_repr())
    &&& RT::dense_swiglu_runtime_rms_norm_matches(
        runtime, forward_config.rms_norm_epsilon,
    )
    &&& RT::model_weights_repr_of(wp).layers == Seq::new(
        wp.num_layers(),
        |i: int| DENSE_WEIGHTS::layer_weights_repr_of(
            &wp.dense_swiglu_layer(i), head_dim,
        ),
    )
    &&& forall|i: int| 0 <= i < layers.len() as int ==>
        #[trigger] layer_execution_ready(runtime, &layers[i],
            &wp.dense_swiglu_layer(i), head_dim, forward_config)
}

// One semantic bridge for every architecture using this exact composition.
// Family adapters prove only that their admitted payload selects `family`.
pub proof fn lemma_forward_result_matches_dispatch(
    wp: &RT::ModelWeightsPerms,
    wr: ModelWeightsRepr,
    family: DenseSwiGluDecoderConfigRepr,
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
        RT::model_weights_architecture_repr_of(wp)
            == ModelWeightsArchitectureRepr::Qwen3(family)
        || RT::model_weights_architecture_repr_of(wp)
            == ModelWeightsArchitectureRepr::Llama3(family),
    ensures
        MA::model_forward_logits_repr(
            wr, RT::model_weights_architecture_repr_of(wp),
            input_ids, positions, pre_kv, slots,
            cu_q, cu_k, max_q, max_k, block_table,
        ) == BD::model_forward_logits_repr(
            dense_swiglu_forward_config_repr(family),
            wr, input_ids, positions, pre_kv, slots,
            cu_q, cu_k, max_q, max_k, block_table,
        ),
        MA::model_forward_kv_reprs(
            wr, RT::model_weights_architecture_repr_of(wp),
            input_ids, positions, pre_kv, slots,
            cu_q, cu_k, max_q, max_k, block_table,
        ) == CC::model_forward_kv_reprs(
            dense_swiglu_forward_config_repr(family),
            wr, input_ids, positions, pre_kv, slots,
            cu_q, cu_k, max_q, max_k, block_table,
        ),
{
    match RT::model_weights_architecture_repr_of(wp) {
        ModelWeightsArchitectureRepr::Qwen3(qwen) => {
            MA::lemma_qwen3_forward_dispatch(
                wr, qwen, input_ids, positions, pre_kv, slots,
                cu_q, cu_k, max_q, max_k, block_table,
            );
        },
        ModelWeightsArchitectureRepr::Llama3(llama) => {
            MA::lemma_llama3_forward_dispatch(
                wr, llama, input_ids, positions, pre_kv, slots,
                cu_q, cu_k, max_q, max_k, block_table,
            );
        },
        ModelWeightsArchitectureRepr::Gemma3Text(_) => {
            assert(false);
        },
        ModelWeightsArchitectureRepr::Gemma4Text(_) => { assert(false); },
    }
    reveal(crate::proof::model::families::dense_swiglu::forward_logits_repr);
    reveal(crate::proof::model::families::dense_swiglu::forward_kv_reprs);
}

#[verifier::spinoff_prover]
pub(crate) fn model_forward(
    runtime: &RT::ModelFamilyRuntime,
    config: &ModelConfig,
    embed_weight: &RT::Tensor,
    layers: &Vec<DENSE_WEIGHTS::DenseSwiGluLayerWeights>,
    final_norm: &RT::Tensor,
    lm_head: &RT::Tensor,
    Tracked(wp): Tracked<&RT::ModelWeightsPerms>,
    input_ids: &RT::Tensor,
    Tracked(ip): Tracked<&RT::TensorPerm>,
    positions: &RT::Tensor,
    Tracked(pp): Tracked<&RT::TensorPerm>,
    kv_caches: &Vec<(RT::Tensor, RT::Tensor)>,
    Tracked(kv_perms): Tracked<&mut RT::KVCachePerms>,
    block_table: &RT::Tensor, Tracked(bt_perm): Tracked<&RT::TensorPerm>,
    slot_mapping: &RT::Tensor, Tracked(sp): Tracked<&RT::TensorPerm>,
    cu_seqlens_q: &RT::Tensor, Tracked(cuq_perm): Tracked<&RT::TensorPerm>,
    cu_seqlens_k: &RT::Tensor, Tracked(cuk_perm): Tracked<&RT::TensorPerm>,
    max_seqlen_q: usize, max_seqlen_k: usize,
    num_seqs: usize,
    Ghost(input_ids_repr): Ghost<IntTensor1D>,
    Ghost(positions_repr): Ghost<IntTensor1D>,
    Ghost(cu_q_repr): Ghost<Seq<int>>,
    Ghost(cu_k_repr): Ghost<Seq<int>>,
    Ghost(bt_repr): Ghost<Seq<Seq<BlockId>>>,
    Ghost(slot_repr): Ghost<Seq<int>>,
    Ghost(head_dim): Ghost<nat>,
    Ghost(forward_config): Ghost<DenseSwiGluForwardConfigRepr>,
) -> (out: (RT::Tensor, Tracked<RT::TensorPerm>))
    requires
        RT::paged_attention_numeric_domain(),
        model_execution_ready(
            runtime, embed_weight, layers, final_norm, lm_head,
            wp, head_dim, forward_config,
        ),
        layers.len() == config.num_layers,
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
        input_ids_repr.len() == positions_repr.len(),
        slot_repr.len() == input_ids_repr.len(),
        cu_q_repr.len() == num_seqs as nat + 1,
        cu_q_repr[0] == 0,
        forall|j: int| 0 <= j < num_seqs as int ==>
            cu_q_repr[j] < #[trigger] cu_q_repr[j + 1],
        cu_q_repr[num_seqs as int] == input_ids_repr.len() as int,
        forall|i: int| 0 <= i < config.num_layers as int ==>
            #[trigger] RT::store_kv_cache_launch_ready(
                input_ids_repr.len(), old(kv_perms).k_repr(i),
                old(kv_perms).v_repr(i), slot_repr,
            ),
        forall|i: int| 0 <= i < config.num_layers as int ==>
            #[trigger] RT::paged_attention_launch_ready(
                input_ids_repr.len(), old(kv_perms).k_repr(i),
                old(kv_perms).v_repr(i), cu_q_repr, cu_k_repr,
                max_seqlen_q as nat, max_seqlen_k as nat, bt_repr,
            ),
    ensures ({ let (logits, lp) = out;
        let pre_kv_reprs: Seq<(KVCacheLayerRepr, KVCacheLayerRepr)> =
            Seq::new(config.num_layers as nat, |i: int|
                (old(kv_perms).k_repr(i), old(kv_perms).v_repr(i)));
        let logits_repr = BD::model_forward_logits_repr(forward_config,
            RT::model_weights_repr_of(wp),
            input_ids_repr, positions_repr,
            pre_kv_reprs, slot_repr,
            cu_q_repr, cu_k_repr,
            max_seqlen_q as nat, max_seqlen_k as nat, bt_repr);
        RT::tensor_repr_2d(
            lp@,
            logits,
            Seq::new(num_seqs as nat, |i: int|
                RT::select_sample_logits_repr(logits_repr, cu_q_repr, i as nat)),
        )
        && final(kv_perms).len() == old(kv_perms).len()
        // Permission-shape preservation (surfaced 2026-08-06 for the
        // verified `engine.step`): the take/put discipline leaves the
        // permission collection in its entry shape.
        && final(kv_perms).extracted() == Set::<int>::empty()
        && RT::kv_perms_ids_distinct(*final(kv_perms))
        && (forall|j: int| 0 <= j < config.num_layers as int ==>
            #[trigger] final(kv_perms).k_id(j) == old(kv_perms).k_id(j)
            && final(kv_perms).v_id(j) == old(kv_perms).v_id(j))
        // Post-store per-layer cache reprs: the full chain of layer KV updates.
        && (forall|j: int| 0 <= j < config.num_layers as int ==>
            #[trigger] final(kv_perms).k_repr(j) == CC::model_forward_kv_reprs(forward_config,
                RT::model_weights_repr_of(wp), input_ids_repr, positions_repr,
                pre_kv_reprs, slot_repr, cu_q_repr, cu_k_repr,
                max_seqlen_q as nat, max_seqlen_k as nat, bt_repr)[j].0
            && final(kv_perms).v_repr(j) == CC::model_forward_kv_reprs(forward_config,
                RT::model_weights_repr_of(wp), input_ids_repr, positions_repr,
                pre_kv_reprs, slot_repr, cu_q_repr, cu_k_repr,
                max_seqlen_q as nat, max_seqlen_k as nat, bt_repr)[j].1)
    }),
{
    proof {
        reveal(BD::model_forward_logits_repr);
        reveal(model_execution_ready);
    }
    broadcast use {
        RT::lemma_linear_repr_shape,
        RT::lemma_rms_norm_repr_shape,
        RT::lemma_add_rms_norm_repr_shape,
        RT::lemma_embed_repr_shape,
    };

    let n: usize = layers.len();

    // Pin the pre-loop per-layer KV reprs (kv_perms is untouched here, so these
    // equal `old(kv_perms)`'s reprs — the `kv_cache_reprs` the spec chain uses).
    let ghost pre_kv_reprs: Seq<(KVCacheLayerRepr, KVCacheLayerRepr)> =
        Seq::new(config.num_layers as nat, |j: int|
            (kv_perms.k_repr(j), kv_perms.v_repr(j)));
    let ghost layers_repr: Seq<LayerWeightsRepr> = RT::model_weights_repr_of(wp).layers;
    proof {
        assert forall|j: int| 0 <= j < config.num_layers as int implies
            #[trigger] RT::store_kv_cache_launch_ready(
                input_ids_repr.len(), pre_kv_reprs[j].0,
                pre_kv_reprs[j].1, slot_repr,
            ) by {};
        assert forall|j: int| 0 <= j < config.num_layers as int implies
            #[trigger] RT::paged_attention_launch_ready(
                input_ids_repr.len(), pre_kv_reprs[j].0, pre_kv_reprs[j].1,
                cu_q_repr, cu_k_repr,
                max_seqlen_q as nat, max_seqlen_k as nat, bt_repr,
            ) by {};
    }

    // Step 1: embed.
    let scope0: Ghost<Set<RT::TensorId>> = Ghost(Set::<RT::TensorId>::empty());
    let tracked embed_p_ref = wp.tracked_borrow_embed_weight();
    let (h0, h0_p) = RT::embed(runtime, input_ids, embed_weight,
        Tracked(ip), Tracked(embed_p_ref),
        Ghost(input_ids_repr), Ghost(wp.embed_weight_repr()), scope0);
    let scope1: Ghost<Set<RT::TensorId>> = Ghost(scope0@.insert(h0.id()));

    if n == 0 {
        // Zero-layer model: rms_norm(embed) → gather last rows → lm_head.
        let tracked fn_p_ref = wp.tracked_borrow_final_norm();
        let tracked lm_p_ref = wp.tracked_borrow_lm_head();
        let (normed, normed_p) = RT::rms_norm(
            runtime, &h0, final_norm,
            Tracked(h0_p.borrow()), Tracked(fn_p_ref),
            Ghost(RT::embed_repr(input_ids_repr, wp.embed_weight_repr())),
            Ghost(wp.final_norm_repr()),
            Ghost(forward_config.rms_norm_epsilon),
            scope1);
        let scope2: Ghost<Set<RT::TensorId>> = Ghost(scope1@.insert(normed.id()));
        let ghost normed_repr: Tensor2D = RT::rms_norm_repr(
            RT::embed_repr(input_ids_repr, wp.embed_weight_repr()),
            wp.final_norm_repr(),
            forward_config.rms_norm_epsilon,
        );
        proof {
            assert(normed_repr.len() == input_ids_repr.len());
            crate::proof::tensor::geometry::lemma_cu_int_bounds(cu_q_repr, num_seqs as int);
            assert forall|k: int| 0 <= k < num_seqs as int implies
                #[trigger] cu_q_repr[k + 1] > 0
                    && cu_q_repr[k + 1] <= normed_repr.len() as int by {
                assert(0 <= cu_q_repr[k]);
                assert(cu_q_repr[k] < cu_q_repr[k + 1]);
                assert(cu_q_repr[k + 1] <= cu_q_repr[num_seqs as int]);
            }
        }
        let (selected, selected_p) = RT::select_last_hidden_rows(
            &normed,
            cu_seqlens_q,
            num_seqs,
            Tracked(normed_p.borrow()),
            Tracked(cuq_perm),
            Ghost(normed_repr),
            Ghost(cu_q_repr),
            scope2,
        );
        let ghost selected_repr: Tensor2D = Seq::new(
            num_seqs as nat,
            |i: int| normed_repr[cu_q_repr[i + 1] - 1],
        );
        let scope3: Ghost<Set<RT::TensorId>> = Ghost(scope2@.insert(selected.id()));
        let (logits, lp) = RT::linear(
            runtime, &selected, lm_head,
            Tracked(selected_p.borrow()), Tracked(lm_p_ref),
            Ghost(selected_repr),
            Ghost(wp.lm_head_repr()),
            scope3);
        proof {
            let ghost full_logits = BD::model_forward_logits_repr(forward_config,
                RT::model_weights_repr_of(wp),
                input_ids_repr,
                positions_repr,
                pre_kv_reprs,
                slot_repr,
                cu_q_repr,
                cu_k_repr,
                max_seqlen_q as nat,
                max_seqlen_k as nat,
                bt_repr,
            );
            assert(full_logits == RT::linear_repr(normed_repr, wp.lm_head_repr()));
            assert forall|i: int| 0 <= i < num_seqs as int implies
                #[trigger] RT::linear_repr(selected_repr, wp.lm_head_repr())[i]
                    == RT::select_sample_logits_repr(
                        full_logits, cu_q_repr, i as nat,
                    ) by {
                BI::linear_selected_row_invariance(
                    normed_repr,
                    wp.lm_head_repr(),
                    selected_repr,
                    i,
                    cu_q_repr[i + 1] - 1,
                );
            }
            assert(RT::linear_repr(selected_repr, wp.lm_head_repr()) =~=
                Seq::new(num_seqs as nat, |i: int|
                    RT::select_sample_logits_repr(
                        full_logits, cu_q_repr, i as nat,
                    )));
        }
        return (logits, lp);
    }

    // Step 2: First layer.
    // Pin ids on both sides of the take just as the general layer loop does.
    // These snapshots make the take/kernel/put frame independent of the
    // solver's treatment of `old(kv_perms)` across the first-layer call.
    let ghost pre_take0_k_ids: Seq<RT::TensorId> =
        Seq::new(n as nat, |j: int| kv_perms.k_id(j));
    let ghost pre_take0_v_ids: Seq<RT::TensorId> =
        Seq::new(n as nat, |j: int| kv_perms.v_id(j));
    let ghost pre_take0_k_reprs: Seq<KVCacheLayerRepr> =
        Seq::new(config.num_layers as nat, |j: int| kv_perms.k_repr(j));
    let ghost pre_take0_v_reprs: Seq<KVCacheLayerRepr> =
        Seq::new(config.num_layers as nat, |j: int| kv_perms.v_repr(j));
    proof {
        assert forall|j: int| 0 <= j < n as int implies
            #[trigger] pre_take0_k_ids[j] == old(kv_perms).k_id(j)
            && pre_take0_v_ids[j] == old(kv_perms).v_id(j) by {};
        assert forall|j: int| 0 <= j < config.num_layers as int implies
            #[trigger] pre_take0_k_reprs[j] == pre_kv_reprs[j].0
            && pre_take0_v_reprs[j] == pre_kv_reprs[j].1 by {};
    }
    let tracked wp0_ref = wp.tracked_borrow_dense_swiglu_layer(0);
    let tracked (k0_perm_t, v0_perm_t) = kv_perms.tracked_take_layer(0);
    let tracked mut k0_perm = k0_perm_t;
    let tracked mut v0_perm = v0_perm_t;

    let ghost embed_repr_g: Tensor2D =
        RT::embed_repr(input_ids_repr, wp.embed_weight_repr());
    // Use the freshly-extracted perms' own reprs rather than reaching for
    // `old(kv_perms).k_repr(0)`.  Avoids cross-call frame chains.
    let ghost k0_in_repr: KVCacheLayerRepr = k0_perm.kv_cache_repr();
    let ghost v0_in_repr: KVCacheLayerRepr = v0_perm.kv_cache_repr();
    // Post-take(0) snapshot (kv_perms unchanged across the kernel call).
    let ghost post_take0_k: Seq<KVCacheLayerRepr> =
        Seq::new(config.num_layers as nat, |j: int| kv_perms.k_repr(j));
    let ghost post_take0_v: Seq<KVCacheLayerRepr> =
        Seq::new(config.num_layers as nat, |j: int| kv_perms.v_repr(j));
    let ghost post_take0_k_ids: Seq<RT::TensorId> =
        Seq::new(n as nat, |j: int| kv_perms.k_id(j));
    let ghost post_take0_v_ids: Seq<RT::TensorId> =
        Seq::new(n as nat, |j: int| kv_perms.v_id(j));

    proof {
        // Chain of id equalities: take preserves k_id/v_id, function-entry
        // precondition ties kv_caches[0] tensor ids to old(kv_perms).k_id/v_id.
        assert(k0_perm.id() == kv_caches[0].0.id());
        assert(v0_perm.id() == kv_caches[0].1.id());
        assert forall|j: int| 0 <= j < n as int implies
            #[trigger] post_take0_k_ids[j] == pre_take0_k_ids[j]
            && post_take0_v_ids[j] == pre_take0_v_ids[j] by {
            // Mention the accessor applications explicitly to instantiate
            // tracked_take_layer's all-index id frame before unfolding the
            // snapshot sequences.
            assert(kv_perms.k_id(j) == pre_take0_k_ids[j]);
            assert(kv_perms.v_id(j) == pre_take0_v_ids[j]);
        };
        // Layer 0 is untouched at this point, so its taken repr is the pinned
        // pre-loop repr (= the spec chain's `kv_cache_reprs[0]`).
        assert(k0_in_repr == pre_kv_reprs[0].0);
        assert(v0_in_repr == pre_kv_reprs[0].1);
        assert(RT::paged_attention_launch_ready(
            input_ids_repr.len(), k0_in_repr, v0_in_repr,
            cu_q_repr, cu_k_repr,
            max_seqlen_q as nat, max_seqlen_k as nat, bt_repr,
        ));
        assert(RT::store_kv_cache_launch_ready(
            input_ids_repr.len(), k0_in_repr, v0_in_repr, slot_repr,
        ));
        assert(*wp0_ref == wp.dense_swiglu_layer(0));
        assert(layer_execution_ready(runtime, &layers[0],
            &wp.dense_swiglu_layer(0), head_dim, forward_config));
        assert(DENSE_WEIGHTS::layer_weights_valid(
            &layers[0], &wp.dense_swiglu_layer(0),
        ));
        assert(RT::dense_swiglu_runtime_qk_norm_matches(
            runtime,
            qk_norm_weights_kind(DENSE_WEIGHTS::qk_norm_weights_repr_of(
                &wp.dense_swiglu_layer(0).qk_norm,
            )),
        ));
        assert(RT::dense_swiglu_runtime_rotary_matches(
            runtime,
            crate::boundary::dense_layer_primitives::layer_attention_geometry_repr(
                DENSE_WEIGHTS::layer_weights_repr_of(
                    &wp.dense_swiglu_layer(0), head_dim,
                ),
            ),
            forward_config.rotary,
        ));
        assert(RT::dense_swiglu_runtime_rotary_matches(
            runtime,
            crate::boundary::dense_layer_primitives::layer_attention_geometry_repr(
                DENSE_WEIGHTS::layer_weights_repr_of(wp0_ref, head_dim),
            ),
            forward_config.rotary,
        ));
        // Layers j >= 1 are still untouched: snapshot equals the pre-loop reprs.
        assert forall|j: int| 1 <= j < config.num_layers as int implies
            #[trigger] post_take0_k[j] == pre_kv_reprs[j].0
            && post_take0_v[j] == pre_kv_reprs[j].1 by {
            assert(kv_perms.k_repr(j) == pre_take0_k_reprs[j]);
            assert(kv_perms.v_repr(j) == pre_take0_v_reprs[j]);
            assert(pre_take0_k_reprs[j] == pre_kv_reprs[j].0);
            assert(pre_take0_v_reprs[j] == pre_kv_reprs[j].1);
        };
    }

    let (mut cur_h, mut cur_hp, mut cur_r, mut cur_rp) = DENSE_EXEC::first_decoder_layer_forward(
        runtime, &layers[0], Tracked(wp0_ref),
        Ghost(head_dim), Ghost(forward_config),
        &h0, Tracked(h0_p.borrow()),
        positions, Tracked(pp),
        &kv_caches[0].0, Tracked(&mut k0_perm),
        &kv_caches[0].1, Tracked(&mut v0_perm),
        block_table, Tracked(bt_perm),
        slot_mapping, Tracked(sp),
        cu_seqlens_q, Tracked(cuq_perm),
        cu_seqlens_k, Tracked(cuk_perm),
        max_seqlen_q, max_seqlen_k,
        Ghost(embed_repr_g),
        Ghost(positions_repr),
        Ghost(cu_q_repr), Ghost(cu_k_repr), Ghost(bt_repr),
        Ghost(k0_in_repr), Ghost(v0_in_repr), Ghost(slot_repr),
        scope1);

    proof {
        // Kernel preserves perm.id().  tracked_take_layer's frame for
        // ids guarantees kv_perms.k_id(0) is unchanged.
        assert(k0_perm.id() == kv_perms.k_id(0));
        assert(v0_perm.id() == kv_perms.v_id(0));
        kv_perms.tracked_put_layer(0, k0_perm, v0_perm);
        // (A) at loop entry (i == 1): put_layer(0) leaves j != 0 unchanged, so
        // for j >= 1 the repr equals the post-take(0) snapshot (= pre-loop).
        assert forall|j: int| 1 <= j < n as int implies
            #[trigger] kv_perms.k_repr(j) == pre_kv_reprs[j].0
            && kv_perms.v_repr(j) == pre_kv_reprs[j].1 by {
            assert(kv_perms.k_id(j) == kv_perms.k_id(j));
        };
        assert forall|j: int| 0 <= j < n as int implies
            #[trigger] kv_perms.k_id(j) == old(kv_perms).k_id(j)
            && kv_perms.v_id(j) == old(kv_perms).v_id(j) by {
            if j == 0 {
                assert(kv_perms.k_id(j) == post_take0_k_ids[j]);
                assert(kv_perms.v_id(j) == post_take0_v_ids[j]);
            } else {
                assert(kv_perms.k_id(j) == post_take0_k_ids[j]);
                assert(kv_perms.v_id(j) == post_take0_v_ids[j]);
            }
            assert(post_take0_k_ids[j] == pre_take0_k_ids[j]);
            assert(post_take0_v_ids[j] == pre_take0_v_ids[j]);
            assert(pre_take0_k_ids[j] == old(kv_perms).k_id(j));
            assert(pre_take0_v_ids[j] == old(kv_perms).v_id(j));
        };
    }

    let mut scope: Ghost<Set<RT::TensorId>> =
        Ghost(scope1@.insert(cur_h.id()).insert(cur_r.id()));

    // The first layer's output reprs are the spec chain's starting point
    // `first.0`/`first.1`: the kernel ensured `cur_*p@.repr_2d() ==
    // `first_decoder_layer_output_repr(layer_weights_repr_of(wp0), embed, k0_in, ...)`,
    // and `wp0 == wp.dense_swiglu_layer(0)`, `k0_in == pre_kv_reprs[0].0`.
    let ghost h1_repr: Tensor2D = cur_hp@.repr_2d();
    let ghost r1_repr: Tensor2D = cur_rp@.repr_2d();
    proof {
        assert(layers_repr[0] == DENSE_WEIGHTS::layer_weights_repr_of(wp0_ref, head_dim));
        assert(h1_repr == BD::first_decoder_layer_output_repr(forward_config, layers_repr[0],
            RT::embed_repr(input_ids_repr, wp.embed_weight_repr()), positions_repr,
            pre_kv_reprs[0].0, pre_kv_reprs[0].1, slot_repr,
            cu_q_repr, cu_k_repr, max_seqlen_q as nat, max_seqlen_k as nat, bt_repr).0);
        assert(r1_repr == BD::first_decoder_layer_output_repr(forward_config, layers_repr[0],
            RT::embed_repr(input_ids_repr, wp.embed_weight_repr()), positions_repr,
            pre_kv_reprs[0].0, pre_kv_reprs[0].1, slot_repr,
            cu_q_repr, cu_k_repr, max_seqlen_q as nat, max_seqlen_k as nat, bt_repr).1);
        BD::lemma_first_decoder_layer_output_repr_shape(
            forward_config, layers_repr[0],
            RT::embed_repr(input_ids_repr, wp.embed_weight_repr()), positions_repr,
            pre_kv_reprs[0].0, pre_kv_reprs[0].1, slot_repr,
            cu_q_repr, cu_k_repr, max_seqlen_q as nat, max_seqlen_k as nat, bt_repr,
        );
    }

    // Post-first-layer cache snapshot + the full post-store cache (computed by
    // folding the remaining layers).  `full_kv` is loop-constant; the KV-fold
    // invariant telescopes it down to the final per-layer reprs at loop exit.
    let ghost kv1_seq: Seq<(KVCacheLayerRepr, KVCacheLayerRepr)> =
        Seq::new(n as nat, |j: int| (kv_perms.k_repr(j), kv_perms.v_repr(j)));
    let ghost full_kv: Seq<(KVCacheLayerRepr, KVCacheLayerRepr)> =
        CC::layer_chain_kv_reprs(forward_config, layers_repr, h1_repr, r1_repr, positions_repr,
            kv1_seq, slot_repr, cu_q_repr, cu_k_repr,
            max_seqlen_q as nat, max_seqlen_k as nat, bt_repr, 1);
    proof {
        // Characterize kv1_seq: index 0 is the first layer's kv update (from
        // first_decoder_layer_forward's kv ensures + put); j >= 1 are still pre-loop.
        assert(kv1_seq[0] == CC::first_decoder_layer_kv_update_repr(forward_config, layers_repr[0],
            RT::embed_repr(input_ids_repr, wp.embed_weight_repr()), positions_repr,
            pre_kv_reprs[0].0, pre_kv_reprs[0].1, slot_repr));
        assert forall|j: int| 1 <= j < n as int implies
            #[trigger] kv1_seq[j] == pre_kv_reprs[j] by {};
        // Instantiate the physical-id frame explicitly before the loop. The
        // attention semantic term must not control whether these quantifiers
        // happen to match during invariant initiation.
        assert forall|j: int| 0 <= j < n as int implies
            #[trigger] kv_caches[j].0.id() == kv_perms.k_id(j)
            && kv_caches[j].1.id() == kv_perms.v_id(j) by {
            assert(kv_perms.k_id(j) == old(kv_perms).k_id(j));
            assert(kv_perms.v_id(j) == old(kv_perms).v_id(j));
            assert(kv_caches[j].0.id() == old(kv_perms).k_id(j));
            assert(kv_caches[j].1.id() == old(kv_perms).v_id(j));
        };
    }

    // Step 3: Layer loop (1..num_layers).  Loop invariant maintains the
    // chain identity: `chain(_, first, _, 1) == chain(_, cur, _, i)`.
    let mut i: usize = 1;
    proof {
        // Pin the established identity frame at the loop boundary, after the
        // semantic-fold setup, rather than relying on later trigger matching.
        assert(forall|j: int| 0 <= j < n as int ==>
            #[trigger] kv_perms.k_id(j) == old(kv_perms).k_id(j)
            && kv_perms.v_id(j) == old(kv_perms).v_id(j));
    }
    while i < n
        invariant
            RT::paged_attention_numeric_domain(),
            1 <= i <= n,
            model_execution_ready(
                runtime, embed_weight, layers, final_norm, lm_head,
                wp, head_dim, forward_config,
            ),
            n == config.num_layers,
            n == layers.len(),
            n == kv_perms.len(),
            n == kv_caches.len(),
            kv_perms.extracted() == Set::<int>::empty(),
            RT::kv_perms_ids_distinct(*kv_perms),
            forall|j: int| 0 <= j < n as int ==>
                #[trigger] kv_perms.k_id(j) == old(kv_perms).k_id(j)
                && kv_perms.v_id(j) == old(kv_perms).v_id(j),
            forall|j: int| 0 <= j < n as int ==>
                #[trigger] kv_caches[j].0.id() == kv_perms.k_id(j)
                && kv_caches[j].1.id() == kv_perms.v_id(j),
            // Kernel inputs that are unchanged across loop iterations.
            RT::int_tensor_repr_1d(*pp, *positions, positions_repr),
            RT::block_table_repr(*bt_perm, *block_table, bt_repr),
            RT::int_tensor_repr_1d(*sp, *slot_mapping, slot_repr),
            RT::int_tensor_repr_1d(*cuq_perm, *cu_seqlens_q, cu_q_repr),
            RT::int_tensor_repr_1d(*cuk_perm, *cu_seqlens_k, cu_k_repr),
            cur_h.id() != cur_r.id(),
            scope@.contains(cur_h.id()),
            scope@.contains(cur_r.id()),
            input_ids_repr.len() == positions_repr.len(),
            slot_repr.len() == input_ids_repr.len(),
            // Per-perm repr bonds and shapes (the chain identity itself is the
            // last two invariant clauses below).
            RT::tensor_repr_2d(cur_hp@, cur_h, cur_hp@.repr_2d()),
            RT::tensor_repr_2d(cur_rp@, cur_r, cur_rp@.repr_2d()),
            cur_hp@.repr_2d().len() == input_ids_repr.len(),
            cur_rp@.repr_2d().len() == input_ids_repr.len(),
            // Chain-fold tracking: carries the `layer_chain_repr` fold identity
            // forward, discharging the former tail `admit` (informal-proof G1).
            layers_repr.len() == n as nat,
            pre_kv_reprs.len() == n as nat,
            layers_repr == RT::model_weights_repr_of(wp).layers,
            forall|j: int| 0 <= j < n as int ==>
                #[trigger] layers_repr[j]
                    == DENSE_WEIGHTS::layer_weights_repr_of(
                        &wp.dense_swiglu_layer(j), head_dim,
                    ),
            forall|j: int| 0 <= j < n as int ==>
                #[trigger] RT::store_kv_cache_launch_ready(
                    input_ids_repr.len(), pre_kv_reprs[j].0,
                    pre_kv_reprs[j].1, slot_repr,
                ),
            forall|j: int| 0 <= j < n as int ==>
                #[trigger] RT::paged_attention_launch_ready(
                    input_ids_repr.len(), pre_kv_reprs[j].0,
                    pre_kv_reprs[j].1, cu_q_repr, cu_k_repr,
                    max_seqlen_q as nat, max_seqlen_k as nat, bt_repr,
                ),
            // Untouched layers j >= i keep their pinned pre-loop repr, so the
            // kernel's `ki_in_repr` equals the spec chain's `pre_kv_reprs[i]`.
            KV_LOOP::suffix_matches(
                kv_perms, pre_kv_reprs, i as int, n as int,
            ),
            // The fold identity: chain from `first` equals chain from `cur` at i.
            BD::layer_chain_repr(forward_config, layers_repr, h1_repr, r1_repr, positions_repr,
                pre_kv_reprs, slot_repr, cu_q_repr, cu_k_repr,
                max_seqlen_q as nat, max_seqlen_k as nat, bt_repr, 1)
            == BD::layer_chain_repr(forward_config, layers_repr, cur_hp@.repr_2d(), cur_rp@.repr_2d(),
                positions_repr, pre_kv_reprs, slot_repr, cu_q_repr, cu_k_repr,
                max_seqlen_q as nat, max_seqlen_k as nat, bt_repr, i as nat),
            // KV-fold telescoping: the full post-store cache equals folding the
            // remaining layers from `cur` at `i` over the current per-layer reprs.
            full_kv == CC::layer_chain_kv_reprs(forward_config, layers_repr,
                cur_hp@.repr_2d(), cur_rp@.repr_2d(), positions_repr,
                Seq::new(n as nat, |j: int| (kv_perms.k_repr(j), kv_perms.v_repr(j))),
                slot_repr, cu_q_repr, cu_k_repr,
                max_seqlen_q as nat, max_seqlen_k as nat, bt_repr, i as nat),
        decreases n - i,
    {
        let next_i = i + 1;
        // KV-fold: snapshot the per-layer reprs entering this iteration (head
        // state, before the take) and pin the invariant instance over them.
        let ghost seq_old: Seq<(KVCacheLayerRepr, KVCacheLayerRepr)> =
            Seq::new(n as nat, |j: int| (kv_perms.k_repr(j), kv_perms.v_repr(j)));
        let ghost pre_take_k_ids: Seq<RT::TensorId> =
            Seq::new(n as nat, |j: int| kv_perms.k_id(j));
        let ghost pre_take_v_ids: Seq<RT::TensorId> =
            Seq::new(n as nat, |j: int| kv_perms.v_id(j));
        let ghost ch_h: Tensor2D = cur_hp@.repr_2d();
        let ghost ch_r: Tensor2D = cur_rp@.repr_2d();
        proof {
            KV_LOOP::suffix_matches_at(
                kv_perms, pre_kv_reprs, i as int, n as int, i as int,
            );
            assert(full_kv == CC::layer_chain_kv_reprs(forward_config, layers_repr, ch_h, ch_r,
                positions_repr, seq_old, slot_repr, cu_q_repr, cu_k_repr,
                max_seqlen_q as nat, max_seqlen_k as nat, bt_repr, i as nat));
            assert(seq_old[i as int] == pre_kv_reprs[i as int]);
            assert forall|j: int| 0 <= j < n as int implies
                #[trigger] pre_take_k_ids[j] == old(kv_perms).k_id(j)
                && pre_take_v_ids[j] == old(kv_perms).v_id(j) by {
                assert(pre_take_k_ids[j] == kv_perms.k_id(j));
                assert(pre_take_v_ids[j] == kv_perms.v_id(j));
                assert(kv_perms.k_id(j) == old(kv_perms).k_id(j));
                assert(kv_perms.v_id(j) == old(kv_perms).v_id(j));
            };
        }

        let tracked wpi_ref = wp.tracked_borrow_dense_swiglu_layer(i as int);
        let tracked (ki_perm_t, vi_perm_t) = kv_perms.tracked_take_layer(i as int);
        let tracked mut ki_perm = ki_perm_t;
        let tracked mut vi_perm = vi_perm_t;

        let ghost old_h_repr: Tensor2D = cur_hp@.repr_2d();
        let ghost old_r_repr: Tensor2D = cur_rp@.repr_2d();
        let ghost ki_in_repr: KVCacheLayerRepr = ki_perm.kv_cache_repr();
        let ghost vi_in_repr: KVCacheLayerRepr = vi_perm.kv_cache_repr();
        // Stable snapshot of the post-take reprs (kv_perms is unchanged across
        // the kernel call, so `put`'s `old(self)` equals this).
        let ghost post_take_k: Seq<KVCacheLayerRepr> =
            Seq::new(n as nat, |j: int| kv_perms.k_repr(j));
        let ghost post_take_v: Seq<KVCacheLayerRepr> =
            Seq::new(n as nat, |j: int| kv_perms.v_repr(j));
        let ghost post_take_k_ids: Seq<RT::TensorId> =
            Seq::new(n as nat, |j: int| kv_perms.k_id(j));
        let ghost post_take_v_ids: Seq<RT::TensorId> =
            Seq::new(n as nat, |j: int| kv_perms.v_id(j));

        proof {
            assert(ki_perm.id() == kv_caches[i as int].0.id());
            assert(vi_perm.id() == kv_caches[i as int].1.id());
            // Layer i is untouched until now, so its taken repr is the pinned
            // pre-loop repr; and the borrowed weights are layer i's.
            assert(ki_in_repr == pre_kv_reprs[i as int].0);
            assert(vi_in_repr == pre_kv_reprs[i as int].1);
            assert(RT::paged_attention_launch_ready(
                input_ids_repr.len(), ki_in_repr, vi_in_repr,
                cu_q_repr, cu_k_repr,
                max_seqlen_q as nat, max_seqlen_k as nat, bt_repr,
            ));
            assert(RT::store_kv_cache_launch_ready(
                input_ids_repr.len(), ki_in_repr, vi_in_repr, slot_repr,
            ));
            assert(*wpi_ref == wp.dense_swiglu_layer(i as int));
            assert(layer_execution_ready(runtime, &layers[i as int],
                &wp.dense_swiglu_layer(i as int), head_dim, forward_config));
            assert(DENSE_WEIGHTS::layer_weights_valid(
                &layers[i as int], &wp.dense_swiglu_layer(i as int),
            ));
            assert(RT::dense_swiglu_runtime_qk_norm_matches(
                runtime,
                qk_norm_weights_kind(DENSE_WEIGHTS::qk_norm_weights_repr_of(
                    &wp.dense_swiglu_layer(i as int).qk_norm,
                )),
            ));
            assert(RT::dense_swiglu_runtime_rotary_matches(
                runtime,
                crate::boundary::dense_layer_primitives::layer_attention_geometry_repr(
                    DENSE_WEIGHTS::layer_weights_repr_of(
                        &wp.dense_swiglu_layer(i as int), head_dim,
                    ),
                ),
                forward_config.rotary,
            ));
            assert(layers_repr[i as int]
                == DENSE_WEIGHTS::layer_weights_repr_of(
                    &wp.dense_swiglu_layer(i as int), head_dim,
                ));
            assert(layers_repr[i as int]
                == DENSE_WEIGHTS::layer_weights_repr_of(wpi_ref, head_dim));
            assert(RT::dense_swiglu_runtime_rotary_matches(
                runtime,
                crate::boundary::dense_layer_primitives::layer_attention_geometry_repr(
                    DENSE_WEIGHTS::layer_weights_repr_of(wpi_ref, head_dim),
                ),
                forward_config.rotary,
            ));
            assert forall|j: int| 0 <= j < n as int implies
                #[trigger] post_take_k_ids[j] == pre_take_k_ids[j]
                && post_take_v_ids[j] == pre_take_v_ids[j] by {};
            // After take (before the kernel and put), layers j > i are still
            // untouched — record (A) into the stable snapshot, one `take` frame
            // from the loop-head invariant; `put` will then frame it forward.
            assert forall|j: int| (i as int + 1) <= j < n as int implies
                #[trigger] post_take_k[j] == pre_kv_reprs[j].0
                && post_take_v[j] == pre_kv_reprs[j].1 by {};
        }

        let (nh, nh_p, nr, nr_p) = DENSE_EXEC::decoder_layer_forward(
            runtime, &layers[i], Tracked(wpi_ref),
            Ghost(head_dim), Ghost(forward_config),
            &cur_h, Tracked(cur_hp.borrow()),
            &cur_r, Tracked(cur_rp.borrow()),
            positions, Tracked(pp),
            &kv_caches[i].0, Tracked(&mut ki_perm),
            &kv_caches[i].1, Tracked(&mut vi_perm),
            block_table, Tracked(bt_perm),
            slot_mapping, Tracked(sp),
            cu_seqlens_q, Tracked(cuq_perm),
            cu_seqlens_k, Tracked(cuk_perm),
            max_seqlen_q, max_seqlen_k,
            Ghost(old_h_repr), Ghost(old_r_repr),
            Ghost(positions_repr),
            Ghost(cu_q_repr), Ghost(cu_k_repr), Ghost(bt_repr),
            Ghost(ki_in_repr), Ghost(vi_in_repr), Ghost(slot_repr),
            scope);

        proof {
            // Kernel preserves perm.id().
            assert(ki_perm.id() == kv_perms.k_id(i as int));
            assert(vi_perm.id() == kv_perms.v_id(i as int));
            kv_perms.tracked_put_layer(i as int, ki_perm, vi_perm);
            // Fire the layer output shape lemma so the loop invariant
            // `cur_hp@.repr_2d().len() == input_ids_repr.len()` is provable
            // after the assignment.
            BD::lemma_decoder_layer_output_repr_shape(forward_config,
                RT::model_weights_repr_of(wp).layers[i as int],
                old_h_repr, old_r_repr, positions_repr,
                ki_in_repr, vi_in_repr, slot_repr,
                cu_q_repr, cu_k_repr,
                max_seqlen_q as nat, max_seqlen_k as nat, bt_repr,
            );
            // Chain k_id/v_id frame across take/kernel/put.  Both
            // ops preserve k_id/v_id for all j (kernel doesn't touch
            // kv_perms; take/put have explicit forall frames).
            assert forall|j: int| 0 <= j < n as int implies
                #[trigger] kv_perms.k_id(j) == old(kv_perms).k_id(j)
                && kv_perms.v_id(j) == old(kv_perms).v_id(j) by {
                if j == i as int {
                    assert(kv_perms.k_id(j) == post_take_k_ids[j]);
                    assert(kv_perms.v_id(j) == post_take_v_ids[j]);
                } else {
                    assert(kv_perms.k_id(j) == post_take_k_ids[j]);
                    assert(kv_perms.v_id(j) == post_take_v_ids[j]);
                }
                assert(post_take_k_ids[j] == pre_take_k_ids[j]);
                assert(post_take_v_ids[j] == pre_take_v_ids[j]);
                assert(pre_take_k_ids[j] == old(kv_perms).k_id(j));
                assert(pre_take_v_ids[j] == old(kv_perms).v_id(j));
            };
            // (A) maintenance: put_layer(i) leaves j != i unchanged, so for
            // j >= i+1 the current repr equals the post-take snapshot (= pre-loop).
            assert(next_i as int == i as int + 1);
            assert forall|j: int| next_i as int <= j < n as int implies
                #[trigger] kv_perms.k_repr(j) == pre_kv_reprs[j].0
                && kv_perms.v_repr(j) == pre_kv_reprs[j].1 by {
                // Mention k_id(j) to fire put_layer's frame forall (its trigger),
                // which also carries k_repr(j)/v_repr(j) == post-take snapshot.
                assert(kv_perms.k_id(j) == kv_perms.k_id(j));
            };
            assert(KV_LOOP::suffix_matches(
                kv_perms, pre_kv_reprs, next_i as int, n as int));
            // (C) maintenance: the kernel output is the spec chain's next step
            // (`ki_in_repr == pre_kv_reprs[i]`, weights match), so one unfold of
            // `layer_chain_repr` at `i` advances the fold to `i+1`.
            assert(nh_p@.repr_2d() == BD::decoder_layer_output_repr(forward_config,
                layers_repr[i as int], old_h_repr, old_r_repr, positions_repr,
                pre_kv_reprs[i as int].0, pre_kv_reprs[i as int].1, slot_repr,
                cu_q_repr, cu_k_repr, max_seqlen_q as nat, max_seqlen_k as nat, bt_repr).0);
            assert(nr_p@.repr_2d() == BD::decoder_layer_output_repr(forward_config,
                layers_repr[i as int], old_h_repr, old_r_repr, positions_repr,
                pre_kv_reprs[i as int].0, pre_kv_reprs[i as int].1, slot_repr,
                cu_q_repr, cu_k_repr, max_seqlen_q as nat, max_seqlen_k as nat, bt_repr).1);
            reveal_with_fuel(BD::layer_chain_repr, 1);
            assert(BD::layer_chain_repr(forward_config, layers_repr, h1_repr, r1_repr, positions_repr,
                    pre_kv_reprs, slot_repr, cu_q_repr, cu_k_repr,
                    max_seqlen_q as nat, max_seqlen_k as nat, bt_repr, 1)
                == BD::layer_chain_repr(forward_config, layers_repr, nh_p@.repr_2d(), nr_p@.repr_2d(),
                    positions_repr, pre_kv_reprs, slot_repr, cu_q_repr, cu_k_repr,
                    max_seqlen_q as nat, max_seqlen_k as nat, bt_repr, (i + 1) as nat));
            // (C') KV-fold maintenance: layer i's post-store repr is its
            // `decoder_layer_kv_update_repr` (from decoder_layer_forward's kv
            // ensures + put); one unfold of `layer_chain_kv_reprs` at `i` advances
            // the KV fold to `i+1` (cur → next, seq → seq.update(i, upd_i)).
            let ghost upd_i = CC::decoder_layer_kv_update_repr(forward_config, layers_repr[i as int],
                old_h_repr, old_r_repr, positions_repr,
                pre_kv_reprs[i as int].0, pre_kv_reprs[i as int].1, slot_repr);
            assert(kv_perms.k_repr(i as int) == upd_i.0);
            assert(kv_perms.v_repr(i as int) == upd_i.1);
            let ghost seq_new = Seq::new(n as nat,
                |j: int| (kv_perms.k_repr(j), kv_perms.v_repr(j)));
            assert(seq_new =~= seq_old.update(i as int, upd_i)) by {
                assert forall|j: int| 0 <= j < n as int && j != i as int implies
                    #[trigger] seq_new[j] == seq_old[j] by {
                    assert(kv_perms.k_id(j) == kv_perms.k_id(j));
                }
            }
            reveal_with_fuel(CC::layer_chain_kv_reprs, 1);
            assert(full_kv == CC::layer_chain_kv_reprs(forward_config, layers_repr,
                nh_p@.repr_2d(), nr_p@.repr_2d(), positions_repr, seq_new,
                slot_repr, cu_q_repr, cu_k_repr,
                max_seqlen_q as nat, max_seqlen_k as nat, bt_repr, (i + 1) as nat));
        }

        cur_h = nh;
        cur_hp = nh_p;
        cur_r = nr;
        cur_rp = nr_p;
        scope = Ghost(scope@.insert(cur_h.id()).insert(cur_r.id()));
        i = next_i;
        proof {
            assert(i == next_i);
            assert(KV_LOOP::suffix_matches(
                kv_perms, pre_kv_reprs, i as int, n as int,
            ));
        }
    }

    // Step 4: Final add_rms_norm + last-row gather + lm_head linear.
    let tracked fn_p_ref = wp.tracked_borrow_final_norm();
    let tracked lm_p_ref = wp.tracked_borrow_lm_head();

    let ghost final_h_repr: Tensor2D = cur_hp@.repr_2d();
    let ghost final_r_repr: Tensor2D = cur_rp@.repr_2d();

    let (final_pair, final_perms) = RT::add_rms_norm(
        runtime, &cur_h, &cur_r, final_norm,
        Tracked(cur_hp.borrow()), Tracked(cur_rp.borrow()), Tracked(fn_p_ref),
        Ghost(final_h_repr), Ghost(final_r_repr),
        Ghost(wp.final_norm_repr()),
        Ghost(forward_config.rms_norm_epsilon),
        scope);
    let final_normed = final_pair.0;
    let final_normed_p = final_perms.0;
    let scope_after_norm: Ghost<Set<RT::TensorId>> =
        Ghost(scope@.insert(final_normed.id()).insert(final_pair.1.id()));

    let ghost final_normed_repr: Tensor2D = RT::add_rms_norm_repr(
        final_h_repr,
        final_r_repr,
        wp.final_norm_repr(),
        forward_config.rms_norm_epsilon,
    ).0;
    let ghost full_logits: Tensor2D =
        RT::linear_repr(final_normed_repr, wp.lm_head_repr());

    // Discharge the chain fold before the gather; its row-count fact provides
    // the exact bounds required by the last-row adapter.
    proof {
        reveal_with_fuel(BD::layer_chain_repr, 1);
        // Loop exit i == n: base case of the fold gives
        // chain(cur, n) == (cur, cur); with the (C) invariant,
        // chain(first = (h1, r1), 1) == (final_h_repr, final_r_repr).
        assert(BD::layer_chain_repr(forward_config, layers_repr, h1_repr, r1_repr, positions_repr,
            pre_kv_reprs, slot_repr, cu_q_repr, cu_k_repr,
            max_seqlen_q as nat, max_seqlen_k as nat, bt_repr, 1)
            == (final_h_repr, final_r_repr));
        // The spec forward starts its chain from `first == (h1, r1)`, so it
        // reduces to lm_head(add_rms_norm(final_h, final_r)).
        assert(BD::model_forward_logits_repr(forward_config, RT::model_weights_repr_of(wp),
            input_ids_repr, positions_repr, pre_kv_reprs, slot_repr,
            cu_q_repr, cu_k_repr, max_seqlen_q as nat, max_seqlen_k as nat, bt_repr)
            == full_logits);
        BD::lemma_model_forward_logits_repr_shape(forward_config,
            RT::model_weights_repr_of(wp),
            input_ids_repr,
            positions_repr,
            pre_kv_reprs,
            slot_repr,
            cu_q_repr,
            cu_k_repr,
            max_seqlen_q as nat,
            max_seqlen_k as nat,
            bt_repr,
        );
        assert(final_normed_repr.len() == input_ids_repr.len());
        crate::proof::tensor::geometry::lemma_cu_int_bounds(cu_q_repr, num_seqs as int);
        assert forall|k: int| 0 <= k < num_seqs as int implies
            #[trigger] cu_q_repr[k + 1] > 0
                && cu_q_repr[k + 1] <= final_normed_repr.len() as int by {
            assert(0 <= cu_q_repr[k]);
            assert(cu_q_repr[k] < cu_q_repr[k + 1]);
            assert(cu_q_repr[k + 1] <= cu_q_repr[num_seqs as int]);
        }
    }

    // Gather only the sampled hidden rows before the vocabulary-width GEMM.
    let (selected, selected_p) = RT::select_last_hidden_rows(
        &final_normed,
        cu_seqlens_q,
        num_seqs,
        Tracked(final_normed_p.borrow()),
        Tracked(cuq_perm),
        Ghost(final_normed_repr),
        Ghost(cu_q_repr),
        scope_after_norm,
    );
    let ghost selected_repr: Tensor2D = Seq::new(
        num_seqs as nat,
        |i: int| final_normed_repr[cu_q_repr[i + 1] - 1],
    );
    let scope_after_select: Ghost<Set<RT::TensorId>> =
        Ghost(scope_after_norm@.insert(selected.id()));

    let (logits, lp) = RT::linear(
        runtime, &selected,
        lm_head,
        Tracked(selected_p.borrow()),
        Tracked(lm_p_ref),
        Ghost(selected_repr),
        Ghost(wp.lm_head_repr()),
        scope_after_select,
    );

    proof {
        assert forall|i: int| 0 <= i < num_seqs as int implies
            #[trigger] RT::linear_repr(selected_repr, wp.lm_head_repr())[i]
                == RT::select_sample_logits_repr(
                    full_logits, cu_q_repr, i as nat,
                ) by {
            BI::linear_selected_row_invariance(
                final_normed_repr,
                wp.lm_head_repr(),
                selected_repr,
                i,
                cu_q_repr[i + 1] - 1,
            );
        }
        assert(RT::linear_repr(selected_repr, wp.lm_head_repr()) =~=
            Seq::new(num_seqs as nat, |i: int|
                RT::select_sample_logits_repr(
                    BD::model_forward_logits_repr(forward_config,
                        RT::model_weights_repr_of(wp),
                        input_ids_repr,
                        positions_repr,
                        pre_kv_reprs,
                        slot_repr,
                        cu_q_repr,
                        cu_k_repr,
                        max_seqlen_q as nat,
                        max_seqlen_k as nat,
                        bt_repr,
                    ),
                    cu_q_repr,
                    i as nat,
                )));

        // KV-fold exit: at i == n the fold base case gives `full_kv` == the final
        // per-layer reprs; and `full_kv` == `model_forward_kv_reprs` (unfold the
        // top-level spec: h1/r1 == first_out, kv1_seq =~= the spec's kv1).
        reveal_with_fuel(CC::layer_chain_kv_reprs, 1);
        let ghost seq_final = Seq::new(n as nat,
            |j: int| (kv_perms.k_repr(j), kv_perms.v_repr(j)));
        assert(full_kv == seq_final);  // invariant at i==n + base case unfold
        let ghost mfkv = CC::model_forward_kv_reprs(forward_config, RT::model_weights_repr_of(wp),
            input_ids_repr, positions_repr, pre_kv_reprs, slot_repr,
            cu_q_repr, cu_k_repr, max_seqlen_q as nat, max_seqlen_k as nat, bt_repr);
        assert(kv1_seq =~= pre_kv_reprs.update(0, CC::first_decoder_layer_kv_update_repr(forward_config,
            layers_repr[0], RT::embed_repr(input_ids_repr, wp.embed_weight_repr()), positions_repr,
            pre_kv_reprs[0].0, pre_kv_reprs[0].1, slot_repr)));
        assert(full_kv == mfkv);
        assert forall|j: int| 0 <= j < config.num_layers as int implies
            #[trigger] kv_perms.k_repr(j) == mfkv[j].0
            && kv_perms.v_repr(j) == mfkv[j].1 by {
            assert(seq_final[j] == (kv_perms.k_repr(j), kv_perms.v_repr(j)));
        }
    }

    (logits, lp)
}


} // verus!
