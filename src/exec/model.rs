//! Closed executable dispatch for model-family forward implementations.
//!
//! Concrete layer loops live in composition modules; family adapters establish
//! only their boundary-specific readiness facts. Callers see one readiness
//! predicate and one architecture-dispatched result.

use crate::model_config::{ModelArchitecture, ModelConfig};
use crate::exec::dense_swiglu_model as DENSE_MODEL;
use crate::exec::model_families::{
    gemma3 as GEMMA_FAMILY, gemma4 as GEMMA4_FAMILY, llama3 as LLAMA_FAMILY, qwen3 as QWEN_FAMILY,
};
use crate::proof::model::architecture as MA;
use crate::{types::*, proof::model::types::*, proof::tensor::types::*};
use crate::boundary::tensor_runtime as RT;
use vstd::prelude::*;

verus! {

// Family-local kernel/adapter domain behind the closed executable dispatch.
// Qwen retains its established per-layer launch predicates. Gemma follows the
// exact evolving hidden/post-store cache fold, including full-cache SWA.
#[verifier::opaque]
pub open spec fn architecture_model_forward_ready(
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
) -> bool {
    match wp.architecture() {
        ModelArchitecture::Gemma4Text => GEMMA4_FAMILY::forward_ready(
            wp, input_ids_repr, positions_repr, pre_kv_reprs, slot_repr,
            cu_q_repr, cu_k_repr, max_seqlen_q, max_seqlen_k, bt_repr, num_seqs),
        ModelArchitecture::Qwen3 => crate::exec::dense_swiglu_decoder::forward_ready(
            wp, input_ids_repr, pre_kv_reprs, slot_repr,
            cu_q_repr, cu_k_repr, max_seqlen_q, max_seqlen_k, bt_repr,
        ),
        ModelArchitecture::Llama3 => crate::exec::dense_swiglu_decoder::forward_ready(
            wp, input_ids_repr, pre_kv_reprs, slot_repr,
            cu_q_repr, cu_k_repr, max_seqlen_q, max_seqlen_k, bt_repr,
        ),
        ModelArchitecture::Gemma3Text => GEMMA_FAMILY::forward_ready(
            wp, input_ids_repr, positions_repr, pre_kv_reprs, slot_repr,
            cu_q_repr, cu_k_repr, max_seqlen_q, max_seqlen_k, bt_repr,
            num_seqs,
        ),
    }
}

// Public executable model boundary. Family-specific forward loops stay behind
// this one closed dispatch point, while callers see only the architecture-
// dispatched semantic capstone. The same runtime sum is persisted by Engine,
// whose architecture-native step admits only a backend-qualified common plan
// capability for the selected family.
#[verifier::rlimit(300)]
pub fn model_forward(
    config: &ModelConfig,
    weights: &RT::ModelWeights,
    runtime: &RT::ModelRuntime,
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
) -> (out: (RT::Tensor, Tracked<RT::TensorPerm>))
    requires
        RT::paged_attention_numeric_domain(),
        RT::model_execution_valid(weights, runtime, wp),
        RT::model_weights_num_layers(weights) == config.num_layers as nat,
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
        architecture_model_forward_ready(
            wp, input_ids_repr, positions_repr,
            Seq::new(config.num_layers as nat, |i: int|
                (old(kv_perms).k_repr(i), old(kv_perms).v_repr(i))),
            slot_repr, cu_q_repr, cu_k_repr,
            max_seqlen_q as nat, max_seqlen_k as nat, bt_repr,
            num_seqs as nat,
        ),
    ensures ({ let (logits, lp) = out;
        let pre_kv_reprs: Seq<(KVCacheLayerRepr, KVCacheLayerRepr)> =
            Seq::new(config.num_layers as nat, |i: int|
                (old(kv_perms).k_repr(i), old(kv_perms).v_repr(i)));
        let architecture_repr = RT::model_weights_architecture_repr_of(wp);
        let logits_repr = MA::model_forward_logits_repr(
            RT::model_weights_repr_of(wp), architecture_repr,
            input_ids_repr, positions_repr, pre_kv_reprs, slot_repr,
            cu_q_repr, cu_k_repr,
            max_seqlen_q as nat, max_seqlen_k as nat, bt_repr,
        );
        RT::tensor_repr_2d(
            lp@,
            logits,
            Seq::new(num_seqs as nat, |i: int|
                RT::select_sample_logits_repr(logits_repr, cu_q_repr, i as nat)),
        )
    }),
    final(kv_perms).len() == old(kv_perms).len(),
    final(kv_perms).extracted() == Set::<int>::empty(),
    RT::kv_perms_ids_distinct(*final(kv_perms)),
    forall|j: int| 0 <= j < config.num_layers as int ==>
        #[trigger] final(kv_perms).k_id(j) == old(kv_perms).k_id(j),
    forall|j: int| 0 <= j < config.num_layers as int ==>
        #[trigger] final(kv_perms).v_id(j) == old(kv_perms).v_id(j),
    ({ let pre_kv_reprs: Seq<(KVCacheLayerRepr, KVCacheLayerRepr)> =
            Seq::new(config.num_layers as nat, |i: int|
                (old(kv_perms).k_repr(i), old(kv_perms).v_repr(i)));
        let architecture_repr = RT::model_weights_architecture_repr_of(wp);
        let post_kv_reprs = MA::model_forward_kv_reprs(
            RT::model_weights_repr_of(wp), architecture_repr,
            input_ids_repr, positions_repr, pre_kv_reprs, slot_repr,
            cu_q_repr, cu_k_repr,
            max_seqlen_q as nat, max_seqlen_k as nat, bt_repr,
        );
        forall|j: int| 0 <= j < config.num_layers as int ==>
            #[trigger] final(kv_perms).k_repr(j) == post_kv_reprs[j].0
    }),
    ({ let pre_kv_reprs: Seq<(KVCacheLayerRepr, KVCacheLayerRepr)> =
            Seq::new(config.num_layers as nat, |i: int|
                (old(kv_perms).k_repr(i), old(kv_perms).v_repr(i)));
        let architecture_repr = RT::model_weights_architecture_repr_of(wp);
        let post_kv_reprs = MA::model_forward_kv_reprs(
            RT::model_weights_repr_of(wp), architecture_repr,
            input_ids_repr, positions_repr, pre_kv_reprs, slot_repr,
            cu_q_repr, cu_k_repr,
            max_seqlen_q as nat, max_seqlen_k as nat, bt_repr,
        );
        forall|j: int| 0 <= j < config.num_layers as int ==>
            #[trigger] final(kv_perms).v_repr(j) == post_kv_reprs[j].1
    }),
{
    // @kernel-bridge-begin exec::model::qualified_kernel_plan_gate
    #[cfg(not(verus_only))]
    if !RT::model_runtime_admitted_by_runtime_gate(runtime, config.num_layers) {
        panic!("model forward rejected an unqualified or changed kernel plan");
    }
    // @kernel-bridge-end exec::model::qualified_kernel_plan_gate
    let ghost pre_kv_reprs: Seq<(KVCacheLayerRepr, KVCacheLayerRepr)> =
        Seq::new(config.num_layers as nat, |i: int|
            (kv_perms.k_repr(i), kv_perms.v_repr(i)));
    proof {
        assert(pre_kv_reprs =~= Seq::new(config.num_layers as nat, |i: int|
            (old(kv_perms).k_repr(i), old(kv_perms).v_repr(i))));
        // Preserve the common physical/permission binding explicitly across
        // family dispatch, before any family forward can mutate the caches.
        assert forall|i: int| 0 <= i < config.num_layers as int implies
            #[trigger] kv_caches[i].0.id() == kv_perms.k_id(i)
            && kv_caches[i].1.id() == kv_perms.v_id(i) by {
            assert(kv_caches[i].0.id() == old(kv_perms).k_id(i));
            assert(kv_caches[i].1.id() == old(kv_perms).v_id(i));
        }
        reveal(RT::model_execution_valid);
        RT::lemma_model_runtime_kernel_plan_qualification_projection(runtime);
        RT::lemma_model_runtime_deployment_config_projection(runtime);
        reveal(architecture_model_forward_ready);
        RT::lemma_model_weights_architecture_repr_valid(weights, runtime, wp);
        assert(RT::model_weights_repr_of(wp).layers.len()
            == config.num_layers as nat);
        MA::lemma_model_forward_kv_reprs_len(
            RT::model_weights_repr_of(wp),
            RT::model_weights_architecture_repr_of(wp),
            input_ids_repr, positions_repr, pre_kv_reprs, slot_repr,
            cu_q_repr, cu_k_repr,
            max_seqlen_q as nat, max_seqlen_k as nat, bt_repr,
        );
    }
    match (weights, runtime) {
        (RT::ModelWeights::Gemma4Text(gemma), RT::ModelRuntime::Gemma4Text(gemma_runtime)) => {
            proof {
                reveal(RT::model_runtime_execution_valid);
                RT::lemma_family_runtime_deployment_config_projection(gemma_runtime);
                assert(GEMMA4_FAMILY::runtime_matches_weights(gemma_runtime, gemma));
                assert(gemma.layers.len() == config.num_layers);
            }
            let out = GEMMA4_FAMILY::model_forward(
                gemma_runtime, gemma, Tracked(wp), input_ids, Tracked(ip), positions, Tracked(pp),
                kv_caches, Tracked(kv_perms), block_table, Tracked(bt_perm), slot_mapping, Tracked(sp),
                cu_seqlens_q, Tracked(cuq_perm), cu_seqlens_k, Tracked(cuk_perm),
                max_seqlen_q, max_seqlen_k, num_seqs,
                Ghost(input_ids_repr), Ghost(positions_repr), Ghost(slot_repr),
                Ghost(cu_q_repr), Ghost(cu_k_repr), Ghost(bt_repr));
            proof {
                crate::boundary::model_families::gemma4::lemma_architecture_repr(wp);
                crate::boundary::model_families::gemma4::lemma_weights_extension_repr(wp);
                MA::lemma_four_norm_forward_composition(RT::model_weights_repr_of(wp),
                    RT::model_weights_architecture_repr_of(wp),
                    input_ids_repr, positions_repr, pre_kv_reprs, slot_repr,
                    cu_q_repr, cu_k_repr, max_seqlen_q as nat, max_seqlen_k as nat, bt_repr);
                let post = MA::model_forward_kv_reprs(RT::model_weights_repr_of(wp),
                    RT::model_weights_architecture_repr_of(wp),
                    input_ids_repr, positions_repr, pre_kv_reprs, slot_repr,
                    cu_q_repr, cu_k_repr, max_seqlen_q as nat, max_seqlen_k as nat, bt_repr);
                assert forall|j: int|
                    #![trigger kv_perms.k_id(j)] #![trigger kv_perms.v_id(j)]
                    #![trigger kv_perms.k_repr(j)] #![trigger kv_perms.v_repr(j)]
                    0 <= j < config.num_layers as int implies {
                    &&& kv_perms.k_id(j) == old(kv_perms).k_id(j)
                    &&& kv_perms.v_id(j) == old(kv_perms).v_id(j)
                    &&& kv_perms.k_repr(j) == post[j].0
                    &&& kv_perms.v_repr(j) == post[j].1
                } by {
                    assert(kv_perms.k_id(j) == old(kv_perms).k_id(j));
                }
            }
            out
        },
        (
            RT::ModelWeights::Qwen3(qwen),
            RT::ModelRuntime::Qwen3(qwen_runtime),
        ) => {
            proof {
                reveal(RT::model_runtime_execution_valid);
                reveal(RT::qwen3_runtime_matches_weights);
                reveal(RT::qwen3_runtime_configuration_valid);
                reveal(RT::physical_model_deployment_config_repr);
                RT::lemma_family_runtime_deployment_config_projection(qwen_runtime);
                reveal(crate::exec::dense_swiglu_decoder::forward_ready);
                assert(wp.architecture() == ModelArchitecture::Qwen3);
                assert(RT::qwen3_model_weights_bound(qwen, wp));
                crate::boundary::model_families::qwen3::
                    lemma_execution_valid_implies_configuration_ready(
                        weights, runtime, wp,
                    );
                assert(qwen.layers.len() == config.num_layers);
                assert(wp.num_layers() == config.num_layers as nat);
                assert(RT::family_runtime_execution_valid(qwen_runtime));
                assert(RT::qwen3_runtime_matches_weights(qwen_runtime, qwen));
                assert forall|i: int| 0 <= i < config.num_layers as int implies
                    #[trigger] RT::store_kv_cache_launch_ready(
                        input_ids_repr.len(), old(kv_perms).k_repr(i),
                        old(kv_perms).v_repr(i), slot_repr,
                    ) by {
                    assert(0 <= i < wp.num_layers() as int);
                    assert(pre_kv_reprs[i].0 == old(kv_perms).k_repr(i));
                    assert(pre_kv_reprs[i].1 == old(kv_perms).v_repr(i));
                }
                assert forall|i: int| 0 <= i < config.num_layers as int implies
                    #[trigger] RT::paged_attention_launch_ready(
                        input_ids_repr.len(), old(kv_perms).k_repr(i),
                        old(kv_perms).v_repr(i), cu_q_repr, cu_k_repr,
                        max_seqlen_q as nat, max_seqlen_k as nat, bt_repr,
                    ) by {
                    assert(0 <= i < wp.num_layers() as int);
                    assert(pre_kv_reprs[i].0 == old(kv_perms).k_repr(i));
                    assert(pre_kv_reprs[i].1 == old(kv_perms).v_repr(i));
                }
            }
            #[cfg(not(verus_only))]
            if config.num_layers > 0
                && !crate::boundary::model_families::qwen3::deployment::
                    reports_backend_qualified(qwen_runtime)
            {
                panic!("Qwen3 forward requires a backend-qualified runtime");
            }
            proof {
                QWEN_FAMILY::lemma_model_execution_ready(
                    qwen_runtime, qwen, wp,
                );
            }
            let out = DENSE_MODEL::model_forward(
                qwen_runtime, config,
                &qwen.embed_weight, &qwen.layers,
                &qwen.final_norm, &qwen.lm_head,
                Tracked(wp),
                input_ids, Tracked(ip), positions, Tracked(pp),
                kv_caches, Tracked(kv_perms),
                block_table, Tracked(bt_perm), slot_mapping, Tracked(sp),
                cu_seqlens_q, Tracked(cuq_perm), cu_seqlens_k, Tracked(cuk_perm),
                max_seqlen_q, max_seqlen_k, num_seqs,
                Ghost(input_ids_repr), Ghost(positions_repr),
                Ghost(cu_q_repr), Ghost(cu_k_repr), Ghost(bt_repr), Ghost(slot_repr),
                Ghost(qwen3_config_repr(qwen.config).geometry.head_dim),
                Ghost(crate::boundary::model_families::qwen3::forward_config_repr()),
            );
            proof {
                crate::boundary::model_families::qwen3::lemma_architecture_repr(wp);
                crate::boundary::model_families::qwen3::
                    lemma_config_valid_implies_forward_config_repr(
                        crate::boundary::model_families::qwen3::
                            weights_extension_repr_of(wp),
                    );
                DENSE_MODEL::lemma_forward_result_matches_dispatch(
                    wp, RT::model_weights_repr_of(wp),
                    crate::boundary::model_families::qwen3::
                        weights_extension_repr_of(wp),
                    input_ids_repr, positions_repr, pre_kv_reprs, slot_repr,
                    cu_q_repr, cu_k_repr,
                    max_seqlen_q as nat, max_seqlen_k as nat, bt_repr,
                );
                let dispatched_logits = MA::model_forward_logits_repr(
                    RT::model_weights_repr_of(wp),
                    RT::model_weights_architecture_repr_of(wp),
                    input_ids_repr, positions_repr, pre_kv_reprs, slot_repr,
                    cu_q_repr, cu_k_repr,
                    max_seqlen_q as nat, max_seqlen_k as nat, bt_repr,
                );
                let dispatched_post = MA::model_forward_kv_reprs(
                    RT::model_weights_repr_of(wp),
                    RT::model_weights_architecture_repr_of(wp),
                    input_ids_repr, positions_repr, pre_kv_reprs, slot_repr,
                    cu_q_repr, cu_k_repr,
                    max_seqlen_q as nat, max_seqlen_k as nat, bt_repr,
                );
                assert(RT::tensor_repr_2d(
                    out.1@,
                    out.0,
                    Seq::new(num_seqs as nat, |i: int|
                        RT::select_sample_logits_repr(
                            dispatched_logits, cu_q_repr, i as nat,
                        )),
                ));
                assert(kv_perms.len() == old(kv_perms).len());
                assert(kv_perms.extracted() == Set::<int>::empty());
                assert(RT::kv_perms_ids_distinct(*kv_perms));
                assert forall|j: int|
                    #![trigger kv_perms.k_id(j)]
                    #![trigger kv_perms.v_id(j)]
                    0 <= j < config.num_layers as int implies
                    kv_perms.k_id(j) == old(kv_perms).k_id(j)
                    && kv_perms.v_id(j) == old(kv_perms).v_id(j) by {
                    assert(kv_perms.k_id(j) == old(kv_perms).k_id(j));
                    assert(kv_perms.v_id(j) == old(kv_perms).v_id(j));
                }
                assert forall|j: int|
                    #![trigger kv_perms.k_repr(j)]
                    #![trigger kv_perms.v_repr(j)]
                    0 <= j < config.num_layers as int implies
                    kv_perms.k_repr(j) == dispatched_post[j].0
                    && kv_perms.v_repr(j) == dispatched_post[j].1 by {
                    assert(kv_perms.k_id(j) == old(kv_perms).k_id(j));
                    assert(kv_perms.v_id(j) == old(kv_perms).v_id(j));
                    assert(kv_perms.k_repr(j) == dispatched_post[j].0);
                    assert(kv_perms.v_repr(j) == dispatched_post[j].1);
                }
            }
            out
        },
        (
            RT::ModelWeights::Llama3(llama),
            RT::ModelRuntime::Llama3(llama_runtime),
        ) => {
            proof {
                reveal(RT::model_runtime_execution_valid);
                reveal(RT::llama3_runtime_matches_weights);
                reveal(RT::llama3_runtime_configuration_valid);
                reveal(RT::physical_model_deployment_config_repr);
                RT::lemma_family_runtime_deployment_config_projection(llama_runtime);
                reveal(crate::exec::dense_swiglu_decoder::forward_ready);
                assert(wp.architecture() == ModelArchitecture::Llama3);
                assert(RT::llama3_model_weights_bound(llama, wp));
                crate::boundary::model_families::llama3::
                    lemma_execution_valid_implies_configuration_ready(
                        weights, runtime, wp,
                    );
                assert(llama.layers.len() == config.num_layers);
                assert(wp.num_layers() == config.num_layers as nat);
                assert(RT::family_runtime_execution_valid(llama_runtime));
                assert(RT::llama3_runtime_matches_weights(llama_runtime, llama));
                assert forall|i: int| 0 <= i < config.num_layers as int implies
                    #[trigger] RT::store_kv_cache_launch_ready(
                        input_ids_repr.len(), old(kv_perms).k_repr(i),
                        old(kv_perms).v_repr(i), slot_repr,
                    ) by {
                    assert(0 <= i < wp.num_layers() as int);
                    assert(pre_kv_reprs[i].0 == old(kv_perms).k_repr(i));
                    assert(pre_kv_reprs[i].1 == old(kv_perms).v_repr(i));
                }
                assert forall|i: int| 0 <= i < config.num_layers as int implies
                    #[trigger] RT::paged_attention_launch_ready(
                        input_ids_repr.len(), old(kv_perms).k_repr(i),
                        old(kv_perms).v_repr(i), cu_q_repr, cu_k_repr,
                        max_seqlen_q as nat, max_seqlen_k as nat, bt_repr,
                    ) by {
                    assert(0 <= i < wp.num_layers() as int);
                    assert(pre_kv_reprs[i].0 == old(kv_perms).k_repr(i));
                    assert(pre_kv_reprs[i].1 == old(kv_perms).v_repr(i));
                }
            }
            #[cfg(not(verus_only))]
            if config.num_layers > 0
                && !crate::boundary::model_families::llama3::deployment::
                    reports_backend_qualified(llama_runtime)
            {
                panic!("Llama3 forward requires a backend-qualified runtime");
            }
            proof {
                LLAMA_FAMILY::lemma_model_execution_ready(
                    llama_runtime, llama, wp,
                );
            }
            let out = DENSE_MODEL::model_forward(
                llama_runtime, config,
                &llama.embed_weight, &llama.layers,
                &llama.final_norm, &llama.lm_head,
                Tracked(wp),
                input_ids, Tracked(ip), positions, Tracked(pp),
                kv_caches, Tracked(kv_perms),
                block_table, Tracked(bt_perm), slot_mapping, Tracked(sp),
                cu_seqlens_q, Tracked(cuq_perm), cu_seqlens_k, Tracked(cuk_perm),
                max_seqlen_q, max_seqlen_k, num_seqs,
                Ghost(input_ids_repr), Ghost(positions_repr),
                Ghost(cu_q_repr), Ghost(cu_k_repr), Ghost(bt_repr), Ghost(slot_repr),
                Ghost(llama3_config_repr(llama.config).geometry.head_dim),
                Ghost(crate::boundary::model_families::llama3::forward_config_repr(
                    llama3_config_repr(llama.config),
                )),
            );
            proof {
                reveal(RT::llama3_model_weights_bound);
                reveal(crate::boundary::model_families::llama3::weights::model_weights_bound);
                crate::boundary::model_families::llama3::lemma_architecture_repr(wp);
                crate::boundary::model_families::llama3::lemma_weights_extension_repr(wp);
                let family = crate::boundary::model_families::llama3::
                    weights_extension_repr_of(wp);
                let physical = llama3_config_repr(llama.config);
                assert(family == physical);
                crate::boundary::model_families::llama3::
                    lemma_config_valid_implies_forward_config_repr(
                        family,
                    );
                assert(dense_swiglu_forward_config_repr(family)
                    == crate::boundary::model_families::llama3::
                        forward_config_repr(physical));
                DENSE_MODEL::lemma_forward_result_matches_dispatch(
                    wp, RT::model_weights_repr_of(wp),
                    family,
                    input_ids_repr, positions_repr, pre_kv_reprs, slot_repr,
                    cu_q_repr, cu_k_repr,
                    max_seqlen_q as nat, max_seqlen_k as nat, bt_repr,
                );
                let dispatched_logits = MA::model_forward_logits_repr(
                    RT::model_weights_repr_of(wp),
                    RT::model_weights_architecture_repr_of(wp),
                    input_ids_repr, positions_repr, pre_kv_reprs, slot_repr,
                    cu_q_repr, cu_k_repr,
                    max_seqlen_q as nat, max_seqlen_k as nat, bt_repr,
                );
                let dispatched_post = MA::model_forward_kv_reprs(
                    RT::model_weights_repr_of(wp),
                    RT::model_weights_architecture_repr_of(wp),
                    input_ids_repr, positions_repr, pre_kv_reprs, slot_repr,
                    cu_q_repr, cu_k_repr,
                    max_seqlen_q as nat, max_seqlen_k as nat, bt_repr,
                );
                assert(RT::tensor_repr_2d(
                    out.1@,
                    out.0,
                    Seq::new(num_seqs as nat, |i: int|
                        RT::select_sample_logits_repr(
                            dispatched_logits, cu_q_repr, i as nat,
                        )),
                ));
                assert(kv_perms.len() == old(kv_perms).len());
                assert(kv_perms.extracted() == Set::<int>::empty());
                assert(RT::kv_perms_ids_distinct(*kv_perms));
                assert forall|j: int|
                    #![trigger kv_perms.k_id(j)]
                    #![trigger kv_perms.v_id(j)]
                    0 <= j < config.num_layers as int implies
                    kv_perms.k_id(j) == old(kv_perms).k_id(j)
                    && kv_perms.v_id(j) == old(kv_perms).v_id(j) by {
                    assert(kv_perms.k_id(j) == old(kv_perms).k_id(j));
                    assert(kv_perms.v_id(j) == old(kv_perms).v_id(j));
                }
                assert forall|j: int|
                    #![trigger kv_perms.k_repr(j)]
                    #![trigger kv_perms.v_repr(j)]
                    0 <= j < config.num_layers as int implies
                    kv_perms.k_repr(j) == dispatched_post[j].0
                    && kv_perms.v_repr(j) == dispatched_post[j].1 by {
                    assert(kv_perms.k_id(j) == old(kv_perms).k_id(j));
                    assert(kv_perms.v_id(j) == old(kv_perms).v_id(j));
                    assert(kv_perms.k_repr(j) == dispatched_post[j].0);
                    assert(kv_perms.v_repr(j) == dispatched_post[j].1);
                }
            }
            out
        },
        (
            RT::ModelWeights::Gemma3Text(gemma),
            RT::ModelRuntime::Gemma3Text(gemma_runtime),
        ) => {
            proof {
                reveal(RT::model_runtime_execution_valid);
                reveal(RT::gemma3_runtime_matches_weights);
                reveal(RT::gemma3_runtime_configuration_valid);
                reveal(RT::physical_model_deployment_config_repr);
                RT::lemma_family_runtime_deployment_config_projection(gemma_runtime);
                assert(wp.architecture() == ModelArchitecture::Gemma3Text);
                assert(RT::gemma3_model_weights_bound(gemma, wp));
                assert(gemma.layers.len() == config.num_layers);
                assert(RT::family_runtime_execution_valid(gemma_runtime));
                assert(RT::gemma3_runtime_matches_weights(gemma_runtime, gemma));
                assert forall|i: int| 0 <= i < config.num_layers as int implies
                    #[trigger] kv_caches[i].0.id() == old(kv_perms).k_id(i)
                    && kv_caches[i].1.id() == old(kv_perms).v_id(i) by {}
                assert(GEMMA_FAMILY::forward_ready(
                    wp, input_ids_repr, positions_repr, pre_kv_reprs,
                    slot_repr, cu_q_repr, cu_k_repr,
                    max_seqlen_q as nat, max_seqlen_k as nat, bt_repr,
                    num_seqs as nat,
                ));
            }
            let out = GEMMA_FAMILY::model_forward(
                gemma_runtime, config, gemma, Tracked(wp),
                input_ids, Tracked(ip), positions, Tracked(pp),
                kv_caches, Tracked(kv_perms),
                block_table, Tracked(bt_perm), slot_mapping, Tracked(sp),
                cu_seqlens_q, Tracked(cuq_perm), cu_seqlens_k, Tracked(cuk_perm),
                max_seqlen_q, max_seqlen_k, num_seqs,
                Ghost(input_ids_repr), Ghost(positions_repr),
                Ghost(cu_q_repr), Ghost(cu_k_repr), Ghost(bt_repr), Ghost(slot_repr),
            );
            proof {
                GEMMA_FAMILY::lemma_forward_result_matches_dispatch(
                    wp,
                    RT::model_weights_repr_of(wp),
                    input_ids_repr, positions_repr, pre_kv_reprs, slot_repr,
                    cu_q_repr, cu_k_repr,
                    max_seqlen_q as nat, max_seqlen_k as nat, bt_repr,
                );
                let dispatched_logits = MA::model_forward_logits_repr(
                    RT::model_weights_repr_of(wp),
                    RT::model_weights_architecture_repr_of(wp),
                    input_ids_repr, positions_repr, pre_kv_reprs, slot_repr,
                    cu_q_repr, cu_k_repr,
                    max_seqlen_q as nat, max_seqlen_k as nat, bt_repr,
                );
                let dispatched_post = MA::model_forward_kv_reprs(
                    RT::model_weights_repr_of(wp),
                    RT::model_weights_architecture_repr_of(wp),
                    input_ids_repr, positions_repr, pre_kv_reprs, slot_repr,
                    cu_q_repr, cu_k_repr,
                    max_seqlen_q as nat, max_seqlen_k as nat, bt_repr,
                );
                assert(RT::tensor_repr_2d(
                    out.1@,
                    out.0,
                    Seq::new(num_seqs as nat, |i: int|
                        RT::select_sample_logits_repr(
                            dispatched_logits, cu_q_repr, i as nat,
                        )),
                ));
                assert(kv_perms.len() == old(kv_perms).len());
                assert(kv_perms.extracted() == Set::<int>::empty());
                assert(RT::kv_perms_ids_distinct(*kv_perms));
                assert forall|j: int|
                    #![trigger kv_perms.k_id(j)]
                    #![trigger kv_perms.v_id(j)]
                    0 <= j < config.num_layers as int implies
                    kv_perms.k_id(j) == old(kv_perms).k_id(j)
                    && kv_perms.v_id(j) == old(kv_perms).v_id(j) by {
                    assert(kv_perms.k_id(j) == old(kv_perms).k_id(j));
                    assert(kv_perms.v_id(j) == old(kv_perms).v_id(j));
                }
                assert forall|j: int|
                    #![trigger kv_perms.k_repr(j)]
                    #![trigger kv_perms.v_repr(j)]
                    0 <= j < config.num_layers as int implies
                    kv_perms.k_repr(j) == dispatched_post[j].0
                    && kv_perms.v_repr(j) == dispatched_post[j].1 by {
                    // Instantiate the family-local wrapper's combined
                    // id/repr postcondition through its stable-id trigger.
                    assert(kv_perms.k_id(j) == old(kv_perms).k_id(j));
                    assert(kv_perms.v_id(j) == old(kv_perms).v_id(j));
                    assert(kv_perms.k_repr(j) == dispatched_post[j].0);
                    assert(kv_perms.v_repr(j) == dispatched_post[j].1);
                }
            }
            out
        },
        _ => {
            #[cfg(verus_only)]
            {
                proof { assert(false); }
                unreached()
            }
            #[cfg(not(verus_only))]
            {
                panic!("model weights and runtime architectures differ")
            }
        },
    }
}

} // verus!
