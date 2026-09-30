//! Family implementations of the executable model-forward contract.

use crate::model_config::{ModelArchitecture, ModelConfig};
use crate::exec::cache_scheduler::StepMode;
use crate::proof::model::architecture as MA;
#[cfg(verus_only)]
use crate::proof::model::graph_cover as GRAPH_COVER;
use crate::{types::*, proof::model::types::*, proof::tensor::types::*};
use crate::boundary::tensor_runtime as RT;
use vstd::prelude::*;

pub mod gemma3;
pub mod gemma4;
pub mod llama3;
pub mod qwen3;

verus! {

// Capability-level graph overlay support. The closed match is exhaustive over
// every served family, so Engine does not name a concrete architecture.
#[verifier::opaque]
pub open spec fn cuda_graph_overlay_supported(
    wp: &RT::ModelWeightsPerms,
) -> bool {
    match wp.architecture() {
        ModelArchitecture::Gemma4Text => true,
        ModelArchitecture::Qwen3 => true,
        ModelArchitecture::Llama3 => true,
        ModelArchitecture::Gemma3Text => true,
    }
}

#[verifier::opaque]
pub open spec fn cuda_graph_decode_cover_ready(
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
) -> bool {
    GRAPH_COVER::decode_cover_ready(
        wr, input_ids, positions, pre_kv, slots,
        cu_q, cu_k, max_q, max_k, block_table,
    )
}

pub proof fn lemma_cuda_graph_decode_cover_ready_unwrap(
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
    requires
        cuda_graph_decode_cover_ready(
            wp, wr, input_ids, positions, pre_kv, slots,
            cu_q, cu_k, max_q, max_k, block_table,
        ),
    ensures
        GRAPH_COVER::decode_cover_ready(
            wr, input_ids, positions, pre_kv, slots,
            cu_q, cu_k, max_q, max_k, block_table,
        ),
{
    reveal(cuda_graph_decode_cover_ready);
}

/// Exact architecture-dispatched result returned to the Engine after replay.
#[verifier::opaque]
pub open spec fn cuda_graph_overlay_exact_result(
    wr: ModelWeightsRepr,
    architecture_repr: ModelWeightsArchitectureRepr,
    input_ids: IntTensor1D,
    positions: IntTensor1D,
    pre_kv: Seq<(KVCacheLayerRepr, KVCacheLayerRepr)>,
    pre_k_ids: Seq<RT::TensorId>,
    pre_v_ids: Seq<RT::TensorId>,
    slots: Seq<int>,
    cu_q: Seq<int>,
    cu_k: Seq<int>,
    max_q: nat,
    max_k: nat,
    bt: Seq<Seq<BlockId>>,
    num_seqs: nat,
    logits: RT::Tensor,
    lp: RT::TensorPerm,
    post_kv: &RT::KVCachePerms,
) -> bool {
    let logits_repr = MA::model_forward_logits_repr(
        wr, architecture_repr, input_ids, positions, pre_kv, slots,
        cu_q, cu_k, max_q, max_k, bt,
    );
    let post = MA::model_forward_kv_reprs(
        wr, architecture_repr, input_ids, positions, pre_kv, slots,
        cu_q, cu_k, max_q, max_k, bt,
    );
    &&& RT::tensor_repr_2d(
        lp, logits,
        Seq::new(num_seqs, |i: int|
            RT::select_sample_logits_repr(logits_repr, cu_q, i as nat)),
    )
    &&& post_kv.len() == pre_kv.len()
    &&& post_kv.extracted() == Set::<int>::empty()
    &&& RT::kv_perms_ids_distinct(*post_kv)
    &&& (forall|j: int| 0 <= j < pre_kv.len() ==>
        #[trigger] post_kv.k_id(j) == pre_k_ids[j]
        && post_kv.v_id(j) == pre_v_ids[j])
    &&& (forall|j: int| 0 <= j < pre_kv.len() ==>
        #[trigger] post_kv.k_repr(j) == post[j].0
        && post_kv.v_repr(j) == post[j].1)
}

/// Result reported by the raw launcher before optional cover refinement.
/// A positive pad count is admitted only through the architecture-neutral
/// pure-decode cover construction.
#[verifier::opaque]
pub open spec fn cuda_graph_overlay_raw_result(
    pads: nat,
    allow_decode_cover: bool,
    wr: ModelWeightsRepr,
    architecture_repr: ModelWeightsArchitectureRepr,
    input_ids: IntTensor1D,
    positions: IntTensor1D,
    pre_kv: Seq<(KVCacheLayerRepr, KVCacheLayerRepr)>,
    pre_k_ids: Seq<RT::TensorId>,
    pre_v_ids: Seq<RT::TensorId>,
    slots: Seq<int>,
    cu_q: Seq<int>,
    cu_k: Seq<int>,
    max_q: nat,
    max_k: nat,
    bt: Seq<Seq<BlockId>>,
    num_seqs: nat,
    logits: RT::Tensor,
    lp: RT::TensorPerm,
    post_kv: &RT::KVCachePerms,
) -> bool {
    &&& (!allow_decode_cover ==> pads == 0)
    &&& (pads == 0 ==> cuda_graph_overlay_exact_result(
        wr, architecture_repr, input_ids, positions, pre_kv,
        pre_k_ids, pre_v_ids, slots, cu_q, cu_k, max_q, max_k, bt,
        num_seqs, logits, lp, post_kv,
    ))
    &&& (pads > 0 ==> {
        &&& input_ids.len() == num_seqs
        &&& max_q == 1
    })
    &&& RT::tensor_repr_2d(
        lp, logits,
        GRAPH_COVER::covered_selected_logits(
            wr, architecture_repr, input_ids, positions, pre_kv, slots,
            cu_q, cu_k, max_q, max_k, bt, pads, num_seqs,
        ),
    )
    &&& post_kv.len() == pre_kv.len()
    &&& post_kv.extracted() == Set::<int>::empty()
    &&& RT::kv_perms_ids_distinct(*post_kv)
    &&& (forall|j: int| 0 <= j < pre_kv.len() ==>
        #[trigger] post_kv.k_id(j) == pre_k_ids[j]
        && post_kv.v_id(j) == pre_v_ids[j])
    &&& (forall|j: int| 0 <= j < pre_kv.len() ==>
        #[trigger] post_kv.k_repr(j)
            == GRAPH_COVER::covered_model_forward_kv_reprs(
                wr, architecture_repr, input_ids, positions, pre_kv, slots,
                cu_q, cu_k, max_q, max_k, bt, pads,
            )[j].0
        && post_kv.v_repr(j)
            == GRAPH_COVER::covered_model_forward_kv_reprs(
                wr, architecture_repr, input_ids, positions, pre_kv, slots,
                cu_q, cu_k, max_q, max_k, bt, pads,
            )[j].1)
}

// Dense-SwiGLU families share one launch-readiness contract; Gemma carries its
// different layer composition in its checked family readiness predicate.
pub proof fn lemma_cuda_graph_overlay_launch_ready(
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
        cuda_graph_overlay_supported(wp),
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
    match wp.architecture() {
        ModelArchitecture::Gemma4Text => {
            reveal(crate::exec::model::architecture_model_forward_ready);
            reveal(gemma4::forward_ready);
        },
        ModelArchitecture::Qwen3 => {
            reveal(crate::exec::model::architecture_model_forward_ready);
            reveal(crate::exec::dense_swiglu_decoder::forward_ready);
        },
        ModelArchitecture::Llama3 => {
            reveal(crate::exec::model::architecture_model_forward_ready);
            reveal(crate::exec::dense_swiglu_decoder::forward_ready);
        },
        ModelArchitecture::Gemma3Text => {
            gemma3::lemma_architecture_forward_ready_implies_launch_ready(
                wp, input_ids_repr, positions_repr, pre_kv_reprs, slot_repr,
                cu_q_repr, cu_k_repr, max_seqlen_q, max_seqlen_k, bt_repr,
                num_seqs,
            );
        },
    }
}

proof fn lemma_cuda_graph_projection_configuration_ready(
    weights: &RT::ModelWeights,
    runtime: &RT::ModelRuntime,
    wp: &RT::ModelWeightsPerms,
)
    requires
        RT::model_execution_valid(weights, runtime, wp),
        RT::model_weights_repr_of(wp).layers.len() > 0,
    ensures
        model_weights_architecture_repr_valid(
            RT::model_weights_repr_of(wp),
            RT::model_weights_architecture_repr_of(wp),
        ),
        MA::request_projection_configuration_ready(
            RT::model_weights_repr_of(wp),
            RT::model_weights_architecture_repr_of(wp),
        ),
{
    RT::lemma_model_weights_architecture_repr_valid(weights, runtime, wp);
    reveal(MA::request_projection_configuration_ready);
    match wp.architecture() {
        ModelArchitecture::Gemma4Text => {
            crate::boundary::model_families::gemma4::lemma_architecture_repr(wp);
            crate::boundary::model_families::gemma4::lemma_weights_extension_repr(wp);
            crate::boundary::model_families::gemma4::lemma_execution_valid_implies_configuration_ready(
                weights, runtime, wp);
            crate::boundary::model_families::gemma4::lemma_extension_attention_configs_valid(wp);
        },
        ModelArchitecture::Qwen3 => {
            crate::boundary::model_families::qwen3::lemma_architecture_repr(wp);
        },
        ModelArchitecture::Llama3 => {
            crate::boundary::model_families::llama3::lemma_architecture_repr(wp);
        },
        ModelArchitecture::Gemma3Text => {
            crate::boundary::model_families::gemma3::lemma_architecture_repr(wp);
            crate::boundary::model_families::gemma3::
                lemma_execution_valid_implies_configuration_ready(
                    weights, runtime, wp,
                );
            reveal(crate::boundary::model_families::gemma3::configuration_ready);
            reveal(model_weights_architecture_repr_valid);
            assert(crate::boundary::model_families::gemma3::
                weights_extension_repr_of(wp).layers.len()
                    == RT::model_weights_repr_of(wp).layers.len());
            assert(RT::model_weights_repr_of(wp).layers.len()
                == wp.num_layers());
            crate::boundary::model_families::gemma3::
                lemma_extension_attention_configs_valid(wp);
            crate::boundary::model_families::gemma3::
                lemma_weights_extension_repr(wp);
        },
    }
}

// Shared checked CUDA-graph adapter. Exact signatures are the zero-padding
// case; pure-decode signatures may replay a larger captured batch through the
// generic cover refinement proved for every enabled family.
// @kernel-bridge-begin exec::model_families::cuda_graph_overlay_contract
fn model_forward_cuda_graph_overlay_checked(
    overlay: &RT::CudaGraphOverlay,
    mode: &StepMode,
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
        RT::cuda_graph_replay_fidelity(),
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
        cu_k_repr.len() == num_seqs as nat + 1,
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
        (input_ids_repr.len() == num_seqs as nat
            && max_seqlen_q as nat == 1) ==>
            crate::exec::model_families::cuda_graph_decode_cover_ready(
                wp, RT::model_weights_repr_of(wp),
                input_ids_repr, positions_repr,
                Seq::new(config.num_layers as nat, |i: int|
                    (old(kv_perms).k_repr(i), old(kv_perms).v_repr(i))),
                slot_repr, cu_q_repr, cu_k_repr,
                max_seqlen_q as nat, max_seqlen_k as nat, bt_repr,
            ),
    ensures ({ let (logits, lp) = out;
        let pre_kv = Seq::new(config.num_layers as nat, |i: int|
            (old(kv_perms).k_repr(i), old(kv_perms).v_repr(i)));
        let pre_k_ids = Seq::new(config.num_layers as nat, |i: int|
            old(kv_perms).k_id(i));
        let pre_v_ids = Seq::new(config.num_layers as nat, |i: int|
            old(kv_perms).v_id(i));
        crate::exec::model_families::cuda_graph_overlay_exact_result(
            RT::model_weights_repr_of(wp),
            RT::model_weights_architecture_repr_of(wp),
            input_ids_repr, positions_repr, pre_kv, pre_k_ids, pre_v_ids,
            slot_repr, cu_q_repr, cu_k_repr,
            max_seqlen_q as nat, max_seqlen_k as nat, bt_repr,
            num_seqs as nat, logits, lp@, final(kv_perms),
        )
    }),
{
    proof {
        RT::lemma_model_weights_architecture_repr_valid(weights, runtime, wp);
    }
    #[cfg(not(verus_only))]
    if !RT::model_runtime_reports_backend_qualified(runtime) {
        panic!("CUDA-graph forward rejected a changed kernel plan");
    }
    let ghost pre_kv = Seq::new(config.num_layers as nat, |i: int|
        (kv_perms.k_repr(i), kv_perms.v_repr(i)));
    let ghost pre_k_ids = Seq::new(config.num_layers as nat, |i: int|
        kv_perms.k_id(i));
    let ghost pre_v_ids = Seq::new(config.num_layers as nat, |i: int|
        kv_perms.v_id(i));
    proof {
        assert(pre_kv =~= Seq::new(config.num_layers as nat, |i: int|
            (old(kv_perms).k_repr(i), old(kv_perms).v_repr(i))));
        assert(pre_k_ids =~= Seq::new(config.num_layers as nat, |i: int|
            old(kv_perms).k_id(i)));
        assert(pre_v_ids =~= Seq::new(config.num_layers as nat, |i: int|
            old(kv_perms).v_id(i)));
    }
    let (logits, lp) = crate::boundary::model_forward_graph::
        model_forward_cuda_graph_overlay_raw(
            overlay, mode, true, config, weights, runtime, Tracked(wp),
            input_ids, Tracked(ip), positions, Tracked(pp),
            kv_caches, Tracked(kv_perms), block_table, Tracked(bt_perm),
            slot_mapping, Tracked(sp), cu_seqlens_q, Tracked(cuq_perm),
            cu_seqlens_k, Tracked(cuk_perm), max_seqlen_q, max_seqlen_k,
            num_seqs, Ghost(input_ids_repr), Ghost(positions_repr),
            Ghost(cu_q_repr), Ghost(cu_k_repr), Ghost(bt_repr),
            Ghost(slot_repr),
        );
    proof {
        let pads = choose|pads: nat|
            crate::exec::model_families::cuda_graph_overlay_raw_result(
                pads, true, RT::model_weights_repr_of(wp),
                RT::model_weights_architecture_repr_of(wp),
                input_ids_repr, positions_repr, pre_kv,
                pre_k_ids, pre_v_ids, slot_repr, cu_q_repr, cu_k_repr,
                max_seqlen_q as nat, max_seqlen_k as nat, bt_repr,
                num_seqs as nat, logits, lp@, kv_perms,
            );
        assert(crate::exec::model_families::cuda_graph_overlay_raw_result(
            pads, true, RT::model_weights_repr_of(wp),
            RT::model_weights_architecture_repr_of(wp),
            input_ids_repr, positions_repr, pre_kv,
            pre_k_ids, pre_v_ids, slot_repr, cu_q_repr, cu_k_repr,
            max_seqlen_q as nat, max_seqlen_k as nat, bt_repr,
            num_seqs as nat, logits, lp@, kv_perms,
        ));
        reveal(crate::exec::model_families::cuda_graph_overlay_raw_result);
        let wr = RT::model_weights_repr_of(wp);
        let architecture_repr = RT::model_weights_architecture_repr_of(wp);
        let exact_logits = MA::model_forward_logits_repr(
            wr, architecture_repr,
            input_ids_repr, positions_repr, pre_kv, slot_repr,
            cu_q_repr, cu_k_repr,
            max_seqlen_q as nat, max_seqlen_k as nat, bt_repr,
        );
        let exact_selected = Seq::new(num_seqs as nat, |i: int|
            RT::select_sample_logits_repr(exact_logits, cu_q_repr, i as nat));
        let exact_post = MA::model_forward_kv_reprs(
            wr, architecture_repr,
            input_ids_repr, positions_repr, pre_kv, slot_repr,
            cu_q_repr, cu_k_repr,
            max_seqlen_q as nat, max_seqlen_k as nat, bt_repr,
        );
        if pads == 0 {
            GRAPH_COVER::lemma_zero_covered_model_forward(
                wr, architecture_repr,
                input_ids_repr, positions_repr, pre_kv, slot_repr,
                cu_q_repr, cu_k_repr,
                max_seqlen_q as nat, max_seqlen_k as nat, bt_repr,
                num_seqs as nat,
            );
        } else {
            assert(input_ids_repr.len() == num_seqs as nat);
            assert(max_seqlen_q as nat == 1);
            lemma_cuda_graph_decode_cover_ready_unwrap(
                wp, wr, input_ids_repr, positions_repr, pre_kv, slot_repr,
                cu_q_repr, cu_k_repr,
                max_seqlen_q as nat, max_seqlen_k as nat, bt_repr,
            );
            GRAPH_COVER::lemma_decode_cover_ready_parts(
                wr, input_ids_repr, positions_repr, pre_kv, slot_repr,
                cu_q_repr, cu_k_repr,
                max_seqlen_q as nat, max_seqlen_k as nat, bt_repr,
            );
            lemma_cuda_graph_projection_configuration_ready(
                weights, runtime, wp,
            );
            assert(bt_repr.len() == num_seqs as nat);
            GRAPH_COVER::lemma_decode_cover_refines_model_forward(
                wr, architecture_repr,
                input_ids_repr, positions_repr, pre_kv, slot_repr,
                cu_q_repr, cu_k_repr,
                max_seqlen_q as nat, max_seqlen_k as nat, bt_repr, pads,
            );
        }
        assert(GRAPH_COVER::covered_selected_logits(
            wr, architecture_repr,
            input_ids_repr, positions_repr, pre_kv, slot_repr,
            cu_q_repr, cu_k_repr,
            max_seqlen_q as nat, max_seqlen_k as nat, bt_repr,
            pads, num_seqs as nat,
        ) == exact_selected);
        assert(RT::tensor_repr_2d(lp@, logits, exact_selected));
        assert(GRAPH_COVER::covered_model_forward_kv_reprs(
            wr, architecture_repr,
            input_ids_repr, positions_repr, pre_kv, slot_repr,
            cu_q_repr, cu_k_repr,
            max_seqlen_q as nat, max_seqlen_k as nat, bt_repr, pads,
        ) == exact_post);
        MA::lemma_model_forward_kv_reprs_len(
            wr, architecture_repr,
            input_ids_repr, positions_repr, pre_kv, slot_repr,
            cu_q_repr, cu_k_repr,
            max_seqlen_q as nat, max_seqlen_k as nat, bt_repr,
        );
        assert(exact_post.len() == pre_kv.len());
        assert forall|j: int| 0 <= j < pre_kv.len() implies
            #[trigger] kv_perms.k_repr(j) == exact_post[j].0
                && kv_perms.v_repr(j) == exact_post[j].1
        by {
            assert(kv_perms.k_repr(j)
                == GRAPH_COVER::covered_model_forward_kv_reprs(
                    wr, architecture_repr,
                    input_ids_repr, positions_repr, pre_kv, slot_repr,
                    cu_q_repr, cu_k_repr,
                    max_seqlen_q as nat, max_seqlen_k as nat, bt_repr, pads,
                )[j].0);
            assert(kv_perms.v_repr(j)
                == GRAPH_COVER::covered_model_forward_kv_reprs(
                    wr, architecture_repr,
                    input_ids_repr, positions_repr, pre_kv, slot_repr,
                    cu_q_repr, cu_k_repr,
                    max_seqlen_q as nat, max_seqlen_k as nat, bt_repr, pads,
                )[j].1);
        }
        assert(kv_perms.len() == pre_kv.len());
        assert(kv_perms.extracted() == Set::<int>::empty());
        assert(RT::kv_perms_ids_distinct(*kv_perms));
        assert forall|j: int| 0 <= j < pre_kv.len() implies
            #[trigger] kv_perms.k_id(j) == pre_k_ids[j]
                && kv_perms.v_id(j) == pre_v_ids[j] by {
            assert(kv_perms.k_id(j) == pre_k_ids[j]);
            assert(kv_perms.v_id(j) == pre_v_ids[j]);
        }
        reveal(crate::exec::model_families::cuda_graph_overlay_exact_result);
        assert(crate::exec::model_families::cuda_graph_overlay_exact_result(
            RT::model_weights_repr_of(wp),
            RT::model_weights_architecture_repr_of(wp),
            input_ids_repr, positions_repr, pre_kv,
            pre_k_ids, pre_v_ids, slot_repr, cu_q_repr, cu_k_repr,
            max_seqlen_q as nat, max_seqlen_k as nat, bt_repr,
            num_seqs as nat, logits, lp@, kv_perms,
        ));
    }
    (logits, lp)
}
// @kernel-bridge-end exec::model_families::cuda_graph_overlay_contract

/// Architecture-neutral entry point for the optional CUDA-graph overlay.
#[verifier::spinoff_prover]
pub(crate) fn model_forward_cuda_graph_overlay(
    overlay: &RT::CudaGraphOverlay,
    mode: &StepMode,
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
        RT::cuda_graph_replay_fidelity(),
        RT::paged_attention_numeric_domain(),
        cuda_graph_overlay_supported(wp),
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
        cu_k_repr.len() == num_seqs as nat + 1,
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
        (input_ids_repr.len() == num_seqs as nat
            && max_seqlen_q as nat == 1) ==>
            cuda_graph_decode_cover_ready(
                wp, RT::model_weights_repr_of(wp),
                input_ids_repr, positions_repr,
                Seq::new(config.num_layers as nat, |i: int|
                    (old(kv_perms).k_repr(i), old(kv_perms).v_repr(i))),
                slot_repr, cu_q_repr, cu_k_repr,
                max_seqlen_q as nat, max_seqlen_k as nat, bt_repr,
            ),
    ensures ({ let (logits, lp) = out;
        let pre_kv = Seq::new(config.num_layers as nat, |i: int|
            (old(kv_perms).k_repr(i), old(kv_perms).v_repr(i)));
        let pre_k_ids = Seq::new(config.num_layers as nat, |i: int|
            old(kv_perms).k_id(i));
        let pre_v_ids = Seq::new(config.num_layers as nat, |i: int|
            old(kv_perms).v_id(i));
        cuda_graph_overlay_exact_result(
            RT::model_weights_repr_of(wp),
            RT::model_weights_architecture_repr_of(wp),
            input_ids_repr, positions_repr, pre_kv, pre_k_ids, pre_v_ids,
            slot_repr, cu_q_repr, cu_k_repr,
            max_seqlen_q as nat, max_seqlen_k as nat, bt_repr,
            num_seqs as nat, logits, lp@, final(kv_perms),
        )
    }),
{
    model_forward_cuda_graph_overlay_checked(
        overlay, mode, config, weights, runtime, Tracked(wp),
        input_ids, Tracked(ip), positions, Tracked(pp),
        kv_caches, Tracked(kv_perms), block_table, Tracked(bt_perm),
        slot_mapping, Tracked(sp), cu_seqlens_q, Tracked(cuq_perm),
        cu_seqlens_k, Tracked(cuk_perm), max_seqlen_q, max_seqlen_k,
        num_seqs, Ghost(input_ids_repr), Ghost(positions_repr),
        Ghost(cu_q_repr), Ghost(cu_k_repr), Ghost(bt_repr), Ghost(slot_repr),
    )
}

} // verus!
