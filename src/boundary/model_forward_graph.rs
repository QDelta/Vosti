//! Architecture-neutral trusted CUDA-graph launcher for model forward.
//!
//! This module states only the external replay contract. The executable
//! canonical forward and the checked padded-to-unpadded projection remain in
//! `exec::model_families`.

use crate::model_config::ModelConfig;
use crate::exec::cache_scheduler::StepMode;
use crate::exec::model::model_forward;
use crate::exec::model_families as MODEL_FAMILIES;
use crate::{types::*, proof::model::types::*, proof::tensor::types::*};
use crate::boundary::tensor_runtime as RT;
use vstd::prelude::*;

#[cfg(not(verus_only))]
use pyo3::prelude::*;

verus! {

// Trusted CUDA launcher for the canonical checked `model_forward`.
//
// The launcher reports the canonical padded execution that CUDA replay ran;
// it does not claim that dummy rows are harmless.  The verified wrapper below
// derives the unpadded contract with `graph_cover`'s refinement theorem.
// @kernel-bridge-begin boundary::model_forward_graph::raw
#[verifier::external_body]
pub(crate) fn model_forward_cuda_graph_overlay_raw(
    overlay: &RT::CudaGraphOverlay,
    mode: &StepMode,
    allow_decode_cover: bool,
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
            MODEL_FAMILIES::cuda_graph_decode_cover_ready(
                wp,
                RT::model_weights_repr_of(wp),
                input_ids_repr, positions_repr,
                Seq::new(config.num_layers as nat, |i: int|
                    (old(kv_perms).k_repr(i), old(kv_perms).v_repr(i))),
                slot_repr, cu_q_repr, cu_k_repr,
                max_seqlen_q as nat, max_seqlen_k as nat, bt_repr,
            ),
    ensures ({ let (logits, lp) = out;
        let pre_kv_reprs: Seq<(KVCacheLayerRepr, KVCacheLayerRepr)> =
            Seq::new(config.num_layers as nat, |i: int|
                (old(kv_perms).k_repr(i), old(kv_perms).v_repr(i)));
        let pre_k_ids = Seq::new(config.num_layers as nat, |i: int|
            old(kv_perms).k_id(i));
        let pre_v_ids = Seq::new(config.num_layers as nat, |i: int|
            old(kv_perms).v_id(i));
        exists|pads: nat|
            MODEL_FAMILIES::cuda_graph_overlay_raw_result(
                pads, allow_decode_cover, RT::model_weights_repr_of(wp),
                RT::model_weights_architecture_repr_of(wp),
                input_ids_repr, positions_repr, pre_kv_reprs,
                pre_k_ids, pre_v_ids, slot_repr, cu_q_repr, cu_k_repr,
                max_seqlen_q as nat, max_seqlen_k as nat, bt_repr,
                num_seqs as nat, logits, lp@, final(kv_perms),
            )
    }),
{
    #[cfg(not(verus_only))]
    {
        // Pure prefill deliberately stays on the direct eager path.  Its
        // large kernels do not benefit enough from launch replay to justify
        // Python/GIL signature probing on every step.  This is still the
        // canonical verified forward, so it has exactly this wrapper's
        // contract without relying on CUDA-graph replay fidelity.
        if matches!(mode, StepMode::Prefill) {
            return model_forward(
                config,
                weights,
                runtime,
                Tracked(wp),
                input_ids,
                Tracked(ip),
                positions,
                Tracked(pp),
                kv_caches,
                Tracked(kv_perms),
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
                num_seqs,
                Ghost(input_ids_repr),
                Ghost(positions_repr),
                Ghost(cu_q_repr),
                Ghost(cu_k_repr),
                Ghost(bt_repr),
                Ghost(slot_repr),
            );
        }

        let mode_code: u8 = match mode {
            StepMode::Prefill => 0,
            StepMode::Decode => 1,
            StepMode::Mixed => 2,
        };
        let (num_tokens, runtime_num_seqs, block_table_width): (usize, usize, usize) =
            pyo3::Python::with_gil(|py| -> pyo3::PyResult<(usize, usize, usize)> {
                let ids = input_ids.inner.bind(py);
                let cuq = cu_seqlens_q.inner.bind(py);
                let bt = block_table.inner.bind(py);
                let tokens = ids.getattr("shape")?.get_item(0)?.extract()?;
                let cuq_len: usize = cuq.getattr("shape")?.get_item(0)?.extract()?;
                let width = bt.getattr("shape")?.get_item(1)?.extract()?;
                Ok((tokens, cuq_len.saturating_sub(1), width))
            })
            .expect("inspect model-forward CUDA-graph signature");
        assert_eq!(runtime_num_seqs, num_seqs, "CUDA-graph sequence-count drift");

        let action: u8 = pyo3::Python::with_gil(|py| -> pyo3::PyResult<u8> {
            overlay.inner.bind(py).call_method1(
                "probe",
                (
                    mode_code,
                    num_tokens,
                    num_seqs,
                    max_seqlen_q,
                    block_table_width,
                    allow_decode_cover,
                ),
            )?.extract()
        })
        .expect("probe model-forward CUDA-graph overlay");

        if action == 1 {
            let replayed = pyo3::Python::with_gil(
                |py| -> pyo3::PyResult<pyo3::Py<pyo3::PyAny>> {
                    Ok(overlay.inner.bind(py).call_method1(
                        "replay",
                        (
                            mode_code,
                            max_seqlen_q,
                            input_ids.inner.bind(py),
                            positions.inner.bind(py),
                            slot_mapping.inner.bind(py),
                            cu_seqlens_q.inner.bind(py),
                            cu_seqlens_k.inner.bind(py),
                            block_table.inner.bind(py),
                            allow_decode_cover,
                        ),
                    )?.unbind())
                },
            )
            .expect("replay captured model_forward");
            return (RT::Tensor { inner: replayed }, Tracked::assume_new());
        }

        if action == 2 {
            pyo3::Python::with_gil(|py| -> pyo3::PyResult<()> {
                overlay.inner.bind(py).call_method1(
                    "capture_begin",
                    (
                        mode_code,
                        num_tokens,
                        num_seqs,
                        max_seqlen_q,
                        block_table_width,
                        input_ids.inner.bind(py),
                    ),
                )?;
                Ok(())
            })
            .expect("begin canonical model_forward CUDA-graph capture");
        } else if action != 0 {
            panic!("invalid CUDA-graph overlay action {action}");
        }

        // The only source of a captured graph is this exact verified eager
        // function.  On an ordinary miss this is just eager execution; under
        // capture it records the same sequence and is replayed immediately
        // below to materialize this step's physical outputs and KV stores.
        let forward = model_forward(
            config,
            weights,
            runtime,
            Tracked(wp),
            input_ids,
            Tracked(ip),
            positions,
            Tracked(pp),
            kv_caches,
            Tracked(kv_perms),
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
            num_seqs,
            Ghost(input_ids_repr),
            Ghost(positions_repr),
            Ghost(cu_q_repr),
            Ghost(cu_k_repr),
            Ghost(bt_repr),
            Ghost(slot_repr),
        );

        if action == 2 {
            let replayed = pyo3::Python::with_gil(
                |py| -> pyo3::PyResult<pyo3::Py<pyo3::PyAny>> {
                    Ok(overlay.inner.bind(py).call_method1(
                        "capture_end_and_replay",
                        (
                            input_ids.inner.bind(py),
                            positions.inner.bind(py),
                            slot_mapping.inner.bind(py),
                            cu_seqlens_q.inner.bind(py),
                            cu_seqlens_k.inner.bind(py),
                            block_table.inner.bind(py),
                            forward.0.inner.bind(py),
                        ),
                    )?.unbind())
                },
            )
            .expect("finish and replay canonical model_forward capture");
            (RT::Tensor { inner: replayed }, forward.1)
        } else {
            forward
        }
    }
    #[cfg(verus_only)]
    {
        unreachable!()
    }
}
// @kernel-bridge-end boundary::model_forward_graph::raw

} // verus!
