//! Checked adapters between the architecture-neutral Engine and model-family
//! contracts.
//!
//! Scheduler and layout facts are established once by `exec::engine`. The
//! closed matches here pass those facts to a uniform operation implemented by
//! each family adapter; no family-specific premise or proof body lives here.

use crate::model_config::ModelArchitecture;
#[cfg(verus_only)]
use crate::boundary::model_families::{
    gemma3 as GEMMA_BOUNDARY, gemma4 as GEMMA4_BOUNDARY, llama3 as LLAMA_BOUNDARY, qwen3 as QWEN_BOUNDARY,
};
use crate::exec::engine::*;
#[cfg(verus_only)]
use crate::proof::model::families as FAMILIES;
#[cfg(verus_only)]
use crate::proof::model::families::dense_swiglu as DENSE;
#[cfg(verus_only)]
use crate::exec::model_families::{gemma3 as GEMMA, gemma4 as GEMMA4};
use crate::exec::request_state::SamplerState;
use crate::{types::*, proof::model::types::*, proof::tensor::types::*};
use crate::boundary::tensor_runtime as RT;
use vstd::prelude::*;

verus! {

// A qualified executable runtime always supplies a model configuration in the
// domain of its family's checked cache-refinement laws.  This is the only
// physical-runtime case split needed by architecture-neutral IBM/refinement
// code.
pub proof fn lemma_execution_cache_refinement_supported(
    weights: &RT::ModelWeights,
    runtime: &RT::ModelRuntime,
    wp: &RT::ModelWeightsPerms,
)
    requires RT::model_execution_valid(weights, runtime, wp),
    ensures
        crate::proof::model::architecture::cache_refinement_supported(
            SemanticModelRepr {
                weights: RT::model_weights_repr_of(wp),
                architecture: RT::model_weights_architecture_repr_of(wp),
            },
        ),
{
    RT::lemma_model_weights_architecture_repr_valid(weights, runtime, wp);
    match wp.architecture() {
        ModelArchitecture::Gemma4Text => {
            GEMMA4_BOUNDARY::lemma_execution_valid_implies_configuration_ready(weights, runtime, wp);
            GEMMA4_BOUNDARY::lemma_architecture_repr(wp);
            GEMMA4_BOUNDARY::lemma_weights_extension_repr(wp);
            GEMMA4_BOUNDARY::lemma_extension_attention_configs_valid(wp);
            reveal(crate::proof::model::architecture::cache_refinement_supported);
        },
        ModelArchitecture::Qwen3 => {
            QWEN_BOUNDARY::lemma_execution_valid_implies_configuration_ready(
                weights, runtime, wp,
            );
            QWEN_BOUNDARY::lemma_architecture_repr(wp);
            reveal(crate::proof::model::architecture::cache_refinement_supported);
            reveal(DENSE::cache_refinement_supported);
            reveal(DENSE::architecture_uses_config);
        },
        ModelArchitecture::Llama3 => {
            LLAMA_BOUNDARY::lemma_execution_valid_implies_configuration_ready(
                weights, runtime, wp,
            );
            LLAMA_BOUNDARY::lemma_architecture_repr(wp);
            reveal(crate::proof::model::architecture::cache_refinement_supported);
            reveal(DENSE::cache_refinement_supported);
            reveal(DENSE::architecture_uses_config);
        },
        ModelArchitecture::Gemma3Text => {
            GEMMA_BOUNDARY::lemma_execution_valid_implies_configuration_ready(
                weights, runtime, wp,
            );
            GEMMA_BOUNDARY::lemma_architecture_repr(wp);
            GEMMA_BOUNDARY::lemma_weights_extension_repr(wp);
            reveal(GEMMA_BOUNDARY::configuration_ready);
            GEMMA_BOUNDARY::lemma_extension_attention_configs_valid(wp);
            reveal(crate::proof::model::architecture::cache_refinement_supported);
        },
    }
}

pub proof fn lemma_architecture_eng_cache_shape_preserved(
    old_e: Engine,
    new_e: Engine,
    emitted: Map<RequestId, TokenId>,
    samples: Map<RequestId, (SamplerState, TokenId)>,
    reprs: StepReprs,
)
    requires
        eng_execution_perms_ok(&old_e),
        architecture_engine_step_relation(
            old_e, new_e, emitted, samples, reprs,
        ),
        eng_cache_shape_ok(&old_e),
        old_e.kv_caches_repr@.len() == reprs.wr.layers.len(),
    ensures eng_cache_shape_ok(&new_e),
{
    let architecture_repr =
        RT::model_weights_architecture_repr_of(&old_e.weights_perms@);
    RT::lemma_model_weights_architecture_repr_valid(
        &old_e.weights, &old_e.runtime, &old_e.weights_perms@,
    );
    assert(model_weights_architecture_repr_valid(
        reprs.wr, architecture_repr,
    ));
    reveal(eng_cache_shape_ok);
    reveal(crate::proof::model::cache::cache_sequence_page_shape);
    crate::proof::model::architecture::lemma_model_forward_cache_shape_preserved(
        reprs.wr, architecture_repr, reprs.input_ids, reprs.positions,
        old_e.kv_caches_repr@, reprs.slots, reprs.cu_q, reprs.cu_k,
        reprs.max_q, reprs.max_k, reprs.bt, old_e.cs.num_blocks as nat);
    assert(new_e.cs.num_blocks == old_e.cs.num_blocks);
}

pub proof fn lemma_engine_architecture_model_forward_ready(
    old_e: Engine,
    reprs: StepReprs,
    wp: &RT::ModelWeightsPerms,
    pre_kv: Seq<(KVCacheLayerRepr, KVCacheLayerRepr)>,
)
    requires
        FAMILIES::engine_forward_context(old_e, reprs, wp, pre_kv),
    ensures
        crate::exec::model::architecture_model_forward_ready(
            wp, reprs.input_ids, reprs.positions, pre_kv, reprs.slots,
            reprs.cu_q, reprs.cu_k, reprs.max_q, reprs.max_k, reprs.bt,
            reprs.scheduled.len(),
        ),
{
    reveal(FAMILIES::engine_forward_context);
    match wp.architecture() {
        ModelArchitecture::Gemma4Text => {
            GEMMA4::lemma_engine_model_forward_ready(old_e, reprs, wp, pre_kv);
        },
        ModelArchitecture::Qwen3 => {
            QWEN_BOUNDARY::lemma_execution_valid_implies_configuration_ready(
                &old_e.weights, &old_e.runtime, wp,
            );
            reveal(crate::exec::model::architecture_model_forward_ready);
            reveal(crate::exec::dense_swiglu_decoder::forward_ready);
        },
        ModelArchitecture::Llama3 => {
            LLAMA_BOUNDARY::lemma_execution_valid_implies_configuration_ready(
                &old_e.weights, &old_e.runtime, wp,
            );
            reveal(crate::exec::model::architecture_model_forward_ready);
            reveal(crate::exec::dense_swiglu_decoder::forward_ready);
        },
        ModelArchitecture::Gemma3Text => {
            GEMMA::lemma_engine_model_forward_ready(
                old_e, reprs, wp, pre_kv,
            );
        },
    }
}

pub proof fn lemma_engine_cuda_graph_decode_cover_ready(
    old_e: Engine,
    reprs: StepReprs,
)
    requires
        crate::exec::model_families::cuda_graph_overlay_supported(
            &old_e.weights_perms@,
        ),
        eng_cache_shape_ok(&old_e),
        step_reprs_wf(old_e, reprs),
        reprs_forward_layout_ok(old_e, reprs),
        reprs_other_writes_miss_plan_rows(reprs),
        reprs.scheduled.len() > 0,
        old_e.kv_caches_repr@.len() == reprs.wr.layers.len(),
        reprs.wr.layers.len() > 0,
        reprs.input_ids.len() == reprs.bt.len(),
        reprs.max_q == 1,
    ensures
        crate::exec::model_families::cuda_graph_decode_cover_ready(
            &old_e.weights_perms@,
            reprs.wr, reprs.input_ids, reprs.positions,
            old_e.kv_caches_repr@, reprs.slots,
            reprs.cu_q, reprs.cu_k, reprs.max_q, reprs.max_k, reprs.bt,
        ),
{
    crate::proof::model::graph_cover::lemma_reprs_decode_cover_ready(
        old_e, reprs,
    );
    reveal(crate::exec::model_families::cuda_graph_decode_cover_ready);
}

} // verus!
