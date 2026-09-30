//! Architecture-neutral serving refinement and trace construction.
//!
//! The public definition lives in `spec`; this module constructs
//! their independent-batch witnesses, preserves semantic/cache provenance for
//! the architecture selected by the engine, and packages actual init, admit,
//! and compute executions into finite observable traces. Family semantics enter
//! only through the closed `model_architecture` contracts.

use crate::model_config::ModelConfig;
use crate::exec::cache_scheduler;
use crate::exec::engine::*;
use crate::proof::reference::independent_batch_model::*;
use crate::proof::model::cache as MODEL_CACHE;
#[cfg(verus_only)]
use crate::proof::reference::request_machine::{request_machine_alive, token_seq_to_int};
use crate::exec::request_state::*;
use crate::{types::*, proof::model::types::*, proof::tensor::types::*};
use crate::boundary::tensor_runtime as RT;
use vstd::prelude::*;

#[cfg(verus_only)]
pub use crate::proof::cache::provenance::{provenance_prefix_candidate, registered_prefix_candidate};
#[cfg(verus_only)]
use vstd::std_specs::hash::obeys_key_model;

verus! {

// Per-rid coherence between engine.cs.live_requests and ibm.machines:
// for every shared rid, the request states are equal.  O(N) over the
// shared keyset; no pairwise enumeration.
// `machine_step_transition_full` preserves view-equality: if two pre-states are
// view-equal and both transition with the SAME sampled `(next_sampler_state,
// token)`, the post-states are view-equal.  This is exactly what keeps the engine
// and abstract sides coherent across a step — `inv(old)` gives view-equal pre
// states, and the deterministic `sample_from_repr` gives both the same sample.
pub proof fn lemma_machine_step_full_preserves_view_eq(
    pre_a: RequestState,
    post_a: RequestState,
    pre_b: RequestState,
    post_b: RequestState,
    next_sampler_state: SamplerState,
    emitted: TokenId,
)
    requires
        request_state_view_eq(pre_a, pre_b),
        crate::proof::reference::request_machine::machine_step_transition_full(pre_a, post_a, next_sampler_state, emitted),
        crate::proof::reference::request_machine::machine_step_transition_full(pre_b, post_b, next_sampler_state, emitted),
    ensures
        request_state_view_eq(post_a, post_b),
{
    lemma_request_lifecycle_view_eq_symmetric(post_b, pre_b);
    lemma_request_lifecycle_view_eq_transitive(post_a, pre_a, pre_b);
    lemma_request_lifecycle_view_eq_transitive(post_a, pre_b, post_b);
    assert(post_a.generated_tokens@ == post_b.generated_tokens@);
    assert(post_a.prompt_tokens@ == post_b.prompt_tokens@);
}

// `should_finish_after_append` depends only on view-level fields (generated-token
// count, max_tokens, eos, ignore_eos, token), so view-equal states finish alike.
// Used to show the engine and abstract sides remove the same requests
// (`shared_rid_keyset`).
pub proof fn lemma_should_finish_respects_view_eq(
    s1: RequestState,
    s2: RequestState,
    emitted: TokenId,
)
    requires request_state_view_eq(s1, s2),
    ensures should_finish_after_append(s1, emitted) == should_finish_after_append(s2, emitted),
{
    lemma_request_lifecycle_view_eq_fields(s1, s2);
    lemma_eos_tokens_from_policy_eq(s1, s2);
    lemma_same_eos_tokens_contains(eos_tokens(s1), eos_tokens(s2), emitted);
}

pub open spec fn shared_rid_coherence(
    cs: &cache_scheduler::CacheScheduler,
    ibm: &IndependentBatchModel,
) -> bool {
    forall|rid: RequestId|
        cs.live_requests@.contains_key(rid) && ibm.machines.contains_key(rid)
        ==> #[trigger] request_state_view_eq(cs.live_requests@[rid],
                ibm.machines[rid].request_state)
}

// Engine-side and IBM-side are over the same set of live requests.
pub open spec fn shared_rid_keyset(
    cs: &cache_scheduler::CacheScheduler,
    ibm: &IndependentBatchModel,
) -> bool {
    cs.live_requests@.dom() == ibm.machines.dom()
}

// The block-table row for `rid` in the engine's residency.  A request's logical
// block table is shared across layers; only the physical cache contents differ
// per layer.  This is the `bt_row` the engine's paged reads index through.
pub open spec fn residency_block_table(
    cs: &cache_scheduler::CacheScheduler,
    rid: RequestId,
) -> Seq<BlockId>
    recommends cs.request_residency@.contains_key(rid),
{
    cs.request_residency@[rid].block_ids@
}

// Precise engine↔machine KV coherence, parameterized by the engine's per-layer
// paged cache repr `eng_kv_repr` (a ghost projection of `engine.kv_caches`, to be
// stored on `Engine` when this is wired into `inv`).  For every shared `rid`,
// every layer, and every position the machine already holds (`pos < kv_tokens`),
// the engine's paged read — through `rid`'s residency block table — equals the
// machine's private *contiguous* read at the same position.
//
// This is exactly the prefix pre-store agreement (P) that the proven relocation
// lift consumes (`model_forward_relocation` / `prefix_positions_relocation_agree`),
// with A = engine paged geometry (`bt = residency_block_table`) and
// B = machine contiguous geometry (slot == pos).  Discharging refinement at
// integration time = feeding this into the relocation hypotheses.
pub open spec fn engine_kv_coherent_at(
    eng_kv_repr: Seq<(KVCacheLayerRepr, KVCacheLayerRepr)>,
    cs: &cache_scheduler::CacheScheduler,
    ibm: &IndependentBatchModel,
    num_layers: nat,
) -> bool {
    forall|rid: RequestId, layer: int, pos: nat|
        #![trigger eng_kv_repr[layer],
            crate::proof::tensor::geometry::block_table_slot(residency_block_table(cs, rid), pos)]
        cs.live_requests@.contains_key(rid)
        && ibm.machines.contains_key(rid)
        && 0 <= layer < num_layers as int
        && pos < ibm.machines[rid].kv_tokens
        ==> {
            let bt = residency_block_table(cs, rid);
            crate::proof::tensor::geometry::cache_at(eng_kv_repr[layer].0,
                crate::proof::tensor::geometry::block_table_slot(bt, pos))
                == crate::proof::tensor::geometry::cache_at(ibm.machines[rid].kv_cache_reprs[layer].0, pos)
            && crate::proof::tensor::geometry::cache_at(eng_kv_repr[layer].1,
                crate::proof::tensor::geometry::block_table_slot(bt, pos))
                == crate::proof::tensor::geometry::cache_at(ibm.machines[rid].kv_cache_reprs[layer].1, pos)
        }
}

// EngineKVCoherent — live over the engine's ghost paged-cache repr
// (`engine.kv_caches_repr`). Its preservation across `step` is proved by the
// structural and persistent semantic refinement paths.
pub open spec fn engine_kv_coherent(
    engine: &Engine,
    ibm: &IndependentBatchModel,
) -> bool {
    engine_kv_coherent_at(engine.kv_caches_repr@, &engine.cs, ibm,
        engine.model_config.num_layers as nat)
}

// Engine ownership/shape invariant (G8): the runtime per-layer caches and their
// ghost repr both have exactly `num_layers` entries.  The analog of `cs_valid`
// for the engine's KV state; without it `engine_kv_coherent` cannot index
// `kv_caches_repr[layer]` safely.
pub open spec fn eng_valid(engine: &Engine) -> bool {
    engine.kv_caches@.len() == engine.model_config.num_layers as nat
    && engine.kv_caches_repr@.len() == engine.model_config.num_layers as nat
}

// Machine-side counterpart of `eng_cache_shape_ok`: every machine's per-layer
// caches have `num_blocks` full pages. The driver establishes this alongside
// `inv` at step 0. Each step preserves it because the machine store and the
// prefix-filled base both keep the page structure.
pub open spec fn mach_cache_shape_ok(
    ibm: &IndependentBatchModel,
    num_blocks: nat,
) -> bool {
    forall|rid: RequestId, layer: int|
        #![trigger ibm.machines[rid].kv_cache_reprs[layer]]
        ibm.machines.contains_key(rid)
        && 0 <= layer < ibm.machines[rid].kv_cache_reprs.len()
        ==> {
            &&& ibm.machines[rid].kv_cache_reprs[layer].0.len() == num_blocks
            &&& ibm.machines[rid].kv_cache_reprs[layer].1.len() == num_blocks
            &&& (forall|p: int| 0 <= p < num_blocks
                ==> (#[trigger] ibm.machines[rid].kv_cache_reprs[layer].0[p]).len()
                    == crate::types::BLOCK_SIZE_SPEC as int)
            &&& (forall|p: int| 0 <= p < num_blocks
                ==> (#[trigger] ibm.machines[rid].kv_cache_reprs[layer].1[p]).len()
                    == crate::types::BLOCK_SIZE_SPEC as int)
        }
}

// Refinement invariant.  Pulls in `cs_valid` and `ibm_valid` so the two
// sides actually have well-formed local state, on top of the cross-state
// coherence predicates.
// The refinement invariant — 7 conjuncts, the relation between the concrete
// engine and the abstract per-request machines that `refinement_step` preserves:
pub open spec fn inv(engine: &Engine, ibm: &IndependentBatchModel) -> bool {
    // 1. Same model on both sides.
    engine.model_config == ibm.model_config
    // 2. The engine's scheduler state is well-formed (paged-cache invariants).
    && cache_scheduler::cs_valid(&engine.cs)
    // 3. The engine's KV cache + its ghost repr have `num_layers` entries.
    && eng_valid(engine)
    // 4. Every abstract machine is well-formed (valid state, KV shape, counters).
    && ibm_valid(*ibm)
    // 5. Both sides track exactly the same set of live request ids.
    && shared_rid_keyset(&engine.cs, ibm)
    // 6. Each shared request has the same (view-level) request state on both sides.
    && shared_rid_coherence(&engine.cs, ibm)
    // 7. Pillar-3 coherence: the engine's paged cache read through a request's
    //    block table equals that request's private contiguous cache (per layer,
    //    per cached position).  This is what lets the next step's batched forward
    //    equal the isolated per-request forward.
    && engine_kv_coherent(engine, ibm)
}

// Strengthened relation used by the observable/prefix-fidelity refinement.
// It keeps model-weight identity and IBM cache fidelity explicit beyond the
// structural `inv`; checked preservation covers every emitting mode and
// non-final KV-only chunks.

// Architecture-neutral semantic state.  Unlike `semantic_inv` above, this
// carries the complete model identity and the common cache-fidelity vocabulary;
// it therefore applies uniformly to every family discharged by
// `model_architecture::lemma_cache_refinement_laws`.
pub open spec fn architecture_semantic_inv(
    engine: &Engine,
    ibm: &IndependentBatchModel,
) -> bool {
    inv(engine, ibm)
    && ibm.wr == RT::model_weights_repr_of(&engine.weights_perms@)
    && ibm.architecture_repr
        == RT::model_weights_architecture_repr_of(&engine.weights_perms@)
    && crate::proof::model::architecture::cache_refinement_supported(
        ibm_semantic_model(*ibm),
    )
    && MODEL_CACHE::ibm_cache_fidelity(ibm)
}

// Persistent architecture-neutral semantic state: live private machines are
// canonical, and every reusable physical provenance chain is canonical for the
// same complete semantic model.
pub open spec fn architecture_persistent_semantic_inv(
    engine: &Engine,
    ibm: &IndependentBatchModel,
) -> bool {
    architecture_semantic_inv(engine, ibm)
    && crate::proof::cache::provenance::provenance_cache_fidelity(
        engine, ibm_semantic_model(*ibm),
    )
}

// Stable executable bundle shared by every architecture admitted through the
// closed runtime dispatcher.  Scheduler companions remain model-opaque; only
// the semantic/provenance conjunct above depends on the selected architecture.
pub open spec fn architecture_persistent_semantic_runtime_inv(
    engine: &Engine,
    ibm: &IndependentBatchModel,
) -> bool {
    architecture_persistent_semantic_inv(engine, ibm)
    && eng_execution_perms_ok(engine)
    && cache_scheduler::free_queue_valid(&engine.cs)
    && engine.model_config.num_layers > 0
    && engine.cs.num_blocks <= u64::MAX / crate::types::BLOCK_SIZE
    && (forall|r: RequestId|
        #[trigger] engine.cs.live_requests@.contains_key(r)
        ==> valid_request_state(engine.cs.live_requests@[r]))
    && cache_scheduler::live_request_step_ready(&engine.cs)
    && phase_aligned(engine, ibm)
    && eng_cache_shape_ok(engine)
    && cache_scheduler::residency_history_aligned(&engine.cs)
    && cache_scheduler::slot_mapping_aligned(&engine.cs)
    && cache_scheduler::tail_write_exclusive(&engine.cs)
    && cache_scheduler::residency_running_aligned(&engine.cs)
    && cache_scheduler::persistent_provenance_closed(&engine.cs)
    && mach_cache_shape_ok(ibm, engine.cs.num_blocks as nat)
}

pub proof fn lemma_architecture_semantic_inv_cache_laws(
    engine: &Engine,
    ibm: &IndependentBatchModel,
)
    requires architecture_semantic_inv(engine, ibm),
    ensures
        MODEL_CACHE::cache_refinement_laws(
            ibm_semantic_model(*ibm),
        ),
{
    reveal(architecture_semantic_inv);
    assert(ibm_valid(*ibm));
    lemma_ibm_cache_refinement_laws(*ibm);
}

// Queue/machine phase alignment: a shared request is running exactly when its
// independent machine has initialized KV.  This is a stable-boundary companion
// rather than a structural `inv` conjunct.
pub open spec fn phase_aligned(engine: &Engine, ibm: &IndependentBatchModel) -> bool {
    forall|rid: RequestId|
        engine.cs.live_requests@.contains_key(rid) && ibm.machines.contains_key(rid)
        ==> (engine.cs.running@.contains(rid)
            <==> #[trigger] ibm.machines[rid].kv_initialized)
}

// Every materialized emitting batch row is sampled from the cold full-history
// reference under the matching pre-step sampler state.
// emitting batch row is sampled from the cold full-history reference under the
// matching pre-step sampler state.  This includes requests removed by commit
// after emitting their final token; KV-only chunk rows are intentionally absent.
pub open spec fn step_samples_match_reference(
    old_ibm: IndependentBatchModel,
    samples: Map<RequestId, (SamplerState, TokenId)>,
    reprs: crate::exec::engine::StepReprs,
) -> bool {
    forall|i: int| #![trigger reprs.scheduled[i], reprs.sample_mask[i]]
        0 <= i < reprs.scheduled.len() && reprs.sample_mask[i] ==> {
            let rid = reprs.scheduled[i];
            let state = old_ibm.machines[rid].request_state;
            let tokens = crate::proof::reference::request_machine::token_seq_to_int(history(state));
            &&& old_ibm.machines.contains_key(rid)
            &&& samples.contains_key(rid)
            &&& samples[rid] == RT::sample_from_repr(
                crate::proof::model::architecture::reference_logits_last_row(
                    crate::proof::reference::independent_batch_model::ibm_semantic_model(old_ibm),
                    tokens,
                ),
                state.sampler_state,
            )
        }
}

pub open spec fn observable_step_agreement(
    old_ibm: IndependentBatchModel,
    emitted: Map<RequestId, TokenId>,
    samples: Map<RequestId, (SamplerState, TokenId)>,
    reprs: crate::exec::engine::StepReprs,
) -> bool {
    step_samples_match_reference(old_ibm, samples, reprs)
    && (forall|rid: RequestId| #[trigger] emitted.contains_key(rid) <==>
        crate::exec::engine::reprs_emits(reprs, rid))
    && forall|i: int| #![trigger reprs.scheduled[i], reprs.sample_mask[i]]
        0 <= i < reprs.scheduled.len() && reprs.sample_mask[i] ==> {
            let rid = reprs.scheduled[i];
            &&& emitted.contains_key(rid)
            &&& emitted[rid] == samples[rid].1
        }
}

// One concrete engine observation together with the abstract states immediately
// before and after it.  All fields are ghost values, but `emitted`, `samples`,
// and `reprs` are taken directly from the executable step's returned views.
pub ghost struct ObservableTraceStep {
    pub pre_engine: Engine,
    pub post_engine: Engine,
    pub pre_ibm: IndependentBatchModel,
    pub post_ibm: IndependentBatchModel,
    pub emitted: Map<RequestId, TokenId>,
    pub samples: Map<RequestId, (SamplerState, TokenId)>,
    pub reprs: crate::exec::engine::StepReprs,
}

pub open spec fn observable_trace_step_agreement(step: ObservableTraceStep) -> bool {
    architecture_persistent_semantic_runtime_inv(
        &step.pre_engine, &step.pre_ibm,
    )
    && architecture_persistent_semantic_runtime_inv(
        &step.post_engine, &step.post_ibm,
    )
    && crate::exec::engine::architecture_engine_step_relation(
        step.pre_engine,
        step.post_engine,
        step.emitted,
        step.samples,
        step.reprs,
    )
    && ibm_step(
        step.pre_ibm,
        step.post_ibm,
        step.emitted.dom(),
        step.samples,
    )
    && observable_step_agreement(
        step.pre_ibm, step.emitted, step.samples, step.reprs,
    )
}

// Package the five already-proved step facts without unfolding their large
// bodies in trace-construction proofs.
#[verifier::spinoff_prover]
pub proof fn lemma_observable_trace_step_agreement_intro(
    step: ObservableTraceStep,
)
    requires
        architecture_persistent_semantic_runtime_inv(
            &step.pre_engine, &step.pre_ibm,
        ),
        architecture_persistent_semantic_runtime_inv(
            &step.post_engine, &step.post_ibm,
        ),
        crate::exec::engine::architecture_engine_step_relation(
            step.pre_engine,
            step.post_engine,
            step.emitted,
            step.samples,
            step.reprs,
        ),
        ibm_step(
            step.pre_ibm,
            step.post_ibm,
            step.emitted.dom(),
            step.samples,
        ),
        observable_step_agreement(
            step.pre_ibm, step.emitted, step.samples, step.reprs,
        ),
    ensures
        observable_trace_step_agreement(step),
{
}

pub open spec fn observable_trace_chain(steps: Seq<ObservableTraceStep>) -> bool {
    (forall|i: int| 0 <= i < steps.len() ==>
        #[trigger] observable_trace_step_agreement(steps[i]))
    && (forall|i: int| 0 <= i && i + 1 < steps.len() ==>
        #[trigger] steps[i].post_engine == steps[i + 1].pre_engine
        && steps[i].post_ibm == steps[i + 1].pre_ibm)
}

// A finite execution trace with explicit endpoints.  This is a safety and
// observational-equivalence object: it does not assert that `step_count`
// suffices to finish every request or that scheduling is fair.
pub ghost struct ObservableServingTrace {
    pub initial_engine: Engine,
    pub final_engine: Engine,
    pub initial_ibm: IndependentBatchModel,
    pub final_ibm: IndependentBatchModel,
    pub steps: Seq<ObservableTraceStep>,
}

// Dynamic serving extends the compute trace with stable-boundary admission
// events.  The event carries both concrete and abstract endpoints, so one chain
// can interleave arbitrary future arrivals with ordinary mixed-batch steps.
pub ghost struct AdmissionTraceStep {
    pub pre_engine: Engine,
    pub post_engine: Engine,
    pub pre_ibm: IndependentBatchModel,
    pub post_ibm: IndependentBatchModel,
    pub request: RequestState,
}

pub ghost enum DynamicServingEvent {
    Admission(AdmissionTraceStep),
    Compute(ObservableTraceStep),
}

pub open spec fn admission_trace_step_agreement(step: AdmissionTraceStep) -> bool {
    let rid = step.request.request_id;
    architecture_persistent_semantic_runtime_inv(
        &step.pre_engine, &step.pre_ibm,
    )
    && architecture_persistent_semantic_runtime_inv(
        &step.post_engine, &step.post_ibm,
    )
    && crate::exec::engine::engine_admission_relation(
        &step.pre_engine, &step.post_engine, step.request,
    )
    && ibm_admission_relation(
        step.pre_ibm, step.post_ibm, &step.post_engine, rid,
    )
    && !step.pre_engine.cs.accepted_requests@.contains_key(rid)
    && step.request.generated_tokens@.len() == 0
    && can_step(step.request)
    && request_history_capacity_safe(step.request)
}

pub open spec fn dynamic_event_agreement(event: DynamicServingEvent) -> bool {
    match event {
        DynamicServingEvent::Admission(step) => admission_trace_step_agreement(step),
        DynamicServingEvent::Compute(step) => observable_trace_step_agreement(step),
    }
}

// Model-family-neutral observable contract over the closed architecture
// forward selected by the engine.
pub open spec fn architecture_step_logits_match_reference(
    old_e: Engine,
    reprs: crate::exec::engine::StepReprs,
) -> bool {
    forall|i: int| #![trigger reprs.scheduled[i], reprs.sample_mask[i]]
        0 <= i < reprs.scheduled.len() && reprs.sample_mask[i] ==> {
            let rid = reprs.scheduled[i];
            let tokens = token_seq_to_int(
                history(old_e.cs.live_requests@[rid]),
            );
            &&& old_e.cs.live_requests@.contains_key(rid)
            &&& RT::select_sample_logits_repr(
                crate::exec::engine::architecture_step_logits_repr(old_e, reprs),
                reprs.cu_q,
                i as nat,
            ) == crate::proof::model::architecture::reference_logits_last_row(
                crate::exec::engine::step_semantic_model(old_e, reprs), tokens,
            )
        }
}

pub open spec fn dynamic_event_pre_engine(event: DynamicServingEvent) -> Engine {
    match event {
        DynamicServingEvent::Admission(step) => step.pre_engine,
        DynamicServingEvent::Compute(step) => step.pre_engine,
    }
}

pub open spec fn dynamic_event_post_engine(event: DynamicServingEvent) -> Engine {
    match event {
        DynamicServingEvent::Admission(step) => step.post_engine,
        DynamicServingEvent::Compute(step) => step.post_engine,
    }
}

pub open spec fn dynamic_event_pre_ibm(
    event: DynamicServingEvent,
) -> IndependentBatchModel {
    match event {
        DynamicServingEvent::Admission(step) => step.pre_ibm,
        DynamicServingEvent::Compute(step) => step.pre_ibm,
    }
}

pub open spec fn dynamic_event_post_ibm(
    event: DynamicServingEvent,
) -> IndependentBatchModel {
    match event {
        DynamicServingEvent::Admission(step) => step.post_ibm,
        DynamicServingEvent::Compute(step) => step.post_ibm,
    }
}

pub open spec fn dynamic_event_admits(
    event: DynamicServingEvent,
    rid: RequestId,
) -> bool {
    match event {
        DynamicServingEvent::Admission(step) => step.request.request_id == rid,
        DynamicServingEvent::Compute(_) => false,
    }
}

pub open spec fn dynamic_trace_chain(events: Seq<DynamicServingEvent>) -> bool {
    (forall|i: int| 0 <= i < events.len() ==>
        #[trigger] dynamic_event_agreement(events[i]))
    && (forall|i: int| 0 <= i && i + 1 < events.len() ==>
        #[trigger] dynamic_event_post_engine(events[i])
            == dynamic_event_pre_engine(events[i + 1])
        && dynamic_event_post_ibm(events[i])
            == dynamic_event_pre_ibm(events[i + 1]))
}

pub open spec fn dynamic_events_do_not_admit(
    events: Seq<DynamicServingEvent>,
    rid: RequestId,
) -> bool {
    forall|i: int| 0 <= i < events.len() ==>
        !#[trigger] dynamic_event_admits(events[i], rid)
}

pub open spec fn dynamic_trace_samples_for(
    events: Seq<DynamicServingEvent>,
    rid: RequestId,
) -> Seq<(SamplerState, TokenId)>
    decreases events.len(),
{
    if events.len() == 0 {
        Seq::empty()
    } else {
        let first = events[0];
        let tail = events.subrange(1, events.len() as int);
        match first {
            DynamicServingEvent::Admission(_) => dynamic_trace_samples_for(tail, rid),
            DynamicServingEvent::Compute(step) => {
                if step.emitted.contains_key(rid) {
                    seq![step.samples[rid]] + dynamic_trace_samples_for(tail, rid)
                } else {
                    dynamic_trace_samples_for(tail, rid)
                }
            },
        }
    }
}

// A per-request dynamic certificate begins at that request's own admission and
// permits arbitrary other arrivals afterward.  Same-id readmission is derived
// from the engine-resident accepted-request ledger: admission inserts the id,
// compute frames the ledger, and every later admission requires absence.
pub ghost struct DynamicRequestTrace {
    pub arrival: AdmissionTraceStep,
    pub final_engine: Engine,
    pub final_ibm: IndependentBatchModel,
    pub events: Seq<DynamicServingEvent>,
}

pub open spec fn dynamic_request_trace(trace: DynamicRequestTrace) -> bool {
    admission_trace_step_agreement(trace.arrival)
    && dynamic_trace_chain(trace.events)
    && (trace.events.len() == 0 ==>
        trace.final_engine == trace.arrival.post_engine
        && trace.final_ibm == trace.arrival.post_ibm)
    && (trace.events.len() > 0 ==>
        dynamic_event_pre_engine(trace.events[0]) == trace.arrival.post_engine
        && dynamic_event_pre_ibm(trace.events[0]) == trace.arrival.post_ibm
        && dynamic_event_post_engine(trace.events[trace.events.len() - 1])
            == trace.final_engine
        && dynamic_event_post_ibm(trace.events[trace.events.len() - 1])
            == trace.final_ibm)
}

// Equality needed for the arriving request's output sequence. Runtime ids and
// lifecycle controls may differ: the theorem compares only a prefix for which
// both executions actually emit samples.
pub open spec fn arrival_request_view_eq(a: RequestState, b: RequestState) -> bool {
    request_sampling_view_eq(a, b)
}

pub open spec fn observable_serving_trace(
    trace: ObservableServingTrace,
    step_count: nat,
) -> bool {
    trace.steps.len() == step_count
    && observable_trace_chain(trace.steps)
    && (trace.steps.len() == 0 ==>
        trace.initial_engine == trace.final_engine
        && trace.initial_ibm == trace.final_ibm)
    && (trace.steps.len() > 0 ==>
        trace.steps[0].pre_engine == trace.initial_engine
        && trace.steps[0].pre_ibm == trace.initial_ibm
        && trace.steps[trace.steps.len() - 1].post_engine == trace.final_engine
        && trace.steps[trace.steps.len() - 1].post_ibm == trace.final_ibm)
}

pub open spec fn prepend_observable_trace(
    first: ObservableTraceStep,
    tail: ObservableServingTrace,
) -> ObservableServingTrace {
    ObservableServingTrace {
        initial_engine: first.pre_engine,
        final_engine: tail.final_engine,
        initial_ibm: first.pre_ibm,
        final_ibm: tail.final_ibm,
        steps: seq![first] + tail.steps,
    }
}

// Sequence-only trace composition is independent of the executable model and
// scheduler proof. Keeping it in a small spinoff query prevents new model-
// family declarations from perturbing the recursive execution theorem.
#[verifier::spinoff_prover]
#[verifier::rlimit(50)]
pub proof fn lemma_prepend_observable_trace(
    first: ObservableTraceStep,
    tail: ObservableServingTrace,
    tail_len: nat,
)
    requires
        observable_trace_step_agreement(first),
        observable_serving_trace(tail, tail_len),
        tail.initial_engine == first.post_engine,
        tail.initial_ibm == first.post_ibm,
    ensures
        observable_serving_trace(
            prepend_observable_trace(first, tail), tail_len + 1,
        ),
{
    let steps = seq![first] + tail.steps;
    assert(tail.steps.len() == tail_len);
    assert(steps.len() == tail_len + 1);
    assert forall|i: int| 0 <= i < steps.len() implies
        #[trigger] observable_trace_step_agreement(steps[i])
    by {
        if i == 0 {
            assert(steps[0] == first);
        } else {
            assert(0 <= i - 1 < tail.steps.len());
            assert(steps[i] == tail.steps[i - 1]);
        }
    }
    assert forall|i: int| 0 <= i && i + 1 < steps.len() implies
        #[trigger] steps[i].post_engine == steps[i + 1].pre_engine
            && steps[i].post_ibm == steps[i + 1].pre_ibm
    by {
        if i == 0 {
            assert(steps[0] == first);
            assert(tail.steps.len() > 0);
            assert(steps[1] == tail.steps[0]);
            assert(tail.steps[0].pre_engine == tail.initial_engine);
            assert(tail.steps[0].pre_ibm == tail.initial_ibm);
        } else {
            assert(0 <= i - 1);
            assert(i < tail.steps.len());
            assert(steps[i] == tail.steps[i - 1]);
            assert(steps[i + 1] == tail.steps[i]);
            assert(tail.steps[i - 1].post_engine
                == tail.steps[i].pre_engine);
            assert(tail.steps[i - 1].post_ibm
                == tail.steps[i].pre_ibm);
        }
    }
    assert(observable_trace_chain(steps));
    if tail.steps.len() == 0 {
        assert(tail.initial_ibm == tail.final_ibm);
        assert(steps.len() == 1);
        assert(steps[steps.len() - 1] == first);
        assert(first.post_engine == tail.final_engine);
        assert(first.post_ibm == tail.final_ibm);
    } else {
        assert(steps[0] == first);
        assert(steps[steps.len() - 1]
            == tail.steps[tail.steps.len() - 1]);
        assert(tail.steps[tail.steps.len() - 1].post_engine
            == tail.final_engine);
        assert(tail.steps[tail.steps.len() - 1].post_ibm
            == tail.final_ibm);
    }
    assert(observable_serving_trace(
        prepend_observable_trace(first, tail), tail_len + 1,
    ));
}

// Per-request observation removes scheduler timing: steps for other requests
// (and empty schedules) disappear, leaving only this request's concrete sample
// sequence in emission order.
pub open spec fn trace_samples_for(
    steps: Seq<ObservableTraceStep>,
    rid: RequestId,
) -> Seq<(SamplerState, TokenId)>
    decreases steps.len(),
{
    if steps.len() == 0 {
        Seq::empty()
    } else {
        let first = steps[0];
        let tail = steps.subrange(1, steps.len() as int);
        if first.emitted.contains_key(rid) {
            seq![first.samples[rid]] + trace_samples_for(tail, rid)
        } else {
            trace_samples_for(tail, rid)
        }
    }
}

// Canonical autoregressive sample sequence for one request.  This deliberately
// uses only request-view data, so two executable `Vec` clones with the same
// prompt/generated views and sampler state induce the same sequence.
pub open spec fn reference_sample_run(
    model: SemanticModelRepr,
    prompt: Seq<TokenId>,
    generated: Seq<TokenId>,
    sampler_state: SamplerState,
    observations: Seq<(SamplerState, TokenId)>,
) -> bool
    decreases observations.len(),
{
    if observations.len() == 0 {
        true
    } else {
        let sample = observations[0];
        sample == RT::sample_from_repr(
            crate::proof::model::architecture::reference_logits_last_row(
                model,
                crate::proof::reference::request_machine::token_seq_to_int(prompt + generated),
            ),
            sampler_state,
        )
        && reference_sample_run(
            model,
            prompt,
            generated.push(sample.1),
            sample.0,
            observations.subrange(1, observations.len() as int),
        )
    }
}

// Determinism of the canonical per-request run, stated up to the common
// available length so finite traces of different wall-clock step counts can be
// compared directly.
pub proof fn lemma_reference_sample_runs_agree(
    model: SemanticModelRepr,
    prompt: Seq<TokenId>,
    generated: Seq<TokenId>,
    sampler_state: SamplerState,
    left: Seq<(SamplerState, TokenId)>,
    right: Seq<(SamplerState, TokenId)>,
    upto: nat,
)
    requires
        reference_sample_run(model, prompt, generated, sampler_state, left),
        reference_sample_run(model, prompt, generated, sampler_state, right),
        upto <= left.len(),
        upto <= right.len(),
    ensures
        forall|i: int| 0 <= i < upto ==>
            #[trigger] left[i] == right[i],
    decreases upto,
{
    if upto > 0 {
        assert(left.len() > 0);
        assert(right.len() > 0);
        assert(left[0] == RT::sample_from_repr(
            crate::proof::model::architecture::reference_logits_last_row(
                model,
                crate::proof::reference::request_machine::token_seq_to_int(prompt + generated),
            ),
            sampler_state,
        ));
        assert(right[0] == RT::sample_from_repr(
            crate::proof::model::architecture::reference_logits_last_row(
                model,
                crate::proof::reference::request_machine::token_seq_to_int(prompt + generated),
            ),
            sampler_state,
        ));
        assert(left[0] == right[0]);
        let left_tail = left.subrange(1, left.len() as int);
        let right_tail = right.subrange(1, right.len() as int);
        lemma_reference_sample_runs_agree(
            model,
            prompt,
            generated.push(left[0].1),
            left[0].0,
            left_tail,
            right_tail,
            (upto - 1) as nat,
        );
        assert forall|i: int| 0 <= i < upto implies
            #[trigger] left[i] == right[i]
        by {
            if i > 0 {
                assert(left[i] == left_tail[i - 1]);
                assert(right[i] == right_tail[i - 1]);
            }
        }
    }
}

pub proof fn lemma_observable_trace_chain_tail(
    steps: Seq<ObservableTraceStep>,
)
    requires
        observable_trace_chain(steps),
        steps.len() > 0,
    ensures
        observable_trace_chain(steps.subrange(1, steps.len() as int)),
{
    let tail = steps.subrange(1, steps.len() as int);
    assert forall|i: int| 0 <= i < tail.len() implies
        #[trigger] observable_trace_step_agreement(tail[i])
    by {
        assert(tail[i] == steps[i + 1]);
    }
    assert forall|i: int| 0 <= i && i + 1 < tail.len() implies
        #[trigger] tail[i].post_engine == tail[i + 1].pre_engine
            && tail[i].post_ibm == tail[i + 1].pre_ibm
    by {
        assert(tail[i] == steps[i + 1]);
        assert(tail[i + 1] == steps[i + 2]);
    }
}

// Once a request is absent, `ibm_step`'s no-admission property prevents it from
// reappearing, so no later projected sample can exist for that request.
pub proof fn lemma_absent_request_has_no_trace_samples(
    steps: Seq<ObservableTraceStep>,
    rid: RequestId,
)
    requires
        observable_trace_chain(steps),
        steps.len() > 0,
        !steps[0].pre_ibm.machines.contains_key(rid),
    ensures
        trace_samples_for(steps, rid).len() == 0,
    decreases steps.len(),
{
    let first = steps[0];
    let tail = steps.subrange(1, steps.len() as int);
    assert(observable_trace_step_agreement(first));
    assert(ibm_step(
        first.pre_ibm, first.post_ibm, first.emitted.dom(), first.samples,
    ));
    assert(!first.emitted.contains_key(rid));
    assert(!first.post_ibm.machines.contains_key(rid));
    if tail.len() > 0 {
        lemma_observable_trace_chain_tail(steps);
        assert(tail[0].pre_ibm == first.post_ibm);
        lemma_absent_request_has_no_trace_samples(tail, rid);
    }
    assert(trace_samples_for(steps, rid) == trace_samples_for(tail, rid));
    assert(trace_samples_for(tail, rid).len() == 0);
}

// A certified concrete/IBM trace projects to the canonical independent
// autoregressive run for every request that exists at the trace's initial
// state, regardless of how many intervening steps schedule other requests.
#[verifier::spinoff_prover]
#[verifier::rlimit(30)]
pub proof fn lemma_observable_steps_request_sample_run(
    steps: Seq<ObservableTraceStep>,
    rid: RequestId,
)
    requires
        observable_trace_chain(steps),
        steps.len() > 0,
        steps[0].pre_ibm.machines.contains_key(rid),
    ensures ({
        let initial = steps[0].pre_ibm.machines[rid].request_state;
        reference_sample_run(
            ibm_semantic_model(steps[0].pre_ibm),
            initial.prompt_tokens@,
            initial.generated_tokens@,
            initial.sampler_state,
            trace_samples_for(steps, rid),
        )
    }),
    decreases steps.len(),
{
    let first = steps[0];
    let tail = steps.subrange(1, steps.len() as int);
    let initial = first.pre_ibm.machines[rid].request_state;
    assert(observable_trace_step_agreement(first));
    assert(ibm_step(
        first.pre_ibm, first.post_ibm, first.emitted.dom(), first.samples,
    ));
    if tail.len() > 0 {
        lemma_observable_trace_chain_tail(steps);
        assert(tail[0].pre_ibm == first.post_ibm);
    }

    if first.emitted.contains_key(rid) {
        assert(observable_step_agreement(
            first.pre_ibm, first.emitted, first.samples, first.reprs,
        ));
        assert(crate::exec::engine::reprs_emits(first.reprs, rid));
        let k = choose|k: int| 0 <= k < first.reprs.scheduled.len()
            && first.reprs.scheduled[k] == rid
            && first.reprs.sample_mask[k];
        assert(step_samples_match_reference(
            first.pre_ibm, first.samples, first.reprs,
        ));
        assert(first.samples[rid] == RT::sample_from_repr(
            crate::proof::model::architecture::reference_logits_last_row(
                ibm_semantic_model(first.pre_ibm),
                crate::proof::reference::request_machine::token_seq_to_int(history(initial)),
            ),
            initial.sampler_state,
        ));
        assert(history(initial)
            == initial.prompt_tokens@ + initial.generated_tokens@);
        assert(first.samples[rid] == RT::sample_from_repr(
            crate::proof::model::architecture::reference_logits_last_row(
                ibm_semantic_model(first.pre_ibm),
                crate::proof::reference::request_machine::token_seq_to_int(
                    initial.prompt_tokens@ + initial.generated_tokens@,
                ),
            ),
            initial.sampler_state,
        ));

        if should_finish_after_append(initial, first.samples[rid].1) {
            assert(!first.post_ibm.machines.contains_key(rid));
            if tail.len() > 0 {
                assert(!tail[0].pre_ibm.machines.contains_key(rid));
                lemma_absent_request_has_no_trace_samples(tail, rid);
            }
            assert(trace_samples_for(tail, rid).len() == 0);
            assert(reference_sample_run(
                ibm_semantic_model(first.pre_ibm),
                initial.prompt_tokens@,
                initial.generated_tokens@.push(first.samples[rid].1),
                first.samples[rid].0,
                trace_samples_for(tail, rid),
            ));
        } else {
            assert(first.post_ibm.machines.contains_key(rid));
            let post = first.post_ibm.machines[rid].request_state;
            assert(crate::proof::reference::request_machine::machine_step_transition_full(
                initial, post, first.samples[rid].0, first.samples[rid].1,
            ));
            if tail.len() > 0 {
                assert(tail[0].pre_ibm.machines.contains_key(rid));
                lemma_observable_steps_request_sample_run(tail, rid);
                assert(reference_sample_run(
                    ibm_semantic_model(tail[0].pre_ibm),
                    post.prompt_tokens@,
                    post.generated_tokens@,
                    post.sampler_state,
                    trace_samples_for(tail, rid),
                ));
            } else {
                assert(trace_samples_for(tail, rid).len() == 0);
            }
            assert(first.post_ibm.wr == first.pre_ibm.wr);
            assert(post.prompt_tokens@ == initial.prompt_tokens@);
            assert(post.generated_tokens@
                == initial.generated_tokens@.push(first.samples[rid].1));
            assert(post.sampler_state == first.samples[rid].0);
            assert(reference_sample_run(
                ibm_semantic_model(first.pre_ibm),
                initial.prompt_tokens@,
                initial.generated_tokens@.push(first.samples[rid].1),
                first.samples[rid].0,
                trace_samples_for(tail, rid),
            ));
        }
        assert(trace_samples_for(steps, rid)
            == seq![first.samples[rid]] + trace_samples_for(tail, rid));
        assert(trace_samples_for(steps, rid).subrange(
            1, trace_samples_for(steps, rid).len() as int,
        ) == trace_samples_for(tail, rid));
        assert(reference_sample_run(
            ibm_semantic_model(first.pre_ibm),
            initial.prompt_tokens@,
            initial.generated_tokens@,
            initial.sampler_state,
            trace_samples_for(steps, rid),
        ));
    } else {
        assert(first.post_ibm.machines.contains_key(rid));
        assert(first.post_ibm.machines[rid] == first.pre_ibm.machines[rid]);
        if tail.len() > 0 {
            assert(tail[0].pre_ibm.machines.contains_key(rid));
            lemma_observable_steps_request_sample_run(tail, rid);
        } else {
            assert(trace_samples_for(tail, rid).len() == 0);
        }
        assert(first.post_ibm.wr == first.pre_ibm.wr);
        assert(trace_samples_for(steps, rid) == trace_samples_for(tail, rid));
    }
}

pub proof fn lemma_observable_serving_trace_request_sample_run(
    trace: ObservableServingTrace,
    rid: RequestId,
)
    requires
        observable_serving_trace(trace, trace.steps.len()),
        trace.initial_ibm.machines.contains_key(rid),
    ensures ({
        let initial = trace.initial_ibm.machines[rid].request_state;
        reference_sample_run(
            ibm_semantic_model(trace.initial_ibm),
            initial.prompt_tokens@,
            initial.generated_tokens@,
            initial.sampler_state,
            trace_samples_for(trace.steps, rid),
        )
    }),
{
    if trace.steps.len() == 0 {
        assert(trace_samples_for(trace.steps, rid).len() == 0);
    } else {
        assert(trace.steps[0].pre_ibm == trace.initial_ibm);
        lemma_observable_steps_request_sample_run(trace.steps, rid);
    }
}

pub proof fn lemma_dynamic_trace_chain_tail(events: Seq<DynamicServingEvent>)
    requires
        dynamic_trace_chain(events),
        events.len() > 0,
    ensures
        dynamic_trace_chain(events.subrange(1, events.len() as int)),
{
    let tail = events.subrange(1, events.len() as int);
    assert forall|i: int| 0 <= i < tail.len() implies
        #[trigger] dynamic_event_agreement(tail[i])
    by {
        assert(tail[i] == events[i + 1]);
    }
    assert forall|i: int| 0 <= i && i + 1 < tail.len() implies
        #[trigger] dynamic_event_post_engine(tail[i])
            == dynamic_event_pre_engine(tail[i + 1])
        && dynamic_event_post_ibm(tail[i])
            == dynamic_event_pre_ibm(tail[i + 1])
    by {
        assert(tail[i] == events[i + 1]);
        assert(tail[i + 1] == events[i + 2]);
    }
}

// Every valid event retains all previously accepted request ids.  Admission
// grows the ledger by one fresh id; compute leaves it byte-for-byte unchanged.
pub proof fn lemma_dynamic_event_preserves_accepted(
    event: DynamicServingEvent,
    rid: RequestId,
)
    requires
        dynamic_event_agreement(event),
        dynamic_event_pre_engine(event).cs.accepted_requests@.contains_key(rid),
    ensures
        dynamic_event_post_engine(event).cs.accepted_requests@.contains_key(rid),
{
    match event {
        DynamicServingEvent::Admission(step) => {
            assert(admission_trace_step_agreement(step));
            assert(crate::exec::engine::engine_admission_relation(
                &step.pre_engine, &step.post_engine, step.request,
            ));
            assert(cache_scheduler::scheduler_admission_relation(
                &step.pre_engine.cs, &step.post_engine.cs, step.request,
            ));
        },
        DynamicServingEvent::Compute(step) => {
            assert(observable_trace_step_agreement(step));
            assert(crate::exec::engine::architecture_engine_step_relation(
                step.pre_engine,
                step.post_engine,
                step.emitted,
                step.samples,
                step.reprs,
            ));
        },
    }
}

// A valid admission event cannot re-admit an id already present in the
// concrete accepted-request ledger; compute events admit nothing.
#[verifier::spinoff_prover]
pub proof fn lemma_dynamic_event_cannot_readmit_accepted(
    event: DynamicServingEvent,
    rid: RequestId,
)
    requires
        dynamic_event_agreement(event),
        dynamic_event_pre_engine(event).cs.accepted_requests@
            .contains_key(rid),
    ensures
        !dynamic_event_admits(event, rid),
{
    match event {
        DynamicServingEvent::Admission(step) => {
            assert(admission_trace_step_agreement(step));
            if step.request.request_id == rid {
                assert(step.pre_engine.cs.accepted_requests@
                    .contains_key(step.request.request_id));
                assert(!step.pre_engine.cs.accepted_requests@
                    .contains_key(step.request.request_id));
            }
        },
        DynamicServingEvent::Compute(_) => {},
    }
}

#[verifier::spinoff_prover]
pub proof fn lemma_dynamic_events_do_not_admit_cons(
    events: Seq<DynamicServingEvent>,
    rid: RequestId,
)
    requires
        events.len() > 0,
        !dynamic_event_admits(events[0], rid),
        dynamic_events_do_not_admit(
            events.subrange(1, events.len() as int), rid,
        ),
    ensures
        dynamic_events_do_not_admit(events, rid),
{
    let tail = events.subrange(1, events.len() as int);
    assert forall|i: int| 0 <= i < events.len() implies
        !#[trigger] dynamic_event_admits(events[i], rid)
    by {
        if i > 0 {
            assert(0 <= i - 1 < tail.len());
            assert(tail[i - 1] == events[i]);
        }
    }
}

// Once `rid` is in the ledger at the start of a valid event chain, no event in
// that chain can admit `rid` again.  This turns the old trace-level uniqueness
// premise into a consequence of executable engine state.
#[verifier::spinoff_prover]
#[verifier::rlimit(30)]
pub proof fn lemma_dynamic_chain_cannot_readmit(
    events: Seq<DynamicServingEvent>,
    rid: RequestId,
)
    requires
        dynamic_trace_chain(events),
        events.len() > 0,
        dynamic_event_pre_engine(events[0]).cs.accepted_requests@
            .contains_key(rid),
    ensures
        dynamic_events_do_not_admit(events, rid),
    decreases events.len(),
{
    let first = events[0];
    let tail = events.subrange(1, events.len() as int);
    assert(dynamic_event_agreement(first));
    lemma_dynamic_event_cannot_readmit_accepted(first, rid);
    lemma_dynamic_event_preserves_accepted(first, rid);

    if tail.len() > 0 {
        lemma_dynamic_trace_chain_tail(events);
        assert(tail[0] == events[1]);
        assert(dynamic_event_post_engine(events[0])
            == dynamic_event_pre_engine(events[1]));
        assert(dynamic_event_pre_engine(tail[0])
            == dynamic_event_post_engine(first));
        assert(dynamic_event_pre_engine(tail[0]).cs.accepted_requests@
            .contains_key(rid));
        lemma_dynamic_chain_cannot_readmit(tail, rid);
    } else {
        assert(dynamic_events_do_not_admit(tail, rid));
    }
    lemma_dynamic_events_do_not_admit_cons(events, rid);
}

pub proof fn lemma_dynamic_no_admit_tail(
    events: Seq<DynamicServingEvent>,
    rid: RequestId,
)
    requires
        events.len() > 0,
        dynamic_events_do_not_admit(events, rid),
    ensures
        dynamic_events_do_not_admit(
            events.subrange(1, events.len() as int), rid,
        ),
{
    let tail = events.subrange(1, events.len() as int);
    assert forall|i: int| 0 <= i < tail.len() implies
        !#[trigger] dynamic_event_admits(tail[i], rid)
    by {
        assert(tail[i] == events[i + 1]);
    }
}

// With trace-level id uniqueness, neither kind of event can resurrect an
// absent request: compute has no admission, and any admission in the suffix is
// for a different id.
pub proof fn lemma_dynamic_absent_request_has_no_samples(
    events: Seq<DynamicServingEvent>,
    rid: RequestId,
)
    requires
        dynamic_trace_chain(events),
        dynamic_events_do_not_admit(events, rid),
        events.len() > 0,
        !dynamic_event_pre_ibm(events[0]).machines.contains_key(rid),
    ensures
        dynamic_trace_samples_for(events, rid).len() == 0,
    decreases events.len(),
{
    let first = events[0];
    let tail = events.subrange(1, events.len() as int);
    assert(dynamic_event_agreement(first));
    assert(!dynamic_event_admits(first, rid));

    match first {
        DynamicServingEvent::Admission(step) => {
            assert(admission_trace_step_agreement(step));
            assert(step.request.request_id != rid);
            assert(ibm_admission_relation(
                step.pre_ibm, step.post_ibm,
                &step.post_engine, step.request.request_id,
            ));
            assert(!step.post_ibm.machines.contains_key(rid));
        },
        DynamicServingEvent::Compute(step) => {
            assert(observable_trace_step_agreement(step));
            assert(ibm_step(
                step.pre_ibm, step.post_ibm,
                step.emitted.dom(), step.samples,
            ));
            assert(!step.emitted.contains_key(rid));
            assert(!step.post_ibm.machines.contains_key(rid));
        },
    }

    if tail.len() > 0 {
        lemma_dynamic_trace_chain_tail(events);
        lemma_dynamic_no_admit_tail(events, rid);
        assert(tail[0] == events[1]);
        assert(dynamic_event_post_engine(events[0])
            == dynamic_event_pre_engine(events[1]));
        assert(dynamic_event_post_ibm(events[0])
            == dynamic_event_pre_ibm(events[1]));
        assert(dynamic_event_pre_ibm(tail[0])
            == dynamic_event_post_ibm(first));
        assert(!dynamic_event_pre_ibm(tail[0]).machines.contains_key(rid));
        lemma_dynamic_absent_request_has_no_samples(tail, rid);
    }
    assert(dynamic_trace_samples_for(events, rid)
        == dynamic_trace_samples_for(tail, rid));
    assert(dynamic_trace_samples_for(tail, rid).len() == 0);
}

// Dynamic analogue of `lemma_observable_steps_request_sample_run`: admissions
// for other ids are stuttering steps for `rid`; compute events reuse the same
// independent-reference sample agreement as the static trace theorem.
#[verifier::spinoff_prover]
proof fn lemma_dynamic_admission_head_sample_run(
    step: AdmissionTraceStep,
    tail: Seq<DynamicServingEvent>,
    rid: RequestId,
)
    requires
        admission_trace_step_agreement(step),
        step.request.request_id != rid,
        step.pre_ibm.machines.contains_key(rid),
        tail.len() > 0 ==> {
            let post = step.post_ibm.machines[rid].request_state;
            &&& dynamic_event_pre_ibm(tail[0]) == step.post_ibm
            &&& reference_sample_run(
                ibm_semantic_model(step.post_ibm),
                post.prompt_tokens@,
                post.generated_tokens@,
                post.sampler_state,
                dynamic_trace_samples_for(tail, rid),
            )
        },
        tail.len() == 0 ==>
            dynamic_trace_samples_for(tail, rid).len() == 0,
    ensures ({
        let initial = step.pre_ibm.machines[rid].request_state;
        reference_sample_run(
            ibm_semantic_model(step.pre_ibm),
            initial.prompt_tokens@,
            initial.generated_tokens@,
            initial.sampler_state,
            dynamic_trace_samples_for(tail, rid),
        )
    }),
{
    assert(ibm_admission_relation(
        step.pre_ibm, step.post_ibm,
        &step.post_engine, step.request.request_id,
    ));
    assert(step.post_ibm.machines.contains_key(rid));
    assert(step.post_ibm.machines[rid] == step.pre_ibm.machines[rid]);
    assert(step.post_ibm.wr == step.pre_ibm.wr);
}

#[verifier::spinoff_prover]
#[verifier::rlimit(30)]
proof fn lemma_dynamic_compute_head_sample_run(
    step: ObservableTraceStep,
    tail: Seq<DynamicServingEvent>,
    rid: RequestId,
)
    requires
        observable_trace_step_agreement(step),
        step.pre_ibm.machines.contains_key(rid),
        step.post_ibm.machines.contains_key(rid) ==> {
            let post = step.post_ibm.machines[rid].request_state;
            reference_sample_run(
                ibm_semantic_model(step.post_ibm),
                post.prompt_tokens@,
                post.generated_tokens@,
                post.sampler_state,
                dynamic_trace_samples_for(tail, rid),
            )
        },
        !step.post_ibm.machines.contains_key(rid) ==>
            dynamic_trace_samples_for(tail, rid).len() == 0,
    ensures ({
        let initial = step.pre_ibm.machines[rid].request_state;
        let samples = if step.emitted.contains_key(rid) {
            seq![step.samples[rid]] + dynamic_trace_samples_for(tail, rid)
        } else {
            dynamic_trace_samples_for(tail, rid)
        };
        reference_sample_run(
            ibm_semantic_model(step.pre_ibm),
            initial.prompt_tokens@,
            initial.generated_tokens@,
            initial.sampler_state,
            samples,
        )
    }),
{
    // This step only prepends to the tail's samples; its trace need not unfold.
    hide(dynamic_trace_samples_for);
    let initial = step.pre_ibm.machines[rid].request_state;
    assert(ibm_step(
        step.pre_ibm, step.post_ibm,
        step.emitted.dom(), step.samples,
    ));
    assert(step.post_ibm.wr == step.pre_ibm.wr);
    assert(step.post_ibm.architecture_repr
        == step.pre_ibm.architecture_repr);
    assert(ibm_semantic_models_equal(step.post_ibm, step.pre_ibm));
    assert(ibm_semantic_model(step.post_ibm)
        == ibm_semantic_model(step.pre_ibm));
    if step.emitted.contains_key(rid) {
        assert(observable_step_agreement(
            step.pre_ibm, step.emitted, step.samples, step.reprs,
        ));
        assert(crate::exec::engine::reprs_emits(step.reprs, rid));
        let k = choose|k: int| 0 <= k < step.reprs.scheduled.len()
            && step.reprs.scheduled[k] == rid
            && step.reprs.sample_mask[k];
        assert(step_samples_match_reference(
            step.pre_ibm, step.samples, step.reprs,
        ));
        assert(step.samples[rid] == RT::sample_from_repr(
            crate::proof::model::architecture::reference_logits_last_row(
                ibm_semantic_model(step.pre_ibm),
                crate::proof::reference::request_machine::token_seq_to_int(history(initial)),
            ),
            initial.sampler_state,
        ));
        assert(history(initial)
            == initial.prompt_tokens@ + initial.generated_tokens@);
        assert(step.samples[rid] == RT::sample_from_repr(
            crate::proof::model::architecture::reference_logits_last_row(
                ibm_semantic_model(step.pre_ibm),
                crate::proof::reference::request_machine::token_seq_to_int(
                    initial.prompt_tokens@ + initial.generated_tokens@,
                ),
            ),
            initial.sampler_state,
        ));

        if should_finish_after_append(initial, step.samples[rid].1) {
            assert(!step.post_ibm.machines.contains_key(rid));
            assert(dynamic_trace_samples_for(tail, rid).len() == 0);
            assert(reference_sample_run(
                ibm_semantic_model(step.pre_ibm),
                initial.prompt_tokens@,
                initial.generated_tokens@.push(step.samples[rid].1),
                step.samples[rid].0,
                dynamic_trace_samples_for(tail, rid),
            ));
        } else {
            assert(step.post_ibm.machines.contains_key(rid));
            let post = step.post_ibm.machines[rid].request_state;
            assert(crate::proof::reference::request_machine::machine_step_transition_full(
                initial, post, step.samples[rid].0, step.samples[rid].1,
            ));
            assert(step.post_ibm.wr == step.pre_ibm.wr);
            assert(post.prompt_tokens@ == initial.prompt_tokens@);
            assert(post.generated_tokens@
                == initial.generated_tokens@.push(step.samples[rid].1));
            assert(post.sampler_state == step.samples[rid].0);
            assert(reference_sample_run(
                ibm_semantic_model(step.pre_ibm),
                initial.prompt_tokens@,
                initial.generated_tokens@.push(step.samples[rid].1),
                step.samples[rid].0,
                dynamic_trace_samples_for(tail, rid),
            ));
        }
        let samples = seq![step.samples[rid]]
            + dynamic_trace_samples_for(tail, rid);
        assert(samples.subrange(1, samples.len() as int)
            == dynamic_trace_samples_for(tail, rid));
        assert(reference_sample_run(
            ibm_semantic_model(step.pre_ibm),
            initial.prompt_tokens@,
            initial.generated_tokens@,
            initial.sampler_state,
            samples,
        ));
    } else {
        assert(step.post_ibm.machines.contains_key(rid));
        assert(step.post_ibm.machines[rid] == step.pre_ibm.machines[rid]);
        assert(step.post_ibm.wr == step.pre_ibm.wr);
        assert(reference_sample_run(
            ibm_semantic_model(step.pre_ibm),
            initial.prompt_tokens@,
            initial.generated_tokens@,
            initial.sampler_state,
            dynamic_trace_samples_for(tail, rid),
        ));
    }
}

#[verifier::spinoff_prover]
pub proof fn lemma_dynamic_events_request_sample_run(
    events: Seq<DynamicServingEvent>,
    rid: RequestId,
)
    requires
        dynamic_trace_chain(events),
        dynamic_events_do_not_admit(events, rid),
        events.len() > 0,
        dynamic_event_pre_ibm(events[0]).machines.contains_key(rid),
    ensures ({
        let initial_ibm = dynamic_event_pre_ibm(events[0]);
        let initial = initial_ibm.machines[rid].request_state;
        reference_sample_run(
            ibm_semantic_model(initial_ibm),
            initial.prompt_tokens@,
            initial.generated_tokens@,
            initial.sampler_state,
            dynamic_trace_samples_for(events, rid),
        )
    }),
    decreases events.len(),
{
    let first = events[0];
    let tail = events.subrange(1, events.len() as int);
    let initial_ibm = dynamic_event_pre_ibm(first);
    let initial = initial_ibm.machines[rid].request_state;
    assert(dynamic_event_agreement(first));
    assert(!dynamic_event_admits(first, rid));
    if tail.len() > 0 {
        lemma_dynamic_trace_chain_tail(events);
        lemma_dynamic_no_admit_tail(events, rid);
        assert(tail[0] == events[1]);
        assert(dynamic_event_post_engine(events[0])
            == dynamic_event_pre_engine(events[1]));
        assert(dynamic_event_post_ibm(events[0])
            == dynamic_event_pre_ibm(events[1]));
        assert(dynamic_event_pre_ibm(tail[0])
            == dynamic_event_post_ibm(first));
    }

    match first {
        DynamicServingEvent::Admission(step) => {
            assert(admission_trace_step_agreement(step));
            assert(step.request.request_id != rid);
            if tail.len() > 0 {
                assert(dynamic_event_pre_ibm(tail[0]).machines.contains_key(rid));
                lemma_dynamic_events_request_sample_run(tail, rid);
            } else {
                assert(dynamic_trace_samples_for(tail, rid).len() == 0);
            }
            lemma_dynamic_admission_head_sample_run(step, tail, rid);
            assert(dynamic_trace_samples_for(events, rid)
                == dynamic_trace_samples_for(tail, rid));
        },
        DynamicServingEvent::Compute(step) => {
            assert(observable_trace_step_agreement(step));
            assert(ibm_step(
                step.pre_ibm, step.post_ibm,
                step.emitted.dom(), step.samples,
            ));

            if step.emitted.contains_key(rid) {
                if should_finish_after_append(initial, step.samples[rid].1) {
                    assert(!step.post_ibm.machines.contains_key(rid));
                    if tail.len() > 0 {
                        assert(!dynamic_event_pre_ibm(tail[0])
                            .machines.contains_key(rid));
                        lemma_dynamic_absent_request_has_no_samples(tail, rid);
                    }
                    assert(dynamic_trace_samples_for(tail, rid).len() == 0);
                } else {
                    assert(step.post_ibm.machines.contains_key(rid));
                    let post = step.post_ibm.machines[rid].request_state;
                    if tail.len() > 0 {
                        assert(dynamic_event_pre_ibm(tail[0])
                            .machines.contains_key(rid));
                        lemma_dynamic_events_request_sample_run(tail, rid);
                        assert(reference_sample_run(
                            ibm_semantic_model(dynamic_event_pre_ibm(tail[0])),
                            post.prompt_tokens@,
                            post.generated_tokens@,
                            post.sampler_state,
                            dynamic_trace_samples_for(tail, rid),
                        ));
                    } else {
                        assert(dynamic_trace_samples_for(tail, rid).len() == 0);
                        assert(reference_sample_run(
                            ibm_semantic_model(step.post_ibm),
                            post.prompt_tokens@,
                            post.generated_tokens@,
                            post.sampler_state,
                            dynamic_trace_samples_for(tail, rid),
                        ));
                    }
                }
            } else {
                assert(step.post_ibm.machines.contains_key(rid));
                let post = step.post_ibm.machines[rid].request_state;
                if tail.len() > 0 {
                    assert(dynamic_event_pre_ibm(tail[0])
                        .machines.contains_key(rid));
                    lemma_dynamic_events_request_sample_run(tail, rid);
                    assert(reference_sample_run(
                        ibm_semantic_model(step.post_ibm),
                        post.prompt_tokens@,
                        post.generated_tokens@,
                        post.sampler_state,
                        dynamic_trace_samples_for(tail, rid),
                    ));
                } else {
                    assert(dynamic_trace_samples_for(tail, rid).len() == 0);
                    assert(reference_sample_run(
                        ibm_semantic_model(step.post_ibm),
                        post.prompt_tokens@,
                        post.generated_tokens@,
                        post.sampler_state,
                        dynamic_trace_samples_for(tail, rid),
                    ));
                }
            }
            lemma_dynamic_compute_head_sample_run(step, tail, rid);
            assert(dynamic_trace_samples_for(events, rid) ==
                if step.emitted.contains_key(rid) {
                    seq![step.samples[rid]] + dynamic_trace_samples_for(tail, rid)
                } else {
                    dynamic_trace_samples_for(tail, rid)
                });
        },
    }
}

pub proof fn lemma_dynamic_request_trace_sample_run(trace: DynamicRequestTrace)
    requires
        dynamic_request_trace(trace),
    ensures ({
        let request = trace.arrival.request;
        reference_sample_run(
            ibm_semantic_model(trace.arrival.pre_ibm),
            request.prompt_tokens@,
            request.generated_tokens@,
            request.sampler_state,
            dynamic_trace_samples_for(trace.events, request.request_id),
        )
    }),
{
    let rid = trace.arrival.request.request_id;
    assert(admission_trace_step_agreement(trace.arrival));
    assert(ibm_admission_relation(
        trace.arrival.pre_ibm,
        trace.arrival.post_ibm,
        &trace.arrival.post_engine,
        rid,
    ));
    assert(crate::exec::engine::engine_admission_relation(
        &trace.arrival.pre_engine,
        &trace.arrival.post_engine,
        trace.arrival.request,
    ));
    assert(trace.arrival.post_engine.cs.live_requests@[rid]
        == trace.arrival.request);
    assert(trace.arrival.post_ibm.machines.contains_key(rid));
    assert(trace.arrival.post_ibm.machines[rid]
        == initialized_request_machine(&trace.arrival.post_engine, rid));
    assert(trace.arrival.post_ibm.machines[rid].request_state
        == trace.arrival.request);
    assert(trace.arrival.post_ibm.wr == trace.arrival.pre_ibm.wr);
    if trace.events.len() > 0 {
        assert(crate::exec::engine::engine_admission_relation(
            &trace.arrival.pre_engine,
            &trace.arrival.post_engine,
            trace.arrival.request,
        ));
        assert(cache_scheduler::scheduler_admission_relation(
            &trace.arrival.pre_engine.cs,
            &trace.arrival.post_engine.cs,
            trace.arrival.request,
        ));
        assert(trace.arrival.post_engine.cs.accepted_requests@
            .contains_key(rid));
        assert(dynamic_event_pre_engine(trace.events[0])
            == trace.arrival.post_engine);
        lemma_dynamic_chain_cannot_readmit(trace.events, rid);
        assert(dynamic_event_pre_ibm(trace.events[0])
            == trace.arrival.post_ibm);
        lemma_dynamic_events_request_sample_run(trace.events, rid);
    } else {
        assert(dynamic_trace_samples_for(trace.events, rid).len() == 0);
    }
}

// Canonical independent-machine state corresponding to a freshly initialized
// engine request.  All requests are waiting, so no KV position has been
// processed yet; sharing the engine's initial cache representation is harmless
// because `kv_tokens == 0` and subsequent abstract steps construct value-typed
// per-machine stores.
pub open spec fn initialized_request_machine(
    engine: &Engine,
    rid: RequestId,
) -> crate::proof::reference::request_machine::RequestMachine
    recommends engine.cs.live_requests@.contains_key(rid),
{
    crate::proof::reference::request_machine::RequestMachine {
        request_state: engine.cs.live_requests@[rid],
        kv_initialized: false,
        kv_tokens: 0,
        kv_cache_reprs: engine.kv_caches_repr@,
    }
}

pub open spec fn initialized_ibm(engine: &Engine) -> IndependentBatchModel {
    IndependentBatchModel {
        model_config: engine.model_config,
        wr: RT::model_weights_repr_of(&engine.weights_perms@),
        architecture_repr:
            RT::model_weights_architecture_repr_of(&engine.weights_perms@),
        machines: Map::new(
            engine.cs.live_requests@.dom(),
            |rid: RequestId| initialized_request_machine(engine, rid),
        ),
    }
}

// Abstract state after one stable-boundary admission.  Existing machines are
// framed; the new request starts uninitialized with zero logical KV tokens and
// a value copy of the engine's current cache representation.  Because no
// position is readable at `kv_tokens == 0`, this does not claim ownership of
// any resident prefix page.
pub open spec fn admitted_ibm(
    old_ibm: IndependentBatchModel,
    new_engine: &Engine,
    rid: RequestId,
) -> IndependentBatchModel
    recommends new_engine.cs.live_requests@.contains_key(rid),
{
    IndependentBatchModel {
        model_config: old_ibm.model_config,
        wr: old_ibm.wr,
        architecture_repr: old_ibm.architecture_repr,
        machines: old_ibm.machines.insert(
            rid, initialized_request_machine(new_engine, rid),
        ),
    }
}

pub open spec fn ibm_admission_relation(
    old_ibm: IndependentBatchModel,
    new_ibm: IndependentBatchModel,
    new_engine: &Engine,
    rid: RequestId,
) -> bool {
    new_ibm.model_config == old_ibm.model_config
    && new_ibm.wr == old_ibm.wr
    && new_ibm.architecture_repr == old_ibm.architecture_repr
    && new_ibm.machines == old_ibm.machines.insert(
        rid, initialized_request_machine(new_engine, rid),
    )
}

// Architecture-neutral stable-boundary admission.  Scheduler state and cache
// tensors are framed by `engine_admission_relation`; the new private machine
// starts at zero KV tokens, so no family-specific cache construction is needed.
#[verifier::spinoff_prover]
pub proof fn architecture_persistent_semantic_refinement_admission(
    old_e: Engine,
    new_e: Engine,
    old_ibm: IndependentBatchModel,
    request: RequestState,
)
    requires
        architecture_persistent_semantic_runtime_inv(&old_e, &old_ibm),
        crate::exec::engine::engine_admission_relation(&old_e, &new_e, request),
        cache_scheduler::cs_valid(&new_e.cs),
        cache_scheduler::free_queue_valid(&new_e.cs),
        eng_execution_perms_ok(&new_e),
        eng_cache_shape_ok(&new_e),
        cache_scheduler::live_request_step_ready(&new_e.cs),
        cache_scheduler::residency_history_aligned(&new_e.cs),
        cache_scheduler::slot_mapping_aligned(&new_e.cs),
        cache_scheduler::tail_write_exclusive(&new_e.cs),
        cache_scheduler::residency_running_aligned(&new_e.cs),
        cache_scheduler::persistent_provenance_closed(&new_e.cs),
        !old_e.cs.accepted_requests@.contains_key(request.request_id),
        request.generated_tokens@.len() == 0,
        can_step(request),
        request_history_capacity_safe(request),
    ensures
        ibm_admission_relation(
            old_ibm,
            admitted_ibm(old_ibm, &new_e, request.request_id),
            &new_e,
            request.request_id,
        ),
        architecture_persistent_semantic_runtime_inv(
            &new_e,
            &admitted_ibm(old_ibm, &new_e, request.request_id),
        ),
{
    let rid = request.request_id;
    let new_ibm = admitted_ibm(old_ibm, &new_e, rid);
    let old_model = ibm_semantic_model(old_ibm);
    let new_model = ibm_semantic_model(new_ibm);

    reveal(architecture_persistent_semantic_runtime_inv);
    assert(cache_scheduler::scheduler_admission_relation(
        &old_e.cs, &new_e.cs, request,
    ));
    assert(new_e.cs.live_requests@
        == old_e.cs.live_requests@.insert(rid, request));
    assert(new_e.cs.running@ == old_e.cs.running@);
    assert(new_e.cs.request_residency@ == old_e.cs.request_residency@);
    assert(new_e.cs.blocks@ == old_e.cs.blocks@);
    assert(new_e.kv_caches_repr@ == old_e.kv_caches_repr@);
    assert(new_ibm.machines == old_ibm.machines.insert(
        rid, initialized_request_machine(&new_e, rid),
    ));
    assert(ibm_admission_relation(old_ibm, new_ibm, &new_e, rid));

    assert(architecture_semantic_inv(&old_e, &old_ibm));
    assert(architecture_persistent_semantic_inv(&old_e, &old_ibm));
    assert(new_e.model_config == old_e.model_config);
    assert(new_ibm.model_config == old_ibm.model_config);
    assert(new_ibm.model_config == new_e.model_config);
    assert(new_ibm.wr == old_ibm.wr);
    assert(new_ibm.architecture_repr == old_ibm.architecture_repr);
    assert(new_model == old_model);
    lemma_ibm_model_architecture_valid_preserved(old_ibm, new_ibm);

    assert(eng_valid(&new_e)) by {
        assert(eng_valid(&old_e));
    }
    assert(ibm_valid(new_ibm)) by {
        assert forall|r: RequestId|
            #[trigger] new_ibm.machines.contains_key(r)
            implies request_machine_alive(
                new_ibm.machines[r], new_ibm.model_config,
            ) && new_ibm.machines[r].request_state.request_id == r
        by {
            if r == rid {
                let machine = new_ibm.machines[r];
                assert(new_e.cs.live_requests@[rid] == request);
                assert(machine == initialized_request_machine(&new_e, rid));
                assert(machine.request_state == request);
                assert(valid_request_state(request));
                assert(request.prompt_tokens@.len() > 0);
                assert(history(request).len() > 0);
                assert(machine.kv_cache_reprs.len()
                    == new_e.model_config.num_layers as nat);
                assert(!machine.kv_initialized);
                assert(machine.kv_tokens == 0);
            } else {
                assert(old_ibm.machines.contains_key(r));
                assert(new_ibm.machines[r] == old_ibm.machines[r]);
                assert(request_machine_alive(
                    old_ibm.machines[r], old_ibm.model_config,
                ));
            }
        }
    }
    assert(shared_rid_keyset(&new_e.cs, &new_ibm)) by {
        assert(shared_rid_keyset(&old_e.cs, &old_ibm));
        assert(new_e.cs.live_requests@.dom()
            == old_e.cs.live_requests@.dom().insert(rid));
        assert(new_ibm.machines.dom()
            == old_ibm.machines.dom().insert(rid));
    }
    assert(shared_rid_coherence(&new_e.cs, &new_ibm)) by {
        assert forall|r: RequestId|
            new_e.cs.live_requests@.contains_key(r)
                && new_ibm.machines.contains_key(r)
            implies #[trigger] request_state_view_eq(
                new_e.cs.live_requests@[r], new_ibm.machines[r].request_state,
            )
        by {
            if r == rid {
                assert(new_e.cs.live_requests@[r] == request);
                assert(new_ibm.machines[r]
                    == initialized_request_machine(&new_e, rid));
                lemma_request_state_view_eq_reflexive(request);
            } else {
                assert(old_e.cs.live_requests@.contains_key(r));
                assert(old_ibm.machines.contains_key(r));
                assert(new_e.cs.live_requests@[r]
                    == old_e.cs.live_requests@[r]);
                assert(new_ibm.machines[r] == old_ibm.machines[r]);
                assert(shared_rid_coherence(&old_e.cs, &old_ibm));
            }
        }
    }
    assert(engine_kv_coherent(&new_e, &new_ibm)) by {
        assert forall|r: RequestId, layer: int, pos: nat|
            #![trigger new_e.kv_caches_repr@[layer],
                crate::proof::tensor::geometry::block_table_slot(
                    residency_block_table(&new_e.cs, r), pos,
                )]
            new_e.cs.live_requests@.contains_key(r)
            && new_ibm.machines.contains_key(r)
            && 0 <= layer < new_e.model_config.num_layers as int
            && pos < new_ibm.machines[r].kv_tokens
            implies {
                let bt = residency_block_table(&new_e.cs, r);
                crate::proof::tensor::geometry::cache_at(
                    new_e.kv_caches_repr@[layer].0,
                    crate::proof::tensor::geometry::block_table_slot(bt, pos),
                ) == crate::proof::tensor::geometry::cache_at(
                    new_ibm.machines[r].kv_cache_reprs[layer].0, pos,
                )
                && crate::proof::tensor::geometry::cache_at(
                    new_e.kv_caches_repr@[layer].1,
                    crate::proof::tensor::geometry::block_table_slot(bt, pos),
                ) == crate::proof::tensor::geometry::cache_at(
                    new_ibm.machines[r].kv_cache_reprs[layer].1, pos,
                )
            }
        by {
            if r == rid {
                assert(new_ibm.machines[r]
                    == initialized_request_machine(&new_e, rid));
                assert(new_ibm.machines[r].kv_tokens == 0);
                assert(false);
            } else {
                assert(old_e.cs.live_requests@.contains_key(r));
                assert(old_ibm.machines.contains_key(r));
                assert(new_ibm.machines[r] == old_ibm.machines[r]);
                assert(residency_block_table(&new_e.cs, r)
                    == residency_block_table(&old_e.cs, r));
                assert(engine_kv_coherent(&old_e, &old_ibm));
            }
        }
    }
    assert(inv(&new_e, &new_ibm));

    assert(new_ibm.wr == RT::model_weights_repr_of(&new_e.weights_perms@));
    assert(new_ibm.architecture_repr
        == RT::model_weights_architecture_repr_of(&new_e.weights_perms@));
    assert(MODEL_CACHE::ibm_cache_fidelity(&new_ibm)) by {
        assert forall|r: RequestId| new_ibm.machines.contains_key(r)
            implies #[trigger] MODEL_CACHE::machine_cache_fidelity(
                new_ibm.machines[r], new_model,
            )
        by {
            if r == rid {
                let machine = new_ibm.machines[r];
                assert(machine == initialized_request_machine(&new_e, rid));
                assert(machine.kv_cache_reprs.len()
                    == new_model.weights.layers.len());
                assert(machine.kv_tokens == 0);
                assert(MODEL_CACHE::cached_prefix_supported(0));
                assert(MODEL_CACHE::cache_reprs_match_canonical(
                    machine.kv_cache_reprs,
                    new_model,
                    token_seq_to_int(history(machine.request_state)),
                    0,
                )) by {
                    reveal(MODEL_CACHE::cache_reprs_match_canonical);
                }
                reveal(MODEL_CACHE::machine_cache_fidelity);
            } else {
                assert(old_ibm.machines.contains_key(r));
                assert(new_ibm.machines[r] == old_ibm.machines[r]);
                assert(MODEL_CACHE::ibm_cache_fidelity(&old_ibm));
            }
        }
    }
    assert(crate::proof::cache::provenance::provenance_cache_fidelity(
        &new_e, new_model,
    )) by {
        assert forall|chain: Seq<BlockId>, request_tokens: Seq<TokenId>,
                c_tokens: nat, layer: int, pos: nat|
            #![trigger crate::proof::cache::provenance::registered_cache_cell_is_canonical(
                &new_e, new_model, chain, request_tokens, layer, pos,
            ), crate::proof::cache::provenance::provenance_prefix_candidate(
                &new_e.cs, chain, request_tokens, c_tokens,
            )]
            crate::proof::cache::provenance::provenance_prefix_candidate(
                &new_e.cs, chain, request_tokens, c_tokens,
            )
            && 0 <= layer < new_model.weights.layers.len()
            && pos < c_tokens
            implies crate::proof::cache::provenance::registered_cache_cell_is_canonical(
                &new_e, new_model, chain, request_tokens, layer, pos,
            )
        by {
            assert(crate::proof::cache::provenance::provenance_prefix_candidate(
                &old_e.cs, chain, request_tokens, c_tokens,
            ));
            assert(crate::proof::cache::provenance::registered_cache_cell_is_canonical(
                &old_e, old_model, chain, request_tokens, layer, pos,
            ));
            reveal(crate::proof::cache::provenance::registered_cache_cell_is_canonical);
        }
    }
    assert(architecture_semantic_inv(&new_e, &new_ibm));
    assert(architecture_persistent_semantic_inv(&new_e, &new_ibm));

    assert(phase_aligned(&new_e, &new_ibm)) by {
        assert forall|r: RequestId|
            new_e.cs.live_requests@.contains_key(r)
                && new_ibm.machines.contains_key(r)
            implies (new_e.cs.running@.contains(r)
                <==> #[trigger] new_ibm.machines[r].kv_initialized)
        by {
            if r == rid {
                assert(!old_e.cs.running@.contains(rid)) by {
                    if old_e.cs.running@.contains(rid) {
                        assert(cache_scheduler::live_covers_queue(&old_e.cs));
                        assert(old_e.cs.live_requests@.contains_key(rid));
                    }
                }
                assert(!new_e.cs.running@.contains(rid));
                assert(new_ibm.machines[r]
                    == initialized_request_machine(&new_e, rid));
                assert(!new_ibm.machines[r].kv_initialized);
            } else {
                assert(old_e.cs.live_requests@.contains_key(r));
                assert(old_ibm.machines.contains_key(r));
                assert(new_ibm.machines[r] == old_ibm.machines[r]);
                assert(phase_aligned(&old_e, &old_ibm));
            }
        }
    }
    assert(mach_cache_shape_ok(&new_ibm, new_e.cs.num_blocks as nat)) by {
        assert forall|r: RequestId, layer: int|
            #![trigger new_ibm.machines[r].kv_cache_reprs[layer]]
            new_ibm.machines.contains_key(r)
            && 0 <= layer < new_ibm.machines[r].kv_cache_reprs.len()
            implies {
                &&& new_ibm.machines[r].kv_cache_reprs[layer].0.len()
                    == new_e.cs.num_blocks as nat
                &&& new_ibm.machines[r].kv_cache_reprs[layer].1.len()
                    == new_e.cs.num_blocks as nat
                &&& (forall|p: int| 0 <= p < new_e.cs.num_blocks as nat
                    ==> (#[trigger]
                        new_ibm.machines[r].kv_cache_reprs[layer].0[p]).len()
                        == crate::types::BLOCK_SIZE_SPEC as int)
                &&& (forall|p: int| 0 <= p < new_e.cs.num_blocks as nat
                    ==> (#[trigger]
                        new_ibm.machines[r].kv_cache_reprs[layer].1[p]).len()
                        == crate::types::BLOCK_SIZE_SPEC as int)
            }
        by {
            if r == rid {
                assert(new_ibm.machines[r]
                    == initialized_request_machine(&new_e, rid));
            } else {
                assert(old_ibm.machines.contains_key(r));
                assert(new_ibm.machines[r] == old_ibm.machines[r]);
                assert(old_e.cs.num_blocks == new_e.cs.num_blocks);
                assert(mach_cache_shape_ok(
                    &old_ibm, old_e.cs.num_blocks as nat,
                ));
            }
        }
    }
    assert forall|r: RequestId|
        #[trigger] new_e.cs.live_requests@.contains_key(r)
        implies valid_request_state(new_e.cs.live_requests@[r])
    by {
        assert(can_step(new_e.cs.live_requests@[r]));
    }
    assert(architecture_persistent_semantic_runtime_inv(
        &new_e, &new_ibm,
    ));
}

// Actual-request initialization for any qualified executable architecture.
// Every initial machine has zero logical KV tokens, so the generic cache
// fidelity invariant is established without inspecting a family layer fold.
#[verifier::spinoff_prover]
pub proof fn architecture_semantic_refinement_initialized(
    engine: &Engine,
)
    requires
        cache_scheduler::cs_valid(&engine.cs),
        eng_execution_perms_ok(engine),
        forall|rid: RequestId|
            #[trigger] engine.cs.live_requests@.contains_key(rid) ==> {
                &&& valid_request_state(engine.cs.live_requests@[rid])
                &&& engine.cs.live_requests@[rid].request_id == rid
            },
    ensures
        architecture_semantic_inv(engine, &initialized_ibm(engine)),
{
    let ibm = initialized_ibm(engine);
    assert(ibm.model_config == engine.model_config);
    assert(ibm.wr == RT::model_weights_repr_of(&engine.weights_perms@));
    assert(ibm.architecture_repr
        == RT::model_weights_architecture_repr_of(&engine.weights_perms@));
    reveal(eng_execution_perms_ok);
    RT::lemma_model_weights_architecture_repr_valid(
        &engine.weights, &engine.runtime, &engine.weights_perms@,
    );
    assert(model_weights_architecture_repr_valid(
        ibm.wr, ibm.architecture_repr,
    ));
    crate::proof::engine::architecture::lemma_execution_cache_refinement_supported(
        &engine.weights, &engine.runtime, &engine.weights_perms@,
    );
    assert(crate::proof::model::architecture::cache_refinement_supported(
        ibm_semantic_model(ibm),
    ));
    assert(ibm.wr.architecture == ibm.model_config.architecture);
    lemma_ibm_model_architecture_valid(ibm);
    assert(ibm.wr.layers.len()
        == engine.model_config.num_layers as nat);

    assert forall|rid: RequestId|
        #[trigger] ibm.machines.contains_key(rid)
        <==> engine.cs.live_requests@.contains_key(rid) by {}
    assert(engine.cs.live_requests@.dom() =~= ibm.machines.dom());
    assert forall|rid: RequestId| #[trigger] ibm.machines.contains_key(rid)
        implies request_machine_alive(
            ibm.machines[rid], ibm.model_config,
        ) && ibm.machines[rid].request_state.request_id == rid
    by {
        let machine = ibm.machines[rid];
        assert(engine.cs.live_requests@.contains_key(rid));
        assert(machine == initialized_request_machine(engine, rid));
        assert(valid_request_state(machine.request_state));
        assert(machine.request_state.prompt_tokens@.len() > 0);
        assert(history(machine.request_state).len() > 0);
        assert(machine.kv_cache_reprs.len()
            == engine.model_config.num_layers as nat);
        assert(!machine.kv_initialized);
        assert(machine.kv_tokens == 0);
    }
    assert(ibm_valid(ibm));
    assert(shared_rid_keyset(&engine.cs, &ibm));
    assert(shared_rid_coherence(&engine.cs, &ibm)) by {
        assert forall|rid: RequestId|
            engine.cs.live_requests@.contains_key(rid)
                && ibm.machines.contains_key(rid)
            implies #[trigger] request_state_view_eq(
                engine.cs.live_requests@[rid],
                ibm.machines[rid].request_state,
            )
        by {
            assert(ibm.machines[rid]
                == initialized_request_machine(engine, rid));
            assert(ibm.machines[rid].request_state
                == engine.cs.live_requests@[rid]);
            lemma_request_state_view_eq_reflexive(
                engine.cs.live_requests@[rid],
            );
        }
    }
    assert(engine_kv_coherent(engine, &ibm)) by {
        assert forall|rid: RequestId, layer: int, pos: nat|
            #![trigger engine.kv_caches_repr@[layer],
                crate::proof::tensor::geometry::block_table_slot(
                    residency_block_table(&engine.cs, rid), pos,
                )]
            engine.cs.live_requests@.contains_key(rid)
            && ibm.machines.contains_key(rid)
            && 0 <= layer < engine.model_config.num_layers as int
            && pos < ibm.machines[rid].kv_tokens
            implies {
                let bt = residency_block_table(&engine.cs, rid);
                crate::proof::tensor::geometry::cache_at(
                    engine.kv_caches_repr@[layer].0,
                    crate::proof::tensor::geometry::block_table_slot(bt, pos),
                ) == crate::proof::tensor::geometry::cache_at(
                    ibm.machines[rid].kv_cache_reprs[layer].0, pos,
                )
                && crate::proof::tensor::geometry::cache_at(
                    engine.kv_caches_repr@[layer].1,
                    crate::proof::tensor::geometry::block_table_slot(bt, pos),
                ) == crate::proof::tensor::geometry::cache_at(
                    ibm.machines[rid].kv_cache_reprs[layer].1, pos,
                )
            }
        by {
            assert(ibm.machines[rid].kv_tokens == 0);
            assert(false);
        }
    }
    assert(inv(engine, &ibm));
    assert(MODEL_CACHE::ibm_cache_fidelity(&ibm)) by {
        assert forall|rid: RequestId| ibm.machines.contains_key(rid)
            implies #[trigger] MODEL_CACHE::machine_cache_fidelity(
                ibm.machines[rid], ibm_semantic_model(ibm),
            )
        by {
            let machine = ibm.machines[rid];
            assert(machine == initialized_request_machine(engine, rid));
            assert(machine.kv_cache_reprs.len()
                == ibm.wr.layers.len());
            assert(machine.kv_tokens == 0);
            assert(MODEL_CACHE::cached_prefix_supported(0));
            assert(MODEL_CACHE::cache_reprs_match_canonical(
                machine.kv_cache_reprs,
                ibm_semantic_model(ibm),
                token_seq_to_int(history(machine.request_state)),
                0,
            )) by {
                reveal(MODEL_CACHE::cache_reprs_match_canonical);
            }
            reveal(MODEL_CACHE::machine_cache_fidelity);
        }
    }
    assert(architecture_semantic_inv(engine, &ibm));
}

// Fresh machines have not materialized KV, exactly matching an empty running
// queue.  This structural fact is independent of model semantics.
pub proof fn lemma_initialized_ibm_phase_aligned(engine: &Engine)
    requires
        engine.cs.running@.len() == 0,
    ensures
        phase_aligned(engine, &initialized_ibm(engine)),
{
    let ibm = initialized_ibm(engine);
    assert forall|rid: RequestId|
        engine.cs.live_requests@.contains_key(rid)
        && ibm.machines.contains_key(rid)
        implies (engine.cs.running@.contains(rid)
            <==> #[trigger] ibm.machines[rid].kv_initialized)
    by {
        assert(!engine.cs.running@.contains(rid));
        assert(ibm.machines[rid] == initialized_request_machine(engine, rid));
        assert(!ibm.machines[rid].kv_initialized);
    }
}

// Every freshly initialized private machine is a value projection of the
// engine cache representation, so the common engine page-shape invariant
// immediately supplies the private-cache shape companion.
pub proof fn lemma_initialized_ibm_cache_shape(
    engine: &Engine,
)
    requires
        eng_cache_shape_ok(engine),
    ensures
        mach_cache_shape_ok(
            &initialized_ibm(engine), engine.cs.num_blocks as nat,
        ),
{
    let ibm = initialized_ibm(engine);
    assert forall|rid: RequestId, layer: int|
        #![trigger ibm.machines[rid].kv_cache_reprs[layer]]
        ibm.machines.contains_key(rid)
        && 0 <= layer < ibm.machines[rid].kv_cache_reprs.len()
        implies {
            &&& ibm.machines[rid].kv_cache_reprs[layer].0.len()
                == engine.cs.num_blocks as nat
            &&& ibm.machines[rid].kv_cache_reprs[layer].1.len()
                == engine.cs.num_blocks as nat
            &&& (forall|p: int| 0 <= p < engine.cs.num_blocks as nat
                ==> (#[trigger]
                    ibm.machines[rid].kv_cache_reprs[layer].0[p]).len()
                    == crate::types::BLOCK_SIZE_SPEC as int)
            &&& (forall|p: int| 0 <= p < engine.cs.num_blocks as nat
                ==> (#[trigger]
                    ibm.machines[rid].kv_cache_reprs[layer].1[p]).len()
                    == crate::types::BLOCK_SIZE_SPEC as int)
        }
    by {
        assert(ibm.machines[rid]
            == initialized_request_machine(engine, rid));
        assert(ibm.machines[rid].kv_cache_reprs
            == engine.kv_caches_repr@);
    }
}

// Actual-request persistent initialization for every qualified executable
// architecture.  All model-dependent work is delegated to the common semantic
// initializer; physical provenance is vacuous because no page exists yet.
#[verifier::spinoff_prover]
pub proof fn architecture_persistent_semantic_refinement_initialized(
    engine: &Engine,
)
    requires
        cache_scheduler::cs_valid(&engine.cs),
        cache_scheduler::free_queue_valid(&engine.cs),
        eng_execution_perms_ok(engine),
        engine.model_config.num_layers > 0,
        engine.cs.num_blocks <= u64::MAX / crate::types::BLOCK_SIZE,
        engine.cs.running@.len() == 0,
        engine.cs.request_residency@.dom().is_empty(),
        engine.cs.blocks@.dom().is_empty(),
        forall|rid: RequestId|
            #[trigger] engine.cs.live_requests@.contains_key(rid) ==> {
                &&& valid_request_state(engine.cs.live_requests@[rid])
                &&& engine.cs.live_requests@[rid].request_id == rid
            },
        cache_scheduler::live_request_step_ready(&engine.cs),
        eng_cache_shape_ok(engine),
        cache_scheduler::residency_running_aligned(&engine.cs),
        cache_scheduler::persistent_provenance_closed(&engine.cs),
    ensures
        architecture_persistent_semantic_runtime_inv(
            engine, &initialized_ibm(engine),
        ),
{
    let ibm = initialized_ibm(engine);
    architecture_semantic_refinement_initialized(engine);
    crate::proof::cache::provenance::lemma_empty_physical_cache_has_provenance_fidelity(
        engine, ibm_semantic_model(ibm),
    );
    assert(architecture_persistent_semantic_inv(engine, &ibm));

    assert(cache_scheduler::residency_history_aligned(&engine.cs));
    assert(cache_scheduler::slot_mapping_aligned(&engine.cs));
    assert(cache_scheduler::tail_write_exclusive(&engine.cs));
    lemma_initialized_ibm_phase_aligned(engine);
    lemma_initialized_ibm_cache_shape(engine);
    assert(architecture_persistent_semantic_runtime_inv(engine, &ibm));
}

// Cache-constructor-independent structural simulation.  Request lifecycle and
// state coherence depend only on the opaque forward payload and `ibm_step`;
// the cache constructor supplies the resulting engine/machine coherence as a
// separate checked fact.
pub proof fn refinement_step_sim_from_coherence(
    old_e: Engine,
    new_e: Engine,
    old_ibm: IndependentBatchModel,
    new_ibm: IndependentBatchModel,
    emitted: Map<RequestId, TokenId>,
    samples: Map<RequestId, (SamplerState, TokenId)>,
    reprs: crate::exec::engine::StepReprs,
    logits_repr: Tensor2D,
    post_kv: Seq<(KVCacheLayerRepr, KVCacheLayerRepr)>,
)
    requires
        inv(&old_e, &old_ibm),
        crate::exec::engine::engine_step_relation_with_payload(
            old_e, new_e, emitted, samples, reprs, logits_repr, post_kv,
        ),
        ibm_step(old_ibm, new_ibm, emitted.dom(), samples),
        engine_kv_coherent(&new_e, &new_ibm),
    ensures
        inv(&new_e, &new_ibm),
{
    // model_config: new_e == old_e == old_ibm == new_ibm.
    assert(new_e.model_config == new_ibm.model_config);
    // cs_valid(new_e): directly from engine_step_relation.
    // eng_valid(new_e): cache shapes preserved + model_config equal ⇒ both lengths
    // still equal num_layers.
    assert(eng_valid(&new_e));
    // shared_rid_coherence: each surviving shared request evolves view-identically
    // on both sides under the same sample map, so coherence is preserved.
    assert forall|rid: RequestId|
        new_e.cs.live_requests@.contains_key(rid) && new_ibm.machines.contains_key(rid)
        implies #[trigger] request_state_view_eq(new_e.cs.live_requests@[rid],
            new_ibm.machines[rid].request_state)
    by {
        // Survivors were live on both sides before the step.
        assert(old_e.cs.live_requests@.contains_key(rid));
        assert(old_ibm.machines.contains_key(rid));
        // inv(old) ⇒ pre-states are view-equal.
        assert(request_state_view_eq(old_e.cs.live_requests@[rid],
            old_ibm.machines[rid].request_state));
        if emitted.contains_key(rid) {
            // Scheduled survivor: both sides apply the full transition with the
            // same sample, so post-states stay view-equal.
            lemma_machine_step_full_preserves_view_eq(
                old_e.cs.live_requests@[rid], new_e.cs.live_requests@[rid],
                old_ibm.machines[rid].request_state, new_ibm.machines[rid].request_state,
                samples[rid].0, samples[rid].1);
        } else {
            // Unscheduled survivor: both sides unchanged.
        }
    }
    // shared_rid_keyset: both sides keep exactly the unfinished requests.  The
    // engine's survival rule (in `engine_step_relation`) and the abstract removal
    // rule (in `ibm_step`) agree because the pre-states are view-equal (so
    // `should_finish_after_append` matches) and the schedules coincide
    // (`selected == emitted.dom`).
    assert(new_e.cs.live_requests@.dom() =~= new_ibm.machines.dom()) by {
        assert forall|rid: RequestId|
            new_e.cs.live_requests@.contains_key(rid) <==> new_ibm.machines.contains_key(rid)
        by {
            if old_e.cs.live_requests@.contains_key(rid) {
                // old keysets equal, pre-states view-equal ⇒ same finish decision.
                assert(old_ibm.machines.contains_key(rid));
                if emitted.contains_key(rid) {
                    lemma_should_finish_respects_view_eq(old_e.cs.live_requests@[rid],
                        old_ibm.machines[rid].request_state, emitted[rid]);
                }
            }
        }
    }
    // ibm_valid(new_ibm): unchanged machines stay alive (from inv(old)); stepped
    // machines stay alive by `ibm_step`'s contract; `wr`/`model_config` unchanged.
    lemma_ibm_model_architecture_valid_preserved(old_ibm, new_ibm);
    assert(ibm_valid(new_ibm)) by {
        assert forall|rid: RequestId| #[trigger] new_ibm.machines.contains_key(rid)
            implies crate::proof::reference::request_machine::request_machine_alive(
                new_ibm.machines[rid], new_ibm.model_config)
                && new_ibm.machines[rid].request_state.request_id == rid
        by {
            assert(old_ibm.machines.contains_key(rid));
            if emitted.contains_key(rid) {
                assert(crate::proof::reference::request_machine::machine_step_transition_full(
                    old_ibm.machines[rid].request_state,
                    new_ibm.machines[rid].request_state,
                    samples[rid].0,
                    samples[rid].1,
                ));
                lemma_request_lifecycle_view_eq_fields(
                    new_ibm.machines[rid].request_state,
                    old_ibm.machines[rid].request_state,
                );
            }
        }
    }
}

// Forward-simulation core parameterized by the concrete model payload. It
// derives post-step coherence from the common store-shaped contract, then
// delegates all model-opaque structure above.

// Architecture-neutral semantic forward simulation. Scheduler and physical
// cache coherence use the payload-opaque structural proof above; the
// only semantic premise is the common IBM cache-extension contract over the
// complete `SemanticModelRepr`.

// Architecture-native one-step refinement for the singleton-forward private
// cache constructor.  All request-machine structure is constructed here; the
// remaining semantic obligations are exactly post-step engine/private cache
// coherence and the common canonical-cache extension law.
pub proof fn architecture_semantic_refinement_step_from_cache_facts(
    old_e: Engine,
    new_e: Engine,
    old_ibm: IndependentBatchModel,
    emitted: Map<RequestId, TokenId>,
    samples: Map<RequestId, (SamplerState, TokenId)>,
    reprs: crate::exec::engine::StepReprs,
)
    requires
        architecture_semantic_inv(&old_e, &old_ibm),
        crate::exec::engine::architecture_engine_step_relation(
            old_e, new_e, emitted, samples, reprs,
        ),
        crate::exec::engine::engine_step_semantic_identity(old_e, new_e),
        engine_kv_coherent(
            &new_e,
            &crate::proof::engine::abstract_step::architecture_stepped_ibm(
                old_e, new_e, old_ibm, emitted, reprs,
            ),
        ),
        MODEL_CACHE::ibm_cache_extension(
            old_ibm,
            crate::proof::engine::abstract_step::architecture_stepped_ibm(
                old_e, new_e, old_ibm, emitted, reprs,
            ),
        ),
    ensures
        architecture_semantic_inv(
            &new_e,
            &crate::proof::engine::abstract_step::architecture_stepped_ibm(
                old_e, new_e, old_ibm, emitted, reprs,
            ),
        ),
        ibm_step(
            old_ibm,
            crate::proof::engine::abstract_step::architecture_stepped_ibm(
                old_e, new_e, old_ibm, emitted, reprs,
            ),
            emitted.dom(),
            samples,
        ),
{
    let new_ibm = crate::proof::engine::abstract_step::architecture_stepped_ibm(
        old_e, new_e, old_ibm, emitted, reprs,
    );
    reveal(architecture_semantic_inv);
    lemma_ibm_valid_semantic_model(old_ibm);
    crate::proof::engine::abstract_step::construct_architecture_abstract_step(
        old_e, new_e, old_ibm, emitted, samples, reprs,
    );
    refinement_step_sim_from_coherence(
        old_e,
        new_e,
        old_ibm,
        new_ibm,
        emitted,
        samples,
        reprs,
        crate::exec::engine::architecture_step_logits_repr(old_e, reprs),
        crate::exec::engine::architecture_engine_post_kv_of(
            old_e, reprs, old_e.kv_caches_repr@,
        ),
    );
    assert(new_ibm.wr == old_ibm.wr);
    assert(new_ibm.architecture_repr == old_ibm.architecture_repr);
    assert(reprs.wr == RT::model_weights_repr_of(
        &old_e.weights_perms@,
    ));
    assert(RT::model_weights_repr_of(&new_e.weights_perms@) == reprs.wr);
    reveal(crate::exec::engine::engine_step_semantic_identity);
    assert(new_ibm.architecture_repr
        == RT::model_weights_architecture_repr_of(
            &new_e.weights_perms@,
        ));
    assert(ibm_semantic_model(new_ibm) == ibm_semantic_model(old_ibm));
    assert(crate::proof::model::architecture::cache_refinement_supported(
        ibm_semantic_model(new_ibm),
    ));
    MODEL_CACHE::lemma_ibm_cache_extension_preserves_fidelity(
        old_ibm, new_ibm,
    );
}

// Architecture-native refinement with post-step coherence fully derived from
// the common scheduler/layout and whole-model relocation proofs.  The sole
// remaining semantic premise is canonical private-cache extension.
pub proof fn architecture_semantic_refinement_step_from_cache_extension(
    old_e: Engine,
    new_e: Engine,
    old_ibm: IndependentBatchModel,
    emitted: Map<RequestId, TokenId>,
    samples: Map<RequestId, (SamplerState, TokenId)>,
    reprs: crate::exec::engine::StepReprs,
)
    requires
        crate::boundary::tensor_runtime::paged_attention_numeric_domain(),
        architecture_semantic_inv(&old_e, &old_ibm),
        phase_aligned(&old_e, &old_ibm),
        crate::exec::engine::eng_cache_shape_ok(&old_e),
        mach_cache_shape_ok(&old_ibm, old_e.cs.num_blocks as nat),
        cache_scheduler::residency_history_aligned(&new_e.cs),
        old_e.model_config.num_layers > 0,
        crate::exec::engine::architecture_engine_step_relation(
            old_e, new_e, emitted, samples, reprs,
        ),
        crate::exec::engine::engine_step_semantic_identity(old_e, new_e),
        MODEL_CACHE::ibm_cache_extension(
            old_ibm,
            crate::proof::engine::abstract_step::architecture_stepped_ibm(
                old_e, new_e, old_ibm, emitted, reprs,
            ),
        ),
    ensures
        architecture_semantic_inv(
            &new_e,
            &crate::proof::engine::abstract_step::architecture_stepped_ibm(
                old_e, new_e, old_ibm, emitted, reprs,
            ),
        ),
        ibm_step(
            old_ibm,
            crate::proof::engine::abstract_step::architecture_stepped_ibm(
                old_e, new_e, old_ibm, emitted, reprs,
            ),
            emitted.dom(),
            samples,
        ),
{
    crate::proof::cache::coherence::derive_architecture_engine_kv_coherent(
        old_e, new_e, old_ibm, emitted, samples, reprs,
    );
    architecture_semantic_refinement_step_from_cache_facts(
        old_e, new_e, old_ibm, emitted, samples, reprs,
    );
}

// Architecture-native one-step refinement with both semantic cache extension
// and post-step physical coherence derived.  The only cache premise is the
// architecture-neutral certificate over reusable pre-step prefixes; no family
// semantics, per-family layout theorem, or assumed IBM extension reaches this
// interface.
pub proof fn architecture_semantic_refinement_step_from_registered_cache(
    old_e: Engine,
    new_e: Engine,
    old_ibm: IndependentBatchModel,
    emitted: Map<RequestId, TokenId>,
    samples: Map<RequestId, (SamplerState, TokenId)>,
    reprs: crate::exec::engine::StepReprs,
)
    requires
        crate::boundary::tensor_runtime::paged_attention_numeric_domain(),
        architecture_semantic_inv(&old_e, &old_ibm),
        phase_aligned(&old_e, &old_ibm),
        crate::proof::cache::provenance::registered_cache_fidelity(
            &old_e, ibm_semantic_model(old_ibm),
        ),
        crate::exec::engine::eng_cache_shape_ok(&old_e),
        mach_cache_shape_ok(&old_ibm, old_e.cs.num_blocks as nat),
        cache_scheduler::residency_history_aligned(&new_e.cs),
        old_e.model_config.num_layers > 0,
        old_e.cs.num_blocks <= u64::MAX,
        crate::exec::engine::architecture_engine_step_relation(
            old_e, new_e, emitted, samples, reprs,
        ),
        crate::exec::engine::engine_step_semantic_identity(old_e, new_e),
    ensures
        architecture_semantic_inv(
            &new_e,
            &crate::proof::engine::abstract_step::architecture_stepped_ibm(
                old_e, new_e, old_ibm, emitted, reprs,
            ),
        ),
        ibm_step(
            old_ibm,
            crate::proof::engine::abstract_step::architecture_stepped_ibm(
                old_e, new_e, old_ibm, emitted, reprs,
            ),
            emitted.dom(),
            samples,
        ),
{
    crate::proof::cache::semantics::derive_architecture_ibm_cache_extension(
        old_e, new_e, old_ibm, emitted, samples, reprs,
    );
    architecture_semantic_refinement_step_from_cache_extension(
        old_e, new_e, old_ibm, emitted, samples, reprs,
    );
}

// Mode-complete persistent one-step refinement for every supported semantic
// model architecture.  Both the IBM cache extension and retained physical
// provenance are derived from common scheduler geometry plus the closed model
// dispatch; no Qwen/Gemma branch appears in this theorem.
#[verifier::spinoff_prover]
pub proof fn architecture_persistent_semantic_refinement_step(
    old_e: Engine,
    new_e: Engine,
    old_ibm: IndependentBatchModel,
    emitted: Map<RequestId, TokenId>,
    samples: Map<RequestId, (SamplerState, TokenId)>,
    reprs: crate::exec::engine::StepReprs,
)
    requires
        crate::boundary::tensor_runtime::paged_attention_numeric_domain(),
        architecture_persistent_semantic_inv(&old_e, &old_ibm),
        phase_aligned(&old_e, &old_ibm),
        crate::exec::engine::eng_cache_shape_ok(&old_e),
        mach_cache_shape_ok(&old_ibm, old_e.cs.num_blocks as nat),
        cache_scheduler::tail_write_exclusive(&old_e.cs),
        cache_scheduler::residency_history_aligned(&old_e.cs),
        cache_scheduler::residency_history_aligned(&new_e.cs),
        old_e.model_config.num_layers > 0,
        old_e.cs.num_blocks <= u64::MAX,
        crate::exec::engine::architecture_engine_step_relation(
            old_e, new_e, emitted, samples, reprs,
        ),
        crate::exec::engine::engine_step_semantic_identity(old_e, new_e),
    ensures
        architecture_persistent_semantic_inv(
            &new_e,
            &crate::proof::engine::abstract_step::architecture_stepped_ibm(
                old_e, new_e, old_ibm, emitted, reprs,
            ),
        ),
        ibm_step(
            old_ibm,
            crate::proof::engine::abstract_step::architecture_stepped_ibm(
                old_e, new_e, old_ibm, emitted, reprs,
            ),
            emitted.dom(),
            samples,
        ),
{
    let new_ibm = crate::proof::engine::abstract_step::architecture_stepped_ibm(
        old_e, new_e, old_ibm, emitted, reprs,
    );
    reveal(architecture_persistent_semantic_inv);
    assert(architecture_semantic_inv(&old_e, &old_ibm));
    assert(crate::proof::cache::provenance::provenance_cache_fidelity(
        &old_e, ibm_semantic_model(old_ibm),
    ));
    crate::proof::cache::provenance::lemma_provenance_fidelity_implies_registered_fidelity(
        &old_e, ibm_semantic_model(old_ibm),
    );
    crate::proof::cache::semantics::derive_architecture_provenance_cache_fidelity(
        old_e, new_e, old_ibm, emitted, samples, reprs,
    );
    architecture_semantic_refinement_step_from_registered_cache(
        old_e, new_e, old_ibm, emitted, samples, reprs,
    );
    assert(ibm_semantic_model(new_ibm) == ibm_semantic_model(old_ibm));
}

// Sampler-free observable theorem for every supported architecture.  Each
// emitting packed row is relocated to the common private singleton forward;
// the generic canonical-prefix law then identifies its last row with the cold
// full-history reference.  No family-specific layout theorem is used here.
#[verifier::spinoff_prover]
pub proof fn lemma_architecture_step_logits_match_reference_with_ibm(
    old_e: Engine,
    new_e: Engine,
    old_ibm: IndependentBatchModel,
    emitted: Map<RequestId, TokenId>,
    samples: Map<RequestId, (SamplerState, TokenId)>,
    reprs: crate::exec::engine::StepReprs,
)
    requires
        RT::paged_attention_numeric_domain(),
        architecture_persistent_semantic_inv(&old_e, &old_ibm),
        phase_aligned(&old_e, &old_ibm),
        eng_cache_shape_ok(&old_e),
        mach_cache_shape_ok(&old_ibm, old_e.cs.num_blocks as nat),
        old_e.model_config.num_layers > 0,
        old_e.cs.num_blocks <= u64::MAX,
        crate::exec::engine::architecture_engine_step_relation(
            old_e, new_e, emitted, samples, reprs,
        ),
    ensures
        architecture_step_logits_match_reference(old_e, reprs),
{
    reveal(architecture_persistent_semantic_inv);
    assert(architecture_semantic_inv(&old_e, &old_ibm));
    assert(inv(&old_e, &old_ibm));
    lemma_ibm_valid_semantic_model(old_ibm);
    let model = ibm_semantic_model(old_ibm);
    assert(crate::exec::engine::step_semantic_model(old_e, reprs) == model);
    crate::proof::cache::provenance::lemma_provenance_fidelity_implies_registered_fidelity(
        &old_e, model,
    );
    reveal(crate::exec::engine::architecture_engine_step_relation);
    assert(cache_scheduler::waiting_unstarted(&old_e.cs));
    reveal(architecture_step_logits_match_reference);
    assert forall|i: int| #![trigger reprs.scheduled[i], reprs.sample_mask[i]]
        0 <= i < reprs.scheduled.len() && reprs.sample_mask[i] implies {
            let rid = reprs.scheduled[i];
            let tokens = token_seq_to_int(
                history(old_e.cs.live_requests@[rid]),
            );
            &&& old_e.cs.live_requests@.contains_key(rid)
            &&& RT::select_sample_logits_repr(
                crate::exec::engine::architecture_step_logits_repr(old_e, reprs),
                reprs.cu_q,
                i as nat,
            ) == crate::proof::model::architecture::reference_logits_last_row(
                crate::exec::engine::step_semantic_model(old_e, reprs), tokens,
            )
        }
    by {
        let rid = reprs.scheduled[i];
        assert(reprs.scheduled.contains(rid));
        assert(old_e.cs.running@.contains(rid)
            || old_e.cs.waiting@.contains(rid));
        assert(cache_scheduler::live_covers_queue(&old_e.cs));
        assert(old_e.cs.live_requests@.contains_key(rid));
        let row = crate::proof::engine::abstract_step::architecture_scheduled_index_of(
            reprs, rid,
        );
        assert(row == i) by {
            assert(reprs.scheduled.no_duplicates());
        }
        crate::proof::engine::abstract_step::lemma_architecture_emitting_processed_tokens_are_history(
            old_e, reprs, i,
        );
        let tokens = crate::proof::engine::abstract_step::architecture_machine_processed_tokens(
            old_e, reprs, rid,
        );
        let prefix_len = crate::proof::engine::abstract_step::architecture_machine_prefix_len(
            reprs, rid,
        );
        let q_len = crate::proof::engine::abstract_step::architecture_machine_query_len(
            reprs, rid,
        );
        let k_len = crate::proof::engine::abstract_step::architecture_machine_key_len(
            reprs, rid,
        );
        let base = crate::proof::engine::abstract_step::architecture_machine_cache_base(
            old_e, old_ibm, reprs, rid,
        );
        let private_logits =
            crate::proof::engine::abstract_step::architecture_machine_logits_after(
                old_e, old_ibm, reprs, rid,
            );
        let cold_logits = MODEL_CACHE::cold_reference_logits_repr(
            model, tokens,
        );
        let engine_logits = crate::exec::engine::architecture_step_logits_repr(
            old_e, reprs,
        );
        let lo = reprs.cu_q[i];
        let hi = reprs.cu_q[i + 1];

        crate::proof::cache::semantics::derive_architecture_machine_canonical_prefix_forward_ready(
            old_e, old_ibm, reprs, rid,
        );
        MODEL_CACHE::lemma_canonical_prefix_forward_cache_fidelity(
            model, tokens, prefix_len, base,
        );
        assert(prefix_len < tokens.len());
        let continuation_logits = MODEL_CACHE::prefix_continuation_logits_repr(
            model, tokens, prefix_len, base,
        );
        crate::proof::engine::abstract_step::lemma_architecture_machine_forward_is_prefix_continuation(
            old_e, old_ibm, reprs, rid,
        );
        assert(private_logits == continuation_logits);

        crate::proof::cache::coherence::derive_architecture_request_projection_common_domain(
            old_e, old_ibm, reprs, rid,
        );
        crate::proof::model::architecture::lemma_model_forward_request_projection_domain_from_common(
            reprs.wr,
            old_ibm.architecture_repr,
            reprs.input_ids,
            reprs.positions,
            old_e.kv_caches_repr@,
            reprs.slots,
            reprs.cu_q,
            reprs.cu_k,
            reprs.max_q,
            reprs.max_k,
            reprs.bt,
            i as nat,
        );
        crate::proof::cache::coherence::derive_architecture_forward_relocation_ready_from_projection(
            old_e, old_ibm, reprs, rid,
        );
        crate::proof::engine::abstract_step::lemma_architecture_machine_forward_matches_engine(
            old_e, old_ibm, reprs, rid,
        );
        crate::proof::model::architecture::lemma_model_forward_logits_repr_shape(
            reprs.wr,
            old_ibm.architecture_repr,
            reprs.input_ids,
            reprs.positions,
            old_e.kv_caches_repr@,
            reprs.slots,
            reprs.cu_q,
            reprs.cu_k,
            reprs.max_q,
            reprs.max_k,
            reprs.bt,
        );
        crate::proof::tensor::geometry::lemma_cu_int_bounds(
            reprs.cu_q, reprs.scheduled.len() as int,
        );
        assert(0 <= lo < hi <= engine_logits.len() as int);
        assert(engine_logits.subrange(lo, hi) == private_logits);

        assert(MODEL_CACHE::prefix_continuation_matches_cold(
            model, tokens, prefix_len, base,
        ));
        reveal(MODEL_CACHE::prefix_continuation_matches_cold);
        assert(tokens.len() == k_len);
        assert(q_len == (hi - lo) as nat);
        assert(q_len as int == hi - lo);
        assert(prefix_len + q_len == k_len);
        assert(q_len > 0);
        assert(cold_logits.subrange(prefix_len as int, k_len as int)
            == continuation_logits);
        assert(engine_logits.subrange(lo, hi)[q_len as int - 1]
            == private_logits[q_len as int - 1]);
        assert(private_logits[q_len as int - 1]
            == continuation_logits[q_len as int - 1]);
        assert(continuation_logits[q_len as int - 1]
            == cold_logits[k_len as int - 1]);
        assert(engine_logits.subrange(lo, hi)[q_len as int - 1]
            == engine_logits[lo + q_len as int - 1]);
        assert(lo + q_len as int - 1 == hi - 1);
        assert(engine_logits[hi - 1]
            == engine_logits.subrange(lo, hi)[q_len as int - 1]);
        crate::proof::model::architecture::lemma_reference_logits_last_row_is_forward_last(
            model, tokens,
        );
        reveal(MODEL_CACHE::cold_reference_logits_repr);
        reveal(RT::select_sample_logits_repr);
    }
}

// Sampling corollary of the architecture-neutral logits theorem.
pub proof fn lemma_architecture_step_samples_match_reference(
    old_e: Engine,
    new_e: Engine,
    old_ibm: IndependentBatchModel,
    emitted: Map<RequestId, TokenId>,
    samples: Map<RequestId, (SamplerState, TokenId)>,
    reprs: crate::exec::engine::StepReprs,
)
    requires
        RT::paged_attention_numeric_domain(),
        architecture_persistent_semantic_inv(&old_e, &old_ibm),
        phase_aligned(&old_e, &old_ibm),
        eng_cache_shape_ok(&old_e),
        mach_cache_shape_ok(&old_ibm, old_e.cs.num_blocks as nat),
        old_e.model_config.num_layers > 0,
        old_e.cs.num_blocks <= u64::MAX,
        crate::exec::engine::architecture_engine_step_relation(
            old_e, new_e, emitted, samples, reprs,
        ),
    ensures
        architecture_step_logits_match_reference(old_e, reprs),
        step_samples_match_reference(old_ibm, samples, reprs),
{
    lemma_architecture_step_logits_match_reference_with_ibm(
        old_e, new_e, old_ibm, emitted, samples, reprs,
    );
    reveal(crate::exec::engine::architecture_engine_step_relation);
    assert(crate::exec::engine::engine_samples_match_logits(
        old_e,
        samples,
        reprs,
        crate::exec::engine::architecture_step_logits_repr(old_e, reprs),
    ));
    assert forall|i: int| #![trigger reprs.scheduled[i], reprs.sample_mask[i]]
        0 <= i < reprs.scheduled.len() && reprs.sample_mask[i] implies {
            let rid = reprs.scheduled[i];
            let state = old_ibm.machines[rid].request_state;
            let tokens = token_seq_to_int(history(state));
            &&& old_ibm.machines.contains_key(rid)
            &&& samples.contains_key(rid)
            &&& samples[rid] == RT::sample_from_repr(
                crate::proof::model::architecture::reference_logits_last_row(
                    ibm_semantic_model(old_ibm), tokens,
                ),
                state.sampler_state,
            )
        }
    by {
        let rid = reprs.scheduled[i];
        assert(reprs.scheduled.contains(rid));
        assert(old_e.cs.running@.contains(rid)
            || old_e.cs.waiting@.contains(rid));
        assert(cache_scheduler::live_covers_queue(&old_e.cs));
        assert(old_e.cs.live_requests@.contains_key(rid));
        assert(old_ibm.machines.contains_key(rid));
        let state = old_ibm.machines[rid].request_state;
        let tokens = token_seq_to_int(history(state));
        assert(request_state_view_eq(
            old_e.cs.live_requests@[rid], state,
        ));
        assert(history(old_e.cs.live_requests@[rid]) == history(state));
        assert(samples.contains_key(rid));
        assert(samples[rid] == RT::sample_from_repr(
            RT::select_sample_logits_repr(
                crate::exec::engine::architecture_step_logits_repr(old_e, reprs),
                reprs.cu_q,
                i as nat,
            ),
            old_e.cs.live_requests@[rid].sampler_state,
        ));
        assert(crate::exec::engine::step_semantic_model(old_e, reprs)
            == ibm_semantic_model(old_ibm));
    }
}

// Observable architecture-native one-step capstone: semantic/provenance state,
// abstract request evolution, reference logits, and emitted samples are proved
// together for the concrete architecture-dispatched engine relation.
#[verifier::spinoff_prover]
pub proof fn architecture_persistent_semantic_refinement_observable_step(
    old_e: Engine,
    new_e: Engine,
    old_ibm: IndependentBatchModel,
    emitted: Map<RequestId, TokenId>,
    samples: Map<RequestId, (SamplerState, TokenId)>,
    reprs: crate::exec::engine::StepReprs,
)
    requires
        RT::paged_attention_numeric_domain(),
        architecture_persistent_semantic_inv(&old_e, &old_ibm),
        phase_aligned(&old_e, &old_ibm),
        eng_cache_shape_ok(&old_e),
        mach_cache_shape_ok(&old_ibm, old_e.cs.num_blocks as nat),
        cache_scheduler::tail_write_exclusive(&old_e.cs),
        cache_scheduler::residency_history_aligned(&old_e.cs),
        cache_scheduler::residency_history_aligned(&new_e.cs),
        old_e.model_config.num_layers > 0,
        old_e.cs.num_blocks <= u64::MAX,
        crate::exec::engine::architecture_engine_step_relation(
            old_e, new_e, emitted, samples, reprs,
        ),
        crate::exec::engine::engine_step_semantic_identity(old_e, new_e),
    ensures
        architecture_step_logits_match_reference(old_e, reprs),
        architecture_persistent_semantic_inv(
            &new_e,
            &crate::proof::engine::abstract_step::architecture_stepped_ibm(
                old_e, new_e, old_ibm, emitted, reprs,
            ),
        ),
        ibm_step(
            old_ibm,
            crate::proof::engine::abstract_step::architecture_stepped_ibm(
                old_e, new_e, old_ibm, emitted, reprs,
            ),
            emitted.dom(),
            samples,
        ),
        observable_step_agreement(old_ibm, emitted, samples, reprs),
{
    reveal(architecture_persistent_semantic_inv);
    assert(architecture_semantic_inv(&old_e, &old_ibm));
    reveal(architecture_semantic_inv);
    lemma_ibm_valid_semantic_model(old_ibm);
    crate::proof::engine::abstract_step::construct_architecture_abstract_step(
        old_e, new_e, old_ibm, emitted, samples, reprs,
    );
    lemma_architecture_step_samples_match_reference(
        old_e, new_e, old_ibm, emitted, samples, reprs,
    );
    architecture_persistent_semantic_refinement_step(
        old_e, new_e, old_ibm, emitted, samples, reprs,
    );
    reveal(crate::exec::engine::architecture_engine_step_relation);
    assert forall|rid: RequestId| #[trigger] emitted.contains_key(rid)
        implies crate::exec::engine::reprs_emits(reprs, rid) by {
        assert(old_e.cs.live_requests@.contains_key(rid));
    }
    assert forall|rid: RequestId|
        #[trigger] crate::exec::engine::reprs_emits(reprs, rid)
        implies emitted.contains_key(rid) by {
        let i = choose|i: int| 0 <= i < reprs.scheduled.len()
            && reprs.scheduled[i] == rid
            && reprs.sample_mask[i];
        assert(reprs.scheduled.contains(rid));
        assert(old_e.cs.running@.contains(rid)
            || old_e.cs.waiting@.contains(rid));
        assert(cache_scheduler::live_covers_queue(&old_e.cs));
        assert(old_e.cs.live_requests@.contains_key(rid));
    }
    assert forall|i: int| #![trigger reprs.scheduled[i], reprs.sample_mask[i]]
        0 <= i < reprs.scheduled.len() && reprs.sample_mask[i] implies {
            let rid = reprs.scheduled[i];
            &&& emitted.contains_key(rid)
            &&& emitted[rid] == samples[rid].1
        }
    by {
        let rid = reprs.scheduled[i];
        assert(reprs.scheduled.contains(rid));
        assert(old_e.cs.running@.contains(rid)
            || old_e.cs.waiting@.contains(rid));
        assert(cache_scheduler::live_covers_queue(&old_e.cs));
        assert(old_e.cs.live_requests@.contains_key(rid));
        assert(emitted.contains_key(rid));
    }
}

// Actual request initialization: run `Engine::init` and construct the matching
// IBM directly from its request-state relation.  The result satisfies the
// complete stable semantic bundle, including inductive scheduler readiness.
pub fn architecture_persistent_semantic_refinement_init_exec(
    config: cache_scheduler::SchedulerConfig,
    num_blocks: u64,
    kv_caches: Vec<(RT::Tensor, RT::Tensor)>,
    kv_perms: Tracked<RT::KVCachePerms>,
    weights: RT::ModelWeights,
    runtime: RT::ModelRuntime,
    weights_perms: Tracked<RT::ModelWeightsPerms>,
    model_config: ModelConfig,
    requests: Vec<RequestState>,
) -> (out: (Engine, Ghost<IndependentBatchModel>))
    requires
        obeys_key_model::<u64>(),
        num_blocks <= u64::MAX / crate::types::BLOCK_SIZE,
        kv_caches.len() == model_config.num_layers,
        RT::model_weights_num_layers(&weights) == model_config.num_layers,
        RT::model_execution_valid(&weights, &runtime, &weights_perms@),
        RT::model_weights_repr_of(&weights_perms@).architecture
            == model_config.architecture,
        kv_perms@.len() == model_config.num_layers as nat,
        RT::kv_perms_ids_distinct(kv_perms@),
        kv_perms@.extracted() == Set::<int>::empty(),
        RT::kv_cache_tensor_ids_match(
            kv_caches@, kv_perms@, model_config.num_layers as nat,
        ),
        RT::kv_perms_page_shape(kv_perms@, num_blocks as nat),
        model_config.num_layers > 0,
        initial_request_batch_ready(requests@),
    ensures
        architecture_persistent_semantic_runtime_inv(&out.0, &out.1@),
        cache_scheduler::prefill_plan_ready(&out.0.cs),
        out.0.cs.running@.len() == 0,
        out.0.cs.running@.len() > 0
            ==> cache_scheduler::decode_plan_ready(&out.0.cs),
        crate::exec::engine::engine_init_request_relation(
            &out.0, config, num_blocks, requests@,
        ),
        forall|k: int| #![trigger requests@[k]]
            0 <= k < requests@.len() ==> {
            let rid = requests@[k].request_id;
            &&& out.1@.machines.contains_key(rid)
            &&& request_state_view_eq(
                out.1@.machines[rid].request_state, requests@[k],
            )
        },
{
    let ghost request_states = requests@;
    let engine = Engine::init(
        config,
        num_blocks,
        kv_caches,
        kv_perms,
        weights,
        runtime,
        weights_perms,
        model_config,
        requests,
    );
    let ghost ibm = initialized_ibm(&engine);
    proof {
        reveal(crate::exec::engine::engine_init_request_relation);
        assert forall|rid: RequestId|
            #[trigger] engine.cs.live_requests@.contains_key(rid) implies {
                &&& valid_request_state(engine.cs.live_requests@[rid])
                &&& engine.cs.live_requests@[rid].request_id == rid
            }
        by {
            let k = choose|k: int| 0 <= k < request_states.len()
                && request_states[k].request_id == rid;
            assert(request_state_view_eq(
                engine.cs.live_requests@[rid], request_states[k],
            ));
            lemma_request_lifecycle_view_eq_fields(
                engine.cs.live_requests@[rid], request_states[k],
            );
            assert(can_step(request_states[k]));
        }
        architecture_persistent_semantic_refinement_initialized(&engine);
        cache_scheduler::lemma_live_request_step_ready_implies_plan_ready(
            &engine.cs,
        );
        assert forall|k: int| #![trigger request_states[k]]
            0 <= k < request_states.len() implies {
            let rid = request_states[k].request_id;
            &&& ibm.machines.contains_key(rid)
            &&& request_state_view_eq(
                ibm.machines[rid].request_state, request_states[k],
            )
        } by {
            let rid = request_states[k].request_id;
            assert(engine.cs.live_requests@.contains_key(rid));
            assert(ibm.machines[rid]
                == initialized_request_machine(&engine, rid));
            assert(request_state_view_eq(
                engine.cs.live_requests@[rid], request_states[k],
            ));
        }
    }
    (engine, Ghost(ibm))
}

pub proof fn lemma_architecture_phase_aligned_preserved(
    old_e: Engine,
    new_e: Engine,
    old_ibm: IndependentBatchModel,
    emitted: Map<RequestId, TokenId>,
    samples: Map<RequestId, (SamplerState, TokenId)>,
    reprs: crate::exec::engine::StepReprs,
)
    requires
        inv(&old_e, &old_ibm),
        phase_aligned(&old_e, &old_ibm),
        crate::exec::engine::architecture_engine_step_relation(
            old_e, new_e, emitted, samples, reprs,
        ),
    ensures
        phase_aligned(
            &new_e,
            &crate::proof::engine::abstract_step::architecture_stepped_ibm(
                old_e, new_e, old_ibm, emitted, reprs,
            ),
        ),
{
    let new_ibm = crate::proof::engine::abstract_step::architecture_stepped_ibm(
        old_e, new_e, old_ibm, emitted, reprs,
    );
    reveal(crate::exec::engine::architecture_engine_step_relation);
    assert forall|rid: RequestId|
        new_e.cs.live_requests@.contains_key(rid)
            && new_ibm.machines.contains_key(rid)
        implies (new_e.cs.running@.contains(rid)
            <==> #[trigger] new_ibm.machines[rid].kv_initialized)
    by {
        assert(old_e.cs.live_requests@.contains_key(rid));
        assert(old_ibm.machines.contains_key(rid));
        assert(new_ibm.machines[rid]
            == crate::proof::engine::abstract_step::architecture_stepped_machine(
                old_e, new_e, old_ibm, emitted, reprs, rid,
            ));
        if emitted.contains_key(rid) {
            assert(new_ibm.machines[rid].kv_initialized);
            assert(crate::exec::engine::reprs_emits(reprs, rid));
            assert(!(emitted.contains_key(rid)
                && should_finish_after_append(
                    old_e.cs.live_requests@[rid], emitted[rid],
                )));
            assert(new_e.cs.running@.contains(rid));
        } else {
            assert(new_ibm.machines[rid] == old_ibm.machines[rid]);
            assert(!crate::exec::engine::reprs_emits(reprs, rid));
            if reprs.scheduled.contains(rid) {
                let k = choose|k: int| 0 <= k < reprs.scheduled.len()
                    && reprs.scheduled[k] == rid;
                assert(!reprs.sample_mask[k]) by {
                    if reprs.sample_mask[k] {
                        assert(crate::exec::engine::reprs_emits(reprs, rid));
                    }
                }
                crate::exec::engine::lemma_kv_only_row_was_waiting(
                    old_e, reprs, k,
                );
                assert(!old_e.cs.running@.contains(rid)) by {
                    assert(cache_scheduler::queue_disjoint(&old_e.cs));
                }
                assert(crate::exec::engine::reprs_parks(reprs, rid));
                assert(!new_e.cs.running@.contains(rid));
                assert(!old_ibm.machines[rid].kv_initialized);
            } else {
                assert(!crate::exec::engine::reprs_parks(reprs, rid));
                assert(new_e.cs.running@.contains(rid)
                    <==> old_e.cs.running@.contains(rid));
            }
        }
    }
}

// Page geometry of every surviving private machine is preserved by the
// architecture-dispatched singleton forward.  The only family obligation is
// the symmetric `lemma_forward_cache_shape_preserved` adapter contract.
pub proof fn lemma_architecture_mach_cache_shape_preserved(
    old_e: Engine,
    new_e: Engine,
    old_ibm: IndependentBatchModel,
    emitted: Map<RequestId, TokenId>,
    samples: Map<RequestId, (SamplerState, TokenId)>,
    reprs: crate::exec::engine::StepReprs,
)
    requires
        architecture_semantic_inv(&old_e, &old_ibm),
        crate::exec::engine::architecture_engine_step_relation(
            old_e, new_e, emitted, samples, reprs,
        ),
        mach_cache_shape_ok(
            &old_ibm, old_e.cs.num_blocks as nat,
        ),
    ensures
        mach_cache_shape_ok(
            &crate::proof::engine::abstract_step::architecture_stepped_ibm(
                old_e, new_e, old_ibm, emitted, reprs,
            ),
            new_e.cs.num_blocks as nat,
        ),
{
    let num_pages = old_e.cs.num_blocks as nat;
    let new_ibm = crate::proof::engine::abstract_step::architecture_stepped_ibm(
        old_e, new_e, old_ibm, emitted, reprs,
    );
    reveal(architecture_semantic_inv);
    reveal(crate::exec::engine::architecture_engine_step_relation);
    lemma_ibm_valid_semantic_model(old_ibm);
    assert(new_e.cs.num_blocks == old_e.cs.num_blocks);
    assert forall|rid: RequestId, layer: int|
        #![trigger new_ibm.machines[rid].kv_cache_reprs[layer]]
        new_ibm.machines.contains_key(rid)
        && 0 <= layer < new_ibm.machines[rid].kv_cache_reprs.len()
        implies {
            &&& new_ibm.machines[rid].kv_cache_reprs[layer].0.len()
                == num_pages
            &&& new_ibm.machines[rid].kv_cache_reprs[layer].1.len()
                == num_pages
            &&& (forall|page: int| 0 <= page < num_pages ==>
                (#[trigger]
                    new_ibm.machines[rid].kv_cache_reprs[layer].0[page]).len()
                    == crate::types::BLOCK_SIZE_SPEC as int)
            &&& (forall|page: int| 0 <= page < num_pages ==>
                (#[trigger]
                    new_ibm.machines[rid].kv_cache_reprs[layer].1[page]).len()
                    == crate::types::BLOCK_SIZE_SPEC as int)
        }
    by {
        assert(old_ibm.machines.contains_key(rid));
        assert(new_ibm.machines[rid]
            == crate::proof::engine::abstract_step::architecture_stepped_machine(
                old_e, new_e, old_ibm, emitted, reprs, rid,
            ));
        if emitted.contains_key(rid) {
            assert(reprs.scheduled.contains(rid));
            let row = crate::proof::engine::abstract_step::architecture_scheduled_index_of(
                reprs, rid,
            );
            let q_len = crate::proof::engine::abstract_step::architecture_machine_query_len(
                reprs, rid,
            );
            let k_len = crate::proof::engine::abstract_step::architecture_machine_key_len(
                reprs, rid,
            );
            let prefix_len =
                crate::proof::engine::abstract_step::architecture_machine_prefix_len(
                    reprs, rid,
                );
            let base = crate::proof::engine::abstract_step::architecture_machine_cache_base(
                old_e, old_ibm, reprs, rid,
            );
            assert(base.len() == old_ibm.model_config.num_layers as nat);
            assert(MODEL_CACHE::cache_sequence_page_shape(
                base, num_pages,
            )) by {
                assert forall|ell: int| 0 <= ell < base.len() implies {
                    &&& (#[trigger] base[ell]).0.len() == num_pages
                    &&& base[ell].1.len() == num_pages
                    &&& (forall|page: int| 0 <= page < num_pages ==>
                        (#[trigger] base[ell].0[page]).len()
                            == crate::types::BLOCK_SIZE_SPEC as int)
                    &&& (forall|page: int| 0 <= page < num_pages ==>
                        (#[trigger] base[ell].1[page]).len()
                            == crate::types::BLOCK_SIZE_SPEC as int)
                } by {
                    let old_cache = old_ibm.machines[rid]
                        .kv_cache_reprs[ell];
                    let engine_cache = old_e.kv_caches_repr@[ell];
                    let bt = reprs.bt[row];
                    crate::proof::model::family_layout::lemma_relocated_prefix_base_shape(
                        old_cache.0, engine_cache.0, bt, prefix_len,
                    );
                    crate::proof::model::family_layout::lemma_relocated_prefix_base_shape(
                        old_cache.1, engine_cache.1, bt, prefix_len,
                    );
                    assert(mach_cache_shape_ok(
                        &old_ibm, num_pages,
                    ));
                    reveal(crate::proof::engine::abstract_step::architecture_machine_cache_base);
                }
            }
            assert(reprs.scheduled.contains(rid));
            crate::proof::engine::abstract_step::lemma_architecture_machine_forward_is_prefix_continuation(
                old_e, old_ibm, reprs, rid,
            );
            crate::proof::tensor::geometry::lemma_cu_int_bounds(
                reprs.cu_q, reprs.scheduled.len() as int,
            );
            assert(0 <= row < reprs.scheduled.len());
            assert(0 <= reprs.cu_q[row]
                < reprs.cu_q[row + 1]
                <= reprs.input_ids.len() as int);
            assert(reprs.input_ids.subrange(
                reprs.cu_q[row], reprs.cu_q[row + 1],
            ).len() == q_len);
            assert(reprs.positions.subrange(
                reprs.cu_q[row], reprs.cu_q[row + 1],
            ).len() == q_len);
            crate::proof::model::architecture::lemma_model_forward_cache_shape_preserved(
                old_ibm.wr,
                old_ibm.architecture_repr,
                reprs.input_ids.subrange(
                    reprs.cu_q[row], reprs.cu_q[row + 1],
                ),
                reprs.positions.subrange(
                    reprs.cu_q[row], reprs.cu_q[row + 1],
                ),
                base,
                crate::proof::reference::request_machine::slots_from(prefix_len, q_len),
                crate::proof::reference::request_machine::seq_lens_for_single(q_len),
                crate::proof::reference::request_machine::seq_lens_for_single(k_len),
                q_len,
                k_len,
                crate::proof::reference::request_machine::singleton_block_rows(k_len),
                num_pages,
            );
            assert(MODEL_CACHE::cache_sequence_page_shape(
                crate::proof::engine::abstract_step::architecture_machine_cache_after(
                    old_e, old_ibm, reprs, rid,
                ),
                num_pages,
            )) by {
                reveal(crate::proof::engine::abstract_step::architecture_machine_cache_after);
            }
            assert(new_ibm.machines[rid].kv_cache_reprs
                == crate::proof::engine::abstract_step::architecture_machine_cache_after(
                    old_e, old_ibm, reprs, rid,
                ));
        } else {
            assert(new_ibm.machines[rid] == old_ibm.machines[rid]);
            assert(mach_cache_shape_ok(&old_ibm, num_pages));
        }
    }
}

// Strongest executable one-step entry point. It carries the persistent semantic
// invariant and returns the
// concrete emissions together with the ghost sample/plan witnesses needed to
// state the observable result: every emitted token is exactly the sample of
// an independent cold full-history reference execution.
//
// @kernel-bridge-begin proof::engine::refinement::architecture_persistent_semantic_refinement_observable_step_exec
pub fn architecture_persistent_semantic_refinement_observable_step_exec(
    engine: &mut Engine,
    Ghost(ibm): Ghost<IndependentBatchModel>,
    graph_overlay: Option<&crate::boundary::tensor_runtime::CudaGraphOverlay>,
) -> (out: (
    cache_scheduler::EmittedTokens,
    Ghost<IndependentBatchModel>,
    Ghost<Map<RequestId, (SamplerState, TokenId)>>,
    Ghost<crate::exec::engine::StepReprs>,
))
    requires
        graph_overlay.is_some()
            ==> crate::boundary::tensor_runtime::cuda_graph_replay_fidelity(),
        graph_overlay.is_some()
            ==> crate::exec::model_families::cuda_graph_overlay_supported(
                &old(engine).weights_perms@,
            ),
        obeys_key_model::<u64>(),
        RT::paged_attention_numeric_domain(),
        architecture_persistent_semantic_runtime_inv(old(engine), &ibm),
    ensures
        architecture_step_logits_match_reference(*old(engine), out.3@),
        architecture_persistent_semantic_runtime_inv(
            final(engine), &out.1@,
        ),
        crate::exec::engine::architecture_engine_step_relation(
            *old(engine), *final(engine), out.0@, out.2@, out.3@,
        ),
        ibm_step(ibm, out.1@, out.0@.dom(), out.2@),
        observable_step_agreement(ibm, out.0@, out.2@, out.3@),
{
    let ghost old_e = *engine;
    proof {
        reveal(architecture_persistent_semantic_runtime_inv);
        cache_scheduler::lemma_live_request_step_ready_implies_plan_ready(
            &old_e.cs,
        );
    }
    let (emitted, samples, reprs) = engine.step(graph_overlay);
    let ghost new_ibm = crate::proof::engine::abstract_step::architecture_stepped_ibm(
        old_e, *engine, ibm, emitted@, reprs@,
    );
    proof {
        architecture_persistent_semantic_refinement_observable_step(
            old_e, *engine, ibm, emitted@, samples@, reprs@,
        );
        lemma_architecture_phase_aligned_preserved(
            old_e, *engine, ibm, emitted@, samples@, reprs@,
        );
        lemma_architecture_mach_cache_shape_preserved(
            old_e, *engine, ibm, emitted@, samples@, reprs@,
        );
        crate::exec::engine::lemma_architecture_live_request_step_ready_preserved(
            old_e, *engine, emitted@, samples@, reprs@,
        );
        assert(architecture_persistent_semantic_runtime_inv(
            engine, &new_ibm,
        ));
    }
    (emitted, Ghost(new_ibm), samples, reprs)
}
// @kernel-bridge-end proof::engine::refinement::architecture_persistent_semantic_refinement_observable_step_exec

// Execute exactly `step_count` engine steps and retain a ghost trace of the
// actual emissions, sample maps, plan representations, and chained IBM states.
// Because the architecture-neutral runtime invariant is inductive, the recursive call
// needs no new scheduler-readiness assumptions.
#[verifier::spinoff_prover]
#[verifier::rlimit(100)]
pub fn persistent_semantic_refinement_trace_exec(
    engine: &mut Engine,
    Ghost(ibm): Ghost<IndependentBatchModel>,
    step_count: usize,
) -> (out: Ghost<ObservableServingTrace>)
    requires
        obeys_key_model::<u64>(),
        crate::boundary::tensor_runtime::paged_attention_numeric_domain(),
        architecture_persistent_semantic_runtime_inv(old(engine), &ibm),
    ensures
        architecture_persistent_semantic_runtime_inv(
            final(engine), &out@.final_ibm,
        ),
        observable_serving_trace(out@, step_count as nat),
        out@.initial_engine == *old(engine),
        out@.final_engine == *final(engine),
        out@.initial_ibm == ibm,
    decreases step_count,
{
    if step_count == 0 {
        let ghost trace = ObservableServingTrace {
            initial_engine: *engine,
            final_engine: *engine,
            initial_ibm: ibm,
            final_ibm: ibm,
            steps: Seq::empty(),
        };
        proof {
            assert(observable_trace_chain(trace.steps));
            assert(observable_serving_trace(trace, 0));
        }
        Ghost(trace)
    } else {
        let ghost pre_engine = *engine;
        let ghost pre_ibm = ibm;
        let (emitted, Ghost(post_ibm), samples, reprs) =
            architecture_persistent_semantic_refinement_observable_step_exec(
                engine, Ghost(ibm), None,
            );
        let ghost first = ObservableTraceStep {
            pre_engine,
            post_engine: *engine,
            pre_ibm,
            post_ibm,
            emitted: emitted@,
            samples: samples@,
            reprs: reprs@,
        };
        proof {
            lemma_observable_trace_step_agreement_intro(first);
        }
        let Ghost(tail) = persistent_semantic_refinement_trace_exec(
            engine, Ghost(post_ibm), step_count - 1,
        );
        let ghost trace = prepend_observable_trace(first, tail);
        proof {
            assert(tail.initial_engine == first.post_engine);
            assert(tail.initial_ibm == post_ibm);
            assert(tail.steps.len() == (step_count - 1) as nat);
            lemma_prepend_observable_trace(
                first, tail, (step_count - 1) as nat,
            );
            assert(observable_serving_trace(trace, step_count as nat));
        }
        Ghost(trace)
    }
}

// End-to-end finite safety/refinement driver: initialize the real engine from
// caller requests, construct the matching independent machines, and execute a
// concrete trace of exactly `step_count` serving steps.  Every trace entry is
// an actual architecture-dispatched engine relation / `ibm_step` pair whose emitted tokens equal
// deterministic cold full-history reference samples.
// @kernel-bridge-begin proof::engine::refinement::persistent_semantic_refinement_init_trace_exec
pub fn persistent_semantic_refinement_init_trace_exec(
    config: cache_scheduler::SchedulerConfig,
    num_blocks: u64,
    kv_caches: Vec<(RT::Tensor, RT::Tensor)>,
    kv_perms: Tracked<RT::KVCachePerms>,
    weights: RT::ModelWeights,
    runtime: RT::ModelRuntime,
    weights_perms: Tracked<RT::ModelWeightsPerms>,
    model_config: ModelConfig,
    requests: Vec<RequestState>,
    step_count: usize,
) -> (out: (Engine, Ghost<ObservableServingTrace>))
    requires
        obeys_key_model::<u64>(),
        crate::boundary::tensor_runtime::paged_attention_numeric_domain(),
        num_blocks <= u64::MAX / crate::types::BLOCK_SIZE,
        kv_caches.len() == model_config.num_layers,
        RT::model_weights_num_layers(&weights) == model_config.num_layers,
        RT::model_execution_valid(&weights, &runtime, &weights_perms@),
        RT::model_weights_repr_of(&weights_perms@).architecture
            == model_config.architecture,
        kv_perms@.len() == model_config.num_layers as nat,
        RT::kv_perms_ids_distinct(kv_perms@),
        kv_perms@.extracted() == Set::<int>::empty(),
        RT::kv_cache_tensor_ids_match(
            kv_caches@, kv_perms@, model_config.num_layers as nat,
        ),
        RT::kv_perms_page_shape(kv_perms@, num_blocks as nat),
        model_config.num_layers > 0,
        initial_request_batch_ready(requests@),
    ensures
        architecture_persistent_semantic_runtime_inv(
            &out.0, &out.1@.final_ibm,
        ),
        observable_serving_trace(out.1@, step_count as nat),
        out.1@.final_engine == out.0,
        crate::exec::engine::engine_init_request_relation(
            &out.1@.initial_engine, config, num_blocks, requests@,
        ),
        forall|k: int| #![trigger requests@[k]]
            0 <= k < requests@.len() ==> {
            let rid = requests@[k].request_id;
            &&& out.1@.initial_ibm.machines.contains_key(rid)
            &&& request_state_view_eq(
                out.1@.initial_ibm.machines[rid].request_state,
                requests@[k],
            )
        },
{
    let ghost request_states = requests@;
    let (mut engine, Ghost(initial_ibm)) =
        architecture_persistent_semantic_refinement_init_exec(
            config, num_blocks, kv_caches, kv_perms,
            weights, runtime, weights_perms, model_config, requests,
        );
    let ghost initial_engine = engine;
    let Ghost(trace) = persistent_semantic_refinement_trace_exec(
        &mut engine, Ghost(initial_ibm), step_count,
    );
    proof {
        assert(trace.initial_engine == initial_engine);
        assert(trace.initial_ibm == initial_ibm);
        assert forall|k: int| #![trigger request_states[k]]
            0 <= k < request_states.len() implies {
                let rid = request_states[k].request_id;
                &&& trace.initial_ibm.machines.contains_key(rid)
                &&& request_state_view_eq(
                    trace.initial_ibm.machines[rid].request_state,
                    request_states[k],
                )
            }
        by {
        }
    }
    (engine, Ghost(trace))
}
// @kernel-bridge-end proof::engine::refinement::persistent_semantic_refinement_init_trace_exec

} // verus!
