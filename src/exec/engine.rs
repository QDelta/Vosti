// Concrete top-level inference engine, derived from the earlier Dafny engine.
//
// Executable port: `step()` follows the Dafny engine shape
// Plan -> ModelForward -> SampleAll -> Commit.  The step body and request-side
// initialization are verified. Runtime tensors enter through explicit tracked
// permission bundles minted by their constructors; numerical kernel behavior
// remains behind the `tensor_runtime` external-body contracts.

use crate::model_config::ModelConfig;
use crate::exec::cache_scheduler::*;
use crate::proof::engine::architecture as EA;
use crate::exec::model::model_forward;
use crate::exec::model_families as MODEL_FAMILIES;
use crate::exec::model_families::model_forward_cuda_graph_overlay;
use crate::proof::model::architecture as MA;
use crate::exec::request_state::*;
use crate::{types::*, proof::model::types::*, proof::tensor::types::*};
use crate::boundary::tensor_runtime as RT;
use vstd::hash_map::HashMapWithView;
use vstd::prelude::*;
#[cfg(verus_only)]
use vstd::std_specs::hash::obeys_key_model;

verus! {

pub struct Engine {
    pub cs: CacheScheduler,
    pub kv_caches: Vec<(RT::Tensor, RT::Tensor)>,
    pub weights: RT::ModelWeights,
    // Persistent executable capability paired with `weights`. Every model
    // family enters through the same qualified runtime interface.
    pub runtime: RT::ModelRuntime,
    pub model_config: ModelConfig,
    // Ghost projection of the per-layer paged cache contents (K, V) the runtime
    // `kv_caches` tensors hold.  Persistent ghost state so the refinement
    // invariant (`engine_kv_coherent`) can talk about cache *contents* without a
    // borrowed `KVCachePerms`.  Maintained by `step`'s store/forward.
    pub kv_caches_repr: Ghost<Seq<(KVCacheLayerRepr, KVCacheLayerRepr)>>,
    // The engine owns weight and KV-cache permissions across steps. `init`
    // mints them once under the allocator trust boundary.
    // `eng_execution_perms_ok` ties `kv_caches_repr` to the owned permissions'
    // representations, preserving cache identity and contents across steps.
    pub weights_perms: Tracked<RT::ModelWeightsPerms>,
    pub kv_perms: Tracked<RT::KVCachePerms>,
    // Output-only telemetry. Neither scheduling nor semantic invariants read it.
    // Refreshed before commit can discard a chunk's/request's residency.
    pub last_step_prefix_reuse: Vec<crate::exec::step_observation::PlannedPrefixReuse>,
}

// Architecture-dispatched permission-ownership invariant: the owned runtime,
// weights, and permissions agree; the runtime tensors have the shape
// `model_forward` requires; and the persistent ghost cache `kv_caches_repr` is
// exactly the owned cache permissions' per-layer representations.  This is the
// execution layer shared by Qwen and qualified Gemma.
pub open spec fn eng_execution_perms_ok(e: &Engine) -> bool {
    RT::model_execution_valid(&e.weights, &e.runtime, &e.weights_perms@)
    && RT::model_weights_repr_of(&e.weights_perms@).architecture
        == e.model_config.architecture
    && RT::model_weights_num_layers(&e.weights) == e.model_config.num_layers as nat
    && e.kv_caches@.len() == e.model_config.num_layers as nat
    && e.kv_perms@.len() == e.model_config.num_layers as nat
    && RT::kv_perms_ids_distinct(e.kv_perms@)
    && e.kv_perms@.extracted() == Set::<int>::empty()
    && (forall|i: int| 0 <= i < e.model_config.num_layers as int ==>
        #[trigger] e.kv_caches@[i].0.id() == e.kv_perms@.k_id(i))
    && (forall|i: int| 0 <= i < e.model_config.num_layers as int ==>
        #[trigger] e.kv_caches@[i].1.id() == e.kv_perms@.v_id(i))
    && e.kv_caches_repr@.len() == e.model_config.num_layers as nat
    && (forall|j: int| 0 <= j < e.model_config.num_layers as int ==>
        #[trigger] e.kv_caches_repr@[j]
            == (e.kv_perms@.k_repr(j), e.kv_perms@.v_repr(j)))
}

// The engine's ghost caches have `cs.num_blocks` pages of BLOCK_SIZE rows per
// layer and side, as required by scheduler slot arithmetic. The trusted
// permissions minted at initialization establish this shape. The per-layer
// store characterization proves that scatter stores preserve it at every step.
// Engine-side `slot_in_cache` obligations then follow from `cs_valid`'s
// block-range facts without a trusted plan-geometry statement.
pub open spec fn eng_cache_shape_ok(e: &Engine) -> bool {
    forall|j: int| 0 <= j < e.kv_caches_repr@.len() ==> {
        &&& (#[trigger] e.kv_caches_repr@[j]).0.len() == e.cs.num_blocks as nat
        &&& e.kv_caches_repr@[j].1.len() == e.cs.num_blocks as nat
        &&& (forall|p: int| 0 <= p < e.kv_caches_repr@[j].0.len()
            ==> (#[trigger] e.kv_caches_repr@[j].0[p]).len()
                == crate::types::BLOCK_SIZE_SPEC as int)
        &&& (forall|p: int| 0 <= p < e.kv_caches_repr@[j].1.len()
            ==> (#[trigger] e.kv_caches_repr@[j].1[p]).len()
                == crate::types::BLOCK_SIZE_SPEC as int)
    }
}

// Exact scheduler/request state established by `Engine::init`.  Kept opaque at
// unrelated call sites: spelling these quantifiers directly in the external
// init contract polluted solver-heavy forward queries in this module.
#[verifier::opaque]
pub open spec fn engine_init_request_relation(
    e: &Engine,
    config: SchedulerConfig,
    num_blocks: u64,
    requests: Seq<RequestState>,
) -> bool {
    e.cs.config == config
    && e.cs.num_blocks == num_blocks
    && e.cs.free_blocks == num_blocks
    && e.cs.running@.len() == 0
    && e.cs.waiting@.len() == requests.len()
    && e.cs.request_residency@.dom().is_empty()
    && e.cs.blocks@.dom().is_empty()
    && e.cs.hash_to_block@.dom().is_empty()
    && e.cs.accepted_requests@.dom() == e.cs.live_requests@.dom()
    && (forall|rid: RequestId|
        #[trigger] e.cs.accepted_requests@.contains_key(rid)
        ==> e.cs.accepted_requests@[rid])
    && (forall|k: int| 0 <= k < requests.len() ==>
        #[trigger] e.cs.waiting@[k] == requests[k].request_id)
    && (forall|rid: RequestId|
        #[trigger] e.cs.live_requests@.contains_key(rid) ==>
        exists|k: int| 0 <= k < requests.len()
            && requests[k].request_id == rid)
    && (forall|k: int| 0 <= k < requests.len() ==>
        e.cs.live_requests@.contains_key(#[trigger] requests[k].request_id))
    && (forall|k: int| 0 <= k < requests.len() ==>
        #[trigger] request_state_view_eq(
            e.cs.live_requests@[requests[k].request_id], requests[k],
        ))
}

// Exact engine-level effect of admitting a request between compute steps.  No
// tensor, permission, weight, or cache representation changes; only the
// scheduler's live map and waiting queue are extended.
pub open spec fn engine_admission_relation(
    before: &Engine,
    after: &Engine,
    request: RequestState,
) -> bool {
    scheduler_admission_relation(&before.cs, &after.cs, request)
    && after.kv_caches@ == before.kv_caches@
    && after.weights == before.weights
    && after.runtime == before.runtime
    && after.model_config == before.model_config
    && after.kv_caches_repr@ == before.kv_caches_repr@
    && after.weights_perms@ == before.weights_perms@
    && after.kv_perms@ == before.kv_perms@
}

// The engine's per-request sampled result `(next_sampler_state, token)` map.
// `step` returns it as ghost data: the view of the map computed by the verified
// `step_core`. `engine_step_relation` takes it as a parameter.
// The same map drives the abstract `ibm_step`, so engine and
// machine evolve their shared request states identically.
pub open spec fn samples_view(
    m: Map<u64, SampleResult>,
) -> Map<RequestId, (crate::exec::request_state::SamplerState, TokenId)> {
    m.map_values(|s: SampleResult| (s.sampler_state, s.token))
}

// Exact sampling tail exported by `step_core`.  This predicate is kept opaque
// at the engine call site and revealed only by the small packaging lemma below;
// otherwise its two row/domain quantifiers interact with every later commit
// invariant in `Engine::step`.
#[verifier::opaque]
pub open spec fn step_core_samples_match(
    plan: &StepPlan,
    live_requests: Map<RequestId, RequestState>,
    logits_repr: Tensor2D,
    samples: Map<RequestId, SampleResult>,
) -> bool {
    (forall|i: int| #![trigger plan.scheduled_ids@[i], plan.sample_mask@[i]]
        0 <= i < plan.scheduled_ids@.len()
            && plan.sample_mask@[i]
            && live_requests.contains_key(plan.scheduled_ids@[i]) ==> {
        let rid = plan.scheduled_ids@[i];
        &&& samples.contains_key(rid)
        &&& samples[rid].sampler_state == RT::sample_from_repr(
            RT::select_sample_logits_repr(
                logits_repr, plan.cu_seqlens_q_repr@, i as nat,
            ),
            live_requests[rid].sampler_state,
        ).0
        &&& samples[rid].token as nat == RT::sample_from_repr(
            RT::select_sample_logits_repr(
                logits_repr, plan.cu_seqlens_q_repr@, i as nat,
            ),
            live_requests[rid].sampler_state,
        ).1
    })
    && (forall|rid: RequestId| #[trigger] samples.contains_key(rid) ==>
        live_requests.contains_key(rid)
            && exists|i: int| 0 <= i < plan.scheduled_ids@.len()
                && plan.scheduled_ids@[i] == rid
                && #[trigger] plan.sample_mask@[i])
}

// The step's plan/weight representations. `step` returns these ghost values,
// constructed from the verified `plan`'s ghost fields and owned weight permissions.
pub ghost struct StepReprs {
    pub wr: ModelWeightsRepr,
    pub input_ids: IntTensor1D,
    pub positions: IntTensor1D,
    pub slots: Seq<int>,
    pub cu_q: Seq<int>,
    pub cu_k: Seq<int>,
    pub bt: Seq<Seq<BlockId>>,
    pub max_q: nat,
    pub max_k: nat,
    // Scheduled request IDs identify the owner of each cu_q batch segment.
    // Per-request decomposition witnesses are defined from these representations,
    // without a trusted witness-construction assumption.
    pub scheduled: Seq<RequestId>,
    // Mirrors `StepPlan::sample_mask`; this is the sole row-effect decision
    // carried into the engine relation and refinement proofs.
    pub sample_mask: Seq<bool>,
}

pub open spec fn step_semantic_model(
    old_e: Engine,
    reprs: StepReprs,
) -> SemanticModelRepr {
    SemanticModelRepr {
        weights: reprs.wr,
        architecture:
            RT::model_weights_architecture_repr_of(&old_e.weights_perms@),
    }
}

// Semantic companion to the model-opaque structural step relation.  Keeping
// this separate prevents recursive family payload equality from entering the
// large scheduler/cache proof contexts while still making complete model
// identity an explicit stable-boundary guarantee.
#[verifier::opaque]
pub open spec fn engine_step_semantic_identity(
    old_e: Engine,
    new_e: Engine,
) -> bool {
    RT::model_weights_architecture_repr_of(&new_e.weights_perms@)
        == RT::model_weights_architecture_repr_of(&old_e.weights_perms@)
}

// Architecture-dispatched view of the step input.
pub open spec fn architecture_step_logits_repr(
    old_e: Engine,
    reprs: StepReprs,
) -> Tensor2D {
    MA::model_forward_logits_repr(
        reprs.wr,
        RT::model_weights_architecture_repr_of(&old_e.weights_perms@),
        reprs.input_ids,
        reprs.positions,
        old_e.kv_caches_repr@,
        reprs.slots,
        reprs.cu_q,
        reprs.cu_k,
        reprs.max_q,
        reprs.max_k,
        reprs.bt,
    )
}


// A request has an observable row effect exactly when its unique scheduled
// row is marked in the plan's sample mask. Keeping this row-indexed avoids a
// second request-id collection whose agreement with the cu partition would
// itself need verification.
pub open spec fn reprs_emits(reprs: StepReprs, rid: RequestId) -> bool {
    exists|i: int| 0 <= i < reprs.scheduled.len()
        && reprs.scheduled[i] == rid
        && #[trigger] reprs.sample_mask[i]
}

pub open spec fn reprs_parks(reprs: StepReprs, rid: RequestId) -> bool {
    exists|i: int| 0 <= i < reprs.scheduled.len()
        && reprs.scheduled[i] == rid
        && !#[trigger] reprs.sample_mask[i]
}

// Model-opaque sampling payload for an Engine step. The scheduler/lifecycle
// relation needs only the concrete logits tensor whose last row is selected
// for each emitting request; the model family chooses that tensor separately.
pub open spec fn engine_samples_match_logits(
    old_e: Engine,
    samples: Map<RequestId, (crate::exec::request_state::SamplerState, TokenId)>,
    reprs: StepReprs,
    logits_repr: Tensor2D,
) -> bool {
    (forall|i: int| #![trigger reprs.scheduled[i], reprs.sample_mask[i]]
        0 <= i < reprs.scheduled.len() && reprs.sample_mask[i] ==> {
            let rid = reprs.scheduled[i];
            &&& samples.contains_key(rid)
            &&& samples[rid] == RT::sample_from_repr(
                RT::select_sample_logits_repr(
                    logits_repr, reprs.cu_q, i as nat,
                ),
                old_e.cs.live_requests@[rid].sampler_state,
            )
        })
    && (forall|rid: RequestId| #[trigger] samples.contains_key(rid) ==>
        reprs_emits(reprs, rid))
}

// Well-formedness of the step's plan reprs: the shape facts
// the verified `plan` establishes, re-exposed on the ghost reprs so the
// plan-layout derivation can split the batch along the cu_q partition.
// Proven by `step` from `plan`'s ensures (`step_plan_shape_ok` +
// `step_plan_commit_ready`).
pub open spec fn step_reprs_wf(old_e: Engine, reprs: StepReprs) -> bool {
    reprs.scheduled.no_duplicates()
    && reprs.sample_mask.len() == reprs.scheduled.len()
    && reprs.input_ids.len() == reprs.positions.len()
    && reprs.slots.len() == reprs.input_ids.len()
    && reprs.cu_q.len() == reprs.scheduled.len() + 1
    && reprs.cu_k.len() == reprs.scheduled.len() + 1
    && reprs.bt.len() == reprs.scheduled.len()
    && reprs.cu_q[0] == 0
    && reprs.cu_k[0] == 0
    && reprs.max_q <= u64::MAX as nat
    && reprs.max_k <= u64::MAX as nat
    && (forall|j: int| 0 <= j < reprs.scheduled.len() as int ==>
        reprs.cu_q[j] < #[trigger] reprs.cu_q[j + 1])
    && (forall|j: int| 0 <= j < reprs.scheduled.len() as int ==>
        reprs.cu_k[j] < #[trigger] reprs.cu_k[j + 1])
    && (forall|j: int| #![trigger reprs.cu_q[j + 1]]
        0 <= j < reprs.scheduled.len() as int ==> {
        let q_len = reprs.cu_q[j + 1] - reprs.cu_q[j];
        let k_len = reprs.cu_k[j + 1] - reprs.cu_k[j];
        &&& q_len <= reprs.max_q as int
        &&& k_len <= reprs.max_k as int
        &&& q_len <= k_len
    })
    && reprs.cu_q[reprs.scheduled.len() as int] == reprs.input_ids.len() as int
    && reprs.wr.layers.len() == old_e.model_config.num_layers as nat
    && reprs.wr == RT::model_weights_repr_of(&old_e.weights_perms@)
}

// Observable-effect policy is separate from tensor-shape well-formedness so
// K/V and geometry proofs do not instantiate its row quantifiers. KV-only
// rows must be admitted prefills, and an admission samples exactly at finality.
#[verifier::opaque]
pub open spec fn reprs_sample_policy(old_e: Engine, reprs: StepReprs) -> bool {
    (forall|i: int| #![trigger reprs.scheduled[i], reprs.sample_mask[i]]
        0 <= i < reprs.scheduled.len() && !reprs.sample_mask[i]
        ==> old_e.cs.waiting@.contains(reprs.scheduled[i]))
    && (forall|i: int| #![trigger reprs.scheduled[i], reprs.sample_mask[i]]
        0 <= i < reprs.scheduled.len()
            && old_e.cs.waiting@.contains(reprs.scheduled[i]) ==> {
        let rid = reprs.scheduled[i];
        let k_len = reprs.cu_k[i + 1] - reprs.cu_k[i];
        reprs.sample_mask[i]
            <==> k_len == old_e.cs.live_requests@[rid].prompt_tokens@.len()
    })
}

pub proof fn lemma_old_running_row_emits(
    old_e: Engine,
    reprs: StepReprs,
    k: int,
)
    requires
        crate::exec::cache_scheduler::queue_disjoint(&old_e.cs),
        step_reprs_wf(old_e, reprs),
        reprs_sample_policy(old_e, reprs),
        0 <= k < reprs.scheduled.len(),
        old_e.cs.running@.contains(reprs.scheduled[k]),
    ensures
        reprs.sample_mask[k],
        reprs_emits(reprs, reprs.scheduled[k]),
{
    if !reprs.sample_mask[k] {
        lemma_kv_only_row_was_waiting(old_e, reprs, k);
        assert(false);
    }
}

pub proof fn lemma_nonempty_step_has_query_tokens(old_e: Engine, reprs: StepReprs)
    requires step_reprs_wf(old_e, reprs), reprs.scheduled.len() > 0,
    ensures reprs.input_ids.len() > 0,
{
    reveal(step_reprs_wf);
    assert forall|j: int| 0 <= j < reprs.scheduled.len() implies
        (#[trigger] reprs.cu_q[j]) < reprs.cu_q[j + 1] by {}
    assert(reprs.cu_q[0] < reprs.cu_q[1]);
    crate::proof::tensor::geometry::lemma_cu_mono(reprs.cu_q, reprs.scheduled.len() as int,
        1, reprs.scheduled.len() as int);
}

pub proof fn lemma_kv_only_row_was_waiting(
    old_e: Engine,
    reprs: StepReprs,
    k: int,
)
    requires
        step_reprs_wf(old_e, reprs),
        reprs_sample_policy(old_e, reprs),
        0 <= k < reprs.scheduled.len(),
        !reprs.sample_mask[k],
    ensures
        old_e.cs.waiting@.contains(reprs.scheduled[k]),
{
    reveal(reprs_sample_policy);
}

pub proof fn lemma_waiting_sample_row_is_final(
    old_e: Engine,
    reprs: StepReprs,
    k: int,
)
    requires
        step_reprs_wf(old_e, reprs),
        reprs_sample_policy(old_e, reprs),
        0 <= k < reprs.scheduled.len(),
        old_e.cs.waiting@.contains(reprs.scheduled[k]),
        reprs.sample_mask[k],
    ensures
        reprs.cu_k[k + 1] - reprs.cu_k[k]
            == old_e.cs.live_requests@[reprs.scheduled[k]]
                .prompt_tokens@.len(),
{
    reveal(reprs_sample_policy);
}

// Package the sampler's two directional row facts into the exact mask
// contract consumed by `CacheScheduler::commit`.  Keeping this conversion in a
// spinoff proof prevents its existential/no-duplicates reasoning from sharing
// a solver context with the much larger engine-step assembly.
#[verifier::spinoff_prover]
pub proof fn lemma_step_core_samples_to_engine_payload_facts(
    old_e: Engine,
    post_plan: &CacheScheduler,
    plan: &StepPlan,
    reprs: StepReprs,
    samples: Map<RequestId, SampleResult>,
    samples_g: Map<RequestId, (SamplerState, TokenId)>,
    core_logits: Tensor2D,
    relation_logits: Tensor2D,
)
    requires
        step_reprs_wf(old_e, reprs),
        reprs_sample_policy(old_e, reprs),
        reprs.scheduled == plan.scheduled_ids@,
        reprs.sample_mask == plan.sample_mask@,
        reprs.cu_q == plan.cu_seqlens_q_repr@,
        post_plan.live_requests@ == old_e.cs.live_requests@,
        live_covers_queue(&old_e.cs),
        live_covers_queue(post_plan),
        waiting_unstarted(&old_e.cs),
        old_e.kv_caches_repr@.len() >= reprs.wr.layers.len(),
        samples_g == samples_view(samples),
        core_logits == relation_logits,
        core_logits.len() == reprs.input_ids.len(),
        step_core_samples_match(
            plan, post_plan.live_requests@, core_logits, samples,
        ),
        forall|k: int| #![trigger plan.scheduled_ids@[k]]
            0 <= k < plan.scheduled_ids@.len()
                ==> post_plan.running@.contains(plan.scheduled_ids@[k]),
    ensures
        forall|k: int| #![trigger plan.scheduled_ids@[k], plan.sample_mask@[k]]
            0 <= k < plan.scheduled_ids@.len()
                ==> (samples.contains_key(plan.scheduled_ids@[k])
                    <==> plan.sample_mask@[k]),
        forall|k: int| #![trigger plan.scheduled_ids@[k], plan.sample_mask@[k]]
            0 <= k < plan.scheduled_ids@.len()
                && !plan.sample_mask@[k]
            ==> post_plan.live_requests@.contains_key(plan.scheduled_ids@[k])
                && post_plan.live_requests@[plan.scheduled_ids@[k]]
                    .generated_tokens@.len() == 0,
        engine_samples_match_logits(
            old_e, samples_g, reprs, relation_logits,
        ),
{
    reveal(step_core_samples_match);
    crate::proof::tensor::geometry::lemma_cu_int_bounds(
        reprs.cu_q, reprs.scheduled.len() as int,
    );
    assert forall|k: int| #![trigger plan.scheduled_ids@[k], plan.sample_mask@[k]]
        0 <= k < plan.scheduled_ids@.len() implies
            (samples.contains_key(plan.scheduled_ids@[k])
                <==> plan.sample_mask@[k])
    by {
        let rid = plan.scheduled_ids@[k];
        assert(post_plan.live_requests@.contains_key(rid));
        if samples.contains_key(rid) {
            let j = choose|j: int| 0 <= j < plan.scheduled_ids@.len()
                && plan.scheduled_ids@[j] == rid
                && plan.sample_mask@[j];
            assert(j == k) by {
                assert(plan.scheduled_ids@.no_duplicates());
            }
        }
    }
    assert forall|k: int| #![trigger plan.scheduled_ids@[k], plan.sample_mask@[k]]
        0 <= k < plan.scheduled_ids@.len() && !plan.sample_mask@[k]
        implies post_plan.live_requests@.contains_key(plan.scheduled_ids@[k])
            && post_plan.live_requests@[plan.scheduled_ids@[k]]
                .generated_tokens@.len() == 0
    by {
        let rid = plan.scheduled_ids@[k];
        assert(!reprs.sample_mask[k]);
        lemma_kv_only_row_was_waiting(old_e, reprs, k);
        assert(old_e.cs.live_requests@.contains_key(rid));
        assert(old_e.cs.live_requests@[rid].generated_tokens@.len() == 0);
    }
    assert(engine_samples_match_logits(
        old_e, samples_g, reprs, relation_logits,
    )) by {
        assert forall|i: int|
            #![trigger reprs.scheduled[i], reprs.sample_mask[i]]
            0 <= i < reprs.scheduled.len() && reprs.sample_mask[i]
            implies {
                let rid = reprs.scheduled[i];
                &&& samples_g.contains_key(rid)
                &&& samples_g[rid] == RT::sample_from_repr(
                    RT::select_sample_logits_repr(
                        relation_logits, reprs.cu_q, i as nat,
                    ),
                    old_e.cs.live_requests@[rid].sampler_state,
                )
            }
        by {
            let rid = reprs.scheduled[i];
            assert(0 <= reprs.cu_q[i]);
            assert(reprs.cu_q[i] < reprs.cu_q[i + 1]);
            assert(reprs.cu_q[i + 1] > 0);
            assert(reprs.cu_q[i + 1]
                <= reprs.cu_q[reprs.scheduled.len() as int]);
            assert(reprs.cu_q[i + 1] <= relation_logits.len());
            assert(post_plan.live_requests@.contains_key(rid));
            assert(samples.contains_key(rid));
            assert(samples_g.contains_key(rid));
            assert(samples_g[rid]
                == (samples[rid].sampler_state, samples[rid].token));
            assert(samples_g[rid].0 == samples[rid].sampler_state);
            assert(samples_g[rid].1 == samples[rid].token);
            let expected = RT::sample_from_repr(
                RT::select_sample_logits_repr(
                    core_logits, plan.cu_seqlens_q_repr@, i as nat,
                ),
                post_plan.live_requests@[rid].sampler_state,
            );
            assert(samples[rid].sampler_state == expected.0);
            assert(samples[rid].token as nat == expected.1);
            assert(expected == RT::sample_from_repr(
                RT::select_sample_logits_repr(
                    relation_logits, reprs.cu_q, i as nat,
                ),
                old_e.cs.live_requests@[rid].sampler_state,
            ));
        }
        assert forall|rid: RequestId|
            #[trigger] samples_g.contains_key(rid)
            implies reprs_emits(reprs, rid)
        by {
            assert(samples.contains_key(rid));
            let i = choose|i: int| 0 <= i < plan.scheduled_ids@.len()
                && plan.scheduled_ids@[i] == rid
                && plan.sample_mask@[i];
            assert(reprs.scheduled[i] == rid);
            assert(reprs.sample_mask[i]);
        }
    }
}

// Translate commit's plan-indexed output facts into the request-indexed
// emission predicate stored in `engine_step_relation`.
#[verifier::spinoff_prover]
pub proof fn lemma_commit_emits_iff_reprs(
    old_e: Engine,
    pre_commit: &CacheScheduler,
    plan: &StepPlan,
    samples: Map<RequestId, SampleResult>,
    emitted: Map<RequestId, TokenId>,
    reprs: StepReprs,
)
    requires
        step_reprs_wf(old_e, reprs),
        reprs.scheduled == plan.scheduled_ids@,
        reprs.sample_mask == plan.sample_mask@,
        pre_commit.live_requests@ == old_e.cs.live_requests@,
        forall|k: int| #![trigger plan.scheduled_ids@[k], plan.sample_mask@[k]]
            0 <= k < plan.scheduled_ids@.len()
                ==> (samples.contains_key(plan.scheduled_ids@[k])
                    <==> plan.sample_mask@[k]),
        emitted.dom().subset_of(samples.dom()),
        forall|r: RequestId| #[trigger] emitted.contains_key(r)
            ==> step_plan_emits(plan, r),
        forall|k: int| #![trigger plan.scheduled_ids@[k], plan.sample_mask@[k]]
            0 <= k < plan.scheduled_ids@.len() && plan.sample_mask@[k]
                ==> emitted.contains_key(plan.scheduled_ids@[k]),
    ensures
        forall|r: RequestId| #[trigger] old_e.cs.live_requests@.contains_key(r)
            ==> (emitted.contains_key(r) <==> reprs_emits(reprs, r)),
{
    assert forall|r: RequestId| #[trigger] old_e.cs.live_requests@.contains_key(r)
        implies (emitted.contains_key(r) <==> reprs_emits(reprs, r))
    by {
        if emitted.contains_key(r) {
            let k = choose|k: int| 0 <= k < plan.scheduled_ids@.len()
                && plan.scheduled_ids@[k] == r
                && plan.sample_mask@[k];
            assert(reprs.scheduled[k] == r);
            assert(reprs.sample_mask[k]);
        }
        if reprs_emits(reprs, r) {
            let k = choose|k: int| 0 <= k < reprs.scheduled.len()
                && reprs.scheduled[k] == r
                && reprs.sample_mask[k];
            assert(pre_commit.live_requests@.contains_key(plan.scheduled_ids@[k]));
            assert(samples.contains_key(plan.scheduled_ids@[k]));
            assert(emitted.contains_key(plan.scheduled_ids@[k]));
        }
    }
}

// A final prefill row may append one token during commit.  Transport the
// transient plan block table across that append into the stable post-step
// residency witness used by refinement.  KV-only and decode rows make this
// predicate vacuous and therefore stay out of the arithmetic proof.
#[verifier::spinoff_prover]
pub proof fn lemma_commit_preserves_prefill_residencies(
    old_e: Engine,
    pre_commit: Engine,
    new_e: Engine,
    plan: &StepPlan,
    emitted: Map<RequestId, TokenId>,
    reprs: StepReprs,
)
    requires
        cs_valid(&old_e.cs),
        cs_valid(&pre_commit.cs),
        cs_valid(&new_e.cs),
        step_reprs_wf(old_e, reprs),
        reprs_sample_policy(old_e, reprs),
        reprs.scheduled == plan.scheduled_ids@,
        reprs.sample_mask == plan.sample_mask@,
        reprs.bt == plan.block_table_repr@,
        reprs.cu_q == plan.cu_seqlens_q_repr@,
        reprs.cu_k == plan.cu_seqlens_k_repr@,
        pre_commit.cs.live_requests@ == old_e.cs.live_requests@,
        forall|k: int| 0 <= k < plan.scheduled_ids@.len()
            ==> #[trigger] plan_slot_segments_at(
                &old_e.cs, &pre_commit.cs, plan, k,
            ),
        forall|k: int| #![trigger plan.scheduled_ids@[k]]
            0 <= k < plan.scheduled_ids@.len()
                ==> pre_commit.cs.running@.contains(plan.scheduled_ids@[k]),
        forall|r: RequestId| #[trigger] reprs.scheduled.contains(r)
            ==> old_e.cs.running@.contains(r) || old_e.cs.waiting@.contains(r),
        forall|r: RequestId| #[trigger] old_e.cs.live_requests@.contains_key(r)
            ==> (emitted.contains_key(r) <==> reprs_emits(reprs, r)),
        forall|r: RequestId| #[trigger] old_e.cs.live_requests@.contains_key(r)
            ==> (new_e.cs.live_requests@.contains_key(r) <==>
                !(emitted.contains_key(r)
                    && should_finish_after_append(
                        old_e.cs.live_requests@[r], emitted[r],
                    ))),
        forall|r: RequestId| #[trigger] new_e.cs.running@.contains(r)
            <==> (pre_commit.cs.running@.contains(r)
                && !step_plan_parks(plan, r)
                && !(emitted.contains_key(r)
                    && should_finish_after_append(
                        pre_commit.cs.live_requests@[r], emitted[r],
                    ))),
        forall|r: RequestId| #[trigger] emitted.contains_key(r)
            && pre_commit.cs.request_residency@.contains_key(r)
            && new_e.cs.request_residency@.contains_key(r)
            ==> new_e.cs.request_residency@[r].block_ids@.len()
                    >= pre_commit.cs.request_residency@[r].block_ids@.len()
                && new_e.cs.request_residency@[r].block_ids@.subrange(
                    0,
                    pre_commit.cs.request_residency@[r].block_ids@.len() as int,
                ) == pre_commit.cs.request_residency@[r].block_ids@
                && new_e.cs.request_residency@[r].cached_prefix_blocks
                    == pre_commit.cs.request_residency@[r].cached_prefix_blocks,
    ensures
        reprs_prefill_residencies(old_e, new_e, reprs),
{
    reveal(reprs_prefill_residencies);
    assert forall|k: int| 0 <= k < reprs.scheduled.len()
        implies #[trigger] reprs_prefill_residency_at(old_e, new_e, reprs, k)
    by {
        let rid = reprs.scheduled[k];
        reveal(reprs_prefill_residency_at);
        if reprs.sample_mask[k] {
            if !old_e.cs.running@.contains(rid)
                && new_e.cs.live_requests@.contains_key(rid) {
                reveal(crate::exec::cache_scheduler::plan_slot_segments_at);
                assert(crate::exec::cache_scheduler::plan_slot_segments_at(
                    &old_e.cs, &pre_commit.cs, plan, k,
                ));
                assert(pre_commit.cs.request_residency@.contains_key(rid));
                let ids_pc = pre_commit.cs.request_residency@[rid].block_ids@;
                let ids_new = new_e.cs.request_residency@[rid].block_ids@;
                assert(reprs.bt[k] == ids_pc);
                assert(emitted.contains_key(rid)) by {
                    assert(old_e.cs.live_requests@.contains_key(rid));
                    assert(reprs.scheduled.contains(rid));
                }
                assert(!step_plan_parks(plan, rid)) by {
                    if step_plan_parks(plan, rid) {
                        let j = choose|j: int| 0 <= j < plan.scheduled_ids@.len()
                            && plan.scheduled_ids@[j] == rid
                            && !plan.sample_mask@[j];
                        assert(j == k) by {
                            assert(plan.scheduled_ids@.no_duplicates());
                        }
                    }
                }
                assert(!(emitted.contains_key(rid)
                    && should_finish_after_append(
                        old_e.cs.live_requests@[rid], emitted[rid],
                    )));
                assert(new_e.cs.running@.contains(rid));
                assert(new_e.cs.request_residency@.contains_key(rid)) by {
                    assert(crate::exec::cache_scheduler::running_has_residency(
                        &new_e.cs,
                    ));
                }
                assert(ids_new.len() >= ids_pc.len());
                assert(ids_new.subrange(0, ids_pc.len() as int) == ids_pc);
                assert(new_e.cs.request_residency@[rid].cached_prefix_blocks
                    == pre_commit.cs.request_residency@[rid]
                        .cached_prefix_blocks);
                let n = old_e.cs.live_requests@[rid].prompt_tokens@.len() as int;
                let q = reprs.cu_q[k + 1] - reprs.cu_q[k];
                let kd = reprs.cu_k[k + 1] - reprs.cu_k[k];
                assert(old_e.cs.waiting@.contains(rid)) by {
                    assert(plan.scheduled_ids@.contains(rid));
                }
                lemma_waiting_sample_row_is_final(old_e, reprs, k);
                assert(kd == n);
                let c_pc = pre_commit.cs.request_residency@[rid]
                    .cached_prefix_blocks as int;
                assert(q == n - c_pc
                    * (crate::types::BLOCK_SIZE_SPEC as int));
                let c_tokens = n - q;
                assert(c_tokens == c_pc
                    * (crate::types::BLOCK_SIZE_SPEC as int));
                assert(0 <= c_pc);
                vstd::arithmetic::div_mod::lemma_fundamental_div_mod_converse_mod(
                    c_tokens,
                    crate::types::BLOCK_SIZE_SPEC as int,
                    c_pc,
                    0,
                );
                vstd::arithmetic::div_mod::lemma_fundamental_div_mod_converse_div(
                    c_tokens,
                    crate::types::BLOCK_SIZE_SPEC as int,
                    c_pc,
                    0,
                );
            }
        }
    }
    assert(reprs_prefill_residencies(old_e, new_e, reprs));
}

// Stable finite-pool bound for every materialized plan block table.  Kept
// separate from `reprs_forward_layout_at` so proofs that reveal detailed row
// payloads do not inherit an unrelated cardinality conjunct.
pub open spec fn step_reprs_block_tables_bounded(
    old_e: Engine,
    reprs: StepReprs,
) -> bool {
    forall|k: int| #![trigger reprs.bt[k]]
        0 <= k < reprs.scheduled.len()
            ==> reprs.bt[k].len() <= old_e.cs.num_blocks
}

// Request-local forward payload exported by the verified scheduler and kept
// in the step relation across commit.  This is the engine-facing form of
// `cache_scheduler::plan_forward_layout_ok`.
pub open spec fn reprs_forward_layout_at(
    old_e: Engine,
    reprs: StepReprs,
    k: int,
) -> bool {
    let rid = reprs.scheduled[k];
    let s0 = reprs.cu_q[k];
    let s1 = reprs.cu_q[k + 1];
    let kd = reprs.cu_k[k + 1] - reprs.cu_k[k];
    &&& old_e.cs.live_requests@.contains_key(rid)
    &&& valid_request_state(old_e.cs.live_requests@[rid])
    &&& 0 <= s0 < s1
    &&& s1 <= reprs.input_ids.len() as int
    &&& crate::proof::tensor::geometry::blocks_needed_for(kd as nat) <= reprs.bt[k].len()
    &&& reprs.bt[k].no_duplicates()
    &&& (forall|l: int| 0 <= l < reprs.bt[k].len() ==>
        #[trigger] reprs.bt[k][l] < old_e.cs.num_blocks)
    &&& (if old_e.cs.running@.contains(rid) {
        let h = history(old_e.cs.live_requests@[rid]);
        old_e.cs.request_residency@.contains_key(rid)
        && reprs.bt[k] == old_e.cs.request_residency@[rid].block_ids@
        && reprs.bt[k].len()
            == crate::proof::tensor::geometry::blocks_needed_for(h.len())
        && s1 == s0 + 1
        && kd == h.len() as int
        && reprs.input_ids[s0] == h[h.len() - 1] as int
        && reprs.positions[s0] == h.len() as int - 1
        && reprs.slots[s0] >= 0
        && reprs.slots[s0] as nat
            == crate::proof::tensor::geometry::block_table_slot(
                reprs.bt[k], (h.len() - 1) as nat,
            )
    } else {
        let n = old_e.cs.live_requests@[rid].prompt_tokens@.len() as int;
        let c = kd - (s1 - s0);
        old_e.cs.waiting@.contains(rid)
        && reprs.bt[k].len()
            == crate::proof::tensor::geometry::blocks_needed_for(n as nat)
        && 0 <= c < kd
        && kd <= n
        && (forall|q: int| s0 <= q < s1 ==> {
            let p = c + q - s0;
            &&& #[trigger] reprs.input_ids[q]
                == old_e.cs.live_requests@[rid].prompt_tokens@[p] as int
            &&& reprs.positions[q] == p
            &&& reprs.slots[q] >= 0
            &&& reprs.slots[q] as nat
                == crate::proof::tensor::geometry::block_table_slot(reprs.bt[k], p as nat)
        })
    })
}

// Uniform row-slot projection for both decode and prefill.  A query entry at
// offset `q - s0` always writes logical key position
// `k_len - q_len + (q - s0)`, regardless of how the scheduler produced the
// row.  Keeping this arithmetic at the Engine layout boundary lets model and
// cache proofs remain phase-agnostic.
pub proof fn lemma_reprs_forward_layout_slot_at(
    old_e: Engine,
    reprs: StepReprs,
    row: int,
    q: int,
)
    requires
        reprs_forward_layout_at(old_e, reprs, row),
        0 <= row < reprs.scheduled.len(),
        reprs.cu_q[row] <= q < reprs.cu_q[row + 1],
    ensures ({
        let s0 = reprs.cu_q[row];
        let s1 = reprs.cu_q[row + 1];
        let k_len = reprs.cu_k[row + 1] - reprs.cu_k[row];
        let logical_pos = k_len - (s1 - s0) + q - s0;
        &&& reprs.slots[q] >= 0
        &&& reprs.slots[q] as nat
            == crate::proof::tensor::geometry::block_table_slot(
                reprs.bt[row], logical_pos as nat,
            )
    }),
{
    reveal(reprs_forward_layout_at);
    let rid = reprs.scheduled[row];
    let s0 = reprs.cu_q[row];
    let s1 = reprs.cu_q[row + 1];
    let k_len = reprs.cu_k[row + 1] - reprs.cu_k[row];
    let logical_pos = k_len - (s1 - s0) + q - s0;
    if old_e.cs.running@.contains(rid) {
        assert(s1 == s0 + 1);
        assert(q == s0);
        assert(k_len
            == crate::exec::request_state::history(
                old_e.cs.live_requests@[rid],
            ).len() as int);
        assert(logical_pos
            == crate::exec::request_state::history(
                old_e.cs.live_requests@[rid],
            ).len() as int - 1);
    } else {
        let prompt = old_e.cs.live_requests@[rid].prompt_tokens@;
        let cached = k_len - (s1 - s0);
        assert(cached == logical_pos - (q - s0));
        assert({
            let p = cached + q - s0;
            &&& reprs.input_ids[q] == prompt[p] as int
            &&& reprs.positions[q] == p
            &&& reprs.slots[q] >= 0
            &&& reprs.slots[q] as nat
                == crate::proof::tensor::geometry::block_table_slot(
                    reprs.bt[row], p as nat,
                )
        });
    }
}

pub open spec fn reprs_forward_layout_ok(old_e: Engine, reprs: StepReprs) -> bool {
    forall|k: int| 0 <= k < reprs.scheduled.len() ==>
        #[trigger] reprs_forward_layout_at(old_e, reprs, k)
}

// Stable engine-facing export of the scheduler's cached-prefix origin: every
// page omitted from a partial-prefill query was already an exact registry
// target at the original step boundary, contains the requester's exact prompt
// tokens at those positions, and was never produced by this same forward.
pub open spec fn reprs_cached_prefix_origin_at(
    old_e: Engine,
    reprs: StepReprs,
    k: int,
) -> bool {
    let rid = reprs.scheduled[k];
    if old_e.cs.running@.contains(rid) {
        true
    } else {
        let q = reprs.cu_q[k + 1] - reprs.cu_q[k];
        let kd = reprs.cu_k[k + 1] - reprs.cu_k[k];
        let c_tokens = kd - q;
        let c = c_tokens / (crate::types::BLOCK_SIZE_SPEC as int);
        let ids = reprs.bt[k];
        &&& old_e.cs.live_requests@.contains_key(rid)
        &&& 0 <= c_tokens
        &&& c_tokens % (crate::types::BLOCK_SIZE_SPEC as int) == 0
        &&& 0 <= c <= ids.len()
        &&& registered_prefix_chain(old_e.cs.blocks@, ids.subrange(0, c))
        &&& token_placement_prefix(
            old_e.cs.blocks@, ids,
            old_e.cs.live_requests@[rid].prompt_tokens@, c_tokens,
        )
        &&& forall|l: int| #![trigger ids[l]] 0 <= l < c ==> {
            let bid = ids[l];
            &&& old_e.cs.blocks@.contains_key(bid)
            &&& old_e.cs.hash_to_block@.contains_key(
                old_e.cs.blocks@[bid].hash_value)
            &&& old_e.cs.hash_to_block@[
                old_e.cs.blocks@[bid].hash_value] == bid
        }
    }
}

#[verifier::opaque]
pub open spec fn reprs_cached_prefix_origins(
    old_e: Engine,
    reprs: StepReprs,
) -> bool {
    forall|k: int| 0 <= k < reprs.scheduled.len() ==>
        #[trigger] reprs_cached_prefix_origin_at(old_e, reprs, k)
}

pub proof fn lemma_plan_cached_prefix_origins_to_reprs(
    old_e: Engine,
    plan: &StepPlan,
    reprs: StepReprs,
)
    requires
        plan_cached_prefix_origins(&old_e.cs, plan),
        reprs.scheduled == plan.scheduled_ids@,
        reprs.cu_q == plan.cu_seqlens_q_repr@,
        reprs.cu_k == plan.cu_seqlens_k_repr@,
        reprs.bt == plan.block_table_repr@,
    ensures
        reprs_cached_prefix_origins(old_e, reprs),
{
    reveal(plan_cached_prefix_origins);
    reveal(reprs_cached_prefix_origins);
    assert forall|k: int| 0 <= k < reprs.scheduled.len() implies
        #[trigger] reprs_cached_prefix_origin_at(old_e, reprs, k)
    by {
        assert(plan_cached_prefix_origin_at(&old_e.cs, plan, k));
    }
}

// Stable bridge from the plan-time admission residency to the post-step
// residency retained by a surviving request.  Commit may append one block for
// the sampled token, but it preserves the entire plan block table as a prefix
// and keeps the cached-page count unchanged.
#[verifier::opaque]
pub open spec fn reprs_prefill_residency_at(
    old_e: Engine,
    new_e: Engine,
    reprs: StepReprs,
    k: int,
) -> bool {
    let rid = reprs.scheduled[k];
    if !reprs.sample_mask[k] || old_e.cs.running@.contains(rid) {
        true
    } else if new_e.cs.live_requests@.contains_key(rid) {
        let n = old_e.cs.live_requests@[rid].prompt_tokens@.len() as int;
        let q = reprs.cu_q[k + 1] - reprs.cu_q[k];
        let c_tokens = n - q;
        let c = c_tokens / (crate::types::BLOCK_SIZE_SPEC as int);
        let new_ids = new_e.cs.request_residency@[rid].block_ids@;
        &&& new_e.cs.request_residency@.contains_key(rid)
        &&& 0 <= c_tokens
        &&& c_tokens % (crate::types::BLOCK_SIZE_SPEC as int) == 0
        &&& new_e.cs.request_residency@[rid].cached_prefix_blocks as int == c
        &&& reprs.bt[k].len() <= new_ids.len()
        &&& new_ids.subrange(0, reprs.bt[k].len() as int) == reprs.bt[k]
    } else {
        true
    }
}

#[verifier::opaque]
pub open spec fn reprs_prefill_residencies(
    old_e: Engine,
    new_e: Engine,
    reprs: StepReprs,
) -> bool {
    forall|k: int| 0 <= k < reprs.scheduled.len() ==>
        #[trigger] reprs_prefill_residency_at(old_e, new_e, reprs, k)
}

// Export the scheduler's transient plan-time residency facts as a stable
// engine row.  Keeping this derivation in a focused lemma avoids exposing its
// quantifiers to either the scheduler admission loop or the full `step` proof.
#[verifier::spinoff_prover]
pub proof fn lemma_plan_row_to_reprs_forward_layout(
    old_e: Engine,
    post_cs: &CacheScheduler,
    plan: &StepPlan,
    reprs: StepReprs,
    k: int,
)
    requires
        cs_valid(&old_e.cs),
        cs_valid(post_cs),
        post_cs.live_requests@ == old_e.cs.live_requests@,
        post_cs.num_blocks == old_e.cs.num_blocks,
        0 <= k < plan.scheduled_ids@.len(),
        post_cs.running@.contains(plan.scheduled_ids@[k]),
        plan_forward_layout_at(&old_e.cs, plan, k),
        plan_residency_extent_at(&old_e.cs, post_cs, plan, k),
        plan_slot_segments_at(&old_e.cs, post_cs, plan, k),
        step_reprs_wf(old_e, reprs),
        reprs.scheduled == plan.scheduled_ids@,
        reprs.input_ids == plan.input_ids_repr@,
        reprs.positions == plan.positions_repr@,
        reprs.slots == plan.slot_mapping_repr@,
        reprs.cu_q == plan.cu_seqlens_q_repr@,
        reprs.cu_k == plan.cu_seqlens_k_repr@,
        reprs.bt == plan.block_table_repr@,
    ensures
        reprs_forward_layout_at(old_e, reprs, k),
{
    reveal(plan_forward_layout_at);
    reveal(reprs_forward_layout_at);
    reveal(plan_slot_segments_at);
    let rid = reprs.scheduled[k];
    let s0 = reprs.cu_q[k];
    let s1 = reprs.cu_q[k + 1];
    let kd = reprs.cu_k[k + 1] - reprs.cu_k[k];
    let ids = post_cs.request_residency@[rid].block_ids@;
    reveal(plan_residency_extent_at);
    assert(reprs.bt[k] == ids);
    assert(ids.no_duplicates()) by {
        assert(residency_block_ids_unique(post_cs));
    }
    assert forall|l: int| 0 <= l < reprs.bt[k].len()
        implies #[trigger] reprs.bt[k][l] < old_e.cs.num_blocks
    by {
        let bid = reprs.bt[k][l];
        assert(bid == ids[l]);
        assert(post_cs.blocks@.contains_key(bid)) by {
            assert(residency_blocks_in_range(post_cs));
        }
        assert(bid < post_cs.num_blocks) by {
            assert(blocks_dom_in_range(post_cs));
        }
    }
    let post_h = history(post_cs.live_requests@[rid]);
    if old_e.cs.running@.contains(rid) {
        let h = history(old_e.cs.live_requests@[rid]);
        assert(post_h == h);
        assert(kd == post_h.len() as int);
        assert(ids.len()
            == crate::proof::tensor::geometry::blocks_needed_for(h.len()));
    } else {
        let n = old_e.cs.live_requests@[rid].prompt_tokens@.len() as int;
        assert(old_e.cs.waiting@.contains(rid));
        assert(old_e.cs.live_requests@[rid].generated_tokens@.len() == 0) by {
            assert(waiting_unstarted(&old_e.cs));
        }
        assert(post_h.len() as int == n);
        assert(ids.len()
            == crate::proof::tensor::geometry::blocks_needed_for(n as nat));
    }
    let tail_id = ids[ids.len() - 1];
    let tail = post_cs.blocks@[tail_id].tokens@.len() as int;
    assert(ids.len() >= 1);
    assert(post_cs.blocks@.contains_key(tail_id));
    assert(1 <= tail);
    assert(tail <= crate::types::BLOCK_SIZE_SPEC as int) by {
        assert(block_token_bound(post_cs));
    }
    assert(crate::proof::tensor::geometry::blocks_needed_for(kd as nat) <= reprs.bt[k].len());
    if old_e.cs.running@.contains(rid) {
        let h = history(old_e.cs.live_requests@[rid]);
        let pos = (h.len() - 1) as nat;
        crate::proof::tensor::geometry::lemma_blocks_needed_covers_pos(pos, h.len());
        assert(reprs.slots[s0]
            == crate::proof::tensor::geometry::block_table_slot(reprs.bt[k], pos) as int);
        assert(reprs.slots[s0] >= 0);
        assert(reprs.slots[s0] as nat
            == crate::proof::tensor::geometry::block_table_slot(reprs.bt[k], pos));
    } else {
        let c = kd - (s1 - s0);
        let c_post = post_cs.request_residency@[rid]
            .cached_prefix_blocks as int
            * (crate::types::BLOCK_SIZE_SPEC as int);
        assert(c == c_post);
        assert forall|q: int| s0 <= q < s1 implies {
            let p = c + q - s0;
            &&& #[trigger] reprs.slots[q] >= 0
            &&& reprs.slots[q] as nat
                == crate::proof::tensor::geometry::block_table_slot(reprs.bt[k], p as nat)
        } by {
            let p = c + q - s0;
            assert(reprs.slots[q]
                == crate::proof::tensor::geometry::block_table_slot(reprs.bt[k], p as nat) as int);
        }
    }
    assert(reprs_forward_layout_at(old_e, reprs, k));
}

// Transport the scheduler's opaque row policy into the engine-facing repr.
// Keeping this bridge small prevents sampling/finality quantifiers from
// contaminating the tensor-layout proof in `step`.
#[verifier::spinoff_prover]
pub proof fn lemma_plan_sample_policy_to_reprs(
    old_e: Engine,
    plan: &StepPlan,
    reprs: StepReprs,
)
    requires
        plan_sample_policy(&old_e.cs, plan),
        queue_disjoint(&old_e.cs),
        step_reprs_wf(old_e, reprs),
        reprs.scheduled == plan.scheduled_ids@,
        reprs.cu_k == plan.cu_seqlens_k_repr@,
        reprs.sample_mask == plan.sample_mask@,
    ensures
        reprs_sample_policy(old_e, reprs),
{
    reveal(reprs_sample_policy);
    reveal(plan_sample_policy);
    assert forall|i: int| #![trigger reprs.scheduled[i], reprs.sample_mask[i]]
        0 <= i < reprs.scheduled.len() && !reprs.sample_mask[i]
        implies old_e.cs.waiting@.contains(reprs.scheduled[i])
    by {
        assert(plan_sample_policy_at(&old_e.cs, plan, i));
        reveal(plan_sample_policy_at);
        assert(plan.scheduled_ids@[i] == reprs.scheduled[i]);
        assert(plan.sample_mask@[i] == reprs.sample_mask[i]);
        assert(!old_e.cs.running@.contains(reprs.scheduled[i]));
    }
    assert forall|i: int| #![trigger reprs.scheduled[i], reprs.sample_mask[i]]
        0 <= i < reprs.scheduled.len()
            && old_e.cs.waiting@.contains(reprs.scheduled[i]) implies {
        let rid = reprs.scheduled[i];
        let k_len = reprs.cu_k[i + 1] - reprs.cu_k[i];
        reprs.sample_mask[i]
            <==> k_len == old_e.cs.live_requests@[rid].prompt_tokens@.len()
    } by {
        let rid = reprs.scheduled[i];
        assert(plan_sample_policy_at(&old_e.cs, plan, i));
        reveal(plan_sample_policy_at);
        assert(plan.scheduled_ids@[i] == reprs.scheduled[i]);
        assert(plan.sample_mask@[i] == reprs.sample_mask[i]);
        assert(plan.cu_seqlens_k_repr@[i] == reprs.cu_k[i]);
        assert(plan.cu_seqlens_k_repr@[i + 1] == reprs.cu_k[i + 1]);
        assert(old_e.cs.live_requests@.contains_key(rid));
        assert(!old_e.cs.running@.contains(rid));
    }
}

// Architecture-dispatched post-store cache view used by the common executable
// step.
pub open spec fn architecture_engine_post_kv_of(
    old_e: Engine,
    reprs: StepReprs,
    pre_kv: Seq<(KVCacheLayerRepr, KVCacheLayerRepr)>,
) -> Seq<(KVCacheLayerRepr, KVCacheLayerRepr)> {
    MA::model_forward_kv_reprs(
        reprs.wr,
        RT::model_weights_architecture_repr_of(&old_e.weights_perms@),
        reprs.input_ids, reprs.positions, pre_kv, reprs.slots,
        reprs.cu_q, reprs.cu_k, reprs.max_q, reprs.max_k, reprs.bt,
    )
}


// Architecture-neutral structural step relation, parameterized only by the
// model payload consumed by sampling and produced for KV persistence. The
// scheduler, request lifecycle, layout, and page-ownership clauses do not
// inspect a family-specific layer fold.
// Per-row slot segments of the plan, transported to the step boundary.
// Decode rows carry one slot (the block-table slot of the
// last pre-step history position, against the pre-step residency, which commit
// only extends); admitted surviving rows carry the uncached suffix
// pointwise against the post-step residency. A named per-k predicate lets the
// quantifier folds through one application.
pub open spec fn reprs_row_slots_at(
    old_e: Engine,
    new_e: Engine,
    reprs: StepReprs,
    k: int,
) -> bool {
    let srid = reprs.scheduled[k];
    let s0 = reprs.cu_q[k];
    let s1 = reprs.cu_q[k + 1];
    0 <= s0 && s1 <= reprs.slots.len() as int
    && if old_e.cs.running@.contains(srid) {
        s1 == s0 + 1
        && (old_e.cs.live_requests@.contains_key(srid)
            ==> old_e.cs.request_residency@.contains_key(srid)
                && reprs.slots[s0] == crate::proof::tensor::geometry::block_table_slot(
                    old_e.cs.request_residency@[srid].block_ids@,
                    (crate::exec::request_state::history(
                        old_e.cs.live_requests@[srid]).len() - 1) as nat) as int)
    } else if reprs.sample_mask[k] {
        new_e.cs.live_requests@.contains_key(srid) ==> {
            let c = new_e.cs.request_residency@[srid].cached_prefix_blocks as int
                * (crate::types::BLOCK_SIZE_SPEC as int);
            let n = new_e.cs.live_requests@[srid].prompt_tokens@.len() as int;
            &&& new_e.cs.request_residency@.contains_key(srid)
            &&& s1 == s0 + (n - c)
            &&& n - c >= 1
            &&& (forall|q: int| s0 <= q < s1
                ==> #[trigger] reprs.slots[q] == crate::proof::tensor::geometry::block_table_slot(
                    new_e.cs.request_residency@[srid].block_ids@,
                    (c + q - s0) as nat) as int)
        }
    } else {
        true
    }
}

// A scheduled row's slots never land in a page that another request's post-step
// block table can reach within its covered prefix (positions below its last
// history position). A named per-k predicate supports quantifier folding.
#[verifier::opaque]
pub open spec fn reprs_rows_disjoint_at(new_e: Engine, reprs: StepReprs, k: int) -> bool {
    let s0 = reprs.cu_q[k];
    let s1 = reprs.cu_q[k + 1];
    forall|q: int, r: RequestId, l: int|
        #![trigger reprs.slots[q], new_e.cs.request_residency@[r].block_ids@[l]]
        s0 <= q < s1
        && r != reprs.scheduled[k]
        && new_e.cs.running@.contains(r)
        && new_e.cs.live_requests@.contains_key(r)
        && new_e.cs.request_residency@.contains_key(r)
        && 0 <= l < new_e.cs.request_residency@[r].block_ids@.len()
        && l < crate::proof::tensor::geometry::blocks_needed_for(
            (crate::exec::request_state::history(
                new_e.cs.live_requests@[r]).len() - 1) as nat)
        ==> reprs.slots[q] / (crate::types::BLOCK_SIZE_SPEC as int)
            != new_e.cs.request_residency@[r].block_ids@[l] as int
}

pub open spec fn reprs_rows_disjoint(new_e: Engine, reprs: StepReprs) -> bool {
    forall|k: int| 0 <= k < reprs.scheduled.len()
        ==> #[trigger] reprs_rows_disjoint_at(new_e, reprs, k)
}

pub open spec fn reprs_row_slots_ok(old_e: Engine, new_e: Engine, reprs: StepReprs) -> bool {
    forall|k: int| 0 <= k < reprs.scheduled.len()
        ==> #[trigger] reprs_row_slots_at(old_e, new_e, reprs, k)
}

// Stable write/write separation for the materialized plan. Different request
// rows never scatter into the same physical KV page. Unlike the post-commit
// residency predicate below, this statement survives even when one of the
// requests finishes and its residency is removed during commit.
#[verifier::opaque]
pub open spec fn reprs_write_pages_disjoint(reprs: StepReprs) -> bool {
    forall|k: int, m: int, q: int, p: int|
        #![trigger reprs.cu_q[k], reprs.cu_q[m], reprs.slots[q], reprs.slots[p]]
        0 <= k < reprs.scheduled.len()
        && 0 <= m < reprs.scheduled.len()
        && k != m
        && reprs.cu_q[k] <= q < reprs.cu_q[k + 1]
        && reprs.cu_q[m] <= p < reprs.cu_q[m + 1]
        ==> reprs.slots[q] / (crate::types::BLOCK_SIZE_SPEC as int)
            != reprs.slots[p] / (crate::types::BLOCK_SIZE_SPEC as int)
}

// Stable isolation for every materialized plan row's complete physical block
// table.  Unlike post-commit residency facts, this remains meaningful when a
// scheduled request finishes and its residency is removed by commit.
#[verifier::opaque]
pub open spec fn reprs_other_writes_miss_plan_rows(
    reprs: StepReprs,
) -> bool {
    forall|k: int, m: int, p: int, l: int|
        #![trigger reprs.bt[k][l], reprs.cu_q[m], reprs.slots[p]]
        0 <= k < reprs.scheduled.len()
        && 0 <= m < reprs.scheduled.len()
        && m != k
        && reprs.cu_q[m] <= p < reprs.cu_q[m + 1]
        && 0 <= l < reprs.bt[k].len()
        ==> reprs.slots[p] / (crate::types::BLOCK_SIZE_SPEC as int)
            != reprs.bt[k][l] as int
}

pub open spec fn engine_step_relation_with_payload(
    old_e: Engine,
    new_e: Engine,
    emitted: Map<RequestId, TokenId>,
    samples: Map<RequestId, (crate::exec::request_state::SamplerState, TokenId)>,
    reprs: StepReprs,
    logits_repr: Tensor2D,
    post_kv: Seq<(KVCacheLayerRepr, KVCacheLayerRepr)>,
) -> bool {
    cs_valid(&new_e.cs)
    && new_e.model_config == old_e.model_config
    && RT::model_weights_repr_of(&new_e.weights_perms@) == reprs.wr
    && new_e.cs.num_blocks == old_e.cs.num_blocks
    && new_e.cs.accepted_requests@ == old_e.cs.accepted_requests@
    && new_e.kv_caches@.len() == old_e.kv_caches@.len()
    && new_e.kv_caches_repr@.len() == old_e.kv_caches_repr@.len()
    && emitted.dom().subset_of(old_e.cs.live_requests@.dom())
    && (forall|r: RequestId|
        #[trigger] new_e.cs.live_requests@.contains_key(r)
        ==> old_e.cs.live_requests@.contains_key(r))
    // Per-request state evolution (from `commit`): each surviving emitted
    // request takes the full transition; every unemitted request is unchanged.
    // A KV-only scheduled row is therefore an abstract stutter.
    && (forall|r: RequestId|
        #[trigger] new_e.cs.live_requests@.contains_key(r) ==>
            if emitted.contains_key(r) {
                crate::proof::reference::request_machine::machine_step_transition_full(
                    old_e.cs.live_requests@[r], new_e.cs.live_requests@[r],
                    samples[r].0, samples[r].1)
            } else {
                new_e.cs.live_requests@[r] == old_e.cs.live_requests@[r]
            })
    // The emitted tokens are the sample-map tokens.
    && (forall|r: RequestId| #[trigger] emitted.contains_key(r) ==>
        samples.contains_key(r)
        && emitted[r] == samples[r].1)
    // Unlike the earlier structural relation, the sample map is not merely a
    // shared input: `step_core` binds every entry to the exact selected row of
    // this step's verified batched forward.
    && engine_samples_match_logits(old_e, samples, reprs, logits_repr)
    // Survival characterization (from `commit`): an old live request survives iff
    // it is not a scheduled request that finished after appending its token.
    // Mirrors `ibm_step`'s removal rule, so the live keysets stay equal.
    && (forall|r: RequestId| #[trigger] old_e.cs.live_requests@.contains_key(r) ==>
        (new_e.cs.live_requests@.contains_key(r) <==>
            !(emitted.contains_key(r)
              && crate::exec::request_state::should_finish_after_append(
                  old_e.cs.live_requests@[r], emitted[r]))))
    // Scheduled (emitted) requests were steppable pre-step (from `plan`): the
    // engine only schedules requests that `can_step`.  Lets the abstract step's
    // `can_step(pre)` obligation be discharged on the matching machine.
    && (forall|r: RequestId| #[trigger] emitted.contains_key(r) ==>
        crate::exec::request_state::can_step(old_e.cs.live_requests@[r]))
    // Surviving live requests are well-formed (from `commit`/`cs_valid`): their
    // post-step states stay valid + have non-empty history.  Lets the abstract
    // step's `request_machine_alive` obligation be discharged.
    && (forall|r: RequestId| #[trigger] new_e.cs.live_requests@.contains_key(r) ==>
        crate::exec::request_state::valid_request_state(new_e.cs.live_requests@[r])
        && crate::exec::request_state::history(new_e.cs.live_requests@[r]).len() > 0)
    // Engine-side post-store cache: the new per-layer ghost cache
    // is exactly the batched cache `model_forward` produces over this step's plan
    // (pre-cache = old ghost cache). `step` calls the verified `model_forward`,
    // whose `ensures` characterizes the
    // full post-store cache as `model_forward_kv_reprs`.
    && new_e.kv_caches_repr@ == post_kv
    // Plan-shape well-formedness on the ghost reprs, plus the
    // row-mask/emitted alignment. Both are proven from the verified
    // plan/step_core/commit ensures.
    && step_reprs_wf(old_e, reprs)
    && reprs_sample_policy(old_e, reprs)
    && step_reprs_block_tables_bounded(old_e, reprs)
    && reprs_forward_layout_ok(old_e, reprs)
    && reprs_cached_prefix_origins(old_e, reprs)
    && reprs_prefill_residencies(old_e, new_e, reprs)
    // The page origin is prefix-closed: an inherited child cannot sit above a
    // parent block ID that was evicted and reused by this admission step.
    && positive_chains_from_pre_or_executed_rows(
        &old_e.cs, &new_e.cs, reprs.scheduled, reprs.bt, reprs.cu_k,
    )
    // Every newly published full page below the admission row's `k_len`
    // retains its exact physical parent chain and tokens across commit, even if that request
    // finishes and loses its residency in the same step.
    && published_admission_row_prefixes(
        &old_e.cs, &new_e.cs, reprs.scheduled, reprs.bt, reprs.cu_k,
    )
    && (forall|r: RequestId| #[trigger] old_e.cs.live_requests@.contains_key(r) ==>
        (emitted.contains_key(r) <==> reprs_emits(reprs, r)))
    // Scheduled requests originate in a pre-step queue: decode
    // rows from `running`, admissions from `waiting`.  Proven from the
    // verified plan's origin export.
    && (forall|r: RequestId| #[trigger] reprs.scheduled.contains(r) ==>
        old_e.cs.running@.contains(r) || old_e.cs.waiting@.contains(r))
    // Running-queue evolution: post-step running is the old
    // queue plus this step's schedule, minus the finished.  Proven from
    // the plan and commit exports.
    && (forall|r: RequestId| #[trigger] new_e.cs.running@.contains(r)
        <==> ((old_e.cs.running@.contains(r) || reprs.scheduled.contains(r))
            && !reprs_parks(reprs, r)
            && !(emitted.contains_key(r)
                && crate::exec::request_state::should_finish_after_append(
                    old_e.cs.live_requests@[r], emitted[r]))))
    // Residency evolution: an unscheduled old-running
    // request keeps its residency verbatim; a scheduled old-running
    // survivor's block table extends its old one (append-only). Proven
    // from the plan running-preservation + commit residency exports.
    && (forall|r: RequestId| #[trigger] old_e.cs.running@.contains(r)
            && !reprs.scheduled.contains(r)
        ==> old_e.cs.request_residency@.contains_key(r)
            && new_e.cs.request_residency@.contains_key(r)
            && new_e.cs.request_residency@[r] == old_e.cs.request_residency@[r])
    && (forall|r: RequestId| #[trigger] reprs.scheduled.contains(r)
            && old_e.cs.running@.contains(r)
            && reprs_emits(reprs, r)
            && new_e.cs.request_residency@.contains_key(r)
        ==> old_e.cs.request_residency@.contains_key(r)
            && new_e.cs.request_residency@[r].block_ids@.len()
                >= old_e.cs.request_residency@[r].block_ids@.len()
            && new_e.cs.request_residency@[r].block_ids@.subrange(0,
                old_e.cs.request_residency@[r].block_ids@.len() as int)
                == old_e.cs.request_residency@[r].block_ids@)
    // The plan's per-row slot segments.
    && reprs_row_slots_ok(old_e, new_e, reprs)
    // Stable write/write page separation, including rows whose requests
    // finish and lose their residencies during commit.
    && reprs_write_pages_disjoint(reprs)
    && reprs_other_writes_miss_plan_rows(reprs)
    // Cross-request page disjointness.
    && reprs_rows_disjoint(new_e, reprs)
}

// Architecture-dispatched serving relation. This uses the same structural
// clauses and the same `StepReprs`; only the opaque whole-model logits and KV
// payload are selected from the closed architecture representation.
pub open spec fn architecture_engine_step_relation(
    old_e: Engine,
    new_e: Engine,
    emitted: Map<RequestId, TokenId>,
    samples: Map<RequestId, (crate::exec::request_state::SamplerState, TokenId)>,
    reprs: StepReprs,
) -> bool {
    engine_step_relation_with_payload(
        old_e, new_e, emitted, samples, reprs,
        architecture_step_logits_repr(old_e, reprs),
        architecture_engine_post_kv_of(
            old_e, reprs, old_e.kv_caches_repr@,
        ),
    )
}


// Architecture-neutral introduction rule for the structural relation.  The
// caller supplies only the two model payloads explicitly; all scheduler,
// lifecycle, layout, and page-ownership obligations are shared.
#[verifier::spinoff_prover]
pub proof fn lemma_engine_step_relation_with_payload_intro(
    old_e: Engine,
    new_e: Engine,
    emitted: Map<RequestId, TokenId>,
    samples: Map<RequestId, (SamplerState, TokenId)>,
    reprs: StepReprs,
    logits_repr: Tensor2D,
    post_kv: Seq<(KVCacheLayerRepr, KVCacheLayerRepr)>,
)
    requires
        cs_valid(&new_e.cs),
        new_e.model_config == old_e.model_config,
        RT::model_weights_repr_of(&new_e.weights_perms@) == reprs.wr,
        new_e.cs.num_blocks == old_e.cs.num_blocks,
        new_e.cs.accepted_requests@ == old_e.cs.accepted_requests@,
        new_e.kv_caches@.len() == old_e.kv_caches@.len(),
        new_e.kv_caches_repr@.len() == old_e.kv_caches_repr@.len(),
        emitted.dom().subset_of(old_e.cs.live_requests@.dom()),
        forall|r: RequestId| #[trigger] new_e.cs.live_requests@.contains_key(r)
            ==> old_e.cs.live_requests@.contains_key(r),
        forall|r: RequestId| #[trigger] new_e.cs.live_requests@.contains_key(r)
            ==> if emitted.contains_key(r) {
                crate::proof::reference::request_machine::machine_step_transition_full(
                    old_e.cs.live_requests@[r], new_e.cs.live_requests@[r],
                    samples[r].0, samples[r].1,
                )
            } else {
                new_e.cs.live_requests@[r] == old_e.cs.live_requests@[r]
            },
        forall|r: RequestId| #[trigger] emitted.contains_key(r)
            ==> samples.contains_key(r) && emitted[r] == samples[r].1,
        engine_samples_match_logits(old_e, samples, reprs, logits_repr),
        forall|r: RequestId| #[trigger] old_e.cs.live_requests@.contains_key(r)
            ==> (new_e.cs.live_requests@.contains_key(r) <==>
                !(emitted.contains_key(r)
                    && should_finish_after_append(
                        old_e.cs.live_requests@[r], emitted[r],
                    ))),
        forall|r: RequestId| #[trigger] emitted.contains_key(r)
            ==> can_step(old_e.cs.live_requests@[r]),
        forall|r: RequestId| #[trigger] new_e.cs.live_requests@.contains_key(r)
            ==> valid_request_state(new_e.cs.live_requests@[r])
                && history(new_e.cs.live_requests@[r]).len() > 0,
        new_e.kv_caches_repr@ == post_kv,
        step_reprs_wf(old_e, reprs),
        reprs_sample_policy(old_e, reprs),
        step_reprs_block_tables_bounded(old_e, reprs),
        reprs_forward_layout_ok(old_e, reprs),
        reprs_cached_prefix_origins(old_e, reprs),
        reprs_prefill_residencies(old_e, new_e, reprs),
        positive_chains_from_pre_or_executed_rows(
            &old_e.cs, &new_e.cs, reprs.scheduled, reprs.bt, reprs.cu_k,
        ),
        published_admission_row_prefixes(
            &old_e.cs, &new_e.cs, reprs.scheduled, reprs.bt, reprs.cu_k,
        ),
        forall|r: RequestId| #[trigger] old_e.cs.live_requests@.contains_key(r)
            ==> (emitted.contains_key(r) <==> reprs_emits(reprs, r)),
        forall|r: RequestId| #[trigger] reprs.scheduled.contains(r)
            ==> old_e.cs.running@.contains(r) || old_e.cs.waiting@.contains(r),
        forall|r: RequestId| #[trigger] new_e.cs.running@.contains(r)
            <==> ((old_e.cs.running@.contains(r) || reprs.scheduled.contains(r))
                && !reprs_parks(reprs, r)
                && !(emitted.contains_key(r)
                    && should_finish_after_append(
                        old_e.cs.live_requests@[r], emitted[r],
                    ))),
        forall|r: RequestId| #[trigger] old_e.cs.running@.contains(r)
                && !reprs.scheduled.contains(r)
            ==> old_e.cs.request_residency@.contains_key(r)
                && new_e.cs.request_residency@.contains_key(r)
                && new_e.cs.request_residency@[r]
                    == old_e.cs.request_residency@[r],
        forall|r: RequestId| #[trigger] reprs.scheduled.contains(r)
                && old_e.cs.running@.contains(r)
                && reprs_emits(reprs, r)
                && new_e.cs.request_residency@.contains_key(r)
            ==> old_e.cs.request_residency@.contains_key(r)
                && new_e.cs.request_residency@[r].block_ids@.len()
                    >= old_e.cs.request_residency@[r].block_ids@.len()
                && new_e.cs.request_residency@[r].block_ids@.subrange(
                    0,
                    old_e.cs.request_residency@[r].block_ids@.len() as int,
                ) == old_e.cs.request_residency@[r].block_ids@,
        reprs_row_slots_ok(old_e, new_e, reprs),
        reprs_write_pages_disjoint(reprs),
        reprs_other_writes_miss_plan_rows(reprs),
        reprs_rows_disjoint(new_e, reprs),
    ensures
        engine_step_relation_with_payload(
            old_e, new_e, emitted, samples, reprs, logits_repr, post_kv,
        ),
{
}

// A live request either remains byte-for-byte unchanged or appends exactly one
// sampled token.  The survival clause removes precisely the append that would
// finish the request, while prompt/max-token fields remain fixed.  Therefore
// the static history-capacity budget is an inductive step invariant.
pub proof fn lemma_live_request_step_ready_preserved_with_payload(
    old_e: Engine,
    new_e: Engine,
    emitted: Map<RequestId, TokenId>,
    samples: Map<RequestId, (crate::exec::request_state::SamplerState, TokenId)>,
    reprs: StepReprs,
    logits_repr: Tensor2D,
    post_kv: Seq<(KVCacheLayerRepr, KVCacheLayerRepr)>,
)
    requires
        engine_step_relation_with_payload(
            old_e, new_e, emitted, samples, reprs, logits_repr, post_kv,
        ),
        live_request_step_ready(&old_e.cs),
    ensures
        live_request_step_ready(&new_e.cs),
{
    assert forall|rid: RequestId|
        #[trigger] new_e.cs.live_requests@.contains_key(rid)
        implies can_step(new_e.cs.live_requests@[rid])
            && request_history_capacity_safe(new_e.cs.live_requests@[rid])
    by {
        assert(old_e.cs.live_requests@.contains_key(rid));
        let pre = old_e.cs.live_requests@[rid];
        let post = new_e.cs.live_requests@[rid];
        assert(can_step(pre));
        assert(request_history_capacity_safe(pre));
        if emitted.contains_key(rid) {
            assert(crate::proof::reference::request_machine::machine_step_transition_full(
                pre, post, samples[rid].0, samples[rid].1,
            ));
            assert(!should_finish_after_append(pre, emitted[rid])) by {
                assert(old_e.cs.live_requests@.contains_key(rid));
            }
            crate::proof::reference::request_machine::lemma_machine_step_survivor_ready(
                pre, post, samples[rid].0, emitted[rid],
            );
            crate::proof::reference::request_machine::lemma_machine_step_preserves_history_capacity(
                pre, post, samples[rid].0, emitted[rid],
            );
        } else {
            assert(post == pre);
        }
    }
}

// Package the post-commit survivor fold outside `Engine::step`'s already-large
// solver context.  The scheduler supplies structural transition and survival
// facts; the request-machine lemma supplies validity for the emitted case.
#[verifier::spinoff_prover]
pub proof fn lemma_commit_survivors_valid(
    before: &CacheScheduler,
    after: &CacheScheduler,
    emitted: Map<RequestId, TokenId>,
    samples: Map<RequestId, (SamplerState, TokenId)>,
)
    requires
        forall|r: RequestId| #[trigger] after.live_requests@.contains_key(r)
            ==> before.live_requests@.contains_key(r),
        forall|r: RequestId| #[trigger] after.live_requests@.contains_key(r)
            ==> if emitted.contains_key(r) {
                crate::proof::reference::request_machine::machine_step_transition_full(
                    before.live_requests@[r], after.live_requests@[r],
                    samples[r].0, samples[r].1,
                )
            } else {
                after.live_requests@[r] == before.live_requests@[r]
            },
        forall|r: RequestId| #[trigger] before.live_requests@.contains_key(r)
            ==> (after.live_requests@.contains_key(r) <==>
                !(emitted.contains_key(r)
                    && should_finish_after_append(
                        before.live_requests@[r], emitted[r],
                    ))),
        forall|r: RequestId| #[trigger] emitted.contains_key(r)
            ==> samples.contains_key(r)
                && emitted[r] == samples[r].1
                && can_step(before.live_requests@[r]),
        forall|r: RequestId| #[trigger] before.live_requests@.contains_key(r)
            ==> valid_request_state(before.live_requests@[r]),
    ensures
        forall|r: RequestId| #[trigger] after.live_requests@.contains_key(r)
            ==> valid_request_state(after.live_requests@[r])
                && history(after.live_requests@[r]).len() > 0,
{
    assert forall|r: RequestId| #[trigger] after.live_requests@.contains_key(r)
        implies valid_request_state(after.live_requests@[r])
            && history(after.live_requests@[r]).len() > 0
    by {
        assert(before.live_requests@.contains_key(r));
        if emitted.contains_key(r) {
            assert(!should_finish_after_append(
                before.live_requests@[r], emitted[r],
            ));
            crate::proof::reference::request_machine::lemma_machine_step_survivor_ready(
                before.live_requests@[r],
                after.live_requests@[r],
                samples[r].0,
                emitted[r],
            );
        } else {
            assert(after.live_requests@[r] == before.live_requests@[r]);
            assert(before.live_requests@[r].prompt_tokens@.len() > 0);
        }
    }
}


pub proof fn lemma_architecture_live_request_step_ready_preserved(
    old_e: Engine,
    new_e: Engine,
    emitted: Map<RequestId, TokenId>,
    samples: Map<RequestId, (crate::exec::request_state::SamplerState, TokenId)>,
    reprs: StepReprs,
)
    requires
        architecture_engine_step_relation(
            old_e, new_e, emitted, samples, reprs,
        ),
        live_request_step_ready(&old_e.cs),
    ensures
        live_request_step_ready(&new_e.cs),
{
    lemma_live_request_step_ready_preserved_with_payload(
        old_e,
        new_e,
        emitted,
        samples,
        reprs,
        architecture_step_logits_repr(old_e, reprs),
        architecture_engine_post_kv_of(
            old_e, reprs, old_e.kv_caches_repr@,
        ),
    );
}

// The materialized plan's writes miss every other row's complete block table.
// Kept as a separate proof boundary because both graph-cover readiness and the
// post-forward cache-fidelity argument consume the same stable geometry.
#[verifier::spinoff_prover]
proof fn lemma_plan_other_writes_miss_plan_rows(
    pre: &CacheScheduler,
    post: &CacheScheduler,
    plan: &StepPlan,
    reprs: StepReprs,
)
    requires
        cs_valid(post),
        residency_history_aligned(post),
        pre_commit_tails_exclusive(post),
        step_plan_shape_ok(plan),
        post.live_requests@ == pre.live_requests@,
        forall|k: int| 0 <= k < plan.scheduled_ids@.len() ==>
            #[trigger] post.running@.contains(plan.scheduled_ids@[k]),
        plan_slot_segments_ok(pre, post, plan),
        plan_forward_layout_ok(pre, plan),
        forall|k: int| 0 <= k < plan.scheduled_ids@.len()
            && !pre.running@.contains(plan.scheduled_ids@[k]) ==>
            #[trigger] admitted_pages_exclusive_post(
                post, plan.scheduled_ids@, k,
            ),
        plan.scheduled_ids@.no_duplicates(),
        reprs.scheduled == plan.scheduled_ids@,
        reprs.cu_q == plan.cu_seqlens_q_repr@,
        reprs.slots == plan.slot_mapping_repr@,
        reprs.bt == plan.block_table_repr@,
    ensures
        reprs_other_writes_miss_plan_rows(reprs),
        reprs_write_pages_disjoint(reprs),
{
    reveal(reprs_other_writes_miss_plan_rows);
    assert forall|k: int, m: int, p: int, l: int|
        #![trigger reprs.bt[k][l], reprs.cu_q[m], reprs.slots[p]]
        0 <= k < reprs.scheduled.len()
        && 0 <= m < reprs.scheduled.len()
        && m != k
        && reprs.cu_q[m] <= p < reprs.cu_q[m + 1]
        && 0 <= l < reprs.bt[k].len()
        implies reprs.slots[p] / (crate::types::BLOCK_SIZE_SPEC as int)
            != reprs.bt[k][l] as int
    by {
        assert(plan_slot_segments_at(pre, post, plan, k));
        assert(plan_slot_segments_at(pre, post, plan, m));
        assert(plan_forward_layout_at(pre, plan, m));
        lemma_plan_slot_page_exclusive(pre, post, plan, m, p);
        let rid_k = reprs.scheduled[k];
        let rid_m = reprs.scheduled[m];
        assert(rid_k != rid_m) by {
            reveal(Seq::no_duplicates);
        }
        reveal(plan_slot_segments_at);
        assert(reprs.bt[k]
            == post.request_residency@[rid_k].block_ids@);
        let target = reprs.bt[k][l];
        assert(post.request_residency@[rid_k]
            .block_ids@.contains(target));
        let page = reprs.slots[p]
            / (crate::types::BLOCK_SIZE_SPEC as int);
        let writer_bid = page as BlockId;
        if page == target as int {
            assert(writer_bid == target);
            assert(post.request_residency@[rid_m]
                .block_ids@.contains(target));
            let holders = residency_holders_of(post, target);
            assert(holders.contains(rid_k));
            assert(holders.contains(rid_m));
            lemma_two_holders(holders, rid_k, rid_m);
            assert(post.blocks@[target].refcount as int
                == holders.len() as int);
            assert(post.blocks@[target].refcount == 1);
            assert(false);
        }
    }
    assert forall|k: int, q: int| #![trigger reprs.cu_q[k], reprs.slots[q]]
        0 <= k < reprs.scheduled.len() && reprs.cu_q[k] <= q < reprs.cu_q[k + 1]
        implies {
            let page = reprs.slots[q] / (crate::types::BLOCK_SIZE_SPEC as int);
            let bid = page as BlockId;
            &&& reprs.bt[k].contains(bid)
            &&& page == bid as int
        }
    by {
        assert(plan_slot_segments_at(pre, post, plan, k));
        assert(plan_forward_layout_at(pre, plan, k));
        lemma_plan_slot_page_exclusive(pre, post, plan, k, q);
        reveal(plan_slot_segments_at);
    }
    lemma_write_page_separation_from_row_coverage(reprs);
}

// Pure row geometry: keep the four-index implication separate from scheduler
// invariants and their ownership/queue quantifiers.
#[verifier::spinoff_prover]
proof fn lemma_write_page_separation_from_row_coverage(reprs: StepReprs)
    requires
        reprs_other_writes_miss_plan_rows(reprs),
        forall|k: int, q: int| #![trigger reprs.cu_q[k], reprs.slots[q]]
            0 <= k < reprs.scheduled.len() && reprs.cu_q[k] <= q < reprs.cu_q[k + 1]
            ==> {
                let page = reprs.slots[q] / (crate::types::BLOCK_SIZE_SPEC as int);
                let bid = page as BlockId;
                &&& reprs.bt[k].contains(bid)
                &&& page == bid as int
            },
    ensures reprs_write_pages_disjoint(reprs),
{
    reveal(reprs_write_pages_disjoint);
    assert forall|k: int, m: int, q: int, p: int|
        #![trigger reprs.cu_q[k], reprs.cu_q[m], reprs.slots[q], reprs.slots[p]]
        0 <= k < reprs.scheduled.len() && 0 <= m < reprs.scheduled.len() && k != m
        && reprs.cu_q[k] <= q < reprs.cu_q[k + 1]
        && reprs.cu_q[m] <= p < reprs.cu_q[m + 1]
        implies reprs.slots[q] / (crate::types::BLOCK_SIZE_SPEC as int)
            != reprs.slots[p] / (crate::types::BLOCK_SIZE_SPEC as int)
    by {
        lemma_write_page_separation_at(reprs, k, m, q, p);
    }
}

proof fn lemma_write_page_separation_at(reprs: StepReprs, k: int, m: int, q: int, p: int)
    requires
        reprs_other_writes_miss_plan_rows(reprs),
        0 <= k < reprs.scheduled.len(),
        0 <= m < reprs.scheduled.len(),
        k != m,
        reprs.cu_q[m] <= p < reprs.cu_q[m + 1],
        reprs.bt[k].contains((reprs.slots[q] / (crate::types::BLOCK_SIZE_SPEC as int)) as BlockId),
        reprs.slots[q] / (crate::types::BLOCK_SIZE_SPEC as int)
            == ((reprs.slots[q] / (crate::types::BLOCK_SIZE_SPEC as int)) as BlockId) as int,
    ensures
        reprs.slots[q] / (crate::types::BLOCK_SIZE_SPEC as int)
            != reprs.slots[p] / (crate::types::BLOCK_SIZE_SPEC as int),
{
    reveal(reprs_other_writes_miss_plan_rows);
    let page = reprs.slots[q] / (crate::types::BLOCK_SIZE_SPEC as int);
    let bid = page as BlockId;
    let l = reprs.bt[k].index_of(bid);
    assert(reprs.bt[k][l] == bid);
    assert(reprs.slots[p] / (crate::types::BLOCK_SIZE_SPEC as int) != reprs.bt[k][l] as int);
}

// Verified core of `step`: batched forward + per-request
// sampling.  The returned sample map is BOUND to the verified pipeline: for
// each true-mask scheduled index `i` whose rid is live, the sample is exactly
// `sample_from_repr(select_sample_logits_repr(model_forward_logits_repr(…),
// cu_q, i), live[rid].sampler_state)`. The batched sampling call still
// computes a candidate for every row; false-mask candidates are omitted from
// the returned map. The verified scheduler supplies the plan-shape facts
// below; `Engine::step` composes this function with verified `plan` and
// `commit`.
#[verifier::spinoff_prover]
#[verifier::rlimit(200)]
pub fn step_core(
    model_config: &ModelConfig,
    weights: &RT::ModelWeights,
    runtime: &RT::ModelRuntime,
    Tracked(wp): Tracked<&RT::ModelWeightsPerms>,
    kv_caches: &Vec<(RT::Tensor, RT::Tensor)>,
    Tracked(kv_perms): Tracked<&mut RT::KVCachePerms>,
    plan: &StepPlan,
    Tracked(plan_perms): Tracked<&StepPlanPerms>,
    live_requests: &vstd::hash_map::HashMapWithView<u64, RequestState>,
    graph_overlay: Option<&RT::CudaGraphOverlay>,
) -> (samples: SampleResults)
    requires
        graph_overlay.is_some() ==> RT::cuda_graph_replay_fidelity(),
        RT::paged_attention_numeric_domain(),
        graph_overlay.is_some() ==> model_config.num_layers > 0,
        graph_overlay.is_some() ==>
            MODEL_FAMILIES::cuda_graph_overlay_supported(wp),
        obeys_key_model::<u64>(),
        RT::model_execution_valid(weights, runtime, wp),
        RT::model_weights_num_layers(weights) == model_config.num_layers as nat,
        step_plan_perms_valid(plan, *plan_perms),
        kv_caches.len() == model_config.num_layers as nat,
        kv_perms.len() == model_config.num_layers as nat,
        RT::kv_perms_ids_distinct(*kv_perms),
        kv_perms.extracted() == Set::<int>::empty(),
        RT::kv_cache_tensor_ids_match(
            kv_caches@, *old(kv_perms), model_config.num_layers as nat,
        ),
        // Established by the verified `plan` through `step_plan_shape_ok`.
        step_plan_shape_ok(plan),
        plan.scheduled_ids@.no_duplicates(),
        crate::exec::model::architecture_model_forward_ready(
            wp, plan.input_ids_repr@, plan.positions_repr@,
            Seq::new(model_config.num_layers as nat, |i: int|
                (kv_perms.k_repr(i), kv_perms.v_repr(i))),
            plan.slot_mapping_repr@,
            plan.cu_seqlens_q_repr@, plan.cu_seqlens_k_repr@,
            plan.max_seqlen_q as nat, plan.max_seqlen_k as nat,
            plan.block_table_repr@, plan.scheduled_ids@.len(),
        ),
        graph_overlay.is_some()
            && plan.input_ids_repr@.len() == plan.scheduled_ids@.len()
            && plan.max_seqlen_q as nat == 1 ==>
                MODEL_FAMILIES::cuda_graph_decode_cover_ready(
                wp, RT::model_weights_repr_of(wp),
                plan.input_ids_repr@, plan.positions_repr@,
                Seq::new(model_config.num_layers as nat, |i: int|
                    (old(kv_perms).k_repr(i), old(kv_perms).v_repr(i))),
                plan.slot_mapping_repr@,
                plan.cu_seqlens_q_repr@, plan.cu_seqlens_k_repr@,
                plan.max_seqlen_q as nat, plan.max_seqlen_k as nat,
                plan.block_table_repr@,
            ),
    ensures
        final(kv_perms).len() == old(kv_perms).len(),
        final(kv_perms).extracted() == Set::<int>::empty(),
        RT::kv_perms_ids_distinct(*final(kv_perms)),
        RT::kv_cache_tensor_ids_match(
            kv_caches@, *final(kv_perms), model_config.num_layers as nat,
        ),
        forall|j: int| 0 <= j < model_config.num_layers as int ==>
            #[trigger] final(kv_perms).k_id(j) == old(kv_perms).k_id(j),
        forall|j: int| 0 <= j < model_config.num_layers as int ==>
            #[trigger] final(kv_perms).v_id(j) == old(kv_perms).v_id(j),
        ({
        let pre_kv: Seq<(KVCacheLayerRepr, KVCacheLayerRepr)> =
            Seq::new(model_config.num_layers as nat, |i: int|
                (old(kv_perms).k_repr(i), old(kv_perms).v_repr(i)));
        let architecture_repr = RT::model_weights_architecture_repr_of(wp);
        let post_kv = MA::model_forward_kv_reprs(
            RT::model_weights_repr_of(wp),
            architecture_repr,
            plan.input_ids_repr@, plan.positions_repr@,
            pre_kv, plan.slot_mapping_repr@,
            plan.cu_seqlens_q_repr@, plan.cu_seqlens_k_repr@,
            plan.max_seqlen_q as nat, plan.max_seqlen_k as nat,
            plan.block_table_repr@);
        forall|j: int| 0 <= j < model_config.num_layers as int ==>
            #[trigger] final(kv_perms).k_repr(j) == post_kv[j].0
        }),
        ({
        let pre_kv: Seq<(KVCacheLayerRepr, KVCacheLayerRepr)> =
            Seq::new(model_config.num_layers as nat, |i: int|
                (old(kv_perms).k_repr(i), old(kv_perms).v_repr(i)));
        let architecture_repr = RT::model_weights_architecture_repr_of(wp);
        let post_kv = MA::model_forward_kv_reprs(
            RT::model_weights_repr_of(wp),
            architecture_repr,
            plan.input_ids_repr@, plan.positions_repr@,
            pre_kv, plan.slot_mapping_repr@,
            plan.cu_seqlens_q_repr@, plan.cu_seqlens_k_repr@,
            plan.max_seqlen_q as nat, plan.max_seqlen_k as nat,
            plan.block_table_repr@);
        forall|j: int| 0 <= j < model_config.num_layers as int ==>
            #[trigger] final(kv_perms).v_repr(j) == post_kv[j].1
        }),
        ({
        let pre_kv: Seq<(KVCacheLayerRepr, KVCacheLayerRepr)> =
            Seq::new(model_config.num_layers as nat, |i: int|
                (old(kv_perms).k_repr(i), old(kv_perms).v_repr(i)));
        let architecture_repr = RT::model_weights_architecture_repr_of(wp);
        let logits_repr = MA::model_forward_logits_repr(
            RT::model_weights_repr_of(wp),
            architecture_repr,
            plan.input_ids_repr@, plan.positions_repr@,
            pre_kv, plan.slot_mapping_repr@,
            plan.cu_seqlens_q_repr@, plan.cu_seqlens_k_repr@,
            plan.max_seqlen_q as nat, plan.max_seqlen_k as nat,
            plan.block_table_repr@);
        step_core_samples_match(plan, live_requests@, logits_repr, samples@)
        }),
{
    let ghost pre_kv: Seq<(KVCacheLayerRepr, KVCacheLayerRepr)> =
        Seq::new(model_config.num_layers as nat, |i: int|
            (kv_perms.k_repr(i), kv_perms.v_repr(i)));
    // A graph is an optional overlay on this exact eager call, not a peer
    // backend.  `model_forward_cuda_graph_overlay` can only populate its
    // private graph cache by invoking `model_forward` below under capture;
    // replay assumes that recorded execution has the same contract.
    proof {
        if graph_overlay.is_some() {
            MODEL_FAMILIES::lemma_cuda_graph_overlay_launch_ready(
                wp, plan.input_ids_repr@, plan.positions_repr@, pre_kv,
                plan.slot_mapping_repr@, plan.cu_seqlens_q_repr@,
                plan.cu_seqlens_k_repr@, plan.max_seqlen_q as nat,
                plan.max_seqlen_k as nat, plan.block_table_repr@,
                plan.scheduled_ids@.len(),
            );
            assert forall|i: int| 0 <= i < model_config.num_layers as int implies
                #[trigger] RT::store_kv_cache_launch_ready(
                    plan.input_ids_repr@.len(), old(kv_perms).k_repr(i),
                    old(kv_perms).v_repr(i), plan.slot_mapping_repr@,
                ) by {
                assert(0 <= i < wp.num_layers() as int);
                assert(pre_kv[i].0 == old(kv_perms).k_repr(i));
                assert(pre_kv[i].1 == old(kv_perms).v_repr(i));
            }
            assert forall|i: int| 0 <= i < model_config.num_layers as int implies
                #[trigger] RT::paged_attention_launch_ready(
                    plan.input_ids_repr@.len(), old(kv_perms).k_repr(i),
                    old(kv_perms).v_repr(i),
                    plan.cu_seqlens_q_repr@, plan.cu_seqlens_k_repr@,
                    plan.max_seqlen_q as nat, plan.max_seqlen_k as nat,
                    plan.block_table_repr@,
                ) by {
                assert(0 <= i < wp.num_layers() as int);
                assert(pre_kv[i].0 == old(kv_perms).k_repr(i));
                assert(pre_kv[i].1 == old(kv_perms).v_repr(i));
                }
        }
    }
    let (logits, logits_perm) = match graph_overlay {
        Some(overlay) => {
            let out = model_forward_cuda_graph_overlay(
                overlay,
                &plan.mode,
                model_config,
                weights,
                runtime,
                Tracked(wp),
                &plan.input_ids,
                Tracked(&plan_perms.input_ids),
                &plan.positions,
                Tracked(&plan_perms.positions),
                kv_caches,
                Tracked(kv_perms),
                &plan.block_table,
                Tracked(&plan_perms.block_table),
                &plan.slot_mapping,
                Tracked(&plan_perms.slot_mapping),
                &plan.cu_seqlens_q,
                Tracked(&plan_perms.cu_seqlens_q),
                &plan.cu_seqlens_k,
                Tracked(&plan_perms.cu_seqlens_k),
                plan.max_seqlen_q,
                plan.max_seqlen_k,
                plan.scheduled_ids.len(),
                Ghost(plan.input_ids_repr@),
                Ghost(plan.positions_repr@),
                Ghost(plan.cu_seqlens_q_repr@),
                Ghost(plan.cu_seqlens_k_repr@),
                Ghost(plan.block_table_repr@),
                Ghost(plan.slot_mapping_repr@),
            );
            proof {
                reveal(MODEL_FAMILIES::cuda_graph_overlay_exact_result);
            }
            out
        },
        None => model_forward(
            model_config,
            weights,
            runtime,
            Tracked(wp),
            &plan.input_ids,
            Tracked(&plan_perms.input_ids),
            &plan.positions,
            Tracked(&plan_perms.positions),
            kv_caches,
            Tracked(kv_perms),
            &plan.block_table,
            Tracked(&plan_perms.block_table),
            &plan.slot_mapping,
            Tracked(&plan_perms.slot_mapping),
            &plan.cu_seqlens_q,
            Tracked(&plan_perms.cu_seqlens_q),
            &plan.cu_seqlens_k,
            Tracked(&plan_perms.cu_seqlens_k),
            plan.max_seqlen_q,
            plan.max_seqlen_k,
            plan.scheduled_ids.len(),
            Ghost(plan.input_ids_repr@),
            Ghost(plan.positions_repr@),
            Ghost(plan.cu_seqlens_q_repr@),
            Ghost(plan.cu_seqlens_k_repr@),
            Ghost(plan.block_table_repr@),
            Ghost(plan.slot_mapping_repr@),
        ),
    };
    proof {
        RT::lemma_model_weights_architecture_repr_valid(weights, runtime, wp);
    }
    let ghost architecture_repr = RT::model_weights_architecture_repr_of(wp);
    let ghost logits_repr = MA::model_forward_logits_repr(
        RT::model_weights_repr_of(wp),
        architecture_repr,
        plan.input_ids_repr@, plan.positions_repr@,
        pre_kv, plan.slot_mapping_repr@,
        plan.cu_seqlens_q_repr@, plan.cu_seqlens_k_repr@,
        plan.max_seqlen_q as nat, plan.max_seqlen_k as nat,
        plan.block_table_repr@);
    let ghost post_kv = MA::model_forward_kv_reprs(
        RT::model_weights_repr_of(wp),
        architecture_repr,
        plan.input_ids_repr@, plan.positions_repr@,
        pre_kv, plan.slot_mapping_repr@,
        plan.cu_seqlens_q_repr@, plan.cu_seqlens_k_repr@,
        plan.max_seqlen_q as nat, plan.max_seqlen_k as nat,
        plan.block_table_repr@);
    proof {
        let selected = Seq::new(plan.scheduled_ids@.len(), |i: int|
            RT::select_sample_logits_repr(
                logits_repr, plan.cu_seqlens_q_repr@, i as nat,
            ));
        assert(RT::tensor_repr_2d(logits_perm@, logits, selected));
        assert(kv_perms.len() == pre_kv.len());
        assert(kv_perms.extracted() == Set::<int>::empty());
        assert(RT::kv_perms_ids_distinct(*kv_perms));
        assert(pre_kv.len() == model_config.num_layers as nat);
        MA::lemma_model_forward_kv_reprs_len(
            RT::model_weights_repr_of(wp), architecture_repr,
            plan.input_ids_repr@, plan.positions_repr@,
            pre_kv, plan.slot_mapping_repr@,
            plan.cu_seqlens_q_repr@, plan.cu_seqlens_k_repr@,
            plan.max_seqlen_q as nat, plan.max_seqlen_k as nat,
            plan.block_table_repr@,
        );
        assert(post_kv.len() == model_config.num_layers as nat);
        assert forall|j: int| 0 <= j < model_config.num_layers as int implies
            #[trigger] kv_perms.k_id(j) == old(kv_perms).k_id(j)
            && kv_perms.v_id(j) == old(kv_perms).v_id(j) by {
            assert(0 <= j < pre_kv.len());
            assert(kv_perms.k_id(j) == old(kv_perms).k_id(j));
            assert(kv_perms.v_id(j) == old(kv_perms).v_id(j));
        }
        assert forall|j: int| 0 <= j < model_config.num_layers as int implies
            #[trigger] kv_perms.k_repr(j) == post_kv[j].0
            && kv_perms.v_repr(j) == post_kv[j].1 by {
            assert(0 <= j < pre_kv.len());
            assert(0 <= j < post_kv.len());
            assert(kv_perms.k_repr(j) == post_kv[j].0);
            assert(kv_perms.v_repr(j) == post_kv[j].1);
        }
        assert forall|j: int| 0 <= j < model_config.num_layers as int implies
            #[trigger] kv_perms.v_repr(j) == post_kv[j].1 by {
            assert(kv_perms.k_repr(j) == post_kv[j].0);
        }
    }
    proof {
        MA::lemma_model_forward_logits_repr_shape(
            RT::model_weights_repr_of(wp),
            architecture_repr,
            plan.input_ids_repr@, plan.positions_repr@,
            pre_kv, plan.slot_mapping_repr@,
            plan.cu_seqlens_q_repr@, plan.cu_seqlens_k_repr@,
            plan.max_seqlen_q as nat, plan.max_seqlen_k as nat,
            plan.block_table_repr@);
        MA::lemma_model_forward_kv_reprs_len(
            RT::model_weights_repr_of(wp),
            architecture_repr,
            plan.input_ids_repr@, plan.positions_repr@,
            pre_kv, plan.slot_mapping_repr@,
            plan.cu_seqlens_q_repr@, plan.cu_seqlens_k_repr@,
            plan.max_seqlen_q as nat, plan.max_seqlen_k as nat,
            plan.block_table_repr@);
        assert(logits_repr.len() == plan.input_ids_repr@.len());
        crate::proof::tensor::geometry::lemma_cu_int_bounds(
            plan.cu_seqlens_q_repr@,
            plan.scheduled_ids@.len() as int,
        );
        assert forall|k: int| 0 <= k < plan.scheduled_ids@.len() as int implies {
            &&& #[trigger] plan.cu_seqlens_q_repr@[k + 1] > 0
            &&& plan.cu_seqlens_q_repr@[k + 1] <= logits_repr.len()
        } by {
            assert(0 <= plan.cu_seqlens_q_repr@[k]);
            assert(plan.cu_seqlens_q_repr@[k]
                < plan.cu_seqlens_q_repr@[k + 1]);
            assert(plan.cu_seqlens_q_repr@[k + 1]
                <= plan.cu_seqlens_q_repr@[
                    plan.scheduled_ids@.len() as int
                ]);
        }
        assert(post_kv.len() == model_config.num_layers as nat);
    }
    // Batch the entire sampling tail: model_forward has already gathered one
    // last hidden row per request before lm_head, so this is one argmax
    // reduction and one host readback for the step.
    let ghost states: Seq<crate::exec::request_state::SamplerState> = Seq::new(
        plan.scheduled_ids@.len(),
        |k: int| if live_requests@.contains_key(plan.scheduled_ids@[k]) {
            live_requests@[plan.scheduled_ids@[k]].sampler_state
        } else {
            vstd::pervasive::arbitrary::<crate::exec::request_state::SamplerState>()
        },
    );
    let ghost selected_logits_repr: Tensor2D = logits_perm@.repr_2d();
    proof {
        assert(selected_logits_repr.len() == plan.scheduled_ids@.len());
        assert forall|k: int| 0 <= k < plan.scheduled_ids@.len() as int implies
            #[trigger] selected_logits_repr[k]
                == RT::select_sample_logits_repr(
                    logits_repr,
                    plan.cu_seqlens_q_repr@,
                    k as nat,
                ) by {}
    }
    let tokens = RT::sample_tokens_rows(
        &logits,
        plan.scheduled_ids.len(),
        Tracked(logits_perm.borrow()),
        Ghost(selected_logits_repr),
        Ghost(states),
    );
    proof {
        assert forall|k: int| 0 <= k < plan.scheduled_ids@.len() as int implies
            #[trigger] RT::sample_from_repr(
                RT::select_sample_logits_repr(
                    logits_repr,
                    plan.cu_seqlens_q_repr@,
                    k as nat,
                ),
                states[k],
            ).0 == states[k] by {
            assert(selected_logits_repr[k] == RT::select_sample_logits_repr(
                logits_repr,
                plan.cu_seqlens_q_repr@,
                k as nat,
            ));
            assert(RT::sample_from_repr(
                selected_logits_repr[k], states[k],
            ).0 == states[k]);
        }
        assert forall|k: int| 0 <= k < plan.scheduled_ids@.len() as int implies
            (#[trigger] tokens@[k]) as nat == RT::sample_from_repr(
                RT::select_sample_logits_repr(
                    logits_repr,
                    plan.cu_seqlens_q_repr@,
                    k as nat,
                ),
                states[k],
            ).1 by {
            assert(selected_logits_repr[k] == RT::select_sample_logits_repr(
                logits_repr,
                plan.cu_seqlens_q_repr@,
                k as nat,
            ));
            assert((tokens@[k]) as nat == RT::sample_from_repr(
                selected_logits_repr[k], states[k],
            ).1);
        }
    }
    let mut samples = vstd::hash_map::HashMapWithView::<u64, SampleResult>::new();
    let mut i: usize = 0;
    while i < plan.scheduled_ids.len()
        invariant
            obeys_key_model::<u64>(),
            i <= plan.scheduled_ids@.len(),
            plan.scheduled_ids@.no_duplicates(),
            plan.sample_mask@.len() == plan.scheduled_ids@.len(),
            plan.cu_seqlens_q_repr@.len() == plan.scheduled_ids@.len() + 1,
            plan.cu_seqlens_q_repr@[0] == 0,
            forall|j: int| 0 <= j < plan.scheduled_ids@.len() as int ==>
                plan.cu_seqlens_q_repr@[j] < #[trigger] plan.cu_seqlens_q_repr@[j + 1],
            plan.cu_seqlens_q_repr@[plan.scheduled_ids@.len() as int]
                == plan.input_ids_repr@.len() as int,
            tokens@.len() == plan.scheduled_ids@.len(),
            states.len() == plan.scheduled_ids@.len(),
            forall|k: int| 0 <= k < plan.scheduled_ids@.len() as int
                    && live_requests@.contains_key(#[trigger] plan.scheduled_ids@[k]) ==>
                states[k] == live_requests@[plan.scheduled_ids@[k]].sampler_state,
            forall|k: int| 0 <= k < plan.scheduled_ids@.len() as int ==>
                (#[trigger] tokens@[k]) as nat == RT::sample_from_repr(
                    RT::select_sample_logits_repr(
                        logits_repr,
                        plan.cu_seqlens_q_repr@,
                        k as nat,
                    ),
                    states[k],
                ).1,
            forall|k: int| 0 <= k < plan.scheduled_ids@.len() as int ==>
                #[trigger] RT::sample_from_repr(
                    RT::select_sample_logits_repr(
                        logits_repr,
                        plan.cu_seqlens_q_repr@,
                        k as nat,
                    ),
                    states[k],
                ).0 == states[k],
            forall|k: int| #![trigger plan.scheduled_ids@[k], plan.sample_mask@[k]]
                    0 <= k < i as int
                    && plan.sample_mask@[k]
                    && live_requests@.contains_key(plan.scheduled_ids@[k]) ==>
                samples@.contains_key(plan.scheduled_ids@[k])
                && samples@[plan.scheduled_ids@[k]].sampler_state
                    == RT::sample_from_repr(
                        RT::select_sample_logits_repr(logits_repr,
                            plan.cu_seqlens_q_repr@, k as nat),
                        live_requests@[plan.scheduled_ids@[k]].sampler_state).0
                && samples@[plan.scheduled_ids@[k]].token as nat
                    == RT::sample_from_repr(
                        RT::select_sample_logits_repr(logits_repr,
                            plan.cu_seqlens_q_repr@, k as nat),
                        live_requests@[plan.scheduled_ids@[k]].sampler_state).1,
            forall|rid: RequestId| #[trigger] samples@.contains_key(rid) ==>
                (exists|k: int| 0 <= k < i as int
                    && plan.scheduled_ids@[k] == rid
                    && #[trigger] plan.sample_mask@[k])
                && live_requests@.contains_key(rid),
        decreases plan.scheduled_ids@.len() - i,
    {
        let rid = plan.scheduled_ids[i];
        if plan.sample_mask[i] {
            let state_opt = live_requests.get(&rid);
            match state_opt {
                Some(state) => {
                    let sampler_state = state.sampler_state;
                    let token = tokens[i];
                    proof {
                        assert(states[i as int] == state.sampler_state);
                        assert((tokens@[i as int]) as nat == RT::sample_from_repr(
                            RT::select_sample_logits_repr(
                                logits_repr,
                                plan.cu_seqlens_q_repr@,
                                i as nat,
                            ),
                            state.sampler_state,
                        ).1);
                        assert(RT::sample_from_repr(
                            RT::select_sample_logits_repr(
                                logits_repr,
                                plan.cu_seqlens_q_repr@,
                                i as nat,
                            ),
                            state.sampler_state,
                        ).0 == state.sampler_state);
                    }
                    proof {
                        // `rid` cannot already be in `samples`: its dom is drawn
                        // from scheduled_ids[0..i], all distinct from index i.
                        if samples@.contains_key(rid) {
                            let k = choose|k: int| 0 <= k < i as int
                                && plan.scheduled_ids@[k] == rid;
                            assert(plan.scheduled_ids@[k]
                                == plan.scheduled_ids@[i as int]);
                            assert(false);
                        }
                    }
                    samples.insert(rid, SampleResult { sampler_state, token });
                },
                None => {},
            }
        }
        i = i + 1;
    }
    proof {
        let ghost entry_kv = Seq::new(model_config.num_layers as nat, |j: int|
            (old(kv_perms).k_repr(j), old(kv_perms).v_repr(j)));
        assert(pre_kv =~= entry_kv) by {
            assert forall|j: int| 0 <= j < pre_kv.len() implies
                #[trigger] pre_kv[j] == entry_kv[j] by {
            }
        }
        assert(pre_kv == entry_kv);
        reveal(step_core_samples_match);
        assert(i as int == plan.scheduled_ids@.len());
        assert forall|j: int| 0 <= j < model_config.num_layers as int implies
            #[trigger] kv_perms.k_repr(j) == post_kv[j].0
            && kv_perms.v_repr(j) == post_kv[j].1 by {
            assert(j < post_kv.len());
            // Instantiate the forward contract using its K-side trigger, then
            // split the pair before rebuilding this quantified conjunction.
            assert(kv_perms.k_repr(j) == post_kv[j].0);
            assert(kv_perms.v_repr(j) == post_kv[j].1);
        }
        assert forall|j: int| 0 <= j < model_config.num_layers as int implies
            #[trigger] kv_perms.v_repr(j) == post_kv[j].1 by {
            assert(kv_perms.k_repr(j) == post_kv[j].0);
        }
        assert forall|k: int| #![trigger plan.scheduled_ids@[k], plan.sample_mask@[k]]
                0 <= k < plan.scheduled_ids@.len() as int
                && plan.sample_mask@[k]
                && live_requests@.contains_key(plan.scheduled_ids@[k])
            implies samples@.contains_key(plan.scheduled_ids@[k])
                && samples@[plan.scheduled_ids@[k]].sampler_state
                    == RT::sample_from_repr(
                        RT::select_sample_logits_repr(
                            logits_repr, plan.cu_seqlens_q_repr@, k as nat,
                        ),
                        live_requests@[plan.scheduled_ids@[k]].sampler_state,
                    ).0
                && samples@[plan.scheduled_ids@[k]].token as nat
                    == RT::sample_from_repr(
                        RT::select_sample_logits_repr(
                            logits_repr, plan.cu_seqlens_q_repr@, k as nat,
                        ),
                        live_requests@[plan.scheduled_ids@[k]].sampler_state,
                    ).1
        by {
            assert(plan.scheduled_ids@.len() > 0);
            assert(0 <= 0int < plan.scheduled_ids@.len() as int);
            assert(step_plan_shape_ok(plan));
            assert forall|j: int| 0 <= j < plan.scheduled_ids@.len() as int
                implies plan.cu_seqlens_q_repr@[j]
                    < #[trigger] plan.cu_seqlens_q_repr@[j + 1] by {
            }
            crate::proof::tensor::geometry::lemma_cu_int_bounds(
                plan.cu_seqlens_q_repr@,
                plan.scheduled_ids@.len() as int,
            );
            crate::proof::tensor::geometry::lemma_cu_int_bounds(plan.cu_seqlens_q_repr@, k);
            assert(0 <= plan.cu_seqlens_q_repr@[k]);
            assert(plan.cu_seqlens_q_repr@[k]
                < plan.cu_seqlens_q_repr@[k + 1]);
            assert(plan.cu_seqlens_q_repr@[k + 1] > 0);
            assert(plan.cu_seqlens_q_repr@[k + 1]
                <= plan.cu_seqlens_q_repr@[
                    plan.scheduled_ids@.len() as int
                ]);
            assert(plan.cu_seqlens_q_repr@[k + 1]
                <= logits_repr.len());
        }
        assert forall|rid: RequestId| #[trigger] samples@.contains_key(rid) implies
            live_requests@.contains_key(rid)
                && exists|k: int| 0 <= k < plan.scheduled_ids@.len() as int
                    && plan.scheduled_ids@[k] == rid
                    && #[trigger] plan.sample_mask@[k] by {
            let k = choose|k: int| 0 <= k < plan.scheduled_ids@.len() as int
                && plan.scheduled_ids@[k] == rid
                && plan.sample_mask@[k];
        }
        assert(kv_perms.len() == old(kv_perms).len());
        assert(kv_perms.extracted() == Set::<int>::empty());
        assert(RT::kv_perms_ids_distinct(*kv_perms));
        assert(model_config.num_layers as nat <= kv_caches@.len());
        assert(model_config.num_layers as nat <= old(kv_perms).len());
        assert forall|j: int| 0 <= j < model_config.num_layers as int
            implies #[trigger] kv_caches@[j].0.id() == kv_perms.k_id(j)
                && kv_caches@[j].1.id() == kv_perms.v_id(j) by {
            RT::lemma_kv_cache_tensor_ids_match_at(
                kv_caches@, *old(kv_perms),
                model_config.num_layers as nat, j,
            );
            assert(kv_perms.k_id(j) == old(kv_perms).k_id(j));
            assert(kv_perms.v_id(j) == old(kv_perms).v_id(j));
        }
        RT::lemma_kv_cache_tensor_ids_match_from_pointwise(
            kv_caches@, *kv_perms, model_config.num_layers as nat,
        );
        assert(RT::kv_cache_tensor_ids_match(
            kv_caches@, *kv_perms, model_config.num_layers as nat,
        ));
        assert forall|j: int| 0 <= j < model_config.num_layers as int implies
            #[trigger] kv_perms.k_id(j) == old(kv_perms).k_id(j)
                && kv_perms.v_id(j) == old(kv_perms).v_id(j) by {
            assert(kv_perms.k_id(j) == old(kv_perms).k_id(j));
            assert(kv_perms.v_id(j) == old(kv_perms).v_id(j));
        }
        assert forall|j: int| 0 <= j < model_config.num_layers as int implies
            #[trigger] kv_perms.v_id(j) == old(kv_perms).v_id(j) by {
            assert(kv_perms.k_id(j) == old(kv_perms).k_id(j));
        }
        assert(step_core_samples_match(
            plan, live_requests@, logits_repr, samples@,
        ));
    }
    samples
}

impl Engine {
    // Assemble an engine from already-bound runtime tensors.  In particular,
    // this does not mint replacement permissions for caller-provided handles:
    // the unique bundles returned by the tensor constructors are moved into
    // the engine and retained across every step.
    fn init_with_scheduler(
        cs: CacheScheduler,
        kv_caches: Vec<(RT::Tensor, RT::Tensor)>,
        kv_perms: Tracked<RT::KVCachePerms>,
        weights: RT::ModelWeights,
        runtime: RT::ModelRuntime,
        weights_perms: Tracked<RT::ModelWeightsPerms>,
        model_config: ModelConfig,
    ) -> (out: Engine)
        requires
            cs_valid(&cs),
            free_queue_valid(&cs),
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
            RT::kv_perms_page_shape(kv_perms@, cs.num_blocks as nat),
        ensures
            out.cs == cs,
            cs_valid(&out.cs),
            free_queue_valid(&out.cs),
            out.kv_caches@ == kv_caches@,
            out.weights == weights,
            out.runtime == runtime,
            out.model_config == model_config,
            eng_execution_perms_ok(&out),
            eng_cache_shape_ok(&out),
    {
        let ghost kv_reprs = Seq::new(model_config.num_layers as nat,
            |j: int| (kv_perms@.k_repr(j), kv_perms@.v_repr(j)));
        assert forall|j: int| 0 <= j < model_config.num_layers as int implies
            #[trigger] kv_reprs[j]
                == (kv_perms@.k_repr(j), kv_perms@.v_repr(j)) by {
        }
        assert forall|i: int| 0 <= i < model_config.num_layers as int implies
            #[trigger] kv_caches@[i].0.id() == kv_perms@.k_id(i) by {
            RT::lemma_kv_cache_tensor_ids_match_at(
                kv_caches@, kv_perms@, model_config.num_layers as nat, i,
            );
        }
        assert forall|i: int| 0 <= i < model_config.num_layers as int implies
            #[trigger] kv_caches@[i].1.id() == kv_perms@.v_id(i) by {
            RT::lemma_kv_cache_tensor_ids_match_at(
                kv_caches@, kv_perms@, model_config.num_layers as nat, i,
            );
        }
        let out = Engine {
            cs,
            kv_caches,
            weights,
            runtime,
            model_config,
            kv_caches_repr: Ghost(kv_reprs),
            weights_perms,
            kv_perms,
            last_step_prefix_reuse: Vec::new(),
        };
        assert(eng_execution_perms_ok(&out)) by {
            reveal(eng_execution_perms_ok);
        }
        out
    }

    // Architecture-dispatched request/scheduler initialization. This admits
    // any weight/runtime pair accepted by `model_execution_valid` and
    // establishes the common executable ownership invariant.
    pub fn init(
        config: SchedulerConfig,
        num_blocks: u64,
        kv_caches: Vec<(RT::Tensor, RT::Tensor)>,
        kv_perms: Tracked<RT::KVCachePerms>,
        weights: RT::ModelWeights,
        runtime: RT::ModelRuntime,
        weights_perms: Tracked<RT::ModelWeightsPerms>,
        model_config: ModelConfig,
        requests: Vec<RequestState>,
    ) -> (out: Engine)
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
            forall|a: int, b: int|
                #![trigger requests@[a].request_id, requests@[b].request_id]
                0 <= a < b < requests@.len() ==>
                    requests@[a].request_id != requests@[b].request_id,
            forall|k: int| 0 <= k < requests@.len() ==> {
                let request = #[trigger] requests@[k];
                &&& request.generated_tokens@.len() == 0
                &&& can_step(request)
                &&& request_history_capacity_safe(request)
            },
        ensures
            cs_valid(&out.cs),
            free_queue_valid(&out.cs),
            engine_init_request_relation(&out, config, num_blocks, requests@),
            out.kv_caches@ == kv_caches@,
            out.weights == weights,
            out.runtime == runtime,
            out.model_config == model_config,
            eng_execution_perms_ok(&out),
            residency_running_aligned(&out.cs),
            persistent_provenance_closed(&out.cs),
            live_request_step_ready(&out.cs),
            eng_cache_shape_ok(&out),
    {
        // @kernel-bridge-begin exec::engine::qualified_kernel_plan_gate
        #[cfg(not(verus_only))]
        if !RT::model_runtime_admitted_by_runtime_gate(
            &runtime,
            model_config.num_layers,
        ) {
            panic!("Engine::init rejected an unqualified or changed kernel plan");
        }
        // @kernel-bridge-end exec::engine::qualified_kernel_plan_gate
        #[cfg(not(verus_only))]
        {
            for i in 0..requests.len() {
                assert!(
                    requests[i].generated_tokens.is_empty(),
                    "Engine::init requires fresh requests",
                );
                assert!(
                    !requests[i].prompt_tokens.is_empty(),
                    "Engine::init requires nonempty prompts",
                );
                assert!(
                    requests[i].max_tokens > 0,
                    "Engine::init requires a positive generation budget",
                );
                let history_capacity = requests[i]
                    .prompt_tokens
                    .len()
                    .checked_add(requests[i].max_tokens)
                    .expect("Engine::init request history capacity overflow");
                assert!(
                    u64::try_from(history_capacity).is_ok(),
                    "Engine::init request history exceeds u64 capacity",
                );
                for j in (i + 1)..requests.len() {
                    assert!(
                        requests[i].request_id != requests[j].request_id,
                        "Engine::init requires pairwise-distinct request ids",
                    );
                }
            }
        }
        let ghost request_states = requests@;
        let cs = CacheScheduler::init_with_requests(config, num_blocks, requests);
        let out = Engine::init_with_scheduler(
            cs, kv_caches, kv_perms, weights, runtime, weights_perms, model_config,
        );
        proof {
            reveal(engine_init_request_relation);
            assert(engine_init_request_relation(
                &out, config, num_blocks, request_states,
            ));
        }
        out
    }

    // Admit a fresh request between calls to `step`.  The scheduler does not
    // allocate blocks here: the next plan performs prefix matching/allocation
    // under the ordinary batching policy.
    pub fn add_request(&mut self, request: RequestState)
        requires
            obeys_key_model::<u64>(),
            cs_valid(&old(self).cs),
            free_queue_valid(&old(self).cs),
            eng_execution_perms_ok(old(self)),
            eng_cache_shape_ok(old(self)),
            residency_history_aligned(&old(self).cs),
            slot_mapping_aligned(&old(self).cs),
            tail_write_exclusive(&old(self).cs),
            residency_running_aligned(&old(self).cs),
            persistent_provenance_closed(&old(self).cs),
            live_request_step_ready(&old(self).cs),
            !old(self).cs.accepted_requests@.contains_key(request.request_id),
            request.generated_tokens@.len() == 0,
            can_step(request),
            request_history_capacity_safe(request),
        ensures
            engine_admission_relation(old(self), final(self), request),
            cs_valid(&final(self).cs),
            free_queue_valid(&final(self).cs),
            eng_execution_perms_ok(final(self)),
            eng_cache_shape_ok(final(self)),
            residency_history_aligned(&final(self).cs),
            slot_mapping_aligned(&final(self).cs),
            tail_write_exclusive(&final(self).cs),
            residency_running_aligned(&final(self).cs),
            persistent_provenance_closed(&final(self).cs),
            live_request_step_ready(&final(self).cs),
    {
        self.cs.add_request(request);
        proof {
            assert(engine_admission_relation(old(self), self, request));
            assert(eng_execution_perms_ok(self));
            assert(eng_cache_shape_ok(self));
        }
    }

    // Checked runtime-facing admission.  Invalid arrivals are rejected before
    // any mutation; accepted arrivals are converted to a fresh RequestState and
    // delegated to the verified stable-boundary transition above.
    pub fn try_add_request(&mut self, request: NewRequest) -> (out: AdmissionStatus)
        requires
            obeys_key_model::<u64>(),
            cs_valid(&old(self).cs),
            free_queue_valid(&old(self).cs),
            eng_execution_perms_ok(old(self)),
            eng_cache_shape_ok(old(self)),
            residency_history_aligned(&old(self).cs),
            slot_mapping_aligned(&old(self).cs),
            tail_write_exclusive(&old(self).cs),
            residency_running_aligned(&old(self).cs),
            persistent_provenance_closed(&old(self).cs),
            live_request_step_ready(&old(self).cs),
        ensures
            cs_valid(&final(self).cs),
            free_queue_valid(&final(self).cs),
            eng_execution_perms_ok(final(self)),
            eng_cache_shape_ok(final(self)),
            residency_history_aligned(&final(self).cs),
            slot_mapping_aligned(&final(self).cs),
            tail_write_exclusive(&final(self).cs),
            residency_running_aligned(&final(self).cs),
            persistent_provenance_closed(&final(self).cs),
            live_request_step_ready(&final(self).cs),
            out == AdmissionStatus::Accepted ==> {
                let rid = request.request_id;
                &&& final(self).cs.waiting@ == old(self).cs.waiting@.push(rid)
                &&& final(self).cs.live_requests@.contains_key(rid)
                &&& final(self).cs.live_requests@[rid].request_id == rid
                &&& final(self).cs.live_requests@[rid].prompt_tokens@
                    == request.prompt_tokens@
                &&& final(self).cs.live_requests@[rid].generated_tokens@.len() == 0
                &&& final(self).cs.live_requests@[rid].max_tokens == request.max_tokens
                &&& same_eos_tokens(
                    eos_tokens(final(self).cs.live_requests@[rid]),
                    request.eos_token_ids@.to_set(),
                )
                &&& final(self).cs.live_requests@[rid].ignore_eos == request.ignore_eos
                &&& final(self).cs.accepted_requests@.contains_key(rid)
            },
            out != AdmissionStatus::Accepted ==> {
                &&& final(self).cs.waiting@ == old(self).cs.waiting@
                &&& final(self).cs.live_requests@ == old(self).cs.live_requests@
                &&& final(self).cs.accepted_requests@
                    == old(self).cs.accepted_requests@
            },
    {
        let rid = request.request_id;
        if self.cs.accepted_requests.contains_key(&rid) {
            return AdmissionStatus::DuplicateRequestId;
        }
        if request.prompt_tokens.len() == 0 {
            return AdmissionStatus::EmptyPrompt;
        }
        if request.eos_token_ids.len() == 0 {
            return AdmissionStatus::EmptyEosTokenIds;
        }
        if request.eos_token_ids.len() > MAX_EOS_TOKEN_IDS {
            return AdmissionStatus::TooManyEosTokenIds;
        }
        if request.max_tokens == 0 {
            return AdmissionStatus::ZeroMaxTokens;
        }

        let ghost admitted_eos_tokens = request.eos_token_ids@.to_set();
        let eos_token_set = EosTokenSet::from_nonempty_bounded(
            &request.eos_token_ids,
        );

        let prompt_len = request.prompt_tokens.len();
        if request.max_tokens > usize::MAX - prompt_len {
            return AdmissionStatus::HistoryCapacityOverflow;
        }
        let prompt_len_u128 = prompt_len as u128;
        if prompt_len_u128 > u64::MAX as u128 {
            return AdmissionStatus::HistoryCapacityOverflow;
        }
        let u64_room = (u64::MAX as u128) - prompt_len_u128;
        if request.max_tokens as u128 > u64_room {
            return AdmissionStatus::HistoryCapacityOverflow;
        }

        let state = RequestState::from_parts(
            rid,
            request.prompt_tokens,
            Vec::<TokenId>::new(),
            SamplerState::empty(),
            request.max_tokens,
            eos_token_set,
            request.ignore_eos,
        );
        proof {
            lemma_same_eos_tokens_transitive(
                eos_tokens(state),
                eos_token_set_view(eos_token_set),
                admitted_eos_tokens,
            );
            lemma_same_eos_tokens_view_eq(
                eos_tokens(state),
                admitted_eos_tokens,
            );
            assert(state.prompt_tokens@.len() > 0);
            assert(state.generated_tokens@.len() == 0);
            assert(state.max_tokens > 0);
            assert(valid_request_state(state));
            assert(!is_finished(state));
            assert(can_step(state));
            assert(state.prompt_tokens@.len() + state.max_tokens as int
                <= usize::MAX as int);
            assert(state.prompt_tokens@.len() + state.max_tokens as int
                <= u64::MAX as int);
            assert(request_history_capacity_safe(state));
        }
        self.add_request(state);
        AdmissionStatus::Accepted
    }

    // Architecture-dispatched top-level: plan → ModelForward → SampleAll
    // → Commit.
    // Returns the per-request emitted-token map for this step.
    //
    // The relation follows from the verified callees' postconditions: `plan`
    // (shape + commit-readiness + live preservation), `step_core` (sample
    // map bound to the verified pipeline; post-store cache reprs), and
    // `commit` (full step characterization).  The empty-schedule return is
    // covered by the architecture-dispatched empty-forward theorem.  The
    // mode-specific plan-readiness side conditions remain explicit on this
    // low-level API.  The semantic runtime wrapper derives them from its
    // inductive all-live-request capacity invariant.
    #[verifier::spinoff_prover]
    #[verifier::rlimit(300)]
    pub fn step(
        &mut self,
        graph_overlay: Option<&RT::CudaGraphOverlay>,
    ) -> (out: (
        EmittedTokens,
        Ghost<Map<RequestId, (crate::exec::request_state::SamplerState, TokenId)>>,
        Ghost<StepReprs>,
    ))
        requires
            graph_overlay.is_some() ==> RT::cuda_graph_replay_fidelity(),
            RT::paged_attention_numeric_domain(),
            graph_overlay.is_some() ==> old(self).model_config.num_layers > 0,
            graph_overlay.is_some() ==>
                MODEL_FAMILIES::cuda_graph_overlay_supported(
                    &old(self).weights_perms@,
                ),
            obeys_key_model::<u64>(),
            cs_valid(&old(self).cs),
            free_queue_valid(&old(self).cs),
            eng_execution_perms_ok(old(self)),
            eng_cache_shape_ok(old(self)),
            old(self).cs.num_blocks <= u64::MAX / crate::types::BLOCK_SIZE,
            prefill_plan_ready(&old(self).cs),
            old(self).cs.running@.len() > 0 ==> decode_plan_ready(&old(self).cs),
            residency_history_aligned(&old(self).cs),
            slot_mapping_aligned(&old(self).cs),
            tail_write_exclusive(&old(self).cs),
            residency_running_aligned(&old(self).cs),
            persistent_provenance_closed(&old(self).cs),
            forall|r: RequestId|
                #[trigger] old(self).cs.live_requests@.contains_key(r)
                ==> valid_request_state(old(self).cs.live_requests@[r]),
        ensures
            cs_valid(&final(self).cs),
            free_queue_valid(&final(self).cs),
            residency_history_aligned(&final(self).cs),
            slot_mapping_aligned(&final(self).cs),
            tail_write_exclusive(&final(self).cs),
            residency_running_aligned(&final(self).cs),
            persistent_provenance_closed(&final(self).cs),
            final(self).cs.accepted_requests@ == old(self).cs.accepted_requests@,
            out.0@.dom().subset_of(old(self).cs.live_requests@.dom()),
            // The full structural step effect, consumed by `refinement_step`'s
            // forward-simulation proof (which cannot call this exec `step`
            // directly).  The sample map is returned as ghost data (the view
            // of what the verified `step_core` computed).
            architecture_engine_step_relation(
                *old(self), *final(self), out.0@, out.1@, out.2@,
            ),
            engine_step_semantic_identity(*old(self), *final(self)),
            // Permission ownership is maintained: same runtime tensors and
            // weights, and the post-step ghost cache is the owned perms'
            // post-forward reprs (`model_forward`'s ensures).
            eng_execution_perms_ok(final(self)),
            eng_cache_shape_ok(final(self)),
            final(self).kv_caches@ == old(self).kv_caches@,
            final(self).weights == old(self).weights,
            final(self).runtime == old(self).runtime,
            final(self).model_config == old(self).model_config,
            final(self).weights_perms@ == old(self).weights_perms@,
    {
        // Queue topology is proved inside scheduler transitions. Engine only
        // carries the certificate; unfolding its linked-list quantifiers here
        // needlessly couples them to forward-row separation proofs.
        hide(free_queue_valid);
        hide(Seq::no_duplicates);
        let device_anchor = RT::model_step_plan_device_anchor(&self.weights);
        let (plan, plan_perms) = self.cs.plan(device_anchor);
        crate::exec::step_observation::record_planned_prefix_reuse(
            &self.cs, &plan.scheduled_ids, &mut self.last_step_prefix_reuse,
        );
        let ghost reprs = StepReprs {
            wr: RT::model_weights_repr_of(&self.weights_perms@),
            input_ids: plan.input_ids_repr@,
            positions: plan.positions_repr@,
            slots: plan.slot_mapping_repr@,
            cu_q: plan.cu_seqlens_q_repr@,
            cu_k: plan.cu_seqlens_k_repr@,
            bt: plan.block_table_repr@,
            max_q: plan.max_seqlen_q as nat,
            max_k: plan.max_seqlen_k as nat,
            scheduled: plan.scheduled_ids@,
            sample_mask: plan.sample_mask@,
        };
        proof {
            RT::lemma_model_weights_architecture_repr_valid(
                &self.weights, &self.runtime,
                &self.weights_perms@,
            );
            RT::lemma_model_execution_valid_num_layers(
                &self.weights, &self.runtime, &self.weights_perms@,
            );
            reveal(plan_residency_extents);
            assert(step_reprs_wf(*old(self), reprs)) by {
                assert(plan.scheduled_ids@.no_duplicates());
                assert(reprs.wr.layers.len()
                    == RT::model_weights_num_layers(&old(self).weights));
            }
            assert forall|k: int| 0 <= k < reprs.scheduled.len() implies
                #[trigger] reprs_forward_layout_at(*old(self), reprs, k)
            by {
                assert(plan_forward_layout_at(&old(self).cs, &plan, k));
                assert(plan_residency_extent_at(
                    &old(self).cs, &self.cs, &plan, k,
                ));
                assert(plan_slot_segments_at(
                    &old(self).cs, &self.cs, &plan, k));
                let rid = plan.scheduled_ids@[k];
                assert(self.cs.running@.contains(rid)) by {
                    assert(reprs.scheduled.contains(rid));
                }
                lemma_plan_row_to_reprs_forward_layout(
                    *old(self), &self.cs, &plan, reprs, k);
            }
            assert(reprs_forward_layout_ok(*old(self), reprs));
            assert(reprs.sample_mask == plan.sample_mask@);
            assert forall|rid: RequestId| #[trigger] reprs.scheduled.contains(rid)
                implies self.cs.running@.contains(rid)
            by {
                assert(plan.scheduled_ids@.contains(rid));
            }
            lemma_plan_sample_policy_to_reprs(*old(self), &plan, reprs);
            lemma_plan_cached_prefix_origins_to_reprs(
                *old(self), &plan, reprs,
            );
            // Stable read/write isolation is a scheduler-geometry fact, so
            // establish it before the forward.  CUDA covering consumes this
            // same fact; it does not depend on graph capacity or policy.
            lemma_plan_other_writes_miss_plan_rows(
                &old(self).cs, &self.cs, &plan, reprs,
            );
        }
        if plan.scheduled_ids.len() == 0 {
            proof {
                lemma_admission_origins_are_executed_rows(
                    &old(self).cs, &self.cs, reprs.scheduled, reprs.bt, reprs.cu_k,
                );
                // Empty schedule: shape facts force an empty plan, so the
                // forward is the identity on the ghost cache.
                assert(plan.cu_seqlens_q_repr@[0] == 0);
                assert(plan.input_ids_repr@.len() == 0);
                assert(reprs.wr.layers.len()
                    == RT::model_weights_num_layers(&old(self).weights));
                RT::lemma_model_weights_architecture_repr_valid(
                    &old(self).weights, &old(self).runtime,
                    &old(self).weights_perms@,
                );
                MA::lemma_model_forward_kv_reprs_empty(
                    reprs.wr,
                    RT::model_weights_architecture_repr_of(
                        &old(self).weights_perms@,
                    ),
                    reprs.input_ids, reprs.positions,
                    old(self).kv_caches_repr@, reprs.slots,
                    reprs.cu_q, reprs.cu_k, reprs.max_q, reprs.max_k, reprs.bt);
                assert(self.kv_caches_repr@
                    == architecture_engine_post_kv_of(
                        *old(self), reprs, old(self).kv_caches_repr@,
                    ));
                let e: Map<RequestId, TokenId> = Map::empty();
                let sm: Map<RequestId, (crate::exec::request_state::SamplerState, TokenId)>
                    = Map::empty();
                assert(e.dom().subset_of(old(self).cs.live_requests@.dom()));
                assert(step_reprs_wf(*old(self), reprs)) by {
                    assert(plan.scheduled_ids@.no_duplicates());
                    assert(reprs.wr.layers.len()
                        == RT::model_weights_num_layers(&old(self).weights));
                }
                assert(forall|r: RequestId| !reprs.scheduled.contains(r));
                assert(reprs_other_writes_miss_plan_rows(reprs));
                assert(step_reprs_block_tables_bounded(
                    *old(self), reprs,
                ));
                assert(tail_write_exclusive(&self.cs)) by {
                    assert(tail_write_exclusive(&old(self).cs));
                    reveal(tail_write_exclusive);
                    assert forall|rid: RequestId|
                        #[trigger] self.cs.running@.contains(rid)
                        && self.cs.live_requests@.contains_key(rid)
                        implies {
                            let ids = self.cs.request_residency@[rid].block_ids@;
                            &&& self.cs.request_residency@.contains_key(rid)
                            &&& ids.len() >= 1
                            &&& self.cs.blocks@.contains_key(ids[ids.len() - 1])
                            &&& self.cs.blocks@[ids[ids.len() - 1]].refcount == 1
                            &&& self.cs.blocks@[ids[ids.len() - 1]].prefix_depth == 0
                            &&& self.cs.blocks@[ids[ids.len() - 1]].hash_value == 0
                        }
                    by {
                        assert(old(self).cs.running@.contains(rid));
                        assert(old(self).cs.live_requests@.contains_key(rid));
                        assert(self.cs.request_residency@[rid]
                            == old(self).cs.request_residency@[rid]);
                        let ids = self.cs.request_residency@[rid].block_ids@;
                        let t = ids[ids.len() - 1];
                        assert(old(self).cs.blocks@[t].prefix_depth == 0);
                        assert(self.cs.blocks@[t].prefix_depth
                            == old(self).cs.blocks@[t].prefix_depth);
                    }
                }
                reveal(reprs_prefill_residencies);
                assert forall|k: int| 0 <= k < reprs.scheduled.len()
                    implies #[trigger] reprs_prefill_residency_at(
                        *old(self), *self, reprs, k,
                    ) by {}
                assert(reprs_prefill_residencies(*old(self), *self, reprs));
                assert(architecture_engine_step_relation(
                    *old(self), *self, e, sm, reprs,
                ));
                assert(old(self).kv_caches_repr@.len()
                    == reprs.wr.layers.len());
                EA::lemma_architecture_eng_cache_shape_preserved(
                    *old(self), *self, e, sm, reprs,
                );
                assert(self.weights_perms@ == old(self).weights_perms@);
                assert(self.weights == old(self).weights);
                assert(self.runtime == old(self).runtime);
                assert(self.model_config == old(self).model_config);
                assert(self.kv_caches@ == old(self).kv_caches@);
                assert(self.kv_perms@ == old(self).kv_perms@);
                assert(self.kv_caches_repr@ == old(self).kv_caches_repr@);
                assert(engine_step_semantic_identity(
                    *old(self), *self,
                )) by {
                    reveal(engine_step_semantic_identity);
                }
                assert(eng_execution_perms_ok(self)) by {
                    reveal(eng_execution_perms_ok);
                }
                assert(forall|r: RequestId| #[trigger] reprs.scheduled.contains(r) ==>
                    old(self).cs.running@.contains(r) || old(self).cs.waiting@.contains(r));
                assert forall|r: RequestId|
                    #![trigger self.cs.running@.contains(r)]
                    self.cs.running@.contains(r)
                    implies (old(self).cs.running@.contains(r)
                        || reprs.scheduled.contains(r)) by {
                    assert(old(self).cs.running@.contains(r));
                }
                assert forall|r: RequestId|
                    #![trigger self.cs.running@.contains(r)]
                    old(self).cs.running@.contains(r) || reprs.scheduled.contains(r)
                    implies self.cs.running@.contains(r) by {
                    if reprs.scheduled.contains(r) {
                        assert(false);
                    }
                }
            }
            return (
                HashMapWithView::<u64, TokenId>::new(),
                Ghost(Map::empty()),
                Ghost(reprs),
            );
        }
        let tracked plan_perms_v = plan_perms.get();
        proof {
            // Non-cs engine fields are untouched by `plan` (field framing);
            // instantiate `eng_execution_perms_ok(old(self))` for
            // `step_core`'s requires.
            assert(self.kv_caches@ == old(self).kv_caches@);
            assert(self.kv_perms@ == old(self).kv_perms@);
            assert forall|i: int| 0 <= i < self.model_config.num_layers as int
                implies #[trigger] self.kv_caches@[i].0.id() == self.kv_perms@.k_id(i)
                    && self.kv_caches@[i].1.id() == self.kv_perms@.v_id(i) by {
                assert(old(self).kv_caches@[i].0.id() == old(self).kv_perms@.k_id(i));
            }
        }
        proof {
            assert(RT::model_execution_valid(
                &self.weights, &self.runtime, &self.weights_perms@,
            ));
            assert(RT::model_weights_num_layers(&self.weights)
                == self.model_config.num_layers as nat);
            assert(self.kv_caches.len() == self.model_config.num_layers as nat);
            assert(self.kv_perms@.len() == self.model_config.num_layers as nat);
            assert(RT::kv_perms_ids_distinct(self.kv_perms@));
            assert(self.kv_perms@.extracted() == Set::<int>::empty());
            assert forall|i: int| 0 <= i < self.model_config.num_layers as int
                implies #[trigger] self.kv_caches@[i].0.id() == self.kv_perms@.k_id(i)
                    && self.kv_caches@[i].1.id() == self.kv_perms@.v_id(i) by {
                assert(old(self).kv_caches@[i].0.id() == old(self).kv_perms@.k_id(i));
                assert(old(self).kv_caches@[i].1.id() == old(self).kv_perms@.v_id(i));
            }
            assert(self.cs.num_blocks == old(self).cs.num_blocks);
            lemma_plan_paged_attention_metadata_ready(
                &old(self).cs, &self.cs, &plan,
            );
            crate::proof::cache::plan_layout::lemma_reprs_store_kv_cache_metadata_ready(
                *old(self), reprs,
            );
            reveal(eng_cache_shape_ok);
            assert forall|i: int| 0 <= i < self.model_config.num_layers as int
                implies #[trigger] RT::paged_attention_launch_ready(
                    plan.input_ids_repr@.len(), self.kv_perms@.k_repr(i),
                    self.kv_perms@.v_repr(i),
                    plan.cu_seqlens_q_repr@, plan.cu_seqlens_k_repr@,
                    plan.max_seqlen_q as nat, plan.max_seqlen_k as nat,
                    plan.block_table_repr@,
                )
            by {
                assert(self.kv_perms@.k_repr(i)
                    == old(self).kv_caches_repr@[i].0);
                assert(self.kv_perms@.v_repr(i)
                    == old(self).kv_caches_repr@[i].1);
                assert(old(self).kv_caches_repr@[i].0.len()
                    == old(self).cs.num_blocks as nat);
                assert(old(self).kv_caches_repr@[i].1.len()
                    == old(self).cs.num_blocks as nat);
                reveal(RT::paged_cache_geometry);
                assert(self.kv_perms@.k_repr(i).len() > 0);
                assert(self.kv_perms@.v_repr(i).len()
                    == self.kv_perms@.k_repr(i).len());
                assert forall|p: int|
                    0 <= p < self.kv_perms@.k_repr(i).len()
                    implies (#[trigger] self.kv_perms@.k_repr(i)[p]).len()
                        == crate::types::BLOCK_SIZE_SPEC as int
                by {
                    assert(self.kv_perms@.k_repr(i)[p]
                        == old(self).kv_caches_repr@[i].0[p]);
                }
                assert forall|p: int|
                    0 <= p < self.kv_perms@.v_repr(i).len()
                    implies (#[trigger] self.kv_perms@.v_repr(i)[p]).len()
                        == crate::types::BLOCK_SIZE_SPEC as int
                by {
                    assert(self.kv_perms@.v_repr(i)[p]
                        == old(self).kv_caches_repr@[i].1[p]);
                }
                assert(RT::paged_cache_geometry(
                    self.kv_perms@.k_repr(i), self.kv_perms@.v_repr(i),
                ));
                assert(RT::paged_attention_metadata_ready(
                    plan.input_ids_repr@.len(),
                    self.kv_perms@.k_repr(i).len(),
                    plan.cu_seqlens_q_repr@, plan.cu_seqlens_k_repr@,
                    plan.max_seqlen_q as nat, plan.max_seqlen_k as nat,
                    plan.block_table_repr@,
                ));
                RT::lemma_paged_attention_launch_ready_from_parts(
                    plan.input_ids_repr@.len(),
                    self.kv_perms@.k_repr(i), self.kv_perms@.v_repr(i),
                    plan.cu_seqlens_q_repr@, plan.cu_seqlens_k_repr@,
                    plan.max_seqlen_q as nat, plan.max_seqlen_k as nat,
                    plan.block_table_repr@,
                );
            }
            assert forall|i: int| 0 <= i < self.model_config.num_layers as int
                implies #[trigger] RT::store_kv_cache_launch_ready(
                    plan.input_ids_repr@.len(), self.kv_perms@.k_repr(i),
                    self.kv_perms@.v_repr(i), plan.slot_mapping_repr@,
                )
            by {
                assert(self.kv_perms@.k_repr(i)
                    == old(self).kv_caches_repr@[i].0);
                assert(self.kv_perms@.v_repr(i)
                    == old(self).kv_caches_repr@[i].1);
                assert(old(self).kv_caches_repr@[i].0.len()
                    == old(self).cs.num_blocks as nat);
                RT::lemma_paged_attention_launch_ready_parts(
                    plan.input_ids_repr@.len(),
                    self.kv_perms@.k_repr(i), self.kv_perms@.v_repr(i),
                    plan.cu_seqlens_q_repr@, plan.cu_seqlens_k_repr@,
                    plan.max_seqlen_q as nat, plan.max_seqlen_k as nat,
                    plan.block_table_repr@,
                );
                assert(RT::store_kv_cache_metadata_ready(
                    plan.input_ids_repr@.len(),
                    self.kv_perms@.k_repr(i).len(),
                    plan.slot_mapping_repr@,
                ));
                RT::lemma_store_kv_cache_launch_ready_from_parts(
                    plan.input_ids_repr@.len(),
                    self.kv_perms@.k_repr(i), self.kv_perms@.v_repr(i),
                    plan.slot_mapping_repr@,
                );
            }
            if graph_overlay.is_some()
                && reprs.input_ids.len() == reprs.scheduled.len()
                && reprs.max_q == 1
            {
                assert(reprs.wr.layers.len() > 0);
                assert(old(self).kv_caches_repr@.len()
                    == reprs.wr.layers.len());
                assert(reprs.input_ids.len() == reprs.bt.len());
                EA::lemma_engine_cuda_graph_decode_cover_ready(
                    *old(self), reprs,
                );
                let pre = Seq::new(self.model_config.num_layers as nat,
                    |i: int| (self.kv_perms@.k_repr(i),
                        self.kv_perms@.v_repr(i)));
                assert(pre =~= old(self).kv_caches_repr@);
                assert(MODEL_FAMILIES::cuda_graph_decode_cover_ready(
                    &self.weights_perms@,
                    reprs.wr, reprs.input_ids, reprs.positions,
                    pre, reprs.slots, reprs.cu_q, reprs.cu_k,
                    reprs.max_q, reprs.max_k, reprs.bt,
                ));
            }
        }
        // The verified core: forward + sampling, with the sample map bound to
        // the verified pipeline (see `step_core`'s ensures).
        proof {
            let pre_kv = Seq::new(self.model_config.num_layers as nat, |i: int| (
                self.kv_perms@.k_repr(i), self.kv_perms@.v_repr(i),
            ));
            assert(reprs.input_ids.len() > 0) by {
                assert(reprs.scheduled.len() > 0);
                lemma_nonempty_step_has_query_tokens(*old(self), reprs);
            }
            EA::lemma_engine_architecture_model_forward_ready(
                *old(self), reprs, &self.weights_perms@, pre_kv,
            );
            assert(crate::exec::model::architecture_model_forward_ready(
                &self.weights_perms@,
                plan.input_ids_repr@,
                plan.positions_repr@,
                Seq::new(self.model_config.num_layers as nat, |i: int| (
                    self.kv_perms@.k_repr(i), self.kv_perms@.v_repr(i),
                )),
                plan.slot_mapping_repr@,
                plan.cu_seqlens_q_repr@,
                plan.cu_seqlens_k_repr@,
                plan.max_seqlen_q as nat,
                plan.max_seqlen_k as nat,
                plan.block_table_repr@,
                plan.scheduled_ids@.len(),
            ));
        }
        let ghost core_logits = MA::model_forward_logits_repr(
            RT::model_weights_repr_of(&self.weights_perms@),
            RT::model_weights_architecture_repr_of(&self.weights_perms@),
            plan.input_ids_repr@,
            plan.positions_repr@,
            Seq::new(self.model_config.num_layers as nat, |i: int| (
                self.kv_perms@.k_repr(i), self.kv_perms@.v_repr(i),
            )),
            plan.slot_mapping_repr@,
            plan.cu_seqlens_q_repr@,
            plan.cu_seqlens_k_repr@,
            plan.max_seqlen_q as nat,
            plan.max_seqlen_k as nat,
            plan.block_table_repr@,
        );
        proof {
            assert(RT::kv_cache_tensor_ids_match(
                self.kv_caches@,
                self.kv_perms@,
                self.model_config.num_layers as nat,
            )) by {
                reveal(RT::kv_cache_tensor_ids_match);
                assert forall|i: int|
                    0 <= i < self.model_config.num_layers as int
                    implies #[trigger] self.kv_caches[i].0.id()
                            == self.kv_perms@.k_id(i)
                        && self.kv_caches[i].1.id()
                            == self.kv_perms@.v_id(i)
                by {
                    RT::lemma_kv_cache_tensor_ids_match_at(
                        self.kv_caches@, self.kv_perms@,
                        self.model_config.num_layers as nat, i,
                    );
                }
            }
        }
        let samples = step_core(
            &self.model_config,
            &self.weights,
            &self.runtime,
            Tracked(self.weights_perms.borrow()),
            &self.kv_caches,
            Tracked(self.kv_perms.borrow_mut()),
            &plan,
            Tracked(&plan_perms_v),
            &self.cs.live_requests,
            graph_overlay,
        );
        // Maintain the persistent ghost cache: after the forward, the owned
        // kv_perms hold the post-store per-layer reprs (`step_core`'s
        // ensures); mirror them into `kv_caches_repr` so the common execution
        // invariant keeps talking about current contents.
        self.kv_caches_repr = Ghost(Seq::new(self.model_config.num_layers as nat,
            |j: int| (self.kv_perms@.k_repr(j), self.kv_perms@.v_repr(j))));
        let ghost samples_g = samples_view(samples@);
        proof {
            // Pre-commit facts: the ghost cache update is exactly the
            // verified forward's post-store cache over the plan reprs.
            let ghost n = self.model_config.num_layers as int;
            let ghost pre_kv = Seq::new(n as nat,
                |j: int| (old(self).kv_perms@.k_repr(j), old(self).kv_perms@.v_repr(j)));
            assert(old(self).kv_caches_repr@ =~= pre_kv);
            assert(core_logits == architecture_step_logits_repr(
                *old(self), reprs,
            ));
            let ghost post_kv = architecture_engine_post_kv_of(
                *old(self), reprs, old(self).kv_caches_repr@,
            );
            RT::lemma_model_weights_architecture_repr_valid(
                &old(self).weights, &old(self).runtime,
                &old(self).weights_perms@,
            );
            MA::lemma_model_forward_kv_reprs_len(
                reprs.wr,
                RT::model_weights_architecture_repr_of(
                    &old(self).weights_perms@,
                ),
                reprs.input_ids, reprs.positions,
                old(self).kv_caches_repr@, reprs.slots,
                reprs.cu_q, reprs.cu_k, reprs.max_q, reprs.max_k, reprs.bt);
            assert(post_kv.len() == n);
            assert forall|j: int| 0 <= j < n
                implies #[trigger] self.kv_caches_repr@[j] == post_kv[j] by {
                assert(self.kv_perms@.k_repr(j) == post_kv[j].0);
                assert(self.kv_perms@.v_repr(j) == post_kv[j].1);
            }
            assert(self.kv_caches_repr@ =~= post_kv);
            // samples_view lookups.
            assert forall|r: RequestId| #[trigger] samples@.contains_key(r)
                implies samples_g.contains_key(r)
                    && samples_g[r].0 == samples@[r].sampler_state
                    && samples_g[r].1 == samples@[r].token by {}
        }
        // Forward has now materialized the scheduled histories. Publish only
        // completed decode pages, before commit appends the next sampled token
        // (which does not yet have KV) or releases a finished request.
        let ghost pre_publication = *self;
        proof {
            assert forall|rid: RequestId| #[trigger] self.cs.running@.contains(rid)
                implies old(self).cs.running@.contains(rid) || old(self).cs.waiting@.contains(rid)
            by {
                if !old(self).cs.running@.contains(rid) {
                    let k = plan.scheduled_ids@.index_of(rid);
                    assert(plan.scheduled_ids@[k] == rid);
                }
            }
            lemma_plan_decode_publication_ready(&old(self).cs, &self.cs);
            assert forall|k: int| 0 <= k < plan.scheduled_ids@.len()
                implies self.cs.running@.contains(#[trigger] plan.scheduled_ids@[k])
            by { assert(plan.scheduled_ids@.contains(plan.scheduled_ids@[k])); }
        }
        let published_decode = self.cs.publish_decode_prefixes(&plan.scheduled_ids);
        proof {
            lemma_decode_publication_plan_frame(
                &old(self).cs, &pre_publication.cs, &self.cs, published_decode@, &plan,
            );
            lemma_decode_publication_plan_origins(
                &old(self).cs, &pre_publication.cs, &self.cs, published_decode@, &plan,
            );
        }
        let ghost pre_commit = *self;
        proof {
            // Commit preconditions: scheduled ⊆ running (plan's evolution
            // clause); commit headroom is `plan`'s by-construction
            // capacity ensures.
            assert forall|k: int|
                #![trigger plan.scheduled_ids@[k]]
                0 <= k < plan.scheduled_ids@.len()
                implies self.cs.running@.contains(plan.scheduled_ids@[k])
            by {
                assert(plan.scheduled_ids@.contains(plan.scheduled_ids@[k]));
            }
            // Unscheduled running requests keep their boundary
            // hash-0 tail (plan preserved the entry fields).
            assert forall|r: RequestId| #[trigger] self.cs.running@.contains(r)
                && self.cs.live_requests@.contains_key(r)
                && !plan.scheduled_ids@.contains(r)
                implies self.cs.blocks@[self.cs.request_residency@[r].block_ids@[
                        self.cs.request_residency@[r].block_ids@.len() - 1]]
                    .hash_value == 0
                    && self.cs.blocks@[self.cs.request_residency@[r].block_ids@[
                        self.cs.request_residency@[r].block_ids@.len() - 1]].prefix_depth == 0
            by {
                assert(old(self).cs.running@.contains(r));
                assert(old(self).cs.live_requests@.contains_key(r));
                assert(self.cs.request_residency@[r]
                    == old(self).cs.request_residency@[r]);
                let t = old(self).cs.request_residency@[r].block_ids@[
                    old(self).cs.request_residency@[r].block_ids@.len() - 1];
                assert(!published_decode@.contains(r));
                lemma_decode_publication_bystander_frame(
                    &pre_publication.cs, &self.cs, published_decode@, r,
                );
                assert(old(self).cs.blocks@[t].hash_value == 0);
                assert(self.cs.blocks@[t].hash_value
                    == old(self).cs.blocks@[t].hash_value);
            }
            // The executable sampler and planner agree exactly on observable
            // rows; KV-only rows are still unstarted when commit parks them.
            RT::lemma_model_weights_architecture_repr_valid(
                &old(self).weights, &old(self).runtime,
                &old(self).weights_perms@,
            );
            MA::lemma_model_forward_logits_repr_shape(
                reprs.wr,
                RT::model_weights_architecture_repr_of(
                    &old(self).weights_perms@,
                ),
                reprs.input_ids, reprs.positions,
                old(self).kv_caches_repr@, reprs.slots,
                reprs.cu_q, reprs.cu_k, reprs.max_q, reprs.max_k, reprs.bt,
            );
            lemma_step_core_samples_to_engine_payload_facts(
                *old(self), &self.cs, &plan, reprs, samples@, samples_g,
                core_logits, architecture_step_logits_repr(*old(self), reprs),
            );
            assert(reprs_other_writes_miss_plan_rows(reprs));
            // Establish this before `commit`: its new persistent-registry
            // frame is quantified over hashes and intentionally unrelated to
            // the row-layout facts below.  Carrying the closed fact across the
            // call keeps those proof domains isolated.
            assert(step_reprs_wf(*old(self), reprs)) by {
                assert(plan.scheduled_ids@.no_duplicates());
                assert(reprs.wr.layers.len()
                    == RT::model_weights_num_layers(&old(self).weights));
            }
            assert(step_reprs_block_tables_bounded(
                *old(self), reprs,
            )) by {
                assert forall|k: int| #![trigger reprs.bt[k]]
                    0 <= k < reprs.scheduled.len() implies
                        reprs.bt[k].len() <= old(self).cs.num_blocks
                by {
                    let rid = reprs.scheduled[k];
                    assert(plan_slot_segments_at(
                        &old(self).cs, &pre_commit.cs, &plan, k,
                    ));
                    reveal(plan_slot_segments_at);
                    assert(reprs.bt[k] == pre_commit.cs
                        .request_residency@[rid].block_ids@);
                    lemma_residency_len_bounded(&pre_commit.cs, rid);
                    assert(pre_commit.cs.num_blocks == old(self).cs.num_blocks);
                }
            }
        }
        let emitted = self.cs.commit(&plan, &samples);
        proof {
            // Assemble the architecture-dispatched relation from
            // plan/step_core/commit.
            lemma_positive_chains_executed_rows_frame(
                &old(self).cs, &pre_commit.cs, &self.cs,
                reprs.scheduled, reprs.bt, reprs.cu_k,
            );
            lemma_published_admission_row_prefixes_frame(
                &old(self).cs, &pre_commit.cs, &self.cs,
                reprs.scheduled, reprs.bt, reprs.cu_k,
            );
            assert(pre_commit.cs.live_requests@ == old(self).cs.live_requests@);
            assert forall|r: RequestId| #[trigger] emitted@.contains_key(r)
                implies old(self).cs.live_requests@.contains_key(r)
                    && samples_g.contains_key(r)
                    && emitted@[r] == samples_g[r].1 by {
                assert(samples@.contains_key(r));
            }
            lemma_commit_survivors_valid(
                &old(self).cs, &self.cs, emitted@, samples_g,
            );
            // Reprs well-formedness + the scheduled/emitted iff.
            assert(step_reprs_wf(*old(self), reprs));
            lemma_commit_emits_iff_reprs(
                *old(self), &pre_commit.cs, &plan, samples@, emitted@, reprs,
            );
            assert forall|r: RequestId| #[trigger] reprs.scheduled.contains(r)
                implies old(self).cs.running@.contains(r)
                    || old(self).cs.waiting@.contains(r) by {
                let k = choose|k: int| 0 <= k < plan.scheduled_ids@.len()
                    && plan.scheduled_ids@[k] == r;
                assert(old(self).cs.running@.contains(plan.scheduled_ids@[k])
                    || old(self).cs.waiting@.contains(plan.scheduled_ids@[k]));
            }
            // A surviving admission retains the exact plan block table as a
            // prefix of its post-commit residency.
            lemma_commit_preserves_prefill_residencies(
                *old(self), pre_commit, *self, &plan, emitted@, reprs,
            );
            // Running evolution: compose the plan's queue growth with the
            // commit's finish-removals (pre-commit live == entry live).
            assert forall|r: RequestId|
                #![trigger self.cs.running@.contains(r)]
                self.cs.running@.contains(r)
                implies ((old(self).cs.running@.contains(r)
                        || reprs.scheduled.contains(r))
                    && !reprs_parks(reprs, r)
                    && !(emitted@.contains_key(r)
                        && crate::exec::request_state::should_finish_after_append(
                            old(self).cs.live_requests@[r], emitted@[r]))) by {
                assert(pre_commit.cs.running@.contains(r));
                assert(pre_commit.cs.live_requests@ == old(self).cs.live_requests@);
            }
            assert forall|r: RequestId|
                #![trigger self.cs.running@.contains(r)]
                (old(self).cs.running@.contains(r) || reprs.scheduled.contains(r))
                    && !reprs_parks(reprs, r)
                    && !(emitted@.contains_key(r)
                        && crate::exec::request_state::should_finish_after_append(
                            old(self).cs.live_requests@[r], emitted@[r]))
                implies self.cs.running@.contains(r) by {
                assert(pre_commit.cs.running@.contains(r));
                assert(pre_commit.cs.live_requests@ == old(self).cs.live_requests@);
            }
            assert(self.cs.num_blocks == pre_commit.cs.num_blocks);
            assert(self.cs.num_blocks == old(self).cs.num_blocks);
            // Residency evolution: plan preserves running residencies, then
            // commit frames the uncommitted and extends the committed.
            assert forall|r: RequestId| #[trigger] old(self).cs.running@.contains(r)
                && !reprs.scheduled.contains(r)
                implies old(self).cs.request_residency@.contains_key(r)
                    && self.cs.request_residency@.contains_key(r)
                    && self.cs.request_residency@[r]
                        == old(self).cs.request_residency@[r] by {
                assert(pre_commit.cs.request_residency@.contains_key(r));
                assert(pre_commit.cs.request_residency@[r]
                    == old(self).cs.request_residency@[r]);
                assert(!emitted@.contains_key(r)) by {
                    if emitted@.contains_key(r) {
                        assert(plan.scheduled_ids@.contains(r));
                    }
                }
            }
            assert forall|r: RequestId| #[trigger] reprs.scheduled.contains(r)
                && old(self).cs.running@.contains(r)
                && reprs_emits(reprs, r)
                && self.cs.request_residency@.contains_key(r)
                implies old(self).cs.request_residency@.contains_key(r)
                    && self.cs.request_residency@[r].block_ids@.len()
                        >= old(self).cs.request_residency@[r].block_ids@.len()
                    && self.cs.request_residency@[r].block_ids@.subrange(0,
                        old(self).cs.request_residency@[r].block_ids@.len() as int)
                        == old(self).cs.request_residency@[r].block_ids@ by {
                assert(pre_commit.cs.request_residency@.contains_key(r));
                assert(pre_commit.cs.request_residency@[r]
                    == old(self).cs.request_residency@[r]);
                if !emitted@.contains_key(r) {
                    assert(self.cs.request_residency@[r]
                        == pre_commit.cs.request_residency@[r]);
                    assert(self.cs.request_residency@[r].block_ids@.subrange(0,
                        self.cs.request_residency@[r].block_ids@.len() as int)
                        =~= self.cs.request_residency@[r].block_ids@);
                }
            }
            // Per-row slot segments: the plan's export, with admitted rows
            // transported through the commit's residency extension.
            assert forall|k: int| 0 <= k < reprs.scheduled.len()
                implies #[trigger] reprs_row_slots_at(*old(self), *self, reprs, k)
            by {
                reveal(reprs_row_slots_at);
                reveal(crate::exec::cache_scheduler::plan_slot_segments_at);
                assert(crate::exec::cache_scheduler::plan_slot_segments_at(
                    &old(self).cs, &pre_commit.cs, &plan, k));
                let srid = reprs.scheduled[k];
                assert(srid == plan.scheduled_ids@[k]);
                let s0 = reprs.cu_q[k];
                let s1 = reprs.cu_q[k + 1];
                assert(s0 == plan.cu_seqlens_q_repr@[k]
                    && s1 == plan.cu_seqlens_q_repr@[k + 1]);
                let bsz = crate::types::BLOCK_SIZE_SPEC as int;
                if old(self).cs.running@.contains(srid) {
                    // Decode row: the plan left running residencies and all
                    // live states untouched, so the export IS the old-state
                    // fact.
                    assert(pre_commit.cs.request_residency@.contains_key(srid)
                        && pre_commit.cs.request_residency@[srid]
                            == old(self).cs.request_residency@[srid]);
                    assert(old(self).cs.request_residency@.contains_key(srid));
                    assert(pre_commit.cs.live_requests@
                        == old(self).cs.live_requests@);
                } else if reprs.sample_mask[k] {
                    if self.cs.live_requests@.contains_key(srid) {
                        // Admitted survivor: c and n are stable across the
                        // commit, and the block table only grew.
                        assert(pre_commit.cs.live_requests@.contains_key(srid));
                        assert(pre_commit.cs.live_requests@
                            == old(self).cs.live_requests@);
                        assert(old(self).cs.waiting@.contains(srid)) by {
                            assert(plan.scheduled_ids@.contains(srid));
                        }
                        assert(crate::exec::cache_scheduler::waiting_unstarted(
                            &old(self).cs));
                        assert(old(self).cs.live_requests@[srid]
                            .generated_tokens@.len() == 0);
                        let n = self.cs.live_requests@[srid]
                            .prompt_tokens@.len() as int;
                        let kd = reprs.cu_k[k + 1] - reprs.cu_k[k];
                        lemma_waiting_sample_row_is_final(
                            *old(self), reprs, k,
                        );
                        assert(kd == n);
                        assert(self.cs.live_requests@[srid].prompt_tokens@
                            == pre_commit.cs.live_requests@[srid].prompt_tokens@)
                        by {
                            if emitted@.contains_key(srid) {
                                assert(self.cs.live_requests@[srid].prompt_tokens@
                                    == pre_commit.cs.live_requests@[srid]
                                        .prompt_tokens@);
                            } else {
                                assert(self.cs.live_requests@[srid]
                                    == pre_commit.cs.live_requests@[srid]);
                            }
                        }
                        assert(crate::exec::request_state::history(
                            pre_commit.cs.live_requests@[srid]).len() as int == n);
                        // Residency: extension (committed) or equality.
                        assert(pre_commit.cs.request_residency@.contains_key(srid));
                        assert(reprs.scheduled.contains(srid));
                        assert(!(emitted@.contains_key(srid)
                            && crate::exec::request_state::should_finish_after_append(
                                old(self).cs.live_requests@[srid], emitted@[srid])));
                        assert(self.cs.running@.contains(srid));
                        assert(self.cs.request_residency@.contains_key(srid)) by {
                            assert(crate::exec::cache_scheduler::running_has_residency(
                                &self.cs));
                        }
                        let ids_pc = pre_commit.cs.request_residency@[srid]
                            .block_ids@;
                        let ids_new = self.cs.request_residency@[srid].block_ids@;
                        assert(self.cs.request_residency@.contains_key(srid)
                            && ids_new.len() >= ids_pc.len()
                            && ids_new.subrange(0, ids_pc.len() as int) == ids_pc
                            && self.cs.request_residency@[srid].cached_prefix_blocks
                                == pre_commit.cs.request_residency@[srid]
                                    .cached_prefix_blocks)
                        by {
                            if emitted@.contains_key(srid) {
                            } else {
                                assert(self.cs.request_residency@[srid]
                                    == pre_commit.cs.request_residency@[srid]);
                                assert(ids_pc.subrange(0, ids_pc.len() as int)
                                    =~= ids_pc);
                            }
                        }
                        let c = self.cs.request_residency@[srid]
                            .cached_prefix_blocks as int * bsz;
                        // Alignment at the post-plan state pins the budget.
                        assert(crate::exec::cache_scheduler::residency_history_aligned(
                            &pre_commit.cs));
                        assert(pre_commit.cs.running@.contains(srid));
                        let tail_pc = pre_commit.cs.blocks@[
                            ids_pc[ids_pc.len() - 1]].tokens@.len() as int;
                        assert(n == (ids_pc.len() - 1) * bsz + tail_pc);
                        assert(tail_pc >= 1);
                        assert(tail_pc <= bsz) by {
                            assert(crate::exec::cache_scheduler::block_token_bound(
                                &pre_commit.cs));
                        }
                        crate::exec::cache_scheduler::lemma_aligned_blocks_needed(
                            n, ids_pc.len() as int, tail_pc);
                        assert forall|q: int| s0 <= q < s1
                            implies #[trigger] reprs.slots[q]
                                == crate::proof::tensor::geometry::block_table_slot(
                                    self.cs.request_residency@[srid].block_ids@,
                                    (c + q - s0) as nat) as int
                        by {
                            assert(reprs.slots[q] == plan.slot_mapping_repr@[q]);
                            assert(plan.slot_mapping_repr@[q]
                                == crate::proof::tensor::geometry::block_table_slot(ids_pc,
                                    (c + q - s0) as nat) as int);
                            let pos = c + q - s0;
                            assert(0 <= pos < n);
                            crate::proof::tensor::geometry::lemma_blocks_needed_covers_pos(
                                pos as nat, n as nat);
                            assert((pos as int) / bsz < ids_pc.len() as int);
                            assert(ids_new[(pos as int) / bsz]
                                == ids_pc[(pos as int) / bsz]) by {
                                assert(ids_new.subrange(0, ids_pc.len() as int)[
                                    (pos as int) / bsz]
                                    == ids_pc[(pos as int) / bsz]);
                            }
                        }
                    }
                }
            }
            // Cross-request page disjointness: a row's write pages are
            // exclusively owned at the pre-commit state, so no OTHER
            // request's covered prefix can reach them.
            assert forall|k: int| 0 <= k < reprs.scheduled.len()
                implies #[trigger] reprs_rows_disjoint_at(*self, reprs, k)
            by {
                reveal(reprs_rows_disjoint_at);
                reveal(crate::exec::cache_scheduler::plan_slot_segments_at);
                let w = reprs.scheduled[k];
                let s0 = reprs.cu_q[k];
                let s1 = reprs.cu_q[k + 1];
                let bsz = crate::types::BLOCK_SIZE_SPEC as int;
                assert(crate::exec::cache_scheduler::plan_slot_segments_at(
                    &old(self).cs, &pre_commit.cs, &plan, k));
                assert(pre_commit.cs.running@.contains(w)) by {
                    assert(plan.scheduled_ids@.contains(w));
                }
                assert(pre_commit.cs.live_requests@.contains_key(w)) by {
                    assert(crate::exec::cache_scheduler::live_covers_queue(&pre_commit.cs));
                }
                let ids_w = pre_commit.cs.request_residency@[w].block_ids@;
                let hist_w = crate::exec::request_state::history(
                    pre_commit.cs.live_requests@[w]).len() as int;
                let tail_w = pre_commit.cs.blocks@[
                    ids_w[ids_w.len() - 1]].tokens@.len() as int;
                assert(pre_commit.cs.request_residency@.contains_key(w));
                assert(ids_w.len() >= 1);
                assert(hist_w == (ids_w.len() - 1) * bsz + tail_w);
                assert(tail_w >= 1);
                assert(tail_w <= bsz) by {
                    assert(crate::exec::cache_scheduler::block_token_bound(&pre_commit.cs));
                }
                assert forall|q: int, r: RequestId, l: int|
                    #![trigger reprs.slots[q],
                        self.cs.request_residency@[r].block_ids@[l]]
                    s0 <= q < s1
                    && r != w
                    && self.cs.running@.contains(r)
                    && self.cs.live_requests@.contains_key(r)
                    && self.cs.request_residency@.contains_key(r)
                    && 0 <= l < self.cs.request_residency@[r].block_ids@.len()
                    && l < crate::proof::tensor::geometry::blocks_needed_for(
                        (crate::exec::request_state::history(
                            self.cs.live_requests@[r]).len() - 1) as nat)
                    implies reprs.slots[q] / bsz
                        != self.cs.request_residency@[r].block_ids@[l] as int
                by {
                    assert(reprs.slots[q] == plan.slot_mapping_repr@[q]);
                    // The writer's page index in its pre-commit table.
                    let ghost pos: nat = if old(self).cs.running@.contains(w) {
                        (hist_w - 1) as nat
                    } else {
                        (pre_commit.cs.request_residency@[w]
                            .cached_prefix_blocks as int * bsz + q - s0) as nat
                    };
                    if old(self).cs.running@.contains(w) {
                        assert(s1 == s0 + 1);
                        assert(q == s0);
                        assert(reprs.slots[q] == crate::proof::tensor::geometry::block_table_slot(
                            ids_w, pos) as int);
                        assert((pos as int) / bsz == ids_w.len() as int - 1) by {
                            vstd::arithmetic::div_mod::
                                lemma_fundamental_div_mod_converse_div(
                                    pos as int, bsz, ids_w.len() as int - 1,
                                    tail_w - 1);
                        }
                        assert(pre_commit.cs.blocks@[
                            ids_w[ids_w.len() - 1]].refcount == 1);
                    } else {
                        let c_w = pre_commit.cs.request_residency@[w]
                            .cached_prefix_blocks as int * bsz;
                        let end_w = reprs.cu_k[k + 1] - reprs.cu_k[k];
                        assert(crate::exec::cache_scheduler::admitted_pages_exclusive_post(
                            &pre_commit.cs, plan.scheduled_ids@, k));
                        assert(s1 == s0 + (end_w - c_w));
                        assert(reprs.slots[q] == crate::proof::tensor::geometry::block_table_slot(
                            ids_w, pos) as int);
                        assert(pos < end_w);
                        assert(plan_residency_extent_at(
                            &old(self).cs, &pre_commit.cs, &plan, k,
                        ));
                        reveal(plan_residency_extent_at);
                        assert(crate::proof::tensor::geometry::blocks_needed_for(end_w as nat)
                            <= ids_w.len());
                        crate::proof::tensor::geometry::lemma_blocks_needed_covers_pos(
                            pos, end_w as nat);
                        assert((pos as int) / bsz < ids_w.len() as int);
                        assert((pos as int) / bsz
                            >= pre_commit.cs.request_residency@[w]
                                .cached_prefix_blocks as int) by {
                            vstd::arithmetic::div_mod::lemma_div_is_ordered(
                                c_w, pos as int, bsz);
                            vstd::arithmetic::div_mod::lemma_fundamental_div_mod_converse_div(
                                c_w, bsz, pre_commit.cs.request_residency@[w]
                                    .cached_prefix_blocks as int, 0,
                            );
                        }
                        assert(pre_commit.cs.blocks@[
                            ids_w[(pos as int) / bsz]].refcount == 1);
                    }
                    let pi = (pos as int) / bsz;
                    assert(0 <= pi < ids_w.len());
                    crate::proof::tensor::geometry::block_table_slot_block(ids_w, pos);
                    assert(reprs.slots[q] / bsz == ids_w[pi] as int);
                    assert(pre_commit.cs.blocks@[ids_w[pi]].refcount == 1);
                    let p = ids_w[pi];
                    // r at the pre-commit state.
                    assert(pre_commit.cs.running@.contains(r)) by {
                        assert(pre_commit.cs.live_requests@
                            == old(self).cs.live_requests@);
                    }
                    assert(pre_commit.cs.live_requests@.contains_key(r)) by {
                        assert(crate::exec::cache_scheduler::live_covers_queue(
                            &pre_commit.cs));
                    }
                    assert(pre_commit.cs.request_residency@.contains_key(r)) by {
                        assert(crate::exec::cache_scheduler::running_has_residency(
                            &pre_commit.cs));
                    }
                    let ids_r_pc = pre_commit.cs.request_residency@[r].block_ids@;
                    let ids_r_new = self.cs.request_residency@[r].block_ids@;
                    // p is exclusively w's: not in r's pre-commit table.
                    assert(!ids_r_pc.contains(p)) by {
                        if ids_r_pc.contains(p) {
                            assert(crate::exec::cache_scheduler::refcount_valid(
                                &pre_commit.cs));
                            let holders = crate::exec::cache_scheduler::
                                residency_holders_of(&pre_commit.cs, p);
                            assert(ids_w[pi] == p);
                            assert(ids_w.contains(p));
                            assert(holders.contains(w));
                            assert(holders.contains(r));
                            crate::exec::cache_scheduler::lemma_two_holders(holders, r, w);
                            assert(pre_commit.cs.blocks@[p].refcount as int >= 2);
                        }
                    }
                    // r's covered prefix at the new state sits inside its
                    // pre-commit table.
                    let hist_r_pc = crate::exec::request_state::history(
                        pre_commit.cs.live_requests@[r]).len() as int;
                    let tail_r_pc = pre_commit.cs.blocks@[
                        ids_r_pc[ids_r_pc.len() - 1]].tokens@.len() as int;
                    assert(hist_r_pc == (ids_r_pc.len() - 1) * bsz + tail_r_pc);
                    assert(tail_r_pc >= 1);
                    assert(tail_r_pc <= bsz) by {
                        assert(crate::exec::cache_scheduler::block_token_bound(
                            &pre_commit.cs));
                    }
                    crate::exec::cache_scheduler::lemma_aligned_blocks_needed(
                        hist_r_pc, ids_r_pc.len() as int, tail_r_pc);
                    let hist_r_new = crate::exec::request_state::history(
                        self.cs.live_requests@[r]).len() as int;
                    assert(hist_r_new - 1 <= hist_r_pc) by {
                        if emitted@.contains_key(r) {
                            assert(hist_r_new == hist_r_pc + 1);
                        } else {
                            assert(self.cs.live_requests@[r]
                                == pre_commit.cs.live_requests@[r]);
                        }
                    }
                    crate::proof::tensor::geometry::lemma_blocks_needed_monotone(
                        (hist_r_new - 1) as nat, hist_r_pc as nat);
                    assert(l < ids_r_pc.len() as int);
                    assert(ids_r_new[l] == ids_r_pc[l]) by {
                        if emitted@.contains_key(r) {
                            assert(ids_r_new.subrange(0, ids_r_pc.len() as int)
                                == ids_r_pc);
                            assert(ids_r_new.subrange(0, ids_r_pc.len() as int)[l]
                                == ids_r_pc[l]);
                        } else {
                            assert(self.cs.request_residency@[r]
                                == pre_commit.cs.request_residency@[r]);
                        }
                    }
                    assert(ids_r_pc[l] != p) by {
                        assert(ids_r_pc.contains(ids_r_pc[l]));
                    }
                }
            }
            assert(reprs_rows_disjoint(*self, reprs));
            lemma_engine_step_relation_with_payload_intro(
                *old(self), *self, emitted@, samples_g, reprs,
                architecture_step_logits_repr(*old(self), reprs),
                architecture_engine_post_kv_of(
                    *old(self), reprs, old(self).kv_caches_repr@,
                ),
            );
            assert(architecture_engine_step_relation(
                *old(self), *self, emitted@, samples_g, reprs,
            ));
            assert(old(self).kv_caches_repr@.len()
                == reprs.wr.layers.len());
            EA::lemma_architecture_eng_cache_shape_preserved(
                *old(self), *self, emitted@, samples_g, reprs,
            );
            assert(self.weights_perms@ == old(self).weights_perms@);
            assert(self.weights == old(self).weights);
            assert(self.kv_caches@ == old(self).kv_caches@);
            assert(self.runtime == old(self).runtime);
            assert(self.model_config == old(self).model_config);
            assert(engine_step_semantic_identity(
                *old(self), *self,
            )) by {
                reveal(engine_step_semantic_identity);
            }
            assert(eng_execution_perms_ok(self)) by {
                reveal(eng_execution_perms_ok);
                assert(RT::model_execution_valid(
                    &self.weights, &self.runtime, &self.weights_perms@,
                ));
                assert(RT::model_weights_repr_of(&self.weights_perms@).architecture
                    == self.model_config.architecture);
                assert(RT::model_weights_num_layers(&self.weights)
                    == self.model_config.num_layers as nat);
                assert(self.kv_caches@.len()
                    == self.model_config.num_layers as nat);
                assert(self.kv_perms@.len()
                    == self.model_config.num_layers as nat);
                assert(RT::kv_perms_ids_distinct(self.kv_perms@));
                assert(self.kv_perms@.extracted() == Set::<int>::empty());
                assert(RT::kv_cache_tensor_ids_match(
                    self.kv_caches@, self.kv_perms@,
                    self.model_config.num_layers as nat,
                ));
                assert forall|i: int|
                    0 <= i < self.model_config.num_layers as int implies
                        #[trigger] self.kv_caches@[i].0.id()
                            == self.kv_perms@.k_id(i) by {
                    RT::lemma_kv_cache_tensor_ids_match_at(
                        self.kv_caches@, self.kv_perms@,
                        self.model_config.num_layers as nat, i,
                    );
                }
                assert forall|i: int|
                    0 <= i < self.model_config.num_layers as int implies
                        #[trigger] self.kv_caches@[i].1.id()
                            == self.kv_perms@.v_id(i) by {
                    RT::lemma_kv_cache_tensor_ids_match_at(
                        self.kv_caches@, self.kv_perms@,
                        self.model_config.num_layers as nat, i,
                    );
                }
                assert(self.kv_caches_repr@.len()
                    == self.model_config.num_layers as nat);
                assert forall|j: int|
                    0 <= j < self.model_config.num_layers as int
                    implies #[trigger] self.kv_caches_repr@[j]
                        == (self.kv_perms@.k_repr(j), self.kv_perms@.v_repr(j))
                by {}
            }
        }
        (emitted, Ghost(samples_g), Ghost(reprs))
    }

}

} // verus!
