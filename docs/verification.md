# Verification

Qwen3, text-only Gemma3/Gemma4, and Llama3 use the same
top-level theorems; Qwen3 and Llama3 share the dense-SwiGLU proof composition,
while Gemma3 and Gemma4 share the four-norm gated composition.

## 1. Read the specification

[`spec.rs`](../src/spec.rs) defines finite
executions from initialization, explicit initial request inputs, per-step
admission and output maps, a fixed per-request `view`, and `deterministic(system)`.
Only the state type is abstract. A request input contains its prompt and initial
sampler state; an output record contains logits and an emitted token. Its only
external types are the opaque `Scalar` and `SamplerState`.

The engine-facing interpretation is
[`interpretation.rs`](../src/proof/serving/interpretation.rs):
`system(model, plan)` supplies engine initialization and structural admission/
compute transitions. Outputs are constructed from the actual emitted map and
the batched-forward rows used by sampling, not from reference outputs. The
interpretation binds one exact semantic model and qualified kernel plan.
Its runtime-identity clause follows from the checked `Engine::step`/`add_request`
postconditions. The abstract specification has no
configurable input/output projection that could erase or replace observations.
Its shared API vocabulary lives in
[`transitions.rs`](../src/proof/serving/transitions.rs), independently
of the public definition and internal trace witnesses. Internal semantic invariants and their checked
preservation live in [`refinement.rs`](../src/proof/serving/refinement.rs).

[`lemma_serving_deterministic`](../src/proof.rs) proves
`deterministic(system(model, plan))` for every model and plan, subject to the
attention numeric-domain premise and the runtime/kernel trust boundaries.
For requests with equal prompts and initial sampler states, the complete
logits/token views are prefix-comparable, even when their IDs, admission times,
batching, cache state, lifecycle controls and output lengths differ.

The constructive connection is split into three layers:

1. [`records.rs`](../src/proof/serving/records.rs) proves exact output
   domains/tokens and identifies each logit row with the row used by sampling.
   The one-step theorem supplies reference-logit equality separately.
2. [`trace.rs`](../src/proof/serving/trace.rs) proves that every
   initialized API-call chain yields a legal record trace. The accepted-ID ledger
   establishes freshness and rules out unknown outputs. Conversely, legal record
   executions reconstruct structural API witnesses with the same records and
   endpoints. The initial semantic invariant, model and kernel plan are preserved.
3. [`consistency.rs`](../src/proof/serving/consistency.rs) proves each
   request's view is a prefix of one internal reference run determined by its
   input and model. The two-trace property follows from prefix comparability.

Neither these reference runs nor IBM witnesses appear in the public predicate
or concrete transition premises. Dependency-closure tests enforce that separation.

The internal `proof::serving::continuation_agreement::lemma_request_prefix_agreement`
also handles requests with partially generated histories. The main theorem
does not depend on it.

The public input record omits EOS tokens, `ignore_eos`, `max_tokens`, and
request IDs: these may affect how long a request runs, but corresponding
outputs must still agree. No theorem here requires progress or completion.
Review must establish that the formal definition and its concrete
interpretation capture the intended claim.

## 2. Check architecture coverage

Architecture coverage passes through four closed gates:

1. `tensor_runtime::model_execution_valid` matches physical weights,
   permissions, runtime architecture, and a backend-qualified plan;
2. `exec::model::model_forward` dispatches to one checked family adapter;
3. `proof::model::architecture` dispatches the corresponding semantic laws; and
4. `cache_refinement_supported` admits the family to the persistent cache proof.

`Engine` contains no concrete family case. `uv run --locked python scripts/project.py architecture` checks
the required adapters, shared interfaces, and closed dispatch boundaries. The current closed cases
are dense text-only Qwen3, Gemma3, Gemma4, and Llama3. Each family has physical
Engine admission, a checked forward caller, semantic laws, and cache refinement.
The generic property theorems cover these admitted profiles under their stated
runtime and kernel premises. The Gemma families share the four-norm model
proofs and cache laws; family-specific geometry stays in their bindings.

The shared system instantiation fixes the complete 256-bit plan identity. The theorem quantifies over dynamic scheduling, batching, and cache
behavior inside one deployment qualification; it does not compare different
kernel configurations or hardware environments. See
[architecture.md](architecture.md) for the executable dispatch and optional
CUDA-graph path.

## 3. Review the family proof contract

Each `src/proof/model/families/<composition>/mod.rs` supplies the same operations for:

- whole-model logits and post-store KV representations;
- cold full-history last-row logits;
- cache-refinement laws;
- packed-request logits/KV isolation;
- output/cache shapes and empty-forward framing;
- per-layer K/V row witnesses and scatter-store equality;
- logical-prefix relocation; and
- engine cache-shape preservation and executable-forward readiness.

Generic proofs call the shared model proofs through family bindings.
Qwen3 and Llama3 use `proof::model::families::dense_swiglu`.
Gemma3 and Gemma4 use the neutral `four_norm_gated_*` semantic fold,
batch/Full-SWA relocation and canonical-prefix cache proofs, and checked decoder.
The cache proof binds a semantic model to its shared decoder through the
checked architecture dispatcher; it does not assume a concrete model tag.
The Gemma adapters fix their respective numerical policies.

## 4. Follow the persistent refinement

`src/proof/engine/refinement.rs` relates the engine to an independent batch
model (IBM), which holds one reference machine per live request.
`architecture_persistent_semantic_runtime_inv` combines scheduler coherence,
model identity, cached-value fidelity, and the runtime premises.

Initialization establishes this invariant. Admission adds a reference machine
without changing existing cached values. Each engine step advances the
scheduled reference machines and preserves the others. The proof then shows
that selected logit rows match cold reference execution and that equal sampler
states produce equal tokens. The executable wrappers connect these results
to the actual engine calls; trace induction extends them to finite executions.

The core step theorem contains no concrete family match. It consumes common
scheduler geometry, `architecture_engine_step_relation`, and the closed model
adapter laws.

## 5. Follow one executable step

```text
architecture_persistent_semantic_refinement_observable_step_exec
  Engine::step
    CacheScheduler::plan
    step_core
      exec::model::model_forward
        exec::model_families::<selected family>::model_forward
      select last query rows
      deterministic sample
    CacheScheduler::commit
  architecture_persistent_semantic_refinement_observable_step
    lemma_architecture_step_samples_match_reference
      lemma_architecture_step_logits_match_reference_with_ibm
    architecture_persistent_semantic_refinement_step
      abstract_step::architecture_stepped_ibm
      proof::cache::semantics
      proof::cache::coherence
```

`Engine::init`, `Engine::step`, scheduler plan/commit, and all supported family model
orchestrations have checked Rust/Verus bodies. Family-specific numerical
primitive effects may remain source-attested external boundaries; layer order,
freshness, the KV permission fold, row selection, and logits construction are
checked composition.

The executable entrypoint additionally requires
`paged_attention_numeric_domain()`. If a CUDA graph overlay is supplied, it also
requires replay fidelity and the common graph capability. The padded-decode
optimization is proved once over the closed architecture dispatcher. Gemma's
family discharge covers both Full and SlidingWindow attention without claiming
window-local KV dependence.

## 6. Separate structural and observable facts

`independent_batch_model::ibm_step` consumes the supplied sample map.
`engine::architecture_engine_step_relation` binds that map to selected logits
from the verified family-dispatched forward. The sampler-free theorem first
proves row equality; the sample theorem is then a congruence result using equal
sampler states.

Cache/model refinement therefore does not assume the output token it is meant
to prove. The top-level one-step property exposes logits equality independently
of sampling policy.

## 7. Review cache coherence and fidelity

The common cache path is:

```text
proof/model/architecture.rs
  family-dispatched layer K/V rows and cache laws

proof/model/family_layout.rs
  neutral physical/logical layout lemmas

proof/cache/plan_witnesses.rs
  neutral unscheduled/no-touch witnesses

proof/engine/abstract_step.rs
  architecture_stepped_ibm

proof/cache/coherence.rs
  architecture-generic engine/IBM coherence

proof/cache/semantics.rs
  IBM cache extension + physical provenance preservation
```

`cache_provenance::provenance_cache_fidelity` certifies canonical values for
resident physical provenance chains independently of a live donor request. It
is preserved across admission, publication, reuse, append, deallocation, and
eviction. Family cache adapters prove that their post-store K/V fold satisfies
the common canonical continuation laws.

For sliding-window attention, these proofs conservatively retain and compare
the complete causal prefix. They validate SWA execution and relocation, not
window-only dependence or memory-efficient KV eviction.

## 8. Review the kernel and runtime boundary

For each external kernel/runtime contract compare:

1. the Verus `requires`/`ensures`;
2. the actual Python/Torch wrapper and tensors passed;
3. the pinned Triton annotation and specialization;
4. the generated ContractIR artifact and checked adapter; and
5. the audit import record and explicit checked consumer.

Do not conflate evidence classes:

- Verus checks model composition and refinement bodies.
- Generated certificates import kernel relational and exact copy/frame properties.
- Source attestation binds reviewed host spans.
- Backend qualification tests the backend assumptions on a particular device;
  it does not prove compiler or GPU correctness.
- Python/Torch object correspondence, physical layout/non-aliasing, opaque numeric
  cells, and compiler behavior remain in the TCB. The KV scatter's logical effect
  is derived through its annotation-generated theorem and checked adapter.

All supported families compose explicit primitive calls in checked adapters. Shared
layout, sampling, and runtime mechanics remain neutral; profile-specific
operations retain family qualification. Exact host/GPU effects remain trusted
unless the claim ledger binds a call to a theorem-reachable imported contract.

## 9. Validation gates

The [setup guide](setup.md#cpu-side-checks) lists verification, regression,
audit, and regeneration commands. Whole-crate Verus verification and fresh
kernel certificates are required for the complete claim; module-focused
verification is partial evidence. Tests and independent diagnostics are not
additional premises of the relational proof.

## Solver-friendly proof boundaries

Keep large invariants opaque in callers that only preserve them through checked
calls. Reveal individual facts through small lemmas when needed. This limits
unfolding without weakening contracts. Collection lemmas should be independent
of scheduler state where possible.

Check quantifier triggers for self-expansion: a rule triggered by `cu[r]` that
introduces `cu[r + 1]` can trigger itself indefinitely. Profile expensive queries
and test both successful callers and callers missing required premises.

After proof changes, measure the affected functions with fresh compilation and
record solver seeds and worker counts. Check alternate seeds for expensive
queries, then verify the whole crate. A faster isolated query can still make
its callers slower.

## Evidence and limitations

The principal checked entrypoints are:

| Property | Entrypoint |
| --- | --- |
| Initialization | `proof::serving::refinement::lemma_init_establishes_serving_inv` |
| One engine step | `proof::serving::refinement::lemma_step_logits_match_reference` |
| Static/dynamic serving determinism | `proof::lemma_serving_deterministic` |
| Executable initialization and step refinement | `proof::engine::refinement::architecture_persistent_semantic_refinement_*_exec` |

The project distinguishes the following evidence classes:

| Evidence | Establishes | Does not establish |
| --- | --- | --- |
| Verus-checked bodies | Logical composition, invariants, refinement, and theorem implications under their premises | Truth of external bodies or uninterpreted numerical functions |
| Generated kernel certificates | Recorded structural or relational properties for exact specializations | General numerical correctness or compiler correctness |
| Source-span and import ledgers | Identity and closed inventory of reviewed contracts and consumers | Behavioral correctness of Python, Torch, or CUDA |
| Backend qualification | Empirical rejection of incompatible environment and configuration bundles | A formal proof of compiler or GPU execution |
| Runtime differential tests | Falsification evidence for concrete model and hardware runs | Universal correctness |

The theorem compares executions within one qualified plan. It does not prove
model numerical correctness, equality between fused and unfused formulations,
or cross-plan/cross-hardware equality. Runtime finiteness remains an explicit
attention premise; backend probes do not establish it universally. Source
attestation does not itself seal the final compiled GPU artifact.

Empirical records must identify the model/configuration, exact prompts and
token comparisons, hardware, contention state, and backend versions. Keep
measurements with their run artifacts, separate from the checked claims.

### Open gaps

1. Memory-efficient SWA eviction requires independence from every KV row
   outside the configured window before evicting those rows or weakening cache
   fidelity.
2. Many numerical kernel cells remain abstract or are imported
   through relational rather than complete real/BF16 numerical theorems.
3. Cross-language correspondence for PyO3 object identity, tensor values,
   compiler behavior, and backend execution remains external.
4. Different legal launch configurations, compiler
   builds, or hardware qualifications are not proved mutually equivalent.
5. Scheduler safety does not establish eventual service or
   completion.
6. MoE requires routing and expert-composition
   contracts; multimodal processing and linear-attention state are outside the
   current model and cache abstractions.
## Soundness rules

The following are part of the review discipline:

1. `TensorPerm` must remain unforgeable and non-cloneable outside trusted
   allocators/binders.
2. Every representation bond must connect the permission and runtime handle to
   the same ghost identity and representation kind.
3. Allocator and binder contracts must state every freshness or alias-separation
   fact used by checked code.
4. Writable Python tensors must not covertly share storage.
5. Runtime bridges must implement the exact reviewed pre/postconditions and
   argument order to which their source spans are bound.
6. Imported certificates must match the deployed source, specialization, and
   launch identity.
7. A qualified deployment seal must bind all behavior-determining static
   choices not already covered by the Engine theorem.

Tracked linearity makes framing local: a kernel receives only the permissions
it touches. Avoid universal freshness and pairwise whole-engine framing
conditions; use named bundle predicates and scheduler invariants so the solver
instantiates only relevant identities.

## Verus design choices

- Mutable heap classes are represented by explicit state-passing structs and
  tracked permissions.
- Opaque Dafny predicates become closed spec functions with targeted reveal.
- Sequence, map, and set reasoning uses `vstd` datatypes.
- Family-private recursive proofs may be split for solver stability, but their
  exported contracts remain paired and architecture-neutral.
