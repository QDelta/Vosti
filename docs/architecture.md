# Architecture

See the [README](../README.md#supported-execution) for supported models,
[verification](verification.md) for proved properties and open gaps,
[models](models.md) for extending a family, and
[deployment](deployment.md) for qualification and startup.

## System boundary

The Rust/PyO3 engine batches requests and stores their KV values in pages.
The Verus proof relates that execution to an independent machine for each
request. This gives request-level determinism for all supported architectures,
under the assumptions in the [verification guide](verification.md).

## Source ownership

The Rust/Verus crate separates these ownership areas:

| Directory | Responsibility |
| --- | --- |
| `src/boundary/` | External bodies and types, uninterpreted numerical specifications, imported kernel certificates, and checked adapters around those declarations |
| `src/types.rs` | Shared engine identities and compiled page constants |
| `src/model_config.rs` | Shared executable configuration schema: architecture tags, geometry, attention kinds, and float-bit parameters |
| `src/exec/` | Verified scheduler mutation, model-forward orchestration, and Engine execution |
| `src/spec.rs` | Human-reviewed engine-independent determinism definition |
| `src/proof.rs` | Checked satisfaction of that definition by the concrete engine |
| `src/proof/` | Checked invariants, semantic lemmas, refinement witnesses, and executable proof drivers |

`audit/` contains the reviewed claim inventory, architecture-ownership manifest,
and generated import records. These files record evidence; the proofs live in Rust/Verus.
`python/vosti_kernels/` contains the Python runtime boundary and family
checkpoint/runtime bindings. `kernels/` is the tracked kernel and verifier
directory in the same repository; build audits pin its source tree.

### Model-family isolation

Each supported family has paired adapters under:

- `src/boundary/model_families/<family>/`;
- `src/exec/model_families/<family>/`; and
- `src/proof/model/families/<composition>/` (shared by compatible families).

Each boundary family defines its checkpoint data and constants in `config.rs`.
Shared geometry and parameter types live in `src/model_config.rs`. Request and
token aliases, together with the compiled page size, live in `src/types.rs`.
Python constants are shared through `kernels/triton_kernels/constants.py`.
Changing a compiled constant also requires reviewing its uses in the
implementation, proofs, and deployment evidence.

The adapters expose common operations backed by the shared `dense_swiglu_*`
and `four_norm_gated_*` decoder semantics and proofs.

`audit/model_architecture_ownership.json` records the supported families, their
proof compositions, and shared kernel interfaces.
`uv run --locked python scripts/project.py architecture` checks required adapter
operations, shared signatures, certificate coverage, and closed Rust dispatch.
It rejects family imports in generic Engine/proof code and imports across family
boundaries. Compilation, Verus, and tests check the implementations separately.

## Two-phase serving

### Phase 1: qualify a deployment

Offline preparation resolves and validates:

- the checkpoint and complete model configuration;
- the target environment and backend;
- the static kernel launch inventory and exact specializations.

It verifies each selected specialization, runs backend probes, and produces a
sealed deployment bundle. The bundle alone cannot enter the Engine.

Checked startup loads weights, binds tracked permissions, and constructs a
closed `ModelRuntime` and `QualifiedKernelPlan`. The latter records the
architecture, the `BackendQualified` state, and a 256-bit `KernelPlanId`
derived from the sealed deployment identity.

The trusted loaders and validators narrow the external assumptions by checking
roles, shapes, dtype, device, layout, attention policy, and aliasing rules.
Their numerical assignment of checkpoint values and their Python-object-to-
ghost equations remain outside Verus.

All families use `static_runtime.admit_runtime` for device/dtype checks,
framework and kernel origins, and sealed-bundle binding. Family runtime classes
retain their numerical policies and immutable launch maps. Their deployment
modules bind the shared workflow to an explicit profile object.

### Phase 2: execute the qualified deployment

`Engine::init` accepts the already selected weights, runtime, permissions,
scheduler configuration, and cache allocation. `model_execution_valid` requires
their architecture tags to agree and requires the runtime plan to be backend-
qualified. The Engine retains that immutable runtime capability.

Once initialized, the serving path performs no model, hardware, or launch-plan
selection. Dynamic scheduling, admission, batching, cache reuse, and eviction
remain runtime choices because they are covered by the Engine relation.

~~~text
Engine::add_request
  -> verified scheduler admission

Engine::step
  -> plan requests and materialize exact step tensors
  -> eager or qualified CUDA-graph model forward
  -> select one logit row per emitting request
  -> deterministic sampling
  -> commit cache and request-state changes
~~~

`Engine::step` calls the architecture-neutral `exec::model::model_forward`,
which performs one closed family dispatch over matching weights and runtime.
Engine itself has no concrete family branch. Its postcondition publishes
`architecture_engine_step_relation`, the common scheduler, layout, cache, logit,
and sampling relation consumed by refinement.

CUDA graph capture/replay is an optional qualified overlay. Exact-signature
replay and padded pure-decode covering share one architecture-neutral checked
adapter; each family supplies the same row-isolation and KV-shape laws. Replay
fidelity remains an external premise.

## Tensor and runtime boundary

### Tracked tensor permissions

`Tensor` is an external executable handle around a Python object. `TensorPerm`
is a linear tracked token that associates the handle's ghost identity with one
semantic representation. Bond predicates such as `tensor_repr_2d` and the
KV-cache bonds connect a runtime handle, permission, and ghost value.

Read-only kernels borrow input permissions and return a fresh output handle and
permission. Mutating kernels such as KV scatter take a tracked mutable
permission whose postcondition advances the corresponding ghost cache value.
Permissions not passed to a call are mechanically framed.

Read-only checkpoint roles may alias, including tied embedding and LM-head
storage, because the weight bundle exposes no mutable borrow. Writable tensors
must be distinct wherever the proof relies on separation.

### External operations

`src/boundary/tensor_runtime.rs` contains the common tensor vocabulary, runtime
sums, representation bonds, and PyO3-facing operations. Focused boundary
modules own model deployment, CUDA graphs, dense-layer primitives, sampling,
and opaque scalar functions.

The boundary includes:

1. Pure or allocating primitives, whose contracts return semantic output
   representations and freshness facts.
2. Mutating primitives such as KV scatter, which exposes the exact cache update needed by
   the refinement proof.
3. Materializers and host adapters. Token, position, slot, cumulative-
   length, block-table, row-selection, and sampling adapters expose exact host
   effects.

Source-span manifests bind reviewed Rust and Python implementations to their
recorded contracts and fail when marked code changes. They detect drift; they
do not prove that Python or CUDA implements the Verus postcondition.

The trusted cache allocator returns all K/V handles and one permission bundle
together. Its contract records page geometry and internal identity separation.
Verified initialization consumes that bundle; it has no verified path for
minting replacement permissions.

## Model composition

`ModelWeightsRepr` contains the compact common layer representation.
`ModelWeightsArchitectureRepr` carries family-specific semantic data projected
from the bound physical weights. Closed sums for physical weights, runtime, and
architecture payload prevent mismatched family combinations.

`proof::model::architecture` is the closed semantic dispatcher. It exposes
whole-model logits, post-store KV semantics, reference logits, request
projection, and cache laws without exposing a family's private proof
implementation. `exec::model` is the corresponding executable dispatcher.

All four families implement the same adapter contracts through two shared
decoder compositions:

- Qwen3 and Llama3 instantiate the shared dense-SwiGLU executable and proof
  composition. Their family adapters supply distinct checkpoint roles,
  immutable configurations, Q/K-normalization policy, and RoPE policy. Tied
  versus untied embedding/LM-head storage is resolved by the immutable
  checkpoint loader and does not select a different forward program.
- Gemma3 and Gemma4 share the four-norm gated composition, with scaled
  embedding, Gemma normalization and GELU behavior, tied output head, and
  per-layer Full or SlidingWindow attention. Each family's adapter supplies
  its geometry and numerical policy.

Shared operations use neutral, parameterized semantics. Geometry and profile
constants remain family data. A new element-wise operation adds a neutral
semantic/kernel contract and is then composed by the family adapter; it does
not require a new scheduler or refinement.

Sliding-window layers currently allocate and retain the complete causal KV
prefix. Window masking and skipped attention work do not license eviction of
rows outside the window.

## Scheduler, cache, and independent model

The cache scheduler owns request lifecycle, block residency, free/cached queues,
prefix lookup, eviction, planning, and commit. Its stable invariants cover:

- valid request and residency maps;
- queue topology and block ownership;
- block-table bounds and sufficient page coverage;
- exclusive writable tails and duplicate-free write slots;
- persistent provenance for reusable prefix pages; and
- alignment between running requests, histories, and cache residency.

`StepReprs` is the architecture-neutral ghost view of a concrete step plan:
packed token/position rows, cumulative lengths, slots, block tables, maxima,
sampling masks, and request order. Model payloads never enter scheduler state.

The `IndependentBatchModel` contains one private contiguous-cache machine per
live request. Refinement relates it to the paged Engine through three distinct
facts:

1. Request-state coherence: Engine and machine views agree for every live
   request.
2. Physical cache coherence: paged Engine reads agree with each machine's
   contiguous cache at resident logical positions.
3. Semantic cache fidelity: machine cache rows equal canonical cold-model KV
   values for the request's causal token prefix.

Request projection and relocation bridge batched paged geometry to singleton
contiguous geometry. The step proof covers decode, fresh prefill, cached partial
prefill, and non-final page-aligned prefill chunks. Non-final chunks update and
publish KV while the abstract request state stutters.

Prefix caching preserves provenance separately from current request ownership.
Retained zero-reference pages may be reused only through the proved registry,
hash, token, and canonical-KV conditions. See
[prefix-cache guide](prefix-cache.md) for the detailed scheduler protocol.

## Verification boundary

See [verification](verification.md) for the specification and proof, and
[kernel contracts](kernel-contracts.md) for generated assumptions and their
checked adapters. The generated [claims](../audit/claims.md) and
[TCB](../audit/tcb.md) inventories list the audited declarations.

## Module map

| Area | Principal modules |
| --- | --- |
| Shared executable vocabulary | `types`, `exec/request_state` |
| Shared model configuration | `model_config` |
| Semantic representations and geometry | `proof/model/types`, `proof/tensor/{types, geometry}` |
| Independent reference model | `proof/reference/{request_machine, independent_batch_model}` |
| Tensor/runtime boundary | `boundary::{tensor_runtime, model_deployment, model_forward_graph, dense_layer_primitives}` |
| Model-family adapters | `boundary/model_families`, `exec/model_families`, `proof/model/families` |
| Concrete serving | `exec::{cache_scheduler, model, engine}` |
| Scheduler proofs | `proof/scheduler` |
| Cache and layout bridges | `proof/cache`, `proof/model/relocation.rs`, `proof/tensor` |
| Refinement and traces | `proof/engine`, `proof/serving` |
| Public specification and interpretation | `spec; proof::serving::{interpretation, transitions}` |

Within `cache_scheduler`, state/types and stable invariants are separated from
planning invariants and from lifecycle mutations. Admission, prefix caching,
reclamation, append, planning, and commit remain separate verified modules.
Small proof modules are also intentional solver boundaries; file size alone is
not a reason to merge them.
