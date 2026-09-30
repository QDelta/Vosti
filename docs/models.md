# Adding a dense model architecture

Adding a dense family requires checkpoint loading, a checked forward
implementation, and proofs of the common model and cache contracts. Families
can share these proofs when they use the same decoder composition. The
scheduler and top-level determinism theorem do not depend on a particular family.

## Files to extend

`audit/model_architecture_ownership.json` lists supported families, their proof
compositions, and kernel interfaces. A new family needs changes in these areas:

| Area | Required change |
| --- | --- |
| `model_config.rs`, `proof/model/types.rs` | Add the shared architecture tag and the semantic configuration representation, including the family extension payload. Keep family-only roles and policy in that payload. |
| `boundary/model_families/<family>/config.rs` | Own the executable checkpoint configuration, fixed family constants, and any checked geometry helpers. Shared geometry and parameter types remain in `model_config.rs`. |
| `boundary/tensor_runtime.rs` | Add closed physical weight/runtime variants, permission projection, execution-validity case, and KV geometry. Reuse the common qualified-plan capability. |
| `boundary/model_deployment.rs` | Add the closed deployment and permission-binding case. |
| `boundary/model_families/<family>/` | Implement the paired deployment, weight, configuration and architecture adapters. Reuse neutral primitives; add family-local primitive adapters only when needed. |
| `exec/model_families/<family>/` | Implement readiness and the checked family forward wrapper. |
| `proof/model/families/<composition>/` | Reuse the shared whole-model, cache, projection, relocation, layer-witness, and Engine-readiness contracts; extend them only for a new composition. |
| `exec/model.rs` | Add one closed matching weight/runtime forward case. Do not add a family case to Engine. |
| `proof/model/architecture.rs` | Add the family adapter to each closed semantic dispatch. Do not import private implementation files. |
| `proof/engine/architecture.rs` | Discharge common Engine readiness through the new adapters. |
| `proof/model/relocation.rs` | Add the closed projection/relocation adapter call. |
| `python/vosti_kernels/model_families/<family>/` | Add checkpoint loading, physical binding, profile selection, deployment binding, runtime capability, and scope data that cannot be neutral. |
| `audit/model_architecture_ownership.json`: `kernel_qualification` | Register the family's static qualification inventory. Reuse shared `kernel_interfaces`; do not create a family-owned import catalog. New geometries must be qualified and covered by the shared generated interface. |
| Shared family discovery | Add the family to boundary/execution `model_families/mod.rs`, bind its proof composition in the ownership manifest, and update Python discovery/support tables, reference/smoke tables, and explicit aggregate gates. |
| Examples and tests | Reuse common serving/benchmark support; add thin checkpoint policy, CPU contract tests, reference differential tests, and per-family GPU smoke gates. |
| `audit/model_architecture_ownership.json` | Register the family, its proof composition, and qualification inventory. Private helper files and experiments need no entries. |

Kernel source identities live once in `python/vosti_kernels/kernel_catalog.json`.
Each family's `scope.json` binds wrapper roles to those kernel names and declares
its launch profiles. `model_profile.load_scope` expands these references for both
runtime admission and audits; family files cannot override the shared identities.

Family discovery is checked across boundary, execution, proof composition, and
runtime layers. `uv run --locked python scripts/project.py architecture` rejects
missing adapters/operations, mismatched shared signatures, cross-family imports,
and concrete family cases in `Engine`. Experiments may cover any subset of models.

## Common proof contract

Every family proof adapter must provide checked operations for:

1. whole-model logits and post-store KV semantics;
2. cold full-history reference logits;
3. cache-refinement support and canonical-cache laws;
4. request-projection readiness and logits/KV isolation;
5. logits/KV shapes and empty-forward framing;
6. per-layer KV rows and scatter-store correspondence;
7. logical-prefix relocation across cache layouts;
8. Engine cache-shape preservation; and
9. executable forward readiness and reduction to the closed semantic dispatch.

Generic proof modules match the architecture tag and call those operations.
They do not depend on whether the implementation is a direct fold or a private
recursive proof.

The family is covered by the top-level theorem only after all these contracts
are connected through the dispatchers. A working forward implementation alone
does not complete that proof.

## Reusing and adding primitives

Prefer neutral, property-oriented primitives:

- shared semantics are parameterized by shape or policy;
- family/profile modules provide concrete geometry and launch choices;
- generated contracts describe exact qualified specializations; and
- the family forward composes those effects in checked Rust/Verus.

For a new element-wise or row-wise operation, add the neutral opaque semantic
operation, offline kernel verification, checked shape and row-projection adapter,
runtime wrapper, and source attestation. Axis-projection certificates stay out
of the engine imports; their launch configurations are qualified offline.
The raw numerical cell and
Python/Torch/CUDA correspondence remain explicit trust unless separately
proved. This addition should not affect scheduler or top-level refinement code.

A new attention or cache operation needs a
state/effect contract, projection and relocation laws, cache-fidelity rule, and
the scheduler premises required by its launch. Linear attention, for example,
cannot be treated as a new element-wise layer because it changes the persistent
state abstraction.

## Profile addition versus family addition

A new size is a profile addition when checkpoint roles, layer order, attention
policy, and semantic operations are unchanged. Add its complete resolved
configuration and static launch inventory to the family scope. Import one proof
case per distinct specialization; profiles sharing a specialization should
reuse its generated declaration.

Fail-closed profile selection must check:

1. the complete resolved model configuration;
2. equality of derived and declared launch inventories;
3. an imported proof case for every distinct required specialization;
4. rejection of unknown specializations; and
5. fresh generated modules and import records.

If a size changes roles, layer composition, state, or attention semantics, it is
a family/architecture change rather than a profile-only addition.

## Shared compositions

Each family implements the same public contracts. The current families share
two decoder compositions:

| Difference | Why it remains family-owned |
| --- | --- |
| Qwen3/Llama3 profile and checkpoint adapters | The two families instantiate one dense-SwiGLU composition but retain different physical roles, Q/K-normalization choices, RoPE policies, and fail-closed profiles |
| Gemma3/Gemma4 checkpoint and profile adapters | Both use the shared four-norm gated composition and Full/SWA proofs; checkpoint roles, geometry, and numerical policy remain family data |

CUDA graph replay and padded decode also use shared proofs. Each family must
satisfy the same row-isolation, padding, and replay adapter contracts.

## Current conservative SWA contract

Sliding-window attention is supported only with complete causal KV storage and
complete-prefix proof premises. A window mask is not evidence that older KV
rows are irrelevant. Memory-efficient eviction requires a separate theorem
proving independence from every row outside the configured window, followed by
weakened cache-fidelity and allocation contracts.

## Validation

Run at least:

~~~bash
uv run --locked python scripts/project.py architecture
make verify
uv run --locked python scripts/project.py examples
make test
python3 scripts/audit/claim_ledger.py
python3 scripts/audit/tcb.py --check
~~~

When kernels or qualification scopes change, also run:

~~~bash
make generate
make check
~~~

Reference-model and GPU smoke/qualification gates are empirical checks, not
substitutes for the proof gates. Record the exact checkpoint, deployment seal,
hardware, backend versions, workload, output-token comparison, and concurrent
GPU state.
