# Kernels and verifier

The kernel verifier translates a restricted Triton subset into a typed IR and
uses Z3 to prove relational
equality of selected output regions under declared input relations.

## Offline tuning screens

Offline launch-configuration screens live in `benchmarks/attention.py` and
`benchmarks/matmul.py`. From the repository root, use the shared environment:

```sh
uv run --locked python kernels/benchmarks/attention.py --help
uv run --locked python kernels/benchmarks/matmul.py --help
```

Use an idle GPU for measurements. `scripts/common/gpu_monitor.py` can record
contention during research runs. The screens retain their seeds, candidate
configurations, source hashes, raw timing samples, and sampled bytewise
row-consistency checks. They do not alter selectors or qualify deployment
configurations; the ordinary verification and deployment gates still apply.

## What we verify

The annotation-driven verifier handles batch/row invariance and selected-row
causal-prefix equality through the same proof rules:

1. Ordered iteration alignment: contributing iterations pair in source
   order. Unequal ranges require proof that every one-sided iteration is an
   exact identity transition; selected grid instances may differ across runs.
2. Tensor-region coverage/equivalence: every tensor-valued input in the
   selected output's backward value slice has an explicit equality assumption,
   and the conservatively inferred regions fit corresponding parts of it.
3. Scalar-value equivalence: scalar expressions broadcast into tensor
   computation, and path conditions for contributing writes, are equal.
4. Output coverage: every cell of every region listed in a proof goal's
   `post(...)` clause must be
   reached by a source write.  The checker constructs and validates concrete
   grid/local-loop witnesses for supported rectangular tilings; unsupported
   tilings fail closed. A named proof goal may cover only part of
   a kernel's memory effects; the consuming framework separately checks that
   its engine contract names every output it relies on.

These checks support a trusted meta-argument: the same translated expression
graph, paired loop iterations, equal scalar values, and equal corresponding
tensor values produce equal selected outputs. Tensor floating-point execution
is not encoded into Z3, so this last semantic lift is part of the verifier TCB.
The inferred regions are conservative bounding rectangles, not a proof that
the physical GPU load trace is identical.

### Proof-gated relational ContractIR

`verify_annotations` returns a canonical `VerifiedDataflowContract`
only after all equality/output obligations and annotation satisfiability pass.
It records the proof's typed `Kernel` and `RelationalProofGoal`, declared and
specialized parameter types, constants, all `same`/`pre`/`post`/`singleton`
clauses, the goal name, source digest, guarded dependencies, and ordered-operation
evidence.
Failed, unsupported, or unqualified proofs produce no accepted artifact.

`normalize_axis_projection_contract` is a separate fail-closed semantic bridge
for the general selected-axis theorem family. It accepts only complete
selected rows on every output, complete selected rows or whole shared regions
on every input, exact singleton/bounds premises, and explicitly shared
non-batch dimensions and scalar parameters. Constant and composite extents are
matched against their declared symbolic origins and proved specialization.
Scalar equalities are discharged only when they are the exact element equality
implied by a complete rank-one projection; every other domain premise is
preserved. A partial-output theorem may be valid regional evidence but cannot
be imported as whole-row equality.

Paged attention consumes the qualified raw theorem directly through checked
engine representation adapters; there is no intermediate ragged certificate.
Batch projection and selected-row prefix equality remain separate annotated
goals over one execution. A shared strict schema validator rejects unknown or
non-canonical ContractIR before either normalization or raw export.

`ir.relational_dataflow` tracks forward facts, backward dependencies, and the
order of operations. Each use is associated with the definitions that reach it,
including branch joins and loop back edges. A later assignment cannot justify
an earlier use; facts from one iteration cannot be assumed after a possibly
empty loop. The [review invariants](#analysis-review-invariants) list the
corresponding regression tests.

Floating operations retain their ordered expression trees. The scalar solver
treats floating values as opaque: equal operands imply equal results for the
same operation, but real-number cancellation and reassociation are unavailable.
Signed-zero literals remain distinct. Annotation arithmetic describes integer
geometry; floating annotation values support equality only.

Qualification also requires `annotation_preconditions_satisfiable: true`.
A separate query checks scalar and quantified premises, tensor-region
equalities, and shape constraints. A bounded witness can establish
satisfiability; proving unsatisfiability requires an unrestricted query.
Unknown results are rejected. Witness-search bounds never restrict the equality
theorem. This check does not prove physical GPU-input realizability or discharge
extra numerical assumptions from the analysis.

`ir.relational_verifier` checks the result against the prepared theorem before
issuing an artifact. The axis normalizer rejects extra analyzer assumptions;
the raw Verus renderer preserves them as explicit uninterpreted preconditions.
Their labels and kinds are recorded, but their meaning is not inferred from
diagnostic text. Interpreting these conditions remains trusted.

Both attention goals use this raw export. Verus checks the adapters that derive
engine-level relations from it. The exported assumptions still depend on the
Python verifier and exporter; they are not independently checked proof objects.

Engine integration bindings and checked representation adapters are described
in the [kernel-contract guide](../docs/kernel-contracts.md#checked-engine-adapters).
They remain separate from the operation-neutral verifier and exporter.

### Generic ContractIR-to-Verus lowering

`ir.verus_contract` lowers a proof-gated relational artifact without
selecting an operation family. It emits typed launch-state/free-variable
records, explicit rectangular tensor-shape predicates, every raw `same`,
`pre`, `post`, `given`, quantified, regional, and `singleton` clause, and
an abstract execution function whose written cells are uninterpreted while
allocation shape and unwritten launch fields are preserved by construction.
For supported whole-launch schemas, the generated `external_body` certificate
states the proved implication from the raw precondition to the raw postcondition
over that execution. A source/grid iterator must not appear as an unbound
annotation coordinate: an iteration-scoped family of premises is not one
caller-chosen integer. Such exports fail closed pending an explicit quantified
input relation.

The generated certificate is a source- and specialization-bound trusted import,
not an independently checkable proof object. A
consumer must still bind its framework tensors and semantic function to the
generated launch state/execution and discharge the generated raw domain. That
adapter is separate so framework policy stays outside the
operation-neutral lowering.

The current lowering rejects syntax it cannot represent exactly, including
floating specialized constants in expressions, ordered/arithmetic operations
over opaque floating values, malformed tensor ranks, and unguarded universal
clauses. From `kernels/`, verify and emit one standalone fragment:

```bash
uv run python scripts/transpile_verus_contract.py \
  triton_kernels/silu_mul.py silu_mul_kernel \
  --constant BLOCK_M=1 --constant BLOCK_N=4096 \
  --standalone --output ./silu-contract.rs \
  --manifest ./silu-contract.json
verus ./silu-contract.rs
```

To emit several named annotation theorems over one opaque execution model,
repeat `--goal`. All goals must verify
for the same kernel specialization. The generated module declares execution
once and keeps each goal's preconditions, including any numeric obligations,
separate. Written outputs are obtained from typed IR, even when a particular
goal does not relate them. This raw export does not replace the engine's
representation adapters. Set `VERUS` to the verifier executable to enable the
positive/negative caller tests in `tests/frontend/test_shared_execution_contract.py`.

Production kernels use this raw export for both relational and exact-effect
goals. Framework integration and residual trust are described in the
[kernel-contract guide](../docs/kernel-contracts.md).

## Trust boundary and current limitations

- The Triton-to-IR translation, type/region analyses, Z3 encoding, and the
  structural-to-value meta-argument are trusted components.
- Proof entrypoints reject every residual type-inference shape equation.  The
  type checker can expose equations to non-proof callers, but a certificate
  cannot silently assume one unless it is first represented as a checked
  contract hypothesis.
- Floating data casts and `tl.log2` are represented as identity in the IR. This
  is adequate only for dependency/equality reasoning when both sides execute
  the same operation; it is not a numerical model. Integer casts are rejected
  unless they are redundant `.to(tl.int32)` casts on scalar loads from trusted
  `tensor(..., int32, ...)` metadata, so narrowing cannot silently change an address. The
  paged-attention `lse` write is a `post(...)` obligation, but its equality proof
  relies on this same-operation abstraction rather than a numerical model of
  `tl.log2`.
- Mask-aware positional facts narrow backward dependencies. Ordered loop
  alignment proves equal contributions over the shared range; one-sided
  iterations require exact identity transitions. Both attention variants use
  these general rules for selected-row causal-prefix equality, with reported
  finiteness assumptions and the backend's zero-lane dot-accumulation rule.
  The dependency-to-value argument remains verifier TCB. Numerical zero does
  not specify a sign: finite `x * 0` retains its sign dependency, and finite
  `x + 0` is not a bitwise state identity. The explicit all-false-row state
  guard does not rely on that algebra.
- `ir/relational_dataflow.py` is the goal-driven composition of the forward
  positional facts, guarded backward dependencies, and the ordinary regional
  equality prover. It accepts straight-line and soundly merged branch bodies,
  stateless source-loop maps, same-range ordered folds, and unequal ascending
  ranges only when a generic identity-transition proof erases every one-sided
  iteration. It rejects unsupported nested recurrence and partial-state
  patterns instead of summarizing one symbolic body execution as a whole
  fold. Its artifact is consumed by production batch-contract paths after
  the shared qualified verifier. Production selected-row paths consume the same
  artifact format and proof entrypoint.
- The bounded suffix-independence evaluator is a regression/falsification
  check, not an unbounded proof component. It fully unrolls a few concrete
  shapes and checks the final symbolic output for suffix variables. This tests
  the translator, recurrence, and epilogue independently of the proof pipeline.
- Every `pre(...)` clause remains an ordinary conditional theorem assumption.
  Premise satisfiability is separate from the equality proof obligation:
  if the conjunction is inconsistent, the Hoare-style result is sound but
  vacuous. Unified artifact qualification additionally requires a satisfying
  typed annotation input pair. Analysis-internal strengthened contexts, such as paired effectful
  iterators, still require satisfiability so a proof tactic cannot manufacture
  vacuity. Framework callers must establish the generated premises for every
  reachable launch.
- `diagnostics/race.py` is a separate, fail-closed SMT pass for logical write
  disjointness across every pair of distinct program instances. It includes
  lexical branch guards, inner-loop ranges, store masks, all pairs of store
  sites on each logical tensor, and quantified premises such as KV-slot
  injectivity. No regional, causal, or relational result depends on this pass.
  It does not prove physical race freedom: the IR discards numeric strides and
  treats different tensor parameters as separate logical address spaces.
  Injective logical-to-physical layout, writable no-alias, address arithmetic,
  and analyzed-to-executed launch correspondence remain explicit external
  obligations. `require_writable_tensors` is runtime defense in depth, not a
  static proof bridge.
- `same(tensor)` means whole-tensor value equality. A sliced `pre(...)` equality
  is preferable when only a request-local region is shared.
- `@params` is trusted tensor-shape and logical element-kind metadata where raw
  Triton pointers do not expose a full allocation type. In particular,
  `tensor(..., int32, ...)` promises a signed-int32 runtime tensor. The front end checks
  annotation scope, grid axes, stride-to-axis mappings, and direct
  block-pointer shapes, but a pointer-offset view's full allocation shape and
  the runtime dtype correspondence still come from `@params` and the launch
  bridge.
- Unchecked block-pointer axes are modeled as physical accesses, not as
  zero-padding. Their in-bounds memory safety must be established by kernel
  preconditions and launch geometry; there is not yet a general memory-safety
  proof pass.
- The source front end checks definite assignment before IR locals are hoisted,
  rejects value reassignment captured by a live block-pointer/alias, and accepts
  unsqueeze indexing only when every other index is a full `:` slice. These are
  deliberate subset restrictions, not Triton language restrictions.
- Cyclic tensor updates are accepted only when region propagation is
  element-wise. The recurrence/iteration semantic lift remains part of the
  trusted region-analysis argument rather than an encoded tensor semantics.

### Deployment backend probes

The verifier necessarily assumes semantics for the Triton operations it
models, including lane locality, exact nonnegative integer address arithmetic,
masking, reductions, shape mappings, and selected IEEE-754 special values.
Verifying the Triton compiler and GPU backend is outside this repository's
scope. Instead, the consuming Vosti deployment pipeline exports every used
operation and assumed property from the exact proved kernel shapes, selects a
static launch configuration, and runs CUDA probes on the deployment host before
sealing that configuration.

`ir/backend_requirements.py` extracts these requirements from the translated
IR. `backend/probe_contract.py` is a pure, fail-closed mapping from each
operation/property pair to its permitted probe family, while
`backend/probes.py` contains the CUDA implementations. Unknown operations,
properties, probe families, schema fields, or failed observations are rejected.
The contract also independently reconstructs the required property set for
conditional arithmetic/cast signatures, so deleting a verifier-used property
from the exporter fails closed. Analysis-consumer labels are a closed set.
These modules do not participate in proof acceptance: the probes are empirical
environment qualification and falsification tests, not formal proofs.

The probes run bounded inputs for every exported obligation,
every applicable declared BF16/FP32 physical dtype, and every exact sealed
`num_warps`/`num_stages` context. They exercise locality, repeatability,
selected arithmetic identities, shape mappings, masks, and memory behavior;
they do not exhaust either floating-point domain. Direct relational tests of
the serving kernels provide a separate empirical check.

The probes do not yet reproduce every block-pointer stride/order expression,
and compiler/artifact provenance remains incomplete. The
proof-rule-to-requirement inventory and probe implementations therefore remain
reviewed trusted code. See [deployment](../docs/deployment.md) for bundle
versioning, environment binding, sealing, and runtime rejection rules.

## Quick start

Run from the project root using its shared environment:

```bash
uv sync --locked

# Core verifier regression suite
CUDA_VISIBLE_DEVICES='' uv run --locked python scripts/project.py test --suite kernels

# Unified batch and selected-row certificates at deployed geometry
make verify-kernels

# Optional independent diagnostics
uv run --locked python scripts/project.py diagnostics
```

The root [`pyproject.toml`](../pyproject.toml) and [`uv.lock`](../uv.lock) own
the environment for both the engine and kernels; there is no nested Python
project or virtual environment. The Make targets preserve the verifier's
working directory while explicitly selecting that root environment.
The CPU proof and regression paths do not require a CUDA device;
backend probes and direct kernel execution do.

## Repository layout

| Path | Responsibility |
| --- | --- |
| `ir/` | Triton subset validation, typed IR, structural/relational analyses, certificate schemas, Z3 checks, and generic Verus lowering |
| `ir/proof_preparation.py` | Shared translation, specialization, and named-goal preparation |
| `ir/smt.py` | Shared scalar SMT encoding and obligation results |
| `ir/regional_obligations.py` | Regional obligations consumed by unified relational analysis; no artifact producer |
| `diagnostics/` | Optional logical write-disjointness and bounded suffix checks |
| `triton_kernels/` | Kernels and runtime defenses reachable from the consuming Vosti framework |
| `backend/` | Closed primitive-obligation registry and empirical CUDA probes |
| `scripts/` | Suffix and standalone-contract drivers |
| `tests/` | Front-end, analysis, diagnostic-independence, backend, and runtime-defense regressions |
| `example.py` | Minimal annotation-driven matmul translation and proof example |

## Pipeline overview

The proof path translates directly from the current source on every run:

```
Triton source (.py)
  -> translate_kernel_source()       # fail-closed subset gate + translation
  -> check_variable_names()          # validate naming conventions
  -> check_tensorindex_readonly()    # forbid TensorIndex reads from written tensors
  -> specialize_kernel_constants()   # substitute block sizes
  -> expand_let_bindings()           # inline let expressions
  -> ensure_tensor_var_sizes_known() # require concrete tensor extents after specialization
  -> infer_types_for_proof()         # type inference; reject residual shape equations
  -> verify_prepared_annotations()
                                     # forward facts + backward demands + ordered alignment
                                     # qualify and return one canonical artifact
```

Logical write-disjointness and bounded suffix evaluation run independently
through `diagnostics/`; neither is a stage of this artifact-producing pipeline.

## Annotation propositions and the supported proof fragment

`ir.annotations` defines one proposition AST. Scalar comparisons and tensor
region equalities compose with `and`, `or`, `not`, `implies`, and integer
`forall`. Equality uses `==`; `=` is reserved for named bindings such as
`singleton(bi left=0 right=0)`. There is no `exists` or public `forall_region`
syntax. The latter names an internal guarded-region normal form only.

Relational lowering accepts positive conjunctions, scalar comparisons,
universally quantified scalar formulas, and scalar-guarded input-region
equalities. Relational output claims are conjunctions of region equalities.
Unsupported connective combinations, nested quantifiers, or regional
antecedents fail closed; representability in the AST is not proof support.
Execution sides and lexical binders survive every lowering stage.

Quantified input-region proofs first try substitutions suggested by shared SMT
term structure, then fall back to execution-iterator candidates. Matching is a
proof-search heuristic, not an equality rule: every substitution must separately
prove the complete guarded domain and demanded region correspondence on both
executions. A failed or unknown domain skips that candidate's region query.
The matcher does not recognize kernel names, particular operators, or tile/page
relationships, and never descends under binders. Explicit quantified contracts
and the generated Verus assumptions are unchanged by this search strategy.

Temporal annotations use `before(...)`, `after(...)`, and `dtype(...)` in the
same typed language. The exact-effect fragment supports guarded universal
copy/frame equalities with established index bounds and dtype identity.
Mutable-state reads, loop-carried values, and unproved arithmetic/conversions
are rejected. Region endpoints must be valid even for empty regions.

## Analysis review invariants

| Concern | Invariant to preserve | Regression examples under `tests/` |
| --- | --- | --- |
| Names and scope | Execution sides and lexical binders remain distinct; fresh internal names cannot invent equality. | `analysis/test_annotation_binder_scope.py`, `analysis/test_symbol_isolation.py`, `analysis/test_relational_names.py` |
| Positional facts | Facts are sufficient conditions; missing facts do not imply negation. Partial writes invalidate affected facts. | `analysis/test_positional.py`, `analysis/test_positional_corners.py` |
| Coordinates | Forward substitution and backward over-approximation preserve broadcast/reduction axes and offsets. | `analysis/test_coordinate_transport.py`, `analysis/test_regions_masked.py` |
| Loops | Arbitrary iterations cannot reuse initializer facts; back edges and live-out effects are covered; one-sided iterations require exact identity. | `analysis/test_loop_boundaries.py`, `analysis/test_identity_transition.py` |
| Provenance | Each use follows reaching definitions, including joins and zero-iteration paths, not later overwrites. | `analysis/test_discrete_definitions.py` |
| Floating values | Ordered opaque operations imply congruence, not real-number algebra. Numerical zero is not bitwise identity; dot-specific assumptions remain distinct. | `analysis/test_scalar_float_values.py`, `analysis/test_bitwise_zero_rules.py` |
| Qualification | Failures, missing evidence, vacuity, and unknown satisfiability cannot qualify. Bounded witness search never restricts the equality theorem. | `analysis/test_annotation_satisfiability.py`, `analysis/test_relational_verifier.py` |
| Consumers | Exact source/configuration, written outputs, theorem, and every extra premise survive import. | `frontend/test_unified_consumers.py`, `frontend/test_verus_contract.py`, `frontend/test_verus_exact_effect.py` |

## The "Verifiable Triton" subset

The translator accepts Triton kernels that follow these rules.

### Allowed constructs

- Tensor access via `tl.make_block_ptr` + `tl.load` / `tl.store`
- Unmasked scalar pointer loads (e.g. `tl.load(cu_seqlens + bi)`) for
  index lookups
- Standard operations: `tl.dot` (2-arg and 3-arg accumulate form), `tl.trans`,
  `tl.arange`, `tl.zeros`, `tl.full`
- Reductions: `tl.max`, `tl.sum` (positional constant propagation requires
  a statically positive reduced extent)
- Element-wise: `tl.exp2`, `tl.maximum`, `tl.where`, `tl.broadcast_to`
- Reshape: `tl.reshape` that only adds or removes size-1 dimensions
- Unsqueeze indexing: tuple indices containing `None` only when every other
  entry is a full `:` slice
- Cast: explicitly modeled floating data casts are equality-only identity
  abstractions; `.to(tl.int32)` is accepted only for scalar loads from
  `tensor(..., int32, ...)`, and unknown/dynamic cast targets are rejected
- `tl.constexpr` branching: specialized at translation time via `specialize={"FLAG": True}`
- Scalar data-dependent branching represented explicitly as IR `If`
- `for` loops with `range(start, stop)` iteration
- Python builtins: `min()`, `max()` (translated to `Min` / `Max` IR nodes)
- Augmented assignments: `+=`, `-=`, `*=`, `/=`, `and=`
- Literals: `float("-inf")`, `float("inf")`

### Kernel interface and proof-goal annotations

Kernels that recover tensor structure from pointer-offset expressions need
unique `@params(...)` and `@grid(...)` comment annotations before the
`@triton.jit` decorator. In
practice this is used for pointer-offset block-ptr bases and scalar pointer
loads that should be reconstructed as tensor accesses. The annotation tells the
translator the grid shape and tensor layouts. One or more named `@verif`
blocks state independent relational proof goals:

```python
# @params(
#   tensor(q, float, shape(B, H, L, D), strides(qb, qh, ql, qd)),
#   tensor(k, float, shape(B, Hkv, S, D), strides(kb, kh, ks, kd)),
#   tensor(o, float, shape(B, H, L, D), strides(ob, oh, ol, od)),
# )
# @grid(B, H, cdiv(L, BLOCK_M))
# @verif(batch_invariance,
#   same(H, D),
#   pre(left(q)[...] == right(q)[...]),
#   post(left(o)[...] == right(o)[...]),
# )
@triton.jit
def my_kernel(q, k, o, ...):
```

The interface annotations support:

- `@grid(dim0, dim1, ...)`: grid dimensions (including `cdiv`, `add`, `sub`, and `mul`)
- `tensor(name, float|int32, shape(...), strides(...))`: explicit logical tensor type, shape, and optional physical stride parameters
- `scalar(name, int|float|bool)`: typed scalar or logical launch parameter

Every `@verif(name, ...)` has its own optional `same(...)`, `pre(...)`, and
`singleton(...)` clauses and a required nonempty `post(...)`. When several
goals exist, callers must select one by name.

Scalar comparisons and tensor-region equalities share one proposition type.
Use `==` for equality, and ordinary `forall(i, implies(guard, relation))` for
guarded universal region equality; `=` is only for named singleton bindings.
Conjunctions can mix scalar and regional premises and group output equalities.
The parser also represents `or`, `not`, and nested `forall`, but proof lowering
rejects unsupported forms explicitly rather than discarding any proposition.
`exists` is not part of the language. See the
[supported proof fragment](#annotation-propositions-and-the-supported-proof-fragment).

Region coordinates support binary integer `min(a, b)` and `max(a, b)`, for
example `0:min(page_size, prefix_end - page_start)`. These are preserved through
the typed proof, canonical theorem, and generic Verus lowering. Selected-row
attention uses this to state logical-prefix equality without constraining
unused rows in the final page. Its separate batch-projection goal still assumes
whole-page equality.

Explicit `left(...)` and `right(...)` input references inside coordinates retain
their run identity, including references to the other run. Logical aliases are
internal to the paired proof environments, not additional theorem premises.
Cross-run dynamic iterator references are rejected; use a shared logical index
with explicit relational constraints instead.

Fully raw-pointer `tl.store(...)` kernels are still outside the supported
subset, even with interface annotations.

### Not supported

- `tl.atomic_*` operations
- Inline assembly (`tl.inline_asm`)
- Tensor/block-valued branch conditions (scalar conditions are modeled)
- `range` with a step argument
- Indirect/gather loads where the index tensor is data-dependent and
  non-contiguous within a tile
- Reading a local that is not definitely assigned on every reaching source
  path, or reassigning a scalar captured by an existing pointer value

## Engine-reachable kernels

| Kernel | File | Translates | Structural proof | Notes |
|--------|------|:----------:|:--------:|-------|
| Matrix multiply | `triton_kernels/matmul.py` | Yes | Yes | 2D grid, k-loop accumulation |
| Paged attention | `triton_kernels/fattn_paged.py` | Yes | Batch: `o` and private `lse`; selected-row: `o` | General dependency and ordered-loop rules establish causal-prefix equality under reported numerical assumptions. The IR-to-value argument remains trusted |
| Sliding-window paged attention | `triton_kernels/fattn_paged_swa.py` | Yes | Yes, for `o` and private `lse` | Uses a windowed numerical mask while conservatively retaining complete causal-prefix KV premises; it does not certify window-local eviction |
| RMSNorm / residual RMSNorm | `triton_kernels/rmsnorm*.py` | Yes | Yes | Both residual outputs are checked; the column grid covers arbitrary logical widths while runtime `next_power_of_2(N)` keeps deployed launches single-tile |
| Gemma row-local primitives | `triton_kernels/{add,gelu_tanh_mul,gemma_qk_norm,gemma_rmsnorm,scaled_embedding}.py` | Yes | Yes | Axis-projection contracts cover the profile-qualified rectangular launches used by vosti-verus |
| RoPE / SiLU-mul / QK norm | `triton_kernels/{rope,silu_mul,qk_norm}.py` | Yes | Yes | Pointwise or row-local certificates; SiLU-mul likewise has complete tiled-width coverage |
| Embedding | `triton_kernels/embedding.py` | Yes | Yes | Embedding IDs are explicit scalar-tensor dependencies |
| KV cache scatter | `triton_kernels/store_kv_cache.py` | Yes | Conditional | Bounded injective slots support the annotated copy/frame theorem. Physical execution and the verifier remain trusted |

## Configuration coverage

The regression suite checks every declared matmul and paged-attention candidate
against its structural obligations. The consuming framework's family-scoped
drivers additionally verify the exact kernel/configuration cases admitted by
each model profile, including both named attention goals. For focused proof and
raw theorem export, `scripts/transpile_verus_contract.py` accepts `--goal` and
explicit specialization constants; it uses the same qualified verifier for
attention and non-attention kernels.

Runtime autotuning is outside the modeled Triton subset. Candidate proof
coverage and offline deterministic selection remain separate launch-bridge
obligations.

## Minimal example

[`example.py`](example.py) validates and translates the annotated matmul
kernel, prints its IR, and runs the annotation-driven relational proof. Run from
`kernels/`:

```bash
uv run python example.py
```
