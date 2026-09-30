# Kernel contracts

The generator turns verified kernel annotations into Verus assumptions and
checked adapters for the engine. For annotation syntax, see
[`kernels/README.md`](../kernels/README.md); for the surrounding proof, see
[verification](verification.md).

## Pipeline and ownership

```text
annotated Triton + fixed specialization
    -> typed kernel IR and typed propositions
    -> qualified relational or exact-effect analysis
    -> source-bound ContractIR
    -> generated raw Verus assumptions
    -> checked representation adapters
    -> engine composition and determinism theorems
```

`ir.kernel_verifier.verify_kernel_goal` selects the analysis from typed
propositions, not kernel, architecture, parameter, or goal names. Preparation
is shared across goals for one source and specialization.

- `ir.relational_dataflow` combines forward positional facts, guarded backward
  dependencies, regional/scalar equality, and ordered operation alignment.
  Successful qualified proofs produce `VerifiedDataflowContract`.
- `ir.exact_effects` checks the supported temporal copy/frame fragment:
  reaching copied values, writer coverage/disjointness, dtype identity, and
  absence of writes in framed regions. It produces `VerifiedExactEffectContract`.
- `ir.verus_contract` and `ir.verus_exact_effect` lower these artifacts to raw
  Verus execution models, premises, conclusions, and imported certificates.
- `scripts/verification/verify_kernel_contracts.py` qualifies the union of model-family
  inventories once. The same qualified cases feed deployment receipts and
  shared attention, rectangular, and mutation interface generation.

Each artifact records its source, specialization, goal, outputs, and analysis
evidence. A relational proof is accepted only if it succeeds and its annotation
premises are satisfiable, including quantified premises. This check does not
establish that the premises describe a physically realizable GPU execution or
discharge numerical assumptions added by the analysis. The analyzer remains trusted.

The [annotation language and supported proof fragment](../kernels/README.md#annotation-propositions-and-the-supported-proof-fragment)
and [analysis review invariants](../kernels/README.md#analysis-review-invariants)
are documented with the verifier.

## Raw Verus assumptions

Goals for the same typed source execution share one opaque execution function
and launch-state representation. Each goal retains its own complete premises.
The written-output inventory comes from IR, not only from the selected goal:
an attention goal may relate `o` without relating the also-written `lse`.
Temporal exports retain before/after states and physical dtype premises.

Analyzer assumptions and external obligations become separate uninterpreted
precondition predicates. Their exact labels and kinds remain in the manifest;
diagnostic text is neither parsed as a theorem nor silently discharged.
Consumers without a representation for a condition must reject it. In
particular, runtime finiteness may remain an explicit assumption.

Unknown schema fields, mixed source executions, duplicate goals, colliding
symbols, and unbound execution iterators are rejected. Transparent helpers
preserve quantified premises and expose usable Verus triggers.

## Checked engine adapters

Identifier bindings are integration declarations, not semantic evidence.
`scripts/verification/engine_kernel_bindings.py` declares attention port/coordinate roles
and the default labels for engine-required properties. A scope can override
`proof_goals` with distinct goal names for exactly those roles. Adding a model
using existing interfaces changes its inventory, not the verifier rules.

- Rectangular adapters infer row/shared roles and semantic dimensions from
  qualified tensor types. The axis normalizer accepts only complete supported
  projections and preserves the exact raw domain.
- Paged-attention adapters infer geometry from explicitly bound tensors. Shared
  layout lemmas establish reshapes, page-table validity, and logical-to-physical
  region correspondence. The compiled engine page size is fixed, not a new
  deployment option or a proof-visible kernel tile.
- Scatter adapters infer source, destination, indices, and row width from the
  qualified typed effect. Checked copy/frame lemmas feed the paged-cache update
  proof; batch invariance alone cannot justify mutation semantics.

Verus checks the adapters' bodies. Generating a file does not verify it.
A new representation may need a new adapter even if the annotation language
already expresses its contract. Unknown layouts are rejected.

Attention batch projection requires equality of complete referenced pages;
selected-row projection requires equality only through the causal endpoint,
including a clipped final page. Both use the same raw execution. Canonical
row packing and whole-launch-to-row mapping are checked under the exact layout
and numerical premises. SWA still conservatively depends on the whole causal
prefix; window-only KV eviction is not proved.

Static alternatives are separately qualified. The interface assembler checks
that their complete logical theories and adapters agree, while retaining
distinct execution/proof identities. The engine interface denotes one fixed
deployed implementation; this comparison does not prove numerical equality
between configurations. Tile selection remains outside the engine theorem.

## Residual trust

The Python analyzer, translation, and raw exporter remain trusted. Tensor
floating-point execution is not modeled bit-for-bit in Z3: congruent ordered
operations plus equal dependencies support a trusted semantic argument.
Physical tensor/object correspondence, non-aliasing, allocation, source-to-launch
correspondence, numeric premises, and compiler/GPU behavior remain external
obligations.

Deployment probes provide bounded empirical evidence, not proofs of these
obligations. Neither the engine theorem nor this bridge proves model numerical
correctness, equivalence to a reference implementation, or equivalence between
fused and unfused model formulations.

## Adapter review

Engine-side tests additionally cover checked adapters, consistently renamed
bindings, altered geometry, shortened regions, and omitted numeric premises.
Negative Verus callers must fail for the intended proof obligation; a timeout
does not establish rejection. Tests and engineering review are not a
machine-checked soundness proof of the analyzer.

## Validation

For a bridge change, run focused producer/exporter/adapter regressions, then
`make verify-kernels` for deployed cases and generated freshness, `make verify-engine`
for whole-crate Verus and compiler/source trust agreement, and the independent
`uv run --locked python scripts/project.py architecture`, `python3 scripts/audit/claim_ledger.py`, `python3 scripts/audit/tcb.py --check`, and
`python3 scripts/effort/account.py --check` gates. Regenerate interfaces with
`make generate`; regenerate TCB accounting with `python3 scripts/audit/tcb.py`.
Effort-accounting procedures are documented with their scripts. CPU Python
tests need loopback socket access for the local HTTP fixture; a denied socket
is an environment failure, not kernel-proof evidence. CUDA tests are separate.
