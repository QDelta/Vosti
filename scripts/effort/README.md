# Source-effort accounting

The source-effort tools are:

- `scripts/effort/account.py`: source selection, Python/Triton classification,
  aggregation and LaTeX generation.
- `scripts/effort/specification.py`: theorem-rooted subdivision of specification
  lines, with compiler-resolved dependency and assumption accounting.
- `scripts/effort/purposes.py`: reviewed purpose ownership for abstract spec,
  concrete instantiation, runtime contracts, and auxiliary proof effort.
- `scripts/effort/dependencies.py`: typed Verus dependency extraction.
- `scripts/effort/rust/`: Rust/Verus syntax classifier and pinned Cargo dependencies.
- `python/tests/test_effort.py`: Python regression tests; Rust tests are
  embedded in the classifier.

Generated files live under `target/effort/`: `effort_dependencies.json` contains
the source-pinned typed graph, `effort.json` contains the detailed per-file
classification, and `effort.tex` contains the publication table. Copy snapshots
into a paper or run archive when needed; they are not tracked source files.

The current table uses three specification columns and includes auxiliary
specification in the Proof column. See "Purpose-based presentation" below.

From the repository root:

```sh
cargo test --offline --locked --manifest-path scripts/effort/rust/Cargo.toml --target-dir target/effort-accounting
uv run --locked python -m pytest -q python/tests/test_effort.py # included in make test
python3 scripts/effort/dependencies.py # refresh typed graph after Rust/compiler-input changes
cargo build --offline --locked --manifest-path scripts/effort/rust/Cargo.toml --target-dir target/effort-accounting
python3 scripts/effort/account.py # regenerate JSON and LaTeX
python3 scripts/effort/account.py --check      # recount sources and check saved outputs
```

`make test` refreshes the typed graph before the Python tests. For standalone
specification-closure tests, run `dependencies.py` first. Missing or stale graphs
fail with a regeneration instruction.

Builds use locked, cached Cargo dependencies and place artifacts under
`target/effort-accounting`. On a machine without cached dependencies, first run:

```sh
cargo build --locked --manifest-path scripts/effort/rust/Cargo.toml --target-dir target/effort-accounting
```

The Python driver uses only the standard library. Its default source root is
this checkout, independent of the current working directory; `--root` selects
another framework checkout. Source selection uses tracked paths; staged and
working-tree edits are counted. The tool checks that the selected files remain
unchanged during counting and that scoped kernel/support bytes match their digests.

The checker compares source hashes, classifications, line ranges, totals,
scope, and kernel digests. Committing already-counted edits or changing
unrelated documentation does not invalidate the count.

## Counting rules and scope

The JSON records syntax-based physical-line counts and
four-component totals. The publication table uses the
whole-file responsibility regrouping described below.
Each nonblank source line belongs to exactly one category:

- Implementation: executable source, including verifier/tooling implementation.
- Specification: contracts, spec functions/constants, explicit ghost/tracked
  state, assumed proof declarations, and structured kernel annotation comments.
- Proof: proof functions/blocks, Verus assertions and reveal/hide expressions,
  and explicit ghost/tracked local bindings.

Specification takes precedence over proof on mixed lines. Documentation and
ordinary comments are excluded. Rust is parsed with `verus_syn`, including
`verus!` contents; Python uses AST/token locations. No general macro expansion
or semantic erasure is performed by the line classifier. The main/auxiliary
subdivision separately uses compiler-resolved declaration identities;
recounting is not proof verification.
The specification-plus-proof sum measures verification source effort,
not verified executable coverage or TCB size. The publication table splits
main/assumed specification, auxiliary specification, and proof.

`source_inventory` is the exact selection policy: tracked engine/model/proof
sources, Rust boundary and selected serving helpers, Python runtime, scoped
Triton sources, verifier infrastructure and certificate/deployment integration.
All supported model families are included; shared files count once. Tests,
benchmarks, ordinary examples, metadata, generated certificates and third-party
dependencies are excluded by selection and limited syntax filters.
Backend qualification probes are included. Entire selected files
count, including optional/CPU fallback paths. The static import walk is not a
complete runtime reachability analysis.

The `python3 scripts/audit/tcb.py` / `python3 scripts/audit/tcb.py --check` commands maintain
`audit/tcb.md` using a different trust-inventory methodology. They are not
interchangeable with the effort counter.

## Purpose-based presentation

The publication table presents abstract specification, concrete
instantiation, and runtime contracts separately. Auxiliary specification
is included in Proof, not in a specification column.

The primary root is
`proof::lemma_serving_deterministic`, which proves
`deterministic(system(model, plan))` for the explicit whole-execution spec.
Initialization, one-step logits, and
the internal certified-trace corollaries are audited claims but
do not independently seed publication specification accounting.

The dependency closure follows declarations, not proof or executable bodies.
It determines which definitions contribute to the specification, while the
purpose rules below assign their categories. `purpose_counts` and `purpose_lines`
partition specification lines into four disjoint sets, alongside the
syntax-based `specification_parts` counts.
JSON schema is `paper.effort.v3`. The generated inventory records each selected
contributing declaration's role and selection rule.

### Ownership rules

- Abstract spec: only `src/spec.rs`, the generic trace,
  system, legality, observation, and consistency definitions. Compiler-closure
  tests limit its dependency closure to this module and the existing opaque
  scalar and sampler-state leaf types. Only states are parameters; input and output
  vocabulary is explicit. Model/deployment scope belongs to the concrete
  instantiation.
- Concrete instantiation: the specialized satisfaction theorem, engine
  interpretation, configurations/weights and model composition, executable
  states and steps, state abstraction and cache-fidelity relations, graph
  padding, and defined tensor/operation/permission/shape/admission vocabulary.
  Concrete observation and trace definitions are not relabeled abstract.
- Runtime contracts: imported/assumed Rust runtime declarations and kernel
  annotations in the Triton component. Annotations are not relabeled trusted
  Rust assumptions. Rust runtime-contract line sets must exactly match the
  separately retained trusted-assumption line sets.
- Auxiliary proof: specification lines outside the retained closure, plus
  selected queue/refcount/provenance-maintenance predicates and helper
  conditions establishing the concrete invariant. These contribute to Proof
  alongside original proof syntax.

`purposes.py` assigns default categories by file and overrides them for named
declarations. The checker rejects new files without a category, stale overrides,
conflicting line assignments, and unreviewed trusted declarations. Shared
definitions count once. Classification does not stop dependency traversal.

Cache meaning and state correspondence belong to concrete instantiation;
invariants maintaining queue/refcount representations belong to auxiliary
proof. This distinction is reviewed per declaration rather than moving all
cache-related specifications together.
