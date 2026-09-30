# Repository guidance

Working rules for contributors and coding agents. Consult the linked guides
for architecture, proof, and setup details.

## Read before changing the system

- `README.md` lists supported models and verification entry points.
- `docs/architecture.md` defines ownership, closed model-family dispatch, and
  two-phase serving.
- `docs/verification.md` explains the theorem, proof structure, trust boundary,
  and open gaps. Update it when those claims change.
- `docs/models.md` defines the extension contract for a
  new family.
- `docs/deployment.md` defines specialization, kernel
  verification, backend probes, sealing, and checked startup assembly.
- `audit/tcb.md` and `audit/claims.md` are generated audit inventories. Read them
  as maps, not as additional proofs.
- `docs/setup.md` lists setup and validation commands.

Use Git history for superseded designs and milestones. Do not retain
development diaries, machine-specific sibling-checkout paths, or personal
agent notes in current documentation.

## Architectural invariants

- Shared model proofs, execution, and top-level properties remain architecture-neutral.
  A family supplies the common closed composition, cache, relocation,
  readiness, weight, and deployment contracts.
- Treat Qwen3, Llama3, and text-only Gemma3/Gemma4 symmetrically at shared boundaries.
  Symmetry means the same obligations and lifecycle, not identical file sizes
  or identical native layer compositions.
- Keep reusable dense layers and their properties family-neutral. A new dense
  family composed from existing layers should require a small family adapter,
  profile, certificates, and tests, without branches throughout the Engine or
  top-level proof.
- Put every `external_body`, uninterpreted declaration, opaque external type,
  and imported raw certificate under `src/boundary/`. Keep checked adapters
  there only when they constrain a colocated declaration.
- Put shared identities in `src/types.rs` and the common configuration schema
  in `src/model_config.rs`. Verified execution belongs in `src/exec/`, the
  public determinism definition in `src/spec.rs`, its satisfaction theorem in
  `src/proof.rs`, and semantic models and supporting proofs in `src/proof/`.
- Preserve tracked permission discipline: mutation uses
  `Tracked<&mut TensorPerm>`, reads use `Tracked<&TensorPerm>`, and allocators
  return owned `Tracked<TensorPerm>`. Do not make `TensorPerm` `Clone` or
  `Copy`, or add an ordinary public constructor.

Serving is two-phase. Offline preparation selects and verifies the complete
kernel plan, qualifies the environment, and seals a deployment identity.
Checked startup assembly constructs an explicit immutable runtime capability.
Online execution performs exact-key lookup only: do not add process-global
deployment state, runtime kernel selection, autotuning, or production
fallbacks.

Sliding-window attention currently retains the complete causal KV
prefix. Do not claim window-local dependency or add KV eviction without the
corresponding proof.

## Editing rules

- Prefer deleting obsolete compatibility paths and extracting neutral shared
  layers over adding mirrored wrappers solely to make directory trees look
  alike.
- Keep family-native differences inside the family composition or its
  specialization evidence. The architecture manifest records supported
  families, proof compositions, and kernel interfaces; private helpers and
  experiment files do not need individual ownership entries.
- Do not hand-edit generated kernel contracts, import records,
  `audit/claims.md`, or `audit/tcb.md`. Use their generators and review both the
  generator and generated diff.
- Preserve unrelated changes in a dirty worktree. Inspect staged and unstaged
  diffs before formatting, generating, reverting, or committing.
- Follow local Rust/Verus formatting. Whole-tree rustfmt is not a passing gate;
  bulk formatting can invalidate source-attested artifacts. Regenerate and
  verify affected artifacts when their source bytes change.
- Do not present tests, linkage checks, backend probes, or bounded falsifiers
  as formal proof. State which gate actually ran and which trust assumptions
  remain.

## Validation

Use the narrowest relevant checks during development, then the complete gate
when a change affects the stated claim. See [setup](docs/setup.md#cpu-side-checks)
for commands and regeneration order. Review generator inputs and outputs
together; successful generation does not replace either verifier.

GPU checks are separate empirical gates and require a checkpoint and qualified
deployment bundle. Select an idle GPU and do not disturb another user's job.
Report unrun checks as unrun.
