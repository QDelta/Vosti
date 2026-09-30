# Audit manifests

This directory records the claimed properties, their supporting declarations,
and model-family interfaces. Audits check these records against the source;
Verus and the kernel verifier check the proofs separately.

| Path | Ownership | Purpose | Primary check |
| --- | --- | --- | --- |
| [claims.md](claims.md) | Generated | Maps reviewed claims to supporting entrypoints, imports, and trusted declarations. | `python3 scripts/audit/claim_ledger.py` |
| [tcb.md](tcb.md) | Generated | Inventories checked code, external boundaries, and verification tooling. | `python3 scripts/audit/tcb.py --check` |
| `claim_surface.json` | Manually reviewed | Enumerates the intent-trusted properties, supporting checked entrypoints, deployment assumptions, certificate-consumer roles, source-pinned trusted declarations, runtime boundaries, and external types. | `python3 scripts/audit/claim_ledger.py` |
| `model_architecture_ownership.json` | Manually reviewed | Lists supported families, proof compositions, qualification inventories, and shared kernel interfaces with their generators and consumers. | `uv run --locked python scripts/project.py architecture` |
| `attention_kernel_interfaces.json` and `*_kernel_interface.json` | Generated | Shared operator receipts bind qualified kernel cases, annotation-derived propositions, generated Verus interfaces, and representation obligations. Families contribute cases, not separate interface copies. | `make verify-kernels` and `python3 scripts/audit/claim_ledger.py` |

The architecture check rejects missing adapter operations, shared-signature
disagreement, forbidden Rust imports/dispatch, and incomplete certificate
inventories. It does not prescribe private helper layouts or model coverage in
tests and benchmarks. Compilation and proof gates check the actual contracts;
the source audit alone cannot establish behavior.

The claim ledger distinguishes two certificate roles:

- `claim_supporting` records at least one explicitly enumerated direct call
  from a checked consumer module.
- `qualification_only` has no such checked consumer and is not presented as a
  theorem-proof dependency.

This classification records direct calls. It is not a
formally verified transitive call graph from every certificate to a top-level
property.

## Updating the records

Edit the two reviewed JSON manifests only as an intentional claim or ownership
change. Do not hand-edit generated interface receipts or `audit/claims.md`. After a
kernel proof, specialization scope, or claim-surface change, run:

```bash
make generate
uv run --locked python scripts/project.py architecture
python3 scripts/audit/claim_ledger.py
```

`make generate` qualifies the complete shared kernel inventory, writes the
generated contracts and interface receipts, and renders `audit/claims.md`. Review
the generator inputs and every generated diff together.

`python3 scripts/audit/claim_ledger.py` validates source, scope, digest, import, trusted-interface,
checked-consumer, and declaration linkage. It does not rerun Verus or the
kernel verifier. Use `make verify` for both provers or `make check` for the
complete CPU gate. Regenerate the separate trust inventory in
`audit/tcb.md` with `python3 scripts/audit/tcb.py`.

See [claims.md](claims.md), [tcb.md](tcb.md), and
[deployment](../docs/deployment.md) for the remaining trusted components.
