#!/usr/bin/env python3
"""Validate and render the closed serving-determinism claim ledger.

This checker is intentionally independent of the kernel proof driver.  The
driver proves kernels and emits canonical import records; this module checks
those records against the framework catalog, generated Verus axioms, explicit
checked consumers, top-level theorem spans, and the complete external runtime
function/type inventory.  It improves fail-closed traceability, but does not
turn source attestation or an ``external_body`` import into a formal proof.
"""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit.tcb import (
    rust_module_name,
    rust_source_files,
    scan_external_type_records,
    scan_external_types,
    scan_rust,
    scan_rust_trust,
    scan_rust_trust_records,
)
from scripts.verification.kernel_interface_registry import load_registry, rectangular_operators_from_source
from scripts.audit.kernel_interface_audit import (validate_attention_interface, attention_inventory_from_source,
                                    validate_rectangular_interface, rectangular_inventory_from_source,
                                    validate_mutation_interface)


SURFACE_PATH = ROOT / "audit" / "claim_surface.json"
OWNERSHIP_PATH = ROOT / "audit" / "model_architecture_ownership.json"
RUNTIME_BRIDGE_SCOPE_PATH = (
    ROOT / "python" / "vosti_kernels" / "runtime_bridge_scope.json"
)
BOUNDARY_ROOT = (ROOT / "src" / "boundary").resolve()
DOC_PATH = ROOT / "audit" / "claims.md"
KERNEL_SOURCE_ROOT = ROOT / "kernels" / "triton_kernels"

SURFACE_FIELDS = {
    "schema_version",
    "deployment_assumptions",
    "top_level_claims",
    "supporting_checked_entrypoints",
    "certificate_consumers",
    "kernel_interface_receipt_sha256",
    "trusted_proof_boundaries",
    "uninterpreted_specifications",
    "trusted_declaration_source_sha256",
    "excluded_nonproduction_trusted_declarations",
    "trusted_runtime_boundaries",
    "trusted_external_types",
}
CLAIM_FIELDS = {
    "name",
    "item",
    "kind",
    "source_span",
    "requires",
    "ensures",
    "limitations",
    "formalization_role",
    "verification_status",
}
SPAN_FIELDS = {"path", "name", "digest"}
CONSUMER_FIELDS = {"interface", "proof_name", "role", "consumers"}
RUNTIME_FIELDS = {"item", "class", "scope", "attestation", "source_span"}
TYPE_FIELDS = {"item", "class", "scope"}
DECLARATION_FIELDS = {"item", "class", "scope"}
NONPRODUCTION_FIELDS = {"item", "category", "scope", "source_sha256"}
RUNTIME_CLASSES = {
    "opaque_host_state",
    "relational_kernel_launch",
    "conditional_kernel_launch",
    "trusted_exact_kernel_effect",
    "trusted_exact_adapter",
    "exact_materializer",
    "permission_binder",
    "permission_allocator",
    "test_tensor_adapter",
}
TYPE_CLASSES = {
    "opaque_host_state",
    "abstract_numeric_value",
    "runtime_identity",
    "python_tensor_handle",
    "ghost_permission",
    "ghost_permission_bundle",
}
PROOF_CLASSES = {
    "kernel_certificate",
    "runtime_shape_axiom",
    "permission_protocol_axiom",
}
UNINTERPRETED_CLASSES = {
    "runtime_identity",
    "tensor_representation",
    "abstract_kernel_semantics",
    "deployment_assumption",
    "sampling_semantics",
    "permission_bundle_projection",
    "kernel_numeric_premise",
    "kernel_physical_premise",
}
SCOPES = {"claim_reachable", "deployment_setup", "test_only"}
NONPRODUCTION_SCOPES = {"example", "test"}
NONPRODUCTION_CATEGORIES = {
    "exec-trusted",
    "proof-trusted",
    "uninterp",
    "external-type",
}
HEX64 = re.compile(r"^[0-9a-f]{64}$")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(f"invalid claim ledger: {message}")


def digest(value: str, description: str) -> str:
    require(isinstance(value, str) and HEX64.fullmatch(value) is not None, description)
    return value


def unique_strings(value, description: str, *, allow_empty: bool = False) -> list[str]:
    require(
        isinstance(value, list)
        and (allow_empty or bool(value))
        and all(isinstance(item, str) and item for item in value)
        and len(set(value)) == len(value),
        description,
    )
    return value


def canonical_sha256(value) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()




def marked_body(source: str, name: str, marker: str) -> str:
    lines = source.splitlines(keepends=True)
    begin_marker = f"{marker}-begin {name}"
    end_marker = f"{marker}-end {name}"
    begins = [i for i, line in enumerate(lines) if line.strip().endswith(begin_marker)]
    ends = [i for i, line in enumerate(lines) if line.strip().endswith(end_marker)]
    require(
        len(begins) == len(ends) == 1 and begins[0] < ends[0],
        f"source span {name!r} must have exactly one {marker} marker pair",
    )
    body = "".join(lines[begins[0] + 1 : ends[0]])
    require(bool(body.strip()), f"source span {name!r} is empty")
    return body


def read_json(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(data, dict), f"{path.relative_to(ROOT)} is not an object")
    return data


def repository_path(relative: str, description: str) -> Path:
    require(isinstance(relative, str) and relative, f"{description} has no path")
    path = (ROOT / relative).resolve()
    require(
        path == ROOT.resolve() or ROOT.resolve() in path.parents,
        f"{description} escapes repository",
    )
    require(path.is_file(), f"{description} is missing: {relative}")
    return path


def certificate_import_sets() -> dict[str, dict]:
    """Shared interfaces are keyed by kernel role, never by owning family."""
    _, interfaces = load_registry(ROOT)
    return {name: {
        **entry,
        "imports_path": repository_path(entry["manifest"], f"{name} interface receipt"),
        "generated_path": repository_path(entry["generated"], f"{name} generated interface"),
    } for name, entry in interfaces.items()}


def runtime_attestation_scope(runtime_bridge_scope: dict) -> dict:
    """Union common attestation spans; no model family governs primitives."""
    qualification, _ = load_registry(ROOT)
    contracts = {}
    for family, entry in qualification.items():
        scope = read_json(repository_path(entry["scope"], f"{family} kernel scope"))
        for contract in scope["kernel_contracts"]:
            spans = contract.get("bridge", {}).get("source_spans", [])
            if not spans:
                continue
            wrapper = contract["wrapper"]
            normalized = {"wrapper": wrapper, "bridge": {"source_spans": spans}}
            require(wrapper not in contracts or contracts[wrapper] == normalized,
                    f"shared runtime attestation differs across families: {wrapper}")
            contracts[wrapper] = normalized
    return {"kernel_contracts": list(contracts.values()),
            "trusted_runtime_bridges": runtime_bridge_scope["bridges"]}


def span_body(span: dict) -> str:
    require(isinstance(span, dict) and set(span) == SPAN_FIELDS, "malformed source span")
    path = (ROOT / span["path"]).resolve()
    require(ROOT.resolve() in path.parents, "source span escapes repository")
    body = marked_body(path.read_text(encoding="utf-8"), span["name"], "@kernel-bridge")
    require(
        hashlib.sha256(body.encode("utf-8")).hexdigest() == digest(span["digest"], "bad span digest"),
        f"source span digest mismatch for {span['name']!r}",
    )
    return body


def validate_consumers(surface: dict, imports: dict[str, dict], certificate_sets: dict) -> dict:
    entries = surface["certificate_consumers"]
    require(isinstance(entries, list), "malformed certificate consumers")
    declared = {}
    for entry in entries:
        require(isinstance(entry, dict) and set(entry) == CONSUMER_FIELDS,
                "malformed certificate consumer")
        proof = entry["proof_name"]
        require(proof in imports and proof not in declared, "duplicate or unrecorded raw import")
        require(entry["interface"] in certificate_sets
                and entry["interface"] == imports[proof]["interface"],
                "raw import interface mismatch")
        require(entry["role"] == "claim_supporting", "unconsumed raw imports are not engine certificates")
        consumers = unique_strings(entry["consumers"], "invalid checked consumer set")
        require(sorted(consumers) == imports[proof]["consumers"],
                f"consumer mismatch for {proof!r}")
        declared[proof] = entry
    require(set(declared) == set(imports), "consumer/import proof coverage differs")
    return declared


def validate_claim_entries(
    claims: object,
    *,
    description: str,
    expected_role: str | None,
) -> list[dict]:
    require(isinstance(claims, list) and bool(claims), f"no {description}")
    names, items = set(), set()
    for claim in claims:
        require(
            isinstance(claim, dict) and set(claim) == CLAIM_FIELDS,
            f"malformed {description} entry",
        )
        require(
            claim["name"] not in names and claim["item"] not in items,
            f"duplicate {description} entry",
        )
        require(claim["kind"] in {"exec", "proof"}, f"invalid {description} kind")
        if expected_role is None:
            require(
                claim["formalization_role"].startswith("intent_trusted_"),
                f"invalid top-level formalization role for {claim['item']!r}",
            )
        else:
            require(
                claim["formalization_role"] == expected_role,
                f"invalid supporting role for {claim['item']!r}",
            )
        require(
            claim["verification_status"] == "verus_checked",
            f"invalid verification status for {claim['item']!r}",
        )
        require(claim["source_span"]["name"] == claim["item"], "claim item/span mismatch")
        body = span_body(claim["source_span"])
        for field in ("requires", "ensures", "limitations"):
            unique_strings(claim[field], f"malformed claim {field}")
        for token in claim["requires"] + claim["ensures"]:
            require(token in body, f"claim token {token!r} absent from {claim['item']!r}")
        names.add(claim["name"])
        items.add(claim["item"])
    return claims


def validate_top_claims(surface: dict) -> list[dict]:
    return validate_claim_entries(
        surface["top_level_claims"],
        description="top-level property",
        expected_role=None,
    )


def validate_supporting_entrypoints(surface: dict) -> list[dict]:
    return validate_claim_entries(
        surface["supporting_checked_entrypoints"],
        description="supporting checked entrypoint",
        expected_role="supporting_checked_entrypoint",
    )


def referenced_span(attestation: str, span_name: str, contracts: dict, bridges: dict) -> bool:
    if attestation.startswith("engine_kernel_contract:"):
        owner = contracts.get(attestation.split(":", 1)[1])
        spans = owner.get("bridge", {}).get("source_spans", []) if owner else []
    elif attestation.startswith("trusted_runtime_bridge:"):
        owner = bridges.get(attestation.split(":", 1)[1])
        spans = owner.get("source_spans", []) if owner else []
    else:
        return False
    matches = [span for span in spans if span["name"] == span_name]
    require(len(matches) == 1, f"attestation span {span_name!r} is not unique")
    span_body(matches[0])
    return True


def validate_runtime_inventory(surface: dict, scope: dict) -> tuple[list[dict], list[dict]]:
    entries = surface["trusted_runtime_boundaries"]
    require(isinstance(entries, list), "malformed runtime inventory")
    declared = {}
    contracts = {contract["wrapper"]: contract for contract in scope["kernel_contracts"]}
    bridges = {bridge["name"]: bridge for bridge in scope.get("trusted_runtime_bridges", [])}
    used_contracts, used_bridges = set(), set()
    for entry in entries:
        require(isinstance(entry, dict) and set(entry) == RUNTIME_FIELDS, "malformed runtime boundary")
        item = entry["item"]
        require(item not in declared, f"duplicate runtime boundary {item!r}")
        require(entry["class"] in RUNTIME_CLASSES, f"unknown runtime class for {item!r}")
        require(entry["scope"] in SCOPES, f"unknown runtime scope for {item!r}")
        if entry["attestation"] == "none":
            require(entry["source_span"] is None, f"unattested boundary {item!r} names a span")
            require(entry["scope"] != "claim_reachable", f"claim-reachable boundary {item!r} is unattested")
        else:
            require(isinstance(entry["source_span"], str) and entry["source_span"], f"attested boundary {item!r} has no span")
            require(referenced_span(entry["attestation"], entry["source_span"], contracts, bridges), f"attestation does not cover {item!r}")
            prefix, owner = entry["attestation"].split(":", 1)
            (used_contracts if prefix == "engine_kernel_contract" else used_bridges).add(owner)
        declared[item] = entry
    actual = {
        record["identity"]
        for path in rust_source_files(ROOT / "src")
        for record in scan_rust_trust_records(path)
        if record["category"] == "exec-trusted"
    }
    require(set(declared) == actual, "trusted runtime function inventory is not closed")
    require(used_contracts == set(contracts), "not every engine kernel contract covers a runtime boundary")
    require(used_bridges == set(bridges), "not every trusted runtime bridge covers a runtime boundary")

    type_entries = surface["trusted_external_types"]
    require(isinstance(type_entries, list), "malformed external type inventory")
    declared_types = {}
    for entry in type_entries:
        require(isinstance(entry, dict) and set(entry) == TYPE_FIELDS, "malformed external type")
        require(entry["class"] in TYPE_CLASSES and entry["scope"] in SCOPES, "invalid external type class/scope")
        require(entry["item"] not in declared_types, f"duplicate external type {entry['item']!r}")
        declared_types[entry["item"]] = entry
    actual_types = {
        record["identity"]
        for path in rust_source_files(ROOT / "src")
        for record in scan_external_type_records(path)
    }
    require(set(declared_types) == actual_types, "trusted external type inventory is not closed")
    return entries, type_entries


def _validate_declaration_entries(
    entries,
    *,
    description: str,
    classes: set[str],
    actual: set[str],
) -> list[dict]:
    require(isinstance(entries, list), f"malformed {description} inventory")
    declared = {}
    for entry in entries:
        require(
            isinstance(entry, dict) and set(entry) == DECLARATION_FIELDS,
            f"malformed {description} declaration",
        )
        item = entry["item"]
        require(isinstance(item, str) and bool(item), f"invalid {description} item")
        require(item not in declared, f"duplicate {description} declaration {item!r}")
        require(entry["class"] in classes, f"unknown {description} class for {item!r}")
        require(entry["scope"] in SCOPES, f"unknown {description} scope for {item!r}")
        declared[item] = entry
    require(set(declared) == actual, f"{description} inventory is not closed")
    return entries


def validate_trusted_declaration_inventory(
    surface: dict,
) -> tuple[list[dict], list[dict], dict[str, str]]:
    actual_proofs = set()
    actual_uninterpreted = set()
    actual_digests = {}
    for path in rust_source_files(ROOT / "src"):
        trust_records = scan_rust_trust_records(path)
        type_records = scan_external_type_records(path)
        require(
            not (trust_records or type_records)
            or BOUNDARY_ROOT in path.resolve().parents,
            f"trusted declaration outside src/boundary: {path.relative_to(ROOT)}",
        )
        for identity, category, _, _ in scan_rust_trust(path):
            if category == "proof-trusted":
                actual_proofs.add(identity)
            elif category == "uninterp":
                actual_uninterpreted.add(identity)
        for record in trust_records:
            identity = record["identity"]
            require(
                identity not in actual_digests,
                f"duplicate trusted declaration identity {identity!r}",
            )
            actual_digests[identity] = record["source_sha256"]
        for record in type_records:
            identity = record["identity"]
            require(
                identity not in actual_digests,
                f"duplicate trusted declaration identity {identity!r}",
            )
            actual_digests[identity] = record["source_sha256"]
    proofs = _validate_declaration_entries(
        surface["trusted_proof_boundaries"],
        description="trusted proof",
        classes=PROOF_CLASSES,
        actual=actual_proofs,
    )
    uninterpreted = _validate_declaration_entries(
        surface["uninterpreted_specifications"],
        description="uninterpreted specification",
        classes=UNINTERPRETED_CLASSES,
        actual=actual_uninterpreted,
    )
    declared_digests = surface["trusted_declaration_source_sha256"]
    require(
        isinstance(declared_digests, dict),
        "malformed trusted declaration source-digest inventory",
    )
    require(
        set(declared_digests) == set(actual_digests),
        "trusted declaration source-digest inventory is not closed",
    )
    for identity, actual_digest in actual_digests.items():
        require(
            digest(
                declared_digests[identity],
                f"invalid source digest for trusted declaration {identity!r}",
            )
            == actual_digest,
            f"trusted declaration source digest mismatch for {identity!r}",
        )
    return proofs, uninterpreted, actual_digests


def validate_nonproduction_trusted_inventory(surface: dict) -> list[dict]:
    actual = {}
    for scope, directory in (
        ("example", ROOT / "examples"),
        ("test", ROOT / "tests"),
    ):
        for path in rust_source_files(directory):
            for record in scan_rust_trust_records(path):
                item = f"{scope}::{record['identity']}"
                require(item not in actual, f"duplicate non-production trust item {item!r}")
                actual[item] = {
                    "category": record["category"],
                    "scope": scope,
                    "source_sha256": record["source_sha256"],
                }
            for record in scan_external_type_records(path):
                item = f"{scope}::{record['identity']}"
                require(item not in actual, f"duplicate non-production trust item {item!r}")
                actual[item] = {
                    "category": "external-type",
                    "scope": scope,
                    "source_sha256": record["source_sha256"],
                }

    entries = surface["excluded_nonproduction_trusted_declarations"]
    require(isinstance(entries, list), "malformed non-production trust inventory")
    declared = {}
    for entry in entries:
        require(
            isinstance(entry, dict) and set(entry) == NONPRODUCTION_FIELDS,
            "malformed non-production trusted declaration",
        )
        item = entry["item"]
        require(
            isinstance(item, str) and item not in declared,
            f"duplicate non-production trusted declaration {item!r}",
        )
        require(
            entry["category"] in NONPRODUCTION_CATEGORIES,
            f"unknown non-production trust category for {item!r}",
        )
        require(
            entry["scope"] in NONPRODUCTION_SCOPES,
            f"unknown non-production trust scope for {item!r}",
        )
        digest(entry["source_sha256"], f"invalid non-production source digest for {item!r}")
        declared[item] = entry
    require(
        set(declared) == set(actual),
        "excluded non-production trusted declaration inventory is not closed",
    )
    for item, live in actual.items():
        require(
            all(declared[item][field] == value for field, value in live.items()),
            f"non-production trusted declaration drift for {item!r}",
        )
    return entries


def validate_assumptions(surface: dict, claims: list[dict]) -> list[dict]:
    assumptions = surface["deployment_assumptions"]
    require(isinstance(assumptions, list) and bool(assumptions), "no deployment assumptions")
    names = set()
    for assumption in assumptions:
        require(set(assumption) == {"name", "predicate", "status", "scope"}, "malformed deployment assumption")
        require(assumption["name"] not in names and assumption["status"] in {"assumed", "runtime_guarded"}, "invalid deployment assumption")
        predicate_name = assumption["predicate"].split("::")[-1]
        require(
            any(
                re.search(rf"\b{re.escape(predicate_name)}\b", requirement)
                is not None
                for claim in claims
                for requirement in claim["requires"]
            ),
            f"assumption {assumption['name']!r} is not exposed by the checked claim surface",
        )
        names.add(assumption["name"])
    return assumptions


def validate_all(surface: dict | None = None, imports: dict[str, dict] | None = None) -> dict:
    surface = read_json(SURFACE_PATH) if surface is None else surface
    require(set(surface) == SURFACE_FIELDS and surface["schema_version"] == 6,
            "unsupported claim-surface schema")
    sets = certificate_import_sets()
    imports = ({name: read_json(entry["imports_path"]) for name, entry in sets.items()}
               if imports is None else imports)
    require(isinstance(imports, dict) and set(imports) == set(sets),
            "receipt records must cover every registered kernel interface")
    pins = surface["kernel_interface_receipt_sha256"]
    require(isinstance(pins, dict) and set(pins) == set(sets), "interface receipt pins are not closed")
    # Adding a producer requires an explicit auditor; never silently skip it.
    rectangular = rectangular_operators_from_source(ROOT)
    require(set(sets) == {"attention", "kv_store"} | set(rectangular), "unimplemented shared interface auditor")
    qualification, _ = load_registry(ROOT)
    imported = validate_attention_interface("attention", sets["attention"], imports["attention"],
                                           pins["attention"], attention_inventory_from_source(qualification, ROOT), ROOT)
    imported.update(validate_mutation_interface("kv_store", sets["kv_store"], imports["kv_store"],
        pins["kv_store"], rectangular_inventory_from_source(qualification, ROOT,
            "store_kv_cache.py", "store_cache_kernel"), ROOT))
    for name, (source, kernel) in sorted(rectangular.items()):
        imported.update(validate_rectangular_interface(name, sets[name], imports[name],
            pins[name], rectangular_inventory_from_source(qualification, ROOT, source, kernel), ROOT))
    consumers = validate_consumers(surface, imported, sets)
    claims = validate_top_claims(surface)
    supporting = validate_supporting_entrypoints(surface)
    require(
        not ({entry["name"] for entry in claims} & {entry["name"] for entry in supporting})
        and not ({entry["item"] for entry in claims} & {entry["item"] for entry in supporting}),
        "top-level and supporting claim surfaces overlap",
    )
    assumptions = validate_assumptions(surface, claims + supporting)
    proofs, uninterpreted, declaration_digests = validate_trusted_declaration_inventory(surface)
    require({p["item"] for p in proofs if p["class"] == "kernel_certificate"} == set(imported),
            "trusted kernel-certificate inventory differs from generated raw imports")
    nonproduction = validate_nonproduction_trusted_inventory(surface)
    runtime_bridge_scope = read_json(RUNTIME_BRIDGE_SCOPE_PATH)
    require(set(runtime_bridge_scope) == {"schema_version", "bridges"}
            and runtime_bridge_scope["schema_version"] == 1, "unsupported runtime-bridge scope")
    runtime, types = validate_runtime_inventory(surface, runtime_attestation_scope(runtime_bridge_scope))
    return {
        "claims": claims, "supporting": supporting, "assumptions": assumptions,
        "imports": imported, "consumers": consumers, "proofs": proofs,
        "uninterpreted": uninterpreted, "declaration_digests": declaration_digests,
        "nonproduction": nonproduction, "runtime": runtime, "types": types,
    }




def render_markdown(ledger: dict) -> str:
    family_role_counts = Counter(
        (family, ledger["consumers"][proof_name]["role"])
        for proof_name, record in ledger["imports"].items()
        for family in record["families"]
    )
    out = [
        "# Serving-determinism property and audit ledger",
        "",
        "Generated by `scripts/audit/claim_ledger.py` from the claim manifest, kernel imports, and trusted declarations. The script checks their source references and this document's freshness. Run the verification commands in [tcb.md](tcb.md) to check the proofs.",
        "",
        "## How to read this ledger",
        "",
        "- Review determines whether the property statements express the intended claim; `make verify` checks their proofs. The `verus_checked` label identifies the verification method, not a fresh result from this generator.",
        "- A `claim_supporting` kernel certificate has at least one explicitly enumerated direct call from a checked consumer module. The ledger validates that complete direct-call set; it does not construct a transitive call graph to a top-level theorem.",
        "- Every listed raw import has a checked consumer. Static implementations of the same interface have separate proof records.",
        "- Each static implementation is separately qualified during the kernel gate and recorded in its interface receipt. The engine proof sees one fixed implementation's relational semantics, without block sizes or cross-configuration equality.",
        "- Certificate counts are abstract imported-theorem counts, not specialization counts. Shared interfaces have no owning model family.",
        "",
        "## Coverage summary",
        "",
        f"The manifest lists {len(ledger['claims'])} properties, {len(ledger['supporting'])} supporting checked entrypoints, and {len(ledger['assumptions'])} deployment assumptions.",
        "",
        "| Family | Applicable shared raw imports |",
        "|---|---:|",
    ]
    for family in sorted(
        {
            family
            for record in ledger["imports"].values()
            for family in record["families"]
        }
    ):
        out.append(
            f"| `{family}` | {family_role_counts[family, 'claim_supporting']} |"
        )
    out.extend([
        "",
        "## Intent-trusted, Verus-checked properties",
        "",
        "Review establishes whether these statements capture the intended properties; Verus checks their proof bodies.",
        "",
        "| Property | Verus item | Formalization role | Verification | Preconditions exposed | Checked conclusion surface | Limitations |",
        "|---|---|---|---|---|---|---|",
    ])
    for claim in ledger["claims"]:
        out.append(
            f"| `{claim['name']}` | `{claim['item']}` | "
            f"`{claim['formalization_role']}` | `{claim['verification_status']}` | "
            f"{', '.join(f'`{item}`' for item in claim['requires'])} | "
            f"{', '.join(f'`{item}`' for item in claim['ensures'])} | "
            f"{', '.join(f'`{item}`' for item in claim['limitations'])} |"
        )
    out.extend([
        "",
        "## Supporting checked entrypoints and corollaries",
        "",
        "These functions connect the proofs to engine initialization and execution, and provide trace corollaries.",
        "",
        "| Entry point | Verus item | Preconditions exposed | Checked conclusion surface | Limitations |",
        "|---|---|---|---|---|",
    ])
    for entry in ledger["supporting"]:
        out.append(
            f"| `{entry['name']}` | `{entry['item']}` | "
            f"{', '.join(f'`{item}`' for item in entry['requires'])} | "
            f"{', '.join(f'`{item}`' for item in entry['ensures'])} | "
            f"{', '.join(f'`{item}`' for item in entry['limitations'])} |"
        )
    out.extend(["", "## Shared raw interfaces and checked consumers", "",
        "| Interface | Applicable families | Imported raw theorem | Separate implementations | Analyzer premises | Direct checked consumers |",
        "|---|---|---|---:|---|---|"])
    for proof_name, record in sorted(ledger["imports"].items()):
        conditions = record["theory"]["analyzer_conditions"]
        premises = [f"{c['kind']}: {c['label']} ({c['predicate_name']})" for c in conditions]
        out.append(
            f"| `{record['interface']}{('/' + record['module']) if record['module'] else ''}` | "
            f"{', '.join(record['families'])} | `{proof_name}` | "
            f"{len(record['implementations'])} | {'; '.join(premises) or 'none'} | "
            f"{', '.join(record['consumers'])} |"
        )
    out.extend(["",
        "Each row describes one interface for a fixed deployed implementation; it does not assert equality across configurations. The audit checks recorded sources, configurations, generated modules, proof imports, and direct checked callers. Verus and the kernel verifier run separately.",
        "", "## Explicit deployment assumptions", ""])
    for assumption in ledger["assumptions"]:
        out.append(f"- `{assumption['predicate']}` ({assumption['status']}): {assumption['scope']}")
    out.extend(["", "## Trusted executable boundary inventory", "", "| Class | Claim-reachable | Deployment setup | Test only |", "|---|---:|---:|---:|"])
    counts = Counter((entry["class"], entry["scope"]) for entry in ledger["runtime"])
    for class_name in sorted({entry["class"] for entry in ledger["runtime"]}):
        out.append(f"| `{class_name}` | {counts[class_name, 'claim_reachable']} | {counts[class_name, 'deployment_setup']} | {counts[class_name, 'test_only']} |")
    unattested = [entry["item"] for entry in ledger["runtime"] if entry["attestation"] == "none"]
    out.extend([
        "",
        f"All {sum(entry['scope'] == 'claim_reachable' for entry in ledger['runtime'])} claim-reachable external executable functions are source-attested. Explicitly unattested non-claim functions: {', '.join(f'`{item}`' for item in unattested) or 'none'}.",
        "",
        f"The closed opaque external-type inventory contains {len(ledger['types'])} items.",
        "",
        "## Closed trusted proof and uninterpreted specification inventories",
        "",
        "| Declaration kind | Count |",
        "|---|---:|",
        f"| Trusted proof functions | {len(ledger['proofs'])} |",
        f"| Uninterpreted specifications | {len(ledger['uninterpreted'])} |",
        "",
        f"All {len(ledger['declaration_digests'])} external-body and uninterpreted declarations covered by this manifest are matched by qualified module/type/name identity and an exact SHA-256 digest of their attributes, signature, and body.",
        "",
        f"A separate closed exclusion inventory source-pins {len(ledger['nonproduction'])} trust-bearing declarations under `tests/` and `examples/`; none is part of the production theorem surface.",
        "",
        "## Remaining systemic TCB",
        "",
        "- The kernel verifier, its Triton-to-IR translation, regional/mask/dependency analyses, and solver encoding must be sound.",
        "- The listed trusted Verus proof functions remain `external_body` imports. Checked adapters can discharge generated raw domains, but this ledger does not prove the verifier's artifact-to-Verus translation.",
        "- Host-only Rust/PyO3 functions are absent from Verus's VIR and from this `external_body` attestation table. They remain unverified executable TCB and are enumerated separately in [tcb.md](tcb.md).",
        "- Python/Torch objects, allocation and physical representation/layout/non-aliasing correspondence, and compiler/CUDA behavior remain outside Verus. The logical KV scatter effect is derived from an annotation-generated import by checked adapters.",
        "- The finite-value attention premise is assumed rather than established from runtime tensors.",
        "- The trace theorems prove arbitrary caller-chosen finite safety and per-request common-prefix determinism, not scheduling fairness, progress, or completion.",
        "",
    ])
    return "\n".join(out)


def main() -> None:
    write = "--write" in sys.argv[1:]
    require(set(sys.argv[1:]) <= {"--write"}, "unknown command-line argument")
    ledger = validate_all()
    expected = render_markdown(ledger)
    if write:
        DOC_PATH.write_text(expected, encoding="utf-8")
        print(f"wrote {DOC_PATH.relative_to(ROOT)}")
    else:
        actual = DOC_PATH.read_text(encoding="utf-8") if DOC_PATH.exists() else None
        require(actual == expected, "audit/claims.md is stale; run `python3 scripts/audit/claim_ledger.py --write`")
        print(
            f"Claim ledger passed: {len(ledger['claims'])} top-level claims, "
            f"{len(ledger['supporting'])} supporting checked entrypoints, "
            f"{len(ledger['imports'])} imported certificates, "
            f"{len(ledger['proofs'])} trusted proof functions, "
            f"{len(ledger['uninterpreted'])} uninterpreted specifications, "
            f"{len(ledger['declaration_digests'])} source-pinned trusted declarations, "
            f"{len(ledger['nonproduction'])} source-pinned non-production exclusions, "
            f"{len(ledger['runtime'])} trusted runtime functions, "
            f"{len(ledger['types'])} external types."
        )


if __name__ == "__main__":
    main()
