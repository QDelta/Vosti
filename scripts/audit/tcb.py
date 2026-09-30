#!/usr/bin/env python3
"""Generate audit/tcb.md: verified-vs-trusted accounting for the paper.

Static analysis over src/ (Rust/Verus), python/vosti_kernels (trusted
executable glue), the proof/bridge orchestration surface, deployment-time
qualification code, and kernels (kernels + checker infrastructure).
Function classification:

  exec-verified   fn with a body checked by Verus
  exec-host       #[cfg(not(verus_only))] fn absent from the Verus build
  exec-trusted    #[verifier::external_body] exec fn (ensures are trusted)
  proof           proof fn with a checked body
  proof-trusted   #[verifier::external_body] proof fn (an axiom with args)
  spec            open/closed spec fn (definitions; not trust, but seam
                  surface when consumed by trusted ensures)
  uninterp        uninterp spec fn (semantic trust points)

LOC is nonblank, noncomment-only lines inside the item (approximate,
brace-matched).  Run:  python3 scripts/audit/tcb.py [--time | --check]
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit.check_kernel_sources import resolve_kernel_sources
from scripts.audit.rust_source import code_only


def integration_source_groups(inventory, scope_files):
    """Partition shared integration sources; other source tables stay separate."""
    qualification = {path for path in inventory
                     if path.startswith("kernels/backend/")
                     or path == "scripts/deployment/common.py"
                     or path == "scripts/prepare_deployment.py"
                     or (re.fullmatch(r"scripts/deployment/model_families/\w+\.py", path)
                         and not path.endswith('_scope.py'))}
    bridge = {
        "Makefile", "scripts/project.py", "python/vosti_kernels/kernel_interfaces.py",
        "python/vosti_kernels/runtime_bridge_scope.json", "audit/claim_surface.json",
        "scripts/audit/tcb.py", "scripts/audit/claim_ledger.py",
        "scripts/audit/check_verus_trust_manifest.py",
        *scope_files,
    } | {path for path in inventory if path.startswith("scripts/") and path not in qualification}
    return sorted(bridge), sorted(qualification)


def rust_module_name(path: Path) -> str:
    """Return a stable qualified module name for project Rust sources."""
    for source_root in (ROOT / "src", ROOT / "tests", ROOT / "examples"):
        try:
            relative = path.resolve().relative_to(source_root.resolve()).with_suffix("")
        except ValueError:
            continue
        parts = list(relative.parts)
        if parts and parts[-1] == "mod":
            parts.pop()
        return "::".join(parts) if parts else source_root.name
    return path.stem


def rust_source_files(directory: Path) -> list[Path]:
    """Return every Rust source below a crate/source subtree."""
    return sorted(directory.rglob("*.rs"))

_VISIBILITY = r"(?:pub(?:\s*\([^)]*\))?\s+)?"
FN_RE = re.compile(
    rf"^\s*{_VISIBILITY}(?:broadcast\s+)?(?:unsafe\s+)?(?:async\s+)?(?:const\s+)?"
    r"(?P<kind>(?:uninterp\s+spec|open\s+spec|closed\s+spec|spec|proof)\s+)?fn\s+"
    r"(?P<name>\w+)"
)
ATTR_EB = "#[verifier::external_body]"
ATTR_CFG_DISABLED = "#[cfg(any())]"
ATTR_CFG_HOST_ONLY = "#[cfg(not(verus_only))]"
TYPE_RE = re.compile(
    rf"^\s*{_VISIBILITY}(?:(?:tracked|ghost)\s+)?struct\s+(?P<name>\w+)"
)
IMPL_FOR_RE = re.compile(
    r"^\s*impl(?:\s*<[^>{}]*>)?\s+[^{}]+\s+for\s+(?P<owner>\w+)\s*\{"
)
IMPL_INHERENT_RE = re.compile(
    r"^\s*impl(?:\s*<[^>{}]*>)?\s+(?P<owner>\w+)\s*\{"
)
INLINE_MODULE_RE = re.compile(rf"^\s*{_VISIBILITY}mod\s+(?P<name>\w+)\s*\{{")


def classify(kind: str | None, eb: bool, host_only: bool = False) -> str:
    if kind is None:
        if host_only:
            return "exec-host"
        return "exec-trusted" if eb else "exec-verified"
    kind = kind.strip()
    if kind.startswith("uninterp"):
        return "uninterp"
    if kind.endswith("spec"):
        return "spec"
    if kind == "proof":
        return "proof-trusted" if eb else "proof"
    return "spec"


def _normalized_attribute(lines: list[str], start: int) -> tuple[str, int]:
    """Read one possibly multiline attribute and normalize insignificant spaces."""
    parts = []
    square_depth = 0
    index = start
    while index < len(lines):
        part = lines[index].strip()
        parts.append(part)
        square_depth += part.count("[") - part.count("]")
        index += 1
        if square_depth <= 0:
            break
    return re.sub(r"\s+", "", "".join(parts)), index


def _item_end(lines: list[str], start: int) -> tuple[int, int]:
    """Return the inclusive item end and non-comment LOC, or fail closed."""
    stack = []
    loc = 0
    index = start
    pairs = {")": "(", "]": "[", "}": "{"}
    while index < len(lines):
        line = lines[index]
        stripped = line.strip()
        if stripped and not stripped.startswith("//"):
            loc += 1
        for column, char in enumerate(line):
            if char in "([{":
                stack.append(char)
            elif char in ")]}":
                if not stack or stack.pop() != pairs[char]:
                    raise ValueError(f"unbalanced declaration starting at line {start + 1}")
                if char == "}" and not stack:
                    # Verus signatures can contain quantified block expressions
                    # and struct literals before the function body. A signature
                    # expression continues into a clause, comma, operator, or
                    # the actual body; it must not truncate the source digest.
                    following = line[column + 1:].strip()
                    lookahead = index + 1
                    while not following and lookahead < len(lines):
                        following = lines[lookahead].strip()
                        lookahead += 1
                    continuation = re.match(
                        r"^(?:[,({)\[\].:+*/%<>=!&|^?\-]|"
                        r"(?:requires|ensures|recommends|decreases|returns|opens_invariants|"
                        r"no_unwind|when|via|where|else|as|is|has|matches)\b)", following)
                    if not continuation:
                        return index, loc
            elif char == ";" and not stack:
                return index, loc
        index += 1
    raise ValueError(f"unterminated declaration starting at line {start + 1}")


def _source_sha256(raw_lines: list[str], start: int, end: int) -> str:
    source = "".join(raw_lines[start : end + 1])
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def _impl_owners(lines: list[str]) -> dict[int, str]:
    """Map source lines inside an impl block to the impl target type."""
    owners = {}
    stack: list[tuple[int, str]] = []
    brace_depth = 0
    for line_number, line in enumerate(lines, 1):
        while stack and brace_depth < stack[-1][0]:
            stack.pop()
        if stack:
            owners[line_number] = stack[-1][1]
        code = line.split("//", 1)[0]
        match = IMPL_FOR_RE.match(code) or IMPL_INHERENT_RE.match(code)
        next_depth = brace_depth + code.count("{") - code.count("}")
        if match and "{" in code:
            stack.append((next_depth, match.group("owner")))
        brace_depth = next_depth
    return owners


def _module_owners(lines: list[str]) -> dict[int, tuple[str, ...]]:
    """Map lines to their enclosing inline modules (verus! is transparent)."""
    owners = {}
    stack = []
    depth = 0
    for line_number, code in enumerate(lines, 1):
        while stack and depth < stack[-1][0]:
            stack.pop()
        owners[line_number] = tuple(name for _, name in stack)
        match = INLINE_MODULE_RE.match(code)
        if match:
            stack.append((depth + 1, match["name"]))
        depth += code.count("{") - code.count("}")
        if depth < 0:
            raise ValueError(f"unbalanced Rust module scope at line {line_number}")
    if depth != 0:
        raise ValueError("unterminated Rust module scope")
    return owners


def _scan_rust_items(path: Path):
    items = []
    source = path.read_text(encoding="utf-8")
    raw_lines = source.splitlines(keepends=True)
    lines = code_only(source).splitlines()
    owners = _impl_owners(lines)
    modules = _module_owners(lines)
    pending_attributes: list[str] = []
    pending_start: int | None = None
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if stripped.startswith("#["):
            if pending_start is None:
                pending_start = i
            attribute, i = _normalized_attribute(lines, i)
            if "external_body" in attribute and attribute != ATTR_EB:
                raise ValueError(
                    f"unsupported external_body attribute at {path}:{i}"
                )
            pending_attributes.append(attribute)
            continue
        m = FN_RE.match(line)
        if m and not stripped.startswith("//"):
            kind = (m.group("kind") or "").strip()
            if ATTR_EB in pending_attributes and kind.endswith("spec"):
                raise ValueError(
                    f"unsupported external_body spec function at {path}:{i + 1}"
                )
            host_only = ATTR_CFG_HOST_ONLY in pending_attributes
            if host_only and (kind or ATTR_EB in pending_attributes):
                raise ValueError(
                    f"unsupported host-only Verus/trusted function at {path}:{i + 1}"
                )
            cat = classify(
                m.group("kind"), ATTR_EB in pending_attributes, host_only
            )
            disabled = ATTR_CFG_DISABLED in pending_attributes
            source_start = pending_start if pending_start is not None else i
            j, loc = _item_end(lines, i)
            line_number = i + 1
            owner = owners.get(line_number)
            module = "::".join((rust_module_name(path), *modules[line_number]))
            identity = (
                f"{module}::{owner}::{m.group('name')}"
                if owner is not None
                else f"{module}::{m.group('name')}"
            )
            items.append(
                {
                    "name": m.group("name"),
                    "identity": identity,
                    "category": cat,
                    "loc": loc,
                    "line": line_number,
                    "end_line": j + 1,
                    "module": module,
                    "disabled": disabled,
                    "source_sha256": _source_sha256(raw_lines, source_start, j),
                }
            )
            pending_attributes = []
            pending_start = None
            i = j + 1
            continue
        type_match = TYPE_RE.match(line)
        if type_match and not stripped.startswith("//"):
            pending_attributes = []
            pending_start = None
            i += 1
            continue
        code = line.split("//", 1)[0]
        if re.search(r"\buninterp\b", code):
            raise ValueError(
                f"unrecognized uninterpreted declaration at {path}:{i + 1}"
            )
        if stripped and not stripped.startswith("//"):
            if ATTR_EB in pending_attributes:
                raise ValueError(
                    f"unrecognized external_body target at {path}:{i + 1}"
                )
            pending_attributes = []
            pending_start = None
        i += 1
    if ATTR_EB in pending_attributes:
        raise ValueError(f"unterminated external_body attribute in {path}")
    return items


def scan_rust(path: Path, *, include_disabled: bool = False):
    """Return active Rust/Verus functions as name/category/LOC/line tuples."""
    return [
        (item["name"], item["category"], item["loc"], item["line"])
        for item in _scan_rust_items(path)
        if include_disabled or not item["disabled"]
    ]


def scan_rust_item_records(path: Path, *, include_disabled: bool = False):
    """Qualified functions with source extents for checked-consumer auditing."""
    return [dict(item) for item in _scan_rust_items(path)
            if include_disabled or not item["disabled"]]


def scan_rust_trust(path: Path, *, include_disabled: bool = False):
    """Return qualified identities for soundness-critical trusted declarations."""
    return [
        (
            item["identity"],
            item["category"],
            item["loc"],
            item["line"],
        )
        for item in _scan_rust_items(path)
        if (include_disabled or not item["disabled"])
        and item["category"] in {"exec-trusted", "proof-trusted", "uninterp"}
    ]


def scan_rust_trust_records(path: Path, *, include_disabled: bool = False):
    """Return source-pinned records for trust-bearing function declarations."""
    return [
        dict(item)
        for item in _scan_rust_items(path)
        if (include_disabled or not item["disabled"])
        and item["category"] in {"exec-trusted", "proof-trusted", "uninterp"}
    ]


def scan_disabled_rust(path: Path):
    return [
        (item["identity"], item["category"], item["loc"], item["line"])
        for item in _scan_rust_items(path)
        if item["disabled"]
    ]


def scan_external_type_records(path: Path, *, include_disabled: bool = False):
    """Return source-pinned external-body structs omitted by fn accounting."""
    items = []
    source = path.read_text(encoding="utf-8")
    raw_lines = source.splitlines(keepends=True)
    lines = code_only(source).splitlines()
    modules = _module_owners(lines)
    pending_attributes: list[str] = []
    pending_start: int | None = None
    index = 0
    while index < len(lines):
        line = lines[index]
        line_number = index + 1
        stripped = line.strip()
        if stripped.startswith("#["):
            if pending_start is None:
                pending_start = index
            attribute, index = _normalized_attribute(lines, index)
            if "external_body" in attribute and attribute != ATTR_EB:
                raise ValueError(
                    f"unsupported external_body attribute at {path}:{index}"
                )
            pending_attributes.append(attribute)
            continue
        match = TYPE_RE.match(line)
        if match and ATTR_EB in pending_attributes:
            disabled = ATTR_CFG_DISABLED in pending_attributes
            source_start = pending_start if pending_start is not None else index
            end, _ = _item_end(lines, index)
            if include_disabled or not disabled:
                items.append(
                    {
                        "name": match.group("name"),
                        "identity": "::".join((rust_module_name(path), *modules[line_number], match.group("name"))),
                        "line": line_number,
                        "disabled": disabled,
                        "source_sha256": _source_sha256(
                            raw_lines, source_start, end
                        ),
                    }
                )
            pending_attributes = []
            pending_start = None
            index = end + 1
            continue
        if stripped and not stripped.startswith("//"):
            if ATTR_EB in pending_attributes and not FN_RE.match(line):
                raise ValueError(
                    f"unrecognized external_body target at {path}:{line_number}"
                )
            pending_attributes = []
            pending_start = None
        index += 1
    return items


def scan_external_types(path: Path):
    """Return external-body structs as name/line tuples."""
    return [
        (item["name"], item["line"])
        for item in scan_external_type_records(path)
    ]


def loc_of(path: Path) -> int:
    n = 0
    for l in path.read_text().splitlines():
        s = l.strip()
        if s and not s.startswith("#") and not s.startswith("//"):
            n += 1
    return n


def checked_output(args: list[str], cwd: Path) -> str:
    result = subprocess.run(args, cwd=cwd, capture_output=True, text=True)
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(f"command failed: {' '.join(args)}: {detail}")
    return result.stdout.strip()


def resolve_kernel_checkout() -> tuple[Path, str]:
    """Resolve the kernel directory and check its scoped source digests."""
    configured = os.environ.get("KERNELS_DIR")
    return resolve_kernel_sources(ROOT, Path(configured) if configured else None)


CATS = [
    "exec-verified",
    "exec-host",
    "exec-trusted",
    "proof",
    "proof-trusted",
    "spec",
    "uninterp",
]


def main() -> None:
    arguments = set(sys.argv[1:])
    if not arguments <= {"--time", "--check"}:
        raise SystemExit("usage: tcb.py [--time] [--check]")
    if "--time" in arguments and "--check" in arguments:
        raise SystemExit("--time and --check are mutually exclusive")
    do_time = "--time" in arguments
    check_only = "--check" in arguments
    out = []
    out.append("# TCB and proof accounting (generated by scripts/audit/tcb.py)\n")
    out.append(
        "This inventory lists checked code, trusted declarations, and "
        "verification and qualification tools. Regenerate it with "
        "`python3 scripts/audit/tcb.py`; `python3 scripts/audit/tcb.py --check` fails if the snapshot is "
        "stale. See [claims.md](claims.md) for the theorem-to-certificate "
        "ledger and [verification](../docs/verification.md) for the supported claim.\n"
    )
    out.append("## How to read this inventory\n")
    out.append(
        "- Verus checks executable and proof bodies against their specifications, "
        "subject to the assumptions listed below."
    )
    out.append(
        "- Host-only executable functions are absent from Verus's VIR. "
        "Trusted executable functions have `external_body`, so their "
        "ensures clauses are assumed. Both are executable TCB."
    )
    out.append(
        "- Trusted proof functions are imported axioms with arguments. "
        "Uninterpreted specifications describe assumed semantics. Ordinary "
        "specification definitions are counted separately."
    )
    out.append(
        "- Python glue, proof drivers, and qualification tooling are "
        "outside Verus. Kernel-verifier infrastructure is trusted; deployment "
        "probes provide empirical checks.\n"
    )
    out.append("## Verification and qualification pipeline\n")
    out.append(
        "See [setup](../docs/setup.md#cpu-side-checks) for CPU verification, "
        "regression, and audit commands, and [deployment](../docs/deployment.md) "
        "for specialization proofs and backend qualification. "
        "GPU/model tests and bounded diagnostics are empirical evidence; "
        "they do not establish unbounded proofs.\n"
    )

    per_module: dict[str, dict[str, int]] = {}
    inventories: dict[str, list[tuple[str, str, str, int]]] = {
        "exec-host": [],
        "exec-trusted": [],
        "proof-trusted": [],
        "uninterp": [],
    }
    external_types: list[tuple[str, str, str, int]] = []
    disabled_items: list[tuple[str, str, int, int]] = []
    for f in rust_source_files(ROOT / "src"):
        module = rust_module_name(f)
        source_path = f.relative_to(ROOT).as_posix()
        items = scan_rust(f)
        disabled_items.extend(
            (identity, category, loc, line)
            for identity, category, loc, line in scan_disabled_rust(f)
        )
        external_types.extend(
            (module, source_path, name, line)
            for name, line in scan_external_types(f)
        )
        agg = {c: 0 for c in CATS}
        cnt = {c: 0 for c in CATS}
        for name, cat, loc, line in items:
            agg[cat] += loc
            cnt[cat] += 1
            if cat in inventories:
                inventories[cat].append((module, source_path, name, line))
        if any(cnt.values()):
            per_module[module] = {
                "loc": {c: agg[c] for c in CATS},
                "cnt": {c: cnt[c] for c in CATS},
            }

    out.append("## Rust/Verus: lines per category, by module\n")
    out.append("| module | exec-verified | exec-host | exec-trusted | proof | proof-trusted | spec | uninterp decls |")
    out.append("|---|---|---|---|---|---|---|---|")
    tot = {c: 0 for c in CATS}
    totc = {c: 0 for c in CATS}
    for mod, d in per_module.items():
        for c in CATS:
            tot[c] += d["loc"][c]
            totc[c] += d["cnt"][c]
        out.append(
            f"| {mod} | {d['loc']['exec-verified']} | {d['loc']['exec-host']} "
            f"| {d['loc']['exec-trusted']} "
            f"| {d['loc']['proof']} | {d['loc']['proof-trusted']} | {d['loc']['spec']} "
            f"| {d['cnt']['uninterp']} |")
    out.append(
        f"| **total** | **{tot['exec-verified']}** | **{tot['exec-host']}** "
        f"| **{tot['exec-trusted']}** "
        f"| **{tot['proof']}** | **{tot['proof-trusted']}** | **{tot['spec']}** "
        f"| **{totc['uninterp']}** |")
    out.append("")
    out.append(f"Function counts: {totc['exec-verified']} verified exec fns, "
               f"{totc['exec-host']} host-only unverified exec fns, "
               f"{totc['exec-trusted']} trusted exec fns (external_body), "
               f"{totc['proof']} checked proof fns, "
               f"{totc['proof-trusted']} trusted proof fns, "
               f"{totc['spec']} spec fns, {totc['uninterp']} uninterpreted spec fns.\n")
    out.append(
        "The authoritative `make verify-engine` gate additionally cross-checks every "
        "active trusted source declaration against the typed Function/Datatype "
        "modes in Verus's emitted VIR; source accounting is not the sole "
        "inventory mechanism.\n"
    )
    out.append(
        "Host-only exec functions are compiled only by ordinary Rust/PyO3 "
        "(`#[cfg(not(verus_only))]`), are absent from the Verus VIR, and are "
        "therefore explicitly counted as unverified executable TCB.\n"
    )
    out.append(
        f"External-body type count: {len(external_types)} opaque executable/tracked structs.\n"
    )
    disabled_counts: dict[str, int] = {}
    disabled_loc: dict[str, int] = {}
    for _, category, loc, _ in disabled_items:
        disabled_counts[category] = disabled_counts.get(category, 0) + 1
        disabled_loc[category] = disabled_loc.get(category, 0) + loc
    if disabled_items:
        detail = ", ".join(
            f"{disabled_counts[category]} {category} functions / "
            f"{disabled_loc[category]} LOC"
            for category in sorted(disabled_counts)
        )
        out.append(
            f"Cfg-disabled items excluded from active totals: {detail}.\n"
        )

    nonproduction = []
    for scope, directory in (
        ("example", ROOT / "examples"),
        ("test", ROOT / "tests"),
    ):
        for path in rust_source_files(directory):
            for record in scan_rust_trust_records(path):
                nonproduction.append(
                    (
                        scope,
                        record["identity"],
                        record["category"],
                        record["source_sha256"],
                    )
                )
            for record in scan_external_type_records(path):
                nonproduction.append(
                    (
                        scope,
                        record["identity"],
                        "external-type",
                        record["source_sha256"],
                    )
                )
    out.append("## Explicitly excluded non-production trust\n")
    if nonproduction:
        out.append(
            "These declarations are source-pinned by `audit/claim_surface.json` "
            "but excluded from the production theorem surface.\n"
        )
        out.append("| scope | declaration | category | source SHA-256 |")
        out.append("|---|---|---|---|")
        for scope, identity, category, source_digest in nonproduction:
            out.append(
                f"| {scope} | `{identity}` | {category} | `{source_digest}` |"
            )
        out.append("")
    else:
        out.append("There are no trust-bearing declarations under `tests/` or `examples/`.\n")

    out.append("## Trusted executable glue (Python)\n")
    py_total = 0
    out.append("| file | LOC |")
    out.append("|---|---|")
    python_runtime_root = ROOT / "python" / "vosti_kernels"
    for f in sorted(
        path
        for path in python_runtime_root.rglob("*.py")
        if "__pycache__" not in path.parts
    ):
        n = loc_of(f)
        py_total += n
        out.append(f"| {f.relative_to(ROOT).as_posix()} | {n} |")
    out.append(f"| **total** | **{py_total}** |\n")

    out.append("## Kernel proof layer (kernels)\n")
    mk, kernel_sources = resolve_kernel_checkout()
    tk = sum(loc_of(f) for f in (mk / "triton_kernels").glob("*.py"))
    ir = sum(loc_of(f) for f in (mk / "ir").glob("*.py") if not f.name.startswith("test_"))
    diagnostics = sum(loc_of(f) for f in (mk / "diagnostics").glob("*.py"))
    ir_test = sum(loc_of(f) for f in (mk / "tests").rglob("*.py"))
    out.append(f"- Scoped kernel/support source digest: `{kernel_sources}`, checked against "
               "every model-family `scope.json`")
    out.append(f"- Triton kernels under proof (incl. wrappers/annotations): {tk} LOC")
    out.append(f"- Verifier infrastructure (translator, region analysis, Z3 backend, "
               f"unified relational verifier; trusted): {ir} LOC")
    out.append(f"- Optional logical-race and bounded-suffix diagnostics "
               f"(not dependencies of the equality proof): {diagnostics} LOC")
    out.append(f"- Verifier test suite: {ir_test} LOC\n")

    out.append("## Verification bridge and proof drivers (trusted)\n")
    # Share the tracked, static-import inventory with effort accounting.
    # New model helpers and binding modules must not need another manual list.
    from scripts.effort.account import source_inventory, script_dependencies, tracked
    inventory, scope_files = source_inventory(ROOT)
    audit_dependencies = script_dependencies(ROOT, tracked(ROOT), [
        "scripts/audit/claim_ledger.py", "scripts/audit/check_verus_trust_manifest.py",
        *(str(path.relative_to(ROOT)) for path in (ROOT / "scripts/verification/diagnostics").glob("verify_*_kernel_races.py")),
    ])
    bridge_paths, qualification_paths = integration_source_groups(
        set(inventory) | audit_dependencies, scope_files)
    bridge_files = [(path, ROOT / path) for path in bridge_paths]
    bridge_total = 0
    out.append("| file | LOC |")
    out.append("|---|---|")
    for label, path in bridge_files:
        n = loc_of(path)
        bridge_total += n
        out.append(f"| {label} | {n} |")
    out.append(f"| **total** | **{bridge_total}** |\n")
    out.append(
        "These files define the claimed properties, select proof cases, and "
        "interpret verifier results, so they are trusted. The suffix checker "
        "covers bounded cases only. Generated records such as "
        "`audit/attention_kernel_interfaces.json` are excluded from the LOC "
        "total; their freshness is checked against current proof results.\n"
    )

    out.append("## Deployment qualification gate (trusted empirical tooling)\n")
    qualification_files = [(path, ROOT / path) for path in qualification_paths]
    qualification_total = 0
    out.append("| file | LOC |")
    out.append("|---|---|")
    for label, path in qualification_files:
        n = loc_of(path)
        qualification_total += n
        out.append(f"| {label} | {n} |")
    out.append(f"| **total** | **{qualification_total}** |\n")
    out.append(
        "These tools select deployment proof cases, run verification and "
        "backend probes, and decide whether to seal a candidate. They are "
        "trusted because a bug could accept insufficient evidence. Passing "
        "probes does not prove compiler or GPU correctness. The requirement "
        "exporter is counted above under `kernels/ir`.\n"
    )

    out.append("## Trust inventories (the named items)\n")
    out.append("### Host-only exec fns (unverified executable TCB)\n")
    for mod, source_path, name, line in inventories["exec-host"]:
        out.append(f"- `{mod}::{name}` (`{source_path}:{line}`)")
    out.append("")
    out.append("### External-body types (opaque executable/tracked representations)\n")
    for mod, source_path, name, line in external_types:
        out.append(f"- `{mod}::{name}` (`{source_path}:{line}`)")
    out.append("")
    out.append("### Uninterpreted spec functions (semantic trust points)\n")
    for mod, source_path, name, line in inventories["uninterp"]:
        out.append(f"- `{mod}::{name}` (`{source_path}:{line}`)")
    out.append("\n### Trusted proof fns (axioms-with-arguments)\n")
    for mod, source_path, name, line in inventories["proof-trusted"]:
        out.append(f"- `{mod}::{name}` (`{source_path}:{line}`)")
    out.append("\n### Trusted exec fns (external_body; ensures are the kernel/wiring contracts)\n")
    for mod, source_path, name, line in inventories["exec-trusted"]:
        out.append(f"- `{mod}::{name}` (`{source_path}:{line}`)")
    out.append("")

    if do_time:
        out.append("## Verification wall-clock\n")
        t0 = time.time()
        r = subprocess.run(["make", "verify-engine"], cwd=ROOT, capture_output=True, text=True)
        dt = time.time() - t0
        res = [l for l in (r.stdout + r.stderr).splitlines() if "verification results" in l]
        status = "PASS" if r.returncode == 0 else f"FAIL (exit {r.returncode})"
        out.append(f"- `make verify-engine` (full engine gate, including orchestration): {status}, {dt:.0f}s — "
                   f"{res[-1].split('::')[-1].strip() if res else 'n/a'}")
        t0 = time.time()
        r = subprocess.run(["make", "verify-kernels"], cwd=ROOT, capture_output=True, text=True)
        dt = time.time() - t0
        status = "PASS" if r.returncode == 0 else f"FAIL (exit {r.returncode})"
        out.append(f"- `make verify-kernels` (deployed certificate gate, including "
                   f"orchestration): {status}, {dt:.0f}s")
        out.append("")

    rendered = "\n".join(out).rstrip() + "\n"
    destination = ROOT / "audit" / "tcb.md"
    if check_only:
        if not destination.is_file() or destination.read_text() != rendered:
            raise SystemExit(
                "audit/tcb.md is stale; run python3 scripts/audit/tcb.py"
            )
        print("audit/tcb.md is fresh")
    else:
        destination.write_text(rendered)
        print(f"wrote audit/tcb.md ({len(out)} lines)")


if __name__ == "__main__":
    main()
