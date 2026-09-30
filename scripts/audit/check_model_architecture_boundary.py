#!/usr/bin/env python3
"""Reject model-family dispatch that leaks outside the reviewed boundaries."""

from __future__ import annotations

import ast
from importlib.util import resolve_name
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.verification.kernel_interface_registry import validate_registry, rectangular_operators_from_source
from scripts.audit.rust_source import code_only


# These files are the closed semantic/proof dispatch surface. A new payload
# variant may be mentioned here, but not in scheduler or unrelated proof code.
PAYLOAD_VARIANT_FILES = {
    "src/boundary/tensor_runtime.rs",
    "src/proof/model/types.rs",
    "src/proof/engine/architecture.rs",
    "src/proof/model/architecture.rs",
    "src/proof/model/relocation.rs",
    "src/exec/dense_swiglu_model.rs",
}

# Physical weights/runtime variants are confined to family construction and
# execution adapters. The Engine itself consumes only their common facade.
EXEC_VARIANT_FILES = {
    "src/boundary/model_deployment.rs",
    "src/boundary/tensor_runtime.rs",
    "src/exec/model.rs",
}

PAYLOAD_VARIANT = re.compile(r"\bModelWeightsArchitectureRepr::[A-Za-z0-9_]+")
EXEC_VARIANT = re.compile(r"\b(?:RT::)?Model(?:Weights|Runtime)::[A-Za-z0-9_]+")
ENGINE_ARCHITECTURE = re.compile(r"\bModelArchitecture::(?P<variant>[A-Za-z0-9_]+)")
ENGINE_RUNTIME = re.compile(r"\b(?:RT::)?ModelRuntime::(?P<variant>[A-Za-z0-9_]+)")

GENERIC_PROOF_DISPATCH_FILES = {
    "src/exec/model.rs",
    "src/proof/engine/architecture.rs",
    "src/proof/model/architecture.rs",
    "src/proof/model/relocation.rs",
}

# The neutral physical facade and its checked deployment assembler are the
# reviewed locations allowed to inspect the closed family-owned weight/runtime
# sums. No scheduler, Engine, or unrelated boundary module receives this
# exception.
GENERIC_PHYSICAL_DISPATCH_FILES = {
    "src/boundary/model_deployment.rs",
    "src/boundary/tensor_runtime.rs",
}

# These existing closed configuration projections need the family-owned data,
# but may not import a family's executable helpers or weight/deployment code.
CONFIGURATION_DATA_IMPORT_FILES = {
    "src/proof/model/types.rs",
    "src/boundary/dense_layer_primitives.rs",
}



# This neutral capability dispatcher is the single reviewed closed dispatch
# for the family-checked CUDA-graph adapters. It exports an
# architecture-neutral predicate/overlay to Engine; ordinary neutral modules
# may not import family implementations even if they mention every supported
# family in comments.
REVIEWED_OPTIONAL_CAPABILITY_DISPATCH_FILES = {
    "src/exec/model_families/mod.rs",
}

COMMON_DEPLOYMENT_ASSEMBLER = "src/boundary/model_deployment.rs"
COMMON_DEPLOYMENT_OPERATIONS = (
    "checkpoint_contents_valid",
    "staged_kernel_plan",
    "backend_qualified_kernel_plan",
    "assemble_qualified_model",
)

CRATE_IMPORT = re.compile(r"\buse\s+crate::(?P<path>[^;]+);")
PHYSICAL_WEIGHT_DECLARATION = re.compile(
    r"\bpub\s+(?:tracked\s+)?struct\s+"
    r"(?P<name>[A-Za-z_][A-Za-z0-9_]*(?:Layer|Model)Weights(?:Perms)?)\b"
)

FAMILY_CONTRACT_OPERATIONS = (
    "forward_logits_repr",
    "forward_kv_reprs",
    "reference_logits_last_row",
    "lemma_reference_logits_last_row_is_forward_last",
    "cache_refinement_supported",
    "semantic_model",
    "lemma_cache_refinement_laws",
    "request_projection_ready",
    "lemma_logits_request_isolation",
    "lemma_logits_repr_shape",
    "lemma_kv_reprs_len",
    "lemma_kv_reprs_empty",
    "lemma_forward_cache_shape_preserved",
    "layer_kv_rows",
    "lemma_forward_layer_store",
    "lemma_layer_kv_rows_shape",
    "lemma_forward_relocation",
    "lemma_kv_request_isolation",
)

# Executable family adapters now contain only the boundary-specific discharge
# needed by their selected shared composition. The public executable contract
# and its dense-SwiGLU implementation live at the neutral dispatch layer.
EXEC_FAMILY_CONTRACT_OPERATIONS: tuple[str, ...] = ()

BOUNDARY_FAMILY_CONTRACT_OPERATIONS = (
    "weights_extension_repr_of",
    "lemma_weights_extension_repr",
    "lemma_architecture_repr",
    "lemma_architecture_repr_implies_tag",
    "configuration_ready",
    "lemma_execution_valid_implies_configuration_ready",
)


def _is_configuration_data_import(path: str, families: set[str]) -> bool:
    match = re.fullmatch(
        r"boundary::model_families::(\w+)::config::(\w+|\{[^{}]+\})", path
    )
    if match is None or match[1] not in families:
        return False
    names = [name.strip() for name in match[2].strip("{}").split(",") if name.strip()]
    return bool(names) and all(re.fullmatch(
        r"[A-Z][A-Za-z0-9]*Config|[A-Z][A-Z0-9_]*", name
    ) for name in names)


def _direct_operation_signature(text: str, operation: str) -> str | None:
    """Return a function's parameter/return signature, excluding its contract."""

    match = re.search(
        rf"(?m)^[ \t]*"
        rf"(?:pub(?:\([^\n)]*\))?[ \t]+)?"
        rf"(?:(?:open|closed|spec|proof|exec|broadcast|unsafe|const|async)[ \t]+)*"
        rf"fn[ \t]+{re.escape(operation)}[ \t]*\(",
        text,
    )
    if match is None:
        return None
    opening = text.find("(", match.start())
    depth = 0
    closing = None
    for index in range(opening, len(text)):
        character = text[index]
        if character == "(":
            depth += 1
        elif character == ")":
            depth -= 1
            if depth == 0:
                closing = index
                break
    if closing is None:
        return None

    suffix_lines: list[str] = []
    for line in text[closing + 1 :].splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if re.match(
            r"^(?:requires|ensures|recommends|decreases)\b", stripped
        ) or stripped.startswith(("{", ";")):
            break
        for delimiter in ("{", ";"):
            if delimiter in stripped:
                prefix = stripped.split(delimiter, 1)[0].strip()
                if prefix:
                    suffix_lines.append(prefix)
                return text[opening : closing + 1] + " ".join(suffix_lines)
        suffix_lines.append(stripped)
    return text[opening : closing + 1] + " ".join(suffix_lines)


def _operation_signature(
    path: Path,
    family_root: Path,
    operation: str,
) -> str | None:
    """Resolve a direct declaration or a mod.rs public re-export."""

    text = _rust(path)
    direct = _direct_operation_signature(text, operation)
    if direct is not None:
        return direct
    if path.name != "mod.rs" or re.search(
        rf"\bpub\s+use\s+[^;]*\b{re.escape(operation)}\b[^;]*;", text
    ) is None:
        return None
    matches = [
        signature
        for candidate in sorted(family_root.rglob("*.rs"))
        if candidate != path
        for signature in [
            _direct_operation_signature(
                _rust(candidate), operation
            )
        ]
        if signature is not None
    ]
    return matches[0] if len(matches) == 1 else None


def _normalize_family_signature(signature: str, families: set[str]) -> str:
    """Erase the family spelling while preserving the adapter's type shape."""

    family_names = "|".join(re.escape(family) for family in sorted(families))
    normalized = re.sub(
        rf"\b(?:{family_names})(?:Text)?ModelWeightsExtensionRepr\b",
        "FamilyModelWeightsExtensionRepr",
        signature,
        flags=re.IGNORECASE,
    )
    normalized = re.sub(
        r"(?<![A-Za-z0-9_])_([A-Za-z][A-Za-z0-9_]*)\s*:",
        r"\1:",
        normalized,
    )
    return re.sub(r"\s+", "", normalized)


def _python_deployment_binding(text: str, operation: str | None = None) -> bool:
    """Recognize direct delegates and explicit partials of the shared API.

    Import aliases are irrelevant, but the source module, bound architecture,
    and delegated operation must agree. This remains a structural audit, not
    a Python behavior proof.
    """
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return False
    generic = {item.asname or item.name for node in tree.body
        if isinstance(node, ast.ImportFrom) and node.level == 3 and node.module is None
        for item in node.names if item.name == "deployment"}
    partials = {item.asname or item.name for node in tree.body
        if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module == "functools"
        for item in node.names if item.name == "partial"}

    def generic_call(node, name):
        return isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) \
            and node.value.id in generic and node.attr == name

    def architecture_arg(node):
        return isinstance(node, ast.Name) and node.id == "ARCHITECTURE"

    assignments = {node.targets[0].id: node.value for node in tree.body
        if isinstance(node, ast.Assign) and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)}
    architecture = assignments.get("ARCHITECTURE")
    if not (isinstance(architecture, ast.Call)
            and (generic_call(architecture.func, "DeploymentArchitecture")
                 or (isinstance(architecture.func, ast.Attribute)
                     and architecture.func.attr == "from_profile"
                     and generic_call(architecture.func.value, "DeploymentArchitecture")))):
        return False
    if operation is None:
        return True
    partial = assignments.get(operation)
    if isinstance(partial, ast.Call) and isinstance(partial.func, ast.Name) \
            and partial.func.id in partials and len(partial.args) == 2 \
            and not partial.keywords and generic_call(partial.args[0], operation) \
            and architecture_arg(partial.args[1]):
        return True
    return any(isinstance(call, ast.Call) and generic_call(call.func, operation)
        and call.args and architecture_arg(call.args[0])
        for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == operation
        for call in ast.walk(node))


def _python_inherited_operation(
    path: Path, runtime_root: Path, operation: str,
) -> bool:
    """Resolve explicit local class inheritance without importing runtime code.

    Only classes actually used as bases count. Wildcard imports, factories,
    dynamic attributes and files outside the reviewed runtime tree fail closed.
    This is an ownership/interface check, not a Python type or behavior proof.
    """
    runtime_root = runtime_root.resolve()
    visited: set[tuple[Path, str]] = set()

    def read(module: Path):
        module = module.resolve()
        if not module.is_relative_to(runtime_root) or not module.is_file():
            return None
        try:
            return ast.parse(module.read_text(encoding="utf-8"))
        except (OSError, SyntaxError):
            return None

    def base_target(module: Path, tree: ast.Module, name: str):
        if any(isinstance(node, ast.ClassDef) and node.name == name for node in tree.body):
            return module, name
        for node in tree.body:
            if not isinstance(node, ast.ImportFrom) or node.module is None:
                continue
            matched = next((item for item in node.names if (item.asname or item.name) == name), None)
            if matched is None or matched.name == "*":
                continue
            if node.level:
                parent = module.parent
                for _ in range(node.level - 1):
                    parent = parent.parent
                target = parent.joinpath(*node.module.split("."))
            elif node.module.startswith(runtime_root.name + "."):
                target = runtime_root.joinpath(*node.module.split(".")[1:])
            else:
                continue
            source = target.with_suffix(".py")
            if not source.is_file():
                source = target / "__init__.py"
            return source, matched.name
        return None

    def class_has(module: Path, name: str) -> bool:
        identity = module.resolve(), name
        if identity in visited:
            return False
        visited.add(identity)
        tree = read(module)
        if tree is None:
            return False
        cls = next((node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == name), None)
        if cls is None:
            return False
        if any(isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == operation for node in cls.body):
            return True
        return bases_have(module, tree, cls)

    def bases_have(module: Path, tree: ast.Module, cls: ast.ClassDef) -> bool:
        for base in cls.bases:
            target = base_target(module, tree, base.id) if isinstance(base, ast.Name) else None
            if target is not None and class_has(*target):
                return True
        return False

    tree = read(path)
    return tree is not None and any(bases_have(path, tree, node)
        for node in tree.body if isinstance(node, ast.ClassDef))


def _is_family_adapter_import(imported_path: str) -> bool:
    """Whether an import names only the reviewed family adapter surface."""

    compact = " ".join(imported_path.split())
    family_root = r"(?:(?:boundary|exec)::)?model_families"
    if re.fullmatch(
        rf"{family_root}\s+as\s+[A-Za-z_][A-Za-z0-9_]*", compact
    ):
        return True
    if re.fullmatch(rf"{family_root}::\{{.*\}}", compact):
        return True
    return bool(
        re.fullmatch(
            rf"{family_root}::[A-Za-z_][A-Za-z0-9_]*"
            r"\s+as\s+[A-Za-z_][A-Za-z0-9_]*",
            compact,
        )
    )


def _declared_physical_weight_families(
    text: str,
    families: set[str],
) -> list[tuple[str, set[str]]]:
    """Return family-owned physical declarations found in neutral source."""

    declarations: list[tuple[str, set[str]]] = []
    for match in PHYSICAL_WEIGHT_DECLARATION.finditer(text):
        normalized = re.sub(r"[^a-z0-9]", "", match.group("name").lower())
        owners = {
            family
            for family in families
            if re.sub(r"[^a-z0-9]", "", family.lower()) in normalized
        }
        if owners:
            declarations.append((match.group("name"), owners))
    return declarations


def _matches(path: str, text: str, pattern: re.Pattern[str]) -> list[str]:
    return [
        f"{path}:{line_number}: {match.group(0)}"
        for line_number, line in enumerate(text.splitlines(), start=1)
        for match in pattern.finditer(line)
    ]

OWNERSHIP_MANIFEST = "audit/model_architecture_ownership.json"
FAMILY_MODULE = re.compile(r"^\s*pub(?:\([^)]*\))?\s+mod\s+(\w+)\s*;", re.MULTILINE)
RUST_CONTRACTS = {
    "mod.rs": BOUNDARY_FAMILY_CONTRACT_OPERATIONS + ("lemma_common_layers_repr",),
    "weights.rs": ("bind_model_weights_perms", "common_layers_repr_of",
        "layer_weights_valid", "layer_weights_common_repr_of", "model_weights_bound"),
    "deployment.rs": ("checkpoint_valid", "init_qualified_runtime",
        "init_staged_runtime_for_tests", "load_checkpoint", "qualify_checkpoint",
        "reports_backend_qualified"),
}
PYTHON_CONTRACTS = {
    "physical.py": ("validate_model_weights_permission_contract",
        "validate_model_weights_runtime_contract"),
    "loader.py": ("inspect_text_checkpoint", "load_text_weights"),
    "deployment.py": ("scope", "scope_sha256", "bind_launches_to_proof_cases",
        "validate_candidate", "seal_candidate", "load_bundle", "validate_runtime_binding"),
    "profile.py": ("model_config", "model_profile_for_config",
        "model_profile_for_name", "model_profiles", "scope", "scope_sha256"),
    "runtime.py": ("config_for_profile", "config_from_bundle", "load_runtime",
        "load_qualified_runtime", "report", "model_config", "runtime_config",
        "verified_for", "static_launch_config", "kernel_entrypoint"),
}


def _rust(path: Path) -> str:
    return code_only(path.read_text())


def _modules(root: Path, relative: str, expected: set[str], errors: list[str]):
    directory = root / relative
    module = directory / "mod.rs"
    if not module.is_file():
        errors.append(f"missing adapter module: {relative}/mod.rs")
        return
    declared = FAMILY_MODULE.findall(_rust(module))
    actual = {path.name for path in directory.iterdir()
              if path.is_dir() and (path / "mod.rs").is_file()}
    if len(declared) != len(set(declared)) or set(declared) != expected or actual != expected:
        errors.append(f"adapter set differs at {relative}: expected={sorted(expected)}, "
                      f"declared={declared}, actual={sorted(actual)}")


def _rust_contract(root: Path, relative: str, operations, errors):
    path = root / relative
    if not path.is_file():
        errors.append(f"missing adapter: {relative}")
        return
    for operation in operations:
        # The compiler resolves reexports outside this directory. This audit
        # checks the named API, not a second Rust module resolver.
        reexport = re.search(rf"\bpub\s+use\s+[^;]*\b{re.escape(operation)}\b[^;]*;", _rust(path))
        if _direct_operation_signature(_rust(path), operation) is None and not reexport:
            errors.append(f"missing adapter operation: {relative}: {operation}")


def _check_adapters(root, families, compositions, errors):
    for layer in ("boundary", "exec"):
        _modules(root, f"src/{layer}/model_families", families, errors)
    proof_root = "src/proof/model/families"
    _modules(root, proof_root, set(compositions.values()), errors)
    for composition in set(compositions.values()):
        _rust_contract(root, f"{proof_root}/{composition}/mod.rs",
                       FAMILY_CONTRACT_OPERATIONS, errors)
    for family in families:
        for file, operations in RUST_CONTRACTS.items():
            _rust_contract(root, f"src/boundary/model_families/{family}/{file}", operations, errors)

    # These operations have the same type shape after erasing family spelling.
    # Native checkpoint/weight records deliberately have different signatures.
    for file, operations in (
        ("mod.rs", RUST_CONTRACTS["mod.rs"]),
        ("weights.rs", ("common_layers_repr_of",)),
    ):
        for operation in operations:
            signatures = set()
            for family in families:
                path = root / f"src/boundary/model_families/{family}/{file}"
                if path.is_file():
                    signature = _operation_signature(path, path.parent, operation)
                    if signature is not None:
                        signatures.add(_normalize_family_signature(signature, families))
            if len(signatures) > 1:
                errors.append(f"adapter signature differs: {file}: {operation}")

    _rust_contract(root, COMMON_DEPLOYMENT_ASSEMBLER, COMMON_DEPLOYMENT_OPERATIONS, errors)
    for family in families:
        path = root / f"src/boundary/model_families/{family}/deployment.rs"
        if not path.is_file():
            continue
        text = _rust(path)
        aliases = re.findall(r"\buse\s+crate::boundary::model_deployment\s+as\s+(\w+)\s*;", text)
        for operation in COMMON_DEPLOYMENT_OPERATIONS:
            if not any(re.search(rf"\b{alias}::{operation}\s*\(", text) for alias in aliases):
                errors.append(f"family deployment bypasses common assembly: {family}: {operation}")


def _check_runtime(root, families, pending, errors):
    runtime_root = root / "python/vosti_kernels"
    directory = runtime_root / "model_families"
    # Static imports may use a family's own helpers, but not another adapter's
    # internals. The common runtime selects adapters through its closed registry.
    for path in runtime_root.rglob("*.py"):
        relative = path.relative_to(runtime_root)
        owner = relative.parts[1] if relative.parts[0] == "model_families" and len(relative.parts) > 2 else None
        package = ".".join(("vosti_kernels", *relative.parts[:-1]))
        for node in ast.walk(ast.parse(path.read_text())):
            imports = []
            if isinstance(node, ast.Import):
                imports = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                module = "." * node.level + (node.module or "")
                module = resolve_name(module, package) if node.level else module
                imports = [module, *(module + "." + alias.name for alias in node.names)]
            for module in imports:
                parts = module.split(".")
                if (parts[:2] == ["vosti_kernels", "model_families"]
                        and len(parts) > 2 and parts[2] in families | pending
                        and parts[2] != owner):
                    errors.append(f"runtime source imports another family: {relative}: {module}")
    actual = {path.name for path in directory.iterdir()
              if path.is_dir() and path.name != "__pycache__"}
    if actual != families | pending:
        errors.append(f"runtime family set differs: {sorted(actual)}")
    for family in families:
        for file, operations in PYTHON_CONTRACTS.items():
            path = directory / family / file
            if not path.is_file():
                errors.append(f"missing runtime adapter: {family}/{file}")
                continue
            text = path.read_text()
            tree = ast.parse(text)
            declared = {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}
            declared.update(method.name for node in tree.body
                if isinstance(node, ast.ClassDef) and node.name in {"Runtime", "QualifiedRuntime"}
                for method in node.body if isinstance(method, ast.FunctionDef))
            for operation in operations:
                if file == "deployment.py":
                    found = _python_deployment_binding(text, operation)
                else:
                    found = operation in declared or (
                        file == "runtime.py"
                        and _python_inherited_operation(path, runtime_root, operation))
                if not found:
                    errors.append(f"missing runtime operation: {family}/{file}: {operation}")
        scope = json.loads((directory / family / "scope.json").read_text())
        profiles = scope.get("model_profiles")
        if (scope.get("engine_reachable") is not False
                or scope.get("status") != "qualification_scope"
                or not isinstance(profiles, list) or not profiles
                or any(not isinstance(p, dict) or set(p) != {"model", "launches"}
                    or not isinstance(p["model"], dict) or not p["model"]
                    or not isinstance(p["launches"], list) or not p["launches"] for p in profiles)):
            errors.append(f"invalid runtime qualification scope: {family}")
    # Required APIs are checked above. Numerical/schema validity and admission
    # behavior have their own profile, loader and deployment tests.


def _check_certificates(root, interfaces, errors):
    directory = root / "src/boundary/backend_certificates"
    expected = {entry["generated"] for entry in interfaces.values()}
    expected |= {"src/boundary/backend_certificates/mod.rs",
                 "src/boundary/backend_certificates/support.rs"}
    actual = {path.relative_to(root).as_posix() for path in directory.rglob("*.rs")}
    if actual != expected:
        errors.append("backend certificate inventory is not closed")
    module = directory / "mod.rs"
    if module.is_file():
        declared = FAMILY_MODULE.findall(_rust(module))
        expected_modules = {"support"} | {Path(e["generated"]).stem for e in interfaces.values()}
        if len(declared) != len(set(declared)) or set(declared) != expected_modules:
            errors.append("backend certificate module declarations differ from inventory")
    support = directory / "support.rs"
    if support.is_file():
        source = support.read_text()
        if "@kernel-import-begin" in source or "external_body" in _rust(support):
            errors.append("backend certificate support must not import a trusted theorem")
    for entry in interfaces.values():
        for relative in (entry["manifest"], entry["generator"],
                         *(m["path"] for m in entry["consumer_modules"])):
            if not (root / relative).is_file():
                errors.append(f"missing shared kernel interface artifact: {relative}")
    # Independent receipt/theory/consumer validation remains in claim_ledger.
    rectangular_operators_from_source(root)


def _check_source_boundaries(root, families, errors):
    for source in sorted((root / "src").rglob("*.rs")):
        relative = source.relative_to(root).as_posix()
        text = _rust(source)
        parts = source.relative_to(root / "src").parts
        owner = (parts[2] if len(parts) > 3 and parts[0] in {"boundary", "exec"}
                 and parts[1] == "model_families" and parts[2] in families else None)
        composition = relative.startswith("src/proof/model/families/")
        if owner is None:
            for declaration, _ in _declared_physical_weight_families(text, families):
                errors.append(f"family physical weight declaration outside owned boundary: "
                              f"{relative}: {declaration}")
        for match in CRATE_IMPORT.finditer(text):
            imported = match.group("path")
            named = {family for family in families if family in imported.lower()}
            if owner and named - {owner}:
                errors.append(f"model family imports another family implementation: {relative}: {imported}")
            if not owner and named:
                if relative in CONFIGURATION_DATA_IMPORT_FILES and _is_configuration_data_import(imported, families):
                    continue
                if relative in GENERIC_PHYSICAL_DISPATCH_FILES | REVIEWED_OPTIONAL_CAPABILITY_DISPATCH_FILES:
                    continue
                if relative in GENERIC_PROOF_DISPATCH_FILES and _is_family_adapter_import(imported):
                    continue
                errors.append(f"family-neutral source imports a family implementation: {relative}: {imported}")
        if not owner and not composition and relative not in (
                PAYLOAD_VARIANT_FILES | REVIEWED_OPTIONAL_CAPABILITY_DISPATCH_FILES):
            errors.extend(f"architecture payload dispatch outside closed boundary: {item}"
                          for item in _matches(relative, text, PAYLOAD_VARIANT))
        if not owner and relative not in EXEC_VARIANT_FILES | REVIEWED_OPTIONAL_CAPABILITY_DISPATCH_FILES:
            errors.extend(f"physical model dispatch outside executable boundary: {item}"
                          for item in _matches(relative, text, EXEC_VARIANT))
        if relative == "src/exec/engine.rs":
            for pattern in (ENGINE_ARCHITECTURE, ENGINE_RUNTIME):
                errors.extend(f"generic Engine contains a concrete family case: {item}"
                              for item in _matches(relative, text, pattern))


def architecture_boundary_errors(root: Path = ROOT) -> list[str]:
    """Audit shared interfaces and dispatch; compiler/proof gates check behavior."""
    errors = []
    try:
        manifest = json.loads((root / OWNERSHIP_MANIFEST).read_text())
        qualification, interfaces = validate_registry(manifest, root)
        families = set(manifest["families"])
        pending = set(manifest.get("pre_admission_kernel_families", []))
        compositions = manifest.get("family_compositions")
        if (not isinstance(compositions, dict) or set(compositions) != families
                or not all(isinstance(v, str) and re.fullmatch(r"[a-z][a-z0-9_]*", v)
                           for v in compositions.values())):
            return ["family_compositions must map every family to a proof composition"]
        _check_adapters(root, families, compositions, errors)
        _check_runtime(root, families, pending, errors)
        _check_certificates(root, interfaces, errors)
        _check_source_boundaries(root, families, errors)
    except (OSError, ValueError, SyntaxError, KeyError, TypeError) as error:
        errors.append(f"invalid architecture inventory: {error}")
    return errors


def main() -> int:
    errors = architecture_boundary_errors()
    if errors:
        print("model architecture boundary check failed:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 1
    print("Model architecture PASS: family interfaces, shared certificates and closed dispatch")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
