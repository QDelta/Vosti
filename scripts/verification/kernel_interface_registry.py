"""Reviewed ownership of shared generated interfaces and family inventories.

An interface is owned once, by its kernel generator, not by a model family.
Family entries only select qualification inventories. This registry is not a
proof receipt: source freshness and exact implementation coverage are checked
by the corresponding generator and admission validator.
"""

from pathlib import Path
import ast
import json
import re


ROOT = Path(__file__).resolve().parents[2]
OWNERSHIP = "audit/model_architecture_ownership.json"
CERTIFICATE_ROOT = "src/boundary/backend_certificates"


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _path(value, root, *, prefix, suffix):
    _require(isinstance(value, str), "kernel registry path must be a string")
    path = Path(value)
    _require(not path.is_absolute() and ".." not in path.parts
             and value == path.as_posix() and value.startswith(prefix + "/")
             and value.endswith(suffix)
             and (root / path).resolve().is_relative_to(root.resolve()),
             f"kernel registry path escapes its reviewed root: {value!r}")
    return path


def validate_registry(ownership, root=ROOT):
    _require(isinstance(ownership, dict) and ownership.get("schema_version") == 10,
             "kernel interface ownership requires schema_version 10")
    _require("certificate_imports" not in ownership,
             "family-owned certificate catalogs have been retired")
    families = ownership.get("families")
    _require(isinstance(families, list) and bool(families)
             and all(isinstance(f, str) and re.fullmatch(r"[a-z][a-z0-9_]*", f) for f in families)
             and len(set(families)) == len(families), "invalid kernel registry families")
    qualification = ownership.get("kernel_qualification")
    pending = ownership.get("pre_admission_kernel_families", [])
    _require(isinstance(pending, list)
             and all(isinstance(f, str) and re.fullmatch(r"[a-z][a-z0-9_]*", f) for f in pending)
             and len(set(pending)) == len(pending) and not set(pending).intersection(families),
             "invalid or already-admitted pre-admission kernel families")
    _require(isinstance(qualification, dict) and set(qualification) == set(families) | set(pending),
             "kernel qualification inventories must cover every family exactly once")
    for family, entry in qualification.items():
        _require(isinstance(entry, dict) and set(entry) == {"scope", "inventory"},
                 f"invalid kernel qualification descriptor: {family}")
        _require(entry["scope"] == f"python/vosti_kernels/model_families/{family}/scope.json",
                 f"kernel qualification scope names another family: {family}")
        _path(entry["scope"], root, prefix="python/vosti_kernels/model_families", suffix=".json")
        _require(entry["inventory"] in {"contract_catalog", "profile_catalog"},
                 f"unknown kernel qualification inventory: {family}")
    interfaces = ownership.get("kernel_interfaces")
    _require(isinstance(interfaces, dict) and bool(interfaces),
             "shared kernel interface registry must be nonempty")
    generated, manifests = set(), set()
    for name, entry in interfaces.items():
        _require(isinstance(name, str) and re.fullmatch(r"[a-z][a-z0-9_]*", name),
                 "invalid shared kernel interface name")
        _require(isinstance(entry, dict) and set(entry) == {
            "generated", "manifest", "generator", "consumer_modules"},
            f"invalid shared kernel interface descriptor: {name}")
        path = _path(entry["generated"], root, prefix=CERTIFICATE_ROOT, suffix=".rs")
        _require(path.parent.as_posix() == CERTIFICATE_ROOT and path.stem not in {"mod", "support"},
                 "shared kernel interface must live directly under the neutral certificate root")
        _path(entry["manifest"], root, prefix="audit", suffix=".json")
        _path(entry["generator"], root, prefix="scripts", suffix=".py")
        # A generic producer can own several different interfaces. Generated
        # artifacts must still have unique ownership; producer code need not.
        for value, seen in ((entry["generated"], generated), (entry["manifest"], manifests)):
            _require(value not in seen, f"duplicate shared kernel interface artifact: {value}")
            seen.add(value)
        modules = entry["consumer_modules"]
        _require(isinstance(modules, list) and bool(modules),
                 f"shared kernel interface has no checked consumers: {name}")
        seen = set()
        for module in modules:
            _require(isinstance(module, dict) and set(module) == {"path", "module"},
                     f"invalid shared kernel interface consumer: {name}")
            path = _path(module["path"], root, prefix="src", suffix=".rs")
            _require(isinstance(module["module"], str)
                     and re.fullmatch(r"[a-z][a-z0-9_]*(::[a-z][a-z0-9_]*)+", module["module"]),
                     f"invalid shared kernel interface module: {name}")
            _require(module["path"] not in generated and module["path"] not in seen,
                     f"duplicate or self-referential kernel interface consumer: {name}")
            seen.add(module["path"])
    _require(not generated.intersection(m["path"] for e in interfaces.values()
                                        for m in e["consumer_modules"]),
             "generated interfaces must not stand in for checked external consumers")
    return qualification, interfaces


def load_registry(root=ROOT):
    ownership = json.loads((root / OWNERSHIP).read_text())
    return validate_registry(ownership, root)


def rectangular_operators_from_source(root=ROOT):
    """Read the shared routing declaration without importing Python kernels.

    This is identity routing, not a theorem or shape binding. The independent
    auditor still checks each interface against its exact source inventory.
    """
    tree = ast.parse((root / "python/vosti_kernels/kernel_interfaces.py").read_text())
    declarations = [node.value for node in tree.body
                    if isinstance(node, ast.Assign) and len(node.targets) == 1
                    and isinstance(node.targets[0], ast.Name)
                    and node.targets[0].id == "RECTANGULAR_KERNEL_INTERFACES"]
    _require(len(declarations) == 1 and isinstance(declarations[0], ast.Dict),
             "rectangular interface routing must be one literal dictionary")
    declaration = declarations[0]
    pairs = [(ast.literal_eval(key), ast.literal_eval(value))
             for key, value in zip(declaration.keys, declaration.values)]
    sources, names = set(), set()
    for source, name in pairs:
        _require(isinstance(source, tuple) and len(source) == 2
                 and all(isinstance(s, str) for s in source)
                 and re.fullmatch(r"[a-z][a-z0-9_]*\.py", source[0])
                 and re.fullmatch(r"[a-z][a-z0-9_]*", source[1])
                 and isinstance(name, str) and re.fullmatch(r"[a-z][a-z0-9_]*", name),
                 "invalid rectangular interface routing entry")
        _require(source not in sources and name not in names,
                 "duplicate rectangular interface routing entry")
        sources.add(source)
        names.add(name)
    return {name: source for source, name in pairs}
