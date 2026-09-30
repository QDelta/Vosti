"""Independent identity/linkage audit of shared generated kernel interfaces.

The source verifier establishes the imported theorems in a separate gate. This
audit checks reviewed receipt identity, complete static inventory coverage,
generated bodies, and actual checked consumers; it never reinterprets an
analyzer condition or authenticates a proof merely from a JSON shape.
"""

import hashlib
import ast
import json

from pathlib import Path
import re

from scripts.audit.tcb import rust_module_name, scan_rust_item_records
from vosti_kernels.model_profile import load_scope
from scripts.audit.rust_source import code_only
from scripts.verification.engine_kernel_bindings import required_kernel_goal_bindings


def require(condition, message):
    if not condition:
        raise ValueError(f"invalid kernel interface audit: {message}")


def canonical_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                      allow_nan=False).encode()).hexdigest()


def _sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def _case_identity(case):
    return canonical_digest({key: case[key] for key in ("source", "kernel", "constants")})


def _goal_inventory(cases, roles):
    """Expected labels come from source-inventory role bindings, not receipts."""
    result = {}
    for contract, constants, _ in cases:
        binding = required_kernel_goal_bindings(contract)
        require(set(roles) <= set(binding), "source inventory lacks required proof roles")
        result[_case_identity(dict(source=contract["source"], kernel=contract["kernel"], constants=constants))] = {
            "raw_" + binding[role] for role in roles}
    return result


def attention_inventory_from_source(qualification, root):
    """Read declarative profile/alternative coverage without importing kernels.

    The fixed page size is declared in the shared constants module; alternatives
    remain literal declarations in the kernel file. Check the import link too.
    No Python evaluation, source verifier, Torch, or GPU is needed by this gate.
    """
    tree = ast.parse((root / "kernels/triton_kernels/fattn_paged.py").read_text())
    shared = ast.parse((root / "kernels/triton_kernels/constants.py").read_text())
    imports = [node for node in tree.body if isinstance(node, ast.ImportFrom)
               and node.module == "triton_kernels.constants" and node.level == 0]
    require(len(imports) == 1 and any(alias.name == "PAGE_SIZE" and alias.asname is None
                                     for alias in imports[0].names),
            "attention must import the shared page size")
    require(not any(isinstance(node, ast.Name) and node.id == "PAGE_SIZE"
                    and isinstance(node.ctx, ast.Store) for node in ast.walk(tree)),
            "attention must not redefine the shared page size")
    literals = {}
    for module, names in ((tree, {"CONFIGS"}), (shared, {"PAGE_SIZE"})):
        for node in module.body:
            if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                name = node.targets[0].id
                if name in names:
                    require(name not in literals, "duplicate attention inventory declaration")
                    literals[name] = ast.literal_eval(node.value)
    require(set(literals) == {"PAGE_SIZE", "CONFIGS"} and type(literals["PAGE_SIZE"]) is int
            and literals["PAGE_SIZE"] > 0, "missing literal attention inventory")
    page_size = literals["PAGE_SIZE"]
    selected = {}
    def constants(config):
        return {key: value for key, value in config.items() if key not in {"num_warps", "num_stages"}}
    def add(contract, config, family):
        values = {**constants(config), "PAGE_BLOCK_SIZE": page_size}
        key = (contract["source"], contract["kernel"], json.dumps(values, sort_keys=True, allow_nan=False))
        if key in selected:
            require(selected[key][0]["source_sha256"] == contract["source_sha256"], "contributor source disagreement")
        selected.setdefault(key, (contract, values, set()))[2].add(family)
    for family, entry in sorted(qualification.items()):
        scope = load_scope(root / entry["scope"])
        contracts = {c["wrapper"]: c for c in scope["kernel_contracts"]}
        for profile in scope["model_profiles"]:
            for launch in profile["launches"]:
                contract = contracts[launch["wrapper"]]
                if contract["evidence"] != "conditional_relational_certificate":
                    continue
                add(contract, launch["config"], family)
                if entry["inventory"] == "contract_catalog":
                    require(contract["source"] == "fattn_paged.py", "unknown contract-catalog attention inventory")
                    for alternative in literals["CONFIGS"]:
                        require(type(alternative["BLOCK_N"]) is int and alternative["BLOCK_N"] > 0,
                                "invalid alternative attention tile")
                        if page_size % alternative["BLOCK_N"] == 0:
                            add(contract, {**alternative, "D_HEAD": launch["config"]["D_HEAD"]}, family)
    return [selected[key] for key in sorted(selected)]


def _components(body):
    """Read the renderer's explicit module grammar, not arbitrary Rust syntax."""
    header = "// @generated by scripts/verification/attention_interface_catalog.py; DO NOT EDIT.\n"
    require(body.startswith(header), "unrecognized catalog header")
    prefix, separator, dispatch = body[len(header):].partition(
        "// Geometry dispatch is shared across architectures.")
    require(bool(separator) and bool(dispatch), "missing checked geometry dispatcher")
    pattern = re.compile(r"^pub mod ([a-z][a-z0-9_]*) \{\nuse vstd::prelude::\*;\nverus! \{\n", re.M)
    starts = list(pattern.finditer(prefix))
    require(starts and starts[0].start() == 0, "missing generated raw modules")
    parts = {}
    for index, start in enumerate(starts):
        end = starts[index + 1].start() if index + 1 < len(starts) else len(prefix)
        module = prefix[start.end():end]
        # Adjacent modules are separated by the catalog's single join newline.
        trailer = "\n}\n\n" if index + 1 < len(starts) else "\n}\n"
        require(module.endswith(trailer), "unrecognized generated module trailer")
        module = module[:-len(trailer)]
        marker = "// Checked consumer of a generated raw kernel contract."
        raw, found, adapter = module.partition("\n}\n" + marker)
        require(bool(found) and start[1] not in parts, "missing or duplicate checked adapter")
        parts[start[1]] = (raw, marker + adapter)
    return parts


def rectangular_inventory_from_source(qualification, root, source_name, kernel_name):
    """Independent static profile inventory; no import or evaluation of kernels."""
    selected = {}
    for family, entry in sorted(qualification.items()):
        scope = load_scope(root / entry["scope"])
        contracts = {c["wrapper"]: c for c in scope["kernel_contracts"]}
        for profile in scope["model_profiles"]:
            for launch in profile["launches"]:
                contract = contracts[launch["wrapper"]]
                if (contract["source"], contract["kernel"]) != (source_name, kernel_name):
                    continue
                constants = {k: v for k, v in launch["config"].items()
                             if k not in {"num_warps", "num_stages"}}
                key = (source_name, kernel_name, json.dumps(constants, sort_keys=True, allow_nan=False))
                if key in selected:
                    require(selected[key][0]["source_sha256"] == contract["source_sha256"],
                            "contributor source disagreement")
                selected.setdefault(key, (contract, constants, set()))[2].add(family)
    return [selected[key] for key in sorted(selected)]


def validate_rectangular_interface(name, descriptor, manifest, expected_digest, cases, root):
    return _validate_row_interface(name, descriptor, manifest, expected_digest, cases, root, mutation=False)


def validate_mutation_interface(name, descriptor, manifest, expected_digest, cases, root):
    return _validate_row_interface(name, descriptor, manifest, expected_digest, cases, root, mutation=True)


def _validate_row_interface(name, descriptor, manifest, expected_digest, cases, root, *, mutation):
    require(canonical_digest(manifest) == expected_digest, "reviewed interface receipt differs")
    catalog_kind = ("geometry_dispatched_mutation_interface" if mutation
                    else "geometry_dispatched_rectangular_interfaces")
    catalog = manifest.get("kind") == catalog_kind
    extra = {"checked_dispatch_sha256", "dimensions", "interfaces"} if catalog else {"checked_adapter_sha256", "raw"}
    require(set(manifest) == {"schema_version", "kind", "generated_body_sha256",
            "inventory_contributions"} | extra
            and manifest["schema_version"] == 1 and manifest["kind"] in (
                {catalog_kind} if mutation else {"rectangular_kernel_interface", catalog_kind}),
            "unsupported row interface schema")
    path = root / descriptor["generated"]
    body = path.read_text()
    require(_sha(body) == manifest["generated_body_sha256"], "generated catalog body differs")
    if catalog:
        parts, dispatch = _rectangular_catalog_components(body, mutation=mutation)
        require(_sha(dispatch) == manifest["checked_dispatch_sha256"], "checked dispatcher body differs")
    else:
        require(body.startswith("// @generated by scripts/verification/verify_rectangular_interfaces.py; DO NOT EDIT.\n")
                and body.endswith("\n} // verus!\n"), "invalid rectangular source envelope")
        _, found, raw_and_adapter = body.partition("\nverus! {\n")
        require(bool(found), "missing rectangular Verus body")
        raw_body, found, adapter_body = raw_and_adapter.partition("\n// BEGIN CHECKED RECTANGULAR ADAPTER\n")
        require(bool(found), "missing checked rectangular adapter")
        adapter_body = adapter_body.removesuffix("\n} // verus!\n")
        parts = {"": (raw_body, adapter_body)}
    contributions = manifest["inventory_contributions"]
    expected = []
    for contract, constants, families in cases:
        source = (root / "kernels/triton_kernels" / contract["source"]).resolve()
        require(source.is_relative_to((root / "kernels/triton_kernels").resolve()), "source escapes kernel tree")
        require(_sha(source.read_text()) == contract["source_sha256"], "kernel source differs from qualified scope")
        expected.append(dict(source=contract["source"], kernel=contract["kernel"], constants=constants,
                             families=sorted(families)))
    receipt_fields = {"execution_identity", "structural_contract_digest"} if mutation else {"execution_identity"}
    require(isinstance(contributions, list) and all(isinstance(c, dict) and set(c) == {
        "source", "kernel", "constants", "families"} | receipt_fields for c in contributions),
        "invalid rectangular static contributions")
    if mutation:
        require(all(isinstance(c["structural_contract_digest"], str)
                    and re.fullmatch(r"[0-9a-f]{64}", c["structural_contract_digest"]) for c in contributions),
                "missing mutation batch qualification")
    require(canonical_digest([{k: v for k, v in c.items() if k not in receipt_fields}
                              for c in contributions]) == canonical_digest(expected),
            "static inventory coverage differs")
    by_execution = {c["execution_identity"]: c for c in contributions}
    require(len(by_execution) == len(contributions), "duplicate contribution execution")
    records, used = {}, set()
    interfaces = manifest["interfaces"] if catalog else [dict(module="", geometry={},
        raw=manifest["raw"], checked_adapter_sha256=manifest["checked_adapter_sha256"])]
    modules, geometries = set(), set()
    if catalog:
        dimensions = manifest["dimensions"]
        require(isinstance(dimensions, list) and all(isinstance(d, str) and re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", d)
                for d in dimensions) and dimensions == sorted(set(dimensions)), "invalid catalog dimensions")
    require(isinstance(interfaces, list) and interfaces, "empty geometry catalog")
    for entry in interfaces:
        require(isinstance(entry, dict) and set(entry) == {"module", "geometry", "raw", "checked_adapter_sha256"},
                "invalid rectangular geometry interface")
        module, geometry = entry["module"], entry["geometry"]
        require(isinstance(geometry, dict) and all(type(v) is int and v >= 0 for v in geometry.values()),
                "invalid geometry extents")
        if catalog:
            require(set(geometry) <= set(dimensions), "geometry keys are not catalog dimensions")
            expected_module = "geometry_" + ("_".join(f"{k}_{v}" for k, v in sorted(geometry.items())) or "dynamic")
            require(module == expected_module, "module differs from geometry")
        key = tuple(sorted(geometry.items()))
        require(module in parts and module not in modules and key not in geometries,
                "missing or duplicate geometry interface")
        modules.add(module)
        geometries.add(key)
        raw_body, adapter_body = parts[module]
        _validate_static_interface(entry["raw"], raw_body, adapter_body, entry["checked_adapter_sha256"],
            by_execution=by_execution, used=used, records=records, name=name, path=path, module=module,
            geometry=geometry, expected_goals=_goal_inventory(cases, ("effect",) if mutation else ("batch",)))
        if mutation:
            require(all(g.get("proof_kind") == "exact_effect" and g.get("state_relation") == "before_after"
                        for i in entry["raw"]["implementations"] for g in i["standalone_contracts"]),
                    "mutation import is not a before/after effect theorem")
    require(modules == set(parts), "unrecorded generated module")
    require(used == set(by_execution), "uncovered static contribution")
    actual = {i["identity"] for i in scan_rust_item_records(path) if i["category"] == "proof-trusted"}
    require(actual == set(records), "imported raw proof inventory is not closed")
    consumers = local_checked_consumers(path, set(records))
    require(all(consumers.values()), "raw import lacks a checked consumer")
    for identity, record in records.items():
        record["families"] = sorted(record["families"])
        record["consumers"] = consumers[identity]
    return records


def _rectangular_catalog_components(body, *, mutation=False):
    generator = "verify_mutation_interfaces" if mutation else "rectangular_interface_catalog"
    require(body.startswith(f"// @generated by scripts/verification/{generator}.py; DO NOT EDIT.\n"),
            "unrecognized rectangular catalog header")
    prefix, marker, dispatch = body.partition("\n// BEGIN CHECKED GEOMETRY DISPATCH\n")
    require(marker and dispatch.startswith("verus! {\n") and dispatch.endswith("} // verus!\n"),
            "missing checked rectangular dispatcher")
    starts = list(re.finditer(r"^pub mod ([A-Za-z_][A-Za-z_0-9]*) \{\nuse super::\*;\nverus! \{\n", prefix, re.M))
    require(starts, "missing rectangular geometry modules")
    parts = {}
    for index, start in enumerate(starts):
        end = starts[index + 1].start() if index + 1 < len(starts) else len(prefix)
        segment = prefix[start.end():end]
        trailer = "\n} // verus!\n}\n" + ("\n" if index + 1 < len(starts) else "")
        require(segment.endswith(trailer), "invalid rectangular geometry trailer")
        segment = segment[:-len(trailer)]
        kind = "MUTATION" if mutation else "RECTANGULAR"
        raw, marker, adapter = segment.partition(f"\n// BEGIN CHECKED {kind} ADAPTER\n")
        require(marker and start[1] not in parts, "missing or duplicate geometry adapter")
        parts[start[1]] = (raw, adapter)
    return parts, dispatch


def local_checked_consumers(path, proof_names):
    """Find complete direct calls to module-local imported proof functions.

    Generated adapters use local calls. Cross-module calls to an import are
    rejected here rather than guessed from an alias. External checked callers
    consume exported adapter lemmas, not the raw imports themselves.
    """
    lines = code_only(path.read_text()).splitlines()
    observed = {name: set() for name in proof_names}
    short_names = {name.rsplit("::", 1)[1] for name in proof_names}
    for item in scan_rust_item_records(path):
        code = "\n".join(lines[item["line"] - 1:item["end_line"]])
        for call in re.finditer(r"\b([A-Za-z_]\w*(?:::[A-Za-z_]\w*)*)\s*\(", code):
            name = call[1]
            if name.rsplit("::", 1)[-1] not in short_names or re.search(r"\bfn\s*$", code[:call.start()]):
                continue
            require("::" not in name, "raw proof call must use its own generated module")
            identity = item["module"] + "::" + name
            require(identity in observed, "raw proof call has no module-local import")
            require(item["category"] in {"proof", "exec-verified"},
                    "raw proof consumed by an unchecked item")
            observed[identity].add(item["identity"])
    return {name: sorted(consumers) for name, consumers in observed.items()}


def _validate_static_interface(raw, raw_body, adapter_body, adapter_sha, *,
                               by_execution, used, records, name, path, module,
                               geometry, expected_goals):
    """Shared exact-identity audit for attention and rectangular raw exports."""
    require(set(raw) == {"schema_version", "kind", "interpretation", "logical_body_sha256", "generated_body_sha256", "implementations"}
            and raw["schema_version"] == 1 and raw["kind"] == "static_implementation_interface"
            and raw["interpretation"] == "one_fixed_deployed_implementation", "invalid static raw interface")
    require(_sha(raw_body) == raw["generated_body_sha256"], "raw interface body differs")
    require(_sha("".join(raw_body.splitlines(keepends=True)[3:])) == raw["logical_body_sha256"],
            "raw logical body differs")
    require(_sha(adapter_body) == adapter_sha, "checked adapter body differs")
    require(raw["implementations"], "empty static implementation interface")
    for implementation in raw["implementations"]:
        require(set(implementation) == {"schema_version", "kind", "execution_identity", "execute_name", "side_type", "output_parameters", "standalone_contracts", "generated_body_sha256"}
                and implementation["schema_version"] == 1 and implementation["kind"] == "shared_execution",
                "invalid raw implementation")
        execution = implementation["execution_identity"]
        require(execution in by_execution and execution not in used, "uncovered or duplicate implementation")
        used.add(execution)
        contribution = by_execution[execution]
        for key, value in geometry.items():
            require(type(contribution["constants"].get(key)) is type(value)
                    and contribution["constants"][key] == value, "geometry specialization differs")
        goals = implementation["standalone_contracts"]
        required_goals = expected_goals.get(_case_identity(contribution))
        require(required_goals is not None and len(goals) == len(required_goals)
                and {g["symbol_prefix"] for g in goals} == required_goals, "raw goal coverage differs")
        for goal in goals:
            require(goal["schema_version"] == 3 and goal["kernel"] == contribution["kernel"], "raw goal source differs")
            identity = "::".join(part for part in (rust_module_name(path), module, goal["certificate_name"]) if part)
            theory = {k: v for k, v in goal.items() if k not in {"raw_contract_digest", "generated_body_sha256"}}
            record = records.setdefault(identity, dict(interface=name, module=module, theory=theory,
                implementations=[], families=set()))
            require(record["theory"] == theory, "static implementations expose different interface records")
            record["families"].update(contribution["families"])
            record["implementations"].append(dict(**contribution, raw_contract_digest=goal["raw_contract_digest"]))


def validate_attention_interface(name, descriptor, manifest, expected_digest, cases, root):
    """Close a geometry-dispatched catalog against separately reviewed receipts.

    expected_digest pins the complete manifest, including all exact proof
    digests and separate implementation identities. It does not stand in for
    running the source verifier or checking the Verus adapter bodies.
    """
    require(canonical_digest(manifest) == expected_digest, "reviewed interface receipt differs")
    require(set(manifest) == {"schema_version", "kind", "generated_body_sha256", "interfaces", "inventory_contributions"}
            and manifest["schema_version"] == 1
            and manifest["kind"] == "geometry_dispatched_attention_interfaces", "unsupported catalog schema")
    path = root / descriptor["generated"]
    body = path.read_text()
    require(_sha(body) == manifest["generated_body_sha256"], "generated catalog body differs")
    parts = _components(body)
    contributions = manifest["inventory_contributions"]
    require(isinstance(contributions, list), "invalid static contributions")
    expected = []
    for contract, constants, families in cases:
        source = (root / "kernels/triton_kernels" / contract["source"]).resolve()
        require(source.is_relative_to((root / "kernels/triton_kernels").resolve()), "source escapes kernel tree")
        require(_sha(source.read_text()) == contract["source_sha256"], "kernel source differs from qualified scope")
        expected.append(dict(source=contract["source"], kernel=contract["kernel"], constants=constants, families=sorted(families)))
    for item in contributions:
        require(isinstance(item, dict) and set(item) == {"source", "kernel", "constants", "families", "execution_identity"},
                "invalid static contribution")
    require(canonical_digest([{k: v for k, v in item.items() if k != "execution_identity"}
                               for item in contributions]) == canonical_digest(expected),
            "static inventory coverage differs")
    by_execution = {item["execution_identity"]: item for item in contributions}
    require(len(by_execution) == len(contributions), "duplicate contribution execution")
    records, used, modules = {}, set(), set()
    for interface in manifest["interfaces"]:
        require(set(interface) == {"head_dim", "sliding_window", "raw", "checked_adapter_sha256"}
                and type(interface["head_dim"]) is int and interface["head_dim"] > 0
                and type(interface["sliding_window"]) is bool, "invalid geometry interface")
        module = f'{"swa" if interface["sliding_window"] else "full"}_d{interface["head_dim"]}'
        require(module in parts and module not in modules, "missing or duplicate geometry module")
        modules.add(module)
        raw_body, adapter_body = parts[module]
        raw = interface["raw"]
        _validate_static_interface(raw, raw_body, adapter_body, interface["checked_adapter_sha256"],
            by_execution=by_execution, used=used, records=records, name=name, path=path,
            module=module, geometry={"D_HEAD": interface["head_dim"]},
            expected_goals=_goal_inventory(cases, ("batch", "selected")))
    require(used == set(by_execution), "uncovered static contribution")
    require(modules == set(parts), "unrecorded generated module")
    actual = {item["identity"] for item in scan_rust_item_records(path) if item["category"] == "proof-trusted"}
    require(actual == set(records), "imported raw proof inventory is not closed")
    consumers = local_checked_consumers(path, set(records))
    require(all(consumers.values()), "raw import lacks a checked consumer")
    for identity, record in records.items():
        record["families"] = sorted(record["families"])
        record["consumers"] = consumers[identity]
    return records
