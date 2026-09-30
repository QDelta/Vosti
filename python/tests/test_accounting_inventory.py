"""TCB integration tables use the shared inventory, not a family-specific list."""

from scripts.audit.tcb import integration_source_groups
from scripts.effort.account import ROOT, script_dependencies, source_inventory


def test_new_integration_helpers_are_counted_without_another_allowlist():
    inventory = {"scripts/verification/engine_kernel_bindings.py", "scripts/deployment/dense_kernel_scope.py",
                 "scripts/new_representation_helper.py", "kernels/ir/annotations.py",
                 "src/exec/engine.rs"}
    bridge, qualification = integration_source_groups(inventory, [])
    assert set(bridge) & inventory == {p for p in inventory if p.startswith("scripts/")}
    assert not qualification


def test_families_are_partitioned_symmetrically_without_double_counting():
    inventory, scopes = set(), []
    for family in ("qwen3", "gemma3", "gemma4", "llama3", "new_family"):
        inventory.update({f"scripts/deployment/model_families/{family}.py",
                          f"scripts/deployment/model_families/{family}_scope.py"})
        scopes.append(f"python/vosti_kernels/model_families/{family}/scope.json")
    inventory.update({"scripts/prepare_deployment.py", "scripts/deployment/common.py", "kernels/backend/probes.py"})
    bridge, qualification = integration_source_groups(inventory, scopes)
    assert set(scopes) <= set(bridge)
    assert set(bridge).isdisjoint(qualification)
    assert set(bridge) | set(qualification) >= inventory | set(scopes)
    assert len(qualification) == 8
    assert bridge == sorted(set(bridge))
    assert qualification == sorted(set(qualification))


def test_script_import_closure_handles_package_imports_cycles_and_unused_names(tmp_path):
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    sources = {
        "entry.py": "from scripts import helper\nfrom . import relative\nimport external\n",
        "helper.py": "from .leaf import name\n",
        "relative.py": "import scripts.helper\n",
        "leaf.py": "from scripts import entry\n",
        "unused.py": "raise RuntimeError('must not be executed')\n",
    }
    for name, source in sources.items():
        (scripts / name).write_text(source)
    files = {"scripts/" + name for name in sources}
    assert script_dependencies(tmp_path, files, ["scripts/entry.py"]) == files - {"scripts/unused.py"}


def test_shared_inventory_includes_dynamically_loaded_family_policies():
    inventory, _ = source_inventory(ROOT)
    families = ROOT / "scripts/deployment/model_families"
    policies = {str(p.relative_to(ROOT)) for p in families.glob("*_scope.py")}
    assert policies
    assert policies <= inventory.keys()
    assert {"scripts/deployment/dense_kernel_scope.py", "scripts/verification/engine_kernel_bindings.py"} <= inventory.keys()
    builders = {str(p.relative_to(ROOT)) for p in families.glob("*.py") if not p.stem.endswith("_scope")}
    assert builders
    assert {"scripts/prepare_deployment.py", *builders} <= inventory.keys()


def test_script_import_closure_follows_nested_qualified_modules(tmp_path):
    sources = {
        "scripts/verification/entry.py": "from scripts.deployment import common\n",
        "scripts/deployment/common.py": "import scripts.audit.registry as registry\n",
        "scripts/audit/registry.py": "from scripts.verification.entry import check\n",
        "scripts/audit/unused.py": "raise RuntimeError('must not be executed')\n",
    }
    for name, source in sources.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source)
    assert script_dependencies(tmp_path, set(sources), ["scripts/verification/entry.py"]) == (
        sources.keys() - {"scripts/audit/unused.py"})


def test_shared_verifier_and_its_generators_remain_in_the_accounted_closure():
    inventory, _ = source_inventory(ROOT)
    assert {
        "scripts/verification/verify_kernel_contracts.py",
        "scripts/verification/verify_attention_interfaces.py",
        "scripts/verification/verify_rectangular_interfaces.py",
        "scripts/verification/verify_mutation_interfaces.py",
        "scripts/verification/kernel_contract_codegen.py",
        "scripts/verification/kernel_interface_codegen.py",
    } <= inventory.keys()


def test_nested_effort_modules_remain_in_the_tcb_import_closure():
    from scripts.effort.account import tracked
    files = tracked(ROOT)
    closure = script_dependencies(ROOT, files, ["scripts/audit/tcb.py"])
    assert {
        "scripts/effort/account.py", "scripts/effort/dependencies.py",
        "scripts/effort/specification.py", "scripts/effort/purposes.py",
    } <= closure
    assert "python/tests/test_effort.py" not in closure
