"""Mutation admission binds both qualification receipts and the checked import."""

from copy import deepcopy
from pathlib import Path

import pytest

from scripts.deployment.common import qualify_proof_case
from scripts.audit.kernel_interface_audit import (
    canonical_digest, rectangular_inventory_from_source, validate_mutation_interface,
)
from scripts.verification.kernel_interface_registry import load_registry
from scripts.verification.verify_mutation_interfaces import selected_cases
from vosti_kernels.kernel_interfaces import load_mutation_interface, validate_mutation_interface_case


ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def cases():
    return [qualify_proof_case(c, constants, ("bfloat16", "float32")).receipt
            for c, constants, _ in selected_cases()]


def test_every_deployed_scatter_geometry_is_admitted(cases):
    manifest = load_mutation_interface()
    assert len(cases) == len(manifest["inventory_contributions"]) == 4
    for case in cases:
        validate_mutation_interface_case(case, manifest)


@pytest.mark.parametrize("change", ["effect_digest", "batch_digest", "kind", "specialization", "source"])
def test_wrong_qualification_is_rejected(cases, change):
    case = deepcopy(cases[0])
    if change == "effect_digest":
        case["semantic_proof"]["contract_digest"] = "0" * 64
    elif change == "batch_digest":
        case["structural_contract_digest"] = "0" * 64
    elif change == "kind":
        case["semantic_proof"]["kind"] = "batch_invariance"
    elif change == "specialization":
        case["specialization"]["BLOCK_M"] += 1
    else:
        case["source"] = "another_kernel.py"
    with pytest.raises(ValueError, match="mutation"):
        validate_mutation_interface_case(case, load_mutation_interface())


def audit(manifest):
    qualification, registry = load_registry()
    return validate_mutation_interface("kv_store", registry["kv_store"], manifest,
        canonical_digest(manifest), rectangular_inventory_from_source(qualification, ROOT,
            "store_kv_cache.py", "store_cache_kernel"), ROOT)


def test_independent_audit_has_only_consumed_effect_imports():
    records = audit(load_mutation_interface())
    assert len(records) == 4
    assert all(r["theory"]["proof_kind"] == "exact_effect" for r in records.values())
    assert all(len(r["consumers"]) == 2 for r in records.values())


@pytest.mark.parametrize("change", ["coverage", "identity", "geometry", "effect_kind", "state_relation",
                                    "adapter", "dispatch", "extra_goal", "interpretation"])
def test_mutation_audit_rejects_structural_drift_even_with_updated_pin(change):
    manifest = load_mutation_interface()
    entry = manifest["interfaces"][0]
    raw = entry["raw"]
    goal = raw["implementations"][0]["standalone_contracts"][0]
    if change == "coverage":
        manifest["inventory_contributions"].pop()
    elif change == "identity":
        raw["implementations"][0]["execution_identity"] = "0" * 64
    elif change == "geometry":
        entry["geometry"]["KVD"] += 1
    elif change == "effect_kind":
        goal["proof_kind"] = "batch_invariance"
    elif change == "state_relation":
        goal["state_relation"] = "left_right"
    elif change == "adapter":
        entry["checked_adapter_sha256"] = "0" * 64
    elif change == "dispatch":
        manifest["checked_dispatch_sha256"] = "0" * 64
    elif change == "extra_goal":
        raw["implementations"][0]["standalone_contracts"].append(deepcopy(goal))
    else:
        raw["interpretation"] = "equal_across_all_configurations"
    with pytest.raises(ValueError, match="invalid kernel interface audit"):
        audit(manifest)
