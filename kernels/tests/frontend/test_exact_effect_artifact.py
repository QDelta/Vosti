"""Exact-effect imports retain source, theorem, and complete proof coverage."""

import copy
import json
from functools import cache

import pytest
import z3

from ir.exact_effect_artifact import (
    VerifiedExactEffectContract, _issue_exact_effect_contract, _plan_document,
)
from ir.exact_effects import prepare_exact_effect_plan, verify_exact_effects
from ir.relational_verifier import verify_annotations
from ir.verus_contract import render_verified_kernel_to_verus
from tests.analysis.test_exact_effects import annotated


@cache
def proved():
    report = verify_exact_effects(annotated(), "store_cache_kernel", {"KVD": 512, "BLOCK_M": 1})
    assert report.proved and report.verified_contract is not None
    return report.verified_contract


def test_round_trip_retains_typed_theorem_and_physical_premises():
    artifact = proved()
    data = artifact.to_data()
    assert VerifiedExactEffectContract.from_data(data) == artifact
    assert artifact.written_tensor_parameters() == ("cache",)
    assert data["plan"]["theorem"]["name"] == "exact_effect"
    assert '"node": "DTypeOf"' in json.dumps(data["plan"]["theorem"])
    assert '"node": "Before"' in json.dumps(data["plan"]["theorem"])
    assert '"node": "After"' in json.dumps(data["plan"]["theorem"])
    assert len(data["plan"]["external_obligations"]) == 3


def test_plan_and_validation_are_solver_free_and_fresh_name_independent(monkeypatch):
    data = proved().to_data()
    def forbidden(*args, **kwargs):
        raise AssertionError("artifact validation must not solve again")
    monkeypatch.setattr(z3, "Solver", forbidden)
    monkeypatch.setattr(z3, "Tactic", forbidden)
    first = prepare_exact_effect_plan(annotated(), "store_cache_kernel", {"KVD": 512, "BLOCK_M": 1})
    for _ in range(100):
        z3.FreshInt("unrelated")
    second = prepare_exact_effect_plan(annotated(), "store_cache_kernel", {"BLOCK_M": 1, "KVD": 512})
    assert _plan_document(first) == _plan_document(second)
    assert VerifiedExactEffectContract.from_data(data).prepared_plan().written_tensors == ("cache",)


@pytest.mark.parametrize("mutation", [
    "source", "constant", "goal", "theorem", "ir", "written", "external", "execution_identity",
    "drop_obligation", "weaken_obligation", "drop_check", "failed_check", "int_check", "reorder_checks",
    "extra_field", "schema_bool", "constant_bool", "duplicate_key",
])
def test_tampered_artifacts_fail_closed(mutation):
    data = copy.deepcopy(proved().to_data())
    if mutation == "source":
        data["source"] += "\n# different source\n"
    elif mutation == "constant":
        data["constants"]["KVD"] = 1024
    elif mutation == "goal":
        data["goal_name"] = "batch_invariance"
    elif mutation == "theorem":
        data["plan"]["theorem"]["pre_conditions"].pop()
    elif mutation == "ir":
        data["plan"]["execution"]["analyzed_kernel"]["grid"]["body"].pop()
    elif mutation == "written":
        data["plan"]["written_tensors"] = []
    elif mutation == "external":
        data["plan"]["external_obligations"] = []
    elif mutation == "execution_identity":
        data["plan"]["execution_identity"] = "0" * 64
    elif mutation == "drop_obligation":
        data["plan"]["obligations"].pop()
        data["checks"].pop()
    elif mutation == "weaken_obligation":
        data["plan"]["obligations"][1]["claim"] = "true"
    elif mutation == "drop_check":
        data["checks"].pop()
    elif mutation == "failed_check":
        data["checks"][0]["proved"] = False
    elif mutation == "int_check":
        data["checks"][0]["proved"] = 1
    elif mutation == "reorder_checks":
        data["checks"].reverse()
    elif mutation == "extra_field":
        data["assumed_framing"] = True
    elif mutation == "schema_bool":
        data["schema_version"] = True
    elif mutation == "constant_bool":
        data["constants"]["BLOCK_M"] = True
    else:
        raw = json.dumps(data)
        raw = raw.replace('"schema_version": 1', '"schema_version": 1, "schema_version": 1')
        with pytest.raises(ValueError, match="duplicate"):
            VerifiedExactEffectContract(raw).to_data()
        return
    with pytest.raises(ValueError):
        VerifiedExactEffectContract.from_data(data)


def test_failed_proof_has_no_artifact_and_diagnostics_cannot_issue_one():
    source = annotated().replace("x_row.to(cache.dtype.element_ty)", "(x_row + 1.0).to(cache.dtype.element_ty)")
    report = verify_exact_effects(source, "store_cache_kernel", {"KVD": 512, "BLOCK_M": 1})
    assert not report.proved and report.verified_contract is None
    plan = prepare_exact_effect_plan(annotated(), "store_cache_kernel", {"KVD": 512, "BLOCK_M": 1})
    with pytest.raises(ValueError, match="incomplete"):
        _issue_exact_effect_contract(plan, [])


def test_relational_and_temporal_goals_share_exact_execution_identity():
    relational = verify_annotations(annotated(), "store_cache_kernel", {"KVD": 512, "BLOCK_M": 1},
                                    goal_name="batch_invariance")
    assert relational.proved and relational.verified_contract is not None
    bundle = render_verified_kernel_to_verus((relational.verified_contract,), symbol_prefix="shared_store")
    assert bundle.execution_identity == proved().execution_identity
