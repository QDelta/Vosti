"""Adversarial checks of the single qualified verifier entrypoint."""

from dataclasses import replace
import json
from pathlib import Path
from unittest import mock

import pytest

from ir import BoolLit
from ir.relational_artifact import VerifiedDataflowContract, _build_verified_dataflow_contract
from ir.relational_verifier import (
    verify_prepared_annotations, verify_annotations,
    validate_dataflow_artifact,
)
from ir.relational_dataflow import prove_prepared_relational_dataflow
from ir.proof_preparation import prepare_annotation_proof


ROOT = Path(__file__).resolve().parents[2] / "triton_kernels"


@pytest.fixture(scope="module")
def proved():
    prepared = prepare_annotation_proof(
        (ROOT / "add.py").read_text(), "add_kernel", {"BLOCK_M": 1, "BLOCK_N": 64},
    )
    return prepared, prove_prepared_relational_dataflow(prepared)


def _artifact(data):
    return VerifiedDataflowContract(json.dumps(data, sort_keys=True, separators=(",", ":")))


def test_qualification_preserves_the_produced_artifact(proved):
    prepared, dataflow = proved
    assert verify_prepared_annotations(prepared).verified_contract == dataflow.verified_contract


@pytest.mark.parametrize("field,value", [
    ("unsupported_reason", "new unsupported construct"),
    ("checks", ()), ("verified_contract", None), ("alignments", ()),
])
def test_unsupported_or_incomplete_report_never_falls_back(proved, field, value):
    prepared, dataflow = proved
    with mock.patch("ir.relational_verifier.prove_prepared_relational_dataflow",
                    return_value=replace(dataflow, **{field: value})):
        with pytest.raises(ValueError):
            verify_prepared_annotations(prepared)


def test_verifier_calls_one_producer_once(proved):
    prepared, dataflow = proved
    with mock.patch("ir.relational_verifier.prove_prepared_relational_dataflow",
                    return_value=dataflow) as produce:
        assert verify_prepared_annotations(prepared).verified_contract == dataflow.verified_contract
    produce.assert_called_once_with(prepared)


@pytest.mark.parametrize("change", ["guard", "operation", "loop_order", "theorem", "specialization"])
def test_well_formed_artifact_mutation_cannot_be_substituted(proved, change):
    prepared, dataflow = proved
    data = dataflow.verified_contract.to_data()
    evidence = data["evidence"]
    if change == "guard":
        demand = evidence["alignments"][0]["paired_demands"][0]
        demand["left"]["guard"]["body"]["value"] = False
    elif change == "operation":
        def mutate(node):
            if isinstance(node, dict):
                if node.get("node") == "BinOp" and node["op"] == "+":
                    node["op"] = "-"
                    return True
                return any(mutate(value) for value in node.values())
            if isinstance(node, list):
                return any(mutate(value) for value in node)
            return False
        assert mutate(evidence["kernel"])
    elif change == "loop_order":
        assert len(evidence["kernel"]["grid"]["iters"]) > 1
        evidence["kernel"]["grid"]["iters"].reverse()
    elif change == "theorem":
        data["theorem_contract"]["goal_name"] = "different_goal"
    else:
        data["theorem_contract"]["constants"]["BLOCK_M"]["value"] = 2
    mutated = _artifact(data)
    mutated.to_data()  # Schema-valid is deliberately weaker than proof binding.
    with pytest.raises(ValueError, match="differs from prepared proof/report"):
        validate_dataflow_artifact(prepared, replace(dataflow, verified_contract=mutated))


def test_changed_live_dependency_report_cannot_reuse_old_artifact(proved):
    prepared, report = proved
    alignment = report.alignments[0]
    demand = alignment.paired_demands[0]
    changed = replace(demand, left=replace(demand.left, guard=replace(demand.left.guard, body=BoolLit(False))))
    report = replace(report, alignments=(replace(alignment, paired_demands=(changed, *alignment.paired_demands[1:])),))
    with pytest.raises(ValueError, match="differs from prepared proof/report"):
        validate_dataflow_artifact(prepared, report)


def test_each_proof_can_pass_but_different_sources_cannot_be_combined(proved):
    prepared, dataflow = proved
    source = prepared.source.replace("(x_block + y_block)", "(x_block - y_block)")
    assert source != prepared.source
    other = prepare_annotation_proof(source, "add_kernel", dict(prepared.constants))
    other_dataflow = prove_prepared_relational_dataflow(other)
    assert other_dataflow.proved
    with pytest.raises(ValueError, match="does not belong to the prepared theorem"):
        validate_dataflow_artifact(other, dataflow)


def test_added_numeric_premise_requires_explicit_consumer_preservation(proved):
    prepared, report = proved
    report = replace(report, used_assumptions=("finite(x)",), external_obligations=("finite(x)",))
    report = replace(report, verified_contract=_build_verified_dataflow_contract(prepared, report))
    with mock.patch("ir.relational_verifier.prove_prepared_relational_dataflow", return_value=report):
        with pytest.raises(ValueError, match="cannot discharge or erase"):
            verify_prepared_annotations(prepared)
        assert verify_prepared_annotations(prepared, preserve_analyzer_conditions=True) is report


@pytest.mark.parametrize("filename,kernel,constants", [
    ("matmul.py", "matmul_kernel", {"BLOCK_M": 16, "BLOCK_N": 64, "BLOCK_K": 64}),
    ("rmsnorm_residual.py", "rmsnorm_residual_kernel", {"BLOCK_M": 1, "BLOCK_N": 1024}),
    ("fattn_paged.py", "fattn_varlen_paged_fwd_block_ptr_kernel",
     {"BLOCK_M": 16, "BLOCK_N": 64, "D_HEAD": 128, "PAGE_BLOCK_SIZE": 64}),
    ("fattn_paged_swa.py", "fattn_varlen_paged_swa_kernel",
     {"BLOCK_M": 16, "BLOCK_N": 64, "D_HEAD": 128, "PAGE_BLOCK_SIZE": 64}),
])
def test_loop_multioutput_and_ragged_contracts_verify(filename, kernel, constants):
    prepared = prepare_annotation_proof(
        (ROOT / filename).read_text(), kernel, constants, goal_name="batch_invariance",
    )
    assert verify_prepared_annotations(prepared).proved


@pytest.mark.parametrize("filename,kernel", [
    ("fattn_paged.py", "fattn_varlen_paged_fwd_block_ptr_kernel"),
    ("fattn_paged_swa.py", "fattn_varlen_paged_swa_kernel"),
])
def test_named_conditional_goals_use_the_same_verifier(filename, kernel):
    source = (ROOT / filename).read_text()
    constants = dict(BLOCK_M=16, BLOCK_N=64, D_HEAD=128, PAGE_BLOCK_SIZE=64)
    with pytest.raises(ValueError, match="cannot discharge or erase"):
        verify_annotations(source, kernel, constants, goal_name="selected_row_prefix_equivalence")
    result = verify_annotations(source, kernel, constants, goal_name="selected_row_prefix_equivalence",
                                preserve_analyzer_conditions=True)
    assert result.verified_contract is not None and result.external_obligations


@pytest.mark.parametrize("value", [None, 0, 1, "false", "true"])
def test_condition_policy_requires_an_explicit_boolean(proved, value):
    with pytest.raises(TypeError, match="must be boolean"):
        verify_prepared_annotations(proved[0], preserve_analyzer_conditions=value)
