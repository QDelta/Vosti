"""Unified artifact identity/schema tests, not an independent proof checker."""

from dataclasses import dataclass, replace
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from ir import Expr
from ir.contract_schema import validate_relational_contract
from ir.relational_artifact import (
    VerifiedDataflowContract, _build_verified_dataflow_contract, _decode, _encode,
)
from ir.relational_contract import relational_contract_data
from ir.relational_dataflow import prove_prepared_relational_dataflow
from ir.proof_preparation import prepare_annotation_proof


ROOT = Path(__file__).resolve().parents[2]


def _prepare(filename="add.py", kernel="add_kernel", constants=None, goal=None):
    return prepare_annotation_proof(
        (ROOT / "triton_kernels" / filename).read_text(), kernel,
        constants or {"BLOCK_M": 1, "BLOCK_N": 64}, goal_name=goal,
    )


@pytest.fixture(scope="module")
def proved():
    prepared = _prepare()
    report = prove_prepared_relational_dataflow(prepared)
    assert report.proved and report.verified_contract is not None
    return prepared, report


def _artifact(data):
    return VerifiedDataflowContract(json.dumps(data, sort_keys=True, separators=(",", ":")))


def test_artifact_preserves_exact_prepared_theorem(proved):
    prepared, report = proved
    artifact = report.verified_contract
    data = artifact.to_data()
    expected = relational_contract_data(
        source=prepared.source, kernel=prepared.kernel,
        declared_parameters=prepared.declared_parameters,
        annotation=prepared.annotation, constants=dict(prepared.constants),
        proof_kind="relational_dataflow",
    )
    assert data["theorem_contract"] == expected
    assert data["annotation_preconditions_satisfiable"] is True
    assert data["evidence"]["required_tensor_reads"] == ["x", "y"]
    assert data["evidence"]["alignments"][0]["paired_demands"]
    assert "details" not in data["evidence"]["checks"][0]
    assert _artifact(data) == artifact
    assert prove_prepared_relational_dataflow(prepared).verified_contract == artifact
    # A raw theorem surface is not a qualified artifact.
    with pytest.raises(ValueError, match="unsupported verified artifact type"):
        validate_relational_contract(
            expected, family="test",
        )


@pytest.mark.parametrize("mutation", [
    lambda r: replace(r, checks=()),
    lambda r: replace(r, checks=(replace(r.checks[0], proved=False),)),
    lambda r: replace(r, unsupported_reason="future analysis"),
    lambda r: replace(r, alignments=()),
    lambda r: replace(r, alignments=(replace(r.alignments[0], theorem_vacuous=True),)),
    lambda r: replace(r, alignments=(replace(r.alignments[0], relevant_statements=()),)),
])
def test_no_artifact_for_failed_unsupported_or_vacuous_reports(proved, mutation):
    prepared, report = proved
    assert _build_verified_dataflow_contract(prepared, mutation(report)) is None


@pytest.mark.parametrize("field,value", [
    ("source_sha256", "0" * 64), ("theorem_sha256", "0" * 64),
    ("goal_name", "other"), ("kernel_name", "other"), ("constants", ()),
])
def test_prepared_proof_identity_cannot_be_substituted(proved, field, value):
    prepared, report = proved
    with pytest.raises(ValueError, match="does not belong"):
        _build_verified_dataflow_contract(prepared, replace(report, **{field: value}))


@pytest.mark.parametrize("path,value", [
    (("extra",), 1), (("schema_version",), True), (("schema_version",), 1),
    (("schema_version",), 2), (("schema_version",), 4),
    (("annotation_preconditions_satisfiable",), "sat"),
    (("annotation_preconditions_satisfiable",), False),
    (("annotation_preconditions_satisfiable",), 1),
    (("theorem_contract", "theorem", "extra"), []),
    (("evidence", "extra"), 1), (("evidence", "checks"), []),
    (("evidence", "checks", 0, "proved"), False),
    (("evidence", "checks", 0, "proved"), 1),
    (("evidence", "alignments"), []),
    (("evidence", "alignments", 0, "theorem_vacuous"), True),
    (("evidence", "alignments", 0, "paired_demands"), []),
    (("evidence", "alignments", 0, "relevant_statements", 0, "ordinal"), 2),
    (("evidence", "alignments", 0, "relevant_statements", 0, "write", "node"), "FutureOp"),
    (("evidence", "alignments", 0, "loop_alignments", 0, "strategy"), "future"),
    (("evidence", "alignments", 0, "paired_demands", 0, "coordinates", "rank"), 99),
    (("evidence", "kernel", "name"), "wrong_kernel"),
    (("evidence", "required_tensor_reads"), ["nonexistent"]),
    (("evidence", "used_assumptions"), ["b", "a"]),
    (("evidence", "kernel", "params", 0, "type", "extra"), None),
])
def test_unknown_malformed_and_unproved_artifacts_fail_closed(proved, path, value):
    data = proved[1].verified_contract.to_data()
    parent = data
    for key in path[:-1]:
        parent = parent[key]
    parent[path[-1]] = value
    with pytest.raises(ValueError):
        _artifact(data).to_data()


def test_unknown_typed_nodes_fail_closed():
    @dataclass(frozen=True)
    class FutureOp(Expr):
        pass

    with pytest.raises(ValueError, match="unsupported typed node"):
        _encode(FutureOp())


def test_float_encoding_retains_signed_zero_and_infinity():
    assert _encode(0.0) != _encode(-0.0)
    for value in (0.0, -0.0, float("-inf"), 0.123):
        assert _decode(_encode(value), float).hex() == value.hex()
    with pytest.raises(ValueError, match="NaN"):
        _encode(float("nan"))


def test_new_operation_requires_schema_review():
    from ir import BinOp, IntLit

    value = _encode(BinOp("unreviewed", IntLit(1), IntLit(2)))
    with pytest.raises(ValueError, match="unsupported binary operation"):
        _decode(value, Expr)


def test_detected_vacuity_is_not_exported():
    prepared = _prepare()
    source = prepared.source.replace("#     left(M) > 0, N > 0,", "#     left(M) > 0, N > 0, N < 0,")
    assert source != prepared.source
    report = prove_prepared_relational_dataflow(prepare_annotation_proof(
        source, "add_kernel", dict(prepared.constants),
    ))
    assert report.proved  # A contradictory precondition gives a valid implication.
    assert any(a.theorem_vacuous for a in report.alignments)
    assert report.verified_contract is None


def test_missing_input_relation_is_not_exported():
    prepared = _prepare()
    source = prepared.source.replace("#     left(y)[b:b+1, 0:N] == right(y)[0:1, 0:N],\n", "")
    assert source != prepared.source
    report = prove_prepared_relational_dataflow(prepare_annotation_proof(
        source, "add_kernel", dict(prepared.constants),
    ))
    assert not report.proved and report.verified_contract is None


def test_every_output_is_retained():
    report = prove_prepared_relational_dataflow(_prepare(
        "rmsnorm_residual.py", "rmsnorm_residual_kernel", {"BLOCK_M": 1, "BLOCK_N": 1024},
    ))
    assert report.proved and report.verified_contract is not None
    data = report.verified_contract.to_data()
    assert len(data["theorem_contract"]["theorem"]["post"]) == 2
    assert len(data["evidence"]["alignments"]) == 2
    data["evidence"]["alignments"].pop()
    with pytest.raises(ValueError, match="output coverage"):
        _artifact(data).to_data()


def test_semantic_changes_affect_identity_but_diagnostics_do_not(proved):
    prepared, report = proved
    original = report.verified_contract
    diagnostic = replace(report, checks=tuple(replace(c, details="different prose") for c in report.checks))
    assert _build_verified_dataflow_contract(prepared, diagnostic) == original
    changed = replace(report, used_assumptions=("finite(x)",), external_obligations=("finite(x)",))
    assert _build_verified_dataflow_contract(prepared, changed).digest != original.digest
    # A well-typed mutation is a different artifact, not an independently
    # refuted proof. Consumers must bind artifact identity to the actual run.
    data = original.to_data()
    data["evidence"]["alignments"][0]["paired_demands"][0]["left"]["guard"]["body"]["value"] = False
    assert _artifact(data).digest != original.digest
    data = original.to_data()
    assert VerifiedDataflowContract(json.dumps(data)).canonical_json != original.canonical_json
    with pytest.raises(ValueError, match="noncanonical"):
        VerifiedDataflowContract(json.dumps(data)).to_data()


@pytest.mark.parametrize("filename,kernel", [
    ("fattn_paged.py", "fattn_varlen_paged_fwd_block_ptr_kernel"),
    ("fattn_paged_swa.py", "fattn_varlen_paged_swa_kernel"),
])
def test_attention_artifact_retains_ordered_loops_and_numeric_obligations(filename, kernel):
    prepared = _prepare(filename, kernel, {
        "BLOCK_M": 16, "BLOCK_N": 64, "D_HEAD": 128, "PAGE_BLOCK_SIZE": 64,
    }, "selected_row_prefix_equivalence")
    report = prove_prepared_relational_dataflow(prepared)
    assert report.proved and report.verified_contract is not None
    data = report.verified_contract.to_data()
    assert data["evidence"]["external_obligations"] == list(report.external_obligations)
    assert report.external_obligations
    loops = data["evidence"]["alignments"][0]["loop_alignments"]
    differing = [loop for loop in loops if loop["range_difference_neutrality"] is not None]
    assert differing and differing[0]["conditional_identities"]
    differing[0]["range_difference_neutrality"]["left_exclusive_checks"][0]["proved"] = False
    with pytest.raises(ValueError, match="unproved range difference"):
        _artifact(data).to_data()


@pytest.mark.parametrize("filename,kernel,constants,goal", [
    ("add.py", "add_kernel", {"BLOCK_M": 1, "BLOCK_N": 64}, None),
    ("fattn_paged.py", "fattn_varlen_paged_fwd_block_ptr_kernel",
     {"BLOCK_M": 16, "BLOCK_N": 64, "D_HEAD": 128, "PAGE_BLOCK_SIZE": 64},
     "selected_row_prefix_equivalence"),
    ("fattn_paged_swa.py", "fattn_varlen_paged_swa_kernel",
     {"BLOCK_M": 16, "BLOCK_N": 64, "D_HEAD": 128, "PAGE_BLOCK_SIZE": 64},
     "selected_row_prefix_equivalence"),
])
def test_canonical_identity_is_independent_of_python_hash_seed(filename, kernel, constants, goal):
    expected = prove_prepared_relational_dataflow(_prepare(filename, kernel, constants, goal))
    assert expected.verified_contract is not None
    code = f"""
from pathlib import Path
from ir.relational_dataflow import prove_relational_dataflow_from_annotations
source = Path('triton_kernels', {filename!r}).read_text()
r = prove_relational_dataflow_from_annotations(source, {kernel!r}, {constants!r}, goal_name={goal!r})
print(r.verified_contract.digest)
"""
    for seed in ("1", "97"):
        result = subprocess.run(
            [sys.executable, "-c", code], cwd=ROOT, check=True, capture_output=True, text=True,
            env={**os.environ, "PYTHONHASHSEED": seed, "PYTHONPATH": str(ROOT)},
        )
        assert result.stdout.strip() == expected.verified_contract.digest
