"""Complete typed annotation premises, not just quantifier-free scalar facts."""

from dataclasses import replace
from pathlib import Path
from unittest import mock

import pytest
import z3

from ir.annotation_satisfiability import annotation_preconditions, check_annotation_satisfiability
from ir.relational_artifact import _build_verified_dataflow_contract
from ir.relational_dataflow import prove_prepared_relational_dataflow
from ir.smt import ProofCheck
from ir.proof_preparation import prepare_annotation_proof


ROOT = Path(__file__).resolve().parents[2] / "triton_kernels"


def _prepare(extra="", *, metadata=False, relation="", shared_metadata=False):
    source = (ROOT / "add.py").read_text()
    if metadata:
        dimension = "M" if shared_metadata else "1"
        source = source.replace("# @params(\n", f"# @params(\n#   tensor(meta, int32, shape({dimension})),\n")
        source = source.replace("    M,\n    N,\n", "    M,\n    N,\n    meta,\n", 1)
    if shared_metadata:
        source = source.replace("#   same(N),", "#   same(N, meta),")
    additions = "\n".join(line for line in (extra + "\n" + relation).splitlines() if line)
    source = source.replace("#     left(M) > 0, N > 0,",
                            "#     left(M) > 0, N > 0," + ("\n" + additions if additions else ""))
    return prepare_annotation_proof(source, "add_kernel", dict(BLOCK_M=1, BLOCK_N=64))


@pytest.mark.parametrize("extra", ["", "#     left(M) > 1000,", "#     offset < sub(0, 1000),"])
def test_bounded_witness_failure_does_not_restrict_valid_annotation(extra):
    prepared = _prepare(extra)
    status, check = check_annotation_satisfiability(prepared, witness_bounds=(1,))
    assert status == "sat" and check.proved
    report = prove_prepared_relational_dataflow(prepared)
    assert report.proved and report.annotation_satisfiability == "sat"
    assert report.verified_contract is not None


@pytest.mark.parametrize("relation", [
    "#     left(meta)[0:1] == right(meta)[0:1],",
    "#     forall(i, implies(and(i >= 0, i < 1), left(meta)[i:i+1] == right(meta)[i:i+1])),",
])
def test_value_region_contradictions_prevent_artifact_issuance(relation):
    prepared = _prepare("#     left(meta)[0] == 0, right(meta)[0] == 1,\n",
                        metadata=True, relation=relation)
    scalar_solver = z3.Solver()
    scalar_solver.add(*prepared.first_config.base_assumptions)
    assert scalar_solver.check() == z3.sat  # The old diagnostic misses the contradiction.
    report = prove_prepared_relational_dataflow(prepared)
    assert report.proved  # A valid conditional equality, but no qualified domain.
    assert report.annotation_satisfiability == "unsat"
    assert report.verified_contract is None


def test_quantified_scalar_contradiction_is_not_ignored():
    prepared = _prepare(
        "#     left(meta)[0] == 0,\n"
        "#     forall(i, implies(and(i >= 0, i < 1), left(meta)[i] == 1)),",
        metadata=True,
    )
    assert check_annotation_satisfiability(prepared)[0] == "unsat"


def test_inactive_quantified_region_does_not_require_its_equality():
    prepared = _prepare("#     left(meta)[0] == 0, right(meta)[0] == 1,\n",
        metadata=True, relation=
        "#     forall(i, implies(and(i >= 0, i < 0), left(meta)[i:i+1] == right(meta)[i:i+1])),")
    assert check_annotation_satisfiability(prepared)[0] == "sat"


def test_whole_tensor_sharing_requires_equal_shapes():
    prepared = _prepare("#     left(M) > 1,", metadata=True, shared_metadata=True)
    assert check_annotation_satisfiability(prepared)[0] == "unsat"


def test_unequal_slice_lengths_are_not_a_satisfiable_region_equality():
    prepared = _prepare()
    source = prepared.source.replace("left(x)[b:b+1, 0:N]", "left(x)[b:b+2, 0:N]")
    prepared = prepare_annotation_proof(source, "add_kernel", dict(prepared.constants))
    assert check_annotation_satisfiability(prepared)[0] == "unsat"


def test_given_clause_equality_is_part_of_annotation_domain():
    prepared = _prepare("#     left(M) > 1,")
    source = prepared.source.replace(
        "right(x)[0:1, 0:N],", "right(x)[0:1, 0:N] given left(M),")
    assert source != prepared.source
    prepared = prepare_annotation_proof(source, "add_kernel", dict(prepared.constants))
    # The slices fit, but given M requires left(M) == right(M) == 1.
    assert check_annotation_satisfiability(prepared)[0] == "unsat"


def test_precondition_encoding_does_not_mutate_proof_context():
    prepared = _prepare()
    before = (dict(prepared.first_config.left_env), dict(prepared.first_config.right_env),
              list(prepared.first_config.base_assumptions))
    assert annotation_preconditions(prepared)
    assert prepared.first_config.left_env == before[0]
    assert prepared.first_config.right_env == before[1]
    assert prepared.first_config.base_assumptions == before[2]


def test_unsat_bounded_search_and_unknown_unrestricted_query_stay_unknown():
    solvers = [mock.Mock() for _ in range(4)]
    for solver, result in zip(solvers, [z3.unsat, z3.unknown, z3.unsat, z3.unknown]):
        solver.check.return_value = result
    prepared = _prepare()
    with mock.patch("ir.annotation_satisfiability.z3.Solver", side_effect=solvers):
        status, check = check_annotation_satisfiability(prepared)
    assert status == "unknown" and not check.proved


@pytest.mark.parametrize("status", ["unknown", "unchecked", "unsat"])
def test_missing_positive_sat_evidence_cannot_export_an_artifact(status):
    prepared = _prepare()
    report = prove_prepared_relational_dataflow(prepared)
    assert report.verified_contract is not None
    assert _build_verified_dataflow_contract(prepared, replace(report, annotation_satisfiability=status)) is None


def test_unknown_satisfiability_preserves_equality_result_but_blocks_qualification():
    prepared = _prepare()
    with mock.patch("ir.relational_dataflow.check_annotation_satisfiability",
                    return_value=("unknown", ProofCheck("annotation_preconditions_satisfiable", False, "unknown"))):
        report = prove_prepared_relational_dataflow(prepared)
    assert report.proved and report.annotation_satisfiability == "unknown"
    assert report.verified_contract is None


@pytest.mark.parametrize("filename,kernel", [
    ("fattn_paged.py", "fattn_varlen_paged_fwd_block_ptr_kernel"),
    ("fattn_paged_swa.py", "fattn_varlen_paged_swa_kernel"),
])
@pytest.mark.parametrize("goal", ["batch_invariance", "selected_row_prefix_equivalence"])
def test_attention_annotations_have_satisfying_typed_input_pairs(filename, kernel, goal):
    prepared = prepare_annotation_proof((ROOT / filename).read_text(), kernel,
        dict(BLOCK_M=16, BLOCK_N=64, D_HEAD=128, PAGE_BLOCK_SIZE=64), goal_name=goal)
    assert check_annotation_satisfiability(prepared)[0] == "sat"
