"""Quantifiers bind bare coordinates, not explicit left/right launch values."""

from pathlib import Path

import pytest
import z3

from ir.annotation_to_config import ann_forall_to_z3
from ir.annotations import (
    AnnAnd, AnnBinOp, AnnComparison, AnnImplies, AnnIndex, ForAllConstraint,
    FreeVar, IntConst, Left, Right,
)
from ir.relational_dataflow import prove_relational_dataflow_from_annotations


@pytest.mark.parametrize("side", [Left, Right])
@pytest.mark.parametrize("different_value", [False, True])
def test_bound_parameter_name_cannot_make_false_theorem_vacuously_pass(side, different_value):
    source = (Path(__file__).resolve().parents[2] / "triton_kernels/add.py").read_text()
    source = source.replace("#     left(M) > 0, N > 0,",
        f"#     left(M) > 0, N > 0,\n#     forall(M, and({side.__name__.lower()}(M) > 0)),")
    if different_value:
        source = source.replace("(x_block + y_block).to", "(x_block + M + y_block).to")
    report = prove_relational_dataflow_from_annotations(source, "add_kernel", dict(BLOCK_M=1, BLOCK_N=64))
    assert (report.verified_contract is not None) == (not different_value)


def _equivalent(left, right):
    solver = z3.Solver()
    solver.add(left != right)
    assert solver.check() == z3.unsat


@pytest.mark.parametrize("side", [Left, Right])
def test_explicit_launch_reference_survives_binder_shadowing(side):
    lhs, rhs = {"n": z3.Int("left_n")}, {"n": z3.Int("right_n")}
    goal = ForAllConstraint(("n",), AnnComparison(">", side("n"), IntConst(0)))
    actual = ann_forall_to_z3(goal, lhs, rhs)
    _equivalent(actual, (lhs if side is Left else rhs)["n"] > 0)
    assert str(lhs["n"]) == "left_n" and str(rhs["n"]) == "right_n"


def test_solver_symbol_name_does_not_capture_a_differently_named_parameter():
    # Binder spelling matches a solver symbol, but not the parameter's name.
    outer = z3.Int("i")
    goal = ForAllConstraint(("i",), AnnComparison(">", Left("length"), IntConst(0)))
    _equivalent(ann_forall_to_z3(goal, {"length": outer}, {}), outer > 0)


def test_bare_binding_reaches_arithmetic_indices_and_nested_predicates():
    tensor = z3.Function("values", z3.IntSort(), z3.IntSort())
    length = z3.Int("length")
    index = FreeVar("i")
    goal = ForAllConstraint(("i",), AnnImplies(
        AnnAnd((AnnComparison(">=", index, IntConst(0)),
                AnnComparison("<", index, Left("length")))),
        AnnComparison("==", AnnIndex(Left("data"), (AnnBinOp("+", index, IntConst(1)),)),
                      AnnIndex(Right("data"), (index,))),
    ))
    # Bare i shadows this unrelated free value; data stays a launch tensor.
    left = dict(i=z3.IntVal(99), data=tensor, length=length)
    right = dict(i=z3.IntVal(99), data=tensor)
    actual = ann_forall_to_z3(goal, left, right)
    i = z3.FreshInt("expected")
    _equivalent(actual, z3.ForAll(i, z3.Implies(z3.And(i >= 0, i < length), tensor(i+1) == tensor(i))))


@pytest.mark.parametrize("names", [(), ("i", "i")])
def test_invalid_binder_lists_fail_closed(names):
    with pytest.raises(ValueError, match="unique bound variables"):
        ann_forall_to_z3(ForAllConstraint(names, AnnComparison("==", IntConst(0), IntConst(0))), {}, {})


@pytest.mark.parametrize("select_related_row", [False, True])
def test_region_schema_placeholder_cannot_alias_user_parameter(select_related_row):
    source = (Path(__file__).resolve().parents[2] / "triton_kernels/add.py").read_text()
    source = source.replace("# @params(\n", "# @params(\n#   scalar(__region_schema_i, int),\n")
    source = source.replace("    M,\n    N,\n", "    M,\n    N,\n    __region_schema_i,\n", 1)
    source = source.replace(
        "#     left(x)[b:b+1, 0:N] == right(x)[0:1, 0:N],",
        "#     forall(i, implies(and(i == left(__region_schema_i), i >= 0, i < left(M)), "
        "left(x)[i:i+1, 0:N] == right(x)[0:1, 0:N])),",
    )
    if select_related_row:
        source = source.replace("#     b >= 0, b < left(M),",
            "#     b >= 0, b < left(M), b == left(__region_schema_i),")
    # Without the last premise, choose M=2, b=1, selector=0: only x[0]
    # agrees with the singleton input, and the selected x[1] is unconstrained.
    report = prove_relational_dataflow_from_annotations(source, "add_kernel", dict(BLOCK_M=1, BLOCK_N=64))
    assert (report.verified_contract is not None) == select_related_row
