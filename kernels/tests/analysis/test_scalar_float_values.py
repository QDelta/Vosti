"""FP source dependencies cannot disappear through real-number identities."""

from pathlib import Path
import struct

import pytest
import z3

from ir import BinOp, FloatLit, FloatType, Var
from ir.annotation_to_config import _z3_binop, _z3_compare
from ir.scalar_values import FLOAT_VALUE, float_literal
from ir.smt import expr_to_z3
from ir.proof_preparation import prepare_annotation_proof
from ir.regional_obligations import relevant_output_writes
from ir.relational_dataflow import prove_relational_dataflow_from_annotations


def _source(expression, shared=False):
    source = (Path(__file__).resolve().parents[2] / "triton_kernels/add.py").read_text()
    source = source.replace("# @params(\n", "# @params(\n#   scalar(scale, float),\n")
    source = source.replace("    M,\n    N,\n", "    M,\n    N,\n    scale,\n", 1)
    if shared:
        source = source.replace("#   same(N),", "#   same(N, scale),")
    return source.replace("        (x_block + y_block).to(o.dtype.element_ty),",
                          f"        (x_block + ({expression}) + y_block).to(o.dtype.element_ty),")


@pytest.mark.parametrize("expression", ["(scale + 1.0) - scale", "scale - scale", "scale * 0.0"])
@pytest.mark.parametrize("shared", [False, True])
def test_source_float_expressions_require_equal_inputs(expression, shared):
    source = _source(expression, shared)
    constants = {"BLOCK_M": 1, "BLOCK_N": 64}
    unified = prove_relational_dataflow_from_annotations(source, "add_kernel", constants)
    assert unified.proved == shared
    assert (unified.verified_contract is not None) == shared


def test_float32_counterexample_and_preserved_source_tree():
    def f32(value):
        return struct.unpack("f", struct.pack("f", value))[0]
    assert f32(f32(2**24 + 1.0) - 2**24) == 0.0
    assert f32(f32(0 + 1.0) - 0) == 1.0
    prepared = prepare_annotation_proof(_source("(scale + 1.0) - scale"), "add_kernel",
                                        {"BLOCK_M": 1, "BLOCK_N": 64})
    value = relevant_output_writes(prepared.kernel, "o")[-1].write.value.value.lhs.rhs
    assert isinstance(value, BinOp) and value.op == "-"
    assert isinstance(value.lhs, BinOp) and value.lhs.op == "+"
    assert value.rhs.name == "scale"


def test_integer_cancellation_remains_available():
    source = _source("(scale + 1) - scale").replace("scalar(scale, float)", "scalar(scale, int)")
    report = prove_relational_dataflow_from_annotations(source, "add_kernel", {"BLOCK_M": 1, "BLOCK_N": 64})
    assert report.verified_contract is not None


def test_legacy_real_binding_cannot_reintroduce_real_arithmetic():
    expression = BinOp("-", Var("x", type=FloatType()), Var("x", type=FloatType()))
    with pytest.raises(ValueError, match="not Z3 reals"):
        expr_to_z3(expression, {"x": z3.Real("x")})


def test_float_literal_signed_zeros_are_not_identified():
    solver = z3.Solver()
    solver.add(float_literal(0.0) != float_literal(-0.0))
    assert solver.check() == z3.sat
    assert float_literal(0.0).eq(float_literal(0.0))


def test_source_float_equality_does_not_assume_nan_reflexivity():
    value = Var("x", type=FloatType())
    compare = expr_to_z3(BinOp("==", value, value), {"x": z3.Const("x", FLOAT_VALUE)})
    solver = z3.Solver()
    solver.add(z3.Not(compare))
    assert solver.check() == z3.sat


@pytest.mark.parametrize("operator", ["+", "-", "*", "//", "min", "max"])
def test_annotation_arithmetic_is_integer_only(operator):
    with pytest.raises(ValueError, match="requires integer"):
        _z3_binop(operator, float_literal(1.0), float_literal(2.0))


def test_floating_annotation_equality_is_value_identity_not_numeric_order():
    value = z3.Const("x", FLOAT_VALUE)
    assert z3.is_true(z3.simplify(_z3_compare("==", value, value)))
    with pytest.raises(ValueError, match="not numeric ordering"):
        _z3_compare(">", value, float_literal(0.0))
