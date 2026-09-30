"""IEEE signed-zero witnesses for relational dependency and identity rules."""

from pathlib import Path
import struct

import pytest

from ir import Assign, BinOp, FloatType, IntLit, Slice, TensorType, Var
from ir.positional import (
    NeutralityAssumptions, PositionalAnalyzer, facts_all_zero, pred_true,
)
from ir.regions import GuardedRegion, regions_write_masked
from ir.relational_dataflow import prove_relational_dataflow_from_annotations


def bits(value):
    return struct.pack(">f", value)


@pytest.mark.parametrize("operand_order", ["x_block * zero", "zero * x_block"])
def test_finite_zero_product_cannot_qualify_unrelated_inputs(operand_order):
    # Empty input equality intentionally says nothing about the selected x.
    # Before the repair this theorem passed with finite(x_block), even though
    # the following two finite executions have unequal output bits.
    assert 1.0 * 0.0 == -1.0 * 0.0
    assert bits(1.0 * 0.0) != bits(-1.0 * 0.0)
    source = (Path(__file__).resolve().parents[2] / "triton_kernels/add.py").read_text()
    source = source.replace(
        "left(x)[b:b+1, 0:N] == right(x)[0:1, 0:N]",
        "left(x)[0:0, 0:N] == right(x)[0:0, 0:N]",
    ).replace(
        "    o_block_ptr = tl.make_block_ptr(\n",
        "    zero = tl.full((BLOCK_M, BLOCK_N), 0.0, dtype=tl.float32)\n"
        f"    result = {operand_order}\n"
        "    o_block_ptr = tl.make_block_ptr(\n",
    ).replace("(x_block + y_block).to(o.dtype.element_ty)", "result.to(o.dtype.element_ty)")
    report = prove_relational_dataflow_from_annotations(
        source, "add_kernel", {"BLOCK_M": 1, "BLOCK_N": 64},
    )
    assert not report.proved
    assert report.verified_contract is None
    assert any(not check.proved and "tensor" in check.name for check in report.checks)


@pytest.mark.parametrize("form", ["x+zero", "zero+x", "x+=zero"])
def test_finite_plus_zero_is_not_bitwise_state_identity(form):
    assert -0.0 + 0.0 == -0.0
    assert bits(-0.0 + 0.0) != bits(-0.0)
    typ = TensorType(FloatType(), [IntLit(4)])
    x, zero = Var(name="x", type=typ), Var(name="zero", type=typ)
    analyzer = PositionalAnalyzer(
        assumptions=NeutralityAssumptions(finite_vars=frozenset({"x"})),
        type_env={"x": typ, "zero": typ}, accumulators=frozenset({"x"}),
    )
    analyzer.env["zero"] = facts_all_zero(1)
    value = zero if form == "x+=zero" else BinOp(
        op="+", lhs=x if form == "x+zero" else zero,
        rhs=zero if form == "x+zero" else x, type=typ,
    )
    analyzer.exec_stmt(Assign(target=x, op="+" if form == "x+=zero" else None, value=value))
    assert analyzer.env["x"].unchanged_from is None


def test_compound_zero_multiply_keeps_old_sign_dependency():
    typ = TensorType(FloatType(), [IntLit(4)])
    x, zero = Var(name="x", type=typ), Var(name="zero", type=typ)
    demanded = GuardedRegion([Slice(IntLit(0), IntLit(4))], pred_true(1))
    regions = {"x": demanded}
    regions_write_masked(
        Assign(target=x, op="*", value=zero), {"x": typ, "zero": typ},
        regions, {"zero": facts_all_zero(1)}, "x", frozenset({"x"}),
    )
    assert regions["x"] == demanded
