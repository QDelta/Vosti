"""Clipped integer regions retain their meaning through parsing and lowering."""

from pathlib import Path

import pytest
import z3

from ir.annotation_to_config import ann_expr_to_ir, ann_expr_to_z3
from ir.annotations import ParseError, parse_annotation_text
from ir.relational_verifier import verify_annotations
from ir.smt import expr_to_z3
from ir.verus_contract import render_verified_contract_to_verus


@pytest.mark.parametrize("lower,upper", [(-3, 8), (0, 0), (4, 2), (9, 9)])
def test_integer_min_max_agree_between_annotation_and_ir(lower, upper):
    condition = parse_annotation_text("max(0, min(lo, hi)) == result")[0]
    env = {"lo": z3.IntVal(lower), "hi": z3.IntVal(upper)}
    expected = max(0, min(lower, upper))
    for value in (
        ann_expr_to_z3(condition.lhs, env, env),
        expr_to_z3(ann_expr_to_ir(condition.lhs), env),
    ):
        assert z3.simplify(value).as_long() == expected


@pytest.mark.parametrize("text", ["min(1) == 1", "max(1, 2, 3) == 3", "minimum(1, 2) == 1"])
def test_clipping_operators_have_closed_binary_syntax(text):
    with pytest.raises(ParseError):
        parse_annotation_text(text)


def test_clipping_is_not_an_implicit_floating_minimum_rule():
    expression = parse_annotation_text("min(left(x), 1) == 0")[0].lhs
    with pytest.raises(ValueError, match="requires integer operands"):
        ann_expr_to_z3(expression, {"x": z3.Real("x")}, {})


def test_clipped_regions_are_preserved_in_verified_artifact_and_verus():
    source = (Path(__file__).resolve().parents[2] / "triton_kernels/add.py").read_text()
    source = source.replace("0:N", "max(0, 0):min(N, N)")
    proof = verify_annotations(source, "add_kernel", {"BLOCK_M": 1, "BLOCK_N": 64})
    rendered = render_verified_contract_to_verus(proof.verified_contract, symbol_prefix="clipped")
    assert "vstd::math::max(0, 0)" in rendered.body
    assert "vstd::math::min(left.N, left.N)" in rendered.body
    assert "left.N == right.N" in rendered.body
