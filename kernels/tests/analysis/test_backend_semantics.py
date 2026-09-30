from __future__ import annotations

import z3

from ir import Cast, FloatType, Var
from ir.smt import expr_to_z3
from ir.scalar_values import FLOAT_VALUE


def _float_cast(name: str, target: str) -> Cast:
    return Cast(
        Var(name, type=FloatType()),
        "float",
        target,
        type=FloatType(),
    )


def test_scalar_cast_encoding_provides_congruence_but_not_identity() -> None:
    left = z3.Const("left", FLOAT_VALUE)
    right = z3.Const("right", FLOAT_VALUE)
    expression = _float_cast("value", "tl.bfloat16")
    encoded_left = expr_to_z3(expression, {"value": left})
    encoded_right = expr_to_z3(expression, {"value": right})

    congruence = z3.Solver()
    congruence.add(left == right, encoded_left != encoded_right)
    assert congruence.check() == z3.unsat

    not_identity = z3.Solver()
    not_identity.add(encoded_left != left)
    assert not_identity.check() == z3.sat


def test_scalar_cast_targets_are_distinct_opaque_operations() -> None:
    value = z3.Const("value", FLOAT_VALUE)
    bf16 = expr_to_z3(_float_cast("value", "tl.bfloat16"), {"value": value})
    fp32 = expr_to_z3(_float_cast("value", "tl.float32"), {"value": value})

    solver = z3.Solver()
    solver.add(bf16 != fp32)
    assert solver.check() == z3.sat
