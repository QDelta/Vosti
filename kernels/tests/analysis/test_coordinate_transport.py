"""Nested scalar expressions must follow the coordinate map of shape ops."""

from dataclasses import dataclass

import pytest

from ir import (
    BinOp, BoolType, BroadcastTo, Expr, IntLit, Max, Min, Squeeze,
    TensorIndex, TensorType, Transpose, Unsqueeze, Var, Where,
)
from ir.positional import (
    NeutralityAssumptions, PositionalAnalyzer, facts_bool_exact,
    free_idx, subst_free_indices,
)
from ir.subst import subst_expr


def _predicate(row, column):
    return BinOp("<", Where(
        BinOp("<", row, IntLit(1)), column,
        BinOp("+", row, column),
    ), IntLit(2))


@pytest.mark.parametrize("operation,dims,indices", [
    (lambda v: Transpose(v, [1, 0]), (2, 3), (free_idx(1), free_idx(0))),
    (lambda v: Unsqueeze(v, 1), (2, 3), (free_idx(0), free_idx(2))),
    (lambda v: Squeeze(v, 0), (1, 3), (IntLit(0), free_idx(0))),
    (lambda v: BroadcastTo(v, [IntLit(4), IntLit(3)]), (1, 3),
     (IntLit(0), free_idx(1))),
])
def test_shape_ops_transport_nested_where(operation, dims, indices):
    typ = TensorType(BoolType(), [IntLit(d) for d in dims])
    predicate = _predicate(free_idx(0), free_idx(1))
    analyzer = PositionalAnalyzer(
        NeutralityAssumptions(), type_env={"mask": typ},
        env={"mask": facts_bool_exact(2, predicate)},
        elem_body={"mask": predicate},
    )
    expression = operation(Var("mask", type=typ))
    rank = 3 if isinstance(expression, Unsqueeze) else (
        1 if isinstance(expression, Squeeze) else 2
    )
    facts, body = analyzer._eval(expression, rank)
    expected = _predicate(*indices)
    assert facts.true_where.body == expected
    assert body == expected


def test_shared_substitution_transports_clipping_and_indexed_coordinates():
    expression = Where(
        BinOp("<", free_idx(0), Var("limit")),
        Min([free_idx(1), IntLit(7)]),
        Max([TensorIndex(Var("offsets"), [free_idx(0)]), free_idx(1)]),
    )
    # Simultaneous substitution must swap indices, not map both to _i0.
    expected = Where(
        BinOp("<", free_idx(1), Var("limit")),
        Min([free_idx(0), IntLit(7)]),
        Max([TensorIndex(Var("offsets"), [free_idx(1)]), free_idx(0)]),
    )
    assert subst_free_indices(expression, {
        "_i0": free_idx(1), "_i1": free_idx(0),
    }) == expected
    assert subst_free_indices is subst_expr


def test_unknown_expression_cannot_silently_retain_old_coordinates():
    @dataclass(frozen=True)
    class FutureExpression(Expr):
        value: Expr

    with pytest.raises(AssertionError, match="unhandled expr"):
        subst_free_indices(FutureExpression(free_idx(0)), {"_i0": IntLit(0)})
