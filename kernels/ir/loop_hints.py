"""Narrow compiler-only loop hints that preserve the ordered recurrence.

Every possible result must be a positive integer literal. A conditional may
test only equality of a declared constexpr parameter and an integer literal;
runtime dimensions, loads, calls, arithmetic and arbitrary flags are excluded.
The backend still owns faithful lowering of these compiler scheduling hints.
"""
from __future__ import annotations

import ast


def positive_static_unroll_hint(node: ast.AST, constexpr_params: set[str]) -> bool:
    if isinstance(node, ast.Constant):
        return type(node.value) is int and node.value > 0
    if not isinstance(node, ast.IfExp):
        return False
    condition = node.test
    if not (
        isinstance(condition, ast.Compare)
        and len(condition.ops) == 1
        and isinstance(condition.ops[0], (ast.Eq, ast.NotEq))
        and len(condition.comparators) == 1
        and isinstance(condition.left, ast.Name)
        and condition.left.id in constexpr_params
        and isinstance(condition.comparators[0], ast.Constant)
        and type(condition.comparators[0].value) is int
    ):
        return False
    return positive_static_unroll_hint(node.body, constexpr_params) and positive_static_unroll_hint(
        node.orelse, constexpr_params
    )
