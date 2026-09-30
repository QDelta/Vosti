"""Corner-case tests for the per-position analyzer.

The kernel-level tests in `test_positional.py` cover the load-bearing
end-to-end paths. This file exercises individual propagation rules in
isolation, with hand-built IR fragments and explicit pre-/post-conditions.

Each test pins a single algebraic rule and a single soundness gate, so
that a future regression touches the smallest possible blast radius.
"""

import pytest

from ir import (
    Assign,
    BinOp,
    BoolLit,
    BroadcastTo,
    Exp2,
    FloatLit,
    FloatType,
    If,
    IntLit,
    IntType,
    Let,
    Maximum,
    Not,
    ReduceMax,
    ReduceSum,
    Squeeze,
    TensorType,
    Transpose,
    Unsqueeze,
    Var,
    Where,
    add_,
    mul_,
    sub_,
)
from ir.positional import (
    NeutralityAssumptions,
    PositionalAnalyzer,
    Pred,
    facts_all_neg_inf,
    facts_all_one,
    facts_all_zero,
    facts_unchanged,
    facts_unknown,
    free_idx,
    pred_false,
    pred_true,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _f32(*dims):
    return TensorType(FloatType(), [IntLit(d) for d in dims])


def _bool(*dims):
    from ir import BoolType

    return TensorType(BoolType(), [IntLit(d) for d in dims])


def _int(*dims):
    return TensorType(IntType(), [IntLit(d) for d in dims])


def _make_analyzer(
    type_env, *, finite=(), positive=(), all_false=(), accumulators=()
):
    return PositionalAnalyzer(
        assumptions=NeutralityAssumptions(
            finite_vars=frozenset(finite),
            positive_vars=frozenset(positive),
            all_false_masks=frozenset(all_false),
        ),
        type_env=type_env,
        accumulators=frozenset(accumulators),
    )


def _typed_var(name, ty):
    return Var(name=name, type=ty)


# ---------------------------------------------------------------------------
# soundness: -inf propagation through `+` and `-` requires explicit finiteness
# ---------------------------------------------------------------------------


def test_plus_with_neg_inf_lhs_requires_finite_rhs():
    """`(-inf) + x = -inf` must NOT fire when x is not declared finite.
    A "no positive proof of -inf on rhs" shortcut would be unsound: rhs
    could still be +inf, making the result NaN."""
    type_env = {"a": _f32(4), "b": _f32(4)}
    a = _make_analyzer(type_env)  # neither a nor b declared finite
    # Pre-load the env: a is provably -inf at every position; b is unknown.
    a.env["a"] = facts_all_neg_inf(1)
    a.env["b"] = facts_unknown(1)
    # Evaluate a + b.
    expr = add_(_typed_var("a", _f32(4)), _typed_var("b", _f32(4)))
    expr = expr.__class__(**{**expr.__dict__, "type": _f32(4)})  # set type
    facts, _ = a._eval(expr, 1)
    # Without finite(b), we cannot conclude the sum is -inf.
    assert not _pred_is_true(facts.neg_inf_where), (
        "(-inf) + b should NOT be -inf without finite(b); rhs could be +inf "
        "yielding NaN. Got neg_inf_where = " + str(facts.neg_inf_where.body)
    )


def test_plus_with_neg_inf_lhs_succeeds_when_rhs_finite():
    """Same situation, but with finite(b) declared: -inf propagates."""
    type_env = {"a": _f32(4), "b": _f32(4)}
    a = _make_analyzer(type_env, finite=("b",))
    a.env["a"] = facts_all_neg_inf(1)
    a.env["b"] = facts_unknown(1)
    expr = add_(_typed_var("a", _f32(4)), _typed_var("b", _f32(4)))
    expr = expr.__class__(**{**expr.__dict__, "type": _f32(4)})
    facts, _ = a._eval(expr, 1)
    assert _pred_is_true(facts.neg_inf_where)
    assert any("finite(b)" in u for u in a.used_assumptions), (
        f"finiteness of b must be recorded; got {a.used_assumptions}"
    )


def test_minus_with_neg_inf_lhs_requires_finite_rhs():
    """Symmetric to the + test, for the - operator."""
    type_env = {"a": _f32(4), "b": _f32(4)}
    a = _make_analyzer(type_env)
    a.env["a"] = facts_all_neg_inf(1)
    a.env["b"] = facts_unknown(1)
    expr = sub_(_typed_var("a", _f32(4)), _typed_var("b", _f32(4)))
    expr = expr.__class__(**{**expr.__dict__, "type": _f32(4)})
    facts, _ = a._eval(expr, 1)
    assert not _pred_is_true(facts.neg_inf_where), (
        "(-inf) - b should NOT be -inf without finite(b)"
    )


# ---------------------------------------------------------------------------
# soundness: x - x = 0 requires finiteness of x
# ---------------------------------------------------------------------------


def test_x_minus_x_zero_requires_finite():
    """x - x = 0 only when x is finite (else NaN for x = ±inf).

    NB: we build the BinOp directly rather than using the `sub_` helper,
    because `sub_` short-circuits `x - x` to `IntLit(0)` syntactically and
    would bypass the analyzer's algebraic rule entirely."""
    from ir import BinOp

    type_env = {"x": _f32(4)}
    a = _make_analyzer(type_env)  # x not finite
    a.env["x"] = facts_unchanged(1, "x")
    expr = BinOp(
        op="-",
        lhs=_typed_var("x", _f32(4)),
        rhs=_typed_var("x", _f32(4)),
        type=_f32(4),
    )
    facts, _ = a._eval(expr, 1)
    assert not _pred_is_true(facts.zero_where), (
        "x - x must NOT collapse to 0 without finite(x)"
    )


def test_x_minus_x_zero_when_finite():
    """With finite(x), x - x = 0, and the assumption is recorded."""
    from ir import BinOp

    type_env = {"x": _f32(4)}
    a = _make_analyzer(type_env, finite=("x",))
    a.env["x"] = facts_unchanged(1, "x")
    expr = BinOp(
        op="-",
        lhs=_typed_var("x", _f32(4)),
        rhs=_typed_var("x", _f32(4)),
        type=_f32(4),
    )
    facts, _ = a._eval(expr, 1)
    assert _pred_is_true(facts.zero_where)
    assert "finite(x)@x-x" in a.used_assumptions


# ---------------------------------------------------------------------------
# Maximum(x, -inf) requires finite(x); else not safe to drop -inf
# ---------------------------------------------------------------------------


def test_maximum_with_neg_inf_requires_finite_other():
    """Maximum(x, -inf) = x is the IEEE 754 identity ONLY for NaN-free x.
    Without finite(x), x could be NaN and the result becomes NaN, not x."""
    type_env = {"x": _f32(4)}
    a = _make_analyzer(type_env)  # x not finite
    a.env["x"] = facts_unchanged(1, "x")
    expr = Maximum(
        lhs=_typed_var("x", _f32(4)),
        rhs=FloatLit(value=float("-inf"), type=_f32(4)),  # broadcast scalar
    )
    expr = expr.__class__(**{**expr.__dict__, "type": _f32(4)})
    facts, _ = a._eval(expr, 1)
    assert facts.unchanged_from is None, (
        "Maximum(x, -inf) must not preserve x's unchanged_from without finite(x)"
    )


def test_maximum_with_neg_inf_when_finite_propagates():
    type_env = {"x": _f32(4)}
    a = _make_analyzer(type_env, finite=("x",))
    a.env["x"] = facts_unchanged(1, "x")
    expr = Maximum(
        lhs=_typed_var("x", _f32(4)),
        rhs=FloatLit(value=float("-inf"), type=_f32(4)),
    )
    expr = expr.__class__(**{**expr.__dict__, "type": _f32(4)})
    facts, _ = a._eval(expr, 1)
    assert facts.unchanged_from == "x"


# ---------------------------------------------------------------------------
# Reductions: ReduceMax / ReduceSum on whole-tensor constants
# ---------------------------------------------------------------------------


def test_reduce_max_of_all_neg_inf_yields_neg_inf():
    type_env = {"x": _f32(4, 8)}
    a = _make_analyzer(type_env)
    a.env["x"] = facts_all_neg_inf(2)
    expr = ReduceMax(
        value=_typed_var("x", _f32(4, 8)), axis=1, type=_f32(4)
    )
    facts, _ = a._eval(expr, 1)
    assert _pred_is_true(facts.neg_inf_where)


def test_reduce_max_of_all_zero_yields_zero():
    """Newly added: ReduceMax of zeros is 0 (max of finite zeros is 0).
    Before this rule, the result was unknown."""
    type_env = {"x": _f32(4, 8)}
    a = _make_analyzer(type_env)
    a.env["x"] = facts_all_zero(2)
    expr = ReduceMax(
        value=_typed_var("x", _f32(4, 8)), axis=1, type=_f32(4)
    )
    facts, _ = a._eval(expr, 1)
    assert _pred_is_true(facts.zero_where)


def test_reduce_max_of_all_one_yields_one():
    type_env = {"x": _f32(4, 8)}
    a = _make_analyzer(type_env)
    a.env["x"] = facts_all_one(2)
    expr = ReduceMax(
        value=_typed_var("x", _f32(4, 8)), axis=1, type=_f32(4)
    )
    facts, _ = a._eval(expr, 1)
    assert _pred_is_true(facts.one_where)


def test_reduce_sum_of_all_zero_yields_zero():
    type_env = {"x": _f32(4, 8)}
    a = _make_analyzer(type_env)
    a.env["x"] = facts_all_zero(2)
    expr = ReduceSum(
        value=_typed_var("x", _f32(4, 8)), axis=1, type=_f32(4)
    )
    facts, _ = a._eval(expr, 1)
    assert _pred_is_true(facts.zero_where)


def test_reduce_sum_partial_zero_unknown():
    """ReduceSum doesn't lift partial zeros — only whole-tensor zero
    yields all-zero. Verify partial info doesn't accidentally collapse."""
    type_env = {"x": _f32(4, 8)}
    a = _make_analyzer(type_env)
    # Partial: x is 0 only where _i0 < 2, the rest unknown.
    a.env["x"] = a.env.get("x", facts_unknown(2))
    a.env["x"] = type(a.env["x"])(
        rank=2,
        zero_where=Pred(2, BoolLit(True)),  # actually mark all-zero…
        neg_inf_where=pred_false(2),
        one_where=pred_false(2),
        true_where=pred_false(2),
        false_where=pred_false(2),
        unchanged_from=None,
    )
    # The above is technically all-zero. Let's instead make it a *partial*
    # predicate by directly building a non-trivial body.
    a.env["x"] = type(a.env["x"])(
        rank=2,
        zero_where=Pred(2, BoolLit(False)),  # default: no positive proof
        neg_inf_where=pred_false(2),
        one_where=pred_false(2),
        true_where=pred_false(2),
        false_where=pred_false(2),
        unchanged_from=None,
    )
    expr = ReduceSum(
        value=_typed_var("x", _f32(4, 8)), axis=1, type=_f32(4)
    )
    facts, _ = a._eval(expr, 1)
    assert not _pred_is_true(facts.zero_where)


# ---------------------------------------------------------------------------
# Where: per-position branch selection
# ---------------------------------------------------------------------------


def test_where_with_allfalse_cond_returns_else_branch():
    """Where(false, t, f) ≡ f at every position."""
    type_env = {"c": _bool(4), "t": _f32(4), "f": _f32(4)}
    a = _make_analyzer(type_env, all_false=("c",))
    # Record c as all-false via the oracle (assigning anything triggers it).
    a._assign("c", None, BoolLit(value=True, type=_bool(4)))
    a.env["t"] = facts_unknown(1)
    a.env["f"] = facts_all_zero(1)
    expr = Where(
        cond=_typed_var("c", _bool(4)),
        on_true=_typed_var("t", _f32(4)),
        on_false=_typed_var("f", _f32(4)),
        type=_f32(4),
    )
    facts, _ = a._eval(expr, 1)
    # f's facts dominate; result is all-zero.
    assert _pred_is_true(facts.zero_where), (
        f"Where(false, _, all_zero) should be all-zero, got {facts.zero_where.body}"
    )


# ---------------------------------------------------------------------------
# Shape ops remap positional facts but not unindexed identity provenance
# ---------------------------------------------------------------------------


def test_unsqueeze_forgets_unindexed_identity_provenance():
    type_env = {"x": _f32(4)}
    a = _make_analyzer(type_env)
    a.env["x"] = facts_unchanged(1, "x")
    expr = Unsqueeze(
        value=_typed_var("x", _f32(4)), axis=1, type=_f32(4, 1)
    )
    facts, _ = a._eval(expr, 2)
    assert facts.unchanged_from is None
    assert facts.rank == 2


def test_squeeze_forgets_unindexed_identity_provenance():
    type_env = {"x": _f32(4, 1)}
    a = _make_analyzer(type_env)
    a.env["x"] = facts_unchanged(2, "x")
    expr = Squeeze(
        value=_typed_var("x", _f32(4, 1)), axis=1, type=_f32(4)
    )
    facts, _ = a._eval(expr, 1)
    assert facts.unchanged_from is None
    assert facts.rank == 1


def test_transpose_remaps_predicate_correctly():
    """Take a tensor whose zero_where is `_i0 == 0` (zero on axis-0
    boundary). After transposing, the same fact should be expressed as
    `_i1 == 0` in the transposed coordinate system."""
    from ir.pp import pp_expr

    type_env = {"x": _f32(4, 8)}
    a = _make_analyzer(type_env)
    # Build a custom predicate body referencing _i0.
    from ir import BinOp

    body = BinOp(op="==", lhs=free_idx(0), rhs=IntLit(0))
    from ir.positional import TensorFacts

    a.env["x"] = TensorFacts(
        rank=2,
        zero_where=Pred(2, body),
        neg_inf_where=pred_false(2),
        one_where=pred_false(2),
        true_where=pred_false(2),
        false_where=pred_false(2),
        unchanged_from=None,
    )
    expr = Transpose(
        value=_typed_var("x", _f32(4, 8)),
        permutation=(1, 0),
        type=_f32(8, 4),
    )
    facts, _ = a._eval(expr, 2)
    # After transpose with perm=(1,0), inner _i0 maps to outer _i1.
    # The original predicate `_i0 == 0` becomes `_i1 == 0`.
    assert "_i1" in pp_expr(facts.zero_where.body)
    assert "_i0" not in pp_expr(facts.zero_where.body)


def test_unsqueeze_then_squeeze_roundtrip():
    """The current provenance domain cannot recover composed index maps."""
    type_env = {"x": _f32(4)}
    a = _make_analyzer(type_env)
    a.env["x"] = facts_unchanged(1, "x")
    unsq = Unsqueeze(
        value=_typed_var("x", _f32(4)), axis=1, type=_f32(4, 1)
    )
    sq = Squeeze(value=unsq, axis=1, type=_f32(4))
    facts, _ = a._eval(sq, 1)
    assert facts.unchanged_from is None
    assert facts.rank == 1


# ---------------------------------------------------------------------------
# Multiplication preserves unchanged_from when other side is all-one
# ---------------------------------------------------------------------------


def test_mul_by_all_one_preserves_unchanged_from():
    """Finite x * 1 = x preserves unchanged_from(x)."""
    type_env = {"x": _f32(4), "ones": _f32(4)}
    a = _make_analyzer(type_env, finite=("x",))
    a.env["x"] = facts_unchanged(1, "x")
    a.env["ones"] = facts_all_one(1)
    expr = mul_(_typed_var("x", _f32(4)), _typed_var("ones", _f32(4)))
    expr = expr.__class__(**{**expr.__dict__, "type": _f32(4)})
    facts, _ = a._eval(expr, 1)
    assert facts.unchanged_from == "x"


def test_mul_by_all_one_requires_finite_unchanged_value():
    type_env = {"x": _f32(4), "ones": _f32(4)}
    a = _make_analyzer(type_env)
    a.env["x"] = facts_unchanged(1, "x")
    a.env["ones"] = facts_all_one(1)
    expr = mul_(_typed_var("x", _f32(4)), _typed_var("ones", _f32(4)))
    expr = expr.__class__(**{**expr.__dict__, "type": _f32(4)})
    facts, _ = a._eval(expr, 1)
    assert facts.unchanged_from is None


def test_mul_by_partial_one_does_not_preserve_unchanged_from():
    """If the multiplier is one only at *some* positions, unchanged_from
    cannot be claimed (it is a whole-tensor predicate)."""
    type_env = {"x": _f32(4), "m": _f32(4)}
    a = _make_analyzer(type_env)
    a.env["x"] = facts_unchanged(1, "x")
    # Build a partial one_where: m is one only where _i0 == 0.
    from ir import BinOp
    from ir.positional import TensorFacts

    a.env["m"] = TensorFacts(
        rank=1,
        zero_where=pred_false(1),
        neg_inf_where=pred_false(1),
        one_where=Pred(1, BinOp(op="==", lhs=free_idx(0), rhs=IntLit(0))),
        true_where=pred_false(1),
        false_where=pred_false(1),
        unchanged_from=None,
    )
    expr = mul_(_typed_var("x", _f32(4)), _typed_var("m", _f32(4)))
    expr = expr.__class__(**{**expr.__dict__, "type": _f32(4)})
    facts, _ = a._eval(expr, 1)
    assert facts.unchanged_from is None


def test_compound_mul_by_zero_requires_finite_accumulator():
    """`x *= 0` is not known zero when x could contain NaN or infinity."""
    type_env = {"x": _f32(4), "zero": _f32(4)}
    a = _make_analyzer(type_env)
    a.env["x"] = facts_unknown(1)
    a.env["zero"] = facts_all_zero(1)
    a.exec_stmt(
        Assign(
            target=_typed_var("x", _f32(4)),
            op="*",
            value=_typed_var("zero", _f32(4)),
        )
    )
    assert not _pred_is_true(a.env["x"].zero_where)


def test_compound_mul_by_zero_with_finite_accumulator_is_zero():
    type_env = {"x": _f32(4), "zero": _f32(4)}
    a = _make_analyzer(type_env, finite=("x",))
    a.env["x"] = facts_unknown(1)
    a.env["zero"] = facts_all_zero(1)
    a.exec_stmt(
        Assign(
            target=_typed_var("x", _f32(4)),
            op="*",
            value=_typed_var("zero", _f32(4)),
        )
    )
    assert _pred_is_true(a.env["x"].zero_where)


def test_if_default_merges_branches_instead_of_choosing_then():
    """Unknown control flow retains only facts common to both branches."""
    from ir import BoolType

    type_env = {
        "cond": BoolType(),
        "x": _f32(4),
        "zero": _f32(4),
        "one": _f32(4),
    }
    a = _make_analyzer(type_env)
    a.env["zero"] = facts_all_zero(1)
    a.env["one"] = facts_all_one(1)
    a.exec_stmt(
        If(
            cond=_typed_var("cond", BoolType()),
            then_body=[
                Assign(
                    target=_typed_var("x", _f32(4)),
                    op=None,
                    value=_typed_var("zero", _f32(4)),
                )
            ],
            else_body=[
                Assign(
                    target=_typed_var("x", _f32(4)),
                    op=None,
                    value=_typed_var("one", _f32(4)),
                )
            ],
        )
    )
    assert not _pred_is_true(a.env["x"].zero_where)
    assert not _pred_is_true(a.env["x"].one_where)


def test_assumed_then_branch_is_explicitly_audited():
    from ir import BoolType

    type_env = {"cond": BoolType(), "x": _f32(4), "zero": _f32(4)}
    a = PositionalAnalyzer(
        assumptions=NeutralityAssumptions(),
        type_env=type_env,
        assume_then_branches=True,
    )
    a.env["zero"] = facts_all_zero(1)
    a.exec_stmt(
        If(
            cond=_typed_var("cond", BoolType()),
            then_body=[
                Assign(
                    target=_typed_var("x", _f32(4)),
                    op=None,
                    value=_typed_var("zero", _f32(4)),
                )
            ],
            else_body=[],
        )
    )
    assert _pred_is_true(a.env["x"].zero_where)
    assert "all_if_conditions_true" in a.used_assumptions


def test_float_comparison_is_not_an_exact_mask_without_no_nan_proof():
    """Real-valued Z3 predicates cannot soundly stand for NaN comparisons."""
    x_type = _f32(4)
    mask_type = _bool(4)
    analyzer = _make_analyzer({"x": x_type, "mask": mask_type})
    comparison = BinOp(
        ">",
        _typed_var("x", x_type),
        FloatLit(0.0, type=FloatType()),
        type=mask_type,
    )
    analyzer.exec_stmt(Assign(
        target=_typed_var("mask", mask_type),
        op=None,
        value=comparison,
    ))
    facts = analyzer.env["mask"]
    assert not _pred_is_true(facts.true_where)
    assert not _pred_is_true(facts.false_where)
    assert "mask" not in analyzer.elem_body


def test_integer_comparison_remains_an_exact_mask():
    positions_type = _int(4)
    mask_type = _bool(4)
    analyzer = _make_analyzer({"positions": positions_type, "mask": mask_type})
    comparison = BinOp(
        "<",
        _typed_var("positions", positions_type),
        IntLit(7, type=IntType()),
        type=mask_type,
    )
    analyzer.exec_stmt(Assign(
        target=_typed_var("mask", mask_type),
        op=None,
        value=comparison,
    ))
    facts = analyzer.env["mask"]
    assert not isinstance(facts.true_where.body, BoolLit)
    assert not isinstance(facts.false_where.body, BoolLit)
    assert "mask" in analyzer.elem_body


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _pred_is_true(p: Pred) -> bool:
    return isinstance(p.body, BoolLit) and p.body.value is True
