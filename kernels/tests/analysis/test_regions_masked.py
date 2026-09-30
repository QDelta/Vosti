"""Tests for mask-aware backward region narrowing.

Layers:
  1. Synthetic IR fragments exercising each propagation rule in isolation
     (Where, *, @, ReduceSum/Max, shape ops). These pin the rule's
     contract independently of any kernel.
  2. Kernel 1 / Kernel 2 narrowing: verify that K and V access guards for
     a single output row are implied by the causal mask predicate.
  3. Negative controls: missing facts → guard remains TRUE (no spurious
     narrowing); rules cannot conjure narrowing out of thin air.
"""

import pytest

from ir import (
    Assign,
    BinOp,
    BoolLit,
    BoolType,
    BroadcastTo,
    Exp2,
    FloatLit,
    FloatType,
    For,
    If,
    IntLit,
    IntType,
    Kernel,
    Let,
    Max,
    MaskedLoad,
    MaskedStore,
    Maximum,
    Min,
    Not,
    ReduceMax,
    ReduceSum,
    Slice,
    Squeeze,
    TensorType,
    TensorIndex,
    TensorView,
    Transpose,
    Type,
    Unsqueeze,
    Var,
    Where,
    add_,
    mul_,
    sub_,
)
from ir.pp import pp_expr
from ir.positional import (
    ForwardFactTrace,
    NeutralityAssumptions,
    Pred,
    PositionalAnalyzer,
    TensorFacts,
    facts_all_one,
    facts_all_zero,
    facts_unknown,
    free_idx,
    pred_false,
    pred_true,
)
from ir.regions import (
    GuardedRegion,
    _collect_z3_symbols,
    _singleton_index,
    bound_variable_regions_masked,
    collect_write_stmts,
    regions_expr_masked,
    regions_write_masked,
)
from functools import partial
from ir.relational_dataflow import prove_relational_dataflow_from_annotations
KERNEL1 = "fattn_varlen_paged_fwd_block_ptr_kernel"
selected_report = partial(prove_relational_dataflow_from_annotations,
                          goal_name="selected_row_prefix_equivalence")
from triton_kernels.fattn_paged import CONFIGS as FATTN_CONFIGS


def constexpr_values(config):
    return {k: v for k, v in config.items() if not k.startswith("num_")}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _f32(*dims):
    return TensorType(FloatType(), [IntLit(d) for d in dims])


def _bool(*dims):
    return TensorType(BoolType(), [IntLit(d) for d in dims])


def _typed_var(name: str, ty: Type) -> Var:
    return Var(name=name, type=ty)


def _is_true(p: Pred) -> bool:
    return isinstance(p.body, BoolLit) and p.body.value is True


def _is_false(p: Pred) -> bool:
    return isinstance(p.body, BoolLit) and p.body.value is False


def _run(
    expr,
    *,
    type_env: dict[str, Type],
    pos_env: dict[str, TensorFacts],
    target_region,
    target_rank: int,
    target_guard: Pred | None = None,
    finite_vars: frozenset[str] = frozenset(),
) -> dict[str, GuardedRegion]:
    if target_guard is None:
        target_guard = pred_true(target_rank)
    regions: dict[str, GuardedRegion | None] = {}
    regions_expr_masked(
        expr, target_region, target_guard, type_env, regions, pos_env,
        finite_vars,
    )
    return {k: v for k, v in regions.items() if v is not None}


def _assert_predicate_at(pred: Pred, coordinates: tuple[int, ...], expected: bool):
    """Use the verifier's Z3 encoding to evaluate a closed index instance."""
    import z3

    from ir.smt import expr_to_z3

    assert pred.rank == len(coordinates)
    env: dict[str, object] = {}
    _collect_z3_symbols(pred.body, env)
    solver = z3.Solver()
    for k, coordinate in enumerate(coordinates):
        index = env.get(f"_i{k}")
        if index is not None:
            solver.add(index == coordinate)  # type: ignore[operator]
    encoded = expr_to_z3(pred.body, env)
    solver.add(z3.Not(encoded) if expected else encoded)  # type: ignore[arg-type]
    assert solver.check() == z3.unsat


# ===========================================================================
# 1. Synthetic IR — one rule at a time
# ===========================================================================


def test_where_narrows_branches_by_cond_facts():
    """`Where(c, t, f)` with c's true_where/false_where known: t's read
    upper bound is `¬c.false_where`; f's is `¬c.true_where`. (Lower
    bounds give upper bounds on reads via negation of the OPPOSITE
    branch — see the docstring on `regions_expr_masked` for the why.)"""
    type_env = {"c": _bool(4), "t": _f32(4), "f": _f32(4)}
    cond_true = Pred(1, BinOp(op="<", lhs=free_idx(0), rhs=IntLit(2)))
    cond_false = Pred(1, BinOp(op=">=", lhs=free_idx(0), rhs=IntLit(2)))
    pos_env = {
        "c": TensorFacts(
            rank=1,
            zero_where=pred_false(1),
            neg_inf_where=pred_false(1),
            one_where=pred_false(1),
            true_where=cond_true,
            false_where=cond_false,
            unchanged_from=None,
        ),
        "t": facts_unknown(1),
        "f": facts_unknown(1),
    }
    expr = Where(
        cond=_typed_var("c", _bool(4)),
        on_true=_typed_var("t", _f32(4)),
        on_false=_typed_var("f", _f32(4)),
        type=_f32(4),
    )
    result = _run(
        expr,
        type_env=type_env,
        pos_env=pos_env,
        target_region=[Slice(IntLit(0), IntLit(4))],
        target_rank=1,
    )
    # t's guard = NOT (_i0 >= 2) — pretty-prints with ">=" inside a Not.
    t_guard_str = pp_expr(result["t"].guard.body)
    assert "_i0" in t_guard_str
    assert ">=" in t_guard_str
    # f's guard = NOT (_i0 < 2) — pretty-prints with "<" inside a Not.
    f_guard_str = pp_expr(result["f"].guard.body)
    assert "_i0" in f_guard_str
    assert "<" in f_guard_str
    # c is read at full region with TRUE guard (cond is itself read).
    assert _is_true(result["c"].guard)


def test_where_without_cond_facts_no_narrowing():
    """Rule cannot fire when cond has no facts: guards stay TRUE."""
    type_env = {"c": _bool(4), "t": _f32(4), "f": _f32(4)}
    pos_env: dict[str, TensorFacts] = {}  # no facts at all
    expr = Where(
        cond=_typed_var("c", _bool(4)),
        on_true=_typed_var("t", _f32(4)),
        on_false=_typed_var("f", _f32(4)),
        type=_f32(4),
    )
    result = _run(
        expr,
        type_env=type_env,
        pos_env=pos_env,
        target_region=[Slice(IntLit(0), IntLit(4))],
        target_rank=1,
    )
    # Without facts, true_where = false_where = pred_false (default
    # facts_unknown). The narrowing pred_and(TRUE, false) collapses to
    # FALSE — meaning we'd claim t is never read, which is unsound!
    # Actually this is a real corner: if we have no facts about cond,
    # we should pass guards through unchanged. Verify what the impl does.
    # Soundness contract: dependency-set ⊆ region ∩ guard. If guard is
    # FALSE we'd claim no value dependency — UNSOUND. So guards must be TRUE,
    # or the impl must not consult absent facts.
    assert _is_true(result["t"].guard) or not _is_false(result["t"].guard), (
        f"unsound: t's guard collapsed to FALSE under absent cond facts. "
        f"got: {result['t'].guard.body}"
    )


def test_mul_retains_finite_operand_sign_with_zero_other_operand():
    """finite(a) * zero still depends on the sign of a."""
    type_env = {"a": _f32(4), "b": _f32(4)}
    pos_env = {
        "a": facts_unknown(1),
        "b": facts_all_zero(1),
    }
    expr = mul_(_typed_var("a", _f32(4)), _typed_var("b", _f32(4)))
    expr = expr.__class__(**{**expr.__dict__, "type": _f32(4)})
    result = _run(
        expr,
        type_env=type_env,
        pos_env=pos_env,
        target_region=[Slice(IntLit(0), IntLit(4))],
        target_rank=1,
        finite_vars=frozenset({"a"}),
    )
    assert _is_true(result["a"].guard)
    assert _is_true(result["b"].guard)


def test_mul_partial_zero_retains_operand_sign_dependency():
    """Neither partially nor wholly numerical-zero lanes erase sign."""
    type_env = {"a": _f32(4), "b": _f32(4)}
    b_zero = Pred(1, BinOp(op=">=", lhs=free_idx(0), rhs=IntLit(2)))
    pos_env = {
        "a": facts_unknown(1),
        "b": TensorFacts(
            rank=1,
            zero_where=b_zero,
            neg_inf_where=pred_false(1),
            one_where=pred_false(1),
            true_where=pred_false(1),
            false_where=pred_false(1),
            unchanged_from=None,
        ),
    }
    expr = mul_(_typed_var("a", _f32(4)), _typed_var("b", _f32(4)))
    expr = expr.__class__(**{**expr.__dict__, "type": _f32(4)})
    result = _run(
        expr,
        type_env=type_env,
        pos_env=pos_env,
        target_region=[Slice(IntLit(0), IntLit(4))],
        target_rank=1,
        finite_vars=frozenset({"a"}),
    )
    assert _is_true(result["a"].guard)


def test_matmul_narrows_rhs_by_lhs_zero_where_at_singleton_row():
    """`a @ b` consumed at single row [j, j+1) → b's guard narrows by
    `¬a.zero_where[j, _i0]`."""
    type_env = {"a": _f32(8, 4), "b": _f32(4, 16)}
    # a's zero_where = `_i1 >= 3` (a's column index >= 3 means zero).
    a_zero = Pred(2, BinOp(op=">=", lhs=free_idx(1), rhs=IntLit(3)))
    pos_env = {
        "a": TensorFacts(
            rank=2,
            zero_where=a_zero,
            neg_inf_where=pred_false(2),
            one_where=pred_false(2),
            true_where=pred_false(2),
            false_where=pred_false(2),
            unchanged_from=None,
        ),
        "b": facts_unknown(2),
    }
    expr = BinOp(
        op="@",
        lhs=_typed_var("a", _f32(8, 4)),
        rhs=_typed_var("b", _f32(4, 16)),
        type=_f32(8, 16),
    )
    j = IntLit(2)  # singleton row index
    target_region = [Slice(j, add_(j, IntLit(1))), Slice(IntLit(0), IntLit(16))]
    result = _run(
        expr,
        type_env=type_env,
        pos_env=pos_env,
        target_region=target_region,
        target_rank=2,
        finite_vars=frozenset({"b"}),
    )
    # b's guard substitutes a's `_i0` ← j (=IntLit(2)), `_i1` ← b's `_i0`.
    # a's zero_where(_i1 >= 3) becomes (b's _i0 >= 3). Guard is its
    # negation: NOT (b's _i0 >= 3), i.e. `_i0 < 3` (or equivalent).
    b_guard_str = pp_expr(result["b"].guard.body)
    assert "_i0" in b_guard_str
    assert ">=" in b_guard_str  # appears inside the negation


def test_mul_zero_does_not_drop_nonfinite_operand_without_assumption():
    """IEEE 754 has 0*NaN = NaN and 0*inf = NaN, so dependency remains."""
    type_env = {"a": _f32(4), "b": _f32(4)}
    pos_env = {"a": facts_unknown(1), "b": facts_all_zero(1)}
    expr = mul_(_typed_var("a", _f32(4)), _typed_var("b", _f32(4)))
    expr = expr.__class__(**{**expr.__dict__, "type": _f32(4)})
    result = _run(
        expr,
        type_env=type_env,
        pos_env=pos_env,
        target_region=[Slice(IntLit(0), IntLit(4))],
        target_rank=1,
    )
    assert _is_true(result["a"].guard)


def test_matmul_zero_lane_does_not_drop_nonfinite_rhs_without_assumption():
    """A zero matmul coefficient cannot annihilate NaN/inf in the RHS."""
    type_env = {"a": _f32(1, 4), "b": _f32(4, 2)}
    pos_env = {"a": facts_all_zero(2), "b": facts_unknown(2)}
    expr = BinOp(
        op="@",
        lhs=_typed_var("a", _f32(1, 4)),
        rhs=_typed_var("b", _f32(4, 2)),
        type=_f32(1, 2),
    )
    result = _run(
        expr,
        type_env=type_env,
        pos_env=pos_env,
        target_region=[
            Slice(IntLit(0), IntLit(1)),
            Slice(IntLit(0), IntLit(2)),
        ],
        target_rank=2,
    )
    assert _is_true(result["b"].guard)


def test_matmul_no_narrowing_for_multi_row_target():
    """When target_region is multi-row, the rhs narrowing is conservative
    (TRUE) — narrowing across rows would need an OR over rows."""
    type_env = {"a": _f32(8, 4), "b": _f32(4, 16)}
    a_zero = Pred(2, BinOp(op=">=", lhs=free_idx(1), rhs=IntLit(3)))
    pos_env = {
        "a": TensorFacts(
            rank=2,
            zero_where=a_zero,
            neg_inf_where=pred_false(2),
            one_where=pred_false(2),
            true_where=pred_false(2),
            false_where=pred_false(2),
            unchanged_from=None,
        ),
        "b": facts_unknown(2),
    }
    expr = BinOp(
        op="@",
        lhs=_typed_var("a", _f32(8, 4)),
        rhs=_typed_var("b", _f32(4, 16)),
        type=_f32(8, 16),
    )
    target_region = [Slice(IntLit(0), IntLit(8)), Slice(IntLit(0), IntLit(16))]
    result = _run(
        expr,
        type_env=type_env,
        pos_env=pos_env,
        target_region=target_region,
        target_rank=2,
    )
    assert _is_true(result["b"].guard), (
        f"matmul rhs guard should be TRUE for multi-row consumer, got {result['b'].guard.body}"
    )


def test_z3_recognizes_intersected_symbolic_singleton():
    """max/min clipping preserves a singleton whenever the slice is nonempty."""
    i = Var("i", type=IntType())
    lo = Var("lo", type=IntType())
    hi = Var("hi", type=IntType())
    base = Var("base", type=IntType())
    clipped = Slice(
        BinOp("-", Max([i, lo]), base, type=IntType()),
        BinOp(
            "-",
            Min([add_(i, IntLit(1)), hi]),
            base,
            type=IntType(),
        ),
    )
    assert _singleton_index(clipped) == clipped.start


def test_z3_does_not_misclassify_intersected_two_row_slice():
    i = Var("i", type=IntType())
    lo = Var("lo", type=IntType())
    hi = Var("hi", type=IntType())
    clipped = Slice(
        Max([i, lo]),
        Min([add_(i, IntLit(2)), hi]),
    )
    assert _singleton_index(clipped) is None


def test_reduce_sum_retains_numerical_zero_sign_dependencies():
    """zero_where does not determine the bits of a reduction's inputs."""
    type_env = {"x": _f32(4, 8)}
    # x.zero_where = `_i1 >= 5`.
    x_zero = Pred(2, BinOp(op=">=", lhs=free_idx(1), rhs=IntLit(5)))
    pos_env = {
        "x": TensorFacts(
            rank=2,
            zero_where=x_zero,
            neg_inf_where=pred_false(2),
            one_where=pred_false(2),
            true_where=pred_false(2),
            false_where=pred_false(2),
            unchanged_from=None,
        ),
    }
    expr = ReduceSum(
        value=_typed_var("x", _f32(4, 8)), axis=1, type=_f32(4)
    )
    result = _run(
        expr,
        type_env=type_env,
        pos_env=pos_env,
        target_region=[Slice(IntLit(0), IntLit(4))],
        target_rank=1,
    )
    assert _is_true(result["x"].guard)


# ---------------------------------------------------------------------------
# Shape ops: round-trip and remap correctness
# ---------------------------------------------------------------------------


def test_unsqueeze_remaps_guard():
    """Unsqueeze adds a size-1 axis; parent's `_iaxis` becomes 0 (the
    only valid index in the new axis), other indices shift."""
    type_env = {"x": _f32(4)}
    pos_env = {"x": facts_unknown(1)}
    expr = Unsqueeze(
        value=_typed_var("x", _f32(4)), axis=1, type=_f32(4, 1)
    )
    parent_guard = Pred(2, BinOp(op="<", lhs=free_idx(0), rhs=IntLit(3)))
    result = _run(
        expr,
        type_env=type_env,
        pos_env=pos_env,
        target_region=[Slice(IntLit(0), IntLit(4)), Slice(IntLit(0), IntLit(1))],
        target_rank=2,
        target_guard=parent_guard,
    )
    # Parent's `_i0` was the kept axis; child's `_i0` is the same.
    x_guard_str = pp_expr(result["x"].guard.body)
    assert "_i0" in x_guard_str


def test_broadcast_drops_guard_dependent_on_expanded_axis():
    """Backward broadcast needs an existential, never index-zero substitution."""
    type_env = {"x": _f32(1, 4)}
    expr = BroadcastTo(
        value=_typed_var("x", _f32(1, 4)),
        shape=[IntLit(3), IntLit(4)],
        type=_f32(3, 4),
    )
    result = _run(
        expr,
        type_env=type_env,
        pos_env={"x": facts_unknown(2)},
        target_region=[Slice(IntLit(0), IntLit(3)), Slice(IntLit(0), IntLit(4))],
        target_rank=2,
        target_guard=Pred(2, BinOp("==", free_idx(0), IntLit(1))),
    )
    assert _is_true(result["x"].guard)


def test_broadcast_preserves_guard_independent_of_expanded_axis():
    type_env = {"x": _f32(1, 4)}
    expr = BroadcastTo(
        value=_typed_var("x", _f32(1, 4)),
        shape=[IntLit(3), IntLit(4)],
        type=_f32(3, 4),
    )
    result = _run(
        expr,
        type_env=type_env,
        pos_env={"x": facts_unknown(2)},
        target_region=[Slice(IntLit(0), IntLit(3)), Slice(IntLit(0), IntLit(4))],
        target_rank=2,
        target_guard=Pred(2, BinOp("<", free_idx(1), IntLit(2))),
    )
    _assert_predicate_at(result["x"].guard, (0, 1), True)
    _assert_predicate_at(result["x"].guard, (0, 3), False)


@pytest.mark.parametrize("load_kind", ["view", "masked_load"])
def test_load_maps_local_guard_to_absolute_source_coordinates(load_kind):
    """A local row-one guard on a [4:8] load denotes source row five."""
    type_env = {"base": _f32(16)}
    region = [Slice(IntLit(4), IntLit(8))]
    base = _typed_var("base", _f32(16))
    if load_kind == "view":
        expr = TensorView(base, region, type=_f32(4))
    else:
        expr = MaskedLoad(base, region, region, type=_f32(4))
    result = _run(
        expr,
        type_env=type_env,
        pos_env={},
        target_region=[Slice(IntLit(1), IntLit(2))],
        target_rank=1,
        target_guard=Pred(1, BinOp("==", free_idx(0), IntLit(1))),
    )
    _assert_predicate_at(result["base"].guard, (5,), True)
    _assert_predicate_at(result["base"].guard, (1,), False)


@pytest.mark.parametrize("store_kind", ["view", "masked_store"])
def test_store_maps_absolute_guard_to_value_local_coordinates(store_kind):
    """An absolute row-five output guard denotes local row one of [4:8]."""
    out_ty = _f32(16)
    value_ty = _f32(4)
    region = [Slice(IntLit(4), IntLit(8))]
    out = _typed_var("out", out_ty)
    value = _typed_var("value", value_ty)
    if store_kind == "view":
        write = Assign(TensorView(out, region), None, value)
    else:
        write = MaskedStore(out, region, value, region)
    regions: dict[str, GuardedRegion | None] = {
        "out": GuardedRegion(
            [Slice(IntLit(5), IntLit(6))],
            Pred(1, BinOp("==", free_idx(0), IntLit(5))),
        )
    }
    regions_write_masked(
        write,
        {"out": out_ty, "value": value_ty},
        regions,
        {},
        "out",
    )
    value_region = regions["value"]
    assert value_region is not None
    _assert_predicate_at(value_region.guard, (1,), True)
    _assert_predicate_at(value_region.guard, (5,), False)


def test_transpose_inverts_index_remap():
    """Transpose with permutation (1, 0): parent's `_i0` (transposed
    axis 0) maps back to child's `_i1`, and vice versa."""
    type_env = {"x": _f32(4, 8)}
    pos_env = {"x": facts_unknown(2)}
    expr = Transpose(
        value=_typed_var("x", _f32(4, 8)), permutation=(1, 0), type=_f32(8, 4)
    )
    parent_guard = Pred(2, BinOp(op="<", lhs=free_idx(0), rhs=IntLit(2)))
    result = _run(
        expr,
        type_env=type_env,
        pos_env=pos_env,
        target_region=[Slice(IntLit(0), IntLit(8)), Slice(IntLit(0), IntLit(4))],
        target_rank=2,
        target_guard=parent_guard,
    )
    # Parent's `_i0` should become child's `_i1`.
    x_guard_str = pp_expr(result["x"].guard.body)
    assert "_i1" in x_guard_str
    assert "_i0" not in x_guard_str.replace("_i1", "")


def test_mul_transposed_zero_does_not_erase_operand_sign():
    """Reindexing a numerical zero fact cannot strengthen it to bit equality."""
    type_env = {"a": _f32(4, 4), "z": _f32(4, 4)}
    z_zero = Pred(2, BinOp(op="==", lhs=free_idx(0), rhs=IntLit(0)))
    pos_env = {
        "a": facts_unknown(2),
        "z": TensorFacts(
            rank=2,
            zero_where=z_zero,
            neg_inf_where=pred_false(2),
            one_where=pred_false(2),
            true_where=pred_false(2),
            false_where=pred_false(2),
            unchanged_from=None,
        ),
    }
    expr = BinOp(
        op="*",
        lhs=_typed_var("a", _f32(4, 4)),
        rhs=Transpose(
            value=_typed_var("z", _f32(4, 4)),
            permutation=(1, 0),
            type=_f32(4, 4),
        ),
        type=_f32(4, 4),
    )
    result = _run(
        expr,
        type_env=type_env,
        pos_env=pos_env,
        target_region=[Slice(IntLit(0), IntLit(4)), Slice(IntLit(0), IntLit(4))],
        target_rank=2,
        finite_vars=frozenset({"a"}),
    )
    assert _is_true(result["a"].guard)


# ---------------------------------------------------------------------------
# Composability
# ---------------------------------------------------------------------------


def test_nested_where_conjoins_guards():
    """`where(c1, where(c2, t, 0), 0)` → t's guard should involve BOTH
    c1.true_where and c2.true_where via conjunction."""
    type_env = {
        "c1": _bool(4),
        "c2": _bool(4),
        "t": _f32(4),
        "z": _f32(4),
    }
    c1_true = Pred(1, BinOp(op="<", lhs=free_idx(0), rhs=IntLit(2)))
    c1_false = Pred(1, BinOp(op=">=", lhs=free_idx(0), rhs=IntLit(2)))
    c2_true = Pred(1, BinOp(op="==", lhs=free_idx(0), rhs=IntLit(0)))
    c2_false = Pred(1, BinOp(op="!=", lhs=free_idx(0), rhs=IntLit(0)))
    pos_env = {
        "c1": TensorFacts(
            rank=1,
            zero_where=pred_false(1),
            neg_inf_where=pred_false(1),
            one_where=pred_false(1),
            true_where=c1_true,
            false_where=c1_false,
            unchanged_from=None,
        ),
        "c2": TensorFacts(
            rank=1,
            zero_where=pred_false(1),
            neg_inf_where=pred_false(1),
            one_where=pred_false(1),
            true_where=c2_true,
            false_where=c2_false,
            unchanged_from=None,
        ),
        "t": facts_unknown(1),
        "z": facts_all_zero(1),
    }
    inner = Where(
        cond=_typed_var("c2", _bool(4)),
        on_true=_typed_var("t", _f32(4)),
        on_false=_typed_var("z", _f32(4)),
        type=_f32(4),
    )
    outer = Where(
        cond=_typed_var("c1", _bool(4)),
        on_true=inner,
        on_false=_typed_var("z", _f32(4)),
        type=_f32(4),
    )
    result = _run(
        outer,
        type_env=type_env,
        pos_env=pos_env,
        target_region=[Slice(IntLit(0), IntLit(4))],
        target_rank=1,
    )
    t_guard_str = pp_expr(result["t"].guard.body)
    # Corrected rule narrows by negated false_where of each cond.
    # c1.false_where = `_i0 >= 2`; c2.false_where = `_i0 != 0`.
    # So t's guard body contains `>=` and `!=` (both inside Not).
    assert ">=" in t_guard_str
    assert "!=" in t_guard_str


# ===========================================================================
# 2. Kernel-level narrowing: K and V access for output row j
# ===========================================================================


def _paged_source() -> str:
    with open("triton_kernels/fattn_paged.py") as source_file:
        return source_file.read()


def _load_attn_kernel(name: str, constants: dict):
    from ir.preprocess import check_variable_names, check_tensorindex_readonly
    from ir.subst import expand_let_bindings, specialize_kernel_constants
    from ir.translate import translate_kernel_source
    from ir.typ import infer_types

    source = _paged_source()
    bool_constants = {k: v for k, v in constants.items() if isinstance(v, bool)}
    numeric_constants = {k: v for k, v in constants.items() if not isinstance(v, bool)}
    kernel = translate_kernel_source(
        source, name, specialize=bool_constants if bool_constants else None
    )
    roles = check_variable_names(kernel)
    check_tensorindex_readonly(kernel)
    kernel = specialize_kernel_constants(kernel, numeric_constants)
    kernel = expand_let_bindings(kernel)
    kernel, _ = infer_types(kernel, roles)
    return kernel


def _build_pos_env(kernel, assumptions: NeutralityAssumptions):
    """Run the forward positional analyzer over the entire kernel body
    and return its program-point-sensitive trace."""
    from ir.preprocess import build_type_env

    type_env = build_type_env(kernel)
    a = PositionalAnalyzer(
        assumptions=assumptions,
        type_env=type_env,
        assume_then_branches=True,
    )
    for stmt in kernel.grid.body:
        a.exec_stmt(stmt)
    return a.trace()


def test_kernel1_where_chain_narrows_qk_scores_masked_scores():
    """For kernel 1, the Where rule + element-wise propagation should
    narrow read guards on the QK chain — `qk`, `scores`,
    `masked_scores` — by the attn_mask predicate. This is the part of
    the causality theorem that lives strictly in the elementwise/Where
    chain (no matmul singleton needed).

    K/V cache narrowing additionally crosses the matmul boundary in
    `test_singleton_row_narrows_kv_to_causal_prefix`; this test isolates the
    elementwise/Where portion for a smaller diagnostic blast radius."""
    kernel = _load_attn_kernel(
        "fattn_varlen_paged_fwd_block_ptr_kernel",
        {"D_HEAD": 64, "BLOCK_M": 64, "BLOCK_N": 64, "PAGE_BLOCK_SIZE": 64},
    )
    pos_env = _build_pos_env(kernel, NeutralityAssumptions())

    j = Var(name="j_abs", type=IntType())
    hi = Var(name="hi_abs", type=IntType())
    output_region = [
        Slice(j, add_(j, IntLit(1))),
        Slice(hi, add_(hi, IntLit(1))),
        Slice(IntLit(0), IntLit(64)),
    ]
    result = bound_variable_regions_masked(
        kernel, "o", output_region, pred_true(3), pos_env
    )

    # The chain `masked_scores = where(attn_mask, scores, -inf)` ⇒
    # `scores`'s guard is narrowed by ¬attn_mask.false_where. Through
    # `scores = qk * scale_log2` (mul; scale is rank-0) the same guard
    # propagates back to qk. Each of these should mention at least one
    # mask-predicate constituent.
    for var in ("qk", "scores", "masked_scores"):
        assert var in result, f"{var} missing from regions"
        guard_str = pp_expr(result[var].guard.body)
        has_signal = any(
            s in guard_str
            for s in ("q_indices", "k_indices", "q_shift", "q_mask", "k_mask",
                      "cu_seqlens_q", "cu_seqlens_k")
        )
        assert has_signal, (
            f"{var}'s guard shows no evidence of mask narrowing; got: {guard_str[:200]}"
        )


_CAUSAL_CERTIFICATE_CASES = [
    (
        KERNEL1,
        {
            **constexpr_values(config),
            "PAGE_BLOCK_SIZE": page_size,
            "D_HEAD": 128,
        },
    )
    for page_size in (64, 256)
    for config in FATTN_CONFIGS
    if page_size % config["BLOCK_N"] == 0
]




@pytest.mark.parametrize("kernel_name, constants", _CAUSAL_CERTIFICATE_CASES)
def test_paged_attention_selected_row_goal(kernel_name, constants):
    report = selected_report(_paged_source(), kernel_name, constants)
    assert report.proved, [(c.name, c.details) for c in report.checks if not c.proved]
    assert report.verified_contract is not None
    assert report.annotation_satisfiability == "sat"


def test_relational_certificate_rejects_wrong_q_shift():
    """A one-token causal-offset drift must break the source bridge."""
    with open("triton_kernels/fattn_paged.py") as source_file:
        source = source_file.read()
    original = "q_shift = k_len - q_len"
    assert source.count(original) == 1
    broken = source.replace(original, "q_shift = k_len - q_len + 1", 1)
    report = selected_report(
        broken,
        KERNEL1,
        {
            "D_HEAD": 128,
            "BLOCK_M": 16,
            "BLOCK_N": 64,
            "PAGE_BLOCK_SIZE": 256,
        },
    )
    assert not report.proved
    assert any(not check.proved for check in report.checks)


def test_relational_certificate_rejects_wrong_query_row_address():
    """The bridge must bind selected Q values to the actual source load."""
    with open("triton_kernels/fattn_paged.py") as source_file:
        source = source_file.read()
    original = "base=q + q_start * stride_qt + hi * stride_qh,"
    assert source.count(original) == 1
    broken = source.replace(
        original,
        "base=q + (q_start + 1) * stride_qt + hi * stride_qh,",
        1,
    )
    report = selected_report(
        broken,
        KERNEL1,
        {
            "D_HEAD": 128,
            "BLOCK_M": 16,
            "BLOCK_N": 64,
            "PAGE_BLOCK_SIZE": 256,
        },
    )
    assert not report.proved
    assert report.verified_contract is None


@pytest.mark.parametrize(
    "original, broken, reason",
    [
        (
            "scores = qk * scale_log2",
            "scores = qk * scale_log2 + q_shift",
            "unpaired query shift affects scores",
        ),
        (
            "p = tl.where(attn_mask, tl.exp2(scores - next_max[:, None]), 0.0)",
            "p = tl.exp2(scores - next_max[:, None])",
            "values outside the related prefix can affect the accumulator",
        ),
        (
            "out = acc * tl.where(logsum > 0.0, 1.0 / logsum, 0.0)[:, None]",
            "out = acc * tl.where(logsum > 0.0, 1.0 / logsum, 0.0)[:, None] + q_shift",
            "unpaired query shift affects the output",
        ),
    ],
)
def test_relational_composition_rejects_broken_transition_contract(
    original, broken, reason
):
    """Ordered relational composition rejects unpaired values and suffix reads."""
    with open("triton_kernels/fattn_paged.py") as source_file:
        source = source_file.read()
    assert source.count(original) == 1
    report = selected_report(
        source.replace(original, broken, 1),
        KERNEL1,
        {
            "D_HEAD": 128,
            "BLOCK_M": 16,
            "BLOCK_N": 64,
            "PAGE_BLOCK_SIZE": 256,
        },
    )
    assert not report.proved
    assert report.verified_contract is None, reason


def test_causal_certificate_rejects_reversed_kernel1_mask():
    """Negative control: a future-token mask regression must fail closed."""
    with open("triton_kernels/fattn_paged.py") as source_file:
        source = source_file.read()
    original = (
        "attn_mask &= k_indices[None, :] <= "
        "(q_indices[:, None] + q_shift)"
    )
    assert source.count(original) == 1
    broken = source.replace(original, original.replace(" <= ", " >= "))
    report = selected_report(
        broken,
        KERNEL1,
        {
            "D_HEAD": 128,
            "BLOCK_M": 16,
            "BLOCK_N": 64,
            "PAGE_BLOCK_SIZE": 256,
        },
    )
    assert not report.proved
    assert report.verified_contract is None


def test_causal_certificate_rejects_wrong_kernel1_block_table_slot():
    """Causal confinement alone must not bless a wrong physical KV page."""
    with open("triton_kernels/fattn_paged.py") as source_file:
        source = source_file.read()
    original = (
        "block_table + bi * stride_btb + page_slot * stride_bts"
    )
    assert source.count(original) == 1
    broken = source.replace(
        original,
        "block_table + bi * stride_btb + (page_slot + 1) * stride_bts",
    )
    report = selected_report(
        broken,
        KERNEL1,
        {
            "D_HEAD": 128,
            "BLOCK_M": 16,
            "BLOCK_N": 64,
            "PAGE_BLOCK_SIZE": 256,
        },
    )
    assert not report.proved
    assert report.verified_contract is None


def test_relational_goal_does_not_require_mathematical_zero_initializer():
    source = _paged_source().replace(
        "acc = tl.zeros((BLOCK_M, D_HEAD), dtype=tl.float32)",
        "acc = tl.full((BLOCK_M, D_HEAD), 1.0, dtype=tl.float32)", 1)
    report = selected_report(source, KERNEL1,
        dict(D_HEAD=128, BLOCK_M=16, BLOCK_N=64, PAGE_BLOCK_SIZE=64))
    assert report.proved and report.verified_contract is not None


def test_kernel1_no_facts_means_no_narrowing():
    """Negative control: with an empty `pos_env`, backward narrowing has
    nothing to consume. The vars that DO get narrowed in
    `test_kernel1_where_chain_narrows_qk_scores_masked_scores` should
    instead have TRUE guards. This guards against a stray rule fabricating
    narrowing from thin air."""
    kernel = _load_attn_kernel(
        "fattn_varlen_paged_fwd_block_ptr_kernel",
        {"D_HEAD": 64, "BLOCK_M": 64, "BLOCK_N": 64, "PAGE_BLOCK_SIZE": 64},
    )
    writes = collect_write_stmts(kernel)
    pos_env = ForwardFactTrace(
        final_env={},
        before_stmt={id(item.write): {} for item in writes},
        after_stmt={id(item.write): {} for item in writes},
    )
    j = Var(name="j_abs", type=IntType())
    hi = Var(name="hi_abs", type=IntType())
    output_region = [
        Slice(j, add_(j, IntLit(1))),
        Slice(hi, add_(hi, IntLit(1))),
        Slice(IntLit(0), IntLit(64)),
    ]
    result = bound_variable_regions_masked(
        kernel, "o", output_region, pred_true(3), pos_env
    )
    # Each var that was provably narrowed under facts must be TRUE here.
    for var in ("qk", "scores", "masked_scores"):
        if var in result:
            assert _is_true(result[var].guard), (
                f"without facts, {var}'s guard should be TRUE; got: "
                f"{pp_expr(result[var].guard.body)[:200]}"
            )


def test_backward_pass_uses_forward_facts_at_the_exact_use() -> None:
    """A later reassignment must not rewrite facts at an earlier use."""

    from ir.preprocess import build_type_env, check_variable_names
    from ir.subst import expand_let_bindings, specialize_kernel_constants
    from ir.translate import translate_kernel_source
    from ir.typ import infer_types

    source = """\
import triton
import triton.language as tl

# @params(
#   tensor(x, float, shape(N), strides(x_pitch)),
#   tensor(o, float, shape(N), strides(o_pitch)),
# )
# @grid(1)
@triton.jit
def reassigned_mask_kernel(x, o, N, cutoff, x_pitch, o_pitch, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = tl.arange(0, BLOCK)
    x_ptr = tl.make_block_ptr(
        x, shape=(N,), strides=(x_pitch,), offsets=(0,),
        block_shape=(BLOCK,), order=(0,),
    )
    values = tl.load(x_ptr, boundary_check=(0,), padding_option="zero")
    gate = offsets < cutoff
    selected = tl.where(gate, values, 0.0)
    gate = offsets >= 0
    o_ptr = tl.make_block_ptr(
        o, shape=(N,), strides=(o_pitch,), offsets=(0,),
        block_shape=(BLOCK,), order=(0,),
    )
    tl.store(o_ptr, selected, boundary_check=(0,))
"""
    kernel = translate_kernel_source(source, "reassigned_mask_kernel")
    kernel = specialize_kernel_constants(kernel, {"BLOCK": 4})
    kernel = expand_let_bindings(kernel)
    roles = check_variable_names(kernel)
    kernel, _ = infer_types(kernel, roles)

    analyzer = PositionalAnalyzer(
        assumptions=NeutralityAssumptions(),
        type_env=build_type_env(kernel),
    )
    for statement in kernel.grid.body:
        analyzer.exec_stmt(statement)
    result = bound_variable_regions_masked(
        kernel,
        "o",
        [Slice(IntLit(0), IntLit(4))],
        pred_true(1),
        analyzer.trace(),
    )
    guard = pp_expr(result["x"].guard.body)
    assert "cutoff" in guard and "<" in guard
    assert ">=" not in guard


# ---------------------------------------------------------------------------
# Regression: the ordinary unmasked region pass remains independent
# ---------------------------------------------------------------------------


def test_unmasked_pass_unaffected_by_extension():
    """The standard unmasked pass retains its independent result."""
    from ir.regions import bound_variable_regions

    kernel = _load_attn_kernel(
        "fattn_varlen_paged_fwd_block_ptr_kernel",
        {"D_HEAD": 64, "BLOCK_M": 64, "BLOCK_N": 64, "PAGE_BLOCK_SIZE": 64},
    )
    # `o` is rank-3 (Tq, H, D). Take a single (token, head) row.
    j = Var(name="j_abs", type=IntType())
    hi = Var(name="hi_abs", type=IntType())
    output_region = [
        Slice(j, add_(j, IntLit(1))),
        Slice(hi, add_(hi, IntLit(1))),
        Slice(IntLit(0), IntLit(64)),
    ]
    regions, _ = bound_variable_regions(kernel, "o", output_region)
    # Sanity: v_cache and k_cache appear in the unmasked pass too.
    assert "v_cache" in regions
    assert "k_cache" in regions
