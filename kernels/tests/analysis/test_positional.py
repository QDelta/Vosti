"""Tests for the per-position forward analyzer (`ir/positional.py`).

Covers two pieces of functionality on top of the same analyzer:

  * **Per-position predicate tracking** — after running PositionalAnalyzer
    on the ki / ti loop body, `attn_mask` has a recognizable `true_where`
    predicate, and `p` has `zero_where` covering positions where the mask
    is false. These are the load-bearing facts for the upcoming
    backward-region extension that will narrow K/V accesses.

  * **Accumulator-neutrality lemma** (`check_iteration_neutral`) — the
    no-op-iteration lemma underlying single-batch causal equivalence:
    iterations beyond the causal cutoff leave `acc`, `logsum`, and
    `scores_max` provably unchanged. Positive tests prove the lemma on
    the deployed paged-attention kernel; negative controls confirm that both
    the mask premise and its explicit state guards are load-bearing.
"""

from ir import (
    Assign,
    BoolLit,
    FloatType,
    For,
    If,
    IntLit,
    Range,
    ReduceSum,
    ReduceMax,
    TensorType,
    Var,
    Zeros,
)
from ir.positional import (
    NeutralityAssumptions,
    PositionalAnalyzer,
    check_iteration_neutral,
    check_iteration_row_neutral,
    facts_all_neg_inf,
)
from ir.preprocess import build_type_env, check_variable_names, check_tensorindex_readonly
from ir.subst import expand_let_bindings, specialize_kernel_constants
from ir.translate import translate_kernel_source
from ir.typ import infer_types


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def find_for_loop(body, var_name):
    for stmt in body:
        match stmt:
            case For(var=Var(name=n)) if n == var_name:
                return stmt
            case If(then_body=t, else_body=e):
                r = find_for_loop(list(t), var_name) or find_for_loop(list(e), var_name)
                if r is not None:
                    return r
            case For(body=b):
                r = find_for_loop(list(b), var_name)
                if r is not None:
                    return r
    return None


def load_kernel(kernel_name: str, constants: dict, source: str | None = None):
    """Translate, specialize, expand let-bindings, and type-infer a
    paged-attention kernel. The positional analyzer
    needs concrete tensor extents and `Expr.type` set on every node."""
    if source is None:
        with open("triton_kernels/fattn_paged.py") as fh:
            source = fh.read()
    bool_constants = {k: v for k, v in constants.items() if isinstance(v, bool)}
    numeric_constants = {k: v for k, v in constants.items() if not isinstance(v, bool)}
    kernel = translate_kernel_source(
        source, kernel_name, specialize=bool_constants if bool_constants else None
    )
    roles = check_variable_names(kernel)
    check_tensorindex_readonly(kernel)
    kernel = specialize_kernel_constants(kernel, numeric_constants)
    kernel = expand_let_bindings(kernel)
    kernel, _ = infer_types(kernel, roles)
    return kernel


def load_kernel1():
    return load_kernel(
        "fattn_varlen_paged_fwd_block_ptr_kernel",
        {"D_HEAD": 64, "BLOCK_M": 64, "BLOCK_N": 64, "PAGE_BLOCK_SIZE": 64},
    )


def loop_body_exit_facts(kernel, loop_var: str, assumptions: NeutralityAssumptions):
    """Facts at the end of a parametric iteration, not after the whole loop."""
    loop = find_for_loop(list(kernel.grid.body), loop_var)
    assert loop is not None, f"no {loop_var} loop found"
    type_env = build_type_env(kernel)
    a = PositionalAnalyzer(
        assumptions,
        type_env,
        assume_then_branches=True,
    )
    for stmt in kernel.grid.body:
        a.exec_stmt(stmt)
    return a.trace().after(loop.body[-1])


def kernel1_assumptions():
    return NeutralityAssumptions(
        all_false_masks=frozenset({"attn_mask"}),
    )


# ---------------------------------------------------------------------------
# per-position predicate tracking
# ---------------------------------------------------------------------------


def test_kernel1_p_zero_where_covers_not_attn_mask():
    """Under no special assumptions beyond the ones needed to run the
    mask-narrowing chain, p's zero_where predicate must cover all
    positions where attn_mask is false (locally in its free indices)."""
    kernel = load_kernel1()
    facts = loop_body_exit_facts(kernel, "ki", NeutralityAssumptions())
    p_facts = facts.get("p")
    mask_facts = facts.get("attn_mask")
    assert p_facts is not None
    assert mask_facts is not None

    # attn_mask has exact false_where from Where + 'and' assignments
    assert mask_facts.false_where.rank == 2
    assert not isinstance(mask_facts.false_where.body, BoolLit), (
        "attn_mask should have a non-trivial false_where built from its "
        "definition, got: "
        + str(mask_facts.false_where.body)
    )

    # p.zero_where includes ¬attn_mask: since p = Where(attn_mask, _, 0),
    # positions where attn_mask.false_where is true are zero-positions of
    # p. Check that attn_mask.false_where is structurally a subterm of
    # p.zero_where (conservatively: we just require p.zero_where to not
    # be "false", i.e. we proved *something*).
    assert not isinstance(p_facts.zero_where.body, BoolLit), (
        "p should have non-trivial zero_where, got: " + str(p_facts.zero_where.body)
    )

    # Specifically: p.zero_where should contain attn_mask.false_where as
    # a disjunct (via the OR from Where's definition). Since we didn't
    # canonicalize, just pretty-print both and check the mask body
    # appears.
    from ir.pp import pp_expr
    mask_str = pp_expr(mask_facts.false_where.body)
    p_zero_str = pp_expr(p_facts.zero_where.body)
    assert mask_str in p_zero_str, (
        f"expected attn_mask.false_where to appear in p.zero_where\n"
        f"mask.false_where = {mask_str}\n"
        f"p.zero_where     = {p_zero_str}"
    )


def test_kernel1_p_zero_where_under_allfalse_mask_is_universal():
    """If the caller discharges `attn_mask` as all-false, p's zero_where
    should simplify to TRUE (p is zero everywhere in this iteration)."""
    kernel = load_kernel1()
    assumptions = NeutralityAssumptions(
        all_false_masks=frozenset({"attn_mask"}),
    )
    facts = loop_body_exit_facts(kernel, "ki", assumptions)
    p_facts = facts["p"]
    assert isinstance(p_facts.zero_where.body, BoolLit)
    assert p_facts.zero_where.body.value is True, (
        f"expected p.zero_where = TRUE when attn_mask is all-false, got "
        f"{p_facts.zero_where.body}"
    )


def test_bool_reduce_max_guard_is_false_under_all_false_mask():
    """Model Triton's promoted max(bool) as integer any(mask), fail closed."""
    from ir import BinOp, BoolType, Grid, Kernel, Param, VarDecl

    mask_ty = TensorType(BoolType(), [IntLit(4), IntLit(8)])
    row_ty = TensorType(BoolType(), [IntLit(4)])
    kernel = Kernel(
        name="bool_reduce_guard",
        params=[Param("mask_input", mask_ty)],
        grid=Grid(
            iters=[],
            decls=[VarDecl(Var("mask"), mask_ty), VarDecl(Var("row_has_any"), row_ty)],
            body=[
                Assign(Var("mask"), None, Var("mask_input")),
                Assign(
                    Var("row_has_any"),
                    None,
                    BinOp(">", ReduceMax(Var("mask"), 1), IntLit(0)),
                ),
            ],
        ),
    )
    kernel, _ = infer_types(kernel)
    a = PositionalAnalyzer(
        assumptions=NeutralityAssumptions(
            all_false_masks=frozenset({"mask"})
        ),
        type_env=build_type_env(kernel),
    )
    for statement in kernel.grid.body:
        a.exec_stmt(statement)
    facts = a.env["row_has_any"]
    assert isinstance(facts.false_where.body, BoolLit)
    assert facts.false_where.body.value is True


def test_whole_false_where_preserves_accumulator_provenance():
    """An explicit false guard selects old state and skips the candidate."""
    from ir import BinOp, BoolType, FloatLit, Where

    ty = TensorType(FloatType(), [IntLit(4), IntLit(8)])
    guard_ty = TensorType(BoolType(), [IntLit(4), IntLit(8)])
    acc = Var("acc", type=ty)
    candidate = Var("candidate", type=ty)
    guard = Var("guard", type=guard_ty)
    report = check_iteration_neutral(
        [
            Assign(target=guard, op=None, value=guard),
            Assign(
                target=acc,
                op=None,
                value=Where(
                    guard,
                    BinOp("*", candidate, FloatLit(0.0), type=ty),
                    acc,
                    type=ty,
                ),
            )
        ],
        accumulators=["acc"],
        assumptions=NeutralityAssumptions(
            finite_vars=frozenset({"candidate"}),
            all_false_masks=frozenset({"guard"}),
        ),
    )
    assert report.proved, report.failures
    # The unselected candidate's x*0 rule would consume finite(candidate).
    assert report.used_assumptions == {"all_false_mask(guard)"}


# ---------------------------------------------------------------------------
# accumulator-neutrality lemma
# ---------------------------------------------------------------------------


def test_kernel1_proved_neutral():
    """Kernel 1 (ki loop) is accumulator-neutral under kernel1_assumptions."""
    kernel = load_kernel1()
    loop = find_for_loop(list(kernel.grid.body), "ki")
    report = check_iteration_neutral(
        list(loop.body),
        accumulators=["acc", "logsum", "scores_max"],
        assumptions=kernel1_assumptions(),
    )
    assert report.proved, f"failures: {report.failures}"


def test_kernel1_neutrality_uses_only_mask_oracle():
    kernel = load_kernel1()
    loop = find_for_loop(list(kernel.grid.body), "ki")
    report = check_iteration_neutral(
        list(loop.body),
        accumulators=["acc", "logsum", "scores_max"],
        assumptions=kernel1_assumptions(),
    )
    assert report.used_assumptions == {"all_false_mask(attn_mask)"}


def test_kernel1_proves_selected_row_neutrality():
    """Whole-mask neutrality is soundly lifted to one masked query row."""
    kernel = load_kernel1()
    loop = find_for_loop(list(kernel.grid.body), "ki")
    report = check_iteration_row_neutral(
        list(loop.body),
        accumulators=["acc", "logsum", "scores_max"],
        mask_name="attn_mask",
        assumptions=kernel1_assumptions(),
    )
    assert report.proved, report.row_separation_failures


def test_row_neutrality_rejects_cross_row_mask_mixing():
    """A transpose makes output row r depend on mask column r."""
    from ir import (
        BoolType, FloatLit, Full, Grid, Kernel, Param, Transpose, VarDecl,
        Where, mm_,
    )

    four = IntLit(4)
    mask_ty = TensorType(BoolType(), [four, four])
    float_ty = TensorType(FloatType(), [four, four])
    kernel = Kernel(
        name="cross_row_mask",
        params=[Param("mask", mask_ty), Param("v", float_ty)],
        grid=Grid(
            iters=[],
            decls=[VarDecl(Var("p"), float_ty), VarDecl(Var("acc"), float_ty)],
            body=[
                # Cut-point assignment lets the explicit all-false oracle model
                # the counterfactual mask tensor.
                Assign(target=Var("mask"), op=None, value=Var("mask")),
                Assign(
                    target=Var("p"),
                    op=None,
                    value=Where(
                        Transpose(Var("mask"), [1, 0]),
                        Full([four, four], FloatLit(1.0)),
                        Full([four, four], FloatLit(0.0)),
                    ),
                ),
                Assign(
                    target=Var("acc"), op=None,
                    value=Where(
                        Transpose(Var("mask"), [1, 0]),
                        mm_(Var("p"), Var("v")), Var("acc"),
                    ),
                ),
            ],
        ),
    )
    kernel, _ = infer_types(kernel)
    report = check_iteration_row_neutral(
        list(kernel.grid.body),
        accumulators=["acc"],
        mask_name="mask",
        assumptions=NeutralityAssumptions(
            finite_vars=frozenset({"acc", "v"}),
            all_false_masks=frozenset({"mask"}),
        ),
    )
    assert report.whole_report.proved
    assert not report.proved
    assert report.row_separation_failures


def test_row_neutrality_rejects_mask_value_escaping_before_cut():
    """The final-mask cut must reject an earlier cross-row alias.

    With an all-false whole mask the iteration below is neutral.  But knowing
    only that one row of the *final* mask is false does not constrain ``tmp``,
    which captured a transpose of the first mask value.  A suffix analysis
    starting after the final assignment would otherwise miss this dependency.
    """
    from ir import (
        BoolType, FloatLit, Full, Grid, Kernel, Param, Transpose, VarDecl,
        Where, mm_,
    )

    four = IntLit(4)
    mask_ty = TensorType(BoolType(), [four, four])
    float_ty = TensorType(FloatType(), [four, four])
    kernel = Kernel(
        name="escaped_mask_before_cut",
        params=[Param("mask_input", mask_ty), Param("v", float_ty)],
        grid=Grid(
            iters=[],
            decls=[
                VarDecl(Var("mask"), mask_ty),
                VarDecl(Var("tmp"), mask_ty),
                VarDecl(Var("p"), float_ty),
                VarDecl(Var("acc"), float_ty),
            ],
            body=[
                Assign(target=Var("mask"), op=None, value=Var("mask_input")),
                Assign(target=Var("tmp"), op=None, value=Transpose(Var("mask"), [1, 0])),
                Assign(target=Var("mask"), op=None, value=Var("mask_input")),
                Assign(
                    target=Var("p"),
                    op=None,
                    value=Where(
                        Var("tmp"),
                        Full([four, four], FloatLit(1.0)),
                        Full([four, four], FloatLit(0.0)),
                    ),
                ),
                Assign(
                    target=Var("acc"), op=None,
                    value=Where(Var("tmp"), mm_(Var("p"), Var("v")), Var("acc")),
                ),
            ],
        ),
    )
    kernel, _ = infer_types(kernel)
    report = check_iteration_row_neutral(
        list(kernel.grid.body),
        accumulators=["acc"],
        mask_name="mask",
        assumptions=NeutralityAssumptions(
            finite_vars=frozenset({"acc", "v"}),
            all_false_masks=frozenset({"mask", "mask_input"}),
        ),
    )
    assert report.whole_report.proved
    assert not report.proved
    assert any(
        "between fact-construction writes" in failure
        for failure in report.row_separation_failures
    )


# ---------------------------------------------------------------------------
# negative controls
# ---------------------------------------------------------------------------


def test_scalar_compound_update_forgets_target_facts_without_changing_rank():
    from ir import FloatLit
    from ir.positional import facts_all_zero, pred_false

    ty = TensorType(FloatType(), [IntLit(2), IntLit(4)])
    analyzer = PositionalAnalyzer(
        NeutralityAssumptions(), type_env={"acc": ty},
        env={"acc": facts_all_zero(2)},
    )
    analyzer.elem_body["acc"] = FloatLit(0.0)
    analyzer.exec_stmt(Assign(target=Var("acc", type=ty), op="+",
                              value=FloatLit(1.0, type=FloatType())))
    assert analyzer.env["acc"].rank == 2
    assert analyzer.env["acc"].zero_where == pred_false(2)
    assert "acc" not in analyzer.elem_body


def test_compound_add_does_not_treat_unknown_rhs_as_finite():
    """No proof of RHS -inf is not a proof that RHS excludes +inf/NaN."""
    ty = TensorType(FloatType(), [IntLit(1)])
    rhs = Var("rhs", type=ty)
    analyzer = PositionalAnalyzer(
        NeutralityAssumptions(),
        type_env={"acc": ty, "rhs": ty},
        env={"acc": facts_all_neg_inf(1)},
    )
    analyzer.exec_stmt(Assign(target=Var("acc"), op="+", value=rhs))
    final = analyzer.env["acc"]
    assert isinstance(final.neg_inf_where.body, BoolLit)
    assert final.neg_inf_where.body.value is False


def test_parametric_loop_forgets_carried_initializer_facts() -> None:
    """The syntactic first-iteration state cannot stand for every iteration."""

    ty = TensorType(FloatType(), [IntLit(1)])
    state = Var("state", type=ty)
    zero = Zeros([IntLit(1)], type=ty)
    update = Assign(target=state, op="+", value=zero)
    loop = For(
        var=Var("i"),
        iters=Range(IntLit(0), IntLit(2)),
        body=[update],
    )
    analyzer = PositionalAnalyzer(
        NeutralityAssumptions(),
        type_env={"state": ty},
    )
    analyzer.exec_stmt(Assign(target=state, op=None, value=zero))
    assert isinstance(analyzer.env["state"].zero_where.body, BoolLit)
    assert analyzer.env["state"].zero_where.body.value is True

    analyzer.exec_stmt(loop)
    before_update = analyzer.trace().before(update)["state"]
    assert isinstance(before_update.zero_where.body, BoolLit)
    assert before_update.zero_where.body.value is False
    assert isinstance(analyzer.env["state"].zero_where.body, BoolLit)
    assert analyzer.env["state"].zero_where.body.value is False


def test_parametric_loop_forgets_noncarried_exit_facts() -> None:
    """A loop-local fact is not a fact about the exit of a possibly empty loop."""

    ty = TensorType(FloatType(), [IntLit(1)])
    temp = Var("temp", type=ty)
    zero = Zeros([IntLit(1)], type=ty)
    define_temp = Assign(target=temp, op=None, value=zero)
    loop = For(
        var=Var("i"),
        iters=Range(IntLit(0), IntLit(2)),
        body=[define_temp],
    )
    analyzer = PositionalAnalyzer(
        NeutralityAssumptions(),
        type_env={"temp": ty},
    )
    analyzer.exec_stmt(loop)
    assert isinstance(analyzer.env["temp"].zero_where.body, BoolLit)
    assert analyzer.env["temp"].zero_where.body.value is False


def test_reduction_does_not_assume_symbolic_axis_is_nonempty():
    extent = Var("N")
    input_type = TensorType(FloatType(), [extent])
    output_type = TensorType(FloatType(), [])
    zeros = Zeros([extent], type=input_type)
    reduction = ReduceSum(zeros, 0, type=output_type)
    analyzer = PositionalAnalyzer(NeutralityAssumptions(), type_env={})
    facts, _ = analyzer._eval(reduction, 0)
    assert isinstance(facts.zero_where.body, BoolLit)
    assert facts.zero_where.body.value is False


def test_broadcast_substitutes_zero_on_expanded_source_axis():
    """A vacuous source predicate must not become a fact at output index 1."""
    from ir import BinOp, BroadcastTo
    from ir.positional import Pred, TensorFacts, free_idx, pred_false
    from ir.pp import pp_expr

    src_ty = TensorType(FloatType(), [IntLit(1), IntLit(4)])
    out_ty = TensorType(FloatType(), [IntLit(3), IntLit(4)])
    vacuous_zero = Pred(
        2, BinOp("==", free_idx(0), IntLit(1))
    )
    x_facts = TensorFacts(
        rank=2,
        zero_where=vacuous_zero,
        neg_inf_where=pred_false(2),
        one_where=pred_false(2),
        true_where=pred_false(2),
        false_where=pred_false(2),
        unchanged_from=None,
    )
    analyzer = PositionalAnalyzer(
        NeutralityAssumptions(),
        type_env={"x": src_ty},
        env={"x": x_facts},
    )
    facts, _ = analyzer._eval(
        BroadcastTo(Var("x", type=src_ty), [IntLit(3), IntLit(4)], type=out_ty),
        2,
    )
    assert "_i0" not in pp_expr(facts.zero_where.body)


def test_kernel1_fails_without_mask_oracle():
    """Without `all_false_masks`, the body genuinely updates the
    accumulators, so neutrality must not be provable."""
    kernel = load_kernel1()
    loop = find_for_loop(list(kernel.grid.body), "ki")
    a = kernel1_assumptions()
    a = NeutralityAssumptions(
        finite_vars=a.finite_vars,
        all_false_masks=frozenset(),
    )
    report = check_iteration_neutral(
        list(loop.body),
        accumulators=["acc", "logsum", "scores_max"],
        assumptions=a,
    )
    assert not report.proved


def test_attention_neutrality_rejects_removed_state_guard():
    """The certificate must fail if an all-false row commits a candidate.

    This source-level mutation recreates the IEEE-unsafe update pattern: the
    candidate may contain NaN/Inf, so the mask premise alone cannot prove the
    old state is preserved.
    """
    guarded = "logsum = tl.where(row_has_any, next_logsum, logsum)"
    cases = [(
        "triton_kernels/fattn_paged.py",
        "fattn_varlen_paged_fwd_block_ptr_kernel",
        "ki",
        {"D_HEAD": 64, "BLOCK_M": 64, "BLOCK_N": 64,
         "PAGE_BLOCK_SIZE": 64},
    )]
    for path, kernel_name, loop_var, constants in cases:
        with open(path) as source_file:
            source = source_file.read()
        assert source.count(guarded) == 1
        broken = source.replace(guarded, "logsum = next_logsum")
        kernel = load_kernel(kernel_name, constants, source=broken)
        loop = find_for_loop(list(kernel.grid.body), loop_var)
        report = check_iteration_neutral(
            list(loop.body),
            accumulators=["acc", "logsum", "scores_max"],
            assumptions=NeutralityAssumptions(
                all_false_masks=frozenset({"attn_mask"})
            ),
        )
        assert not report.proved
        assert any(name == "logsum" for name, _ in report.failures)


def test_default_assumptions_prove_nothing():
    """With the empty assumption bundle, the kernel should not be provable.
    This guards against any future baked-in rule that would
    silently bypass the explicit assumption system."""
    kernel = load_kernel1()
    loop = find_for_loop(list(kernel.grid.body), "ki")
    report = check_iteration_neutral(
        list(loop.body),
        accumulators=["acc", "logsum", "scores_max"],
        assumptions=NeutralityAssumptions(),
    )
    assert not report.proved
    assert report.used_assumptions == set()
