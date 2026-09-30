"""Soundness tests for generic regional identity transitions."""

from ir import (
    Assign,
    BinOp,
    BoolLit,
    BoolType,
    FloatType,
    For,
    Grid,
    If,
    IntLit,
    Kernel,
    Param,
    Range,
    Slice,
    TensorType,
    Transpose,
    Var,
    VarDecl,
    Where,
    Zeros,
    add_,
)
from ir.identity_transition import (
    AllFalseFact,
    AvailableRegionalFact,
    check_region_identity_transition,
    prove_fact_requirement_covered,
)
from ir.positional import NeutralityAssumptions, pred_true
from ir.regions import GuardedRegion
from ir.typ import infer_types


FOUR = IntLit(4)
MASK_TYPE = TensorType(BoolType(), [FOUR, FOUR])
STATE_TYPE = TensorType(FloatType(), [FOUR, FOUR])


def _typed_body(update, *, prefix=()):
    kernel = Kernel(
        name="generic_identity",
        params=[
            Param("predicate", MASK_TYPE),
            Param("candidate", STATE_TYPE),
            Param("predicate_input", MASK_TYPE),
        ],
        grid=Grid(
            iters=[],
            decls=[
                VarDecl(Var("carried"), STATE_TYPE),
                VarDecl(Var("derived_predicate"), MASK_TYPE),
                VarDecl(Var("escaped"), MASK_TYPE),
            ],
            body=[*prefix, update],
        ),
    )
    kernel, _ = infer_types(kernel)
    return list(kernel.grid.body)


def _selected_row(variable: str = "row") -> tuple[Var, GuardedRegion]:
    row = Var(variable)
    return row, GuardedRegion(
        [Slice(row, add_(row, IntLit(1))), Slice(IntLit(0), FOUR)],
        pred_true(2),
    )


def _row_premises(row: Var) -> list[BinOp]:
    return [
        BinOp(">=", row, IntLit(0)),
        BinOp("<", row, FOUR),
    ]


def test_input_fact_derives_conditional_selected_region_identity():
    """No assignment or attention-specific variable name is required."""
    body = _typed_body(
        Assign(
            target=Var("carried"),
            op=None,
            value=Where(Var("predicate"), Var("candidate"), Var("carried")),
        )
    )
    row, demand = _selected_row()
    report = check_region_identity_transition(
        body,
        {"carried": demand},
        AllFalseFact("predicate"),
    )

    assert report.proved, report.failures
    assert report.used_assumptions == ("all_false_mask(predicate)",)
    assert report.external_assumptions == ()
    assert report.states[0].final_facts.unchanged_from == "carried"
    assert report.states[0].dependencies == (
        "candidate",
        "carried",
        "predicate",
    )
    assert len(report.requirements) == 1
    requirement = report.requirements[0]
    coverage = prove_fact_requirement_covered(
        requirement,
        AvailableRegionalFact(AllFalseFact("predicate"), demand),
        _row_premises(row),
        check_name="generic_selected_fact_coverage",
    )
    assert coverage.proved, coverage.details


def test_arbitrary_discarded_candidate_does_not_enter_identity_proof():
    """A false selection is exact; the discarded value needs no FP premise."""
    body = _typed_body(
        Assign(
            target=Var("carried"),
            op=None,
            value=Where(
                Var("predicate"),
                BinOp("/", Var("candidate"), Var("candidate")),
                Var("carried"),
            ),
        )
    )
    _, demand = _selected_row()
    report = check_region_identity_transition(
        body,
        {"carried": demand},
        AllFalseFact("predicate"),
    )

    assert report.proved, report.failures
    assert report.used_assumptions == ("all_false_mask(predicate)",)


def test_cross_axis_dependency_emits_uncovered_fact_requirement():
    """The generic rule remains conditional; row coverage fails separately."""
    body = _typed_body(
        Assign(
            target=Var("carried"),
            op=None,
            value=Where(
                Transpose(Var("predicate"), [1, 0]),
                Var("candidate"),
                Var("carried"),
            ),
        )
    )
    row, demand = _selected_row()
    report = check_region_identity_transition(
        body,
        {"carried": demand},
        AllFalseFact("predicate"),
    )

    assert report.proved, report.failures
    assert len(report.requirements) == 1
    coverage = prove_fact_requirement_covered(
        report.requirements[0],
        AvailableRegionalFact(AllFalseFact("predicate"), demand),
        _row_premises(row),
        check_name="cross_axis_selected_fact_coverage",
    )
    assert not coverage.proved


def test_fact_coverage_cannot_discharge_a_different_fact():
    body = _typed_body(
        Assign(
            target=Var("carried"),
            op=None,
            value=Where(Var("predicate"), Var("candidate"), Var("carried")),
        )
    )
    row, demand = _selected_row()
    report = check_region_identity_transition(
        body,
        {"carried": demand},
        AllFalseFact("predicate"),
    )

    coverage = prove_fact_requirement_covered(
        report.requirements[0],
        AvailableRegionalFact(AllFalseFact("predicate_input"), demand),
        _row_premises(row),
        check_name="mismatched_fact_coverage",
    )
    assert not coverage.proved


def test_removing_state_selection_rejects_identity():
    body = _typed_body(Assign(target=Var("carried"), op=None, value=Var("candidate")))
    _, demand = _selected_row()
    report = check_region_identity_transition(
        body,
        {"carried": demand},
        AllFalseFact("predicate"),
    )

    assert not report.proved
    assert not report.whole_report.proved
    assert report.whole_report.failures


def test_reindexed_prestate_is_not_misclassified_as_identity():
    """A square transpose preserves shape but not pointwise state values."""
    body = _typed_body(
        Assign(
            target=Var("carried"),
            op=None,
            value=Where(
                Var("predicate"),
                Var("candidate"),
                Transpose(Var("carried"), [1, 0]),
            ),
        )
    )
    _, demand = _selected_row()
    report = check_region_identity_transition(
        body,
        {"carried": demand},
        AllFalseFact("predicate"),
    )

    assert not report.proved
    assert report.whole_report.failures


def test_numerical_additive_identity_is_not_bitwise_neutral():
    """Finite x + 0 is not an identity when x can be negative zero."""
    body = _typed_body(
        Assign(
            target=Var("carried"),
            op="+",
            value=Where(
                Var("predicate"),
                Var("candidate"),
                Zeros([FOUR, FOUR]),
            ),
        )
    )
    _, demand = _selected_row()
    report = check_region_identity_transition(
        body,
        {"carried": demand},
        AllFalseFact("predicate"),
        NeutralityAssumptions(finite_vars=frozenset({"carried"})),
    )

    assert not report.whole_report.proved
    assert not report.proved
    assert any(name == "carried" for name, _ in report.whole_report.failures)


def test_nested_pre_cut_fact_read_cannot_escape():
    prefix = (
        If(
            cond=BoolLit(True),
            then_body=[
                Assign(
                    target=Var("escaped"),
                    op=None,
                    value=Transpose(Var("derived_predicate"), [1, 0]),
                )
            ],
            else_body=[
                Assign(
                    target=Var("escaped"),
                    op=None,
                    value=Var("derived_predicate"),
                )
            ],
        ),
        Assign(
            target=Var("derived_predicate"),
            op=None,
            value=Var("predicate_input"),
        ),
    )
    body = _typed_body(
        Assign(
            target=Var("carried"),
            op=None,
            value=Where(Var("escaped"), Var("candidate"), Var("carried")),
        ),
        prefix=prefix,
    )
    _, demand = _selected_row()
    report = check_region_identity_transition(
        body,
        {"carried": demand},
        AllFalseFact("derived_predicate"),
    )

    assert report.whole_report.proved
    assert not report.proved
    assert any("read before its first full definition" in f for f in report.failures)


def test_nested_fold_fails_closed_until_ordered_induction_exists():
    nested = For(
        var=Var("i"),
        iters=Range(IntLit(0), FOUR),
        body=[
            Assign(
                target=Var("carried"),
                op=None,
                value=Where(Var("predicate"), Var("candidate"), Var("carried")),
            )
        ],
    )
    body = _typed_body(nested)
    _, demand = _selected_row()
    report = check_region_identity_transition(
        body,
        {"carried": demand},
        AllFalseFact("predicate"),
    )

    assert not report.proved
    assert "nested loops are not supported" in " ".join(report.failures)
