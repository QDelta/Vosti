"""Adversarial tests for logical cross-program write disjointness."""

from itertools import product
from pathlib import Path

import z3

from ir import (
    BinOp,
    FloatType,
    For,
    Grid,
    GridIter,
    If,
    IntLit,
    IntType,
    Kernel,
    MaskedStore,
    Param,
    TensorType,
    add_,
    index_,
    lit_,
    mul_,
    range_,
    slice_,
    v_,
)
from diagnostics.race import (
    LogicalRaceConfig,
    collect_write_footprints,
    prove_logical_write_disjointness,
    prove_logical_write_disjointness_from_annotations,
)


def _store(base: str, starts, stops, masks) -> MaskedStore:
    return MaskedStore(
        base=v_(base),
        region=[slice_(start, stop) for start, stop in zip(starts, stops, strict=True)],
        value=IntLit(0),
        mask=[slice_(start, stop) for start, stop in masks],
    )


def _kernel(
    body,
    *,
    grid_iters=None,
    params=None,
    name="race_test",
) -> Kernel:
    if grid_iters is None:
        grid_iters = [GridIter(v_("pid"), range_(lit_(0), lit_(4)))]
    if params is None:
        params = [Param("out", TensorType(FloatType(), [lit_(16)]))]
    return Kernel(
        name=name,
        params=params,
        grid=Grid(iters=grid_iters, decls=[], body=body),
    )


def _prove(kernel: Kernel, **kwargs):
    return prove_logical_write_disjointness(
        LogicalRaceConfig(kernel=kernel, env={}, **kwargs)
    )


def _failed_races(result):
    return [
        check
        for check in result.checks
        if check.name.startswith("logical_write_disjoint:") and not check.proved
    ]


def test_row_partition_is_disjoint():
    pid = v_("pid")
    kernel = _kernel(
        [_store("out", [pid], [add_(pid, lit_(1))], [(lit_(0), lit_(16))])]
    )
    assert _prove(kernel).ok


def test_dropped_program_axis_finds_race():
    kernel = _kernel(
        [_store("out", [lit_(0)], [lit_(1)], [(lit_(0), lit_(16))])]
    )
    result = _prove(kernel)
    assert not result.ok
    assert _failed_races(result)
    assert "race_0_0_a_grid_pid" in _failed_races(result)[0].details


def test_partial_tile_mask_remains_disjoint():
    pid = v_("pid")
    start = mul_(pid, lit_(4))
    kernel = _kernel(
        [_store("out", [start], [add_(start, lit_(4))], [(lit_(0), lit_(10))])],
        grid_iters=[GridIter(pid, range_(lit_(0), lit_(3)))],
    )
    assert _prove(kernel).ok


def test_tile_boundary_off_by_one_finds_race():
    pid = v_("pid")
    start = mul_(pid, lit_(4))
    kernel = _kernel(
        [_store("out", [start], [add_(start, lit_(5))], [(lit_(0), lit_(16))])]
    )
    assert not _prove(kernel).ok


def test_affine_interval_solver_matches_small_exhaustive_oracle():
    """Compare SMT acceptance with explicit cell enumeration on small grids."""

    for grid_size, stride, width, offset, mask_stop in product(
        range(1, 5),
        range(4),
        range(1, 4),
        (-1, 0),
        (1, 4, 7),
    ):
        pid = v_("pid")
        start = add_(mul_(pid, lit_(stride)), lit_(offset))
        stop = add_(start, lit_(width))
        kernel = _kernel(
            [_store("out", [start], [stop], [(lit_(0), lit_(mask_stop))])],
            grid_iters=[GridIter(pid, range_(lit_(0), lit_(grid_size)))],
        )

        writes = []
        for program in range(grid_size):
            first = program * stride + offset
            writes.append(
                {
                    cell
                    for cell in range(first, first + width)
                    if 0 <= cell < mask_stop
                }
            )
        collision = any(
            writes[first] & writes[second]
            for first in range(grid_size)
            for second in range(first + 1, grid_size)
        )
        result = _prove(kernel)
        assert result.ok == (not collision), (
            grid_size,
            stride,
            width,
            offset,
            mask_stop,
            result.checks,
        )


def test_ignored_second_grid_axis_finds_race():
    row = v_("row")
    head = v_("head")
    kernel = _kernel(
        [_store("out", [row], [add_(row, lit_(1))], [(lit_(0), lit_(2))])],
        grid_iters=[
            GridIter(row, range_(lit_(0), lit_(2))),
            GridIter(head, range_(lit_(0), lit_(2))),
        ],
        params=[Param("out", TensorType(FloatType(), [lit_(2)]))],
    )
    assert not _prove(kernel).ok


def test_guard_restricting_store_to_one_program_is_disjoint():
    pid = v_("pid")
    only_zero = BinOp("==", pid, lit_(0))
    kernel = _kernel(
        [
            If(
                cond=only_zero,
                then_body=[
                    _store("out", [lit_(0)], [lit_(1)], [(lit_(0), lit_(16))])
                ],
                else_body=[],
            )
        ]
    )
    assert _prove(kernel).ok


def test_mutually_exclusive_per_program_branches_can_race_across_programs():
    pid = v_("pid")
    kernel = _kernel(
        [
            If(
                cond=BinOp("==", pid, lit_(0)),
                then_body=[
                    _store("out", [lit_(0)], [lit_(1)], [(lit_(0), lit_(16))])
                ],
                else_body=[
                    _store("out", [lit_(0)], [lit_(1)], [(lit_(0), lit_(16))])
                ],
            )
        ],
        grid_iters=[GridIter(pid, range_(lit_(0), lit_(2)))],
    )
    result = _prove(kernel)
    assert not result.ok
    assert any("site0:site1" in check.name for check in _failed_races(result))


def test_inner_loop_iterations_are_sequential_not_program_identities():
    pid = v_("pid")
    loop = For(
        var=v_("r"),
        iters=range_(lit_(0), lit_(3)),
        body=[
            _store("out", [pid], [add_(pid, lit_(1))], [(lit_(0), lit_(16))])
        ],
    )
    kernel = _kernel([loop])
    footprints = collect_write_footprints(kernel)
    assert [context.var_name for context in footprints[0].loops] == ["r"]
    assert _prove(kernel).ok


def _kv_store_kernel():
    pid = v_("pid")
    slot = index_(v_("slot_mapping"), [pid])
    return _kernel(
        [
            _store(
                "cache",
                [slot, lit_(0)],
                [add_(slot, lit_(1)), lit_(2)],
                [(lit_(0), lit_(8)), (lit_(0), lit_(2))],
            )
        ],
        grid_iters=[GridIter(pid, range_(lit_(0), lit_(4)))],
        params=[
            Param("slot_mapping", TensorType(IntType(), [lit_(4)])),
            Param("cache", TensorType(FloatType(), [lit_(8), lit_(2)])),
        ],
        name="store_kv_cache",
    )


def test_kv_scatter_injectivity_proves_disjointness():
    slots = z3.Function("slot_mapping", z3.IntSort(), z3.IntSort())
    i, j = z3.Ints("i j")
    injective = z3.ForAll(
        [i, j],
        z3.Implies(
            z3.And(i >= 0, i < 4, j >= 0, j < 4, slots(i) == slots(j)),
            i == j,
        ),
    )
    result = prove_logical_write_disjointness(
        LogicalRaceConfig(
            kernel=_kv_store_kernel(),
            env={"slot_mapping": slots},
            assumptions=[injective],
        )
    )
    assert result.ok


def test_kv_scatter_without_injectivity_finds_race():
    slots = z3.Function("slot_mapping", z3.IntSort(), z3.IntSort())
    result = prove_logical_write_disjointness(
        LogicalRaceConfig(
            kernel=_kv_store_kernel(),
            env={"slot_mapping": slots},
        )
    )
    assert not result.ok
    assert _failed_races(result)


def test_translated_kv_kernel_uses_its_quantified_annotation_contract():
    """Exercise the source translator and annotation environment together."""
    source = Path("triton_kernels/store_kv_cache.py").read_text()
    constants = {"KVD": 128, "BLOCK_M": 1}
    result = prove_logical_write_disjointness_from_annotations(
        source,
        "store_cache_kernel",
        constants,
        goal_name="batch_invariance",
        timeout_ms=5000,
    )
    assert result.ok


def test_race_driver_selects_one_named_goal_from_multi_goal_kernel():
    source = Path("triton_kernels/fattn_paged.py").read_text()
    result = prove_logical_write_disjointness_from_annotations(
        source,
        "fattn_varlen_paged_fwd_block_ptr_kernel",
        {
            "BLOCK_M": 16,
            "BLOCK_N": 64,
            "D_HEAD": 128,
            "PAGE_BLOCK_SIZE": 64,
        },
        goal_name="batch_invariance",
        timeout_ms=5000,
    )
    assert result.ok


def test_translated_kv_kernel_fails_if_injectivity_clause_is_removed():
    source = Path("triton_kernels/store_kv_cache.py").read_text()
    injectivity = (
        "#     forall(i, j, implies(and(i >= 0, i < left(M), j >= 0, "
        "j < left(M), left(slot_mapping)[i] >= 0, "
        "left(slot_mapping)[i] == left(slot_mapping)[j]), i == j)),\n"
    )
    assert injectivity in source
    mutated = source.replace(injectivity, "", 1)
    result = prove_logical_write_disjointness_from_annotations(
        mutated,
        "store_cache_kernel",
        {"KVD": 128, "BLOCK_M": 1},
        goal_name="batch_invariance",
        timeout_ms=5000,
    )
    assert not result.ok
    assert _failed_races(result)


def test_race_preconditions_exclude_right_and_cross_run_constraints():
    from ir.annotation_to_config import build_left_launch_assumptions
    from ir.annotations import (
        IntConst,
        Left,
        RelationalProofGoal,
        Right,
        AnnComparison,
    )

    annotation = RelationalProofGoal(
        name="race_projection",
        pre_conditions=[
            AnnComparison(">", Left("M"), IntConst(0)),
            AnnComparison("==", Right("M"), IntConst(1)),
            AnnComparison("==", Left("N"), Right("N")),
        ],
        post_conditions=[],
    )
    left_env = {"M": z3.Int("left_M"), "N": z3.Int("left_N")}
    right_env = {"M": z3.Int("right_M"), "N": z3.Int("right_N")}
    assumptions = build_left_launch_assumptions(
        annotation, left_env, right_env
    )
    assert len(assumptions) == 1
    assert str(assumptions[0]) == "0 < left_M"


def test_contradictory_preconditions_are_vacuously_race_free():
    pid = v_("pid")
    kernel = _kernel(
        [_store("out", [pid], [add_(pid, lit_(1))], [(lit_(0), lit_(16))])]
    )
    result = _prove(kernel, assumptions=[z3.BoolVal(False)])
    assert result.ok
    assert any(check.name == "vacuous_preconditions" for check in result.checks)
    assert result.diagnostics[0].name == "race_preconditions_satisfiable"
    assert "unsat" in result.diagnostics[0].details


def test_contradictory_preconditions_do_not_hide_malformed_footprints():
    pid = v_("pid")
    kernel = _kernel(
        [
            _store(
                "out",
                [pid, lit_(0)],
                [add_(pid, lit_(1)), lit_(1)],
                [(lit_(0), lit_(16)), (lit_(0), lit_(1))],
            )
        ]
    )
    result = _prove(kernel, assumptions=[z3.BoolVal(False)])
    assert not result.ok
    assert any(
        check.name == "write_footprint_well_formed:site0" and not check.proved
        for check in result.checks
    )


def test_distinct_writable_bases_are_separate_logical_spaces():
    pid = v_("pid")
    body = [
        _store("a", [pid], [add_(pid, lit_(1))], [(lit_(0), lit_(4))]),
        _store("b", [pid], [add_(pid, lit_(1))], [(lit_(0), lit_(4))]),
    ]
    params = [
        Param("a", TensorType(FloatType(), [lit_(4)])),
        Param("b", TensorType(FloatType(), [lit_(4)])),
    ]
    kernel = _kernel(body, params=params)

    # This pass proves a logical, per-tensor property only.  Physical no-alias
    # between ``a`` and ``b`` is a separate launch obligation and cannot be
    # manufactured by passing opaque allocation names into the SMT checker.
    assert _prove(kernel).ok


def test_no_global_store_fails_closed():
    result = _prove(_kernel([]))
    assert not result.ok
    assert result.checks[0].name == "global_store_present"
