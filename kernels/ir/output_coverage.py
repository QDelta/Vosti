"""Fail-closed coverage obligations for annotated output regions.

Regional equivalence only implies equality for cells that the kernel actually
writes.  This module constructs concrete program-iterator witnesses for the
rectangular tiling patterns accepted by the verifier.  The resulting formula
is quantifier-free: its free point variables stand for an arbitrary selected
output cell.

Witness synthesis is deliberately incomplete.  If a tiling shape is outside
the supported patterns, verification rejects it instead of assuming coverage.
"""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import fields, is_dataclass
from itertools import product

import z3

from . import BinOp, Expr, IntLit, Kernel, Region, TensorIndex, Var, collect_expr_vars
from .regions import ConditionalRegion


Z3Val = z3.ArithRef | z3.BoolRef | z3.FuncDeclRef
ExprTranslator = Callable[[Expr, dict[str, Z3Val]], z3.ArithRef | z3.BoolRef]


class UnsupportedOutputCoverage(ValueError):
    """The output tiling is outside the conservative witness subset."""


def _child_exprs(expr: Expr) -> list[Expr]:
    children: list[Expr] = []
    if not is_dataclass(expr):
        return children
    for field in fields(expr):
        value = getattr(expr, field.name)
        if isinstance(value, Expr):
            children.append(value)
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            children.extend(item for item in value if isinstance(item, Expr))
    return children


def _positive_iterator_tiles(expr: Expr, iterator: str) -> set[int]:
    """Return literal positive coefficients from ``iterator * coefficient``."""
    tiles: set[int] = set()
    match expr:
        case BinOp(op="*", lhs=Var(name=name), rhs=IntLit(value=value)):
            if name == iterator and value > 0:
                tiles.add(value)
        case BinOp(op="*", lhs=IntLit(value=value), rhs=Var(name=name)):
            if name == iterator and value > 0:
                tiles.add(value)
    for child in _child_exprs(expr):
        tiles |= _positive_iterator_tiles(child, iterator)
    return tiles


def _affine_iterator_offset(
    expr: Expr,
    iterator: str,
    env: dict[str, Z3Val],
    translate: ExprTranslator,
) -> z3.ArithRef | None:
    """Return ``offset`` when ``expr`` is ``iterator - offset``.

    This deliberately recognizes only coefficient-one affine forms.  The
    returned expression is used to propose a concrete coverage witness; all
    iterator bounds, lexical guards, and destination-region constraints are
    still checked by the solver before that witness can establish coverage.
    """

    match expr:
        case Var(name=name) if name == iterator:
            return z3.IntVal(0)
        case BinOp(op="-", lhs=lhs, rhs=rhs):
            if iterator in collect_expr_vars(rhs):
                return None
            offset = _affine_iterator_offset(lhs, iterator, env, translate)
            if offset is None:
                return None
            translated = translate(rhs, env)
            return offset + translated  # pyright: ignore[reportOperatorIssue, reportReturnType]
        case BinOp(op="+", lhs=lhs, rhs=rhs):
            if iterator in collect_expr_vars(rhs):
                return None
            offset = _affine_iterator_offset(lhs, iterator, env, translate)
            if offset is None:
                return None
            translated = translate(rhs, env)
            return offset - translated  # pyright: ignore[reportOperatorIssue, reportReturnType]
    return None


def _affine_iterator_tiles(
    expr: Expr,
    iterator: str,
    env: dict[str, Z3Val],
    translate: ExprTranslator,
) -> list[tuple[int, z3.ArithRef]]:
    """Find positive ``(iterator - offset) * tile`` subexpressions."""

    matches: list[tuple[int, z3.ArithRef]] = []
    match expr:
        case BinOp(op="*", lhs=lhs, rhs=IntLit(value=tile)) if tile > 0:
            offset = _affine_iterator_offset(lhs, iterator, env, translate)
            if offset is not None:
                matches.append((tile, offset))
        case BinOp(op="*", lhs=IntLit(value=tile), rhs=rhs) if tile > 0:
            offset = _affine_iterator_offset(rhs, iterator, env, translate)
            if offset is not None:
                matches.append((tile, offset))
    for child in _child_exprs(expr):
        matches.extend(
            _affine_iterator_tiles(child, iterator, env, translate)
        )
    return matches


def _iterator_occurs_in_tensor_index(expr: Expr, iterator: str) -> bool:
    if isinstance(expr, TensorIndex) and any(
        iterator in collect_expr_vars(index) for index in expr.indices
    ):
        return True
    return any(
        _iterator_occurs_in_tensor_index(child, iterator)
        for child in _child_exprs(expr)
    )


def _integer_literals(expr: Expr) -> set[int]:
    """Return integer boundary values appearing in one guard expression."""

    values = {expr.value} if isinstance(expr, IntLit) else set()
    for child in _child_exprs(expr):
        values |= _integer_literals(child)
    return values


def _index_terms(expr: Expr) -> list[Expr]:
    """Collect scalar index terms, excluding tensor values themselves."""
    if isinstance(expr, TensorIndex):
        terms: list[Expr] = []
        for index in expr.indices:
            terms.append(index)
            terms.extend(_index_terms(index))
        return terms
    terms = [expr] if isinstance(expr, Var) else []
    for child in _child_exprs(expr):
        terms.extend(_index_terms(child))
    return terms


def _deduplicate_z3(values: Sequence[z3.ArithRef]) -> list[z3.ArithRef]:
    unique: dict[str, z3.ArithRef] = {}
    for value in values:
        unique.setdefault(value.sexpr(), value)
    return list(unique.values())


def _candidate_witnesses(
    iterator: str,
    points: Sequence[z3.ArithRef],
    output_region: Region,
    written_region: Region,
    conditions: Sequence[Expr],
    env: dict[str, Z3Val],
    translate: ExprTranslator,
) -> list[z3.ArithRef]:
    axes = [
        axis
        for axis, region_slice in enumerate(written_region)
        if iterator in (
            collect_expr_vars(region_slice.start)
            | collect_expr_vars(region_slice.stop)
        )
    ]

    # Zero handles inner lane loops paired with a tiled grid iterator (for
    # example output_index = program_id * BLOCK_M + lane).  Its declared loop
    # bounds are still proved below, so an invalid zero is never assumed.
    candidates: list[z3.ArithRef] = [z3.IntVal(0)]
    for axis in axes:
        point = points[axis]
        selected = output_region[axis]
        written = written_region[axis]
        tiles = (
            _positive_iterator_tiles(written.start, iterator)
            | _positive_iterator_tiles(written.stop, iterator)
        )
        affine_tiles = _affine_iterator_tiles(
            written.start, iterator, env, translate
        ) + _affine_iterator_tiles(
            written.stop, iterator, env, translate
        )
        indexed_iterator = _iterator_occurs_in_tensor_index(
            written.start, iterator
        ) or _iterator_occurs_in_tensor_index(written.stop, iterator)
        if tiles:
            selected_start = translate(selected.start, env)
            assert isinstance(selected_start, z3.ArithRef)
            for tile in sorted(tiles):
                candidates.extend(
                    [
                        point / tile,
                        (point - selected_start) / tile,
                    ]
                )
        if affine_tiles:
            selected_start = translate(selected.start, env)
            assert isinstance(selected_start, z3.ArithRef)
            for tile, offset in affine_tiles:
                candidates.extend(
                    [
                        point / tile + offset,
                        (point - selected_start) / tile + offset,
                    ]
                )
        if not tiles and not affine_tiles:
            # Unit-width grid dimensions (batch/head, for example) use the
            # output coordinate directly.
            candidates.append(point)

        # A data-dependent destination can still have a simple source-row
        # witness.  For cache[slot[row]], for example, choosing row=b makes the
        # write address syntactically equal to the selected cache[slot[b]]
        # address; no inverse or injectivity assumption about slot is used.
        # This applies even when the row is itself tiled (program * BLOCK+r).
        if indexed_iterator:
            for bound in (selected.start, selected.stop):
                for term in _index_terms(bound):
                    translated = translate(term, env)
                    if isinstance(translated, z3.ArithRef):
                        candidates.append(translated)

    # A grid dimension may select an output without appearing in its address.
    # For example, a third program id can select Q, K, or V while the written
    # tensors themselves are two-dimensional.  Try the finite integer
    # boundaries mentioned by that write's lexical guards and their immediate
    # neighbours.  These are merely concrete witness candidates: iterator
    # bounds and every guard are still proved below, so adding a candidate
    # cannot weaken the coverage obligation.
    for condition in conditions:
        if iterator not in collect_expr_vars(condition):
            continue
        for boundary in _integer_literals(condition):
            candidates.extend(
                z3.IntVal(value)
                for value in (boundary - 1, boundary, boundary + 1)
            )

    return _deduplicate_z3(candidates)


def build_output_coverage_claim(
    *,
    name: str,
    kernel: Kernel,
    output_region: Region,
    written: Sequence[ConditionalRegion],
    loop_ranges: Mapping[str, tuple[Expr, Expr]],
    env: dict[str, Z3Val],
    translate: ExprTranslator,
    max_witness_combinations: int = 256,
) -> z3.BoolRef:
    """Build ``selected(point) -> covered(point)`` for an arbitrary point.

    Grid iterators and only the local iterators enclosing a particular write
    receive concrete witnesses.  A bounded Cartesian search combines the
    per-iterator candidates.  Exceeding the bound is an unsupported proof
    shape, never a successful proof.
    """

    if not output_region or not written:
        raise UnsupportedOutputCoverage("output region or write set is empty")

    points = [z3.FreshInt(f"{name}_point_{axis}") for axis in range(len(output_region))]
    selected = z3.And(
        *[
            z3.And(
                point >= translate(region_slice.start, env),
                point < translate(region_slice.stop, env),
            )
            for point, region_slice in zip(points, output_region)
        ]
    )

    grid_iter_names = {grid_iter.var.name for grid_iter in kernel.grid.iters}
    alternatives: list[z3.BoolRef] = []
    for conditional in written:
        if len(conditional.region) != len(points):
            raise UnsupportedOutputCoverage("output write rank mismatch")

        active_iter_names = sorted(grid_iter_names | set(conditional.iterators))
        witness_options: list[list[z3.ArithRef]] = []
        combination_count = 1
        for iterator in active_iter_names:
            if iterator not in loop_ranges:
                raise UnsupportedOutputCoverage(
                    f"missing declared range for iterator {iterator}"
                )
            options = _candidate_witnesses(
                iterator,
                points,
                output_region,
                conditional.region,
                conditional.conditions,
                env,
                translate,
            )
            if not options:
                raise UnsupportedOutputCoverage(
                    f"no coverage witness candidates for iterator {iterator}"
                )
            combination_count *= len(options)
            if combination_count > max_witness_combinations:
                raise UnsupportedOutputCoverage(
                    "coverage witness search exceeds "
                    f"{max_witness_combinations} combinations"
                )
            witness_options.append(options)

        assignments = product(*witness_options) if witness_options else [()]
        for witnesses in assignments:
            witness_env = {
                **env,
                **dict(zip(active_iter_names, witnesses, strict=True)),
            }
            bounds = [
                z3.And(
                    witness_env[iterator]
                    >= translate(loop_ranges[iterator][0], witness_env),
                    witness_env[iterator]
                    < translate(loop_ranges[iterator][1], witness_env),
                )
                for iterator in active_iter_names
            ]
            alternatives.append(
                z3.And(
                    *bounds,
                    *[
                        translate(condition, witness_env)
                        for condition in conditional.conditions
                    ],
                    *[
                        z3.And(
                            point >= translate(region_slice.start, witness_env),
                            point < translate(region_slice.stop, witness_env),
                        )
                        for point, region_slice in zip(
                            points, conditional.region, strict=True
                        )
                    ],
                )
            )

    return z3.Implies(selected, z3.Or(*alternatives))
