"""Logical write-disjointness checks for translated Triton kernels.

This module proves that distinct program instances in one kernel launch cannot
write the same *logical tensor cell of the same tensor parameter*.  It
intentionally does not claim physical race freedom: that additionally requires
injective tensor layouts, no-alias evidence for distinct writable parameters,
and a sound connection from the analyzed launch to the executed launch.

The checker is kept separate from regional value-dependency analysis.  Grid
iterations represent concurrent Triton programs; ``For`` iterations inside one
program are sequential, but their values still affect each store footprint.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import combinations_with_replacement

import z3

from ir import (
    Assign,
    Expr,
    For,
    GridIter,
    If,
    Kernel,
    Let,
    MaskedStore,
    Not,
    Range,
    Stmt,
    TensorType,
)
from ir.smt import ProofCheck, ProofResult, Z3Val, expr_to_z3
from ir.proof_preparation import prepare_annotation_proof


@dataclass(frozen=True)
class LoopContext:
    """One lexically enclosing, sequential loop."""

    var_name: str
    iters: Range


@dataclass(frozen=True)
class WriteFootprint:
    """A global store together with the control context that reaches it."""

    site: int
    conditions: tuple[Expr, ...]
    loops: tuple[LoopContext, ...]
    store: MaskedStore


@dataclass(frozen=True)
class LogicalRaceConfig:
    """Inputs to the logical write-disjointness proof.

    ``env`` and ``assumptions`` describe a single kernel launch.  TensorIndex
    functions in ``env`` are shared by both program copies.

    Different tensor parameters are distinct logical address spaces.  Whether
    they alias in physical storage is deliberately outside this pass and must
    not be discharged with caller-supplied names or allocation identifiers.
    """

    kernel: Kernel
    env: dict[str, Z3Val]
    assumptions: Sequence[z3.BoolRef | bool] = ()
    timeout_ms: int = 1000


def collect_write_footprints(kernel: Kernel) -> list[WriteFootprint]:
    """Collect global stores with their lexical guards and loop bounds."""

    footprints: list[WriteFootprint] = []

    def visit(
        statements: Sequence[Stmt],
        conditions: tuple[Expr, ...],
        loops: tuple[LoopContext, ...],
    ) -> None:
        for statement in statements:
            match statement:
                case MaskedStore() as store:
                    footprints.append(
                        WriteFootprint(
                            site=len(footprints),
                            conditions=conditions,
                            loops=loops,
                            store=store,
                        )
                    )
                case For(var=var, iters=iters, body=body):
                    visit(
                        body,
                        conditions,
                        loops + (LoopContext(var.name, iters),),
                    )
                case If(cond=cond, then_body=then_body, else_body=else_body):
                    visit(then_body, conditions + (cond,), loops)
                    visit(else_body, conditions + (Not(cond),), loops)
                case Assign() | Let():
                    # Assign writes an SSA/local tensor value; Let should have
                    # been expanded before proof, but neither is global memory.
                    continue
                case _:
                    raise AssertionError(f"unhandled statement: {statement}")

    visit(kernel.grid.body, (), ())
    return footprints


def _assumptions_satisfiability_diagnostic(
    config: LogicalRaceConfig,
) -> tuple[str, ProofCheck]:
    quantified = [
        assumption
        for assumption in config.assumptions
        if isinstance(assumption, z3.QuantifierRef)
    ]
    assumptions = list(config.assumptions)
    if quantified:
        # Z3 routinely returns unknown when asked for a model of the deployed
        # universal array clauses. Classify only the quantifier-free core; the
        # quantified premises remain part of each conditional proof query.
        assumptions = [
            assumption
            for assumption in assumptions
            if not isinstance(assumption, z3.QuantifierRef)
        ]

    solver = z3.Solver()
    solver.set("timeout", config.timeout_ms)
    solver.add(*assumptions)
    status = solver.check()
    if status == z3.sat:
        details = (
            "quantifier-free core sat; quantified assumptions are conditional"
            if quantified
            else "sat"
        )
        return "sat", ProofCheck(
            name="race_preconditions_satisfiable", proved=True, details=details
        )
    if status == z3.unsat:
        details = "unsat preconditions"
    else:
        details = f"unknown: {solver.reason_unknown()}"
    return ("unsat" if status == z3.unsat else "unknown"), ProofCheck(
        name="race_preconditions_satisfiable", proved=False, details=details
    )


def _copy_env(
    config: LogicalRaceConfig,
    footprint: WriteFootprint,
    side: str,
    pair_name: str,
) -> dict[str, Z3Val]:
    env = dict(config.env)
    for grid_iter in config.kernel.grid.iters:
        env[grid_iter.var.name] = z3.FreshInt(
            f"{pair_name}_{side}_grid_{grid_iter.var.name}"
        )
    for loop in footprint.loops:
        env[loop.var_name] = z3.FreshInt(f"{pair_name}_{side}_loop_{loop.var_name}")
    return env


def _range_clauses(
    grid_iters: Sequence[GridIter],
    loops: Sequence[LoopContext],
    env: dict[str, Z3Val],
) -> list[z3.BoolRef]:
    clauses: list[z3.BoolRef] = []
    for grid_iter in grid_iters:
        value = env[grid_iter.var.name]
        clauses.extend(
            [
                value >= expr_to_z3(grid_iter.iters.start, env),  # type: ignore[operator]
                value < expr_to_z3(grid_iter.iters.stop, env),  # type: ignore[operator]
            ]
        )
    for loop in loops:
        value = env[loop.var_name]
        clauses.extend(
            [
                value >= expr_to_z3(loop.iters.start, env),  # type: ignore[operator]
                value < expr_to_z3(loop.iters.stop, env),  # type: ignore[operator]
            ]
        )
    return clauses


def _validate_footprint(kernel: Kernel, footprint: WriteFootprint) -> str | None:
    store = footprint.store
    if len(store.region) != len(store.mask):
        return (
            f"store site {footprint.site} has region rank {len(store.region)} "
            f"but mask rank {len(store.mask)}"
        )
    param_type = next(
        (param.type for param in kernel.params if param.name == store.base.name),
        None,
    )
    if not isinstance(param_type, TensorType):
        return f"store site {footprint.site} target {store.base.name!r} is not a tensor parameter"
    if len(param_type.dims) != len(store.region):
        return (
            f"store site {footprint.site} target {store.base.name!r} has rank "
            f"{len(param_type.dims)} but store region rank {len(store.region)}"
        )
    return None


def _prove_pair_disjoint(
    config: LogicalRaceConfig,
    first: WriteFootprint,
    second: WriteFootprint,
) -> ProofCheck:
    pair_name = f"race_{first.site}_{second.site}"
    first_env = _copy_env(config, first, "a", pair_name)
    second_env = _copy_env(config, second, "b", pair_name)

    clauses: list[z3.BoolRef | bool] = list(config.assumptions)
    clauses.extend(
        _range_clauses(config.kernel.grid.iters, first.loops, first_env)
    )
    clauses.extend(
        _range_clauses(config.kernel.grid.iters, second.loops, second_env)
    )

    # Only distinct Triton program tuples may execute concurrently.  Inner
    # loop iterations are deliberately excluded from this distinction.
    program_diff = [
        first_env[grid_iter.var.name] != second_env[grid_iter.var.name]
        for grid_iter in config.kernel.grid.iters
    ]
    clauses.append(z3.Or(*program_diff) if program_diff else z3.BoolVal(False))
    clauses.extend(expr_to_z3(cond, first_env) for cond in first.conditions)
    clauses.extend(expr_to_z3(cond, second_env) for cond in second.conditions)

    for dimension, (first_region, first_mask, second_region, second_mask) in enumerate(
        zip(
            first.store.region,
            first.store.mask,
            second.store.region,
            second.store.mask,
            strict=True,
        )
    ):
        cell = z3.FreshInt(f"{pair_name}_cell_{dimension}")
        for interval, env in (
            (first_region, first_env),
            (first_mask, first_env),
            (second_region, second_env),
            (second_mask, second_env),
        ):
            clauses.extend(
                [
                    cell >= expr_to_z3(interval.start, env),
                    cell < expr_to_z3(interval.stop, env),
                ]
            )

    solver = z3.Solver()
    solver.set("timeout", config.timeout_ms)
    solver.add(*clauses)
    status = solver.check()
    name = (
        f"logical_write_disjoint:{first.store.base.name}:"
        f"site{first.site}:site{second.site}"
    )
    if status == z3.unsat:
        return ProofCheck(name=name, proved=True, details="unsat")
    if status == z3.sat:
        return ProofCheck(name=name, proved=False, details=str(solver.model()))
    return ProofCheck(
        name=name,
        proved=False,
        details=f"unknown: {solver.reason_unknown()}",
    )


def prove_logical_write_disjointness(config: LogicalRaceConfig) -> ProofResult:
    """Prove cross-program logical write disjointness, failing closed.

    This result is not a physical-memory race certificate.  The caller must
    separately establish that logical tensor cells map injectively to physical
    addresses and that different writable tensor parameters do not alias.
    """

    footprints = collect_write_footprints(config.kernel)
    if not footprints:
        return ProofResult(
            checks=[
                ProofCheck(
                    name="global_store_present",
                    proved=False,
                    details="kernel contains no MaskedStore obligation",
                )
            ]
        )

    satisfiability, diagnostic = _assumptions_satisfiability_diagnostic(config)
    diagnostics = [diagnostic]
    checks: list[ProofCheck] = []

    for footprint in footprints:
        error = _validate_footprint(config.kernel, footprint)
        checks.append(
            ProofCheck(
                name=f"write_footprint_well_formed:site{footprint.site}",
                proved=error is None,
                details="valid" if error is None else error,
            )
        )
    if any(not check.proved for check in checks):
        return ProofResult(checks=checks, diagnostics=diagnostics)

    if satisfiability == "unsat":
        checks.append(
            ProofCheck(
                name="vacuous_preconditions",
                proved=True,
                details="quantifier-free theorem preconditions are unsatisfiable",
            )
        )
        return ProofResult(checks=checks, diagnostics=diagnostics)

    by_base: dict[str, list[WriteFootprint]] = {}
    for footprint in footprints:
        by_base.setdefault(footprint.store.base.name, []).append(footprint)

    for base, base_footprints in sorted(by_base.items()):
        for first, second in combinations_with_replacement(base_footprints, 2):
            checks.append(_prove_pair_disjoint(config, first, second))

    return ProofResult(checks=checks, diagnostics=diagnostics)


def prove_logical_write_disjointness_from_annotations(
    source: str,
    kernel_name: str,
    constants: dict[str, int | float | bool],
    *,
    goal_name: str | None = None,
    timeout_ms: int = 1000,
) -> ProofResult:
    """Run the logical race proof from the same validated source context.

    This is deliberately separate from relational equality verification: logical
    write disjointness is diagnostic until physical layout and allocation
    identity are connected to the framework launch contract.
    """

    prepared = prepare_annotation_proof(
        source, kernel_name, constants, goal_name=goal_name
    )
    from ir.annotation_to_config import build_left_launch_assumptions

    launch_assumptions = build_left_launch_assumptions(
        prepared.annotation,
        prepared.first_config.left_env,
        prepared.first_config.right_env,
    )
    race_result = prove_logical_write_disjointness(
        LogicalRaceConfig(
            kernel=prepared.kernel,
            env=prepared.first_config.left_env,
            assumptions=launch_assumptions,
            timeout_ms=timeout_ms,
        )
    )
    return race_result
