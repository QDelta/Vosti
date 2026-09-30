"""Shared regional SMT obligations used by the unified relational analysis.

This module does not prepare source goals or issue proof artifacts.
"""
from dataclasses import dataclass, field
from collections.abc import Sequence
from . import (
    Arange, Assign, BinOp, BoolLit, BoolType, BroadcastTo, Cast, Exp2, Expr,
    FloatLit, For, Full, If, IntLit, IntType, Kernel, Let, Log2, MaskedLoad,
    MaskedStore, Max, Maximum, Min, Not, ReduceMax, ReduceSum, Region, Rsqrt,
    Sigmoid, Squeeze, Stmt, TensorIndex, TensorType, TensorView, Transpose,
    Type, Unsqueeze, Var, Where, Zeros,
)
from .preprocess import build_type_env
from .names import fresh_name
from .instantiation import suggest_instantiations
from .regions import (
    bound_variable_regions, collect_expr_vars, collect_write_stmts,
    get_read_tensors, ConditionalRegion, ConditionalWrite, GuardedRegion,
)
from .output_coverage import UnsupportedOutputCoverage, build_output_coverage_claim
from .smt import (
    Z3Val, ProofCheck, ProofResult, expr_to_z3, z3_prove, z3_satisfiable,
    z3_satisfiability_diagnostic,
)
import z3
from z3.z3util import get_vars
from itertools import product


@dataclass(frozen=True)
class IterRange:
    start: Expr
    stop: Expr


@dataclass(frozen=True)
class SingletonAnnotation:
    """Optional side-specific singleton witness expressions for a loop variable."""

    left: Expr | None = None
    right: Expr | None = None


def extract_loop_iter_ranges(kernel: Kernel) -> dict[str, IterRange]:
    ranges: dict[str, IterRange] = {}

    for grid_iter in kernel.grid.iters:
        ranges[grid_iter.var.name] = IterRange(
            start=grid_iter.iters.start,
            stop=grid_iter.iters.stop,
        )

    def visit(stmt_list: Sequence[Stmt]):
        for stmt in stmt_list:
            match stmt:
                case For(var=var, iters=iters, body=body):
                    ranges[var.name] = IterRange(
                        start=iters.start,
                        stop=iters.stop,
                    )
                    visit(body)
                case If(then_body=then_body, else_body=else_body):
                    visit(then_body)
                    visit(else_body)
                case Assign() | Let() | MaskedStore():
                    pass
                case _:
                    raise AssertionError(f"unhandled stmt: {stmt}")

    visit(kernel.grid.body)
    return ranges


def _written_tensor_names(
    write: Assign | MaskedStore, type_env: dict[str, Type]
) -> set[str]:
    match write:
        case Assign(target=Var(name=name)):
            return {name} if isinstance(type_env[name], TensorType) else set()
        case Assign(target=TensorView(base=base)):
            return {base.name}
        case MaskedStore(base=base):
            return {base.name}
        case _:
            raise AssertionError(f"unhandled write: {write}")


def relevant_output_writes(
    kernel: Kernel, output_tensor: str
) -> list[ConditionalWrite]:
    """Return writes in the conservative backward value slice of an output.

    This is shared proof infrastructure rather than a renderer detail.  In
    particular, the relational dataflow analysis uses the exact same slice to
    account for the static operations and path predicates whose dynamic
    instances must align between the two executions.
    """
    type_env = build_type_env(kernel)
    writes = collect_write_stmts(kernel)
    live = {output_tensor}
    selected: set[int] = set()
    changed = True
    while changed:
        changed = False
        for index, conditional_write in enumerate(writes):
            if index in selected:
                continue
            write = conditional_write.write
            if not (_written_tensor_names(write, type_env) & live):
                continue
            selected.add(index)
            live |= get_read_tensors(write.value, type_env)
            changed = True
    return [write for index, write in enumerate(writes) if index in selected]


def _collect_scalar_value_exprs(expr: Expr, out: list[Expr]) -> None:
    """Collect scalar expressions broadcast into tensor computation.

    Tensor inputs are handled by region assumptions.  Scalar expressions are
    a separate semantic dependency: the same tensor footprint is insufficient
    when, for example, a singleton program id or an unconstrained ``eps`` is
    added to the output value.
    """
    if not isinstance(expr.type, TensorType):
        if not isinstance(expr, (IntLit, FloatLit, BoolLit)):
            out.append(expr)
        return

    match expr:
        case Var():
            return
        case TensorView() | MaskedLoad() | TensorIndex():
            return
        case Zeros():
            return
        case Full(value=value):
            _collect_scalar_value_exprs(value, out)
        case Arange(start=start, stop=stop):
            _collect_scalar_value_exprs(start, out)
            _collect_scalar_value_exprs(stop, out)
        case BinOp(lhs=lhs, rhs=rhs) | Maximum(lhs=lhs, rhs=rhs):
            _collect_scalar_value_exprs(lhs, out)
            _collect_scalar_value_exprs(rhs, out)
        case Min(args=args) | Max(args=args):
            for arg in args:
                _collect_scalar_value_exprs(arg, out)
        case Where(cond=cond, on_true=on_true, on_false=on_false):
            _collect_scalar_value_exprs(cond, out)
            _collect_scalar_value_exprs(on_true, out)
            _collect_scalar_value_exprs(on_false, out)
        case (
            ReduceMax(value=value)
            | ReduceSum(value=value)
            | Exp2(value=value)
            | Sigmoid(value=value)
            | Rsqrt(value=value)
            | Log2(value=value)
            | Cast(value=value)
            | Not(value=value)
            | Unsqueeze(value=value)
            | Squeeze(value=value)
            | Transpose(value=value)
            | BroadcastTo(value=value)
        ):
            _collect_scalar_value_exprs(value, out)
        case _:
            raise AssertionError(f"unhandled value expression: {expr}")


def _collect_block_tensor_reads(expr: Expr, out: set[str]) -> None:
    """Collect tensor-valued memory reads that need a value assumption.

    Scalar TensorIndex reads are instead related through Z3 expressions (for
    example sequence lengths used in control/address arithmetic).
    """
    match expr:
        case TensorView(base=base) | MaskedLoad(base=base):
            out.add(base.name)
        case TensorIndex():
            return
        case Var(name=name):
            out.add(name)
        case IntLit() | FloatLit() | BoolLit() | Zeros() | Arange():
            return
        case Full(value=value):
            _collect_block_tensor_reads(value, out)
        case BinOp(lhs=lhs, rhs=rhs) | Maximum(lhs=lhs, rhs=rhs):
            _collect_block_tensor_reads(lhs, out)
            _collect_block_tensor_reads(rhs, out)
        case Min(args=args) | Max(args=args):
            for arg in args:
                _collect_block_tensor_reads(arg, out)
        case Where(cond=cond, on_true=on_true, on_false=on_false):
            _collect_block_tensor_reads(cond, out)
            _collect_block_tensor_reads(on_true, out)
            _collect_block_tensor_reads(on_false, out)
        case (
            ReduceMax(value=value)
            | ReduceSum(value=value)
            | Exp2(value=value)
            | Sigmoid(value=value)
            | Rsqrt(value=value)
            | Log2(value=value)
            | Cast(value=value)
            | Not(value=value)
            | Unsqueeze(value=value)
            | Squeeze(value=value)
            | Transpose(value=value)
            | BroadcastTo(value=value)
        ):
            _collect_block_tensor_reads(value, out)
        case _:
            raise AssertionError(f"unhandled value expression: {expr}")


@dataclass(frozen=True)
class ScopedScalarDependency:
    """A value-relevant scalar and the local loops enclosing its write."""

    expression: Expr
    iterators: frozenset[str]


def collect_scoped_output_value_dependencies(
    kernel: Kernel,
    outputs: Sequence[str],
    *,
    defer_discrete_assignment_scalars: bool = False,
) -> tuple[list[ScopedScalarDependency], set[str]]:
    """Return scalar dependencies with lexical loop scope and tensor inputs."""
    scoped_scalars: list[ScopedScalarDependency] = []
    block_tensor_reads: set[str] = set()
    for output in outputs:
        for conditional_write in relevant_output_writes(kernel, output):
            scalar_exprs: list[Expr] = []
            write = conditional_write.write
            value = write.value
            target_type: Type | None = None
            if isinstance(write, Assign):
                if isinstance(write.target, Var):
                    target_type = write.target.type
                elif isinstance(write.target, TensorView):
                    target_type = write.target.type
            target_elem_type = (
                target_type.elem_type
                if isinstance(target_type, TensorType)
                else target_type
            )
            if not (
                defer_discrete_assignment_scalars
                and isinstance(target_elem_type, (IntType, BoolType))
            ):
                _collect_scalar_value_exprs(value, scalar_exprs)
            _collect_block_tensor_reads(value, block_tensor_reads)
            scalar_exprs.extend(conditional_write.conditions)
            for expression in scalar_exprs:
                dependency = ScopedScalarDependency(
                    expression=expression,
                    iterators=conditional_write.iterators,
                )
                if dependency not in scoped_scalars:
                    scoped_scalars.append(dependency)
    return scoped_scalars, block_tensor_reads


def region_equiv_claim(
    left_region: Region,
    right_region: Region,
    left_env: dict[str, Z3Val],
    right_env: dict[str, Z3Val],
    left_eq_region: Region,
    right_eq_region: Region,
):
    assert (
        len(left_region)
        == len(right_region)
        == len(left_eq_region)
        == len(right_eq_region)
    )

    clauses = []
    for left_slice, right_slice, left_equiv, right_equiv in zip(
        left_region,
        right_region,
        left_eq_region,
        right_eq_region,
    ):
        left_start = expr_to_z3(left_slice.start, left_env)
        left_stop = expr_to_z3(left_slice.stop, left_env)
        right_start = expr_to_z3(right_slice.start, right_env)
        right_stop = expr_to_z3(right_slice.stop, right_env)

        left_equiv_start = expr_to_z3(left_equiv.start, left_env)
        left_equiv_stop = expr_to_z3(left_equiv.stop, left_env)
        right_equiv_start = expr_to_z3(right_equiv.start, right_env)
        right_equiv_stop = expr_to_z3(right_equiv.stop, right_env)

        clauses.append(left_start >= left_equiv_start)  # pyright: ignore[reportOperatorIssue]
        clauses.append(left_stop <= left_equiv_stop)  # pyright: ignore[reportOperatorIssue]
        clauses.append(right_start >= right_equiv_start)  # pyright: ignore[reportOperatorIssue]
        clauses.append(right_stop <= right_equiv_stop)  # pyright: ignore[reportOperatorIssue]

        clauses.append(left_start - left_equiv_start == right_start - right_equiv_start)  # pyright: ignore[reportOperatorIssue]
        clauses.append(left_stop - left_equiv_stop == right_stop - right_equiv_stop)  # pyright: ignore[reportOperatorIssue]

    return z3.And(*clauses)


def point_region_equiv_claim(
    left_coordinates: list[z3.ArithRef],
    right_coordinates: list[z3.ArithRef],
    left_env: dict[str, Z3Val],
    right_env: dict[str, Z3Val],
    left_eq_region: Region,
    right_eq_region: Region,
) -> z3.BoolRef:
    """Pointwise membership in one declared ordinal-mapped equality region."""

    assert (
        len(left_coordinates)
        == len(right_coordinates)
        == len(left_eq_region)
        == len(right_eq_region)
    )
    clauses: list[z3.BoolRef] = []
    for left_coordinate, right_coordinate, left_equiv, right_equiv in zip(
        left_coordinates,
        right_coordinates,
        left_eq_region,
        right_eq_region,
    ):
        left_start = expr_to_z3(left_equiv.start, left_env)
        left_stop = expr_to_z3(left_equiv.stop, left_env)
        right_start = expr_to_z3(right_equiv.start, right_env)
        right_stop = expr_to_z3(right_equiv.stop, right_env)
        clauses.extend(
            (
                left_coordinate >= left_start,
                left_coordinate < left_stop,
                right_coordinate >= right_start,
                right_coordinate < right_stop,
                left_coordinate - left_start == right_coordinate - right_start,
            )
        )
    return z3.And(*clauses)


@dataclass(frozen=True)
class TensorAssumption:
    left_region: Region
    right_region: Region
    # Additional tensor-specific facts that must be derivable from the current
    # proof context before this assumption can be used.
    conditions: tuple[z3.BoolRef | bool, ...] = ()
    # Universally quantified regional schemas carry their bound SMT symbols
    # and domain separately.  A proof may instantiate them with dynamic loop
    # ordinals only after discharging every schema condition.
    schema_variables: tuple[str, ...] = ()
    schema_symbols: tuple[z3.ArithRef, ...] = ()
    schema_conditions: tuple[z3.BoolRef | bool, ...] = ()

    def claim(
        self,
        left_region: Region,
        right_region: Region,
        left_env: dict[str, Z3Val],
        right_env: dict[str, Z3Val],
    ) -> z3.BoolRef:
        return region_equiv_claim(
            left_region,
            right_region,
            left_env,
            right_env,
            self.left_region,
            self.right_region,
        )


@dataclass(frozen=True)
class EquivProofConfig:
    """Instance-specific inputs for the region-equivalence proof."""

    kernel: Kernel
    output_tensor: str
    left_output_region: Region
    right_output_region: Region
    tensor_assumptions: dict[str, TensorAssumption]
    left_env: dict[str, Z3Val]
    right_env: dict[str, Z3Val]
    base_assumptions: list[z3.BoolRef | bool]
    singleton_annotations: dict[str, SingletonAnnotation] = field(default_factory=dict)


def prove_region_equivalence(
    config: EquivProofConfig,
    *,
    left_dependency_regions: dict[str, Region] | None = None,
    right_dependency_regions: dict[str, Region] | None = None,
    left_guarded_dependency_regions: dict[str, GuardedRegion] | None = None,
    right_guarded_dependency_regions: dict[str, GuardedRegion] | None = None,
    required_tensor_reads: set[str] | None = None,
    ordered_range_difference_iterators: frozenset[str] = frozenset(),
    defer_discrete_assignment_scalars: bool = False,
) -> ProofResult:
    """General structural-equivalence proof driver.

    Given a kernel and left/right output-region specifications, proves:
      1. Each loop variable is either range-equal, singleton-relevant, or is
         explicitly delegated to an ordered range-difference proof.
      2. Under those conditions, every input tensor's accessed region on the
         left is equivalent to its accessed region on the right (modulo
         the declared tensor_assumptions).
      3. Scalar expressions that influence output values or contributing
         control-flow are equal.

    ``ordered_range_difference_iterators`` does not itself erase any dynamic
    iteration.  It only permits the common intersection to be checked at one
    shared ordinal; the caller must separately prove exact identity on both
    one-sided range differences before exporting a theorem.

    The final semantic lift from these structural facts to output-value
    equality is a trusted meta-argument about the translated IR, not an SMT
    encoding of floating-point tensor execution. See README.md.
    """
    inferred_left_regions, left_written = bound_variable_regions(
        config.kernel, config.output_tensor, config.left_output_region
    )
    inferred_right_regions, right_written = bound_variable_regions(
        config.kernel, config.output_tensor, config.right_output_region
    )
    left_regions = (
        inferred_left_regions
        if left_dependency_regions is None
        else left_dependency_regions
    )
    right_regions = (
        inferred_right_regions
        if right_dependency_regions is None
        else right_dependency_regions
    )

    loop_ranges = extract_loop_iter_ranges(config.kernel)

    checks: list[ProofCheck] = []
    diagnostics: list[ProofCheck] = []

    left_env = config.left_env
    right_env = config.right_env
    base_assumptions = config.base_assumptions
    singleton_annotations = config.singleton_annotations

    # Z3 commonly returns unknown for the quantified monotonicity clauses used
    # by varlen kernels. Check the quantifier-free core as an early diagnostic;
    # the quantified clauses remain ordinary conditional Hoare premises.
    quantifier_free_assumptions = [
        assumption
        for assumption in base_assumptions
        if not isinstance(assumption, z3.QuantifierRef)
    ]
    satisfiability, diagnostic = z3_satisfiability_diagnostic(
        "quantifier_free_preconditions_satisfiable",
        quantifier_free_assumptions,
    )
    diagnostics.append(diagnostic)
    if satisfiability == "unsat":
        return ProofResult(
            checks=[
                ProofCheck(
                    name="vacuous_preconditions",
                    proved=True,
                    details="quantifier-free theorem preconditions are unsatisfiable",
                )
            ],
            diagnostics=diagnostics,
        )

    def output_coverage(
        name: str,
        output_region: Region,
        written: list[ConditionalRegion],
        env: dict[str, Z3Val],
    ) -> ProofCheck:
        """Prove every annotated output cell is reached by a source write."""

        try:
            claim = build_output_coverage_claim(
                name=name,
                kernel=config.kernel,
                output_region=output_region,
                written=written,
                loop_ranges={
                    iterator: (iter_range.start, iter_range.stop)
                    for iterator, iter_range in loop_ranges.items()
                },
                env=env,
                translate=expr_to_z3,
            )
        except UnsupportedOutputCoverage as error:
            return ProofCheck(name, False, str(error))
        return z3_prove(name, base_assumptions, claim, timeout=10_000)

    coverage_checks = [
        output_coverage(
            "left_output_fully_written",
            config.left_output_region,
            left_written,
            left_env,
        ),
        output_coverage(
            "right_output_fully_written",
            config.right_output_region,
            right_written,
            right_env,
        ),
    ]
    checks.extend(coverage_checks)

    output_iter_names = {
        grid_iter.var.name for grid_iter in config.kernel.grid.iters
    } | {
        iterator
        for conditional in (*left_written, *right_written)
        for iterator in conditional.iterators
    }

    def iter_assumptions(
        env: dict[str, Z3Val],
        iter_names: set[str] | frozenset[str] | None = None,
    ) -> list[z3.BoolRef]:
        clauses = []
        selected_names = output_iter_names if iter_names is None else iter_names
        for iter_name in sorted(selected_names):
            iter_range = loop_ranges[iter_name]
            iter_var = env[iter_name]
            clauses.append(iter_var >= expr_to_z3(iter_range.start, env))  # pyright: ignore[reportOperatorIssue]
            clauses.append(iter_var < expr_to_z3(iter_range.stop, env))  # pyright: ignore[reportOperatorIssue]
        return clauses

    def output_nonempty(written: list[ConditionalRegion], env: dict[str, Z3Val]):
        """Disjunction: at least one conditional region is active and nonempty."""
        alternatives = []
        for cr in written:
            cond_clauses = [expr_to_z3(c, env) for c in cr.conditions]
            region_clauses = [
                expr_to_z3(sl.start, env) < expr_to_z3(sl.stop, env)  # pyright: ignore[reportOperatorIssue]
                for sl in cr.region
            ]
            alternatives.append(z3.And(*(cond_clauses + region_clauses)))
        if not alternatives:
            return z3.BoolVal(False)
        return z3.Or(*alternatives)

    def left(
        override_env: dict[str, Z3Val] | None = None,
        extra_iter_names: set[str] | frozenset[str] = frozenset(),
    ):
        env = left_env.copy()
        if override_env is not None:
            env.update(override_env)
        return (
            iter_assumptions(env, output_iter_names | set(extra_iter_names)),
            output_nonempty(left_written, env),
        )

    def right(
        override_env: dict[str, Z3Val] | None = None,
        extra_iter_names: set[str] | frozenset[str] = frozenset(),
    ):
        env = right_env.copy()
        if override_env is not None:
            env.update(override_env)
        return (
            iter_assumptions(env, output_iter_names | set(extra_iter_names)),
            output_nonempty(right_written, env),
        )

    # --- Step 1: loop-variable classification ---
    iter_mode: dict[str, str] = {}
    iter_relation_assumptions: list[z3.BoolRef | bool] = []

    left_iter_assumptions, left_nonempty = left()
    right_iter_assumptions, right_nonempty = right()

    # All later relational checks assume both selected executions reach a
    # non-empty output region. Reject an inconsistent quantifier-free execution
    # context as an early diagnostic. Quantified clauses are omitted from this
    # SAT query because Z3 cannot generally return SAT for these kernels; they
    # remain assumptions of the conditional result.
    left_context = z3_satisfiable(
        "left_execution_context_satisfiable",
        quantifier_free_assumptions
        + left_iter_assumptions
        + [left_nonempty],
    )
    right_context = z3_satisfiable(
        "right_execution_context_satisfiable",
        quantifier_free_assumptions
        + right_iter_assumptions
        + [right_nonempty],
    )
    checks.extend([left_context, right_context])
    if not left_context.proved or not right_context.proved:
        return ProofResult(checks=checks, diagnostics=diagnostics)

    for iter_name, iter_range in loop_ranges.items():
        deps = collect_expr_vars(iter_range.start) | collect_expr_vars(iter_range.stop)
        depends_on_singleton_outer = any(
            iter_mode.get(dep) == "singleton" for dep in deps
        )

        left_start = expr_to_z3(iter_range.start, left_env)
        left_stop = expr_to_z3(iter_range.stop, left_env)
        right_start = expr_to_z3(iter_range.start, right_env)
        right_stop = expr_to_z3(iter_range.stop, right_env)
        same_range_check = z3_prove(
            f"iter_{iter_name}_same_range",
            base_assumptions
            + iter_relation_assumptions,
            z3.And(left_start == right_start, left_stop == right_stop),
        )

        if same_range_check.proved:
            iter_mode[iter_name] = "range_equal"
            iter_relation_assumptions.append(
                left_env[iter_name] == right_env[iter_name]
            )
            checks.append(same_range_check)
            continue

        # --- Try effective range equal ---
        # When syntactic bounds differ but the set of "effectful" values
        # (those that actually produce output) is provably the same,
        # we can still treat the variable as range-equal.
        #
        # We build left_eff(v) := v∈left_range ∧ left_nonempty(v)
        # and        right_eff(v) := v∈right_range ∧ right_nonempty(v)
        # using ONLY the current variable's range constraint (not all
        # iter_assumptions) to avoid spurious failures from unrelated
        # unconstrained loop variables.  Other variables referenced
        # inside output_nonempty are constrained by base_assumptions
        # and iter_relation_assumptions passed to z3_prove.
        eff_v = z3.FreshInt(f"{iter_name}_eff")
        left_env_eff = {**left_env, iter_name: eff_v}
        right_env_eff = {**right_env, iter_name: eff_v}
        left_nonempty_eff = output_nonempty(left_written, left_env_eff)
        right_nonempty_eff = output_nonempty(right_written, right_env_eff)
        left_effective = z3.And(
            eff_v >= left_start,  # pyright: ignore[reportOperatorIssue]
            eff_v < left_stop,  # pyright: ignore[reportOperatorIssue]
            left_nonempty_eff,
        )
        right_effective = z3.And(
            eff_v >= right_start,  # pyright: ignore[reportOperatorIssue]
            eff_v < right_stop,  # pyright: ignore[reportOperatorIssue]
            right_nonempty_eff,
        )
        eff_range_check = z3_prove(
            f"iter_{iter_name}_effective_range_equal",
            base_assumptions + iter_relation_assumptions,
            left_effective == right_effective,
        )

        if eff_range_check.proved:
            iter_mode[iter_name] = "effective_range_equal"
            iter_relation_assumptions.append(
                left_env[iter_name] == right_env[iter_name]
            )
            checks.append(eff_range_check)
            continue

        lv1, lv2 = (z3.FreshInt(f"{iter_name}_l{i}") for i in (1, 2))
        rv1, rv2 = (z3.FreshInt(f"{iter_name}_r{i}") for i in (1, 2))

        lv1_assumptions, lv1_nonempty = left(
            {iter_name: lv1}, {iter_name}
        )
        lv2_assumptions, lv2_nonempty = left(
            {iter_name: lv2}, {iter_name}
        )
        rv1_assumptions, rv1_nonempty = right(
            {iter_name: rv1}, {iter_name}
        )
        rv2_assumptions, rv2_nonempty = right(
            {iter_name: rv2}, {iter_name}
        )

        annotation = singleton_annotations.get(iter_name)
        if annotation is not None and annotation.left is not None:
            left_singleton_goal = z3.Implies(
                z3.And(*(lv1_assumptions + [lv1_nonempty])),
                lv1 == expr_to_z3(annotation.left, left_env),
            )
        else:
            # Existence is already established by the side-specific
            # execution-context SAT gate.  Here prove uniqueness only; an
            # existential with implication would be vacuous, while re-encoding
            # the full multi-iterator witness makes nonlinear grid cases return
            # unknown without strengthening the checked context.
            left_singleton_goal = z3.Implies(
                z3.And(
                    *(
                        lv1_assumptions
                        + [lv1_nonempty]
                        + lv2_assumptions
                        + [lv2_nonempty]
                    )
                ),
                lv1 == lv2,
            )

        if annotation is not None and annotation.right is not None:
            right_singleton_goal = z3.Implies(
                z3.And(*(rv1_assumptions + [rv1_nonempty])),
                rv1 == expr_to_z3(annotation.right, right_env),
            )
        else:
            right_singleton_goal = z3.Implies(
                z3.And(
                    *(
                        rv1_assumptions
                        + [rv1_nonempty]
                        + rv2_assumptions
                        + [rv2_nonempty]
                    )
                ),
                rv1 == rv2,
            )

        singleton_claim = z3.And(left_singleton_goal, right_singleton_goal)

        singleton_check = z3_prove(
            f"iter_{iter_name}_singleton",
            base_assumptions + iter_relation_assumptions,
            singleton_claim,
        )

        if singleton_check.proved:
            # If this loop's bounds depend on an outer singleton loop var,
            # singleton classification alone is not enough: the two sides may
            # pick different singleton witnesses. Keep this conservative reject.
            if depends_on_singleton_outer:
                iter_mode[iter_name] = "error"
                checks.append(
                    ProofCheck(
                        name=f"iter_{iter_name}",
                        proved=False,
                        details=(
                            "range depends on singleton outer loopvar and "
                            "could not prove same_range"
                        ),
                    )
                )
            else:
                iter_mode[iter_name] = "singleton"
                if annotation is not None:
                    if annotation.left is not None:
                        iter_relation_assumptions.append(
                            left_env[iter_name] == expr_to_z3(annotation.left, left_env)
                        )
                    if annotation.right is not None:
                        iter_relation_assumptions.append(
                            right_env[iter_name]
                            == expr_to_z3(annotation.right, right_env)
                        )
                checks.append(singleton_check)
        else:
            if iter_name in ordered_range_difference_iterators:
                iter_mode[iter_name] = "ordered_range_difference"
                iter_relation_assumptions.append(
                    left_env[iter_name] == right_env[iter_name]
                )
                checks.append(
                    ProofCheck(
                        name=f"iter_{iter_name}_ordered_shared_ordinal",
                        proved=True,
                        details=(
                            "common iterations paired by one ascending "
                            "unit-stride source ordinal; one-sided iterations "
                            "remain caller obligations"
                        ),
                    )
                )
                continue
            iter_mode[iter_name] = "error"
            checks.append(
                ProofCheck(
                    name=f"iter_{iter_name}",
                    proved=False,
                    details=(
                        f"same_range_failed={same_range_check.details}; "
                        f"effective_range_failed={eff_range_check.details}; "
                        f"singleton_failed={singleton_check.details}"
                    ),
                )
            )

    # --- Step 2: region equivalence for every input tensor ---
    region_assumptions = (
        left_iter_assumptions
        + [left_nonempty]
        + right_iter_assumptions
        + [right_nonempty]
        + iter_relation_assumptions
    )

    # Iterator pairing is itself an added proof assumption.  Equal ranges do
    # not guarantee that the two sides have a common effectful iterator (their
    # path conditions may select disjoint points).  Without this joint SAT
    # check, every region and scalar claim below can become vacuous even though
    # the separate left/right context checks above both succeeded.
    relational_context = z3_satisfiable(
        "relational_execution_context_satisfiable",
        quantifier_free_assumptions + region_assumptions,
    )
    checks.append(relational_context)
    if not relational_context.proved:
        return ProofResult(checks=checks, diagnostics=diagnostics)

    # Regional equality alone does not imply output-value equality if a
    # value-relevant scalar differs. Prove every scalar expression broadcast
    # into the output's backward value slice, plus every path condition that
    # selects a contributing write.
    scoped_scalar_exprs, block_tensor_reads = collect_scoped_output_value_dependencies(
        config.kernel,
        (config.output_tensor,),
        defer_discrete_assignment_scalars=defer_discrete_assignment_scalars,
    )
    deferred_obligations: tuple[str, ...] = ()
    if defer_discrete_assignment_scalars:
        complete_scalar_exprs, _ = collect_scoped_output_value_dependencies(
            config.kernel,
            (config.output_tensor,),
        )
        deferred_obligations = tuple(
            f"discrete_assignment_scalar_{index}"
            for index, dependency in enumerate(complete_scalar_exprs)
            if dependency not in scoped_scalar_exprs
        )

    for index, dependency in enumerate(scoped_scalar_exprs):
        expr = dependency.expression
        name = f"value_scalar_equiv_{index}"
        try:
            left_value = expr_to_z3(expr, left_env)
            right_value = expr_to_z3(expr, right_env)
        except (AssertionError, KeyError, TypeError, ValueError, z3.Z3Exception) as error:
            checks.append(
                ProofCheck(
                    name=name,
                    proved=False,
                    details=f"cannot relate value scalar {expr}: {error}",
                )
            )
            continue
        checks.append(
            z3_prove(
                name,
                base_assumptions
                + region_assumptions
                + iter_assumptions(left_env, dependency.iterators)
                + iter_assumptions(right_env, dependency.iterators),
                left_value == right_value,
            )
        )

    # Every tensor-valued memory read in the output value slice must have an
    # explicit value-equality assumption. Previously the driver iterated only
    # over assumptions that happened to be present, silently accepting omitted
    # inputs.
    tensor_param_names = {
        param.name
        for param in config.kernel.params
        if isinstance(param.type, TensorType)
    }
    required_tensor_assumptions = (
        block_tensor_reads
        if required_tensor_reads is None
        else required_tensor_reads
    ) & tensor_param_names
    missing_tensor_assumptions = sorted(
        required_tensor_assumptions - set(config.tensor_assumptions)
    )
    for tensor in missing_tensor_assumptions:
        checks.append(
            ProofCheck(
                name=f"tensor_assumption_{tensor}",
                proved=False,
                details="tensor-valued input read has no pre(...) region equality or tensor same(...) assumption",
            )
        )

    # A required value input with no inferred region is not "unused": it means
    # the backward analysis failed to connect a syntactically contributing
    # read to the selected output.  Treat that as an analyzer failure rather
    # than silently skipping the tensor's region obligation below.
    missing_inferred_regions = sorted(
        required_tensor_assumptions
        - (set(left_regions) & set(right_regions))
    )
    for tensor in missing_inferred_regions:
        checks.append(
            ProofCheck(
                name=f"tensor_region_inferred_{tensor}",
                proved=False,
                details=(
                    "required tensor-valued input read has no inferred region "
                    "on one or both sides"
                ),
            )
        )

    for tensor, assumption in config.tensor_assumptions.items():
        if tensor not in left_regions or tensor not in right_regions:
            # A shared pre(...) may cover inputs used by a different post(...) output.
            # Extra assumptions are harmless; required-but-missing assumptions
            # were rejected above.
            continue
        tensor_iter_names = (
            {
                name
                for region in (left_regions[tensor], right_regions[tensor])
                for region_slice in region
                for bound in (region_slice.start, region_slice.stop)
                for name in collect_expr_vars(bound)
                if name in loop_ranges
            }
            | {
                name
                for condition in assumption.conditions
                if isinstance(condition, z3.AstRef)
                for symbol in get_vars(condition)
                for name in loop_ranges
                if (
                    isinstance(left_env[name], z3.AstRef)
                    and symbol.eq(left_env[name])
                ) or (
                    isinstance(right_env[name], z3.AstRef)
                    and symbol.eq(right_env[name])
                )
            }
        )
        tensor_iter_assumptions = (
            iter_assumptions(left_env, tensor_iter_names)
            + iter_assumptions(right_env, tensor_iter_names)
        )
        activity: z3.BoolRef | bool = True
        guard_alignment_check: ProofCheck | None = None
        guarded_left_coordinates: list[z3.ArithRef] | None = None
        guarded_right_coordinates: list[z3.ArithRef] | None = None
        if (
            left_guarded_dependency_regions is not None
            and right_guarded_dependency_regions is not None
            and tensor in left_guarded_dependency_regions
            and tensor in right_guarded_dependency_regions
        ):
            from .positional import subst_free_indices

            left_guarded = left_guarded_dependency_regions[tensor]
            right_guarded = right_guarded_dependency_regions[tensor]
            if not (
                len(left_guarded.region)
                == len(right_guarded.region)
                == left_guarded.guard.rank
                == right_guarded.guard.rank
            ):
                checks.append(
                    ProofCheck(
                        name=f"tensor_guard_alignment_{tensor}",
                        proved=False,
                        details="guard and region ranks differ",
                    )
                )
                continue
            guard_left_env = dict(left_env)
            guard_right_env = dict(right_env)
            left_mapping: dict[str, Expr] = {}
            right_mapping: dict[str, Expr] = {}
            left_membership: list[z3.BoolRef] = []
            right_membership: list[z3.BoolRef] = []
            guarded_left_coordinates = []
            guarded_right_coordinates = []
            try:
                for axis, (left_interval, right_interval) in enumerate(
                    zip(left_guarded.region, right_guarded.region)
                ):
                    coordinate_name = fresh_name(
                        f"__tensor_guard_{tensor}_{axis}",
                        set(guard_left_env) | set(guard_right_env),
                    )
                    coordinate = z3.FreshInt(coordinate_name)
                    coordinate_expr = Var(coordinate_name, type=IntType())
                    guard_left_env[coordinate_name] = coordinate
                    guard_right_env[coordinate_name] = coordinate
                    left_mapping[f"_i{axis}"] = BinOp(
                        "+", left_interval.start, coordinate_expr
                    )
                    right_mapping[f"_i{axis}"] = BinOp(
                        "+", right_interval.start, coordinate_expr
                    )
                    guarded_left_coordinates.append(
                        expr_to_z3(left_mapping[f"_i{axis}"], guard_left_env)
                    )
                    guarded_right_coordinates.append(
                        expr_to_z3(right_mapping[f"_i{axis}"], guard_right_env)
                    )
                    left_extent = (
                        expr_to_z3(left_interval.stop, guard_left_env)
                        - expr_to_z3(left_interval.start, guard_left_env)
                    )
                    right_extent = (
                        expr_to_z3(right_interval.stop, guard_right_env)
                        - expr_to_z3(right_interval.start, guard_right_env)
                    )
                    left_membership.extend(
                        (coordinate >= 0, coordinate < left_extent)
                    )
                    right_membership.extend(
                        (coordinate >= 0, coordinate < right_extent)
                    )
                left_guard = expr_to_z3(
                    subst_free_indices(left_guarded.guard.body, left_mapping),
                    guard_left_env,
                )
                right_guard = expr_to_z3(
                    subst_free_indices(right_guarded.guard.body, right_mapping),
                    guard_right_env,
                )
                left_active = z3.And(*left_membership, left_guard)
                right_active = z3.And(*right_membership, right_guard)
                activity = z3.Or(left_active, right_active)
                guard_alignment_check = z3_prove(
                    f"tensor_guard_alignment_{tensor}",
                    base_assumptions
                    + region_assumptions
                    + tensor_iter_assumptions,
                    left_active == right_active,
                    timeout=10_000,
                )
                checks.append(guard_alignment_check)
            except (
                AssertionError,
                KeyError,
                TypeError,
                z3.Z3Exception,
            ) as error:
                checks.append(
                    ProofCheck(
                        name=f"tensor_guard_alignment_{tensor}",
                        proved=False,
                        details=f"cannot encode guarded demand: {error}",
                    )
                )
                continue
        schema_candidates: list[
            tuple[
                dict[str, Z3Val],
                dict[str, Z3Val],
                tuple[z3.BoolRef | bool, ...],
                str,
            ]
        ] = []
        if assumption.schema_variables:
            if not (
                len(assumption.schema_variables)
                == len(assumption.schema_symbols)
            ):
                checks.append(
                    ProofCheck(
                        name=f"tensor_region_equiv_{tensor}",
                        proved=False,
                        details="malformed quantified region schema metadata",
                    )
                )
                continue
            # Matching proposes terms only. Every candidate, including hints,
            # must pass the unchanged domain and demanded-region obligations.
            patterns, demands = [], []
            for pattern_region, demand_region, env in (
                (assumption.left_region, left_regions[tensor], left_env),
                (assumption.right_region, right_regions[tensor], right_env),
            ):
                schema_env = dict(env)
                schema_env.update(zip(assumption.schema_variables, assumption.schema_symbols))
                patterns.extend(
                    expr_to_z3(bound, schema_env)
                    for sl in pattern_region for bound in (sl.start, sl.stop)
                )
                demands.extend(
                    expr_to_z3(bound, env)
                    for sl in demand_region for bound in (sl.start, sl.stop)
                )
            bindings = suggest_instantiations(patterns, demands, assumption.schema_symbols)
            bindings.extend(tuple(left_env[name] for name in names) for names in product(
                sorted(loop_ranges), repeat=len(assumption.schema_variables)))
            seen_bindings = set()
            for values in bindings:
                key = tuple(value.get_id() for value in values)
                if key in seen_bindings:
                    continue
                seen_bindings.add(key)
                candidate_left = dict(left_env)
                candidate_right = dict(right_env)
                substitutions = []
                for variable, symbol, value in zip(
                    assumption.schema_variables,
                    assumption.schema_symbols,
                    values,
                ):
                    # Instantiate both sides with ONE logical coordinate.
                    # Cross-run equality/coverage is still an SMT obligation.
                    candidate_left[variable] = value
                    candidate_right[variable] = value
                    substitutions.append((symbol, value))
                instantiated = tuple(
                    z3.substitute(condition, *substitutions)
                    if isinstance(condition, z3.ExprRef)
                    else condition
                    for condition in assumption.schema_conditions
                )
                schema_candidates.append(
                    (
                        candidate_left,
                        candidate_right,
                        instantiated,
                        ",".join(str(value) for value in values),
                    )
                )
        else:
            schema_candidates.append((left_env, right_env, (), "none"))

        proved_schema: tuple[ProofCheck, ProofCheck] | None = None
        failed_schemas: list[str] = []
        for (
            candidate_left,
            candidate_right,
            schema_conditions,
            schema_label,
        ) in schema_candidates:
            substitutions = tuple(
                (symbol, candidate_left[variable])
                for variable, symbol in zip(
                    assumption.schema_variables,
                    assumption.schema_symbols,
                )
            )
            ordinary_conditions = tuple(
                z3.substitute(condition, *substitutions)
                if substitutions and isinstance(condition, z3.ExprRef)
                else condition
                for condition in assumption.conditions
            )
            all_conditions = ordinary_conditions + schema_conditions
            condition_check = z3_prove(
                f"tensor_condition_{tensor}",
                base_assumptions + region_assumptions + tensor_iter_assumptions,
                z3.Implies(activity, z3.And(*all_conditions)),
            )
            if not condition_check.proved:
                failed_schemas.append(
                    f"{schema_label}: domain={condition_check.details}; region=not attempted"
                )
                continue  # A region proof cannot make a failed domain admissible.
            if (
                guarded_left_coordinates is not None
                and guarded_right_coordinates is not None
            ):
                tensor_claim = point_region_equiv_claim(
                    guarded_left_coordinates,
                    guarded_right_coordinates,
                    candidate_left,
                    candidate_right,
                    assumption.left_region,
                    assumption.right_region,
                )
            else:
                tensor_claim = assumption.claim(
                    left_regions[tensor],
                    right_regions[tensor],
                    candidate_left,
                    candidate_right,
                )
            region_check = z3_prove(
                f"tensor_region_equiv_{tensor}",
                base_assumptions
                + region_assumptions
                + tensor_iter_assumptions
                + [z3.Implies(activity, z3.And(*all_conditions))],
                z3.Implies(activity, tensor_claim),
            )
            if condition_check.proved and region_check.proved:
                proved_schema = (condition_check, region_check)
                break
            failed_schemas.append(
                f"{schema_label}: domain={condition_check.details}; "
                f"region={region_check.details}"
            )

        if proved_schema is None:
            checks.append(
                ProofCheck(
                    name=f"tensor_condition_{tensor}",
                    proved=False,
                    details="no quantified-region instantiation proved: "
                    + " | ".join(failed_schemas),
                )
            )
            checks.append(
                ProofCheck(
                    name=f"tensor_region_equiv_{tensor}",
                    proved=False,
                    details="no quantified-region instantiation covers demand",
                )
            )
            continue
        condition_check, region_check = proved_schema
        if assumption.conditions or assumption.schema_conditions:
            checks.append(condition_check)
        checks.append(region_check)

    return ProofResult(
        checks=checks,
        diagnostics=diagnostics,
        deferred_obligations=deferred_obligations,
    )
