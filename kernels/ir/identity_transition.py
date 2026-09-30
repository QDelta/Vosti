"""Generic region-local exact-identity transitions.

This module composes the existing per-position forward facts with the guarded
backward dependency analysis.  It does not define another expression or
statement IR: every proof record below points at regions and facts over the
same typed kernel IR consumed by the structural verifier.

The first supported numeric fact is deliberately narrow: a named Boolean
tensor is universally false in a counterfactual execution.  If that makes the
whole iteration preserve each carried state exactly, backward analysis then
computes the smallest currently representable guarded region on which the
actual execution must agree with that counterfactual.  The result is a
conditional identity rule; a caller must separately prove the returned fact
requirements.
"""

from __future__ import annotations

from dataclasses import dataclass

import z3
from .names import fresh_name

from . import (
    Assign,
    BoolLit,
    BoolType,
    Expr,
    For,
    Grid,
    If,
    IntType,
    Kernel,
    Let,
    MaskedStore,
    Param,
    Stmt,
    TensorType,
    TensorView,
    Type,
    Var,
    collect_expr_vars,
)
from .positional import (
    NeutralityAssumptions,
    NeutralityReport,
    PositionalAnalyzer,
    TensorFacts,
    check_iteration_neutral,
    collect_statement_type_env,
    facts_unknown,
    subst_free_indices,
)
from .regions import (
    GuardedRegion,
    _collect_z3_symbols,
    bound_variable_regions_masked,
)
from .smt import ProofCheck, expr_to_z3, z3_prove


@dataclass(frozen=True)
class AllFalseFact:
    """Counterfactual value fact for one Boolean tensor."""

    variable: str


@dataclass(frozen=True)
class RegionalFactRequirement:
    """A fact that must hold on a guarded value-dependency region."""

    fact: AllFalseFact
    state: str
    demand: GuardedRegion


@dataclass(frozen=True)
class AvailableRegionalFact:
    """A caller-proved fact on an unconditional guarded region."""

    fact: AllFalseFact
    demand: GuardedRegion


@dataclass(frozen=True)
class StateIdentityEvidence:
    """Per-state evidence produced by one conditional transition proof."""

    state: str
    demand: GuardedRegion
    final_facts: TensorFacts
    dependencies: tuple[str, ...]
    fact_requirement: RegionalFactRequirement | None


@dataclass(frozen=True)
class IdentityTransitionReport:
    """Conditional exact-identity result for one source-loop body.

    ``proved`` means the analysis has established the implication represented
    by ``requirements``; it does not mean those regional requirements are true
    in a particular caller context.  Each requirement must be discharged by a
    separate structural or arithmetic proof before an unmatched iteration can
    be erased.
    """

    fact: AllFalseFact
    fact_facts: TensorFacts
    whole_report: NeutralityReport
    states: tuple[StateIdentityEvidence, ...]
    requirements: tuple[RegionalFactRequirement, ...]
    failures: tuple[str, ...]
    used_assumptions: tuple[str, ...]
    external_assumptions: tuple[str, ...]

    @property
    def proved(self) -> bool:
        return self.whole_report.proved and not self.failures


def _external_assumptions(
    used: set[str],
    fact: AllFalseFact,
) -> tuple[str, ...]:
    discharged = f"all_false_mask({fact.variable})"
    return tuple(sorted(assumption for assumption in used if assumption != discharged))


def _writes_variable(stmt: Stmt, variable: str) -> bool:
    match stmt:
        case Assign(target=Var(name=name)):
            return name == variable
        case Assign(target=TensorView(base=base)):
            return base.name == variable
        case MaskedStore(base=base):
            return base.name == variable
        case For(body=body):
            return any(_writes_variable(inner, variable) for inner in body)
        case If(then_body=then_body, else_body=else_body):
            return any(
                _writes_variable(inner, variable) for inner in (*then_body, *else_body)
            )
        case Let():
            return False
        case _:
            raise AssertionError(f"unhandled statement: {stmt!r}")


def _reads_variable(
    stmt: Stmt,
    variable: str,
) -> bool:
    """Audit reads recursively so a pre-cut value cannot escape via nesting."""

    match stmt:
        case Assign(value=value) | Let(value=value) | MaskedStore(value=value):
            return variable in collect_expr_vars(value)
        case If(cond=cond, then_body=then_body, else_body=else_body):
            return variable in collect_expr_vars(cond) or any(
                _reads_variable(inner, variable) for inner in (*then_body, *else_body)
            )
        case For(iters=iters, body=body):
            return (
                variable in collect_expr_vars(iters.start)
                or variable in collect_expr_vars(iters.stop)
                or any(_reads_variable(inner, variable) for inner in body)
            )
        case _:
            raise AssertionError(f"unhandled statement: {stmt!r}")


def _fragment_failures(
    stmts: list[Stmt],
    type_env: dict[str, Type],
) -> list[str]:
    """Reject constructs not represented by guarded tensor dependencies."""

    failures: list[str] = []
    for stmt in stmts:
        match stmt:
            case For():
                failures.append(
                    "nested loops are not supported by identity transitions"
                )
            case If():
                failures.append(
                    "source If control is not represented by the identity "
                    "transition dependency proof"
                )
            case Let():
                failures.append(
                    "Let bindings must be expanded before identity analysis"
                )
            case Assign(target=TensorView()):
                failures.append(
                    "partial tensor assignments are not supported by identity "
                    "transitions"
                )
            case Assign(target=Var(name=name)):
                if not isinstance(type_env.get(name), TensorType):
                    failures.append(
                        f"{name}: scalar assignment is not represented by "
                        "guarded tensor dependencies"
                    )
            case MaskedStore():
                failures.append(
                    "memory side effects cannot be erased by a state identity "
                    "transition"
                )
            case _:
                raise AssertionError(f"unhandled statement: {stmt!r}")
    return failures


def _fact_cut_tail(
    body: list[Stmt],
    fact: AllFalseFact,
) -> tuple[list[Stmt], int, list[str]]:
    """Return the suffix that may observe the fact and audit the cut seam."""

    failures: list[str] = []
    full_write_indices = [
        index
        for index, stmt in enumerate(body)
        if isinstance(stmt, Assign)
        and isinstance(stmt.target, Var)
        and stmt.target.name == fact.variable
    ]
    all_writes = sum(_writes_variable(stmt, fact.variable) for stmt in body)

    if not full_write_indices:
        if all_writes:
            failures.append(
                f"{fact.variable}: fact is only partially or nestedly written; "
                "a sound cut requires top-level full assignments"
            )
        # An unwritten fact is an input/cut-point value at body entry.
        return body, 0, failures

    if all_writes != len(full_write_indices):
        failures.append(
            f"{fact.variable}: fact construction contains a partial or nested "
            "write; the value-fact cut requires top-level full assignments"
        )

    first = full_write_indices[0]
    last = full_write_indices[-1]
    first_stmt = body[first]
    assert isinstance(first_stmt, Assign)
    if first_stmt.op is not None:
        failures.append(
            f"{fact.variable}: first fact write is compound and may depend on "
            "an unconstrained previous value"
        )

    for index in range(first, last + 1):
        stmt = body[index]
        if not (
            isinstance(stmt, Assign)
            and isinstance(stmt.target, Var)
            and stmt.target.name == fact.variable
        ):
            failures.append(
                f"{fact.variable}: non-fact statement at index {index} occurs "
                "between fact-construction writes"
            )

    for index, stmt in enumerate(body[:first]):
        if _reads_variable(stmt, fact.variable):
            failures.append(
                f"{fact.variable}: value is read before its first full "
                f"definition at statement {index}"
            )

    tail_start = last + 1
    return body[tail_start:], tail_start, failures


def _fact_facts_at_cut(
    body: list[Stmt],
    tail_start: int,
    fact: AllFalseFact,
    type_env: dict[str, Type],
    entry_facts: dict[str, TensorFacts],
    entry_element_bodies: dict[str, Expr],
) -> TensorFacts:
    """Compute source-derived facts immediately after fact construction."""

    fact_type = type_env.get(fact.variable)
    rank = len(fact_type.dims) if isinstance(fact_type, TensorType) else 0
    analyzer = PositionalAnalyzer(
        assumptions=NeutralityAssumptions(),
        type_env=type_env,
        env=dict(entry_facts),
        elem_body=dict(entry_element_bodies),
    )
    for stmt in body[:tail_start]:
        analyzer.exec_stmt(stmt)
    return analyzer.env.get(fact.variable, facts_unknown(rank))


def _counterfactual_assumptions(
    assumptions: NeutralityAssumptions,
    fact: AllFalseFact,
) -> NeutralityAssumptions:
    return NeutralityAssumptions(
        finite_vars=assumptions.finite_vars,
        positive_vars=assumptions.positive_vars,
        all_false_masks=assumptions.all_false_masks | {fact.variable},
    )


def check_region_identity_transition(
    loop_body: list[Stmt],
    state_demands: dict[str, GuardedRegion],
    fact: AllFalseFact,
    assumptions: NeutralityAssumptions = NeutralityAssumptions(),
    *,
    entry_facts: dict[str, TensorFacts] | None = None,
    entry_element_bodies: dict[str, Expr] | None = None,
) -> IdentityTransitionReport:
    """Derive a conditional regional identity rule for one iteration.

    The forward half proves whole-state identity in the counterfactual run
    where ``fact.variable`` is false everywhere.  The guarded backward half
    starts at each demanded state region and computes exactly which region of
    that fact can influence it.  Other inputs are held equal between the
    actual and counterfactual executions by the standard deterministic
    regional-dataflow meta-rule.

    Nested loops are rejected in this first fragment.  Treating a single
    symbolic nested body execution as the whole nested fold would otherwise
    be unsound; the ordered loop framework will provide that induction later.
    """

    type_env = collect_statement_type_env(loop_body)
    if entry_facts is None:
        entry_facts = {}
    if entry_element_bodies is None:
        entry_element_bodies = {}
    failures: list[str] = []
    if not state_demands:
        failures.append("identity transition has no demanded carried state")
    if fact.variable in state_demands:
        failures.append("the counterfactual fact cannot itself be carried state")
    fact_type = type_env.get(fact.variable)
    fact_is_boolean = isinstance(fact_type, TensorType) and isinstance(
        fact_type.elem_type, BoolType
    )
    if not fact_is_boolean:
        failures.append(
            f"{fact.variable}: all-false fact does not name a boolean value"
        )

    failures.extend(_fragment_failures(loop_body, type_env))

    for state, demand in state_demands.items():
        state_type = type_env.get(state)
        if not isinstance(state_type, TensorType):
            failures.append(f"{state}: demanded state is not a tensor")
        elif len(state_type.dims) != len(demand.region):
            failures.append(
                f"{state}: demand rank {len(demand.region)} differs from state "
                f"rank {len(state_type.dims)}"
            )

    counterfactual = _counterfactual_assumptions(assumptions, fact)
    try:
        whole = check_iteration_neutral(
            loop_body,
            accumulators=list(state_demands),
            assumptions=counterfactual,
        )
    except (AssertionError, AttributeError, KeyError, TypeError, ValueError) as error:
        whole = NeutralityReport(
            proved=False,
            failures=[],
            used_assumptions=set(),
            final_facts={},
        )
        failures.append(f"counterfactual forward analysis failed: {error}")

    # This first exact-identity fragment deliberately excludes algebraic
    # floating-point identities.  Facts that select an existing state value
    # are exact even for signed zero and NaNs; rules consuming finiteness or
    # positivity need their own backend-semantic proof before they may erase a
    # dynamic iteration.
    arithmetic_assumptions = sorted(
        assumption
        for assumption in whole.used_assumptions
        if not assumption.startswith("all_false_mask(")
    )
    if arithmetic_assumptions:
        failures.append(
            "exact identity transition consumed unsupported arithmetic "
            "assumptions: " + ", ".join(arithmetic_assumptions)
        )

    tail, tail_start, cut_failures = _fact_cut_tail(loop_body, fact)
    failures.extend(cut_failures)
    fact_facts = facts_unknown(
        len(fact_type.dims) if isinstance(fact_type, TensorType) else 0
    )
    if not failures:
        try:
            fact_facts = _fact_facts_at_cut(
                loop_body,
                tail_start,
                fact,
                type_env,
                entry_facts,
                entry_element_bodies,
            )
        except (
            AssertionError,
            AttributeError,
            KeyError,
            TypeError,
            ValueError,
        ) as error:
            failures.append(f"source fact analysis failed: {error}")
    if failures or not whole.proved:
        return IdentityTransitionReport(
            fact=fact,
            fact_facts=fact_facts,
            whole_report=whole,
            states=(),
            requirements=(),
            failures=tuple(failures),
            used_assumptions=tuple(sorted(whole.used_assumptions)),
            external_assumptions=_external_assumptions(whole.used_assumptions, fact),
        )

    synthetic = Kernel(
        name="regional_identity_tail",
        params=[Param(name, typ) for name, typ in type_env.items()],
        grid=Grid(iters=[], decls=[], body=tail),
    )
    analyzer = PositionalAnalyzer(
        assumptions=NeutralityAssumptions(),
        type_env=type_env,
    )
    for stmt in tail:
        analyzer.exec_stmt(stmt)
    trace = analyzer.trace()

    evidence: list[StateIdentityEvidence] = []
    requirements: list[RegionalFactRequirement] = []
    for state, demand in state_demands.items():
        try:
            dependencies = bound_variable_regions_masked(
                synthetic,
                state,
                demand.region,
                demand.guard,
                trace,
            )
        except (
            AssertionError,
            AttributeError,
            KeyError,
            TypeError,
            ValueError,
        ) as error:
            failures.append(f"{state}: regional dependency analysis failed: {error}")
            continue
        fact_region = dependencies.get(fact.variable)
        requirement = (
            RegionalFactRequirement(fact, state, fact_region)
            if fact_region is not None
            else None
        )
        if requirement is not None:
            requirements.append(requirement)
        final_facts = whole.final_facts[state]
        evidence.append(
            StateIdentityEvidence(
                state=state,
                demand=demand,
                final_facts=final_facts,
                dependencies=tuple(sorted(dependencies)),
                fact_requirement=requirement,
            )
        )

    return IdentityTransitionReport(
        fact=fact,
        fact_facts=fact_facts,
        whole_report=whole,
        states=tuple(evidence),
        requirements=tuple(requirements),
        failures=tuple(failures),
        used_assumptions=tuple(sorted(whole.used_assumptions)),
        external_assumptions=_external_assumptions(whole.used_assumptions, fact),
    )


def prove_fact_requirement_covered(
    requirement: RegionalFactRequirement,
    available: AvailableRegionalFact,
    premises: list[Expr],
    *,
    check_name: str,
) -> ProofCheck:
    """Prove rectangular coverage of a required fact region.

    This first composition rule accepts only an unconditional available fact
    region.  A guarded available fact needs a separate implication proof and
    therefore fails closed for now.  The requirement's own guard may be
    arbitrary: bounding its whole rectangle inside an unconditional available
    rectangle is conservative.
    """

    if requirement.fact != available.fact:
        return ProofCheck(
            check_name,
            False,
            "available and required facts describe different variables or values",
        )
    required = requirement.demand
    available_region = available.demand
    if not (
        isinstance(available_region.guard.body, BoolLit)
        and available_region.guard.body.value is True
    ):
        return ProofCheck(
            check_name,
            False,
            "guarded available facts are not supported by this coverage rule",
        )
    if len(required.region) != len(available_region.region):
        return ProofCheck(check_name, False, "fact regions have different ranks")

    env: dict[str, z3.ExprRef] = {}
    expressions = [*premises]
    for region in (required.region, available_region.region):
        for sl in region:
            expressions.extend((sl.start, sl.stop))
    for expr in expressions:
        _collect_z3_symbols(expr, env)
    try:
        encoded_premises = [expr_to_z3(expr, env) for expr in premises]
        bounds = []
        for inner, outer in zip(required.region, available_region.region):
            inner_start = expr_to_z3(inner.start, env)
            inner_stop = expr_to_z3(inner.stop, env)
            outer_start = expr_to_z3(outer.start, env)
            outer_stop = expr_to_z3(outer.stop, env)
            bounds.extend((inner_start >= outer_start, inner_stop <= outer_stop))
        return z3_prove(check_name, encoded_premises, z3.And(*bounds))
    except (AssertionError, KeyError, TypeError, z3.Z3Exception) as error:
        return ProofCheck(check_name, False, f"cannot encode fact coverage: {error}")


def prove_report_requirement_on_context(
    report: IdentityTransitionReport,
    requirement: RegionalFactRequirement,
    *,
    env: dict[str, z3.ExprRef | z3.FuncDeclRef],
    assumptions: list[z3.BoolRef | bool],
    context: z3.BoolRef | bool,
    check_name: str,
) -> ProofCheck:
    """Prove a source-derived fact on one dynamic iteration context.

    Every coordinate in the required guarded region is quantified implicitly
    by the SMT validity check.  Unlike ``AvailableRegionalFact``, the fact used
    here is tied to the positional analysis snapshot at this report's checked
    construction cut, so callers cannot supply an unrelated availability
    claim.  ``context`` is expected to describe one side of an ordered range
    difference; proving that connection remains the range composer's job.
    """

    if not report.proved:
        return ProofCheck(check_name, False, "identity transition is not proved")
    if report.external_assumptions:
        return ProofCheck(
            check_name,
            False,
            "identity transition retains external assumptions",
        )
    if requirement not in report.requirements or requirement.fact != report.fact:
        return ProofCheck(
            check_name,
            False,
            "requirement is not issued by this identity transition",
        )

    demand = requirement.demand
    fact_false = report.fact_facts.false_where
    rank = len(demand.region)
    if demand.guard.rank != rank or fact_false.rank != rank:
        return ProofCheck(
            check_name,
            False,
            "fact, guard, and demanded region have different ranks",
        )

    local_env = dict(env)
    occupied = (
        set(local_env)
        | collect_expr_vars(demand.guard.body)
        | collect_expr_vars(fact_false.body)
    )
    for interval in demand.region:
        occupied |= collect_expr_vars(interval.start) | collect_expr_vars(interval.stop)
    coordinates: list[Var] = []
    for axis in range(rank):
        name = fresh_name(f"__identity_coordinate_{axis}", occupied)
        occupied.add(name)
        coordinate = Var(name, type=IntType())
        coordinates.append(coordinate)
        local_env[name] = z3.FreshInt(name)

    substitution = {
        f"_i{axis}": coordinate for axis, coordinate in enumerate(coordinates)
    }
    try:
        guard = expr_to_z3(
            subst_free_indices(demand.guard.body, substitution),
            local_env,
        )
        false_at_coordinate = expr_to_z3(
            subst_free_indices(fact_false.body, substitution),
            local_env,
        )
        membership = []
        for coordinate, interval in zip(coordinates, demand.region):
            value = local_env[coordinate.name]
            start = expr_to_z3(
                subst_free_indices(interval.start, substitution),
                local_env,
            )
            stop = expr_to_z3(
                subst_free_indices(interval.stop, substitution),
                local_env,
            )
            membership.extend((value >= start, value < stop))
        claim = z3.Implies(
            z3.And(context, guard, *membership),
            false_at_coordinate,
        )
        return z3_prove(check_name, assumptions, claim)
    except (AssertionError, KeyError, TypeError, z3.Z3Exception) as error:
        return ProofCheck(
            check_name,
            False,
            f"cannot encode source-derived regional fact: {error}",
        )
