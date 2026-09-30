"""Goal-driven relational dataflow over the existing typed kernel IR.

This module composes the per-position forward analysis with guarded backward
dependency propagation, then discharges the resulting rectangular equality
obligations through the ordinary relational proof driver.  It deliberately
does not define another kernel IR.

The supported fragment includes straight-line control flow, stateless source
maps, same-range ordered folds, and unequal ascending ranges whose one-sided
iterations are proved to be exact identity transitions. A single symbolic
loop-body execution is used only after loop-carried facts are forgotten and
the back edge has an explicit region-stable recurrence summary.

Production batch and conditional selected-row consumers use this analysis's
canonical artifact through the shared relational verifier. Regional SMT checks
are internal proof components, not a second artifact producer. Selected-row attention uses
the same named-goal entrypoint, with no separate attention proof producer.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
from typing import Literal, TYPE_CHECKING

if TYPE_CHECKING:
    from .relational_artifact import VerifiedDataflowContract

import z3
from .names import fresh_name
from .annotation_satisfiability import SatisfiabilityStatus, check_annotation_satisfiability

from . import (
    Assign,
    BinOp,
    BoolLit,
    BoolType,
    Expr,
    FloatType,
    For,
    If,
    IntType,
    Let,
    MaskedStore,
    Region,
    Stmt,
    TensorType,
    TensorView,
    Type,
    Var,
    collect_expr_vars,
)
from .annotation_to_config import build_config_from_annotation
from .identity_transition import (
    AllFalseFact,
    IdentityTransitionReport,
    RegionalFactRequirement,
    check_region_identity_transition,
    prove_report_requirement_on_context,
)
from .positional import (
    ForwardFactTrace,
    NeutralityAssumptions,
    PositionalAnalyzer,
    pred_true,
    subst_free_indices,
)
from .preprocess import build_type_env
from .reaching_definitions import reaching_definitions
from .regions import (
    ConditionalWrite,
    GuardedRegion,
    WriteTraversalPlan,
    bound_variable_regions_masked,
    collect_write_stmts,
    get_read_tensors,
    plan_write_stmts,
)
from .relational_contract import relational_theorem_digest
from .regional_obligations import (
    EquivProofConfig,
    ScopedScalarDependency,
    collect_scoped_output_value_dependencies,
    extract_loop_iter_ranges,
    prove_region_equivalence,
    relevant_output_writes,
)
from .proof_preparation import PreparedAnnotationProof, prepare_annotation_proof
from .smt import ProofCheck, ProofResult, expr_to_z3, z3_prove, z3_satisfiable


@dataclass(frozen=True)
class AxiswiseOrdinalMap:
    """Coordinate correspondence induced by a rectangular equality clause.

    Axis ``k`` at relative offset ``d`` in the left rectangle corresponds to
    axis ``k`` at the same relative offset in the right rectangle.  Keeping
    this in the proof state prevents a pair of equal-size but permuted
    reduction domains from being described as the same demand.
    """

    rank: int


@dataclass(frozen=True)
class PairedDemand:
    """Two guarded dependency hulls with an explicit coordinate map."""

    tensor: str
    left: GuardedRegion
    right: GuardedRegion
    coordinates: AxiswiseOrdinalMap
    justification: str


@dataclass(frozen=True)
class RelevantStatementAlignment:
    """A relevant typed-IR write shared by both executions.

    The existing ``write.value`` expression tree is the ordered operation
    schema.  Referencing it directly avoids a second expression IR and keeps
    opcode, type, static attributes, and operand order tied to the exact object
    consumed by the regional proof.
    """

    ordinal: int
    write: Assign | MaskedStore
    iterators: tuple[str, ...]
    control_predicates: tuple[Expr, ...]

    @property
    def target(self) -> str:
        match self.write:
            case Assign(target=Var(name=name)):
                return name
            case Assign(target=TensorView(base=base)):
                return base.name
            case MaskedStore(base=base):
                return base.name
            case _:
                raise AssertionError(f"unhandled relevant write: {self.write!r}")


@dataclass(frozen=True)
class OneSidedIterationObligation:
    """What a range proof must establish before one iteration is erased.

    This is deliberately an obligation, not a certificate. The local
    transition analysis proves that the named states are preserved *if* the
    returned regional facts hold. ``prove_range_difference_neutrality`` must
    still prove both that the iteration occurs on only ``execution`` and that
    every requirement holds there.
    """

    iterator: str
    execution: Literal["left", "right"]
    fact: AllFalseFact
    state_values: tuple[str, ...]
    regional_fact_requirements: tuple[RegionalFactRequirement, ...]


@dataclass(frozen=True)
class ConditionalIterationIdentity:
    """Exact local state identity for either execution of one loop body.

    Both reports refer to the same typed source body but use the respective
    left/right demanded regions.  Inclusion in an alignment means only that
    the conditional implication was proved without external arithmetic
    assumptions; its regional fact premises remain open.
    """

    fact: AllFalseFact
    state_values: tuple[str, ...]
    left: IdentityTransitionReport
    right: IdentityTransitionReport

    def obligation(
        self,
        iterator: str,
        execution: Literal["left", "right"],
    ) -> OneSidedIterationObligation:
        if execution == "left":
            requirements = self.left.requirements
        elif execution == "right":
            requirements = self.right.requirements
        else:
            raise ValueError("execution must be 'left' or 'right'")
        return OneSidedIterationObligation(
            iterator=iterator,
            execution=execution,
            fact=self.fact,
            state_values=self.state_values,
            regional_fact_requirements=requirements,
        )


@dataclass(frozen=True)
class RangeDifferenceNeutrality:
    """Proof that every iteration present on only one side is an identity.

    This does not by itself prove a relational loop: iterations in the range
    intersection must still have the ordered operation/input alignment carried
    by ``LoopAlignment``.  It is the complementary evidence needed to erase
    the symmetric range difference.
    """

    iterator: str
    identity: ConditionalIterationIdentity
    left_exclusive_checks: tuple[ProofCheck, ...]
    right_exclusive_checks: tuple[ProofCheck, ...]

    @property
    def proved(self) -> bool:
        checks = self.left_exclusive_checks + self.right_exclusive_checks
        return bool(checks) and all(check.proved for check in checks)


def prove_range_difference_neutrality(
    loop: For,
    identity: ConditionalIterationIdentity,
    *,
    left_env: dict[str, z3.ExprRef | z3.FuncDeclRef],
    right_env: dict[str, z3.ExprRef | z3.FuncDeclRef],
    assumptions: list[z3.BoolRef | bool],
) -> RangeDifferenceNeutrality:
    """Discharge both one-sided range obligations with one shared ordinal.

    Source ``Range`` loops are ascending, unit-stride integer sequences.  A
    shared integer therefore pairs the intersection in source order; the two
    exclusive contexts below describe exactly the iterations that must be
    erased on one execution.  Relational assumptions mentioning this iterator
    are instantiated at the same shared ordinal before regional facts are
    proved.
    """

    iterator = loop.var.name
    expected_states = set(identity.state_values)
    if not (
        identity.left.proved
        and identity.right.proved
        and not identity.left.external_assumptions
        and not identity.right.external_assumptions
        and identity.left.fact == identity.fact
        and identity.right.fact == identity.fact
        and {state.state for state in identity.left.states} == expected_states
        and {state.state for state in identity.right.states} == expected_states
    ):
        failed = ProofCheck(
            f"iter_{iterator}_conditional_identity",
            False,
            "conditional identity reports do not prove the declared state/fact rule",
        )
        return RangeDifferenceNeutrality(
            iterator=iterator,
            identity=identity,
            left_exclusive_checks=(failed,),
            right_exclusive_checks=(),
        )

    shared = z3.FreshInt(f"{iterator}_ordered")
    left = dict(left_env)
    right = dict(right_env)
    try:
        left_iterator = left[iterator]
        right_iterator = right[iterator]
        if not isinstance(left_iterator, z3.ArithRef) or not isinstance(
            right_iterator, z3.ArithRef
        ):
            raise TypeError("loop iterators must be integer SMT values")
        left[iterator] = shared
        right[iterator] = shared
        left_start = expr_to_z3(loop.iters.start, left)
        left_stop = expr_to_z3(loop.iters.stop, left)
        right_start = expr_to_z3(loop.iters.start, right)
        right_stop = expr_to_z3(loop.iters.stop, right)
        left_in = z3.And(shared >= left_start, shared < left_stop)
        right_in = z3.And(shared >= right_start, shared < right_stop)
        left_exclusive = z3.And(left_in, z3.Not(right_in))
        right_exclusive = z3.And(right_in, z3.Not(left_in))
        substitutions = ((left_iterator, shared), (right_iterator, shared))
        instantiated_assumptions = [
            z3.substitute(assumption, *substitutions)
            if isinstance(assumption, z3.ExprRef)
            else assumption
            for assumption in assumptions
        ]
    except (AssertionError, KeyError, TypeError, z3.Z3Exception) as error:
        failed = ProofCheck(
            f"iter_{iterator}_range_difference_context",
            False,
            f"cannot encode ordered range difference: {error}",
        )
        return RangeDifferenceNeutrality(
            iterator=iterator,
            identity=identity,
            left_exclusive_checks=(failed,),
            right_exclusive_checks=(),
        )

    left_requirements = {
        requirement.state: requirement for requirement in identity.left.requirements
    }
    right_requirements = {
        requirement.state: requirement for requirement in identity.right.requirements
    }

    def demand_rectangle_nonempty(
        requirement: RegionalFactRequirement,
        env: dict[str, z3.ExprRef | z3.FuncDeclRef],
    ) -> z3.BoolRef:
        # A dynamic loop instance only participates in the relational fold for
        # a state when its backward-demand rectangle is nonempty.  Omitting the
        # guard here is conservative: it admits additional inactive instances
        # instead of silently erasing an active one.
        return z3.And(
            *(
                expr_to_z3(interval.start, env)
                < expr_to_z3(interval.stop, env)
                for interval in requirement.demand.region
            )
        )

    left_checks_list: list[ProofCheck] = []
    for requirement in identity.left.requirements:
        counterpart = right_requirements.get(requirement.state)
        if counterpart is None:
            left_checks_list.append(
                ProofCheck(
                    f"iter_{iterator}_left_exclusive_{requirement.state}_"
                    f"{identity.fact.variable}_false",
                    False,
                    "opposite execution has no matching state demand",
                )
            )
            continue
        try:
            paired_context = z3.And(
                left_exclusive,
                demand_rectangle_nonempty(counterpart, right),
            )
        except (AssertionError, KeyError, TypeError, z3.Z3Exception) as error:
            left_checks_list.append(
                ProofCheck(
                    f"iter_{iterator}_left_exclusive_{requirement.state}_"
                    f"{identity.fact.variable}_false",
                    False,
                    f"cannot encode opposite state demand: {error}",
                )
            )
            continue
        left_checks_list.append(
            prove_report_requirement_on_context(
                identity.left,
                requirement,
                env=left,
                assumptions=instantiated_assumptions,
                context=paired_context,
                check_name=(
                    f"iter_{iterator}_left_exclusive_{requirement.state}_"
                    f"{identity.fact.variable}_false"
                ),
            )
        )

    right_checks_list: list[ProofCheck] = []
    for requirement in identity.right.requirements:
        counterpart = left_requirements.get(requirement.state)
        if counterpart is None:
            right_checks_list.append(
                ProofCheck(
                    f"iter_{iterator}_right_exclusive_{requirement.state}_"
                    f"{identity.fact.variable}_false",
                    False,
                    "opposite execution has no matching state demand",
                )
            )
            continue
        try:
            paired_context = z3.And(
                right_exclusive,
                demand_rectangle_nonempty(counterpart, left),
            )
        except (AssertionError, KeyError, TypeError, z3.Z3Exception) as error:
            right_checks_list.append(
                ProofCheck(
                    f"iter_{iterator}_right_exclusive_{requirement.state}_"
                    f"{identity.fact.variable}_false",
                    False,
                    f"cannot encode opposite state demand: {error}",
                )
            )
            continue
        right_checks_list.append(
            prove_report_requirement_on_context(
                identity.right,
                requirement,
                env=right,
                assumptions=instantiated_assumptions,
                context=paired_context,
                check_name=(
                    f"iter_{iterator}_right_exclusive_{requirement.state}_"
                    f"{identity.fact.variable}_false"
                ),
            )
        )

    left_checks = tuple(left_checks_list)
    right_checks = tuple(right_checks_list)
    if not left_checks and not right_checks:
        failed = ProofCheck(
            f"iter_{iterator}_range_difference_fact_requirements",
            False,
            "conditional identity has no regional fact requirements",
        )
        left_checks = (failed,)
    return RangeDifferenceNeutrality(
        iterator=iterator,
        identity=identity,
        left_exclusive_checks=left_checks,
        right_exclusive_checks=right_checks,
    )


@dataclass(frozen=True)
class LoopAlignment:
    """Proved correspondence for one grid or sequential iteration sequence."""

    iterator: str
    scope: str
    strategy: str
    proof_check: str
    carried_values: tuple[str, ...] = ()
    live_out_values: tuple[str, ...] = ()
    identity_state_values: tuple[str, ...] = ()
    conditional_identities: tuple[ConditionalIterationIdentity, ...] = ()
    range_difference_neutrality: RangeDifferenceNeutrality | None = None


@dataclass(frozen=True)
class ControlAlignment:
    """A value-relevant path predicate proved equal in the two runs."""

    predicate: Expr
    iterators: tuple[str, ...]
    proof_check: str


@dataclass(frozen=True)
class DiscreteValueAlignment:
    """Equality at one relevant integer/Boolean tensor definition.

    ``exact_element_relation`` means equality was proved from the two
    per-position expressions after mapping relative region coordinates.
    ``same_operation_congruence`` means every tensor operand was already
    aligned and the shared typed-IR expression fixes operand/reduction order.
    """

    variable: str
    statement_ordinal: int
    strategy: str
    proof_checks: tuple[str, ...]


@dataclass(frozen=True)
class OrderedAlignmentSummary:
    """Auditable evidence for ordered relevant-operation correspondence."""

    output_tensor: str
    paired_demands: tuple[PairedDemand, ...]
    relevant_statements: tuple[RelevantStatementAlignment, ...]
    loop_alignments: tuple[LoopAlignment, ...]
    control_alignments: tuple[ControlAlignment, ...]
    discrete_value_alignments: tuple[DiscreteValueAlignment, ...] = ()
    theorem_vacuous: bool = False


@dataclass(frozen=True)
class RelationalDataflowReport:
    """Proof result and explicit assumptions for one named relational goal."""

    kernel_name: str
    goal_name: str
    source_sha256: str
    constants: tuple[tuple[str, int | float | bool], ...]
    theorem_sha256: str
    checks: tuple[ProofCheck, ...]
    diagnostics: tuple[ProofCheck, ...]
    required_tensor_reads: tuple[str, ...]
    used_assumptions: tuple[str, ...]
    external_obligations: tuple[str, ...]
    alignments: tuple[OrderedAlignmentSummary, ...]
    unsupported_reason: str | None
    verified_contract: VerifiedDataflowContract | None = None
    annotation_satisfiability: SatisfiabilityStatus = "unchecked"

    @property
    def proved(self) -> bool:
        return bool(self.checks) and all(check.proved for check in self.checks)


class UnsupportedOrderedProvenance(ValueError):
    """The existing relational proof cannot be accounted as an alignment."""


def _statement_schema(
    ordinal: int,
    write,
) -> RelevantStatementAlignment:
    statement = write.write
    if not isinstance(statement, (Assign, MaskedStore)):
        raise AssertionError(f"unhandled relevant write: {statement!r}")
    return RelevantStatementAlignment(
        ordinal=ordinal,
        write=statement,
        iterators=tuple(sorted(write.iterators)),
        control_predicates=tuple(write.conditions),
    )


def _find_check(result: ProofResult, name: str) -> ProofCheck | None:
    return next((check for check in result.checks if check.name == name), None)


def _proved_iterator_strategy(
    result: ProofResult,
    iterator: str,
) -> tuple[str, str]:
    modes = (
        ("same_range", "same_range"),
        ("effective_range_equal", "filtered_relevant_range"),
        ("singleton", "singleton_relevant"),
        ("ordered_shared_ordinal", "ordered_range_difference"),
    )
    proved = [
        (f"iter_{iterator}_{check_name}", strategy)
        for check_name, strategy in modes
        if (
            (check := _find_check(result, f"iter_{iterator}_{check_name}")) is not None
            and check.proved
        )
    ]
    if len(proved) != 1:
        raise UnsupportedOrderedProvenance(
            f"iterator {iterator!r} has {len(proved)} proved alignment modes"
        )
    return proved[0]


def _source_loops(statements: list[Stmt]) -> list[For]:
    loops: list[For] = []
    for statement in statements:
        match statement:
            case For(body=body):
                loops.append(statement)
                loops.extend(_source_loops(list(body)))
            case If(then_body=then_body, else_body=else_body):
                loops.extend(_source_loops(list(then_body)))
                loops.extend(_source_loops(list(else_body)))
            case Assign() | Let() | MaskedStore():
                continue
            case _:
                raise AssertionError(f"unhandled statement: {statement!r}")
    return loops


def _region_vars(region: Region) -> set[str]:
    return {
        name
        for sl in region
        for bound in (sl.start, sl.stop)
        for name in collect_expr_vars(bound)
    }


def _write_reads(write: Assign | MaskedStore) -> set[str]:
    reads = collect_expr_vars(write.value)
    match write:
        case Assign(target=Var(name=name), op=op):
            if op is not None:
                reads.add(name)
        case Assign(target=TensorView(base=base, region=region)):
            # A partial update preserves the rest of the old tensor.
            reads.add(base.name)
            reads |= _region_vars(region)
        case MaskedStore(region=region, mask=mask):
            reads |= _region_vars(region) | _region_vars(mask)
        case _:
            raise AssertionError(f"unhandled write: {write!r}")
    return reads


def _direct_assigned_values(statements: list[Stmt]) -> set[str]:
    assigned: set[str] = set()
    for statement in statements:
        match statement:
            case Assign(target=Var(name=name)):
                assigned.add(name)
            case Assign(target=TensorView(base=base)):
                assigned.add(base.name)
            case MaskedStore(base=base):
                assigned.add(base.name)
            case If(then_body=then_body, else_body=else_body):
                assigned |= _direct_assigned_values(list(then_body))
                assigned |= _direct_assigned_values(list(else_body))
            case For() | Let():
                continue
            case _:
                raise AssertionError(f"unhandled statement: {statement!r}")
    return assigned


def _all_statement_reads(statements: list[Stmt]) -> set[str]:
    reads: set[str] = set()
    for statement in statements:
        match statement:
            case Assign() | MaskedStore():
                reads |= _write_reads(statement)
            case Let(value=value):
                reads |= collect_expr_vars(value)
            case If(cond=cond, then_body=then_body, else_body=else_body):
                reads |= collect_expr_vars(cond)
                reads |= _all_statement_reads(list(then_body))
                reads |= _all_statement_reads(list(else_body))
            case For(iters=iters, body=body):
                reads |= collect_expr_vars(iters.start)
                reads |= collect_expr_vars(iters.stop)
                reads |= _all_statement_reads(list(body))
            case _:
                raise AssertionError(f"unhandled statement: {statement!r}")
    return reads


def _loop_carried_values(loop: For) -> frozenset[str]:
    """Values read before a definite same-iteration definition."""

    assigned = _direct_assigned_values(list(loop.body))

    def scan(
        statements: list[Stmt],
        definitely_defined: set[str],
    ) -> tuple[set[str], set[str]]:
        carried: set[str] = set()
        defined = set(definitely_defined)
        for statement in statements:
            match statement:
                case Assign(target=target) as write:
                    carried |= (_write_reads(write) & assigned) - defined
                    if isinstance(target, Var):
                        defined.add(target.name)
                    # A TensorView assignment does not definitely initialize
                    # the complete tensor for the next read.
                case MaskedStore() as write:
                    carried |= (_write_reads(write) & assigned) - defined
                case Let(var=var, value=value):
                    carried |= (collect_expr_vars(value) & assigned) - defined
                    defined.add(var.name)
                case If(cond=cond, then_body=then_body, else_body=else_body):
                    carried |= (collect_expr_vars(cond) & assigned) - defined
                    then_carried, then_defined = scan(list(then_body), defined)
                    else_carried, else_defined = scan(list(else_body), defined)
                    carried |= then_carried | else_carried
                    defined = then_defined & else_defined
                case For(iters=iters, body=body):
                    nested_reads = (
                        collect_expr_vars(iters.start)
                        | collect_expr_vars(iters.stop)
                        | _all_statement_reads(list(body))
                    )
                    carried |= (nested_reads & assigned) - defined
                    # A nested range may be empty, so none of its definitions
                    # are definite after the loop.
                case _:
                    raise AssertionError(f"unhandled statement: {statement!r}")
        return carried, defined

    carried, _ = scan(list(loop.body), set())
    return frozenset(carried)


def _kernel_loop_carried(
    prepared: PreparedAnnotationProof,
) -> dict[int, frozenset[str]]:
    return {
        id(loop): _loop_carried_values(loop)
        for loop in _source_loops(list(prepared.kernel.grid.body))
    }


def _loop_live_out_values(
    loop: For,
    iterator: str,
    all_writes: list[ConditionalWrite],
) -> frozenset[str]:
    """Relevant loop definitions whose values are observed after the loop."""

    body_positions = [
        index
        for index, conditional in enumerate(all_writes)
        if iterator in conditional.iterators
    ]
    if not body_positions:
        return frozenset()
    last_body_position = max(body_positions)
    later_reads = {
        variable
        for conditional in all_writes[last_body_position + 1 :]
        if iterator not in conditional.iterators
        for variable in (
            _write_reads(conditional.write)
            | {
                name
                for condition in conditional.conditions
                for name in collect_expr_vars(condition)
            }
        )
    }
    return frozenset(_direct_assigned_values(list(loop.body)) & later_reads)


def _loop_effect_values(
    iterator: str,
    all_writes: list[ConditionalWrite],
) -> frozenset[str]:
    """Externally visible memory writes performed by one loop iteration."""

    return frozenset(
        _statement_schema(index, conditional).target
        for index, conditional in enumerate(all_writes)
        if iterator in conditional.iterators
        and isinstance(conditional.write, MaskedStore)
    )


def _value_control_relevant_writes(
    prepared: PreparedAnnotationProof,
    output_tensor: str,
) -> list[ConditionalWrite]:
    """Conservative value/control slice including scalar definitions.

    ``relevant_output_writes`` intentionally follows tensor value flow; loop
    erasure also has to preserve loop-defined scalar predicates that control a
    later relevant write.  This local fixed point follows both value operands
    and enclosing path predicates without introducing a second statement IR.
    """

    writes = collect_write_stmts(prepared.kernel)
    live = {output_tensor}
    selected: set[int] = set()
    changed = True
    while changed:
        changed = False
        for index, conditional in enumerate(writes):
            if index in selected:
                continue
            target = _statement_schema(index, conditional).target
            if target not in live:
                continue
            selected.add(index)
            live |= _write_reads(conditional.write)
            for condition in conditional.conditions:
                live |= collect_expr_vars(condition)
            changed = True
    return [write for index, write in enumerate(writes) if index in selected]


def _identity_demands(
    states: frozenset[str],
    dependencies: dict[str, GuardedRegion],
    output_tensor: str,
    output_region: Region,
) -> dict[str, GuardedRegion] | None:
    demands: dict[str, GuardedRegion] = {}
    # These are independent state obligations, not an execution order. Freeze
    # their order before constructing transition evidence from the state set.
    for state in sorted(states):
        if state == output_tensor:
            demands[state] = GuardedRegion(
                output_region,
                pred_true(len(output_region)),
            )
        elif state in dependencies:
            demands[state] = dependencies[state]
        else:
            # A one-sided erasure rule may not silently omit observable state.
            return None
    return demands


def _conditional_iteration_identities(
    loop: For,
    states: frozenset[str],
    left_dependencies: dict[str, GuardedRegion],
    right_dependencies: dict[str, GuardedRegion],
    output_tensor: str,
    left_output_region: Region,
    right_output_region: Region,
    type_env: dict[str, Type],
    trace: ForwardFactTrace,
) -> tuple[ConditionalIterationIdentity, ...]:
    """Discover exact all-false selection identities without naming kernels."""

    if not states:
        return ()
    left_demands = _identity_demands(
        states,
        left_dependencies,
        output_tensor,
        left_output_region,
    )
    right_demands = _identity_demands(
        states,
        right_dependencies,
        output_tensor,
        right_output_region,
    )
    if left_demands is None or right_demands is None:
        return ()

    body = list(loop.body)
    if not body:
        return ()
    # Identity must hold at an arbitrary iteration, not just initialization.
    # The forward trace at body entry has already invalidated loop writes;
    # the snapshot before the For still contains first-iteration-only facts.
    entry_facts = trace.before(body[0])
    entry_element_bodies = trace.element_bodies_before(body[0])
    referenced = _all_statement_reads(body) | _direct_assigned_values(body)
    candidates: list[str] = []
    for name in sorted(referenced):
        typ = type_env.get(name)
        if (
            isinstance(typ, TensorType)
            and isinstance(typ.elem_type, BoolType)
            and name not in states
        ):
            candidates.append(name)
    identities: list[ConditionalIterationIdentity] = []
    expected_states = set(states)
    for name in candidates:
        fact = AllFalseFact(name)
        left = check_region_identity_transition(
            body,
            left_demands,
            fact,
            entry_facts=entry_facts,
            entry_element_bodies=entry_element_bodies,
        )
        right = check_region_identity_transition(
            body,
            right_demands,
            fact,
            entry_facts=entry_facts,
            entry_element_bodies=entry_element_bodies,
        )
        assumption = f"all_false_mask({name})"
        if not (
            left.proved
            and right.proved
            and not left.external_assumptions
            and not right.external_assumptions
            and assumption in left.used_assumptions
            and assumption in right.used_assumptions
            and {state.state for state in left.states} == expected_states
            and {state.state for state in right.states} == expected_states
        ):
            continue
        identities.append(
            ConditionalIterationIdentity(
                fact=fact,
                state_values=tuple(sorted(states)),
                left=left,
                right=right,
            )
        )
    return tuple(identities)


def _loop_context_issue(prepared: PreparedAnnotationProof) -> str | None:
    """Explain why the ordinary relational environment cannot bind a loop."""

    available_left = set(prepared.first_config.left_env)
    available_right = set(prepared.first_config.right_env)
    for iterator, bounds in extract_loop_iter_ranges(prepared.kernel).items():
        needed = collect_expr_vars(bounds.start) | collect_expr_vars(bounds.stop)
        missing_left = needed - available_left
        missing_right = needed - available_right
        if missing_left or missing_right:
            return (
                f"iterator {iterator!r} bounds lack relational bindings: "
                f"left={sorted(missing_left)}, right={sorted(missing_right)}"
            )
    return None


def _loop_alignments(
    prepared: PreparedAnnotationProof,
    output_tensor: str,
    left_output_region: Region,
    right_output_region: Region,
    left_dependencies: dict[str, GuardedRegion],
    right_dependencies: dict[str, GuardedRegion],
    result: ProofResult,
    carried_by_loop: dict[int, frozenset[str]],
    write_plan: WriteTraversalPlan,
    type_env: dict[str, Type],
    trace: ForwardFactTrace,
) -> tuple[LoopAlignment, ...]:
    alignments: list[LoopAlignment] = []
    for grid_iter in prepared.kernel.grid.iters:
        iterator = grid_iter.var.name
        proof_check, strategy = _proved_iterator_strategy(result, iterator)
        alignments.append(
            LoopAlignment(
                iterator=iterator,
                scope="grid",
                strategy=strategy,
                proof_check=proof_check,
            )
        )

    all_writes = relevant_output_writes(prepared.kernel, output_tensor)
    identity_writes = _value_control_relevant_writes(prepared, output_tensor)
    relevant_write_ids = {id(write.write) for write in all_writes}
    parameter_names = {parameter.name for parameter in prepared.kernel.params}
    for loop in _source_loops(list(prepared.kernel.grid.body)):
        iterator = loop.var.name
        proof_check, proved_strategy = _proved_iterator_strategy(result, iterator)
        if proved_strategy not in {"same_range", "ordered_range_difference"}:
            raise UnsupportedOrderedProvenance(
                f"source loop {iterator!r} requires an order-preserving "
                "same-range or exact one-sided-identity fold, "
                f"got {proved_strategy}"
            )
        body_positions = [
            index
            for index, conditional in enumerate(all_writes)
            if iterator in conditional.iterators
        ]
        if not body_positions:
            continue
        relevant_body_targets = {
            _statement_schema(index, all_writes[index]).target
            for index in body_positions
        }
        carried = carried_by_loop[id(loop)] & relevant_body_targets
        first_body_position = min(body_positions)
        initialized = parameter_names | {
            alignment.target
            for alignment in (
                _statement_schema(index, conditional)
                for index, conditional in enumerate(all_writes[:first_body_position])
            )
        }
        missing_initializers = carried - initialized
        if missing_initializers:
            raise UnsupportedOrderedProvenance(
                f"source loop {iterator!r} carries values without an initializer: "
                f"{sorted(missing_initializers)}"
            )

        nested_recurrences = [
            recurrence
            for recurrence in write_plan.recurrences
            if any(
                iterator in write.iterators
                and id(write.write) in relevant_write_ids
                and len(write.iterators) > 1
                for write in recurrence.writes
            )
        ]
        if nested_recurrences:
            raise UnsupportedOrderedProvenance(
                f"source loop {iterator!r} contains a nested recurrence; "
                "nested fold-state alignment is not implemented"
            )

        recurrence_values = {
            variable
            for recurrence in write_plan.recurrences
            if any(
                iterator in write.iterators and id(write.write) in relevant_write_ids
                for write in recurrence.writes
            )
            for variable in recurrence.variables
        }
        missing_recurrences = carried - recurrence_values
        if missing_recurrences:
            raise UnsupportedOrderedProvenance(
                f"source loop {iterator!r} has unaccounted back-edge values: "
                f"{sorted(missing_recurrences)}"
            )
        partial_state_writes = [
            conditional.write
            for conditional in all_writes
            if iterator in conditional.iterators
            and _statement_schema(0, conditional).target in carried
            and not (
                isinstance(conditional.write, Assign)
                and isinstance(conditional.write.target, Var)
            )
        ]
        if partial_state_writes:
            raise UnsupportedOrderedProvenance(
                f"source loop {iterator!r} carries state through a partial "
                "tensor or memory write"
            )
        identity_body_targets = {
            _statement_schema(index, conditional).target
            for index, conditional in enumerate(identity_writes)
            if iterator in conditional.iterators
        }
        identity_carried = carried_by_loop[id(loop)] & identity_body_targets
        live_out = _loop_live_out_values(loop, iterator, identity_writes)
        identity_states = (
            identity_carried | live_out | _loop_effect_values(iterator, identity_writes)
        )
        conditional_identities = _conditional_iteration_identities(
            loop,
            identity_states,
            left_dependencies,
            right_dependencies,
            output_tensor,
            left_output_region,
            right_output_region,
            type_env,
            trace,
        )
        range_difference_neutrality = None
        strategy = "same_range_fold" if carried else "same_range_map"
        if proved_strategy == "ordered_range_difference":
            range_context = _iteration_context_assumptions(
                prepared.first_config,
                result,
                exclude_iterators=frozenset({iterator}),
            )
            attempted = tuple(
                prove_range_difference_neutrality(
                    loop,
                    identity,
                    left_env=prepared.first_config.left_env,
                    right_env=prepared.first_config.right_env,
                    assumptions=[
                        *prepared.first_config.base_assumptions,
                        *range_context,
                    ],
                )
                for identity in conditional_identities
            )
            range_difference_neutrality = next(
                (neutrality for neutrality in attempted if neutrality.proved),
                None,
            )
            if range_difference_neutrality is None:
                failed_details = [
                    f"{neutrality.identity.fact.variable}: "
                    + "; ".join(
                        check.details
                        for check in (
                            neutrality.left_exclusive_checks
                            + neutrality.right_exclusive_checks
                        )
                        if not check.proved
                    )
                    for neutrality in attempted
                ]
                raise UnsupportedOrderedProvenance(
                    f"source loop {iterator!r} has unequal ranges but no "
                    "conditional identity discharges both one-sided "
                    "differences"
                    + (": " + " | ".join(failed_details) if failed_details else "")
                )
            strategy = "ordered_common_range_with_identity_difference"
        alignments.append(
            LoopAlignment(
                iterator=iterator,
                scope="source",
                strategy=strategy,
                proof_check=proof_check,
                carried_values=tuple(sorted(carried)),
                live_out_values=tuple(sorted(live_out)),
                identity_state_values=tuple(sorted(identity_states)),
                conditional_identities=conditional_identities,
                range_difference_neutrality=range_difference_neutrality,
            )
        )
    return tuple(alignments)


def _control_alignments(
    prepared: PreparedAnnotationProof,
    output_tensor: str,
    result: ProofResult,
) -> tuple[ControlAlignment, ...]:
    scoped, _ = collect_scoped_output_value_dependencies(
        prepared.kernel,
        (output_tensor,),
        defer_discrete_assignment_scalars=True,
    )
    controls: list[ControlAlignment] = []
    for relevant_write in relevant_output_writes(prepared.kernel, output_tensor):
        for condition in relevant_write.conditions:
            dependency = ScopedScalarDependency(
                expression=condition,
                iterators=relevant_write.iterators,
            )
            try:
                index = scoped.index(dependency)
            except ValueError as error:
                raise UnsupportedOrderedProvenance(
                    "relevant control predicate is absent from scalar obligations: "
                    f"{condition!r}"
                ) from error
            proof_check = f"value_scalar_equiv_{index}"
            check = _find_check(result, proof_check)
            if check is None or not check.proved:
                raise UnsupportedOrderedProvenance(
                    f"control predicate lacks a proved equality check: {proof_check}"
                )
            alignment = ControlAlignment(
                predicate=condition,
                iterators=tuple(sorted(relevant_write.iterators)),
                proof_check=proof_check,
            )
            if alignment not in controls:
                controls.append(alignment)
    return tuple(controls)


def _assigned_name(write: Assign | MaskedStore) -> str:
    return _statement_schema(0, ConditionalWrite([], write, frozenset())).target


def _is_discrete_tensor_name(name: str, type_env: dict[str, Type]) -> bool:
    typ = type_env.get(name)
    return isinstance(typ, TensorType) and isinstance(
        typ.elem_type, (IntType, BoolType)
    )


def _iteration_context_assumptions(
    config: EquivProofConfig,
    result: ProofResult,
    *,
    exclude_iterators: frozenset[str] = frozenset(),
) -> list[z3.BoolRef | bool]:
    """Reconstruct the proved per-dynamic-instance iterator context."""

    assumptions: list[z3.BoolRef | bool] = []
    ranges = extract_loop_iter_ranges(config.kernel)
    for iterator, bounds in ranges.items():
        if iterator in exclude_iterators:
            continue
        left_value = config.left_env[iterator]
        right_value = config.right_env[iterator]
        assumptions.extend(
            (
                left_value >= expr_to_z3(bounds.start, config.left_env),
                left_value < expr_to_z3(bounds.stop, config.left_env),
                right_value >= expr_to_z3(bounds.start, config.right_env),
                right_value < expr_to_z3(bounds.stop, config.right_env),
            )
        )
        paired = any(
            (
                (check := _find_check(result, f"iter_{iterator}_{suffix}"))
                is not None
                and check.proved
            )
            for suffix in (
                "same_range",
                "effective_range_equal",
                "ordered_shared_ordinal",
            )
        )
        if paired:
            assumptions.append(left_value == right_value)
        singleton = config.singleton_annotations.get(iterator)
        if singleton is not None:
            if singleton.left is not None:
                assumptions.append(
                    left_value == expr_to_z3(singleton.left, config.left_env)
                )
            if singleton.right is not None:
                assumptions.append(
                    right_value == expr_to_z3(singleton.right, config.right_env)
                )
    return assumptions


def _prove_exact_discrete_cut(
    *,
    variable: str,
    body: Expr,
    left: GuardedRegion,
    right: GuardedRegion,
    config: EquivProofConfig,
    result: ProofResult,
) -> tuple[ProofCheck, ...]:
    """Prove equality of exact per-position bodies under ordinal mapping."""

    if len(left.region) != len(right.region):
        return (
            ProofCheck(
                f"discrete_{variable}_rank",
                False,
                "left/right demanded regions have different ranks",
            ),
        )
    left_env = dict(config.left_env)
    right_env = dict(config.right_env)
    left_mapping: dict[str, Expr] = {}
    right_mapping: dict[str, Expr] = {}
    deltas: list[z3.ArithRef] = []
    left_extents: list[z3.ArithRef] = []
    right_extents: list[z3.ArithRef] = []
    try:
        for axis, (left_interval, right_interval) in enumerate(
            zip(left.region, right.region)
        ):
            delta_name = fresh_name(
                f"__relational_discrete_{variable}_{axis}", set(left_env) | set(right_env),
            )
            delta_var = Var(delta_name, type=IntType())
            delta = z3.FreshInt(delta_name)
            left_env[delta_name] = delta
            right_env[delta_name] = delta
            deltas.append(delta)
            left_mapping[f"_i{axis}"] = BinOp(
                "+", left_interval.start, delta_var
            )
            right_mapping[f"_i{axis}"] = BinOp(
                "+", right_interval.start, delta_var
            )
            left_extents.append(
                expr_to_z3(left_interval.stop, left_env)
                - expr_to_z3(left_interval.start, left_env)
            )
            right_extents.append(
                expr_to_z3(right_interval.stop, right_env)
                - expr_to_z3(right_interval.start, right_env)
            )
        left_value = expr_to_z3(
            subst_free_indices(body, left_mapping), left_env
        )
        right_value = expr_to_z3(
            subst_free_indices(body, right_mapping), right_env
        )
        assumptions = [
            *config.base_assumptions,
            *_iteration_context_assumptions(config, result),
        ]
        satisfiable_assumptions = [
            assumption
            for assumption in assumptions
            if not isinstance(assumption, z3.QuantifierRef)
        ]
        nonempty = [
            *(extent > 0 for extent in left_extents),
            *(extent > 0 for extent in right_extents),
        ]
        extent_claim = z3.And(
            *(left_extent == right_extent for left_extent, right_extent in zip(
                left_extents, right_extents
            ))
        )
        coordinate_domain = [
            clause
            for delta, extent in zip(deltas, left_extents)
            for clause in (delta >= 0, delta < extent)
        ]
    except (AssertionError, KeyError, TypeError, ValueError, z3.Z3Exception) as error:
        return (
            ProofCheck(
                f"discrete_{variable}_encoding",
                False,
                f"cannot encode exact element relation: {error}",
            ),
        )

    return (
        z3_satisfiable(
            f"discrete_{variable}_demand_context_satisfiable",
            satisfiable_assumptions + nonempty,
        ),
        z3_prove(
            f"discrete_{variable}_demand_extents_equal",
            assumptions + nonempty,
            extent_claim,
        ),
        z3_prove(
            f"discrete_{variable}_exact_element_relation",
            assumptions + nonempty + coordinate_domain,
            left_value == right_value,
            timeout=10_000,
        ),
    )


def discrete_definition_graph(kernel, writes, type_env):
    """Return definition ordinals, operand predecessors, and consumed roots.

    A negative predecessor represents an input/undefined/unmodeled version of
    a locally written value and cannot be silently justified by a later write.
    Control reads conservatively require every definition, since their source
    guard may have been evaluated before its nested writes.
    """
    definitions = {
        ordinal: conditional for ordinal, conditional in enumerate(writes)
        if _is_discrete_tensor_name(_assigned_name(conditional.write), type_env)
    }
    by_id = {id(conditional.write): ordinal for ordinal, conditional in definitions.items()}
    names = {_assigned_name(conditional.write) for conditional in definitions.values()}
    reaching = reaching_definitions(kernel) if definitions else {}

    def predecessors(conditional):
        result = set()
        for read in _write_reads(conditional.write) & names:
            result.update(by_id.get(key, -1) for key in
                          reaching[id(conditional.write)].get(read, frozenset({0})))
        return result

    dependencies = {ordinal: predecessors(conditional)
                    for ordinal, conditional in definitions.items()}
    terminal = set()
    for ordinal, conditional in enumerate(writes):
        if ordinal not in definitions:
            terminal.update(predecessors(conditional))
        for condition in conditional.conditions:
            controls = collect_expr_vars(condition) & names
            terminal.update(index for index, definition in definitions.items()
                            if _assigned_name(definition.write) in controls)
    return definitions, dependencies, terminal


def _discrete_value_alignments(
    prepared: PreparedAnnotationProof,
    output_tensor: str,
    left_dependencies: dict[str, GuardedRegion],
    right_dependencies: dict[str, GuardedRegion],
    config: EquivProofConfig,
    result: ProofResult,
    trace: ForwardFactTrace,
    type_env: dict[str, Type],
) -> tuple[tuple[DiscreteValueAlignment, ...], tuple[ProofCheck, ...]]:
    """Close the definitions actually consumed at demanded discrete cuts.

    A variable name is not a reaching definition: a later aligned overwrite
    cannot justify an earlier consumer. We prove each reaching definition over
    the variable's union demand. Congruence uses its specific operand versions,
    not an unrelated later definition with the same name.
    This reuses the existing write slice and fact trace, not a second SSA IR.
    """

    writes = relevant_output_writes(prepared.kernel, output_tensor)
    definitions, dependencies, terminal = discrete_definition_graph(prepared.kernel, writes, type_env)

    aligned: set[int] = set()
    records: list[DiscreteValueAlignment] = []
    checks: list[ProofCheck] = []
    pending = set(terminal)
    while pending:
        progress = False
        for ordinal in sorted(pending):
            if ordinal not in definitions:
                continue
            conditional = definitions[ordinal]
            variable = _assigned_name(conditional.write)
            left = left_dependencies.get(variable)
            right = right_dependencies.get(variable)
            if left is None or right is None:
                continue
            exact_body = trace.element_bodies_after(conditional.write).get(variable)
            if exact_body is not None:
                direct_checks = _prove_exact_discrete_cut(
                    variable=variable, body=exact_body, left=left, right=right,
                    config=config, result=result,
                )
                definition_checks = tuple(
                    replace(check, name=f"definition_{ordinal}:{check.name}")
                    for check in direct_checks
                )
                checks.extend(definition_checks)
                if not all(check.proved for check in definition_checks):
                    return tuple(records), tuple(checks)
                strategy = "exact_element_relation"
            else:
                if not (isinstance(conditional.write, Assign)
                        and isinstance(conditional.write.target, Var)):
                    continue  # Partial state needs a regional update proof, not plain congruence.
                tensor_reads = get_read_tensors(conditional.write.value, type_env)
                scalar_names = collect_expr_vars(conditional.write.value) - tensor_reads
                needed = dependencies[ordinal] - aligned
                if needed or scalar_names:
                    new = needed - pending
                    if new:
                        pending.update(new)
                        progress = True
                    continue
                check = ProofCheck(
                    f"definition_{ordinal}:discrete_{variable}_same_operation_congruence",
                    True,
                    "every reaching discrete operand definition is aligned; "
                    "the shared typed IR fixes operation and reduction order",
                )
                checks.append(check)
                definition_checks = (check,)
                strategy = "same_operation_congruence"
            records.append(DiscreteValueAlignment(
                variable=variable, statement_ordinal=ordinal, strategy=strategy,
                proof_checks=tuple(check.name for check in definition_checks),
            ))
            pending.remove(ordinal)
            aligned.add(ordinal)
            progress = True
            break
        if not progress:
            for ordinal in sorted(pending):
                checks.append(
                    ProofCheck(
                        f"discrete_definition_{ordinal}_alignment",
                        False,
                        "no exact element relation or closed same-operation "
                        "congruence proof",
                    )
                )
            break
    return tuple(records), tuple(checks)


def _paired_demands(
    output_tensor: str,
    left_output_region: Region,
    right_output_region: Region,
    left_dependencies: dict[str, GuardedRegion],
    right_dependencies: dict[str, GuardedRegion],
    tensor_parameters: set[str],
) -> tuple[PairedDemand, ...]:
    left_names = set(left_dependencies)
    right_names = set(right_dependencies)
    if left_names != right_names:
        raise UnsupportedOrderedProvenance(
            "left/right backward slices name different values: "
            f"left_only={sorted(left_names - right_names)}, "
            f"right_only={sorted(right_names - left_names)}"
        )

    output_rank = len(left_output_region)
    if output_rank != len(right_output_region):
        raise UnsupportedOrderedProvenance("paired output regions have different ranks")
    demands = [
        PairedDemand(
            tensor=output_tensor,
            left=GuardedRegion(left_output_region, pred_true(output_rank)),
            right=GuardedRegion(right_output_region, pred_true(output_rank)),
            coordinates=AxiswiseOrdinalMap(output_rank),
            justification="annotated_post_coordinate_order",
        )
    ]
    for name in sorted(left_names):
        if name == output_tensor:
            continue
        left = left_dependencies[name]
        right = right_dependencies[name]
        if len(left.region) != len(right.region):
            raise UnsupportedOrderedProvenance(
                f"paired demand {name!r} has different ranks"
            )
        justification = (
            f"tensor_region_equiv_{name}"
            if name in tensor_parameters
            else "same_static_operation_transfer"
        )
        demand = PairedDemand(
            tensor=name,
            left=left,
            right=right,
            coordinates=AxiswiseOrdinalMap(len(left.region)),
            justification=justification,
        )
        if demand not in demands:
            demands.append(demand)
    return tuple(demands)


def _build_alignment_summary(
    prepared: PreparedAnnotationProof,
    output_tensor: str,
    left_output_region: Region,
    right_output_region: Region,
    left_dependencies: dict[str, GuardedRegion],
    right_dependencies: dict[str, GuardedRegion],
    tensor_parameters: set[str],
    result: ProofResult,
    carried_by_loop: dict[int, frozenset[str]],
    write_plan: WriteTraversalPlan,
    trace: ForwardFactTrace,
    discrete_value_alignments: tuple[DiscreteValueAlignment, ...],
) -> OrderedAlignmentSummary:
    theorem_vacuous = any(
        check.name == "vacuous_preconditions" and check.proved
        for check in result.checks
    )
    writes = relevant_output_writes(prepared.kernel, output_tensor)
    if not writes:
        raise UnsupportedOrderedProvenance(
            f"output {output_tensor!r} has no relevant write"
        )
    statements = tuple(
        _statement_schema(ordinal, write) for ordinal, write in enumerate(writes)
    )
    if not any(statement.target == output_tensor for statement in statements):
        raise UnsupportedOrderedProvenance(
            f"ordered slice for {output_tensor!r} does not reach an output write"
        )

    loops = (
        ()
        if theorem_vacuous
        else _loop_alignments(
            prepared,
            output_tensor,
            left_output_region,
            right_output_region,
            left_dependencies,
            right_dependencies,
            result,
            carried_by_loop,
            write_plan,
            build_type_env(prepared.kernel),
            trace,
        )
    )
    controls = (
        () if theorem_vacuous else _control_alignments(prepared, output_tensor, result)
    )
    return OrderedAlignmentSummary(
        output_tensor=output_tensor,
        paired_demands=_paired_demands(
            output_tensor,
            left_output_region,
            right_output_region,
            left_dependencies,
            right_dependencies,
            tensor_parameters,
        ),
        relevant_statements=statements,
        loop_alignments=loops,
        control_alignments=controls,
        discrete_value_alignments=discrete_value_alignments,
        theorem_vacuous=theorem_vacuous,
    )


def prove_prepared_relational_dataflow(
    prepared: PreparedAnnotationProof,
) -> RelationalDataflowReport:
    """Prove one prepared goal using forward facts plus masked dependencies.

    Numeric pruning begins from unconditional forward facts and may retry one
    masked dependency with an explicit finite-value obligation. Every such
    obligation is retained in the report; the analysis does not discharge it.
    """

    goal_name = prepared.annotation.name
    report_identity = {
        "kernel_name": prepared.kernel.name,
        "goal_name": goal_name,
        "source_sha256": hashlib.sha256(
            prepared.source.encode("utf-8")
        ).hexdigest(),
        "constants": prepared.constants,
        "theorem_sha256": relational_theorem_digest(prepared.annotation),
    }
    loop_context_issue = _loop_context_issue(prepared)
    if _source_loops(list(prepared.kernel.grid.body)) and loop_context_issue:
        return RelationalDataflowReport(
            **report_identity,
            checks=(
                ProofCheck(
                    name="loop_alignment_context_unsupported",
                    proved=False,
                    details=loop_context_issue,
                ),
            ),
            diagnostics=(),
            required_tensor_reads=(),
            used_assumptions=(),
            external_obligations=(),
            alignments=(),
            unsupported_reason="loop_alignment_context",
        )

    type_env = build_type_env(prepared.kernel)
    carried_by_loop = _kernel_loop_carried(prepared)
    write_plan = plan_write_stmts(prepared.kernel, type_env)
    analyzer = PositionalAnalyzer(
        assumptions=NeutralityAssumptions(),
        type_env=type_env,
        assume_then_branches=False,
    )
    for statement in prepared.kernel.grid.body:
        analyzer.exec_stmt(statement)
    trace = analyzer.trace()

    tensor_parameters = {
        parameter.name
        for parameter in prepared.kernel.params
        if isinstance(parameter.type, TensorType)
    }
    checks: list[ProofCheck] = []
    diagnostics: list[ProofCheck] = []
    required_reads: set[str] = set()
    alignments: list[OrderedAlignmentSummary] = []
    numeric_assumptions: set[str] = set()

    constants = dict(prepared.constants)
    for post_index in range(len(prepared.annotation.post_conditions)):
        config = build_config_from_annotation(
            prepared.kernel,
            prepared.annotation,
            specialized_constants=constants,
            post_index=post_index,
        )
        rank = len(config.left_output_region)
        _, syntactic_tensor_reads = collect_scoped_output_value_dependencies(
            prepared.kernel, (config.output_tensor,)
        )

        def attempt(
            finite_vars: frozenset[str],
        ) -> tuple[
            dict[str, GuardedRegion],
            dict[str, GuardedRegion],
            set[str],
            ProofResult,
        ]:
            left = bound_variable_regions_masked(
                prepared.kernel,
                config.output_tensor,
                config.left_output_region,
                pred_true(rank),
                trace,
                finite_vars=finite_vars,
            )
            right = bound_variable_regions_masked(
                prepared.kernel,
                config.output_tensor,
                config.right_output_region,
                pred_true(rank),
                trace,
                finite_vars=finite_vars,
            )
            required = (
                (set(left) | set(right))
                & syntactic_tensor_reads
                & tensor_parameters
            )
            proof = prove_region_equivalence(
                config,
                left_dependency_regions={
                    name: guarded.region for name, guarded in left.items()
                },
                right_dependency_regions={
                    name: guarded.region for name, guarded in right.items()
                },
                left_guarded_dependency_regions=left,
                right_guarded_dependency_regions=right,
                required_tensor_reads=required,
                ordered_range_difference_iterators=frozenset(
                    loop.var.name
                    for loop in _source_loops(list(prepared.kernel.grid.body))
                ),
                defer_discrete_assignment_scalars=True,
            )
            return left, right, required, proof

        chosen_finite_vars = frozenset()
        left_dependencies, right_dependencies, post_required_reads, result = attempt(
            chosen_finite_vars
        )
        if not result.checks_ok:
            finite_candidates_set: set[str] = set()
            for conditional in relevant_output_writes(
                prepared.kernel, config.output_tensor
            ):
                value = conditional.write.value
                if not isinstance(value, BinOp) or value.op not in {"*", "@"}:
                    continue
                reaching = trace.before(conditional.write)
                left_names = get_read_tensors(value.lhs, type_env)
                right_names = get_read_tensors(value.rhs, type_env)

                def may_supply_zero(names: set[str]) -> bool:
                    return any(
                        (facts := reaching.get(name)) is not None
                        and not (
                            isinstance(facts.zero_where.body, BoolLit)
                            and facts.zero_where.body.value is False
                        )
                        for name in names
                    )

                if may_supply_zero(left_names):
                    finite_candidates_set |= right_names
                if may_supply_zero(right_names):
                    finite_candidates_set |= left_names
            finite_candidates = sorted(
                name
                for name in finite_candidates_set
                if (
                    isinstance((typ := type_env.get(name)), TensorType)
                    and isinstance(typ.elem_type, FloatType)
                    and name not in tensor_parameters
                )
            )
            for candidate in finite_candidates:
                candidate_attempt = attempt(frozenset({candidate}))
                if candidate_attempt[3].checks_ok:
                    chosen_finite_vars = frozenset({candidate})
                    (
                        left_dependencies,
                        right_dependencies,
                        post_required_reads,
                        result,
                    ) = candidate_attempt
                    break
        numeric_assumptions |= {
            f"finite({name})@masked-backward-dependency"
            for name in chosen_finite_vars
        }
        required_reads.update(post_required_reads)
        checks.extend(
            ProofCheck(
                name=f"{config.output_tensor}:{check.name}",
                proved=check.proved,
                details=check.details,
            )
            for check in result.checks
        )
        diagnostics.extend(
            ProofCheck(
                name=f"{config.output_tensor}:{diagnostic.name}",
                proved=diagnostic.proved,
                details=diagnostic.details,
            )
            for diagnostic in result.diagnostics
        )
        discrete_alignments: tuple[DiscreteValueAlignment, ...] = ()
        discrete_checks: tuple[ProofCheck, ...] = ()
        if result.checks_ok:
            discrete_alignments, discrete_checks = _discrete_value_alignments(
                prepared,
                config.output_tensor,
                left_dependencies,
                right_dependencies,
                config,
                result,
                trace,
                type_env,
            )
            checks.extend(
                ProofCheck(
                    name=f"{config.output_tensor}:{check.name}",
                    proved=check.proved,
                    details=check.details,
                )
                for check in discrete_checks
            )
        deferred_composition_proved = (
            not result.deferred_obligations
            or (
                bool(discrete_alignments)
                and bool(discrete_checks)
                and all(check.proved for check in discrete_checks)
            )
        )
        if result.deferred_obligations:
            checks.append(
                ProofCheck(
                    name=f"{config.output_tensor}:deferred_discrete_scalar_composition",
                    proved=deferred_composition_proved,
                    details=(
                        "every skipped discrete-scalar dependency reaches an "
                        "exact or same-operation aligned terminal discrete cut"
                        if deferred_composition_proved
                        else "regional proof left discrete scalar obligations "
                        "without a closed terminal cut: "
                        + ", ".join(result.deferred_obligations)
                    ),
                )
            )
        if (
            result.checks_ok
            and all(check.proved for check in discrete_checks)
            and deferred_composition_proved
        ):
            try:
                alignment = _build_alignment_summary(
                    prepared,
                    config.output_tensor,
                    config.left_output_region,
                    config.right_output_region,
                    left_dependencies,
                    right_dependencies,
                    tensor_parameters,
                    result,
                    carried_by_loop,
                    write_plan,
                    trace,
                    discrete_alignments,
                )
            except (AssertionError, UnsupportedOrderedProvenance) as error:
                checks.append(
                    ProofCheck(
                        name=f"{config.output_tensor}:ordered_provenance_alignment",
                        proved=False,
                        details=str(error),
                    )
                )
            else:
                alignments.append(alignment)
                checks.append(
                    ProofCheck(
                        name=f"{config.output_tensor}:ordered_provenance_alignment",
                        proved=True,
                        details=(
                            f"{len(alignment.relevant_statements)} relevant writes, "
                            f"{len(alignment.loop_alignments)} loop alignments, "
                            f"{len(alignment.control_alignments)} control alignments"
                        ),
                    )
                )

    report = RelationalDataflowReport(
        **report_identity,
        checks=tuple(checks),
        diagnostics=tuple(diagnostics),
        required_tensor_reads=tuple(sorted(required_reads)),
        used_assumptions=tuple(
            sorted(analyzer.used_assumptions | numeric_assumptions)
        ),
        external_obligations=tuple(sorted(numeric_assumptions)),
        alignments=tuple(alignments),
        unsupported_reason=None,
    )
    if report.proved:
        status, diagnostic = check_annotation_satisfiability(prepared)
        report = replace(report, annotation_satisfiability=status,
                         diagnostics=(*report.diagnostics, diagnostic))
    # Satisfiability qualifies artifact issuance; it does not change the
    # conditional equality theorem or any of the premises used to prove it.
    from .relational_artifact import _build_verified_dataflow_contract

    return replace(report, verified_contract=_build_verified_dataflow_contract(prepared, report))


def prove_relational_dataflow_from_annotations(
    source: str,
    kernel_name: str,
    constants: dict[str, int | float | bool],
    *,
    goal_name: str | None = None,
) -> RelationalDataflowReport:
    """Translate, select one goal, and run the relational dataflow proof."""

    prepared = prepare_annotation_proof(
        source,
        kernel_name,
        constants,
        goal_name=goal_name,
    )
    return prove_prepared_relational_dataflow(prepared)
