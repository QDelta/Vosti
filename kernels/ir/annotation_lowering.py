"""Lower typed propositions into the dependency verifier's supported fragment.

The public AST describes logic; these internal schemas describe proof-search
inputs. Lowering is equivalence-preserving, never a weakening of a premise or
postcondition. Unsupported connectives/polarities fail before proof or export.
"""

from dataclasses import dataclass, field, fields, is_dataclass
from typing import Union

from .annotations import (
    AnnAnd, AnnComparison, AnnImplies, ForAllConstraint, IntConst,
    Prop, RegionEquiv, RelationalProofGoal, SingletonSpec, Before, After, DTypeOf,
)


class UnsupportedProposition(ValueError):
    """Well-formed annotation outside the current proof-search fragment."""


# Only this scalar fragment is currently translated to SMT.
AnnBoolExpr = Union[AnnComparison, AnnAnd, AnnImplies]


@dataclass(frozen=True)
class GuardedRegionEquality:
    """Internal schema: forall(vars, implies(when, relation))."""

    vars: list[str]
    when: AnnBoolExpr
    relation: RegionEquiv


Condition = Union[AnnComparison, RegionEquiv, GuardedRegionEquality, ForAllConstraint]


@dataclass
class LoweredProofGoal:
    name: str
    pre_conditions: list[Condition]
    post_conditions: list[RegionEquiv]
    singletons: list[SingletonSpec] = field(default_factory=list)
    same_vars: set[str] = field(default_factory=set)


def _scalar(prop: Prop) -> bool:
    match prop:
        case AnnComparison(op=op):
            if op not in {"==", "<", "<=", ">", ">="}:
                raise UnsupportedProposition(f"Unsupported scalar comparison {op!r}; equality uses '=='")
            return True
        case AnnAnd(args=args):
            return bool(args) and all(_scalar(arg) for arg in args)
        case AnnImplies(antecedent=a, consequent=b):
            return _scalar(a) and _scalar(b)
        case _:
            return False


def lower_proof_goal(goal: RelationalProofGoal | LoweredProofGoal) -> LoweredProofGoal:
    """Check the proof fragment and lower positive universal input relations.

    Conjunctions flatten at pre/post level. A forall may contain scalar formulas,
    regional equalities, conjunctions, and scalar-guarded implications. Nested
    quantifiers, disjunction/negation, regional antecedents, unquantified guarded
    relations, and non-regional postconditions are not supported yet.
    """
    def reject_temporal(value):
        if isinstance(value, (Before, After, DTypeOf)):
            raise UnsupportedProposition("temporal states and dtype propositions require the exact-effect verifier")
        if isinstance(value, (list, tuple)):
            for item in value:
                reject_temporal(item)
        elif is_dataclass(value):
            for item in fields(value):
                reject_temporal(getattr(value, item.name))
    reject_temporal(goal)
    if isinstance(goal, LoweredProofGoal):
        return goal
    pre: list[Condition] = []
    post: list[RegionEquiv] = []

    def fail(location: str, prop: Prop, reason: str = "") -> None:
        raise UnsupportedProposition(
            f"{goal.name} {location}: unsupported proposition {type(prop).__name__}"
            + (f" ({reason})" if reason else "")
        )

    def inputs(prop: Prop, bound: tuple[str, ...] = (), guard: Prop | None = None) -> None:
        # Keep scalar quantifier bodies intact, rather than changing their SMT
        # instantiation patterns by unnecessarily splitting a conjunction.
        if bound and _scalar(prop):
            pre.append(ForAllConstraint(list(bound),
                prop if guard is None else AnnImplies(guard, prop)))
            return
        match prop:
            case AnnAnd(args=args) if args:
                for arg in args:
                    inputs(arg, bound, guard)
            case ForAllConstraint(vars=variables, body=body):
                if bound:
                    fail("pre", prop, "nested quantifiers are not supported")
                if not variables or len(set(variables)) != len(variables):
                    fail("pre", prop, "forall requires unique bound variables")
                if set(variables) & goal.same_vars:
                    fail("pre", prop, "bound variables must not shadow same parameters")
                inputs(body, tuple(variables), guard)
            case AnnImplies(antecedent=a, consequent=b):
                if not bound or not _scalar(a):
                    fail("pre", prop, "requires a scalar guard under forall")
                inputs(b, bound, a if guard is None else AnnAnd([guard, a]))
            case AnnComparison():
                if not _scalar(prop):
                    fail("pre", prop)
                pre.append(prop)
            case RegionEquiv():
                if bound:
                    when = guard if guard is not None else AnnComparison("==", IntConst(0), IntConst(0))
                    pre.append(GuardedRegionEquality(list(bound), when, prop))
                else:
                    pre.append(prop)
            case _:
                fail("pre", prop)

    def outputs(prop: Prop) -> None:
        match prop:
            case AnnAnd(args=args) if args:
                for arg in args:
                    outputs(arg)
            case RegionEquiv():
                post.append(prop)
            case _:
                fail("post", prop, "only conjunctions of output region equalities are supported")

    for prop in goal.pre_conditions:
        inputs(prop)
    for prop in goal.post_conditions:
        outputs(prop)
    return LoweredProofGoal(goal.name, pre, post, list(goal.singletons), set(goal.same_vars))
