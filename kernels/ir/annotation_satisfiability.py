"""Existence of a typed input pair satisfying the complete declared annotation.

This is a qualification check, not part of the universal equality derivation.
Tensor values use the existing abstract scalar sorts. Numeric/backend premises
introduced by the analyzer are not interpreted here, nor is a satisfying model
a proof of physical GPU-input realizability.
"""

from typing import TYPE_CHECKING, Literal

import z3

from . import TensorType
from .annotation_lowering import GuardedRegionEquality

from .annotations import Left, RegionEquiv
from .annotation_to_config import ann_expr_to_z3, ann_bool_expr_to_z3, _eval_given_expr

if TYPE_CHECKING:
    from .proof_preparation import PreparedAnnotationProof

SatisfiabilityStatus = Literal["sat", "unsat", "unknown", "unchecked"]


def annotation_preconditions(prepared: "PreparedAnnotationProof") -> list[z3.BoolRef | bool]:
    """Encode scalar, quantified, regional, shared-value, and shape premises.

    The paired environments already encode explicit sharing and specialization.
    Regional equality includes in-bounds equal-size slices and pointwise value
    equality. Its optional `given` clause keeps the ordinary per-run semantics.
    """
    from .smt import expr_to_z3

    config = prepared.first_config
    left, right = config.left_env, config.right_env
    types = {parameter.name: parameter.type for parameter in prepared.kernel.params}
    facts = list(config.base_assumptions)
    for name, typ in types.items():
        if not isinstance(typ, TensorType):
            continue
        for dimension in typ.dims:
            a, b = expr_to_z3(dimension, left), expr_to_z3(dimension, right)
            facts.extend((a >= 0, b >= 0))
            if left[name].eq(right[name]):
                facts.append(a == b)

    def region(relation: RegionEquiv, bound: dict) -> z3.BoolRef:
        starts, stops, functions = [], [], []
        clauses = []
        for ref in (relation.left, relation.right):
            env = left if isinstance(ref.side, Left) else right
            typ = types[ref.side.name]
            if not isinstance(typ, TensorType) or len(ref.slices) != len(typ.dims):
                raise ValueError("regional precondition has inconsistent tensor rank")
            begin = [ann_expr_to_z3(s.start, left, right, bound_env=bound) for s in ref.slices]
            end = [ann_expr_to_z3(s.stop, left, right, bound_env=bound) for s in ref.slices]
            for lo, hi, dimension in zip(begin, end, typ.dims):
                clauses.extend((lo >= 0, lo <= hi, hi <= expr_to_z3(dimension, env)))
            starts.append(begin)
            stops.append(end)
            functions.append(env[ref.side.name])
        if not starts[0] or len(starts[0]) != len(starts[1]):
            raise ValueError("regional precondition has inconsistent tensor rank")
        offsets = [z3.FreshInt("annotation_cell") for _ in starts[0]]
        clauses.extend(b-a == d-c for a,b,c,d in zip(starts[0], stops[0], starts[1], stops[1]))
        active = z3.And(*[z3.And(i >= 0, i < hi-lo)
                         for i,lo,hi in zip(offsets, starts[0], stops[0])])
        clauses.append(z3.ForAll(offsets, z3.Implies(active,
            functions[0](*[lo+i for lo,i in zip(starts[0], offsets)]) ==
            functions[1](*[lo+i for lo,i in zip(starts[1], offsets)]))))
        if relation.given is not None:
            clauses.append(_eval_given_expr(relation.given, {**left, **bound}) ==
                           _eval_given_expr(relation.given, {**right, **bound}))
        return z3.And(*clauses)

    for condition in prepared.annotation.pre_conditions:
        if isinstance(condition, RegionEquiv):
            facts.append(region(condition, {}))
        elif isinstance(condition, GuardedRegionEquality):
            bound = {name: z3.FreshInt(name) for name in condition.vars}
            facts.append(z3.ForAll(list(bound.values()), z3.Implies(
                ann_bool_expr_to_z3(condition.when, left, right, bound_env=bound),
                region(condition.relation, bound))))
    return facts


def check_annotation_satisfiability(
    prepared: "PreparedAnnotationProof", *, timeout_ms: int = 1000,
    witness_bounds: tuple[int, ...] = (1, 4, 16),
):
    """Find a model, or report unrestricted unsat/unknown without conflation.

    Bounded attempts constrain only witness search, never the equality proof.
    Their unsat/unknown answers cannot establish that the original domain is
    empty. The unrestricted query remains the only source of an unsat result.
    Only status, not the chosen model/search strategy, enters artifact identity.
    """
    from .smt import ProofCheck

    if type(timeout_ms) is not int or timeout_ms <= 0 or any(
        type(bound) is not int or bound < 0 for bound in witness_bounds
    ):
        raise ValueError("invalid annotation satisfiability search budget")
    facts = annotation_preconditions(prepared)
    config = prepared.first_config
    symbols = {value.get_id(): value
        for value in [*config.left_env.values(), *config.right_env.values()]
        if isinstance(value, z3.ExprRef) and z3.is_int(value) and z3.is_const(value)
        and value.decl().kind() == z3.Z3_OP_UNINTERPRETED}
    # The small-model attempt is cheap for quantified ragged metadata. Always
    # try the unrestricted formula too if it fails, including for negative or
    # large scalar values that the initial box cannot contain.
    attempts = [witness_bounds[0], None, *witness_bounds[1:]] if witness_bounds else [None]
    for bound in attempts:
        solver = z3.Solver()
        solver.set(timeout=timeout_ms)
        solver.add(*facts)
        if bound is not None:
            solver.add(*[z3.And(value >= -bound, value <= bound) for value in symbols.values()])
        result = solver.check()
        if result == z3.sat:
            return "sat", ProofCheck("annotation_preconditions_satisfiable", True, "sat typed annotation premises")
        if result == z3.unsat and bound is None:
            return "unsat", ProofCheck("annotation_preconditions_satisfiable", False, "unsat typed annotation premises")
    return "unknown", ProofCheck("annotation_preconditions_satisfiable", False, "unknown typed annotation satisfiability")
