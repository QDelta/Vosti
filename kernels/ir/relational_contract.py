"""Canonical serialization of typed relational theorem inputs.

This module defines no proof artifact or proof-success assertion. The unified
relational analyzer owns the sole qualified artifact, including its evidence.
"""

from __future__ import annotations

import hashlib
import json
import math

from . import (
    BinOp,
    BoolLit,
    BoolType,
    Expr,
    FloatLit,
    FloatType,
    IntLit,
    IntType,
    Kernel,
    TensorType,
    Type,
    Var,
)
from .annotation_lowering import (
    GuardedRegionEquality, LoweredProofGoal, lower_proof_goal,
)

from .annotations import (
    AnnAnd,
    AnnBinOp,
    AnnComparison,
    AnnImplies,
    AnnIndex,
    AnnSlice,
    ForAllConstraint,
    FreeVar,
    IntConst,
    Left,
    RelationalProofGoal,
    RegionEquiv,
    RegionRef,
    Right,
    SingletonSpec,
)


SCHEMA_VERSION = 5


def _ir_expr_data(expr: Expr) -> dict:
    match expr:
        case Var(name=name):
            return {"kind": "var", "name": name}
        case IntLit(value=value):
            return {"kind": "int", "value": value}
        case FloatLit(value=value):
            if not math.isfinite(value):
                raise ValueError("non-finite shape expression")
            return {"kind": "float", "value": value}
        case BoolLit(value=value):
            return {"kind": "bool", "value": value}
        case BinOp(op=op, lhs=lhs, rhs=rhs):
            return {
                "kind": "binary",
                "op": op,
                "lhs": _ir_expr_data(lhs),
                "rhs": _ir_expr_data(rhs),
            }
        case _:
            raise TypeError(f"unsupported contract shape expression: {type(expr).__name__}")


def _type_data(typ: Type) -> dict:
    match typ:
        case IntType():
            return {"kind": "int"}
        case FloatType():
            return {"kind": "float"}
        case BoolType():
            return {"kind": "bool"}
        case TensorType(elem_type=elem_type, dims=dims):
            return {
                "kind": "tensor",
                "element": _type_data(elem_type),
                "shape": [_ir_expr_data(dim) for dim in dims],
            }
        case _:
            raise TypeError(f"unsupported contract parameter type: {type(typ).__name__}")


def _ann_expr_data(expr) -> dict:
    match expr:
        case Left(name=name):
            return {"kind": "side", "side": "left", "name": name}
        case Right(name=name):
            return {"kind": "side", "side": "right", "name": name}
        case FreeVar(name=name):
            return {"kind": "free", "name": name}
        case IntConst(value=value):
            return {"kind": "int", "value": value}
        case AnnBinOp(op=op, lhs=lhs, rhs=rhs):
            return {
                "kind": "binary",
                "op": op,
                "lhs": _ann_expr_data(lhs),
                "rhs": _ann_expr_data(rhs),
            }
        case AnnIndex(base=base, indices=indices):
            return {
                "kind": "index",
                "base": _ann_expr_data(base),
                "indices": [_ann_expr_data(index) for index in indices],
            }
        case _:
            raise TypeError(f"unsupported annotation expression: {type(expr).__name__}")


def _slice_data(value: AnnSlice) -> dict:
    return {
        "start": _ann_expr_data(value.start),
        "stop": _ann_expr_data(value.stop),
    }


def _region_data(value: RegionRef) -> dict:
    return {
        "tensor": value.side.name,
        "side": "left" if isinstance(value.side, Left) else "right",
        "slices": [_slice_data(item) for item in value.slices],
    }


def _bool_data(value) -> dict:
    match value:
        case AnnComparison(op=op, lhs=lhs, rhs=rhs):
            return {
                "kind": "comparison",
                "op": op,
                "lhs": _ann_expr_data(lhs),
                "rhs": _ann_expr_data(rhs),
            }
        case AnnAnd(args=args):
            return {"kind": "and", "args": [_bool_data(arg) for arg in args]}
        case AnnImplies(antecedent=antecedent, consequent=consequent):
            return {
                "kind": "implies",
                "antecedent": _bool_data(antecedent),
                "consequent": _bool_data(consequent),
            }
        case _:
            raise TypeError(f"unsupported annotation boolean: {type(value).__name__}")


def _condition_data(value) -> dict:
    match value:
        case AnnComparison(op=op, lhs=lhs, rhs=rhs):
            return {
                "kind": "scalar",
                "op": op,
                "lhs": _ann_expr_data(lhs),
                "rhs": _ann_expr_data(rhs),
            }
        case RegionEquiv(left=left, right=right, given=given):
            return {
                "kind": "region_equality",
                "left": _region_data(left),
                "right": _region_data(right),
                "given": None if given is None else _ann_expr_data(given),
            }
        case ForAllConstraint(vars=vars, body=body):
            return {
                "kind": "forall",
                "variables": list(vars),
                "body": _bool_data(body),
            }
        case GuardedRegionEquality(vars=vars, when=when, relation=relation):
            return {
                "kind": "forall_region",
                "variables": list(vars),
                "when": _bool_data(when),
                "relation": _condition_data(relation),
            }
        case _:
            raise TypeError(f"unsupported annotation condition: {type(value).__name__}")


def _singleton_data(value: SingletonSpec) -> dict:
    return {
        "variable": value.var,
        "left": _ann_expr_data(value.left),
        "right": _ann_expr_data(value.right),
    }


def relational_theorem_data(annotation: RelationalProofGoal | LoweredProofGoal) -> dict:
    """Serialize the theorem in the common proof fragment's normal form.

    Conjunction flattening and guarded universal schemas preserve the source
    proposition's meaning. The artifact also binds the exact source digest.
    Proof, satisfiability checking, and export consume this same lowering; an
    unsupported proposition cannot be silently erased by serialization.
    """

    annotation = lower_proof_goal(annotation)
    return {
        "same": sorted(annotation.same_vars),
        "pre": [_condition_data(item) for item in annotation.pre_conditions],
        "post": [_condition_data(item) for item in annotation.post_conditions],
        "singletons": [_singleton_data(item) for item in annotation.singletons],
    }


def relational_theorem_digest(annotation: RelationalProofGoal) -> str:
    """Hash the exact canonical relational theorem consumed by a proof pass."""

    canonical = json.dumps(
        relational_theorem_data(annotation),
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def relational_tensor_surface(annotation: RelationalProofGoal) -> dict:
    """Return an order-independent inventory of regional tensor relations.

    The surface deliberately records only source tensor identities and sides;
    it does not assign operation roles such as query, key, or value.  Duplicate
    pairs remain duplicate entries so an importer can fail closed if the proof
    goal adds a second, unaccounted-for relation for the same tensors.
    """

    annotation = lower_proof_goal(annotation)

    def endpoint(condition: RegionEquiv) -> dict[str, str]:
        return {
            "left": condition.left.side.name,
            "right": condition.right.side.name,
        }

    def ordered(items: list[dict[str, str]]) -> list[dict[str, str]]:
        return sorted(items, key=lambda item: (item["left"], item["right"]))

    return {
        "pre": ordered([
            endpoint(
                condition.relation
                if isinstance(condition, GuardedRegionEquality)
                else condition
            )
            for condition in annotation.pre_conditions
            if isinstance(condition, (RegionEquiv, GuardedRegionEquality))
        ]),
        "post": ordered([endpoint(condition) for condition in annotation.post_conditions]),
    }


def _constant_data(value: int | float | bool) -> dict:
    if isinstance(value, bool):
        return {"kind": "bool", "value": value}
    if isinstance(value, int):
        return {"kind": "int", "value": value}
    if isinstance(value, float) and math.isfinite(value):
        return {"kind": "float", "value": value}
    raise TypeError(f"unsupported or non-finite specialized constant: {value!r}")


def relational_contract_data(
    *,
    source: str,
    kernel: Kernel,
    declared_parameters: tuple,
    annotation: RelationalProofGoal,
    constants: dict[str, int | float | bool],
    proof_kind: str,
) -> dict:
    """Serialize typed theorem inputs, without asserting that a proof passed."""

    specialized_parameters = {parameter.name: parameter for parameter in kernel.params}
    declared_by_name = {parameter.name: parameter for parameter in declared_parameters}
    if len(specialized_parameters) != len(kernel.params):
        raise ValueError("duplicate specialized kernel parameter")
    if len(declared_by_name) != len(declared_parameters):
        raise ValueError("duplicate declared kernel parameter")
    missing_declared = specialized_parameters.keys() - declared_by_name.keys()
    if missing_declared:
        raise ValueError(
            f"specialized kernel parameters lack declarations: {sorted(missing_declared)}"
        )

    document = {
        "schema_version": SCHEMA_VERSION,
        "proof_kind": proof_kind,
        "goal_name": annotation.name,
        "kernel": kernel.name,
        "source_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
        "constants": {
            name: _constant_data(value) for name, value in sorted(constants.items())
        },
        "parameters": [
            {
                "name": parameter.name,
                "declared_type": _type_data(declared_by_name[parameter.name].type),
                "type": _type_data(parameter.type),
            }
            for parameter in kernel.params
        ],
        "theorem": relational_theorem_data(annotation),
    }
    return document
