"""Strict schema validation shared by relational-contract normalizers.

Unified contracts are proof-gated producer artifacts. Semantic
normalizers still validate their complete serialized form before assigning a
stronger meaning to it: malformed, non-canonical, or newly extended syntax is
rejected until this trusted module is updated deliberately.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .relational_artifact import VerifiedDataflowContract


RAW_FIELDS = {
    "schema_version",
    "proof_kind",
    "goal_name",
    "kernel",
    "source_sha256",
    "constants",
    "parameters",
    "theorem",
}
THEOREM_FIELDS = {
    "same",
    "pre",
    "post",
    "singletons",
}
PARAMETER_FIELDS = {"name", "declared_type", "type"}
TENSOR_TYPE_FIELDS = {"kind", "element", "shape"}
SCALAR_KINDS = {"int", "float", "bool"}
SHAPE_BINARY_OPS = {"+", "-", "*", "//", "%", "cdiv", "min", "max"}
COMPARISON_OPS = {"==", ">", ">=", "<", "<="}
_SHA256_RE = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True)
class ValidatedRelationalContract:
    raw: dict
    parameters: dict[str, dict]
    constants: dict[str, dict]
    theorem: dict
    used_assumptions: tuple[str, ...] = ()
    external_obligations: tuple[str, ...] = ()


def expect(condition: bool, description: str, *, family: str) -> None:
    if not condition:
        raise ValueError(f"not a complete {family} contract: {description}")


def int_expr(value: int) -> dict:
    return {"kind": "int", "value": value}


def side_expr(side: str, name: str) -> dict:
    return {"kind": "side", "side": side, "name": name}


def free_expr(name: str) -> dict:
    return {"kind": "free", "name": name}


def add_one(value: dict) -> dict:
    return {"kind": "binary", "op": "+", "lhs": value, "rhs": int_expr(1)}


def scalar_condition(op: str, lhs: dict, rhs: dict) -> dict:
    return {"kind": "scalar", "op": op, "lhs": lhs, "rhs": rhs}


def index_expr(side: str, tensor: str, indices: list[dict]) -> dict:
    return {
        "kind": "index",
        "base": side_expr(side, tensor),
        "indices": indices,
    }


def _name(value, description: str, family: str) -> str:
    expect(isinstance(value, str) and bool(value), description, family=family)
    return value


def shape_expression(value: dict, description: str, *, family: str) -> dict:
    expect(isinstance(value, dict), f"{description} is not an expression", family=family)
    kind = value.get("kind")
    if kind == "var":
        expect(set(value) == {"kind", "name"}, f"{description} has extra fields", family=family)
        _name(value.get("name"), f"{description} has a malformed variable", family)
    elif kind == "int":
        expect(
            set(value) == {"kind", "value"}
            and isinstance(value.get("value"), int)
            and not isinstance(value.get("value"), bool),
            f"{description} has a malformed integer",
            family=family,
        )
    elif kind == "binary":
        expect(
            set(value) == {"kind", "op", "lhs", "rhs"}
            and value.get("op") in SHAPE_BINARY_OPS,
            f"{description} has a malformed binary expression",
            family=family,
        )
        shape_expression(value["lhs"], description, family=family)
        shape_expression(value["rhs"], description, family=family)
    else:
        raise ValueError(
            f"not a complete {family} contract: {description} has unsupported kind {kind!r}"
        )
    return value


def parameter_type(parameter: dict, field: str, *, family: str) -> dict:
    typ = parameter.get(field)
    expect(isinstance(typ, dict), f"parameter {parameter.get('name')!r} has no {field}", family=family)
    kind = typ.get("kind")
    if kind == "tensor":
        expect(set(typ) == TENSOR_TYPE_FIELDS, f"malformed tensor {field}", family=family)
        element = typ.get("element")
        expect(
            isinstance(element, dict)
            and set(element) == {"kind"}
            and element.get("kind") in SCALAR_KINDS,
            f"tensor {parameter.get('name')!r} has malformed element type",
            family=family,
        )
        shape = typ.get("shape")
        expect(
            isinstance(shape, list) and bool(shape),
            f"tensor {parameter.get('name')!r} has no axes",
            family=family,
        )
        for index, dimension in enumerate(shape):
            shape_expression(
                dimension,
                f"tensor {parameter.get('name')!r} axis {index}",
                family=family,
            )
    else:
        expect(
            set(typ) == {"kind"} and kind in SCALAR_KINDS,
            f"parameter {parameter.get('name')!r} has unsupported scalar type",
            family=family,
        )
    return typ


def specialize_shape_expression(value: dict, constants: dict[str, dict]) -> dict:
    kind = value["kind"]
    if kind == "var":
        return constants.get(value["name"], value)
    if kind == "int":
        return value
    return {
        "kind": "binary",
        "op": value["op"],
        "lhs": specialize_shape_expression(value["lhs"], constants),
        "rhs": specialize_shape_expression(value["rhs"], constants),
    }


def specialize_type(typ: dict, constants: dict[str, dict]) -> dict:
    if typ["kind"] != "tensor":
        return typ
    return {
        "kind": "tensor",
        "element": typ["element"],
        "shape": [
            specialize_shape_expression(dimension, constants)
            for dimension in typ["shape"]
        ],
    }


def _validate_constant(name: str, value: dict, family: str) -> None:
    _name(name, "malformed specialized constant name", family)
    expect(
        isinstance(value, dict) and set(value) == {"kind", "value"},
        f"specialized constant {name!r} is malformed",
        family=family,
    )
    kind = value.get("kind")
    raw_value = value.get("value")
    valid = (
        (kind == "bool" and isinstance(raw_value, bool))
        or (kind == "int" and isinstance(raw_value, int) and not isinstance(raw_value, bool))
        or (kind == "float" and isinstance(raw_value, float) and math.isfinite(raw_value))
    )
    expect(valid, f"specialized constant {name!r} has invalid value", family=family)


def annotation_expression(value: dict, description: str, *, family: str) -> dict:
    expect(isinstance(value, dict), f"{description} is not an expression", family=family)
    kind = value.get("kind")
    if kind == "side":
        expect(
            set(value) == {"kind", "side", "name"}
            and value.get("side") in {"left", "right"},
            f"{description} has malformed side reference",
            family=family,
        )
        _name(value.get("name"), f"{description} has malformed side name", family)
    elif kind == "free":
        expect(set(value) == {"kind", "name"}, f"{description} has malformed free variable", family=family)
        _name(value.get("name"), f"{description} has malformed free name", family)
    elif kind == "int":
        expect(
            set(value) == {"kind", "value"}
            and isinstance(value.get("value"), int)
            and not isinstance(value.get("value"), bool),
            f"{description} has malformed integer",
            family=family,
        )
    elif kind == "binary":
        expect(
            set(value) == {"kind", "op", "lhs", "rhs"}
            and value.get("op") in SHAPE_BINARY_OPS,
            f"{description} has malformed binary expression",
            family=family,
        )
        annotation_expression(value["lhs"], description, family=family)
        annotation_expression(value["rhs"], description, family=family)
    elif kind == "index":
        expect(set(value) == {"kind", "base", "indices"}, f"{description} has malformed index", family=family)
        base = value.get("base")
        expect(
            isinstance(base, dict)
            and set(base) == {"kind", "side", "name"}
            and base.get("kind") == "side"
            and base.get("side") in {"left", "right"},
            f"{description} has malformed index base",
            family=family,
        )
        _name(base.get("name"), f"{description} has malformed index tensor", family)
        indices = value.get("indices")
        expect(isinstance(indices, list) and bool(indices), f"{description} has no indices", family=family)
        for index in indices:
            annotation_expression(index, description, family=family)
    else:
        raise ValueError(
            f"not a complete {family} contract: {description} has unsupported kind {kind!r}"
        )
    return value


def _boolean_expression(value: dict, description: str, family: str) -> None:
    expect(isinstance(value, dict), f"{description} is not boolean", family=family)
    kind = value.get("kind")
    if kind == "comparison":
        expect(
            set(value) == {"kind", "op", "lhs", "rhs"}
            and value.get("op") in COMPARISON_OPS,
            f"{description} has malformed comparison",
            family=family,
        )
        annotation_expression(value["lhs"], description, family=family)
        annotation_expression(value["rhs"], description, family=family)
    elif kind == "and":
        expect(set(value) == {"kind", "args"}, f"{description} has malformed conjunction", family=family)
        args = value.get("args")
        expect(isinstance(args, list) and bool(args), f"{description} has empty conjunction", family=family)
        for arg in args:
            _boolean_expression(arg, description, family)
    elif kind == "implies":
        expect(
            set(value) == {"kind", "antecedent", "consequent"},
            f"{description} has malformed implication",
            family=family,
        )
        _boolean_expression(value["antecedent"], description, family)
        _boolean_expression(value["consequent"], description, family)
    else:
        raise ValueError(
            f"not a complete {family} contract: {description} has unsupported boolean kind {kind!r}"
        )


def _region(value: dict, description: str, side: str, family: str) -> None:
    expect(
        isinstance(value, dict)
        and set(value) == {"tensor", "side", "slices"}
        and value.get("side") == side,
        f"{description} has malformed {side} region",
        family=family,
    )
    _name(value.get("tensor"), f"{description} has malformed tensor", family)
    slices = value.get("slices")
    expect(isinstance(slices, list) and bool(slices), f"{description} has no slices", family=family)
    for item in slices:
        expect(isinstance(item, dict) and set(item) == {"start", "stop"}, f"{description} has malformed slice", family=family)
        annotation_expression(item["start"], description, family=family)
        annotation_expression(item["stop"], description, family=family)


def condition(value: dict, description: str, *, family: str) -> dict:
    expect(isinstance(value, dict), f"{description} is not a condition", family=family)
    kind = value.get("kind")
    if kind == "scalar":
        expect(
            set(value) == {"kind", "op", "lhs", "rhs"}
            and value.get("op") in COMPARISON_OPS,
            f"{description} has malformed scalar constraint",
            family=family,
        )
        annotation_expression(value["lhs"], description, family=family)
        annotation_expression(value["rhs"], description, family=family)
    elif kind == "region_equality":
        expect(
            set(value) == {"kind", "left", "right", "given"},
            f"{description} has malformed region equality",
            family=family,
        )
        _region(value["left"], description, "left", family)
        _region(value["right"], description, "right", family)
        expect(
            value["left"]["tensor"] == value["right"]["tensor"],
            f"{description} compares different tensor parameters",
            family=family,
        )
        if value["given"] is not None:
            annotation_expression(value["given"], description, family=family)
    elif kind == "forall":
        expect(
            set(value) == {"kind", "variables", "body"},
            f"{description} has malformed universal constraint",
            family=family,
        )
        variables = value.get("variables")
        expect(
            isinstance(variables, list)
            and bool(variables)
            and len(set(variables)) == len(variables),
            f"{description} has malformed bound variables",
            family=family,
        )
        for variable in variables:
            _name(variable, f"{description} has malformed bound variable", family)
        _boolean_expression(value["body"], description, family)
    elif kind == "forall_region":
        expect(
            set(value) == {"kind", "variables", "when", "relation"},
            f"{description} has malformed quantified region equality",
            family=family,
        )
        variables = value.get("variables")
        expect(
            isinstance(variables, list)
            and bool(variables)
            and len(set(variables)) == len(variables),
            f"{description} has malformed bound variables",
            family=family,
        )
        for variable in variables:
            _name(variable, f"{description} has malformed bound variable", family)
        _boolean_expression(value["when"], description, family)
        relation = value.get("relation")
        expect(
            isinstance(relation, dict)
            and relation.get("kind") == "region_equality",
            f"{description} does not quantify a region equality",
            family=family,
        )
        condition(relation, description, family=family)
    else:
        raise ValueError(
            f"not a complete {family} contract: {description} has unsupported condition kind {kind!r}"
        )
    return value


def region_pair(value: dict, *, family: str) -> tuple[dict, dict]:
    condition(value, "tensor condition", family=family)
    expect(value.get("kind") == "region_equality", "tensor condition is not region equality", family=family)
    return value["left"], value["right"]


def shape_dimensions(parameter: dict, *, family: str) -> list[dict]:
    typ = parameter_type(parameter, "declared_type", family=family)
    expect(typ.get("kind") == "tensor", "all regional parameters must be tensors", family=family)
    return typ["shape"]


def side_extent(value: dict, side: str) -> dict:
    kind = value["kind"]
    if kind == "var":
        return side_expr(side, value["name"])
    if kind == "int":
        return value
    return {
        "kind": "binary",
        "op": value["op"],
        "lhs": side_extent(value["lhs"], side),
        "rhs": side_extent(value["rhs"], side),
    }


def shape_symbols(value: dict) -> set[str]:
    if value["kind"] == "var":
        return {value["name"]}
    if value["kind"] == "int":
        return set()
    return shape_symbols(value["lhs"]) | shape_symbols(value["rhs"])


def is_full_slice(value: dict, dimension: dict, side: str) -> bool:
    return value == {"start": int_expr(0), "stop": side_extent(dimension, side)}


def validate_relational_contract(
    verified: VerifiedDataflowContract,
    *,
    family: str,
    preserve_analyzer_conditions: bool = False,
) -> ValidatedRelationalContract:
    """Read an exact theorem without dropping analyzer-added conditions.

    Consumers must explicitly opt into preserving the separate analyzer
    condition envelope. Otherwise it must be empty (as required by the
    axis format). Opting in does not discharge any condition: the
    consumer must retain both tuples in its emitted theorem. The full unified
    artifact remains the provenance identity, not a relabeled regional proof.
    This is schema validation, not proof authentication.
    """
    from .relational_artifact import VerifiedDataflowContract

    if isinstance(verified, VerifiedDataflowContract):
        data = verified.to_data()  # Validate complete typed evidence, not just the theorem.
        evidence = data["evidence"]
        expect(
            preserve_analyzer_conditions
            or (not evidence["used_assumptions"] and not evidence["external_obligations"]),
            "consumer cannot discharge or erase unified analyzer assumptions/obligations",
            family=family,
        )
        validated = validate_relational_contract_data(
            data["theorem_contract"], family=family, proof_kind="relational_dataflow",
        )
        return replace(
            validated,
            used_assumptions=tuple(evidence["used_assumptions"]),
            external_obligations=tuple(evidence["external_obligations"]),
        )
    raise ValueError(f"not a complete {family} contract: unsupported verified artifact type")


def validate_relational_contract_data(
    raw: dict, *, family: str, proof_kind: str,
) -> ValidatedRelationalContract:
    """Validate a theorem surface; this does not attest a successful proof."""
    expect(isinstance(raw, dict), "malformed raw contract", family=family)
    expect(set(raw) == RAW_FIELDS, "malformed raw contract fields", family=family)
    expect(raw.get("schema_version") == 5, "unsupported raw contract schema", family=family)
    expect(raw.get("proof_kind") == proof_kind, "wrong proof kind", family=family)
    _name(raw.get("goal_name"), "malformed proof-goal name", family)
    _name(raw.get("kernel"), "malformed kernel name", family)
    expect(
        isinstance(raw.get("source_sha256"), str)
        and _SHA256_RE.fullmatch(raw["source_sha256"]) is not None,
        "malformed source digest",
        family=family,
    )

    constants = raw.get("constants")
    expect(isinstance(constants, dict), "malformed specialized constants", family=family)
    for name, value in constants.items():
        _validate_constant(name, value, family)

    parameters_list = raw.get("parameters")
    expect(isinstance(parameters_list, list), "missing typed parameters", family=family)
    expect(
        all(isinstance(item, dict) and set(item) == PARAMETER_FIELDS for item in parameters_list),
        "malformed typed parameter",
        family=family,
    )
    parameters: dict[str, dict] = {}
    for parameter in parameters_list:
        name = _name(parameter.get("name"), "malformed parameter name", family)
        expect(name not in parameters, "duplicate parameter", family=family)
        declared_type = parameter_type(parameter, "declared_type", family=family)
        specialized_type = parameter_type(parameter, "type", family=family)
        expect(
            specialize_type(declared_type, constants) == specialized_type,
            f"specialized type for {name!r} does not match its declared type",
            family=family,
        )
        parameters[name] = parameter

    theorem = raw.get("theorem")
    expect(
        isinstance(theorem, dict) and set(theorem) == THEOREM_FIELDS,
        "missing or malformed theorem",
        family=family,
    )
    same = theorem.get("same")
    expect(
        isinstance(same, list)
        and all(isinstance(name, str) and bool(name) for name in same)
        and len(set(same)) == len(same),
        "malformed shared-name set",
        family=family,
    )
    for field in ("pre", "post"):
        values = theorem.get(field)
        expect(isinstance(values, list), f"malformed theorem {field}", family=family)
        for index, value in enumerate(values):
            condition(value, f"{field} condition {index}", family=family)
    expect(bool(theorem["post"]), "missing output theorem", family=family)
    expect(
        all(item.get("kind") == "region_equality" for item in theorem["post"]),
        "postconditions must be region equalities",
        family=family,
    )
    singletons = theorem.get("singletons")
    expect(isinstance(singletons, list), "malformed singleton mappings", family=family)
    for index, singleton in enumerate(singletons):
        expect(
            isinstance(singleton, dict)
            and set(singleton) == {"variable", "left", "right"},
            f"malformed singleton mapping {index}",
            family=family,
        )
        _name(singleton.get("variable"), f"malformed singleton variable {index}", family)
        annotation_expression(singleton["left"], f"singleton {index}", family=family)
        annotation_expression(singleton["right"], f"singleton {index}", family=family)

    return ValidatedRelationalContract(
        raw=raw,
        parameters=parameters,
        constants=constants,
        theorem=theorem,
    )
