"""Normalize a proved regional theorem into a generic axis-projection contract.

An axis-projection contract says that one axis of every projected input/output
is reduced from an arbitrary full extent to a singleton, while other inputs
are shared in full.  The normalizer contains no operation names or numerical
semantics.  It rejects partial trailing regions and unclassified tensor
parameters so consumers may soundly interpret a projected output region as a
whole logical row.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json

from .contract_schema import (
    add_one as _add_one,
    expect as _schema_expect,
    free_expr as _free,
    index_expr as _index,
    int_expr as _int,
    is_full_slice as _is_full_slice,
    parameter_type as _schema_parameter_type,
    region_pair as _schema_region_pair,
    scalar_condition as _scalar,
    shape_dimensions as _schema_shape_dimensions,
    shape_symbols as _shape_symbols,
    side_expr as _side,
    validate_relational_contract,
)
from .relational_artifact import VerifiedDataflowContract


AXIS_PROJECTION_SCHEMA_VERSION = 2
_FAMILY = "axis-projection"


@dataclass(frozen=True)
class VerifiedAxisProjectionContract:
    canonical_json: str

    @property
    def digest(self) -> str:
        return hashlib.sha256(self.canonical_json.encode("utf-8")).hexdigest()

    def to_data(self) -> dict:
        data = json.loads(self.canonical_json)
        if not isinstance(data, dict):
            raise ValueError("axis-projection contract must encode an object")
        return data


def _expect(condition: bool, message: str) -> None:
    _schema_expect(condition, message, family=_FAMILY)


def _parameter_type(parameter: dict, field: str) -> dict:
    return _schema_parameter_type(parameter, field, family=_FAMILY)


def _shape_dimensions(parameter: dict) -> list[dict]:
    return _schema_shape_dimensions(parameter, family=_FAMILY)


def _region_pair(condition: dict) -> tuple[dict, dict]:
    left, right = _schema_region_pair(condition, family=_FAMILY)
    _expect(condition.get("given") is None, "given-dependent regions are not axis projections")
    return left, right


def _projected_tensor(
    condition: dict,
    parameter: dict,
    *,
    batch_symbol: str,
    selector: str,
) -> bool:
    left, right = _region_pair(condition)
    if left["tensor"] != parameter["name"]:
        return False
    shape = _shape_dimensions(parameter)
    if shape[0] != {"kind": "var", "name": batch_symbol}:
        return False
    left_slices = left.get("slices")
    right_slices = right.get("slices")
    if not isinstance(left_slices, list) or not isinstance(right_slices, list):
        return False
    if len(left_slices) != len(shape) or len(right_slices) != len(shape):
        return False
    selected = _free(selector)
    if left_slices[0] != {"start": selected, "stop": _add_one(selected)}:
        return False
    if right_slices[0] != {"start": _int(0), "stop": _int(1)}:
        return False
    return all(
        _is_full_slice(left_slice, dimension, "left")
        and _is_full_slice(right_slice, dimension, "left")
        for left_slice, right_slice, dimension in zip(
            left_slices[1:], right_slices[1:], shape[1:]
        )
    )


def _shared_tensor(condition: dict, parameter: dict) -> bool:
    left, right = _region_pair(condition)
    if left["tensor"] != parameter["name"]:
        return False
    shape = _shape_dimensions(parameter)
    left_slices = left.get("slices")
    right_slices = right.get("slices")
    if not isinstance(left_slices, list) or not isinstance(right_slices, list):
        return False
    if len(left_slices) != len(shape) or len(right_slices) != len(shape):
        return False
    return all(
        _is_full_slice(left_slice, dimension, "left")
        and _is_full_slice(right_slice, dimension, "left")
        for left_slice, right_slice, dimension in zip(
            left_slices, right_slices, shape
        )
    )


def _infer_projection(post: dict, parameters: dict[str, dict]) -> tuple[str, str]:
    left, right = _region_pair(post)
    name = left["tensor"]
    _expect(name in parameters, f"postcondition tensor {name!r} is not a parameter")
    shape = _shape_dimensions(parameters[name])
    left_slices = left.get("slices")
    right_slices = right.get("slices")
    _expect(
        isinstance(left_slices, list)
        and isinstance(right_slices, list)
        and len(left_slices) == len(shape)
        and len(right_slices) == len(shape),
        "postcondition rank does not match its output tensor",
    )
    first = left_slices[0]
    selector_expr = first.get("start") if isinstance(first, dict) else None
    _expect(
        isinstance(selector_expr, dict)
        and selector_expr.get("kind") == "free"
        and isinstance(selector_expr.get("name"), str),
        "left output does not start at a free selector",
    )
    selector = selector_expr["name"]
    _expect(
        shape[0].get("kind") == "var" and isinstance(shape[0].get("name"), str),
        "projected output has no symbolic batch axis",
    )
    batch_symbol = shape[0]["name"]
    _expect(
        _projected_tensor(
            post,
            parameters[name],
            batch_symbol=batch_symbol,
            selector=selector,
        ),
        "postcondition is not a complete selected-row equality",
    )
    return batch_symbol, selector


def _remove_required_scalar(
    conditions: list[dict], required: dict, description: str
) -> None:
    matches = [index for index, condition in enumerate(conditions) if condition == required]
    _expect(len(matches) == 1, f"expected exactly one {description}")
    conditions.pop(matches[0])


def _remove_projection_implied_preconditions(
    conditions: list[dict],
    projected_inputs: list[str],
    parameters: dict[str, dict],
    selector: str,
) -> list[dict]:
    """Remove only scalar equalities entailed by a complete rank-1 projection."""

    removed = []
    for name in projected_inputs:
        if len(_shape_dimensions(parameters[name])) != 1:
            continue
        required = _scalar(
            "==",
            _index("left", name, [_free(selector)]),
            _index("right", name, [_int(0)]),
        )
        matches = [
            index for index, condition in enumerate(conditions)
            if condition == required
        ]
        _expect(
            len(matches) <= 1,
            f"duplicate projection-implied scalar equality for {name!r}",
        )
        if matches:
            removed.append(conditions.pop(matches[0]))
    return removed


def normalize_axis_projection_contract(
    verified: VerifiedDataflowContract,
) -> VerifiedAxisProjectionContract:
    """Derive a complete selected-axis theorem from a successful raw contract."""

    validated = validate_relational_contract(verified, family=_FAMILY)
    raw = validated.raw
    parameters = validated.parameters
    constants = validated.constants
    theorem = validated.theorem
    posts = theorem["post"]
    pre = theorem["pre"]
    _expect(
        theorem.get("singletons") == [],
        "explicit loop singleton mappings require another contract family",
    )

    batch_symbol, selector = _infer_projection(posts[0], parameters)
    projected_outputs = []
    for post in posts:
        left, _ = _region_pair(post)
        name = left["tensor"]
        _expect(name not in projected_outputs, f"duplicate projected output {name!r}")
        _expect(
            name in parameters
            and _projected_tensor(
                post,
                parameters[name],
                batch_symbol=batch_symbol,
                selector=selector,
            ),
            f"output {name!r} does not use the common complete projection",
        )
        projected_outputs.append(name)

    scalar_conditions = [item for item in pre if item.get("kind") != "region_equality"]
    region_conditions = [item for item in pre if item.get("kind") == "region_equality"]
    _expect(
        len(scalar_conditions) + len(region_conditions) == len(pre),
        "unsupported precondition kind",
    )

    shared_names = theorem.get("same")
    _expect(
        isinstance(shared_names, list)
        and all(isinstance(name, str) and name for name in shared_names)
        and len(set(shared_names)) == len(shared_names),
        "malformed shared-name set",
    )
    tensor_parameters = {
        name for name, parameter in parameters.items()
        if parameter["type"].get("kind") == "tensor"
    }
    scalar_parameters = set(parameters) - tensor_parameters
    same_tensor_parameters = set(shared_names) & tensor_parameters
    same_scalar_parameters = set(shared_names) & scalar_parameters
    shared_inputs = sorted(same_tensor_parameters)
    for name in shared_inputs:
        _expect(
            name not in projected_outputs,
            f"output tensor {name!r} is declared wholly shared",
        )

    projected_inputs = []
    for condition in region_conditions:
        left, _ = _region_pair(condition)
        name = left["tensor"]
        _expect(name in parameters, f"input tensor {name!r} is not a parameter")
        _expect(
            name not in projected_outputs,
            f"output tensor {name!r} is also an input assumption",
        )
        _expect(
            name not in projected_inputs and name not in shared_inputs,
            f"duplicate input assumption for {name!r}",
        )
        if _projected_tensor(
            condition,
            parameters[name],
            batch_symbol=batch_symbol,
            selector=selector,
        ):
            projected_inputs.append(name)
        elif _shared_tensor(condition, parameters[name]):
            shared_inputs.append(name)
        else:
            raise ValueError(
                "not a complete axis-projection contract: "
                f"input {name!r} is neither a complete projection nor fully shared"
            )

    classified = set(projected_inputs) | set(shared_inputs) | set(projected_outputs)
    _expect(
        classified == tensor_parameters,
        f"unclassified tensor parameters {sorted(tensor_parameters - classified)}",
    )
    _expect(projected_inputs, "no projected input")

    required_shared_dimensions = set()
    for name in projected_inputs + projected_outputs:
        shape = _shape_dimensions(parameters[name])
        for dimension in shape[1:]:
            required_shared_dimensions.update(_shape_symbols(dimension))
    for name in shared_inputs:
        for dimension in _shape_dimensions(parameters[name]):
            required_shared_dimensions.update(_shape_symbols(dimension))
    shared_dimensions = (
        set(shared_names) - same_tensor_parameters - same_scalar_parameters
    )
    _expect(
        required_shared_dimensions == shared_dimensions,
        "shared dimensions must be exactly the non-batch tensor dimensions",
    )
    _expect(
        same_scalar_parameters == scalar_parameters,
        f"unclassified scalar parameters {sorted(scalar_parameters - same_scalar_parameters)}",
    )

    remaining = list(scalar_conditions)
    _remove_required_scalar(
        remaining,
        _scalar("==", _side("right", batch_symbol), _int(1)),
        "right singleton batch extent",
    )
    _remove_required_scalar(
        remaining,
        _scalar(">=", _free(selector), _int(0)),
        "nonnegative selector",
    )
    _remove_required_scalar(
        remaining,
        _scalar("<", _free(selector), _side("left", batch_symbol)),
        "selector upper bound",
    )
    derived_preconditions = _remove_projection_implied_preconditions(
        remaining,
        projected_inputs,
        parameters,
        selector,
    )
    _expect(
        all(item.get("kind") == "scalar" for item in remaining),
        "quantified domain conditions require another contract family",
    )

    document = {
        "schema_version": AXIS_PROJECTION_SCHEMA_VERSION,
        "kind": "axis_projection",
        "raw_contract_digest": verified.digest,
        "kernel": raw["kernel"],
        "source_sha256": raw["source_sha256"],
        "constants": constants,
        "batch_symbol": batch_symbol,
        "selector": selector,
        "projected_inputs": sorted(projected_inputs),
        "shared_inputs": sorted(shared_inputs),
        "projected_outputs": sorted(projected_outputs),
        "parameter_types": {
            name: parameters[name]["type"]
            for name in sorted(classified | same_scalar_parameters)
        },
        "declared_parameter_types": {
            name: parameters[name]["declared_type"]
            for name in sorted(classified | same_scalar_parameters)
        },
        "shared_dimensions": sorted(shared_dimensions),
        "shared_scalar_parameters": sorted(same_scalar_parameters),
        "derived_preconditions": derived_preconditions,
        "domain_preconditions": remaining,
    }
    canonical = json.dumps(document, sort_keys=True, separators=(",", ":"))
    return VerifiedAxisProjectionContract(canonical_json=canonical)
