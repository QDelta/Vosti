"""Fail-closed correspondence between exported assumptions and GPU probes.

This module is intentionally pure: it states which probe family is allowed to
test each backend operation and which properties that family implements.  The
CUDA implementations live in :mod:`backend.probes`.
"""

from __future__ import annotations


_COMMON_LANE = {"lane_local", "deterministic_for_fixed_inputs"}

ANALYSIS_CONSUMERS = frozenset(
    {
        "logical_race",
        "mask_aware_regional",
        "output_coverage",
        "positional",
        "regional",
    }
)

# Triton's transcendental elementwise primitives consume fp32/fp64 values.
# Engine kernels reach them through explicit casts or fp32 accumulators; casts
# are qualified separately.  Keep this list closed so the probe runner does
# not manufacture an unsupported bf16 invocation and then mistake that
# compilation failure for evidence about the primitive's declared property.
FLOAT32_ONLY_OPERATIONS = frozenset(
    {
        "triton.exp2",
        "triton.sigmoid",
        "triton.rsqrt",
        "triton.log2",
    }
)

_ARITHMETIC_IDENTITIES = {
    "negative_infinity_plus_finite_is_negative_infinity",
    "finite_self_subtraction_is_positive_zero",
    "negative_infinity_minus_finite_is_negative_infinity",
    "finite_zero_product_is_numerical_zero",
    "finite_one_multiplicative_identity",
    "negative_infinity_times_positive_is_negative_infinity",
}

PROBE_CONTRACTS = {
    "triton.elementwise.binary": (
        "elementwise_binary",
        _COMMON_LANE
        | {
            "declared_comparison_semantics",
            "declared_boolean_semantics",
            "declared_nonnegative_integer_semantics",
        }
        | _ARITHMETIC_IDENTITIES,
    ),
    "triton.assignment_update": (
        "elementwise_binary",
        {
            "lane_local_update",
            "deterministic_for_fixed_inputs",
            "declared_boolean_semantics",
            "declared_nonnegative_integer_semantics",
        }
        | _ARITHMETIC_IDENTITIES,
    ),
    "triton.maximum": (
        "elementwise_binary",
        {
            "lane_local",
            "maximum_finite_and_negative_infinity_returns_finite_operand",
        },
    ),
    "triton.exp2": (
        "elementwise_unary",
        {
            "lane_local",
            "exp2_negative_infinity_is_positive_zero",
            "exp2_positive_zero_is_one",
        },
    ),
    "triton.sigmoid": ("elementwise_unary", _COMMON_LANE),
    "triton.rsqrt": ("elementwise_unary", _COMMON_LANE),
    "triton.log2": ("elementwise_unary", _COMMON_LANE),
    "triton.logical_not": (
        "elementwise_unary",
        {"lane_local", "boolean_complement"},
    ),
    "triton.where": (
        "where",
        {"lane_local_selection", "unselected_lane_has_no_value_dependency"},
    ),
    "triton.cast": (
        "cast",
        _COMMON_LANE
        | {
            "signed_int32_identity",
            "positive_zero_is_preserved",
            "one_is_preserved",
            "negative_infinity_is_preserved",
        },
    ),
    "triton.reduce_max": (
        "reduction",
        {
            "output_uses_only_declared_reduction_axis",
            "deterministic_for_fixed_inputs",
            "constant_value_is_preserved",
            "negative_infinity_identity",
            "bool_promotes_to_int32",
        },
    ),
    "triton.reduce_sum": (
        "reduction",
        {
            "output_uses_only_declared_reduction_axis",
            "deterministic_for_fixed_inputs",
            "all_zero_sum_is_numerical_zero",
        },
    ),
    "triton.dot": (
        "dot",
        {
            "output_region_uses_only_matching_reduction_axis",
            "finite_zero_lane_has_no_contribution",
            "deterministic_for_fixed_inputs",
        },
    ),
    "triton.zeros": (
        "constructor",
        {"all_lanes_are_positive_zero", "deterministic_shape"},
    ),
    "triton.full": (
        "constructor",
        {"all_lanes_equal_scalar", "deterministic_shape"},
    ),
    "triton.arange": (
        "constructor",
        {"lane_value_matches_logical_index", "deterministic_shape"},
    ),
    "triton.shape.unsqueeze": ("shape_mapping", {"logical_index_bijection"}),
    "triton.shape.squeeze": ("shape_mapping", {"logical_index_bijection"}),
    "triton.broadcast_to": (
        "shape_mapping",
        {"output_lane_maps_to_expected_source_lane"},
    ),
    "triton.trans": (
        "shape_mapping",
        {"output_lane_maps_to_permuted_source_lane"},
    ),
    "triton.load.scalar": ("memory", {"result_depends_only_on_addressed_element"}),
    "triton.block_pointer.view": (
        "memory",
        {"logical_region_maps_to_declared_base_region"},
    ),
    "triton.load.block": (
        "memory",
        {"in_mask_lane_reads_declared_region", "out_of_mask_lane_is_positive_zero"},
    ),
    "triton.store.block": (
        "memory",
        {"in_mask_lane_writes_declared_region", "out_of_mask_lane_does_not_write"},
    ),
    "triton.program_id": ("control", {"axis_id_enumerates_declared_grid"}),
    "triton.control.range": (
        "control",
        {"executes_declared_half_open_iteration_space"},
    ),
    "triton.control.if": ("control", {"executes_only_selected_branch"}),
    "triton.scalar.min": (
        "scalar",
        {
            "deterministic_for_fixed_inputs",
            "declared_nonnegative_integer_semantics",
        },
    ),
    "triton.scalar.max": (
        "scalar",
        {
            "deterministic_for_fixed_inputs",
            "declared_nonnegative_integer_semantics",
        },
    ),
}

if not FLOAT32_ONLY_OPERATIONS <= PROBE_CONTRACTS.keys():
    raise AssertionError("float32-only operation has no closed probe contract")


def _element(spec: dict | None) -> str | None:
    return None if spec is None else spec.get("element")


def _required_properties(requirement: dict, supported: set[str]) -> set[str]:
    """Independently reconstruct the semantics used for this IR signature.

    Most primitive kinds always require their full closed contract.  Arithmetic
    and casts are conditional on the operator and abstract element kind.  This
    second reconstruction is intentional: deleting a property from the IR
    exporter must not silently weaken deployment qualification.
    """

    operation = requirement["operation"]
    if operation == "triton.reduce_max":
        properties = {
            "output_uses_only_declared_reduction_axis",
            "deterministic_for_fixed_inputs",
            "constant_value_is_preserved",
        }
        inputs = requirement.get("inputs", [])
        if inputs and _element(inputs[0]) == "abstract_bool":
            properties.add("bool_promotes_to_int32")
        else:
            properties.add("negative_infinity_identity")
        return properties
    if operation not in {
        "triton.elementwise.binary",
        "triton.assignment_update",
        "triton.cast",
    }:
        return set(supported)

    if operation == "triton.cast":
        properties = set(_COMMON_LANE)
        proof_kind = requirement.get("attributes", {}).get("proof_kind")
        if proof_kind == "float":
            properties |= {
                "positive_zero_is_preserved",
                "one_is_preserved",
                "negative_infinity_is_preserved",
            }
        elif proof_kind == "int32":
            properties.add("signed_int32_identity")
        else:
            raise ValueError(f"unsupported cast proof kind {proof_kind!r}")
        return properties

    properties = {
        (
            "lane_local"
            if operation == "triton.elementwise.binary"
            else "lane_local_update"
        ),
        "deterministic_for_fixed_inputs",
    }
    attributes = requirement.get("attributes", {})
    operator = attributes.get("operator")
    inputs = requirement.get("inputs", [])
    output = requirement.get("output")
    elements = {_element(spec) for spec in inputs}
    output_element = _element(output)
    if operator in {"<", "<=", ">", ">=", "==", "!="}:
        properties.add("declared_comparison_semantics")
    if operator in {"and", "or"}:
        properties.add("declared_boolean_semantics")
    if elements == {"abstract_int"} and output_element == "abstract_int":
        properties.add("declared_nonnegative_integer_semantics")
    if operator == "+":
        if output_element == "abstract_float":
            properties.add("negative_infinity_plus_finite_is_negative_infinity")
    elif operator == "-":
        properties.add("finite_self_subtraction_is_positive_zero")
        if output_element == "abstract_float":
            properties.add("negative_infinity_minus_finite_is_negative_infinity")
    elif operator == "*":
        properties |= {
            "finite_zero_product_is_numerical_zero",
            "finite_one_multiplicative_identity",
        }
        if output_element == "abstract_float":
            properties.add("negative_infinity_times_positive_is_negative_infinity")
    return properties


def validate_requirement_probe_coverage(requirement: dict) -> None:
    operation = requirement["operation"]
    contract = PROBE_CONTRACTS.get(operation)
    if contract is None:
        raise ValueError(f"no backend property coverage for {operation!r}")
    expected_probe, supported = contract
    if requirement.get("probe") != expected_probe:
        raise ValueError(
            f"backend operation {operation!r} requires probe family "
            f"{expected_probe!r}, not {requirement.get('probe')!r}"
        )
    properties = set(requirement["properties"])
    if not properties or not properties <= supported:
        raise ValueError(
            f"backend probe does not cover properties for {operation!r}: "
            f"unknown={sorted(properties - supported)}"
        )
    required = _required_properties(requirement, supported)
    if not required <= properties:
        raise ValueError(
            f"backend requirement omits verifier-used properties for {operation!r}: "
            f"missing={sorted(required - properties)}"
        )
    consumers = requirement.get("consumers")
    if (
        not isinstance(consumers, list)
        or not consumers
        or any(not isinstance(consumer, str) for consumer in consumers)
        or len(set(consumers)) != len(consumers)
        or not set(consumers) <= ANALYSIS_CONSUMERS
    ):
        unknown = (
            sorted(set(consumers) - ANALYSIS_CONSUMERS)
            if isinstance(consumers, list)
            and all(isinstance(consumer, str) for consumer in consumers)
            else []
        )
        raise ValueError(
            "backend requirement has invalid analysis consumers: "
            f"consumers={consumers!r}, unknown={unknown}"
        )
