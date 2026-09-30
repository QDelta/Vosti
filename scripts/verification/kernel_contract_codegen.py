"""Generate Verus contracts and checked adapters from proved kernel ContractIR.

This module does not know operation semantics.  Axis projections have one
supported lowering: the raw relational theorem plus a rectangular framework
adapter.  Any unrecognized precondition, tensor role, type, or output shape fails closed instead of being omitted.
"""

from __future__ import annotations

import hashlib
import math
import re


_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_SEMANTIC_FIELDS = {
    "kind",
    "contract_digest",
    "proof_name",
    "specialization",
    "arguments",
    "outputs",
    "domain_policy",
}
_SEMANTIC_OPTIONAL_FIELDS = {
    "verus_lowering",
    "raw_contract_digest",
}
_ARGUMENT_FIELDS = {"name", "type", "kernel", "mode"}
_OUTPUT_FIELDS = {"kernel", "function", "type"}
_AXIS_FIELDS = {
    "schema_version",
    "kind",
    "raw_contract_digest",
    "kernel",
    "source_sha256",
    "constants",
    "batch_symbol",
    "selector",
    "projected_inputs",
    "shared_inputs",
    "projected_outputs",
    "parameter_types",
    "declared_parameter_types",
    "shared_dimensions",
    "shared_scalar_parameters",
    "derived_preconditions",
    "domain_preconditions",
}
_VERUS_SEQUENCE_TYPES = {
    "Tensor1D": ("float", 1),
    "Tensor2D": ("float", 2),
    "IntTensor1D": ("int", 1),
}
_VERUS_SCALAR_TYPES = {
    "Scalar": "float",
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(f"invalid semantic kernel bridge: {message}")


def _identifier(value, description: str) -> str:
    _require(isinstance(value, str) and _IDENTIFIER.fullmatch(value) is not None, description)
    return value


def typed_constants(values: dict[str, int | float | bool]) -> dict:
    result = {}
    for name, value in sorted(values.items()):
        if isinstance(value, bool):
            result[name] = {"kind": "bool", "value": value}
        elif isinstance(value, int):
            result[name] = {"kind": "int", "value": value}
        elif isinstance(value, float) and math.isfinite(value):
            result[name] = {"kind": "float", "value": value}
        else:
            raise ValueError(f"unsupported or non-finite deployed constant {value!r}")
    return result


def validate_artifact_origin(
    framework_contract: dict,
    artifact,
    *,
    source: str,
    constants: dict[str, int | float | bool],
) -> None:
    """Independently bind a producer artifact to this exact proof invocation.

    Artifact digests already make accidental substitution unlikely, but an
    importer should not rely on a digest alone to recover its meaning.  This
    check explicitly re-establishes kernel identity, whole-source identity,
    and the complete specialization before any Verus axiom is rendered.
    """

    from ir.relational_artifact import VerifiedDataflowContract

    data = artifact.to_data()
    if isinstance(artifact, VerifiedDataflowContract):
        data = data["theorem_contract"]
    _require(
        data.get("kernel") == framework_contract.get("kernel"),
        "proved artifact names a different kernel",
    )
    _require(
        data.get("source_sha256")
        == hashlib.sha256(source.encode("utf-8")).hexdigest(),
        "proved artifact names a different whole kernel source",
    )
    expected_constants = typed_constants(constants)
    _require(
        data.get("constants") == expected_constants,
        "proved artifact specialization differs from the deployed proof case",
    )


def _require_matching_parameter_type(
    axis: dict, kernel_name: str, verus_type: str
) -> str:
    parameter_types = axis.get("parameter_types")
    _require(isinstance(parameter_types, dict), "axis contract has no parameter types")
    kernel_type = parameter_types.get(kernel_name)
    _require(isinstance(kernel_type, dict), f"kernel argument {kernel_name!r} has no type")
    kind = kernel_type.get("kind")
    if verus_type in _VERUS_SCALAR_TYPES:
        _require(
            kernel_type == {"kind": _VERUS_SCALAR_TYPES[verus_type]},
            f"framework type {verus_type!r} does not match kernel argument {kernel_name!r}",
        )
        return "scalar"

    _require(
        verus_type in _VERUS_SEQUENCE_TYPES,
        f"unsupported Verus argument type {verus_type!r}",
    )
    _require(
        kind == "tensor",
        f"framework type {verus_type!r} does not match kernel argument {kernel_name!r}",
    )
    element = kernel_type.get("element")
    shape = kernel_type.get("shape")
    expected_element, expected_rank = _VERUS_SEQUENCE_TYPES[verus_type]
    _require(
        element == {"kind": expected_element}
        and isinstance(shape, list)
        and len(shape) == expected_rank,
        f"framework type {verus_type!r} does not match kernel argument {kernel_name!r}",
    )
    return "tensor"


def _require_string_list(axis: dict, field: str) -> list[str]:
    value = axis.get(field)
    _require(
        isinstance(value, list)
        and all(isinstance(item, str) and item for item in value)
        and len(set(value)) == len(value),
        f"axis contract has malformed {field}",
    )
    return value


def validate_axis_projection_binding(axis_contract, binding: dict) -> dict:
    """Validate inferred roles against the closed axis contract surface."""

    wrapper = axis_contract.to_data()["kernel"]
    _require(
        isinstance(binding, dict) and set(binding) == _SEMANTIC_FIELDS,
        f"malformed binding for {wrapper!r}",
    )
    _require(binding["kind"] == "axis_projection", "binding kind does not match generator")

    axis = axis_contract.to_data()
    _require(set(axis) == _AXIS_FIELDS, "kernel certificate has unsupported fields")
    _require(axis.get("schema_version") == 2, "unsupported axis contract schema")
    _require(axis.get("kind") == "axis_projection", "kernel certificate has wrong kind")
    _require(
        isinstance(axis.get("parameter_types"), dict)
        and isinstance(axis.get("declared_parameter_types"), dict)
        and set(axis["parameter_types"]) == set(axis["declared_parameter_types"]),
        "kernel certificate has malformed parameter type maps",
    )
    _require_string_list(axis, "shared_dimensions")
    _require(
        isinstance(axis.get("derived_preconditions"), list),
        "kernel certificate has malformed derived preconditions",
    )
    _require(binding["contract_digest"] == axis_contract.digest, "proved contract digest mismatch")
    _require(
        axis["constants"]
        == typed_constants(binding["specialization"]),
        "binding specialization differs from proved contract",
    )
    _require(binding["domain_policy"] == "raw_preconditions",
             "only exact raw preconditions are supported")

    arguments = binding["arguments"]
    _require(isinstance(arguments, list) and arguments, "binding has no arguments")
    names = []
    kernel_names = set()
    projected = {}
    shared_tensors = {}
    shared_scalars = {}
    for argument in arguments:
        _require(
            isinstance(argument, dict) and set(argument) == _ARGUMENT_FIELDS,
            "malformed framework argument",
        )
        name = _identifier(argument["name"], "invalid Verus argument name")
        _require(name not in names, f"duplicate Verus argument {name!r}")
        names.append(name)
        typ = argument["type"]
        _require(
            typ in _VERUS_SEQUENCE_TYPES or typ in _VERUS_SCALAR_TYPES,
            f"unsupported Verus argument type {typ!r}",
        )
        kernel_name = _identifier(argument["kernel"], "invalid kernel argument name")
        _require(kernel_name not in kernel_names, f"duplicate kernel argument {kernel_name!r}")
        kernel_names.add(kernel_name)
        parameter_kind = _require_matching_parameter_type(axis, kernel_name, typ)
        if argument["mode"] == "projected":
            _require(parameter_kind == "tensor", "scalar arguments cannot be projected")
            projected[kernel_name] = argument
        elif argument["mode"] == "shared":
            if parameter_kind == "tensor":
                shared_tensors[kernel_name] = argument
            else:
                shared_scalars[kernel_name] = argument
        else:
            raise ValueError(
                "invalid semantic kernel bridge: "
                f"unknown argument mode {argument['mode']!r}"
            )
    _require(
        set(projected) == set(_require_string_list(axis, "projected_inputs")),
        "projected input binding differs from proved contract",
    )
    _require(
        set(shared_tensors) == set(_require_string_list(axis, "shared_inputs")),
        "shared tensor binding differs from proved contract",
    )
    _require(
        set(shared_scalars)
        == set(_require_string_list(axis, "shared_scalar_parameters")),
        "shared scalar binding differs from proved contract",
    )

    outputs = binding["outputs"]
    _require(isinstance(outputs, list) and outputs, "binding has no outputs")
    output_kernels = []
    output_functions = []
    for output in outputs:
        _require(
            isinstance(output, dict) and set(output) == _OUTPUT_FIELDS,
            "malformed output binding",
        )
        kernel_name = _identifier(output["kernel"], "invalid kernel output name")
        function = _identifier(
            output["function"], "invalid framework semantic function"
        )
        _require(
            kernel_name not in output_kernels,
            "duplicate kernel output binding",
        )
        _require(
            function not in output_functions,
            "duplicate semantic output function",
        )
        output_kernels.append(kernel_name)
        output_functions.append(function)
        output_type = output["type"]
        _require(
            output_type in _VERUS_SEQUENCE_TYPES,
            f"unsupported Verus output type {output_type!r}",
        )
        _require(
            _require_matching_parameter_type(axis, kernel_name, output_type)
            == "tensor",
            "projected output is not a tensor",
        )
    _require(
        set(output_kernels)
        == set(_require_string_list(axis, "projected_outputs")),
        "output binding differs from proved contract",
    )
    _identifier(binding["proof_name"], "invalid generated proof name")
    return axis


def validate_raw_rectangular_contract(raw_contract):
    from ir.contract_schema import validate_relational_contract
    return validate_relational_contract(raw_contract, family="raw rectangular adapter").raw


def render_raw_rectangular_axis_adapter(
    raw_contract, axis_contract, raw_verus_contract, binding: dict,
) -> str:
    """Render a raw ContractIR import plus a checked rectangular adapter.

    Supports row-projected rank-one/rank-two inputs, shared rank-one/rank-two
    tensor inputs, shared scalar parameters, and row-projected rank-two
    outputs.  Output allocations are synthesized from their declared symbolic
    shapes, so an output need not match an input shape.  Unsupported roles or
    shapes fail closed.

    The raw theorem itself is emitted by ``kernels.ir.verus_contract``;
    the proof below only instantiates its exact pre/post predicates with
    rectangular framework sequences.  No numerical operation semantics are
    introduced here.
    """

    wrapper = axis_contract.to_data()["kernel"]
    _require(
        isinstance(binding, dict)
        and set(binding) == _SEMANTIC_FIELDS | _SEMANTIC_OPTIONAL_FIELDS,
        f"malformed raw rectangular binding for {wrapper!r}",
    )
    _require(
        binding["verus_lowering"] == "raw_rectangular",
        "raw rectangular renderer selected by a different lowering",
    )

    # Reuse the operation-neutral role/type/domain validator before applying
    # the only supported Verus lowering.
    core_binding = {field: binding[field] for field in _SEMANTIC_FIELDS}
    axis = validate_axis_projection_binding(
        axis_contract, core_binding
    )
    raw = validate_raw_rectangular_contract(raw_contract)
    _require(
        binding["raw_contract_digest"] == raw_contract.digest,
        "raw ContractIR digest differs from the binding",
    )
    _require(
        axis["raw_contract_digest"] == raw_contract.digest,
        "axis artifact does not derive from the supplied raw ContractIR",
    )
    _require(
        raw_verus_contract.raw_contract_digest == raw_contract.digest,
        "rendered raw Verus fragment has the wrong ContractIR digest",
    )
    _require(
        raw.get("proof_kind") in {"regional_equivalence", "relational_dataflow"},
        "raw rectangular adapter requires a relational equality theorem",
    )
    arguments = binding["arguments"]
    projected_arguments = [
        argument for argument in arguments if argument["mode"] == "projected"
    ]
    shared_arguments = [
        argument for argument in arguments if argument["mode"] == "shared"
    ]
    _require(
        all(
            argument["type"] in {"Tensor2D", "IntTensor1D"}
            for argument in projected_arguments
        ),
        "raw rectangular adapter supports only projected Tensor2D and "
        "IntTensor1D inputs",
    )
    _require(
        all(
            argument["type"] in {"Tensor1D", "Tensor2D", "Scalar"}
            for argument in shared_arguments
        ),
        "raw rectangular adapter supports only shared Tensor1D, Tensor2D, "
        "and Scalar arguments",
    )
    outputs = binding["outputs"]
    _require(
        all(output["type"] == "Tensor2D" for output in outputs),
        "raw rectangular adapter requires Tensor2D outputs",
    )
    expected_parameter_names = {
        *(argument["kernel"] for argument in arguments),
        *(output["kernel"] for output in outputs),
    }
    _require(
        {
            parameter.get("name")
            for parameter in raw.get("parameters", [])
            if isinstance(parameter, dict)
        }
        == expected_parameter_names,
        "raw rectangular ContractIR parameter surface differs from the binding",
    )
    batch_symbol = axis["batch_symbol"]
    parameter_types = axis["parameter_types"]
    constants = axis["constants"]

    def shape_variables(expression: dict) -> set[str]:
        kind = expression.get("kind")
        if kind == "var":
            return {_identifier(expression.get("name"), "invalid shape dimension")}
        if kind == "int":
            return set()
        if kind == "binary":
            return shape_variables(expression["lhs"]) | shape_variables(
                expression["rhs"]
            )
        raise ValueError(
            "invalid semantic kernel bridge: unsupported tensor shape expression"
        )

    def shape_expression(expression: dict) -> str:
        kind = expression.get("kind")
        if kind == "int":
            value = expression.get("value")
            _require(isinstance(value, int) and value >= 0, "negative tensor extent")
            return str(value)
        if kind == "var":
            name = _identifier(expression.get("name"), "invalid shape dimension")
            _require(name != batch_symbol, "batch dimension used as a row width")
            _require(name not in constants, "specialized dimension was not reduced")
            return name
        if kind == "binary":
            operator = expression.get("op")
            _require(operator in {"+", "*"}, "unsupported tensor shape operator")
            return (
                f"({shape_expression(expression['lhs'])} {operator} "
                f"{shape_expression(expression['rhs'])})"
            )
        raise ValueError(
            "invalid semantic kernel bridge: unsupported tensor shape expression"
        )

    dynamic_dimensions = sorted(
        {
            name
            for kernel_name in expected_parameter_names
            for expression in parameter_types[kernel_name].get("shape", [])
            for name in shape_variables(expression)
            if name != batch_symbol and name not in constants
        }
    )

    for argument in projected_arguments:
        shape = parameter_types[argument["kernel"]]["shape"]
        _require(
            len(shape) in {1, 2}
            and shape[0] == {"kind": "var", "name": batch_symbol}
            and (
                (argument["type"] == "IntTensor1D" and len(shape) == 1)
                or (argument["type"] == "Tensor2D" and len(shape) == 2)
            ),
            f"projected tensor {argument['kernel']!r} does not have its "
            "declared row-projected rank",
        )

    for output in outputs:
        shape = parameter_types[output["kernel"]]["shape"]
        _require(
            len(shape) == 2
            and shape[0] == {"kind": "var", "name": batch_symbol},
            f"projected output {output['kernel']!r} is not [batch, columns]",
        )

    _require(
        len(raw_verus_contract.output_functions) == len(outputs),
        "raw rectangular fragment output count differs from the binding",
    )
    _require(
        raw_verus_contract.output_parameters
        == tuple(output["kernel"] for output in outputs),
        "raw rectangular fragment output order differs from the binding",
    )

    proof_name = binding["proof_name"]
    proof_prefix = proof_name.removesuffix("_certificate")
    output_repr_names = {
        output["kernel"]: (
            proof_prefix + "_repr"
            if len(outputs) == 1
            else proof_prefix + "_" + output["kernel"] + "_repr"
        )
        for output in outputs
    }
    mapped_repr_names = {
        output["kernel"]: output_repr_names[output["kernel"]].removesuffix(
            "_repr"
        ) + "_mapped_repr"
        for output in outputs
    }
    allocation_names = {
        output["kernel"]: proof_prefix + "_" + output["kernel"] + "_allocation"
        for output in outputs
    }
    argument_signature = ",\n    ".join(
        f"{argument['name']}: {argument['type']}" for argument in arguments
    )
    dimension_signature = ",\n    ".join(
        f"{name}: nat" for name in dynamic_dimensions
    )
    input_signature = ",\n    ".join(
        part for part in (argument_signature, dimension_signature) if part
    )
    proof_signature = ",\n    ".join(
        part
        for part in (
            argument_signature,
            "rows: nat",
            dimension_signature,
            "i: nat",
        )
        if part
    )
    equivalence_signature = ",\n    ".join(
        part
        for part in (argument_signature, "rows: nat", dimension_signature)
        if part
    )
    input_names = ", ".join(
        [argument["name"] for argument in arguments] + dynamic_dimensions
    )
    singleton_names = ", ".join(
        [
            f"seq![{argument['name']}[i as int]]"
            if argument["mode"] == "projected"
            else argument["name"]
            for argument in arguments
        ]
        + dynamic_dimensions
    )
    mapped_singleton_names = ", ".join(
        [
            f"seq![{argument['name']}[_row]]"
            if argument["mode"] == "projected"
            else argument["name"]
            for argument in arguments
        ]
        + dynamic_dimensions
    )

    def render_side_fields(*, singleton: bool, semantic: bool = False) -> str:
        fields = []
        for argument in arguments:
            value = argument["name"]
            if singleton and argument["mode"] == "projected":
                value = f"seq![{value}[i as int]]"
            fields.append(f"{argument['kernel']}: {value}")
        for output in outputs:
            output_rows = (
                f"{projected_arguments[0]['name']}.len()"
                if semantic
                else ("1" if singleton else "rows")
            )
            output_width = shape_expression(
                parameter_types[output["kernel"]]["shape"][1]
            )
            fields.append(
                f"{output['kernel']}: {allocation_names[output['kernel']]}("
                f"{output_rows}, {output_width})"
            )
        batch_value = (
            f"{projected_arguments[0]['name']}.len() as int"
            if semantic
            else ("1" if singleton else "rows as int")
        )
        fields.append(f"{batch_symbol}: {batch_value}")
        fields.extend(f"{name}: {name} as int" for name in dynamic_dimensions)
        return ",\n        ".join(fields)

    side_fields = ",\n        ".join(
        render_side_fields(singleton=False).split(",\n        ")
    )
    singleton_side_fields = ",\n        ".join(
        render_side_fields(singleton=True).split(",\n        ")
    )
    repr_side_fields = render_side_fields(singleton=False, semantic=True)
    shape_requirements = []
    for argument in projected_arguments:
        shape = parameter_types[argument["kernel"]]["shape"]
        if argument["type"] == "IntTensor1D":
            shape_requirements.append(f"{argument['name']}.len() == rows")
        else:
            width = shape_expression(shape[1])
            shape_requirements.append(
                f"TS::tensor2d_shape({argument['name']}, rows, {width})"
            )
    for argument in shared_arguments:
        if argument["type"] == "Tensor1D":
            shape = parameter_types[argument["kernel"]]["shape"]
            _require(len(shape) == 1, "shared Tensor1D has the wrong rank")
            shape_requirements.append(
                f"{argument['name']}.len() == {shape_expression(shape[0])}"
            )
        elif argument["type"] == "Tensor2D":
            shape = parameter_types[argument["kernel"]]["shape"]
            _require(len(shape) == 2, "shared Tensor2D has the wrong rank")
            shape_requirements.append(
                f"TS::tensor2d_shape({argument['name']}, "
                f"{shape_expression(shape[0])}, {shape_expression(shape[1])})"
            )
    shape_requires = ",\n        ".join(shape_requirements)
    input_shape_assertions = "\n".join(
        f"        assert forall|r: int| 0 <= r < rows implies\n"
        f"            (#[trigger] {argument['name']}[r]).len() == "
        f"{shape_expression(parameter_types[argument['kernel']]['shape'][1])} by {{}};"
        for argument in projected_arguments
        if argument["type"] == "Tensor2D"
    )

    def projected_region_assertion(argument: dict) -> str:
        if argument["type"] == "IntTensor1D":
            return (
                f"        assert forall|r0: int| 0 <= r0 < 1 implies\n"
                f"            #[trigger] left.{argument['kernel']}[i as int + r0]\n"
                f"                == right.{argument['kernel']}[r0] by {{\n"
                f"            assert(r0 == 0);\n"
                f"        }};"
            )
        return (
            f"        assert forall|r0: int, r1: int|\n"
            f"            0 <= r0 < 1 && 0 <= r1 < "
            f"{shape_expression(parameter_types[argument['kernel']]['shape'][1])} implies\n"
            f"            #[trigger] left.{argument['kernel']}[i as int + r0][r1]\n"
            f"                == right.{argument['kernel']}[r0][r1] by {{\n"
            f"            assert(r0 == 0);\n"
            f"        }};"
        )

    region_assertions = "\n".join(
        projected_region_assertion(argument)
        for argument in projected_arguments
    )
    side_type = raw_verus_contract.side_type
    free_type = raw_verus_contract.free_type
    selector = axis["selector"]
    execute = raw_verus_contract.execute_name
    raw_pre = raw_verus_contract.pre_name
    raw_post = raw_verus_contract.post_name
    raw_certificate = raw_verus_contract.certificate_name
    output_after = dict(
        zip(
            (output["kernel"] for output in outputs),
            raw_verus_contract.output_functions,
        )
    )

    allocation_definitions = "\n\n".join(
        f"""pub open spec fn {allocation_names[output['kernel']]}(
    rows: nat,
    cols: nat,
) -> Tensor2D {{
    Seq::new(rows, |_row: int| Seq::new(cols, |_col: int|
        generated_kernel_allocation_cell()))
}}"""
        for output in outputs
    )
    repr_definitions = "\n\n".join(
        f"""pub open spec fn {output_repr_names[output['kernel']]}(
    {input_signature},
) -> Tensor2D {{
    {output_after[output['kernel']]}({side_type} {{
        {repr_side_fields},
    }})
}}

pub open spec fn {mapped_repr_names[output['kernel']]}(
    {input_signature},
) -> Tensor2D {{
    Seq::new({projected_arguments[0]['name']}.len(), |_row: int|
        {output_repr_names[output['kernel']]}({mapped_singleton_names})[0])
}}"""
        for output in outputs
    )
    ensures = ",\n        ".join(
        f"{output_repr_names[output['kernel']]}({singleton_names})\n"
        f"            == seq![{output_repr_names[output['kernel']]}({input_names})[i as int]]"
        for output in outputs
    )
    dimension_assertions = "\n".join(
        f"    assert(left_after.{name} == {name} as int);\n"
        f"    assert(right_after.{name} == {name} as int);"
        for name in dynamic_dimensions
    )
    post_assertions = "\n".join(
        f"""    assert forall|c: int| 0 <= c < {shape_expression(parameter_types[output['kernel']]['shape'][1])} implies
        (#[trigger] left_after.{output['kernel']}[i as int][c])
            == right_after.{output['kernel']}[0][c] by {{
        assert(0 <= 0 < ((free.{selector} + 1) - free.{selector}));
        assert(0 <= c < {shape_expression(parameter_types[output['kernel']]['shape'][1])});
        assert(left_after.{output['kernel']}[free.{selector} + 0][0 + c]
            == right_after.{output['kernel']}[0 as int][0 + c]);
    }};
    assert(left_after.{output['kernel']}[i as int]
        =~= right_after.{output['kernel']}[0]);"""
        for output in outputs
    )
    repr_assertions = "\n".join(
        f"""    reveal({output_repr_names[output['kernel']]});
    assert({output_repr_names[output['kernel']]}({input_names})[i as int]
        == left_after.{output['kernel']}[i as int]);
    assert({output_repr_names[output['kernel']]}({singleton_names})[0]
        == right_after.{output['kernel']}[0]);
    assert({output_repr_names[output['kernel']]}({singleton_names})
        =~= seq![{output_repr_names[output['kernel']]}({input_names})[i as int]]);"""
        for output in outputs
    )
    equivalence_ensures = ",\n        ".join(
        f"{output_repr_names[output['kernel']]}({input_names})\n"
        f"            == {mapped_repr_names[output['kernel']]}({input_names})"
        for output in outputs
    )
    equivalence_assertions = "\n".join(
        f"""    assert forall|_row: int| 0 <= _row < rows implies
        (#[trigger] {output_repr_names[output['kernel']]}({input_names})[_row])
            == {mapped_repr_names[output['kernel']]}({input_names})[_row] by {{
        {proof_name}({', '.join(argument['name'] for argument in arguments)}, rows, {', '.join(dynamic_dimensions) + ',' if dynamic_dimensions else ''} _row as nat);
        reveal({mapped_repr_names[output['kernel']]});
        assert({output_repr_names[output['kernel']]}({singleton_names.replace('i as int', '_row')})[0]
            == {output_repr_names[output['kernel']]}({input_names})[_row]);
    }};
    assert({output_repr_names[output['kernel']]}({input_names})
        =~= {mapped_repr_names[output['kernel']]}({input_names}));"""
        for output in outputs
    )
    equivalence_name = proof_prefix + "_launch_equivalence"
    allocation_reveals = "\n".join(
        f"        reveal({allocation_names[output['kernel']]});"
        for output in outputs
    )

    domain_name = proof_prefix + "_domain"
    domain_arguments = ", ".join(
        [argument["name"] for argument in arguments] + ["rows"] + dynamic_dimensions)
    domain_definition = f"""// Exact source-annotation domain at the chosen projection. No domain
// policy table weakens or interprets the source predicate.
pub open spec fn {domain_name}(
    {proof_signature},
) -> bool {{
    let left = {side_type} {{ {side_fields} }};
    let right = {side_type} {{ {singleton_side_fields} }};
    let free = {free_type} {{ {selector}: i as int }};
    {raw_pre}(left, right, free)
}}
"""
    projection_requires = shape_requires + f",\n        {domain_name}({domain_arguments}, i)"
    equivalence_requires = shape_requires + (
        f",\n        forall|row: nat| row < rows ==>\n"
        f"            #[trigger] {domain_name}({domain_arguments}, row)")
    domain_reveal = f"    reveal({domain_name});\n"

    adapter = f"""
// Canonical framework values for the raw launch semantics.  Each pre-launch
// output is a generated rectangular allocation with the exact declared shape;
// the generated execute function replaces every logical output cell.
{allocation_definitions}

{repr_definitions}

{domain_definition}

// Checked rectangular instantiation of the generated raw ContractIR theorem.
pub proof fn {proof_name}(
    {proof_signature},
)
    requires
        {projection_requires},
        i < rows,
    ensures
        {ensures},
{{
{domain_reveal}
    let left = {side_type} {{
        {side_fields},
    }};
    let right = {side_type} {{
        {singleton_side_fields},
    }};
    let free = {free_type} {{ {selector}: i as int }};
    assert({raw_pre}(left, right, free)) by {{
{allocation_reveals}
{input_shape_assertions}
{region_assertions}
    }}
    {raw_certificate}(left, right, free);
    let left_after = {execute}(left);
    let right_after = {execute}(right);
    assert({raw_post}(left_after, right_after, free));
    reveal({raw_post});
    reveal({execute});
    assert(left_after.{batch_symbol} == rows as int);
    assert(right_after.{batch_symbol} == 1);
{dimension_assertions}
    assert(0 <= i as int && i as int + 1 <= left_after.{batch_symbol});
{post_assertions}
{repr_assertions}
}}

// The engine may use the mapped representation as its total row-local
// semantics.  For every valid rectangular launch, this checked theorem shows
// that it is exactly the generated raw whole-launch representation.
pub proof fn {equivalence_name}(
    {equivalence_signature},
)
    requires
        {equivalence_requires},
    ensures
        {equivalence_ensures},
{{
{equivalence_assertions}
}}
"""
    return adapter
