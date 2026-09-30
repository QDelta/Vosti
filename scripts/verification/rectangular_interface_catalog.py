"""Geometry-dispatched row interfaces inferred from qualified tensor types.

Shape symbols identify semantic geometry; launch-only constants are never
dispatcher inputs. Alternatives at the SAME geometry must still export the
same complete raw theory. Dispatch does not assert numerical equivalence of
different geometries or implementations. Unknown geometry returns None.
"""

from dataclasses import dataclass
import hashlib
import json
import re

from ir.axis_projection_contract import normalize_axis_projection_contract
from scripts.verification.rectangular_interface_codegen import (
    RenderedRectangularInterface, inferred_binding, render_rectangular_interface,
)


def _require(condition, message):
    if not condition:
        raise ValueError("rectangular catalog: " + message)


def _symbols(expression):
    kind = expression.get("kind")
    if kind == "var":
        name = expression.get("name")
        _require(isinstance(name, str) and re.fullmatch(r"[a-zA-Z_][a-zA-Z_0-9]*", name),
                 "invalid shape symbol")
        return {name}
    if kind == "int":
        _require(type(expression.get("value")) is int and expression["value"] >= 0,
                 "invalid tensor extent")
        return set()
    _require(kind == "binary" and expression.get("op") in {"+", "*"},
             "unsupported shape expression")
    return _symbols(expression["lhs"]) | _symbols(expression["rhs"])


def _expression(expression, batch, rows):
    _symbols(expression)  # shared validation, including literal extent bounds
    if expression["kind"] == "var":
        return rows if expression["name"] == batch else expression["name"]
    if expression["kind"] == "int":
        return str(expression["value"])
    return (f"({_expression(expression['lhs'], batch, rows)} {expression['op']} "
            f"{_expression(expression['rhs'], batch, rows)})")


def _shape_symbols(types):
    return {symbol for typ in types.values() for expr in typ.get("shape", [])
            for symbol in _symbols(expr)}


def has_static_shape_geometry(implementations):
    """Whether specialization fixes semantic tensor extents, not launch tiles."""
    result = False
    for contracts in implementations:
        _require(len(contracts) == 1, "exactly one row-projection goal required")
        axis = normalize_axis_projection_contract(contracts[0]).to_data()
        result |= bool(_shape_symbols(axis["declared_parameter_types"]) & axis["constants"].keys())
    return result


def _layout(axis, binding, rows):
    conditions = []
    for argument in binding["arguments"]:
        typ = axis["parameter_types"][argument["kernel"]]
        if typ["kind"] != "tensor":
            continue
        shape = [_expression(expr, axis["batch_symbol"], rows) for expr in typ["shape"]]
        name = argument["name"]
        if len(shape) == 1:
            conditions.append(f"{name}.len() == {shape[0]}")
        else:
            _require(len(shape) == 2, "unsupported tensor rank")
            conditions.append(f"TS::tensor2d_shape({name}, {shape[0]}, {shape[1]})")
    return conditions


@dataclass(frozen=True)
class RectangularGeometryInterface:
    module: str
    geometry: tuple[tuple[str, int], ...]
    interface: RenderedRectangularInterface


@dataclass(frozen=True)
class RenderedRectangularCatalog:
    body: str
    dispatch_body: str
    interfaces: tuple[RectangularGeometryInterface, ...]
    dimensions: tuple[str, ...]

    def manifest(self):
        return json.dumps(dict(
            schema_version=1, kind="geometry_dispatched_rectangular_interfaces",
            generated_body_sha256=hashlib.sha256(self.body.encode()).hexdigest(),
            checked_dispatch_sha256=hashlib.sha256(self.dispatch_body.encode()).hexdigest(),
            dimensions=list(self.dimensions),
            interfaces=[dict(module=entry.module, geometry=dict(entry.geometry),
                raw=json.loads(entry.interface.raw.manifest()),
                checked_adapter_sha256=hashlib.sha256(entry.interface.adapter_body.encode()).hexdigest())
                for entry in self.interfaces],
        ), sort_keys=True, indent=2) + "\n"


def render_rectangular_catalog(implementations):
    """Infer branches from declared tensor shapes, not a per-kernel dimension table.

    Remaining static differences at equal geometry fail the existing complete
    theory check; we never split on arbitrary implementation/config identities.
    """
    implementations = tuple(tuple(group) for group in implementations)
    _require(bool(implementations), "no qualified implementations")
    groups = {}
    declaration = source = static_names = None
    for contracts in implementations:
        _require(len(contracts) == 1, "exactly one row-projection goal required")
        axis = normalize_axis_projection_contract(contracts[0]).to_data()
        current_source = (axis["kernel"], axis["source_sha256"])
        signature = {key: axis[key] for key in ("declared_parameter_types", "batch_symbol",
            "projected_inputs", "shared_inputs", "shared_scalar_parameters", "projected_outputs")}
        shape_names = _shape_symbols(axis["declared_parameter_types"]) - {axis["batch_symbol"]}
        static = shape_names & axis["constants"].keys()
        _require(axis["batch_symbol"] not in axis["constants"], "batch dimension cannot be specialized")
        if declaration is None:
            declaration, source, static_names = signature, current_source, static
        _require(signature == declaration, "different declared tensor interfaces")
        _require(current_source == source, "mixed kernel sources")
        _require(static == static_names, "inconsistent shape specialization surface")
        geometry = tuple((name, axis["constants"][name]["value"]) for name in sorted(static))
        _require(all(type(value) is int and value >= 0 for _, value in geometry),
                 "geometry requires nonnegative integer extents")
        groups.setdefault(geometry, []).append(contracts)

    dimensions = tuple(sorted(shape_names))
    modules, entries = [], []
    branches = {name: [] for name in ("geometry_valid", "launch_valid", "raw_output", "mapped_output", "proof", "row_map_proof")}
    common_binding = None
    for geometry, contracts in sorted(groups.items()):
        interface = render_rectangular_interface(tuple(contracts))
        axis_contract = normalize_axis_projection_contract(contracts[0][0])
        axis = axis_contract.to_data()
        own_fragment = next(bundle.contracts[0] for bundle in interface.raw.implementations
                            if bundle.contracts[0].raw_contract_digest == contracts[0][0].digest)
        binding = inferred_binding(axis_contract, contracts[0][0], own_fragment)
        surface = (binding["arguments"], binding["outputs"])
        if common_binding is None:
            common_binding = surface
        _require(surface == common_binding, "incompatible projected/shared argument surface")
        module = "geometry_" + ("_".join(f"{name}_{value}" for name, value in geometry) or "dynamic")
        entries.append(RectangularGeometryInterface(module, geometry, interface))
        modules.append(f"pub mod {module} {{\nuse super::*;\nverus! {{\n"
                       + interface.raw.body + "\n// BEGIN CHECKED RECTANGULAR ADAPTER\n"
                       + interface.adapter_body + "\n} // verus!\n}\n")
        dynamic = sorted(_shape_symbols(axis["parameter_types"]) - {axis["batch_symbol"]})
        _require(set(dynamic) == set(dimensions) - dict(geometry).keys(), "unhandled shape specialization")
        arguments = [arg["name"] for arg in binding["arguments"]]
        rows = next(arg["name"] for arg in binding["arguments"] if arg["mode"] == "projected") + ".len()"
        call = ", ".join(arguments + dynamic)
        proof_call = ", ".join(arguments + [rows] + dynamic)
        layout = _layout(axis, binding, rows)
        layout.append(f"forall|row: nat| row < {rows} ==>\n"
                      f"            #[trigger] {module}::row_projection_domain({proof_call}, row)")
        condition = " && ".join(f"{name} == {value}" for name, value in geometry) or "true"
        output_names = ["row_projection" + ("_" + output["kernel"] if len(binding["outputs"]) > 1 else "")
                        for output in binding["outputs"]]
        def output(mapped):
            expressions = [f"{module}::{name}_{'mapped_repr' if mapped else 'repr'}({call})" for name in output_names]
            return "Some(" + (expressions[0] if len(expressions) == 1 else "(" + ", ".join(expressions) + ")") + ")"
        expressions = dict(geometry_valid="true", launch_valid="\n        && ".join(f"({term})" for term in layout),
            raw_output=output(False), mapped_output=output(True),
            proof=f"{module}::row_projection_launch_equivalence({proof_call});")
        expressions["row_map_proof"] = "\n        ".join(
            f"assert(mapped{'.' + str(index) if len(output_names) > 1 else ''} =~= row_map{'.' + str(index) if len(output_names) > 1 else ''});"
            for index in range(len(output_names)))
        for name, expr in expressions.items():
            branches[name].append(f"if {condition} {{\n        {expr}\n    }}")

    def dispatch(name):
        fallback = "assert(false);" if name == "proof" else "" if name == "row_map_proof" else "false" if name.endswith("valid") else "None"
        return " else ".join(branches[name]) + f" else {{ {fallback} }}"
    arguments, outputs = common_binding
    input_names = [arg["name"] for arg in arguments] + list(dimensions)
    _require(len(input_names) == len(set(input_names)), "colliding input/geometry symbols")
    inputs = ", ".join([f"{arg['name']}: {arg['type']}" for arg in arguments]
                       + [f"{name}: nat" for name in dimensions])
    call = ", ".join(input_names)
    output_type = outputs[0]["type"] if len(outputs) == 1 else "(" + ", ".join(o["type"] for o in outputs) + ")"
    singleton_call = ", ".join([f"seq![{arg['name']}[row]]" if arg["mode"] == "projected" else arg["name"]
                                for arg in arguments] + list(dimensions))
    row_expressions = [f"Seq::new({rows}, |row: int| raw_output({singleton_call}).unwrap()"
                       + (f".{index}" if len(outputs) > 1 else "") + "[0])" for index in range(len(outputs))]
    row_map = row_expressions[0] if len(outputs) == 1 else "(" + ", ".join(row_expressions) + ")"
    dispatch_body = f'''verus! {{
pub open spec fn geometry_valid({', '.join(f'{name}: nat' for name in dimensions)}) -> bool {{
    {dispatch('geometry_valid')}
}}

pub open spec fn launch_valid({inputs}) -> bool {{
    {dispatch('launch_valid')}
}}

pub open spec fn raw_output({inputs}) -> Option<{output_type}> {{
    {dispatch('raw_output')}
}}

pub open spec fn mapped_output({inputs}) -> Option<{output_type}> {{
    {dispatch('mapped_output')}
}}

// A public row map independent of the generated geometry module names.
pub open spec fn row_mapped_output({inputs}) -> Option<{output_type}> {{
    if geometry_valid({', '.join(dimensions)}) {{ Some({row_map}) }} else {{ None }}
}}

pub proof fn checked_row_map({inputs})
    ensures mapped_output({call}) == row_mapped_output({call}),
{{
    let mapped = mapped_output({call}).unwrap();
    let row_map = row_mapped_output({call}).unwrap();
    {dispatch('row_map_proof')}
}}

pub proof fn checked_launch_equivalence({inputs})
    requires launch_valid({call}),
    ensures geometry_valid({', '.join(dimensions)}),
        raw_output({call}).is_some(),
        raw_output({call}) == mapped_output({call}),
        raw_output({call}) == row_mapped_output({call}),
{{
    checked_row_map({call});
    {dispatch('proof')}
}}
}} // verus!
'''
    _require("external_body" not in dispatch_body and "assume(" not in dispatch_body,
             "dispatcher cannot introduce trusted proofs")
    header = '''// @generated by scripts/verification/rectangular_interface_catalog.py; DO NOT EDIT.
#![allow(non_snake_case)]
use vstd::prelude::*;
use crate::{boundary::scalar::{Scalar}, proof::tensor::types::{Tensor1D, Tensor2D, IntTensor1D}};
use crate::proof::tensor::shape as TS;
#[cfg(verus_only)]
use crate::boundary::backend_certificates::support::generated_kernel_allocation_cell;

'''
    body = header + "\n".join(modules) + "\n// BEGIN CHECKED GEOMETRY DISPATCH\n" + dispatch_body
    return RenderedRectangularCatalog(body, dispatch_body, tuple(entries), dimensions)
