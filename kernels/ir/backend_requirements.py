"""Export concrete backend assumptions from one specialized verifier IR.

This module is deliberately not imported by any proof pass.  It turns the
already validated and typed IR into deployment-time test requirements; a bug
here can weaken empirical qualification, but cannot make a kernel proof pass.

The exported properties are relational/locality properties used by the batch-
invariance analyses, not full numerical-correctness specifications.  Device
probes are falsifiers for these assumptions and do not prove the Triton
compiler correct for every input.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import hashlib
import json

from . import (
    Arange,
    Assign,
    BinOp,
    BoolLit,
    BoolType,
    BroadcastTo,
    Cast,
    Exp2,
    Expr,
    FloatLit,
    FloatType,
    For,
    Full,
    If,
    IntLit,
    IntType,
    Kernel,
    Let,
    Log2,
    MaskedLoad,
    MaskedStore,
    Max,
    Maximum,
    Min,
    Not,
    ReduceMax,
    ReduceSum,
    Region,
    Rsqrt,
    Sigmoid,
    Squeeze,
    Stmt,
    TensorIndex,
    TensorType,
    TensorView,
    Transpose,
    Type,
    Unsqueeze,
    Var,
    Where,
    Zeros,
)
from .pp import pp_expr
from .preprocess import build_type_env


SCHEMA = "vosti.backend-requirements.v2"


@dataclass(frozen=True)
class BackendRequirement:
    site: str
    operation: str
    probe: str
    properties: tuple[str, ...]
    consumers: tuple[str, ...]
    inputs: tuple[dict, ...]
    output: dict | None
    attributes: tuple[tuple[str, object], ...] = ()

    def signature_dict(self) -> dict:
        return {
            "operation": self.operation,
            "probe": self.probe,
            "properties": list(self.properties),
            "consumers": list(self.consumers),
            "inputs": list(self.inputs),
            "output": self.output,
            "attributes": dict(self.attributes),
        }


def _aggregate_requirements(requirements: Sequence[BackendRequirement]) -> list[dict]:
    """Deduplicate probe-equivalent sites while retaining their provenance."""

    grouped: dict[bytes, tuple[dict, list[str]]] = {}
    for requirement in requirements:
        signature = requirement.signature_dict()
        key = _canonical_bytes(signature)
        if key not in grouped:
            grouped[key] = (signature, [])
        grouped[key][1].append(requirement.site)

    result = []
    for key in sorted(grouped):
        signature, sites = grouped[key]
        body = {"sites": sorted(sites), **signature}
        digest = hashlib.sha256(_canonical_bytes(body)).hexdigest()
        result.append({"id": digest[:20], **body})
    return result


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _element_name(typ: Type) -> str:
    match typ:
        case FloatType():
            return "abstract_float"
        case IntType():
            return "abstract_int"
        case BoolType():
            return "abstract_bool"
        case _:
            raise TypeError(f"unsupported backend element type: {typ}")


def _type_spec(typ: Type | None) -> dict:
    if typ is None:
        raise ValueError("backend requirement encountered an untyped expression")
    match typ:
        case TensorType(elem_type=elem_type, dims=dims):
            return {
                "element": _element_name(elem_type),
                "shape": [pp_expr(dim) for dim in dims],
                "shape_is_concrete": all(isinstance(dim, IntLit) for dim in dims),
            }
        case IntType() | FloatType() | BoolType():
            return {
                "element": _element_name(typ),
                "shape": [],
                "shape_is_concrete": True,
            }
        case _:
            raise TypeError(f"unsupported backend type: {typ}")


def _expr_spec(expr: Expr) -> dict:
    return _type_spec(expr.type)


def _has_float_elements(expr: Expr) -> bool:
    typ = expr.type
    if isinstance(typ, TensorType):
        typ = typ.elem_type
    return isinstance(typ, FloatType)


def _has_int_elements(expr: Expr) -> bool:
    typ = expr.type
    if isinstance(typ, TensorType):
        typ = typ.elem_type
    return isinstance(typ, IntType)


def _has_bool_elements(expr: Expr) -> bool:
    typ = expr.type
    if isinstance(typ, TensorType):
        typ = typ.elem_type
    return isinstance(typ, BoolType)


class _Collector:
    def __init__(self, kernel: Kernel) -> None:
        self.kernel = kernel
        self.type_env = build_type_env(kernel)
        self.requirements: list[BackendRequirement] = []

    def add(
        self,
        *,
        site: str,
        operation: str,
        probe: str,
        properties: Sequence[str],
        consumers: Sequence[str],
        inputs: Sequence[Expr] = (),
        output: Expr | None = None,
        output_type: Type | None = None,
        attributes: dict[str, object] | None = None,
    ) -> None:
        if output is not None and output_type is not None:
            raise ValueError("backend requirement has two output type sources")
        self.requirements.append(
            BackendRequirement(
                site=site,
                operation=operation,
                probe=probe,
                properties=tuple(dict.fromkeys(properties)),
                consumers=tuple(dict.fromkeys(consumers)),
                inputs=tuple(_expr_spec(expr) for expr in inputs),
                output=(
                    _expr_spec(output)
                    if output is not None
                    else _type_spec(output_type)
                    if output_type is not None
                    else None
                ),
                attributes=tuple(sorted((attributes or {}).items())),
            )
        )

    def expr(self, expr: Expr, site: str) -> None:
        match expr:
            case Var() | IntLit() | FloatLit() | BoolLit():
                return
            case BinOp(op=op, lhs=lhs, rhs=rhs):
                if op == "@":
                    self.add(
                        site=site,
                        operation="triton.dot",
                        probe="dot",
                        properties=(
                            "output_region_uses_only_matching_reduction_axis",
                            "finite_zero_lane_has_no_contribution",
                            "deterministic_for_fixed_inputs",
                        ),
                        consumers=("regional", "mask_aware_regional"),
                        inputs=(lhs, rhs),
                        output=expr,
                    )
                else:
                    properties = [
                        "lane_local",
                        "deterministic_for_fixed_inputs",
                    ]
                    consumers = ["regional"]
                    if op in {"<", "<=", ">", ">=", "==", "!="}:
                        properties.append("declared_comparison_semantics")
                        consumers.extend(("positional", "mask_aware_regional"))
                    elif op in {"and", "or"}:
                        properties.append("declared_boolean_semantics")
                        consumers.extend(("positional", "mask_aware_regional"))
                    elif _has_int_elements(lhs) and _has_int_elements(rhs):
                        # Deployed specialized integer operands are
                        # nonnegative shapes, indices, or loop bounds.
                        properties.append("declared_nonnegative_integer_semantics")
                        consumers.extend(("positional", "output_coverage"))
                    if op == "+":
                        consumers.append("positional")
                        if _has_float_elements(expr):
                            properties.append(
                                "negative_infinity_plus_finite_is_negative_infinity"
                            )
                    if op == "-":
                        properties.append("finite_self_subtraction_is_positive_zero")
                        consumers.append("positional")
                        if _has_float_elements(expr):
                            properties.append(
                                "negative_infinity_minus_finite_is_negative_infinity"
                            )
                    if op == "*":
                        properties.extend(
                            (
                                "finite_zero_product_is_numerical_zero",
                                "finite_one_multiplicative_identity",
                            )
                        )
                        consumers.extend(("mask_aware_regional", "positional"))
                        if _has_float_elements(expr):
                            properties.append(
                                "negative_infinity_times_positive_is_negative_infinity"
                            )
                    self.add(
                        site=site,
                        operation="triton.elementwise.binary",
                        probe="elementwise_binary",
                        properties=properties,
                        consumers=consumers,
                        inputs=(lhs, rhs),
                        output=expr,
                        attributes={"operator": op},
                    )
                self.expr(lhs, f"{site}.lhs")
                self.expr(rhs, f"{site}.rhs")
            case Min(args=args) | Max(args=args):
                self.add(
                    site=site,
                    operation=f"triton.scalar.{type(expr).__name__.lower()}",
                    probe="scalar",
                    properties=(
                        "deterministic_for_fixed_inputs",
                        "declared_nonnegative_integer_semantics",
                    ),
                    consumers=("regional", "positional", "output_coverage"),
                    inputs=args,
                    output=expr,
                )
                for index, arg in enumerate(args):
                    self.expr(arg, f"{site}.args[{index}]")
            case Zeros(shape=shape):
                self.add(
                    site=site,
                    operation="triton.zeros",
                    probe="constructor",
                    properties=("all_lanes_are_positive_zero", "deterministic_shape"),
                    consumers=("regional", "positional"),
                    output=expr,
                )
                self.exprs(shape, f"{site}.shape")
            case Full(shape=shape, value=value):
                self.add(
                    site=site,
                    operation="triton.full",
                    probe="constructor",
                    properties=("all_lanes_equal_scalar", "deterministic_shape"),
                    consumers=("regional", "positional"),
                    inputs=(value,),
                    output=expr,
                )
                self.expr(value, f"{site}.value")
                self.exprs(shape, f"{site}.shape")
            case Arange(start=start, stop=stop):
                self.add(
                    site=site,
                    operation="triton.arange",
                    probe="constructor",
                    properties=(
                        "lane_value_matches_logical_index",
                        "deterministic_shape",
                    ),
                    consumers=("regional", "positional"),
                    inputs=(start, stop),
                    output=expr,
                )
                self.expr(start, f"{site}.start")
                self.expr(stop, f"{site}.stop")
            case Where(cond=cond, on_true=on_true, on_false=on_false):
                self.add(
                    site=site,
                    operation="triton.where",
                    probe="where",
                    properties=(
                        "lane_local_selection",
                        "unselected_lane_has_no_value_dependency",
                    ),
                    consumers=("regional", "mask_aware_regional", "positional"),
                    inputs=(cond, on_true, on_false),
                    output=expr,
                )
                self.expr(cond, f"{site}.cond")
                self.expr(on_true, f"{site}.on_true")
                self.expr(on_false, f"{site}.on_false")
            case ReduceMax(value=value, axis=axis) | ReduceSum(value=value, axis=axis):
                is_max = isinstance(expr, ReduceMax)
                properties = [
                    "output_uses_only_declared_reduction_axis",
                    "deterministic_for_fixed_inputs",
                ]
                if is_max:
                    properties.append("constant_value_is_preserved")
                    if _has_bool_elements(value):
                        properties.append("bool_promotes_to_int32")
                    else:
                        properties.append("negative_infinity_identity")
                else:
                    properties.append("all_zero_sum_is_numerical_zero")
                self.add(
                    site=site,
                    operation="triton.reduce_max" if is_max else "triton.reduce_sum",
                    probe="reduction",
                    properties=properties,
                    consumers=("regional", "mask_aware_regional", "positional"),
                    inputs=(value,),
                    output=expr,
                    attributes={"axis": axis},
                )
                self.expr(value, f"{site}.value")
            case Exp2(value=value):
                self.add(
                    site=site,
                    operation="triton.exp2",
                    probe="elementwise_unary",
                    properties=(
                        "lane_local",
                        "exp2_negative_infinity_is_positive_zero",
                        "exp2_positive_zero_is_one",
                    ),
                    consumers=("regional", "positional"),
                    inputs=(value,),
                    output=expr,
                )
                self.expr(value, f"{site}.value")
            case Sigmoid(value=value) | Rsqrt(value=value) | Log2(value=value):
                self.add(
                    site=site,
                    operation=f"triton.{type(expr).__name__.lower()}",
                    probe="elementwise_unary",
                    properties=("lane_local", "deterministic_for_fixed_inputs"),
                    consumers=("regional",),
                    inputs=(value,),
                    output=expr,
                )
                self.expr(value, f"{site}.value")
            case Cast(value=value, kind=kind, target=target):
                properties = ["lane_local", "deterministic_for_fixed_inputs"]
                consumers = ["regional"]
                if kind == "float":
                    properties.extend(
                        (
                            "positive_zero_is_preserved",
                            "one_is_preserved",
                            "negative_infinity_is_preserved",
                        )
                    )
                    consumers.extend(("positional", "mask_aware_regional"))
                else:
                    properties.append("signed_int32_identity")
                    consumers.append("positional")
                self.add(
                    site=site,
                    operation="triton.cast",
                    probe="cast",
                    properties=properties,
                    consumers=consumers,
                    inputs=(value,),
                    output=expr,
                    attributes={"proof_kind": kind, "target": target},
                )
                self.expr(value, f"{site}.value")
            case Not(value=value):
                self.add(
                    site=site,
                    operation="triton.logical_not",
                    probe="elementwise_unary",
                    properties=("lane_local", "boolean_complement"),
                    consumers=("regional", "positional"),
                    inputs=(value,),
                    output=expr,
                )
                self.expr(value, f"{site}.value")
            case Maximum(lhs=lhs, rhs=rhs):
                self.add(
                    site=site,
                    operation="triton.maximum",
                    probe="elementwise_binary",
                    properties=(
                        "lane_local",
                        "maximum_finite_and_negative_infinity_returns_finite_operand",
                    ),
                    consumers=("regional", "positional"),
                    inputs=(lhs, rhs),
                    output=expr,
                )
                self.expr(lhs, f"{site}.lhs")
                self.expr(rhs, f"{site}.rhs")
            case Unsqueeze(value=value, axis=axis) | Squeeze(value=value, axis=axis):
                self.add(
                    site=site,
                    operation=f"triton.shape.{type(expr).__name__.lower()}",
                    probe="shape_mapping",
                    properties=("logical_index_bijection",),
                    consumers=("regional", "positional"),
                    inputs=(value,),
                    output=expr,
                    attributes={"axis": axis},
                )
                self.expr(value, f"{site}.value")
            case BroadcastTo(value=value, shape=shape):
                self.add(
                    site=site,
                    operation="triton.broadcast_to",
                    probe="shape_mapping",
                    properties=("output_lane_maps_to_expected_source_lane",),
                    consumers=("regional", "positional"),
                    inputs=(value,),
                    output=expr,
                )
                self.expr(value, f"{site}.value")
                self.exprs(shape, f"{site}.shape")
            case Transpose(value=value, permutation=permutation):
                self.add(
                    site=site,
                    operation="triton.trans",
                    probe="shape_mapping",
                    properties=("output_lane_maps_to_permuted_source_lane",),
                    consumers=("regional", "positional"),
                    inputs=(value,),
                    output=expr,
                    attributes={"permutation": list(permutation)},
                )
                self.expr(value, f"{site}.value")
            case TensorIndex(base=base, indices=indices):
                self.add(
                    site=site,
                    operation="triton.load.scalar",
                    probe="memory",
                    properties=("result_depends_only_on_addressed_element",),
                    consumers=("regional",),
                    inputs=indices,
                    output=expr,
                    attributes={"base": base.name},
                )
                self.exprs(indices, f"{site}.indices")
            case TensorView(base=base, region=region):
                self.add(
                    site=site,
                    operation="triton.block_pointer.view",
                    probe="memory",
                    properties=("logical_region_maps_to_declared_base_region",),
                    consumers=("regional",),
                    output=expr,
                    attributes={"base": base.name},
                )
                self.region(region, f"{site}.region")
            case MaskedLoad(base=base, region=region, mask=mask):
                self.add(
                    site=site,
                    operation="triton.load.block",
                    probe="memory",
                    properties=(
                        "in_mask_lane_reads_declared_region",
                        "out_of_mask_lane_is_positive_zero",
                    ),
                    consumers=("regional", "positional"),
                    output=expr,
                    attributes={"base": base.name},
                )
                self.region(region, f"{site}.region")
                self.region(mask, f"{site}.mask")
            case _:
                raise AssertionError(
                    f"backend requirement collector has no rule for {expr!r}"
                )

    def exprs(self, exprs: Sequence[Expr], site: str) -> None:
        for index, expr in enumerate(exprs):
            self.expr(expr, f"{site}[{index}]")

    def region(self, region: Region, site: str) -> None:
        for index, region_slice in enumerate(region):
            self.expr(region_slice.start, f"{site}[{index}].start")
            self.expr(region_slice.stop, f"{site}[{index}].stop")

    def stmt(self, stmt: Stmt, site: str) -> None:
        match stmt:
            case Let(value=value):
                self.expr(value, f"{site}.value")
            case Assign(target=target, op=op, value=value):
                if op is not None:
                    if op == "@":
                        raise AssertionError(
                            "augmented matmul has no declared backend semantics"
                        )
                    properties = [
                        "lane_local_update",
                        "deterministic_for_fixed_inputs",
                    ]
                    consumers = ["regional", "positional"]
                    if op in {"and", "or"} and _has_bool_elements(target):
                        properties.append("declared_boolean_semantics")
                        consumers.append("mask_aware_regional")
                    elif _has_int_elements(target) and _has_int_elements(value):
                        properties.append("declared_nonnegative_integer_semantics")
                        consumers.append("output_coverage")
                    if op == "+":
                        consumers.append("positional")
                        if _has_float_elements(target):
                            properties.append(
                                "negative_infinity_plus_finite_is_negative_infinity"
                            )
                    if op == "-":
                        properties.append("finite_self_subtraction_is_positive_zero")
                        consumers.append("positional")
                        if _has_float_elements(target):
                            properties.append(
                                "negative_infinity_minus_finite_is_negative_infinity"
                            )
                    if op == "*":
                        properties.extend(
                            (
                                "finite_zero_product_is_numerical_zero",
                                "finite_one_multiplicative_identity",
                            )
                        )
                        consumers.extend(("mask_aware_regional", "positional"))
                        if _has_float_elements(target):
                            properties.append(
                                "negative_infinity_times_positive_is_negative_infinity"
                            )
                    self.add(
                        site=site,
                        operation="triton.assignment_update",
                        probe="elementwise_binary",
                        properties=properties,
                        consumers=consumers,
                        inputs=(target, value),
                        output_type=target.type,
                        attributes={"operator": op},
                    )
                if isinstance(target, TensorView):
                    self.expr(target, f"{site}.target")
                self.expr(value, f"{site}.value")
            case MaskedStore(base=base, region=region, value=value, mask=mask):
                self.add(
                    site=site,
                    operation="triton.store.block",
                    probe="memory",
                    properties=(
                        "in_mask_lane_writes_declared_region",
                        "out_of_mask_lane_does_not_write",
                    ),
                    consumers=("regional", "output_coverage", "logical_race"),
                    inputs=(value,),
                    attributes={"base": base.name},
                )
                self.region(region, f"{site}.region")
                self.region(mask, f"{site}.mask")
                self.expr(value, f"{site}.value")
            case For(iters=iters, body=body):
                self.add(
                    site=site,
                    operation="triton.control.range",
                    probe="control",
                    properties=("executes_declared_half_open_iteration_space",),
                    consumers=("regional", "output_coverage"),
                    inputs=(iters.start, iters.stop),
                )
                self.expr(iters.start, f"{site}.start")
                self.expr(iters.stop, f"{site}.stop")
                self.stmts(body, f"{site}.body")
            case If(cond=cond, then_body=then_body, else_body=else_body):
                self.add(
                    site=site,
                    operation="triton.control.if",
                    probe="control",
                    properties=("executes_only_selected_branch",),
                    consumers=("regional", "output_coverage"),
                    inputs=(cond,),
                )
                self.expr(cond, f"{site}.cond")
                self.stmts(then_body, f"{site}.then")
                self.stmts(else_body, f"{site}.else")
            case _:
                raise AssertionError(
                    f"backend requirement collector has no statement rule for {stmt!r}"
                )

    def stmts(self, statements: Sequence[Stmt], site: str) -> None:
        for index, statement in enumerate(statements):
            self.stmt(statement, f"{site}[{index}]")

    def collect(self) -> list[BackendRequirement]:
        for index, grid_iter in enumerate(self.kernel.grid.iters):
            self.add(
                site=f"grid.iters[{index}]",
                operation="triton.program_id",
                probe="control",
                properties=("axis_id_enumerates_declared_grid",),
                consumers=("regional", "output_coverage", "logical_race"),
                inputs=(grid_iter.iters.start, grid_iter.iters.stop),
                attributes={"axis": index},
            )
        self.stmts(self.kernel.grid.body, "grid.body")
        return self.requirements


def build_backend_requirement_manifest(
    *,
    kernel: Kernel,
    source_name: str,
    source_sha256: str,
    constants: dict[str, int | float | bool],
    physical_float_dtypes: Sequence[str],
) -> dict:
    """Build a canonical, fail-closed requirement manifest for one case."""

    if not physical_float_dtypes:
        raise ValueError("backend requirements need a nonempty physical float domain")
    collected = _Collector(kernel).collect()
    requirements = _aggregate_requirements(collected)
    if not requirements:
        raise ValueError("specialized kernel emitted no backend requirements")
    body = {
        "schema": SCHEMA,
        "kernel": {
            "source": source_name,
            "source_sha256": source_sha256,
            "name": kernel.name,
            "specialization": dict(sorted(constants.items())),
        },
        "physical_float_dtypes": sorted(set(physical_float_dtypes)),
        "site_count": len(collected),
        "requirements": requirements,
    }
    return {
        **body,
        "manifest_sha256": hashlib.sha256(_canonical_bytes(body)).hexdigest(),
    }
