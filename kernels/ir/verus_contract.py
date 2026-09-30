"""Fail-closed lowering of proved kernel contracts to raw Verus specs.

This backend is deliberately operation- and framework-neutral.  It consumes
validated unified artifacts, preserving any analyzer conditions as explicit
uninterpreted premises, and renders the theorem's raw domain, postcondition, and
program-instance mapping. Temporal goals delegate to the typed before/after
lowerer, sharing the execution layout and output functions below.
Framework-specific semantic functions and checked
representation adapters belong to the consumer.

Tensor shapes are explicit predicates in the output.  The source ContractIR
uses shaped tensor types, whereas a Verus ``Seq<Seq<...>>`` permits ragged
values; omitting these predicates would silently strengthen the imported
kernel theorem.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re

from .contract_schema import ValidatedRelationalContract, validate_relational_contract
from .relational_artifact import VerifiedDataflowContract


_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_RESERVED = {
    "as", "break", "const", "continue", "crate", "else", "enum", "extern",
    "false", "fn", "for", "if", "impl", "in", "let", "loop", "match",
    "mod", "move", "mut", "pub", "ref", "return", "self", "Self", "static",
    "struct", "super", "trait", "true", "type", "unsafe", "use", "where",
    "while", "async", "await", "dyn", "abstract", "become", "box", "do",
    "final", "macro", "override", "priv", "typeof", "unsized", "virtual",
    "yield", "try",
}
_ARITHMETIC_OPS = {"+", "-", "*", "//", "%", "cdiv", "min", "max"}
_ORDER_OPS = {">", ">=", "<", "<="}


@dataclass(frozen=True)
class RenderedAnalyzerCondition:
    """Exact analyzer label paired with a required, uninterpreted predicate.

    This is an audit mapping, not a translation of prose into numeric logic.
    The predicate's interpretation at the kernel boundary remains trusted.
    """

    kind: str
    label: str
    predicate_name: str


@dataclass(frozen=True)
class RenderedVerusContract:
    """One deterministic raw Verus contract fragment."""

    kernel: str
    raw_contract_digest: str
    symbol_prefix: str
    side_type: str
    free_type: str
    program_type: str | None
    pre_name: str
    post_name: str
    singleton_name: str | None
    execute_name: str
    certificate_name: str
    output_parameters: tuple[str, ...]
    output_functions: tuple[str, ...]
    analyzer_conditions: tuple[RenderedAnalyzerCondition, ...]
    body: str
    dtype_parameters: tuple[str, ...] = ()

    @property
    def body_sha256(self) -> str:
        return hashlib.sha256(self.body.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class RenderedVerusKernel:
    """Several proved goals over one opaque kernel execution model.

    ``body`` declares the execution model once. Individual contracts remain
    self-contained for standalone checks; do not concatenate their bodies.
    The execution identity binds source, specialization, typed parameters and
    analyzed IR, independently of goal names and theorem evidence.
    """

    execution_identity: str
    execute_name: str
    side_type: str
    output_parameters: tuple[str, ...]
    contracts: tuple[RenderedVerusContract, ...]
    body: str


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(f"invalid generic Verus contract: {message}")


def _identifier(value: object, description: str) -> str:
    _require(
        isinstance(value, str)
        and _IDENTIFIER.fullmatch(value) is not None
        and value not in _RESERVED,
        description,
    )
    return value


def _pascal(value: str) -> str:
    parts = [part for part in value.split("_") if part]
    _require(bool(parts), "empty generated type name")
    result = "".join(part[0].upper() + part[1:] for part in parts)
    return _identifier(result, "invalid generated Verus type name")


def _typed_constant(value: dict) -> int | float | bool:
    kind = value["kind"]
    raw = value["value"]
    if kind == "bool":
        return bool(raw)
    if kind == "int":
        return int(raw)
    if kind == "float":
        return float(raw)
    raise AssertionError(kind)


def _scalar_verus_type(kind: str) -> str:
    if kind == "int":
        return "int"
    if kind == "bool":
        return "bool"
    if kind == "float":
        return "Scalar"
    raise ValueError(f"invalid generic Verus contract: unsupported scalar kind {kind!r}")


def _tensor_verus_type(typ: dict) -> str:
    _require(typ.get("kind") == "tensor", "expected a tensor type")
    element = _scalar_verus_type(typ["element"]["kind"])
    for _ in typ["shape"]:
        element = f"Seq<{element}>"
    return element


def _shape_symbols(expr: dict) -> set[str]:
    kind = expr["kind"]
    if kind == "var":
        return {expr["name"]}
    if kind == "int":
        return set()
    return _shape_symbols(expr["lhs"]) | _shape_symbols(expr["rhs"])


def _expr_names(expr: dict) -> tuple[set[str], set[str], set[str]]:
    """Return side names, free names, and indexed tensor names."""

    kind = expr["kind"]
    if kind == "side":
        return {expr["name"]}, set(), set()
    if kind == "free":
        return set(), {expr["name"]}, set()
    if kind == "int":
        return set(), set(), set()
    if kind == "binary":
        left = _expr_names(expr["lhs"])
        right = _expr_names(expr["rhs"])
        return tuple(left[i] | right[i] for i in range(3))  # type: ignore[return-value]
    if kind == "index":
        side_names = {expr["base"]["name"]}
        free_names: set[str] = set()
        indexed = {expr["base"]["name"]}
        for index in expr["indices"]:
            names = _expr_names(index)
            side_names |= names[0]
            free_names |= names[1]
            indexed |= names[2]
        return side_names, free_names, indexed
    raise AssertionError(kind)


def _bool_names(value: dict) -> tuple[set[str], set[str], set[str]]:
    kind = value["kind"]
    if kind == "comparison":
        left = _expr_names(value["lhs"])
        right = _expr_names(value["rhs"])
        return tuple(left[i] | right[i] for i in range(3))  # type: ignore[return-value]
    if kind == "and":
        result = (set(), set(), set())
        for item in value["args"]:
            names = _bool_names(item)
            result = tuple(result[i] | names[i] for i in range(3))  # type: ignore[assignment]
        return result
    if kind == "implies":
        left = _bool_names(value["antecedent"])
        right = _bool_names(value["consequent"])
        return tuple(left[i] | right[i] for i in range(3))  # type: ignore[return-value]
    raise AssertionError(kind)


def _condition_names(value: dict) -> tuple[set[str], set[str], set[str]]:
    kind = value["kind"]
    if kind == "scalar":
        left = _expr_names(value["lhs"])
        right = _expr_names(value["rhs"])
        return tuple(left[i] | right[i] for i in range(3))  # type: ignore[return-value]
    if kind == "forall":
        side, free, indexed = _bool_names(value["body"])
        free -= set(value["variables"])
        return side, free, indexed
    if kind == "forall_region":
        domain = _bool_names(value["when"])
        relation = _condition_names(value["relation"])
        combined = tuple(domain[i] | relation[i] for i in range(3))
        combined[1].difference_update(value["variables"])
        return combined  # type: ignore[return-value]
    if kind == "region_equality":
        side_names: set[str] = set()
        free_names: set[str] = set()
        indexed = {value["left"]["tensor"], value["right"]["tensor"]}
        for side in ("left", "right"):
            for item in value[side]["slices"]:
                for endpoint in ("start", "stop"):
                    names = _expr_names(item[endpoint])
                    side_names |= names[0]
                    free_names |= names[1]
                    indexed |= names[2]
        if value["given"] is not None:
            names = _expr_names(value["given"])
            side_names |= names[0]
            free_names |= names[1]
            indexed |= names[2]
        return side_names, free_names, indexed
    raise AssertionError(kind)


def _expr_free_vars(expr: dict) -> set[str]:
    return _expr_names(expr)[1]


def _flatten_and(value: dict) -> list[dict]:
    if value["kind"] == "and":
        return [item for arg in value["args"] for item in _flatten_and(arg)]
    return [value]


def _bounded_quantifier(value: dict) -> None:
    variables = set(value["variables"])
    body = value["body"]
    _require(
        body["kind"] == "implies",
        "forall bodies must be guarded implications",
    )
    comparisons = [
        item for item in _flatten_and(body["antecedent"])
        if item["kind"] == "comparison"
    ]
    lower: set[str] = set()
    upper: set[str] = set()
    for comparison in comparisons:
        lhs = comparison["lhs"]
        rhs = comparison["rhs"]
        op = comparison["op"]
        if lhs.get("kind") == "free" and lhs["name"] in variables:
            if lhs["name"] not in _expr_free_vars(rhs):
                if op in {">", ">="}:
                    lower.add(lhs["name"])
                elif op in {"<", "<="}:
                    upper.add(lhs["name"])
        if rhs.get("kind") == "free" and rhs["name"] in variables:
            if rhs["name"] not in _expr_free_vars(lhs):
                if op in {"<", "<="}:
                    lower.add(rhs["name"])
                elif op in {">", ">="}:
                    upper.add(rhs["name"])
    missing = sorted(name for name in variables if name not in lower or name not in upper)
    _require(
        not missing,
        f"forall variables lack explicit lower and upper guards: {missing}",
    )


class _ExecutionLayout:
    """Shared launch-state shape and opaque output lowering for all proof kinds."""

    dtype_parameters: tuple[str, ...] = ()

    def _dtype_definition(self) -> str:
        if not self.dtype_parameters:
            return ""
        return f"""#[verifier::external_body]
#[verifier::ext_equal]
pub struct {_pascal(self.execution_prefix)}DType {{
    _private: (),
}}

"""

    def _constant(self, name: str) -> str:
        value = self.constants[name]
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, int):
            return str(value)
        raise ValueError(
            "invalid generic Verus contract: floating specialized constants "
            "cannot be materialized as opaque Scalar values"
        )

    def _shape_expr(self, value: dict, side: str) -> str:
        kind = value["kind"]
        if kind == "int":
            return str(value["value"])
        if kind == "var":
            name = value["name"]
            if name in self.constants:
                return self._constant(name)
            return f"{side}.{name}"
        lhs = self._shape_expr(value["lhs"], side)
        rhs = self._shape_expr(value["rhs"], side)
        if value["op"] == "cdiv":
            return f"{self.cdiv_name}({lhs}, {rhs})"
        if value["op"] in {"min", "max"}:
            return f"vstd::math::{value['op']}({lhs}, {rhs})"
        actual = "/" if value["op"] == "//" else value["op"]
        return f"({lhs} {actual} {rhs})"

    def _shape_conditions(self, side: str, name: str) -> list[str]:
        typ = self.tensor_types[name]
        dimensions = [self._shape_expr(item, side) for item in typ["shape"]]
        tensor = f"{side}.{name}"
        conditions = [f"0 <= {dimension}" for dimension in dimensions]
        conditions.append(f"{tensor}.len() as int == {dimensions[0]}")
        for axis in range(1, len(dimensions)):
            variables = [f"_s{i}" for i in range(axis)]
            access = tensor + "".join(f"[{variable}]" for variable in variables)
            bounds: list[str] = []
            prefix = tensor
            for variable in variables:
                bounds.append(f"0 <= {variable} < {prefix}.len() as int")
                prefix += f"[{variable}]"
            binders = ", ".join(f"{variable}: int" for variable in variables)
            conditions.append(
                f"forall|{binders}| {' && '.join(bounds)} ==> "
                f"(#[trigger] {access}).len() as int == {dimensions[axis]}"
            )
        return conditions

    @staticmethod
    def _conjunction(conditions: list[str], indent: str = "    ") -> str:
        if not conditions:
            return indent + "true"
        return "\n".join(f"{indent}&&& {condition}" for condition in conditions)

    def _output_definition(self, name: str) -> tuple[str, str]:
        typ = self.tensor_types[name]
        rank = len(typ["shape"])
        cell_name = f"{self.execution_prefix}_{name}_cell"
        output_name = f"{self.execution_prefix}_{name}_after"
        indices = [f"_i{axis}" for axis in range(rank)]
        signature = ", ".join(
            [f"before: {self.side_type}"]
            + [f"{index}: int" for index in indices]
        )
        element_type = _scalar_verus_type(typ["element"]["kind"])
        value = f"{cell_name}(before, {', '.join(indices)})"
        for axis in reversed(range(rank)):
            prefix = f"before.{name}" + "".join(
                f"[{indices[previous]}]" for previous in range(axis)
            )
            value = f"Seq::new({prefix}.len(), |{indices[axis]}: int| {value})"
        definition = f"""
pub uninterp spec fn {cell_name}({signature}) -> {element_type};

pub open spec fn {output_name}(before: {self.side_type})
    -> {_tensor_verus_type(typ)}
{{
    {value}
}}
"""
        return output_name, definition

    def _side_fields(self) -> list[tuple[str, str]]:
        side_fields: list[tuple[str, str]] = []
        for name, parameter in self.parameters.items():
            typ = parameter["declared_type"]
            if typ["kind"] == "tensor":
                side_fields.append((name, _tensor_verus_type(typ)))
            elif name not in self.constants:
                side_fields.append((name, _scalar_verus_type(typ["kind"])))
        existing = {name for name, _ in side_fields}
        for name, kind in sorted(self.scalar_kinds.items()):
            if name not in existing and name not in self.constants:
                side_fields.append((name, _scalar_verus_type(kind)))
        for name, _ in side_fields:
            _identifier(name, f"invalid side field {name!r}")
        existing = {name for name, _ in side_fields}
        for name in self.dtype_parameters:
            _require(name in self.tensor_types, "dtype metadata is not a tensor parameter")
            field = f"__element_dtype_{name}"
            _require(field not in existing, "dtype metadata field collides with a source field")
            side_fields.append((field, _pascal(self.execution_prefix) + "DType"))
        return side_fields


class _Renderer(_ExecutionLayout):
    def __init__(
        self, verified: VerifiedDataflowContract,
        validated: ValidatedRelationalContract, symbol_prefix: str,
        *, execution_prefix: str | None = None,
        dtype_parameters: tuple[str, ...] = (),
    ):
        self.verified = verified
        self.dtype_parameters = dtype_parameters
        self.raw = validated.raw
        self.parameters = validated.parameters
        self.constants = {
            name: _typed_constant(value)
            for name, value in validated.constants.items()
        }
        self.theorem = validated.theorem
        self.analyzer_conditions = tuple(
            RenderedAnalyzerCondition(kind, label, f"{symbol_prefix}_{kind}_{index}")
            for kind, labels in (
                ("used_assumption", validated.used_assumptions),
                ("external_obligation", validated.external_obligations),
            )
            for index, label in enumerate(labels)
        )
        self.kernel = _identifier(self.raw["kernel"], "invalid kernel name")
        self.prefix = _identifier(symbol_prefix, "invalid generated symbol prefix")
        self.execution_prefix = _identifier(
            execution_prefix or self.prefix, "invalid execution symbol prefix"
        )
        self.cdiv_name = self.prefix + "_cdiv"
        self.side_type = _pascal(self.execution_prefix) + "Side"
        self.free_type = _pascal(self.prefix) + "Free"
        self.program_type = (
            _pascal(self.prefix) + "Programs"
            if self.theorem["singletons"] else None
        )
        self.scalar_kinds: dict[str, str] = {}
        self.tensor_types: dict[str, dict] = {}
        self.free_names: set[str] = set()
        self.quantified_definitions: list[str] = []
        self.output_names = tuple(dict.fromkeys(
            condition["left"]["tensor"]
            for condition in self.theorem["post"]
        ))
        self._build_environment()

    def _build_environment(self) -> None:
        for name in self.constants:
            _identifier(name, f"invalid specialized constant name {name!r}")
        for name, parameter in self.parameters.items():
            _identifier(name, f"invalid parameter name {name!r}")
            _require(
                name not in self.constants,
                f"parameter {name!r} collides with a specialized constant",
            )
            typ = parameter["declared_type"]
            if typ["kind"] == "tensor":
                self.tensor_types[name] = parameter["type"]
                for dimension in typ["shape"]:
                    for symbol in _shape_symbols(dimension):
                        _identifier(symbol, f"invalid shape symbol {symbol!r}")
                        _require(
                            symbol not in self.tensor_types,
                            f"shape symbol {symbol!r} collides with a tensor",
                        )
                        prior = self.scalar_kinds.setdefault(symbol, "int")
                        _require(
                            prior == "int",
                            f"shape symbol {symbol!r} is not integer-valued",
                        )
            else:
                prior = self.scalar_kinds.setdefault(name, typ["kind"])
                _require(
                    prior == typ["kind"],
                    f"scalar parameter {name!r} has conflicting types",
                )

        all_conditions = self.theorem["pre"] + self.theorem["post"]
        referenced_side: set[str] = set()
        indexed_tensors: set[str] = set()
        for condition in all_conditions:
            side, free, indexed = _condition_names(condition)
            referenced_side |= side
            self.free_names |= free
            indexed_tensors |= indexed
            if condition["kind"] == "forall":
                _bounded_quantifier(condition)
                for variable in condition["variables"]:
                    _identifier(
                        variable,
                        f"invalid quantified variable {variable!r}",
                    )
            elif condition["kind"] == "forall_region":
                _bounded_quantifier({
                    "kind": "forall",
                    "variables": condition["variables"],
                    "body": {
                        "kind": "implies",
                        "antecedent": condition["when"],
                        "consequent": condition["when"],
                    },
                })
                for variable in condition["variables"]:
                    _identifier(
                        variable,
                        f"invalid quantified region variable {variable!r}",
                    )
        for singleton in self.theorem["singletons"]:
            _identifier(singleton["variable"], "invalid singleton program variable")
            for endpoint in ("left", "right"):
                side, free, indexed = _expr_names(singleton[endpoint])
                referenced_side |= side
                self.free_names |= free
                indexed_tensors |= indexed

        for name in referenced_side | set(self.theorem["same"]):
            _identifier(name, f"invalid referenced name {name!r}")
            if (
                name not in self.tensor_types
                and name not in self.constants
                and name not in self.scalar_kinds
            ):
                # This mirrors annotation_to_config: annotation-only logical
                # dimensions (for example total key rows ``Tk``) are Z3 Ints
                # even when they are not concrete Triton parameters.
                self.scalar_kinds[name] = "int"
        for name in indexed_tensors:
            _require(name in self.tensor_types, f"indexed name {name!r} is not a tensor")
        for name in self.free_names:
            _identifier(name, f"invalid free variable {name!r}")
        self.free_names -= set(self.constants)

        # The regional analyzer interprets source/grid iterators in the
        # current execution context. A whole-launch theorem cannot replace
        # that scoped family with one caller-chosen integer (e.g. ki=0 would
        # require equality for only one cache page). Only explicit quantified
        # schemas, whose bound names have been removed above, are exported.
        unbound_iterators = (
            self.free_names | referenced_side | set(self.theorem["same"])
        ) & set(self.verified.execution_iterator_names())
        _require(not unbound_iterators,
                 "unbound execution iterators in whole-launch theorem: "
                 f"{sorted(unbound_iterators)}; use explicit quantified input relations")

        for name in self.theorem["same"]:
            _require(
                name in self.tensor_types
                or name in self.scalar_kinds
                or name in self.constants,
                f"shared name {name!r} has no type",
            )

        for condition in all_conditions:
            self._check_condition(condition)
        singleton_variables = [
            singleton["variable"] for singleton in self.theorem["singletons"]
        ]
        _require(
            len(singleton_variables) == len(set(singleton_variables)),
            "duplicate singleton program variable",
        )
        for singleton in self.theorem["singletons"]:
            _require(
                self._expr_type(singleton["left"]) == "int"
                and self._expr_type(singleton["right"]) == "int",
                f"singleton {singleton['variable']!r} coordinates must be integers",
            )

    def _expr_type(self, expr: dict, bound: set[str] | None = None) -> str:
        if bound is None:
            bound = set()
        kind = expr["kind"]
        if kind == "int":
            return "int"
        if kind == "free":
            if expr["name"] in bound:
                return "int"
            if expr["name"] in self.constants:
                value = self.constants[expr["name"]]
                if isinstance(value, bool):
                    return "bool"
                if isinstance(value, float):
                    return "float"
                return "int"
            _require(
                expr["name"] in self.free_names or expr["name"] in bound,
                f"unbound free variable {expr['name']!r}",
            )
            return "int"
        if kind == "side":
            name = expr["name"]
            if name in self.constants:
                value = self.constants[name]
                if isinstance(value, bool):
                    return "bool"
                if isinstance(value, float):
                    return "float"
                return "int"
            _require(name not in self.tensor_types, f"bare tensor reference {name!r}")
            return self.scalar_kinds[name]
        if kind == "binary":
            _require(expr["op"] in _ARITHMETIC_OPS, "unsupported arithmetic operation")
            _require(
                self._expr_type(expr["lhs"], bound) == "int"
                and self._expr_type(expr["rhs"], bound) == "int",
                f"arithmetic operation {expr['op']!r} requires integers",
            )
            return "int"
        if kind == "index":
            name = expr["base"]["name"]
            typ = self.tensor_types[name]
            _require(
                len(expr["indices"]) == len(typ["shape"]),
                f"tensor index rank mismatch for {name!r}",
            )
            _require(
                all(self._expr_type(index, bound) == "int" for index in expr["indices"]),
                f"tensor {name!r} has a non-integer index",
            )
            return typ["element"]["kind"]
        raise AssertionError(kind)

    def _check_bool(self, value: dict, bound: set[str]) -> None:
        kind = value["kind"]
        if kind == "comparison":
            lhs_type = self._expr_type(value["lhs"], bound)
            rhs_type = self._expr_type(value["rhs"], bound)
            _require(lhs_type == rhs_type, "comparison operands have different types")
            if value["op"] in _ORDER_OPS:
                _require(lhs_type == "int", "ordered comparisons require integers")
            return
        if kind == "and":
            for item in value["args"]:
                self._check_bool(item, bound)
            return
        if kind == "implies":
            self._check_bool(value["antecedent"], bound)
            self._check_bool(value["consequent"], bound)
            return
        raise AssertionError(kind)

    def _check_region(self, value: dict, bound: set[str] | None = None) -> None:
        if bound is None:
            bound = set()
        name = value["left"]["tensor"]
        _require(name in self.tensor_types, f"regional name {name!r} is not a tensor")
        rank = len(self.tensor_types[name]["shape"])
        _require(
            len(value["left"]["slices"]) == rank
            and len(value["right"]["slices"]) == rank,
            f"region rank mismatch for {name!r}",
        )
        for side in ("left", "right"):
            for item in value[side]["slices"]:
                _require(
                    self._expr_type(item["start"], bound) == "int"
                    and self._expr_type(item["stop"], bound) == "int",
                    "region endpoints must be integers",
                )
        if value["given"] is not None:
            _require(
                self._expr_type(value["given"], bound) == "int",
                "given expression must be integer",
            )

    def _check_condition(self, value: dict) -> None:
        kind = value["kind"]
        if kind == "scalar":
            self._check_bool(
                {
                    "kind": "comparison",
                    "op": value["op"],
                    "lhs": value["lhs"],
                    "rhs": value["rhs"],
                },
                set(),
            )
        elif kind == "forall":
            self._check_bool(value["body"], set(value["variables"]))
        elif kind == "region_equality":
            self._check_region(value)
        elif kind == "forall_region":
            bound = set(value["variables"])
            self._check_bool(value["when"], bound)
            self._check_region(value["relation"], bound)
        else:
            raise AssertionError(kind)


    def _expr(
        self,
        expr: dict,
        *,
        evaluation_side: str | None = None,
        bound: frozenset[str] = frozenset(),
    ) -> str:
        kind = expr["kind"]
        if kind == "int":
            return str(expr["value"])
        if kind == "free":
            name = expr["name"]
            if name in bound:
                # Keep annotation binders distinct from launch-state names and
                # the generated regional offsets, even if the source reused them.
                return f"_q{sorted(bound).index(name)}"
            if name in self.constants:
                return self._constant(name)
            return name if name not in self.free_names else f"free.{name}"
        if kind == "side":
            name = expr["name"]
            if name in self.constants:
                return self._constant(name)
            side = evaluation_side or expr["side"]
            return f"{side}.{name}"
        if kind == "binary":
            lhs = self._expr(
                expr["lhs"], evaluation_side=evaluation_side, bound=bound
            )
            rhs = self._expr(
                expr["rhs"], evaluation_side=evaluation_side, bound=bound
            )
            op = expr["op"]
            if op == "cdiv":
                return f"{self.cdiv_name}({lhs}, {rhs})"
            if op in {"min", "max"}:
                return f"vstd::math::{op}({lhs}, {rhs})"
            actual = "/" if op == "//" else op
            return f"({lhs} {actual} {rhs})"
        if kind == "index":
            side = evaluation_side or expr["base"]["side"]
            result = f"{side}.{expr['base']['name']}"
            for index in expr["indices"]:
                result += (
                    f"[{self._expr(index, evaluation_side=evaluation_side, bound=bound)}]"
                )
            return result
        raise AssertionError(kind)

    def _comparison(
        self, value: dict, *, bound: frozenset[str] = frozenset()
    ) -> str:
        op = value["op"]
        return (
            f"{self._expr(value['lhs'], bound=bound)} {op} "
            f"{self._expr(value['rhs'], bound=bound)}"
        )

    def _bool(
        self, value: dict, *, bound: frozenset[str] = frozenset()
    ) -> str:
        kind = value["kind"]
        if kind == "comparison":
            return self._comparison(value, bound=bound)
        if kind == "and":
            return "(" + " && ".join(
                self._bool(item, bound=bound) for item in value["args"]
            ) + ")"
        if kind == "implies":
            return (
                f"({self._bool(value['antecedent'], bound=bound)} ==> "
                f"{self._bool(value['consequent'], bound=bound)})"
            )
        raise AssertionError(kind)



    def _region(
        self,
        value: dict,
        *,
        bound: frozenset[str] = frozenset(),
    ) -> str:
        name = value["left"]["tensor"]
        typ = self.tensor_types[name]
        rank = len(typ["shape"])
        left_slices = value["left"]["slices"]
        right_slices = value["right"]["slices"]
        conditions: list[str] = []
        for axis in range(rank):
            left_start = self._expr(left_slices[axis]["start"], bound=bound)
            left_stop = self._expr(left_slices[axis]["stop"], bound=bound)
            right_start = self._expr(right_slices[axis]["start"], bound=bound)
            right_stop = self._expr(right_slices[axis]["stop"], bound=bound)
            left_extent = self._shape_expr(typ["shape"][axis], "left")
            right_extent = self._shape_expr(typ["shape"][axis], "right")
            conditions.extend([
                f"0 <= {left_start}",
                f"{left_start} <= {left_stop}",
                f"{left_stop} <= {left_extent}",
                f"0 <= {right_start}",
                f"{right_start} <= {right_stop}",
                f"{right_stop} <= {right_extent}",
                f"({left_stop} - {left_start}) == ({right_stop} - {right_start})",
            ])
        if value["given"] is not None:
            left_given = self._expr(
                value["given"], evaluation_side="left", bound=bound
            )
            right_given = self._expr(
                value["given"], evaluation_side="right", bound=bound
            )
            conditions.append(f"{left_given} == {right_given}")

        offsets = [f"_r{axis}" for axis in range(rank)]
        bounds = []
        left_access = f"left.{name}"
        right_access = f"right.{name}"
        for axis, offset in enumerate(offsets):
            left_start = self._expr(left_slices[axis]["start"], bound=bound)
            left_stop = self._expr(left_slices[axis]["stop"], bound=bound)
            right_start = self._expr(right_slices[axis]["start"], bound=bound)
            bounds.append(f"0 <= {offset} < ({left_stop} - {left_start})")
            left_access += f"[{left_start} + {offset}]"
            right_access += f"[{right_start} + {offset}]"
        binders = ", ".join(f"{offset}: int" for offset in offsets)
        conditions.append(
            f"forall|{binders}| {' && '.join(bounds)} ==> "
            f"#[trigger] {left_access} == {right_access}"
        )
        return "(" + " && ".join(conditions) + ")"

    def _condition(self, value: dict) -> str:
        kind = value["kind"]
        if kind == "scalar":
            return self._comparison(value)
        if kind == "region_equality":
            return self._region(value)
        if kind in {"forall", "forall_region"}:
            bound = frozenset(value["variables"])
            coordinates = ", ".join(f"_q{i}" for i in range(len(bound)))
            binders = ", ".join(f"_q{i}: int" for i in range(len(bound)))
            if kind == "forall":
                expression = self._bool(value["body"], bound=bound)
            else:
                expression = (
                    f"{self._bool(value['when'], bound=bound)} ==> "
                    f"{self._region(value['relation'], bound=bound)}"
                )
            # A transparent definition supplies a stable application trigger.
            # Asking Verus to infer one through nested regional quantifiers and
            # page-index arithmetic fails for otherwise valid annotations.
            name = f"{self.prefix}_quantified_{len(self.quantified_definitions)}"
            self.quantified_definitions.append(f"""
pub open spec fn {name}(
    left: {self.side_type}, right: {self.side_type}, free: {self.free_type},
    {binders},
) -> bool {{
    {expression}
}}
""")
            return (
                f"forall|{binders}| #[trigger] {name}(left, right, free, {coordinates})"
            )
        raise AssertionError(kind)




    def render(self, *, include_execution: bool = True) -> RenderedVerusContract:
        side_fields = self._side_fields()
        # A goal may mention only a subset of the writes (attention's selected
        # row goal omits LSE). Unmentioned writes are still opaque outputs, not
        # unchanged inputs. Single-goal and bundled exports use the same rule.
        effective_outputs = self.verified.written_tensor_parameters()
        _require(set(self.output_names) <= set(effective_outputs), "execution omits proved outputs")

        free_fields = sorted(self.free_names)
        pre_conditions: list[str] = []
        for side in ("left", "right"):
            for name in self.tensor_types:
                pre_conditions.extend(self._shape_conditions(side, name))
        for name in self.theorem["same"]:
            if name not in self.constants:
                pre_conditions.append(f"left.{name} == right.{name}")
        # Relational analysis compares the same compiled element types. When
        # temporal goals make that metadata explicit, do not extend the
        # theorem to comparisons across physical dtype specializations.
        for name in self.dtype_parameters:
            pre_conditions.append(f"left.__element_dtype_{name} == right.__element_dtype_{name}")
        pre_conditions.extend(self._condition(item) for item in self.theorem["pre"])
        condition_definitions = ""
        for condition in self.analyzer_conditions:
            pre_conditions.append(f"{condition.predicate_name}(left, right, free)")
            condition_definitions += f"""
// Additional analyzer premise; its exact label is retained in the manifest.
// This predicate is not discharged here. Its kernel-level interpretation
// (including local values and the relevant executions) remains trusted.
pub uninterp spec fn {condition.predicate_name}(
    left: {self.side_type}, right: {self.side_type}, free: {self.free_type},
) -> bool;
"""
        post_conditions = [self._condition(item) for item in self.theorem["post"]]
        pre_conditions = list(dict.fromkeys(pre_conditions))
        post_conditions = list(dict.fromkeys(post_conditions))
        quantified_definitions = "".join(self.quantified_definitions)

        side_body = "\n".join(
            f"    pub {name}: {typ}," for name, typ in side_fields
        )
        free_body = "\n".join(f"    pub {name}: int," for name in free_fields)
        pre_name = self.prefix + "_raw_pre"
        post_name = self.prefix + "_raw_post"
        singleton_name = (
            self.prefix + "_raw_singletons"
            if self.theorem["singletons"] else None
        )
        program_definition = ""
        singleton_definition = ""
        if self.theorem["singletons"]:
            assert self.program_type is not None and singleton_name is not None
            programs = [item["variable"] for item in self.theorem["singletons"]]
            program_fields = "\n".join(f"    pub {name}: int," for name in programs)
            program_definition = f"\npub struct {self.program_type} {{\n{program_fields}\n}}\n"
            singleton_conditions = []
            for item in self.theorem["singletons"]:
                variable = item["variable"]
                singleton_conditions.extend([
                    f"left_programs.{variable} == {self._expr(item['left'])}",
                    f"right_programs.{variable} == {self._expr(item['right'])}",
                ])
            singleton_definition = f"""
pub open spec fn {singleton_name}(
    left_programs: {self.program_type},
    right_programs: {self.program_type},
    left: {self.side_type},
    right: {self.side_type},
    free: {self.free_type},
) -> bool {{
{self._conjunction(singleton_conditions)}
}}
"""

        output_functions: dict[str, str] = {}
        output_definitions: list[str] = []
        for name in effective_outputs:
            function, definition = self._output_definition(name)
            output_functions[name] = function
            output_definitions.append(definition)
        output_definitions_text = "".join(output_definitions)

        execute_name = self.execution_prefix + "_execute"
        certificate_name = self.prefix + "_certificate"
        execute_fields = []
        for name, _ in side_fields:
            if name in effective_outputs:
                execute_fields.append(
                    f"        {name}: {self.execution_prefix}_{name}_after(before),"
                )
            else:
                execute_fields.append(f"        {name}: before.{name},")
        execute_body = "\n".join(execute_fields)

        side_definition = f"""{self._dtype_definition()}pub struct {self.side_type} {{
{side_body}
}}

""" if include_execution else ""
        execution_definition = f"""{output_definitions_text}
pub open spec fn {execute_name}(before: {self.side_type}) -> {self.side_type} {{
    {self.side_type} {{
{execute_body}
    }}
}}""" if include_execution else ""

        body = f"""// Raw ContractIR lowering for {self.kernel}
// Proved relational contract: {self.verified.digest}
// Whole Triton source: {self.raw['source_sha256']}

pub open spec fn {self.cdiv_name}(numerator: int, denominator: int) -> int {{
    (numerator + denominator - 1) / denominator
}}

{side_definition}pub struct {self.free_type} {{
{free_body}
}}
{program_definition}{quantified_definitions}{condition_definitions}
{execution_definition}

pub open spec fn {pre_name}(
    left: {self.side_type},
    right: {self.side_type},
    free: {self.free_type},
) -> bool {{
{self._conjunction(pre_conditions)}
}}

pub open spec fn {post_name}(
    left: {self.side_type},
    right: {self.side_type},
    free: {self.free_type},
) -> bool {{
{self._conjunction(post_conditions)}
}}
{singleton_definition}"""
        body += f"""
// Trusted import of the relational theorem proved for the source and
// specialization named above.  The generated execute function preserves
// launch metadata and allocation shape; only written output cells are opaque.
#[verifier::external_body]
pub proof fn {certificate_name}(
    left: {self.side_type},
    right: {self.side_type},
    free: {self.free_type},
)
    requires {pre_name}(left, right, free),
    ensures {post_name}(
        {execute_name}(left), {execute_name}(right), free,
    ),
{{
    unreachable!()
}}
"""
        return RenderedVerusContract(
            kernel=self.kernel,
            raw_contract_digest=self.verified.digest,
            symbol_prefix=self.prefix,
            side_type=self.side_type,
            free_type=self.free_type,
            program_type=self.program_type,
            pre_name=pre_name,
            post_name=post_name,
            singleton_name=singleton_name,
            execute_name=execute_name,
            certificate_name=certificate_name,
            output_parameters=tuple(self.output_names),
            output_functions=tuple(output_functions[name] for name in self.output_names),
            analyzer_conditions=self.analyzer_conditions,
            body=body,
            dtype_parameters=self.dtype_parameters,
        )


def render_verified_contract_to_verus(
    verified: VerifiedDataflowContract,
    *,
    symbol_prefix: str | None = None,
) -> RenderedVerusContract:
    """Render exact relational premises, including any analyzer conditions.

    Analyzer labels are not parsed as formulas. Each becomes an uninterpreted
    predicate over the two launch states and theorem coordinates, required by
    the raw import. Its trusted interpretation is recorded by the manifest.
    """

    validated = validate_relational_contract(
        verified, family="generic Verus lowering", preserve_analyzer_conditions=True,
    )
    raw = validated.raw
    kernel = raw.get("kernel")
    _identifier(kernel, "invalid kernel name")
    prefix = symbol_prefix or f"{kernel}_{verified.digest[:12]}"
    return _Renderer(verified, validated, prefix).render()


def render_verified_kernel_to_verus(
    contracts: tuple, *, symbol_prefix: str,
) -> RenderedVerusKernel:
    """Bind relational and temporal goals to one execution, without an equivalence axiom.

    This groups exact artifacts, not kernels with different configurations.
    Each goal keeps its own pre/post and additional premises. Output sets may
    differ (e.g. attention's batch goal also covers LSE): the common execution
    updates all written parameters from typed IR, without adding a relation
    about unmentioned outputs to any goal.
    Incompatible logical launch-state schemas are rejected for now.
    """
    from .exact_effect_artifact import VerifiedExactEffectContract
    from .verus_exact_effect import _TemporalRenderer

    _require(bool(contracts), "kernel bundle has no proved goals")
    _identifier(symbol_prefix, "invalid execution symbol prefix")
    dtype_parameters = tuple(sorted({
        p["name"] for contract in contracts if isinstance(contract, VerifiedExactEffectContract)
        for p in contract.to_data()["plan"]["execution"]["parameters"]
        if p["type"]["kind"] == "tensor"
    }))

    def renderer_for(contract, validated, prefix):
        if isinstance(contract, VerifiedExactEffectContract):
            return _TemporalRenderer(contract, prefix, execution_prefix=symbol_prefix,
                                     dtype_parameters=dtype_parameters)
        return _Renderer(contract, validated, prefix, execution_prefix=symbol_prefix,
                         dtype_parameters=dtype_parameters)

    entries = []
    identity = None
    names: set[str] = set()
    types: set[str] = {_pascal(symbol_prefix) + "Side"}
    if dtype_parameters:
        types.add(_pascal(symbol_prefix) + "DType")
    for contract in contracts:
        if isinstance(contract, VerifiedExactEffectContract):
            data = contract.to_data()
            validated = None
            raw = {**data["plan"]["execution"], "goal_name": data["goal_name"]}
            analyzed = raw["analyzed_kernel"]
        else:
            _require(isinstance(contract, VerifiedDataflowContract), "unsupported kernel proof artifact")
            validated = validate_relational_contract(
                contract, family="shared execution lowering", preserve_analyzer_conditions=True,
            )
            raw = validated.raw
            analyzed = contract.to_data()["evidence"]["kernel"]
        goal = _identifier(raw["goal_name"], "invalid goal name")
        _require(goal not in names, "duplicate goal in kernel bundle")
        names.add(goal)
        signature = {key: raw[key] for key in ("kernel", "source_sha256", "constants", "parameters")}
        signature["analyzed_kernel"] = analyzed
        canonical = json.dumps(signature, sort_keys=True, separators=(",", ":"))
        if identity is None:
            identity = canonical
        _require(canonical == identity, "kernel bundle mixes source, specialization or typed execution")
        renderer = renderer_for(contract, validated, f"{symbol_prefix}_{goal}")
        for name in (renderer.free_type, renderer.program_type):
            if name is not None:
                _require(name not in types, "generated goal type names collide")
                types.add(name)
        entries.append((goal, contract, validated, renderer))
    entries.sort(key=lambda entry: entry[0])
    layout = entries[0][3]._side_fields()
    for _, _, _, renderer in entries:
        _require(renderer._side_fields() == layout, "goals have incompatible logical launch-state schemas")
    outputs = entries[0][1].written_tensor_parameters()
    standalone, fragments = [], []
    for index, (goal, contract, validated, renderer) in enumerate(entries):
        standalone.append(renderer.render())
        # Renderers collect quantified helper definitions; use a fresh instance
        # rather than re-rendering mutable lowering state.
        fragment = renderer_for(contract, validated, f"{symbol_prefix}_{goal}").render(
                                 include_execution=index == 0)
        fragments.append(fragment.body)
    assert identity is not None
    return RenderedVerusKernel(
        execution_identity=hashlib.sha256(identity.encode()).hexdigest(),
        execute_name=symbol_prefix + "_execute", side_type=_pascal(symbol_prefix) + "Side",
        output_parameters=outputs, contracts=tuple(standalone), body="\n".join(fragments),
    )


def transpile_verified_kernel_source(
    source: str,
    kernel_name: str,
    constants: dict[str, int | float | bool],
    *,
    symbol_prefix: str | None = None,
    goal_name: str = "batch_invariance",
) -> RenderedVerusContract:
    """Verify a named goal and render its unified artifact directly.

    The raw renderer preserves every analyzer-added condition as an explicit
    uninterpreted precondition. No separate architecture-specific proof path
    is selected for conditional or selected-row goals.
    """

    from .proof_preparation import prepare_kernel_proofs
    from .kernel_verifier import verify_kernel_goal
    from .exact_effect_artifact import VerifiedExactEffectContract
    from .verus_exact_effect import render_verified_exact_effect_to_verus

    prepared = prepare_kernel_proofs(source, kernel_name, constants)
    artifact = verify_kernel_goal(prepared, goal_name, preserve_analyzer_conditions=True)
    if isinstance(artifact, VerifiedExactEffectContract):
        return render_verified_exact_effect_to_verus(artifact, symbol_prefix=symbol_prefix)
    return render_verified_contract_to_verus(
        artifact,
        symbol_prefix=symbol_prefix,
    )


def transpile_verified_kernel_goals(
    source: str, kernel_name: str, constants: dict[str, int | float | bool],
    *, goal_names: tuple[str, ...], symbol_prefix: str | None = None,
) -> RenderedVerusKernel:
    """Prepare one source once and prove every requested goal before emission."""
    from .proof_preparation import prepare_kernel_proofs
    from .kernel_verifier import verify_kernel_goal

    _require(bool(goal_names), "kernel bundle has no requested goals")
    _require(len(set(goal_names)) == len(goal_names), "duplicate requested goal")
    prepared = prepare_kernel_proofs(source, kernel_name, constants)
    contracts = []
    for goal in goal_names:
        contracts.append(verify_kernel_goal(prepared, goal, preserve_analyzer_conditions=True))
    return render_verified_kernel_to_verus(tuple(contracts), symbol_prefix=symbol_prefix or kernel_name)


def render_contract_manifest(contract: RenderedVerusContract | RenderedVerusKernel) -> str:
    """Return a canonical audit record for a rendered raw Verus fragment."""

    if isinstance(contract, RenderedVerusKernel):
        return json.dumps({
            "schema_version": 1,
            "kind": "shared_execution",
            "execution_identity": contract.execution_identity,
            "execute_name": contract.execute_name,
            "side_type": contract.side_type,
            "output_parameters": list(contract.output_parameters),
            # Per-goal body hashes describe self-contained fragments, not
            # slices of the deduplicated module. The module has its own hash.
            "standalone_contracts": [json.loads(render_contract_manifest(goal))
                                     for goal in contract.contracts],
            "generated_body_sha256": hashlib.sha256(contract.body.encode()).hexdigest(),
        }, indent=2, sort_keys=True) + "\n"

    from .verus_exact_effect import RenderedVerusExactEffectContract

    temporal_fields = ({"proof_kind": "exact_effect", "state_relation": "before_after"}
                       if isinstance(contract, RenderedVerusExactEffectContract) else {})
    if contract.dtype_parameters:
        temporal_fields["dtype_parameters"] = list(contract.dtype_parameters)
    return json.dumps(
        {
            "schema_version": 3,
            **temporal_fields,
            "kernel": contract.kernel,
            "raw_contract_digest": contract.raw_contract_digest,
            "symbol_prefix": contract.symbol_prefix,
            "pre_name": contract.pre_name,
            "post_name": contract.post_name,
            "singleton_name": contract.singleton_name,
            "execute_name": contract.execute_name,
            "certificate_name": contract.certificate_name,
            "output_parameters": list(contract.output_parameters),
            "output_functions": list(contract.output_functions),
            "analyzer_conditions": [
                {
                    "kind": condition.kind,
                    "label": condition.label,
                    "predicate_name": condition.predicate_name,
                }
                for condition in contract.analyzer_conditions
            ],
            "generated_body_sha256": contract.body_sha256,
        },
        indent=2,
        sort_keys=True,
    ) + "\n"


def render_standalone_verus_module(contract: RenderedVerusContract | RenderedVerusKernel) -> str:
    """Wrap a raw fragment in the smallest Verus module needed to check it."""

    return f"""#![allow(non_snake_case)]
use vstd::prelude::*;

verus! {{

#[verifier::external_body]
#[verifier::ext_equal]
pub struct Scalar {{
    _private: (),
}}

{contract.body}
}} // verus!

fn main() {{}}
"""
