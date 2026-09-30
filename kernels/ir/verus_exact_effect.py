"""Typed temporal propositions lowered over the shared raw kernel execution."""

from dataclasses import dataclass

from . import annotations as ann
from .exact_effect_artifact import VerifiedExactEffectContract
from .verus_contract import (
    RenderedAnalyzerCondition, RenderedVerusContract, _ExecutionLayout,
    _identifier, _pascal, _require, _shape_symbols, _typed_constant,
)


@dataclass(frozen=True)
class RenderedTemporalHelper:
    """Typed source proposition and lexical arguments of a transparent helper.

    Consumer adapters use this association, not traversal-order suffixes. It
    supplies names only; any derived consumer facts still need Verus checking.
    Region helpers additionally take one offset per source-region dimension.
    """

    proposition: ann.Prop
    binders: tuple[str, ...]
    name: str


@dataclass(frozen=True)
class RenderedVerusExactEffectContract(RenderedVerusContract):
    """Pre(before, free); post(before, after, free); one execution, not two."""

    helpers: tuple[RenderedTemporalHelper, ...] = ()
    dtype_type: str = ""


class _TemporalRenderer(_ExecutionLayout):
    def __init__(self, verified: VerifiedExactEffectContract, symbol_prefix: str, *,
                 execution_prefix: str | None = None, dtype_parameters: tuple[str, ...] | None = None):
        data, plan = verified._validated()
        self.verified = verified
        self.raw = data["plan"]["execution"]
        self.parameters = {p["name"]: p for p in self.raw["parameters"]}
        self.constants = {k: _typed_constant(v) for k, v in self.raw["constants"].items()}
        self.goal, = [g for g in plan.prepared.goals if g.name == plan.goal_name]
        self.kernel = _identifier(self.raw["kernel"], "invalid kernel name")
        self.prefix = _identifier(symbol_prefix, "invalid generated symbol prefix")
        self.execution_prefix = _identifier(execution_prefix or symbol_prefix, "invalid execution prefix")
        self.cdiv_name = self.prefix + "_cdiv"
        self.side_type = _pascal(self.execution_prefix) + "Side"
        self.free_type = _pascal(self.prefix) + "Free"
        self.program_type = None
        self.tensor_types, self.scalar_kinds = {}, {}
        for name, parameter in self.parameters.items():
            _identifier(name, "invalid parameter name")
            typ = parameter["declared_type"]
            if typ["kind"] == "tensor":
                self.tensor_types[name] = parameter["type"]
                for dim in typ["shape"]:
                    for symbol in _shape_symbols(dim):
                        _identifier(symbol, "invalid shape symbol")
                        self.scalar_kinds[symbol] = "int"
            else:
                self.scalar_kinds[name] = typ["kind"]
        _require(not (self.scalar_kinds.keys() & self.tensor_types.keys()), "tensor/shape names collide")
        self.dtype_parameters = tuple(sorted(self.tensor_types)) if dtype_parameters is None else dtype_parameters
        _require(set(self.tensor_types) <= set(self.dtype_parameters), "execution omits dtype metadata")
        self.output_names = plan.written_tensors
        self.quantified_definitions = []
        self.region_definitions = []
        self.helpers = []
        self.analyzer_conditions = tuple(RenderedAnalyzerCondition(
            "external_obligation", label, f"{self.prefix}_external_obligation_{i}",
        ) for i, label in enumerate(data["plan"]["external_obligations"]))

    def _term(self, expr, bound):
        match expr:
            case ann.IntConst(value=value):
                return str(value)
            case ann.FreeVar(name=name):
                if name in bound:
                    return bound[name]
                if name in self.constants:
                    return self._constant(name)
                _require(name in self.scalar_kinds, "unbound temporal term")
                return f"before.{name}"
            case ann.DTypeOf(name=name):
                _require(name in self.dtype_parameters, "unbound dtype term")
                return f"before.__element_dtype_{name}"
            case ann.AnnIndex(base=ann.Before(name=name), indices=indices):
                return f"before.{name}" + "".join(f"[{self._term(i, bound)}]" for i in indices)
            case ann.AnnBinOp(op=op, lhs=a, rhs=b):
                left, right = self._term(a, bound), self._term(b, bound)
                if op == "cdiv":
                    return f"{self.cdiv_name}({left}, {right})"
                if op in {"min", "max"}:
                    return f"vstd::math::{op}({left}, {right})"
                _require(op in {"+", "-", "*", "//", "%"}, "unsupported temporal arithmetic")
                return f"({left} {'/' if op == '//' else op} {right})"
            case _:
                raise ValueError(f"unsupported temporal term {type(expr).__name__}")

    def _prop(self, prop, bound=None):
        bound = {} if bound is None else bound
        match prop:
            case ann.AnnComparison(op=op, lhs=a, rhs=b):
                _require(op in {"==", "<", "<=", ">", ">="}, "unsupported temporal comparison")
                return f"({self._term(a, bound)} {op} {self._term(b, bound)})"
            case ann.AnnAnd(args=args) | ann.AnnOr(args=args):
                join = " && " if isinstance(prop, ann.AnnAnd) else " || "
                return "(" + join.join(self._prop(p, bound) for p in args) + ")"
            case ann.AnnNot(body=body):
                return f"!({self._prop(body, bound)})"
            case ann.AnnImplies(antecedent=a, consequent=b):
                return f"({self._prop(a, bound)} ==> {self._prop(b, bound)})"
            case ann.ForAllConstraint(vars=names, body=body):
                scope = dict(bound)
                for name in names:
                    _require(name not in scope, "shadowed temporal binder")
                    scope[name] = f"_q{len(scope)}"
                value = self._prop(body, scope)
                helper = f"{self.prefix}_quantified_{len(self.quantified_definitions)}"
                self.helpers.append(RenderedTemporalHelper(prop, tuple(scope), helper))
                args = ", ".join(f"{v}: int" for v in scope.values())
                self.quantified_definitions.append(f"""
pub open spec fn {helper}(before: {self.side_type}, after: {self.side_type},
    free: {self.free_type}, {args}) -> bool {{
    {value}
}}
""")
                binders = ", ".join(f"{scope[n]}: int" for n in names)
                call = f"{helper}(before, after, free, {', '.join(scope.values())})"
                return f"(forall|{binders}| #[trigger] {call})"
            case ann.RegionEquiv(left=left, right=right, given=None):
                _require(isinstance(left.side, ann.After) and isinstance(right.side, ann.Before),
                         "temporal region must preserve after/before orientation")
                conditions, limits = [], []
                lhs, rhs = f"after.{left.side.name}", f"before.{right.side.name}"
                for i, (a, b) in enumerate(zip(left.slices, right.slices, strict=True)):
                    start, stop = self._term(a.start, bound), self._term(a.stop, bound)
                    src, end = self._term(b.start, bound), self._term(b.stop, bound)
                    dst_dim = self._shape_expr(self.tensor_types[left.side.name]["shape"][i], "before")
                    src_dim = self._shape_expr(self.tensor_types[right.side.name]["shape"][i], "before")
                    conditions += [f"0 <= {start} <= {stop} <= {dst_dim}",
                                   f"0 <= {src} <= {end} <= {src_dim}",
                                   f"({stop} - {start}) == ({end} - {src})"]
                    limits.append(f"0 <= _r{i} < ({stop} - {start})")
                    lhs += f"[{start} + _r{i}]"
                    rhs += f"[{src} + _r{i}]"
                binders = ", ".join(f"_r{i}: int" for i in range(len(limits)))
                helper = f"{self.prefix}_region_{len(self.region_definitions)}"
                self.helpers.append(RenderedTemporalHelper(prop, tuple(bound), helper))
                args = [*bound.values(), *(f"_r{i}" for i in range(len(limits)))]
                signature = ", ".join(f"{arg}: int" for arg in args)
                # A transparent cell predicate supplies a stable trigger even
                # when source offsets contain indirect metadata arithmetic.
                self.region_definitions.append(f"""
pub open spec fn {helper}(before: {self.side_type}, after: {self.side_type},
    free: {self.free_type}, {signature}) -> bool {{
    {' && '.join(limits)} ==> {lhs} == {rhs}
}}
""")
                conditions.append(f"(forall|{binders}| #[trigger] {helper}(before, after, free, {', '.join(args)}))")
                return "(" + " && ".join(conditions) + ")"
            case _:
                raise ValueError(f"unsupported temporal proposition {type(prop).__name__}")

    def render(self, *, include_execution=True):
        outputs = self.output_names
        fields = self._side_fields()
        pre = [p for name in self.tensor_types for p in self._shape_conditions("before", name)]
        pre += [self._prop(p) for p in self.goal.pre_conditions]
        post = [self._prop(p) for p in self.goal.post_conditions]
        conditions = ""
        for condition in self.analyzer_conditions:
            pre.append(f"{condition.predicate_name}(before, free)")
            conditions += f"pub uninterp spec fn {condition.predicate_name}(before: {self.side_type}, free: {self.free_type}) -> bool;\n"
        functions, definitions = [], []
        for name in outputs:
            function, definition = self._output_definition(name)
            functions.append(function)
            definitions.append(definition)
        execute = self.execution_prefix + "_execute"
        pre_name, post_name = self.prefix + "_raw_pre", self.prefix + "_raw_post"
        certificate = self.prefix + "_certificate"
        execution = ""
        if include_execution:
            field_defs = "\n".join(f"    pub {name}: {typ}," for name, typ in fields)
            initializers = "\n".join(f"        {name}: " +
                (f"{self.execution_prefix}_{name}_after(before)" if name in outputs else f"before.{name}") + ","
                for name, _ in fields)
            execution = f"""{self._dtype_definition()}pub struct {self.side_type} {{
{field_defs}
}}
{''.join(definitions)}
pub open spec fn {execute}(before: {self.side_type}) -> {self.side_type} {{
    {self.side_type} {{
{initializers}
    }}
}}
"""
        body = f"""// Exact temporal ContractIR lowering for {self.kernel}
// Proved exact-effect contract: {self.verified.digest}
// Whole Triton source: {self.raw['source_sha256']}

pub open spec fn {self.cdiv_name}(numerator: int, denominator: int) -> int {{
    (numerator + denominator - 1) / denominator
}}

{execution}
pub struct {self.free_type} {{}}
{''.join(self.quantified_definitions)}
{''.join(self.region_definitions)}
{conditions}
pub open spec fn {pre_name}(before: {self.side_type}, free: {self.free_type}) -> bool {{
    let after = before;
{self._conjunction(pre)}
}}

pub open spec fn {post_name}(before: {self.side_type}, after: {self.side_type}, free: {self.free_type}) -> bool {{
{self._conjunction(post)}
}}

// Trusted import of the proved copy/frame theorem, retaining all premises.
// This is about one execution's before/after states, not batch invariance.
#[verifier::external_body]
pub proof fn {certificate}(before: {self.side_type}, free: {self.free_type})
    requires {pre_name}(before, free),
    ensures {post_name}(before, {execute}(before), free),
{{
    unreachable!()
}}
"""
        return RenderedVerusExactEffectContract(
            kernel=self.kernel, raw_contract_digest=self.verified.digest, symbol_prefix=self.prefix,
            side_type=self.side_type, free_type=self.free_type, program_type=None,
            pre_name=pre_name, post_name=post_name, singleton_name=None,
            execute_name=execute, certificate_name=certificate,
            output_parameters=tuple(outputs), output_functions=tuple(functions),
            analyzer_conditions=self.analyzer_conditions, body=body,
            dtype_parameters=self.dtype_parameters,
            helpers=tuple(self.helpers),
            dtype_type=_pascal(self.execution_prefix) + "DType",
        )


def render_verified_exact_effect_to_verus(verified: VerifiedExactEffectContract, *, symbol_prefix: str | None = None):
    _require(isinstance(verified, VerifiedExactEffectContract), "expected verified exact-effect evidence")
    prefix = symbol_prefix or f"exact_effect_{verified.digest[:12]}"
    return _TemporalRenderer(verified, prefix).render()
