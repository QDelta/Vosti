"""Fail-closed exact-copy and framing analysis over the shared typed IR.

This is not a relational certificate and does not infer functional effects
from batch invariance. Each write is modeled at its reaching definition and
lane coordinate. Copy obligations require coverage and agreement of *every*
possible writer; framing requires absence of any writer. Floating arithmetic,
loop-carried values, mutable-input reads and unsupported IR fail closed.

Obligation construction is solver-free and reproducible. Successful reports
carry a source-bound artifact; reports themselves remain mutable diagnostics.
Physical dtype equality is an annotation premise, never inferred from FloatType.
"""
from dataclasses import dataclass, field, fields, is_dataclass
from typing import TYPE_CHECKING

import z3

from . import (Assign, BinOp, BoolType, Cast, FloatType, For, If, IntLit,
              IntType, MaskedLoad, MaskedStore, TensorIndex, TensorType,
              TensorView, Var, collect_type_vars)
from .annotations import (After, AnnAnd, AnnBinOp, AnnComparison, AnnImplies,
                          AnnIndex, AnnNot, AnnOr, Before, DTypeOf,
                          ForAllConstraint, FreeVar, IntConst, RegionEquiv)
from .proof_preparation import PreparedKernelProofs, prepare_kernel_proofs
from .smt import ProofCheck, z3_prove, z3_satisfiable
from .subst import subst_expr

if TYPE_CHECKING:
    from .exact_effect_artifact import VerifiedExactEffectContract


class UnsupportedEffect(ValueError):
    pass


@dataclass(frozen=True)
class CopyOrigin:
    tensor: str
    coordinates: tuple[z3.ArithRef, ...]
    valid: z3.BoolRef
    dtype: z3.ExprRef
    identity: z3.BoolRef


@dataclass(frozen=True)
class WriteEffect:
    tensor: str
    variables: tuple[z3.ArithRef, ...]
    guard: z3.BoolRef
    coordinates: tuple[z3.ArithRef, ...]
    origin: CopyOrigin


@dataclass(frozen=True)
class EffectClause:
    variables: tuple[z3.ArithRef, ...]
    guard: z3.BoolRef
    relation: RegionEquiv
    bindings: dict[str, z3.ArithRef]


@dataclass(frozen=True)
class EffectObligation:
    """One exact formula to check, not a successful solver receipt."""

    name: str
    assumptions: tuple[z3.BoolRef, ...]
    claim: z3.BoolRef | None  # None means require a satisfiable context.

    def solve(self, timeout: int) -> ProofCheck:
        if self.claim is None:
            return z3_satisfiable(self.name, self.assumptions, timeout)
        check = z3_prove(self.name, self.assumptions, self.claim, timeout)
        if self.name.startswith("copy_coverage_") and check.details == "unknown":
            # QE preserves the exists/div coverage formula; no assumed witness.
            reduced = z3.TryFor(z3.Then("simplify", "qe", "simplify"), timeout)(self.claim).as_expr()
            check = z3_prove(self.name, self.assumptions, reduced, timeout)
        return check


@dataclass(frozen=True)
class ExactEffectPlan:
    prepared: PreparedKernelProofs
    goal_name: str
    obligations: tuple[EffectObligation, ...]
    written_tensors: tuple[str, ...]


@dataclass
class ExactEffectReport:
    kernel_name: str
    goal_name: str
    checks: list[ProofCheck] = field(default_factory=list)
    written_tensors: tuple[str, ...] = ()
    unsupported_reason: str | None = None
    verified_contract: "VerifiedExactEffectContract | None" = None

    @property
    def proved(self) -> bool:
        return self.unsupported_reason is None and bool(self.checks) and all(c.proved for c in self.checks)


def _and(items):
    return z3.And(*items)


def _same(left, right):
    if len(left) != len(right):
        raise UnsupportedEffect("coordinate rank differs")
    return _and(a == b for a, b in zip(left, right))


class _Analysis:
    def __init__(self, prepared: PreparedKernelProofs, goal):
        self.prepared, self.goal = prepared, goal
        self.fresh_count = 0
        self.params = {p.name: p.type for p in prepared.kernel.params}
        self.env = {}
        for param in prepared.kernel.params:
            for name in collect_type_vars(param.type):
                self.env.setdefault(name, z3.Int(name))
            if isinstance(param.type, (IntType, BoolType)):
                self.env[param.name] = z3.Int(param.name) if isinstance(param.type, IntType) else z3.Bool(param.name)
            elif isinstance(param.type, TensorType):
                if isinstance(param.type.elem_type, IntType):
                    self.env[param.name] = z3.Function(param.name, *([z3.IntSort()] * (len(param.type.dims) + 1)))
            else:
                raise UnsupportedEffect("exact-copy analysis does not support floating scalar parameters")
        for name, value in prepared.constants:
            if type(value) is not int:
                raise UnsupportedEffect("exact-copy analysis requires integer specializations")
            self.env[name] = z3.IntVal(value)
        dtype_sort = z3.DeclareSort("ExactEffectDType")
        self.dtypes = {name: z3.Const("dtype:" + name, dtype_sort)
                       for name, typ in self.params.items() if isinstance(typ, TensorType)}
        self.read_bounds = []
        self.writes = []
        self.mutable = set()
        self.obligations = []
        self.assumptions = []
        self._inventory(prepared.kernel.grid.body)

    def fresh_int(self):
        # Source and annotation identifiers cannot contain '!'. Unlike
        # FreshInt, these names do not depend on process-global Z3 allocation.
        name = f"exact_effect!{self.fresh_count}"
        self.fresh_count += 1
        return z3.Int(name)

    def _inventory(self, statements):
        for statement in statements:
            match statement:
                case MaskedStore(base=base) | Assign(target=TensorView(base=base)):
                    if base.name not in self.params:
                        raise UnsupportedEffect("partial local-tensor writes are unsupported")
                    self.mutable.add(base.name)
                case Assign(target=Var(name=name)):
                    if name in self.params:
                        raise UnsupportedEffect("whole-parameter assignment is unsupported")
                case For(body=body):
                    self._inventory(body)
                case If(then_body=yes, else_body=no):
                    self._inventory(yes)
                    self._inventory(no)
                case _:
                    raise UnsupportedEffect(f"unsupported exact-effect statement {type(statement).__name__}")

    def tensor(self, name):
        typ = self.params.get(name)
        if not isinstance(typ, TensorType):
            raise UnsupportedEffect(f"not a tensor parameter: {name}")
        return typ

    def _integer(self, value):
        if not z3.is_int(value):
            raise UnsupportedEffect("expected integer term")
        return value

    def integer_operation(self, op, a, b):
        a, b = self._integer(a), self._integer(b)
        if op in {"//", "%", "cdiv"}:
            denominator = z3.simplify(b)
            if not z3.is_int_value(denominator) or denominator.as_long() <= 0:
                raise UnsupportedEffect("division requires a positive static denominator")
        operations = {"+": lambda: a + b, "-": lambda: a - b, "*": lambda: a * b,
                      "//": lambda: a / b, "%": lambda: a % b,
                      "cdiv": lambda: (a + b - 1) / b,
                      "min": lambda: z3.If(a < b, a, b), "max": lambda: z3.If(a > b, a, b)}
        if op not in operations:
            raise UnsupportedEffect(f"unsupported integer operation {op}")
        return operations[op]()

    def ann_expr(self, expr, bound, guard):
        match expr:
            case IntConst(value=value):
                return z3.IntVal(value)
            case FreeVar(name=name):
                value = bound.get(name, self.env.get(name))
                if value is None or isinstance(value, z3.FuncDeclRef):
                    raise UnsupportedEffect(f"unbound or non-scalar annotation name {name}")
                return self._integer(value)
            case DTypeOf(name=name):
                self.tensor(name)
                return self.dtypes[name]
            case AnnBinOp(op=op, lhs=a, rhs=b):
                return self.integer_operation(op, self.ann_expr(a, bound, guard), self.ann_expr(b, bound, guard))
            case AnnIndex(base=Before(name=name), indices=indices):
                typ = self.tensor(name)
                if not isinstance(typ.elem_type, IntType) or name in self.mutable:
                    raise UnsupportedEffect("annotation index requires immutable integer metadata")
                coords = tuple(self._integer(self.ann_expr(i, bound, guard)) for i in indices)
                self.read_bounds.append((guard, self.in_bounds(name, coords, self.env)))
                return self.env[name](*coords)
            case _:
                raise UnsupportedEffect(f"unsupported exact-effect term {type(expr).__name__}")

    def bind(self, names, bound):
        if not names or len(set(names)) != len(names) or any(n in bound or n in self.env or n in self.params for n in names):
            raise UnsupportedEffect("annotation binder shadows an existing name")
        variables = tuple(self.fresh_int() for _ in names)
        return variables, {**bound, **dict(zip(names, variables))}

    def scalar(self, prop, bound, guard):
        match prop:
            case AnnComparison(op=op, lhs=left, rhs=right):
                a, b = self.ann_expr(left, bound, guard), self.ann_expr(right, bound, guard)
                if a.sort() != b.sort():
                    raise UnsupportedEffect("comparison mixes dtype and integer terms")
                if op == "==":
                    return a == b
                self._integer(a); self._integer(b)
                if op not in {"<", "<=", ">", ">="}:
                    raise UnsupportedEffect("unsupported comparison")
                return {"<": lambda: a < b, "<=": lambda: a <= b, ">": lambda: a > b, ">=": lambda: a >= b}[op]()
            case AnnAnd(args=args) | AnnOr(args=args):
                if not args:
                    raise UnsupportedEffect("empty connective")
                values = []
                active = guard
                for p in args:
                    value = self.scalar(p, bound, active)
                    values.append(value)
                    # A conservative well-definedness order for indexed
                    # metadata terms; the returned logical connective is
                    # unchanged. Bounds must precede a guarded index use.
                    active = z3.And(active, value if isinstance(prop, AnnAnd) else z3.Not(value))
                return (z3.And if isinstance(prop, AnnAnd) else z3.Or)(*values)
            case AnnNot(body=body):
                return z3.Not(self.scalar(body, bound, guard))
            case AnnImplies(antecedent=a, consequent=b):
                antecedent = self.scalar(a, bound, guard)
                return z3.Implies(antecedent, self.scalar(b, bound, z3.And(guard, antecedent)))
            case ForAllConstraint(vars=names, body=body):
                variables, scope = self.bind(names, bound)
                return z3.ForAll(variables, self.scalar(body, scope, guard))
            case _:
                raise UnsupportedEffect("exact-effect guards and premises must be scalar propositions")

    def clauses(self, prop, bound=None, variables=(), guard=None):
        bound = {} if bound is None else bound
        guard = z3.BoolVal(True) if guard is None else guard
        match prop:
            case AnnAnd(args=args):
                if not args:
                    raise UnsupportedEffect("empty postcondition")
                return [c for p in args for c in self.clauses(p, bound, variables, guard)]
            case AnnImplies(antecedent=a, consequent=b):
                return self.clauses(b, bound, variables, z3.And(guard, self.scalar(a, bound, guard)))
            case ForAllConstraint(vars=names, body=body):
                fresh, scope = self.bind(names, bound)
                return self.clauses(body, scope, variables + fresh, guard)
            case RegionEquiv(left=left, right=right, given=None):
                if not isinstance(left.side, After) or not isinstance(right.side, Before):
                    raise UnsupportedEffect("exact effects require after(destination) == before(source)")
                return [EffectClause(variables, guard, prop, bound)]
            case _:
                raise UnsupportedEffect("unsupported exact-effect postcondition")

    def ir_expr(self, expr, env, guard):
        # The admitted integer cast is a source-validated int32 identity. All
        # other arithmetic here is integer address/control arithmetic only.
        match expr:
            case BinOp(op=op, lhs=a, rhs=b) if op in {"+", "-", "*", "//", "%", "cdiv"}:
                return self.integer_operation(op, self.ir_expr(a, env, guard), self.ir_expr(b, env, guard))
            case TensorIndex(base=base, indices=indices):
                typ = self.tensor(base.name)
                if base.name in self.mutable or not isinstance(typ.elem_type, IntType):
                    raise UnsupportedEffect("address/control reads must be immutable integer metadata")
                coords = tuple(self._integer(self.ir_expr(i, env, guard)) for i in indices)
                self.read_bounds.append((guard, self.in_bounds(base.name, coords, env)))
                return self.env[base.name](*coords)
            case Cast(value=value, kind="int32", target="tl.int32"):
                if not isinstance(value, TensorIndex) or not isinstance(value.type, IntType):
                    raise UnsupportedEffect("int32 cast is not a translated metadata identity")
                return self.ir_expr(value, env, guard)
            case Var(name=name):
                if name not in env or isinstance(env[name], z3.FuncDeclRef):
                    raise UnsupportedEffect("unbound scalar IR value")
                return env[name]
            case IntLit(value=value):
                return z3.IntVal(value)
            case BinOp(op=op, lhs=a, rhs=b) if op in {"<", "<=", ">", ">=", "==", "and", "or"}:
                a, b = self.ir_expr(a, env, guard), self.ir_expr(b, env, guard)
                return {"<": lambda: a < b, "<=": lambda: a <= b, ">": lambda: a > b, ">=": lambda: a >= b,
                        "==": lambda: a == b, "and": lambda: z3.And(a, b), "or": lambda: z3.Or(a, b)}[op]()
            case _:
                raise UnsupportedEffect(f"unsupported address/control expression {type(expr).__name__}")

    def in_bounds(self, tensor, coords, env):
        typ = self.tensor(tensor)
        if len(coords) != len(typ.dims):
            raise UnsupportedEffect("tensor index rank differs")
        return _and(z3.And(c >= 0, c < self.ir_expr(d, env, z3.BoolVal(True))) for c, d in zip(coords, typ.dims))

    def region(self, region, env, guard):
        return tuple((self.ir_expr(s.start, env, guard), self.ir_expr(s.stop, env, guard)) for s in region)

    def origin(self, expr, coordinates, env, guard):
        match expr:
            case MaskedLoad(base=base, region=region) | TensorView(base=base, region=region):
                if base.name in self.mutable:
                    raise UnsupportedEffect("mutable-state loads require ordered state reasoning")
                self.tensor(base.name)
                bounds = self.region(region, env, guard)
                if len(bounds) != len(coordinates):
                    raise UnsupportedEffect("copy source rank differs")
                source = tuple(lo + c for c, (lo, hi) in zip(coordinates, bounds))
                valid = _and(z3.And(c >= 0, c < hi - lo) for c, (lo, hi) in zip(coordinates, bounds))
                valid = z3.And(valid, self.in_bounds(base.name, source, env))
                if isinstance(expr, MaskedLoad):
                    masks = self.region(expr.mask, env, guard)
                    if len(masks) != len(source):
                        raise UnsupportedEffect("load mask rank differs")
                    valid = z3.And(valid, _and(z3.And(c >= lo, c < hi) for c, (lo, hi) in zip(source, masks)))
                return CopyOrigin(base.name, source, valid, self.dtypes[base.name], z3.BoolVal(True))
            case Cast(value=value, kind="float", target=target):
                if not target.endswith(".dtype.element_ty"):
                    raise UnsupportedEffect("floating conversion is not an element-dtype identity")
                target_name = target.removesuffix(".dtype.element_ty")
                if not isinstance(self.tensor(target_name).elem_type, FloatType):
                    raise UnsupportedEffect("floating cast target is not a floating tensor")
                origin = self.origin(value, coordinates, env, guard)
                if not isinstance(self.tensor(origin.tensor).elem_type, FloatType):
                    raise UnsupportedEffect("floating cast source is not a floating tensor")
                return CopyOrigin(origin.tensor, origin.coordinates, origin.valid, self.dtypes[target_name],
                                  z3.And(origin.identity, origin.dtype == self.dtypes[target_name]))
            case _:
                raise UnsupportedEffect(f"stored value is not a proved input copy: {type(expr).__name__}")

    def walk(self, statements, env, definitions, variables, guard):
        # Loop and branch bodies do not leak definitions. An entry read of a
        # local assigned inside that scope is rejected rather than treated as
        # its previous iteration value. Sequential assignments are snapshots.
        for statement in statements:
            match statement:
                case Assign(target=Var(name=name), value=value, op=None):
                    self.reject_mutable_reads(value)
                    definitions[name] = subst_expr(value, definitions)
                case For(var=var, iters=iters, body=body):
                    value = self.fresh_int()
                    lower, upper = self.ir_expr(iters.start, env, guard), self.ir_expr(iters.stop, env, guard)
                    assigned = self.assigned(body)
                    inner = {k: v for k, v in definitions.items() if k not in assigned}
                    self.walk(body, {**env, var.name: value}, inner, variables + (value,),
                              z3.And(guard, value >= lower, value < upper))
                    for name in assigned:
                        definitions.pop(name, None)
                case If(cond=condition, then_body=yes, else_body=no):
                    condition = self.ir_expr(subst_expr(condition, definitions), env, guard)
                    self.walk(yes, env, definitions.copy(), variables, z3.And(guard, condition))
                    self.walk(no, env, definitions.copy(), variables, z3.And(guard, z3.Not(condition)))
                    for name in self.assigned(yes) | self.assigned(no):
                        definitions.pop(name, None)
                case MaskedStore(base=base, region=region, value=value) | Assign(target=TensorView(base=base, region=region), value=value, op=None):
                    bounds = self.region(region, env, guard)
                    lanes = tuple(self.fresh_int() for _ in bounds)
                    coords = tuple(lo + c for c, (lo, hi) in zip(lanes, bounds))
                    active = z3.And(guard, _and(z3.And(c >= 0, c < hi - lo) for c, (lo, hi) in zip(lanes, bounds)))
                    if isinstance(statement, MaskedStore):
                        masks = self.region(statement.mask, env, guard)
                        if len(masks) != len(coords):
                            raise UnsupportedEffect("store mask rank differs")
                        active = z3.And(active, _and(z3.And(c >= lo, c < hi) for c, (lo, hi) in zip(coords, masks)))
                    self.check("store_bounds", [active], self.in_bounds(base.name, coords, env))
                    origin = self.origin(subst_expr(value, definitions), lanes, env, active)
                    # The implicit store conversion must preserve bits too.
                    origin = CopyOrigin(origin.tensor, origin.coordinates, origin.valid, origin.dtype,
                                        z3.And(origin.identity, origin.dtype == self.dtypes[base.name]))
                    self.writes.append(WriteEffect(base.name, variables + lanes, active, coords, origin))
                case _:
                    raise UnsupportedEffect("unsupported exact-effect write or update")

    def assigned(self, statements):
        result = set()
        for statement in statements:
            if isinstance(statement, Assign) and isinstance(statement.target, Var):
                result.add(statement.target.name)
            elif isinstance(statement, For):
                result |= self.assigned(statement.body)
            elif isinstance(statement, If):
                result |= self.assigned(statement.then_body) | self.assigned(statement.else_body)
        return result

    def reject_mutable_reads(self, value):
        if isinstance(value, (MaskedLoad, TensorView, TensorIndex)) and value.base.name in self.mutable:
            raise UnsupportedEffect("mutable-state loads require ordered state reasoning")
        if isinstance(value, (list, tuple)):
            for item in value:
                self.reject_mutable_reads(item)
        elif is_dataclass(value):
            for item in fields(value):
                if item.name != "type":
                    self.reject_mutable_reads(getattr(value, item.name))

    def check(self, name, assumptions, claim):
        label = f"{name}_{len(self.obligations)}"
        self.obligations.append(EffectObligation(label, tuple(self.assumptions + assumptions), claim))

    def satisfiable(self, name, assumptions):
        self.obligations.append(EffectObligation(name, tuple(self.assumptions + assumptions), None))

    def prove_clause(self, clause):
        relation = clause.relation
        dst, src = relation.left.side.name, relation.right.side.name
        dt, st = self.tensor(dst), self.tensor(src)
        if type(dt.elem_type) != type(st.elem_type):
            raise UnsupportedEffect("copy region element sorts differ")
        if len(relation.left.slices) != len(dt.dims) or len(relation.right.slices) != len(st.dims):
            raise UnsupportedEffect("postcondition region rank differs")
        left = tuple((self._integer(self.ann_expr(s.start, clause.bindings, clause.guard)),
                      self._integer(self.ann_expr(s.stop, clause.bindings, clause.guard))) for s in relation.left.slices)
        right = tuple((self._integer(self.ann_expr(s.start, clause.bindings, clause.guard)),
                       self._integer(self.ann_expr(s.stop, clause.bindings, clause.guard))) for s in relation.right.slices)
        if len(left) != len(right):
            raise UnsupportedEffect("postcondition coordinate rank differs")
        offsets = tuple(self.fresh_int() for _ in left)
        demand = z3.And(clause.guard, _and(z3.And(c >= 0, c < hi - lo) for c, (lo, hi) in zip(offsets, left)))
        target = tuple(lo + c for c, (lo, _) in zip(offsets, left))
        source = tuple(lo + c for c, (lo, _) in zip(offsets, right))
        self.check("region_extents", [clause.guard], _and(z3.And(a <= b, c <= d, b - a == d - c) for (a, b), (c, d) in zip(left, right)))
        # The exported shaped-region proposition includes endpoint bounds,
        # even for an empty region. Per-cell bounds alone would be vacuous in
        # that case and would not justify the stronger exported proposition.
        self.check("region_bounds", [clause.guard], _and(
            z3.And(lo >= 0, hi <= self.ir_expr(dim, self.env, clause.guard))
            for bounds, typ in ((left, dt), (right, st))
            for (lo, hi), dim in zip(bounds, typ.dims)
        ))
        self.check("post_bounds", [demand], z3.And(self.in_bounds(dst, target, self.env), self.in_bounds(src, source, self.env)))
        self.satisfiable(f"post_nonvacuous_{len(self.obligations)}", [demand])
        writes = [w for w in self.writes if w.tensor == dst]
        if dst == src:
            self.check("frame_coordinates", [demand], _same(target, source))
            for write in writes:
                self.check("frame_no_write", [demand, write.guard], z3.Not(_same(write.coordinates, target)))
        else:
            if src in self.mutable:
                raise UnsupportedEffect("copy postcondition references mutable source")
            coverage = []
            for write in writes:
                match = _same(write.coordinates, target)
                same_source = z3.BoolVal(write.origin.tensor == src)
                equality = z3.And(same_source, write.origin.valid, write.origin.identity,
                                  _same(write.origin.coordinates, source))
                self.check("all_writers_copy", [demand, write.guard, match], equality)
                coverage.append(z3.Exists(write.variables, z3.And(write.guard, match)))
            self.check("copy_coverage", [demand], z3.Or(*coverage))

    def prove_disjoint_writes(self):
        # Independent program instances cannot race, even if they would write
        # the same value. Be conservative about repeated sequential writes too.
        for i, left in enumerate(self.writes):
            for j, right in enumerate(self.writes):
                if j < i or left.tensor != right.tensor:
                    continue
                fresh = tuple(self.fresh_int() for _ in right.variables)
                replacements = tuple(zip(right.variables, fresh))
                guard = z3.substitute(right.guard, *replacements)
                coords = tuple(z3.substitute(c, *replacements) for c in right.coordinates)
                identity = _same(left.variables, fresh) if i == j else z3.BoolVal(False)
                self.check("write_disjoint", [left.guard, guard, _same(left.coordinates, coords)], identity)

    def run(self):
        if self.goal.same_vars or self.goal.singletons:
            raise UnsupportedEffect("exact effects do not use cross-execution sharing or singleton witnesses")
        if not self.goal.post_conditions:
            raise UnsupportedEffect("empty exact-effect goal")
        self.assumptions = [self.scalar(p, {}, z3.BoolVal(True)) for p in self.goal.pre_conditions]
        for name, typ in self.params.items():
            if isinstance(typ, TensorType):
                self.assumptions.extend(self.ir_expr(d, self.env, z3.BoolVal(True)) >= 0 for d in typ.dims)
        self.satisfiable("annotation_preconditions_sat", [])
        clauses = [c for p in self.goal.post_conditions for c in self.clauses(p)]
        env, variables, guard = dict(self.env), (), z3.BoolVal(True)
        for iterator in self.prepared.kernel.grid.iters:
            value = self.fresh_int()
            lower, upper = self.ir_expr(iterator.iters.start, env, guard), self.ir_expr(iterator.iters.stop, env, guard)
            guard = z3.And(guard, value >= lower, value < upper)
            env[iterator.var.name] = value
            variables += (value,)
        self.walk(self.prepared.kernel.grid.body, env, {}, variables, guard)
        if not self.writes:
            raise UnsupportedEffect("no supported parameter writes")
        self.prove_disjoint_writes()
        for clause in clauses:
            self.prove_clause(clause)
        for guard, bounds in self.read_bounds:
            self.check("metadata_read_bounds", [guard], bounds)


def prepare_exact_effect_plan(source: str, kernel_name: str, constants: dict, *, goal_name: str = "exact_effect") -> ExactEffectPlan:
    """Derive all obligations from source without querying any solver."""
    prepared = prepare_kernel_proofs(source, kernel_name, constants)
    return prepare_prepared_exact_effect_plan(prepared, goal_name=goal_name)


def prepare_prepared_exact_effect_plan(prepared: PreparedKernelProofs, *, goal_name: str = "exact_effect") -> ExactEffectPlan:
    """Use the same prepared typed execution as other named source goals."""
    goal, = [goal for goal in prepared.goals if goal.name == goal_name]
    analysis = _Analysis(prepared, goal)
    analysis.run()
    return ExactEffectPlan(prepared, goal_name, tuple(analysis.obligations), tuple(sorted(analysis.mutable)))


def verify_prepared_exact_effects(prepared: PreparedKernelProofs, *, goal_name: str = "exact_effect", timeout: int = 5000) -> ExactEffectReport:
    """Check exact-copy/frame propositions; never produce relational evidence."""
    report = ExactEffectReport(prepared.kernel.name, goal_name)
    try:
        plan = prepare_prepared_exact_effect_plan(prepared, goal_name=goal_name)
        report.checks = [obligation.solve(timeout) for obligation in plan.obligations]
        report.written_tensors = plan.written_tensors
        if report.proved:
            from .exact_effect_artifact import _issue_exact_effect_contract
            report.verified_contract = _issue_exact_effect_contract(plan, report.checks)
    except (UnsupportedEffect, ValueError, TypeError, AssertionError, z3.Z3Exception) as error:
        report.unsupported_reason = str(error)
    return report


def verify_exact_effects(source: str, kernel_name: str, constants: dict, *, goal_name: str = "exact_effect", timeout: int = 5000) -> ExactEffectReport:
    try:
        prepared = prepare_kernel_proofs(source, kernel_name, constants)
    except (ValueError, TypeError, AssertionError) as error:
        return ExactEffectReport(kernel_name, goal_name, unsupported_reason=str(error))
    return verify_prepared_exact_effects(prepared, goal_name=goal_name, timeout=timeout)
