"""Convert one parsed relational proof goal into an EquivProofConfig.

This module bridges the annotation AST (from annotations.py) and the
regional proof obligations. It builds Z3 environments and translates
annotation expressions into IR Expr nodes (for regions) and Z3 formulas
(for scalar constraints).
"""

from __future__ import annotations

from .scalar_values import FLOAT_VALUE, float_literal, is_float_value
from .names import fresh_name

from . import (
    Kernel,
    Expr,
    Var,
    IntLit,
    BinOp,
    Min,
    Max,
    Slice,
    TensorIndex,
    Region,
    Type,
    BoolType,
    FloatType,
    TensorType,
    collect_type_vars,
)
from .annotation_lowering import (
    AnnBoolExpr, Condition, GuardedRegionEquality, LoweredProofGoal, lower_proof_goal,
)

from .annotations import (
    AnnExpr,
    AnnBinOp,
    AnnIndex,
    AnnSlice,
    AnnComparison,
    AnnAnd,
    AnnImplies,
    Left,
    Right,
    FreeVar,
    IntConst,
    RegionRef,
    RegionEquiv,
    ForAllConstraint,
    SingletonSpec,
    RelationalProofGoal,
)
import z3

Z3Val = z3.ExprRef | z3.FuncDeclRef


# ---------------------------------------------------------------------------
# AnnExpr → IR Expr  (for building Region slices)
# ---------------------------------------------------------------------------


def ann_expr_to_ir(
    expr: AnnExpr, reference_names: dict[tuple[type, str], str] | None = None,
) -> Expr:
    """Convert an annotation expression to an IR Expr.

    Same-run references use the ordinary environment. Cross-run references
    receive aliases bound to the original side by build_config_from_annotation;
    they must not silently become same-run values during region reasoning.
    """
    match expr:
        case Left(name=name) | Right(name=name):
            return Var((reference_names or {}).get((type(expr), name), name))
        case FreeVar(name=name):
            return Var(name)
        case IntConst(value=value):
            return IntLit(value)
        case AnnBinOp(op=op, lhs=lhs, rhs=rhs):
            if op in {"min", "max"}:
                return (Min if op == "min" else Max)([
                    ann_expr_to_ir(lhs, reference_names), ann_expr_to_ir(rhs, reference_names),
                ])
            return BinOp(
                op, ann_expr_to_ir(lhs, reference_names), ann_expr_to_ir(rhs, reference_names),
            )
        case AnnIndex(base=base, indices=indices):
            return TensorIndex(
                base=ann_expr_to_ir(base, reference_names),
                indices=[ann_expr_to_ir(idx, reference_names) for idx in indices],
            )
        case _:
            raise ValueError(f"Cannot convert annotation expr to IR: {expr}")


# ---------------------------------------------------------------------------
# AnnExpr → Z3  (for scalar constraints)
# ---------------------------------------------------------------------------


def _z3_binop(op: str, l: z3.ExprRef, r: z3.ExprRef) -> z3.ArithRef:
    """Apply a binary arithmetic operator in Z3."""
    if not (z3.is_int(l) and z3.is_int(r)):
        raise ValueError("annotation arithmetic requires integer operands")
    if op == "+":
        return l + r  # type: ignore[return-value]
    if op == "-":
        return l - r  # type: ignore[return-value]
    if op == "*":
        return l * r  # type: ignore[return-value]
    if op == "%":
        return l % r  # type: ignore[return-value]
    if op == "//":
        return l / r  # type: ignore[return-value]
    if op == "cdiv":
        return (l + r - 1) / r  # type: ignore[return-value]
    if op in {"min", "max"}:
        return z3.If(l <= r if op == "min" else l >= r, l, r)
    raise ValueError(f"Unsupported binary op for Z3: {op}")


def _z3_compare(op: str, l: z3.ExprRef, r: z3.ExprRef) -> z3.BoolRef:
    """Apply a comparison operator in Z3."""
    if op == "==":
        return l == r  # type: ignore[return-value]
    if is_float_value(l) or is_float_value(r):
        raise ValueError("floating annotation values support equality, not numeric ordering")
    if op == ">":
        return l > r  # type: ignore[return-value]
    if op == ">=":
        return l >= r  # type: ignore[return-value]
    if op == "<":
        return l < r  # type: ignore[return-value]
    if op == "<=":
        return l <= r  # type: ignore[return-value]
    raise ValueError(f"Unsupported comparison op: {op}")


def ann_expr_to_z3(
    expr: AnnExpr,
    left_env: dict[str, Z3Val],
    right_env: dict[str, Z3Val],
    *,
    bound_env: dict[str, z3.ExprRef] | None = None,
) -> z3.ExprRef:
    """Convert an annotation expression to a Z3 formula.

    Left/Right references are resolved via the corresponding env.
    Bare names resolve lexical quantifier bindings before shared/free values.
    Explicit Left/Right references always denote launch values, never binders.
    """
    match expr:
        case Left(name=name):
            return left_env[name]  # type: ignore[return-value]
        case Right(name=name):
            return right_env[name]  # type: ignore[return-value]
        case FreeVar(name=name):
            if bound_env is not None and name in bound_env:
                return bound_env[name]
            return left_env[name]  # type: ignore[return-value]  # free vars are the same in both
        case IntConst(value=value):
            return z3.IntVal(value)
        case AnnBinOp(op=op, lhs=lhs, rhs=rhs):
            return _z3_binop(
                op, ann_expr_to_z3(lhs, left_env, right_env, bound_env=bound_env),
                ann_expr_to_z3(rhs, left_env, right_env, bound_env=bound_env),
            )
        case AnnIndex(base=base, indices=indices):
            func = left_env[base.name] if isinstance(base, Left) else right_env[base.name]
            assert isinstance(func, z3.FuncDeclRef), f"Expected Z3 Function for {base.name}, got {type(func)}"
            return func(*[
                ann_expr_to_z3(idx, left_env, right_env, bound_env=bound_env)
                for idx in indices
            ])  # type: ignore[return-value]
        case _:
            raise ValueError(f"Cannot convert annotation expr to Z3: {expr}")


def ann_bool_expr_to_z3(
    expr: AnnBoolExpr,
    left_env: dict[str, Z3Val],
    right_env: dict[str, Z3Val],
    *,
    bound_env: dict[str, z3.ExprRef] | None = None,
) -> z3.BoolRef:
    """Convert an annotation boolean expression to a Z3 boolean formula."""
    match expr:
        case AnnComparison(op=op, lhs=lhs, rhs=rhs):
            return _z3_compare(
                op, ann_expr_to_z3(lhs, left_env, right_env, bound_env=bound_env),
                ann_expr_to_z3(rhs, left_env, right_env, bound_env=bound_env),
            )
        case AnnAnd(args=args):
            return z3.And(*[ann_bool_expr_to_z3(a, left_env, right_env, bound_env=bound_env) for a in args])
        case AnnImplies(antecedent=ante, consequent=cons):
            return z3.Implies(
                ann_bool_expr_to_z3(ante, left_env, right_env, bound_env=bound_env),
                ann_bool_expr_to_z3(cons, left_env, right_env, bound_env=bound_env),
            )
        case _:
            raise ValueError(f"Cannot convert annotation bool expr to Z3: {expr}")


def ann_forall_to_z3(
    constraint: ForAllConstraint,
    left_env: dict[str, Z3Val],
    right_env: dict[str, Z3Val],
) -> z3.BoolRef:
    """Convert a ForAllConstraint to a Z3 ForAll expression.

    Quantifier bindings have their own namespace. Fresh Z3 symbols also avoid
    capturing a launch expression that happens to use the same solver name.
    """
    if not constraint.vars or len(set(constraint.vars)) != len(constraint.vars):
        raise ValueError("forall requires unique bound variables")
    bound_z3_vars = [z3.FreshInt(name) for name in constraint.vars]
    body = ann_bool_expr_to_z3(
        constraint.body, left_env, right_env,
        bound_env=dict(zip(constraint.vars, bound_z3_vars)),
    )
    return z3.ForAll(bound_z3_vars, body)


def _ann_expr_uses_right(expr: AnnExpr) -> bool:
    match expr:
        case Right():
            return True
        case Left() | FreeVar() | IntConst():
            return False
        case AnnBinOp(lhs=lhs, rhs=rhs):
            return _ann_expr_uses_right(lhs) or _ann_expr_uses_right(rhs)
        case AnnIndex(base=base, indices=indices):
            return isinstance(base, Right) or any(
                _ann_expr_uses_right(index) for index in indices
            )
        case _:
            raise ValueError(f"unsupported annotation expression: {expr!r}")


def _ann_bool_uses_right(expr: AnnBoolExpr) -> bool:
    match expr:
        case AnnComparison(lhs=lhs, rhs=rhs):
            return _ann_expr_uses_right(lhs) or _ann_expr_uses_right(rhs)
        case AnnAnd(args=args):
            return any(_ann_bool_uses_right(argument) for argument in args)
        case AnnImplies(antecedent=antecedent, consequent=consequent):
            return _ann_bool_uses_right(antecedent) or _ann_bool_uses_right(
                consequent
            )
        case _:
            raise ValueError(f"unsupported annotation Boolean: {expr!r}")


def build_left_launch_assumptions(
    annotation: RelationalProofGoal | LoweredProofGoal,
    left_env: dict[str, Z3Val],
    right_env: dict[str, Z3Val],
) -> list[z3.BoolRef]:
    """Project relational ``@pre`` clauses to one full/left launch.

    Right-only and cross-run facts cannot justify race freedom within the full
    launch. Region equalities are value relations and are likewise irrelevant.
    """

    annotation = lower_proof_goal(annotation)
    assumptions: list[z3.BoolRef] = []
    for condition in annotation.pre_conditions:
        if isinstance(condition, (RegionEquiv, GuardedRegionEquality)):
            continue
        if isinstance(condition, AnnComparison):
            if _ann_expr_uses_right(condition.lhs) or _ann_expr_uses_right(
                condition.rhs
            ):
                continue
            assumptions.append(
                _z3_compare(
                    condition.op,
                    ann_expr_to_z3(condition.lhs, left_env, right_env),
                    ann_expr_to_z3(condition.rhs, left_env, right_env),
                )
            )
            continue
        if isinstance(condition, ForAllConstraint):
            if not _ann_bool_uses_right(condition.body):
                assumptions.append(
                    ann_forall_to_z3(condition, left_env, right_env)
                )
            continue
        raise ValueError(f"unsupported launch precondition: {condition!r}")
    return assumptions


def _eval_given_expr(
    expr: AnnExpr,
    env: dict[str, Z3Val],
) -> z3.ArithRef | z3.BoolRef:
    """Evaluate a given-clause expression using a single environment.

    All names (FreeVar, Left, Right) resolve from the same env.
    This is used to evaluate the expr once with left_env and once with right_env.
    """
    match expr:
        case Left(name=name) | Right(name=name) | FreeVar(name=name):
            return env[name]  # type: ignore[return-value]
        case IntConst(value=value):
            return z3.IntVal(value)
        case AnnBinOp(op=op, lhs=lhs, rhs=rhs):
            return _z3_binop(op, _eval_given_expr(lhs, env), _eval_given_expr(rhs, env))
        case AnnIndex(base=base, indices=indices):
            func = env[base.name]
            assert isinstance(func, z3.FuncDeclRef)
            return func(*[_eval_given_expr(idx, env) for idx in indices])  # type: ignore[return-value]
        case _:
            raise ValueError(f"Cannot evaluate given expr: {expr}")


# ---------------------------------------------------------------------------
# Environment construction
# ---------------------------------------------------------------------------


def _collect_referenced_names(annotation: LoweredProofGoal) -> tuple[set[str], set[str], set[str]]:
    """Collect all Left, Right, and FreeVar names referenced in the annotation.

    Returns (left_names, right_names, free_names).
    """
    left_names: set[str] = set()
    right_names: set[str] = set()
    free_names: set[str] = set()

    def visit_expr(
        expr: AnnExpr,
        bound: frozenset[str] = frozenset(),
    ) -> None:
        match expr:
            case Left(name=name):
                left_names.add(name)
            case Right(name=name):
                right_names.add(name)
            case FreeVar(name=name):
                if name not in bound:
                    free_names.add(name)
            case AnnBinOp(lhs=lhs, rhs=rhs):
                visit_expr(lhs, bound)
                visit_expr(rhs, bound)
            case AnnIndex(base=base, indices=indices):
                if isinstance(base, Left):
                    left_names.add(base.name)
                else:
                    right_names.add(base.name)
                for idx in indices:
                    visit_expr(idx, bound)
            case IntConst():
                pass

    def visit_slice(s: AnnSlice, bound: frozenset[str] = frozenset()) -> None:
        visit_expr(s.start, bound)
        visit_expr(s.stop, bound)

    def visit_region(
        ref: RegionRef,
        bound: frozenset[str] = frozenset(),
    ) -> None:
        if isinstance(ref.side, Left):
            left_names.add(ref.side.name)
        else:
            right_names.add(ref.side.name)
        for s in ref.slices:
            visit_slice(s, bound)

    def visit_bool_expr(
        expr: AnnBoolExpr,
        bound: frozenset[str] = frozenset(),
    ) -> None:
        match expr:
            case AnnComparison(lhs=lhs, rhs=rhs):
                visit_expr(lhs, bound)
                visit_expr(rhs, bound)
            case AnnAnd(args=args):
                for a in args:
                    visit_bool_expr(a, bound)
            case AnnImplies(antecedent=ante, consequent=cons):
                visit_bool_expr(ante, bound)
                visit_bool_expr(cons, bound)

    def visit_condition(cond: Condition) -> None:
        match cond:
            case RegionEquiv(left=left, right=right, given=given):
                visit_region(left)
                visit_region(right)
                if given is not None:
                    visit_expr(given)
            case AnnComparison(lhs=lhs, rhs=rhs):
                visit_expr(lhs)
                visit_expr(rhs)
            case ForAllConstraint(vars=vars, body=body):
                visit_bool_expr(body, frozenset(vars))
            case GuardedRegionEquality(vars=vars, when=when, relation=relation):
                bound = frozenset(vars)
                visit_bool_expr(when, bound)
                visit_region(relation.left, bound)
                visit_region(relation.right, bound)
                if relation.given is not None:
                    visit_expr(relation.given, bound)

    for cond in annotation.pre_conditions:
        visit_condition(cond)
    for cond in annotation.post_conditions:
        visit_condition(cond)

    return left_names, right_names, free_names


def _find_equality_bindings(
    pre_conditions: list[Condition],
) -> tuple[dict[str, int], dict[str, int], set[str]]:
    """Extract environment bindings from equality constraints.

    Returns:
        left_const: {param_name: const_value} for left(P) == <int>
        right_const: {param_name: const_value} for right(P) == <int>
        shared: set of param names where left(P) == right(P)
    """
    left_const: dict[str, int] = {}
    right_const: dict[str, int] = {}
    shared: set[str] = set()

    for cond in pre_conditions:
        if not isinstance(cond, AnnComparison):
            continue
        if cond.op != "==":
            continue

        lhs, rhs = cond.lhs, cond.rhs

        # left(P) == <int>
        if isinstance(lhs, Left) and isinstance(rhs, IntConst):
            left_const[lhs.name] = rhs.value
        # right(P) == <int>
        elif isinstance(lhs, Right) and isinstance(rhs, IntConst):
            right_const[lhs.name] = rhs.value
        # <int> = left(P)
        elif isinstance(rhs, Left) and isinstance(lhs, IntConst):
            left_const[rhs.name] = lhs.value
        # <int> = right(P)
        elif isinstance(rhs, Right) and isinstance(lhs, IntConst):
            right_const[rhs.name] = lhs.value
        # left(P) == right(P) or right(P) == left(P)
        elif isinstance(lhs, Left) and isinstance(rhs, Right) and lhs.name == rhs.name:
            shared.add(lhs.name)
        elif isinstance(lhs, Right) and isinstance(rhs, Left) and lhs.name == rhs.name:
            shared.add(lhs.name)

    return left_const, right_const, shared


def _is_binding_constraint(cond: Condition) -> bool:
    """Check if a constraint is a pure binding (right(M) == 1, left(N) == right(N)).

    These are used for env construction, not as Z3 base assumptions.
    """
    if not isinstance(cond, AnnComparison):
        return False
    if cond.op != "==":
        return False
    lhs, rhs = cond.lhs, cond.rhs
    # right(P) == <int> or left(P) == <int>
    if isinstance(lhs, (Left, Right)) and isinstance(rhs, IntConst):
        return True
    if isinstance(rhs, (Left, Right)) and isinstance(lhs, IntConst):
        return True
    # left(P) == right(P) or right(P) == left(P)
    if isinstance(lhs, Left) and isinstance(rhs, Right) and lhs.name == rhs.name:
        return True
    if isinstance(lhs, Right) and isinstance(rhs, Left) and lhs.name == rhs.name:
        return True
    return False


def _is_tensor_param(kernel: Kernel, name: str) -> bool:
    """Check if a kernel parameter is a tensor (needs Z3 Function, not Int)."""
    for param in kernel.params:
        if param.name == name and isinstance(param.type, TensorType):
            return True
    return False


def _tensor_param_ndim(kernel: Kernel, name: str) -> int:
    """Get number of dimensions for a tensor parameter."""
    for param in kernel.params:
        if param.name == name and isinstance(param.type, TensorType):
            return len(param.type.dims)
    raise ValueError(f"No tensor param named {name}")


def _kernel_param_type(kernel: Kernel, name: str) -> Type | None:
    for param in kernel.params:
        if param.name == name:
            return param.type
    return None


def _z3_sort_for_type(typ: Type | None) -> z3.SortRef:
    if isinstance(typ, BoolType):
        return z3.BoolSort()
    if isinstance(typ, FloatType):
        return FLOAT_VALUE
    return z3.IntSort()


def _z3_value(value: int | float | bool, typ: Type | None = None):
    if isinstance(typ, BoolType) or isinstance(value, bool):
        return z3.BoolVal(bool(value))
    if isinstance(typ, FloatType) or isinstance(value, float):
        return float_literal(float(value))
    return z3.IntVal(int(value))


def build_config_from_annotation(
    kernel: Kernel,
    annotation: RelationalProofGoal | LoweredProofGoal,
    specialized_constants: dict[str, int | float | bool] | None = None,
    post_index: int = 0,
):
    """Build an EquivProofConfig from parsed annotations and a kernel.

    Steps:
    1. Identify all referenced parameter names
    2. Build left_env and right_env from binding constraints
    3. Extract base assumptions from non-binding scalar constraints
    4. Extract tensor assumptions from RegionEquiv in @pre
    5. Extract one output region from @post, selected by ``post_index``
    6. Build singleton annotations from @singleton

    Args:
        specialized_constants: Constexpr values (e.g. {"D": 64, "BLOCK_M": 128}).
            These become concrete Z3 IntVals in both envs for free vars.
    """
    # Regional proof structures are needed only when preparing a proof config.
    from .regional_obligations import (
        EquivProofConfig,
        TensorAssumption,
        SingletonAnnotation,
        extract_loop_iter_ranges,
    )

    annotation = lower_proof_goal(annotation)

    # Build singleton annotations from parsed @singleton specs
    singleton_annotations: dict[str, SingletonAnnotation] = {}
    if specialized_constants is None:
        specialized_constants = {}

    left_names, right_names, free_names = _collect_referenced_names(annotation)
    left_const, right_const, shared = _find_equality_bindings(annotation.pre_conditions)

    # @same vars are shared and appear as Left nodes in the AST
    shared |= annotation.same_vars
    # @same vars appear as Left references; also register them as right references
    # so they get processed as param_names below
    for name in annotation.same_vars:
        if name in left_names:
            right_names.add(name)

    # All param names referenced as left()/right() or declared shared.
    # A name appearing only in @same (not elsewhere in @pre/@post) must still
    # receive a shared environment binding; otherwise value-dependency checks
    # see an unbound parameter and, before those checks existed, the @same was
    # simply inert.
    # Every source input and symbolic tensor dimension also receives a Z3
    # binding, even when the theorem leaves it unconstrained.  The typed IR may
    # mention such a symbol in a loop bound, guard, or inferred dependency
    # region.  Omitting it makes generic analysis depend on irrelevant @pre
    # mentions and, worse, turns a missing theorem constraint into a KeyError
    # instead of an ordinary unprovable obligation.
    kernel_param_names = {p.name for p in kernel.params}
    symbolic_dimension_names = {
        name
        for parameter in kernel.params
        for name in collect_type_vars(parameter.type)
    }
    kernel_input_names = kernel_param_names | symbolic_dimension_names
    param_names = (
        left_names | right_names | annotation.same_vars | kernel_input_names
    )
    # Remove tensor names that appear in region equivs (they're tensor assumptions, not scalar params)
    tensor_assumption_names: set[str] = set()
    for cond in annotation.pre_conditions:
        if isinstance(cond, RegionEquiv):
            tensor_assumption_names.add(cond.left.side.name)
            tensor_assumption_names.add(cond.right.side.name)
        elif isinstance(cond, GuardedRegionEquality):
            tensor_assumption_names.add(cond.relation.left.side.name)
            tensor_assumption_names.add(cond.relation.right.side.name)
    for cond in annotation.post_conditions:
        tensor_assumption_names.add(cond.left.side.name)
        tensor_assumption_names.add(cond.right.side.name)

    # Names index source bindings, not solver identities. Independently declared
    # inputs must never alias through suffixes (M/right versus a parameter MR)
    # or generated witness names. Only explicit sharing reuses a solver term.
    left_env: dict[str, Z3Val] = {}
    right_env: dict[str, Z3Val] = {}

    # Process scalar kernel params (not tensors used in region equivs)
    for name in sorted(param_names):
        if name in tensor_assumption_names and not _is_tensor_param(kernel, name):
            continue

        if _is_tensor_param(kernel, name):
            # Tensor param referenced as scalar index (e.g., cu_seqlens_q in constraints)
            ndim = _tensor_param_ndim(kernel, name)
            tensor_type = _kernel_param_type(kernel, name)
            assert isinstance(tensor_type, TensorType)
            sorts = [z3.IntSort()] * ndim + [
                _z3_sort_for_type(tensor_type.elem_type)
            ]
            if name in shared:
                func = z3.FreshFunction(*sorts)
                left_env[name] = func
                right_env[name] = func
            else:
                left_env[name] = z3.FreshFunction(*sorts)
                right_env[name] = z3.FreshFunction(*sorts)
            continue

        # Regular scalar param (use concrete value if specialized away)
        param_type = _kernel_param_type(kernel, name)
        if name in specialized_constants:
            val = _z3_value(specialized_constants[name], param_type)
            left_env[name] = val
            right_env[name] = val
            continue
        if name in shared:
            var = z3.FreshConst(_z3_sort_for_type(param_type), prefix=name)
            left_env[name] = var
            right_env[name] = var
        else:
            if name in left_const:
                left_env[name] = _z3_value(left_const[name], param_type)
            elif name not in left_env:
                left_env[name] = z3.FreshConst(
                    _z3_sort_for_type(param_type), prefix=f"{name}_left"
                )

            if name in right_const:
                right_env[name] = _z3_value(right_const[name], param_type)
            elif name not in right_env:
                right_env[name] = z3.FreshConst(
                    _z3_sort_for_type(param_type), prefix=f"{name}_right"
                )

    # Reject bare kernel params not in @same or specialized_constants.
    # Such params get the SAME Z3 variable on both sides, which is unsound.
    # _collect_referenced_names already excludes each quantifier's own bound
    # occurrences. A binder elsewhere cannot authorize a bare launch parameter.
    bare_kernel_params = (
        free_names & kernel_input_names
        - set(specialized_constants.keys())
    )
    if bare_kernel_params:
        raise ValueError(
            f"Kernel parameter(s) {bare_kernel_params} appear bare (without "
            f"left()/right() wrapper) and are not in @same. Add them to "
            f"@same(...) or use explicit left(...)/right(...) references."
        )

    # Add free variables to both envs.
    # If a free var matches a specialized constant, use its concrete value.
    for name in sorted(free_names):
        if name not in left_env:
            if name in specialized_constants:
                val = _z3_value(specialized_constants[name])
                left_env[name] = val
                right_env[name] = val
            else:
                var = z3.FreshInt(name)
                left_env[name] = var
                right_env[name] = var

    # Add loop variables (grid iters + for loops)
    loop_ranges = extract_loop_iter_ranges(kernel)
    for loop_name in loop_ranges:
        if loop_name not in left_env:
            left_env[loop_name] = z3.FreshInt(f"{loop_name}_left")
        if loop_name not in right_env:
            right_env[loop_name] = z3.FreshInt(f"{loop_name}_right")

    # Region IR is evaluated in one run's environment. Retain any explicit
    # reference to the other run through a collision-free logical alias, whose
    # value/function is that original run's input in BOTH environments. This
    # preserves the annotation AST without adding a second expression IR.
    aliases: dict[tuple[type, str], str] = {}
    reserved = (
        set(left_env) | set(right_env) | {decl.var.name for decl in kernel.grid.decls}
    )

    def lower_coordinate(expr: AnnExpr, owner: type) -> Expr:
        references: dict[tuple[type, str], str] = {}

        def visit(value: AnnExpr) -> None:
            match value:
                case Left(name=name) | Right(name=name):
                    side = type(value)
                    if side is owner or name in shared:
                        return
                    # Dynamic iterator witnesses are instantiated separately
                    # by the relational fold proof, not frozen input aliases.
                    if name in loop_ranges and name not in kernel_input_names:
                        raise ValueError(
                            "cross-run iterator coordinates require a shared logical index"
                        )
                    key = (side, name)
                    if key not in aliases:
                        alias = fresh_name(
                            "__annotation_" + ("left_" if side is Left else "right_") + name,
                            reserved,
                        )
                        reserved.add(alias)
                        aliases[key] = alias
                        source_env = left_env if side is Left else right_env
                        left_env[alias] = right_env[alias] = source_env[name]
                    references[key] = aliases[key]
                case AnnBinOp(lhs=lhs, rhs=rhs):
                    visit(lhs)
                    visit(rhs)
                case AnnIndex(base=base, indices=indices):
                    visit(base)
                    for index in indices:
                        visit(index)
                case IntConst() | FreeVar():
                    pass
                case _:
                    raise ValueError(f"unsupported region coordinate: {value!r}")

        visit(expr)
        return ann_expr_to_ir(expr, references)

    def lower_region(ref: RegionRef) -> Region:
        return [
            Slice(lower_coordinate(s.start, type(ref.side)),
                  lower_coordinate(s.stop, type(ref.side)))
            for s in ref.slices
        ]

    for spec in annotation.singletons:
        singleton_annotations[spec.var] = SingletonAnnotation(
            left=lower_coordinate(spec.left, Left),
            right=lower_coordinate(spec.right, Right),
        )

    # --- Extract base assumptions ---
    base_assumptions: list[z3.BoolRef | bool] = []
    for cond in annotation.pre_conditions:
        if isinstance(cond, (RegionEquiv, GuardedRegionEquality)):
            continue  # tensor assumptions handled below
        if isinstance(cond, ForAllConstraint):
            base_assumptions.append(ann_forall_to_z3(cond, left_env, right_env))
            continue
        assert isinstance(cond, AnnComparison)
        # Binding equalities also remain proof assumptions.  Environment
        # substitution is only an optimization: deleting the original clauses
        # lets duplicate/conflicting bindings be overwritten, and lets a
        # side-specific constant disappear when combined with left(X)=right(X).
        # Keeping every clause makes the satisfiability gate detect both cases.
        lhs_z3 = ann_expr_to_z3(cond.lhs, left_env, right_env)
        rhs_z3 = ann_expr_to_z3(cond.rhs, left_env, right_env)
        base_assumptions.append(_z3_compare(cond.op, lhs_z3, rhs_z3))

    # --- Extract tensor assumptions from @pre ---
    tensor_assumptions: dict[str, TensorAssumption] = {}
    # Placeholders are shared by schema coordinate name within this config,
    # but must never alias a launch parameter with an internal-looking name.
    schema_bindings: dict[str, z3.ArithRef] = {}
    for cond in annotation.pre_conditions:
        schema_variables: tuple[str, ...] = ()
        schema_symbols: tuple[z3.ArithRef, ...] = ()
        schema_conditions: tuple[z3.BoolRef | bool, ...] = ()
        if isinstance(cond, GuardedRegionEquality):
            relation = cond.relation
            if not cond.vars or len(set(cond.vars)) != len(cond.vars):
                raise ValueError(
                    "guarded region equality requires unique bound variables"
                )
            collisions = set(cond.vars) & (
                set(left_env) | set(right_env)
            )
            if collisions:
                raise ValueError(
                    "guarded region equality bound variable shadows an existing name: "
                    f"{sorted(collisions)}"
                )
            schema_variables = tuple(cond.vars)
            for name in schema_variables:
                if name not in schema_bindings:
                    schema_bindings[name] = z3.FreshInt(f"__region_schema_{name}")
            schema_symbols = tuple(schema_bindings[name] for name in schema_variables)
            temporary_left = dict(left_env)
            temporary_right = dict(right_env)
            for name, symbol in zip(schema_variables, schema_symbols):
                temporary_left[name] = symbol
                temporary_right[name] = symbol
            schema_conditions = (
                ann_bool_expr_to_z3(cond.when, temporary_left, temporary_right),
            )
        elif isinstance(cond, RegionEquiv):
            relation = cond
            temporary_left = left_env
            temporary_right = right_env
        else:
            continue
        tensor_name = relation.left.side.name
        right_tensor_name = relation.right.side.name
        if tensor_name != right_tensor_name:
            raise ValueError(
                "@pre region equality must compare the same tensor on both "
                f"sides, got {tensor_name!r} and {right_tensor_name!r}"
            )
        if tensor_name in tensor_assumptions:
            raise ValueError(
                f"multiple @pre region equalities for {tensor_name!r} are "
                "not yet supported; combine them into one covering region"
            )
        left_region = lower_region(relation.left)
        right_region = lower_region(relation.right)
        # Handle optional "given expr" clause — the condition is that
        # the expr evaluates to the same value on both sides.
        # We evaluate the expr with left_env (free vars → left) and
        # right_env (free vars → right), then assert equality.
        conditions: tuple[z3.BoolRef | bool, ...] = ()
        if relation.given is not None:
            left_val = _eval_given_expr(relation.given, temporary_left)
            right_val = _eval_given_expr(relation.given, temporary_right)
            conditions = (left_val == right_val,)
        tensor_assumptions[tensor_name] = TensorAssumption(
            left_region=left_region,
            right_region=right_region,
            conditions=conditions,
            schema_variables=schema_variables,
            schema_symbols=schema_symbols,
            schema_conditions=schema_conditions,
        )

    # A tensor named in @same denotes whole-tensor value equality.  This is
    # intentionally stronger than a sliced @pre assumption, but it is useful
    # for immutable model weights shared by both executions.  Previously such
    # names were silently ignored because @same was only wired into the scalar
    # Z3 environment.
    for name in annotation.same_vars:
        if not _is_tensor_param(kernel, name):
            continue
        for param in kernel.params:
            if param.name == name:
                assert isinstance(param.type, TensorType)
                full_region = [Slice(IntLit(0), dim) for dim in param.type.dims]
                tensor_assumptions[name] = TensorAssumption(
                    left_region=full_region,
                    right_region=full_region,
                )
                break

    # --- Extract output region from @post ---
    if not annotation.post_conditions:
        raise ValueError("@post must contain at least one region equivalence")

    if post_index < 0 or post_index >= len(annotation.post_conditions):
        raise IndexError(
            f"post_index {post_index} outside 0..{len(annotation.post_conditions) - 1}"
        )
    post = annotation.post_conditions[post_index]
    if post.left.side.name != post.right.side.name:
        raise ValueError(
            "@post region equality must compare the same tensor on both "
            f"sides, got {post.left.side.name!r} and {post.right.side.name!r}"
        )
    output_tensor = post.left.side.name
    left_output_region = lower_region(post.left)
    right_output_region = lower_region(post.right)

    return EquivProofConfig(
        kernel=kernel,
        output_tensor=output_tensor,
        left_output_region=left_output_region,
        right_output_region=right_output_region,
        tensor_assumptions=tensor_assumptions,
        left_env=left_env,
        right_env=right_env,
        base_assumptions=base_assumptions,
        singleton_annotations=singleton_annotations,
    )
