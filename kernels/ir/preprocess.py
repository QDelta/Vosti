from typing import Literal

from . import *

# ---------------------------------------------------------------------------
# Kernel validity: variable-name checks (no-shadowing + scope)
# ---------------------------------------------------------------------------

VarRole = Literal["param", "param_size", "grid", "local_var", "loop_var", "let_bind"]


def _check_expr_scope(expr: Expr, scope: set[str], errors: list[str]) -> None:
    """Check that every Var in *expr* is in *scope*; append to *errors*."""
    for name in collect_expr_vars(expr):
        if name not in scope:
            errors.append(f"variable '{name}' used before declaration or out of scope")


def _check_type_scope(typ: Type, scope: set[str], errors: list[str]) -> None:
    """Check Var references inside a Type (e.g. tensor dim expressions)."""
    for name in collect_type_vars(typ):
        if name not in scope:
            errors.append(f"variable '{name}' used before declaration or out of scope")


def _check_target_scope(
    target: Var | TensorView, scope: set[str], errors: list[str]
) -> None:
    """Check Var references in an assignment target."""
    match target:
        case Var(name=name):
            if name not in scope:
                errors.append(
                    f"variable '{name}' used before declaration or out of scope"
                )
        case TensorView(base=base, region=region):
            if base.name not in scope:
                errors.append(
                    f"variable '{base.name}' used before declaration or out of scope"
                )
            for sl in region:
                _check_expr_scope(sl.start, scope, errors)
                _check_expr_scope(sl.stop, scope, errors)


def check_variable_names(kernel: Kernel) -> dict[str, VarRole]:
    """Validate variable names in *kernel*: no shadowing and proper scoping.

    Combines two checks in a single ordered walk:

    1. **No shadowing**: every binding-site name (params, size vars, grid
       iters, grid decls, For-loop vars, Let bindings) must be globally
       unique.
    2. **Scope**: every ``Var`` reference in an expression must refer to a
       name that has already been declared and is currently in scope.

    Scope rules:

    * Params and param-size vars are in scope everywhere.
    * Grid iter vars are in scope for grid decls and the grid body.
    * Grid decl vars are in scope for the grid body.
    * ``For`` loop vars are in scope only within their body.
    * ``Let`` bindings are in scope for subsequent statements at the
      same level (and deeper).

    Raises ``ValueError`` listing all problems if any are found.
    Returns the complete ``name → role`` map on success.
    """
    roles: dict[str, VarRole] = {}
    errors: list[str] = []

    def _register(name: str, role: VarRole) -> None:
        if name in roles:
            errors.append(
                f"'{name}': already declared as '{roles[name]}', now as '{role}'"
            )
        else:
            roles[name] = role

    # --- Phase 1: params and param-size vars (always in scope) -----------
    scope: set[str] = set()

    for p in kernel.params:
        _register(p.name, "param")
        scope.add(p.name)

    for p in kernel.params:
        for v in collect_type_vars(p.type):
            existing_role = roles.get(v)
            if existing_role is None:
                _register(v, "param_size")
                scope.add(v)
            elif existing_role == "param":
                errors.append(
                    f"'{v}': already declared as 'param', now as 'param_size'"
                )
            elif existing_role == "param_size":
                pass
            else:
                raise AssertionError(f"unexpected role: {existing_role}")

    # --- Phase 2: grid iter vars -----------------------------------------
    # Check iter range exprs against current scope, then add iter var.
    for it in kernel.grid.iters:
        _check_expr_scope(it.iters.start, scope, errors)
        _check_expr_scope(it.iters.stop, scope, errors)
        _register(it.var.name, "grid")
        scope.add(it.var.name)

    # --- Phase 3: grid decl vars ----------------------------------------
    # Check decl type exprs, then add decl var.
    for d in kernel.grid.decls:
        _check_type_scope(d.type, scope, errors)
        _register(d.var.name, "local_var")
        scope.add(d.var.name)

    # --- Phase 4: body walk (For/Let/Assign) in textual order ------------
    def _walk_body(stmts: Sequence[Stmt], scope: set[str]) -> None:
        for stmt in stmts:
            match stmt:
                case Let(var=var, value=value):
                    _check_expr_scope(value, scope, errors)
                    _register(var.name, "let_bind")
                    scope.add(var.name)
                case Assign(target=target, value=value):
                    _check_target_scope(target, scope, errors)
                    _check_expr_scope(value, scope, errors)
                case For(var=var, iters=iters, body=body):
                    _check_expr_scope(iters.start, scope, errors)
                    _check_expr_scope(iters.stop, scope, errors)
                    _register(var.name, "loop_var")
                    # For-loop var is only in scope inside the body
                    _walk_body(body, scope | {var.name})
                case If(cond=cond, then_body=then_body, else_body=else_body):
                    _check_expr_scope(cond, scope, errors)
                    # Bindings inside branches don't leak out
                    _walk_body(then_body, scope.copy())
                    _walk_body(else_body, scope.copy())
                case MaskedStore(base=base, region=region, value=value, mask=mask):
                    if base.name not in scope:
                        errors.append(
                            f"variable '{base.name}' used before declaration or out of scope"
                        )
                    for sl in region:
                        _check_expr_scope(sl.start, scope, errors)
                        _check_expr_scope(sl.stop, scope, errors)
                    for sl in mask:
                        _check_expr_scope(sl.start, scope, errors)
                        _check_expr_scope(sl.stop, scope, errors)
                    _check_expr_scope(value, scope, errors)
                case _:
                    raise AssertionError(f"unhandled stmt: {stmt}")

    _walk_body(kernel.grid.body, scope)

    if errors:
        raise ValueError(
            "check_variable_names: problems found:\n  " + "\n  ".join(errors)
        )
    return roles


def build_type_env(kernel: Kernel) -> dict[str, Type]:
    """Build a flat global type env for *kernel* in a single pass.

    Covers: kernel params, symbolic size variables (Var references inside
    param/decl type dims), Grid iter vars, Grid decl vars, and For loop vars
    (any nesting depth).  Let-binding names are intentionally excluded: they
    are added dynamically by ``_annotate_stmt`` as the body is annotated.

    Requires ``check_variable_names`` to have been called previously (the env
    is flat and correct only when all names are globally unique).
    """
    env: dict[str, Type] = {}

    # Kernel params
    for p in kernel.params:
        env[p.name] = p.type

    # Symbolic size vars from param and decl type dims
    for p in kernel.params:
        for v in collect_type_vars(p.type):
            if v not in env:
                env[v] = IntType()
    for d in kernel.grid.decls:
        for v in collect_type_vars(d.type):
            if v not in env:
                env[v] = IntType()

    # Grid iter vars
    for it in kernel.grid.iters:
        env[it.var.name] = IntType()

    # Grid decl vars
    for d in kernel.grid.decls:
        env[d.var.name] = d.type

    # For loop vars at any nesting depth
    def _add_for_vars(stmts: Sequence[Stmt]) -> None:
        for stmt in stmts:
            match stmt:
                case For(var=var, body=body):
                    env[var.name] = IntType()
                    _add_for_vars(body)
                case If(then_body=then_body, else_body=else_body):
                    _add_for_vars(then_body)
                    _add_for_vars(else_body)
                case Assign() | Let():
                    pass
                case MaskedStore():
                    pass
                case _:
                    raise AssertionError(f"unhandled stmt: {stmt}")

    _add_for_vars(kernel.grid.body)
    return env


def collect_readonly_tensors(kernel: Kernel) -> set[str]:
    """Return the set of tensor-typed names that are never written.

    Covers both kernel params and grid decl local vars.  A tensor is
    considered "written" if it appears as the target of any ``Assign``
    statement (either as a plain ``Var`` or as the base of a
    ``TensorView``).
    """
    type_env = build_type_env(kernel)

    # All tensor-typed names
    all_tensors: set[str] = set()
    for name, typ in type_env.items():
        if isinstance(typ, TensorType):
            all_tensors.add(name)

    # Collect written tensor names
    written: set[str] = set()

    def _walk(stmts: Sequence[Stmt]) -> None:
        for stmt in stmts:
            match stmt:
                case Assign(target=target):
                    match target:
                        case Var(name=name):
                            if name in all_tensors:
                                written.add(name)
                        case TensorView(base=base):
                            written.add(base.name)
                case For(body=body):
                    _walk(body)
                case If(then_body=then_body, else_body=else_body):
                    _walk(then_body)
                    _walk(else_body)
                case Let():
                    pass
                case MaskedStore(base=base):
                    written.add(base.name)
                case _:
                    raise AssertionError(f"unhandled stmt: {stmt}")

    _walk(kernel.grid.body)
    return all_tensors - written


def check_tensorindex_readonly(kernel: Kernel) -> None:
    """Assert every TensorIndex base is a read-only tensor.

    Walks all expressions in the kernel body.  For each ``TensorIndex``
    node found, verifies that its ``base`` tensor is never written.
    This is required for soundly modeling tensor accesses as
    uninterpreted functions in verification.

    Raises ``ValueError`` listing all violations.
    """
    readonly = collect_readonly_tensors(kernel)
    errors: list[str] = []

    def _check_expr(expr: Expr) -> None:
        match expr:
            case Var() | IntLit() | FloatLit() | BoolLit():
                pass
            case BinOp(lhs=lhs, rhs=rhs) | Maximum(lhs=lhs, rhs=rhs):
                _check_expr(lhs)
                _check_expr(rhs)
            case Min(args=args) | Max(args=args):
                for a in args:
                    _check_expr(a)
            case Zeros(shape=shape):
                for s in shape:
                    _check_expr(s)
            case Full(shape=shape, value=value):
                for s in shape:
                    _check_expr(s)
                _check_expr(value)
            case Arange(start=start, stop=stop):
                _check_expr(start)
                _check_expr(stop)
            case Where(cond=cond, on_true=on_true, on_false=on_false):
                _check_expr(cond)
                _check_expr(on_true)
                _check_expr(on_false)
            case ReduceMax(value=v) | ReduceSum(value=v) | Exp2(value=v) | Sigmoid(value=v) | Rsqrt(value=v) | Log2(value=v) | Cast(value=v) | Not(value=v):
                _check_expr(v)
            case Unsqueeze(value=value) | Squeeze(value=value):
                _check_expr(value)
            case BroadcastTo(value=value, shape=shape):
                _check_expr(value)
                for s in shape:
                    _check_expr(s)
            case Transpose(value=value):
                _check_expr(value)
            case TensorView(base=base, region=region):
                for sl in region:
                    _check_expr(sl.start)
                    _check_expr(sl.stop)
            case TensorIndex(base=base, indices=indices):
                if base.name not in readonly:
                    errors.append(f"TensorIndex base '{base.name}' is not read-only")
                for idx in indices:
                    _check_expr(idx)
            case MaskedLoad(base=base, region=region, mask=mask):
                for sl in region:
                    _check_expr(sl.start)
                    _check_expr(sl.stop)
                for sl in mask:
                    _check_expr(sl.start)
                    _check_expr(sl.stop)
            case _:
                raise AssertionError(f"unhandled expr: {expr}")

    def _check_stmts(stmts: Sequence[Stmt]) -> None:
        for stmt in stmts:
            match stmt:
                case Assign(target=target, value=value):
                    match target:
                        case Var():
                            pass
                        case TensorView(region=region):
                            for sl in region:
                                _check_expr(sl.start)
                                _check_expr(sl.stop)
                    _check_expr(value)
                case For(iters=iters, body=body):
                    _check_expr(iters.start)
                    _check_expr(iters.stop)
                    _check_stmts(body)
                case If(cond=cond, then_body=then_body, else_body=else_body):
                    _check_expr(cond)
                    _check_stmts(then_body)
                    _check_stmts(else_body)
                case Let(value=value):
                    _check_expr(value)
                case MaskedStore(region=region, value=value, mask=mask):
                    for sl in region:
                        _check_expr(sl.start)
                        _check_expr(sl.stop)
                    for sl in mask:
                        _check_expr(sl.start)
                        _check_expr(sl.stop)
                    _check_expr(value)
                case _:
                    raise AssertionError(f"unhandled stmt: {stmt}")

    _check_stmts(kernel.grid.body)

    if errors:
        raise ValueError(
            "check_tensorindex_readonly: violations found:\n  " + "\n  ".join(errors)
        )
