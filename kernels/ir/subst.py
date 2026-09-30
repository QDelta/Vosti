from . import *


def subst_expr(expr: Expr, map: dict[str, Expr]) -> Expr:
    if not map:
        return expr
    match expr:
        case Var(name=name):
            return map.get(name, expr)
        case IntLit() | FloatLit() | BoolLit():
            return expr
        case BinOp(op=op, lhs=lhs, rhs=rhs):
            return BinOp(op, subst_expr(lhs, map), subst_expr(rhs, map))
        case Min(args=args):
            return Min([subst_expr(arg, map) for arg in args])
        case Max(args=args):
            return Max([subst_expr(arg, map) for arg in args])
        case Zeros(shape=shape):
            return Zeros([subst_expr(dim, map) for dim in shape])
        case Full(shape=shape, value=value):
            return Full(
                shape=[subst_expr(dim, map) for dim in shape],
                value=subst_expr(value, map),
            )
        case Arange(start=start, stop=stop):
            return Arange(start=subst_expr(start, map), stop=subst_expr(stop, map))
        case Where(cond=cond, on_true=on_true, on_false=on_false):
            return Where(
                cond=subst_expr(cond, map),
                on_true=subst_expr(on_true, map),
                on_false=subst_expr(on_false, map),
            )
        case ReduceMax(value=value, axis=axis):
            return ReduceMax(value=subst_expr(value, map), axis=axis)
        case ReduceSum(value=value, axis=axis):
            return ReduceSum(value=subst_expr(value, map), axis=axis)
        case Exp2(value=value) | Sigmoid(value=value) | Rsqrt(value=value) | Log2(value=value) | Not(value=value):
            return type(expr)(value=subst_expr(value, map))
        case Cast(value=value, kind=kind, target=target):
            return Cast(value=subst_expr(value, map), kind=kind, target=target)
        case Maximum(lhs=lhs, rhs=rhs):
            return Maximum(lhs=subst_expr(lhs, map), rhs=subst_expr(rhs, map))
        case Unsqueeze(value=value, axis=axis):
            return Unsqueeze(value=subst_expr(value, map), axis=axis)
        case Squeeze(value=value, axis=axis):
            return Squeeze(value=subst_expr(value, map), axis=axis)
        case BroadcastTo(value=value, shape=shape):
            return BroadcastTo(
                value=subst_expr(value, map),
                shape=[subst_expr(dim, map) for dim in shape],
            )
        case Transpose(value=value, permutation=permutation):
            return Transpose(value=subst_expr(value, map), permutation=permutation)
        case TensorIndex(base=base, indices=indices):
            assert base.name not in map
            return TensorIndex(
                base=base,
                indices=[subst_expr(idx, map) for idx in indices],
            )
        case TensorView(base=base, region=region):
            assert base.name not in map
            return TensorView(
                base=base,
                region=[
                    Slice(
                        start=subst_expr(sl.start, map),
                        stop=subst_expr(sl.stop, map),
                    )
                    for sl in region
                ],
            )
        case MaskedLoad(base=base, region=region, mask=mask):
            assert base.name not in map
            return MaskedLoad(
                base=base,
                region=[
                    Slice(
                        start=subst_expr(sl.start, map),
                        stop=subst_expr(sl.stop, map),
                    )
                    for sl in region
                ],
                mask=[
                    Slice(
                        start=subst_expr(sl.start, map),
                        stop=subst_expr(sl.stop, map),
                    )
                    for sl in mask
                ],
            )
        case _:
            raise AssertionError(f"unhandled expr: {expr}")


def subst_type(typ: Type, map: dict[str, Expr]) -> Type:
    if not map:
        return typ
    match typ:
        case TensorType(elem_type=elem_type, dims=dims):
            return TensorType(
                elem_type=subst_type(elem_type, map),
                dims=[subst_expr(dim, map) for dim in dims],
            )
        case IntType() | FloatType() | BoolType():
            return typ
        case _:
            raise AssertionError(f"unhandled type: {typ}")


def subst_target(target: Var | TensorView, map: dict[str, Expr]) -> Var | TensorView:
    if not map:
        return target
    match target:
        case Var(name=name):
            assert name not in map
            return target
        case TensorView(base=base, region=region):
            assert base.name not in map
            return TensorView(
                base=base,
                region=[
                    Slice(
                        start=subst_expr(sl.start, map),
                        stop=subst_expr(sl.stop, map),
                    )
                    for sl in region
                ],
            )


def specialize_kernel_constants(
    kernel: Kernel, constants: dict[str, int | float | bool]
) -> Kernel:
    if not constants:
        return kernel

    global_names = {param.name for param in kernel.params}
    for param in kernel.params:
        global_names.update(collect_type_vars(param.type))
    unknown = set(constants) - global_names
    if unknown:
        raise ValueError(
            "specialize_kernel_constants: constants are not kernel parameters "
            f"or tensor dimensions: {sorted(unknown)}"
        )

    literal_map = {name: lit_(value) for name, value in constants.items()}

    def specialize_stmt(stmt: Stmt) -> Stmt:
        match stmt:
            case Assign(target=target, op=op, value=value):
                return Assign(
                    target=subst_target(target, literal_map),
                    op=op,
                    value=subst_expr(value, literal_map),
                )
            case For(var=var, iters=iters, body=body):
                return For(
                    var=var,
                    iters=Range(
                        start=subst_expr(iters.start, literal_map),
                        stop=subst_expr(iters.stop, literal_map),
                    ),
                    body=[specialize_stmt(item) for item in body],
                )
            case Let(var=var, value=value):
                # A let-bound name must not coincide with a specialized constant
                if var.name in literal_map:
                    raise ValueError(
                        f"specialize_kernel_constants: let-bound variable '{var.name}' conflicts with a specialized constant"
                    )
                return Let(var=var, value=subst_expr(value, literal_map))
            case If(cond=cond, then_body=then_body, else_body=else_body):
                return If(
                    cond=subst_expr(cond, literal_map),
                    then_body=[specialize_stmt(item) for item in then_body],
                    else_body=[specialize_stmt(item) for item in else_body],
                )
            case MaskedStore(base=base, region=region, value=value, mask=mask):
                assert base.name not in literal_map
                return MaskedStore(
                    base=base,
                    region=[
                        Slice(
                            start=subst_expr(sl.start, literal_map),
                            stop=subst_expr(sl.stop, literal_map),
                        )
                        for sl in region
                    ],
                    value=subst_expr(value, literal_map),
                    mask=[
                        Slice(
                            start=subst_expr(sl.start, literal_map),
                            stop=subst_expr(sl.stop, literal_map),
                        )
                        for sl in mask
                    ],
                )
            case _:
                raise AssertionError(f"unhandled stmt: {stmt}")

    rewritten_params = [
        Param(name=param.name, type=subst_type(param.type, literal_map))
        for param in kernel.params
        if param.name not in constants
    ]

    rewritten_grid = Grid(
        iters=[
            GridIter(
                var=iter_var.var,
                iters=Range(
                    start=subst_expr(iter_var.iters.start, literal_map),
                    stop=subst_expr(iter_var.iters.stop, literal_map),
                ),
            )
            for iter_var in kernel.grid.iters
        ],
        decls=[
            VarDecl(var=decl.var, type=subst_type(decl.type, literal_map))
            for decl in kernel.grid.decls
        ],
        body=[specialize_stmt(stmt) for stmt in kernel.grid.body],
    )

    return Kernel(name=kernel.name, params=rewritten_params, grid=rewritten_grid)


# ---------------------------------------------------------------------------
# Let-binding expansion pass
# ---------------------------------------------------------------------------


def expand_let_bindings(kernel: Kernel) -> Kernel:
    def expand_stmts(
        stmts: Sequence[Stmt], incoming: dict[str, Expr]
    ) -> Sequence[Stmt]:
        # Let bindings are lexical.  Each branch/loop receives the bindings
        # visible at entry, but bindings created inside it cannot rewrite its
        # sibling or following outer statements.
        let_map = dict(incoming)
        new_stmts: list[Stmt] = []
        for stmt in stmts:
            new_stmt = expand_stmt(stmt, let_map)
            if new_stmt is not None:
                new_stmts.append(new_stmt)
        return new_stmts

    def expand_stmt(stmt: Stmt, let_map: dict[str, Expr]) -> Stmt | None:
        match stmt:
            case Assign(target=target, op=op, value=value):
                return Assign(
                    target=subst_target(target, let_map),
                    op=op,
                    value=subst_expr(value, let_map),
                )
            case For(var=var, iters=iters, body=body):
                return For(
                    var=var,
                    iters=Range(
                        start=subst_expr(iters.start, let_map),
                        stop=subst_expr(iters.stop, let_map),
                    ),
                    body=expand_stmts(body, let_map),
                )
            case Let(var=var, value=value):
                value_vars = collect_expr_vars(value)
                assert var.name not in value_vars
                expanded_value = subst_expr(value, let_map)
                let_map[var.name] = expanded_value
                return None
            case If(cond=cond, then_body=then_body, else_body=else_body):
                return If(
                    cond=subst_expr(cond, let_map),
                    then_body=expand_stmts(then_body, let_map),
                    else_body=expand_stmts(else_body, let_map),
                )
            case MaskedStore(base=base, region=region, value=value, mask=mask):
                assert base.name not in let_map
                return MaskedStore(
                    base=base,
                    region=[
                        Slice(
                            start=subst_expr(sl.start, let_map),
                            stop=subst_expr(sl.stop, let_map),
                        )
                        for sl in region
                    ],
                    value=subst_expr(value, let_map),
                    mask=[
                        Slice(
                            start=subst_expr(sl.start, let_map),
                            stop=subst_expr(sl.stop, let_map),
                        )
                        for sl in mask
                    ],
                )
            case _:
                raise AssertionError(f"unhandled stmt: {stmt}")

    expanded_grid = Grid(
        iters=kernel.grid.iters,
        decls=kernel.grid.decls,
        body=expand_stmts(kernel.grid.body, {}),
    )

    return Kernel(name=kernel.name, params=kernel.params, grid=expanded_grid)
