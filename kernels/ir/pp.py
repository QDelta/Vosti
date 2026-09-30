from . import *


def pretty_kernel(kernel: Kernel) -> str:
    return pp_kernel(kernel)


def pp_kernel(kernel: Kernel) -> str:
    params = ", ".join(f"{p.name}: {pp_type(p.type)}" for p in kernel.params)
    header = f"def {kernel.name}({params}):"
    body = pp_grid(kernel.grid, indent=2)
    return "\n".join([header, body])


def pp_type(typ: Type) -> str:
    match typ:
        case IntType():
            return "int"
        case FloatType():
            return "float"
        case BoolType():
            return "bool"
        case TensorType(elem_type=elem_type, dims=dims):
            shape = ", ".join(pp_expr(d) for d in dims)
            return f"tensor<{pp_type(elem_type)}>[{shape}]"
        case _:
            raise AssertionError(f"unhandled type: {typ}")


def pp_expr(expr: Expr) -> str:
    match expr:
        case Var(name=name):
            return name
        case IntLit(value=value):
            return str(value)
        case FloatLit(value=value):
            return str(value)
        case BoolLit(value=value):
            return "true" if value else "false"
        case BinOp(op=op, lhs=lhs, rhs=rhs):
            return f"({pp_expr(lhs)} {op} {pp_expr(rhs)})"
        case Min(args=args):
            parts = ", ".join(pp_expr_no_paren(a) for a in args)
            return f"min({parts})"
        case Max(args=args):
            parts = ", ".join(pp_expr_no_paren(a) for a in args)
            return f"max({parts})"
        case Zeros(shape=shape):
            dims = ", ".join(pp_expr_no_paren(s) for s in shape)
            return f"zeros({dims})"
        case Full(shape=shape, value=value):
            dims = ", ".join(pp_expr_no_paren(s) for s in shape)
            return f"full({dims}, {pp_expr_no_paren(value)})"
        case Arange(start=start, stop=stop):
            return f"arange({pp_expr_no_paren(start)}, {pp_expr_no_paren(stop)})"
        case Where(cond=cond, on_true=on_true, on_false=on_false):
            return f"where({pp_expr_no_paren(cond)}, {pp_expr_no_paren(on_true)}, {pp_expr_no_paren(on_false)})"
        case ReduceMax(value=value, axis=axis):
            return f"max({pp_expr_no_paren(value)}, axis={axis})"
        case ReduceSum(value=value, axis=axis):
            return f"sum({pp_expr_no_paren(value)}, axis={axis})"
        case Exp2(value=value) | Sigmoid(value=value) | Rsqrt(value=value) | Log2(value=value) | Not(value=value):
            name = type(expr).__name__.lower()
            return f"{name}({pp_expr_no_paren(value)})"
        case Cast(value=value, kind=_, target=target):
            return f"cast({pp_expr_no_paren(value)}, {target})"
        case Maximum(lhs=lhs, rhs=rhs):
            return f"maximum({pp_expr_no_paren(lhs)}, {pp_expr_no_paren(rhs)})"
        case Unsqueeze(value=value, axis=axis):
            return f"unsqueeze({pp_expr_no_paren(value)}, {axis})"
        case Squeeze(value=value, axis=axis):
            return f"squeeze({pp_expr_no_paren(value)}, {axis})"
        case BroadcastTo(value=value, shape=shape):
            dims = ", ".join(pp_expr_no_paren(s) for s in shape)
            return f"broadcast_to({pp_expr_no_paren(value)}, ({dims}))"
        case Transpose(value=value, permutation=permutation):
            perm = ", ".join(str(p) for p in permutation)
            return f"transpose({pp_expr_no_paren(value)}, ({perm}))"
        case TensorView(base=base, region=region):
            rendered = ", ".join(pp_slice(sl) for sl in region)
            return f"{pp_expr(base)}[{rendered}]"
        case TensorIndex(base=base, indices=indices):
            rendered = ", ".join(pp_expr_no_paren(idx) for idx in indices)
            return f"{pp_expr(base)}[{rendered}]"
        case MaskedLoad(base=base, region=region, mask=mask):
            rgn = ", ".join(pp_slice(sl) for sl in region)
            msk = ", ".join(pp_slice(sl) for sl in mask)
            return f"masked_load({pp_expr(base)}[{rgn}], mask=[{msk}])"
        case _:
            raise AssertionError(f"unhandled expr: {expr}")


def pp_expr_no_paren(expr: Expr) -> str:
    if isinstance(expr, BinOp):
        return f"{pp_expr(expr.lhs)} {expr.op} {pp_expr(expr.rhs)}"
    else:
        return pp_expr(expr)


def pp_slice(slice: Slice) -> str:
    return f"{pp_expr(slice.start)}:{pp_expr(slice.stop)}"


def pp_stmt(stmt: Stmt, indent: int) -> str:
    pad = " " * indent
    match stmt:
        case Assign(target=target, op=op, value=value):
            op_str = f" {op}= " if op is not None else " = "
            return f"{pad}{pp_expr_no_paren(target)}{op_str}{pp_expr_no_paren(value)}"
        case For(var=var, iters=iters, body=body):
            header = f"{pad}for({pp_expr(var)} = {pp_range(iters)}):"
            body_str = "\n".join(pp_stmt(s, indent + 2) for s in body)
            return "\n".join([header, body_str])
        case If(cond=cond, then_body=then_body, else_body=else_body):
            header = f"{pad}if {pp_expr_no_paren(cond)}:"
            then_str = "\n".join(pp_stmt(s, indent + 2) for s in then_body)
            if else_body:
                else_header = f"{pad}else:"
                else_str = "\n".join(pp_stmt(s, indent + 2) for s in else_body)
                return "\n".join([header, then_str, else_header, else_str])
            return "\n".join([header, then_str])
        case Let(var=var, value=value):
            return f"{pad}let {pp_expr(var)} = {pp_expr_no_paren(value)}"
        case MaskedStore(base=base, region=region, value=value, mask=mask):
            rgn = ", ".join(pp_slice(sl) for sl in region)
            msk = ", ".join(pp_slice(sl) for sl in mask)
            return f"{pad}masked_store({pp_expr(base)}[{rgn}], mask=[{msk}]) = {pp_expr_no_paren(value)}"
        case _:
            raise AssertionError(f"unhandled stmt: {stmt}")


def pp_decl(decl: VarDecl, indent: int) -> str:
    pad = " " * indent
    return f"{pad}var {decl.var.name}: {pp_type(decl.type)}"


def pp_grid(grid: Grid, indent: int) -> str:
    pad = " " * indent
    iters = ", ".join(f"{pp_expr(i.var)} = {pp_range(i.iters)}" for i in grid.iters)
    header = f"{pad}grid({iters}):"
    decls = "\n".join(pp_decl(d, indent + 2) for d in grid.decls)
    body = "\n".join(pp_stmt(s, indent + 2) for s in grid.body)
    if decls:
        return "\n".join([header, decls, body])
    return "\n".join([header, body])


def pp_range(rng: Range) -> str:
    return f"{pp_expr(rng.start)}..{pp_expr(rng.stop)}"
