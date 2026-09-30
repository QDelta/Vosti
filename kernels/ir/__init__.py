from dataclasses import dataclass, field
from collections.abc import Sequence


def _check_instance(name: str, value: object, typ: type | tuple[type, ...]) -> None:
    if not isinstance(value, typ):
        raise TypeError(f"{name} must be {typ}, got {type(value)}")


def _check_sequence(
    name: str, value: object, elem_type: type | tuple[type, ...]
) -> None:
    if not isinstance(value, Sequence):
        raise TypeError(f"{name} must be sequence, got {type(value)}")
    for idx, elem in enumerate(value):
        if not isinstance(elem, elem_type):
            raise TypeError(f"{name}[{idx}] must be {elem_type}, got {type(elem)}")


@dataclass(frozen=True)
class Type:
    pass


@dataclass(frozen=True)
class IntType(Type):
    pass


@dataclass(frozen=True)
class FloatType(Type):
    pass


@dataclass(frozen=True)
class BoolType(Type):
    pass


@dataclass(frozen=True)
class TensorType(Type):
    elem_type: Type
    dims: Sequence["Expr"]

    def __post_init__(self) -> None:
        _check_instance("elem_type", self.elem_type, Type)
        _check_sequence("dims", self.dims, Expr)


@dataclass(frozen=True)
class Expr:
    type: Type | None = field(default=None, kw_only=True, compare=False)


@dataclass(frozen=True)
class Var(Expr):
    name: str

    def __post_init__(self) -> None:
        _check_instance("name", self.name, str)


@dataclass(frozen=True)
class IntLit(Expr):
    value: int

    def __post_init__(self) -> None:
        _check_instance("value", self.value, int)


@dataclass(frozen=True)
class FloatLit(Expr):
    value: float

    def __post_init__(self) -> None:
        _check_instance("value", self.value, float)


@dataclass(frozen=True)
class BoolLit(Expr):
    value: bool

    def __post_init__(self) -> None:
        _check_instance("value", self.value, bool)


@dataclass(frozen=True)
class BinOp(Expr):
    op: str
    lhs: Expr
    rhs: Expr

    def __post_init__(self) -> None:
        _check_instance("op", self.op, str)
        _check_instance("lhs", self.lhs, Expr)
        _check_instance("rhs", self.rhs, Expr)


@dataclass(frozen=True)
class Min(Expr):
    args: Sequence[Expr]

    def __post_init__(self) -> None:
        _check_sequence("args", self.args, Expr)


@dataclass(frozen=True)
class Max(Expr):
    args: Sequence[Expr]

    def __post_init__(self) -> None:
        _check_sequence("args", self.args, Expr)


@dataclass(frozen=True)
class Zeros(Expr):
    shape: Sequence[Expr]

    def __post_init__(self) -> None:
        _check_sequence("shape", self.shape, Expr)


@dataclass(frozen=True)
class Full(Expr):
    shape: Sequence[Expr]
    value: Expr

    def __post_init__(self) -> None:
        _check_sequence("shape", self.shape, Expr)
        _check_instance("value", self.value, Expr)


@dataclass(frozen=True)
class Arange(Expr):
    start: Expr
    stop: Expr

    def __post_init__(self) -> None:
        _check_instance("start", self.start, Expr)
        _check_instance("stop", self.stop, Expr)


@dataclass(frozen=True)
class Where(Expr):
    cond: Expr
    on_true: Expr
    on_false: Expr

    def __post_init__(self) -> None:
        _check_instance("cond", self.cond, Expr)
        _check_instance("on_true", self.on_true, Expr)
        _check_instance("on_false", self.on_false, Expr)


@dataclass(frozen=True)
class ReduceMax(Expr):
    value: Expr
    axis: int

    def __post_init__(self) -> None:
        _check_instance("value", self.value, Expr)
        _check_instance("axis", self.axis, int)


@dataclass(frozen=True)
class ReduceSum(Expr):
    value: Expr
    axis: int

    def __post_init__(self) -> None:
        _check_instance("value", self.value, Expr)
        _check_instance("axis", self.axis, int)


@dataclass(frozen=True)
class Exp2(Expr):
    value: Expr

    def __post_init__(self) -> None:
        _check_instance("value", self.value, Expr)


@dataclass(frozen=True)
class Sigmoid(Expr):
    value: Expr

    def __post_init__(self) -> None:
        _check_instance("value", self.value, Expr)


@dataclass(frozen=True)
class Rsqrt(Expr):
    value: Expr

    def __post_init__(self) -> None:
        _check_instance("value", self.value, Expr)


@dataclass(frozen=True)
class Log2(Expr):
    """A Triton ``tl.log2`` operation.

    The structural verifier uses only pointwise congruence for this node.  Its
    concrete floating-point behavior is a backend qualification obligation,
    not a numerical theorem of this IR.
    """

    value: Expr

    def __post_init__(self) -> None:
        _check_instance("value", self.value, Expr)


@dataclass(frozen=True)
class Cast(Expr):
    """A dependency-preserving Triton data cast with an explicit target.

    Keeping the cast in IR makes the verifier's congruence assumption visible
    to deployment-time backend qualification.  Address-changing integer casts
    remain rejected by the frontend.
    """

    value: Expr
    kind: str
    target: str

    def __post_init__(self) -> None:
        _check_instance("value", self.value, Expr)
        _check_instance("kind", self.kind, str)
        _check_instance("target", self.target, str)


@dataclass(frozen=True)
class Not(Expr):
    value: Expr

    def __post_init__(self) -> None:
        _check_instance("value", self.value, Expr)


@dataclass(frozen=True)
class Maximum(Expr):
    lhs: Expr
    rhs: Expr

    def __post_init__(self) -> None:
        _check_instance("lhs", self.lhs, Expr)
        _check_instance("rhs", self.rhs, Expr)


@dataclass(frozen=True)
class Unsqueeze(Expr):
    """Insert a size-1 dimension at `axis`."""

    value: Expr
    axis: int

    def __post_init__(self) -> None:
        _check_instance("value", self.value, Expr)


@dataclass(frozen=True)
class Squeeze(Expr):
    """Remove the dimension at `axis` (must be size 1)."""

    value: Expr
    axis: int

    def __post_init__(self) -> None:
        _check_instance("value", self.value, Expr)


@dataclass(frozen=True)
class BroadcastTo(Expr):
    value: Expr
    shape: Sequence[Expr]

    def __post_init__(self) -> None:
        _check_instance("value", self.value, Expr)
        _check_sequence("shape", self.shape, Expr)


@dataclass(frozen=True)
class Transpose(Expr):
    value: Expr
    permutation: Sequence[int]

    def __post_init__(self) -> None:
        _check_instance("value", self.value, Expr)
        _check_sequence("permutation", self.permutation, int)


@dataclass(frozen=True)
class Slice:
    start: Expr
    stop: Expr

    def __post_init__(self) -> None:
        _check_instance("start", self.start, Expr)
        _check_instance("stop", self.stop, Expr)


Region = Sequence[Slice]


@dataclass(frozen=True)
class TensorIndex(Expr):
    base: Var
    indices: Sequence[Expr]

    def __post_init__(self) -> None:
        _check_instance("base", self.base, Var)
        _check_sequence("indices", self.indices, Expr)


@dataclass(frozen=True)
class TensorView(Expr):
    base: Var
    region: Region

    def __post_init__(self) -> None:
        _check_instance("base", self.base, Var)
        _check_sequence("region", self.region, Slice)


@dataclass(frozen=True)
class MaskedLoad(Expr):
    """Load a region from a tensor param, zero-padding positions outside mask."""

    base: Var
    region: Region
    mask: Region

    def __post_init__(self) -> None:
        _check_instance("base", self.base, Var)
        _check_sequence("region", self.region, Slice)
        _check_sequence("mask", self.mask, Slice)


@dataclass(frozen=True)
class Stmt:
    pass


@dataclass(frozen=True)
class VarDecl:
    var: Var
    type: Type

    def __post_init__(self) -> None:
        _check_instance("var", self.var, Var)
        _check_instance("type", self.type, Type)


@dataclass(frozen=True)
class Let(Stmt):
    var: Var
    value: Expr

    def __post_init__(self) -> None:
        _check_instance("var", self.var, Var)
        _check_instance("value", self.value, Expr)


@dataclass(frozen=True)
class Assign(Stmt):
    target: Var | TensorView
    op: str | None
    value: Expr

    def __post_init__(self) -> None:
        _check_instance("target", self.target, (Var, TensorView))
        if self.op is not None:
            _check_instance("op", self.op, str)
        _check_instance("value", self.value, Expr)


@dataclass(frozen=True)
class MaskedStore(Stmt):
    """Store to a region of a tensor param, skipping positions outside mask."""

    base: Var
    region: Region
    value: Expr
    mask: Region

    def __post_init__(self) -> None:
        _check_instance("base", self.base, Var)
        _check_sequence("region", self.region, Slice)
        _check_instance("value", self.value, Expr)
        _check_sequence("mask", self.mask, Slice)


@dataclass(frozen=True)
class Range:
    start: Expr
    stop: Expr

    def __post_init__(self) -> None:
        _check_instance("start", self.start, Expr)
        _check_instance("stop", self.stop, Expr)


@dataclass(frozen=True)
class For(Stmt):
    var: Var
    iters: Range
    body: Sequence[Stmt]

    def __post_init__(self) -> None:
        _check_instance("var", self.var, Var)
        _check_instance("iters", self.iters, Range)
        _check_sequence("body", self.body, Stmt)


@dataclass(frozen=True)
class If(Stmt):
    cond: Expr
    then_body: Sequence[Stmt]
    else_body: Sequence[Stmt]

    def __post_init__(self) -> None:
        _check_instance("cond", self.cond, Expr)
        _check_sequence("then_body", self.then_body, Stmt)
        _check_sequence("else_body", self.else_body, Stmt)


@dataclass(frozen=True)
class Grid:
    iters: Sequence["GridIter"]
    decls: Sequence[VarDecl]
    body: Sequence[Stmt]

    def __post_init__(self) -> None:
        _check_sequence("iters", self.iters, GridIter)
        _check_sequence("decls", self.decls, VarDecl)
        _check_sequence("body", self.body, Stmt)


@dataclass(frozen=True)
class GridIter:
    var: Var
    iters: Range

    def __post_init__(self) -> None:
        _check_instance("var", self.var, Var)
        _check_instance("iters", self.iters, Range)


@dataclass(frozen=True)
class Param:
    name: str
    type: Type

    def __post_init__(self) -> None:
        _check_instance("name", self.name, str)
        _check_instance("type", self.type, Type)


@dataclass(frozen=True)
class Kernel:
    name: str
    params: Sequence[Param]
    grid: Grid

    def __post_init__(self) -> None:
        _check_instance("name", self.name, str)
        _check_sequence("params", self.params, Param)
        _check_instance("grid", self.grid, Grid)


@dataclass(frozen=True)
class Program:
    kernel: Kernel

    def __post_init__(self) -> None:
        _check_instance("kernel", self.kernel, Kernel)


def v_(name: str) -> Var:
    return Var(name)


def let_(name: str, value: Expr) -> Let:
    return Let(var=Var(name), value=value)


def if_(cond: Expr, then_body: Sequence[Stmt], else_body: Sequence[Stmt] = ()) -> If:
    return If(cond=cond, then_body=then_body, else_body=else_body)


def lit_(value: int | float | bool) -> Expr:
    match value:
        case bool() as b:
            return BoolLit(b, type=BoolType())
        case int() as i:
            return IntLit(i, type=IntType())
        case float() as f:
            return FloatLit(f, type=FloatType())


def add_(lhs: Expr, rhs: Expr) -> Expr:
    if lhs == IntLit(0):
        return rhs
    if rhs == IntLit(0):
        return lhs
    return BinOp("+", lhs, rhs)


def sub_(lhs: Expr, rhs: Expr) -> Expr:
    if rhs == IntLit(0):
        return lhs
    if lhs == rhs:
        return IntLit(0)
    # Simplify (x + c) - x → c  and  (c + x) - x → c
    if isinstance(lhs, BinOp) and lhs.op == "+":
        if lhs.rhs == rhs:
            return lhs.lhs
        if lhs.lhs == rhs:
            return lhs.rhs
    return BinOp("-", lhs, rhs)


def mul_(lhs: Expr, rhs: Expr) -> BinOp:
    return BinOp("*", lhs, rhs)


def div_(lhs: Expr, rhs: Expr) -> BinOp:
    return BinOp("/", lhs, rhs)


def floordiv_(lhs: Expr, rhs: Expr) -> BinOp:
    return BinOp("//", lhs, rhs)


def lt_(lhs: Expr, rhs: Expr) -> BinOp:
    return BinOp("<", lhs, rhs)


def ge_(lhs: Expr, rhs: Expr) -> BinOp:
    return BinOp(">=", lhs, rhs)


def and_(lhs: Expr, rhs: Expr) -> BinOp:
    return BinOp("and", lhs, rhs)


def cdiv_(lhs: Expr, rhs: Expr) -> BinOp:
    return BinOp("cdiv", lhs, rhs)


def mm_(lhs: Expr, rhs: Expr) -> BinOp:
    return BinOp("@", lhs, rhs)


def range_(start: Expr, stop: Expr) -> Range:
    return Range(start=start, stop=stop)


def slice_(start: Expr, stop: Expr) -> Slice:
    return Slice(start=start, stop=stop)


def index_(base: Var, indices: Sequence[Expr]) -> TensorIndex:
    return TensorIndex(base=base, indices=indices)


def view_(base: Var, region: Region) -> TensorView:
    return TensorView(base=base, region=region)


def masked_load_(base: Var, region: Region, mask: Region) -> MaskedLoad:
    return MaskedLoad(base=base, region=region, mask=mask)


def masked_store_(base: Var, region: Region, value: Expr, mask: Region) -> MaskedStore:
    return MaskedStore(base=base, region=region, value=value, mask=mask)


def min_(exprs: Sequence[Expr]) -> Expr:
    exprs = _dedup_exprs(_flatten_min(exprs))
    if len(exprs) == 1:
        return exprs[0]
    else:
        return Min(exprs)


def max_(exprs: Sequence[Expr]) -> Expr:
    exprs = _dedup_exprs(_flatten_max(exprs))
    if len(exprs) == 1:
        return exprs[0]
    else:
        return Max(exprs)


def _dedup_exprs(exprs: Sequence[Expr]) -> Sequence[Expr]:
    unique = []
    for expr in exprs:
        if not any(expr == other for other in unique):
            unique.append(expr)
    return unique


def _flatten_min(exprs: Sequence[Expr]) -> Sequence[Expr]:
    flattened = []
    for expr in exprs:
        if isinstance(expr, Min):
            flattened += _flatten_min(expr.args)
        else:
            flattened.append(expr)
    return flattened


def _flatten_max(exprs: Sequence[Expr]) -> Sequence[Expr]:
    flattened = []
    for expr in exprs:
        if isinstance(expr, Max):
            flattened += _flatten_max(expr.args)
        else:
            flattened.append(expr)
    return flattened


def collect_type_vars(typ: Type) -> set[str]:
    """Collect all Var names referenced inside a Type (e.g. tensor dims)."""
    match typ:
        case TensorType(elem_type=et, dims=dims):
            names = collect_type_vars(et)
            for d in dims:
                names |= collect_expr_vars(d)
            return names
        case IntType() | FloatType() | BoolType():
            return set()
        case _:
            raise AssertionError(f"unhandled type: {typ}")


def collect_expr_vars(expr: Expr) -> set[str]:
    """Collect all Var names referenced in an expression."""
    match expr:
        case Var(name=name):
            return {name}
        case IntLit() | FloatLit() | BoolLit():
            return set()
        case BinOp(lhs=lhs, rhs=rhs) | Maximum(lhs=lhs, rhs=rhs):
            return collect_expr_vars(lhs) | collect_expr_vars(rhs)
        case Min(args=args) | Max(args=args):
            result: set[str] = set()
            for a in args:
                result |= collect_expr_vars(a)
            return result
        case (
            Exp2(value=value)
            | Sigmoid(value=value)
            | Rsqrt(value=value)
            | Log2(value=value)
            | Cast(value=value)
            | Not(value=value)
            | ReduceMax(value=value)
            | ReduceSum(value=value)
            | Unsqueeze(value=value)
            | Squeeze(value=value)
            | Transpose(value=value)
        ):
            return collect_expr_vars(value)
        case Zeros(shape=shape):
            result = set()
            for s in shape:
                result |= collect_expr_vars(s)
            return result
        case Full(shape=shape, value=value):
            result = collect_expr_vars(value)
            for s in shape:
                result |= collect_expr_vars(s)
            return result
        case Arange(start=start, stop=stop):
            return collect_expr_vars(start) | collect_expr_vars(stop)
        case Where(cond=cond, on_true=on_true, on_false=on_false):
            return (
                collect_expr_vars(cond)
                | collect_expr_vars(on_true)
                | collect_expr_vars(on_false)
            )
        case BroadcastTo(value=value, shape=shape):
            result = collect_expr_vars(value)
            for s in shape:
                result |= collect_expr_vars(s)
            return result
        case TensorIndex(base=base, indices=indices):
            result = {base.name}
            for idx in indices:
                result |= collect_expr_vars(idx)
            return result
        case TensorView(base=base, region=region):
            result = {base.name}
            for sl in region:
                result |= collect_expr_vars(sl.start)
                result |= collect_expr_vars(sl.stop)
            return result
        case MaskedLoad(base=base, region=region, mask=mask):
            result = {base.name}
            for sl in region:
                result |= collect_expr_vars(sl.start)
                result |= collect_expr_vars(sl.stop)
            for sl in mask:
                result |= collect_expr_vars(sl.start)
                result |= collect_expr_vars(sl.stop)
            return result
        case _:
            raise AssertionError(f"unhandled expr: {expr}")
