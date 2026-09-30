from .preprocess import VarRole, build_type_env, check_variable_names
from . import *


@dataclass(frozen=True)
class ShapeEquation:
    """A shape constraint exported from type inference.

    Represents the requirement that `lhs == rhs`, where one side is a
    known constant (`IntLit`) and the other is a symbolic expression.
    """

    lhs: Expr
    rhs: Expr
    context: str


_ARITH_OPS = {"+", "-", "*", "/", "//", "%", "cdiv"}
_CMP_OPS = {"<", "<=", ">", ">="}
_LOGIC_OPS = {"and"}


def _const_fold(expr: Expr) -> Expr:
    """Try to constant-fold an expression to an IntLit."""
    match expr:
        case IntLit(value=v):
            return IntLit(v, type=IntType())
        case BinOp(op=op, lhs=lhs, rhs=rhs):
            fl = _const_fold(lhs)
            fr = _const_fold(rhs)
            if isinstance(fl, IntLit) and isinstance(fr, IntLit):
                a, b = fl.value, fr.value
                result: int | None = None
                if op == "+":
                    result = a + b
                elif op == "-":
                    result = a - b
                elif op == "*":
                    result = a * b
                elif op == "//" and b != 0:
                    result = a // b
                if result is not None:
                    return IntLit(result, type=IntType())
            return expr
        case _:
            return expr


def _scalar_type(typ: Type) -> Type:
    """Return the scalar element type (unwraps TensorType)."""
    match typ:
        case TensorType(elem_type=et):
            return et
        case IntType() | FloatType() | BoolType():
            return typ
        case _:
            raise AssertionError(f"unhandled type: {typ}")


def _is_tensor(typ: Type) -> bool:
    return isinstance(typ, TensorType)


def _tensor_dims(typ: Type) -> Sequence[Expr]:
    assert isinstance(typ, TensorType)
    return typ.dims


def _promote_scalar(a: Type, b: Type) -> Type:
    """Promote two scalar types: int+float -> float, etc."""
    if isinstance(a, FloatType) or isinstance(b, FloatType):
        return FloatType()
    return a  # both int


def _dims_equal(d1: Expr, d2: Expr) -> bool:
    """Compare two dimension expressions for equality.

    Compares IntLit by value (ignoring the type annotation), so that
    IntLit(4) and IntLit(4, type=IntType()) are treated as equal.
    """
    if isinstance(d1, IntLit) and isinstance(d2, IntLit):
        return d1.value == d2.value
    return d1 == d2


def _has_constant_tensor_dims(typ: Type) -> bool:
    """Return True if typ is a TensorType with all IntLit dims."""
    if not isinstance(typ, TensorType):
        return True  # scalars are always "constant"
    return all(isinstance(d, IntLit) for d in typ.dims)


# ---------------------------------------------------------------------------
# Bidirectional type checking: unify / typeinfer / typecheck
# ---------------------------------------------------------------------------

import dataclasses


def unify(t1: Type, t2: Type, equations: list[ShapeEquation], context: str) -> Type:
    """Unify two types, emitting ShapeEquations for symbolic dim mismatches.

    Returns the unified type, preferring constant (IntLit) dims.
    """
    if not isinstance(t1, TensorType) and not isinstance(t2, TensorType):
        if t1 != t2:
            raise TypeError(f"{context}: type mismatch: {t1} vs {t2}")
        return t1
    if not isinstance(t1, TensorType) or not isinstance(t2, TensorType):
        raise TypeError(f"{context}: tensor/scalar mismatch: {t1} vs {t2}")
    if t1.elem_type != t2.elem_type:
        raise TypeError(
            f"{context}: elem type mismatch: {t1.elem_type} vs {t2.elem_type}"
        )
    if len(t1.dims) != len(t2.dims):
        raise TypeError(f"{context}: rank mismatch: {len(t1.dims)} vs {len(t2.dims)}")
    new_dims: list[Expr] = []
    for i, (d1, d2) in enumerate(zip(t1.dims, t2.dims)):
        if _dims_equal(d1, d2):
            new_dims.append(d1)
        elif isinstance(d1, IntLit) and not isinstance(d2, IntLit):
            equations.append(
                ShapeEquation(lhs=d2, rhs=d1, context=f"{context} dim {i}")
            )
            new_dims.append(d1)
        elif isinstance(d2, IntLit) and not isinstance(d1, IntLit):
            equations.append(
                ShapeEquation(lhs=d1, rhs=d2, context=f"{context} dim {i}")
            )
            new_dims.append(d2)
        elif isinstance(d1, IntLit) and isinstance(d2, IntLit):
            raise TypeError(f"{context}: dim {i} mismatch: {d1.value} vs {d2.value}")
        else:
            raise TypeError(f"{context}: dim {i} both symbolic: {d1} vs {d2}")
    return TensorType(t1.elem_type, new_dims)


def _unify_tensor_shapes(
    lt: Type,
    rt: Type,
    equations: list[ShapeEquation],
    context: str,
) -> Sequence[Expr]:
    """Unify two tensor shapes, returning the unified dims."""
    assert _is_tensor(lt) and _is_tensor(rt)
    unified = unify(
        TensorType(_scalar_type(lt), _tensor_dims(lt)),
        TensorType(_scalar_type(lt), _tensor_dims(rt)),
        equations,
        context,
    )
    return _tensor_dims(unified)


def _infer_binop_type(
    op: str,
    lt: Type,
    rt: Type,
    equations: list[ShapeEquation],
) -> Type:
    """Compute the result type of a BinOp from its operand types."""
    if op == "@":
        if not (_is_tensor(lt) and _is_tensor(rt)):
            raise TypeError("@ requires tensor operands")
        ld, rd = _tensor_dims(lt), _tensor_dims(rt)
        if len(ld) < 2 or len(rd) < 2:
            raise TypeError("@ requires at least 2-D tensors")
        if not _dims_equal(ld[-1], rd[-2]):
            raise TypeError(f"@ inner dims mismatch: {ld[-1]} vs {rd[-2]}")
        return TensorType(FloatType(), list(ld[:-1]) + [rd[-1]])
    if op in _CMP_OPS:
        if _is_tensor(lt) or _is_tensor(rt):
            if _is_tensor(lt) and _is_tensor(rt):
                shape = _unify_tensor_shapes(lt, rt, equations, f"{op} shape")
            elif _is_tensor(lt):
                shape = _tensor_dims(lt)
            else:
                shape = _tensor_dims(rt)
            return TensorType(BoolType(), shape)
        return BoolType()
    if op in _LOGIC_OPS:
        if _scalar_type(lt) != BoolType():
            raise TypeError(f"`{op}` lhs must be bool, got {lt}")
        if _scalar_type(rt) != BoolType():
            raise TypeError(f"`{op}` rhs must be bool, got {rt}")
        if _is_tensor(lt) or _is_tensor(rt):
            if _is_tensor(lt) and _is_tensor(rt):
                shape = _unify_tensor_shapes(lt, rt, equations, f"{op} shape")
            elif _is_tensor(lt):
                shape = _tensor_dims(lt)
            else:
                shape = _tensor_dims(rt)
            return TensorType(BoolType(), shape)
        return BoolType()
    if op in _ARITH_OPS:
        if op == "cdiv":
            if _scalar_type(lt) != IntType():
                raise TypeError(f"cdiv lhs must be int, got {lt}")
            if _scalar_type(rt) != IntType():
                raise TypeError(f"cdiv rhs must be int, got {rt}")
        elem = _promote_scalar(_scalar_type(lt), _scalar_type(rt))
        if _is_tensor(lt) and _is_tensor(rt):
            shape = _unify_tensor_shapes(lt, rt, equations, f"{op} shape")
            return TensorType(elem, shape)
        if _is_tensor(lt):
            return TensorType(elem, _tensor_dims(lt))
        if _is_tensor(rt):
            return TensorType(elem, _tensor_dims(rt))
        return elem
    raise TypeError(f"unknown binop: {op}")


def _infer_squeeze_type(vt: Type, axis: int, equations: list[ShapeEquation]) -> Type:
    """Compute the result type of Squeeze, emitting equation if dim is symbolic."""
    if not _is_tensor(vt):
        raise TypeError(f"squeeze requires tensor, got {vt}")
    dims = list(_tensor_dims(vt))
    if axis < 0 or axis >= len(dims):
        raise TypeError(f"squeeze axis {axis} out of range for {len(dims)}-D")
    d = dims[axis]
    if isinstance(d, IntLit):
        if d.value != 1:
            raise TypeError(f"squeeze axis {axis} dim is {d.value}, not 1")
    else:
        equations.append(
            ShapeEquation(
                lhs=d, rhs=IntLit(1, type=IntType()), context=f"squeeze axis {axis}"
            )
        )
    new_dims = dims[:axis] + dims[axis + 1 :]
    return TensorType(_scalar_type(vt), new_dims)


def _infer_unsqueeze_type(vt: Type, axis: int) -> Type:
    """Compute the result type of Unsqueeze."""
    if not _is_tensor(vt):
        raise TypeError(f"unsqueeze requires tensor, got {vt}")
    dims = list(_tensor_dims(vt))
    if axis < 0 or axis > len(dims):
        raise TypeError(f"unsqueeze axis {axis} out of range for {len(dims)}-D")
    new_dims = dims[:axis] + [IntLit(1, type=IntType())] + dims[axis:]
    return TensorType(_scalar_type(vt), new_dims)


def _infer_tensorview(
    base: Var,
    region: Sequence[Slice],
    env: dict[str, Type],
    equations: list[ShapeEquation],
) -> TensorView:
    """Typeinfer a TensorView: annotate slices, compute symbolic dims."""
    a_base = Var(base.name, type=env[base.name])
    a_region = [
        Slice(
            start=typeinfer(sl.start, env, equations),
            stop=typeinfer(sl.stop, env, equations),
        )
        for sl in region
    ]
    for i, sl in enumerate(a_region):
        assert sl.start.type is not None and sl.stop.type is not None
        if sl.start.type != IntType():
            raise TypeError(f"index slice {i} start must be int, got {sl.start.type}")
        if sl.stop.type != IntType():
            raise TypeError(f"index slice {i} stop must be int, got {sl.stop.type}")
    bt = env[base.name]
    if not _is_tensor(bt):
        raise TypeError(f"index requires tensor, got {bt}")
    bdims = _tensor_dims(bt)
    if len(region) != len(bdims):
        raise TypeError(f"index region rank {len(region)} != tensor rank {len(bdims)}")
    new_dims = [sub_(sl.stop, sl.start) for sl in a_region]
    t = TensorType(_scalar_type(bt), new_dims)
    return TensorView(a_base, a_region, type=t)


def _infer_tensorindex(
    base: Var,
    indices: Sequence[Expr],
    env: dict[str, Type],
    equations: list[ShapeEquation],
) -> TensorIndex:
    """Typeinfer a TensorIndex: annotate point indices, result is scalar."""
    a_base = Var(base.name, type=env[base.name])
    a_indices = [typeinfer(idx, env, equations) for idx in indices]
    for i, idx in enumerate(a_indices):
        assert idx.type is not None
        if idx.type != IntType():
            raise TypeError(f"index axis {i} must be int, got {idx.type}")
    bt = env[base.name]
    if not _is_tensor(bt):
        raise TypeError(f"index requires tensor, got {bt}")
    bdims = _tensor_dims(bt)
    if len(indices) != len(bdims):
        raise TypeError(f"index rank {len(indices)} != tensor rank {len(bdims)}")
    return TensorIndex(a_base, a_indices, type=_scalar_type(bt))


def typeinfer(expr: Expr, env: dict[str, Type], equations: list[ShapeEquation]) -> Expr:
    """Infer the type of *expr* bottom-up. Returns a fully annotated Expr."""
    match expr:
        case IntLit(value=v):
            return IntLit(v, type=IntType())
        case FloatLit(value=v):
            return FloatLit(v, type=FloatType())
        case BoolLit(value=v):
            return BoolLit(v, type=BoolType())
        case Var(name=name):
            if name not in env:
                raise TypeError(f"undefined variable: {name}")
            return Var(name, type=env[name])
        case BinOp(op=op, lhs=lhs, rhs=rhs):
            a_lhs = typeinfer(lhs, env, equations)
            a_rhs = typeinfer(rhs, env, equations)
            assert a_lhs.type is not None and a_rhs.type is not None
            t = _infer_binop_type(op, a_lhs.type, a_rhs.type, equations)
            return BinOp(op, a_lhs, a_rhs, type=t)
        case Min(args=args):
            a_args = [typeinfer(a, env, equations) for a in args]
            for i, a in enumerate(a_args):
                if a.type != IntType():
                    raise TypeError(f"min arg {i} must be int, got {a.type}")
            return Min(a_args, type=IntType())
        case Max(args=args):
            a_args = [typeinfer(a, env, equations) for a in args]
            for i, a in enumerate(a_args):
                if a.type != IntType():
                    raise TypeError(f"max arg {i} must be int, got {a.type}")
            return Max(a_args, type=IntType())
        case Zeros(shape=shape):
            a_shape = [typeinfer(s, env, equations) for s in shape]
            for i, s in enumerate(a_shape):
                if s.type != IntType():
                    raise TypeError(f"zeros shape dim {i} must be int, got {s.type}")
            return Zeros(a_shape, type=TensorType(FloatType(), a_shape))
        case Full(shape=shape, value=value):
            a_shape = [typeinfer(s, env, equations) for s in shape]
            for i, s in enumerate(a_shape):
                if s.type != IntType():
                    raise TypeError(f"full shape dim {i} must be int, got {s.type}")
            a_value = typeinfer(value, env, equations)
            assert a_value.type is not None
            if _is_tensor(a_value.type):
                raise TypeError(f"full value must be scalar, got {a_value.type}")
            t = TensorType(_scalar_type(a_value.type), a_shape)
            return Full(a_shape, a_value, type=t)
        case Arange(start=start, stop=stop):
            a_start = typeinfer(start, env, equations)
            a_stop = typeinfer(stop, env, equations)
            assert a_start.type is not None and a_stop.type is not None
            if a_start.type != IntType():
                raise TypeError(f"arange start must be int, got {a_start.type}")
            if a_stop.type != IntType():
                raise TypeError(f"arange stop must be int, got {a_stop.type}")
            return Arange(
                a_start, a_stop, type=TensorType(IntType(), [sub_(a_stop, a_start)])
            )
        case Where(cond=cond, on_true=on_true, on_false=on_false):
            a_cond = typeinfer(cond, env, equations)
            a_true = typeinfer(on_true, env, equations)
            a_false = typeinfer(on_false, env, equations)
            ct, tt, ft = a_cond.type, a_true.type, a_false.type
            assert ct is not None and tt is not None and ft is not None
            # cond must be a bool tensor (not scalar)
            if not _is_tensor(ct):
                raise TypeError(f"where cond must be a tensor, got {ct}")
            if _scalar_type(ct) != BoolType():
                raise TypeError(f"where cond must be bool tensor, got {ct}")
            if _scalar_type(tt) != _scalar_type(ft):
                raise TypeError(
                    f"where branches must have same scalar type: {tt} vs {ft}"
                )
            # Each branch must be a tensor of the same shape as cond, or a scalar.
            cond_shape = _tensor_dims(ct)
            for label, bt in [("on_true", tt), ("on_false", ft)]:
                if _is_tensor(bt):
                    _ = _unify_tensor_shapes(ct, bt, equations, f"where {label} shape")
            # Result shape = cond's shape, elem type = branches' scalar type.
            result_type: Type = TensorType(_scalar_type(tt), cond_shape)
            return Where(a_cond, a_true, a_false, type=result_type)
        case ReduceMax(value=value, axis=axis):
            a_value = typeinfer(value, env, equations)
            assert a_value.type is not None
            vt = a_value.type
            if not _is_tensor(vt):
                raise TypeError(f"reduce requires tensor, got {vt}")
            dims = list(_tensor_dims(vt))
            if axis < 0 or axis >= len(dims):
                raise TypeError(f"axis {axis} out of range for {len(dims)}-D")
            new_dims = dims[:axis] + dims[axis + 1 :]
            # Triton promotes sub-32-bit integer reductions (including
            # int1/bool masks) to a 32-bit integer before reducing.  Modeling
            # max(bool) as bool is observably wrong when its result is later
            # compared with zero, as in an explicit any(mask) guard.
            result_elem = (
                IntType() if _scalar_type(vt) == BoolType()
                else _scalar_type(vt)
            )
            t = (
                result_elem
                if len(new_dims) == 0
                else TensorType(result_elem, new_dims)
            )
            return ReduceMax(a_value, axis, type=t)
        case ReduceSum(value=value, axis=axis):
            a_value = typeinfer(value, env, equations)
            assert a_value.type is not None
            vt = a_value.type
            if not _is_tensor(vt):
                raise TypeError(f"reduce requires tensor, got {vt}")
            dims = list(_tensor_dims(vt))
            if axis < 0 or axis >= len(dims):
                raise TypeError(f"axis {axis} out of range for {len(dims)}-D")
            new_dims = dims[:axis] + dims[axis + 1 :]
            result_elem = (
                IntType() if _scalar_type(vt) == BoolType()
                else _scalar_type(vt)
            )
            t = (
                result_elem
                if len(new_dims) == 0
                else TensorType(result_elem, new_dims)
            )
            return ReduceSum(a_value, axis, type=t)
        case Exp2(value=value) | Sigmoid(value=value) | Rsqrt(value=value) | Log2(value=value):
            a_value = typeinfer(value, env, equations)
            assert a_value.type is not None
            name = type(expr).__name__.lower()
            if _scalar_type(a_value.type) != FloatType():
                raise TypeError(f"{name} requires float operand, got {a_value.type}")
            return type(expr)(a_value, type=a_value.type)
        case Cast(value=value, kind=kind, target=target):
            a_value = typeinfer(value, env, equations)
            assert a_value.type is not None
            if kind == "float":
                elem_type: Type = FloatType()
            elif kind == "int32":
                elem_type = IntType()
            else:
                raise TypeError(f"unsupported cast proof kind {kind!r}")
            result_type: Type = elem_type
            if isinstance(a_value.type, TensorType):
                result_type = TensorType(elem_type, a_value.type.dims)
            return Cast(a_value, kind, target, type=result_type)
        case Not(value=value):
            a_value = typeinfer(value, env, equations)
            assert a_value.type is not None
            if _scalar_type(a_value.type) != BoolType():
                raise TypeError(f"not requires bool operand, got {a_value.type}")
            return Not(a_value, type=a_value.type)
        case Maximum(lhs=lhs, rhs=rhs):
            a_lhs = typeinfer(lhs, env, equations)
            a_rhs = typeinfer(rhs, env, equations)
            assert a_lhs.type is not None and a_rhs.type is not None
            t = unify(a_lhs.type, a_rhs.type, equations, "maximum")
            return Maximum(a_lhs, a_rhs, type=t)
        case Unsqueeze(value=value, axis=axis):
            a_value = typeinfer(value, env, equations)
            assert a_value.type is not None
            t = _infer_unsqueeze_type(a_value.type, axis)
            return Unsqueeze(a_value, axis, type=t)
        case Squeeze(value=value, axis=axis):
            a_value = typeinfer(value, env, equations)
            assert a_value.type is not None
            t = _infer_squeeze_type(a_value.type, axis, equations)
            return Squeeze(a_value, axis, type=t)
        case BroadcastTo(value=value, shape=shape):
            a_value = typeinfer(value, env, equations)
            a_shape = [typeinfer(s, env, equations) for s in shape]
            assert a_value.type is not None
            vt = a_value.type
            if not _is_tensor(vt):
                raise TypeError(f"broadcast_to requires tensor, got {vt}")
            vdims = _tensor_dims(vt)
            if len(vdims) != len(a_shape):
                raise TypeError(
                    f"broadcast_to rank mismatch: {len(vdims)} vs {len(a_shape)}"
                )
            for vd, sd in zip(vdims, a_shape):
                if not _dims_equal(vd, IntLit(1)) and not _dims_equal(vd, sd):
                    raise TypeError(
                        f"broadcast_to dim incompatible: {vd} cannot broadcast to {sd}"
                    )
            return BroadcastTo(
                a_value, a_shape, type=TensorType(_scalar_type(vt), a_shape)
            )
        case Transpose(value=value, permutation=permutation):
            a_value = typeinfer(value, env, equations)
            assert a_value.type is not None
            vt = a_value.type
            if not _is_tensor(vt):
                raise TypeError(f"transpose requires tensor, got {vt}")
            dims = _tensor_dims(vt)
            if sorted(permutation) != list(range(len(dims))):
                raise TypeError(f"invalid permutation {permutation}")
            new_dims = [dims[p] for p in permutation]
            return Transpose(
                a_value, permutation, type=TensorType(_scalar_type(vt), new_dims)
            )
        case TensorView(base=base, region=region):
            return _infer_tensorview(base, region, env, equations)
        case TensorIndex(base=base, indices=indices):
            return _infer_tensorindex(base, indices, env, equations)
        case MaskedLoad(base=base, region=region, mask=mask):
            # Type the view part (same logic as _infer_tensorview)
            a_base = Var(base.name, type=env[base.name])
            a_region = [
                Slice(
                    start=typeinfer(sl.start, env, equations),
                    stop=typeinfer(sl.stop, env, equations),
                )
                for sl in region
            ]
            a_mask = [
                Slice(
                    start=typeinfer(sl.start, env, equations),
                    stop=typeinfer(sl.stop, env, equations),
                )
                for sl in mask
            ]
            for i, sl in enumerate(a_region):
                assert sl.start.type is not None and sl.stop.type is not None
                if sl.start.type != IntType():
                    raise TypeError(
                        f"masked_load region slice {i} start must be int, got {sl.start.type}"
                    )
                if sl.stop.type != IntType():
                    raise TypeError(
                        f"masked_load region slice {i} stop must be int, got {sl.stop.type}"
                    )
            for i, sl in enumerate(a_mask):
                assert sl.start.type is not None and sl.stop.type is not None
                if sl.start.type != IntType():
                    raise TypeError(
                        f"masked_load mask slice {i} start must be int, got {sl.start.type}"
                    )
                if sl.stop.type != IntType():
                    raise TypeError(
                        f"masked_load mask slice {i} stop must be int, got {sl.stop.type}"
                    )
            bt = env[base.name]
            if not _is_tensor(bt):
                raise TypeError(f"masked_load requires tensor, got {bt}")
            bdims = _tensor_dims(bt)
            if len(region) != len(bdims):
                raise TypeError(
                    f"masked_load region rank {len(region)} != tensor rank {len(bdims)}"
                )
            if len(mask) != len(bdims):
                raise TypeError(
                    f"masked_load mask rank {len(mask)} != tensor rank {len(bdims)}"
                )
            new_dims = [sub_(sl.stop, sl.start) for sl in a_region]
            t = TensorType(_scalar_type(bt), new_dims)
            return MaskedLoad(a_base, a_region, a_mask, type=t)
        case _:
            raise AssertionError(f"unhandled expr: {expr}")


def typecheck(
    expr: Expr,
    env: dict[str, Type],
    expected: Type,
    equations: list[ShapeEquation],
) -> Expr:
    """Check *expr* against *expected* type, propagating top-down.

    For shape-transparent nodes (Unsqueeze, Squeeze, Transpose, Exp2,
    arith BinOp, Where, Maximum), propagates *expected* to children.
    For all other forms, falls back to ``typeinfer`` + ``unify``.
    """
    match expr:
        # --- Shape-transparent: propagate expected ---
        case Unsqueeze(value=value, axis=axis):
            if isinstance(expected, TensorType) and 0 <= axis < len(expected.dims):
                inner = TensorType(
                    expected.elem_type,
                    list(expected.dims[:axis]) + list(expected.dims[axis + 1 :]),
                )
                a_value = typecheck(value, env, inner, equations)
            else:
                a_value = typeinfer(value, env, equations)
            assert a_value.type is not None
            t = _infer_unsqueeze_type(a_value.type, axis)
            return Unsqueeze(a_value, axis, type=t)

        case Squeeze(value=value, axis=axis):
            if isinstance(expected, TensorType):
                inner = TensorType(
                    expected.elem_type,
                    list(expected.dims[:axis])
                    + [IntLit(1, type=IntType())]
                    + list(expected.dims[axis:]),
                )
                a_value = typecheck(value, env, inner, equations)
            else:
                a_value = typeinfer(value, env, equations)
            assert a_value.type is not None
            t = _infer_squeeze_type(a_value.type, axis, equations)
            return Squeeze(a_value, axis, type=t)

        case Transpose(value=value, permutation=permutation):
            if isinstance(expected, TensorType):
                inv_perm = [0] * len(permutation)
                for i, p in enumerate(permutation):
                    inv_perm[p] = i
                inner = TensorType(
                    expected.elem_type,
                    [expected.dims[inv_perm[i]] for i in range(len(expected.dims))],
                )
                a_value = typecheck(value, env, inner, equations)
            else:
                a_value = typeinfer(value, env, equations)
            assert a_value.type is not None
            vt = a_value.type
            if not _is_tensor(vt):
                raise TypeError(f"transpose requires tensor, got {vt}")
            dims = _tensor_dims(vt)
            if sorted(permutation) != list(range(len(dims))):
                raise TypeError(f"invalid permutation {permutation}")
            new_dims = [dims[p] for p in permutation]
            return Transpose(
                a_value, permutation, type=TensorType(_scalar_type(vt), new_dims)
            )

        case Exp2(value=value) | Sigmoid(value=value) | Rsqrt(value=value) | Log2(value=value):
            a_value = typecheck(value, env, expected, equations)
            assert a_value.type is not None
            name = type(expr).__name__.lower()
            if _scalar_type(a_value.type) != FloatType():
                raise TypeError(f"{name} requires float operand, got {a_value.type}")
            return type(expr)(a_value, type=a_value.type)

        case Cast():
            inferred = typeinfer(expr, env, equations)
            assert inferred.type is not None
            unified = unify(inferred.type, expected, equations, "cast result")
            return dataclasses.replace(inferred, type=unified)

        case Maximum(lhs=lhs, rhs=rhs):
            a_lhs = typecheck(lhs, env, expected, equations)
            a_rhs = typecheck(rhs, env, expected, equations)
            assert a_lhs.type is not None and a_rhs.type is not None
            t = unify(a_lhs.type, a_rhs.type, equations, "maximum")
            return Maximum(a_lhs, a_rhs, type=t)

        case BinOp(op=op, lhs=lhs, rhs=rhs) if op in _ARITH_OPS:
            # Propagate expected to both; scalars simply ignore it
            a_lhs = typecheck(lhs, env, expected, equations)
            a_rhs = typecheck(rhs, env, expected, equations)
            assert a_lhs.type is not None and a_rhs.type is not None
            t = _infer_binop_type(op, a_lhs.type, a_rhs.type, equations)
            return BinOp(op, a_lhs, a_rhs, type=t)

        case Where(cond=cond, on_true=on_true, on_false=on_false):
            a_cond = typeinfer(cond, env, equations)
            a_true = typecheck(on_true, env, expected, equations)
            a_false = typecheck(on_false, env, expected, equations)
            ct, tt, ft = a_cond.type, a_true.type, a_false.type
            assert ct is not None and tt is not None and ft is not None
            # cond must be a bool tensor (not scalar)
            if not _is_tensor(ct):
                raise TypeError(f"where cond must be a tensor, got {ct}")
            if _scalar_type(ct) != BoolType():
                raise TypeError(f"where cond must be bool tensor, got {ct}")
            if _scalar_type(tt) != _scalar_type(ft):
                raise TypeError(
                    f"where branches must have same scalar type: {tt} vs {ft}"
                )
            # Each branch must be a tensor of the same shape as cond, or a scalar.
            cond_shape = _tensor_dims(ct)
            for label, bt in [("on_true", tt), ("on_false", ft)]:
                if _is_tensor(bt):
                    _ = _unify_tensor_shapes(ct, bt, equations, f"where {label} shape")
            # Result shape = cond's shape, elem type = branches' scalar type.
            result_type: Type = TensorType(_scalar_type(tt), cond_shape)
            return Where(a_cond, a_true, a_false, type=result_type)

        case (
            Var()
            | IntLit()
            | FloatLit()
            | BoolLit()
            | BinOp()
            | Min()
            | Max()
            | Zeros()
            | Full()
            | Arange()
            | ReduceMax()
            | ReduceSum()
            | Not()
            | BroadcastTo()
            | TensorView()
            | TensorIndex()
            | MaskedLoad()
        ):
            # Fallback: typeinfer then unify with expected
            result = typeinfer(expr, env, equations)
            assert result.type is not None
            # Only unify if both are tensor (scalar expected with scalar
            # inferred is fine; mismatches caught at statement level)
            if isinstance(result.type, TensorType) and isinstance(expected, TensorType):
                unified = unify(result.type, expected, equations, "check")
                return dataclasses.replace(result, type=unified)
            return result

        case _:
            raise AssertionError(f"unhandled expr: {expr}")


def annotate_expr(
    expr: Expr,
    env: dict[str, Type],
    equations: list[ShapeEquation] | None = None,
    expected: Type | None = None,
) -> Expr:
    """Convenience entry point: dispatches to typecheck or typeinfer."""
    if equations is None:
        equations = []
    if expected is not None:
        return typecheck(expr, env, expected, equations)
    return typeinfer(expr, env, equations)


def _annotate_target(
    target: Var | TensorView,
    env: dict[str, Type],
    equations: list[ShapeEquation],
    expected: Type | None = None,
) -> Var | TensorView:
    """Annotate an assignment target.

    When *expected* is provided (from the value expression), unify
    TensorView dims with expected constant dims.
    """
    match target:
        case Var(name=name):
            return Var(name, type=env[name])
        case TensorView(base=base, region=region):
            a_base = Var(base.name, type=env[base.name])
            a_region = [
                Slice(
                    start=annotate_expr(sl.start, env, equations),
                    stop=annotate_expr(sl.stop, env, equations),
                )
                for sl in region
            ]
            bt = env[base.name]
            new_dims = [sub_(sl.stop, sl.start) for sl in a_region]
            t: Type = TensorType(_scalar_type(bt), new_dims)
            if expected is not None:
                t = unify(t, expected, equations, "target")
            return TensorView(a_base, a_region, type=t)


def _annotate_stmt(
    stmt: Stmt,
    env: dict[str, Type],
    equations: list[ShapeEquation],
    role_map: dict[str, VarRole],
) -> Stmt:
    """Annotate all expressions in a statement.

    Uses bidirectional type checking for assignments:
    1. If target shape is constant → propagate to value as expected.
    2. Else infer value type → if constant, propagate back to target.
    3. If neither is constant → error.
    """
    match stmt:
        case Assign(target=Var(name=tname)) if role_map.get(tname) == "let_bind":
            raise TypeError(f"assign: cannot assign to let-bound name '{tname}'")
        case Assign(target=target, op=op, value=value):
            # Step 1: Get target type (from env, possibly symbolic for TensorView)
            raw_target = _annotate_target(target, env, equations)
            assert raw_target.type is not None
            target_type = raw_target.type

            if _has_constant_tensor_dims(target_type):
                # Target is constant → propagate to value
                a_value = annotate_expr(value, env, equations, expected=target_type)
                a_target = raw_target
            else:
                # Target is symbolic → infer value, try to propagate back
                a_value = annotate_expr(value, env, equations)
                assert a_value.type is not None
                if _has_constant_tensor_dims(a_value.type):
                    # Value is constant → propagate to target
                    a_target = _annotate_target(
                        target, env, equations, expected=a_value.type
                    )
                else:
                    raise TypeError(
                        f"assign: neither target nor value has constant tensor dims: target={target_type}, value={a_value.type}"
                    )

            assert a_target.type is not None and a_value.type is not None
            # tensor/scalar consistency (plain assign only).
            # For augmented assigns (target op= value), _infer_binop_type already
            # handles tensor-scalar mixing: e.g. tensor += scalar produces a tensor
            # result, so the value type will match the target.
            if op is None and _is_tensor(a_target.type) != _is_tensor(a_value.type):
                raise TypeError(
                    f"assign type mismatch: target is {a_target.type}, value is {a_value.type}"
                )
            # element type match
            if _scalar_type(a_target.type) != _scalar_type(a_value.type):
                raise TypeError(
                    f"assign element type mismatch: {_scalar_type(a_target.type)} vs {_scalar_type(a_value.type)}"
                )
            return Assign(target=a_target, op=op, value=a_value)
        case For(var=var, iters=iters, body=body):
            a_var = Var(var.name, type=IntType())
            a_start = annotate_expr(iters.start, env, equations)
            a_stop = annotate_expr(iters.stop, env, equations)
            assert a_start.type is not None and a_stop.type is not None
            if a_start.type != IntType():
                raise TypeError(f"for loop start must be int, got {a_start.type}")
            if a_stop.type != IntType():
                raise TypeError(f"for loop stop must be int, got {a_stop.type}")
            a_iters = Range(start=a_start, stop=a_stop)
            # No child env needed: check_variable_names guarantees the loop var
            # name is globally unique and already pre-seeded in env.
            a_body = [_annotate_stmt(s, env, equations, role_map) for s in body]
            return For(var=a_var, iters=a_iters, body=a_body)
        case Let(var=var, value=value):
            name = var.name
            # Annotate value *before* adding to env (no self-reference).
            a_value = annotate_expr(value, env, equations)
            assert a_value.type is not None
            if a_value.type != IntType():
                raise TypeError(
                    f"let '{name}': binding must be int, got {a_value.type}"
                )
            # Register in env for subsequent statements.
            env[name] = IntType()
            return Let(var=Var(name, type=IntType()), value=a_value)
        case If(cond=cond, then_body=then_body, else_body=else_body):
            a_cond = annotate_expr(cond, env, equations)
            assert a_cond.type is not None
            if a_cond.type != BoolType():
                raise TypeError(f"if condition must be bool, got {a_cond.type}")
            a_then = [_annotate_stmt(s, env, equations, role_map) for s in then_body]
            a_else = [_annotate_stmt(s, env, equations, role_map) for s in else_body]
            return If(cond=a_cond, then_body=a_then, else_body=a_else)
        case MaskedStore(base=base, region=region, value=value, mask=mask):
            # Annotate the view region and mask
            a_base = Var(base.name, type=env[base.name])
            a_region = [
                Slice(
                    start=annotate_expr(sl.start, env, equations),
                    stop=annotate_expr(sl.stop, env, equations),
                )
                for sl in region
            ]
            a_mask = [
                Slice(
                    start=annotate_expr(sl.start, env, equations),
                    stop=annotate_expr(sl.stop, env, equations),
                )
                for sl in mask
            ]
            bt = env[base.name]
            if not _is_tensor(bt):
                raise TypeError(f"masked_store requires tensor, got {bt}")
            new_dims = [sub_(sl.stop, sl.start) for sl in a_region]
            target_type: Type = TensorType(_scalar_type(bt), new_dims)

            if _has_constant_tensor_dims(target_type):
                a_value = annotate_expr(value, env, equations, expected=target_type)
            else:
                a_value = annotate_expr(value, env, equations)
                assert a_value.type is not None
                if not _has_constant_tensor_dims(a_value.type):
                    raise TypeError(
                        f"masked_store: neither target nor value has constant tensor dims: target={target_type}, value={a_value.type}"
                    )

            assert a_value.type is not None
            if _is_tensor(target_type) != _is_tensor(a_value.type):
                raise TypeError(
                    f"masked_store type mismatch: target is {target_type}, value is {a_value.type}"
                )
            if _scalar_type(target_type) != _scalar_type(a_value.type):
                raise TypeError(
                    f"masked_store element type mismatch: {_scalar_type(target_type)} vs {_scalar_type(a_value.type)}"
                )
            return MaskedStore(base=a_base, region=a_region, value=a_value, mask=a_mask)
        case _:
            raise AssertionError(f"unhandled stmt: {stmt}")


def _annotate_type(typ: Type, env: dict[str, Type]) -> Type:
    """Annotate expressions inside type nodes (e.g. TensorType dims)."""
    match typ:
        case TensorType(elem_type=et, dims=dims):
            a_dims = [annotate_expr(d, env) for d in dims]
            return TensorType(elem_type=_annotate_type(et, env), dims=a_dims)
        case IntType() | FloatType() | BoolType():
            return typ
        case _:
            raise AssertionError(f"unhandled type: {typ}")


def infer_types(
    kernel: Kernel, role_map: dict[str, VarRole] | None = None
) -> tuple[Kernel, list[ShapeEquation]]:
    """Return a new Kernel with every Expr annotated with its inferred type,
    plus a list of shape equations that must hold.

    Uses bidirectional type checking: constant target types propagate to
    value expressions, resolving symbolic TensorView dims.

    ``role_map`` may carry the result of a pre-specialization name check. This
    pass always revalidates the current kernel, while the optional earlier map
    also detects inconsistent roles for names preserved by specialization.
    """
    checked_roles = check_variable_names(kernel)
    if role_map is not None:
        # Specialization/let expansion may legitimately remove names after an
        # earlier check, so the current kernel is authoritative.  Reject only a
        # direct disagreement for names still present in both maps.
        disagreements = {
            name
            for name in checked_roles.keys() & role_map.keys()
            if checked_roles[name] != role_map[name]
        }
        if disagreements:
            raise ValueError(
                "infer_types: stale or inconsistent variable roles: "
                f"{sorted(disagreements)}"
            )
    _roles = checked_roles
    # 1. Build environment using the shared flat builder.
    env = build_type_env(kernel)

    equations: list[ShapeEquation] = []

    # 2. Annotate params
    annotated_params = [
        Param(name=p.name, type=_annotate_type(p.type, env)) for p in kernel.params
    ]

    # 3. Annotate grid iters
    annotated_iters = [
        GridIter(
            var=Var(it.var.name, type=IntType()),
            iters=Range(
                start=annotate_expr(it.iters.start, env),
                stop=annotate_expr(it.iters.stop, env),
            ),
        )
        for it in kernel.grid.iters
    ]

    # 4. Annotate decls
    annotated_decls = [
        VarDecl(
            var=Var(d.var.name, type=d.type),
            type=_annotate_type(d.type, env),
        )
        for d in kernel.grid.decls
    ]

    # 5. Annotate body statements (collecting shape equations).
    # No child-env copy is needed: check_variable_names guarantees all For/Let
    # names are globally unique and already in env.
    annotated_body = [
        _annotate_stmt(s, env, equations, _roles) for s in kernel.grid.body
    ]

    annotated_grid = Grid(
        iters=annotated_iters,
        decls=annotated_decls,
        body=annotated_body,
    )

    typed_kernel = Kernel(
        name=kernel.name,
        params=annotated_params,
        grid=annotated_grid,
    )
    # 6. Verify all inferred tensor types have constant dims
    _verify_stmt_constant_dims(annotated_body)

    return typed_kernel, equations


def infer_types_for_proof(
    kernel: Kernel,
    role_map: dict[str, VarRole] | None = None,
    *,
    context: str = "proof input",
) -> Kernel:
    """Infer types and reject unencoded shape hypotheses.

    ``infer_types`` exposes symbolic-to-constant equations for callers that
    intend to discharge them.  The current proof pipelines do not put those
    equations in their contracts, so accepting one there would silently add an
    assumption.  Keep that fail-closed rule in one shared entry point.
    """

    typed, equations = infer_types(kernel, role_map)
    if equations:
        raise ValueError(
            f"{context} has residual shape equations that are not contract "
            f"hypotheses: {equations}"
        )
    return typed


def _verify_constant_tensor_dims(expr: Expr) -> None:
    """Verify that all inferred TensorType dims are IntLit constants.

    Skips Var nodes since their types come from declarations rather
    than inference. All other nodes (including TensorView) must have
    constant dims after bidirectional type checking.
    """
    if isinstance(expr, Var):
        return

    # Check this node's type
    if isinstance(expr.type, TensorType):
        for i, d in enumerate(expr.type.dims):
            folded = _const_fold(d)
            if not isinstance(folded, IntLit):
                raise TypeError(
                    f"inferred tensor dim {i} is not constant on {type(expr).__name__}: {d}"
                )

    # Recurse into sub-expressions
    match expr:
        case IntLit() | FloatLit() | BoolLit():
            pass
        case BinOp(lhs=lhs, rhs=rhs) | Maximum(lhs=lhs, rhs=rhs):
            _verify_constant_tensor_dims(lhs)
            _verify_constant_tensor_dims(rhs)
        case Min(args=args) | Max(args=args):
            for a in args:
                _verify_constant_tensor_dims(a)
        case Zeros(shape=shape):
            for s in shape:
                _verify_constant_tensor_dims(s)
        case Full(shape=shape, value=value):
            for s in shape:
                _verify_constant_tensor_dims(s)
            _verify_constant_tensor_dims(value)
        case Arange(start=start, stop=stop):
            _verify_constant_tensor_dims(start)
            _verify_constant_tensor_dims(stop)
        case Where(cond=cond, on_true=on_true, on_false=on_false):
            _verify_constant_tensor_dims(cond)
            _verify_constant_tensor_dims(on_true)
            _verify_constant_tensor_dims(on_false)
        case ReduceMax(value=v) | ReduceSum(value=v) | Exp2(value=v) | Sigmoid(value=v) | Rsqrt(value=v) | Log2(value=v) | Cast(value=v) | Not(value=v):
            _verify_constant_tensor_dims(v)
        case Unsqueeze(value=value) | Squeeze(value=value):
            _verify_constant_tensor_dims(value)
        case BroadcastTo(value=value, shape=shape):
            _verify_constant_tensor_dims(value)
            for s in shape:
                _verify_constant_tensor_dims(s)
        case Transpose(value=value):
            _verify_constant_tensor_dims(value)
        case TensorView():
            pass  # already handled above
        case TensorIndex(base=base, indices=indices):
            _verify_constant_tensor_dims(base)
            for idx in indices:
                _verify_constant_tensor_dims(idx)
        case MaskedLoad(base=base, region=region, mask=mask):
            _verify_constant_tensor_dims(base)
            for sl in region:
                _verify_constant_tensor_dims(sl.start)
                _verify_constant_tensor_dims(sl.stop)
            for sl in mask:
                _verify_constant_tensor_dims(sl.start)
                _verify_constant_tensor_dims(sl.stop)
        case _:
            raise AssertionError(f"unhandled expr: {expr}")


def _verify_stmt_constant_dims(stmts: Sequence[Stmt]) -> None:
    for stmt in stmts:
        match stmt:
            case Assign(target=target, value=value):
                _verify_constant_tensor_dims(target)
                _verify_constant_tensor_dims(value)
            case For(var=_, iters=iters, body=body):
                _verify_constant_tensor_dims(iters.start)
                _verify_constant_tensor_dims(iters.stop)
                _verify_stmt_constant_dims(body)
            case Let(var=_, value=value):
                _verify_constant_tensor_dims(value)
            case If(cond=cond, then_body=then_body, else_body=else_body):
                _verify_constant_tensor_dims(cond)
                _verify_stmt_constant_dims(then_body)
                _verify_stmt_constant_dims(else_body)
            case MaskedStore(base=base, region=region, value=value, mask=mask):
                _verify_constant_tensor_dims(base)
                for sl in region:
                    _verify_constant_tensor_dims(sl.start)
                    _verify_constant_tensor_dims(sl.stop)
                _verify_constant_tensor_dims(value)
                for sl in mask:
                    _verify_constant_tensor_dims(sl.start)
                    _verify_constant_tensor_dims(sl.stop)
            case _:
                raise AssertionError(f"unhandled stmt: {stmt}")
