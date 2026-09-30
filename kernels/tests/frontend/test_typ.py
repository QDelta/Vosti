import pytest

from tests.fixtures import matmul_kernel
from ir.pp import pretty_kernel, pp_expr
from ir.preprocess import check_variable_names, build_type_env
from ir.subst import specialize_kernel_constants, expand_let_bindings
from ir.typ import *


def check_all_exprs_typed(expr: Expr) -> None:
    """Assert every Expr node in the tree has type != None."""
    assert expr.type is not None, f"untyped expr: {expr}"
    match expr:
        case IntLit() | FloatLit() | BoolLit() | Var():
            pass
        case BinOp(lhs=lhs, rhs=rhs):
            check_all_exprs_typed(lhs)
            check_all_exprs_typed(rhs)
        case Min(args=args) | Max(args=args):
            for a in args:
                check_all_exprs_typed(a)
        case Zeros(shape=shape):
            for s in shape:
                check_all_exprs_typed(s)
        case Full(shape=shape, value=value):
            for s in shape:
                check_all_exprs_typed(s)
            check_all_exprs_typed(value)
        case Arange(start=start, stop=stop):
            check_all_exprs_typed(start)
            check_all_exprs_typed(stop)
        case Where(cond=cond, on_true=on_true, on_false=on_false):
            check_all_exprs_typed(cond)
            check_all_exprs_typed(on_true)
            check_all_exprs_typed(on_false)
        case ReduceMax(value=value) | ReduceSum(value=value) | Exp2(value=value) | Sigmoid(value=value) | Rsqrt(value=value) | Log2(value=value) | Cast(value=value) | Not(value=value):
            check_all_exprs_typed(value)
        case Maximum(lhs=lhs, rhs=rhs):
            check_all_exprs_typed(lhs)
            check_all_exprs_typed(rhs)
        case Unsqueeze(value=value) | Squeeze(value=value):
            check_all_exprs_typed(value)
        case BroadcastTo(value=value, shape=shape):
            check_all_exprs_typed(value)
            for s in shape:
                check_all_exprs_typed(s)
        case Transpose(value=value):
            check_all_exprs_typed(value)
        case TensorView(base=base, region=region):
            check_all_exprs_typed(base)
            for sl in region:
                check_all_exprs_typed(sl.start)
                check_all_exprs_typed(sl.stop)
        case TensorIndex(base=base, indices=indices):
            check_all_exprs_typed(base)
            for idx in indices:
                check_all_exprs_typed(idx)
        case MaskedLoad(base=base, region=region, mask=mask):
            check_all_exprs_typed(base)
            for sl in region:
                check_all_exprs_typed(sl.start)
                check_all_exprs_typed(sl.stop)
            for sl in mask:
                check_all_exprs_typed(sl.start)
                check_all_exprs_typed(sl.stop)
        case _:
            raise AssertionError(f"unhandled expr: {expr}")


def check_all_stmts_typed(stmts: Sequence[Stmt]) -> None:
    for stmt in stmts:
        match stmt:
            case Assign(target=target, value=value):
                check_all_exprs_typed(target)
                check_all_exprs_typed(value)
            case For(var=var, iters=iters, body=body):
                check_all_exprs_typed(var)
                check_all_exprs_typed(iters.start)
                check_all_exprs_typed(iters.stop)
                check_all_stmts_typed(body)
            case Let(var=var, value=value):
                check_all_exprs_typed(var)
                check_all_exprs_typed(value)
            case If(cond=cond, then_body=then_body, else_body=else_body):
                check_all_exprs_typed(cond)
                check_all_stmts_typed(then_body)
                check_all_stmts_typed(else_body)
            case MaskedStore(base=base, region=region, value=value, mask=mask):
                check_all_exprs_typed(base)
                for sl in region:
                    check_all_exprs_typed(sl.start)
                    check_all_exprs_typed(sl.stop)
                for sl in mask:
                    check_all_exprs_typed(sl.start)
                    check_all_exprs_typed(sl.stop)
                check_all_exprs_typed(value)
            case _:
                raise AssertionError(f"unhandled stmt: {stmt}")


def check_kernel_fully_typed(kernel: Kernel) -> None:
    """Assert every expression in the kernel has a type annotation."""
    for it in kernel.grid.iters:
        check_all_exprs_typed(it.var)
        check_all_exprs_typed(it.iters.start)
        check_all_exprs_typed(it.iters.stop)
    for d in kernel.grid.decls:
        check_all_exprs_typed(d.var)
    check_all_stmts_typed(kernel.grid.body)


def test_matmul() -> None:
    kernel = matmul_kernel()
    specialized = specialize_kernel_constants(
        kernel, {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32}
    )
    typed, equations = infer_types(specialized)
    check_kernel_fully_typed(typed)

    # Check that the pretty-printer still works on the typed kernel
    print(pretty_kernel(typed))
    print()

    # Print exported shape equations
    print(f"  Shape equations ({len(equations)}):")
    for eq in equations:
        print(f"    {eq.context}: {pp_expr(eq.lhs)} == {pp_expr(eq.rhs)}")
    assert equations == [], "source-derived matmul geometry should type-check exactly"

    # Spot check specific types via the canonical type env builder
    env = build_type_env(typed)

    assert env["k"] == IntType()
    _check_tensor_type(env["acc"], FloatType(), [128, 128])
    _check_tensor_type(env["block_a"], FloatType(), [128, 32])
    _check_tensor_type(env["block_b"], FloatType(), [32, 128])


def test_proof_type_inference_rejects_residual_shape_equations() -> None:
    """A proof pipeline cannot silently assume a symbolic dimension is four."""
    n = v_("N")
    kernel = Kernel(
        name="residual_shape_equation",
        params=[Param("x", TensorType(FloatType(), [n]))],
        grid=Grid(
            iters=[],
            decls=[VarDecl(v_("tmp"), TensorType(FloatType(), [lit_(4)]))],
            body=[
                Assign(
                    target=v_("tmp"),
                    op=None,
                    value=view_(v_("x"), [Slice(lit_(0), n)]),
                )
            ],
        ),
    )

    _, equations = infer_types(kernel)
    assert equations
    with pytest.raises(ValueError, match="residual shape equations"):
        infer_types_for_proof(kernel)


def infer_expr_type(expr: Expr, env: dict[str, Type]) -> Type:
    """Convenience wrapper: infer an expression's type and return it."""
    result = typeinfer(expr, env, [])
    assert result.type is not None
    return result.type


def test_scalar_exprs() -> None:
    """Test type inference on scalar expressions."""
    env: dict[str, Type] = {"x": IntType(), "y": FloatType(), "b": BoolType()}

    assert infer_expr_type(IntLit(42), env) == IntType()
    assert infer_expr_type(FloatLit(3.14), env) == FloatType()
    assert infer_expr_type(BoolLit(True), env) == BoolType()
    assert infer_expr_type(Var("x"), env) == IntType()

    # int + int = int
    assert infer_expr_type(BinOp("+", IntLit(1), IntLit(2)), env) == IntType()
    # int + float = float
    assert infer_expr_type(BinOp("+", Var("x"), Var("y")), env) == FloatType()
    # comparison = bool
    assert infer_expr_type(BinOp("<", IntLit(1), IntLit(2)), env) == BoolType()

    # Min/Max
    assert infer_expr_type(Min([IntLit(1), IntLit(2)]), env) == IntType()
    assert infer_expr_type(Max([IntLit(1), IntLit(2)]), env) == IntType()

    print("  scalar exprs OK")


def _check_tensor_type(actual: Type, elem: Type, dim_values: list[int]) -> None:
    """Assert tensor type matches expected elem type and dim values."""
    assert isinstance(actual, TensorType), f"expected TensorType, got {actual}"
    assert actual.elem_type == elem, f"elem type: {actual.elem_type} != {elem}"
    assert len(actual.dims) == len(dim_values), (
        f"rank: {len(actual.dims)} != {len(dim_values)}"
    )
    for i, (d, v) in enumerate(zip(actual.dims, dim_values)):
        assert isinstance(d, IntLit), f"dim {i} not IntLit: {d}"
        assert d.value == v, f"dim {i}: {d.value} != {v}"


def test_tensor_exprs() -> None:
    """Test type inference on tensor-producing expressions."""
    env: dict[str, Type] = {
        "t": TensorType(FloatType(), [IntLit(4), IntLit(8)]),
        "s": TensorType(FloatType(), [IntLit(8), IntLit(3)]),
        "mask": TensorType(BoolType(), [IntLit(4), IntLit(8)]),
    }

    # Zeros
    z = Zeros([IntLit(2), IntLit(3)])
    _check_tensor_type(infer_expr_type(z, env), FloatType(), [2, 3])

    # Full
    f = Full([IntLit(4)], FloatLit(float("-inf")))
    _check_tensor_type(infer_expr_type(f, env), FloatType(), [4])

    # Arange
    ar = Arange(IntLit(0), IntLit(16))
    art = infer_expr_type(ar, env)
    assert isinstance(art, TensorType)
    assert art.elem_type == IntType()

    # Matmul
    mm = BinOp("@", Var("t"), Var("s"))
    _check_tensor_type(infer_expr_type(mm, env), FloatType(), [4, 3])

    # ReduceSum
    rs = ReduceSum(Var("t"), 1)
    _check_tensor_type(infer_expr_type(rs, env), FloatType(), [4])

    # ReduceMax
    rm = ReduceMax(Var("t"), 0)
    _check_tensor_type(infer_expr_type(rm, env), FloatType(), [8])

    # Triton promotes int1 reductions to int32.
    _check_tensor_type(
        infer_expr_type(ReduceMax(Var("mask"), 1), env), IntType(), [4]
    )
    _check_tensor_type(
        infer_expr_type(ReduceSum(Var("mask"), 1), env), IntType(), [4]
    )

    # Unsqueeze
    u = Unsqueeze(Var("t"), 0)
    _check_tensor_type(infer_expr_type(u, env), FloatType(), [1, 4, 8])

    # Squeeze — need a tensor with a size-1 dim
    env_sq: dict[str, Type] = {
        "s1": TensorType(FloatType(), [IntLit(1), IntLit(8)]),
    }
    sq = Squeeze(Var("s1"), 0)
    _check_tensor_type(infer_expr_type(sq, env_sq), FloatType(), [8])

    # Transpose
    tp = Transpose(Var("t"), [1, 0])
    _check_tensor_type(infer_expr_type(tp, env), FloatType(), [8, 4])

    # BroadcastTo
    bt = BroadcastTo(Var("t"), [IntLit(4), IntLit(8)])
    _check_tensor_type(infer_expr_type(bt, env), FloatType(), [4, 8])

    # TensorIndex (point access) returns scalar
    idx = TensorIndex(Var("t"), [IntLit(1), IntLit(2)])
    assert infer_expr_type(idx, env) == FloatType()

    # --- Tensor-Scalar operations ---
    t_env: dict[str, Type] = {
        "tf": TensorType(FloatType(), [IntLit(4), IntLit(8)]),
        "ti": TensorType(IntType(), [IntLit(4), IntLit(8)]),
        "tb": TensorType(BoolType(), [IntLit(4), IntLit(8)]),
        "x": IntType(),
        "y": FloatType(),
        "b": BoolType(),
    }

    # tensor + scalar -> tensor
    ts_add = infer_expr_type(BinOp("+", Var("tf"), Var("y")), t_env)
    _check_tensor_type(ts_add, FloatType(), [4, 8])

    # scalar + tensor -> tensor
    st_add = infer_expr_type(BinOp("+", Var("y"), Var("tf")), t_env)
    _check_tensor_type(st_add, FloatType(), [4, 8])

    # tensor * scalar -> tensor
    ts_mul = infer_expr_type(BinOp("*", Var("tf"), Var("y")), t_env)
    _check_tensor_type(ts_mul, FloatType(), [4, 8])

    # tensor + int_scalar -> tensor (int promoted to float)
    ts_int = infer_expr_type(BinOp("+", Var("tf"), Var("x")), t_env)
    _check_tensor_type(ts_int, FloatType(), [4, 8])

    # tensor < scalar -> tensor<bool>
    ts_cmp = infer_expr_type(BinOp("<", Var("tf"), Var("y")), t_env)
    _check_tensor_type(ts_cmp, BoolType(), [4, 8])

    # scalar < tensor -> tensor<bool>
    st_cmp = infer_expr_type(BinOp("<", Var("y"), Var("tf")), t_env)
    _check_tensor_type(st_cmp, BoolType(), [4, 8])

    # Where(tensor_bool, scalar_float, scalar_float) -> tensor<float>
    wh = infer_expr_type(Where(Var("tb"), FloatLit(1.0), FloatLit(0.0)), t_env)
    _check_tensor_type(wh, FloatType(), [4, 8])

    # Where(tensor_bool, tensor_float, scalar_float) -> tensor<float>
    wh2 = infer_expr_type(Where(Var("tb"), Var("tf"), FloatLit(0.0)), t_env)
    _check_tensor_type(wh2, FloatType(), [4, 8])

    # Where(tensor_bool, tensor_float, tensor_float) -> tensor<float>
    wh3 = infer_expr_type(Where(Var("tb"), Var("tf"), Var("tf")), t_env)
    _check_tensor_type(wh3, FloatType(), [4, 8])

    print("  tensor exprs OK")


def test_type_errors() -> None:
    """Test that type errors are raised for invalid expressions."""
    env: dict[str, Type] = {
        "tf": TensorType(FloatType(), [IntLit(4), IntLit(8)]),
        "tf2": TensorType(FloatType(), [IntLit(4), IntLit(8)]),
        "tf_other": TensorType(FloatType(), [IntLit(3), IntLit(8)]),
        "ti": TensorType(IntType(), [IntLit(4), IntLit(8)]),
        "t1d": TensorType(FloatType(), [IntLit(4)]),
        "t1_8": TensorType(FloatType(), [IntLit(1), IntLit(8)]),
        "tb": TensorType(BoolType(), [IntLit(4), IntLit(8)]),
        "x": IntType(),
        "y": FloatType(),
        "b": BoolType(),
    }

    def expect_error(expr_or_fn, msg: str) -> None:
        """Assert that evaluating expr_or_fn raises TypeError."""
        try:
            if callable(expr_or_fn):
                _ = expr_or_fn()
            else:
                _ = infer_expr_type(expr_or_fn, env)
            assert False, f"expected TypeError for: {msg}"
        except TypeError:
            pass

    # --- unify errors ---
    eqs: list[ShapeEquation] = []
    # scalar type mismatch
    expect_error(
        lambda: unify(IntType(), FloatType(), eqs, "test"), "unify int vs float"
    )
    # tensor vs scalar
    expect_error(
        lambda: unify(IntType(), TensorType(FloatType(), [IntLit(4)]), eqs, "test"),
        "unify scalar vs tensor",
    )
    # elem type mismatch
    expect_error(
        lambda: unify(
            TensorType(FloatType(), [IntLit(4)]),
            TensorType(IntType(), [IntLit(4)]),
            eqs,
            "test",
        ),
        "unify elem type mismatch",
    )
    # rank mismatch
    expect_error(
        lambda: unify(
            TensorType(FloatType(), [IntLit(4)]),
            TensorType(FloatType(), [IntLit(4), IntLit(8)]),
            eqs,
            "test",
        ),
        "unify rank mismatch",
    )
    # constant dim conflict
    expect_error(
        lambda: unify(
            TensorType(FloatType(), [IntLit(4)]),
            TensorType(FloatType(), [IntLit(5)]),
            eqs,
            "test",
        ),
        "unify dim mismatch 4 vs 5",
    )

    # --- BinOp errors ---
    # @ on non-tensor
    expect_error(BinOp("@", Var("x"), Var("tf")), "@ with non-tensor lhs")
    # @ on 1-D tensor
    expect_error(BinOp("@", Var("t1d"), Var("tf")), "@ on 1-D tensor")
    # @ inner dim mismatch
    expect_error(BinOp("@", Var("tf"), Var("tf2")), "@ inner dim mismatch")
    # cdiv with float operand
    expect_error(BinOp("cdiv", Var("y"), IntLit(2)), "cdiv with float lhs")
    expect_error(BinOp("cdiv", IntLit(2), Var("y")), "cdiv with float rhs")
    # 'and' with non-bool
    expect_error(BinOp("and", Var("x"), Var("b")), "and with int lhs")
    expect_error(BinOp("and", Var("b"), Var("x")), "and with int rhs")

    # --- Min/Max with non-int ---
    expect_error(Min([Var("y")]), "min with float")
    expect_error(Max([Var("y")]), "max with float")

    # --- Zeros/Full shape errors ---
    expect_error(Zeros([Var("y")]), "zeros with float dim")
    expect_error(Full([Var("y")], FloatLit(0.0)), "full with float dim")
    # Full with tensor value
    expect_error(Full([IntLit(4)], Var("tf")), "full with tensor value")

    # --- Arange errors ---
    expect_error(Arange(Var("y"), IntLit(10)), "arange with float start")
    expect_error(Arange(IntLit(0), Var("y")), "arange with float stop")

    # --- Where errors ---
    # non-bool tensor condition (int tensor)
    expect_error(
        Where(Var("ti"), IntLit(1), IntLit(2)),
        "where with int tensor cond",
    )
    # scalar condition (not tensor)
    expect_error(
        Where(Var("b"), IntLit(1), IntLit(2)),
        "where with scalar bool cond",
    )
    # non-tensor condition (scalar int)
    expect_error(
        Where(Var("x"), IntLit(1), IntLit(2)),
        "where with scalar int cond",
    )
    # branch type mismatch
    expect_error(
        Where(Var("tb"), IntLit(1), FloatLit(2.0)),
        "where branch scalar type mismatch",
    )

    # --- Reduce on non-tensor ---
    expect_error(ReduceSum(Var("x"), 0), "reduce_sum on scalar")
    expect_error(ReduceMax(Var("x"), 0), "reduce_max on scalar")
    # axis out of range
    expect_error(ReduceSum(Var("tf"), 5), "reduce_sum axis out of range")
    expect_error(ReduceMax(Var("tf"), -1), "reduce_max negative axis")

    # --- Exp2 with non-float ---
    expect_error(Exp2(Var("x")), "exp2 with int")
    expect_error(Exp2(Var("ti")), "exp2 with tensor<int>")

    # --- Maximum type mismatch ---
    expect_error(
        Maximum(Var("tf"), Var("ti")),
        "maximum float vs int tensor",
    )

    # --- Unsqueeze/Squeeze errors ---
    expect_error(Unsqueeze(Var("x"), 0), "unsqueeze on scalar")
    expect_error(Unsqueeze(Var("tf"), 5), "unsqueeze axis out of range")
    expect_error(Squeeze(Var("x"), 0), "squeeze on scalar")
    expect_error(Squeeze(Var("tf"), 5), "squeeze axis out of range")
    # squeeze on non-1 dim
    expect_error(Squeeze(Var("tf"), 0), "squeeze dim 4 != 1")

    # --- Transpose errors ---
    expect_error(Transpose(Var("x"), [0, 1]), "transpose on scalar")
    expect_error(Transpose(Var("tf"), [0, 0]), "transpose bad permutation")

    # --- BroadcastTo errors ---
    expect_error(BroadcastTo(Var("x"), [IntLit(4)]), "broadcast_to on scalar")
    # rank mismatch
    expect_error(
        BroadcastTo(Var("tf"), [IntLit(4)]),
        "broadcast_to rank mismatch",
    )
    # incompatible dim (not 1 and not matching)
    expect_error(
        BroadcastTo(Var("tf"), [IntLit(4), IntLit(16)]),
        "broadcast_to dim incompatible",
    )

    # --- TensorView errors ---
    # index on non-tensor
    expect_error(
        TensorView(Var("x"), [Slice(IntLit(0), IntLit(1))]),
        "index on scalar",
    )
    # wrong number of slices
    expect_error(
        TensorView(Var("tf"), [Slice(IntLit(0), IntLit(1))]),
        "index region rank mismatch",
    )
    # non-int slice bounds
    expect_error(
        TensorView(
            Var("tf"),
            [Slice(FloatLit(0.0), IntLit(1)), Slice(IntLit(0), IntLit(1))],
        ),
        "index slice with float start",
    )

    # --- TensorIndex errors ---
    # index on non-tensor
    expect_error(
        TensorIndex(Var("x"), [IntLit(0)]),
        "index on scalar",
    )
    # wrong number of axes
    expect_error(
        TensorIndex(Var("tf"), [IntLit(0)]),
        "index rank mismatch",
    )
    # non-int axis
    expect_error(
        TensorIndex(Var("tf"), [FloatLit(0.0), IntLit(1)]),
        "index axis with float",
    )

    # --- Undefined variable ---
    expect_error(Var("undefined"), "undefined variable")

    print("  type errors OK")


def _build_let_test_kernel(body_stmts: list[Stmt]) -> Kernel:
    """Construct a minimal kernel for let-binding tests.

    Params: M (int), N (int), B (int)
    Grid iters: grid_i in 0..1 (trivial)
    Decls: none
    """
    return Kernel(
        name="let_test",
        params=[
            Param("M", IntType()),
            Param("N", IntType()),
            Param("B", IntType()),
        ],
        grid=Grid(
            iters=[GridIter(Var("grid_i"), Range(IntLit(0), IntLit(1)))],
            decls=[],
            body=body_stmts,
        ),
    )


def test_let_bindings() -> None:
    """Tests for Let typing and expansion."""

    # ------------------------------------------------------------------
    # 1. Happy path: chained lets mirroring the user's example.
    # ------------------------------------------------------------------
    #   for(i = 0..M):
    #     for(j = 0..N):
    #       let foo = i + j
    #       let bar = foo * B       # bar expands to (i + j) * B
    # ------------------------------------------------------------------
    M, N, B = v_("M"), v_("N"), v_("B")
    i, j = v_("i"), v_("j")

    inner_body = [
        let_("foo", add_(i, j)),
        let_("bar", mul_(v_("foo"), B)),
    ]
    for_j = For(var=j, iters=Range(lit_(0), N), body=inner_body)
    for_i = For(var=i, iters=Range(lit_(0), M), body=[for_j])

    kernel = _build_let_test_kernel([for_i])
    typed, _ = infer_types(kernel)
    check_kernel_fully_typed(typed)

    # After expansion there should be no Let nodes in the body.
    expanded = expand_let_bindings(typed)

    def _collect_let_binds(stmts):
        found = []
        for s in stmts:
            match s:
                case Let():
                    _ = found.append(s)
                case For(body=body):
                    found += _collect_let_binds(body)
                case If(then_body=then_body, else_body=else_body):
                    found += _collect_let_binds(then_body)
                    found += _collect_let_binds(else_body)
                case Assign():
                    pass
                case _:
                    raise AssertionError(f"unhandled stmt: {s}")
        return found

    assert _collect_let_binds(expanded.grid.body) == [], (
        "expand_let_bindings should remove all Let nodes"
    )

    # The innermost For body should now have zero statements (both lets gone).
    inner_for = expanded.grid.body[0]
    assert isinstance(inner_for, For)
    inner_inner_for = inner_for.body[0]
    assert isinstance(inner_inner_for, For)
    assert inner_inner_for.body == [], (
        "all let stmts should be gone from the expanded inner body"
    )

    print("  let_bindings happy path OK")

    # ------------------------------------------------------------------
    # 2. Expansion substitution correctness.
    # ------------------------------------------------------------------
    #   let foo = i + j
    #   let bar = foo * B    # should expand to (i + j) * B, not Var("foo") * B
    # Build a single-loop kernel that uses the let vars in an assignment.
    # We use a trivial decl to give the assignment target a home.
    acc = v_("acc")
    n_param = v_("N")
    kernel2 = Kernel(
        name="let_subst",
        params=[
            Param("N", IntType()),
            Param("B", IntType()),
        ],
        grid=Grid(
            iters=[GridIter(v_("i"), Range(lit_(0), n_param))],
            decls=[VarDecl(acc, IntType())],
            body=[
                let_("foo", v_("i")),
                let_("bar", mul_(v_("foo"), v_("B"))),
                Assign(target=acc, op=None, value=v_("bar")),
            ],
        ),
    )
    typed2, _ = infer_types(kernel2)
    expanded2 = expand_let_bindings(typed2)

    # The single remaining statement should be:  acc = i * B
    assert len(expanded2.grid.body) == 1
    assign = expanded2.grid.body[0]
    assert isinstance(assign, Assign)
    # value should be i * B  (BinOp("*", Var("i"), Var("B")))
    val = assign.value
    assert isinstance(val, BinOp) and val.op == "*", f"expected i*B, got {val}"
    assert isinstance(val.lhs, Var) and val.lhs.name == "i", (
        f"expected lhs=i, got {val.lhs}"
    )
    assert isinstance(val.rhs, Var) and val.rhs.name == "B", (
        f"expected rhs=B, got {val.rhs}"
    )

    print("  let_bindings substitution correctness OK")

    # ------------------------------------------------------------------
    # 3. Error: redeclaration of a let name in the same scope.
    # check_variable_names (not infer_types) catches duplicate Let names.
    # ------------------------------------------------------------------
    kernel_redecl = _build_let_test_kernel(
        [
            let_("foo", v_("M")),
            let_("foo", v_("N")),  # error: re-declaration
        ]
    )
    try:
        _ = check_variable_names(kernel_redecl)
        assert False, "expected ValueError for re-declaration"
    except ValueError:
        pass
    print("  let_bindings redeclaration error OK")

    # ------------------------------------------------------------------
    # 4. Error: non-int let binding (floats not allowed).
    # ------------------------------------------------------------------
    kernel_float = Kernel(
        name="let_float",
        params=[Param("scale", FloatType())],
        grid=Grid(
            iters=[GridIter(v_("i"), Range(lit_(0), lit_(1)))],
            decls=[],
            body=[let_("foo", v_("scale"))],
        ),
    )
    try:
        _ = infer_types(kernel_float)
        assert False, "expected TypeError for float let binding"
    except TypeError:
        pass
    print("  let_bindings non-int error OK")

    # ------------------------------------------------------------------
    # 5. Error: assignment to a let-bound name.
    #    check_variable_names provides the role_map; infer_types uses it
    #    to guard against writing to let-bound names.
    # ------------------------------------------------------------------
    kernel_assign = _build_let_test_kernel(
        [
            let_("foo", v_("M")),
            Assign(target=v_("foo"), op=None, value=v_("N")),  # error: assign to let
        ]
    )
    try:
        _assign_roles = check_variable_names(kernel_assign)
        _ = infer_types(kernel_assign, _assign_roles)
        assert False, "expected TypeError for assign to let-bound name"
    except TypeError:
        pass
    print("  let_bindings assign-to-let error OK")

    # ------------------------------------------------------------------
    # 6. Error: let name shadows an existing param.
    # With the global-env model, check_variable_names (not infer_types)
    # is responsible for catching cross-scope name conflicts.
    # ------------------------------------------------------------------
    kernel_shadow = _build_let_test_kernel(
        [
            let_("M", v_("N")),  # error: M is already a param
        ]
    )
    try:
        _ = check_variable_names(kernel_shadow)
        assert False, "expected ValueError for let shadowing param"
    except ValueError:
        pass
    print("  let_bindings shadow-param error OK")

    # Positive check: roles dict for a well-formed kernel.
    from tests.fixtures import matmul_kernel

    roles = check_variable_names(matmul_kernel())
    assert roles["a"] == "param"
    assert roles["M"] == "param_size"
    assert roles["i"] == "grid"
    assert roles["acc"] == "local_var"
    assert roles["k"] == "loop_var"
    print("  check_variable_names roles dict OK")

    # ------------------------------------------------------------------
    # 7. A let binding inside a For body is not visible after the loop.
    # Type inference re-checks lexical scope rather than relying on a caller.
    # ------------------------------------------------------------------
    acc2 = v_("acc2")
    kernel_scope = Kernel(
        name="let_scope",
        params=[Param("M", IntType())],
        grid=Grid(
            iters=[GridIter(v_("i"), Range(lit_(0), v_("M")))],
            decls=[VarDecl(acc2, IntType())],
            body=[
                For(
                    var=v_("j"),
                    iters=Range(lit_(0), lit_(1)),
                    body=[let_("inner", v_("i"))],
                ),
                Assign(target=acc2, op=None, value=v_("inner")),
            ],
        ),
    )
    with pytest.raises(ValueError, match="out of scope"):
        infer_types(kernel_scope)
    print("  let_bindings lexical scope OK")

    print("  test_let_bindings PASSED")


def main() -> None:
    print("=== Type Inference Tests ===")

    print("test_scalar_exprs...")
    test_scalar_exprs()

    print("test_tensor_exprs...")
    test_tensor_exprs()

    print("test_type_errors...")
    test_type_errors()

    print("test_matmul...")
    test_matmul()

    print("test_fattn...")
    test_fattn()

    print("test_let_bindings...")
    test_let_bindings()

    print("\nAll type inference tests PASSED!")


def test_specialization_rejects_unknown_names() -> None:
    with pytest.raises(ValueError, match="not kernel parameters or tensor dimensions"):
        specialize_kernel_constants(matmul_kernel(), {"NOT_A_LAUNCH_CONSTANT": 1})


def test_let_expansion_does_not_leak_across_branches() -> None:
    out = v_("out")
    kernel = Kernel(
        name="branch_scope",
        params=[],
        grid=Grid(
            iters=[],
            decls=[VarDecl(out, IntType())],
            body=[
                If(
                    cond=lit_(True),
                    then_body=[let_("branch_only", lit_(1))],
                    else_body=[
                        Assign(target=out, op=None, value=v_("branch_only"))
                    ],
                )
            ],
        ),
    )

    expanded = expand_let_bindings(kernel)
    branch = expanded.grid.body[0]
    assert isinstance(branch, If)
    assert branch.then_body == []
    assert isinstance(branch.else_body[0], Assign)
    assert branch.else_body[0].value == v_("branch_only")


if __name__ == "__main__":
    main()
