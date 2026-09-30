"""Tests for TensorIndex support: uninterpreted functions, read-only checks, Z3 translation."""

from ir import *
from tests.fixtures import matmul_kernel
from ir.pp import pretty_kernel, pp_expr
from ir.preprocess import (
    check_variable_names,
    build_type_env,
    collect_readonly_tensors,
    check_tensorindex_readonly,
)
from ir.subst import specialize_kernel_constants, expand_let_bindings
from ir.typ import infer_types
from ir.regions import bound_variable_regions
import z3


# ---------------------------------------------------------------------------
# collect_readonly_tensors
# ---------------------------------------------------------------------------


def test_collect_readonly_tensors_matmul():
    """In matmul, a and b are read-only; c is written."""
    kernel = matmul_kernel()
    readonly = collect_readonly_tensors(kernel)
    assert "a" in readonly, "a should be read-only"
    assert "b" in readonly, "b should be read-only"
    assert "c" not in readonly, "c is written"
    # Local vars that are written should not be in the set.
    assert "acc" not in readonly
    assert "block_a" not in readonly
    assert "block_b" not in readonly
    print("  collect_readonly_tensors matmul OK")


def test_collect_readonly_tensors_minimal():
    """Test with a hand-built minimal kernel."""
    # Kernel with:
    #   param t: tensor<int>[N]    (never written -> read-only)
    #   param s: tensor<int>[N]    (written via plain Var -> NOT read-only)
    #   decl local: tensor<int>[4] (written via TensorView -> NOT read-only)
    #   decl buf: tensor<int>[4]   (never written -> read-only)
    N = v_("N")
    kernel = Kernel(
        name="test_readonly",
        params=[
            Param("t", TensorType(IntType(), [N])),
            Param("s", TensorType(IntType(), [N])),
        ],
        grid=Grid(
            iters=[GridIter(v_("i"), range_(lit_(0), lit_(1)))],
            decls=[
                VarDecl(v_("local"), TensorType(IntType(), [IntLit(4)])),
                VarDecl(v_("buf"), TensorType(IntType(), [IntLit(4)])),
            ],
            body=[
                Assign(target=v_("s"), op=None, value=v_("t")),
                Assign(
                    target=view_(v_("local"), [slice_(lit_(0), lit_(4))]),
                    op=None,
                    value=v_("t"),
                ),
            ],
        ),
    )
    readonly = collect_readonly_tensors(kernel)
    assert "t" in readonly, "t is never written"
    assert "buf" in readonly, "buf is never written"
    assert "s" not in readonly, "s is written (plain Var target)"
    assert "local" not in readonly, "local is written (TensorView target)"
    # Non-tensor params should not appear.
    assert "N" not in readonly, "N is a scalar param, not a tensor"
    assert "i" not in readonly, "i is a grid iter var, not a tensor"
    print("  collect_readonly_tensors minimal OK")


# ---------------------------------------------------------------------------
# check_tensorindex_readonly
# ---------------------------------------------------------------------------


def test_check_tensorindex_readonly_matmul():
    """Matmul has no TensorIndex, so the check trivially passes."""
    kernel = matmul_kernel()
    check_tensorindex_readonly(kernel)  # should not raise
    print("  check_tensorindex_readonly matmul OK")


def test_check_tensorindex_readonly_error():
    """Error when TensorIndex base is a written tensor."""
    # Kernel: param t: tensor<int>[N]
    #   t[...] = ...            # t is written
    #   x = t[i]                # TensorIndex on written t -> error
    N = v_("N")
    kernel = Kernel(
        name="bad_tensorindex",
        params=[
            Param("t", TensorType(IntType(), [N])),
        ],
        grid=Grid(
            iters=[GridIter(v_("i"), range_(lit_(0), N))],
            decls=[VarDecl(v_("x"), IntType())],
            body=[
                Assign(
                    target=view_(v_("t"), [slice_(lit_(0), N)]),
                    op=None,
                    value=v_("t"),
                ),
                Assign(
                    target=v_("x"),
                    op=None,
                    value=index_(v_("t"), [v_("i")]),
                ),
            ],
        ),
    )
    try:
        check_tensorindex_readonly(kernel)
        assert False, "expected ValueError for written tensor in TensorIndex"
    except ValueError as e:
        assert "t" in str(e), f"error should mention 't': {e}"
    print("  check_tensorindex_readonly error OK")


def test_check_tensorindex_readonly_nested():
    """Error when TensorIndex in indices of another access uses a written tensor.

    Scenario: a[b[i]] where b is written -> should fail on b.
    """
    N = v_("N")
    kernel = Kernel(
        name="nested_tensorindex",
        params=[
            Param("a", TensorType(IntType(), [N])),
            Param("b", TensorType(IntType(), [N])),
        ],
        grid=Grid(
            iters=[GridIter(v_("i"), range_(lit_(0), N))],
            decls=[VarDecl(v_("x"), IntType())],
            body=[
                # b is written
                Assign(
                    target=view_(v_("b"), [slice_(lit_(0), N)]),
                    op=None,
                    value=v_("a"),
                ),
                # a[b[i]] — a is read-only (OK), b is written (ERROR)
                Assign(
                    target=v_("x"),
                    op=None,
                    value=index_(v_("a"), [index_(v_("b"), [v_("i")])]),
                ),
            ],
        ),
    )
    try:
        check_tensorindex_readonly(kernel)
        assert False, "expected ValueError for written 'b' in nested TensorIndex"
    except ValueError as e:
        assert "b" in str(e), f"error should mention 'b': {e}"
        # 'a' should NOT be in the error since it's read-only
        assert "'a'" not in str(e), f"error should not mention 'a': {e}"
    print("  check_tensorindex_readonly nested OK")


def test_check_tensorindex_readonly_in_slice():
    """TensorIndex in slice bounds: a[b[i]:b[i+1]] where b is read-only passes."""
    N = v_("N")
    kernel = Kernel(
        name="tensorindex_in_slice",
        params=[
            Param("a", TensorType(IntType(), [N])),
            Param("b", TensorType(IntType(), [N])),
        ],
        grid=Grid(
            iters=[GridIter(v_("i"), range_(lit_(0), N))],
            decls=[VarDecl(v_("out"), TensorType(IntType(), [IntLit(1)]))],
            body=[
                # a[b[i]:b[i]+1] — both a and b are read-only
                Assign(
                    target=v_("out"),
                    op=None,
                    value=view_(
                        v_("a"),
                        [
                            slice_(
                                index_(v_("b"), [v_("i")]),
                                add_(index_(v_("b"), [v_("i")]), lit_(1)),
                            )
                        ],
                    ),
                ),
            ],
        ),
    )
    check_tensorindex_readonly(kernel)  # should not raise
    print("  check_tensorindex_readonly in_slice OK")


# ---------------------------------------------------------------------------
# expr_to_z3 with TensorIndex
# ---------------------------------------------------------------------------


def test_expr_to_z3_tensorindex_1d():
    """TensorIndex on a 1D tensor becomes f(idx)."""
    from ir.smt import expr_to_z3

    f = z3.Function("arr", z3.IntSort(), z3.IntSort())
    env = {"arr": f, "i": z3.Int("i")}

    # arr[i]
    expr = index_(v_("arr"), [v_("i")])
    result = expr_to_z3(expr, env)  # pyright: ignore[reportArgumentType]
    expected = f(z3.Int("i"))
    assert z3.is_true(z3.simplify(result == expected)), (
        f"expected {expected}, got {result}"
    )
    print("  expr_to_z3 tensorindex 1D OK")


def test_expr_to_z3_tensorindex_2d():
    """TensorIndex on a 2D tensor becomes f(idx0, idx1)."""
    from ir.smt import expr_to_z3

    f = z3.Function("mat", z3.IntSort(), z3.IntSort(), z3.IntSort())
    env = {"mat": f, "r": z3.Int("r"), "c": z3.Int("c")}

    # mat[r, c]
    expr = index_(v_("mat"), [v_("r"), v_("c")])
    result = expr_to_z3(expr, env)  # pyright: ignore[reportArgumentType]
    expected = f(z3.Int("r"), z3.Int("c"))
    assert z3.is_true(z3.simplify(result == expected)), (
        f"expected {expected}, got {result}"
    )
    print("  expr_to_z3 tensorindex 2D OK")


def test_expr_to_z3_tensorindex_in_arithmetic():
    """TensorIndex result participates in arithmetic: arr[i] + 1."""
    from ir.smt import expr_to_z3

    f = z3.Function("arr", z3.IntSort(), z3.IntSort())
    env = {"arr": f, "i": z3.Int("i")}

    # arr[i] + 1
    expr = add_(index_(v_("arr"), [v_("i")]), lit_(1))
    result = expr_to_z3(expr, env)  # pyright: ignore[reportArgumentType]
    expected = f(z3.Int("i")) + 1  # pyright: ignore[reportOperatorIssue]
    assert z3.is_true(z3.simplify(result == expected)), (
        f"expected {expected}, got {result}"
    )
    print("  expr_to_z3 tensorindex arithmetic OK")


def test_expr_to_z3_tensorindex_complex_index():
    """TensorIndex with computed index: arr[i + 1]."""
    from ir.smt import expr_to_z3

    f = z3.Function("arr", z3.IntSort(), z3.IntSort())
    env = {"arr": f, "i": z3.Int("i")}

    # arr[i + 1]
    expr = index_(v_("arr"), [add_(v_("i"), lit_(1))])
    result = expr_to_z3(expr, env)  # pyright: ignore[reportArgumentType]
    expected = f(z3.Int("i") + 1)
    assert z3.is_true(z3.simplify(result == expected)), (
        f"expected {expected}, got {result}"
    )
    print("  expr_to_z3 tensorindex complex index OK")


def test_expr_to_z3_shared_function():
    """Same Z3 Function in two envs gives congruence: f(x) == f(x)."""
    from ir.smt import expr_to_z3, z3_prove

    f = z3.Function("cu", z3.IntSort(), z3.IntSort())
    x = z3.Int("x")
    left_env = {"cu": f, "bi": z3.Int("bi_L"), "x": x}
    right_env = {"cu": f, "bi": z3.Int("bi_R"), "x": x}

    # When bi_L == bi_R, cu[bi_L] == cu[bi_R] holds by congruence.
    expr = index_(v_("cu"), [v_("bi")])
    left_val = expr_to_z3(expr, left_env)  # pyright: ignore[reportArgumentType]
    right_val = expr_to_z3(expr, right_env)  # pyright: ignore[reportArgumentType]

    check = z3_prove(
        "congruence",
        [left_env["bi"] == right_env["bi"]],
        left_val == right_val,
    )
    assert check.proved, f"congruence should hold: {check.details}"
    print("  expr_to_z3 shared function congruence OK")


def test_expr_to_z3_different_indices():
    """Different indices give different (unrelated) results."""
    from ir.smt import expr_to_z3

    f = z3.Function("arr", z3.IntSort(), z3.IntSort())
    env = {"arr": f, "i": z3.Int("i"), "j": z3.Int("j")}

    expr_i = index_(v_("arr"), [v_("i")])
    expr_j = index_(v_("arr"), [v_("j")])
    val_i = expr_to_z3(expr_i, env)  # pyright: ignore[reportArgumentType]
    val_j = expr_to_z3(expr_j, env)  # pyright: ignore[reportArgumentType]

    # arr[i] == arr[j] should NOT be provable without i == j
    solver = z3.Solver()
    solver.add(z3.Not(val_i == val_j))
    assert solver.check() == z3.sat, "arr[i] != arr[j] should be satisfiable"

    # But arr[i] == arr[j] should be provable when i == j
    solver2 = z3.Solver()
    solver2.add(env["i"] == env["j"])
    solver2.add(z3.Not(val_i == val_j))
    assert solver2.check() == z3.unsat, "arr[i] == arr[j] should hold when i == j"
    print("  expr_to_z3 different indices OK")


# ---------------------------------------------------------------------------
# Pretty-printing TensorIndex expressions
# ---------------------------------------------------------------------------


def test_pp_tensorindex():
    """pp_expr renders TensorIndex correctly."""
    # 1D: arr[i]
    expr1 = index_(v_("arr"), [v_("i")])
    assert pp_expr(expr1) == "arr[i]"

    # 2D: mat[i, j]
    expr2 = index_(v_("mat"), [v_("i"), v_("j")])
    assert pp_expr(expr2) == "mat[i, j]"

    # Complex index: arr[i + 1]
    expr3 = index_(v_("arr"), [add_(v_("i"), lit_(1))])
    assert pp_expr(expr3) == "arr[i + 1]"

    # TensorIndex in arithmetic: arr[i] + 1
    expr4 = add_(index_(v_("arr"), [v_("i")]), lit_(1))
    result = pp_expr(expr4)
    assert "arr[i]" in result, f"expected arr[i] in: {result}"
    assert "+ 1" in result, f"expected + 1 in: {result}"

    # Nested: arr[brr[i]]
    expr5 = index_(v_("arr"), [index_(v_("brr"), [v_("i")])])
    assert pp_expr(expr5) == "arr[brr[i]]"

    print("  pp_tensorindex OK")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    print("=== TensorIndex Tests ===")

    print("test_collect_readonly_tensors...")
    test_collect_readonly_tensors_matmul()
    test_collect_readonly_tensors_minimal()

    print("test_check_tensorindex_readonly...")
    test_check_tensorindex_readonly_matmul()
    test_check_tensorindex_readonly_error()
    test_check_tensorindex_readonly_nested()
    test_check_tensorindex_readonly_in_slice()

    print("test_expr_to_z3_tensorindex...")
    test_expr_to_z3_tensorindex_1d()
    test_expr_to_z3_tensorindex_2d()
    test_expr_to_z3_tensorindex_in_arithmetic()
    test_expr_to_z3_tensorindex_complex_index()
    test_expr_to_z3_shared_function()
    test_expr_to_z3_different_indices()

    print("test_pp_tensorindex...")
    test_pp_tensorindex()

    print("\nAll TensorIndex tests PASSED!")


if __name__ == "__main__":
    main()
