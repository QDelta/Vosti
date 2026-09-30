import pytest

from ir import *
from tests.fixtures import matmul_kernel
from ir.pp import pretty_kernel
from ir.subst import specialize_kernel_constants, expand_let_bindings
from ir.typ import infer_types
from ir.regions import (
    _singleton_index,
    bound_variable_regions,
    collect_write_stmts,
    ConditionalRegion,
    order_write_stmts,
    plan_write_stmts,
)


def test_isolated_direct_write_is_not_dropped():
    vec = TensorType(FloatType(), [IntLit(4)])
    kernel = Kernel(
        name="direct_write",
        params=[Param("x", vec), Param("o", vec)],
        grid=Grid(
            iters=[],
            decls=[],
            body=[Assign(target=v_("o"), op=None, value=v_("x"))],
        ),
    )
    kernel, _ = infer_types(kernel)
    regions, writes = bound_variable_regions(
        kernel, "o", [Slice(IntLit(1), IntLit(2))]
    )
    assert "x" in regions
    assert _singleton_index(regions["x"][0]) is not None
    assert writes


def test_write_tracks_only_its_enclosing_local_loops():
    vec = TensorType(FloatType(), [IntLit(4)])
    kernel = Kernel(
        name="loop_scopes",
        params=[Param("x", vec), Param("o", vec)],
        grid=Grid(
            iters=[GridIter(v_("program"), range_(lit_(0), lit_(4)))],
            decls=[],
            body=[
                For(
                    var=v_("first"),
                    iters=range_(lit_(0), lit_(4)),
                    body=[Assign(target=v_("o"), op=None, value=v_("x"))],
                ),
                For(
                    var=v_("second"),
                    iters=range_(lit_(0), lit_(4)),
                    body=[Assign(target=v_("o"), op=None, value=v_("x"))],
                ),
            ],
        ),
    )
    writes = collect_write_stmts(kernel)
    assert [write.iterators for write in writes] == [
        frozenset({"first"}),
        frozenset({"second"}),
    ]


def test_single_variable_matmul_recurrence_is_rejected():
    """A self-loop is a cyclic SCC even when it contains only one write."""
    two_by_two = TensorType(FloatType(), [IntLit(2), IntLit(2)])
    kernel = Kernel(
        name="matmul_recurrence",
        params=[
            Param("w", two_by_two),
            Param("o", two_by_two),
        ],
        grid=Grid(
            iters=[GridIter(v_("i"), range_(lit_(0), lit_(1)))],
            decls=[VarDecl(v_("acc"), two_by_two)],
            body=[
                Assign(target=v_("acc"), op=None, value=mm_(v_("acc"), v_("w"))),
                Assign(target=v_("o"), op=None, value=v_("acc")),
            ],
        ),
    )
    kernel, _ = infer_types(kernel)
    with pytest.raises(AssertionError, match="cyclic vars in matmul lhs"):
        order_write_stmts(kernel, {"w": two_by_two, "o": two_by_two,
                                   "acc": two_by_two, "i": IntType()})


def test_elementwise_cycle_is_walked_in_reverse_source_order():
    """The first write in an SCC must see demand propagated by the last."""
    vec = TensorType(FloatType(), [IntLit(4)])
    kernel = Kernel(
        name="elementwise_recurrence",
        params=[Param("data", vec), Param("o", vec)],
        grid=Grid(
            iters=[GridIter(v_("i"), range_(lit_(0), lit_(1)))],
            decls=[VarDecl(v_("state"), vec), VarDecl(v_("next_state"), vec)],
            body=[
                Assign(
                    target=v_("next_state"), op=None,
                    value=add_(v_("state"), v_("data")),
                ),
                Assign(target=v_("state"), op=None, value=v_("next_state")),
                Assign(target=v_("o"), op=None, value=v_("state")),
            ],
        ),
    )
    kernel, _ = infer_types(kernel)
    regions, _ = bound_variable_regions(
        kernel, "o", [Slice(IntLit(2), IntLit(3))]
    )
    assert "data" in regions
    assert _singleton_index(regions["data"][0]) is not None


def test_compound_assignment_records_preupdate_recurrence():
    vec = TensorType(FloatType(), [IntLit(4)])
    kernel = Kernel(
        name="compound_recurrence",
        params=[Param("data", vec), Param("o", vec)],
        grid=Grid(
            iters=[],
            decls=[VarDecl(v_("state"), vec)],
            body=[
                Assign(target=v_("state"), op="+", value=v_("data")),
                Assign(target=v_("o"), op=None, value=v_("state")),
            ],
        ),
    )
    kernel, _ = infer_types(kernel)
    plan = plan_write_stmts(
        kernel,
        {"data": vec, "o": vec, "state": vec},
    )
    assert len(plan.recurrences) == 1
    assert plan.recurrences[0].variables == frozenset({"state"})
    assert plan.recurrences[0].writes[0].write.op == "+"


def pp_slice(sl: Slice) -> str:
    from ir.pp import pp_expr

    return f"{pp_expr(sl.start)}:{pp_expr(sl.stop)}"


def pp_regions(regions: dict[str, Region], written: list[ConditionalRegion]) -> None:
    print("=== Computed regions ===")
    for var, region in sorted(regions.items()):
        slices_str = ", ".join(f"{pp_slice(sl)}" for sl in region)
        print(f"  {var}: [{slices_str}]")

    print()
    print("=== Output written region ===")
    from ir.pp import pp_expr_no_paren

    for cr in written:
        region_str = ", ".join(pp_slice(sl) for sl in cr.region)
        if cr.conditions:
            conds_str = " and ".join(pp_expr_no_paren(c) for c in cr.conditions)
            print(f"  [{region_str}] when {conds_str}")
        else:
            print(f"  [{region_str}]")


def test_scalar_regions_preserve_rank0_and_omit_unrelated_tensor():
    print("--- test_scalar_regions_preserve_rank0_and_omit_unrelated_tensor ---")
    kernel = Kernel(
        name="scalar_copy",
        params=[
            Param("a", TensorType(IntType(), [])),
            Param("b", TensorType(IntType(), [IntLit(4)])),
            Param("o", TensorType(IntType(), [])),
        ],
        grid=Grid(
            iters=[GridIter(v_("i"), range_(lit_(0), lit_(1)))],
            decls=[VarDecl(v_("tmp"), TensorType(IntType(), []))],
            body=[
                Assign(target=v_("tmp"), op=None, value=v_("a")),
                Assign(target=v_("o"), op=None, value=v_("tmp")),
            ],
        ),
    )
    kernel, _ = infer_types(kernel)

    regions, written = bound_variable_regions(kernel, "o", [])
    pp_regions(regions, written)

    assert regions["o"] == []
    assert regions["a"] == []
    assert "b" not in regions
    assert len(written) == 1
    assert written[0].region == []
    print("  scalar rank-0 regions OK")


def test_untouched_tensor_region_is_omitted():
    print("--- test_untouched_tensor_region_is_omitted ---")
    kernel = Kernel(
        name="vector_copy",
        params=[
            Param("a", TensorType(IntType(), [IntLit(4)])),
            Param("b", TensorType(IntType(), [IntLit(4)])),
            Param("o", TensorType(IntType(), [IntLit(4)])),
        ],
        grid=Grid(
            iters=[GridIter(v_("i"), range_(lit_(0), lit_(1)))],
            decls=[VarDecl(v_("tmp"), TensorType(IntType(), [IntLit(4)]))],
            body=[
                Assign(target=v_("tmp"), op=None, value=v_("a")),
                Assign(target=v_("o"), op=None, value=v_("tmp")),
            ],
        ),
    )
    kernel, _ = infer_types(kernel)

    full_region = [Slice(IntLit(0), IntLit(4))]
    regions, written = bound_variable_regions(kernel, "o", full_region)
    pp_regions(regions, written)

    assert regions["o"] == full_region
    assert regions["a"] == full_region
    assert regions["tmp"] == full_region
    assert "b" not in regions
    assert len(written) == 1
    assert written[0].region == full_region
    print("  untouched tensor omitted OK")


def test_matmul_regions():
    print("--- test_matmul_regions ---")
    kernel = specialize_kernel_constants(
        matmul_kernel(),
        {
            "BLOCK_M": 128,
            "BLOCK_N": 128,
            "BLOCK_K": 32,
        },
    )
    kernel = expand_let_bindings(kernel)
    kernel, _ = infer_types(kernel)
    print(pretty_kernel(kernel))
    print()
    x = Var("x")

    # Output region: c[x:x+1, 0:N]
    output_region = [
        Slice(x, BinOp("+", x, IntLit(1))),
        Slice(IntLit(0), Var("N")),
    ]

    regions, written = bound_variable_regions(kernel, "c", output_region)
    pp_regions(regions, written)


if __name__ == "__main__":
    test_scalar_regions_preserve_rank0_and_omit_unrelated_tensor()
    test_untouched_tensor_region_is_omitted()
    test_matmul_regions()
