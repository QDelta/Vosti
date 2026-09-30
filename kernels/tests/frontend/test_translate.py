"""
Tests for the Triton-to-IR translator and subset validator.

Validates:
  1. The translator produces structurally correct IR for matmul and fattn kernels
  2. The subset validator accepts valid kernels and rejects invalid patterns
  3. The translated IR can be pretty-printed (round-trip sanity check)
"""

from __future__ import annotations

from pathlib import Path
import textwrap

import pytest

from ir import (
    Kernel,
    Grid,
    GridIter,
    Param,
    For,
    Assign,
    Let,
    If,
    MaskedLoad,
    MaskedStore,
    Squeeze,
    Var,
    BinOp,
    Zeros,
    Full,
    Arange,
    Where,
    ReduceMax,
    ReduceSum,
    Exp2,
    Maximum,
    Transpose,
    BroadcastTo,
    Unsqueeze,
    TensorIndex,
    TensorType,
    IntType,
    FloatType,
    IntLit,
    Slice,
)
from ir.translate import (
    translate_kernel_source,
    translate_kernel_file,
    TranslationError,
)
from ir.validate_subset import validate_triton_subset, SubsetViolation
from ir.pp import pretty_kernel


# ---------------------------------------------------------------------------
# Paths to Triton kernel source files
# ---------------------------------------------------------------------------
_TRITON_DIR = Path(__file__).resolve().parents[2] / "triton_kernels"
def _read_source(name: str) -> str:
    return (_TRITON_DIR / name).read_text(encoding="utf-8")


def _minimal_source(
    body: str,
    *,
    signature: str = "BLOCK: tl.constexpr",
    directives: tuple[str, ...] = (),
    decorators: tuple[str, ...] = ("@triton.jit",),
) -> str:
    annotation = "".join(f"#   {directive},\n" for directive in directives)
    decorated = "\n".join(decorators)
    indented_body = textwrap.indent(textwrap.dedent(body).strip(), "    ")
    return (
        "import triton\nimport triton.language as tl\n\n"
        f"# @params(\n{annotation}# )\n# @grid(1)\n"
        f"{decorated}\ndef bad_kernel({signature}):\n{indented_body}\n"
    )


# ===================================================================
# Subset validator tests
# ===================================================================


class TestSubsetValidator:
    """Test that the subset validator correctly accepts/rejects patterns."""

    def test_matmul_is_valid(self) -> None:
        source = _read_source("matmul.py")
        violations = validate_triton_subset(source, "matmul_kernel")
        assert violations == [], f"Unexpected violations: {violations}"

    def test_rejects_raw_pointer_load(self) -> None:
        source = textwrap.dedent("""\
            import triton
            import triton.language as tl

            @triton.jit
            def bad_kernel(x, N, stride_x, BLOCK: tl.constexpr):
                i = tl.program_id(0)
                offs = i * BLOCK + tl.arange(0, BLOCK)
                mask = offs < N
                vals = tl.load(x + offs * stride_x, mask=mask, other=0.0)
                tl.store(x + offs * stride_x, vals, mask=mask)
        """)
        violations = validate_triton_subset(source, "bad_kernel")
        assert len(violations) >= 1
        msgs = [str(v) for v in violations]
        assert any("raw pointer" in m.lower() or "mask" in m.lower() for m in msgs), \
            f"Expected raw pointer violation, got: {msgs}"

    def test_rejects_unrecognized_vector_pointer_load(self) -> None:
        source = textwrap.dedent("""\
            import triton
            import triton.language as tl

            @triton.jit
            def bad_kernel(x, BLOCK: tl.constexpr):
                values = tl.load(x + tl.arange(0, BLOCK))
        """)
        violations = validate_triton_subset(source, "bad_kernel")
        assert any("pointer expression" in str(violation) for violation in violations)

    def test_rejects_atomic(self) -> None:
        source = textwrap.dedent("""\
            import triton
            import triton.language as tl

            @triton.jit
            def atomic_kernel(x, BLOCK: tl.constexpr):
                i = tl.program_id(0)
                tl.atomic_add(x + i, 1.0)
        """)
        violations = validate_triton_subset(source, "atomic_kernel")
        assert len(violations) >= 1
        assert any("atomic" in str(v).lower() for v in violations)

    def test_rejects_ptr_store(self) -> None:
        """tl.store with raw pointer (not block_ptr) should be rejected."""
        source = textwrap.dedent("""\
            import triton
            import triton.language as tl

            @triton.jit
            def raw_store_kernel(x, N, stride_x, BLOCK: tl.constexpr):
                i = tl.program_id(0)
                offs = i * BLOCK + tl.arange(0, BLOCK)
                tl.store(x + offs * stride_x, tl.zeros((BLOCK,), dtype=tl.float32))
        """)
        violations = validate_triton_subset(source, "raw_store_kernel")
        assert len(violations) >= 1
        assert any("raw pointer" in str(v).lower() for v in violations)

    def test_paged_block_ptr_kernel_is_valid(self) -> None:
        """The paged attention kernel should pass validation."""
        source = _read_source("fattn_paged.py")
        violations = validate_triton_subset(source, "fattn_varlen_paged_fwd_block_ptr_kernel")
        assert violations == [], f"Unexpected violations: {violations}"

    @pytest.mark.parametrize(
        ("operation", "arguments", "keyword"),
        [
            ("load", "p", "mask=True"),
            ("load", "p", 'padding_option="nan"'),
            ("store", "p, block", "mask=True"),
            ("store", "p, block", 'cache_modifier=".wb"'),
        ],
    )
    def test_rejects_unmodeled_block_pointer_options(
        self, operation: str, arguments: str, keyword: str
    ) -> None:
        source = textwrap.dedent(f"""\
            import triton
            import triton.language as tl

            @triton.jit
            def bad_kernel(x, BLOCK: tl.constexpr):
                p = tl.make_block_ptr(
                    x, shape=(BLOCK,), strides=(1,), offsets=(0,),
                    block_shape=(BLOCK,), order=(0,),
                )
                block = tl.zeros((BLOCK,), dtype=tl.float32)
                tl.{operation}({arguments}, boundary_check=(0,), {keyword})
        """)
        violations = validate_triton_subset(source, "bad_kernel")
        assert any("not modeled" in str(v) for v in violations), violations

    def test_reassigned_block_pointer_is_not_still_accepted(self) -> None:
        source = textwrap.dedent("""\
            import triton
            import triton.language as tl

            @triton.jit
            def bad_kernel(x, BLOCK: tl.constexpr):
                p = tl.make_block_ptr(
                    x, shape=(BLOCK,), strides=(1,), offsets=(0,),
                    block_shape=(BLOCK,), order=(0,),
                )
                p = x + tl.arange(0, BLOCK)
                block = tl.load(p, boundary_check=(0,), padding_option="zero")
        """)
        violations = validate_triton_subset(source, "bad_kernel")
        assert any("pointer expression" in str(v) for v in violations), violations

    @pytest.mark.parametrize("call", ["tl.load()", "tl.store()", "tl.store(p)"])
    def test_rejects_missing_memory_arguments(self, call: str) -> None:
        source = textwrap.dedent(f"""\
            import triton
            import triton.language as tl

            @triton.jit
            def bad_kernel(x, BLOCK: tl.constexpr):
                p = tl.make_block_ptr(
                    x, shape=(BLOCK,), strides=(1,), offsets=(0,),
                    block_shape=(BLOCK,), order=(0,),
                )
                {call}
        """)
        violations = validate_triton_subset(source, "bad_kernel")
        assert any("argument" in str(v) for v in violations), violations


# ===================================================================
# Translator tests
# ===================================================================


class TestTranslateMatmul:
    """Test translation of the matmul kernel."""

    def test_translates_without_error(self) -> None:
        source = _read_source("matmul.py")
        kernel = translate_kernel_source(source, "matmul_kernel")
        assert isinstance(kernel, Kernel)
        assert kernel.name == "matmul_kernel"

    def test_has_grid_iters(self) -> None:
        source = _read_source("matmul.py")
        kernel = translate_kernel_source(source, "matmul_kernel")
        assert len(kernel.grid.iters) == 2  # i, j

    def test_has_tensor_params(self) -> None:
        source = _read_source("matmul.py")
        kernel = translate_kernel_source(source, "matmul_kernel")
        tensor_params = [p for p in kernel.params if isinstance(p.type, TensorType)]
        assert len(tensor_params) == 3  # a, b, c
        names = {p.name for p in tensor_params}
        assert names == {"a", "b", "c"}

    def test_body_has_for_loop(self) -> None:
        source = _read_source("matmul.py")
        kernel = translate_kernel_source(source, "matmul_kernel")
        for_stmts = [s for s in kernel.grid.body if isinstance(s, For)]
        assert len(for_stmts) >= 1, "Should have at least one For loop (k-loop)"

    def test_body_has_masked_store(self) -> None:
        source = _read_source("matmul.py")
        kernel = translate_kernel_source(source, "matmul_kernel")
        stores = [s for s in kernel.grid.body if isinstance(s, MaskedStore)]
        assert len(stores) >= 1, "Should have a MaskedStore for writing output"

    def test_for_loop_has_dot_product(self) -> None:
        source = _read_source("matmul.py")
        kernel = translate_kernel_source(source, "matmul_kernel")
        for_stmts = [s for s in kernel.grid.body if isinstance(s, For)]
        assert len(for_stmts) >= 1

        # Check that the for loop body contains a dot product (mm_)
        for_body = for_stmts[0].body
        has_dot = any(
            isinstance(s, Assign) and isinstance(s.value, BinOp) and s.value.op == "@"
            for s in for_body
        )
        assert has_dot, "For loop should contain a matrix multiply"

    def test_pretty_prints(self) -> None:
        source = _read_source("matmul.py")
        kernel = translate_kernel_source(source, "matmul_kernel")
        output = pretty_kernel(kernel)
        assert "matmul_kernel" in output
        assert "grid(" in output


# ===================================================================
# Error handling tests
# ===================================================================


class TestTranslationErrors:
    """Test that the translator gives clear errors for unsupported patterns."""

    @pytest.mark.parametrize(
        "statement",
        [
            'block = tl.load(p, boundary_check=(0,), padding_option="zero", cache_modifier=".ca")',
            "tl.store(p, block, boundary_check=(0,), mask=True)",
        ],
    )
    def test_translator_rejects_unmodeled_memory_options(
        self, statement: str
    ) -> None:
        source = textwrap.dedent(f"""\
            import triton
            import triton.language as tl

            # @params(
            #   tensor(x, float, shape(N), strides(stride_x)),
            # )
            # @grid(1)
            @triton.jit
            def bad_kernel(x, N, stride_x, BLOCK: tl.constexpr):
                i = tl.program_id(0)
                p = tl.make_block_ptr(
                    x, shape=(N,), strides=(stride_x,), offsets=(0,),
                    block_shape=(BLOCK,), order=(0,),
                )
                block = tl.zeros((BLOCK,), dtype=tl.float32)
                {statement}
        """)
        with pytest.raises(TranslationError, match="keyword .* is not modeled"):
            translate_kernel_source(source, "bad_kernel")

    def test_missing_kernel(self) -> None:
        with pytest.raises(TranslationError, match="not found"):
            translate_kernel_source("def foo(): pass", "nonexistent_kernel")

    def test_rejects_non_triton_function_with_kernel_name(self) -> None:
        source = textwrap.dedent("""\
            # @params(
            # )
            # @grid(1)
            def not_a_kernel():
                pass
        """)
        with pytest.raises(TranslationError, match="not decorated with @triton.jit"):
            translate_kernel_source(source, "not_a_kernel")

    def test_rejects_duplicate_top_level_kernel_names(self) -> None:
        source = textwrap.dedent("""\
            import triton

            # @params(
            # )
            # @grid(1)
            @triton.jit
            def duplicate():
                pass

            # @params(
            # )
            # @grid(1)
            @triton.jit
            def duplicate():
                pass
        """)
        with pytest.raises(TranslationError, match="exactly one top-level"):
            translate_kernel_source(source, "duplicate")

    def test_missing_kernel_interface_annotations(self) -> None:
        source = textwrap.dedent("""\
            import triton
            import triton.language as tl

            @triton.jit
            def bare_kernel(N: tl.constexpr):
                i = tl.program_id(0)
        """)
        with pytest.raises(ValueError, match="missing @params/@grid"):
            translate_kernel_source(source, "bare_kernel")

    def test_missing_param_type_annotation(self) -> None:
        """Scalar param used as float without param() annotation raises TypeError."""
        source = textwrap.dedent("""\
            import triton
            import triton.language as tl

            # @params(
            #   tensor(x, float, shape(M, N), strides(stride_xm, stride_xn)),
            #   tensor(o, float, shape(M, N), strides(stride_om, stride_on)),
            # )
            # @grid(cdiv(M, BLOCK_M))
            @triton.jit
            def bad_kernel(
                x, o, M, N,
                stride_xm, stride_xn, stride_om, stride_on,
                scale,
                BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
            ):
                row = tl.program_id(axis=0) * BLOCK_M
                x_block_ptr = tl.make_block_ptr(x, shape=(M, N), strides=(stride_xm, stride_xn), offsets=(row, 0), block_shape=(BLOCK_M, BLOCK_N), order=(0, 1))
                x_block = tl.load(x_block_ptr, boundary_check=(0, 1), padding_option="zero").to(tl.float32)
                result = tl.math.rsqrt(scale)
                o_block_ptr = tl.make_block_ptr(o, shape=(M, N), strides=(stride_om, stride_on), offsets=(row, 0), block_shape=(BLOCK_M, BLOCK_N), order=(0, 1))
                tl.store(o_block_ptr, x_block, boundary_check=(0, 1))
        """)
        from ir.subst import specialize_kernel_constants
        from ir.preprocess import check_variable_names
        from ir.typ import infer_types

        kernel = translate_kernel_source(source, "bad_kernel")
        kernel = specialize_kernel_constants(kernel, {"BLOCK_M": 1, "BLOCK_N": 256})
        roles = check_variable_names(kernel)
        with pytest.raises(TypeError, match="rsqrt requires float operand"):
            infer_types(kernel, roles)

    def test_unsupported_range_with_step(self) -> None:
        source = textwrap.dedent("""\
            import triton
            import triton.language as tl

            # @params(
            # )
            # @grid(N)
            @triton.jit
            def bad_kernel(N: tl.constexpr):
                i = tl.program_id(0)
                for k in range(0, N, 2):
                    pass
        """)
        with pytest.raises(TranslationError, match="step"):
            translate_kernel_source(source, "bad_kernel")

    def test_missing_interface_does_not_borrow_previous_kernel(self) -> None:
        source = textwrap.dedent("""\
            import triton
            import triton.language as tl

            # @params(
            # )
            # @grid(1)
            @triton.jit
            def first_kernel():
                i = tl.program_id(0)

            @triton.jit
            def target_kernel():
                i = tl.program_id(0)
        """)
        with pytest.raises(ValueError, match="missing @params/@grid"):
            translate_kernel_source(source, "target_kernel")

    def test_rejects_unknown_params_directive(self) -> None:
        source = textwrap.dedent("""\
            import triton
            import triton.language as tl

            # @params(
            #   trust_me(ignored),
            # )
            # @grid(1)
            @triton.jit
            def bad_kernel():
                i = tl.program_id(0)
        """)
        with pytest.raises(ValueError, match="Unknown @params directive"):
            translate_kernel_source(source, "bad_kernel")

    def test_tensor_element_type_does_not_depend_on_stride_presence(self) -> None:
        source = textwrap.dedent("""\
            import triton
            import triton.language as tl

            # @params(
            #   tensor(values, float, shape(N)),
            #   tensor(indices, int32, shape(N)),
            # )
            # @grid(1)
            @triton.jit
            def kernel(values, indices):
                i = tl.program_id(0)
        """)
        kernel = translate_kernel_source(source, "kernel")
        types = {param.name: param.type for param in kernel.params}
        assert types["values"] == TensorType(FloatType(), [Var("N")])
        assert types["indices"] == TensorType(IntType(), [Var("N")])

    def test_explicit_stride_role_does_not_depend_on_parameter_name(self) -> None:
        source = textwrap.dedent("""\
            import triton
            import triton.language as tl

            # @params(
            #   tensor(x, float, shape(N), strides(pitch)),
            # )
            # @grid(1)
            @triton.jit
            def kernel(x, N, pitch, BLOCK: tl.constexpr):
                i = tl.program_id(0)
                p = tl.make_block_ptr(
                    x, shape=(N,), strides=(pitch,), offsets=(0,),
                    block_shape=(BLOCK,), order=(0,),
                )
                block = tl.load(p, boundary_check=(0,), padding_option="zero")
        """)
        kernel = translate_kernel_source(source, "kernel")
        assert {parameter.name for parameter in kernel.params} == {"x", "BLOCK"}

    @pytest.mark.parametrize(
        "directives, expected",
        [
            (
                (
                    "scalar(x, int)",
                    "tensor(x, float, shape(N))",
                ),
                "Duplicate @params name 'x'",
            ),
            (
                ("tensor(x, float, shape(M, N), strides(pitch, pitch))",),
                "duplicate stride parameters",
            ),
            (
                (
                    "scalar(pitch, int)",
                    "tensor(x, float, shape(N), strides(pitch))",
                ),
                "both scalar values and tensor strides",
            ),
        ],
    )
    def test_rejects_conflicting_params_roles(
        self, directives: tuple[str, ...], expected: str
    ) -> None:
        source = _minimal_source(
            "i = tl.program_id(0)",
            signature="x, M, N, pitch",
            directives=directives,
        )
        with pytest.raises(ValueError, match=expected):
            translate_kernel_source(source, "bad_kernel")

    def test_rejects_verif_grid_program_id_mismatch(self) -> None:
        source = textwrap.dedent("""\
            import triton
            import triton.language as tl

            # @params(
            # )
            # @grid(1, 1)
            @triton.jit
            def bad_kernel():
                i = tl.program_id(0)
        """)
        with pytest.raises(TranslationError, match="grid axes"):
            translate_kernel_source(source, "bad_kernel")

    def test_rejects_duplicate_program_id_axis(self) -> None:
        source = textwrap.dedent("""\
            import triton
            import triton.language as tl

            # @params(
            # )
            # @grid(1)
            @triton.jit
            def bad_kernel():
                i = tl.program_id(0)
                j = tl.program_id(0)
        """)
        with pytest.raises(TranslationError, match="Multiple tl.program_id"):
            translate_kernel_source(source, "bad_kernel")

    def test_rejects_late_program_id_assignment(self) -> None:
        source = textwrap.dedent("""\
            import triton
            import triton.language as tl

            # @params(
            #   scalar(n, int),
            # )
            # @grid(1)
            @triton.jit
            def bad_kernel(n):
                i = tl.program_id(0)
                tmp = n + 1
                j = tl.program_id(0)
        """)
        with pytest.raises(TranslationError, match="initial grid-iterator prefix"):
            translate_kernel_source(source, "bad_kernel")

    def test_rejects_for_else_instead_of_dropping_else_body(self) -> None:
        source = textwrap.dedent("""\
            import triton
            import triton.language as tl

            # @params(
            #   scalar(n, int),
            # )
            # @grid(1)
            @triton.jit
            def bad_kernel(n):
                i = tl.program_id(0)
                for k in range(n):
                    tmp = k + 1
                else:
                    tmp = n + 1
        """)
        with pytest.raises(TranslationError, match=r"for \.\.\. else"):
            translate_kernel_source(source, "bad_kernel")

    def test_rejects_block_pointer_shape_disagreeing_with_params(self) -> None:
        source = textwrap.dedent("""\
            import triton
            import triton.language as tl

            # @params(
            #   tensor(x, float, shape(1, N), strides(stride_xm, stride_xn)),
            # )
            # @grid(1)
            @triton.jit
            def bad_kernel(x, M, N, stride_xm, stride_xn):
                i = tl.program_id(0)
                p = tl.make_block_ptr(
                    x, shape=(M, N), strides=(stride_xm, stride_xn),
                    offsets=(0, 0), block_shape=(1, N), order=(0, 1),
                )
                block = tl.load(p, boundary_check=(0, 1), padding_option="zero")
        """)
        with pytest.raises(TranslationError, match="disagrees with @params"):
            translate_kernel_source(source, "bad_kernel")

    def test_rejects_reshape_that_drops_non_singleton_block_axis(self) -> None:
        source = textwrap.dedent("""\
            import triton
            import triton.language as tl

            # @params(
            #   tensor(x, float, shape(M, N), strides(stride_xm, stride_xn)),
            # )
            # @grid(1)
            @triton.jit
            def bad_kernel(x, M, N, stride_xm, stride_xn):
                i = tl.program_id(0)
                p = tl.make_block_ptr(
                    x, shape=(M, N), strides=(stride_xm, stride_xn),
                    offsets=(0, 0), block_shape=(2, N), order=(0, 1),
                )
                block = tl.reshape(
                    tl.load(p, boundary_check=(0, 1), padding_option="zero"),
                    (N,),
                )
        """)
        with pytest.raises(TranslationError, match="removing size-1"):
            translate_kernel_source(source, "bad_kernel")

    def test_rejects_unmapped_block_pointer_base_term(self) -> None:
        source = textwrap.dedent("""\
            import triton
            import triton.language as tl

            # @params(
            #   tensor(x, float, shape(M, N), strides(stride_xm, stride_xn)),
            # )
            # @grid(1)
            @triton.jit
            def bad_kernel(x, M, N, offset, stride_xm, stride_xn):
                i = tl.program_id(0)
                p = tl.make_block_ptr(
                    x + offset, shape=(M, N),
                    strides=(stride_xm, stride_xn), offsets=(0, 0),
                    block_shape=(1, N), order=(0, 1),
                )
                block = tl.load(p, boundary_check=(0, 1), padding_option="zero")
        """)
        with pytest.raises(TranslationError, match="Unsupported bare term"):
            translate_kernel_source(source, "bad_kernel")

    def test_rejects_masked_scalar_pointer_load_instead_of_dropping_mask(self) -> None:
        source = textwrap.dedent("""\
            import triton
            import triton.language as tl

            # @params(
            #   tensor(ids, int32, shape(N)),
            # )
            # @grid(1)
            @triton.jit
            def bad_kernel(ids, N):
                i = tl.program_id(0)
                value = tl.load(ids + i, mask=i < N, other=0)
        """)
        violations = validate_triton_subset(source, "bad_kernel")
        assert any("unmasked" in str(violation) for violation in violations)
        with pytest.raises(TranslationError, match="Scalar pointer loads must be unmasked"):
            translate_kernel_source(source, "bad_kernel")

    def test_unchecked_block_pointer_axis_is_not_modeled_as_padding(self) -> None:
        from ir import Assign, MaskedLoad

        source = textwrap.dedent("""\
            import triton
            import triton.language as tl

            # @params(
            #   tensor(x, float, shape(M, N), strides(stride_xm, stride_xn)),
            # )
            # @grid(1)
            @triton.jit
            def kernel(x, M, N, stride_xm, stride_xn):
                i = tl.program_id(0)
                p = tl.make_block_ptr(
                    x, shape=(M, N), strides=(stride_xm, stride_xn),
                    offsets=(0, 1), block_shape=(1, 1), order=(0, 1),
                )
                block = tl.load(p, boundary_check=(0,), padding_option="zero")
        """)
        kernel = translate_kernel_source(source, "kernel")
        load = next(
            stmt.value
            for stmt in kernel.grid.body
            if isinstance(stmt, Assign) and isinstance(stmt.value, MaskedLoad)
        )
        assert load.mask[1] == load.region[1]

    def test_rejects_parameter_reassignment_instead_of_dropping_it(self) -> None:
        source = textwrap.dedent("""\
            import triton
            import triton.language as tl

            # @params(
            #   scalar(scale, float),
            # )
            # @grid(1)
            @triton.jit
            def bad_kernel(scale):
                i = tl.program_id(0)
                scale = scale * 2.0
        """)
        with pytest.raises(TranslationError, match="Reassignment of kernel parameter"):
            translate_kernel_source(source, "bad_kernel")

    def test_rejects_augmented_parameter_reassignment(self) -> None:
        source = textwrap.dedent("""\
            import triton
            import triton.language as tl

            # @params(
            #   scalar(scale, float),
            # )
            # @grid(1)
            @triton.jit
            def bad_kernel(scale):
                i = tl.program_id(0)
                scale *= 2.0
        """)
        with pytest.raises(TranslationError, match="Augmented assignment"):
            translate_kernel_source(source, "bad_kernel")

    def test_rejects_unsupported_expression_call_instead_of_dropping_it(self) -> None:
        source = textwrap.dedent("""\
            import triton
            import triton.language as tl

            # @params(
            #   scalar(x, int),
            # )
            # @grid(1)
            @triton.jit
            def bad_kernel(x):
                i = tl.program_id(0)
                helper_with_side_effect(x)
        """)
        with pytest.raises(TranslationError, match="Unsupported expression call"):
            translate_kernel_source(source, "bad_kernel")

    def test_rejects_nested_three_argument_dot_instead_of_dropping_accumulator(self) -> None:
        source = textwrap.dedent("""\
            import triton
            import triton.language as tl

            # @params(
            # )
            # @grid(1)
            @triton.jit
            def bad_kernel():
                i = tl.program_id(0)
                a = tl.zeros((1, 1), dtype=tl.float32)
                b = tl.zeros((1, 1), dtype=tl.float32)
                acc = tl.zeros((1, 1), dtype=tl.float32)
                out = tl.dot(a, b, acc) + acc
        """)
        with pytest.raises(TranslationError, match="complete right-hand side"):
            translate_kernel_source(source, "bad_kernel")

    def test_rejects_nested_return_instead_of_dropping_it(self) -> None:
        source = textwrap.dedent("""\
            import triton
            import triton.language as tl

            # @params(
            #   scalar(x, int),
            # )
            # @grid(1)
            @triton.jit
            def bad_kernel(x):
                i = tl.program_id(0)
                if x > 0:
                    x = x + 1
                    return
        """)
        with pytest.raises(TranslationError, match="single top-level"):
            translate_kernel_source(source, "bad_kernel")

    def test_rejects_reduction_options_that_change_translated_shape(self) -> None:
        source = _minimal_source("""
            i = tl.program_id(0)
            values = tl.zeros((BLOCK,), dtype=tl.float32)
            total = tl.sum(values, axis=0, keep_dims=True)
        """)
        with pytest.raises(TranslationError, match="keep_dims.*not modeled"):
            translate_kernel_source(source, "bad_kernel")

    def test_rejects_cast_options_instead_of_erasing_them(self) -> None:
        source = _minimal_source("""
            i = tl.program_id(0)
            values = tl.zeros((BLOCK,), dtype=tl.float32)
            converted = values.to(tl.int32, bitcast=True)
        """)
        with pytest.raises(TranslationError, match="bitcast.*not modeled"):
            translate_kernel_source(source, "bad_kernel")

    def test_rejects_narrowing_integer_cast_without_int32_metadata(self) -> None:
        source = _minimal_source("""
            i = tl.program_id(0)
            values = tl.arange(0, BLOCK)
            narrowed = values.to(tl.int32)
        """)
        with pytest.raises(TranslationError, match="only for scalar loads"):
            translate_kernel_source(source, "bad_kernel")

    def test_rejects_unknown_cast_target_instead_of_erasing_it(self) -> None:
        source = _minimal_source(
            """
            i = tl.program_id(0)
            values = tl.zeros((BLOCK,), dtype=tl.float32)
            converted = values.to(OUTPUT_DTYPE)
            """,
            signature="OUTPUT_DTYPE: tl.constexpr, BLOCK: tl.constexpr",
        )
        with pytest.raises(TranslationError, match="Cast target.*not modeled"):
            translate_kernel_source(source, "bad_kernel")

    def test_rejects_integer_tensor_constructor_in_float_only_ir(self) -> None:
        source = _minimal_source("""
            i = tl.program_id(0)
            values = tl.full((BLOCK,), 256, dtype=tl.uint8)
        """)
        with pytest.raises(TranslationError, match="integer element types are not modeled"):
            translate_kernel_source(source, "bad_kernel")

    def test_rejects_non_unsqueeze_reshape_with_leading_one(self) -> None:
        source = _minimal_source("""
            i = tl.program_id(0)
            values = tl.zeros((2, 3), dtype=tl.float32)
            reordered = tl.reshape(values, (1, 3, 2))
        """)
        with pytest.raises(TranslationError, match="not obtained by adding"):
            translate_kernel_source(source, "bad_kernel")

    @pytest.mark.parametrize("index", ["values[1:, None]", "values[0, None]"])
    def test_rejects_nontrivial_index_hidden_inside_unsqueeze(
        self, index: str
    ) -> None:
        source = _minimal_source(f"""
            i = tl.program_id(0)
            values = tl.zeros((BLOCK,), dtype=tl.float32)
            expanded = {index}
        """)
        with pytest.raises(TranslationError, match="only `None` and full `:`"):
            translate_kernel_source(source, "bad_kernel")

    def test_rejects_reassignment_of_captured_block_pointer_offset(self) -> None:
        source = textwrap.dedent("""\
            import triton
            import triton.language as tl

            # @params(
            #   tensor(x, float, shape(N), strides(stride_x)),
            # )
            # @grid(1)
            @triton.jit
            def bad_kernel(x, N, stride_x, BLOCK: tl.constexpr):
                i = tl.program_id(0)
                offset = i * BLOCK
                p = tl.make_block_ptr(
                    x, shape=(N,), strides=(stride_x,), offsets=(offset,),
                    block_shape=(BLOCK,), order=(0,),
                )
                offset += BLOCK
                values = tl.load(
                    p, boundary_check=(0,), padding_option="zero"
                )
        """)
        with pytest.raises(TranslationError, match="captured by live pointer"):
            translate_kernel_source(source, "bad_kernel")

    def test_rejects_tensor_valued_raw_pointer_offset(self) -> None:
        source = textwrap.dedent("""\
            import triton
            import triton.language as tl

            # @params(
            #   tensor(table, int32, shape(M, N), strides(stride_tm, stride_tn)),
            # )
            # @grid(1)
            @triton.jit
            def bad_kernel(
                table, M, N, stride_tm, stride_tn, BLOCK: tl.constexpr,
            ):
                i = tl.program_id(0)
                columns = tl.arange(0, BLOCK)
                values = tl.load(
                    table + i * stride_tm + columns * stride_tn
                )
        """)
        with pytest.raises(TranslationError, match="tensor-valued offsets"):
            translate_kernel_source(source, "bad_kernel")

    @pytest.mark.parametrize(
        "body",
        [
            """
    for k in range(K):
        tmp = tl.zeros((BLOCK,), dtype=tl.float32)
    out_ptr = tl.make_block_ptr(
        o, shape=(N,), strides=(stride_o,), offsets=(pid * BLOCK,),
        block_shape=(BLOCK,), order=(0,)
    )
    tl.store(out_ptr, tmp, boundary_check=(0,))
""",
            """
    if pid == 0:
        tmp = tl.zeros((BLOCK,), dtype=tl.float32)
    out_ptr = tl.make_block_ptr(
        o, shape=(N,), strides=(stride_o,), offsets=(pid * BLOCK,),
        block_shape=(BLOCK,), order=(0,)
    )
    tl.store(out_ptr, tmp, boundary_check=(0,))
""",
        ],
        ids=["zero-trip-loop", "one-sided-branch"],
    )
    def test_rejects_possibly_undefined_local_after_control_flow(
        self, body: str
    ) -> None:
        source = f"""
import triton
import triton.language as tl

# @params(
#   tensor(o, float, shape(N), strides(stride_o)),
# )
# @grid(cdiv(N, BLOCK))
@triton.jit
def bad_kernel(o, N, K, stride_o, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
{body}
"""
        with pytest.raises(TranslationError, match="tmp.*may be undefined"):
            translate_kernel_source(source, "bad_kernel")

    def test_legacy_unbounded_integer_tensor_directive_is_rejected(self) -> None:
        source = _minimal_source(
            "i = tl.program_id(0)",
            signature="indices, N",
            directives=("int_tensor(indices, N)",),
        )
        with pytest.raises(ValueError, match="Unknown @params directive 'int_tensor'"):
            translate_kernel_source(source, "bad_kernel")

    def test_rejects_specializing_a_runtime_parameter(self) -> None:
        source = _minimal_source(
            """
            i = tl.program_id(0)
            if flag:
                value = 1
            """,
            signature="flag",
            directives=("scalar(flag, bool)",),
        )
        with pytest.raises(TranslationError, match="Only tl.constexpr"):
            translate_kernel_source(source, "bad_kernel", specialize={"flag": True})

    def test_branch_local_block_pointer_does_not_escape(self) -> None:
        source = _minimal_source(
            """
            i = tl.program_id(0)
            if flag:
                p = tl.make_block_ptr(
                    x, shape=(N,), strides=(stride_x,), offsets=(0,),
                    block_shape=(BLOCK,), order=(0,),
                )
            values = tl.zeros((BLOCK,), dtype=tl.float32)
            tl.store(p, values, boundary_check=(0,))
            """,
            signature="x, N, stride_x, flag, BLOCK: tl.constexpr",
            directives=(
                "scalar(flag, bool)",
                "tensor(x, float, shape(N), strides(stride_x))",
            ),
        )
        with pytest.raises(TranslationError, match="p.*may be undefined"):
            translate_kernel_source(source, "bad_kernel")

    def test_make_block_ptr_cannot_discard_unknown_options(self) -> None:
        source = _minimal_source(
            """
            i = tl.program_id(0)
            p = tl.make_block_ptr(
                x, shape=(N,), strides=(stride_x,), offsets=(0,),
                block_shape=(BLOCK,), order=(0,), mystery=True,
            )
            """,
            signature="x, N, stride_x, BLOCK: tl.constexpr",
            directives=("tensor(x, float, shape(N), strides(stride_x))",),
        )
        with pytest.raises(TranslationError, match="unmodeled keyword"):
            translate_kernel_source(source, "bad_kernel")

    def test_rejects_unmodeled_kernel_decorator(self) -> None:
        source = _minimal_source(
            "i = tl.program_id(0)",
            signature="",
            decorators=("@change_launch_semantics", "@triton.jit"),
        )
        with pytest.raises(TranslationError, match="decorator.*not modeled"):
            translate_kernel_source(source, "bad_kernel")

    def test_rejects_runtime_autotune_decorator(self) -> None:
        source = _minimal_source(
            "i = tl.program_id(0)",
            decorators=(
                "@triton.autotune(configs=CONFIGS, key=['BLOCK'])",
                "@triton.jit",
            ),
        )
        with pytest.raises(TranslationError, match="decorator.*not modeled"):
            translate_kernel_source(source, "bad_kernel")

    def test_load_type_comes_from_its_own_region(self) -> None:
        source = textwrap.dedent("""\
            import triton
            import triton.language as tl

            # @params(
            #   tensor(x, float, shape(M, N), strides(stride_xm, stride_xn)),
            # )
            # @grid(1)
            @triton.jit
            def kernel(x, M, N, stride_xm, stride_xn):
                i = tl.program_id(0)
                wide_ptr = tl.make_block_ptr(
                    x, shape=(M, N), strides=(stride_xm, stride_xn),
                    offsets=(0, 0), block_shape=(1, N), order=(0, 1),
                )
                wide = tl.load(
                    wide_ptr, boundary_check=(0, 1), padding_option="zero"
                )
                scalar_ptr = tl.make_block_ptr(
                    x, shape=(M, N), strides=(stride_xm, stride_xn),
                    offsets=(0, 0), block_shape=(1, 1), order=(0, 1),
                )
                scalar = tl.load(
                    scalar_ptr, boundary_check=(0, 1), padding_option="zero"
                )
        """)
        kernel = translate_kernel_source(source, "kernel")
        decl_types = {decl.var.name: decl.type for decl in kernel.grid.decls}
        assert decl_types["wide"] == TensorType(FloatType(), [IntLit(1), Var("N")])
        assert decl_types["scalar"] == TensorType(
            FloatType(), [IntLit(1), IntLit(1)]
        )


# ===================================================================
# Integration: translated IR structure
# ===================================================================


class TestTranslatedVerification:
    """Test that translated kernels pass the full verification pipeline."""

    def test_matmul_passes_preprocessing(self) -> None:
        """Translated matmul should pass check_variable_names + infer_types."""
        from ir.preprocess import check_variable_names, check_tensorindex_readonly
        from ir.subst import specialize_kernel_constants, expand_let_bindings
        from ir.typ import infer_types
        from ir.proof_preparation import ensure_tensor_var_sizes_known

        source = _read_source("matmul.py")
        kernel = translate_kernel_source(source, "matmul_kernel")

        roles = check_variable_names(kernel)
        check_tensorindex_readonly(kernel)
        kernel = specialize_kernel_constants(
            kernel, {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32}
        )
        kernel = expand_let_bindings(kernel)
        ensure_tensor_var_sizes_known(kernel)
        kernel, _ = infer_types(kernel, roles)

        # Verify the specialized kernel looks right
        output = pretty_kernel(kernel)
        assert "cdiv" not in output or "BLOCK" not in output  # blocks are specialized
        assert "64" in output  # BLOCK_M/N specialized to 64

    def test_matmul_z3_verification(self) -> None:
        """Translated matmul should pass Z3 row equivalence proof."""
        from ir.preprocess import check_variable_names, check_tensorindex_readonly
        from ir.subst import specialize_kernel_constants, expand_let_bindings
        from ir.typ import infer_types
        from ir.regional_obligations import prove_region_equivalence, EquivProofConfig, TensorAssumption
        from ir.proof_preparation import ensure_tensor_var_sizes_known
        import z3

        source = _read_source("matmul.py")
        kernel = translate_kernel_source(source, "matmul_kernel")
        roles = check_variable_names(kernel)
        check_tensorindex_readonly(kernel)
        kernel = specialize_kernel_constants(
            kernel, {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32}
        )
        kernel = expand_let_bindings(kernel)
        ensure_tensor_var_sizes_known(kernel)
        kernel, _ = infer_types(kernel, roles)

        M, N, K, x = z3.Ints("M N K x")

        config = EquivProofConfig(
            kernel=kernel,
            output_tensor="c",
            left_output_region=[
                Slice(Var("x"), BinOp("+", Var("x"), IntLit(1))),
                Slice(IntLit(0), Var("N")),
            ],
            right_output_region=[
                Slice(IntLit(0), IntLit(1)),
                Slice(IntLit(0), Var("N")),
            ],
            tensor_assumptions={
                "a": TensorAssumption(
                    [Slice(Var("x"), BinOp("+", Var("x"), IntLit(1))),
                     Slice(IntLit(0), Var("K"))],
                    [Slice(IntLit(0), IntLit(1)),
                     Slice(IntLit(0), Var("K"))],
                ),
                "b": TensorAssumption(
                    [Slice(IntLit(0), Var("K")),
                     Slice(IntLit(0), Var("N"))],
                    [Slice(IntLit(0), Var("K")),
                     Slice(IntLit(0), Var("N"))],
                ),
            },
            left_env={
                "M": M, "N": N, "K": K, "x": x,
                "i": z3.Int("iL"), "j": z3.Int("jL"), "k": z3.Int("kL"),
            },
            right_env={
                "M": z3.IntVal(1), "N": N, "K": K,
                "i": z3.Int("iR"), "j": z3.Int("jR"), "k": z3.Int("kR"),
            },
            base_assumptions=[M > 0, N > 0, K > 0, x >= 0, x < M],
        )

        result = prove_region_equivalence(config)
        for check in result.checks:
            assert check.proved, f"Check '{check.name}' failed: {check.details}"

    def test_matmul_grid_ranges_from_annotation(self) -> None:
        """Translated matmul should use the cdiv ranges declared by @grid."""
        source = _read_source("matmul.py")
        kernel = translate_kernel_source(source, "matmul_kernel")
        output = pretty_kernel(kernel)
        assert "_grid_dim" not in output, \
            f"Grid ranges should come from annotation, not placeholders: {output}"
        assert "cdiv" in output, "Grid ranges should use cdiv"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
