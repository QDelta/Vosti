"""
Coarse source gate for the "Verifiable Triton" subset.

Rejects:
  - Raw vector-pointer loads/stores (scalar index-array loads are allowed)
  - tl.atomic_* operations
  - tl.inline_asm / tl.extra
  - Unsupported operations (tl.libdevice.*, etc.)

The translator performs the authoritative statement, shape, and control-flow
checks after this gate. Memory-call forms are checked in both layers because
silently erased pointer options would invalidate the translated memory model.

Usage:
    from ir.validate_subset import validate_triton_subset
    errors = validate_triton_subset(source, "matmul_kernel")
    if errors:
        for e in errors:
            print(e)
"""

from __future__ import annotations

import ast
from dataclasses import dataclass

from .loop_hints import positive_static_unroll_hint


@dataclass
class SubsetViolation:
    """A single violation of the Verifiable Triton subset."""

    message: str
    lineno: int | None = None

    def __str__(self) -> str:
        loc = f" (line {self.lineno})" if self.lineno else ""
        return f"VIOLATION{loc}: {self.message}"


def validate_triton_subset(source: str, kernel_name: str) -> list[SubsetViolation]:
    """Validate that a kernel stays within the verifiable Triton subset.

    Returns a list of violations (empty = valid).
    """
    tree = ast.parse(source)
    func = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == kernel_name:
            func = node
            break

    if func is None:
        return [SubsetViolation(f"Kernel function '{kernel_name}' not found")]

    constexpr_params = {
        argument.arg for argument in (*func.args.posonlyargs, *func.args.args, *func.args.kwonlyargs)
        if isinstance(argument.annotation, ast.Attribute)
        and isinstance(argument.annotation.value, ast.Name)
        and argument.annotation.value.id == "tl"
        and argument.annotation.attr == "constexpr"
    }
    checker = _SubsetChecker(constexpr_params)
    checker.visit(func)
    return checker.violations


# Allowed tl.* functions
_ALLOWED_TL_FUNCTIONS = {
    # Memory
    "make_block_ptr",
    "load",
    "store",
    # Arithmetic
    "dot",
    "exp2",
    "log2",
    "maximum",
    "minimum",
    "abs",
    "sigmoid",
    # Reductions
    "max",
    "min",
    "sum",
    # Shape manipulation
    "reshape",
    "broadcast_to",
    "trans",
    "cast",
    "view",
    # Creation
    "zeros",
    "full",
    "arange",
    # Control
    "where",
    "cdiv",
    "program_id",
    "constexpr",
    "debug_barrier",
    "range",
}

# Allowed tl.math.* functions
_ALLOWED_TL_MATH_FUNCTIONS = {
    "rsqrt",
}

# Source-call forms whose omitted arguments could change translated semantics.
# Memory operations have additional context-sensitive checks below.
_TL_CALL_SHAPES: dict[str, tuple[int, int, frozenset[str]]] = {
    "zeros": (1, 1, frozenset({"dtype"})),
    "full": (2, 2, frozenset({"dtype"})),
    "arange": (2, 2, frozenset()),
    "where": (3, 3, frozenset()),
    "dot": (2, 3, frozenset()),
    "trans": (1, 1, frozenset()),
    "exp2": (1, 1, frozenset()),
    "log2": (1, 1, frozenset()),
    "maximum": (2, 2, frozenset()),
    "minimum": (2, 2, frozenset()),
    "abs": (1, 1, frozenset()),
    "sigmoid": (1, 1, frozenset()),
    "max": (1, 2, frozenset({"axis"})),
    "min": (1, 2, frozenset({"axis"})),
    "sum": (1, 2, frozenset({"axis"})),
    "reshape": (2, 2, frozenset()),
    "broadcast_to": (2, 2, frozenset()),
    "cast": (2, 2, frozenset()),
    "view": (2, 2, frozenset()),
    "cdiv": (2, 2, frozenset()),
    "program_id": (0, 1, frozenset({"axis"})),
    "debug_barrier": (0, 0, frozenset()),
    "range": (1, 2, frozenset({"loop_unroll_factor"})),
}

# Banned tl.* functions
_BANNED_TL_FUNCTIONS = {
    "atomic_add",
    "atomic_max",
    "atomic_min",
    "atomic_cas",
    "atomic_xchg",
    "atomic_and",
    "atomic_or",
    "atomic_xor",
    "inline_asm",
}


class _SubsetChecker(ast.NodeVisitor):
    def __init__(self, constexpr_params: set[str]) -> None:
        self.constexpr_params = constexpr_params
        self.violations: list[SubsetViolation] = []
        self.block_ptr_vars: set[str] = set()  # vars assigned from make_block_ptr

    def _add(self, msg: str, node: ast.AST) -> None:
        lineno = getattr(node, "lineno", None)
        self.violations.append(SubsetViolation(msg, lineno))

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        func = node.func

        # Check for tl.* calls
        if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
            if func.value.id == "tl":
                method = func.attr

                # Banned functions
                if method in _BANNED_TL_FUNCTIONS:
                    self._add(
                        f"tl.{method} is not in the verifiable subset (no atomics allowed)",
                        node,
                    )

                # Unknown functions
                if method not in _ALLOWED_TL_FUNCTIONS and method not in _BANNED_TL_FUNCTIONS:
                    self._add(
                        f"tl.{method} is not recognized in the verifiable subset",
                        node,
                    )

                shape = _TL_CALL_SHAPES.get(method)
                if shape is not None:
                    self._check_call_shape(node, method, *shape)

                if method == "range":
                    for keyword in node.keywords:
                        if keyword.arg == "loop_unroll_factor" and not positive_static_unroll_hint(
                            keyword.value, self.constexpr_params
                        ):
                            self._add("tl.range unroll factor must be positive literals selected only by constexpr equality", keyword)

                # tl.load: must use block_ptr or simple scalar ptr
                if method == "load":
                    self._check_load(node)

                # tl.store: must use block_ptr
                if method == "store":
                    self._check_store(node)

        # Check for nested tl.math.* — allow specific functions
        if (
            isinstance(func, ast.Attribute)
            and isinstance(func.value, ast.Attribute)
            and isinstance(func.value.value, ast.Name)
            and func.value.value.id == "tl"
        ):
            if func.value.attr == "math":
                if func.attr not in _ALLOWED_TL_MATH_FUNCTIONS:
                    self._add(
                        f"tl.math.{func.attr} is not in the verifiable subset",
                        node,
                    )
                elif func.attr == "rsqrt":
                    self._check_call_shape(
                        node, "math.rsqrt", 1, 1, frozenset()
                    )
            elif func.value.attr == "libdevice":
                self._add(
                    f"tl.libdevice.{func.attr} is not in the verifiable subset",
                    node,
                )

        # Triton value casts are method calls rather than tl.* calls.
        if isinstance(func, ast.Attribute) and func.attr == "to":
            self._check_call_shape(node, "value.to", 1, 1, frozenset())

        self.generic_visit(node)

    def _check_call_shape(
        self,
        node: ast.Call,
        operation: str,
        minimum: int,
        maximum: int,
        allowed_keywords: frozenset[str],
    ) -> None:
        if not minimum <= len(node.args) <= maximum:
            expected = str(minimum) if minimum == maximum else f"{minimum}..{maximum}"
            self._add(
                f"tl.{operation} requires {expected} positional arguments",
                node,
            )
        self._reject_unmodeled_keywords(
            node, set(allowed_keywords), f"tl.{operation}"
        )
        has_keyword_axis = any(
            keyword.arg == "axis" for keyword in node.keywords
        )
        duplicate_axis = (
            operation in {"max", "min", "sum"}
            and len(node.args) == 2
            and has_keyword_axis
        ) or (
            operation == "program_id" and len(node.args) == 1 and has_keyword_axis
        )
        if duplicate_axis:
            self._add(f"tl.{operation} axis is specified twice", node)

    def visit_Assign(self, node: ast.Assign) -> None:
        # Track the current value, not whether a name was ever a block pointer.
        # Leaving a reassigned name in this set would let a later raw-pointer
        # load/store inherit the old block-pointer classification.
        if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            target = node.targets[0].id
            if isinstance(node.value, ast.Call) and self._is_tl_call(node.value, "make_block_ptr"):
                self.block_ptr_vars.add(target)
            else:
                self.block_ptr_vars.discard(target)
        self.generic_visit(node)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        if isinstance(node.target, ast.Name):
            self.block_ptr_vars.discard(node.target.id)
        self.generic_visit(node)

    def _reject_unmodeled_keywords(
        self, node: ast.Call, allowed: set[str], operation: str
    ) -> None:
        for keyword in node.keywords:
            if keyword.arg is None:
                self._add(f"{operation} keyword expansion is not modeled", keyword)
            elif keyword.arg not in allowed:
                self._add(
                    f"{operation} keyword {keyword.arg!r} is not modeled",
                    keyword,
                )

    def _check_load(self, node: ast.Call) -> None:
        """Verify tl.load uses block_ptr or is a simple scalar load."""
        if not node.args:
            self._add("tl.load requires a pointer argument", node)
            return

        arg0 = node.args[0]

        # Block pointer load: tl.load(block_ptr_var, ...)
        if isinstance(arg0, ast.Name) and arg0.id in self.block_ptr_vars:
            if len(node.args) != 1:
                self._add("tl.load(block_ptr) requires exactly one positional argument", node)
            self._reject_unmodeled_keywords(
                node, {"boundary_check", "padding_option"}, "tl.load(block_ptr)"
            )
            # Verify it has boundary_check and padding_option="zero"
            has_boundary = any(kw.arg == "boundary_check" for kw in node.keywords)
            padding = None
            for kw in node.keywords:
                if kw.arg == "padding_option" and isinstance(kw.value, ast.Constant):
                    padding = kw.value.value

            if not has_boundary:
                self._add(
                    "tl.load(block_ptr) should specify boundary_check for verification",
                    node,
                )
            if padding != "zero" and has_boundary:
                self._add(
                    'tl.load(block_ptr) padding other than "zero" is not modeled',
                    node,
                )
            return

        # Scalar pointer load (e.g. tl.load(cu_seqlens_q + bi))
        # This is allowed for loading from 1D index arrays
        if self._is_simple_scalar_ptr(arg0):
            if len(node.args) != 1 or node.keywords:
                self._add(
                    "Scalar pointer loads must be unmasked tl.load(pointer) calls",
                    node,
                )
            return

        # Raw pointer with mask — NOT allowed
        if any(kw.arg == "mask" for kw in node.keywords):
            self._add(
                "tl.load with raw pointer + mask is not in the verifiable subset. "
                "Use tl.make_block_ptr instead.",
                node,
            )
            return

        # Complex pointer arithmetic — NOT allowed
        if self._has_complex_ptr_arithmetic(arg0):
            self._add(
                "tl.load with pointer arithmetic is not in the verifiable subset. "
                "Use tl.make_block_ptr instead.",
                node,
            )
            return

        # Do not silently accept a pointer form merely because the coarse
        # "complex arithmetic" heuristic did not recognize it (for example a
        # vector `tl.arange` offset).  The allow-list above is exhaustive.
        self._add(
            "tl.load pointer expression is not in the verifiable subset",
            node,
        )

    def _check_store(self, node: ast.Call) -> None:
        """Verify tl.store uses block_ptr."""
        if len(node.args) < 2:
            self._add("tl.store requires pointer and value arguments", node)
            return

        arg0 = node.args[0]

        # Block pointer store
        if isinstance(arg0, ast.Name) and arg0.id in self.block_ptr_vars:
            if len(node.args) != 2:
                self._add("tl.store(block_ptr) requires exactly two positional arguments", node)
            self._reject_unmodeled_keywords(
                node, {"boundary_check"}, "tl.store(block_ptr)"
            )
            return

        # Raw pointer store — NOT allowed
        self._add(
            "tl.store with raw pointer is not in the verifiable subset. "
            "Use tl.make_block_ptr instead.",
            node,
        )

    def _is_simple_scalar_ptr(self, node: ast.expr) -> bool:
        """Check if pointer expr is a simple `base + scalar_offset` for 1D indexing."""
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            return self._is_scalar_offset(node.left) and self._is_scalar_offset(
                node.right
            )
        return False

    def _is_scalar_offset(self, node: ast.expr) -> bool:
        """Syntactic scalar arithmetic accepted for index-array lookups.

        Calls (notably ``tl.arange``) and subscripts are deliberately excluded.
        The translator performs the authoritative tensor/stride check later.
        """
        if isinstance(node, (ast.Name, ast.Constant)):
            return True
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            return self._is_scalar_offset(node.operand)
        if isinstance(node, ast.BinOp) and isinstance(
            node.op, (ast.Add, ast.Sub, ast.Mult)
        ):
            return self._is_scalar_offset(node.left) and self._is_scalar_offset(
                node.right
            )
        return False

    def _has_complex_ptr_arithmetic(self, node: ast.expr) -> bool:
        """Detect complex pointer arithmetic patterns (stride-based indexing)."""
        # If the expression contains multiplication with stride vars, it's complex
        for child in ast.walk(node):
            if isinstance(child, ast.BinOp) and isinstance(child.op, ast.Mult):
                return True
            if isinstance(child, ast.Subscript):
                # ptr[offsets] patterns
                return True
        return False

    @staticmethod
    def _is_tl_call(node: ast.expr, method: str) -> bool:
        if not isinstance(node, ast.Call):
            return False
        func = node.func
        return (
            isinstance(func, ast.Attribute)
            and func.attr == method
            and isinstance(func.value, ast.Name)
            and func.value.id == "tl"
        )
