"""
Translate a restricted subset of Triton kernel source code into the verification IR.

The "Verifiable Triton" subset requires:
  - Tensor block access via tl.make_block_ptr + tl.load/tl.store
  - Scalar pointer loads (e.g. tl.load(cu_seqlens + bi)) for index lookups
  - tl.reshape only for removing/adding size-1 dimensions (maps to Squeeze/Unsqueeze)
  - No tl.atomic_*, no inline PTX
  - Scalar conditions become explicit IR branches; tensor conditions are rejected

Usage:
    from ir.translate import translate_kernel_source
    kernel_ir = translate_kernel_source(open("triton_kernels/matmul.py").read(), "matmul_kernel")
"""

from __future__ import annotations

import ast
import re
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field

from .loop_hints import positive_static_unroll_hint

from . import (
    Expr,
    Var,
    IntLit,
    FloatLit,
    BoolLit,
    BinOp,
    Min,
    Max,
    Zeros,
    Full,
    Arange,
    Where,
    ReduceMax,
    ReduceSum,
    Exp2,
    Sigmoid,
    Rsqrt,
    Log2,
    Cast,
    Not,
    Maximum,
    Unsqueeze,
    Squeeze,
    BroadcastTo,
    Transpose,
    Slice,
    TensorIndex,
    TensorView,
    MaskedLoad,
    Stmt,
    VarDecl,
    Let,
    Assign,
    MaskedStore,
    Range,
    For,
    If,
    Grid,
    GridIter,
    Param,
    Kernel,
    Type,
    IntType,
    FloatType,
    BoolType,
    TensorType,
    v_,
    lit_,
    add_,
    sub_,
    mul_,
    div_,
    floordiv_,
    lt_,
    ge_,
    and_,
    cdiv_,
    mm_,
    range_,
    slice_,
    index_,
    masked_load_,
    masked_store_,
)
from .annotations import _extract_annotation_block, kernel_comment_prologue


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class TranslationError(ValueError):
    """Raised when the Triton source uses a construct outside the verifiable subset."""

    def __init__(self, message: str, node: ast.AST | None = None):
        loc = ""
        if node is not None and hasattr(node, "lineno"):
            loc = f" (line {node.lineno})"
        super().__init__(f"{message}{loc}")
        self.node = node


# ---------------------------------------------------------------------------
# Kernel interface annotations: @params(...) and @grid(...)
# ---------------------------------------------------------------------------


@dataclass
class TensorSpec:
    """Parsed logical tensor type, shape, and optional physical strides."""

    dims: list[Expr]  # shape dimensions as IR exprs
    stride_map: dict[str, int]  # stride_param_name -> dimension index
    # Integer metadata is specifically signed int32 in the verified deployment.
    # This makes an explicit `.to(tl.int32)` a semantic identity rather than an
    # unchecked narrowing operation in address arithmetic.
    is_int32: bool = False


@dataclass
class KernelInterface:
    """Shared typed interface and launch geometry for every proof goal."""

    grid_ranges: list[Expr]  # one per axis
    scalar_params: list[tuple[str, Type]]
    tensor_specs: dict[str, TensorSpec]  # tensor_name -> spec


def _parse_interface_expr(s: str) -> Expr:
    """Parse a simple expression string into an IR Expr.

    Supports: identifiers, integers, cdiv(a, b), add(a, b).
    """
    s = s.strip()
    # Integer literal
    if s.isdigit():
        return lit_(int(s))
    # Function calls: cdiv(a, b), add(a, b)
    m = re.match(r"(\w+)\((.+)\)$", s)
    if m:
        func_name = m.group(1)
        # Split args carefully (handle nested parens)
        args_str = m.group(2)
        args = _split_args(args_str)
        if func_name == "cdiv" and len(args) == 2:
            return cdiv_(_parse_interface_expr(args[0]), _parse_interface_expr(args[1]))
        if func_name == "add" and len(args) == 2:
            return add_(_parse_interface_expr(args[0]), _parse_interface_expr(args[1]))
        if func_name == "sub" and len(args) == 2:
            return sub_(_parse_interface_expr(args[0]), _parse_interface_expr(args[1]))
        if func_name == "mul" and len(args) == 2:
            return mul_(_parse_interface_expr(args[0]), _parse_interface_expr(args[1]))
        raise ValueError(f"Unknown function in kernel interface annotation: {func_name}")
    # Identifier
    if re.match(r"^[A-Za-z_]\w*$", s):
        return v_(s)
    raise ValueError(f"Cannot parse kernel interface expression: {s!r}")


def _split_args(s: str) -> list[str]:
    """Split comma-separated args, respecting nested parentheses."""
    args: list[str] = []
    depth = 0
    current: list[str] = []
    for ch in s:
        if ch == "(":
            depth += 1
            current.append(ch)
        elif ch == ")":
            depth -= 1
            if depth < 0:
                raise ValueError(f"Unbalanced ')' in argument list: {s!r}")
            current.append(ch)
        elif ch == "," and depth == 0:
            args.append("".join(current).strip())
            current = []
        else:
            current.append(ch)
    if current:
        args.append("".join(current).strip())
    if depth != 0:
        raise ValueError(f"Unbalanced parentheses in argument list: {s!r}")
    return args


def _parse_kernel_interface(
    source: str, parameter_names: set[str]
) -> KernelInterface | None:
    """Parse the unique ``@params`` and ``@grid`` blocks in one prologue."""

    params_text = _extract_annotation_block(source, "params")
    grid_text = _extract_annotation_block(source, "grid")
    if params_text is None and grid_text is None:
        return None
    if params_text is None or grid_text is None:
        missing = "@params" if params_text is None else "@grid"
        raise ValueError(f"Kernel interface is missing {missing}")

    grid_args = _split_args(grid_text)
    if not grid_args or any(not arg.strip() for arg in grid_args):
        raise ValueError("@grid must declare at least one non-empty axis")
    grid_ranges = [_parse_interface_expr(argument) for argument in grid_args]

    directives = _split_args(params_text)
    scalar_params: list[tuple[str, Type]] = []
    tensor_specs: dict[str, TensorSpec] = {}
    declared_names: set[str] = set()

    for directive in directives:
        directive = directive.strip().rstrip(",")
        if not directive:
            continue

        m = re.fullmatch(r"([A-Za-z_]\w*)\((.*)\)", directive)
        if not m:
            raise ValueError(f"Malformed @params directive: {directive!r}")

        kind = m.group(1)
        args_str = m.group(2)
        args = _split_args(args_str)

        if kind == "scalar":
            if len(args) != 2:
                raise ValueError(f"scalar(...) requires name and type: {directive!r}")
            pname = args[0].strip()
            if re.fullmatch(r"[A-Za-z_]\w*", pname) is None:
                raise ValueError(f"Invalid @params scalar name {pname!r}")
            ptype_str = args[1].strip()
            if pname in declared_names:
                raise ValueError(f"Duplicate @params name {pname!r}")
            if ptype_str == "int":
                ptype: Type = IntType()
            elif ptype_str == "float":
                ptype = FloatType()
            elif ptype_str == "bool":
                ptype = BoolType()
            else:
                raise ValueError(f"Unsupported @params scalar type {ptype_str!r}")
            scalar_params.append((pname, ptype))
            declared_names.add(pname)
        elif kind == "tensor":
            if len(args) not in {3, 4}:
                raise ValueError(
                    "tensor(...) requires name, element type, shape(...), and "
                    f"optional strides(...): {directive!r}"
                )
            tname = args[0].strip()
            if re.fullmatch(r"[A-Za-z_]\w*", tname) is None:
                raise ValueError(f"Invalid @params tensor name {tname!r}")
            if tname in declared_names:
                raise ValueError(f"Duplicate @params name {tname!r}")
            elem_type = args[1].strip()
            if elem_type not in {"float", "int32"}:
                raise ValueError(
                    f"Unsupported tensor element type {elem_type!r} for {tname!r}"
                )
            shape_match = re.fullmatch(r"shape\((.*)\)", args[2].strip())
            if shape_match is None:
                raise ValueError(f"Tensor {tname!r} requires an explicit shape(...)")
            shape_args = _split_args(shape_match.group(1))
            dims = [_parse_interface_expr(item) for item in shape_args]
            if not dims:
                raise ValueError(f"@params tensor {tname!r} has no dimensions")
            stride_items: list[str] = []
            if len(args) == 4:
                stride_match = re.fullmatch(r"strides\((.*)\)", args[3].strip())
                if stride_match is None:
                    raise ValueError(
                        f"Tensor {tname!r} fourth argument must be strides(...)"
                    )
                stride_items = [item.strip() for item in _split_args(stride_match.group(1))]
                if len(stride_items) != len(dims):
                    raise ValueError(
                        f"Tensor {tname!r} declares {len(dims)} dimensions but "
                        f"{len(stride_items)} strides"
                    )
                invalid_strides = sorted(
                    item
                    for item in stride_items
                    if re.fullmatch(r"[A-Za-z_]\w*", item) is None
                )
                if invalid_strides:
                    raise ValueError(
                        f"Tensor {tname!r} has invalid stride names: "
                        f"{invalid_strides}"
                    )
                if len(set(stride_items)) != len(stride_items):
                    raise ValueError(
                        f"Tensor {tname!r} declares duplicate stride parameters"
                    )
                undeclared = sorted(set(stride_items) - parameter_names)
                if undeclared:
                    raise ValueError(
                        f"Tensor {tname!r} names strides absent from the kernel "
                        f"signature: {undeclared}"
                    )
            smap = {sname: si for si, sname in enumerate(stride_items)}
            tensor_specs[tname] = TensorSpec(
                dims=dims,
                stride_map=smap,
                is_int32=(elem_type == "int32"),
            )
            declared_names.add(tname)
        else:
            raise ValueError(f"Unknown @params directive {kind!r}")

    declared_strides = {
        stride for spec in tensor_specs.values() for stride in spec.stride_map
    }
    scalar_stride_overlap = sorted(
        declared_strides & {name for name, _ in scalar_params}
    )
    if scalar_stride_overlap:
        raise ValueError(
            "@params names cannot be both scalar values and tensor strides: "
            f"{scalar_stride_overlap}"
        )
    return KernelInterface(
        grid_ranges=grid_ranges,
        scalar_params=scalar_params,
        tensor_specs=tensor_specs,
    )


# ---------------------------------------------------------------------------
# BlockPtr descriptor — captures info from tl.make_block_ptr(...)
# ---------------------------------------------------------------------------


@dataclass
class BlockPtrInfo:
    """Parsed representation of a tl.make_block_ptr call."""

    base_tensor: str  # name of the tensor param (e.g. "q", "k")
    shape: list[ast.expr]  # AST nodes for each dim of shape=(...)
    strides: list[ast.expr]  # source stride expression for each block-ptr dim
    offsets: list[ast.expr]  # AST nodes for each dim of offsets=(...)
    block_shape: list[ast.expr]  # AST nodes for each dim of block_shape=(...)
    # The full AST of the base argument (to detect pointer-offset patterns)
    base_ast: ast.expr | None = None


# ---------------------------------------------------------------------------
# Translation context
# ---------------------------------------------------------------------------


@dataclass
class TranslationContext:
    """Mutable state carried through translation."""

    # Maps Triton variable names to IR Var nodes
    vars: dict[str, Var] = field(default_factory=dict)

    # Maps variable names to their BlockPtrInfo (for tl.make_block_ptr results)
    block_ptrs: dict[str, BlockPtrInfo] = field(default_factory=dict)

    # Kernel parameters (positional names from the function signature)
    param_names: list[str] = field(default_factory=list)

    # constexpr parameters
    constexpr_params: set[str] = field(default_factory=set)

    # Stride parameters — maps (tensor_name, dim_index) -> stride_param_name
    # We don't need strides in IR, but we track them to skip them in params
    stride_params: set[str] = field(default_factory=set)

    # Tensor parameter shapes: tensor_name -> list of dim var names
    tensor_shapes: dict[str, list[str]] = field(default_factory=dict)

    # Grid iterators discovered from tl.program_id calls
    grid_iters: dict[int, str] = field(default_factory=dict)  # axis -> var_name

    # Local variable declarations accumulated during translation
    decls: list[VarDecl] = field(default_factory=list)

    # Track which names are declared as decls to avoid duplicates
    declared_vars: set[str] = field(default_factory=set)

    # Parsed shared kernel interface (if present)
    kernel_interface: KernelInterface | None = None

    # Pointer aliases: local_name -> full base AST expression
    # e.g. q_base -> q + q_start * stride_qt + hi * stride_qh
    ptr_aliases: dict[str, ast.expr] = field(default_factory=dict)

    # Track variables assigned from arange-based expressions:
    # name -> (start_ast, size_ast)  e.g. offs_m -> (qi, BLOCK_M) for `offs_m = qi + arange(0, BLOCK_M)`
    arange_vars: dict[str, tuple[ast.expr, ast.expr]] = field(default_factory=dict)

    # Track scalar-equivalent expressions for vectorized variables.
    # When page_ids = tl.load(block_table + bi*stride + page_slots*stride),
    # and page_slots comes from k_indices // PBS, the scalar equiv uses the tile-start.
    # name -> IR Expr for the scalar representation at tile start
    scalar_equiv: dict[str, Expr] = field(default_factory=dict)

    # Translation-time local types.  These are used only to validate source
    # reshape syntax before the fully checked IR type pass runs.
    local_types: dict[str, Type] = field(default_factory=dict)

    def get_var(self, name: str) -> Var:
        if name not in self.vars:
            self.vars[name] = v_(name)
        return self.vars[name]

    def declare_local(self, name: str, typ: Type) -> None:
        if name not in self.declared_vars:
            self.declared_vars.add(name)
            self.decls.append(VarDecl(self.get_var(name), typ))


# ---------------------------------------------------------------------------
# AST helpers
# ---------------------------------------------------------------------------


def _is_tl_call(node: ast.expr, method: str) -> bool:
    """Check if node is tl.<method>(...) or triton.language.<method>(...)."""
    if not isinstance(node, ast.Call):
        return False
    func = node.func
    if isinstance(func, ast.Attribute) and func.attr == method:
        if isinstance(func.value, ast.Name) and func.value.id == "tl":
            return True
    return False


def _is_tl_math_call(node: ast.expr, method: str) -> bool:
    """Check if node is tl.math.<method>(...)."""
    if not isinstance(node, ast.Call):
        return False
    func = node.func
    if (
        isinstance(func, ast.Attribute)
        and func.attr == method
        and isinstance(func.value, ast.Attribute)
        and func.value.attr == "math"
        and isinstance(func.value.value, ast.Name)
        and func.value.value.id == "tl"
    ):
        return True
    return False


def _is_tl_attr(node: ast.expr, method: str) -> bool:
    """Check if node is tl.<method>."""
    if isinstance(node, ast.Attribute) and node.attr == method:
        if isinstance(node.value, ast.Name) and node.value.id == "tl":
            return True
    return False


def _get_keyword(call: ast.Call, name: str) -> ast.expr | None:
    for kw in call.keywords:
        if kw.arg == name:
            return kw.value
    return None


def _require_call_shape(
    call: ast.Call,
    *,
    operation: str,
    positional: int,
    keywords: set[str],
) -> None:
    """Reject syntax whose runtime semantics this translator would erase."""
    if len(call.args) != positional:
        raise TranslationError(
            f"{operation} requires exactly {positional} positional arguments",
            call,
        )
    for keyword in call.keywords:
        if keyword.arg is None or keyword.arg not in keywords:
            name = "** expansion" if keyword.arg is None else repr(keyword.arg)
            raise TranslationError(
                f"{operation} keyword {name} is not modeled",
                keyword,
            )


_SCOPED_METADATA_FIELDS = (
    "block_ptrs",
    "ptr_aliases",
    "arange_vars",
    "scalar_equiv",
    "local_types",
)


@contextmanager
def _metadata_scope(ctx: TranslationContext) -> Iterator[None]:
    """Keep source-derived translation metadata in its lexical control scope.

    Block pointers and the auxiliary arange/scalar facts are compile-time
    descriptions, not IR values.  Letting a fact created in one branch leak
    into its sibling (or out of a loop) can erase a memory access or attach the
    wrong address to it.  Existing outer facts remain visible in the body.
    """
    saved = {
        field_name: dict(getattr(ctx, field_name))
        for field_name in _SCOPED_METADATA_FIELDS
    }
    try:
        yield
    finally:
        for field_name, value in saved.items():
            setattr(ctx, field_name, value)


def _translate_scoped_body(
    body: Sequence[ast.stmt], ctx: TranslationContext
) -> list[Stmt]:
    translated: list[Stmt] = []
    with _metadata_scope(ctx):
        for statement in body:
            translated.extend(translate_stmt(statement, ctx))
    return translated


def _tuple_elts(node: ast.expr) -> list[ast.expr]:
    """Extract elements from a Tuple AST node."""
    if isinstance(node, ast.Tuple):
        return list(node.elts)
    return [node]


def _const_int(node: ast.expr) -> int | None:
    """Extract a constant integer from an AST node."""
    if isinstance(node, ast.Constant) and isinstance(node.value, int):
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        v = _const_int(node.operand)
        if v is not None:
            return -v
    return None


def _program_id_axis(call: ast.Call) -> int:
    keyword_axis = _get_keyword(call, "axis")
    axis_node = keyword_axis if keyword_axis is not None else (
        call.args[0] if call.args else None
    )
    axis = _const_int(axis_node) if axis_node is not None else None
    if axis is None:
        raise TranslationError("tl.program_id axis must be a constant", call)
    return axis


def _cast_target(dtype_node: ast.expr, ctx: TranslationContext) -> tuple[str, str]:
    """Return ``(proof kind, source identity)`` for a modeled cast target."""
    if (
        isinstance(dtype_node, ast.Attribute)
        and isinstance(dtype_node.value, ast.Name)
        and dtype_node.value.id == "tl"
    ):
        if dtype_node.attr in {"float16", "float32", "float64", "bfloat16"}:
            return "float", f"tl.{dtype_node.attr}"
        if dtype_node.attr.startswith(("int", "uint")):
            return dtype_node.attr, f"tl.{dtype_node.attr}"

    # `output.dtype.element_ty`: recover the declared output element kind.
    if (
        isinstance(dtype_node, ast.Attribute)
        and dtype_node.attr == "element_ty"
        and isinstance(dtype_node.value, ast.Attribute)
        and dtype_node.value.attr == "dtype"
        and isinstance(dtype_node.value.value, ast.Name)
        and ctx.kernel_interface is not None
    ):
        tensor_spec = ctx.kernel_interface.tensor_specs.get(
            dtype_node.value.value.id
        )
        if tensor_spec is not None:
            kind = "int32" if tensor_spec.is_int32 else "float"
            return kind, ast.unparse(dtype_node)
    raise TranslationError(
        f"Cast target {ast.unparse(dtype_node)!r} is not modeled",
        dtype_node,
    )


def _cast_target_kind(dtype_node: ast.expr, ctx: TranslationContext) -> str:
    return _cast_target(dtype_node, ctx)[0]


def _require_float_constructor_dtype(
    call: ast.Call, ctx: TranslationContext
) -> None:
    dtype = _get_keyword(call, "dtype")
    if dtype is None:
        raise TranslationError(
            f"{ast.unparse(call.func)} requires an explicit modeled floating dtype",
            call,
        )
    if _cast_target_kind(dtype, ctx) != "float":
        raise TranslationError(
            f"{ast.unparse(call.func)} integer element types are not modeled",
            dtype,
        )


def _translate_modeled_cast(
    value_node: ast.expr,
    dtype_node: ast.expr,
    call: ast.Call,
    ctx: TranslationContext,
) -> Expr:
    """Retain modeled data casts while rejecting narrowing addresses.

    The regional theorem uses only congruence for floating data casts, while
    positional analysis uses explicitly qualified sentinel identities.  An
    integer cast can change an address, mask, or loop bound, so it is accepted
    only for a scalar load from trusted signed-int32 metadata.
    """
    value = translate_expr(value_node, ctx)
    target_kind, target_name = _cast_target(dtype_node, ctx)
    if target_kind == "float":
        return Cast(value, target_kind, target_name)
    if target_kind == "int32" and isinstance(value, TensorIndex):
        if ctx.kernel_interface is not None:
            tensor_spec = ctx.kernel_interface.tensor_specs.get(value.base.name)
            if tensor_spec is not None and tensor_spec.is_int32:
                return Cast(value, target_kind, target_name)
    raise TranslationError(
        "Integer casts are modeled as identity only for scalar loads from an "
        "@params tensor(..., int32, ...)",
        call,
    )


# ---------------------------------------------------------------------------
# Expression translator
# ---------------------------------------------------------------------------


def translate_expr(node: ast.expr, ctx: TranslationContext) -> Expr:
    """Translate a Python AST expression to an IR Expr."""

    # --- Constants ---
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool):
            return lit_(node.value)
        if isinstance(node.value, int):
            return lit_(node.value)
        if isinstance(node.value, float):
            return lit_(node.value)
        raise TranslationError(f"Unsupported constant: {node.value!r}", node)

    # --- Variables ---
    if isinstance(node, ast.Name):
        return ctx.get_var(node.id)

    # --- Unary ops ---
    if isinstance(node, ast.UnaryOp):
        if isinstance(node.op, ast.USub):
            operand = translate_expr(node.operand, ctx)
            if isinstance(operand, IntLit):
                return lit_(-operand.value)
            if isinstance(operand, FloatLit):
                return lit_(-operand.value)
            return BinOp("-", lit_(0), operand)
        if isinstance(node.op, ast.Not):
            return Not(translate_expr(node.operand, ctx))
        if isinstance(node.op, ast.Invert):
            return Not(translate_expr(node.operand, ctx))

    # --- Binary ops ---
    if isinstance(node, ast.BinOp):
        lhs = translate_expr(node.left, ctx)
        rhs = translate_expr(node.right, ctx)
        op_map: dict[type, str] = {
            ast.Add: "+",
            ast.Sub: "-",
            ast.Mult: "*",
            ast.Div: "/",
            ast.FloorDiv: "//",
            ast.Mod: "%",
            ast.MatMult: "@",
            ast.BitAnd: "and",
            ast.BitOr: "or",
        }
        op_type = type(node.op)
        if op_type in op_map:
            op_str = op_map[op_type]
            # Preserve source value operations. The integer geometry helpers
            # may cancel (x+c)-x or x-x, which is not valid IEEE arithmetic.
            if op_str == "*":
                return mul_(lhs, rhs)
            if op_str == "@":
                return mm_(lhs, rhs)
            return BinOp(op_str, lhs, rhs)
        raise TranslationError(f"Unsupported binary op: {type(node.op).__name__}", node)

    # --- Boolean ops ---
    if isinstance(node, ast.BoolOp):
        values = [translate_expr(v, ctx) for v in node.values]
        if isinstance(node.op, ast.And):
            result = values[0]
            for v in values[1:]:
                result = and_(result, v)
            return result
        if isinstance(node.op, ast.Or):
            result = values[0]
            for v in values[1:]:
                result = BinOp("or", result, v)
            return result

    # --- Compare ops ---
    if isinstance(node, ast.Compare):
        if len(node.ops) == 1 and len(node.comparators) == 1:
            lhs = translate_expr(node.left, ctx)
            rhs = translate_expr(node.comparators[0], ctx)
            op = node.ops[0]
            if isinstance(op, ast.Lt):
                return lt_(lhs, rhs)
            if isinstance(op, ast.LtE):
                return BinOp("<=", lhs, rhs)
            if isinstance(op, ast.Gt):
                return lt_(rhs, lhs)
            if isinstance(op, ast.GtE):
                return ge_(lhs, rhs)
            if isinstance(op, ast.Eq):
                return BinOp("==", lhs, rhs)
            if isinstance(op, ast.NotEq):
                return BinOp("!=", lhs, rhs)

    # --- Subscript (indexing / slicing) ---
    if isinstance(node, ast.Subscript):
        return _translate_subscript(node, ctx)

    # --- Attribute access ---
    if isinstance(node, ast.Attribute):
        # e.g. scores_max[:, None] -> handled by subscript
        pass

    # --- Function calls ---
    if isinstance(node, ast.Call):
        return _translate_call(node, ctx)

    # --- IfExp (ternary) ---
    if isinstance(node, ast.IfExp):
        return Where(
            translate_expr(node.test, ctx),
            translate_expr(node.body, ctx),
            translate_expr(node.orelse, ctx),
        )

    raise TranslationError(f"Unsupported expression: {ast.dump(node)}", node)


def _translate_subscript(node: ast.Subscript, ctx: TranslationContext) -> Expr:
    """Translate tensor[i] or tensor[i, j] or tensor[:, None] etc."""
    base = translate_expr(node.value, ctx)
    sl = node.slice

    # Single index: tensor[i]
    if isinstance(sl, ast.Constant) or isinstance(sl, ast.Name) or isinstance(sl, ast.BinOp):
        idx = translate_expr(sl, ctx)
        if isinstance(base, Var):
            return index_(base, [idx])
        raise TranslationError(f"Subscript on non-Var base: {ast.dump(node)}", node)

    # Tuple index: tensor[i, j, ...] or tensor[:, None]
    if isinstance(sl, ast.Tuple):
        elts = sl.elts
        # Check for None elements (unsqueeze) and slice elements
        has_none = any(
            isinstance(e, ast.Constant) and e.value is None for e in elts
        )
        has_slice = any(isinstance(e, ast.Slice) for e in elts)

        if has_none:
            # This is broadcasting/unsqueezing: x[None, :] or x[:, None]
            # Only a literal full slice is an identity.  Silently dropping an
            # integer index or a nontrivial slice here would translate a
            # different tensor value and can invalidate dependency regions.
            for element in elts:
                if isinstance(element, ast.Constant) and element.value is None:
                    continue
                if (
                    isinstance(element, ast.Slice)
                    and element.lower is None
                    and element.upper is None
                    and element.step is None
                ):
                    continue
                raise TranslationError(
                    "Unsqueeze indexing supports only `None` and full `:` slices",
                    element,
                )
            result = base
            for i, e in enumerate(elts):
                if isinstance(e, ast.Constant) and e.value is None:
                    result = Unsqueeze(result, i)
            return result

        if not has_slice:
            # Pure integer indexing
            indices = [translate_expr(e, ctx) for e in elts]
            if isinstance(base, Var):
                return index_(base, indices)

    raise TranslationError(f"Unsupported subscript: {ast.dump(node)}", node)


def _translate_call(node: ast.Call, ctx: TranslationContext) -> Expr:
    """Translate a function call expression."""

    # --- tl.zeros ---
    if _is_tl_call(node, "zeros"):
        _require_float_constructor_dtype(node, ctx)
        shape_node = node.args[0]
        shape = [translate_expr(e, ctx) for e in _tuple_elts(shape_node)]
        return Zeros(shape)

    # --- tl.full ---
    if _is_tl_call(node, "full"):
        _require_float_constructor_dtype(node, ctx)
        shape_node = node.args[0]
        value_node = node.args[1]
        shape = [translate_expr(e, ctx) for e in _tuple_elts(shape_node)]
        value = translate_expr(value_node, ctx)
        return Full(shape, value)

    # --- tl.arange ---
    if _is_tl_call(node, "arange"):
        start = translate_expr(node.args[0], ctx)
        stop = translate_expr(node.args[1], ctx)
        return Arange(start, stop)

    # --- tl.where ---
    if _is_tl_call(node, "where"):
        cond = translate_expr(node.args[0], ctx)
        on_true = translate_expr(node.args[1], ctx)
        on_false = translate_expr(node.args[2], ctx)
        return Where(cond, on_true, on_false)

    # --- tl.dot ---
    if _is_tl_call(node, "dot"):
        if len(node.args) != 2 or node.keywords:
            raise TranslationError(
                "Three-argument tl.dot is supported only as the complete "
                "right-hand side of an assignment",
                node,
            )
        lhs = translate_expr(node.args[0], ctx)
        rhs = translate_expr(node.args[1], ctx)
        return mm_(lhs, rhs)

    # --- tl.trans ---
    if _is_tl_call(node, "trans"):
        value = translate_expr(node.args[0], ctx)
        return Transpose(value, [1, 0])

    # --- tl.exp2 ---
    if _is_tl_call(node, "exp2"):
        return Exp2(translate_expr(node.args[0], ctx))

    # --- tl.sigmoid ---
    if _is_tl_call(node, "sigmoid"):
        return Sigmoid(translate_expr(node.args[0], ctx))

    # --- tl.math.rsqrt ---
    if _is_tl_math_call(node, "rsqrt"):
        return Rsqrt(translate_expr(node.args[0], ctx))

    # --- tl.maximum ---
    if _is_tl_call(node, "maximum"):
        lhs = translate_expr(node.args[0], ctx)
        rhs = translate_expr(node.args[1], ctx)
        return Maximum(lhs, rhs)

    # --- tl.minimum ---
    if _is_tl_call(node, "minimum"):
        lhs = translate_expr(node.args[0], ctx)
        rhs = translate_expr(node.args[1], ctx)
        return Min([lhs, rhs])

    # --- tl.log2 dependency-only abstraction ---
    # The node is retained so deployment qualification can test the concrete
    # backend operation.  Proof passes use only pointwise congruence.
    if _is_tl_call(node, "log2"):
        return Log2(translate_expr(node.args[0], ctx))

    # --- tl.max (reduction) ---
    if _is_tl_call(node, "max"):
        value = translate_expr(node.args[0], ctx)
        axis_node = _get_keyword(node, "axis")
        if axis_node is None and len(node.args) > 1:
            axis_node = node.args[1]
        axis = _const_int(axis_node) if axis_node else 0
        if axis is None:
            raise TranslationError("tl.max axis must be a constant", node)
        return ReduceMax(value, axis)

    # --- tl.sum (reduction) ---
    if _is_tl_call(node, "sum"):
        value = translate_expr(node.args[0], ctx)
        axis_node = _get_keyword(node, "axis")
        if axis_node is None and len(node.args) > 1:
            axis_node = node.args[1]
        axis = _const_int(axis_node) if axis_node else 0
        if axis is None:
            raise TranslationError("tl.sum axis must be a constant", node)
        return ReduceSum(value, axis)

    # --- tl.broadcast_to ---
    if _is_tl_call(node, "broadcast_to"):
        value = translate_expr(node.args[0], ctx)
        shape_node = node.args[1]
        shape = [translate_expr(e, ctx) for e in _tuple_elts(shape_node)]
        return BroadcastTo(value, shape)

    # --- tl.cdiv ---
    if _is_tl_call(node, "cdiv"):
        lhs = translate_expr(node.args[0], ctx)
        rhs = translate_expr(node.args[1], ctx)
        return cdiv_(lhs, rhs)

    # --- tl.load (scalar from pointer arithmetic, e.g. cu_seqlens_q + bi) ---
    if _is_tl_call(node, "load"):
        return _translate_tl_load(node, ctx)

    # --- tl.reshape ---
    if _is_tl_call(node, "reshape"):
        return _translate_reshape(node, ctx)

    # --- tl.cast (dependency-preserving data cast) ---
    if _is_tl_call(node, "cast"):
        return _translate_modeled_cast(node.args[0], node.args[1], node, ctx)

    # --- tl.program_id --- (should be handled at stmt level, but just in case)
    if _is_tl_call(node, "program_id"):
        axis = _program_id_axis(node)
        if axis in ctx.grid_iters:
            return ctx.get_var(ctx.grid_iters[axis])
        raise TranslationError("tl.program_id outside of grid setup", node)

    # --- min() builtin ---
    if isinstance(node.func, ast.Name) and node.func.id == "min":
        args = [translate_expr(a, ctx) for a in node.args]
        return Min(args)

    # --- max() builtin ---
    if isinstance(node.func, ast.Name) and node.func.id == "max":
        args = [translate_expr(a, ctx) for a in node.args]
        return Max(args)

    # --- float() / int() builtins (e.g. float("-inf")) ---
    if isinstance(node.func, ast.Name) and node.func.id == "float":
        if len(node.args) == 1 and isinstance(node.args[0], ast.Constant):
            val = node.args[0].value
            if val == "-inf":
                return lit_(float("-inf"))
            if val == "inf":
                return lit_(float("inf"))
            return lit_(float(val))

    if isinstance(node.func, ast.Name) and node.func.id == "int":
        if len(node.args) == 1 and isinstance(node.args[0], ast.Constant):
            return lit_(int(node.args[0].value))

    # --- range() ---
    if isinstance(node.func, ast.Name) and node.func.id == "range":
        # This shouldn't appear as an expression, only in for loops
        raise TranslationError("range() as expression is not supported", node)

    # --- Method calls: x.to(dtype) ---
    if isinstance(node.func, ast.Attribute):
        method = node.func.attr
        if method == "to":
            return _translate_modeled_cast(
                node.func.value, node.args[0], node, ctx
            )

    raise TranslationError(f"Unsupported call: {ast.dump(node)}", node)


def _translate_tl_load(node: ast.Call, ctx: TranslationContext) -> Expr:
    """Translate tl.load(...).

    Two forms:
    1. tl.load(block_ptr, boundary_check=(...), padding_option="zero")
       -> MaskedLoad (resolved later when we know the block_ptr info)
    2. tl.load(cu_seqlens_q + bi) — scalar pointer load
       -> TensorIndex
    """
    arg0 = node.args[0]

    # Form 1: tl.load(block_ptr_var, ...)
    if isinstance(arg0, ast.Name) and arg0.id in ctx.block_ptrs:
        return _translate_block_ptr_load(arg0.id, node, ctx)

    # Form 2: unmasked scalar pointer load — tl.load(base + offset).
    # We translate ptr + offset as a TensorIndex.
    if len(node.args) != 1 or node.keywords:
        raise TranslationError(
            "Scalar pointer loads must be unmasked `tl.load(pointer)` calls; "
            "vector masks and `other` values are not modeled",
            node,
        )
    return _translate_scalar_ptr_load(arg0, node, ctx)


def _parse_boundary_axes(node: ast.Call, rank: int) -> set[int]:
    boundary = _get_keyword(node, "boundary_check")
    if boundary is None:
        return set()
    axes: set[int] = set()
    for axis_node in _tuple_elts(boundary):
        axis = _const_int(axis_node)
        if axis is None or axis < 0 or axis >= rank:
            raise TranslationError(
                f"boundary_check axis must be a constant in 0..{rank - 1}",
                axis_node,
            )
        if axis in axes:
            raise TranslationError(f"Duplicate boundary_check axis {axis}", axis_node)
        axes.add(axis)
    return axes


def _translate_block_ptr_load(ptr_name: str, node: ast.Call, ctx: TranslationContext) -> Expr:
    """Translate a tl.load(block_ptr) into a MaskedLoad."""
    _require_call_shape(
        node,
        operation="tl.load(block_ptr)",
        positional=1,
        keywords={"boundary_check", "padding_option"},
    )
    padding = _get_keyword(node, "padding_option")
    if not (
        isinstance(padding, ast.Constant) and padding.value == "zero"
    ):
        raise TranslationError(
            'tl.load(block_ptr) requires padding_option="zero"', node
        )
    info = ctx.block_ptrs[ptr_name]
    base_var = ctx.get_var(info.base_tensor)
    boundary_axes = _parse_boundary_axes(node, len(info.shape))

    # Check if the base is a pointer-offset expression (not a simple Name)
    # and we have an @params tensor specification for this tensor
    if (
        info.base_ast is not None
        and not isinstance(info.base_ast, ast.Name)
        and ctx.kernel_interface is not None
        and info.base_tensor in ctx.kernel_interface.tensor_specs
    ):
        return _translate_ptr_offset_load(info, ctx, boundary_axes)

    # Simple case: base is just a tensor name
    # Build the region: for each dim, slice from offset to offset + block_size
    region: list[Slice] = []
    for i, (off_ast, bs_ast) in enumerate(zip(info.offsets, info.block_shape)):
        off = translate_expr(off_ast, ctx)
        bs = translate_expr(bs_ast, ctx)
        region.append(slice_(off, add_(off, bs)))

    # Build the mask: for boundary_check axes, use 0..shape[i]; others use full range
    mask: list[Slice] = []
    for i, shape_ast in enumerate(info.shape):
        if i in boundary_axes:
            shape_expr = translate_expr(shape_ast, ctx)
            mask.append(slice_(lit_(0), shape_expr))
        else:
            # An unchecked axis is not zero-padded.  Preserve every physical
            # access in the dependency model; memory safety is a separate
            # precondition obligation.
            mask.append(region[i])

    return masked_load_(base_var, region, mask)


def _extract_additive_terms(node: ast.expr) -> list[ast.expr]:
    """Flatten ``a + b + c`` into ``[a, b, c]``."""
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return _extract_additive_terms(node.left) + _extract_additive_terms(node.right)
    return [node]


def _parse_base_ptr_offsets(
    base_ast: ast.expr, stride_params: set[str]
) -> dict[str, ast.expr]:
    """Parse a pointer-offset expression like `q + q_start * stride_qt + hi * stride_qh`.

    Returns a mapping from stride param name to the offset expression:
        {"stride_qt": q_start_ast, "stride_qh": hi_ast}
    """
    offsets: dict[str, ast.expr] = {}
    terms = _extract_additive_terms(base_ast)
    saw_base = False
    for term in terms:
        if isinstance(term, ast.Name):
            if saw_base:
                raise TranslationError(
                    f"Unsupported bare term {term.id!r} in block pointer base", term
                )
            saw_base = True
            continue
        if isinstance(term, ast.BinOp) and isinstance(term.op, ast.Mult):
            # Check if one side is a stride param
            left_name = term.left.id if isinstance(term.left, ast.Name) else None
            right_name = term.right.id if isinstance(term.right, ast.Name) else None
            if left_name in stride_params:
                stride_name = left_name
                offset = term.right
            elif right_name in stride_params:
                stride_name = right_name
                offset = term.left
            else:
                raise TranslationError(
                    "Every block pointer base offset must be multiplied by a "
                    "declared stride parameter",
                    term,
                )
            if stride_name in offsets:
                raise TranslationError(
                    f"Duplicate offset for stride {stride_name!r}", term
                )
            offsets[stride_name] = offset
            continue
        raise TranslationError(
            f"Unsupported term in block pointer base: {ast.unparse(term)}", term
        )

    if not saw_base:
        raise TranslationError("Block pointer base has no tensor parameter", base_ast)

    return offsets


def _build_ptr_offset_region(
    info: BlockPtrInfo, ctx: TranslationContext, boundary_axes: set[int]
) -> tuple[list[Slice], list[Slice], list[int]]:
    """Build absolute N-dimensional region from a pointer-offset block_ptr.

    Returns (region, mask, squeeze_dims) where squeeze_dims are the tensor dims
    that have size 1 (from base pointer offsets with no block_ptr dim).

    Example: q_base = q + q_start * stride_qt + hi * stride_qh
             make_block_ptr(q_base, shape=(q_len, D), offsets=(li, 0), block_shape=(BLOCK_M, D))
    Tensor q is [Tq, H, D]. Stride map: stride_qt→dim0, stride_qh→dim1, stride_qd→dim2.
    Base offsets: stride_qt→q_start, stride_qh→hi.

    Reconstruction:
      dim0 (Tq): stride offset q_start, block_ptr dim0 offset li, block BLOCK_M
                  → [q_start + li : q_start + li + BLOCK_M]
      dim1 (H):  stride offset hi, NO block_ptr dim (absorbed into base)
                  → [hi : hi + 1]  ← squeeze_dim
      dim2 (D):  no stride offset, block_ptr dim1 offset 0, block D
                  → [0 : D]
    """
    assert ctx.kernel_interface is not None
    tspec = ctx.kernel_interface.tensor_specs[info.base_tensor]
    full_ndim = len(tspec.dims)

    # Parse the base pointer expression to get stride->offset mapping
    base_offsets = _parse_base_ptr_offsets(info.base_ast, ctx.stride_params)

    # Map source strides to full tensor dimensions.  This is the authoritative
    # correspondence; tuple position alone is insufficient once the base
    # pointer absorbs one or more dimensions.
    block_dim_for_tensor_dim: dict[int, int] = {}
    for block_dim, stride_ast in enumerate(info.strides):
        if not isinstance(stride_ast, ast.Name):
            raise TranslationError(
                "make_block_ptr strides must be named kernel stride parameters",
                stride_ast,
            )
        stride_name = stride_ast.id
        if stride_name not in tspec.stride_map:
            raise TranslationError(
                f"Stride {stride_name!r} is not declared for tensor "
                f"{info.base_tensor!r} in @params",
                stride_ast,
            )
        tensor_dim = tspec.stride_map[stride_name]
        if tensor_dim in block_dim_for_tensor_dim:
            raise TranslationError(
                f"Multiple block-pointer dimensions map to tensor axis {tensor_dim}",
                stride_ast,
            )
        block_dim_for_tensor_dim[tensor_dim] = block_dim

    # Determine which tensor dims have stride offsets from the base expression.
    stride_offset_exprs: dict[int, ast.expr] = {}
    for stride_name, offset_expr in base_offsets.items():
        if stride_name not in tspec.stride_map:
            raise TranslationError(
                f"Base offset stride {stride_name!r} is not declared for "
                f"tensor {info.base_tensor!r}",
                offset_expr,
            )
        mapped_dim = tspec.stride_map[stride_name]
        stride_offset_exprs[mapped_dim] = offset_expr

    covered_dims = set(block_dim_for_tensor_dim) | set(stride_offset_exprs)
    if covered_dims != set(range(full_ndim)):
        raise TranslationError(
            f"Block pointer for {info.base_tensor!r} does not describe every "
            f"tensor axis: covered={sorted(covered_dims)}, rank={full_ndim}",
            info.base_ast,
        )

    dim_regions: list[tuple[Expr, Expr]] = []
    squeeze_dims: list[int] = []

    for dim_idx in range(full_ndim):
        base_offset = (
            translate_expr(stride_offset_exprs[dim_idx], ctx)
            if dim_idx in stride_offset_exprs
            else lit_(0)
        )
        if dim_idx in block_dim_for_tensor_dim:
            block_dim = block_dim_for_tensor_dim[dim_idx]
            block_offset = translate_expr(info.offsets[block_dim], ctx)
            block_size = translate_expr(info.block_shape[block_dim], ctx)
            start = add_(base_offset, block_offset)
            dim_regions.append((start, add_(start, block_size)))
        else:
            dim_regions.append((base_offset, add_(base_offset, lit_(1))))
            squeeze_dims.append(dim_idx)

    region = [slice_(start, stop) for start, stop in dim_regions]

    # Build mask from the make_block_ptr shape + base offsets.
    # The boundary_check in make_block_ptr clamps to the shape passed to it,
    # which is a sub-region starting at the base pointer offset.
    # In absolute coords: for each block_ptr dim, the valid range is
    # [stride_offset : stride_offset + block_ptr_shape_dim].
    # For dims not covered by block_ptr dims, use the full tensor dimension.
    mask_slices: list[Slice] = []
    for dim_idx in range(full_ndim):
        if dim_idx in block_dim_for_tensor_dim:
            block_dim = block_dim_for_tensor_dim[dim_idx]
            if block_dim in boundary_axes:
                base_offset = (
                    translate_expr(stride_offset_exprs[dim_idx], ctx)
                    if dim_idx in stride_offset_exprs
                    else lit_(0)
                )
                shape_dim = translate_expr(info.shape[block_dim], ctx)
                mask_slices.append(
                    slice_(base_offset, add_(base_offset, shape_dim))
                )
            else:
                start, stop = dim_regions[dim_idx]
                mask_slices.append(slice_(start, stop))
        else:
            start, stop = dim_regions[dim_idx]
            mask_slices.append(slice_(start, stop))
    mask = mask_slices

    return region, mask, squeeze_dims


def _translate_ptr_offset_load(
    info: BlockPtrInfo, ctx: TranslationContext, boundary_axes: set[int]
) -> Expr:
    """Translate a block_ptr load where the base is a pointer-offset expression."""
    base_var = ctx.get_var(info.base_tensor)
    region, mask, squeeze_dims = _build_ptr_offset_region(
        info, ctx, boundary_axes
    )

    result: Expr = masked_load_(base_var, region, mask)

    # Auto-squeeze size-1 dimensions from base pointer offsets
    for offset, dim in enumerate(squeeze_dims):
        result = Squeeze(result, dim - offset)

    return result


def _translate_scalar_ptr_load(arg0: ast.expr, node: ast.Call, ctx: TranslationContext) -> Expr:
    """Translate tl.load(base_ptr + offset) as TensorIndex.

    For multi-dimensional tensors with an @params annotation providing a stride_map,
    decompose ``ptr + i * stride_a + j * stride_b`` into ``tensor[i, j]`` using
    the stride→dim mapping.  For 1-D tensors (like cu_seqlens) all additive index
    terms are combined into a single flat index.
    """
    # Resolve ptr alias if the argument is a variable name
    resolved_arg0 = arg0
    if isinstance(arg0, ast.Name) and arg0.id in ctx.ptr_aliases:
        resolved_arg0 = ctx.ptr_aliases[arg0.id]

    # Check if we can use stride-based multi-dim decomposition
    base_name = _extract_base_tensor_name(resolved_arg0, ctx)
    base_var = ctx.get_var(base_name)
    tensor_offsets = sorted({
        child.id
        for child in ast.walk(resolved_arg0)
        if isinstance(child, ast.Name)
        and child.id != base_name
        and isinstance(ctx.local_types.get(child.id), TensorType)
    })
    if tensor_offsets:
        raise TranslationError(
            "Raw pointer loads must use scalar offsets; tensor-valued offsets "
            f"are not modeled: {tensor_offsets}",
            arg0,
        )
    if (
        ctx.kernel_interface is None
        or base_name not in ctx.kernel_interface.tensor_specs
    ):
        raise TranslationError(
            f"Scalar pointer base {base_name!r} has no @params tensor declaration",
            arg0,
        )
    tspec = ctx.kernel_interface.tensor_specs[base_name]
    if tspec.stride_map:
        return _translate_multidim_scalar_load(
            resolved_arg0, base_name, base_var, tspec, ctx
        )
    if len(tspec.dims) != 1:
        raise TranslationError(
            f"Scalar pointer base {base_name!r} has rank {len(tspec.dims)} "
            "but no stride-to-axis declarations",
            arg0,
        )

    # Fallback: 1-D tensor — combine all indices
    parsed_base, indices = _parse_ptr_expr(resolved_arg0, ctx)
    if parsed_base != base_name:
        raise TranslationError(
            f"Scalar pointer base changed during alias resolution: "
            f"{parsed_base!r} != {base_name!r}",
            arg0,
        )
    if len(indices) > 1:
        combined = indices[0]
        for idx in indices[1:]:
            combined = add_(combined, idx)
        return index_(base_var, [combined])
    if not indices:
        indices = [lit_(0)]
    return index_(base_var, indices)




def _translate_multidim_scalar_load(
    arg0: ast.expr,
    base_name: str,
    base_var: Var,
    tspec: TensorSpec,
    ctx: TranslationContext,
) -> Expr:
    """Decompose ``ptr + i * stride_a + j * stride_b`` into ``tensor[i, j]``."""
    terms = _extract_additive_terms(arg0)
    dim_indices: dict[int, Expr] = {}

    for term in terms:
        # Skip the base tensor name
        if isinstance(term, ast.Name) and term.id == base_name:
            continue
        # Look for `expr * stride` or `stride * expr`
        if isinstance(term, ast.BinOp) and isinstance(term.op, ast.Mult):
            stride_name: str | None = None
            index_ast: ast.expr | None = None
            if isinstance(term.left, ast.Name) and term.left.id in tspec.stride_map:
                stride_name = term.left.id
                index_ast = term.right
            elif isinstance(term.right, ast.Name) and term.right.id in tspec.stride_map:
                stride_name = term.right.id
                index_ast = term.left
            if stride_name is not None and index_ast is not None:
                dim = tspec.stride_map[stride_name]
                if dim in dim_indices:
                    raise TranslationError(
                        f"Multiple scalar-pointer terms map to tensor axis {dim}",
                        term,
                    )
                # Use scalar equivalent for vectorized variables (e.g. page_slots)
                if isinstance(index_ast, ast.Name) and index_ast.id in ctx.scalar_equiv:
                    dim_indices[dim] = ctx.scalar_equiv[index_ast.id]
                else:
                    dim_indices[dim] = translate_expr(index_ast, ctx)
                continue
        raise TranslationError(
            f"Cannot map scalar-pointer term {ast.unparse(term)!r} to a "
            f"declared stride of tensor {base_name!r}",
            term,
        )

    # Build index list in dimension order
    ndim = len(tspec.dims)
    indices: list[Expr] = []
    for d in range(ndim):
        if d in dim_indices:
            indices.append(dim_indices[d])
        else:
            indices.append(lit_(0))

    return index_(base_var, indices)


def _try_record_arange_var(
    name: str, value_node: ast.expr, ctx: TranslationContext
) -> None:
    """If value_node is an arange-based expression, record in ctx.arange_vars.

    Also records scalar equivalents for vectorized expressions derived from
    arange variables (e.g. page_slots = k_indices // PAGE_BLOCK_SIZE).

    Patterns:
      tl.arange(0, N)           → (Constant(0), N)
      expr + tl.arange(0, N)    → (expr, N)
      arange_var // C            → scalar_equiv = start // C
      arange_var % C             → scalar_equiv = start % C
    """
    arange_info = _extract_arange_info(value_node, ctx)
    if arange_info is not None:
        ctx.arange_vars[name] = arange_info
        return

    # Check for floor-div/mod on an arange variable: page_slots = k_indices // PBS
    if isinstance(value_node, ast.BinOp) and isinstance(value_node.op, (ast.FloorDiv, ast.Mod)):
        lhs = value_node.left
        if isinstance(lhs, ast.Name) and lhs.id in ctx.arange_vars:
            start_ast, size_ast = ctx.arange_vars[lhs.id]
            start_expr = translate_expr(start_ast, ctx)
            rhs_expr = translate_expr(value_node.right, ctx)
            if isinstance(value_node.op, ast.FloorDiv):
                ctx.scalar_equiv[name] = floordiv_(start_expr, rhs_expr)
            else:  # Mod
                ctx.scalar_equiv[name] = BinOp("%", start_expr, rhs_expr)
                # Also record as an arange-like var for range inference:
                # page_offsets = k_tile_start % PBS, size = min(BLOCK_N, PBS - offset)
                # For simplicity, record size as the arange size divided by nothing —
                # actually, when BLOCK_N <= PBS, page_offsets is a contiguous range of size BLOCK_N
                ctx.arange_vars[name] = (value_node, size_ast)  # size same as parent arange


def _extract_arange_info(
    node: ast.expr, ctx: TranslationContext | None = None
) -> tuple[ast.expr, ast.expr] | None:
    """Extract (start_ast, size_ast) from an arange-based expression.

    Returns None if not arange-based. If ctx is provided, also checks
    ctx.arange_vars for variable references.
    """
    # Check if this is a known arange variable
    if ctx is not None and isinstance(node, ast.Name) and node.id in ctx.arange_vars:
        return ctx.arange_vars[node.id]

    # Direct: tl.arange(0, N)
    if isinstance(node, ast.Call):
        func = node.func
        if (
            isinstance(func, ast.Attribute)
            and func.attr == "arange"
            and len(node.args) >= 2
        ):
            return (node.args[0], node.args[1])

    # Offset: expr + arange_expr or arange_expr + expr
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        right_info = _extract_arange_info(node.right, ctx)
        if right_info is not None:
            return (node.left, right_info[1])
        left_info = _extract_arange_info(node.left, ctx)
        if left_info is not None:
            return (node.right, left_info[1])

    return None



def _parse_ptr_expr(node: ast.expr, ctx: TranslationContext) -> tuple[str, list[Expr]]:
    """Parse a pointer expression like `cu_seqlens_q + bi` into (tensor_name, [index])."""
    if isinstance(node, ast.Name):
        return node.id, []

    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        # Recursively parse left side
        base_name, left_indices = _parse_ptr_expr(node.left, ctx)
        right_expr = translate_expr(node.right, ctx)
        return base_name, left_indices + [right_expr]

    raise TranslationError(f"Cannot parse pointer expression: {ast.dump(node)}", node)


def _translate_reshape(node: ast.Call, ctx: TranslationContext) -> Expr:
    """Translate tl.reshape(x, new_shape).

    Only supports removing size-1 dimensions (Squeeze).
    """
    inner = translate_expr(node.args[0], ctx)
    target_shape_node = node.args[1]
    target_dims = _tuple_elts(target_shape_node)

    # If the inner is a MaskedLoad from a block_ptr, use block_shape to identify
    # which dims are size-1 (the region arithmetic like (bi+1)-bi doesn't simplify
    # to IntLit(1), but block_shape entries are AST constants we can check).
    if isinstance(inner, MaskedLoad):
        loaded_ndim = len(inner.region)
        target_ndim = len(target_dims)
        if loaded_ndim > target_ndim:
            # Try to find the BlockPtrInfo for this load to get block_shape
            block_shape_exprs: list[Expr] | None = None
            inner_ast = node.args[0]  # the tl.load(...) AST node
            if isinstance(inner_ast, ast.Call) and _is_tl_call(inner_ast, "load"):
                ptr_arg = inner_ast.args[0]
                if isinstance(ptr_arg, ast.Name) and ptr_arg.id in ctx.block_ptrs:
                    bpi = ctx.block_ptrs[ptr_arg.id]
                    block_shape_exprs = [
                        translate_expr(dim, ctx) for dim in bpi.block_shape
                    ]
            if block_shape_exprs is None:
                raise TranslationError(
                    "Cannot establish source dimensions for squeeze reshape", node
                )

            target_shape = [translate_expr(dim, ctx) for dim in target_dims]

            def find_removed(
                source_index: int, target_index: int
            ) -> list[int] | None:
                if source_index == len(block_shape_exprs):
                    return [] if target_index == len(target_shape) else None
                source_dim = block_shape_exprs[source_index]
                if (
                    target_index < len(target_shape)
                    and source_dim == target_shape[target_index]
                ):
                    kept = find_removed(source_index + 1, target_index + 1)
                    if kept is not None:
                        return kept
                if source_dim == IntLit(1):
                    removed = find_removed(source_index + 1, target_index)
                    if removed is not None:
                        return [source_index, *removed]
                return None

            removed_axes = find_removed(0, 0)
            if removed_axes is None:
                raise TranslationError(
                    "tl.reshape target is not obtained by removing size-1 "
                    "block-pointer dimensions",
                    node,
                )
            result: Expr = inner
            for removed_before, axis in enumerate(removed_axes):
                result = Squeeze(result, axis - removed_before)
            return result

    # Adding size-1 dims (Unsqueeze): e.g. tl.reshape(acc, (1, BLOCK_M, BLOCK_N)).
    # Require an exact source shape; accepting merely because the target begins
    # with 1s can silently erase a genuine data-reordering reshape.
    target_shape = [translate_expr(dim, ctx) for dim in target_dims]
    target_const_dims = [_const_int(dim) for dim in target_dims]

    leading_ones = 0
    for c in target_const_dims:
        if c == 1:
            leading_ones += 1
        else:
            break

    if leading_ones > 0:
        source_type = _infer_local_type(inner, ctx, ctx.local_types)
        if not isinstance(source_type, TensorType):
            raise TranslationError(
                "Cannot establish source tensor dimensions for reshape", node
            )
        expected_shape = [lit_(1)] * leading_ones + list(source_type.dims)
        if target_shape != expected_shape:
            raise TranslationError(
                "tl.reshape target is not obtained by adding leading size-1 "
                "dimensions",
                node,
            )
        result = inner
        for i in range(leading_ones):
            result = Unsqueeze(result, 0)
        return result

    # Generic reshape — try to identify squeeze/unsqueeze pattern
    raise TranslationError(
        "tl.reshape is only supported for adding/removing size-1 dims",
        node,
    )


# ---------------------------------------------------------------------------
# make_block_ptr parser
# ---------------------------------------------------------------------------


def _parse_make_block_ptr(node: ast.Call, ctx: TranslationContext) -> BlockPtrInfo:
    """Parse a tl.make_block_ptr(...) call into a BlockPtrInfo."""
    allowed_keywords = {
        "base", "shape", "strides", "offsets", "block_shape", "order"
    }
    if len(node.args) > 1:
        raise TranslationError(
            "make_block_ptr accepts at most the base as a positional argument",
            node,
        )
    keyword_names = [keyword.arg for keyword in node.keywords]
    if any(name is None or name not in allowed_keywords for name in keyword_names):
        raise TranslationError("make_block_ptr has an unmodeled keyword", node)
    if len(keyword_names) != len(set(keyword_names)):
        raise TranslationError("make_block_ptr has a duplicate keyword", node)
    provided = set(keyword_names)
    if node.args:
        if "base" in provided:
            raise TranslationError("make_block_ptr base is specified twice", node)
        provided.add("base")
    missing = allowed_keywords - provided
    if missing:
        raise TranslationError(
            f"make_block_ptr is missing arguments: {sorted(missing)}", node
        )

    # Extract arguments
    base_arg = node.args[0] if node.args else _get_keyword(node, "base")
    shape_arg = _get_keyword(node, "shape") or (node.args[1] if len(node.args) > 1 else None)
    strides_arg = _get_keyword(node, "strides")
    offsets_arg = _get_keyword(node, "offsets") or (node.args[3] if len(node.args) > 3 else None)
    block_shape_arg = _get_keyword(node, "block_shape") or (node.args[4] if len(node.args) > 4 else None)

    if any(a is None for a in [base_arg, shape_arg, strides_arg, offsets_arg, block_shape_arg]):
        raise TranslationError("make_block_ptr missing required arguments", node)

    # Parse base tensor name (may have pointer offset: q + q_start * stride_qt + ...)
    base_tensor = _extract_base_tensor_name(base_arg, ctx)

    # Resolve the actual base AST: if base_arg is a ptr alias name, use the alias expression
    resolved_base_ast = base_arg
    if isinstance(base_arg, ast.Name) and base_arg.id in ctx.ptr_aliases:
        resolved_base_ast = ctx.ptr_aliases[base_arg.id]

    shape = _tuple_elts(shape_arg)
    strides = _tuple_elts(strides_arg)
    offsets = _tuple_elts(offsets_arg)
    block_shape = _tuple_elts(block_shape_arg)
    if not (len(shape) == len(strides) == len(offsets) == len(block_shape)):
        raise TranslationError(
            "make_block_ptr shape/strides/offsets/block_shape ranks differ", node
        )
    order_arg = _get_keyword(node, "order")
    assert order_arg is not None
    order = [_const_int(axis) for axis in _tuple_elts(order_arg)]
    if any(axis is None for axis in order) or sorted(order) != list(range(len(shape))):
        raise TranslationError(
            "make_block_ptr order must be a constant permutation of its axes",
            order_arg,
        )

    if ctx.kernel_interface is None or base_tensor not in ctx.kernel_interface.tensor_specs:
        raise TranslationError(
            f"Block-pointer base {base_tensor!r} has no @params tensor declaration",
            node,
        )
    tspec = ctx.kernel_interface.tensor_specs[base_tensor]
    stride_dims: list[int] = []
    for stride in strides:
        if not isinstance(stride, ast.Name) or stride.id not in tspec.stride_map:
            raise TranslationError(
                f"Block-pointer stride {ast.unparse(stride)!r} is not declared "
                f"for tensor {base_tensor!r}",
                stride,
            )
        stride_dims.append(tspec.stride_map[stride.id])
    if len(set(stride_dims)) != len(stride_dims):
        raise TranslationError("Duplicate tensor axis in block-pointer strides", node)

    if isinstance(resolved_base_ast, ast.Name):
        if stride_dims != list(range(len(tspec.dims))):
            raise TranslationError(
                f"Direct block pointer for {base_tensor!r} must cover tensor "
                "axes in declaration order",
                node,
            )
        translated_shape = [translate_expr(dim, ctx) for dim in shape]
        if translated_shape != list(tspec.dims):
            raise TranslationError(
                f"Block-pointer shape for {base_tensor!r} disagrees with @params",
                node,
            )
    else:
        # Strictly parse now so no source offset can be silently discarded.
        base_offsets = _parse_base_ptr_offsets(resolved_base_ast, ctx.stride_params)
        unknown_offsets = set(base_offsets) - set(tspec.stride_map)
        if unknown_offsets:
            raise TranslationError(
                f"Pointer base for {base_tensor!r} uses strides declared for "
                f"another tensor: {sorted(unknown_offsets)}",
                node,
            )
        offset_dims = {
            tspec.stride_map[name] for name in base_offsets
        }
        if set(stride_dims) | offset_dims != set(range(len(tspec.dims))):
            raise TranslationError(
                f"Pointer-offset block for {base_tensor!r} does not cover all "
                "declared tensor axes",
                node,
            )

    return BlockPtrInfo(
        base_tensor=base_tensor,
        shape=shape,
        strides=strides,
        offsets=offsets,
        block_shape=block_shape,
        base_ast=resolved_base_ast,
    )


def _extract_base_tensor_name(node: ast.expr, ctx: TranslationContext | None) -> str:
    """Extract the tensor parameter name from a base pointer expression.

    Handles: `q`, `q + q_start * stride_qt + hi * stride_qh`, etc.
    Also resolves pointer aliases: if `q_base` is an alias for `q + ...`, returns `q`.
    """
    if isinstance(node, ast.Name):
        name = node.id
        # Resolve pointer alias if context is available
        if ctx is not None and name in ctx.ptr_aliases:
            return _extract_base_tensor_name(ctx.ptr_aliases[name], ctx)
        return name
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return _extract_base_tensor_name(node.left, ctx)
    raise TranslationError(f"Cannot extract base tensor from: {ast.dump(node)}", node)


# ---------------------------------------------------------------------------
# Statement translator
# ---------------------------------------------------------------------------


def _collect_var_names(expr: Expr, names: set[str]) -> None:
    """Recursively collect all Var names from an IR expression."""
    if isinstance(expr, Var):
        names.add(expr.name)
    elif isinstance(expr, BinOp):
        _collect_var_names(expr.lhs, names)
        _collect_var_names(expr.rhs, names)
    elif isinstance(expr, (IntLit, FloatLit, BoolLit)):
        pass
    # Add more cases as needed


def _ast_uses_name(node: ast.AST | None, name: str) -> bool:
    return node is not None and any(
        isinstance(child, ast.Name) and child.id == name
        for child in ast.walk(node)
    )


def _pointer_metadata_captures(ctx: TranslationContext, name: str) -> bool:
    """Whether live erased pointer metadata captured the current value of name.

    Triton pointer aliases and block pointers are values: changing a source
    variable later does not mutate an already-created pointer.  The translator
    stores their defining AST, so reject such reassignment instead of expanding
    that AST with the newer IR value.
    """

    if any(_ast_uses_name(alias, name) for alias in ctx.ptr_aliases.values()):
        return True
    for info in ctx.block_ptrs.values():
        nodes = [
            info.base_ast,
            *info.shape,
            *info.strides,
            *info.offsets,
            *info.block_shape,
        ]
        if any(_ast_uses_name(node, name) for node in nodes):
            return True
    return False


def _invalidate_value_metadata(ctx: TranslationContext, name: str) -> None:
    """Forget derived facts that describe the previous value of a local."""

    ctx.arange_vars.pop(name, None)
    ctx.scalar_equiv.pop(name, None)


def _is_ptr_alias_expr(node: ast.expr, ctx: TranslationContext) -> bool:
    """Check if an expression is a pointer alias: tensor + offset * stride + ...

    Returns True if the expression is an additive chain where the leftmost
    leaf is a tensor parameter name and stride parameters appear in the offsets.
    """
    if not isinstance(node, ast.BinOp) or not isinstance(node.op, ast.Add):
        return False
    # The leftmost leaf must be a simple Name (tensor param or known alias)
    leftmost = node.left
    while isinstance(leftmost, ast.BinOp) and isinstance(leftmost.op, ast.Add):
        leftmost = leftmost.left
    if not isinstance(leftmost, ast.Name):
        return False
    base_name = leftmost.id
    # Resolve aliases
    if base_name in ctx.ptr_aliases:
        base_name = _extract_base_tensor_name(ctx.ptr_aliases[base_name], ctx)
    if (
        ctx.kernel_interface is None
        or base_name not in ctx.kernel_interface.tensor_specs
    ):
        return False
    # Check that stride params appear in the expression
    for child in ast.walk(node):
        if isinstance(child, ast.Name) and child.id in ctx.stride_params:
            return True
    return False


def _is_scalar_arithmetic(node: ast.expr) -> bool:
    """Check if an AST expression is pure scalar arithmetic involving floor division.

    Only returns True for expressions containing ``//`` (floor-div), used to
    emit ``Let`` bindings for derived index computations like ``hkvi = hi * Hkv // H``.
    """
    if isinstance(node, ast.BinOp):
        if isinstance(node.op, ast.FloorDiv):
            return True
        if isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Mod)):
            return _is_scalar_arithmetic(node.left) or _is_scalar_arithmetic(node.right)
    return False


def translate_stmt(node: ast.stmt, ctx: TranslationContext) -> list[Stmt]:
    """Translate a Python AST statement to IR Stmts."""

    # --- Assignment: x = expr ---
    if isinstance(node, ast.Assign):
        return _translate_assign(node, ctx)

    # --- Augmented assignment: x += expr, x *= expr, etc. ---
    if isinstance(node, ast.AugAssign):
        return _translate_aug_assign(node, ctx)

    # --- For loop ---
    if isinstance(node, ast.For):
        return _translate_for(node, ctx)

    # --- If statement ---
    if isinstance(node, ast.If):
        return _translate_if(node, ctx)

    # --- Expression statement (e.g. tl.store(...)) ---
    if isinstance(node, ast.Expr):
        if isinstance(node.value, ast.Call):
            return _translate_expr_stmt(node.value, ctx)
        # A function docstring has no execution semantics.  Every other
        # expression statement must be rejected: silently dropping a helper
        # call or an unsupported side effect would prove a different program.
        if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            return []
        raise TranslationError("Unsupported expression statement", node)

    # --- Return (early return for guard conditions) ---
    if isinstance(node, ast.Return):
        # Early returns are handled by wrapping the rest in an If
        return []

    raise TranslationError(f"Unsupported statement: {ast.dump(node)}", node)


def _translate_assign(node: ast.Assign, ctx: TranslationContext) -> list[Stmt]:
    """Translate `target = value`."""
    if len(node.targets) != 1:
        raise TranslationError("Multiple assignment targets not supported", node)

    target_node = node.targets[0]

    # Simple variable assignment
    if isinstance(target_node, ast.Name):
        name = target_node.id
        value_node = node.value

        # Rebinding a kernel argument is legal Python/Triton, but treating the
        # assignment as if it never happened is not a sound translation.  The
        # IR currently enforces global no-shadowing, so require the source to
        # introduce a fresh local name instead.
        if name in ctx.param_names:
            raise TranslationError(
                f"Reassignment of kernel parameter {name!r} is not supported; "
                "use a fresh local variable",
                node,
            )
        if name in ctx.block_ptrs or name in ctx.ptr_aliases:
            raise TranslationError(
                f"Reassignment of pointer metadata local {name!r} is not supported",
                node,
            )
        if _pointer_metadata_captures(ctx, name):
            raise TranslationError(
                f"Reassignment of {name!r} captured by live pointer metadata "
                "is not supported",
                node,
            )
        _invalidate_value_metadata(ctx, name)

        # --- tl.program_id --- becomes a grid iterator (skip, handled at grid level)
        if _is_tl_call(value_node, "program_id"):
            raise TranslationError(
                "tl.program_id assignments must form the function's initial "
                "grid-iterator prefix",
                node,
            )

        # --- tl.make_block_ptr --- save info, no IR stmt
        if _is_tl_call(value_node, "make_block_ptr"):
            if name in ctx.vars:
                raise TranslationError(
                    f"Block-pointer local {name!r} shadows an existing value", node
                )
            info = _parse_make_block_ptr(value_node, ctx)
            ctx.block_ptrs[name] = info
            return []

        # --- Pointer alias: x_base = tensor + offset * stride + ... ---
        if _is_ptr_alias_expr(value_node, ctx):
            if name in ctx.vars:
                raise TranslationError(
                    f"Pointer alias {name!r} shadows an existing value", node
                )
            ctx.ptr_aliases[name] = value_node
            return []  # Don't emit IR; resolved when make_block_ptr uses it

        # --- Track arange-based assignments for tensor pointer decomposition ---
        _try_record_arange_var(name, value_node, ctx)

        # --- tl.load(scalar_ptr) --- becomes a Let binding
        if _is_tl_call(value_node, "load"):
            arg0 = value_node.args[0]
            if isinstance(arg0, ast.Name) and arg0.id in ctx.block_ptrs:
                # Block pointer load
                value = _translate_block_ptr_load(arg0.id, value_node, ctx)
                _remember_local_type(name, value, ctx)
                return [Assign(target=ctx.get_var(name), op=None, value=value)]
            else:
                # Scalar pointer load -> Let binding
                value = translate_expr(value_node, ctx)
                # If this is a multi-dim index (e.g. block_table[bi, page_slot]),
                # record as scalar_equiv for use in tensor pointer decomposition
                if isinstance(value, TensorIndex):
                    ctx.scalar_equiv[name] = value
                _remember_local_type(name, value, ctx)
                return [Let(var=ctx.get_var(name), value=value)]

        # --- tl.reshape(tl.load(block_ptr), shape) --- combined load+reshape
        if _is_tl_call(value_node, "reshape"):
            value = _translate_reshape(value_node, ctx)
            _remember_local_type(name, value, ctx)
            return [Assign(target=ctx.get_var(name), op=None, value=value)]

        # --- tl.dot(a, b, acc) --- with accumulator as 3rd arg
        if _is_tl_call(value_node, "dot") and len(value_node.args) >= 3:
            if len(value_node.args) != 3 or value_node.keywords:
                raise TranslationError(
                    "Three-argument tl.dot must have exactly `(a, b, acc)`",
                    value_node,
                )
            lhs = translate_expr(value_node.args[0], ctx)
            rhs = translate_expr(value_node.args[1], ctx)
            # acc is the 3rd argument; result is acc + dot(a, b)
            acc_name = value_node.args[2]
            if isinstance(acc_name, ast.Name) and acc_name.id == name:
                return [Assign(target=ctx.get_var(name), op="+", value=mm_(lhs, rhs))]
            else:
                # dot with different accumulator
                dot_result = mm_(lhs, rhs)
                acc_expr = translate_expr(acc_name, ctx)
                value = add_(acc_expr, dot_result)
                _remember_local_type(name, value, ctx)
                return [Assign(target=ctx.get_var(name), op=None, value=value)]

        # --- tl.cast(x, dtype) --- dependency-preserving data cast
        if _is_tl_call(value_node, "cast"):
            value = translate_expr(value_node, ctx)
            _remember_local_type(name, value, ctx)
            return [Assign(target=ctx.get_var(name), op=None, value=value)]

        # --- Pure scalar arithmetic → Let binding ---
        if _is_scalar_arithmetic(value_node):
            value = translate_expr(value_node, ctx)
            _remember_local_type(name, value, ctx)
            return [Let(var=ctx.get_var(name), value=value)]

        # --- Regular assignment ---
        value = translate_expr(value_node, ctx)
        _remember_local_type(name, value, ctx)
        return [Assign(target=ctx.get_var(name), op=None, value=value)]

    raise TranslationError(f"Unsupported assignment target: {ast.dump(target_node)}", node)


def _translate_aug_assign(node: ast.AugAssign, ctx: TranslationContext) -> list[Stmt]:
    """Translate x += expr, x *= expr, etc."""
    if not isinstance(node.target, ast.Name):
        raise TranslationError("Augmented assignment only supported for simple variables", node)

    name = node.target.id
    if name in ctx.param_names:
        raise TranslationError(
            f"Augmented assignment to kernel parameter {name!r} is not supported; "
            "use a fresh local variable",
            node,
        )
    if _pointer_metadata_captures(ctx, name):
        raise TranslationError(
            f"Reassignment of {name!r} captured by live pointer metadata is "
            "not supported",
            node,
        )
    _invalidate_value_metadata(ctx, name)
    target = ctx.get_var(name)

    op_map = {
        ast.Add: "+",
        ast.Sub: "-",
        ast.Mult: "*",
        ast.Div: "/",
        ast.BitAnd: "and",
        ast.BitOr: "or",
    }
    op_type = type(node.op)
    if op_type not in op_map:
        raise TranslationError(f"Unsupported augmented op: {type(node.op).__name__}", node)

    op_str = op_map[op_type]
    value = translate_expr(node.value, ctx)
    return [Assign(target=target, op=op_str, value=value)]


def _translate_for(node: ast.For, ctx: TranslationContext) -> list[Stmt]:
    """Translate `for var in range(...):`."""
    if not isinstance(node.target, ast.Name):
        raise TranslationError("For loop target must be a simple variable", node)
    if node.orelse:
        raise TranslationError("for ... else is not supported", node)

    var_name = node.target.id
    var = ctx.get_var(var_name)

    # Parse the range expression
    iter_node = node.iter
    if isinstance(iter_node, ast.Call) and isinstance(iter_node.func, ast.Name) and iter_node.func.id == "range":
        start, stop = _parse_range_call(iter_node, ctx)
    elif _is_tl_call(iter_node, "range"):
        # A statically selected unroll hint changes compiler scheduling, not the ordered
        # iteration domain or recurrence. Other tl.range attributes are not
        # admitted: in particular, do not silently erase a step or unknown flag.
        start, stop = _parse_range_call(iter_node, ctx, triton_range=True)
    else:
        raise TranslationError("For loop must iterate over range()", node)

    # Source-derived pointer/arange facts created in a loop body cannot be used
    # after the loop without a separate definite-assignment proof.
    body_stmts = _translate_scoped_body(node.body, ctx)

    return [For(var=var, iters=range_(start, stop), body=body_stmts)]


def _parse_range_call(node: ast.Call, ctx: TranslationContext, *, triton_range: bool = False) -> tuple[Expr, Expr]:
    """Parse range(stop) or range(start, stop)."""
    seen = set()
    for keyword in node.keywords:
        if not triton_range or keyword.arg != "loop_unroll_factor" or keyword.arg in seen:
            raise TranslationError("range keyword is not modeled or is duplicated", keyword)
        if not positive_static_unroll_hint(keyword.value, ctx.constexpr_params):
            raise TranslationError("tl.range unroll factor must be positive literals selected only by constexpr equality", keyword)
        seen.add(keyword.arg)
    if len(node.args) == 1:
        return lit_(0), translate_expr(node.args[0], ctx)
    if len(node.args) == 2:
        return translate_expr(node.args[0], ctx), translate_expr(node.args[1], ctx)
    raise TranslationError("range() with step is not supported", node)


def _translate_if(node: ast.If, ctx: TranslationContext) -> list[Stmt]:
    """Translate if/else."""
    cond = translate_expr(node.test, ctx)

    # Check for early return pattern: if cond: return
    if len(node.body) == 1 and isinstance(node.body[0], ast.Return):
        # This is a guard: if (bad_condition): return
        # Wrap remaining code (from orelse) in If with negated condition
        then_body_stmts = _translate_scoped_body(node.orelse, ctx)
        return [If(cond=Not(cond), then_body=then_body_stmts, else_body=[])]

    # Translate siblings from the same incoming metadata environment.  In
    # particular, a block pointer declared in `then` must never classify an
    # otherwise raw pointer load/store in `else`.
    then_body = _translate_scoped_body(node.body, ctx)
    else_body = _translate_scoped_body(node.orelse, ctx)

    return [If(cond=cond, then_body=then_body, else_body=else_body)]


def _translate_expr_stmt(node: ast.Call, ctx: TranslationContext) -> list[Stmt]:
    """Translate expression statements like tl.store(...)."""

    # --- tl.store ---
    if _is_tl_call(node, "store"):
        return _translate_store(node, ctx)

    # A barrier constrains scheduling but has no value or memory effect in the
    # race-free subset checked here.
    if _is_tl_call(node, "debug_barrier"):
        return []

    raise TranslationError(
        f"Unsupported expression call: {ast.unparse(node.func)}", node
    )


def _translate_store(node: ast.Call, ctx: TranslationContext) -> list[Stmt]:
    """Translate tl.store(block_ptr, value, boundary_check=(...))."""
    _require_call_shape(
        node,
        operation="tl.store(block_ptr)",
        positional=2,
        keywords={"boundary_check"},
    )
    ptr_node = node.args[0]
    value_node = node.args[1]

    if isinstance(ptr_node, ast.Name) and ptr_node.id in ctx.block_ptrs:
        info = ctx.block_ptrs[ptr_node.id]
        base_var = ctx.get_var(info.base_tensor)
        value = translate_expr(value_node, ctx)
        boundary_axes = _parse_boundary_axes(node, len(info.shape))

        # Check for pointer-offset base with @params metadata
        if (
            info.base_ast is not None
            and not isinstance(info.base_ast, ast.Name)
            and ctx.kernel_interface is not None
            and info.base_tensor in ctx.kernel_interface.tensor_specs
        ):
            region, mask, squeeze_dims = _build_ptr_offset_region(
                info, ctx, boundary_axes
            )
            # Unsqueeze the stored value for size-1 dims (reverse of load's squeeze)
            for dim in squeeze_dims:
                value = Unsqueeze(value, dim)
            return [masked_store_(base_var, region, value, mask)]

        # Simple case: base is just a tensor name
        region_simple: list[Slice] = []
        for off_ast, bs_ast in zip(info.offsets, info.block_shape):
            off = translate_expr(off_ast, ctx)
            bs = translate_expr(bs_ast, ctx)
            region_simple.append(slice_(off, add_(off, bs)))

        mask_simple: list[Slice] = []
        for axis, shape_ast in enumerate(info.shape):
            if axis in boundary_axes:
                shape_expr = translate_expr(shape_ast, ctx)
                mask_simple.append(slice_(lit_(0), shape_expr))
            else:
                mask_simple.append(region_simple[axis])

        return [masked_store_(base_var, region_simple, value, mask_simple)]

    raise TranslationError("tl.store with non-block-ptr target not supported", node)


# ---------------------------------------------------------------------------
# Top-level: extract kernel function and build Grid + Kernel
# ---------------------------------------------------------------------------


def _find_kernel_func(source: str, kernel_name: str) -> ast.FunctionDef:
    """Find the @triton.jit function definition by name."""
    tree = ast.parse(source)
    matches = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == kernel_name
    ]
    if not matches:
        raise TranslationError(
            f"Top-level kernel function {kernel_name!r} not found in source"
        )
    if len(matches) != 1:
        raise TranslationError(
            f"Expected exactly one top-level kernel function {kernel_name!r}, "
            f"found {len(matches)}"
        )
    func = matches[0]
    has_triton_jit = any(
        isinstance(decorator, ast.Attribute)
        and isinstance(decorator.value, ast.Name)
        and decorator.value.id == "triton"
        and decorator.attr == "jit"
        for decorator in func.decorator_list
    )
    if not has_triton_jit:
        raise TranslationError(
            f"Function {kernel_name!r} is not decorated with @triton.jit",
            func,
        )
    _validate_kernel_signature(func)
    return func


def _validate_kernel_signature(func: ast.FunctionDef) -> None:
    """Reject Python signature/decorator semantics not represented in the IR."""
    arguments = func.args
    if arguments.posonlyargs or arguments.kwonlyargs:
        raise TranslationError(
            "Positional-only and keyword-only kernel parameters are not supported",
            func,
        )
    if arguments.vararg is not None or arguments.kwarg is not None:
        raise TranslationError("Variadic kernel parameters are not supported", func)
    if arguments.defaults or any(default is not None for default in arguments.kw_defaults):
        raise TranslationError("Default kernel arguments are not supported", func)

    for argument in arguments.args:
        if argument.annotation is not None and not _is_tl_attr(
            argument.annotation, "constexpr"
        ):
            raise TranslationError(
                f"Kernel parameter {argument.arg!r} has an unmodeled annotation",
                argument,
            )

    jit_count = 0
    for decorator in func.decorator_list:
        if (
            isinstance(decorator, ast.Attribute)
            and isinstance(decorator.value, ast.Name)
            and decorator.value.id == "triton"
            and decorator.attr == "jit"
        ):
            jit_count += 1
            continue
        raise TranslationError(
            f"Kernel decorator {ast.unparse(decorator)!r} is not modeled",
            decorator,
        )
    if jit_count != 1:
        raise TranslationError("Kernel must have exactly one @triton.jit decorator", func)


def _classify_params(
    func: ast.FunctionDef,
) -> tuple[list[str], set[str]]:
    """Return function parameter names and the ``tl.constexpr`` subset.

    Stride parameters are declared structurally by ``@params``.  Their Python
    names deliberately carry no semantic meaning.
    """
    all_names: list[str] = []
    constexpr: set[str] = set()

    for arg in func.args.args:
        name = arg.arg
        all_names.append(name)

        # Check for tl.constexpr annotation
        if arg.annotation is not None:
            ann = arg.annotation
            if _is_tl_attr(ann, "constexpr"):
                constexpr.add(name)

    return all_names, constexpr


def parse_kernel_interface_for_kernel(
    source: str, kernel_name: str
) -> KernelInterface:
    """Return the checked ``@params``/``@grid`` interface for one kernel."""

    function = _find_kernel_func(source, kernel_name)
    param_names, _ = _classify_params(function)
    annotation_source = kernel_comment_prologue(source, kernel_name)
    assert annotation_source is not None
    annotation = _parse_kernel_interface(annotation_source, set(param_names))
    if annotation is None:
        raise ValueError(f"Kernel {kernel_name!r} is missing @params/@grid")
    return annotation


def _broadcast_shape(t1: Type, t2: Type) -> list[Expr] | None:
    """Compute broadcast shape of two types, or None if not both tensors."""
    if not isinstance(t1, TensorType) or not isinstance(t2, TensorType):
        if isinstance(t1, TensorType):
            return list(t1.dims) if t1.dims else None
        if isinstance(t2, TensorType):
            return list(t2.dims) if t2.dims else None
        return None
    if not t1.dims or not t2.dims:
        return list(t1.dims) or list(t2.dims) or None
    max_ndim = max(len(t1.dims), len(t2.dims))
    l_padded = [lit_(1)] * (max_ndim - len(t1.dims)) + list(t1.dims)
    r_padded = [lit_(1)] * (max_ndim - len(t2.dims)) + list(t2.dims)
    result: list[Expr] = []
    for ld, rd in zip(l_padded, r_padded):
        if isinstance(ld, IntLit) and ld.value == 1:
            result.append(rd)
        elif isinstance(rd, IntLit) and rd.value == 1:
            result.append(ld)
        else:
            # This is a provisional local type; infer_types later rejects
            # incompatible dimensions.
            result.append(ld)
    return result


def _infer_local_type(
    value: Expr,
    ctx: TranslationContext,
    var_types: dict[str, Type] | None = None,
) -> Type:
    """Infer the type of a local variable from its first assignment RHS.

    Recursively propagates types through expression trees to distinguish
    scalar (IntType/FloatType) from tensor (TensorType) results.
    ``var_types`` maps already-inferred variable names to their types so
    that expressions like ``scores_max_prev = scores_max`` resolve correctly.
    """
    if var_types is None:
        var_types = {}
    # --- Constructors with known shapes ---
    if isinstance(value, Zeros):
        return TensorType(FloatType(), list(value.shape))
    if isinstance(value, Full):
        return TensorType(FloatType(), list(value.shape))
    if isinstance(value, Arange):
        # Simplify dim: if start is 0, dim is just stop
        if isinstance(value.start, IntLit) and value.start.value == 0:
            return TensorType(IntType(), [value.stop])
        return TensorType(IntType(), [BinOp("-", value.stop, value.start)])

    def _slice_extent(region_slice: Slice) -> Expr:
        """Recover the exact block extent from the translator's slice form."""
        stop = region_slice.stop
        if isinstance(region_slice.start, IntLit) and region_slice.start.value == 0:
            return stop
        if (
            isinstance(stop, BinOp)
            and stop.op == "+"
            and stop.lhs == region_slice.start
        ):
            return stop.rhs
        return BinOp("-", stop, region_slice.start)

    # --- Loads ---
    if isinstance(value, MaskedLoad):
        # Derive the type from this load's own region.  Looking up an arbitrary
        # BlockPtrInfo by base tensor is unsound when two pointers into the same
        # tensor have different block shapes, and it couples type inference to
        # translation metadata that should be lexically scoped.
        dims = [_slice_extent(region_slice) for region_slice in value.region]
        elem_type: Type = FloatType()
        if ctx.kernel_interface is not None:
            tensor_spec = ctx.kernel_interface.tensor_specs.get(value.base.name)
            if tensor_spec is not None and tensor_spec.is_int32:
                elem_type = IntType()
        return TensorType(elem_type, dims)

    # --- Squeeze: remove one dim from inner type ---
    if isinstance(value, Squeeze):
        inner_type = _infer_local_type(value.value, ctx, var_types)
        if isinstance(inner_type, TensorType) and inner_type.dims:
            new_dims = list(inner_type.dims)
            if 0 <= value.axis < len(new_dims):
                new_dims.pop(value.axis)
            return TensorType(inner_type.elem_type, new_dims)
        return inner_type

    # --- Unsqueeze: add one dim (size 1) to inner type ---
    if isinstance(value, Unsqueeze):
        inner_type = _infer_local_type(value.value, ctx, var_types)
        if isinstance(inner_type, TensorType) and inner_type.dims:
            new_dims = list(inner_type.dims)
            new_dims.insert(value.axis, lit_(1))
            return TensorType(inner_type.elem_type, new_dims)
        # If inner is scalar, result is a 1-element tensor
        if isinstance(inner_type, (IntType, FloatType, BoolType)):
            return TensorType(inner_type, [])
        return TensorType(FloatType(), [])

    # --- BroadcastTo: use the explicit shape ---
    if isinstance(value, BroadcastTo):
        if value.shape:
            inner_type = _infer_local_type(value.value, ctx, var_types)
            elem = inner_type.elem_type if isinstance(inner_type, TensorType) else FloatType()
            return TensorType(elem, list(value.shape))
        return TensorType(FloatType(), [])

    # --- Transpose: swap dims ---
    if isinstance(value, Transpose):
        inner_type = _infer_local_type(value.value, ctx, var_types)
        if isinstance(inner_type, TensorType) and inner_type.dims:
            old_dims = list(inner_type.dims)
            new_dims = [old_dims[p] for p in value.permutation if p < len(old_dims)]
            return TensorType(inner_type.elem_type, new_dims)
        return inner_type

    # --- Reductions: remove one dim ---
    if isinstance(value, (ReduceMax, ReduceSum)):
        inner_type = _infer_local_type(value.value, ctx, var_types)
        if isinstance(inner_type, TensorType) and inner_type.dims:
            new_dims = list(inner_type.dims)
            if 0 <= value.axis < len(new_dims):
                new_dims.pop(value.axis)
            if new_dims:
                return TensorType(inner_type.elem_type, new_dims)
            return TensorType(inner_type.elem_type, [])
        return TensorType(FloatType(), [])

    # --- Where: shape from condition ---
    if isinstance(value, Where):
        cond_type = _infer_local_type(value.cond, ctx, var_types)
        if isinstance(cond_type, TensorType) and cond_type.dims:
            return TensorType(FloatType(), list(cond_type.dims))
        return TensorType(FloatType(), [])

    # --- Pointwise unary operations: propagate shape from operand ---
    if isinstance(value, (Exp2, Sigmoid, Rsqrt, Log2, Cast)):
        return _infer_local_type(value.value, ctx, var_types)
    if isinstance(value, Maximum):
        return _infer_local_type(value.lhs, ctx, var_types)

    # --- Min/Max with tensor args ---
    if isinstance(value, Min):
        # Check if any arg is tensor
        for a in value.args:
            t = _infer_local_type(a, ctx, var_types)
            if isinstance(t, TensorType):
                return t
        return IntType()

    # --- BinOp: check for matmul or tensor operands ---
    if isinstance(value, BinOp):
        if value.op == "@":
            # Matmul [M, K] × [K, N] → [M, N]
            lhs_t = _infer_local_type(value.lhs, ctx, var_types)
            rhs_t = _infer_local_type(value.rhs, ctx, var_types)
            dims: list[Expr] = []
            if isinstance(lhs_t, TensorType) and len(lhs_t.dims) >= 1:
                dims.append(lhs_t.dims[0])
            if isinstance(rhs_t, TensorType) and len(rhs_t.dims) >= 2:
                dims.append(rhs_t.dims[1])
            elif isinstance(rhs_t, TensorType) and len(rhs_t.dims) == 1:
                dims.append(rhs_t.dims[0])
            return TensorType(FloatType(), dims)
        # Comparison ops produce bool tensors
        lhs_type = _infer_local_type(value.lhs, ctx, var_types)
        rhs_type = _infer_local_type(value.rhs, ctx, var_types)
        broadcast_dims = _broadcast_shape(lhs_type, rhs_type)
        if value.op in ("<", ">", "<=", ">=", "==", "!=", "and", "or"):
            if broadcast_dims is not None:
                return TensorType(BoolType(), broadcast_dims)
            if isinstance(lhs_type, TensorType):
                return TensorType(BoolType(), list(lhs_type.dims))
            if isinstance(rhs_type, TensorType):
                return TensorType(BoolType(), list(rhs_type.dims))
            return BoolType()
        # Arithmetic: if either operand is tensor, result is tensor
        if broadcast_dims is not None:
            elem = lhs_type.elem_type if isinstance(lhs_type, TensorType) else (
                rhs_type.elem_type if isinstance(rhs_type, TensorType) else FloatType()
            )
            return TensorType(elem, broadcast_dims)
        if isinstance(lhs_type, TensorType):
            return TensorType(lhs_type.elem_type, list(lhs_type.dims))
        if isinstance(rhs_type, TensorType):
            return TensorType(rhs_type.elem_type, list(rhs_type.dims))
        return IntType()

    # --- Scalars ---
    if isinstance(value, IntLit):
        return IntType()
    if isinstance(value, FloatLit):
        return FloatType()

    # --- Min/Max builtins (scalar) ---
    if isinstance(value, (Min, Max)):
        return IntType()

    # --- Var: look up previously-inferred type, default scalar ---
    if isinstance(value, Var):
        if value.name in var_types:
            return var_types[value.name]
        return IntType()

    # Fallback: scalar
    return IntType()


def _remember_local_type(name: str, value: Expr, ctx: TranslationContext) -> None:
    """Record the best source-order type available for later reshape checks."""
    ctx.local_types[name] = _infer_local_type(value, ctx, ctx.local_types)


def _emit_var_decls(
    stmts: list[Stmt],
    ctx: TranslationContext,
    param_names: set[str],
    grid_var_names: set[str],
    constexpr_params: set[str],
) -> None:
    """Scan translated body and emit VarDecls for local variables.

    Walks the statement tree, finds Assign targets that are Var nodes
    not already declared as params/grid/constexpr, and calls
    ctx.declare_local() with an inferred type.
    """
    seen: set[str] = set()
    var_types: dict[str, Type] = {}
    skip = param_names | grid_var_names | constexpr_params

    def _scan(stmts: list[Stmt]) -> None:
        for s in stmts:
            if isinstance(s, Assign) and isinstance(s.target, Var):
                name = s.target.name
                if name not in skip and name not in seen:
                    seen.add(name)
                    typ = _infer_local_type(s.value, ctx, var_types)
                    var_types[name] = typ
                    ctx.declare_local(name, typ)
            if isinstance(s, For):
                _scan(s.body)
            if isinstance(s, If):
                _scan(s.then_body)
                _scan(s.else_body)

    _scan(stmts)


def _broadcast_unsqueezes(
    stmts: list[Stmt], var_types: dict[str, Type]
) -> list[Stmt]:
    """Insert BroadcastTo around Unsqueeze nodes in binary operations.

    Triton performs implicit broadcasting but the verification IR requires
    explicit BroadcastTo.  For each Assign/AugAssign whose target has a
    known TensorType, wrap any Unsqueeze child that participates in a
    BinOp with a BroadcastTo to the target shape.
    """

    def _target_shape(name: str) -> list[Expr] | None:
        t = var_types.get(name)
        if isinstance(t, TensorType) and t.dims:
            return list(t.dims)
        return None

    def _wrap_unsqueeze(expr: Expr, shape: list[Expr]) -> Expr:
        """Recursively wrap Unsqueeze operands of BinOp with BroadcastTo."""
        if isinstance(expr, BinOp):
            new_lhs = _wrap_unsqueeze(expr.lhs, shape)
            new_rhs = _wrap_unsqueeze(expr.rhs, shape)
            return BinOp(expr.op, new_lhs, new_rhs)
        if isinstance(expr, (Exp2, Sigmoid, Rsqrt, Log2)):
            return type(expr)(_wrap_unsqueeze(expr.value, shape))
        if isinstance(expr, Cast):
            return Cast(
                _wrap_unsqueeze(expr.value, shape), expr.kind, expr.target
            )
        if isinstance(expr, Maximum):
            return Maximum(_wrap_unsqueeze(expr.lhs, shape),
                           _wrap_unsqueeze(expr.rhs, shape))
        if isinstance(expr, Where):
            return Where(_wrap_unsqueeze(expr.cond, shape),
                         _wrap_unsqueeze(expr.on_true, shape),
                         _wrap_unsqueeze(expr.on_false, shape))
        if isinstance(expr, ReduceMax):
            return ReduceMax(_wrap_unsqueeze(expr.value, shape), expr.axis)
        if isinstance(expr, ReduceSum):
            return ReduceSum(_wrap_unsqueeze(expr.value, shape), expr.axis)
        if isinstance(expr, Unsqueeze):
            return BroadcastTo(expr, shape)
        return expr

    result: list[Stmt] = []
    for s in stmts:
        if isinstance(s, Assign) and isinstance(s.target, Var):
            shape = _target_shape(s.target.name)
            if shape and s.op is not None:
                # AugAssign: wrap unsqueezes in the value to match target shape
                new_value = _wrap_unsqueeze(s.value, shape)
                result.append(Assign(s.target, s.op, new_value))
                continue
            if shape and s.op is None:
                # Regular Assign: recursively wrap Unsqueeze in BinOps.
                # Skip if value is itself an Unsqueeze (e.g. o_block = unsqueeze(...))
                if not isinstance(s.value, Unsqueeze):
                    new_value = _wrap_unsqueeze(s.value, shape)
                    if new_value is not s.value:
                        result.append(Assign(s.target, s.op, new_value))
                        continue
        if isinstance(s, For):
            new_body = _broadcast_unsqueezes(s.body, var_types)
            result.append(For(s.var, s.iters, new_body))
            continue
        if isinstance(s, If):
            new_then = _broadcast_unsqueezes(s.then_body, var_types)
            new_else = _broadcast_unsqueezes(s.else_body, var_types)
            result.append(If(s.cond, new_then, new_else))
            continue
        result.append(s)
    return result


def _scalar_assigns_to_lets(
    stmts: list[Stmt], scalar_names: set[str]
) -> list[Stmt]:
    """Convert Assign statements targeting scalar variables to Let bindings.

    The verification pipeline's region analysis (regions.py) only handles
    tensor Assign targets.  Scalar assignments like ``si = i * BLOCK_N``
    must be expressed as ``Let(si, i * BLOCK_N)`` instead.
    """
    result: list[Stmt] = []
    for s in stmts:
        if isinstance(s, Assign) and isinstance(s.target, Var):
            if s.target.name in scalar_names and s.op is None:
                result.append(Let(var=s.target, value=s.value))
                continue
        if isinstance(s, For):
            new_body = _scalar_assigns_to_lets(s.body, scalar_names)
            result.append(For(s.var, s.iters, new_body))
            continue
        if isinstance(s, If):
            new_then = _scalar_assigns_to_lets(s.then_body, scalar_names)
            new_else = _scalar_assigns_to_lets(s.else_body, scalar_names)
            result.append(If(s.cond, new_then, new_else))
            continue
        result.append(s)
    return result


def _extract_grid_iters(
    func_body: list[ast.stmt], ctx: TranslationContext
) -> tuple[dict[int, str], list[ast.stmt]]:
    """Extract tl.program_id assignments from the beginning of the function body.

    Returns (axis_to_varname, remaining_stmts).
    """
    grid_iters: dict[int, str] = {}
    remaining: list[ast.stmt] = []
    deferred_stmts: list[ast.stmt] = []  # synthetic stmts to prepend to remaining
    done_with_grid = False

    for stmt in func_body:
        if done_with_grid:
            remaining.append(stmt)
            continue

        if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1:
            target = stmt.targets[0]

            # Direct: li = tl.program_id(axis=2)
            if isinstance(target, ast.Name) and _is_tl_call(stmt.value, "program_id"):
                axis = _program_id_axis(stmt.value)
                if axis in grid_iters:
                    raise TranslationError(
                        f"Multiple tl.program_id assignments for axis {axis}", stmt
                    )
                if target.id in ctx.param_names or target.id in grid_iters.values():
                    raise TranslationError(
                        f"Grid iterator {target.id!r} shadows an existing name", stmt
                    )
                grid_iters[axis] = target.id
                ctx.get_var(target.id)
                continue

            # With multiplication: li = tl.program_id(axis=2) * BLOCK_M
            # Decompose into: grid iter _pid_N for the raw program_id,
            # plus a synthetic assignment li = _pid_N * BLOCK_M in the body.
            if isinstance(target, ast.Name) and isinstance(stmt.value, ast.BinOp):
                pid_node = None
                multiplier_node = None
                if _is_tl_call(stmt.value.left, "program_id"):
                    pid_node = stmt.value.left
                    multiplier_node = stmt.value.right
                elif _is_tl_call(stmt.value.right, "program_id"):
                    pid_node = stmt.value.right
                    multiplier_node = stmt.value.left

                if pid_node is not None and multiplier_node is not None:
                    axis = _program_id_axis(pid_node)
                    if axis in grid_iters:
                        raise TranslationError(
                            f"Multiple tl.program_id assignments for axis {axis}", stmt
                        )
                    if target.id in ctx.param_names or target.id in grid_iters.values():
                        raise TranslationError(
                            f"Grid-derived local {target.id!r} shadows an existing name",
                            stmt,
                        )
                    # Create a synthetic grid iter for the raw program_id
                    pid_var_name = f"_pid_{axis}"
                    if pid_var_name in ctx.param_names or pid_var_name in grid_iters.values():
                        raise TranslationError(
                            f"Synthetic grid iterator {pid_var_name!r} shadows an existing name",
                            stmt,
                        )
                    grid_iters[axis] = pid_var_name
                    ctx.get_var(pid_var_name)
                    # Emit: li = _pid_N * BLOCK_M as a synthetic AST assignment
                    synth_assign = ast.Assign(
                        targets=[ast.Name(id=target.id, ctx=ast.Store())],
                        value=ast.BinOp(
                            left=ast.Name(id=pid_var_name, ctx=ast.Load()),
                            op=stmt.value.op,
                            right=multiplier_node,
                        ),
                        lineno=stmt.lineno,
                        col_offset=stmt.col_offset,
                    )
                    deferred_stmts.append(synth_assign)
                    continue

        done_with_grid = True
        remaining.append(stmt)

    return grid_iters, deferred_stmts + remaining


def _handle_early_return(stmts: list[ast.stmt], ctx: TranslationContext) -> list[ast.stmt]:
    """Handle patterns like `if li >= q_len: return` by wrapping remaining in if-else."""
    result: list[ast.stmt] = []
    i = 0
    while i < len(stmts):
        stmt = stmts[i]
        if (
            isinstance(stmt, ast.If)
            and len(stmt.body) == 1
            and isinstance(stmt.body[0], ast.Return)
            and not stmt.orelse
        ):
            # Guard pattern: if cond: return
            # Wrap everything after this in: if not cond: <rest>
            remaining = stmts[i + 1:]
            guard_if = ast.If(
                test=stmt.test,
                body=[ast.Return(value=None)],
                orelse=remaining,
                lineno=stmt.lineno,
                col_offset=stmt.col_offset,
            )
            # We don't emit the guard directly; instead we note the condition
            # and wrap the rest
            result.append(guard_if)
            return result
        else:
            result.append(stmt)
        i += 1
    return result


def _validate_return_usage(func: ast.FunctionDef) -> None:
    """Accept only the single top-level guard-return shape we translate.

    ``_handle_early_return`` rewrites ``if cond: return`` plus the remaining
    top-level statements into one guarded region.  A nested return or a second
    return would otherwise be dropped by ``translate_stmt`` and change the
    source program's control flow.
    """
    allowed: ast.Return | None = None
    for stmt in func.body:
        if (
            isinstance(stmt, ast.If)
            and len(stmt.body) == 1
            and isinstance(stmt.body[0], ast.Return)
            and not stmt.orelse
        ):
            if allowed is not None:
                raise TranslationError(
                    "Only one top-level early-return guard is supported", stmt
                )
            allowed = stmt.body[0]

    for node in ast.walk(func):
        if isinstance(node, ast.Return) and node is not allowed:
            raise TranslationError(
                "Return is supported only as a single top-level "
                "`if condition: return` guard",
                node,
            )


def _validate_definite_assignments(
    statements: Sequence[ast.stmt],
    initially_defined: set[str],
) -> None:
    """Reject reads of source locals that are not defined on every path.

    The IR declares tensor locals at grid scope.  Without this source check, a
    name first assigned in a zero-trip loop or only one branch would appear
    unconditionally available after translation.
    """

    local_names = set(initially_defined)
    for node in ast.walk(ast.Module(body=list(statements), type_ignores=[])):
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            local_names.update(
                target.id for target in targets if isinstance(target, ast.Name)
            )
        elif isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Name):
            local_names.add(node.target.id)
        elif isinstance(node, ast.For) and isinstance(node.target, ast.Name):
            local_names.add(node.target.id)

    def check_reads(node: ast.AST | None, defined: set[str]) -> None:
        if node is None:
            return
        for child in ast.walk(node):
            if (
                isinstance(child, ast.Name)
                and isinstance(child.ctx, ast.Load)
                and child.id in local_names
                and child.id not in defined
            ):
                raise TranslationError(
                    f"Local value {child.id!r} may be undefined on this path",
                    child,
                )

    def analyze_block(
        body: Sequence[ast.stmt], defined_at_entry: set[str]
    ) -> tuple[set[str], bool]:
        defined = set(defined_at_entry)
        falls_through = True
        for statement in body:
            if not falls_through:
                break
            match statement:
                case ast.Assign(targets=targets, value=value):
                    check_reads(value, defined)
                    for target in targets:
                        if isinstance(target, ast.Name):
                            defined.add(target.id)
                case ast.AugAssign(target=target, value=value):
                    if isinstance(target, ast.Name) and target.id not in defined:
                        raise TranslationError(
                            f"Local value {target.id!r} may be undefined before update",
                            target,
                        )
                    check_reads(value, defined)
                case ast.For(target=target, iter=iterator, body=loop_body):
                    check_reads(iterator, defined)
                    loop_defined = set(defined)
                    if isinstance(target, ast.Name):
                        loop_defined.add(target.id)
                    analyze_block(loop_body, loop_defined)
                    # A source range may be empty, so new loop-body bindings
                    # and the loop target are not available after the loop.
                case ast.If(test=test, body=then_body, orelse=else_body):
                    check_reads(test, defined)
                    then_defined, then_falls = analyze_block(then_body, defined)
                    else_defined, else_falls = analyze_block(else_body, defined)
                    continuing = [
                        branch_defined
                        for branch_defined, branch_falls in (
                            (then_defined, then_falls),
                            (else_defined, else_falls),
                        )
                        if branch_falls
                    ]
                    falls_through = bool(continuing)
                    if continuing:
                        defined = set.intersection(*continuing)
                case ast.Expr(value=value):
                    check_reads(value, defined)
                case ast.Return(value=value):
                    check_reads(value, defined)
                    falls_through = False
                case _:
                    # Unsupported statements are rejected by translate_stmt;
                    # still audit their reads so this pass never hides one.
                    check_reads(statement, defined)
        return defined, falls_through

    analyze_block(statements, initially_defined)


def _infer_tensor_params(
    param_names: list[str],
    constexpr_params: set[str],
    stride_params: set[str],
    func_body: list[ast.stmt],
) -> tuple[list[str], list[str]]:
    """Separate tensor params from scalar params.

    Tensor params are those that appear as base in make_block_ptr or pointer loads.
    The rest (minus strides and constexprs) are scalar dimension params.
    """
    # Walk the AST to find all names used as base in make_block_ptr
    tensor_names: set[str] = set()
    for node in ast.walk(ast.Module(body=func_body, type_ignores=[])):
        if _is_tl_call(node, "make_block_ptr") and isinstance(node, ast.Call):
            base_arg = node.args[0] if node.args else _get_keyword(node, "base")
            if base_arg:
                tensor_names.add(_extract_base_tensor_name(base_arg, None))
        # Also check tl.load(ptr + offset) patterns
        if _is_tl_call(node, "load") and isinstance(node, ast.Call):
            arg0 = node.args[0]
            if isinstance(arg0, ast.BinOp):
                name = _extract_base_tensor_name(arg0, None)
                # Only add if it's a known parameter (not a block_ptr var)
                if name in param_names:
                    tensor_names.add(name)

    tensor_params = [n for n in param_names if n in tensor_names]
    scalar_params = [
        n
        for n in param_names
        if n not in tensor_names and n not in stride_params and n not in constexpr_params
    ]
    return tensor_params, scalar_params


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def translate_kernel_source(
    source: str,
    kernel_name: str,
    *,
    specialize: dict[str, bool] | None = None,
) -> Kernel:
    """Translate a Triton kernel from Python source into the verification IR.

    Args:
        source: Python source code containing the @triton.jit kernel.
        kernel_name: Name of the kernel function to translate.
        specialize: Optional dict mapping boolean constexpr parameter names to
                    translation-time branch values.

    Returns:
        An ir.Kernel instance.
    """
    # This public entry point is the proof front end.  Validation must not be a
    # caller convention: every analysis pass should fail before translation if
    # the source contains a memory form or Triton operation outside the audited
    # subset.
    from .validate_subset import validate_triton_subset

    violations = validate_triton_subset(source, kernel_name)
    if violations:
        details = "; ".join(str(violation) for violation in violations)
        raise TranslationError(
            f"Kernel {kernel_name!r} is outside the verifiable Triton subset: "
            f"{details}"
        )

    func = _find_kernel_func(source, kernel_name)
    _validate_return_usage(func)
    ctx = TranslationContext()

    # 1. Classify parameters
    param_names, constexpr_params = _classify_params(func)
    ctx.param_names = param_names
    ctx.constexpr_params = constexpr_params

    if specialize:
        unknown = set(specialize) - constexpr_params
        if unknown:
            raise TranslationError(
                "Only tl.constexpr parameters may be translation-specialized: "
                f"{sorted(unknown)}",
                func,
            )
        non_bool = {
            name: value
            for name, value in specialize.items()
            if not isinstance(value, bool)
        }
        if non_bool:
            raise TranslationError(
                "Translation-time branch specialization requires booleans: "
                f"{non_bool}",
                func,
            )

    # 2. Parse required @params/@grid annotations from this kernel's contiguous
    # comment prologue, never from an earlier function in the same file.
    annotation_source = kernel_comment_prologue(source, kernel_name)
    assert annotation_source is not None  # func was found above
    annot = _parse_kernel_interface(annotation_source, set(param_names))
    if annot is None:
        raise ValueError(
            f"Kernel '{kernel_name}' is missing @params/@grid annotations."
        )
    ctx.kernel_interface = annot
    stride_params = {
        name for spec in annot.tensor_specs.values() for name in spec.stride_map
    }
    ctx.stride_params = stride_params
    unknown_tensors = set(annot.tensor_specs) - set(param_names)
    if unknown_tensors:
        raise TranslationError(
            "@params declares tensors absent from the kernel signature: "
            f"{sorted(unknown_tensors)}",
            func,
        )

    # 3. Identify tensor vs scalar params
    tensor_param_names, scalar_param_names = _infer_tensor_params(
        param_names, constexpr_params, stride_params, func.body
    )
    # Add any annotation-specified tensors not detected by inference
    for tname in annot.tensor_specs:
        if tname not in tensor_param_names and tname in param_names:
            tensor_param_names.append(tname)
            if tname in scalar_param_names:
                scalar_param_names.remove(tname)

    # 4. Build tensor shapes from annotation
    tensor_shapes: dict[str, list[str]] = {}
    for tname, tspec in annot.tensor_specs.items():
        dim_names: list[str] = []
        for d in tspec.dims:
            if isinstance(d, Var):
                dim_names.append(d.name)
            else:
                # Complex expression — use a generated name
                dim_names.append(f"_annot_dim_{tname}_{len(dim_names)}")
        tensor_shapes[tname] = dim_names

    ctx.tensor_shapes = tensor_shapes

    # 5. Extract grid iterators
    grid_iters, remaining_body = _extract_grid_iters(func.body, ctx)
    ctx.grid_iters = grid_iters
    expected_axes = set(range(len(annot.grid_ranges)))
    actual_axes = set(grid_iters)
    if actual_axes != expected_axes:
        raise TranslationError(
            "@grid axes must exactly match tl.program_id axes: "
            f"declared={sorted(expected_axes)}, actual={sorted(actual_axes)}",
            func,
        )

    # 6. Specialize constexpr params if requested
    if specialize:
        remaining_body = _specialize_body(remaining_body, specialize)

    # 7. Check source control-flow before hoisting IR local declarations.
    _validate_definite_assignments(
        remaining_body,
        set(param_names) | set(grid_iters.values()),
    )

    # 8. Handle early returns
    remaining_body = _handle_early_return(remaining_body, ctx)

    # 9. Translate body statements
    body_stmts: list[Stmt] = []
    for stmt in remaining_body:
        body_stmts.extend(translate_stmt(stmt, ctx))

    # 9b. Emit VarDecls for local variables
    all_param_names = set(tensor_param_names) | set(scalar_param_names)
    all_param_names |= {name for name, _ in annot.scalar_params}
    _emit_var_decls(body_stmts, ctx, all_param_names,
                    set(grid_iters.values()), constexpr_params)

    # 9c. Insert explicit BroadcastTo around Unsqueeze nodes in BinOps
    # (Triton does implicit broadcasting; the verification IR needs explicit)
    var_types_map: dict[str, Type] = {d.var.name: d.type for d in ctx.decls}
    body_stmts = _broadcast_unsqueezes(body_stmts, var_types_map)

    # 9d. Convert scalar Assigns to Let bindings.
    # The verification pipeline's region analysis expects only tensor Assigns;
    # scalar assignments (IntType, FloatType, BoolType) should be Lets.
    scalar_var_names: set[str] = {
        d.var.name for d in ctx.decls
        if isinstance(d.type, (IntType, FloatType, BoolType))
    }
    body_stmts = _scalar_assigns_to_lets(body_stmts, scalar_var_names)
    # Remove VarDecls for variables that became Lets
    ctx.decls = [d for d in ctx.decls if d.var.name not in scalar_var_names]

    # 10. Build IR Params
    ir_params: list[Param] = []

    # Tensor params
    for tname in tensor_param_names:
        tspec = annot.tensor_specs[tname]
        # Pointer element types are not present in the Python AST.  Require
        # the annotation to state integer metadata explicitly; the absence of
        # stride parameters is not a sound proxy for element type.
        elem_type: Type = IntType() if tspec.is_int32 else FloatType()
        ir_params.append(Param(tname, TensorType(elem_type, list(tspec.dims))))

    # Collect all dimension variable names — these are implicitly in scope
    # as param_size and must NOT also appear as params
    dim_vars: set[str] = set()
    for tspec in annot.tensor_specs.values():
        for d in tspec.dims:
            _collect_var_names(d, dim_vars)

    # Extra scalar types from @params
    scalar_param_types = {pname: ptype for pname, ptype in annot.scalar_params}

    # Scalar params (excluding dimension vars that are already param_size)
    for sname in scalar_param_names:
        if sname not in dim_vars:
            if sname in scalar_param_types:
                ir_params.append(Param(sname, scalar_param_types[sname]))
            else:
                ir_params.append(Param(sname, IntType()))

    # Extra params from annotation not already in the kernel signature
    existing_param_names = {p.name for p in ir_params} | dim_vars
    for pname, ptype in annot.scalar_params:
        if pname not in existing_param_names:
            ir_params.append(Param(pname, ptype))

    # Constexpr params (BLOCK_M, BLOCK_N, etc.) — always IntType
    # Skip any that are also tensor dimension vars (already param_size)
    for cname in sorted(constexpr_params):
        if specialize and cname in specialize:
            continue  # specialized away
        if cname in dim_vars:
            continue  # already a tensor dimension (param_size)
        ir_params.append(Param(cname, IntType()))

    # 11. Build Grid from annotation grid ranges
    grid_iter_list: list[GridIter] = []
    for axis in sorted(grid_iters.keys()):
        var_name = grid_iters[axis]
        var = ctx.get_var(var_name)
        range_stop = annot.grid_ranges[axis]
        grid_iter_list.append(
            GridIter(var, range_(lit_(0), range_stop))
        )

    grid = Grid(
        iters=grid_iter_list,
        decls=ctx.decls,
        body=body_stmts,
    )

    return Kernel(name=kernel_name, params=ir_params, grid=grid)


def _specialize_body(
    body: list[ast.stmt], specialize: dict[str, bool]
) -> list[ast.stmt]:
    """Inline constexpr specializations by evaluating if-branches."""
    result: list[ast.stmt] = []
    for stmt in body:
        if isinstance(stmt, ast.If):
            test = stmt.test
            # Check if test is a simple Name that we're specializing
            if isinstance(test, ast.Name) and test.id in specialize:
                val = specialize[test.id]
                if val:
                    result.extend(_specialize_body(stmt.body, specialize))
                else:
                    result.extend(_specialize_body(stmt.orelse, specialize))
                continue
            # Recurse into both branches of non-specialized ifs
            new_if = ast.If(
                test=stmt.test,
                body=_specialize_body(stmt.body, specialize),
                orelse=_specialize_body(stmt.orelse, specialize),
                lineno=stmt.lineno,
                col_offset=stmt.col_offset,
            )
            result.append(new_if)
            continue
        if isinstance(stmt, ast.For):
            new_for = ast.For(
                target=stmt.target,
                iter=stmt.iter,
                body=_specialize_body(stmt.body, specialize),
                orelse=stmt.orelse,
                lineno=stmt.lineno,
                col_offset=stmt.col_offset,
            )
            result.append(new_for)
            continue
        result.append(stmt)
    return result


# ---------------------------------------------------------------------------
# Convenience: translate from a file
# ---------------------------------------------------------------------------


def translate_kernel_file(
    filepath: str,
    kernel_name: str,
    **kwargs,
) -> Kernel:
    """Read a .py file and translate the named kernel."""
    with open(filepath) as f:
        source = f.read()
    return translate_kernel_source(source, kernel_name, **kwargs)
