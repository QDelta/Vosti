"""GPU falsifiers for assumptions exported by ``ir.backend_requirements``.

These probes deliberately test only batch-invariance-relevant locality,
determinism, mask identities, and logical memory mapping.  They are empirical
deployment evidence, not a proof of Triton or of kernel numerical correctness.
"""

from __future__ import annotations

from collections import defaultdict
from contextvars import ContextVar
import json
import math
from typing import Callable

import torch
import triton
import triton.language as tl

from .probe_contract import FLOAT32_ONLY_OPERATIONS, validate_requirement_probe_coverage


def _equal_bits(left: torch.Tensor, right: torch.Tensor) -> bool:
    """Exact tensor equality for determinism/locality, including signed zero."""
    return (
        left.shape == right.shape
        and left.dtype == right.dtype
        and torch.equal(
            left.contiguous().reshape(-1).view(torch.uint8),
            right.contiguous().reshape(-1).view(torch.uint8),
        )
    )


@triton.jit
def _binary_kernel(x, y, out, n: tl.constexpr, block: tl.constexpr, op: tl.constexpr):
    offsets = tl.arange(0, block)
    mask = offsets < n
    a = tl.load(x + offsets, mask=mask, other=0)
    b = tl.load(y + offsets, mask=mask, other=1)
    if op == 0:
        value = a + b
    elif op == 1:
        value = a - b
    elif op == 2:
        value = a * b
    elif op == 3:
        value = a / b
    elif op == 4:
        value = a // b
    elif op == 5:
        value = a % b
    elif op == 6:
        value = a < b
    elif op == 7:
        value = a <= b
    elif op == 8:
        value = a > b
    elif op == 9:
        value = a >= b
    elif op == 10:
        value = a == b
    elif op == 11:
        value = a != b
    elif op == 12:
        value = a & b
    elif op == 13:
        value = a | b
    elif op == 14:
        value = (a + b - 1) // b
    elif op == 15:
        value = tl.maximum(a, b)
    else:
        value = tl.minimum(a, b)
    tl.store(out + offsets, value, mask=mask)


@triton.jit
def _unary_kernel(x, out, n: tl.constexpr, block: tl.constexpr, op: tl.constexpr):
    offsets = tl.arange(0, block)
    mask = offsets < n
    value = tl.load(x + offsets, mask=mask, other=1)
    if op == 0:
        value = tl.exp2(value)
    elif op == 1:
        value = tl.sigmoid(value)
    elif op == 2:
        value = tl.rsqrt(value)
    elif op == 3:
        value = tl.log2(value)
    else:
        value = ~value
    tl.store(out + offsets, value, mask=mask)


@triton.jit
def _where_kernel(cond, x, y, out, n: tl.constexpr, block: tl.constexpr):
    offsets = tl.arange(0, block)
    mask = offsets < n
    c = tl.load(cond + offsets, mask=mask, other=0)
    a = tl.load(x + offsets, mask=mask, other=0)
    b = tl.load(y + offsets, mask=mask, other=0)
    tl.store(out + offsets, tl.where(c != 0, a, b), mask=mask)


@triton.jit
def _cast_kernel(x, out, n: tl.constexpr, block: tl.constexpr):
    offsets = tl.arange(0, block)
    mask = offsets < n
    value = tl.load(x + offsets, mask=mask, other=0)
    tl.store(out + offsets, value.to(out.dtype.element_ty), mask=mask)


@triton.jit
def _reduce_kernel(
    x,
    out,
    rows: tl.constexpr,
    cols: tl.constexpr,
    axis: tl.constexpr,
    op: tl.constexpr,
):
    row_offsets = tl.arange(0, rows)[:, None]
    col_offsets = tl.arange(0, cols)[None, :]
    value = tl.load(x + row_offsets * cols + col_offsets)
    if op == 0:
        result = tl.sum(value, axis=axis)
    elif op == 1:
        result = tl.max(value, axis=axis)
    else:
        result = tl.max(value > 0, axis=axis)
    size: tl.constexpr = cols if axis == 0 else rows
    tl.store(out + tl.arange(0, size), result)


@triton.jit
def _dot_kernel(
    a,
    b,
    out,
    m: tl.constexpr,
    n: tl.constexpr,
    k: tl.constexpr,
):
    mi = tl.arange(0, m)[:, None]
    ki = tl.arange(0, k)[None, :]
    kj = tl.arange(0, k)[:, None]
    nj = tl.arange(0, n)[None, :]
    av = tl.load(a + mi * k + ki)
    bv = tl.load(b + kj * n + nj)
    value = tl.dot(av, bv)
    tl.store(out + mi * n + nj, value)


@triton.jit
def _constructor_kernel(
    out,
    n: tl.constexpr,
    block: tl.constexpr,
    op: tl.constexpr,
):
    offsets = tl.arange(0, block)
    mask = offsets < n
    if op == 0:
        value = tl.zeros((block,), tl.float32)
    elif op == 1:
        value = tl.full((block,), 3.0, tl.float32)
    else:
        value = offsets
    tl.store(out + offsets, value, mask=mask)


@triton.jit
def _reshape_kernel(
    x,
    out,
    in0: tl.constexpr,
    in1: tl.constexpr,
    out0: tl.constexpr,
    out1: tl.constexpr,
    op: tl.constexpr,
):
    flat = tl.load(x + tl.arange(0, in0 * in1))
    value = tl.reshape(flat, (in0, in1))
    if op == 0:
        value = tl.trans(value)
    elif op == 1:
        value = tl.broadcast_to(value, (out0, out1))
    flat_out = tl.reshape(value, (out0 * out1,))
    tl.store(out + tl.arange(0, out0 * out1), flat_out)


@triton.jit
def _reshape_1_2(x, out, a: tl.constexpr, b: tl.constexpr):
    value = tl.load(x + tl.arange(0, a * b))
    value = tl.reshape(value, (a, b))
    tl.store(out + tl.arange(0, a * b), tl.reshape(value, (a * b,)))


@triton.jit
def _reshape_2_1(x, out, a: tl.constexpr, b: tl.constexpr):
    value = tl.load(x + tl.arange(0, a * b))
    value = tl.reshape(value, (a, b))
    value = tl.reshape(value, (a * b,))
    tl.store(out + tl.arange(0, a * b), value)


@triton.jit
def _reshape_2_3(
    x,
    out,
    in0: tl.constexpr,
    in1: tl.constexpr,
    out0: tl.constexpr,
    out1: tl.constexpr,
    out2: tl.constexpr,
):
    size: tl.constexpr = in0 * in1
    value = tl.load(x + tl.arange(0, size))
    value = tl.reshape(value, (in0, in1))
    value = tl.reshape(value, (out0, out1, out2))
    tl.store(out + tl.arange(0, size), tl.reshape(value, (size,)))


@triton.jit
def _reshape_3_2(
    x,
    out,
    in0: tl.constexpr,
    in1: tl.constexpr,
    in2: tl.constexpr,
    out0: tl.constexpr,
    out1: tl.constexpr,
):
    size: tl.constexpr = in0 * in1 * in2
    value = tl.load(x + tl.arange(0, size))
    value = tl.reshape(value, (in0, in1, in2))
    value = tl.reshape(value, (out0, out1))
    tl.store(out + tl.arange(0, size), tl.reshape(value, (size,)))


@triton.jit
def _reshape_3_4(
    x,
    out,
    in0: tl.constexpr,
    in1: tl.constexpr,
    in2: tl.constexpr,
    out0: tl.constexpr,
    out1: tl.constexpr,
    out2: tl.constexpr,
    out3: tl.constexpr,
):
    size: tl.constexpr = in0 * in1 * in2
    value = tl.load(x + tl.arange(0, size))
    value = tl.reshape(value, (in0, in1, in2))
    value = tl.reshape(value, (out0, out1, out2, out3))
    tl.store(
        out + tl.arange(0, size),
        tl.reshape(value, (size,)),
    )


@triton.jit
def _reshape_4_3(
    x,
    out,
    in0: tl.constexpr,
    in1: tl.constexpr,
    in2: tl.constexpr,
    in3: tl.constexpr,
    out0: tl.constexpr,
    out1: tl.constexpr,
    out2: tl.constexpr,
):
    size: tl.constexpr = in0 * in1 * in2 * in3
    value = tl.load(x + tl.arange(0, size))
    value = tl.reshape(value, (in0, in1, in2, in3))
    value = tl.reshape(value, (out0, out1, out2))
    tl.store(
        out + tl.arange(0, size),
        tl.reshape(value, (size,)),
    )


@triton.jit
def _block_load_probe(
    x,
    out,
    logical_rows: tl.constexpr,
    logical_cols: tl.constexpr,
    block_rows: tl.constexpr,
    block_cols: tl.constexpr,
):
    pointer = tl.make_block_ptr(
        x,
        shape=(logical_rows, logical_cols),
        strides=(logical_cols, 1),
        offsets=(0, 0),
        block_shape=(block_rows, block_cols),
        order=(1, 0),
    )
    value = tl.load(pointer, boundary_check=(0, 1), padding_option="zero")
    flat = tl.reshape(value, (block_rows * block_cols,))
    tl.store(out + tl.arange(0, block_rows * block_cols), flat)


@triton.jit
def _block_store_probe(
    x,
    out,
    logical_rows: tl.constexpr,
    logical_cols: tl.constexpr,
    block_rows: tl.constexpr,
    block_cols: tl.constexpr,
):
    value = tl.load(x + tl.arange(0, block_rows * block_cols))
    value = tl.reshape(value, (block_rows, block_cols))
    pointer = tl.make_block_ptr(
        out,
        shape=(logical_rows, logical_cols),
        strides=(logical_cols, 1),
        offsets=(0, 0),
        block_shape=(block_rows, block_cols),
        order=(1, 0),
    )
    tl.store(pointer, value, boundary_check=(0, 1))


@triton.jit
def _block_load_probe3(
    x,
    out,
    logical0: tl.constexpr,
    logical1: tl.constexpr,
    logical2: tl.constexpr,
    block0: tl.constexpr,
    block1: tl.constexpr,
    block2: tl.constexpr,
):
    pointer = tl.make_block_ptr(
        x,
        shape=(logical0, logical1, logical2),
        strides=(logical1 * logical2, logical2, 1),
        offsets=(0, 0, 0),
        block_shape=(block0, block1, block2),
        order=(2, 1, 0),
    )
    value = tl.load(pointer, boundary_check=(0, 1, 2), padding_option="zero")
    flat = tl.reshape(value, (block0 * block1 * block2,))
    tl.store(out + tl.arange(0, block0 * block1 * block2), flat)


@triton.jit
def _block_load_probe4(
    x,
    out,
    logical0: tl.constexpr,
    logical1: tl.constexpr,
    logical2: tl.constexpr,
    logical3: tl.constexpr,
    block0: tl.constexpr,
    block1: tl.constexpr,
    block2: tl.constexpr,
    block3: tl.constexpr,
):
    pointer = tl.make_block_ptr(
        x,
        shape=(logical0, logical1, logical2, logical3),
        strides=(logical1 * logical2 * logical3, logical2 * logical3, logical3, 1),
        offsets=(0, 0, 0, 0),
        block_shape=(block0, block1, block2, block3),
        order=(3, 2, 1, 0),
    )
    value = tl.load(pointer, boundary_check=(0, 1, 2, 3), padding_option="zero")
    flat = tl.reshape(value, (block0 * block1 * block2 * block3,))
    tl.store(out + tl.arange(0, block0 * block1 * block2 * block3), flat)


@triton.jit
def _block_store_probe3(
    x,
    out,
    logical0: tl.constexpr,
    logical1: tl.constexpr,
    logical2: tl.constexpr,
    block0: tl.constexpr,
    block1: tl.constexpr,
    block2: tl.constexpr,
):
    value = tl.load(x + tl.arange(0, block0 * block1 * block2))
    value = tl.reshape(value, (block0, block1, block2))
    pointer = tl.make_block_ptr(
        out,
        shape=(logical0, logical1, logical2),
        strides=(logical1 * logical2, logical2, 1),
        offsets=(0, 0, 0),
        block_shape=(block0, block1, block2),
        order=(2, 1, 0),
    )
    tl.store(pointer, value, boundary_check=(0, 1, 2))


@triton.jit
def _block_store_probe4(
    x,
    out,
    logical0: tl.constexpr,
    logical1: tl.constexpr,
    logical2: tl.constexpr,
    logical3: tl.constexpr,
    block0: tl.constexpr,
    block1: tl.constexpr,
    block2: tl.constexpr,
    block3: tl.constexpr,
):
    value = tl.load(x + tl.arange(0, block0 * block1 * block2 * block3))
    value = tl.reshape(value, (block0, block1, block2, block3))
    pointer = tl.make_block_ptr(
        out,
        shape=(logical0, logical1, logical2, logical3),
        strides=(logical1 * logical2 * logical3, logical2 * logical3, logical3, 1),
        offsets=(0, 0, 0, 0),
        block_shape=(block0, block1, block2, block3),
        order=(3, 2, 1, 0),
    )
    tl.store(pointer, value, boundary_check=(0, 1, 2, 3))


@triton.jit
def _scalar_load_probe(x, out, selected: tl.constexpr):
    tl.store(out, tl.load(x + selected))


@triton.jit
def _program_id_probe(out, axis: tl.constexpr):
    if axis == 0:
        program = tl.program_id(0)
    elif axis == 1:
        program = tl.program_id(1)
    else:
        program = tl.program_id(2)
    tl.store(out + program, program)


@triton.jit
def _control_probe(out, select, count):
    # Keep both values runtime-visible.  The deployed IR contains data-dependent
    # early exits and loop bounds, so a constexpr-only probe would exercise a
    # materially easier compiler path than the assumption it is meant to test.
    value = 0
    if select:
        for _ in range(0, count):
            value += 1
    else:
        value = -1
    tl.store(out, value)


_ACTIVE_LAUNCH_META: ContextVar[dict | None] = ContextVar(
    "vosti_probe_launch_meta", default=None
)


class _LaunchConfiguredKernel:
    """Inject the exact sealed scheduling meta into every primitive launch."""

    def __init__(self, kernel):
        self._kernel = kernel

    def __getattr__(self, name):
        return getattr(self._kernel, name)

    def __getitem__(self, grid):
        launcher = self._kernel[grid]

        def launch(*args, **kwargs):
            meta = _ACTIVE_LAUNCH_META.get()
            if meta is None:
                raise RuntimeError("primitive probe has no sealed launch meta")
            if "num_warps" in kwargs or "num_stages" in kwargs:
                raise RuntimeError("primitive probe launch meta was supplied twice")
            return launcher(
                *args,
                **kwargs,
                num_warps=meta["num_warps"],
                num_stages=meta["num_stages"],
            )

        return launch


for _kernel_name in (
    "_binary_kernel",
    "_unary_kernel",
    "_where_kernel",
    "_cast_kernel",
    "_reduce_kernel",
    "_dot_kernel",
    "_constructor_kernel",
    "_reshape_kernel",
    "_reshape_1_2",
    "_reshape_2_1",
    "_reshape_2_3",
    "_reshape_3_2",
    "_reshape_3_4",
    "_reshape_4_3",
    "_block_load_probe",
    "_block_store_probe",
    "_block_load_probe3",
    "_block_load_probe4",
    "_block_store_probe3",
    "_block_store_probe4",
    "_scalar_load_probe",
    "_program_id_probe",
    "_control_probe",
):
    globals()[_kernel_name] = _LaunchConfiguredKernel(globals()[_kernel_name])


_BINARY_OPS = {
    "+": 0,
    "-": 1,
    "*": 2,
    "/": 3,
    "//": 4,
    "%": 5,
    "<": 6,
    "<=": 7,
    ">": 8,
    ">=": 9,
    "==": 10,
    "!=": 11,
    "and": 12,
    "or": 13,
    "cdiv": 14,
}


def _shape(spec: dict) -> tuple[int, ...]:
    if not spec["shape_is_concrete"]:
        return ()
    return tuple(int(item) for item in spec["shape"])


def _numel(spec: dict | None) -> int:
    if spec is None:
        return 1
    shape = _shape(spec)
    return math.prod(shape) if shape else 1


def _block(n: int) -> int:
    if n <= 0:
        raise ValueError(f"invalid probe size {n}")
    value = triton.next_power_of_2(n)
    if value > 65536:
        raise ValueError(f"flat probe block {value} exceeds supported prototype limit")
    return value


_PHYSICAL_FLOAT_DTYPES = {
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


def _probe_float_dtype(requirement: dict) -> torch.dtype:
    name = requirement.get("_probe_float_dtype", "float32")
    try:
        return _PHYSICAL_FLOAT_DTYPES[name]
    except KeyError as error:
        raise ValueError(f"unsupported physical float probe dtype {name!r}") from error


def _dtype(spec: dict | None, requirement: dict) -> torch.dtype:
    element = "abstract_float" if spec is None else spec["element"]
    if element == "abstract_float":
        return _probe_float_dtype(requirement)
    if element == "abstract_bool":
        return torch.bool
    return torch.int32


def _sequence(n: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if dtype == torch.bool:
        return torch.arange(n, device=device, dtype=torch.int32).remainder(2).bool()
    return torch.arange(n, device=device, dtype=dtype)


def _equal_unchanged(before: torch.Tensor, after: torch.Tensor, changed: int) -> bool:
    mask = torch.ones(before.numel(), dtype=torch.bool, device=before.device)
    mask[changed] = False
    return _equal_bits(before.flatten()[mask], after.flatten()[mask])


def _run_binary(requirement: dict, device: torch.device) -> None:
    operator = requirement["attributes"].get("operator")
    if requirement["operation"] == "triton.maximum":
        op = 15
        operator = "maximum"
    elif requirement["operation"] == "triton.minimum":
        op = 16
        operator = "minimum"
    else:
        if operator not in _BINARY_OPS:
            raise ValueError(f"unsupported binary operator {operator!r}")
        op = _BINARY_OPS[operator]
    n = max(_numel(requirement["output"]), 3)
    lhs_spec = requirement["inputs"][0] if requirement["inputs"] else None
    rhs_spec = requirement["inputs"][1] if len(requirement["inputs"]) > 1 else lhs_spec
    lhs_dtype = _dtype(lhs_spec, requirement)
    rhs_dtype = _dtype(rhs_spec, requirement)
    output_dtype = _dtype(requirement.get("output"), requirement)
    if operator in {"and", "or"}:
        x = torch.arange(n, device=device, dtype=torch.int32).remainder(2).bool()
        y = torch.arange(1, n + 1, device=device, dtype=torch.int32).remainder(2).bool()
    elif lhs_dtype.is_floating_point:
        x = torch.linspace(0.5, 2.5, n, device=device, dtype=lhs_dtype)
        y = torch.linspace(1.5, 3.5, n, device=device, dtype=rhs_dtype)
        if operator in {"<", "<=", ">", ">=", "==", "!="}:
            x[:3] = torch.tensor([-1.0, 0.0, 1.0], device=device, dtype=lhs_dtype)
            y[:3] = torch.tensor([0.0, 0.0, 0.0], device=device, dtype=rhs_dtype)
    else:
        x = torch.arange(n, device=device, dtype=lhs_dtype).remainder(7) + 1
        y = torch.arange(n - 1, -1, -1, device=device, dtype=rhs_dtype).remainder(5) + 1
        if operator in {"<", "<=", ">", ">=", "==", "!="}:
            x[:3] = torch.tensor([0, 1, 2], device=device, dtype=lhs_dtype)
            y[:3] = torch.tensor([1, 1, 1], device=device, dtype=rhs_dtype)
    out = torch.empty(
        n, device=device, dtype=torch.bool if 6 <= op <= 13 else output_dtype
    )
    _binary_kernel[(1,)](x, y, out, n=n, block=_block(n), op=op)
    first = out.clone()
    properties = set(requirement["properties"])
    if properties & {
        "declared_comparison_semantics",
        "declared_boolean_semantics",
        "declared_nonnegative_integer_semantics",
    }:
        reference_x = x.cpu()
        reference_y = y.cpu()
        expected = {
            "+": lambda: reference_x + reference_y,
            "-": lambda: reference_x - reference_y,
            "*": lambda: reference_x * reference_y,
            "/": lambda: reference_x / reference_y,
            "//": lambda: reference_x // reference_y,
            "%": lambda: reference_x % reference_y,
            "<": lambda: reference_x < reference_y,
            "<=": lambda: reference_x <= reference_y,
            ">": lambda: reference_x > reference_y,
            ">=": lambda: reference_x >= reference_y,
            "==": lambda: reference_x == reference_y,
            "!=": lambda: reference_x != reference_y,
            "and": lambda: torch.logical_and(reference_x.bool(), reference_y.bool()),
            "or": lambda: torch.logical_or(reference_x.bool(), reference_y.bool()),
            "cdiv": lambda: (reference_x + reference_y - 1) // reference_y,
            "maximum": lambda: torch.maximum(reference_x, reference_y),
            "minimum": lambda: torch.minimum(reference_x, reference_y),
        }.get(operator)
        if expected is None:
            raise ValueError(
                f"no declared-semantics oracle for binary operator {operator!r}"
            )
        if not _equal_bits(first.cpu(), expected().to(first.dtype)):
            raise AssertionError(
                f"binary operator {operator!r} differs from its declared semantics"
            )
    _binary_kernel[(1,)](x, y, out, n=n, block=_block(n), op=op)
    if not _equal_bits(first, out):
        raise AssertionError("binary operation is nondeterministic")
    if x.dtype == torch.bool:
        x[0] = torch.logical_not(x[0])
    else:
        x[0] = x[0] + 7
    _binary_kernel[(1,)](x, y, out, n=n, block=_block(n), op=op)
    if not _equal_unchanged(first, out, 0):
        raise AssertionError("binary operation changed an unrelated lane")
    if "finite_zero_product_is_numerical_zero" in properties:
        x.fill_(-2)
        y.zero_()
        _binary_kernel[(1,)](x, y, out, n=n, block=_block(n), op=op)
        if not torch.all(out == 0):
            raise AssertionError("finite multiplication by zero was not zero")
        x.zero_()
        y.fill_(2)
        _binary_kernel[(1,)](x, y, out, n=n, block=_block(n), op=op)
        if not torch.all(out == 0):
            raise AssertionError("zero multiplied by a finite value was not zero")
    if "negative_infinity_plus_finite_is_negative_infinity" in properties:
        x.fill_(float("-inf"))
        y.fill_(2)
        _binary_kernel[(1,)](x, y, out, n=n, block=_block(n), op=op)
        if not torch.all(torch.isneginf(out)):
            raise AssertionError("-inf plus finite was not -inf")
        x.fill_(2)
        y.fill_(float("-inf"))
        _binary_kernel[(1,)](x, y, out, n=n, block=_block(n), op=op)
        if not torch.all(torch.isneginf(out)):
            raise AssertionError("finite plus -inf was not -inf")
    if "finite_self_subtraction_is_positive_zero" in properties:
        x.copy_(torch.arange(1, n + 1, device=device, dtype=lhs_dtype))
        y.copy_(x)
        _binary_kernel[(1,)](x, y, out, n=n, block=_block(n), op=op)
        if not _equal_bits(out, torch.zeros_like(out)):
            raise AssertionError("finite self-subtraction was not zero")
        if out.dtype.is_floating_point and torch.any(torch.signbit(out)):
            raise AssertionError(
                "finite self-subtraction did not produce positive zero"
            )
    if "negative_infinity_minus_finite_is_negative_infinity" in properties:
        x.fill_(float("-inf"))
        y.fill_(2)
        _binary_kernel[(1,)](x, y, out, n=n, block=_block(n), op=op)
        if not torch.all(torch.isneginf(out)):
            raise AssertionError("-inf minus finite was not -inf")
    if "finite_one_multiplicative_identity" in properties:
        x.copy_(torch.arange(1, n + 1, device=device, dtype=lhs_dtype))
        y.fill_(1)
        _binary_kernel[(1,)](x, y, out, n=n, block=_block(n), op=op)
        if not _equal_bits(out, x):
            raise AssertionError("multiplication by one changed a finite value")
        y.copy_(x)
        x.fill_(1)
        _binary_kernel[(1,)](x, y, out, n=n, block=_block(n), op=op)
        if not _equal_bits(out, y):
            raise AssertionError("one times a finite value changed that value")
    if "negative_infinity_times_positive_is_negative_infinity" in properties:
        x.fill_(float("-inf"))
        y.fill_(2)
        _binary_kernel[(1,)](x, y, out, n=n, block=_block(n), op=op)
        if not torch.all(torch.isneginf(out)):
            raise AssertionError("-inf times positive was not -inf")
        x.fill_(2)
        y.fill_(float("-inf"))
        _binary_kernel[(1,)](x, y, out, n=n, block=_block(n), op=op)
        if not torch.all(torch.isneginf(out)):
            raise AssertionError("positive times -inf was not -inf")
    if "maximum_finite_and_negative_infinity_returns_finite_operand" in properties:
        x.fill_(2)
        y.fill_(float("-inf"))
        _binary_kernel[(1,)](x, y, out, n=n, block=_block(n), op=op)
        if not _equal_bits(out, x):
            raise AssertionError("maximum(x, -inf) did not preserve finite x")
        x.fill_(float("-inf"))
        y.fill_(2)
        _binary_kernel[(1,)](x, y, out, n=n, block=_block(n), op=op)
        if not _equal_bits(out, y):
            raise AssertionError("maximum(-inf, x) did not preserve finite x")


def _run_unary(requirement: dict, device: torch.device) -> None:
    operation = requirement["operation"]
    op = {
        "triton.exp2": 0,
        "triton.sigmoid": 1,
        "triton.rsqrt": 2,
        "triton.log2": 3,
        "triton.logical_not": 4,
    }[operation]
    n = max(_numel(requirement["output"]), 2)
    dtype = torch.bool if op == 4 else _probe_float_dtype(requirement)
    x = (
        torch.arange(n, device=device, dtype=torch.int32).remainder(2).bool()
        if op == 4
        else torch.arange(1, n + 1, device=device, dtype=dtype)
    )
    out = torch.empty(n, device=device, dtype=dtype)
    _unary_kernel[(1,)](x, out, n=n, block=_block(n), op=op)
    first = out.clone()
    if op == 4 and not _equal_bits(first, torch.logical_not(x)):
        raise AssertionError("logical not did not complement boolean lanes")
    _unary_kernel[(1,)](x, out, n=n, block=_block(n), op=op)
    if not _equal_bits(first, out):
        raise AssertionError("unary operation is nondeterministic")
    if op == 4:
        x[0] = torch.logical_not(x[0])
    else:
        x[0] = x[0] + 1
    _unary_kernel[(1,)](x, out, n=n, block=_block(n), op=op)
    if not _equal_unchanged(first, out, 0):
        raise AssertionError("unary operation changed an unrelated lane")
    if operation == "triton.exp2":
        x.fill_(float("-inf"))
        x[0] = 0
        _unary_kernel[(1,)](x, out, n=n, block=_block(n), op=op)
        if out[0].item() != 1 or not _equal_bits(out[1:], torch.zeros_like(out[1:])):
            raise AssertionError("exp2 sentinel identities failed")


def _run_where(requirement: dict, device: torch.device) -> None:
    n = max(_numel(requirement["output"]), 2)
    cond = torch.arange(n, device=device, dtype=torch.int32) % 2
    dtype = _dtype(requirement["output"], requirement)
    x = torch.arange(n, device=device, dtype=dtype)
    y = x + 100
    out = torch.empty_like(x)
    _where_kernel[(1,)](cond, x, y, out, n=n, block=_block(n))
    expected = torch.where(cond.bool(), x, y)
    if not _equal_bits(out, expected):
        raise AssertionError("where did not select the declared lane")
    unselected = x.clone()
    unselected[cond == 0] += 1000
    _where_kernel[(1,)](cond, unselected, y, out, n=n, block=_block(n))
    if not _equal_bits(out, expected):
        raise AssertionError("where depended on an unselected true branch")
    unselected_false = y.clone()
    unselected_false[cond != 0] += 1000
    _where_kernel[(1,)](cond, x, unselected_false, out, n=n, block=_block(n))
    if not _equal_bits(out, expected):
        raise AssertionError("where depended on an unselected false branch")


def _run_cast(requirement: dict, device: torch.device) -> None:
    n = max(_numel(requirement["output"]), 4)
    source_dtype = _dtype(requirement["inputs"][0], requirement)
    target = requirement["attributes"]["target"]
    dtype = (
        torch.bfloat16
        if "element_ty" in target or target == "tl.bfloat16"
        else torch.float32
    )
    if requirement["attributes"]["proof_kind"] == "int32":
        dtype = torch.int32
        values = torch.arange(n, device=device, dtype=torch.int32)
    else:
        values = torch.linspace(-2, 2, n, device=device, dtype=source_dtype)
        values[:3] = torch.tensor(
            [0.0, 1.0, float("-inf")], device=device, dtype=source_dtype
        )
    out = torch.empty(n, device=device, dtype=dtype)
    _cast_kernel[(1,)](values, out, n=n, block=_block(n))
    first = out.clone()
    _cast_kernel[(1,)](values, out, n=n, block=_block(n))
    if not _equal_bits(first, out):
        raise AssertionError("cast is nondeterministic")
    if requirement["attributes"]["proof_kind"] == "float":
        if out[0].item() != 0 or out[1].item() != 1 or not torch.isneginf(out[2]):
            raise AssertionError("cast did not preserve 0, 1, and -inf")
    elif not _equal_bits(first, values):
        raise AssertionError("signed int32 identity cast changed a value")
    values[-1] = values[-1] + 1
    _cast_kernel[(1,)](values, out, n=n, block=_block(n))
    if not _equal_unchanged(first, out, n - 1):
        raise AssertionError("cast changed an unrelated lane")


def _run_reduction(requirement: dict, device: torch.device) -> None:
    input_shape = _shape(requirement["inputs"][0])
    if len(input_shape) == 1:
        rows, cols = 1, input_shape[0]
        axis = 1
    elif len(input_shape) == 2:
        rows, cols = input_shape
        axis = int(requirement["attributes"]["axis"])
    else:
        raise ValueError(f"unsupported reduction shape {input_shape}")
    if not (
        triton.next_power_of_2(rows) == rows and triton.next_power_of_2(cols) == cols
    ):
        raise ValueError("reduction probe currently requires power-of-two axes")
    is_max = requirement["operation"] == "triton.reduce_max"
    dtype = _dtype(requirement["inputs"][0], requirement)
    out_size = cols if axis == 0 else rows
    if dtype == torch.bool:
        x = _sequence(rows * cols, device=device, dtype=dtype)
        out = torch.empty(out_size, device=device, dtype=torch.int32)
        reduction_op = 2
    else:
        x = torch.arange(1, rows * cols + 1, device=device, dtype=dtype)
        out = torch.empty(out_size, device=device, dtype=dtype)
        reduction_op = int(is_max)
    _reduce_kernel[(1,)](x, out, rows=rows, cols=cols, axis=axis, op=reduction_op)
    first = out.clone()
    _reduce_kernel[(1,)](x, out, rows=rows, cols=cols, axis=axis, op=reduction_op)
    if not _equal_bits(first, out):
        raise AssertionError("reduction is nondeterministic for fixed inputs")
    if dtype == torch.bool:
        x[0] = torch.logical_not(x[0])
    else:
        x[0] += 100
    _reduce_kernel[(1,)](x, out, rows=rows, cols=cols, axis=axis, op=reduction_op)
    changed = 0
    if not _equal_unchanged(first, out, changed):
        raise AssertionError("reduction changed an unrelated output lane")
    if dtype == torch.bool:
        for value in (False, True):
            x.fill_(value)
            _reduce_kernel[(1,)](
                x, out, rows=rows, cols=cols, axis=axis, op=reduction_op
            )
            if not _equal_bits(out, torch.full_like(out, int(value))):
                raise AssertionError(
                    f"bool reduce_max did not preserve constant value {value}"
                )
        if out.dtype != torch.int32:
            raise AssertionError("bool max did not promote to int32")
        return
    if is_max:
        x.fill_(float("-inf"))
    else:
        x.zero_()
    _reduce_kernel[(1,)](x, out, rows=rows, cols=cols, axis=axis, op=int(is_max))
    expected = torch.full_like(out, float("-inf")) if is_max else torch.zeros_like(out)
    if not (_equal_bits(out, expected) if is_max else bool(torch.all(out == 0))):
        raise AssertionError("reduction constant-zero/negative-infinity property failed")
    if is_max and "constant_value_is_preserved" in requirement["properties"]:
        for value in (0.0, 1.0, -3.0):
            x.fill_(value)
            _reduce_kernel[(1,)](
                x, out, rows=rows, cols=cols, axis=axis, op=int(is_max)
            )
            if not _equal_bits(out, torch.full_like(out, value)):
                raise AssertionError(
                    f"reduce_max did not preserve constant value {value}"
                )


def _run_dot(requirement: dict, device: torch.device) -> None:
    lhs_shape = _shape(requirement["inputs"][0])
    rhs_shape = _shape(requirement["inputs"][1])
    if len(lhs_shape) != 2 or len(rhs_shape) != 2:
        raise ValueError(f"unsupported dot shapes {lhs_shape}, {rhs_shape}")
    m, k = lhs_shape
    rk, n = rhs_shape
    if rk != k:
        raise ValueError("dot reduction dimensions disagree")
    dtype = _probe_float_dtype(requirement)
    a = torch.randn((m, k), device=device, dtype=dtype)
    b = torch.randn((k, n), device=device, dtype=dtype)
    out = torch.empty((m, n), device=device, dtype=torch.float32)
    _dot_kernel[(1,)](a, b, out, m=m, n=n, k=k)
    first = out.clone()
    _dot_kernel[(1,)](a, b, out, m=m, n=n, k=k)
    if not _equal_bits(first, out):
        raise AssertionError("dot is nondeterministic")
    a[0] += 1
    _dot_kernel[(1,)](a, b, out, m=m, n=n, k=k)
    if m > 1 and not _equal_bits(first[1:], out[1:]):
        raise AssertionError("dot lhs row changed unrelated output rows")
    _dot_kernel[(1,)](a, b, out, m=m, n=n, k=k)
    before_column_change = out.clone()
    b[:, 0] += 1
    _dot_kernel[(1,)](a, b, out, m=m, n=n, k=k)
    if n > 1 and not _equal_bits(before_column_change[:, 1:], out[:, 1:]):
        raise AssertionError("dot rhs column changed unrelated output columns")
    a[:, 0] = 0
    _dot_kernel[(1,)](a, b, out, m=m, n=n, k=k)
    zero_first = out.clone()
    b[0] += 100
    _dot_kernel[(1,)](a, b, out, m=m, n=n, k=k)
    if not _equal_bits(zero_first, out):
        raise AssertionError("zero lhs reduction lane contributed to dot")
    b[0] = 0
    _dot_kernel[(1,)](a, b, out, m=m, n=n, k=k)
    zero_rhs = out.clone()
    a[:, 0] += 100
    _dot_kernel[(1,)](a, b, out, m=m, n=n, k=k)
    if not _equal_bits(zero_rhs, out):
        raise AssertionError("zero rhs reduction lane contributed to dot")
    # The dot primitive, unlike elementwise multiplication, declares a
    # zero-lane rule. Exercise both signs with no other contributions so a
    # numerical-equality comparator cannot hide an output zero-sign change.
    for zero in (0.0, -0.0):
        a.fill_(zero)
        for finite in (1.0, -1.0):
            b.fill_(finite)
            _dot_kernel[(1,)](a, b, out, m=m, n=n, k=k)
            if not _equal_bits(out, torch.zeros_like(out)):
                raise AssertionError("dot zero-product accumulation was not positive zero")


def _run_constructor(requirement: dict, device: torch.device) -> None:
    n = _numel(requirement["output"])
    dtype = _dtype(requirement["output"], requirement)
    out = torch.empty(n, device=device, dtype=dtype)
    operation = requirement["operation"]
    op = {"triton.zeros": 0, "triton.full": 1, "triton.arange": 2}[operation]
    _constructor_kernel[(1,)](out, n=n, block=_block(n), op=op)
    first = out.clone()
    expected = (
        torch.zeros_like(out)
        if op == 0
        else torch.full_like(out, 3)
        if op == 1
        else torch.arange(n, device=device, dtype=dtype)
    )
    if not _equal_bits(out, expected):
        raise AssertionError(f"{operation} logical lane mapping failed")
    out.zero_()
    _constructor_kernel[(1,)](out, n=n, block=_block(n), op=op)
    if not _equal_bits(out, first):
        raise AssertionError(f"{operation} shape or values were nondeterministic")


def _run_shape(requirement: dict, device: torch.device) -> None:
    input_shape = _shape(requirement["inputs"][0])
    output_shape = _shape(requirement["output"])
    if math.prod(input_shape or (1,)) != math.prod(output_shape or (1,)):
        if requirement["operation"] != "triton.broadcast_to":
            raise ValueError("non-broadcast shape op changed element count")
    operation = requirement["operation"]
    if operation in {"triton.shape.unsqueeze", "triton.shape.squeeze"}:
        n = _numel(requirement["inputs"][0])
        dtype = _dtype(requirement["inputs"][0], requirement)
        x = _sequence(n, device=device, dtype=dtype)
        out = torch.empty_like(x)
        ranks = (len(input_shape), len(output_shape))
        if ranks == (1, 2):
            _reshape_1_2[(1,)](x, out, a=output_shape[0], b=output_shape[1])
        elif ranks == (2, 1):
            _reshape_2_1[(1,)](x, out, a=input_shape[0], b=input_shape[1])
        elif ranks == (2, 3):
            _reshape_2_3[(1,)](
                x,
                out,
                in0=input_shape[0],
                in1=input_shape[1],
                out0=output_shape[0],
                out1=output_shape[1],
                out2=output_shape[2],
            )
        elif ranks == (3, 2):
            _reshape_3_2[(1,)](
                x,
                out,
                in0=input_shape[0],
                in1=input_shape[1],
                in2=input_shape[2],
                out0=output_shape[0],
                out1=output_shape[1],
            )
        elif ranks == (3, 4):
            _reshape_3_4[(1,)](
                x,
                out,
                in0=input_shape[0],
                in1=input_shape[1],
                in2=input_shape[2],
                out0=output_shape[0],
                out1=output_shape[1],
                out2=output_shape[2],
                out3=output_shape[3],
            )
        elif ranks == (4, 3):
            _reshape_4_3[(1,)](
                x,
                out,
                in0=input_shape[0],
                in1=input_shape[1],
                in2=input_shape[2],
                in3=input_shape[3],
                out0=output_shape[0],
                out1=output_shape[1],
                out2=output_shape[2],
            )
        else:
            raise ValueError(f"unsupported exact reshape ranks {ranks}")
        if not _equal_bits(out, x):
            raise AssertionError("reshape changed logical lane order")
        return
    if len(input_shape) != 2 or len(output_shape) != 2:
        raise ValueError(f"unsupported shape mapping {input_shape}->{output_shape}")
    in0, in1 = input_shape
    out0, out1 = output_shape
    dtype = _dtype(requirement["inputs"][0], requirement)
    x = _sequence(in0 * in1, device=device, dtype=dtype)
    out = torch.empty(out0 * out1, device=device, dtype=dtype)
    op = 0 if operation == "triton.trans" else 1
    _reshape_kernel[(1,)](x, out, in0=in0, in1=in1, out0=out0, out1=out1, op=op)
    source = x.reshape(in0, in1)
    expected = source.T if op == 0 else torch.broadcast_to(source, (out0, out1))
    if not _equal_bits(out.reshape(out0, out1), expected):
        raise AssertionError(f"{operation} logical mapping failed")


def _run_memory(requirement: dict, device: torch.device) -> None:
    operation = requirement["operation"]
    if operation == "triton.load.scalar":
        dtype = _dtype(requirement["output"], requirement)
        x = torch.tensor([3, 5, 7], device=device, dtype=dtype)
        out = torch.empty(1, device=device, dtype=dtype)
        _scalar_load_probe[(1,)](x, out, selected=1)
        if out.item() != 5:
            raise AssertionError("scalar load read the wrong element")
        x[0] = 99
        _scalar_load_probe[(1,)](x, out, selected=1)
        if out.item() != 5:
            raise AssertionError("scalar load depended on an unaddressed element")
        return
    if operation == "triton.block_pointer.view":
        synthetic = {
            **requirement,
            "operation": "triton.load.block",
            "output": requirement["output"],
        }
        _run_memory(synthetic, device)
        return
    spec = (
        requirement["output"]
        if operation == "triton.load.block"
        else requirement["inputs"][0]
    )
    dtype = _dtype(spec, requirement)
    shape = _shape(spec)
    if len(shape) == 1:
        block_rows, block_cols = 1, shape[0]
    elif len(shape) == 2:
        block_rows, block_cols = shape
    elif len(shape) in {3, 4}:
        if math.prod(shape) > 65536:
            raise ValueError("memory block exceeds supported prototype limit")
        shrink_axis = next(
            (index for index, extent in enumerate(shape) if extent > 1), None
        )
        if shrink_axis is None:
            raise ValueError("memory padding probe needs one non-singleton axis")
        logical_shape = list(shape)
        logical_shape[shrink_axis] -= 1
        logical_numel = math.prod(logical_shape)
        slices = tuple(slice(0, extent) for extent in logical_shape)
        if operation == "triton.load.block":
            x = _sequence(max(logical_numel, 1), device=device, dtype=dtype) + 1
            out = torch.empty(math.prod(shape), device=device, dtype=dtype)
            if len(shape) == 3:
                _block_load_probe3[(1,)](
                    x,
                    out,
                    logical0=logical_shape[0],
                    logical1=logical_shape[1],
                    logical2=logical_shape[2],
                    block0=shape[0],
                    block1=shape[1],
                    block2=shape[2],
                )
            else:
                _block_load_probe4[(1,)](
                    x,
                    out,
                    logical0=logical_shape[0],
                    logical1=logical_shape[1],
                    logical2=logical_shape[2],
                    logical3=logical_shape[3],
                    block0=shape[0],
                    block1=shape[1],
                    block2=shape[2],
                    block3=shape[3],
                )
            expected = torch.zeros(shape, device=device, dtype=dtype)
            expected[slices] = x.reshape(logical_shape)
            if not _equal_bits(out.reshape(shape), expected):
                raise AssertionError("ranked block load mapping or padding failed")
        elif operation == "triton.store.block":
            x = _sequence(math.prod(shape), device=device, dtype=dtype) + 1
            out = torch.full((math.prod(shape),), -9, device=device, dtype=dtype)
            if len(shape) == 3:
                _block_store_probe3[(1,)](
                    x,
                    out,
                    logical0=logical_shape[0],
                    logical1=logical_shape[1],
                    logical2=logical_shape[2],
                    block0=shape[0],
                    block1=shape[1],
                    block2=shape[2],
                )
            else:
                _block_store_probe4[(1,)](
                    x,
                    out,
                    logical0=logical_shape[0],
                    logical1=logical_shape[1],
                    logical2=logical_shape[2],
                    logical3=logical_shape[3],
                    block0=shape[0],
                    block1=shape[1],
                    block2=shape[2],
                    block3=shape[3],
                )
            expected = x.reshape(shape)[slices].flatten()
            if not _equal_bits(out[:logical_numel], expected):
                raise AssertionError("ranked block store mapping failed")
            if not _equal_bits(
                out[logical_numel:], torch.full_like(out[logical_numel:], -9)
            ):
                raise AssertionError("ranked block store wrote outside boundary")
        else:
            raise ValueError(f"unsupported ranked memory operation {operation}")
        return
    else:
        raise ValueError(f"unsupported memory block shape {shape}")
    if block_rows * block_cols > 65536:
        raise ValueError("memory block exceeds supported prototype limit")
    logical_rows = block_rows
    logical_cols = max(block_cols - 1, 0)
    if operation == "triton.load.block":
        x = (
            _sequence(max(logical_rows * logical_cols, 1), device=device, dtype=dtype)
            + 1
        )
        out = torch.empty(block_rows * block_cols, device=device, dtype=dtype)
        _block_load_probe[(1,)](
            x,
            out,
            logical_rows=logical_rows,
            logical_cols=logical_cols,
            block_rows=block_rows,
            block_cols=block_cols,
        )
        matrix = out.reshape(block_rows, block_cols)
        if logical_cols and not _equal_bits(
            matrix[:, :logical_cols], x.reshape(logical_rows, logical_cols)
        ):
            raise AssertionError("block load mapped an in-bounds lane incorrectly")
        if not _equal_bits(
            matrix[:, logical_cols:], torch.zeros_like(matrix[:, logical_cols:])
        ):
            raise AssertionError("block load padding was not positive zero")
    elif operation == "triton.store.block":
        x = _sequence(block_rows * block_cols, device=device, dtype=dtype) + 1
        out = torch.full((block_rows * block_cols,), -9, device=device, dtype=dtype)
        _block_store_probe[(1,)](
            x,
            out,
            logical_rows=logical_rows,
            logical_cols=logical_cols,
            block_rows=block_rows,
            block_cols=block_cols,
        )
        written = out[: logical_rows * logical_cols]
        expected = x.reshape(block_rows, block_cols)[:, :logical_cols].flatten()
        if not _equal_bits(written, expected):
            raise AssertionError("block store mapped an in-bounds lane incorrectly")
        if not _equal_bits(
            out[logical_rows * logical_cols :],
            torch.full_like(out[logical_rows * logical_cols :], -9),
        ):
            raise AssertionError("block store wrote outside its logical boundary")
    else:
        raise ValueError(f"unsupported memory operation {operation}")


def _run_control(requirement: dict, device: torch.device) -> None:
    out = torch.empty(4, device=device, dtype=torch.int32)
    if requirement["operation"] == "triton.program_id":
        axis = int(requirement["attributes"]["axis"])
        if axis not in {0, 1, 2}:
            raise ValueError(f"unsupported program_id axis {axis}")
        grid = [1, 1, 1]
        grid[axis] = 4
        _program_id_probe[tuple(grid)](out, axis=axis)
        if not _equal_bits(out, torch.arange(4, device=device, dtype=torch.int32)):
            raise AssertionError(
                f"program_id axis {axis} did not enumerate the launch grid"
            )
    else:
        _control_probe[(1,)](out, select=True, count=3)
        if out[0].item() != 3:
            raise AssertionError("control flow did not execute selected loop body")
        _control_probe[(1,)](out, select=True, count=1)
        if out[0].item() != 1:
            raise AssertionError("control flow ignored its runtime loop bound")
        _control_probe[(1,)](out, select=False, count=3)
        if out[0].item() != -1:
            raise AssertionError("control flow did not execute selected branch")


def _run_scalar(requirement: dict, device: torch.device) -> None:
    # Scalar min/max uses the same lane-local primitive at a one-lane shape.
    synthetic = {
        **requirement,
        "operation": (
            "triton.minimum"
            if requirement["operation"] == "triton.scalar.min"
            else "triton.maximum"
        ),
        "attributes": {},
        "inputs": requirement["inputs"]
        or [{"element": "abstract_int", "shape": [], "shape_is_concrete": True}],
        "output": requirement["output"]
        or {
            "element": "abstract_int",
            "shape": [],
            "shape_is_concrete": True,
        },
    }
    while len(synthetic["inputs"]) < 2:
        synthetic["inputs"].append(synthetic["inputs"][0])
    _run_binary(synthetic, device)


_RUNNERS: dict[str, Callable[[dict, torch.device], None]] = {
    "elementwise_binary": _run_binary,
    "elementwise_unary": _run_unary,
    "where": _run_where,
    "cast": _run_cast,
    "reduction": _run_reduction,
    "dot": _run_dot,
    "constructor": _run_constructor,
    "shape_mapping": _run_shape,
    "memory": _run_memory,
    "control": _run_control,
    "scalar": _run_scalar,
}


def _signature(requirement: dict) -> str:
    body = {
        key: value for key, value in requirement.items() if key not in {"id", "sites"}
    }
    return json.dumps(body, sort_keys=True, separators=(",", ":"))


def _float_probe_dtype_names(requirement: dict) -> tuple[str | None, ...]:
    specs = [*requirement.get("inputs", [])]
    output = requirement.get("output")
    if output is not None:
        specs.append(output)
    if not any(spec.get("element") == "abstract_float" for spec in specs):
        return (None,)
    names = requirement.get("_physical_float_dtypes")
    if (
        not isinstance(names, list)
        or not names
        or names != sorted(set(names))
        or any(name not in _PHYSICAL_FLOAT_DTYPES for name in names)
    ):
        raise ValueError(f"invalid physical float probe domain {names!r}")
    if requirement["operation"] in FLOAT32_ONLY_OPERATIONS:
        if "float32" not in names:
            raise ValueError(
                f"{requirement['operation']} requires float32 qualification"
            )
        return ("float32",)
    return tuple(names)


def _sealed_launch_meta_by_case(candidate: dict) -> dict[str, list[dict]]:
    contexts: dict[str, set[tuple[int, int]]] = defaultdict(set)
    for launch in candidate.get("launches", []):
        case_id = launch.get("kernel_case_id")
        config = launch.get("config")
        if not isinstance(case_id, str) or not isinstance(config, dict):
            raise ValueError("candidate has a malformed sealed launch")
        warps = config.get("num_warps")
        stages = config.get("num_stages")
        if (
            isinstance(warps, bool)
            or not isinstance(warps, int)
            or warps <= 0
            or isinstance(stages, bool)
            or not isinstance(stages, int)
            or stages <= 0
        ):
            raise ValueError("sealed launch has invalid num_warps/num_stages")
        contexts[case_id].add((warps, stages))
    return {
        case_id: [
            {"num_warps": warps, "num_stages": stages}
            for warps, stages in sorted(values)
        ]
        for case_id, values in contexts.items()
    }


def run_candidate_requirements(candidate: dict) -> list[dict]:
    """Run each distinct probe signature once and return one result per ID."""

    if not torch.cuda.is_available():
        raise RuntimeError("backend qualification requires CUDA")
    device = torch.device("cuda", torch.cuda.current_device())
    launch_meta_by_case = _sealed_launch_meta_by_case(candidate)
    by_requirement_id: dict[str, dict] = {}
    meta_by_requirement_id: dict[str, set[tuple[int, int]]] = defaultdict(set)
    for case in candidate["kernel_cases"]:
        case_id = case.get("case_id")
        launch_meta_configs = launch_meta_by_case.get(case_id)
        if not launch_meta_configs:
            raise ValueError(f"kernel case {case_id!r} has no sealed launch meta")
        manifest = case["backend_requirements"]
        physical_float_dtypes = manifest["physical_float_dtypes"]
        for requirement in manifest["requirements"]:
            probe_requirement = {
                **requirement,
                "_physical_float_dtypes": physical_float_dtypes,
            }
            requirement_id = requirement["id"]
            previous = by_requirement_id.setdefault(
                requirement_id, probe_requirement
            )
            if _signature(previous) != _signature(probe_requirement):
                raise ValueError(
                    f"requirement ID {requirement_id} has conflicting signatures"
                )
            meta_by_requirement_id[requirement_id].update(
                (meta["num_warps"], meta["num_stages"])
                for meta in launch_meta_configs
            )

    grouped: dict[str, list[dict]] = defaultdict(list)
    for requirement_id, requirement in by_requirement_id.items():
        launch_meta_configs = [
            {"num_warps": warps, "num_stages": stages}
            for warps, stages in sorted(meta_by_requirement_id[requirement_id])
        ]
        probe_requirement = {
            **requirement,
            "_launch_meta_configs": launch_meta_configs,
        }
        grouped[_signature(probe_requirement)].append(probe_requirement)

    results = []
    for signature in sorted(grouped):
        requirements = grouped[signature]
        representative = requirements[0]
        passed = True
        details = "passed empirical backend probe"
        try:
            validate_requirement_probe_coverage(representative)
            runner = _RUNNERS.get(representative["probe"])
            if runner is None:
                raise ValueError(
                    f"unsupported probe family {representative['probe']!r}"
                )
            tested_dtypes = []
            for launch_meta in representative["_launch_meta_configs"]:
                token = _ACTIVE_LAUNCH_META.set(launch_meta)
                try:
                    for dtype_name in _float_probe_dtype_names(representative):
                        probe_requirement = dict(representative)
                        if dtype_name is not None:
                            probe_requirement["_probe_float_dtype"] = dtype_name
                            if dtype_name not in tested_dtypes:
                                tested_dtypes.append(dtype_name)
                        runner(probe_requirement, device)
                        torch.cuda.synchronize(device)
                finally:
                    _ACTIVE_LAUNCH_META.reset(token)
            launch_detail = ",".join(
                f"warps={meta['num_warps']}/stages={meta['num_stages']}"
                for meta in representative["_launch_meta_configs"]
            )
            details = (
                f"passed empirical backend probe at {launch_detail}"
                if not tested_dtypes
                else "passed empirical backend probes for applicable physical "
                f"float dtypes {','.join(tested_dtypes)} at {launch_detail}"
            )
        except Exception as error:
            passed = False
            details = f"{type(error).__name__}: {error}"
        for requirement in requirements:
            results.append(
                {
                    "requirement_id": requirement["id"],
                    "passed": passed,
                    "details": details,
                    "launch_meta_configs": representative[
                        "_launch_meta_configs"
                    ],
                }
            )
    # Multiple case manifests can intentionally share an obligation ID.  The
    # semantic signature is then identical, so one result is sufficient.
    unique = {}
    for result in results:
        previous = unique.get(result["requirement_id"])
        if previous is not None and previous != result:
            raise AssertionError("one requirement ID received conflicting results")
        unique[result["requirement_id"]] = result
    return [unique[key] for key in sorted(unique)]
