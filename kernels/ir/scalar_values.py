"""Opaque floating values for relational congruence, distinct from Z3 integers.

No real-number or IEEE algebra is assumed here. Operations are deterministic
functions of their ordered operands and static identity. Interpreting them as
the deployed floating operations remains part of the backend trust boundary.
"""

import struct
import z3


FLOAT_VALUE = z3.DeclareSort("KernelFloatValue")


def is_float_value(value: z3.ExprRef) -> bool:
    return value.sort() == FLOAT_VALUE


def float_literal(value: float) -> z3.ExprRef:
    # Preserve the source literal's bits, notably the sign of zero. Different
    # literals are not assumed unequal: target rounding may identify them.
    bits = int.from_bytes(struct.pack(">d", float(value)), "big")
    return z3.Function("kernel_float_literal", z3.IntSort(), FLOAT_VALUE)(bits)


def float_operation(name: str, *operands: z3.ExprRef, predicate: bool = False) -> z3.ExprRef:
    result_sort = z3.BoolSort() if predicate else FLOAT_VALUE
    return z3.Function(f"kernel_float[{name}]", *(operand.sort() for operand in operands),
                       result_sort)(*operands)
