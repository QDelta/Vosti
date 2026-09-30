"""Bounded causal suffix-independence checker.

Checks, for a translated kernel at small CONCRETE shapes, that chosen output
lanes are independent of chosen input lanes ("suffix" positions). This bounded
symbolic evaluator is independent pressure on translation, recurrence, masks,
and the epilogue; the mask-aware dependency/relational passes provide the
separate unbounded conditional certificate used by the framework.

Method: symbolically evaluate the kernel IR with
  - all index arithmetic, loop bounds, masks, and int-tensor contents
    CONCRETE (loops fully unrolled, `where` conditions on index-derived
    masks decide at evaluation time);
  - float-tensor contents SYMBOLIC (one Z3 Real constant per lane);
and check that the Z3 terms of the target output lanes contain none of the
designated suffix variables.  Symbolic evaluation is exact with respect to a
small set of documented IEEE-flavoured annihilation axioms (the trusted
semantics of this checker):

  A1  where(False, a, b) = b   and   where(True, a, b) = a
  A2  x + 0 = 0 + x = x
  A3  x * 0 = 0 * x = 0        (requires FINITE tensor values: no inf/NaN)
  A4  x * 1 = 1 * x = x
  A5  maximum(-inf, x) = maximum(x, -inf) = x ; max-reduce identity is -inf
  A6  exp2(-inf) = 0
  A7  (-inf) - r = -inf        (r a finite real)
  A8  (-inf) * c = -inf        (c a concrete positive finite scalar)
  A9  float_cast(0/1/-inf) preserves that exact sentinel; other cast results
      are opaque deterministic functions of their input

`where` with a SYMBOLIC condition (e.g. the final `0 < logsum` guard) builds
a Z3 If — sound for independence: the If term's free variables are the union
of its parts'.

The certificate is syntactic: the output lane's term simply does not mention
the suffix variables.  This implies equality under any replacement of those
variables; the checker does not separately rerun an equivalent two-copy query.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import z3

from ir import (
    Arange,
    Assign,
    BinOp,
    BoolLit,
    BroadcastTo,
    Cast,
    Exp2,
    FloatLit,
    For,
    Full,
    Grid,
    If,
    IntLit,
    Kernel,
    Let,
    Log2,
    MaskedLoad,
    MaskedStore,
    Max,
    Maximum,
    Min,
    Not,
    ReduceMax,
    ReduceSum,
    Squeeze,
    TensorIndex,
    Transpose,
    Unsqueeze,
    Var,
    Where,
    Zeros,
)

# Lane values: Python int/bool (concrete index math), float (concrete
# scalars; 0.0 is the annihilating zero), the NINF sentinel, the absorbing
# JUNK sentinel (+inf / NaN arising in fully-masked padding lanes: their
# arithmetic result never matters because the mask discards it — storing
# JUNK to a tracked in-mask lane is a hard error, so JUNK can never fake an
# independence certificate), or a Z3 term.
NINF = object()
JUNK = object()


def _is_zero(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and v == 0


def _is_one(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and v == 1


@dataclass
class TensorVal:
    shape: tuple[int, ...]
    lanes: list  # row-major

    def idx(self, ix: tuple[int, ...]) -> int:
        flat = 0
        for d, i in enumerate(ix):
            flat = flat * self.shape[d] + i
        return flat

    def get(self, ix: tuple[int, ...]):
        return self.lanes[self.idx(ix)]

    def set(self, ix: tuple[int, ...], v) -> None:
        self.lanes[self.idx(ix)] = v


def _iter_ix(shape):
    if not shape:
        yield ()
        return
    ix = [0] * len(shape)
    while True:
        yield tuple(ix)
        d = len(shape) - 1
        while d >= 0:
            ix[d] += 1
            if ix[d] < shape[d]:
                break
            ix[d] = 0
            d -= 1
        if d < 0:
            return


def _concrete(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _add(a, b):
    if a is JUNK or b is JUNK:
        return JUNK
    if a is NINF or b is NINF:
        # A7 (and -inf + -inf): only reached via max/scores paths.
        return NINF
    if _concrete(a) and _concrete(b):
        return a + b
    if _is_zero(a):
        return b  # A2
    if _is_zero(b):
        return a
    return _to_real(a) + _to_real(b)


def _sub(a, b):
    if a is JUNK or b is JUNK:
        return JUNK
    if a is NINF and b is NINF:
        return JUNK  # NaN
    if a is NINF:
        return NINF  # A7
    if b is NINF:
        return JUNK  # +inf
    if _concrete(a) and _concrete(b):
        return a - b
    if _is_zero(b):
        return a  # A2
    return _to_real(a) - _to_real(b)


def _mul(a, b):
    if a is JUNK or b is JUNK:
        return JUNK
    if a is NINF or b is NINF:
        # Preserve the only exact -inf multiplication needed by the kernel
        # epilogue.  Zero, negative, symbolic, and -inf counterparts can
        # produce NaN or +inf, which this evaluator intentionally represents
        # as JUNK.  JUNK is absorbing and may never reach a tracked store.
        other = b if a is NINF else a
        if _concrete(other) and other > 0:
            return NINF  # A8
        return JUNK
    if _concrete(a) and _concrete(b):
        return a * b
    if _is_zero(a) or _is_zero(b):
        return 0.0  # A3 (finiteness axiom)
    if _is_one(a):
        return b  # A4
    if _is_one(b):
        return a
    return _to_real(a) * _to_real(b)


def _maximum(a, b):
    if a is JUNK or b is JUNK:
        return JUNK
    if a is NINF:
        return b  # A5
    if b is NINF:
        return a
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return max(a, b)
    return z3.If(z3.ToReal(a) >= z3.ToReal(b), a, b) if False else _z3max(a, b)


def _z3max(a, b):
    ar = _to_real(a)
    br = _to_real(b)
    return z3.If(ar >= br, ar, br)


def _to_real(v):
    if isinstance(v, bool):
        raise TypeError("bool in real context")
    if isinstance(v, (int, float)):
        return z3.RealVal(v)
    return v


_EXP2 = z3.Function("exp2", z3.RealSort(), z3.RealSort())
_LOG2 = z3.Function("log2", z3.RealSort(), z3.RealSort())


def _exp2(v):
    if v is JUNK:
        return JUNK
    if v is NINF:
        return 0.0  # A6
    if isinstance(v, (int, float)):
        return _EXP2(z3.RealVal(v))
    return _EXP2(v)


def _log2(v):
    if v is JUNK or v is NINF:
        return JUNK
    if isinstance(v, (int, float)):
        return _LOG2(z3.RealVal(v))
    return _LOG2(v)


def _cast(v, kind: str, target: str):
    if kind == "int32":
        # Translation admits this only for an int32 metadata load cast back
        # to tl.int32, so it is a representation identity.
        return v
    assert kind == "float"
    if v is JUNK:
        return JUNK
    if v is NINF:
        return NINF
    if _is_zero(v):
        return 0.0
    if _is_one(v):
        return 1.0
    cast = z3.Function(
        f"float_cast[{target}]", z3.RealSort(), z3.RealSort()
    )
    if isinstance(v, (int, float)):
        return cast(z3.RealVal(v))
    return cast(v)


class Evaluator:
    """One grid instance of a specialized kernel, symbolically evaluated."""

    def __init__(self, kernel: Kernel, scalar_env: dict[str, int],
                 tensors: dict[str, TensorVal]):
        self.kernel = kernel
        self.env: dict[str, Any] = dict(scalar_env)
        self.tensors = tensors  # params: int tensors concrete, float symbolic
        self.stores: dict[str, TensorVal] = {}
        self.tracked_outputs: set[str] = set()

    # ---- expression evaluation ------------------------------------------
    def e(self, expr) -> Any:
        if isinstance(expr, IntLit):
            return expr.value
        if isinstance(expr, FloatLit):
            v = expr.value
            if v == float("-inf"):
                return NINF
            return float(v)
        if isinstance(expr, BoolLit):
            return bool(expr.value)
        if isinstance(expr, Var):
            if expr.name in self.env:
                return self.env[expr.name]
            raise KeyError(f"unbound var {expr.name}")
        if isinstance(expr, BinOp):
            return self.binop(expr)
        if isinstance(expr, (Min, Max)):
            vals = [self.e(a) for a in expr.args]
            assert all(isinstance(v, int) for v in vals)
            return min(vals) if isinstance(expr, Min) else max(vals)
        if isinstance(expr, Zeros):
            shape = tuple(self.e(s) for s in expr.shape)
            return TensorVal(shape, [0.0] * _size(shape))
        if isinstance(expr, Full):
            shape = tuple(self.e(s) for s in expr.shape)
            v = self.e(expr.value)
            return TensorVal(shape, [v] * _size(shape))
        if isinstance(expr, Arange):
            a, b = self.e(expr.start), self.e(expr.stop)
            return TensorVal((b - a,), list(range(a, b)))
        if isinstance(expr, Where):
            return self.where(expr)
        if isinstance(expr, ReduceMax):
            return self.reduce(expr.value, expr.axis, _maximum, NINF)
        if isinstance(expr, ReduceSum):
            return self.reduce(expr.value, expr.axis, _add, 0.0)
        if isinstance(expr, Exp2):
            return self.map1(expr.value, _exp2)
        if isinstance(expr, Log2):
            return self.map1(expr.value, _log2)
        if isinstance(expr, Cast):
            return self.map1(
                expr.value,
                lambda value: _cast(value, expr.kind, expr.target),
            )
        if isinstance(expr, Not):
            v = self.e(expr.value)
            if isinstance(v, TensorVal):
                return TensorVal(v.shape, [not x for x in v.lanes])
            return not v
        if isinstance(expr, Maximum):
            return self.map2(self.e(expr.lhs), self.e(expr.rhs), _maximum)
        if isinstance(expr, Unsqueeze):
            v = self.e(expr.value)
            shape = list(v.shape)
            shape.insert(expr.axis, 1)
            return TensorVal(tuple(shape), list(v.lanes))
        if isinstance(expr, Squeeze):
            v = self.e(expr.value)
            assert v.shape[expr.axis] == 1
            shape = list(v.shape)
            del shape[expr.axis]
            return TensorVal(tuple(shape), list(v.lanes))
        if isinstance(expr, BroadcastTo):
            v = self.e(expr.value)
            shape = tuple(self.e(s) for s in expr.shape)
            return _broadcast(v, shape)
        if isinstance(expr, Transpose):
            v = self.e(expr.value)
            perm = tuple(expr.permutation)
            shape = tuple(v.shape[p] for p in perm)
            out = TensorVal(shape, [None] * _size(shape))
            for ix in _iter_ix(v.shape):
                out.set(tuple(ix[p] for p in perm), v.get(ix))
            return out
        if isinstance(expr, TensorIndex):
            base = self.tensor(expr.base.name)
            ix = tuple(self.e(i) for i in expr.indices)
            return base.get(ix)
        if isinstance(expr, MaskedLoad):
            return self.masked_load(expr)
        raise NotImplementedError(f"expr {type(expr).__name__}")

    def binop(self, expr: BinOp):
        op = expr.op
        a = self.e(expr.lhs)
        b = self.e(expr.rhs)
        if op == "@":
            return self.matmul(a, b)
        if isinstance(a, TensorVal) or isinstance(b, TensorVal):
            return self.map2(a, b, lambda x, y: self.scalar_binop(op, x, y))
        return self.scalar_binop(op, a, b)

    def scalar_binop(self, op, a, b):
        if op == "+":
            return _add(a, b)
        if op == "-":
            return _sub(a, b)
        if op == "*":
            return _mul(a, b)
        if op == "/":
            if a is JUNK or b is JUNK:
                return JUNK
            if isinstance(a, int) and isinstance(b, int):
                raise NotImplementedError("int / int")
            return _to_real(a) / _to_real(b)
        if op == "//":
            return a // b
        if op == "%":
            return a % b
        if op == "cdiv":
            return -(-a // b)
        if op in ("<", "<=", ">", ">="):
            if isinstance(a, (int, float)) and isinstance(b, (int, float)) \
                    and a is not NINF and b is not NINF:
                return {"<": a < b, "<=": a <= b,
                        ">": a > b, ">=": a >= b}[op]
            # Symbolic comparison (e.g. `0 < logsum`).
            if a is JUNK or b is JUNK or a is NINF or b is NINF:
                # Only reached on discarded padding lanes; the result feeds a
                # `where` whose output is JUNK either way.
                return False
            ar, br = _to_real(a), _to_real(b)
            return {"<": ar < br, "<=": ar <= br,
                    ">": ar > br, ">=": ar >= br}[op]
        if op in ("==", "!="):
            r = a == b
            return r if op == "==" else not r
        if op in ("and", "&"):
            return a and b
        if op in ("or", "|"):
            return a or b
        raise NotImplementedError(f"binop {op}")

    def where(self, expr: Where):
        c = self.e(expr.cond)

        # A uniform concrete condition selects a whole branch exactly.  Do
        # not evaluate the discarded branch: it may contain IEEE-invalid
        # padding arithmetic that is semantically irrelevant by A1.
        if isinstance(c, bool):
            return self.e(expr.on_true if c else expr.on_false)
        if isinstance(c, TensorVal) and c.lanes and all(
            isinstance(lane, bool) for lane in c.lanes
        ) and all(lane == c.lanes[0] for lane in c.lanes):
            selected = self.e(expr.on_true if c.lanes[0] else expr.on_false)
            if isinstance(selected, TensorVal):
                return _broadcast_or_scalar(selected, c.shape)
            return TensorVal(c.shape, [selected] * _size(c.shape))

        a = self.e(expr.on_true)
        b = self.e(expr.on_false)
        shape = None
        for v in (c, a, b):
            if isinstance(v, TensorVal):
                shape = v.shape
        if shape is None:
            return self.where_lane(c, a, b)
        cb = _broadcast_or_scalar(c, shape)
        ab = _broadcast_or_scalar(a, shape)
        bb = _broadcast_or_scalar(b, shape)
        out = TensorVal(shape, [None] * _size(shape))
        for ix in _iter_ix(shape):
            out.set(ix, self.where_lane(_lane(cb, ix), _lane(ab, ix), _lane(bb, ix)))
        return out

    @staticmethod
    def where_lane(c, a, b):
        if isinstance(c, bool):
            return a if c else b  # A1
        # Symbolic condition.  A NINF branch (only the lse epilogue) is
        # encoded as a distinguished inert constant: it can only flow into
        # UNTRACKED output lanes (`masked_store` skips them), so its value
        # never matters for the certificate.
        if a is JUNK or b is JUNK:
            return JUNK
        if a is NINF:
            a = z3.Real("__ninf")
        if b is NINF:
            b = z3.Real("__ninf")
        return z3.If(c, _to_real(a), _to_real(b))

    def reduce(self, value, axis, f, ident):
        v = self.e(value)
        shape = list(v.shape)
        shape.pop(axis)
        out = TensorVal(tuple(shape), [ident] * _size(tuple(shape)))
        for ix in _iter_ix(v.shape):
            oix = tuple(x for d, x in enumerate(ix) if d != axis)
            out.set(oix, f(out.get(oix), v.get(ix)))
        return out

    def map1(self, value, f):
        v = self.e(value)
        if isinstance(v, TensorVal):
            return TensorVal(v.shape, [f(x) for x in v.lanes])
        return f(v)

    def map2(self, a, b, f):
        if not isinstance(a, TensorVal) and not isinstance(b, TensorVal):
            return f(a, b)
        shape = a.shape if isinstance(a, TensorVal) else b.shape
        ab = _broadcast_or_scalar(a, shape)
        bb = _broadcast_or_scalar(b, shape)
        out = TensorVal(shape, [None] * _size(shape))
        for ix in _iter_ix(shape):
            out.set(ix, f(_lane(ab, ix), _lane(bb, ix)))
        return out

    def matmul(self, a: TensorVal, b: TensorVal) -> TensorVal:
        (m, k), (k2, n) = a.shape, b.shape
        assert k == k2
        out = TensorVal((m, n), [0.0] * (m * n))
        for i in range(m):
            for j in range(n):
                acc = 0.0
                for inner in range(k):
                    acc = _add(
                        acc,
                        _mul(a.get((i, inner)), b.get((inner, j))),
                    )
                out.set((i, j), acc)
        return out

    # ---- tensors, loads, stores -----------------------------------------
    def tensor(self, name: str) -> TensorVal:
        if name in self.stores:
            return self.stores[name]
        return self.tensors[name]

    def region_ix(self, region) -> list[tuple[int, int]]:
        out = []
        for sl in region:
            out.append((self.e(sl.start), self.e(sl.stop)))
        return out

    def masked_load(self, expr: MaskedLoad) -> TensorVal:
        base = self.tensor(expr.base.name)
        reg = self.region_ix(expr.region)
        msk = self.region_ix(expr.mask)
        shape = tuple(hi - lo for lo, hi in reg)
        out = TensorVal(shape, [None] * _size(shape))
        for ix in _iter_ix(shape):
            src = tuple(lo + i for (lo, _), i in zip(reg, ix))
            inb = all(mlo <= s < mhi for s, (mlo, mhi) in zip(src, msk)) \
                and all(0 <= s < d for s, d in zip(src, base.shape))
            out.set(ix, base.get(src) if inb else 0.0)
        return out

    def masked_store(self, st: MaskedStore) -> None:
        name = st.base.name
        if name not in self.tracked_outputs:
            return  # e.g. lse: not verified, avoid NINF-under-symbolic-where
        if name not in self.stores:
            base = self.tensors[name]
            self.stores[name] = TensorVal(base.shape, list(base.lanes))
        dst = self.stores[name]
        reg = self.region_ix(st.region)
        msk = self.region_ix(st.mask)
        val = self.e(st.value)
        shape = tuple(hi - lo for lo, hi in reg)
        assert val.shape == shape, (val.shape, shape)
        for ix in _iter_ix(shape):
            tgt = tuple(lo + i for (lo, _), i in zip(reg, ix))
            inb = all(mlo <= t < mhi for t, (mlo, mhi) in zip(tgt, msk)) \
                and all(0 <= t < d for t, d in zip(tgt, dst.shape))
            if inb:
                v = val.get(ix)
                assert v is not JUNK, \
                    f"JUNK stored to tracked lane {name}{tgt} — unsound"
                dst.set(tgt, v)

    # ---- statements ------------------------------------------------------
    def run_stmts(self, stmts) -> None:
        for st in stmts:
            if isinstance(st, Let):
                self.env[st.var.name] = self.e(st.value)
            elif isinstance(st, Assign):
                assert isinstance(st.target, Var), "TensorView assign unsupported"
                v = self.e(st.value)
                if st.op is None:
                    self.env[st.target.name] = v
                else:
                    cur = self.env[st.target.name]
                    self.env[st.target.name] = self.map2(
                        cur, v, lambda x, y: self.scalar_binop(st.op, x, y))
            elif isinstance(st, MaskedStore):
                self.masked_store(st)
            elif isinstance(st, For):
                lo, hi = self.e(st.iters.start), self.e(st.iters.stop)
                for i in range(lo, hi):
                    self.env[st.var.name] = i
                    self.run_stmts(st.body)
            elif isinstance(st, If):
                c = self.e(st.cond)
                assert isinstance(c, bool), "symbolic If condition"
                self.run_stmts(st.then_body if c else st.else_body)
            else:
                # VarDecl handled at grid level; ignore unknown no-ops loudly.
                raise NotImplementedError(f"stmt {type(st).__name__}")

    def run_grid_instance(self, pids: dict[str, int],
                          tracked_outputs: set[str]) -> None:
        self.tracked_outputs = tracked_outputs
        g: Grid = self.kernel.grid
        for it in g.iters:
            self.env[it.var.name] = pids[it.var.name]
        self.run_stmts(g.body)


def _size(shape) -> int:
    n = 1
    for d in shape:
        n *= d
    return n


def _lane(v, ix):
    return v.get(ix) if isinstance(v, TensorVal) else v


def _broadcast(v: TensorVal, shape) -> TensorVal:
    out = TensorVal(tuple(shape), [None] * _size(shape))
    for ix in _iter_ix(shape):
        src = tuple(0 if v.shape[d] == 1 else ix[d] for d in range(len(shape)))
        out.set(ix, v.get(src))
    return out


def _broadcast_or_scalar(v, shape):
    if not isinstance(v, TensorVal):
        return v
    if v.shape == shape:
        return v
    return _broadcast(v, shape)


def free_reals(term) -> set[str]:
    """Free 0-ary Real constants of a lane value."""
    out: set[str] = set()
    if term is NINF or isinstance(term, (int, float, bool)):
        return out
    stack = [term]
    seen = set()
    while stack:
        t = stack.pop()
        if t.get_id() in seen:
            continue
        seen.add(t.get_id())
        if z3.is_const(t) and t.decl().kind() == z3.Z3_OP_UNINTERPRETED \
                and t.num_args() == 0:
            out.add(t.decl().name())
        stack.extend(t.children())
    return out


def check_lane_independence(lane_term, suffix_vars: set[str]) -> tuple[bool, set[str]]:
    """True iff the lane's term mentions no suffix variable."""
    used = free_reals(lane_term)
    bad = used & suffix_vars
    return (len(bad) == 0, bad)
