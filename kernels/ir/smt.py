"""Shared SMT encoding and obligation results; not an artifact producer."""
from dataclasses import dataclass, field
from . import (
    BinOp, BoolLit, Cast, Expr, FloatLit, IntLit, Log2, Max, Min, Not,
    TensorIndex, Var,
)
from .scalar_values import float_literal, float_operation, is_float_value
import z3

Z3Val = z3.ExprRef | z3.FuncDeclRef


@dataclass(frozen=True)
class ProofCheck:
    name: str
    proved: bool
    details: str


@dataclass(frozen=True)
class ProofResult:
    """SMT obligations and diagnostics, not a standalone kernel certificate."""

    checks: list[ProofCheck]
    diagnostics: list[ProofCheck] = field(default_factory=list)
    deferred_obligations: tuple[str, ...] = ()

    @property
    def checks_ok(self) -> bool:
        """Whether emitted checks pass, before any explicit composition cut."""

        return bool(self.checks) and all(check.proved for check in self.checks)

    @property
    def ok(self) -> bool:
        # An analyzer that accidentally emits no obligations, or leaves a
        # caller-composed cut open, must fail closed.
        return self.checks_ok and not self.deferred_obligations


def z3_prove(name: str, assumptions, claim, timeout: int = 1000) -> ProofCheck:
    solver = z3.Solver()
    solver.set("timeout", timeout)
    solver.add(*assumptions)
    solver.add(z3.Not(claim))
    status = solver.check()
    if status == z3.unsat:
        return ProofCheck(name=name, proved=True, details="unsat")
    if status == z3.sat:
        return ProofCheck(name=name, proved=False, details=str(solver.model()))
    return ProofCheck(name=name, proved=False, details=str(status))


def z3_satisfiable(name: str, assumptions, timeout: int = 1000) -> ProofCheck:
    """Require that an analysis-internal proof context has at least one model.

    Use this only when the analyzer itself has strengthened the theorem
    premises (for example by pairing loop iterators).  Unsatisfiability of the
    source-declared theorem precondition is instead a sound, vacuous Hoare
    theorem and must not be rejected by this helper.
    """
    solver = z3.Solver()
    solver.set("timeout", timeout)
    solver.add(*assumptions)
    status = solver.check()
    if status == z3.sat:
        return ProofCheck(name=name, proved=True, details="sat")
    if status == z3.unsat:
        return ProofCheck(name=name, proved=False, details="unsat preconditions")
    return ProofCheck(name=name, proved=False, details=str(status))


def z3_satisfiability_diagnostic(
    name: str,
    assumptions,
    timeout: int = 1000,
) -> tuple[str, ProofCheck]:
    """Classify theorem-premise satisfiability without making it an obligation."""

    solver = z3.Solver()
    solver.set("timeout", timeout)
    solver.add(*assumptions)
    status = solver.check()
    if status == z3.sat:
        return "sat", ProofCheck(name=name, proved=True, details="sat")
    if status == z3.unsat:
        return "unsat", ProofCheck(
            name=name,
            proved=False,
            details="unsat theorem preconditions; theorem is vacuous",
        )
    return "unknown", ProofCheck(name=name, proved=False, details=str(status))


def expr_to_z3(expr: Expr, env: dict[str, Z3Val]) -> z3.ExprRef:
    """Encode integer geometry exactly and floating values by congruence only."""
    match expr:
        case IntLit(value=value):
            return z3.IntVal(value)
        case FloatLit(value=value):
            return float_literal(value)
        case Var(name=name):
            value = env[name]
            if isinstance(value, z3.ExprRef) and z3.is_real(value):
                raise ValueError("floating values require opaque bindings, not Z3 reals")
            return value  # pyright: ignore[reportReturnType]
        case BinOp(op=op, lhs=lhs_expr, rhs=rhs_expr):
            lhs = expr_to_z3(lhs_expr, env)
            rhs = expr_to_z3(rhs_expr, env)
            if is_float_value(lhs) or is_float_value(rhs) or op == "/":
                comparisons = {"<", "<=", ">", ">=", "==", "!="}
                if op not in comparisons | {"+", "-", "*", "/", "//", "%"}:
                    raise ValueError(f"unsupported floating value operation: {op}")
                # Source FP comparisons are operations too: even x == x need
                # not be true (NaN). This differs from contract equality.
                return float_operation(op, lhs, rhs, predicate=op in comparisons)
            if op == "+":
                return lhs + rhs  # pyright: ignore[reportReturnType]
            if op == "-":
                return lhs - rhs  # pyright: ignore[reportOperatorIssue]
            if op == "*":
                return lhs * rhs  # pyright: ignore[reportReturnType]
            if op == "/":
                return lhs / rhs  # pyright: ignore[reportOperatorIssue]
            if op == "//":
                return lhs / rhs  # pyright: ignore[reportOperatorIssue]
            if op == "%":
                return lhs % rhs  # pyright: ignore[reportOperatorIssue]
            if op == "cdiv":
                return (lhs + rhs - 1) / rhs  # pyright: ignore[reportOperatorIssue, reportReturnType]
            if op == "<":
                return lhs < rhs  # pyright: ignore[reportOperatorIssue]
            if op == "<=":
                return lhs <= rhs  # pyright: ignore[reportOperatorIssue]
            if op == ">":
                return lhs > rhs  # pyright: ignore[reportOperatorIssue]
            if op == ">=":
                return lhs >= rhs  # pyright: ignore[reportOperatorIssue]
            if op == "==":
                return lhs == rhs
            if op == "!=":
                return lhs != rhs
            if op == "and":
                return z3.And(lhs, rhs)
            if op == "or":
                return z3.Or(lhs, rhs)
            raise AssertionError(f"unsupported op: {op}")
        case Min(args=args):
            z3_args = [expr_to_z3(a, env) for a in args]
            if any(is_float_value(arg) for arg in z3_args):
                return float_operation("min", *z3_args)
            result = z3_args[0]
            for arg in z3_args[1:]:
                result = z3.If(result <= arg, result, arg)  # pyright: ignore[reportOperatorIssue]
            return result  # pyright: ignore[reportReturnType]
        case Max(args=args):
            z3_args = [expr_to_z3(a, env) for a in args]
            if any(is_float_value(arg) for arg in z3_args):
                return float_operation("max", *z3_args)
            result = z3_args[0]
            for arg in z3_args[1:]:
                result = z3.If(result >= arg, result, arg)  # pyright: ignore[reportOperatorIssue]
            return result  # pyright: ignore[reportReturnType]
        case Not(value=value):
            return z3.Not(expr_to_z3(value, env))  # pyright: ignore[reportReturnType]
        case Cast(value=value, kind=kind, target=target):
            if kind == "int32":
                # The frontend admits this node only for an @params int32
                # tensor scalar load cast back to tl.int32. Source and
                # target representations are therefore already identical.
                return expr_to_z3(value, env)
            # A concrete cast is deterministic, so congruence is available,
            # but it is not mathematical identity (notably for narrowing
            # float casts).  Keep each target opaque here.  Sentinel facts
            # needed by mask analysis are modeled separately and exported as
            # deployment-time backend obligations.
            assert kind == "float"
            operand = expr_to_z3(value, env)
            return float_operation(f"cast:{kind}:{target}", operand)
        case Log2(value=value):
            operand = expr_to_z3(value, env)
            return float_operation("log2", operand)
        case BoolLit(value=value):
            return z3.BoolVal(value)
        case TensorIndex(base=base, indices=indices):
            func = env[base.name]
            assert isinstance(func, z3.FuncDeclRef)
            z3_indices = [expr_to_z3(idx, env) for idx in indices]
            value = func(*z3_indices)
            if z3.is_real(value):
                raise ValueError("floating tensor reads require opaque values, not Z3 reals")
            return value  # pyright: ignore[reportReturnType]
        case _:
            raise AssertionError(f"unhandled expr: {expr}")
