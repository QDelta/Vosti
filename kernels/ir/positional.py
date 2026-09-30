"""Per-position forward analysis.

Symbolically evaluates a loop body to find positions with known values and
prove accumulator identity. Guarded backward analysis in `ir/regions.py` uses
these facts to narrow dependency regions. `check_iteration_neutral` certifies
a no-op on `acc` when its final TensorFacts have `unchanged_from == acc`.

For each loop-body variable, the analysis tracks:

  * numeric tensors:
      - `zero_where`:    lower bound on positions where the value is 0.
      - `neg_inf_where`: lower bound on positions where the value is -inf.
      - `one_where`:     lower bound on positions where the value is 1.
      - `unchanged_from = v`: equality at every position to v's pre-iteration value.
  * boolean tensors:
      - `true_where`:  the exact value predicate if known, else FALSE.
      - `false_where`: dually.

Lower-bound semantics: `zero_where(T) = P` means "T[i] is numerically zero
for every i satisfying P; elsewhere unknown". It does not distinguish +0
from -0. A consumer may erase a dependency only when its own bitwise rule
allows it: finite(x) does not erase the sign dependency of x * 0, and adding
zero is not bitwise state identity. The dot rule separately relies on its
declared zero-initialized accumulation semantics.

For boolean tensors fully expressible from the IR (Var reads, BinOps,
broadcasts, etc.), structural interpretation computes `true_where` exactly.

Each tensor has a rank `r`; its predicates live in a scope with `r` free
index variables named `_i{0..r-1}` (outermost = `_i0`). Ops that reshape
the tensor (BroadcastTo, Unsqueeze, Squeeze, Transpose) reindex the
predicates accordingly. A rank-0 scalar has no free indices.
"""

from dataclasses import dataclass, field
from . import *
# Coordinate transport is ordinary simultaneous IR substitution. Reuse its
# complete, fail-closed visitor; a partial visitor can leave stale coordinates
# inside nested expressions such as Where, Min, or Max.
from .subst import subst_expr as subst_free_indices
from .reaching_definitions import assigned_variables


# ---------------------------------------------------------------------------
# Caller-supplied assumptions
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NeutralityAssumptions:
    """Explicit, named preconditions the analyzer may rely on.

    These properties cannot be derived from the IR alone. Used rules are
    recorded in `PositionalAnalyzer.used_assumptions`. Each assumption must
    hold at every point where its rule is applied, not necessarily throughout
    the kernel.
    """

    finite_vars: frozenset[str] = frozenset()
    """Variables whose values are guaranteed to contain no NaN and no ±inf
    at lemma-application points. Enables rules requiring a finite other
    operand: for example, (-inf) + finite = -inf, whereas
    (-inf) + (+inf) would be NaN."""

    positive_vars: frozenset[str] = frozenset()
    """Variables whose values are guaranteed to be > 0 at lemma-application
    points. Enables (-inf) * positive = -inf."""

    all_false_masks: frozenset[str] = frozenset()
    """Boolean variables that are provably all-false at fragment entry and
    after every (re-)assignment inside the analyzed body. Typically discharged
    by a separate regional proof (e.g., proving the causal mask is all-false
    beyond the cutoff)."""


# ---------------------------------------------------------------------------
# Predicate representation
# ---------------------------------------------------------------------------

FREE_IDX_PREFIX = "_i"


def free_idx(k: int) -> Var:
    """The k-th canonical free-index variable, `_i{k}`."""
    return Var(f"{FREE_IDX_PREFIX}{k}")


@dataclass(frozen=True)
class Pred:
    """A boolean predicate over the first `rank` canonical free indices.

    `body` is an IR boolean expression that may reference `_i0, _i1, ...`
    up to `_i{rank-1}`. A predicate with rank 0 is a plain scalar bool.
    """

    rank: int
    body: Expr  # BoolLit | BinOp('and'/'or'/comparison) | Not | ...

    def __post_init__(self) -> None:
        assert isinstance(self.rank, int) and self.rank >= 0


def pred_true(rank: int) -> Pred:
    return Pred(rank, BoolLit(True))


def pred_false(rank: int) -> Pred:
    return Pred(rank, BoolLit(False))


def _is_true(p: Pred) -> bool:
    return isinstance(p.body, BoolLit) and p.body.value is True


def _is_false(p: Pred) -> bool:
    return isinstance(p.body, BoolLit) and p.body.value is False


def pred_and(a: Pred, b: Pred) -> Pred:
    assert a.rank == b.rank, f"rank mismatch: {a.rank} vs {b.rank}"
    if _is_false(a) or _is_false(b):
        return pred_false(a.rank)
    if _is_true(a):
        return b
    if _is_true(b):
        return a
    return Pred(a.rank, BinOp("and", a.body, b.body))


def pred_or(a: Pred, b: Pred) -> Pred:
    assert a.rank == b.rank, f"rank mismatch: {a.rank} vs {b.rank}"
    if _is_true(a) or _is_true(b):
        return pred_true(a.rank)
    if _is_false(a):
        return b
    if _is_false(b):
        return a
    return Pred(a.rank, BinOp("or", a.body, b.body))


def pred_not(a: Pred) -> Pred:
    if _is_true(a):
        return pred_false(a.rank)
    if _is_false(a):
        return pred_true(a.rank)
    # Double-negation elimination.
    if isinstance(a.body, Not):
        return Pred(a.rank, a.body.value)
    return Pred(a.rank, Not(a.body))


# ---------------------------------------------------------------------------
# Index remapping for shape-changing ops
# ---------------------------------------------------------------------------


def _rank_of(expr: Expr) -> int:
    t = expr.type
    assert t is not None, f"expression {expr} has no type; infer_types must have run"
    if isinstance(t, TensorType):
        return len(t.dims)
    return 0


# ---------------------------------------------------------------------------
# Per-tensor facts
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TensorFacts:
    """Known facts about a tensor variable's per-position values.

    Numeric *_where predicates are lower bounds on the positions where
    the value equals the named constant; boolean true_where/false_where
    are exact when the defining expression is known, otherwise lower
    bounds. `unchanged_from = v` means equality at every position to v's
    pre-iteration value. Accumulator neutrality requires this fact at the
    end of the body.
    """

    rank: int
    zero_where: Pred
    neg_inf_where: Pred
    one_where: Pred
    true_where: Pred
    false_where: Pred
    unchanged_from: str | None = None


def facts_unknown(rank: int) -> TensorFacts:
    return TensorFacts(
        rank=rank,
        zero_where=pred_false(rank),
        neg_inf_where=pred_false(rank),
        one_where=pred_false(rank),
        true_where=pred_false(rank),
        false_where=pred_false(rank),
        unchanged_from=None,
    )


def facts_all_zero(rank: int) -> TensorFacts:
    return TensorFacts(
        rank=rank,
        zero_where=pred_true(rank),
        neg_inf_where=pred_false(rank),
        one_where=pred_false(rank),
        true_where=pred_false(rank),
        false_where=pred_false(rank),
        unchanged_from=None,
    )


def facts_all_one(rank: int) -> TensorFacts:
    return TensorFacts(
        rank=rank,
        zero_where=pred_false(rank),
        neg_inf_where=pred_false(rank),
        one_where=pred_true(rank),
        true_where=pred_false(rank),
        false_where=pred_false(rank),
        unchanged_from=None,
    )


def facts_all_neg_inf(rank: int) -> TensorFacts:
    return TensorFacts(
        rank=rank,
        zero_where=pred_false(rank),
        neg_inf_where=pred_true(rank),
        one_where=pred_false(rank),
        true_where=pred_false(rank),
        false_where=pred_false(rank),
        unchanged_from=None,
    )


def facts_all_false(rank: int) -> TensorFacts:
    """Boolean tensor: all positions are false."""
    return TensorFacts(
        rank=rank,
        zero_where=pred_false(rank),
        neg_inf_where=pred_false(rank),
        one_where=pred_false(rank),
        true_where=pred_false(rank),
        false_where=pred_true(rank),
        unchanged_from=None,
    )


def facts_unchanged(rank: int, var_name: str) -> TensorFacts:
    """The pre-iteration value of `var_name`, whole-tensor."""
    return TensorFacts(
        rank=rank,
        zero_where=pred_false(rank),
        neg_inf_where=pred_false(rank),
        one_where=pred_false(rank),
        true_where=pred_false(rank),
        false_where=pred_false(rank),
        unchanged_from=var_name,
    )


def facts_bool_exact(rank: int, true_body: Expr) -> TensorFacts:
    """Boolean tensor whose exact value (at free indices) is `true_body`.

    `true_where` is `true_body` (lower bound = exact); `false_where` is
    `¬true_body`. If `true_body` simplifies to a BoolLit, we use the
    all-true/all-false specializations.
    """
    if isinstance(true_body, BoolLit):
        if true_body.value:
            return TensorFacts(
                rank=rank,
                zero_where=pred_false(rank),
                neg_inf_where=pred_false(rank),
                one_where=pred_false(rank),
                true_where=pred_true(rank),
                false_where=pred_false(rank),
                unchanged_from=None,
            )
        return facts_all_false(rank)
    return TensorFacts(
        rank=rank,
        zero_where=pred_false(rank),
        neg_inf_where=pred_false(rank),
        one_where=pred_false(rank),
        true_where=Pred(rank, true_body),
        false_where=Pred(rank, Not(true_body)),
        unchanged_from=None,
    )


@dataclass(frozen=True)
class ForwardFactTrace:
    """Forward facts at exact statement boundaries.

    Backward demand propagation must inspect the reaching definitions at a
    use, not the final environment after later reassignments. Statement
    identity is analysis metadata over the existing typed IR, not another IR.
    """

    final_env: dict[str, TensorFacts]
    before_stmt: dict[int, dict[str, TensorFacts]]
    after_stmt: dict[int, dict[str, TensorFacts]]
    final_elem_body: dict[str, Expr] = field(default_factory=dict)
    before_stmt_elem_body: dict[int, dict[str, Expr]] = field(default_factory=dict)
    after_stmt_elem_body: dict[int, dict[str, Expr]] = field(default_factory=dict)

    def before(self, stmt: Stmt) -> dict[str, TensorFacts]:
        facts = self.before_stmt.get(id(stmt))
        if facts is None:
            raise ValueError(
                "forward fact trace has no entry for backward-analyzed statement"
            )
        return facts

    def after(self, stmt: Stmt) -> dict[str, TensorFacts]:
        facts = self.after_stmt.get(id(stmt))
        if facts is None:
            raise ValueError("forward fact trace has no exit for analyzed statement")
        return facts

    def element_bodies_before(self, stmt: Stmt) -> dict[str, Expr]:
        """Return exact per-position expressions reaching ``stmt``.

        These expressions are analysis facts over the existing typed IR.  In
        particular, preserving them across a loop boundary lets a local
        transition proof expand a pre-loop value such as
        ``tile_start + arange`` instead of treating it as an unrelated input
        tensor.
        """

        bodies = self.before_stmt_elem_body.get(id(stmt))
        if bodies is None:
            raise ValueError(
                "forward fact trace has no element-body entry for statement"
            )
        return bodies

    def element_bodies_after(self, stmt: Stmt) -> dict[str, Expr]:
        bodies = self.after_stmt_elem_body.get(id(stmt))
        if bodies is None:
            raise ValueError(
                "forward fact trace has no element-body exit for statement"
            )
        return bodies


# ---------------------------------------------------------------------------
# Forward analyzer
# ---------------------------------------------------------------------------


@dataclass
class PositionalAnalyzer:
    """Forward pass computing per-position facts for each variable.

    Usage:
        a = PositionalAnalyzer(assumptions, type_env, accumulators={...})
        for stmt in loop_body:
            a.exec_stmt(stmt)
        facts = a.env   # dict[str, TensorFacts]

    Variables in `accumulators` are pre-initialized with
    `unchanged_from = name`, modelling "the pre-iteration value of name".
    Subsequent rules propagate `unchanged_from` through identity-preserving
    operations so we can prove no-op iterations.
    """

    assumptions: NeutralityAssumptions
    type_env: dict[str, Type]
    accumulators: frozenset[str] = field(default_factory=frozenset)
    assume_then_branches: bool = False
    env: dict[str, TensorFacts] = field(default_factory=dict)
    used_assumptions: set[str] = field(default_factory=set)
    # Per-variable symbolic body (Expr over free indices) when we can
    # compute it exactly. Used to build `true_where` for boolean tensors
    # and for lookups from broadcasts.
    elem_body: dict[str, Expr] = field(default_factory=dict)
    before_stmt: dict[int, dict[str, TensorFacts]] = field(default_factory=dict)
    after_stmt: dict[int, dict[str, TensorFacts]] = field(default_factory=dict)
    before_stmt_elem_body: dict[int, dict[str, Expr]] = field(default_factory=dict)
    after_stmt_elem_body: dict[int, dict[str, Expr]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Pre-initialize each accumulator's facts with unchanged_from=name.
        # The rank is read from the type_env; if unknown, default to 0
        # (the analyzer will refresh on first read with the actual rank).
        for name in self.accumulators:
            ty = self.type_env.get(name)
            if isinstance(ty, TensorType):
                rank = len(ty.dims)
            else:
                rank = 0
            self.env[name] = facts_unchanged(rank, name)

        # An all-false fact may describe an input/cut-point value that is not
        # assigned inside the analyzed fragment.  Initialize it at entry as
        # well as overriding later assignments in `_assign`.  This makes the
        # assumption mean the same thing at every program point and lets the
        # generic identity-transition analysis handle either an input fact or
        # a value constructed inside the iteration.
        for name in self.assumptions.all_false_masks:
            typ = self.type_env.get(name)
            if typ is None:
                continue
            if isinstance(typ, TensorType):
                if not isinstance(typ.elem_type, BoolType):
                    raise ValueError(
                        f"all-false fact {name!r} does not name a boolean tensor"
                    )
                rank = len(typ.dims)
            elif isinstance(typ, BoolType):
                rank = 0
            else:
                raise ValueError(
                    f"all-false fact {name!r} does not name a boolean value"
                )
            self.env[name] = facts_all_false(rank)
            self.elem_body[name] = BoolLit(False)

    # -- assumption helpers ------------------------------------------------

    def _underlying_var(self, expr: Expr) -> str | None:
        """Peel off shape-only wrappers (Broadcast/Unsqueeze/Squeeze/Transpose)
        to find the underlying Var name, for finiteness / positivity
        lookups. Returns None if the expression is anything else at its core."""
        match expr:
            case Var(name=n):
                return n
            case Unsqueeze(value=v) | Squeeze(value=v) | BroadcastTo(value=v) | Transpose(value=v):
                return self._underlying_var(v)
            case _:
                return None

    def _is_finite(self, expr: Expr | None, label: str) -> bool:
        if expr is None:
            return False
        name = self._underlying_var(expr)
        if name is None:
            return False
        if name in self.assumptions.finite_vars:
            self.used_assumptions.add(f"finite({name})@{label}")
            return True
        return False

    def _is_named_finite(self, name: str | None, label: str) -> bool:
        """Discharge finiteness through an `unchanged_from` provenance."""
        if name is not None and name in self.assumptions.finite_vars:
            self.used_assumptions.add(f"finite({name})@{label}")
            return True
        return False

    def _is_positive(self, expr: Expr | None, label: str) -> bool:
        if expr is None:
            return False
        name = self._underlying_var(expr)
        if name is None:
            return False
        if name in self.assumptions.positive_vars:
            self.used_assumptions.add(f"positive({name})@{label}")
            return True
        return False

    # -- shape helpers ----------------------------------------------------

    def _rank(self, expr: Expr) -> int:
        return _rank_of(expr)

    # -- public API -------------------------------------------------------

    def trace(self) -> ForwardFactTrace:
        return ForwardFactTrace(
            final_env=dict(self.env),
            before_stmt=dict(self.before_stmt),
            after_stmt=dict(self.after_stmt),
            final_elem_body=dict(self.elem_body),
            before_stmt_elem_body=dict(self.before_stmt_elem_body),
            after_stmt_elem_body=dict(self.after_stmt_elem_body),
        )

    def exec_stmt(self, stmt: Stmt) -> None:
        self.before_stmt[id(stmt)] = dict(self.env)
        self.before_stmt_elem_body[id(stmt)] = dict(self.elem_body)
        self._exec_stmt(stmt)
        self.after_stmt[id(stmt)] = dict(self.env)
        self.after_stmt_elem_body[id(stmt)] = dict(self.elem_body)

    def _forget(self, names) -> None:
        for name in names:
            typ = self.type_env.get(name)
            rank = len(typ.dims) if isinstance(typ, TensorType) else 0
            self.env[name] = facts_unknown(rank)
            self.elem_body.pop(name, None)

    def _exec_stmt(self, stmt: Stmt) -> None:
        match stmt:
            case Let(var=var, value=value):
                self._bind(var.name, value)
            case Assign(target=Var(name=name), op=op, value=value):
                self._assign(name, op, value)
            case Assign(target=TensorView(base=base)):
                # A partial update does not establish a whole-tensor fact.
                self._forget((base.name,))
            case For(var=var, body=body):
                # Execute one parametric iteration.  Facts for values carried
                # across the back edge cannot be taken from the syntactically
                # first iteration: doing so can use an initializer fact (for
                # example acc==0) in every iteration.  Forget them at both
                # boundaries; facts established by definitions local to the
                # iteration remain available inside the body.
                # Derive effects here rather than trusting a caller-supplied
                # loop inventory. Include nested writes and the induction
                # variable; local definitions re-establish facts when reached.
                assigned = assigned_variables(body) | {var.name}
                self._forget(assigned)
                for inner in body:
                    self.exec_stmt(inner)
                self._forget(assigned)
            case If(then_body=then_body, else_body=else_body):
                if self.assume_then_branches:
                    # This mode is used only after an external proof has
                    # established the selected output lies on the translated
                    # non-early-return path. Keep the dependency explicit in
                    # the audit trail instead of silently choosing a branch.
                    self.used_assumptions.add("all_if_conditions_true")
                    for inner in then_body:
                        self.exec_stmt(inner)
                    return

                # Default: analyze both branches from the same pre-state and
                # retain only lower-bound facts true after either branch.
                # Taking one branch unconditionally is unsound for a general
                # trusted analysis API.
                then_analyzer = PositionalAnalyzer(
                    assumptions=self.assumptions,
                    type_env=self.type_env,
                    # The current accumulator state is already copied in env;
                    # re-running __post_init__ for them would reset it.
                    accumulators=frozenset(),
                    assume_then_branches=False,
                    env=dict(self.env),
                    used_assumptions=set(self.used_assumptions),
                    elem_body=dict(self.elem_body),
                    before_stmt=self.before_stmt,
                    after_stmt=self.after_stmt,
                    before_stmt_elem_body=self.before_stmt_elem_body,
                    after_stmt_elem_body=self.after_stmt_elem_body,
                )
                else_analyzer = PositionalAnalyzer(
                    assumptions=self.assumptions,
                    type_env=self.type_env,
                    accumulators=frozenset(),
                    assume_then_branches=False,
                    env=dict(self.env),
                    used_assumptions=set(self.used_assumptions),
                    elem_body=dict(self.elem_body),
                    before_stmt=self.before_stmt,
                    after_stmt=self.after_stmt,
                    before_stmt_elem_body=self.before_stmt_elem_body,
                    after_stmt_elem_body=self.after_stmt_elem_body,
                )
                for inner in then_body:
                    then_analyzer.exec_stmt(inner)
                for inner in else_body:
                    else_analyzer.exec_stmt(inner)

                merged_env: dict[str, TensorFacts] = {}
                for name in then_analyzer.env.keys() | else_analyzer.env.keys():
                    ty = self.type_env.get(name)
                    rank = len(ty.dims) if isinstance(ty, TensorType) else 0
                    left = then_analyzer.env.get(name, facts_unknown(rank))
                    right = else_analyzer.env.get(name, facts_unknown(rank))
                    assert left.rank == right.rank
                    merged_env[name] = TensorFacts(
                        rank=left.rank,
                        zero_where=pred_and(left.zero_where, right.zero_where),
                        neg_inf_where=pred_and(
                            left.neg_inf_where, right.neg_inf_where
                        ),
                        one_where=pred_and(left.one_where, right.one_where),
                        true_where=pred_and(left.true_where, right.true_where),
                        false_where=pred_and(left.false_where, right.false_where),
                        unchanged_from=(
                            left.unchanged_from
                            if left.unchanged_from == right.unchanged_from
                            else None
                        ),
                    )
                self.env = merged_env
                self.elem_body = {
                    name: body
                    for name, body in then_analyzer.elem_body.items()
                    if else_analyzer.elem_body.get(name) == body
                }
                self.used_assumptions = (
                    then_analyzer.used_assumptions
                    | else_analyzer.used_assumptions
                )
            case MaskedStore(base=base):
                self._forget((base.name,))
            case _:
                raise ValueError(f"unsupported forward fact statement: {stmt!r}")

    def _bind(self, name: str, value: Expr) -> None:
        # Lets are single-assignment; treat like an unconditional assign.
        self._assign(name, None, value)

    def _assign(self, name: str, op: str | None, value: Expr) -> None:
        rank = self._rank(value)
        if op is not None:
            target_type = self.type_env.get(name)
            target_rank = (
                len(target_type.dims) if isinstance(target_type, TensorType)
                else self.env[name].rank if name in self.env else rank
            )
            if target_rank != rank:
                # Compound broadcast updates write the whole target. Until
                # this rule transports RHS facts into target coordinates,
                # forget them rather than mixing ranks or retaining old facts.
                self.env[name] = facts_unknown(target_rank)
                self.elem_body.pop(name, None)
                return
        new_facts, new_body = self._eval(value, rank)

        if op is None:
            facts = new_facts
            body = new_body
        elif op == "and":
            prev = self.env.get(name, facts_unknown(rank))
            prev_body = self.elem_body.get(name)
            # Boolean AND: true where BOTH true; false where EITHER false.
            facts = TensorFacts(
                rank=rank,
                zero_where=pred_false(rank),
                neg_inf_where=pred_false(rank),
                one_where=pred_false(rank),
                true_where=pred_and(prev.true_where, new_facts.true_where),
                false_where=pred_or(prev.false_where, new_facts.false_where),
                unchanged_from=None,
            )
            if prev_body is not None and new_body is not None:
                body = BinOp("and", prev_body, new_body)
            else:
                body = None
        elif op in ("+", "*"):
            # Numeric compound assign. Check finiteness of the
            # accumulator's PRE-update value via the target name
            # (semantically: "before this statement, name had a finite
            # value"). For `*` we don't propagate -inf.
            prev = self.env.get(name, facts_unknown(rank))
            prev_finite = name in self.assumptions.finite_vars
            if prev_finite:
                self.used_assumptions.add(f"finite({name})@{name}{op}=")
            if op == "+":
                zw = pred_and(prev.zero_where, new_facts.zero_where)
                niw = pred_false(rank)
                if prev_finite:
                    niw = pred_or(niw, new_facts.neg_inf_where)
                # Propagating -inf from the old value also requires the RHS
                # to be finite.  `neg_inf_where == FALSE` means only "we did
                # not prove -inf anywhere"; it is not evidence that the RHS
                # excludes +inf or NaN.
                if self._is_finite(value, f"{name}+=rhs"):
                    niw = pred_or(niw, prev.neg_inf_where)
                # Numerical zero is not a bitwise additive identity:
                # finite(-0) + (+0) is +0. A sign-aware rule would need
                # stronger facts than zero_where and finiteness.
                unchanged = None
                facts = TensorFacts(
                    rank=rank,
                    zero_where=zw,
                    neg_inf_where=niw,
                    one_where=pred_false(rank),
                    true_where=pred_false(rank),
                    false_where=pred_false(rank),
                    unchanged_from=unchanged,
                )
                body = None
            else:  # "*"
                # A zero operand yields numerical zero only when the other
                # operand is finite: 0*NaN and 0*inf are NaN.
                zw = pred_false(rank)
                if self._is_finite(value, f"{name}*=rhs"):
                    zw = pred_or(zw, prev.zero_where)
                if prev_finite:
                    zw = pred_or(zw, new_facts.zero_where)
                # Identity for *=: target is unchanged when the new
                # multiplier is whole-tensor one.
                if (
                    _is_true(new_facts.one_where)
                    and prev.unchanged_from is not None
                    and prev_finite
                ):
                    unchanged = prev.unchanged_from
                else:
                    unchanged = None
                facts = TensorFacts(
                    rank=rank,
                    zero_where=zw,
                    neg_inf_where=pred_false(rank),
                    one_where=pred_false(rank),
                    true_where=pred_false(rank),
                    false_where=pred_false(rank),
                    unchanged_from=unchanged,
                )
                body = None
        else:
            facts = facts_unknown(rank)
            body = None

        # Oracle override for all_false_masks: treat the variable as a
        # boolean tensor that is identically false at every position.
        if name in self.assumptions.all_false_masks:
            facts = facts_all_false(rank)
            body = BoolLit(False)
            self.used_assumptions.add(f"all_false_mask({name})")

        self.env[name] = facts
        if body is not None:
            self.elem_body[name] = body
        elif name in self.elem_body:
            del self.elem_body[name]

    # -- expression evaluation: returns (facts, optional_elem_body) ------

    def _eval(self, expr: Expr, rank: int) -> tuple[TensorFacts, Expr | None]:
        """Evaluate an expression to a TensorFacts, and if possible an
        exact element-body expression over free indices.
        """
        match expr:
            case IntLit(value=v):
                body = IntLit(v)
                f = facts_unknown(rank)
                if v == 0:
                    f = facts_all_zero(rank)
                return f, body
            case FloatLit(value=v):
                import math

                body = FloatLit(v)
                if v == 0.0:
                    return facts_all_zero(rank), body
                if v == float("-inf"):
                    return facts_all_neg_inf(rank), body
                return facts_unknown(rank), body
            case BoolLit(value=v):
                body = BoolLit(v)
                if v:
                    return facts_bool_exact(rank, BoolLit(True)), body
                return facts_all_false(rank), body
            case Var(name=n):
                # Use the variable's recorded facts, if any. For bool
                # tensors with an elem_body, pick that up too.
                if n in self.assumptions.all_false_masks:
                    self.used_assumptions.add(f"all_false_mask({n})")
                facts = self.env.get(n, facts_unknown(rank))
                body = self.elem_body.get(n)
                if body is None:
                    # Default symbolic body: for scalars, the var itself;
                    # for tensors, an element lookup indexed by the
                    # canonical free indices. This keeps element-bodies
                    # composable when a var is read without having been
                    # analyzed (e.g., input parameters, loop vars).
                    if rank == 0:
                        body = Var(n)
                    else:
                        body = TensorIndex(
                            Var(n), [free_idx(k) for k in range(rank)]
                        )
                return facts, body
            case Zeros():
                return facts_all_zero(rank), FloatLit(0.0)
            case Full(value=vexpr):
                inner_facts, inner_body = self._eval(vexpr, 0)
                if _is_true(inner_facts.zero_where):
                    return facts_all_zero(rank), inner_body
                if _is_true(inner_facts.neg_inf_where):
                    return facts_all_neg_inf(rank), inner_body
                return facts_unknown(rank), inner_body
            case BoolLit(value=v):
                return facts_bool_exact(rank, BoolLit(v)), BoolLit(v)
            case Arange():
                # Element value is _i0 itself, but we don't track
                # positive/negative-integer facts in this pass.
                return facts_unknown(rank), free_idx(0)
            case Where(cond=c, on_true=t, on_false=f):
                return self._eval_where(c, t, f, rank)
            case BinOp(op=op, lhs=l, rhs=r):
                return self._eval_binop(op, l, r, rank)
            case Not(value=v):
                inner, inner_body = self._eval(v, rank)
                facts = TensorFacts(
                    rank=rank,
                    zero_where=pred_false(rank),
                    neg_inf_where=pred_false(rank),
                    one_where=pred_false(rank),
                    true_where=inner.false_where,
                    false_where=inner.true_where,
                    unchanged_from=None,
                )
                body = Not(inner_body) if inner_body is not None else None
                return facts, body
            case Maximum(lhs=l, rhs=r):
                # Maximum(x, -inf) = x is an IEEE 754 identity requiring x to be
                # NaN-free. Per-position: at positions where one operand is
                # provably -inf, the result is the other operand at those
                # positions; preserves whole-tensor `unchanged_from` if the
                # other side is whole-tensor unchanged.
                lf, lb = self._eval(l, rank)
                rf, rb = self._eval(r, rank)
                # Whole-tensor: r all -inf → result = lf (assumes l finite).
                if (
                    _is_true(rf.neg_inf_where)
                    and self._is_finite(l, "Max(x,-inf)")
                ):
                    return lf, lb
                if (
                    _is_true(lf.neg_inf_where)
                    and self._is_finite(r, "Max(-inf,x)")
                ):
                    return rf, rb
                return facts_unknown(rank), None
            case Exp2(value=v):
                inner, _ = self._eval(v, rank)
                # Exp2 algebraic identities:
                #   Exp2(-inf) = 0  (positions where inner is -inf)
                #   Exp2(0)    = 1  (positions where inner is 0)
                # On finite inputs Exp2 is nonnegative and never NaN.
                return TensorFacts(
                    rank=rank,
                    zero_where=inner.neg_inf_where,
                    neg_inf_where=pred_false(rank),
                    one_where=inner.zero_where,
                    true_where=pred_false(rank),
                    false_where=pred_false(rank),
                    unchanged_from=None,
                ), None
            case Cast(value=v):
                inner, _ = self._eval(v, rank)
                # IEEE casts preserve the exact sentinels used by mask
                # analysis (0, 1, and -inf), but not arbitrary values or
                # finiteness under narrowing.  These identities are exported
                # as deployment-time backend obligations.
                return TensorFacts(
                    rank=rank,
                    zero_where=inner.zero_where,
                    neg_inf_where=inner.neg_inf_where,
                    one_where=inner.one_where,
                    true_where=inner.true_where,
                    false_where=inner.false_where,
                    unchanged_from=None,
                ), None
            case Log2():
                # No numerical log2 identity is needed by the structural
                # proof.  Retain only its pointwise dependency elsewhere.
                return facts_unknown(rank), None
            case ReduceMax(value=v, axis=axis):
                # Whole-tensor lifting: ReduceMax of an everywhere-c
                # tensor is c at every output position only when the reduced
                # axis is known non-empty.  Symbolic extents are not enough:
                # the IR type system itself does not prove positivity. We support
                # the three constants that arise in the kernels and in
                # downstream proofs: -inf, 0, 1. Partial-position facts
                # would require per-axis tracking, which is unsupported.
                assert isinstance(v.type, TensorType)
                reduced_extent = v.type.dims[axis]
                if not (
                    isinstance(reduced_extent, IntLit)
                    and reduced_extent.value > 0
                ):
                    return facts_unknown(rank), None
                inner, _ = self._eval(v, self._rank(v))
                if isinstance(v.type.elem_type, BoolType):
                    # Triton promotes bool to int32 for tl.max.  On a known
                    # all-false/all-true nonempty axis the numeric result is
                    # therefore exactly 0/1.  Partial boolean information is
                    # deliberately not reduced into an existential predicate.
                    if _is_true(inner.false_where):
                        return facts_all_zero(rank), IntLit(0)
                    if _is_true(inner.true_where):
                        return facts_all_one(rank), IntLit(1)
                    return facts_unknown(rank), None
                if _is_true(inner.neg_inf_where):
                    return facts_all_neg_inf(rank), None
                if _is_true(inner.zero_where):
                    return facts_all_zero(rank), None
                if _is_true(inner.one_where):
                    return facts_all_one(rank), None
                return facts_unknown(rank), None
            case ReduceSum(value=v, axis=axis):
                # Whole-tensor lifting: sum of zeros (any non-empty axis)
                # is zero. Sum of -inf is -inf when the rest are finite,
                # but tracking that would need per-axis information; we
                # keep it conservative.
                assert isinstance(v.type, TensorType)
                reduced_extent = v.type.dims[axis]
                if not (
                    isinstance(reduced_extent, IntLit)
                    and reduced_extent.value > 0
                ):
                    return facts_unknown(rank), None
                inner, _ = self._eval(v, self._rank(v))
                if _is_true(inner.zero_where):
                    return facts_all_zero(rank), None
                return facts_unknown(rank), None
            case Unsqueeze(value=v, axis=axis):
                inner, inner_body = self._eval(v, self._rank(v))
                return self._reindex_unsqueeze(inner, rank, axis), (
                    self._reindex_body_unsqueeze(inner_body, rank, axis)
                    if inner_body is not None
                    else None
                )
            case Squeeze(value=v, axis=axis):
                inner, inner_body = self._eval(v, self._rank(v))
                return self._reindex_squeeze(inner, rank, axis), (
                    self._reindex_body_squeeze(inner_body, rank, axis)
                    if inner_body is not None
                    else None
                )
            case BroadcastTo(value=v):
                # Same rank on both sides in our IR (BroadcastTo doesn't
                # add dims here; Unsqueeze does). It only expands size-1
                # dims.  A broadcast output index on such an axis reads source
                # index 0, so predicates must substitute `_ik -> 0`.
                inner, inner_body = self._eval(v, self._rank(v))
                if inner.rank == rank:
                    assert isinstance(v.type, TensorType)
                    mapping = {
                        f"{FREE_IDX_PREFIX}{k}": IntLit(0)
                        for k, dim in enumerate(v.type.dims)
                        if dim == IntLit(1)
                    }
                    if not mapping:
                        return inner, inner_body
                    facts = TensorFacts(
                        rank=rank,
                        zero_where=Pred(
                            rank,
                            subst_free_indices(inner.zero_where.body, mapping),
                        ),
                        neg_inf_where=Pred(
                            rank,
                            subst_free_indices(inner.neg_inf_where.body, mapping),
                        ),
                        one_where=Pred(
                            rank,
                            subst_free_indices(inner.one_where.body, mapping),
                        ),
                        true_where=Pred(
                            rank,
                            subst_free_indices(inner.true_where.body, mapping),
                        ),
                        false_where=Pred(
                            rank,
                            subst_free_indices(inner.false_where.body, mapping),
                        ),
                        # Broadcasting a size-1 axis changes the coordinate
                        # map.  A bare provenance name cannot express that
                        # map, so it must not certify pointwise state identity.
                        unchanged_from=None,
                    )
                    body = (
                        subst_free_indices(inner_body, mapping)
                        if inner_body is not None
                        else None
                    )
                    return facts, body
                return facts_unknown(rank), None
            case Transpose(value=v, permutation=perm):
                inner, inner_body = self._eval(v, self._rank(v))
                return self._reindex_transpose(inner, perm), (
                    self._reindex_body_transpose(inner_body, perm)
                    if inner_body is not None
                    else None
                )
            case TensorIndex(base=b, indices=idxs):
                # Reading a specific element of a tensor. If we have the
                # base's elem_body, substitute the index expressions in;
                # otherwise keep the indexing as a symbolic read against
                # the input tensor. In both cases `rank == 0` because
                # TensorIndex fully indexes out.
                base_body = self.elem_body.get(b.name)
                if base_body is None:
                    # Input tensor: leave as TensorIndex(Var(b), idxs).
                    # This preserves compositionality for downstream
                    # element-body builders.
                    return facts_unknown(rank), TensorIndex(Var(b.name), idxs)
                mapping: dict[str, Expr] = {}
                for k, idx in enumerate(idxs):
                    mapping[f"{FREE_IDX_PREFIX}{k}"] = idx
                substituted = subst_free_indices(base_body, mapping)
                return facts_unknown(rank), substituted
            case MaskedLoad():
                # Black box: we don't model what's in memory.
                return facts_unknown(rank), None
            case _:
                return facts_unknown(rank), None

    # -- Where ------------------------------------------------------------

    def _eval_where(
        self, c: Expr, t: Expr, f: Expr, rank: int
    ) -> tuple[TensorFacts, Expr | None]:
        cf, cb = self._eval(c, rank)

        # A condition known at every output position selects one branch
        # exactly.  Besides being more precise than merging positional
        # facts below, this is what preserves whole-tensor provenance such
        # as ``unchanged_from`` through an explicit state guard.  Requiring
        # a universal (BoolLit true) predicate is important: a merely
        # position-dependent condition cannot justify whole-tensor equality.
        if _is_true(cf.true_where):
            return self._eval(t, rank)
        if _is_true(cf.false_where):
            return self._eval(f, rank)

        tf, tb = self._eval(t, rank)
        ff, fb = self._eval(f, rank)

        # Positional zero: (cond is true AND t is zero) OR (cond is false AND f is zero).
        zero = pred_or(
            pred_and(cf.true_where, tf.zero_where),
            pred_and(cf.false_where, ff.zero_where),
        )
        # Positional neg_inf: same structure.
        neg_inf = pred_or(
            pred_and(cf.true_where, tf.neg_inf_where),
            pred_and(cf.false_where, ff.neg_inf_where),
        )
        # Positional one: same structure.
        one = pred_or(
            pred_and(cf.true_where, tf.one_where),
            pred_and(cf.false_where, ff.one_where),
        )
        facts = TensorFacts(
            rank=rank,
            zero_where=zero,
            neg_inf_where=neg_inf,
            one_where=one,
            true_where=pred_false(rank),
            false_where=pred_false(rank),
            unchanged_from=None,
        )
        # Represent "if cond then t else f" as a
        # Where node in IR for reuse by downstream consumers.
        if cb is not None and tb is not None and fb is not None:
            body = Where(cb, tb, fb)
        else:
            body = None
        return facts, body

    # -- BinOp ------------------------------------------------------------

    def _eval_binop(
        self, op: str, l: Expr, r: Expr, rank: int
    ) -> tuple[TensorFacts, Expr | None]:
        lf, lb = self._eval(l, rank if _rank_of(l) == rank else _rank_of(l))
        rf, rb = self._eval(r, rank if _rank_of(r) == rank else _rank_of(r))

        # For predicates to compose, we only trust them when both sides
        # share `rank`. Mixed-rank cases (scalar × tensor) still give
        # meaningful facts when the scalar side lifts trivially, handled
        # case-by-case below.
        same_rank = lf.rank == rank and rf.rank == rank
        scalar_rhs = rf.rank == 0 and lf.rank == rank
        scalar_lhs = lf.rank == 0 and rf.rank == rank

        # Helper: lift a rank-0 predicate to `rank`.
        def lift(p: Pred) -> Pred:
            return Pred(rank, p.body) if p.rank == 0 else p

        def lift_facts(f: TensorFacts) -> TensorFacts:
            if f.rank == 0:
                return TensorFacts(
                    rank=rank,
                    zero_where=Pred(rank, f.zero_where.body),
                    neg_inf_where=Pred(rank, f.neg_inf_where.body),
                    one_where=Pred(rank, f.one_where.body),
                    true_where=Pred(rank, f.true_where.body),
                    false_where=Pred(rank, f.false_where.body),
                    # Scalar-to-tensor lifting changes coordinates and cannot
                    # preserve the whole-state identity represented by a bare
                    # provenance name.
                    unchanged_from=None,
                )
            return f

        lf2 = lift_facts(lf) if scalar_lhs else lf
        rf2 = lift_facts(rf) if scalar_rhs else rf
        can_combine = same_rank or scalar_rhs or scalar_lhs

        body: Expr | None = None
        if lb is not None and rb is not None:
            body = BinOp(op, lb, rb)

        if not can_combine:
            return facts_unknown(rank), body

        if op == "and":
            return TensorFacts(
                rank=rank,
                zero_where=pred_false(rank),
                neg_inf_where=pred_false(rank),
                one_where=pred_false(rank),
                true_where=pred_and(lf2.true_where, rf2.true_where),
                false_where=pred_or(lf2.false_where, rf2.false_where),
                unchanged_from=None,
            ), body
        if op == "or":
            return TensorFacts(
                rank=rank,
                zero_where=pred_false(rank),
                neg_inf_where=pred_false(rank),
                one_where=pred_false(rank),
                true_where=pred_or(lf2.true_where, rf2.true_where),
                false_where=pred_and(lf2.false_where, rf2.false_where),
                unchanged_from=None,
            ), body
        if op in ("<", "<=", ">", ">=", "==", "!="):
            # Z3 predicates in the masked-region pass use mathematical
            # Int/Bool semantics and have no NaN value.  Treating a floating
            # comparison as an exact predicate would therefore be unsound for
            # NaN inputs (and could later narrow a value dependency).  The
            # attention masks we certify compare integer positions, so keep
            # exact comparison facts only on discrete operands.  A future
            # floating comparison rule must carry an explicit no-NaN
            # assumption through the audit trail.
            def discrete(expr: Expr) -> bool:
                typ = expr.type
                if isinstance(typ, TensorType):
                    typ = typ.elem_type
                return isinstance(typ, (IntType, BoolType))

            if not (discrete(l) and discrete(r)):
                # Also discard elem_body: retaining it could make a later
                # boolean equality reconstruct an "exact" predicate through
                # this unsupported comparison.
                return facts_unknown(rank), None
            if isinstance(lb, (IntLit, BoolLit)) and isinstance(
                rb, (IntLit, BoolLit)
            ):
                lv = lb.value
                rv = rb.value
                result = {
                    "<": lv < rv,
                    "<=": lv <= rv,
                    ">": lv > rv,
                    ">=": lv >= rv,
                    "==": lv == rv,
                    "!=": lv != rv,
                }[op]
                exact = BoolLit(result)
                return facts_bool_exact(rank, exact), exact
            if body is not None:
                return facts_bool_exact(rank, body), body
            return facts_unknown(rank), None
        if op == "+":
            zw = pred_and(lf2.zero_where, rf2.zero_where)
            # IEEE 754: (-inf) + finite = -inf, but (-inf) + (+inf) = NaN.
            # Propagate -inf only when `_is_finite` proves the other operand
            # excludes NaN and ±inf. `_is_false(neg_inf_where)` is insufficient:
            # it means no -inf positions were proved, not that none exist.
            niw = pred_false(rank)
            if self._is_finite(r, "(-inf)+x"):
                niw = pred_or(niw, lf2.neg_inf_where)
            if self._is_finite(l, "x+(-inf)"):
                niw = pred_or(niw, rf2.neg_inf_where)
            # Adding either sign of zero need not preserve the bit pattern
            # of another zero, even when every operand is finite.
            return TensorFacts(
                rank=rank,
                zero_where=zw,
                neg_inf_where=niw,
                one_where=pred_false(rank),
                true_where=pred_false(rank),
                false_where=pred_false(rank),
                unchanged_from=None,
            ), body
        if op == "-":
            # x - x = 0 requires the same whole-tensor unchanged_from name
            # and finiteness, since -inf-(-inf) is NaN. This supports
            # alpha = exp2(0) = 1 in fully masked iterations.
            zw = pred_false(rank)
            if (
                lf2.unchanged_from is not None
                and lf2.unchanged_from == rf2.unchanged_from
            ):
                # Need finiteness of the unchanged variable so the
                # subtraction doesn't produce NaN.
                if lf2.unchanged_from in self.assumptions.finite_vars:
                    self.used_assumptions.add(
                        f"finite({lf2.unchanged_from})@x-x"
                    )
                    zw = pred_true(rank)
            # (-inf) - finite = -inf; same finiteness gate as `+`.
            # `_is_false(neg_inf_where)` is not a sound substitute (it
            # only says "no proof of -inf", not "proof of not -inf").
            niw = pred_false(rank)
            if self._is_finite(r, "(-inf)-x"):
                niw = pred_or(niw, lf2.neg_inf_where)
            return TensorFacts(
                rank=rank,
                zero_where=zw,
                neg_inf_where=niw,
                one_where=pred_false(rank),
                true_where=pred_false(rank),
                false_where=pred_false(rank),
                unchanged_from=None,
            ), body
        if op == "*":
            zw = pred_false(rank)
            if self._is_finite(r, "0*x"):
                zw = pred_or(zw, lf2.zero_where)
            if self._is_finite(l, "x*0"):
                zw = pred_or(zw, rf2.zero_where)
            niw = pred_false(rank)
            if self._is_positive(r, "(-inf)*x"):
                niw = pred_or(niw, lf2.neg_inf_where)
            if self._is_positive(l, "x*(-inf)"):
                niw = pred_or(niw, rf2.neg_inf_where)
            # x * 1 = x preserves unchanged_from when one side is
            # whole-tensor one.
            unchanged: str | None = None
            if (
                _is_true(rf2.one_where)
                and lf2.unchanged_from is not None
                and self._is_named_finite(
                    lf2.unchanged_from, "x*1 identity"
                )
            ):
                unchanged = lf2.unchanged_from
            elif (
                _is_true(lf2.one_where)
                and rf2.unchanged_from is not None
                and self._is_named_finite(
                    rf2.unchanged_from, "1*x identity"
                )
            ):
                unchanged = rf2.unchanged_from
            return TensorFacts(
                rank=rank,
                zero_where=zw,
                neg_inf_where=niw,
                one_where=pred_false(rank),
                true_where=pred_false(rank),
                false_where=pred_false(rank),
                unchanged_from=unchanged,
            ), body
        if op == "@":
            # 0 @ x = 0 (and x @ 0 = 0): whole-tensor zero absorbs in
            # matmul, given the other side is finite (a single ±inf could
            # produce NaN via 0*inf). For partial zero info we'd need
            # per-row/column tracking, which is unsupported.
            if _is_true(lf2.zero_where) and self._is_finite(r, "0@x"):
                return facts_all_zero(rank), None
            if _is_true(rf2.zero_where) and self._is_finite(l, "x@0"):
                return facts_all_zero(rank), None
            return facts_unknown(rank), None
        return facts_unknown(rank), body

    # -- reindex helpers --------------------------------------------------

    def _reindex_unsqueeze(
        self, inner: TensorFacts, out_rank: int, axis: int
    ) -> TensorFacts:
        # Inner rank r → outer rank r+1. Free-index _i{k} in inner maps
        # to _i{k if k < axis else k+1} in outer.
        mapping = {}
        for k in range(inner.rank):
            out_k = k if k < axis else k + 1
            mapping[f"{FREE_IDX_PREFIX}{k}"] = free_idx(out_k)
        return TensorFacts(
            rank=out_rank,
            zero_where=Pred(out_rank, subst_free_indices(inner.zero_where.body, mapping)),
            neg_inf_where=Pred(
                out_rank, subst_free_indices(inner.neg_inf_where.body, mapping)
            ),
            one_where=Pred(out_rank, subst_free_indices(inner.one_where.body, mapping)),
            true_where=Pred(out_rank, subst_free_indices(inner.true_where.body, mapping)),
            false_where=Pred(
                out_rank, subst_free_indices(inner.false_where.body, mapping)
            ),
            # The provenance domain has a different rank.  Until provenance
            # carries an explicit coordinate map, this is not an identity.
            unchanged_from=None,
        )

    def _reindex_body_unsqueeze(
        self, body: Expr, out_rank: int, axis: int
    ) -> Expr:
        mapping = {}
        for k in range(out_rank - 1):
            out_k = k if k < axis else k + 1
            mapping[f"{FREE_IDX_PREFIX}{k}"] = free_idx(out_k)
        return subst_free_indices(body, mapping)

    def _reindex_squeeze(
        self, inner: TensorFacts, out_rank: int, axis: int
    ) -> TensorFacts:
        # Inner r+1 → outer r. Inner's _i{axis} is dropped (assumed size
        # 1, so no quantification). Inner _i{k} for k > axis maps to
        # _i{k-1}.
        mapping: dict[str, Expr] = {f"{FREE_IDX_PREFIX}{axis}": IntLit(0)}
        for k in range(inner.rank):
            if k == axis:
                continue
            out_k = k if k < axis else k - 1
            mapping[f"{FREE_IDX_PREFIX}{k}"] = free_idx(out_k)
        return TensorFacts(
            rank=out_rank,
            zero_where=Pred(out_rank, subst_free_indices(inner.zero_where.body, mapping)),
            neg_inf_where=Pred(
                out_rank, subst_free_indices(inner.neg_inf_where.body, mapping)
            ),
            one_where=Pred(out_rank, subst_free_indices(inner.one_where.body, mapping)),
            true_where=Pred(out_rank, subst_free_indices(inner.true_where.body, mapping)),
            false_where=Pred(
                out_rank, subst_free_indices(inner.false_where.body, mapping)
            ),
            unchanged_from=None,
        )

    def _reindex_body_squeeze(self, body: Expr, out_rank: int, axis: int) -> Expr:
        mapping: dict[str, Expr] = {f"{FREE_IDX_PREFIX}{axis}": IntLit(0)}
        for k in range(out_rank + 1):
            if k == axis:
                continue
            out_k = k if k < axis else k - 1
            mapping[f"{FREE_IDX_PREFIX}{k}"] = free_idx(out_k)
        return subst_free_indices(body, mapping)

    def _reindex_transpose(self, inner: TensorFacts, perm) -> TensorFacts:
        # Outer _i{k} came from inner _i{perm[k]}; so inner _i{j} maps to
        # outer _i{inv_perm[j]}.
        inv = [0] * len(perm)
        for k, p in enumerate(perm):
            inv[p] = k
        mapping = {f"{FREE_IDX_PREFIX}{j}": free_idx(inv[j]) for j in range(len(perm))}
        return TensorFacts(
            rank=inner.rank,
            zero_where=Pred(
                inner.rank, subst_free_indices(inner.zero_where.body, mapping)
            ),
            neg_inf_where=Pred(
                inner.rank, subst_free_indices(inner.neg_inf_where.body, mapping)
            ),
            one_where=Pred(
                inner.rank, subst_free_indices(inner.one_where.body, mapping)
            ),
            true_where=Pred(
                inner.rank, subst_free_indices(inner.true_where.body, mapping)
            ),
            false_where=Pred(
                inner.rank, subst_free_indices(inner.false_where.body, mapping)
            ),
            unchanged_from=(
                inner.unchanged_from
                if tuple(perm) == tuple(range(inner.rank))
                else None
            ),
        )

    def _reindex_body_transpose(self, body: Expr, perm) -> Expr:
        inv = [0] * len(perm)
        for k, p in enumerate(perm):
            inv[p] = k
        mapping = {f"{FREE_IDX_PREFIX}{j}": free_idx(inv[j]) for j in range(len(perm))}
        return subst_free_indices(body, mapping)


# ---------------------------------------------------------------------------
# Public driver: accumulator-neutrality lemma
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NeutralityReport:
    """Outcome of `check_iteration_neutral`.

    `proved` is True iff every accumulator's facts at end-of-body have
    `unchanged_from == acc_name`: the body provably leaves the
    accumulator equal, at every position, to its pre-iteration value.

    `failures` records (acc_name, final_TensorFacts) for accumulators
    that were not proved unchanged; the caller can inspect these for
    diagnostic purposes.

    `used_assumptions` is the audit trail of named assumptions actually
    relied on by rule firings during the analysis. Useful for: (a)
    discharging the obligations externally; (b) tightening over-declared
    assumption bundles.
    """

    proved: bool
    failures: list[tuple[str, "TensorFacts"]]
    used_assumptions: set[str]
    final_facts: dict[str, "TensorFacts"]


def check_iteration_neutral(
    loop_body: list[Stmt],
    accumulators: list[str],
    assumptions: NeutralityAssumptions = NeutralityAssumptions(),
) -> NeutralityReport:
    """Run the per-position analyzer over `loop_body` and report whether
    every named accumulator ends up provably unchanged from its
    pre-iteration value (i.e. `unchanged_from == acc_name`).

    The analyzer derives the `type_env` for accumulator ranks by walking
    the body and collecting `Var.type` annotations (set by `infer_types`).
    """
    type_env = collect_statement_type_env(loop_body)
    a = PositionalAnalyzer(
        assumptions=assumptions,
        type_env=type_env,
        accumulators=frozenset(accumulators),
    )
    for stmt in loop_body:
        a.exec_stmt(stmt)
    failures: list[tuple[str, TensorFacts]] = []
    for acc in accumulators:
        facts = a.env.get(acc)
        if facts is None or facts.unchanged_from != acc:
            failures.append((acc, facts if facts is not None else facts_unknown(0)))
    return NeutralityReport(
        proved=not failures,
        failures=failures,
        used_assumptions=a.used_assumptions,
        final_facts={
            acc: a.env.get(acc, facts_unknown(0)) for acc in accumulators
        },
    )


@dataclass(frozen=True)
class RowNeutralityReport:
    """Selected-row lifting of the whole-tensor neutrality lemma.

    The proof has two independently checked parts:

    1. replacing the named mask by an all-false tensor leaves every
       accumulator unchanged (`whole_report`); and
    2. a selected accumulator row depends on at most the corresponding mask
       row (`row_separation_failures` is empty).

    Therefore, whenever the selected mask row is all false, the selected
    accumulator row is unchanged even if other mask rows are effectful.
    This conclusion uses the same structural-to-value dependency
    meta-argument as the region verifier.
    """

    proved: bool
    whole_report: NeutralityReport
    row_separation_failures: list[str]
    used_assumptions: set[str]


def check_iteration_row_neutral(
    loop_body: list[Stmt],
    accumulators: list[str],
    mask_name: str,
    assumptions: NeutralityAssumptions = NeutralityAssumptions(),
) -> RowNeutralityReport:
    """Check selected-row all-false identity for an attention iteration.

    The identity and dependency reasoning is implemented by the generic
    regional transition checker. This adapter constructs a singleton
    first-axis demand and proves that every returned fact requirement is
    covered by the corresponding singleton first-axis mask region.
    """
    from .identity_transition import (
        AllFalseFact,
        AvailableRegionalFact,
        check_region_identity_transition,
        prove_fact_requirement_covered,
    )
    from .regions import GuardedRegion

    failures: list[str] = []
    type_env = collect_statement_type_env(loop_body)
    mask_type = type_env.get(mask_name)
    if not isinstance(mask_type, TensorType) or len(mask_type.dims) < 1:
        failures.append(f"{mask_name}: missing tensor type with a row axis")
    selected_row = Var("__selected_row", type=IntType())
    state_demands: dict[str, GuardedRegion] = {}
    for acc in accumulators:
        acc_type = type_env.get(acc)
        if not isinstance(acc_type, TensorType) or len(acc_type.dims) < 1:
            failures.append(f"{acc}: missing tensor type with a row axis")
            continue
        if (
            isinstance(mask_type, TensorType)
            and mask_type.dims
            and acc_type.dims[0] != mask_type.dims[0]
        ):
            failures.append(
                f"{acc}: row extent {acc_type.dims[0]} differs from mask "
                f"extent {mask_type.dims[0]}"
            )
            continue
        state_demands[acc] = GuardedRegion(
            [
                Slice(selected_row, add_(selected_row, IntLit(1))),
                *[Slice(IntLit(0), dim) for dim in acc_type.dims[1:]],
            ],
            pred_true(len(acc_type.dims)),
        )

    transition = check_region_identity_transition(
        loop_body,
        state_demands,
        AllFalseFact(mask_name),
        assumptions,
    )
    failures.extend(transition.failures)

    if isinstance(mask_type, TensorType) and len(mask_type.dims) >= 1:
        available = AvailableRegionalFact(
            AllFalseFact(mask_name),
            GuardedRegion(
                [
                    Slice(selected_row, add_(selected_row, IntLit(1))),
                    *[Slice(IntLit(0), dim) for dim in mask_type.dims[1:]],
                ],
                pred_true(len(mask_type.dims)),
            ),
        )
        premises = [
            BinOp(">=", selected_row, IntLit(0)),
            BinOp("<", selected_row, mask_type.dims[0]),
        ]
        for requirement in transition.requirements:
            check = prove_fact_requirement_covered(
                requirement,
                available,
                premises,
                check_name=f"{requirement.state}_{mask_name}_same_row",
            )
            if check.proved:
                continue
            failures.append(
                f"{requirement.state}: selected output row may depend on "
                f"another {mask_name} row: "
                f"{check.details}"
            )

    return RowNeutralityReport(
        proved=transition.proved and not failures,
        whole_report=transition.whole_report,
        row_separation_failures=failures,
        used_assumptions=set(transition.used_assumptions),
    )


def collect_statement_type_env(stmts: list[Stmt]) -> dict[str, Type]:
    """Walk `stmts` and harvest a {var_name -> Type} mapping from every
    Var that carries a `.type`. Used by `check_iteration_neutral` so the
    caller need not pass an explicit type_env.

    Recurses through dataclass children but skips the `type` field on
    Expr / Stmt nodes to avoid descending into Type definitions (which
    are themselves dataclasses but contain no Var references).
    """
    from dataclasses import fields, is_dataclass

    env: dict[str, Type] = {}

    def visit_any(x: object) -> None:
        if isinstance(x, Var) and x.type is not None:
            env.setdefault(x.name, x.type)
        # Regions and ranges are dataclasses too; variables used only in a
        # Slice bound (for example an attention head index) must still enter
        # the synthetic type environment used by the row-separation checker.
        if is_dataclass(x):
            for f in fields(x):
                if f.name == "type":
                    continue
                visit_any(getattr(x, f.name))
        elif isinstance(x, (list, tuple)):
            for item in x:
                visit_any(item)

    for s in stmts:
        visit_any(s)
    return env
