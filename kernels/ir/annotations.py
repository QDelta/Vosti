"""Typed annotation AST and parser for named proof goals.

Relational goals use left/right executions. Exact-effect goals instead use
before/after tensor states of one execution and may require dtype(x) == dtype(y).
These are ordinary terms/region references inside the same proposition AST;
each verifier explicitly rejects the other path's unsupported constructs.

Kernel interface metadata lives in the separate ``@params`` and ``@grid``
blocks parsed by :mod:`ir.translate`.  This module parses one or more named
``@verif`` proof goals from the same Triton kernel prologue:

    # @verif(batch_projection,
    #   same(N, K),
    #   pre(
    #   right(M) == 1,
    #   M > 0, N > 0, K > 0,
    #   x >= 0, x < left(M),
    #   forall(i, j, implies(i < j, left(cu)[i] < left(cu)[j])),
    #   left(a)[x:x+1, 0:K] == right(a)[0:1, 0:K],
    #   left(k)[...] == right(k)[...] given pi,
    # )
    #   ),
    #   post(
    #   left(c)[x:x+1, 0:N] == right(c)[0:1, 0:N]
    #   ),
    #   singleton(bi left=x right=0),
    # )

Scalar variables listed in ``same(...)`` are shared between left and right
(left(X) == right(X)). Tensor parameters in @same denote whole-tensor value
equality and should be reserved for genuinely shared immutable buffers.
Within ``pre(...)`` and ``post(...)``, they must be written as bare names (X), not left(X) or right(X).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Union
import ast
import re


# ---------------------------------------------------------------------------
# Annotation AST nodes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Left:
    """Reference to the left version of a kernel parameter: left(M)."""
    name: str


@dataclass(frozen=True)
class Right:
    """Reference to the right version of a kernel parameter: right(M)."""
    name: str


@dataclass(frozen=True)
class Before:
    """Tensor state before one execution, distinct from a relational side."""
    name: str


@dataclass(frozen=True)
class After:
    """Tensor state after one execution, distinct from a relational side."""
    name: str


@dataclass(frozen=True)
class DTypeOf:
    """The physical element dtype of a tensor parameter (not an integer)."""
    name: str


@dataclass(frozen=True)
class FreeVar:
    """A free variable not specific to left/right: x."""
    name: str


@dataclass(frozen=True)
class IntConst:
    """Integer literal."""
    value: int


@dataclass(frozen=True)
class AnnBinOp:
    """Binary integer operation, including min/max for clipped regions."""
    op: str
    lhs: AnnExpr
    rhs: AnnExpr


@dataclass(frozen=True)
class AnnIndex:
    """Indexed access: left(cu_seqlens_q)[x] or left(cu_seqlens_q)[x+1]."""
    base: Left | Right | Before | After
    indices: list[AnnExpr]


AnnExpr = Union[Left, Right, Before, After, DTypeOf, FreeVar, IntConst, AnnBinOp, AnnIndex]


@dataclass(frozen=True)
class AnnSlice:
    """A slice in a region: start:stop."""
    start: AnnExpr
    stop: AnnExpr


@dataclass(frozen=True)
class RegionRef:
    """A tensor with region slices: left(a)[x:x+1, 0:K]."""
    side: Left | Right | Before | After
    slices: list[AnnSlice]


class Prop:
    """A proposition, distinct from scalar terms and tensor-region references."""


@dataclass(frozen=True)
class RegionEquiv(Prop):
    """Tensor region equality, between executions or before/after states.

    Optional given clause: left(t)[...] == right(t)[...] given expr
    The given expr is evaluated with both left_env and right_env;
    the condition is that the two evaluations are equal.
    """
    left: RegionRef
    right: RegionRef
    given: AnnExpr | None = None


@dataclass(frozen=True)
class AnnAnd(Prop):
    """Conjunction of propositions."""
    args: list[Prop]


@dataclass(frozen=True)
class AnnOr(Prop):
    """Disjunction; representable even when proof search does not support it."""
    args: list[Prop]


@dataclass(frozen=True)
class AnnNot(Prop):
    """Negation of a proposition."""
    body: Prop


@dataclass(frozen=True)
class AnnImplies(Prop):
    """Implication: implies(antecedent, consequent)."""
    antecedent: Prop
    consequent: Prop


@dataclass(frozen=True)
class AnnComparison(Prop):
    """Scalar comparison, at any proposition nesting depth."""
    op: str  # ==, >, >=, <, <=
    lhs: AnnExpr
    rhs: AnnExpr


@dataclass(frozen=True)
class ForAllConstraint(Prop):
    """Universal quantification: forall(i, j, implies(i < j, ...))."""
    vars: list[str]
    body: Prop


@dataclass(frozen=True)
class SingletonSpec:
    """Singleton annotation: @singleton(bi, left=x, right=0)."""
    var: str
    left: AnnExpr
    right: AnnExpr


@dataclass
class RelationalProofGoal:
    """One named theorem (historical class name shared by both proof paths)."""
    name: str
    pre_conditions: list[Prop]
    post_conditions: list[Prop]
    singletons: list[SingletonSpec] = field(default_factory=list)
    same_vars: set[str] = field(default_factory=set)


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

# Token types
TOK_IDENT = "IDENT"
TOK_INT = "INT"
TOK_LPAREN = "LPAREN"
TOK_RPAREN = "RPAREN"
TOK_LBRACKET = "LBRACKET"
TOK_RBRACKET = "RBRACKET"
TOK_COLON = "COLON"
TOK_COMMA = "COMMA"
TOK_PLUS = "PLUS"
TOK_MINUS = "MINUS"
TOK_STAR = "STAR"
TOK_PERCENT = "PERCENT"  # %
TOK_DSLASH = "DSLASH"  # //
TOK_EQ = "EQ"        # =
TOK_EQEQ = "EQEQ"   # ==
TOK_GT = "GT"         # >
TOK_GE = "GE"         # >=
TOK_LT = "LT"         # <
TOK_LE = "LE"         # <=
TOK_EOF = "EOF"


@dataclass
class Token:
    type: str
    value: str
    pos: int


_TOKEN_PATTERNS = [
    (r"==", TOK_EQEQ),
    (r">=", TOK_GE),
    (r"<=", TOK_LE),
    (r"//", TOK_DSLASH),
    (r"=", TOK_EQ),
    (r">", TOK_GT),
    (r"<", TOK_LT),
    (r"\(", TOK_LPAREN),
    (r"\)", TOK_RPAREN),
    (r"\[", TOK_LBRACKET),
    (r"\]", TOK_RBRACKET),
    (r":", TOK_COLON),
    (r",", TOK_COMMA),
    (r"\+", TOK_PLUS),
    (r"-", TOK_MINUS),
    (r"\*", TOK_STAR),
    (r"%", TOK_PERCENT),
    (r"[A-Za-z_]\w*", TOK_IDENT),
    (r"\d+", TOK_INT),
    (r"\s+", None),  # skip whitespace
]

_TOKEN_RE = re.compile("|".join(f"(?P<T{i}>{pat})" for i, (pat, _) in enumerate(_TOKEN_PATTERNS)))


def _tokenize(text: str) -> list[Token]:
    tokens: list[Token] = []
    pos = 0
    while pos < len(text):
        m = _TOKEN_RE.match(text, pos)
        if m is None:
            snippet = text[pos : pos + 20]
            raise ParseError(
                f"Unexpected character {text[pos]!r} at pos {pos}: {snippet!r}"
            )
        for i, (_, tok_type) in enumerate(_TOKEN_PATTERNS):
            if m.group(f"T{i}") is not None:
                if tok_type is not None:
                    tokens.append(Token(type=tok_type, value=m.group(), pos=m.start()))
                break
        pos = m.end()
    tokens.append(Token(type=TOK_EOF, value="", pos=len(text)))
    return tokens


# ---------------------------------------------------------------------------
# Recursive descent parser
# ---------------------------------------------------------------------------


class ParseError(ValueError):
    pass


class _Parser:
    def __init__(self, tokens: list[Token], same_vars: set[str] | None = None):
        self.tokens = tokens
        self.pos = 0
        self.same_vars: set[str] = same_vars or set()

    def peek(self) -> Token:
        return self.tokens[self.pos]

    def advance(self) -> Token:
        tok = self.tokens[self.pos]
        self.pos += 1
        return tok

    def expect(self, tok_type: str) -> Token:
        tok = self.advance()
        if tok.type != tok_type:
            raise ParseError(f"Expected {tok_type}, got {tok.type} ({tok.value!r}) at pos {tok.pos}")
        return tok

    def at(self, tok_type: str) -> bool:
        return self.peek().type == tok_type

    def at_any(self, *tok_types: str) -> bool:
        return self.peek().type in tok_types

    # --- Expression parsing ---

    def parse_atom(self) -> AnnExpr:
        """Parse atomic expression: left(...), right(...), integer, @same var, or free variable."""
        tok = self.peek()

        if tok.type == TOK_IDENT and tok.value == "dtype":
            self.advance()
            self.expect(TOK_LPAREN)
            name = self.expect(TOK_IDENT).value
            self.expect(TOK_RPAREN)
            return DTypeOf(name)

        if tok.type == TOK_IDENT and tok.value in ("left", "right", "before", "after"):
            side_name = tok.value
            self.advance()
            self.expect(TOK_LPAREN)
            name_tok = self.expect(TOK_IDENT)
            self.expect(TOK_RPAREN)

            # Reject left(X)/right(X) for @same variables
            if name_tok.value in self.same_vars:
                raise ParseError(
                    f"Variable '{name_tok.value}' is declared in @same and must be "
                    f"written as a bare name, not {side_name}({name_tok.value})"
                )

            side = {"left": Left, "right": Right, "before": Before, "after": After}[side_name](name_tok.value)

            # Check for indexing: left(cu_seqlens_q)[...]
            if self.at(TOK_LBRACKET):
                self.advance()
                indices: list[AnnExpr] = [self.parse_expr()]
                while self.at(TOK_COMMA):
                    self.advance()
                    indices.append(self.parse_expr())
                self.expect(TOK_RBRACKET)
                return AnnIndex(base=side, indices=indices)

            return side

        if tok.type == TOK_INT:
            self.advance()
            return IntConst(int(tok.value))

        if tok.type == TOK_IDENT:
            # Could be a function call like cdiv(a, b), a @same var, or a free variable
            self.advance()
            if self.at(TOK_LPAREN):
                # Function call
                func_name = tok.value
                self.advance()
                arg1 = self.parse_expr()
                self.expect(TOK_COMMA)
                arg2 = self.parse_expr()
                self.expect(TOK_RPAREN)
                if func_name not in ("cdiv", "add", "sub", "mul", "min", "max"):
                    raise ParseError(f"Unknown function: {func_name}")
                op = {"add": "+", "sub": "-", "mul": "*"}.get(func_name, func_name)
                return AnnBinOp(op=op, lhs=arg1, rhs=arg2)

            # @same variable: bare name → Left(name) (shared, so left == right)
            if tok.value in self.same_vars:
                node: Left | Right = Left(tok.value)
                # Handle optional indexing: cu_seqlens_q[...]
                if self.at(TOK_LBRACKET):
                    self.advance()
                    indices = [self.parse_expr()]
                    while self.at(TOK_COMMA):
                        self.advance()
                        indices.append(self.parse_expr())
                    self.expect(TOK_RBRACKET)
                    return AnnIndex(base=node, indices=indices)
                return node

            return FreeVar(tok.value)

        if tok.type == TOK_LPAREN:
            self.advance()
            expr = self.parse_expr()
            self.expect(TOK_RPAREN)
            return expr

        raise ParseError(f"Unexpected token {tok.type} ({tok.value!r}) at pos {tok.pos}")

    def parse_mul_expr(self) -> AnnExpr:
        """Parse multiplicative: atom ((* | % | //) atom)*."""
        left = self.parse_atom()
        while self.at_any(TOK_STAR, TOK_PERCENT, TOK_DSLASH):
            tok = self.advance()
            if tok.type == TOK_STAR:
                op = "*"
            elif tok.type == TOK_PERCENT:
                op = "%"
            else:
                op = "//"
            right = self.parse_atom()
            left = AnnBinOp(op, left, right)
        return left

    def parse_expr(self) -> AnnExpr:
        """Parse additive: mul_expr ((+|-) mul_expr)*."""
        left = self.parse_mul_expr()
        while self.at_any(TOK_PLUS, TOK_MINUS):
            op = "+" if self.advance().type == TOK_PLUS else "-"
            right = self.parse_mul_expr()
            left = AnnBinOp(op, left, right)
        return left

    # --- Region parsing ---

    def parse_slice(self) -> AnnSlice:
        """Parse start:stop."""
        start = self.parse_expr()
        self.expect(TOK_COLON)
        stop = self.parse_expr()
        return AnnSlice(start=start, stop=stop)

    def parse_region_ref(self, side: Left | Right | Before | After) -> RegionRef:
        """Parse [slice, slice, ...] after a left/right identifier.

        The side and opening bracket have already been identified by the caller;
        the bracket is consumed here.
        """
        self.expect(TOK_LBRACKET)
        slices: list[AnnSlice] = [self.parse_slice()]
        while self.at(TOK_COMMA):
            self.advance()
            slices.append(self.parse_slice())
        self.expect(TOK_RBRACKET)
        return RegionRef(side=side, slices=slices)

    # --- Proposition parsing ---

    def parse_comparison(self) -> AnnComparison:
        """Parse a scalar comparison; equality has only one spelling."""
        lhs = self.parse_expr()
        operators = {
            TOK_EQEQ: "==", TOK_GT: ">", TOK_GE: ">=",
            TOK_LT: "<", TOK_LE: "<=",
        }
        if self.at(TOK_EQ):
            raise ParseError("Use '==' for proposition equality; '=' is only for named bindings")
        if self.peek().type not in operators:
            raise ParseError(f"Expected comparison operator, got {self.peek().value!r}")
        op = operators[self.advance().type]
        rhs = self.parse_expr()
        return AnnComparison(op, lhs, rhs)

    def parse_prop(self) -> Prop:
        """Parse propositions independently of the supported proof fragment."""
        tok = self.peek()
        if tok.type == TOK_IDENT and tok.value in {"and", "or"}:
            self.advance()
            self.expect(TOK_LPAREN)
            args = [self.parse_prop()]
            while self.at(TOK_COMMA):
                self.advance()
                args.append(self.parse_prop())
            self.expect(TOK_RPAREN)
            return AnnAnd(args) if tok.value == "and" else AnnOr(args)
        if tok.type == TOK_IDENT and tok.value == "not":
            self.advance()
            self.expect(TOK_LPAREN)
            body = self.parse_prop()
            self.expect(TOK_RPAREN)
            return AnnNot(body)
        if tok.type == TOK_IDENT and tok.value == "implies":
            self.advance()
            self.expect(TOK_LPAREN)
            antecedent = self.parse_prop()
            self.expect(TOK_COMMA)
            consequent = self.parse_prop()
            self.expect(TOK_RPAREN)
            return AnnImplies(antecedent, consequent)
        if tok.type == TOK_IDENT and tok.value == "forall":
            return self.parse_forall()
        if tok.type == TOK_IDENT and tok.value == "forall_region":
            raise ParseError("forall_region was removed; use forall(i, implies(guard, region_equality))")
        if tok.type == TOK_IDENT and tok.value == "exists":
            raise ParseError("exists is not part of the annotation language")
        if self._is_region_start():
            left_ref = self._parse_side_region()
            if self.at(TOK_EQ):
                raise ParseError("Use '==' for proposition equality; '=' is only for named bindings")
            self.expect(TOK_EQEQ)
            if not self._is_region_start():
                raise ParseError("Region equality requires two tensor-region operands")
            right_ref = self._parse_side_region()
            given = None
            if self.at(TOK_IDENT) and self.peek().value == "given":
                self.advance()
                given = self.parse_expr()
            return RegionEquiv(left_ref, right_ref, given)
        return self.parse_comparison()

    def parse_forall(self) -> ForAllConstraint:
        """Parse forall(i, j, proposition), with integer binders."""
        self.expect(TOK_IDENT)
        self.expect(TOK_LPAREN)
        bound_vars = []
        reserved = {"and", "or", "not", "implies", "forall",
                    "left", "right", "before", "after", "dtype", "exists", "forall_region"}
        while self.at(TOK_IDENT) and self.peek().value not in reserved:
            saved = self.pos
            name = self.advance().value
            if not self.at(TOK_COMMA):
                self.pos = saved
                break
            self.advance()
            bound_vars.append(name)
        if not bound_vars or len(set(bound_vars)) != len(bound_vars):
            raise ParseError("forall requires unique bound variables")
        body = self.parse_prop()
        self.expect(TOK_RPAREN)
        return ForAllConstraint(bound_vars, body)

    def _is_region_start(self) -> bool:
        """Look ahead to check if current position starts a region ref: left/right(name)[start:stop, ...].

        Distinguishes from index expressions left(name)[expr] by checking
        for a colon inside the brackets (regions have slices, indices don't).
        """
        if not (self.at(TOK_IDENT) and self.peek().value in ("left", "right", "before", "after")):
            return False
        # Peek further: left ( name ) [ ... : ...
        saved = self.pos
        self.advance()  # left/right
        if not self.at(TOK_LPAREN):
            self.pos = saved
            return False
        self.advance()  # (
        if not self.at(TOK_IDENT):
            self.pos = saved
            return False
        self.advance()  # name
        if not self.at(TOK_RPAREN):
            self.pos = saved
            return False
        self.advance()  # )
        if not self.at(TOK_LBRACKET):
            self.pos = saved
            return False
        self.advance()  # [
        # Scan for a colon before the matching ']' to distinguish slice from index
        depth = 1
        has_colon = False
        while self.pos < len(self.tokens) and depth > 0:
            tok = self.advance()
            if tok.type == TOK_LBRACKET:
                depth += 1
            elif tok.type == TOK_RBRACKET:
                depth -= 1
            elif tok.type == TOK_COLON and depth == 1:
                has_colon = True
                break
        self.pos = saved
        return has_colon

    def _parse_side_region(self) -> RegionRef:
        """Parse left(name)[slices] or right(name)[slices]."""
        side_name = self.advance().value  # left or right
        self.expect(TOK_LPAREN)
        name = self.expect(TOK_IDENT).value
        self.expect(TOK_RPAREN)
        side = {"left": Left, "right": Right, "before": Before, "after": After}[side_name](name)
        return self.parse_region_ref(side)

    def parse_condition_list(self) -> list[Prop]:
        """Parse comma-separated conditions until EOF."""
        conditions: list[Prop] = []
        if self.at(TOK_EOF):
            return conditions
        conditions.append(self.parse_prop())
        while self.at(TOK_COMMA):
            self.advance()
            if self.at(TOK_EOF):
                break  # trailing comma
            conditions.append(self.parse_prop())
        if not self.at(TOK_EOF):
            tok = self.peek()
            raise ParseError(
                f"Expected ',' or end of input, got {tok.type} ({tok.value!r}) "
                f"at pos {tok.pos}. Parsed {len(conditions)} condition(s) — "
                f"is a comma missing?"
            )
        return conditions


# ---------------------------------------------------------------------------
# Source extraction + top-level API
# ---------------------------------------------------------------------------


def kernel_comment_prologue(source: str, kernel_name: str) -> str | None:
    """Return the contiguous comment prologue for one function.

    The prologue ends immediately before the function's first decorator (or
    ``def``) and never crosses an earlier non-comment source line.  Both the
    proof goals and ``@params``/``@grid`` metadata use this same boundary.
    """
    lines = source.splitlines()
    tree = ast.parse(source)
    functions = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == kernel_name
    ]
    if not functions:
        return None
    if len(functions) != 1:
        raise ValueError(
            f"Expected exactly one top-level function named {kernel_name!r}, "
            f"found {len(functions)}"
        )
    function = functions[0]
    first_code_line = min(
        [function.lineno] + [decorator.lineno for decorator in function.decorator_list]
    )
    cursor = first_code_line - 2  # line before decorator/def, zero based
    while cursor >= 0:
        stripped = lines[cursor].strip()
        if stripped and not stripped.startswith("#"):
            break
        cursor -= 1
    return "\n".join(lines[cursor + 1 : first_code_line - 1])


def _extract_annotation_blocks(
    source: str, tag: str, kernel_name: str | None = None
) -> list[str]:
    """Extract every ``# @tag(...)`` block in one kernel prologue.

    If kernel_name is provided, only searches the comment block immediately
    preceding the ``def kernel_name(`` line. Otherwise searches the entire source.

    Returns the text between each pair of outer parentheses in source order.
    Most annotations use :func:`_extract_annotation_block`, which additionally
    requires uniqueness.  ``@verif`` intentionally uses this repeated form.
    """
    if kernel_name is not None:
        # Restrict the search to the contiguous comment prologue immediately
        # before this function's first decorator.  Searching every earlier
        # source line can silently borrow a different kernel's annotations when
        # the requested kernel omits one of the tags.
        prologue = kernel_comment_prologue(source, kernel_name)
        if prologue is None:
            return []
        source = prologue

    lines = source.splitlines()
    content_lines: list[str] = []
    blocks: list[str] = []
    in_block = False
    depth = 0
    prefix = f"# @{tag}("

    for line in lines:
        stripped = line.strip()
        if not in_block:
            if stripped.startswith(prefix):
                in_block = True
                # Content after the opening tag
                rest = stripped[len(prefix):]
                depth = 1 + rest.count("(") - rest.count(")")
                if depth < 0:
                    raise ParseError(f"Unbalanced closing parenthesis in @{tag}")
                # Check if block closes on same line
                if depth <= 0:
                    # Remove trailing )
                    rest = rest.rstrip()
                    if rest.endswith(")"):
                        rest = rest[:-1]
                    if rest:
                        content_lines.append(rest)
                    blocks.append(" ".join(content_lines))
                    content_lines = []
                    in_block = False
                else:
                    if rest:
                        content_lines.append(rest)
        else:
            if stripped.startswith("#"):
                content = stripped[1:].strip()
                depth += content.count("(") - content.count(")")
                if depth < 0:
                    raise ParseError(f"Unbalanced closing parenthesis in @{tag}")
                if depth <= 0:
                    # Last line — strip the final closing paren
                    content = content.rstrip()
                    if content.endswith(")"):
                        content = content[:-1]
                    if content:
                        content_lines.append(content)
                    blocks.append(" ".join(content_lines))
                    content_lines = []
                    in_block = False
                else:
                    content_lines.append(content)
            else:
                raise ParseError(
                    f"Non-comment line interrupts @{tag} annotation block"
                )

    if in_block:
        raise ParseError(f"Unterminated @{tag} annotation block")
    return blocks


def _extract_annotation_block(
    source: str, tag: str, kernel_name: str | None = None
) -> str | None:
    """Extract one unique ``# @tag(...)`` block, if present."""

    blocks = _extract_annotation_blocks(source, tag, kernel_name)
    if not blocks:
        return None
    if len(blocks) != 1:
        raise ParseError(f"Duplicate @{tag} annotation block")
    return blocks[0]


def _split_top_level(text: str) -> list[str]:
    """Split comma-separated annotation clauses without reparsing expressions."""

    parts: list[str] = []
    current: list[str] = []
    paren_depth = 0
    bracket_depth = 0
    for character in text:
        if character == "(":
            paren_depth += 1
        elif character == ")":
            paren_depth -= 1
        elif character == "[":
            bracket_depth += 1
        elif character == "]":
            bracket_depth -= 1
        if paren_depth < 0 or bracket_depth < 0:
            raise ParseError("Unbalanced delimiter in @verif annotation")
        if character == "," and paren_depth == 0 and bracket_depth == 0:
            part = "".join(current).strip()
            if part:
                parts.append(part)
            current = []
        else:
            current.append(character)
    if paren_depth != 0 or bracket_depth != 0:
        raise ParseError("Unbalanced delimiter in @verif annotation")
    tail = "".join(current).strip()
    if tail:
        parts.append(tail)
    return parts


def _goal_clause(text: str) -> tuple[str, str]:
    match = re.fullmatch(r"([A-Za-z_]\w*)\((.*)\)", text, re.DOTALL)
    if match is None:
        raise ParseError(f"Malformed @verif clause {text!r}")
    return match.group(1), match.group(2).strip()


def parse_annotation_text(text: str, same_vars: set[str] | None = None) -> list[Prop]:
    """Parse a raw annotation text string into a list of conditions."""
    tokens = _tokenize(text)
    parser = _Parser(tokens, same_vars=same_vars)
    return parser.parse_condition_list()


def _parse_singleton_text(text: str) -> list[SingletonSpec]:
    """Parse singleton annotation text: var, left=expr, right=expr.

    Multiple singletons can be comma-separated at top level, where each
    singleton is: var left=expr right=expr
    """
    tokens = _tokenize(text)
    parser = _Parser(tokens)
    specs: list[SingletonSpec] = []
    while not parser.at(TOK_EOF):
        var_name = parser.expect(TOK_IDENT).value
        # Expect "left" "=" expr
        left_kw = parser.expect(TOK_IDENT)
        if left_kw.value != "left":
            raise ParseError(f"Expected 'left', got {left_kw.value!r}")
        parser.expect(TOK_EQ)
        left_expr = parser.parse_expr()
        # Expect "right" "=" expr
        right_kw = parser.expect(TOK_IDENT)
        if right_kw.value != "right":
            raise ParseError(f"Expected 'right', got {right_kw.value!r}")
        parser.expect(TOK_EQ)
        right_expr = parser.parse_expr()
        specs.append(SingletonSpec(var=var_name, left=left_expr, right=right_expr))
        # Optional comma between singletons
        if parser.at(TOK_COMMA):
            parser.advance()
    return specs


def _parse_same_text(text: str) -> set[str]:
    """Parse @same annotation text: comma-separated variable names.

    Example: "N, K, H" → {"N", "K", "H"}
    """
    tokens = _tokenize(text)
    parser = _Parser(tokens)
    names: set[str] = set()
    if parser.at(TOK_EOF):
        return names
    name_tok = parser.expect(TOK_IDENT)
    names.add(name_tok.value)
    while parser.at(TOK_COMMA):
        parser.advance()
        if parser.at(TOK_EOF):
            break  # trailing comma
        name_tok = parser.expect(TOK_IDENT)
        names.add(name_tok.value)
    if not parser.at(TOK_EOF):
        tok = parser.peek()
        raise ParseError(
            f"Expected ',' or end of @same, got {tok.type} "
            f"({tok.value!r}) at pos {tok.pos}"
        )
    return names


def _parse_verif_goal_text(text: str) -> RelationalProofGoal:
    parts = _split_top_level(text)
    if not parts or re.fullmatch(r"[A-Za-z_]\w*", parts[0]) is None:
        raise ParseError("@verif must start with a proof-goal name")
    name = parts[0]
    clauses: dict[str, str] = {}
    for raw_clause in parts[1:]:
        clause, body = _goal_clause(raw_clause)
        if clause not in {"same", "pre", "post", "singleton"}:
            raise ParseError(f"Unknown @{clause} clause inside @verif")
        if clause in clauses:
            raise ParseError(f"Duplicate {clause}(...) clause in @verif {name!r}")
        clauses[clause] = body
    if "post" not in clauses:
        raise ParseError(f"@verif {name!r} must contain post(...)")

    same_vars = _parse_same_text(clauses.get("same", ""))
    pre_conditions = parse_annotation_text(
        clauses.get("pre", ""), same_vars=same_vars
    )
    post_conditions = parse_annotation_text(clauses["post"], same_vars=same_vars)
    if not post_conditions:
        raise ParseError(f"@verif {name!r} has an empty post(...)")

    singletons = _parse_singleton_text(clauses.get("singleton", ""))
    singleton_names = [spec.var for spec in singletons]
    if len(set(singleton_names)) != len(singleton_names):
        raise ParseError("Duplicate variable in singleton(...) clause")
    return RelationalProofGoal(
        name=name,
        pre_conditions=pre_conditions,
        post_conditions=post_conditions,
        singletons=singletons,
        same_vars=same_vars,
    )


def parse_verif_goals(
    source: str, kernel_name: str | None = None
) -> tuple[RelationalProofGoal, ...]:
    """Parse all named relational goals preceding one Triton kernel."""

    legacy_tags = (
        "same",
        "pre",
        "post",
        "singleton",
        "witness",
        "causal_selected_row",
    )
    present_legacy = [
        tag
        for tag in legacy_tags
        if _extract_annotation_blocks(source, tag, kernel_name)
    ]
    if present_legacy:
        raise ParseError(
            "Legacy standalone proof annotations are not supported; place "
            f"these clauses inside a named @verif: {present_legacy}"
        )

    goals = tuple(
        _parse_verif_goal_text(text)
        for text in _extract_annotation_blocks(source, "verif", kernel_name)
    )
    names = [goal.name for goal in goals]
    if len(set(names)) != len(names):
        raise ParseError(f"Duplicate @verif proof-goal name: {names}")
    return goals


def parse_verif_goal(
    source: str,
    kernel_name: str | None = None,
    goal_name: str | None = None,
) -> RelationalProofGoal | None:
    """Select one named goal, requiring an explicit name when there are several."""

    goals = parse_verif_goals(source, kernel_name)
    if goal_name is not None:
        matches = [goal for goal in goals if goal.name == goal_name]
        if not matches:
            raise ParseError(f"No @verif proof goal named {goal_name!r}")
        return matches[0]
    if not goals:
        return None
    if len(goals) != 1:
        raise ParseError(
            "Kernel has multiple @verif proof goals; select one by name"
        )
    return goals[0]
