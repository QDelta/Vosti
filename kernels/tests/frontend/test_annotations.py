"""Tests for the annotation parser and annotation-driven proof flow."""

import pytest
from ir.annotations import (
    _tokenize,
    _parse_same_text,
    parse_annotation_text,
    parse_verif_goal,
    parse_verif_goals,
    _extract_annotation_block,
    ParseError,
    Left,
    Right,
    FreeVar,
    IntConst,
    AnnBinOp,
    AnnIndex,
    AnnSlice,
    RegionRef,
    RegionEquiv,
    AnnComparison,
    TOK_IDENT,
    TOK_INT,
    TOK_EQEQ,
    TOK_EQ,
    TOK_GT,
    TOK_GE,
    TOK_LT,
    TOK_LE,
    TOK_LPAREN,
    TOK_RPAREN,
    TOK_LBRACKET,
    TOK_RBRACKET,
    TOK_COLON,
    TOK_COMMA,
    TOK_PLUS,
    TOK_MINUS,
    TOK_STAR,
    TOK_PERCENT,
    TOK_EOF,
)


# ---------------------------------------------------------------------------
# Tokenizer tests
# ---------------------------------------------------------------------------


class TestTokenizer:
    def test_basic_tokens(self):
        tokens = _tokenize("left(M) > 0")
        types = [t.type for t in tokens]
        assert types == [TOK_IDENT, TOK_LPAREN, TOK_IDENT, TOK_RPAREN, TOK_GT, TOK_INT, TOK_EOF]

    def test_comparison_operators(self):
        tokens = _tokenize("a == b >= c <= d > e < f")
        ops = [t.type for t in tokens if t.type not in (TOK_IDENT, TOK_EOF)]
        assert ops == [TOK_EQEQ, TOK_GE, TOK_LE, TOK_GT, TOK_LT]

    def test_arithmetic(self):
        tokens = _tokenize("x + 1 * y - 2 % 3")
        ops = [t.type for t in tokens if t.type not in (TOK_IDENT, TOK_INT, TOK_EOF)]
        assert ops == [TOK_PLUS, TOK_STAR, TOK_MINUS, TOK_PERCENT]

    def test_brackets_and_colons(self):
        tokens = _tokenize("[0:M, x:x+1]")
        types = [t.type for t in tokens if t.type != TOK_EOF]
        assert types == [
            TOK_LBRACKET, TOK_INT, TOK_COLON, TOK_IDENT, TOK_COMMA,
            TOK_IDENT, TOK_COLON, TOK_IDENT, TOK_PLUS, TOK_INT, TOK_RBRACKET,
        ]

    def test_eq_vs_eqeq(self):
        tokens = _tokenize("a = 1, b == 2")
        eq_tokens = [t for t in tokens if t.type in (TOK_EQ, TOK_EQEQ)]
        assert eq_tokens[0].type == TOK_EQ
        assert eq_tokens[1].type == TOK_EQEQ

    def test_rejects_unrecognized_characters_instead_of_skipping_them(self):
        with pytest.raises(ParseError, match="Unexpected character"):
            _tokenize("left(M) > 0 @ left(N) > 0")


# ---------------------------------------------------------------------------
# Parser tests
# ---------------------------------------------------------------------------


class TestParser:
    def test_scalar_constraint_gt(self):
        conds = parse_annotation_text("left(M) > 0")
        assert len(conds) == 1
        c = conds[0]
        assert isinstance(c, AnnComparison)
        assert c.op == ">"
        assert c.lhs == Left("M")
        assert c.rhs == IntConst(0)

    def test_scalar_constraint_eq(self):
        conds = parse_annotation_text("right(M) == 1")
        assert len(conds) == 1
        c = conds[0]
        assert isinstance(c, AnnComparison)
        assert c.op == "=="
        assert c.lhs == Right("M")
        assert c.rhs == IntConst(1)

    def test_shared_constraint(self):
        conds = parse_annotation_text("left(N) == right(N)")
        assert len(conds) == 1
        c = conds[0]
        assert isinstance(c, AnnComparison)
        assert c.op == "=="
        assert c.lhs == Left("N")
        assert c.rhs == Right("N")

    def test_free_var_constraint(self):
        conds = parse_annotation_text("x >= 0")
        assert len(conds) == 1
        c = conds[0]
        assert isinstance(c, AnnComparison)
        assert c.lhs == FreeVar("x")
        assert c.rhs == IntConst(0)

    def test_arithmetic_expr(self):
        conds = parse_annotation_text("x < left(M) + 1")
        c = conds[0]
        assert isinstance(c, AnnComparison)
        assert c.op == "<"
        assert isinstance(c.rhs, AnnBinOp)
        assert c.rhs.op == "+"
        assert c.rhs.lhs == Left("M")
        assert c.rhs.rhs == IntConst(1)

    def test_modulo(self):
        conds = parse_annotation_text("left(H) % left(Hkv) == 0")
        c = conds[0]
        assert isinstance(c, AnnComparison)
        assert c.op == "=="
        assert isinstance(c.lhs, AnnBinOp)
        assert c.lhs.op == "%"

    def test_region_equiv(self):
        conds = parse_annotation_text(
            "left(a)[x:x+1, 0:left(K)] == right(a)[0:1, 0:right(K)]"
        )
        assert len(conds) == 1
        c = conds[0]
        assert isinstance(c, RegionEquiv)
        assert c.left.side == Left("a")
        assert len(c.left.slices) == 2
        assert c.left.slices[0].start == FreeVar("x")
        assert isinstance(c.left.slices[0].stop, AnnBinOp)
        assert c.right.side == Right("a")
        assert c.right.slices[0].start == IntConst(0)
        assert c.right.slices[0].stop == IntConst(1)

    def test_quantified_region_equiv_has_explicit_binder_and_domain(self):
        from ir.annotations import ForAllConstraint, AnnImplies
        conds = parse_annotation_text(
            "forall(tile, implies(and(tile >= 0, tile < 4), "
            "left(a)[tile:tile+1] == right(a)[tile:tile+1]))"
        )
        assert len(conds) == 1
        condition = conds[0]
        assert isinstance(condition, ForAllConstraint)
        assert condition.vars == ["tile"]
        assert isinstance(condition.body, AnnImplies)
        assert condition.body.consequent.left.slices[0].start == FreeVar("tile")
        assert condition.body.consequent.right.slices[0].start == FreeVar("tile")

    def test_multiple_conditions(self):
        text = "right(M) == 1, left(N) == right(N), left(M) > 0"
        conds = parse_annotation_text(text)
        assert len(conds) == 3
        assert all(isinstance(c, AnnComparison) for c in conds)

    def test_index_expr(self):
        conds = parse_annotation_text(
            "left(cu)[x+1] - left(cu)[x] == right(cu)[1] - right(cu)[0]"
        )
        c = conds[0]
        assert isinstance(c, AnnComparison)
        assert isinstance(c.lhs, AnnBinOp)
        assert c.lhs.op == "-"
        assert isinstance(c.lhs.lhs, AnnIndex)
        assert c.lhs.lhs.base == Left("cu")

    def test_cdiv_function(self):
        conds = parse_annotation_text("x < cdiv(left(M), 64)")
        c = conds[0]
        assert isinstance(c.rhs, AnnBinOp)
        assert c.rhs.op == "cdiv"

    def test_trailing_comma(self):
        conds = parse_annotation_text("left(M) > 0,")
        assert len(conds) == 1

    def test_forall_implies_and(self):
        from ir.annotations import ForAllConstraint, AnnImplies, AnnAnd, AnnComparison
        conds = parse_annotation_text(
            "forall(i, j, implies(i < j, and(left(cu_q)[i] < left(cu_q)[j], left(cu_k)[i] < left(cu_k)[j])))"
        )
        assert len(conds) == 1
        c = conds[0]
        assert isinstance(c, ForAllConstraint)
        assert c.vars == ["i", "j"]
        assert isinstance(c.body, AnnImplies)
        assert isinstance(c.body.antecedent, AnnComparison)
        assert isinstance(c.body.consequent, AnnAnd)
        assert len(c.body.consequent.args) == 2


# ---------------------------------------------------------------------------
# Source extraction tests
# ---------------------------------------------------------------------------


class TestExtractAnnotation:
    def test_extract_pre(self):
        source = """\
# @pre(
#   right(M) == 1,
#   left(M) > 0,
# )
"""
        text = _extract_annotation_block(source, "pre")
        assert text is not None
        assert "right(M) == 1" in text
        assert "left(M) > 0" in text

    def test_extract_post(self):
        source = """\
# @post(
#   left(c)[x:x+1, 0:N] == right(c)[0:1, 0:N]
# )
"""
        text = _extract_annotation_block(source, "post")
        assert text is not None
        assert "left(c)" in text

    def test_extract_same(self):
        source = """\
# @same(N, K)
# @pre(
#   right(M) == 1,
# )
"""
        text = _extract_annotation_block(source, "same")
        assert text is not None
        assert "N" in text
        assert "K" in text

    def test_no_annotation(self):
        source = "# just a comment\n"
        assert _extract_annotation_block(source, "pre") is None

    def test_empty_annotation_is_distinct_from_missing(self):
        assert _extract_annotation_block("# @post()\n", "post") == ""

    def test_duplicate_annotation_block_is_rejected(self):
        source = "# @post(left(a)[0:1] == right(a)[0:1])\n" \
                 "# @post(left(b)[0:1] == right(b)[0:1])\n"
        with pytest.raises(ParseError, match="Duplicate @post"):
            _extract_annotation_block(source, "post")

    def test_unterminated_annotation_block_is_rejected(self):
        with pytest.raises(ParseError, match="Unterminated @pre"):
            _extract_annotation_block("# @pre(\n# left(M) > 0\n", "pre")

    def test_parse_verif_goal_full(self):
        source = """\
# @verif(batch_invariance,
#   same(N, K),
#   pre(
#     right(M) == 1,
#     left(a)[x:x+1, 0:K] == right(a)[0:1, 0:K],
#   ),
#   post(
#     left(c)[x:x+1, 0:N] == right(c)[0:1, 0:N]
#   ),
# )
"""
        result = parse_verif_goal(source)
        assert result is not None
        assert result.name == "batch_invariance"
        assert result.same_vars == {"N", "K"}
        assert len(result.pre_conditions) == 2
        assert len(result.post_conditions) == 1
        assert isinstance(result.pre_conditions[0], AnnComparison)
        assert isinstance(result.pre_conditions[1], RegionEquiv)
        assert isinstance(result.post_conditions[0], RegionEquiv)

    def test_multiple_named_verif_goals_are_independent(self):
        source = """\
# @verif(first, post(left(a)[0:1] == right(a)[0:1]))
# @verif(second, same(N), post(left(b)[0:N] == right(b)[0:N]))
"""
        goals = parse_verif_goals(source)
        assert [goal.name for goal in goals] == ["first", "second"]
        assert goals[0].same_vars == set()
        assert goals[1].same_vars == {"N"}
        with pytest.raises(ParseError, match="multiple @verif"):
            parse_verif_goal(source)
        assert parse_verif_goal(source, goal_name="second") == goals[1]

    def test_duplicate_verif_goal_name_is_rejected(self):
        source = """\
# @verif(same_name, post(left(a)[0:1] == right(a)[0:1]))
# @verif(same_name, post(left(b)[0:1] == right(b)[0:1]))
"""
        with pytest.raises(ParseError, match="Duplicate @verif proof-goal"):
            parse_verif_goals(source)

    @pytest.mark.parametrize(
        "legacy",
        [
            "# @same(N)",
            "# @pre(left(N) > 0)",
            "# @post(left(a)[0:1] == right(a)[0:1])",
            "# @singleton(bi left=0 right=0)",
            "# @witness(i left=0 right=0)",
            "# @causal_selected_row(roles(query=q))",
        ],
    )
    def test_named_verif_rejects_legacy_standalone_clauses(self, legacy):
        source = (
            f"{legacy}\n"
            "# @verif(goal, post(left(a)[0:1] == right(a)[0:1]))\n"
        )
        with pytest.raises(ParseError, match="Legacy standalone"):
            parse_verif_goals(source)

    def test_kernel_scope_does_not_borrow_previous_annotations(self):
        source = """\
# @verif(first, pre(left(M) > 0), post(left(a)[0:1] == right(a)[0:1]))
def first_kernel():
    pass

# @verif(target, post(left(b)[0:1] == right(b)[0:1]))
def target_kernel():
    pass
"""
        result = parse_verif_goal(source, "target_kernel")
        assert result is not None
        assert result.pre_conditions == []
        assert len(result.post_conditions) == 1
        assert result.post_conditions[0].left.side.name == "b"


# ---------------------------------------------------------------------------
# Binding symmetry tests — ensure x = y and y = x produce the same result
# ---------------------------------------------------------------------------


class TestBindingSymmetry:
    """Verify that _find_equality_bindings and _is_binding_constraint
    are order-independent: x = y and y = x yield the same bindings."""

    def test_right_const_both_orderings(self):
        from ir.annotation_to_config import _find_equality_bindings, _is_binding_constraint

        # right(M) == 1
        cond1 = AnnComparison(op="==", lhs=Right("M"), rhs=IntConst(1))
        # 1 = right(M)
        cond2 = AnnComparison(op="==", lhs=IntConst(1), rhs=Right("M"))

        l1, r1, s1 = _find_equality_bindings([cond1])
        l2, r2, s2 = _find_equality_bindings([cond2])
        assert r1 == r2 == {"M": 1}
        assert l1 == l2 == {}
        assert s1 == s2 == set()

        assert _is_binding_constraint(cond1)
        assert _is_binding_constraint(cond2)

    def test_left_const_both_orderings(self):
        from ir.annotation_to_config import _find_equality_bindings, _is_binding_constraint

        # left(M) == 5
        cond1 = AnnComparison(op="==", lhs=Left("M"), rhs=IntConst(5))
        # 5 = left(M)
        cond2 = AnnComparison(op="==", lhs=IntConst(5), rhs=Left("M"))

        l1, r1, s1 = _find_equality_bindings([cond1])
        l2, r2, s2 = _find_equality_bindings([cond2])
        assert l1 == l2 == {"M": 5}
        assert r1 == r2 == {}
        assert s1 == s2 == set()

        assert _is_binding_constraint(cond1)
        assert _is_binding_constraint(cond2)

    def test_shared_both_orderings(self):
        from ir.annotation_to_config import _find_equality_bindings, _is_binding_constraint

        # left(N) == right(N)
        cond1 = AnnComparison(op="==", lhs=Left("N"), rhs=Right("N"))
        # right(N) == left(N)
        cond2 = AnnComparison(op="==", lhs=Right("N"), rhs=Left("N"))

        l1, r1, s1 = _find_equality_bindings([cond1])
        l2, r2, s2 = _find_equality_bindings([cond2])
        assert s1 == s2 == {"N"}
        assert l1 == l2 == {}
        assert r1 == r2 == {}

        assert _is_binding_constraint(cond1)
        assert _is_binding_constraint(cond2)

    def test_non_binding_is_not_binding(self):
        from ir.annotation_to_config import _is_binding_constraint

        # left(M) > 0 — not a binding
        assert not _is_binding_constraint(
            AnnComparison(op=">", lhs=Left("M"), rhs=IntConst(0))
        )
        # left(X) == right(Y) where X != Y — not a binding, it's a constraint
        assert not _is_binding_constraint(
            AnnComparison(op="==", lhs=Left("X"), rhs=Right("Y"))
        )

    def test_eqeq_also_recognized_as_binding(self):
        from ir.annotation_to_config import _find_equality_bindings, _is_binding_constraint

        # right(M) == 1 (using == instead of =)
        cond = AnnComparison(op="==", lhs=Right("M"), rhs=IntConst(1))
        _, r, _ = _find_equality_bindings([cond])
        assert r == {"M": 1}
        assert _is_binding_constraint(cond)

    def test_reversed_annotations_full_proof(self):
        """Matmul proof succeeds even when annotation operand order is reversed."""
        from ir.relational_dataflow import prove_relational_dataflow_from_annotations

        # Read real matmul source and reverse the binding orderings
        with open("triton_kernels/matmul.py") as f:
            source = f.read()

        # Reverse the equality operands.
        source = source.replace("right(M) == 1", "1 == right(M)")

        result = prove_relational_dataflow_from_annotations(
            source, "matmul_kernel",
            {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32},
        )
        assert result.proved, [
            f"{c.name}: {c.details}" for c in result.checks if not c.proved
        ]


# ---------------------------------------------------------------------------
# @same annotation tests
# ---------------------------------------------------------------------------


class TestSame:
    def test_parse_same_text(self):
        names = _parse_same_text("N, K, H")
        assert names == {"N", "K", "H"}

    def test_parse_same_text_trailing_comma(self):
        names = _parse_same_text("N, K,")
        assert names == {"N", "K"}

    def test_parse_same_text_single(self):
        names = _parse_same_text("N")
        assert names == {"N"}

    def test_parse_same_text_empty(self):
        names = _parse_same_text("")
        assert names == set()

    def test_parse_same_text_rejects_missing_comma(self):
        with pytest.raises(ParseError, match="Expected ',' or end of @same"):
            _parse_same_text("N K")

    def test_same_var_parsed_as_left(self):
        """Bare @same var in expression becomes Left node."""
        conds = parse_annotation_text("N > 0", same_vars={"N"})
        assert len(conds) == 1
        c = conds[0]
        assert isinstance(c, AnnComparison)
        assert c.lhs == Left("N")

    def test_same_var_in_arithmetic(self):
        """@same var in arithmetic: N + 1."""
        conds = parse_annotation_text("x < N + 1", same_vars={"N"})
        c = conds[0]
        assert isinstance(c, AnnComparison)
        assert isinstance(c.rhs, AnnBinOp)
        assert c.rhs.lhs == Left("N")

    def test_same_var_with_indexing(self):
        """@same var with index: cu[0] == 0."""
        conds = parse_annotation_text("cu[0] == 0", same_vars={"cu"})
        c = conds[0]
        assert isinstance(c, AnnComparison)
        assert isinstance(c.lhs, AnnIndex)
        assert c.lhs.base == Left("cu")

    def test_same_var_in_region_slice(self):
        """@same var used inside region slice: left(a)[0:N]."""
        conds = parse_annotation_text(
            "left(a)[0:N, 0:K] == right(a)[0:N, 0:K]",
            same_vars={"N", "K"},
        )
        c = conds[0]
        assert isinstance(c, RegionEquiv)
        # N in slices should be Left("N")
        assert c.left.slices[0].stop == Left("N")
        assert c.left.slices[1].stop == Left("K")

    def test_reject_left_of_same_var(self):
        """left(N) is rejected when N is in @same."""
        with pytest.raises(ParseError, match="declared in @same"):
            parse_annotation_text("left(N) > 0", same_vars={"N"})

    def test_reject_right_of_same_var(self):
        """right(N) is rejected when N is in @same."""
        with pytest.raises(ParseError, match="declared in @same"):
            parse_annotation_text("right(N) == 1", same_vars={"N"})

    def test_non_same_var_still_free(self):
        """Variables not in @same are still parsed as FreeVar."""
        conds = parse_annotation_text("x >= 0, N > 0", same_vars={"N"})
        assert conds[0].lhs == FreeVar("x")
        assert conds[1].lhs == Left("N")

    def test_parse_verif_goal_with_same(self):
        """Nested same/pre/post clauses share one proof-goal scope."""
        source = """\
# @verif(batch_invariance,
#   same(N, K),
#   pre(
#     right(M) == 1,
#     N > 0,
#     left(a)[x:x+1, 0:K] == right(a)[0:1, 0:K],
#   ),
#   post(
#     left(c)[x:x+1, 0:N] == right(c)[0:1, 0:N]
#   ),
# )
def my_kernel():
    pass
"""
        result = parse_verif_goal(source, "my_kernel")
        assert result is not None
        assert result.same_vars == {"N", "K"}
        assert len(result.pre_conditions) == 3
        assert len(result.post_conditions) == 1
        # N > 0 should use Left("N")
        assert result.pre_conditions[1].lhs == Left("N")

    def test_matmul_proof_with_same(self):
        """Matmul proof succeeds with @same annotations."""
        from ir.relational_dataflow import prove_relational_dataflow_from_annotations

        with open("triton_kernels/matmul.py") as f:
            source = f.read()

        # Verify matmul.py scopes same(...) within a named proof goal.
        assert "#   same(N, K)" in source

        result = prove_relational_dataflow_from_annotations(
            source, "matmul_kernel",
            {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32},
        )
        assert result.proved, [
            f"{c.name}: {c.details}" for c in result.checks if not c.proved
        ]


# ---------------------------------------------------------------------------
# End-to-end proof tests
# ---------------------------------------------------------------------------


class TestProofFromAnnotations:
    def test_proof_entrypoint_rejects_source_outside_validated_subset(self):
        """Callers cannot bypass the source-semantics gate."""
        from ir.relational_dataflow import prove_relational_dataflow_from_annotations

        with open("triton_kernels/matmul.py") as f:
            source = f.read()
        source = source.replace(
            'tl.load(block_ptr_a, boundary_check=(0, 1), padding_option="zero")',
            "tl.load(block_ptr_a)",
            1,
        )
        with pytest.raises(ValueError, match="outside the verifiable Triton subset"):
            prove_relational_dataflow_from_annotations(
                source,
                "matmul_kernel",
                {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32},
            )

    def test_proof_entrypoint_rejects_empty_postcondition(self):
        from ir.relational_dataflow import prove_relational_dataflow_from_annotations

        with open("triton_kernels/add.py") as f:
            source = f.read()
        source = source.replace(
            "#   post(\n"
            "#     left(o)[b:b+1, 0:N] == right(o)[0:1, 0:N]\n"
            "#   ),",
            "#   post(),",
        )
        with pytest.raises(ValueError, match="empty post"):
            prove_relational_dataflow_from_annotations(
                source,
                "add_kernel",
                {"BLOCK_M": 1, "BLOCK_N": 64},
            )

    def test_fattn_paged_singleton_width_is_independent(self):
        """Unused rectangular padding must not constrain singleton width."""
        with open("triton_kernels/fattn_paged.py") as f:
            source = f.read()
        annotation = parse_verif_goal(
            source,
            "fattn_varlen_paged_fwd_block_ptr_kernel",
            "batch_invariance",
        )
        assert annotation is not None
        assert "MAX_NUM_PAGES" not in annotation.same_vars

    def test_matmul(self):
        from ir.relational_dataflow import prove_relational_dataflow_from_annotations

        with open("triton_kernels/matmul.py") as f:
            source = f.read()

        result = prove_relational_dataflow_from_annotations(
            source, "matmul_kernel",
            {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32},
        )
        assert result.proved, [
            f"{c.name}: {c.details}" for c in result.checks if not c.proved
        ]

    def test_fattn_paged_block_ptr(self):
        from ir.relational_dataflow import prove_relational_dataflow_from_annotations

        with open("triton_kernels/fattn_paged.py") as f:
            source = f.read()

        result = prove_relational_dataflow_from_annotations(
            source, "fattn_varlen_paged_fwd_block_ptr_kernel",
            {"D_HEAD": 64, "BLOCK_M": 64, "BLOCK_N": 64, "PAGE_BLOCK_SIZE": 64},
            goal_name="batch_invariance",
        )
        assert result.proved, [
            f"{c.name}: {c.details}" for c in result.checks if not c.proved
        ]

    def test_store_kv_cache_data_dependent_output_coverage(self):
        """The selected source row is a checked witness for cache[slot[row]]."""
        from ir.relational_dataflow import prove_relational_dataflow_from_annotations

        with open("triton_kernels/store_kv_cache.py") as f:
            source = f.read()

        result = prove_relational_dataflow_from_annotations(
            source,
            "store_cache_kernel",
            {"KVD": 64, "BLOCK_M": 1},
            goal_name="batch_invariance",
        )
        assert result.proved, [
            f"{c.name}: {c.details}" for c in result.checks if not c.proved
        ]


# ---------------------------------------------------------------------------
# Trusted-driver soundness regressions
# ---------------------------------------------------------------------------


class TestProofDriverSoundness:
    def test_postcondition_must_be_fully_written(self):
        """One nonempty written column cannot justify whole-row equality."""
        from ir.relational_dataflow import prove_relational_dataflow_from_annotations

        with open("triton_kernels/add.py") as f:
            source = f.read()
        original = (
            "        offsets=(row, col),\n"
            "        block_shape=(BLOCK_M, BLOCK_N),\n"
            "        order=(0, 1),\n"
            "    )\n"
            "    tl.store("
        )
        shifted = (
            "        offsets=(row, col + N - 1),\n"
            "        block_shape=(BLOCK_M, BLOCK_N),\n"
            "        order=(0, 1),\n"
            "    )\n"
            "    tl.store("
        )
        assert source.count(original) == 1
        source = source.replace(original, shifted)
        result = prove_relational_dataflow_from_annotations(
            source,
            "add_kernel",
            {"BLOCK_M": 1, "BLOCK_N": 64},
        )
        coverage = [
            check for check in result.checks
            if check.name.endswith("output_fully_written")
        ]
        assert coverage
        assert any(not check.proved for check in coverage)
        assert result.verified_contract is None

    def test_empty_inner_loop_cannot_vacuously_hide_output_scalar(self):
        """A loop bound applies only to values computed inside that loop."""
        from ir.relational_dataflow import prove_relational_dataflow_from_annotations

        with open("triton_kernels/matmul.py") as f:
            source = f.read()
        marker = "    block_ptr_c = tl.make_block_ptr(\n"
        assert source.count(marker) == 1
        source = source.replace(
            marker,
            # This term is zero whenever the k loop is nonempty, but equals
            # program_id(0) when K == 0.  Batch and singleton executions then
            # disagree only in the empty-loop case.
            "    acc += (i * (1 - tl.minimum(K, 1))) * 1.0\n\n" + marker,
        )
        result = prove_relational_dataflow_from_annotations(
            source,
            "matmul_kernel",
            {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32},
        )
        assert not result.proved
        assert any(
            check.name.startswith("c:value_scalar_equiv_")
            and not check.proved
            for check in result.checks
        )
        assert result.verified_contract is None

    def test_iterator_pairing_context_must_be_jointly_satisfiable(self):
        import z3

        from ir import (
            Assign,
            BinOp,
            FloatType,
            Grid,
            GridIter,
            If,
            IntLit,
            IntType,
            Kernel,
            Param,
            Range,
            Slice,
            TensorType,
            Var,
        )
        from ir.regional_obligations import EquivProofConfig, TensorAssumption, prove_region_equivalence

        tensor_type = TensorType(FloatType(), [IntLit(2)])
        i = Var("i", type=IntType())
        kernel = Kernel(
            name="disjoint_effectful_iterators",
            params=[
                Param("x", tensor_type),
                Param("o", tensor_type),
                Param("selected", IntType()),
            ],
            grid=Grid(
                iters=[GridIter(i, Range(IntLit(0), IntLit(2)))],
                decls=[],
                body=[
                    If(
                        BinOp("==", Var("i"), Var("selected")),
                        [Assign(Var("o"), None, Var("x"))],
                        [],
                    )
                ],
            ),
        )
        whole = [Slice(IntLit(0), IntLit(2))]
        i_left, i_right = z3.Ints("disjoint_i_left disjoint_i_right")
        config = EquivProofConfig(
            kernel=kernel,
            output_tensor="o",
            left_output_region=whole,
            right_output_region=whole,
            tensor_assumptions={"x": TensorAssumption(whole, whole)},
            left_env={"i": i_left, "selected": z3.IntVal(0)},
            right_env={"i": i_right, "selected": z3.IntVal(1)},
            base_assumptions=[],
        )
        result = prove_region_equivalence(config)
        check = next(
            item
            for item in result.checks
            if item.name == "relational_execution_context_satisfiable"
        )
        assert not check.proved
        assert "unsat" in check.details

    def test_cross_tensor_precondition_is_rejected(self):
        """A left(x)==right(y) assumption says nothing about right(x)."""
        from ir.relational_dataflow import prove_relational_dataflow_from_annotations

        with open("triton_kernels/add.py") as f:
            source = f.read()
        source = source.replace(
            "left(x)[b:b+1, 0:N] == right(x)[0:1, 0:N]",
            "left(x)[b:b+1, 0:N] == right(y)[0:1, 0:N]",
        )
        with pytest.raises(ValueError, match="same tensor"):
            prove_relational_dataflow_from_annotations(
                source,
                "add_kernel",
                {"BLOCK_M": 1, "BLOCK_N": 64},
            )

    def test_cross_tensor_postcondition_is_rejected(self):
        from ir.relational_dataflow import prove_relational_dataflow_from_annotations

        with open("triton_kernels/add.py") as f:
            source = f.read()
        source = source.replace(
            "left(o)[b:b+1, 0:N] == right(o)[0:1, 0:N]",
            "left(o)[b:b+1, 0:N] == right(y)[0:1, 0:N]",
        )
        with pytest.raises(ValueError, match="same tensor"):
            prove_relational_dataflow_from_annotations(
                source,
                "add_kernel",
                {"BLOCK_M": 1, "BLOCK_N": 64},
            )

    def test_missing_value_input_assumption_is_rejected(self):
        """An accessed tensor omitted from @pre must not be silently ignored."""
        from ir.relational_dataflow import prove_relational_dataflow_from_annotations

        with open("triton_kernels/add.py") as f:
            source = f.read()
        source = source.replace(
            "#     left(y)[b:b+1, 0:N] == right(y)[0:1, 0:N],\n", ""
        )
        result = prove_relational_dataflow_from_annotations(
            source,
            "add_kernel",
            {"BLOCK_M": 1, "BLOCK_N": 64},
        )
        assert not result.proved
        assert any(
            check.name == "o:tensor_assumption_y" and not check.proved
            for check in result.checks
        )

    def test_singleton_program_id_in_output_value_is_rejected(self):
        """Equal read regions do not justify adding different batch row ids."""
        from ir.relational_dataflow import prove_relational_dataflow_from_annotations

        with open("triton_kernels/add.py") as f:
            source = f.read()
        source = source.replace(
            "(x_block + y_block).to(o.dtype.element_ty)",
            "(x_block + y_block + row).to(o.dtype.element_ty)",
        )
        result = prove_relational_dataflow_from_annotations(
            source,
            "add_kernel",
            {"BLOCK_M": 1, "BLOCK_N": 64},
        )
        assert not result.proved
        assert any(
            "value_scalar_equiv" in check.name and not check.proved
            for check in result.checks
        )

    def test_value_parameter_not_in_same_is_rejected(self):
        """Different eps values can change RMSNorm despite identical regions."""
        from ir.relational_dataflow import prove_relational_dataflow_from_annotations

        with open("triton_kernels/rmsnorm.py") as f:
            source = f.read()
        source = source.replace("#   same(N, eps)", "#   same(N)")
        result = prove_relational_dataflow_from_annotations(
            source,
            "rmsnorm_kernel",
            {"BLOCK_M": 1, "BLOCK_N": 64},
        )
        assert not result.proved
        assert any(
            "value_scalar_equiv" in check.name and not check.proved
            for check in result.checks
        )

    def test_every_post_output_is_actually_proved(self):
        """A bad second output must fail even when the first output is valid."""
        from ir.relational_dataflow import prove_relational_dataflow_from_annotations

        with open("triton_kernels/rmsnorm_residual.py") as f:
            source = f.read()
        source = source.replace(
            "tl.store(ro_block_ptr, hidden.to(residual_out.dtype.element_ty)",
            "tl.store(ro_block_ptr, (hidden + row).to(residual_out.dtype.element_ty)",
        )
        result = prove_relational_dataflow_from_annotations(
            source,
            "rmsnorm_residual_kernel",
            {"BLOCK_M": 1, "BLOCK_N": 64},
        )
        assert not result.proved
        assert any(
            check.name.startswith("residual_out:value_scalar_equiv_")
            and not check.proved
            for check in result.checks
        )

    def test_named_goal_may_cover_only_one_store(self):
        """A proof goal may state a sound theorem about one visible effect."""
        from ir.relational_dataflow import prove_relational_dataflow_from_annotations

        with open("triton_kernels/rmsnorm_residual.py") as f:
            source = f.read()
        source = source.replace(
            "#     left(o)[b:b+1, 0:N] == right(o)[0:1, 0:N],\n"
            "#     left(residual_out)[b:b+1, 0:N] == right(residual_out)[0:1, 0:N]\n",
            "#     left(o)[b:b+1, 0:N] == right(o)[0:1, 0:N]\n",
        )
        result = prove_relational_dataflow_from_annotations(
            source,
            "rmsnorm_residual_kernel",
            {"BLOCK_M": 1, "BLOCK_N": 64},
        )
        assert result.proved
        assert result.verified_contract is not None
        post = result.verified_contract.to_data()["theorem_contract"]["theorem"]["post"]
        assert [condition["left"]["tensor"] for condition in post] == ["o"]

    def test_contradictory_scalar_preconditions_are_vacuously_valid(self):
        """A Hoare theorem with contradictory premises is sound but vacuous."""
        from ir.relational_dataflow import prove_relational_dataflow_from_annotations

        with open("triton_kernels/add.py") as f:
            source = f.read()
        source = source.replace("left(M) > 0", "left(M) < 0")
        result = prove_relational_dataflow_from_annotations(
            source,
            "add_kernel",
            {"BLOCK_M": 1, "BLOCK_N": 64},
        )
        assert result.proved
        assert result.verified_contract is None
        assert result.annotation_satisfiability == "unsat"
        assert any(
            check.name == "o:quantifier_free_preconditions_satisfiable"
            and not check.proved
            and "vacuous" in check.details
            for check in result.diagnostics
        )
        assert any(
            check.name == "o:vacuous_preconditions" and check.proved
            for check in result.checks
        )

    def test_contradictory_binding_preconditions_are_vacuously_valid(self):
        """Contradictory bindings remain visible as a vacuity diagnostic."""
        from ir.relational_dataflow import prove_relational_dataflow_from_annotations

        with open("triton_kernels/add.py") as f:
            source = f.read()
        source = source.replace(
            "#     right(M) == 1,",
            "#     left(M) == right(M),\n"
            "#     left(M) == 1,\n"
            "#     right(M) == 2,",
        )
        result = prove_relational_dataflow_from_annotations(
            source,
            "add_kernel",
            {"BLOCK_M": 1, "BLOCK_N": 64},
        )
        assert result.proved
        assert result.verified_contract is None
        assert result.annotation_satisfiability == "unsat"
        assert any(
            check.name == "o:quantifier_free_preconditions_satisfiable"
            and not check.proved
            and "vacuous" in check.details
            for check in result.diagnostics
        )

    def test_quantified_preconditions_are_ordinary_conditional_assumptions(self):
        """Quantified Hoare premises do not require separate witness syntax."""
        from ir.relational_dataflow import prove_relational_dataflow_from_annotations

        with open("triton_kernels/store_kv_cache.py") as f:
            source = f.read()
        assert "@witness" not in source
        result = prove_relational_dataflow_from_annotations(
            source,
            "store_cache_kernel",
            {"KVD": 64, "BLOCK_M": 1},
            goal_name="batch_invariance",
        )
        assert result.proved

    def test_tensor_same_is_a_real_whole_tensor_assumption(self):
        """Embedding's immutable shared weight must be checked, not ignored."""
        from ir.relational_dataflow import prove_relational_dataflow_from_annotations

        with open("triton_kernels/embedding.py") as f:
            source = f.read()
        result = prove_relational_dataflow_from_annotations(
            source,
            "embedding_kernel",
            {"D": 3072, "BLOCK_M": 1, "BLOCK_D": 4096},
        )
        assert result.proved, [
            f"{c.name}: {c.details}" for c in result.checks if not c.proved
        ]
        assert any(
            check.name == "o:tensor_region_equiv_weight"
            for check in result.checks
        )


# ---------------------------------------------------------------------------
# Launch configuration verification: prove all candidate configs
# ---------------------------------------------------------------------------


def test_empty_proof_result_fails_closed() -> None:
    from ir.smt import ProofCheck, ProofResult

    assert not ProofResult(checks=[]).ok
    deferred = ProofResult(
        checks=[ProofCheck("regional_fragment", True, "proved")],
        deferred_obligations=("discrete_assignment_scalar_0",),
    )
    assert deferred.checks_ok
    assert not deferred.ok


class TestLaunchConfigurationVerification:
    def _read_source(self, name: str) -> str:
        with open(f"triton_kernels/{name}") as f:
            return f.read()

    def _verify_matmul_configs(self, source_file: str, kernel_name: str):
        from triton_kernels.matmul import CONFIGS
        from ir.relational_dataflow import prove_relational_dataflow_from_annotations

        source = self._read_source(source_file)
        for cfg in CONFIGS:
            constants = {k: v for k, v in cfg.items() if not k.startswith("num_")}
            result = prove_relational_dataflow_from_annotations(source, kernel_name, constants)
            label = ", ".join(f"{k}={v}" for k, v in constants.items())
            assert result.proved, (
                f"{kernel_name}({label}) failed: "
                + str([f"{c.name}: {c.details}" for c in result.checks if not c.proved])
            )

    def test_matmul_all_launch_configs(self):
        self._verify_matmul_configs("matmul.py", "matmul_kernel")

    def test_fattn_paged_all_launch_configs(self):
        from triton_kernels.fattn_paged import CONFIGS, PAGE_SIZE
        from ir.relational_dataflow import prove_relational_dataflow_from_annotations

        source = self._read_source("fattn_paged.py")
        page_block_size = PAGE_SIZE
        for cfg in CONFIGS:
            constants = {k: v for k, v in cfg.items() if not k.startswith("num_")}
            if page_block_size % constants["BLOCK_N"] != 0:
                continue
            full_constants = {
                **constants,
                "D_HEAD": 64,
                "PAGE_BLOCK_SIZE": page_block_size,
            }
            result = prove_relational_dataflow_from_annotations(
                source, "fattn_varlen_paged_fwd_block_ptr_kernel", full_constants,
                goal_name="batch_invariance",
            )
            label = ", ".join(f"{k}={v}" for k, v in full_constants.items())
            assert result.proved, (
                f"fattn_varlen_paged_fwd_block_ptr_kernel({label}) failed: "
                + str([f"{c.name}: {c.details}" for c in result.checks if not c.proved])
            )


# ---------------------------------------------------------------------------
# Fix 1: Missing commas between conditions must be rejected
# ---------------------------------------------------------------------------


class TestMissingCommaRejection:
    def test_missing_comma_two_conditions(self):
        """Two conditions without a comma → ParseError."""
        with pytest.raises(ParseError, match="is a comma missing"):
            parse_annotation_text("left(M) > 0 N > 0")

    def test_missing_comma_after_several(self):
        """Missing comma before last condition → ParseError."""
        with pytest.raises(ParseError, match="is a comma missing"):
            parse_annotation_text("left(M) > 0, N > 0, K > 0 H > 0")

    def test_valid_comma_separated(self):
        """Properly comma-separated conditions still work."""
        conds = parse_annotation_text("left(M) > 0, N > 0")
        assert len(conds) == 2

    def test_trailing_comma_still_ok(self):
        """Trailing comma is still accepted."""
        conds = parse_annotation_text("left(M) > 0,")
        assert len(conds) == 1


# ---------------------------------------------------------------------------
# Fix 2: Bare kernel params not in @same must be rejected
# ---------------------------------------------------------------------------


class TestBareKernelParamRejection:
    @staticmethod
    def _make_kernel(param_names_and_types):
        """Helper to build a Kernel with proper IR types."""
        from ir import Kernel, Param, IntType, FloatType, TensorType, Var, Grid
        params = []
        for name, kind in param_names_and_types:
            if kind == "int":
                params.append(Param(name, IntType()))
            elif kind == "float":
                params.append(Param(name, FloatType()))
            elif kind == "tensor2d":
                d0, d1 = Var(f"{name}_d0"), Var(f"{name}_d1")
                params.append(Param(name, TensorType(FloatType(), [d0, d1])))
            else:
                raise ValueError(kind)
        return Kernel(name="test_kernel", params=params, grid=Grid(iters=[], decls=[], body=[]))

    def test_bare_kernel_param_rejected(self):
        """A kernel param used bare (not in @same) should raise ValueError."""
        from ir.annotation_to_config import build_config_from_annotation

        kernel = self._make_kernel([
            ("M", "int"), ("N", "int"),
            ("a", "tensor2d"), ("c", "tensor2d"),
        ])
        # M appears bare — not in @same, not specialized
        annotation = parse_verif_goal(
            "# @verif(test,\n"
            "#   pre(\n"
            "#   right(N) == 1,\n"
            "#   M > 0,\n"
            "#   left(a)[0:1, 0:left(N)] == right(a)[0:1, 0:right(N)],\n"
            "#   ),\n"
            "#   post(\n"
            "#   left(c)[0:1, 0:left(N)] == right(c)[0:1, 0:right(N)]\n"
            "#   ),\n"
            "# )\n"
        )
        assert annotation is not None
        with pytest.raises(ValueError, match="bare"):
            build_config_from_annotation(kernel, annotation)

    def test_free_var_non_param_still_works(self):
        """Free vars that aren't kernel params (e.g., batch index x) are fine."""
        from ir.annotation_to_config import build_config_from_annotation

        kernel = self._make_kernel([
            ("M", "int"), ("a", "tensor2d"), ("c", "tensor2d"),
        ])
        annotation = parse_verif_goal(
            "# @verif(test,\n"
            "#   pre(\n"
            "#   right(M) == 1,\n"
            "#   x >= 0,\n"
            "#   left(a)[x:x+1, 0:left(M)] == right(a)[0:1, 0:right(M)],\n"
            "#   ),\n"
            "#   post(\n"
            "#   left(c)[x:x+1, 0:left(M)] == right(c)[0:1, 0:right(M)]\n"
            "#   ),\n"
            "# )\n"
        )
        assert annotation is not None
        # Should not raise — x is not a kernel param
        config = build_config_from_annotation(kernel, annotation)
        assert config is not None

    def test_unmentioned_symbolic_dimensions_receive_side_specific_bindings(self):
        """Typed-IR shape symbols remain encodable without extra @pre clauses."""
        from ir.annotation_to_config import build_config_from_annotation

        kernel = self._make_kernel([
            ("a", "tensor2d"), ("c", "tensor2d"),
        ])
        annotation = parse_verif_goal(
            "# @verif(test,\n"
            "#   pre(\n"
            "#   left(a)[0:1, 0:1] == right(a)[0:1, 0:1]\n"
            "#   ),\n"
            "#   post(\n"
            "#   left(c)[0:1, 0:1] == right(c)[0:1, 0:1]\n"
            "#   ),\n"
            "# )\n"
        )
        assert annotation is not None
        config = build_config_from_annotation(kernel, annotation)
        assert "a_d0" in config.left_env
        assert "a_d0" in config.right_env
        assert not config.left_env["a_d0"].eq(config.right_env["a_d0"])

    def test_bare_symbolic_dimension_rejected(self):
        """A shape symbol is a side-specific input unless explicitly shared."""
        from ir.annotation_to_config import build_config_from_annotation

        kernel = self._make_kernel([
            ("a", "tensor2d"), ("c", "tensor2d"),
        ])
        annotation = parse_verif_goal(
            "# @verif(test,\n"
            "#   pre(\n"
            "#   a_d0 > 0,\n"
            "#   left(a)[0:1, 0:1] == right(a)[0:1, 0:1]\n"
            "#   ),\n"
            "#   post(\n"
            "#   left(c)[0:1, 0:1] == right(c)[0:1, 0:1]\n"
            "#   ),\n"
            "# )\n"
        )
        assert annotation is not None
        with pytest.raises(ValueError, match="bare"):
            build_config_from_annotation(kernel, annotation)

    def test_quantified_region_binder_is_schema_local(self):
        from ir.annotation_to_config import build_config_from_annotation

        kernel = self._make_kernel([
            ("a", "tensor2d"), ("c", "tensor2d"),
        ])
        annotation = parse_verif_goal(
            "# @verif(test,\n"
            "#   pre(\n"
            "#   forall(tile, implies(and(tile >= 0, tile < 2), "
            "left(a)[tile:tile+1, 0:1] == "
            "right(a)[tile:tile+1, 0:1]))\n"
            "#   ),\n"
            "#   post(\n"
            "#   left(c)[0:1, 0:1] == right(c)[0:1, 0:1]\n"
            "#   ),\n"
            "# )\n"
        )
        assert annotation is not None
        config = build_config_from_annotation(kernel, annotation)
        assumption = config.tensor_assumptions["a"]
        assert assumption.schema_variables == ("tile",)
        assert len(assumption.schema_symbols) == 1
        assert len(assumption.schema_conditions) == 1
        assert "tile" not in config.left_env
        assert "tile" not in config.right_env

    def test_quantified_region_binder_cannot_capture_an_outer_free_name(self):
        from ir.annotation_to_config import build_config_from_annotation

        kernel = self._make_kernel([
            ("a", "tensor2d"), ("c", "tensor2d"),
        ])
        annotation = parse_verif_goal(
            "# @verif(test,\n"
            "#   pre(\n"
            "#   tile >= 0,\n"
            "#   forall(tile, implies(and(tile >= 0, tile < 2), "
            "left(a)[tile:tile+1, 0:1] == "
            "right(a)[tile:tile+1, 0:1]))\n"
            "#   ),\n"
            "#   post(\n"
            "#   left(c)[0:1, 0:1] == right(c)[0:1, 0:1]\n"
            "#   ),\n"
            "# )\n"
        )
        assert annotation is not None
        with pytest.raises(ValueError, match="shadows an existing name"):
            build_config_from_annotation(kernel, annotation)

    def test_specialized_constant_bare_ok(self):
        """Specialized constants used bare are fine (they become concrete IntVals)."""
        from ir.annotation_to_config import build_config_from_annotation

        kernel = self._make_kernel([
            ("M", "int"), ("D", "int"),
            ("a", "tensor2d"), ("c", "tensor2d"),
        ])
        annotation = parse_verif_goal(
            "# @verif(test,\n"
            "#   pre(\n"
            "#   right(M) == 1,\n"
            "#   left(a)[0:1, 0:D] == right(a)[0:1, 0:D],\n"
            "#   ),\n"
            "#   post(\n"
            "#   left(c)[0:1, 0:D] == right(c)[0:1, 0:D]\n"
            "#   ),\n"
            "# )\n"
        )
        assert annotation is not None
        # D=64 is specialized — bare usage is fine
        config = build_config_from_annotation(kernel, annotation, {"D": 64})
        assert config is not None

    def test_same_var_bare_ok(self):
        """@same vars used bare are fine."""
        from ir.annotation_to_config import build_config_from_annotation

        kernel = self._make_kernel([
            ("M", "int"), ("N", "int"),
            ("a", "tensor2d"), ("c", "tensor2d"),
        ])
        annotation = parse_verif_goal(
            "# @verif(test,\n"
            "#   same(N),\n"
            "#   pre(\n"
            "#   right(M) == 1,\n"
            "#   N > 0,\n"
            "#   left(a)[0:1, 0:N] == right(a)[0:1, 0:N],\n"
            "#   ),\n"
            "#   post(\n"
            "#   left(c)[0:1, 0:N] == right(c)[0:1, 0:N]\n"
            "#   ),\n"
            "# )\n"
        )
        assert annotation is not None
        # N is in @same — bare usage is fine (it's parsed as Left, not FreeVar)
        config = build_config_from_annotation(kernel, annotation)
        assert config is not None

    def test_float_param_uses_opaque_value_sort(self):
        from ir.annotation_to_config import build_config_from_annotation

        kernel = self._make_kernel([
            ("scale", "float"), ("a", "tensor2d"), ("c", "tensor2d"),
        ])
        annotation = parse_verif_goal(
            "# @verif(test,\n"
            "#   same(scale),\n"
            "#   pre(\n"
            "#   left(a)[0:1, 0:1] == right(a)[0:1, 0:1]\n"
            "#   ),\n"
            "#   post(\n"
            "#   left(c)[0:1, 0:1] == right(c)[0:1, 0:1]\n"
            "#   ),\n"
            "# )\n"
        )
        assert annotation is not None
        config = build_config_from_annotation(kernel, annotation)
        assert str(config.left_env["scale"].sort()) == "KernelFloatValue"
