from dataclasses import replace
import hashlib
from pathlib import Path

import pytest
import z3

from ir import (
    Assign,
    FloatType,
    For,
    IntLit,
    IntType,
    MaskedLoad,
    MaskedStore,
    Range,
    Slice,
    TensorType,
    Var,
    collect_expr_vars,
)
from ir.relational_dataflow import (
    _loop_carried_values,
    prove_range_difference_neutrality,
    prove_relational_dataflow_from_annotations,
)
from ir.identity_transition import prove_report_requirement_on_context
from ir.proof_preparation import prepare_annotation_proof


ROOT = Path(__file__).resolve().parents[2]


def _source(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def _add_source() -> str:
    """Add a named intermediate to the deployed add kernel for diagnostics."""

    source = _source("triton_kernels/add.py")
    source = source.replace(
        "    o_block_ptr = tl.make_block_ptr(\n",
        "    result = x_block + y_block\n\n"
        "    o_block_ptr = tl.make_block_ptr(\n",
    )
    return source.replace(
        "        (x_block + y_block).to(o.dtype.element_ty),",
        "        result.to(o.dtype.element_ty),",
    )


def test_add_straight_line_goal_is_proved() -> None:
    report = prove_relational_dataflow_from_annotations(
        _add_source(),
        "add_kernel",
        {"BLOCK_M": 1, "BLOCK_N": 64},
    )
    assert report.proved, [
        f"{check.name}: {check.details}" for check in report.checks if not check.proved
    ]
    assert report.required_tensor_reads == ("x", "y")
    assert report.used_assumptions == ()
    assert report.external_obligations == ()
    assert report.unsupported_reason is None
    assert len(report.alignments) == 1
    alignment = report.alignments[0]
    assert alignment.output_tensor == "o"
    assert [(loop.iterator, loop.strategy) for loop in alignment.loop_alignments] == [
        ("_pid_0", "singleton_relevant"),
        ("_pid_1", "same_range"),
    ]
    assert {demand.tensor for demand in alignment.paired_demands} >= {
        "o",
        "x",
        "y",
    }
    assert all(
        demand.coordinates.rank == len(demand.left.region) == len(demand.right.region)
        for demand in alignment.paired_demands
    )
    result_write = next(
        statement
        for statement in alignment.relevant_statements
        if statement.target == "result"
    )
    assert result_write.write.value.op == "+"
    assert (
        result_write.write.value.lhs.name,
        result_write.write.value.rhs.name,
    ) == ("x_block", "y_block")
    assert any(
        check.name == "o:ordered_provenance_alignment" and check.proved
        for check in report.checks
    )


def test_silu_straight_line_goal_is_proved() -> None:
    report = prove_relational_dataflow_from_annotations(
        _source("triton_kernels/silu_mul.py"),
        "silu_mul_kernel",
        {"BLOCK_M": 1, "BLOCK_N": 64},
    )
    assert report.proved, [
        f"{check.name}: {check.details}" for check in report.checks if not check.proved
    ]
    assert report.required_tensor_reads == ("x", "y")


def test_input_relation_order_does_not_assign_semantic_roles() -> None:
    source = _add_source()
    x_relation = "#     left(x)[b:b+1, 0:N] == right(x)[0:1, 0:N],\n"
    y_relation = "#     left(y)[b:b+1, 0:N] == right(y)[0:1, 0:N],\n"
    assert x_relation + y_relation in source
    source = source.replace(x_relation + y_relation, y_relation + x_relation)
    report = prove_relational_dataflow_from_annotations(
        source,
        "add_kernel",
        {"BLOCK_M": 1, "BLOCK_N": 64},
    )
    assert report.proved
    assert report.required_tensor_reads == ("x", "y")


def test_same_kernel_operand_reordering_preserves_relational_alignment() -> None:
    """Static operand order is recorded, but is shared by the two runs."""

    source = _add_source().replace(
        "    result = x_block + y_block\n",
        "    result = y_block + x_block\n",
    )
    report = prove_relational_dataflow_from_annotations(
        source,
        "add_kernel",
        {"BLOCK_M": 1, "BLOCK_N": 64},
    )
    assert report.proved
    result_write = next(
        statement
        for statement in report.alignments[0].relevant_statements
        if statement.target == "result"
    )
    assert (
        result_write.write.value.lhs.name,
        result_write.write.value.rhs.name,
    ) == ("y_block", "x_block")


def test_value_relevant_aligned_branch_is_accounted() -> None:
    source = _add_source().replace(
        "    result = x_block + y_block\n",
        "    result = x_block + y_block\n"
        "    if row < M:\n"
        "        result = result + 1.0\n",
    )
    report = prove_relational_dataflow_from_annotations(
        source,
        "add_kernel",
        {"BLOCK_M": 1, "BLOCK_N": 64},
    )
    assert report.proved
    controls = report.alignments[0].control_alignments
    assert len(controls) == 1
    assert controls[0].predicate.op == "<"
    assert controls[0].proof_check == "value_scalar_equiv_0"


def test_value_relevant_divergent_branch_is_rejected() -> None:
    source = _add_source().replace(
        "    result = x_block + y_block\n",
        "    result = x_block + y_block\n"
        "    if row < 1:\n"
        "        result = result + 1.0\n",
    )
    report = prove_relational_dataflow_from_annotations(
        source,
        "add_kernel",
        {"BLOCK_M": 1, "BLOCK_N": 64},
    )
    assert not report.proved
    assert report.alignments == ()
    assert any(
        check.name.startswith("o:value_scalar_equiv_") and not check.proved
        for check in report.checks
    )


def test_local_variable_rename_does_not_change_proof() -> None:
    source = _add_source()
    source = source.replace(
        "    result = x_block + y_block\n",
        "    combined = x_block + y_block\n",
    ).replace(
        "        result.to(o.dtype.element_ty),",
        "        combined.to(o.dtype.element_ty),",
    )
    report = prove_relational_dataflow_from_annotations(
        source,
        "add_kernel",
        {"BLOCK_M": 1, "BLOCK_N": 64},
    )
    assert report.proved


def test_shifted_input_relation_is_rejected() -> None:
    source = _add_source().replace(
        "right(x)[0:1, 0:N]",
        "right(x)[0:1, 1:N+1]",
    )
    report = prove_relational_dataflow_from_annotations(
        source,
        "add_kernel",
        {"BLOCK_M": 1, "BLOCK_N": 64},
    )
    assert not report.proved
    assert report.alignments == ()
    assert any(
        check.name == "o:tensor_region_equiv_x" and not check.proved
        for check in report.checks
    )


def test_stateless_source_loop_is_an_ordered_map() -> None:
    report = prove_relational_dataflow_from_annotations(
        _source("triton_kernels/embedding.py"),
        "embedding_kernel",
        {"D": 3072, "BLOCK_M": 1, "BLOCK_D": 4096},
    )
    assert report.proved
    source_loops = [
        loop for loop in report.alignments[0].loop_alignments if loop.scope == "source"
    ]
    assert [
        (loop.iterator, loop.strategy, loop.carried_values) for loop in source_loops
    ] == [("r", "same_range_map", ())]


def test_matmul_source_loop_is_an_ordered_accumulator_fold() -> None:
    report = prove_relational_dataflow_from_annotations(
        _source("triton_kernels/matmul.py"),
        "matmul_kernel",
        {"BLOCK_M": 16, "BLOCK_N": 64, "BLOCK_K": 64},
        goal_name="batch_invariance",
    )
    assert report.proved
    source_loops = [
        loop for loop in report.alignments[0].loop_alignments if loop.scope == "source"
    ]
    assert [
        (loop.iterator, loop.strategy, loop.carried_values) for loop in source_loops
    ] == [("k", "same_range_fold", ("acc",))]


def test_guarded_fold_exposes_conditional_identity_as_open_obligation() -> None:
    source = _source("triton_kernels/matmul.py").replace(
        "        acc = tl.dot(block_a, block_b, acc)\n",
        "        next_acc = tl.dot(block_a, block_b, acc)\n"
        "        offsets = tl.arange(0, BLOCK_N)\n"
        "        gate = tl.broadcast_to((offsets < k - 1)[None, :], "
        "(BLOCK_M, BLOCK_N))\n"
        "        acc = tl.where(gate, next_acc, acc)\n",
    )
    report = prove_relational_dataflow_from_annotations(
        source,
        "matmul_kernel",
        {"BLOCK_M": 16, "BLOCK_N": 64, "BLOCK_K": 64},
        goal_name="batch_invariance",
    )

    assert report.proved
    loop = next(
        loop for loop in report.alignments[0].loop_alignments if loop.scope == "source"
    )
    assert loop.identity_state_values == ("acc",)
    assert len(loop.conditional_identities) == 1
    identity = loop.conditional_identities[0]
    assert identity.fact.variable == "gate"
    assert identity.state_values == ("acc",)
    assert identity.left.proved and identity.right.proved
    assert identity.left.external_assumptions == ()
    assert collect_expr_vars(identity.left.fact_facts.false_where.body) == {
        "_i1",
        "k",
    }

    obligation = identity.obligation(loop.iterator, "left")
    assert obligation.iterator == "k"
    assert obligation.execution == "left"
    assert obligation.state_values == ("acc",)
    assert len(obligation.regional_fact_requirements) == 1
    assert obligation.regional_fact_requirements[0].fact.variable == "gate"

    prepared = prepare_annotation_proof(
        source,
        "matmul_kernel",
        {"BLOCK_M": 16, "BLOCK_N": 64, "BLOCK_K": 64},
        goal_name="batch_invariance",
    )
    left_env = prepared.first_config.left_env
    neutral_at_one = prove_report_requirement_on_context(
        identity.left,
        obligation.regional_fact_requirements[0],
        env=left_env,
        assumptions=prepared.first_config.base_assumptions,
        context=left_env["k"] == 1,
        check_name="left_k_one_gate_false",
    )
    assert neutral_at_one.proved, neutral_at_one.details
    not_neutral_at_two = prove_report_requirement_on_context(
        identity.left,
        obligation.regional_fact_requirements[0],
        env=left_env,
        assumptions=prepared.first_config.base_assumptions,
        context=left_env["k"] == 2,
        check_name="left_k_two_gate_false",
    )
    assert not not_neutral_at_two.proved

    source_loop = next(
        loop
        for loop in prepared.kernel.grid.body
        if isinstance(loop, For) and loop.var.name == "k"
    )
    left_range_env = dict(prepared.first_config.left_env)
    right_range_env = dict(prepared.first_config.right_env)
    left_k = z3.Int("test_left_k_extent")
    right_k = z3.Int("test_right_k_extent")
    left_range_env["K"] = left_k
    right_range_env["K"] = right_k
    difference = prove_range_difference_neutrality(
        source_loop,
        identity,
        left_env=left_range_env,
        right_env=right_range_env,
        assumptions=[left_k == 64, right_k == 128],
    )
    assert difference.proved, [
        check.details
        for check in (
            difference.left_exclusive_checks + difference.right_exclusive_checks
        )
        if not check.proved
    ]
    nonneutral_difference = prove_range_difference_neutrality(
        source_loop,
        identity,
        left_env=left_range_env,
        right_env=right_range_env,
        assumptions=[left_k == 64, right_k == 192],
    )
    assert not nonneutral_difference.proved
    fabricated_state_surface = prove_range_difference_neutrality(
        source_loop,
        replace(identity, state_values=("acc", "omitted_live_out")),
        left_env=left_range_env,
        right_env=right_range_env,
        assumptions=[left_k == 64, right_k == 128],
    )
    assert not fabricated_state_surface.proved


def test_unequal_fold_ranges_compose_with_proved_one_sided_identity() -> None:
    """The exported alignment closes, rather than merely exposing, the tail."""

    source = _source("triton_kernels/matmul.py")
    source = source.replace(
        "range(tl.cdiv(K, BLOCK_K))",
        "range(tl.cdiv(M, BLOCK_K))",
    ).replace(
        "        acc = tl.dot(block_a, block_b, acc)\n",
        "        next_acc = tl.dot(block_a, block_b, acc)\n"
        "        offsets = tl.arange(0, BLOCK_N)\n"
        "        gate = tl.broadcast_to((offsets >= k * BLOCK_K)[None, :], "
        "(BLOCK_M, BLOCK_N))\n"
        "        acc = tl.where(gate, next_acc, acc)\n",
    )
    report = prove_relational_dataflow_from_annotations(
        source,
        "matmul_kernel",
        {"BLOCK_M": 16, "BLOCK_N": 64, "BLOCK_K": 64},
        goal_name="batch_invariance",
    )

    assert report.proved, [
        f"{check.name}: {check.details}" for check in report.checks if not check.proved
    ]
    loop = next(
        loop for loop in report.alignments[0].loop_alignments if loop.scope == "source"
    )
    assert loop.strategy == "ordered_common_range_with_identity_difference"
    assert loop.range_difference_neutrality is not None
    assert loop.range_difference_neutrality.proved
    assert loop.range_difference_neutrality.identity.fact.variable == "gate"


def test_unconditional_update_does_not_expose_identity_rule() -> None:
    source = _source("triton_kernels/matmul.py").replace(
        "        acc = tl.dot(block_a, block_b, acc)\n",
        "        next_acc = tl.dot(block_a, block_b, acc)\n"
        "        offsets = tl.arange(0, BLOCK_N)\n"
        "        gate = tl.broadcast_to((offsets < k)[None, :], "
        "(BLOCK_M, BLOCK_N))\n"
        "        acc = next_acc\n",
    )
    report = prove_relational_dataflow_from_annotations(
        source,
        "matmul_kernel",
        {"BLOCK_M": 16, "BLOCK_N": 64, "BLOCK_K": 64},
        goal_name="batch_invariance",
    )

    assert report.proved
    loop = next(
        loop for loop in report.alignments[0].loop_alignments if loop.scope == "source"
    )
    assert loop.identity_state_values == ("acc",)
    assert loop.conditional_identities == ()


def test_conditional_identity_must_cover_non_carried_live_out_state() -> None:
    source = _source("triton_kernels/matmul.py")
    source = (
        source.replace(
            "    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)\n",
            "    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)\n"
            "    last = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)\n",
        )
        .replace(
            "        acc = tl.dot(block_a, block_b, acc)\n",
            "        next_acc = tl.dot(block_a, block_b, acc)\n"
            "        offsets = tl.arange(0, BLOCK_N)\n"
            "        gate = tl.broadcast_to((offsets < k)[None, :], "
            "(BLOCK_M, BLOCK_N))\n"
            "        acc = tl.where(gate, next_acc, acc)\n"
            "        last = block_a\n",
        )
        .replace(
            "    block_ptr_c = tl.make_block_ptr(\n",
            "    acc = acc + last\n\n    block_ptr_c = tl.make_block_ptr(\n",
        )
    )
    report = prove_relational_dataflow_from_annotations(
        source,
        "matmul_kernel",
        {"BLOCK_M": 16, "BLOCK_N": 64, "BLOCK_K": 64},
        goal_name="batch_invariance",
    )

    assert report.proved
    loop = next(
        loop for loop in report.alignments[0].loop_alignments if loop.scope == "source"
    )
    assert loop.carried_values == ("acc",)
    assert loop.live_out_values == ("acc", "last")
    assert loop.identity_state_values == ("acc", "last")
    assert loop.conditional_identities == ()


def test_matmul_fold_rejects_unaligned_iteration_count() -> None:
    source = _source("triton_kernels/matmul.py").replace(
        "range(tl.cdiv(K, BLOCK_K))",
        "range(tl.cdiv(M, BLOCK_K))",
    )
    report = prove_relational_dataflow_from_annotations(
        source,
        "matmul_kernel",
        {"BLOCK_M": 16, "BLOCK_N": 64, "BLOCK_K": 64},
        goal_name="batch_invariance",
    )
    assert not report.proved
    assert report.alignments == ()
    assert any(
        check.name == "c:ordered_provenance_alignment"
        and not check.proved
        and "no conditional identity" in check.details
        for check in report.checks
    )


def test_memory_read_after_partial_loop_store_is_carried_state() -> None:
    """A loop-mutated parameter cannot be misclassified as a stateless map."""

    vec = TensorType(FloatType(), [IntLit(4)])
    full = [Slice(IntLit(0), IntLit(4))]
    cache = Var("cache", type=vec)
    value = Var("value", type=vec)
    loaded = Var("loaded", type=vec)
    loop = For(
        var=Var("i", type=IntType()),
        iters=Range(IntLit(0), IntLit(2)),
        body=[
            MaskedStore(
                base=cache,
                region=full,
                value=value,
                mask=full,
            ),
            Assign(
                target=loaded,
                op=None,
                value=MaskedLoad(
                    base=cache,
                    region=full,
                    mask=full,
                    type=vec,
                ),
            ),
        ],
    )
    assert _loop_carried_values(loop) == frozenset({"cache"})


def test_selected_attention_closes_with_quantified_prefix_and_neutral_tail() -> None:
    source = _source("triton_kernels/fattn_paged.py")
    report = prove_relational_dataflow_from_annotations(
        source,
        "fattn_varlen_paged_fwd_block_ptr_kernel",
        {
            "BLOCK_M": 16,
            "BLOCK_N": 64,
            "D_HEAD": 128,
            "PAGE_BLOCK_SIZE": 64,
        },
        goal_name="selected_row_prefix_equivalence",
    )
    assert report.proved
    assert report.kernel_name == "fattn_varlen_paged_fwd_block_ptr_kernel"
    assert report.goal_name == "selected_row_prefix_equivalence"
    assert report.source_sha256 == hashlib.sha256(source.encode("utf-8")).hexdigest()
    assert dict(report.constants) == {
        "BLOCK_M": 16,
        "BLOCK_N": 64,
        "D_HEAD": 128,
        "PAGE_BLOCK_SIZE": 64,
    }
    assert len(report.theorem_sha256) == 64
    assert report.unsupported_reason is None
    assert report.external_obligations == (
        "finite(v_block)@masked-backward-dependency",
    )
    assert any(
        check.name == "o:iter__pid_2_singleton" and check.proved
        for check in report.checks
    )
    assert any(
        check.name == "o:iter_ki_ordered_shared_ordinal" and check.proved
        for check in report.checks
    )
    assert all(check.proved for check in report.checks)
    assert len(report.alignments) == 1
    alignment = report.alignments[0]
    ki = next(loop for loop in alignment.loop_alignments if loop.iterator == "ki")
    assert ki.strategy == "ordered_common_range_with_identity_difference"
    assert ki.range_difference_neutrality is not None
    assert ki.range_difference_neutrality.proved
    assert ki.range_difference_neutrality.identity.fact.variable == "attn_mask"
    assert {
        (cut.variable, cut.strategy) for cut in alignment.discrete_value_alignments
    } == {
        ("attn_mask", "exact_element_relation"),
        ("row_has_any", "same_operation_congruence"),
    }


@pytest.mark.parametrize("block_m", [16, 32, 64])
def test_sliding_attention_closes_with_neutral_leading_tiles(block_m: int) -> None:
    report = prove_relational_dataflow_from_annotations(
        _source("triton_kernels/fattn_paged_swa.py"),
        "fattn_varlen_paged_swa_kernel",
        {"BLOCK_M": block_m, "BLOCK_N": 64, "D_HEAD": 256, "PAGE_BLOCK_SIZE": 64},
        goal_name="selected_row_prefix_equivalence",
    )
    assert report.proved, report.unsupported_reason
    ki = next(loop for loop in report.alignments[0].loop_alignments if loop.iterator == "ki")
    assert ki.strategy == "ordered_common_range_with_identity_difference"
    assert ki.range_difference_neutrality is not None
    assert ki.range_difference_neutrality.proved


def test_sliding_attention_rejects_skipping_a_partly_visible_tile() -> None:
    source = _source("triton_kernels/fattn_paged_swa.py").replace(
        "window_k_start // BLOCK_N, tl.cdiv(causal_k_end, BLOCK_N)",
        "tl.cdiv(window_k_start, BLOCK_N), tl.cdiv(causal_k_end, BLOCK_N)",
    )
    report = prove_relational_dataflow_from_annotations(
        source, "fattn_varlen_paged_swa_kernel",
        {"BLOCK_M": 16, "BLOCK_N": 64, "D_HEAD": 256, "PAGE_BLOCK_SIZE": 64},
        goal_name="selected_row_prefix_equivalence",
    )
    assert not report.proved
    assert any(check.name == "o:ordered_provenance_alignment" and not check.proved for check in report.checks)


def test_selected_attention_rejects_prefix_schema_missing_final_tile() -> None:
    source = _source("triton_kernels/fattn_paged.py").replace(
        "prefix_tile < cdiv(left(Tk) - left(Tq) + selected_left_row + 1, BLOCK_N)",
        "prefix_tile + 1 < cdiv(left(Tk) - left(Tq) + selected_left_row + 1, BLOCK_N)",
    )
    report = prove_relational_dataflow_from_annotations(
        source,
        "fattn_varlen_paged_fwd_block_ptr_kernel",
        {
            "BLOCK_M": 16,
            "BLOCK_N": 64,
            "D_HEAD": 128,
            "PAGE_BLOCK_SIZE": 64,
        },
        goal_name="selected_row_prefix_equivalence",
    )
    assert not report.proved
    assert {
        "o:tensor_condition_k_cache",
        "o:tensor_region_equiv_k_cache",
        "o:tensor_condition_v_cache",
        "o:tensor_region_equiv_v_cache",
    } <= {check.name for check in report.checks if not check.proved}


def test_selected_attention_rejects_nonidentity_one_sided_iteration() -> None:
    source = _source("triton_kernels/fattn_paged.py").replace(
        "acc = tl.where(row_has_any[:, None], next_acc, acc)",
        "acc = next_acc",
    )
    report = prove_relational_dataflow_from_annotations(
        source,
        "fattn_varlen_paged_fwd_block_ptr_kernel",
        {
            "BLOCK_M": 16,
            "BLOCK_N": 64,
            "D_HEAD": 128,
            "PAGE_BLOCK_SIZE": 64,
        },
        goal_name="selected_row_prefix_equivalence",
    )
    assert not report.proved
    assert any(
        check.name == "o:ordered_provenance_alignment"
        and not check.proved
        and "no conditional identity" in check.details
        for check in report.checks
    )


def test_selected_attention_rejects_query_dependent_key_loop_origin() -> None:
    """The generic gate closes a provenance gap in the specialized checker."""

    source = _source("triton_kernels/fattn_paged.py")
    loop = "for ki in tl.range(tl.cdiv(causal_k_end, BLOCK_N),"
    assert source.count(loop) == 1, "attention loop changed; update the negative fixture"
    source = source.replace(
        loop,
        "for ki in tl.range(q_len, q_len + tl.cdiv(causal_k_end, BLOCK_N),",
    )
    constants = {
        "BLOCK_M": 16,
        "BLOCK_N": 64,
        "D_HEAD": 128,
        "PAGE_BLOCK_SIZE": 64,
    }
    generic = prove_relational_dataflow_from_annotations(
        source,
        "fattn_varlen_paged_fwd_block_ptr_kernel",
        constants,
        goal_name="selected_row_prefix_equivalence",
    )
    assert not generic.proved
    assert any(
        check.name == "o:ordered_provenance_alignment" and not check.proved
        for check in generic.checks
    )



def test_selected_attention_accepts_reordered_independent_state_computations() -> None:
    source = _source("triton_kernels/fattn_paged.py").replace(
        "        next_acc = acc * alpha[:, None]\n"
        "        next_logsum = logsum * alpha + tl.sum(p, axis=1)",
        "        next_logsum = logsum * alpha + tl.sum(p, axis=1)\n"
        "        next_acc = acc * alpha[:, None]",
    )
    constants = {
        "BLOCK_M": 16,
        "BLOCK_N": 64,
        "D_HEAD": 128,
        "PAGE_BLOCK_SIZE": 64,
    }
    generic = prove_relational_dataflow_from_annotations(
        source,
        "fattn_varlen_paged_fwd_block_ptr_kernel",
        constants,
        goal_name="selected_row_prefix_equivalence",
    )
    assert generic.proved
