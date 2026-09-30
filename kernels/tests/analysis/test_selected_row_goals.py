"""Selected-row goals use the same artifact and rules as every other relation."""

import hashlib
from pathlib import Path

import pytest

from ir.annotations import parse_verif_goal
from ir.relational_contract import relational_tensor_surface
from ir.relational_verifier import verify_annotations


KERNEL = "fattn_varlen_paged_fwd_block_ptr_kernel"
CONSTANTS = dict(D_HEAD=128, BLOCK_M=16, BLOCK_N=64, PAGE_BLOCK_SIZE=64)
GOAL = "selected_row_prefix_equivalence"


def _source():
    return (Path(__file__).resolve().parents[2] / "triton_kernels/fattn_paged.py").read_text()


def _verify(source, constants=CONSTANTS):
    return verify_annotations(source, KERNEL, constants, goal_name=GOAL,
                              preserve_analyzer_conditions=True)


def test_selected_goal_exports_only_the_common_typed_artifact():
    source = _source()
    report = _verify(source)
    data = report.verified_contract.to_data()
    assert set(data) == {"schema_version", "theorem_contract", "evidence", "annotation_preconditions_satisfiable"}
    raw = data["theorem_contract"]
    assert raw["proof_kind"] == "relational_dataflow"
    assert raw["kernel"] == KERNEL and raw["goal_name"] == GOAL
    assert raw["source_sha256"] == hashlib.sha256(source.encode()).hexdigest()
    assert report.required_tensor_reads == ("k_cache", "q", "v_cache")
    assert tuple(a.output_tensor for a in report.alignments) == ("o",)
    assert report.used_assumptions == report.external_obligations == ("finite(v_block)@masked-backward-dependency",)


def test_selected_artifact_is_canonical_over_constant_order():
    first = _verify(_source())
    second = _verify(_source(), dict(reversed(list(CONSTANTS.items()))))
    assert first.verified_contract == second.verified_contract


def test_selected_inputs_have_no_positional_semantic_roles():
    source = _source()
    q_line = next(line for line in source.splitlines() if line.startswith("#     left(q)[selected_left_row:"))
    prefix = [line for line in source.splitlines() if line.startswith("#     forall(prefix_tile")]
    assert len(prefix) == 2
    old = "\n".join((q_line, *prefix))
    assert old in source
    reordered = source.replace(old, "\n".join((prefix[1], q_line, prefix[0])), 1)
    assert relational_tensor_surface(parse_verif_goal(source, KERNEL, GOAL)) == relational_tensor_surface(
        parse_verif_goal(reordered, KERNEL, GOAL))
    assert _verify(reordered).required_tensor_reads == ("k_cache", "q", "v_cache")


@pytest.mark.parametrize("old,new", [
    ("q_shift = k_len - q_len", "q_shift = k_len - q_len + 1"),
    ("right(o)[selected_right_row:selected_right_row+1, 0:H, 0:D_HEAD]",
     "right(o)[selected_right_row+1:selected_right_row+2, 0:H, 0:D_HEAD]"),
    ("@verif(selected_row_prefix_equivalence,", "@verif(a_different_goal,"),
    ("left(v_cache)[left(block_table)[0, prefix_tile", "left(q)[left(block_table)[0, prefix_tile"),
])
def test_invalid_source_relation_cannot_export_a_qualified_artifact(old, new):
    source = _source()
    assert old in source
    with pytest.raises(ValueError):
        _verify(source.replace(old, new, 1))
