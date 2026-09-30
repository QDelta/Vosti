"""The selected-row annotation covers the logical prefix, not page padding."""

from pathlib import Path

import pytest
import z3

from ir.annotation_to_config import ann_expr_to_z3
from ir.annotations import parse_verif_goal
from ir.annotation_lowering import GuardedRegionEquality, lower_proof_goal
from ir.relational_dataflow import prove_relational_dataflow_from_annotations


ROOT = Path(__file__).resolve().parents[2] / "triton_kernels"
CASES = [
    ("fattn_paged.py", "fattn_varlen_paged_fwd_block_ptr_kernel"),
    ("fattn_paged_swa.py", "fattn_varlen_paged_swa_kernel"),
]
CONSTANTS = {"BLOCK_M": 16, "BLOCK_N": 64, "D_HEAD": 128, "PAGE_BLOCK_SIZE": 64}


@pytest.mark.parametrize("filename,kernel", CASES)
def test_annotation_cache_regions_are_exactly_the_logical_prefix(filename, kernel):
    goal = parse_verif_goal((ROOT / filename).read_text(), kernel, "selected_row_prefix_equivalence")
    relations = [item for item in lower_proof_goal(goal).pre_conditions if isinstance(item, GuardedRegionEquality)]
    assert len(relations) == 2
    # Symbolically check partial-page coverage, including e=0, page endpoints,
    # and arbitrary longer prefixes. The address-map dimension is separate.
    e, tile, lane = z3.Ints("effective_position tile lane")
    left = {**{name: z3.IntVal(value) for name, value in CONSTANTS.items()},
            "Tk": e + 1, "Tq": z3.IntVal(1), "selected_left_row": z3.IntVal(0),
            "selected_right_row": z3.IntVal(0), "prefix_tile": tile}
    right = dict(left)
    for item in relations:
        for region in (item.relation.left, item.relation.right):
            interval = region.slices[1]
            start = ann_expr_to_z3(interval.start, left, right)
            stop = ann_expr_to_z3(interval.stop, left, right)
            solver = z3.Solver()
            solver.add(e >= 0, tile >= 0, tile * 64 <= e, lane >= 0)
            solver.add(z3.And(lane >= start, lane < stop) !=
                       z3.And(lane < 64, tile * 64 + lane <= e))
            assert solver.check() == z3.unsat


@pytest.mark.parametrize("filename,kernel", CASES)
def test_generic_proof_rejects_reading_one_token_beyond_the_prefix(filename, kernel):
    source = (ROOT / filename).read_text()
    if filename == "fattn_paged.py":
        old, new = "<= (q_indices[:, None] + q_shift)", "<= (q_indices[:, None] + q_shift + 1)"
    else:
        old, new = "q_indices[:, None] + q_shift\n        )", "q_indices[:, None] + q_shift + 1\n        )"
    assert source.count(old) == 1
    report = prove_relational_dataflow_from_annotations(
        source.replace(old, new), kernel, CONSTANTS,
        goal_name="selected_row_prefix_equivalence",
    )
    assert not report.proved
    assert report.verified_contract is None
    assert any(not check.proved and "tensor" in check.name for check in report.checks)
