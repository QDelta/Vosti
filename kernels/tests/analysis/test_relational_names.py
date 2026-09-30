"""The generic proof must follow annotations and IR, not attention names."""

from pathlib import Path
import re

import pytest

from ir.relational_dataflow import prove_relational_dataflow_from_annotations


@pytest.mark.parametrize("filename,entry", [
    ("fattn_paged.py", "fattn_varlen_paged_fwd_block_ptr_kernel"),
    ("fattn_paged_swa.py", "fattn_varlen_paged_swa_kernel"),
])
def test_selected_region_goal_survives_kernel_tensor_state_and_tile_renaming(filename, entry):
    source = (Path(__file__).resolve().parents[2] / "triton_kernels" / filename).read_text()
    renames = {
        entry: "masked_recurrent_operator",
        "selected_row_prefix_equivalence": "chosen_output_regions_equal",
        "q": "queries", "k_cache": "key_storage", "v_cache": "value_storage",
        "block_table": "address_map", "o": "result_storage",
        "attn_mask": "visible_positions", "row_has_any": "has_contribution",
        "acc": "weighted_state", "scores_max": "maximum_state",
        "logsum": "normalizer_state", "v_block": "loaded_values",
        "ki": "fold_step", "prefix_tile": "assumption_index",
        "BLOCK_M": "ROW_TILE", "BLOCK_N": "COLUMN_TILE",
        "selected_left_row": "left_selection", "selected_right_row": "right_selection",
    }
    source = re.sub(r"\b(?:" + "|".join(map(re.escape, renames)) + r")\b",
                    lambda match: renames[match.group()], source)
    report = prove_relational_dataflow_from_annotations(
        source, "masked_recurrent_operator",
        {"ROW_TILE": 16, "COLUMN_TILE": 64, "D_HEAD": 128, "PAGE_BLOCK_SIZE": 64},
        goal_name="chosen_output_regions_equal",
    )
    assert report.proved, [(check.name, check.details) for check in report.checks if not check.proved]
    assert report.verified_contract is not None
    assert report.alignments[0].output_tensor == "result_storage"
    assert report.external_obligations == ("finite(loaded_values)@masked-backward-dependency",)
    fold = next(item for item in report.alignments[0].loop_alignments if item.iterator == "fold_step")
    assert fold.range_difference_neutrality.proved
    assert fold.range_difference_neutrality.identity.fact.variable == "visible_positions"
