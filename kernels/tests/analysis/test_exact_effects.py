"""Exact effects are separate from relational equality and fail closed."""
from pathlib import Path

import pytest

from ir.annotations import After, Before, DTypeOf, parse_annotation_text, parse_verif_goal
from ir.annotation_lowering import UnsupportedProposition, lower_proof_goal
from ir.exact_effects import verify_exact_effects

ROOT = Path(__file__).resolve().parents[2]
SOURCE = (ROOT / "triton_kernels/store_kv_cache.py").read_text()
GOAL = SOURCE[SOURCE.index("# @verif(exact_effect,"):
              SOURCE.index("# @kernel-bridge-begin store_kv_cache::store_cache_kernel")]


def annotated(source=SOURCE, goal=GOAL):
    assert source.count(GOAL) == 1
    return source.replace(GOAL, goal)


def verify(source=None, width=512):
    return verify_exact_effects(annotated() if source is None else source,
                               "store_cache_kernel", {"KVD": width, "BLOCK_M": 1})


@pytest.mark.parametrize("width", [512, 1024, 2048, 4096])
def test_real_scatter_copy_and_frame(width):
    report = verify(width=width)
    assert report.proved, (report.unsupported_reason, [(c.name, c.details) for c in report.checks if not c.proved])
    assert report.written_tensors == ("cache",)
    names = [c.name for c in report.checks]
    assert any(n.startswith("copy_coverage") for n in names)
    assert any(n.startswith("all_writers_copy") for n in names)
    assert any(n.startswith("frame_no_write") for n in names)
    assert any(n.startswith("write_disjoint") for n in names)


@pytest.mark.parametrize("mutation", ["arithmetic", "narrowing", "wrong_slot", "missing_row", "missing_dtype", "missing_unique", "wrong_source", "false_premise", "wrong_frame"])
def test_invalid_effects_rejected(mutation):
    source, goal = SOURCE, GOAL
    if mutation == "arithmetic":
        source = source.replace("x_row.to(cache.dtype.element_ty)", "(x_row + 1.0).to(cache.dtype.element_ty)")
    elif mutation == "narrowing":
        source = source.replace("x_row.to(cache.dtype.element_ty)", "x_row.to(tl.float32).to(cache.dtype.element_ty)")
    elif mutation == "wrong_slot":
        source = source.replace("offsets=(s, 0)", "offsets=(s + 1, 0)")
    elif mutation == "missing_row":
        source = source.replace("        tl.store(cache_ptr,", "        if cur_row > 0:\n            tl.store(cache_ptr,")
    elif mutation == "missing_dtype":
        goal = goal.replace("#     dtype(x) == dtype(cache),\n", "")
    elif mutation == "missing_unique":
        a = goal.index("#     forall(i, j,")
        b = goal.index("#   ),", a)
        goal = goal[:a] + goal[b:]
    elif mutation == "wrong_source":
        source = source.replace("offsets=(cur_row, 0)", "offsets=(cur_row + 1, 0)")
    elif mutation == "false_premise":
        goal = goal.replace("M >= 0,", "M < 0,")
    else:
        goal = goal.replace("before(cache)[s:s+1, 0:KVD]", "before(cache)[0:1, 0:KVD]")
    assert source != SOURCE or goal != GOAL
    report = verify(annotated(source, goal))
    assert not report.proved


def test_no_kernel_or_parameter_name_dispatch():
    import re
    source = annotated()
    for before, after in (("store_cache_kernel", "scatter_rows"), ("cache", "output_state"),
                          ("slot_mapping", "destinations"), ("x", "values")):
        source = re.sub(r"\b" + before + r"\b", after, source)
    report = verify_exact_effects(source, "scatter_rows", {"KVD": 512, "BLOCK_M": 1})
    assert report.proved, (report.unsupported_reason, [(c.name, c.details) for c in report.checks if not c.proved])


VECTOR_SOURCE = '''import triton
import triton.language as tl
# @params(
#   tensor(values, float, shape(N), strides(sv)),
#   tensor(result, float, shape(N), strides(sr)),
# )
# @grid(cdiv(N, BLOCK))
# @verif(exact_effect,
#   pre(N > 0, dtype(values) == dtype(result)),
#   post(after(result)[0:N] == before(values)[0:N]),
# )
@triton.jit
def vector_copy(values, result, N, sv, sr, BLOCK: tl.constexpr):
    start = tl.program_id(0) * BLOCK
    src = tl.make_block_ptr(values, shape=(N,), strides=(sv,), offsets=(start,), block_shape=(BLOCK,), order=(0,))
    dst = tl.make_block_ptr(result, shape=(N,), strides=(sr,), offsets=(start,), block_shape=(BLOCK,), order=(0,))
    value = tl.load(src, boundary_check=(0,), padding_option="zero")
    tl.store(dst, value, boundary_check=(0,))
'''


@pytest.mark.parametrize("block,drop_tail", [(1, False), (8, False), (8, True)])
def test_independent_vector_copy_with_tail(block, drop_tail):
    source = VECTOR_SOURCE
    if drop_tail:
        source = source.replace("# @grid(cdiv(N, BLOCK))", "# @grid(cdiv(sub(N, 1), BLOCK))")
    report = verify_exact_effects(source, "vector_copy", {"BLOCK": block})
    assert report.proved == (not drop_tail), (report.unsupported_reason, [(c.name, c.details) for c in report.checks if not c.proved])
    if drop_tail:
        assert report.unsupported_reason is None
        assert any(c.name.startswith("copy_coverage") and not c.proved for c in report.checks)


@pytest.mark.parametrize("out_of_bounds_empty", [False, True])
def test_region_endpoint_bounds_also_apply_when_empty(out_of_bounds_empty):
    source = VECTOR_SOURCE.replace("pre(N > 0,", "pre(N >= 0,")
    if out_of_bounds_empty:
        # N == 0 gives [-1:-1]; N > 0 gives the valid full [0:N] region.
        # Per-cell bounds/coverage would miss the out-of-bounds empty case.
        source = source.replace("[0:N]", "[min(0, sub(N, 1)):add(min(0, sub(N, 1)), N)]")
    report = verify_exact_effects(source, "vector_copy", {"BLOCK": 1})
    assert report.proved == (not out_of_bounds_empty), report.unsupported_reason
    if out_of_bounds_empty:
        assert any(c.name.startswith("region_bounds") and not c.proved for c in report.checks)


def test_reaching_definition_is_a_snapshot_not_a_later_assignment():
    source = SOURCE.replace("        tl.store(cache_ptr, x_row.to(cache.dtype.element_ty), boundary_check=(0, 1))",
        "        saved = x_row\n        x_row = x_row + 1.0\n"
        "        tl.store(cache_ptr, saved.to(cache.dtype.element_ty), boundary_check=(0, 1))")
    report = verify(annotated(source))
    assert report.proved, report.unsupported_reason


def test_duplicate_writes_are_rejected_even_if_values_agree():
    line = "        tl.store(cache_ptr, x_row.to(cache.dtype.element_ty), boundary_check=(0, 1))"
    report = verify(annotated(SOURCE.replace(line, line + "\n" + line)))
    assert not report.proved
    assert any(c.name.startswith("write_disjoint") and not c.proved for c in report.checks)


def test_unmasked_sentinel_write_is_not_a_valid_frame():
    source = SOURCE.replace("tl.store(cache_ptr, x_row.to(cache.dtype.element_ty), boundary_check=(0, 1))",
                            "tl.store(cache_ptr, x_row.to(cache.dtype.element_ty))")
    report = verify(annotated(source))
    assert not report.proved


def test_unused_mutable_state_read_is_not_accepted_as_order_independent():
    source = SOURCE.replace("        tl.store(cache_ptr,", "        unused = tl.load(cache_ptr, boundary_check=(0, 1), padding_option=\"zero\")\n        tl.store(cache_ptr,")
    report = verify(annotated(source))
    assert not report.proved
    assert "mutable-state" in report.unsupported_reason


def test_temporal_regions_remain_composable_typed_props():
    dtype, region = parse_annotation_text("dtype(x) == dtype(cache), after(cache)[0:1] == before(x)[0:1]")
    assert isinstance(dtype.lhs, DTypeOf)
    assert isinstance(region.left.side, After) and isinstance(region.right.side, Before)


@pytest.mark.parametrize("prop", ["after(cache)[0:1] == before(x)[0:1]", "dtype(x) == dtype(cache)",
                                    "before(slots)[0] == 0"])
def test_relational_lowerer_cannot_erase_effect_terms(prop):
    goal = parse_verif_goal(f"# @verif(test, pre({prop}), post(left(o)[0:1] == right(o)[0:1]))")
    with pytest.raises(UnsupportedProposition, match="exact-effect verifier"):
        lower_proof_goal(goal)


@pytest.mark.parametrize("premise", ["dtype(x) < dtype(cache)", "dtype(x) == 1", "dtype(missing) == dtype(x)",
                                       "before(slot_mapping)[M] == 0", "forall(M, M == 1)"])
def test_bad_types_scopes_and_metadata_bounds_fail(premise):
    report = verify(annotated(goal=GOAL.replace("#   pre(\n", f"#   pre(\n#     {premise},\n")))
    assert not report.proved
