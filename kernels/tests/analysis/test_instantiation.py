"""Structural matching suggests substitutions; only obligations justify them."""
from pathlib import Path
import re

import pytest
import z3

from ir.instantiation import suggest_instantiations
from ir.relational_dataflow import prove_relational_dataflow_from_annotations
from ir import regional_obligations
from ir.smt import ProofCheck


@pytest.mark.parametrize("names", [("p", "lookup"), ("coordinate", "unrelated_name")])
@pytest.mark.parametrize("block,page", [(64, 64), (32, 128), (96, 64)])
def test_structural_coordinate_has_no_name_or_geometry_special_case(names, block, page):
    p, i, a, b = z3.Ints(f"{names[0]} index source_row destination_row")
    lookup = z3.Function(names[1], z3.IntSort(), z3.IntSort(), z3.IntSort())
    coordinate = i * block / page
    result = suggest_instantiations([lookup(a, p)], [lookup(b, coordinate)], [p])
    assert len(result) == 1 and result[0][0].eq(coordinate)
    # The row difference is deliberately NOT proved by the matcher.
    solver = z3.Solver()
    solver.add(a != b)
    assert solver.check() == z3.sat


def test_multiple_binders_repeated_occurrences_and_stable_deduplication():
    p, q, i, j = z3.Ints("p q i j")
    f = z3.Function("f", *([z3.IntSort()] * 4))
    result = suggest_instantiations([f(p, q, p)], [f(i, j, i), f(i, j, i)], [p, q])
    assert len(result) == 1
    assert all(a.eq(b) for a, b in zip(result[0], (i, j)))
    assert not suggest_instantiations([f(p, q, p)], [f(i, j, j)], [p, q])


def test_other_operators_sorts_and_no_capture():
    p, i = z3.Ints("p i")
    f = z3.Function("f", z3.IntSort(), z3.IntSort())
    g = z3.Function("g", z3.IntSort(), z3.IntSort())
    assert not suggest_instantiations([f(p)], [g(i)], [p])
    assert not suggest_instantiations([f(p)], [f(p + 1)], [p])
    assert not suggest_instantiations([f(p)], [z3.ForAll(i, f(i) > 0)], [p])
    assert not suggest_instantiations([f(p)], [f(z3.Var(0, z3.IntSort()))], [p])
    assert not suggest_instantiations([f(p)], [f(i)], [p, z3.Real("r")])
    assert not suggest_instantiations([p], [i + 1], [p])
    assert not suggest_instantiations([f(p)], [f(i)], [])
    with pytest.raises(ValueError, match="distinct constants"):
        suggest_instantiations([f(p)], [f(i)], [p, p])
    with pytest.raises(ValueError, match="distinct constants"):
        suggest_instantiations([f(p)], [f(i)], [p + 1])
    with pytest.raises(ValueError, match="distinct constants"):
        suggest_instantiations([f(p)], [f(i)], [z3.IntVal(0)])


def _source(domain="i >= 0, i < left(M)", right_row="0:1"):
    source = (Path(__file__).resolve().parents[2] / "triton_kernels/add.py").read_text()
    return source.replace(
        "left(x)[b:b+1, 0:N] == right(x)[0:1, 0:N]",
        f"forall(i, implies(and({domain}), "
        f"left(x)[i:i+1, 0:N] == right(x)[{right_row}, 0:N]))",
    )


@pytest.mark.parametrize("hint", [None, "empty", "incorrect"])
@pytest.mark.parametrize("case", ["valid", "uncovered", "wrong_region"])
def test_hints_cannot_bypass_domain_or_region_proofs(monkeypatch, hint, case):
    if hint is not None:
        suggestions = [] if hint == "empty" else [(z3.IntVal(-1),), (z3.IntVal(123),)]
        monkeypatch.setattr(regional_obligations, "suggest_instantiations", lambda *args: suggestions.copy())
    source = _source(
        domain="i >= 0, i < b" if case == "uncovered" else "i >= 0, i < left(M)",
        right_row="1:2" if case == "wrong_region" else "0:1",
    )
    report = prove_relational_dataflow_from_annotations(source, "add_kernel", dict(BLOCK_M=1, BLOCK_N=64))
    assert (report.verified_contract is not None) == (case == "valid")


def test_unknown_domain_is_not_accepted_or_followed_by_region_query(monkeypatch):
    original = regional_obligations.z3_prove
    queries = []

    def prove(name, *args, **kwargs):
        queries.append(name)
        if name == "tensor_condition_x":
            return ProofCheck(name, False, "unknown")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(regional_obligations, "z3_prove", prove)
    report = prove_relational_dataflow_from_annotations(_source(), "add_kernel", dict(BLOCK_M=1, BLOCK_N=64))
    assert report.verified_contract is None
    assert "tensor_condition_x" in queries
    assert "tensor_region_equiv_x" not in queries


@pytest.mark.parametrize("block,page", [(32, 128), (64, 256)])
def test_quantified_indirect_regions_with_unequal_tiles_and_renamed_symbols(block, page):
    source = (Path(__file__).resolve().parents[2] / "triton_kernels/fattn_paged.py").read_text()
    names = dict(block_table="mapping", cache_page="ordinal", ki="iteration",
                 fattn_varlen_paged_fwd_block_ptr_kernel="renamed_kernel")
    source = re.sub(r"\b(" + "|".join(names) + r")\b", lambda m: names[m[0]], source)
    report = prove_relational_dataflow_from_annotations(
        source, "renamed_kernel",
        dict(BLOCK_M=16, BLOCK_N=block, D_HEAD=128, PAGE_BLOCK_SIZE=page),
        goal_name="batch_invariance",
    )
    assert report.verified_contract is not None, [
        (check.name, check.details) for check in report.checks if not check.proved
    ]
