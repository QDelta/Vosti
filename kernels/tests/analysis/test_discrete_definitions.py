"""Every consumed version must align, not merely the last variable definition."""

import json
from pathlib import Path

import pytest

from ir import Assign, BoolLit, For, Grid, If, IntLit, Kernel, Range, Var
from ir.reaching_definitions import reaching_definitions
from ir.relational_artifact import VerifiedDataflowContract
from ir.relational_dataflow import prove_relational_dataflow_from_annotations


def _source(first: str, second: str, *, kind: str = "bool") -> str:
    source = (Path(__file__).resolve().parents[2] / "triton_kernels/add.py").read_text()
    def value(dimension):
        op = "<" if kind == "bool" else "+"
        return f"tl.broadcast_to((tl.arange(0, BLOCK_N) {op} {dimension})[None, :], (BLOCK_M, BLOCK_N))"
    marker = "    o_block_ptr = tl.make_block_ptr(\n"
    assert source.count(marker) == 1
    source = source.replace(marker,
        f"    cut = {value(first)}\n"
        "    first = x_block + cut.to(tl.float32)\n"
        f"    cut = {value(second)}\n"
        "    result = first + cut.to(tl.float32) + y_block\n" + marker)
    return source.replace("        (x_block + y_block).to(o.dtype.element_ty),",
                          "        result.to(o.dtype.element_ty),")


def _prove(source):
    return prove_relational_dataflow_from_annotations(
        source, "add_kernel", {"BLOCK_M": 1, "BLOCK_N": 64},
    )


@pytest.mark.parametrize("kind", ["bool", "int"])
@pytest.mark.parametrize("first,second", [("M", "N"), ("N", "M")])
def test_unaligned_definition_cannot_be_hidden_by_reuse(kind, first, second):
    # This is a concrete witness satisfying the annotation: b=0, left M=2,
    # right M=1, shared N=64, and equal zero-filled x/y rows. Lane 1 differs.
    def lane(m):
        dimensions = {"M": m, "N": 64}
        if kind == "bool":
            return float(1 < dimensions[first]) + float(1 < dimensions[second])
        return float(1 + dimensions[first]) + float(1 + dimensions[second])
    assert lane(2) != lane(1)
    report = _prove(_source(first, second, kind=kind))
    assert not report.proved
    assert report.verified_contract is None
    assert any("exact_element_relation" in check.name and not check.proved
               for check in report.checks)


@pytest.fixture(scope="module", params=["bool", "int"])
def aligned(request):
    report = _prove(_source("N", "N + 1", kind=request.param))
    assert report.proved and report.verified_contract is not None
    return report


def test_valid_reuse_records_every_relevant_definition(aligned):
    alignment, = aligned.alignments
    cuts = [cut for cut in alignment.discrete_value_alignments if cut.variable == "cut"]
    assert len(cuts) == 2
    assert {cut.statement_ordinal for cut in cuts} == {
        statement.ordinal for statement in alignment.relevant_statements
        if statement.target == "cut"
    }
    for cut in cuts:
        assert all(name.startswith(f"definition_{cut.statement_ordinal}:")
                   for name in cut.proof_checks)
    assert aligned.verified_contract.to_data()["schema_version"] == 3


def test_congruence_uses_operand_version_at_the_reduction():
    source = _source("M", "N", kind="int").replace(
        "    first = x_block + cut.to(tl.float32)\n",
        "    count = tl.sum(cut, axis=1)\n"
        "    first = x_block + count[:, None].to(tl.float32)\n",
    )
    report = _prove(source)
    assert not report.proved and report.verified_contract is None


def _write(name="v", value=1):
    return Assign(Var(name), None, IntLit(value))


def _reaching(statements):
    return reaching_definitions(Kernel("metadata", [], Grid([], [], statements)))


def test_reaching_definitions_keep_earlier_consumers_distinct():
    first, use_first, second, use_second = _write(), _write("use1"), _write(), _write("use2")
    before = _reaching([first, use_first, second, use_second])
    assert before[id(use_first)]["v"] == {id(first)}
    assert before[id(use_second)]["v"] == {id(second)}


def test_reaching_definitions_join_conditional_overwrites():
    initial, replacement, after = _write(), _write(), _write("after")
    branch = If(BoolLit(True), [replacement], [])
    before = _reaching([initial, branch, after])
    assert before[id(after)]["v"] == {id(initial), id(replacement)}
    before = _reaching([branch, after])
    assert before[id(after)]["v"] == {0, id(replacement)}


def test_reaching_definitions_include_loop_back_edges_and_zero_iterations():
    initial, use, replacement, after = _write(), _write("use"), _write(), _write("after")
    loop = For(Var("i"), Range(IntLit(0), IntLit(2)), [use, replacement])
    before = _reaching([initial, loop, after])
    assert before[id(use)]["v"] == {id(initial), id(replacement)}
    assert before[id(after)]["v"] == {id(initial), id(replacement)}
    loop = For(Var("i"), Range(IntLit(0), IntLit(2)), [replacement, use])
    before = _reaching([initial, loop, after])
    assert before[id(use)]["v"] == {id(replacement)}
    assert before[id(after)]["v"] == {id(initial), id(replacement)}


@pytest.mark.parametrize("change", ["missing", "all_missing", "wrong_statement", "duplicate", "unknown_check", "other_definition"])
def test_artifact_rejects_incomplete_or_misbound_definition_evidence(aligned, change):
    data = aligned.verified_contract.to_data()
    cuts = data["evidence"]["alignments"][0]["discrete_value_alignments"]
    assert len(cuts) == 2
    if change == "missing":
        cuts.pop()
    elif change == "all_missing":
        cuts.clear()
    elif change == "wrong_statement":
        cuts[0]["statement_ordinal"] = 0
    elif change == "duplicate":
        cuts.append(cuts[0])
    elif change == "other_definition":
        cuts[0]["proof_checks"] = cuts[1]["proof_checks"]
    else:
        cuts[0]["proof_checks"] = ["not_proved"]
    artifact = VerifiedDataflowContract(json.dumps(data, sort_keys=True, separators=(",", ":")))
    with pytest.raises(ValueError, match="discrete definition"):
        artifact.to_data()
