"""Explicit run references must survive region-coordinate lowering."""

from pathlib import Path
from dataclasses import replace

import pytest
import z3

from ir import IntType, Param, TensorType, Var
from ir.annotation_to_config import build_config_from_annotation
from ir.annotations import parse_verif_goal
from ir.relational_verifier import verify_annotations
from ir.relational_dataflow import prove_relational_dataflow_from_annotations
from ir.smt import expr_to_z3
from ir.proof_preparation import prepare_annotation_proof


CONSTANTS = {"BLOCK_M": 1, "BLOCK_N": 64}


def source():
    return (Path(__file__).resolve().parents[2] / "triton_kernels/add.py").read_text().replace(
        "right(M) == 1", "right(M) > 0",
    )


def offset_input(text, name):
    return text.replace(f"right({name})[0:1, 0:N]",
                        f"right({name})[left(M)-right(M):left(M)-right(M)+1, 0:N]")


def test_cross_run_coordinate_cannot_be_erased_into_same_run_zero():
    # For M_left=3, M_right=2 this premise equates left x[b] with right
    # x[1], not x[0]. Previously both M references became right(M), and
    # the false row-zero conclusion was accepted.
    report = prove_relational_dataflow_from_annotations(
        offset_input(source(), "x"), "add_kernel", CONSTANTS,
    )
    assert not report.proved
    assert report.verified_contract is None


@pytest.mark.parametrize("collision", [False, True])
def test_aligned_cross_run_regions_prove_and_preserve_explicit_theorem(collision):
    text = source().replace("left(M) > 0, N > 0",
                            "left(M) > right(M), left(M) < 2 * right(M), N > 0")
    for name in ("x", "y", "o"):
        text = offset_input(text, name)
    if collision:
        text = text.replace("b >= 0, b < left(M),",
                            "b >= 0, b < left(M), __annotation_left_M == 999,")
    result = verify_annotations(text, "add_kernel", CONSTANTS)
    assert result.proved
    # Aliases are implementation details of the paired environments. The
    # exported theorem retains exactly the original left/right expression.
    theorem = result.verified_contract.to_data()["theorem_contract"]["theorem"]
    assert "__annotation_left_M_" not in str(theorem)
    post = theorem["post"][0]
    assert post["right"]["slices"][0]["start"]["lhs"]["side"] == "left"
    assert post["right"]["slices"][0]["start"]["rhs"]["side"] == "right"


def test_cross_run_iterator_requires_explicit_shared_logical_index():
    text = source().replace("right(x)[0:1, 0:N]", "right(x)[left(_pid_0):left(_pid_0)+1, 0:N]")
    with pytest.raises(ValueError, match="cross-run iterator coordinates"):
        prepare_annotation_proof(text, "add_kernel", CONSTANTS)


@pytest.mark.parametrize("owner,foreign", [("left", "right"), ("right", "left")])
def test_cross_run_indexed_coordinates_keep_both_function_and_index_sides(owner, foreign):
    prepared = prepare_annotation_proof(source(), "add_kernel", CONSTANTS)
    kernel = replace(prepared.kernel, params=[*prepared.kernel.params,
        Param("indices", TensorType(IntType(), [Var("M")]))])
    old = "left(x)[b:b+1, 0:N]" if owner == "left" else "right(x)[0:1, 0:N]"
    coordinate = f"{foreign}(indices)[{foreign}(M)-1]"
    text = source().replace(old, f"{owner}(x)[{coordinate}:{coordinate}+1, 0:N]")
    goal = parse_verif_goal(text, "add_kernel", "batch_invariance")
    config = build_config_from_annotation(kernel, goal, CONSTANTS)
    owner_env = config.left_env if owner == "left" else config.right_env
    foreign_env = config.right_env if owner == "left" else config.left_env
    relation = config.tensor_assumptions["x"]
    region = relation.left_region if owner == "left" else relation.right_region
    actual = expr_to_z3(region[0].start, owner_env)
    expected = foreign_env["indices"](foreign_env["M"] - 1)
    assert z3.eq(actual, expected)
