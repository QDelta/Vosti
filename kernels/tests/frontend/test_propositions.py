"""One proposition language, with an explicit fail-closed proof fragment."""

from pathlib import Path

import pytest

from ir.annotations import (
    AnnAnd, AnnComparison, AnnImplies, AnnNot, AnnOr, ForAllConstraint,
    ParseError, Prop, RegionEquiv, parse_annotation_text, parse_verif_goal,
)
from ir.annotation_lowering import (
    GuardedRegionEquality, UnsupportedProposition, lower_proof_goal,
)
from ir.relational_contract import relational_theorem_data
from ir.relational_verifier import verify_annotations


REGION = "left(x)[i:i+1] == right(x)[i:i+1]"
OUTPUT = "left(o)[0:1] == right(o)[0:1]"


def goal(pre, post=OUTPUT):
    return parse_verif_goal(f"# @verif(test, pre({pre}), post({post}))")


@pytest.mark.parametrize("text,typ", [
    ("i == 0", AnnComparison), (REGION, RegionEquiv),
    (f"and(i == 0, {REGION})", AnnAnd),
    (f"or(i == 0, {REGION})", AnnOr),
    (f"not({REGION})", AnnNot),
    (f"implies({REGION}, i == 0)", AnnImplies),
    (f"forall(i, implies(i >= 0, {REGION}))", ForAllConstraint),
    (f"forall(i, forall(j, {REGION}))", ForAllConstraint),
])
def test_composable_prop_ast(text, typ):
    prop, = parse_annotation_text(text)
    assert isinstance(prop, Prop) and isinstance(prop, typ)


def test_scalar_comparison_has_one_type_at_every_depth():
    direct, nested = parse_annotation_text("i == 0, and(i == 0)")
    assert direct == nested.args[0]


def test_positive_conjunctions_lower_in_pre_and_post():
    lowered = lower_proof_goal(goal(f"and(i >= 0, {REGION})", f"and({OUTPUT}, {OUTPUT})"))
    assert [type(p) for p in lowered.pre_conditions] == [AnnComparison, RegionEquiv]
    assert lowered.post_conditions == [parse_annotation_text(OUTPUT)[0]] * 2


def test_universal_mixed_conjunction_preserves_all_premises_and_guards():
    lowered = lower_proof_goal(goal(
        f"forall(i, j, implies(i >= 0, and(i < 4, {REGION})))"))
    scalar, region = lowered.pre_conditions
    assert isinstance(scalar, ForAllConstraint)
    assert scalar.vars == ["i", "j"]
    assert isinstance(scalar.body, AnnImplies)
    assert isinstance(region, GuardedRegionEquality)
    assert region.vars == ["i", "j"]
    assert scalar.body.antecedent == region.when


def test_nested_implications_accumulate_guards_without_strengthening():
    lowered = lower_proof_goal(goal(
        f"forall(i, implies(i >= 0, implies(i < 4, {REGION})))"))
    relation, = lowered.pre_conditions
    assert isinstance(relation.when, AnnAnd)
    assert relation.when.args == parse_annotation_text("i >= 0, i < 4")


def test_guarded_scalar_equality_is_not_hoisted_into_a_launch_binding():
    from ir.annotation_to_config import _find_equality_bindings

    lowered = lower_proof_goal(goal(
        f"forall(i, implies(i < 0, and(right(M) == 1, {REGION})))"))
    assert _find_equality_bindings(lowered.pre_conditions) == ({}, {}, set())


def test_unknown_prop_nodes_fail_closed():
    class FutureProp(Prop):
        pass

    annotation = goal("i == 0")
    annotation.pre_conditions.append(FutureProp())
    with pytest.raises(UnsupportedProposition, match="FutureProp"):
        lower_proof_goal(annotation)


def test_region_binder_cannot_shadow_a_shared_parameter():
    annotation = parse_verif_goal(
        f"# @verif(test, same(i), pre(forall(i, {REGION})), post({OUTPUT}))")
    with pytest.raises(UnsupportedProposition, match="shadow same parameters"):
        lower_proof_goal(annotation)


@pytest.mark.parametrize("text", [
    f"or(i == 0, {REGION})", f"not({REGION})",
    f"forall(i, implies({REGION}, i == 0))",
    f"forall(i, implies(or(i == 0, i == 1), {REGION}))",
    f"forall(i, forall(j, {REGION}))",
    f"implies(i >= 0, {REGION})",
])
def test_unsupported_props_parse_but_never_lower_or_export(text):
    annotation = goal(text)
    assert annotation is not None
    with pytest.raises(UnsupportedProposition, match="pre: unsupported proposition"):
        lower_proof_goal(annotation)
    with pytest.raises(UnsupportedProposition):
        relational_theorem_data(annotation)


@pytest.mark.parametrize("text", ["i == 0", f"or({OUTPUT}, {OUTPUT})", f"forall(i, {OUTPUT})"])
def test_post_is_prop_but_unsupported_conclusions_cannot_be_dropped(text):
    annotation = goal("i == 0", text)
    assert isinstance(annotation.post_conditions[0], Prop)
    with pytest.raises(UnsupportedProposition, match="post: unsupported proposition"):
        lower_proof_goal(annotation)


@pytest.mark.parametrize("text", [
    "i = 0", "and(i = 0)", "forall(i, i = 0)",
    REGION.replace("==", "="),
])
def test_single_equals_is_not_proposition_equality(text):
    with pytest.raises(ParseError, match="Use '=='"):
        parse_annotation_text(text)


@pytest.mark.parametrize("text", [
    f"exists(i, {REGION})", f"forall_region(i, i >= 0, {REGION})",
    "and(0, i == 0)", "left(x)[0:1] == 0", "0 == left(x)[0:1]",
    "forall(i, i, i == 0)",
])
def test_removed_keywords_and_ill_typed_operands_are_rejected(text):
    with pytest.raises(ParseError):
        parse_annotation_text(text)


def test_named_bindings_still_use_single_equals():
    annotation = parse_verif_goal(
        f"# @verif(test, post({OUTPUT}), singleton(bi left=0 right=0))")
    assert annotation.singletons[0].var == "bi"


@pytest.mark.parametrize("unsupported", ["or(left(M) > 0, left(M) > 1)", "not(left(M) == 0)"])
def test_qualified_verifier_rejects_unsupported_premises(unsupported):
    source = (Path(__file__).resolve().parents[2] / "triton_kernels/add.py").read_text()
    source = source.replace("#   pre(", f"#   pre(\n#     {unsupported},")
    with pytest.raises(UnsupportedProposition):
        verify_annotations(source, "add_kernel", dict(BLOCK_M=1, BLOCK_N=64))


def test_conjunctive_post_verifies_every_output_region():
    source = (Path(__file__).resolve().parents[2] / "triton_kernels/add.py").read_text()
    original = "left(o)[b:b+1, 0:N] == right(o)[0:1, 0:N]"
    assert original in source
    source = source.replace(original, f"and({original}, {original})")
    report = verify_annotations(source, "add_kernel", dict(BLOCK_M=1, BLOCK_N=64))
    assert report.verified_contract is not None
    assert len(report.verified_contract.to_data()["theorem_contract"]["theorem"]["post"]) == 2
