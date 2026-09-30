"""Consumers use unified provenance without erasing its conditional meaning."""

from dataclasses import replace
import json
from pathlib import Path
from unittest import mock

import pytest

from ir.relational_artifact import VerifiedDataflowContract
from ir.contract_schema import validate_relational_contract
from ir.relational_verifier import validate_dataflow_artifact
from ir.relational_dataflow import prove_prepared_relational_dataflow
from ir.proof_preparation import prepare_annotation_proof
from ir.verus_contract import (
    render_contract_manifest, render_verified_contract_to_verus,
    transpile_verified_kernel_source,
)


ROOT = Path(__file__).resolve().parents[2] / "triton_kernels"


@pytest.fixture(scope="module", params=[
    ("add.py", "add_kernel", {"BLOCK_M": 1, "BLOCK_N": 64}),
    ("fattn_paged.py", "fattn_varlen_paged_fwd_block_ptr_kernel",
     {"BLOCK_M": 16, "BLOCK_N": 64, "PAGE_BLOCK_SIZE": 64, "D_HEAD": 128}),
    ("fattn_paged_swa.py", "fattn_varlen_paged_swa_kernel",
     {"BLOCK_M": 16, "BLOCK_N": 64, "PAGE_BLOCK_SIZE": 64, "D_HEAD": 128}),
])
def proved(request):
    filename, kernel, constants = request.param
    prepared = prepare_annotation_proof((ROOT / filename).read_text(), kernel, constants,
                                        goal_name="batch_invariance")
    unified = prove_prepared_relational_dataflow(prepared)
    validate_dataflow_artifact(prepared, unified)
    return prepared, unified


@pytest.fixture(scope="module")
def exportable(proved):
    return proved


def _strict_surface(artifact):
    return validate_relational_contract(artifact, family="consumer regression")


def _artifact(data):
    return VerifiedDataflowContract(json.dumps(data, sort_keys=True, separators=(",", ":")))


def test_schema_validation_preserves_complete_theorem(proved):
    prepared, unified = proved
    validated = validate_relational_contract(unified.verified_contract, family="test")
    assert validated.raw == unified.verified_contract.to_data()["theorem_contract"]
    assert validated.raw["kernel"] == prepared.kernel.name


@pytest.mark.parametrize("change", ["noncanonical", "unknown_condition"])
def test_raw_consumer_rejects_malformed_contracts(proved, change):
    _, unified = proved
    data = unified.verified_contract.to_data()
    if change == "noncanonical":
        artifact = VerifiedDataflowContract(json.dumps(data, indent=2))
    else:
        data["theorem_contract"]["theorem"]["pre"].append({"kind": "future_condition"})
        artifact = _artifact(data)
    for consume in (_strict_surface, render_verified_contract_to_verus):
        with pytest.raises(ValueError):
            consume(artifact)


def test_raw_verus_preserves_exact_surface_and_unified_identity(exportable):
    _, unified = exportable
    new = render_verified_contract_to_verus(unified.verified_contract, symbol_prefix="reviewed")
    assert new.raw_contract_digest == unified.verified_contract.digest
    assert new.raw_contract_digest in new.body
    assert new.output_parameters == tuple(dict.fromkeys(a.output_tensor for a in unified.alignments))
    assert f"requires {new.pre_name}(left, right, free)" in new.body


@pytest.mark.parametrize("field", ["used_assumptions", "external_obligations"])
def test_no_consumer_may_silently_drop_analyzer_conditions(exportable, field):
    prepared, unified = exportable
    data = unified.verified_contract.to_data()
    data["evidence"][field] = ["finite(example)"]
    artifact = _artifact(data)
    artifact.to_data()  # Well-formed conditional artifact, not an unconditional theorem.
    with pytest.raises(ValueError, match="cannot discharge or erase"):
        _strict_surface(artifact)
    rendered = render_verified_contract_to_verus(artifact)
    manifest = json.loads(render_contract_manifest(rendered))
    assert manifest["schema_version"] == 3
    assert len(manifest["analyzer_conditions"]) == 1
    condition = manifest["analyzer_conditions"][0]
    assert condition["kind"] == {
        "used_assumptions": "used_assumption", "external_obligations": "external_obligation",
    }[field]
    assert condition["label"] == "finite(example)"
    name = condition["predicate_name"]
    assert f"pub uninterp spec fn {name}(" in rendered.body
    pre_body = rendered.body.split(f"pub open spec fn {rendered.pre_name}(")[1].split(
        f"pub open spec fn {rendered.post_name}("
    )[0]
    assert f"{name}(left, right, free)" in pre_body
    assert f"requires {rendered.pre_name}(left, right, free)" in rendered.body
    assert rendered.raw_contract_digest == artifact.digest


def test_condition_labels_are_preserved_without_becoming_source_code(exportable):
    _, unified = exportable
    data = unified.verified_contract.to_data()
    label = 'finite(x)\n}\npub axiom unsafe_assumption(); // "'
    data["evidence"]["used_assumptions"] = [label]
    data["evidence"]["external_obligations"] = [label]
    rendered = render_verified_contract_to_verus(_artifact(data))
    conditions = json.loads(render_contract_manifest(rendered))["analyzer_conditions"]
    assert [condition["label"] for condition in conditions] == [label, label]
    assert len({condition["predicate_name"] for condition in conditions}) == 2
    assert label not in rendered.body
    assert "unsafe_assumption" not in rendered.body
    for condition in conditions:
        assert f"{condition['predicate_name']}(left, right, free)" in rendered.body


@pytest.mark.parametrize("filename,kernel", [
    ("fattn_paged.py", "fattn_varlen_paged_fwd_block_ptr_kernel"),
    ("fattn_paged_swa.py", "fattn_varlen_paged_swa_kernel"),
])
def test_selected_row_raw_export_retains_real_numeric_premise(filename, kernel):
    prepared = prepare_annotation_proof(
        (ROOT / filename).read_text(), kernel,
        {"BLOCK_M": 16, "BLOCK_N": 64, "PAGE_BLOCK_SIZE": 64, "D_HEAD": 128},
        goal_name="selected_row_prefix_equivalence",
    )
    report = prove_prepared_relational_dataflow(prepared)
    assert report.proved and report.verified_contract is not None
    assert report.external_obligations == ("finite(v_block)@masked-backward-dependency",)
    assert report.used_assumptions == report.external_obligations
    rendered = render_verified_contract_to_verus(report.verified_contract)
    assert [condition.kind for condition in rendered.analyzer_conditions] == [
        "used_assumption", "external_obligation",
    ]
    for condition in rendered.analyzer_conditions:
        assert condition.label == report.external_obligations[0]
        assert f"{condition.predicate_name}(left, right, free)" in rendered.body
    assert rendered.singleton_name is not None
    assert rendered.raw_contract_digest == report.verified_contract.digest
    with pytest.raises(ValueError, match="cannot discharge or erase"):
        _strict_surface(report.verified_contract)


@pytest.mark.parametrize("change", ["failed", "missing", "unknown"])
def test_consumers_validate_evidence_not_only_nested_theorem(proved, change):
    prepared, unified = proved
    data = unified.verified_contract.to_data()
    if change == "failed":
        data["evidence"]["checks"][0]["proved"] = False
    elif change == "missing":
        data["evidence"]["alignments"] = []
    else:
        data["evidence"]["extra"] = True
    for consume in (_strict_surface, render_verified_contract_to_verus):
        with pytest.raises(ValueError):
            consume(_artifact(data))


def test_transpilation_emits_unified_not_regional_provenance():
    source = (ROOT / "add.py").read_text().replace(
        "@verif(batch_invariance,", "@verif(selected_value_equality,",
    )
    constants = {"BLOCK_M": 1, "BLOCK_N": 64}
    prepared = prepare_annotation_proof(source, "add_kernel", constants,
                                        goal_name="selected_value_equality")
    unified = prove_prepared_relational_dataflow(prepared)
    rendered = transpile_verified_kernel_source(source, "add_kernel", constants,
                                                goal_name="selected_value_equality")
    assert rendered.raw_contract_digest == unified.verified_contract.digest


@pytest.mark.parametrize("field,value", [
    ("checks", ()), ("verified_contract", None),
    ("unsupported_reason", "not supported"),
])
def test_source_transpilation_never_falls_back_to_regional(proved, field, value):
    prepared, unified = proved
    with mock.patch("ir.relational_verifier.prove_prepared_relational_dataflow",
                    return_value=replace(unified, **{field: value})):
        with pytest.raises(ValueError, match="relational verifier"):
            transpile_verified_kernel_source(prepared.source, prepared.kernel.name,
                                              dict(prepared.constants))


def test_source_transpilation_preserves_the_complete_condition_envelope(exportable):
    prepared, unified = exportable
    from ir.relational_artifact import _build_verified_dataflow_contract

    conditional = replace(unified, external_obligations=("finite(example)",))
    conditional = replace(
        conditional, verified_contract=_build_verified_dataflow_contract(prepared, conditional),
    )
    assert render_verified_contract_to_verus(conditional.verified_contract).analyzer_conditions
    with mock.patch("ir.relational_verifier.prove_prepared_relational_dataflow",
                    return_value=conditional):
        rendered = transpile_verified_kernel_source(prepared.source, prepared.kernel.name,
                                                     dict(prepared.constants))
        assert [condition.label for condition in rendered.analyzer_conditions] == ["finite(example)"]
        assert rendered.raw_contract_digest == conditional.verified_contract.digest
        condition, = rendered.analyzer_conditions
        assert f"{condition.predicate_name}(left, right, free)" in rendered.body
