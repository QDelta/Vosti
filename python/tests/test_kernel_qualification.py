"""One source qualification supplies deployment receipts and raw Verus input."""

from copy import deepcopy
import hashlib
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest import mock

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "kernels")]
import scripts.deployment.common as qualification
import scripts.verification.kernel_qualification_inventory as inventory
import scripts.verification.verify_kernel_contracts as driver


def artifact(kind, digest, count):
    result = mock.Mock(spec=kind, digest=digest)
    result.to_data.return_value = ({'checks': [1] * count} if kind is qualification.VerifiedExactEffectContract
                                  else {'evidence': {'checks': [1] * count}})
    return result


def fixture_contract(tmp_path, *, conditional=True):
    directory = tmp_path / "triton_kernels"
    directory.mkdir()
    source = "test source\n"
    (directory / "test.py").write_text(source)
    return dict(source="test.py", kernel="test_kernel", wrapper="fixture",
                source_sha256=hashlib.sha256(source.encode()).hexdigest(),
                evidence="conditional_relational_certificate" if conditional else "regional_certificate")


def test_source_mismatch_rejected_before_analysis(tmp_path):
    contract = fixture_contract(tmp_path)
    contract["source_sha256"] = "0" * 64
    with mock.patch.object(qualification, "KERNEL_ROOT", tmp_path), \
         mock.patch.object(qualification, "prepare_kernel_proofs") as prepare:
        with pytest.raises(ValueError, match="source differs"):
            qualification.qualify_proof_case(contract, {}, ("float32",))
        prepare.assert_not_called()


@pytest.mark.parametrize("conditional", [False, True])
@pytest.mark.parametrize("missing_goal", [None, "batch", "selected"])
def test_receipt_and_raw_artifacts_share_the_same_qualification(tmp_path, conditional, missing_goal):
    contract = fixture_contract(tmp_path, conditional=conditional)
    batch = None if missing_goal == "batch" else artifact(qualification.VerifiedDataflowContract, "batch-digest", 1)
    selected = None if missing_goal == "selected" else artifact(qualification.VerifiedDataflowContract, "selected-digest", 2)
    prepared = SimpleNamespace(kernel=object())
    with mock.patch.object(qualification, "KERNEL_ROOT", tmp_path), \
         mock.patch.object(qualification, "prepare_kernel_proofs", return_value=prepared) as prepare, \
         mock.patch.object(qualification, "build_backend_requirement_manifest", return_value={}), \
         mock.patch.object(qualification, "verify_kernel_goal", side_effect=[batch, selected]) as prove:
        if missing_goal == "batch" or (conditional and missing_goal == "selected"):
            with pytest.raises(ValueError, match="incompatible proof kind"):
                qualification.qualify_proof_case(contract, {"D": 128}, ("float32",))
            return
        result = qualification.qualify_proof_case(contract, {"D": 128}, ("float32",))
        assert result.contracts == ((batch, selected) if conditional else (batch,))
        assert result.receipt["structural_contract_digest"] == batch.digest
        prepare.assert_called_once_with("test source\n", "test_kernel", {"D": 128})
        assert prove.call_args_list[0] == mock.call(prepared, "batch_invariance", preserve_analyzer_conditions=False)
        if conditional:
            assert result.receipt["semantic_proof"]["contract_digest"] == selected.digest
            assert prove.call_args_list[1] == mock.call(prepared, "selected_row_prefix_equivalence",
                                                       preserve_analyzer_conditions=True)
        else:
            assert result.receipt["semantic_proof"] is None
            assert prove.call_count == 1


def test_qualification_rejects_unknown_evidence_role():
    with pytest.raises(ValueError, match="unknown kernel qualification evidence role"):
        qualification.required_proof_goals({"evidence": "arbitrary"})


def test_mutation_batch_proof_does_not_claim_the_mutation_effect():
    assert qualification.required_proof_goals({"evidence": "exact_effect_certificate"}) == (
        "batch_invariance", "exact_effect")
    with pytest.raises(ValueError, match="unknown.*evidence role"):
        qualification.required_proof_goals({"evidence": "trusted_mutation_with_structural_certificate"})


@pytest.mark.parametrize("effect_ok", [False, True])
def test_exact_effect_receipt_requires_its_own_proof(tmp_path, effect_ok):
    contract = fixture_contract(tmp_path, conditional=False)
    contract["evidence"] = "exact_effect_certificate"
    batch = artifact(qualification.VerifiedDataflowContract, "batch", 1)
    effect = artifact(qualification.VerifiedExactEffectContract, "effect", 1)
    prepared = SimpleNamespace(kernel=object())
    with mock.patch.object(qualification, "KERNEL_ROOT", tmp_path), \
         mock.patch.object(qualification, "prepare_kernel_proofs", return_value=prepared), \
         mock.patch.object(qualification, "build_backend_requirement_manifest", return_value={}), \
         mock.patch.object(qualification, "verify_kernel_goal", side_effect=[batch, effect if effect_ok else None]) as prove:
        if not effect_ok:
            with pytest.raises(ValueError, match="effect qualification exported an incompatible proof kind"):
                qualification.qualify_proof_case(contract, {}, ("float32",))
            return
        result = qualification.qualify_proof_case(contract, {}, ("float32",))
        assert result.contracts == (batch, effect)
        assert result.receipt["semantic_proof"] == dict(kind="exact_effect", contract_digest="effect", check_count=1)
        assert prove.call_args_list == [mock.call(prepared, "batch_invariance", preserve_analyzer_conditions=False),
                                        mock.call(prepared, "exact_effect", preserve_analyzer_conditions=True)]


def test_real_mutation_cases_qualify_both_goals_symmetrically():
    from ir.exact_effect_artifact import VerifiedExactEffectContract
    from ir.verus_contract import render_verified_kernel_to_verus
    cases = [(c, constants, families) for c, constants, families in inventory.selected_kernel_cases()
             if c["evidence"] == "exact_effect_certificate"]
    assert {constants["KVD"] for _, constants, _ in cases} == {512, 1024, 2048, 4096}
    assert set().union(*(families for _, _, families in cases)) == {"qwen3", "llama3", "gemma3", "gemma4"}
    for contract, constants, _ in cases:
        qualified = qualification.qualify_proof_case(contract, constants, ("bfloat16", "float32"))
        assert len(qualified.contracts) == 2
        effect = qualified.contracts[1]
        assert isinstance(effect, VerifiedExactEffectContract)
        assert qualified.receipt["semantic_proof"] == dict(
            kind="exact_effect", contract_digest=effect.digest, check_count=len(effect.to_data()["checks"]))
        assert render_verified_kernel_to_verus(qualified.contracts, symbol_prefix="raw").execution_identity == effect.execution_identity


def test_real_batch_certificate_cannot_replace_missing_dtype_effect(tmp_path):
    contract, constants, _ = next((c, k, f) for c, k, f in inventory.selected_kernel_cases()
                                  if c["evidence"] == "exact_effect_certificate")
    contract = deepcopy(contract)
    source = (qualification.KERNEL_ROOT / "triton_kernels" / contract["source"]).read_text()
    weakened = source.replace("#     dtype(x) == dtype(cache),\n", "")
    assert weakened != source
    (tmp_path / "triton_kernels").mkdir()
    (tmp_path / "triton_kernels" / contract["source"]).write_text(weakened)
    contract["source_sha256"] = hashlib.sha256(weakened.encode()).hexdigest()
    with mock.patch.object(qualification, "KERNEL_ROOT", tmp_path), \
         mock.patch.object(qualification, "verify_kernel_goal", wraps=qualification.verify_kernel_goal) as prove:
        with pytest.raises(ValueError, match="exact-effect verifier"):
            qualification.qualify_proof_case(contract, constants, ("bfloat16", "float32"))
        assert [call.args[1] for call in prove.call_args_list] == ['batch_invariance', 'exact_effect']


@pytest.mark.parametrize('binding', [
    {'batch': 'same', 'effect': 'same'}, {'batch': 'only'},
    {'batch': 'b', 'effect': 'e', 'extra': 'x'}, {'batch': 1, 'effect': 'e'},
])
def test_invalid_goal_role_bindings_fail_closed(binding):
    with pytest.raises(ValueError, match='proof-goal binding'):
        qualification.required_proof_goals({'evidence': 'exact_effect_certificate', 'proof_goals': binding})


def test_renamed_goals_qualify_and_mutation_receipt_is_admitted(tmp_path):
    from scripts.verification.mutation_interface_codegen import render_mutation_interface
    from vosti_kernels.kernel_interfaces import validate_mutation_interface_case
    import json
    contract, constants, families = next((c, k, f) for c, k, f in inventory.selected_kernel_cases()
                                        if c['evidence'] == 'exact_effect_certificate')
    contract = deepcopy(contract)
    source = (qualification.KERNEL_ROOT / 'triton_kernels' / contract['source']).read_text()
    source = source.replace('@verif(batch_invariance,', '@verif(row_equality,')
    source = source.replace('@verif(exact_effect,', '@verif(copy_and_frame,')
    contract['proof_goals'] = dict(batch='row_equality', effect='copy_and_frame')
    contract['source_sha256'] = hashlib.sha256(source.encode()).hexdigest()
    (tmp_path / 'triton_kernels').mkdir()
    (tmp_path / 'triton_kernels' / contract['source']).write_text(source)
    with mock.patch.object(qualification, 'KERNEL_ROOT', tmp_path):
        case = qualification.qualify_proof_case(contract, constants, ('float32',))
    body, manifest = render_mutation_interface(((contract, constants, families, case),))
    assert 'raw_copy_and_frame_certificate' in body
    validate_mutation_interface_case(case.receipt, json.loads(manifest))


def test_effect_role_cannot_be_filled_by_relational_evidence(tmp_path):
    contract = fixture_contract(tmp_path, conditional=False)
    contract['evidence'] = 'exact_effect_certificate'
    batch = artifact(qualification.VerifiedDataflowContract, 'batch', 1)
    with mock.patch.object(qualification, 'KERNEL_ROOT', tmp_path), \
         mock.patch.object(qualification, 'prepare_kernel_proofs', return_value=SimpleNamespace(kernel=object())), \
         mock.patch.object(qualification, 'verify_kernel_goal', return_value=batch):
        with pytest.raises(ValueError, match='effect qualification exported an incompatible proof kind'):
            qualification.qualify_proof_case(contract, {}, ('float32',))


def test_family_cannot_overwrite_shared_catalog():
    with mock.patch.object(driver, "selected_kernel_cases") as select:
        with pytest.raises(ValueError, match="complete family inventory"):
            driver.verify(families=["example"], write=True)
        select.assert_not_called()


@pytest.mark.parametrize("difference", ["alias", "batch_role", "source", "goals", "typed_constant"])
def test_static_case_deduplication_preserves_execution_and_goals(difference):
    left = dict(source="test.py", kernel="kernel", source_sha256="a" * 64,
                wrapper="left_alias", evidence="regional_certificate")
    right = deepcopy(left)
    constants = {"BLOCK": 1}
    if difference == "alias":
        right["wrapper"] = "right_alias"
    elif difference == "batch_role":
        right["evidence"] = "structural_batch_decomposition_not_numeric_correctness"
    elif difference == "source":
        right["source_sha256"] = "b" * 64
    elif difference == "goals":
        right["evidence"] = "conditional_relational_certificate"
    elif difference == "typed_constant":
        constants = {"BLOCK": True}
    with mock.patch.object(inventory, "load_registry", return_value=({"a": {}, "b": {}}, {})), \
         mock.patch.object(inventory, "family_cases", side_effect=[[(left, {"BLOCK": 1})], [(right, constants)]]):
        if difference in {"source", "goals"}:
            with pytest.raises(ValueError, match="contributors disagree"):
                inventory.selected_kernel_cases()
        else:
            cases = inventory.selected_kernel_cases()
            assert len(cases) == (2 if difference == "typed_constant" else 1)
            if len(cases) == 1:
                assert cases[0][2] == {"a", "b"}


@pytest.mark.parametrize("families", [[], ["unknown"]])
def test_unknown_or_empty_family_filter_fails_closed(families):
    with pytest.raises(ValueError, match="empty or unknown"):
        inventory.selected_kernel_cases(families)
