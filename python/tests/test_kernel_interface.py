"""A common interface must not become cross-configuration output equality."""

import json
from dataclasses import replace
from pathlib import Path

import pytest
import scripts.verification.kernel_interface_codegen as kernel_interface_codegen

from test_attention_raw_adapter import VERUS, proved_goals, test_raw_attention_batch_adapter as check_adapter
from scripts.verification.kernel_interface_codegen import render_kernel_interface
from ir.relational_verifier import verify_annotations
from scripts.verification.paged_attention_adapter_codegen import render_checked_paged_attention_interface
from scripts.verification.engine_kernel_bindings import FULL_ATTENTION_BINDING, SLIDING_ATTENTION_BINDING


def test_interface_comparison_is_not_attention_specific():
    source = (Path(__file__).resolve().parents[2] / "kernels/triton_kernels/silu_mul.py").read_text()
    alternatives = tuple((verify_annotations(source, "silu_mul_kernel",
        dict(BLOCK_M=m, BLOCK_N=1024), goal_name="batch_invariance").verified_contract,)
        for m in (1, 4))
    interface = render_kernel_interface(alternatives, symbol_prefix="silu")
    assert len(interface.implementations) == 2
    assert interface.body.count("#[verifier::external_body]") == 1
    changed_source = source + "\n# Distinct source identity.\n"
    changed = verify_annotations(changed_source, "silu_mul_kernel",
        dict(BLOCK_M=4, BLOCK_N=1024), goal_name="batch_invariance").verified_contract
    with pytest.raises(ValueError, match="mixes kernel sources"):
        render_kernel_interface((alternatives[0], (changed,)), symbol_prefix="silu")


@pytest.mark.parametrize('field', ['label', 'kind', 'predicate_name'])
def test_interface_checks_manifest_only_premise_identity(monkeypatch, field):
    alternatives = (proved_goals(False, block_m=16), proved_goals(False, block_m=32))
    original = kernel_interface_codegen.render_verified_kernel_to_verus

    def corrupt_record(contracts, **kwargs):
        bundle = original(contracts, **kwargs)
        if contracts == alternatives[1]:
            batch, selected = bundle.contracts
            conditions = list(selected.analyzer_conditions)
            conditions[0] = replace(conditions[0], **{field: 'different_interpretation'})
            selected = replace(selected, analyzer_conditions=tuple(conditions))
            # Keep the rendered theory unchanged: this exercises the separate
            # metadata guard, not artifact qualification or the text comparison.
            bundle = replace(bundle, contracts=(batch, selected))
        return bundle

    monkeypatch.setattr(kernel_interface_codegen, 'render_verified_kernel_to_verus', corrupt_record)
    with pytest.raises(ValueError, match='different logical contract records'):
        render_kernel_interface(alternatives, symbol_prefix='raw')


@pytest.mark.parametrize("swa", [False, True], ids=["full", "swa"])
def test_static_alternatives_preserve_separate_execution_identities(swa):
    alternatives = (proved_goals(swa, block_m=16), proved_goals(swa, block_m=32))
    result = render_kernel_interface(alternatives, symbol_prefix="raw")
    assert len(result.implementations) == 2
    assert len({impl.execution_identity for impl in result.implementations}) == 2
    assert result.body.count("pub open spec fn raw_execute(") == 1
    assert result.body.count("#[verifier::external_body]") == 2
    assert result.body.count("pub uninterp spec fn raw_o_cell(") == 1
    assert "outputs\n// are not equated" in result.body
    manifest = json.loads(result.manifest())
    assert manifest["interpretation"] == "one_fixed_deployed_implementation"
    assert "execution_identity" not in manifest
    assert len(manifest["implementations"]) == 2
    assert manifest["implementations"][0]["execution_identity"] != manifest["implementations"][1]["execution_identity"]
    for impl in result.implementations:
        conditions = impl.contracts[1].analyzer_conditions
        assert {condition.kind for condition in conditions} == {"used_assumption", "external_obligation"}
        for condition in conditions:
            assert condition.predicate_name in result.body
    assert result == render_kernel_interface(tuple(reversed(alternatives)), symbol_prefix="raw")


@pytest.mark.parametrize("case", ["empty", "duplicate", "mixed_bundle", "missing_goal", "different_geometry"])
def test_interface_rejects_weakened_or_ambiguous_exports(case):
    a = proved_goals(False, block_m=16)
    b = proved_goals(False, block_m=32)
    implementations, error = {
        "empty": ((), "no qualified implementation"),
        "duplicate": ((a, a), "duplicate implementation"),
        "mixed_bundle": (((a[0], b[1]),), "mixes source, specialization"),
        "missing_goal": ((a, b[:1]), "different logical contracts"),
        "different_geometry": ((a, proved_goals(False, head_dim=256)), "different logical contracts"),
    }[case]
    with pytest.raises(ValueError, match=error):
        render_kernel_interface(implementations, symbol_prefix="raw")


@pytest.mark.skipif(VERUS is None, reason="set VERUS to check the complete static attention interface")
@pytest.mark.parametrize("swa", [False, True], ids=["full", "swa"])
def test_verus_checks_static_attention_interface(tmp_path, swa):
    interface = render_checked_paged_attention_interface(
        (proved_goals(swa, block_m=16), proved_goals(swa, block_m=32)), symbol_prefix="raw",
        binding=SLIDING_ATTENTION_BINDING if swa else FULL_ATTENTION_BINDING)
    # Unfiltered Verus checks all adapter proofs against the exported abstract
    # raw contracts, including the whole-launch/canonical operation binding.
    check_adapter(tmp_path, swa, "positive", adapter_override=interface)
