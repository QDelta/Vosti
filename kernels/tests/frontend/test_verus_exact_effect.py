"""Temporal lowering retains premises and shares the relational execution."""

import json
import os
import shutil
import subprocess
import sys
from functools import cache

import pytest

from ir.exact_effects import verify_exact_effects
from ir.kernel_verifier import verify_kernel_goal
from ir.proof_preparation import prepare_kernel_proofs
from ir.relational_verifier import verify_annotations
from ir.verus_contract import render_contract_manifest, render_standalone_verus_module, render_verified_kernel_to_verus
from ir.verus_exact_effect import render_verified_exact_effect_to_verus
from tests.analysis.test_exact_effects import annotated
from tests.analysis.test_exact_effects import ROOT
from tests.frontend.test_exact_effect_artifact import proved


@cache
def relational():
    report = verify_annotations(annotated(), "store_cache_kernel", {"KVD": 512, "BLOCK_M": 1})
    assert report.proved and report.verified_contract is not None
    return report.verified_contract


def test_temporal_rendering_has_real_dtype_and_before_after_copy_frame():
    rendered = render_verified_exact_effect_to_verus(proved(), symbol_prefix="scatter")
    body = rendered.body
    assert "pub __element_dtype_x: ScatterDType" in body
    assert "before.__element_dtype_x == before.__element_dtype_cache" in body
    assert "before.x[_q0 + _r0][0 + _r1]" in body
    assert "before.cache[_q0 + _r0][0 + _r1]" in body
    assert "after.cache[before.slot_mapping[_q0] + _r0][0 + _r1]" in body
    assert "ensures scatter_raw_post(before, scatter_execute(before), free)" in body
    assert "__element_dtype_x: before.__element_dtype_x" in body
    assert "before.cache.len()" in body
    assert body.count("pub uninterp spec fn scatter_external_obligation_") == 3
    manifest = json.loads(render_contract_manifest(rendered))
    assert manifest["proof_kind"] == "exact_effect" and manifest["state_relation"] == "before_after"


def test_mixed_goals_share_one_execution_and_dtypes_independent_of_order():
    inputs = (proved(), relational())
    one = render_verified_kernel_to_verus(inputs, symbol_prefix="store")
    two = render_verified_kernel_to_verus(tuple(reversed(inputs)), symbol_prefix="store")
    assert one == two
    assert one.execution_identity == proved().execution_identity
    assert one.body.count("pub struct StoreSide") == 1
    assert one.body.count("pub struct StoreDType") == 1
    assert one.body.count("pub open spec fn store_execute(") == 1
    assert one.body.count("pub uninterp spec fn store_cache_cell(") == 1
    assert len(one.contracts) == 2
    assert all(c.execute_name == "store_execute" for c in one.contracts)
    assert all("pub __element_dtype_cache: StoreDType" in c.body for c in one.contracts)
    relational_body = next(c.body for c in one.contracts if c.symbol_prefix.endswith("batch_invariance"))
    assert "left.__element_dtype_cache == right.__element_dtype_cache" in relational_body


def test_mixed_bundle_rejects_different_sources_or_configs():
    other = verify_exact_effects(annotated(), "store_cache_kernel", {"KVD": 1024, "BLOCK_M": 1})
    assert other.proved
    with pytest.raises(ValueError, match="mixes source, specialization"):
        render_verified_kernel_to_verus((relational(), other.verified_contract), symbol_prefix="store")


def test_generic_entrypoint_dispatches_by_typed_props_and_requires_premise_opt_in():
    source = annotated().replace("@verif(exact_effect,", "@verif(write_result,")
    prepared = prepare_kernel_proofs(source, "store_cache_kernel", {"KVD": 512, "BLOCK_M": 1})
    with pytest.raises(ValueError, match="preserve physical"):
        verify_kernel_goal(prepared, "write_result")
    artifact = verify_kernel_goal(prepared, "write_result", preserve_analyzer_conditions=True)
    assert artifact.to_data()["goal_name"] == "write_result"


@pytest.mark.parametrize("invalid", [False, True])
def test_cli_exports_mixed_goals_only_after_both_proofs_pass(tmp_path, invalid):
    source = annotated()
    if invalid:
        source = source.replace("#     dtype(x) == dtype(cache),\n", "")
    path = tmp_path / "scatter.py"
    path.write_text(source)
    output, manifest = tmp_path / "contract.rs", tmp_path / "manifest.json"
    result = subprocess.run([
        sys.executable, str(ROOT / "scripts/transpile_verus_contract.py"), str(path), "store_cache_kernel",
        "--constant", "KVD=512", "--constant", "BLOCK_M=1",
        "--goal", "batch_invariance", "--goal", "exact_effect", "--symbol-prefix", "store",
        "--output", str(output), "--manifest", str(manifest),
    ], capture_output=True, text=True, timeout=120)
    if invalid:
        assert result.returncode != 0 and "exact-effect verifier" in result.stderr
        assert not output.exists() and not manifest.exists()
    else:
        assert result.returncode == 0, result.stdout + result.stderr
        assert output.read_text().count("pub open spec fn store_execute(") == 1
        assert len(json.loads(manifest.read_text())["standalone_contracts"]) == 2


@pytest.mark.parametrize("mixed", [False, True])
def test_generated_temporal_and_shared_modules_pass_verus(tmp_path, mixed):
    verus = os.environ.get("VERUS") or shutil.which("verus")
    if not verus:
        pytest.skip("Verus executable not configured")
    contract = (render_verified_kernel_to_verus((proved(), relational()), symbol_prefix="store") if mixed
                else render_verified_exact_effect_to_verus(proved(), symbol_prefix="store"))
    path = tmp_path / "contract.rs"
    path.write_text(render_standalone_verus_module(contract))
    result = subprocess.run([verus, str(path)], text=True, capture_output=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("remove_dtype", [False, True])
def test_checked_copy_frame_caller_needs_dtype_premise(tmp_path, remove_dtype):
    verus = os.environ.get("VERUS") or shutil.which("verus")
    if not verus:
        pytest.skip("Verus executable not configured")
    rendered = render_verified_exact_effect_to_verus(proved(), symbol_prefix="store")
    # The weakened caller predicate differs only by the physical dtype equality.
    start = rendered.body.index("pub open spec fn store_raw_pre(")
    end = rendered.body.index("pub open spec fn store_raw_post(", start)
    weak = rendered.body[start:end].replace("store_raw_pre", "caller_pre")
    if remove_dtype:
        weak = weak.replace("    &&& (before.__element_dtype_x == before.__element_dtype_cache)\n", "")
    caller = '''
pub proof fn consume_copy_and_frame(before: StoreSide, free: StoreFree, i: int, j: int, s: int)
    requires caller_pre(before, free),
        0 <= i < before.M, before.slot_mapping[i] >= 0, 0 <= j < 512,
        0 <= s < before.NUM_SLOTS,
        forall|r: int| 0 <= r < before.M ==> (#[trigger] before.slot_mapping[r]) != s,
    ensures
        store_execute(before).cache[before.slot_mapping[i]][j] == before.x[i][j],
        store_execute(before).cache[s][j] == before.cache[s][j],
{
    store_certificate(before, free);
    let after = store_execute(before);
    assert(store_quantified_2(before, after, free, i));
    assert forall|r: int| #[trigger] store_quantified_3(before, after, free, s, r) by {}
    assert(store_quantified_4(before, after, free, s));
    assert(store_region_0(before, after, free, i, 0, j));
    assert(store_region_1(before, after, free, s, 0, j));
}
'''
    source = render_standalone_verus_module(rendered).replace("} // verus!", weak + caller + "} // verus!")
    path = tmp_path / "caller.rs"
    path.write_text(source)
    result = subprocess.run([verus, str(path)], text=True, capture_output=True, timeout=120)
    if remove_dtype:
        assert result.returncode != 0 and "precondition not satisfied" in result.stdout + result.stderr
    else:
        assert result.returncode == 0, result.stdout + result.stderr
