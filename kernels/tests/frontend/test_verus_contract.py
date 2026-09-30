"""Tests for proof-gated generic ContractIR-to-Verus lowering."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from functools import cache
from ir.relational_artifact import VerifiedDataflowContract
from ir.relational_verifier import verify_annotations
from ir.relational_dataflow import prove_prepared_relational_dataflow
from ir.proof_preparation import prepare_annotation_proof
from tests.frontend.attention_contract_fixtures import iteration_scoped_attention_source
from ir.verus_contract import (
    render_contract_manifest,
    render_standalone_verus_module,
    render_verified_contract_to_verus,
    transpile_verified_kernel_source,
)


_KERNEL_DIR = Path(__file__).resolve().parents[2] / "triton_kernels"


def _source(name: str) -> str:
    return (_KERNEL_DIR / name).read_text(encoding="utf-8")


@cache
def _proved_silu_contract() -> VerifiedDataflowContract:
    result = verify_annotations(
        _source("silu_mul.py"),
        "silu_mul_kernel",
        {"BLOCK_M": 1, "BLOCK_N": 4096},
    )
    assert result.proved and result.verified_contract is not None
    return result.verified_contract


def test_silu_lowering_preserves_domain_shapes_regions_and_post() -> None:
    contract = render_verified_contract_to_verus(
        _proved_silu_contract(), symbol_prefix="silu_mul_deployed"
    )

    assert "pub struct SiluMulDeployedSide" in contract.body
    assert "pub x: Seq<Seq<Scalar>>" in contract.body
    assert "left.x.len() as int == left.M" in contract.body
    assert "(#[trigger] left.x[_s0]).len() as int == left.N" in contract.body
    assert "left.N == right.N" in contract.body
    assert "right.M == 1" in contract.body
    assert "(numerator + denominator - 1) / denominator" in contract.body
    assert "if 0 <= numerator" not in contract.body
    assert "free.b < left.M" in contract.body
    assert "left.x[free.b + _r0][0 + _r1]" in contract.body
    assert "right.x[0 + _r0][0 + _r1]" in contract.body
    assert "left.o[free.b + _r0][0 + _r1]" in contract.body
    assert contract.pre_name == "silu_mul_deployed_raw_pre"
    assert contract.post_name == "silu_mul_deployed_raw_post"
    assert contract.singleton_name is None
    assert contract.execute_name == "silu_mul_deployed_execute"
    assert contract.certificate_name == "silu_mul_deployed_certificate"
    assert contract.output_parameters == ("o",)
    assert contract.output_functions == ("silu_mul_deployed_o_after",)
    assert "Seq::new(before.o.len()" in contract.body
    assert "x: before.x" in contract.body
    assert "o: silu_mul_deployed_o_after(before)" in contract.body
    assert "#[verifier::external_body]" in contract.body
    assert "requires silu_mul_deployed_raw_pre(left, right, free)" in contract.body
    assert "silu_mul_deployed_execute(left)" in contract.body

    manifest = json.loads(render_contract_manifest(contract))
    assert manifest["raw_contract_digest"] == contract.raw_contract_digest
    assert manifest["generated_body_sha256"] == contract.body_sha256
    assert manifest["certificate_name"] == contract.certificate_name
    assert manifest["output_parameters"] == list(contract.output_parameters)
    assert manifest["output_functions"] == list(contract.output_functions)
    assert manifest["schema_version"] == 3
    assert manifest["analyzer_conditions"] == []
    assert contract.analyzer_conditions == ()
    standalone = render_standalone_verus_module(contract)
    assert "use vstd::prelude::*;" in standalone
    assert contract.body in standalone


@pytest.mark.parametrize(
    ("filename", "kernel", "constants"),
    [
        ("rmsnorm.py", "rmsnorm_kernel", {"BLOCK_M": 1, "BLOCK_N": 1024}),
        (
            "rmsnorm_residual.py",
            "rmsnorm_residual_kernel",
            {"BLOCK_M": 1, "BLOCK_N": 1024},
        ),
        ("silu_mul.py", "silu_mul_kernel", {"BLOCK_M": 1, "BLOCK_N": 4096}),
        (
            "embedding.py",
            "embedding_kernel",
            {"D": 1024, "BLOCK_M": 1, "BLOCK_D": 1024},
        ),
        (
            "qk_norm.py",
            "head_rms_norm_kernel",
            {"H": 16, "D": 128, "BLOCK_M": 1},
        ),
        (
            "store_kv_cache.py",
            "store_cache_kernel",
            {"KVD": 1024, "BLOCK_M": 1},
        ),
        ("rope.py", "rope_kernel", {"D": 128, "HD": 64, "BLOCK_M": 1}),
        (
            "matmul.py",
            "matmul_kernel",
            {"BLOCK_M": 16, "BLOCK_N": 64, "BLOCK_K": 64},
        ),
        (
            "fattn_paged.py",
            "fattn_varlen_paged_fwd_block_ptr_kernel",
            {
                "BLOCK_M": 16,
                "BLOCK_N": 64,
                "D_HEAD": 128,
                "PAGE_BLOCK_SIZE": 64,
            },
        ),
    ],
)
def test_supported_raw_contract_surfaces_lower(
    filename: str,
    kernel: str,
    constants: dict[str, int | bool],
) -> None:
    source = _source(filename)
    rendered = transpile_verified_kernel_source(source, kernel, constants)
    assert rendered.kernel == kernel
    assert rendered.raw_contract_digest in rendered.body
    assert rendered.pre_name in rendered.body
    assert rendered.post_name in rendered.body


def test_paged_lowering_preserves_explicit_quantifiers_and_singleton() -> None:
    rendered = transpile_verified_kernel_source(
        _source("fattn_paged.py"),
        "fattn_varlen_paged_fwd_block_ptr_kernel",
        {
            "BLOCK_M": 16,
            "BLOCK_N": 64,
            "D_HEAD": 128,
            "PAGE_BLOCK_SIZE": 64,
        },
        symbol_prefix="paged_deployed",
    )
    assert "forall|_q0: int, _q1: int| #[trigger] paged_deployed_quantified_" in rendered.body
    assert "left.block_table[free.x][" in rendered.body
    assert "right.block_table[0][" in rendered.body
    assert 'pub ki: int' not in rendered.body
    assert "pub struct PagedDeployedPrograms" in rendered.body
    assert "left_programs.bi == free.x" in rendered.body
    assert "right_programs.bi == 0" in rendered.body


@pytest.mark.parametrize('swa', [False, True])
def test_attention_verifier_rejects_first_page_only_equality(swa):
    filename = 'fattn_paged_swa.py' if swa else 'fattn_paged.py'
    kernel = 'fattn_varlen_paged_swa_kernel' if swa else 'fattn_varlen_paged_fwd_block_ptr_kernel'
    source = _source(filename)
    bound = ('cache_page < cdiv(left(cu_seqlens_k)[x+1] - '
             'left(cu_seqlens_k)[x], PAGE_BLOCK_SIZE)')
    assert source.count(bound) == 2
    with pytest.raises(ValueError, match="failed or unsupported dataflow proof"):
        verify_annotations(source.replace(bound, 'cache_page < 1'), kernel,
                           dict(BLOCK_M=16, BLOCK_N=64, D_HEAD=128, PAGE_BLOCK_SIZE=64))


@pytest.mark.parametrize('swa', [False, True])
def test_raw_export_rejects_iteration_scoped_attention_premises(swa):
    filename = 'fattn_paged_swa.py' if swa else 'fattn_paged.py'
    kernel = 'fattn_varlen_paged_swa_kernel' if swa else 'fattn_varlen_paged_fwd_block_ptr_kernel'
    constants = dict(BLOCK_M=16, BLOCK_N=64, D_HEAD=128, PAGE_BLOCK_SIZE=64)
    # The regional proof still accepts its iteration-scoped schema. It must
    # not be exported as one caller-selected cache-page equality implying
    # equality of the whole request's output.
    report = verify_annotations(iteration_scoped_attention_source(filename), kernel, constants)
    assert report.proved
    with pytest.raises(ValueError, match="unbound execution iterators.*ki"):
        render_verified_contract_to_verus(report.verified_contract)


def test_raw_cli_writes_no_artifact_for_unbound_iterations(tmp_path):
    output, manifest = tmp_path / 'contract.rs', tmp_path / 'contract.json'
    source = tmp_path / 'iteration_scoped.py'
    source.write_text(iteration_scoped_attention_source('fattn_paged.py'))
    result = subprocess.run([
        sys.executable, str(_KERNEL_DIR.parent / 'scripts/transpile_verus_contract.py'),
        str(source), 'fattn_varlen_paged_fwd_block_ptr_kernel',
        '--constant', 'BLOCK_M=16', '--constant', 'BLOCK_N=64',
        '--constant', 'D_HEAD=128', '--constant', 'PAGE_BLOCK_SIZE=64',
        '--output', str(output), '--manifest', str(manifest),
    ], text=True, capture_output=True, timeout=120)
    assert result.returncode != 0
    assert 'unbound execution iterators' in result.stderr
    assert not output.exists() and not manifest.exists()


@pytest.mark.parametrize("filename,kernel", [
    ("fattn_paged.py", "fattn_varlen_paged_fwd_block_ptr_kernel"),
    ("fattn_paged_swa.py", "fattn_varlen_paged_swa_kernel"),
])
def test_source_entrypoint_preserves_conditional_goal_premises(filename, kernel):
    from ir.relational_verifier import verify_annotations
    constants = dict(BLOCK_M=16, BLOCK_N=64, D_HEAD=128, PAGE_BLOCK_SIZE=64)
    source = _source(filename)
    report = verify_annotations(source, kernel, constants,
        goal_name="selected_row_prefix_equivalence", preserve_analyzer_conditions=True)
    direct = render_verified_contract_to_verus(report.verified_contract, symbol_prefix="selected_raw")
    through_source = transpile_verified_kernel_source(source, kernel, constants,
        goal_name="selected_row_prefix_equivalence", symbol_prefix="selected_raw")
    assert through_source == direct
    assert through_source.analyzer_conditions
    for condition in through_source.analyzer_conditions:
        assert condition.predicate_name + "(left, right, free)" in through_source.body


@cache
def _selected_paged_raw_contract() -> VerifiedDataflowContract:
    source = _source("fattn_paged.py")
    constants = {
        "BLOCK_M": 16,
        "BLOCK_N": 64,
        "D_HEAD": 128,
        "PAGE_BLOCK_SIZE": 64,
    }
    prepared = prepare_annotation_proof(
        source,
        "fattn_varlen_paged_fwd_block_ptr_kernel",
        constants,
        goal_name="selected_row_prefix_equivalence",
    )
    report = prove_prepared_relational_dataflow(prepared)
    assert report.proved and report.verified_contract is not None
    return report.verified_contract


def test_quantified_region_schema_lowers_with_bound_prefix_variable() -> None:
    artifact = _selected_paged_raw_contract()
    rendered = render_verified_contract_to_verus(
        artifact, symbol_prefix="paged_selected"
    )
    quantified = [condition for condition in artifact.to_data()["theorem_contract"]["theorem"]["pre"]
                  if condition["kind"] in {"forall", "forall_region"}]
    assert rendered.body.count("forall|_q0: int| #[trigger] paged_selected_quantified_") == len(quantified)
    assert "free.prefix_tile" not in rendered.body
    for index in range(len(quantified)):
        assert f"pub open spec fn paged_selected_quantified_{index}(" in rendered.body
    assert "#![auto]" not in rendered.body
    assert "left.k_cache[left.block_table[0]" in rendered.body
    assert "left.v_cache[left.block_table[0]" in rendered.body


@pytest.mark.parametrize("binder", ["left", "right", "free", "_r0"])
def test_quantified_binders_cannot_capture_generated_names(binder) -> None:
    original = _selected_paged_raw_contract()
    raw = original.to_data()

    def rename(value):
        if isinstance(value, list):
            return [rename(item) for item in value]
        if not isinstance(value, dict):
            return value
        if value == {"kind": "free", "name": "prefix_tile"}:
            return {"kind": "free", "name": binder}
        return {key: rename(item) for key, item in value.items()}

    for index, condition in enumerate(raw["theorem_contract"]["theorem"]["pre"]):
        if condition["kind"] == "forall_region":
            assert condition["variables"] == ["prefix_tile"]
            raw["theorem_contract"]["theorem"]["pre"][index] = {
                **rename(condition), "variables": [binder],
            }
    renamed = VerifiedDataflowContract(
        canonical_json=json.dumps(raw, sort_keys=True, separators=(",", ":")),
    )
    before = render_verified_contract_to_verus(original, symbol_prefix="bound_names")
    after = render_verified_contract_to_verus(renamed, symbol_prefix="bound_names")
    assert after.body.replace(renamed.digest, original.digest) == before.body


def test_unbounded_quantified_region_fails_closed() -> None:
    raw = _selected_paged_raw_contract().to_data()
    condition = next(
        condition
        for condition in raw["theorem_contract"]["theorem"]["pre"]
        if condition["kind"] == "forall_region"
    )
    assert condition["when"]["kind"] == "and"
    condition["when"]["args"] = condition["when"]["args"][:1]
    malformed = VerifiedDataflowContract(
        canonical_json=json.dumps(raw, sort_keys=True, separators=(",", ":"))
    )
    with pytest.raises(ValueError, match="explicit lower and upper guards"):
        render_verified_contract_to_verus(malformed)


def test_unbounded_forall_fails_closed() -> None:
    raw = _proved_silu_contract().to_data()
    raw["theorem_contract"]["theorem"]["pre"].append(
        {
            "kind": "forall",
            "variables": ["i"],
            "body": {
                "kind": "comparison",
                "op": ">=",
                "lhs": {"kind": "free", "name": "i"},
                "rhs": {"kind": "int", "value": 0},
            },
        }
    )
    malformed = VerifiedDataflowContract(
        canonical_json=json.dumps(raw, sort_keys=True, separators=(",", ":"))
    )
    with pytest.raises(ValueError, match="guarded implications"):
        render_verified_contract_to_verus(malformed)


def test_tensor_index_rank_mismatch_fails_closed() -> None:
    raw = _proved_silu_contract().to_data()
    raw["theorem_contract"]["theorem"]["pre"].append(
        {
            "kind": "scalar",
            "op": "==",
            "lhs": {
                "kind": "index",
                "base": {"kind": "side", "side": "left", "name": "x"},
                "indices": [{"kind": "int", "value": 0}],
            },
            "rhs": {
                "kind": "index",
                "base": {"kind": "side", "side": "right", "name": "x"},
                "indices": [{"kind": "int", "value": 0}],
            },
        }
    )
    malformed = VerifiedDataflowContract(
        canonical_json=json.dumps(raw, sort_keys=True, separators=(",", ":"))
    )
    with pytest.raises(ValueError, match="tensor index rank mismatch"):
        render_verified_contract_to_verus(malformed)


def test_non_integer_singleton_coordinate_fails_closed() -> None:
    raw = _proved_silu_contract().to_data()
    raw["theorem_contract"]["theorem"]["singletons"].append(
        {
            "variable": "bi",
            "left": {
                "kind": "index",
                "base": {"kind": "side", "side": "left", "name": "x"},
                "indices": [
                    {"kind": "int", "value": 0},
                    {"kind": "int", "value": 0},
                ],
            },
            "right": {"kind": "int", "value": 0},
        }
    )
    malformed = VerifiedDataflowContract(
        canonical_json=json.dumps(raw, sort_keys=True, separators=(",", ":"))
    )
    with pytest.raises(ValueError, match="coordinates must be integers"):
        render_verified_contract_to_verus(malformed)
