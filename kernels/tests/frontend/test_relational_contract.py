"""Tests for proof-gated, canonical relational contract export."""

import json
from pathlib import Path

import pytest

from ir.axis_projection_contract import normalize_axis_projection_contract
from ir.relational_artifact import VerifiedDataflowContract
from ir.relational_dataflow import prove_relational_dataflow_from_annotations


ROOT = Path(__file__).resolve().parents[2]
MATMUL = ROOT / "triton_kernels" / "matmul.py"
CONFIG = {"BLOCK_M": 16, "BLOCK_N": 64, "BLOCK_K": 64}
SILU_MUL = ROOT / "triton_kernels" / "silu_mul.py"
SILU_CONFIG = {"BLOCK_M": 1, "BLOCK_N": 4096}
EMBEDDING = ROOT / "triton_kernels" / "embedding.py"
EMBEDDING_CONFIG = {"D": 1024, "BLOCK_M": 1, "BLOCK_D": 1024}
RMSNORM = ROOT / "triton_kernels" / "rmsnorm.py"
RMSNORM_CONFIG = {"BLOCK_M": 1, "BLOCK_N": 1024}
QKV_MATMUL = ROOT / "triton_kernels" / "qkv_matmul.py"
QKV_CONFIG = {"BLOCK_M": 16, "BLOCK_N": 32, "BLOCK_K": 128}


def test_passed_proof_exports_exact_typed_contract():
    source = MATMUL.read_text(encoding="utf-8")
    result = prove_relational_dataflow_from_annotations(source, "matmul_kernel", CONFIG)

    assert result.proved
    assert result.verified_contract is not None
    data = result.verified_contract.to_data()
    assert data["theorem_contract"]["kernel"] == "matmul_kernel"
    assert data["theorem_contract"]["goal_name"] == "batch_invariance"
    assert data["theorem_contract"]["parameters"][0] == {
        "name": "a",
        "declared_type": {
            "kind": "tensor",
            "element": {"kind": "float"},
            "shape": [
                {"kind": "var", "name": "M"},
                {"kind": "var", "name": "K"},
            ],
        },
        "type": {
            "kind": "tensor",
            "element": {"kind": "float"},
            "shape": [
                {"kind": "var", "name": "M"},
                {"kind": "var", "name": "K"},
            ],
        },
    }
    assert data["theorem_contract"]["theorem"]["same"] == ["K", "N"]
    assert data["theorem_contract"]["theorem"]["post"][0]["kind"] == "region_equality"


def test_failed_proof_cannot_export_contract():
    source = MATMUL.read_text(encoding="utf-8").replace(
        "right(c)[0:1, 0:N]", "right(c)[1:2, 0:N]"
    )
    result = prove_relational_dataflow_from_annotations(source, "matmul_kernel", CONFIG)

    assert not result.proved
    assert result.verified_contract is None


def test_contract_digest_binds_source_and_constants():
    source = MATMUL.read_text(encoding="utf-8")
    first = prove_relational_dataflow_from_annotations(source, "matmul_kernel", CONFIG)
    second = prove_relational_dataflow_from_annotations(source, "matmul_kernel", dict(reversed(CONFIG.items())))

    assert first.verified_contract is not None
    assert second.verified_contract is not None
    assert first.verified_contract.canonical_json == second.verified_contract.canonical_json
    assert first.verified_contract.digest == second.verified_contract.digest


def test_matmul_normalizes_to_complete_axis_projection():
    result = prove_relational_dataflow_from_annotations(
        MATMUL.read_text(encoding="utf-8"), "matmul_kernel", CONFIG
    )
    assert result.verified_contract is not None

    projection = normalize_axis_projection_contract(result.verified_contract).to_data()
    assert projection["batch_symbol"] == "M"
    assert projection["selector"] == "x"
    assert projection["projected_inputs"] == ["a"]
    assert projection["shared_inputs"] == ["b"]
    assert projection["projected_outputs"] == ["c"]
    assert projection["parameter_types"]["a"]["element"] == {"kind": "float"}
    assert len(projection["parameter_types"]["a"]["shape"]) == 2
    assert projection["shared_dimensions"] == ["K", "N"]
    assert projection["shared_scalar_parameters"] == []
    assert projection["derived_preconditions"] == []
    assert projection["domain_preconditions"] == []


def test_qkv_normalizes_three_outputs_over_compact_projection_grid():
    result = prove_relational_dataflow_from_annotations(
        QKV_MATMUL.read_text(encoding="utf-8"),
        "qkv_matmul_kernel",
        QKV_CONFIG,
    )
    assert result.proved
    assert result.verified_contract is not None

    projection = normalize_axis_projection_contract(result.verified_contract).to_data()
    assert projection["batch_symbol"] == "M"
    assert projection["selector"] == "x"
    assert projection["projected_inputs"] == ["a"]
    assert projection["shared_inputs"] == ["wk", "wq", "wv"]
    assert projection["projected_outputs"] == ["ok", "oq", "ov"]
    assert projection["shared_dimensions"] == ["K", "KVN", "QN"]
    assert len(projection["domain_preconditions"]) == 3


def test_silu_mul_normalizes_both_inputs_as_complete_projections():
    result = prove_relational_dataflow_from_annotations(
        SILU_MUL.read_text(encoding="utf-8"), "silu_mul_kernel", SILU_CONFIG
    )
    assert result.proved
    assert result.verified_contract is not None

    projection = normalize_axis_projection_contract(result.verified_contract).to_data()
    assert projection["batch_symbol"] == "M"
    assert projection["selector"] == "b"
    assert projection["projected_inputs"] == ["x", "y"]
    assert projection["shared_inputs"] == []
    assert projection["projected_outputs"] == ["o"]
    assert projection["shared_dimensions"] == ["N"]
    assert projection["domain_preconditions"] == []


def test_embedding_normalizes_specialized_extent_and_whole_shared_tensor():
    result = prove_relational_dataflow_from_annotations(
        EMBEDDING.read_text(encoding="utf-8"),
        "embedding_kernel",
        EMBEDDING_CONFIG,
    )
    assert result.proved
    assert result.verified_contract is not None

    projection = normalize_axis_projection_contract(result.verified_contract).to_data()
    assert projection["projected_inputs"] == ["ids"]
    assert projection["shared_inputs"] == ["weight"]
    assert projection["projected_outputs"] == ["o"]
    assert projection["shared_dimensions"] == ["D", "V"]
    assert projection["shared_scalar_parameters"] == []
    assert projection["declared_parameter_types"]["o"]["shape"][1] == {
        "kind": "var",
        "name": "D",
    }
    assert projection["parameter_types"]["o"]["shape"][1] == {
        "kind": "int",
        "value": 1024,
    }
    assert len(projection["derived_preconditions"]) == 1
    assert projection["derived_preconditions"][0]["kind"] == "scalar"
    assert projection["domain_preconditions"] == []


def test_rmsnorm_preserves_typed_shared_scalar_and_rank_one_tensor():
    result = prove_relational_dataflow_from_annotations(
        RMSNORM.read_text(encoding="utf-8"),
        "rmsnorm_kernel",
        RMSNORM_CONFIG,
    )
    assert result.proved
    assert result.verified_contract is not None

    projection = normalize_axis_projection_contract(result.verified_contract).to_data()
    assert projection["projected_inputs"] == ["x"]
    assert projection["shared_inputs"] == ["w"]
    assert projection["shared_scalar_parameters"] == ["eps"]
    assert projection["parameter_types"]["eps"] == {"kind": "float"}
    assert projection["parameter_types"]["w"]["shape"] == [
        {"kind": "var", "name": "N"}
    ]
    assert projection["domain_preconditions"] == []


def test_declared_shape_cannot_disagree_with_proved_specialization():
    result = prove_relational_dataflow_from_annotations(
        EMBEDDING.read_text(encoding="utf-8"),
        "embedding_kernel",
        EMBEDDING_CONFIG,
    )
    assert result.verified_contract is not None
    data = result.verified_contract.to_data()
    output = next(item for item in data["theorem_contract"]["parameters"] if item["name"] == "o")
    output["type"]["shape"][1]["value"] = 2048
    modified = VerifiedDataflowContract(
        canonical_json=json.dumps(data, sort_keys=True, separators=(",", ":"))
    )

    with pytest.raises(ValueError, match="specialized type .* does not match"):
        normalize_axis_projection_contract(modified)


def test_theorem_trailing_extent_must_match_kernel_evidence():
    result = prove_relational_dataflow_from_annotations(
        MATMUL.read_text(encoding="utf-8"), "matmul_kernel", CONFIG
    )
    assert result.verified_contract is not None
    data = result.verified_contract.to_data()
    output = next(item for item in data["theorem_contract"]["parameters"] if item["name"] == "c")
    output["declared_type"]["shape"][1]["name"] = "K"
    output["type"]["shape"][1]["name"] = "K"
    modified = VerifiedDataflowContract(
        canonical_json=json.dumps(data, sort_keys=True, separators=(",", ":"))
    )

    with pytest.raises(ValueError, match="parameter types differ"):
        normalize_axis_projection_contract(modified)


def test_whole_shared_tensor_cannot_disappear_during_normalization():
    result = prove_relational_dataflow_from_annotations(
        EMBEDDING.read_text(encoding="utf-8"),
        "embedding_kernel",
        EMBEDDING_CONFIG,
    )
    assert result.verified_contract is not None
    data = result.verified_contract.to_data()
    data["theorem_contract"]["theorem"]["same"].remove("weight")
    modified = VerifiedDataflowContract(
        canonical_json=json.dumps(data, sort_keys=True, separators=(",", ":"))
    )

    with pytest.raises(ValueError, match="unclassified tensor parameters"):
        normalize_axis_projection_contract(modified)


def test_shared_scalar_parameter_cannot_disappear_during_normalization():
    result = prove_relational_dataflow_from_annotations(
        RMSNORM.read_text(encoding="utf-8"),
        "rmsnorm_kernel",
        RMSNORM_CONFIG,
    )
    assert result.verified_contract is not None
    data = result.verified_contract.to_data()
    data["theorem_contract"]["theorem"]["same"].remove("eps")
    modified = VerifiedDataflowContract(
        canonical_json=json.dumps(data, sort_keys=True, separators=(",", ":"))
    )

    with pytest.raises(ValueError, match="unclassified scalar parameters"):
        normalize_axis_projection_contract(modified)


def test_nonmatching_index_equality_remains_a_domain_precondition():
    result = prove_relational_dataflow_from_annotations(
        EMBEDDING.read_text(encoding="utf-8"),
        "embedding_kernel",
        EMBEDDING_CONFIG,
    )
    assert result.verified_contract is not None
    data = result.verified_contract.to_data()
    indexed = next(
        item
        for item in data["theorem_contract"]["theorem"]["pre"]
        if item.get("kind") == "scalar" and item.get("op") == "=="
        and item.get("rhs", {}).get("kind") == "index"
    )
    indexed["rhs"]["indices"][0]["value"] = 1
    modified = VerifiedDataflowContract(
        canonical_json=json.dumps(data, sort_keys=True, separators=(",", ":"))
    )

    projection = normalize_axis_projection_contract(modified).to_data()
    assert projection["derived_preconditions"] == []
    assert projection["domain_preconditions"] == [indexed]


def test_partial_output_region_cannot_normalize_as_whole_row():
    source = MATMUL.read_text(encoding="utf-8").replace(
        "0:N] == right(c)[0:1, 0:N]",
        "0:N-1] == right(c)[0:1, 0:N-1]",
    )
    result = prove_relational_dataflow_from_annotations(source, "matmul_kernel", CONFIG)
    assert result.proved
    assert result.verified_contract is not None

    with pytest.raises(ValueError, match="complete selected-row equality"):
        normalize_axis_projection_contract(result.verified_contract)


def test_unbound_shared_scalar_cannot_disappear_during_normalization():
    result = prove_relational_dataflow_from_annotations(
        MATMUL.read_text(encoding="utf-8"), "matmul_kernel", CONFIG
    )
    assert result.verified_contract is not None
    data = result.verified_contract.to_data()
    data["theorem_contract"]["theorem"]["same"].append("ambient")
    modified = VerifiedDataflowContract(
        canonical_json=json.dumps(data, sort_keys=True, separators=(",", ":"))
    )

    with pytest.raises(ValueError, match="shared dimensions must be exactly"):
        normalize_axis_projection_contract(modified)


def test_unknown_theorem_field_cannot_disappear_during_normalization():
    result = prove_relational_dataflow_from_annotations(
        MATMUL.read_text(encoding="utf-8"), "matmul_kernel", CONFIG
    )
    assert result.verified_contract is not None
    data = result.verified_contract.to_data()
    data["theorem_contract"]["theorem"]["future_assumption"] = []
    modified = VerifiedDataflowContract(
        canonical_json=json.dumps(data, sort_keys=True, separators=(",", ":"))
    )

    with pytest.raises(ValueError, match="missing or malformed theorem"):
        normalize_axis_projection_contract(modified)
