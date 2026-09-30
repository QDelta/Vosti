from __future__ import annotations

import hashlib
from pathlib import Path

from ir.backend_requirements import (
    SCHEMA,
    build_backend_requirement_manifest,
)
from ir.proof_preparation import prepare_annotation_proof


KERNEL_DIR = Path(__file__).resolve().parents[2] / "triton_kernels"


def _manifest(filename: str, kernel_name: str, constants: dict) -> dict:
    path = KERNEL_DIR / filename
    source = path.read_text(encoding="utf-8")
    prepared = prepare_annotation_proof(
        source, kernel_name, constants, goal_name="batch_invariance"
    )
    return build_backend_requirement_manifest(
        kernel=prepared.kernel,
        source_name=filename,
        source_sha256=hashlib.sha256(source.encode("utf-8")).hexdigest(),
        constants=constants,
        physical_float_dtypes=("bfloat16", "float32"),
    )


def test_attention_exports_previously_erased_operations_with_exact_shapes() -> None:
    manifest = _manifest(
        "fattn_paged.py",
        "fattn_varlen_paged_fwd_block_ptr_kernel",
        {
            "D_HEAD": 128,
            "BLOCK_M": 16,
            "BLOCK_N": 64,
            "PAGE_BLOCK_SIZE": 64,
        },
    )

    assert manifest["schema"] == SCHEMA
    assert manifest["site_count"] > len(manifest["requirements"])
    casts = [
        requirement
        for requirement in manifest["requirements"]
        if requirement["operation"] == "triton.cast"
    ]
    assert {tuple(cast["output"]["shape"]) for cast in casts} == {
        ("16", "64"),
        ("16", "128"),
    }
    assert all("negative_infinity_is_preserved" in cast["properties"] for cast in casts)

    log2 = [
        requirement
        for requirement in manifest["requirements"]
        if requirement["operation"] == "triton.log2"
    ]
    assert len(log2) == 1
    assert log2[0]["inputs"][0]["shape"] == ["16"]
    assert log2[0]["output"]["shape"] == ["16"]

    program_ids = [
        requirement
        for requirement in manifest["requirements"]
        if requirement["operation"] == "triton.program_id"
    ]
    assert {item["attributes"]["axis"] for item in program_ids} == {0, 1, 2}

    binary_by_operator = {
        requirement["attributes"]["operator"]: requirement
        for requirement in manifest["requirements"]
        if requirement["operation"] == "triton.elementwise.binary"
    }
    assert "declared_comparison_semantics" in binary_by_operator["<"]["properties"]
    assert "declared_boolean_semantics" in binary_by_operator["and"]["properties"]
    assert (
        "declared_nonnegative_integer_semantics"
        in binary_by_operator["%"]["properties"]
    )

    arithmetic_properties = {
        property_name
        for requirement in manifest["requirements"]
        if requirement["operation"]
        in {"triton.elementwise.binary", "triton.assignment_update"}
        for property_name in requirement["properties"]
    }
    assert {
        "negative_infinity_plus_finite_is_negative_infinity",
        "finite_self_subtraction_is_positive_zero",
        "negative_infinity_minus_finite_is_negative_infinity",
        "finite_zero_product_is_numerical_zero",
        "finite_one_multiplicative_identity",
        "negative_infinity_times_positive_is_negative_infinity",
    } <= arithmetic_properties
    assert "finite_zero_additive_identity" not in arithmetic_properties

    reduce_max = [
        requirement
        for requirement in manifest["requirements"]
        if requirement["operation"] == "triton.reduce_max"
    ]
    assert reduce_max
    assert all(
        "constant_value_is_preserved" in requirement["properties"]
        for requirement in reduce_max
    )


def test_matmul_exports_tile_local_dot_and_masked_memory_contracts() -> None:
    manifest = _manifest(
        "matmul.py",
        "matmul_kernel",
        {"BLOCK_M": 16, "BLOCK_N": 64, "BLOCK_K": 64},
    )
    operations = {item["operation"] for item in manifest["requirements"]}
    assert {
        "triton.dot",
        "triton.load.block",
        "triton.store.block",
        "triton.cast",
    } <= operations

    dot = next(
        item for item in manifest["requirements"] if item["operation"] == "triton.dot"
    )
    assert dot["inputs"][0]["shape"] == ["16", "64"]
    assert dot["inputs"][1]["shape"] == ["64", "64"]
    assert dot["output"]["shape"] == ["16", "64"]
    assert "finite_zero_lane_has_no_contribution" in dot["properties"]


def test_permitted_address_cast_exports_signed_int32_identity() -> None:
    manifest = _manifest(
        "embedding.py",
        "embedding_kernel",
        {"D": 3072, "BLOCK_M": 1, "BLOCK_D": 4096},
    )
    integer_casts = [
        requirement
        for requirement in manifest["requirements"]
        if requirement["operation"] == "triton.cast"
        and requirement["attributes"]["proof_kind"] == "int32"
    ]
    assert integer_casts
    assert all(
        "signed_int32_identity" in requirement["properties"]
        for requirement in integer_casts
    )


def test_manifest_is_canonical_and_binds_specialization() -> None:
    first = _manifest(
        "matmul.py",
        "matmul_kernel",
        {"BLOCK_M": 16, "BLOCK_N": 64, "BLOCK_K": 64},
    )
    same = _manifest(
        "matmul.py",
        "matmul_kernel",
        {"BLOCK_K": 64, "BLOCK_N": 64, "BLOCK_M": 16},
    )
    different = _manifest(
        "matmul.py",
        "matmul_kernel",
        {"BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_K": 64},
    )
    assert first == same
    assert first["manifest_sha256"] != different["manifest_sha256"]


def test_requirement_exporter_is_not_a_proof_dependency() -> None:
    for name in ("proof_preparation.py", "regional_obligations.py", "smt.py",
                 "relational_verifier.py", "relational_dataflow.py"):
        proof_driver = (KERNEL_DIR.parent / "ir" / name).read_text(encoding="utf-8")
        assert "backend_requirements" not in proof_driver
