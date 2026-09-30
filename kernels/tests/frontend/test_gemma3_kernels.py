from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from backend.probe_contract import validate_requirement_probe_coverage
from ir.backend_requirements import build_backend_requirement_manifest
from diagnostics.race import prove_logical_write_disjointness_from_annotations
from ir.relational_dataflow import prove_relational_dataflow_from_annotations
from ir.proof_preparation import prepare_annotation_proof


KERNEL_DIR = Path(__file__).resolve().parents[2] / "triton_kernels"
GEMMA3_CASES = (
    (
        "gemma_rmsnorm.py",
        "gemma_rmsnorm_kernel",
        {"BLOCK_M": 1, "BLOCK_N": 4096},
    ),
    (
        "gelu_tanh_mul.py",
        "gelu_tanh_mul_kernel",
        {"BLOCK_M": 1, "BLOCK_N": 4096},
    ),
    (
        "add.py",
        "add_kernel",
        {"BLOCK_M": 1, "BLOCK_N": 4096},
    ),
    (
        "scaled_embedding.py",
        "scaled_embedding_kernel",
        {"BLOCK_M": 1, "BLOCK_D": 4096},
    ),
    (
        "gemma_qk_norm.py",
        "gemma_head_rms_norm_kernel",
        {"H": 8, "D": 256, "BLOCK_M": 1},
    ),
    (
        "gemma_qk_norm.py",
        "gemma_head_rms_norm_kernel",
        {"H": 4, "D": 256, "BLOCK_M": 1},
    ),
    (
        "rope.py",
        "rope_kernel",
        {"D": 256, "HD": 128, "BLOCK_M": 1},
    ),
    # Every Gemma 3 4B projection currently resolves to the conservative
    # default matmul tile.  N and K remain symbolic in the decomposition
    # theorem, so this one specialization covers q/k/v/o, gate/up, down, and
    # the tied LM head; the closed runtime inventory records their exact
    # individual shapes.
    (
        "matmul.py",
        "matmul_kernel",
        {"BLOCK_M": 16, "BLOCK_N": 64, "BLOCK_K": 64},
    ),
    (
        "store_kv_cache.py",
        "store_cache_kernel",
        {"KVD": 1024, "BLOCK_M": 1},
    ),
    (
        "fattn_paged.py",
        "fattn_varlen_paged_fwd_block_ptr_kernel",
        {
            "D_HEAD": 256,
            "PAGE_BLOCK_SIZE": 64,
            "BLOCK_M": 16,
            "BLOCK_N": 64,
        },
    ),
    (
        "fattn_paged_swa.py",
        "fattn_varlen_paged_swa_kernel",
        {
            "D_HEAD": 256,
            "PAGE_BLOCK_SIZE": 64,
            "BLOCK_M": 16,
            "BLOCK_N": 64,
        },
    ),
)

GEMMA3_SELECTED_ROW_CASES = (
    (
        "fattn_paged.py",
        "fattn_varlen_paged_fwd_block_ptr_kernel",
    ),
    (
        "fattn_paged_swa.py",
        "fattn_varlen_paged_swa_kernel",
    ),
)


def _source(filename: str) -> str:
    return (KERNEL_DIR / filename).read_text(encoding="utf-8")


@pytest.mark.parametrize("filename,kernel,constants", GEMMA3_CASES)
def test_gemma3_runtime_kernels_prove_batch_invariant_and_write_disjoint(
    filename: str, kernel: str, constants: dict
) -> None:
    source = _source(filename)
    relational = prove_relational_dataflow_from_annotations(
        source, kernel, constants, goal_name="batch_invariance"
    )
    assert relational.proved, relational.unsupported_reason

    race = prove_logical_write_disjointness_from_annotations(
        source,
        kernel,
        constants,
        goal_name="batch_invariance",
        timeout_ms=5000,
    )
    assert race.ok, [
        (check.name, check.details) for check in race.checks if not check.proved
    ]


@pytest.mark.parametrize("filename,kernel,constants", GEMMA3_CASES)
def test_gemma3_runtime_backend_assumptions_have_probe_rules(
    filename: str, kernel: str, constants: dict
) -> None:
    source = _source(filename)
    prepared = prepare_annotation_proof(
        source, kernel, constants, goal_name="batch_invariance"
    )
    manifest = build_backend_requirement_manifest(
        kernel=prepared.kernel,
        source_name=filename,
        source_sha256=hashlib.sha256(source.encode("utf-8")).hexdigest(),
        constants=constants,
        physical_float_dtypes=("bfloat16", "float32"),
    )
    assert manifest["requirements"]
    for requirement in manifest["requirements"]:
        validate_requirement_probe_coverage(requirement)


@pytest.mark.parametrize(
    "filename,kernel", GEMMA3_SELECTED_ROW_CASES
)
def test_gemma3_attention_proves_logical_causal_prefix_relocation(
    filename: str, kernel: str
) -> None:
    source = _source(filename)
    constants = {
        "D_HEAD": 256,
        "PAGE_BLOCK_SIZE": 64,
        "BLOCK_M": 16,
        "BLOCK_N": 64,
    }
    generic = prove_relational_dataflow_from_annotations(
        source,
        kernel,
        constants,
        goal_name="selected_row_prefix_equivalence",
    )
    assert generic.proved, generic.unsupported_reason

    assert generic.verified_contract is not None
    assert generic.used_assumptions == generic.external_obligations == (
        "finite(v_block)@masked-backward-dependency",
    )


def test_gemma_rmsnorm_decomposition_proof_does_not_fix_weight_semantics() -> None:
    source = _source("gemma_rmsnorm.py").replace(
        "normalized * (1.0 + w_block[None, :])",
        "normalized * w_block[None, :]",
    )
    result = prove_relational_dataflow_from_annotations(
        source,
        "gemma_rmsnorm_kernel",
        {"BLOCK_M": 1, "BLOCK_N": 4096},
        goal_name="batch_invariance",
    )
    # Batch invariance intentionally abstracts operation semantics: changing
    # one row-local expression does not invalidate the decomposition theorem.
    assert result.proved


def test_scaled_embedding_requires_equal_scale_between_runs() -> None:
    source = _source("scaled_embedding.py").replace(
        "same(V, D, weight, scale)", "same(V, D, weight)"
    )
    result = prove_relational_dataflow_from_annotations(
        source,
        "scaled_embedding_kernel",
        {"BLOCK_M": 1, "BLOCK_D": 4096},
        goal_name="batch_invariance",
    )
    assert not result.proved
    assert any(
        "value_scalar_equiv" in check.name and not check.proved
        for check in result.checks
    )


def test_sliding_attention_requires_equal_window_between_runs() -> None:
    source = _source("fattn_paged_swa.py").replace(
        "same(H, Hkv, D_HEAD, scale_log2, window_size)",
        "same(H, Hkv, D_HEAD, scale_log2)",
    )
    with pytest.raises(ValueError, match="window_size"):
        prove_relational_dataflow_from_annotations(
            source,
            "fattn_varlen_paged_swa_kernel",
            {
                "D_HEAD": 256,
                "PAGE_BLOCK_SIZE": 64,
                "BLOCK_M": 16,
                "BLOCK_N": 64,
            },
            goal_name="batch_invariance",
        )
