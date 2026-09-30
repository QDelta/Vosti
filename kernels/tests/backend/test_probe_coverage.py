from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

import backend.probes as probes
from backend.probe_contract import validate_requirement_probe_coverage
from backend.probes import _control_probe, _float_probe_dtype_names
from ir.backend_requirements import build_backend_requirement_manifest
from ir.proof_preparation import prepare_annotation_proof


ROOT = Path(__file__).resolve().parents[2]


def _requirements(filename: str, kernel: str, constants: dict) -> list[dict]:
    source = (ROOT / "triton_kernels" / filename).read_text(encoding="utf-8")
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
    return manifest["requirements"]


def test_every_exported_attention_property_has_an_explicit_probe_rule() -> None:
    requirements = _requirements(
        "fattn_paged.py",
        "fattn_varlen_paged_fwd_block_ptr_kernel",
        {
            "D_HEAD": 128,
            "BLOCK_M": 16,
            "BLOCK_N": 64,
            "PAGE_BLOCK_SIZE": 64,
        },
    )
    for requirement in requirements:
        validate_requirement_probe_coverage(requirement)


@pytest.mark.parametrize(
    "filename,kernel,constants",
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
        (
            "rope.py",
            "rope_kernel",
            {"D": 128, "HD": 64, "BLOCK_M": 1},
        ),
    ],
)
def test_every_engine_kernel_operation_has_property_coverage(
    filename: str, kernel: str, constants: dict
) -> None:
    for requirement in _requirements(filename, kernel, constants):
        validate_requirement_probe_coverage(requirement)


def test_new_unimplemented_property_fails_closed() -> None:
    requirement = _requirements(
        "matmul.py",
        "matmul_kernel",
        {"BLOCK_M": 16, "BLOCK_N": 64, "BLOCK_K": 64},
    )[0]
    requirement = {**requirement, "properties": ["new_unimplemented_assumption"]}
    with pytest.raises(ValueError, match="does not cover properties"):
        validate_requirement_probe_coverage(requirement)


def test_omitted_verifier_used_property_fails_closed() -> None:
    requirements = _requirements(
        "fattn_paged.py",
        "fattn_varlen_paged_fwd_block_ptr_kernel",
        {
            "D_HEAD": 128,
            "BLOCK_M": 16,
            "BLOCK_N": 64,
            "PAGE_BLOCK_SIZE": 64,
        },
    )
    requirement = next(
        item
        for item in requirements
        if item["operation"] == "triton.elementwise.binary"
        and item["attributes"].get("operator") == "+"
        and item["output"]["element"] == "abstract_float"
    )
    requirement = {
        **requirement,
        "properties": [
            property_name
            for property_name in requirement["properties"]
            if property_name != "negative_infinity_plus_finite_is_negative_infinity"
        ],
    }
    with pytest.raises(ValueError, match="omits verifier-used properties"):
        validate_requirement_probe_coverage(requirement)


def test_wrong_probe_family_fails_closed() -> None:
    requirement = _requirements(
        "matmul.py",
        "matmul_kernel",
        {"BLOCK_M": 16, "BLOCK_N": 64, "BLOCK_K": 64},
    )[0]
    requirement = {**requirement, "probe": "constructor"}
    with pytest.raises(ValueError, match="requires probe family"):
        validate_requirement_probe_coverage(requirement)


def test_unknown_analysis_consumer_fails_closed() -> None:
    requirement = _requirements(
        "matmul.py",
        "matmul_kernel",
        {"BLOCK_M": 16, "BLOCK_N": 64, "BLOCK_K": 64},
    )[0]
    requirement = {**requirement, "consumers": ["future_unreviewed_analysis"]}
    with pytest.raises(ValueError, match="invalid analysis consumers"):
        validate_requirement_probe_coverage(requirement)


def test_duplicate_analysis_consumer_fails_closed() -> None:
    requirement = _requirements(
        "matmul.py",
        "matmul_kernel",
        {"BLOCK_M": 16, "BLOCK_N": 64, "BLOCK_K": 64},
    )[0]
    consumer = requirement["consumers"][0]
    requirement = {**requirement, "consumers": [consumer, consumer]}
    with pytest.raises(ValueError, match="invalid analysis consumers"):
        validate_requirement_probe_coverage(requirement)


def test_control_probe_keeps_branch_and_loop_bound_runtime_visible() -> None:
    # Deployed attention has a data-dependent early exit and several kernels
    # have runtime loop bounds.  A constexpr-only probe would compile a
    # different, easier control-flow path.
    assert _control_probe.constexprs == []


def test_float_requirements_expand_to_the_sealed_physical_domain() -> None:
    requirement = next(
        item
        for item in _requirements(
            "matmul.py",
            "matmul_kernel",
            {"BLOCK_M": 16, "BLOCK_N": 64, "BLOCK_K": 64},
        )
        if item["operation"] == "triton.dot"
    )
    requirement = {
        **requirement,
        "_physical_float_dtypes": ["bfloat16", "float32"],
    }
    assert _float_probe_dtype_names(requirement) == ("bfloat16", "float32")


def test_float32_only_primitive_uses_only_applicable_sealed_dtype() -> None:
    requirement = next(
        item
        for item in _requirements(
            "silu_mul.py",
            "silu_mul_kernel",
            {"BLOCK_M": 1, "BLOCK_N": 4096},
        )
        if item["operation"] == "triton.sigmoid"
    )
    requirement = {
        **requirement,
        "_physical_float_dtypes": ["bfloat16", "float32"],
    }
    assert _float_probe_dtype_names(requirement) == ("float32",)


def test_float32_only_primitive_fails_when_sealed_domain_omits_float32() -> None:
    requirement = next(
        item
        for item in _requirements(
            "silu_mul.py",
            "silu_mul_kernel",
            {"BLOCK_M": 1, "BLOCK_N": 4096},
        )
        if item["operation"] == "triton.sigmoid"
    )
    requirement = {
        **requirement,
        "_physical_float_dtypes": ["bfloat16"],
    }
    with pytest.raises(ValueError, match="requires float32 qualification"):
        _float_probe_dtype_names(requirement)


def test_nonfloat_requirements_run_once_without_a_float_variant() -> None:
    requirement = next(
        item
        for item in _requirements(
            "embedding.py",
            "embedding_kernel",
            {"D": 1024, "BLOCK_M": 1, "BLOCK_D": 1024},
        )
        if item["operation"] == "triton.load.scalar"
    )
    requirement = {
        **requirement,
        "_physical_float_dtypes": ["bfloat16", "float32"],
    }
    assert _float_probe_dtype_names(requirement) == (None,)


def test_candidate_runner_executes_every_sealed_dtype_and_launch_meta(
    monkeypatch,
) -> None:
    requirement = next(
        item
        for item in _requirements(
            "matmul.py",
            "matmul_kernel",
            {"BLOCK_M": 16, "BLOCK_N": 64, "BLOCK_K": 64},
        )
        if item["operation"] == "triton.dot"
    )
    candidate = {
        "launches": [
            {
                "kernel_case_id": "matmul-case",
                "config": {"num_warps": 2, "num_stages": 4},
            },
            {
                "kernel_case_id": "matmul-case",
                "config": {"num_warps": 8, "num_stages": 2},
            },
        ],
        "kernel_cases": [
            {
                "case_id": "matmul-case",
                "backend_requirements": {
                    "physical_float_dtypes": ["bfloat16", "float32"],
                    "requirements": [requirement],
                }
            }
        ]
    }
    observed = []

    def fake_runner(probe_requirement, _device) -> None:
        observed.append(
            (
                probe_requirement.get("_probe_float_dtype"),
                dict(probes._ACTIVE_LAUNCH_META.get()),
            )
        )

    monkeypatch.setattr(probes.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(probes.torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(probes.torch.cuda, "synchronize", lambda _device: None)
    monkeypatch.setitem(probes._RUNNERS, "dot", fake_runner)

    results = probes.run_candidate_requirements(candidate)
    assert observed == [
        ("bfloat16", {"num_warps": 2, "num_stages": 4}),
        ("float32", {"num_warps": 2, "num_stages": 4}),
        ("bfloat16", {"num_warps": 8, "num_stages": 2}),
        ("float32", {"num_warps": 8, "num_stages": 2}),
    ]
    assert results == [
        {
            "requirement_id": requirement["id"],
            "passed": True,
            "details": (
                "passed empirical backend probes for applicable physical float "
                "dtypes bfloat16,float32 at warps=2/stages=4,warps=8/stages=2"
            ),
            "launch_meta_configs": [
                {"num_warps": 2, "num_stages": 4},
                {"num_warps": 8, "num_stages": 2},
            ],
        }
    ]
