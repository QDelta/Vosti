"""Architecture-neutral deployment identity and qualification utilities."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess



EVIDENCE_KIND = "empirical_falsification_not_formal_proof"
BACKEND_REQUIREMENTS_SCHEMA = "vosti.backend-requirements.v2"
_PROOF_INDEPENDENT_CONFIG_FIELDS = frozenset({"num_warps", "num_stages"})
_MANIFEST_FIELDS = {
    "schema",
    "kernel",
    "physical_float_dtypes",
    "site_count",
    "requirements",
    "manifest_sha256",
}
_REQUIREMENT_FIELDS = {
    "id",
    "sites",
    "operation",
    "probe",
    "properties",
    "consumers",
    "inputs",
    "output",
    "attributes",
}
_REPORT_FIELDS = {
    "schema",
    "candidate_sha256",
    "environment",
    "tested_device_index",
    "evidence_kind",
    "results",
}
_RESULT_FIELDS = {"requirement_id", "passed", "details", "launch_meta_configs"}


def is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def digest(value: object) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def proof_specialization(config: dict) -> dict:
    """Project a launch config to source-level logical parameters."""

    return {
        key: value
        for key, value in config.items()
        if key not in _PROOF_INDEPENDENT_CONFIG_FIELDS
    }


def discover_backend_environment() -> dict:
    """Observe deployment-relevant CUDA/compiler facts, excluding UUIDs."""

    import torch
    import triton

    if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
        raise RuntimeError("deployment preparation requires an accessible CUDA device")
    devices = []
    for index in range(torch.cuda.device_count()):
        properties = torch.cuda.get_device_properties(index)
        devices.append(
            {
                "name": properties.name,
                "compute_capability": f"{properties.major}.{properties.minor}",
                "total_memory_bytes": int(properties.total_memory),
            }
        )
    compatibility = {
        (device["compute_capability"], device["name"]) for device in devices
    }
    if len(compatibility) != 1:
        raise RuntimeError(
            "one static deployment currently requires homogeneous visible GPUs: "
            f"{devices}"
        )
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=driver_version",
                "--format=csv,noheader",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=10.0,
        )
        versions = sorted(set(completed.stdout.split()))
    except (
        OSError,
        subprocess.CalledProcessError,
        subprocess.TimeoutExpired,
    ) as error:
        raise RuntimeError(
            "deployment preparation cannot determine the NVIDIA driver version"
        ) from error
    if len(versions) != 1:
        raise RuntimeError(
            f"deployment preparation observed ambiguous driver versions: {versions}"
        )
    return {
        "backend": "cuda",
        "devices": devices,
        "torch_version": torch.__version__,
        "triton_version": triton.__version__,
        "cuda_runtime": torch.version.cuda,
        "driver_version": versions[0],
    }


def required_probe_ids(candidate: dict) -> set[str]:
    return set(required_probe_contexts(candidate))


def required_probe_contexts(candidate: dict) -> dict[str, list[dict]]:
    """Return each primitive obligation's exact sealed warp/stage contexts."""

    contexts_by_case: dict[str, set[tuple[int, int]]] = {}
    for launch in candidate["launches"]:
        config = launch["config"]
        contexts_by_case.setdefault(launch["kernel_case_id"], set()).add(
            (int(config["num_warps"]), int(config["num_stages"]))
        )
    contexts_by_requirement: dict[str, set[tuple[int, int]]] = {}
    for case in candidate["kernel_cases"]:
        contexts = contexts_by_case.get(case["case_id"])
        if not contexts:
            raise ValueError(
                f"deployment proof case {case['case_id']} has no sealed launch"
            )
        for requirement in case["backend_requirements"]["requirements"]:
            contexts_by_requirement.setdefault(requirement["id"], set()).update(
                contexts
            )
    return {
        requirement_id: [
            {"num_warps": warps, "num_stages": stages}
            for warps, stages in sorted(contexts)
        ]
        for requirement_id, contexts in contexts_by_requirement.items()
    }


def validate_backend_requirement_manifest(
    manifest: object,
    *,
    source: str,
    source_sha256: str,
    kernel: str,
    specialization: dict,
    runtime_dtype: str,
) -> set[str]:
    """Validate one architecture-independent primitive probe manifest."""

    if not isinstance(manifest, dict) or set(manifest) != _MANIFEST_FIELDS:
        raise ValueError("backend requirement manifest differs from its schema")
    if manifest.get("schema") != BACKEND_REQUIREMENTS_SCHEMA:
        raise ValueError("backend requirement manifest has an unknown schema")
    manifest_body = {
        key: value for key, value in manifest.items() if key != "manifest_sha256"
    }
    if digest(manifest_body) != manifest.get("manifest_sha256"):
        raise ValueError("backend requirement manifest digest is invalid")
    if manifest.get("kernel") != {
        "source": source,
        "source_sha256": source_sha256,
        "name": kernel,
        "specialization": specialization,
    }:
        raise ValueError("backend requirement manifest names the wrong proof case")
    physical_dtypes = manifest.get("physical_float_dtypes")
    if (
        not isinstance(physical_dtypes, list)
        or not all(isinstance(dtype, str) for dtype in physical_dtypes)
        or physical_dtypes != sorted(set(physical_dtypes))
        or runtime_dtype not in physical_dtypes
        or "float32" not in physical_dtypes
    ):
        raise ValueError("backend requirement physical dtype domain is invalid")
    requirements = manifest.get("requirements")
    if not isinstance(requirements, list) or not requirements:
        raise ValueError("backend requirement manifest is empty")
    if (
        isinstance(manifest.get("site_count"), bool)
        or not isinstance(manifest.get("site_count"), int)
        or manifest["site_count"] < len(requirements)
    ):
        raise ValueError("backend requirement manifest site count is invalid")
    requirement_ids: set[str] = set()
    for requirement in requirements:
        if not isinstance(requirement, dict) or set(requirement) != _REQUIREMENT_FIELDS:
            raise ValueError("backend requirement differs from the closed schema")
        requirement_body = {
            key: value for key, value in requirement.items() if key != "id"
        }
        requirement_id = requirement.get("id")
        if (
            not isinstance(requirement_id, str)
            or requirement_id != digest(requirement_body)[:20]
        ):
            raise ValueError("backend requirement identity is invalid")
        if requirement_id in requirement_ids:
            raise ValueError("backend requirement identity is duplicated")
        requirement_ids.add(requirement_id)
        if not requirement.get("sites") or not requirement.get("properties"):
            raise ValueError("backend requirement has no sites or properties")
    return requirement_ids


def validate_backend_qualification_report(
    candidate: dict,
    report: dict,
    *,
    report_schema: str,
) -> str:
    """Validate exact architecture-neutral probe coverage and return its digest."""

    if set(report) != _REPORT_FIELDS:
        raise ValueError("backend report differs from the closed schema")
    if report.get("schema") != report_schema:
        raise ValueError("unknown backend qualification report schema")
    if report.get("evidence_kind") != EVIDENCE_KIND:
        raise ValueError("backend report misstates its evidence kind")
    tested_device_index = report.get("tested_device_index")
    devices = candidate.get("environment", {}).get("devices", [])
    if (
        isinstance(tested_device_index, bool)
        or not isinstance(tested_device_index, int)
        or not 0 <= tested_device_index < len(devices)
    ):
        raise ValueError("backend report has no tested device index")
    if report.get("candidate_sha256") != candidate.get("candidate_sha256"):
        raise ValueError("backend report was produced for a different candidate")
    if report.get("environment") != candidate.get("environment"):
        raise ValueError("backend report environment differs from candidate")
    results = report.get("results")
    if not isinstance(results, list):
        raise ValueError("backend report has no result list")
    by_id = {}
    for result in results:
        if set(result) != _RESULT_FIELDS:
            raise ValueError("malformed backend probe result")
        requirement_id = result["requirement_id"]
        if (
            not isinstance(requirement_id, str)
            or not isinstance(result["passed"], bool)
            or not isinstance(result["details"], str)
            or not result["details"]
        ):
            raise ValueError("malformed backend probe result values")
        if requirement_id in by_id:
            raise ValueError(f"duplicate backend result {requirement_id}")
        by_id[requirement_id] = result
    required_contexts = required_probe_contexts(candidate)
    required = set(required_contexts)
    if set(by_id) != required:
        raise ValueError(
            "backend result coverage differs from requirements: "
            f"missing={sorted(required - by_id.keys())}, "
            f"extra={sorted(by_id.keys() - required)}"
        )
    failed = [item for item in by_id.values() if item["passed"] is not True]
    if failed:
        raise ValueError(f"backend qualification failed: {failed}")
    for requirement_id, result in by_id.items():
        if result.get("launch_meta_configs") != required_contexts[requirement_id]:
            raise ValueError(
                "backend qualification launch-meta coverage differs from "
                f"the sealed plan for {requirement_id}"
            )
    return digest(report)
