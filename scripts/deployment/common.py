"""Architecture-neutral helpers for offline kernel qualification bundles."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import hashlib
import importlib
import json
from pathlib import Path
import sys
from typing import Callable


ROOT = Path(__file__).resolve().parents[2]
KERNEL_ROOT = ROOT / "kernels"
PYTHON_ROOT = ROOT / "python"
for path in (ROOT, PYTHON_ROOT, KERNEL_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from ir.backend_requirements import build_backend_requirement_manifest  # noqa: E402
from ir.proof_preparation import prepare_kernel_proofs  # noqa: E402
from ir.kernel_verifier import verify_kernel_goal  # noqa: E402
from ir.relational_artifact import VerifiedDataflowContract  # noqa: E402
from ir.exact_effect_artifact import VerifiedExactEffectContract  # noqa: E402
from backend.probes import run_candidate_requirements  # noqa: E402
from vosti_kernels.backend_evidence import (  # noqa: E402
    EVIDENCE_KIND,
    digest,
    discover_backend_environment,
    proof_specialization,
)
from vosti_kernels.deployment import candidate_digest, source_provenance  # noqa: E402
from vosti_kernels.kernel_selection import select_static_launches  # noqa: E402
from vosti_kernels.kernel_modules import validate_kernel_sources  # noqa: E402
from scripts.verification.engine_kernel_bindings import required_kernel_goal_bindings  # noqa: E402


CheckpointProfile = Callable[[Path], tuple[dict, dict, str]]
PrepareCandidate = Callable[[str | Path], dict]
SealCandidate = Callable[[dict, dict], dict]


@dataclass(frozen=True)
class DeploymentCandidateArchitecture:
    """Family inputs needed by the common offline candidate compiler."""

    family_label: str
    candidate_schema: str
    scope: Callable[[], dict]
    scope_sha256: Callable[[], str]
    checkpoint_profile: CheckpointProfile
    launch_inventory: Callable[[dict], list[dict]]
    bind_launches_to_proof_cases: Callable[..., list[dict]]
    validate_candidate: Callable[[dict], dict]


@dataclass(frozen=True)
class DeploymentPreparationArchitecture:
    """Family policy needed by the common prove/probe/seal operation."""

    family_label: str
    report_schema: str
    prepare_candidate: PrepareCandidate
    seal_candidate: SealCandidate


def _write_json(path: Path, value: dict) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def prepare_and_seal_deployment(
    architecture: DeploymentPreparationArchitecture,
    model: str | Path,
    output: str | Path,
) -> dict[str, dict]:
    """Prove, probe, and seal one qualified non-engine deployment bundle."""

    output_path = Path(output).expanduser().resolve()
    if output_path.exists() and any(output_path.iterdir()):
        raise ValueError(
            f"refusing to overwrite nonempty output directory: {output_path}"
        )
    output_path.mkdir(parents=True, exist_ok=True)

    candidate = architecture.prepare_candidate(model)
    _write_json(output_path / "deployment-candidate.json", candidate)
    results = run_candidate_requirements(candidate)
    report = {
        "schema": architecture.report_schema,
        "candidate_sha256": candidate["candidate_sha256"],
        "environment": discover_backend_environment(),
        "tested_device_index": __import__("torch").cuda.current_device(),
        "evidence_kind": EVIDENCE_KIND,
        "results": results,
    }
    _write_json(output_path / "backend-qualification-report.json", report)
    deployment = architecture.seal_candidate(candidate, report)
    _write_json(output_path / "deployment.json", deployment)
    return {
        "candidate": candidate,
        "report": report,
        "deployment": deployment,
    }


def prepare_proof_case(
    contract: dict,
    constants: dict,
    physical_float_dtypes: Sequence[str],
) -> dict:
    """Prove and describe one exact kernel specialization for any family."""

    return qualify_proof_case(contract, constants, physical_float_dtypes).receipt


@dataclass(frozen=True)
class QualifiedKernelCase:
    """One qualification produces both deployment evidence and raw contracts.

    Keeping the artifacts avoids reconstructing a theorem from a receipt or
    rerunning a different producer for the Verus interface.
    """

    receipt: dict
    contracts: tuple[VerifiedDataflowContract | VerifiedExactEffectContract, ...]


def required_proof_goals(contract: dict) -> tuple[str, ...]:
    return tuple(required_kernel_goal_bindings(contract).values())


def qualify_proof_case(
    contract: dict,
    constants: dict,
    physical_float_dtypes: Sequence[str],
) -> QualifiedKernelCase:
    """Qualify one exact source execution, with all required named goals."""

    source_file = contract["source"]
    kernel_name = contract["kernel"]
    goals = required_kernel_goal_bindings(contract)
    source_path = KERNEL_ROOT / "triton_kernels" / source_file
    source = source_path.read_text(encoding="utf-8")
    source_sha256 = hashlib.sha256(source.encode("utf-8")).hexdigest()
    if source_sha256 != contract["source_sha256"]:
        raise ValueError(f"kernel source differs from its qualification scope: {source_file}")
    kernel_proofs = prepare_kernel_proofs(source, kernel_name, constants)
    # Roles specify what the ENGINE needs, not how a proof is performed. The
    # shared verifier dispatches by typed propositions for every bound goal.
    qualified = {}
    counts = {}
    for role, goal in goals.items():
        artifact = verify_kernel_goal(kernel_proofs, goal, preserve_analyzer_conditions=role != "batch")
        expected = VerifiedExactEffectContract if role == "effect" else VerifiedDataflowContract
        if not isinstance(artifact, expected):
            raise ValueError(f"{role} qualification exported an incompatible proof kind: {kernel_name}")
        data = artifact.to_data()
        counts[role] = len(data["checks"] if role == "effect" else data["evidence"]["checks"])
        qualified[role] = artifact
    contracts = tuple(qualified.values())
    requirements = build_backend_requirement_manifest(
        kernel=kernel_proofs.kernel,
        source_name=source_file,
        source_sha256=source_sha256,
        constants=constants,
        physical_float_dtypes=physical_float_dtypes,
    )
    semantic = None
    if "selected" in qualified:
        semantic = {
            # Receipt role, not the artifact encoding: do not let a successful
            # batch goal stand in for the required selected-row relation.
            "kind": "conditional_selected_row",
            "contract_digest": qualified["selected"].digest,
            "check_count": counts["selected"],
        }
    if "effect" in qualified:
        semantic = {
            "kind": "exact_effect",
            "contract_digest": qualified["effect"].digest,
            "check_count": counts["effect"],
        }
    identity = {
        "source": source_file,
        "source_sha256": source_sha256,
        "kernel": kernel_name,
        "specialization": dict(sorted(constants.items())),
    }
    receipt = {
        "case_id": digest(identity)[:20],
        **identity,
        "structural_contract_digest": qualified["batch"].digest,
        "structural_check_count": counts["batch"],
        "semantic_proof": semantic,
        "backend_requirements": requirements,
    }
    return QualifiedKernelCase(receipt, tuple(contracts))


def _selected_proof_cases(
    deployment_scope: dict,
    launches: list[dict],
) -> list[tuple[dict, dict]]:
    contracts = {
        contract["wrapper"]: contract
        for contract in deployment_scope["kernel_contracts"]
    }
    selected = []
    seen = set()
    for launch in launches:
        contract = contracts[launch["wrapper"]]
        constants = proof_specialization(launch["config"])
        kernel_module = importlib.import_module(
            f"triton_kernels.{contract['module']}"
        )
        compiled_page_size = getattr(kernel_module, "PAGE_SIZE", None)
        if compiled_page_size is not None:
            constants["PAGE_BLOCK_SIZE"] = int(compiled_page_size)
        identity = (
            contract["source"],
            contract["kernel"],
            tuple(sorted(constants.items())),
        )
        if identity in seen:
            continue
        seen.add(identity)
        selected.append((contract, constants))
    return selected


def prepare_deployment_candidate(
    architecture: DeploymentCandidateArchitecture,
    model: str | Path,
    *,
    environment: dict | None = None,
) -> dict:
    """Compile one exact, qualified-but-not-engine-reachable candidate."""

    model_path = Path(model).expanduser().resolve()
    resolved_config, profile, config_sha256 = architecture.checkpoint_profile(
        model_path
    )
    profile_name = profile["model"]["name"]
    observed_environment = (
        discover_backend_environment() if environment is None else environment
    )
    if observed_environment.get("backend") != "cuda":
        raise ValueError(
            f"{architecture.family_label} deployment qualification requires CUDA"
        )

    deployment_scope = architecture.scope()
    validate_kernel_sources(KERNEL_ROOT, deployment_scope)
    launches = select_static_launches(
        deployment_scope["kernel_contracts"],
        architecture.launch_inventory(profile),
    )
    if launches != profile["launches"]:
        raise ValueError(
            "kernel-owned selectors differ from the attested model profile"
        )
    proof_cases = [
        prepare_proof_case(
            contract,
            constants,
            ("bfloat16", "float32"),
        )
        for contract, constants in _selected_proof_cases(
            deployment_scope, launches
        )
    ]
    launches = architecture.bind_launches_to_proof_cases(
        proof_cases,
        profile_name=profile_name,
        launches=launches,
    )
    body = {
        "schema": architecture.candidate_schema,
        "qualified": False,
        "engine_reachable": False,
        "architecture": deployment_scope["architecture"],
        "model": {
            "catalog_name": profile_name,
            "directory_name": model_path.name,
            "config_sha256": config_sha256,
            "resolved_config": json.loads(json.dumps(resolved_config)),
        },
        "environment": observed_environment,
        "runtime": {
            "dtype": "bfloat16",
            "device_type": deployment_scope["runtime"]["device_type"],
        },
        "selection_policy": {
            "batch_dimensions_used": [],
            "model_dimensions_used": True,
            "hardware_profile_observed": True,
            "architecture_scope": deployment_scope["architecture"],
        },
        "provenance": source_provenance(ROOT),
        "scope_sha256": architecture.scope_sha256(),
        "launches": launches,
        "kernel_cases": proof_cases,
    }
    candidate = {**body, "candidate_sha256": candidate_digest(body)}
    architecture.validate_candidate(candidate)
    return candidate


__all__ = [
    "DeploymentCandidateArchitecture",
    "DeploymentPreparationArchitecture",
    "prepare_and_seal_deployment",
    "prepare_deployment_candidate",
    "prepare_proof_case",
]
