"""Gemma-4 checkpoint adapter for the shared offline candidate compiler."""

import hashlib
from pathlib import Path

from scripts.deployment.common import DeploymentCandidateArchitecture, prepare_deployment_candidate
from vosti_kernels.model_families.gemma4 import deployment, profile
from vosti_kernels.model_families.gemma4.loader import inspect_text_checkpoint


def _checkpoint_profile(model_path: Path) -> tuple[dict, dict, str]:
    resolved = inspect_text_checkpoint(model_path).as_runtime_dict()
    return resolved, profile.model_profile_for_config(resolved), hashlib.sha256(
        (model_path / "config.json").read_bytes()).hexdigest()


CANDIDATE_ARCHITECTURE = DeploymentCandidateArchitecture(
    family_label="Gemma 4", candidate_schema=deployment.CANDIDATE_SCHEMA,
    scope=deployment.scope, scope_sha256=deployment.scope_sha256,
    checkpoint_profile=_checkpoint_profile, launch_inventory=profile.launch_inventory,
    bind_launches_to_proof_cases=deployment.bind_launches_to_proof_cases,
    validate_candidate=deployment.validate_candidate)


def prepare_candidate(model: str | Path, *, environment: dict | None = None) -> dict:
    return prepare_deployment_candidate(CANDIDATE_ARCHITECTURE, model, environment=environment)
