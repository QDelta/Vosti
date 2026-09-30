"""Prepare an unsealed backend-qualification candidate for Llama 3."""

from __future__ import annotations

import hashlib
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[3]
KERNEL_ROOT = ROOT / "kernels"
PYTHON_ROOT = ROOT / "python"
for path in (PYTHON_ROOT, KERNEL_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from scripts.deployment.common import (  # noqa: E402
    DeploymentCandidateArchitecture,
    prepare_deployment_candidate,
)
from vosti_kernels.model_families.llama3.deployment import (  # noqa: E402
    CANDIDATE_SCHEMA,
    bind_launches_to_proof_cases,
    scope,
    scope_sha256,
    validate_candidate,
)
from vosti_kernels.model_families.llama3.loader import (  # noqa: E402
    inspect_text_checkpoint,
)
from vosti_kernels.model_families.llama3.profile import (  # noqa: E402
    launch_inventory,
    model_profile_for_config,
)


def _checkpoint_profile(model_path: Path) -> tuple[dict, dict, str]:
    config_path = model_path / "config.json"
    if not config_path.is_file():
        raise ValueError(f"model directory has no config.json: {model_path}")
    config_bytes = config_path.read_bytes()
    config = inspect_text_checkpoint(model_path)
    resolved_config = config.as_runtime_dict()
    profile = model_profile_for_config(resolved_config)
    return resolved_config, profile, hashlib.sha256(config_bytes).hexdigest()


CANDIDATE_ARCHITECTURE = DeploymentCandidateArchitecture(
    family_label="Llama 3",
    candidate_schema=CANDIDATE_SCHEMA,
    scope=scope,
    scope_sha256=scope_sha256,
    checkpoint_profile=_checkpoint_profile,
    launch_inventory=launch_inventory,
    bind_launches_to_proof_cases=bind_launches_to_proof_cases,
    validate_candidate=validate_candidate,
)


def prepare_candidate(
    model: str | Path,
    *,
    environment: dict | None = None,
) -> dict:
    """Validate the checkpoint and prove its exact static primitive plan."""

    return prepare_deployment_candidate(
        CANDIDATE_ARCHITECTURE,
        model,
        environment=environment,
    )


__all__ = ["CANDIDATE_ARCHITECTURE", "prepare_candidate"]
