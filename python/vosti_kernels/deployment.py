"""Architecture-neutral qualification schema for model runtimes.

Family modules supply an exact immutable scope plus profile/config resolvers.
This module validates source/proof inventory, static launch plans, backend
reports, seals, and runtime binding without importing any model family.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import subprocess
from typing import Any, Callable
from .kernel_interfaces import (load_attention_interfaces, validate_attention_interface_case,
                                load_mutation_interface, validate_mutation_interface_case,
                                load_rectangular_interface, validate_rectangular_interface_case,
                                RECTANGULAR_KERNEL_INTERFACES)

from .backend_evidence import (
    digest,
    proof_specialization,
    validate_backend_qualification_report,
    validate_backend_requirement_manifest,
)

CANDIDATE_SCHEMA = "vosti.deployment-candidate.v5"
REPORT_SCHEMA = "vosti.backend-qualification-report.v5"
DEPLOYMENT_SCHEMA = "vosti.static-deployment.v5"

_CANDIDATE_FIELDS = {
    "schema",
    "qualified",
    "engine_reachable",
    "architecture",
    "model",
    "environment",
    "runtime",
    "selection_policy",
    "provenance",
    "scope_sha256",
    "launches",
    "kernel_cases",
    "candidate_sha256",
}


def _standard_semantic_proof_kind(contract: dict) -> str | None:
    """Return the semantic certificate required by a reusable kernel role."""

    if contract.get("evidence") == "conditional_relational_certificate":
        return "conditional_selected_row"
    if contract.get("evidence") == "exact_effect_certificate":
        return "exact_effect"
    return None


@dataclass(frozen=True, slots=True, init=False)
class DeploymentArchitecture:
    """One family binding for the generic qualification workflow."""

    _scope: dict[str, Any] = field(repr=False)
    scope_sha256: str
    model_config: Callable[[dict], dict] = field(repr=False)
    model_profile_for_config: Callable[[dict], dict] = field(repr=False)
    model_profile_for_name: Callable[[str | None], dict] = field(repr=False)
    semantic_proof_kind: Callable[[dict], str | None] = field(repr=False)

    def __init__(
        self,
        *,
        scope: dict[str, Any],
        scope_sha256: str,
        model_config: Callable[[dict], dict],
        model_profile_for_config: Callable[[dict], dict],
        model_profile_for_name: Callable[[str | None], dict],
        semantic_proof_kind: Callable[
            [dict], str | None
        ] = _standard_semantic_proof_kind,
    ) -> None:
        detached_scope = json.loads(json.dumps(scope))
        if (
            detached_scope.get("status") != "qualification_scope"
            or detached_scope.get("engine_reachable") is not False
        ):
            raise ValueError(
                "architecture qualification scope must remain outside the engine"
            )
        if digest(detached_scope) != scope_sha256:
            raise ValueError("architecture qualification scope digest is invalid")
        object.__setattr__(self, "_scope", detached_scope)
        object.__setattr__(self, "scope_sha256", digest(detached_scope))
        object.__setattr__(self, "model_config", model_config)
        object.__setattr__(
            self, "model_profile_for_config", model_profile_for_config
        )
        object.__setattr__(
            self, "model_profile_for_name", model_profile_for_name
        )
        object.__setattr__(self, "semantic_proof_kind", semantic_proof_kind)

    def scope(self) -> dict[str, Any]:
        return json.loads(json.dumps(self._scope))

    @classmethod
    def from_profile(cls, profile):
        return cls(scope=profile.scope(), scope_sha256=profile.scope_sha256(),
                   model_config=profile.model_config,
                   model_profile_for_config=profile.model_profile_for_config,
                   model_profile_for_name=profile.model_profile_for_name)


_CASE_FIELDS = {
    "case_id",
    "source",
    "source_sha256",
    "kernel",
    "specialization",
    "structural_contract_digest",
    "structural_check_count",
    "semantic_proof",
    "backend_requirements",
}
_LAUNCH_FIELDS = {
    "wrapper",
    "sites",
    "key",
    "config",
    "proof_specialization",
    "kernel_case_id",
}


def scope(architecture: DeploymentArchitecture) -> dict[str, Any]:
    """Return a detached copy of the exact architecture scope."""

    return architecture.scope()


def scope_sha256(architecture: DeploymentArchitecture) -> str:
    return architecture.scope_sha256


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def source_provenance(root: str | Path) -> dict:
    """Best-effort base revision, not an attestation of the working files."""
    try:
        revision = subprocess.run(
            ["git", "-C", str(Path(root).resolve()), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        revision = None
    return {"base_revision": revision or None}


def candidate_digest(candidate: dict) -> str:
    """Identity of qualification inputs, excluding informational provenance."""
    return digest({key: value for key, value in candidate.items()
                   if key not in {"candidate_sha256", "provenance"}})


def _contracts_by_wrapper(
    architecture: DeploymentArchitecture,
) -> dict[str, dict]:
    contracts = architecture.scope()["kernel_contracts"]
    result = {contract["wrapper"]: contract for contract in contracts}
    if len(result) != len(contracts):
        raise RuntimeError("architecture scope has duplicate kernel wrappers")
    return result


def bind_launches_to_proof_cases(
    architecture: DeploymentArchitecture,
    proof_cases: list[dict],
    *,
    profile_name: str | None = None,
    launches: list[dict] | None = None,
) -> list[dict]:
    """Bind every exact scoped launch to one source/specialization case."""

    by_identity = {}
    for case in proof_cases:
        identity = (
            case["source"],
            case["kernel"],
            tuple(sorted(case["specialization"].items())),
        )
        if identity in by_identity:
            raise ValueError(f"duplicate proof case {identity}")
        by_identity[identity] = case
    contracts = _contracts_by_wrapper(architecture)
    profile = architecture.model_profile_for_name(profile_name)
    scoped_launches = profile["launches"] if launches is None else launches
    bound = []
    launched_case_ids = set()
    for launch in scoped_launches:
        contract = contracts[launch["wrapper"]]
        configured = proof_specialization(launch["config"])
        matches = [
            case
            for identity, case in by_identity.items()
            if identity[0] == contract["source"]
            and identity[1] == contract["kernel"]
            and all(
                case["specialization"].get(key) == value
                for key, value in configured.items()
            )
        ]
        if len(matches) != 1:
            raise ValueError(
                f"launch {launch['wrapper']!r} has {len(matches)} matching "
                "proof cases"
            )
        case = matches[0]
        launched_case_ids.add(case["case_id"])
        bound.append(
            {
                **launch,
                "proof_specialization": case["specialization"],
                "kernel_case_id": case["case_id"],
            }
        )
    if launched_case_ids != {case["case_id"] for case in proof_cases}:
        raise ValueError("proof inventory contains an unlaunched case")
    return bound


def validate_candidate(
    architecture: DeploymentArchitecture,
    candidate: dict,
) -> dict[str, dict]:
    """Validate one exact family scope, proof inventory, and launch plan."""

    scope = architecture.scope()

    if set(candidate) != _CANDIDATE_FIELDS:
        raise ValueError("deployment candidate differs from the closed schema")
    if candidate.get("schema") != CANDIDATE_SCHEMA:
        raise ValueError("unknown deployment candidate schema")
    if candidate.get("qualified") is not False:
        raise ValueError("deployment candidate must be explicitly unqualified")
    if candidate.get("engine_reachable") is not False:
        raise ValueError("deployment candidate must remain outside the engine")
    if candidate.get("architecture") != scope["architecture"]:
        raise ValueError("deployment candidate names the wrong architecture")
    provenance = candidate.get("provenance")
    if (not isinstance(provenance, dict) or set(provenance) != {"base_revision"}
            or (provenance["base_revision"] is not None
                and not isinstance(provenance["base_revision"], str))):
        raise ValueError("deployment candidate has invalid provenance metadata")
    if candidate.get("scope_sha256") != scope_sha256(architecture):
        raise ValueError("deployment candidate scope digest is invalid")
    if candidate_digest(candidate) != candidate.get("candidate_sha256"):
        raise ValueError("deployment candidate content digest is invalid")

    runtime = candidate.get("runtime")
    scope_dtype = scope["runtime"]["dtype"]
    if not isinstance(scope_dtype, str) or not scope_dtype:
        raise ValueError("architecture scope has no runtime dtype")
    expected_runtime = {
        "dtype": scope_dtype.removeprefix("torch."),
        "device_type": scope["runtime"]["device_type"],
    }
    if runtime != expected_runtime:
        raise ValueError("deployment candidate runtime differs from its scope")
    if candidate.get("selection_policy") != {
        "batch_dimensions_used": [],
        "model_dimensions_used": True,
        "hardware_profile_observed": True,
        "architecture_scope": scope["architecture"],
    }:
        raise ValueError("deployment candidate selection policy differs from scope")
    environment = candidate.get("environment")
    if not isinstance(environment, dict) or environment.get("backend") != "cuda":
        raise ValueError("deployment candidate requires an observed CUDA environment")
    model = candidate.get("model")
    if not isinstance(model, dict) or set(model) != {
        "catalog_name",
        "directory_name",
        "config_sha256",
        "resolved_config",
    }:
        raise ValueError("deployment candidate has invalid model metadata")
    if not isinstance(model["directory_name"], str) or not model["directory_name"]:
        raise ValueError("deployment candidate has no model directory identity")
    if not _is_sha256(model["config_sha256"]):
        raise ValueError("deployment candidate has no config digest")
    try:
        profile = architecture.model_profile_for_name(model["catalog_name"])
    except ValueError as error:
        raise ValueError("deployment candidate names an unknown model profile") from error
    expected_config = architecture.model_config(profile)
    if model["resolved_config"] != expected_config:
        raise ValueError("deployment candidate resolved config is outside scope")

    proof_cases = candidate.get("kernel_cases")
    launches = candidate.get("launches")
    if not isinstance(proof_cases, list) or not isinstance(launches, list):
        raise ValueError("deployment candidate has no cases or launches")
    contracts = _contracts_by_wrapper(architecture)
    cases_by_id: dict[str, dict] = {}
    attention_interfaces = None
    mutation_interface = None
    rectangular_interfaces = {}
    for case in proof_cases:
        if not isinstance(case, dict) or set(case) != _CASE_FIELDS:
            raise ValueError("proof case differs from its closed schema")
        identity = {
            "source": case.get("source"),
            "source_sha256": case.get("source_sha256"),
            "kernel": case.get("kernel"),
            "specialization": case.get("specialization"),
        }
        case_id = case.get("case_id")
        if not isinstance(case_id, str) or case_id != digest(identity)[:20]:
            raise ValueError("proof case identity is invalid")
        if case_id in cases_by_id:
            raise ValueError("proof case identity is duplicated")
        source_contracts = [
            contract
            for contract in contracts.values()
            if contract["source"] == case["source"]
            and contract["kernel"] == case["kernel"]
        ]
        if len(source_contracts) != 1:
            raise ValueError("proof case is absent from the kernel catalog")
        if case["source_sha256"] != source_contracts[0]["source_sha256"]:
            raise ValueError("proof case source digest differs from scope")
        if not _is_sha256(case.get("structural_contract_digest")) or not (
            isinstance(case.get("structural_check_count"), int)
            and not isinstance(case.get("structural_check_count"), bool)
            and case["structural_check_count"] > 0
        ):
            raise ValueError("proof case has no structural certificate")
        semantic = case.get("semantic_proof")
        semantic_kind = architecture.semantic_proof_kind(source_contracts[0])
        if semantic_kind is not None:
            if not (
                isinstance(semantic, dict)
                and set(semantic) == {"kind", "contract_digest", "check_count"}
                and semantic.get("kind") == semantic_kind
                and _is_sha256(semantic.get("contract_digest"))
                and isinstance(semantic.get("check_count"), int)
                and not isinstance(semantic.get("check_count"), bool)
                and semantic["check_count"] > 0
            ):
                raise ValueError(
                    "proof case lacks its required semantic certificate"
                )
        elif semantic is not None:
            raise ValueError(
                "proof case claims an unexpected semantic certificate"
            )
        if semantic_kind == "conditional_selected_row":
            if attention_interfaces is None:
                attention_interfaces = load_attention_interfaces()
            validate_attention_interface_case(case, attention_interfaces)
        if semantic_kind == "exact_effect":
            if mutation_interface is None:
                mutation_interface = load_mutation_interface()
            validate_mutation_interface_case(case, mutation_interface)
        interface_name = RECTANGULAR_KERNEL_INTERFACES.get((case["source"], case["kernel"]))
        if interface_name is not None:
            if interface_name not in rectangular_interfaces:
                rectangular_interfaces[interface_name] = load_rectangular_interface(interface_name)
            validate_rectangular_interface_case(case, rectangular_interfaces[interface_name])
        validate_backend_requirement_manifest(
            case.get("backend_requirements"),
            source=case["source"],
            source_sha256=case["source_sha256"],
            kernel=case["kernel"],
            specialization=case["specialization"],
            runtime_dtype=runtime["dtype"],
        )
        cases_by_id[case_id] = case

    expected_launches = bind_launches_to_proof_cases(
        architecture, proof_cases, profile_name=model["catalog_name"]
    )
    if launches != expected_launches:
        raise ValueError("launch plan differs from the closed scope")
    for launch in launches:
        if set(launch) != _LAUNCH_FIELDS:
            raise ValueError("launch differs from its closed schema")
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in launch["config"].values()
        ):
            raise ValueError("launch config must contain positive integers")
        if launch["kernel_case_id"] not in cases_by_id:
            raise ValueError("launch references an unknown proof case")
    return cases_by_id


def seal_candidate(
    architecture: DeploymentArchitecture,
    candidate: dict,
    report: dict,
) -> dict:
    """Seal backend evidence while preserving non-engine status."""

    validate_candidate(architecture, candidate)
    report_sha256 = validate_backend_qualification_report(
        candidate,
        report,
        report_schema=REPORT_SCHEMA,
    )
    deployment = {
        "schema": DEPLOYMENT_SCHEMA,
        "qualified": True,
        "engine_reachable": False,
        "candidate_sha256": candidate["candidate_sha256"],
        "qualification_report_sha256": report_sha256,
        "architecture": candidate["architecture"],
        "model": candidate["model"],
        "environment": candidate["environment"],
        "runtime": candidate["runtime"],
        "scope_sha256": candidate["scope_sha256"],
        "launches": candidate["launches"],
    }
    return {**deployment, "deployment_sha256": digest(deployment)}


def load_bundle(
    architecture: DeploymentArchitecture,
    directory: str | Path,
) -> dict[str, dict]:
    root = Path(directory).expanduser().resolve()
    candidate = json.loads((root / "deployment-candidate.json").read_text())
    report = json.loads(
        (root / "backend-qualification-report.json").read_text()
    )
    deployment = json.loads((root / "deployment.json").read_text())
    reconstructed = seal_candidate(architecture, candidate, report)
    if deployment != reconstructed:
        raise ValueError("deployment differs from its candidate/report")
    return {
        "candidate": candidate,
        "report": report,
        "deployment": deployment,
    }


def validate_runtime_binding(
    architecture: DeploymentArchitecture,
    bundle: dict,
    *,
    resolved_config: dict,
    model_config_sha256: str,
    environment: dict,
) -> dict:
    """Bind one reconstructed deployment seal to the runtime about to execute."""

    if not isinstance(bundle, dict) or set(bundle) != {
        "candidate",
        "report",
        "deployment",
    }:
        raise ValueError("runtime received a malformed deployment bundle")
    candidate = bundle["candidate"]
    report = bundle["report"]
    deployment = bundle["deployment"]
    reconstructed = seal_candidate(architecture, candidate, report)
    if deployment != reconstructed:
        raise ValueError("runtime deployment bundle does not reconstruct exactly")
    if deployment.get("engine_reachable") is not False:
        raise ValueError("runtime deployment bundle illegally claims engine reachability")
    expected_config = json.loads(json.dumps(resolved_config))
    if deployment["model"]["resolved_config"] != expected_config:
        raise ValueError("runtime model config differs from the deployment seal")
    if deployment["model"]["config_sha256"] != model_config_sha256:
        raise ValueError("runtime model config digest differs from the deployment seal")
    if deployment["scope_sha256"] != scope_sha256(architecture):
        raise ValueError("runtime scope differs from the deployment seal")
    if deployment["environment"] != environment:
        raise ValueError("runtime environment differs from the deployment seal")
    profile = architecture.model_profile_for_config(expected_config)
    if deployment["model"]["catalog_name"] != profile["model"]["name"]:
        raise ValueError("runtime model profile differs from the deployment seal")
    projected = [
        {
            key: value
            for key, value in launch.items()
            if key not in {"kernel_case_id", "proof_specialization"}
        }
        for launch in deployment["launches"]
    ]
    if projected != profile["launches"]:
        raise ValueError("runtime launches differ from its source scope")
    for launch in deployment["launches"]:
        configured = proof_specialization(launch["config"])
        if any(
            launch["proof_specialization"].get(key) != value
            for key, value in configured.items()
        ):
            raise ValueError("runtime launch proof specialization differs from config")
    return deployment


__all__ = [
    "CANDIDATE_SCHEMA",
    "DEPLOYMENT_SCHEMA",
    "REPORT_SCHEMA",
    "DeploymentArchitecture",
    "source_provenance",
    "candidate_digest",
    "bind_launches_to_proof_cases",
    "load_bundle",
    "seal_candidate",
    "scope",
    "scope_sha256",
    "validate_runtime_binding",
    "validate_candidate",
]
