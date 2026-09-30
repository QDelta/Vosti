"""Architecture-neutral helpers for immutable model-profile registries."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Sequence


def detached(value: Any) -> Any:
    """Return a JSON-detached copy of profile data."""

    return json.loads(json.dumps(value))


def canonical(value: Any) -> Any:
    """Return the canonical JSON form accepted by deployment profiles."""

    try:
        return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))
    except (TypeError, ValueError) as error:
        raise ValueError("model profiles require canonical JSON data") from error


def load_scope(path: str | Path) -> dict[str, Any]:
    """Expand per-wrapper bindings using the shared kernel source catalog."""
    path = Path(path)
    with path.open(encoding="utf-8") as scope_file:
        value = json.load(scope_file)
    if not isinstance(value, dict):
        raise RuntimeError("model profile scope must be a JSON object")
    catalog = json.loads((path.parents[2] / "kernel_catalog.json").read_text())
    if set(catalog) != {"kernels", "runtime_support_sources"}:
        raise RuntimeError("kernel catalog has an invalid schema")
    if "runtime_support_sources" in value:
        raise RuntimeError("runtime support sources belong to the shared catalog")
    identities = {"module", "entrypoint", "source", "source_sha256"}
    kernels = catalog["kernels"]
    if not isinstance(kernels, dict) or not kernels or any(
            not isinstance(entry, dict) or set(entry) != identities for entry in kernels.values()):
        raise RuntimeError("kernel catalog has invalid source identities")
    for contract in value["kernel_contracts"]:
        if identities & set(contract):
            raise RuntimeError("family binding must not override a shared kernel identity")
        kernel = contract["kernel"]
        if kernel not in kernels:
            raise RuntimeError(f"unknown kernel catalog identity: {kernel}")
        contract.update(kernels[kernel])
    value["runtime_support_sources"] = catalog["runtime_support_sources"]
    return value


def scope_sha256(scope: dict[str, Any]) -> str:
    payload = json.dumps(
        scope, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


# @kernel-bridge-begin vosti_kernels::profile_validation
def is_sha256(value: object) -> bool:
    return (isinstance(value, str) and len(value) == 64
            and all(character in "0123456789abcdef" for character in value))


def validate_kernel_catalog(
    contracts: object, *, fields: set[str], family_label: str,
) -> set[str]:
    """Check the common exact catalog schema, source hashes and wrapper identities."""

    if not isinstance(contracts, list) or not contracts:
        raise RuntimeError(f"{family_label} scope has no kernel contracts")
    wrappers = set()
    for contract in contracts:
        if not isinstance(contract, dict) or set(contract) != fields:
            raise RuntimeError(f"{family_label} scope has an invalid kernel contract")
        if not is_sha256(contract["source_sha256"]):
            raise RuntimeError(f"{family_label} scope has an invalid kernel source digest")
        wrapper = contract["wrapper"]
        if not isinstance(wrapper, str) or not wrapper:
            raise RuntimeError(f"{family_label} scope has an invalid kernel wrapper")
        if wrapper in wrappers:
            raise RuntimeError(f"{family_label} scope has duplicate kernel wrappers")
        wrappers.add(wrapper)
    return wrappers


def validate_scope_metadata(
    scope: dict[str, Any], *, fields: set[str], architecture: str, family_label: str,
) -> None:
    """Validate the common closed qualification envelope, not model semantics."""

    if set(scope) != fields:
        raise RuntimeError(f"{family_label} scope differs from its closed schema")
    if scope.get("schema") != "vosti.model-runtime-scope.v1":
        raise RuntimeError(f"{family_label} scope has an unsupported schema")
    if (scope.get("status") != "qualification_scope"
            or scope.get("engine_reachable") is not False):
        raise RuntimeError(f"{family_label} qualification scope must remain fail-closed")
    if scope.get("architecture") != architecture:
        raise RuntimeError(f"{family_label} scope names the wrong architecture")
    if scope.get("runtime") != {"dtype": "torch.bfloat16", "device_type": "cuda"}:
        raise RuntimeError(f"{family_label} scope has unsupported runtime metadata")


def validate_profile_registry(
    profiles: list[dict[str, Any]], *,
    validate_model: Callable[[object], None],
    profile_config: Callable[[dict[str, Any]], dict[str, Any]],
    family_label: str,
) -> list[dict[str, Any]]:
    """Validate distinct exact profiles before family-specific launch checks."""

    if not isinstance(profiles, list) or not profiles:
        raise RuntimeError(f"{family_label} scope has no model profiles")
    names, configs = set(), set()
    for profile in profiles:
        if not isinstance(profile, dict) or set(profile) != {"model", "launches"}:
            raise RuntimeError(f"{family_label} scope has an invalid model profile")
        validate_model(profile["model"])
        name = profile["model"]["name"]
        if name in names:
            raise RuntimeError(f"{family_label} scope has duplicate model names")
        names.add(name)
        config_key = json.dumps(canonical(profile_config(profile)), sort_keys=True)
        if config_key in configs:
            raise RuntimeError(f"{family_label} scope has duplicate model configs")
        configs.add(config_key)
    return profiles
# @kernel-bridge-end vosti_kernels::profile_validation


def select_profile_by_name(
    profiles: Sequence[dict[str, Any]],
    name: str | None,
    *,
    profile_name: Callable[[dict[str, Any]], str],
    family_label: str,
) -> dict[str, Any]:
    if name is None:
        if len(profiles) != 1:
            raise ValueError(f"{family_label} model profile name is required")
        return detached(profiles[0])
    matches = [profile for profile in profiles if profile_name(profile) == name]
    if len(matches) != 1:
        raise ValueError(f"unknown {family_label} model profile {name!r}")
    return detached(matches[0])


def select_profile_by_config(
    profiles: Sequence[dict[str, Any]],
    config: dict[str, Any],
    *,
    profile_config: Callable[[dict[str, Any]], dict[str, Any]],
    profile_name: Callable[[dict[str, Any]], str],
    family_label: str,
) -> dict[str, Any]:
    if not isinstance(config, dict):
        raise ValueError(f"{family_label} runtime requires a config dictionary")
    actual = canonical(config)
    matches = [
        profile
        for profile in profiles
        if canonical(profile_config(profile)) == actual
    ]
    if len(matches) == 1:
        return detached(matches[0])
    names = [profile_name(profile) for profile in profiles]
    expected_configs = [canonical(profile_config(profile)) for profile in profiles]
    differing_by_profile = [
        sorted(
            field
            for field in set(expected) | set(actual)
            if expected.get(field) != actual.get(field)
        )
        for expected in expected_configs
    ]
    nearest_index = min(
        range(len(differing_by_profile)),
        key=lambda index: len(differing_by_profile[index]),
    )
    raise ValueError(
        f"{family_label} config is outside the exact model profiles; "
        f"catalog={names}; nearest={names[nearest_index]!r}; "
        f"differing fields={differing_by_profile[nearest_index]}"
    )
