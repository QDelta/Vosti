"""Validated declarative scope for exact text-only Qwen3 profiles."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ...dense_launch_plan import (
    QK_NORM_RMS,
    required_model_launch_inventory,
    validate_dense_launch,
    validate_model_launch_inventory,
)
from ...model_profile import (
    validate_kernel_catalog,
    validate_scope_metadata,
    validate_profile_registry,
    detached,
    load_scope,
    scope_sha256 as _scope_sha256,
    select_profile_by_config,
    select_profile_by_name,
)


SCOPE_SCHEMA = "vosti.model-runtime-scope.v1"
SCOPE_STATUS = "qualification_scope"

_SCOPE_PATH = Path(__file__).with_name("scope.json")
_SCOPE = load_scope(_SCOPE_PATH)
_SCOPE_FIELDS = {
    "schema",
    "status",
    "engine_reachable",
    "architecture",
    "model_profiles",
    "runtime",
    "kernel_contracts",
    "runtime_support_sources",
}
_MODEL_FIELDS = {
    "name",
    "model_type",
    "vocab_size",
    "hidden_size",
    "intermediate_size",
    "num_hidden_layers",
    "num_attention_heads",
    "num_key_value_heads",
    "head_dim",
    "max_position_embeddings",
    "rms_norm_eps",
    "rope_theta",
    "tie_word_embeddings",
    "attention_bias",
    "attention_dropout",
    "hidden_act",
    "use_sliding_window",
    "sliding_window",
    "rope_scaling",
}
_POSITIVE_INTEGER_FIELDS = {
    "vocab_size",
    "hidden_size",
    "intermediate_size",
    "num_hidden_layers",
    "num_attention_heads",
    "num_key_value_heads",
    "head_dim",
    "max_position_embeddings",
}
_CONFIG_FIELDS = _MODEL_FIELDS - {"name"}
_CONTRACT_FIELDS = {
    "wrapper",
    "module",
    "entrypoint",
    "source",
    "kernel",
    "source_sha256",
    "outputs",
    "evidence",
    "bridge",
}


def model_config(profile: dict[str, Any]) -> dict[str, Any]:
    model = dict(profile["model"])
    model.pop("name")
    return detached(model)


def project_config(config: dict[str, Any]) -> dict[str, Any]:
    """Project a Hugging Face config to the exact supported semantic fields."""

    if not isinstance(config, dict):
        raise ValueError("Qwen3 config must be a dictionary")
    missing = sorted(field for field in _CONFIG_FIELDS if field not in config)
    if missing:
        raise ValueError(f"Qwen3 config lacks required fields {missing}")
    return detached({field: config[field] for field in sorted(_CONFIG_FIELDS)})


def model_shape(profile: dict[str, Any]) -> dict[str, int | str]:
    model = profile["model"]
    return {
        "name": model["name"],
        "hidden": model["hidden_size"],
        "intermediate_half": model["intermediate_size"],
        "head_dim": model["head_dim"],
        "num_heads": model["num_attention_heads"],
        "num_kv_heads": model["num_key_value_heads"],
    }


def _deployment_model(profile: dict[str, Any]) -> dict[str, Any]:
    return {
        "vocab_size": profile["model"]["vocab_size"],
        "geometry": model_shape(profile),
        "composition": {"qk_norm": QK_NORM_RMS},
    }


def launch_inventory(profile: dict[str, Any]) -> list[dict[str, Any]]:
    """Return Qwen3 call sites without selecting kernel configurations."""

    return required_model_launch_inventory(_deployment_model(profile))


def _validate_model(model: object) -> None:
    if not isinstance(model, dict) or set(model) != _MODEL_FIELDS:
        raise RuntimeError("Qwen3 profile has invalid model metadata")
    if not isinstance(model["name"], str) or not model["name"]:
        raise RuntimeError("Qwen3 profile has no model name")
    if model["model_type"] != "qwen3":
        raise RuntimeError("Qwen3 profile names another model type")
    for field in _POSITIVE_INTEGER_FIELDS:
        value = model[field]
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise RuntimeError(f"Qwen3 profile has invalid {field}")
    for field in ("rms_norm_eps", "rope_theta"):
        value = model[field]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or value <= 0
        ):
            raise RuntimeError(f"Qwen3 profile has invalid {field}")
    if (
        model["num_attention_heads"] % model["num_key_value_heads"] != 0
        or model["head_dim"] % 2 != 0
    ):
        raise RuntimeError("Qwen3 profile has invalid head geometry")
    if model["hidden_act"] != "silu":
        raise RuntimeError("Qwen3 profile requires SiLU")
    if model["attention_bias"] is not False or model["attention_dropout"] != 0.0:
        raise RuntimeError("Qwen3 profile requires bias-free deterministic attention")
    if (
        model["use_sliding_window"] is not False
        or model["sliding_window"] is not None
        or model["rope_scaling"] is not None
    ):
        raise RuntimeError("Qwen3 profile requires full attention and default RoPE")
    if not isinstance(model["tie_word_embeddings"], bool):
        raise RuntimeError("Qwen3 profile has invalid embedding tying metadata")


def _validate_scope() -> None:
    validate_scope_metadata(
        _SCOPE, fields=_SCOPE_FIELDS, architecture="qwen3", family_label="Qwen3",
    )

    profiles = _SCOPE.get("model_profiles")
    contracts = _SCOPE.get("kernel_contracts")
    if not isinstance(profiles, list) or not profiles:
        raise RuntimeError("Qwen3 scope has no model profiles")
    wrappers = validate_kernel_catalog(
        contracts, fields=_CONTRACT_FIELDS, family_label="Qwen3",
    )

    for profile in validate_profile_registry(
        profiles, validate_model=_validate_model,
        profile_config=model_config, family_label="Qwen3",
    ):
        launches = profile["launches"]
        if not isinstance(launches, list) or not launches:
            raise RuntimeError("Qwen3 profile has no launches")
        for launch in launches:
            if not isinstance(launch, dict) or set(launch) != {
                "wrapper",
                "sites",
                "key",
                "config",
            }:
                raise RuntimeError("Qwen3 profile has an invalid launch")
            if launch["wrapper"] not in wrappers:
                raise RuntimeError("Qwen3 profile references an unknown wrapper")
            try:
                validate_dense_launch(launch)
            except ValueError as error:
                raise RuntimeError(
                    f"Qwen3 profile has an invalid launch: {error}"
                ) from error
        try:
            validate_model_launch_inventory(_deployment_model(profile), launches)
        except ValueError as error:
            raise RuntimeError(
                f"Qwen3 profile launch inventory is invalid: {error}"
            ) from error


def scope() -> dict[str, Any]:
    return detached(_SCOPE)


def scope_sha256() -> str:
    return _scope_sha256(_SCOPE)


def model_profiles() -> list[dict[str, Any]]:
    return detached(_SCOPE["model_profiles"])


def model_profile_for_name(name: str | None = None) -> dict[str, Any]:
    return select_profile_by_name(
        _SCOPE["model_profiles"],
        name,
        profile_name=lambda profile: profile["model"]["name"],
        family_label="Qwen3",
    )


def model_profile_for_config(config: dict[str, Any]) -> dict[str, Any]:
    return select_profile_by_config(
        _SCOPE["model_profiles"],
        project_config(config),
        profile_config=model_config,
        profile_name=lambda profile: profile["model"]["name"],
        family_label="Qwen3",
    )


_validate_scope()


__all__ = [
    "model_config",
    "launch_inventory",
    "model_profile_for_config",
    "model_profile_for_name",
    "model_profiles",
    "model_shape",
    "project_config",
    "scope",
    "scope_sha256",
]
