"""Validated declarative scope for exact text-only Llama 3 profiles."""

# @kernel-bridge-begin vosti_kernels::llama3_profile_registry
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from ...dense_launch_plan import (
    QK_NORM_DISABLED,
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
    "transformers_architecture",
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
    "rope_factor",
    "rope_low_frequency_factor",
    "rope_high_frequency_factor",
    "rope_original_max_position_embeddings",
    "tie_word_embeddings",
    "hidden_act",
    "attention_bias",
    "attention_dropout",
    "mlp_bias",
    "pretraining_tp",
    "attention_kind",
    "rope_scaling_kind",
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
    "rope_original_max_position_embeddings",
}
_POSITIVE_FLOAT_FIELDS = {
    "rms_norm_eps",
    "rope_theta",
    "rope_factor",
    "rope_low_frequency_factor",
    "rope_high_frequency_factor",
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
_FIXED_COMPOSITION = {
    "model_type": "llama",
    "transformers_architecture": "LlamaForCausalLM",
    "hidden_act": "silu",
    "attention_bias": False,
    "attention_dropout": 0.0,
    "mlp_bias": False,
    "pretraining_tp": 1,
    "attention_kind": "full_attention",
    "rope_scaling_kind": "llama3",
}


def model_config(profile: dict[str, Any]) -> dict[str, Any]:
    model = dict(profile["model"])
    model.pop("name")
    return detached(model)


def runtime_model_config(profile: dict[str, Any]) -> dict[str, Any]:
    """Return the flat, sealed configuration consumed by shared primitives."""

    return model_config(profile)


def project_config(config: dict[str, Any]) -> dict[str, Any]:
    """Project a normalized loader config to all forward-changing fields."""

    if not isinstance(config, dict):
        raise ValueError("Llama 3 config must be a dictionary")
    missing = sorted(field for field in _CONFIG_FIELDS if field not in config)
    if missing:
        raise ValueError(f"Llama 3 config lacks required fields {missing}")
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
        "composition": {"qk_norm": QK_NORM_DISABLED},
    }


def launch_inventory(profile: dict[str, Any]) -> list[dict[str, Any]]:
    """Return Llama 3 call sites without selecting kernel configurations."""

    return required_model_launch_inventory(_deployment_model(profile))


def _validate_model(model: object) -> None:
    if not isinstance(model, dict) or set(model) != _MODEL_FIELDS:
        raise RuntimeError("Llama 3 profile has invalid model metadata")
    if not isinstance(model["name"], str) or not model["name"]:
        raise RuntimeError("Llama 3 profile has no model name")
    for field in _POSITIVE_INTEGER_FIELDS:
        value = model[field]
        if type(value) is not int or value <= 0:
            raise RuntimeError(f"Llama 3 profile has invalid {field}")
    for field in _POSITIVE_FLOAT_FIELDS:
        value = model[field]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or value <= 0
        ):
            raise RuntimeError(f"Llama 3 profile has invalid {field}")
    if (
        model["num_attention_heads"] % model["num_key_value_heads"]
        or model["head_dim"] % 2
    ):
        raise RuntimeError("Llama 3 profile has invalid head geometry")
    if model["rope_high_frequency_factor"] <= model["rope_low_frequency_factor"]:
        raise RuntimeError("Llama 3 profile has invalid RoPE frequency factors")
    if type(model["tie_word_embeddings"]) is not bool:
        raise RuntimeError("Llama 3 profile has invalid tie_word_embeddings")
    for field, expected in _FIXED_COMPOSITION.items():
        value = model[field]
        if value != expected or type(value) is not type(expected):
            raise RuntimeError(
                f"Llama 3 profile requires exact {field}={expected!r}"
            )


def _validate_scope() -> None:
    validate_scope_metadata(
        _SCOPE, fields=_SCOPE_FIELDS, architecture="llama3", family_label="Llama 3",
    )

    profiles = _SCOPE.get("model_profiles")
    contracts = _SCOPE.get("kernel_contracts")
    if not isinstance(profiles, list) or not profiles:
        raise RuntimeError("Llama 3 scope has no model profiles")
    wrappers = validate_kernel_catalog(
        contracts, fields=_CONTRACT_FIELDS, family_label="Llama 3",
    )
    if "qk_norm" in wrappers:
        raise RuntimeError("Llama 3 scope must not import Q/K normalization")

    for profile in validate_profile_registry(
        profiles, validate_model=_validate_model,
        profile_config=model_config, family_label="Llama 3",
    ):
        launches = profile["launches"]
        if not isinstance(launches, list) or not launches:
            raise RuntimeError("Llama 3 profile has no launches")
        if any(launch.get("wrapper") == "qk_norm" for launch in launches):
            raise RuntimeError("Llama 3 profile must not launch Q/K normalization")
        for launch in launches:
            if not isinstance(launch, dict) or set(launch) != {
                "wrapper",
                "sites",
                "key",
                "config",
            }:
                raise RuntimeError("Llama 3 profile has an invalid launch")
            if launch["wrapper"] not in wrappers:
                raise RuntimeError("Llama 3 profile references an unknown wrapper")
            try:
                validate_dense_launch(launch)
            except ValueError as error:
                raise RuntimeError(
                    f"Llama 3 profile has an invalid launch: {error}"
                ) from error
        try:
            validate_model_launch_inventory(_deployment_model(profile), launches)
        except ValueError as error:
            raise RuntimeError(
                f"Llama 3 profile launch inventory is invalid: {error}"
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
        family_label="Llama 3",
    )


def model_profile_for_config(config: dict[str, Any]) -> dict[str, Any]:
    return select_profile_by_config(
        _SCOPE["model_profiles"],
        project_config(config),
        profile_config=model_config,
        profile_name=lambda profile: profile["model"]["name"],
        family_label="Llama 3",
    )


_validate_scope()

# @kernel-bridge-end vosti_kernels::llama3_profile_registry

__all__ = [
    "model_config",
    "launch_inventory",
    "model_profile_for_config",
    "model_profile_for_name",
    "model_profiles",
    "model_shape",
    "project_config",
    "runtime_model_config",
    "scope",
    "scope_sha256",
]
