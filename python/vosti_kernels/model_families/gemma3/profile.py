"""Validated declarative scope for qualified text-only Gemma 3 profiles.

Kernel contracts are shared by the architecture.  Each model profile binds one
exact loader configuration to its own static launch inventory.  Selection is
by the complete resolved configuration, never by a size label or a partial
geometry match.
"""

# @kernel-bridge-begin vosti_kernels::gemma3_profile_registry
from __future__ import annotations

from pathlib import Path
from typing import Any

from ...dense_launch_plan import validate_launch_inventory
from ...model_profile import (
    validate_kernel_catalog,
    is_sha256 as _is_sha256,
    validate_scope_metadata,
    validate_profile_registry,
    detached as _detached,
    load_scope,
    scope_sha256 as _scope_sha256,
    select_profile_by_config,
    select_profile_by_name,
)


SCOPE_SCHEMA = "vosti.model-runtime-scope.v1"
SCOPE_STATUS = "qualification_scope"

_SCOPE_PATH = Path(__file__).with_name("scope.json")
_SCOPE = load_scope(_SCOPE_PATH)

_MODEL_FIELDS = {
    "name",
    "vocab_size",
    "hidden_size",
    "intermediate_size",
    "num_hidden_layers",
    "num_attention_heads",
    "num_key_value_heads",
    "head_dim",
    "max_position_embeddings",
    "rms_norm_eps",
    "query_pre_attn_scalar",
    "sliding_window",
    "local_rope_theta",
    "global_rope_theta",
    "global_rope_factor",
    "attention_pattern",
}
_SCOPE_FIELDS = {
    "schema",
    "status",
    "engine_reachable",
    "architecture",
    "runtime",
    "deferred_obligations",
    "trusted_framework_operations",
    "runtime_support_sources",
    "kernel_contracts",
    "model_profiles",
}
_CONTRACT_FIELDS = {
    "wrapper",
    "module",
    "entrypoint",
    "source",
    "kernel",
    "source_sha256",
    "evidence",
}
_POSITIVE_INTEGER_MODEL_FIELDS = {
    "vocab_size",
    "hidden_size",
    "intermediate_size",
    "num_hidden_layers",
    "num_attention_heads",
    "num_key_value_heads",
    "head_dim",
    "max_position_embeddings",
    "sliding_window",
}
_POSITIVE_NUMBER_MODEL_FIELDS = {
    "rms_norm_eps",
    "query_pre_attn_scalar",
    "local_rope_theta",
    "global_rope_theta",
    "global_rope_factor",
}


def _attention_schedule(model: dict[str, Any]) -> list[str]:
    pattern = model["attention_pattern"]
    period = int(pattern["period"])
    offset = int(pattern["full_attention_offset"])
    return [
        "full_attention" if index % period == offset else "sliding_attention"
        for index in range(int(model["num_hidden_layers"]))
    ]


def model_config(profile: dict[str, Any]) -> dict[str, Any]:
    """Project one validated profile to the checkpoint-loader config."""

    model = dict(profile["model"])
    model.pop("name")
    model.pop("attention_pattern")
    model["layer_types"] = _attention_schedule(profile["model"])
    return _detached(model)


def launch_inventory(profile: dict[str, Any]) -> list[dict[str, Any]]:
    """Return Gemma 3 call sites without selecting kernel configurations."""

    model = profile["model"]
    hidden = int(model["hidden_size"])
    intermediate = int(model["intermediate_size"])
    head_dim = int(model["head_dim"])
    q_width = int(model["num_attention_heads"]) * head_dim
    kv_width = int(model["num_key_value_heads"]) * head_dim
    return [
        {"wrapper": "gemma3_scaled_embed", "sites": ["token_embedding"],
         "key": {"width": hidden}},
        {"wrapper": "gemma3_rms_norm", "sites": [
            "input_norm", "post_attention_norm", "pre_feedforward_norm",
            "post_feedforward_norm", "final_norm",
        ], "key": {"width": hidden}},
        {"wrapper": "qkv_linear", "sites": ["qkv_projection"],
         "key": {"q_width": q_width, "kv_width": kv_width, "k": hidden}},
        {"wrapper": "linear", "sites": ["o_projection"],
         "key": {"n": hidden, "k": q_width}},
        {"wrapper": "linear", "sites": ["gate_up_projection"],
         "key": {"n": 2 * intermediate, "k": hidden}},
        {"wrapper": "linear", "sites": ["down_projection"],
         "key": {"n": hidden, "k": intermediate}},
        {"wrapper": "linear", "sites": ["tied_lm_head"],
         "key": {"n": int(model["vocab_size"]), "k": hidden}},
        {"wrapper": "gemma3_qk_norm", "sites": ["q_norm"], "key": {
            "heads": int(model["num_attention_heads"]), "head_dim": head_dim,
        }},
        {"wrapper": "gemma3_qk_norm", "sites": ["k_norm"], "key": {
            "heads": int(model["num_key_value_heads"]), "head_dim": head_dim,
        }},
        {"wrapper": "rotary_embed",
         "sites": ["rotary_embed_q", "rotary_embed_k"],
         "key": {"width": head_dim}},
        {"wrapper": "store_kv_cache", "sites": ["store_k", "store_v"],
         "key": {"kvd": kv_width}},
        {"wrapper": "paged_attention", "sites": ["full_attention"],
         "key": {"head_dim": head_dim}},
        {"wrapper": "paged_attention_swa",
         "sites": ["sliding_attention"], "key": {
             "head_dim": head_dim, "window_size": int(model["sliding_window"]),
         }},
        {"wrapper": "gemma3_add",
         "sites": ["attention_residual_add", "feedforward_residual_add"],
         "key": {"width": hidden}},
        {"wrapper": "gemma3_gelu_tanh_mul", "sites": ["mlp_activation"],
         "key": {"width": intermediate}},
    ]


def _validate_model(model: object) -> None:
    if not isinstance(model, dict) or set(model) != _MODEL_FIELDS:
        raise RuntimeError("Gemma 3 profile has invalid model metadata")
    if not isinstance(model["name"], str) or not model["name"]:
        raise RuntimeError("Gemma 3 profile has no model name")
    for field in _POSITIVE_INTEGER_MODEL_FIELDS:
        value = model[field]
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise RuntimeError(
                f"Gemma 3 profile has invalid {field}"
            )
    for field in _POSITIVE_NUMBER_MODEL_FIELDS:
        value = model[field]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or value <= 0
        ):
            raise RuntimeError(f"Gemma 3 profile has invalid {field}")
    if (
        model["num_attention_heads"] % model["num_key_value_heads"] != 0
        or model["head_dim"] % 2 != 0
    ):
        raise RuntimeError("Gemma 3 profile has invalid head geometry")
    pattern = model["attention_pattern"]
    if not isinstance(pattern, dict) or set(pattern) != {
        "period",
        "full_attention_offset",
    }:
        raise RuntimeError("Gemma 3 profile has invalid attention pattern")
    period = pattern["period"]
    offset = pattern["full_attention_offset"]
    if (
        isinstance(period, bool)
        or not isinstance(period, int)
        or period <= 0
        or isinstance(offset, bool)
        or not isinstance(offset, int)
        or offset < 0
        or offset >= period
    ):
        raise RuntimeError("Gemma 3 profile has invalid attention schedule")


def _validate_scope() -> None:
    validate_scope_metadata(
        _SCOPE, fields=_SCOPE_FIELDS, architecture="gemma3_text", family_label="Gemma 3",
    )

    contracts = _SCOPE.get("kernel_contracts")
    profiles = _SCOPE.get("model_profiles")
    wrappers = validate_kernel_catalog(
        contracts, fields=_CONTRACT_FIELDS, family_label="Gemma 3",
    )
    if any(
        not isinstance(value, str) or not value
        for contract in contracts
        for value in contract.values()
    ):
        raise RuntimeError("Gemma 3 scope has invalid contract metadata")
    support_sources = _SCOPE.get("runtime_support_sources")
    if not isinstance(support_sources, list) or any(
        not isinstance(source, dict) or set(source) != {"source", "source_sha256"}
        for source in support_sources
    ):
        raise RuntimeError("Gemma 3 scope has invalid support sources")
    if any(
        not isinstance(source["source"], str) or not source["source"]
        for source in support_sources
    ):
        raise RuntimeError("Gemma 3 scope has invalid support source names")
    for source in support_sources:
        if not _is_sha256(source.get("source_sha256")):
            raise RuntimeError("Gemma 3 scope has an invalid source digest")
    for field in ("deferred_obligations", "trusted_framework_operations"):
        values = _SCOPE.get(field)
        if not isinstance(values, list) or any(
            not isinstance(value, str) or not value for value in values
        ):
            raise RuntimeError(f"Gemma 3 scope has invalid {field}")

    known_wrappers = set(wrappers)
    for profile in validate_profile_registry(
        profiles, validate_model=_validate_model,
        profile_config=model_config, family_label="Gemma 3",
    ):

        launches = profile["launches"]
        if not isinstance(launches, list) or not launches:
            raise RuntimeError("Gemma 3 profile has no launches")
        sites: set[str] = set()
        identities: set[tuple[str, tuple[tuple[str, int], ...]]] = set()
        for launch in launches:
            if not isinstance(launch, dict) or set(launch) != {
                "wrapper",
                "sites",
                "key",
                "config",
            }:
                raise RuntimeError("Gemma 3 launch differs from its schema")
            wrapper = launch.get("wrapper")
            key = launch.get("key")
            launch_sites = launch.get("sites")
            if wrapper not in known_wrappers or not isinstance(key, dict):
                raise RuntimeError("Gemma 3 launch has an unknown identity")
            if not isinstance(launch_sites, list) or not launch_sites:
                raise RuntimeError("Gemma 3 launch has no call sites")
            if not key or any(
                not isinstance(field, str)
                or not field
                or isinstance(value, bool)
                or not isinstance(value, int)
                or value <= 0
                for field, value in key.items()
            ):
                raise RuntimeError("Gemma 3 launch has an invalid key")
            identity = (wrapper, tuple(sorted(key.items())))
            if identity in identities:
                raise RuntimeError("Gemma 3 launch identity is duplicated")
            identities.add(identity)
            config = launch.get("config")
            if not isinstance(config, dict) or any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or value <= 0
                for value in config.values()
            ):
                raise RuntimeError("Gemma 3 launch has invalid config")
            for site in launch_sites:
                if not isinstance(site, str) or not site or site in sites:
                    raise RuntimeError("Gemma 3 launch site is duplicated")
                sites.add(site)
        if {launch["wrapper"] for launch in launches} != known_wrappers:
            raise RuntimeError("Gemma 3 profile does not cover every wrapper")
        try:
            validate_launch_inventory(launch_inventory(profile), launches)
        except ValueError as error:
            raise RuntimeError(
                f"Gemma 3 profile launch inventory is invalid: {error}"
            ) from error


_validate_scope()


def scope() -> dict[str, Any]:
    return _detached(_SCOPE)


def scope_sha256() -> str:
    return _scope_sha256(_SCOPE)


def model_profiles() -> list[dict[str, Any]]:
    return _detached(_SCOPE["model_profiles"])


def model_profile_for_name(name: str | None = None) -> dict[str, Any]:
    return select_profile_by_name(
        _SCOPE["model_profiles"],
        name,
        profile_name=lambda profile: profile["model"]["name"],
        family_label="Gemma 3",
    )


def model_profile_for_config(config: dict[str, Any]) -> dict[str, Any]:
    return select_profile_by_config(
        _SCOPE["model_profiles"],
        config,
        profile_config=model_config,
        profile_name=lambda profile: profile["model"]["name"],
        family_label="Gemma 3",
    )


# @kernel-bridge-end vosti_kernels::gemma3_profile_registry


__all__ = [
    "SCOPE_SCHEMA",
    "SCOPE_STATUS",
    "model_config",
    "launch_inventory",
    "model_profile_for_config",
    "model_profile_for_name",
    "model_profiles",
    "scope",
    "scope_sha256",
]
