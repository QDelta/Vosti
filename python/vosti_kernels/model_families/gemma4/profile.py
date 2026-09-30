"""Exact Gemma-4 profiles and model-static primitive call-site inventories."""

from pathlib import Path

from ...dense_launch_plan import launch_identity, validate_launch_inventory, validate_dense_launch
from ...model_profile import (canonical, detached, load_scope, scope_sha256 as _digest,
    select_profile_by_config, select_profile_by_name, validate_kernel_catalog)
from .loader import config_from_runtime_dict, FULL_ATTENTION, SLIDING_ATTENTION

SCOPE_SCHEMA = "vosti.model-runtime-scope.v1"
SCOPE_STATUS = "qualification_scope"
_CONTRACT_FIELDS = {
    "wrapper", "module", "entrypoint", "source", "kernel", "source_sha256", "evidence",
}


def validate_launch(launch: dict) -> None:
    wrapper = launch.get("wrapper")
    if wrapper == "scaled_embed":
        validate_dense_launch({**launch, "wrapper": "embed"})
    elif wrapper == "paged_attention_swa":
        key = launch.get("key", {})
        if set(key) != {"head_dim", "window_size"} or type(key["window_size"]) is not int or key["window_size"] <= 0:
            raise ValueError("SWA launch requires static head dimension and window")
        validate_dense_launch({**launch, "wrapper": "paged_attention", "key": {"head_dim": key["head_dim"]}})
    elif wrapper in {"residual_add", "gelu_tanh_mul", "scale", "softcap"}:
        key, config = launch.get("key"), launch.get("config")
        if not isinstance(key, dict) or set(key) != {"width"} or not isinstance(config, dict) or set(config) != {
            "BLOCK_M", "BLOCK_N", "num_warps", "num_stages"
        }:
            raise ValueError("pointwise launch has invalid key/config schema")
        if any(type(value) is not int or value <= 0 for value in (*key.values(), *config.values())):
            raise ValueError("pointwise launch requires positive integers")
    else:
        validate_dense_launch(launch)


def model_config(profile: dict) -> dict:
    model = dict(profile["model"])
    model.pop("name")
    return detached(config_from_runtime_dict(model).as_runtime_dict())


def launch_inventory(profile: dict) -> list[dict]:
    """No runtime dimensions or tile choices; repeated static sites share keys."""
    config = config_from_runtime_dict(model_config(profile))
    entries, by_key = [], {}

    def add(wrapper, site, key):
        identity = launch_identity(wrapper, key)
        if identity not in by_key:
            by_key[identity] = {"wrapper": wrapper, "sites": [], "key": dict(key)}
            entries.append(by_key[identity])
        if site not in by_key[identity]["sites"]:
            by_key[identity]["sites"].append(site)

    h, ff = config.hidden_size, config.intermediate_size
    add("scaled_embed", "token_embedding", {"width": h})
    for site in ("input_norm", "post_attention_norm", "pre_feedforward_norm",
                 "post_feedforward_norm", "final_norm"):
        add("rms_norm", site, {"width": h})
    for i, kind in enumerate(config.layer_types):
        geometry = config.attention_geometry(i)
        prefix = "local" if kind == SLIDING_ATTENTION else "global"
        add("qkv_linear", f"{prefix}.qkv_projection",
            {"q_width": geometry.query_width, "kv_width": geometry.kv_width, "k": h})
        add("linear", f"{prefix}.o_projection", {"n": h, "k": geometry.query_width})
        for role, heads in (("q_norm", geometry.query_heads), ("k_norm", geometry.kv_heads),
                            ("v_norm", geometry.kv_heads)):
            add("qk_norm", f"{prefix}.{role}", {"heads": heads, "head_dim": geometry.head_dim})
        for role in ("rotary_embed_q", "rotary_embed_k"):
            add("rotary_embed", f"{prefix}.{role}", {"width": geometry.head_dim})
        for role in ("store_k", "store_v"):
            add("store_kv_cache", f"{prefix}.{role}", {"kvd": geometry.kv_width})
        if kind == FULL_ATTENTION:
            add("paged_attention", "full_attention", {"head_dim": geometry.head_dim})
        else:
            add("paged_attention_swa", "sliding_attention",
                {"head_dim": geometry.head_dim, "window_size": config.sliding_window})
    for site, n, k in (("gate_up_projection", 2 * ff, h), ("down_projection", h, ff),
                       ("tied_lm_head", config.vocab_size, h)):
        add("linear", site, {"n": n, "k": k})
    for site in ("attention_residual_add", "feedforward_residual_add"):
        add("residual_add", site, {"width": h})
    add("gelu_tanh_mul", "mlp_activation", {"width": ff})
    add("scale", "layer_output_scale", {"width": h})
    if config.final_logit_softcapping is not None:
        add("softcap", "final_logits_softcap", {"width": config.vocab_size})
    return entries


def scope() -> dict:
    value = load_scope(Path(__file__).with_name("scope.json"))
    if (value.get("schema") != SCOPE_SCHEMA or value.get("status") != SCOPE_STATUS
            or value.get("architecture") != "gemma4_text" or value.get("engine_reachable") is not False
            or value.get("runtime") != {"dtype": "torch.bfloat16", "device_type": "cuda"}):
        raise ValueError("Gemma 4 has invalid qualification scope metadata")
    contracts = value["kernel_contracts"]
    try:
        wrappers = validate_kernel_catalog(
            contracts, fields=_CONTRACT_FIELDS, family_label="Gemma 4",
        )
    except RuntimeError as error:
        raise ValueError(str(error)) from error
    profiles = value["model_profiles"]
    if not profiles or len({p["model"]["name"] for p in profiles}) != len(profiles):
        raise ValueError("Gemma 4 scope must name distinct exact profiles")
    for profile in profiles:
        config = model_config(profile)
        if canonical(config) != canonical({k:v for k,v in profile["model"].items() if k != "name"}):
            raise ValueError("Gemma 4 scope model is not in canonical runtime form")
        validate_launch_inventory(launch_inventory(profile), profile["launches"])
        for launch in profile["launches"]:
            validate_launch(launch)
        if {entry["wrapper"] for entry in profile["launches"]} != wrappers:
            raise ValueError("Gemma 4 profile does not cover its complete kernel catalog")
    return detached(value)


def scope_sha256() -> str:
    return _digest(scope())


def model_profiles() -> list[dict]:
    return scope()["model_profiles"]


def model_profile_for_name(name: str | None = None) -> dict:
    return select_profile_by_name(model_profiles(), name,
        profile_name=lambda p: p["model"]["name"], family_label="Gemma 4")


def model_profile_for_config(config: dict) -> dict:
    return select_profile_by_config(model_profiles(), config, profile_config=model_config,
        profile_name=lambda p: p["model"]["name"], family_label="Gemma 4")
