"""Static launch policy for models composed from the shared dense primitives.

The policy is architecture-neutral: callers supply decoder geometry, any
additional model-level matrix multiplications, and the pinned kernel-config
selectors.  No request or batch dimension is an input.
"""

from __future__ import annotations

from typing import Callable


KernelSelector = Callable[..., dict]

QK_NORM_DISABLED = "disabled"
QK_NORM_RMS = "rms_norm"
QK_NORM_KINDS = frozenset({QK_NORM_DISABLED, QK_NORM_RMS})

DENSE_CONFIG_FIELDS_BY_WRAPPER = {
    "rms_norm": {"BLOCK_M", "BLOCK_N", "num_warps", "num_stages"},
    "add_rms_norm": {"BLOCK_M", "BLOCK_N", "num_warps", "num_stages"},
    "silu_and_mul": {"BLOCK_M", "BLOCK_N", "num_warps", "num_stages"},
    "embed": {"BLOCK_M", "BLOCK_D", "D", "num_warps", "num_stages"},
    "qk_norm": {"BLOCK_M", "H", "D", "num_warps", "num_stages"},
    "store_kv_cache": {"BLOCK_M", "KVD", "num_warps", "num_stages"},
    "rotary_embed": {"BLOCK_M", "D", "HD", "num_warps", "num_stages"},
    "linear": {
        "BLOCK_M",
        "BLOCK_N",
        "BLOCK_K",
        "num_warps",
        "num_stages",
    },
    "qkv_linear": {
        "BLOCK_M",
        "BLOCK_N",
        "BLOCK_K",
        "num_warps",
        "num_stages",
    },
    "paged_attention": {
        "BLOCK_M",
        "BLOCK_N",
        "D_HEAD",
        "num_warps",
        "num_stages",
    },
}
DENSE_KEY_FIELDS_BY_WRAPPER = {
    "rms_norm": {"width"},
    "add_rms_norm": {"width"},
    "silu_and_mul": {"width"},
    "embed": {"width"},
    "qk_norm": {"heads", "head_dim"},
    "store_kv_cache": {"kvd"},
    "rotary_embed": {"width"},
    "linear": {"n", "k"},
    "qkv_linear": {"q_width", "kv_width", "k"},
    "paged_attention": {"head_dim"},
}


def launch_identity(
    wrapper: str, key: dict
) -> tuple[str, tuple[tuple[str, int], ...]]:
    return wrapper, tuple(sorted((name, int(value)) for name, value in key.items()))


def _matmul_sites(shape: dict) -> tuple[tuple[str, int, int], ...]:
    q_width = shape["num_heads"] * shape["head_dim"]
    return (
        ("hidden_to_hidden", shape["hidden"], shape["hidden"]),
        ("o_projection", shape["hidden"], q_width),
        (
            "gate_up_projection",
            2 * shape["intermediate_half"],
            shape["hidden"],
        ),
        ("down_projection", shape["hidden"], shape["intermediate_half"]),
    )


def _qk_norm_kind(composition: object) -> str:
    if not isinstance(composition, dict):
        raise ValueError("dense model has no closed composition")
    kind = composition.get("qk_norm")
    if not isinstance(kind, str) or kind not in QK_NORM_KINDS:
        raise ValueError(f"dense model has unsupported qk_norm choice {kind!r}")
    return kind


def derive_static_launch_plan(
    shape: dict,
    additional_matmul_nk: tuple[tuple[int, int], ...],
    *,
    composition: dict,
    selectors: dict[str, KernelSelector],
) -> list[dict]:
    """Derive one complete static plan through kernel-owned selectors."""

    kvd = shape["num_kv_heads"] * shape["head_dim"]
    qk_norm_kind = _qk_norm_kind(composition)
    launches = [
        {
            "wrapper": "rms_norm",
            "sites": ["rms_norm"],
            "key": {"width": shape["hidden"]},
        },
        {
            "wrapper": "add_rms_norm",
            "sites": ["add_rms_norm"],
            "key": {"width": shape["hidden"]},
        },
        {
            "wrapper": "silu_and_mul",
            "sites": ["silu_and_mul"],
            "key": {"width": shape["intermediate_half"]},
        },
        {
            "wrapper": "embed",
            "sites": ["embed"],
            "key": {"width": shape["hidden"]},
        },
        {
            "wrapper": "store_kv_cache",
            "sites": ["store_k", "store_v"],
            "key": {"kvd": kvd},
        },
        {
            "wrapper": "rotary_embed",
            "sites": ["rotary_embed_q", "rotary_embed_k"],
            "key": {"width": shape["head_dim"]},
        },
    ]
    if qk_norm_kind == QK_NORM_RMS:
        launches[4:4] = [
            {
                "wrapper": "qk_norm",
                "sites": ["q_norm"],
                "key": {
                    "heads": shape["num_heads"],
                    "head_dim": shape["head_dim"],
                },
            },
            {
                "wrapper": "qk_norm",
                "sites": ["k_norm"],
                "key": {
                    "heads": shape["num_kv_heads"],
                    "head_dim": shape["head_dim"],
                },
            },
        ]
    launches.append(
        {
            "wrapper": "qkv_linear",
            "sites": ["qkv_projection"],
            "key": {
                "q_width": shape["num_heads"] * shape["head_dim"],
                "kv_width": shape["num_kv_heads"] * shape["head_dim"],
                "k": shape["hidden"],
            },
        }
    )
    matmul_by_nk: dict[tuple[int, int], dict] = {}
    sites = list(_matmul_sites(shape))
    sites.extend(
        (f"model_matmul_{index}", n, k)
        for index, (n, k) in enumerate(additional_matmul_nk)
    )
    for site, n, k in sites:
        if (n, k) in matmul_by_nk:
            matmul_by_nk[(n, k)]["sites"].append(site)
            continue
        launch = {
            "wrapper": "linear",
            "sites": [site],
            "key": {"n": n, "k": k},
        }
        matmul_by_nk[(n, k)] = launch
        launches.append(launch)
    launches.append(
        {
            "wrapper": "paged_attention",
            "sites": ["paged_attention"],
            "key": {"head_dim": shape["head_dim"]},
        }
    )
    return materialize_static_launch_plan(launches, selectors)


def materialize_static_launch_plan(
    inventory: list[dict], selectors: dict[str, KernelSelector]
) -> list[dict]:
    """Apply each kernel's colocated selector exactly once before serving."""

    launches = []
    for entry in inventory:
        wrapper = entry["wrapper"]
        selector = selectors.get(wrapper)
        if selector is None:
            raise ValueError(f"no config selector for kernel wrapper {wrapper!r}")
        config = selector(**entry["key"])
        if not isinstance(config, dict) or not config:
            raise ValueError(f"selector for {wrapper!r} returned no config")
        launches.append({**entry, "config": dict(config)})
    return launches


def required_model_launch_inventory(model: dict) -> list[dict]:
    """Derive the exact shared-primitive call-site inventory from model data."""

    if not isinstance(model, dict) or not isinstance(model.get("geometry"), dict):
        raise ValueError("deployment model has no launch geometry")
    geometry = model["geometry"]
    dimension_names = (
        "hidden",
        "intermediate_half",
        "num_heads",
        "num_kv_heads",
        "head_dim",
    )
    dimensions = {}
    for name in dimension_names:
        value = geometry.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"deployment model dimension {name!r} is invalid")
        dimensions[name] = value
    vocab_size = model.get("vocab_size")
    if (
        isinstance(vocab_size, bool)
        or not isinstance(vocab_size, int)
        or vocab_size <= 0
    ):
        raise ValueError("deployment model vocabulary size is invalid")

    hidden = dimensions["hidden"]
    intermediate = dimensions["intermediate_half"]
    num_heads = dimensions["num_heads"]
    num_kv_heads = dimensions["num_kv_heads"]
    head_dim = dimensions["head_dim"]
    qk_norm_kind = _qk_norm_kind(model.get("composition"))
    q_width = num_heads * head_dim
    kv_width = num_kv_heads * head_dim

    inventory = []
    by_identity = {}

    def add(wrapper: str, site: str, key: dict) -> None:
        identity = launch_identity(wrapper, key)
        previous = by_identity.get(identity)
        if previous is None:
            previous = {"wrapper": wrapper, "sites": [], "key": dict(key)}
            by_identity[identity] = previous
            inventory.append(previous)
        if site in previous["sites"]:
            raise ValueError(f"duplicate model launch site {site!r}")
        previous["sites"].append(site)

    add("rms_norm", "rms_norm", {"width": hidden})
    add("add_rms_norm", "add_rms_norm", {"width": hidden})
    add("silu_and_mul", "silu_and_mul", {"width": intermediate})
    add("embed", "embed", {"width": hidden})
    if qk_norm_kind == QK_NORM_RMS:
        add("qk_norm", "q_norm", {"heads": num_heads, "head_dim": head_dim})
        add("qk_norm", "k_norm", {"heads": num_kv_heads, "head_dim": head_dim})
    add("store_kv_cache", "store_k", {"kvd": kv_width})
    add("store_kv_cache", "store_v", {"kvd": kv_width})
    add("rotary_embed", "rotary_embed_q", {"width": head_dim})
    add("rotary_embed", "rotary_embed_k", {"width": head_dim})
    add(
        "qkv_linear",
        "qkv_projection",
        {"q_width": q_width, "kv_width": kv_width, "k": hidden},
    )
    for site, n, k in (
        ("hidden_to_hidden", hidden, hidden),
        ("o_projection", hidden, q_width),
        ("gate_up_projection", 2 * intermediate, hidden),
        ("down_projection", hidden, intermediate),
        ("model_matmul_0", vocab_size, hidden),
    ):
        add("linear", site, {"n": n, "k": k})
    add("paged_attention", "paged_attention", {"head_dim": head_dim})
    return inventory


def validate_launch_inventory(
    expected_entries: list[dict], launches: object
) -> None:
    """Reject missing, extra, duplicate, or misattributed launch sites."""

    if not isinstance(launches, list):
        raise ValueError("deployment candidate has no launch plan")
    expected = {
        launch_identity(entry["wrapper"], entry["key"]): entry
        for entry in expected_entries
    }
    actual = {}
    for launch in launches:
        if not isinstance(launch, dict):
            raise ValueError("deployment launch is malformed")
        wrapper = launch.get("wrapper")
        key = launch.get("key")
        sites = launch.get("sites")
        if not isinstance(wrapper, str) or not isinstance(key, dict):
            raise ValueError("deployment launch has no static identity")
        identity = launch_identity(wrapper, key)
        if identity in actual:
            raise ValueError(f"deployment launch plan duplicates {wrapper} key {key}")
        if (
            not isinstance(sites, list)
            or len(sites) != len(set(sites))
            or any(not isinstance(site, str) or not site for site in sites)
        ):
            raise ValueError("deployment launch has invalid call-site coverage")
        actual[identity] = launch

    missing = sorted(set(expected) - set(actual))
    extra = sorted(set(actual) - set(expected))
    if missing or extra:
        raise ValueError(
            "deployment launch plan differs from the model-derived inventory: "
            f"missing={missing}, extra={extra}"
        )
    for identity, expected_entry in expected.items():
        launch = actual[identity]
        if launch["sites"] != expected_entry["sites"]:
            raise ValueError(
                "deployment launch call sites differ from the model-derived "
                f"inventory for {launch['wrapper']} {launch['key']}: "
                f"expected={expected_entry['sites']}, got={launch['sites']}"
            )


def validate_model_launch_inventory(model: dict, launches: object) -> None:
    """Validate the call-site inventory for a shared dense model."""

    validate_launch_inventory(required_model_launch_inventory(model), launches)


def validate_dense_launch(launch: dict) -> None:
    """Validate one launch for the shared dense primitive catalog."""

    wrapper = launch.get("wrapper")
    if wrapper not in DENSE_CONFIG_FIELDS_BY_WRAPPER:
        raise ValueError(f"unknown shared dense wrapper {wrapper!r}")
    key = launch.get("key")
    config = launch.get("config")
    if not isinstance(key, dict) or set(key) != DENSE_KEY_FIELDS_BY_WRAPPER[wrapper]:
        raise ValueError("dense launch has an invalid static key")
    if (
        not isinstance(config, dict)
        or set(config) != DENSE_CONFIG_FIELDS_BY_WRAPPER[wrapper]
    ):
        raise ValueError("dense launch has an incomplete static config")
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value <= 0
        for value in (*key.values(), *config.values())
    ):
        raise ValueError("dense launch key and config values must be positive integers")
    if wrapper == "embed" and key["width"] != config["D"]:
        raise ValueError("embedding launch key and config disagree")
    if wrapper in {"rms_norm", "add_rms_norm"} and (
        config["BLOCK_N"] < key["width"]
    ):
        raise ValueError("reduction block width does not cover its key")
    if wrapper == "qk_norm" and (
        key["heads"] != config["H"] or key["head_dim"] != config["D"]
    ):
        raise ValueError("qk-norm launch key and config disagree")
    if wrapper == "store_kv_cache" and key["kvd"] != config["KVD"]:
        raise ValueError("KV-store launch key and config disagree")
    if wrapper == "rotary_embed" and (
        key["width"] != config["D"] or config["HD"] * 2 != config["D"]
    ):
        raise ValueError("rotary launch key and config disagree")
    if wrapper == "paged_attention" and key["head_dim"] != config["D_HEAD"]:
        raise ValueError("paged-attention launch key and config disagree")


__all__ = [
    "DENSE_CONFIG_FIELDS_BY_WRAPPER",
    "DENSE_KEY_FIELDS_BY_WRAPPER",
    "QK_NORM_DISABLED",
    "QK_NORM_KINDS",
    "QK_NORM_RMS",
    "derive_static_launch_plan",
    "launch_identity",
    "materialize_static_launch_plan",
    "required_model_launch_inventory",
    "validate_dense_launch",
    "validate_launch_inventory",
    "validate_model_launch_inventory",
]
