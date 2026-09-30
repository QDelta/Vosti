"""Text-only Gemma 3 checkpoint inspection and weight loading.

This module deliberately does not configure or qualify the serving runtime.
It resolves the Transformers text config, rejects unsupported model
conventions, ignores multimodal weights by construction, and returns the exact
ordered tensor role schema consumed by the Rust checkpoint boundary.
"""

# @kernel-bridge-begin vosti_kernels::gemma3_checkpoint_loader
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from pathlib import Path
from typing import Any

import torch
from transformers import AutoConfig

from ...attention_geometry import AttentionGeometry
from ... import checkpoint
from .physical import GEMMA3_LAYER_WEIGHT_ROLES


SLIDING_ATTENTION = "sliding_attention"
FULL_ATTENTION = "full_attention"

@dataclass(frozen=True)
class Gemma3TextConfig:
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    max_position_embeddings: int
    rms_norm_eps: float
    query_pre_attn_scalar: float
    sliding_window: int
    local_rope_theta: float
    global_rope_theta: float
    global_rope_factor: float
    layer_types: tuple[str, ...]

    def as_runtime_dict(self) -> dict[str, Any]:
        return asdict(self)

    def attention_geometry(self, layer: int) -> AttentionGeometry:
        if type(layer) is not int or not 0 <= layer < self.num_hidden_layers:
            raise ValueError("attention layer index is outside model geometry")
        return AttentionGeometry(self.num_attention_heads, self.num_key_value_heads, self.head_dim)


def _value(config: Any, name: str, default: Any = None) -> Any:
    if isinstance(config, dict):
        return config.get(name, default)
    return getattr(config, name, default)


def _require_positive_int(config: Any, name: str) -> int:
    value = _value(config, name)
    if type(value) is not int or value <= 0:
        raise ValueError(f"Gemma 3 text config requires positive {name}")
    return value


def _rope_entry(parameters: dict[str, Any], kind: str) -> dict[str, Any]:
    entry = parameters.get(kind)
    if not isinstance(entry, dict):
        raise ValueError(f"Gemma 3 text config requires {kind} RoPE parameters")
    return entry


def parse_text_config(config: Any) -> Gemma3TextConfig:
    """Resolve and validate the dense text submodel of a Gemma 3 config."""

    text = _value(config, "text_config") or config
    if _value(text, "model_type") != "gemma3_text":
        raise ValueError("checkpoint is not a Gemma 3 text model")

    vocab_size = _require_positive_int(text, "vocab_size")
    hidden_size = _require_positive_int(text, "hidden_size")
    intermediate_size = _require_positive_int(text, "intermediate_size")
    num_hidden_layers = _require_positive_int(text, "num_hidden_layers")
    num_attention_heads = _require_positive_int(text, "num_attention_heads")
    num_key_value_heads = _require_positive_int(text, "num_key_value_heads")
    head_dim = _require_positive_int(text, "head_dim")
    max_position_embeddings = _require_positive_int(
        text, "max_position_embeddings"
    )
    sliding_window = _require_positive_int(text, "sliding_window")

    if num_attention_heads % num_key_value_heads != 0:
        raise ValueError("Gemma 3 query heads must be divisible by KV heads")
    if _value(text, "hidden_activation") != "gelu_pytorch_tanh":
        raise ValueError("Gemma 3 loader supports only gelu_pytorch_tanh")
    if _value(text, "attention_bias") is not False:
        raise ValueError("Gemma 3 loader requires bias-free text attention")
    if float(_value(text, "attention_dropout", 0.0)) != 0.0:
        raise ValueError("Gemma 3 loader requires zero attention dropout")
    if _value(text, "tie_word_embeddings") is not True:
        raise ValueError("Gemma 3 loader requires tied token embeddings")
    if _value(text, "use_bidirectional_attention", False) is not False:
        raise ValueError("Gemma 3 loader does not support bidirectional attention")
    if _value(text, "attn_logit_softcapping") is not None:
        raise ValueError("Gemma 3 loader does not support attention softcapping")
    if _value(text, "final_logit_softcapping") is not None:
        raise ValueError("Gemma 3 loader does not support final-logit softcapping")

    rms_norm_eps = float(_value(text, "rms_norm_eps"))
    query_pre_attn_scalar = float(_value(text, "query_pre_attn_scalar"))
    if rms_norm_eps <= 0.0 or query_pre_attn_scalar <= 0.0:
        raise ValueError("Gemma 3 norm epsilon and attention scalar must be positive")

    layer_types = tuple(_value(text, "layer_types", ()))
    if len(layer_types) != num_hidden_layers:
        raise ValueError("Gemma 3 layer_types must cover every decoder layer")
    unsupported = set(layer_types) - {SLIDING_ATTENTION, FULL_ATTENTION}
    if unsupported:
        raise ValueError(f"unsupported Gemma 3 attention kinds: {sorted(unsupported)}")

    rope_parameters = _value(text, "rope_parameters")
    if not isinstance(rope_parameters, dict):
        raise ValueError("Gemma 3 text config requires per-attention RoPE parameters")
    local_rope = _rope_entry(rope_parameters, SLIDING_ATTENTION)
    global_rope = _rope_entry(rope_parameters, FULL_ATTENTION)
    if local_rope.get("rope_type") != "default":
        raise ValueError("Gemma 3 sliding attention requires default RoPE")
    if global_rope.get("rope_type") != "linear":
        raise ValueError("Gemma 3 full attention requires linear RoPE")
    local_rope_theta = float(local_rope.get("rope_theta", 0.0))
    global_rope_theta = float(global_rope.get("rope_theta", 0.0))
    global_rope_factor = float(global_rope.get("factor", 0.0))
    if min(local_rope_theta, global_rope_theta, global_rope_factor) <= 0.0:
        raise ValueError("Gemma 3 RoPE theta/factor values must be positive")

    return Gemma3TextConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        head_dim=head_dim,
        max_position_embeddings=max_position_embeddings,
        rms_norm_eps=rms_norm_eps,
        query_pre_attn_scalar=query_pre_attn_scalar,
        sliding_window=sliding_window,
        local_rope_theta=local_rope_theta,
        global_rope_theta=global_rope_theta,
        global_rope_factor=global_rope_factor,
        layer_types=layer_types,
    )


def _layer_checkpoint_keys(layer: int) -> dict[str, str | tuple[str, str]]:
    prefix = f"language_model.model.layers.{layer}"
    return {
        "input_norm": f"{prefix}.input_layernorm.weight",
        "q_proj": f"{prefix}.self_attn.q_proj.weight",
        "k_proj": f"{prefix}.self_attn.k_proj.weight",
        "v_proj": f"{prefix}.self_attn.v_proj.weight",
        "q_norm": f"{prefix}.self_attn.q_norm.weight",
        "k_norm": f"{prefix}.self_attn.k_norm.weight",
        "o_proj": f"{prefix}.self_attn.o_proj.weight",
        "post_attn_norm": f"{prefix}.post_attention_layernorm.weight",
        "pre_feedforward_norm": f"{prefix}.pre_feedforward_layernorm.weight",
        "gate_up_proj": (
            f"{prefix}.mlp.gate_proj.weight",
            f"{prefix}.mlp.up_proj.weight",
        ),
        "down_proj": f"{prefix}.mlp.down_proj.weight",
        "post_feedforward_norm": f"{prefix}.post_feedforward_layernorm.weight",
    }


def expected_checkpoint_shapes(config: Gemma3TextConfig) -> dict[str, tuple[int, ...]]:
    """Return the exact text-only raw checkpoint key/shape inventory."""

    hidden = config.hidden_size
    intermediate = config.intermediate_size
    q_width = config.num_attention_heads * config.head_dim
    kv_width = config.num_key_value_heads * config.head_dim
    expected: dict[str, tuple[int, ...]] = {
        "language_model.model.embed_tokens.weight": (config.vocab_size, hidden),
        "language_model.model.norm.weight": (hidden,),
    }
    role_shapes = {
        "input_norm": (hidden,),
        "q_proj": (q_width, hidden),
        "k_proj": (kv_width, hidden),
        "v_proj": (kv_width, hidden),
        "q_norm": (config.head_dim,),
        "k_norm": (config.head_dim,),
        "o_proj": (hidden, q_width),
        "post_attn_norm": (hidden,),
        "pre_feedforward_norm": (hidden,),
        "down_proj": (hidden, intermediate),
        "post_feedforward_norm": (hidden,),
    }
    for layer in range(config.num_hidden_layers):
        for role, key in _layer_checkpoint_keys(layer).items():
            if role == "gate_up_proj":
                gate_key, up_key = key
                expected[gate_key] = (intermediate, hidden)
                expected[up_key] = (intermediate, hidden)
            else:
                assert isinstance(key, str)
                expected[key] = role_shapes[role]
    return expected


def validate_checkpoint_layout(model_path: str | Path, config: Gemma3TextConfig) -> None:
    """Validate exact text keys, shapes and BF16 dtype without materializing data."""
    checkpoint.validate_layout(Path(model_path), expected_checkpoint_shapes(config),
                               text_prefix='language_model.', label='Gemma 3')


def inspect_text_checkpoint(model_path: str | Path) -> Gemma3TextConfig:
    """Resolve config and validate the checkpoint without materializing weights."""

    resolved = Path(model_path).expanduser().resolve()
    hf_config = AutoConfig.from_pretrained(resolved, local_files_only=True)
    config = parse_text_config(hf_config)
    validate_checkpoint_layout(resolved, config)
    return config


def _load_raw_tensors(
    model_path: Path, keys: set[str], *, device: str | torch.device,
) -> dict[str, torch.Tensor]:
    return checkpoint.load_tensors(model_path, keys, device=device, label='Gemma 3')


def load_text_weights(
    model_path: str | Path,
    *,
    device: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Load only Gemma's text weights in the exact Rust role order.

    The Rust checkpoint boundary can materialize this object into the closed
    model facade.  This function alone performs no runtime qualification and
    mints no Verus permission bundle.
    """

    resolved = Path(model_path).expanduser().resolve()
    config = inspect_text_checkpoint(resolved)
    expected = expected_checkpoint_shapes(config)
    raw = _load_raw_tensors(resolved, set(expected), device=device)
    embed = raw["language_model.model.embed_tokens.weight"]
    layers = []
    for layer_index in range(config.num_hidden_layers):
        keys = _layer_checkpoint_keys(layer_index)
        values: list[torch.Tensor] = []
        for role in GEMMA3_LAYER_WEIGHT_ROLES:
            key = keys[role]
            if role == "gate_up_proj":
                gate_key, up_key = key
                value = torch.cat((raw[gate_key], raw[up_key]), dim=0).contiguous()
            else:
                assert isinstance(key, str)
                value = raw[key]
            values.append(value)
        layers.append(tuple(values))

    loaded = {
        "architecture": "gemma3_text",
        "config": config.as_runtime_dict(),
        "model_config_sha256": hashlib.sha256(
            (resolved / "config.json").read_bytes()
        ).hexdigest(),
        "embed_weight": embed,
        "layers": layers,
        "attention_kinds": list(config.layer_types),
        "final_norm": raw["language_model.model.norm.weight"],
        # Gemma 3 ties its output projection to the token embedding matrix.
        "lm_head": embed,
    }
    # Keep the checkpoint parser independent of the runtime module at import
    # time, but exercise the exact future permission-binder contract before a
    # caller can consume loaded tensors.
    from .physical import validate_model_weights_runtime_contract

    validate_model_weights_runtime_contract(
        loaded["embed_weight"],
        loaded["layers"],
        loaded["attention_kinds"],
        loaded["final_norm"],
        loaded["lm_head"],
        loaded["config"],
        config.num_hidden_layers,
    )
    return loaded
# @kernel-bridge-end vosti_kernels::gemma3_checkpoint_loader
