"""Text-only Qwen3 checkpoint inspection and weight loading.

This module is intentionally outside the verified core. It loads a local
Transformers Qwen checkpoint and returns PyTorch tensors that the Rust
boundary wraps as tensor handles. Runtime qualification is a separate phase
and has no loader side effect.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM

from . import physical as qwen3_physical
from .profile import model_profile_for_config, project_config


# @kernel-bridge-begin vosti_kernels::qwen3_checkpoint_loader
def inspect_text_checkpoint(model_path: str | Path) -> dict:
    """Validate and project one supported dense Qwen3 text config."""

    config_path = Path(model_path).expanduser().resolve() / "config.json"
    config_bytes = config_path.read_bytes()
    resolved_config = project_config(json.loads(config_bytes))
    model_profile_for_config(resolved_config)
    return resolved_config


def load_text_weights(
    model_path: str | Path,
    *,
    device: str | torch.device = "cpu",
) -> dict:
    """Load Qwen3 weights in the exact Rust role order without qualification."""

    resolved_path = Path(model_path).expanduser().resolve()
    resolved_config = inspect_text_checkpoint(resolved_path)
    model = AutoModelForCausalLM.from_pretrained(resolved_path, dtype=torch.bfloat16).to(
        device
    )
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)

    embed_weight = model.model.embed_tokens.weight.contiguous()

    layers = []
    for layer in model.model.layers:
        attn = layer.self_attn
        mlp = layer.mlp
        gate_up = torch.cat(
            [mlp.gate_proj.weight.contiguous(), mlp.up_proj.weight.contiguous()],
            dim=0,
        ).contiguous()
        layers.append(
            (
                layer.input_layernorm.weight.contiguous(),
                attn.q_proj.weight.contiguous(),
                attn.k_proj.weight.contiguous(),
                attn.v_proj.weight.contiguous(),
                attn.q_norm.weight.contiguous(),
                attn.k_norm.weight.contiguous(),
                attn.o_proj.weight.contiguous(),
                layer.post_attention_layernorm.weight.contiguous(),
                gate_up,
                mlp.down_proj.weight.contiguous(),
            )
        )

    final_norm = model.model.norm.weight.contiguous()
    lm_head = model.lm_head.weight.contiguous()
    qwen3_physical.validate_model_weights_runtime_contract(
        embed_weight,
        layers,
        final_norm,
        lm_head,
        {
            **resolved_config,
            "num_heads": int(resolved_config["num_attention_heads"]),
            "num_kv_heads": int(resolved_config["num_key_value_heads"]),
            "device": device,
            "dtype": embed_weight.dtype,
        },
        int(resolved_config["num_hidden_layers"]),
    )
    return {
        "architecture": "qwen3",
        "config": resolved_config,
        "model_config_sha256": hashlib.sha256(
            (resolved_path / "config.json").read_bytes()
        ).hexdigest(),
        "embed_weight": embed_weight,
        "layers": layers,
        "attention_kinds": ["full_attention"] * len(layers),
        "final_norm": final_norm,
        "lm_head": lm_head,
    }
# @kernel-bridge-end vosti_kernels::qwen3_checkpoint_loader
