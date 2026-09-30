"""Architecture-neutral RoPE table construction from sealed model config."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math
from typing import Any

import torch


@dataclass(frozen=True)
class PrecomputedTables:
    """Immutable model-policy RoPE tables selected by runtime positions."""

    cos: torch.Tensor
    sin: torch.Tensor

    def __post_init__(self) -> None:
        if self.cos.dim() != 2 or self.sin.dim() != 2:
            raise ValueError("precomputed RoPE tables must be rank-2")
        if self.cos.shape != self.sin.shape:
            raise ValueError("precomputed RoPE cos/sin shapes must agree")
        if self.cos.device != self.sin.device or self.cos.dtype != self.sin.dtype:
            raise ValueError("precomputed RoPE cos/sin placement must agree")

    def select(self, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Select table rows without inspecting runtime position values."""

        if positions.dim() != 1:
            raise ValueError("RoPE positions must be rank-1")
        if positions.dtype not in (torch.int32, torch.int64):
            raise ValueError("RoPE positions must have integer dtype")
        if positions.device != self.cos.device:
            raise ValueError("RoPE positions and precomputed tables must colocate")
        return (
            self.cos.index_select(0, positions),
            self.sin.index_select(0, positions),
        )


def _positive_float(config: Mapping[str, Any], name: str) -> float:
    value = config.get(name)
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or value <= 0
    ):
        raise ValueError(f"RoPE config requires finite positive {name}")
    return float(value)


def _positive_int(config: Mapping[str, Any], name: str) -> int:
    value = config.get(name)
    if type(value) is not int or value <= 0:
        raise ValueError(f"RoPE config requires positive {name}")
    return value


def inverse_frequencies(
    *,
    head_dim: int,
    theta: float,
    scaling: Mapping[str, Any] | None,
    device: torch.device,
) -> torch.Tensor:
    """Build the exact float32 inverse-frequency vector for one policy."""

    if type(head_dim) is not int or head_dim <= 0 or head_dim % 2:
        raise ValueError("RoPE requires a positive even head dimension")
    if (
        isinstance(theta, bool)
        or not isinstance(theta, (int, float))
        or not math.isfinite(float(theta))
        or theta <= 0
    ):
        raise ValueError("RoPE requires a finite positive theta")
    inv_freq = 1.0 / (
        float(theta)
        ** (
            torch.arange(0, head_dim, 2, dtype=torch.int64, device=device).to(
                dtype=torch.float32
            )
            / head_dim
        )
    )
    if scaling is None:
        return inv_freq
    if isinstance(scaling, Mapping) and scaling.get("rope_type") == "proportional":
        if set(scaling) != {"rope_type", "partial_rotary_factor", "factor"}:
            raise ValueError("proportional RoPE differs from the closed policy schema")
        proportion = scaling["partial_rotary_factor"]
        if (
            isinstance(proportion, bool)
            or not isinstance(proportion, (int, float))
            or not math.isfinite(float(proportion))
            or not 0 <= proportion <= 1
        ):
            raise ValueError("proportional RoPE requires a finite fraction in [0, 1]")
        factor = _positive_float(scaling, "factor")
        angles = int(proportion * head_dim // 2)
        # Keep the full-head frequency denominator and half-rotation pairing.
        # Zero frequencies encode identity pairs in the unrotated subspace.
        return torch.cat((inv_freq[:angles], torch.zeros(
            head_dim // 2 - angles, dtype=torch.float32, device=device
        ))) / factor
    if not isinstance(scaling, Mapping) or set(scaling) != {
        "rope_type",
        "factor",
        "low_freq_factor",
        "high_freq_factor",
        "original_max_position_embeddings",
    }:
        raise ValueError("RoPE scaling differs from the closed policy schema")
    if scaling.get("rope_type") != "llama3":
        raise ValueError("unsupported RoPE policy")

    factor = _positive_float(scaling, "factor")
    low_factor = _positive_float(scaling, "low_freq_factor")
    high_factor = _positive_float(scaling, "high_freq_factor")
    old_context = _positive_int(scaling, "original_max_position_embeddings")
    if high_factor <= low_factor:
        raise ValueError("Llama 3 RoPE requires high_freq_factor > low_freq_factor")

    low_wavelength = old_context / low_factor
    high_wavelength = old_context / high_factor
    wavelength = 2 * math.pi / inv_freq
    scaled = torch.where(wavelength > low_wavelength, inv_freq / factor, inv_freq)
    smooth = (old_context / wavelength - low_factor) / (high_factor - low_factor)
    smoothed = (1 - smooth) * scaled / factor + smooth * scaled
    medium = ~(wavelength < high_wavelength) * ~(wavelength > low_wavelength)
    return torch.where(medium, smoothed, scaled)


def tables(
    positions: torch.Tensor,
    *,
    head_dim: int,
    theta: float,
    scaling: Mapping[str, Any] | None,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return half-width cos/sin rows consumed by the verified RoPE kernel."""

    if positions.dim() != 1:
        raise ValueError("RoPE positions must be rank-1")
    inv_freq = inverse_frequencies(
        head_dim=head_dim,
        theta=theta,
        scaling=scaling,
        device=positions.device,
    )
    position_ids = positions.unsqueeze(0)
    inv_expanded = inv_freq[None, :, None].expand(position_ids.shape[0], -1, 1)
    position_expanded = position_ids[:, None, :].float()
    frequencies = (inv_expanded.float() @ position_expanded.float()).transpose(1, 2)
    half = frequencies.squeeze(0)
    return half.cos().to(dtype), half.sin().to(dtype)


def precompute_tables(
    *,
    max_positions: int,
    head_dim: int,
    theta: float,
    scaling: Mapping[str, Any] | None,
    device: torch.device | str,
    dtype: torch.dtype,
) -> PrecomputedTables:
    """Build one table pair from model-static geometry and RoPE policy."""

    if type(max_positions) is not int or max_positions <= 0:
        raise ValueError("RoPE requires a positive maximum position count")
    positions = torch.arange(
        max_positions,
        dtype=torch.int64,
        device=torch.device(device),
    )
    cos, sin = tables(
        positions,
        head_dim=head_dim,
        theta=theta,
        scaling=scaling,
        dtype=dtype,
    )
    return PrecomputedTables(cos=cos, sin=sin)


def runtime_scaling(config: Mapping[str, Any]) -> dict[str, Any] | None:
    """Project a sealed flat runtime config to one closed RoPE policy."""

    kind = config.get("rope_scaling_kind", "none")
    if kind == "none":
        return None
    if kind != "llama3":
        raise ValueError(f"unsupported sealed RoPE scaling kind {kind!r}")
    return {
        "rope_type": "llama3",
        "factor": config.get("rope_factor"),
        "low_freq_factor": config.get("rope_low_frequency_factor"),
        "high_freq_factor": config.get("rope_high_frequency_factor"),
        "original_max_position_embeddings": config.get(
            "rope_original_max_position_embeddings"
        ),
    }


__all__ = [
    "PrecomputedTables",
    "inverse_frequencies",
    "precompute_tables",
    "runtime_scaling",
    "tables",
]
