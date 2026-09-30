"""Qwen3 policy adapter for the architecture-neutral dense kernel scope."""

from __future__ import annotations

from pathlib import Path

from scripts.deployment.dense_kernel_scope import DenseKernelScope, validate_bridge_surfaces

from vosti_kernels.dense_launch_plan import QK_NORM_RMS
from vosti_kernels.model_families.qwen3.profile import (
    model_profiles,
    model_shape,
)


SCOPE = DenseKernelScope(
    architecture="qwen3",
    scope_path=(
        Path(__file__).resolve().parents[3]
        / "python/vosti_kernels/model_families/qwen3/scope.json"
    ),
    model_profiles=model_profiles,
    model_shape=model_shape,
    qk_norm_kind=QK_NORM_RMS,
)

REQUIRED_POST_TENSORS = SCOPE.required_post_tensors
DEPLOYED_SHAPES = SCOPE.deployed_shapes
verified_model_shapes = SCOPE.verified_model_shapes
contract_for = SCOPE.contract_for
validate_post_surface = SCOPE.validate_post_surface
validate_contract_catalog = SCOPE.validate_contract_catalog
verified_model_shape = SCOPE.verified_model_shape
selected_launch_configs = SCOPE.selected_launch_configs
selected_deployment_cases = SCOPE.selected_deployment_cases
deployed_cases = SCOPE.deployed_cases
validate_deployed_case_coverage = SCOPE.validate_deployed_case_coverage
linear_static_policy = SCOPE.linear_static_policy


__all__ = [
    "DEPLOYED_SHAPES",
    "REQUIRED_POST_TENSORS",
    "SCOPE",
    "contract_for",
    "deployed_cases",
    "linear_static_policy",
    "selected_deployment_cases",
    "selected_launch_configs",
    "validate_bridge_surfaces",
    "validate_contract_catalog",
    "validate_deployed_case_coverage",
    "validate_post_surface",
    "verified_model_shape",
    "verified_model_shapes",
]
