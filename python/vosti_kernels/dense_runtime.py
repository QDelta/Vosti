"""Shared immutable runtime capability for qualified dense model families.

Family modules provide a closed profile registry and deployment adapter. This
module owns source attestation, immutable launch-plan binding, and the narrow
primitive capability consumed by checked Rust composition. There is no
process-global selected model and no online launch selection.
"""

from __future__ import annotations

from collections.abc import Callable
import json
import os
from types import MappingProxyType
from typing import Any

import torch

from .physical import PAGE_SIZE
from . import primitive_runtime as PRIMITIVE_RUNTIME
from . import rotary as ROTARY
from .static_runtime import PrimitiveRuntimeState, freeze_launch_inventory, admit_runtime


# @kernel-bridge-begin vosti_kernels::dense_runtime_capability
SHAPE_MATCH_KEYS = (
    "hidden_size",
    "intermediate_size",
    "num_heads",
    "num_kv_heads",
    "head_dim",
)


def _dict_config_str(config: dict) -> str:
    return ",".join(
        f"{key}={value}" for key, value in sorted(config.items())
    )


def _static_launch_inventory(
    launches: list[dict[str, Any]],
) -> tuple[dict[str, str], MappingProxyType]:
    """Bind the common immutable inventory plus dense diagnostic labels."""

    keyed, _sites = freeze_launch_inventory(launches)
    pinned = {
        f"{launch['wrapper']} ({','.join(launch['sites'])})": _dict_config_str(launch["config"])
        for launch in launches
    }
    return pinned, keyed


class DenseRuntimeFamily:
    """Closed family policy used to construct independent runtime objects."""

    def __init__(
        self,
        *,
        architecture: str,
        label: str,
        family_module: str,
        scope_schema: str,
        scope_status: str,
        scope: dict[str, Any],
        model_config: Callable[[dict[str, Any]], dict[str, Any]],
        runtime_model_config: Callable[[dict[str, Any]], dict[str, Any]],
        model_profile_for_config: Callable[[dict[str, Any]], dict[str, Any]],
        model_profile_for_name: Callable[[str | None], dict[str, Any]],
        deployment: object,
        runtime_type: type[Runtime] | None = None,
    ) -> None:
        if scope.get("architecture") != architecture:
            raise RuntimeError(
                f"{label} runtime received a scope for another architecture"
            )
        self.architecture = architecture
        self.label = label
        self.family_module = family_module
        self.scope_schema = scope_schema
        self.scope_status = scope_status
        self.scope = json.loads(json.dumps(scope))
        self.model_config_from_profile = model_config
        self.runtime_model_config_from_profile = runtime_model_config
        self.model_profile_for_config = model_profile_for_config
        self.model_profile_for_name = model_profile_for_name
        self.deployment = deployment
        self.runtime_type = runtime_type or Runtime

        self.page_size = PAGE_SIZE
        self.verified_runtime = dict(self.scope["runtime"])
        self.kernel_contracts = tuple(self.scope["kernel_contracts"])
        self.contracts_by_wrapper = {
            contract["wrapper"]: contract for contract in self.kernel_contracts
        }
        if len(self.contracts_by_wrapper) != len(self.kernel_contracts):
            raise RuntimeError(f"{label} scope contains duplicate kernel wrappers")

        self.kernel_modules = tuple(
            dict.fromkeys(
                contract["module"] for contract in self.kernel_contracts
            )
        )
        module_sources: dict[str, str] = {}
        for contract in self.kernel_contracts:
            previous = module_sources.setdefault(
                contract["module"], contract["source"]
            )
            if previous != contract["source"]:
                raise RuntimeError(
                    f"{label} scope maps one kernel module to multiple sources"
                )
        self.module_sources = MappingProxyType(module_sources)

        self.verified_model_shapes = tuple(
            {
                "name": profile["model"]["name"],
                "hidden_size": profile["model"]["hidden_size"],
                "intermediate_size": profile["model"]["intermediate_size"],
                "num_heads": profile["model"]["num_attention_heads"],
                "num_kv_heads": profile["model"]["num_key_value_heads"],
                "head_dim": profile["model"]["head_dim"],
            }
            for profile in self.scope["model_profiles"]
        )
        if not self.verified_model_shapes:
            raise RuntimeError(f"{label} scope has no model profiles")

    @property
    def package_name(self) -> str:
        return f"vosti_kernels.model_families.{self.family_module}"

    def config_for_profile(self, profile_name: str | None = None) -> dict[str, Any]:
        return self.model_config_from_profile(
            self.model_profile_for_name(profile_name)
        )

    def config_from_bundle(
        self, deployment_bundle: str | os.PathLike[str]
    ) -> dict[str, Any]:
        bundle = self.deployment.load_bundle(deployment_bundle)
        deployment_model = bundle["deployment"]["model"]
        config = deployment_model["resolved_config"]
        profile = self.model_profile_for_config(config)
        if deployment_model["catalog_name"] != profile["model"]["name"]:
            raise ValueError(
                f"{self.label} bundle model name differs from its exact config"
            )
        return self.model_config_from_profile(profile)

    def matched_scope_shape(self, config: dict[str, Any]) -> dict | None:
        actual = {
            "hidden_size": config["hidden_size"],
            "intermediate_size": config["intermediate_size"],
            "num_heads": config["num_heads"],
            "num_kv_heads": config["num_kv_heads"],
            "head_dim": config["head_dim"],
        }
        for model_shape in self.verified_model_shapes:
            if all(
                model_shape[key] == actual[key] for key in SHAPE_MATCH_KEYS
            ):
                return dict(model_shape)
        return None

    def verified_scope_error(self, config: dict[str, Any]) -> str | None:
        actual_runtime = {
            "dtype": str(config["dtype"]),
            "device_type": torch.device(config["device"]).type,
        }
        if (
            actual_runtime == self.verified_runtime
            and self.matched_scope_shape(config) is not None
        ):
            return None
        actual_shape = {key: config[key] for key in SHAPE_MATCH_KEYS}
        names = [shape["name"] for shape in self.verified_model_shapes]
        return (
            f"runtime model/config is outside the {self.label} deployed proof "
            f"scope (attested shapes={names}, runtime={self.verified_runtime}; "
            f"got shape={actual_shape}, runtime={actual_runtime})"
        )

    def _install_static_launch_plan(
        self,
        kernel_digests: dict[str, str],
        bundle: dict[str, Any],
        profile: dict[str, Any],
    ) -> tuple[dict[str, str], MappingProxyType]:
        candidate = bundle.get("candidate")
        deployment = bundle.get("deployment")
        if not isinstance(candidate, dict) or not isinstance(deployment, dict):
            raise RuntimeError("verified serving received a malformed bundle")
        modules_by_source = {
            source: module for module, source in self.module_sources.items()
        }
        for case in candidate["kernel_cases"]:
            source_name = case["source"]
            module_name = modules_by_source.get(source_name)
            if module_name is None:
                raise RuntimeError(
                    f"deployment proof case uses unknown source {source_name!r}"
                )
            actual_digest = kernel_digests[f"triton_kernels.{module_name}"]
            if case["source_sha256"] != actual_digest:
                raise RuntimeError(
                    f"deployment proof source differs from loaded {source_name}: "
                    f"proved={case['source_sha256']}, loaded={actual_digest}"
                )

        try:
            selected_profile = self.model_profile_for_config(
                candidate["model"]["resolved_config"]
            )
            if selected_profile["model"]["name"] != profile["model"]["name"]:
                raise ValueError(
                    f"sealed bundle selected another {self.label} profile"
                )
            projected_launches = [
                {
                    field: launch[field]
                    for field in ("wrapper", "sites", "key", "config")
                }
                for launch in deployment["launches"]
            ]
            if projected_launches != profile["launches"]:
                raise ValueError(
                    "sealed launches differ from the exact model profile"
                )
        except (KeyError, TypeError, ValueError) as error:
            raise RuntimeError(
                f"static launch-plan profile validation failed: {error}"
            ) from error
        return _static_launch_inventory(projected_launches)

    def load_runtime(
        self,
        config: dict[str, Any],
        *,
        kernel_root: str | os.PathLike[str] | None = None,
        deployment_bundle: dict[str, Any] | str | os.PathLike[str] | None = None,
        framework_root: str | os.PathLike[str] | None = None,
        model_config_sha256: str | None = None,
        device: str | None = None,
        dtype: torch.dtype = torch.bfloat16,
        environment: dict[str, Any] | None = None,
    ) -> Runtime:
        profile = self.model_profile_for_config(config)
        admission = admit_runtime(
            family_module=self.family_module, scope=self.scope, config=config,
            deployment=self.deployment, kernel_root=kernel_root,
            deployment_bundle=deployment_bundle, framework_root=framework_root,
            model_config_sha256=model_config_sha256, device=device, dtype=dtype,
            environment=environment,
        )
        runtime_config = {
            **self.runtime_model_config_from_profile(profile),
            "num_heads": int(config["num_attention_heads"]),
            "num_kv_heads": int(config["num_key_value_heads"]),
            "device": admission.device,
            "dtype": admission.dtype,
        }
        scope_error = self.verified_scope_error(runtime_config)
        if scope_error is not None:
            raise ValueError(scope_error)
        if admission.bundle is not None:
            pinned_configs, static_launch_plan = self._install_static_launch_plan(
                admission.digests, admission.bundle, profile
            )
        else:
            pinned_configs, static_launch_plan = _static_launch_inventory(profile["launches"])

        return self.runtime_type(
            family=self,
            config=runtime_config,
            profile=profile,
            modules=admission.modules,
            origins=admission.origins,
            digests=admission.digests,
            kernel_root=admission.kernel_root,
            framework_root=admission.framework_root,
            qualification=admission.qualification,
            pinned_configs=pinned_configs,
            static_launch_plan=static_launch_plan,
        )

    def load_qualified_runtime(
        self, config: dict[str, Any], **kwargs
    ) -> QualifiedRuntime:
        return QualifiedRuntime(self.load_runtime(config, **kwargs))

    def runtime_for_tests(self, config: dict[str, Any]) -> Runtime:
        return self.runtime_type(
            family=self,
            config=config,
            profile=None,
            modules={},
            origins={},
            digests={},
            kernel_root=None,
            framework_root=None,
            qualification=None,
            pinned_configs={},
            static_launch_plan=MappingProxyType({}),
            test_only=True,
        )


class Runtime(PrimitiveRuntimeState):
    """One isolated source-attested implementation of primitive operations."""

    def __init__(
        self,
        *,
        family: DenseRuntimeFamily,
        config: dict[str, Any],
        profile: dict[str, Any] | None,
        modules: dict[str, object],
        origins: dict[str, str],
        digests: dict[str, str],
        kernel_root: str | None,
        framework_root: str | None,
        qualification: dict[str, Any] | None,
        pinned_configs: dict[str, str],
        static_launch_plan: MappingProxyType,
        test_only: bool = False,
    ) -> None:
        self._family = family
        self._config = MappingProxyType(dict(config))
        super().__init__(
            scope=family.scope, modules=modules, origins=origins, digests=digests,
            kernel_root=kernel_root,
            qualification=qualification, profile=profile,
            static_launch_plan=static_launch_plan,
        )
        self._framework_root = framework_root
        self._pinned_configs = MappingProxyType(dict(pinned_configs))
        self._test_only = bool(test_only)
        self._rotary_tables = (
            ROTARY.precompute_tables(
                max_positions=int(self._config["max_position_embeddings"]),
                head_dim=int(self._config["head_dim"]),
                theta=float(self._config["rope_theta"]),
                scaling=ROTARY.runtime_scaling(self._config),
                device=torch.device(self._config["device"]),
                dtype=self._config["dtype"],
            )
            if self._qualification is not None
            else None
        )

    def model_config(self) -> dict[str, Any]:
        if self._profile is None:
            raise RuntimeError(
                f"test-only {self._family.label} runtime has no model profile"
            )
        return self._family.model_config_from_profile(self._profile)

    def runtime_config(self) -> MappingProxyType:
        return self._config

    def report(self) -> dict[str, Any]:
        return {
            **super().report(),
            "verified": bool(self._modules) and not self._test_only,
            "reason": (
                f"loaded from {self._kernel_root}; "
                "deployed scope matched"
                if self._modules
                else "explicit test-only fallback runtime"
            ),
            "config_deterministic": bool(self._static_launch_plan),
            "pinned_configs": dict(self._pinned_configs),
            "runtime_scope": [
                dict(shape) for shape in self._family.verified_model_shapes
            ],
            "runtime_scope_matched": (
                self._family.matched_scope_shape(dict(self._config)) or {}
            ).get("name"),
            "runtime_requirements": dict(self._family.verified_runtime),
            "runtime_scope_supported": (
                self._family.verified_scope_error(dict(self._config)) is None
                if self._profile is not None
                else False
            ),
            "framework_root": self._framework_root,
        }

    def verified_for(self, tensor: torch.Tensor) -> object | None:
        if self._test_only:
            return None
        return super().verified_for(tensor)

    def rotary_tables(
        self, positions: torch.Tensor, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return position rows from the sealed model's immutable RoPE table."""

        if self._rotary_tables is None:
            return ROTARY.tables(
                positions,
                head_dim=int(self._config["head_dim"]),
                theta=float(self._config["rope_theta"]),
                scaling=ROTARY.runtime_scaling(self._config),
                dtype=dtype,
            )
        if dtype != self._rotary_tables.cos.dtype:
            raise ValueError("RoPE input dtype differs from the sealed runtime")
        return self._rotary_tables.select(positions)


class QualifiedRuntime(PRIMITIVE_RUNTIME.QualifiedPrimitiveRuntime):
    """Shared qualification guard with the dense family's additional identity check."""

    def __init__(
        self, runtime: Runtime, *, family: DenseRuntimeFamily | None = None,
    ) -> None:
        selected_family = family or getattr(runtime, "_family", None)
        if not isinstance(selected_family, DenseRuntimeFamily):
            raise TypeError("qualified dense runtime requires a family policy")
        super().__init__(runtime)
        self._family = selected_family

    def _checked_runtime(self) -> Runtime:
        if (hasattr(self._runtime, "_family")
                and self._runtime._family is not self._family):
            raise RuntimeError(
                f"qualified {self._family.label} runtime identity changed after binding"
            )
        return super()._checked_runtime()

    def rotary_tables(
        self, positions: torch.Tensor, dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self._checked_runtime().rotary_tables(positions, dtype)
# @kernel-bridge-end vosti_kernels::dense_runtime_capability


__all__ = [
    "DenseRuntimeFamily",
    "QualifiedRuntime",
    "Runtime",
    "SHAPE_MATCH_KEYS",
    "_dict_config_str",
    "_static_launch_inventory",
]
