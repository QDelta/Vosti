"""Attested direct-kernel catalogs and deployed dense proof cases.

This module contains no proof pass.  Structural and race drivers consume the
same source identities, output surfaces, and configuration matrix without
depending on one another's acceptance results.
"""

from __future__ import annotations

import hashlib
import importlib
import json

import os
from pathlib import Path
import sys

_FRAMEWORK_ROOT = Path(__file__).resolve().parents[2]
_PYTHON_ROOT = str(Path(_FRAMEWORK_ROOT) / "python")
if _PYTHON_ROOT not in sys.path:
    sys.path.insert(0, _PYTHON_ROOT)
_KERNEL_ROOT = str(Path(_FRAMEWORK_ROOT) / "kernels")
if (Path(_KERNEL_ROOT) / "ir").is_dir() and _KERNEL_ROOT not in sys.path:
    sys.path.insert(0, _KERNEL_ROOT)

from vosti_kernels.model_profile import load_scope
from ir.annotations import parse_verif_goal
from ir.annotation_lowering import lower_proof_goal
from triton_kernels.fattn_paged import CONFIGS as FATTN_CONFIGS
from triton_kernels.fattn_paged import PAGE_SIZE
from triton_kernels.fattn_paged import select_config as select_fattn_paged_config
from triton_kernels.matmul import (
    CONFIG_INDEX_BY_NK,
    select_config as select_matmul_config,
)
from vosti_kernels.dense_launch_plan import (
    QK_NORM_KINDS,
    derive_static_launch_plan,
)
from scripts.verification.engine_kernel_bindings import required_kernel_goal_bindings
from scripts.verification.kernel_annotation_identity import annotation_contract_digest
from scripts.audit.runtime_bridge_scope import (
    validate_marked_source_spans,
    validate_runtime_bridge_scope,
    validate_source_span_schema,
)


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
_RUNTIME_FIELDS = {"dtype", "device_type"}
_EVIDENCE_KINDS = {
    "regional_certificate",
    "conditional_relational_certificate",
    "exact_effect_certificate",
}
_BRIDGE_REQUIRED_FIELDS = {"source_spans", "kernel_annotation_digest"}


def constexpr_values(config: dict) -> dict:
    return {
        key: value
        for key, value in config.items()
        if key not in {"num_warps", "num_stages"}
    }


def _kernel_selectors(scope: "DenseKernelScope") -> dict[str, object]:
    """Load every selector from the same module as its kernel."""

    selectors = {}
    for contract in scope.contracts:
        module = importlib.import_module(
            f"triton_kernels.{contract['module']}"
        )
        selector = getattr(module, "select_config", None)
        if not callable(selector):
            raise ValueError(
                f"kernel module {contract['module']!r} has no select_config"
            )
        selectors[contract["wrapper"]] = selector
    return selectors


class DenseKernelScope:
    """One explicit offline family view over the shared dense kernel policy."""

    def __init__(
        self,
        *,
        architecture: str,
        scope_path: str | os.PathLike[str],
        model_profiles,
        model_shape,
        qk_norm_kind: str,
    ) -> None:
        if not isinstance(architecture, str) or not architecture:
            raise ValueError("dense kernel scope requires an architecture")
        if qk_norm_kind not in QK_NORM_KINDS:
            raise ValueError("dense kernel scope has an invalid Q/K-norm choice")
        self.verified_scope = load_scope(scope_path)
        self.architecture = architecture
        self.qk_norm_kind = qk_norm_kind
        self.model_profiles = tuple(model_profiles())
        self.model_shape = model_shape
        self.contracts = tuple(self.verified_scope["kernel_contracts"])
        self.contracts_by_kernel = {
            (contract["source"], contract["kernel"]): contract
            for contract in self.contracts
        }
        self.contracts_by_wrapper = {
            contract["wrapper"]: contract for contract in self.contracts
        }
        self.required_post_tensors = {
            (contract["source"], contract["kernel"]): set(contract["outputs"])
            for contract in self.contracts
        }
        self.deployed_shapes = tuple(
            self.model_shape(profile) for profile in self.model_profiles
        )

    def verified_model_shapes(self) -> tuple[dict, ...]:
        return verified_model_shapes(self)

    def contract_for(self, source_file: str, kernel_name: str) -> dict:
        return contract_for(self, source_file, kernel_name)

    def validate_post_surface(
        self, source_file: str, kernel_name: str, source: str
    ) -> set[str]:
        return validate_post_surface(self, source_file, kernel_name, source)

    def validate_contract_catalog(self) -> None:
        validate_contract_catalog(self)

    def verified_model_shape(self, name: str) -> dict:
        return verified_model_shape(self, name)

    def selected_launch_configs(
        self,
        shape: dict,
        additional_matmul_nk: tuple[tuple[int, int], ...] = (),
    ) -> list[dict]:
        return selected_launch_configs(self, shape, additional_matmul_nk)

    def selected_deployment_cases(
        self,
        shape: dict,
        additional_matmul_nk: tuple[tuple[int, int], ...] = (),
    ) -> list[tuple[str, str, dict]]:
        return selected_deployment_cases(self, shape, additional_matmul_nk)

    def linear_static_policy(self) -> dict:
        return linear_static_policy(self)

    def deployed_cases(self) -> list:
        return deployed_cases(self)

    def validate_deployed_case_coverage(self, cases: list) -> None:
        validate_deployed_case_coverage(self, cases)


def verified_model_shapes(scope: DenseKernelScope) -> tuple[dict, ...]:
    return tuple(dict(shape) for shape in scope.deployed_shapes)


def contract_for(
    scope: DenseKernelScope,
    source_file: str,
    kernel_name: str,
) -> dict:
    try:
        return scope.contracts_by_kernel[(source_file, kernel_name)]
    except KeyError as error:
        raise ValueError(
            f"no engine kernel contract for {(source_file, kernel_name)}"
        ) from error


def validate_post_surface(
    scope: DenseKernelScope,
    source_file: str,
    kernel_name: str,
    source: str,
) -> set[str]:
    key = (source_file, kernel_name)
    required = scope.required_post_tensors.get(key)
    if required is None:
        raise ValueError(f"no framework output-surface contract for {key}")
    goal = required_kernel_goal_bindings(contract_for(scope, source_file, kernel_name))["batch"]
    annotation = parse_verif_goal(source, kernel_name, goal)
    if annotation is None:
        raise ValueError(f"no bound batch @verif goal {goal!r} for {key}")
    annotation = lower_proof_goal(annotation)
    named = set()
    for post in annotation.post_conditions:
        left_name = post.left.side.name
        right_name = post.right.side.name
        if left_name != right_name:
            raise ValueError(
                f"@post for {key} compares different tensors: "
                f"{left_name!r} and {right_name!r}"
            )
        named.add(left_name)
    missing = required - named
    if missing:
        raise ValueError(f"@post for {key} omits required outputs {sorted(missing)}")
    return named


def validate_contract_catalog(scope: DenseKernelScope) -> None:
    verified_scope = scope.verified_scope
    if set(verified_scope) != _SCOPE_FIELDS:
        raise ValueError(
            "verified scope fields differ from the closed schema: "
            f"expected={sorted(_SCOPE_FIELDS)}, got={sorted(verified_scope)}"
        )
    if verified_scope["architecture"] != scope.architecture:
        raise ValueError("direct kernel scope identifies another architecture")
    if verified_scope.get("schema") != "vosti.model-runtime-scope.v1":
        raise ValueError("direct kernel scope has an unsupported schema")
    if (
        verified_scope.get("status") != "qualification_scope"
        or verified_scope.get("engine_reachable") is not False
    ):
        raise ValueError("direct kernel qualification scope must remain fail-closed")
    validate_runtime_bridge_scope()
    if (
        not isinstance(verified_scope["model_profiles"], list)
        or not scope.model_profiles
    ):
        raise ValueError("verified model-profile catalog must be a nonempty list")
    shape_names = set()
    shape_geometries = set()
    for profile in scope.model_profiles:
        if not isinstance(profile, dict) or set(profile) != {"model", "launches"}:
            raise ValueError("verified model profile differs from the closed schema")
        model_shape = scope.model_shape(profile)
        name = model_shape["name"]
        if not isinstance(name, str) or not name:
            raise ValueError("verified model-shape names must be nonempty strings")
        if name in shape_names:
            raise ValueError(f"duplicate verified model-shape name {name!r}")
        if any(
            not isinstance(value, int) or value <= 0
            for field, value in model_shape.items()
            if field != "name"
        ):
            raise ValueError("verified model-shape values must be positive integers")
        geometry = tuple(
            model_shape[field] for field in sorted(set(model_shape) - {"name"})
        )
        if geometry in shape_geometries:
            raise ValueError(f"duplicate verified model geometry for {name!r}")
        expected_launches = selected_launch_configs(
            scope,
            model_shape,
            ((profile["model"]["vocab_size"], model_shape["hidden"]),),
        )
        if profile["launches"] != expected_launches:
            raise ValueError(
                f"verified model profile {name!r} differs from the pinned "
                "offline shared-dense launch policy"
            )
        shape_names.add(name)
        shape_geometries.add(geometry)
    runtime = verified_scope.get("runtime")
    if not isinstance(runtime, dict) or set(runtime) != _RUNTIME_FIELDS:
        raise ValueError("verified runtime requirements differ from the closed schema")
    if runtime != {
        "dtype": "torch.bfloat16",
        "device_type": "cuda",
    }:
        raise ValueError(f"unsupported verified runtime requirements: {runtime!r}")
    required_fields = {
        "wrapper",
        "module",
        "entrypoint",
        "source",
        "kernel",
        "source_sha256",
        "outputs",
        "evidence",
    }
    optional_fields = {"bridge", "proof_goals"}
    wrappers = set()
    kernels = set()
    for contract in scope.contracts:
        missing = required_fields - contract.keys()
        extra = contract.keys() - required_fields - optional_fields
        if missing or extra:
            raise ValueError(
                f"invalid engine kernel contract fields: missing={sorted(missing)}, "
                f"extra={sorted(extra)}"
            )
        required_kernel_goal_bindings(contract)
        wrapper = contract["wrapper"]
        key = (contract["source"], contract["kernel"])
        if wrapper in wrappers:
            raise ValueError(f"duplicate engine wrapper contract {wrapper!r}")
        if key in kernels:
            raise ValueError(f"duplicate concrete kernel contract {key}")
        if not contract["outputs"]:
            raise ValueError(f"engine kernel contract {wrapper!r} has no outputs")
        if not _is_digest(contract["source_sha256"]):
            raise ValueError(
                f"engine kernel contract {wrapper!r} has no source digest"
            )
        if contract["evidence"] not in _EVIDENCE_KINDS:
            raise ValueError(
                f"engine kernel contract {wrapper!r} has unknown evidence "
                f"{contract['evidence']!r}"
            )
        bridge = contract.get("bridge")
        if bridge is None:
            raise ValueError(
                f"engine kernel contract {wrapper!r} has no attested bridge"
            )
        if not isinstance(bridge, dict) or set(bridge) != _BRIDGE_REQUIRED_FIELDS:
            raise ValueError(
                f"invalid bridge contract fields for {wrapper!r}; "
                "only source spans and annotation identity are accepted"
            )
        validate_source_span_schema(
            f"bridge contract for {wrapper!r}", bridge["source_spans"]
        )
        if not _is_digest(bridge["kernel_annotation_digest"]):
            raise ValueError(f"invalid kernel annotation digest for {wrapper!r}")
        wrappers.add(wrapper)
        kernels.add(key)

def _is_digest(value) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _require_digest(wrapper: str, surface: str, expected: str, actual: str) -> None:
    if actual != expected:
        raise ValueError(
            f"bridge {surface} digest mismatch for {wrapper!r}: "
            f"expected {expected}, got {actual}"
        )


def validate_bridge_surfaces(
    contract: dict,
    kernel_source: str,
    source_overrides: dict[str, str] | None = None,
) -> None:
    """Fail closed when any reviewed source or annotation surface changes.

    This is deliberately operation-agnostic: it hashes named source spans and
    the parsed kernel annotation, but does not interpret their semantics. The
    cross-language implication remains an explicit reviewed TCB item.
    """

    bridge = contract.get("bridge")
    if bridge is None:
        return

    wrapper = contract["wrapper"]
    _require_digest(
        wrapper,
        "kernel annotation",
        bridge["kernel_annotation_digest"],
        annotation_contract_digest(kernel_source, contract["kernel"]),
    )
    validate_marked_source_spans(
        wrapper, bridge["source_spans"], source_overrides
    )


def verified_model_shape(scope: DenseKernelScope, name: str) -> dict:
    """Return one normalized shape from the closed verified catalog."""

    matches = [
        shape for shape in scope.deployed_shapes if shape["name"] == name
    ]
    if len(matches) != 1:
        raise ValueError(f"model {name!r} is not in the verified shape catalog")
    return dict(matches[0])


def selected_launch_configs(
    scope: DenseKernelScope,
    shape: dict,
    additional_matmul_nk: tuple[tuple[int, int], ...] = (),
) -> list[dict]:
    """Resolve the exact launch policy for one model geometry.

    ``additional_matmul_nk`` covers model constants not needed by the engine
    proof shape (currently the vocabulary projection).  No batch-dependent
    dimension appears in this API.
    """

    return derive_static_launch_plan(
        shape,
        additional_matmul_nk,
        composition={"qk_norm": scope.qk_norm_kind},
        selectors=_kernel_selectors(scope),
    )


def linear_static_policy(scope: DenseKernelScope) -> dict:
    """Return this family's exact non-batch linear deployment policy."""

    fallback_key = (1, 1)
    if fallback_key in CONFIG_INDEX_BY_NK:
        raise ValueError("linear fallback probe collides with a selected shape")
    selected_nk = {
        (launch["key"]["n"], launch["key"]["k"])
        for shape in scope.deployed_shapes
        for launch in selected_launch_configs(scope, shape)
        if launch["wrapper"] == "linear"
    }
    return {
        "default": dict(select_matmul_config(*fallback_key)),
        "by_nk": [
            {
                "n": n,
                "k": k,
                "config": dict(select_matmul_config(n, k)),
            }
            for n, k in sorted(selected_nk & CONFIG_INDEX_BY_NK.keys())
        ],
    }


def _shape_cases(
    scope: DenseKernelScope,
    shape: dict,
    *,
    attention_configs: tuple[dict, ...],
    additional_matmul_nk: tuple[tuple[int, int], ...] = (),
) -> list[tuple[str, str, dict]]:
    cases = []
    identities = set()

    def add(source: str, kernel: str, constants: dict) -> None:
        identity = (source, kernel, tuple(sorted(constants.items())))
        if identity not in identities:
            identities.add(identity)
            cases.append((source, kernel, dict(constants)))

    launches = selected_launch_configs(scope, shape, additional_matmul_nk)
    for launch in launches:
        if launch["wrapper"] == "paged_attention":
            continue
        contract = scope.contracts_by_wrapper[launch["wrapper"]]
        add(
            contract["source"],
            contract["kernel"],
            constexpr_values(launch["config"]),
        )

    for config in attention_configs:
        constants = constexpr_values(config)
        if PAGE_SIZE % constants["BLOCK_N"] != 0:
            raise ValueError(
                f"attention config {constants} does not divide page size "
                f"{PAGE_SIZE} for {shape['name']}"
            )
        contract = scope.contracts_by_wrapper["paged_attention"]
        add(
            contract["source"],
            contract["kernel"],
            {
                **constants,
                "D_HEAD": shape["head_dim"],
                "PAGE_BLOCK_SIZE": PAGE_SIZE,
            },
        )
    return cases


def selected_deployment_cases(
    scope: DenseKernelScope,
    shape: dict,
    additional_matmul_nk: tuple[tuple[int, int], ...] = (),
) -> list[tuple[str, str, dict]]:
    """Return only the exact specializations selected for one deployment."""

    attention = select_fattn_paged_config(shape["head_dim"])
    return _shape_cases(
        scope,
        shape,
        attention_configs=(attention,),
        additional_matmul_nk=additional_matmul_nk,
    )


def deployed_cases(scope: DenseKernelScope) -> list:
    """Return the union proof surface used by the repository-wide gate."""

    cases = []
    identities = set()
    for shape in scope.deployed_shapes:
        compatible = tuple(
            config
            for config in FATTN_CONFIGS
            if PAGE_SIZE % config["BLOCK_N"] == 0
        )
        for source, kernel, constants in _shape_cases(
            scope,
            shape, attention_configs=compatible
        ):
            identity = (source, kernel, tuple(sorted(constants.items())))
            if identity not in identities:
                identities.add(identity)
                cases.append((source, kernel, constants))
    return cases


def validate_deployed_case_coverage(
    scope: DenseKernelScope,
    cases: list,
) -> None:
    """Require the generated proof matrix to match the attested catalog."""

    proved_kernels = {(source, kernel) for source, kernel, _ in cases}
    declared_kernels = set(scope.required_post_tensors)
    if proved_kernels != declared_kernels:
        raise ValueError(
            "deployed proof cases and engine contracts differ: "
            f"proof-only={sorted(proved_kernels - declared_kernels)}, "
            f"contract-only={sorted(declared_kernels - proved_kernels)}"
        )
