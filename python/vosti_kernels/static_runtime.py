"""Shared explicit runtime identity and immutable primitive launch bindings."""

import json
import os
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

from .dense_launch_plan import launch_identity
from . import primitive_runtime
from .primitive_runtime import QualifiedPrimitiveRuntime


# @kernel-bridge-begin vosti_kernels::static_primitive_runtime
def freeze_launch_inventory(launches) -> tuple[MappingProxyType, MappingProxyType]:
    """Detach one admitted inventory into immutable kernel-key and call-site maps."""

    keyed, sites = {}, {}
    for launch in launches:
        identity = launch_identity(launch["wrapper"], launch["key"])
        selected = MappingProxyType(dict(launch["config"]))
        if identity in keyed:
            raise RuntimeError("static launch inventory duplicates a kernel key")
        keyed[identity] = selected
        for site in launch["sites"]:
            if site in sites:
                raise RuntimeError("static launch inventory duplicates a call site")
            sites[site] = selected
    if not keyed:
        raise RuntimeError("static launch plan is empty")
    return MappingProxyType(keyed), MappingProxyType(sites)


class PrimitiveRuntimeState:
    """Shared source identity, immutable bindings and primitive lookup."""

    def __init__(self, *, scope, modules, origins, digests, kernel_root,
                 qualification, profile, static_launch_plan):
        self._scope = json.loads(json.dumps(scope))
        self._modules = MappingProxyType(dict(modules))
        self._origins = MappingProxyType(dict(origins))
        self._digests = MappingProxyType(dict(digests))
        self._kernel_root = kernel_root
        self._qualification = None if qualification is None else json.loads(json.dumps(qualification))
        self._profile = None if profile is None else json.loads(json.dumps(profile))
        self._static_launch_plan = static_launch_plan
        self._contracts = {entry["wrapper"]: dict(entry) for entry in self._scope["kernel_contracts"]}
        self._kernel_namespace = SimpleNamespace(**self._modules)

    def binding_identity(self):
        return primitive_runtime.runtime_binding_identity(self._qualification)

    def report(self):
        return {"architecture": self._scope["architecture"], "schema": self._scope["schema"],
            "status": self._scope["status"], **self.binding_identity(),
            "deployment_sha256": None if self._qualification is None else self._qualification["deployment_sha256"],
            "model_profile": None if self._profile is None else self._profile["model"]["name"],
            "kernel_root": self._kernel_root,
            "module_origins": dict(self._origins),
            "module_source_sha256": dict(self._digests),
            "launches": [] if self._profile is None else json.loads(json.dumps(self._profile["launches"]))}

    def verified_for(self, tensor):
        if not tensor.is_cuda:
            raise RuntimeError("the source-attested pipeline received a non-CUDA tensor")
        return self._kernel_namespace

    def static_launch_config(self, wrapper, key):
        try:
            return self._static_launch_plan[launch_identity(wrapper, key)]
        except KeyError as error:
            raise RuntimeError(f"{wrapper} key {key} is absent from the static launch plan") from error

    def kernel_entrypoint(self, kernels, wrapper):
        try:
            contract = self._contracts[wrapper]
        except KeyError as error:
            raise RuntimeError(f"no qualified primitive {wrapper!r}") from error
        return getattr(getattr(kernels, contract["module"]), contract["entrypoint"])


class StaticPrimitiveRuntime(PrimitiveRuntimeState):
    """Flat model config and site bindings for geometry-dependent operations."""

    def __init__(self, *, scope, config, modules, origins, digests, kernel_root,
                 qualification, profile, device, dtype):
        launches = profile["launches"] if qualification is None else qualification["launches"]
        keyed, self._site_configs = freeze_launch_inventory(launches)
        super().__init__(
            scope=scope, modules=modules, origins=origins, digests=digests,
            kernel_root=kernel_root,
            qualification=qualification, profile=profile, static_launch_plan=keyed,
        )
        # Runtime schemas are flat except for the immutable layer schedule.
        detached = json.loads(json.dumps(config))
        if "layer_types" in detached:
            detached["layer_types"] = tuple(detached["layer_types"])
        self._config = MappingProxyType(detached)
        self._device, self._dtype = str(device), dtype

    def _config_for(self, site):
        try:
            return self._site_configs[site]
        except KeyError as error:
            raise RuntimeError(f"call site {site!r} has no static launch") from error

    def report(self):
        return {
            **super().report(),
            "trusted_framework_operations": list(self._scope.get("trusted_framework_operations", ())),
            "deferred_obligations": list(self._scope.get("deferred_obligations", ())),
        }

    def model_config(self):
        return json.loads(json.dumps(dict(self._config)))

    def runtime_config(self):
        return MappingProxyType({**dict(self._config), "device": self._device, "dtype": self._dtype,
            "num_heads": self._config["num_attention_heads"], "num_kv_heads": self._config["num_key_value_heads"]})


def attest_framework_package(family_module, configured_root):
    """Bind loaded package origins to the requested checkout for every family."""
    import importlib
    import sys
    from .kernel_modules import realpath, module_origin, source_sha256

    framework_root = realpath(configured_root)
    package_dir = Path(framework_root) / "python/vosti_kernels"
    families_dir = package_dir / "model_families"
    family_dir = families_dir / family_module
    family_package = f"vosti_kernels.model_families.{family_module}"
    packages = {"vosti_kernels": package_dir,
                "vosti_kernels.model_families": families_dir,
                family_package: family_dir}
    expected = {name: directory / "__init__.py" for name, directory in packages.items()}
    expected.update({f"{family_package}.{name}": family_dir / f"{name}.py"
                     for name in ("deployment", "profile", "physical", "runtime")})
    expected.update({f"vosti_kernels.{name}": package_dir / f"{name}.py" for name in (
        "backend_evidence", "dense_launch_plan", "dense_runtime", "deployment",
        "kernels", "kernel_modules", "static_runtime", "kernel_selection",
        "model_profile", "physical", "primitive_runtime", "rotary")})
    for name, path in ((f"{family_package}.loader", family_dir / "loader.py"),
                       ("vosti_kernels.graph_overlay", package_dir / "graph_overlay.py")):
        if name in sys.modules:
            expected[name] = path
    origins = {name: module_origin(importlib.import_module(name), path,
                                  package_dir=packages.get(name))
               for name, path in expected.items()}
    origins[f"{family_package}.scope"] = realpath(family_dir / "scope.json")
    origins["vosti_kernels.kernel_catalog"] = realpath(package_dir / "kernel_catalog.json")
    return framework_root, origins, {name: source_sha256(path) for name, path in origins.items()}


def admit_runtime(*, family_module, scope, config, deployment, kernel_root=None,
                  deployment_bundle=None, framework_root=None, model_config_sha256=None,
                  device=None, dtype=None, environment=None):
    """Check execution platform, source origins and the sealed deployment once."""
    import torch
    from .backend_evidence import discover_backend_environment
    from .kernel_modules import realpath, load_attested_modules

    selected = device or os.environ.get("CUDA_DEVICE", "cuda:0")
    if torch.device(selected).type != "cuda":
        raise ValueError("source-attested runtime requires a CUDA device")
    if dtype is None:
        dtype = torch.bfloat16
    if dtype != torch.bfloat16:
        raise ValueError("source-attested runtime requires bfloat16")
    root = realpath(kernel_root or os.environ.get("VOSTI_KERNEL_ROOT")
                    or Path(__file__).parents[2] / "kernels")
    framework_root, framework_origins, framework_digests = attest_framework_package(
        family_module, framework_root or os.environ.get("VOSTI_FRAMEWORK_ROOT")
        or Path(__file__).parents[2])
    modules, origins, digests = load_attested_modules(root, scope, label=scope["architecture"])
    qualification = bundle = None
    if deployment_bundle is not None:
        if model_config_sha256 is None:
            raise ValueError("qualified runtime requires the model config digest")
        bundle = deployment.load_bundle(deployment_bundle) if isinstance(deployment_bundle, (str, os.PathLike)) else deployment_bundle
        observed = discover_backend_environment() if environment is None else environment
        qualification = deployment.validate_runtime_binding(bundle, resolved_config=config,
            model_config_sha256=model_config_sha256, environment=observed)
    return SimpleNamespace(device=selected, dtype=dtype, kernel_root=root,
        framework_root=framework_root, modules=modules,
        origins={**framework_origins, **origins}, digests={**framework_digests, **digests},
        qualification=qualification, bundle=bundle)


def load_family_runtime(runtime_type, *, family_module, scope, profile, config, deployment, **kwargs):
    admission = admit_runtime(family_module=family_module, scope=scope, config=config,
                              deployment=deployment, **kwargs)
    return runtime_type(admission.modules, admission.origins, admission.digests,
                        admission.kernel_root, admission.qualification, profile,
                        admission.device, admission.dtype)
# @kernel-bridge-end vosti_kernels::static_primitive_runtime
