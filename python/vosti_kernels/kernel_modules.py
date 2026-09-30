"""Shared source/origin checks for a deployment's exact kernel module catalog."""

import hashlib
import importlib
import os
from pathlib import Path
import sys


# @kernel-bridge-begin vosti_kernels::kernel_module_attestation
def realpath(path) -> str:
    return os.path.realpath(os.path.abspath(os.fspath(path)))


def source_sha256(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as source_file:
        for chunk in iter(lambda: source_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def module_origin(module, expected_file, *, package_dir=None) -> str:
    """Check file, import-spec origin, and (for packages) the entire search path."""

    expected = realpath(expected_file)
    module_file = getattr(module, "__file__", None)
    if not isinstance(module_file, str) or realpath(module_file) != expected:
        raise RuntimeError(
            f"module {getattr(module, '__name__', '<unknown>')!r} loaded from "
            f"{module_file!r}, expected {expected!r}"
        )
    spec_origin = getattr(getattr(module, "__spec__", None), "origin", None)
    if not isinstance(spec_origin, str) or realpath(spec_origin) != expected:
        raise RuntimeError(
            f"module {getattr(module, '__name__', '<unknown>')!r} has import "
            f"origin {spec_origin!r}, expected {expected!r}"
        )
    if package_dir is not None:
        expected_package_dir = realpath(package_dir)
        package_path = getattr(module, "__path__", None)
        resolved_paths = [realpath(path) for path in package_path] if package_path is not None else []
        if resolved_paths != [expected_package_dir]:
            raise RuntimeError(
                f"package {getattr(module, '__name__', '<unknown>')!r} search "
                f"path is {resolved_paths!r}, expected {[expected_package_dir]!r}"
            )
    return expected


def validate_kernel_sources(root: str, scope: dict) -> None:
    """Check the executed source catalog, independent of Git and unrelated files."""
    package_dir = Path(realpath(root)) / "triton_kernels"
    if package_dir.resolve() != package_dir:
        raise RuntimeError("kernel package path is not canonical")
    support = scope.get("runtime_support_sources", [])
    if not {"__init__.py", "runtime_contracts.py", "constants.py"} <= {entry["source"] for entry in support}:
        raise RuntimeError("kernel scope must bind its initializer, runtime contracts, and constants")
    for entry in [*support, *scope["kernel_contracts"]]:
        source = entry["source"]
        if Path(source).name != source or not source.endswith(".py"):
            raise RuntimeError(f"invalid kernel source {source!r}")
        if source_sha256(package_dir / source) != entry["source_sha256"]:
            raise RuntimeError(f"kernel/support source {source} differs from its scope digest")


def load_attested_modules(root: str, scope: dict, *, label: str) -> tuple[dict, dict, dict]:
    """Load the explicitly bound kernel and support sources, with no Git gate."""
    from .physical import PAGE_SIZE

    root = realpath(root)
    package_dir = str(Path(root) / "triton_kernels")
    if not Path(package_dir).is_dir():
        raise RuntimeError(f"no triton_kernels directory under {root}")
    if realpath(package_dir) != package_dir:
        raise RuntimeError(f"{label} kernel package path is not canonical")
    validate_kernel_sources(root, scope)
    sys.path[:] = [path for path in sys.path if not path or realpath(path) != root]
    sys.path.insert(0, root)
    importlib.invalidate_caches()
    origins, digests, modules = {}, {}, {}
    package = importlib.import_module("triton_kernels")
    origins["triton_kernels"] = module_origin(
        package, Path(package_dir) / "__init__.py", package_dir=package_dir,
    )
    digests["triton_kernels"] = source_sha256(origins["triton_kernels"])

    def attest(qualified, source, expected_digest):
        # Catalog entries must refer to direct files in the pinned package.
        if Path(source).name != source or not source.endswith(".py"):
            raise RuntimeError(f"{label} has invalid kernel source {source!r}")
        module = importlib.import_module(qualified)
        path = module_origin(module, Path(package_dir) / source)
        actual = source_sha256(path)
        if actual != expected_digest:
            raise RuntimeError(f"{label} kernel/support source {source} differs from its scope digest")
        origins[qualified], digests[qualified] = path, actual
        return module

    support_sources = scope.get("runtime_support_sources", [])
    for support in support_sources:
        source = support["source"]
        name = "triton_kernels" if source == "__init__.py" else f"triton_kernels.{Path(source).stem}"
        attest(name, source, support["source_sha256"])
    for contract in scope["kernel_contracts"]:
        name = contract["module"]
        if contract["source"] != f"{name}.py":
            raise RuntimeError(f"{label} module/source identity disagrees")
        module = attest(f"triton_kernels.{name}", contract["source"], contract["source_sha256"])
        if not callable(getattr(module, contract["entrypoint"], None)) or not callable(getattr(module, "select_config", None)):
            raise RuntimeError(f"{label} module {name} lacks its entrypoint or colocated selector")
        modules[name] = module
    for name, module in modules.items():
        page_size = getattr(module, "PAGE_SIZE", None)
        if page_size is not None and page_size != PAGE_SIZE:
            raise RuntimeError(f"{name} PAGE_SIZE differs from Engine PAGE_SIZE={PAGE_SIZE}")
    from triton.runtime.autotuner import Autotuner

    autotuners = sorted(f"{name}.{attribute}" for name, module in modules.items()
                       for attribute, value in vars(module).items() if isinstance(value, Autotuner))
    if autotuners:
        raise RuntimeError(f"{label} runtime modules expose timing autotuners: {autotuners}")
    return modules, origins, digests
# @kernel-bridge-end vosti_kernels::kernel_module_attestation
