"""CPU-only loader rejection tests; synthetic modules never execute kernels."""

from copy import deepcopy
import importlib
import shutil
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock

from triton.runtime.autotuner import Autotuner

from vosti_kernels import kernel_modules
from vosti_kernels import backend_evidence
from vosti_kernels.physical import PAGE_SIZE


class KernelModuleTests(unittest.TestCase):
    families = ("qwen3", "llama3", "gemma3", "gemma4")

    def setUp(self):
        self.root = Path(__file__).resolve().parents[2] / "kernels"
        self.scopes = [
            importlib.import_module(
                f"vosti_kernels.model_families.{family}.profile"
            ).scope()
            for family in self.families
        ]

    def modules(self, scope):
        package_dir = self.root / "triton_kernels"
        modules = {}

        def add(name, source):
            module = types.ModuleType(name)
            module.__file__ = str(package_dir / source)
            module.__spec__ = types.SimpleNamespace(origin=module.__file__)
            modules[name] = module
            return module

        package = add("triton_kernels", "__init__.py")
        package.__path__ = [str(package_dir)]
        for support in scope.get("runtime_support_sources", []):
            if support["source"] != "__init__.py":
                add(f"triton_kernels.{Path(support['source']).stem}", support["source"])
        for contract in scope["kernel_contracts"]:
            name = f"triton_kernels.{contract['module']}"
            module = modules.get(name)
            if module is None:
                module = add(name, contract["source"])
            setattr(module, contract["entrypoint"], lambda: None)
            module.select_config = lambda: None
        return modules

    def load(self, scope, modules):
        with mock.patch.object(sys, "path", list(sys.path)), mock.patch.object(
            kernel_modules.importlib, "import_module", side_effect=modules.__getitem__,
        ):
            return kernel_modules.load_attested_modules(
                str(self.root), scope, label=scope["architecture"],
            )

    def test_all_catalogs_bind_exact_source_origins_and_digests(self):
        for scope in self.scopes:
            with self.subTest(family=scope["architecture"]):
                modules, origins, digests = self.load(scope, self.modules(scope))
                self.assertEqual(set(modules), {c["module"] for c in scope["kernel_contracts"]})
                self.assertIn("triton_kernels", origins)
                for name, path in origins.items():
                    self.assertEqual(digests[name], kernel_modules.source_sha256(path))

    def test_all_catalogs_reject_wrong_origin_digest_entrypoint_and_autotuner(self):
        for scope in self.scopes:
            for fault in ("file", "origin", "package_path", "digest", "entrypoint", "selector",
                          "page_size", "autotuner", "source_path"):
                changed = deepcopy(scope)
                modules = self.modules(changed)
                contract = changed["kernel_contracts"][0]
                module = modules[f"triton_kernels.{contract['module']}"]
                if fault == "file":
                    module.__file__ += ".wrong"
                elif fault == "origin":
                    module.__spec__.origin += ".wrong"
                elif fault == "package_path":
                    modules["triton_kernels"].__path__.append("/unreviewed")
                elif fault == "digest":
                    contract["source_sha256"] = "0" * 64
                elif fault == "entrypoint":
                    setattr(module, contract["entrypoint"], None)
                elif fault == "selector":
                    module.select_config = None
                elif fault == "page_size":
                    module.PAGE_SIZE = PAGE_SIZE + 1
                elif fault == "autotuner":
                    module.unused_tuner = object.__new__(Autotuner)
                else:
                    contract["source"] = "../outside.py"
                with self.subTest(family=scope["architecture"], fault=fault):
                    with self.assertRaises(RuntimeError):
                        self.load(changed, modules)

    def test_explicit_support_catalog_cannot_drop_initializer_or_change_digest(self):
        for scope in self.scopes:
            for fault in ("missing_initializer", "missing_contracts", "missing_constants", "digest"):
                changed = deepcopy(scope)
                if fault.startswith("missing_"):
                    removed = {"missing_initializer": "__init__.py",
                               "missing_contracts": "runtime_contracts.py",
                               "missing_constants": "constants.py"}[fault]
                    changed["runtime_support_sources"] = [
                        entry for entry in changed["runtime_support_sources"]
                        if entry["source"] != removed
                    ]
                else:
                    changed["runtime_support_sources"][0]["source_sha256"] = "0" * 64
                with self.subTest(family=scope["architecture"], fault=fault):
                    with self.assertRaises(RuntimeError):
                        self.load(changed, self.modules(changed))

    def test_all_runtime_loaders_work_without_consulting_git(self):
        for family, scope in zip(self.families, self.scopes):
            runtime = importlib.import_module(
                f"vosti_kernels.model_families.{family}.runtime"
            )
            config = runtime.config_for_profile(scope["model_profiles"][0]["model"]["name"])
            with self.subTest(family=family), mock.patch.object(
                backend_evidence.subprocess, "run", side_effect=AssertionError("startup consulted Git"),
            ):
                loaded = runtime.load_runtime(config, kernel_root=str(self.root))
                self.assertEqual(loaded.model_config(), config)

    def test_source_export_ignores_unrelated_files_but_rejects_support_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shutil.copytree(self.root / "triton_kernels", root / "triton_kernels")
            (root / "README.md").write_text("unrelated documentation")
            for scope in self.scopes:
                kernel_modules.validate_kernel_sources(str(root), scope)
            constants = root / "triton_kernels/constants.py"
            original = constants.read_text()
            constants.write_text(original.replace("PAGE_SIZE = 64", "PAGE_SIZE = 32"))
            for scope in self.scopes:
                with self.subTest(family=scope["architecture"]), self.assertRaisesRegex(RuntimeError, "scope digest"):
                    kernel_modules.validate_kernel_sources(str(root), scope)
            constants.write_text(original)
            (root / "triton_kernels/runtime_contracts.py").write_text("# changed support code\n")
            for scope in self.scopes:
                with self.subTest(family=scope["architecture"]), self.assertRaisesRegex(RuntimeError, "scope digest"):
                    kernel_modules.validate_kernel_sources(str(root), scope)

    def test_kernel_package_cannot_redirect_outside_the_checkout(self):
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "triton_kernels").symlink_to(self.root / "triton_kernels")
            with self.assertRaisesRegex(RuntimeError, "not canonical"):
                kernel_modules.load_attested_modules(
                    directory, self.scopes[0], label="test",
                )


if __name__ == "__main__":
    unittest.main()
