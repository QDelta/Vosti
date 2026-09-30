from __future__ import annotations

import unittest
from unittest import mock

import torch

from vosti_kernels import primitive_runtime
from vosti_kernels.static_runtime import PrimitiveRuntimeState
from vosti_kernels.model_families.gemma3 import profile as gemma3_profile
from vosti_kernels.model_families.gemma3 import runtime as gemma3_runtime
from vosti_kernels.model_families.gemma4 import profile as gemma4_profile
from vosti_kernels.model_families.gemma4 import runtime as gemma4_runtime
from vosti_kernels.model_families.llama3 import runtime as llama3_runtime
from vosti_kernels.model_families.qwen3 import runtime as qwen3_runtime


class _FakeRuntime:
    def __init__(self) -> None:
        self.config = {"hidden_size": 4}
        self.calls = []

    def runtime_config(self):
        return self.config

    def verified_for(self, tensor):
        self.calls.append(("verified_for", tensor))
        return "kernels"

    def static_launch_config(self, wrapper, key):
        self.calls.append(("static_launch_config", wrapper, key))
        return {"sealed": 1}

    def kernel_entrypoint(self, kernels, wrapper):
        self.calls.append(("kernel_entrypoint", kernels, wrapper))
        return "entrypoint"


class PrimitiveRuntimeTests(unittest.TestCase):
    def test_all_families_share_the_qualification_guard(self) -> None:
        for family in (qwen3_runtime, llama3_runtime, gemma3_runtime, gemma4_runtime):
            with self.subTest(family=family.__name__):
                self.assertTrue(issubclass(
                    family.QualifiedRuntime, primitive_runtime.QualifiedPrimitiveRuntime,
                ))

    def test_live_binding_checks_do_not_construct_diagnostic_reports(self) -> None:
        geometry = dict(hidden_size=16, num_heads=2, num_kv_heads=1, head_dim=8)
        runtimes = (
            (qwen3_runtime, qwen3_runtime.runtime_for_tests(
                **geometry, rms_norm_eps=1e-6,
            )),
            (llama3_runtime, llama3_runtime.runtime_for_tests(**geometry)),
            (gemma3_runtime, gemma3_runtime.Runtime(
                modules={}, origins={}, digests={}, kernel_root="test-only",
                qualification=None,
                profile=gemma3_profile.model_profile_for_name("gemma-3-4b-it-text"),
                device="cpu", dtype=torch.float32,
            )),
            (gemma4_runtime, gemma4_runtime.Runtime(
                modules={}, origins={}, digests={}, kernel_root="test-only",
                qualification=None,
                selected_profile=gemma4_profile.model_profile_for_name("gemma-4-31b-it-text"),
                device="cpu", dtype=torch.float32,
            )),
        )
        qualification = {
            "candidate_sha256": "a" * 64,
            "qualification_report_sha256": "b" * 64,
            "deployment_sha256": "c" * 64,
        }
        for family, runtime in runtimes:
            with self.subTest(family=family.__name__):
                self.assertEqual(runtime.binding_identity(), {
                    "engine_reachable": False,
                    "backend_qualified": False,
                    "qualification": None,
                })
                # Synthetic identity only exercises CPU guards, not backend
                # qualification or numerical execution.
                runtime._qualification = dict(qualification)
                identity = runtime.binding_identity()
                report = runtime.report()
                self.assertEqual(identity, {key: report[key] for key in identity})
                self.assertIsInstance(runtime, PrimitiveRuntimeState)
                report["module_origins"]["unreviewed"] = "not-a-real-source"
                report["launches"].append({"unreviewed": True})
                fresh = runtime.report()
                self.assertNotIn("unreviewed", fresh["module_origins"])
                self.assertNotIn({"unreviewed": True}, fresh["launches"])
                common_fields = {
                    "architecture", "schema", "status", "engine_reachable",
                    "backend_qualified", "qualification", "deployment_sha256",
                    "model_profile", "kernel_root", "module_origins",
                    "module_source_sha256", "launches",
                }
                self.assertLessEqual(common_fields, set(fresh))
                capability = family.QualifiedRuntime(runtime)
                expected = runtime.runtime_config()
                with mock.patch.object(runtime, "report", side_effect=AssertionError(
                    "primitive guard must not build a diagnostic report"
                )):
                    self.assertEqual(capability.runtime_config(), expected)
                    for key in qualification:
                        runtime._qualification[key] = "changed"
                        with self.subTest(field=key), self.assertRaisesRegex(
                            RuntimeError, "identity changed"
                        ):
                            capability.runtime_config()
                        runtime._qualification[key] = qualification[key]
                    runtime._qualification = None
                    with self.assertRaisesRegex(RuntimeError, "identity changed"):
                        capability.runtime_config()
                    runtime._qualification = dict(qualification)
                    self.assertEqual(capability.runtime_config(), expected)
                    if hasattr(runtime, "_family"):
                        runtime._family = object()
                        with self.assertRaisesRegex(RuntimeError, "identity changed"):
                            capability.runtime_config()

    def test_explicit_capability_delegates_without_runtime_selection(self) -> None:
        runtime = _FakeRuntime()

        self.assertEqual(
            primitive_runtime.runtime_config_or_none(runtime), runtime.config
        )
        self.assertEqual(primitive_runtime.runtime_config(runtime), runtime.config)
        self.assertEqual(
            primitive_runtime.verified_for(runtime, "tensor"), "kernels"
        )
        self.assertEqual(
            primitive_runtime.static_launch_config(
                runtime, "linear", {"n": 4}
            ),
            {"sealed": 1},
        )
        self.assertEqual(
            primitive_runtime.kernel_entrypoint(
                runtime, "kernels", "linear"
            ),
            "entrypoint",
        )
        self.assertEqual(
            runtime.calls,
            [
                ("verified_for", "tensor"),
                ("static_launch_config", "linear", {"n": 4}),
                ("kernel_entrypoint", "kernels", "linear"),
            ],
        )

    def test_none_runtime_has_only_the_explicit_fallback(self) -> None:
        self.assertIsNone(primitive_runtime.runtime_config_or_none(None))
        self.assertIsNone(primitive_runtime.verified_for(None, "tensor"))

    def test_runtime_configuration_must_be_a_mapping(self) -> None:
        runtime = _FakeRuntime()
        runtime.config = "not-a-mapping"
        with self.assertRaisesRegex(TypeError, "must be a mapping"):
            primitive_runtime.runtime_config(runtime)


if __name__ == "__main__":
    unittest.main()
