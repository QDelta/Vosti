import copy
import unittest
from unittest import mock

import torch

from vosti_kernels.model_families.gemma3 import profile as gemma3_scope
from vosti_kernels.model_families.gemma3.loader import (
    FULL_ATTENTION,
    SLIDING_ATTENTION,
)
from vosti_kernels.model_families.gemma3.runtime import (
    QualifiedRuntime,
    _validate_model_config,
    config_for_profile,
    load_qualified_runtime,
    load_runtime,
)


GEMMA3_4B_PROFILE = "gemma-3-4b-it-text"
GEMMA3_12B_PROFILE = "gemma-3-12b-it-text"
GEMMA3_27B_PROFILE = "gemma-3-27b-it-text"


class FakeRuntime:
    def __init__(self, report):
        self.current_report = report
        self.paged_attention_calls = []
        self.qk_norm_calls = []

    def report(self):
        return self.current_report

    def binding_identity(self):
        return self.current_report

    def paged_attention(self, q, k_cache, v_cache, step, **kwargs):
        self.paged_attention_calls.append(
            (q, k_cache, v_cache, step, kwargs)
        )
        return "qualified_attention"

    def runtime_config(self):
        return {"hidden_size": 2560, "num_heads": 8}

    def qk_norm(self, q, k, q_weight, k_weight):
        self.qk_norm_calls.append((q, k, q_weight, k_weight))
        return ("normalized_q", "normalized_k")

    def verified_for(self, tensor):
        return ("verified", tensor)

    def static_launch_config(self, wrapper, key):
        return (wrapper, tuple(sorted(key.items())))

    def kernel_entrypoint(self, kernels, wrapper):
        return (kernels, wrapper)


class Gemma3RuntimeScopeTests(unittest.TestCase):
    def test_scope_has_exact_4b_12b_and_27b_text_profiles(self) -> None:
        config = config_for_profile(GEMMA3_4B_PROFILE)
        self.assertEqual(config["hidden_size"], 2560)
        self.assertEqual(config["head_dim"], 256)
        self.assertEqual(len(config["layer_types"]), 34)
        self.assertEqual(
            [index for index, kind in enumerate(config["layer_types"]) if kind == FULL_ATTENTION],
            [5, 11, 17, 23, 29],
        )

        medium = config_for_profile(GEMMA3_12B_PROFILE)
        self.assertEqual(medium["hidden_size"], 3840)
        self.assertEqual(medium["intermediate_size"], 15360)
        self.assertEqual(medium["head_dim"], 256)
        self.assertEqual(medium["num_attention_heads"], 16)
        self.assertEqual(medium["num_key_value_heads"], 8)
        self.assertEqual(len(medium["layer_types"]), 48)
        self.assertEqual(
            [
                index
                for index, kind in enumerate(medium["layer_types"])
                if kind == FULL_ATTENTION
            ],
            [5, 11, 17, 23, 29, 35, 41, 47],
        )

        wide = config_for_profile(GEMMA3_27B_PROFILE)
        self.assertEqual(wide["hidden_size"], 5376)
        self.assertEqual(wide["intermediate_size"], 21504)
        self.assertEqual(wide["head_dim"], 128)
        self.assertEqual(wide["num_attention_heads"], 32)
        self.assertEqual(wide["num_key_value_heads"], 16)
        self.assertEqual(len(wide["layer_types"]), 62)
        self.assertEqual(
            [
                index
                for index, kind in enumerate(wide["layer_types"])
                if kind == FULL_ATTENTION
            ],
            [5, 11, 17, 23, 29, 35, 41, 47, 53, 59],
        )

        runtime = load_runtime(config)
        report = runtime.report()
        self.assertEqual(report["architecture"], "gemma3_text")
        self.assertEqual(report["status"], "qualification_scope")
        self.assertFalse(report["engine_reachable"])
        self.assertFalse(report["backend_qualified"])
        self.assertIsNone(report["deployment_sha256"])
        self.assertEqual(len(report["launches"]), 15)
        self.assertIn(
            "memory_efficient_sliding_window_kv_eviction_and_its_window_only_dependency_proof",
            report["deferred_obligations"],
        )
        self.assertEqual(runtime.runtime_config()["device"], "cuda:0")
        self.assertIs(runtime.runtime_config()["dtype"], torch.bfloat16)

        medium_report = load_runtime(medium).report()
        self.assertEqual(medium_report["model_profile"], GEMMA3_12B_PROFILE)
        self.assertEqual(len(medium_report["launches"]), 15)
        medium_by_wrapper = {
            launch["wrapper"]: launch
            for launch in medium_report["launches"]
            if launch["wrapper"] not in {"linear", "gemma3_qk_norm"}
        }
        self.assertEqual(
            medium_by_wrapper["gemma3_scaled_embed"]["config"]["D"], 3840
        )
        self.assertEqual(
            medium_by_wrapper["gemma3_rms_norm"]["config"]["BLOCK_N"], 4096
        )
        self.assertEqual(
            medium_by_wrapper["paged_attention"]["config"]["D_HEAD"], 256
        )

        wide_report = load_runtime(wide).report()
        self.assertEqual(wide_report["model_profile"], GEMMA3_27B_PROFILE)
        self.assertEqual(len(wide_report["launches"]), 15)
        by_wrapper = {
            launch["wrapper"]: launch
            for launch in wide_report["launches"]
            if launch["wrapper"] not in {"linear", "gemma3_qk_norm"}
        }
        self.assertEqual(
            by_wrapper["gemma3_scaled_embed"]["config"]["D"], 5376
        )
        self.assertEqual(
            by_wrapper["gemma3_rms_norm"]["config"]["BLOCK_N"], 8192
        )
        self.assertEqual(
            by_wrapper["paged_attention"]["config"]["D_HEAD"], 128
        )

    def test_qualified_capability_rejects_unsealed_staged_runtime(self) -> None:
        runtime = FakeRuntime(
            {
                "engine_reachable": False,
                "backend_qualified": False,
                "qualification": None,
            }
        )
        with self.assertRaisesRegex(ValueError, "sealed backend bundle"):
            QualifiedRuntime(runtime)

    def test_qualified_capability_revalidates_identity_before_primitive_use(self) -> None:
        qualification = {
            "candidate_sha256": "a" * 64,
            "qualification_report_sha256": "b" * 64,
            "deployment_sha256": "c" * 64,
        }
        runtime = FakeRuntime(
            {
                "engine_reachable": False,
                "backend_qualified": True,
                "qualification": qualification,
            }
        )
        capability = QualifiedRuntime(runtime)
        self.assertEqual(capability.runtime_config()["hidden_size"], 2560)

        runtime.current_report = {
            **runtime.current_report,
            "qualification": {**qualification, "deployment_sha256": "e" * 64},
        }
        with self.assertRaisesRegex(RuntimeError, "identity changed"):
            capability.runtime_config()
        for method, args in (
            ("scaled_embed", ("ids", "weight")),
            ("rms_norm", ("x", "weight", "input_norm")),
            ("qk_norm", ("q", "k", "qw", "kw", FULL_ATTENTION)),
            ("rotary_embed", ("q", "k", "positions", FULL_ATTENTION)),
            ("paged_attention", ("q", "kc", "vc", "bt", "cuq", "cuk", 1, 2, FULL_ATTENTION)),
            ("add", ("x", "y", "attention_residual_add")),
            ("gelu_tanh_mul", ("gate", "up")),
        ):
            with self.subTest(method=method), self.assertRaisesRegex(RuntimeError, "identity changed"):
                getattr(capability, method)(*args)

    def test_qualified_capability_exposes_neutral_primitive_protocol(self) -> None:
        qualification = {
            "candidate_sha256": "a" * 64,
            "qualification_report_sha256": "b" * 64,
            "deployment_sha256": "c" * 64,
        }
        runtime = FakeRuntime(
            {
                "engine_reachable": False,
                "backend_qualified": True,
                "qualification": qualification,
            }
        )
        capability = QualifiedRuntime(runtime)

        self.assertEqual(capability.runtime_config()["hidden_size"], 2560)
        for kind in (SLIDING_ATTENTION, FULL_ATTENTION):
            self.assertEqual(capability.qk_norm("q", "k", "qw", "kw", kind),
                             ("normalized_q", "normalized_k"))
        self.assertEqual(runtime.qk_norm_calls, [("q", "k", "qw", "kw")] * 2)
        with self.assertRaisesRegex(ValueError, "attention kind"):
            capability.qk_norm("q", "k", "qw", "kw", "unknown_attention")
        self.assertEqual(len(runtime.qk_norm_calls), 2)
        self.assertEqual(
            capability.verified_for("tensor"), ("verified", "tensor")
        )
        self.assertEqual(
            capability.static_launch_config("linear", {"k": 2, "n": 1}),
            ("linear", (("k", 2), ("n", 1))),
        )
        self.assertEqual(
            capability.kernel_entrypoint("kernels", "linear"),
            ("kernels", "linear"),
        )
        self.assertEqual(
            capability.paged_attention(
                "q", "k_cache", "v_cache", "block_table",
                "cu_q", "cu_k", 1, 7, FULL_ATTENTION,
            ),
            "qualified_attention",
        )
        _, _, _, step, kwargs = runtime.paged_attention_calls[-1]
        self.assertEqual(step.block_table, "block_table")
        self.assertEqual(step.cu_seqlens_q, "cu_q")
        self.assertEqual(step.cu_seqlens_k, "cu_k")
        self.assertEqual(step.max_seqlen_q, 1)
        self.assertEqual(step.max_seqlen_k, 7)
        self.assertEqual(kwargs["attention_kind"], FULL_ATTENTION)
        self.assertIs(kwargs["value_checks"], False)

    def test_qualified_loader_is_the_only_capability_constructor(self) -> None:
        qualification = {
            "candidate_sha256": "a" * 64,
            "qualification_report_sha256": "b" * 64,
            "deployment_sha256": "c" * 64,
        }
        runtime = FakeRuntime(
            {
                "engine_reachable": False,
                "backend_qualified": True,
                "qualification": qualification,
            }
        )
        config = config_for_profile(GEMMA3_4B_PROFILE)
        with mock.patch(
            "vosti_kernels.model_families.gemma3.runtime.load_runtime",
            return_value=runtime,
        ) as staged_loader:
            capability = load_qualified_runtime(
                config,
                deployment_bundle="bundle",
                model_config_sha256="f" * 64,
                kernel_root="kernels",
                framework_root="framework",
                device="cuda:7",
                dtype=torch.bfloat16,
                environment={"backend": "cuda"},
            )
        self.assertIsInstance(capability, QualifiedRuntime)
        staged_loader.assert_called_once_with(
            config,
            kernel_root="kernels",
            deployment_bundle="bundle",
            framework_root="framework",
            model_config_sha256="f" * 64,
            device="cuda:7",
            dtype=torch.bfloat16,
            environment={"backend": "cuda"},
        )

    def test_runtime_rejects_unqualified_device_or_dtype(self) -> None:
        config = config_for_profile(GEMMA3_4B_PROFILE)
        with self.assertRaisesRegex(ValueError, "CUDA device"):
            load_runtime(config, device="cpu")
        with self.assertRaisesRegex(ValueError, "bfloat16"):
            load_runtime(config, dtype=torch.float32)

    def test_config_geometry_and_attention_schedule_drift_fail_closed(self) -> None:
        for field, value in (
            ("head_dim", 128),
            ("sliding_window", 2048),
        ):
            with self.subTest(field=field):
                config = config_for_profile(GEMMA3_4B_PROFILE)
                config[field] = value
                with self.assertRaisesRegex(ValueError, field):
                    _validate_model_config(config)

        config = config_for_profile(GEMMA3_4B_PROFILE)
        config["layer_types"][0] = FULL_ATTENTION
        with self.assertRaisesRegex(ValueError, "layer_types"):
            _validate_model_config(config)

    def test_profile_registry_requires_explicit_selection_when_ambiguous(self) -> None:
        original = gemma3_scope.scope()
        alternate = copy.deepcopy(original["model_profiles"][0])
        alternate["model"]["name"] = "synthetic-gemma3-profile"
        alternate["model"]["max_position_embeddings"] = 65536
        expanded = copy.deepcopy(original)
        expanded["model_profiles"].append(alternate)

        with mock.patch.object(gemma3_scope, "_SCOPE", expanded):
            with self.assertRaisesRegex(ValueError, "profile name is required"):
                config_for_profile()
            selected = config_for_profile("synthetic-gemma3-profile")
            self.assertEqual(selected["max_position_embeddings"], 65536)
            self.assertEqual(
                gemma3_scope.model_profile_for_config(selected)["model"]["name"],
                "synthetic-gemma3-profile",
            )

    def test_profile_registry_schema_drift_fails_closed(self) -> None:
        original = gemma3_scope.scope()

        duplicate_config = copy.deepcopy(original)
        duplicate = copy.deepcopy(duplicate_config["model_profiles"][0])
        duplicate["model"]["name"] = "duplicate-config-profile"
        duplicate_config["model_profiles"].append(duplicate)
        with mock.patch.object(gemma3_scope, "_SCOPE", duplicate_config):
            with self.assertRaisesRegex(RuntimeError, "duplicate model configs"):
                gemma3_scope._validate_scope()

        incomplete_launches = copy.deepcopy(original)
        incomplete_launches["model_profiles"][0]["launches"].pop()
        with mock.patch.object(gemma3_scope, "_SCOPE", incomplete_launches):
            with self.assertRaisesRegex(RuntimeError, "cover every wrapper"):
                gemma3_scope._validate_scope()

    def test_rope_tables_distinguish_local_and_global_conventions(self) -> None:
        runtime = load_runtime(
            config_for_profile(GEMMA3_4B_PROFILE)
        )
        positions = torch.tensor([0, 8], dtype=torch.int32)
        local_cos, local_sin = runtime._rope_tables(
            positions,
            attention_kind=SLIDING_ATTENTION,
            dtype=torch.float32,
        )
        global_cos, global_sin = runtime._rope_tables(
            positions,
            attention_kind=FULL_ATTENTION,
            dtype=torch.float32,
        )

        self.assertTrue(torch.equal(local_cos[0], global_cos[0]))
        self.assertTrue(torch.equal(local_sin[0], global_sin[0]))
        self.assertFalse(torch.equal(local_cos[1], global_cos[1]))
        self.assertFalse(torch.equal(local_sin[1], global_sin[1]))


if __name__ == "__main__":
    unittest.main()
