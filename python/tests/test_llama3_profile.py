import unittest

from vosti_kernels.model_families.llama3 import runtime as llama3_runtime
from vosti_kernels.model_families.llama3.profile import (
    model_config,
    model_profile_for_config,
    model_profile_for_name,
    model_profiles,
    model_shape,
    project_config,
    scope,
)


class Llama3ProfileTests(unittest.TestCase):
    def _config(self, name: str = "llama-3.1-8b") -> dict:
        return model_config(model_profile_for_name(name))

    def test_registry_admits_all_exact_llama_geometries(self) -> None:
        self.assertEqual(
            {profile["model"]["name"] for profile in model_profiles()},
            {"llama-3.1-8b", "llama-3.2-3b", "llama-3.3-70b"},
        )
        config = self._config("llama-3.3-70b")
        profile = model_profile_for_config(config)
        self.assertEqual(profile["model"]["name"], "llama-3.3-70b")
        self.assertEqual(
            model_shape(profile),
            {
                "name": "llama-3.3-70b",
                "hidden": 8192,
                "intermediate_half": 28672,
                "head_dim": 128,
                "num_heads": 64,
                "num_kv_heads": 8,
            },
        )

    def test_exact_profile_seals_complete_forward_configuration(self) -> None:
        config = self._config()
        self.assertEqual(
            model_profile_for_config(config)["model"]["name"],
            "llama-3.1-8b",
        )
        self.assertEqual(
            model_shape(model_profile_for_config(config)),
            {
                "name": "llama-3.1-8b",
                "hidden": 4096,
                "intermediate_half": 14336,
                "head_dim": 128,
                "num_heads": 32,
                "num_kv_heads": 8,
            },
        )
        self.assertEqual(config["rms_norm_eps"], 1e-5)
        self.assertEqual(config["rope_scaling_kind"], "llama3")
        self.assertEqual(config["attention_kind"], "full_attention")
        self.assertFalse(config["tie_word_embeddings"])
        self.assertEqual(project_config({**config, "ignored": "runtime"}), config)

    def test_every_forward_changing_field_fails_closed(self) -> None:
        mutations = {
            "hidden_size": 4097,
            "num_hidden_layers": 31,
            "rms_norm_eps": 1e-6,
            "rope_theta": 10_000.0,
            "rope_factor": 4.0,
            "attention_kind": "sliding_attention",
            "hidden_act": "gelu",
            "attention_bias": True,
            "mlp_bias": True,
            "pretraining_tp": 2,
            "tie_word_embeddings": True,
        }
        for field, value in mutations.items():
            with self.subTest(field=field):
                changed = self._config()
                changed[field] = value
                with self.assertRaisesRegex(ValueError, "exact model profiles"):
                    model_profile_for_config(changed)

        missing = self._config()
        missing.pop("rope_low_frequency_factor")
        with self.assertRaisesRegex(ValueError, "lacks required fields"):
            model_profile_for_config(missing)

    def test_launch_and_contract_inventories_exclude_qk_norm(self) -> None:
        contract_wrappers = [
            contract["wrapper"] for contract in scope()["kernel_contracts"]
        ]
        self.assertNotIn("qk_norm", contract_wrappers)
        for profile in model_profiles():
            with self.subTest(profile=profile["model"]["name"]):
                launch_wrappers = [
                    launch["wrapper"] for launch in profile["launches"]
                ]
                self.assertNotIn("qk_norm", launch_wrappers)
                self.assertEqual(len(profile["launches"]), 12)
                self.assertEqual(set(launch_wrappers), set(contract_wrappers))
                self.assertTrue(
                    all(
                        "batch" not in launch["key"]
                        for launch in profile["launches"]
                    )
                )

    def test_explicit_test_runtimes_are_isolated_and_immutable(self) -> None:
        left = llama3_runtime.runtime_for_tests(
            hidden_size=16,
            num_heads=2,
            num_kv_heads=1,
            head_dim=8,
            intermediate_size=24,
        )
        right = llama3_runtime.runtime_for_tests(
            hidden_size=32,
            num_heads=4,
            num_kv_heads=2,
            head_dim=8,
            intermediate_size=48,
        )
        self.assertEqual(left.runtime_config()["hidden_size"], 16)
        self.assertEqual(right.runtime_config()["hidden_size"], 32)
        self.assertEqual(left.runtime_config()["rope_scaling_kind"], "llama3")
        with self.assertRaises(TypeError):
            left.runtime_config()["hidden_size"] = 17
        self.assertFalse(left.report()["backend_qualified"])
        with self.assertRaisesRegex(ValueError, "sealed backend bundle"):
            llama3_runtime.QualifiedRuntime(left)


if __name__ == "__main__":
    unittest.main()
