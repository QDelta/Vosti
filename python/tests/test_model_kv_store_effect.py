import unittest

from scripts.checks.kv_store_effect import (
    FAMILIES,
    _normalize_config,
    _validate_runtime_report,
)


class _ConfigObject:
    def as_runtime_dict(self):
        return {"head_dim": 256}


def _report(architecture: str) -> dict:
    digest = "a" * 64
    return {
        "architecture": architecture,
        "engine_reachable": False,
        "backend_qualified": True,
        "deployment_sha256": digest,
        "qualification": {
            "candidate_sha256": digest,
            "qualification_report_sha256": digest,
            "deployment_sha256": digest,
        },
        "module_origins": {"triton_kernels.store_kv_cache": "/kernel.py"},
        "module_source_sha256": {
            "triton_kernels.store_kv_cache": digest,
        },
    }


class ModelKvStoreEffectTests(unittest.TestCase):
    def test_registry_covers_all_supported_families(self) -> None:
        self.assertEqual(set(FAMILIES), {"qwen3", "gemma3", "gemma4", "llama3"})
        self.assertEqual(
            {family["architecture"] for family in FAMILIES.values()},
            {"qwen3", "gemma3_text", "gemma4_text", "llama3"},
        )

    def test_normalizes_mapping_and_object_checkpoint_configs(self) -> None:
        self.assertEqual(_normalize_config({"head_dim": 128}), {"head_dim": 128})
        self.assertEqual(_normalize_config(_ConfigObject()), {"head_dim": 256})
        with self.assertRaisesRegex(TypeError, "no runtime config"):
            _normalize_config(object())

    def test_accepts_qualified_runtime_reports_for_every_family(self) -> None:
        for architecture in ("qwen3", "gemma3_text", "gemma4_text", "llama3"):
            with self.subTest(architecture=architecture):
                _validate_runtime_report(_report(architecture), architecture)

    def test_rejects_unqualified_or_unattested_runtime_reports(self) -> None:
        report = _report("qwen3")
        report["backend_qualified"] = False
        with self.assertRaisesRegex(RuntimeError, "not backend-qualified"):
            _validate_runtime_report(report, "qwen3")

        report = _report("gemma3_text")
        report["module_source_sha256"] = {}
        with self.assertRaisesRegex(RuntimeError, "origins and digests"):
            _validate_runtime_report(report, "gemma3_text")


if __name__ == "__main__":
    unittest.main()
