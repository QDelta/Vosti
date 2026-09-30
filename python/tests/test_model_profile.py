"""Regression checks for shared profile validation and family-specific policy."""

from copy import deepcopy
import unittest
from unittest import mock

from vosti_kernels.model_profile import validate_profile_registry, validate_scope_metadata
from vosti_kernels.model_families.gemma3 import profile as gemma3
from vosti_kernels.model_families.gemma4 import profile as gemma4
from vosti_kernels.model_families.llama3 import profile as llama3
from vosti_kernels.model_families.qwen3 import profile as qwen3


class ProfileValidationTests(unittest.TestCase):
    # Gemma 4 validates canonical runtime configs through its checkpoint parser,
    # rather than these three families' explicit closed model-field schemas.
    closed_envelope_families = (qwen3, llama3, gemma3)

    def test_all_family_profiles_roundtrip(self):
        for family in (*self.closed_envelope_families, gemma4):
            for profile in family.model_profiles():
                with self.subTest(family=family.__name__, model=profile["model"]["name"]):
                    self.assertEqual(
                        family.model_profile_for_config(family.model_config(profile)),
                        profile,
                    )

    def test_closed_metadata_rejections_are_shared(self):
        for family in self.closed_envelope_families:
            original = family.scope()
            for field, value in (
                ("schema", "unknown"),
                ("status", "engine_ready"),
                ("engine_reachable", True),
                ("engine_reachable", 0),
                ("architecture", "another_family"),
                ("runtime", {"dtype": "torch.float32", "device_type": "cuda"}),
                ("extra_field", 1),
            ):
                changed = {**original, field: value}
                with self.subTest(family=family.__name__, field=field, value=value):
                    with mock.patch.object(family, "_SCOPE", changed):
                        with self.assertRaises(RuntimeError):
                            family._validate_scope()
            validate_scope_metadata(
                original, fields=set(original),
                architecture=original["architecture"], family_label="test",
            )

    def test_registry_keeps_family_model_validation(self):
        for family in self.closed_envelope_families:
            original = family.scope()
            changes = {
                "duplicate name": lambda profiles: profiles.append(deepcopy(profiles[0])),
                "duplicate config": lambda profiles: profiles.append({
                    **deepcopy(profiles[0]),
                    "model": {**profiles[0]["model"], "name": "different-name"},
                }),
                "invalid model": lambda profiles: profiles[0]["model"].update(hidden_size=0),
                "invalid profile": lambda profiles: profiles[0].update(extra_field=True),
                "missing launches": lambda profiles: profiles[0].update(launches=[]),
            }
            for description, mutate in changes.items():
                changed = deepcopy(original)
                mutate(changed["model_profiles"])
                with self.subTest(family=family.__name__, mutation=description):
                    with mock.patch.object(family, "_SCOPE", changed):
                        with self.assertRaises(RuntimeError):
                            family._validate_scope()
            before = deepcopy(original)
            with mock.patch.object(family, "_SCOPE", original):
                family._validate_scope()
            self.assertEqual(original, before)

    def test_registry_requires_nonempty_list(self):
        for profiles in (None, {}, [], ()):
            with self.subTest(profiles=profiles), self.assertRaises(RuntimeError):
                validate_profile_registry(
                    profiles, validate_model=lambda model: None,
                    profile_config=lambda profile: profile["model"], family_label="test",
                )

    def test_every_family_rejects_malformed_kernel_catalogs(self):
        for family in (*self.closed_envelope_families, gemma4):
            for fault in ("empty", "schema", "digest", "wrapper", "duplicate"):
                changed = family.scope()
                contracts = changed["kernel_contracts"]
                if fault == "empty":
                    contracts.clear()
                elif fault == "schema":
                    contracts[0]["extra_field"] = True
                elif fault == "digest":
                    contracts[0]["source_sha256"] = "G" * 64
                elif fault == "wrapper":
                    contracts[0]["wrapper"] = ""
                else:
                    contracts.append(deepcopy(contracts[0]))
                with self.subTest(family=family.__name__, fault=fault):
                    if family is gemma4:
                        with mock.patch.object(family, "load_scope", return_value=changed):
                            with self.assertRaises(ValueError):
                                family.scope()
                    else:
                        with mock.patch.object(family, "_SCOPE", changed):
                            with self.assertRaises(RuntimeError):
                                family._validate_scope()


if __name__ == "__main__":
    unittest.main()
