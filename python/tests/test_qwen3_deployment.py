import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from scripts.deployment.model_families import qwen3 as qwen3_deployment_bundle
from scripts.deployment.model_families.qwen3_scope import selected_launch_configs
from scripts.deployment.common import _selected_proof_cases
from deployment_contract_cases import proof_digest_fixture, check_required_effect_receipts
from vosti_kernels import primitive_runtime
from vosti_kernels.backend_evidence import (
    BACKEND_REQUIREMENTS_SCHEMA,
    EVIDENCE_KIND,
    digest,
    required_probe_contexts,
    required_probe_ids,
)
from vosti_kernels.model_families.qwen3 import runtime as qwen3_runtime
from vosti_kernels.deployment import candidate_digest
from vosti_kernels.model_families.qwen3.deployment import (
    validate_candidate,
    CANDIDATE_SCHEMA,
    DEPLOYMENT_SCHEMA,
    REPORT_SCHEMA,
    bind_launches_to_proof_cases,
    load_bundle,
    scope,
    scope_sha256,
    seal_candidate,
    validate_runtime_binding,
)
from vosti_kernels.model_families.qwen3.profile import (
    model_config,
    model_profile_for_config,
    model_profile_for_name,
    model_shape,
)


class FakeRuntime:
    def __init__(self, report):
        self.current_report = report

    def report(self):
        return self.current_report

    def binding_identity(self):
        return self.current_report

    def model_config(self):
        return {"model_type": "qwen3"}

    def runtime_config(self):
        return {"hidden_size": 1024, "num_heads": 16}

    def verified_for(self, tensor):
        return ("verified", tensor)

    def static_launch_config(self, wrapper, key):
        return (wrapper, tuple(sorted(key.items())))

    def kernel_entrypoint(self, kernels, wrapper):
        return (kernels, wrapper)


class Qwen3DeploymentTests(unittest.TestCase):
    def test_each_mutation_case_requires_exact_effect_evidence(self):
        check_required_effect_receipts(self, scope(), self._candidate, validate_candidate)

    def _profile(self):
        return model_profile_for_name("qwen3-0.6b")

    def _model_config(self):
        return model_config(model_profile_for_name("qwen3-8b"))

    def _proof_cases(self, profile):
        cases = []
        requirement_body = {
            "sites": ["grid.body[0]"],
            "operation": "triton.zeros",
            "probe": "constructor",
            "properties": ["all_lanes_are_positive_zero"],
            "consumers": ["regional"],
            "inputs": [],
            "output": {
                "element": "abstract_float",
                "shape": ["1"],
                "shape_is_concrete": True,
            },
            "attributes": {},
        }
        requirement = {"id": digest(requirement_body)[:20], **requirement_body}
        for contract, specialization in _selected_proof_cases(scope(), profile["launches"]):
            identity = {
                "source": contract["source"],
                "source_sha256": contract["source_sha256"],
                "kernel": contract["kernel"],
                "specialization": specialization,
            }
            identity_digest = digest(identity)
            manifest_body = {
                "schema": BACKEND_REQUIREMENTS_SCHEMA,
                "kernel": {
                    "source": identity["source"],
                    "source_sha256": identity["source_sha256"],
                    "name": identity["kernel"],
                    "specialization": identity["specialization"],
                },
                "physical_float_dtypes": ["bfloat16", "float32"],
                "site_count": 1,
                "requirements": [requirement],
            }
            structural, semantic = proof_digest_fixture(contract, specialization)
            cases.append(
                {
                    "case_id": identity_digest[:20],
                    **identity,
                    "structural_contract_digest": structural,
                    "structural_check_count": 1,
                    "semantic_proof": semantic,
                    "backend_requirements": {
                        **manifest_body,
                        "manifest_sha256": digest(manifest_body),
                    },
                }
            )
        return cases

    def _candidate(self):
        profile = self._profile()
        proof_cases = self._proof_cases(profile)
        deployment_scope = scope()
        body = {
            "schema": CANDIDATE_SCHEMA,
            "qualified": False,
            "engine_reachable": False,
            "architecture": "qwen3",
            "model": {
                "catalog_name": profile["model"]["name"],
                "config_sha256": "1" * 64,
                "directory_name": "test-model",
                "resolved_config": model_config(profile),
            },
            "environment": {
                "backend": "cuda",
                "devices": [
                    {
                        "name": "test",
                        "compute_capability": "8.6",
                        "total_memory_bytes": 1,
                    }
                ],
            },
            "runtime": {
                "dtype": "bfloat16",
                "device_type": "cuda",
            },
            "selection_policy": {
                "batch_dimensions_used": [],
                "model_dimensions_used": True,
                "hardware_profile_observed": True,
                "architecture_scope": "qwen3",
            },
            "provenance": {"base_revision": "f" * 40},
            "scope_sha256": scope_sha256(),
            "launches": bind_launches_to_proof_cases(
                proof_cases,
                profile_name=profile["model"]["name"],
            ),
            "kernel_cases": proof_cases,
        }
        return {**body, "candidate_sha256": candidate_digest(body)}

    def _report(self, candidate):
        return {
            "schema": REPORT_SCHEMA,
            "candidate_sha256": candidate["candidate_sha256"],
            "environment": candidate["environment"],
            "tested_device_index": 0,
            "evidence_kind": EVIDENCE_KIND,
            "results": [
                {
                    "requirement_id": requirement_id,
                    "passed": True,
                    "details": "ok",
                    "launch_meta_configs": required_probe_contexts(candidate)[
                        requirement_id
                    ],
                }
                for requirement_id in sorted(required_probe_ids(candidate))
            ],
        }

    @staticmethod
    def _redigest(candidate):
        candidate["candidate_sha256"] = candidate_digest(candidate)

    def test_model_profiles_materialize_the_pinned_dense_policy(self):
        for name in ("qwen3-0.6b", "qwen3-8b"):
            profile = model_profile_for_name(name)
            shape = model_shape(profile)
            derived = selected_launch_configs(
                shape,
                ((profile["model"]["vocab_size"], shape["hidden"]),),
            )
            self.assertEqual(profile["launches"], derived)
            self.assertTrue(all("batch" not in launch["key"] for launch in derived))

    def test_exact_config_selection_fails_closed(self):
        profile = model_profile_for_config(self._model_config())
        self.assertEqual(model_shape(profile)["name"], "qwen3-8b")
        changed = self._model_config()
        changed["hidden_size"] += 1
        with self.assertRaisesRegex(ValueError, "exact model profiles"):
            model_profile_for_config(changed)

    def test_candidate_checkpoint_projection_uses_exact_profile(self):
        config = model_config(self._profile())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            raw = json.dumps(config).encode()
            (path / "config.json").write_bytes(raw)
            resolved, profile, config_sha256 = (
                qwen3_deployment_bundle.CANDIDATE_ARCHITECTURE.checkpoint_profile(
                    path
                )
            )
        self.assertEqual(resolved, config)
        self.assertEqual(profile["model"]["name"], "qwen3-0.6b")
        self.assertEqual(config_sha256, hashlib.sha256(raw).hexdigest())

    def test_seal_preserves_non_engine_status(self):
        candidate = self._candidate()
        deployment = seal_candidate(candidate, self._report(candidate))
        self.assertEqual(deployment["schema"], DEPLOYMENT_SCHEMA)
        self.assertTrue(deployment["qualified"])
        self.assertFalse(deployment["engine_reachable"])
        self.assertNotIn("provenance", deployment)
        self.assertEqual(deployment["scope_sha256"], scope_sha256())

    def test_candidate_requires_the_exact_profile_launches(self):
        candidate = self._candidate()
        report = self._report(candidate)
        candidate["launches"].pop()
        self._redigest(candidate)
        report["candidate_sha256"] = candidate["candidate_sha256"]
        with self.assertRaisesRegex(ValueError, "closed scope"):
            seal_candidate(candidate, report)

    def test_candidate_binds_scope_and_validates_provenance_shape(self):
        mutations = {
            "provenance": ("provenance", {"base_revision": 123}, "provenance metadata"),
            "scope": ("scope_sha256", "0" * 64, "scope digest"),
        }
        for name, (field, value, message) in mutations.items():
            with self.subTest(name=name):
                candidate = self._candidate()
                candidate[field] = value
                self._redigest(candidate)
                with self.assertRaisesRegex(ValueError, message):
                    seal_candidate(candidate, self._report(candidate))

    def test_candidate_rejects_source_or_backend_evidence_drift(self):
        candidate = self._candidate()
        case = candidate["kernel_cases"][0]
        old_case_id = case["case_id"]
        case["source_sha256"] = "0" * 64
        identity = {
            field: case[field]
            for field in ("source", "source_sha256", "kernel", "specialization")
        }
        case["case_id"] = digest(identity)[:20]
        for launch in candidate["launches"]:
            if launch["kernel_case_id"] == old_case_id:
                launch["kernel_case_id"] = case["case_id"]
        self._redigest(candidate)
        with self.assertRaisesRegex(ValueError, "source digest differs"):
            seal_candidate(candidate, self._report(candidate))

        candidate = self._candidate()
        report = self._report(candidate)
        report["results"].pop()
        with self.assertRaisesRegex(ValueError, "coverage differs"):
            seal_candidate(candidate, report)

    def test_runtime_loader_reconstructs_all_bundle_records(self):
        candidate = self._candidate()
        report = self._report(candidate)
        deployment = seal_candidate(candidate, report)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for filename, value in (
                ("deployment-candidate.json", candidate),
                ("backend-qualification-report.json", report),
                ("deployment.json", deployment),
            ):
                (root / filename).write_text(json.dumps(value), encoding="utf-8")
            self.assertEqual(load_bundle(root)["deployment"], deployment)
            deployment["engine_reachable"] = True
            (root / "deployment.json").write_text(
                json.dumps(deployment), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "differs"):
                load_bundle(root)

    def test_runtime_binding_rejects_every_bound_identity_drift(self):
        candidate = self._candidate()
        report = self._report(candidate)
        bundle = {
            "candidate": candidate,
            "report": report,
            "deployment": seal_candidate(candidate, report),
        }
        arguments = {
            "resolved_config": model_config(self._profile()),
            "model_config_sha256": "1" * 64,
            "environment": candidate["environment"],
        }
        self.assertEqual(
            validate_runtime_binding(bundle, **arguments), bundle["deployment"]
        )
        for name, changed, message in (
            ("model_config_sha256", "2" * 64, "config digest"),
            ("environment", {"backend": "other"}, "environment"),
        ):
            with self.subTest(name=name):
                with self.assertRaisesRegex(ValueError, message):
                    validate_runtime_binding(
                        bundle,
                        **{**arguments, name: changed},
                    )

    def test_runtime_installs_one_immutable_complete_launch_plan(self):
        candidate = self._candidate()
        deployment = seal_candidate(candidate, self._report(candidate))
        bundle = {"candidate": candidate, "deployment": deployment}
        kernel_digests = {
            f"triton_kernels.{contract['module']}": contract["source_sha256"]
            for contract in scope()["kernel_contracts"]
        }

        bad_bundle = copy.deepcopy(bundle)
        bad_bundle["candidate"]["kernel_cases"][0]["source_sha256"] = "0" * 64
        with self.assertRaisesRegex(RuntimeError, "proof source differs"):
            qwen3_runtime._FAMILY._install_static_launch_plan(
                kernel_digests, bad_bundle, self._profile()
            )

        incomplete = copy.deepcopy(bundle)
        incomplete["deployment"]["launches"].pop()
        with self.assertRaisesRegex(RuntimeError, "exact model profile"):
            qwen3_runtime._FAMILY._install_static_launch_plan(
                kernel_digests, incomplete, self._profile()
            )

        _, plan = qwen3_runtime._FAMILY._install_static_launch_plan(
            kernel_digests, bundle, self._profile()
        )
        runtime = qwen3_runtime.runtime_for_tests(
            hidden_size=1024,
            num_heads=16,
            num_kv_heads=8,
            head_dim=128,
            intermediate_size=3072,
            rms_norm_eps=1e-6,
        )
        with mock.patch.object(runtime, "_static_launch_plan", plan):
            launch = runtime.static_launch_config(
                "linear", {"n": 1024, "k": 1024}
            )
            self.assertEqual(launch["BLOCK_M"], 16)
            with self.assertRaises(TypeError):
                launch["BLOCK_M"] = 32

    def test_qualified_capability_is_passed_explicitly(self):
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
        capability = qwen3_runtime.QualifiedRuntime(runtime)
        self.assertEqual(
            primitive_runtime.runtime_config(capability)["hidden_size"],
            1024,
        )

    def test_qualified_capability_revalidates_explicit_identity(self):
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
        capability = qwen3_runtime.QualifiedRuntime(runtime)
        self.assertEqual(capability.runtime_config()["hidden_size"], 1024)
        runtime.current_report = {
            **runtime.current_report,
            "qualification": {**qualification, "deployment_sha256": "e" * 64},
        }
        with self.assertRaisesRegex(RuntimeError, "identity changed"):
            capability.runtime_config()


if __name__ == "__main__":
    unittest.main()
