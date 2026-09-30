from pathlib import Path
from functools import partial
import importlib
import hashlib
import tempfile
import unittest
from unittest import mock

from scripts.deployment.common import (
    DeploymentCandidateArchitecture,
    prepare_deployment_candidate,
)
from vosti_kernels.backend_evidence import digest
from vosti_kernels.deployment import (
    DeploymentArchitecture,
    scope,
    scope_sha256,
    source_provenance,
)


def synthetic_architecture() -> DeploymentArchitecture:
    architecture_scope = {
        "status": "qualification_scope",
        "engine_reachable": False,
        "architecture": "synthetic_dense",
        "kernel_contracts": [],
    }
    profile = {"model": {"name": "synthetic"}, "launches": []}
    return DeploymentArchitecture(
        scope=architecture_scope,
        scope_sha256=digest(architecture_scope),
        model_config=lambda selected: {"name": selected["model"]["name"]},
        model_profile_for_config=lambda _config: profile,
        model_profile_for_name=lambda _name: profile,
    )


class DeploymentArchitectureTests(unittest.TestCase):
    def test_missing_git_only_omits_optional_provenance(self):
        with mock.patch("vosti_kernels.deployment.subprocess.run", side_effect=FileNotFoundError):
            self.assertEqual(source_provenance(Path.cwd()), {"base_revision": None})

    def test_build_tree_revision_is_not_part_of_any_family_deployment_scope(self):
        for family in ("qwen3", "llama3", "gemma3", "gemma4"):
            profile = importlib.import_module(f"vosti_kernels.model_families.{family}.profile")
            deployment = importlib.import_module(f"vosti_kernels.model_families.{family}.deployment")
            family_scope = profile.scope()
            architecture = DeploymentArchitecture(
                scope=family_scope, scope_sha256=digest(family_scope),
                model_config=profile.model_config,
                model_profile_for_config=profile.model_profile_for_config,
                model_profile_for_name=profile.model_profile_for_name,
            )
            with self.subTest(family=family):
                self.assertEqual(scope_sha256(architecture), deployment.scope_sha256())
                self.assertNotIn("kernel_commit", scope(architecture))

    def test_every_family_binds_the_same_deployment_operations(self) -> None:
        from vosti_kernels import deployment as generic

        operations = (
            "scope", "scope_sha256", "bind_launches_to_proof_cases",
            "validate_candidate", "seal_candidate", "load_bundle", "validate_runtime_binding",
        )
        for family in ("qwen3", "llama3", "gemma3", "gemma4"):
            adapter = importlib.import_module(
                f"vosti_kernels.model_families.{family}.deployment"
            )
            for operation in operations:
                with self.subTest(family=family, operation=operation):
                    binding = getattr(adapter, operation)
                    self.assertIsInstance(binding, partial)
                    self.assertIs(binding.func, getattr(generic, operation))
                    self.assertEqual(binding.args, (adapter.ARCHITECTURE,))
                    self.assertFalse(binding.keywords)

    def test_scope_is_detached_and_digest_bound(self) -> None:
        architecture = synthetic_architecture()
        detached = scope(architecture)
        detached["architecture"] = "mutated"
        self.assertEqual(
            scope(architecture)["architecture"],
            "synthetic_dense",
        )
        self.assertEqual(
            scope_sha256(architecture),
            digest(scope(architecture)),
        )

    def test_scope_must_remain_outside_engine(self) -> None:
        scope = {
            "status": "qualification_scope",
            "engine_reachable": True,
        }
        with self.assertRaisesRegex(ValueError, "outside the engine"):
            DeploymentArchitecture(
                scope=scope,
                scope_sha256=digest(scope),
                model_config=lambda profile: profile,
                model_profile_for_config=lambda config: config,
                model_profile_for_name=lambda name: {"name": name},
            )

    def test_architecture_binding_is_immutable(self) -> None:
        architecture = synthetic_architecture()
        with self.assertRaises(AttributeError):
            architecture.scope_sha256 = "0" * 64

    def test_semantic_proof_policy_defaults_from_kernel_evidence(self) -> None:
        architecture = synthetic_architecture()
        self.assertEqual(architecture.semantic_proof_kind({"evidence": "exact_effect_certificate"}), "exact_effect")
        self.assertEqual(
            architecture.semantic_proof_kind(
                {"evidence": "conditional_relational_certificate"}
            ),
            "conditional_selected_row",
        )
        self.assertIsNone(
            architecture.semantic_proof_kind(
                {
                    "evidence": (
                        "structural_batch_decomposition_not_numeric_correctness"
                    )
                }
            )
        )

    def test_scope_digest_is_checked_at_binding(self) -> None:
        scope = {
            "status": "qualification_scope",
            "engine_reachable": False,
        }
        with self.assertRaisesRegex(ValueError, "digest"):
            DeploymentArchitecture(
                scope=scope,
                scope_sha256="0" * 64,
                model_config=lambda profile: profile,
                model_profile_for_config=lambda config: config,
                model_profile_for_name=lambda name: {"name": name},
            )


class DeploymentCandidateArchitectureTests(unittest.TestCase):
    def test_proof_case_uses_only_the_unified_verifier(self):
        from scripts.deployment.common import KERNEL_ROOT, prepare_proof_case
        from ir.relational_dataflow import prove_prepared_relational_dataflow
        from ir.proof_preparation import prepare_annotation_proof

        contract = {"source": "add.py", "kernel": "add_kernel", "evidence": "regional_certificate",
                    "source_sha256": hashlib.sha256((KERNEL_ROOT / "triton_kernels/add.py").read_bytes()).hexdigest()}
        constants = {"BLOCK_M": 1, "BLOCK_N": 64}
        prepared = prepare_annotation_proof(
            (KERNEL_ROOT / "triton_kernels/add.py").read_text(), "add_kernel", constants,
        )
        unified = prove_prepared_relational_dataflow(prepared)
        with mock.patch("ir.relational_verifier.prove_prepared_relational_dataflow",
                        return_value=unified) as produce:
            case = prepare_proof_case(contract, constants, ("float32",))
        produce.assert_called_once()
        self.assertEqual(case["structural_contract_digest"], unified.verified_contract.digest)
        self.assertEqual(case["structural_check_count"], len(unified.checks))

    def test_common_compiler_assembles_family_neutral_candidate(self) -> None:
        deployment_scope = {
            "architecture": "synthetic_dense",
            "runtime": {"device_type": "cuda"},
            "kernel_contracts": [
                {
                    "wrapper": "linear",
                    "module": "matmul",
                    "source": "matmul.py",
                    "kernel": "matmul_kernel",
                }
            ],
        }
        profile = {
            "model": {"name": "synthetic-1b"},
            "launches": [
                {
                    "wrapper": "linear",
                    "config": {
                        "BLOCK_M": 16,
                        "BLOCK_N": 64,
                        "BLOCK_K": 64,
                        "num_warps": 4,
                        "num_stages": 3,
                    },
                }
            ],
        }
        validated = []
        architecture = DeploymentCandidateArchitecture(
            family_label="Synthetic",
            candidate_schema="synthetic.candidate.v1",
            scope=lambda: deployment_scope,
            scope_sha256=lambda: "b" * 64,
            checkpoint_profile=lambda _path: (
                {"hidden_size": 1024},
                profile,
                "c" * 64,
            ),
            launch_inventory=lambda _profile: [{
                "wrapper": "linear",
                "sites": ["linear"],
                "key": {"n": 1024, "k": 1024},
            }],
            bind_launches_to_proof_cases=lambda cases, *, profile_name, launches: [
                {
                    "profile": profile_name,
                    "case_id": cases[0]["case_id"],
                }
            ],
            validate_candidate=lambda candidate: validated.append(candidate),
        )
        proof_case = {
            "case_id": "case-1",
            "source": "matmul.py",
            "kernel": "matmul_kernel",
        }
        environment = {"backend": "cuda", "devices": [{"name": "fixture"}]}
        with tempfile.TemporaryDirectory() as directory, mock.patch(
            "scripts.deployment.common.validate_kernel_sources",
        ) as check_sources, mock.patch(
            "scripts.deployment.common.source_provenance",
            return_value={"base_revision": None},
        ), mock.patch(
            "scripts.deployment.common.prepare_proof_case",
            return_value=proof_case,
        ), mock.patch(
            "scripts.deployment.common.select_static_launches",
            return_value=profile["launches"],
        ):
            candidate = prepare_deployment_candidate(
                architecture,
                Path(directory),
                environment=environment,
            )

        self.assertEqual(candidate["schema"], "synthetic.candidate.v1")
        self.assertFalse(candidate["qualified"])
        self.assertFalse(candidate["engine_reachable"])
        self.assertEqual(candidate["model"]["catalog_name"], "synthetic-1b")
        self.assertEqual(
            candidate["model"]["resolved_config"],
            {"hidden_size": 1024},
        )
        self.assertEqual(candidate["provenance"], {"base_revision": None})
        self.assertEqual(candidate["scope_sha256"], "b" * 64)
        check_sources.assert_called_once()
        self.assertEqual(validated, [candidate])


if __name__ == "__main__":
    unittest.main()
