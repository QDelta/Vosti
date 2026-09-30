import json
from pathlib import Path
import tempfile
import unittest

from vosti_kernels.backend_evidence import (
    BACKEND_REQUIREMENTS_SCHEMA,
    EVIDENCE_KIND,
    digest,
    required_probe_contexts,
)
from vosti_kernels.deployment import CANDIDATE_SCHEMA, DEPLOYMENT_SCHEMA, REPORT_SCHEMA, candidate_digest
from vosti_kernels.kernel_interfaces import (
    load_attention_interfaces, load_rectangular_interface, RECTANGULAR_KERNEL_INTERFACES,
    rectangular_implementation_records,
    load_mutation_interface,
)
from scripts.deployment.common import _selected_proof_cases


def proof_digest_fixture(contract, specialization):
    """Use installed interface identities; backend evidence below is synthetic.

    This is a deployment-schema fixture, not a new qualification run. In
    particular it must not bypass the real generated-interface admission gate.
    """
    rectangular = RECTANGULAR_KERNEL_INTERFACES.get((contract["source"], contract["kernel"]))
    if contract["evidence"] == "exact_effect_certificate":
        manifest = load_mutation_interface()
        matches = [c for c in manifest["inventory_contributions"]
                   if c["source"] == contract["source"] and c["kernel"] == contract["kernel"]
                   and c["constants"] == specialization]
        assert len(matches) == 1
        implementations = [i for entry in manifest["interfaces"] for i in entry["raw"]["implementations"]
                           if i["execution_identity"] == matches[0]["execution_identity"]]
        assert len(implementations) == 1
        goal, = implementations[0]["standalone_contracts"]
        return matches[0]["structural_contract_digest"], {
            "kind": "exact_effect", "contract_digest": goal["raw_contract_digest"], "check_count": 1,
        }
    if contract["evidence"] != "conditional_relational_certificate" and rectangular is None:
        return "a" * 64, None
    manifest = load_rectangular_interface(rectangular) if rectangular else load_attention_interfaces()
    matches = [c for c in manifest["inventory_contributions"]
               if c["source"] == contract["source"] and c["kernel"] == contract["kernel"]
               and c["constants"] == specialization]
    assert len(matches) == 1, (contract["wrapper"], specialization)
    records = (rectangular_implementation_records(manifest) if rectangular else
               [i for interface in manifest["interfaces"] for i in interface["raw"]["implementations"]])
    implementations = [i for i in records
                       if i["execution_identity"] == matches[0]["execution_identity"]]
    assert len(implementations) == 1
    goals = {c["symbol_prefix"]: c["raw_contract_digest"]
             for c in implementations[0]["standalone_contracts"]}
    if rectangular:
        return goals["raw_batch_invariance"], None
    return goals["raw_batch_invariance"], {
        "kind": "conditional_selected_row",
        "contract_digest": goals["raw_selected_row_prefix_equivalence"],
        "check_count": 1,
    }


def _requirement_manifest(source, source_sha256, kernel, specialization):
    requirement_body = {
        "sites": ["synthetic.qualification.site"],
        "operation": "triton.program_id",
        "probe": "control",
        "properties": ["synthetic schema fixture"],
        "consumers": [],
        "inputs": [],
        "output": None,
        "attributes": {"axis": 0},
    }
    requirement = {
        "id": digest(requirement_body)[:20],
        **requirement_body,
    }
    body = {
        "schema": BACKEND_REQUIREMENTS_SCHEMA,
        "kernel": {
            "source": source,
            "source_sha256": source_sha256,
            "name": kernel,
            "specialization": specialization,
        },
        "physical_float_dtypes": ["bfloat16", "float32"],
        "site_count": 1,
        "requirements": [requirement],
    }
    return {**body, "manifest_sha256": digest(body)}


def candidate_fixture(deployment, runtime, profile_name):
    deployment_scope = deployment.scope()
    profile = deployment.ARCHITECTURE.model_profile_for_name(profile_name)
    cases = []
    for contract, specialization in _selected_proof_cases(deployment_scope, profile["launches"]):
        identity = {
            "source": contract["source"],
            "source_sha256": contract["source_sha256"],
            "kernel": contract["kernel"],
            "specialization": specialization,
        }
        structural, semantic = proof_digest_fixture(contract, specialization)
        cases.append(
            {
                "case_id": digest(identity)[:20],
                **identity,
                "structural_contract_digest": structural,
                "structural_check_count": 1,
                "semantic_proof": semantic,
                "backend_requirements": _requirement_manifest(
                    contract["source"],
                    contract["source_sha256"],
                    contract["kernel"],
                    specialization,
                ),
            }
        )
    launches = deployment.bind_launches_to_proof_cases(
        cases, profile_name=profile_name
    )
    environment = {
        "backend": "cuda",
        "devices": [{"name": "fixture GPU"}],
        "torch_version": "fixture",
        "triton_version": "fixture",
        "cuda_runtime": "fixture",
        "driver_version": "fixture",
    }
    body = {
        "schema": CANDIDATE_SCHEMA,
        "qualified": False,
        "engine_reachable": False,
        "architecture": deployment_scope["architecture"],
        "model": {
            "catalog_name": profile["model"]["name"],
            "directory_name": profile_name,
            "config_sha256": "c" * 64,
            "resolved_config": json.loads(json.dumps(
                runtime.config_for_profile(profile_name)
            )),
        },
        "environment": environment,
        "runtime": {
            "dtype": "bfloat16",
            "device_type": "cuda",
        },
        "selection_policy": {
            "batch_dimensions_used": [],
            "model_dimensions_used": True,
            "hardware_profile_observed": True,
            "architecture_scope": deployment_scope["architecture"],
        },
        "provenance": {"base_revision": "d" * 40},
        "scope_sha256": deployment.scope_sha256(),
        "launches": launches,
        "kernel_cases": cases,
    }
    return {**body, "candidate_sha256": candidate_digest(body)}


def report_fixture(candidate):
    contexts = required_probe_contexts(candidate)
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
                "details": "passed synthetic schema fixture",
                "launch_meta_configs": launch_meta,
            }
            for requirement_id, launch_meta in contexts.items()
        ],
    }


def check_required_effect_receipts(self, scope, candidate_factory, validate):
    kernels = {c["kernel"] for c in scope["kernel_contracts"]
               if c["evidence"] == "exact_effect_certificate"}
    checked = set()
    for index, case in enumerate(candidate_factory()["kernel_cases"]):
        if case["kernel"] not in kernels:
            continue
        checked.add(case["kernel"])
        for kind in (None, "conditional_selected_row", "batch_invariance"):
            candidate = candidate_factory()
            effect = candidate["kernel_cases"][index]["semantic_proof"]
            candidate["kernel_cases"][index]["semantic_proof"] = None if kind is None else {**effect, "kind": kind}
            candidate["candidate_sha256"] = candidate_digest(candidate)
            with self.subTest(kernel=case["kernel"], wrong_kind=kind):
                with self.assertRaisesRegex(ValueError, "semantic certificate"):
                    validate(candidate)
        for field in ("effect", "structural"):
            candidate = candidate_factory()
            changed = candidate["kernel_cases"][index]
            if field == "effect":
                changed["semantic_proof"]["contract_digest"] = "0" * 64
            else:
                changed["structural_contract_digest"] = "0" * 64
            candidate["candidate_sha256"] = candidate_digest(candidate)
            with self.subTest(kernel=case["kernel"], wrong_digest=field):
                with self.assertRaisesRegex(ValueError, "mutation .* differs"):
                    validate(candidate)
    self.assertEqual(checked, kernels)


class DeploymentContractCases:
    """Shared deployment fail-closed cases; family tests supply static bindings."""

    def candidate(self):
        return candidate_fixture(self.deployment, self.runtime, self.profile_name)

    def test_seals_exact_qualified_but_non_engine_deployment(self):
        candidate = self.candidate()
        self.deployment.validate_candidate(candidate)
        deployment = self.deployment.seal_candidate(candidate, report_fixture(candidate))

        self.assertEqual(deployment["schema"], DEPLOYMENT_SCHEMA)
        self.assertTrue(deployment["qualified"])
        self.assertFalse(deployment["engine_reachable"])
        self.assertEqual(deployment["architecture"], self.deployment.scope()["architecture"])
        self.assertEqual(len(deployment["launches"]), self.expected_launch_count)

    def test_candidate_scope_source_and_launch_drift_fail_closed(self):
        mutations = []
        candidate = self.candidate()
        candidate["engine_reachable"] = True
        mutations.append((candidate, "outside the engine"))
        candidate = self.candidate()
        candidate["scope_sha256"] = "0" * 64
        mutations.append((candidate, "scope digest"))
        candidate = self.candidate()
        candidate["kernel_cases"][0]["source_sha256"] = "0" * 64
        mutations.append((candidate, "identity"))
        candidate = self.candidate()
        candidate["launches"] = candidate["launches"][:-1]
        mutations.append((candidate, "launch plan"))
        candidate = self.candidate()
        candidate["model"]["catalog_name"] = "unknown-profile"
        mutations.append((candidate, "unknown model profile"))

        for malformed, message in mutations:
            malformed["candidate_sha256"] = candidate_digest(malformed)
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    self.deployment.validate_candidate(malformed)

    def test_report_coverage_and_engine_status_cannot_be_forged(self):
        candidate = self.candidate()
        report = report_fixture(candidate)
        report["results"] = []
        with self.assertRaisesRegex(ValueError, "coverage"):
            self.deployment.seal_candidate(candidate, report)

        candidate = self.candidate()
        report = report_fixture(candidate)
        report["results"][0]["passed"] = False
        with self.assertRaisesRegex(ValueError, "qualification failed"):
            self.deployment.seal_candidate(candidate, report)

    def test_each_attention_case_requires_selected_row_evidence(self):
        attention = {
            contract["kernel"] for contract in self.deployment.scope()["kernel_contracts"]
            if contract["wrapper"] in {"paged_attention", "paged_attention_swa"}
        }
        checked = set()
        for index, case in enumerate(self.candidate()["kernel_cases"]):
            if case["kernel"] not in attention:
                continue
            checked.add(case["kernel"])
            with self.subTest(kernel=case["kernel"]):
                candidate = self.candidate()
                candidate["kernel_cases"][index]["semantic_proof"] = None
                candidate["candidate_sha256"] = candidate_digest(candidate)
                with self.assertRaisesRegex(ValueError, "semantic certificate"):
                    self.deployment.validate_candidate(candidate)
        self.assertEqual(checked, attention)

    def test_each_mutation_case_requires_exact_effect_evidence(self):
        check_required_effect_receipts(self, self.deployment.scope(), self.candidate,
                                       self.deployment.validate_candidate)

    def test_bundle_round_trip_reconstructs_the_seal(self):
        candidate = self.candidate()
        report = report_fixture(candidate)
        deployment = self.deployment.seal_candidate(candidate, report)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name, value in (
                ("deployment-candidate.json", candidate),
                ("backend-qualification-report.json", report),
                ("deployment.json", deployment),
            ):
                (root / name).write_text(json.dumps(value), encoding="utf-8")
            loaded = self.deployment.load_bundle(root)
            selected_config = self.runtime.config_from_bundle(root)
        self.assertEqual(loaded["deployment"], deployment)
        self.assertEqual(
            selected_config, self.runtime.config_for_profile(self.profile_name)
        )

    def test_provenance_does_not_change_the_seal_or_runtime_admission(self):
        candidate = self.candidate()
        report = report_fixture(candidate)
        sealed = self.deployment.seal_candidate(candidate, report)
        for revision in (None, "a-different-base-revision"):
            candidate["provenance"] = {"base_revision": revision}
            self.assertEqual(candidate_digest(candidate), candidate["candidate_sha256"])
            self.assertEqual(self.deployment.seal_candidate(candidate, report), sealed)
            bundle = {"candidate": candidate, "report": report, "deployment": sealed}
            self.assertEqual(self.deployment.validate_runtime_binding(
                bundle, resolved_config=self.runtime.config_for_profile(self.profile_name),
                model_config_sha256="c" * 64, environment=candidate["environment"],
            ), sealed)

    def test_runtime_binding_requires_exact_model_source_and_environment(self):
        candidate = self.candidate()
        report = report_fixture(candidate)
        bundle = {
            "candidate": candidate,
            "report": report,
            "deployment": self.deployment.seal_candidate(candidate, report),
        }
        deployment = self.deployment.validate_runtime_binding(
            bundle,
            resolved_config=self.runtime.config_for_profile(self.profile_name),
            model_config_sha256="c" * 64,
            environment=candidate["environment"],
        )
        self.assertFalse(deployment["engine_reachable"])

        arguments = {
            "resolved_config": self.runtime.config_for_profile(self.profile_name),
            "model_config_sha256": "0" * 64,
            "environment": candidate["environment"],
        }
        for field, value, message in (
            ("model_config_sha256", "0" * 64, "config digest"),
            ("environment", {"backend": "cuda"}, "environment"),
        ):
            with self.subTest(field=field):
                changed = dict(arguments)
                changed["model_config_sha256"] = "c" * 64
                changed[field] = value
                with self.assertRaisesRegex(ValueError, message):
                    self.deployment.validate_runtime_binding(bundle, **changed)
