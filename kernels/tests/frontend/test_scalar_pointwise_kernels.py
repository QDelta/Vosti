"""Neutral pointwise contracts needed by Gemma-4, independently of its model proof."""

from pathlib import Path
import hashlib

import pytest

from diagnostics.race import prove_logical_write_disjointness_from_annotations
from ir.relational_dataflow import prove_relational_dataflow_from_annotations
from ir.proof_preparation import prepare_annotation_proof
from backend.probe_contract import validate_requirement_probe_coverage
from ir.backend_requirements import build_backend_requirement_manifest


@pytest.mark.parametrize("name", ("scale", "softcap"))
def test_scalar_pointwise_batch_relation_and_write_disjointness(name):
    source = (Path(__file__).resolve().parents[2] / "triton_kernels" / f"{name}.py").read_text()
    kernel = f"{name}_kernel"
    constants = {"BLOCK_M": 1, "BLOCK_N": 1024}
    relational = prove_relational_dataflow_from_annotations(source, kernel, constants, goal_name="batch_invariance")
    assert relational.proved, relational.unsupported_reason
    race = prove_logical_write_disjointness_from_annotations(source, kernel, constants,
        goal_name="batch_invariance", timeout_ms=5000)
    assert race.ok, [(c.name, c.details) for c in race.checks if not c.proved]


@pytest.mark.parametrize("name", ("scale", "softcap"))
def test_scalar_pointwise_backend_requirements_have_probe_rules(name):
    source = (Path(__file__).resolve().parents[2] / "triton_kernels" / f"{name}.py").read_text()
    constants = {"BLOCK_M": 1, "BLOCK_N": 1024}
    prepared = prepare_annotation_proof(source, f"{name}_kernel", constants, goal_name="batch_invariance")
    manifest = build_backend_requirement_manifest(kernel=prepared.kernel,
        source_name=f"{name}.py", source_sha256=hashlib.sha256(source.encode()).hexdigest(),
        constants=constants, physical_float_dtypes=("bfloat16", "float32"))
    assert manifest["requirements"]
    for requirement in manifest["requirements"]:
        validate_requirement_probe_coverage(requirement)
