"""Independent receipt, source, import and checked-consumer closure checks."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from scripts.verification.kernel_interface_registry import load_registry
from scripts.audit.kernel_interface_audit import (
    attention_inventory_from_source, canonical_digest, local_checked_consumers,
    validate_attention_interface,
)
from scripts.audit.claim_ledger import validate_all, validate_consumers


@pytest.fixture(scope="module")
def inputs():
    qualification, registry = load_registry()
    descriptor = registry["attention"]
    manifest = json.loads((ROOT / descriptor["manifest"]).read_text())
    return descriptor, manifest, attention_inventory_from_source(qualification, ROOT)


def check(inputs, manifest=None, *, root=ROOT, pinned=None):
    descriptor, original, cases = inputs
    return validate_attention_interface("attention", descriptor, original if manifest is None else manifest,
                                        canonical_digest(original) if pinned is None else pinned, cases, root)


def test_current_raw_interface_and_claim_ledger_are_closed(inputs):
    records = check(inputs)
    ledger = validate_all()
    assert records == {k: v for k, v in ledger["imports"].items() if v["interface"] == "attention"}
    assert len(records) == 10
    assert sum(len(r["implementations"]) for r in records.values()) == 14
    assert all(len(r["consumers"]) == 1 for r in records.values())
    assert len({r["module"] for r in records.values()}) == 5
    assert all(p["scope"] == "claim_reachable" for p in ledger["proofs"] if p["class"] == "kernel_certificate")


def test_independent_inventory_matches_actual_producer(inputs):
    from scripts.verification.verify_attention_interfaces import selected_cases
    def identity(cases):
        return [(c["source"], c["kernel"], constants, sorted(families)) for c, constants, families in cases]
    assert identity(inputs[2]) == identity(selected_cases())


@pytest.mark.parametrize("mutation", ["wrong_import", "redefinition", "nonliteral", "duplicate"])
def test_attention_inventory_rejects_broken_constant_link(tmp_path, mutation):
    directory = tmp_path / "kernels/triton_kernels"
    directory.mkdir(parents=True)
    source = (ROOT / "kernels/triton_kernels/fattn_paged.py").read_text()
    constants = (ROOT / "kernels/triton_kernels/constants.py").read_text()
    if mutation == "wrong_import":
        source = source.replace("from triton_kernels.constants import PAGE_SIZE",
                                "from unrelated.constants import PAGE_SIZE")
    elif mutation == "redefinition":
        source += "\nPAGE_SIZE = 32\n"
    elif mutation == "nonliteral":
        constants = constants.replace("PAGE_SIZE = 64", "PAGE_SIZE = int('64')")
    else:
        constants += "\nPAGE_SIZE = 32\n"
    (directory / "fattn_paged.py").write_text(source)
    (directory / "constants.py").write_text(constants)
    with pytest.raises(ValueError):
        attention_inventory_from_source({}, tmp_path)


def test_claim_gate_does_not_import_verifier_or_gpu_packages():
    result = subprocess.run([sys.executable, "-c", '''
import sys
sys.path.insert(0, ".")
from scripts.audit.claim_ledger import validate_all
validate_all()
assert not {"z3", "torch", "triton", "ir.relational_verifier"}.intersection(sys.modules)
'''], cwd=ROOT, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr


def test_unreviewed_receipt_mutation_is_rejected(inputs):
    manifest = deepcopy(inputs[1])
    manifest["interfaces"][0]["raw"]["implementations"][0]["standalone_contracts"][0]["raw_contract_digest"] = "0" * 64
    with pytest.raises(ValueError, match="reviewed interface receipt differs"):
        check(inputs, manifest)


@pytest.mark.parametrize("mutation", ["missing_contribution", "duplicate_execution", "missing_implementation",
    "duplicate_implementation", "missing_goal", "cross_configuration_meaning", "wrong_geometry",
    "raw_body", "logical_body", "adapter_body", "condition_drift"])
def test_structural_mutations_fail_even_if_a_new_receipt_pin_is_supplied(inputs, mutation):
    manifest = deepcopy(inputs[1])
    raw = manifest["interfaces"][0]["raw"]
    if mutation == "missing_contribution":
        manifest["inventory_contributions"].pop()
    elif mutation == "duplicate_execution":
        manifest["inventory_contributions"][1]["execution_identity"] = manifest["inventory_contributions"][0]["execution_identity"]
    elif mutation == "missing_implementation":
        raw["implementations"].pop()
    elif mutation == "duplicate_implementation":
        raw["implementations"].append(deepcopy(raw["implementations"][0]))
    elif mutation == "missing_goal":
        raw["implementations"][0]["standalone_contracts"].pop()
    elif mutation == "cross_configuration_meaning":
        raw["interpretation"] = "outputs_equal_across_all_implementations"
    elif mutation == "wrong_geometry":
        manifest["interfaces"][0]["head_dim"] = 64
    elif mutation in {"raw_body", "logical_body"}:
        raw["generated_body_sha256" if mutation == "raw_body" else "logical_body_sha256"] = "0" * 64
    elif mutation == "adapter_body":
        manifest["interfaces"][0]["checked_adapter_sha256"] = "0" * 64
    elif mutation == "condition_drift":
        raw["implementations"][0]["standalone_contracts"][1]["analyzer_conditions"][0]["label"] = "different premise"
    with pytest.raises(ValueError, match="invalid kernel interface audit"):
        check(inputs, manifest, pinned=canonical_digest(manifest))


def test_source_body_mutation_is_not_blessed_by_receipt_pin(inputs, tmp_path):
    descriptor = inputs[0]
    path = tmp_path / descriptor["generated"]
    path.parent.mkdir(parents=True)
    path.write_text((ROOT / descriptor["generated"]).read_text().replace("unreachable!()", "assume(false);", 1))
    with pytest.raises(ValueError, match="generated catalog body differs"):
        check(inputs, root=tmp_path)


def test_consumer_calls_in_comments_cannot_count_as_checked_calls(tmp_path):
    path = tmp_path / "fixture.rs"
    path.write_text('''pub mod raw {
#[verifier::external_body]
pub proof fn imported() {}
pub proof fn consumer() {
    // imported();
    let text = "imported()";
}
}
''')
    assert local_checked_consumers(path, {"fixture::raw::imported"}) == {"fixture::raw::imported": []}


def test_raw_consumer_cannot_be_external(tmp_path):
    path = tmp_path / "fixture.rs"
    path.write_text('''pub mod raw {
#[verifier::external_body]
pub proof fn imported() {}
#[verifier::external_body]
pub proof fn consumer() {
    imported();
}
}
''')
    with pytest.raises(ValueError, match="unchecked item"):
        local_checked_consumers(path, {"fixture::raw::imported"})


@pytest.mark.parametrize("mutation", ["omit", "wrong_function", "wrong_interface", "qualification_only"])
def test_claim_consumer_declarations_match_actual_calls(inputs, mutation):
    surface = json.loads((ROOT / "audit/claim_surface.json").read_text())
    if mutation == "omit":
        surface["certificate_consumers"].pop()
    elif mutation == "wrong_function":
        surface["certificate_consumers"][0]["consumers"] = ["proof::unrelated"]
    elif mutation == "wrong_interface":
        surface["certificate_consumers"][0]["interface"] = "another"
    else:
        surface["certificate_consumers"][0]["role"] = "qualification_only"
    with pytest.raises(ValueError):
        validate_consumers(surface, check(inputs), {"attention": inputs[0]})
