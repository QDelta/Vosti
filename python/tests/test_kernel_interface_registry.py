"""Shared interfaces have single ownership and closed family inventories."""

from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from scripts.verification.kernel_interface_registry import load_registry, validate_registry, rectangular_operators_from_source


def ownership():
    return json.loads((ROOT / "audit/model_architecture_ownership.json").read_text())


def test_shared_registry_has_no_family_owner():
    inventories, interfaces = load_registry()
    assert set(inventories) == {"qwen3", "llama3", "gemma3", "gemma4"}
    assert set(interfaces) == {"attention", "linear", "qkv", "rms_norm", "residual_rms_norm", "offset_rms_norm",
                               "add", "silu_mul", "gelu_tanh_mul", "scale", "softcap", "embedding", "scaled_embedding",
                               "head_rms_norm", "offset_head_rms_norm", "rotary", "kv_store"}
    for entry in inventories.values():
        assert set(entry) == {"scope", "inventory"}
    assert {m["module"] for m in interfaces["attention"]["consumer_modules"]} >= {
        "boundary::attention_operator", "boundary::tensor_runtime",
        "boundary::four_norm_gated_primitives"}


@pytest.mark.parametrize("mutation", [
    "old_schema", "old_catalog", "missing_family", "extra_family", "wrong_scope", "unknown_inventory",
    "empty_interfaces", "unknown_field", "family_directory", "escape", "duplicate_generated",
    "duplicate_manifest", "missing_consumer", "self_consumer", "duplicate_consumer",
    "invalid_module", "reserved_module",
])
def test_invalid_registry_is_rejected(mutation):
    record = ownership()
    entry = record["kernel_interfaces"]["attention"]
    if mutation == "old_schema":
        record["schema_version"] = 8
    elif mutation == "old_catalog":
        record["certificate_imports"] = {}
    elif mutation == "missing_family":
        del record["kernel_qualification"]["qwen3"]
    elif mutation == "extra_family":
        record["kernel_qualification"]["unknown"] = {}
    elif mutation == "wrong_scope":
        record["kernel_qualification"]["qwen3"]["scope"] = record["kernel_qualification"]["gemma3"]["scope"]
    elif mutation == "unknown_inventory":
        record["kernel_qualification"]["qwen3"]["inventory"] = "anything"
    elif mutation == "empty_interfaces":
        record["kernel_interfaces"] = {}
    elif mutation == "unknown_field":
        entry["owner"] = "qwen3"
    elif mutation == "family_directory":
        entry["generated"] = "src/boundary/backend_certificates/model_families/attention.rs"
    elif mutation == "escape":
        entry["manifest"] = "audit/../../outside.json"
    elif mutation.startswith("duplicate_") and mutation != "duplicate_consumer":
        other = deepcopy(entry)
        other.update(generated="src/boundary/backend_certificates/other.rs",
                     manifest="audit/other.json", generator="scripts/other.py")
        field = mutation.removeprefix("duplicate_")
        other[field] = entry[field]
        record["kernel_interfaces"]["other"] = other
    elif mutation == "missing_consumer":
        entry["consumer_modules"] = []
    elif mutation == "self_consumer":
        entry["consumer_modules"][0]["path"] = entry["generated"]
    elif mutation == "duplicate_consumer":
        entry["consumer_modules"].append(entry["consumer_modules"][0])
    elif mutation == "invalid_module":
        entry["consumer_modules"][0]["module"] = "not/a/module"
    elif mutation == "reserved_module":
        entry["generated"] = "src/boundary/backend_certificates/support.rs"
    else:
        raise AssertionError(mutation)
    with pytest.raises(ValueError):
        validate_registry(record)


def test_new_family_needs_no_new_interface_owner():
    record = ownership()
    record["families"].append("example")
    record["kernel_qualification"]["example"] = {
        "scope": "python/vosti_kernels/model_families/example/scope.json",
        "inventory": "profile_catalog",
    }
    _, interfaces = validate_registry(record)
    assert interfaces == record["kernel_interfaces"]


def test_shared_generator_has_distinct_artifact_owners():
    _, interfaces = load_registry()
    assert interfaces["linear"]["generator"] == interfaces["qkv"]["generator"]
    assert interfaces["linear"]["generated"] != interfaces["qkv"]["generated"]


def test_auditor_reads_the_same_declarative_routing_without_execution():
    from vosti_kernels.kernel_interfaces import RECTANGULAR_KERNEL_INTERFACES
    assert rectangular_operators_from_source() == {
        name: source for source, name in RECTANGULAR_KERNEL_INTERFACES.items()}


@pytest.mark.parametrize("text", [
    "dict()", "{('a.py', 'a'): 'one', ('a.py', 'a'): 'two'}",
    "{('a.py', 'a'): 'same', ('b.py', 'b'): 'same'}", "{('../a.py', 'a'): 'one'}",
])
def test_ambiguous_or_nonliteral_routing_rejected(tmp_path, text):
    path = tmp_path / "python/vosti_kernels/kernel_interfaces.py"
    path.parent.mkdir(parents=True)
    path.write_text("RECTANGULAR_KERNEL_INTERFACES = " + text)
    with pytest.raises(ValueError):
        rectangular_operators_from_source(tmp_path)


def test_pre_admission_inventory_does_not_need_a_family_owned_interface():
    record = ownership()
    record["pre_admission_kernel_families"] = ["example"]
    record["kernel_qualification"]["example"] = {
        "scope": "python/vosti_kernels/model_families/example/scope.json",
        "inventory": "profile_catalog",
    }
    inventories, interfaces = validate_registry(record)
    assert "example" in inventories
    assert interfaces == record["kernel_interfaces"]
