"""Raw rectangular interfaces close actual deployment and proof receipts."""

from copy import deepcopy
import json
from pathlib import Path

import pytest

from scripts.verification.kernel_interface_registry import load_registry, rectangular_operators_from_source
from scripts.audit.kernel_interface_audit import (canonical_digest, rectangular_inventory_from_source,
                                    validate_rectangular_interface)
from vosti_kernels.kernel_interfaces import (load_rectangular_interface,
                                             rectangular_implementation_records,
                                             validate_rectangular_interface_case)

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(params=sorted(rectangular_operators_from_source()))
def interface(request):
    return request.param


def cases(manifest):
    by_execution = {i["execution_identity"]: i for i in rectangular_implementation_records(manifest)}
    return [dict(source=c["source"], kernel=c["kernel"], specialization=c["constants"],
                 structural_contract_digest=by_execution[c["execution_identity"]]["standalone_contracts"][0]["raw_contract_digest"])
            for c in manifest["inventory_contributions"]]


def audit(manifest, name, *, pin=None):
    inventories, registry = load_registry()
    expected = rectangular_inventory_from_source(inventories, ROOT, *rectangular_operators_from_source()[name])
    return validate_rectangular_interface(name, registry[name], manifest,
        pin or canonical_digest(manifest), expected, ROOT)


def test_all_static_implementations_have_checked_consumers_and_admit(interface):
    manifest = load_rectangular_interface(interface)
    assert len(manifest["inventory_contributions"]) == {
        "linear": 13, "qkv": 4, "rms_norm": 3, "residual_rms_norm": 3,
        "offset_rms_norm": 2, "add": 1, "silu_mul": 1, "gelu_tanh_mul": 1,
        "scale": 1, "softcap": 1, "embedding": 4, "scaled_embedding": 3,
        "head_rms_norm": 10, "offset_head_rms_norm": 5, "rotary": 3}[interface]
    for case in cases(manifest):
        validate_rectangular_interface_case(case, manifest)
    records = audit(manifest, interface)
    modules = [entry["module"] + "::" for entry in manifest["interfaces"]] if "interfaces" in manifest else [""]
    assert set(records) == {f"boundary::backend_certificates::{interface}::{module}raw_batch_invariance_certificate" for module in modules}
    assert all(record["consumers"] for record in records.values())
    from scripts.verification.verify_rectangular_interfaces import selected_cases
    inventories, _ = load_registry()
    independently_collected = rectangular_inventory_from_source(inventories, ROOT,
        *rectangular_operators_from_source()[interface])
    def identity(items):
        return [(c["source"], c["kernel"], constants, sorted(families)) for c, constants, families in items]
    assert identity(selected_cases(interface)) == identity(independently_collected)


@pytest.mark.parametrize("mutation", ["digest", "tile", "source", "duplicate", "missing_goal", "execution"])
def test_mismatched_deployment_proof_is_not_admitted(mutation, interface):
    manifest = load_rectangular_interface(interface)
    case = deepcopy(cases(manifest)[0])
    if mutation == "digest":
        case["structural_contract_digest"] = "0" * 64
    elif mutation == "tile":
        case["specialization"]["BLOCK_K"] = 123
    elif mutation == "source":
        case["source"] = "other.py"
    elif mutation == "duplicate":
        manifest["inventory_contributions"].append(deepcopy(manifest["inventory_contributions"][0]))
    else:
        identity = manifest["inventory_contributions"][0]["execution_identity"]
        record = next(i for i in rectangular_implementation_records(manifest) if i["execution_identity"] == identity)
        if mutation == "missing_goal":
            record["standalone_contracts"].clear()
        else:
            record["execution_identity"] = "0" * 64
    with pytest.raises(ValueError):
        validate_rectangular_interface_case(case, manifest)


@pytest.mark.parametrize("mutation", ["missing_case", "duplicate_execution", "missing_implementation", "missing_goal",
    "raw_body", "logical_body", "adapter_body", "interpretation"])
def test_repinning_does_not_hide_structural_receipt_drift(mutation, interface):
    manifest = load_rectangular_interface(interface)
    entry = manifest["interfaces"][0] if "interfaces" in manifest else manifest
    raw = entry["raw"]
    if mutation == "missing_case":
        manifest["inventory_contributions"].pop()
    elif mutation == "duplicate_execution":
        contributions = manifest["inventory_contributions"]
        if len(contributions) == 1:
            contributions.append(deepcopy(contributions[0]))
        else:
            contributions[1]["execution_identity"] = contributions[0]["execution_identity"]
    elif mutation == "missing_implementation":
        raw["implementations"].pop()
    elif mutation == "missing_goal":
        raw["implementations"][0]["standalone_contracts"].clear()
    elif mutation == "raw_body":
        raw["generated_body_sha256"] = "0" * 64
    elif mutation == "logical_body":
        raw["logical_body_sha256"] = "0" * 64
    elif mutation == "adapter_body":
        entry["checked_adapter_sha256"] = "0" * 64
    else:
        raw["interpretation"] = "all_implementations_have_equal_outputs"
    with pytest.raises(ValueError):
        audit(manifest, interface)


def test_new_digest_requires_explicit_review(interface):
    manifest = load_rectangular_interface(interface)
    pin = canonical_digest(manifest)
    rectangular_implementation_records(manifest)[0]["standalone_contracts"][0]["raw_contract_digest"] = "0" * 64
    with pytest.raises(ValueError, match="reviewed interface receipt differs"):
        audit(manifest, interface, pin=pin)


def test_unhandled_registered_generator_fails_before_qualification(monkeypatch):
    import scripts.verification.verify_kernel_contracts as driver
    qualification, registry = load_registry()
    registry["unhandled"] = {}
    monkeypatch.setattr(driver, "load_registry", lambda: (qualification, registry))
    monkeypatch.setattr(driver, "selected_kernel_cases", lambda _: [])
    with pytest.raises(ValueError, match="has no generator"):
        driver.verify()


def test_runtime_only_trusts_raw_execution_not_row_map(interface):
    if interface == "rotary":
        from scripts.audit.tcb import scan_rust_item_records
        for module in ("dense_layer_primitives", "four_norm_gated_primitives"):
            path = ROOT / f"src/boundary/{module}.rs"
            source = path.read_text()
            record = next(r for r in scan_rust_item_records(path) if r["name"] == "rotary_embed_raw")
            raw = "\n".join(source.splitlines()[record["line"] - 1:record["end_line"]])
            assert "#[verifier::external_body]\npub fn rotary_embed(" not in source
            assert "rotary_component_raw_output(" in raw and "rotary_component_layout(" in raw
            assert "rotary_embed_repr(" not in raw
        return
    if interface in {"head_rms_norm", "offset_head_rms_norm"}:
        modules = ["four_norm_gated_primitives"]
        if interface == "head_rms_norm":
            modules.append("tensor_runtime")
        from scripts.audit.tcb import scan_rust_item_records
        for module in modules:
            path = ROOT / f"src/boundary/{module}.rs"
            source = path.read_text()
            functions = ["qk_norm"] + (["value_norm"] if module == "four_norm_gated_primitives" and interface == "head_rms_norm" else [])
            for function in functions:
                record = next(r for r in scan_rust_item_records(path) if r["name"] == function + "_raw")
                raw = "\n".join(source.splitlines()[record["line"] - 1:record["end_line"]])
                assert f"#[verifier::external_body]\npub fn {function}(" not in source
                assert "raw_output(" in raw and "layout(" in raw
                assert f"{function}_repr(" not in raw
            assert "checked_head_norm_binding(" in source
        return
    if interface in {"embedding", "scaled_embedding"}:
        module, function, prefix = (("tensor_runtime", "embed", "plain") if interface == "embedding"
                                    else ("four_norm_gated_primitives", "scaled_embed", "scaled"))
        path = ROOT / f"src/boundary/{module}.rs"
        source = path.read_text()
        from scripts.audit.tcb import scan_rust_item_records
        record = next(r for r in scan_rust_item_records(path) if r["name"] == function + "_raw")
        raw = "\n".join(source.splitlines()[record["line"] - 1:record["end_line"]])
        assert f"#[verifier::external_body]\npub fn {function}(" not in source
        assert f"EMBED::{prefix}_raw_output" in raw
        assert ("EMBED::layout" if prefix == "plain" else "EMBED::scaled_layout") in raw
        assert f"{function}_repr(" not in raw and f"EMBED::{prefix}_output(" not in raw
        assert f"EMBED::checked_{prefix}_binding(" in source
        return
    if interface in {"add", "silu_mul", "gelu_tanh_mul", "scale", "softcap"}:
        module = "tensor_runtime" if interface == "silu_mul" else "four_norm_gated_primitives"
        function = "silu_and_mul" if interface == "silu_mul" else interface
        source = (ROOT / f"src/boundary/{module}.rs").read_text()
        from scripts.audit.tcb import scan_rust_item_records
        record = next(r for r in scan_rust_item_records(ROOT / f"src/boundary/{module}.rs")
                      if r["identity"] == f"boundary::{module}::{function}_raw")
        raw = "\n".join(source.splitlines()[record["line"] - 1:record["end_line"]])
        assert f"#[verifier::external_body]\npub fn {function}(" not in source
        assert f"PW::{interface}_raw_output" in raw
        assert "PW::layout" in raw or "PW::binary_layout" in raw
        assert f"PW::{interface}_output(" not in raw
        assert f"PW::checked_{interface}_binding(" in source
        return
    if interface == "offset_rms_norm":
        source = (ROOT / "src/boundary/four_norm_gated_primitives.rs").read_text()
        assert "#[verifier::external_body]\npub fn rms_norm(" not in source
        raw = source.split("fn rms_norm_raw(", 1)[1].split("pub fn add(", 1)[0]
        assert "NORM::layout" in raw and "norm_raw_output" in raw
        assert "norm_repr(" not in raw
        assert "checked_norm_binding(input_repr, weight_repr, policy)" in source
        assert "NORM::checked_offset_binding" in source and "NORM::checked_rms_binding" in source
        return
    source = (ROOT / "src/boundary/tensor_runtime.rs").read_text()
    function = {"linear": "linear", "qkv": "qkv_linear", "rms_norm": "rms_norm",
                "residual_rms_norm": "add_rms_norm"}[interface]
    if interface in {"rms_norm", "residual_rms_norm"}:
        prefix = "rms" if interface == "rms_norm" else "residual"
        raw_output = f"NORM::{prefix}_raw_output"
        mapped = f"NORM::{prefix}_output"
        layout = "NORM::layout" if prefix == "rms" else "NORM::residual_layout"
        binding = f"NORM::checked_{prefix}_binding"
    else:
        namespace = interface.upper()
        raw_output, mapped, layout, binding = (f"{namespace}::{name}" for name in (
            "raw_output", "output", "layout", "checked_runtime_binding"))
    assert f"#[verifier::external_body]\npub fn {function}(" not in source
    raw = source.split(f"fn {function}_raw(", 1)[1].split(
        f"// @kernel-bridge-end boundary::tensor_runtime::{function}", 1)[0]
    assert raw_output in raw and layout in raw
    assert f"{function}_repr(" not in raw and mapped + "(" not in raw
    assert binding + "(" in source
