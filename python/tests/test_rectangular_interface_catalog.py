"""Geometry dispatch derives solely from typed contracts, not model tables."""

from functools import cache
import json
import os
from pathlib import Path
import shutil
import subprocess
from copy import deepcopy
from types import SimpleNamespace

import pytest

from spec_test_support import page_constants_source

from ir.relational_verifier import verify_annotations
from scripts.verification.kernel_qualification_inventory import selected_kernel_cases
from scripts.verification.rectangular_interface_catalog import render_rectangular_catalog
from scripts.verification.verify_rectangular_interfaces import OPERATORS, render_qualified_cases
from scripts.audit.kernel_interface_audit import canonical_digest, validate_rectangular_interface
from vosti_kernels.kernel_interfaces import validate_rectangular_interface_case

ROOT = Path(__file__).resolve().parents[2]
VERUS = os.environ.get("VERUS") or shutil.which("verus")
KERNELS = tuple(sorted({contract["kernel"] for contract, _, _ in selected_kernel_cases()
                       if contract["evidence"] not in {
                           "conditional_relational_certificate", "exact_effect_certificate"}}))


@cache
def contracts(kernel):
    result = []
    for contract, constants, _ in selected_kernel_cases():
        if contract["kernel"] == kernel:
            source = (ROOT / "kernels/triton_kernels" / contract["source"]).read_text()
            report = verify_annotations(source, kernel, constants, goal_name="batch_invariance")
            assert report.proved and report.verified_contract is not None
            result.append((report.verified_contract,))
    assert result
    return tuple(result)


@cache
def catalog(kernel):
    return render_rectangular_catalog(contracts(kernel))


@pytest.mark.parametrize("kernel", KERNELS)
def test_catalog_covers_every_qualified_geometry_and_retains_execution_identity(kernel):
    rendered = catalog(kernel)
    assert rendered == render_rectangular_catalog(tuple(reversed(contracts(kernel))))
    manifest = json.loads(rendered.manifest())
    implementations = [record for entry in manifest["interfaces"] for record in entry["raw"]["implementations"]]
    assert len(implementations) == len(contracts(kernel))
    assert len({record["execution_identity"] for record in implementations}) == len(implementations)
    assert all(not name.startswith("BLOCK_") for name in rendered.dimensions)
    assert "external_body" not in rendered.dispatch_body and "assume(" not in rendered.dispatch_body
    assert rendered.body.count("#[verifier::external_body]") == len(rendered.interfaces)


def fixture(path):
    return f'''use vstd::prelude::*;
#[path = "{ROOT / 'src/proof/tensor/seq_flatten.rs'}"] pub mod seq_flatten;
pub mod fixture_types {{
use vstd::prelude::*;
verus! {{
#[verifier::external_body]
#[verifier::ext_equal]
pub struct Scalar {{ _private: () }}
pub type Tensor1D = Seq<Scalar>;
pub type Tensor2D = Seq<Tensor1D>;
pub type Tensor3D = Seq<Tensor2D>;
pub type Tensor4D = Seq<Tensor3D>;
pub type IntTensor1D = Seq<int>;
}}
}}
pub mod proof {{
pub mod model {{ pub mod types {{ pub use crate::fixture_types::*; }} }}
pub mod tensor {{
pub mod types {{ pub use crate::fixture_types::*; }}
pub use crate::seq_flatten as seq_flatten;

#[path = "{ROOT / 'src/proof/tensor/shape.rs'}"] pub mod shape;

}}
}}
pub mod boundary {{
pub mod scalar {{ pub use crate::fixture_types::*; }}
pub mod backend_certificates {{
pub mod support {{
use vstd::prelude::*;
use crate::boundary::scalar::Scalar;
verus! {{ pub uninterp spec fn generated_kernel_allocation_cell() -> Scalar; }}
}}
}}
}}
#[path = "{path}"] pub mod catalog;

mod types {{
    pub use crate::fixture_types::*;
    use vstd::prelude::*;
    verus! {{ {page_constants_source()} }}
}}

fn main() {{}}
'''


def run_verus(tmp_path, body, *, succeeds):
    generated = tmp_path / "catalog.rs"
    generated.write_text(body)
    root = tmp_path / "check.rs"
    root.write_text(fixture(generated))
    result = subprocess.run([VERUS, "--cfg", "verus_only", str(root)],
                            capture_output=True, text=True, timeout=120)
    if succeeds:
        assert result.returncode == 0 and "0 errors" in result.stdout, result.stdout + result.stderr
    else:
        assert result.returncode != 0
        assert any(message in result.stderr for message in (
            "assertion failed", "precondition not satisfied", "postcondition not satisfied")), result.stderr


@pytest.mark.skipif(VERUS is None, reason="set VERUS for catalog proofs")
@pytest.mark.parametrize("kernel", KERNELS)
def test_verus_checks_complete_catalog_dispatch(tmp_path, kernel):
    run_verus(tmp_path, catalog(kernel).body, succeeds=True)


@pytest.mark.skipif(VERUS is None, reason="set VERUS for negative catalog proofs")
@pytest.mark.parametrize("mutation", ["wrong_geometry", "missing_domain", "wrong_output", "wrong_row_map"])
def test_catalog_dispatch_does_not_hide_a_wrong_binding(tmp_path, mutation):
    rendered = catalog("embedding_kernel" if mutation != "wrong_output" else "qkv_matmul_kernel")
    body = rendered.body
    if mutation == "wrong_geometry":
        first, second = rendered.interfaces[:2]
        start = body.index("pub open spec fn raw_output(")
        end = body.index("pub open spec fn mapped_output(", start)
        segment = body[start:end]
        assert first.module in segment
        body = body[:start] + segment.replace(first.module, second.module) + body[end:]
    elif mutation == "missing_domain":
        line = next(line for line in body.splitlines(True) if line.startswith("    requires launch_valid("))
        body = body.replace(line, "")
    elif mutation == "wrong_row_map":
        start = body.index("pub open spec fn row_mapped_output(")
        end = body.index("pub proof fn checked_row_map(", start)
        segment = body[start:end]
        assert "seq![input_ids[row]]" in segment
        body = body[:start] + segment.replace("seq![input_ids[row]]", "seq![input_ids[0]]") + body[end:]
    else:
        start = body.index("pub open spec fn raw_output(")
        end = body.index("pub open spec fn mapped_output(", start)
        segment = body[start:end]
        assert "row_projection_oq_repr" in segment
        body = body[:start] + segment.replace("row_projection_oq_repr", "row_projection_ok_repr") + body[end:]
    run_verus(tmp_path, body, succeeds=False)


def test_no_duplicate_implementation_or_mixed_source():
    a = contracts("embedding_kernel")
    with pytest.raises(ValueError, match="duplicate implementation"):
        render_rectangular_catalog(a + (a[0],))
    with pytest.raises(ValueError, match="different declared tensor interfaces|mixed kernel sources"):
        render_rectangular_catalog(a + contracts("scaled_embedding_kernel"))


@pytest.mark.skipif(VERUS is None, reason="set VERUS for unsupported geometry proof")
def test_unknown_geometry_has_no_output(tmp_path):
    rendered = catalog("embedding_kernel")
    assert rendered.dimensions == ("D", "V")
    body = rendered.body + '''
verus! {
proof fn unsupported(ids: IntTensor1D, weight: Tensor2D, D: nat, V: nat)
    requires !geometry_valid(D, V),
    ensures raw_output(ids, weight, D, V).is_none(),
        mapped_output(ids, weight, D, V).is_none(),
        row_mapped_output(ids, weight, D, V).is_none(),
        !launch_valid(ids, weight, D, V),
{}
}
'''
    run_verus(tmp_path, body, succeeds=True)


@cache
def export(kernel):
    selected = [case for case in selected_kernel_cases() if case[0]["kernel"] == kernel]
    return render_qualified_cases(tuple((*case, SimpleNamespace(contracts=raw))
                                        for case, raw in zip(selected, contracts(kernel))))


def audit_export(tmp_path, kernel, body, manifest):
    path = tmp_path / "src/boundary/backend_certificates/example.rs"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body)
    cases = [case for case in selected_kernel_cases() if case[0]["kernel"] == kernel]
    source = cases[0][0]["source"]
    directory = tmp_path / "kernels/triton_kernels"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / source).write_text((ROOT / "kernels/triton_kernels" / source).read_text())
    return validate_rectangular_interface("example", {"generated": path.relative_to(tmp_path).as_posix()},
                                          manifest, canonical_digest(manifest), cases, tmp_path)


@pytest.mark.parametrize("kernel", KERNELS)
def test_shared_producer_auditor_and_admission_support_both_catalog_shapes(tmp_path, kernel):
    body, serialized = export(kernel)
    manifest = json.loads(serialized)
    installed = next((name for name, (_, entrypoint) in OPERATORS.items() if entrypoint == kernel), None)
    if installed is not None:
        assert body == (ROOT / f"src/boundary/backend_certificates/{installed}.rs").read_text()
        assert serialized == (ROOT / f"audit/{installed}_kernel_interface.json").read_text()
    records = audit_export(tmp_path, kernel, body, manifest)
    assert records and all(record["consumers"] for record in records.values())
    by_execution = {i["execution_identity"]: i for entry in (
        manifest["interfaces"] if "interfaces" in manifest else [dict(raw=manifest["raw"])])
        for i in entry["raw"]["implementations"]}
    for contribution in manifest["inventory_contributions"]:
        case = dict(source=contribution["source"], kernel=kernel, specialization=contribution["constants"],
            structural_contract_digest=by_execution[contribution["execution_identity"]]["standalone_contracts"][0]["raw_contract_digest"])
        validate_rectangular_interface_case(case, manifest)
        case["structural_contract_digest"] = "0" * 64
        with pytest.raises(ValueError, match="proof differs"):
            validate_rectangular_interface_case(case, manifest)


@pytest.mark.parametrize("mutation", ["geometry", "missing_interface", "duplicate_interface", "missing_case",
    "raw_body", "adapter_body", "dispatch_body", "wrong_module", "dimension", "extra_trusted"])
def test_geometry_auditor_rejects_drift_even_with_repinning(tmp_path, mutation):
    body, serialized = export("embedding_kernel")
    manifest = json.loads(serialized)
    if mutation == "geometry":
        manifest["interfaces"][0]["geometry"]["D"] += 1
    elif mutation == "missing_interface":
        manifest["interfaces"].pop()
    elif mutation == "duplicate_interface":
        manifest["interfaces"].append(deepcopy(manifest["interfaces"][0]))
    elif mutation == "missing_case":
        manifest["inventory_contributions"].pop()
    elif mutation == "raw_body":
        manifest["interfaces"][0]["raw"]["generated_body_sha256"] = "0" * 64
    elif mutation == "adapter_body":
        manifest["interfaces"][0]["checked_adapter_sha256"] = "0" * 64
    elif mutation == "dispatch_body":
        manifest["checked_dispatch_sha256"] = "0" * 64
    elif mutation == "wrong_module":
        manifest["interfaces"][0]["module"] = "unrelated"
    elif mutation == "dimension":
        manifest["dimensions"].remove("D")
    else:
        import hashlib
        marker = "pub open spec fn geometry_valid("
        assert marker in body
        body = body.replace(marker, "#[verifier::external_body]\npub proof fn extra_assumption() {}\n" + marker)
        manifest["generated_body_sha256"] = hashlib.sha256(body.encode()).hexdigest()
        dispatch = body.split("\n// BEGIN CHECKED GEOMETRY DISPATCH\n")[1]
        manifest["checked_dispatch_sha256"] = hashlib.sha256(dispatch.encode()).hexdigest()
    with pytest.raises(ValueError):
        audit_export(tmp_path, "embedding_kernel", body, manifest)


def test_untyped_static_differences_do_not_become_dispatch_keys():
    source = (ROOT / "kernels/triton_kernels/silu_mul.py").read_text()
    source = source.replace("#     right(M) == 1,", "#     right(M) == 1,\n#     N > LIMIT,")
    source = source.replace("BLOCK_N: tl.constexpr,", "BLOCK_N: tl.constexpr,\n    LIMIT: tl.constexpr,")
    cases = []
    for limit in (7, 8):
        report = verify_annotations(source, "silu_mul_kernel", dict(BLOCK_M=1, BLOCK_N=1024, LIMIT=limit),
                                    goal_name="batch_invariance")
        assert report.proved and report.verified_contract is not None
        cases.append((report.verified_contract,))
    with pytest.raises(ValueError, match="different logical contracts"):
        render_rectangular_catalog(tuple(cases))


@pytest.mark.skipif(VERUS is None, reason="set VERUS for a new typed geometry")
def test_new_kernel_and_shape_symbol_need_no_dispatch_table(tmp_path):
    source = (ROOT / "kernels/triton_kernels/embedding.py").read_text()
    import re
    source = re.sub(r"\bD\b", "FEATURES", source).replace("embedding_kernel", "example_lookup")
    cases = []
    for width in (32, 96):
        report = verify_annotations(source, "example_lookup",
            dict(FEATURES=width, BLOCK_M=1, BLOCK_D=128), goal_name="batch_invariance")
        assert report.proved and report.verified_contract is not None
        cases.append((report.verified_contract,))
    rendered = render_rectangular_catalog(tuple(cases))
    assert rendered.dimensions == ("FEATURES", "V")
    assert [dict(entry.geometry) for entry in rendered.interfaces] == [{"FEATURES": 32}, {"FEATURES": 96}]
    run_verus(tmp_path, rendered.body, succeeds=True)


def test_deployment_loads_geometry_manifest_and_rejects_unknown_schema(tmp_path, monkeypatch):
    from vosti_kernels import kernel_interfaces as admission
    body, serialized = export("embedding_kernel")
    generated = tmp_path / "src/boundary/backend_certificates/example.rs"
    generated.parent.mkdir(parents=True)
    generated.write_text(body)
    receipt = tmp_path / "audit/example_kernel_interface.json"
    receipt.parent.mkdir()
    receipt.write_text(serialized)
    monkeypatch.setattr(admission, "__file__", str(tmp_path / "python/vosti_kernels/kernel_interfaces.py"))
    monkeypatch.setattr(admission, "RECTANGULAR_KERNEL_INTERFACES", {("embedding.py", "embedding_kernel"): "example"})
    assert admission.load_rectangular_interface("example") == json.loads(serialized)
    broken = json.loads(serialized)
    broken["schema_version"] = 999
    receipt.write_text(json.dumps(broken))
    with pytest.raises(ValueError, match="invalid rectangular interface manifest"):
        admission.load_rectangular_interface("example")


def test_inference_preserves_each_artifacts_own_execution_binding(monkeypatch):
    import scripts.verification.rectangular_interface_catalog as renderer
    original = renderer.inferred_binding
    def checked(axis, raw, fragment):
        assert fragment.raw_contract_digest == raw.digest
        return original(axis, raw, fragment)
    monkeypatch.setattr(renderer, "inferred_binding", checked)
    for order in (contracts("matmul_kernel"), tuple(reversed(contracts("matmul_kernel")))):
        render_rectangular_catalog(order)
