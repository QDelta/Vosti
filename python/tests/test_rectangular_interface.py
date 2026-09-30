"""Typed row interfaces need no handwritten model/operation bindings."""

from functools import cache
import os
from pathlib import Path
import re
import shutil
import subprocess

import pytest

from spec_test_support import page_constants_source

from ir.relational_verifier import verify_annotations
from scripts.verification.kernel_qualification_inventory import selected_kernel_cases
from scripts.verification.rectangular_interface_codegen import render_rectangular_interface

ROOT = Path(__file__).resolve().parents[2]
VERUS = os.environ.get("VERUS") or shutil.which("verus")


def row_cases():
    cases = {}
    for contract, constants, _ in selected_kernel_cases():
        if contract["evidence"] in {
            "conditional_relational_certificate", "exact_effect_certificate"
        }:
            continue
        cases.setdefault(contract["kernel"], (contract, constants))
    return sorted(cases.items())


CASES = row_cases()


@cache
def interface(kernel):
    contract, constants = dict(CASES)[kernel]
    source = (ROOT / "kernels/triton_kernels" / contract["source"]).read_text()
    report = verify_annotations(source, kernel, constants, goal_name="batch_invariance")
    assert report.proved and report.verified_contract is not None
    return render_rectangular_interface(((report.verified_contract,),))


@pytest.mark.parametrize("kernel", [name for name, _ in CASES])
def test_every_current_row_kernel_infers_a_binding(kernel):
    rendered = interface(kernel)
    assert "row_projection_launch_equivalence" in rendered.adapter_body
    assert "external_body" not in rendered.adapter_body
    assert "assume(" not in rendered.adapter_body
    assert rendered.body.count("#[verifier::external_body]") == 1
    assert "BLOCK_M" not in rendered.body and "BLOCK_K" not in rendered.body


def verus_module(body):
    return f'''#![allow(non_snake_case)]
use vstd::prelude::*;
pub mod fixture_types {{
    use vstd::prelude::*;
    pub type Tensor2D = Seq<Seq<super::Scalar>>;
    pub type Tensor3D = Seq<Tensor2D>;
    pub type Tensor4D = Seq<Tensor3D>;
}}
#[path = "{ROOT / 'src/proof/tensor/shape.rs'}"]
mod TS;
verus! {{
#[verifier::external_body]
#[verifier::ext_equal]
pub struct Scalar {{ _private: () }}
pub type Tensor1D = Seq<Scalar>;
pub type Tensor2D = Seq<Tensor1D>;
pub type IntTensor1D = Seq<int>;
pub uninterp spec fn generated_kernel_allocation_cell() -> Scalar;
{body}
}}

mod proof {{
pub mod model {{ pub mod types {{ pub use crate::fixture_types::*; }} }}
pub mod tensor {{
pub mod types {{ pub use crate::fixture_types::*; }}

}}
}}

mod types {{
    pub use crate::fixture_types::*;
    use vstd::prelude::*;
    verus! {{ {page_constants_source()} }}
}}

fn main() {{}}
'''


@pytest.mark.skipif(VERUS is None, reason="set VERUS to check inferred rectangular adapters")
@pytest.mark.parametrize("kernel", [name for name, _ in CASES])
def test_verus_checks_every_inferred_row_adapter(tmp_path, kernel):
    path = tmp_path / "inferred_row.rs"
    path.write_text(verus_module(interface(kernel).body))
    result = subprocess.run([VERUS, str(path)], capture_output=True, text=True, timeout=120)
    assert result.returncode == 0 and "0 errors" in result.stdout, result.stdout + result.stderr


@pytest.mark.skipif(VERUS is None, reason="set VERUS for negative proof regression")
@pytest.mark.parametrize("mutation", ["missing_domain", "missing_raw_theorem"])
def test_inferred_adapter_does_not_hide_missing_premises(tmp_path, mutation):
    rendered = interface("qkv_matmul_kernel" if mutation == "missing_domain" else "matmul_kernel")
    body = rendered.adapter_body
    if mutation == "missing_domain":
        old = next(line + "\n" for line in body.splitlines()
                   if line.startswith("        row_projection_domain("))
    else:
        old = "    raw_batch_invariance_certificate(left, right, free);\n"
    assert old in body
    body = body.replace(old, "", 1)
    path = tmp_path / "weakened_row.rs"
    path.write_text(verus_module(rendered.raw.body + body))
    result = subprocess.run([VERUS, str(path)], capture_output=True, text=True, timeout=120)
    assert result.returncode != 0
    assert re.search(r"(assertion failed|precondition not satisfied|postcondition not satisfied)",
                     result.stderr), result.stdout + result.stderr


def test_static_matmul_alternatives_share_theory_not_execution():
    source = (ROOT / "kernels/triton_kernels/matmul.py").read_text()
    contracts = tuple((verify_annotations(source, "matmul_kernel",
        dict(BLOCK_M=m, BLOCK_N=n, BLOCK_K=k), goal_name="batch_invariance").verified_contract,)
        for m, n, k in ((16, 64, 64), (32, 64, 64), (16, 32, 256)))
    rendered = render_rectangular_interface(contracts)
    assert len({r.execution_identity for r in rendered.raw.implementations}) == 3
    assert rendered == render_rectangular_interface(tuple(reversed(contracts)))
    assert rendered.body.count("#[verifier::external_body]") == 1


def test_unknown_kernel_name_needs_no_new_binding_table():
    source = (ROOT / "kernels/triton_kernels/silu_mul.py").read_text()
    source = source.replace("silu_mul_kernel", "example_row_operation")
    raw = verify_annotations(source, "example_row_operation",
        dict(BLOCK_M=1, BLOCK_N=1024), goal_name="batch_invariance").verified_contract
    assert "row_projection_mapped_repr" in render_rectangular_interface(((raw,),)).adapter_body


@pytest.mark.skipif(VERUS is None, reason="set VERUS to check a new source domain")
def test_new_source_domain_needs_no_handwritten_policy(tmp_path):
    source = (ROOT / "kernels/triton_kernels/matmul.py").read_text()
    source = source.replace("#     right(M) == 1,", "#     right(M) == 1,\n#     N > 7,")
    raw = verify_annotations(source, "matmul_kernel",
        dict(BLOCK_M=16, BLOCK_N=64, BLOCK_K=64), goal_name="batch_invariance").verified_contract
    rendered = render_rectangular_interface(((raw,),))
    assert "row_projection_domain" in rendered.adapter_body
    assert re.search(r"left.N\)? > \(?7", rendered.raw.body)
    path = tmp_path / "new_domain.rs"
    path.write_text(verus_module(rendered.body))
    result = subprocess.run([VERUS, str(path)], capture_output=True, text=True, timeout=120)
    assert result.returncode == 0 and "0 errors" in result.stdout, result.stdout + result.stderr
