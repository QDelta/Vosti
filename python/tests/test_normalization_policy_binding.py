"""Check actual policy wiring against raw kernels, without numeric equivalence."""

import os
from pathlib import Path
import re
import shutil
import subprocess

import pytest

from spec_test_support import page_constants_source

from scripts.audit.tcb import scan_rust_item_records

ROOT = Path(__file__).resolve().parents[2]
VERUS = os.environ.get("VERUS") or shutil.which("verus")


def item(relative, name):
    path = ROOT / relative
    records = [r for r in scan_rust_item_records(path) if r["name"] == name]
    assert len(records) == 1, (relative, name)
    record = records[0]
    return "".join(path.read_text().splitlines(True)[record["line"] - 1:record["end_line"]])


@pytest.mark.skipif(VERUS is None, reason="set VERUS to check policy/raw binding")
@pytest.mark.parametrize("mutation", [None, "wrong_kernel", "offset_epsilon", "direct_epsilon", "missing_layout"])
def test_actual_normalization_policy_binding(tmp_path, mutation):
    relative = "src/boundary/four_norm_gated_primitives.rs"
    raw = item(relative, "norm_raw_output")
    if mutation == "wrong_kernel":
        assert "NORM::offset_raw_output" in raw
        raw = raw.replace("NORM::offset_raw_output", "NORM::rms_raw_output")
    elif mutation == "offset_epsilon":
        assert "unit_offset_norm_epsilon_repr()" in raw
        raw = raw.replace("unit_offset_norm_epsilon_repr()", "FloatParameterBits { bits: 0 }")
    elif mutation == "direct_epsilon":
        assert "float_parameter_scalar_repr(epsilon)" in raw
        raw = raw.replace("float_parameter_scalar_repr(epsilon)",
                          "float_parameter_scalar_repr(unit_offset_norm_epsilon_repr())")
    proof = item(relative, "checked_norm_binding")
    if mutation == "missing_layout":
        assert "    requires NORM::layout(input, weight),\n" in proof
        proof = proof.replace("    requires NORM::layout(input, weight),\n", "")

    # Minimal scalar/type boundary; production definitions and proof bodies
    # below are read verbatim. Full make verify covers the surrounding runtime.
    scalar_parameter = item("src/boundary/scalar.rs", "float_parameter_scalar_repr")
    epsilon = item("src/proof/model/types.rs", "unit_offset_norm_epsilon_repr")
    constant_name = re.search(r"bits: (\w+)", epsilon).group(1)
    definitions = [match.group()
        for path in (ROOT / "src/boundary/model_families").glob("*/config.rs")
        if (match := re.search(rf"^pub const {re.escape(constant_name)}:.*$", path.read_text(), re.M))]
    assert len(definitions) == 1, (constant_name, definitions)
    epsilon_constant = definitions[0]
    tensor_runtime = "\n".join(item("src/boundary/tensor_runtime.rs", name) for name in (
        "rms_norm_kernel_cell_repr", "rms_norm_kernel_row_repr", "offset_rms_norm_kernel_cell_repr"))
    source = f'''use vstd::prelude::*;
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
pub struct FloatParameterBits {{ pub bits: u64 }}
pub enum NormPolicyRepr {{ UnitOffset, Direct(FloatParameterBits) }}
{epsilon_constant}
{epsilon}
{scalar_parameter}
}}
}}
pub mod proof {{
pub mod model {{ pub mod types {{ pub use crate::fixture_types::*; }} }}
pub mod tensor {{
pub mod types {{ pub use crate::fixture_types::*; }}

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
#[path = "{ROOT / 'src/boundary/backend_certificates/rms_norm.rs'}"] pub mod rms_norm;
#[path = "{ROOT / 'src/boundary/backend_certificates/residual_rms_norm.rs'}"] pub mod residual_rms_norm;
#[path = "{ROOT / 'src/boundary/backend_certificates/offset_rms_norm.rs'}"] pub mod offset_rms_norm;
}}
#[path = "{ROOT / 'src/boundary/normalization_operator.rs'}"] pub mod normalization_operator;
}}
pub mod RT {{
use vstd::prelude::*;
use crate::types::*;
use crate::proof::model::types::*;
use crate::proof::tensor::types::*;
use crate::boundary::normalization_operator as NORM;
verus! {{ {tensor_runtime} }}
}}
pub mod ROWS {{
use vstd::prelude::*;
use crate::types::*;
use crate::proof::model::types::*;
use crate::proof::tensor::types::*;
use crate::RT;
verus! {{ {item('src/proof/model/four_norm_gated/layers.rs', 'norm_row')} }}
}}
use crate::types::*;
use crate::proof::model::types::*;
use crate::proof::tensor::types::*;
use crate::boundary::normalization_operator as NORM;
verus! {{
{item(relative, 'norm_repr')}
{raw}
{proof}
}}

mod types {{
    pub use crate::fixture_types::*;
    use vstd::prelude::*;
    verus! {{ {page_constants_source()} }}
}}

fn main() {{}}
'''
    path = tmp_path / "norm_policy.rs"
    path.write_text(source)
    result = subprocess.run([VERUS, "--cfg", "verus_only", str(path)],
        capture_output=True, text=True, timeout=120)
    if mutation is None:
        assert result.returncode == 0 and "0 errors" in result.stdout, result.stdout + result.stderr
    else:
        assert result.returncode != 0
        assert any(s in result.stderr for s in (
            "assertion failed", "precondition not satisfied", "postcondition not satisfied")), result.stderr
