"""Coverage and rejection checks for the installed, geometry-bound raw catalog."""

import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

import pytest

from spec_test_support import page_constants_source

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "kernels")]
from scripts.verification.verify_attention_interfaces import selected_cases
from scripts.verification.attention_interface_catalog import render_attention_catalog

VERUS = os.environ.get("VERUS") or shutil.which("verus")


def test_catalog_covers_current_qualification_inventories():
    manifest = json.loads((ROOT / "audit/attention_kernel_interfaces.json").read_text())
    body = (ROOT / "src/boundary/backend_certificates/attention.rs").read_text()
    assert manifest["generated_body_sha256"] == hashlib.sha256(body.encode()).hexdigest()
    expected = [dict(source=c["source"], kernel=c["kernel"], constants=constants, families=sorted(owners))
                for c, constants, owners in selected_cases()]
    assert [{k: v for k, v in item.items() if k != "execution_identity"}
            for item in manifest["inventory_contributions"]] == expected
    assert {item["execution_identity"] for item in manifest["inventory_contributions"]} == {
        impl["execution_identity"] for interface in manifest["interfaces"]
        for impl in interface["raw"]["implementations"]}
    assert {(x["sliding_window"], x["head_dim"]) for x in manifest["interfaces"]} == {
        (False, 128), (False, 256), (False, 512), (True, 128), (True, 256)}
    assert sum(len(x["raw"]["implementations"]) for x in manifest["interfaces"]) == len(expected)
    dispatch = body.split("// Geometry dispatch is shared across architectures.", 1)[1]
    assert "external_body" not in dispatch and "assume(" not in dispatch
    assert "BLOCK_M" not in dispatch and "BLOCK_N" not in dispatch
    assert "else { None }" in dispatch


def test_catalog_rejects_empty_inventory():
    with pytest.raises(ValueError, match="no qualified implementations"):
        render_attention_catalog(())


@pytest.mark.skipif(VERUS is None, reason="set VERUS to check catalog binding rejection")
@pytest.mark.parametrize("case", ["unsupported_geometry", "missing_geometry", "missing_layout", "missing_numeric"])
def test_catalog_binding_premises(tmp_path, case):
    # Check real production definitions. The small type shell mirrors the
    # semantic tensor aliases; the full crate separately checks the same proof
    # and all raw adapter bodies against its actual semantic types.
    header = f'''
use vstd::prelude::*;
#[path = "{ROOT / 'src/boundary/scalar.rs'}"] pub mod scalar_boundary;
mod fixture_types {{
    use vstd::prelude::*;
    verus! {{
        pub struct FloatParameterBits {{ pub bits: u64 }}
        pub enum AttentionKind {{ Full, SlidingWindow }}
        pub struct AttentionGeometryRepr {{
            pub num_attention_heads: nat, pub num_key_value_heads: nat, pub head_dim: nat,
        }}
        pub enum AttentionScaleRepr {{ InverseSqrtHeadDim, InverseSqrtParameter(FloatParameterBits) }}
        pub struct AttentionParametersRepr {{ pub geometry: AttentionGeometryRepr, pub scale: AttentionScaleRepr }}
    }}
    pub type Scalar = super::scalar_boundary::Scalar;
    pub type BlockId = u64;
    pub type Tensor1D = Seq<Scalar>;
    pub type Tensor2D = Seq<Tensor1D>;
    pub type Tensor3D = Seq<Tensor2D>;
    pub type Tensor4D = Seq<Tensor3D>;
    pub type KVCacheLayerRepr = Tensor3D;
    pub type IntTensor2D = Seq<Seq<int>>;
}}
#[path = "{ROOT / 'src/proof/tensor/geometry.rs'}"] pub mod common;
#[path = "{ROOT / 'src/proof/tensor/seq_flatten.rs'}"] pub mod seq_flatten;
mod proof {{
pub mod model {{ pub mod types {{ pub use crate::fixture_types::*; }} }}
pub mod tensor {{
pub mod types {{ pub use crate::fixture_types::*; }}
pub use crate::common as geometry;
pub use crate::seq_flatten as seq_flatten;

    #[path = "{ROOT / 'src/proof/tensor/attention_projection.rs'}"] pub mod attention_projection;
    #[path = "{ROOT / 'src/proof/tensor/paged.rs'}"] pub mod paged;
    #[path = "{ROOT / 'src/proof/tensor/shape.rs'}"] pub mod shape;
    #[path = "{ROOT / 'src/proof/tensor/layout.rs'}"] pub mod layout;

}}
}}
mod boundary {{
pub use crate::scalar_boundary as scalar;

    pub mod backend_certificates {{
        #[path = "{ROOT / 'src/boundary/backend_certificates/support.rs'}"] pub mod support;
        #[path = "{ROOT / 'src/boundary/backend_certificates/attention.rs'}"] pub mod attention;
    }}
    #[path = "{ROOT / 'src/boundary/attention_operator.rs'}"] pub mod attention_operator;
}}

mod model_config {{ pub use crate::fixture_types::*; }}
mod types {{
    pub use crate::fixture_types::*;
    use vstd::prelude::*;
    verus! {{ {page_constants_source()} }}
}}

fn main() {{}}
'''
    caller = '''
mod probe {
use vstd::prelude::*;
use crate::{types::*, proof::model::types::*, proof::tensor::types::*};
use crate::boundary::backend_certificates::{attention as RAW, support as SUP};
use crate::boundary::attention_operator as AO;
verus! {
pub proof fn caller(
    q: Tensor2D, k: KVCacheLayerRepr, v: KVCacheLayerRepr,
    cu_q: Seq<int>, cu_k: Seq<int>, table: Seq<Seq<BlockId>>, max_q: nat, max_k: nat,
    kind: AttentionKind, parameters: AttentionParametersRepr, window: nat,
)
    requires
        __GEOMETRY__
        __LAYOUT__
        __NUMERIC__
{
    AO::checked_runtime_binding(q, k, v, cu_q, cu_k, table, max_q, max_k, kind, parameters, window);
}
}
}
'''
    premises = {
        "GEOMETRY": "RAW::binding_valid(kind, parameters.geometry, window),",
        "LAYOUT": "RAW::layout_ready(q, k, v, table, cu_q, cu_k, max_q, max_k, parameters.geometry),",
        "NUMERIC": "RAW::numeric_requirements(q, k, v, table, cu_q, cu_k, kind, parameters.geometry, "
                   "AO::scale_log2(parameters), window, SUP::generated_kernel_allocation_cell()),",
    }
    if case == "unsupported_geometry":
        caller = '''
mod probe {
use vstd::prelude::*;
use crate::{types::*, proof::model::types::*, proof::tensor::types::*};
use crate::boundary::backend_certificates::attention as RAW;
verus! {
pub proof fn caller(kind: AttentionKind, scale: Scalar, fill: Scalar, window: nat)
    ensures
        RAW::row_operation(kind, AttentionGeometryRepr {
            num_attention_heads: 8, num_key_value_heads: 4, head_dim: 64,
        }, scale, window, fill).is_none(),
        RAW::row_operation(AttentionKind::SlidingWindow, AttentionGeometryRepr {
            num_attention_heads: 8, num_key_value_heads: 4, head_dim: 128,
        }, scale, 0, fill).is_none(),
{}
}
}
'''
    else:
        premises[case.removeprefix("missing_").upper()] = ""
        for name, text in premises.items():
            caller = caller.replace(f"__{name}__", text)
    path = tmp_path / "catalog_probe.rs"
    path.write_text(header + caller)
    result = subprocess.run([VERUS, str(path), "--verify-only-module", "probe", "--verify-function", "caller"],
                            capture_output=True, text=True, timeout=120)
    if case == "unsupported_geometry":
        assert result.returncode == 0 and "0 errors" in result.stdout, result.stdout + result.stderr
    else:
        assert result.returncode != 0, "missing binding premise was accepted"
        errors = [x for x in re.findall(r"^error: (.+)$", result.stderr, re.MULTILINE)
                  if not x.startswith("aborting due to")]
        assert errors and set(errors) == {"precondition not satisfied"}, result.stdout + result.stderr
        assert "AO::checked_runtime_binding" in result.stderr
