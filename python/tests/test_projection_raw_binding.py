"""Raw projection kernels plus engine layouts, checked without numeric axioms."""

import os
from pathlib import Path
import shutil
import subprocess

import pytest

from spec_test_support import page_constants_source

ROOT = Path(__file__).resolve().parents[2]
VERUS = os.environ.get("VERUS") or shutil.which("verus")


@pytest.mark.skipif(VERUS is None, reason="set VERUS for raw engine bindings")
@pytest.mark.parametrize("name,mutation", [
    ("linear", None), ("linear", "transpose"), ("linear", "missing_layout"),
    ("qkv", None), ("qkv", "missing_layout"), ("qkv", "output_swap"),
    ("qkv", "missing_domain"),
    ("normalization", None), ("normalization", "rms_layout"),
    ("normalization", "residual_layout"), ("normalization", "residual_output_swap"),
    ("normalization", "rms_epsilon"), ("normalization", "residual_epsilon"),
    ("normalization", "offset_epsilon"), ("normalization", "offset_kernel"),
    ("pointwise", None), ("pointwise", "add_width"),
    ("pointwise", "binary_layout"), ("pointwise", "scale_shape"),
    ("pointwise", "softcap_parameter"), ("pointwise", "pointwise_kernel"),
    ("embedding", None), ("embedding", "embedding_layout"),
    ("embedding", "embedding_token"), ("embedding", "embedding_scalar"),
    ("embedding", "embedding_kernel"),
    ("head_normalization", None),
    ("head_normalization", "head_layout"), ("head_normalization", "head_eps"),
    ("head_normalization", "head_kernel"), ("head_normalization", "head_row"),
    ("rotary", None),
    ("rotary", "rotary_layout"), ("rotary", "rotary_tables"),
    ("rotary", "rotary_row"),
])
def test_projection_binding(tmp_path, name, mutation):
    source = (ROOT / f"src/boundary/{name}_operator.rs").read_text()
    if mutation in {"rotary_layout", "rotary_tables", "rotary_row"}:
        before, after = {
            "rotary_layout": ("    requires layout(input, cos, sin),", "    requires true,"),
            "rotary_tables": ("raw_output(seq![row], seq![cos], seq![sin])[0]", "raw_output(seq![row], seq![sin], seq![cos])[0]"),
            "rotary_row": ("row_output(input[r], cos[r], sin[r])", "row_output(input[r], cos[0], sin[r])"),
        }[mutation]
        assert before in source
        source = source.replace(before, after)
    if mutation in {"head_layout", "head_eps", "head_kernel", "head_row"}:
        before, after = {
            "head_layout": ("    requires layout(input, weight, offset),", "    requires true,"),
            "head_eps": ("raw_output(seq![row], weight, eps, offset)[0]",
                "raw_output(seq![row], weight, crate::boundary::backend_certificates::support::generated_kernel_allocation_cell(), offset)[0]"),
            "head_kernel": ("raw_output(seq![row], weight, eps, offset)[0]", "raw_output(seq![row], weight, eps, !offset)[0]"),
            "head_row": ("row_output(input[r], weight, eps, offset)", "row_output(input[0], weight, eps, offset)"),
        }[mutation]
        assert before in source
        source = source.replace(before, after)
    if mutation in {"embedding_layout", "embedding_token", "embedding_scalar", "embedding_kernel"}:
        before, after = {
            "embedding_layout": ("    requires layout(weight),", ""),
            "embedding_token": ("plain_row_output(ids[row], weight)", "plain_row_output(ids[row + 1], weight)"),
            "embedding_scalar": ("scaled_raw_output(seq![token], weight, width)[0]",
                "SCALED::raw_output(seq![token], weight, crate::boundary::backend_certificates::support::generated_kernel_allocation_cell(), width, weight.len()).unwrap()[0]"),
            "embedding_kernel": ("scaled_raw_output(seq![token], weight, width)[0]", "plain_raw_output(seq![token], weight)[0]"),
        }[mutation]
        assert before in source
        source = source.replace(before, after)
    if mutation in {"add_width", "binary_layout", "scale_shape", "softcap_parameter", "pointwise_kernel"}:
        before, after = {
            "add_width": (" input.len() > 0 ==> width(input) > 0,", ""),
            "binary_layout": ("requires binary_layout(input, other),", "requires layout(input),"),
            "scale_shape": (" other.len() == 1,", ""),
            "softcap_parameter": ("softcap_raw_output(seq![row], other)[0]", "softcap_raw_output(seq![row], crate::boundary::backend_certificates::support::generated_kernel_allocation_cell())[0]"),
            "pointwise_kernel": ("gelu_tanh_mul_raw_output(seq![row], seq![other])[0]", "silu_mul_raw_output(seq![row], seq![other])[0]"),
        }[mutation]
        assert before in source
        source = source.replace(before, after)
    if mutation == "transpose":
        source = source.replace("use crate::proof::tensor::layout::transposed_weights;", "")
        source = source.replace("verus! {", """verus! {
pub open spec fn transposed_weights(weights: Tensor2D, width: nat) -> Tensor2D {
    Seq::new(width, |k: int| Seq::new(weights.len(), |n: int| weights[k][n]))
}
""")
        # A wrong transpose is still a deterministic operation. To test layout
        # correspondence, fix the engine cell's canonical transpose separately.
        source = source.replace("transposed_weights(weights, row.len())",
            "Seq::new(row.len(), |k: int| Seq::new(weights.len(), |n: int| weights[n][k]))")
    elif mutation == "missing_layout":
        premise = ("layout(input, weights, width)" if name == "linear"
                   else "layout(input, qw, kw, vw, width)")
        assert f"    requires {premise},\n" in source
        source = source.replace(f"    requires {premise},\n", "")
    elif mutation == "output_swap":
        assert "(raw.0[0], raw.1[0], raw.2[0])" in source
        source = source.replace("(raw.0[0], raw.1[0], raw.2[0])", "(raw.0[0], raw.2[0], raw.1[0])")
    elif mutation == "missing_domain":
        assert "&& width > 0" in source
        source = source.replace("&& width > 0", "")
    elif mutation in {"rms_layout", "residual_layout"}:
        premise = ("layout(input, weight)" if mutation == "rms_layout"
                   else "residual_layout(input, residual, weight)")
        assert f"    requires {premise},\n" in source
        source = source.replace(f"    requires {premise},\n", "")
    elif mutation == "residual_output_swap":
        assert "(raw.0[0], raw.1[0])" in source
        source = source.replace("(raw.0[0], raw.1[0])", "(raw.1[0], raw.0[0])")
    elif mutation in {"rms_epsilon", "residual_epsilon", "offset_epsilon"}:
        call = {"rms_epsilon": "rms_raw_output(seq![row], weight, eps)",
                "residual_epsilon": "residual_raw_output(seq![row], seq![residual], weight, eps)",
                "offset_epsilon": "offset_raw_output(seq![row], weight, eps)"}[mutation]
        assert call in source
        source = source.replace(call, call.removesuffix("eps)")
            + "crate::boundary::backend_certificates::support::generated_kernel_allocation_cell())")
    elif mutation == "offset_kernel":
        call = "offset_raw_output(seq![row], weight, eps)"
        assert call in source
        source = source.replace(call, "rms_raw_output(seq![row], weight, eps)")
    operator = tmp_path / f"{name}_operator.rs"
    operator.write_text(source)
    interfaces = [name] if name != "normalization" else ["rms_norm", "residual_rms_norm", "offset_rms_norm"]
    if name == "pointwise":
        interfaces = ["add", "silu_mul", "gelu_tanh_mul", "scale", "softcap"]
    if name == "embedding":
        interfaces = ["embedding", "scaled_embedding"]
    if name == "head_normalization":
        interfaces = ["head_rms_norm", "offset_head_rms_norm"]
    imports = "\n".join(
        f'#[path = "{ROOT / f"src/boundary/backend_certificates/{interface}.rs"}"] pub mod {interface};'
        for interface in interfaces)
    module = f'''use vstd::prelude::*;
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
    #[path = "{ROOT / 'src/proof/tensor/layout.rs'}"] pub mod layout;

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
        {imports}
    }}
    #[path = "{operator}"] pub mod {name}_operator;
}}

mod types {{
    pub use crate::fixture_types::*;
    use vstd::prelude::*;
    verus! {{ {page_constants_source()} }}
}}

fn main() {{}}
'''
    path = tmp_path / f"{name}_binding.rs"
    path.write_text(module)
    result = subprocess.run([VERUS, "--cfg", "verus_only", str(path)],
                            capture_output=True, text=True, timeout=120)
    if mutation is None:
        assert result.returncode == 0 and "0 errors" in result.stdout, result.stdout + result.stderr
    else:
        assert result.returncode != 0
        assert any(msg in result.stderr for msg in (
            "assertion failed", "precondition not satisfied", "postcondition not satisfied")), result.stderr
