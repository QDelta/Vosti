"""Check architecture-neutral attention projection laws, without kernel axioms."""

import os
from pathlib import Path
import re
import shutil
import subprocess

import pytest

from spec_test_support import page_constants_source

ROOT = Path(__file__).resolve().parents[2]
VERUS = os.environ.get("VERUS") or shutil.which("verus")


@pytest.mark.skipif(VERUS is None, reason="set VERUS to check attention projection")
@pytest.mark.parametrize("case", ["positive", "wrong_owner", "unordered_offsets", "different_operation"])
def test_attention_projection(tmp_path, case):
    header = f'''
use vstd::prelude::*;
#[path = "{ROOT / 'src/boundary/scalar.rs'}"] pub mod scalar_boundary;
mod fixture_types {{
    use vstd::prelude::*;
    verus! {{ pub struct FloatParameterBits {{ pub bits: u64 }} }}
    pub type Scalar = super::scalar_boundary::Scalar;
    pub type BlockId = u64;
    pub type Tensor1D = Seq<Scalar>;
    pub type Tensor2D = Seq<Tensor1D>;
    pub type Tensor3D = Seq<Tensor2D>;
    pub type Tensor4D = Seq<Tensor3D>;
    pub type KVCacheLayerRepr = Tensor3D;
}}
#[path = "{ROOT / 'src/proof/tensor/geometry.rs'}"] pub mod common;
mod model_config {{ pub use crate::fixture_types::FloatParameterBits; }}
mod proof {{
pub mod model {{ pub mod types {{ pub use crate::fixture_types::*; }} }}
pub mod tensor {{
pub mod types {{ pub use crate::fixture_types::*; }}
pub use crate::common as geometry;

    #[path = "{ROOT / 'src/proof/tensor/shape.rs'}"] pub mod shape;
    #[path = "{ROOT / 'src/proof/tensor/attention_projection.rs'}"] pub mod attention_projection;

}}
}}
use proof::tensor::attention_projection::*;
use fixture_types::*;

mod types {{
    pub use crate::fixture_types::*;
    use vstd::prelude::*;
    verus! {{ {page_constants_source()} }}
}}


mod boundary {{ pub use crate::scalar_boundary as scalar;
}}
fn main() {{}}
'''
    if case == "positive":
        caller = '''
proof fn caller(q: Tensor1D) {
    lemma_owner(seq![0int, 2int, 5int], 2, 1, 3);
    assert(owner(seq![0int, 2int, 5int], 2, 3) == 1);
    let op: RowOperation = |q: Tensor1D, k: Tensor2D, v: Tensor2D| q;
    lemma_batch_projection(op, q.len(), seq![q, q, q], seq![], seq![],
        seq![0int, 1int, 3int], seq![0int, 65int, 132int], seq![seq![0u64], seq![1u64, 0u64]], 1);
}
'''
    elif case == "wrong_owner":
        caller = 'proof fn caller() { lemma_owner(seq![0int, 2int, 5int], 2, 0, 3); }'
    elif case == "unordered_offsets":
        caller = 'proof fn caller() { lemma_owner(seq![0int, 5int, 2int], 2, 0, 1); }'
    else:
        caller = '''
proof fn caller(a: RowOperation, b: RowOperation, q: Tensor1D) {
    // One fixed operation is essential; changing the operation is not covered.
    assert(row_output(a, 1, q, seq![], seq![]) == row_output(b, 1, q, seq![], seq![]));
}
'''
    path = tmp_path / "attention_projection.rs"
    path.write_text(header + '\nverus! {\n' + caller + '\n}\n')
    result = subprocess.run([VERUS, str(path)], capture_output=True, text=True, timeout=120)
    if case == "positive":
        assert result.returncode == 0 and "0 errors" in result.stdout, result.stdout + result.stderr
    else:
        failures = [line for line in re.findall(r"^error: (.+)$", result.stderr, re.MULTILINE)
                    if not line.startswith("aborting due to")]
        expected = "assertion failed" if case == "different_operation" else "precondition not satisfied"
        assert result.returncode != 0 and failures == [expected], result.stdout + result.stderr
