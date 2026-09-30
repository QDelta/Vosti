"""Check real reshape proofs and reject incomplete geometry/equality premises."""

import os
from pathlib import Path
import re
import shutil
import subprocess

import pytest

from spec_test_support import page_constants_source

ROOT = Path(__file__).resolve().parents[2]
VERUS = os.environ.get("VERUS") or shutil.which("verus")


def layout_module():
    return f'''
use vstd::prelude::*;
mod fixture_types {{
    use vstd::prelude::*;
    pub type Scalar = int;
    pub type Tensor1D = Seq<Scalar>;
    pub type Tensor2D = Seq<Tensor1D>;
    pub type Tensor3D = Seq<Tensor2D>;
    pub type Tensor4D = Seq<Tensor3D>;
}}
#[path = "{ROOT / 'src/proof/tensor/seq_flatten.rs'}"]
pub mod seq_flatten;
mod proof {{
pub mod model {{ pub mod types {{ pub use crate::fixture_types::*; }} }}
pub mod tensor {{
pub mod types {{ pub use crate::fixture_types::*; }}
pub use crate::seq_flatten as seq_flatten;

    #[path = "{ROOT / 'src/proof/tensor/shape.rs'}"]
    pub mod shape;
    #[path = "{ROOT / 'src/proof/tensor/layout.rs'}"]
    pub mod layout;

}}
}}
use proof::tensor::shape as TS;
use proof::tensor::layout::*;

mod types {{
    pub use crate::fixture_types::*;
    use vstd::prelude::*;
    verus! {{ {page_constants_source()} }}
}}


mod boundary {{ pub mod scalar {{ pub use crate::fixture_types::Scalar; }}
}}
fn main() {{}}
'''


@pytest.mark.skipif(VERUS is None, reason="set VERUS to check tensor layout proofs")
@pytest.mark.parametrize("case", ["positive", "bad_width", "missing_row_equality", "short_region", "missing_region_equality", "ragged_output"])
def test_tensor_layout(tmp_path, case):
    if case == "positive":
        caller = '''
    let rows = seq![seq![1int, 2int, 3int, 4int], seq![5int, 6int, 7int, 8int]];
    assert(TS::tensor2d_shape(rows, 2, 4));
    lemma_split_last_axis_shape(rows, 2, 2, 2);
    let split = split_last_axis(rows, 2, 2);
    assert(split[1][1][1] == 8);
    lemma_split_last_axis_subrange(rows, 2, 2, 1, 2);
    assert(TS::tensor3d_shape(seq![rows, rows], 2, 2, 4));
    lemma_split_last_axis_3d_shape(seq![rows, rows], 2, 2, 2, 2);
    assert(split_last_axis_3d(seq![rows], 2, 2)[0]
        == split_last_axis_3d(seq![rows, rows], 2, 2)[1]);
    lemma_merge_last_axis_shape(split, 2, 2, 2);
    assert(merge_last_axis(split.subrange(1, 2))
        =~= merge_last_axis(split).subrange(1, 2));
    lemma_merge_last_axis_region_equality(split, seq![split[1]], 1, 0, 1, 2, 2);
    lemma_tensor3d_allocation_shape(2, 2, 2, 7int);
'''
    elif case == "bad_width":
        caller = 'lemma_split_last_axis_shape(seq![seq![1int, 2int, 3int]], 1, 2, 2);'
    elif case == "missing_row_equality":
        caller = '''lemma_split_last_axis_row_equality(
            seq![seq![1int, 2int]], seq![seq![1int, 3int]], 0, 0, 1, 2);'''
    elif case == "short_region":
        caller = '''lemma_merge_last_axis_region_equality(
            seq![seq![seq![1int]]], seq![seq![seq![1int]]], 0, 0, 2, 1, 1);'''
    elif case == "missing_region_equality":
        caller = '''lemma_merge_last_axis_region_equality(
            seq![seq![seq![1int]]], seq![seq![seq![2int]]], 0, 0, 1, 1, 1);'''
    else:
        caller = 'lemma_merge_last_axis_shape(seq![seq![seq![1int], seq![2int, 3int]]], 1, 2, 2);'
    source = tmp_path / "tensor_layout.rs"
    source.write_text(layout_module() + '\nverus! { proof fn caller() {\n' + caller + '\n} }\n')
    result = subprocess.run([VERUS, str(source)], capture_output=True, text=True, timeout=120)
    if case == "positive":
        assert result.returncode == 0 and "0 errors" in result.stdout, result.stdout + result.stderr
    else:
        failures = [line for line in re.findall(r"^error: (.+)$", result.stderr, re.MULTILINE)
                    if not line.startswith("aborting due to")]
        assert result.returncode != 0 and failures == ["precondition not satisfied"], result.stdout + result.stderr
