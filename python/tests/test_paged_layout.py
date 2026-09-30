"""Checked page-table adapter groundwork; no imported attention axiom needed."""

import os
from pathlib import Path
import re
import shutil
import subprocess

import pytest

from spec_test_support import page_constants_source


ROOT = Path(__file__).resolve().parents[2]
VERUS = os.environ.get("VERUS") or shutil.which("verus")


def module_source():
    # Use the actual common arithmetic and engine metadata definitions. The
    # page-table proofs use the production integer aliases, not Scalar values.
    return f'''
use vstd::prelude::*;
mod fixture_types {{
    use vstd::prelude::*;
    pub type BlockId = u64;
    pub type Scalar = int;
    pub type Tensor1D = Seq<Scalar>;
    pub type KVCacheLayerRepr = Seq<Seq<Tensor1D>>;
    pub type IntTensor2D = Seq<Seq<int>>;
}}
#[path = "{ROOT / 'src/proof/tensor/geometry.rs'}"]
pub mod common;
mod boundary {{
pub mod scalar {{ pub use crate::fixture_types::Scalar; }}
 pub mod backend_certificates {{
    #[path = "{ROOT / 'src/boundary/backend_certificates/support.rs'}"]
    pub mod support;
}} }}
#[path = "{ROOT / 'src/proof/tensor/paged.rs'}"]
pub mod paged_layout;
use paged_layout::*;

mod proof {{
pub mod model {{ pub mod types {{ pub use crate::fixture_types::*; }} }}
pub mod tensor {{
pub mod types {{ pub use crate::fixture_types::*; }}
pub use crate::common as geometry;
pub use crate::paged_layout as paged;

}}
}}

mod types {{
    pub use crate::fixture_types::*;
    use vstd::prelude::*;
    verus! {{ {page_constants_source()} }}
}}

fn main() {{}}
'''


@pytest.mark.skipif(VERUS is None, reason="set VERUS to check paged layout proofs")
@pytest.mark.parametrize("case", ["positive", "empty_pool", "narrow_table", "missing_prefix", "partial_last_page"])
def test_checked_page_table_materialization(tmp_path, case):
    source = module_source()
    if case == "positive":
        source += '''
verus! {
proof fn caller() {
    let table = seq![seq![3u64], seq![1u64, 2u64, 0u64]];
    assert(page_table_ids_valid(table, 4));
    lemma_canonical_page_table(table, 4);
    reveal_with_fuel(page_table_width, 3);
    assert(page_table_width(table) == 3);
    let padded = rectangular_page_table(table, 3);
    assert(padded[0] =~= seq![3int, 0int, 0int]);
    assert(padded[1] =~= seq![1int, 2int, 0int]);
    assert(padded[0][0] == rectangular_page_table(seq![seq![3u64]], 1)[0][0]);
    // Actual attention adapter: equal data at different physical page IDs,
    // with different rectangular table widths.
    let page = Seq::new(64, |slot: int| seq![0int]);
    lemma_rectangular_cache_page_equality(
        seq![page], seq![0u64], 1, seq![page, page], seq![1u64], 3, 1, 0);
}
}
'''
    elif case == "empty_pool":
        source += '''
verus! {
proof fn caller() {
    // Logical validity is vacuous, but padding zero cannot belong to no pages.
    let table: Seq<Seq<u64>> = seq![seq![]];
    assert(page_table_ids_valid(table, 0));
    lemma_rectangular_page_table_domain(table, 1, 0);
}
}
'''
    elif case == "narrow_table":
        source += '''
verus! {
proof fn caller() {
    // A narrow rectangular allocation cannot preserve every logical entry.
    lemma_rectangular_page_table_preserves_entries(seq![seq![0u64, 1u64]], 1);
}
}
'''
    elif case == "missing_prefix":
        source += '''
verus! {
proof fn caller() {
    // Valid page IDs and geometry do not establish equality of cached data.
    let zero = Seq::new(64, |slot: int| seq![0int]);
    let different = Seq::new(64, |slot: int| if slot == 0 { seq![1int] } else { seq![0int] });
    lemma_rectangular_cache_page_equality(
        seq![zero], seq![0u64], 1, seq![zero, different], seq![1u64], 2, 1, 0);
}
}
'''
    else:
        source += '''
verus! {
proof fn caller() {
    let zero = Seq::new(64, |slot: int| seq![0int]);
    let tail = Seq::new(64, |slot: int| if slot == 0 { seq![0int] } else { seq![1int] });
    let a = seq![zero, zero];
    let b = seq![zero, tail];
    let row = seq![0u64, 1u64];
    // The first 65 tokens agree, but the required complete last page does not.
    assert forall|pos: nat| pos < 65 implies
        common::cache_at(a, common::block_table_slot(row, pos))
            == common::cache_at(b, common::block_table_slot(row, pos)) by {
        if pos < 64 {
            assert(pos / 64 == 0);
        } else {
            assert(pos == 64);
        }
    }
    lemma_rectangular_cache_page_equality(a, row, 2, b, row, 2, 2, 1);
}
}
'''
    path = tmp_path / "page_table.rs"
    path.write_text(source)
    result = subprocess.run([VERUS, str(path)], capture_output=True, text=True, timeout=120)
    if case == "positive":
        assert result.returncode == 0 and "0 errors" in result.stdout, result.stdout + result.stderr
    else:
        assert result.returncode != 0 and "precondition not satisfied" in result.stderr, result.stdout + result.stderr
        failures = [line for line in re.findall(r"^error: (.+)$", result.stderr, re.MULTILINE)
                    if not line.startswith("aborting due to")]
        assert failures == ["precondition not satisfied"], result.stdout + result.stderr
        assert "page_table.rs" in result.stderr
