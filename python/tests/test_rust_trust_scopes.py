"""Trust declarations in generated inline modules keep distinct identities."""

from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from scripts.audit.tcb import scan_rust_item_records, scan_rust_trust_records, scan_external_type_records, rust_source_files
from scripts.audit.rust_source import code_only


def test_inline_modules_impls_and_macros_have_distinct_identities(tmp_path):
    path = tmp_path / "fixture.rs"
    path.write_text('''
pub mod first {
verus! {
#[verifier::external_body]
pub proof fn imported() {}
#[verifier::external_body]
pub struct Handle {}
impl Handle {
pub uninterp spec fn value(&self) -> int;
}
}
mod nested {
#[verifier::external_body]
pub proof fn imported() {}
}
}
pub(crate) mod second {
#[verifier::external_body]
pub proof fn imported() {}
#[verifier::external_body]
pub struct Handle {}
}
#[verifier::external_body]
pub proof fn imported() {}
''')
    names = {r["identity"] for r in scan_rust_trust_records(path)}
    assert names == {"fixture::first::imported", "fixture::first::Handle::value",
                     "fixture::first::nested::imported", "fixture::second::imported", "fixture::imported"}
    assert {r["identity"] for r in scan_external_type_records(path)} == {
        "fixture::first::Handle", "fixture::second::Handle"}


def test_literals_and_nested_comments_cannot_change_scope_or_create_trust(tmp_path):
    path = tmp_path / "fixture.rs"
    source = '''
pub mod real {
fn strings<'a>() {
    let a = " } // not a comment \\\" still a string ";
    let b = r###" } \" mod fake {
#[verifier::external_body]
pub proof fn fake() {}
"###;
    let c = '}';
    let d = '\\u{007d}';
}
/* } /* { */
#[verifier::external_body]
pub proof fn fake() {}
*/
#[verifier::external_body]
pub proof fn real() {}
}
'''
    path.write_text(source)
    records = scan_rust_trust_records(path)
    assert [r["identity"] for r in records] == ["fixture::real::real"]
    masked = code_only(source)
    assert len(masked) == len(source)
    assert [i for i, c in enumerate(masked) if c == "\n"] == [i for i, c in enumerate(source) if c == "\n"]
    assert "strings<'a>" in masked
    assert "fake" not in masked


@pytest.mark.parametrize("source", ['/* nested /* */', 'r##"unfinished"#', '"unfinished'])
def test_unterminated_lexical_regions_are_rejected(source):
    with pytest.raises(ValueError, match="unterminated"):
        code_only(source)


def test_current_production_trust_identities_are_unique():
    records = [r for path in rust_source_files(ROOT / "src") for r in
               scan_rust_trust_records(path) + scan_external_type_records(path)]
    assert len(records) == len({r["identity"] for r in records})
    attention = [r for r in records if r["identity"].startswith("boundary::backend_certificates::attention::")]
    assert len(attention) == 30
    assert len({r["identity"].split("::")[3] for r in attention}) == 5


def test_function_extents_do_not_include_the_next_declaration(tmp_path):
    path = tmp_path / "fixture.rs"
    path.write_text('''pub mod a {
pub proof fn first() {
    let x = "}";
}
pub proof fn second() {}
}
''')
    records = scan_rust_item_records(path)
    assert [(r["identity"], r["line"], r["end_line"]) for r in records] == [
        ("fixture::a::first", 2, 4), ("fixture::a::second", 5, 5)]


def test_contract_literals_and_quantifiers_do_not_truncate_function(tmp_path):
    path = tmp_path / "fixture.rs"
    source = '''
#[verifier::external_body]
pub proof fn imported(x: int)
    requires
        condition(Witness { value: x }),
        forall|y: int| { y == x || y != x },
    ensures
        if x == 0 { true } else { x != 0 },
{
    unreachable!()
}
pub proof fn caller() {}
'''
    path.write_text(source)
    records = scan_rust_item_records(path)
    assert [(r["name"], r["line"], r["end_line"]) for r in records] == [
        ("imported", 3, 11), ("caller", 12, 12)]
    before = scan_rust_trust_records(path)[0]["source_sha256"]
    path.write_text(source.replace("unreachable!()", "assume(false);"))
    assert scan_rust_trust_records(path)[0]["source_sha256"] != before
