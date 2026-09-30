"""Check internal arbitrary-start continuation vocabulary and agreement."""

import os
from pathlib import Path
import re
import shutil
import subprocess

import pytest

from spec_test_support import compiler_closure

ROOT = Path(__file__).resolve().parents[2]
VERUS = os.environ.get("VERUS") or shutil.which("verus")


def test_continuation_vocabulary_has_no_engine_dependencies():
    from scripts.audit.rust_source import code_only

    source = code_only((ROOT / "src/proof/serving/continuation_agreement/vocabulary.rs").read_text())
    assert "pub use crate::boundary::sampler::SamplerState;" in source
    source = source.replace("pub use crate::boundary::sampler::SamplerState;", "")
    assert "pub use crate::types::{RequestId, TokenId};" in source
    source = source.replace("pub use crate::types::{RequestId, TokenId};", "")
    assert not re.search(r"\b(crate|super|extern)\b|#\s*\[\s*path\b", source)
    assert not re.search(r"external_body|external_type|\bassume\s*\(|\badmit\s*\(", source)
    assert re.search(r"struct\s+TraceInterface\s*<\s*State\s*,\s*Event\s*>", source)


def test_continuation_vocabulary_compiler_dependency_closure():
    closure = compiler_closure("proof::serving::continuation_agreement::vocabulary::request_consistent")
    assert {node["path"] for node in closure.values()} == {
        "src/proof/serving/continuation_agreement/vocabulary.rs", "src/boundary/sampler.rs"}


def test_preserved_continuation_agreement_guarantee():
    closure = compiler_closure("proof::serving::continuation_agreement::lemma_request_prefix_agreement")
    assert "vosti_verus::proof::serving::continuation_agreement::vocabulary::request_consistent" in closure
    assert "vosti_verus::proof::serving::continuation_agreement::interpretation::engine_trace_interface" in closure


@pytest.mark.skipif(VERUS is None, reason="set VERUS to check the standalone continuation vocabulary")
@pytest.mark.parametrize("case", [
    "projection", "unequal_lengths", "nondeterminism_witness", "consumer",
    "false_agreement_claim", "broken_chain", "different_output", "different_sampler",
    "input_fields", "input_sampler",
])
def test_standalone_continuation_agreement_vocabulary(tmp_path, case):
    # This compilation has no Engine, IBM, tensor, family, or refinement modules.
    header = f'''
use vstd::prelude::*;
#[path = "{ROOT / 'src/types.rs'}"] mod types;
mod boundary {{
    #[path = "{ROOT / 'src/boundary/sampler.rs'}"] pub mod sampler;
}}
#[path = "{ROOT / 'src/proof/serving/continuation_agreement/vocabulary.rs'}"] mod continuation_agreement_vocabulary;
use continuation_agreement_vocabulary::*;
fn main() {{}}
verus! {{
type Event = (int, int, RequestId, Option<Observation>);
type System = TraceInterface<int, Event>;
spec fn sampler() -> SamplerState {{ arbitrary() }}
spec fn observation(token: TokenId) -> Observation {{ (sampler(), token) }}
spec fn request(prompt: Seq<TokenId>) -> SamplingInput {{
    SamplingInput {{ prompt, generated: Seq::empty(), sampler: sampler() }}
}}
spec fn system() -> System {{
    TraceInterface {{
        admissible: |s: int| s >= 0,
        before: |e: Event| e.0,
        after: |e: Event| e.1,
        step: |e: Event| e.1 == e.0 + 1,
        input: |s: int, rid: u64| Some(request(seq![42u64])),
        output: |e: Event, rid: u64| if e.2 == rid {{ e.3 }} else {{ None }},
    }}
}}
spec fn run(value: u64) -> Trace<int, Event> {{
    Trace {{ initial: 0, final_state: 1,
        events: seq![(0int, 1int, 7u64, Some(observation(value)))] }}
}}
'''
    bodies = {
        "consumer": "",
        "projection": '''
            reveal_with_fuel(observations, 5);
            let events = seq![(0int, 1int, 7u64, Some(observation(10))),
                (1int, 2int, 8u64, Some(observation(99))),
                (2int, 3int, 7u64, None), (3int, 4int, 7u64, Some(observation(11)))];
            assert(observations(system().output, events, 7u64) == seq![observation(10), observation(11)]);
        ''',
        "unequal_lengths": '''
            assert(common_prefix_equal(seq![observation(10)], seq![observation(10), observation(11)]));
            assert(common_prefix_equal(Seq::<Observation>::empty(), seq![observation(99)]));
        ''',
        "nondeterminism_witness": '''
            assert(legal_trace(system(), run(10)));
            assert(legal_trace(system(), run(11)));
            assert(observations(system().output, run(10).events, 7u64)[0] == observation(10));
            assert(observations(system().output, run(11).events, 7u64)[0] == observation(11));
            assert(!request_consistent(system(), run(10), run(11), 7u64, 7u64));
        ''',
        "false_agreement_claim": 'assert(request_consistent(system(), run(10), run(11), 7u64, 7u64));',
        "broken_chain": '''
            let trace = Trace { initial: 0int, final_state: 3int,
                events: seq![(0int, 1int, 7u64, Some(observation(10))),
                    (2int, 3int, 7u64, Some(observation(10)))] };
            assert(legal_trace(system(), trace));
        ''',
        "input_fields": '''
            let a = request(seq![1u64]);
            let b = request(seq![2u64]);
            assert(a.prompt[0] != b.prompt[0]);
            assert(a != b);
            let c = SamplingInput { prompt: a.prompt, generated: seq![3u64], sampler: a.sampler };
            assert(a.generated.len() != c.generated.len());
            assert(a != c);
        ''',
        "different_sampler": 'assert(common_prefix_equal(seq![(a, 10u64)], seq![(b, 10u64)]));',
        "input_sampler": '''
            let left = SamplingInput { prompt: seq![1u64], generated: Seq::empty(), sampler: a };
            let right = SamplingInput { prompt: left.prompt, generated: left.generated, sampler: b };
            assert(left != right);
        ''',
        "different_output": 'assert(common_prefix_equal(seq![observation(10)], seq![observation(11)]));',
    }
    consumer = '''
proof fn consume(sys: System, left: Trace<int, Event>, right: Trace<int, Event>, a: u64, b: u64)
    requires
        request_consistent(sys, left, right, a, b), legal_trace(sys, left), legal_trace(sys, right),
        (sys.input)(left.initial, a).is_some(),
        (sys.input)(left.initial, a) == (sys.input)(right.initial, b),
    ensures common_prefix_equal(observations(sys.output, left.events, a),
        observations(sys.output, right.events, b)),
{
    assert(request_consistent(sys, left, right, a, b));
}
''' if case == "consumer" else ""
    source = tmp_path / "standalone.rs"
    signature = ("proof fn test(a: SamplerState, b: SamplerState) requires a != b {\n"
                 if case in {"different_sampler", "input_sampler"} else "proof fn test() {\n")
    source.write_text(header + signature + bodies[case] + "\n}\n" + consumer + "}\n")
    result = subprocess.run([VERUS, str(source)], text=True, capture_output=True, timeout=90)
    output = result.stdout + result.stderr
    if case in {"projection", "unequal_lengths", "nondeterminism_witness", "consumer", "input_fields", "input_sampler"}:
        assert result.returncode == 0 and "0 errors" in result.stdout, output
    else:
        errors = re.findall(r"^error: (.+)$", result.stderr, re.MULTILINE)
        errors = [e for e in errors if not e.startswith("aborting due to")]
        assert result.returncode != 0 and errors == ["assertion failed"], output
