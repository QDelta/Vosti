"""Explicit execution vocabulary and checked engine-to-record boundaries."""

import json
import os
from pathlib import Path
import re
import shutil
import subprocess

import pytest

from spec_test_support import compiler_closure

ROOT = Path(__file__).resolve().parents[2]
VERUS = os.environ.get("VERUS") or shutil.which("verus")


def test_execution_spec_closure_has_only_existing_leaf_types():
    closure = compiler_closure("spec::deterministic")
    assert {node["path"] for node in closure.values()} == {
        "src/spec.rs",
        "src/boundary/sampler.rs",
        "src/boundary/scalar.rs",
    }


def test_record_construction_does_not_use_reference_outputs():
    for root in ("initialized", "record", "transition", "compute_outputs"):
        closure = compiler_closure("proof::serving::interpretation::" + root)
        assert not [name for name in closure if name.startswith((
            "vosti_verus::proof::engine::refinement::",
            "vosti_verus::proof::serving::continuation_agreement::",
            "vosti_verus::proof::serving::refinement::",
            "vosti_verus::proof::serving::records::",
            "vosti_verus::proof::serving::trace::",
            "vosti_verus::proof::serving::consistency::",
            "vosti_verus::proof::reference::independent_batch_model::",
        )) or "reference_logits_last_row" in name]


def test_execution_spec_only_abstracts_state():
    from scripts.audit.rust_source import code_only

    source = code_only((ROOT / "src/spec.rs").read_text())
    assert re.search(r"struct\s+System\s*<\s*State\s*>", source)
    assert not re.search(r"external_body|external_type|\bassume\s*\(|\badmit\s*\(", source)


def test_public_claim_instantiates_the_generic_spec_without_certificate_premises():
    from scripts.effort.specification import accounting_claims

    claims = json.loads((ROOT / "audit/claim_surface.json").read_text())
    roots = accounting_claims(claims)
    assert [claim["item"] for claim in roots] == [
        "proof::lemma_serving_deterministic"]
    closure = compiler_closure(roots[0]["item"])
    assert "vosti_verus::spec::deterministic" in closure
    assert "vosti_verus::proof::serving::interpretation::system" in closure
    # Whole-execution legality must not carry proof witnesses or agreement
    # with reference outputs in the theorem's contract.
    assert not [name for name in closure if name.startswith((
        "vosti_verus::proof::serving::continuation_agreement::",
        "vosti_verus::proof::serving::refinement::",
        "vosti_verus::proof::serving::trace::",
        "vosti_verus::proof::serving::consistency::",
    ))]


def test_engine_transition_has_no_refinement_certificate_premise():
    # Shared structural transitions must not depend on the refinement invariant,
    # IBM steps, reference runs, or output-agreement proofs.
    closure = compiler_closure("proof::serving::transitions::event_valid")
    forbidden = (
        "vosti_verus::proof::reference::independent_batch_model::",
        "vosti_verus::proof::engine::refinement::",
        "vosti_verus::proof::serving::continuation_agreement::",
        "vosti_verus::proof::serving::refinement::",
    )
    assert not [name for name in closure if name.startswith(forbidden)]


def test_engine_event_has_no_certificate_fields():
    closure = compiler_closure("proof::serving::transitions::Event")
    assert not [name for name in closure if name.startswith((
        "vosti_verus::proof::reference::independent_batch_model::",
        "vosti_verus::proof::engine::refinement::",
        "vosti_verus::proof::serving::continuation_agreement::",
    ))]


def test_one_public_determinism_definition_and_satisfaction_entrypoint():
    from scripts.audit.rust_source import code_only

    definitions = []
    for path in (ROOT / "src").rglob("*.rs"):
        source = code_only(path.read_text())
        if re.search(r"spec\s+fn\s+(?:serving_)?deterministic\s*[<(]", source):
            definitions.append(path.relative_to(ROOT).as_posix())
        assert "lemma_sampler_traces_deterministic" not in source
    assert definitions == ["src/spec.rs"]
    entrypoint = code_only((ROOT / "src/proof.rs").read_text())
    assert re.findall(r"proof\s+fn\s+(\w+)", entrypoint) == ["lemma_serving_deterministic"]
    assert "spec::deterministic(interpretation::system(model, plan))" in entrypoint
    assert "consistency::lemma_request_consistent" in entrypoint
    assert not list((ROOT / "src/proof").glob("*.rs"))
    assert {p.name for p in (ROOT / "src/proof").iterdir() if p.is_dir() and any(p.rglob("*.rs"))} == {
        "serving", "engine", "model", "cache", "scheduler", "tensor", "reference",
    }
    library = code_only((ROOT / "src/lib.rs").read_text())
    assert set(re.findall(r"pub mod (\w+);", library)) == {
        "types", "model_config", "exec", "boundary", "spec", "proof",
    }
    assert "pub use" not in library
    assert not (ROOT / "src/model").exists()


def test_primary_serving_modules_do_not_depend_on_supplementary_trace_proof():
    from scripts.audit.rust_source import code_only

    # Check implementation imports/calls too, not only theorem contracts.
    for path in (ROOT / "src/proof/serving").glob("*.rs"):
        source = code_only(path.read_text())
        if path.name == "mod.rs":
            source = source.replace("pub(crate) mod continuation_agreement;", "")
        assert "continuation_agreement" not in source, path


@pytest.mark.skipif(VERUS is None, reason="set VERUS to check the explicit execution spec")
@pytest.mark.parametrize("case", [
    "view", "prefix", "registry", "consumer", "nondeterminism", "late_admission",
    "duplicate_id", "readmission", "unknown_output", "wrong_initial_state", "state_count",
    "different_logits", "different_tokens",
])
def test_standalone_execution_spec(tmp_path, case):
    # Scalar's file also declares parameter-interpretation functions. Supply
    # their small parameter datatype, not model or engine semantics. These
    # unused functions do not enter the public predicate's checked closure.
    header = f'''
use vstd::prelude::*;
#[path = "{ROOT / 'src/model_config.rs'}"] mod model_config;
#[path = "{ROOT / 'src/types.rs'}"] mod types;
mod boundary {{
    #[path = "{ROOT / 'src/boundary/sampler.rs'}"] pub mod sampler;
    #[path = "{ROOT / 'src/boundary/scalar.rs'}"] pub mod scalar;
}}
#[path = "{ROOT / 'src/spec.rs'}"] mod execution_spec;
use execution_spec::*;

fn main() {{}}
verus! {{
spec fn request() -> RequestInput {{
    RequestInput {{ prompt: seq![42u64], initial_sampler_state: arbitrary() }}
}}
spec fn output(token: TokenId) -> OutputRecord {{
    OutputRecord {{ logits: seq![arbitrary::<Scalar>()], token }}
}}
spec fn emit(rid: RequestId, token: TokenId) -> Step {{
    Step {{ admitted: Map::empty(), outputs: Map::empty().insert(rid, output(token)) }}
}}
spec fn admit(rid: RequestId) -> Step {{
    Step {{ admitted: Map::empty().insert(rid, request()), outputs: Map::empty() }}
}}
spec fn initial() -> Map<RequestId, RequestInput> {{ Map::empty().insert(7u64, request()) }}
spec fn system() -> System<int> {{
    System {{ init: |s: int, inputs| s == 0 && inputs == initial(),
        step: |s: int, record: Step, next: int| next == s + 1 }}
}}
spec fn run(token: TokenId) -> Execution<int> {{
    Execution {{ initial_requests: initial(), states: seq![0int, 1int],
        steps: seq![emit(7, token)] }}
}}
'''
    bodies = {
        "view": '''
            reveal_with_fuel(view, 5);
            let steps = seq![emit(7, 10), emit(8, 99), admit(9), emit(7, 11)];
            assert(view(steps, 7) == seq![output(10), output(11)]);
        ''',
        "prefix": '''
            assert(prefix_comparable(seq![output(10)], seq![output(10), output(11)]));
            assert(prefix_comparable(Seq::empty(), seq![output(99)]));
        ''',
        "registry": '''
            reveal_with_fuel(requests_after, 4);
            let steps = seq![admit(8), emit(7, 10), emit(8, 11)];
            assert(records_well_formed(initial(), steps));
            let inputs = requests_after(initial(), steps);
            assert(inputs.contains_key(7));
            assert(inputs.contains_key(8));
            assert(inputs[7] == inputs[8]);
        ''',
        "consumer": '',
        "nondeterminism": '''
            assert(legal_execution(system(), run(10)));
            assert(legal_execution(system(), run(11)));
            assert(requests_after(initial(), run(10).steps) =~= initial());
            assert(requests_after(initial(), run(11).steps) =~= initial());
            assert(view(run(10).steps, 7)[0].token == 10);
            assert(view(run(11).steps, 7)[0].token == 11);
            assert(!request_consistent(system(), run(10), run(11), 7, 7));
            assert(!deterministic(system()));
        ''',
        "duplicate_id": 'assert(records_well_formed(initial(), seq![admit(7)]));',
        "readmission": '''
            reveal_with_fuel(requests_after, 4);
            assert(records_well_formed(initial(), seq![admit(8), emit(8, 10), admit(8)]));
        ''',
        "late_admission": '''
            reveal_with_fuel(view, 5);
            reveal_with_fuel(requests_after, 5);
            let dynamic = System { init: |s: int, inputs: Map<RequestId, RequestInput>|
                s == 0 && inputs == Map::empty(),
                step: |s: int, record: Step, next: int| next == s + 1 };
            let left = Execution { initial_requests: Map::empty(), states: seq![0int, 1int, 2int, 3int, 4int],
                steps: seq![admit(8), admit(9), emit(9, 99), emit(8, 10)] };
            let right = Execution { initial_requests: Map::empty(), states: seq![0int, 1int, 2int],
                steps: seq![admit(17), emit(17, 10)] };
            assert(legal_execution(dynamic, left));
            assert(legal_execution(dynamic, right));
            assert(requests_after(left.initial_requests, left.steps)[8]
                == requests_after(right.initial_requests, right.steps)[17]);
            assert(view(left.steps, 8) == view(right.steps, 17));
            assert(request_consistent(dynamic, left, right, 8, 17));
        ''',
        "unknown_output": 'assert(records_well_formed(initial(), seq![emit(9, 10)]));',
        "wrong_initial_state": '''
            let execution = Execution { initial_requests: initial(), states: seq![1int], steps: Seq::empty() };
            assert(legal_execution(system(), execution));
        ''',
        "state_count": '''
            let execution = Execution { initial_requests: initial(), states: seq![0int], steps: seq![emit(7, 10)] };
            assert(legal_execution(system(), execution));
        ''',
        "different_logits": '''
            let left = OutputRecord { logits: seq![a], token: 10 };
            let right = OutputRecord { logits: seq![b], token: 10 };
            assert(prefix_comparable(seq![left], seq![right]));
        ''',
        "different_tokens": 'assert(prefix_comparable(seq![output(10)], seq![output(11)]));',
    }
    consumer = '''
proof fn consume(system: System<int>, left: Execution<int>, right: Execution<int>, a: u64, b: u64)
    requires
        deterministic(system), legal_execution(system, left), legal_execution(system, right),
        requests_after(left.initial_requests, left.steps).contains_key(a),
        requests_after(right.initial_requests, right.steps).contains_key(b),
        requests_after(left.initial_requests, left.steps)[a]
            == requests_after(right.initial_requests, right.steps)[b],
    ensures prefix_comparable(view(left.steps, a), view(right.steps, b)),
{ assert(request_consistent(system, left, right, a, b)); }
''' if case == "consumer" else ''
    signature = ("proof fn test(a: Scalar, b: Scalar) requires a != b {\n"
                 if case == "different_logits" else "proof fn test() {\n")
    source = tmp_path / "standalone.rs"
    source.write_text(header + signature + bodies[case] + "\n}\n" + consumer + "}\n")
    result = subprocess.run([VERUS, str(source)], text=True, capture_output=True, timeout=90)
    output = result.stdout + result.stderr
    if case in {"view", "prefix", "registry", "consumer", "nondeterminism", "late_admission"}:
        assert result.returncode == 0 and "0 errors" in result.stdout, output
    else:
        errors = re.findall(r"^error: (.+)$", result.stderr, re.MULTILINE)
        errors = [e for e in errors if not e.startswith("aborting due to")]
        assert result.returncode != 0 and errors == ["assertion failed"], output
