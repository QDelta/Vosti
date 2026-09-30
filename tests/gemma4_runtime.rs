//! CPU-only host protocol tests. Mocked bundle loading here is not backend or
//! model qualification; it isolates Rust's independent retained-config checks.

use vosti_verus::model_config::{AttentionKind, DenseGeometry, FloatParameterBits};
use vosti_verus::boundary::model_families::gemma4::config::Gemma4Config;
use pyo3::prelude::*;
use std::panic::{catch_unwind, AssertUnwindSafe};
use vosti_verus::boundary::model_families::gemma4::deployment;

fn parameter(value: f64) -> FloatParameterBits {
    FloatParameterBits { bits: value.to_bits() }
}

fn config() -> Gemma4Config {
    Gemma4Config {
        geometry: DenseGeometry {
            vocab_size: 262144, hidden_size: 5376, intermediate_size: 21504,
            num_layers: 60, num_attention_heads: 32, num_key_value_heads: 16,
            head_dim: 256, max_position_embeddings: 262144,
        },
        num_global_key_value_heads: 4, global_head_dim: 512,
        rms_norm_epsilon: parameter(1e-6), sliding_window: 1024,
        local_rope_theta: parameter(1e4), global_rope_theta: parameter(1e6),
        global_rope_factor: parameter(1.0), global_partial_rotary_factor: parameter(0.25),
        attention_k_eq_v: true, final_logit_softcap: Some(parameter(30.0)),
    }
}

fn kinds() -> Vec<AttentionKind> {
    (0..60).map(|i| if (i + 1) % 6 == 0 { AttentionKind::Full }
        else { AttentionKind::SlidingWindow }).collect()
}

#[test]
fn runtime_host_protocol_preserves_exact_config_and_rejects_drift() {
    let staged = deployment::init_staged_runtime_for_tests("gemma-4-31b-it-text");
    assert!(!deployment::reports_backend_qualified(&staged));

    Python::with_gil(|py| {
        let fixture = pyo3::types::PyModule::from_code_bound(py, r#"
import contextlib
import copy
import math
from types import SimpleNamespace
from unittest.mock import patch
from vosti_kernels.model_families.gemma4 import runtime

@contextlib.contextmanager
def patched(field=None, no_softcap=False):
    original = runtime.config_for_profile('gemma-4-31b-it-text')
    retained = copy.deepcopy(original)
    report = dict(architecture='gemma4_text', backend_qualified=True,
                  deployment_sha256='a' * 64)
    if no_softcap:
        retained['final_logit_softcapping'] = None
    if field in retained:
        value = retained[field]
        if isinstance(value, bool):
            retained[field] = not value
        elif isinstance(value, int):
            retained[field] = value + 1
        elif isinstance(value, float):
            retained[field] = math.nextafter(value, math.inf)
        else:
            retained[field] = list(reversed(value))
    elif field == 'wrong_architecture':
        report['architecture'] = 'gemma3_text'
    elif field == 'unqualified':
        report['backend_qualified'] = False
    elif field == 'missing_identity':
        report['deployment_sha256'] = None
    fake = SimpleNamespace(model_config=lambda: copy.deepcopy(retained),
                           report=lambda: copy.deepcopy(report))
    with patch.object(runtime, 'config_from_bundle', return_value=original), \
         patch.object(runtime, 'load_qualified_runtime', return_value=fake) as loader:
        yield
        assert loader.call_count == 1
        assert loader.call_args.kwargs == dict(deployment_bundle='mock-bundle',
                                               model_config_sha256='b' * 64)

fields = list(runtime.config_for_profile('gemma-4-31b-it-text')) + ['wrong_architecture', 'unqualified', 'missing_identity']
"#, "gemma4_runtime_host_fixture.py", "gemma4_runtime_host_fixture").unwrap();

        let fields: Vec<String> = fixture.getattr("fields").unwrap().extract().unwrap();
        let cases = std::iter::once((None, false))
            .chain(std::iter::once((None, true)))
            .chain(fields.iter().map(|field| (Some(field.as_str()), false)));
        for (field, no_softcap) in cases {
            let context = fixture.getattr("patched").unwrap().call1((field, no_softcap)).unwrap();
            context.call_method0("__enter__").unwrap();
            let outcome = catch_unwind(AssertUnwindSafe(|| {
                let mut expected = config();
                if no_softcap { expected.final_logit_softcap = None; }
                let runtime = deployment::init_qualified_runtime(
                    "mock-bundle", &"b".repeat(64), expected, kinds());
                assert!(deployment::reports_backend_qualified(&runtime));
            }));
            // Restore Python patches even when the Rust boundary panics.
            context.call_method1("__exit__", (py.None(), py.None(), py.None())).unwrap();
            assert_eq!(outcome.is_ok(), field.is_none(), "case {field:?}, no_softcap={no_softcap}");
        }
    });
}
