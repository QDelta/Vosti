//! End-to-end CPU checkpoint IO for the shared four-norm role adapters.

use pyo3::prelude::*;
use vosti_verus::boundary::model_families::gemma4::deployment::{
    bind_checkpoint, init_model_kv_caches, load_checkpoint,
};

fn initialize_python_imports() {
    // Rust runs tests on separate host threads. Resolve Transformers' lazy
    // export once before those threads import the two family fixture modules.
    // Acquire Once outside the GIL so a waiting test cannot block its owner.
    static IMPORTS: std::sync::Once = std::sync::Once::new();
    IMPORTS.call_once(|| Python::with_gil(|py| {
        py.import_bound("transformers").unwrap().getattr("AutoConfig").unwrap();
    }));
}

struct Fixture {
    directory: Py<PyAny>,
    path: String,
    sha256: String,
}

impl Drop for Fixture {
    fn drop(&mut self) {
        Python::with_gil(|py| {
            self.directory.bind(py).call_method0("cleanup").expect("remove owned tiny fixture");
        });
    }
}

fn fixture(cap: Option<f64>) -> Fixture {
    initialize_python_imports();
    Python::with_gil(|py| {
        let module = PyModule::from_code_bound(py, concat!(
            include_str!("../python/tests/test_gemma4_loader.py"),
            r#"

def rust_checkpoint_fixture(cap):
    import hashlib
    import os
    directory = tempfile.TemporaryDirectory(prefix="gemma4-rust-checkpoint-",
        dir=os.environ.get("TMPDIR"))
    root = Path(directory.name)
    Gemma4LoaderTests()._write(root)
    config = json.loads((root / "config.json").read_text())
    config["final_logit_softcapping"] = cap
    (root / "config.json").write_text(json.dumps(config))
    return directory, str(root), hashlib.sha256((root / "config.json").read_bytes()).hexdigest()
"#), "gemma4_checkpoint_fixture.py", "gemma4_checkpoint_fixture").unwrap();
        let (directory, path, sha256) = module.getattr("rust_checkpoint_fixture").unwrap()
            .call1((cap,)).unwrap().extract().unwrap();
        Fixture { directory, path, sha256 }
    })
}

#[test]
fn checkpoint_preserves_exact_config_and_raw_projection_identity() {
    let fixture = fixture(Some(30.0));
    let checkpoint = load_checkpoint(&fixture.path, "cpu");
    assert_eq!(checkpoint.model_config_sha256, fixture.sha256);
    assert_eq!(checkpoint.weights.config.global_head_dim, 8);
    assert_eq!(checkpoint.weights.config.geometry.head_dim, 4);
    assert_eq!(checkpoint.weights.config.num_global_key_value_heads, 1);
    assert_eq!(checkpoint.weights.config.geometry.num_key_value_heads, 2);
    assert_eq!(checkpoint.weights.config.global_partial_rotary_factor.bits, 0.25f64.to_bits());
    assert_eq!(checkpoint.weights.config.final_logit_softcap.unwrap().bits, 30.0f64.to_bits());
    let (weights, _perms, digest) = bind_checkpoint(checkpoint);
    assert_eq!(digest, fixture.sha256);
    assert_eq!(weights.layers.len(), 2);
    Python::with_gil(|py| {
        assert!(weights.embed_weight.inner.bind(py).is(weights.lm_head.inner.bind(py)));
        assert!(weights.layers[1].k_proj.inner.bind(py).is(weights.layers[1].v_proj.inner.bind(py)));
        assert!(!weights.layers[0].k_proj.inner.bind(py).is(weights.layers[0].v_proj.inner.bind(py)));
        for layer in &weights.layers {
            let value: f64 = layer.layer_scale.as_ref().unwrap().inner.bind(py)
                .call_method0("item").unwrap().extract().unwrap();
            assert_eq!(value, 0.25);
        }
    });
}

#[test]
fn checkpoint_preserves_disabled_softcap() {
    let fixture = fixture(None);
    let checkpoint = load_checkpoint(&fixture.path, "cpu");
    assert!(checkpoint.weights.config.final_logit_softcap.is_none());
    let (weights, _perms, digest) = bind_checkpoint(checkpoint);
    assert!(weights.config.final_logit_softcap.is_none());
    assert_eq!(digest, fixture.sha256);
}

#[test]
fn gemma4_cache_allocation_preserves_per_layer_geometry_and_independence() {
    let fixture = fixture(Some(30.0));
    let (weights, _weights_perms, _) = bind_checkpoint(load_checkpoint(&fixture.path, "cpu"));
    // Exercise empty storage and both sides of the compiled 64-slot page size.
    for capacity in [0, 1, 64, 65] {
        let (caches, _cache_perms) = init_model_kv_caches(&weights, capacity);
        assert_eq!(caches.len(), 2);
        Python::with_gil(|py| {
            let mut pointers = Vec::new();
            for (i, (k, v)) in caches.iter().enumerate() {
                for tensor in [k, v] {
                    let bound = tensor.inner.bind(py);
                    let shape: Vec<usize> = bound.getattr("shape").unwrap().extract().unwrap();
                    let (kv_heads, head_dim) = if i == 0 { (2, 4) } else { (1, 8) };
                    assert_eq!(shape, vec![capacity.div_ceil(64), 64, kv_heads, head_dim]);
                    if capacity > 0 {
                        let pointer: usize = bound.call_method0("data_ptr").unwrap().extract().unwrap();
                        assert!(!pointers.contains(&pointer));
                        pointers.push(pointer);
                    }
                }
                assert!(!k.inner.bind(py).is(v.inner.bind(py)));
            }
        });
    }
}

#[test]
#[should_panic(expected = "finite positive final_logit_softcapping")]
fn checkpoint_rejects_invalid_softcap() {
    let fixture = fixture(Some(-1.0));
    let _ = load_checkpoint(&fixture.path, "cpu");
}

#[test]
fn gemma3_checkpoint_still_loads_through_shared_role_adapter() {
    initialize_python_imports();
    let fixture = Python::with_gil(|py| {
        let module = PyModule::from_code_bound(py, concat!(
            include_str!("../python/tests/test_gemma3_loader.py"),
            r#"

def rust_checkpoint_fixture():
    import hashlib
    import os
    directory = tempfile.TemporaryDirectory(prefix="gemma3-rust-checkpoint-",
        dir=os.environ.get("TMPDIR"))
    root = Path(directory.name)
    writer = Gemma3CheckpointTests()
    writer.setUp()
    writer._write_checkpoint(root)
    return directory, str(root), hashlib.sha256((root / "config.json").read_bytes()).hexdigest()
"#), "gemma3_checkpoint_fixture.py", "gemma3_checkpoint_fixture").unwrap();
        let (directory, path, sha256) = module.getattr("rust_checkpoint_fixture").unwrap()
            .call0().unwrap().extract().unwrap();
        Fixture { directory, path, sha256 }
    });
    let _checkpoint = vosti_verus::boundary::model_families::gemma3::deployment::load_checkpoint(
        &fixture.path, "cpu",
    );
}
