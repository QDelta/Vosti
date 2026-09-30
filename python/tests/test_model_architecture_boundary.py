"""Architecture checks cover interfaces and dependencies, not file symmetry."""

import json
import shutil

import pytest

from scripts.audit.check_model_architecture_boundary import (
    ROOT, architecture_boundary_errors, _is_configuration_data_import,
    _normalize_family_signature, _python_deployment_binding, _python_inherited_operation,
)


@pytest.fixture
def checkout(tmp_path):
    # Use real sources instead of maintaining a second synthetic model tree.
    for relative in ("src", "audit", "python/vosti_kernels"):
        shutil.copytree(ROOT / relative, tmp_path / relative,
                        ignore=shutil.ignore_patterns("__pycache__"))
    manifest = json.loads((tmp_path / "audit/model_architecture_ownership.json").read_text())
    for entry in manifest["kernel_interfaces"].values():
        path = tmp_path / entry["generator"]
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / entry["generator"], path)
    return tmp_path


def append(root, relative, text):
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text((path.read_text() if path.exists() else "") + text)


def mutate_json(root, relative, mutate):
    path = root / relative
    data = json.loads(path.read_text())
    mutate(data)
    path.write_text(json.dumps(data))


def test_repository_respects_closed_dispatch_surface():
    assert architecture_boundary_errors() == []


def test_comments_strings_benchmarks_and_helper_names_need_no_exceptions(checkout):
    append(checkout, "src/exec/engine.rs", '\n// Gemma3: ModelArchitecture::Gemma3Text\n'
           '/* use crate::boundary::model_families::gemma3::weights; */\n'
           'const EXAMPLE: &str = "ModelRuntime::Qwen3";\n')
    append(checkout, "scripts/gemma3_experiment.py", '# An intentionally Gemma3-only experiment.\n')
    append(checkout, "src/boundary/model_families/gemma3/shape_helper.rs", '// Private helper.\n')
    append(checkout, "src/proof/model/families/dense_swiglu/geometry.rs", '// Shared helper.\n')
    assert architecture_boundary_errors(checkout) == []


@pytest.mark.parametrize("path,text,error", [
    ("src/exec/engine.rs", "fn example() { ModelArchitecture::FutureFamily; }", "generic Engine"),
    ("src/exec/cache_scheduler/mod.rs", "fn example() { ModelWeightsArchitectureRepr::Qwen3; }", "payload dispatch"),
    ("src/proof.rs", "use crate::boundary::model_families::gemma3::weights;", "family-neutral source imports"),
    ("src/exec/model.rs", "use crate::exec::model_families::gemma3::readiness;", "family-neutral source imports"),
    ("src/boundary/model_families/qwen3/mod.rs", "use crate::boundary::model_families::llama3::weights;", "another family"),
    ("python/vosti_kernels/model_families/qwen3/runtime.py", "from ..llama3 import loader", "another family"),
    ("python/vosti_kernels/model_families/qwen3/runtime.py", "import vosti_kernels.model_families.gemma3.loader", "another family"),
    ("python/vosti_kernels/dense_runtime.py", "from .model_families.gemma4 import loader", "another family"),
    ("src/boundary/tensor_runtime.rs", "pub struct Qwen3LayerWeights {}", "physical weight declaration"),
    ("src/boundary/backend_certificates/support.rs", "#[verifier::external_body] pub proof fn example() {}", "trusted theorem"),
    ("src/boundary/backend_certificates/extra.rs", "pub proof fn example() {}", "inventory is not closed"),
])
def test_forbidden_boundaries_are_rejected(checkout, path, text, error):
    append(checkout, path, "\n" + text + "\n")
    assert any(error in e for e in architecture_boundary_errors(checkout))


@pytest.mark.parametrize("path,operation", [
    ("src/boundary/model_families/gemma3/mod.rs", "weights_extension_repr_of"),
    ("src/proof/model/families/dense_swiglu/mod.rs", "forward_logits_repr"),
    ("src/boundary/model_families/qwen3/weights.rs", "model_weights_bound"),
    ("python/vosti_kernels/model_families/gemma4/runtime.py", "load_qualified_runtime"),
    ("python/vosti_kernels/model_families/llama3/loader.py", "inspect_text_checkpoint"),
])
def test_missing_api_is_rejected_even_with_a_commented_declaration(checkout, path, operation):
    file = checkout / path
    text = file.read_text()
    keyword = "def" if path.endswith(".py") else "fn"
    assert f"{keyword} {operation}(" in text
    text = text.replace(f"{keyword} {operation}(", f"{keyword} removed_operation(")
    comment = "#" if path.endswith(".py") else "//"
    file.write_text(text + f"\n{comment} {keyword} {operation}() {{}}\n")
    assert any(operation in e for e in architecture_boundary_errors(checkout))


def test_family_must_use_common_deployment_assembly(checkout):
    path = checkout / "src/boundary/model_families/gemma3/deployment.rs"
    text = path.read_text().replace("COMMON_DEPLOYMENT::assemble_qualified_model(", "unreviewed_assembly(")
    path.write_text(text)
    assert any("bypasses common assembly" in e for e in architecture_boundary_errors(checkout))


@pytest.mark.parametrize("relative", [
    "src/exec/model_families/gemma3/mod.rs",
    "src/boundary/model_families/qwen3/weights.rs",
    "python/vosti_kernels/model_families/llama3/physical.py",
])
def test_missing_family_adapter_is_rejected(checkout, relative):
    (checkout / relative).unlink()
    assert architecture_boundary_errors(checkout)


def test_composition_mapping_covers_every_family(checkout):
    mutate_json(checkout, "audit/model_architecture_ownership.json",
                lambda m: m["family_compositions"].pop("gemma3"))
    assert any("map every family" in e for e in architecture_boundary_errors(checkout))


def test_runtime_scope_must_have_exact_profile_and_launch_records(checkout):
    mutate_json(checkout, "python/vosti_kernels/model_families/gemma3/scope.json",
                lambda m: m["model_profiles"][0].update(extra=True))
    assert any("qualification scope" in e for e in architecture_boundary_errors(checkout))


def test_configuration_imports_exclude_helpers_and_wildcards():
    prefix = "boundary::model_families::example::config::"
    assert _is_configuration_data_import(prefix + "{ExampleConfig, EXAMPLE_EPSILON}", {"example"})
    for name in ("*", "attention_geometry", "{ExampleConfig, *}", "ExampleConfig as Alias"):
        assert not _is_configuration_data_import(prefix + name, {"example"})


def test_normalized_signature_preserves_type_shape():
    families = {"example_dense", "another"}
    left = "(perms: &RT::ModelWeightsPerms) -> Example_denseModelWeightsExtensionRepr"
    right = "(perms: &RT::ModelWeightsPerms) -> AnotherTextModelWeightsExtensionRepr"
    assert _normalize_family_signature(left, families) == _normalize_family_signature(right, families)
    assert _normalize_family_signature(left, families) != _normalize_family_signature(right.replace("&RT", "RT"), families)


def test_generic_deployment_binding_checks_operation_and_architecture():
    prefix = ("from ... import deployment as shared\nfrom functools import partial\n"
              "ARCHITECTURE = shared.DeploymentArchitecture()\n")
    binding = "seal_candidate = partial(shared.seal_candidate, ARCHITECTURE)\n"
    assert _python_deployment_binding(prefix + binding, "seal_candidate")
    for wrong in (binding.replace("shared.seal_candidate", "shared.validate_candidate"),
                  binding.replace(", ARCHITECTURE)", ", other)"), "# " + binding):
        assert not _python_deployment_binding(prefix + wrong, "seal_candidate")


def test_runtime_inheritance_is_static_and_rejects_cycles(tmp_path):
    base = tmp_path / "base.py"
    child = tmp_path / "runtime.py"
    base.write_text("raise RuntimeError('must not execute')\nclass Base:\n    def report(self): pass\n")
    child.write_text("from .base import Base\nclass Runtime(Base): pass\n")
    assert _python_inherited_operation(child, tmp_path, "report")
    assert not _python_inherited_operation(child, tmp_path, "missing")
    base.write_text("from .runtime import Runtime\nclass Base(Runtime): pass\n")
    assert not _python_inherited_operation(child, tmp_path, "report")
