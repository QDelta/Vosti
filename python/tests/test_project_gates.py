"""Orchestration preserves fail-closed verification and explicit GPU admission."""

from pathlib import Path
from unittest.mock import Mock

import pytest

from scripts import project


@pytest.fixture
def gates(monkeypatch, tmp_path):
    monkeypatch.setenv("VERUS_CLAIM_TARGET_DIR", str(tmp_path / "build"))
    monkeypatch.delenv("VERUS_CLAIM_VIR_LOG_DIR", raising=False)
    gate = project.Gates()
    gate.require_verus = Mock()
    gate.architecture = Mock()
    gate.run = Mock()
    return gate


def test_engine_deletes_stale_vir_before_verifier_and_checks_fresh_output(gates):
    gates.vir_dir.mkdir(parents=True)
    vir = gates.vir_dir / "crate-simple.vir"
    vir.write_text("stale focused result")

    def run(*args, **kwargs):
        if args[0] == gates.cargo_verus:
            assert not vir.exists()
            assert "--fwd-verus-args-to" in args
            vir.write_text("fresh whole-crate result")

    gates.run.side_effect = run
    gates.verify_engine()
    calls = gates.run.call_args_list
    assert calls[0].args[:4] == ("cargo", "clean", "-p", "vosti-verus")
    assert calls[-1].args[1:] == (project.ROOT / "scripts/audit/check_verus_trust_manifest.py", vir)
    assert (gates.target / "CACHEDIR.TAG").read_bytes().startswith(
        b"Signature: 8a477f597d28d172789f06886806bc55")


@pytest.mark.parametrize("output", [None, ""])
def test_engine_rejects_missing_or_empty_vir(gates, output):
    def run(*args, **kwargs):
        if args[0] == gates.cargo_verus and output is not None:
            (gates.vir_dir / "crate-simple.vir").write_text(output)

    gates.run.side_effect = run
    with pytest.raises(ValueError, match="fresh, nonempty"):
        gates.verify_engine()
    assert len(gates.run.call_args_list) == 2


def test_failure_stops_aggregate_gate(gates):
    gates.verify = Mock(side_effect=RuntimeError("failed"))
    gates.examples = Mock()
    with pytest.raises(RuntimeError):
        gates.check()
    gates.examples.assert_not_called()


def test_complete_check_preserves_all_old_cpu_stages(gates):
    order = []
    for name in ("verify", "examples", "test", "diagnostics"):
        setattr(gates, name, lambda name=name: order.append(name))
    gates.python = lambda script, *args: order.append((script, args))
    gates.check()
    assert order == ["verify", "examples", "test", ("scripts/audit/claim_ledger.py", ()),
                     "diagnostics", ("scripts/audit/tcb.py", ("--check",))]


def test_verify_includes_both_provers(gates):
    order = []
    gates.verify_engine = lambda: order.append("engine")
    gates.verify_kernels = lambda: order.append("kernels")
    gates.verify()
    assert order == ["engine", "kernels"]


def test_rust_gate_runs_shared_http_tests_and_builds_all_examples(gates):
    gates.test("rust")
    assert gates.env["CUDA_VISIBLE_DEVICES"] == ""
    assert [call.args for call in gates.run.call_args_list] == [
        ("cargo", "test", "--locked", "--features", "openai-server",
         "--lib", "--tests", "--", "--nocapture"),
        ("cargo", "build", "--locked", "--features", "openai-server", "--examples"),
    ]


def test_generate_always_uses_full_inventory(gates):
    gates.env["FAMILY"] = "gemma4"
    gates.generate()
    assert gates.run.call_args_list[1].args[1:] == (
        project.ROOT / "scripts/verification/verify_kernel_contracts.py", "--write-generated")
    assert gates.run.call_args_list[2].args[1:] == (
        project.ROOT / "scripts/audit/claim_ledger.py", "--write")


def test_missing_verus_is_an_error(monkeypatch):
    monkeypatch.setenv("VERUS", "/nonexistent/verus")
    with pytest.raises(ValueError, match="must not skip"):
        project.Gates().require_verus()


def test_all_independent_diagnostics_are_retained(gates):
    gates.diagnostics()
    invoked = {Path(call.args[1]) for call in gates.run.call_args_list}
    assert set((project.ROOT / "scripts").glob("verify_*_kernel_races.py")) <= invoked
    assert project.ROOT / "kernels/scripts/verify_suffix_independence.py" in invoked


def test_cpu_suites_disable_cuda_and_include_kernel_regressions(gates):
    gates.env["CUDA_VISIBLE_DEVICES"] = "3"
    gates.test()
    assert gates.env["CUDA_VISIBLE_DEVICES"] == ""
    assert [call.args[0] for call in gates.run.call_args_list] == [
        "cargo", "cargo", project.sys.executable, project.sys.executable, project.sys.executable,
        project.sys.executable]
    assert gates.run.call_args_list[2].args[1] == project.ROOT / "scripts/effort/dependencies.py"
    assert gates.run.call_args_list[-1].kwargs["cwd"] == project.ROOT / "kernels"


@pytest.mark.parametrize("missing", ["FAMILY", "CUDA_VISIBLE_DEVICES", "MODEL_PATH", "VOSTI_DEPLOYMENT_BUNDLE"])
def test_gpu_missing_inputs_rejected_before_any_subprocess(gates, tmp_path, missing):
    gates.env.update(FAMILY="gemma4", CUDA_VISIBLE_DEVICES="3",
                     MODEL_PATH=str(tmp_path), VOSTI_DEPLOYMENT_BUNDLE=str(tmp_path))
    gates.env.pop(missing)
    with pytest.raises(ValueError, match=missing):
        gates.test_gpu()
    gates.run.assert_not_called()


@pytest.mark.parametrize("family", ["qwen3", "gemma3", "llama3", "gemma4"])
def test_families_use_same_sequential_gpu_gate(gates, tmp_path, family):
    gates.env.update(FAMILY=family, CUDA_VISIBLE_DEVICES="3",
                     MODEL_PATH=str(tmp_path), VOSTI_DEPLOYMENT_BUNDLE=str(tmp_path))
    gates.test_gpu()
    calls = gates.run.call_args_list
    assert [Path(call.args[1]).name for call in calls] == [
        "check_kernel_sources.py", "kv_store_effect.py", "engine.py",
        "reference.py", "causal_confinement.py"]
    assert all(call.args[-1] == family for call in calls[1:4])


@pytest.mark.parametrize("makelevel", ["0", "1"])
def test_makefile_public_surface_and_clean_preserves_environment(monkeypatch, makelevel):
    import subprocess

    monkeypatch.setenv("MAKELEVEL", makelevel)
    makefile = (project.ROOT / "Makefile").read_text()
    assert ".DEFAULT_GOAL := help" in makefile
    assert "make -C" not in makefile
    result = subprocess.run(["make", "--no-print-directory", "-n", "clean"], cwd=project.ROOT,
                            check=True, capture_output=True, text=True)
    assert result.stdout.strip() == "cargo clean"
