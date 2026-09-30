"""CPU-only coverage of the shared launch and deployment entry points."""

import importlib
import os
import subprocess
import sys
from unittest.mock import patch

import pytest

from scripts import launch, prepare_deployment
from scripts.common import model_paths


@pytest.mark.parametrize("family", model_paths.supported_families())
@pytest.mark.parametrize("kind", ("engine", "server"))
def test_launch_preserves_build_mode_and_selects_an_existing_example(family, kind):
    command = launch.launch_command(kind, family, [])
    expected = ["uv", "run", "--locked", "cargo", "run"]
    if kind == "server":
        expected += ["--release", "--features", "openai-server"]
    assert command == [*expected, "--example", f"verus_{kind}_{family}"]
    assert (launch.ROOT / "examples" / f"verus_{kind}_{family}.rs").is_file()


def test_launch_preserves_environment_and_replaces_process():
    family = model_paths.supported_families()[0]
    original = {"LD_LIBRARY_PATH": "/existing/lib", "PYTHONPATH": "/existing/python",
                "CUDA_VISIBLE_DEVICES": "3", "MODEL_PATH": "/model with spaces",
                "VOSTI_DEPLOYMENT_BUNDLE": "/sealed bundle"}
    with patch.dict(os.environ, original, clear=True), \
            patch.object(launch.sysconfig, "get_config_var", return_value="/python/lib"), \
            patch.object(launch.os, "chdir") as chdir, \
            patch.object(launch.os, "execvpe") as execute:
        launch.main(["--kind", "engine", "--family", family, "--", "--flag", "value with spaces"])
        assert dict(os.environ) == original
    chdir.assert_called_once_with(launch.ROOT)
    executable, command, env = execute.call_args.args
    assert executable == "uv"
    assert command[-3:] == ["--", "--flag", "value with spaces"]
    assert env == {**original, "LD_LIBRARY_PATH": "/python/lib:/existing/lib",
                   "PYTHONPATH": f"{launch.ROOT}:{launch.ROOT / 'python'}:/existing/python",
                   "VOSTI_FRAMEWORK_ROOT": str(launch.ROOT)}


def test_launch_has_no_empty_search_path_components():
    with patch.object(launch.sysconfig, "get_config_var", return_value="/python/lib"):
        env = launch.launch_environment({"LD_LIBRARY_PATH": "", "PYTHONPATH": ""})
    assert env["LD_LIBRARY_PATH"] == "/python/lib"
    assert env["PYTHONPATH"] == f"{launch.ROOT}:{launch.ROOT / 'python'}"


def test_missing_python_library_is_an_error():
    with patch.object(launch.sysconfig, "get_config_var", return_value=None), \
            pytest.raises(RuntimeError, match="library directory"):
        launch.launch_environment({})


@pytest.mark.parametrize("family", model_paths.supported_families())
def test_preparation_binds_existing_family_policy_without_running_it(family):
    architecture = prepare_deployment.preparation_architecture(family)
    candidate = importlib.import_module(f"scripts.deployment.model_families.{family}")
    deployment = importlib.import_module(f"vosti_kernels.model_families.{family}.deployment")
    assert architecture.family_label == candidate.CANDIDATE_ARCHITECTURE.family_label
    assert architecture.report_schema == deployment.REPORT_SCHEMA
    assert architecture.prepare_candidate is candidate.prepare_candidate
    assert architecture.seal_candidate is deployment.seal_candidate


def test_preparation_cli_passes_model_and_output_to_shared_pipeline(tmp_path, capsys):
    family = model_paths.supported_families()[0]
    architecture = object()
    bundle = {"candidate": {"model": {"catalog_name": "synthetic"},
                            "environment": {"devices": [{"name": "test device"}]}},
              "report": {"results": [1, 2]}}
    with patch.object(prepare_deployment, "preparation_architecture", return_value=architecture) as select, \
            patch("scripts.deployment.common.prepare_and_seal_deployment", return_value=bundle) as prepare:
        prepare_deployment.main(["--family", family, "/checkpoint", "--output", str(tmp_path)])
    select.assert_called_once_with(family)
    prepare.assert_called_once_with(architecture, "/checkpoint", tmp_path)
    assert "[BACKEND PROBES] 2 passed on test device" in capsys.readouterr().out


@pytest.mark.parametrize("module", (launch, prepare_deployment))
def test_cli_rejects_unknown_family_before_execution(module):
    with pytest.raises(SystemExit) as error:
        module.main(["--family", "../unknown"])
    assert error.value.code == 2


def test_family_discovery_uses_installed_scopes(tmp_path):
    scopes = tmp_path / "python/vosti_kernels/model_families"
    for name in ("new_dense", "another_dense"):
        (scopes / name).mkdir(parents=True)
        (scopes / name / "scope.json").write_text("{}")
    (scopes / "not_a_family").mkdir()
    with patch.object(model_paths, "ROOT", tmp_path):
        assert model_paths.supported_families() == ("another_dense", "new_dense")


@pytest.mark.parametrize("script", ("launch.py", "prepare_deployment.py"))
def test_help_works_outside_repo_without_pythonpath_or_gpu(script, tmp_path):
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": ""}
    env.pop("PYTHONPATH", None)
    result = subprocess.run([sys.executable, str(launch.ROOT / "scripts" / script), "--help"],
                            cwd=tmp_path, env=env, capture_output=True, text=True, check=True)
    assert "--family" in result.stdout
    assert all(family in result.stdout for family in model_paths.supported_families())
