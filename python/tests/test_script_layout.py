"""Moved tools resolve their checkout without shell-specific Python paths."""

import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("script", (
    "checks/engine.py", "checks/reference.py", "checks/reference_suite.py",
    "checks/kv_store_effect.py", "effort/account.py",
    "common/gpu_monitor.py", "serving_benchmark/workloads/sharegpt.py",
    "verification/verify_kernel_contracts.py",
    "verification/verify_attention_interfaces.py",
    "verification/verify_rectangular_interfaces.py",
    "verification/verify_mutation_interfaces.py",
))
def test_help_outside_checkout_without_pythonpath(script, tmp_path):
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": ""}
    env.pop("PYTHONPATH", None)
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / script), "--help"],
        cwd=tmp_path, env=env, capture_output=True, text=True, check=True,
    )
    assert "usage:" in result.stdout


@pytest.mark.parametrize("script, root_key", (
    ("effort/dependencies.py", "ROOT"),
    ("checks/causal_confinement.py", "repo_root"),
    ("audit/tcb.py", "ROOT"),
    ("audit/claim_ledger.py", "ROOT"),
    ("audit/check_model_architecture_boundary.py", "ROOT"),
    ("audit/check_project_name.py", "ROOT"),
    ("audit/check_verus_trust_manifest.py", "ROOT"),
    ("deployment/common.py", "ROOT"),
    ("common/model_paths.py", "ROOT"),
    ("common/telemetry.py", "ROOT"),
))
def test_non_cli_modules_resolve_root_without_running_checks(script, root_key, tmp_path):
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": ""}
    env.pop("PYTHONPATH", None)
    # run_path does not run the guarded main: no verifier or CUDA launch.
    code = ("import runpy, sys; from pathlib import Path; "
            "namespace = runpy.run_path(sys.argv[1]); "
            "assert Path(namespace[sys.argv[2]]).resolve() == Path(sys.argv[3])")
    subprocess.run([sys.executable, "-c", code, str(ROOT / "scripts" / script), root_key, str(ROOT)],
                   cwd=tmp_path, env=env, capture_output=True, text=True, check=True)
