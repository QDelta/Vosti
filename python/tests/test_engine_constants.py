"""Definition ownership and lightweight imports, not support for other values."""

import ast
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

import pytest

from kernels.triton_kernels.constants import PAGE_SIZE

ROOT = Path(__file__).resolve().parents[2]


def test_one_python_page_definition_and_matching_rust_counterpart():
    definitions = []
    for directory in ("python/vosti_kernels", "kernels/triton_kernels", "scripts"):
        for path in (ROOT / directory).rglob("*.py"):
            for node in ast.parse(path.read_text()).body:
                if (isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant)
                        and any(isinstance(target, ast.Name) and target.id == "PAGE_SIZE"
                                for target in node.targets)):
                    definitions.append(path.relative_to(ROOT).as_posix())
    assert definitions == ["kernels/triton_kernels/constants.py"]
    rust = (ROOT / "src/types.rs").read_text()
    assert int(re.search(r"pub const BLOCK_SIZE: u64 = (\d+);", rust)[1]) == PAGE_SIZE


def test_python_consumers_share_page_size_and_eos_bound():
    from vosti_kernels import physical, serving_workload
    from scripts.determinism_tests import vosti_worker
    from scripts.serving_benchmark.workloads import sharegpt
    from triton_kernels import fattn_paged, fattn_paged_swa

    for consumer in (physical, serving_workload, vosti_worker, sharegpt, fattn_paged, fattn_paged_swa):
        assert consumer.PAGE_SIZE == PAGE_SIZE
    assert serving_workload.make_graph_warmup_prompt_ids.__kwdefaults__["block_size"] == PAGE_SIZE
    rust = (ROOT / "src/exec/request_state.rs").read_text()
    assert int(re.search(r"pub const MAX_EOS_TOKEN_IDS: usize = (\d+);", rust)[1]) == serving_workload.MAX_EOS_TOKEN_IDS


def test_standalone_scripts_import_without_gpu_libraries(tmp_path):
    for relative in ("scripts/determinism_tests/vosti_worker.py",
                     "scripts/serving_benchmark/workloads/sharegpt.py"):
        # No site-packages or inherited path: exercise the direct-script bootstrap.
        result = subprocess.run(
            [sys.executable, "-S", str(ROOT / relative), "--help"], cwd=tmp_path,
            env={**os.environ, "PYTHONPATH": ""}, capture_output=True, text=True,
        )
        assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("family", ("qwen3", "llama3", "gemma3", "gemma4"))
def test_importing_engine_does_not_preselect_the_kernel_tree(tmp_path, family):
    exported = tmp_path / "exported"
    shutil.copytree(ROOT / "kernels/triton_kernels", exported / "triton_kernels")
    result = subprocess.run([sys.executable, "-c", '''
import importlib
import sys
runtime = importlib.import_module(f"vosti_kernels.model_families.{sys.argv[2]}.runtime")
profile = importlib.import_module(f"vosti_kernels.model_families.{sys.argv[2]}.profile")
assert "triton_kernels" not in sys.modules
config = runtime.config_for_profile(profile.scope()["model_profiles"][0]["model"]["name"])
loaded = runtime.load_runtime(config, kernel_root=sys.argv[1])
assert loaded.model_config() == config
''', str(exported), family], cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": os.pathsep.join((str(ROOT), str(ROOT / "python"))),
             "CUDA_VISIBLE_DEVICES": ""}, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
