"""Shared benchmark helpers do not depend on an engine or the optional agent stack."""

import json
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import pytest

from scripts.common.artifacts import package_inventory, save, shared_helper_sources
from scripts.common.timing import interval_union_s, inference_accounting


def test_inference_accounting_overlap_and_absent_cache():
    assert interval_union_s([(0,3),(1,2),(2,4),(5,6)]) == 5
    row = dict(usage=dict(prompt_tokens=10, completion_tokens=3, cached_tokens=6),
               ttft_s=1, last_content_s=3, request_s=4,
               request_started_monotonic_s=0, request_finished_monotonic_s=4)
    out = inference_accounting([row,row],10)
    assert out['model_request_sum_s'] == 8 and out['model_request_active_wall_s'] == 4
    assert out['uncached_prompt_tokens'] == 8 and out['generation_fraction_of_visible_wait'] == 2/3
    assert out['subsequent_decode_tokens_estimate'] == 4
    assert out['prefill_client_proxy_tokens_s'] == 4 and out['decode_client_proxy_tokens_s'] == 1
    row['usage'].pop('cached_tokens')
    row.update(ttft_s=None,last_content_s=None)
    out = inference_accounting([row],10)
    assert out['uncached_prompt_tokens'] is None and out['prefill_client_proxy_tokens_s'] is None
    assert out['no_visible_text_request_sum_s'] == 4


def test_inference_accounting_empty_group():
    out = inference_accounting([],10)
    assert out['model_request_sum_s'] == 0
    assert out['output_tokens'] == 0


def test_atomic_json_updates_replace_the_prior_snapshot(tmp_path):
    output = tmp_path / "progress.json"
    save(output, {"status": "running"})
    save(output, {"status": "complete"})
    assert json.loads(output.read_text()) == {"status": "complete"}
    assert not output.with_suffix(".json.tmp").exists()


def test_json_serialization_failure_preserves_previous_snapshot(tmp_path):
    output = tmp_path / "progress.json"
    save(output, {"status": "running"})
    with pytest.raises(TypeError):
        save(output, {"bad": object()})
    assert json.loads(output.read_text()) == {"status": "running"}


def test_package_inventory_uses_the_selected_worker_interpreter():
    with patch("scripts.common.artifacts.subprocess.check_output", return_value='[["example", "1"]]') as run:
        assert package_inventory(Path("/worker/python")) == [["example", "1"]]
    command = run.call_args.args[0]
    assert command[:2] == ["/worker/python", "-c"]
    assert "sorted(" in command[2]


def test_snapshots_include_shared_helpers_and_future_nested_modules(tmp_path):
    for path in ("artifacts.py", "process_lifecycle.py", "timing.py", "nested/helper.py"):
        file = tmp_path / path
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text("# helper")
    (tmp_path / "not-source.json").write_text("{}")
    with patch("scripts.common.artifacts.__file__", str(tmp_path / "artifacts.py")):
        assert {p.relative_to(tmp_path).as_posix() for p in shared_helper_sources()} == {
            "artifacts.py", "process_lifecycle.py", "timing.py", "nested/helper.py"}


def test_retained_harness_sources_cover_relocated_helpers():
    from scripts.common import gpu_monitor, model_paths, telemetry, tokenizers
    sources = shared_helper_sources()
    assert len(sources) == len(set(sources))
    assert {Path(module.__file__).resolve() for module in (
        gpu_monitor, model_paths, telemetry, tokenizers)} <= set(sources)


def test_metadata_and_workload_helpers_do_not_import_gpu_stack():
    subprocess.run([sys.executable, "-c", """
import sys
from scripts.common import gpu_monitor, model_paths, telemetry, tokenizers
from scripts.serving_benchmark.workloads import sharegpt
assert not {'torch', 'triton', 'transformers'}.intersection(sys.modules)
"""], cwd=Path(__file__).resolve().parents[2], check=True)
