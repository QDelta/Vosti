"""Exercise resume admission through the real CLIs, stopping before execution."""
import importlib
import json
import sys
from unittest.mock import Mock

import pytest


class ExecutionReached(RuntimeError):
    pass


@pytest.fixture(params=['campaign', 'report_campaign', 'expanded_suite'])
def controller(request, monkeypatch, tmp_path):
    name = request.param
    module = importlib.import_module(f'scripts.determinism_tests.{name}')
    output = tmp_path / 'run'
    args = ['--output', str(output)]
    boundary = Mock(side_effect=ExecutionReached)

    def check_output(command, **kwargs):
        if command[:2] == ['git', 'rev-parse']:
            return 'source'
        assert command[:2] == ['git', 'status'] or command[0] == 'nvidia-smi'
        return ''

    monkeypatch.setattr(module.subprocess, 'check_output', check_output)
    if name == 'campaign':
        args += ['--stack-root', '/stack']
        monkeypatch.setattr(module, 'CHECKPOINTS', ())
        monkeypatch.setattr(module, 'logical_matrix', lambda: [])
        monkeypatch.setattr(module, 'source_identity', lambda: dict(framework='source', kernel='kernel'))
        monkeypatch.setattr(module, 'package_inventory', lambda _: [])
        monkeypatch.setattr(module.subprocess, 'call', boundary)
        filename, source_field, config_field = 'campaign.json', 'source', 'inventories'
    elif name == 'report_campaign':
        mode = next(mode.key for mode in module.EXECUTION_CONFIGS if mode.engine == 'vllm')
        args += ['--stack-root', '/stack', '--mode', mode]
        monkeypatch.setattr(module, 'REPORT_MODELS', ())
        monkeypatch.setattr(module, 'table_matrix', lambda: [])
        monkeypatch.setattr(module, 'package_inventory', lambda _: [])
        monkeypatch.setattr(module, 'ThreadPoolExecutor', boundary)
        filename, source_field, config_field = 'inputs.json', 'source', 'inventories'
    else:
        args += ['--pilot', '--worker-python', sys.executable]
        monkeypatch.setattr(module, 'deterministic_prompt', lambda path, length, seed: [seed] * length)
        monkeypatch.setattr(module, 'SuiteRunner', boundary)
        filename, source_field, config_field = 'inputs.json', 'controller_sha256', 'seed'

    def run(resume=False):
        monkeypatch.setattr(sys, 'argv', [name, *args, *(['--resume'] if resume else [])])
        module.main()

    with pytest.raises(ExecutionReached):
        run()
    boundary.reset_mock()
    raw = output / 'retained-result.bin'
    raw.write_bytes(b'original raw evidence')
    return run, output / filename, raw, boundary, source_field, config_field


@pytest.mark.parametrize('change', [None, 'source', 'configuration', 'extra-field'])
def test_resume_requires_exact_manifest_without_rewriting_evidence(controller, change):
    run, manifest, raw, boundary, source_field, config_field = controller
    if change:
        previous = json.loads(manifest.read_text())
        field = source_field if change == 'source' else config_field if change == 'configuration' else 'extra'
        previous[field] = 'changed'
        manifest.write_text(json.dumps(previous))
    before = manifest.read_bytes(), raw.read_bytes()
    if change:
        with pytest.raises(RuntimeError, match='changed; start a new'):
            run(resume=True)
        boundary.assert_not_called()
    else:
        with pytest.raises(ExecutionReached):
            run(resume=True)
        boundary.assert_called_once()
    assert (manifest.read_bytes(), raw.read_bytes()) == before
