"""One preparation CLI retains distinct CPU-only workload/trial presets."""
import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import Mock

import pytest

from scripts.serving_benchmark import prepare


PRESETS = (
    ('multi-session-replay', ['--stack-root', '/stack', '--checkpoint', prepare.CHECKPOINTS[0].key,
                             '--mode', 'vosti-padded-graph']),
    ('pilot', ['--stack-root', '/stack']),
    ('multi-workloads', []),
    ('phase-workloads', []),
    ('multi-trials', ['--stack-root', '/stack', '--workloads', '/workloads']),
    ('phase-trials', ['--stack-root', '/stack', '--workloads', '/workloads']),
    ('capacity-trials', ['--root', str(prepare.ROOT), '--stack-root', '/stack', '--build', '/build']),
)


@pytest.mark.parametrize('preset, options', PRESETS)
def test_dispatch_uses_the_selected_preset_without_work(monkeypatch, tmp_path, preset, options):
    handler = Mock()
    monkeypatch.setattr(prepare, '_' + preset.replace('-', '_'), handler)
    output = tmp_path / 'new'
    prepare.main([preset, *options, '--output', str(output)])
    handler.assert_called_once()
    args, parser = handler.call_args.args
    assert args.preset == preset and args.output == output
    assert parser is args.command_parser
    assert not output.exists()


@pytest.mark.parametrize('preset, options', PRESETS)
def test_existing_output_is_never_overwritten(tmp_path, preset, options):
    sentinel = tmp_path / 'keep'
    sentinel.write_text('previous run')
    with pytest.raises(FileExistsError):
        prepare.main([preset, *options, '--output', str(tmp_path)])
    assert sentinel.read_text() == 'previous run'
    assert list(tmp_path.iterdir()) == [sentinel]


@pytest.mark.parametrize('preset, options', PRESETS)
def test_help_outside_checkout_without_pythonpath(tmp_path, preset, options):
    env = {**os.environ, 'CUDA_VISIBLE_DEVICES': ''}
    env.pop('PYTHONPATH', None)
    result = subprocess.run([sys.executable, str(prepare.ROOT / 'scripts/serving_benchmark/prepare.py'),
        preset, '--help'], cwd=tmp_path, env=env, capture_output=True, text=True, check=True)
    assert '--output' in result.stdout


def test_pilot_replay_cannot_silently_add_phase_work(tmp_path):
    output = tmp_path / 'new'
    with pytest.raises(SystemExit) as error:
        prepare.main(['pilot', '--stack-root', '/stack', '--output', str(output),
                      '--replay-jobs', '/saved/jobs.json', '--phase-screen'])
    assert error.value.code == 2
    assert not output.exists()


def test_missing_or_unknown_preset_is_rejected():
    for argv in ([], ['unknown']):
        with pytest.raises(SystemExit) as error:
            prepare.main(argv)
        assert error.value.code == 2
