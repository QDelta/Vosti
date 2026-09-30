"""Native orchestration has explicit subcommands, without invoking GPU work."""
from unittest.mock import Mock

import pytest

from scripts.determinism_tests import native_report


OPTIONS = {
    'prepare': ['--stack-root', '/stack'],
    'run': ['--checkpoint', native_report.NATIVE_CHECKPOINTS[-1].key, '--gpu-index', '3', '--worker-root', '/worker',
            '--worker-python', '/worker/python', '--binary', '/binary', '--deployment-bundle', '/bundle'],
    'smoke': ['--checkpoint', native_report.REPORT_MODELS[0], '--worker-python', '/worker/python',
              '--binary', '/binary', '--deployment-bundle', '/bundle'],
}


@pytest.mark.parametrize('command', OPTIONS)
def test_dispatch(monkeypatch, tmp_path, command):
    handler = Mock()
    monkeypatch.setattr(native_report, '_' + command, handler)
    native_report.main([command, *OPTIONS[command], '--output', str(tmp_path / 'new')])
    handler.assert_called_once()
    args, parser = handler.call_args.args
    assert args.command == command and args.command_parser is parser
    assert not (tmp_path / 'new').exists()


@pytest.mark.parametrize('command', OPTIONS)
def test_help_does_not_launch_work(command):
    with pytest.raises(SystemExit) as error:
        native_report.main([command, '--help'])
    assert error.value.code == 0


def test_missing_subcommand_rejected():
    with pytest.raises(SystemExit) as error:
        native_report.main([])
    assert error.value.code == 2


def test_smoke_failure_prevents_full_report(monkeypatch, tmp_path):
    monkeypatch.setattr(native_report.subprocess, 'run', Mock())
    monkeypatch.setattr(native_report.subprocess, 'check_output', Mock(return_value='source'))
    monkeypatch.setattr(native_report, 'gpu_pids', Mock(return_value=set()))
    monkeypatch.setattr(native_report, 'make_plan', Mock(return_value={}))
    monkeypatch.setattr(native_report, 'source_identity', Mock(return_value='worker'))
    monkeypatch.setattr(native_report, 'package_inventory', Mock(return_value=[]))
    monkeypatch.setattr(native_report, 'file_hash', Mock(return_value='hash'))
    monkeypatch.setattr(native_report, 'SuiteRunner', Mock())
    smoke = Mock(return_value={'status': 'mismatch'})
    report = Mock()
    monkeypatch.setattr(native_report, 'run_smoke', smoke)
    monkeypatch.setattr(native_report, 'run_report', report)
    with pytest.raises(RuntimeError, match='full suite was not started'):
        native_report.main(['run', *OPTIONS['run'], '--output', str(tmp_path / 'new')])
    smoke.assert_called_once()
    report.assert_not_called()
    assert (tmp_path / 'new/smoke/summary.json').exists()
    assert (tmp_path / 'new/error.json').exists()
