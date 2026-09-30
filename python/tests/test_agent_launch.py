"""CPU-only regression tests for terminal-independent campaign ownership."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from unittest.mock import patch, Mock

import pytest

from scripts.agent_benchmark.launch import run_jobs
from scripts.agent_benchmark.prepare import save
from scripts.common.artifacts import shared_helper_sources


def test_prepare_help_without_optional_packages():
    result = subprocess.run(
        [sys.executable, '-S', '-m', 'scripts.agent_benchmark.prepare', '--help'],
        capture_output=True, text=True, check=True,
    )
    assert '--agent-source' in result.stdout


def test_prepare_missing_packages_gives_setup_instruction(tmp_path):
    result = subprocess.run(
        [sys.executable, '-S', '-m', 'scripts.agent_benchmark.prepare',
         '--output', str(tmp_path / 'new'), '--agent-source', '/agent',
         '--model', 'test', '--mode', 'test'],
        capture_output=True, text=True,
    )
    assert result.returncode == 2
    assert 'README.md#environment' in result.stderr
    assert not (tmp_path / 'new').exists()


def test_worker_survives_launcher_exit_with_regular_file_stdout(tmp_path):
    log = tmp_path/'worker.log'
    release = tmp_path/'release'
    # The child prints after the launcher has exited and its captured pipe has
    # closed. This reproduces the lifetime boundary of a terminal/tool launch.
    worker = ('import os,stat,time; from pathlib import Path; '
              f'p=Path({str(release)!r}); '
              'print(stat.S_ISREG(os.fstat(1).st_mode),flush=True); '
              '\nwhile not p.exists(): time.sleep(.01)\n'
              'print("AFTER_LAUNCHER_EXIT",flush=True)')
    launcher = ('import os; from scripts.common.process_lifecycle import detached; '
                f'c=detached({[sys.executable,"-c",worker]!r},{str(log)!r},cwd=os.getcwd(),env=os.environ); '
                'print(c.pid,flush=True)')
    subprocess.run([sys.executable,'-c',launcher], check=True, capture_output=True, timeout=10)
    release.touch()
    deadline = time.monotonic()+5
    while 'AFTER_LAUNCHER_EXIT' not in log.read_text():
        if time.monotonic() > deadline:
            pytest.fail('detached child did not survive launcher exit')
        time.sleep(.01)
    assert log.read_text().splitlines() == ['True','AFTER_LAUNCHER_EXIT']


def test_failed_job_has_receipt_and_does_not_start_next(tmp_path):
    save(tmp_path/'driver-plan.json', dict(cwd=os.getcwd(), files={}, jobs=[
        dict(mode='first',output=str(tmp_path/'first'),command=[sys.executable,'-c','raise SystemExit(3)']),
        dict(mode='second',output=str(tmp_path/'second'),command=[sys.executable,'-c','pass'])]))
    with pytest.raises(RuntimeError,match='exited 3'):
        run_jobs(tmp_path)
    assert json.loads((tmp_path/'receipts/first-exited.json').read_text())['returncode'] == 3
    assert json.loads((tmp_path/'failed.json').read_text())['complete'] is False
    assert not (tmp_path/'receipts/second-started.json').exists()


def test_source_change_prevents_execution(tmp_path):
    source = Path(__file__)
    save(tmp_path/'driver-plan.json', dict(cwd=os.getcwd(), files={str(source):'wrong'}, jobs=[
        dict(mode='first',output=str(tmp_path/'first'),command=[sys.executable,'-c','pass'])]))
    with pytest.raises(RuntimeError,match='changed campaign source'):
        run_jobs(tmp_path)
    assert not (tmp_path/'receipts/first-started.json').exists()


def test_launcher_locks_shared_helpers_without_starting_a_worker(tmp_path):
    from scripts.agent_benchmark import launch
    output = tmp_path / 'run'
    argv = ['launch', '--previous-pilot', str(tmp_path / 'pilot'), '--output', str(output),
            '--qualified-mode', str(tmp_path / 'serving'), 'mode']
    with patch.object(sys, 'argv', argv), \
            patch.object(launch, 'detached', return_value=Mock(pid=123)), \
            patch.object(launch, 'process_start_time_ticks', return_value=456):
        launch.main()
    plan = json.loads((output / 'driver-plan.json').read_text())
    assert {str(path) for path in shared_helper_sources()} <= plan['files'].keys()
