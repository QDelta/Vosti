"""Durable sequential agent campaigns, with file-backed logs and exit receipts."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from scripts.common.gpu_monitor import process_start_time_ticks
from scripts.common.artifacts import shared_helper_sources
from scripts.common.process_lifecycle import detached
from .prepare import save, sha


def run_jobs(root):
    plan = json.loads((root/'driver-plan.json').read_text())
    completed = []
    try:
        for job in plan['jobs']:
            for path, digest in plan['files'].items():
                if sha(path) != digest:
                    raise RuntimeError(f'changed campaign source: {path}')
            started = time.time()
            save(root/'receipts'/f'{job["mode"]}-started.json', dict(started_unix_s=started,
                command=job['command']))
            print('START', job['mode'], flush=True)
            # Inference, telemetry and grading inherit this regular file, not
            # the optional terminal reader. Losing that reader is harmless.
            with (root/f'{job["mode"]}.log').open('x') as log:
                result = subprocess.run(job['command'], cwd=plan['cwd'],
                    stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
            save(root/'receipts'/f'{job["mode"]}-exited.json', dict(returncode=result.returncode,
                started_unix_s=started, finished_unix_s=time.time()))
            if result.returncode:
                raise RuntimeError(f'{job["mode"]} exited {result.returncode}; remaining modes not started')
            output = Path(job['output'])
            for name in ['inference-complete.json','grading-complete.json','summary/measurements.json']:
                if not (output/name).is_file():
                    raise RuntimeError(f'missing completion evidence: {output/name}')
            completed.append(job['mode'])
            print('COMPLETE', job['mode'], flush=True)
        save(root/'complete.json', dict(complete=True, modes=completed, finished_unix_s=time.time()))
    except BaseException:
        save(root/'failed.json', dict(complete=False, completed_modes=completed,
            error=traceback.format_exc(), finished_unix_s=time.time()))
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--previous-pilot', type=Path)
    p.add_argument('--qualified-mode', nargs=2, action='append', metavar=('CAMPAIGN','MODE'))
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    a = p.parse_args()
    root = a.output.resolve()
    if a.worker:
        run_jobs(root)
        return
    if not a.previous_pilot or not a.qualified_mode:
        p.error('--previous-pilot and --qualified-mode are required')
    modes = [mode for _, mode in a.qualified_mode]
    if len(modes) != len(set(modes)) or any(Path(m).name != m or m in ('.','..') for m in modes):
        p.error('mode IDs must be unique directory names')
    root.mkdir(parents=True, exist_ok=False)
    cwd = Path(__file__).resolve().parents[2]
    jobs = []
    for campaign, mode in a.qualified_mode:
        output = root/mode
        command = [sys.executable, '-u', '-m', 'scripts.agent_benchmark.campaign',
            '--previous-pilot', str(a.previous_pilot.resolve()),
            '--serving-campaign', str(Path(campaign).resolve()),
            '--mode', mode, '--output', str(output)]
        jobs.append(dict(mode=mode, output=str(output), command=command))
    files = {str(f.resolve()): sha(f) for f in [*Path(__file__).parent.iterdir(), *shared_helper_sources()]
             if f.suffix in ('.py', '.jinja')}
    save(root/'driver-plan.json', dict(cwd=str(cwd), jobs=jobs, files=files))
    env = dict(os.environ, MSWEA_SILENT_STARTUP='1',
        MSWEA_GLOBAL_CONFIG_DIR=str(root/'agent-global-config'), HF_HUB_OFFLINE='1',
        TOKENIZERS_PARALLELISM='false', PYTHONUNBUFFERED='1')
    command = [sys.executable, '-u', '-m', 'scripts.agent_benchmark.launch',
               '--worker', '--output', str(root)]
    child = detached(command, root/'supervisor.log', cwd=cwd, env=env)
    receipt = dict(pid=child.pid, start_time_ticks=process_start_time_ticks(child.pid),
                   command=command, log=str(root/'supervisor.log'))
    save(root/'launcher.json', receipt)
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
