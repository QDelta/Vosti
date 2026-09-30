"""Run a validated job list against one owned server; wrap with GPU telemetry.

This worker never launches another GPU job while a server is alive. It retains
raw server logs, commands, endpoint evidence and each client's result. A failed
client stops the trial; no measurement is silently retried in a warm cache.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

import httpx

from scripts.common.process_lifecycle import wait_for_child_exit_without_reaping
from scripts.serving_benchmark.multi_turn import digest, write_new
from scripts.serving_benchmark.server_config import clean_environment

ROOT = Path(__file__).resolve().parents[2]


def child_exited(pid: int) -> bool:
    return os.waitid(os.P_PID, pid, os.WEXITED | os.WNOWAIT | os.WNOHANG) is not None


def gpu_pids(index: int) -> set[int]:
    text = subprocess.check_output(['nvidia-smi', '-i', str(index),
        '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True, timeout=10)
    return {int(row) for row in text.splitlines() if row.strip()}


def group_gpu_pids(leader: int, index: int) -> set[int]:
    owned = set()
    for pid in gpu_pids(index):
        try:
            if os.getpgid(pid) == leader:
                owned.add(pid)
        except ProcessLookupError:
            pass
    return owned


def stop_server(server, index: int):
    # The direct leader is kept unreaped, preventing its PID/group ID from being
    # reused. Only this fresh owned process group is ever signalled.
    ownership_error = None
    try:
        owned = group_gpu_pids(server.pid, index)
    except Exception as error:
        # A diagnostic failure must not bypass termination of our own server.
        owned, ownership_error = set(), error
    try:
        try:
            os.killpg(server.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            wait_for_child_exit_without_reaping(server.pid, timeout_s=90)
        except RuntimeError:
            os.killpg(server.pid, signal.SIGKILL)
            wait_for_child_exit_without_reaping(server.pid, timeout_s=30)
        deadline = time.monotonic() + 45
        while True:
            if not owned & gpu_pids(index):
                # Allow an in-flight telemetry lookup to finish before reaping.
                time.sleep(10.25)
                if not owned & gpu_pids(index):
                    break
            if time.monotonic() >= deadline:
                raise RuntimeError('owned server GPU contexts did not finish teardown')
            time.sleep(.1)
        if ownership_error is not None:
            raise RuntimeError('could not verify GPU ownership during teardown') from ownership_error
    except BaseException:
        try:
            os.killpg(server.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        raise
    finally:
        server.wait(timeout=30)


def wait_ready(server, base_url: str, timeout: float):
    deadline = time.monotonic() + timeout
    with httpx.Client(timeout=2) as client:
        while time.monotonic() < deadline:
            if child_exited(server.pid):
                raise RuntimeError('server exited before health check; inspect server.log')
            try:
                if client.get(base_url + '/health').status_code == 200:
                    return
            except httpx.HTTPError:
                pass
            time.sleep(.5)
    raise RuntimeError('server readiness timeout')


def endpoint_evidence(base_url: str) -> dict:
    evidence = {}
    with httpx.Client(timeout=30) as client:
        for endpoint in ('/server_info', '/metrics', '/v1/models'):
            try:
                response = client.get(base_url + endpoint)
                evidence[endpoint] = dict(status=response.status_code, text=response.text)
            except httpx.HTTPError as error:
                evidence[endpoint] = dict(error=str(error))
    return evidence


def client_command(spec: dict, job: dict, result: Path) -> list[str]:
    common = ['--tokenizer', spec['checkpoint']['path'], '--model', spec['served_name'],
              '--base-url', spec['base_url'], '--engine-label', spec['execution']['key'],
              '--context-limit', str(spec['settings']['context_limit']), '--output', str(result)]
    if job['kind'] == 'phase':
        return [sys.executable, '-m', 'scripts.serving_benchmark.phases', 'run',
                '--workload', job['workload'], *common]
    if job['kind'] in {'multi_session', 'multi_session_arrivals'}:
        command = [sys.executable, '-m', 'scripts.serving_benchmark.multi_turn', 'run',
                '--workload', job['workload'],
                '--concurrency', str(job['concurrency']), '--stagger-seconds', '0', *common]
        for path in job['warmup'] if isinstance(job['warmup'], list) else [job['warmup']]:
            command += ['--warmup-workload', path]
        if job['kind'] == 'multi_session_arrivals':
            command += ['--request-rate', str(job['request_rate']), '--arrival-process', job['arrival_process'],
                        '--arrival-seed', str(job['arrival_seed'])]
            if job.get('arrival_trace'):
                command += ['--arrival-trace', job['arrival_trace']]
        return command
    raise ValueError(f'unsupported job kind {job["kind"]}')


def validate_jobs(spec: dict, jobs: list[dict], tokenizer) -> None:
    """Validate everything before starting a model; prevent cross-job warm hits."""
    from scripts.serving_benchmark.multi_turn import resolve_arrivals, validate_warmup
    from scripts.common.tokenizers import tokenizer_artifact_sha256
    from scripts.serving_benchmark.phases import validate

    if not jobs or len({job['id'] for job in jobs}) != len(jobs):
        raise ValueError('expected nonempty jobs with unique IDs')
    artifacts = tokenizer_artifact_sha256(Path(spec['checkpoint']['path']))
    prefixes = set()
    for job in jobs:
        if not job['id'] or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_' for c in job['id']):
            raise ValueError('job ID must be a simple filename component')
        data = json.loads(Path(job['workload']).read_text())
        if data['tokenizer_artifact_sha256'] != artifacts:
            raise ValueError('job tokenizer artifacts differ from checkpoint')
        if job['kind'] == 'phase':
            validate(data, tokenizer, spec['settings']['context_limit'])
            concurrency = data['concurrency']
            prompts = [row['prompt'] for name in ('warmup', 'measured') for row in data[name]]
        elif job['kind'] in {'multi_session', 'multi_session_arrivals'}:
            paths = job['warmup'] if isinstance(job['warmup'], list) else [job['warmup']]
            stages = [json.loads(Path(path).read_text()) for path in paths]
            warmup = stages if isinstance(job['warmup'], list) else stages[0]
            validate_warmup(data, warmup, tokenizer, spec['settings']['context_limit'])
            concurrency = job['concurrency']
            if job['kind'] == 'multi_session_arrivals':
                if job['request_rate'] is None:
                    raise ValueError('arrival rate requires a completed common-rate pilot')
                resolve_arrivals(data, request_rate=job['request_rate'], arrival_process=job['arrival_process'],
                    arrival_seed=job['arrival_seed'], concurrency=concurrency, stagger_seconds=0,
                    arrival_trace=json.loads(Path(job['arrival_trace']).read_text()) if job.get('arrival_trace') else None)
            prompts = [row['initial_prompt'] for manifest in [data, *stages] for row in manifest['sessions']]
        else:
            raise ValueError('unknown job kind')
        if concurrency <= 0 or (job['kind'] != 'multi_session_arrivals'
                               and concurrency > spec['settings']['max_sequences']):
            raise ValueError('job concurrency exceeds declared server capacity')
        # Shared prefixes inside one job are intentional for multi-session replay;
        # separate jobs must still have disjoint prefixes in this server lifetime.
        job_prefixes = {tuple(tokenizer.encode(prompt)[:16]) for prompt in prompts}
        if job_prefixes & prefixes:
            raise ValueError('jobs share an initial prefix; use disjoint inputs or a fresh server')
        prefixes.update(job_prefixes)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', type=Path, required=True)
    parser.add_argument('--jobs', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--startup-timeout', type=float, default=1800)
    parser.add_argument('--validate-only', action='store_true')
    args = parser.parse_args()
    if not math.isfinite(args.startup_timeout) or args.startup_timeout <= 0:
        parser.error('startup timeout must be positive and finite')
    spec, jobs = (json.loads(path.read_text()) for path in (args.spec, args.jobs))
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(spec['checkpoint']['path'], local_files_only=True)
    validate_jobs(spec, jobs, tokenizer)
    if args.validate_only:
        print('Validated job geometry, tokenizer artifacts and cross-job disjointness; no server launched.')
        return
    if gpu_pids(spec['settings']['gpu_index']):
        raise RuntimeError('selected GPU is occupied; no server launched')
    # Fail before launch if another listener already owns the requested port.
    with socket.socket() as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        probe.bind(('127.0.0.1', spec['settings']['port']))
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    vv_python = ROOT / '.venv/bin/python'
    metadata = json.loads(subprocess.check_output([str(vv_python), '-c',
        'import site,sysconfig,json; print(json.dumps(dict(libdir=sysconfig.get_config_var("LIBDIR"), sites=site.getsitepackages())))'], text=True))
    env = clean_environment(dict(os.environ), spec, python_libdir=metadata['libdir'],
                            python_site_packages=metadata['sites'])
    write_new(output / 'inputs.json', dict(spec=spec, jobs=jobs, spec_sha256=digest(spec), jobs_sha256=digest(jobs),
        source=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()))
    for key in ('TRITON_CACHE_DIR', 'TORCHINDUCTOR_CACHE_DIR', 'CUDA_CACHE_PATH'):
        Path(env[key]).mkdir(parents=True, exist_ok=True)
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f'trial received signal {signum}')
    signal.signal(signal.SIGTERM, interrupted)
    status = dict(complete=False, jobs=[])
    with (output / 'server.log').open('xb') as log:
        server = subprocess.Popen(spec['command'], cwd=ROOT, env=env, stdout=log,
                                  stderr=subprocess.STDOUT, start_new_session=True)
        try:
            wait_ready(server, spec['base_url'], args.startup_timeout)
            write_new(output / 'server-before.json', endpoint_evidence(spec['base_url']))
            for job in jobs:
                result_path = output / f'{job["id"]}.json'
                command = client_command(spec, job, result_path)
                write_new(output / f'{job["id"]}-command.json', dict(command=command))
                with (output / f'{job["id"]}.log').open('xb') as client_log:
                    code = subprocess.call(command, cwd=ROOT, env=env, stdout=client_log, stderr=subprocess.STDOUT)
                status['jobs'].append(dict(id=job['id'], returncode=code, result=str(result_path)))
                if code:
                    raise RuntimeError(f'client job {job["id"]} failed; inspect retained result/log')
            status['complete'] = True
        except BaseException as error:
            status['error'] = repr(error)
            raise
        finally:
            try:
                if not child_exited(server.pid):
                    write_new(output / 'server-after.json', endpoint_evidence(spec['base_url']))
                stop_server(server, spec['settings']['gpu_index'])
            except BaseException as error:
                status.update(complete=False, teardown_error=repr(error))
                raise
            finally:
                write_new(output / 'status.json', status)


if __name__ == '__main__':
    main()
