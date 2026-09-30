"""One exclusive-GPU aligned trial, with owned-process teardown on failure."""
import argparse
import asyncio
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import time

import httpx

from scripts.serving_benchmark.aligned.measure import measure, summarize
from scripts.serving_benchmark.server_config import clean_environment
from scripts.serving_benchmark.server_trial import gpu_pids, stop_server, wait_ready, endpoint_evidence
from scripts.serving_benchmark.multi_turn import write_new

ROOT = Path(__file__).resolve().parents[3]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--spec', type=Path, required=True)
    p.add_argument('--inputs', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    spec, data = (json.loads(f.read_text()) for f in (a.spec, a.inputs))
    gpu = spec['settings']['gpu_index']
    if type(gpu) is not int or gpu < 0 or gpu_pids(gpu):
        raise RuntimeError('selected GPU not eligible/idle')
    native = spec['execution']['engine'] == 'vosti'
    if not native:
        with socket.socket() as probe:
            # Match the server listener: old connections in TIME_WAIT must not
            # reject sequential reuse, but an active listener still conflicts.
            probe.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
            probe.bind(('127.0.0.1',spec['settings']['port']))
    output = a.output.resolve(); output.mkdir(parents=True,exist_ok=False)
    frozen = Path(spec['benchmark_frozen_root'])
    metadata = json.loads(subprocess.check_output([str(frozen/'.venv/bin/python'),'-c',
        'import site,sysconfig,json; print(json.dumps([sysconfig.get_config_var("LIBDIR"),site.getsitepackages()]))'],text=True))
    env = clean_environment(dict(os.environ), spec, python_libdir=metadata[0], python_site_packages=metadata[1])
    for key in ('TRITON_CACHE_DIR','TORCHINDUCTOR_CACHE_DIR','CUDA_CACHE_PATH'):
        Path(env[key]).mkdir(parents=True,exist_ok=True)
    if native:
        env.update(ALIGNED_INPUTS=str(a.inputs.resolve()),ALIGNED_OUTPUT=str(output))

    def interrupted(signum, frame):
        raise KeyboardInterrupt(signum)

    signal.signal(signal.SIGTERM, interrupted)
    with (output/'server.log').open('x') as log:
        server = subprocess.Popen(spec['command'], cwd=frozen if native else ROOT, env=env, stdin=subprocess.DEVNULL,
            stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:
            if native:
                # Keep ownership until teardown; do not reap before stop_server.
                from scripts.common.process_lifecycle import wait_for_child_exit_without_reaping
                wait_for_child_exit_without_reaping(server.pid, timeout_s=7200)
                if not (output/'complete.json').exists():
                    raise RuntimeError('native driver failed; inspect server.log')
            else:
                wait_ready(server,spec['base_url'],1200)
                if not list((output/'records').glob('installed-*.json')):
                    raise RuntimeError('alignment hook not installed')
                write_new(output/'server-before.json',endpoint_evidence(spec['base_url']))
                print('SERVER READY',spec['execution']['key'],'GPU',gpu,flush=True)

                async def run():
                    async with httpx.AsyncClient(timeout=900,trust_env=False,
                            limits=httpx.Limits(max_connections=8)) as client:
                        await measure(client,spec,data,output)

                asyncio.run(run())
                write_new(output/'server-after.json',endpoint_evidence(spec['base_url']))
            rows = summarize(data,output)
            print('RESULT',json.dumps(rows),flush=True)
        finally:
            stop_server(server,gpu)
    write_new(output/'trial-complete.json',dict(complete=True,finished_unix_s=time.time()))


if __name__ == '__main__':
    main()
