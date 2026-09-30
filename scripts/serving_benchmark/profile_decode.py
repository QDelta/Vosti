"""Profile warmed native decode using an owned Nsight Systems session.

Diagnostic only: node-level CUDA Graph tracing perturbs execution. Never use
these request timings as benchmark results. Wrap this driver in GPU telemetry.
The supplied launch spec must already reference a qualified deployment bundle.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
import site
import socket
import subprocess
import sysconfig
import time

import httpx

from scripts.serving_benchmark.multi_turn import digest, write_new
from scripts.serving_benchmark.run import read_server_metrics, send_request
from scripts.serving_benchmark.server_config import clean_environment
from scripts.serving_benchmark.server_trial import ROOT, gpu_pids, stop_server, wait_ready


async def wave(spec, prompts, tokens):
    async with httpx.AsyncClient(timeout=600) as client:
        start = time.perf_counter()
        results = await asyncio.gather(*[
            send_request(client, base_url=spec['base_url'], endpoint='completions',
                model=spec['served_name'], request={**prompt, 'max_tokens': tokens},
                index=i, scheduled_offset_s=0, benchmark_start=start, require_usage=True)
            for i, prompt in enumerate(prompts)
        ])
        metrics = await read_server_metrics(client, spec['base_url'])
    if any(not row['success'] or row['output_tokens'] != tokens for row in results):
        raise RuntimeError('profile wave did not complete the requested output lengths')
    return dict(requests=results, metrics=metrics)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', type=Path, required=True)
    parser.add_argument('--workload', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--deployment-bundle', type=Path,
        help='Fresh bundle sealed to this checkout; overrides only the saved bundle path')
    parser.add_argument('--concurrency', type=int, default=4)
    parser.add_argument('--tokens', type=int, default=256)
    args = parser.parse_args()
    spec = json.loads(args.spec.read_text())
    if args.deployment_bundle:
        spec['environment']['VOSTI_DEPLOYMENT_BUNDLE'] = str(args.deployment_bundle.resolve())
    workload = json.loads(args.workload.read_text())
    if spec['execution']['engine'] != 'vosti':
        parser.error('this diagnostic requires native graph counters')
    if not 0 < args.concurrency <= min(len(workload['sessions']), spec['settings']['max_sequences']):
        parser.error('concurrency exceeds workload or server capacity')
    if args.tokens <= 1:
        parser.error('decode profiling requires more than one output token')
    if gpu_pids(spec['settings']['gpu_index']):
        raise RuntimeError('selected GPU is occupied')
    if subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT):
        raise RuntimeError('profile requires a frozen clean checkout')
    with socket.socket() as probe:
        probe.bind(('127.0.0.1', spec['settings']['port']))
    args.output.mkdir(parents=True, exist_ok=False)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(spec['checkpoint']['path'], local_files_only=True)
    prompts = [dict(prompt=s['initial_prompt'], prompt_tokens=s['initial_tokens'])
               for s in workload['sessions'][:args.concurrency]]
    if any(len(tokenizer.encode(p['prompt'])) != p['prompt_tokens']
           or p['prompt_tokens'] + args.tokens > spec['settings']['context_limit'] for p in prompts):
        raise ValueError('profile prompt-token geometry does not match checkpoint or context limit')
    session = f'vosti-decode-{os.getpid()}'
    env = clean_environment(dict(os.environ), spec,
        python_libdir=sysconfig.get_config_var('LIBDIR'), python_site_packages=site.getsitepackages())
    launch = ['nsys', 'launch', '--session-new', session, '--trace=cuda',
              '--cuda-graph-trace=node', *spec['command']]
    write_new(args.output / 'inputs.json', dict(spec=spec, workload_sha256=digest(workload),
        command=launch, tokens=args.tokens, concurrency=args.concurrency,
        source=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        diagnostic_only=True))
    with (args.output / 'server.log').open('xb') as log:
        server = subprocess.Popen(launch, cwd=ROOT, env=env, stdout=log,
                                  stderr=subprocess.STDOUT, start_new_session=True)
        collecting = False
        try:
            wait_ready(server, spec['base_url'], 600)
            for i in range(2):
                result = asyncio.run(wave(spec, prompts, args.tokens))
                write_new(args.output / f'warmup-{i}.json', result)
                print(f'WARMUP {i} complete', flush=True)
            before = result['metrics']['engine']['cuda_graph']
            subprocess.run(['nsys', 'start', '--session', session, '--sample=none',
                '--cpuctxsw=none', '--output', str(args.output / 'decode')], check=True)
            collecting = True
            result = asyncio.run(wave(spec, prompts, args.tokens))
            write_new(args.output / 'profiled-wave.json', result)
            subprocess.run(['nsys', 'stop', '--session', session], check=True)
            collecting = False
            after = result['metrics']['engine']['cuda_graph']
            if after['capture_count'] != before['capture_count']:
                raise RuntimeError('new graph capture inside profiled wave')
            write_new(args.output / 'complete.json', dict(diagnostic_only=True,
                graph_before=before, graph_after=after, status='complete'))
            print('PROFILE complete; instrumented timings are not performance results', flush=True)
        finally:
            if collecting:
                subprocess.run(['nsys', 'stop', '--session', session], check=False)
            subprocess.run(['nsys', 'shutdown', '--session', session, '--kill=sigterm'], check=False)
            stop_server(server, spec['settings']['gpu_index'])


if __name__ == '__main__':
    main()
