"""Run the H200 core matrix sequentially from a clean, preferably frozen checkout."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from scripts.common.artifacts import package_inventory, save

from scripts.determinism_tests.protocol import (
    CAMPAIGN_SEEDS, CHECKPOINTS, EXECUTION_CONFIGS, TESTS, logical_matrix,
)
from scripts.determinism_tests.rank_divergence import analyze_summary

ROOT = Path(__file__).resolve().parents[2]


def read(path):
    return json.loads(path.read_text())


def source_identity():
    from scripts.audit.check_kernel_sources import resolve_kernel_sources
    result = {}
    if subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT):
        raise RuntimeError('framework checkout is dirty')
    result['framework'] = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    result['kernel'] = resolve_kernel_sources(ROOT)[1]
    return result


def gpu_pids(index):
    return subprocess.check_output(['nvidia-smi', '-i', str(index),
        '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True).strip()


def wait_for_idle(index):
    while gpu_pids(index):
        print('WAIT: GPU occupied; no job launched or signalled', flush=True)
        time.sleep(30)


def classify_summary(summary):
    if summary is None:
        return 'error'
    if 'relation_pass' in summary:
        if set(summary['relation_pass']) != set(TESTS):
            return 'error'
        return 'pass' if all(summary['relation_pass'].values()) else 'mismatch'
    return 'pass' if summary['status'] == 'pass' else 'mismatch'


def deferred_keys(cases, reason, planned_keys):
    """Defer explicit checkpoint/mode pairs without changing their result states."""
    if not cases:
        if reason:
            raise ValueError('deferral reason requires at least one case')
        return set()
    if not reason or not reason.strip():
        raise ValueError('deferral requires a nonempty reason')
    known = {key.rsplit('/', 1)[0] for key in planned_keys}
    if not set(cases) <= known:
        raise ValueError(f'unknown deferred checkpoint/mode pair: {set(cases) - known}')
    return {key for key in planned_keys if key.rsplit('/', 1)[0] in cases}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--stack-root', type=Path, required=True)
    parser.add_argument('--gpu-index', type=int, default=0)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--defer-case', action='append', default=[], metavar='CHECKPOINT/MODE',
                        help='Leave this pair unresolved while running other cases; never counts as tested')
    parser.add_argument('--defer-reason')
    parser.add_argument('--defer-evidence', type=Path,
                        help='Existing failure log supporting the explicit deferral')
    args = parser.parse_args()
    if args.gpu_index < 0:
        parser.error("--gpu-index must be nonnegative")
    base = args.output.resolve()
    if base.exists() and not args.resume:
        raise FileExistsError(f'{base} exists; use --resume')
    base.mkdir(parents=True, exist_ok=True)
    identity = source_identity()
    workers = {name: args.stack_root.resolve() / name / '.venv/bin/python'
               for name in ('vosti', 'vllm', 'sglang')}
    inventories = {name: package_inventory(python) for name, python in workers.items()}
    for checkpoint in CHECKPOINTS:
        for name in ('config.json', 'tokenizer_config.json', 'generation_config.json'):
            if not (Path(checkpoint.path) / name).is_file():
                raise FileNotFoundError(f'{checkpoint.key}: missing {name}')
    config = dict(source=identity, root=str(ROOT), gpu_index=args.gpu_index,
        workers={key: str(value) for key, value in workers.items()}, inventories=inventories,
        checkpoints=[asdict(value) for value in CHECKPOINTS], seeds=list(CAMPAIGN_SEEDS),
        modes=[asdict(value) for value in EXECUTION_CONFIGS], relations=list(TESTS))
    config_path = base / 'campaign.json'
    if config_path.exists() and read(config_path) != config:
        raise RuntimeError('campaign source, packages, or configuration changed; start a new campaign')
    save(config_path, config)
    save(base / 'matrix.json', logical_matrix())
    states = read(base / 'status.json') if (base / 'status.json').exists() else {}
    env = {**os.environ, 'CUDA_VISIBLE_DEVICES': str(args.gpu_index), 'CUDA_DEVICE': 'cuda:0',
           'PYTHONPATH': f'{ROOT}:{ROOT / "python"}', 'TOKENIZERS_PARALLELISM': 'false',
           'LD_LIBRARY_PATH': '/usr/lib/x86_64-linux-gnu', 'PYO3_PYTHON': str(workers['vosti']),
           'CARGO_TARGET_DIR': str(base / 'build')}
    planned = [(f'{checkpoint.key}/{mode.key}/seed-{seed}', checkpoint, mode, seed)
               for seed in CAMPAIGN_SEEDS for checkpoint in CHECKPOINTS for mode in EXECUTION_CONFIGS]
    deferred = deferred_keys(args.defer_case, args.defer_reason, [p[0] for p in planned])
    if deferred:
        if args.defer_evidence is None:
            parser.error('--defer-case requires --defer-evidence')
        evidence = args.defer_evidence.resolve()
        evidence_hash = hashlib.sha256(evidence.read_bytes()).hexdigest()
        directory = base / 'deferrals'
        directory.mkdir(exist_ok=True)
        receipt = directory / f'{time.time_ns()}.json'
        save(receipt, dict(source=identity, keys=sorted(deferred), reason=args.defer_reason,
                           evidence=str(evidence), evidence_sha256=evidence_hash))
        print('EXPLICIT DEFERRAL', receipt, 'unresolved suites:', len(deferred), flush=True)
    elif args.defer_evidence is not None:
        parser.error('--defer-evidence requires --defer-case')
    for key, _, _, _ in planned:
        states.setdefault(key, {'status': 'unrun'})
    save(base / 'status.json', states)

    def command(label, argv, *, gpu=True):
        if source_identity() != identity:
            raise RuntimeError('source changed during campaign')
        if gpu:
            wait_for_idle(args.gpu_index)
        log_dir = base / 'commands' / label
        log_dir.mkdir(parents=True, exist_ok=True)
        attempt = len(list(log_dir.glob('attempt-*.json'))) + 1
        save(log_dir / f'attempt-{attempt}.json', argv)
        print('START', label, 'attempt', attempt, flush=True)
        with (log_dir / f'attempt-{attempt}.log').open('x') as log:
            code = subprocess.call(argv, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        print('FINISH', label, 'exit', code, flush=True)
        return code

    build = ['cargo', 'build', '--release', '--features', 'openai-server']
    for family in sorted({checkpoint.model for _, checkpoint, _, _ in planned}):
        for kind in ('engine', 'server'):
            build += ['--example', f'verus_{kind}_{family}']
    if command('build', build, gpu=False):
        raise RuntimeError('build failed')
    for key, checkpoint, mode, seed in planned:
        if states[key]['status'] in ('pass', 'mismatch'):
            continue
        if key in deferred:
            print('DEFERRED', key, 'state retained:', states[key]['status'], flush=True)
            continue
        if inventories[mode.engine] != package_inventory(workers[mode.engine]):
            raise RuntimeError(f'{mode.engine} packages changed during campaign')
        bundle = base / 'deployments' / identity['framework'] / checkpoint.key
        if mode.engine == 'vosti' and not (bundle / 'deployment.json').exists():
            telemetry = base / 'deployment-telemetry' / identity['framework'] / f'{checkpoint.key}.json'
            telemetry.parent.mkdir(parents=True, exist_ok=True)
            if bundle.exists() or telemetry.exists():
                raise RuntimeError(f'Inspect incomplete deployment before retrying: {bundle}')
            prepare = [str(workers['vosti']), str(ROOT / 'scripts/common/gpu_monitor.py'),
                '--gpu-index', str(args.gpu_index), '--output', str(telemetry), '--',
                str(workers['vosti']), str(ROOT / 'scripts/prepare_deployment.py'), '--family', checkpoint.model,
                checkpoint.path, '--output', str(bundle)]
            if command(f'deploy-{checkpoint.key}', prepare):
                raise RuntimeError('deployment qualification failed; inspect retained evidence')
        output = base / 'suites' / key
        summary_path = output / 'summary.json'
        if summary_path.exists():
            summary = read(summary_path)
        else:
            states[key] = {'status': 'running', 'output': str(output), 'source': identity}
            save(base / 'status.json', states)
            argv = [str(workers['vosti']), '-m', 'scripts.determinism_tests.run',
                '--hardware', 'h200', '--gpu-index', str(args.gpu_index),
                '--model', checkpoint.model, '--model-path', checkpoint.path,
                '--execution-config', mode.key, '--seed', str(seed),
                '--worker-python', str(workers[mode.engine]), '--output', str(output)]
            if mode.engine == 'vosti':
                argv += ['--vosti-binary', str(base / f'build/release/examples/verus_engine_{checkpoint.model}'),
                         '--deployment-bundle', str(bundle)]
            if output.exists():
                argv.append('--resume')
            code = command(key, argv)
            summary = read(summary_path) if summary_path.exists() else None
            if summary is None:
                states[key] = {'status': 'error', 'returncode': code, 'output': str(output), 'source': identity,
                               'note': 'Not a numerical mismatch; inspect worker and telemetry logs.'}
                save(base / 'status.json', states)
                # Avoid multiplying a harness/configuration error across seeds.
                # Previously completed results and all unrun cells remain explicit.
                raise RuntimeError(f'{key} did not produce a valid summary; inspect before continuing')
        states[key] = dict(status=classify_summary(summary), summary=str(summary_path),
                           relation_pass=summary['relation_pass'], source=states[key].get('source', identity))
        save(base / 'status.json', states)
        if states[key]['status'] == 'error':
            raise RuntimeError(f'{key}: incomplete relation summary')
        save(output / 'rank-divergence.json', analyze_summary(key, summary_path))
        print('RESULT', key, states[key]['status'], summary['relation_pass'], flush=True)
    unresolved = {key: state for key, state in states.items() if state['status'] not in ('pass', 'mismatch')}
    if unresolved:
        save(base / 'partial.json', dict(status='partial', suites=len(planned),
                                        unresolved=unresolved, deferred=sorted(deferred)))
        print('PARTIAL: unresolved suites remain; no complete.json emitted', flush=True)
        return
    save(base / 'complete.json', {'status': 'complete', 'suites': len(planned),
                                 'logical_relations': len(planned) * len(TESTS)})


if __name__ == '__main__':
    main()
