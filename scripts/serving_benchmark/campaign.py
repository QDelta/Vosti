"""Execute prepared performance trials sequentially against a frozen checkout."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from scripts.common.telemetry import load_complete_telemetry
from scripts.common.artifacts import package_inventory, save
from scripts.serving_benchmark.server_trial import gpu_pids


def read(path: Path):
    return json.loads(path.read_text())


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def job_input_files(job):
    """Bind every workload, ordered warmup stage and saved arrival trace."""
    files = {}
    for key in ('workload', 'warmup', 'arrival_trace'):
        if key in job:
            paths = job[key] if isinstance(job[key], list) else [job[key]]
            for path in paths:
                files[path] = file_hash(Path(path))
    return files


def source_identity(root: Path) -> dict:
    from scripts.audit.check_kernel_sources import resolve_kernel_sources
    result = {}
    if subprocess.check_output(['git', 'status', '--porcelain'], cwd=root):
        raise RuntimeError('framework serving checkout is dirty')
    result['framework'] = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
    result['kernel'] = resolve_kernel_sources(root)[1]
    return result


def checked_alias(alias: Path, target: Path):
    if alias.resolve() == target.resolve():
        return
    if alias.exists() or alias.is_symlink():
        raise RuntimeError(f'refusing to replace an existing artifact: {alias}')
    alias.parent.mkdir(parents=True, exist_ok=True)
    alias.symlink_to(target, target_is_directory=True)


def load_exclusions(path: Path | None, pairs: set[str]) -> dict[str, str]:
    """Read explicit checkpoint/mode omissions; never infer missing trials."""
    exclusions = read(path) if path is not None else {}
    if (not isinstance(exclusions, dict) or not set(exclusions) <= pairs
            or any(not isinstance(reason, str) or not reason.strip()
                   for reason in exclusions.values())):
        raise ValueError('exclusions must map planned checkpoint/mode pairs to nonempty reasons')
    return exclusions


def validate_rate_exclusions(plans: list[dict], exclusions: dict[str, str]):
    for plan in plans:
        rates = plan.get('rate_selection')
        if rates and rates.get('exclusions', {}) != exclusions:
            raise RuntimeError('rate calibration and execution must use the same exclusions')


def inspect_trial(output: Path, telemetry: Path) -> dict:
    load_complete_telemetry(telemetry)
    status = read(output / 'status.json')
    if not status.get('complete') or not status.get('jobs') or any(j['returncode'] for j in status['jobs']):
        raise RuntimeError('server trial did not finish all client jobs')
    inputs = read(output / 'inputs.json')
    if [j['id'] for j in status['jobs']] != [j['id'] for j in inputs['jobs']]:
        raise RuntimeError('server trial completed jobs differ from its input list')
    results = []
    for job in status['jobs']:
        result_path = Path(job['result'])
        result = read(result_path)
        if not result.get('complete'):
            raise RuntimeError(f'incomplete measurement: {result_path}')
        results.append(dict(id=job['id'], path=str(result_path), sha256=file_hash(result_path)))
    return dict(status='complete', output=str(output), telemetry=str(telemetry),
                telemetry_sha256=file_hash(telemetry), results=results)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True, help='Frozen serving checkout')
    parser.add_argument('--stack-root', type=Path, required=True)
    parser.add_argument('--build', type=Path, required=True, help='Built release servers from the frozen source')
    parser.add_argument('--plan', type=Path, action='append', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--exclusions', type=Path,
                        help='JSON map of planned checkpoint/mode pairs to reasons; leaves trials unrun')
    args = parser.parse_args()
    root, output, stack, build = (p.resolve() for p in (args.root, args.output, args.stack_root, args.build))
    if output.exists() and not args.resume:
        raise FileExistsError('output exists; use --resume')
    output.mkdir(parents=True, exist_ok=True)
    identity = source_identity(root)
    plans = [read(path) for path in args.plan]
    if any(plan['framework_source'] != identity['framework'] for plan in plans):
        raise RuntimeError('plan source does not match frozen serving source')
    trials = [trial for plan in plans for trial in plan['trials']]
    if not trials or len({t['id'] for t in trials}) != len(trials):
        raise RuntimeError('expected nonempty plans with unique trial IDs')
    files = {str(path.resolve()): file_hash(path) for path in args.plan}
    specs = {trial['id']: read(Path(trial['spec'])) for trial in trials}
    pairs = {f'{spec["checkpoint"]["key"]}/{spec["execution"]["key"]}' for spec in specs.values()}
    exclusions = load_exclusions(args.exclusions, pairs)
    validate_rate_exclusions(plans, exclusions)
    selected_specs = [spec for spec in specs.values()
                      if f'{spec["checkpoint"]["key"]}/{spec["execution"]["key"]}' not in exclusions]
    engines = {'vosti'} | {spec['execution']['engine'] for spec in selected_specs}
    workers = {name: stack / name / '.venv/bin/python' for name in sorted(engines)}
    packages = {name: package_inventory(python) for name, python in workers.items()}
    if args.exclusions:
        files[str(args.exclusions.resolve())] = file_hash(args.exclusions)
    for trial in trials:
        for key in ('spec', 'jobs'):
            files[trial[key]] = file_hash(Path(trial[key]))
        for job in read(Path(trial['jobs'])):
            files.update(job_input_files(job))
    models = {spec['checkpoint']['model'] for spec in selected_specs if spec['execution']['engine'] == 'vosti'}
    binaries = {str(path): file_hash(path) for model in sorted(models)
                for path in [build / 'release/examples' / f'verus_server_{model}']}
    config = dict(root=str(root), source=identity, workers={k: str(v) for k, v in workers.items()},
        packages=packages, input_files=files, binaries=binaries, trials=trials, exclusions=exclusions,
        coordinator_file_sha256=file_hash(Path(__file__)),
        coordinator_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip())
    config_path = output / 'campaign.json'
    if config_path.exists() and read(config_path) != config:
        raise RuntimeError('performance campaign inputs, packages, source or coordinator changed')
    save(config_path, config)
    states = read(output / 'status.json') if (output / 'status.json').exists() else {
        t['id']: dict(status='unrun') for t in trials}
    if set(states) != {trial['id'] for trial in trials}:
        raise RuntimeError('saved trial states differ from the plan')
    save(output / 'status.json', states)

    def check_inputs():
        if source_identity(root) != identity or file_hash(Path(__file__)) != config['coordinator_file_sha256']:
            raise RuntimeError('source changed during performance campaign')
        for path, expected in {**files, **binaries}.items():
            if file_hash(Path(path)) != expected:
                raise RuntimeError(f'campaign artifact changed: {path}')

    def monitored(label: str, command: list[str], gpu: int) -> tuple[int, Path]:
        check_inputs()
        while gpu_pids(gpu):
            print('WAIT: GPU occupied; no job launched or signalled', flush=True)
            time.sleep(30)
        directory = output / 'commands' / label
        directory.mkdir(parents=True, exist_ok=True)
        attempt = len(list(directory.glob('attempt-*.json'))) + 1
        telemetry = directory / f'telemetry-{attempt}.json'
        argv = [str(workers['vosti']), str(root / 'scripts/common/gpu_monitor.py'),
                '--gpu-index', str(gpu), '--output', str(telemetry), '--', *command]
        save(directory / f'attempt-{attempt}.json', dict(command=argv))
        print('START', label, 'attempt', attempt, flush=True)
        env = {key: value for key, value in os.environ.items()
               if not key.startswith(('VOSTI_', 'VLLM_', 'SGLANG_'))}
        env.update(CUDA_VISIBLE_DEVICES=str(gpu), CUDA_DEVICE='cuda:0',
                   PYTHONPATH=f'{root}:{root / "python"}', TOKENIZERS_PARALLELISM='false')
        with (directory / f'attempt-{attempt}.log').open('x') as log:
            code = subprocess.call(argv, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT)
        print('FINISH', label, 'exit', code, flush=True)
        return code, telemetry

    for trial in trials:
        key = trial['id']
        if states[key]['status'] == 'complete':
            prior = states[key]
            observed = inspect_trial(Path(prior['output']), Path(prior['telemetry']))
            if observed != prior:
                raise RuntimeError(f'completed trial evidence changed: {key}')
            continue
        spec = read(Path(trial['spec']))
        pair = f'{spec["checkpoint"]["key"]}/{spec["execution"]["key"]}'
        if pair in exclusions:
            print('EXCLUDED', key, exclusions[pair], 'state retained:', states[key]['status'], flush=True)
            continue
        engine = spec['execution']['engine']
        if packages[engine] != package_inventory(workers[engine]):
            raise RuntimeError(f'{engine} package inventory changed')
        gpu = spec['settings']['gpu_index']
        if engine == 'vosti':
            checkpoint = spec['checkpoint']
            bundle = output / 'deployments' / identity['framework'] / checkpoint['key']
            receipt_path = output / 'deployment-receipts' / f'{checkpoint["key"]}.json'
            if not receipt_path.exists():
                if bundle.exists():
                    raise RuntimeError(f'inspect incomplete deployment before retrying: {bundle}')
                command = [str(workers['vosti']), str(root / 'scripts/prepare_deployment.py'), '--family', checkpoint['model'],
                           checkpoint['path'], '--output', str(bundle)]
                code, telemetry = monitored('deploy-' + checkpoint['key'], command, gpu)
                if code:
                    raise RuntimeError('deployment qualification failed; evidence retained')
                load_complete_telemetry(telemetry)
                receipt_path.parent.mkdir(exist_ok=True)
                save(receipt_path, dict(telemetry=str(telemetry), telemetry_sha256=file_hash(telemetry),
                                       deployment_sha256=file_hash(bundle / 'deployment.json')))
            receipt = read(receipt_path)
            load_complete_telemetry(Path(receipt['telemetry']))
            if (file_hash(Path(receipt['telemetry'])) != receipt['telemetry_sha256']
                    or file_hash(bundle / 'deployment.json') != receipt['deployment_sha256']):
                raise RuntimeError('qualified deployment evidence changed')
            checked_alias(Path(spec['environment']['VOSTI_DEPLOYMENT_BUNDLE']), bundle)
            checked_alias(Path(spec['command'][0]).parents[2], build)
        trial_base = Path(trial['output'])
        trial_output = trial_base
        attempt = 1
        while trial_output.exists():
            attempt += 1
            trial_output = trial_base.with_name(trial_base.name + f'-retry-{attempt}')
        states[key] = dict(status='running', output=str(trial_output))
        save(output / 'status.json', states)
        command = [str(workers['vosti']), '-m', 'scripts.serving_benchmark.server_trial',
                   '--spec', trial['spec'], '--jobs', trial['jobs'], '--output', str(trial_output)]
        code, telemetry = monitored(key, command, gpu)
        try:
            if code:
                raise RuntimeError(f'trial exited {code}')
            states[key] = inspect_trial(trial_output, telemetry)
        except Exception as error:
            states[key] = dict(status='error', output=str(trial_output), telemetry=str(telemetry), error=str(error))
            save(output / 'status.json', states)
            raise RuntimeError(f'{key}: inspect failed trial before continuing') from error
        save(output / 'status.json', states)
        print('RESULT', key, 'complete', flush=True)
    unresolved = {key: state for key, state in states.items() if state['status'] != 'complete'}
    if unresolved:
        save(output / 'partial.json', dict(status='partial', trials=len(trials),
            unresolved=unresolved, exclusions=exclusions))
        print('PARTIAL: excluded performance trials remain unrun', flush=True)
        return
    save(output / 'complete.json', dict(status='complete', trials=len(trials),
        scope=[plan['scope'] for plan in plans], cells=sum(plan['cell_count'] for plan in plans)))


if __name__ == '__main__':
    main()
