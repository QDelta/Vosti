"""Full-logit chunk and prefix-reuse relations at long/window-boundary lengths."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import subprocess

from scripts.determinism_tests.protocol import (
    CAMPAIGN_SEEDS, CHECKPOINTS, EXECUTION_CONFIGS, MODELS, SCHEMA_VERSION,
    compare_row_artifacts, deterministic_prompt, row_comparison_semantics, sha256_json,
)
from scripts.determinism_tests.run import SuiteRunner, _arm, _row_path
from scripts.determinism_tests.vosti_worker import PAGE_SIZE


def lengths_for(model_path: Path, profile: str) -> tuple[int, ...]:
    if profile == 'long-context':
        return (8192, 32768)
    if profile != 'window-boundary':
        raise ValueError('unknown context profile')
    config = json.loads((model_path / 'config.json').read_text())
    text = config.get('text_config', config)
    window = text.get('sliding_window')
    if type(window) is not int or window <= 1 or text.get('use_sliding_window') is False:
        return ()
    return (window - 1, window, window + 1)


def make_inputs(model, path: Path, *, profile: str, length: int, seed: int) -> dict:
    if length not in lengths_for(path, profile):
        raise ValueError('length is not part of the checkpoint context profile')
    budgets = [512, 2048, 4096] if profile == 'long-context' else [64, 256, 512]
    if max(budgets) >= length:
        raise ValueError('profile requires every chunk budget to be smaller than the prompt')
    return dict(schema_version=SCHEMA_VERSION, model=asdict(model), model_path=str(path),
        profile=profile, length=length, seed=seed, chunk_budgets=budgets,
        prompt=deterministic_prompt(path, length=length, seed=seed + length))


def context_arm(inputs: dict, execution, calls: list[dict], *, caching: bool, budget: int) -> dict:
    arm = _arm(inputs=inputs, execution=execution, calls=calls,
               prefix_caching=caching, max_num_batched_tokens=budget)
    arm['engine']['max_model_len'] = inputs['length'] + 2
    # This is engine cache capacity, not a runtime-selected kernel config.
    arm['engine']['num_blocks'] = (inputs['length'] + 2 + PAGE_SIZE - 1) // PAGE_SIZE + 32
    return arm


def run_case(runner, inputs: dict, execution, *, hardware: str) -> dict:
    def call(label):
        return dict(label=label, prompts=[inputs['prompt']], max_tokens=2,
                    record_last_rows=True, record_generated_position=0)
    rows = []
    for budget in inputs['chunk_budgets']:
        directory, result = runner.run_arm(f'chunk-{budget}', context_arm(
            inputs, execution, [call('chunk')], caching=False, budget=budget))
        rows.append(_row_path(directory, result, 0, 0))
    # Donor and reuse run in one engine lifetime. A separate cold engine avoids
    # treating unrelated setup/warmup cache state as the cold reference.
    cold_dir, cold = runner.run_arm('prefix-cold', context_arm(
        inputs, execution, [call('cold')], caching=True, budget=4096))
    warm_dir, warm = runner.run_arm('prefix-reuse', context_arm(
        inputs, execution, [call('donor'), call('reuse')], caching=True, budget=4096))
    cold_cached = cold['calls'][0]['requests'][0]['num_cached_tokens']
    donor_cached = warm['calls'][0]['requests'][0]['num_cached_tokens']
    warm_cached = warm['calls'][1]['requests'][0]['num_cached_tokens']
    witness = dict(cold_cache_empty=cold_cached == 0, donor_cache_empty=donor_cached == 0,
                   warm_prefix_observed=type(warm_cached) is int and 0 < warm_cached <= inputs['length'])
    comparisons = dict(different_chunk=[compare_row_artifacts(rows[0], row) for row in rows[1:]],
        cold_vs_warm=[compare_row_artifacts(_row_path(cold_dir, cold, 0, 0), _row_path(warm_dir, warm, 1, 0))])
    relation_pass = {name: all(row['bitwise_equal'] and row['finite_left'] and row['finite_right'] for row in values)
                     for name, values in comparisons.items()}
    status = 'invalid' if not all(witness.values()) else 'pass' if all(relation_pass.values()) else 'mismatch'
    return dict(schema_version=SCHEMA_VERSION, hardware=hardware, profile=inputs['profile'],
        length=inputs['length'], model=inputs['model'], model_path=inputs['model_path'], seed=inputs['seed'],
        execution=asdict(execution), inputs_sha256=sha256_json(inputs),
        comparison_semantics=row_comparison_semantics(), comparisons=comparisons,
        relation_pass=relation_pass, witness=witness, cold_cached_tokens=cold_cached,
        donor_cached_tokens=donor_cached, warm_cached_tokens=warm_cached,
        arms=runner.arm_records, status=status)


def matrix(profile: str) -> list[dict]:
    return [dict(checkpoint=model.key, model=model.model, model_path=model.path, profile=profile,
                 length=length, execution_config=mode.key, seed=seed)
            for model in CHECKPOINTS for length in lengths_for(Path(model.path), profile)
            for mode in EXECUTION_CONFIGS for seed in CAMPAIGN_SEEDS]


def write_new(path, data):
    with path.open('x') as stream:
        json.dump(data, stream, indent=2, sort_keys=True)
        stream.write('\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', choices=('long-context', 'window-boundary'), required=True)
    parser.add_argument('--matrix', action='store_true', help='List cases without running GPU work')
    parser.add_argument('--model', choices=[model.key for model in MODELS])
    parser.add_argument('--model-path', type=Path)
    parser.add_argument('--length', type=int)
    parser.add_argument('--seed', type=lambda value: int(value, 0), default=CAMPAIGN_SEEDS[0])
    parser.add_argument('--execution-config', choices=[mode.key for mode in EXECUTION_CONFIGS])
    parser.add_argument('--worker-python', type=Path)
    parser.add_argument('--worker-root', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--vosti-binary', type=Path)
    parser.add_argument('--deployment-bundle', type=Path)
    parser.add_argument('--gpu-index', type=int, default=0)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if args.gpu_index < 0:
        parser.error("--gpu-index must be nonnegative")
    if args.matrix:
        cases = matrix(args.profile)
        write_new(args.output, dict(status='planned_not_executed', cases=cases,
                                   case_count=len(cases), relation_cells=2 * len(cases)))
        return
    if any(value is None for value in (args.model, args.model_path, args.length, args.execution_config, args.worker_python)):
        parser.error('execution requires model, model-path, length, execution-config and worker-python')
    model = next(model for model in MODELS if model.key == args.model)
    execution = next(mode for mode in EXECUTION_CONFIGS if mode.key == args.execution_config)
    inputs = make_inputs(model, args.model_path.resolve(), profile=args.profile, length=args.length, seed=args.seed)
    inputs['execution'] = asdict(execution)
    inputs['worker_root'] = str(args.worker_root.resolve())
    inputs['worker_source'] = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=args.worker_root, text=True).strip()
    fields = {}
    if execution.engine == 'vosti':
        if args.vosti_binary is None or args.deployment_bundle is None:
            parser.error('Vosti requires its built engine and qualified deployment bundle')
        fields = dict(vosti_binary=str(args.vosti_binary.resolve()), deployment_bundle=str(args.deployment_bundle.resolve()))
        from hashlib import sha256
        inputs['vosti_artifacts'] = dict(binary_sha256=sha256(args.vosti_binary.read_bytes()).hexdigest(),
            deployment_sha256=sha256((args.deployment_bundle / 'deployment.json').read_bytes()).hexdigest())
    if args.output.exists() and not args.resume:
        raise FileExistsError('output exists; use --resume')
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / 'inputs.json'
    if path.exists():
        if json.loads(path.read_text()) != inputs:
            raise RuntimeError('resumed context inputs or worker source changed')
    else:
        write_new(path, inputs)
    if (args.output / 'summary.json').exists():
        raise RuntimeError('case already has a completed summary')
    runner = SuiteRunner(output=args.output, worker_python=Path(os.path.abspath(args.worker_python)),
        worker_root=args.worker_root, gpu_index=args.gpu_index, execution=execution, arm_engine_fields=fields)
    result = run_case(runner, inputs, execution, hardware='h200')
    write_new(args.output / 'summary.json', result)
    print(json.dumps(dict(status=result['status'], relation_pass=result['relation_pass'], witness=result['witness'])))
    if result['status'] == 'invalid' or (result['status'] == 'mismatch' and execution.strict):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
