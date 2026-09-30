"""Small SGLang worker pilot and request-identity smoke for the report suite."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np

from scripts.determinism_tests.protocol import (
    CHECKPOINTS, EXECUTION_CONFIGS, MODELS, SCHEMA_VERSION, deterministic_prompt,
    compare_row_artifacts, row_comparison_semantics,
)
from scripts.determinism_tests.rank_divergence import ranking_divergence
from scripts.determinism_tests.run import SuiteRunner, _arm, _row_path
from scripts.determinism_tests.call_inputs import teacher_prefixes
from scripts.determinism_tests import call_inputs


PILOT_LENGTHS = (17, 63, 64, 65, 127, 129, 257, 385)
RELATIONS = ('batch', 'chunk', 'pd', 'cache')


def batch_groups(count: int) -> list[list[int]]:
    if count < 8 or count % 8:
        raise ValueError('prompt count must be a positive multiple of eight')
    return [order[start:start + size]
            for size in (2, 4, 8)
            for order in (list(range(count)), list(reversed(range(count))))
            for start in range(0, count, size)]


def aggregate(comparisons: dict, witnesses: dict) -> dict:
    if set(comparisons) != set(RELATIONS) or any(not comparisons[key] for key in RELATIONS):
        raise ValueError('every relation must have measured comparisons')
    valid = bool(witnesses) and all(value is True for value in witnesses.values())
    cells = {}
    for relation, rows in comparisons.items():
        finite = all(row['finite_left'] and row['finite_right'] for row in rows)
        equal = sum(row['bitwise_equal'] for row in rows)
        cells[relation] = dict(comparisons=len(rows), equal=equal,
            mismatches=len(rows) - equal,
            top_token_differences=sum(row.get('argmax_equal') is False for row in rows),
            status='invalid' if not valid or not finite else 'pass' if equal == len(rows) else 'mismatch')
    status = ('invalid' if any(row['status'] == 'invalid' for row in cells.values())
              else 'mismatch' if any(row['status'] == 'mismatch' for row in cells.values()) else 'pass')
    return dict(status=status, cells=cells)


def run_pilot(runner, inputs, execution):
    comparisons = {key: [] for key in RELATIONS}
    witnesses = {}

    def call(label, prompts, *, output=1, all_rows=False):
        return dict(label=label, prompts=prompts, max_tokens=output,
                    record_last_rows=True, record_all_rows=all_rows)

    def arm(name, calls, *, budget=4096, caching=False):
        spec = _arm(inputs=inputs, execution=execution, calls=calls,
                    max_num_batched_tokens=budget, prefix_caching=caching)
        spec['engine']['max_model_len'] = 2048
        print(f'START {name}: {len(calls)} calls', flush=True)
        started = time.monotonic()
        directory, result = runner.run_arm(name, spec)
        if execution.engine == 'sglang':
            backend = result['backend_evidence']
            witnesses[f'{name}:cache_setting'] = backend['disable_radix_cache'] is (not caching)
            witnesses[f'{name}:backend'] = backend['attention_backend'] == execution.attention_backend
            witnesses[f'{name}:deterministic'] = backend['enable_deterministic_inference'] is True
        if not caching:
            witnesses[f'{name}:zero_cache_hits'] = all(
                request['num_cached_tokens'] == 0
                for value in result['calls'] for request in value['requests'])
        print(f'FINISH {name}: {time.monotonic() - started:.2f}s', flush=True)
        return directory, result

    def compare(relation, label, left, right, *, lifetime):
        value = compare_row_artifacts(left, right)
        if value['shape_equal'] and value['dtype_equal'] and value['finite_left'] and value['finite_right']:
            value['ranking'] = ranking_divergence(np.load(left, allow_pickle=False), np.load(right, allow_pickle=False))
        comparisons[relation].append(dict(value, label=label, lifetime=lifetime))

    prompts = inputs['batch_prompts']
    singles_dir, singles = arm('batch-isolated-singles',
        [call(f'single-{i}', [prompt]) for i, prompt in enumerate(prompts)])
    groups = batch_groups(len(prompts))
    batch_dir, batch = arm('batch-compositions',
        [call(f'composition-{i}', [prompts[index] for index in group])
         for i, group in enumerate(groups)])
    for group_index, group in enumerate(groups):
        for offset, prompt_index in enumerate(group):
            compare('batch', f'composition-{group_index}/prompt-{prompt_index}',
                    _row_path(singles_dir, singles, prompt_index, 0),
                    _row_path(batch_dir, batch, group_index, offset), lifetime='shared_engine_cache_disabled')
    for index in (0, len(prompts) - 1):
        directory, result = arm(f'batch-fresh-single-{index}', [call('fresh', [prompts[index]])])
        compare('batch', f'fresh-vs-isolated-{index}', _row_path(directory, result, 0, 0),
                _row_path(singles_dir, singles, index, 0), lifetime='fresh_process_spot_check')
        group_index = next(i for i, group in enumerate(groups) if len(group) == 8)
        compare('batch', f'fresh-vs-batch-{index}', _row_path(directory, result, 0, 0),
                _row_path(batch_dir, batch, group_index, groups[group_index].index(index)),
                lifetime='fresh_process_spot_check')

    chunk_calls = [call(f'length-{len(prompt)}', [prompt]) for prompt in inputs['chunk_prompts']]
    full_dir, full = arm('chunk-full', chunk_calls)
    for budget in (64, 256):
        directory, result = arm(f'chunk-{budget}', chunk_calls, budget=budget)
        witnesses[f'chunk-{budget}:configured_budget'] = (
            result['engine_kwargs'].get('chunked_prefill_size') == budget)
        for index, prompt in enumerate(inputs['chunk_prompts']):
            compare('chunk', f'length-{len(prompt)}/budget-{budget}',
                    _row_path(full_dir, full, index, 0), _row_path(directory, result, index, 0),
                    lifetime='fresh_engine_per_budget_shared_engine_across_prompts')

    prompt = inputs['pd_prompt']
    gen_dir, gen = arm('pd-generate', [call('generate', [prompt], output=16, all_rows=True)])
    request = gen['calls'][0]['requests'][0]
    outputs = request['output_token_ids']
    rows = request['output_rows']
    if len(outputs) != 16 or len(rows) != len(outputs):
        raise RuntimeError('missing generated token or prediction row')
    teacher_dir, teacher = arm('pd-teacher',
        [call(f'prediction-{i}', [prefix]) for i, prefix in enumerate(teacher_prefixes(prompt, outputs))])
    for index, record in enumerate(rows):
        compare('pd', f'prediction-{index}', gen_dir / 'rows' / record['artifact'],
                _row_path(teacher_dir, teacher, index, 0), lifetime='fresh_teacher_engine_cache_disabled')

    cache_prompts = inputs['cache_prompts']
    cold_dir, cold = arm('cache-cold', [call(f'cold-{i}', [prompt]) for i, prompt in enumerate(cache_prompts)])
    for index, (prompt, donor_length) in enumerate(zip(cache_prompts, (128, 384), strict=True)):
        directory, result = arm(f'cache-reuse-{index}',
            [call('donor', [prompt[:donor_length]]), call('reuse', [prompt])], caching=True)
        donor = result['calls'][0]['requests'][0]
        warm = result['calls'][1]['requests'][0]
        witnesses[f'cache-{index}:cold_donor'] = donor['num_cached_tokens'] == 0
        witnesses[f'cache-{index}:positive_bounded_hit'] = 0 < warm['num_cached_tokens'] <= donor_length
        compare('cache', f'donor-{donor_length}/query-{len(prompt)}',
                _row_path(cold_dir, cold, index, 0), _row_path(directory, result, 1, 0),
                lifetime='fresh_donor_reuse_pair_vs_cache_disabled_cold_engine')
    return dict(aggregate(comparisons, witnesses), comparisons=comparisons, witnesses=witnesses)


def run_identity_smoke(runner, inputs, execution):
    prompts = inputs['chunk_prompts']
    calls = [dict(label=label, prompts=values, max_tokens=2,
                  record_last_rows=True, record_all_rows=True)
             for label, values in (('forward', prompts), ('reverse', list(reversed(prompts))))]
    spec = _arm(inputs=inputs, execution=execution, calls=calls,
                prefix_caching=False, max_num_batched_tokens=64)
    spec['engine'].update(max_model_len=2048, trace_requests=True)
    directory, result = runner.run_arm('request-identity-split-batches', spec)
    metadata = [row['metadata'] for value in result['calls']
                for request in value['requests'] for row in request['output_rows']]
    witness = dict(all_rows_identified=len(metadata) == 12 and all(row.get('request_id') for row in metadata),
                   zero_cache_hits=all(request['num_cached_tokens'] == 0
                       for value in result['calls'] for request in value['requests']),
                   cache_disabled=result['backend_evidence']['disable_radix_cache'] is True,
                   selected_fa3=result['backend_evidence']['attention_backend'] == 'fa3',
                   deterministic=result['backend_evidence']['enable_deterministic_inference'] is True)
    # Include intermediate sampling observations, not only final selected rows.
    from scripts.determinism_tests.protocol import load_observer_records
    records = load_observer_records(directory / 'rows')
    witness['chunk_observed'] = any(0 < row.get('metadata', {}).get('query_tokens', 0) <= 64
                                  and row.get('metadata', {}).get('prefix_tokens', 0) > 0 for row in records)
    return dict(status='pass' if all(value is True for value in witness.values()) else 'invalid',
                cells={}, witnesses=witness, observed_rows=len(records), selected_rows=len(metadata))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    kind = parser.add_mutually_exclusive_group(required=True)
    kind.add_argument('--pilot', action='store_true')
    kind.add_argument('--request-identity-smoke', action='store_true')
    parser.add_argument('--checkpoint', choices=('llama3-8b', 'gemma3-4b'), default='llama3-8b')
    parser.add_argument('--execution-config', choices=('sglang-deterministic-fa3',), default='sglang-deterministic-fa3')
    parser.add_argument('--seed', type=int, default=1592598566)
    parser.add_argument('--worker-python', type=Path, required=True)
    parser.add_argument('--worker-root', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--gpu-index', type=int, default=0)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if args.gpu_index < 0:
        parser.error("--gpu-index must be nonnegative")
    root = args.worker_root.resolve()
    if subprocess.check_output(['git', 'status', '--porcelain'], cwd=root):
        raise RuntimeError('use a clean frozen worker source for the pilot')
    checkpoint = next(value for value in CHECKPOINTS if value.key == args.checkpoint)
    model = next(value for value in MODELS if value.key == checkpoint.model)
    execution = next(value for value in EXECUTION_CONFIGS if value.key == args.execution_config)
    path = Path(checkpoint.path)
    def prompt(length, offset):
        return deterministic_prompt(path, length=length, seed=args.seed + offset)
    inputs = dict(schema_version=SCHEMA_VERSION,
        profile='request-identity-smoke-v1' if args.request_identity_smoke else 'expanded-pilot-v1',
        model=asdict(model), model_path=str(path), checkpoint=checkpoint.key, seed=args.seed,
        execution=asdict(execution), gpu_index=args.gpu_index,
        worker_root=str(root), worker_python=os.path.abspath(args.worker_python),
        worker_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(),
        controller_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        input_builder_sha256=hashlib.sha256(Path(call_inputs.__file__).read_bytes()).hexdigest(),
        batch_prompts=[prompt(length, i) for i, length in enumerate(PILOT_LENGTHS)],
        chunk_prompts=[prompt(length, 100 + i) for i, length in enumerate((65, 257, 1089))],
        pd_prompt=prompt(257, 200), cache_prompts=[prompt(385, 300 + i) for i in range(2)],
        comparison_semantics=row_comparison_semantics())
    if args.resume:
        if args.request_identity_smoke:
            parser.error('smoke runs require a new output directory')
        if (args.output / 'summary.json').exists():
            raise RuntimeError('pilot already completed; do not overwrite it')
        previous = json.loads((args.output / 'inputs.json').read_text())
        if previous != inputs:
            raise RuntimeError('pilot inputs or source changed; start a new pilot')
    else:
        args.output.mkdir(parents=True, exist_ok=False)
        (args.output / 'inputs.json').write_text(json.dumps(inputs, indent=2) + '\n')
    runner = SuiteRunner(output=args.output, worker_python=Path(os.path.abspath(args.worker_python)),
                         gpu_index=args.gpu_index, execution=execution, worker_root=root)
    started = time.monotonic()
    result = (run_identity_smoke if args.request_identity_smoke else run_pilot)(runner, inputs, execution)
    result.update(profile=inputs['profile'], elapsed_seconds=time.monotonic() - started,
                  elapsed_seconds_scope='current_controller_invocation_not_previous_attempts',
                  arms=runner.arm_records, comparison_semantics=row_comparison_semantics(),
                  raw_logits_bytes=sum(file.stat().st_size for file in args.output.rglob('*.npy')))
    # Resumed controllers may reuse most arms. Do not describe their short
    # invocation time as the total cost of the measured experiment.
    result['valid_arm_seconds'] = 0.0
    for arm in runner.arm_records:
        data = Path(arm['telemetry']['path']).read_bytes()
        if hashlib.sha256(data).hexdigest() != arm['telemetry']['sha256']:
            raise RuntimeError('arm telemetry changed before timing aggregation')
        telemetry = json.loads(data)
        result['valid_arm_seconds'] += telemetry['finished_unix_s'] - telemetry['started_unix_s']
    (args.output / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({key: result[key] for key in ('status', 'cells', 'elapsed_seconds', 'raw_logits_bytes')}), flush=True)
    if result['status'] == 'invalid':
        raise SystemExit(2)


if __name__ == '__main__':
    main()
