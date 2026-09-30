"""Vosti schedule/cache/graph stress, checked against fresh single requests."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path

from scripts.determinism_tests.protocol import (
    BATCH_LENGTHS, CAMPAIGN_SEEDS, CHECKPOINTS, EXECUTION_CONFIGS, MODELS,
    compare_row_artifacts, deterministic_prompt, row_comparison_semantics, sha256_json,
)
from scripts.determinism_tests.run import SuiteRunner, _arm, _row_path
from scripts.determinism_tests.context_relations import write_new
from scripts.serving_benchmark.campaign import source_identity


OUTPUT_TOKENS = 16
ARRIVALS = [0, 0, 2, 4, 8, 12, 16, 20]
RELATIONS = ('queued_batch', 'staggered_arrivals', 'cache_pressure', 'padded_cover')
MODE = EXECUTION_CONFIGS[0]


def make_inputs(checkpoint, seed):
    path = Path(checkpoint.path)
    return dict(checkpoint=asdict(checkpoint), model=asdict(next(m for m in MODELS if m.key == checkpoint.model)),
        model_path=str(path), seed=seed, output_tokens=OUTPUT_TOKENS, arrivals=ARRIVALS,
        prompts=[deterministic_prompt(path, length=n, seed=seed + 1000 + i) for i, n in enumerate(BATCH_LENGTHS)],
        primer=[deterministic_prompt(path, length=17, seed=seed + 2000 + i) for i in range(4)])


def stress_arm(inputs, prompts, *, arrivals=None, blocks=128, primer=False, budget=128):
    call = dict(label='measured', prompts=prompts, max_tokens=OUTPUT_TOKENS,
                record_last_rows=True, record_generated_position=OUTPUT_TOKENS - 1)
    if arrivals is not None:
        call['arrival_steps'] = arrivals
    arm = _arm(inputs=inputs, execution=MODE, calls=[call], prefix_caching=True,
               max_num_batched_tokens=budget)
    arm['engine'].update(trace_steps=True, num_blocks=blocks, max_num_seqs=4)
    if primer:
        arm['engine']['graph_primer'] = dict(prompts=inputs['primer'], max_tokens=4)
    return arm


def schedule_witness(steps, *, measured_base, output_tokens, graph_before, graph_after):
    counts, previous_pages = {}, {}
    overlap, replacements, measured_steps = False, 0, 0
    for step in steps:
        pages = {page[0]: page[1:] for page in step['prefix_pages']}
        if step['request_id_base'] == measured_base:
            measured_steps += 1
            if step['arrived'] and any(0 < count < output_tokens for count in counts.values()):
                overlap = True
            replacements += sum(pages.get(bid) != page for bid, page in previous_pages.items())
            for rid, token in zip(step['scheduled'], step['emitted']):
                if token is not None:
                    counts[rid] = counts.get(rid, 0) + 1
        previous_pages = pages
    cover = graph_after['cover_replay_count'] - graph_before['cover_replay_count']
    replay = cover + graph_after['replay_count'] - graph_before['replay_count']
    return dict(measured_steps=measured_steps, arrivals_during_decode=overlap,
                registered_prefix_replacements=replacements, measured_graph_replays=replay,
                measured_cover_replays=cover)


def run_case(runner, inputs):
    references, reference_tokens = [], []
    for index, prompt in enumerate(inputs['prompts']):
        directory, result = runner.run_arm(f'single-{index:02}', stress_arm(inputs, [prompt], budget=4096))
        references.append(_row_path(directory, result, 0, 0))
        reference_tokens.append(result['calls'][0]['requests'][0]['output_token_ids'])
    comparisons, witnesses, relation_pass = {}, {}, {}
    for relation in RELATIONS:
        staggered = relation != 'queued_batch'
        primer = relation == 'padded_cover'
        directory, result = runner.run_arm(relation, stress_arm(inputs, inputs['prompts'],
            arrivals=inputs['arrivals'] if staggered else [0] * len(inputs['prompts']),
            blocks=32 if relation == 'cache_pressure' else 128, primer=primer))
        invocation = result['backend_evidence']['invocations'][0]
        trace = invocation['step_trace']
        trace_path = Path(trace['path'])
        if hashlib.sha256(trace_path.read_bytes()).hexdigest() != trace['sha256']:
            raise RuntimeError('native step-trace hash changed')
        steps = [json.loads(line) for line in trace_path.read_text().splitlines()]
        witness = schedule_witness(steps, measured_base=len(inputs['prompts']) + (4 if primer else 0),
            output_tokens=OUTPUT_TOKENS, graph_before=invocation['graph_warmup_stats'],
            graph_after=invocation['graph_stats'])
        witness['valid'] = (witness['measured_steps'] > 0 and witness['measured_graph_replays'] > 0
            and (not staggered or witness['arrivals_during_decode'])
            and (relation != 'cache_pressure' or witness['registered_prefix_replacements'] > 0)
            and (not primer or witness['measured_cover_replays'] > 0))
        rows = []
        for index, reference in enumerate(references):
            comparison = compare_row_artifacts(reference, _row_path(directory, result, 0, index))
            comparison['output_tokens_equal'] = reference_tokens[index] == result['calls'][0]['requests'][index]['output_token_ids']
            rows.append(comparison)
        comparisons[relation], witnesses[relation] = rows, witness
        relation_pass[relation] = all(row['bitwise_equal'] and row['finite_left'] and row['finite_right']
                                      and row['output_tokens_equal'] for row in rows)
    status = 'invalid' if not all(w['valid'] for w in witnesses.values()) else 'pass' if all(relation_pass.values()) else 'mismatch'
    return dict(status=status, model=inputs['model'], model_path=inputs['model_path'], seed=inputs['seed'],
        execution=asdict(MODE), hardware='h200', inputs_sha256=sha256_json(inputs),
        comparison_semantics=row_comparison_semantics(), comparisons=comparisons,
        relation_pass=relation_pass, witnesses=witnesses, arms=runner.arm_records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--matrix', action='store_true')
    parser.add_argument('--checkpoint', choices=[c.key for c in CHECKPOINTS])
    parser.add_argument('--seed', type=int, default=CAMPAIGN_SEEDS[0])
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
        cases = [dict(checkpoint=c.key, model_path=c.path, seed=s, execution_config=MODE.key)
                 for c in CHECKPOINTS for s in CAMPAIGN_SEEDS]
        write_new(args.output, dict(status='planned_not_executed', cases=cases, case_count=len(cases),
            relations=list(RELATIONS), relation_cells=len(cases) * len(RELATIONS),
            prompt_lengths=list(BATCH_LENGTHS), generated_tokens=OUTPUT_TOKENS, arrivals=ARRIVALS,
            max_sequences=4, chunk_budget=128, cache_blocks=[128, 32], reference_chunk_budget=4096))
        return
    if any(v is None for v in (args.checkpoint, args.worker_python, args.vosti_binary, args.deployment_bundle)):
        parser.error('execution requires checkpoint, worker-python, vosti-binary and deployment-bundle')
    inputs = make_inputs(next(c for c in CHECKPOINTS if c.key == args.checkpoint), args.seed)
    inputs.update(worker_root=str(args.worker_root.resolve()), source=source_identity(args.worker_root),
        controller_source=source_identity(Path(__file__).resolve().parents[2]),
        binary_sha256=hashlib.sha256(args.vosti_binary.read_bytes()).hexdigest(),
        deployment_sha256=hashlib.sha256((args.deployment_bundle / 'deployment.json').read_bytes()).hexdigest())
    if args.output.exists() and not args.resume:
        raise FileExistsError('output exists; use --resume')
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / 'inputs.json'
    if path.exists():
        if json.loads(path.read_text()) != inputs:
            raise RuntimeError('resumed stress inputs or source changed')
    else:
        write_new(path, inputs)
    if (args.output / 'summary.json').exists():
        raise RuntimeError('stress case already completed')
    runner = SuiteRunner(output=args.output, worker_python=Path(os.path.abspath(args.worker_python)),
        gpu_index=args.gpu_index, worker_root=args.worker_root, execution=MODE,
        arm_engine_fields=dict(vosti_binary=str(args.vosti_binary.resolve()),
                               deployment_bundle=str(args.deployment_bundle.resolve())))
    result = run_case(runner, inputs)
    write_new(args.output / 'summary.json', result)
    print(json.dumps(dict(status=result['status'], relation_pass=result['relation_pass'], witnesses=result['witnesses'])))
    if result['status'] != 'pass':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
