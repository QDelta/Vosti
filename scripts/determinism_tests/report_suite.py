"""Expanded single-seed report execution; retain raw evidence and partial cells."""
from dataclasses import asdict
import json
from pathlib import Path
import time

import numpy as np

from scripts.determinism_tests.protocol import compare_row_artifacts, load_observer_records, row_comparison_semantics
from scripts.determinism_tests.rank_divergence import ranking_divergence
from scripts.determinism_tests.run import _arm, _row_path
from scripts.determinism_tests.call_inputs import teacher_prefixes
from scripts.determinism_tests.vllm_observation import public_schedule_events


def prefill_witness(rows, length, budget):
    cursor = 0
    pieces = []
    for row in rows:
        prefix, query = row.get('prefix_tokens'), row.get('query_tokens')
        if prefix is None or query is None or prefix >= length:
            continue
        if type(prefix) is not int or type(query) is not int or prefix != cursor or not 0 < query <= budget:
            return dict(valid=False, pieces=pieces, reason='noncontiguous_or_oversized_prefill')
        cursor += query
        pieces.append(query)
    return dict(valid=cursor == length and (len(pieces) == 1 if budget >= length else len(pieces) > 1),
                pieces=pieces, consumed=cursor, expected=length)


def summarize(comparisons, expected, witnesses):
    cells = {}
    for relation, count in expected.items():
        rows = comparisons[relation]
        equal = sum(row['bitwise_equal'] for row in rows)
        valid = all(item['valid'] for item in witnesses[relation]) and all(
            row['finite_left'] and row['finite_right'] for row in rows)
        state = ('incomplete' if len(rows) != count else 'invalid' if not valid else
                 'pass' if equal == count else 'mismatch')
        cells[relation] = dict(status=state, expected=count, compared=len(rows),
                              equal=equal, mismatches=len(rows)-equal,
                              top_token_differences=sum(row.get('argmax_equal') is False for row in rows))
    return cells


def run_report(runner, plan, execution):
    if execution.engine == 'vosti' and not runner.arm_engine_fields.get('multi_call'):
        raise RuntimeError('native report requires the multi-call adapter and qualified deployment')
    expected = dict(batch=288 + 2*len(plan['fresh_spots']),
                    chunk=len(plan['chunk_prompts'])*len(plan['chunk_budgets']),
                    pd=len(plan['pd_prompts'])*plan['output_tokens'], cache=len(plan['cache_cases'])+1)
    comparisons = {key: [] for key in expected}
    witnesses = {key: [] for key in expected}
    started = time.time()

    def save():
        value = dict(profile=plan['profile'], checkpoint=plan['checkpoint'], seed=plan['seed'],
                     execution=asdict(execution), expected=expected,
                     cells=summarize(comparisons, expected, witnesses), comparisons=comparisons,
                     witnesses=witnesses, arms=runner.arm_records, started_unix_s=started,
                     comparison_semantics=row_comparison_semantics())
        temporary = runner.output / 'progress.tmp'
        temporary.write_text(json.dumps(value, indent=2) + '\n')
        temporary.replace(runner.output / 'progress.json')
        return value

    def call(label, prompts, output=2, all_rows=False):
        return dict(label=label, prompts=prompts, max_tokens=output, record_last_rows=True,
                    record_generated_position=output-1 if all_rows else 0, record_all_rows=all_rows)

    def arm(relation, name, calls, caching=False, budget=4096):
        spec = _arm(inputs=plan, execution=execution, calls=calls,
                    prefix_caching=caching, max_num_batched_tokens=budget)
        spec['engine'].update(gpu_memory_utilization=.40, max_model_len=33024,
            trace_requests=True, max_prefill_tokens=budget, max_total_tokens=73728,
            kv_cache_memory_bytes=12*1024**3, graph_prefill_cap=4096)
        print(f'START {plan["checkpoint"]}/{execution.key}/{name}: {len(calls)} calls', flush=True)
        directory, result = runner.run_arm(name, spec)
        if execution.engine == 'vllm':
            # Validate every schedule/result namespace association, not merely
            # the singleton chunk witnesses. Original result.json stays intact.
            for value in result['calls']:
                public_schedule_events(value, result['vllm_version'])
        if not caching:
            witnesses[relation].append(dict(label=f'{name}:cold_state', valid=all(
                request['num_cached_tokens'] == 0 for value in result['calls'] for request in value['requests'])))
        if execution.engine == 'sglang':
            backend = result['backend_evidence']
            witnesses[relation].append(dict(label=f'{name}:backend_settings', valid=(
                backend['disable_radix_cache'] is (not caching)
                and backend['enable_deterministic_inference'] is (execution.mode == 'deterministic')
                and (execution.attention_backend is None or backend['attention_backend'] == execution.attention_backend))))
        save()
        return directory, result

    def compare(relation, label, left, right):
        row = compare_row_artifacts(left, right)
        if row['shape_equal'] and row['dtype_equal'] and row['finite_left'] and row['finite_right']:
            row['ranking'] = ranking_divergence(np.load(left, allow_pickle=False), np.load(right, allow_pickle=False))
        comparisons[relation].append(dict(row, label=label))

    prompts = plan['batch_prompts']
    sd, singles = arm('batch', 'isolated-singles', [call(str(i), [prompt]) for i, prompt in enumerate(prompts)])
    bd, batches = arm('batch', 'compositions', [call(str(i), [prompts[j] for j in group])
                          for i, group in enumerate(plan['batch_groups'])])
    batch_records = load_observer_records(bd / 'rows') if execution.engine == 'sglang' else None
    for index, group in enumerate(plan['batch_groups']):
        ids = {request['request_id'] for request in batches['calls'][index]['requests']}
        together = (any(row.get('metadata', {}).get('request_id') in ids and row['batch_size'] > 1
                        for row in batch_records) if batch_records is not None
                    else any(len(event)>1 for event in batches['calls'][index]['schedule_events']))
        witnesses['batch'].append(dict(label=f'composition-{index}:observed_batch', valid=together))
        for offset, prompt_index in enumerate(group):
            compare('batch', f'composition-{index}/prompt-{prompt_index}',
                    _row_path(sd, singles, prompt_index, 0), _row_path(bd, batches, index, offset))
    for index in plan['fresh_spots']:
        fd, fresh = arm('batch', f'fresh-single-{index}', [call('fresh', [prompts[index]])])
        compare('batch', f'fresh-vs-isolated-{index}', _row_path(fd, fresh, 0, 0), _row_path(sd, singles, index, 0))
        group_index = next(i for i, group in enumerate(plan['batch_groups']) if len(group)==8 and index in group)
        compare('batch', f'fresh-vs-batch-{index}', _row_path(fd, fresh, 0, 0),
                _row_path(bd, batches, group_index, plan['batch_groups'][group_index].index(index)))
    save()

    chunk_calls = [call(str(len(prompt)), [prompt]) for prompt in plan['chunk_prompts']]
    def chunk_arm(budget):
        directory, result = arm('chunk', f'chunk-{budget}', chunk_calls, budget=budget)
        records = load_observer_records(directory / 'rows') if execution.engine == 'sglang' else None
        for index, prompt in enumerate(plan['chunk_prompts']):
            request = result['calls'][index]['requests'][0]
            rid = request['request_id']
            events = (public_schedule_events(result['calls'][index], result['vllm_version'])
                      if execution.engine == 'vllm' else result['calls'][index].get('schedule_events', []))
            rows = ([row['metadata'] for row in records if row.get('metadata', {}).get('request_id') == rid]
                    if records is not None else [row for event in events
                                                for row in event if row['request_id'] == rid])
            witnesses['chunk'].append(dict(prefill_witness(rows, len(prompt), budget), label=f'{len(prompt)}/{budget}'))
        return directory, result
    full_dir, full = chunk_arm(plan['full_budget'])
    for budget in plan['chunk_budgets']:
        directory, result = chunk_arm(budget)
        for index, prompt in enumerate(plan['chunk_prompts']):
            compare('chunk', f'{len(prompt)}/{budget}', _row_path(full_dir, full, index, 0), _row_path(directory, result, index, 0))
        save()

    gen_dir, generated = arm('pd', 'generate', [call(str(len(prompt)), [prompt], plan['output_tokens'], True)
                                               for prompt in plan['pd_prompts']])
    teacher_calls = []
    positions = []
    for index, prompt in enumerate(plan['pd_prompts']):
        request = generated['calls'][index]['requests'][0]
        if len(request['output_rows']) != plan['output_tokens']:
            raise RuntimeError('generation omitted prediction rows')
        for position, prefix in enumerate(teacher_prefixes(prompt, request['output_token_ids'])):
            positions.append((index, position))
            teacher_calls.append(call(f'{index}/{position}', [prefix]))
    td, teacher = arm('pd', 'teacher', teacher_calls)
    for index, (prompt_index, position) in enumerate(positions):
        row = generated['calls'][prompt_index]['requests'][0]['output_rows'][position]
        compare('pd', f'{prompt_index}/{position}', gen_dir/'rows'/row['artifact'], _row_path(td, teacher, index, 0))
    save()

    cd, cold = arm('cache', 'cold-prefix-references', [call(case['label'], [case['query']]) for case in plan['cache_cases']])
    for index, case in enumerate(plan['cache_cases']):
        directory, result = arm('cache', f'prefix-{case["label"]}',
            [call('donor', [case['donor']]), call('reuse', [case['query']])], caching=True)
        donor, warm = [value['requests'][0] for value in result['calls']]
        hits = warm['num_cached_tokens']
        witnesses['cache'].append(dict(label=case['label'], cached_tokens=hits,
            valid=donor['num_cached_tokens']==0 and 0 <= hits <= case['prefix_limit']
                  and (not case['require_positive'] or hits > 0)))
        compare('cache', case['label'], _row_path(cd, cold, index, 0), _row_path(directory, result, 1, 0))
        save()

    donor = call('generated-donor', [plan['generated_donor']], plan['output_tokens'])
    donor['record_last_rows'] = False
    followup = call('generated-followup', [])
    followup['output_prefix_from'] = dict(call=0, request=0, count=plan['output_tokens']-1, suffix=plan['generated_suffix'])
    wd, warm = arm('cache', 'generated-prefix-reuse', [donor, followup], caching=True)
    query = warm['calls'][1]['requests'][0]['input_token_ids']
    fd, fresh = arm('cache', 'generated-prefix-cold', [call('cold', [query])])
    hits = warm['calls'][1]['requests'][0]['num_cached_tokens']
    witnesses['cache'].append(dict(label='generated-prefix', cached_tokens=hits,
        valid=warm['calls'][0]['requests'][0]['num_cached_tokens']==0 and
              len(plan['generated_donor']) < hits <= len(plan['generated_donor'])+plan['output_tokens']-1))
    compare('cache', 'generated-prefix', _row_path(fd, fresh, 0, 0), _row_path(wd, warm, 1, 0))
    result = save()
    result['status'] = ('invalid' if any(value['status'] in ('invalid', 'incomplete') for value in result['cells'].values())
                        else 'mismatch' if any(value['status']=='mismatch' for value in result['cells'].values()) else 'pass')
    with (runner.output/'summary.json').open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    return result
