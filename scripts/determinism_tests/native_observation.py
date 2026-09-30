"""Read-only native plan witnesses, outside CUDA graph capture.

materialize_step_plan in src/exec/cache_scheduler/mod.rs calls seq_lens_tensor
exactly twice, in Q then K order. Record those host inputs, not inferred token
counts or padded graph buffers. The source-bound worker and strict pair checks
make drift fail closed.
"""
import json
import os


def observe_lengths(lengths):
    step = os.environ.get('VOSTI_LOGITS_OBSERVER_STEP')
    phase = os.environ.get('VOSTI_LOGITS_OBSERVER_PHASE')
    if not step or not phase:
        raise RuntimeError('native layout observation lacks call/step identity')
    values = list(lengths)
    if any(type(value) is not int for value in values):
        raise ValueError('native cumulative lengths must be integers')
    with open(os.environ['VOSTI_NATIVE_LAYOUT_PATH'], 'a') as output:
        output.write(json.dumps(dict(engine_step=step, phase=phase, lengths=values)) + '\n')


def schedule_events(layouts, steps):
    grouped = {}
    for row in layouts:
        grouped.setdefault(row['engine_step'], []).append(row)
    result = {}
    for step in steps:
        key = step['engine_step']
        if key in result:
            raise ValueError('duplicate native step')
        pair = grouped.pop(key, [])
        ids = step['scheduled']
        if not ids and not pair:
            result[key] = []
            continue
        if len(pair) != 2 or pair[0]['phase'] != pair[1]['phase']:
            raise ValueError('native step lacks exactly one Q/K materializer pair')
        q, k = (row['lengths'] for row in pair)
        if (len(q) != len(ids)+1 or len(k) != len(q) or q[0] != 0 or k[0] != 0
                or any(type(x) is not int or x < 0 for x in q+k)):
            raise ValueError('invalid native cumulative length endpoints')
        rows = []
        for i, rid in enumerate(ids):
            query, total = q[i+1]-q[i], k[i+1]-k[i]
            if query <= 0 or total < query:
                raise ValueError('invalid native query/prefix lengths')
            rows.append(dict(request_id=str(rid), query_tokens=query, prefix_tokens=total-query))
        result[key] = rows
    if grouped:
        raise ValueError('native layouts refer to unrecorded steps')
    return result
