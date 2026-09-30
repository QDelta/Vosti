"""Identical cached-context cohorts and result checks across aligned adapters."""
import asyncio
import json
import math
from pathlib import Path
import statistics
import time

from scripts.serving_benchmark.aligned.protocol import Cohort
from scripts.serving_benchmark.cached_phases import prepare, validate
from scripts.serving_benchmark.multi_turn import write_new


def inputs(tokenizer, checkpoint, contexts, measured=5):
    raw = prepare(tokenizer, checkpoint, seed=42, contexts=contexts, waves=measured+1)
    validate(raw)
    cases = []
    for cell in raw.pop('cells'):
        rows = cell['warmup']+cell['measured']
        for wave, offset in enumerate(range(0, len(rows), 4)):
            cases.append(dict(name=f'{cell["kind"][0]}{cell["cached_tokens"]}w{wave}',
                phase='warmup' if wave<2 else 'measured', kind=cell['kind'],
                cached_tokens=cell['cached_tokens'], query_tokens=cell['query_tokens'],
                rows=rows[offset:offset+4]))
    return dict(raw, schema='vosti.aligned-phases.v1', cases=cases,
                contexts=list(contexts), measured_waves=measured, warmup_waves=2)


def body(spec, prompt, rid, output):
    if spec['execution']['engine'] == 'sglang':
        data = dict(input_ids=prompt, rid=rid, stream=False,
            sampling_params=dict(temperature=0, top_p=1, max_new_tokens=output, ignore_eos=True))
    else:
        data = dict(model=spec['served_name'], prompt=prompt, request_id=rid,
            temperature=0, top_p=1, max_tokens=output, ignore_eos=True, stream=False)
    return json.dumps(data).encode()


async def wave(client, spec, payloads):
    engine = spec['execution']['engine']
    url = spec['base_url']+('/generate' if engine == 'sglang' else '/v1/completions')
    start = time.perf_counter()

    async def request(payload):
        launched = time.perf_counter()-start
        r = await client.post(url, content=payload, headers={'Content-Type':'application/json'})
        r.raise_for_status()
        return dict(launched_s=launched, received_s=time.perf_counter()-start, response=r.json())

    rows = await asyncio.gather(*(request(p) for p in payloads))
    return dict(client_wall_s=time.perf_counter()-start, requests=rows)


def check(record, case):
    c = Cohort(case['name'], case['cached_tokens'], case['query_tokens'])
    if (record['batches'] != c.outputs or record['prefix_lens'] != [c.cached]*4
            or record['query_lens'] != [c.query]*4 or len(set(record['request_ids'])) != 4
            or not math.isfinite(record['engine_wall_s']) or record['engine_wall_s'] <= 0):
        raise ValueError('incorrect batch/cache geometry or timing')
    decode = record.get('decode_batches', [])
    if len(decode) != c.outputs-1:
        raise ValueError('wrong number of measured decode steps')
    for index, batch in enumerate(decode, 1):
        if (set(batch['request_ids']) != set(record['request_ids'])
                or len(batch['request_ids']) != 4 or batch['seq_lens'] != [c.cached+index+1]*4):
            raise ValueError('decode batch/context drift')
    tokens = 508 if c.query == 1 else 4*c.query
    if record['measured_tokens'] != tokens:
        raise ValueError('wrong measured numerator')
    if (not math.isfinite(record['tokens_per_s'])
            or abs(record['tokens_per_s']-tokens/record['engine_wall_s']) > 1e-6):
        raise ValueError('rate disagrees with raw interval')


async def measure(client, spec, data, output):
    for case in data['cases']:
        c = Cohort(case['name'], case['cached_tokens'], case['query_tokens'])
        payloads = [body(spec, r['prompt'], rid, c.outputs) for r,rid in zip(case['rows'],c.ids())]
        donors = None
        if c.cached:
            donors = await wave(client, spec, [body(spec, r['donor'], f'donor{c.name}r{i}', 1)
                                             for i,r in enumerate(case['rows'])])
        result = await wave(client, spec, payloads)
        timing = json.loads((output/'records'/f'{c.name}.json').read_text())
        check(timing, case)
        for rid, row in zip(c.ids(), result['requests']):
            response = row['response']
            if spec['execution']['engine'] == 'sglang':
                meta = response['meta_info']
                cache = meta.get('cached_tokens')
                assert meta['id'] == rid
            else:
                meta = response['usage']
                cache = (meta.get('prompt_tokens_details') or {}).get('cached_tokens')
                assert response['id'] == f'cmpl-{rid}'
            if (cache != c.cached or meta['prompt_tokens'] != c.cached+c.query
                    or meta['completion_tokens'] != c.outputs):
                raise RuntimeError(f'HTTP/scheduler geometry disagrees: {meta}')
        write_new(output/f'{c.name}.json', dict(timing, phase=case['phase'], donor=donors, **result))
        print('WAVE', spec['execution']['key'], c.name, case['phase'],
              round(timing['engine_wall_s']*1000,3), 'ms', flush=True)


def summarize(data, output):
    groups = {}
    for case in data['cases']:
        record = json.loads((output/f'{case["name"]}.json').read_text())
        check(record, case)
        if case['phase'] == 'measured':
            groups.setdefault((case['kind'],case['cached_tokens']),[]).append(record)
    rows = []
    for (kind,cached),records in groups.items():
        if len(records) != data['measured_waves']:
            raise ValueError('missing measured wave')
        times = [r['engine_wall_s'] for r in records]
        median = statistics.median(times)
        rows.append(dict(kind=kind,cached_tokens=cached,query_tokens=records[0]['query_tokens'],
            batch_size=4, measured_waves=len(times), engine_wall_s=times,
            median_ms=median*1000, min_ms=min(times)*1000, max_ms=max(times)*1000,
            tokens_per_s=records[0]['measured_tokens']/median))
    write_new(output/'result.json',dict(complete=True,rows=rows,
        timing='Engine scheduling/forward/sampling; HTTP/admission excluded. '
        'Prefill Q256; decode 127 subsequent steps after Q1 first-token pass, normal overlap.'))
    return rows
