"""Exact cached-KV phase sweep; token IDs avoid tokenizer boundary drift."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
import random
import time

from scripts.common.timing import interval_union_s
from scripts.serving_benchmark.multi_turn import write_new
from scripts.common.tokenizers import tokenizer_artifact_sha256
from scripts.serving_benchmark.run import read_server_metrics, send_request

CONTEXTS = (0, 4096, 8192, 12288, 16384)


def prepare(tokenizer, checkpoint, seed=42, contexts=CONTEXTS, waves=2):
    pool = sorted(set(tokenizer.encode(
        ' '.join(f' word{i} alpha beta gamma delta number{i}' for i in range(200)),
        add_special_tokens=False)) - set(tokenizer.all_special_ids))
    if len(pool) < 12:
        raise ValueError('insufficient ordinary token IDs')
    rng = random.Random(seed)
    # SGLang can reuse even one token. Distinct multi-token prefixes alone
    # do not isolate waves from prior requests in a token-granular radix cache.
    starters = sorted(set(tokenizer.get_vocab().values()) - set(tokenizer.all_special_ids))
    needed = len(contexts)*2*(1+waves)*4
    if len(starters) < needed:
        raise ValueError('insufficient distinct first tokens for isolated requests')
    starters = iter(rng.sample(starters,needed))
    cells = []
    for context in contexts:
        if context < 0 or context % 4096:
            raise ValueError('expected a nonnegative 4K-aligned cache length')
        for kind, query, output in [('prefill',256,1),('decode',1,128)]:
            cell = dict(id=f'{kind}-c{context}', kind=kind, cached_tokens=context,
                        query_tokens=query, output_tokens=output, warmup=[], measured=[])
            for phase, count in [('warmup',1),('measured',waves)]:
                for index in range(count*4):
                    prompt = [next(starters)] + rng.choices(pool, k=context+query-1)
                    donor = None
                    if context:
                        tail = rng.choice([t for t in pool if t != prompt[context]])
                        donor = prompt[:context]+[tail]
                    cell[phase].append(dict(index=index,prompt=prompt,
                        prompt_tokens=len(prompt),max_tokens=output,donor=donor))
            cells.append(cell)
    return dict(schema='vosti.cached-phase-input.v2',checkpoint=checkpoint,
        tokenizer_artifact_sha256=tokenizer_artifact_sha256(Path(checkpoint['path'])),
        seed=seed,concurrency=4,waves=waves,cells=cells,
        donor_policy='L+1 token donor; measured query diverges after exactly L tokens')


def validate(data):
    seen = set()
    for c in data['cells']:
        for phase, count in [('warmup',1),('measured',data['waves'])]:
            if len(c[phase]) != count*data['concurrency']:
                raise ValueError('wrong wave size')
            for r in c[phase]:
                length = c['cached_tokens']
                if len(r['prompt']) != length+c['query_tokens'] or r['prompt_tokens'] != len(r['prompt']):
                    raise ValueError('prompt geometry mismatch')
                if r['max_tokens'] != c['output_tokens']:
                    raise ValueError('output geometry mismatch')
                key = r['prompt'][0]
                if key in seen:
                    raise ValueError('accidental cross-request prefix reuse')
                seen.add(key)
                donor = r['donor']
                if length:
                    if (len(donor) != length+1 or donor[:length] != r['prompt'][:length]
                            or donor[length] == r['prompt'][length]):
                        raise ValueError('donor must diverge after exactly L cached tokens')
                elif donor is not None:
                    raise ValueError('zero cache must not have a donor')


def observed_cache(record, engine, *, sglang_zero_omission_verified=False):
    value = record['server_cached_prompt_tokens']
    source = 'explicit_usage'
    # Pinned SGLang UsageProcessor._details_if_cached omits the details object
    # precisely for zero counts. Do not generalize this inference to engines
    # or versions that have not been source-checked by the coordinator.
    if value is None and engine == 'sglang' and sglang_zero_omission_verified:
        usage = record['server_usage']
        if usage.get('prompt_tokens_details') is None:
            value, source = 0, 'zero_omitted_by_pinned_sglang_usage_schema'
    return value, source


def wave_metrics(records, elapsed, query):
    if not records or not all(r['success'] for r in records):
        raise ValueError('cannot summarize failed wave')
    intervals = [(r['first_output_event_offset_s'],r['last_output_event_offset_s']) for r in records]
    decode_seconds = None
    if all(a is not None and b is not None and b >= a for a,b in intervals):
        decode_seconds = interval_union_s(intervals)
    decode_tokens = sum(max(0,r['output_tokens']-1) for r in records)
    return dict(wave_wall_s=elapsed,new_input_tokens=len(records)*query,
        new_input_tokens_per_s=len(records)*query/elapsed,
        decode_tokens=decode_tokens,decode_active_union_s=decode_seconds,
        decode_tokens_per_active_s=decode_tokens/decode_seconds if decode_seconds else None,
        decode_start_spread_s=max(a for a,b in intervals)-min(a for a,b in intervals)
            if all(a is not None for a,b in intervals) else None,
        output_tokens_per_wave_s=sum(r['output_tokens'] for r in records)/elapsed)


async def measure(client, spec, data, output):
    engine = spec['execution']['engine']
    zero_verified = spec.get('benchmark_sglang_zero_omission_verified',False)
    async def wave(rows):
        start = time.perf_counter()
        records = await asyncio.gather(*(send_request(client,base_url=spec['base_url'],
            endpoint='completions',model=spec['served_name'],request=r,index=r['index'],
            scheduled_offset_s=0,benchmark_start=start,require_usage=True) for r in rows))
        return records,time.perf_counter()-start
    all_results=[]
    for cell in data['cells']:
        cell_output=output/cell['id'];cell_output.mkdir()
        phases={}
        for phase in ['warmup','measured']:
            waves=[]
            for offset in range(0,len(cell[phase]),data['concurrency']):
                rows=cell[phase][offset:offset+data['concurrency']]
                donors=[];donor_s=0
                if cell['cached_tokens']:
                    donor_rows=[dict(index=r['index'],prompt=r['donor'],
                        prompt_tokens=len(r['donor']),max_tokens=1) for r in rows]
                    donors,donor_s=await wave(donor_rows)
                    if not all(r['success'] for r in donors):
                        write_new(cell_output/f'{phase}-{offset}-donor-failed.json',donors)
                        raise RuntimeError('donor failed')
                before=await read_server_metrics(client,spec['base_url'])
                records,elapsed=await wave(rows)
                after=await read_server_metrics(client,spec['base_url'])
                for r in records:
                    actual,source=observed_cache(r,engine,sglang_zero_omission_verified=zero_verified)
                    r['cache_evidence']=dict(actual=actual,source=source,expected=cell['cached_tokens'],
                        exact=actual==cell['cached_tokens'])
                result=dict(records=records,donors=donors,donor_wall_s=donor_s,
                    metrics=wave_metrics(records,elapsed,cell['query_tokens']) if all(r['success'] for r in records) else None,
                    server_before=before,server_after=after)
                write_new(cell_output/f'{phase}-{offset//data["concurrency"]}.json',result)
                if not all(r['success'] and r['cache_evidence']['exact'] for r in records):
                    raise RuntimeError(f'{cell["id"]}: request failure or non-exact cache geometry')
                if engine=='vosti' and phase=='measured':
                    a,b=before['engine']['cuda_graph'],after['engine']['cuda_graph']
                    result['graph_delta']={k:b[k]-a[k] for k in ['capture_count','replay_count','cover_replay_count','eager_count']}
                    if b['poisoned_reason'] is not None or result['graph_delta']['capture_count']:
                        raise RuntimeError(f'{cell["id"]}: native graph capture/poison in measured wave')
                waves.append(result)
                print('WAVE',cell['id'],phase,offset//4,'cache',cell['cached_tokens'],
                    'seconds',round(elapsed,3),flush=True)
            phases[phase]=waves
        measured=phases['measured']
        wall=sum(w['metrics']['wave_wall_s'] for w in measured)
        decode_s=sum(w['metrics']['decode_active_union_s'] or 0 for w in measured)
        row=dict(id=cell['id'],kind=cell['kind'],cached_tokens=cell['cached_tokens'],
            query_tokens=cell['query_tokens'],concurrency=4,measured_requests=len(measured)*4,
            prefill_tokens_per_s=sum(w['metrics']['new_input_tokens'] for w in measured)/wall
                if cell['kind']=='prefill' else None,
            decode_tokens_per_s=sum(w['metrics']['decode_tokens'] for w in measured)/decode_s
                if cell['kind']=='decode' and decode_s else None,
            measured_wall_s=wall,phases=phases)
        write_new(cell_output/'result.json',row);all_results.append(row)
        print('CELL COMPLETE',cell['id'],'prefill',row['prefill_tokens_per_s'],
              'decode',row['decode_tokens_per_s'],flush=True)
    write_new(output/'result.json',dict(complete=True,rows=all_results,
        timing='Prefill: new query tokens / concurrent wave wall (includes first-token and HTTP overhead). '
        'Decode: subsequent output tokens / union of first-to-last visible-output intervals; '
        'excludes initial prefill-only time, not overlapping prefill of lagging requests or HTTP overhead. '
        'Concurrency is not a physical batch guarantee. KV grows during generation.'))
