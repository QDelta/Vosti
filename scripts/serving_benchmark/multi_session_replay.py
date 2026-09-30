"""Seeded multi-session replay preset and pure timing/report helpers.

Execution uses the general multi-turn serving runner; historical campaign
controllers and their private deployment directories are not required.
"""
import copy
from pathlib import Path
import random
import statistics

from scripts.serving_benchmark.multi_turn import (
    SCHEMA, exact_text, prepare_arrival_trace, validate_workload,
)
from scripts.common.tokenizers import tokenizer_artifact_sha256

METRICS = ('wall_s', 'initial_ttft_ms', 'followup_ttft_ms', 'followup_tpot_ms')


def prepare_inputs(tokenizer, checkpoint, seed=42):
    if type(seed) is not int or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    rng = random.Random(seed)
    shared = exact_text(tokenizer,8193,rng,special=True)
    common = tokenizer.encode(shared)
    sessions = []
    for sid in range(4):
        prompt = exact_text(tokenizer,16385,rng,special=True,prefix=shared+'\n\n')
        assert tokenizer.encode(prompt)[:8193] == common
        turns = [dict(suffix='',suffix_tokens=0,max_tokens=768,delay_s=0.0)]
        for _ in range(5):
            suffix=exact_text(tokenizer,256,rng,special=False,prefix='\n\n')
            turns.append(dict(suffix=suffix,suffix_tokens=256,max_tokens=768,delay_s=.5))
        sessions.append(dict(id=str(sid),initial_prompt=prompt,initial_tokens=16385,turns=turns))
    measured=dict(schema=SCHEMA,seed=seed,sessions=sessions,tokenizer=checkpoint.path,
        tokenizer_artifact_sha256=tokenizer_artifact_sha256(Path(checkpoint.path)),
        shared_prefix_tokens=8193,description='Shared-prefix-warm; 4 independent sessions x 6 turns; real output text retained.')
    warm_rng=random.Random(seed+1)
    warm_prompt=exact_text(tokenizer,16385,warm_rng,special=True)
    assert tokenizer.encode(warm_prompt)[:64] != common[:64]
    # A common cached donor makes all four decode rows start together. Sweep
    # every key-width bucket used by the measured six turns, plus one page.
    # The four rows then cover smaller decode batches through padded graphs.
    warmup=copy.deepcopy(measured)
    warmup.update(seed=seed+1,description='Excluded decode-shape sweep; separate prefix from measurement.')
    warmup['sessions']=[dict(id=str(sid),initial_prompt=warm_prompt,initial_tokens=16385,
        turns=[dict(suffix='',suffix_tokens=0,max_tokens=5952,delay_s=0.0)]) for sid in range(4)]
    prefix_warmup=exact_text(tokenizer,8197,rng,special=True,prefix=shared+'\n\n')
    assert tokenizer.encode(prefix_warmup)[:8193]==common
    for data in (measured,warmup): validate_workload(data,tokenizer,32768)
    return dict(measured=measured,warmup=warmup,shared_prefix_warmup=prefix_warmup,
        donor_warmup=warm_prompt,arrivals=prepare_arrival_trace(measured,request_rate=2.0,seed=seed))



def union_seconds(intervals):
    end, total = None, 0.0
    for a, b in sorted(intervals):
        total += b-a if end is None or a > end else max(0, b-end)
        end = b if end is None else max(end, b)
    return total


def summarize(rows):
    prefill, generation = [], []
    for r in rows:
        a, m, b = (r['send_started_offset_s'], r['first_output_event_offset_s'], r['completed_offset_s'])
        if m is not None:
            prefill.append((a, m))
            generation.append((m, b))
    return dict(requests=len(rows), successful=sum(r['success'] for r in rows),
        prompt_tokens=sum(r['prompt_tokens'] for r in rows),
        output_tokens=sum(r['output_tokens'] for r in rows),
        cached_tokens=sum(r['server_cached_prompt_tokens'] or 0 for r in rows),
        uncached_tokens=sum(r['prompt_tokens']-(r['server_cached_prompt_tokens'] or 0) for r in rows),
        prefill_wait_sum_s=sum(b-a for a,b in prefill), prefill_wait_union_s=union_seconds(prefill),
        generation_sum_s=sum(b-a for a,b in generation), generation_union_s=union_seconds(generation),
        request_union_s=union_seconds([(r['send_started_offset_s'],r['completed_offset_s']) for r in rows]),
        mean_ttft_ms=statistics.mean([r['ttft_s']*1000 for r in rows if r['ttft_s'] is not None])
            if prefill else None,
        mean_tpot_ms=statistics.mean([r['tpot_s']*1000 for r in rows if r['tpot_s'] is not None])
            if any(r['tpot_s'] is not None for r in rows) else None)



def compare_reports(current, reference):
    assert current['source']==reference['source']
    assert current['unsupported']==reference['unsupported']
    previous={(r['model'],r['mode']):r for r in reference['rows'] if r['status']=='complete'}
    complete=[r for r in current['rows'] if r['status']=='complete']
    best={(r['model'],metric):min(x[metric] for x in complete if x['model']==r['model'])
          for r in complete for metric in METRICS}
    rows=[]
    for row in complete:
        before=previous[(row['model'],row['mode'])]
        rows.append(dict(model=row['model'],mode=row['mode'],
            current={k:row[k] for k in METRICS},reference={k:before[k] for k in METRICS},
            geometry_warning=row.get('geometry_warning'),
            change_pct={k:100*(row[k]/before[k]-1) for k in METRICS},
            pct_of_current_best={k:100*row[k]/best[row['model'],k] for k in METRICS}))
    return rows
