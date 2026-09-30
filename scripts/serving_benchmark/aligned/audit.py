"""Recompute aligned matrix tables from raw waves; reject incomplete coverage."""
import argparse
import csv
import json
from pathlib import Path
import statistics
import tarfile

from scripts.common.telemetry import load_complete_telemetry
from scripts.serving_benchmark.aligned.measure import check
from scripts.serving_benchmark.campaign import file_hash
from scripts.serving_benchmark.multi_turn import write_new

MODES = {'vosti-padded-graph','vllm-fast-auto','vllm-invariant-flash-attn',
         'vllm-invariant-triton-attn','sglang-fast-auto','sglang-deterministic-fa3',
         'sglang-deterministic-triton'}
MODELS = ('gemma3-4b','llama3-8b','gemma3-27b','gemma4-31b')


def read(path): return json.loads(Path(path).read_text())


def audit(root, require_full=False):
    plan, completion = read(root/'plan.json'),read(root/'complete.json')
    if not completion['complete'] or completion['trials'] != len(plan['jobs']):
        raise ValueError('incomplete campaign')
    # Audit the retained controller snapshot, not a later edited working tree.
    # External input/launch artifacts remain immutable and are checked in place.
    import hashlib
    checkout = Path(__file__).resolve().parents[3]
    with tarfile.open(root/'controller-source.tar.gz') as archive:
        names=set(archive.getnames())
        for path,digest in plan['files'].items():
            p=Path(path)
            name=str(p.relative_to(checkout)) if p.is_relative_to(checkout) else None
            actual=(hashlib.sha256(archive.extractfile(name).read()).hexdigest()
                    if name in names else file_hash(p))
            if actual!=digest: raise ValueError(f'source/input hash mismatch: {path}')
    build=plan['build']
    if file_hash(Path(build['binary']))!=build['binary_sha256']:
        raise ValueError('native binary changed')
    keys={(j['model'],j['mode']) for j in plan['jobs']}
    if len(keys)!=len(plan['jobs']): raise ValueError('duplicate trials')
    if require_full:
        expected={(model,mode) for model in MODELS for mode in
            (MODES if model!='gemma4-31b' else
             MODES-{'sglang-deterministic-fa3','vllm-invariant-flash-attn'} |
             {'vllm-invariant-fa3-local-triton-global'})}
        if (keys!=expected or plan['contexts']!=[0,4096,8192,12288,16384]
                or plan['measured_waves']!=5 or plan['warmup_waves']!=2):
            raise ValueError('does not cover the approved full matrix')
    table=[]; waves=0
    for job in plan['jobs']:
        trial=root/job['model']/job['mode'];run=trial/'run'
        load_complete_telemetry(trial/'telemetry.json')
        if read(trial/'exit.json')['returncode'] or not read(run/'trial-complete.json')['complete']:
            raise ValueError('failed trial')
        data=read(job['inputs'])
        if data['contexts']!=plan['contexts'] or data['measured_waves']!=plan['measured_waves']:
            raise ValueError('inconsistent workload')
        spec=read(trial/'launch.json');native=spec['execution']['engine']=='vosti'
        if (type(spec['settings']['gpu_index']) is not int
                or spec['settings']['gpu_index'] not in plan['gpus']):
            raise ValueError('invalid GPU index')
        if not native:
            receipts=list((run/'records').glob('installed-*.json'))
            if not receipts: raise ValueError('missing installed hook evidence')
            for receipt in receipts:
                r=read(receipt);source=r.get('source',r.get('scheduler_source'))
                if file_hash(Path(source))!=r['sha256']: raise ValueError('engine source changed')
        groups={}
        for case in data['cases']:
            r=read(run/f'{case["name"]}.json');check(r,case);waves+=1
            cached,query=case['cached_tokens'],case['query_tokens']
            outputs=128 if query==1 else 1
            if native:
                if len(r['generated_ids'])!=4 or any(len(x)!=outputs for x in r['generated_ids']):
                    raise ValueError('native output length mismatch')
                if case['phase']=='measured':
                    before,after=r['graph_before'],r['graph_after']
                    if before['capture_count']!=after['capture_count'] or after['poisoned_reason'] is not None:
                        raise ValueError('native graph capture/poison')
                    if query==1:
                        b=r['graph_decode_before']
                        if after['replay_count']-b['replay_count']!=127 or after['eager_count']!=b['eager_count']:
                            raise ValueError('native measured decode not entirely graph replay')
            else:
                if len(r['requests'])!=4: raise ValueError('missing responses')
                for response in r['requests']:
                    payload=response['response']
                    if spec['execution']['engine']=='sglang':
                        meta=payload['meta_info'];cache=meta.get('cached_tokens')
                    else:
                        meta=payload['usage'];cache=(meta.get('prompt_tokens_details') or {}).get('cached_tokens')
                    if (cache!=cached or meta['prompt_tokens']!=cached+query or meta['completion_tokens']!=outputs):
                        raise ValueError('response accounting mismatch')
            if r['phase']!=case['phase']: raise ValueError('wave role mismatch')
            groups.setdefault((case['kind'],cached),{'warmup':[],'measured':[]})[case['phase']].append(r)
        retained={(r['kind'],r['cached_tokens']):r for r in read(run/'result.json')['rows']}
        for key,phases in groups.items():
            if len(phases['warmup'])!=2 or len(phases['measured'])!=plan['measured_waves']:
                raise ValueError('wrong repetition coverage')
            measured=phases['measured'];times=[r['engine_wall_s'] for r in measured]
            median=statistics.median(times);rate=measured[0]['measured_tokens']/median
            if abs(retained[key]['tokens_per_s']-rate)>1e-6 or retained[key]['engine_wall_s']!=times:
                raise ValueError('retained summary disagrees with raw intervals')
            table.append(dict(model=job['model'],mode=job['mode'],gpu=spec['settings']['gpu_index'],
                kind=key[0],cached_tokens=key[1],query_tokens=measured[0]['query_tokens'],
                batch_size=4,waves=len(times),median_ms=median*1000,min_ms=min(times)*1000,
                max_ms=max(times)*1000,tokens_per_s=rate))
    if len(table)!=plan['expected_points']: raise ValueError('missing points')
    return dict(complete=True,full_matrix=require_full,trials=len(keys),points=len(table),
                checked_waves=waves,rows=table)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('root',type=Path)
    p.add_argument('--require-full',action='store_true')
    p.add_argument('--output',type=Path)
    a=p.parse_args();result=audit(a.root.resolve(),a.require_full)
    if a.output:
        a.output.mkdir(parents=True,exist_ok=False)
        write_new(a.output/'audit.json',result)
        with (a.output/'measurements.csv').open('x',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(result['rows'][0]))
            writer.writeheader();writer.writerows(result['rows'])
        lines=['# Aligned controlled-phase results','',
            'Actual batch size four; new-query tokens/s for prefill, subsequent decode tokens/s for decode.',
            'Engine intervals exclude HTTP/admission waiting; contexts denote starting cached lengths.','']
        for model in MODELS:
            for kind in ('prefill','decode'):
                rows=[r for r in result['rows'] if r['model']==model and r['kind']==kind]
                if not rows: continue
                contexts=sorted({r['cached_tokens'] for r in rows})
                lines += [f'## {model}: {kind} tokens/s','',
                    '| Mode | '+' | '.join(str(c) for c in contexts)+' |',
                    '|---|'+'---:|'*len(contexts)]
                for mode in sorted({r['mode'] for r in rows}):
                    values={r['cached_tokens']:r['tokens_per_s'] for r in rows if r['mode']==mode}
                    lines.append('| '+mode+' | '+' | '.join(f'{values[c]:,.1f}' for c in contexts)+' |')
                lines.append('')
        (a.output/'REPORT.md').write_text('\n'.join(lines))
    print(json.dumps({k:v for k,v in result.items() if k!='rows'}))


if __name__=='__main__': main()
