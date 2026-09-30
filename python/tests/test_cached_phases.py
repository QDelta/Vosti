import copy
from pathlib import Path
from unittest.mock import patch

import pytest

from scripts.serving_benchmark.cached_phases import prepare,validate,observed_cache,wave_metrics
from scripts.serving_benchmark.aligned.launch import adapt


class Tokenizer:
    all_special_ids=[0]
    def encode(self,text,add_special_tokens=False):return list(range(1,200))
    def get_vocab(self):return {str(i):i for i in range(1000)}


def workload():
    with patch('scripts.serving_benchmark.cached_phases.tokenizer_artifact_sha256',return_value={}):
        return prepare(Tokenizer(),dict(path='/model'),contexts=(0,4096),waves=2)


def test_seeded_geometry_and_divergent_donor():
    a=workload();assert a==workload();validate(a)
    for c in a['cells']:
        for r in c['warmup']+c['measured']:
            n=c['cached_tokens']
            assert len(r['prompt'])==n+c['query_tokens']
            if n:
                assert len(r['donor'])==n+1
                assert r['donor'][:n]==r['prompt'][:n]
                assert r['donor'][n]!=r['prompt'][n]


def test_matching_extra_token_and_short_donor_rejected():
    for change in ['same','short']:
        d=workload();c=d['cells'][-1];r=c['measured'][0];n=c['cached_tokens']
        if change=='same':r['donor'][n]=r['prompt'][n]
        else:r['donor']=r['donor'][:n]
        with pytest.raises(ValueError,match='diverge'):validate(d)


def test_single_token_overlap_rejected_even_with_different_remaining_prefix():
    d=workload();a,b=d['cells'][0]['measured'][:2]
    b['prompt'][0]=a['prompt'][0]
    assert b['prompt'][1:16]!=a['prompt'][1:16]
    with pytest.raises(ValueError,match='cross-request prefix reuse'):validate(d)


def test_unknown_cache_not_silently_zero_except_pinned_sglang_schema():
    row=dict(server_cached_prompt_tokens=None,server_usage=dict(prompt_tokens=1))
    assert observed_cache(row,'vosti')[0] is None
    assert observed_cache(row,'sglang')[0] is None
    assert observed_cache(row,'sglang',sglang_zero_omission_verified=True)==(0,'zero_omitted_by_pinned_sglang_usage_schema')
    row['server_cached_prompt_tokens']=4096
    assert observed_cache(row,'sglang')[0]==4096


def test_decode_uses_union_not_summed_or_whole_request_times():
    rows=[dict(success=True,first_output_event_offset_s=a,last_output_event_offset_s=b,output_tokens=128)
          for a,b in [(2,6),(3,7),(2,6),(3,7)]]
    m=wave_metrics(rows,8,1)
    assert m['decode_active_union_s']==5
    assert m['decode_tokens']==4*127
    assert m['decode_tokens_per_active_s']==4*127/5
    assert m['decode_start_spread_s']==1
    m=wave_metrics(rows,8,256)
    assert m['new_input_tokens_per_s']==4*256/8


def test_failed_wave_not_reported_as_throughput():
    with pytest.raises(ValueError):wave_metrics([dict(success=False)],1,256)


@pytest.mark.parametrize('engine,flags',[
    ('vosti',[]),('vllm',['--port','1','--max-model-len','1','--max-num-seqs','1','--max-num-batched-tokens','1']),
    ('sglang',['--port','1','--context-length','1','--max-running-requests','1','--max-prefill-tokens','1','--chunked-prefill-size','1'])])
def test_both_gpu_specs_keep_backend_and_capacity(engine,flags):
    original=dict(command=['server']+flags,settings={},environment={},
        execution=dict(engine=engine,key='mode',attention_backend='qualified'))
    original_copy=copy.deepcopy(original)
    s=adapt(original,2,Path('/cache'),Path('/frozen'))
    assert original==original_copy
    assert s['execution']==original['execution']
    assert s['environment']['CUDA_VISIBLE_DEVICES']=='2'
    assert s['settings']['max_batched_tokens']==1024
    assert s['settings']['vosti_num_blocks']==1152
    assert s['base_url']=='http://127.0.0.1:18402'


def test_sglang_deterministic_triton_chunk_admits_default_alignment():
    original=dict(command=['server','--port','1','--context-length','1',
        '--max-running-requests','1','--max-prefill-tokens','1','--chunked-prefill-size','1'],
        settings={},environment={},execution=dict(engine='sglang',key='sglang-deterministic-triton'))
    spec=adapt(original,3,Path('/cache'),Path('/frozen'))
    for flag in ['--max-prefill-tokens','--chunked-prefill-size']:
        assert spec['command'][spec['command'].index(flag)+1]=='4096'
    assert spec['settings']['max_batched_tokens']==4096
    assert 'SGLANG_TRITON_PREFILL_TRUNCATION_ALIGN_SIZE' not in spec['environment']
