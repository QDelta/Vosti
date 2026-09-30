"""CPU regressions for the opt-in controlled-phase scheduler adapter."""
import importlib
import json
import sys
from types import SimpleNamespace as NS

import pytest

from scripts.serving_benchmark.aligned.protocol import Cohort, admission, check_batch, identify


def test_admission():
    c = Cohort('wave0', 8192, 256)
    assert admission([]) is None
    assert admission(['ordinary']) is None
    for n in range(1, 4):
        assert admission(c.ids()[:n]) is False
    assert admission(c.ids()[::-1]) == c
    for ids in [c.ids()+['ordinary'], c.ids()[:2]*2,
                c.ids()[:3]+Cohort('other', 8192, 256).ids()[:1]]:
        with pytest.raises(ValueError):
            admission(ids)
    for rid in ['aligned_../bad_0_256_0', 'aligned_bad_0_256_4', 'aligned_bad_0_0_0']:
        with pytest.raises(ValueError):
            identify(rid)


def test_actual_geometry():
    c = Cohort('wave0', 8192, 256)
    check_batch(c, c.ids(), [8192]*4, [256]*4, is_extend=True)
    for ids, prefix, query, extend in [
        (c.ids()[:1], [8192], [256], True),
        (c.ids(), [8191]*4, [257]*4, True),
        (c.ids(), [8192]*4, [1]*4, False),
    ]:
        with pytest.raises(ValueError):
            check_batch(c, ids, prefix, query, is_extend=extend)


@pytest.mark.parametrize('overlap', [False, True])
def test_hook_barrier_and_timing(tmp_path, monkeypatch, overlap):
    hook = importlib.import_module('scripts.serving_benchmark.aligned.sglang_hook')
    clock = [1.0]
    calls = []
    monkeypatch.setattr(hook.time, 'perf_counter', lambda: clock[0])
    monkeypatch.setenv('VOSTI_ALIGNED_PHASE_RECORDS', str(tmp_path))
    monkeypatch.setitem(sys.modules, 'torch', NS(cuda=NS(synchronize=lambda: calls.append('sync'))))

    class Scheduler:
        enable_overlap = overlap

        def get_next_batch_to_run(self, *, running_batch, last_batch):
            calls.append('schedule')
            batch = NS(reqs=self.waiting_queue, prefix_lens=[8192]*4,
                extend_lens=[256]*4, forward_mode=NS(is_extend=lambda: True))
            self.waiting_queue = []
            return NS(batch_to_run=batch, running_batch=running_batch)

        def run_batch(self, batch):
            calls.append('forward')
            clock[0] += .125
            return 'result'

        def launch_batch_sample_if_needed(self, result, batch):
            calls.append('sample')
            clock[0] += .025

    hook.install(NS(Scheduler=Scheduler, NextBatchPlan=NS, __file__=__file__))
    scheduler = Scheduler()
    empty = NS(reqs=[])
    for n in range(2):
        cohort = Cohort(f'wave{n}', 8192, 256)
        requests = [NS(rid=rid, finished=lambda: False) for rid in cohort.ids()]
        if n == 0:
            scheduler.waiting_queue = requests[:1]
            assert scheduler.get_next_batch_to_run(empty, None).batch_to_run is None
            assert calls == []
            clock[0] += .2
        scheduler.waiting_queue = requests
        batch = scheduler.get_next_batch_to_run(empty, None).batch_to_run
        assert scheduler.run_batch(batch) == 'result'
        if overlap:
            assert not (tmp_path/f'wave{n}.json').exists()
            scheduler.launch_batch_sample_if_needed('result', batch)
        record = json.loads((tmp_path/f'wave{n}.json').read_text())
        assert record['engine_wall_s'] == pytest.approx(.150 if overlap else .125)
        assert record['admission_wait_s'] == pytest.approx(.2 if n == 0 else 0)
        assert record['prefix_lens'] == [8192]*4
        assert record['batches'] == 1
        if overlap:
            for req in requests:
                req.sampling_params = NS(max_new_tokens=1)
            decode = NS(reqs=requests,
                forward_mode=NS(is_extend=lambda: False, is_decode=lambda: True))
            scheduler.run_batch(decode)
            scheduler.launch_batch_sample_if_needed('result', decode)
            assert json.loads((tmp_path/f'lookahead-wave{n}.json').read_text())['batch_size'] == 4
            with pytest.raises(RuntimeError):
                scheduler.run_batch(decode)
    assert calls.count('schedule') == 2
    assert calls.count('sync') == 4


@pytest.mark.parametrize('overlap', [False, True])
def test_sglang_continuous_decode(tmp_path, monkeypatch, overlap):
    hook = importlib.import_module('scripts.serving_benchmark.aligned.sglang_hook')
    clock, syncs = [0.0], []
    monkeypatch.setattr(hook.time, 'perf_counter', lambda: clock[0])
    monkeypatch.setenv('VOSTI_ALIGNED_PHASE_RECORDS', str(tmp_path))
    monkeypatch.setitem(sys.modules, 'torch', NS(cuda=NS(synchronize=lambda: syncs.append(clock[0]))))
    cohort = Cohort('decode', 4096, 1)
    reqs = [NS(rid=rid, finished=lambda: False) for rid in cohort.ids()]
    class Scheduler:
        enable_overlap = overlap
        def get_next_batch_to_run(self, *, running_batch, last_batch):
            self.waiting_queue=[]
            return NS(batch_to_run=self.next, running_batch=running_batch)
        def run_batch(self,batch): clock[0] += .01
        def launch_batch_sample_if_needed(self,result,batch): pass
    hook.install(NS(Scheduler=Scheduler,NextBatchPlan=NS,__file__=__file__))
    s=Scheduler();s.waiting_queue=reqs
    for step in range(128):
        s.next=NS(reqs=reqs,prefix_lens=[4096]*4,extend_lens=[1]*4,
            forward_mode=NS(is_extend=lambda: step==0,is_decode=lambda: step>0),
            seq_lens_cpu=NS(tolist=lambda: [4096+step+1]*4))
        batch=s.get_next_batch_to_run(NS(reqs=[]),None).batch_to_run
        s.run_batch(batch)
        if overlap: s.launch_batch_sample_if_needed(None,batch)
    record=json.loads((tmp_path/'decode.json').read_text())
    assert record['engine_wall_s'] == pytest.approx(1.27)
    assert record['prefill_wall_s'] == pytest.approx(.01)
    assert record['batches']==128 and len(record['decode_batches'])==127
    assert len(syncs)==3  # initial boundary, prefill completion, final decode completion


def test_vllm_ids():
    from scripts.serving_benchmark.aligned.vllm_hook import canonical
    assert canonical('cmpl-aligned_wave_0_256_0-0') == 'aligned_wave_0_256_0'
    assert canonical('cmpl-aligned_wave_0_256_0-0-deadbeef') == 'aligned_wave_0_256_0'
    with pytest.raises(ValueError): canonical('cmpl-aligned_bad-unrecognized')
    assert canonical('ordinary') == 'ordinary'


def test_result_rejects_decode_drift():
    from scripts.serving_benchmark.aligned.measure import check
    c=dict(name='test',cached_tokens=4096,query_tokens=1)
    ids=[0,1,2,3]
    r=dict(batches=128,prefix_lens=[4096]*4,query_lens=[1]*4,request_ids=ids,
        engine_wall_s=1,measured_tokens=508,tokens_per_s=508,
        decode_batches=[dict(request_ids=ids,seq_lens=[4096+i+1]*4) for i in range(1,128)])
    check(r,c)
    r['decode_batches'][50]['request_ids']=[0,1,2]
    with pytest.raises(ValueError): check(r,c)


@pytest.mark.parametrize('field', ['engine_wall_s','tokens_per_s'])
@pytest.mark.parametrize('value', [float('nan'),float('inf')])
def test_result_rejects_nonfinite(field,value):
    from scripts.serving_benchmark.aligned.measure import check
    c=dict(name='test',cached_tokens=0,query_tokens=256)
    r=dict(batches=1,prefix_lens=[0]*4,query_lens=[256]*4,request_ids=[0,1,2,3],
        engine_wall_s=1,measured_tokens=1024,tokens_per_s=1024,decode_batches=[])
    r[field]=value
    with pytest.raises(ValueError): check(r,c)


@pytest.mark.parametrize('query', [1,256])
def test_vllm_barrier_and_continuous_timing(tmp_path, monkeypatch, query):
    from scripts.serving_benchmark.aligned import vllm_hook as hook
    clock,syncs=[0.0],[]
    monkeypatch.setattr(hook.time,'perf_counter',lambda:clock[0])
    monkeypatch.setenv('VOSTI_ALIGNED_PHASE_RECORDS',str(tmp_path))
    monkeypatch.setitem(sys.modules,'torch',NS(cuda=NS(is_initialized=lambda:True,
        synchronize=lambda:syncs.append(clock[0]))))
    class UniProcExecutor:
        def sample_tokens(self): clock[0]+=.01
    monkeypatch.setitem(sys.modules,'vllm.v1.executor.uniproc_executor',NS(UniProcExecutor=UniProcExecutor))
    class Scheduler:
        def __init__(self): self.requests={};self.steps=0
        def schedule(self):
            n=query if self.steps==0 else 1
            for r in self.requests.values(): r.num_computed_tokens+=n
            self.steps+=1
            return NS(num_scheduled_tokens={rid:n for rid in self.requests},
                preempted_req_ids=[],scheduled_spec_decode_tokens={})
    class EngineCore:
        def __init__(self):
            self.model_executor=UniProcExecutor();self.scheduler=Scheduler()
            self.async_scheduling=True;self.batch_queue_size=2
        def add_request(self,r): self.scheduler.requests[r.request_id]=r
    hook.install(NS(EngineCore=EngineCore,__file__=__file__))
    e=EngineCore();c=Cohort('test',4096,query)
    for index,rid in enumerate(c.ids()):
        e.add_request(NS(request_id=f'cmpl-{rid}-0',max_tokens=c.outputs,
            num_prompt_tokens=4096+query,num_computed_tokens=4096))
        assert len(e.scheduler.requests) == (4 if index==3 else 0)
    for _ in range(c.outputs):
        e.scheduler.schedule();e.model_executor.sample_tokens()
    record=json.loads((tmp_path/'test.json').read_text())
    assert record['engine_wall_s']==pytest.approx(1.27 if query==1 else .01)
    assert record['batches']==c.outputs
    assert len(syncs)==(3 if query==1 else 2)
