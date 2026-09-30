"""Single-GPU vLLM adapter, preserving its default asynchronous scheduling.

Admission is buffered before the scheduler until the entire cohort arrives.
Timing surrounds normal schedule/execute/sample, not HTTP or output formatting.
Only interval boundaries synchronize; no per-decode-step GPU synchronization.
"""
import hashlib
import json
import os
import re
from pathlib import Path
import time

from scripts.serving_benchmark.aligned.protocol import admission, identify, check_batch


def canonical(rid):
    if rid.startswith('cmpl-aligned_'):
        # The API appends completion index -0, then InputProcessor appends an
        # eight-hex internal uniqueness suffix. Do not disable that behavior.
        match = re.fullmatch(r'cmpl-(aligned_[A-Za-z0-9]+_\d+_\d+_[0-3])-0(?:-[0-9a-f]{8})?',rid)
        if match is None:
            raise ValueError(f'unrecognized vLLM aligned request identity: {rid}')
        return match[1]
    return rid


def install(module):
    import torch
    from vllm.v1.executor.uniproc_executor import UniProcExecutor

    cls = module.EngineCore
    if getattr(cls, '_aligned_hook', False):
        return
    directory = Path(os.environ['VOSTI_ALIGNED_PHASE_RECORDS'])
    original_init, original_add = cls.__init__, cls.add_request

    def init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        if type(self.model_executor) is not UniProcExecutor or not torch.cuda.is_initialized():
            raise RuntimeError('aligned vLLM adapter requires the existing in-process single-GPU executor')
        directory.mkdir(parents=True, exist_ok=True)
        (directory/f'installed-{os.getpid()}.json').write_text(json.dumps(dict(
            source=module.__file__, sha256=hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest(),
            async_scheduling=self.async_scheduling, batch_queue_size=self.batch_queue_size,
            executor=type(self.model_executor).__name__)))
        self._aligned_pending = []
        self._aligned_state = None
        self._aligned_completed = {}
        schedule, sample = self.scheduler.schedule, self.model_executor.sample_tokens

        def scheduled(*args, **kwargs):
            state = self._aligned_state
            if state is not None and 'started' not in state:
                torch.cuda.synchronize()
                state['started'] = time.perf_counter()
            if state is not None and state.get('prefill_finished') and 'decode_started' not in state:
                state['decode_started'] = time.perf_counter()
            result = schedule(*args, **kwargs)
            ids = list(result.num_scheduled_tokens)
            normalized = [canonical(rid) for rid in ids]
            self._aligned_current = False
            if any(identify(rid) is not None for rid in normalized):
                cohort = admission(normalized)
                if state is None:
                    done = self._aligned_completed.get(getattr(cohort, 'name', None))
                    if done is None or done['lookahead'] or set(result.num_scheduled_tokens.values()) != {1}:
                        raise RuntimeError('unexpected vLLM execution after interval completion')
                    done['lookahead'] += 1
                    (directory/f'lookahead-{cohort.name}.json').write_text(json.dumps(dict(
                        request_ids=normalized, included_in_timing=False)))
                else:
                    if cohort != state['cohort']:
                        raise RuntimeError('actual vLLM batch split/mixed')
                    queries = [result.num_scheduled_tokens[rid] for rid in ids]
                    prefixes = [self.scheduler.requests[rid].num_computed_tokens-q
                                for rid,q in zip(ids,queries)]
                    if state['batches'] == 0:
                        check_batch(cohort, normalized, prefixes, queries, is_extend=True)
                        state.update(request_ids=normalized, prefix_lens=prefixes, query_lens=queries)
                    else:
                        expected = cohort.cached + state['batches']
                        if cohort.query != 1 or queries != [1]*4 or prefixes != [expected]*4:
                            raise RuntimeError('actual vLLM decode geometry changed')
                        state['decode_batches'].append(dict(request_ids=normalized,
                                                           seq_lens=[p+1 for p in prefixes]))
                    if result.preempted_req_ids or any(result.scheduled_spec_decode_tokens.values()):
                        raise RuntimeError('preemption or speculative tokens in controlled batch')
                    state['batches'] += 1
                    self._aligned_current = True
            elif state is not None:
                raise RuntimeError('admitted aligned vLLM cohort did not execute')
            return result

        def sampled(*args, **kwargs):
            result = sample(*args, **kwargs)
            if not self._aligned_current:
                return result
            state = self._aligned_state
            cohort = state['cohort']
            if cohort.query == 1 and state['batches'] == 1:
                torch.cuda.synchronize()
                state['prefill_wall_s'] = time.perf_counter()-state['started']
                state['prefill_finished'] = True
            elif state['batches'] == cohort.outputs:
                torch.cuda.synchronize()
                end = time.perf_counter()
                state.pop('cohort')
                start = state.get('decode_started', state['started'])
                tokens = 4*127 if cohort.query == 1 else 4*cohort.query
                record = dict(state, name=cohort.name, cached_tokens=cohort.cached,
                    query_tokens=cohort.query, engine_wall_s=end-start, ended=end,
                    measured_tokens=tokens, tokens_per_s=tokens/(end-start),
                    async_scheduling=self.async_scheduling, batch_queue_size=self.batch_queue_size)
                self._aligned_completed[cohort.name] = dict(lookahead=0)
                self._aligned_state = None
                self._aligned_current = False
                with (directory/f'{cohort.name}.json').open('x') as out:
                    json.dump(record, out, indent=2)
            return result

        self.scheduler.schedule = scheduled
        self.model_executor.sample_tokens = sampled

    def add(self, request, *args, **kwargs):
        cohort = identify(canonical(request.request_id))
        if cohort is None:
            if self._aligned_pending or self._aligned_state:
                raise RuntimeError('unrelated request during aligned cohort')
            return original_add(self, request, *args, **kwargs)
        if self._aligned_state is not None:
            raise RuntimeError('overlapping aligned cohorts')
        if not self._aligned_pending:
            self._aligned_admission_start = time.perf_counter()
        if request.max_tokens != cohort.outputs or request.num_prompt_tokens != cohort.cached+cohort.query:
            raise RuntimeError('aligned request geometry mismatch')
        self._aligned_pending.append((request,args,kwargs))
        ready = admission([canonical(r.request_id) for r,_,_ in self._aligned_pending])
        if ready is False:
            return
        self._aligned_state = dict(cohort=cohort, batches=0, decode_batches=[],
            admission_wait_s=time.perf_counter()-self._aligned_admission_start)
        for req,a,k in self._aligned_pending:
            original_add(self, req, *a, **k)
        self._aligned_pending = []

    cls.__init__, cls.add_request = init, add
    cls._aligned_hook = True
