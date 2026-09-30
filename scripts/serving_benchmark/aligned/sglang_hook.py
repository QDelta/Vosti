"""Opt-in scheduler admission/timing hook; never changes installed SGLang files.

This adapter measures prefill or a continuous 127-step decode interval. It
retains the normal scheduler, attention backend, overlap mode and CUDA graphs.
No tensor copies, file IO or GPU synchronizations occur per layer. Two GPU
synchronizations bound each isolated measured/warmup prefill interval.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import time

from scripts.serving_benchmark.aligned.protocol import admission, check_batch, identify


def install(module):
    import torch

    cls = module.Scheduler
    if getattr(cls, '_aligned_prefill_hook', False):
        return
    output = Path(os.environ['VOSTI_ALIGNED_PHASE_RECORDS'])
    output.mkdir(parents=True, exist_ok=True)
    (output/f'installed-{os.getpid()}.json').write_text(json.dumps(dict(
        scheduler_source=module.__file__,
        sha256=hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest())))
    original_next, original_run = cls.get_next_batch_to_run, cls.run_batch
    original_sample = cls.launch_batch_sample_if_needed

    def next_batch(self, running_batch, last_batch):
        pending = [r.rid for r in self.waiting_queue]
        cohort = admission(pending)
        if cohort is False:
            since = getattr(self, '_aligned_wait_since', None)
            if since is None:
                self._aligned_wait_since = time.perf_counter()
            elif time.perf_counter()-since > 30:
                raise RuntimeError('aligned admission timed out waiting for four requests')
            return module.NextBatchPlan(batch_to_run=None, running_batch=running_batch)
        if cohort is not None:
            if getattr(self, '_aligned_active', None) is not None:
                raise RuntimeError('overlapping controlled cohorts')
            previous = list(running_batch.reqs) + (list(last_batch.reqs) if last_batch else [])
            if any(not r.finished() for r in previous) or getattr(self, 'result_queue', []):
                # Let the standard event loop drain previous result processing.
                return module.NextBatchPlan(batch_to_run=None, running_batch=running_batch)
            torch.cuda.synchronize()
            started = time.perf_counter()
            self._aligned_active = dict(cohort=cohort, started=started,
                admission_wait_s=started-(getattr(self, '_aligned_wait_since', None) or started),
                batches=0)
            self._aligned_wait_since = started
        state = getattr(self, '_aligned_active', None)
        if state is not None and state.get('prefill_finished') and 'decode_started' not in state:
            # The first-token GPU work completed at the preceding boundary.
            # Start before scheduling the first decode; do not sync per token.
            state['decode_started'] = time.perf_counter()
        plan = original_next(self, running_batch=running_batch, last_batch=last_batch)
        if cohort is not None and plan.batch_to_run is None:
            raise RuntimeError('four admitted requests did not produce a batch')
        return plan

    def finish(self):
        state = self._aligned_active
        if state['batches'] != state['cohort'].outputs:
            raise RuntimeError('interval did not execute the expected number of batches')
        torch.cuda.synchronize()
        ended = time.perf_counter()
        cohort = state.pop('cohort')
        start = state.get('decode_started', state['started'])
        tokens = 4*127 if cohort.query == 1 else 4*cohort.query
        record = dict(state, ended=ended, engine_wall_s=ended-start,
            name=cohort.name, cached_tokens=cohort.cached, query_tokens=cohort.query,
            new_input_tokens=4*cohort.query, measured_tokens=tokens, tokens_per_s=tokens/(ended-start),
            scheduler_pid=os.getpid(), endpoint='GPU complete after measured sampling',
            timing='engine scheduling + forward + sampling; no HTTP/admission waiting; decode keeps overlap')
        self._aligned_active = None
        self._aligned_wait_since = None
        if not hasattr(self, '_aligned_prefilled'):
            self._aligned_prefilled = {}
        self._aligned_prefilled[cohort.name] = dict(cohort=cohort, lookahead_batches=0)
        with (output/f'{cohort.name}.json').open('x') as stream:
            json.dump(record, stream, indent=2)

    def run_batch(self, batch, *args, **kwargs):
        ids = [r.rid for r in batch.reqs]
        controlled = any(identify(rid) is not None for rid in ids)
        prefill = controlled and batch.forward_mode.is_extend()
        if controlled:
            state = getattr(self, '_aligned_active', None)
            if state is None:
                # The overlap scheduler can launch one unused decode before
                # observing max_new_tokens=1 in result processing. Preserve it:
                # this pilot measures pure prefill, not request-total GPU work.
                cohort = admission(ids)
                done = getattr(self, '_aligned_prefilled', {}).get(getattr(cohort, 'name', None))
                if (done is None or done['cohort'] != cohort or not self.enable_overlap
                        or not batch.forward_mode.is_decode() or done['lookahead_batches']
                        or any(r.sampling_params.max_new_tokens != cohort.outputs for r in batch.reqs)):
                    raise RuntimeError('unexpected aligned execution outside prefill interval')
                done['lookahead_batches'] += 1
                with (output/f'lookahead-{cohort.name}.json').open('x') as stream:
                    json.dump(dict(request_ids=ids, batch_size=len(ids), query_tokens=1,
                        forward_mode='decode', included_in_prefill_timing=False,
                        reason='normal overlap lookahead before first-token completion processing'), stream)
            else:
                if state['batches'] == 0:
                    check_batch(state['cohort'], ids, batch.prefix_lens, batch.extend_lens,
                                is_extend=prefill)
                    state.update(request_ids=ids, prefix_lens=list(batch.prefix_lens),
                                 query_lens=list(batch.extend_lens), overlap=bool(self.enable_overlap),
                                 decode_batches=[])
                else:
                    if (state['cohort'].query != 1 or not batch.forward_mode.is_decode()
                            or len(ids) != 4 or set(ids) != set(state['cohort'].ids())):
                        raise RuntimeError('decode cohort split or changed execution phase')
                    lengths = batch.seq_lens_cpu.tolist()
                    expected = state['cohort'].cached + state['batches'] + 1
                    if lengths != [expected]*4:
                        raise RuntimeError(f'decode context mismatch: {lengths} != {expected}')
                    state['decode_batches'].append(dict(request_ids=ids, seq_lens=lengths))
                state['batches'] += 1
        result = original_run(self, batch, *args, **kwargs)
        if controlled and not self.enable_overlap:
            sampled(self)
        return result

    def sampled(self):
        state = getattr(self, '_aligned_active', None)
        if state is None:
            return  # unused normal overlap lookahead after completion
        if state['cohort'].query == 1 and state['batches'] == 1:
            torch.cuda.synchronize()
            state['prefill_wall_s'] = time.perf_counter()-state['started']
            state['prefill_finished'] = True
        elif state['batches'] == state['cohort'].outputs:
            finish(self)

    def sample(self, result, batch):
        returned = original_sample(self, result, batch)
        if batch is not None and any(identify(r.rid) is not None for r in batch.reqs):
            sampled(self)
        return returned

    cls.get_next_batch_to_run = next_batch
    cls.run_batch = run_batch
    cls.launch_batch_sample_if_needed = sample
    cls._aligned_prefill_hook = True
