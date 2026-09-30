"""Multi-call, one-engine native adapter for expanded correctness workloads."""
import hashlib
import json
import os
from pathlib import Path
import sysconfig

from scripts.determinism_tests.call_inputs import resolve_call
from scripts.determinism_tests.native_observation import schedule_events
from scripts.determinism_tests.protocol import load_observer_records, sha256_json
from scripts.determinism_tests.step_trace import index_emissions, validate_arrivals


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def assemble_calls(calls, completed, records, steps, layouts, *, caching, copy_row):
    """Validate identities and all emitted rows before returning selected logits."""
    if len(calls) != len(completed):
        raise ValueError('native multi-call results have wrong call count')
    indexed = index_emissions(records, steps)
    schedules = schedule_events(layouts, steps)
    results = []
    next_id = 0
    for index, (declared, actual) in enumerate(zip(calls, completed, strict=True)):
        call = resolve_call(declared, results)
        prompts, outputs = actual['input_token_ids'], actual['output_token_ids']
        base = actual['request_id_base']
        if (actual['label'] != call['label'] or prompts != call['prompts'] or base != next_id
                or len(outputs) != len(prompts) or actual['cache_isolated'] is not (not caching)):
            raise ValueError('native call identity or cache isolation differs from input')
        next_id += len(prompts)
        validate_arrivals(steps, request_id_base=base, arrival_steps=[0]*len(prompts))
        phase = f'call-{index}'
        if any(row['phase'] != phase for row in layouts
               if row['engine_step'].startswith(f'{base}:')):
            raise ValueError('native layout phase differs from call')
        page = actual['cache_page_tokens']
        if type(page) is not int or page <= 0:
            raise ValueError('invalid compiled cache page geometry')
        call_steps = [step for step in steps if step['request_id_base'] == base]
        hits = {}
        for step in call_steps:
            for rid, blocks in zip(step['scheduled'], step['cached_prefix_blocks'], strict=True):
                if rid not in hits:
                    if type(blocks) is not int or blocks < 0:
                        raise ValueError('native initial cache observation absent')
                    hits[rid] = blocks*page
        requests = []
        for offset, (prompt, output) in enumerate(zip(prompts, outputs, strict=True)):
            rid = base+offset
            rows = indexed.pop(rid, [])
            if (len(output) != call['max_tokens'] or len(rows) != len(output)
                    or any(row['argmax'] != token or not row['finite'] or row.get('phase') != phase
                           for row, token in zip(rows, output, strict=True))):
                raise ValueError('native output rows do not match complete emitted sequence')
            cached = hits.get(rid)
            if cached is None or cached > len(prompt) or (not caching and cached != 0):
                raise ValueError('native initial cache state disagrees with declaration')
            position = call.get('record_generated_position')
            if position is None:
                position = len(output)-1
            if type(position) is not int or not 0 <= position < len(output):
                raise ValueError('native retained position out of bounds')
            selected = {}
            positions = (range(len(rows)) if call.get('record_all_rows') else
                         [position] if call.get('record_last_rows') else [])
            for i in positions:
                row = copy_row(rows[i], index, offset, i)
                row['metadata'].update(request_id=str(rid), generated_position=i)
                selected[i] = row
            request = dict(request_id=str(rid), input_token_ids=prompt, prompt_length=len(prompt),
                prompt_sha256=sha256_json(prompt), output_token_ids=output,
                num_cached_tokens=cached, last_row=selected.get(position))
            if call.get('record_all_rows'):
                request['output_rows'] = [selected[i] for i in range(len(rows))]
            requests.append(request)
        stats = actual['graph_stats']
        if isinstance(stats, str):
            stats = json.loads(stats)
        if (not isinstance(stats, dict) or stats.get('poisoned_reason') is not None
                or 'cover_replay_count' not in stats):
            raise ValueError('native graph overlay absent or poisoned')
        results.append(dict(label=call['label'], requests=requests,
            schedule_events=[schedules[step['engine_step']] for step in call_steps],
            graph_stats=stats, cache_isolated=actual['cache_isolated']))
    if indexed or any(step['request_id_base'] not in {value['request_id_base'] for value in completed}
                      for step in steps):
        raise ValueError('native observations contain unknown requests or calls')
    return results


def run_calls(*, arm, binary, deployment_bundle, invocation_dir, artifact_dir):
    from scripts.determinism_tests.vosti_worker import ROOT, _copy_row, _run_binary_with_identifiable_teardown
    invocation_dir.mkdir(parents=True, exist_ok=False)
    calls_path = invocation_dir/'calls.json'
    results_path = invocation_dir/'call-results.json'
    observer = invocation_dir/'observer'
    trace = invocation_dir/'step-trace.jsonl'
    layout = invocation_dir/'layout.jsonl'
    calls_path.write_text(json.dumps(dict(prefix_caching=arm['engine']['prefix_caching'], calls=arm['calls'])))
    env = os.environ.copy()
    # Native embeds this worker's Python; retain the suite hooks and site-packages.
    env['LD_LIBRARY_PATH'] = os.pathsep.join(filter(None, [sysconfig.get_config_var('LIBDIR'), env.get('LD_LIBRARY_PATH')]))
    paths = env.get('PYTHONPATH', '').split(os.pathsep)
    purelib = str(Path(sysconfig.get_paths()['purelib']).resolve())
    env['PYTHONPATH'] = os.pathsep.join(dict.fromkeys([*filter(None, paths), purelib]))
    for name in ('VOSTI_BENCH_INPUT', 'VOSTI_BENCH_WARMUP_INPUT', 'VOSTI_BENCH_GRAPH_PRIMER_INPUT',
                 'VOSTI_OUTPUT_TOKENS', 'VOSTI_LOGITS_OBSERVER_STEP'):
        env.pop(name, None)
    env.update(MODEL_PATH=str(Path(arm['model_path']).resolve()), CUDA_DEVICE='cuda:0',
        VOSTI_FRAMEWORK_ROOT=str(ROOT), VOSTI_DEPLOYMENT_BUNDLE=str(deployment_bundle),
        VOSTI_BENCH='1', VOSTI_BENCH_CALLS=str(calls_path), VOSTI_BENCH_CALL_RESULTS=str(results_path),
        VOSTI_MAX_BATCHED_TOKENS=str(arm['engine']['max_num_batched_tokens']),
        VOSTI_MAX_SEQS=str(arm['engine']['max_num_seqs']), VOSTI_NUM_BLOCKS=str(arm['engine']['num_blocks']),
        VOSTI_CUDA_GRAPH='1', VOSTI_GRAPH_WARMUP_ROUNDS='0', VOSTI_QUIET_OUTPUT='1',
        VOSTI_KERNELS_LOGITS_OBSERVER='1', VOSTI_LOGITS_OBSERVER_DIR=str(observer),
        VOSTI_LOGITS_OBSERVER_PHASE='initialization', VOSTI_ENGINE_STEP_TRACE=str(trace),
        VOSTI_NATIVE_LAYOUT_OBSERVER='1', VOSTI_NATIVE_LAYOUT_PATH=str(layout))
    stdout_path, stderr_path = invocation_dir/'stdout.log', invocation_dir/'stderr.log'
    with stdout_path.open('xb') as stdout, stderr_path.open('xb') as stderr:
        code = _run_binary_with_identifiable_teardown(binary, env=env, stdout=stdout, stderr=stderr)
    if code != 0:
        raise RuntimeError(f'native multi-call worker exited {code}; see {stderr_path}')
    if 'backend_qualified=true' not in stdout_path.read_text():
        raise RuntimeError('native multi-call worker did not report qualified runtime')

    def copy(row, call, request, position):
        return _copy_row(row, observer, artifact_dir,
            label=f'call-{call:03d}-request-{request:02d}-position-{position:03d}',
            call_index=call, request_index=request)

    completed = json.loads(results_path.read_text())
    calls = assemble_calls(arm['calls'], completed, load_observer_records(observer), read_jsonl(trace),
        read_jsonl(layout), caching=arm['engine']['prefix_caching'], copy_row=copy)
    evidence = dict(binary=str(binary), deployment_bundle=str(deployment_bundle), multi_call=True,
        stdout=str(stdout_path), stderr=str(stderr_path),
        call_graph_stats=[call['graph_stats'] for call in calls],
        observations={name: dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
                      for name, path in [('steps',trace), ('layout',layout), ('calls',results_path)]})
    return calls, evidence
