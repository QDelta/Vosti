"""One live SWE-bench mode, invoked under the existing GPU telemetry wrapper."""
from __future__ import annotations

import argparse
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
import importlib.metadata
import json
import os
from pathlib import Path
import statistics
import subprocess
import time
import traceback

import httpx
from minisweagent.agents.default import DefaultAgent
from transformers import AutoTokenizer

from scripts.serving_benchmark.server_config import clean_environment
from scripts.serving_benchmark.server_trial import endpoint_evidence, gpu_pids, stop_server, wait_ready
from .prepare import save, sha
from .runtime import TaskEnvironment, TimedModel


def distribution(values):
    values = sorted(v for v in values if v is not None)
    if not values:
        return None
    def percentile(p):
        x = (len(values)-1)*p
        lo = int(x)
        return values[lo] + (values[min(lo+1,len(values)-1)]-values[lo])*(x-lo)
    return dict(n=len(values), mean=statistics.mean(values), p50=percentile(.5),
                p95=percentile(.95), max=max(values))


def prompt_token_count(tokenizer, messages):
    tokens = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
    # Transformers 5 may return BatchEncoding (a Mapping, not a dict).
    return len(tokens['input_ids'] if isinstance(tokens, Mapping) else tokens)


def task(row, environment, tokenizer, plan, spec, output, campaign_start, agent_config):
    start = time.monotonic()
    model = TimedModel(spec['base_url'], spec['served_name'], tokenizer, plan, output)
    model.deadline = start + plan['task_timeout_s']
    agent = DefaultAgent(model, environment, **agent_config)
    status, info, error = 'NotStarted', '', None
    print('TASK START', row['instance_id'], flush=True)
    try:
        status, info = agent.run(row['problem_statement'])
    except Exception:
        status, error = 'InfrastructureOrAPIError', traceback.format_exc()
    end = time.monotonic()
    patch = environment.patch()
    result = dict(instance_id=row['instance_id'], exit_status=status, error=error,
        start_offset_s=start-campaign_start, finish_offset_s=end-campaign_start,
        task_wall_s=end-start, calls=model.n_calls, generated_tokens=model.generated,
        model_request_s=sum(x['request_s'] for x in model.records),
        tool_s=sum(x['elapsed_s'] for x in environment.records),
        requests=model.records, submitted_text=info)
    save(output/'trajectory.json', dict(messages=agent.messages))
    save(output/'result.json', result)
    save(output/'prediction.json', dict(instance_id=row['instance_id'], model_name_or_path=spec['execution']['key'], model_patch=patch))
    print('TASK DONE', row['instance_id'], status, 'calls', model.n_calls,
          'tokens', model.generated, 'seconds', round(end-start, 1), flush=True)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pilot', type=Path, required=True)
    p.add_argument('--spec', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    plan = json.loads((a.pilot/'plan.json').read_text())
    for path, expected in plan['files'].items():
        if sha(path) != expected:
            raise RuntimeError(f'changed plan input: {path}')
    rows = json.loads((a.pilot/'instances.json').read_text())
    images = json.loads((a.pilot/'images.json').read_text())
    config = json.loads((a.pilot/'agent-config.json').read_text())
    spec = json.loads(a.spec.read_text())
    gpu = spec['settings']['gpu_index']
    assert spec['settings']['max_sequences'] == plan['concurrency']
    assert spec['settings']['context_limit'] == plan['context_limit']
    if gpu_pids(gpu):
        raise RuntimeError(f'GPU{gpu} not idle')
    save(a.output/'inputs.json', dict(plan=plan, spec=spec, images=images,
        runtime_files={str(f.resolve()):sha(f) for f in Path(__file__).parent.glob('*.py')},
        packages={d.metadata['Name']:d.version for d in importlib.metadata.distributions()}))
    tokenizer = AutoTokenizer.from_pretrained(spec['checkpoint']['path'], local_files_only=True)
    environments, server = [], None
    try:
        for row in rows:
            out = a.output/row['instance_id']
            out.mkdir()
            env = TaskEnvironment(images[row['instance_id']]['id'], plan, row['base_commit'], out)
            environments.append(env)
            print('CONTAINER READY', row['instance_id'], flush=True)
        # Server uses its frozen runtime, not this CPU-side agent environment.
        root_python = Path(spec['environment']['VOSTI_FRAMEWORK_ROOT'])/'.venv/bin/python' if spec['execution']['engine']=='vosti' else None
        lib = json.loads(subprocess.check_output([str(root_python),'-c',
            'import json,sysconfig,site; print(json.dumps([sysconfig.get_config_var("LIBDIR"),site.getsitepackages()]))'], text=True)) if root_python else ['', []]
        env = clean_environment(dict(os.environ), spec, python_libdir=lib[0], python_site_packages=lib[1])
        with (a.output/'server.log').open('x') as log:
            server = subprocess.Popen(spec['command'], env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            wait_ready(server, spec['base_url'], 1200)
            warm = dict(model=spec['served_name'], messages=[dict(role='system', content='Be brief.'),
                dict(role='user', content='Reply with the word ready.')],
                temperature=0, top_p=1, max_tokens=16, stream=False)
            with httpx.Client(timeout=300, trust_env=False) as client:
                response = client.post(spec['base_url']+'/v1/chat/completions', json=warm)
                response.raise_for_status()
                save(a.output/'warmup.json', dict(request=warm, response=response.json()))
                if response.json()['usage']['prompt_tokens'] != prompt_token_count(tokenizer, warm['messages']):
                    raise RuntimeError('warmup server/client chat tokenization differs')
            save(a.output/'server-before.json', endpoint_evidence(spec['base_url']))
            print('MEASURED TASKS START', spec['execution']['key'], flush=True)
            start = time.monotonic()
            with ThreadPoolExecutor(max_workers=plan['concurrency']) as pool:
                futures = [pool.submit(task, row, environment, tokenizer, plan, spec,
                    a.output/row['instance_id'], start, config) for row, environment in zip(rows,environments)]
                results = [f.result() for f in futures]
            elapsed = time.monotonic()-start
            save(a.output/'server-after.json', endpoint_evidence(spec['base_url']))
            requests = [q for r in results for q in r['requests']]
            save(a.output/'summary.json', dict(complete=True, mode=spec['execution']['key'],
                attempted=len(results), infrastructure_errors=sum(r['error'] is not None for r in results),
                measured_campaign_s=elapsed, task_wall_s=distribution([r['task_wall_s'] for r in results]),
                ttft_s=distribution([q['ttft_s'] for q in requests]),
                tpot_estimate_s=distribution([q['tpot_estimate_s'] for q in requests]),
                request_latency_s=distribution([q['request_s'] for q in requests]),
                total_generated_tokens=sum(r['generated_tokens'] for r in results),
                output_tokens_per_campaign_s=sum(r['generated_tokens'] for r in results)/elapsed,
                input_tokens_per_campaign_s=sum(q['usage']['prompt_tokens'] for q in requests)/elapsed,
                tasks_per_hour=len(results)*3600/elapsed, task_results=results,
                quality_score='pending official evaluator; Submitted is not resolved'))
            save(a.output/'predictions.json', [json.loads((a.output/r['instance_id']/'prediction.json').read_text()) for r in rows])
            print('MODE COMPLETE', spec['execution']['key'], round(elapsed, 1), flush=True)
    finally:
        try:
            if server is not None:
                stop_server(server, gpu)
        finally:
            for environment in environments:
                environment.cleanup()


if __name__ == '__main__':
    main()
