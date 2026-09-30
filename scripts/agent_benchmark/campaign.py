"""Replay frozen tasks on explicitly selected, qualified serving modes."""
from __future__ import annotations

import argparse
from copy import deepcopy
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import time

from scripts.common.telemetry import load_complete_telemetry
from scripts.serving_benchmark.campaign import inspect_trial, source_identity
from scripts.common.artifacts import package_inventory, shared_helper_sources
from scripts.serving_benchmark.server_trial import gpu_pids
from .prepare import save, sha


def configure_launch(original, plan, output, chat_template=None):
    """Adjust capacity and cache paths without changing the qualified backend."""
    spec = deepcopy(original)
    concurrency, context = plan['concurrency'], plan['context_limit']
    if concurrency <= 0 or context <= 0:
        raise ValueError('positive concurrency and context required')
    engine = spec['execution']['engine']
    def replace(flag, value):
        if spec['command'].count(flag) != 1:
            raise ValueError(f'expected exactly one {flag}')
        spec['command'][spec['command'].index(flag)+1] = str(value)
    if engine == 'vosti':
        if concurrency > spec['settings']['max_sequences']:
            raise ValueError('native concurrency exceeds qualified capacity')
        spec['environment'].update(VOSTI_MAX_SEQS=str(concurrency), VOSTI_MAX_MODEL_LEN=str(context))
    elif engine == 'vllm':
        replace('--max-num-seqs', concurrency)
        replace('--max-model-len', context)
    elif engine == 'sglang':
        replace('--max-running-requests', concurrency)
        replace('--context-length', context)
        # The built-in warmup sends images for VLM checkpoints, whereas this
        # harness deliberately accepts text only. run.py still performs and
        # validates a real text-chat warmup before any measured agent task.
        if '--skip-server-warmup' not in spec['command']:
            spec['command'].append('--skip-server-warmup')
        # Retain qualified graph coverage, which also covers smaller batches.
    else:
        raise ValueError(f'unsupported serving engine: {engine}')
    if chat_template is not None and engine != 'vosti':
        if '--chat-template' in spec['command']:
            raise ValueError('qualified template override requires explicit reconciliation')
        spec['command'].extend(['--chat-template', str(chat_template)])
    spec['settings'].update(max_sequences=concurrency, context_limit=context)
    for key, sub in [('TRITON_CACHE_DIR','triton'),('TORCHINDUCTOR_CACHE_DIR','inductor'),('CUDA_CACHE_PATH','cuda')]:
        spec['environment'][key] = str(output/'compiler-cache'/spec['execution']['key']/sub)
    return spec


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--previous-pilot', type=Path, required=True)
    p.add_argument('--serving-campaign', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--mode', default='vosti-padded-graph', help='Exact mode ID in the qualified serving campaign')
    p.add_argument('--concurrency', type=int, help='Default: preserve prior pilot concurrency')
    a = p.parse_args()
    mode = a.mode
    root = Path(__file__).resolve().parents[2]
    output = a.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    previous = json.loads((a.previous_pilot/'plan.json').read_text())
    serving = json.loads((a.serving_campaign/'campaign.json').read_text())
    state = json.loads((a.serving_campaign/'status.json').read_text())[mode]
    frozen = Path(serving['root'])
    assert source_identity(frozen) == serving['source']
    assert inspect_trial(Path(state['output']), Path(state['telemetry'])) == state
    for path, expected in {**serving['input_files'], **serving['binaries'], **previous['files']}.items():
        assert sha(path) == expected, path
    old_trial = next(t for t in serving['trials'] if t['id'] == mode)
    spec = json.loads(Path(old_trial['spec']).read_text())
    assert spec['execution']['key'] == mode
    engine = spec['execution']['engine']
    worker = Path(serving['workers'][engine])
    assert package_inventory(worker) == serving['packages'][engine]
    assert importlib.metadata.version('mini-swe-agent') == '1.17.5'
    assert importlib.metadata.version('swebench') == '5.0.2'
    checkpoint = Path(spec['checkpoint']['path'])
    generation = json.loads((checkpoint/'generation_config.json').read_text())
    from transformers import AutoTokenizer
    from .runtime import generation_eos_strings
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    plan = dict(previous, model=spec['checkpoint']['key'], modes=[mode],
        concurrency=a.concurrency if a.concurrency is not None else previous['concurrency'],
        eos_token_ids=generation['eos_token_id'],
        generation_config_sha256=sha(checkpoint/'generation_config.json'),
        subset_reused_from=str(a.previous_pilot.resolve()),
        comparison_note='Same frozen issues, agent prompts and per-task budgets; live trajectories may differ. '
            'Compare concurrency explicitly; not an equal-work speedup measurement.')
    plan['eos_strings'] = generation_eos_strings(tokenizer, plan)
    # Validate the actual default template instead of adding unsupported API kwargs.
    messages=[dict(role='user', content='Reply with the word ready.')]
    default = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
    explicit = tokenizer.apply_chat_template(messages, tokenize=True,
        add_generation_prompt=True, enable_thinking=False)
    assert default == explicit, 'checkpoint default thinking policy needs explicit integration'
    plan['thinking_policy'] = 'checkpoint default, verified equal to enable_thinking=False'
    plan['chat_content_policy'] = 'string text; normalize baseline content arrays before the unmodified checkpoint template'
    template_path = output/'chat-template.jinja' if engine != 'vosti' else None
    spec = configure_launch(spec, plan, output, template_path)
    files = {}
    for name in ['instances.json', 'grading-instances.json', 'agent-config.json', 'images.json']:
        data = json.loads((a.previous_pilot/name).read_text())
        save(output/name, data)
        assert sha(output/name) == sha(a.previous_pilot/name), name
        files[str(output/name)] = sha(output/name)
    plan['files'] = files
    if template_path is not None:
        prefix = (Path(__file__).parent/'text_content.jinja').read_text()
        if not isinstance(tokenizer.chat_template, str):
            raise ValueError('select an explicit checkpoint template before replay')
        with template_path.open('x') as stream:
            stream.write(prefix + tokenizer.chat_template)
        files[str(template_path)] = sha(template_path)
    save(output/'plan.json', plan)
    images = json.loads((output/'images.json').read_text())
    assert set(images) == {r['instance_id'] for r in json.loads((output/'instances.json').read_text())}
    for item in images.values():
        subprocess.run(['docker','image','inspect',item['id']], check=True,
            stdout=subprocess.DEVNULL, timeout=30)
    spec_path = output/'launches'/f'{mode}.json'
    save(spec_path, spec)
    if engine == 'vosti':
        bundle = Path(spec['environment']['VOSTI_DEPLOYMENT_BUNDLE'])/'deployment.json'
        deployment_receipt = json.loads((a.serving_campaign/'deployment-receipts'/
            f'{spec["checkpoint"]["key"]}.json').read_text())
        assert sha(bundle) == deployment_receipt['deployment_sha256']
        assert sha(deployment_receipt['telemetry']) == deployment_receipt['telemetry_sha256']
        load_complete_telemetry(Path(deployment_receipt['telemetry']))
        files[str(bundle)] = sha(bundle)
    scripts = {str(f.resolve()):sha(f) for f in [*Path(__file__).parent.iterdir(), *shared_helper_sources()]
               if f.suffix in ('.py','.jinja')}
    with tarfile.open(output/'harness-run.tar.gz', 'x:gz') as archive:
        for path in sorted(scripts):
            archive.add(path, arcname=str(Path(path).relative_to(root)))
    locked = {**files, **scripts, **serving['input_files'], **serving['binaries'],
        str(output/'plan.json'):sha(output/'plan.json'), str(spec_path):sha(spec_path),
        spec['command'][0]:sha(spec['command'][0])}
    for path in [output/'harness-run.tar.gz', a.previous_pilot/'plan.json',
                 a.serving_campaign/'campaign.json', a.serving_campaign/'status.json',
                 Path(old_trial['spec'])]:
        locked[str(path.resolve())] = sha(path)
    for name in ['config.json','generation_config.json','tokenizer.json','tokenizer_config.json','chat_template.jinja']:
        path = checkpoint/name
        if path.is_file():
            locked[str(path)] = sha(path)
    save(output/'execution-lock.json', dict(source=serving['source'], files=locked,
        prior_serving_campaign=str(a.serving_campaign.resolve()),
        prior_serving_campaign_sha256=sha(a.serving_campaign/'campaign.json'),
        packages={d.metadata['Name']:d.version for d in importlib.metadata.distributions()}))
    def check_inputs():
        assert source_identity(frozen) == serving['source']
        for path, expected in locked.items():assert sha(path) == expected, path
    gpu = spec['settings']['gpu_index']
    while gpu_pids(gpu):
        print('WAIT: GPU occupied; no job launched or signalled', flush=True)
        time.sleep(30)
    check_inputs()
    telemetry = output/'telemetry'/f'{mode}.json'
    telemetry.parent.mkdir(exist_ok=True)
    env = dict(os.environ, MSWEA_SILENT_STARTUP='1',
        MSWEA_GLOBAL_CONFIG_DIR=str(output/'agent-global-config'), HF_HUB_OFFLINE='1',
        TOKENIZERS_PARALLELISM='false')
    command = [sys.executable,str(root/'scripts/common/gpu_monitor.py'),
        '--gpu-index',str(gpu),'--output',str(telemetry),'--',sys.executable,
        '-m','scripts.agent_benchmark.run','--pilot',str(output),'--spec',str(spec_path),
        '--output',str(output/'trials'/mode)]
    save(output/'commands'/f'{mode}.json', dict(command=command))
    print('MODE START', plan['model'], mode, 'tasks',len(images), flush=True)
    subprocess.run(command, check=True, cwd=root, env=env)
    receipt = load_complete_telemetry(telemetry)
    check_inputs()
    save(output/'receipts'/f'{mode}.json', receipt)
    summary = json.loads((output/'trials'/mode/'summary.json').read_text())
    assert summary['complete'] and summary['attempted'] == len(images)
    if summary['infrastructure_errors']:
        raise RuntimeError('infrastructure errors: preserve attempts for diagnosis; do not grade as model failures')
    save(output/'inference-complete.json', dict(complete=True,modes=[mode],
        tasks=len(images), finished_unix_s=time.time(), grading='pending official evaluation'))
    print('INFERENCE COMPLETE; START OFFICIAL GRADING', flush=True)
    subprocess.run([sys.executable,'-m','scripts.agent_benchmark.evaluate',
        '--pilot',str(output)], check=True, cwd=root, env=env)


if __name__ == '__main__':
    main()
