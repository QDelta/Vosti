"""Freeze a seeded SWE-bench Lite pilot; never expose grading fields to agents."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess

DATASET = 'SWE-bench/SWE-bench_Lite'
PUBLIC_FIELDS = ('instance_id', 'repo', 'base_commit', 'problem_statement')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(data, stream, indent=2, sort_keys=True)
        stream.write('\n')


def select(rows, seed, count, difficulty=None):
    if difficulty is not None:
        rows = [r for r in rows if r.get('difficulty') == difficulty]
    if not 0 < count <= len(rows):
        raise ValueError('invalid subset size')
    if len({r['instance_id'] for r in rows}) != len(rows):
        raise ValueError('duplicate instance IDs')
    return sorted(rows, key=lambda r: hashlib.sha256(
        f'{seed}:{r["instance_id"]}'.encode()).hexdigest())[:count]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--agent-source', type=Path, required=True)
    parser.add_argument('--model', required=True, help='Checkpoint key in the qualified serving launch')
    parser.add_argument('--mode', action='append', required=True, help='Qualified mode; repeat as needed')
    parser.add_argument('--concurrency', type=int, default=1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--count', type=int, default=10)
    parser.add_argument('--difficulty')
    parser.add_argument('--revision')
    parser.add_argument('--dataset-arrow', type=Path, help='Reuse a cached test split; requires its pinned revision')
    args = parser.parse_args()
    if args.concurrency <= 0 or len(set(args.mode)) != len(args.mode):
        parser.error('positive concurrency and unique modes required')
    root = args.output.resolve()
    if args.dataset_arrow and not args.revision:
        parser.error('--dataset-arrow requires --revision')
    try:
        from datasets import Dataset, load_dataset
        from huggingface_hub import HfApi
        import yaml
    except ImportError as error:
        parser.error(f'{error}; install the optional agent environment described in '
                     'scripts/agent_benchmark/README.md#environment')
    revision = args.revision or HfApi().dataset_info(DATASET).sha
    rows = (list(Dataset.from_file(str(args.dataset_arrow))) if args.dataset_arrow else
            list(load_dataset(DATASET, revision=revision, split='test', cache_dir=str(root/'hf-cache'))))
    chosen = select(rows, args.seed, args.count, args.difficulty)
    config_path = args.agent_source/'src/minisweagent/config/extra/swebench.yaml'
    agent = yaml.safe_load(config_path.read_text())['agent']
    agent.update(step_limit=40, cost_limit=0)
    instances = []
    for row in chosen:
        public = {k: row[k] for k in PUBLIC_FIELDS}
        public['image_tag'] = ('docker.io/swebench/sweb.eval.x86_64.' +
            row['instance_id'].replace('__', '_1776_') + ':latest').lower()
        instances.append(public)
    save(root/'instances.json', instances)
    # Host-side evaluator data only. Never mounted or included in prompts.
    save(root/'grading-instances.json', chosen)
    save(root/'agent-config.json', agent)
    save(root/'packages.json', {d.metadata['Name']: d.version for d in importlib.metadata.distributions()})
    plan = dict(dataset=DATASET, revision=revision, split='test', seed=args.seed,
        difficulty=args.difficulty, eligible_count=sum(args.difficulty is None or r.get('difficulty') == args.difficulty for r in rows),
        dataset_arrow_sha256=sha(args.dataset_arrow) if args.dataset_arrow else None,
        selection='ascending sha256(seed:instance_id), independent of model results',
        model=args.model, modes=args.mode, concurrency=args.concurrency, context_limit=40960,
        max_tokens_per_call=2048, max_generated_tokens_per_task=16384,
        task_timeout_s=1800, tool_timeout_s=60, container_cpus=2, container_memory='8g',
        agent_commit=subprocess.check_output(['git','rev-parse','HEAD'], cwd=args.agent_source, text=True).strip(),
        upstream_agent_config_sha256=sha(config_path),
        files={str(root/name): sha(root/name) for name in ['instances.json','grading-instances.json','agent-config.json','packages.json']},
        sampling=dict(temperature=0, top_p=1, ignore_eos=False),
        context_policy='stop task before exceeding context; no silent truncation',
        timing='exclude environment preparation and grading; include tools in task wall time',
        attempt_policy='one attempt per issue per mode; no best-of or silent retries')
    save(root/'plan.json', plan)
    print(json.dumps(dict(plan=str(root/'plan.json'), instances=[r['instance_id'] for r in instances]), indent=2))


if __name__ == '__main__':
    main()
