"""Small architecture-neutral qualification of actual padded-cover replay."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from scripts.determinism_tests.native_report import mapping_calls, check_cover_subset
from scripts.determinism_tests.protocol import (
    CHECKPOINTS, EXECUTION_CONFIGS, MODELS, PROMPT_SEED, build_inputs,
    row_comparison_semantics,
)
from scripts.determinism_tests.run import SuiteRunner, _arm
from scripts.serving_benchmark.campaign import source_identity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', choices=[c.key for c in CHECKPOINTS], required=True)
    parser.add_argument('--worker-root', type=Path, required=True)
    parser.add_argument('--worker-python', type=Path, required=True)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--deployment-bundle', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpu-index', type=int, default=0)
    parser.add_argument('--seed', type=lambda x: int(x, 0), default=PROMPT_SEED)
    args = parser.parse_args()
    if args.gpu_index < 0:
        parser.error("--gpu-index must be nonnegative")
    checkpoint = next(c for c in CHECKPOINTS if c.key == args.checkpoint)
    model = next(m for m in MODELS if m.key == checkpoint.model)
    inputs = build_inputs(model, Path(checkpoint.path), seed=args.seed)
    prompts = [inputs['batch_prompts'][i] for i in (3, 4, 5)]
    calls = mapping_calls(prompts, inputs['chunk_prompt'])
    fields = dict(vosti_binary=str(args.binary.resolve()),
        deployment_bundle=str(args.deployment_bundle.resolve()),
        multi_call=True, trace_steps=True, num_blocks=64)
    manifest = dict(inputs=inputs, calls=calls, engine=fields,
        worker_source=source_identity(args.worker_root.resolve()),
        controller_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        binary_sha256=hashlib.sha256(args.binary.read_bytes()).hexdigest(),
        deployment_sha256=hashlib.sha256(
            (args.deployment_bundle / 'deployment.json').read_bytes()).hexdigest(),
        gpu_index=args.gpu_index, worker_python=str(args.worker_python.absolute()))
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'inputs.json').write_text(json.dumps(manifest, indent=2) + '\n')
    execution = next(m for m in EXECUTION_CONFIGS if m.key == 'vosti-padded-graph')
    runner = SuiteRunner(output=args.output, worker_root=args.worker_root,
        worker_python=args.worker_python.absolute(), gpu_index=args.gpu_index,
        execution=execution, arm_engine_fields=fields)
    arm = _arm(inputs=inputs, execution=execution, calls=calls, prefix_caching=False)
    directory, result = runner.run_arm('cover-smoke', arm)
    # Requires at least two measured cover replays and compares all three
    # output-token logits for each of the two covered requests (six pairs).
    comparisons = check_cover_subset(directory, result)
    passed = all(row['bitwise_equal'] and row['finite_left'] and row['finite_right']
                 for row in comparisons)
    summary = dict(status='pass' if passed else 'mismatch', comparisons=comparisons,
        comparison_semantics=row_comparison_semantics(), arms=runner.arm_records,
        graph_stats=[call['graph_stats'] for call in result['calls']])
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(dict(status=summary['status'], pairs=len(comparisons))), flush=True)
    if not passed:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
