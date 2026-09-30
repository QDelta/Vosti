"""Native report preparation, qualified-checkpoint runs, and adapter smoke checks.

All subcommands retain raw evidence and require qualified native execution.
A smoke check never substitutes for the full four-relation report.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import threading
import time

from scripts.common.artifacts import package_inventory, save
from scripts.common.telemetry import load_complete_telemetry
from scripts.determinism_tests.protocol import EXECUTION_CONFIGS, NATIVE_CHECKPOINTS, compare_row_artifacts
from scripts.determinism_tests.report_plan import make_plan, REPORT_MODELS
from scripts.determinism_tests.report_suite import prefill_witness, run_report
from scripts.determinism_tests.run import SuiteRunner, _arm, _row_path
from scripts.serving_benchmark.campaign import file_hash, source_identity
from scripts.serving_benchmark.server_trial import gpu_pids

def call(label, prompts, count=3):
    return dict(label=label, prompts=prompts, max_tokens=count, record_last_rows=True,
                record_generated_position=0, record_all_rows=True)


def mapping_calls(prompts, long):
    # Exact graphs take precedence over cover. Batch two must be new after
    # capturing batch three; repeating singleton/three cannot exercise cover.
    return [call('single', prompts[:1]), call('mixed', prompts),
            call('reverse', list(reversed(prompts))), call('single-again', prompts[:1]),
            call('long', [long]), call('cover-subset', prompts[:2])]


def check_cover_subset(directory, result):
    before, subset = result['calls'][-2:]
    mixed = result['calls'][1]
    if (subset['label'] != 'cover-subset' or len(subset['requests']) != 2
            or len(mixed['requests']) != 3
            or subset['graph_stats']['cover_replay_count'] - before['graph_stats']['cover_replay_count'] < 2):
        raise RuntimeError('native subset smoke did not exercise both padded-cover decode steps')
    comparisons = []
    for reference, request in zip(mixed['requests'][:2], subset['requests'], strict=True):
        if (reference['input_token_ids'] != request['input_token_ids']
                or len(reference['output_rows']) != 3 or len(request['output_rows']) != 3):
            raise RuntimeError('native cover comparison has different inputs or missing output rows')
        for left, right in zip(reference['output_rows'], request['output_rows'], strict=True):
            comparisons.append(compare_row_artifacts(directory/'rows'/left['artifact'],
                                                     directory/'rows'/right['artifact']))
    return comparisons


def run_smoke(runner, plan):
    """Exercise the same native adapter witnesses for any catalog checkpoint."""
    def run(name, calls, caching=False, budget=4096):
        arm = _arm(inputs=plan, execution=runner.execution, calls=calls, prefix_caching=caching,
                   max_num_batched_tokens=budget)
        arm['engine']['max_model_len'] = 33024
        return runner.run_arm(name, arm)

    prompts = [plan['batch_prompts'][i] for i in (6,12,19)]
    directory, result = run('mapping-and-capacity', mapping_calls(prompts, plan['pd_prompts'][-1]))
    comparisons = []
    for left_call, left_index, right_call, right_index in [(0,0,1,0), (0,0,3,0), (1,0,2,2), (1,1,2,1), (1,2,2,0)]:
        comparisons.append(compare_row_artifacts(_row_path(directory,result,left_call,left_index),
                                                 _row_path(directory,result,right_call,right_index)))
    long = result['calls'][4]
    witness = prefill_witness([row for event in long['schedule_events'] for row in event], 32768, 4096)
    if not witness['valid']:
        raise RuntimeError('native long-context schedule witness failed')
    comparisons.extend(check_cover_subset(directory, result))
    full_dir, full = run('full-prefill-capacity', [call('full', [plan['pd_prompts'][-1]])], budget=32768)
    full_witness = prefill_witness([row for event in full['calls'][0]['schedule_events'] for row in event], 32768, 32768)
    if not full_witness['valid']:
        raise RuntimeError('native full-prefill capacity witness failed')
    comparisons.append(compare_row_artifacts(_row_path(directory,result,4,0), _row_path(full_dir,full,0,0)))
    warm_dir, warm = run('generated-prefix', [call('donor', [plan['generated_donor']], 128),
        dict(call('reuse', [], 2), output_prefix_from=dict(call=0, request=0, count=127,
                                                        suffix=plan['generated_suffix']))], caching=True)
    query = warm['calls'][1]['requests'][0]
    if not 1024 < query['num_cached_tokens'] <= 1151:
        raise RuntimeError('native generated-prefix hit did not extend into generated tokens')
    cold_dir, cold = run('generated-cold', [call('cold', [query['input_token_ids']], 2)])
    comparisons.append(compare_row_artifacts(_row_path(warm_dir,warm,1,0), _row_path(cold_dir,cold,0,0)))
    summary = dict(status='pass' if all(row['bitwise_equal'] for row in comparisons) else 'mismatch',
                   comparisons=comparisons, chunk_witness=witness, full_prefill_witness=full_witness,
                   cached_tokens=query['num_cached_tokens'],
                   arms=runner.arm_records)
    return summary


def _smoke(args, parser):
    if args.gpu_index < 0:
        parser.error("--gpu-index must be nonnegative")
    root = Path(__file__).resolve().parents[2]
    if subprocess.check_output(['git','status','--porcelain'], cwd=root):
        raise RuntimeError('native smoke requires clean frozen source')
    if subprocess.check_output(['nvidia-smi','-i',str(args.gpu_index),'--query-compute-apps=pid','--format=csv,noheader'], text=True).strip():
        raise RuntimeError('native smoke requires the selected GPU idle; do not join the baseline cohort independently')
    args.output.mkdir(parents=True, exist_ok=False)
    plan = make_plan(args.checkpoint)
    execution = next(mode for mode in EXECUTION_CONFIGS if mode.key=='vosti-padded-graph')
    fields = dict(vosti_binary=str(args.binary.resolve()), deployment_bundle=str(args.deployment_bundle.resolve()),
                  multi_call=True, trace_steps=True, num_blocks=1152)
    manifest = dict(plan=plan, engine=fields, worker_python=str(args.worker_python.absolute()),
        source=subprocess.check_output(['git','rev-parse','HEAD'], cwd=root, text=True).strip(),
        binary_sha256=hashlib.sha256(args.binary.read_bytes()).hexdigest(),
        deployment_file_sha256=hashlib.sha256((args.deployment_bundle/'deployment.json').read_bytes()).hexdigest())
    (args.output/'inputs.json').write_text(json.dumps(manifest, indent=2)+'\n')
    runner = SuiteRunner(output=args.output, worker_root=root, worker_python=args.worker_python.absolute(),
                         gpu_index=args.gpu_index, execution=execution, arm_engine_fields=fields)

    summary = run_smoke(runner, plan)
    summary['manifest_sha256'] = hashlib.sha256((args.output/'inputs.json').read_bytes()).hexdigest()
    (args.output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(dict(status=summary['status'], comparisons=len(summary['comparisons']))), flush=True)
    if summary['status'] != 'pass':
        raise SystemExit(2)


def _prepare(args, parser):
    if args.gpu_index < 0:
        parser.error("--gpu-index must be nonnegative")
    root = Path(__file__).resolve().parents[2]
    source = subprocess.check_output(['git','rev-parse','HEAD'], cwd=root, text=True).strip()
    if subprocess.check_output(['git','status','--porcelain'], cwd=root):
        raise RuntimeError('native pipeline requires clean frozen source')
    worker = args.stack_root.resolve()/'vosti/.venv/bin/python'
    args.output.mkdir(parents=True, exist_ok=False)
    plans = {key: make_plan(key) for key in REPORT_MODELS}
    binaries = {key: root/'target/release/examples'/f'verus_engine_{plan["model"]["key"]}'
                for key,plan in plans.items()}
    manifest = dict(source=source, root=str(root), gpu=args.gpu_index,
                    stack_root=str(args.stack_root.resolve()), plans=plans,
                    binary_sha256={key:hashlib.sha256(path.read_bytes()).hexdigest() for key,path in binaries.items()})
    (args.output/'inputs.json').write_text(json.dumps(manifest, indent=2)+'\n')

    def state(phase, **extra):
        value = dict(phase=phase, updated_unix_s=time.time(), **extra)
        temporary = args.output/'status.tmp'
        temporary.write_text(json.dumps(value, indent=2)+'\n')
        temporary.replace(args.output/'status.json')
        print(json.dumps(value), flush=True)

    state('waiting_for_idle_gpu')
    while subprocess.check_output(['nvidia-smi','-i',str(args.gpu_index),'--query-compute-apps=pid','--format=csv,noheader'], text=True).strip():
        time.sleep(30)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu_index), VOSTI_FRAMEWORK_ROOT=str(root),
               PYTHONPATH=os.pathsep.join([str(root/'python'),str(root),str(root/'kernels')]))
    env['TRITON_CACHE_DIR'] = str(args.output/'compiler-cache/triton')
    env['TORCHINDUCTOR_CACHE_DIR'] = str(args.output/'compiler-cache/torchinductor')
    temporary = args.output/'tmp'
    temporary.mkdir()
    env['TMPDIR'] = str(temporary)

    def run(label, command):
        state('running', step=label)
        with (args.output/f'{label}.stdout.log').open('xb') as stdout, (args.output/f'{label}.stderr.log').open('xb') as stderr:
            result = subprocess.run(command, cwd=root, env=env, stdout=stdout, stderr=stderr)
        if result.returncode:
            raise RuntimeError(f'{label} exited {result.returncode}; inspect preserved logs')

    try:
        mapping, peaks, totals = {}, [], []
        for key, plan in plans.items():
            bundle = args.output/'deployments'/key
            telemetry = args.output/f'deploy-{key}.telemetry.json'
            run(f'deploy-{key}', [str(worker), str(root/'scripts/common/gpu_monitor.py'),
                '--gpu-index',str(args.gpu_index),'--output',str(telemetry),'--',str(worker),
                str(root/'scripts/prepare_deployment.py'), '--family', plan['model']['key'],
                plan['model_path'], '--output',str(bundle)])
            load_complete_telemetry(telemetry)
            smoke = args.output/'smokes'/key
            run(f'smoke-{key}', [str(worker), '-u','-m','scripts.determinism_tests.native_report', 'smoke',
                '--gpu-index',str(args.gpu_index),'--checkpoint',key,'--worker-python',str(worker),'--binary',str(binaries[key]),
                '--deployment-bundle',str(bundle),'--output',str(smoke)])
            summary = json.loads((smoke/'summary.json').read_text())
            if summary['status'] != 'pass':
                raise RuntimeError('native adapter smoke did not pass')
            peaks.append(max(arm['telemetry']['metric_summary']['memory_used_mib']['max'] for arm in summary['arms']))
            totals.extend(float(arm['telemetry']['gpu']['memory_total_mib']) for arm in summary['arms'])
            mapping[key] = dict(vosti_binary=str(binaries[key]), deployment_bundle=str(bundle))
        concurrency = 2 if sum(peaks) < .85*min(totals) else 1
        state('native_capacity_qualified', individual_peak_mib=peaks, concurrency=concurrency)
        mapping_path = args.output/'native-bundles.json'
        mapping_path.write_text(json.dumps(mapping, indent=2)+'\n')
        run('report', [str(worker),'-u','-m','scripts.determinism_tests.report_campaign',
            '--gpu-index',str(args.gpu_index),'--stack-root',str(args.stack_root.resolve()),'--output',str(args.output/'run'),
            '--mode','vosti-padded-graph','--native-bundles',str(mapping_path),'--concurrency',str(concurrency)])
        statuses = json.loads((args.output/'run/status.json').read_text())
        done = all(statuses[f'{key}/vosti-padded-graph']['status'] in ('pass','mismatch') for key in REPORT_MODELS)
        state('native_rows_finished' if done else 'native_rows_incomplete', report=str(args.output/'run'))
    except Exception as error:
        state('failed', error=repr(error))
        raise


def _run(args, parser):
    root = Path(__file__).resolve().parents[2]
    worker_root = args.worker_root.resolve()
    if args.num_blocks <= 0 or args.gpu_index < 0:
        raise ValueError('invalid cache capacity or GPU index')
    # Untracked unrelated work is neither executed nor part of the attestation.
    subprocess.run(['git', 'diff', '--exit-code', 'HEAD', '--'], cwd=root, check=True)
    controller_files = sorted((root/'scripts/determinism_tests').glob('*.py'))
    for path in controller_files:
        subprocess.run(['git', 'ls-files', '--error-unmatch', str(path)],
                       cwd=root, check=True, stdout=subprocess.DEVNULL)
    if gpu_pids(args.gpu_index):
        raise RuntimeError('selected GPU is occupied; no processes were signalled')
    plan = make_plan(args.checkpoint)
    mode = next(m for m in EXECUTION_CONFIGS if m.key == 'vosti-padded-graph')
    fields = dict(vosti_binary=str(args.binary.resolve()),
                  deployment_bundle=str(args.deployment_bundle.resolve()),
                  multi_call=True, trace_steps=True, num_blocks=args.num_blocks)
    bound_paths = [args.binary.resolve(),
                   *(args.deployment_bundle.resolve()/name for name in
                     ('deployment.json', 'deployment-candidate.json', 'backend-qualification-report.json')),
                   *controller_files]
    manifest = dict(plan=plan, gpu=args.gpu_index, engine=fields,
        controller_source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
        worker_root=str(worker_root), worker_source=source_identity(worker_root),
        worker_python=str(args.worker_python.absolute()),
        packages=package_inventory(args.worker_python),
        files={str(path):file_hash(path) for path in bound_paths})
    args.output.mkdir(parents=True, exist_ok=False)
    save(args.output/'inputs.json', manifest)
    phase = 'smoke'
    done = threading.Event()

    def heartbeat():
        while not done.wait(30):
            arms = sorted((args.output/phase/'arms').glob('*/arm.json'), key=lambda p:p.stat().st_mtime)
            label = arms[-1].parent.name if arms else 'initializing'
            event = dict(checkpoint=args.checkpoint, phase=phase, arm=label, unix_s=time.time())
            save(args.output/'heartbeat.json', event)
            print('HEARTBEAT', json.dumps(event), flush=True)

    def runner(directory):
        directory.mkdir()
        return SuiteRunner(output=directory, worker_root=worker_root,
            worker_python=args.worker_python.absolute(), gpu_index=args.gpu_index,
            execution=mode, arm_engine_fields=fields)

    monitor = threading.Thread(target=heartbeat, daemon=True)
    monitor.start()
    try:
        print('SMOKE START', args.checkpoint, flush=True)
        smoke = run_smoke(runner(args.output/'smoke'), plan)
        save(args.output/'smoke/summary.json', smoke)
        if smoke['status'] != 'pass':
            raise RuntimeError('adapter/capacity smoke did not pass; full suite was not started')
        print('SMOKE PASS', args.checkpoint, flush=True)
        phase = 'suite'
        result = run_report(runner(args.output/'suite'), plan, mode)
        if source_identity(worker_root) != manifest['worker_source']:
            raise RuntimeError('qualified worker source changed during execution')
        if any(file_hash(Path(path)) != sha for path,sha in manifest['files'].items()):
            raise RuntimeError('bound controller, binary or deployment changed during execution')
        save(args.output/'complete.json', dict(status=result['status'], cells=result['cells'],
             finished_unix_s=time.time(), manifest_sha256=file_hash(args.output/'inputs.json')))
        print('FINISHED', args.checkpoint, json.dumps(result['cells']), flush=True)
        if result['status'] != 'pass':
            raise RuntimeError('suite did not pass; retained comparisons distinguish mismatches from invalid runs')
    except Exception as error:
        save(args.output/'error.json', dict(phase=phase,error=repr(error),unix_s=time.time()))
        raise
    finally:
        done.set()
        monitor.join()


def build_parser():
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest='command', required=True)

    def command(name, description, handler):
        parser = commands.add_parser(name, help=description, description=description)
        parser.set_defaults(handler=handler, command_parser=parser)
        return parser

    parser = command('prepare', 'Qualify deployments and run both native report rows.', _prepare)
    parser.add_argument('--stack-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpu-index', type=int, default=0)

    parser = command('run', 'Smoke-check and run one checkpoint from an existing qualified worker.', _run)
    parser.add_argument('--checkpoint', choices=[c.key for c in NATIVE_CHECKPOINTS], required=True)
    parser.add_argument('--gpu-index', type=int, required=True)
    parser.add_argument('--worker-root', type=Path, required=True)
    parser.add_argument('--worker-python', type=Path, required=True)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--deployment-bundle', type=Path, required=True)
    parser.add_argument('--num-blocks', type=int, default=768)
    parser.add_argument('--output', type=Path, required=True)

    parser = command('smoke', 'Check native row mapping, cache reuse and capacity without running the full report.', _smoke)
    parser.add_argument('--checkpoint', choices=REPORT_MODELS, required=True)
    parser.add_argument('--worker-python', type=Path, required=True)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--deployment-bundle', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpu-index', type=int, default=0)

    return root


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.handler(args, args.command_parser)


if __name__ == '__main__':
    main()
