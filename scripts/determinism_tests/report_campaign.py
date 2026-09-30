"""Run selected available report rows; keep all 14 rows visible in the manifest."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import threading

from scripts.determinism_tests.protocol import EXECUTION_CONFIGS, sha256_json
from scripts.determinism_tests.report_plan import make_plan, table_matrix, REPORT_MODELS
from scripts.determinism_tests.report_suite import run_report
from scripts.determinism_tests.run import SuiteRunner
from scripts.common.gpu_monitor import process_start_time_ticks
from scripts.common.artifacts import package_inventory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stack-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mode', action='append', required=True,
                        choices=[mode.key for mode in EXECUTION_CONFIGS])
    parser.add_argument('--native-bundles', type=Path,
                        help='JSON mapping each report checkpoint to vosti_binary and deployment_bundle')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--concurrency', type=int, choices=(1,2), default=2)
    parser.add_argument('--gpu-index', type=int, default=0)
    args = parser.parse_args()
    if args.gpu_index < 0:
        parser.error("--gpu-index must be nonnegative")
    root = Path(__file__).resolve().parents[2]
    if subprocess.check_output(['git', 'status', '--porcelain'], cwd=root):
        raise RuntimeError('report requires clean frozen source')
    if subprocess.check_output(['nvidia-smi', '-i', str(args.gpu_index), '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip():
        raise RuntimeError('report requires the selected GPU initially idle')
    source = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
    plans = {key: make_plan(key) for key in REPORT_MODELS}
    native = {}
    if 'vosti-padded-graph' in args.mode:
        if args.native_bundles is None:
            raise ValueError('native report requires --native-bundles')
        native = json.loads(args.native_bundles.read_text())
        if set(native) != set(REPORT_MODELS):
            raise ValueError('native bundle mapping must cover both report checkpoints')
        for record in native.values():
            for key in ('vosti_binary', 'deployment_bundle'):
                record[key] = str(Path(record[key]).resolve())
            binary = Path(record['vosti_binary'])
            bundle = Path(record['deployment_bundle'])/'deployment.json'
            record.update(binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
                          deployment_file_sha256=hashlib.sha256(bundle.read_bytes()).hexdigest())
    inventories = {}
    for engine in sorted({mode.engine for mode in EXECUTION_CONFIGS if mode.key in args.mode}):
        worker = args.stack_root/engine/'.venv/bin/python'
        inventories[engine] = package_inventory(worker)
    manifest = dict(source=source, root=str(root), stack_root=str(args.stack_root.resolve()),
                    gpu=args.gpu_index, seed=1592598566, concurrency=args.concurrency, matrix=table_matrix(),
                    selected_modes=args.mode, plans=plans)
    manifest['inventories'] = inventories
    manifest['native_bundles'] = native
    if args.resume:
        previous = json.loads((args.output/'inputs.json').read_text())
        if previous != manifest:
            raise RuntimeError('report inputs, packages, path, or source changed; start a new campaign')
    else:
        args.output.mkdir(parents=True, exist_ok=False)
        (args.output/'inputs.json').write_text(json.dumps(manifest, indent=2)+'\n')
    statuses = {f'{row["checkpoint"]}/{row["execution"]}': dict(status='unrun') for row in manifest['matrix']}
    if (args.output/'status.json').exists():
        statuses = json.loads((args.output/'status.json').read_text())
    lock = threading.Lock()
    def status(key, value):
        with lock:
            statuses[key] = value
            temp = args.output/'status.tmp'
            temp.write_text(json.dumps(statuses, indent=2)+'\n')
            temp.replace(args.output/'status.json')
    identity = f'{os.getpid()}:{process_start_time_ticks(os.getpid())}'
    def run(model, mode):
        key = f'{model}/{mode.key}'
        output = args.output/'suites'/model/mode.key
        output.mkdir(parents=True, exist_ok=True)
        if (output/'summary.json').exists():
            result = json.loads((output/'summary.json').read_text())
            status(key, dict(status=result['status'], cells=result['cells'], summary=str(output/'summary.json')))
            return
        status(key, dict(status='running', progress=str(output/'progress.json')))
        runner = SuiteRunner(output=output, worker_python=args.stack_root/mode.engine/'.venv/bin/python',
                             worker_root=root, gpu_index=args.gpu_index, execution=mode, cohort_root=identity,
                             arm_engine_fields=(dict(vosti_binary=native[model]['vosti_binary'],
                                 deployment_bundle=native[model]['deployment_bundle'],
                                 multi_call=True, trace_steps=True, num_blocks=1152)
                                 if mode.engine == 'vosti' else None))
        try:
            result = run_report(runner, plans[model], mode)
            status(key, dict(status=result['status'], cells=result['cells'], summary=str(output/'summary.json')))
        except Exception as error:
            status(key, dict(status='invalid', error=repr(error), progress=str(output/'progress.json')))
            print(f'INVALID {key}: {error}', flush=True)
    modes = [next(mode for mode in EXECUTION_CONFIGS if mode.key==key) for key in args.mode]
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = [pool.submit(run, model, mode) for mode in modes for model in REPORT_MODELS]
        for future in futures:
            future.result()
    # Selected-mode runs remain partial until every matrix row has a result.
    complete = all(value['status'] in ('pass','mismatch') for value in statuses.values())
    result = dict(status='complete' if complete else 'partial', cases=statuses,
                  manifest_sha256=sha256_json(manifest))
    (args.output/('complete.json' if complete else 'partial.json')).write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    main()
