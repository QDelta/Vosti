"""Official grading with a least-privilege container factory and pinned images.

Only container creation is replaced: patch application, tests and scoring remain
the installed official SWE-bench implementation. Runs after all inference so
grading CPU/IO cannot perturb serving measurements.
"""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import select
import time

from .prepare import save, sha


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pilot', type=Path, required=True)
    p.add_argument('--wait-pid', type=int, help='Wait for the owned inference campaign to exit via Linux pidfd')
    a = p.parse_args()
    root = a.pilot.resolve()
    if a.wait_pid is not None:
        fd = os.pidfd_open(a.wait_pid)
        try:
            command = Path(f'/proc/{a.wait_pid}/cmdline').read_bytes().split(b'\0')
            if b'scripts.agent_benchmark.campaign' not in command or str(root).encode() not in command:
                raise RuntimeError('wait PID is not this pilot campaign')
            print('WAITING for inference campaign exit using pidfd', a.wait_pid, flush=True)
            select.select([fd], [], [])
        finally:
            os.close(fd)
    if not (root/'inference-complete.json').is_file():
        raise RuntimeError('finish inference before running grading')
    from swebench.harness import run_evaluation as harness
    plan = json.loads((root/'plan.json').read_text())
    images = json.loads((root/'images.json').read_text())
    grading = root/'grading'
    grading.mkdir(exist_ok=False)
    os.chdir(grading)
    save(grading/'identity.json', dict(swebench_version=importlib.metadata.version('swebench'),
        factory_sha256=sha(__file__), harness_sha256=sha(harness.__file__),
        dataset_sha256=sha(root/'grading-instances.json'), images=images,
        adaptation='container creation only: pinned local image, no network/mounts/GPU, 2 CPUs/8GB, no added capabilities, BLAS/OpenMP thread limits 2'))

    def create_container(test_spec, client, run_id, logger):
        iid = test_spec.instance_id
        image = images[iid]['id']
        client.images.get(image)  # Never pull a floating tag while grading.
        return client.containers.create(image=image, name=f'sweb.eval.{iid.lower()}.{run_id}',
            user='root', detach=True, command='tail -f /dev/null', network_mode='none',
            cap_drop=['ALL'], security_opt=['no-new-privileges'], nano_cpus=2_000_000_000,
            environment={k:'2' for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']},
            mem_limit='8g', pids_limit=256, labels={'vosti.swebench-pilot-grading':'true'})

    harness.create_container = create_container
    for mode in plan['modes']:
        start = time.monotonic()
        predictions = root/'trials'/mode/'predictions.json'
        run_id = f'{root.name}-{mode}'
        print('GRADE START', mode, flush=True)
        harness.main(dataset_name=str(root/'grading-instances.json'), split='test',
            instance_ids=list(images), predictions_path=str(predictions), max_workers=4,
            open_file_limit=4096, run_id=run_id, timeout=600, rewrite_reports=False,
            modal=False, report_dir=str(grading/'reports'))
        save(grading/f'{mode}-receipt.json',dict(predictions_sha256=sha(predictions),
            elapsed_s=time.monotonic()-start, run_id=run_id))
        print('GRADE DONE', mode, flush=True)
    save(root/'grading-complete.json', dict(complete=True, modes=plan['modes']))
    from .report import render
    render(root)


if __name__ == '__main__':
    main()
