"""Validate the pilot's official grader using reference patches, never model inputs."""
import argparse
import json
import os
from pathlib import Path

from .prepare import save, sha


def main():
    from swebench.harness import run_evaluation as harness
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pilot', type=Path, required=True)
    a = p.parse_args()
    root = a.pilot.resolve()
    assert (root/'grading-complete.json').exists()
    out = root/'grading-control'
    out.mkdir(exist_ok=False)
    rows = json.loads((root/'grading-instances.json').read_text())
    images = json.loads((root/'images.json').read_text())
    save(out/'predictions.json', [dict(instance_id=r['instance_id'], model_name_or_path='gold-control', model_patch=r['patch']) for r in rows])
    save(out/'identity.json', dict(script_sha256=sha(__file__), harness_sha256=sha(harness.__file__),
        dataset_sha256=sha(root/'grading-instances.json'), predictions_sha256=sha(out/'predictions.json')))

    def create_container(test_spec, client, run_id, logger):
        return client.containers.create(image=images[test_spec.instance_id]['id'],
            name=f'sweb.eval.{test_spec.instance_id.lower()}.{run_id}', user='root', detach=True,
            command='tail -f /dev/null', network_mode='none', cap_drop=['ALL'],
            security_opt=['no-new-privileges'], nano_cpus=2_000_000_000, mem_limit='8g', pids_limit=256,
            environment={k:'2' for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']},
            labels={'vosti.swebench-pilot-grading':'true'})

    harness.create_container = create_container
    os.chdir(out)
    run_id = root.name+'-gold-control'
    harness.main(dataset_name=str(root/'grading-instances.json'), split='test',
        instance_ids=[r['instance_id'] for r in rows], predictions_path=str(out/'predictions.json'),
        max_workers=4, open_file_limit=4096, run_id=run_id,
        timeout=600, rewrite_reports=False, modal=False, report_dir=str(out/'reports'))
    result = json.loads((out/'reports'/f'gold-control.{run_id}.json').read_text())
    save(out/'complete.json', dict(complete=True, passed=result['resolved_instances'], total=len(rows)))
    print('GOLD CONTROL', result['resolved_instances'], '/', len(rows), flush=True)


if __name__ == '__main__':
    main()
