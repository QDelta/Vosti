"""File-backed aligned matrix with per-model summaries and exclusive GPU trials."""
import argparse
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import time
import traceback

from scripts.common.process_lifecycle import detached
from scripts.common.artifacts import package_inventory, shared_helper_sources
from scripts.common.telemetry import load_complete_telemetry
from scripts.common.gpu_monitor import process_start_time_ticks
from scripts.serving_benchmark.aligned.measure import inputs
from scripts.serving_benchmark.aligned.launch import adapt
from scripts.serving_benchmark.campaign import source_identity, file_hash
from scripts.serving_benchmark.multi_turn import write_new
from scripts.serving_benchmark.server_trial import gpu_pids

ROOT = Path(__file__).resolve().parents[3]
MODELS = ('gemma3-4b','llama3-8b','gemma3-27b','gemma4-31b')


def read(path):
    return json.loads(Path(path).read_text())


def run_trial(source, input_path, trial, gpu, build, locked):
    if any(file_hash(Path(p)) != sha for p,sha in locked.items()):
        raise RuntimeError('controller source changed during campaign')
    spec = read(source)
    frozen = Path(spec['benchmark_frozen_root'])
    spec = adapt(spec,gpu,trial/'cache',frozen)
    if spec['execution']['engine'] == 'vosti':
        spec['command'] = [build['binary']]
    else:
        spec['environment']['VOSTI_ALIGNED_PHASE_RECORDS'] = str(trial/'run/records')
        spec['environment']['PYTHONPATH'] = ':'.join([
            str(ROOT/'scripts/serving_benchmark/aligned/hooks'),str(ROOT),spec['environment']['PYTHONPATH']])
    write_new(trial/'launch.json',spec)
    command = [sys.executable,'-m','scripts.common.gpu_monitor','--gpu-index',str(gpu),
        '--output',str(trial/'telemetry.json'),'--',sys.executable,
        '-m','scripts.serving_benchmark.aligned.worker','--spec',str(trial/'launch.json'),
        '--inputs',str(input_path),'--output',str(trial/'run')]
    print('START',trial.parent.name,trial.name,'GPU',gpu,flush=True)
    started = time.time()
    with (trial/'worker.log').open('x') as log:
        proc = subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        write_new(trial/'started.json',dict(pid=proc.pid,start_ticks=process_start_time_ticks(proc.pid),
            started_unix_s=started,gpu=gpu,command=command))
        code = proc.wait()
    write_new(trial/'exit.json',dict(returncode=code,started_unix_s=started,finished_unix_s=time.time()))
    if code:
        raise RuntimeError(f'{trial}: worker exited {code}')
    load_complete_telemetry(trial/'telemetry.json')
    if not read(trial/'run/trial-complete.json')['complete']:
        raise RuntimeError('missing worker completion')
    if any(file_hash(Path(p)) != sha for p,sha in locked.items()):
        raise RuntimeError('controller source changed during measured trial')
    result = read(trial/'run/result.json')
    print('COMPLETE',trial.parent.name,trial.name,flush=True)
    return dict(model=trial.parent.name,mode=trial.name,gpu=gpu,**result)


def campaign(a):
    out = a.output.resolve();out.mkdir(parents=True,exist_ok=True)
    build = read(a.native_build/'build.json')
    if file_hash(Path(build['binary'])) != build['binary_sha256']:
        raise RuntimeError('native benchmark binary changed')
    sources = [p for p in (ROOT/'scripts/serving_benchmark/aligned').rglob('*') if p.suffix in ('.py','.rs')]
    sources += [ROOT/'scripts/serving_benchmark'/f'{n}.py' for n in
        ('cached_phases','server_trial','server_config','multi_turn','campaign')]
    sources += shared_helper_sources()
    locked = {str(p):file_hash(p) for p in sources}
    if file_hash(Path(__file__).with_name('native.rs')) != build['driver_sha256']:
        raise RuntimeError('native driver requires rebuild')
    from transformers import AutoTokenizer
    jobs = []
    packages = {}
    for model in a.models:
        launches = sorted((a.launches/model).glob('*/launch.json'),
                          key=lambda p:(p.parent.name!='vosti-padded-graph',p.parent.name))
        if a.modes:
            launches = [p for p in launches if p.parent.name in a.modes]
            missing = set(a.modes) - {p.parent.name for p in launches}
            if missing:
                raise ValueError(f'{model}: missing qualified launch modes {sorted(missing)}')
        if not launches:
            raise ValueError('empty requested model/mode set')
        checkpoint = read(launches[0])['checkpoint']
        tokenizer = AutoTokenizer.from_pretrained(checkpoint['path'],local_files_only=True)
        data = inputs(tokenizer,checkpoint,a.contexts,a.measured)
        path = out/model/'inputs.json';write_new(path,data)
        locked[str(path)] = file_hash(path)
        for source in launches:
            spec = read(source)
            if source_identity(Path(spec['benchmark_frozen_root'])) != build['source']:
                raise RuntimeError('frozen engine source differs from build')
            engine = spec['execution']['engine']
            if engine not in packages:
                python = Path(spec['benchmark_frozen_root'])/'.venv/bin/python' if engine=='vosti' else Path(spec['command'][0])
                packages[engine] = package_inventory(python)
            locked[str(source)] = file_hash(source)
            jobs.append(dict(source=str(source),model=model,mode=source.parent.name,inputs=str(path)))
    with tarfile.open(out/'controller-source.tar.gz','x:gz') as archive:
        for p in sources: archive.add(p,arcname=str(p.relative_to(ROOT)))
    write_new(out/'plan.json',dict(jobs=jobs,files=locked,build=build,packages=packages,
        contexts=a.contexts,measured_waves=a.measured,warmup_waves=2,gpus=a.gpus,
        expected_trials=len(jobs),expected_points=len(jobs)*2*len(a.contexts),seed=42))
    results = []
    # Finish one model before advancing; never run two jobs on one GPU.
    for model in a.models:
        pending = [job for job in jobs if job['model']==model]
        with ThreadPoolExecutor(max_workers=len(a.gpus)) as pool:
            active = {}
            while pending or active:
                used = set(active.values())
                for gpu in a.gpus:
                    if pending and gpu not in used and not gpu_pids(gpu):
                        job = pending.pop(0)
                        trial = out/model/job['mode'];trial.mkdir()
                        future = pool.submit(run_trial,Path(job['source']),Path(job['inputs']),trial,gpu,build,locked)
                        active[future]=gpu
                if active:
                    done,_ = wait(active,timeout=30,return_when=FIRST_COMPLETED)
                    for future in done:
                        del active[future]
                        results.append(future.result())
                elif pending:
                    print('WAIT: eligible GPUs occupied by external jobs; no benchmark launched',flush=True)
                    time.sleep(30)
        write_new(out/model/'summary.json',[r for r in results if r['model']==model])
        print('MODEL COMPLETE',model,flush=True)
    write_new(out/'measurements.json',results)
    write_new(out/'complete.json',dict(complete=True,trials=len(results),finished_unix_s=time.time()))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--native-build',type=Path,required=True)
    p.add_argument('--launches',type=Path,required=True,
                   help='Qualified launch specs laid out as MODEL/MODE/launch.json.')
    p.add_argument('--models',nargs='+',choices=MODELS,default=list(MODELS))
    p.add_argument('--modes',nargs='+')
    p.add_argument('--contexts',nargs='+',type=int,default=[0,4096,8192,12288,16384])
    p.add_argument('--measured',type=int,default=5)
    p.add_argument('--gpus',nargs='+',type=int,required=True)
    p.add_argument('--detach',action='store_true')
    a = p.parse_args()
    if any(gpu < 0 for gpu in a.gpus) or len(set(a.gpus)) != len(a.gpus):
        p.error('--gpus must contain distinct nonnegative indices')
    if a.detach:
        out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
        command=[sys.executable,'-m','scripts.serving_benchmark.aligned.campaign',
                 *[v for v in sys.argv[1:] if v!='--detach']]
        proc=detached(command,out/'supervisor.log',cwd=ROOT,env=dict(os.environ,PYTHONPATH=str(ROOT)))
        write_new(out/'supervisor.json',dict(pid=proc.pid,start_ticks=process_start_time_ticks(proc.pid),command=command))
        print(json.dumps(dict(pid=proc.pid,output=str(out))),flush=True)
    else:
        try: campaign(a)
        except BaseException:
            write_new(a.output/'failed.json',dict(error=traceback.format_exc()))
            raise


if __name__=='__main__':
    main()
