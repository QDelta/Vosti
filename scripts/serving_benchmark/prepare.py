"""Prepare reproducible serving workloads and trial plans; never launch a GPU job."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.common.tokenizers import tokenizer_artifact_sha256
from scripts.determinism_tests.protocol import CHECKPOINTS, EXECUTION_CONFIGS
from scripts.serving_benchmark.matrix import performance_matrix
from scripts.serving_benchmark.multi_turn import (
    prepare as prepare_multi_turn, prepare_arrival_trace, validate_warmup, write_new,
)
from scripts.serving_benchmark.phases import prepare as prepare_phase, validate as validate_phase
from scripts.serving_benchmark.server_config import (
    PERFORMANCE_EXECUTION_CONFIGS, ServerSettings, server_spec,
)
from scripts.serving_benchmark.server_trial import validate_jobs
from scripts.serving_benchmark.multi_session_replay import prepare_inputs


def _multi_session_replay(args, parser):
    if args.seed < 0 or len(args.mode) != len(set(args.mode)):
        parser.error('seed must be nonnegative and modes must be unique')
    settings = ServerSettings(gpu_index=args.gpu_index, port=args.port, context_limit=32768)
    settings.validate()
    root = (args.root or ROOT).resolve()
    source = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    checkpoint = next(c for c in CHECKPOINTS if c.key == args.checkpoint)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(checkpoint.path, local_files_only=True)
    inputs = prepare_inputs(tokenizer, checkpoint, seed=args.seed)

    def donor(prompt, description):
        return dict(inputs['measured'], description=description, sessions=[dict(
            id='donor', initial_prompt=prompt, initial_tokens=len(tokenizer.encode(prompt)),
            turns=[dict(suffix='', suffix_tokens=0, max_tokens=1, delay_s=0.0)])])

    artifacts = dict(measured=inputs['measured'], arrivals=inputs['arrivals'],
        **{'decode-donor': donor(inputs['donor_warmup'], 'Excluded decode-warmup cache donor.'),
           'decode-warmup': inputs['warmup'],
           'prefix-warmup': donor(inputs['shared_prefix_warmup'], 'Excluded shared-prefix cache donor.')})
    paths = {name: output / f'{name}.json' for name in artifacts}
    for name, data in artifacts.items():
        write_new(paths[name], data)
    jobs = [dict(id='multi-session-replay', kind='multi_session_arrivals',
        workload=str(paths['measured']),
        warmup=[str(paths[name]) for name in ('decode-donor', 'decode-warmup', 'prefix-warmup')],
        concurrency=4, request_rate=inputs['arrivals']['request_rate'],
        arrival_seed=args.seed, arrival_process='poisson', arrival_trace=str(paths['arrivals']))]
    jobs_path = output / 'jobs.json'
    trials = []
    for key in args.mode:
        mode = next(m for m in PERFORMANCE_EXECUTION_CONFIGS if m.key == key)
        spec = server_spec(root=root, stack=args.stack_root.resolve(), checkpoint=checkpoint,
            mode=mode, settings=settings, cache=output / 'compiler-cache' / key,
            build=output / 'build', bundle=output / 'deployments' / source / checkpoint.key)
        validate_jobs(spec, jobs, tokenizer)
        path = output / f'{key}-launch.json'
        write_new(path, spec)
        trials.append(dict(id=f'{checkpoint.key}-{key}', spec=str(path), jobs=str(jobs_path),
                          output=str(output / 'trials' / key), status='planned'))
    write_new(jobs_path, jobs)
    write_new(output / 'plan.json', dict(framework_source=source, checkpoint=checkpoint.key,
        trials=trials, cell_count=len(trials), status='planned_not_executed',
        scope='Multi-session replay: four sessions, six turns, saved request-based Poisson arrivals.',
        note='Warmup stages are ordered and excluded. Retain generated output text; '
             'audit realized cache hits and token geometry. Mode selection is not backend qualification.'))
    print(f'Prepared {len(trials)} multi-session replay trial(s); jobs: {jobs_path}', flush=True)


def phase_screen_jobs(tokenizer, checkpoint, artifacts, output, seed):
    """One measured wave per base phase; multi-turn remains the main workload."""
    jobs = []
    for index, (kind, context, query, count) in enumerate((
        ('cold_prefill', 0, 4096, 1),
        ('decode', 8192, 0, 64),
        ('cached_extension', 8192, 128, 1),
    )):
        data = prepare_phase(tokenizer, kind=kind, context=context, query=query,
            output=count, concurrency=1, waves=1, seed=seed + 100 + index)
        data.update(tokenizer=checkpoint.path, tokenizer_artifact_sha256=artifacts)
        path = output / f'phase-{kind}.json'
        write_new(path, data)
        jobs.append(dict(id=f'phase-{kind}', kind='phase', workload=str(path)))
    return jobs


def _pilot(args, parser):
    if args.phase_screen and args.replay_jobs:
        parser.error('--phase-screen cannot modify saved replay jobs')
    root = (args.root or Path(__file__).resolve().parents[2]).resolve()
    source = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    checkpoint = next(c for c in CHECKPOINTS if c.key == args.checkpoint)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(checkpoint.path, local_files_only=True)
    if args.replay_jobs:
        jobs = json.loads(args.replay_jobs.read_text())
    else:
        artifacts = tokenizer_artifact_sha256(Path(checkpoint.path))
        paths, manifests = {}, {}
        for name, seed in (('measured', args.seed), ('warmup', args.seed + 1)):
            data = prepare_multi_turn(tokenizer, sessions=4, turns=args.turns, initial_tokens=8192,
                suffix_tokens=128, output_tokens=args.output_tokens, think_seconds=.5, seed=seed,
                suffix_includes_separator=True)
            data.update(tokenizer=checkpoint.path, tokenizer_artifact_sha256=artifacts)
            manifests[name] = data
            paths[name] = output / f'{name}.json'
            write_new(paths[name], data)
        trace = prepare_arrival_trace(manifests['measured'], request_rate=args.request_rate, seed=args.seed)
        trace_path = output / 'arrivals.json'
        write_new(trace_path, trace)
        jobs = [dict(id='multi-turn', kind='multi_session_arrivals', workload=str(paths['measured']),
            warmup=str(paths['warmup']), concurrency=4, request_rate=args.request_rate,
            arrival_seed=args.seed, arrival_process='poisson', arrival_trace=str(trace_path))]
        if args.phase_screen:
            jobs = phase_screen_jobs(tokenizer, checkpoint, artifacts, output, args.seed) + jobs
    jobs_path = output / 'jobs.json'
    write_new(jobs_path, jobs)
    settings = ServerSettings(context_limit=16384, vosti_num_blocks=1024)
    trials = []
    modes = args.mode or ['vllm-invariant-flash-attn', 'sglang-deterministic-fa3', 'vosti-padded-graph']
    if len(modes) != len(set(modes)):
        parser.error('duplicate modes')
    for key in modes:
        mode = next(m for m in PERFORMANCE_EXECUTION_CONFIGS if m.key == key)
        spec = server_spec(root=root, stack=args.stack_root.resolve(), checkpoint=checkpoint,
            mode=mode, settings=settings, cache=output / 'compiler-cache' / key,
            build=output / 'build', bundle=output / 'deployments' / source / checkpoint.key)
        validate_jobs(spec, jobs, tokenizer)
        spec_path = output / f'{key}-launch.json'
        write_new(spec_path, spec)
        trials.append(dict(id=key, spec=str(spec_path), jobs=str(jobs_path),
                           output=str(output / 'trials' / key), status='planned'))
    write_new(output / 'plan.json', dict(framework_source=source, trials=trials, cell_count=len(trials),
        scope='Short multi-turn compatibility/performance pilot, not a full matrix or capacity study',
        status='planned_not_executed', checkpoint=checkpoint.key,
        note='Illustrative common offered rate, not calibrated capacity. Audit cache/token geometry, '
             'backend selection, measured graph capture/compilation and GPU contention before timing claims.'))
    print(f'Prepared {len(trials)} trial(s); jobs: {jobs_path}', flush=True)


def _multi_workloads(args, parser):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    from transformers import AutoTokenizer

    records = []
    for checkpoint in CHECKPOINTS:
        if not checkpoint.performance:
            continue
        tokenizer = AutoTokenizer.from_pretrained(checkpoint.path, local_files_only=True)
        artifacts = tokenizer_artifact_sha256(Path(checkpoint.path))
        for context, generated in ((8192, 512), (32768, 1024)):
            for repetition in range(3):
                pair = {}
                paths = {}
                for name, sessions, seed in [('measured', 8, 42 + repetition * 100),
                                               ('warmup', 4, 43 + repetition * 100)]:
                    data = prepare_multi_turn(tokenizer, sessions=sessions, turns=4,
                        initial_tokens=context, suffix_tokens=128, output_tokens=generated,
                        think_seconds=.5, seed=seed, suffix_includes_separator=True)
                    data.update(tokenizer=checkpoint.path, tokenizer_artifact_sha256=artifacts)
                    pair[name] = data
                    paths[name] = output / f'{checkpoint.key}-context-{context}-rep-{repetition}-{name}.json'
                validate_warmup(pair['measured'], pair['warmup'], tokenizer, context_limit=40960)
                for name, data in pair.items():
                    write_new(paths[name], data)
                records.append(dict(checkpoint=checkpoint.key, context=context,
                    output_tokens=generated, repetition=repetition,
                    measured=str(paths['measured']), warmup=str(paths['warmup'])))
                print('PREPARED', checkpoint.key, context, repetition, flush=True)
    root = Path(__file__).resolve().parents[2]
    write_new(output / 'index.json', dict(status='prepared_not_executed', records=records,
        framework_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(),
        note='Common manifests for all engine modes; verify actual token geometry after serving.'))


def phase_cells():
    # One workload per model/shape/repetition, shared by all seven engine modes.
    return [row for row in performance_matrix()
            if row['execution_config'] == EXECUTION_CONFIGS[0].key
            and row['kind'] in {'cold_prefill', 'decode', 'cached_extension'}]


def _phase_workloads(args, parser):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    from transformers import AutoTokenizer

    records = []
    cells = phase_cells()
    for checkpoint in CHECKPOINTS:
        if not checkpoint.performance:
            continue
        tokenizer = AutoTokenizer.from_pretrained(checkpoint.path, local_files_only=True)
        artifacts = tokenizer_artifact_sha256(Path(checkpoint.path))
        for index, cell in enumerate(cells):
            if cell['checkpoint'] != checkpoint.key:
                continue
            context = cell.get('cached_tokens', cell.get('initial_context_tokens', 0))
            query = cell.get('query_tokens', 0)
            seed = 70000 + index  # Disjoint across shapes, concurrency and repeats.
            data = prepare_phase(tokenizer, kind=cell['kind'], context=context, query=query,
                output=cell['output_tokens'], concurrency=cell['concurrency'], waves=4, seed=seed)
            data.update(tokenizer=checkpoint.path, tokenizer_artifact_sha256=artifacts)
            validate_phase(data, tokenizer, context_limit=40960)
            path = output / (f"{checkpoint.key}-{cell['kind']}-c{context}-q{query}"
                             f"-concurrency{cell['concurrency']}-rep{cell['repetition']}.json")
            write_new(path, data)
            records.append(dict(cell={k: v for k, v in cell.items() if k != 'execution_config'},
                                workload=str(path), seed=seed))
            print('PREPARED', path.name, flush=True)
    write_new(output / 'index.json', dict(status='prepared_not_executed', records=records,
        note='Four measured waves and one disjoint warmup wave per cell. '
             'Donor calls are retained separately and excluded from measurement. '
             'Observe actual reused KV and effective uncached queries at runtime.'))


def _multi_trials(args, parser):
    root = args.root.resolve() if args.root else Path(__file__).resolve().parents[2]
    source = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    records = json.loads((args.workloads / 'index.json').read_text())['records']
    rates = json.loads(args.rates.read_text()) if args.rates else None
    if rates and rates.get('schema') != 'vosti.common-offered-rates.v1':
        raise ValueError('expected completed common-rate calibration')
    from transformers import AutoTokenizer
    trials = []
    for checkpoint in CHECKPOINTS:
        if not checkpoint.performance:
            continue
        tokenizer = AutoTokenizer.from_pretrained(checkpoint.path, local_files_only=True)
        for context in (8192, 32768):
            rows = sorted((row for row in records if row['checkpoint'] == checkpoint.key
                           and row['context'] == context), key=lambda row: row['repetition'])
            if [row['repetition'] for row in rows] != [0, 1, 2]:
                raise ValueError('expected exactly three repetitions per checkpoint/context')
            for load in (('light', 'moderate') if rates else (1, 4)):
                jobs = []
                for row in rows:
                    job = dict(id=f'rep-{row["repetition"]}', kind='multi_session',
                        workload=row['measured'], warmup=row['warmup'], concurrency=load,
                        checkpoint=checkpoint.key, context=context, repetition=row['repetition'])
                    if rates:
                        job.update(kind='multi_session_arrivals', concurrency=8,
                                   request_rate=rates['rates'][checkpoint.key][load],
                                   arrival_process='poisson', arrival_seed=42 + row['repetition'], load=load)
                        trace = prepare_arrival_trace(json.loads(Path(row['measured']).read_text()),
                            request_rate=job['request_rate'], seed=job['arrival_seed'])
                        trace_path = output / f'{checkpoint.key}-context{context}-{load}-rep{row["repetition"]}-arrivals.json'
                        write_new(trace_path, trace)
                        job['arrival_trace'] = str(trace_path)
                    jobs.append(job)
                group = f'{checkpoint.key}-context{context}-load{load}'
                jobs_path = output / f'{group}-jobs.json'
                for mode in EXECUTION_CONFIGS:
                    name = f'{group}-{mode.key}'
                    spec = server_spec(root=root, stack=args.stack_root.resolve(), checkpoint=checkpoint,
                        mode=mode, settings=ServerSettings(), cache=output / 'compiler-cache' / name,
                        build=output / 'build', bundle=output / 'deployments' / source / checkpoint.key)
                    if mode == EXECUTION_CONFIGS[0]:
                        validate_jobs(spec, jobs, tokenizer)
                        write_new(jobs_path, jobs)
                    spec_path = output / f'{name}-launch.json'
                    write_new(spec_path, spec)
                    trials.append(dict(id=name, spec=str(spec_path), jobs=str(jobs_path),
                                       output=str(output / 'trials' / name), status='planned'))
                print('VALIDATED', group, flush=True)
    write_new(output / 'plan.json', dict(status='planned_not_executed', framework_source=source,
        trials=trials, cell_count=len(trials) * 3, rate_selection=rates,
        scope='offered-rate multi-session' if rates else 'closed-loop multi-session',
        note='Fresh server per context/load/mode prevents cross-profile cache contamination. '
             'Build/qualification, capacity pilot and continuous contention telemetry are required before measurement.'))


def _phase_trials(args, parser):
    root = args.root.resolve() if args.root else Path(__file__).resolve().parents[2]
    source = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    records = json.loads((args.workloads / 'index.json').read_text())['records']
    if len(records) != 300:
        raise ValueError('expected all 300 engine-independent phase workloads')
    from transformers import AutoTokenizer
    trials = []
    for checkpoint in CHECKPOINTS:
        if not checkpoint.performance:
            continue
        rows = [record for record in records if record['cell']['checkpoint'] == checkpoint.key]
        if len(rows) != 60:
            raise ValueError(f'{checkpoint.key}: expected sixty shape/repetition jobs')
        jobs = [dict(id=Path(row['workload']).stem, kind='phase', workload=row['workload'],
                     matrix_cell=row['cell']) for row in rows]
        jobs_path = output / f'{checkpoint.key}-jobs.json'
        tokenizer = AutoTokenizer.from_pretrained(checkpoint.path, local_files_only=True)
        for mode in EXECUTION_CONFIGS:
            name = f'{checkpoint.key}-{mode.key}'
            spec = server_spec(root=root, stack=args.stack_root.resolve(), checkpoint=checkpoint,
                mode=mode, settings=ServerSettings(), cache=output / 'compiler-cache' / name,
                build=output / 'build', bundle=output / 'deployments' / source / checkpoint.key)
            # Same job geometry for every mode; validate once per tokenizer.
            if mode == EXECUTION_CONFIGS[0]:
                validate_jobs(spec, jobs, tokenizer)
                write_new(jobs_path, jobs)
            spec_path = output / f'{name}-launch.json'
            write_new(spec_path, spec)
            trials.append(dict(id=name, spec=str(spec_path), jobs=str(jobs_path),
                               output=str(output / 'trials' / name), status='planned'))
        print('VALIDATED', checkpoint.key, '60 disjoint jobs across seven modes', flush=True)
    write_new(output / 'plan.json', dict(status='planned_not_executed', framework_source=source,
        trials=trials, cell_count=sum(len(json.loads(Path(t['jobs']).read_text())) for t in trials),
        scope='Phase workloads only; multi-session and arrival-rate trials remain separate.',
        required_before_measurement=['freeze source and check all package inventories',
            'build servers and qualify fresh Vosti deployment bundles',
            'run capacity and graph-warmup pilots before full trials',
            'wrap every server trial with continuous GPU contention telemetry']))


def _capacity_trials(args, parser):
    root, output = args.root.resolve(), args.output.resolve()
    source = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
    output.mkdir(parents=True, exist_ok=False)
    from transformers import AutoTokenizer
    settings = ServerSettings()
    trials = []
    for index, checkpoint in enumerate(CHECKPOINTS):
        if not checkpoint.performance:
            continue
        tokenizer = AutoTokenizer.from_pretrained(checkpoint.path, local_files_only=True)
        data = prepare_phase(tokenizer, kind='decode', context=settings.context_limit - 256,
                       query=0, output=256, concurrency=4, waves=1, seed=99000 + index)
        data.update(tokenizer=checkpoint.path,
                    tokenizer_artifact_sha256=tokenizer_artifact_sha256(Path(checkpoint.path)))
        workload_path = output / f'{checkpoint.key}-workload.json'
        write_new(workload_path, data)
        jobs = [dict(id='capacity', kind='phase', workload=str(workload_path))]
        jobs_path = output / f'{checkpoint.key}-jobs.json'
        for mode in EXECUTION_CONFIGS:
            name = f'capacity-{checkpoint.key}-{mode.key}'
            spec = server_spec(root=root, stack=args.stack_root.resolve(), checkpoint=checkpoint,
                mode=mode, settings=settings, cache=output / 'compiler-cache' / name,
                build=args.build.resolve(), bundle=output / 'deployments' / source / checkpoint.key)
            if mode == EXECUTION_CONFIGS[0]:
                validate_jobs(spec, jobs, tokenizer)
                write_new(jobs_path, jobs)
            spec_path = output / f'{name}-launch.json'
            write_new(spec_path, spec)
            trials.append(dict(id=name, spec=str(spec_path), jobs=str(jobs_path),
                               output=str(output / 'trials' / name), status='planned'))
        print('PREPARED capacity', checkpoint.key, flush=True)
    write_new(output / 'plan.json', dict(status='planned_not_executed', framework_source=source,
        trials=trials, cell_count=len(trials), scope='near-limit C4 capacity probes, not benchmark cells',
        note='40,704 input tokens plus 256 generated, concurrency four, one disjoint warmup wave '
             'and one probe wave. Confirms this workload executes; inspect actual batching and '
             'graph/memory telemetry. Does not establish arbitrary-workload memory feasibility.'))


def build_parser():
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="preset", required=True)

    def command(name, description, handler):
        parser = commands.add_parser(name, help=description, description=description)
        parser.set_defaults(handler=handler, command_parser=parser)
        return parser

    parser = command('multi-session-replay',
        'Prepare four-session, six-turn replay with ordered warmup and seeded request arrivals.',
        _multi_session_replay)
    parser.add_argument('--checkpoint', choices=tuple(c.key for c in CHECKPOINTS), required=True)
    parser.add_argument('--mode', action='append', required=True,
        choices=tuple(m.key for m in PERFORMANCE_EXECUTION_CONFIGS),
        help='Explicit modes to prepare; availability must be qualified separately')
    parser.add_argument('--stack-root', type=Path, required=True)
    parser.add_argument('--root', type=Path, help='Frozen serving checkout; defaults to this checkout')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--gpu-index', type=int, default=0)
    parser.add_argument('--port', type=int, default=18300)

    parser = command('pilot',
        'Prepare one short, shared multi-turn pilot for the three comparison engines.', _pilot)
    parser.add_argument('--checkpoint', choices=tuple(c.key for c in CHECKPOINTS if c.performance),
                        default='gemma3-4b')
    parser.add_argument('--stack-root', type=Path, required=True)
    parser.add_argument('--root', type=Path,
        help='Clean serving checkout; defaults to this checkout')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--request-rate', type=float, default=.5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--turns', type=int, default=3,
        help='Turns per session for a new workload; replay-jobs keeps the saved geometry')
    parser.add_argument('--output-tokens', type=int, default=256,
        help='Exact requested output length for a new workload')
    parser.add_argument('--mode', action='append',
        choices=tuple(mode.key for mode in PERFORMANCE_EXECUTION_CONFIGS),
        help='Run only selected modes; by default prepare all three')
    parser.add_argument('--replay-jobs', type=Path,
        help='Reuse an existing pilot jobs file, including its saved workloads and arrival trace')
    parser.add_argument('--phase-screen', action='store_true',
        help='Add one small measured wave each of cold prefill, decode and cached extension')

    parser = command('multi-workloads',
        'Prepare engine-independent multi-session manifests; no GPU or HTTP execution.', _multi_workloads)
    parser.add_argument('--output', type=Path, required=True)

    parser = command('phase-workloads',
        'Freeze token-counted phase workloads for every performance matrix geometry.', _phase_workloads)
    parser.add_argument('--output', type=Path, required=True)

    parser = command('multi-trials',
        'Prepare closed-loop multi-session trials, or arrivals after common calibration.', _multi_trials)
    parser.add_argument('--workloads', type=Path, required=True)
    parser.add_argument('--stack-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--root', type=Path, help='Frozen serving checkout; defaults to this checkout')
    parser.add_argument('--rates', type=Path,
                        help='Common-rate selector output; omit for closed-loop C1/C4 trials')

    parser = command('phase-trials',
        'Group the 2,100 phase cells into 35 sequential server trials; no GPU launch.', _phase_trials)
    parser.add_argument('--workloads', type=Path, required=True)
    parser.add_argument('--stack-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--root', type=Path, help='Frozen serving checkout; defaults to this checkout')

    parser = command('capacity-trials',
        'Prepare near-limit C4 capacity probes, separate from benchmark matrix cells.', _capacity_trials)
    parser.add_argument('--root', type=Path, required=True, help='Frozen serving checkout')
    parser.add_argument('--stack-root', type=Path, required=True)
    parser.add_argument('--build', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)

    return root


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.handler(args, args.command_parser)


if __name__ == '__main__':
    main()
