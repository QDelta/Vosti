"""Choose shared per-model offered rates from completed C4 multi-session runs.

This is a finite-workload pilot reference, not a saturation-capacity estimate.
All engines receive the same two rates for a model, including both contexts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics

from scripts.common.telemetry import load_complete_telemetry
from scripts.determinism_tests.protocol import CHECKPOINTS, EXECUTION_CONFIGS
from scripts.serving_benchmark.multi_turn import complete_result, digest, write_new


def select_rates(observations: list[dict], *, excluded_pairs: set[tuple[str, str]] | None = None) -> dict:
    excluded_pairs = excluded_pairs or set()
    pairs = {(model.key, mode.key) for model in CHECKPOINTS if model.performance for mode in EXECUTION_CONFIGS}
    if not excluded_pairs <= pairs:
        raise ValueError('unknown excluded performance checkpoint/mode pair')
    expected = {(model.key, mode.key, context, rep)
                for model in CHECKPOINTS if model.performance for mode in EXECUTION_CONFIGS
                for context in (8192, 32768) for rep in range(3)
                if (model.key, mode.key) not in excluded_pairs}
    indexed = {}
    for row in observations:
        key = tuple(row[name] for name in ('checkpoint', 'execution_config', 'context', 'repetition'))
        if key in indexed:
            raise ValueError(f'duplicate pilot cell {key}')
        rate = row['request_throughput_per_s']
        if not math.isfinite(rate) or rate <= 0:
            raise ValueError('pilot throughput must be positive and finite')
        indexed[key] = rate
    if indexed.keys() != expected:
        raise ValueError('need every planned C4 pilot outside explicit exclusions; both contexts and three repetitions')
    result = {}
    for model in CHECKPOINTS:
        if not model.performance:
            continue
        references = [dict(execution_config=mode.key, context=context,
            median_requests_per_s=statistics.median(indexed[(model.key, mode.key, context, rep)] for rep in range(3)))
            for mode in EXECUTION_CONFIGS for context in (8192, 32768)
            if (model.key, mode.key) not in excluded_pairs]
        if not references:
            raise ValueError(f'no measured reference remains for {model.key}')
        limiting = min(references, key=lambda row: row['median_requests_per_s'])
        base = limiting['median_requests_per_s']
        result[model.key] = dict(light=base * .5, moderate=base * .8,
                                limiting_reference=limiting, references=references,
                                excluded_modes=sorted(mode for checkpoint, mode in excluded_pairs if checkpoint == model.key))
    return result


def load_observation(entry: dict) -> dict:
    result_path, telemetry_path = Path(entry['result']), Path(entry['telemetry'])
    load_complete_telemetry(telemetry_path)
    telemetry = json.loads(telemetry_path.read_text())
    command = telemetry.get('command', [])
    if ('scripts.serving_benchmark.server_trial' not in command or '--output' not in command
            or Path(command[command.index('--output') + 1]).resolve() != result_path.parent.resolve()):
        raise ValueError('telemetry does not cover the result server trial')
    result = json.loads(result_path.read_text())
    if (result.get('schema') != 'vosti.multi-turn-result.v1' or not complete_result(result)
            or result.get('warmup') is None or not complete_result(result['warmup'])
            or result['concurrency'] != 4 or result['stagger_seconds'] != 0 or result.get('request_rate') is not None
            or result['engine_label'] != entry['execution_config']):
        raise ValueError('pilot must be complete, warmed, closed-loop C4 with the declared engine')
    if digest(result['workload']) != result['workload_sha256']:
        raise ValueError('pilot workload digest mismatch')
    if digest(result['warmup']['workload']) != result['warmup']['workload_sha256']:
        raise ValueError('pilot warmup digest mismatch')
    checkpoint = next(model for model in CHECKPOINTS if model.key == entry['checkpoint'])
    if (Path(result['workload']['tokenizer']).resolve() != Path(checkpoint.path).resolve()
            or result['workload']['seed'] != 42 + entry['repetition'] * 100):
        raise ValueError('pilot checkpoint or repetition does not match its workload')
    sessions = result['workload']['sessions']
    output = {8192: 512, 32768: 1024}[entry['context']]
    if (len(sessions) != 8 or any(s['initial_tokens'] != entry['context'] or len(s['turns']) != 4
            or any(t['max_tokens'] != output for t in s['turns']) for s in sessions)):
        raise ValueError('pilot has different session/context/output geometry')
    followups = [row for row in result['requests'] if row['turn'] > 0]
    if not followups or any(row['server_cached_prompt_tokens'] is None
                           or row['server_cached_prompt_tokens'] <= 0 for row in followups):
        raise ValueError('pilot followups lack observed prefix reuse')
    return {**entry, 'request_throughput_per_s': result['summary']['request_throughput_per_s'],
            'result_sha256': hashlib.sha256(result_path.read_bytes()).hexdigest(),
            'telemetry_sha256': hashlib.sha256(telemetry_path.read_bytes()).hexdigest(),
            'followup_cache_fraction': result['followup_turns']['cached_token_fraction_observed']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pilots', type=Path, required=True,
                        help='JSON list of checkpoint, execution_config, context, repetition, result and telemetry')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--exclusions', type=Path,
                        help='JSON map of omitted checkpoint/mode pairs to reasons')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('refusing to overwrite existing rate selection')
    from scripts.serving_benchmark.campaign import load_exclusions
    pairs = {f'{model.key}/{mode.key}' for model in CHECKPOINTS if model.performance
             for mode in EXECUTION_CONFIGS}
    exclusions = load_exclusions(args.exclusions, pairs)
    excluded_pairs = {tuple(pair.split('/')) for pair in exclusions}
    observations = [load_observation(row) for row in json.loads(args.pilots.read_text())]
    write_new(args.output, dict(schema='vosti.common-offered-rates.v1',
        rates=select_rates(observations, excluded_pairs=excluded_pairs), exclusions=exclusions,
        observations=observations, rule='0.5 and 0.8 times the lowest per-mode/context median of three C4 repeats',
        note='Same per-model rates across all modes and both contexts. Finite workload reference only; '
             'does not establish steady-state capacity. Retain actual causal delays and achieved throughput.'))


if __name__ == '__main__':
    main()
