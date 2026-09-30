"""Concrete one-seed workloads for the 14-row determinism report table."""
from dataclasses import asdict
import json
from pathlib import Path
import random

from scripts.determinism_tests.protocol import NATIVE_CHECKPOINTS as CHECKPOINTS
from scripts.determinism_tests.protocol import EXECUTION_CONFIGS, MODELS, deterministic_prompt


SEED = 1592598566
REPORT_MODELS = ('llama3-8b', 'gemma3-4b')
BATCH_LENGTHS = (17, 31, 32, 33, 63, 64, 65, 127, 128, 129, 255, 256, 257,
                 511, 512, 513, 767, 1023, 1024, 1025, 1535, 2047, 2048, 2049,
                 3071, 4095, 4096, 4097, 6143, 8191, 8192, 8193)
CHUNK_BUDGETS = (64, 128, 256, 512, 1024)
PD_LENGTHS = (257, 8192, 32768)
OUTPUT_TOKENS = 128


def compositions():
    shuffled = list(range(32))
    random.Random(SEED + 9000).shuffle(shuffled)
    return [order[offset:offset + size] for size in (2, 4, 8)
            for order in (list(range(32)), list(reversed(range(32))), shuffled)
            for offset in range(0, 32, size)]


def lengths(window=None):
    values = {2048, 8192, 32768}
    for boundary in (*CHUNK_BUDGETS, *((window,) if window else ())):
        values.update((boundary - 1, boundary, boundary + 1))
    return sorted(values)


def table_matrix():
    return [dict(checkpoint=model, execution=mode.key, seed=SEED,
                 relations=['batch', 'chunk', 'pd', 'cache'])
            for model in REPORT_MODELS for mode in EXECUTION_CONFIGS]


def make_plan(checkpoint_key):
    if checkpoint_key not in {checkpoint.key for checkpoint in CHECKPOINTS}:
        raise ValueError('checkpoint is outside the supported checkpoint catalog')
    checkpoint = next(value for value in CHECKPOINTS if value.key == checkpoint_key)
    model = next(value for value in MODELS if value.key == checkpoint.model)
    path = Path(checkpoint.path)
    config = json.loads((path / 'config.json').read_text())
    text = config.get('text_config', config)
    window = text.get('sliding_window') if text.get('use_sliding_window', True) else None
    if window is not None and (type(window) is not int or window < 2):
        raise ValueError('invalid checkpoint sliding window')
    def prompt(length, offset):
        return deterministic_prompt(path, length=length, seed=SEED + offset)
    chunk_lengths = lengths(window)
    cache = []
    donors = sorted({15, 16, 17, 127, 128, 129, 8192,
                     *((window - 1, window, window + 1) if window else (1023, 1024, 1025))})
    for index, count in enumerate(donors):
        query = prompt(count + 128, 3000 + index)
        cache.append(dict(label=f'partial-{count}', donor=query[:count], query=query,
                          prefix_limit=count, require_positive=count >= 128))
    for index, count in enumerate((129, 1025)):
        query = prompt(count, 4000 + index)
        cache.append(dict(label=f'full-{count}', donor=query, query=query,
                          prefix_limit=count, require_positive=True))
    shared = prompt(1024, 5000)
    left = prompt(128, 5001)[1:] + prompt(1, 5001)
    right = prompt(128, 5002)[1:] + prompt(1, 5002)
    if left == right:
        raise RuntimeError('divergent suffix construction failed')
    cache.append(dict(label='divergent-suffix', donor=shared + left, query=shared + right,
                      prefix_limit=1024, require_positive=True))
    return dict(profile='report-one-seed-v1', checkpoint=checkpoint_key, model=asdict(model),
        model_path=str(path), seed=SEED, window=window,
        batch_prompts=[prompt(length, i) for i, length in enumerate(BATCH_LENGTHS)],
        batch_groups=compositions(), fresh_spots=[0, 10, 21, 31],
        chunk_prompts=[prompt(length, 1000 + i) for i, length in enumerate(chunk_lengths)],
        chunk_budgets=list(CHUNK_BUDGETS), full_budget=max(chunk_lengths),
        pd_prompts=[prompt(length, 2000 + i) for i, length in enumerate(PD_LENGTHS)],
        output_tokens=OUTPUT_TOKENS, cache_cases=cache,
        generated_donor=prompt(1024, 6000), generated_suffix=prompt(128, 6001))
