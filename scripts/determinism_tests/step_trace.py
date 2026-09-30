"""Associate sampled rows with requests under non-rectangular native schedules."""
from __future__ import annotations


def index_emissions(records: list[dict], steps: list[dict]) -> dict[int, list[dict]]:
    """Check every scheduled row, retaining only rows which actually emitted.

    Chunk-only steps can have logits without emitting a token. Request IDs and
    row order come from the native driver, not from argmax matching or an assumed
    fixed batch size. A future dynamic-arrival driver can use the same format.
    """
    by_step = {}
    for record in records:
        key = record.get('engine_step')
        if not key:
            raise ValueError('logit row lacks native engine-step identity')
        index = record['batch_index']
        group = by_step.setdefault(key, {})
        if type(index) is not int or index < 0 or index in group:
            raise ValueError('ambiguous logit row index within native step')
        group[index] = record
    seen = set()
    last_steps = {}
    outputs = {}
    for step in steps:
        key = step['engine_step']
        base, number = step['request_id_base'], step['step']
        if (type(base) is not int or base < 0 or type(number) is not int or number < 0
                or key in seen or key != f'{base}:{number}'
                or number <= last_steps.get(base, -1)):
            raise ValueError('duplicate or inconsistent native step identity')
        seen.add(key)
        last_steps[base] = number
        scheduled, emitted = step['scheduled'], step['emitted']
        if (len(scheduled) != len(set(scheduled)) or len(emitted) != len(scheduled)
                or len(step['cached_prefix_blocks']) != len(scheduled)
                or any(type(rid) is not int or rid < step['request_id_base'] for rid in scheduled)):
            raise ValueError('invalid scheduled request metadata')
        rows = by_step.pop(key, {})
        if set(rows) != set(range(len(scheduled))):
            raise ValueError('scheduled requests and observed rows do not match')
        for index, (rid, token) in enumerate(zip(scheduled, emitted)):
            if token is None:
                continue
            if type(token) is not int or token < 0 or rows[index]['argmax'] != token:
                raise ValueError('emitted token disagrees with observed logit argmax')
            outputs.setdefault(rid, []).append(rows[index])
    if by_step:
        raise ValueError('logit rows refer to unrecorded native steps')
    return outputs


def select_emission_rows(indexed: dict[int, list[dict]], *, request_id_base: int,
                         batch_size: int, max_tokens: int, position: int) -> list[dict]:
    if not 0 <= position < max_tokens:
        raise ValueError('retained generated-token position is out of bounds')
    sequences = [indexed.get(request_id_base + i, []) for i in range(batch_size)]
    if any(len(sequence) != max_tokens for sequence in sequences):
        raise ValueError('native trace has missing or excess request emissions')
    return [sequence[position] for sequence in sequences]


def validate_arrivals(steps: list[dict], *, request_id_base: int, arrival_steps: list[int]) -> None:
    expected = {}
    for index, step in enumerate(arrival_steps):
        if type(step) is not int or step < 0:
            raise ValueError('invalid declared arrival step')
        expected.setdefault(step, []).append(request_id_base + index)
    actual = {step['step']: step['arrived'] for step in steps
              if step['request_id_base'] == request_id_base and step.get('arrived')}
    if actual != expected:
        raise ValueError('native arrivals do not match the declared request schedule')
