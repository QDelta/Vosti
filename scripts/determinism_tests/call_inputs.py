"""Construct prediction inputs and resolve followups from earlier engine calls."""
from __future__ import annotations


def teacher_prefixes(prompt: list[int], generated: list[int]) -> list[list[int]]:
    if not prompt or not generated:
        raise ValueError('teacher forcing requires a prompt and generated tokens')
    # Prediction i consumes only outputs before i, never the target token itself.
    return [prompt + generated[:position] for position in range(len(generated))]


def resolve_call(call, completed):
    source = call.get('output_prefix_from')
    if source is None:
        return call
    if set(source) != {'call', 'request', 'count', 'suffix'}:
        raise ValueError('generated-prefix source requires call, request, count and suffix')
    ci, ri, count = (source[key] for key in ('call', 'request', 'count'))
    if any(type(value) is not int or value < 0 for value in (ci, ri, count)):
        raise ValueError('generated-prefix indices/count must be nonnegative integers')
    if ci >= len(completed) or ri >= len(completed[ci]['requests']):
        raise ValueError('generated-prefix source must be an already completed request')
    request = completed[ci]['requests'][ri]
    if count > len(request['output_token_ids']):
        raise ValueError('generated-prefix count exceeds the observed generation')
    suffix = source['suffix']
    if not isinstance(suffix, list) or any(type(token) is not int or token < 0 for token in suffix):
        raise ValueError('generated-prefix suffix must be nonnegative token IDs')
    return dict(call, prompts=[request['input_token_ids'] + request['output_token_ids'][:count] + suffix])
