"""Opt-in request identity at SGLang's sampling boundary, test harness only."""
from contextvars import ContextVar


SAMPLING_REQUESTS = ContextVar('vosti_sglang_sampling_requests', default=None)


def request_metadata(forward_batch):
    ids = forward_batch.rids
    if not ids or any(not isinstance(value, str) or not value for value in ids):
        raise RuntimeError('SGLang forward batch has no usable request IDs')
    if len(set(ids)) != len(ids):
        raise RuntimeError('SGLang forward batch repeats a request ID')
    rows = [dict(request_id=value, forward_mode=str(forward_batch.forward_mode)) for value in ids]
    for attribute, field in (('extend_seq_lens_cpu', 'query_tokens'),
                             ('extend_prefix_lens_cpu', 'prefix_tokens')):
        values = getattr(forward_batch, attribute, None)
        if values is not None:
            if len(values) < len(rows):
                raise RuntimeError('SGLang forward lengths do not cover request IDs')
            for index, row in enumerate(rows):
                row[field] = int(values[index])
    return rows
