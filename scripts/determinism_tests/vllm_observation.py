"""Test-only recording of vLLM scheduler outputs without changing scheduling."""
import re


def public_schedule_events(call, version):
    """Associate retained internal IDs with public result IDs, without mutation.

    vLLM 0.28 InputProcessor.assign_request_id uses external_id + '-' +
    random_uuid()[:8]. The offline LLM worker uses unique decimal public IDs.
    Require an exact bijection; never guess from token content or row order.
    """
    public = [request['request_id'] for request in call['requests']]
    if len(set(public)) != len(public) or any(not isinstance(rid,str) or not rid.isdecimal() for rid in public):
        raise ValueError('vLLM report requires unique decimal public request IDs')
    internal = {row['request_id'] for event in call['schedule_events'] for row in event}
    mapping = {}
    for rid in internal:
        if rid in public:
            mapping[rid] = rid
            continue
        match = re.fullmatch(r'([0-9]+)-[0-9a-f]{8}', rid) if version == '0.28.0' else None
        if match is None or match[1] not in public:
            raise ValueError(f'unsupported or unknown vLLM internal request ID: {rid}')
        mapping[rid] = match[1]
    if len(mapping) != len(public) or set(mapping.values()) != set(public):
        raise ValueError('vLLM scheduler/public request mapping is not bijective')
    return [[dict(row, request_id=mapping[row['request_id']], scheduler_request_id=row['request_id'])
             for row in event] for event in call['schedule_events']]


def schedule_rows(output):
    prefix = {request.req_id: int(request.num_computed_tokens) for request in output.scheduled_new_reqs}
    cached = output.scheduled_cached_reqs
    prefix.update(zip(cached.req_ids, map(int, cached.num_computed_tokens), strict=True))
    if set(prefix) != set(output.num_scheduled_tokens):
        raise RuntimeError('vLLM scheduled IDs do not match their prefix metadata')
    rows = [dict(request_id=rid, prefix_tokens=prefix[rid], query_tokens=int(count))
            for rid, count in output.num_scheduled_tokens.items()]
    if any(row['prefix_tokens'] < 0 or row['query_tokens'] <= 0 for row in rows):
        raise RuntimeError('vLLM returned invalid schedule geometry')
    return rows


def observe_scheduler(llm):
    scheduler = llm.llm_engine.engine_core.engine_core.scheduler
    original = scheduler.schedule
    events = []

    def observed(*args, **kwargs):
        output = original(*args, **kwargs)
        events.append(schedule_rows(output))
        return output

    scheduler.schedule = observed
    return events
