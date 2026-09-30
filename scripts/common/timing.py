"""Client-visible inference accounting; these are not GPU kernel timings."""


def interval_union_s(intervals):
    total, end = 0.0, float('-inf')
    for start, stop in sorted(intervals):
        if stop < start:
            raise ValueError('negative interval')
        total += max(0.0, stop-max(start, end))
        end = max(end, stop)
    return total


def inference_accounting(records, campaign_s):
    prompt = sum(r['usage']['prompt_tokens'] for r in records)
    output = sum(r['usage']['completion_tokens'] for r in records)
    cached = [r['usage'].get('prompt_tokens_details', {}).get('cached_tokens',
              r['usage'].get('cached_tokens')) for r in records]
    cache_known = all(v is not None for v in cached)
    uncached = prompt-sum(cached) if cache_known else None
    visible = [r for r in records if r['ttft_s'] is not None]
    ttft = sum(r['ttft_s'] for r in visible)
    generation = sum(r['last_content_s']-r['ttft_s'] for r in visible)
    tail = sum(r['request_s']-r['last_content_s'] for r in visible)
    unknown = sum(r['request_s'] for r in records if r['ttft_s'] is None)
    total = sum(r['request_s'] for r in records)
    assert abs(total-(ttft+generation+tail+unknown)) < 1e-6*max(total, 1)
    timestamps_known = all('request_started_monotonic_s' in r for r in records)
    active = interval_union_s([(r['request_started_monotonic_s'], r['request_finished_monotonic_s'])
                              for r in records]) if timestamps_known else None
    # The first output is produced by prefill, not a subsequent decode step.
    decode_estimate = sum(max(0, r['usage']['completion_tokens']-1) for r in records)
    visible_decode = sum(max(0, r['usage']['completion_tokens']-1) for r in visible)
    visible_uncached = sum(r['usage']['prompt_tokens']-c for r,c in zip(records,cached)
                           if r['ttft_s'] is not None) if cache_known else None
    return dict(completed_calls=len(records), visible_timing_calls=len(visible),
        prompt_tokens=prompt, cached_prompt_tokens=sum(cached) if cache_known else None,
        uncached_prompt_tokens=uncached, output_tokens=output,
        subsequent_decode_tokens_estimate=decode_estimate,
        model_request_sum_s=total, model_request_active_wall_s=active,
        time_to_first_text_sum_s=ttft, generation_span_sum_s=generation,
        stream_tail_sum_s=tail, no_visible_text_request_sum_s=unknown,
        generation_fraction_of_visible_wait=generation/(ttft+generation) if ttft+generation else None,
        uncached_prompt_tokens_per_campaign_s=uncached/campaign_s if uncached is not None else None,
        output_tokens_per_campaign_s=output/campaign_s,
        output_tokens_per_active_request_wall_s=output/active if active else None,
        prefill_client_proxy_tokens_s=visible_uncached/ttft if visible_uncached is not None and ttft else None,
        decode_client_proxy_tokens_s=visible_decode/generation if generation else None,
        gpu_only_inference_s=None,
        note='Request sums overlap across sessions. TTFT includes queueing and first-token production; generation includes scheduling and transport. Decode token count subtracts one output per call, not instrumented kernel work. Missing cache counts remain unknown.')
