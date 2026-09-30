"""Export detailed task and request metrics alongside official task outcomes."""
import csv
import json
from pathlib import Path

from .prepare import save, sha
from .run import distribution
from scripts.common.timing import inference_accounting


def export_csv(path, rows):
    with path.open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]) if rows else [])
        writer.writeheader()
        writer.writerows(rows)


def render(root):
    plan = json.loads((root/'plan.json').read_text())
    destination = root/'summary'
    destination.mkdir(exist_ok=False)
    tasks, requests, modes = [], [], []
    for mode in plan['modes']:
        trial = root/'trials'/mode
        summary = json.loads((trial/'summary.json').read_text())
        receipt = json.loads((root/'grading'/f'{mode}-receipt.json').read_text())
        grade = json.loads((root/'grading/reports'/f'{mode}.{receipt["run_id"]}.json').read_text())
        assert summary['attempted'] == len(json.loads((root/'instances.json').read_text()))
        cached_total, prompt_total, cache_known = 0, 0, True
        for task in summary['task_results']:
            iid = task['instance_id']
            tasks.append(dict(mode=mode, instance_id=iid, resolved=iid in grade['resolved_ids'],
                exit_status=task['exit_status'], infrastructure_error=task['error'] is not None,
                task_wall_s=task['task_wall_s'], model_request_s=task['model_request_s'], tool_s=task['tool_s'],
                model_calls=task['calls'], generated_tokens=task['generated_tokens']))
            for number, request in enumerate(task['requests'],1):
                usage = request['usage']
                cached = usage.get('prompt_tokens_details', {}).get('cached_tokens', usage.get('cached_tokens'))
                cache_known &= cached is not None
                cached_total += cached or 0
                prompt_total += usage['prompt_tokens']
                requests.append(dict(mode=mode, instance_id=iid, call=number,
                    prompt_tokens=usage['prompt_tokens'], completion_tokens=usage['completion_tokens'],
                    cached_prompt_tokens=cached, ttft_ms=request['ttft_s']*1000 if request['ttft_s'] is not None else None,
                    tpot_estimate_ms=request['tpot_estimate_s']*1000 if request['tpot_estimate_s'] is not None else None,
                    request_latency_s=request['request_s'], finish_reason=request['finish_reason']))
        accounting = inference_accounting([r for t in summary['task_results'] for r in t['requests']],
                                           summary['measured_campaign_s'])
        modes.append(dict(mode=mode, resolved=grade['resolved_instances'], attempted=summary['attempted'],
            inference=accounting,
            initial_requests=inference_accounting([t['requests'][0] for t in summary['task_results'] if t['requests']],
                                                   summary['measured_campaign_s']),
            followup_requests=inference_accounting([r for t in summary['task_results'] for r in t['requests'][1:]],
                                                    summary['measured_campaign_s']),
            initial_ttft_ms=distribution([r['ttft_ms'] for r in requests if r['mode']==mode and r['call']==1]),
            followup_ttft_ms=distribution([r['ttft_ms'] for r in requests if r['mode']==mode and r['call']>1]),
            tool_s=sum(t['tool_s'] for t in summary['task_results']),
            infrastructure_errors=summary['infrastructure_errors'],
            grading_errors=grade.get('error_instances'), campaign_s=summary['measured_campaign_s'],
            resolved_per_hour=grade['resolved_instances']*3600/summary['measured_campaign_s'],
            ttft_ms=distribution([r['ttft_ms'] for r in requests if r['mode']==mode]),
            tpot_estimate_ms=distribution([r['tpot_estimate_ms'] for r in requests if r['mode']==mode]),
            request_latency_s=summary['request_latency_s'], task_wall_s=summary['task_wall_s'],
            output_tokens_s=summary['output_tokens_per_campaign_s'],
            input_tokens_s=summary['input_tokens_per_campaign_s'],
            cached_prompt_fraction=cached_total/prompt_total if cache_known and prompt_total else None,
            generated_tokens=summary['total_generated_tokens']))
    export_csv(destination/'requests.csv', requests)
    export_csv(destination/'tasks.csv', tasks)
    save(destination/'measurements.json', dict(modes=modes, requests=requests, tasks=tasks))
    text = ['# SWE-bench Lite live-agent pilot', '',
        f'{plan["model"]} bf16, {plan["concurrency"]} concurrent agent session(s), '
        f'the same seeded {len(json.loads((root/"instances.json").read_text()))} issues per mode.',
        'One attempt per issue; resolved means the official evaluator passed the required tests, not merely agent submission.',
        'Different generations change tool actions, token lengths, and task trajectories. This is not a fixed-workload speedup measurement.', '',
        '| Mode | Resolved | Campaign s | Task wall p50/p95 s | TTFT p50/p95 ms | TPOT estimate p50/p95 ms | Output tok/s | Cached prompt fraction |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for row in modes:
        def pair(key):
            v=row[key]
            return f'{v["p50"]:.2f} / {v["p95"]:.2f}' if v else 'unavailable'
        hit = f'{row["cached_prompt_fraction"]:.1%}' if row['cached_prompt_fraction'] is not None else 'unavailable'
        text.append(f'| {row["mode"]} | {row["resolved"]}/{row["attempted"]} | {row["campaign_s"]:.1f} | {pair("task_wall_s")} | {pair("ttft_ms")} | {pair("tpot_estimate_ms")} | {row["output_tokens_s"]:.1f} | {hit} |')
    text += ['', '| Mode | Initial TTFT p50/p95 ms | Follow-up TTFT p50/p95 ms | Tool time s | Infrastructure errors |',
        '|---|---:|---:|---:|---:|']
    for row in modes:
        text.append(f'| {row["mode"]} | {pair("initial_ttft_ms")} | {pair("followup_ttft_ms")} | {row["tool_s"]:.2f} | {row["infrastructure_errors"]} |')
    text += ['', '## Inference time and token accounting', '',
        '| Mode | Active request wall s | Sum request s | Sum TTFT s | Sum generation s | Uncached prompt tokens | Cached prompt tokens | Output tokens | Generation share |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    def fmt(value):
        return 'unavailable' if value is None else f'{value:,.2f}'
    for row in modes:
        a = row['inference']
        keys = ['model_request_active_wall_s','model_request_sum_s','time_to_first_text_sum_s',
                'generation_span_sum_s','uncached_prompt_tokens','cached_prompt_tokens','output_tokens']
        text.append('| '+row['mode']+' | '+' | '.join(fmt(a[k]) for k in keys)+
                    ' | '+fmt(a['generation_fraction_of_visible_wait']*100 if a['generation_fraction_of_visible_wait'] is not None else None)+'% |')
    text += ['', '| Mode | Uncached prompt tok/campaign s | Output tok/campaign s | Output tok/active-request s | Prefill client proxy tok/s | Decode client proxy tok/s |',
        '|---|---:|---:|---:|---:|---:|']
    for row in modes:
        a = row['inference']
        text.append('| '+row['mode']+' | '+' | '.join(fmt(a[k]) for k in [
            'uncached_prompt_tokens_per_campaign_s','output_tokens_per_campaign_s',
            'output_tokens_per_active_request_wall_s','prefill_client_proxy_tokens_s','decode_client_proxy_tokens_s'])+' |')
    text += ['', 'Active request wall time is the union of client request intervals, excluding gaps with no outstanding request. Summed request times overlap across sessions and are not GPU busy time.',
        'Prefill client proxy is uncached prompt tokens / summed TTFT. Decode client proxy is subsequent output tokens / summed first-to-last-text time. These are per-request service-rate estimates, not aggregate GPU phase throughput. The first output token is attributed to prefill; EOS-only calls have no visible phase timing.',
        'Generation share is generation span / (TTFT + generation span), excluding stream tails and calls without visible text. Raw server metrics are preserved separately; no comparable GPU-only timer is available across all engines.',
        f'Difficulty filter: {plan.get("difficulty")!r}; seeded selection is independent of model outcomes.',
        '', 'TTFT is client time to first nonempty streamed text. TPOT is the first-to-last text interval divided by output tokens minus one; transport chunks are not individual GPU tokens.',
        'Task timing includes model calls and tools; model startup, environment setup, warmup, patch extraction, and final grading are excluded. Campaign time includes dispatch gaps and result persistence.',
        'Output throughput counts all generated tokens, including unsuccessful tasks; input throughput counts cached prompt tokens too. No failed attempts are dropped.',
        'This pilot is too small to establish score equivalence or general correctness. Per-task outcomes, grading errors, raw events, server counters and GPU telemetry remain available.', '',
        '[Per-request metrics](requests.csv) | [Per-task outcomes and metrics](tasks.csv) | [Full measurements](measurements.json)', '']
    with (destination/'REPORT.md').open('x') as stream:
        stream.write('\n'.join(text))
    save(destination/'sha256.json', {str(f):sha(f) for f in destination.iterdir() if f.is_file()})
    print('REPORT READY', destination/'REPORT.md', flush=True)
