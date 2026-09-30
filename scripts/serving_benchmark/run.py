#!/usr/bin/env python3
"""Drive an OpenAI-compatible server with a reproducible ShareGPT arrival trace."""

from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import socket
import sys
import time
from typing import Any

import httpx

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.serving_benchmark.protocol import (  # noqa: E402
    SseDecoder,
    arrival_offsets,
    parse_stream_payload,
    summarize_requests,
)


def load_workload(
    path: Path,
    tokenizer,
    max_requests: int | None,
    endpoint: str,
) -> tuple[list[dict], dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        requests = payload
        metadata = {}
    else:
        requests = payload.get("requests")
        metadata = payload.get("meta", {})
    if not isinstance(requests, list) or not requests:
        raise ValueError(f"workload contains no requests: {path}")
    if max_requests is not None:
        requests = requests[:max_requests]
    prepared = []
    for index, request in enumerate(requests):
        if not isinstance(request, dict):
            raise ValueError(f"request {index} is not an object")
        prompt = request.get("prompt")
        max_tokens = request.get("max_tokens")
        if not isinstance(prompt, str) or not prompt:
            raise ValueError(f"request {index} has no prompt")
        if type(max_tokens) is not int or max_tokens <= 0:
            raise ValueError(f"request {index} has invalid max_tokens")
        prompt_tokens = (
            tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=True,
                add_generation_prompt=True,
            )
            if endpoint == "chat"
            else tokenizer.encode(prompt)
        )
        prepared.append(
            {
                "prompt": prompt,
                "max_tokens": max_tokens,
                "prompt_tokens": len(prompt_tokens),
            }
        )
    return prepared, metadata


async def read_server_metrics(client: httpx.AsyncClient, base_url: str) -> dict | None:
    try:
        response = await client.get(f"{base_url}/metrics", timeout=10.0)
        if response.status_code != 200:
            return None
        value = response.json()
        return value if isinstance(value, dict) else None
    except (httpx.HTTPError, ValueError):
        return None


async def send_request(
    client: httpx.AsyncClient,
    *,
    base_url: str,
    endpoint: str,
    model: str,
    request: dict,
    index: int,
    scheduled_offset_s: float,
    benchmark_start: float,
    retain_output: bool = False,
    require_usage: bool = False,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "max_tokens": request["max_tokens"],
        "temperature": 0.0,
        "top_p": 1.0,
        "stream": True,
        "ignore_eos": True,
        "stream_options": {"include_usage": True},
    }
    if endpoint == "completions":
        url = f"{base_url}/v1/completions"
        payload["prompt"] = request["prompt"]
    else:
        url = f"{base_url}/v1/chat/completions"
        payload["messages"] = [{"role": "user", "content": request["prompt"]}]

    deadline = benchmark_start + scheduled_offset_s
    delay = deadline - time.perf_counter()
    if delay > 0:
        await asyncio.sleep(delay)
    send_started = time.perf_counter()

    output_event_times: list[float] = []
    reported_token_times: list[float] = []
    output_tokens_from_usage: int | None = None
    finish_reason: str | None = None
    status_code: int | None = None
    done = False
    error: str | None = None
    server_usage: dict[str, Any] = {}
    output_text: list[str] = []
    output_ids: list[int] = []

    def consume(data: str, received: float) -> None:
        nonlocal finish_reason, output_tokens_from_usage, done
        event = parse_stream_payload(data)
        usage = event.get("usage")
        if isinstance(usage, dict):
            server_usage.update(usage)
            if type(usage.get("completion_tokens")) is int:
                output_tokens_from_usage = usage["completion_tokens"]
        if event["kind"] == "output":
            output_event_times.append(received)
            reported_token_times.extend([received] * len(event["token_ids"]))
            if retain_output:
                output_text.append(event["text"])
                output_ids.extend(event["token_ids"])
        if event.get("finish_reason") is not None:
            finish_reason = str(event["finish_reason"])
        if event["kind"] == "error":
            raise RuntimeError(str(event.get("message")))
        if event["kind"] == "done":
            done = True

    try:
        async with client.stream("POST", url, json=payload) as response:
            status_code = response.status_code
            if response.status_code != 200:
                body = (await response.aread()).decode("utf-8", errors="replace")
                raise RuntimeError(f"HTTP {response.status_code}: {body[:1000]}")
            decoder = SseDecoder()
            async for line in response.aiter_lines():
                received = time.perf_counter()
                for data in decoder.feed_line(line):
                    consume(data, received)
                if done:
                    break
            for data in decoder.finish():
                consume(data, time.perf_counter())
    except (httpx.HTTPError, RuntimeError, ValueError, json.JSONDecodeError) as exception:
        error = repr(exception)
    completed = time.perf_counter()
    server_prompt_tokens = server_usage.get("prompt_tokens")
    details = server_usage.get("prompt_tokens_details")
    cached_tokens = details.get("cached_tokens") if isinstance(details, dict) else None
    if require_usage and error is None:
        if (
            output_tokens_from_usage is None
            or type(server_prompt_tokens) is not int
            or server_prompt_tokens != request["prompt_tokens"]
        ):
            error = "missing usage or server/client prompt-token count mismatch"
        elif cached_tokens is not None and (
            type(cached_tokens) is not int or not 0 <= cached_tokens <= server_prompt_tokens
        ):
            error = "invalid server cached-token count"

    output_tokens = (
        output_tokens_from_usage
        if output_tokens_from_usage is not None
        else (
            len(reported_token_times)
            if reported_token_times
            else len(output_event_times)
        )
    )
    output_token_count_source = (
        "usage"
        if output_tokens_from_usage is not None
        else (
            "reported_token_ids"
            if reported_token_times
            else "stream_output_events"
        )
    )
    success = (
        error is None
        and status_code == 200
        and done
        and (
            output_tokens_from_usage is not None
            or bool(reported_token_times)
            or bool(output_event_times)
        )
        and output_tokens == request["max_tokens"]
    )
    if not success and error is None:
        error = (
            "incomplete stream: "
            f"done={done}, output_events={len(output_event_times)}, "
            f"reported_token_ids={len(reported_token_times)}, "
            f"output_tokens={output_tokens}, "
            f"expected={request['max_tokens']}"
        )
    first_output = output_event_times[0] if output_event_times else None
    latency_s = completed - send_started
    ttft_s = first_output - send_started if first_output is not None else None
    inter_output_event = [
        right - left
        for left, right in zip(output_event_times, output_event_times[1:])
    ]
    tpot_s = (
        (completed - first_output) / (output_tokens - 1)
        if first_output is not None and output_tokens > 1
        else None
    )
    result = {
        "index": index,
        "success": success,
        "error": error,
        "status_code": status_code,
        "scheduled_offset_s": scheduled_offset_s,
        "send_started_offset_s": send_started - benchmark_start,
        "scheduling_lag_s": max(0.0, send_started - deadline),
        "first_output_event_offset_s": (
            first_output - benchmark_start if first_output is not None else None
        ),
        "last_output_event_offset_s": (
            output_event_times[-1] - benchmark_start if output_event_times else None
        ),
        "completed_offset_s": completed - benchmark_start,
        "latency_s": latency_s,
        "ttft_s": ttft_s,
        "tpot_s": tpot_s,
        "inter_output_event_latency_s": inter_output_event,
        "stream_output_events": len(output_event_times),
        "stream_reported_token_ids": len(reported_token_times),
        "visible_output_observed": bool(output_event_times),
        "prompt_tokens": request["prompt_tokens"],
        "requested_output_tokens": request["max_tokens"],
        "output_tokens": output_tokens,
        "output_token_count_source": output_token_count_source,
        "finish_reason": finish_reason,
        "server_usage": server_usage,
        "server_prompt_tokens": server_prompt_tokens,
        "server_cached_prompt_tokens": cached_tokens,
    }
    if retain_output:
        result.update(generated_text="".join(output_text), generated_token_ids=output_ids)
    return result


async def execute_workload(
    client: httpx.AsyncClient,
    *,
    base_url: str,
    endpoint: str,
    model: str,
    requests: list[dict],
    request_rate: float,
    arrival_process: str,
    seed: int,
) -> tuple[list[dict], float, list[float]]:
    offsets = arrival_offsets(len(requests), request_rate, seed, arrival_process)
    benchmark_start = time.perf_counter()
    tasks = [
        asyncio.create_task(
            send_request(
                client,
                base_url=base_url,
                endpoint=endpoint,
                model=model,
                request=request,
                index=index,
                scheduled_offset_s=offsets[index],
                benchmark_start=benchmark_start,
            )
        )
        for index, request in enumerate(requests)
    ]
    records = await asyncio.gather(*tasks)
    duration_s = time.perf_counter() - benchmark_start
    return list(records), duration_s, offsets


async def async_main(args: argparse.Namespace) -> dict:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    measured, workload_metadata = load_workload(
        args.workload, tokenizer, args.max_requests, args.endpoint
    )
    warmup = None
    warmup_metadata = None
    if args.warmup_workload is not None:
        warmup, warmup_metadata = load_workload(
            args.warmup_workload,
            tokenizer,
            args.max_warmup_requests,
            args.endpoint,
        )
        measured_prompts = {request["prompt"] for request in measured}
        if any(request["prompt"] in measured_prompts for request in warmup):
            raise ValueError("warmup and measured workloads contain identical prompts")
    del tokenizer

    timeout = httpx.Timeout(args.timeout)
    limits = httpx.Limits(
        max_connections=args.max_connections,
        max_keepalive_connections=args.max_connections,
    )
    headers = {"Authorization": f"Bearer {args.api_key}"} if args.api_key else {}
    async with httpx.AsyncClient(timeout=timeout, limits=limits, headers=headers) as client:
        health = await client.get(f"{args.base_url}/health", timeout=10.0)
        health.raise_for_status()
        warmup_result = None
        if warmup is not None:
            print(f"warmup: {len(warmup)} requests at {args.warmup_request_rate} req/s")
            records, duration_s, offsets = await execute_workload(
                client,
                base_url=args.base_url,
                endpoint=args.endpoint,
                model=args.model,
                requests=warmup,
                request_rate=args.warmup_request_rate,
                arrival_process=args.arrival_process,
                seed=args.seed + 1,
            )
            warmup_result = {
                "duration_s": duration_s,
                "summary": summarize_requests(records, duration_s),
                "arrival_offsets_s": offsets,
                "metadata": warmup_metadata,
                "requests": records,
            }
            if args.settle_seconds:
                await asyncio.sleep(args.settle_seconds)

        metrics_before = await read_server_metrics(client, args.base_url)
        print(
            f"measured: {len(measured)} requests at {args.request_rate} req/s "
            f"({args.arrival_process})"
        )
        records, duration_s, offsets = await execute_workload(
            client,
            base_url=args.base_url,
            endpoint=args.endpoint,
            model=args.model,
            requests=measured,
            request_rate=args.request_rate,
            arrival_process=args.arrival_process,
            seed=args.seed,
        )
        metrics_after = await read_server_metrics(client, args.base_url)

    summary = summarize_requests(records, duration_s)
    return {
        "schema": "vosti.openai-serving-benchmark.v2",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "client_host": socket.gethostname(),
        "client_platform": platform.platform(),
        "declared_engine_label": args.engine_label,
        "base_url": args.base_url,
        "endpoint": args.endpoint,
        "model": args.model,
        "tokenizer": str(args.tokenizer),
        "workload": {
            "path": str(args.workload),
            "sha256": hashlib.sha256(args.workload.read_bytes()).hexdigest(),
            "metadata": workload_metadata,
            "requests": len(measured),
        },
        "arrival": {
            "process": args.arrival_process,
            "request_rate_per_s": args.request_rate,
            "seed": args.seed,
            "offsets_s": offsets,
        },
        "timing_scope": {
            "client_request_construction": "excluded",
            "client_scheduling_delay": "excluded from per-request latency",
            "network_and_server_queueing": "included",
            "server_tokenization": "included",
            "server_detokenization": "included",
            "time_to_first_token": (
                "request send start to first visible output SSE event"
            ),
            "inter_output_event_latency": (
                "between visible output SSE events; an event may represent "
                "zero, one, or multiple generated tokens"
            ),
            "end_to_end": "request send start to completed SSE stream",
        },
        "warmup": warmup_result,
        "server_metrics_before": metrics_before,
        "server_metrics_after": metrics_after,
        "summary": summary,
        "requests": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--engine-label", required=True)
    parser.add_argument("--endpoint", choices=("completions", "chat"), default="completions")
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--workload", type=Path, required=True)
    parser.add_argument("--warmup-workload", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--request-rate", type=float, required=True)
    parser.add_argument("--warmup-request-rate", type=float)
    parser.add_argument("--arrival-process", choices=("poisson", "constant"), default="poisson")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-requests", type=int)
    parser.add_argument("--max-warmup-requests", type=int)
    parser.add_argument("--max-connections", type=int, default=1024)
    parser.add_argument("--timeout", type=float, default=3600.0)
    parser.add_argument("--settle-seconds", type=float, default=1.0)
    parser.add_argument("--api-key")
    args = parser.parse_args()
    args.base_url = args.base_url.rstrip("/")
    args.tokenizer = args.tokenizer.expanduser().resolve()
    args.workload = args.workload.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    if args.warmup_workload is not None:
        args.warmup_workload = args.warmup_workload.expanduser().resolve()
    if args.warmup_request_rate is None:
        args.warmup_request_rate = args.request_rate
    if args.request_rate <= 0 or args.warmup_request_rate <= 0:
        parser.error("request rates must be positive")
    if args.max_connections <= 0 or args.timeout <= 0 or args.settle_seconds < 0:
        parser.error("connection/timeout limits must be positive and settle time nonnegative")
    if args.max_requests is not None and args.max_requests <= 0:
        parser.error("--max-requests must be positive")
    if args.max_warmup_requests is not None and args.max_warmup_requests <= 0:
        parser.error("--max-warmup-requests must be positive")
    for path in (args.tokenizer, args.workload):
        if not path.exists():
            parser.error(f"path does not exist: {path}")
    if args.warmup_workload is not None and not args.warmup_workload.is_file():
        parser.error(f"warmup workload does not exist: {args.warmup_workload}")
    if args.output.exists():
        parser.error(f"refusing to overwrite benchmark artifact: {args.output}")

    result = asyncio.run(async_main(args))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result["summary"], indent=2, sort_keys=True))
    warmup_failures = (
        result["warmup"]["summary"]["failed_requests"]
        if result["warmup"] is not None
        else 0
    )
    if warmup_failures or result["summary"]["failed_requests"]:
        raise SystemExit("benchmark completed with failed requests; inspect the retained artifact")


if __name__ == "__main__":
    main()
