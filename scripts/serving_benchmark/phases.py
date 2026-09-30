"""Token-counted prefill, decode and cached-extension HTTP measurements.

Prepare manifests once and replay unchanged across engines. Donor calls and
disjoint warmup are outside measured waves. This measures serving latency, not
isolated GPU kernels; concurrency does not guarantee a physical batch size.
"""
from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import random
import socket
import time

import httpx

from scripts.common.tokenizers import tokenizer_artifact_sha256
from scripts.serving_benchmark.multi_turn import (
    digest, exact_text, summarize_turns, write_new,
)
from scripts.serving_benchmark.run import read_server_metrics, send_request


SCHEMA = "vosti.phase-workload.v1"


def prepare(tokenizer, *, kind: str, context: int, query: int, output: int,
            concurrency: int, waves: int, seed: int) -> dict:
    if kind not in {"cold_prefill", "decode", "cached_extension"}:
        raise ValueError("unknown phase kind")
    if any(type(v) is not int or v <= 0 for v in (output, concurrency, waves)):
        raise ValueError("output, concurrency and waves must be positive integers")
    if type(context) is not int or type(query) is not int or min(context, query) < 0:
        raise ValueError("context and query must be nonnegative integers")
    if ((kind == "cold_prefill" and (context != 0 or query <= 0 or output != 1))
            or (kind == "decode" and (context <= 0 or query != 0 or output <= 1))
            or (kind == "cached_extension" and (context <= 0 or query <= 0 or output != 1))):
        raise ValueError("invalid geometry for phase kind")
    rng = random.Random(seed)
    phases = {}
    for name, count in (("warmup", 1), ("measured", waves)):
        rows = []
        for index in range(count * concurrency):
            donor = None
            if kind == "cached_extension":
                donor = exact_text(tokenizer, context, rng, special=True)
                prompt = exact_text(tokenizer, context + query, rng,
                                    special=True, prefix=donor + "\n\n")
                if tokenizer.encode(prompt)[:context] != tokenizer.encode(donor):
                    raise ValueError("extension changed the donor token prefix")
            else:
                prompt = exact_text(tokenizer, context + query, rng, special=True)
            rows.append(dict(index=index, prompt=prompt, prompt_tokens=context + query,
                             max_tokens=output, donor=donor))
        phases[name] = rows
    result = dict(schema=SCHEMA, kind=kind, context_tokens=context, query_tokens=query,
                  output_tokens=output, concurrency=concurrency, waves=waves, seed=seed,
                  **phases)
    validate(result, tokenizer, context_limit=context + query + output)
    return result


def validate(data: dict, tokenizer, context_limit: int) -> None:
    if data.get("schema") != SCHEMA:
        raise ValueError("unknown phase workload schema")
    kind = data["kind"]
    context, query, output = (data[k] for k in ("context_tokens", "query_tokens", "output_tokens"))
    if kind not in {"cold_prefill", "decode", "cached_extension"}:
        raise ValueError("unknown phase kind")
    if any(type(data[k]) is not int or data[k] <= 0
           for k in ("concurrency", "waves", "output_tokens")):
        raise ValueError("invalid workload counts")
    if any(type(v) is not int or v < 0 for v in (context, query)):
        raise ValueError("invalid context or query")
    if ((kind == "cold_prefill" and (context != 0 or query <= 0 or output != 1))
            or (kind == "decode" and (context <= 0 or query != 0 or output <= 1))
            or (kind == "cached_extension" and (context <= 0 or query <= 0 or output != 1))):
        raise ValueError("invalid geometry for phase kind")
    prefixes = set()
    for name, waves in (("warmup", 1), ("measured", data["waves"])):
        if len(data[name]) != waves * data["concurrency"]:
            raise ValueError("wrong number of wave requests")
        for index, row in enumerate(data[name]):
            if (row["index"] != index or row["max_tokens"] != output
                    or row["prompt_tokens"] != context + query):
                raise ValueError("inconsistent request metadata")
            ids = tokenizer.encode(row["prompt"])
            if len(ids) != context + query or len(ids) + output > context_limit:
                raise ValueError("prompt geometry changed or context limit exceeded")
            # Reject even short accidental common prefixes in prepared phases.
            # Actual runtime cache counters remain authoritative.
            prefix = tuple(ids[:min(16, len(ids))])
            if prefix in prefixes:
                raise ValueError("warmup/measured prompts share an initial token prefix")
            prefixes.add(prefix)
            if kind == "cached_extension":
                donor = tokenizer.encode(row["donor"])
                if len(donor) != context or donor != ids[:context]:
                    raise ValueError("donor geometry or token prefix changed")
            elif row["donor"] is not None:
                raise ValueError("unexpected donor")


def cache_evidence(record: dict, kind: str, nominal_context: int) -> dict:
    cached = record["server_cached_prompt_tokens"]
    if not record["success"]:
        status = "request_failed"
    elif cached is None:
        status = "unknown"
    elif kind == "cached_extension":
        status = ("observed_hit" if 0 < cached <= nominal_context
                  else "missing_hit" if cached == 0 else "unexpected_extra_reuse")
    else:
        status = "observed_cold" if cached == 0 else "unexpected_reuse"
    return dict(status=status, nominal_cached_tokens=nominal_context if kind == "cached_extension" else 0,
                observed_cached_tokens=cached,
                realized_uncached_prompt_tokens=(record["prompt_tokens"] - cached
                                                  if cached is not None else None),
                nominal_prefix_coverage=(cached / nominal_context
                    if kind == "cached_extension" and cached is not None else None))


async def execute_phase(client, *, data: dict, name: str, base_url: str, model: str) -> dict:
    records, donors, wave_results = [], [], []
    duration = 0.0
    concurrency = data["concurrency"]

    async def wave(rows):
        start = time.perf_counter()
        result = await asyncio.gather(*(send_request(
            client, base_url=base_url, endpoint="completions", model=model,
            request=row, index=row["index"], scheduled_offset_s=0,
            benchmark_start=start, require_usage=True) for row in rows))
        return result, time.perf_counter() - start

    for offset in range(0, len(data[name]), concurrency):
        rows = data[name][offset:offset + concurrency]
        wave_index = offset // concurrency
        if data["kind"] == "cached_extension":
            donor_rows = [dict(prompt=row["donor"], prompt_tokens=data["context_tokens"],
                               max_tokens=1, index=row["index"]) for row in rows]
            donor_records, donor_duration = await wave(donor_rows)
            donors.append(dict(wave=wave_index, requests=donor_records, duration_s=donor_duration))
            if not all(r["success"] for r in donor_records):
                break
        before = await read_server_metrics(client, base_url)
        measured, elapsed = await wave(rows)
        after = await read_server_metrics(client, base_url)
        for record in measured:
            record.update(wave=wave_index, cache_evidence=cache_evidence(
                record, data["kind"], data["context_tokens"]))
        records.extend(measured)
        duration += elapsed
        wave_results.append(dict(wave=wave_index, duration_s=elapsed,
                                 server_metrics_before=before, server_metrics_after=after))
        if not all(r["success"] for r in measured):
            break
    complete = (len(records) == len(data[name]) and all(r["success"] for r in records))
    summary = summarize_turns(records, duration)
    summary["throughput_scope"] = "sum of measured wave durations; excludes donor, warmup and inter-wave gaps"
    valid_cache = all(r["cache_evidence"]["status"] in {"observed_hit", "observed_cold"} for r in records)
    return dict(complete=complete, cache_geometry_observed=complete and valid_cache,
                planned_requests=len(data[name]), unattempted_requests=len(data[name]) - len(records),
                requests=records, donors=donors, waves=wave_results, summary=summary)


async def run(args, data: dict) -> dict:
    headers = {"Authorization": f"Bearer {args.api_key}"} if args.api_key else None
    async with httpx.AsyncClient(timeout=args.timeout, headers=headers,
            limits=httpx.Limits(max_connections=data["concurrency"] + 4)) as client:
        (await client.get(f"{args.base_url}/health")).raise_for_status()
        warmup = await execute_phase(client, data=data, name="warmup", base_url=args.base_url, model=args.model)
        measured = (await execute_phase(client, data=data, name="measured", base_url=args.base_url, model=args.model)
                    if warmup["complete"] else None)
    return dict(schema="vosti.phase-result.v1", engine_label=args.engine_label,
                created_utc=datetime.now(timezone.utc).isoformat(), host=socket.gethostname(),
                model=args.model, context_limit=args.context_limit,
                workload=data, workload_sha256=digest(data), warmup=warmup, measured=measured,
                complete=bool(measured and measured["complete"]),
                client_source_sha256={name: digest((Path(__file__).parent / name).read_text())
                    for name in ("phases.py", "multi_turn.py", "run.py", "protocol.py")},
                note="Client TTFT includes serving overhead; decode TPOT excludes initial TTFT. "
                     "SSE events need not be individual tokens. Cached context is nominal: "
                     "full-page caching and last-logit recomputation can reduce observed reuse. "
                     "Compare realized uncached prompt counts before claiming matched cache geometry.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    generate, execute = commands.add_parser("prepare"), commands.add_parser("run")
    for command in (generate, execute):
        command.add_argument("--tokenizer", type=Path, required=True)
        command.add_argument("--output", type=Path, required=True)
    generate.add_argument("--kind", choices=("cold_prefill", "decode", "cached_extension"), required=True)
    generate.add_argument("--context", type=int, default=0)
    generate.add_argument("--query", type=int, default=0)
    generate.add_argument("--output-tokens", type=int, default=1)
    generate.add_argument("--concurrency", type=int, default=1)
    generate.add_argument("--waves", type=int, default=4)
    generate.add_argument("--seed", type=int, default=42)
    execute.add_argument("--workload", type=Path, required=True)
    execute.add_argument("--base-url", default="http://127.0.0.1:8000")
    execute.add_argument("--model", required=True)
    execute.add_argument("--engine-label", required=True)
    execute.add_argument("--context-limit", type=int, default=40960)
    execute.add_argument("--timeout", type=float, default=3600)
    execute.add_argument("--api-key")
    execute.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("refusing to overwrite existing artifact")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    artifacts = tokenizer_artifact_sha256(args.tokenizer)
    if args.command == "prepare":
        result = prepare(tokenizer, kind=args.kind, context=args.context, query=args.query,
            output=args.output_tokens, concurrency=args.concurrency, waves=args.waves, seed=args.seed)
        result.update(tokenizer=str(args.tokenizer.resolve()), tokenizer_artifact_sha256=artifacts)
    else:
        data = json.loads(args.workload.read_text())
        if not math.isfinite(args.timeout) or args.timeout <= 0 or args.context_limit <= 0:
            parser.error("timeout and context limit must be positive and finite")
        if data["tokenizer_artifact_sha256"] != artifacts:
            parser.error("tokenizer artifacts differ from workload")
        validate(data, tokenizer, args.context_limit)
        if args.validate_only:
            print("Valid workload and tokenizer; no HTTP requests sent.")
            return
        args.base_url = args.base_url.rstrip("/")
        result = asyncio.run(run(args, data))
    write_new(args.output, result)
    if args.command == "run" and not result["complete"]:
        raise SystemExit("incomplete measurement; evidence retained")


if __name__ == "__main__":
    main()
