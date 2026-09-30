"""Synthetic, closed-loop multi-turn serving through the common HTTP client.

Prepare once per tokenizer, then replay the same manifest on every engine.
Answers are generated normally and carried forward as text, not substituted.
"""

from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random
import socket
import sys
import time

import httpx

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.common.tokenizers import tokenizer_artifact_sha256
from scripts.serving_benchmark.protocol import arrival_offsets, summarize_requests
from scripts.serving_benchmark.run import read_server_metrics, send_request


SCHEMA = "vosti.synthetic-multi-turn.v1"
WORDS = "river forest stone garden cloud water mountain book paper window light field road tree lake".split()


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def write_new(path: Path, value: dict) -> None:
    content = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(content)


def exact_text(tokenizer, target: int, rng: random.Random, *, special: bool,
               prefix: str = "") -> str:
    """Make token-counted synthetic text, verifying the decode/encode round trip."""
    if target <= 0:
        raise ValueError("text token length must be positive")
    for _ in range(8):
        text = " ".join(rng.choices(WORDS, k=target * 2 + 32))
        ids = tokenizer.encode(text, add_special_tokens=False)
        take = min(target, len(ids))
        for _ in range(8):
            if not 0 < take <= len(ids):
                break
            candidate = prefix + tokenizer.decode(ids[:take], skip_special_tokens=False)
            actual = len(tokenizer.encode(candidate, add_special_tokens=special))
            if actual == target:
                return candidate
            take += target - actual
    raise ValueError(f"cannot construct {target} tokens for this tokenizer")


def prepare(tokenizer, *, sessions: int, turns: int, initial_tokens: int,
            suffix_tokens: int, output_tokens: int, think_seconds: float, seed: int,
            suffix_includes_separator: bool = False) -> dict:
    for value in (sessions, turns, initial_tokens, suffix_tokens, output_tokens):
        if type(value) is not int or value <= 0:
            raise ValueError("counts must be positive integers")
    if not math.isfinite(think_seconds) or think_seconds < 0:
        raise ValueError("think time must be finite and nonnegative")
    rng = random.Random(seed)
    conversations = []
    for session in range(sessions):
        initial = exact_text(tokenizer, initial_tokens, rng, special=True)
        conversation = []
        for turn in range(turns):
            # The separator is deliberately included in actual suffix accounting.
            if turn == 0:
                suffix = ""
            elif suffix_includes_separator:
                suffix = exact_text(tokenizer, suffix_tokens, rng, special=False, prefix="\n\n")
            else:
                suffix = "\n\n" + exact_text(tokenizer, suffix_tokens, rng, special=False)
            conversation.append({
                "suffix": suffix,
                "suffix_tokens": len(tokenizer.encode(suffix, add_special_tokens=False)),
                "max_tokens": output_tokens,
                "delay_s": 0.0 if turn == 0 else think_seconds,
            })
        conversations.append({"id": str(session), "initial_prompt": initial,
                              "initial_tokens": initial_tokens, "turns": conversation})
    if len({s["initial_prompt"] for s in conversations}) != sessions:
        raise ValueError("duplicate initial prompts; increase initial context length")
    return {"schema": SCHEMA, "seed": seed, "sessions": conversations,
            "description": "Synthetic raw completions; real answers carried forward as text. "
                           "Suffix counts include separators. No guaranteed cache-hit rate."}


def validate_workload(workload: dict, tokenizer, context_limit: int) -> None:
    if workload.get("schema") != SCHEMA or not workload.get("sessions"):
        raise ValueError("not a nonempty synthetic multi-turn workload")
    seen = set()
    for session in workload["sessions"]:
        if session["id"] in seen:
            raise ValueError("duplicate session ID")
        seen.add(session["id"])
        prompt = session["initial_prompt"]
        if not isinstance(prompt, str) or not prompt:
            raise ValueError("empty initial prompt")
        if len(tokenizer.encode(prompt)) != session["initial_tokens"]:
            raise ValueError("initial token count changed")
        if not session["turns"] or session["turns"][0]["suffix"] != "":
            raise ValueError("first turn must use the initial prompt without a suffix")
        nominal = session["initial_tokens"]
        for turn in session["turns"]:
            if not isinstance(turn["suffix"], str):
                raise ValueError("suffix must be text")
            count = len(tokenizer.encode(turn["suffix"], add_special_tokens=False))
            if count != turn["suffix_tokens"]:
                raise ValueError("suffix token count changed")
            if type(turn["max_tokens"]) is not int or turn["max_tokens"] <= 0:
                raise ValueError("invalid output budget")
            if not math.isfinite(turn["delay_s"]) or turn["delay_s"] < 0:
                raise ValueError("invalid think delay")
            nominal += count + turn["max_tokens"]
            if nominal > context_limit:
                raise ValueError("nominal conversation exceeds context limit; no truncation allowed")


def summarize_turns(records: list[dict], duration: float) -> dict:
    not_started = not records and duration == 0
    result = summarize_requests(records, 1.0 if not_started else duration)
    if not_started:
        result["duration_s"] = 0.0
        for field in ("request_throughput_per_s", "input_token_throughput_per_s",
                      "output_token_throughput_per_s", "total_token_throughput_per_s"):
            result[field] = None
    successful = [r for r in records if r["success"]]
    observed = [r for r in successful if r["server_cached_prompt_tokens"] is not None]
    denominator = sum(r["prompt_tokens"] for r in observed)
    result.update(
        cache_observed_requests=len(observed),
        cache_unknown_requests=len(successful) - len(observed),
        cached_token_fraction_observed=(
            sum(r["server_cached_prompt_tokens"] for r in observed) / denominator
            if denominator else None
        ),
        ttft_observed_requests=sum(r["ttft_s"] is not None for r in successful),
        # Denominator is always the whole lifecycle, not a claimed steady interval.
        throughput_scope="whole workload duration, including think time and drain",
    )
    if not_started:
        result["throughput_scope"] = "measurement not started"
    return result


def geometry_differences(left: dict, right: dict) -> list[dict]:
    def indexed(result):
        rows = {(r["session_id"], r["turn"]): r for r in result["requests"]}
        if len(rows) != len(result["requests"]):
            raise ValueError("duplicate turn records")
        return rows

    a, b = indexed(left), indexed(right)
    differences = []
    for key in sorted(a.keys() | b.keys()):
        if key not in a or key not in b:
            differences.append({"session_id": key[0], "turn": key[1], "missing_turn": True})
            continue
        changed = {field: [a[key][field], b[key][field]] for field in (
            "prompt_tokens", "requested_output_tokens", "output_tokens",
            "previous_input_common_prefix_tokens",
        ) if a[key][field] != b[key][field]}
        if changed:
            differences.append({"session_id": key[0], "turn": key[1], "differences": changed})
    return differences


def complete_result(result: dict) -> bool:
    return (result["complete"] and result["unattempted_turns"] == 0
            and len(result["requests"]) == result["planned_turns"]
            and all(row["success"] for row in result["requests"]))


def warmup_results(result):
    """Read ordered warmup evidence, including the original single-stage format."""
    if result.get("warmup_stages") is not None:
        if result.get("warmup") is not None:
            raise ValueError("ambiguous warmup evidence")
        return result["warmup_stages"]
    return [result["warmup"]] if result.get("warmup") is not None else []


def compare_results(left: dict, right: dict) -> dict:
    """Compare declared setup and actual geometry, never infer backend identity."""
    for result in (left, right):
        if result.get("schema") != "vosti.multi-turn-result.v1":
            raise ValueError("not a multi-turn result")
        if digest(result["workload"]) != result["workload_sha256"]:
            raise ValueError("result workload digest mismatch")
        if result.get("arrival_trace") is not None:
            validate_arrival_trace(result["arrival_trace"], result["workload"])
            if (result["offered_trace"] != result["arrival_trace"]["events"]
                    or any(result[key] != result["arrival_trace"][key]
                           for key in ("request_rate", "arrival_process", "arrival_seed"))):
                raise ValueError("result differs from saved arrival trace")
        for warmup in warmup_results(result):
            if digest(warmup["workload"]) != warmup["workload_sha256"]:
                raise ValueError("warmup workload digest mismatch")
    fields = ("workload_sha256", "concurrency", "context_limit", "stagger_seconds",
              "client_source_sha256")
    setup_differences = [field for field in fields if left[field] != right[field]]
    for field in ("request_rate", "arrival_process", "arrival_seed", "offered_trace"):
        if left.get(field) != right.get(field):
            setup_differences.append(field)
    warm_a, warm_b = warmup_results(left), warmup_results(right)
    if [r["workload_sha256"] for r in warm_a] != [r["workload_sha256"] for r in warm_b]:
        setup_differences.append("warmup_workload_sha256")
    warm_differences = [difference for a, b in zip(warm_a, warm_b)
                        for difference in geometry_differences(a, b)]
    differences = geometry_differences(left, right)
    complete = all(complete_result(r) and all(complete_result(w) for w in warmup_results(r))
                   for r in (left, right))
    return {
        "complete": complete, "setup_differences": setup_differences,
        "geometry_differences": differences,
        "warmup_geometry_differences": warm_differences,
        "matched_setup_and_geometry": complete and not setup_differences
                                      and not differences and not warm_differences,
        "left": {key: left[key] for key in ("engine_label", "summary", "cold_turns", "followup_turns")},
        "right": {key: right[key] for key in ("engine_label", "summary", "cold_turns", "followup_turns")},
        "note": "No timing normalization or speedup claim. Hardware, precision, weights, backend, "
                "warmup and contention evidence must also match. Cache counters are observations, "
                "not assumed equal; missing counters do not establish cache equivalence.",
    }


def turn_keys(workload: dict) -> list[tuple[str, int]]:
    sessions = workload.get("sessions")
    if workload.get("schema") != SCHEMA or not isinstance(sessions, list) or not sessions:
        raise ValueError("arrival trace requires a nonempty multi-turn workload")
    seen = set()
    for session in sessions:
        sid = session.get("id")
        if not isinstance(sid, str) or not sid or sid in seen or not session.get("turns"):
            raise ValueError("arrival trace requires unique session IDs and nonempty turns")
        seen.add(sid)
    return [(session["id"], turn)
            for turn in range(max(len(s["turns"]) for s in sessions))
            for session in sessions if turn < len(session["turns"])]


def offered_trace(workload: dict, *, request_rate: float | None, process: str,
                  seed: int, concurrency: int, stagger_seconds: float) -> list[dict]:
    """A turn-major offered trace fixed before any answers are generated.

    Causal dependencies can delay dispatch. The trace is not a promise that a
    finite set of sessions can sustain the requested rate.
    """
    if request_rate is None:
        return []
    sessions = workload["sessions"]
    if concurrency < len(sessions) or stagger_seconds != 0:
        raise ValueError("offered-rate mode needs a lane per session and zero stagger")
    keys = turn_keys(workload)
    offsets = arrival_offsets(len(keys), request_rate, seed, process)
    return [dict(session_id=session, turn=turn, offset_s=offset)
            for (session, turn), offset in zip(keys, offsets, strict=True)]


def prepare_arrival_trace(workload: dict, *, request_rate: float,
                          process: str = "poisson", seed: int = 42) -> dict:
    """Engine-independent artifact; saved offsets, not RNG regeneration, drive replay."""
    turn_keys(workload)
    events = offered_trace(workload, request_rate=request_rate, process=process,
                          seed=seed, concurrency=len(workload["sessions"]), stagger_seconds=0)
    result = dict(schema="vosti.multi-turn-arrival-trace.v1", workload_sha256=digest(workload),
                  request_rate=request_rate, arrival_process=process, arrival_seed=seed,
                  ordering="turn-major", first_arrival="zero", events=events)
    result["sha256"] = digest(result)
    validate_arrival_trace(result, workload)
    return result


def validate_arrival_trace(trace: dict, workload: dict) -> None:
    if trace.get("schema") != "vosti.multi-turn-arrival-trace.v1":
        raise ValueError("unknown arrival trace schema")
    if trace.get("sha256") != digest({k: v for k, v in trace.items() if k != "sha256"}):
        raise ValueError("arrival trace digest mismatch")
    if trace.get("workload_sha256") != digest(workload):
        raise ValueError("arrival trace workload mismatch")
    rate = trace.get("request_rate")
    if (type(rate) not in (int, float) or not math.isfinite(rate) or rate <= 0
            or type(trace.get("arrival_seed")) is not int
            or trace.get("arrival_process") not in ("poisson", "constant")
            or trace.get("ordering") != "turn-major" or trace.get("first_arrival") != "zero"):
        raise ValueError("invalid arrival trace settings")
    events, keys = trace.get("events"), turn_keys(workload)
    if not isinstance(events, list) or not keys or len(events) != len(keys):
        raise ValueError("arrival trace must cover every turn exactly once")
    previous = 0.0
    for index, (row, key) in enumerate(zip(events, keys, strict=True)):
        if (not isinstance(row, dict) or (row.get("session_id"), row.get("turn")) != key
                or type(row.get("turn")) is not int):
            raise ValueError("arrival trace session/turn ordering mismatch")
        offset = row.get("offset_s")
        if (type(offset) not in (int, float) or not math.isfinite(offset)
                or offset < previous or (index == 0 and offset != 0)):
            raise ValueError("invalid arrival trace offset")
        previous = offset


def resolve_arrivals(workload: dict, *, arrival_trace: dict | None = None,
                     request_rate: float | None = None, arrival_process: str | None = None,
                     arrival_seed: int | None = None, concurrency: int,
                     stagger_seconds: float) -> tuple[dict, dict | None]:
    settings = dict(request_rate=request_rate, arrival_process=arrival_process, arrival_seed=arrival_seed)
    if arrival_trace is not None:
        validate_arrival_trace(arrival_trace, workload)
        for key, value in settings.items():
            if value is not None and value != arrival_trace[key]:
                raise ValueError(f"{key} conflicts with saved arrival trace")
        settings = {key: arrival_trace[key] for key in settings}
    else:
        settings["arrival_process"] = arrival_process if arrival_process is not None else "poisson"
        settings["arrival_seed"] = arrival_seed if arrival_seed is not None else 42
        if request_rate is not None:
            arrival_trace = prepare_arrival_trace(workload, request_rate=request_rate,
                process=settings["arrival_process"], seed=settings["arrival_seed"])
    if arrival_trace is not None and (concurrency < len(workload["sessions"]) or stagger_seconds != 0):
        raise ValueError("offered-rate mode needs a lane per session and zero stagger")
    return settings, arrival_trace


async def execute_sessions(client, *, tokenizer, workload: dict, base_url: str,
                           model: str, concurrency: int, context_limit: int,
                           stagger_seconds: float = 0.0, request_rate: float | None = None,
                           arrival_process: str | None = None, arrival_seed: int | None = None,
                           arrival_trace: dict | None = None,
                           request_observer=None) -> dict:
    validate_workload(workload, tokenizer, context_limit)
    if concurrency <= 0 or not math.isfinite(stagger_seconds) or stagger_seconds < 0:
        raise ValueError("invalid concurrency or stagger")
    sessions = workload["sessions"]
    _, arrival_trace = resolve_arrivals(workload, request_rate=request_rate,
        arrival_process=arrival_process, arrival_seed=arrival_seed, arrival_trace=arrival_trace,
        concurrency=concurrency, stagger_seconds=stagger_seconds)
    trace = arrival_trace["events"] if arrival_trace is not None else []
    offered = {(row["session_id"], row["turn"]): row["offset_s"] for row in trace}
    records, failures = [], []
    tokenizer_lock = asyncio.Lock()
    started = time.perf_counter()

    async def encode(text, **kwargs):
        # Fast tokenizers can mutate padding/truncation settings during encode.
        # Serialize access to this instance without blocking the SSE event loop.
        async with tokenizer_lock:
            return await asyncio.to_thread(tokenizer.encode, text, **kwargs)

    async def lane(lane_index):
        await asyncio.sleep(lane_index * stagger_seconds)
        # Fixed session-to-lane assignment keeps the work manifest unchanged.
        for session in sessions[lane_index::concurrency]:
            history = session["initial_prompt"]
            nominal = session["initial_tokens"]
            previous_ids = []
            ready = started if trace else time.perf_counter()
            for turn_index, turn in enumerate(session["turns"]):
                prompt = history + turn["suffix"]
                # Long-context tokenization must not block other SSE readers.
                ids = await encode(prompt)
                nominal += turn["suffix_tokens"]
                if len(ids) + turn["max_tokens"] > context_limit:
                    failures.append({"session_id": session["id"], "turn": turn_index,
                                     "error": "actual context exceeds limit; session aborted",
                                     "prompt_tokens": len(ids)})
                    break
                # Count the retained previous-input prefix, not a supposed KV hit.
                common = 0
                for left, right in zip(previous_ids, ids):
                    if left != right:
                        break
                    common += 1
                dependency_ready = ready - started + turn["delay_s"]
                offered_offset = offered.get((session["id"], turn_index))
                scheduled = max(dependency_ready, offered_offset) if offered_offset is not None else dependency_ready
                row = await send_request(
                    client, base_url=base_url, endpoint="completions", model=model,
                    request={"prompt": prompt, "prompt_tokens": len(ids),
                             "max_tokens": turn["max_tokens"]},
                    index=turn_index, scheduled_offset_s=scheduled,
                    benchmark_start=started, retain_output=True, require_usage=True,
                )
                row.update(session_id=session["id"], turn=turn_index,
                           prompt_sha256=digest(prompt), prompt_token_ids_sha256=digest(ids),
                           nominal_prompt_tokens=nominal,
                           prompt_token_drift=len(ids) - nominal,
                           previous_input_tokens=len(previous_ids),
                           previous_input_common_prefix_tokens=common,
                           think_seconds=turn["delay_s"])
                if offered_offset is not None:
                    row.update(offered_offset_s=offered_offset,
                               dependency_ready_offset_s=dependency_ready,
                               dependency_delay_s=max(0.0, dependency_ready - offered_offset),
                               offered_to_send_s=row["send_started_offset_s"] - offered_offset,
                               offered_to_completion_s=row["completed_offset_s"] - offered_offset)
                row["retokenized_output_tokens"] = len(await encode(
                    row["generated_text"], add_special_tokens=False
                ))
                records.append(row)
                if request_observer is not None:
                    request_observer(row)
                if not row["success"]:
                    failures.append({"session_id": session["id"], "turn": turn_index,
                                     "error": row["error"]})
                    break
                history = prompt + row["generated_text"]
                nominal += row["output_tokens"]
                previous_ids = ids
                # Think time starts when the stream completes; client accounting
                # overhead is not silently added to the intended delay.
                ready = started + row["completed_offset_s"]

    await asyncio.gather(*(lane(i) for i in range(min(concurrency, len(sessions)))))
    duration = time.perf_counter() - started
    order = {s["id"]: i for i, s in enumerate(sessions)}
    records.sort(key=lambda row: (order[row["session_id"]], row["turn"]))
    planned = sum(len(s["turns"]) for s in sessions)
    return {
        "complete": not failures and len(records) == planned,
        "planned_turns": planned, "unattempted_turns": planned - len(records),
        "session_failures": failures,
        "summary": summarize_turns(records, duration),
        "cold_turns": summarize_turns([r for r in records if r["turn"] == 0], duration),
        "followup_turns": summarize_turns([r for r in records if r["turn"] > 0], duration),
        "requests": records,
        "offered_trace": trace,
    }


def validate_warmup(workload, warmup, tokenizer, context_limit):
    validate_workload(workload, tokenizer, context_limit)
    if isinstance(warmup, list):
        if not warmup:
            raise ValueError("warmup sequence must be nonempty")
        for stage in warmup:
            if not isinstance(stage, dict):
                raise ValueError("warmup stages must be workloads")
            validate_warmup(workload, stage, tokenizer, context_limit)
        return
    if warmup is None:
        return
    validate_workload(warmup, tokenizer, context_limit)
    if warmup.get("tokenizer_artifact_sha256") != workload.get("tokenizer_artifact_sha256"):
        raise ValueError("warmup tokenizer artifacts differ from measured workload")
    measured = {tuple(tokenizer.encode(s["initial_prompt"])) for s in workload["sessions"]}
    if any(tuple(tokenizer.encode(s["initial_prompt"])) in measured for s in warmup["sessions"]):
        raise ValueError("warmup repeats a measured initial prompt; use a distinct workload")


async def run(args, tokenizer, workload, warmup=None):
    # Validate both phases before even contacting the server.
    validate_warmup(workload, warmup, tokenizer, args.context_limit)
    trace_path = getattr(args, "arrival_trace", None)
    arrivals, trace_artifact = resolve_arrivals(workload,
        arrival_trace=json.loads(Path(trace_path).read_text()) if trace_path else None,
        request_rate=getattr(args, "request_rate", None),
        arrival_process=getattr(args, "arrival_process", None),
        arrival_seed=getattr(args, "arrival_seed", None),
        concurrency=args.concurrency, stagger_seconds=args.stagger_seconds)
    trace = trace_artifact["events"] if trace_artifact is not None else []
    headers = {"Authorization": f"Bearer {args.api_key}"} if args.api_key else {}
    async with httpx.AsyncClient(
        timeout=args.timeout, headers=headers,
        limits=httpx.Limits(max_connections=args.concurrency,
                           max_keepalive_connections=args.concurrency),
    ) as client:
        health = await client.get(f"{args.base_url}/health", timeout=10)
        health.raise_for_status()
        execution = dict(tokenizer=tokenizer, base_url=args.base_url,
            model=args.model, concurrency=args.concurrency, context_limit=args.context_limit,
            stagger_seconds=args.stagger_seconds)
        before_warmup = await read_server_metrics(client, args.base_url)
        stages = warmup if isinstance(warmup, list) else [warmup] if warmup is not None else []
        stage_results = []
        for stage in stages:
            stage_result = await execute_sessions(client, workload=stage, **execution)
            stage_result.update(workload=stage, workload_sha256=digest(stage))
            stage_results.append(stage_result)
            if not complete_result(stage_result):
                break
        warmup_result = stage_results[0] if stage_results and not isinstance(warmup, list) else None
        before = (await read_server_metrics(client, args.base_url)
                  if warmup is not None else before_warmup)
        if any(not complete_result(stage) for stage in stage_results):
            planned = sum(len(s["turns"]) for s in workload["sessions"])
            result = dict(complete=False, planned_turns=planned, unattempted_turns=planned,
                          session_failures=[{"error": "warmup failed; measurement not started"}],
                          requests=[], summary=summarize_turns([], 0),
                          cold_turns=summarize_turns([], 0), followup_turns=summarize_turns([], 0))
        else:
            result = await execute_sessions(client, workload=workload, **execution, **arrivals,
                                            arrival_trace=trace_artifact)
        after = await read_server_metrics(client, args.base_url)
    return {"schema": "vosti.multi-turn-result.v1",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "client_host": socket.gethostname(), "engine_label": args.engine_label,
            "model": args.model, "base_url": args.base_url,
            "workload_sha256": digest(workload), "workload": workload,
            "concurrency": args.concurrency, "stagger_seconds": args.stagger_seconds,
            "context_limit": args.context_limit,
            **arrivals, "offered_trace": trace, "arrival_trace": trace_artifact,
            "client_source_sha256": {name: hashlib.sha256(
                (Path(__file__).parent / name).read_bytes()).hexdigest()
                for name in ("multi_turn.py", "run.py", "protocol.py")},
            "timing_scope": "HTTP send to SSE completion; TTFT is first visible output event. "
                            "Lifecycle throughput includes client processing, think time and drain. "
                            "Optional warmup is excluded and retained separately. "
                            "Offered-rate traces are fixed; causal/think delays and realized send lag are recorded. "
                            "Cold means first turn, not verified cache miss; followup is not guaranteed hit.",
            "warmup": warmup_result,
            **({"warmup_stages": stage_results} if isinstance(warmup, list) else {}),
            "server_metrics_before_warmup": before_warmup,
            "server_metrics_before": before, "server_metrics_after": after, **result}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    generate = commands.add_parser("prepare")
    trace_parser = commands.add_parser("prepare-trace", help="Save a seeded offered-arrival trace; CPU only")
    trace_parser.add_argument("--workload", type=Path, required=True)
    trace_parser.add_argument("--request-rate", type=float, required=True)
    trace_parser.add_argument("--arrival-process", choices=("poisson", "constant"), default="poisson")
    trace_parser.add_argument("--arrival-seed", type=int, default=42)
    trace_parser.add_argument("--output", type=Path, required=True)
    execute = commands.add_parser("run")
    compare = commands.add_parser("compare")
    compare.add_argument("--left", type=Path, required=True)
    compare.add_argument("--right", type=Path, required=True)
    compare.add_argument("--output", type=Path, required=True)
    for child in (generate, execute):
        child.add_argument("--tokenizer", type=Path, required=True)
        child.add_argument("--output", type=Path, required=True)
    generate.add_argument("--sessions", type=int, default=16)
    generate.add_argument("--turns", type=int, default=4)
    generate.add_argument("--initial-tokens", type=int, default=8192)
    generate.add_argument("--suffix-tokens", type=int, default=256)
    generate.add_argument("--suffix-includes-separator", action="store_true",
                          help="Count the separator inside the requested suffix token budget")
    generate.add_argument("--output-tokens", type=int, default=1024)
    generate.add_argument("--think-seconds", type=float, default=0.5)
    generate.add_argument("--seed", type=int, default=42)
    execute.add_argument("--workload", type=Path, required=True)
    execute.add_argument("--warmup-workload", type=Path, action="append",
                         help="Excluded warmup workload; repeat to execute stages in order in the same server")
    execute.add_argument("--base-url", default="http://127.0.0.1:8000")
    execute.add_argument("--model", required=True)
    execute.add_argument("--engine-label", required=True)
    execute.add_argument("--concurrency", type=int, default=4)
    execute.add_argument("--context-limit", type=int, default=16384)
    execute.add_argument("--stagger-seconds", type=float, default=0.1)
    execute.add_argument("--request-rate", type=float,
                         help="Fixed offered request rate; requires --stagger-seconds 0 and a lane per session")
    execute.add_argument("--arrival-process", choices=("poisson", "constant"),
                         help="Default: saved trace setting, otherwise poisson")
    execute.add_argument("--arrival-seed", type=int, help="Default: saved trace setting, otherwise 42")
    execute.add_argument("--arrival-trace", type=Path,
                         help="Replay saved offsets; explicit arrival flags must agree with the artifact")
    execute.add_argument("--timeout", type=float, default=3600)
    execute.add_argument("--api-key")
    execute.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("refusing to overwrite existing artifact")
    if args.command == "prepare-trace":
        result = prepare_arrival_trace(json.loads(args.workload.read_text()),
            request_rate=args.request_rate, process=args.arrival_process, seed=args.arrival_seed)
        write_new(args.output, result)
        print(f"Saved {len(result['events'])} arrivals; sha256={result['sha256']}")
        return
    if args.command == "compare":
        result = compare_results(json.loads(args.left.read_text()), json.loads(args.right.read_text()))
        write_new(args.output, result)
        print(json.dumps({key: result[key] for key in (
            "complete", "matched_setup_and_geometry", "setup_differences", "geometry_differences",
            "warmup_geometry_differences"
        )}, indent=2))
        return
    from transformers import AutoTokenizer
    tokenizer_path = args.tokenizer.expanduser().resolve()
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    artifacts = tokenizer_artifact_sha256(tokenizer_path)
    if args.command == "prepare":
        result = prepare(tokenizer, sessions=args.sessions, turns=args.turns,
                         initial_tokens=args.initial_tokens, suffix_tokens=args.suffix_tokens,
                         output_tokens=args.output_tokens, think_seconds=args.think_seconds,
                         seed=args.seed, suffix_includes_separator=args.suffix_includes_separator)
        result.update(tokenizer=str(tokenizer_path), tokenizer_artifact_sha256=artifacts)
    else:
        if args.concurrency <= 0 or args.context_limit <= 0 or not math.isfinite(args.timeout) or args.timeout <= 0:
            parser.error("concurrency, context limit and timeout must be positive")
        if not math.isfinite(args.stagger_seconds) or args.stagger_seconds < 0:
            parser.error("stagger must be finite and nonnegative")
        workload = json.loads(args.workload.read_text())
        if workload["tokenizer_artifact_sha256"] != artifacts:
            parser.error("tokenizer artifacts differ from workload")
        stages = [json.loads(path.read_text()) for path in args.warmup_workload or []]
        warmup = stages if len(stages) > 1 else stages[0] if stages else None
        validate_warmup(workload, warmup, tokenizer, args.context_limit)
        resolve_arrivals(workload, request_rate=args.request_rate, arrival_process=args.arrival_process,
            arrival_seed=args.arrival_seed,
            arrival_trace=json.loads(args.arrival_trace.read_text()) if args.arrival_trace else None,
            concurrency=args.concurrency, stagger_seconds=args.stagger_seconds)
        if args.validate_only:
            print("Valid workload and tokenizer; no HTTP requests sent.")
            return
        args.base_url = args.base_url.rstrip("/")
        result = asyncio.run(run(args, tokenizer, workload, warmup))
    write_new(args.output, result)
    if args.command == "run":
        print(json.dumps(result["summary"], indent=2))
        if not result["complete"]:
            raise SystemExit("incomplete campaign; failures retained in result")


if __name__ == "__main__":
    main()
