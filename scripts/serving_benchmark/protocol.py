"""Pure arrival-schedule, SSE, and metric helpers for serving benchmarks."""

from __future__ import annotations

import json
import math
import random
from typing import Any, Iterable


def arrival_offsets(
    count: int,
    request_rate: float,
    seed: int,
    process: str,
) -> list[float]:
    """Return deterministic open-loop send offsets in seconds."""

    if count <= 0:
        raise ValueError("arrival schedule requires at least one request")
    if not math.isfinite(request_rate) or request_rate <= 0:
        raise ValueError("request rate must be positive and finite")
    if process not in {"poisson", "constant"}:
        raise ValueError(f"unknown arrival process {process!r}")

    rng = random.Random(seed)
    offsets = [0.0]
    for _ in range(1, count):
        interval = (
            rng.expovariate(request_rate)
            if process == "poisson"
            else 1.0 / request_rate
        )
        offsets.append(offsets[-1] + interval)
    return offsets


class SseDecoder:
    """Decode SSE data fields from the line stream returned by HTTP clients."""

    def __init__(self) -> None:
        self._data: list[str] = []

    def feed_line(self, line: str) -> list[str]:
        if line == "":
            if not self._data:
                return []
            data = "\n".join(self._data)
            self._data.clear()
            return [data]
        if line.startswith(":"):
            return []
        if line.startswith("data:"):
            self._data.append(line.removeprefix("data:").lstrip())
        return []

    def finish(self) -> list[str]:
        if not self._data:
            return []
        data = "\n".join(self._data)
        self._data.clear()
        return [data]


def parse_stream_payload(data: str) -> dict[str, Any]:
    """Classify one OpenAI-compatible SSE data payload."""

    if data == "[DONE]":
        return {"kind": "done"}
    payload = json.loads(data)
    if not isinstance(payload, dict):
        raise ValueError("stream payload is not a JSON object")
    if "error" in payload:
        error = payload["error"]
        message = error.get("message") if isinstance(error, dict) else str(error)
        return {"kind": "error", "message": message}

    usage = payload.get("usage")
    choices = payload.get("choices")
    if not choices:
        if isinstance(usage, dict):
            return {"kind": "usage", "usage": usage}
        return {"kind": "metadata"}
    if not isinstance(choices, list) or not isinstance(choices[0], dict):
        raise ValueError("stream choices are malformed")
    choice = choices[0]
    finish_reason = choice.get("finish_reason")
    if "text" in choice:
        text = choice.get("text")
    else:
        delta = choice.get("delta")
        if not isinstance(delta, dict) or "content" not in delta:
            text = None
        else:
            text = delta.get("content")
    if text is not None and not isinstance(text, str):
        raise ValueError("stream output content is not a string")

    raw_token_ids = choice.get("token_ids")
    if raw_token_ids is None:
        raw_token_id = choice.get("token_id")
        token_ids = [] if raw_token_id is None else [raw_token_id]
    else:
        if not isinstance(raw_token_ids, list):
            raise ValueError("stream token_ids is not a list")
        token_ids = raw_token_ids
    if any(type(token_id) is not int for token_id in token_ids):
        raise ValueError("stream token IDs are not integers")

    # A terminal chunk may also carry its final output delta. Empty,
    # nonterminal chunks are metadata unless token IDs make their output
    # semantics explicit.
    has_output = bool(token_ids) or bool(text)
    if has_output:
        return {
            "kind": "output",
            "text": text or "",
            "token_ids": token_ids,
            "finish_reason": finish_reason,
            "usage": usage if isinstance(usage, dict) else None,
        }
    if finish_reason is not None:
        return {
            "kind": "finish",
            "finish_reason": finish_reason,
            "usage": usage if isinstance(usage, dict) else None,
        }
    return {"kind": "metadata", **({"usage": usage} if isinstance(usage, dict) else {})}


def percentile(values: Iterable[float], fraction: float) -> float | None:
    values = sorted(float(value) for value in values)
    if not values:
        return None
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("percentile fraction must lie in [0, 1]")
    position = (len(values) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return values[lower]
    weight = position - lower
    return values[lower] * (1.0 - weight) + values[upper] * weight


def latency_distribution_ms(values_s: Iterable[float]) -> dict[str, float | None]:
    values_ms = [float(value) * 1000.0 for value in values_s]
    return {
        "mean": sum(values_ms) / len(values_ms) if values_ms else None,
        "p50": percentile(values_ms, 0.50),
        "p90": percentile(values_ms, 0.90),
        "p99": percentile(values_ms, 0.99),
    }


def summarize_requests(records: list[dict[str, Any]], duration_s: float) -> dict[str, Any]:
    if duration_s <= 0:
        raise ValueError("benchmark duration must be positive")
    successful = [record for record in records if record["success"]]
    failed = [record for record in records if not record["success"]]
    input_tokens = sum(int(record["prompt_tokens"]) for record in successful)
    output_tokens = sum(int(record["output_tokens"]) for record in successful)
    inter_output_event = [
        interval
        for record in successful
        for interval in record["inter_output_event_latency_s"]
    ]
    return {
        "duration_s": duration_s,
        "requests": len(records),
        "successful_requests": len(successful),
        "failed_requests": len(failed),
        "request_throughput_per_s": len(successful) / duration_s,
        "input_token_throughput_per_s": input_tokens / duration_s,
        "output_token_throughput_per_s": output_tokens / duration_s,
        "total_token_throughput_per_s": (input_tokens + output_tokens) / duration_s,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "end_to_end_latency_ms": latency_distribution_ms(
            record["latency_s"] for record in successful
        ),
        "time_to_first_token_ms": latency_distribution_ms(
            record["ttft_s"]
            for record in successful
            if record["ttft_s"] is not None
        ),
        "time_per_output_token_ms": latency_distribution_ms(
            record["tpot_s"]
            for record in successful
            if record["tpot_s"] is not None
        ),
        "inter_output_event_latency_ms": latency_distribution_ms(inter_output_event),
        "client_scheduling_lag_ms": latency_distribution_ms(
            record["scheduling_lag_s"] for record in records
        ),
    }
