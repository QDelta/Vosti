#!/usr/bin/env python3
"""Generate reproducible raw-ShareGPT workloads for cache-aware benchmarks.

The natural workload follows the common offline-serving shape: sample the
first user/assistant turn, keep raw (untemplated) prompts, cap completions at
512 tokens, and reject requests that exceed the 2,048-token context bound.

The prefix-pair workload models branching generation.  Each real ShareGPT
prompt is followed by two different answer-style suffixes.  Donors are placed
before consumers so a token-budget-limited scheduler can register the donor's
full prefix pages before admitting its paired consumer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import statistics
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kernels.triton_kernels.constants import PAGE_SIZE
from scripts.common.tokenizers import tokenizer_artifact_sha256


MIN_TOKENS = 4
MAX_PROMPT_TOKENS = 1024
MAX_OUTPUT_TOKENS = 512
MAX_CONTEXT_TOKENS = 2048


def percentile(values: list[int], fraction: float) -> int:
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * fraction)]


def distribution(values: list[int]) -> dict[str, float | int]:
    return {
        "mean": statistics.fmean(values),
        "p50": percentile(values, 0.50),
        "p90": percentile(values, 0.90),
        "min": min(values),
        "max": max(values),
        "total": sum(values),
    }


def first_turn(record: Any) -> tuple[str, str] | None:
    if not isinstance(record, dict):
        return None
    conversations = record.get("conversations")
    if not isinstance(conversations, list) or len(conversations) < 2:
        return None

    def text(row: Any) -> str | None:
        if not isinstance(row, dict):
            return None
        value = row.get("value", row.get("content"))
        return value if isinstance(value, str) and value.strip() else None

    prompt = text(conversations[0])
    completion = text(conversations[1])
    if prompt is None or completion is None:
        return None
    return prompt, completion


def eligible_rows(dataset: list[Any], tokenizer) -> list[dict[str, Any]]:
    candidates = []
    for source_index, record in enumerate(dataset):
        turn = first_turn(record)
        if turn is not None:
            candidates.append((source_index, *turn))

    rows = []
    for offset in range(0, len(candidates), 512):
        batch = candidates[offset : offset + 512]
        # We only need to distinguish an over-limit row from an in-range row.
        # Truncating one token past each limit avoids spending minutes on a few
        # pathological ShareGPT entries containing hundreds of thousands of
        # tokens while preserving the exact lengths of every eligible row.
        prompt_ids = tokenizer(
            [row[1] for row in batch],
            truncation=True,
            max_length=MAX_PROMPT_TOKENS + 1,
        )["input_ids"]
        completion_ids = tokenizer(
            [row[2] for row in batch],
            truncation=True,
            max_length=MAX_OUTPUT_TOKENS + 1,
        )["input_ids"]
        for (source_index, prompt, _), prompt_row, completion_row in zip(
            batch, prompt_ids, completion_ids
        ):
            prompt_len = len(prompt_row)
            completion_len = len(completion_row)
            output_len = min(completion_len, MAX_OUTPUT_TOKENS)
            if prompt_len < MIN_TOKENS or output_len < MIN_TOKENS:
                continue
            if prompt_len > MAX_PROMPT_TOKENS:
                continue
            if prompt_len + output_len > MAX_CONTEXT_TOKENS:
                continue
            rows.append(
                {
                    "source_index": source_index,
                    "prompt": prompt,
                    "prompt_tokens": prompt_len,
                    "max_tokens": output_len,
                }
            )
    return rows


def natural_sample(
    rows: list[dict[str, Any]], count: int, seed: int, excluded: set[int]
) -> list[dict[str, Any]]:
    candidates = [row for row in rows if row["source_index"] not in excluded]
    selected = random.Random(seed).sample(candidates, count)
    excluded.update(row["source_index"] for row in selected)
    return selected


def common_prefix_tokens(tokenizer, left: str, right: str) -> int:
    left_ids = tokenizer.encode(left)
    right_ids = tokenizer.encode(right)
    count = 0
    for left_id, right_id in zip(left_ids, right_ids):
        if left_id != right_id:
            break
        count += 1
    return count


def prefix_pair_sample(
    rows: list[dict[str, Any]], pair_count: int, seed: int, excluded: set[int], tokenizer
) -> tuple[list[dict[str, Any]], list[int]]:
    candidates = [
        row
        for row in rows
        if row["source_index"] not in excluded
        and 96 <= row["prompt_tokens"] <= MAX_PROMPT_TOKENS - 32
    ]
    selected = random.Random(seed).sample(candidates, pair_count)
    excluded.update(row["source_index"] for row in selected)
    donors = []
    consumers = []
    shared_blocks = []
    for row in selected:
        shared = row["prompt"].rstrip() + "\n\n"
        donor_prompt = shared + "Please provide a detailed answer."
        consumer_prompt = shared + "Please provide a concise answer."
        donor_len = len(tokenizer.encode(donor_prompt))
        consumer_len = len(tokenizer.encode(consumer_prompt))
        if max(donor_len, consumer_len) + row["max_tokens"] > MAX_CONTEXT_TOKENS:
            raise ValueError("prefix-pair request exceeds the context bound")
        common = common_prefix_tokens(tokenizer, donor_prompt, consumer_prompt)
        blocks = common // PAGE_SIZE
        if blocks == 0:
            raise ValueError("prefix-pair request has no complete shared page")
        shared_blocks.append(blocks)
        base = {
            "max_tokens": row["max_tokens"],
            "source_index": row["source_index"],
            "shared_prefix_tokens": common,
            "shared_prefix_blocks": blocks,
        }
        donors.append(
            {
                **base,
                "prompt": donor_prompt,
                "branch": "donor",
                "arrival_phase": 0,
            }
        )
        consumers.append(
            {
                **base,
                "prompt": consumer_prompt,
                "branch": "consumer",
                "arrival_phase": 1,
            }
        )
    return donors + consumers, shared_blocks


def request_payload(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    payload = []
    for row in rows:
        request = {"prompt": row["prompt"], "max_tokens": int(row["max_tokens"])}
        if "arrival_phase" in row:
            request["arrival_phase"] = int(row["arrival_phase"])
        payload.append(request)
    return payload


def metadata(
    kind: str,
    seed: int,
    rows: list[dict[str, Any]],
    tokenizer,
    dataset_sha256: str,
    tokenizer_sha256: dict[str, str],
    **extra: Any,
) -> dict[str, Any]:
    prompt_lengths = [len(tokenizer.encode(row["prompt"])) for row in rows]
    output_lengths = [int(row["max_tokens"]) for row in rows]
    return {
        "kind": kind,
        "seed": seed,
        "n_requests": len(rows),
        "tokenizer": tokenizer.name_or_path,
        "tokenizer_artifact_sha256": tokenizer_sha256,
        "dataset_sha256": dataset_sha256,
        "source_indices": [int(row["source_index"]) for row in rows],
        "prompt_len_dist": distribution(prompt_lengths),
        "output_len_dist": distribution(output_lengths),
        "total_output_tokens": sum(output_lengths),
        **extra,
    }


def write_workload(path: Path, rows: list[dict[str, Any]], meta: dict[str, Any]) -> None:
    path.write_text(
        json.dumps({"meta": meta, "requests": request_payload(rows)}, indent=2)
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument(
        "--tokenizer-path",
        dest="tokenizer_path",
        type=Path,
        required=True,
        help="local tokenizer directory",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--requests", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup-seed", type=int, default=43)
    args = parser.parse_args()
    if args.requests <= 0 or args.requests % 2:
        parser.error("--requests must be a positive even number")

    dataset_path = args.dataset.expanduser().resolve()
    tokenizer_path = args.tokenizer_path.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    raw = dataset_path.read_bytes()
    dataset_sha256 = hashlib.sha256(raw).hexdigest()
    dataset = json.loads(raw)
    if not isinstance(dataset, list):
        raise ValueError("ShareGPT dataset must be a JSON list")
    tokenizer_sha256 = tokenizer_artifact_sha256(tokenizer_path)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    rows = eligible_rows(dataset, tokenizer)
    excluded: set[int] = set()

    natural = natural_sample(rows, args.requests, args.seed, excluded)
    natural_warmup = natural_sample(rows, args.requests, args.warmup_seed, excluded)
    pairs, pair_blocks = prefix_pair_sample(
        rows, args.requests // 2, args.seed, excluded, tokenizer
    )
    pairs_warmup, warmup_pair_blocks = prefix_pair_sample(
        rows, args.requests // 2, args.warmup_seed, excluded, tokenizer
    )

    products = {
        f"sharegpt_seed{args.seed}.json": (
            natural,
            metadata(
                "sharegpt-natural",
                args.seed,
                natural,
                tokenizer,
                dataset_sha256,
                tokenizer_sha256,
            ),
        ),
        f"sharegpt_seed{args.warmup_seed}_warmup.json": (
            natural_warmup,
            metadata(
                "sharegpt-natural-warmup",
                args.warmup_seed,
                natural_warmup,
                tokenizer,
                dataset_sha256,
                tokenizer_sha256,
            ),
        ),
        f"sharegpt_prefix_pairs_seed{args.seed}.json": (
            pairs,
            metadata(
                "sharegpt-prefix-pairs",
                args.seed,
                pairs,
                tokenizer,
                dataset_sha256,
                tokenizer_sha256,
                pair_count=args.requests // 2,
                arrival_phases=[args.requests // 2, args.requests // 2],
                shared_prefix_blocks_total=sum(pair_blocks),
                shared_prefix_blocks_dist=distribution(pair_blocks),
            ),
        ),
        f"sharegpt_prefix_pairs_seed{args.warmup_seed}_warmup.json": (
            pairs_warmup,
            metadata(
                "sharegpt-prefix-pairs-warmup",
                args.warmup_seed,
                pairs_warmup,
                tokenizer,
                dataset_sha256,
                tokenizer_sha256,
                pair_count=args.requests // 2,
                arrival_phases=[args.requests // 2, args.requests // 2],
                shared_prefix_blocks_total=sum(warmup_pair_blocks),
                shared_prefix_blocks_dist=distribution(warmup_pair_blocks),
            ),
        ),
    }
    manifest = {
        "dataset": str(dataset_path),
        "dataset_sha256": dataset_sha256,
        "tokenizer_path": str(tokenizer_path),
        "tokenizer_artifact_sha256": tokenizer_sha256,
        "eligible_rows": len(rows),
        "page_size": PAGE_SIZE,
        "workloads": {},
    }
    for filename, (product_rows, meta) in products.items():
        path = output_dir / filename
        write_workload(path, product_rows, meta)
        manifest["workloads"][filename] = meta
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
