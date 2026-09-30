#!/usr/bin/env python3
"""Find the first differing entry in two logit-ranked token sequences."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.determinism_tests.protocol import sha256_json  # noqa: E402


SCHEMA_VERSION = 2


def ranking_divergence(lhs: np.ndarray, rhs: np.ndarray) -> dict[str, Any]:
    """Compare descending ``(token_id, logit)`` sequences exactly."""

    if lhs.ndim != 1 or rhs.ndim != 1 or lhs.shape != rhs.shape:
        raise ValueError("rank divergence requires equal one-dimensional rows")
    if lhs.dtype != np.float32 or rhs.dtype != np.float32:
        raise ValueError("rank divergence requires float32 rows")
    if not np.isfinite(lhs).all() or not np.isfinite(rhs).all():
        raise ValueError("rank divergence requires finite rows")

    token_ids = np.arange(lhs.size, dtype=np.int64)
    # Descending logit is the primary order; token ID makes exact ties stable.
    lhs_order = np.lexsort((token_ids, -lhs))
    rhs_order = np.lexsort((token_ids, -rhs))
    lhs_logits = lhs[lhs_order]
    rhs_logits = rhs[rhs_order]
    token_differs = lhs_order != rhs_order
    # This analyzer reports exact float32-logit divergence, not merely numeric
    # inequality. IEEE +0.0 and -0.0 therefore differ.
    lhs_logit_bits = np.ascontiguousarray(lhs_logits).view(np.uint32)
    rhs_logit_bits = np.ascontiguousarray(rhs_logits).view(np.uint32)
    logit_differs = lhs_logit_bits != rhs_logit_bits
    differing = np.flatnonzero(token_differs | logit_differs)

    result: dict[str, Any] = {
        "vocab_size": int(lhs.size),
        "ordered_pairs_identical": not bool(differing.size),
        "first_divergent_rank": None,
        "divergence_reason": None,
        "token_differs_at_first_divergence": False,
        "top_token_differs": False,
        "tie_affected": False,
    }
    if not differing.size:
        return result

    offset = int(differing[0])
    lhs_token = int(lhs_order[offset])
    rhs_token = int(rhs_order[offset])
    token_difference = bool(token_differs[offset])
    logit_difference = bool(logit_differs[offset])
    if token_difference and logit_difference:
        reason = "token_and_logit"
    elif token_difference:
        reason = "token_only"
    else:
        reason = "logit_only"
    lhs_logit = lhs_logits[offset]
    rhs_logit = rhs_logits[offset]
    lhs_tie_count = int(np.count_nonzero(lhs == lhs_logit))
    rhs_tie_count = int(np.count_nonzero(rhs == rhs_logit))
    result.update(
        {
            "first_divergent_rank": offset + 1,
            "divergence_reason": reason,
            "token_differs_at_first_divergence": token_difference,
            "top_token_differs": offset == 0 and token_difference,
            "left_token_id": lhs_token,
            "right_token_id": rhs_token,
            "left_rank_logit": float(lhs_logit),
            "right_rank_logit": float(rhs_logit),
            "left_token_logit_in_right": float(rhs[lhs_token]),
            "right_token_logit_in_left": float(lhs[rhs_token]),
            "left_rank_tie_count": lhs_tie_count,
            "right_rank_tie_count": rhs_tie_count,
            "tie_affected": lhs_tie_count > 1 or rhs_tie_count > 1,
        }
    )
    return result


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _recorded_rows(summary: dict[str, Any]) -> dict[Path, dict[str, Any]]:
    records: dict[Path, dict[str, Any]] = {}
    for arm in summary["arms"]:
        result_path = Path(arm["result"]).resolve()
        result = json.loads(result_path.read_text(encoding="utf-8"))
        for call in result["calls"]:
            for request in call["requests"]:
                record = request.get("last_row")
                if record is None:
                    continue
                path = (result_path.parent / "rows" / record["artifact"]).resolve()
                prior = records.get(path)
                if prior is not None and prior != record:
                    raise RuntimeError(f"conflicting row records for {path}")
                records[path] = record
    return records


def _load_verified_row(
    path: Path,
    records: dict[Path, dict[str, Any]],
    cache: dict[Path, np.ndarray],
) -> np.ndarray:
    resolved = path.resolve()
    if resolved in cache:
        return cache[resolved]
    record = records.get(resolved)
    if record is None:
        raise RuntimeError(f"comparison row is absent from retained results: {resolved}")
    row = np.load(resolved, allow_pickle=False)
    if list(row.shape) != record["shape"] or str(row.dtype) != record["dtype"]:
        raise RuntimeError(f"row metadata differs from retained artifact: {resolved}")
    digest = hashlib.sha256(row.tobytes(order="C")).hexdigest()
    if digest != record["sha256"]:
        raise RuntimeError(f"row hash differs from retained record: {resolved}")
    if row.dtype != np.float32 or row.ndim != 1 or not np.isfinite(row).all():
        raise RuntimeError(f"rank analysis requires a finite float32 row: {resolved}")
    cache[resolved] = row
    return row


def analyze_summary(label: str, summary_path: Path) -> dict[str, Any]:
    summary_path = summary_path.expanduser().resolve()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    records = _recorded_rows(summary)
    cache: dict[Path, np.ndarray] = {}
    comparisons = []
    for relation, relation_comparisons in summary["comparisons"].items():
        for comparison_index, comparison in enumerate(relation_comparisons):
            left_path = Path(comparison["left"]).resolve()
            right_path = Path(comparison["right"]).resolve()
            left = _load_verified_row(left_path, records, cache)
            right = _load_verified_row(right_path, records, cache)
            ranked = ranking_divergence(left, right)
            recorded_bitwise_equal = bool(comparison["bitwise_equal"])
            rescored_bitwise_equal = bool(ranked["ordered_pairs_identical"])
            ranked.update(
                {
                    "relation": relation,
                    "comparison_index": comparison_index,
                    "left": str(left_path),
                    "right": str(right_path),
                    "source_recorded_bitwise_equal": recorded_bitwise_equal,
                    "rescored_bitwise_equal": rescored_bitwise_equal,
                    "recorded_bitwise_equal_matches_rescore": (
                        recorded_bitwise_equal == rescored_bitwise_equal
                    ),
                    "source_recorded_argmax_equal": bool(
                        comparison["argmax_equal"]
                    ),
                }
            )
            comparisons.append(ranked)

    model_paths = set()
    for arm in summary["arms"]:
        result = json.loads(Path(arm["result"]).read_text(encoding="utf-8"))
        model_paths.add(str(Path(result["model_path"]).resolve()))
    if len(model_paths) != 1:
        raise RuntimeError(f"summary spans multiple model paths: {summary_path}")
    return {
        "label": label,
        "summary": str(summary_path),
        "summary_sha256": _file_sha256(summary_path),
        "hardware": summary["hardware"],
        "model_key": summary["model"]["key"],
        "model_path": model_paths.pop(),
        "execution": summary["execution"],
        "input_sha256": summary["inputs_sha256"],
        "comparison_count": len(comparisons),
        "ordered_pairs_identical_count": sum(
            row["ordered_pairs_identical"] for row in comparisons
        ),
        "top_token_difference_count": sum(
            row["top_token_differs"] for row in comparisons
        ),
        "comparisons": comparisons,
    }


def _parse_summary(value: str) -> tuple[str, Path]:
    label, separator, raw_path = value.partition("=")
    if not separator or not label or not raw_path:
        raise argparse.ArgumentTypeError("summary must be LABEL=PATH")
    return label, Path(raw_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--summary",
        action="append",
        required=True,
        type=_parse_summary,
        metavar="LABEL=PATH",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    labels = [label for label, _path in args.summary]
    if len(labels) != len(set(labels)):
        raise RuntimeError("summary labels must be unique")
    analyses = [analyze_summary(label, path) for label, path in args.summary]
    comparisons = [row for analysis in analyses for row in analysis["comparisons"]]
    payload = {
        "schema_version": SCHEMA_VERSION,
        "analyzer": str(Path(__file__).resolve()),
        "analyzer_sha256": _file_sha256(Path(__file__).resolve()),
        "metric": {
            "primary_order": "descending_float32_logit",
            "tie_breaker": "ascending_token_id",
            "entry": "(token_id, float32_logit)",
            "logit_equality": "exact_float32_element_bits",
            "first_divergent_rank_is_one_based": True,
            "softmax_materialized": False,
        },
        "summary_count": len(analyses),
        "comparison_count": len(comparisons),
        "ordered_pairs_identical_count": sum(
            row["ordered_pairs_identical"] for row in comparisons
        ),
        "top_token_difference_count": sum(
            row["top_token_differs"] for row in comparisons
        ),
        "analyses": analyses,
    }
    payload["result_sha256"] = sha256_json(payload)
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({key: payload[key] for key in (
        "summary_count",
        "comparison_count",
        "ordered_pairs_identical_count",
        "top_token_difference_count",
        "result_sha256",
    )}, sort_keys=True))


if __name__ == "__main__":
    main()
