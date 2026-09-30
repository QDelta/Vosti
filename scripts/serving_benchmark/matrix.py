"""Materialize the approved performance plan; this does not launch servers."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.determinism_tests.protocol import CHECKPOINTS, EXECUTION_CONFIGS, sha256_json


def performance_matrix() -> list[dict]:
    shapes = []
    for concurrency in (1, 4):
        for query in (512, 2048, 8192):
            shapes.append(dict(kind="cold_prefill", cached_tokens=0, query_tokens=query,
                               output_tokens=1, concurrency=concurrency))
        for context in (1024, 8192, 32768):
            shapes.append(dict(kind="decode", initial_context_tokens=context,
                               output_tokens=256, concurrency=concurrency))
        for cached in (8192, 32768):
            for query in (128, 1024):
                shapes.append(dict(kind="cached_extension", cached_tokens=cached,
                                   query_tokens=query, output_tokens=1, concurrency=concurrency))
    for context, output in ((8192, 512), (32768, 1024)):
        workload = dict(initial_context_tokens=context, output_tokens=output,
                        appended_tokens=128, turns=4, sessions=8, think_seconds=0.5)
        for concurrency in (1, 4):
            shapes.append(dict(kind="multi_session", concurrency=concurrency, **workload))
        for load in ("light", "moderate"):
            shapes.append(dict(kind="multi_session_arrivals", load=load,
                               request_rate=None, rate_selection="shared_per_model_pilot",
                               **workload))
    return [dict(hardware="h200", checkpoint=model.key, model=model.model,
                 model_path=model.path, execution_config=mode.key,
                 repetition=repetition, **shape)
            for model in CHECKPOINTS if model.performance
            for mode in EXECUTION_CONFIGS
            for repetition in range(3)
            for shape in shapes]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    cells = performance_matrix()
    payload = dict(schema="vosti.performance-plan.v1", cells=cells, cell_count=len(cells),
        execution_status="planned_not_executed", logit_observers=False,
        sampling=dict(temperature=0, ignore_eos=True),
        timing="client TTFT, TPOT, inter-token latency, whole-workload throughput",
        acceptance=["verify token geometry and actual cache hits", "reject GPU contention",
                    "exclude disjoint warmup; record measured graph capture/compilation"],
        caveats=["concurrency is not a guarantee of physical batch size",
                 "TTFT is not pure GPU prefill time",
                 "eight-session runs do not establish reliable extreme-tail latency",
                 "open-loop rates must be calibrated once per model and shared across engines"])
    payload["matrix_sha256"] = sha256_json(payload)
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output:
        with args.output.open("x") as stream:
            stream.write(encoded)
    else:
        print(encoded, end="")


if __name__ == "__main__":
    main()
