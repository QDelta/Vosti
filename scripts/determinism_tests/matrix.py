#!/usr/bin/env python3
"""Print or materialize the agreed cross-engine determinism matrix."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from scripts.determinism_tests.protocol import (
    EXECUTION_CONFIGS,
    CAMPAIGN_SEEDS,
    CHECKPOINTS,
    HARDWARE,
    MODELS,
    SCHEMA_VERSION,
    TESTS,
    logical_matrix,
    sha256_json,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--pretty", action="store_true")
    parser.add_argument("--seed", action="append", type=lambda value: int(value, 0),
                        help="retain one seed; repeat for more (default: three campaign seeds)")
    parser.add_argument(
        "--hardware",
        action="append",
        choices=HARDWARE,
        help="retain one hardware target; repeat to retain more than one",
    )
    args = parser.parse_args()
    selected_hardware = tuple(args.hardware or ("h200",))
    selected_seeds = tuple(args.seed or CAMPAIGN_SEEDS)
    suites = logical_matrix(hardware=selected_hardware, seeds=selected_seeds)
    strict = sum(bool(row["strict"]) for row in suites)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "hardware": list(selected_hardware),
        "models": [asdict(model) for model in MODELS],
        "checkpoints": [asdict(checkpoint) for checkpoint in CHECKPOINTS],
        "seeds": list(selected_seeds),
        "execution_configs": [asdict(config) for config in EXECUTION_CONFIGS],
        "tests": list(TESTS),
        "logical_suite_count": len(suites),
        "strict_suite_count": strict,
        "observational_suite_count": len(suites) - strict,
        "suites": suites,
    }
    payload["matrix_sha256"] = sha256_json(payload)
    encoded = json.dumps(payload, indent=2 if args.pretty else None, sort_keys=True) + "\n"
    if args.output:
        output = args.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.exists():
            raise FileExistsError(f"refusing to overwrite {output}")
        output.write_text(encoded, encoding="utf-8")
    else:
        print(encoded, end="")


if __name__ == "__main__":
    main()
