#!/usr/bin/env python3
"""Run one supported model engine and validate its common serving surface."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[2]
FAMILIES = {
    "qwen3": {
        "architecture": "qwen3",
        "expected_requests": 1,
    },
    "gemma3": {
        "architecture": "gemma3_text",
        "expected_requests": 1,
    },
    "llama3": {
        "architecture": "llama3",
        "expected_requests": 1,
    },
    "gemma4": {
        "architecture": "gemma4_text",
        "expected_requests": 1,
    },
}


def read_output_tokens(path: Path) -> list[list[int]]:
    indexed: list[tuple[int, list[int]]] = []
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        try:
            raw_index, raw_tokens = raw_line.split("\t", 1)
            index = int(raw_index)
            tokens = [int(token) for token in raw_tokens.split(",") if token]
        except ValueError as error:
            raise ValueError(
                f"{path}:{line_number} is not an indexed token record"
            ) from error
        indexed.append((index, tokens))
    indexed.sort()
    if [index for index, _ in indexed] != list(range(len(indexed))):
        raise ValueError(f"{path} has non-contiguous request indexes")
    return [tokens for _, tokens in indexed]


def validate_smoke_result(
    *,
    architecture: str,
    expected_requests: int,
    expected_tokens_per_request: int,
    stdout: str,
    output_path: Path,
) -> list[list[int]]:
    marker = (
        f"MODEL_RUNTIME architecture={architecture} "
        "backend_qualified=true"
    )
    if stdout.splitlines().count(marker) != 1:
        raise ValueError(f"engine output lacks exactly one {marker!r} marker")
    outputs = read_output_tokens(output_path)
    if len(outputs) != expected_requests:
        raise ValueError(
            f"engine emitted {len(outputs)} request records, "
            f"expected {expected_requests}"
        )
    lengths = [len(tokens) for tokens in outputs]
    if lengths != [expected_tokens_per_request] * expected_requests:
        raise ValueError(
            f"engine output lengths are {lengths}, expected "
            f"{expected_tokens_per_request} per request"
        )
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("family", choices=tuple(FAMILIES))
    parser.add_argument("--max-tokens", type=int, default=2)
    args = parser.parse_args()
    if args.max_tokens <= 0:
        parser.error("--max-tokens must be positive")

    family = FAMILIES[args.family]
    with tempfile.TemporaryDirectory(prefix="vosti-engine-smoke-") as directory:
        output_path = Path(directory) / "output_tokens.tsv"
        environment = os.environ.copy()
        environment.update(
            {
                "VOSTI_BENCH": "1",
                "VOSTI_IGNORE_EOS": "1",
                "VOSTI_MAX_TOKENS": str(args.max_tokens),
                "VOSTI_OUTPUT_TOKENS": str(output_path),
                "VOSTI_QUIET_OUTPUT": "1",
            }
        )
        completed = subprocess.run(
            [sys.executable, str(ROOT / "scripts/launch.py"), "--kind", "engine", "--family", args.family],
            cwd=ROOT,
            env=environment,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        print(completed.stdout, end="")
        if completed.returncode != 0:
            raise RuntimeError(
                f"{args.family} Engine smoke run failed with "
                f"exit code {completed.returncode}"
            )
        outputs = validate_smoke_result(
            architecture=str(family["architecture"]),
            expected_requests=int(family["expected_requests"]),
            expected_tokens_per_request=args.max_tokens,
            stdout=completed.stdout,
            output_path=output_path,
        )
    print(
        f"{args.family} Engine smoke check passed: "
        f"{len(outputs)} requests, {sum(map(len, outputs))} output tokens"
    )


if __name__ == "__main__":
    main()
