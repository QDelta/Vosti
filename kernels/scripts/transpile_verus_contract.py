#!/usr/bin/env python3
"""Verify one annotated Triton kernel and emit its raw Verus contract.

Example:
    python scripts/transpile_verus_contract.py \
        triton_kernels/silu_mul.py silu_mul_kernel \
        --constant BLOCK_M=1 --constant BLOCK_N=4096

The command emits nothing unless the kernel proof succeeds.  Its output is the
framework-neutral raw domain/post/singleton surface; a consumer still needs a
checked representation adapter and a source-bound imported kernel theorem.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from ir.verus_contract import (  # noqa: E402
    render_contract_manifest,
    render_standalone_verus_module,
    transpile_verified_kernel_source,
    transpile_verified_kernel_goals,
)


def _constant(value: str) -> tuple[str, int | float | bool]:
    name, separator, raw = value.partition("=")
    if not separator or not name:
        raise argparse.ArgumentTypeError("constants must use NAME=JSON_VALUE")
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as error:
        raise argparse.ArgumentTypeError(
            f"constant {name!r} is not a JSON scalar"
        ) from error
    if not isinstance(parsed, (bool, int, float)):
        raise argparse.ArgumentTypeError(
            f"constant {name!r} must be a bool, integer, or finite float"
        )
    if isinstance(parsed, float) and not (-float("inf") < parsed < float("inf")):
        raise argparse.ArgumentTypeError(f"constant {name!r} must be finite")
    return name, parsed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("kernel")
    parser.add_argument(
        "--constant",
        action="append",
        default=[],
        type=_constant,
        metavar="NAME=JSON_VALUE",
    )
    parser.add_argument("--symbol-prefix")
    parser.add_argument("--goal", action="append",
                        help="named annotation goal; repeat to share one execution model")
    parser.add_argument(
        "--standalone",
        action="store_true",
        help="wrap the fragment in a minimal Verus module for syntax checking",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()

    constants: dict[str, int | float | bool] = {}
    for name, value in args.constant:
        if name in constants:
            parser.error(f"duplicate constant {name!r}")
        constants[name] = value

    source = args.source.read_text(encoding="utf-8")
    goals = args.goal or ["batch_invariance"]
    if len(goals) == 1:
        rendered = transpile_verified_kernel_source(
            source, args.kernel, constants, symbol_prefix=args.symbol_prefix, goal_name=goals[0],
        )
    else:
        rendered = transpile_verified_kernel_goals(
            source, args.kernel, constants, symbol_prefix=args.symbol_prefix, goal_names=tuple(goals),
        )
    output = (
        render_standalone_verus_module(rendered)
        if args.standalone
        else rendered.body
    )
    if args.output is None:
        print(output, end="")
    else:
        args.output.write_text(output, encoding="utf-8")
    if args.manifest is not None:
        args.manifest.write_text(
            render_contract_manifest(rendered), encoding="utf-8"
        )


if __name__ == "__main__":
    main()
