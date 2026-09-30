#!/usr/bin/env python3
"""Independently prove Qwen3 logical cross-program write disjointness.

This pass is deliberately separate from regional, causal, and relational
acceptance.  It proves a per-logical-tensor property only.  Physical layout,
cross-parameter no-alias, address arithmetic, and launch correspondence remain
explicit obligations and are never synthesized from opaque allocation names.

Run through the shared project environment:

    uv run --locked python scripts/verification/diagnostics/verify_qwen3_kernel_races.py
"""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
for path in (ROOT, ROOT / "python", ROOT / "kernels"):
    sys.path.insert(0, str(path))

from diagnostics.race import prove_logical_write_disjointness_from_annotations  # noqa: E402

from scripts.deployment.model_families.qwen3_scope import (  # noqa: E402
    REQUIRED_POST_TENSORS,
    deployed_cases,
    validate_contract_catalog,
    validate_deployed_case_coverage,
    validate_post_surface,
)


def main() -> None:
    validate_contract_catalog()
    cases = deployed_cases()
    validate_deployed_case_coverage(cases)

    all_ok = True
    checked_surfaces: set[tuple[str, str]] = set()
    for source_file, kernel_name, constants in cases:
        source = (ROOT / "kernels/triton_kernels" / source_file).read_text()
        label = ", ".join(f"{key}={value}" for key, value in constants.items())
        try:
            key = (source_file, kernel_name)
            if key not in checked_surfaces:
                validate_post_surface(source_file, kernel_name, source)
                checked_surfaces.add(key)
            result = prove_logical_write_disjointness_from_annotations(
                source,
                kernel_name,
                constants,
                goal_name="batch_invariance",
                timeout_ms=5000,
            )
        except Exception as error:
            all_ok = False
            print(f"[FAILED] {kernel_name}({label})")
            print(f"         raised {type(error).__name__}: {error}")
            continue

        if result.ok:
            print(f"[LOGICAL WRITE-DISJOINTNESS PASS] {kernel_name}({label})")
        else:
            all_ok = False
            print(f"[FAILED] {kernel_name}({label})")
            for check in result.checks:
                if not check.proved:
                    print(f"         logical-write-race:{check.name}: {check.details}")

    if not all_ok:
        print("Some deployed logical write-disjointness cases FAILED.")
        raise SystemExit(1)

    print(
        f"All {len(cases)} deployed logical write-disjointness cases passed for "
        f"{len(REQUIRED_POST_TENSORS)} engine-reachable kernels."
    )
    print("NOT DISCHARGED by this pass:")
    print("  - injective logical-to-physical writable layouts")
    print("  - physical no-alias between distinct writable tensor parameters")
    print("  - address-arithmetic and memory-safety obligations")
    print("  - correspondence between analyzed and executed launch arguments")


if __name__ == "__main__":
    main()
