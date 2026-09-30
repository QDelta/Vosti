#!/usr/bin/env python3
"""Empirical causal-confinement check for the paged-attention kernel.

The verified model defines attention row `j` of a request as a function of
its query row and the causally visible K/V window
[0, k_len - q_len + j + 1).  kernels's mask-aware dependency analysis now
proves this property conditionally for the translated kernel. This independent
hardware check pressures the remaining analyzer-to-compiled-kernel and
framework-launch bridge: perturbing cache content beyond a row's causal extent
must leave that row BIT-identical.

Cases:
  1. perturb a page no request maps        -> all rows identical
  2. perturb mapped pages beyond k_len     -> all rows identical
  3. perturb a position visible only to late rows of request A
       -> A's earlier rows and ALL of request B identical;
          A's later rows must actually change (vacuity guard)

Run from the repo root on a CUDA host:
    uv run --locked python scripts/checks/causal_confinement.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[2]
kernel_root = os.path.abspath(
    os.environ.get(
        "VOSTI_KERNEL_ROOT",
        os.path.join(repo_root, "kernels"),
    )
)
sys.path.insert(0, kernel_root)

import torch  # noqa: E402
from triton_kernels.fattn_paged import (  # noqa: E402
    PAGE_SIZE,
    fattn_varlen_paged_fwd_block_ptr,
    select_config,
)

DEV, DT = "cuda", torch.bfloat16
H, HKV, D = 4, 2, 128
PAGE = PAGE_SIZE
NUM_PAGES = 8
MAXP = 3

# Request A: q_len 5, k_len 2*PAGE+2 (3 deployed-size pages: 0,1,2).
# Request B: q_len 3, k_len 3.
Q_LENS = [5, 3]
K_LENS = [2 * PAGE + 2, 3]
CU_Q = [0, 5, 8]
CU_K = [0, K_LENS[0], K_LENS[0] + K_LENS[1]]
BT = [[0, 1, 2], [3, 0, 0]]  # B pads with page 0 (never read: 1 page suffices)


def run(q, k_cache, v_cache, cu_q, cu_k, bt):
    launch_config = select_config(D)
    out = fattn_varlen_paged_fwd_block_ptr(
        q, k_cache, v_cache, cu_q, cu_k,
        max(Q_LENS), max(K_LENS),
        softmax_scale=D ** -0.5,
        block_table=bt,
        launch_config=launch_config,
    )
    return out.clone()


def rows_equal(a, b, rows):
    return [bool(torch.equal(a[r], b[r])) for r in rows]


def main() -> None:
    torch.manual_seed(0)
    tq = CU_Q[-1]
    q = torch.randn(tq, H, D, device=DEV, dtype=DT)
    k_cache = torch.randn(NUM_PAGES, PAGE, HKV, D, device=DEV, dtype=DT)
    v_cache = torch.randn(NUM_PAGES, PAGE, HKV, D, device=DEV, dtype=DT)
    cu_q = torch.tensor(CU_Q, device=DEV, dtype=torch.int32)
    cu_k = torch.tensor(CU_K, device=DEV, dtype=torch.int32)
    bt = torch.tensor(BT, device=DEV, dtype=torch.int32)

    base = run(q, k_cache, v_cache, cu_q, cu_k, bt)
    failures = []

    # Case 1: pages nobody maps (4..7).
    kc, vc = k_cache.clone(), v_cache.clone()
    kc[4:] = torch.randn_like(kc[4:])
    vc[4:] = torch.randn_like(vc[4:])
    out = run(q, kc, vc, cu_q, cu_k, bt)
    if not torch.equal(out, base):
        failures.append("case1: unmapped-page perturbation changed some output row")

    # Case 2: mapped page 2 beyond A's k_len (offsets 2..PAGE-1 are beyond
    # k_len_A), plus B's page 3 beyond its k_len 3.
    kc, vc = k_cache.clone(), v_cache.clone()
    kc[2, 2:] = torch.randn_like(kc[2, 2:])
    vc[2, 2:] = torch.randn_like(vc[2, 2:])
    kc[3, 3:] = torch.randn_like(kc[3, 3:])
    vc[3, 3:] = torch.randn_like(vc[3, 3:])
    out = run(q, kc, vc, cu_q, cu_k, bt)
    if not torch.equal(out, base):
        failures.append("case2: beyond-k_len perturbation changed some output row")

    # Case 3: perturb position 2*PAGE-1 of A (page 1, offset PAGE-1).  Causal
    # extents make this position invisible to A's rows 0,1 and visible to
    # rows 2,3,4.
    kc, vc = k_cache.clone(), v_cache.clone()
    kc[1, PAGE - 1] = torch.randn_like(kc[1, PAGE - 1])
    vc[1, PAGE - 1] = torch.randn_like(vc[1, PAGE - 1])
    out = run(q, kc, vc, cu_q, cu_k, bt)
    eq = rows_equal(out, base, range(tq))
    # A rows 0,1 (global rows 0,1) and B rows (global 5,6,7) must be identical.
    for r in [0, 1, 5, 6, 7]:
        if not eq[r]:
            failures.append(f"case3: causally-invisible perturbation changed row {r}")
    # Vacuity guard: at least one of A's late rows must change.
    if all(eq[r] for r in [2, 3, 4]):
        failures.append("case3: perturbation invisible to ALL rows — test is vacuous")

    if failures:
        for f in failures:
            print(f"FAILED: {f}")
        sys.exit(1)
    changed = [r for r in [2, 3, 4] if not eq[r]]
    print(f"causal-confinement check passed: unmapped/beyond-k_len perturbations "
          f"bit-invisible; position-{2 * PAGE - 1} perturbation confined to "
          f"late rows {changed}")


if __name__ == "__main__":
    main()
