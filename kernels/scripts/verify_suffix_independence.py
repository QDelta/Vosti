#!/usr/bin/env python3
"""Bounded symbolic causal suffix-independence check for paged attention.

For the translated `fattn_varlen_paged_fwd_block_ptr_kernel` at small
concrete shapes, symbolically evaluates the kernel (diagnostics/suffix_independence)
and certifies, per query row j, that the output row's Z3 terms mention NO
K/V lane beyond the row's causal window [0, k_len - q_len + j + 1) — while
the last row (which sees everything) DOES mention the final position's lanes
(vacuity guard).

This is supporting evidence for causal confinement, not an unbounded theorem:
it symbolically checks selected concrete sequence lengths and a reduced head
dimension. The guarded-region analysis separately reasons about symbolic rows;
neither result is yet composed into the top annotation theorem.

Usage:
    uv run python scripts/verify_suffix_independence.py
"""

from __future__ import annotations

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

import z3  # noqa: E402

from ir.subst import expand_let_bindings, specialize_kernel_constants  # noqa: E402
from diagnostics.suffix_independence import (  # noqa: E402
    Evaluator,
    TensorVal,
    check_lane_independence,
    free_reals,
)
from ir.translate import translate_kernel_source  # noqa: E402

from triton_kernels.fattn_paged import PAGE_SIZE, select_config  # noqa: E402

# Concrete instances.  The two TOY configs (2x2 tiles, 2-slot pages) keep
# the multi-page gather and per-block causal offsets cheap to exercise; the
# DEPLOYED-TILE config is derived from the SAME launch rule inference uses
# (`select_config` at the compiled engine page size), so the mask
# and control structure verified here is the one the launched binary
# specializes.  Sequence length remains the one extrapolated dimension
# (bounded unrolling; the hardware falsifier covers deployed lengths).
B, H, HKV = 1, 1, 1

TOY = dict(BLOCK_M=2, BLOCK_N=2, D_HEAD=2, PAGE=2, NUM_PAGES=4)
CONFIGS = [
    dict(Q_LEN=2, K_LEN=4, MAXP=2, BT=[[0, 1]], **TOY),
    dict(Q_LEN=3, K_LEN=5, MAXP=3, BT=[[0, 1, 2]], **TOY),
]


def deployed_config() -> dict:
    """The tile config the engine actually launches, at its page size."""
    page = PAGE_SIZE
    cfg = select_config(8)
    bm, bn = cfg["BLOCK_M"], cfg["BLOCK_N"]
    # D_HEAD is deliberately REDUCED (deployed: 128; here: 8, ~40s — 128
    # exceeds 30min).  Sound reduction for causality: the kernel masks on
    # POSITIONS, never on head dims — every (pos, d) lane enters/leaves the
    # causal window together, so d_head only lengthens dot products without
    # touching mask or control structure.  Override: VOSTI_SUFFIX_D_HEAD.
    d_head = int(os.environ.get("VOSTI_SUFFIX_D_HEAD", "8"))
    # Two query tiles and two k-steps: exercises the per-tile causal offset
    # arithmetic at the REAL tile geometry.
    q_len = bm + 1
    k_len = bn + 1
    maxp = -(-k_len // page)
    return dict(Q_LEN=q_len, K_LEN=k_len, MAXP=maxp,
                BT=[list(range(maxp))],
                BLOCK_M=bm, BLOCK_N=bn, D_HEAD=d_head, PAGE=page,
                NUM_PAGES=maxp + 1, DEPLOYED=True)


def sym_tensor(name: str, shape) -> TensorVal:
    lanes = []

    def build(prefix, dims):
        if not dims:
            lanes.append(z3.Real(f"{name}_" + "_".join(map(str, prefix))))
            return
        for i in range(dims[0]):
            build(prefix + [i], dims[1:])

    build([], list(shape))
    return TensorVal(tuple(shape), lanes)


def int_tensor(shape, data) -> TensorVal:
    flat = []

    def flatten(x):
        if isinstance(x, list):
            for y in x:
                flatten(y)
        else:
            flat.append(int(x))

    flatten(data)
    return TensorVal(tuple(shape), flat)


def kv_var(bt, page_size: int, name: str, pos: int, h: int, d: int) -> str:
    page, slot = bt[0][pos // page_size], pos % page_size
    return f"{name}_{page}_{slot}_{h}_{d}"


def check_config(source: str, cfg: dict) -> list[str]:
    q_len, k_len, maxp, bt = cfg["Q_LEN"], cfg["K_LEN"], cfg["MAXP"], cfg["BT"]
    BLOCK_M, BLOCK_N = cfg["BLOCK_M"], cfg["BLOCK_N"]
    D_HEAD, PAGE, NUM_PAGES = cfg["D_HEAD"], cfg["PAGE"], cfg["NUM_PAGES"]
    needed_pages = -(-k_len // PAGE) if PAGE > 0 else 0
    config_errors = []
    if min(q_len, k_len, maxp, BLOCK_M, BLOCK_N, D_HEAD, PAGE, NUM_PAGES) <= 0:
        config_errors.append("all suffix-checker dimensions must be positive")
    if q_len > k_len:
        config_errors.append(f"q_len={q_len} exceeds k_len={k_len}")
    if PAGE % BLOCK_N != 0:
        config_errors.append(
            f"page={PAGE} is not divisible by BLOCK_N={BLOCK_N}")
    if maxp < needed_pages:
        config_errors.append(
            f"block table width {maxp} does not cover {needed_pages} logical pages")
    if len(bt) != B or any(len(row) != maxp for row in bt):
        config_errors.append(f"block table shape does not equal ({B}, {maxp})")
    if any(page_id < 0 or page_id >= NUM_PAGES for row in bt for page_id in row):
        config_errors.append(
            f"block table contains a page outside [0, {NUM_PAGES})")
    if config_errors:
        return [f"invalid suffix-checker config: {error}" for error in config_errors]
    kernel = translate_kernel_source(
        source, "fattn_varlen_paged_fwd_block_ptr_kernel"
    )
    kernel = specialize_kernel_constants(kernel, {
        "BLOCK_M": BLOCK_M, "BLOCK_N": BLOCK_N,
        "D_HEAD": D_HEAD, "PAGE_BLOCK_SIZE": PAGE,
    })
    kernel = expand_let_bindings(kernel)

    scalar_env = {
        "B": B, "H": H, "Hkv": HKV,
        "Tq": q_len, "Tk": k_len,
        "NUM_PAGES": NUM_PAGES, "MAX_NUM_PAGES": maxp,
        "max_seqlen_q": q_len, "max_seqlen_k": k_len,
        "scale_log2": 1,
    }
    tensors = {
        "q": sym_tensor("q", (q_len, H, D_HEAD)),
        "k_cache": sym_tensor("k", (NUM_PAGES, PAGE, HKV, D_HEAD)),
        "v_cache": sym_tensor("v", (NUM_PAGES, PAGE, HKV, D_HEAD)),
        "o": TensorVal((q_len, H, D_HEAD), [0.0] * (q_len * H * D_HEAD)),
        "lse": TensorVal((H, q_len), [0.0] * (H * q_len)),
        "block_table": int_tensor((B, maxp), bt),
        "cu_seqlens_q": int_tensor((B + 1,), [0, q_len]),
        "cu_seqlens_k": int_tensor((B + 1,), [0, k_len]),
    }

    ev = Evaluator(kernel, scalar_env, tensors)
    grid_vars = [it.var.name for it in kernel.grid.iters]
    assert len(grid_vars) == 3, grid_vars
    # Run EVERY grid instance (all q-blocks; B and H are 1 here).
    q_blocks = -(-q_len // BLOCK_M)
    for bi in range(B):
        for hi in range(H):
            for mb in range(q_blocks):
                ev.run_grid_instance(
                    {grid_vars[0]: bi, grid_vars[1]: hi, grid_vars[2]: mb},
                    tracked_outputs={"o"},
                )
    out = ev.stores["o"]

    failures = []
    for j in range(q_len):
        extent = k_len - q_len + j + 1          # visible positions [0, extent)
        suffix = set()
        for pos in range(extent, k_len):
            for h in range(HKV):
                for d in range(D_HEAD):
                    suffix.add(kv_var(bt, PAGE, "k", pos, h, d))
                    suffix.add(kv_var(bt, PAGE, "v", pos, h, d))
        used_all: set[str] = set()
        for h in range(H):
            for d in range(D_HEAD):
                term = out.get((j, h, d))
                ok, bad = check_lane_independence(term, suffix)
                used_all |= free_reals(term)
                if not ok:
                    failures.append(
                        f"(q{q_len},k{k_len}) row {j}: lane (h={h}, d={d}) "
                        f"depends on causally-invisible {sorted(bad)}")
        # Vacuity guard: the row must genuinely use its own window.
        window_vars = {kv_var(bt, PAGE, "k", p, h, d)
                       for p in range(extent)
                       for h in range(HKV) for d in range(D_HEAD)}
        if not (used_all & window_vars):
            failures.append(f"(q{q_len},k{k_len}) row {j}: uses no K window "
                            f"vars — vacuous")

    # Vacuity guard on the last row: it must see the final position.
    last_used = set()
    for h in range(H):
        for d in range(D_HEAD):
            last_used |= free_reals(out.get((q_len - 1, h, d)))
    final_pos_vars = {kv_var(bt, PAGE, "k", k_len - 1, h, d)
                      for h in range(HKV) for d in range(D_HEAD)}
    if not (last_used & final_pos_vars):
        failures.append(f"(q{q_len},k{k_len}) last row does not depend on the "
                        f"final position — not exercising the full window")
    return failures


def main() -> None:
    import time
    with open(os.path.join(REPO_ROOT, "triton_kernels", "fattn_paged.py")) as f:
        source = f.read()
    dep = deployed_config()
    failures = []
    for cfg in CONFIGS + [dep]:
        t0 = time.monotonic()
        failures += check_config(source, cfg)
        dt = time.monotonic() - t0
        tag = "DEPLOYED-TILE" if cfg.get("DEPLOYED") else "toy"
        print(f"  [{tag}] q={cfg['Q_LEN']} k={cfg['K_LEN']} "
              f"BLOCK_M={cfg['BLOCK_M']} BLOCK_N={cfg['BLOCK_N']} "
              f"page={cfg['PAGE']} d_head={cfg['D_HEAD']}: {dt:.1f}s")
    if failures:
        for f in failures:
            print(f"FAILED: {f}")
        sys.exit(1)
    shapes = ", ".join(f"(q={c['Q_LEN']},k={c['K_LEN']})"
                       for c in CONFIGS + [dep])
    print(f"bounded suffix-independence CHECKED for {shapes}")
    # ATTESTATION: the tile geometry verified above is the one the launch
    # rule selects at the engine's page size (bug-#6-class guard: if the
    # rule or page size ever changes, this line changes — and the parent's
    # `uv run --locked python scripts/project.py diagnostics` compares page sizes, so drift fails loudly).
    print(f"ATTESTED: launch rule selects BLOCK_M={dep['BLOCK_M']}, "
          f"BLOCK_N={dep['BLOCK_N']} at page={dep['PAGE']} — checked above "
          f"(d_head={dep['D_HEAD']}; sequence length remains "
          f"falsifier-covered)")


if __name__ == "__main__":
    main()
