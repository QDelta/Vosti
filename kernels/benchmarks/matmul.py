"""Offline fixed-(N,K) projection screen with decode-first comparison data.

No deployed selector is changed. New tiles are unadmitted candidates until
the regular proof, backend qualification, and serving regression gates pass.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import statistics
import sys

import torch
import triton
from triton.testing import do_bench_cudagraph

KERNEL_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(KERNEL_ROOT))
from triton_kernels.matmul import CONFIGS, matmul, select_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--n', type=int, required=True)
    parser.add_argument('--k', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if min(args.n, args.k) <= 0:
        parser.error('N and K must be positive')
    candidates = [select_config(args.n, args.k), *[CONFIGS[i] for i in (10, 11, 12, 13)]]
    for bm, bn, bk, warps in ((16, 16, 256, 2), (16, 16, 512, 4), (16, 32, 512, 4), (32, 64, 256, 4)):
        candidates.append(dict(BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk, num_warps=warps, num_stages=3))
    candidates = list({json.dumps(c, sort_keys=True): c for c in candidates}.values())
    args.output.mkdir(parents=True, exist_ok=False)
    report = dict(n=args.n, k=args.k, seed=42, torch=torch.__version__, triton=triton.__version__,
        gpu=torch.cuda.get_device_name(), deployed=select_config(args.n, args.k), candidates=candidates,
        source_sha256=hashlib.sha256((KERNEL_ROOT / 'triton_kernels/matmul.py').read_bytes()).hexdigest(),
        driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), cases=[])
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    b = torch.randn((args.n, args.k), device='cuda', dtype=torch.bfloat16).T * .25
    for rows in (1, 4, 128, 512):
        a = torch.randn((rows, args.k), device='cuda', dtype=b.dtype) * .25
        reference = (a.float() @ b.float()).to(a.dtype)
        case = dict(rows=rows, measurements=[])
        order = list(range(len(candidates)))
        random.Random(43).shuffle(order)
        for index in order:
            config = candidates[index]
            result = matmul(a, b, launch_config=config)
            torch.testing.assert_close(result, reference, atol=.05, rtol=.03)
            for row in {0, rows - 1}:
                singleton = matmul(a[row:row + 1], b, launch_config=config)
                if not torch.equal(result[row:row + 1].contiguous().view(torch.uint8), singleton.view(torch.uint8)):
                    raise RuntimeError(f'candidate {config} fails selected-row bitwise check')
            samples = [do_bench_cudagraph(lambda: matmul(a, b, launch_config=config), rep=50) * 1000 for _ in range(3)]
            measured = dict(index=index, config=config, median_us=statistics.median(samples),
                samples_us=samples, selected_rows_bitwise=True,
                max_abs_vs_reference=float((result.float() - reference.float()).abs().max().item()))
            case['measurements'].append(measured)
            print(rows, index, round(measured['median_us'], 2), flush=True)
        report['cases'].append(case)
        (args.output / 'results.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
