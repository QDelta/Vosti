"""Small offline attention launch screen; never selects from runtime shapes.

The same candidate is measured across all three phases. Results are candidate
evidence only: admission still requires the normal kernel/deployment gates.
Run under scripts/common/gpu_monitor.py on the reserved idle device.
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
from triton_kernels.fattn_paged import fattn_varlen_paged_fwd_block_ptr, select_config
from triton_kernels.fattn_paged_swa import (
    fattn_varlen_paged_swa, select_config as select_sliding_config,
)


def exact(a, b):
    return a.shape == b.shape and a.dtype == b.dtype and torch.equal(
        a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))


def deployed_configs(head_dim, window_size):
    return {'full': select_config(head_dim),
            'sliding': select_sliding_config(head_dim, window_size)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--head-dim', type=int, default=256)
    parser.add_argument('--heads', type=int, default=8)
    parser.add_argument('--kv-heads', type=int, default=4)
    parser.add_argument('--window-size', type=int, default=1024)
    parser.add_argument('--kind', choices=('full', 'sliding', 'both'), default='both')
    args = parser.parse_args()
    if min(args.head_dim, args.heads, args.kv_heads, args.window_size) <= 0 or args.heads % args.kv_heads:
        parser.error('invalid positive head/window geometry')
    args.output.mkdir(parents=True, exist_ok=False)
    configs = [{**select_config(args.head_dim), 'num_warps': w, 'num_stages': s}
               for w in (2, 4, 8) for s in (1, 2, 3)]
    deployed = deployed_configs(args.head_dim, args.window_size)
    report = dict(geometry={k: v for k, v in vars(args).items() if k != 'output'},
        seed=42, torch=torch.__version__, triton=triton.__version__, cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(), deployed_by_kind=deployed, candidates=configs,
        source_sha256={name: hashlib.sha256((KERNEL_ROOT / 'triton_kernels' / name).read_bytes()).hexdigest()
                       for name in ('fattn_paged.py', 'fattn_paged_swa.py')},
        driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), cases=[])
    for batch, q_len, k_len in ((4, 1, 8192), (4, 128, 8192), (1, 512, 512)):
        torch.manual_seed(42)
        q = torch.randn((batch * q_len, args.heads, args.head_dim), device='cuda', dtype=torch.bfloat16) * .25
        pages = batch * k_len // 64
        k = torch.randn((pages, 64, args.kv_heads, args.head_dim), device='cuda', dtype=q.dtype) * .25
        v = torch.randn_like(k) * .25
        table = torch.randperm(pages, device='cuda', dtype=torch.int32).reshape(batch, -1)
        cu_q = torch.arange(batch + 1, device='cuda', dtype=torch.int32) * q_len
        cu_k = torch.arange(batch + 1, device='cuda', dtype=torch.int32) * k_len
        one_q = torch.tensor([0, q_len], device='cuda', dtype=torch.int32)
        one_k = torch.tensor([0, k_len], device='cuda', dtype=torch.int32)
        row_q = torch.tensor([0, 1], device='cuda', dtype=torch.int32)
        for kind, wrapper in (('full', fattn_varlen_paged_fwd_block_ptr), ('sliding', fattn_varlen_paged_swa)):
            if args.kind not in ('both', kind):
                continue
            def run(config, query=q, lengths_q=cu_q, lengths_k=cu_k, rows=q_len, blocks=table):
                return wrapper(query, k, v, lengths_q, lengths_k, rows, k_len,
                    softmax_scale=args.head_dim ** -.5, block_table=blocks, value_checks=False,
                    launch_config=config, **({'window_size': args.window_size} if kind == 'sliding' else {}))
            reference = run(deployed[kind])
            case = dict(batch=batch, q_len=q_len, k_len=k_len, kind=kind,
                        deployed_config=deployed[kind], candidates=[])
            order = list(range(len(configs)))
            random.Random(43).shuffle(order)
            for index in order:
                config = configs[index]
                result = run(config)
                torch.testing.assert_close(result, reference, atol=.003, rtol=.03)
                singleton = run(config, q[:q_len], one_q, one_k, q_len, table[:1])
                selected = run(config, q[q_len - 1:q_len], row_q, one_k, 1, table[:1])
                consistent = exact(result[:q_len], singleton) and exact(result[q_len - 1:q_len], selected)
                if not consistent:
                    raise RuntimeError(f'candidate {config} violates tested row consistency')
                samples = [do_bench_cudagraph(lambda: run(config), rep=50) * 1000 for _ in range(3)]
                row = dict(index=index, config=config, median_us=statistics.median(samples), samples_us=samples,
                    batch_and_selected_row_bitwise=True, matches_deployed=exact(result, reference),
                    max_abs_vs_deployed=float((result.float() - reference.float()).abs().max().item()))
                case['candidates'].append(row)
                print(kind, batch, q_len, k_len, index, round(row['median_us'], 2), flush=True)
            report['cases'].append(case)
            (args.output / 'results.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
