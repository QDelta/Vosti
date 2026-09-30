"""Multi-page SWA checks; exact comparisons include signed zero."""
from __future__ import annotations

import pytest
import torch

from triton_kernels.fattn_paged_swa import fattn_varlen_paged_swa, select_config


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")


def _exact(left, right):
    assert left.shape == right.shape and left.dtype == right.dtype
    assert torch.equal(left.contiguous().view(torch.uint8), right.contiguous().view(torch.uint8))


def _run(q, k, v, table, q_lengths, k_lengths, window=512):
    cu_q = torch.tensor([0, *q_lengths], device=q.device, dtype=torch.int32).cumsum(0, dtype=torch.int32)
    cu_k = torch.tensor([0, *k_lengths], device=q.device, dtype=torch.int32).cumsum(0, dtype=torch.int32)
    return fattn_varlen_paged_swa(
        q, k, v, cu_q, cu_k, max(q_lengths), max(k_lengths),
        window_size=window, softmax_scale=q.shape[-1] ** -0.5,
        block_table=table, launch_config=select_config(q.shape[-1], window),
        value_checks=False,
    )


@pytest.mark.parametrize("k_len", [512, 513, 575, 576, 8192])
@pytest.mark.parametrize("q_len", [1, 17, 65])
def test_sliding_attention_window_boundaries(k_len, q_len):
    torch.manual_seed(42)
    heads, kv_heads, dim, window = 8, 4, 256, 512
    pages = (k_len + 63) // 64
    q = torch.randn((q_len, heads, dim), device="cuda", dtype=torch.bfloat16) * 0.25
    k = torch.randn((pages, 64, kv_heads, dim), device="cuda", dtype=q.dtype) * 0.25
    v = torch.randn_like(k) * 0.25
    table = torch.arange(pages, device="cuda", dtype=torch.int32)[None, :]
    actual = _run(q, k, v, table, [q_len], [k_len])

    # Batch padding with a different request must not change any selected row.
    other_q = torch.randn((3, heads, dim), device="cuda", dtype=q.dtype)
    batched = _run(torch.cat([other_q, q]), k, v, table.repeat(2, 1), [3, q_len], [127, k_len])
    _exact(actual, batched[3:])
    # Physically relocate every page without changing logical KV.
    relocated = _run(q, k.flip(0), v.flip(0), table.flip(1).contiguous(), [q_len], [k_len])
    _exact(actual, relocated)

    for row in sorted({0, q_len - 1, min(15, q_len - 1), min(16, q_len - 1)}):
        position = k_len - q_len + row
        # A singleton decode and a differently aligned partial prefill select
        # the same query and logical prefix as the original query block.
        singleton = _run(q[row:row + 1], k, v, table, [1], [position + 1])
        partial = _run(q[row:], k, v, table, [q_len - row], [k_len])
        _exact(actual[row:row + 1], singleton)
        _exact(actual[row:row + 1], partial[:1])

        first = max(0, position - window + 1)
        keys = k.flatten(0, 1)[first:position + 1].repeat_interleave(heads // kv_heads, dim=1).float()
        values = v.flatten(0, 1)[first:position + 1].repeat_interleave(heads // kv_heads, dim=1).float()
        probs = torch.softmax(torch.einsum("hd,khd->hk", q[row].float(), keys) * dim ** -0.5, dim=-1)
        expected = torch.einsum("hk,khd->hd", probs, values).to(q.dtype)
        torch.testing.assert_close(actual[row], expected, rtol=0.03, atol=0.003)

    # All keys before the earliest row's window are invisible (this empirical
    # check does not weaken the exported all-prefix contract or allow eviction).
    expired = max(0, k_len - q_len - window + 1)
    changed_k, changed_v = k.clone(), v.clone()
    changed_k.flatten(0, 1)[:expired] += 10
    changed_v.flatten(0, 1)[:expired] -= 10
    _exact(actual, _run(q, changed_k, changed_v, table, [q_len], [k_len]))


def test_sliding_attention_graph_replay():
    torch.manual_seed(43)
    q = torch.randn((1, 8, 256), device="cuda", dtype=torch.bfloat16)
    k = torch.randn((128, 64, 4, 256), device="cuda", dtype=q.dtype)
    v = torch.randn_like(k)
    table = torch.arange(128, device="cuda", dtype=torch.int32)[None, :]
    cu_q = torch.tensor([0, 1], device="cuda", dtype=torch.int32)
    cu_k = torch.tensor([0, 8192], device="cuda", dtype=torch.int32)

    def run():
        return fattn_varlen_paged_swa(
            q, k, v, cu_q, cu_k, 1, 8192, window_size=512,
            softmax_scale=256 ** -0.5, block_table=table,
            launch_config=select_config(256, 512), value_checks=False,
        )

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            run()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = run()
    for _ in range(3):
        q.copy_(torch.randn_like(q))
        graph.replay()
        _exact(captured, run())
