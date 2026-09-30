"""Small 31B-geometry GPU screens; neither full-model nor engine proof evidence."""

from types import SimpleNamespace

import pytest
import torch

from vosti_kernels.model_families.gemma4 import runtime
from vosti_kernels.model_families.gemma4.loader import FULL_ATTENTION, SLIDING_ATTENTION


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")


def exact(left, right):
    assert left.shape == right.shape and left.dtype == right.dtype
    assert torch.equal(left.contiguous().view(torch.uint8), right.contiguous().view(torch.uint8))


@pytest.fixture(scope="module")
def capability():
    # Raw source-attested primitive surface only. No sealed deployment or
    # engine-admission claim is made by these independently callable kernels.
    return runtime.load_runtime(runtime.config_for_profile("gemma-4-31b-it-text"), device="cuda:0")


def run_attention(capability, kind, q, k, v, table, q_lengths, k_lengths):
    step = SimpleNamespace(
        cu_seqlens_q=torch.tensor([0, *q_lengths], device=q.device, dtype=torch.int32).cumsum(0, dtype=torch.int32),
        cu_seqlens_k=torch.tensor([0, *k_lengths], device=q.device, dtype=torch.int32).cumsum(0, dtype=torch.int32),
        max_seqlen_q=max(q_lengths), max_seqlen_k=max(k_lengths), block_table=table)
    return capability.paged_attention(q, k, v, step, attention_kind=kind)


@pytest.mark.parametrize("kind,kv_heads,dim", [(FULL_ATTENTION, 4, 512), (SLIDING_ATTENTION, 16, 256)])
@pytest.mark.parametrize("q_len,k_len", [(1, 129), (17, 1089)])
def test_attention_reference_batch_rows_and_page_relocation(capability, kind, kv_heads, dim, q_len, k_len):
    generator = torch.Generator(device="cuda").manual_seed(20260907)
    heads, page_size = 32, 64
    pages = (k_len + page_size - 1) // page_size
    q = torch.randn((q_len, heads, dim), device="cuda", dtype=torch.bfloat16, generator=generator) * 0.125
    k = torch.randn((pages, page_size, kv_heads, dim), device="cuda", dtype=q.dtype, generator=generator) * 0.125
    v = torch.randn(k.shape, device="cuda", dtype=q.dtype, generator=generator) * 0.125
    table = torch.arange(pages, device="cuda", dtype=torch.int32)[None, :]
    actual = run_attention(capability, kind, q, k, v, table, [q_len], [k_len])

    # Independent double-precision reference with bottom-right causal masking,
    # unit attention scale, and the exact 1024-token sliding-window convention.
    logical_k = k.flatten(0, 1)[:k_len].repeat_interleave(heads // kv_heads, dim=1).double()
    logical_v = v.flatten(0, 1)[:k_len].repeat_interleave(heads // kv_heads, dim=1).double()
    scores = torch.einsum("qhd,khd->hqk", q.double(), logical_k)
    query_pos = torch.arange(k_len - q_len, k_len, device=q.device)[:, None]
    key_pos = torch.arange(k_len, device=q.device)[None, :]
    visible = key_pos <= query_pos
    if kind == SLIDING_ATTENTION:
        visible &= key_pos > query_pos - 1024
    expected = torch.einsum("hqk,khd->qhd", scores.masked_fill(~visible[None], -float("inf")).softmax(-1), logical_v)
    torch.testing.assert_close(actual.double(), expected, rtol=0.03, atol=0.003)

    other = torch.randn((3, heads, dim), device=q.device, dtype=q.dtype, generator=generator)
    batched = run_attention(capability, kind, torch.cat([other, q]), k, v,
        table.repeat(2, 1), [3, q_len], [127, k_len])
    exact(actual, batched[3:])
    # Reverse physical placement and the block table together.
    moved = run_attention(capability, kind, q, k.flip(0).contiguous(), v.flip(0).contiguous(),
        table.flip(1).contiguous(), [q_len], [k_len])
    exact(actual, moved)


@pytest.mark.parametrize("kind,kv_heads,dim", [(FULL_ATTENTION, 4, 512), (SLIDING_ATTENTION, 16, 256)])
def test_value_normalization_reference_and_rows(capability, kind, kv_heads, dim):
    generator = torch.Generator(device="cuda").manual_seed(20260907)
    values = torch.randn((3, kv_heads * dim), device="cuda", dtype=torch.bfloat16, generator=generator)
    actual = capability.value_norm(values, attention_kind=kind)
    x = values.float().reshape(3, kv_heads, dim)
    expected = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6)
    torch.testing.assert_close(actual.float(), expected, rtol=0.008, atol=0.008)
    for row in range(3):
        exact(actual[row:row+1], capability.value_norm(values[row:row+1].clone(), attention_kind=kind))
