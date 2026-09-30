"""CPU checks of the independent SDPA oracle, not kernel proof evidence."""
import pytest
import torch

from triton_kernels.fattn_paged import _sdpa_reference


@pytest.mark.parametrize("q_len,k_len", [(4, 4), (1, 7), (3, 7), (7, 3), (3, 0)])
def test_bottom_right_causal_grouped_attention(q_len, k_len):
    torch.manual_seed(123)
    q = torch.randn(q_len, 4, 8)
    k_cache = torch.randn(5, 2, 2, 8)
    v_cache = torch.randn_like(k_cache)
    # Non-contiguous physical page ordering, with padding that must be ignored.
    block_table = torch.tensor([[3, 1, 4, 0, 2]])
    out, lse, _ = _sdpa_reference(
        q, k_cache, v_cache, torch.tensor([0, q_len]),
        torch.tensor([0, k_len]), q_len, k_len,
        block_table=block_table, return_attn_probs=True,
    )
    for row in range(q_len):
        visible = max(0, k_len - q_len + row + 1)
        for head in range(4):
            if visible == 0:
                assert torch.equal(out[row, head], torch.zeros(8))
                assert torch.isneginf(lse[head, row])
                continue
            keys = torch.stack([k_cache[block_table[0, p // 2], p % 2, head // 2]
                                for p in range(visible)]).double()
            values = torch.stack([v_cache[block_table[0, p // 2], p % 2, head // 2]
                                  for p in range(visible)]).double()
            scores = (keys @ q[row, head].double()) * 8 ** -0.5
            expected = torch.softmax(scores, dim=0) @ values
            torch.testing.assert_close(out[row, head].double(), expected, atol=1e-6, rtol=1e-5)
            torch.testing.assert_close(lse[head, row].double(), torch.logsumexp(scores, 0),
                                       atol=1e-6, rtol=1e-5)


def test_ragged_rows_ignore_poisoned_unused_pages_and_offsets():
    torch.manual_seed(321)
    q = torch.randn(5, 4, 8)
    k = torch.randn(7, 2, 2, 8)
    v = torch.randn_like(k)
    tables = torch.tensor([[2, 0, 6], [5, 1, 6]])
    cu_q, cu_k = torch.tensor([0, 2, 5]), torch.tensor([0, 3, 7])
    expected, expected_lse, _ = _sdpa_reference(
        q, k, v, cu_q, cu_k, 3, 4, block_table=tables, return_attn_probs=True)
    k[0, 1] = float('nan')
    v[0, 1] = float('nan')
    k[6] = float('nan')
    v[6] = float('nan')
    actual, actual_lse, _ = _sdpa_reference(
        q, k, v, cu_q, cu_k, 3, 4, block_table=tables, return_attn_probs=True)
    assert torch.equal(actual, expected)
    assert torch.equal(actual_lse, expected_lse)
