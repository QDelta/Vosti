from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from triton_kernels.add import add
from triton_kernels.fattn_paged_swa import fattn_varlen_paged_swa
from triton_kernels.gelu_tanh_mul import gelu_tanh_mul, torch_gelu_tanh_mul
from triton_kernels.gemma_qk_norm import (
    gemma_head_rms_norm,
    torch_gemma_head_rms_norm,
)
from triton_kernels.gemma_rmsnorm import gemma_rmsnorm, torch_gemma_rmsnorm
from triton_kernels.rope import rope
from triton_kernels.scaled_embedding import (
    scaled_embedding,
    torch_scaled_embedding,
)


def test_gemma3_torch_references_preserve_architecture_specific_semantics() -> None:
    x = torch.tensor([[1.0, -2.0, 0.5, 3.0]], dtype=torch.bfloat16)
    y = torch.tensor([[0.5, 1.5, -1.0, 2.0]], dtype=torch.bfloat16)
    weight = torch.tensor([0.0, 0.5, -0.25, 1.0], dtype=torch.bfloat16)
    eps = 1e-6

    x_float = x.float()
    expected_norm = (
        x_float
        * torch.rsqrt(x_float.square().mean(dim=-1, keepdim=True) + eps)
        * (1.0 + weight.float())
    ).to(x.dtype)
    torch.testing.assert_close(
        torch_gemma_rmsnorm(x, weight, eps), expected_norm, rtol=0, atol=0
    )
    torch.testing.assert_close(
        torch_gelu_tanh_mul(x, y),
        F.gelu(x, approximate="tanh") * y,
        rtol=0,
        atol=0,
    )

    embedding_weight = torch.arange(20, dtype=torch.float32).reshape(5, 4).to(
        torch.bfloat16
    )
    ids = torch.tensor([3, 1], dtype=torch.int32)
    expected_embedding = F.embedding(ids, embedding_weight) * torch.tensor(
        math.sqrt(4), dtype=embedding_weight.dtype
    )
    torch.testing.assert_close(
        torch_scaled_embedding(ids, embedding_weight),
        expected_embedding,
        rtol=0,
        atol=0,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_gemma3_rowwise_triton_kernels_match_torch_at_4b_widths() -> None:
    torch.manual_seed(0)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    rows = 3

    hidden = torch.randn((rows, 2560), device=device, dtype=dtype)
    residual = torch.randn_like(hidden)
    norm_weight = torch.randn((2560,), device=device, dtype=dtype)
    norm_actual = gemma_rmsnorm(
        hidden,
        norm_weight,
        1e-6,
        launch_config={
            "BLOCK_M": 1,
            "BLOCK_N": 4096,
            "num_warps": 8,
            "num_stages": 2,
        },
    )
    torch.testing.assert_close(
        norm_actual,
        torch_gemma_rmsnorm(hidden, norm_weight, 1e-6),
        rtol=0.02,
        atol=0.02,
    )

    add_actual = add(
        hidden,
        residual,
        launch_config={
            "BLOCK_M": 1,
            "BLOCK_N": 4096,
            "num_warps": 8,
            "num_stages": 2,
        },
    )
    torch.testing.assert_close(add_actual, hidden + residual, rtol=0, atol=0)

    gate = torch.randn((rows, 10240), device=device, dtype=dtype) * 0.5
    up = torch.randn_like(gate) * 0.5
    gelu_actual = gelu_tanh_mul(
        gate,
        up,
        launch_config={
            "BLOCK_M": 1,
            "BLOCK_N": 4096,
            "num_warps": 8,
            "num_stages": 2,
        },
    )
    torch.testing.assert_close(
        gelu_actual,
        torch_gelu_tanh_mul(gate, up),
        rtol=0.02,
        atol=0.02,
    )

    embedding_weight = torch.randn((19, 2560), device=device, dtype=dtype)
    ids = torch.tensor([18, 3, 7], device=device, dtype=torch.int32)
    embedding_actual = scaled_embedding(
        ids,
        embedding_weight,
        launch_config={
            "D": 2560,
            "BLOCK_M": 1,
            "BLOCK_D": 4096,
            "num_warps": 8,
            "num_stages": 2,
        },
    )
    torch.testing.assert_close(
        embedding_actual,
        torch_scaled_embedding(ids, embedding_weight),
        rtol=0,
        atol=0,
    )

    head_values = torch.randn((rows, 8 * 256), device=device, dtype=dtype)
    head_weight = torch.randn((256,), device=device, dtype=dtype)
    head_actual = gemma_head_rms_norm(
        head_values,
        head_weight,
        8,
        1e-6,
        launch_config={
            "H": 8,
            "D": 256,
            "BLOCK_M": 1,
            "num_warps": 8,
            "num_stages": 2,
        },
    )
    torch.testing.assert_close(
        head_actual,
        torch_gemma_head_rms_norm(head_values, head_weight, 8, 1e-6),
        rtol=0.02,
        atol=0.02,
    )

    rope_values = torch.randn((rows * 8, 256), device=device, dtype=dtype)
    cos = torch.randn((rows * 8, 128), device=device, dtype=dtype)
    sin = torch.randn((rows * 8, 128), device=device, dtype=dtype)
    rope_actual = rope(
        rope_values,
        cos,
        sin,
        launch_config={
            "D": 256,
            "HD": 128,
            "BLOCK_M": 1,
            "num_warps": 8,
            "num_stages": 2,
        },
    )
    rope_float = rope_values.float()
    rope_expected = torch.cat(
        [
            rope_float[:, :128] * cos.float()
            - rope_float[:, 128:] * sin.float(),
            rope_float[:, 128:] * cos.float()
            + rope_float[:, :128] * sin.float(),
        ],
        dim=-1,
    ).to(dtype)
    torch.testing.assert_close(rope_actual, rope_expected, rtol=0, atol=0)


def _dense_swa_reference(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    q_lengths: tuple[int, ...],
    k_lengths: tuple[int, ...],
    block_table: torch.Tensor,
    *,
    window_size: int,
    scale: float,
) -> torch.Tensor:
    result = torch.empty_like(q)
    q_start = 0
    num_heads = q.shape[1]
    num_kv_heads = k_cache.shape[2]
    for batch_index, (q_len, k_len) in enumerate(zip(q_lengths, k_lengths)):
        page_id = int(block_table[batch_index, 0].item())
        q_shift = k_len - q_len
        for query_index in range(q_len):
            query_position = q_shift + query_index
            first_key = max(0, query_position - window_size + 1)
            for head_index in range(num_heads):
                kv_head = head_index * num_kv_heads // num_heads
                query = q[q_start + query_index, head_index].float()
                keys = k_cache[
                    page_id, first_key : query_position + 1, kv_head
                ].float()
                values = v_cache[
                    page_id, first_key : query_position + 1, kv_head
                ].float()
                probabilities = torch.softmax(keys @ query * scale, dim=0)
                result[q_start + query_index, head_index] = (
                    probabilities[:, None] * values
                ).sum(dim=0).to(q.dtype)
        q_start += q_len
    return result


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_gemma3_sliding_attention_matches_dense_mask_and_ignores_old_kv() -> None:
    torch.manual_seed(1)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    q_lengths = (3, 2)
    k_lengths = (6, 5)
    total_q = sum(q_lengths)
    q = torch.randn((total_q, 8, 256), device=device, dtype=dtype) * 0.25
    k_cache = torch.randn((2, 64, 4, 256), device=device, dtype=dtype) * 0.25
    v_cache = torch.randn_like(k_cache) * 0.25
    block_table = torch.tensor([[0], [1]], device=device, dtype=torch.int32)
    cu_q = torch.tensor([0, 3, 5], device=device, dtype=torch.int32)
    cu_k = torch.tensor([0, 6, 11], device=device, dtype=torch.int32)
    launch_config = {
        "D_HEAD": 256,
        "BLOCK_M": 16,
        "BLOCK_N": 64,
        "num_warps": 8,
        "num_stages": 2,
    }
    window_size = 4
    scale = 256.0**-0.5

    actual = fattn_varlen_paged_swa(
        q,
        k_cache,
        v_cache,
        cu_q,
        cu_k,
        max(q_lengths),
        max(k_lengths),
        window_size=window_size,
        softmax_scale=scale,
        block_table=block_table,
        launch_config=launch_config,
    )
    expected = _dense_swa_reference(
        q,
        k_cache,
        v_cache,
        q_lengths,
        k_lengths,
        block_table,
        window_size=window_size,
        scale=scale,
    )
    torch.testing.assert_close(actual, expected, rtol=0.03, atol=0.03)

    # For the last query in each request, positions strictly before
    # query_position-window+1 are numerically invisible even though the kernel
    # still receives and conservatively reasons about the complete KV prefix.
    perturbed_k = k_cache.clone()
    perturbed_v = v_cache.clone()
    perturbed_k[0, :2] += 10
    perturbed_v[0, :2] -= 10
    perturbed_k[1, :1] += 10
    perturbed_v[1, :1] -= 10
    perturbed = fattn_varlen_paged_swa(
        q,
        perturbed_k,
        perturbed_v,
        cu_q,
        cu_k,
        max(q_lengths),
        max(k_lengths),
        window_size=window_size,
        softmax_scale=scale,
        block_table=block_table,
        launch_config=launch_config,
    )
    torch.testing.assert_close(actual[[2, 4]], perturbed[[2, 4]], rtol=0, atol=0)
