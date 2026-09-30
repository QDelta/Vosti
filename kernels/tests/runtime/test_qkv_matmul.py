from __future__ import annotations

import pytest
import torch

from triton_kernels.matmul import matmul
from triton_kernels.qkv_matmul import qkv_matmul


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize(
    "config",
    [
        {"BLOCK_M": 16, "BLOCK_N": 64, "BLOCK_K": 64,
         "num_warps": 2, "num_stages": 4},
        {"BLOCK_M": 16, "BLOCK_N": 32, "BLOCK_K": 128,
         "num_warps": 2, "num_stages": 3},
    ],
)
def test_qkv_matches_three_matmuls_bitwise(config: dict) -> None:
    torch.manual_seed(42)
    device = "cuda"
    a = torch.randn((7, 96), device=device, dtype=torch.bfloat16)
    wq = torch.randn((96, 80), device=device, dtype=torch.bfloat16)
    wk = torch.randn((96, 31), device=device, dtype=torch.bfloat16)
    wv = torch.randn((96, 31), device=device, dtype=torch.bfloat16)

    actual = qkv_matmul(a, wq, wk, wv, launch_config=config)
    expected = tuple(
        matmul(a, weight, launch_config=config) for weight in (wq, wk, wv)
    )

    assert all(torch.equal(got, want) for got, want in zip(actual, expected))
    assert len({tensor.data_ptr() for tensor in actual}) == 3
