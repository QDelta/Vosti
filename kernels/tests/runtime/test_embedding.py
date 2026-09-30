from __future__ import annotations

import pytest
import torch

from triton_kernels.embedding import embedding


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_embedding_padded_cover_3072() -> None:
    device = "cuda"
    weight = torch.randn((19, 3072), device=device, dtype=torch.bfloat16)
    ids = torch.tensor([18, 3, 7], device=device, dtype=torch.int32)

    actual = embedding(
        ids,
        weight,
        launch_config={
            "D": 3072,
            "BLOCK_M": 1,
            "BLOCK_D": 4096,
            "num_warps": 4,
            "num_stages": 3,
        },
    )

    torch.testing.assert_close(
        actual,
        torch.nn.functional.embedding(ids, weight),
        rtol=0,
        atol=0,
    )
