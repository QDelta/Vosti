"""Small GPU correctness and exact row-invariance screens for neutral operators."""

import pytest
import torch

from triton_kernels import scale, softcap


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float32))
@pytest.mark.parametrize("name,width", (("scale", 5376), ("softcap", 262144)))
def test_scalar_pointwise_reference_and_batch_rows(name, width, dtype):
    generator = torch.Generator(device="cuda").manual_seed(20260907)
    x = torch.randn((3, width), generator=generator, device="cuda", dtype=dtype)
    module = scale if name == "scale" else softcap
    config = module.select_config(width)
    argument = torch.tensor([0.375], dtype=dtype, device="cuda") if name == "scale" else 30.0
    operation = getattr(module, name)
    batched = operation(x, argument, launch_config=config)
    expected = ((x.float() * argument.float()) if name == "scale" else
                30.0 * torch.tanh(x.float() / 30.0)).to(dtype)
    torch.testing.assert_close(batched, expected, rtol=0.008 if dtype == torch.bfloat16 else 1e-5,
                               atol=0.001 if dtype == torch.bfloat16 else 1e-5)
    for row in range(3):
        single = operation(x[row:row+1].clone(), argument, launch_config=config)
        assert torch.equal(batched[row:row+1].contiguous().view(torch.uint8),
                           single.contiguous().view(torch.uint8))
