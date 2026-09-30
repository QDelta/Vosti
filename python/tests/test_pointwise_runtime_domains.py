"""CPU checks of the physical domains used by checked pointwise bindings."""

import pytest
import torch

from triton_kernels import add, gelu_tanh_mul, scale, silu_mul, softcap


@pytest.mark.parametrize("rows", [0, 2])
def test_add_rejects_zero_width_before_launch(rows):
    x = torch.empty((rows, 0))
    with pytest.raises(ValueError, match="positive row width"):
        add.add(x, x, launch_config={})


@pytest.mark.parametrize("module", [add, gelu_tanh_mul, silu_mul])
def test_binary_pointwise_rejects_mismatched_shape(module):
    operation = getattr(module, module.__name__.rsplit(".", 1)[-1])
    with pytest.raises(AssertionError, match="Shape mismatch"):
        operation(torch.empty((2, 4)), torch.empty((2, 5)), launch_config={})


@pytest.mark.parametrize("shape", [(), (2,), (1, 1)])
def test_scale_requires_exact_scalar_tensor_shape(shape):
    with pytest.raises(ValueError, match="one-element weight"):
        scale.scale(torch.empty((2, 4)), torch.empty(shape), launch_config={})


@pytest.mark.parametrize("cap", [True, 0, -1, float("inf"), float("nan")])
def test_softcap_parameter_domain(cap):
    with pytest.raises(ValueError, match="finite positive cap"):
        softcap.softcap(torch.empty((2, 4)), cap, launch_config={})
