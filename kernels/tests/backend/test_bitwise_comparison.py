import torch

from backend.probes import _equal_bits


def test_signed_zero_is_a_mismatch_but_identical_nan_payloads_match():
    positive = torch.tensor([0.0])
    negative = torch.tensor([-0.0])
    assert torch.equal(positive, negative)
    assert not _equal_bits(positive, negative)
    nan = torch.tensor([0x7fc00001], dtype=torch.int32).view(torch.float32)
    other = torch.tensor([0x7fc00002], dtype=torch.int32).view(torch.float32)
    assert _equal_bits(nan, nan.clone())
    assert not _equal_bits(nan, other)


def test_bitwise_comparison_requires_shape_and_dtype_but_not_layout():
    value = torch.arange(12, dtype=torch.float32).reshape(3, 4).t()
    assert not value.is_contiguous()
    assert _equal_bits(value, value.contiguous())
    assert not _equal_bits(value, value.flatten())
    assert not _equal_bits(value, value.double())
    assert _equal_bits(torch.tensor(0.0), torch.tensor(0.0))
    assert _equal_bits(torch.empty(0), torch.empty(0))
