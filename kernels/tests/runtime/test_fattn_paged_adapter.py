from __future__ import annotations

import math

import pytest
import torch

from triton_kernels.fattn_paged import _prepare_inputs


class DeviceValueRead(AssertionError):
    pass


class FakeCudaTensor:
    """Metadata-only tensor double that rejects every device-value read."""

    def __init__(self, shape: tuple[int, ...], *, dtype=torch.float16):
        self.shape = shape
        self.ndim = len(shape)
        self.dtype = dtype
        self.device = torch.device("cuda")
        self.is_cuda = True

    def stride(self, dim: int) -> int:
        del dim
        return 1

    def contiguous(self):
        return self

    def to(self, *, device=None, dtype=None):
        del device, dtype
        return self

    def numel(self) -> int:
        return math.prod(self.shape)

    def __getitem__(self, key):
        raise DeviceValueRead(f"device value read at {key!r}")


def valid_inputs():
    return {
        "q": FakeCudaTensor((2, 16, 128)),
        "k_cache": FakeCudaTensor((2, 64, 8, 128)),
        "v_cache": FakeCudaTensor((2, 64, 8, 128)),
        "cu_seqlens_q": FakeCudaTensor((3,), dtype=torch.int32),
        "cu_seqlens_k": FakeCudaTensor((3,), dtype=torch.int32),
        "max_seqlen_q": 1,
        "max_seqlen_k": 2,
        "dropout_p": 0.0,
        "softmax_scale": None,
        "softcap": 0.0,
        "alibi_slopes": None,
        "block_table": FakeCudaTensor((2, 1), dtype=torch.int32),
    }


def test_verified_caller_mode_performs_no_device_value_reads() -> None:
    prepared = _prepare_inputs(**valid_inputs(), value_checks=False)
    assert prepared[6:11] == (2, 16, 8, 128, 2)


def test_checked_mode_retains_device_value_validation_by_default() -> None:
    with pytest.raises(DeviceValueRead):
        _prepare_inputs(**valid_inputs())


def test_verified_caller_mode_keeps_host_shape_checks() -> None:
    arguments = valid_inputs()
    arguments["q"] = FakeCudaTensor((16, 128))
    with pytest.raises(ValueError, match="q must have shape"):
        _prepare_inputs(**arguments, value_checks=False)
