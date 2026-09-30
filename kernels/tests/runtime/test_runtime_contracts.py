"""Tests for optional runtime writable-layout defenses.

These checks do not constitute a static logical-to-physical proof bridge.
"""

import pytest
import torch

from triton_kernels.runtime_contracts import (
    require_dense_launch_tensors,
    require_int32_tensors,
    require_writable_tensors,
)


def test_distinct_contiguous_writable_tensors_pass():
    require_writable_tensors(
        first=torch.empty((2, 3)),
        second=torch.empty((2, 3)),
    )


def test_noncontiguous_writable_tensor_is_rejected():
    transposed = torch.empty((2, 3)).transpose(0, 1)
    with pytest.raises(ValueError, match="contiguous row-major"):
        require_writable_tensors(output=transposed)


def test_shared_storage_is_rejected_even_for_disjoint_views():
    storage = torch.empty(8)
    with pytest.raises(ValueError, match="alias one storage"):
        require_writable_tensors(first=storage[:4], second=storage[4:])


def test_empty_tensors_need_no_alias_evidence():
    storage = torch.empty(0)
    require_writable_tensors(first=storage, second=storage)


def test_non_tensor_is_rejected():
    with pytest.raises(TypeError, match="not a torch.Tensor"):
        require_writable_tensors(output=object())  # type: ignore[arg-type]


def test_int32_metadata_dtype_is_checked():
    require_int32_tensors(indices=torch.tensor([0, 1], dtype=torch.int32))
    with pytest.raises(TypeError, match="must have dtype torch.int32"):
        require_int32_tensors(indices=torch.tensor([0, 1], dtype=torch.int64))


def test_dense_launch_allows_read_aliases_but_separates_writes():
    read = torch.empty((2, 3))
    require_dense_launch_tensors(
        readonly={"first": read, "second": read},
        writable={"out": torch.empty_like(read)},
    )


def test_dense_launch_rejects_read_write_storage_alias():
    storage = torch.empty(12)
    with pytest.raises(ValueError, match="aliases writable tensor"):
        require_dense_launch_tensors(
            readonly={"input": storage[:6].reshape(2, 3)},
            writable={"output": storage[6:].reshape(2, 3)},
        )


def test_dense_launch_rejects_noncontiguous_read_tensor():
    with pytest.raises(ValueError, match="contiguous row-major"):
        require_dense_launch_tensors(
            readonly={"input": torch.empty((2, 3)).transpose(0, 1)},
            writable={"output": torch.empty((2, 3))},
        )


def test_dense_launch_requires_both_tensor_roles():
    with pytest.raises(ValueError, match="require read-only and writable"):
        require_dense_launch_tensors(readonly={}, writable={"output": torch.empty(1)})
