"""Runtime defenses for assumptions that remain outside the static proofs.

These checks do not themselves connect the Python launch to the analyzed IR;
that correspondence remains a framework/deployment proof obligation.
"""

from collections.abc import Mapping

import torch


def require_writable_tensors(**tensors: torch.Tensor) -> None:
    """Check injective row-major layouts and pairwise distinct storage.

    Empty tensors are excluded from alias comparison because no address is
    written and independent empty storages may share data pointer 0.
    """

    storage_owners: dict[int, str] = {}
    for name, tensor in tensors.items():
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"writable {name!r} is not a torch.Tensor")
        if not tensor.is_contiguous():
            raise ValueError(
                f"writable tensor {name!r} must have contiguous row-major layout"
            )
        if tensor.numel() == 0:
            continue
        storage_pointer = tensor.untyped_storage().data_ptr()
        previous = storage_owners.get(storage_pointer)
        if previous is not None:
            raise ValueError(
                f"writable tensors {previous!r} and {name!r} alias one storage"
            )
        storage_owners[storage_pointer] = name


# @kernel-bridge-begin runtime_contracts::dense_launch_tensors
def require_dense_launch_tensors(
    *,
    readonly: Mapping[str, torch.Tensor],
    writable: Mapping[str, torch.Tensor],
) -> None:
    """Check the physical layout and alias premises of one kernel launch.

    Read/read aliasing is harmless and deliberately allowed. Every nonempty
    writable tensor must own storage distinct from all other writable tensors
    and every read-only input. All tensors must be dense row-major on one
    device, which makes logical indices injective physical addresses for the
    launch shapes passed to Triton.
    """

    if not readonly or not writable:
        raise ValueError("dense launch checks require read-only and writable tensors")
    tensors = {**readonly, **writable}
    if len(tensors) != len(readonly) + len(writable):
        raise ValueError("read-only and writable launch tensor names must be distinct")

    device = None
    for name, tensor in tensors.items():
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"launch tensor {name!r} is not a torch.Tensor")
        if not tensor.is_contiguous():
            raise ValueError(
                f"launch tensor {name!r} must have contiguous row-major layout"
            )
        if device is None:
            device = tensor.device
        elif tensor.device != device:
            raise ValueError("every launch tensor must be on one device")

    writable_storage: dict[int, str] = {}
    for name, tensor in writable.items():
        if tensor.numel() == 0:
            continue
        pointer = tensor.untyped_storage().data_ptr()
        previous = writable_storage.get(pointer)
        if previous is not None:
            raise ValueError(
                f"writable launch tensors {previous!r} and {name!r} alias one storage"
            )
        writable_storage[pointer] = name

    for name, tensor in readonly.items():
        if tensor.numel() == 0:
            continue
        writable_name = writable_storage.get(tensor.untyped_storage().data_ptr())
        if writable_name is not None:
            raise ValueError(
                f"read-only launch tensor {name!r} aliases writable tensor "
                f"{writable_name!r}"
            )
# @kernel-bridge-end runtime_contracts::dense_launch_tensors


def require_int32_tensors(**tensors: torch.Tensor) -> None:
    """Check the runtime dtype promised by ``@params tensor(..., int32)``."""
    for name, tensor in tensors.items():
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"int32 metadata {name!r} is not a torch.Tensor")
        if tensor.dtype != torch.int32:
            raise TypeError(
                f"int32 metadata tensor {name!r} must have dtype torch.int32, "
                f"got {tensor.dtype}"
            )
