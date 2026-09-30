"""Source-derived deployed-kernel fixtures used by verifier tests."""

from pathlib import Path

from ir import Kernel
from ir.translate import translate_kernel_source


_KERNEL_DIR = Path(__file__).resolve().parent.parent / "triton_kernels"
def _translate(
    filename: str,
    kernel_name: str,
    *,
    specialize: dict[str, bool] | None = None,
) -> Kernel:
    source = (_KERNEL_DIR / filename).read_text(encoding="utf-8")
    return translate_kernel_source(source, kernel_name, specialize=specialize)


def matmul_kernel() -> Kernel:
    return _translate("matmul.py", "matmul_kernel")
