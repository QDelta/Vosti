"""Check every family's declared kernel and support source digests."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / 'python') not in sys.path:
    sys.path.insert(0, str(ROOT / 'python'))
from vosti_kernels.kernel_modules import validate_kernel_sources
from vosti_kernels.model_profile import load_scope


def resolve_kernel_sources(root: Path = ROOT, kernel_root: Path | None = None) -> tuple[Path, str]:
    checkout = (kernel_root if kernel_root is not None else root / 'kernels').expanduser().resolve()
    if not (checkout / 'triton_kernels').is_dir():
        raise RuntimeError(f'no kernel sources at {checkout}')
    scopes = sorted((root / 'python/vosti_kernels/model_families').glob('*/scope.json'))
    if not scopes:
        raise RuntimeError('no model-family kernel scopes found')
    sources = {}
    for path in scopes:
        scope = load_scope(path)
        validate_kernel_sources(str(checkout), scope)
        for entry in [*scope['kernel_contracts'], *scope['runtime_support_sources']]:
            sources[entry['source']] = entry['source_sha256']
    digest = hashlib.sha256(json.dumps(sources, sort_keys=True).encode()).hexdigest()
    return checkout, digest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kernel-root', type=Path)
    args = parser.parse_args()
    checkout, digest = resolve_kernel_sources(kernel_root=args.kernel_root)
    print(f'Kernel sources PASS: {digest} ({checkout})')
