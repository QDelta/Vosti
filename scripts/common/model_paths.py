"""Shared checkpoint layout and installed-family discovery for local tooling."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def supported_families() -> tuple[str, ...]:
    """Discover installed family scopes without importing GPU/runtime modules."""
    return tuple(sorted(path.parent.name for path in
                        (ROOT / "python/vosti_kernels/model_families").glob("*/scope.json")))


def checkpoint_path(name: str) -> str:
    """Resolve a checkpoint beneath VOSTI_MODEL_ROOT (default: repo/models)."""
    root = Path(os.environ.get("VOSTI_MODEL_ROOT", str(ROOT / "models")))
    return str(root.expanduser().resolve() / name)
