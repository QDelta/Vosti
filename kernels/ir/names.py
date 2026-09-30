"""Capture-free logical names for temporary coordinates in the typed IR."""

from collections.abc import Collection


def fresh_name(stem: str, occupied: Collection[str]) -> str:
    """Choose a deterministic unused name; the caller owns its scope binding."""
    name = stem
    while name in occupied:
        name += "_"
    return name
