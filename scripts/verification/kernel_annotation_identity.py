"""Architecture-neutral identity for parsed kernel proof annotations."""

from __future__ import annotations

from dataclasses import fields, is_dataclass
import hashlib
import json

from ir.annotations import parse_verif_goals


def _canonical_annotation_value(value):
    if is_dataclass(value):
        return {
            "node": type(value).__name__,
            **{
                field.name: _canonical_annotation_value(getattr(value, field.name))
                for field in fields(value)
            },
        }
    if isinstance(value, set):
        return sorted(_canonical_annotation_value(item) for item in value)
    if isinstance(value, (list, tuple)):
        return [_canonical_annotation_value(item) for item in value]
    return value


def annotation_contract_digest(source: str, kernel_name: str) -> str:
    """Digest every parsed semantic annotation, independent of layout."""

    goals = parse_verif_goals(source, kernel_name)
    if not goals:
        raise ValueError(f"no @verif proof goal for {kernel_name!r}")
    payload = json.dumps(
        _canonical_annotation_value(goals),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


__all__ = ["annotation_contract_digest"]
