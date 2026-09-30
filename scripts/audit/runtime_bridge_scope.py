"""Architecture-neutral registry and validation for trusted runtime bridges."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
SCOPE_PATH = ROOT / "python" / "vosti_kernels" / "runtime_bridge_scope.json"
_SOURCE_SPAN_FIELDS = {"path", "name", "digest"}
_RUNTIME_BRIDGE_FIELDS = {"name", "source_spans"}


def _is_digest(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _read_scope() -> dict:
    with SCOPE_PATH.open(encoding="utf-8") as scope_file:
        return json.load(scope_file)


def validate_source_span_schema(owner: str, spans) -> None:
    if not isinstance(spans, list) or not spans:
        raise ValueError(f"{owner} has no source spans")
    identities = set()
    for span in spans:
        if not isinstance(span, dict) or set(span) != _SOURCE_SPAN_FIELDS:
            raise ValueError(f"invalid source span in {owner}")
        identity = (span["path"], span["name"])
        if not all(isinstance(item, str) and item for item in identity):
            raise ValueError(f"invalid source span identity in {owner}")
        if identity in identities:
            raise ValueError(f"duplicate source span {identity} for {owner}")
        identities.add(identity)
        if not _is_digest(span["digest"]):
            raise ValueError(f"invalid source span digest for {owner}: {identity}")


def runtime_bridges() -> tuple[dict, ...]:
    scope = _read_scope()
    if set(scope) != {"schema_version", "bridges"} or scope["schema_version"] != 1:
        raise ValueError("runtime bridge scope has an unsupported schema")
    bridges = scope["bridges"]
    if not isinstance(bridges, list) or not bridges:
        raise ValueError("runtime bridge scope must contain a nonempty bridge list")
    names = set()
    for bridge in bridges:
        if not isinstance(bridge, dict) or set(bridge) != _RUNTIME_BRIDGE_FIELDS:
            raise ValueError("invalid trusted runtime bridge fields")
        name = bridge["name"]
        if not isinstance(name, str) or not name or name in names:
            raise ValueError("invalid or duplicate trusted runtime bridge name")
        names.add(name)
        validate_source_span_schema(
            f"trusted runtime bridge {name!r}", bridge["source_spans"]
        )
    return tuple(bridges)


def runtime_bridge_for(name: str) -> dict:
    matches = [bridge for bridge in runtime_bridges() if bridge["name"] == name]
    if len(matches) != 1:
        raise ValueError(f"no trusted runtime bridge named {name!r}")
    return matches[0]


def marked_source_digest(source: str, name: str) -> str:
    """Digest an explicitly delimited source span in any text language."""

    begin_marker = f"@kernel-bridge-begin {name}"
    end_marker = f"@kernel-bridge-end {name}"
    lines = source.splitlines(keepends=True)
    begin = [
        index for index, line in enumerate(lines) if line.strip().endswith(begin_marker)
    ]
    end = [
        index for index, line in enumerate(lines) if line.strip().endswith(end_marker)
    ]
    if len(begin) != 1 or len(end) != 1 or begin[0] >= end[0]:
        raise ValueError(
            f"bridge source span {name!r} must have exactly one marker pair"
        )
    body = "".join(lines[begin[0] + 1 : end[0]])
    if not body.strip():
        raise ValueError(f"bridge source span {name!r} is empty")
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def validate_marked_source_spans(
    owner: str,
    spans: list[dict],
    source_overrides: dict[str, str] | None = None,
) -> None:
    source_overrides = source_overrides or {}
    framework_root = os.path.realpath(ROOT)
    for span in spans:
        relative_path = span["path"]
        full_path = os.path.realpath(os.path.join(framework_root, relative_path))
        if os.path.commonpath((framework_root, full_path)) != framework_root:
            raise ValueError(f"bridge source path escapes repository: {relative_path!r}")
        span_source = (
            source_overrides[relative_path]
            if relative_path in source_overrides
            else Path(full_path).read_text(encoding="utf-8")
        )
        actual = marked_source_digest(span_source, span["name"])
        if actual != span["digest"]:
            raise ValueError(
                "bridge source span "
                f"{relative_path}:{span['name']} digest mismatch for {owner!r}: "
                f"expected {span['digest']}, got {actual}"
            )


def validate_runtime_bridge_surfaces(
    bridge: dict,
    source_overrides: dict[str, str] | None = None,
) -> None:
    """Fail closed when a reviewed non-kernel runtime boundary changes."""

    validate_source_span_schema(bridge.get("name", "runtime bridge"), bridge["source_spans"])
    validate_marked_source_spans(
        bridge["name"], bridge["source_spans"], source_overrides
    )


def validate_runtime_bridge_scope() -> None:
    for bridge in runtime_bridges():
        validate_runtime_bridge_surfaces(bridge)


def refresh_runtime_bridge_scope() -> None:
    """Refresh digests for the already-reviewed closed span inventory."""

    scope = _read_scope()
    for bridge in scope["bridges"]:
        for span in bridge["source_spans"]:
            source = (ROOT / span["path"]).read_text(encoding="utf-8")
            span["digest"] = marked_source_digest(source, span["name"])
    SCOPE_PATH.write_text(json.dumps(scope, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    arguments = set(sys.argv[1:])
    if arguments == {"--write"}:
        refresh_runtime_bridge_scope()
    elif arguments:
        raise ValueError("usage: runtime_bridge_scope.py [--write]")
    validate_runtime_bridge_scope()
    return 0


__all__ = [
    "marked_source_digest",
    "refresh_runtime_bridge_scope",
    "runtime_bridge_for",
    "runtime_bridges",
    "validate_marked_source_spans",
    "validate_runtime_bridge_scope",
    "validate_runtime_bridge_surfaces",
    "validate_source_span_schema",
]


if __name__ == "__main__":
    raise SystemExit(main())
