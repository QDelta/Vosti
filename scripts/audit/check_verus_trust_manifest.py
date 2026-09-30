#!/usr/bin/env python3
"""Cross-check the source trust inventory against Verus's typed VIR.

``scripts/audit/tcb.py`` remains the source-span/digest authority, but a
source scanner should not be the only mechanism deciding which declarations
are trusted.  This checker consumes ``crate-simple.vir`` emitted by the same
Verus invocation that proves the crate and compares Verus's typed declaration
modes with the source inventory.

The VIR text format is itself a tool interface rather than a proof object.  We
therefore parse only top-level Function/Datatype records and fail closed if
the expected fields or source correspondence are absent.
"""

from __future__ import annotations

from collections import Counter
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit.tcb import (
    rust_source_files,
    scan_external_type_records,
    scan_rust_trust_records,
)


_ITEM_START = re.compile(
    r'^\(@ "(?P<path>[^"\n]+):(?P<line>[0-9]+):[0-9]+:[^"\n]*" '
    r'\((?P<kind>Function|Datatype)\b'
)

_ASSUME_START = "(> AssertAssume :is_assume true"
_ASSUME_END = re.compile(r"\s:msg\s+(?:None|\(Some\b)", re.DOTALL)


def _paren_delta(text: str) -> int:
    """Count S-expression parentheses while ignoring quoted VIR strings."""
    delta = 0
    in_string = False
    escaped = False
    for character in text:
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
        elif character == '"':
            in_string = True
        elif character == "(":
            delta += 1
        elif character == ")":
            delta -= 1
    if in_string and not text.endswith("\n"):
        raise ValueError("unterminated VIR string")
    return delta


def _normalized_path(path: str) -> str:
    candidate = Path(path)
    if candidate.is_absolute():
        try:
            candidate = candidate.resolve().relative_to(ROOT)
        except ValueError:
            return path
    return candidate.as_posix()


def compiler_trust_records(vir_path: Path) -> list[tuple[str, int, str]]:
    """Return ``(source path, line, category)`` from typed Verus VIR."""
    records: list[tuple[str, int, str]] = []
    current: dict[str, object] | None = None

    with vir_path.open(encoding="utf-8") as vir_file:
        for raw_line in vir_file:
            if current is None:
                match = _ITEM_START.match(raw_line)
                if match is None:
                    continue
                path = _normalized_path(match.group("path"))
                if not path.startswith("src/"):
                    continue
                current = {
                    "path": path,
                    "line": int(match.group("line")),
                    "kind": match.group("kind"),
                    "depth": 0,
                    "metadata": "",
                    "field_tail": "",
                    "external": False,
                    "assume": False,
                    "assume_detail": "",
                    "assume_probe": "",
                }

            current["depth"] = int(current["depth"]) + _paren_delta(raw_line)
            metadata = str(current["metadata"])
            # The declaration mode/transparency is in the short header. Keep
            # enough normalized text to span pretty-printer line breaks.
            if len(metadata) < 32_768:
                current["metadata"] = (metadata + " " + raw_line.strip())[:32_768]
            field_window = str(current["field_tail"]) + raw_line
            current["field_tail"] = field_window[-256:]
            if re.search(r":is_external_body\s+true\b", field_window):
                current["external"] = True

            # Newer Verus releases lower ownership-resolution checks to
            # `AssertAssume(is_assume=true, HasResolved(...))`. These are
            # compiler-generated move/borrow bookkeeping, not source
            # `assume`/`admit` operations, and their enclosing functions are
            # still verified. Fail closed on every other true Assume node.
            assume_probe = str(current["assume_probe"])
            if not assume_probe:
                assume_start = raw_line.find(_ASSUME_START)
                if assume_start >= 0:
                    assume_probe = raw_line[assume_start:]
            else:
                assume_probe += raw_line
            if assume_probe:
                if re.search(r"\bHasResolved\b", assume_probe):
                    assume_probe = ""
                elif _ASSUME_END.search(assume_probe) is not None:
                    current["assume"] = True
                    current["assume_detail"] = re.sub(
                        r"\s+", " ", assume_probe,
                    )[:512]
                    assume_probe = ""
                if len(assume_probe) > 32_768:
                    raise ValueError(
                        "could not classify typed VIR assume node at "
                        f"{current['path']}:{current['line']}"
                    )
            current["assume_probe"] = assume_probe

            if int(current["depth"]) != 0:
                continue

            normalized = re.sub(r"\s+", " ", str(current["metadata"]))
            path = str(current["path"])
            line = int(current["line"])
            kind = str(current["kind"])
            if current["assume_probe"]:
                # Older/fixture VIR may omit the trailing `:msg` field used
                # by newer pretty-printers. Reaching the declaration boundary
                # without seeing `HasResolved` is still enough to reject this
                # true Assume node; the parser must not depend on an optional
                # field before failing closed.
                current["assume"] = True
                current["assume_detail"] = re.sub(
                    r"\s+", " ", str(current["assume_probe"]),
                )[:512]
                current["assume_probe"] = ""
            if kind == "Function" and bool(current["assume"]):
                raise ValueError(
                    f"typed function contains assume/admit at {path}:{line}: "
                    f"{current['assume_detail']}"
                )
            if kind == "Function" and bool(current["external"]):
                mode_match = re.search(
                    r":owning_module\s+\S+\s+:mode\s+(Exec|Proof|Spec)\b",
                    normalized,
                )
                if mode_match is None:
                    raise ValueError(
                        f"typed external function lacks a mode at {path}:{line}"
                    )
                mode = mode_match.group(1)
                if mode == "Exec":
                    category = "exec-trusted"
                elif mode == "Proof":
                    category = "proof-trusted"
                elif "(BodyVisibility Uninterpreted)" in normalized:
                    category = "uninterp"
                else:
                    raise ValueError(
                        f"unsupported trusted spec function at {path}:{line}"
                    )
                records.append((path, line, category))
            elif kind == "Datatype" and re.search(
                r":transparency\s+\(DatatypeTransparency\s+Never\b", normalized
            ):
                records.append((path, line, "external-type"))
            current = None

    if current is not None:
        raise ValueError(
            f"unterminated VIR {current['kind']} record at "
            f"{current['path']}:{current['line']}"
        )
    if not records:
        raise ValueError("typed VIR contains no project trusted declarations")
    return records


def source_trust_records() -> list[tuple[str, int, str]]:
    records: list[tuple[str, int, str]] = []
    for path in rust_source_files(ROOT / "src"):
        relative = path.relative_to(ROOT).as_posix()
        records.extend(
            (relative, int(record["line"]), str(record["category"]))
            for record in scan_rust_trust_records(path)
        )
        records.extend(
            (relative, int(record["line"]), "external-type")
            for record in scan_external_type_records(path)
        )
    return records


def validate_compiler_trust_manifest(vir_path: Path) -> Counter:
    compiler = Counter(compiler_trust_records(vir_path))
    source = Counter(source_trust_records())
    if compiler != source:
        missing = sorted((source - compiler).elements())
        undeclared = sorted((compiler - source).elements())
        raise ValueError(
            "source and compiler trusted-declaration inventories differ: "
            f"missing_from_vir={missing!r}, missing_from_source={undeclared!r}"
        )
    return compiler


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        raise SystemExit("usage: check_verus_trust_manifest.py PATH/TO/crate-simple.vir")
    records = validate_compiler_trust_manifest(Path(argv[1]))
    counts = Counter(category for _, _, category in records.elements())
    print(
        "compiler/source trust inventory agrees: "
        + ", ".join(f"{category}={counts[category]}" for category in sorted(counts))
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
