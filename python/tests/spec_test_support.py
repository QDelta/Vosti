"""Compiler-resolved dependency checks shared by specification tests."""

from pathlib import Path
import re

from scripts.effort.dependencies import load_dependencies

ROOT = Path(__file__).resolve().parents[2]


def page_constants_source():
    """Copy the actual compiled/spec page constants into small Verus harnesses."""
    source = (ROOT / "src/types.rs").read_text()
    items = re.findall(
        r"^(?:#\[verifier::inline\]\n)?pub (?:spec )?const BLOCK_SIZE(?:_SPEC)?:[^;]+;",
        source, re.M,
    )
    assert len(items) == 2
    return "\n".join(items)


def compiler_closure(root):
    inventory = load_dependencies(ROOT)
    declarations = inventory["declarations"]
    pending = ["vosti_verus::" + root]
    seen = set()
    while pending:
        name = pending.pop()
        if name in seen:
            continue
        seen.add(name)
        declaration = declarations[name]
        pending.extend(declaration["references"])
    return {name: declarations[name] for name in seen}
