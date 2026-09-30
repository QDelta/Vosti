"""Keep optional diagnostics and retired entrypoints out of the proof path."""

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_core_modules_do_not_import_optional_diagnostics():
    # Checking every IR module also covers transitive dependencies of the
    # public verifier, including imports inside helper functions.
    for path in (ROOT / "ir").glob("*.py"):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
                if not node.module:
                    names += [alias.name for alias in node.names]
            elif isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            else:
                continue
            assert not any("diagnostics" in name.split(".") for name in names), (path.name, names)


def test_retired_modules_are_not_kept_as_compatibility_entrypoints():
    for name in ("verif.py", "race.py", "suffix_independence.py"):
        assert not (ROOT / "ir" / name).exists()
    assert not list((ROOT / "tests/passes").glob("*.py"))


def test_smt_utilities_do_not_depend_on_proof_drivers():
    tree = ast.parse((ROOT / "ir/smt.py").read_text())
    imports = {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
    assert not imports & {
        "proof_preparation", "regional_obligations", "regions", "relational_dataflow",
        "relational_verifier", "relational_artifact",
    }
