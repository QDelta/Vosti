"""Keep the shared kernel verifier separate from optional diagnostics."""

from __future__ import annotations

import ast
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]


def _imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(), filename=str(path))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
        elif isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
    return imported


class VerifierPassLayoutTests(unittest.TestCase):
    def test_structural_driver_does_not_import_optional_passes(self) -> None:
        imports = _imported_modules(
            ROOT / "scripts/verification" / "verify_kernel_contracts.py"
        )
        self.assertNotIn("diagnostics.race", imports)
        self.assertNotIn("diagnostics.suffix_independence", imports)

    def test_project_gates_keep_optional_drivers_separate(self) -> None:
        from unittest.mock import patch
        from scripts.project import Gates

        runner = Gates()
        with patch.object(runner, "python") as invoke:
            runner.verify_kernels()
            self.assertEqual([call.args[0] for call in invoke.call_args_list],
                             ["scripts/audit/check_kernel_sources.py", "scripts/verification/verify_kernel_contracts.py"])
        with patch.object(runner, "python") as invoke:
            runner.diagnostics()
            paths = {Path(call.args[0]).name for call in invoke.call_args_list}
            race_drivers = {path.name for path in
                            (ROOT / "scripts/verification/diagnostics").glob("verify_*_kernel_races.py")}
            self.assertTrue(race_drivers)
            self.assertTrue(race_drivers <= paths)
            self.assertIn("verify_suffix_independence.py", paths)
