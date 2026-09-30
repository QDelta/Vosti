from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from scripts.checks.engine import read_output_tokens, validate_smoke_result


class ModelEngineSmokeTests(unittest.TestCase):
    def test_accepts_complete_common_runtime_and_output_record(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tokens.tsv"
            path.write_text("0\t11,12\n", encoding="utf-8")
            outputs = validate_smoke_result(
                architecture="qwen3",
                expected_requests=1,
                expected_tokens_per_request=2,
                stdout="MODEL_RUNTIME architecture=qwen3 backend_qualified=true\n",
                output_path=path,
            )
        self.assertEqual(outputs, [[11, 12]])

    def test_rejects_missing_runtime_identity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tokens.tsv"
            path.write_text("0\t11,12\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "lacks exactly one"):
                validate_smoke_result(
                    architecture="gemma3_text",
                    expected_requests=1,
                    expected_tokens_per_request=2,
                    stdout="",
                    output_path=path,
                )

    def test_rejects_noncontiguous_request_indexes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tokens.tsv"
            path.write_text("1\t11,12\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "non-contiguous"):
                read_output_tokens(path)

    def test_rejects_incomplete_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tokens.tsv"
            path.write_text("0\t11\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "output lengths"):
                validate_smoke_result(
                    architecture="gemma3_text",
                    expected_requests=1,
                    expected_tokens_per_request=2,
                    stdout=(
                        "MODEL_RUNTIME architecture=gemma3_text "
                        "backend_qualified=true\n"
                    ),
                    output_path=path,
                )
