import hashlib
import json
import tempfile
from pathlib import Path
import unittest

import numpy as np

from scripts.checks.reference import (
    FAMILIES,
    _checkpoint_manifest,
    _checkpoint_unchanged,
    _combine_error_summaries,
    _load_array,
    _load_single_observed_row,
    _logit_error_summary,
    _read_outputs,
    _read_single_output,
)


class ModelReferenceTests(unittest.TestCase):
    def test_every_supported_family_has_one_reference_configuration(self) -> None:
        self.assertEqual(set(FAMILIES), {"qwen3", "gemma3", "gemma4", "llama3"})
        for family in FAMILIES.values():
            self.assertEqual(
                set(family),
                {"architecture", "default_model", "default_prompt"},
            )
            self.assertTrue(family["default_prompt"])

    def test_single_output_parser_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tokens.tsv"
            path.write_text("0\t17\n", encoding="utf-8")
            self.assertEqual(_read_single_output(path), 17)
            path.write_text("0\t17,18\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "not one token"):
                _read_single_output(path)

    def test_multi_output_parser_orders_and_validates_indexes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tokens.tsv"
            path.write_text("1\t19\n0\t17,18\n", encoding="utf-8")
            self.assertEqual(_read_outputs(path), [[17, 18], [19]])
            path.write_text("2\t19\n0\t17\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "not contiguous"):
                _read_outputs(path)

    def test_logit_error_summary_checks_shape_dtype_and_finiteness(self) -> None:
        reference = np.array([1.0, 2.0, -3.0], dtype=np.float32)
        actual = np.array([1.25, 1.5, -3.0], dtype=np.float32)
        self.assertEqual(
            _logit_error_summary(reference, actual),
            {"max_abs_error": 0.5, "mean_abs_error": 0.25},
        )
        with self.assertRaisesRegex(RuntimeError, "shapes differ"):
            _logit_error_summary(reference, actual[:2])
        with self.assertRaisesRegex(RuntimeError, "float32"):
            _logit_error_summary(reference, actual.astype(np.float64))
        with self.assertRaisesRegex(RuntimeError, "finite"):
            _logit_error_summary(
                reference, np.array([1.0, np.inf, -3.0], dtype=np.float32)
            )

    def test_combined_error_summary_weights_every_element(self) -> None:
        left = np.array([0.0], dtype=np.float32)
        right = np.array([1.0], dtype=np.float32)
        zeros = np.zeros(3, dtype=np.float32)
        self.assertEqual(
            _combine_error_summaries(
                [(left, right), (zeros, zeros)], label="unit"
            ),
            {"max_abs_error": 1.0, "mean_abs_error": 0.25},
        )

    def test_array_loader_checks_digest_shape_and_dtype(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            array = np.array([[1.0, -0.0]], dtype=np.float32)
            path = root / "array.npy"
            np.save(path, array, allow_pickle=False)
            descriptor = {
                "artifact": path.name,
                "shape": [1, 2],
                "dtype": "float32",
                "sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest(),
            }
            np.testing.assert_array_equal(
                _load_array(root, descriptor, label="unit"), array
            )
            descriptor["sha256"] = "0" * 64
            with self.assertRaisesRegex(RuntimeError, "digest"):
                _load_array(root, descriptor, label="unit")

    def test_observed_row_loader_validates_source_and_digest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            row = np.array([1.0, -0.0, 3.0], dtype=np.float32)
            artifact = root / "row.npy"
            np.save(artifact, row, allow_pickle=False)
            digest = hashlib.sha256(row.tobytes(order="C")).hexdigest()
            record = {
                "pid": 1,
                "call_sequence": 0,
                "batch_index": 0,
                "source": "vosti.sampling_boundary",
                "artifact": artifact.name,
                "sha256": digest,
            }
            (root / "observer-1.jsonl").write_text(
                json.dumps(record) + "\n", encoding="utf-8"
            )
            np.testing.assert_array_equal(_load_single_observed_row(root), row)
            record["source"] = "other"
            (root / "observer-1.jsonl").write_text(
                json.dumps(record) + "\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(RuntimeError, "unexpected.*source"):
                _load_single_observed_row(root)

    def test_checkpoint_manifest_binds_weights_and_detects_changes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "config.json").write_text('{"model_type":"unit"}\n')
            weight = root / "model.safetensors"
            weight.write_bytes(b"unit-weights")
            manifest = _checkpoint_manifest(root)
            self.assertEqual(
                [row["name"] for row in manifest["files"]],
                ["model.safetensors", "config.json"],
            )
            _checkpoint_unchanged(root, manifest)
            weight.write_bytes(b"changed-unit-weights")
            with self.assertRaisesRegex(RuntimeError, "changed"):
                _checkpoint_unchanged(root, manifest)

    def test_checkpoint_manifest_accepts_root_alias_but_rejects_escaped_weight(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory).resolve()
            root = base / "checkpoint"
            root.mkdir()
            (root / "config.json").write_text('{"model_type":"unit"}\n')
            (root / "model.safetensors").write_bytes(b"weights")
            alias = base / "alias"
            alias.symlink_to(root, target_is_directory=True)
            self.assertEqual(_checkpoint_manifest(alias), _checkpoint_manifest(root))
            external = base / "outside.safetensors"
            external.write_bytes(b"outside")
            (root / "model-escape.safetensors").symlink_to(external)
            with self.assertRaisesRegex(RuntimeError, "escapes"):
                _checkpoint_manifest(alias)


if __name__ == "__main__":
    unittest.main()
