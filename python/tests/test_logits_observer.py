import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

from scripts.determinism_tests.logits_observer import (
    observe_logits_row,
    observe_logits_rows,
)


class LogitsObserverTests(unittest.TestCase):
    def test_structured_rows_are_lossless_and_indexed(self):
        with tempfile.TemporaryDirectory() as raw:
            with mock.patch.dict(
                os.environ,
                {
                    "VOSTI_LOGITS_OBSERVER_DIR": raw,
                    "VOSTI_LOGITS_OBSERVER_PHASE": "measured",
                    "VOSTI_LOGITS_OBSERVER_STEP": "3:7",
                },
                clear=False,
            ):
                observe_logits_rows(
                    torch.tensor([[1.0, 3.0], [-2.0, 4.0]], dtype=torch.bfloat16),
                    source="unit-test",
                )
            lines = []
            for path in Path(raw).glob("observer-*.jsonl"):
                lines.extend(path.read_text().splitlines())
            records = [json.loads(line) for line in lines]
            self.assertEqual([row["batch_index"] for row in records], [0, 1])
            self.assertEqual({row["source"] for row in records}, {"unit-test"})
            self.assertEqual({row["phase"] for row in records}, {"measured"})
            self.assertEqual({row["engine_step"] for row in records}, {"3:7"})
            arrays = [np.load(Path(raw) / row["artifact"]) for row in records]
            np.testing.assert_array_equal(arrays[0], np.array([1.0, 3.0], np.float32))
            np.testing.assert_array_equal(arrays[1], np.array([-2.0, 4.0], np.float32))
            self.assertTrue(all(row["finite"] for row in records))
            self.assertEqual([row["argmax"] for row in records], [1, 1])

    def test_single_row_digest_preserves_logical_batch_index(self):
        with tempfile.TemporaryDirectory() as raw:
            digest = Path(raw) / "digests.txt"
            with mock.patch.dict(
                os.environ, {"VOSTI_LOGITS_DIGEST": str(digest)}, clear=False
            ):
                observe_logits_row(
                    torch.tensor([1.0, 2.0]), 7, source="unit-test-single"
                )
            self.assertEqual(digest.read_text().split()[0], "7")


if __name__ == "__main__":
    unittest.main()
