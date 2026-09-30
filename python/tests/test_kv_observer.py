import hashlib
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

from scripts.determinism_tests.kv_observer import (
    load_kv_observer_records,
    observe_kv_store,
)


class KvObserverTests(unittest.TestCase):
    def test_records_active_inputs_and_physical_rows(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            k = torch.tensor(
                [[[1.0, 2.0]], [[9.0, 9.0]], [[3.0, 4.0]]],
                dtype=torch.bfloat16,
            )
            v = k + 10
            k_cache = torch.zeros((2, 2, 1, 2), dtype=torch.bfloat16)
            v_cache = torch.zeros_like(k_cache)
            slots = torch.tensor([0, -1, 3], dtype=torch.int32)
            k_cache.reshape(-1, 1, 2)[[0, 3]] = k[[0, 2]]
            v_cache.reshape(-1, 1, 2)[[0, 3]] = v[[0, 2]]

            with mock.patch.dict(
                os.environ, {"VOSTI_KV_OBSERVER_DIR": raw}, clear=False
            ):
                observe_kv_store(
                    k,
                    v,
                    k_cache,
                    v_cache,
                    slots,
                    source="unit-test",
                )

            records = load_kv_observer_records(Path(raw))
            self.assertEqual(len(records), 1)
            record = records[0]
            self.assertEqual(record["source"], "unit-test")
            for name in ("input_k", "input_v", "stored_k", "stored_v", "slots"):
                descriptor = record[name]
                array = np.load(Path(raw) / descriptor["artifact"], allow_pickle=False)
                self.assertEqual(descriptor["shape"], list(array.shape))
                self.assertEqual(descriptor["dtype"], str(array.dtype))
                self.assertEqual(
                    descriptor["sha256"],
                    hashlib.sha256(array.tobytes(order="C")).hexdigest(),
                )
            np.testing.assert_array_equal(
                np.load(Path(raw) / record["slots"]["artifact"]),
                np.array([0, 3], dtype=np.int64),
            )
            np.testing.assert_array_equal(
                np.load(Path(raw) / record["input_k"]["artifact"]),
                np.load(Path(raw) / record["stored_k"]["artifact"]),
            )

    def test_rejects_duplicate_active_slots(self) -> None:
        with tempfile.TemporaryDirectory() as raw, mock.patch.dict(
            os.environ, {"VOSTI_KV_OBSERVER_DIR": raw}, clear=False
        ):
            tensor = torch.zeros((2, 1, 2), dtype=torch.bfloat16)
            cache = torch.zeros((1, 2, 1, 2), dtype=torch.bfloat16)
            with self.assertRaisesRegex(RuntimeError, "injective"):
                observe_kv_store(
                    tensor,
                    tensor.clone(),
                    cache,
                    cache.clone(),
                    torch.tensor([0, 0], dtype=torch.int32),
                    source="unit-test",
                )


if __name__ == "__main__":
    unittest.main()
