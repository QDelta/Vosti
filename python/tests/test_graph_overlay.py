import json
import unittest

import torch

from vosti_kernels.graph_overlay import CudaGraphOverlay, GraphSignature


class CudaGraphOverlayProtocolTests(unittest.TestCase):
    def test_warm_then_capture_is_per_instance(self):
        first = CudaGraphOverlay()
        second = CudaGraphOverlay()
        signature = (1, 64, 64, 1, 4)

        self.assertEqual(first.probe(*signature), CudaGraphOverlay.EAGER)
        self.assertEqual(first.probe(*signature), CudaGraphOverlay.CAPTURE)
        self.assertEqual(second.probe(*signature), CudaGraphOverlay.EAGER)

        first_stats = json.loads(first.stats_json())
        second_stats = json.loads(second.stats_json())
        self.assertEqual(first_stats["warmed_count"], 1)
        self.assertEqual(second_stats["warmed_count"], 1)
        self.assertEqual(first_stats["graph_count"], 0)
        self.assertEqual(second_stats["graph_count"], 0)

    def test_empty_schedule_stays_eager(self):
        overlay = CudaGraphOverlay()
        self.assertEqual(
            overlay.probe(0, 0, 0, 0, 0),
            CudaGraphOverlay.EAGER,
        )
        self.assertEqual(
            overlay.probe(0, 0, 0, 0, 0),
            CudaGraphOverlay.EAGER,
        )

    def test_pure_prefill_policy_stays_eager(self):
        overlay = CudaGraphOverlay()
        signature = (0, 4096, 4, 1024, 16)
        self.assertEqual(overlay.probe(*signature), CudaGraphOverlay.EAGER)
        self.assertEqual(overlay.probe(*signature), CudaGraphOverlay.EAGER)
        stats = json.loads(overlay.stats_json())
        self.assertEqual(stats["graph_count"], 0)
        self.assertEqual(stats["warmed_count"], 0)

    def test_negative_signature_is_rejected(self):
        overlay = CudaGraphOverlay()
        with self.assertRaises(ValueError):
            overlay.probe(1, 64, 64, 1, -1)

    def test_pure_decode_uses_smallest_covering_capture(self):
        overlay = CudaGraphOverlay()
        large = GraphSignature(1, 64, 64, 1, 4)
        medium = GraphSignature(1, 32, 32, 1, 3)
        overlay._graphs[large] = {}
        overlay._graphs[medium] = {}

        actual = GraphSignature(1, 17, 17, 1, 2)
        self.assertEqual(overlay._covering_signature(actual), medium)
        self.assertEqual(overlay.probe(*actual.as_tuple()), CudaGraphOverlay.REPLAY)

    def test_disabled_cover_policy_does_not_reuse_a_padded_decode_graph(self):
        overlay = CudaGraphOverlay()
        captured = GraphSignature(1, 64, 64, 1, 4)
        actual = GraphSignature(1, 17, 17, 1, 2)
        overlay._graphs[captured] = {}

        self.assertIsNone(
            overlay._covering_signature(actual, allow_decode_cover=False)
        )
        self.assertEqual(
            overlay.probe(*actual.as_tuple(), allow_decode_cover=False),
            CudaGraphOverlay.EAGER,
        )
        self.assertEqual(
            overlay.probe(*actual.as_tuple(), allow_decode_cover=False),
            CudaGraphOverlay.CAPTURE,
        )

    def test_same_batch_wider_table_is_not_a_cover(self):
        overlay = CudaGraphOverlay()
        wider = GraphSignature(1, 16, 16, 1, 9)
        actual = GraphSignature(1, 16, 16, 1, 5)
        overlay._graphs[wider] = {}

        self.assertIsNone(overlay._covering_signature(actual))
        self.assertEqual(overlay.probe(*actual.as_tuple()), CudaGraphOverlay.EAGER)
        self.assertEqual(overlay.probe(*actual.as_tuple()), CudaGraphOverlay.CAPTURE)

    def test_covering_does_not_cross_modes_or_capture_mixed(self):
        overlay = CudaGraphOverlay()
        overlay._graphs[GraphSignature(1, 64, 64, 1, 4)] = {}
        mixed = (2, 32, 16, 4, 2)
        self.assertEqual(overlay.probe(*mixed), CudaGraphOverlay.EAGER)
        self.assertEqual(overlay.probe(*mixed), CudaGraphOverlay.EAGER)
        stats = json.loads(overlay.stats_json())
        self.assertEqual(stats["graph_count"], 1)
        self.assertEqual(stats["warmed_count"], 0)

    def test_decode_cover_installs_inert_padding(self):
        actual = GraphSignature(1, 2, 2, 1, 2)
        captured_signature = GraphSignature(1, 4, 4, 1, 3)
        buffers = {
            "input_ids": torch.full((4,), 99, dtype=torch.int32),
            "positions": torch.full((4,), 99, dtype=torch.int64),
            "slot_mapping": torch.full((4,), 99, dtype=torch.int32),
            "cu_seqlens_q": torch.full((5,), 99, dtype=torch.int32),
            "cu_seqlens_k": torch.full((5,), 99, dtype=torch.int32),
            "block_table": torch.full((4, 3), 99, dtype=torch.int32),
        }
        entry = {
            "inputs": buffers,
            "decode_cu_q_template": torch.arange(5, dtype=torch.int32),
            "decode_pad_offsets": torch.arange(1, 5, dtype=torch.int32),
            "cover_fill_state": None,
        }
        current = {
            "input_ids": torch.tensor([7, 8], dtype=torch.int32),
            "positions": torch.tensor([11, 12], dtype=torch.int64),
            "slot_mapping": torch.tensor([2, 3], dtype=torch.int32),
            "cu_seqlens_q": torch.tensor([0, 1, 2], dtype=torch.int32),
            "cu_seqlens_k": torch.tensor([0, 5, 12], dtype=torch.int32),
            "block_table": torch.tensor([[4, 5], [6, 7]], dtype=torch.int32),
        }

        CudaGraphOverlay._install_decode_cover_inputs(
            entry, current, actual, captured_signature
        )

        self.assertEqual(buffers["input_ids"].tolist(), [7, 8, 0, 0])
        self.assertEqual(buffers["positions"].tolist(), [11, 12, 0, 0])
        self.assertEqual(buffers["slot_mapping"].tolist(), [2, 3, -1, -1])
        self.assertEqual(buffers["cu_seqlens_q"].tolist(), [0, 1, 2, 3, 4])
        self.assertEqual(buffers["cu_seqlens_k"].tolist(), [0, 5, 12, 13, 14])
        self.assertEqual(
            buffers["block_table"].tolist(),
            [[4, 5, 0], [6, 7, 0], [0, 0, 0], [0, 0, 0]],
        )

        # The shape-static padding remains installed, but K endpoints must
        # track the new real cumulative endpoint on every replay.
        current["cu_seqlens_k"] = torch.tensor(
            [0, 6, 14], dtype=torch.int32
        )
        CudaGraphOverlay._install_decode_cover_inputs(
            entry, current, actual, captured_signature
        )
        self.assertEqual(buffers["cu_seqlens_k"].tolist(), [0, 6, 14, 15, 16])

    def test_exact_replay_invalidates_cached_cover_padding(self):
        actual = GraphSignature(1, 2, 2, 1, 2)
        captured_signature = GraphSignature(1, 4, 4, 1, 3)
        buffers = {
            "input_ids": torch.full((4,), 99, dtype=torch.int32),
            "positions": torch.full((4,), 99, dtype=torch.int64),
            "slot_mapping": torch.full((4,), 99, dtype=torch.int32),
            "cu_seqlens_q": torch.arange(5, dtype=torch.int32),
            "cu_seqlens_k": torch.arange(5, dtype=torch.int32),
            "block_table": torch.full((4, 3), 99, dtype=torch.int32),
        }
        entry = {
            "inputs": buffers,
            "decode_cu_q_template": torch.arange(5, dtype=torch.int32),
            "decode_pad_offsets": torch.arange(1, 5, dtype=torch.int32),
            "cover_fill_state": None,
        }
        cover = {
            "input_ids": torch.tensor([7, 8], dtype=torch.int32),
            "positions": torch.tensor([11, 12], dtype=torch.int64),
            "slot_mapping": torch.tensor([2, 3], dtype=torch.int32),
            "cu_seqlens_q": torch.tensor([0, 1, 2], dtype=torch.int32),
            "cu_seqlens_k": torch.tensor([0, 5, 12], dtype=torch.int32),
            "block_table": torch.tensor([[4, 5], [6, 7]], dtype=torch.int32),
        }
        CudaGraphOverlay._install_decode_cover_inputs(
            entry, cover, actual, captured_signature
        )
        self.assertEqual(entry["cover_fill_state"], (2, 2))

        exact = {
            "input_ids": torch.tensor([1, 2, 3, 4], dtype=torch.int32),
            "positions": torch.tensor([1, 2, 3, 4], dtype=torch.int64),
            "slot_mapping": torch.tensor([20, 21, 22, 23], dtype=torch.int32),
            "cu_seqlens_q": torch.arange(5, dtype=torch.int32),
            "cu_seqlens_k": torch.tensor([0, 4, 8, 12, 16], dtype=torch.int32),
            "block_table": torch.tensor(
                [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]],
                dtype=torch.int32,
            ),
        }
        CudaGraphOverlay._install_exact_inputs(entry, exact)
        self.assertIsNone(entry["cover_fill_state"])

        CudaGraphOverlay._install_decode_cover_inputs(
            entry, cover, actual, captured_signature
        )
        self.assertEqual(buffers["slot_mapping"].tolist(), [2, 3, -1, -1])
        self.assertEqual(
            buffers["block_table"].tolist(),
            [[4, 5, 0], [6, 7, 0], [0, 0, 0], [0, 0, 0]],
        )


if __name__ == "__main__":
    unittest.main()
