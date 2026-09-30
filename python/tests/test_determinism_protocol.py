from scripts.common.model_paths import checkpoint_path
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from scripts.determinism_tests.protocol import (
    EXECUTION_CONFIGS,
    ALL_EXECUTION_CONFIGS,
    CAMPAIGN_SEEDS,
    CHECKPOINTS,
    MODELS,
    ROW_COMPARISON_EQUALITY,
    ROW_COMPARISON_VERSION,
    TESTS,
    build_inputs,
    compare_row_artifacts,
    logical_matrix,
    _model_token_domain,
)
from scripts.determinism_tests.run import (
    _retained_arm_is_reusable,
    _selected_backend_from_logs,
)
from scripts.determinism_tests.sglang_worker import _rows_at_generated_position
from scripts.common.process_lifecycle import (
    wait_for_child_exit_without_reaping,
)
from scripts.determinism_tests.rank_divergence import ranking_divergence
from scripts.determinism_tests.vllm_worker import (
    _requires_language_model_only,
    _resolved_vocab_size,
    _text_projection_engine_overrides,
)
from scripts.determinism_tests.vosti_worker import (
    _disjoint_warmup,
    _final_records,
    _invocation_plan,
    _parse_output_tokens,
    _records_at_position,
    _wait_for_nvml_release,
)


class DeterminismProtocolTests(unittest.TestCase):
    def test_catalog_matrix_has_588_relations(self):
        matrix = logical_matrix()
        self.assertEqual(len(matrix), 588)
        self.assertEqual(sum(row["strict"] for row in matrix), 420)
        self.assertEqual(len(CHECKPOINTS), 7)
        self.assertEqual(len({row["seed"] for row in matrix}), 3)
        self.assertIn("sglang-deterministic-fa3", {config.key for config in EXECUTION_CONFIGS})
        self.assertNotIn("sglang-deterministic-fa4", {config.key for config in EXECUTION_CONFIGS})
        self.assertIn("sglang-deterministic-fa4", {config.key for config in ALL_EXECUTION_CONFIGS})
        fa3_rows = [row for row in matrix if row["execution_config"] == "sglang-deterministic-fa3"]
        self.assertEqual({row["checkpoint"] for row in fa3_rows}, {checkpoint.key for checkpoint in CHECKPOINTS})
        fa3_config = next(config for config in EXECUTION_CONFIGS if config.key == "sglang-deterministic-fa3")
        self.assertEqual(fa3_config.attention_backend, "fa3")
        self.assertEqual(len(EXECUTION_CONFIGS), 7)
        self.assertEqual(len(MODELS), 4)
        self.assertEqual(len(TESTS), 4)
        self.assertEqual(MODELS[0].key, "qwen3")
        self.assertEqual(MODELS[0].default_path, checkpoint_path("Qwen3-8B"))
        self.assertEqual(MODELS[1].key, "gemma3")
        self.assertEqual(MODELS[2].key, "llama3")
        self.assertEqual(
            MODELS[2].default_path, checkpoint_path("Llama-3.1-8B")
        )
        self.assertEqual(MODELS[3].key, "gemma4")
        self.assertEqual(MODELS[3].default_path, checkpoint_path("gemma-4-31b-it"))
        self.assertTrue(MODELS[3].language_model_only)

    def test_single_seed_matrix_has_196_h200_relations(self):
        matrix = logical_matrix(hardware=("h200",), seeds=CAMPAIGN_SEEDS[:1])
        self.assertEqual(len(matrix), 196)
        self.assertEqual(sum(row["strict"] for row in matrix), 140)
        self.assertEqual({row["hardware"] for row in matrix}, {"h200"})

    def test_matrix_rejects_unknown_hardware(self):
        with self.assertRaisesRegex(ValueError, "unknown.*hardware"):
            logical_matrix(hardware=("future-gpu",))

    def test_prompt_generation_is_reproducible_and_excludes_special_ids(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            (root / "config.json").write_text(
                json.dumps({"model_type": "test", "vocab_size": 100})
            )
            (root / "generation_config.json").write_text(
                json.dumps({"bos_token_id": 2, "eos_token_id": [1], "pad_token_id": 0})
            )
            (root / "tokenizer_config.json").write_text(
                json.dumps({"added_tokens_decoder": {"3": {}}})
            )
            first = build_inputs(MODELS[0], root)
            second = build_inputs(MODELS[0], root)
            self.assertEqual(first, second)
            other = build_inputs(MODELS[0], root, seed=CAMPAIGN_SEEDS[1])
            self.assertNotEqual(first["token_ids_sha256"], other["token_ids_sha256"])
            self.assertEqual(other["seed"], CAMPAIGN_SEEDS[1])
            self.assertEqual([len(row) for row in first["batch_prompts"]], [17, 63, 64, 65, 127, 129, 257, 385])
            self.assertTrue(all(row[0] == 2 for row in first["batch_prompts"]))
            forbidden = {0, 1, 2, 3}
            self.assertTrue(
                all(token not in forbidden for row in first["batch_prompts"] for token in row[1:])
            )

    def test_tokenizer_vocab_fallback_supports_sparse_model_config(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            (root / "config.json").write_text(
                json.dumps({"text_config": {"model_type": "sparse-test"}})
            )
            (root / "generation_config.json").write_text(
                json.dumps({"bos_token_id": 2, "eos_token_id": 1, "pad_token_id": 0})
            )
            (root / "tokenizer_config.json").write_text(
                json.dumps({"added_tokens_decoder": {"5": {}}})
            )
            (root / "tokenizer.json").write_text(
                json.dumps({"model": {"vocab": {str(index): index for index in range(4)}}})
            )
            vocab_size, bos_token_id, excluded = _model_token_domain(root)
            self.assertEqual(vocab_size, 6)
            self.assertEqual(bos_token_id, 2)
            self.assertEqual(excluded, {0, 1, 2, 5})

    def test_row_comparison_reports_exact_mismatch(self):
        with tempfile.TemporaryDirectory() as raw:
            left = Path(raw) / "left.npy"
            right = Path(raw) / "right.npy"
            np.save(left, np.array([1.0, 2.0, 4.0], dtype=np.float32))
            np.save(right, np.array([1.0, 3.0, 4.0], dtype=np.float32))
            result = compare_row_artifacts(left, right)
            self.assertFalse(result["bitwise_equal"])
            self.assertEqual(result["mismatch_count"], 1)
            self.assertEqual(result["first_mismatch_index"], 1)
            self.assertEqual(result["max_abs_diff"], 1.0)

    def test_row_comparison_distinguishes_signed_zero_bits(self):
        with tempfile.TemporaryDirectory() as raw:
            left = Path(raw) / "left.npy"
            right = Path(raw) / "right.npy"
            np.save(left, np.array([0.0], dtype=np.float32))
            np.save(right, np.array([-0.0], dtype=np.float32))
            result = compare_row_artifacts(left, right)
            self.assertFalse(result["bitwise_equal"])
            self.assertEqual(result["mismatch_count"], 1)
            self.assertEqual(result["first_mismatch_index"], 0)
            self.assertEqual(result["max_abs_diff"], 0.0)
            self.assertEqual(
                result["comparison_semantics"],
                {
                    "version": ROW_COMPARISON_VERSION,
                    "equality": ROW_COMPARISON_EQUALITY,
                    "mismatch_count_unit": "elements",
                },
            )

    def test_row_comparison_rejects_equal_values_with_different_dtypes(self):
        with tempfile.TemporaryDirectory() as raw:
            left = Path(raw) / "left.npy"
            right = Path(raw) / "right.npy"
            np.save(left, np.array([1.0], dtype=np.float32))
            np.save(right, np.array([1.0], dtype=np.float64))
            result = compare_row_artifacts(left, right)
            self.assertFalse(result["dtype_equal"])
            self.assertFalse(result["bitwise_equal"])
            self.assertIsNone(result["mismatch_count"])

    def test_sglang_selects_row_by_generated_token_position(self):
        records = [
            {
                "batch_index": 0,
                "batch_size": 1,
                "argmax": 41,
                "metadata": {"token_position": 62},
            },
            {
                "batch_index": 0,
                "batch_size": 1,
                "argmax": 99,
                "metadata": {"token_position": 63},
            },
        ]
        selected = _rows_at_generated_position(
            records,
            prompts=[list(range(63))],
            output_token_ids=[[41]],
            generated_position=0,
        )
        self.assertEqual(selected, [records[0]])

    def test_sglang_rejects_row_that_did_not_produce_returned_token(self):
        records = [{
            "batch_index": 0,
            "batch_size": 1,
            "argmax": 41,
            "metadata": {"token_position": 62},
        }]
        with self.assertRaisesRegex(RuntimeError, "returned greedy token"):
            _rows_at_generated_position(
                records,
                prompts=[list(range(63))],
                output_token_ids=[[42]],
                generated_position=0,
            )

    @patch("scripts.common.process_lifecycle.os.waitid")
    def test_gpu_child_exit_wait_retains_process_identity(self, waitid):
        waitid.side_effect = [None, SimpleNamespace(si_pid=17)]
        wait_for_child_exit_without_reaping(17, timeout_s=1.0, poll_s=0.0)
        self.assertEqual(waitid.call_count, 2)
        for call in waitid.call_args_list:
            self.assertTrue(call.args[2] & os.WNOWAIT)

    def test_rank_divergence_reports_same_token_with_different_logit(self):
        left = np.array([3.0, 2.0, 1.0], dtype=np.float32)
        right = np.array([4.0, 2.0, 1.0], dtype=np.float32)
        result = ranking_divergence(left, right)
        self.assertEqual(result["first_divergent_rank"], 1)
        self.assertEqual(result["divergence_reason"], "logit_only")
        self.assertFalse(result["token_differs_at_first_divergence"])
        self.assertFalse(result["top_token_differs"])

    def test_rank_divergence_distinguishes_signed_zero_bits(self):
        left = np.array([1.0, 0.0], dtype=np.float32)
        right = np.array([1.0, -0.0], dtype=np.float32)
        result = ranking_divergence(left, right)
        self.assertFalse(result["ordered_pairs_identical"])
        self.assertEqual(result["first_divergent_rank"], 2)
        self.assertEqual(result["divergence_reason"], "logit_only")
        self.assertFalse(result["token_differs_at_first_divergence"])
        self.assertFalse(result["top_token_differs"])

    def test_rank_divergence_reports_first_different_ranked_token(self):
        left = np.array([3.0, 2.0, 1.0], dtype=np.float32)
        right = np.array([3.0, 1.0, 2.5], dtype=np.float32)
        result = ranking_divergence(left, right)
        self.assertEqual(result["first_divergent_rank"], 2)
        self.assertEqual(result["divergence_reason"], "token_and_logit")
        self.assertTrue(result["token_differs_at_first_divergence"])
        self.assertFalse(result["top_token_differs"])
        self.assertEqual((result["left_token_id"], result["right_token_id"]), (1, 2))

    def test_rank_divergence_uses_token_id_to_break_logit_ties(self):
        left = np.array([3.0, 2.0, 2.0], dtype=np.float32)
        right = np.array([3.0, 1.5, 2.0], dtype=np.float32)
        result = ranking_divergence(left, right)
        self.assertEqual(result["first_divergent_rank"], 2)
        self.assertTrue(result["tie_affected"])

    def test_rank_divergence_accepts_identical_ordered_pairs(self):
        row = np.array([3.0, 2.0, 1.0], dtype=np.float32)
        result = ranking_divergence(row, row.copy())
        self.assertTrue(result["ordered_pairs_identical"])
        self.assertIsNone(result["first_divergent_rank"])

    def test_selected_backend_is_extracted_from_engine_logs(self):
        self.assertEqual(
            _selected_backend_from_logs(
                "vllm",
                "Using FLASH_ATTN attention backend out of potential backends",
                "",
            )["selected_attention_backend"],
            "FLASH_ATTN",
        )
        self.assertEqual(
            _selected_backend_from_logs(
                "sglang",
                "",
                "server_args=ServerArgs(attention_backend='fa3')",
            )["selected_attention_backend"],
            "fa3",
        )

    def test_resume_skips_rejected_telemetry(self):
        with tempfile.TemporaryDirectory() as raw:
            original = Path(raw)
            arm = {"engine": {}, "payload": "fixed"}
            (original / "arm.json").write_text(json.dumps(arm))
            (original / "telemetry.json").write_text(
                json.dumps({"status": "invalid_contended"})
            )
            self.assertFalse(
                _retained_arm_is_reusable(
                    original / "arm.json",
                    original / "telemetry.json",
                    arm,
                    "test-arm",
                )
            )
            (original / "telemetry.json").write_text(
                json.dumps({"status": "complete"})
            )
            self.assertTrue(
                _retained_arm_is_reusable(
                    original / "arm.json",
                    original / "telemetry.json",
                    arm,
                    "test-arm",
                )
            )

    def test_vosti_warmup_preserves_shapes_and_changes_cache_pages(self):
        prompts = [list(range(80)), list(range(100, 180))]
        warmup = _disjoint_warmup(prompts, 1000)
        self.assertEqual([len(row) for row in warmup], [80, 80])
        self.assertTrue(
            {tuple(row[:64]) for row in prompts}.isdisjoint(
                {tuple(row[:64]) for row in warmup}
            )
        )

    def test_vllm_uses_loaded_padded_output_vocabulary(self):
        model_config = SimpleNamespace(get_vocab_size=lambda: 262208)
        llm = SimpleNamespace(
            llm_engine=SimpleNamespace(
                vllm_config=SimpleNamespace(model_config=model_config)
            )
        )
        self.assertEqual(_resolved_vocab_size(llm), 262208)

    def test_vllm_text_projection_disables_outer_multimodal_runtime(self):
        gemma = {
            "architecture": "renaming-does-not-control-policy",
            "language_model_only": True,
        }
        qwen = {
            "architecture": "also-not-semantic",
            "language_model_only": False,
        }
        self.assertTrue(_requires_language_model_only(gemma))
        self.assertEqual(
            _text_projection_engine_overrides(gemma),
            {
                "language_model_only": True,
                "hf_overrides": {"is_mm_prefix_lm": False},
            },
        )
        self.assertFalse(_requires_language_model_only(qwen))
        self.assertEqual(_text_projection_engine_overrides(qwen), {})
        with self.assertRaisesRegex(RuntimeError, "explicit boolean"):
            _requires_language_model_only({"architecture": "anything.text_model"})

    def test_vosti_output_token_record_parser(self):
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "tokens.tsv"
            path.write_text("0\t3,4\n1\t5\n", encoding="utf-8")
            self.assertEqual(_parse_output_tokens(path), [[3, 4], [5]])

    @patch("scripts.determinism_tests.vosti_worker._compute_process_pids")
    def test_vosti_waits_for_nvml_to_release_exited_child(self, compute_pids):
        compute_pids.side_effect = [{17}, {17}, set(), set()]
        _wait_for_nvml_release(17, timeout_s=1.0, poll_s=0.0, settle_s=0.0)
        self.assertEqual(compute_pids.call_count, 4)

    @patch("scripts.determinism_tests.vosti_worker._compute_process_pids")
    def test_vosti_nvml_release_timeout_fails_closed(self, compute_pids):
        compute_pids.return_value = {17}
        with self.assertRaisesRegex(RuntimeError, "retained.*PID 17"):
            _wait_for_nvml_release(
                17, timeout_s=0.0, poll_s=0.0, settle_s=0.0
            )

    def test_vosti_final_records_allow_non_emitting_chunk_observations(self):
        records = [
            {"batch_index": 0, "call_sequence": 0},
            {"batch_index": 0, "call_sequence": 1},
            {"batch_index": 1, "call_sequence": 2},
        ]
        final = _final_records(records, batch_size=2, max_tokens=1)
        self.assertEqual([record["batch_index"] for record in final], [0, 1])

    def test_vosti_can_retain_prefill_row_while_request_stays_live(self):
        records = [
            {"batch_index": 0, "call_sequence": 0},
            {"batch_index": 0, "call_sequence": 1},
        ]
        retained = _records_at_position(
            records, batch_size=1, max_tokens=2, position=0
        )
        self.assertEqual(retained[0]["call_sequence"], 0)

    def test_vosti_two_call_arm_shares_one_engine_lifetime(self):
        cold = {"label": "cold", "record_last_rows": True}
        warm = {"label": "warm", "record_last_rows": True}
        self.assertEqual(_invocation_plan([cold, warm]), [(1, warm, cold)])

    def test_vosti_rejects_unrepresentable_multi_call_arm(self):
        with self.assertRaisesRegex(RuntimeError, "one or two calls"):
            _invocation_plan([{}, {}, {}])


if __name__ == "__main__":
    unittest.main()
