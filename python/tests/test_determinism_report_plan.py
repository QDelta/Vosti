import unittest
import json
import tempfile
from pathlib import Path
from unittest.mock import patch
from scripts.determinism_tests.protocol import Checkpoint, CHECKPOINTS, NATIVE_CHECKPOINTS
from scripts.determinism_tests.report_plan import (
    SEED, BATCH_LENGTHS, PD_LENGTHS, OUTPUT_TOKENS, compositions, lengths, table_matrix, make_plan,
)


class ReportPlanTests(unittest.TestCase):
    def test_catalog_extension_keeps_seed_and_full_relation_coverage(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            (path/'config.json').write_text(json.dumps({'text_config': {'sliding_window': 1024}}))
            checkpoint = Checkpoint('gemma4-31b', 'gemma4', tmp)
            with patch('scripts.determinism_tests.report_plan.CHECKPOINTS', (checkpoint,)), \
                 patch('scripts.determinism_tests.report_plan.deterministic_prompt',
                       side_effect=lambda path, length, seed: [seed] * length):
                plan = make_plan('gemma4-31b')
                with self.assertRaises(ValueError):
                    make_plan('unknown')
        self.assertEqual(plan['seed'], SEED)
        self.assertEqual([len(p) for p in plan['batch_prompts']], list(BATCH_LENGTHS))
        self.assertEqual([len(p) for p in plan['pd_prompts']], list(PD_LENGTHS))
        self.assertEqual(plan['chunk_budgets'], [64, 128, 256, 512, 1024])
        self.assertEqual(plan['output_tokens'], 128)
        self.assertEqual(plan['full_budget'], 32768)

    def test_exactly_two_models_one_seed_seven_modes(self):
        self.assertIn('gemma4-12b', {c.key for c in NATIVE_CHECKPOINTS})
        self.assertNotIn('gemma4-12b', {c.key for c in CHECKPOINTS})
        matrix = table_matrix()
        self.assertEqual(len(matrix), 14)
        self.assertEqual({row['checkpoint'] for row in matrix}, {'llama3-8b', 'gemma3-4b'})
        self.assertEqual({row['seed'] for row in matrix}, {SEED})
        self.assertEqual(sum(len(row['relations']) for row in matrix), 56)
        self.assertNotIn('sglang-deterministic-fa4', {row['execution'] for row in matrix})

    def test_batch_coverage_keeps_all_32_prompts_and_changes_companions(self):
        self.assertEqual(len(BATCH_LENGTHS), 32)
        groups = compositions()
        self.assertEqual(sum(map(len, groups)), 288)
        for index in range(32):
            for size in (2, 4, 8):
                self.assertEqual(sum(index in group for group in groups if len(group) == size), 3)
        self.assertGreater(len({tuple(sorted(group)) for group in groups if len(group) == 2}), 16)

    def test_long_context_and_boundary_cases_are_not_silently_dropped(self):
        self.assertTrue({2048, 8192, 32768, 4095, 4096, 4097} <= set(lengths(4096)))
        for size in (64, 128, 256, 512, 1024):
            self.assertTrue({size-1, size, size+1} <= set(lengths()))
        self.assertEqual(PD_LENGTHS, (257, 8192, 32768))
        self.assertEqual(OUTPUT_TOKENS, 128)
