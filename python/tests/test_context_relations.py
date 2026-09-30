import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from scripts.determinism_tests.context_relations import context_arm, lengths_for, make_inputs, run_case
from scripts.determinism_tests.protocol import EXECUTION_CONFIGS, MODELS
from scripts.determinism_tests.run import SuiteRunner


class ContextInputTests(unittest.TestCase):
    def test_lengths_come_from_checkpoint_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / 'config.json').write_text(json.dumps(dict(text_config=dict(sliding_window=1024))))
            self.assertEqual(lengths_for(path, 'window-boundary'), (1023, 1024, 1025))
            self.assertEqual(lengths_for(path, 'long-context'), (8192, 32768))
            (path / 'config.json').write_text(json.dumps(dict(sliding_window=1024, use_sliding_window=False)))
            self.assertEqual(lengths_for(path, 'window-boundary'), ())

    def test_chunk_budgets_and_cache_capacity_fit_requested_context(self):
        with tempfile.TemporaryDirectory() as directory, \
             patch('scripts.determinism_tests.context_relations.deterministic_prompt',
                   side_effect=lambda path, length, seed: [1] * length):
            inputs = make_inputs(MODELS[0], Path(directory), profile='long-context', length=32768, seed=1)
            arm = context_arm(inputs, EXECUTION_CONFIGS[0], [], caching=False, budget=512)
            self.assertTrue(all(b < len(inputs['prompt']) for b in inputs['chunk_budgets']))
            self.assertEqual(arm['engine']['max_model_len'], 32770)
            self.assertGreater(arm['engine']['num_blocks'] * 64, 32770)

    def test_worker_root_can_be_separate_from_controller(self):
        runner = SuiteRunner(output=Path('/tmp/results'), worker_python=Path('/python'),
                             gpu_index=3, execution=EXECUTION_CONFIGS[0], worker_root=Path('/tmp/frozen'))
        self.assertEqual(runner.worker_root, Path('/tmp/frozen'))

    def test_worker_launch_uses_frozen_cwd_not_just_pythonpath(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            frozen = (output / 'frozen').resolve()
            runner = SuiteRunner(output=output, worker_python=Path('/python'), gpu_index=3,
                                 execution=EXECUTION_CONFIGS[0], worker_root=frozen)
            with patch('scripts.determinism_tests.run.subprocess.run', side_effect=RuntimeError('stop before GPU')) as launch, \
                 patch.object(Path, 'symlink_to'):
                with self.assertRaisesRegex(RuntimeError, 'stop before GPU'):
                    runner.run_arm('probe', dict(engine={}))
            self.assertEqual(launch.call_args.kwargs['cwd'], frozen)
            self.assertTrue(launch.call_args.kwargs['env']['PYTHONPATH'].startswith(str(frozen)))


class FakeRunner:
    def __init__(self, output, *, hit=True, signed_zero=False):
        self.output, self.hit, self.signed_zero = output, hit, signed_zero
        self.arm_records, self.arm_inputs = [], []

    def run_arm(self, name, arm):
        directory = self.output / name
        (directory / 'rows').mkdir(parents=True)
        self.arm_inputs.append(arm)
        calls = []
        for index, call in enumerate(arm['calls']):
            row = np.array([1., 0.], dtype=np.float32)
            if self.signed_zero and name == 'chunk-4096':
                row[1] = -0.
            artifact = f'{index}.npy'
            np.save(directory / 'rows' / artifact, row)
            cached = 64 if call['label'] == 'reuse' and self.hit else 0
            calls.append(dict(requests=[dict(num_cached_tokens=cached, last_row=dict(artifact=artifact))]))
        self.arm_records.append(dict(name=name))
        return directory, dict(calls=calls)


class ContextExecutionTests(unittest.TestCase):
    def inputs(self):
        return dict(profile='long-context', length=8192, seed=1, model={}, model_path='/model',
                    prompt=[1] * 8192, chunk_budgets=[512, 2048, 4096])

    def test_real_reuse_witness_is_required_even_when_rows_match(self):
        for hit, status in ((True, 'pass'), (False, 'invalid')):
            with tempfile.TemporaryDirectory() as directory:
                runner = FakeRunner(Path(directory), hit=hit)
                result = run_case(runner, self.inputs(), EXECUTION_CONFIGS[0], hardware='h200')
                self.assertEqual(result['status'], status)
                self.assertEqual(len(runner.arm_records), 5)
                self.assertEqual([c['label'] for c in runner.arm_inputs[-1]['calls']], ['donor', 'reuse'])
                self.assertTrue(all(c['record_generated_position'] == 0 for a in runner.arm_inputs for c in a['calls']))

    def test_signed_zero_difference_is_a_chunk_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            runner = FakeRunner(Path(directory), signed_zero=True)
            result = run_case(runner, self.inputs(), EXECUTION_CONFIGS[0], hardware='h200')
            self.assertEqual(result['status'], 'mismatch')
            self.assertFalse(result['relation_pass']['different_chunk'])
            self.assertTrue(result['relation_pass']['cold_vs_warm'])
