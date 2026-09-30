import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

from scripts.serving_benchmark import prepare as prepare_pilot


class PilotReplayTests(unittest.TestCase):
    def test_phase_screen_is_small_and_uses_distinct_reproducible_seeds(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(
                prepare_pilot, 'prepare_phase', side_effect=lambda _, **kw: kw) as prepare:
            jobs = prepare_pilot.phase_screen_jobs(
                object(), SimpleNamespace(path='/checkpoint'), {'config': 'hash'}, Path(directory), 42)
            rows = [json.loads(Path(job['workload']).read_text()) for job in jobs]
        self.assertEqual([row['seed'] for row in rows], [142, 143, 144])
        self.assertEqual([row['kind'] for row in rows], ['cold_prefill', 'decode', 'cached_extension'])
        self.assertTrue(all(row['waves'] == row['concurrency'] == 1 for row in rows))
        self.assertEqual([row['output'] for row in rows], [1, 64, 1])
        self.assertEqual(prepare.call_count, 3)

    def test_single_mode_reuses_jobs_without_regenerating_trace(self):
        self.check_replay('gemma3-4b', 'vosti-padded-graph')

    def test_new_dense_checkpoint_accepts_explicit_triton_baselines(self):
        for mode in ('vllm-invariant-triton-attn', 'sglang-deterministic-triton'):
            with self.subTest(mode=mode):
                self.check_replay('gemma4-31b', mode, frozen_worker=True)

    def test_extended_modes_reuse_exact_saved_jobs(self):
        for mode in ('vllm-fast-auto', 'sglang-fast-auto',
                     'vllm-invariant-fa3-local-triton-global', 'vllm-invariant-auto'):
            with self.subTest(mode=mode):
                self.check_replay('gemma4-31b', mode, frozen_worker=True)

    def check_replay(self, checkpoint, mode, frozen_worker=False):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            previous = base / 'previous.json'
            jobs = [dict(id='multi-turn', kind='multi_session_arrivals',
                         workload='/existing/measured.json', warmup='/existing/warmup.json',
                         arrival_trace='/existing/arrivals.json', request_rate=.5)]
            previous.write_text(json.dumps(jobs))
            output = base / 'candidate'
            argv = ['prepare', 'pilot', '--stack-root', str(base), '--output', str(output),
                    '--checkpoint', checkpoint, '--mode', mode, '--replay-jobs', str(previous)]
            if frozen_worker:
                argv += ['--root', str(base / 'frozen')]
            with patch('sys.argv', argv), \
                 patch.object(prepare_pilot.subprocess, 'check_output', return_value='candidate\n'), \
                 patch('transformers.AutoTokenizer.from_pretrained', return_value=object()), \
                 patch.object(prepare_pilot, 'validate_jobs') as validate, \
                 patch.object(prepare_pilot, 'server_spec', return_value={'server': 'candidate'}) as spec, \
                 patch.object(prepare_pilot, 'prepare_multi_turn', side_effect=AssertionError('must not regenerate')), \
                 patch.object(prepare_pilot, 'prepare_arrival_trace', side_effect=AssertionError('must not resample')):
                prepare_pilot.main()
            self.assertEqual(json.loads((output / 'jobs.json').read_text()), jobs)
            plan = json.loads((output / 'plan.json').read_text())
            self.assertEqual(plan['framework_source'], 'candidate')
            self.assertEqual(plan['cell_count'], 1)
            self.assertEqual(plan['checkpoint'], checkpoint)
            self.assertEqual([t['id'] for t in plan['trials']], [mode])
            validate.assert_called_once()
            if frozen_worker:
                self.assertEqual(spec.call_args.kwargs['root'], (base / 'frozen').resolve())


if __name__ == '__main__':
    unittest.main()
