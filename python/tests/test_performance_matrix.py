import unittest
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

from scripts.serving_benchmark.matrix import performance_matrix
from scripts.serving_benchmark import prepare as prepare_multi_trials
from scripts.serving_benchmark.multi_turn import SCHEMA, validate_arrival_trace, write_new
from scripts.determinism_tests.protocol import CHECKPOINTS, EXECUTION_CONFIGS


class PerformanceMatrixTests(unittest.TestCase):
    def test_arrival_trial_preparation_shares_saved_trace_across_all_modes(self):
        checkpoint = SimpleNamespace(key='test', path='/test', performance=True)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            records = []
            for context in (8192, 32768):
                for repetition in range(3):
                    path = root / f'{context}-{repetition}.json'
                    data = dict(schema=SCHEMA, seed=repetition, context=context,
                                sessions=[dict(id=str(i), turns=[{}, {}]) for i in range(8)])
                    write_new(path, data)
                    records.append(dict(checkpoint='test', context=context, repetition=repetition,
                                        measured=str(path), warmup=str(path)))
            write_new(root / 'index.json', dict(records=records))
            write_new(root / 'rates.json', dict(schema='vosti.common-offered-rates.v1',
                                               rates=dict(test=dict(light=2, moderate=4))))
            output = root / 'trials'
            argv = ['prepare', 'multi-trials', '--workloads', str(root), '--stack-root', str(root),
                    '--rates', str(root / 'rates.json'), '--output', str(output)]
            tokenizer_module = SimpleNamespace(AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a, **kw: None))
            with patch.object(prepare_multi_trials, 'CHECKPOINTS', [checkpoint]), \
                 patch.object(prepare_multi_trials, 'validate_jobs'), \
                 patch.object(prepare_multi_trials, 'server_spec', return_value={}), \
                 patch.object(prepare_multi_trials.subprocess, 'check_output', return_value='source\n'), \
                 patch.dict('sys.modules', {'transformers': tokenizer_module}), \
                 patch('sys.argv', argv), patch('builtins.print'):
                prepare_multi_trials.main()
            trials = json.loads((output / 'plan.json').read_text())['trials']
            self.assertEqual(len(trials), 4 * len(EXECUTION_CONFIGS))
            for jobs_path in {trial['jobs'] for trial in trials}:
                self.assertEqual(sum(trial['jobs'] == jobs_path for trial in trials), len(EXECUTION_CONFIGS))
                jobs = json.loads(Path(jobs_path).read_text())
                self.assertEqual(len(jobs), 3)
                for job in jobs:
                    trace = json.loads(Path(job['arrival_trace']).read_text())
                    validate_arrival_trace(trace, json.loads(Path(job['workload']).read_text()))
                    self.assertEqual(trace['arrival_seed'], 42 + job['repetition'])
                    self.assertEqual(trace['request_rate'], job['request_rate'])

    def test_all_models_modes_repetitions_and_shapes_are_present(self):
        cells = performance_matrix()
        checkpoints = {model.key for model in CHECKPOINTS if model.performance}
        modes = {mode.key for mode in EXECUTION_CONFIGS}
        self.assertEqual(len(cells), len(checkpoints) * len(modes) * 3 * 28)
        self.assertEqual({row["execution_config"] for row in cells}, modes)
        self.assertEqual({row["repetition"] for row in cells}, {0, 1, 2})
        self.assertEqual({row["checkpoint"] for row in cells}, checkpoints)

    def test_arrival_rates_are_explicitly_pending_not_invented(self):
        rows = [row for row in performance_matrix() if row["kind"] == "multi_session_arrivals"]
        self.assertTrue(rows)
        self.assertTrue(all(row["request_rate"] is None for row in rows))
        self.assertEqual({row["load"] for row in rows}, {"light", "moderate"})

    def test_prefill_and_cached_extension_have_one_output_token(self):
        for row in performance_matrix():
            if row["kind"] in {"cold_prefill", "cached_extension"}:
                self.assertEqual(row["output_tokens"], 1)


if __name__ == "__main__":
    unittest.main()
