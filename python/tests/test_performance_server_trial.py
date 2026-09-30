import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts.serving_benchmark.phases import prepare
from scripts.serving_benchmark.multi_turn import prepare as prepare_multi, prepare_arrival_trace, write_new
from scripts.serving_benchmark.server_trial import (
    client_command, group_gpu_pids, stop_server, validate_jobs,
)


class Tokenizer:
    def encode(self, text, add_special_tokens=True):
        return ([0] if add_special_tokens else []) + list(text.encode())

    def decode(self, ids, skip_special_tokens=False):
        return bytes(i for i in ids if i).decode()


def spec():
    return dict(checkpoint=dict(path='/checkpoint'), served_name='test', base_url='http://test',
                execution=dict(key='test-mode'), settings=dict(context_limit=100, max_sequences=4))


class TrialPreparationTests(unittest.TestCase):
    def test_saved_arrivals_are_forwarded_and_validated_before_launch(self):
        tokenizer = Tokenizer()
        kwargs = dict(sessions=2, turns=2, initial_tokens=40, suffix_tokens=4,
                      output_tokens=2, think_seconds=0)
        measured = prepare_multi(tokenizer, seed=1, **kwargs)
        warmup = prepare_multi(tokenizer, seed=2, **kwargs)
        for data in (measured, warmup):
            data['tokenizer_artifact_sha256'] = 'artifact'
        trace = prepare_arrival_trace(measured, request_rate=2, seed=7)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, data in (('workload', measured), ('warmup', warmup), ('arrival_trace', trace)):
                write_new(root / f'{name}.json', data)
            job = dict(id='first', kind='multi_session_arrivals', concurrency=2,
                       request_rate=2, arrival_process='poisson', arrival_seed=7,
                       **{name: str(root / f'{name}.json') for name in ('workload', 'warmup', 'arrival_trace')})
            command = client_command(spec(), job, root / 'result.json')
            self.assertEqual(command[command.index('--arrival-trace') + 1], job['arrival_trace'])
            with patch('scripts.common.tokenizers.tokenizer_artifact_sha256', return_value='artifact'):
                validate_jobs(spec(), [job], tokenizer)
                with self.assertRaisesRegex(ValueError, 'conflicts'):
                    validate_jobs(spec(), [{**job, 'arrival_seed': 8}], tokenizer)

    def test_command_is_common_client_not_logit_observer(self):
        for job, module in ((dict(kind='phase', workload='/phase.json'), 'phases'),
                            (dict(kind='multi_session', workload='/multi.json', warmup='/warmup.json', concurrency=4), 'multi_turn')):
            command = client_command(spec(), job, Path('/result.json'))
            self.assertIn(f'scripts.serving_benchmark.{module}', command)
            self.assertNotIn('scripts.determinism_tests', ' '.join(command))
            self.assertEqual(command[command.index('--context-limit') + 1], '100')

    def test_validate_jobs_checks_cross_job_prefix_reuse(self):
        tokenizer = Tokenizer()
        data = prepare(tokenizer, kind='cold_prefill', context=0, query=40, output=1,
                       concurrency=1, waves=1, seed=1)
        data['tokenizer_artifact_sha256'] = 'artifact'
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'workload.json'
            path.write_text(json.dumps(data))
            job = dict(id='first', kind='phase', workload=str(path))
            with patch('scripts.common.tokenizers.tokenizer_artifact_sha256', return_value='artifact'):
                validate_jobs(spec(), [job], tokenizer)
                with self.assertRaisesRegex(ValueError, 'share an initial prefix'):
                    validate_jobs(spec(), [job, {**job, 'id': 'second'}], tokenizer)
                with self.assertRaisesRegex(ValueError, 'unique IDs'):
                    validate_jobs(spec(), [job, job], tokenizer)
                with self.assertRaisesRegex(ValueError, 'filename'):
                    validate_jobs(spec(), [{**job, 'id': '../escape'}], tokenizer)
                too_small = copy.deepcopy(spec())
                too_small['settings']['context_limit'] = 40
                with self.assertRaisesRegex(ValueError, 'context limit'):
                    validate_jobs(too_small, [job], tokenizer)


class TrialOwnershipTests(unittest.TestCase):
    def test_failed_diagnostic_still_terminates_owned_server(self):
        server = SimpleNamespace(pid=123, wait=lambda **kwargs: None)
        with patch('scripts.serving_benchmark.server_trial.group_gpu_pids', side_effect=RuntimeError('NVML unavailable')), \
             patch('scripts.serving_benchmark.server_trial.gpu_pids', return_value=set()), \
             patch('scripts.serving_benchmark.server_trial.os.killpg') as kill, \
             patch('scripts.serving_benchmark.server_trial.wait_for_child_exit_without_reaping'), \
             patch('scripts.serving_benchmark.server_trial.time.sleep'):
            with self.assertRaisesRegex(RuntimeError, 'could not verify GPU ownership'):
                stop_server(server, 3)
        self.assertTrue(kill.called)
        self.assertEqual({call.args[0] for call in kill.call_args_list}, {123})

    def test_gpu_pids_are_filtered_by_owned_process_group(self):
        with patch('scripts.serving_benchmark.server_trial.gpu_pids', return_value={10, 20}), \
             patch('scripts.serving_benchmark.server_trial.os.getpgid', side_effect=lambda pid: 123 if pid == 10 else 456):
            self.assertEqual(group_gpu_pids(123, 3), {10})

    def test_foreign_gpu_job_is_never_signalled_or_waited_on(self):
        server = SimpleNamespace(pid=123, wait=lambda **kwargs: None)
        with patch('scripts.serving_benchmark.server_trial.group_gpu_pids', return_value={10}), \
             patch('scripts.serving_benchmark.server_trial.gpu_pids', return_value={20}), \
             patch('scripts.serving_benchmark.server_trial.os.killpg') as kill, \
             patch('scripts.serving_benchmark.server_trial.wait_for_child_exit_without_reaping'), \
             patch('scripts.serving_benchmark.server_trial.time.sleep'):
            stop_server(server, 3)
        self.assertEqual([call.args[0] for call in kill.call_args_list], [123])
