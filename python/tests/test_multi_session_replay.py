import unittest
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

from python.tests.test_serving_multi_turn import CharacterTokenizer
from scripts.serving_benchmark.multi_session_replay import prepare_inputs
from scripts.serving_benchmark.multi_turn import validate_arrival_trace
from scripts.serving_benchmark.multi_session_replay import union_seconds, compare_reports, METRICS
from scripts.serving_benchmark import prepare
from scripts.serving_benchmark.server_trial import client_command, validate_jobs
from scripts.serving_benchmark.campaign import job_input_files


class MultiSessionReplayTests(unittest.TestCase):
    def test_geometry_seed_and_per_request_arrivals(self):
        tokenizer = CharacterTokenizer()
        with patch('scripts.serving_benchmark.multi_session_replay.tokenizer_artifact_sha256', return_value='test'):
            a = prepare_inputs(tokenizer, SimpleNamespace(path='/unused'))
            b = prepare_inputs(tokenizer, SimpleNamespace(path='/unused'))
        self.assertEqual(a, b)
        measured = a['measured']
        self.assertEqual(len(measured['sessions']), 4)
        common = tokenizer.encode(a['shared_prefix_warmup'])[:8193]
        private = []
        for session in measured['sessions']:
            ids = tokenizer.encode(session['initial_prompt'])
            self.assertEqual(len(ids), 16385)
            self.assertEqual(ids[:8193], common)
            private.append(tuple(ids[8193:]))
            self.assertEqual(len(session['turns']), 6)
            self.assertEqual([t['max_tokens'] for t in session['turns']], [768]*6)
            self.assertEqual([t['suffix_tokens'] for t in session['turns']], [0]+[256]*5)
        self.assertEqual(len(set(private)), 4)
        validate_arrival_trace(a['arrivals'], measured)
        self.assertEqual(a['arrivals']['request_rate'], 2)
        self.assertEqual(len(a['arrivals']['events']), 24)
        self.assertEqual(len({(e['session_id'],e['turn']) for e in a['arrivals']['events']}),24)
        self.assertNotEqual(tokenizer.encode(a['donor_warmup'])[:64],common[:64])
        for s in a['warmup']['sessions']:
            self.assertEqual(s['turns'][0]['max_tokens'],5952)

    def test_phase_unions_do_not_double_count_concurrency(self):
        self.assertEqual(union_seconds([(0,4),(1,2),(3,5),(8,9)]),6)

    def test_another_seed_changes_prompts_and_arrivals_reproducibly(self):
        tokenizer = CharacterTokenizer()
        checkpoint = SimpleNamespace(path='/unused')
        with patch('scripts.serving_benchmark.multi_session_replay.tokenizer_artifact_sha256', return_value='test'):
            original = prepare_inputs(tokenizer, checkpoint)
            first = prepare_inputs(tokenizer, checkpoint, seed=43)
            second = prepare_inputs(tokenizer, checkpoint, seed=43)
        self.assertEqual(first, second)
        self.assertEqual(first['measured']['seed'], 43)
        self.assertEqual(first['warmup']['seed'], 44)
        self.assertEqual(first['arrivals']['arrival_seed'], 43)
        self.assertNotEqual(original['measured']['sessions'], first['measured']['sessions'])
        self.assertNotEqual(original['arrivals']['events'], first['arrivals']['events'])
        validate_arrival_trace(first['arrivals'], first['measured'])
        common = tokenizer.encode(first['shared_prefix_warmup'])[:8193]
        self.assertNotEqual(tokenizer.encode(first['donor_warmup'])[:64], common[:64])
        for session in first['measured']['sessions']:
            ids = tokenizer.encode(session['initial_prompt'])
            self.assertEqual(len(ids), 16385)
            self.assertEqual(ids[:8193], common)
            self.assertEqual(len(session['turns']), 6)
            self.assertEqual([t['max_tokens'] for t in session['turns']], [768]*6)
            self.assertEqual([t['suffix_tokens'] for t in session['turns']], [0]+[256]*5)

    def test_invalid_seed(self):
        with self.assertRaises(ValueError):
            prepare_inputs(CharacterTokenizer(), SimpleNamespace(path='/unused'), seed=-1)

    def test_seed_comparison_and_per_model_normalization(self):
        def row(model,mode,value):
            return dict(model=model,mode=mode,status='complete',**{k:value for k in METRICS})
        reference=dict(source='same',unsupported=[],rows=[row('a','x',10),row('a','y',20),row('b','x',100)])
        current=dict(source='same',unsupported=[],rows=[row('a','x',15),row('a','y',10),row('b','x',200)])
        compared=compare_reports(current,reference)
        self.assertEqual([r['change_pct']['wall_s'] for r in compared],[50,-50,100])
        self.assertEqual([r['pct_of_current_best']['wall_s'] for r in compared],[150,100,100])
        current['rows'][0]['geometry_warning']='Length drift must remain visible'
        self.assertEqual(compare_reports(current,reference)[0]['geometry_warning'], 'Length drift must remain visible')


if __name__ == '__main__': unittest.main()


def test_preparation_emits_reproducible_runnable_plans(monkeypatch, tmp_path):
    tokenizer = CharacterTokenizer()
    monkeypatch.setitem(sys.modules, 'transformers', SimpleNamespace(
        AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a, **kw: tokenizer)))
    monkeypatch.setattr('scripts.serving_benchmark.multi_session_replay.tokenizer_artifact_sha256', lambda _: 'test')
    monkeypatch.setattr('scripts.common.tokenizers.tokenizer_artifact_sha256', lambda _: 'test')
    modes = ['vosti-padded-graph', 'vllm-invariant-flash-attn', 'sglang-deterministic-fa3']
    outputs = [tmp_path / 'first', tmp_path / 'second']
    for output in outputs:
        prepare.main(['multi-session-replay', '--checkpoint', prepare.CHECKPOINTS[0].key, '--stack-root', '/stack',
                      '--gpu-index', '3', '--output', str(output),
                      *[arg for mode in modes for arg in ('--mode', mode)]])
    for name in ('measured', 'arrivals', 'decode-donor', 'decode-warmup', 'prefix-warmup'):
        assert (outputs[0] / f'{name}.json').read_bytes() == (outputs[1] / f'{name}.json').read_bytes()
    output = outputs[0]
    plan = json.loads((output / 'plan.json').read_text())
    jobs = json.loads((output / 'jobs.json').read_text())
    job = jobs[0]
    assert [Path(path).stem for path in job['warmup']] == ['decode-donor', 'decode-warmup', 'prefix-warmup']
    assert len(plan['trials']) == 3 and plan['status'] == 'planned_not_executed'
    assert job['request_rate'] == 2 and job['arrival_seed'] == 42 and job['concurrency'] == 4
    files = job_input_files(job)
    assert set(files) == {job['workload'], job['arrival_trace'], *job['warmup']}
    altered = Path(job['warmup'][1])
    original = altered.read_text()
    altered.write_text(original + '\n')
    assert job_input_files(job)[str(altered)] != files[str(altered)]
    altered.write_text(original)
    assert set(job_input_files(dict(job, warmup=job['warmup'][0]))) == {
        job['workload'], job['arrival_trace'], job['warmup'][0]}
    for trial, mode in zip(plan['trials'], modes):
        spec = json.loads(Path(trial['spec']).read_text())
        assert spec['execution']['key'] == mode
        assert spec['environment']['CUDA_VISIBLE_DEVICES'] == '3'
        assert spec['settings']['context_limit'] == 32768
        assert trial['jobs'] == str(output / 'jobs.json')
        validate_jobs(spec, jobs, tokenizer)
        command = client_command(spec, job, output / 'result.json')
        assert [command[i+1] for i, arg in enumerate(command) if arg == '--warmup-workload'] == job['warmup']
        with unittest.TestCase().assertRaisesRegex(ValueError, 'share an initial prefix'):
            validate_jobs(spec, [job, dict(job, id='second')], tokenizer)
