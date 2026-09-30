import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from scripts.determinism_tests.schedule_stress import (
    ARRIVALS, OUTPUT_TOKENS, RELATIONS, run_case, schedule_witness,
)


def events(base=8, *, reclaim=True, overlap=True):
    old = [[1, 1, None, [7] * 64]]
    return [dict(request_id_base=0, prefix_pages=old),
            dict(request_id_base=base, prefix_pages=old, arrived=[base], scheduled=[base], emitted=[1]),
            dict(request_id_base=base, prefix_pages=[[1, 1, None, [8] * 64]] if reclaim else old,
                 arrived=[base + 1] if overlap else [], scheduled=[base], emitted=[1])]


class FakeRunner:
    def __init__(self, root, *, reclaim=True, mismatch=False):
        self.root, self.reclaim, self.mismatch = root, reclaim, mismatch
        self.arm_records, self.inputs = [], []

    def run_arm(self, name, arm):
        self.inputs.append((name, arm))
        directory = self.root / name
        (directory / 'rows').mkdir(parents=True)
        requests = []
        for i, _ in enumerate(arm['calls'][0]['prompts']):
            row = np.array([0., 1.], dtype=np.float32)
            if self.mismatch and name == 'staggered_arrivals' and i == 0:
                row[0] = -0.
            np.save(directory / 'rows' / f'{i}.npy', row)
            requests.append(dict(last_row=dict(artifact=f'{i}.npy'), output_token_ids=[1] * OUTPUT_TOKENS))
        trace = directory / 'trace.jsonl'
        base = 12 if name == 'padded_cover' else 8
        trace.write_text('\n'.join(json.dumps(s) for s in events(base, reclaim=self.reclaim)))
        invocation = dict(step_trace=dict(path=str(trace), sha256=hashlib.sha256(trace.read_bytes()).hexdigest()),
                          graph_warmup_stats=dict(replay_count=0, cover_replay_count=0),
                          graph_stats=dict(replay_count=1, cover_replay_count=1))
        self.arm_records.append(dict(name=name))
        return directory, dict(calls=[dict(requests=requests)], backend_evidence=dict(invocations=[invocation]))


class ScheduleStressTests(unittest.TestCase):
    def inputs(self):
        return dict(model={}, model_path='/model', seed=1, prompts=[[i] * 17 for i in range(8)],
                    primer=[[i] * 17 for i in range(4)], arrivals=ARRIVALS)

    def test_witness_requires_real_overlap_reclamation_and_measured_graph_replays(self):
        for enabled in (True, False):
            witness = schedule_witness(events(reclaim=enabled, overlap=enabled), measured_base=8,
                output_tokens=16, graph_before=dict(replay_count=2, cover_replay_count=1),
                graph_after=dict(replay_count=3, cover_replay_count=2))
            self.assertEqual(witness['arrivals_during_decode'], enabled)
            self.assertEqual(witness['registered_prefix_replacements'], int(enabled))
            self.assertEqual(witness['measured_graph_replays'], 2)

    def test_all_four_variants_are_compared_to_fresh_singles(self):
        with tempfile.TemporaryDirectory() as directory:
            runner = FakeRunner(Path(directory))
            result = run_case(runner, self.inputs())
            self.assertEqual(result['status'], 'pass')
            self.assertEqual(set(result['relation_pass']), set(RELATIONS))
            self.assertEqual(len(runner.arm_records), 12)
            pressure = next(a for name, a in runner.inputs if name == 'cache_pressure')
            self.assertEqual(pressure['engine']['num_blocks'], 32)
            self.assertEqual(pressure['calls'][0]['arrival_steps'], ARRIVALS)

    def test_identical_logits_without_reclamation_do_not_pass_pressure(self):
        with tempfile.TemporaryDirectory() as directory:
            result = run_case(FakeRunner(Path(directory), reclaim=False), self.inputs())
            self.assertEqual(result['status'], 'invalid')
            self.assertFalse(result['witnesses']['cache_pressure']['valid'])

    def test_signed_zero_mismatch_is_not_hidden_by_equal_generated_tokens(self):
        with tempfile.TemporaryDirectory() as directory:
            result = run_case(FakeRunner(Path(directory), mismatch=True), self.inputs())
            self.assertEqual(result['status'], 'mismatch')
            self.assertFalse(result['relation_pass']['staggered_arrivals'])
