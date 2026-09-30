import copy
from pathlib import Path
import tempfile
import unittest

from scripts.determinism_tests.step_trace import index_emissions, select_emission_rows, validate_arrivals
from scripts.determinism_tests.vosti_worker import _write_prepared


class EngineStepTraceTests(unittest.TestCase):
    def test_native_arrival_witness_matches_declared_schedule(self):
        steps = [dict(request_id_base=3, step=0, arrived=[3]),
                 dict(request_id_base=3, step=7, arrived=[4])]
        validate_arrivals(steps, request_id_base=3, arrival_steps=[0, 7])
        for arrivals in ([0, 0], [0], [0, 8]):
            with self.assertRaises(ValueError):
                validate_arrivals(steps, request_id_base=3, arrival_steps=arrivals)

    def test_optional_arrival_column_preserves_old_prepared_format(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'prepared.tsv'
            _write_prepared(path, [[1, 2], [3]], 4)
            self.assertEqual(path.read_text(), '4\t1,2\n4\t3\n')
            _write_prepared(path, [[1, 2], [3]], 4, [0, 7])
            self.assertEqual(path.read_text(), '4\t1,2\t0\n4\t3\t7\n')
            for bad in ([0], [0, -1], [0, True]):
                with self.assertRaises(ValueError):
                    _write_prepared(path, [[1, 2], [3]], 4, bad)

    def fixture(self):
        steps = [dict(engine_step='0:0', request_id_base=0, step=0,
                      scheduled=[0, 1], emitted=[5, None], cached_prefix_blocks=[0, 0]),
                 dict(engine_step='0:1', request_id_base=0, step=1,
                      scheduled=[1, 0], emitted=[6, 7], cached_prefix_blocks=[1, 1])]
        rows = [dict(engine_step=key, batch_index=index, argmax=token)
                for key, index, token in [('0:0', 0, 5), ('0:0', 1, 9), ('0:1', 0, 6), ('0:1', 1, 7)]]
        return rows, steps

    def test_reordered_requests_and_chunk_only_rows_are_mapped_explicitly(self):
        rows, steps = self.fixture()
        indexed = index_emissions(rows, steps)
        self.assertEqual([r['argmax'] for r in indexed[0]], [5, 7])
        self.assertEqual([r['argmax'] for r in indexed[1]], [6])
        self.assertEqual(select_emission_rows(indexed, request_id_base=0, batch_size=1,
                                             max_tokens=2, position=1)[0]['argmax'], 7)
        with self.assertRaises(ValueError):
            select_emission_rows(indexed, request_id_base=0, batch_size=2, max_tokens=2, position=1)
        with self.assertRaises(ValueError):
            index_emissions(rows, list(reversed(steps)))

    def test_missing_duplicate_and_untraced_rows_are_rejected(self):
        rows, steps = self.fixture()
        for bad in (rows[:-1], rows + [rows[0]],
                    [{**rows[0], 'engine_step': '99:0'}, *rows[1:]],
                    [{k: v for k, v in rows[0].items() if k != 'engine_step'}, *rows[1:]]):
            with self.assertRaises(ValueError):
                index_emissions(bad, steps)

    def test_changed_emission_and_ambiguous_schedule_are_rejected(self):
        rows, steps = self.fixture()
        for field, value in [('emitted', [8, None]), ('scheduled', [0, 0]),
                             ('engine_step', 'invalid'), ('cached_prefix_blocks', [])]:
            bad = copy.deepcopy(steps)
            bad[0][field] = value
            with self.assertRaises(ValueError):
                index_emissions(rows, bad)
        with self.assertRaises(ValueError):
            index_emissions(rows, steps + [steps[0]])
