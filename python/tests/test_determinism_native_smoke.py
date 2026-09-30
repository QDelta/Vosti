import copy
from pathlib import Path
import unittest
from unittest.mock import patch

from scripts.determinism_tests.native_report import mapping_calls, check_cover_subset


class NativeSmokeTests(unittest.TestCase):
    def test_new_smaller_batch_follows_capture_and_records_decode_rows(self):
        calls = mapping_calls([[1]*65, [2]*257, [3]*1025], [4]*32768)
        self.assertEqual([len(c['prompts']) for c in calls], [1,3,3,1,1,2])
        self.assertEqual(calls[-1]['prompts'], calls[1]['prompts'][:2])
        self.assertTrue(all(c['record_all_rows'] and c['max_tokens']==3 for c in calls))

    def test_cover_counter_delta_and_all_prediction_comparisons(self):
        requests = [dict(input_token_ids=[i], output_rows=[dict(artifact=f'{i}-{j}.npy')
                                                         for j in range(3)]) for i in range(3)]
        result = dict(calls=[{}, dict(requests=requests),
                            dict(graph_stats=dict(cover_replay_count=5)),
                            dict(label='cover-subset', requests=copy.deepcopy(requests[:2]),
                                 graph_stats=dict(cover_replay_count=7))])
        with patch('scripts.determinism_tests.native_report.compare_row_artifacts',
                   return_value=dict(bitwise_equal=True)) as compare:
            self.assertEqual(len(check_cover_subset(Path('/rows'), result)), 6)
            self.assertEqual(compare.call_count, 6)
            for change in ('no-cover', 'one-cover', 'input', 'missing-row'):
                bad = copy.deepcopy(result)
                if change == 'no-cover': bad['calls'][-1]['graph_stats']['cover_replay_count'] = 5
                if change == 'one-cover': bad['calls'][-1]['graph_stats']['cover_replay_count'] = 6
                if change == 'input': bad['calls'][-1]['requests'][0]['input_token_ids'] = [99]
                if change == 'missing-row': bad['calls'][-1]['requests'][0]['output_rows'].pop()
                with self.subTest(change=change), self.assertRaises(RuntimeError):
                    check_cover_subset(Path('/rows'), bad)
