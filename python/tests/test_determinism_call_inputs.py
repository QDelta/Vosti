import unittest
from types import SimpleNamespace as NS
from scripts.determinism_tests.call_inputs import resolve_call
from scripts.determinism_tests.vllm_observation import schedule_rows


class CallInputTests(unittest.TestCase):
    def test_schedule_metadata_preserves_new_and_cached_requests(self):
        output = NS(scheduled_new_reqs=[NS(req_id='a', num_computed_tokens=0)],
                    scheduled_cached_reqs=NS(req_ids=['b'], num_computed_tokens=[128]),
                    num_scheduled_tokens={'b': 64, 'a': 17})
        self.assertEqual(schedule_rows(output), [dict(request_id='b', prefix_tokens=128, query_tokens=64),
                                               dict(request_id='a', prefix_tokens=0, query_tokens=17)])
        output.num_scheduled_tokens['c'] = 1
        with self.assertRaises(RuntimeError):
            schedule_rows(output)

    def test_resolves_only_actual_previous_output(self):
        prior = [dict(requests=[dict(input_token_ids=[1, 2], output_token_ids=[3, 4, 5])])]
        call = dict(label='reuse', output_prefix_from=dict(call=0, request=0, count=2, suffix=[6]))
        self.assertEqual(resolve_call(call, prior)['prompts'], [[1, 2, 3, 4, 6]])
        self.assertNotIn('prompts', call)

    def test_rejects_future_output_and_invalid_counts(self):
        prior = [dict(requests=[dict(input_token_ids=[1], output_token_ids=[2])])]
        for ci, ri, count in [(1, 0, 1), (0, 1, 1), (0, 0, 2), (0, 0, -1), (0, 0, True)]:
            with self.assertRaises(ValueError):
                resolve_call(dict(output_prefix_from=dict(call=ci, request=ri, count=count, suffix=[])), prior)
