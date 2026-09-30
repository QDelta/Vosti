import copy
import unittest
from scripts.determinism_tests.vllm_observation import public_schedule_events
from scripts.determinism_tests.report_suite import prefill_witness


class RequestIdentityTests(unittest.TestCase):
    def test_maps_exact_versioned_bijection_without_mutating_evidence(self):
        call = dict(requests=[dict(request_id='0')], schedule_events=[
            [dict(request_id='0-0123abcd', prefix_tokens=0, query_tokens=64)],
            [dict(request_id='0-0123abcd', prefix_tokens=64, query_tokens=1)], []])
        original = copy.deepcopy(call)
        events = public_schedule_events(call, '0.28.0')
        self.assertEqual(call, original)
        self.assertTrue(prefill_witness([row for event in events for row in event],65,64)['valid'])
        self.assertEqual(events[0][0]['scheduler_request_id'],'0-0123abcd')

    def test_rejects_unknown_versions_missing_and_ambiguous_mappings(self):
        for ids, version in [(['0-0123abcd'],'0.29.0'), (['1-0123abcd'],'0.28.0'),
                             (['0-0123abcd','0-87654321'],'0.28.0'), ([], '0.28.0')]:
            call = dict(requests=[dict(request_id='0')], schedule_events=[[dict(request_id=rid) for rid in ids]])
            with self.subTest(ids=ids,version=version), self.assertRaises(ValueError):
                public_schedule_events(call,version)

    def test_exact_public_ids_remain_supported(self):
        call = dict(requests=[dict(request_id='0'),dict(request_id='1')],
                    schedule_events=[[dict(request_id='1'),dict(request_id='0')]])
        self.assertEqual([r['request_id'] for r in public_schedule_events(call,'0.28.0')[0]],['1','0'])
