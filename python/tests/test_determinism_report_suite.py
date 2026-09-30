import unittest
from scripts.determinism_tests.report_suite import prefill_witness, summarize


class ReportSuiteTests(unittest.TestCase):
    def test_prefill_requires_complete_contiguous_geometry(self):
        rows = [dict(prefix_tokens=0, query_tokens=64), dict(prefix_tokens=64, query_tokens=1)]
        self.assertTrue(prefill_witness(rows, 65, 64)['valid'])
        self.assertFalse(prefill_witness(rows, 65, 128)['valid'])
        self.assertFalse(prefill_witness(rows[:1], 65, 64)['valid'])
        self.assertFalse(prefill_witness(rows + rows, 65, 64)['valid'])
        self.assertFalse(prefill_witness([dict(prefix_tokens=0, query_tokens=65)], 65, 64)['valid'])

    def test_unrun_or_invalid_cells_never_pass(self):
        row = dict(bitwise_equal=True, finite_left=True, finite_right=True, argmax_equal=True)
        self.assertEqual(summarize({'pd':[row]}, {'pd':2}, {'pd':[]})['pd']['status'], 'incomplete')
        self.assertEqual(summarize({'pd':[row]}, {'pd':1}, {'pd':[{'valid':False}]})['pd']['status'], 'invalid')
