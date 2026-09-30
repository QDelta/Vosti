import unittest

from scripts.determinism_tests.campaign import classify_summary, deferred_keys
from scripts.determinism_tests.protocol import TESTS


class CampaignClassificationTests(unittest.TestCase):
    def test_worker_failure_is_not_a_numerical_mismatch(self):
        self.assertEqual(classify_summary(None), "error")

    def test_default_mode_difference_is_still_reported_as_mismatch(self):
        self.assertEqual(classify_summary({"status": "observed-difference",
            "relation_pass": {test: test != "batch_vs_single" for test in TESTS}}), "mismatch")

    def test_all_equal_is_a_pass(self):
        self.assertEqual(classify_summary({"relation_pass": {test: True for test in TESTS}}), "pass")

    def test_missing_relations_cannot_pass(self):
        self.assertEqual(classify_summary({"relation_pass": {}}), "error")


class CampaignDeferralTests(unittest.TestCase):
    keys = ['alpha/fa4/seed-1', 'alpha/fa4/seed-2', 'alpha/triton/seed-1', 'beta/fa4/seed-1']

    def test_only_explicit_model_backend_pair_is_deferred(self):
        self.assertEqual(deferred_keys(['alpha/fa4'], 'retained startup failure', self.keys),
                         {'alpha/fa4/seed-1', 'alpha/fa4/seed-2'})
        self.assertEqual(deferred_keys([], None, self.keys), set())

    def test_unknown_pair_and_missing_reason_are_rejected(self):
        for cases, reason in ((['alpha'], 'reason'), (['alpha/fa4'], ''),
                              (['alpha/fa4'], '  '), ([], 'reason')):
            with self.assertRaises(ValueError):
                deferred_keys(cases, reason, self.keys)
