import unittest

from scripts.determinism_tests.protocol import CHECKPOINTS, EXECUTION_CONFIGS
from scripts.serving_benchmark.calibrate_rates import select_rates


def observations():
    return [dict(checkpoint=model.key, execution_config=mode.key, context=context,
                 repetition=rep, request_throughput_per_s=[10., 2., 3.][rep])
            for model in CHECKPOINTS if model.performance for mode in EXECUTION_CONFIGS
            for context in (8192, 32768) for rep in range(3)]


class CommonRateTests(unittest.TestCase):
    def test_medians_then_slowest_reference_select_common_rates(self):
        rows = observations()
        for row in rows[:3]:
            row['request_throughput_per_s'] *= .5
        result = select_rates(rows)
        first_model = rows[0]['checkpoint']
        self.assertEqual(result[first_model]['light'], .75)
        self.assertAlmostEqual(result[first_model]['moderate'], 1.2)
        for model, rates in result.items():
            self.assertEqual(len(rates['references']), 14)
            if model != first_model:
                self.assertEqual(rates['light'], 1.5)

    def test_missing_duplicate_or_invalid_evidence_cannot_select_rates(self):
        rows = observations()
        for bad in (rows[:-1], rows + [rows[0]],
                    [{**rows[0], 'request_throughput_per_s': float('nan')}, *rows[1:]]):
            with self.assertRaises(ValueError):
                select_rates(bad)

    def test_explicit_exclusion_changes_reference_set_not_per_engine_rate(self):
        rows = observations()
        pair = (rows[0]['checkpoint'], rows[0]['execution_config'])
        kept = [row for row in rows if (row['checkpoint'], row['execution_config']) != pair]
        with self.assertRaises(ValueError):
            select_rates(kept)
        rates = select_rates(kept, excluded_pairs={pair})
        self.assertEqual(len(rates[pair[0]]['references']), 12)
        self.assertEqual(rates[pair[0]]['excluded_modes'], [pair[1]])
        self.assertEqual(rates[pair[0]]['light'], 1.5)
        for model, entry in rates.items():
            if model != pair[0]:
                self.assertEqual(len(entry['references']), 14)
                self.assertEqual(entry['excluded_modes'], [])
        with self.assertRaises(ValueError):
            select_rates(kept[:-1], excluded_pairs={pair})
        with self.assertRaises(ValueError):
            select_rates(rows, excluded_pairs={pair})

    def test_unknown_exclusion_or_model_with_no_reference_is_rejected(self):
        rows = observations()
        with self.assertRaisesRegex(ValueError, 'unknown excluded'):
            select_rates(rows, excluded_pairs={('unknown', 'unknown')})
        model = rows[0]['checkpoint']
        pairs = {(row['checkpoint'], row['execution_config']) for row in rows if row['checkpoint'] == model}
        with self.assertRaisesRegex(ValueError, 'no measured reference'):
            select_rates([row for row in rows if row['checkpoint'] != model], excluded_pairs=pairs)
