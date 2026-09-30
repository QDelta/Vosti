import unittest
from types import SimpleNamespace

from scripts.determinism_tests.expanded_suite import (
    RELATIONS, aggregate, batch_groups, teacher_prefixes,
)
from scripts.determinism_tests.sglang_worker import _rows_at_generated_position
from scripts.determinism_tests.sglang_observation import SAMPLING_REQUESTS, request_metadata
from scripts.determinism_tests.hooks.sitecustomize import _patch_sglang_runner


class ExpandedSuiteTests(unittest.TestCase):
    def test_every_prompt_is_compared_in_each_size_and_order(self):
        groups = batch_groups(8)
        self.assertEqual(len(groups), 14)
        self.assertEqual(sum(map(len, groups)), 48)
        for size in (2, 4, 8):
            subset = [group for group in groups if len(group) == size]
            self.assertEqual(sorted(index for group in subset for index in group), sorted(list(range(8)) * 2))
        self.assertIn(list(reversed(range(8))), groups)

    def test_invalid_prompt_counts_rejected(self):
        for count in (0, 1, 7, 9):
            with self.assertRaises(ValueError):
                batch_groups(count)

    def test_teacher_forcing_excludes_target_token(self):
        self.assertEqual(teacher_prefixes([10, 20], [3, 4, 5]),
                         [[10, 20], [10, 20, 3], [10, 20, 3, 4]])
        with self.assertRaises(ValueError):
            teacher_prefixes([10], [])

    def test_every_generation_row_is_mapped_by_position(self):
        prompts, outputs = [[10, 20], [30, 40, 50]], [[6, 7, 8], [9, 10, 11]]
        records = [dict(batch_index=request, batch_size=2,
                        metadata=dict(token_position=len(prompts[request]) + position - 1),
                        argmax=outputs[request][position])
                   for position in range(3) for request in range(2)]
        for position in range(3):
            rows = _rows_at_generated_position(records, prompts=prompts,
                       output_token_ids=outputs, generated_position=position)
            self.assertEqual([row['argmax'] for row in rows], [output[position] for output in outputs])

    def test_request_ids_handle_reordered_and_split_batches(self):
        records = [dict(batch_index=0, batch_size=1, argmax=6,
                        metadata=dict(request_id='second', token_position=2)),
                   dict(batch_index=0, batch_size=1, argmax=5,
                        metadata=dict(request_id='first', token_position=1))]
        rows = _rows_at_generated_position(records, prompts=[[1, 2], [1, 2, 3]],
            output_token_ids=[[5], [6]], generated_position=0, request_ids=['first', 'second'])
        self.assertEqual([row['argmax'] for row in rows], [5, 6])
        with self.assertRaises(RuntimeError):
            _rows_at_generated_position(records, prompts=[[1, 2], [1, 2, 3]],
                output_token_ids=[[5], [6]], generated_position=0, request_ids=['first', 'missing'])

    def test_request_metadata_has_chunk_geometry(self):
        batch = SimpleNamespace(rids=['a', 'b'], forward_mode='EXTEND',
                                extend_seq_lens_cpu=[64, 17], extend_prefix_lens_cpu=[128, 0])
        self.assertEqual(request_metadata(batch), [
            dict(request_id='a', forward_mode='EXTEND', query_tokens=64, prefix_tokens=128),
            dict(request_id='b', forward_mode='EXTEND', query_tokens=17, prefix_tokens=0)])
        batch.rids = ['a', 'a']
        with self.assertRaises(RuntimeError):
            request_metadata(batch)

    def test_sampling_identity_context_is_reset_even_on_failure(self):
        class Runner:
            def sample(self, logits, batch):
                self.seen = SAMPLING_REQUESTS.get()
                raise ValueError('sample failed')
        module = SimpleNamespace(ModelRunner=Runner)
        _patch_sglang_runner(module)
        instance = Runner()
        with self.assertRaises(ValueError):
            instance.sample(None, SimpleNamespace(rids=['x'], forward_mode='DECODE'))
        self.assertEqual(instance.seen, [dict(request_id='x', forward_mode='DECODE')])
        self.assertIsNone(SAMPLING_REQUESTS.get())

    def rows(self):
        return {name: [dict(bitwise_equal=True, finite_left=True, finite_right=True,
                            argmax_equal=True)] for name in RELATIONS}

    def test_missing_relation_never_passes(self):
        rows = self.rows()
        rows['cache'] = []
        with self.assertRaises(ValueError):
            aggregate(rows, {'cache_hit': True})

    def test_no_cache_witness_invalidates_instead_of_passing(self):
        self.assertEqual(aggregate(self.rows(), {'cache_hit': False})['status'], 'invalid')
        self.assertEqual(aggregate(self.rows(), {})['status'], 'invalid')

    def test_mismatch_and_top_token_difference_are_separate(self):
        rows = self.rows()
        rows['pd'][0]['bitwise_equal'] = False
        result = aggregate(rows, {'cache_hit': True})
        self.assertEqual(result['status'], 'mismatch')
        self.assertEqual(result['cells']['pd']['top_token_differences'], 0)
        rows['pd'][0]['argmax_equal'] = False
        self.assertEqual(aggregate(rows, {'cache_hit': True})['cells']['pd']['top_token_differences'], 1)

    def test_nonfinite_rows_are_invalid(self):
        rows = self.rows()
        rows['batch'][0]['finite_left'] = False
        self.assertEqual(aggregate(rows, {'cache_hit': True})['status'], 'invalid')


if __name__ == '__main__':
    unittest.main()
