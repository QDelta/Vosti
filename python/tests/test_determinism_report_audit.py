import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from scripts.determinism_tests.protocol import compare_row_artifacts, sha256_json
from scripts.determinism_tests.rank_divergence import ranking_divergence
from scripts.determinism_tests.report_audit import array, audit_row, backend_evidence, check_comparison, expected_labels


class ReportAuditTests(unittest.TestCase):
    def test_telemetry_binds_to_selected_device_not_a_fixed_gpu(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = dict(checkpoint='test', seed=1, model_path='/checkpoint')
            execution = dict(engine='vosti')
            arm = dict(model_path=plan['model_path'], execution=execution, calls=[])
            arm_hash = sha256_json(arm)
            (root/'arm.json').write_text(json.dumps(arm))
            # Deliberately fail the next validation so no full row fixture is needed.
            (root/'result.json').write_text(json.dumps(dict(arm_sha256=arm_hash, calls=[{}])))
            for gpu in (0, 5):
                (root/'telemetry.json').write_text(json.dumps(dict(status='complete', gpu=dict(index=gpu))))
                telemetry = dict(path=str(root/'telemetry.json'),
                                 sha256=hashlib.sha256((root/'telemetry.json').read_bytes()).hexdigest())
                report = dict(checkpoint='test', seed=1, execution=execution,
                              arms=[dict(result=str(root/'result.json'), arm_sha256=arm_hash, telemetry=telemetry)])
                (root/'report.json').write_text(json.dumps(report))
                with self.assertRaisesRegex(ValueError, 'result call count'):
                    audit_row(root/'report.json', plan, gpu)
                with self.assertRaisesRegex(ValueError, 'selected-GPU telemetry'):
                    audit_row(root/'report.json', plan, gpu+1)

    def test_auto_flash_version_is_bound_to_log_without_mutating_result(self):
        result = dict(engine='vllm', backend_evidence=dict(selected_attention_backend='FLASH_ATTN',
                      attention_config_flash_attn_version=None))
        before = copy.deepcopy(result)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'stdout.log'
            path.write_text('INFO [flash_attn.py:866] Using FlashAttention version 3\n')
            observed = backend_evidence(result, Path(directory))
            self.assertEqual(observed['selected_flash_attn_version'], 3)
            self.assertEqual(observed['version_evidence']['path'], str(path))
            self.assertEqual(len(observed['version_evidence']['sha256']), 64)
            self.assertEqual(result, before)
            result['backend_evidence']['attention_config_flash_attn_version'] = 2
            with self.assertRaises(ValueError): backend_evidence(result, Path(directory))
            result['backend_evidence']['attention_config_flash_attn_version'] = None
            for logs in ('', 'Using FlashAttention version 3',
                         'INFO [flash_attn.py:866] Using FlashAttention version 3\n'
                         'INFO [flash_attn.py:866] Using FlashAttention version 2\n'):
                path.write_text(logs)
                with self.assertRaises(ValueError): backend_evidence(result, Path(directory))
        triton = dict(engine='vllm', backend_evidence=dict(selected_attention_backend='TRITON_ATTN'))
        self.assertEqual(backend_evidence(triton, Path('/missing')), triton['backend_evidence'])

    def test_independent_byte_recheck_includes_signed_zero_and_rank(self):
        with tempfile.TemporaryDirectory() as directory:
            left, right = Path(directory)/'left.npy', Path(directory)/'right.npy'
            a, b = np.array([1.,0.], dtype=np.float32), np.array([1.,-0.], dtype=np.float32)
            np.save(left,a); np.save(right,b)
            row = dict(compare_row_artifacts(left,right), label='zero', ranking=ranking_divergence(a,b))
            self.assertFalse(check_comparison(row))
            for key, value in [('bitwise_equal',True), ('mismatch_count',0), ('argmax_equal',False)]:
                with self.subTest(key=key), self.assertRaises(ValueError):
                    check_comparison(dict(row, **{key:value}))
            wrong = copy.deepcopy(row)
            wrong['ranking']['first_divergent_rank'] = 1
            with self.assertRaises(ValueError): check_comparison(wrong)
        array.cache_clear()

    def test_declared_labels_cover_all_predictions_and_permutations(self):
        plan = dict(batch_groups=[[0,1],[1,0]], fresh_spots=[1], chunk_prompts=[[1,2],[1,2,3]],
                    chunk_budgets=[1,2], pd_prompts=[[1],[1,2]], output_tokens=3,
                    cache_cases=[{'label':'partial'}])
        labels = expected_labels(plan)
        self.assertEqual({key:len(value) for key,value in labels.items()}, dict(batch=6,chunk=4,pd=6,cache=2))
        self.assertIn('1/2', labels['pd'])
        self.assertIn('generated-prefix', labels['cache'])
