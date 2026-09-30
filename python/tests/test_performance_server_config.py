import json
from dataclasses import replace
from pathlib import Path
import unittest

from scripts.determinism_tests.protocol import CHECKPOINTS, EXECUTION_CONFIGS, MODELS
from scripts.serving_benchmark.server_config import (
    ServerSettings, VLLM_INVARIANT_AUTO, VLLM_LOCAL_FA3_GLOBAL_TRITON,
    clean_environment, server_spec,
)


class PerformanceServerConfigurationTests(unittest.TestCase):
    def spec(self, mode, checkpoint=CHECKPOINTS[0], bundle=Path('/bundle')):
        return server_spec(root=Path('/repo'), stack=Path('/stack'), checkpoint=checkpoint,
            mode=mode, settings=ServerSettings(), cache=Path('/cache'), build=Path('/build'), bundle=bundle)

    def test_all_modes_and_models_share_common_serving_limits(self):
        for model in CHECKPOINTS:
            for mode in EXECUTION_CONFIGS:
                result = self.spec(mode, model)
                self.assertEqual(result['settings']['context_limit'], 40960)
                self.assertEqual(result['settings']['max_sequences'], 4)
                self.assertEqual(result['environment']['CUDA_VISIBLE_DEVICES'], '0')
                self.assertFalse(result['logit_observers'])

    def test_defaults_unpinned_and_deterministic_backends_explicit(self):
        for mode in EXECUTION_CONFIGS:
            result = self.spec(mode)
            command = result['command']
            if mode.engine == 'vllm':
                self.assertIn('--enable-prompt-tokens-details', command)
            if mode.mode == 'fast':
                self.assertNotIn('--attention-backend', command)
                self.assertNotIn('--attention-config', command)
                self.assertNotIn('--sampling-backend', command)
            if mode.engine == 'vllm' and mode.mode == 'invariant':
                attention = json.loads(command[command.index('--attention-config') + 1])
                self.assertEqual(attention['backend'], mode.attention_backend)
                if mode.attention_backend == 'FLASH_ATTN':
                    self.assertEqual(attention['flash_attn_version'], 3)
                self.assertEqual(result['environment']['VLLM_BATCH_INVARIANT'], '1')
            if mode.engine == 'sglang' and mode.mode == 'deterministic':
                self.assertIn('--enable-deterministic-inference', command)
                self.assertEqual(command[command.index('--attention-backend') + 1], mode.attention_backend)
            if mode.engine == 'sglang':
                self.assertIn('--cuda-graph-max-bs-decode', command)
                self.assertIn('--enable-cache-report', command)

    def test_text_projection_is_model_metadata_not_name_heuristic(self):
        for checkpoint in CHECKPOINTS:
            renamed = replace(checkpoint, key='neutral', path='/neutral-checkpoint')
            command = self.spec(EXECUTION_CONFIGS[1], renamed)['command']
            model = next(model for model in MODELS if model.key == checkpoint.model)
            self.assertEqual('--language-model-only' in command, model.language_model_only)

    def test_mixed_invariant_mode_routes_by_attention_kind_not_model_name(self):
        for checkpoint in CHECKPOINTS:
            with self.subTest(checkpoint=checkpoint.key):
                spec = self.spec(VLLM_LOCAL_FA3_GLOBAL_TRITON, checkpoint)
                command = spec['command']
                attention = json.loads(command[command.index('--attention-config') + 1])
                self.assertEqual(attention, dict(backend='FLASH_ATTN', flash_attn_version=3,
                    backend_per_kind=dict(sliding_window='FLASH_ATTN', full_attention='TRITON_ATTN')))
                self.assertEqual(spec['environment']['VLLM_BATCH_INVARIANT'], '1')
                self.assertEqual(spec['execution']['key'], VLLM_LOCAL_FA3_GLOBAL_TRITON.key)

    def test_invariant_auto_enables_invariance_without_pinning_attention(self):
        spec = self.spec(VLLM_INVARIANT_AUTO)
        self.assertEqual(spec['environment']['VLLM_BATCH_INVARIANT'], '1')
        self.assertNotIn('--attention-config', spec['command'])
        self.assertNotIn('--attention-backend', spec['command'])

    def test_vosti_requires_bundle_and_padded_graph(self):
        with self.assertRaisesRegex(ValueError, 'qualified deployment'):
            self.spec(EXECUTION_CONFIGS[0], bundle=None)
        env = self.spec(EXECUTION_CONFIGS[0])['environment']
        self.assertEqual(env['VOSTI_CUDA_GRAPH'], '1')
        self.assertEqual(env['VOSTI_DEPLOYMENT_BUNDLE'], '/bundle')

    def test_inherited_observers_hooks_and_engine_overrides_removed(self):
        base = dict(PYTHONPATH='/repo/scripts/determinism_tests/hooks',
                    VOSTI_SGLANG_LOGITS_OBSERVER='1', VOSTI_LOGITS_OBSERVER_DIR='/rows',
                    VLLM_BATCH_INVARIANT='1', SGLANG_ATTENTION_BACKEND='wrong', PATH='/bin')
        spec = self.spec(EXECUTION_CONFIGS[4])
        env = clean_environment(base, spec, python_libdir='/lib', python_site_packages=['/site'])
        self.assertNotIn('VOSTI_SGLANG_LOGITS_OBSERVER', env)
        self.assertNotIn('VLLM_BATCH_INVARIANT', env)
        self.assertNotIn('SGLANG_ATTENTION_BACKEND', env)
        self.assertNotIn('hooks', env['PYTHONPATH'])
        self.assertEqual(env['PATH'], '/bin')

    def test_invalid_limits_rejected(self):
        for settings in (ServerSettings(context_limit=0), ServerSettings(port=80),
                         ServerSettings(baseline_memory_fraction=float('nan'))):
            with self.assertRaises(ValueError):
                settings.validate()
