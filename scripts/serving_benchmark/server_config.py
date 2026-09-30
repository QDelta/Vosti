"""Shared, inspectable server launch configuration for the performance matrix.

These are benchmark launch settings, not model/kernel deployment selection.
No backend is pinned for default modes. Vosti always loads a qualified bundle.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path

from scripts.determinism_tests.protocol import Checkpoint, ExecutionConfig, EXECUTION_CONFIGS, MODELS


# Serving-only extension; do not expand the deterministic experiment matrix.
VLLM_LOCAL_FA3_GLOBAL_TRITON = ExecutionConfig(
    'vllm-invariant-fa3-local-triton-global', 'vllm', 'invariant',
    'FLASH_ATTN', True, 'enabled')
VLLM_INVARIANT_AUTO = ExecutionConfig(
    'vllm-invariant-auto', 'vllm', 'invariant', None, True, 'enabled')
PERFORMANCE_EXECUTION_CONFIGS = EXECUTION_CONFIGS + (
    VLLM_LOCAL_FA3_GLOBAL_TRITON, VLLM_INVARIANT_AUTO,
)


@dataclass(frozen=True)
class ServerSettings:
    gpu_index: int = 0
    port: int = 18300
    context_limit: int = 40960
    max_sequences: int = 4
    max_batched_tokens: int = 4096
    vosti_num_blocks: int = 2560
    baseline_memory_fraction: float = 0.9

    def validate(self):
        for name in ('context_limit', 'max_sequences', 'max_batched_tokens', 'vosti_num_blocks'):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f'{name} must be a positive integer')
        if type(self.gpu_index) is not int or self.gpu_index < 0:
            raise ValueError('GPU index must be nonnegative')
        if type(self.port) is not int or not 1024 <= self.port <= 65535:
            raise ValueError('port must be an unprivileged TCP port')
        if not math.isfinite(self.baseline_memory_fraction) or not 0 < self.baseline_memory_fraction < 1:
            raise ValueError('baseline memory fraction must be between zero and one')


def server_spec(*, root: Path, stack: Path, checkpoint: Checkpoint, mode: ExecutionConfig,
                settings: ServerSettings, cache: Path, build: Path, bundle: Path | None) -> dict:
    settings.validate()
    model = next(item for item in MODELS if item.key == checkpoint.model)
    env = dict(CUDA_VISIBLE_DEVICES=str(settings.gpu_index), CUDA_DEVICE='cuda:0',
        TOKENIZERS_PARALLELISM='false', PYTHONPATH=f'{root}:{root / "python"}',
        TRITON_CACHE_DIR=str(cache / 'triton'), TORCHINDUCTOR_CACHE_DIR=str(cache / 'torchinductor'),
        CUDA_CACHE_PATH=str(cache / 'cuda'))
    served_name = checkpoint.key
    common = ['--model', checkpoint.path, '--served-model-name', served_name,
              '--host', '127.0.0.1', '--port', str(settings.port), '--dtype', 'bfloat16']
    if mode.engine == 'vosti':
        if bundle is None:
            raise ValueError('Vosti requires a qualified deployment bundle')
        command = [str(build / 'release/examples' / f'verus_server_{checkpoint.model}')]
        env.update(MODEL_PATH=checkpoint.path, VOSTI_DEPLOYMENT_BUNDLE=str(bundle),
            VOSTI_SERVED_MODEL_NAME=served_name, VOSTI_SERVER_HOST='127.0.0.1',
            VOSTI_SERVER_PORT=str(settings.port), VOSTI_CUDA_GRAPH='1',
            VOSTI_NUM_BLOCKS=str(settings.vosti_num_blocks), VOSTI_MAX_SEQS=str(settings.max_sequences),
            VOSTI_MAX_BATCHED_TOKENS=str(settings.max_batched_tokens),
            VOSTI_MAX_MODEL_LEN=str(settings.context_limit), VOSTI_ADMISSION_QUEUE='64',
            VOSTI_FRAMEWORK_ROOT=str(root))
    elif mode.engine == 'vllm':
        command = [str(stack / 'vllm/.venv/bin/python'), '-m', 'vllm.entrypoints.openai.api_server',
            *common, '--max-model-len', str(settings.context_limit),
            '--max-num-seqs', str(settings.max_sequences),
            '--max-num-batched-tokens', str(settings.max_batched_tokens),
            '--gpu-memory-utilization', str(settings.baseline_memory_fraction),
            '--enable-prefix-caching', '--enable-prompt-tokens-details', '--seed', '0']
        env['VLLM_BATCH_INVARIANT'] = '1' if mode.mode == 'invariant' else '0'
        if mode.attention_backend:
            attention = dict(backend=mode.attention_backend)
            if mode.attention_backend == 'FLASH_ATTN':
                attention['flash_attn_version'] = 3
            if mode == VLLM_LOCAL_FA3_GLOBAL_TRITON:
                attention['backend_per_kind'] = dict(
                    sliding_window='FLASH_ATTN', full_attention='TRITON_ATTN')
            command += ['--attention-config', json.dumps(attention, sort_keys=True)]
        if model.language_model_only:
            command += ['--language-model-only', '--hf-overrides', '{"is_mm_prefix_lm":false}']
    elif mode.engine == 'sglang':
        command = [str(stack / 'sglang/.venv/bin/python'), '-m', 'sglang.launch_server', *common,
            '--context-length', str(settings.context_limit),
            '--max-running-requests', str(settings.max_sequences),
            '--max-prefill-tokens', str(settings.max_batched_tokens),
            '--chunked-prefill-size', str(settings.max_batched_tokens),
            '--mem-fraction-static', str(settings.baseline_memory_fraction),
            '--cuda-graph-max-bs-decode', str(settings.max_sequences),
            '--enable-cache-report', '--random-seed', '0']
        if mode.attention_backend:
            command += ['--attention-backend', mode.attention_backend]
        if mode.mode == 'deterministic':
            command += ['--enable-deterministic-inference', '--sampling-backend', 'pytorch']
    else:
        raise ValueError(f'unsupported engine {mode.engine}')
    return dict(command=command, environment=env, settings=asdict(settings),
                checkpoint=asdict(checkpoint), execution=asdict(mode),
                base_url=f'http://127.0.0.1:{settings.port}', served_name=served_name,
                logit_observers=False)


def clean_environment(inherited: dict[str, str], spec: dict, *, python_libdir: str,
                      python_site_packages: list[str]) -> dict[str, str]:
    # Never inherit a previous experiment's mode, hooks, bundle or CUDA selector.
    result = {key: value for key, value in inherited.items()
              if not key.startswith(('VOSTI_', 'VLLM_', 'SGLANG_'))
              and key not in {'PYTHONPATH', 'CUDA_VISIBLE_DEVICES', 'CUDA_DEVICE'}}
    result.update(spec['environment'])
    if spec['execution']['engine'] == 'vosti':
        result['PYTHONPATH'] += ':' + ':'.join(python_site_packages)
        result['LD_LIBRARY_PATH'] = python_libdir + ':' + result.get('LD_LIBRARY_PATH', '')
    return result
