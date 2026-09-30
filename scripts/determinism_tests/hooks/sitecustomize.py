"""Install test-only logit and KV observers in engine Python workers.

This directory is prepended to ``PYTHONPATH`` only by the determinism suite.
Python imports ``sitecustomize`` during interpreter startup, which makes the
patch survive process-spawn boundaries without modifying either installed
engine or the Vosti kernel package.
"""

from __future__ import annotations

from importlib.abc import Loader, MetaPathFinder
from importlib.machinery import PathFinder
import os
import sys


_SGLANG_TARGET = "sglang.srt.layers.sampler"
_SGLANG_RUNNER_TARGET = "sglang.srt.model_executor.model_runner"
_VOSTI_TARGET = "vosti_kernels.kernels"


def _patch_sglang(module) -> None:
    from scripts.determinism_tests.logits_observer import observe_logits_rows

    sampler = module.Sampler
    if getattr(sampler, "_vosti_determinism_observer", False):
        return
    original_forward = sampler.forward

    def observed_forward(self, logits_output, *args, **kwargs):
        positions = kwargs.get("positions")
        if positions is None:
            if len(args) < 5:
                raise RuntimeError("cannot observe SGLang sampler positions")
            positions = args[4]
        token_positions = positions.detach().to(device="cpu").reshape(-1).tolist()
        if len(token_positions) != int(logits_output.next_token_logits.shape[0]):
            raise RuntimeError("SGLang sampler positions do not match logit rows")
        metadata = [{"token_position": int(position)} for position in token_positions]
        if os.environ.get("VOSTI_SGLANG_REQUEST_OBSERVER") == "1":
            from scripts.determinism_tests.sglang_observation import SAMPLING_REQUESTS
            requests = SAMPLING_REQUESTS.get()
            if requests is None or len(requests) != len(metadata):
                raise RuntimeError("SGLang sampler has no matching request-ID context")
            metadata = [dict(row, **request) for row, request in zip(metadata, requests, strict=True)]
        observe_logits_rows(
            logits_output.next_token_logits,
            source="sglang.sampler.next_token_logits",
            row_metadata=metadata,
        )
        return original_forward(self, logits_output, *args, **kwargs)

    sampler.forward = observed_forward
    sampler._vosti_determinism_observer = True


def _patch_sglang_runner(module) -> None:
    from scripts.determinism_tests.sglang_observation import SAMPLING_REQUESTS, request_metadata
    runner = module.ModelRunner
    if getattr(runner, '_vosti_request_observer', False):
        return
    original = runner.sample

    def observed_sample(self, logits_output, forward_batch):
        token = SAMPLING_REQUESTS.set(request_metadata(forward_batch))
        try:
            return original(self, logits_output, forward_batch)
        finally:
            SAMPLING_REQUESTS.reset(token)

    runner.sample = observed_sample
    runner._vosti_request_observer = True


def _patch_vosti(module) -> None:
    if os.environ.get("VOSTI_NATIVE_LAYOUT_OBSERVER") == "1":
        from scripts.determinism_tests.native_observation import observe_lengths
        if not getattr(module, "_vosti_layout_observer", False):
            original_lengths = module.seq_lens_tensor

            def observed_lengths(lengths, *args, **kwargs):
                result = original_lengths(lengths, *args, **kwargs)
                observe_lengths(lengths)
                return result

            module.seq_lens_tensor = observed_lengths
            module._vosti_layout_observer = True

    if os.environ.get("VOSTI_KERNELS_LOGITS_OBSERVER") == "1":
        from scripts.determinism_tests.logits_observer import observe_logits_row

        if not getattr(module, "_vosti_logits_observer", False):
            def observed_digest(row, index):
                observe_logits_row(
                    row,
                    index,
                    source="vosti.sampling_boundary",
                )

            # Both select_sample_logits and sample_tokens_rows resolve this
            # global at call time.
            module._maybe_digest_logits = observed_digest
            module._vosti_logits_observer = True

    if os.environ.get("VOSTI_KERNELS_KV_OBSERVER") == "1":
        from functools import wraps
        from scripts.determinism_tests.kv_observer import observe_kv_store

        if not getattr(module, "_vosti_kv_observer", False):
            original_store = module.store_kv_cache_from_verified_caller

            @wraps(original_store)
            def observed_store(k, v, k_cache, v_cache, slot_mapping, runtime=None):
                result = original_store(
                    k, v, k_cache, v_cache, slot_mapping, runtime
                )
                observe_kv_store(
                    k,
                    v,
                    k_cache,
                    v_cache,
                    slot_mapping,
                    source="vosti.store_kv_cache_from_verified_caller",
                )
                return result

            module.store_kv_cache_from_verified_caller = observed_store
            module._vosti_kv_observer = True


class _ObserverLoader(Loader):
    def __init__(self, wrapped: Loader, patch) -> None:
        self.wrapped = wrapped
        self.patch = patch

    def create_module(self, spec):
        create = getattr(self.wrapped, "create_module", None)
        return create(spec) if create is not None else None

    def exec_module(self, module) -> None:
        self.wrapped.exec_module(module)
        self.patch(module)


class _ObserverFinder(MetaPathFinder):
    def __init__(self, target: str, patch) -> None:
        self.target = target
        self.patch = patch

    def find_spec(self, fullname, path, target=None):
        if fullname != self.target:
            return None
        # Ask the ordinary path finder directly so this finder cannot recurse.
        spec = PathFinder.find_spec(fullname, path, target)
        if spec is None or spec.loader is None:
            raise ImportError(
                f"cannot find {self.target} for determinism observation"
            )
        spec.loader = _ObserverLoader(spec.loader, self.patch)
        return spec


if os.environ.get("VOSTI_SGLANG_LOGITS_OBSERVER") == "1":
    sys.meta_path.insert(0, _ObserverFinder(_SGLANG_TARGET, _patch_sglang))
    if os.environ.get("VOSTI_SGLANG_REQUEST_OBSERVER") == "1":
        sys.meta_path.insert(0, _ObserverFinder(_SGLANG_RUNNER_TARGET, _patch_sglang_runner))
if (
    os.environ.get("VOSTI_KERNELS_LOGITS_OBSERVER") == "1"
    or os.environ.get("VOSTI_KERNELS_KV_OBSERVER") == "1"
):
    sys.meta_path.insert(0, _ObserverFinder(_VOSTI_TARGET, _patch_vosti))
