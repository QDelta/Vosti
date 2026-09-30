#!/usr/bin/env python3
"""Execute one fresh-engine vLLM arm and retain its selected raw-logit rows."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.determinism_tests.protocol import SCHEMA_VERSION, sha256_json  # noqa: E402
from scripts.determinism_tests.call_inputs import resolve_call


def _resolved_vocab_size(llm: Any) -> int:
    """Use vLLM's loaded architecture, including padded output embeddings."""

    model_config = llm.llm_engine.vllm_config.model_config
    vocab_size = int(model_config.get_vocab_size())
    if vocab_size <= 0:
        raise RuntimeError(f"vLLM resolved an invalid vocabulary size {vocab_size}")
    return vocab_size


def _requires_language_model_only(model: dict[str, Any]) -> bool:
    """Read the explicit text-projection policy from the experiment arm."""

    value = model.get("language_model_only")
    if not isinstance(value, bool):
        raise RuntimeError("model.language_model_only must be an explicit boolean")
    return value


def _text_projection_engine_overrides(model: dict[str, Any]) -> dict[str, Any]:
    """Disable multimodal prefix-LM handling for an explicit text projection."""

    if not _requires_language_model_only(model):
        return {}
    return {
        "language_model_only": True,
        # vLLM 0.26 otherwise retains the outer model's multimodal prefix-LM
        # classification and can unnecessarily require FlashAttention 4.
        "hf_overrides": {"is_mm_prefix_lm": False},
    }


def _attention_engine_overrides(backend: str | None) -> dict[str, Any]:
    """Keep the suite's FA3 arm explicit as newer engines add FA versions."""
    if not backend:
        return {}
    config: dict[str, Any] = {"backend": backend}
    if backend == "FLASH_ATTN":
        config["flash_attn_version"] = 3
    return {"attention_config": config}


def _raw_row(candidates: dict, vocab_size: int) -> np.ndarray:
    token_ids = {int(token_id) for token_id in candidates}
    expected = set(range(vocab_size))
    if token_ids != expected:
        missing = sorted(expected - token_ids)[:8]
        extra = sorted(token_ids - expected)[:8]
        raise RuntimeError(
            "vLLM did not return one raw logit for every vocabulary entry: "
            f"got={len(token_ids)} expected={vocab_size} missing={missing} extra={extra}"
        )
    row = np.empty(vocab_size, dtype=np.float32)
    for token_id, candidate in candidates.items():
        row[int(token_id)] = np.float32(candidate.logprob)
    return row


def _write_row(
    artifact_dir: Path,
    label: str,
    row: np.ndarray,
    *,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    filename = f"{label}.npy"
    path = artifact_dir / filename
    with path.open("xb") as handle:
        np.save(handle, row, allow_pickle=False)
    row_bytes = row.tobytes(order="C")
    return {
        "artifact": filename,
        "sha256": hashlib.sha256(row_bytes).hexdigest(),
        "shape": list(row.shape),
        "dtype": str(row.dtype),
        "finite": bool(np.isfinite(row).all()),
        "argmax": int(np.argmax(row)),
        "metadata": metadata,
    }


def _resolved_backend(llm: Any) -> dict[str, Any]:
    evidence: dict[str, Any] = {}
    try:
        config = llm.llm_engine.vllm_config.attention_config
        evidence["attention_config_backend"] = str(getattr(config, "backend", None))
        evidence["attention_config_flash_attn_version"] = getattr(
            config, "flash_attn_version", None
        )
    except Exception as error:  # noqa: BLE001
        evidence["attention_config_error"] = repr(error)
    try:
        compilation = llm.llm_engine.vllm_config.compilation_config
        evidence["cudagraph_mode"] = str(getattr(compilation, "cudagraph_mode", None))
        sizes = getattr(compilation, "cudagraph_capture_sizes", None)
        evidence["cudagraph_capture_sizes"] = list(sizes) if sizes else []
    except Exception as error:  # noqa: BLE001
        evidence["compilation_config_error"] = repr(error)
    return evidence


def _shutdown(llm: Any) -> None:
    renderer = getattr(llm.llm_engine, "renderer", None)
    if renderer is not None:
        renderer.shutdown()
    engine_core = getattr(llm.llm_engine, "engine_core", None)
    if engine_core is not None:
        try:
            engine_core.shutdown(timeout=60)
        except TypeError:
            engine_core.shutdown()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    args = parser.parse_args()

    arm = json.loads(args.arm.read_text(encoding="utf-8"))
    if arm.get("schema_version") != SCHEMA_VERSION:
        raise RuntimeError("unsupported determinism arm schema")
    mode = arm["execution"]["mode"]
    expected_env = "1" if mode == "invariant" else "0"
    if os.environ.get("VLLM_BATCH_INVARIANT", "0") != expected_env:
        raise RuntimeError("VLLM_BATCH_INVARIANT does not match arm mode")

    import torch
    import vllm
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    model_path = Path(arm["model_path"]).resolve()
    engine_kwargs: dict[str, Any] = {
        "model": str(model_path),
        "skip_tokenizer_init": True,
        "gpu_memory_utilization": float(arm["engine"]["gpu_memory_utilization"]),
        "enable_prefix_caching": bool(arm["engine"]["prefix_caching"]),
        "enable_chunked_prefill": True,
        "max_num_seqs": int(arm["engine"]["max_num_seqs"]),
        "max_num_batched_tokens": int(arm["engine"]["max_num_batched_tokens"]),
        "max_model_len": int(arm["engine"]["max_model_len"]),
        "max_logprobs": -1,
        "logprobs_mode": "raw_logits",
    }
    backend = arm["execution"].get("attention_backend")
    if 'kv_cache_memory_bytes' in arm['engine']:
        engine_kwargs['kv_cache_memory_bytes'] = int(arm['engine']['kv_cache_memory_bytes'])
    engine_kwargs.update(_attention_engine_overrides(backend))
    engine_kwargs.update(_text_projection_engine_overrides(arm["model"]))

    args.artifact_dir.mkdir(parents=True, exist_ok=False)
    llm = LLM(**engine_kwargs)
    events = None
    if arm['engine'].get('trace_requests', False):
        from scripts.determinism_tests.vllm_observation import observe_scheduler
        events = observe_scheduler(llm)
    vocab_size = _resolved_vocab_size(llm)
    call_results = []
    try:
        backend_evidence = _resolved_backend(llm)
        for call_index, call in enumerate(arm["calls"]):
            call = resolve_call(call, call_results)
            event_start = len(events) if events is not None else 0
            prompts = [
                TokensPrompt(prompt_token_ids=[int(token) for token in prompt])
                for prompt in call["prompts"]
            ]
            max_tokens = int(call["max_tokens"])
            sampling = SamplingParams(
                temperature=0.0,
                max_tokens=max_tokens,
                ignore_eos=True,
                detokenize=False,
                logprobs=-1 if call.get("record_last_rows", False) else None,
            )
            outputs = llm.generate(prompts, sampling, use_tqdm=False)
            if len(outputs) != len(prompts):
                raise RuntimeError("vLLM returned the wrong number of requests")
            requests = []
            for request_index, output in enumerate(outputs):
                completion = output.outputs[0]
                token_ids = [int(token) for token in completion.token_ids]
                if len(token_ids) != max_tokens:
                    raise RuntimeError(
                        f"call {call_index} request {request_index} returned "
                        f"{len(token_ids)} tokens, expected {max_tokens}"
                    )
                row_record = None
                if call.get("record_last_rows", False):
                    if not completion.logprobs or len(completion.logprobs) != max_tokens:
                        raise RuntimeError("vLLM raw-logit rows are missing")
                    requested_position = call.get("record_generated_position")
                    position = max_tokens - 1 if requested_position is None else int(requested_position)
                    if not 0 <= position < max_tokens:
                        raise RuntimeError("vLLM generated position is out of range")
                    row = _raw_row(completion.logprobs[position], vocab_size)
                    label = f"call-{call_index:02d}-request-{request_index:02d}-last"
                    row_record = _write_row(
                        args.artifact_dir,
                        label,
                        row,
                        metadata={
                            "call_index": call_index,
                            "request_index": request_index,
                            "generated_position": position,
                            "source": "vllm-public-raw_logits",
                        },
                    )
                    if not row_record["finite"]:
                        raise RuntimeError("vLLM returned a non-finite raw-logit row")
                requests.append(
                    {
                        "prompt_length": len(call["prompts"][request_index]),
                        "input_token_ids": call['prompts'][request_index],
                        "request_id": output.request_id,
                        "prompt_sha256": sha256_json(call["prompts"][request_index]),
                        "output_token_ids": token_ids,
                        "num_cached_tokens": int(output.num_cached_tokens or 0),
                        "last_row": row_record,
                    }
                )
                if call.get("record_all_rows", False):
                    if not completion.logprobs or len(completion.logprobs) != max_tokens:
                        raise RuntimeError("vLLM raw-logit rows are missing")
                    rows = []
                    for position, candidates in enumerate(completion.logprobs):
                        record = _write_row(args.artifact_dir,
                            f"call-{call_index:02d}-request-{request_index:02d}-position-{position:04d}",
                            _raw_row(candidates, vocab_size), metadata={
                                "call_index": call_index, "request_index": request_index,
                                "generated_position": position, "source": "vllm-public-raw_logits"})
                        if not record["finite"]:
                            raise RuntimeError("vLLM returned a non-finite generation row")
                        if record["argmax"] != token_ids[position]:
                            raise RuntimeError("vLLM generation row disagrees with greedy output")
                        rows.append(record)
                    requests[-1]["output_rows"] = rows
            call_results.append({"label": call["label"], "requests": requests})
            if events is not None:
                call_results[-1]['schedule_events'] = events[event_start:]
    finally:
        _shutdown(llm)

    result = {
        "schema_version": SCHEMA_VERSION,
        "engine": "vllm",
        "mode": mode,
        "hostname": socket.gethostname(),
        "gpu": torch.cuda.get_device_name(0),
        "vllm_version": vllm.__version__,
        "torch_version": torch.__version__,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "runtime_paths": {
            key: os.environ.get(key)
            for key in (
                "TMPDIR",
                "TORCHINDUCTOR_CACHE_DIR",
                "TRITON_CACHE_DIR",
                "CUDA_CACHE_PATH",
            )
        },
        "arm_sha256": sha256_json(arm),
        "model_path": str(model_path),
        "engine_kwargs": engine_kwargs,
        "backend_evidence": backend_evidence,
        "calls": call_results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
