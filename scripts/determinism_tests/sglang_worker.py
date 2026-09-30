#!/usr/bin/env python3
"""Execute one fresh-engine SGLang arm with full raw-logit observation."""

from __future__ import annotations

import argparse
from dataclasses import asdict, is_dataclass
import inspect
import json
import multiprocessing as mp
import os
from pathlib import Path
import socket
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.determinism_tests.protocol import (  # noqa: E402
    SCHEMA_VERSION,
    load_observer_records,
    sha256_json,
)
from scripts.common.process_lifecycle import (  # noqa: E402
    compute_process_pids,
    wait_for_child_exit_without_reaping,
    wait_for_nvml_release,
)
from scripts.determinism_tests.call_inputs import resolve_call


def _cuda_graph_kwargs(server_args_class, max_batch_size: int) -> dict[str, int]:
    """Keep the decode capture limit fixed across the ServerArgs API rename."""
    fields = inspect.signature(server_args_class).parameters
    for name in ("cuda_graph_max_bs_decode", "cuda_graph_max_bs"):
        if name in fields:
            return {name: max_batch_size}
    raise RuntimeError("SGLang exposes no recognized decode CUDA-graph batch limit")


def _cuda_graph_evidence(server_args) -> dict[str, Any]:
    # New SGLang preserves raw constructor inputs on ServerArgs; auto-selected
    # values live in its resolved projection, not those raw attributes.
    resolved = server_args.resolved_dict() if hasattr(server_args, "resolved_dict") else vars(server_args)
    config = resolved.get("cuda_graph_config")
    if isinstance(config, dict):
        return {"cuda_graph_config": config}
    if is_dataclass(config):
        return {"cuda_graph_config": asdict(config)}
    return {"cuda_graph_max_bs": int(resolved["cuda_graph_max_bs"])}


def _backend_evidence(server_args) -> dict[str, Any]:
    resolved = server_args.resolved_dict() if hasattr(server_args, "resolved_dict") else vars(server_args)
    return {**{name: resolved[name] for name in (
        "attention_backend", "sampling_backend", "enable_deterministic_inference",
        "disable_cuda_graph", "disable_radix_cache")}, **_cuda_graph_evidence(server_args)}


def _rows_at_generated_position(
    records: list[dict[str, Any]],
    *,
    prompts: list[list[int]],
    output_token_ids: list[list[int]],
    generated_position: int,
    request_ids: list[str] | None = None,
) -> list[dict[str, Any]]:
    expected_batch = len(prompts)
    if len(output_token_ids) != expected_batch:
        raise RuntimeError("SGLang output-token count does not match prompts")
    if request_ids is not None and (len(request_ids) != expected_batch or len(set(request_ids)) != expected_batch):
        raise RuntimeError("SGLang request IDs do not identify the submitted prompts")
    selected: list[dict[str, Any]] = []
    for request_index, (prompt, tokens) in enumerate(
        zip(prompts, output_token_ids, strict=True)
    ):
        if not 0 <= generated_position < len(tokens):
            raise RuntimeError("SGLang generated position is out of range")
        target_position = len(prompt) + generated_position - 1
        matches = [
            row
            for row in records
            if (row.get('metadata', {}).get('request_id') == request_ids[request_index]
                if request_ids is not None else int(row["batch_index"]) == request_index)
            and int(row.get("metadata", {}).get("token_position", -1))
            == target_position
        ]
        if len(matches) != 1:
            raise RuntimeError(
                "SGLang observer did not identify exactly one requested logit row: "
                f"request={request_index} target_position={target_position} "
                f"matches={len(matches)}"
            )
        record = matches[0]
        if request_ids is None and int(record["batch_size"]) != expected_batch:
            raise RuntimeError("SGLang selected observer call has the wrong batch size")
        expected_token = int(tokens[generated_position])
        if int(record["argmax"]) != expected_token:
            raise RuntimeError(
                "SGLang selected logit row did not produce the returned greedy token: "
                f"request={request_index} argmax={record['argmax']} "
                f"returned={expected_token}"
            )
        selected.append(record)
    return selected


def _row_record(record: dict[str, Any], call_index: int, request_index: int) -> dict[str, Any]:
    return {
        "artifact": record["artifact"],
        "sha256": record["sha256"],
        "shape": record["shape"],
        "dtype": record["observed_dtype"],
        "finite": bool(record["finite"]),
        "argmax": int(record["argmax"]),
        "metadata": {
            "call_index": call_index,
            "request_index": request_index,
            "source": record["source"],
            "observer_pid": record["pid"],
            "observer_call_sequence": record["call_sequence"],
            "token_position": int(record["metadata"]["token_position"]),
            **{key: record['metadata'][key] for key in ('request_id', 'forward_mode', 'query_tokens', 'prefix_tokens')
               if key in record['metadata']},
        },
    }


def _shutdown_with_identifiable_teardown(
    llm: Any, engine_children: list[mp.Process]
) -> dict[str, Any]:
    """Retain SGLang's GPU children until NVML has released their PIDs."""

    child_pids = [
        int(process.pid) for process in engine_children if process.pid is not None
    ]
    gpu_child_pids = sorted(set(child_pids) & compute_process_pids())
    llm.shutdown()
    retained_pids: list[int] = []
    try:
        # Retain every direct engine child before waiting on NVML. In particular,
        # this keeps the scheduler's PID, parent, and start time visible to the
        # independent contention monitor throughout CUDA-context teardown.
        for pid in child_pids:
            wait_for_child_exit_without_reaping(pid)
            retained_pids.append(pid)
        for pid in gpu_child_pids:
            wait_for_nvml_release(pid, owner="SGLang")
    finally:
        for process in engine_children:
            process.join(timeout=1.0)
    return {
        "engine_child_pids": sorted(child_pids),
        "gpu_child_pids": gpu_child_pids,
        "retained_until_exit_pids": sorted(retained_pids),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    args = parser.parse_args()

    arm = json.loads(args.arm.read_text(encoding="utf-8"))
    if arm.get("schema_version") != SCHEMA_VERSION:
        raise RuntimeError("unsupported determinism arm schema")
    if os.environ.get("VOSTI_SGLANG_LOGITS_OBSERVER") != "1":
        raise RuntimeError("SGLang logits observer environment is not enabled")
    trace_requests = bool(arm['engine'].get('trace_requests', False))
    if trace_requests != (os.environ.get('VOSTI_SGLANG_REQUEST_OBSERVER') == '1'):
        raise RuntimeError('SGLang request observer does not match the declared arm')

    import sglang as sgl
    import torch
    from sglang.srt.layers.sampler import Sampler
    from sglang.srt.server_args import ServerArgs

    if not getattr(Sampler, "_vosti_determinism_observer", False):
        raise RuntimeError("SGLang sampler observer was not installed by sitecustomize")

    execution = arm["execution"]
    deterministic = execution["mode"] == "deterministic"
    engine_kwargs: dict[str, Any] = {
        "model_path": arm["model_path"],
        "skip_tokenizer_init": True,
        "mem_fraction_static": float(arm["engine"]["gpu_memory_utilization"]),
        "max_running_requests": int(arm["engine"]["max_num_seqs"]),
        "max_prefill_tokens": int(arm['engine'].get('max_prefill_tokens', 4096)),
        "chunked_prefill_size": int(arm["engine"]["max_num_batched_tokens"]),
        "context_length": int(arm["engine"]["max_model_len"]),
        "random_seed": 0,
        "disable_radix_cache": not bool(arm["engine"]["prefix_caching"]),
    }
    engine_kwargs.update(_cuda_graph_kwargs(ServerArgs, int(arm["engine"]["max_num_seqs"])))
    if 'max_total_tokens' in arm['engine']:
        engine_kwargs['max_total_tokens'] = int(arm['engine']['max_total_tokens'])
    if 'graph_prefill_cap' in arm['engine']:
        engine_kwargs['cuda_graph_max_bs_prefill'] = int(arm['engine']['graph_prefill_cap'])
    backend = execution.get("attention_backend")
    if backend:
        engine_kwargs["attention_backend"] = backend
    if deterministic:
        engine_kwargs["enable_deterministic_inference"] = True
        engine_kwargs["sampling_backend"] = "pytorch"

    args.artifact_dir.mkdir(parents=True, exist_ok=False)
    preexisting_child_pids = {
        int(process.pid)
        for process in mp.active_children()
        if process.pid is not None
    }
    llm = sgl.Engine(**engine_kwargs)
    engine_children = [
        process
        for process in mp.active_children()
        if process.pid is not None and int(process.pid) not in preexisting_child_pids
    ]
    if not engine_children:
        raise RuntimeError("SGLang engine launched no observable direct children")
    call_results = []
    try:
        backend_evidence = _backend_evidence(llm.server_args)
        seen_artifacts = {
            row["artifact"] for row in load_observer_records(args.artifact_dir)
        }
        for call_index, call in enumerate(arm["calls"]):
            call = resolve_call(call, call_results)
            prompts = [[int(token) for token in prompt] for prompt in call["prompts"]]
            max_tokens = int(call["max_tokens"])
            sampling = [
                {
                    "temperature": 0.0,
                    "max_new_tokens": max_tokens,
                    "ignore_eos": True,
                }
                for _ in prompts
            ]
            request_ids = [f'vvdt-{call_index}-{index}' for index in range(len(prompts))] if trace_requests else None
            outputs = llm.generate(input_ids=prompts, sampling_params=sampling,
                                   **({'rid': request_ids} if request_ids is not None else {}))
            if len(outputs) != len(prompts):
                raise RuntimeError("SGLang returned the wrong number of requests")
            if request_ids is not None and [output.get('meta_info', {}).get('id') for output in outputs] != request_ids:
                raise RuntimeError('SGLang returned unexpected request identities or ordering')
            output_token_ids = [
                [int(token) for token in output["output_ids"]] for output in outputs
            ]
            for request_index, token_ids in enumerate(output_token_ids):
                if len(token_ids) != max_tokens:
                    raise RuntimeError(
                        f"call {call_index} request {request_index} returned "
                        f"{len(token_ids)} tokens, expected {max_tokens}"
                    )
            all_records = load_observer_records(args.artifact_dir)
            new_records = [
                record for record in all_records if record["artifact"] not in seen_artifacts
            ]
            seen_artifacts.update(record["artifact"] for record in new_records)
            requested_position = call.get("record_generated_position")
            generated_position = (
                max_tokens - 1
                if requested_position is None
                else int(requested_position)
            )
            selected_rows = (
                _rows_at_generated_position(
                    new_records,
                    prompts=prompts,
                    output_token_ids=output_token_ids,
                    generated_position=generated_position,
                    request_ids=request_ids,
                )
                if call.get("record_last_rows", False)
                else [None] * len(prompts)
            )
            all_position_rows = (
                [_rows_at_generated_position(
                    new_records, prompts=prompts, output_token_ids=output_token_ids,
                    generated_position=position,
                    request_ids=request_ids,
                ) for position in range(max_tokens)]
                if call.get("record_all_rows", False) else None
            )
            requests = []
            for request_index, (output, token_ids) in enumerate(
                zip(outputs, output_token_ids, strict=True)
            ):
                meta = output.get("meta_info", {})
                row_record = (
                    _row_record(selected_rows[request_index], call_index, request_index)
                    if call.get("record_last_rows", False)
                    else None
                )
                if row_record is not None and not row_record["finite"]:
                    raise RuntimeError("SGLang returned a non-finite raw-logit row")
                requests.append(
                    {
                        "prompt_length": len(prompts[request_index]),
                        "input_token_ids": prompts[request_index],
                        "request_id": request_ids[request_index] if request_ids is not None else None,
                        "prompt_sha256": sha256_json(prompts[request_index]),
                        "output_token_ids": token_ids,
                        "num_cached_tokens": int(meta.get("cached_tokens", 0) or 0),
                        "cache_metrics": {
                            key: value
                            for key, value in meta.items()
                            if "cache" in key.lower()
                            and isinstance(value, (int, float))
                            and not isinstance(value, bool)
                        },
                        "last_row": row_record,
                    }
                )
                if all_position_rows is not None:
                    rows = [_row_record(values[request_index], call_index, request_index)
                            for values in all_position_rows]
                    if not all(row["finite"] for row in rows):
                        raise RuntimeError("SGLang returned a non-finite generation row")
                    requests[-1]["output_rows"] = rows
            call_results.append({"label": call["label"], "requests": requests})
    finally:
        teardown_evidence = _shutdown_with_identifiable_teardown(
            llm, engine_children
        )

    result = {
        "schema_version": SCHEMA_VERSION,
        "engine": "sglang",
        "mode": execution["mode"],
        "hostname": socket.gethostname(),
        "gpu": torch.cuda.get_device_name(0),
        "sglang_version": sgl.__version__,
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
        "model_path": str(Path(arm["model_path"]).resolve()),
        "engine_kwargs": engine_kwargs,
        "backend_evidence": backend_evidence,
        "teardown_evidence": teardown_evidence,
        "calls": call_results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
