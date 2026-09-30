#!/usr/bin/env python3
"""Execute one Vosti arm through qualified, padded-CUDA-graph binaries."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import sysconfig
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from kernels.triton_kernels.constants import PAGE_SIZE  # noqa: E402

from scripts.determinism_tests.protocol import (  # noqa: E402
    SCHEMA_VERSION,
    _model_token_domain,
    load_observer_records,
    sha256_json,
)
from scripts.common.process_lifecycle import (  # noqa: E402
    NVML_INFLIGHT_SETTLE_S,
    NVML_RELEASE_POLL_S,
    NVML_RELEASE_TIMEOUT_S,
    compute_process_pids as _compute_process_pids,
    wait_for_nvml_release as _shared_wait_for_nvml_release,
)


def _write_prepared(path: Path, prompts: list[list[int]], max_tokens: int,
                    arrival_steps: list[int] | None = None) -> None:
    if arrival_steps is not None and (len(arrival_steps) != len(prompts)
            or any(type(step) is not int or step < 0 for step in arrival_steps)):
        raise ValueError("arrival steps must be one nonnegative integer per prompt")
    lines = [
        f"{max_tokens}\t{','.join(str(int(token)) for token in prompt)}"
        + (f"\t{arrival_steps[index]}" if arrival_steps is not None else "")
        for index, prompt in enumerate(prompts)
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _disjoint_warmup(prompts: list[list[int]], vocab_size: int) -> list[list[int]]:
    """Construct shape-identical prompts with disjoint first cache pages."""

    measured_pages = {
        tuple(prompt[:PAGE_SIZE]) for prompt in prompts if len(prompt) >= PAGE_SIZE
    }
    for delta in range(1, vocab_size):
        warmup = [
            [int((token + delta) % vocab_size) for token in prompt]
            for prompt in prompts
        ]
        warmup_pages = {
            tuple(prompt[:PAGE_SIZE])
            for prompt in warmup
            if len(prompt) >= PAGE_SIZE
        }
        if measured_pages.isdisjoint(warmup_pages):
            return warmup
    raise RuntimeError("could not construct prefix-disjoint Vosti warmup prompts")


def _parse_output_tokens(path: Path) -> list[list[int]]:
    rows: dict[int, list[int]] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        request_id, raw_tokens = raw_line.split("\t", maxsplit=1)
        rows[int(request_id)] = (
            [int(token) for token in raw_tokens.split(",")]
            if raw_tokens
            else []
        )
    if set(rows) != set(range(len(rows))):
        raise RuntimeError("Vosti output-token record has non-contiguous request IDs")
    return [rows[index] for index in range(len(rows))]


def _wait_for_nvml_release(
    pid: int,
    *,
    timeout_s: float = NVML_RELEASE_TIMEOUT_S,
    poll_s: float = NVML_RELEASE_POLL_S,
    settle_s: float = NVML_INFLIGHT_SETTLE_S,
) -> None:
    _shared_wait_for_nvml_release(
        pid,
        compute_pids=_compute_process_pids,
        timeout_s=timeout_s,
        poll_s=poll_s,
        settle_s=settle_s,
        owner="Vosti",
    )


def _run_binary_with_identifiable_teardown(
    binary: Path, *, env: dict[str, str], stdout, stderr
) -> int:
    process = subprocess.Popen([str(binary)], env=env, stdout=stdout, stderr=stderr)
    # WNOWAIT leaves the exited child as a zombie. Its /proc start time and
    # parent remain observable to the independent telemetry monitor while the
    # CUDA driver removes the corresponding NVML compute-process row.
    os.waitid(os.P_PID, process.pid, os.WEXITED | os.WNOWAIT)
    try:
        _wait_for_nvml_release(process.pid)
    finally:
        returncode = process.wait()
    return returncode


def _json_log_record(stdout_text: str, prefix: str) -> dict[str, Any] | None:
    matches = [
        json.loads(line.removeprefix(prefix))
        for line in stdout_text.splitlines()
        if line.startswith(prefix)
    ]
    if len(matches) > 1:
        raise RuntimeError(f"Vosti emitted multiple {prefix.strip()} records")
    return matches[0] if matches else None


def _copy_row(
    record: dict[str, Any],
    observer_dir: Path,
    artifact_dir: Path,
    *,
    label: str,
    call_index: int,
    request_index: int,
) -> dict[str, Any]:
    filename = f"{label}.npy"
    destination = artifact_dir / filename
    with destination.open("xb") as output, (
        observer_dir / record["artifact"]
    ).open("rb") as source:
        shutil.copyfileobj(source, output)
    return {
        "artifact": filename,
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
        },
    }


def _records_at_position(
    records: list[dict[str, Any]],
    *,
    batch_size: int,
    max_tokens: int,
    position: int,
) -> list[dict[str, Any]]:
    if not 0 <= position < max_tokens:
        raise RuntimeError("Vosti retained generation position is out of bounds")
    minimum = batch_size * max_tokens
    if len(records) < minimum:
        raise RuntimeError(
            "Vosti observer has fewer rows than emitted tokens: "
            f"got={len(records)} minimum={minimum}"
        )
    emitted = records[-minimum:]
    selected = emitted[position * batch_size : (position + 1) * batch_size]
    by_index = {int(record["batch_index"]): record for record in selected}
    if set(by_index) != set(range(batch_size)):
        raise RuntimeError("Vosti retained sampling call has incomplete batch indices")
    return [by_index[index] for index in range(batch_size)]


def _final_records(
    records: list[dict[str, Any]], *, batch_size: int, max_tokens: int
) -> list[dict[str, Any]]:
    return _records_at_position(
        records,
        batch_size=batch_size,
        max_tokens=max_tokens,
        position=max_tokens - 1,
    )


def _request_results(
    *,
    prompts: list[list[int]],
    output_tokens: list[list[int]],
    final_records: list[dict[str, Any]],
    observer_dir: Path,
    artifact_dir: Path,
    call_index: int,
    record_rows: bool,
    record_position: int,
    cached_tokens: int,
) -> list[dict[str, Any]]:
    requests = []
    for request_index, (prompt, tokens) in enumerate(zip(prompts, output_tokens)):
        row = (
            _copy_row(
                final_records[request_index],
                observer_dir,
                artifact_dir,
                label=(
                    f"call-{call_index:02d}-request-{request_index:02d}-"
                    f"position-{record_position:02d}"
                ),
                call_index=call_index,
                request_index=request_index,
            )
            if record_rows
            else None
        )
        if row is not None:
            row["metadata"]["generated_position"] = record_position
        if row is not None and not row["finite"]:
            raise RuntimeError("Vosti returned a non-finite raw-logit row")
        if row is not None and int(row["argmax"]) != int(tokens[record_position]):
            raise RuntimeError(
                "Vosti retained row does not select its emitted token"
            )
        requests.append(
            {
                "prompt_length": len(prompt),
                "prompt_sha256": sha256_json(prompt),
                "output_token_ids": tokens,
                "num_cached_tokens": cached_tokens if len(prompts) == 1 else 0,
                "last_row": row,
            }
        )
    return requests


def _run_invocation(
    *,
    arm: dict[str, Any],
    binary: Path,
    deployment_bundle: Path,
    invocation_dir: Path,
    artifact_dir: Path,
    call_index: int,
    measured_call: dict[str, Any],
    vocab_size: int,
    warmup_call: dict[str, Any] | None = None,
) -> tuple[dict[str, Any] | None, dict[str, Any], dict[str, Any]]:
    invocation_dir.mkdir(parents=True, exist_ok=False)
    measured_prompts = [
        [int(token) for token in prompt] for prompt in measured_call["prompts"]
    ]
    measured_max_tokens = int(measured_call["max_tokens"])
    if warmup_call is None:
        warmup_prompts = _disjoint_warmup(measured_prompts, vocab_size)
        warmup_max_tokens = measured_max_tokens
    else:
        warmup_prompts = [
            [int(token) for token in prompt] for prompt in warmup_call["prompts"]
        ]
        warmup_max_tokens = int(warmup_call["max_tokens"])
        # The native prepared-pair driver reads each workload's own prompts
        # and output budgets. A generated-prefix donor is intentionally shorter
        # than its follow-up and may generate many more tokens. Preserve those
        # independent geometries in one engine lifetime.

    prepared_path = invocation_dir / "measured.tsv"
    warmup_path = invocation_dir / "warmup.tsv"
    output_path = invocation_dir / "output-tokens.tsv"
    observer_dir = invocation_dir / "observer"
    stdout_path = invocation_dir / "stdout.log"
    stderr_path = invocation_dir / "stderr.log"
    measured_arrivals = measured_call.get("arrival_steps")
    warmup_arrivals = warmup_call.get("arrival_steps") if warmup_call else measured_arrivals
    if (measured_arrivals is not None or warmup_arrivals is not None) and not arm["engine"].get("trace_steps", False):
        raise ValueError("arrival schedules require native step-trace mapping")
    _write_prepared(prepared_path, measured_prompts, measured_max_tokens, measured_arrivals)
    _write_prepared(warmup_path, warmup_prompts, warmup_max_tokens, warmup_arrivals)

    env = os.environ.copy()
    python_lib = sysconfig.get_config_var("LIBDIR")
    old_ld = env.get("LD_LIBRARY_PATH")
    env["LD_LIBRARY_PATH"] = (
        str(python_lib) if not old_ld else f"{python_lib}:{old_ld}"
    )
    purelib = str(Path(sysconfig.get_paths()["purelib"]).resolve())
    python_paths = [
        path for path in env.get("PYTHONPATH", "").split(os.pathsep) if path
    ]
    if purelib not in python_paths:
        python_paths.append(purelib)
    env["PYTHONPATH"] = os.pathsep.join(python_paths)
    env.update(
        {
            "MODEL_PATH": str(Path(arm["model_path"]).resolve()),
            "CUDA_DEVICE": "cuda:0",
            "VOSTI_FRAMEWORK_ROOT": str(ROOT),
            "VOSTI_BENCH": "1",
            "VOSTI_BENCH_INPUT": str(prepared_path),
            "VOSTI_BENCH_WARMUP_INPUT": str(warmup_path),
            "VOSTI_OUTPUT_TOKENS": str(output_path),
            "VOSTI_MAX_BATCHED_TOKENS": str(
                int(arm["engine"]["max_num_batched_tokens"])
            ),
            "VOSTI_MAX_SEQS": str(int(arm["engine"]["max_num_seqs"])),
            "VOSTI_NUM_BLOCKS": str(int(arm["engine"]["num_blocks"])),
            "VOSTI_CUDA_GRAPH": "1",
            "VOSTI_GRAPH_WARMUP_ROUNDS": "0",
            "VOSTI_CACHE_STATS": "1",
            "VOSTI_QUIET_OUTPUT": "1",
            "VOSTI_KERNELS_LOGITS_OBSERVER": "1",
            "VOSTI_LOGITS_OBSERVER_DIR": str(observer_dir),
            "VOSTI_LOGITS_OBSERVER_PHASE": "warmup",
        }
    )
    env["VOSTI_DEPLOYMENT_BUNDLE"] = str(deployment_bundle)
    env.pop("VOSTI_LOGITS_OBSERVER_STEP", None)
    env.pop("VOSTI_ENGINE_STEP_TRACE", None)
    trace_path = invocation_dir / "step-trace.jsonl" if arm["engine"].get("trace_steps", False) else None
    if trace_path is not None:
        env["VOSTI_ENGINE_STEP_TRACE"] = str(trace_path)
    # Do not inherit an unrecorded primer from the caller's environment.
    env.pop("VOSTI_BENCH_GRAPH_PRIMER_INPUT", None)
    primer = arm["engine"].get("graph_primer")
    if primer is not None:
        primer_path = invocation_dir / "graph-primer.tsv"
        _write_prepared(primer_path, primer["prompts"], int(primer["max_tokens"]))
        env["VOSTI_BENCH_GRAPH_PRIMER_INPUT"] = str(primer_path)

    with stdout_path.open("xb") as stdout, stderr_path.open("xb") as stderr:
        returncode = _run_binary_with_identifiable_teardown(
            binary, env=env, stdout=stdout, stderr=stderr
        )
    if returncode != 0:
        raise RuntimeError(
            f"Vosti invocation failed with code {returncode}; see {stderr_path}"
        )
    stdout_text = stdout_path.read_text(encoding="utf-8", errors="replace")
    if "backend_qualified=true" not in stdout_text:
        raise RuntimeError("Vosti did not report a qualified model runtime")
    measured_outputs = _parse_output_tokens(output_path)
    if len(measured_outputs) != len(measured_prompts) or any(
        len(tokens) != measured_max_tokens for tokens in measured_outputs
    ):
        raise RuntimeError("Vosti measured output-token record has the wrong shape")

    records = load_observer_records(observer_dir)
    warmup_records = [record for record in records if record.get("phase") == "warmup"]
    measured_records = [
        record for record in records if record.get("phase") == "measured"
    ]
    primer_records = [record for record in records if record.get("phase") == "primer"]
    if (primer is None) != (len(primer_records) == 0):
        raise RuntimeError("Vosti graph primer observation differs from the declared arm")
    if len(warmup_records) + len(measured_records) + len(primer_records) != len(records):
        raise RuntimeError(
            "Vosti observer has records without an explicit execution phase"
        )
    measured_position_raw = measured_call.get("record_generated_position")
    measured_position = (
        measured_max_tokens - 1
        if measured_position_raw is None
        else int(measured_position_raw)
    )
    indexed = None
    warmup_base = len(primer["prompts"]) if primer else 0
    if trace_path is not None:
        from scripts.determinism_tests.step_trace import index_emissions, select_emission_rows, validate_arrivals
        steps = [json.loads(line) for line in trace_path.read_text().splitlines()]
        indexed = index_emissions(records, steps)
        validate_arrivals(steps, request_id_base=warmup_base,
                          arrival_steps=warmup_arrivals if warmup_arrivals is not None else [0] * len(warmup_prompts))
        validate_arrivals(steps, request_id_base=warmup_base + len(warmup_prompts),
                          arrival_steps=measured_arrivals if measured_arrivals is not None else [0] * len(measured_prompts))
        if primer:
            validate_arrivals(steps, request_id_base=0, arrival_steps=[0] * len(primer["prompts"]))
        measured_selected = select_emission_rows(indexed,
            request_id_base=warmup_base + len(warmup_prompts), batch_size=len(measured_prompts),
            max_tokens=measured_max_tokens, position=measured_position)
    else:
        measured_selected = _records_at_position(
            measured_records, batch_size=len(measured_prompts),
            max_tokens=measured_max_tokens, position=measured_position)
    cache_stats = _json_log_record(stdout_text, "CACHE_STATS ") or {
        "admitted_requests": 0,
        "requests_with_reuse": 0,
        "reused_prefix_blocks": 0,
    }
    measured_cached_tokens = int(cache_stats["reused_prefix_blocks"]) * PAGE_SIZE
    measured_result = {
        "label": measured_call["label"],
        "requests": _request_results(
            prompts=measured_prompts,
            output_tokens=measured_outputs,
            final_records=measured_selected,
            observer_dir=observer_dir,
            artifact_dir=artifact_dir,
            call_index=call_index,
            record_rows=bool(measured_call.get("record_last_rows", False)),
            record_position=measured_position,
            cached_tokens=measured_cached_tokens,
        ),
    }

    warmup_result = None
    if warmup_call is not None:
        warmup_position_raw = warmup_call.get("record_generated_position")
        warmup_position = (
            warmup_max_tokens - 1
            if warmup_position_raw is None
            else int(warmup_position_raw)
        )
        if indexed is not None:
            warmup_selected = select_emission_rows(indexed,
                request_id_base=warmup_base, batch_size=len(warmup_prompts),
                max_tokens=warmup_max_tokens, position=warmup_position)
            warmup_outputs = [[int(row["argmax"]) for row in indexed[warmup_base + index]]
                              for index in range(len(warmup_prompts))]
        else:
            warmup_selected = _records_at_position(
                warmup_records, batch_size=len(warmup_prompts),
                max_tokens=warmup_max_tokens, position=warmup_position)
            warmup_emissions = warmup_records[-(len(warmup_prompts) * warmup_max_tokens):]
            warmup_outputs = [
                [int(warmup_emissions[step * len(warmup_prompts) + index]["argmax"])
                 for step in range(warmup_max_tokens)]
                for index in range(len(warmup_prompts))
            ]
        warmup_result = {
            "label": warmup_call["label"],
            "requests": _request_results(
                prompts=warmup_prompts,
                output_tokens=warmup_outputs,
                final_records=warmup_selected,
                observer_dir=observer_dir,
                artifact_dir=artifact_dir,
                call_index=call_index - 1,
                record_rows=bool(warmup_call.get("record_last_rows", False)),
                record_position=warmup_position,
                cached_tokens=0,
            ),
        }

    graph_stats = _json_log_record(stdout_text, "GRAPH_STATS ")
    graph_warmup_stats = _json_log_record(stdout_text, "GRAPH_WARMUP_STATS ")
    if graph_stats is None or graph_stats.get("poisoned_reason") is not None:
        raise RuntimeError("Vosti padded CUDA graph was absent or poisoned")
    if "cover_replay_count" not in graph_stats:
        raise RuntimeError("Vosti graph stats do not expose padded-cover replay")
    if measured_call.get("require_graph_replay", False):
        if graph_warmup_stats is None:
            raise RuntimeError("Vosti decode workload lacks graph warmup evidence")
        measured_replays = (
            int(graph_stats.get("replay_count", 0))
            + int(graph_stats.get("cover_replay_count", 0))
            - int(graph_warmup_stats.get("replay_count", 0))
            - int(graph_warmup_stats.get("cover_replay_count", 0))
        )
        if measured_replays <= 0:
            raise RuntimeError("Vosti measured decode did not replay a CUDA graph")
    invocation_evidence = {
        "binary": str(binary),
        "deployment_bundle": str(deployment_bundle),
        "step_trace": None if trace_path is None else dict(
            path=str(trace_path), sha256=hashlib.sha256(trace_path.read_bytes()).hexdigest()),
        "graph_stats": graph_stats,
        "graph_warmup_stats": graph_warmup_stats,
        "cache_stats": cache_stats,
        "graph_primer_observed_rows": len(primer_records),
        "stdout": str(stdout_path),
        "stderr": str(stderr_path),
    }
    return warmup_result, measured_result, invocation_evidence


def _invocation_plan(
    calls: list[dict[str, Any]],
) -> list[tuple[int, dict[str, Any], dict[str, Any] | None]]:
    """Map one logical arm to the engine lifetimes supported by the driver."""

    if len(calls) == 1:
        return [(0, calls[0], None)]
    if len(calls) == 2:
        # Both calls belong to one arm and therefore one engine lifetime, just
        # as they do in the vLLM and SGLang workers. The native driver exposes
        # this as its warmup-then-measured prepared-workload pair.
        return [(1, calls[1], calls[0])]
    raise RuntimeError("Vosti arms support one or two calls")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    args = parser.parse_args()

    arm = json.loads(args.arm.read_text(encoding="utf-8"))
    if arm.get("schema_version") != SCHEMA_VERSION:
        raise RuntimeError("unsupported determinism arm schema")
    if arm["execution"]["key"] != "vosti-padded-graph":
        raise RuntimeError("Vosti worker requires the qualified padded-graph config")
    if os.environ.get("VOSTI_KERNELS_LOGITS_OBSERVER") != "1":
        raise RuntimeError("Vosti logits observer environment is not enabled")

    binary = Path(arm["engine"]["vosti_binary"]).resolve()
    deployment_bundle = Path(arm["engine"]["deployment_bundle"]).resolve()
    if not binary.is_file():
        raise FileNotFoundError(binary)
    if not (deployment_bundle / "deployment.json").is_file():
        raise FileNotFoundError(deployment_bundle / "deployment.json")
    model_path = Path(arm["model_path"]).resolve()
    vocab_size, _bos_token_id, _excluded = _model_token_domain(model_path)

    args.artifact_dir.mkdir(parents=True, exist_ok=False)
    invocations = args.artifact_dir.parent / "invocations"
    invocations.mkdir(exist_ok=False)
    call_results: list[dict[str, Any]] = []
    invocation_evidence = []
    calls = arm["calls"]
    if arm['engine'].get('multi_call', False):
        from scripts.determinism_tests.native_calls import run_calls
        call_results, evidence = run_calls(arm=arm, binary=binary, deployment_bundle=deployment_bundle,
            invocation_dir=invocations/'calls', artifact_dir=args.artifact_dir)
        invocation_evidence.append(evidence)
    for call_index, measured_call, warmup_call in ([] if arm['engine'].get('multi_call', False) else _invocation_plan(calls)):
        warmup, measured, evidence = _run_invocation(
            arm=arm,
            binary=binary,
            deployment_bundle=deployment_bundle,
            invocation_dir=(
                invocations / "warm-and-measured"
                if warmup_call is not None
                else invocations / f"call-{call_index:02d}"
            ),
            artifact_dir=args.artifact_dir,
            call_index=call_index,
            measured_call=measured_call,
            warmup_call=warmup_call,
            vocab_size=vocab_size,
        )
        if warmup_call is not None:
            if warmup is None:
                raise RuntimeError("Vosti paired invocation omitted its first call")
            call_results.append(warmup)
        call_results.append(measured)
        invocation_evidence.append(evidence)

    deployment = json.loads(
        (deployment_bundle / "deployment.json").read_text(encoding="utf-8")
    )
    result = {
        "schema_version": SCHEMA_VERSION,
        "engine": "vosti",
        "mode": arm["execution"]["mode"],
        "hostname": socket.gethostname(),
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
        "backend_evidence": {
            "qualified": True,
            "deployment_sha256": deployment["deployment_sha256"],
            "scope_sha256": deployment["scope_sha256"],
            "cuda_graph": "padded-cover-enabled",
            "invocations": invocation_evidence,
        },
        "calls": call_results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
