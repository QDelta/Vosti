#!/usr/bin/env python3
"""Run the four deterministic relations for one model/execution configuration."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.common.telemetry import load_complete_telemetry  # noqa: E402
from scripts.determinism_tests.protocol import (  # noqa: E402
    ALL_EXECUTION_CONFIGS,
    MODELS,
    PROMPT_SEED,
    SCHEMA_VERSION,
    build_inputs,
    compare_row_artifacts,
    row_comparison_semantics,
    sha256_json,
)


def _by_key(values, key: str):
    for value in values:
        if value.key == key:
            return value
    raise KeyError(key)


def _arm(
    *,
    inputs: dict[str, Any],
    execution: Any,
    calls: list[dict[str, Any]],
    prefix_caching: bool = False,
    max_num_batched_tokens: int = 4096,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "model": inputs["model"],
        "model_path": inputs["model_path"],
        "execution": asdict(execution),
        "engine": {
            "prefix_caching": prefix_caching,
            "gpu_memory_utilization": 0.80,
            "max_num_seqs": 8,
            "max_num_batched_tokens": max_num_batched_tokens,
            "max_model_len": 1216,
        },
        "calls": calls,
    }


def _row_path(arm_dir: Path, result: dict[str, Any], call: int, request: int) -> Path:
    record = result["calls"][call]["requests"][request]["last_row"]
    if not record:
        raise RuntimeError("arm has no retained row")
    return arm_dir / "rows" / record["artifact"]


def _selected_backend_from_logs(
    engine: str, stdout_text: str, stderr_text: str
) -> dict[str, str] | None:
    """Extract the backend that the engine says it actually selected."""

    combined = stdout_text + "\n" + stderr_text
    if engine == "vllm":
        match = re.search(r"Using ([A-Z0-9_]+) attention backend", combined)
        if match:
            return {
                "selected_attention_backend": match.group(1),
                "selection_evidence": "vllm attention-backend selection log",
            }
    elif engine == "sglang":
        # Explicit and normalized in ServerArgs after deterministic-mode
        # rewriting.  Auto selection can be reported by lower-level logs.
        match = re.search(r"attention_backend='([^']+)'", combined)
        if match and match.group(1) != "None":
            return {
                "selected_attention_backend": match.group(1),
                "selection_evidence": "sglang normalized ServerArgs log",
            }
        patterns = (
            r"Using ([A-Za-z0-9_]+) attention backend",
            r"attention backend(?: is|:|=) ['\"]?([A-Za-z0-9_]+)",
        )
        for pattern in patterns:
            match = re.search(pattern, combined, flags=re.IGNORECASE)
            if match and match.group(1).lower() not in {"none", "auto"}:
                return {
                    "selected_attention_backend": match.group(1),
                    "selection_evidence": "sglang attention-backend selection log",
                }
    return None


def _enrich_backend_evidence(
    arm_dir: Path, result_path: Path, result: dict[str, Any], engine: str
) -> dict[str, Any]:
    evidence = _selected_backend_from_logs(
        engine,
        (arm_dir / "stdout.log").read_text(encoding="utf-8", errors="replace"),
        (arm_dir / "stderr.log").read_text(encoding="utf-8", errors="replace"),
    )
    if evidence is None:
        return result
    retained = result.setdefault("backend_evidence", {})
    for key, value in evidence.items():
        prior = retained.get(key)
        if prior is not None and prior != value:
            raise RuntimeError(
                f"conflicting retained backend evidence for {arm_dir}: "
                f"{key}={prior!r}, log={value!r}"
            )
        retained[key] = value
    result_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def _retained_arm_is_reusable(
    arm_path: Path, telemetry_path: Path, expected_arm: dict[str, Any], name: str
) -> bool:
    """Validate retained arm identity and accept only completed telemetry."""

    prior_arm = json.loads(arm_path.read_text(encoding="utf-8"))
    if sha256_json(prior_arm) != sha256_json(expected_arm):
        raise RuntimeError(f"cannot resume {name}: retained arm configuration differs")
    retained_telemetry = json.loads(telemetry_path.read_text(encoding="utf-8"))
    return retained_telemetry.get("status") == "complete"


class SuiteRunner:
    def __init__(
        self,
        *,
        output: Path,
        worker_python: Path,
        gpu_index: int,
        execution: Any,
        arm_engine_fields: dict[str, Any] | None = None,
        worker_root: Path | None = None,
        cohort_root: str | None = None,
    ) -> None:
        self.output = output
        self.worker_python = worker_python
        self.gpu_index = gpu_index
        self.execution = execution
        self.arm_engine_fields = arm_engine_fields or {}
        self.arm_records: list[dict[str, Any]] = []
        self.worker_root = worker_root.resolve() if worker_root else ROOT
        self.cohort_root = cohort_root

    def run_arm(self, name: str, arm: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
        root = self.worker_root
        arm["engine"].update(self.arm_engine_fields)
        if self.cohort_root is not None:
            arm['concurrency_policy'] = 'owned_correctness_cohort'
        base_dir = self.output / "arms" / name
        arm_dir = base_dir
        attempt = 1
        while arm_dir.exists():
            arm_path = arm_dir / "arm.json"
            result_path = arm_dir / "result.json"
            telemetry_path = arm_dir / "telemetry.json"
            if arm_path.is_file() and result_path.is_file() and telemetry_path.is_file():
                if not _retained_arm_is_reusable(
                    arm_path, telemetry_path, arm, name
                ):
                    attempt += 1
                    arm_dir = self.output / "arms" / f"{name}-retry-{attempt}"
                    continue
                telemetry = load_complete_telemetry(telemetry_path, allow_cohort=self.cohort_root is not None)
                result = json.loads(result_path.read_text(encoding="utf-8"))
                result = _enrich_backend_evidence(
                    arm_dir, result_path, result, self.execution.engine
                )
                self.arm_records.append(
                    {
                        "name": name,
                        "attempt": attempt,
                        "reused": True,
                        "arm_sha256": sha256_json(arm),
                        "result": str(result_path),
                        "telemetry": telemetry,
                    }
                )
                return arm_dir, result
            attempt += 1
            arm_dir = self.output / "arms" / f"{name}-retry-{attempt}"
        arm_dir.mkdir(parents=True, exist_ok=False)
        arm_path = arm_dir / "arm.json"
        result_path = arm_dir / "result.json"
        telemetry_path = arm_dir / "telemetry.json"
        arm_path.write_text(json.dumps(arm, indent=2, sort_keys=True) + "\n")
        command = [
            sys.executable,
            str(root / "scripts/common/gpu_monitor.py"),
            "--gpu-index",
            str(self.gpu_index),
            "--output",
            str(telemetry_path),
            *(['--cohort-root', self.cohort_root] if self.cohort_root is not None else []),
            "--",
            str(self.worker_python),
            "-m",
            f"scripts.determinism_tests.{self.execution.engine}_worker",
            "--arm",
            str(arm_path),
            "--output",
            str(result_path),
            "--artifact-dir",
            str(arm_dir / "rows"),
        ]
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(self.gpu_index)
        # Root /tmp is shared across users and can be capacity-constrained.
        # Keep every compiler and tempfile artifact attributable to this arm
        # under the external experiment record instead.
        arm_tmp = arm_dir / "tmp"
        arm_tmp.mkdir()
        short_tmp = Path(tempfile.gettempdir()) / (
            "dt-"
            + sha256_json(
                {
                    "suite": str(self.output),
                    "arm": name,
                    "attempt": attempt,
                }
            )[:16]
        )
        if short_tmp.exists() or short_tmp.is_symlink():
            if not short_tmp.is_symlink() or short_tmp.resolve() != arm_tmp.resolve():
                raise RuntimeError(f"short TMPDIR alias collision at {short_tmp}")
        else:
            short_tmp.symlink_to(arm_tmp, target_is_directory=True)
        env["TMPDIR"] = str(short_tmp)
        compiler_cache = self.output / "compiler-cache"
        compiler_cache.mkdir(exist_ok=True)
        # Fresh engine state does not require recompiling identical source.
        # Sharing only the on-disk compiler cache within one model/config
        # suite keeps the compiled artifact fixed across its comparison arms.
        env["TORCHINDUCTOR_CACHE_DIR"] = str(compiler_cache / "torchinductor")
        env["TRITON_CACHE_DIR"] = str(compiler_cache / "triton")
        env["CUDA_CACHE_PATH"] = str(compiler_cache / "cuda-cache")
        python_paths = [str(root)]
        if self.execution.engine == "vllm":
            env["VLLM_BATCH_INVARIANT"] = (
                "1" if self.execution.mode == "invariant" else "0"
            )
            env["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
        elif self.execution.engine == "sglang":
            env["VOSTI_SGLANG_LOGITS_OBSERVER"] = "1"
            env["VOSTI_SGLANG_REQUEST_OBSERVER"] = "1" if arm['engine'].get('trace_requests', False) else "0"
            env["VOSTI_LOGITS_OBSERVER_DIR"] = str(arm_dir / "rows")
            python_paths = [
                str(root / "scripts/determinism_tests/hooks"),
                str(root / "python"),
                str(root),
            ]
        elif self.execution.engine == "vosti":
            env["VOSTI_KERNELS_LOGITS_OBSERVER"] = "1"
            python_paths = [
                str(root / "scripts/determinism_tests/hooks"),
                str(root / "python"),
                str(root),
            ]
        else:
            raise RuntimeError(
                f"execution engine {self.execution.engine!r} is not implemented"
            )
        env["PYTHONPATH"] = os.pathsep.join(python_paths) + (
            os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
        )
        with (arm_dir / "stdout.log").open("xb") as stdout, (
            arm_dir / "stderr.log"
        ).open("xb") as stderr:
            completed = subprocess.run(command, cwd=root, env=env, stdout=stdout, stderr=stderr)
        telemetry = load_complete_telemetry(telemetry_path, allow_cohort=self.cohort_root is not None)
        if completed.returncode != 0:
            raise RuntimeError(
                f"arm {name} failed with code {completed.returncode}; "
                f"see {arm_dir / 'stderr.log'}"
            )
        result = json.loads(result_path.read_text(encoding="utf-8"))
        result = _enrich_backend_evidence(
            arm_dir, result_path, result, self.execution.engine
        )
        self.arm_records.append(
            {
                "name": name,
                "attempt": attempt,
                "reused": False,
                "arm_sha256": sha256_json(arm),
                "result": str(result_path),
                "telemetry": telemetry,
            }
        )
        return arm_dir, result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execution-config", required=True)
    parser.add_argument("--model", choices=[model.key for model in MODELS], required=True)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--hardware", choices=("h200", "a100"), required=True)
    parser.add_argument("--gpu-index", type=int, required=True)
    parser.add_argument("--worker-python", type=Path, required=True)
    parser.add_argument("--vosti-binary", type=Path)
    parser.add_argument("--deployment-bundle", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seed", type=lambda value: int(value, 0), default=PROMPT_SEED)
    parser.add_argument("--generated-prefix-only", action="store_true",
                        help="qualify reuse of generated KV pages (Vosti padded graph only)")
    args = parser.parse_args()

    execution = _by_key(ALL_EXECUTION_CONFIGS, args.execution_config)
    if args.generated_prefix_only and execution.key != "vosti-padded-graph":
        parser.error("--generated-prefix-only requires vosti-padded-graph")
    if execution.engine not in {"vosti", "vllm", "sglang"}:
        parser.error("unsupported execution engine")
    arm_engine_fields: dict[str, Any] = {}
    if execution.engine == "vosti":
        if args.vosti_binary is None or args.deployment_bundle is None:
            parser.error(
                "Vosti requires --vosti-binary and --deployment-bundle"
            )
        binary = args.vosti_binary.expanduser().resolve()
        deployment_bundle = args.deployment_bundle.expanduser().resolve()
        if not binary.is_file():
            parser.error(f"Vosti binary does not exist: {binary}")
        if not (deployment_bundle / "deployment.json").is_file():
            parser.error(
                f"deployment bundle is incomplete: {deployment_bundle}"
            )
        arm_engine_fields = {
            "vosti_binary": str(binary),
            "deployment_bundle": str(deployment_bundle),
            "num_blocks": 64,
        }
    model = _by_key(MODELS, args.model)
    model_path = (args.model_path or Path(model.default_path)).expanduser().resolve()
    output = args.output.expanduser().resolve()
    if output.exists() and not args.resume:
        raise FileExistsError(f"refusing to overwrite {output}; pass --resume")
    if not output.exists():
        output.mkdir(parents=True)
    inputs = build_inputs(model, model_path, seed=args.seed)
    inputs_path = output / "inputs.json"
    if inputs_path.exists():
        retained_inputs = json.loads(inputs_path.read_text(encoding="utf-8"))
        if retained_inputs != inputs:
            raise RuntimeError("cannot resume: retained deterministic inputs differ")
    else:
        inputs_path.write_text(
            json.dumps(inputs, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    if (output / "summary.json").exists():
        raise RuntimeError("suite already has a completed summary")
    runner = SuiteRunner(
        output=output,
        # Do not resolve this path: a virtual environment's ``python`` is
        # normally a symlink to the system executable, and resolving it would
        # silently discard that environment's site-packages.
        worker_python=Path(os.path.abspath(args.worker_python.expanduser())),
        gpu_index=args.gpu_index,
        execution=execution,
        arm_engine_fields=arm_engine_fields,
    )
    comparisons: dict[str, list[dict[str, Any]]] = {}
    if args.generated_prefix_only:
        from scripts.determinism_tests.generated_prefix import run_check
        run_check(runner, inputs, execution, hardware=args.hardware)
        return

    batch_arm = _arm(
        inputs=inputs,
        execution=execution,
        calls=[{
            "label": "batch",
            "prompts": inputs["batch_prompts"],
            "max_tokens": 1,
            "record_last_rows": True,
        }],
    )
    batch_dir, batch_result = runner.run_arm("batch-vs-single-batch", batch_arm)
    singleton_rows = []
    for index, prompt in enumerate(inputs["batch_prompts"]):
        single_arm = _arm(
            inputs=inputs,
            execution=execution,
            calls=[{
                "label": f"single-{index}",
                "prompts": [prompt],
                "max_tokens": 1,
                "record_last_rows": True,
            }],
        )
        single_dir, single_result = runner.run_arm(
            f"batch-vs-single-single-{index:02d}", single_arm
        )
        singleton_rows.append(_row_path(single_dir, single_result, 0, 0))
    comparisons["batch_vs_single"] = [
        compare_row_artifacts(
            _row_path(batch_dir, batch_result, 0, index), singleton_rows[index]
        )
        for index in range(len(singleton_rows))
    ]

    chunk_rows = []
    for budget in inputs["chunk_budgets"]:
        arm = _arm(
            inputs=inputs,
            execution=execution,
            max_num_batched_tokens=int(budget),
            calls=[{
                "label": f"chunk-{budget}",
                "prompts": [inputs["chunk_prompt"]],
                "max_tokens": 1,
                "record_last_rows": True,
            }],
        )
        arm_dir, result = runner.run_arm(f"different-chunk-{budget}", arm)
        chunk_rows.append(_row_path(arm_dir, result, 0, 0))
    comparisons["different_chunk"] = [
        compare_row_artifacts(chunk_rows[0], path) for path in chunk_rows[1:]
    ]

    generate_arm = _arm(
        inputs=inputs,
        execution=execution,
        calls=[{
            "label": "decode-generation",
            "prompts": [inputs["generation_prompt"]],
            "max_tokens": int(inputs["generation_tokens"]),
            "record_last_rows": True,
            "require_graph_replay": execution.engine == "vosti",
        }],
    )
    generate_dir, generate_result = runner.run_arm("prefill-vs-decode-generate", generate_arm)
    generated = generate_result["calls"][0]["requests"][0]["output_token_ids"]
    teacher_prompt = inputs["generation_prompt"] + generated[:-1]
    teacher_arm = _arm(
        inputs=inputs,
        execution=execution,
        calls=[{
            "label": "teacher-prefill",
            "prompts": [teacher_prompt],
            "max_tokens": 1,
            "record_last_rows": True,
        }],
    )
    teacher_dir, teacher_result = runner.run_arm("prefill-vs-decode-teacher", teacher_arm)
    comparisons["prefill_vs_decode"] = [
        compare_row_artifacts(
            _row_path(generate_dir, generate_result, 0, 0),
            _row_path(teacher_dir, teacher_result, 0, 0),
        )
    ]

    cache_max_tokens = 2 if execution.engine == "vosti" else 1
    cache_record_position = 0 if execution.engine == "vosti" else None
    cache_arm = _arm(
        inputs=inputs,
        execution=execution,
        prefix_caching=True,
        calls=[
            {
                "label": "cold-cache-donor",
                "prompts": [inputs["cache_prompt"]],
                "max_tokens": cache_max_tokens,
                "record_last_rows": True,
                "record_generated_position": cache_record_position,
            },
            {
                "label": "warm",
                "prompts": [inputs["cache_prompt"]],
                "max_tokens": cache_max_tokens,
                "record_last_rows": True,
                "record_generated_position": cache_record_position,
            },
        ],
    )
    cache_dir, cache_result = runner.run_arm("cold-vs-warm", cache_arm)
    cached_tokens = cache_result["calls"][1]["requests"][0]["num_cached_tokens"]
    if cached_tokens <= 0:
        raise RuntimeError("warm arm reported no cached tokens")
    comparisons["cold_vs_warm"] = [
        {
            **compare_row_artifacts(
                _row_path(cache_dir, cache_result, 0, 0),
                _row_path(cache_dir, cache_result, 1, 0),
            ),
            "warm_num_cached_tokens": cached_tokens,
        }
    ]

    relation_pass = {
        name: all(row["bitwise_equal"] for row in rows)
        for name, rows in comparisons.items()
    }
    all_equal = all(relation_pass.values())
    summary = {
        "schema_version": SCHEMA_VERSION,
        "hardware": args.hardware,
        "model": asdict(model),
        "execution": asdict(execution),
        "inputs_sha256": sha256_json(inputs),
        "comparison_semantics": row_comparison_semantics(),
        "arms": runner.arm_records,
        "comparisons": comparisons,
        "relation_pass": relation_pass,
        "all_bitwise_equal": all_equal,
        "acceptance": "strict" if execution.strict else "observational",
        "status": (
            "pass"
            if all_equal
            else "fail" if execution.strict else "observed-difference"
        ),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    if execution.strict and not all_equal:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
