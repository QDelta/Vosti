#!/usr/bin/env python3
"""Compare every retained Vosti strict-suite row with Transformers.

The strict determinism suite already records complete float32 vocabulary rows
for batch/singleton, chunking, decode/prefill, and cold/warm-prefix relations.
This checker reconstructs each row's exact token context from ``inputs.json``
and the arm outputs, evaluates the same context with Transformers, and emits a
source- and artifact-bound empirical report.  It does not reclassify a failed
or incomplete strict suite as valid evidence.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch
import transformers


ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT, ROOT / "python"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from scripts.checks.reference import (  # noqa: E402
    FAMILIES,
    _checkpoint_manifest,
    _checkpoint_unchanged,
    _clean_framework_commit,
    _combine_error_summaries,
    _load_reference_model,
    _sha256_file,
    _write_json_report,
)
from scripts.determinism_tests.protocol import (  # noqa: E402
    SCHEMA_VERSION,
    TESTS,
    row_comparison_semantics,
    sha256_json,
)


REPORT_SCHEMA = "vosti.model-reference-suite.v1"


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"cannot load {label} JSON at {path}: {error}") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} JSON is not an object")
    return value


def _expected_arm_names(inputs: dict[str, Any]) -> set[str]:
    batch_prompts = inputs.get("batch_prompts")
    chunk_budgets = inputs.get("chunk_budgets")
    if not isinstance(batch_prompts, list) or not batch_prompts:
        raise RuntimeError("strict-suite inputs have no batch prompts")
    if not isinstance(chunk_budgets, list) or not chunk_budgets:
        raise RuntimeError("strict-suite inputs have no chunk budgets")
    return {
        "batch-vs-single-batch",
        *{
            f"batch-vs-single-single-{index:02d}"
            for index in range(len(batch_prompts))
        },
        *{f"different-chunk-{int(budget)}" for budget in chunk_budgets},
        "prefill-vs-decode-generate",
        "prefill-vs-decode-teacher",
        "cold-vs-warm",
    }


def _validated_suite(summary_path: Path) -> tuple[dict, dict, Path]:
    summary_path = summary_path.expanduser().resolve()
    suite_root = summary_path.parent
    summary = _load_json(summary_path, label="strict-suite summary")
    inputs_path = suite_root / "inputs.json"
    inputs = _load_json(inputs_path, label="strict-suite inputs")
    if (
        summary.get("schema_version") != SCHEMA_VERSION
        or summary.get("acceptance") != "strict"
        or summary.get("status") != "pass"
        or summary.get("all_bitwise_equal") is not True
        or summary.get("relation_pass") != {name: True for name in TESTS}
    ):
        raise RuntimeError("reference input is not a passing strict suite")
    if summary.get("comparison_semantics") != row_comparison_semantics():
        raise RuntimeError("strict suite uses unexpected comparison semantics")
    if summary.get("inputs_sha256") != sha256_json(inputs):
        raise RuntimeError("strict-suite summary does not bind inputs.json")
    if inputs.get("schema_version") != SCHEMA_VERSION:
        raise RuntimeError("strict-suite inputs use an unexpected schema")
    token_inputs = {
        "batch": inputs.get("batch_prompts"),
        "chunk": inputs.get("chunk_prompt"),
        "generation": inputs.get("generation_prompt"),
        "cache": inputs.get("cache_prompt"),
    }
    if inputs.get("token_ids_sha256") != sha256_json(token_inputs):
        raise RuntimeError("strict-suite inputs do not bind their token IDs")
    if summary.get("model") != inputs.get("model"):
        raise RuntimeError("strict-suite summary and inputs name different models")
    arms = summary.get("arms")
    if not isinstance(arms, list) or not arms:
        raise RuntimeError("strict suite has no arms")
    arm_names = [arm.get("name") for arm in arms if isinstance(arm, dict)]
    if len(arm_names) != len(arms) or set(arm_names) != _expected_arm_names(inputs):
        raise RuntimeError("strict suite does not contain its exact required arms")
    if len(set(arm_names)) != len(arm_names):
        raise RuntimeError("strict suite contains duplicate arms")
    for arm in arms:
        telemetry = arm.get("telemetry") if isinstance(arm, dict) else None
        if not isinstance(telemetry, dict) or telemetry.get("status") != "complete":
            raise RuntimeError("strict suite contains incomplete telemetry")
        result_path = Path(str(arm.get("result"))).resolve()
        if (
            not result_path.is_relative_to(suite_root / "arms")
            or result_path.name != "result.json"
        ):
            raise RuntimeError("strict-suite arm result escapes its evidence root")
        telemetry_path = Path(str(telemetry.get("path"))).resolve()
        if (
            telemetry_path != result_path.parent / "telemetry.json"
            or telemetry.get("sha256") != _sha256_file(telemetry_path)
        ):
            raise RuntimeError("strict-suite telemetry identity is inconsistent")
    return summary, inputs, suite_root


def _reference_model_path(requested: Path | None, inputs: dict) -> Path:
    """Resolve the checkpoint recorded by this suite, not a family default."""

    recorded = inputs.get("model_path")
    if not isinstance(recorded, str) or not recorded:
        raise RuntimeError("strict suite has no checkpoint path")
    suite_model_path = Path(recorded).expanduser().resolve()
    if requested is None:
        return suite_model_path
    model_path = requested.expanduser().resolve()
    if model_path != suite_model_path:
        raise RuntimeError("reference checkpoint path differs from strict suite")
    return model_path


def _generation_tokens(results: dict[str, dict], expected_count: int) -> list[int]:
    try:
        calls = results["prefill-vs-decode-generate"]["calls"]
        requests = calls[0]["requests"]
        tokens = requests[0]["output_token_ids"]
    except (KeyError, IndexError, TypeError) as error:
        raise RuntimeError("strict suite has no decode-generation witness") from error
    if not isinstance(tokens, list) or len(tokens) != expected_count:
        raise RuntimeError("decode-generation witness has the wrong token count")
    return [int(token) for token in tokens]


def _prompt_for_request(
    arm_name: str,
    request_index: int,
    request: dict,
    inputs: dict,
    generation_tokens: list[int],
) -> list[int]:
    if arm_name == "batch-vs-single-batch":
        prompt = inputs["batch_prompts"][request_index]
    elif arm_name.startswith("batch-vs-single-single-"):
        prompt = inputs["batch_prompts"][int(arm_name.rsplit("-", 1)[1])]
    elif arm_name.startswith("different-chunk-"):
        prompt = inputs["chunk_prompt"]
    elif arm_name == "prefill-vs-decode-generate":
        prompt = inputs["generation_prompt"]
    elif arm_name == "prefill-vs-decode-teacher":
        extra = int(request["prompt_length"]) - len(inputs["generation_prompt"])
        if not 0 <= extra <= len(generation_tokens):
            raise RuntimeError("teacher-prefill prompt length is inconsistent")
        prompt = inputs["generation_prompt"] + generation_tokens[:extra]
    elif arm_name == "cold-vs-warm":
        prompt = inputs["cache_prompt"]
    else:
        raise RuntimeError(f"unsupported strict-suite arm {arm_name!r}")
    prompt = [int(token) for token in prompt]
    if len(prompt) != int(request.get("prompt_length", -1)):
        raise RuntimeError(f"{arm_name} request prompt length is inconsistent")
    if sha256_json(prompt) != request.get("prompt_sha256"):
        raise RuntimeError(f"{arm_name} request prompt digest is inconsistent")
    return prompt


def _load_engine_row(
    arm_dir: Path,
    descriptor: dict,
) -> np.ndarray:
    artifact = arm_dir / "rows" / str(descriptor.get("artifact"))
    if not artifact.resolve().is_relative_to(arm_dir / "rows"):
        raise RuntimeError("strict-suite row artifact escapes its evidence root")
    row = np.load(artifact, allow_pickle=False)
    if row.ndim != 1 or row.dtype != np.float32:
        raise RuntimeError("strict-suite row is not a float32 vocabulary vector")
    if descriptor.get("shape") != list(row.shape):
        raise RuntimeError("strict-suite row shape disagrees with its record")
    if descriptor.get("sha256") != hashlib.sha256(
        row.tobytes(order="C")
    ).hexdigest():
        raise RuntimeError("strict-suite row digest disagrees with its record")
    if not np.isfinite(row).all() or descriptor.get("finite") is not True:
        raise RuntimeError("strict-suite row is not finite")
    return row


def _collect_contexts(
    summary: dict,
    inputs: dict,
) -> tuple[
    dict[tuple[int, ...], list[dict[str, Any]]],
    str,
    str,
    list[dict[str, str]],
]:
    results = {}
    arm_dirs = {}
    arm_provenance = []
    for arm in summary["arms"]:
        arm_name = str(arm["name"])
        result_path = Path(str(arm["result"])).resolve()
        arm_path = result_path.parent / "arm.json"
        arm_sha256 = str(arm.get("arm_sha256"))
        retained_arm = _load_json(arm_path, label=f"{arm_name} arm")
        result = _load_json(result_path, label=f"{arm_name} result")
        if (
            sha256_json(retained_arm) != arm_sha256
            or result.get("arm_sha256") != arm_sha256
        ):
            raise RuntimeError(f"{arm_name} arm identity is inconsistent")
        if (
            result.get("schema_version") != SCHEMA_VERSION
            or result.get("engine") != "vosti"
            or Path(str(result.get("model_path"))).resolve()
            != Path(str(inputs["model_path"])).resolve()
        ):
            raise RuntimeError(f"{arm_name} result identity is inconsistent")
        results[arm_name] = result
        arm_dirs[arm_name] = result_path.parent
        arm_provenance.append(
            {
                "name": arm_name,
                "arm_sha256": arm_sha256,
                "result_sha256": _sha256_file(result_path),
                "telemetry_sha256": str(arm["telemetry"]["sha256"]),
            }
        )
    generation_tokens = _generation_tokens(
        results, int(inputs.get("generation_tokens", -1))
    )
    contexts: dict[tuple[int, ...], list[dict[str, Any]]] = {}
    framework_commits = set()
    deployments = set()
    for arm_name, result in results.items():
        backend = result.get("backend_evidence")
        if not isinstance(backend, dict) or backend.get("qualified") is not True:
            raise RuntimeError(f"{arm_name} is not qualified Engine evidence")
        framework_commit = backend.get("framework_commit")
        deployment = backend.get("deployment_sha256")
        if not isinstance(framework_commit, str) or len(framework_commit) != 40:
            raise RuntimeError(f"{arm_name} has no framework commit")
        if not isinstance(deployment, str) or len(deployment) != 64:
            raise RuntimeError(f"{arm_name} has no deployment identity")
        framework_commits.add(framework_commit)
        deployments.add(deployment)
        arm_dir = arm_dirs[arm_name]
        calls = result.get("calls")
        if not isinstance(calls, list) or not calls:
            raise RuntimeError(f"{arm_name} has no calls")
        if any(not isinstance(call, dict) for call in calls):
            raise RuntimeError(f"{arm_name} has a malformed call")
        expected_request_counts = (
            [len(inputs["batch_prompts"])]
            if arm_name == "batch-vs-single-batch"
            else [1, 1]
            if arm_name == "cold-vs-warm"
            else [1]
        )
        if [len(call.get("requests", [])) for call in calls] != (
            expected_request_counts
        ):
            raise RuntimeError(f"{arm_name} has unexpected call/request coverage")
        for call_index, call in enumerate(calls):
            requests = call.get("requests") if isinstance(call, dict) else None
            if not isinstance(requests, list) or not requests:
                raise RuntimeError(f"{arm_name} call has no requests")
            for request_index, request in enumerate(requests):
                prompt = _prompt_for_request(
                    arm_name,
                    request_index,
                    request,
                    inputs,
                    generation_tokens,
                )
                tokens = [int(token) for token in request.get("output_token_ids", [])]
                expected_token_count = (
                    len(generation_tokens)
                    if arm_name == "prefill-vs-decode-generate"
                    else 2
                    if arm_name == "cold-vs-warm"
                    else 1
                )
                if len(tokens) != expected_token_count:
                    raise RuntimeError(f"{arm_name} has the wrong output token count")
                descriptor = request.get("last_row")
                if not isinstance(descriptor, dict):
                    raise RuntimeError(f"{arm_name} request has no retained row")
                metadata = descriptor.get("metadata")
                if not isinstance(metadata, dict):
                    raise RuntimeError(f"{arm_name} retained row has no metadata")
                position = int(metadata.get("generated_position", -1))
                if not 0 <= position < len(tokens):
                    raise RuntimeError(f"{arm_name} retained row position is invalid")
                expected_position = (
                    len(generation_tokens) - 1
                    if arm_name == "prefill-vs-decode-generate"
                    else 0
                )
                if position != expected_position:
                    raise RuntimeError(
                        f"{arm_name} retained the wrong generation position"
                    )
                if int(descriptor.get("argmax", -1)) != tokens[position]:
                    raise RuntimeError(f"{arm_name} retained row disagrees with output")
                context = tuple(prompt + tokens[:position])
                contexts.setdefault(context, []).append(
                    {
                        "arm": arm_name,
                        "call_index": call_index,
                        "request_index": request_index,
                        "generated_position": position,
                        "output_token": tokens[position],
                        "row_sha256": descriptor["sha256"],
                        "row": _load_engine_row(arm_dir, descriptor),
                    }
                )
        if arm_name == "cold-vs-warm" and (
            int(calls[0]["requests"][0].get("num_cached_tokens", -1)) != 0
            or int(calls[1]["requests"][0].get("num_cached_tokens", 0)) <= 0
        ):
            raise RuntimeError("cold/warm witness has invalid cache-reuse telemetry")
    expected_contexts = len(inputs["batch_prompts"]) + 3
    expected_rows = (
        2 * len(inputs["batch_prompts"])
        + len(inputs["chunk_budgets"])
        + 4
    )
    if len(contexts) != expected_contexts or sum(map(len, contexts.values())) != (
        expected_rows
    ):
        raise RuntimeError("strict suite has incomplete retained-row coverage")
    for observations in contexts.values():
        if len(observations) < 2:
            raise RuntimeError("strict suite has an unpaired retained context")
        first = observations[0]["row"]
        if any(
            first.shape != observation["row"].shape
            or first.dtype != observation["row"].dtype
            or first.tobytes(order="C")
            != observation["row"].tobytes(order="C")
            for observation in observations[1:]
        ):
            raise RuntimeError("strict suite retained rows are not bitwise equal")
    if len(framework_commits) != 1 or len(deployments) != 1:
        raise RuntimeError("strict suite mixes framework or deployment identities")
    return (
        contexts,
        next(iter(framework_commits)),
        next(iter(deployments)),
        arm_provenance,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("summary", type=Path)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-reference-margin", type=float, default=0.1)
    parser.add_argument("--max-logit-abs-error", type=float, default=1.0)
    args = parser.parse_args()
    if args.min_reference_margin < 0.0:
        parser.error("--min-reference-margin must be nonnegative")
    if args.max_logit_abs_error < 0.0:
        parser.error("--max-logit-abs-error must be nonnegative")
    output_path = args.output.expanduser().resolve()
    if output_path.exists():
        parser.error("--output already exists")
    if output_path.is_relative_to(ROOT):
        parser.error("--output must be outside the source checkout")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    validator_commit = _clean_framework_commit()
    summary, inputs, suite_root = _validated_suite(args.summary)
    family = summary.get("model", {}).get("key")
    if family not in FAMILIES or inputs.get("model", {}).get("key") != family:
        raise RuntimeError("strict suite names an unsupported or inconsistent model")
    model_path = _reference_model_path(args.model, inputs)
    checkpoint = _checkpoint_manifest(model_path)
    contexts, framework_commit, deployment, arm_provenance = _collect_contexts(
        summary, inputs
    )

    device = torch.device(os.environ.get("CUDA_DEVICE", "cuda:0"))
    model, language_model = _load_reference_model(family, model_path, device)
    context_reports = []
    all_pairs = []
    minimum_margin = float("inf")
    with torch.inference_mode():
        for context, observations in contexts.items():
            input_ids = torch.tensor([context], dtype=torch.long, device=device)
            attention_mask = torch.ones_like(input_ids)
            logits = language_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
            ).logits
            reference = logits[0, -1].float().cpu().contiguous().numpy()
            top_values, top_indices = logits[0, -1].float().topk(2)
            reference_argmax = int(top_indices[0].item())
            reference_margin = float((top_values[0] - top_values[1]).item())
            minimum_margin = min(minimum_margin, reference_margin)
            pairs = [(reference, observation["row"]) for observation in observations]
            errors = _combine_error_summaries(
                pairs, label="strict-suite Engine/reference logits"
            )
            for observation in observations:
                if observation["output_token"] != reference_argmax:
                    raise AssertionError(
                        f"{observation['arm']} selected "
                        f"{observation['output_token']}, Transformers selected "
                        f"{reference_argmax}"
                    )
            all_pairs.extend(pairs)
            context_reports.append(
                {
                    "context_length": len(context),
                    "context_sha256": sha256_json(list(context)),
                    "reference_argmax": reference_argmax,
                    "reference_margin": reference_margin,
                    "max_abs_error": errors["max_abs_error"],
                    "mean_abs_error": errors["mean_abs_error"],
                    "engine_rows": [
                        {key: value for key, value in observation.items() if key != "row"}
                        for observation in observations
                    ],
                }
            )
            del input_ids, attention_mask, logits, top_values, top_indices
    aggregate = _combine_error_summaries(
        all_pairs, label="strict-suite Engine/reference logits"
    )
    del language_model, model
    gc.collect()
    torch.cuda.empty_cache()
    _checkpoint_unchanged(model_path, checkpoint)

    if minimum_margin < args.min_reference_margin:
        raise AssertionError(
            f"minimum Transformers margin {minimum_margin:.6g} is below "
            f"{args.min_reference_margin:.6g}"
        )
    if aggregate["max_abs_error"] > args.max_logit_abs_error:
        raise AssertionError(
            f"Engine/reference max absolute error "
            f"{aggregate['max_abs_error']:.6g} exceeds "
            f"{args.max_logit_abs_error:.6g}"
        )

    report = {
        "schema": REPORT_SCHEMA,
        "status": "pass",
        "validator_framework_commit": validator_commit,
        "strict_suite": {
            "summary": str(args.summary.expanduser().resolve()),
            "summary_sha256": _sha256_file(args.summary.expanduser().resolve()),
            "inputs_sha256": summary["inputs_sha256"],
            "framework_commit": framework_commit,
            "deployment_sha256": deployment,
            "comparison_semantics": summary["comparison_semantics"],
            "arms": arm_provenance,
        },
        "model": {
            "family": family,
            "path": str(model_path),
            "checkpoint": checkpoint,
        },
        "reference_environment": {
            "torch": str(torch.__version__),
            "transformers": str(transformers.__version__),
            "cuda_runtime": torch.version.cuda,
            "device": str(device),
            "device_name": torch.cuda.get_device_name(device),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
        "thresholds": {
            "min_reference_margin": args.min_reference_margin,
            "max_logit_abs_error": args.max_logit_abs_error,
        },
        "comparison": {
            "unique_contexts": len(contexts),
            "engine_rows": len(all_pairs),
            "minimum_reference_margin": minimum_margin,
            "max_abs_error": aggregate["max_abs_error"],
            "mean_abs_error": aggregate["mean_abs_error"],
            "all_argmax_equal": True,
        },
        "contexts": context_reports,
    }
    _write_json_report(args.output, report)
    print(
        f"{family} strict-suite Transformers comparison passed: "
        f"contexts={len(contexts)} rows={len(all_pairs)} "
        f"minimum_reference_margin={minimum_margin:.6g} "
        f"max_logit_abs_error={aggregate['max_abs_error']:.6g} "
        f"mean_logit_abs_error={aggregate['mean_abs_error']:.6g}"
    )


if __name__ == "__main__":
    main()
