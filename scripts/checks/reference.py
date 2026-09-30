#!/usr/bin/env python3
"""Compare Engine logits, tokens, and optional KV writes with Transformers."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT, ROOT / "python"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from scripts.determinism_tests.protocol import load_observer_records  # noqa: E402
from scripts.determinism_tests.kv_observer import (  # noqa: E402
    load_kv_observer_records,
)


from scripts.common.model_paths import checkpoint_path

FAMILIES = {
    "qwen3": {
        "architecture": "qwen3",
        "default_model": checkpoint_path("Qwen3-0.6B"),
        "default_prompt": "The capital of France is",
    },
    "gemma3": {
        "architecture": "gemma3_text",
        "default_model": checkpoint_path("gemma-3-4b-it"),
        "default_prompt": "The capital of France is",
    },
    "llama3": {
        "architecture": "llama3",
        "default_model": checkpoint_path("Llama-3.1-8B"),
        "default_prompt": "The capital of France is Paris. The capital of Germany is",
    },
    "gemma4": {
        "architecture": "gemma4_text",
        "default_model": checkpoint_path("gemma-4-31b-it"),
        "default_prompt": "The capital of France is",
    },
}

REPORT_SCHEMA = "vosti.model-reference.v1"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _clean_framework_commit() -> str:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=ROOT,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout
    if status:
        raise RuntimeError("reference evidence requires a clean framework checkout")
    return commit


def _checkpoint_manifest(model_path: Path) -> dict:
    model_path = model_path.expanduser().resolve()
    paths = sorted(model_path.glob("model*.safetensors"))
    index = model_path / "model.safetensors.index.json"
    if index.is_file():
        paths.append(index)
    config = model_path / "config.json"
    if not paths or not config.is_file():
        raise RuntimeError("reference checkpoint has no config or safetensors weights")
    paths.append(config)
    files = []
    for path in paths:
        resolved = path.resolve()
        if not resolved.is_relative_to(model_path):
            raise RuntimeError("reference checkpoint file escapes its model directory")
        stat = resolved.stat()
        files.append(
            {
                "name": resolved.relative_to(model_path).as_posix(),
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "sha256": _sha256_file(resolved),
            }
        )
    canonical = json.dumps(
        [{key: row[key] for key in ("name", "size", "sha256")} for row in files],
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return {"sha256": hashlib.sha256(canonical).hexdigest(), "files": files}


def _checkpoint_unchanged(model_path: Path, manifest: dict) -> None:
    for record in manifest["files"]:
        stat = (model_path / record["name"]).stat()
        if stat.st_size != record["size"] or stat.st_mtime_ns != record["mtime_ns"]:
            raise RuntimeError("reference checkpoint changed during comparison")


def _write_json_report(path: Path, report: dict) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def _deployment_evidence(
    family_name: str,
    family: dict,
    model_path: Path,
) -> dict:
    raw_bundle = os.environ.get("VOSTI_DEPLOYMENT_BUNDLE")
    if not raw_bundle:
        raise RuntimeError("retained reference evidence requires a deployment bundle")
    bundle = Path(raw_bundle).expanduser().resolve()
    deployment_path = bundle / "deployment.json"
    deployment = json.loads(deployment_path.read_text(encoding="utf-8"))
    config_sha256 = _sha256_file(model_path / "config.json")
    if (
        deployment.get("qualified") is not True
        or deployment.get("architecture") != family["architecture"]
        or deployment.get("model", {}).get("config_sha256") != config_sha256
    ):
        raise RuntimeError(
            f"{family_name} deployment is not qualified for this architecture and checkpoint"
        )
    return {
        "bundle": str(bundle),
        "deployment_sha256": deployment["deployment_sha256"],
        "deployment_file_sha256": _sha256_file(deployment_path),
        "scope_sha256": deployment["scope_sha256"],
    }


def _read_outputs(path: Path) -> list[list[int]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    indexed = []
    for line in lines:
        raw_index, raw_tokens = line.split("\t", 1)
        indexed.append(
            (int(raw_index), [int(token) for token in raw_tokens.split(",") if token])
        )
    indexed.sort()
    if [index for index, _ in indexed] != list(range(len(indexed))):
        raise RuntimeError("Engine output request indexes are not contiguous")
    return [tokens for _, tokens in indexed]


def _read_single_output(path: Path) -> int:
    outputs = _read_outputs(path)
    if len(outputs) != 1 or len(outputs[0]) != 1:
        raise RuntimeError(
            f"Engine output is not one token for request zero: {outputs!r}"
        )
    return outputs[0][0]


def _cache_rows(
    past_key_values: object,
    new_rows: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    if new_rows <= 0:
        raise RuntimeError("reference cache slice must contain at least one row")
    try:
        layers = list(past_key_values)
    except TypeError as error:
        raise RuntimeError("Transformers returned a non-iterable KV cache") from error
    if not layers:
        raise RuntimeError("Transformers returned an empty KV cache")
    observed = []
    for layer_index, layer in enumerate(layers):
        if not isinstance(layer, (tuple, list)) or len(layer) < 2:
            raise RuntimeError(
                f"Transformers cache layer {layer_index} has no K/V pair"
            )
        key, value = layer[:2]
        if (
            not isinstance(key, torch.Tensor)
            or not isinstance(value, torch.Tensor)
            or key.shape != value.shape
            or key.dim() != 4
            or int(key.shape[0]) != 1
            or int(key.shape[2]) < new_rows
        ):
            raise RuntimeError(
                f"Transformers cache layer {layer_index} has unsupported shape"
            )
        # Transformers uses [batch, heads, sequence, head_dim]; the Engine
        # scatter consumes [sequence, heads, head_dim].
        observed.append(
            (
                key[0, :, -new_rows:, :]
                .transpose(0, 1)
                .float()
                .cpu()
                .contiguous()
                .numpy(),
                value[0, :, -new_rows:, :]
                .transpose(0, 1)
                .float()
                .cpu()
                .contiguous()
                .numpy(),
            )
        )
    return observed


def _load_reference_model(
    family: str,
    model_path: Path,
    device: torch.device,
):
    if family in {"qwen3", "llama3"}:
        model = AutoModelForCausalLM.from_pretrained(
            model_path, dtype=torch.bfloat16
        ).to(device)
        language_model = model
    elif family == "gemma4":
        from transformers import Gemma4ForConditionalGeneration

        model = Gemma4ForConditionalGeneration.from_pretrained(
            model_path, dtype=torch.bfloat16
        ).to(device)
        language_model = model
    else:
        from transformers import Gemma3ForConditionalGeneration

        model = Gemma3ForConditionalGeneration.from_pretrained(
            model_path, dtype=torch.bfloat16
        ).to(device)
        # Text-only inference still uses the conditional-generation wrapper:
        # it skips the vision tower when no pixels are supplied and owns the
        # tied LM head that turns the nested text model's hidden states into
        # logits.
        language_model = model
    model.eval()
    return model, language_model


def _reference_sequence(
    family: str,
    model_path: Path,
    prompt_ids: list[int],
    device: torch.device,
    max_tokens: int,
    *,
    check_kv: bool,
) -> tuple[
    list[np.ndarray],
    list[int],
    list[float],
    list[list[tuple[np.ndarray, np.ndarray]]],
]:
    current_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    model, language_model = _load_reference_model(family, model_path, device)
    rows = []
    tokens = []
    margins = []
    kv_steps = []
    past_key_values = None
    total_length = len(prompt_ids)
    with torch.inference_mode():
        for _ in range(max_tokens):
            attention_mask = torch.ones(
                (1, total_length), dtype=torch.long, device=device
            )
            outputs = language_model(
                input_ids=current_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
            )
            logits = outputs.logits
            row = logits[0, -1].float()
            top_values, top_indices = row.topk(2)
            token = int(top_indices[0].item())
            rows.append(row.cpu().contiguous().numpy())
            tokens.append(token)
            margins.append(float((top_values[0] - top_values[1]).item()))
            if check_kv:
                kv_steps.append(
                    _cache_rows(outputs.past_key_values, int(current_ids.shape[1]))
                )
            past_key_values = outputs.past_key_values
            current_ids = top_indices[:1].reshape(1, 1)
            total_length += 1
            del outputs, logits, row, top_values, top_indices, attention_mask
    del language_model, model, current_ids, past_key_values
    gc.collect()
    torch.cuda.empty_cache()
    return rows, tokens, margins, kv_steps


def _load_observed_rows(directory: Path) -> list[np.ndarray]:
    records = load_observer_records(directory)
    rows = []
    for record in records:
        if record.get("source") != "vosti.sampling_boundary":
            raise RuntimeError(
                f"unexpected Engine logit source: {record.get('source')!r}"
            )
        path = directory / str(record["artifact"])
        row = np.load(path, allow_pickle=False)
        if hashlib.sha256(row.tobytes(order="C")).hexdigest() != record.get(
            "sha256"
        ):
            raise RuntimeError(
                "Engine logit artifact disagrees with its recorded digest"
            )
        if row.ndim != 1 or row.dtype != np.float32:
            raise RuntimeError(
                "Engine logit artifact is not one float32 vocabulary row"
            )
        rows.append(row)
    return rows


def _load_single_observed_row(directory: Path) -> np.ndarray:
    rows = _load_observed_rows(directory)
    if len(rows) != 1:
        raise RuntimeError(
            f"Engine produced {len(rows)} observed logit rows, expected one"
        )
    return rows[0]


def _logit_error_summary(reference: np.ndarray, actual: np.ndarray) -> dict[str, float]:
    if reference.shape != actual.shape:
        raise RuntimeError(
            f"Engine/reference logit shapes differ: {actual.shape} != {reference.shape}"
        )
    if reference.dtype != np.float32 or actual.dtype != np.float32:
        raise RuntimeError("Engine/reference logits must both be float32 observations")
    if not np.isfinite(reference).all() or not np.isfinite(actual).all():
        raise RuntimeError("Engine/reference logits must be finite")
    error = np.abs(actual - reference)
    return {
        "max_abs_error": float(error.max(initial=0.0)),
        "mean_abs_error": float(error.mean()) if error.size else 0.0,
    }


def _combine_error_summaries(
    pairs: list[tuple[np.ndarray, np.ndarray]],
    *,
    label: str,
) -> dict[str, float]:
    if not pairs:
        raise RuntimeError(f"{label} comparison has no arrays")
    maximum = 0.0
    total = 0.0
    count = 0
    for reference, actual in pairs:
        if reference.shape != actual.shape:
            raise RuntimeError(
                f"{label} shapes differ: {actual.shape} != {reference.shape}"
            )
        if reference.dtype != np.float32 or actual.dtype != np.float32:
            raise RuntimeError(f"{label} arrays must be float32 observations")
        if not np.isfinite(reference).all() or not np.isfinite(actual).all():
            raise RuntimeError(f"{label} arrays must be finite")
        error = np.abs(actual - reference)
        maximum = max(maximum, float(error.max(initial=0.0)))
        total += float(error.sum(dtype=np.float64))
        count += int(error.size)
    return {
        "max_abs_error": maximum,
        "mean_abs_error": total / count if count else 0.0,
    }


def _load_array(directory: Path, descriptor: object, *, label: str) -> np.ndarray:
    if not isinstance(descriptor, dict):
        raise RuntimeError(f"KV observer {label} descriptor is malformed")
    array = np.load(directory / str(descriptor.get("artifact")), allow_pickle=False)
    raw = array.tobytes(order="C")
    if descriptor.get("shape") != list(array.shape):
        raise RuntimeError(f"KV observer {label} shape disagrees with its record")
    if descriptor.get("dtype") != str(array.dtype):
        raise RuntimeError(f"KV observer {label} dtype disagrees with its record")
    if descriptor.get("sha256") != hashlib.sha256(raw).hexdigest():
        raise RuntimeError(f"KV observer {label} digest disagrees with its record")
    return array


def _compare_kv_steps(
    directory: Path,
    reference_steps: list[list[tuple[np.ndarray, np.ndarray]]],
) -> dict[str, float | int | bool]:
    records = load_kv_observer_records(directory)
    expected_count = sum(len(layers) for layers in reference_steps)
    if len(records) != expected_count:
        raise RuntimeError(
            f"Engine produced {len(records)} KV-store records, expected "
            f"{expected_count}"
        )
    pairs = []
    record_index = 0
    for step_index, layers in enumerate(reference_steps):
        for layer_index, (reference_k, reference_v) in enumerate(layers):
            record = records[record_index]
            record_index += 1
            if record.get("source") != (
                "vosti.store_kv_cache_from_verified_caller"
            ):
                raise RuntimeError("unexpected Engine KV observer source")
            input_k = _load_array(directory, record.get("input_k"), label="input K")
            input_v = _load_array(directory, record.get("input_v"), label="input V")
            stored_k = _load_array(
                directory, record.get("stored_k"), label="stored K"
            )
            stored_v = _load_array(
                directory, record.get("stored_v"), label="stored V"
            )
            slots = _load_array(directory, record.get("slots"), label="slots")
            if slots.ndim != 1 or slots.dtype != np.int64:
                raise RuntimeError("Engine KV observer slots are not int64 rows")
            if input_k.tobytes(order="C") != stored_k.tobytes(order="C"):
                raise AssertionError(
                    f"Engine K scatter changed values at step {step_index}, "
                    f"layer {layer_index}"
                )
            if input_v.tobytes(order="C") != stored_v.tobytes(order="C"):
                raise AssertionError(
                    f"Engine V scatter changed values at step {step_index}, "
                    f"layer {layer_index}"
                )
            pairs.extend(((reference_k, stored_k), (reference_v, stored_v)))
    errors = _combine_error_summaries(pairs, label="Engine/reference K/V")
    return {
        **errors,
        "steps": len(reference_steps),
        "layers_per_step": len(reference_steps[0]),
        "store_records": len(records),
        "compared_arrays": len(pairs),
        "stored_rows_equal_input_rows": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("family", choices=tuple(FAMILIES))
    parser.add_argument("--prompt")
    parser.add_argument("--max-tokens", type=int, default=1)
    parser.add_argument("--check-kv", action="store_true")
    parser.add_argument("--min-reference-margin", type=float, default=1.0)
    parser.add_argument("--max-logit-abs-error", type=float, default=1.0)
    parser.add_argument("--max-kv-abs-error", type=float, default=1.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.max_tokens <= 0:
        parser.error("--max-tokens must be positive")
    if args.min_reference_margin < 0.0:
        parser.error("--min-reference-margin must be nonnegative")
    if args.max_logit_abs_error < 0.0:
        parser.error("--max-logit-abs-error must be nonnegative")
    if args.max_kv_abs_error < 0.0:
        parser.error("--max-kv-abs-error must be nonnegative")
    if args.output is not None and args.output.expanduser().resolve().exists():
        parser.error("--output already exists")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    family = FAMILIES[args.family]
    prompt = args.prompt or str(family["default_prompt"])
    model_path = Path(
        os.path.expanduser(os.environ.get("MODEL_PATH", family["default_model"]))
    ).resolve()
    framework_commit = None
    checkpoint = None
    deployment = None
    if args.output is not None:
        if args.output.expanduser().resolve().is_relative_to(ROOT):
            parser.error("--output must be outside the source checkout")
        framework_commit = _clean_framework_commit()
        checkpoint = _checkpoint_manifest(model_path)
        deployment = _deployment_evidence(
            args.family, family, model_path
        )
    device = torch.device(os.environ.get("CUDA_DEVICE", "cuda:0"))
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    prompt_ids = [int(token) for token in tokenizer.encode(prompt)]
    if not prompt_ids:
        raise RuntimeError("reference prompt encoded to no tokens")
    reference_rows, expected_tokens, reference_margins, reference_kv = (
        _reference_sequence(
            args.family,
            model_path,
            prompt_ids,
            device,
            args.max_tokens,
            check_kv=args.check_kv,
        )
    )
    reference_margin = min(reference_margins)
    if reference_margin < args.min_reference_margin:
        raise RuntimeError(
            f"{args.family} reference sequence has unstable minimum top-logit margin "
            f"{reference_margin:.6g}, below {args.min_reference_margin:.6g}; "
            "choose a more discriminating prompt or explicitly lower the threshold"
        )

    with tempfile.TemporaryDirectory(prefix="vosti-reference-") as directory:
        temporary_root = Path(directory)
        output_path = temporary_root / "output_tokens.tsv"
        observer_dir = temporary_root / "logits"
        kv_observer_dir = temporary_root / "kv"
        environment = os.environ.copy()
        observer_paths = [
            ROOT / "scripts/determinism_tests/hooks",
            ROOT / "python",
            ROOT,
        ]
        if environment.get("PYTHONPATH"):
            observer_paths.extend(
                Path(path)
                for path in environment["PYTHONPATH"].split(os.pathsep)
                if path
            )
        environment.update(
            {
                "MODEL_PATH": str(model_path),
                "PYTHONPATH": os.pathsep.join(map(str, observer_paths)),
                "VOSTI_BENCH": "1",
                "VOSTI_IGNORE_EOS": "1",
                "VOSTI_KERNELS_LOGITS_OBSERVER": "1",
                "VOSTI_LOGITS_OBSERVER_DIR": str(observer_dir),
                "VOSTI_MAX_TOKENS": str(args.max_tokens),
                "VOSTI_OUTPUT_TOKENS": str(output_path),
                "VOSTI_QUIET_OUTPUT": "1",
                "VOSTI_CUDA_GRAPH": "0",
            }
        )
        if args.check_kv:
            environment.update(
                {
                    "VOSTI_KERNELS_KV_OBSERVER": "1",
                    "VOSTI_KV_OBSERVER_DIR": str(kv_observer_dir),
                }
            )
        environment["VOSTI_PROMPT_IDS"] = ",".join(map(str, prompt_ids))
        completed = subprocess.run(
            [sys.executable, str(ROOT / "scripts/launch.py"), "--kind", "engine", "--family", args.family],
            cwd=ROOT,
            env=environment,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        print(completed.stdout, end="")
        if completed.returncode != 0:
            raise RuntimeError(
                f"{args.family} Engine reference run failed with exit code "
                f"{completed.returncode}"
            )
        outputs = _read_outputs(output_path)
        if len(outputs) != 1 or len(outputs[0]) != args.max_tokens:
            raise RuntimeError(
                f"Engine output shape is {[len(output) for output in outputs]}, "
                f"expected one request with {args.max_tokens} tokens"
            )
        actual_tokens = outputs[0]
        actual_rows = _load_observed_rows(observer_dir)
        if len(actual_rows) != args.max_tokens:
            raise RuntimeError(
                f"Engine produced {len(actual_rows)} observed logit rows, "
                f"expected {args.max_tokens}"
            )
        kv_errors = (
            _compare_kv_steps(kv_observer_dir, reference_kv)
            if args.check_kv
            else None
        )
    if actual_tokens != expected_tokens:
        raise AssertionError(
            f"{args.family} generated tokens differ from Transformers: "
            f"Engine={actual_tokens}, reference={expected_tokens}"
        )
    errors = _combine_error_summaries(
        list(zip(reference_rows, actual_rows)), label="Engine/reference logits"
    )
    if errors["max_abs_error"] > args.max_logit_abs_error:
        raise AssertionError(
            f"{args.family} Engine/reference logit max absolute error "
            f"{errors['max_abs_error']:.6g} exceeds "
            f"{args.max_logit_abs_error:.6g}"
        )
    if kv_errors is not None and kv_errors["max_abs_error"] > args.max_kv_abs_error:
        raise AssertionError(
            f"{args.family} Engine/reference K/V max absolute error "
            f"{kv_errors['max_abs_error']:.6g} exceeds "
            f"{args.max_kv_abs_error:.6g}"
        )
    if args.output is not None:
        assert framework_commit is not None
        assert checkpoint is not None
        assert deployment is not None
        _checkpoint_unchanged(model_path, checkpoint)
        _write_json_report(
            args.output,
            {
                "schema": REPORT_SCHEMA,
                "status": "pass",
                "validator_framework_commit": framework_commit,
                "deployment": deployment,
                "model": {
                    "family": args.family,
                    "path": str(model_path),
                    "checkpoint": checkpoint,
                },
                "prompt": {
                    "text": prompt,
                    "token_ids": prompt_ids,
                    "token_ids_sha256": hashlib.sha256(
                        json.dumps(prompt_ids, separators=(",", ":")).encode("utf-8")
                    ).hexdigest(),
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
                    "max_kv_abs_error": args.max_kv_abs_error,
                },
                "comparison": {
                    "tokens": actual_tokens,
                    "decoded_completion": tokenizer.decode(actual_tokens),
                    "steps": args.max_tokens,
                    "all_tokens_equal": True,
                    "minimum_reference_margin": reference_margin,
                    "logits": errors,
                    "kv": kv_errors,
                },
            },
        )
    kv_summary = (
        ""
        if kv_errors is None
        else (
            f" kv_max_abs_error={kv_errors['max_abs_error']:.6g}"
            f" kv_mean_abs_error={kv_errors['mean_abs_error']:.6g}"
        )
    )
    print(
        f"{args.family} Transformers reference check passed: "
        f"tokens={actual_tokens} reference_min_margin={reference_margin:.6g} "
        f"max_logit_abs_error={errors['max_abs_error']:.6g} "
        f"mean_logit_abs_error={errors['mean_abs_error']:.6g}"
        f"{kv_summary}"
    )


if __name__ == "__main__":
    main()
