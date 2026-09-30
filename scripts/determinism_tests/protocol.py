"""Declarative protocol and artifact helpers for determinism tests."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import random
from typing import Any, Iterable

from scripts.common.model_paths import checkpoint_path

SCHEMA_VERSION = 3
ROW_COMPARISON_VERSION = 2
ROW_COMPARISON_EQUALITY = "exact_shape_dtype_and_element_bytes"
PROMPT_SEED = 0x5EED_2026
CAMPAIGN_SEEDS = (PROMPT_SEED, PROMPT_SEED + 10_000, PROMPT_SEED + 20_000)
BATCH_LENGTHS = (17, 63, 64, 65, 127, 129, 257, 385)
CHUNK_PROMPT_LENGTH = 1089
CHUNK_BUDGETS = (64, 128, 256, 512)
GENERATION_PROMPT_LENGTH = 257
GENERATION_TOKENS = 16
CACHE_PROMPT_LENGTH = 321
TESTS = ("batch_vs_single", "different_chunk", "prefill_vs_decode", "cold_vs_warm")
HARDWARE = ("h200", "a100")


def row_comparison_semantics() -> dict[str, Any]:
    """Describe the independently versioned logit-comparison contract."""

    return {
        "version": ROW_COMPARISON_VERSION,
        "equality": ROW_COMPARISON_EQUALITY,
        "mismatch_count_unit": "elements",
    }


@dataclass(frozen=True)
class ExecutionConfig:
    key: str
    engine: str
    mode: str
    attention_backend: str | None
    strict: bool
    cuda_graph: str


EXECUTION_CONFIGS = (
    ExecutionConfig("vosti-padded-graph", "vosti", "qualified", None, True, "padded-cover"),
    ExecutionConfig("vllm-fast-auto", "vllm", "fast", None, False, "enabled"),
    ExecutionConfig("vllm-invariant-flash-attn", "vllm", "invariant", "FLASH_ATTN", True, "enabled"),
    ExecutionConfig("vllm-invariant-triton-attn", "vllm", "invariant", "TRITON_ATTN", True, "enabled"),
    ExecutionConfig("sglang-fast-auto", "sglang", "fast", None, False, "enabled"),
    ExecutionConfig("sglang-deterministic-fa3", "sglang", "deterministic", "fa3", True, "enabled"),
    ExecutionConfig("sglang-deterministic-triton", "sglang", "deterministic", "triton", True, "enabled"),
)

# Retain the FA4 execution key for explicitly requested or historical runs.
ALL_EXECUTION_CONFIGS = EXECUTION_CONFIGS + (
    ExecutionConfig("sglang-deterministic-fa4", "sglang", "deterministic", "fa4", True, "enabled"),
)


@dataclass(frozen=True)
class ModelSpec:
    key: str
    architecture: str
    default_path: str
    language_model_only: bool


MODELS = (
    ModelSpec(
        "qwen3",
        "Qwen3ForCausalLM",
        checkpoint_path("Qwen3-8B"),
        False,
    ),
    ModelSpec(
        "gemma3",
        "Gemma3ForConditionalGeneration.text_model",
        checkpoint_path("gemma-3-4b-it"),
        True,
    ),
    ModelSpec(
        "llama3",
        "LlamaForCausalLM",
        checkpoint_path("Llama-3.1-8B"),
        False,
    ),
    ModelSpec(
        "gemma4",
        "Gemma4ForConditionalGeneration.text_model",
        checkpoint_path("gemma-4-31b-it"),
        True,
    ),
)


@dataclass(frozen=True)
class Checkpoint:
    key: str
    model: str
    path: str
    performance: bool = True


CHECKPOINTS = (
    Checkpoint("llama3-3b", "llama3", checkpoint_path("Llama-3.2-3B")),
    Checkpoint("gemma3-4b", "gemma3", checkpoint_path("gemma-3-4b-it")),
    Checkpoint("llama3-8b", "llama3", checkpoint_path("Llama-3.1-8B")),
    Checkpoint("gemma3-12b", "gemma3", checkpoint_path("gemma-3-12b-it")),
    Checkpoint("gemma3-27b", "gemma3", checkpoint_path("gemma-3-27b-it")),
    Checkpoint("gemma4-31b", "gemma4", checkpoint_path("gemma-4-31b-it")),
    Checkpoint("qwen3-8b", "qwen3", checkpoint_path("Qwen3-8B"), performance=False),
)

# Additional native-only qualification targets, not baseline-engine admission.
NATIVE_CHECKPOINTS = CHECKPOINTS + (
    Checkpoint("gemma4-12b", "gemma4", checkpoint_path("gemma-4-12b-it"), performance=False),
)


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def sha256_token_ids(rows: Iterable[Iterable[int]]) -> str:
    return sha256_json([[int(token) for token in row] for row in rows])


def _model_token_domain(model_path: Path) -> tuple[int, int, set[int]]:
    config = json.loads((model_path / "config.json").read_text(encoding="utf-8"))
    text = config.get("text_config") or config
    tokenizer_config = json.loads(
        (model_path / "tokenizer_config.json").read_text(encoding="utf-8")
    )
    raw_vocab_size = text.get("vocab_size") or config.get("vocab_size")
    if raw_vocab_size is None:
        tokenizer = json.loads(
            (model_path / "tokenizer.json").read_text(encoding="utf-8")
        )
        base_vocab = tokenizer.get("model", {}).get("vocab")
        if not isinstance(base_vocab, (dict, list)) or not base_vocab:
            raise ValueError("model has no usable config or tokenizer vocabulary")
        added_ids = [
            int(token) for token in tokenizer_config.get("added_tokens_decoder", {})
        ]
        vocab_size = max(len(base_vocab), max(added_ids, default=-1) + 1)
    else:
        vocab_size = int(raw_vocab_size)
    if vocab_size <= 0:
        raise ValueError("model vocabulary must be positive")
    generation = json.loads(
        (model_path / "generation_config.json").read_text(encoding="utf-8")
    )
    bos_token_id = int(generation["bos_token_id"])
    excluded = {int(token) for token in tokenizer_config.get("added_tokens_decoder", {})}
    for key in ("bos_token_id", "pad_token_id"):
        value = generation.get(key)
        if isinstance(value, int):
            excluded.add(value)
    eos = generation.get("eos_token_id", [])
    excluded.update([eos] if isinstance(eos, int) else map(int, eos))
    return vocab_size, bos_token_id, excluded


def deterministic_prompt(
    model_path: Path,
    *,
    length: int,
    seed: int,
) -> list[int]:
    if length <= 0:
        raise ValueError("prompt length must be positive")
    vocab_size, bos_token_id, excluded = _model_token_domain(model_path)
    rng = random.Random(seed)
    prompt = [bos_token_id]
    while len(prompt) < length:
        token = rng.randrange(vocab_size)
        if token not in excluded:
            prompt.append(token)
    return prompt


def build_inputs(model: ModelSpec, model_path: Path, *, seed: int = PROMPT_SEED) -> dict[str, Any]:
    batch = [
        deterministic_prompt(model_path, length=length, seed=seed + index)
        for index, length in enumerate(BATCH_LENGTHS)
    ]
    chunk = deterministic_prompt(
        model_path, length=CHUNK_PROMPT_LENGTH, seed=seed + 100
    )
    generation = deterministic_prompt(
        model_path, length=GENERATION_PROMPT_LENGTH, seed=seed + 200
    )
    cache = deterministic_prompt(
        model_path, length=CACHE_PROMPT_LENGTH, seed=seed + 300
    )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "model": asdict(model),
        "model_path": str(model_path.resolve()),
        "seed": seed,
        "batch_lengths": list(BATCH_LENGTHS),
        "batch_prompts": batch,
        "chunk_prompt": chunk,
        "chunk_budgets": list(CHUNK_BUDGETS),
        "generation_prompt": generation,
        "generation_tokens": GENERATION_TOKENS,
        "cache_prompt": cache,
    }
    payload["token_ids_sha256"] = sha256_json(
        {
            "batch": batch,
            "chunk": chunk,
            "generation": generation,
            "cache": cache,
        }
    )
    return payload


def logical_matrix(*, hardware: Iterable[str] = ("h200",),
                   seeds: Iterable[int] = CAMPAIGN_SEEDS) -> list[dict[str, Any]]:
    selected_hardware = tuple(hardware)
    selected_seeds = tuple(seeds)
    unknown_hardware = set(selected_hardware) - set(HARDWARE)
    if unknown_hardware:
        raise ValueError(
            f"unknown determinism-test hardware: {sorted(unknown_hardware)}"
        )
    return [
        {
            "hardware": hardware,
            "model": checkpoint.model,
            "checkpoint": checkpoint.key,
            "model_path": checkpoint.path,
            "seed": seed,
            "execution_config": config.key,
            "test": test,
            "strict": config.strict,
        }
        for hardware in selected_hardware
        for seed in selected_seeds
        for checkpoint in CHECKPOINTS
        for config in EXECUTION_CONFIGS
        for test in TESTS
    ]


def load_observer_records(directory: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted(directory.glob("observer-*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                record = json.loads(line)
                record["index_file"] = path.name
                records.append(record)
    return sorted(
        records,
        key=lambda row: (row["call_sequence"], row["batch_index"], row["pid"]),
    )


def compare_row_artifacts(left: Path, right: Path) -> dict[str, Any]:
    import numpy as np

    lhs = np.load(left, allow_pickle=False)
    rhs = np.load(right, allow_pickle=False)
    shape_equal = lhs.shape == rhs.shape
    dtype_equal = lhs.dtype == rhs.dtype
    result: dict[str, Any] = {
        "left": str(left),
        "right": str(right),
        "comparison_semantics": row_comparison_semantics(),
        "shape_equal": shape_equal,
        "dtype_equal": dtype_equal,
        "finite_left": bool(np.isfinite(lhs).all()),
        "finite_right": bool(np.isfinite(rhs).all()),
    }
    if not shape_equal or not dtype_equal:
        result.update({"bitwise_equal": False, "mismatch_count": None})
        return result
    lhs_bytes = (
        np.ascontiguousarray(lhs).view(np.uint8).reshape(lhs.size, lhs.itemsize)
    )
    rhs_bytes = (
        np.ascontiguousarray(rhs).view(np.uint8).reshape(rhs.size, rhs.itemsize)
    )
    mismatch = np.flatnonzero(np.any(lhs_bytes != rhs_bytes, axis=1))
    result.update(
        {
            "bitwise_equal": not bool(mismatch.size),
            "mismatch_count": int(mismatch.size),
            "first_mismatch_index": int(mismatch[0]) if mismatch.size else None,
            "max_abs_diff": float(np.max(np.abs(lhs - rhs))) if mismatch.size else 0.0,
            "argmax_left": int(np.argmax(lhs)),
            "argmax_right": int(np.argmax(rhs)),
            "argmax_equal": int(np.argmax(lhs)) == int(np.argmax(rhs)),
        }
    )
    return result
