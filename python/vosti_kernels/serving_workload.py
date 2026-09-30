"""Tokenizer-owned example and benchmark workload preparation."""

from __future__ import annotations

from collections.abc import Mapping
import json
import os
from pathlib import Path

from transformers import AutoTokenizer
from kernels.triton_kernels.constants import PAGE_SIZE

from . import kernels

# Fixed implementation bound, not a tunable limit. The Rust counterpart is in
# src/exec/request_state.rs. Changing both values is insufficient: EosTokenSet's
# three-field representation, lifecycle logic, and proofs must be reviewed and
# updated, then reverified and retested.
MAX_EOS_TOKEN_IDS = 3


def load_serving_tokenizer(model_path: str | Path):
    """Load the tokenizer owned by one long-lived serving process."""

    return AutoTokenizer.from_pretrained(Path(model_path).expanduser().resolve())


def encode_serving_prompt(tokenizer, prompt: str) -> list[int]:
    """Tokenize one raw OpenAI-completions prompt."""

    if not isinstance(prompt, str) or not prompt:
        raise ValueError("serving prompt must be a nonempty string")
    return [int(token) for token in tokenizer.encode(prompt)]


def encode_serving_chat(tokenizer, messages_json: str) -> list[int]:
    """Apply the checkpoint chat template to one OpenAI message sequence."""

    messages = json.loads(messages_json)
    if (
        not isinstance(messages, list)
        or not messages
        or any(
            not isinstance(message, dict)
            or not isinstance(message.get("role"), str)
            or not isinstance(message.get("content"), str)
            for message in messages
        )
    ):
        raise ValueError("chat messages must contain string role/content fields")
    encoded = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
    )
    if isinstance(encoded, Mapping):
        encoded = encoded.get("input_ids")
    if not isinstance(encoded, list) or any(
        type(token) is not int for token in encoded
    ):
        raise ValueError("chat template did not return a flat integer token list")
    return [int(token) for token in encoded]


def decode_serving_tokens(tokenizer, token_ids: list[int]) -> str:
    """Decode generated token IDs for an OpenAI response."""

    if not isinstance(token_ids, list) or any(
        type(token) is not int for token in token_ids
    ):
        raise ValueError("generated token IDs must be an integer list")
    return tokenizer.decode(
        token_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )


def normalize_eos_token_ids(value: object) -> list[int]:
    """Normalize Hugging Face's scalar-or-list EOS metadata fail-closed."""

    values = [value] if type(value) is int else value
    if (
        not isinstance(values, list)
        or not values
        or any(type(token) is not int or token < 0 for token in values)
    ):
        raise ValueError(
            "generation config requires a nonempty nonnegative eos_token_id list"
        )
    if len(values) != len(set(values)):
        raise ValueError("generation config has duplicate eos_token_id entries")
    if len(values) > MAX_EOS_TOKEN_IDS:
        raise ValueError(
            f"generation config has more than {MAX_EOS_TOKEN_IDS} EOS token ids"
        )
    return list(values)


def load_generation_eos_token_ids(model_path: str | Path) -> list[int]:
    """Load alternative single-token terminators from one checkpoint."""

    path = Path(model_path).expanduser().resolve() / "generation_config.json"
    if not path.is_file():
        raise ValueError(f"checkpoint has no generation_config.json: {path.parent}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("generation_config.json must contain an object")
    return normalize_eos_token_ids(payload.get("eos_token_id"))


def make_graph_warmup_prompt_ids(
    prompt_ids: list[list[int]],
    *,
    vocab_size: int,
    rounds: int,
    block_size: int = PAGE_SIZE,
) -> list[list[list[int]]]:
    """Build shape-identical, prefix-disjoint graph-capture workloads."""

    if vocab_size <= 1:
        raise ValueError("graph warmup requires a vocabulary with at least two tokens")
    if rounds < 0:
        raise ValueError("graph warmup rounds must be nonnegative")
    if block_size <= 0:
        raise ValueError("graph warmup block size must be positive")

    def first_pages(sequences: list[list[int]]) -> set[tuple[int, ...]]:
        return {
            tuple(sequence[:block_size])
            for sequence in sequences
            if len(sequence) >= block_size
        }

    occupied = first_pages(prompt_ids)
    warmup_rounds: list[list[list[int]]] = []
    for _ in range(rounds):
        for delta in range(1, vocab_size):
            candidate_pages = {
                tuple((token + delta) % vocab_size for token in sequence[:block_size])
                for sequence in prompt_ids
                if len(sequence) >= block_size
            }
            if candidate_pages.isdisjoint(occupied):
                break
        else:
            raise RuntimeError(
                "could not construct prefix-disjoint CUDA-graph warmup prompts"
            )
        transformed = [
            [(token + delta) % vocab_size for token in sequence]
            for sequence in prompt_ids
        ]
        occupied.update(candidate_pages)
        warmup_rounds.append(transformed)
    return warmup_rounds


def resolve_prompt_ids(entries: list[dict], prompts: list[str], tokenizer):
    """Use attested workload IDs when present and validate them fail-closed."""

    provided = ["prompt_token_ids" in entry for entry in entries]
    if any(provided) and not all(provided):
        raise ValueError(
            "benchmark workload must provide prompt_token_ids for every request or none"
        )
    encoded_prompts = [
        [int(token) for token in tokenizer.encode(prompt)] for prompt in prompts
    ]
    if not all(provided):
        return encoded_prompts, "tokenizer"

    prompt_ids = []
    for index, (entry, encoded) in enumerate(zip(entries, encoded_prompts)):
        raw_ids = entry["prompt_token_ids"]
        if not isinstance(raw_ids, list) or any(
            type(token) is not int or token < 0 for token in raw_ids
        ):
            raise ValueError(f"request {index} has malformed prompt_token_ids")
        if raw_ids != encoded:
            raise ValueError(
                f"request {index} prompt_token_ids disagree with the tokenizer"
            )
        prompt_ids.append(list(raw_ids))
    return prompt_ids, "workload"


def load_tokenizer_workload(
    model_path: str | Path,
    vocab_size: int,
    default_prompts: list[str],
) -> dict:
    """Prepare tokenizer policy and requests without loading model weights."""

    model_path = Path(model_path).expanduser().resolve()
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    prompts_file = os.environ.get("VOSTI_PROMPTS_FILE")
    max_tokens_list = []
    if prompts_file:
        with open(prompts_file) as workload_file:
            payload = json.load(workload_file)
        entries = payload if isinstance(payload, list) else payload["requests"]
        base_prompts = [entry["prompt"] for entry in entries]
        if entries and "max_tokens" in entries[0]:
            max_tokens_list = [int(entry["max_tokens"]) for entry in entries]
        arrival_phases = [int(entry.get("arrival_phase", 0)) for entry in entries]
    else:
        prompts_env = os.environ.get("VOSTI_PROMPTS")
        base_prompts = (
            [prompt for prompt in prompts_env.split("||") if prompt]
            if prompts_env
            else list(default_prompts)
        )
        arrival_phases = [0] * len(base_prompts)
    if os.environ.get("VOSTI_RAW_PROMPTS"):
        prompts = list(base_prompts)
    else:
        prompts = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for prompt in base_prompts
        ]
    if prompts_file:
        prompt_ids, prompt_ids_source = resolve_prompt_ids(entries, prompts, tokenizer)
    else:
        prompt_ids = [tokenizer.encode(prompt) for prompt in prompts]
        prompt_ids_source = "tokenizer"
    try:
        graph_warmup_rounds = int(os.environ.get("VOSTI_GRAPH_WARMUP_ROUNDS", "0"))
    except ValueError as error:
        raise ValueError("VOSTI_GRAPH_WARMUP_ROUNDS must be an integer") from error
    return {
        "tokenizer": tokenizer,
        "prompts": prompts,
        "prompt_ids": prompt_ids,
        "prompt_ids_source": prompt_ids_source,
        "graph_warmup_prompt_ids": make_graph_warmup_prompt_ids(
            prompt_ids,
            vocab_size=vocab_size,
            rounds=graph_warmup_rounds,
            block_size=kernels._BLOCK_SIZE,
        ),
        "max_tokens_list": max_tokens_list,
        "arrival_phases": arrival_phases,
        "eos_token_ids": load_generation_eos_token_ids(model_path),
    }
