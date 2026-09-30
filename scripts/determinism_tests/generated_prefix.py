"""Targeted full-logit regression for reuse of previously generated KV pages."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.determinism_tests.protocol import (
    SCHEMA_VERSION, compare_row_artifacts, deterministic_prompt,
    row_comparison_semantics, sha256_json,
)
from scripts.determinism_tests.run import _arm, _row_path
from scripts.determinism_tests.vosti_worker import PAGE_SIZE


def expected_cached_tokens(prompt_length: int, output_length: int) -> int:
    if prompt_length <= 0 or output_length <= 0:
        raise ValueError("prompt and output must be nonempty")
    # The last sampled token has not been forwarded. Metadata fullness alone
    # must not count it as materialized KV, even at a page boundary.
    return ((prompt_length + output_length - 1) // PAGE_SIZE) * PAGE_SIZE


def check_witness(*, baseline_tokens, repeated_tokens, prompt_length,
                  cold_cached, warm_cached, warmup_graph):
    expected = expected_cached_tokens(prompt_length, len(baseline_tokens))
    return {
        "donor_generation_repeated_exactly": baseline_tokens == repeated_tokens,
        "cold_cache_empty": cold_cached == 0,
        "exact_materialized_page_count": warm_cached == expected,
        "generated_pages_reused": isinstance(warm_cached, int)
            and warm_cached > (prompt_length // PAGE_SIZE) * PAGE_SIZE,
        "donor_padded_cover_replayed": isinstance(warmup_graph, dict)
            and warmup_graph.get("cover_replay_count", 0) > 0,
    }


def run_check(runner, inputs, execution, *, hardware: str):
    if execution.key != "vosti-padded-graph":
        raise ValueError("generated-prefix qualification requires Vosti padded graphs")
    model_path = Path(inputs["model_path"])
    base_seed = int(inputs["seed"])
    prompt = deterministic_prompt(model_path, length=1024, seed=base_seed + 400)
    suffix = deterministic_prompt(model_path, length=66, seed=base_seed + 401)[1:]
    # Capture a wider two-sequence decode graph before the singleton donor.
    # This forces genuine padded-cover replay rather than just exact replay.
    primer = {"prompts": [
        deterministic_prompt(model_path, length=1280, seed=base_seed + seed)
        for seed in (402, 403)
    ], "max_tokens": 4}
    donor = {"label": "donor", "prompts": [prompt], "max_tokens": 128,
             "record_last_rows": True, "require_graph_replay": True}

    def arm(calls):
        result = _arm(inputs=inputs, execution=execution, calls=calls, prefix_caching=True)
        result["engine"]["max_model_len"] = 2048
        result["engine"]["graph_primer"] = primer
        return result

    generated_dir, generated = runner.run_arm("generated-prefix-donor", arm([donor]))
    tokens = generated["calls"][0]["requests"][0]["output_token_ids"]
    if len(tokens) != donor["max_tokens"]:
        raise RuntimeError("donor output budget was not satisfied")
    followup = prompt + tokens + suffix
    manifest = {"graph_primer": primer, "prompt": prompt, "donor_output": tokens, "suffix": suffix,
                "followup": followup, "expected_cached_tokens": expected_cached_tokens(1024, 128)}
    manifest_path = runner.output / "generated-prefix-inputs.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != manifest:
            raise RuntimeError("resumed generated-prefix inputs differ")
    else:
        with manifest_path.open("x") as stream:
            json.dump(manifest, stream, indent=2, sort_keys=True)
            stream.write("\n")
    measured = {"label": "followup", "prompts": [followup], "max_tokens": 2,
                "record_last_rows": True, "record_generated_position": 0,
                "require_graph_replay": True}
    cold_dir, cold = runner.run_arm("generated-prefix-cold", arm([measured]))
    warm_dir, warm = runner.run_arm("generated-prefix-reuse", arm([donor, measured]))
    cold_request = cold["calls"][0]["requests"][0]
    warm_request = warm["calls"][1]["requests"][0]
    repeated = warm["calls"][0]["requests"][0]["output_token_ids"]
    warmup_graph = warm["backend_evidence"]["invocations"][0]["graph_warmup_stats"]
    witness = check_witness(
        baseline_tokens=tokens, repeated_tokens=repeated, prompt_length=len(prompt),
        cold_cached=cold_request["num_cached_tokens"],
        warm_cached=warm_request["num_cached_tokens"], warmup_graph=warmup_graph,
    )
    comparisons = {
        "donor_repeat": compare_row_artifacts(_row_path(generated_dir, generated, 0, 0),
                                              _row_path(warm_dir, warm, 0, 0)),
        "followup_cold_vs_generated_prefix": compare_row_artifacts(
            _row_path(cold_dir, cold, 0, 0), _row_path(warm_dir, warm, 1, 0)),
    }
    valid = all(witness.values())
    equal = all(row["bitwise_equal"] and row["finite_left"] and row["finite_right"]
                for row in comparisons.values())
    summary = {
        "schema_version": SCHEMA_VERSION, "hardware": hardware,
        "model": inputs["model"], "model_path": inputs["model_path"],
        "relation": "generated_prefix_reuse", "inputs_sha256": sha256_json(manifest),
        "comparison_semantics": row_comparison_semantics(), "arms": runner.arm_records,
        "witness": witness, "comparisons": comparisons,
        "expected_cached_tokens": manifest["expected_cached_tokens"],
        "observed_cached_tokens": warm_request["num_cached_tokens"],
        "status": "invalid" if not valid else "pass" if equal else "fail",
    }
    with (runner.output / "summary.json").open("x") as stream:
        json.dump(summary, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    if summary["status"] != "pass":
        raise SystemExit(1)
