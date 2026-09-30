import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from scripts.checks.reference_suite import (
    _collect_contexts,
    _expected_arm_names,
    _prompt_for_request,
    _reference_model_path,
    _validated_suite,
)
from scripts.determinism_tests.protocol import (
    SCHEMA_VERSION,
    TESTS,
    row_comparison_semantics,
    sha256_json,
)


class ModelReferenceSuiteTests(unittest.TestCase):
    def _inputs(self, root: Path) -> dict:
        inputs = {
            "schema_version": SCHEMA_VERSION,
            "model": {
                "key": "synthetic",
                "architecture": "SyntheticForCausalLM",
                "default_path": str(root / "model"),
                "language_model_only": False,
            },
            "model_path": str(root / "model"),
            "batch_prompts": [[7, 8], [7, 9, 10]],
            "chunk_prompt": [7, 11, 12],
            "chunk_budgets": [4, 8],
            "generation_prompt": [7, 13],
            "generation_tokens": 3,
            "cache_prompt": [7, 14, 15],
        }
        inputs["token_ids_sha256"] = sha256_json(
            {
                "batch": inputs["batch_prompts"],
                "chunk": inputs["chunk_prompt"],
                "generation": inputs["generation_prompt"],
                "cache": inputs["cache_prompt"],
            }
        )
        return inputs

    def _summary(self, root: Path, inputs: dict) -> dict:
        arms = []
        for name in sorted(_expected_arm_names(inputs)):
            arm_dir = root / "arms" / name
            arm_dir.mkdir(parents=True)
            telemetry_path = arm_dir / "telemetry.json"
            telemetry_path.write_text('{"status":"complete"}\n', encoding="utf-8")
            telemetry_sha256 = hashlib.sha256(telemetry_path.read_bytes()).hexdigest()
            arms.append(
                {
                    "name": name,
                    "arm_sha256": "a" * 64,
                    "result": str(arm_dir / "result.json"),
                    "telemetry": {
                        "status": "complete",
                        "path": str(telemetry_path),
                        "sha256": telemetry_sha256,
                    },
                }
            )
        return {
            "schema_version": SCHEMA_VERSION,
            "acceptance": "strict",
            "status": "pass",
            "all_bitwise_equal": True,
            "relation_pass": {name: True for name in TESTS},
            "comparison_semantics": row_comparison_semantics(),
            "inputs_sha256": sha256_json(inputs),
            "model": inputs["model"],
            "arms": arms,
        }

    @staticmethod
    def _request(prompt: list[int], tokens: list[int], position: int) -> dict:
        return {
            "prompt_length": len(prompt),
            "prompt_sha256": sha256_json(prompt),
            "output_token_ids": tokens,
            "num_cached_tokens": 0,
            "_position": position,
        }

    @staticmethod
    def _write_row(arm_dir: Path, request: dict, row: np.ndarray) -> None:
        rows = arm_dir / "rows"
        rows.mkdir(exist_ok=True)
        digest = hashlib.sha256(row.tobytes(order="C")).hexdigest()
        artifact = f"row-{digest}.npy"
        np.save(rows / artifact, row, allow_pickle=False)
        request["last_row"] = {
            "artifact": artifact,
            "shape": list(row.shape),
            "dtype": str(row.dtype),
            "finite": True,
            "sha256": digest,
            "argmax": int(np.argmax(row)),
            "metadata": {"generated_position": request.pop("_position")},
        }

    def _write_results(self, root: Path, summary: dict, inputs: dict) -> None:
        generated = [1, 0, 1]
        batch_rows = [
            np.array([2.0, 1.0], dtype=np.float32),
            np.array([1.0, 2.0], dtype=np.float32),
        ]
        chunk_row = np.array([3.0, 1.0], dtype=np.float32)
        generation_row = np.array([1.0, 4.0], dtype=np.float32)
        cache_row = np.array([5.0, 1.0], dtype=np.float32)
        results = {}

        batch_requests = []
        for prompt, row in zip(inputs["batch_prompts"], batch_rows, strict=True):
            request = self._request(prompt, [int(np.argmax(row))], 0)
            batch_requests.append((request, row))
        results["batch-vs-single-batch"] = [batch_requests]
        for index, (prompt, row) in enumerate(
            zip(inputs["batch_prompts"], batch_rows, strict=True)
        ):
            request = self._request(prompt, [int(np.argmax(row))], 0)
            results[f"batch-vs-single-single-{index:02d}"] = [[(request, row)]]
        for budget in inputs["chunk_budgets"]:
            request = self._request(inputs["chunk_prompt"], [0], 0)
            results[f"different-chunk-{budget}"] = [[(request, chunk_row)]]
        request = self._request(inputs["generation_prompt"], generated, 2)
        results["prefill-vs-decode-generate"] = [[(request, generation_row)]]
        teacher_prompt = inputs["generation_prompt"] + generated[:-1]
        request = self._request(teacher_prompt, [1], 0)
        results["prefill-vs-decode-teacher"] = [[(request, generation_row)]]
        cold = self._request(inputs["cache_prompt"], [0, 1], 0)
        warm = self._request(inputs["cache_prompt"], [0, 1], 0)
        warm["num_cached_tokens"] = 2
        results["cold-vs-warm"] = [[(cold, cache_row)], [(warm, cache_row)]]

        for arm in summary["arms"]:
            name = arm["name"]
            arm_dir = Path(arm["result"]).parent
            retained_arm = {"name": name}
            arm_sha256 = sha256_json(retained_arm)
            arm["arm_sha256"] = arm_sha256
            (arm_dir / "arm.json").write_text(
                json.dumps(retained_arm), encoding="utf-8"
            )
            calls = []
            for requests in results[name]:
                serialized = []
                for request, row in requests:
                    self._write_row(arm_dir, request, row)
                    serialized.append(request)
                calls.append({"requests": serialized})
            result = {
                "schema_version": SCHEMA_VERSION,
                "engine": "vosti",
                "arm_sha256": arm_sha256,
                "model_path": inputs["model_path"],
                "backend_evidence": {
                    "qualified": True,
                    "framework_commit": "b" * 40,
                    "deployment_sha256": "c" * 64,
                },
                "calls": calls,
            }
            Path(arm["result"]).write_text(json.dumps(result), encoding="utf-8")

    def test_suite_validation_requires_exact_coverage_and_input_binding(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = self._inputs(root)
            summary = self._summary(root, inputs)
            (root / "inputs.json").write_text(json.dumps(inputs), encoding="utf-8")
            (root / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
            _, loaded_inputs, loaded_root = _validated_suite(root / "summary.json")
            self.assertEqual(loaded_inputs, inputs)
            self.assertEqual(loaded_root, root.resolve())

            summary["arms"].pop()
            (root / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "exact required arms"):
                _validated_suite(root / "summary.json")

    def test_prompt_reconstruction_checks_digest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            inputs = self._inputs(Path(directory))
            tokens = [1, 0, 1]
            prompt = inputs["generation_prompt"] + tokens[:-1]
            request = self._request(prompt, [1], 0)
            self.assertEqual(
                _prompt_for_request(
                    "prefill-vs-decode-teacher", 0, request, inputs, tokens
                ),
                prompt,
            )
            request["prompt_sha256"] = "0" * 64
            with self.assertRaisesRegex(RuntimeError, "digest"):
                _prompt_for_request(
                    "prefill-vs-decode-teacher", 0, request, inputs, tokens
                )

    def test_reference_checkpoint_uses_explicit_suite_path_not_family_default(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = self._inputs(root)
            variant = root / "model-variant"
            inputs["model_path"] = str(variant)
            self.assertEqual(_reference_model_path(None, inputs), variant.resolve())
            self.assertEqual(
                _reference_model_path(variant, inputs),
                variant.resolve(),
            )
            with self.assertRaisesRegex(RuntimeError, "differs from strict suite"):
                _reference_model_path(root / "model", inputs)

    def test_context_collection_rechecks_every_bitwise_relation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = self._inputs(root)
            summary = self._summary(root, inputs)
            self._write_results(root, summary, inputs)
            contexts, framework, deployment, provenance = _collect_contexts(
                summary, inputs
            )
            self.assertEqual(len(contexts), 5)
            self.assertEqual(sum(map(len, contexts.values())), 10)
            self.assertEqual(framework, "b" * 40)
            self.assertEqual(deployment, "c" * 64)
            self.assertEqual(len(provenance), 8)

            single = next(
                arm
                for arm in summary["arms"]
                if arm["name"] == "batch-vs-single-single-00"
            )
            result_path = Path(single["result"])
            result = json.loads(result_path.read_text(encoding="utf-8"))
            request = result["calls"][0]["requests"][0]
            changed = np.array([2.0, -0.0], dtype=np.float32)
            np.save(
                result_path.parent / "rows" / request["last_row"]["artifact"],
                changed,
                allow_pickle=False,
            )
            request["last_row"]["sha256"] = hashlib.sha256(
                changed.tobytes(order="C")
            ).hexdigest()
            result_path.write_text(json.dumps(result), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "not bitwise equal"):
                _collect_contexts(summary, inputs)


if __name__ == "__main__":
    unittest.main()
