import copy
import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import httpx

from scripts.determinism_tests.protocol import CHECKPOINTS
from scripts.serving_benchmark.phases import cache_evidence, execute_phase, prepare, run, validate
from scripts.serving_benchmark.prepare import phase_cells


class CharacterTokenizer:
    def encode(self, text, add_special_tokens=True):
        return ([0] if add_special_tokens else []) + list(text.encode())

    def decode(self, ids, skip_special_tokens=False):
        return bytes(i for i in ids if i).decode()


def workload(kind="cached_extension"):
    return prepare(CharacterTokenizer(), kind=kind, context=0 if kind == "cold_prefill" else 40,
                   query=0 if kind == "decode" else 20, output=4 if kind == "decode" else 1,
                   concurrency=2, waves=2, seed=1)


class PhaseProtocolTests(unittest.TestCase):
    def test_full_phase_matrix_has_one_manifest_per_engine_independent_cell(self):
        cells = phase_cells()
        checkpoints = {model.key for model in CHECKPOINTS if model.performance}
        self.assertEqual(len(cells), len(checkpoints) * 20 * 3)
        self.assertEqual({row['checkpoint'] for row in cells}, checkpoints)
        self.assertEqual({row['concurrency'] for row in cells}, {1, 4})

    def test_exact_geometry_and_disjoint_warmup(self):
        for kind in ("cold_prefill", "decode", "cached_extension"):
            data = workload(kind)
            validate(data, CharacterTokenizer(), 64)
            self.assertEqual(len(data["warmup"]), 2)
            self.assertEqual(len(data["measured"]), 4)
            for row in data["measured"]:
                self.assertEqual(len(CharacterTokenizer().encode(row["prompt"])), row["prompt_tokens"])
                if kind == "cached_extension":
                    self.assertEqual(row["prompt"][:len(row["donor"])], row["donor"])

    def test_changed_prefix_counts_and_overflow_rejected(self):
        for field, value in (("donor", "z" * 39), ("prompt_tokens", 61), ("max_tokens", 2)):
            data = workload()
            data["measured"][0][field] = value
            with self.assertRaises(ValueError):
                validate(data, CharacterTokenizer(), 64)
        with self.assertRaisesRegex(ValueError, "context limit"):
            validate(workload(), CharacterTokenizer(), 60)

    def test_shared_warmup_prefix_rejected(self):
        data = workload()
        data["measured"][0] = copy.deepcopy(data["warmup"][0])
        with self.assertRaisesRegex(ValueError, "share an initial"):
            validate(data, CharacterTokenizer(), 64)

    def test_cache_unknown_cold_partial_and_unexpected_distinct(self):
        record = dict(success=True, prompt_tokens=60, server_cached_prompt_tokens=None)
        self.assertEqual(cache_evidence(record, "cached_extension", 40)["status"], "unknown")
        for cached, status in ((0, "missing_hit"), (32, "observed_hit"), (41, "unexpected_extra_reuse")):
            record["server_cached_prompt_tokens"] = cached
            evidence = cache_evidence(record, "cached_extension", 40)
            self.assertEqual(evidence["status"], status)
            self.assertEqual(evidence["realized_uncached_prompt_tokens"], 60 - cached)
        self.assertEqual(cache_evidence(record, "cold_prefill", 0)["status"], "unexpected_reuse")


class FakeServer:
    def __init__(self, *, fail_donor=False, cache=True):
        self.fail_donor, self.cache = fail_donor, cache
        self.calls = []

    async def __call__(self, request):
        if request.url.path == "/health":
            return httpx.Response(200)
        if request.url.path == "/metrics":
            return httpx.Response(200, json={"calls": len(self.calls)})
        body = json.loads(request.content)
        self.calls.append(body)
        prompt_tokens = len(body["prompt"]) + 1
        if self.fail_donor and prompt_tokens == 40:
            return httpx.Response(500, text="injected donor failure")
        usage = dict(prompt_tokens=prompt_tokens, completion_tokens=body["max_tokens"])
        if self.cache:
            usage["prompt_tokens_details"] = {"cached_tokens": 32 if prompt_tokens == 60 else 0}
        event = dict(choices=[dict(text="x" * body["max_tokens"], finish_reason="length")], usage=usage)
        return httpx.Response(200, text=f"data: {json.dumps(event)}\n\ndata: [DONE]\n\n")


class PhaseExecutionTests(unittest.IsolatedAsyncioTestCase):
    async def test_donors_precede_waves_and_are_excluded_from_summary(self):
        fake = FakeServer()
        async with httpx.AsyncClient(transport=httpx.MockTransport(fake)) as client:
            result = await execute_phase(client, data=workload(), name="measured",
                                         base_url="http://test", model="fake")
        self.assertTrue(result["complete"])
        self.assertTrue(result["cache_geometry_observed"])
        self.assertEqual(result["summary"]["requests"], 4)
        self.assertEqual(len(result["donors"]), 2)
        self.assertEqual([len(row["prompt"]) + 1 for row in fake.calls], [40, 40, 60, 60] * 2)
        self.assertEqual(result["summary"]["duration_s"], sum(w["duration_s"] for w in result["waves"]))
        self.assertEqual(result["waves"][0]["server_metrics_before"], {"calls": 2})
        self.assertEqual(result["waves"][0]["server_metrics_after"], {"calls": 4})
        self.assertTrue(all(row["ignore_eos"] for row in fake.calls))

    async def test_failed_donor_retains_evidence_and_skips_measurement(self):
        fake = FakeServer(fail_donor=True)
        async with httpx.AsyncClient(transport=httpx.MockTransport(fake)) as client:
            result = await execute_phase(client, data=workload(), name="measured",
                                         base_url="http://test", model="fake")
        self.assertFalse(result["complete"])
        self.assertEqual(result["unattempted_requests"], 4)
        self.assertEqual(len(result["donors"][0]["requests"]), 2)
        self.assertIsNone(result["summary"]["output_token_throughput_per_s"])
        json.dumps(result, allow_nan=False)

    async def test_unknown_cache_preserves_timings_without_claiming_cache_geometry(self):
        fake = FakeServer(cache=False)
        async with httpx.AsyncClient(transport=httpx.MockTransport(fake)) as client:
            result = await execute_phase(client, data=workload(), name="measured",
                                         base_url="http://test", model="fake")
        self.assertTrue(result["complete"])
        self.assertFalse(result["cache_geometry_observed"])
        self.assertEqual(result["summary"]["cache_unknown_requests"], 4)

    async def test_warmup_failure_skips_measured_phase(self):
        fake = FakeServer(fail_donor=True)
        client = httpx.AsyncClient(transport=httpx.MockTransport(fake))
        args = SimpleNamespace(api_key=None, timeout=30, base_url="http://test", model="fake",
                               engine_label="fake", context_limit=64)
        with patch("scripts.serving_benchmark.phases.httpx.AsyncClient", return_value=client):
            result = await run(args, workload())
        self.assertFalse(result["complete"])
        self.assertIsNone(result["measured"])
        self.assertFalse(result["warmup"]["complete"])


if __name__ == "__main__":
    unittest.main()
