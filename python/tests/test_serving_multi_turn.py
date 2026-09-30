import asyncio
import copy
import json
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import httpx

from scripts.serving_benchmark.multi_turn import (
    compare_results, digest, exact_text, execute_sessions, offered_trace, prepare, run,
    prepare_arrival_trace, resolve_arrivals, validate_arrival_trace,
    validate_warmup, validate_workload, write_new,
)
from scripts.serving_benchmark.protocol import parse_stream_payload
from scripts.serving_benchmark.run import send_request


class CharacterTokenizer:
    def encode(self, text, add_special_tokens=True):
        return ([0] if add_special_tokens else []) + [ord(c) for c in text]

    def decode(self, ids, skip_special_tokens=False):
        return "".join(chr(i) for i in ids if i != 0)


def workload(sessions=2, turns=3):
    return prepare(CharacterTokenizer(), sessions=sessions, turns=turns,
                   initial_tokens=40, suffix_tokens=5, output_tokens=2,
                   think_seconds=0, seed=1)


class ExactSuffixBudgetTests(unittest.TestCase):
    def test_offered_trace_is_fixed_in_turn_major_order(self):
        data = workload(sessions=2, turns=3)
        trace = offered_trace(data, request_rate=2, process="constant", seed=42,
                              concurrency=2, stagger_seconds=0)
        self.assertEqual([(r["session_id"], r["turn"]) for r in trace],
                         [("0", 0), ("1", 0), ("0", 1), ("1", 1), ("0", 2), ("1", 2)])
        self.assertEqual([r["offset_s"] for r in trace], [0, .5, 1, 1.5, 2, 2.5])
        for kwargs in (dict(concurrency=1, stagger_seconds=0), dict(concurrency=2, stagger_seconds=.1)):
            with self.assertRaisesRegex(ValueError, "lane per session"):
                offered_trace(data, request_rate=2, process="constant", seed=42, **kwargs)
        with self.assertRaisesRegex(ValueError, "positive and finite"):
            offered_trace(data, request_rate=float("nan"), process="constant", seed=42,
                          concurrency=2, stagger_seconds=0)

    def test_explicit_budget_includes_separator(self):
        data = prepare(CharacterTokenizer(), sessions=2, turns=3,
            initial_tokens=40, suffix_tokens=5, output_tokens=2,
            think_seconds=0, seed=1, suffix_includes_separator=True)
        for session in data["sessions"]:
            for turn in session["turns"][1:]:
                self.assertTrue(turn["suffix"].startswith("\n\n"))
                self.assertEqual(turn["suffix_tokens"], 5)

    def test_default_preserves_previous_separator_accounting(self):
        for session in workload()["sessions"]:
            self.assertEqual(session["turns"][1]["suffix_tokens"], 7)


class ArrivalArtifactTests(unittest.TestCase):
    def test_seeded_artifacts_are_byte_reproducible_and_ignore_global_rng(self):
        data = workload()
        a = prepare_arrival_trace(data, request_rate=2, seed=42)
        state = random.getstate()
        try:
            random.seed(999)
            random.random()
            b = prepare_arrival_trace(data, request_rate=2, seed=42)
        finally:
            random.setstate(state)
        with tempfile.TemporaryDirectory() as directory:
            left, right = Path(directory) / 'a.json', Path(directory) / 'b.json'
            write_new(left, a)
            write_new(right, b)
            self.assertEqual(left.read_bytes(), right.read_bytes())
            validate_arrival_trace(json.loads(left.read_text()), data)
        self.assertNotEqual(a['events'], prepare_arrival_trace(data, request_rate=2, seed=43)['events'])
        self.assertEqual(a['events'][0]['offset_s'], 0)

    def test_trace_rejects_tampering_wrong_workload_and_overrides(self):
        data = workload()
        trace = prepare_arrival_trace(data, request_rate=2)
        changed = copy.deepcopy(trace)
        changed['events'][1]['offset_s'] += 1
        with self.assertRaisesRegex(ValueError, 'digest'):
            validate_arrival_trace(changed, data)
        other = copy.deepcopy(data)
        other['sessions'][0]['turns'][1]['delay_s'] = 1
        with self.assertRaisesRegex(ValueError, 'workload mismatch'):
            validate_arrival_trace(trace, other)
        for override in (dict(request_rate=3), dict(arrival_seed=43), dict(arrival_process='constant')):
            with self.assertRaisesRegex(ValueError, 'conflicts'):
                resolve_arrivals(data, arrival_trace=trace, concurrency=2, stagger_seconds=0, **override)

    def test_structural_validation_does_not_depend_on_digest_alone(self):
        data = workload()
        for mutation in ('missing', 'duplicate', 'reordered', 'negative', 'nan', 'first'):
            trace = prepare_arrival_trace(data, request_rate=2)
            if mutation == 'missing':
                trace['events'].pop()
            elif mutation == 'duplicate':
                trace['events'][1] = copy.deepcopy(trace['events'][0])
            elif mutation == 'reordered':
                trace['events'][0], trace['events'][1] = trace['events'][1], trace['events'][0]
            else:
                trace['events'][0 if mutation == 'first' else 1]['offset_s'] = {
                    'negative': -1, 'nan': float('nan'), 'first': .1}[mutation]
            trace['sha256'] = digest({k: v for k, v in trace.items() if k != 'sha256'})
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                validate_arrival_trace(trace, data)

    def test_prepare_trace_cli_is_reproducible_across_processes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'workload.json'
            write_new(source, workload())
            command = [sys.executable, '-m', 'scripts.serving_benchmark.multi_turn', 'prepare-trace',
                       '--workload', str(source), '--request-rate', '2', '--arrival-seed', '7']
            for name in ('a.json', 'b.json'):
                subprocess.run(command + ['--output', str(root / name)], check=True, capture_output=True)
            self.assertEqual((root / 'a.json').read_bytes(), (root / 'b.json').read_bytes())
            retry = subprocess.run(command + ['--output', str(root / 'a.json')], capture_output=True)
            self.assertNotEqual(retry.returncode, 0)


class FakeServer:
    def __init__(self, *, usage=True, cache=None, output="x", short=False, fail_first=False):
        self.usage = usage
        self.cache = cache
        self.output = output
        self.short = short
        self.fail_first = fail_first
        self.payloads = []
        self.active = 0
        self.peak = 0

    async def __call__(self, request):
        body = json.loads(request.content)
        self.payloads.append(body)
        first = len(self.payloads) == 1
        self.active += 1
        self.peak = max(self.peak, self.active)
        await asyncio.sleep(0.005)
        self.active -= 1
        if first and self.fail_first:
            return httpx.Response(500, text="injected failure")
        count = body["max_tokens"] - int(self.short)
        event = {"choices": [{"text": self.output * count, "finish_reason": "length"}]}
        if self.usage:
            event["usage"] = {"prompt_tokens": len(body["prompt"]) + 1,
                              "completion_tokens": count}
            if self.cache is not None:
                event["usage"]["prompt_tokens_details"] = {"cached_tokens": self.cache}
        return httpx.Response(200, text=f"data: {json.dumps(event)}\n\ndata: [DONE]",
                              headers={"content-type": "text/event-stream"})


class MultiTurnTests(unittest.IsolatedAsyncioTestCase):
    async def test_saved_trace_replayed_without_rng_and_survives_response_differences(self):
        data = workload(sessions=2, turns=2)
        for session in data['sessions']:
            session['turns'][1]['delay_s'] = .01
        trace = prepare_arrival_trace(data, request_rate=10000, seed=7)
        results = []
        for output in ('x', 'y'):
            async with httpx.AsyncClient(transport=httpx.MockTransport(FakeServer(output=output))) as client:
                with patch('scripts.serving_benchmark.multi_turn.arrival_offsets',
                           side_effect=AssertionError('replay must not resample')):
                    result = await execute_sessions(client, tokenizer=CharacterTokenizer(), workload=data,
                        base_url='http://test', model=output, concurrency=2, context_limit=200,
                        arrival_trace=json.loads(json.dumps(trace)))
            self.assertTrue(result['complete'])
            self.assertEqual(result['offered_trace'], trace['events'])
            for first, second in zip(result['requests'][::2], result['requests'][1::2]):
                self.assertGreaterEqual(second['send_started_offset_s'], first['completed_offset_s'] + .01)
            results.append(result)
        self.assertEqual(results[0]['offered_trace'], results[1]['offered_trace'])
        self.assertNotEqual(results[0]['requests'][1]['prompt_sha256'], results[1]['requests'][1]['prompt_sha256'])

    async def test_offered_rate_retains_causal_delay_without_changing_trace(self):
        data = workload(sessions=2, turns=2)
        for session in data["sessions"]:
            session["turns"][1]["delay_s"] = .02
        async with httpx.AsyncClient(transport=httpx.MockTransport(FakeServer())) as client:
            result = await execute_sessions(client, tokenizer=CharacterTokenizer(), workload=data,
                base_url="http://test", model="test", concurrency=2, context_limit=200,
                request_rate=10000, arrival_process="constant", arrival_seed=7)
        self.assertTrue(result["complete"])
        for row, expected in zip(result["offered_trace"], [0, .0001, .0002, .0003], strict=True):
            self.assertAlmostEqual(row["offset_s"], expected)
        for a, b in zip(result["requests"][::2], result["requests"][1::2]):
            self.assertAlmostEqual(b["dependency_ready_offset_s"], a["completed_offset_s"] + .02)
            self.assertGreater(b["dependency_delay_s"], 0)
            self.assertGreaterEqual(b["send_started_offset_s"], b["dependency_ready_offset_s"])
            self.assertAlmostEqual(b["offered_to_send_s"], b["send_started_offset_s"] - b["offered_offset_s"])

    async def run_campaign(self, fake, data, warmup=None, **overrides):
        args = SimpleNamespace(api_key=None, timeout=30, concurrency=2,
                               base_url="http://test", model="test", context_limit=200,
                               stagger_seconds=0, engine_label="fake")
        vars(args).update(overrides)

        async def handler(request):
            if request.url.path == "/health":
                return httpx.Response(200)
            return await fake(request)

        # Construct before patching the factory; all requests stay CPU-local.
        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        metrics = AsyncMock(side_effect=["before-warmup", "before-measurement", "after"])
        with patch("scripts.serving_benchmark.multi_turn.httpx.AsyncClient", return_value=client), \
             patch("scripts.serving_benchmark.multi_turn.read_server_metrics", metrics):
            return await run(args, CharacterTokenizer(), data, warmup)

    async def test_result_embeds_trace_and_comparison_rejects_modified_offsets(self):
        data = workload(sessions=2, turns=2)
        trace = prepare_arrival_trace(data, request_rate=10000, seed=7)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'trace.json'
            write_new(path, trace)
            result = await self.run_campaign(FakeServer(), data, arrival_trace=path)
        self.assertEqual(result['arrival_trace'], trace)
        self.assertEqual(result['arrival_seed'], 7)
        self.assertEqual(result['request_rate'], 10000)
        self.assertTrue(compare_results(result, copy.deepcopy(result))['matched_setup_and_geometry'])
        changed = copy.deepcopy(result)
        changed['offered_trace'] = copy.deepcopy(changed['offered_trace'])
        changed['offered_trace'][1]['offset_s'] += .1
        with self.assertRaisesRegex(ValueError, 'saved arrival trace'):
            compare_results(result, changed)

    async def test_warmup_is_separate_and_precedes_measured_requests(self):
        data = workload(sessions=1, turns=2)
        warmup = copy.deepcopy(data)
        warmup["sessions"][0]["initial_prompt"] = "z" * 39
        fake = FakeServer()
        result = await self.run_campaign(fake, data, warmup)
        self.assertTrue(result["complete"])
        self.assertTrue(result["warmup"]["complete"])
        self.assertEqual(result["summary"]["requests"], 2)
        self.assertEqual(result["warmup"]["summary"]["requests"], 2)
        self.assertEqual(fake.payloads[0]["prompt"], "z" * 39)
        self.assertEqual(fake.payloads[2]["prompt"], data["sessions"][0]["initial_prompt"])
        self.assertEqual(result["server_metrics_before_warmup"], "before-warmup")
        self.assertEqual(result["server_metrics_before"], "before-measurement")
        self.assertEqual(result["server_metrics_after"], "after")
        # Offsets belong to the phase's own lifecycle, including client setup.
        self.assertLess(result["requests"][0]["scheduled_offset_s"],
                        result["summary"]["duration_s"])
        self.assertTrue(compare_results(result, copy.deepcopy(result))["matched_setup_and_geometry"])
        changed = copy.deepcopy(result)
        changed["warmup"]["requests"][1]["prompt_tokens"] += 1
        report = compare_results(result, changed)
        self.assertFalse(report["matched_setup_and_geometry"])
        self.assertEqual(len(report["warmup_geometry_differences"]), 1)
        changed = copy.deepcopy(result)
        changed["warmup"] = None
        self.assertEqual(compare_results(result, changed)["setup_differences"],
                         ["warmup_workload_sha256"])
        changed = copy.deepcopy(result)
        changed["warmup"]["workload"]["seed"] += 1
        with self.assertRaisesRegex(ValueError, "warmup workload digest"):
            compare_results(result, changed)

    async def test_failed_warmup_retains_evidence_and_skips_measurement(self):
        data = workload(sessions=1, turns=2)
        warmup = copy.deepcopy(data)
        warmup["sessions"][0]["initial_prompt"] = "z" * 39
        fake = FakeServer(fail_first=True)
        result = await self.run_campaign(fake, data, warmup)
        self.assertFalse(result["complete"])
        self.assertFalse(result["warmup"]["complete"])
        self.assertEqual(len(fake.payloads), 1)
        self.assertEqual(result["requests"], [])
        self.assertEqual(result["unattempted_turns"], 2)
        self.assertEqual(result["summary"]["duration_s"], 0)
        self.assertIsNone(result["summary"]["request_throughput_per_s"])
        self.assertEqual(len(result["warmup"]["requests"]), 1)
        self.assertFalse(compare_results(result, result)["complete"])
        json.dumps(result, allow_nan=False)

    async def test_no_warmup_keeps_existing_behavior(self):
        result = await self.run_campaign(FakeServer(), workload(sessions=1, turns=1))
        self.assertTrue(result["complete"])
        self.assertIsNone(result["warmup"])
        self.assertEqual(result["server_metrics_before"], result["server_metrics_before_warmup"])

    async def test_ordered_warmup_stages_are_excluded_and_compared(self):
        data = workload(sessions=1, turns=1)
        stages = [copy.deepcopy(data) for _ in range(3)]
        for stage, char in zip(stages, 'xyz'):
            stage['sessions'][0]['initial_prompt'] = char * 39
        fake = FakeServer()
        result = await self.run_campaign(fake, data, stages)
        self.assertTrue(result['complete'])
        self.assertIsNone(result['warmup'])
        self.assertEqual([r['workload'] for r in result['warmup_stages']], stages)
        self.assertEqual([p['prompt'] for p in fake.payloads],
                         [s['sessions'][0]['initial_prompt'] for s in stages] +
                         [data['sessions'][0]['initial_prompt']])
        self.assertEqual(result['summary']['requests'], 1)
        self.assertTrue(compare_results(result, copy.deepcopy(result))['matched_setup_and_geometry'])
        changed = copy.deepcopy(result)
        changed['warmup_stages'].reverse()
        self.assertIn('warmup_workload_sha256', compare_results(result, changed)['setup_differences'])
        changed = copy.deepcopy(result)
        changed['warmup_stages'][1]['workload']['seed'] += 1
        with self.assertRaisesRegex(ValueError, 'warmup workload digest'):
            compare_results(result, changed)
        changed = copy.deepcopy(result)
        changed['warmup_stages'][1]['requests'][0]['prompt_tokens'] += 1
        self.assertFalse(compare_results(result, changed)['matched_setup_and_geometry'])

    async def test_failed_stage_skips_later_stages_and_measurement(self):
        data = workload(sessions=1, turns=1)
        stage = copy.deepcopy(data)
        stage['sessions'][0]['initial_prompt'] = 'z' * 39
        fake = FakeServer(fail_first=True)
        result = await self.run_campaign(fake, data, [stage, stage])
        self.assertFalse(result['complete'])
        self.assertEqual(len(fake.payloads), 1)
        self.assertEqual(len(result['warmup_stages']), 1)
        self.assertEqual(result['requests'], [])
        self.assertFalse(compare_results(result, result)['complete'])

    async def test_all_stages_are_validated_before_http(self):
        data = workload(sessions=1, turns=1)
        stage = copy.deepcopy(data)
        stage['sessions'][0]['initial_prompt'] = 'z' * 39
        bad = copy.deepcopy(stage)
        bad['tokenizer_artifact_sha256'] = 'wrong'
        fake = FakeServer()
        with self.assertRaisesRegex(ValueError, 'tokenizer artifacts'):
            await self.run_campaign(fake, data, [stage, bad])
        self.assertEqual(fake.payloads, [])

    async def run_workload(self, fake, data=None, concurrency=2, context_limit=200):
        async with httpx.AsyncClient(transport=httpx.MockTransport(fake)) as client:
            return await execute_sessions(
                client, tokenizer=CharacterTokenizer(), workload=data or workload(),
                base_url="http://test", model="test", concurrency=concurrency,
                context_limit=context_limit,
            )

    async def test_real_answers_carried_forward_and_concurrency_bounded(self):
        fake = FakeServer(cache=0)
        data = workload(sessions=4)
        result = await self.run_workload(fake, data)
        self.assertTrue(result["complete"])
        self.assertEqual(fake.peak, 2)
        self.assertEqual(result["planned_turns"], 12)
        self.assertEqual(result["cold_turns"]["requests"], 4)
        self.assertEqual(result["followup_turns"]["requests"], 8)
        sent = {p["prompt"] for p in fake.payloads}
        for session in data["sessions"]:
            history = session["initial_prompt"]
            for turn in session["turns"]:
                prompt = history + turn["suffix"]
                self.assertIn(prompt, sent)
                history = prompt + "xx"
        self.assertTrue(all(p["ignore_eos"] and p["max_tokens"] == 2 for p in fake.payloads))
        self.assertTrue(all(r["prompt_token_drift"] == 0 for r in result["requests"]))
        self.assertEqual(result["summary"]["cached_token_fraction_observed"], 0)

    async def test_unknown_cache_is_not_zero_and_drift_is_reported(self):
        result = await self.run_workload(FakeServer(output="xy"))
        self.assertTrue(result["complete"])
        self.assertIsNone(result["summary"]["cached_token_fraction_observed"])
        self.assertEqual(result["summary"]["cache_unknown_requests"], 6)
        self.assertEqual([r["prompt_token_drift"] for r in result["requests"]], [0, 2, 4] * 2)
        self.assertTrue(all(r["retokenized_output_tokens"] == 4 for r in result["requests"]))

    async def test_failed_session_stops_but_other_sessions_finish(self):
        result = await self.run_workload(FakeServer(fail_first=True))
        self.assertFalse(result["complete"])
        self.assertEqual(result["summary"]["successful_requests"], 3)
        self.assertEqual(result["unattempted_turns"], 2)
        self.assertEqual(len(result["session_failures"]), 1)

    async def test_missing_usage_short_output_invalid_cache_fail_closed(self):
        for fake in (FakeServer(usage=False), FakeServer(short=True), FakeServer(cache=-1)):
            with self.subTest(fake=fake):
                result = await self.run_workload(fake)
                self.assertFalse(result["complete"])
                self.assertEqual(len(result["requests"]), 2)
                self.assertEqual(result["unattempted_turns"], 4)

    async def test_actual_context_overflow_is_retained_not_truncated(self):
        # Nominal total is 60, but the answer's text re-tokenizes much longer.
        result = await self.run_workload(FakeServer(output="x" * 15),
                                         workload(sessions=1, turns=3), context_limit=70)
        self.assertFalse(result["complete"])
        self.assertEqual(len(result["requests"]), 1)
        self.assertIn("actual context", result["session_failures"][0]["error"])

    async def test_think_delay_is_after_previous_completion(self):
        data = workload(sessions=1, turns=2)
        data["sessions"][0]["turns"][1]["delay_s"] = 0.02
        result = await self.run_workload(FakeServer(), data)
        a, b = result["requests"]
        self.assertAlmostEqual(b["scheduled_offset_s"] - a["completed_offset_s"], 0.02)
        self.assertGreaterEqual(b["send_started_offset_s"], b["scheduled_offset_s"])

    async def test_hidden_output_keeps_timing_unknown_but_usage_validates(self):
        result = await self.run_workload(FakeServer(output=""))
        self.assertTrue(result["complete"])
        self.assertIsNone(result["summary"]["time_to_first_token_ms"]["mean"])
        self.assertEqual(result["summary"]["ttft_observed_requests"], 0)
        self.assertEqual([r["prompt_token_drift"] for r in result["requests"]], [0, -2, -4] * 2)

    async def test_usage_on_metadata_and_eof_finish_are_consumed(self):
        payload = {"choices": [{"text": "", "finish_reason": None}],
                   "usage": {"prompt_tokens": 2, "completion_tokens": 1}}
        self.assertEqual(parse_stream_payload(json.dumps(payload))["usage"], payload["usage"])

        def handler(request):
            return httpx.Response(200, text="data: " + json.dumps(payload) + "\n\ndata: [DONE]")
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            row = await send_request(client, base_url="http://test", endpoint="completions",
                                     model="test", request={"prompt": "x", "prompt_tokens": 2, "max_tokens": 1},
                                     index=0, scheduled_offset_s=0, benchmark_start=0,
                                     retain_output=True, require_usage=True)
        self.assertTrue(row["success"])
        self.assertEqual(row["generated_text"], "")

    async def test_comparison_requires_setup_complete_turns_and_actual_geometry(self):
        data = workload()
        result = await self.run_workload(FakeServer(), data)
        result.update(schema="vosti.multi-turn-result.v1", workload=data,
                      workload_sha256=digest(data), concurrency=2, context_limit=200,
                      stagger_seconds=0, client_source_sha256={"client": "test"},
                      engine_label="test")
        self.assertTrue(compare_results(result, copy.deepcopy(result))["matched_setup_and_geometry"])
        changed = copy.deepcopy(result)
        changed["requests"][1]["prompt_tokens"] += 1
        report = compare_results(result, changed)
        self.assertFalse(report["matched_setup_and_geometry"])
        self.assertEqual(len(report["geometry_differences"]), 1)
        changed = copy.deepcopy(result)
        changed["concurrency"] = 3
        self.assertEqual(compare_results(result, changed)["setup_differences"], ["concurrency"])
        changed = copy.deepcopy(result)
        changed["complete"] = False
        self.assertFalse(compare_results(result, changed)["matched_setup_and_geometry"])
        changed = copy.deepcopy(result)
        changed["requests"].pop()
        self.assertFalse(compare_results(result, changed)["complete"])
        changed = copy.deepcopy(result)
        changed["workload"]["seed"] += 1
        with self.assertRaisesRegex(ValueError, "digest"):
            compare_results(result, changed)


class PreparationTests(unittest.TestCase):
    def test_warmup_validation_rejects_duplicate_prompts_and_tokenizer_drift(self):
        data = workload()
        with self.assertRaisesRegex(ValueError, "repeats"):
            validate_warmup(data, copy.deepcopy(data), CharacterTokenizer(), 200)
        warmup = copy.deepcopy(data)
        warmup["tokenizer_artifact_sha256"] = {"tokenizer": "different"}
        with self.assertRaisesRegex(ValueError, "tokenizer artifacts"):
            validate_warmup(data, warmup, CharacterTokenizer(), 200)
        warmup = copy.deepcopy(data)
        warmup["sessions"][0]["initial_tokens"] += 1
        with self.assertRaisesRegex(ValueError, "initial token"):
            validate_warmup(data, warmup, CharacterTokenizer(), 200)

    def test_exact_text_and_reproducible_manifest(self):
        for special in (True, False):
            text = exact_text(CharacterTokenizer(), 32, random.Random(1), special=special)
            self.assertEqual(len(CharacterTokenizer().encode(text, add_special_tokens=special)), 32)
        self.assertEqual(workload(), workload())

    def test_validation_rejects_nominal_overflow_and_changed_token_counts(self):
        with self.assertRaisesRegex(ValueError, "nominal"):
            validate_workload(workload(), CharacterTokenizer(), 50)
        data = copy.deepcopy(workload())
        data["sessions"][0]["initial_tokens"] += 1
        with self.assertRaisesRegex(ValueError, "initial token"):
            validate_workload(data, CharacterTokenizer(), 200)

    def test_invalid_parameters_and_artifact_preservation(self):
        with self.assertRaises(ValueError):
            prepare(CharacterTokenizer(), sessions=1, turns=2, initial_tokens=32,
                    suffix_tokens=4, output_tokens=2, think_seconds=float("nan"), seed=1)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            write_new(path, {"original": True})
            with self.assertRaises(FileExistsError):
                write_new(path, {"replacement": True})
            self.assertEqual(json.loads(path.read_text()), {"original": True})


if __name__ == "__main__":
    unittest.main()
