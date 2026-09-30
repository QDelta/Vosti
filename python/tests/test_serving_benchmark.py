import asyncio
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from threading import Thread
import unittest

import httpx
import pytest

from scripts.serving_benchmark.protocol import (
    SseDecoder,
    arrival_offsets,
    parse_stream_payload,
    percentile,
    summarize_requests,
)
from scripts.serving_benchmark.run import execute_workload


class FakeOpenAIHandler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:
        content_length = int(self.headers["Content-Length"])
        request = json.loads(self.rfile.read(content_length))
        if self.path != "/v1/completions":
            self.send_error(404)
            return

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        token_groups = (
            [[0]]
            if request["max_tokens"] == 1
            else [[0], list(range(1, request["max_tokens"]))]
        )
        for token_ids in token_groups:
            chunk = {
                "choices": [
                    {
                        "text": "x" * len(token_ids),
                        "finish_reason": None,
                        "token_ids": token_ids,
                    }
                ]
            }
            self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
            self.wfile.flush()
        finish = {
            "choices": [{"text": "", "finish_reason": "length"}],
            "usage": {"completion_tokens": request["max_tokens"]},
        }
        self.wfile.write(f"data: {json.dumps(finish)}\n\n".encode())
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()

    def log_message(self, format: str, *args: object) -> None:
        del format, args


class ServingBenchmarkProtocolTests(unittest.TestCase):
    def test_constant_arrivals_are_exact_and_poisson_is_reproducible(self) -> None:
        self.assertEqual(arrival_offsets(4, 2.0, 7, "constant"), [0.0, 0.5, 1.0, 1.5])
        self.assertEqual(
            arrival_offsets(8, 3.0, 11, "poisson"),
            arrival_offsets(8, 3.0, 11, "poisson"),
        )
        self.assertNotEqual(
            arrival_offsets(8, 3.0, 11, "poisson"),
            arrival_offsets(8, 3.0, 12, "poisson"),
        )

    def test_sse_decoder_handles_comments_and_multiline_data(self) -> None:
        decoder = SseDecoder()
        self.assertEqual(decoder.feed_line(": keepalive"), [])
        self.assertEqual(decoder.feed_line("data: first"), [])
        self.assertEqual(decoder.feed_line("data: second"), [])
        self.assertEqual(decoder.feed_line(""), ["first\nsecond"])
        self.assertEqual(decoder.finish(), [])

    def test_openai_stream_payload_classification(self) -> None:
        output = parse_stream_payload(
            '{"choices":[{"text":"x","finish_reason":null,"token_id":3}]}'
        )
        self.assertEqual(
            output,
            {
                "kind": "output",
                "text": "x",
                "token_ids": [3],
                "finish_reason": None,
                "usage": None,
            },
        )
        terminal_output = parse_stream_payload(
            '{"choices":[{"text":"xy","finish_reason":"length",'
            '"token_ids":[4,5]}],"usage":{"completion_tokens":2}}'
        )
        self.assertEqual(terminal_output["kind"], "output")
        self.assertEqual(terminal_output["token_ids"], [4, 5])
        self.assertEqual(terminal_output["finish_reason"], "length")
        finish = parse_stream_payload(
            '{"choices":[{"text":"","finish_reason":"length"}],'
            '"usage":{"completion_tokens":4}}'
        )
        self.assertEqual(finish["kind"], "finish")
        self.assertEqual(finish["usage"]["completion_tokens"], 4)
        self.assertEqual(parse_stream_payload("[DONE]"), {"kind": "done"})

    def test_summary_separates_success_and_failure(self) -> None:
        records = [
            {
                "success": True,
                "prompt_tokens": 10,
                "output_tokens": 4,
                "latency_s": 2.0,
                "ttft_s": 0.5,
                "tpot_s": 0.5,
                "inter_output_event_latency_s": [0.4, 0.5, 0.6],
                "scheduling_lag_s": 0.01,
            },
            {
                "success": False,
                "prompt_tokens": 20,
                "output_tokens": 0,
                "latency_s": 1.0,
                "ttft_s": None,
                "tpot_s": None,
                "inter_output_event_latency_s": [],
                "scheduling_lag_s": 0.02,
            },
        ]
        summary = summarize_requests(records, 2.0)
        self.assertEqual(summary["successful_requests"], 1)
        self.assertEqual(summary["failed_requests"], 1)
        self.assertEqual(summary["output_token_throughput_per_s"], 2.0)
        self.assertEqual(summary["time_to_first_token_ms"]["p50"], 500.0)
        self.assertEqual(percentile([1.0, 3.0], 0.5), 2.0)

    def test_summary_allows_successful_generation_without_visible_output(self) -> None:
        record = {
            "success": True,
            "prompt_tokens": 5,
            "output_tokens": 2,
            "latency_s": 0.2,
            "ttft_s": None,
            "tpot_s": None,
            "inter_output_event_latency_s": [],
            "scheduling_lag_s": 0.0,
        }
        summary = summarize_requests([record], 1.0)
        self.assertEqual(summary["successful_requests"], 1)
        self.assertIsNone(summary["time_to_first_token_ms"]["mean"])

    @pytest.mark.loopback
    def test_open_loop_client_completes_real_sse_exchange(self) -> None:
        server = ThreadingHTTPServer(("127.0.0.1", 0), FakeOpenAIHandler)
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()

        async def run_client() -> tuple[list[dict], float, list[float]]:
            async with httpx.AsyncClient(timeout=5.0) as client:
                return await execute_workload(
                    client,
                    base_url=f"http://127.0.0.1:{server.server_port}",
                    endpoint="completions",
                    model="test-model",
                    requests=[
                        {"prompt": "first", "prompt_tokens": 1, "max_tokens": 3},
                        {"prompt": "second", "prompt_tokens": 2, "max_tokens": 2},
                    ],
                    request_rate=100.0,
                    arrival_process="constant",
                    seed=7,
                )

        try:
            records, duration_s, offsets = asyncio.run(run_client())
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5.0)

        self.assertGreater(duration_s, 0.0)
        self.assertEqual(offsets, [0.0, 0.01])
        self.assertTrue(all(record["success"] for record in records), records)
        self.assertEqual([record["output_tokens"] for record in records], [3, 2])
        self.assertEqual(
            [record["stream_output_events"] for record in records], [2, 2]
        )
        self.assertEqual(
            [record["stream_reported_token_ids"] for record in records], [3, 2]
        )
        self.assertEqual([record["finish_reason"] for record in records], ["length"] * 2)


if __name__ == "__main__":
    unittest.main()
