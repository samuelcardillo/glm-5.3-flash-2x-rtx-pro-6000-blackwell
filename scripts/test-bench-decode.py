#!/usr/bin/env python3
import importlib.util
import pathlib
import unittest

HERE = pathlib.Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("bench_decode", HERE / "bench-decode.py")


def load_module():
    module = importlib.util.module_from_spec(SPEC)
    SPEC.loader.exec_module(module)
    return module


class DecodeBenchmarkTests(unittest.TestCase):
    def test_fragmented_sse_preserves_three_output_channels_and_usage(self):
        mod = load_module()
        chunks = [
            b'data: {"choices":[{"delta":{"reasoning":"r1"},"finish_reason":null}]}\n\n',
            b'data: {"choices":[{"delta":{"content":"answer","reasoning_content":"rc"},',
            b'"finish_reason":"stop"}],"usage":{"prompt_tokens":3,"completion_tokens":2}}\n\n',
            b'data: [DO', b'NE]\n\n',
        ]
        parsed = mod.parse_sse(chunks)
        self.assertEqual(parsed["content"], "answer")
        self.assertEqual(parsed["reasoning"], "r1")
        self.assertEqual(parsed["reasoning_content"], "rc")
        self.assertEqual(parsed["finish_reason"], "stop")
        self.assertEqual(parsed["usage"], {"prompt_tokens": 3, "completion_tokens": 2})

    def test_stream_fails_closed_for_missing_done_empty_or_malformed_data(self):
        mod = load_module()
        cases = [
            [b'data: {"choices":[{"delta":{"content":"x"},"finish_reason":"stop"}]}\n\n'],
            [b'data: {not json}\n\ndata: [DONE]\n\n'],
            [b'data: {"choices":[{"delta":{},"finish_reason":"stop"}],"usage":{"completion_tokens":0}}\n\ndata: [DONE]\n\n'],
            [b'data: {"choices":[{"delta":{"content":"x"},"finish_reason":"stop"}],"usage":{"completion_tokens":1}}\n\ndata: [DONE]\n\n'],
            [b'data: {"choices":[{"delta":{"content":"x"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1}}\n\ndata: [DONE]\n\ndata: [DONE]\n\n'],
        ]
        for chunks in cases:
            with self.subTest(chunks=chunks), self.assertRaises(mod.StreamError):
                mod.parse_sse(chunks)

    def test_explicit_sse_error_fails_closed(self):
        mod = load_module()
        chunks = [b'event: error\ndata: {"error":{"message":"bad"}}\n\n']
        with self.assertRaisesRegex(mod.StreamError, "SSE error"):
            mod.parse_sse(chunks)

    def test_metric_deltas_include_mtp_totals_acceptance_and_positions(self):
        mod = load_module()
        before = (
            'vllm:spec_decode_num_drafts_total{engine="0",model_name="m"} 10\n'
            'vllm:spec_decode_num_drafts_created{engine="0",model_name="m"} 999\n'
            'vllm:spec_decode_num_draft_tokens_total{engine="0",model_name="m"} 40\n'
            'vllm:spec_decode_num_accepted_tokens_total{engine="0",model_name="m"} 20\n'
            'vllm:spec_decode_num_accepted_tokens_per_pos_total{engine="0",model_name="m",position="0"} 8\n'
        )
        after = (
            'vllm:spec_decode_num_drafts_total{engine="0",model_name="m"} 20\n'
            'vllm:spec_decode_num_drafts_created{engine="0",model_name="m"} 999\n'
            'vllm:spec_decode_num_draft_tokens_total{engine="0",model_name="m"} 80\n'
            'vllm:spec_decode_num_accepted_tokens_total{engine="0",model_name="m"} 52\n'
            'vllm:spec_decode_num_accepted_tokens_per_pos_total{engine="0",model_name="m",position="0"} 18\n'
        )
        delta = mod.mtp_metric_delta(before, after)
        self.assertEqual(delta["draft_steps"], 10.0)
        self.assertEqual(delta["draft_tokens"], 40.0)
        self.assertEqual(delta["accepted_tokens"], 32.0)
        self.assertEqual(delta["acceptance_rate"], 0.8)
        self.assertEqual(delta["accepted_tokens_per_step"], 3.2)
        self.assertEqual(delta["per_position_acceptance"], {"0": 1.0})

    def test_response_chunks_use_nonblocking_read1_boundaries(self):
        mod = load_module()
        class Response:
            def __init__(self):
                self.parts = [b"first", b"second", b""]
            def read1(self, size):
                return self.parts.pop(0)
            def read(self, size):
                raise AssertionError("buffer-filling read would corrupt TTFT")
        self.assertEqual(list(mod.iter_response_chunks(Response())), [b"first", b"second"])

    def test_result_and_summary_have_reproducibility_fields(self):
        mod = load_module()
        result = mod.make_result(
            fixture_id="public-short-prose-v1", fixture=b"public fixture", health_before=True,
            health_after=True, warmups=2, run_index=1, started=10.0, first_token=10.2,
            finished=11.2, parsed={"content":"x", "reasoning":"", "reasoning_content":"",
            "usage":{"prompt_tokens":4,"completion_tokens":10}, "finish_reason":"stop"}, mtp={"draft_tokens":5.0})
        for key in ("fixture_id", "fixture_sha256", "health_before", "health_after", "warmup_count", "run_index", "ttft_seconds", "decode_seconds", "decode_tokens_per_second", "wall_seconds", "prompt_tokens", "completion_tokens", "finish_reason", "mtp"):
            self.assertIn(key, result)
        self.assertEqual(result["decode_tokens_per_second"], 9.0)
        summary = mod.summarize([{"ttft_seconds": 1.0, "decode_tokens_per_second": 2.0}, {"ttft_seconds": 3.0, "decode_tokens_per_second": 4.0}])
        self.assertEqual(summary["run_count"], 2)
        self.assertIn("median", summary["ttft_seconds"])
        self.assertIn("p90", summary["decode_tokens_per_second"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
