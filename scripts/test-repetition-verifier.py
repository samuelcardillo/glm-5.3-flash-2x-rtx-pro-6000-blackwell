#!/usr/bin/env python3
import importlib.util
import json
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "verify-repetition.py"
SPEC = importlib.util.spec_from_file_location("repetition_verifier", SCRIPT)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"could not load {SCRIPT}")
mod = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mod)


class FakeResponse:
    def __init__(self, lines):
        self.lines = [line.encode() for line in lines]

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def __iter__(self):
        return iter(self.lines)


def sse(event):
    return f"data: {json.dumps(event)}\n"


class RepetitionVerifierTests(unittest.TestCase):
    def test_repeated_phrase_must_dominate_output(self):
        phrase = "archive ctx was audited"
        coherent = " ".join(f"unique{i}" for i in range(300)) + " " + " ".join(
            [phrase] * 12
        )
        degenerate = " ".join([phrase] * 20)
        self.assertFalse(mod.detect_loop(coherent)[0])
        self.assertTrue(mod.detect_loop(degenerate)[0])

    def call_stream(self, lines):
        with mock.patch.object(mod.urllib.request, "urlopen", return_value=FakeResponse(lines)):
            return mod.stream_one("http://unused", {"stream": True}, 1)

    def test_empty_stream_fails(self):
        with self.assertRaisesRegex(RuntimeError, "empty SSE stream"):
            self.call_stream([])

    def test_sse_error_object_fails(self):
        with self.assertRaisesRegex(RuntimeError, "server SSE error"):
            self.call_stream([sse({"error": {"message": "engine failed"}})])

    def test_non_object_sse_event_fails(self):
        with self.assertRaisesRegex(RuntimeError, "not an object"):
            self.call_stream(["data: []\n"])

    def test_empty_generated_output_fails(self):
        event = {
            "choices": [{"delta": {}, "finish_reason": "stop"}],
            "usage": {"completion_tokens": 0},
        }
        with self.assertRaisesRegex(RuntimeError, "empty generated output"):
            self.call_stream([sse(event), "data: [DONE]\n"])

    def test_stream_without_done_fails(self):
        event = {"choices": [{"delta": {"content": "partial"}, "finish_reason": None}]}
        with self.assertRaisesRegex(RuntimeError, r"without \[DONE\]"):
            self.call_stream([sse(event)])

    def test_stream_without_finish_reason_fails(self):
        with self.assertRaisesRegex(RuntimeError, "finish_reason"):
            self.call_stream(["data: [DONE]\n"])

    def test_stream_without_usage_fails(self):
        event = {"choices": [{"delta": {}, "finish_reason": "stop"}]}
        with self.assertRaisesRegex(RuntimeError, "usage"):
            self.call_stream([sse(event), "data: [DONE]\n"])

    def test_valid_stream_passes(self):
        events = [
            sse({"choices": [{"delta": {"content": "OK"}, "finish_reason": None}]}),
            sse(
                {
                    "choices": [{"delta": {}, "finish_reason": "stop"}],
                    "usage": {"completion_tokens": 1},
                }
            ),
            "data: [DONE]\n",
        ]
        result = self.call_stream(events)
        self.assertEqual(result["content"], "OK")
        self.assertEqual(result["finish_reason"], "stop")
        self.assertFalse(result["loop"])


if __name__ == "__main__":
    unittest.main(verbosity=2)