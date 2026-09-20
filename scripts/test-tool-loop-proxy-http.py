#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import http.client
import json
import pathlib
import signal
import socket
import subprocess
import sys
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "tool-loop-guard.py"


def load_module():
    spec = importlib.util.spec_from_file_location("tool_loop_guard_http", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeUpstreamHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, format: str, *args):
        return

    def do_POST(self):  # noqa: N802
        length = int(self.headers["Content-Length"])
        self.server.last_body = self.rfile.read(length)  # type: ignore[attr-defined]
        self.server.last_headers = dict(self.headers)  # type: ignore[attr-defined]
        body = self.server.response_body  # type: ignore[attr-defined]
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class ProxyHttpTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.guard = load_module()
        cls.upstream = ThreadingHTTPServer(("127.0.0.1", 0), FakeUpstreamHandler)
        cls.upstream.response_body = b""  # type: ignore[attr-defined]
        cls.upstream_thread = threading.Thread(
            target=cls.upstream.serve_forever, daemon=True
        )
        cls.upstream_thread.start()
        upstream_url = f"http://127.0.0.1:{cls.upstream.server_port}"
        cls.proxy = cls.guard.GuardingProxyServer(("127.0.0.1", 0), upstream_url, 10)
        cls.proxy_thread = threading.Thread(target=cls.proxy.serve_forever, daemon=True)
        cls.proxy_thread.start()
        cls.base = f"http://127.0.0.1:{cls.proxy.server_port}"
        cls.command = 'rg "WhereToGo.uasset" 2>$null'

    @classmethod
    def tearDownClass(cls):
        cls.proxy.shutdown()
        cls.proxy.server_close()
        cls.upstream.shutdown()
        cls.upstream.server_close()

    def payload(self):
        messages = [{"role": "user", "content": "inspect"}]
        for index in range(2):
            call_id = f"call_{index}"
            messages += [
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{
                        "id": call_id,
                        "type": "function",
                        "function": {
                            "name": "exec_command",
                            "arguments": json.dumps({"cmd": self.command}),
                        },
                    }],
                },
                {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": "Process exited with code 1\nOutput:\n",
                },
            ]
        return {
            "model": "overlord-testing",
            "messages": messages,
            "tools": [{
                "type": "function",
                "function": {"name": "exec_command", "parameters": {"type": "object"}},
            }],
            "stream": True,
            "stream_options": {"include_usage": True},
        }

    @staticmethod
    def stream_for(command: str, *, usage: bool = True) -> bytes:
        first = {
            "id": "chatcmpl-test",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "overlord-testing",
            "system_fingerprint": "fp_test",
            "choices": [{
                "index": 0,
                "delta": {
                    "role": "assistant",
                    "tool_calls": [{
                        "index": 0,
                        "id": "next",
                        "type": "function",
                        "function": {"name": "exec_command", "arguments": command[:10]},
                    }],
                },
                "finish_reason": None,
            }],
        }
        second = {
            "id": "chatcmpl-test",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "overlord-testing",
            "choices": [{
                "index": 0,
                "delta": {"tool_calls": [{"index": 0, "function": {"arguments": command[10:]}}]},
                "finish_reason": "tool_calls",
            }],
        }
        events = [first, second]
        if usage:
            events.append({
                "id": "chatcmpl-test",
                "object": "chat.completion.chunk",
                "created": 1,
                "model": "overlord-testing",
                "choices": [],
                "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
            })
        return (
            "".join("data: " + json.dumps(event) + "\n\n" for event in events)
            + "data: [DONE]\n\n"
        ).encode()

    def post(self, path: str, payload=None):
        body = json.dumps(self.payload() if payload is None else payload).encode()
        connection = http.client.HTTPConnection(
            "127.0.0.1", self.proxy.server_port, timeout=15
        )
        connection.putrequest("POST", path)
        connection.putheader("Content-Type", "application/json")
        connection.putheader("Content-Length", str(len(body)))
        connection.putheader("Authorization", "Bearer test-only")
        connection.putheader("Connection", "X-Remove-Me")
        connection.putheader("Connection", "X-Remove-Second")
        connection.putheader("X-Remove-Me", "secret")
        connection.putheader("X-Remove-Second", "secret-two")
        connection.putheader("Proxy-Connection", "keep-alive")
        connection.putheader("Trailer", "X-Later")
        connection.endheaders(body)
        response = connection.getresponse()
        response_body = response.read()
        headers = dict(response.getheaders())
        connection.close()
        return headers, response_body

    def test_query_path_is_guarded_and_safe_stream_is_byte_identical(self):
        safe = self.stream_for(json.dumps({"cmd": "Get-ChildItem Content\\WorldMap"}))
        self.upstream.response_body = safe  # type: ignore[attr-defined]
        headers, body = self.post("/v1/chat/completions?trace=test")
        self.assertEqual(body, safe)
        self.assertEqual(headers.get("Content-Type"), "text/event-stream; charset=utf-8")
        self.assertIsNone(headers.get("X-GLM-Loop-Guard"))
        forwarded = json.loads(self.upstream.last_body)  # type: ignore[attr-defined]
        self.assertEqual(forwarded["repetition_penalty"], 1.1)
        upstream_headers = self.upstream.last_headers  # type: ignore[attr-defined]
        self.assertIn("Authorization", upstream_headers)
        self.assertNotIn("X-Remove-Me", upstream_headers)
        self.assertNotIn("X-Remove-Second", upstream_headers)
        self.assertNotIn("Proxy-Connection", upstream_headers)
        self.assertNotIn("Trailer", upstream_headers)

    def test_unarmed_passthrough_preserves_upstream_content_type(self):
        safe = self.stream_for(json.dumps({"cmd": "safe"}))
        self.upstream.response_body = safe  # type: ignore[attr-defined]
        headers, body = self.post(
            "/v1/chat/completions",
            {"model": "overlord-testing", "messages": [{"role": "user", "content": "hello"}]},
        )
        self.assertEqual(body, safe)
        self.assertEqual(headers.get("Content-Type"), "text/event-stream")

    def test_split_repeated_stream_is_blocked_and_usage_is_preserved(self):
        self.upstream.response_body = self.stream_for(  # type: ignore[attr-defined]
            json.dumps({"cmd": self.command})
        )
        headers, body = self.post("/v1/chat/completions")
        self.assertEqual(headers.get("X-GLM-Loop-Guard"), "blocked")
        events = [
            json.loads(line[6:])
            for line in body.decode().splitlines()
            if line.startswith("data: {")
        ]
        calls = [
            call
            for event in events
            for choice in event.get("choices", [])
            for call in choice.get("delta", {}).get("tool_calls", [])
        ]
        self.assertEqual(calls, [])
        self.assertIn("LOOP_GUARD", events[0]["choices"][0]["delta"]["content"])
        self.assertEqual(events[-1]["choices"], [])
        self.assertEqual(events[-1]["usage"]["total_tokens"], 12)

    def test_non_loopback_upstream_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "loopback"):
            self.guard.GuardingProxyServer(("127.0.0.1", 0), "http://203.0.113.1:8000", 1)

    def test_localhost_must_resolve_only_to_loopback(self):
        with mock.patch.object(
            self.guard.socket,
            "getaddrinfo",
            return_value=[(2, 1, 6, "", ("203.0.113.10", 8000))],
        ):
            with self.assertRaisesRegex(ValueError, "loopback"):
                self.guard.GuardingProxyServer(
                    ("127.0.0.1", 0), "http://localhost:8000", 1
                )

    def test_sigint_exits_proxy_promptly(self):
        probe = socket.socket()
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
        probe.close()
        process = subprocess.Popen(
            [
                "bash", "-c", 'trap "" INT; exec "$@"', "bash",
                sys.executable,
                str(MODULE_PATH),
                "--bind", "127.0.0.1",
                "--port", str(port),
                "--upstream", f"http://127.0.0.1:{self.upstream.server_port}",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        try:
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                try:
                    with socket.create_connection(("127.0.0.1", port), timeout=0.1):
                        break
                except OSError:
                    time.sleep(0.02)
            else:
                self.fail("proxy did not start")
            process.send_signal(signal.SIGINT)
            self.assertEqual(process.wait(timeout=2), 0)
        finally:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
