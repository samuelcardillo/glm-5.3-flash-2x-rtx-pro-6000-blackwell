#!/usr/bin/env python3
"""OpenAI-compatible reverse proxy that prevents repeated failed tool-call loops."""
from __future__ import annotations

import argparse
import copy
import hashlib
import http.client
import ipaddress
import json
import logging
import re
import signal
import socket
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, NamedTuple
from urllib.parse import urlsplit

BASELINE_REPETITION_PENALTY = 1.05
ARMED_REPETITION_PENALTY = 1.10
MAX_REQUEST_BYTES = 128 * 1024 * 1024
MAX_GUARDED_RESPONSE_BYTES = 64 * 1024 * 1024
MAX_CONCURRENT_REQUESTS = 16
CLIENT_READ_TIMEOUT = 30
LOOP_GUARD_MESSAGE = (
    "LOOP_GUARD: The model requested the same tool action after two identical "
    "failed results. The repeated call was blocked. Start a new turn and use a "
    "different path or inspection strategy."
)
HOP_BY_HOP = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
    "proxy-connection",
}
LOG = logging.getLogger("tool-loop-guard")


class GuardState(NamedTuple):
    armed: bool
    fingerprints: frozenset[str]
    client_requested_stream: bool


def _arguments(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {"_raw": value}
        if isinstance(parsed, dict):
            return parsed
    return {"_raw": value}


def function_fingerprint(function: dict[str, Any]) -> str:
    name = str(function.get("name") or "")
    arguments = _arguments(function.get("arguments"))
    if name == "exec_command":
        if "_raw" in arguments:
            relevant = {"raw": arguments["_raw"]}
        else:
            relevant = {
                "command": arguments.get("cmd", arguments.get("command")),
                "workdir": arguments.get("workdir"),
            }
    else:
        relevant = arguments
    return json.dumps(
        {"name": name, "arguments": relevant},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                for key in ("text", "output", "content"):
                    if key in item:
                        parts.append(_content_text(item[key]))
        return "\n".join(parts)
    if isinstance(content, dict):
        return json.dumps(content, ensure_ascii=False, sort_keys=True)
    return "" if content is None else str(content)


def _failed_tool_result(message: dict[str, Any]) -> bool:
    text = _content_text(message.get("content"))
    match = re.match(
        r"\s*(?:process\s+)?(?:exited with code|exit code)\s*[:=]?\s*(-?\d+)",
        text,
        re.I,
    )
    if match:
        return int(match.group(1)) != 0
    try:
        parsed = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        parsed = None
    if isinstance(parsed, dict):
        for key in ("exit_code", "returncode"):
            value = parsed.get(key)
            if type(value) is int:
                return value != 0
        status = parsed.get("status")
        if type(status) is int and 400 <= status <= 599:
            return True
        if isinstance(status, str) and status.lower() in {
            "error",
            "failed",
            "failure",
            "timeout",
            "timed_out",
        }:
            return True
        if parsed.get("ok") is False or parsed.get("success") is False:
            return True
        if parsed.get("error") not in (None, "", False):
            return True
    lowered = text.lstrip().lower()
    return (
        lowered.startswith("traceback (")
        or lowered.startswith("exception:")
        or lowered.startswith("timeout:")
        or lowered.startswith("transport error:")
    )


def _result_fingerprint(message: dict[str, Any]) -> str:
    normalized = re.sub(r"\s+", " ", _content_text(message.get("content")).strip())
    return hashlib.sha256(normalized.encode()).hexdigest()


def _completed_turns(messages: list[Any]) -> list[set[tuple[str, str]]]:
    results: dict[str, dict[str, Any]] = {}
    for message in messages:
        if isinstance(message, dict) and message.get("role") == "tool":
            call_id = message.get("tool_call_id")
            if isinstance(call_id, str):
                results[call_id] = message

    turns: list[set[tuple[str, str]]] = []
    for message in messages:
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        calls = message.get("tool_calls")
        if not isinstance(calls, list):
            continue
        completed: set[tuple[str, str]] = set()
        had_completed_call = False
        for call in calls:
            if not isinstance(call, dict):
                continue
            call_id = call.get("id")
            function = call.get("function")
            result = results.get(call_id) if isinstance(call_id, str) else None
            if isinstance(function, dict) and result is not None:
                had_completed_call = True
                if _failed_tool_result(result):
                    completed.add(
                        (function_fingerprint(function), _result_fingerprint(result))
                    )
        if had_completed_call:
            turns.append(completed)
    return turns


def _repeated_failed_fingerprints(messages: list[Any]) -> frozenset[str]:
    last_user_index = max(
        (
            index
            for index, message in enumerate(messages)
            if isinstance(message, dict) and message.get("role") == "user"
        ),
        default=-1,
    )
    turns = _completed_turns(messages[last_user_index + 1 :])
    if len(turns) < 2:
        return frozenset()
    repeated = turns[-2] & turns[-1]
    return frozenset(fingerprint for fingerprint, _ in repeated)


def _at_least(value: Any, minimum: float) -> float:
    if type(value) in (int, float) and value >= minimum:
        return float(value)
    return minimum


def protect_payload(payload: dict[str, Any]) -> tuple[dict[str, Any], GuardState]:
    protected = dict(payload)
    client_requested_stream = protected.get("stream") is True
    messages = protected.get("messages")
    fingerprints = (
        _repeated_failed_fingerprints(messages)
        if isinstance(messages, list)
        else frozenset()
    )
    armed = bool(fingerprints)
    tools = protected.get("tools")
    agent_request = isinstance(tools, list) and bool(tools)
    if not agent_request and isinstance(messages, list):
        agent_request = any(
            isinstance(message, dict) and bool(message.get("tool_calls"))
            for message in messages
        )
    if agent_request:
        minimum = ARMED_REPETITION_PENALTY if armed else BASELINE_REPETITION_PENALTY
        protected["repetition_penalty"] = _at_least(
            protected.get("repetition_penalty"), minimum
        )
    return protected, GuardState(armed, fingerprints, client_requested_stream)


def _response_fingerprints(response: dict[str, Any]) -> set[str]:
    fingerprints: set[str] = set()
    choices = response.get("choices")
    if not isinstance(choices, list):
        return fingerprints
    for choice in choices:
        if not isinstance(choice, dict):
            continue
        message = choice.get("message")
        if not isinstance(message, dict):
            continue
        calls = message.get("tool_calls")
        if not isinstance(calls, list):
            continue
        for call in calls:
            if isinstance(call, dict) and isinstance(call.get("function"), dict):
                fingerprints.add(function_fingerprint(call["function"]))
    return fingerprints


def _choice_fingerprints(choice: dict[str, Any]) -> set[str]:
    message = choice.get("message")
    if not isinstance(message, dict):
        return set()
    calls = message.get("tool_calls")
    if not isinstance(calls, list):
        return set()
    return {
        function_fingerprint(call["function"])
        for call in calls
        if isinstance(call, dict) and isinstance(call.get("function"), dict)
    }


def sanitize_response(
    response: dict[str, Any], state: GuardState
) -> tuple[dict[str, Any], bool]:
    if not state.armed:
        return response, False
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("guarded upstream response has no choices")
    offending = {
        index
        for index, choice in enumerate(choices)
        if isinstance(choice, dict)
        and _choice_fingerprints(choice) & state.fingerprints
    }
    if not offending:
        return response, False
    filtered = copy.deepcopy(response)
    for index in offending:
        choice = filtered["choices"][index]
        choice["message"] = {"role": "assistant", "content": LOOP_GUARD_MESSAGE}
        choice["finish_reason"] = "stop"
    return filtered, True


def _parse_sse_events(wire: bytes) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    saw_done = False
    for raw_line in wire.decode("utf-8").splitlines():
        if not raw_line.startswith("data:"):
            continue
        data = raw_line[5:].strip()
        if data == "[DONE]":
            saw_done = True
            continue
        if not data:
            continue
        event = json.loads(data)
        if not isinstance(event, dict):
            raise ValueError("SSE event must be an object")
        events.append(event)
    if not saw_done:
        raise ValueError("guarded SSE response is missing [DONE]")
    return events


def _stream_choice_fingerprints(
    events: list[dict[str, Any]],
) -> dict[int, set[str]]:
    assembled: dict[int, dict[int, dict[str, str]]] = {}
    for event in events:
        choices = event.get("choices")
        if not isinstance(choices, list):
            continue
        for choice in choices:
            if not isinstance(choice, dict) or type(choice.get("index")) is not int:
                continue
            choice_index = choice["index"]
            delta = choice.get("delta")
            if not isinstance(delta, dict):
                continue
            calls = delta.get("tool_calls")
            if not isinstance(calls, list):
                continue
            for position, call in enumerate(calls):
                if not isinstance(call, dict):
                    continue
                call_index = call.get("index", position)
                if type(call_index) is not int:
                    raise ValueError("streamed tool call index must be an integer")
                assembled_call = assembled.setdefault(choice_index, {}).setdefault(
                    call_index, {"name": "", "arguments": ""}
                )
                function = call.get("function")
                if isinstance(function, dict):
                    name = function.get("name")
                    arguments = function.get("arguments")
                    if isinstance(name, str):
                        assembled_call["name"] += name
                    if isinstance(arguments, str):
                        assembled_call["arguments"] += arguments
    return {
        choice_index: {
            function_fingerprint(function)
            for _, function in sorted(calls.items())
        }
        for choice_index, calls in assembled.items()
    }


def sanitize_sse_response(
    wire: bytes, state: GuardState
) -> tuple[bytes, bool]:
    if not state.armed:
        return wire, False
    events = _parse_sse_events(wire)
    fingerprints = _stream_choice_fingerprints(events)
    offending = {
        index
        for index, values in fingerprints.items()
        if values & state.fingerprints
    }
    if not offending:
        return wire, False

    template = next((event for event in events if event.get("choices")), {})
    synthetic = {
        key: value
        for key, value in template.items()
        if key not in {"choices", "usage"}
    }
    synthetic["choices"] = [
        {
            "index": index,
            "delta": {"role": "assistant", "content": LOOP_GUARD_MESSAGE},
            "logprobs": None,
            "finish_reason": "stop",
        }
        for index in sorted(offending)
    ]

    outgoing: list[dict[str, Any]] = []
    inserted = False
    for event in events:
        choices = event.get("choices")
        if isinstance(choices, list):
            kept = [
                choice
                for choice in choices
                if not (
                    isinstance(choice, dict)
                    and choice.get("index") in offending
                )
            ]
            if kept:
                copied = dict(event)
                copied["choices"] = kept
                outgoing.append(copied)
            elif not choices and event.get("usage") is not None:
                if not inserted:
                    outgoing.append(synthetic)
                    inserted = True
                outgoing.append(event)
        else:
            outgoing.append(event)
    if not inserted:
        outgoing.append(synthetic)
    encoded = "".join(
        "data: " + json.dumps(event, ensure_ascii=False, separators=(",", ":")) + "\n\n"
        for event in outgoing
    )
    return (encoded + "data: [DONE]\n\n").encode(), True


def encode_sse_response(response: dict[str, Any]) -> bytes:
    chunks: list[dict[str, Any]] = []
    choices = response.get("choices")
    if not isinstance(choices, list):
        raise ValueError("upstream response has no choices")
    for choice in choices:
        if not isinstance(choice, dict):
            continue
        message = choice.get("message")
        if not isinstance(message, dict):
            message = {}
        delta = {key: value for key, value in message.items() if value is not None}
        chunks.append(
            {
                "index": choice.get("index", 0),
                "delta": delta,
                "finish_reason": choice.get("finish_reason"),
            }
        )
    event = {
        "id": response.get("id"),
        "object": "chat.completion.chunk",
        "created": response.get("created"),
        "model": response.get("model"),
        "choices": chunks,
    }
    events = [event]
    if response.get("usage") is not None:
        events.append(
            {
                "id": response.get("id"),
                "object": "chat.completion.chunk",
                "created": response.get("created"),
                "model": response.get("model"),
                "choices": [],
                "usage": response["usage"],
            }
        )
    wire = "".join(
        "data: " + json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n\n"
        for item in events
    )
    return (wire + "data: [DONE]\n\n").encode()


class GuardingProxyHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "GLMToolLoopGuard/1.0"

    @property
    def upstream(self):
        return self.server.upstream  # type: ignore[attr-defined]

    def setup(self) -> None:
        super().setup()
        self.connection.settimeout(CLIENT_READ_TIMEOUT)
        self._response_started = False

    def log_message(self, format: str, *args: Any) -> None:
        status = args[1] if len(args) > 1 else "-"
        LOG.info(
            "client=%s method=%s path=%s status=%s",
            self.client_address[0],
            self.command,
            urlsplit(self.path).path,
            status,
        )

    def _request_headers(self, body_length: int | None = None) -> dict[str, str]:
        nominated = {
            token.strip().lower()
            for value in self.headers.get_all("Connection", [])
            for token in value.split(",")
            if token.strip()
        }
        headers = {
            key: value
            for key, value in self.headers.items()
            if key.lower()
            not in HOP_BY_HOP
            | nominated
            | {"host", "content-length", "accept-encoding"}
        }
        headers["Accept-Encoding"] = "identity"
        if body_length is not None:
            headers["Content-Length"] = str(body_length)
        return headers

    def _connection(self) -> http.client.HTTPConnection:
        return http.client.HTTPConnection(
            self.upstream.hostname,
            self.upstream.port,
            timeout=self.server.upstream_timeout,  # type: ignore[attr-defined]
        )

    def _send_headers(
        self,
        status: int,
        headers: list[tuple[str, str]],
        *,
        content_length: int | None = None,
        content_type: str | None = None,
        blocked: bool = False,
    ) -> None:
        self._response_started = True
        self.send_response(status)
        if content_type is None:
            upstream_content_types = [
                value for key, value in headers if key.lower() == "content-type"
            ]
            if upstream_content_types:
                content_type = upstream_content_types[-1]
        nominated = {
            token.strip().lower()
            for key, value in headers
            if key.lower() == "connection"
            for token in value.split(",")
            if token.strip()
        }
        for key, value in headers:
            lower = key.lower()
            if lower in HOP_BY_HOP | nominated | {
                "content-length",
                "content-type",
                "server",
                "date",
            }:
                continue
            self.send_header(key, value)
        if content_type is not None:
            self.send_header("Content-Type", content_type)
        if content_length is not None:
            self.send_header("Content-Length", str(content_length))
        else:
            self.send_header("Connection", "close")
            self.close_connection = True
        if blocked:
            self.send_header("X-GLM-Loop-Guard", "blocked")
        self.end_headers()

    def _send_upstream_error(self, error: Exception) -> None:
        if self._response_started:
            self.close_connection = True
            LOG.warning("upstream stream interrupted: %s", type(error).__name__)
            return
        self.send_error(502, f"upstream unavailable: {type(error).__name__}")

    @staticmethod
    def _read_limited(upstream: http.client.HTTPResponse) -> bytes:
        body = upstream.read(MAX_GUARDED_RESPONSE_BYTES + 1)
        if len(body) > MAX_GUARDED_RESPONSE_BYTES:
            raise ValueError("guarded upstream response too large")
        return body

    def _proxy_without_body(self) -> None:
        connection = self._connection()
        try:
            connection.request(self.command, self.path, headers=self._request_headers())
            upstream = connection.getresponse()
            self._send_headers(upstream.status, upstream.getheaders())
            while chunk := upstream.read(64 * 1024):
                self.wfile.write(chunk)
        except (OSError, http.client.HTTPException) as error:
            self._send_upstream_error(error)
        finally:
            connection.close()

    def do_GET(self) -> None:  # noqa: N802
        self._proxy_without_body()

    def do_HEAD(self) -> None:  # noqa: N802
        self._proxy_without_body()

    def do_POST(self) -> None:  # noqa: N802
        raw_length = self.headers.get("Content-Length")
        if raw_length is None:
            self.send_error(411, "Content-Length required")
            return
        try:
            length = int(raw_length)
        except ValueError:
            self.send_error(400, "invalid Content-Length")
            return
        if length < 0 or length > MAX_REQUEST_BYTES:
            self.send_error(413, "request body too large")
            return
        try:
            body = self.rfile.read(length)
        except (OSError, TimeoutError):
            self.send_error(408, "request body timeout")
            return
        if len(body) != length:
            self.send_error(400, "incomplete request body")
            return
        state = GuardState(False, frozenset(), False)
        request_path = urlsplit(self.path).path.rstrip("/")
        if request_path.endswith("/v1/chat/completions"):
            try:
                payload = json.loads(body)
                if not isinstance(payload, dict):
                    raise ValueError("request body must be an object")
                payload, state = protect_payload(payload)
                body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode()
            except (json.JSONDecodeError, ValueError) as error:
                self.send_error(400, f"invalid chat completion request: {error}")
                return

        connection = self._connection()
        try:
            connection.request(
                "POST",
                self.path,
                body=body,
                headers=self._request_headers(len(body)),
            )
            upstream = connection.getresponse()
            if not state.armed or upstream.status < 200 or upstream.status >= 300:
                self._send_headers(upstream.status, upstream.getheaders())
                while chunk := upstream.read(64 * 1024):
                    self.wfile.write(chunk)
                return

            response_body = self._read_limited(upstream)
            try:
                if state.client_requested_stream:
                    outgoing, blocked = sanitize_sse_response(response_body, state)
                    content_type = "text/event-stream; charset=utf-8"
                else:
                    response = json.loads(response_body)
                    if not isinstance(response, dict):
                        raise ValueError("response body must be an object")
                    response, blocked = sanitize_response(response, state)
                    outgoing = json.dumps(
                        response, ensure_ascii=False, separators=(",", ":")
                    ).encode()
                    content_type = "application/json"
                if blocked:
                    signature = hashlib.sha256(
                        "\n".join(sorted(state.fingerprints)).encode()
                    ).hexdigest()[:16]
                    LOG.warning(
                        "loop_guard_blocked signature=%s repeated_failures=2",
                        signature,
                    )
            except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
                LOG.error("guarded upstream response invalid: %s", error)
                self.send_error(502, "invalid guarded upstream response")
                return
            self._send_headers(
                upstream.status,
                upstream.getheaders(),
                content_length=len(outgoing),
                content_type=content_type,
                blocked=blocked,
            )
            self.wfile.write(outgoing)
        except (OSError, http.client.HTTPException, ValueError) as error:
            self._send_upstream_error(error)
        finally:
            connection.close()


class GuardingProxyServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True
    request_queue_size = 64

    def __init__(
        self,
        address: tuple[str, int],
        upstream: str,
        upstream_timeout: float,
    ):
        parsed = urlsplit(upstream)
        if parsed.scheme != "http" or parsed.hostname is None or parsed.path not in ("", "/"):
            raise ValueError("upstream must be an HTTP origin without path, query, or fragment")
        if parsed.query or parsed.fragment or parsed.username or parsed.password:
            raise ValueError("upstream must not contain credentials, query, or fragment")
        try:
            is_loopback = ipaddress.ip_address(parsed.hostname).is_loopback
        except ValueError:
            try:
                resolved = socket.getaddrinfo(
                    parsed.hostname,
                    parsed.port or 80,
                    type=socket.SOCK_STREAM,
                )
            except socket.gaierror:
                resolved = []
            is_loopback = bool(resolved) and all(
                ipaddress.ip_address(item[4][0]).is_loopback for item in resolved
            )
        if not is_loopback:
            raise ValueError("upstream must resolve explicitly to loopback")
        self.upstream = parsed
        self.upstream_timeout = upstream_timeout
        self._request_slots = threading.BoundedSemaphore(MAX_CONCURRENT_REQUESTS)
        super().__init__(address, GuardingProxyHandler)

    def process_request(self, request: Any, client_address: Any) -> None:
        if not self._request_slots.acquire(blocking=False):
            try:
                request.sendall(
                    b"HTTP/1.1 503 Service Unavailable\r\n"
                    b"Connection: close\r\nContent-Length: 0\r\n\r\n"
                )
            finally:
                self.shutdown_request(request)
            return
        try:
            super().process_request(request, client_address)
        except Exception:
            self._request_slots.release()
            raise

    def process_request_thread(self, request: Any, client_address: Any) -> None:
        try:
            super().process_request_thread(request, client_address)
        finally:
            self._request_slots.release()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bind", required=True)
    parser.add_argument("--port", required=True, type=int)
    parser.add_argument("--upstream", required=True)
    parser.add_argument("--upstream-timeout", type=float, default=3600)
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535:
        parser.error("port must be 1..65535")
    if args.upstream_timeout <= 0:
        parser.error("upstream-timeout must be positive")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    server = GuardingProxyServer(
        (args.bind, args.port), args.upstream, args.upstream_timeout
    )
    try:
        bind_is_loopback = ipaddress.ip_address(args.bind).is_loopback
    except ValueError:
        bind_is_loopback = args.bind == "localhost"
    if not bind_is_loopback:
        LOG.warning(
            "non-loopback bind has no built-in auth/TLS; restrict port %d to trusted LAN/Tailnet clients",
            args.port,
        )
    LOG.info(
        "ready bind=%s port=%d upstream=%s baseline_penalty=%.2f armed_penalty=%.2f",
        args.bind,
        args.port,
        args.upstream,
        BASELINE_REPETITION_PENALTY,
        ARMED_REPETITION_PENALTY,
    )
    # Background jobs started by non-interactive Bash inherit SIGINT as ignored.
    # Restore Python's handler so supervisor cleanup can terminate the proxy.
    signal.signal(signal.SIGINT, signal.default_int_handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
