#!/usr/bin/env python3
"""Measure active decode while a distinct long prefill overlaps it."""
import argparse
import hashlib
import importlib.util
import json
import pathlib
import threading
import time
import urllib.request
from typing import Any


def require_distinct_prefixes(decoder_fixture: bytes, prefill_fixture: bytes) -> None:
    if hashlib.sha256(decoder_fixture).digest() == hashlib.sha256(prefill_fixture).digest():
        raise ValueError("mixed benchmark prefixes must be distinct")


def analyze_timeline(events, a_completion_tokens: int) -> dict[str, Any]:
    timeline: dict[str, list[float]] = {}
    for name, timestamp in events:
        timeline.setdefault(name, []).append(float(timestamp))
    required = ("a_submitted", "a_token", "b_submitted")
    if any(not timeline.get(name) for name in required):
        raise ValueError("incomplete concurrency timeline")
    a_start = timeline["a_submitted"][0]
    first_a = timeline["a_token"][0]
    b_start = timeline["b_submitted"][0]
    if b_start < first_a:
        raise ValueError("decoder A must reach first token before prefill B starts")
    if not timeline.get("b_first_token"):
        raise ValueError("incomplete concurrency timeline")
    overlap = [stamp for stamp in timeline["a_token"] if stamp >= b_start]
    if timeline.get("a_finished"):
        overlap = [stamp for stamp in overlap if stamp <= timeline["a_finished"][-1]]
    overlap_duration = round(overlap[-1] - overlap[0], 15) if len(overlap) >= 2 else None
    a_finish = timeline.get("a_finished", [timeline["a_token"][-1]])[-1]
    b_finish = timeline.get("b_finished", [timeline["b_first_token"][-1]])[-1]
    active_decode_duration = a_finish - first_a
    active_decode_rate = (
        (a_completion_tokens - 1) / active_decode_duration
        if a_completion_tokens > 1 and active_decode_duration > 0
        else None
    )
    return {
        "a_ttft_seconds": first_a - a_start,
        "a_wall_seconds": a_finish - a_start,
        "a_completion_tokens": a_completion_tokens,
        "a_overlap_token_count": len(overlap),
        "a_overlap_itl_seconds": overlap_duration / (len(overlap) - 1) if overlap_duration is not None else None,
        "a_overlap_tokens_per_second": (len(overlap) - 1) / overlap_duration if overlap_duration and overlap_duration > 0 else None,
        "a_active_decode_tokens_per_second": active_decode_rate,
        "combined_makespan_seconds": max(a_finish, b_finish) - a_start,
        "b_ttft_seconds": timeline["b_first_token"][0] - b_start,
        "b_queue_seconds": timeline["b_first_token"][0] - b_start,
        "b_wall_seconds": b_finish - b_start,
        "ordering_valid": True,
    }


def _load_decode():
    spec = importlib.util.spec_from_file_location("strict_decode", pathlib.Path(__file__).with_name("bench-decode.py"))
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load strict stream parser")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _stream(base: str, payload: dict[str, Any], label: str, events: list, lock: threading.Lock,
            first_token_event: threading.Event | None = None):
    request = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    with lock:
        events.append((f"{label}_submitted", time.perf_counter()))
    chunks, buffer, first = [], b"", True
    with urllib.request.urlopen(request, timeout=900) as response:
        reader = getattr(response, "read1", response.read)
        while True:
            chunk = reader(4096)
            if not chunk:
                break
            chunks.append(chunk)
            buffer += chunk
            normalized = buffer.replace(b"\r\n", b"\n")
            parts = normalized.split(b"\n\n")
            buffer = parts.pop()
            for part in parts:
                for line in part.splitlines():
                    if not line.startswith(b"data: ") or line[6:] == b"[DONE]":
                        continue
                    try:
                        obj = json.loads(line[6:])
                        delta = obj.get("choices", [{}])[0].get("delta", {})
                    except (json.JSONDecodeError, IndexError, AttributeError):
                        continue
                    if any(delta.get(field) for field in ("content", "reasoning", "reasoning_content")):
                        now = time.perf_counter()
                        with lock:
                            events.append((f"{label}_token" if label == "a" else f"{label}_first_token", now))
                        if first_token_event is not None and first:
                            first_token_event.set()
                        first = False
    with lock:
        events.append((f"{label}_finished", time.perf_counter()))
    return _load_decode().parse_sse(chunks)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="overlord-testing")
    parser.add_argument("--decoder-fixture", required=True)
    parser.add_argument("--prefill-fixture", required=True)
    parser.add_argument("--decoder-max-tokens", type=int, default=1024)
    parser.add_argument("--prefill-max-tokens", type=int, default=1)
    args = parser.parse_args(argv)
    a_bytes = pathlib.Path(args.decoder_fixture).read_bytes()
    b_bytes = pathlib.Path(args.prefill_fixture).read_bytes()
    require_distinct_prefixes(a_bytes, b_bytes)
    base = args.base_url.rstrip("/")
    common = {"model": args.model, "temperature": 0, "stream": True, "stream_options": {"include_usage": True}}
    payload_a = {**common, "messages": [{"role": "user", "content": a_bytes.decode()}], "max_tokens": args.decoder_max_tokens}
    payload_b = {**common, "messages": [{"role": "user", "content": b_bytes.decode()}], "max_tokens": args.prefill_max_tokens}
    events, lock, first_a, results = [], threading.Lock(), threading.Event(), {}
    def run_a(): results["a"] = _stream(base, payload_a, "a", events, lock, first_a)
    def run_b(): results["b"] = _stream(base, payload_b, "b", events, lock)
    thread_a = threading.Thread(target=run_a)
    thread_a.start()
    if not first_a.wait(timeout=300):
        raise RuntimeError("decoder A did not reach first token")
    thread_b = threading.Thread(target=run_b)
    thread_b.start()
    thread_a.join(); thread_b.join()
    receipt = analyze_timeline(events, results["a"]["usage"]["completion_tokens"])
    receipt.update({"schema_version": 1, "decoder_fixture_sha256": hashlib.sha256(a_bytes).hexdigest(),
                    "prefill_fixture_sha256": hashlib.sha256(b_bytes).hexdigest(),
                    "b_prompt_tokens": results["b"]["usage"].get("prompt_tokens")})
    print(json.dumps(receipt, sort_keys=True, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
