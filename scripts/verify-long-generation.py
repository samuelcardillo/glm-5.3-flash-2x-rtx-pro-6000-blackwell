#!/usr/bin/env python3
"""Bounded strict-SSE probe for generations exceeding 2,048 tokens."""
import argparse
import hashlib
import importlib.util
import json
import pathlib
import threading
import urllib.error
import urllib.request
from typing import Any, Sequence


def detect_repetition(tokens: Sequence[Any], min_period: int = 8, max_period: int = 256, repeats: int = 4):
    if repeats < 2 or min_period < 1 or max_period < min_period:
        raise ValueError("invalid repetition bounds")
    upper = min(max_period, len(tokens) // repeats)
    for period in range(min_period, upper + 1):
        matched = 0
        needed = period * (repeats - 1)
        for index in range(period, len(tokens)):
            if tokens[index] == tokens[index - period]:
                matched += 1
                if matched >= needed:
                    return {"start": index + 1 - period * repeats, "period": period, "repeats": repeats}
            else:
                matched = 0
    return None


def validate_long_result(parsed: dict[str, Any], requested_max_tokens: int, health_after: bool) -> None:
    if requested_max_tokens <= 2048:
        raise ValueError("probe must request more than 2,048 output tokens")
    completion = parsed.get("usage", {}).get("completion_tokens")
    if not isinstance(completion, int) or completion <= 2048:
        raise ValueError("generation did not exceed 2,048 output tokens")
    if not parsed.get("finish_reason"):
        raise ValueError("generation has no finish reason")
    if not health_after:
        raise ValueError("service unhealthy after long generation")


def require_distinct(first: bytes, second: bytes) -> None:
    if hashlib.sha256(first).digest() == hashlib.sha256(second).digest():
        raise ValueError("concurrent prefix must be distinct")


def _load_decode():
    spec = importlib.util.spec_from_file_location("strict_decode", pathlib.Path(__file__).with_name("bench-decode.py"))
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load strict SSE parser")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _health(base: str) -> bool:
    try:
        with urllib.request.urlopen(base + "/health", timeout=30):
            return True
    except (OSError, urllib.error.HTTPError):
        return False


def _payload(model: str, prompt: str, max_tokens: int):
    return {"model": model, "messages": [{"role": "user", "content": prompt}], "temperature": 0,
            "max_tokens": max_tokens, "stream": True, "stream_options": {"include_usage": True}}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="overlord-testing")
    parser.add_argument("--fixture", required=True)
    parser.add_argument("--max-tokens", type=int, default=2304)
    parser.add_argument("--concurrent-prefix-file")
    parser.add_argument("--min-period", type=int, default=8)
    parser.add_argument("--max-period", type=int, default=256)
    parser.add_argument("--repeats", type=int, default=4)
    args = parser.parse_args(argv)
    if args.max_tokens <= 2048:
        raise ValueError("--max-tokens must exceed 2048")
    fixture = pathlib.Path(args.fixture).read_bytes()
    base = args.base_url.rstrip("/")
    decode = _load_decode()
    long_payload = _payload(args.model, fixture.decode(), args.max_tokens)
    concurrent_receipt = None
    if args.concurrent_prefix_file:
        other = pathlib.Path(args.concurrent_prefix_file).read_bytes()
        require_distinct(fixture, other)
        results = {}
        def long_run(): results["long"] = decode._run(base, long_payload)
        def prefix_run(): results["prefix"] = decode._run(base, _payload(args.model, other.decode(), 1))
        long_thread = threading.Thread(target=long_run)
        prefix_thread = threading.Thread(target=prefix_run)
        long_thread.start(); prefix_thread.start(); long_thread.join(); prefix_thread.join()
        started, first, finished, parsed = results["long"]
        p_started, p_first, p_finished, p_parsed = results["prefix"]
        concurrent_receipt = {"fixture_sha256": hashlib.sha256(other).hexdigest(),
                              "ttft_seconds": p_first - p_started, "wall_seconds": p_finished - p_started,
                              "prompt_tokens": p_parsed["usage"].get("prompt_tokens")}
    else:
        started, first, finished, parsed = decode._run(base, long_payload)
    healthy = _health(base)
    validate_long_result(parsed, args.max_tokens, healthy)
    findings = {}
    channels = {}
    for field in ("content", "reasoning", "reasoning_content"):
        text = parsed[field]
        tokens = text.split()
        findings[field] = detect_repetition(tokens, args.min_period, args.max_period, args.repeats)
        channels[field] = {"sha256": hashlib.sha256(text.encode()).hexdigest(), "whitespace_token_count": len(tokens)}
    if any(finding is not None for finding in findings.values()):
        raise RuntimeError("bounded repetition detector found a periodic loop")
    receipt = {
        "schema_version": 1, "fixture_sha256": hashlib.sha256(fixture).hexdigest(),
        "requested_max_tokens": args.max_tokens, "completion_tokens": parsed["usage"]["completion_tokens"],
        "finish_reason": parsed["finish_reason"], "ttft_seconds": first - started,
        "wall_seconds": finished - started, "health_after": healthy, "channels": channels,
        "repetition_bounds": {"min_period": args.min_period, "max_period": args.max_period, "repeats": args.repeats},
        "repetition_findings": findings, "concurrent_prefix": concurrent_receipt,
        "scope_note": "A bounded finite probe is not a guarantee that autoregressive repetition is impossible.",
    }
    print(json.dumps(receipt, sort_keys=True, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
