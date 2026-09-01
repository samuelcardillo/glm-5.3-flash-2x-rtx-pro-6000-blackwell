#!/usr/bin/env python3
"""Strict, reproducible OpenAI-compatible streaming decode benchmark."""
import argparse
import hashlib
import json
import math
import statistics
import time
import urllib.error
import urllib.request
from typing import Any, Iterable


class StreamError(ValueError):
    pass


def _events(chunks: Iterable[bytes]):
    buffer = b""
    for chunk in chunks:
        buffer += chunk
        while True:
            normalized = buffer.replace(b"\r\n", b"\n")
            marker = normalized.find(b"\n\n")
            if marker < 0:
                break
            event, buffer = normalized[:marker], normalized[marker + 2:]
            if event:
                yield event.decode("utf-8", errors="strict")
    if buffer.strip():
        raise StreamError("incomplete SSE event")


def parse_sse(chunks: Iterable[bytes]) -> dict[str, Any]:
    output = {"content": "", "reasoning": "", "reasoning_content": ""}
    usage = None
    finish_reason = None
    done = False
    for event in _events(chunks):
        lines = event.splitlines()
        event_type = next((line[6:].strip() for line in lines if line.startswith("event:")), "message")
        data_lines = [line[5:].lstrip() for line in lines if line.startswith("data:")]
        if not data_lines:
            raise StreamError("SSE event has no data")
        data = "\n".join(data_lines)
        if event_type == "error":
            raise StreamError("SSE error: " + data[:200])
        if data == "[DONE]":
            if done:
                raise StreamError("duplicate [DONE]")
            done = True
            continue
        if done:
            raise StreamError("data after [DONE]")
        try:
            payload = json.loads(data)
        except (json.JSONDecodeError, UnicodeError) as exc:
            raise StreamError("malformed SSE JSON") from exc
        if "error" in payload:
            raise StreamError("SSE error payload")
        choices = payload.get("choices")
        if choices:
            if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
                raise StreamError("expected exactly one choice")
            choice = choices[0]
            delta = choice.get("delta", {})
            if not isinstance(delta, dict):
                raise StreamError("invalid delta")
            for field in output:
                value = delta.get(field)
                if value is not None:
                    if not isinstance(value, str):
                        raise StreamError("non-text output delta")
                    output[field] += value
            if choice.get("finish_reason") is not None:
                finish_reason = choice["finish_reason"]
        if payload.get("usage") is not None:
            usage = payload["usage"]
    if not done:
        raise StreamError("stream ended before [DONE]")
    if finish_reason is None:
        raise StreamError("stream has no finish reason")
    if (not isinstance(usage, dict)
            or type(usage.get("prompt_tokens")) is not int
            or type(usage.get("completion_tokens")) is not int
            or usage["prompt_tokens"] < 0
            or usage["completion_tokens"] < 0):
        raise StreamError("stream has no valid usage")
    if usage["completion_tokens"] <= 0 or not any(output.values()):
        raise StreamError("empty completion")
    return {**output, "usage": usage, "finish_reason": finish_reason}


def _metric_samples(text: str) -> dict[tuple[str, tuple[tuple[str, str], ...]], float]:
    values = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        try:
            left, raw_value = line.rsplit(None, 1)
            value = float(raw_value)
        except ValueError:
            continue
        if "{" in left and left.endswith("}"):
            name, labels_text = left[:-1].split("{", 1)
            labels = []
            for part in labels_text.split(","):
                key, val = part.split("=", 1)
                labels.append((key.strip(), val.strip().strip('"')))
            label_key = tuple(sorted(labels))
        else:
            name, label_key = left, ()
        values[(name, label_key)] = value
    return values


def mtp_metric_delta(before: str, after: str) -> dict[str, Any]:
    old, new = _metric_samples(before), _metric_samples(after)

    def delta_name(name: str) -> float:
        return sum(
            value - old.get((metric_name, labels), 0.0)
            for (metric_name, labels), value in new.items()
            if metric_name == name
        )

    draft_steps = delta_name("vllm:spec_decode_num_drafts_total")
    draft_tokens = delta_name("vllm:spec_decode_num_draft_tokens_total")
    accepted = delta_name("vllm:spec_decode_num_accepted_tokens_total")
    positions = {}
    for (name, labels), value in new.items():
        if name != "vllm:spec_decode_num_accepted_tokens_per_pos_total":
            continue
        position = dict(labels).get("position")
        if position is not None:
            positions[position] = (
                (value - old.get((name, labels), 0.0)) / draft_steps
                if draft_steps > 0
                else None
            )
    return {
        "draft_steps": draft_steps,
        "draft_tokens": draft_tokens,
        "accepted_tokens": accepted,
        "acceptance_rate": accepted / draft_tokens if draft_tokens > 0 else None,
        "accepted_tokens_per_step": accepted / draft_steps if draft_steps > 0 else None,
        "per_position_acceptance": dict(
            sorted(positions.items(), key=lambda item: int(item[0]))
        ),
    }


def make_result(*, fixture_id: str, fixture: bytes, health_before: bool, health_after: bool,
                warmups: int, run_index: int, started: float, first_token: float,
                finished: float, parsed: dict[str, Any], mtp: dict[str, Any]) -> dict[str, Any]:
    completion = parsed["usage"]["completion_tokens"]
    decode_duration = max(0.0, finished - first_token)
    return {
        "fixture_id": fixture_id,
        "fixture_sha256": hashlib.sha256(fixture).hexdigest(),
        "health_before": health_before,
        "health_after": health_after,
        "warmup_count": warmups,
        "run_index": run_index,
        "ttft_seconds": first_token - started,
        "decode_seconds": decode_duration,
        "decode_tokens_per_second": max(0, completion - 1) / decode_duration if decode_duration > 0 else None,
        "wall_seconds": finished - started,
        "prompt_tokens": parsed["usage"].get("prompt_tokens"),
        "completion_tokens": completion,
        "finish_reason": parsed["finish_reason"],
        "mtp": mtp,
        "output": {key: parsed[key] for key in ("content", "reasoning", "reasoning_content")},
    }


def _percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(quantile * len(ordered)) - 1)]


def summarize(runs: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {"run_count": len(runs)}
    for field in ("ttft_seconds", "decode_seconds", "decode_tokens_per_second", "wall_seconds"):
        values = [run[field] for run in runs if isinstance(run.get(field), (int, float))]
        result[field] = {"median": statistics.median(values), "p90": _percentile(values, 0.9)} if values else None
    return result


def _get(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=15) as response:
        return response.read()


def _health(base_url: str) -> bool:
    try:
        _get(base_url.rstrip("/") + "/health")
        return True
    except (OSError, urllib.error.HTTPError):
        return False


def iter_response_chunks(response, size: int = 4096):
    reader = getattr(response, "read1", response.read)
    while True:
        chunk = reader(size)
        if not chunk:
            return
        yield chunk


def _inspect_output_events(buffer: bytes) -> tuple[bytes, bool]:
    normalized = buffer.replace(b"\r\n", b"\n")
    parts = normalized.split(b"\n\n")
    remainder = parts.pop()
    for part in parts:
        data = b"\n".join(line[5:].lstrip() for line in part.splitlines() if line.startswith(b"data:"))
        if not data or data == b"[DONE]":
            continue
        try:
            payload = json.loads(data)
            delta = payload.get("choices", [{}])[0].get("delta", {})
        except (json.JSONDecodeError, IndexError, AttributeError):
            continue
        if any(delta.get(field) for field in ("content", "reasoning", "reasoning_content")):
            return remainder, True
    return remainder, False


def _run(base_url: str, payload: dict[str, Any]):
    request = urllib.request.Request(base_url.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    started = time.perf_counter()
    chunks, inspect_buffer, first = [], b"", None
    with urllib.request.urlopen(request, timeout=600) as response:
        for chunk in iter_response_chunks(response):
            inspect_buffer += chunk
            inspect_buffer, has_output = _inspect_output_events(inspect_buffer)
            if first is None and has_output:
                first = time.perf_counter()
            chunks.append(chunk)
    finished = time.perf_counter()
    return started, first or finished, finished, parse_sse(chunks)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="overlord-testing")
    parser.add_argument("--fixture-id", required=True)
    parser.add_argument("--fixture-file", required=True)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--max-tokens", type=int, default=512)
    args = parser.parse_args(argv)
    fixture = open(args.fixture_file, "rb").read()
    prompt = fixture.decode("utf-8")
    payload = {"model": args.model, "messages": [{"role": "user", "content": prompt}], "temperature": 0, "stream": True, "stream_options": {"include_usage": True}, "max_tokens": args.max_tokens}
    for _ in range(args.warmups):
        _run(args.base_url, payload)
    runs = []
    for index in range(args.runs):
        health_before = _health(args.base_url)
        metrics_before = _get(args.base_url.rstrip("/") + "/metrics").decode()
        started, first, finished, parsed = _run(args.base_url, payload)
        metrics_after = _get(args.base_url.rstrip("/") + "/metrics").decode()
        runs.append(make_result(fixture_id=args.fixture_id, fixture=fixture, health_before=health_before,
            health_after=_health(args.base_url), warmups=args.warmups, run_index=index, started=started,
            first_token=first, finished=finished, parsed=parsed, mtp=mtp_metric_delta(metrics_before, metrics_after)))
    print(json.dumps({"schema_version": 1, "runs": runs, "summary": summarize(runs)}, sort_keys=True, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
