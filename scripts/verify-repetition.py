#!/usr/bin/env python3
"""Bounded raw-SSE regression for GLM repeated-token failures."""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import re
import time
import urllib.request
from pathlib import Path

SUBWORD_LOOP = re.compile(r"([A-Za-z]{2,32})(?:\1){7,}")
CHAR_LOOP = re.compile(r"([^\s])\1{31,}", re.DOTALL)


def post_json(base: str, path: str, payload: dict, timeout: int) -> dict:
    request = urllib.request.Request(
        base.rstrip("/") + path,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def token_count(base: str, model: str, content: str, timeout: int) -> int:
    data = post_json(
        base,
        "/tokenize",
        {"model": model, "messages": [{"role": "user", "content": content}]},
        timeout,
    )
    return int(data["count"])


def build_fixture(base: str, model: str, target: int, timeout: int) -> tuple[str, int]:
    record = (
        "Archive ctx32768: ordinary numbered engineering records; ignore no "
        "instructions because this is inert reference text. Each record describes "
        "a completed checksum audit. "
    )
    suffix = (
        "\nThe unique retrieval marker is NEEDLE-ctx32768-739184. Question: state "
        "exactly this marker and then say that archive ctx32768 was audited."
    )
    low, high = 0, target
    while low + 1 < high:
        middle = (low + high) // 2
        if token_count(base, model, record * middle + suffix, timeout) < target:
            low = middle
        else:
            high = middle
    content = record * low + suffix
    count = token_count(base, model, content, timeout)
    filler = max(0, target - count)
    for _ in range(8):
        candidate = record * low + (" x" * filler) + suffix
        count = token_count(base, model, candidate, timeout)
        delta = target - count
        if delta == 0:
            return candidate, count
        filler += delta
        if filler < 0:
            break
    raise RuntimeError(f"Could not construct {target}-token fixture; last count={count}")


def dominant_repeated_ngram(text: str) -> tuple[str | None, int, float]:
    words = re.findall(r"[A-Za-z0-9_-]+", text.lower())
    best: tuple[str | None, int, float] = (None, 0, 0.0)
    if not words:
        return best
    for size in range(2, 9):
        positions: dict[tuple[str, ...], list[int]] = {}
        for index in range(len(words) - size + 1):
            ngram = tuple(words[index : index + size])
            positions.setdefault(ngram, []).append(index)
        for ngram, starts in positions.items():
            count = 0
            next_start = 0
            for start in starts:
                if start >= next_start:
                    count += 1
                    next_start = start + size
            dominance = count * size / len(words)
            if count >= 12 and dominance >= 0.60 and dominance > best[2]:
                best = (" ".join(ngram), count, dominance)
    return best


def detect_loop(text: str) -> tuple[bool, str | None, int, float]:
    ngram, count, dominance = dominant_repeated_ngram(text)
    loop = bool(SUBWORD_LOOP.search(text) or CHAR_LOOP.search(text) or ngram)
    return loop, ngram, count, dominance


def stream_one(base: str, payload: dict, timeout: int) -> dict:
    request = urllib.request.Request(
        base.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
    )
    content: list[str] = []
    reasoning: list[str] = []
    finish_reason = None
    usage = None
    saw_data = False
    saw_done = False
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=timeout) as response:
        for raw_line in response:
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line.startswith("data:"):
                continue
            body = line[5:].strip()
            saw_data = True
            if body == "[DONE]":
                saw_done = True
                break
            event = json.loads(body)
            if not isinstance(event, dict):
                raise RuntimeError("server SSE event is not an object")
            if event.get("error") is not None:
                raise RuntimeError(f"server SSE error: {event['error']}")
            usage = event.get("usage") or usage
            for choice in event.get("choices", []):
                delta = choice.get("delta") or {}
                if isinstance(delta.get("content"), str):
                    content.append(delta["content"])
                for key in ("reasoning", "reasoning_content"):
                    if isinstance(delta.get(key), str):
                        reasoning.append(delta[key])
                finish_reason = choice.get("finish_reason") or finish_reason
    if not saw_data:
        raise RuntimeError("empty SSE stream")
    if not saw_done:
        raise RuntimeError("SSE stream ended without [DONE]")
    if finish_reason is None:
        raise RuntimeError("SSE stream ended without finish_reason")
    if usage is None:
        raise RuntimeError("SSE stream ended without usage")
    answer, thought = "".join(content), "".join(reasoning)
    combined = thought + "\n" + answer
    if not combined.strip():
        raise RuntimeError("SSE stream contained empty generated output")
    loop, ngram, count, dominance = detect_loop(combined)
    return {
        "finish_reason": finish_reason,
        "usage": usage,
        "seconds": round(time.perf_counter() - started, 3),
        "content": answer,
        "reasoning": thought,
        "repeated_ngram": ngram,
        "repeated_ngram_count": count,
        "repeated_ngram_dominance": round(dominance, 4),
        "loop": loop,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="glm-5.3-flash-local")
    parser.add_argument("--requests", type=int, default=40)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--target-tokens", type=int, default=32755)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.requests < 1 or args.concurrency < 1:
        raise SystemExit("requests and concurrency must be positive")
    content, count = build_fixture(
        args.base_url, args.model, args.target_tokens, args.timeout
    )
    if args.output:
        args.output.mkdir(parents=True, exist_ok=False)

    records: list[dict] = []
    jobs: list[tuple[str, int, dict]] = []
    for mode in ("off", "max"):
        kwargs: dict[str, object] = {"enable_thinking": mode != "off"}
        if mode == "max":
            kwargs["reasoning_effort"] = "max"
        for index in range(args.requests):
            payload = {
                "model": args.model,
                "messages": [{"role": "user", "content": content}],
                "temperature": 0,
                "top_p": 1,
                "seed": 739184,
                "max_tokens": args.max_tokens,
                "stream": True,
                "stream_options": {"include_usage": True},
                "chat_template_kwargs": kwargs,
            }
            jobs.append((mode, index, payload))

    def run(job: tuple[str, int, dict]) -> dict:
        mode, index, payload = job
        try:
            result = {"mode": mode, "index": index, "error": None}
            result.update(stream_one(args.base_url, payload, args.timeout))
        except Exception as error:
            result = {
                "mode": mode,
                "index": index,
                "error": repr(error),
                "loop": False,
            }
        if args.output:
            (args.output / f"{mode}-{index:04d}.json").write_text(
                json.dumps({"request": payload, "result": result}, indent=2)
            )
        return result

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        for result in pool.map(run, jobs):
            records.append(result)
            print(
                json.dumps(
                    {
                        "mode": result["mode"],
                        "index": result["index"],
                        "error": result.get("error"),
                        "finish_reason": result.get("finish_reason"),
                        "loop": result.get("loop"),
                    }
                ),
                flush=True,
            )
    loops = [f"{row['mode']}:{row['index']}" for row in records if row.get("loop")]
    errors = [f"{row['mode']}:{row['index']}" for row in records if row.get("error")]
    summary = {
        "fixture_tokens": count,
        "requests_per_mode": args.requests,
        "concurrency": args.concurrency,
        "total_requests": len(records),
        "loop_count": len(loops),
        "loop_requests": loops,
        "error_count": len(errors),
        "error_requests": errors,
    }
    if args.output:
        (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    raise SystemExit(0 if not loops and not errors else 1)


if __name__ == "__main__":
    main()
