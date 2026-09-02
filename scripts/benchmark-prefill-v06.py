#!/usr/bin/env python3
"""Measure exact-length prefill. Adapted from tpurtell v0.6 (Apache-2.0)."""

from __future__ import annotations

import argparse
import http.client
import json
import statistics
import subprocess
import time
import urllib.request
from pathlib import Path
from urllib.parse import urlparse


def verify_container_profile(container: str, model: str) -> dict:
    result = subprocess.run(["docker", "inspect", container], text=True, capture_output=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"cannot inspect benchmark container {container!r}")
    document = json.loads(result.stdout)
    command = document[0].get("Config", {}).get("Cmd") if isinstance(document, list) and len(document) == 1 else None
    if not isinstance(command, list) or not all(isinstance(item, str) for item in command):
        raise RuntimeError("container has no inspectable command")
    def option(name: str) -> str:
        positions = [index for index, value in enumerate(command) if value == name]
        if len(positions) != 1 or positions[0] + 1 >= len(command):
            raise RuntimeError(f"container command has invalid {name}")
        return command[positions[0] + 1]
    observed = {
        "container": container,
        "served_model_name": option("--served-model-name"),
        "max_model_len": option("--max-model-len"),
        "kv_cache_dtype": option("--kv-cache-dtype"),
    }
    if observed["served_model_name"] != model or observed["max_model_len"] != "1048576" or observed["kv_cache_dtype"] != "fp8_ds_mla":
        raise RuntimeError(f"container profile mismatch: {observed!r}")
    return observed


def post_json(base_url: str, path: str, payload: dict, timeout: float = 3600) -> dict:
    server_url = base_url.rstrip("/").removesuffix("/v1")
    request = urllib.request.Request(
        server_url + path,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def exact_prompt(base_url: str, model: str, target_tokens: int, nonce: str) -> str:
    """Build a prompt with an exact round-trip token count and unique first block."""
    header = (
        f"Independent prefill qualification {nonce}. This nonce makes the first "
        "cache block unique; ignore it.\n"
    )
    unit = (
        "Slate rivers cross quiet valleys while copper clocks mark patient hours. "
        "This is ordinary benchmark filler with no instructions.\n"
    )

    def tokenize(text: str) -> list[int]:
        return post_json(
            base_url, "/tokenize", {"model": model, "prompt": text}, 300
        )["tokens"]

    def detokenize(tokens: list[int]) -> str:
        return post_json(
            base_url, "/detokenize", {"model": model, "tokens": tokens}, 300
        )["prompt"]

    header_ids = tokenize(header)
    if target_tokens <= len(header_ids):
        raise ValueError(f"target {target_tokens} is too small")
    unit_ids = tokenize(unit)
    source_ids = tokenize(
        header + unit * (target_tokens // max(1, len(unit_ids)) + 3)
    )
    wanted = target_tokens
    for _ in range(8):
        prompt = detokenize(source_ids[:wanted])
        actual = len(tokenize(prompt))
        if actual == target_tokens:
            return prompt
        wanted += target_tokens - actual
        if wanted <= len(header_ids) or wanted > len(source_ids):
            break
    raise RuntimeError(f"could not construct an exact {target_tokens}-token prompt")


def time_to_first_token(base_url: str, model: str, prompt: str) -> dict:
    parsed = urlparse(base_url)
    if parsed.hostname is None:
        raise ValueError(f"base URL has no host: {base_url!r}")
    connection_type = (
        http.client.HTTPSConnection
        if parsed.scheme == "https"
        else http.client.HTTPConnection
    )
    connection = connection_type(parsed.hostname, parsed.port, timeout=3600)
    path = parsed.path.rstrip("/") + "/completions"
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": 1,
        "min_tokens": 1,
        "ignore_eos": True,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
        "return_token_ids": True,
        "cache_prompt": False,
    }
    started = time.perf_counter()
    connection.request(
        "POST",
        path,
        body=json.dumps(payload),
        headers={"Content-Type": "application/json"},
    )
    response = connection.getresponse()
    if response.status != 200:
        error = response.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {response.status}: {error}")
    first_token = None
    usage = None
    saw_done = False
    finish_reason = None
    while True:
        raw_line = response.readline()
        if not raw_line:
            break
        observed = time.perf_counter()
        line = raw_line.decode("utf-8", errors="replace").strip()
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if not data:
            continue
        if data == "[DONE]":
            saw_done = True
            break
        event = json.loads(data)
        if isinstance(event.get("usage"), dict):
            usage = event["usage"]
        for choice in event.get("choices", []):
            if choice.get("finish_reason") is not None:
                finish_reason = choice["finish_reason"]
            token_ids = choice.get("token_ids")
            if first_token is None and isinstance(token_ids, list) and token_ids:
                first_token = observed
    connection.close()
    if first_token is None:
        raise RuntimeError("stream returned no token IDs")
    if not saw_done or finish_reason != "length":
        raise RuntimeError(f"incomplete stream: done={saw_done}, finish={finish_reason!r}")
    prompt_tokens = usage.get("prompt_tokens") if isinstance(usage, dict) else None
    if isinstance(prompt_tokens, bool) or not isinstance(prompt_tokens, int) or prompt_tokens <= 0:
        raise RuntimeError(f"invalid prompt token usage: {prompt_tokens!r}")
    return {
        "prompt_tokens": prompt_tokens,
        "ttft_seconds": first_token - started,
        "effective_prompt_tokens_per_second": prompt_tokens / (first_token - started),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", required=True)
    parser.add_argument("--container", default="glm53-flash-2x-rtxpro6000")
    parser.add_argument("--profile", required=True, choices=["fp8"])
    parser.add_argument(
        "--prompt-tokens",
        nargs="+",
        type=int,
        default=[2048, 8192, 32768, 65536, 128000],
    )
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    verified_profile = verify_container_profile(args.container, args.model)

    warmup = exact_prompt(args.base_url, args.model, 512, f"{args.profile}-warmup")
    time_to_first_token(args.base_url, args.model, warmup)

    points = []
    for target in args.prompt_tokens:
        depth_warmup = exact_prompt(
            args.base_url, args.model, target, f"{args.profile}-{target}-warmup"
        )
        time_to_first_token(args.base_url, args.model, depth_warmup)
        runs = []
        for run in range(args.runs):
            prompt = exact_prompt(
                args.base_url, args.model, target, f"{args.profile}-{target}-{run}"
            )
            result = time_to_first_token(args.base_url, args.model, prompt)
            if result["prompt_tokens"] != target:
                raise RuntimeError(
                    f"server counted {result['prompt_tokens']} prompt tokens, expected {target}"
                )
            runs.append(result)
            print(
                f"{args.profile} {target:,} run {run + 1}/{args.runs}: "
                f"{result['effective_prompt_tokens_per_second']:.1f} tok/s, "
                f"TTFT {result['ttft_seconds']:.3f} s",
                flush=True,
            )
        rates = [run["effective_prompt_tokens_per_second"] for run in runs]
        ttfts = [run["ttft_seconds"] for run in runs]
        points.append(
            {
                "prompt_tokens": target,
                "effective_prompt_tokens_per_second": {
                    "median": statistics.median(rates),
                    "min": min(rates),
                    "max": max(rates),
                },
                "ttft_seconds": {
                    "median": statistics.median(ttfts),
                    "min": min(ttfts),
                    "max": max(ttfts),
                },
                "runs": runs,
            }
        )

    report = {
        "schema": "glm53-prefill-depth.v1",
        "method": (
            "C1 exact-length unique prompts; client request to first streamed token; "
            "server tokenization and one-token handoff included; no prefix reuse"
        ),
        "model": args.model,
        "kv_cache_profile": args.profile,
        "verified_runtime_profile": verified_profile,
        "runs_per_point": args.runs,
        "points": points,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
