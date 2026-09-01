#!/usr/bin/env python3
"""Cold-prefill and prefix-cache benchmark with exact tokenize calibration."""
import argparse
import hashlib
import importlib.util
import json
import pathlib
import urllib.error
import urllib.request
from typing import Any, Callable

PUBLIC_TARGETS = (8192, 32768, 131072, 261900)


class ResetError(RuntimeError):
    pass


def run_salt(fixture_id: str, run_index: int) -> str:
    return hashlib.sha256(f"{fixture_id}\0{run_index}".encode()).hexdigest()[:16]


def calibrate_prompt(target_tokens: int, salt: str, tokenize: Callable[[str], int]) -> tuple[str, int]:
    if target_tokens <= 0:
        raise ValueError("target_tokens must be positive")
    prefix = f"PUBLIC-SYNTHETIC-{salt}"
    base_count = tokenize(prefix)
    filler = max(0, target_tokens - base_count)
    count = base_count
    for _ in range(16):
        prompt = prefix + (" x" * filler)
        count = tokenize(prompt)
        if count == target_tokens:
            return prompt, count
        filler += target_tokens - count
        if filler < 0:
            break
    raise ValueError(f"tokenizer cannot construct exact {target_tokens}-token synthetic prompt; last count={count}")


def _metrics(text: str) -> dict[str, float]:
    result = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        try:
            key, value = line.rsplit(None, 1)
            result[key.split("{", 1)[0]] = float(value)
        except ValueError:
            continue
    return result


def cache_metric_delta(before: str, after: str, block_size: int) -> dict[str, Any]:
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    old, new = _metrics(before), _metrics(after)
    hits = new.get("vllm:prefix_cache_hits_total", 0) - old.get("vllm:prefix_cache_hits_total", 0)
    queries = new.get("vllm:prefix_cache_queries_total", 0) - old.get("vllm:prefix_cache_queries_total", 0)
    aligned = int(hits // block_size) * block_size
    return {
        "hit_tokens_raw": hits,
        "hit_tokens_block_aligned": aligned,
        "queried_tokens": queries,
        "block_aligned_hit_rate": aligned / queries if queries > 0 else None,
        "block_size": block_size,
    }


def classify_reset(status: int) -> dict[str, bool]:
    available = 200 <= status < 300
    return {"available": available, "cold_claim_allowed": available}


def require_cold_reset(status: int) -> None:
    if not classify_reset(status)["cold_claim_allowed"]:
        raise ResetError(f"cache reset unavailable (HTTP {status}); refusing cold claim")


def require_clean_process(metrics: str) -> dict[str, Any]:
    values = _metrics(metrics)
    required = ("vllm:prompt_tokens_total", "vllm:prefix_cache_queries_total")
    if any(name not in values for name in required):
        raise ResetError("clean-process proof metrics are unavailable")
    if any(values[name] != 0 for name in required):
        raise ResetError("clean-process proof failed: inference/cache counters are nonzero")
    return {"method": "clean-process", "cold_claim_allowed": True}


def verify_usage(expected_prompt_tokens: int, usage: dict[str, Any]) -> None:
    actual = usage.get("prompt_tokens")
    if actual != expected_prompt_tokens:
        raise ValueError(f"prompt token mismatch: expected {expected_prompt_tokens}, got {actual}")
    if not isinstance(usage.get("completion_tokens"), int):
        raise ValueError("missing completion token usage")


def require_health(before: bool, after: bool) -> None:
    if not before or not after:
        raise RuntimeError(f"health check failed (before={before}, after={after})")


def _health(base_url: str) -> bool:
    try:
        with urllib.request.urlopen(base_url.rstrip("/") + "/health", timeout=30):
            return True
    except (OSError, urllib.error.HTTPError):
        return False


def _post_json(url: str, payload: dict[str, Any], timeout=600) -> dict[str, Any]:
    request = urllib.request.Request(url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def _get_text(url: str) -> str:
    with urllib.request.urlopen(url, timeout=30) as response:
        return response.read().decode()


def _reset(base_url: str) -> int:
    request = urllib.request.Request(base_url.rstrip("/") + "/reset_prefix_cache", data=b"{}", headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return response.status
    except urllib.error.HTTPError as exc:
        return exc.code


def _load_decode_module():
    path = pathlib.Path(__file__).with_name("bench-decode.py")
    spec = importlib.util.spec_from_file_location("strict_decode", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load strict SSE implementation")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="overlord-testing")
    parser.add_argument("--target", type=int, choices=PUBLIC_TARGETS, required=True)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--cold-method", choices=("reset-endpoint", "clean-process"), default="reset-endpoint")
    args = parser.parse_args(argv)
    if args.cold_method == "clean-process" and args.runs != 1:
        parser.error("--cold-method clean-process requires --runs 1 and a fresh runtime")
    base = args.base_url.rstrip("/")
    decode = _load_decode_module()
    receipts = []
    for index in range(args.runs):
        health_before = _health(base)
        salt = run_salt(f"public-{args.target}-v1", index)
        def tokenize(text):
            response = _post_json(base + "/tokenize", {"model": args.model, "messages": [{"role": "user", "content": text}]})
            count = response.get("count", response.get("num_tokens"))
            if not isinstance(count, int):
                tokens = response.get("tokens")
                if not isinstance(tokens, list):
                    raise ValueError("/tokenize returned no count")
                count = len(tokens)
            return count
        prompt, exact_count = calibrate_prompt(args.target, salt, tokenize)
        if args.cold_method == "reset-endpoint":
            status = _reset(base)
            require_cold_reset(status)
            cold_proof = {"method": "reset-endpoint", **classify_reset(status)}
            before = _get_text(base + "/metrics")
        else:
            before = _get_text(base + "/metrics")
            cold_proof = require_clean_process(before)
        payload = {"model": args.model, "messages": [{"role": "user", "content": prompt}], "temperature": 0, "max_tokens": 1, "stream": True, "stream_options": {"include_usage": True}}
        cold_started, cold_first, cold_finished, cold = decode._run(base, payload)
        middle = _get_text(base + "/metrics")
        warm_started, warm_first, warm_finished, warm = decode._run(base, payload)
        after = _get_text(base + "/metrics")
        health_after = _health(base)
        require_health(health_before, health_after)
        verify_usage(exact_count, cold["usage"])
        verify_usage(exact_count, warm["usage"])
        receipts.append({
            "fixture_id": f"public-{args.target}-v1", "fixture_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            "target_chat_tokens": args.target, "exact_chat_tokens": exact_count, "salt": salt,
            "health_before": health_before, "health_after": health_after,
            "reset": cold_proof, "cold_ttft_seconds": cold_first - cold_started,
            "cold_wall_seconds": cold_finished - cold_started, "warm_ttft_seconds": warm_first - warm_started,
            "warm_wall_seconds": warm_finished - warm_started,
            "cold_cache_delta": cache_metric_delta(before, middle, args.block_size),
            "warm_cache_delta": cache_metric_delta(middle, after, args.block_size),
        })
    print(json.dumps({"schema_version": 1, "runs": receipts}, sort_keys=True, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
