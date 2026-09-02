#!/usr/bin/env python3
"""Run glmrt's seven semantic content contracts against a stock vLLM API."""

from __future__ import annotations

import argparse
import ast
import json
import re
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PromptCase:
    category: str
    prompt: str
    max_tokens: int


CASES = {
    "code": PromptCase(
        "code",
        "Write a Python function merge_intervals(intervals) that merges overlapping "
        "integer intervals. Include type hints, a short docstring, and three assert-based "
        "examples. Return only one Python code block.",
        320,
    ),
    "math": PromptCase(
        "reasoning",
        "A shop discounts a $240 jacket by 25%, then applies 8% sales tax to the "
        "discounted price. What is the final price? Show the calculation briefly.",
        128,
    ),
    "fable": PromptCase(
        "creative-prose",
        "Write a self-contained fable of 140 to 170 words about two parrots who disagree "
        "about sharing credit. End with a one-sentence moral.",
        256,
    ),
    "hello": PromptCase("short-response", "hi", 32),
    "topic": PromptCase(
        "exposition",
        "Explain virtual memory to a junior programmer in five concise bullet points, "
        "including paging, page faults, and the role of the TLB.",
        384,
    ),
    "structured-json": PromptCase(
        "structured-output",
        "Return only a JSON object describing a file edit with keys path, operation, "
        "line_start, line_end, and rationale. Use path src/cache.rs, operation replace, "
        "lines 41 through 47, and a one-sentence rationale about removing a redundant copy.",
        128,
    ),
    "multilingual": PromptCase(
        "multilingual",
        "請用繁體中文，以四個簡短條列解釋什麼是寫入時複製（copy-on-write），"
        "並包含一個行程 fork 後修改記憶體頁面的例子。",
        384,
    ),
}


def validate_case_content(case_id: str, content: str) -> dict:
    """Check prompt-visible contracts without executing generated code."""

    issues: list[str] = []
    stripped = content.strip()
    if not stripped:
        issues.append("response is empty")
    elif case_id == "code":
        match = re.fullmatch(
            r"\s*```(?:python|py)?\s*\n(?P<code>.*)\n```\s*",
            content,
            flags=re.DOTALL | re.IGNORECASE,
        )
        if match is None:
            issues.append("response is not exactly one Python code block")
        else:
            try:
                tree = ast.parse(match.group("code"))
            except SyntaxError:
                issues.append("Python code does not parse")
            else:
                functions = [
                    node
                    for node in tree.body
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and node.name == "merge_intervals"
                ]
                if len(functions) != 1:
                    issues.append("merge_intervals function is missing or duplicated")
                else:
                    fn = functions[0]
                    if (
                        not fn.args.args
                        or fn.args.args[0].annotation is None
                        or fn.returns is None
                    ):
                        issues.append("merge_intervals lacks requested type hints")
                    if ast.get_docstring(fn) is None:
                        issues.append("merge_intervals lacks a docstring")
                if sum(isinstance(node, ast.Assert) for node in ast.walk(tree)) < 3:
                    issues.append("fewer than three assert examples were provided")
    elif case_id == "math":
        normalized = stripped.replace(",", "")
        if re.search(r"(?<![0-9])(?:\$\s*)?194\.4(?:0)?(?![0-9])", normalized) is None:
            issues.append("response does not contain the correct final price 194.40")
        if "240" not in normalized or "25" not in normalized or "8" not in normalized:
            issues.append("response does not show the requested calculation inputs")
    elif case_id == "fable":
        words = re.findall(r"\b[\w'-]+\b", stripped, flags=re.UNICODE)
        if not 140 <= len(words) <= 170:
            issues.append(f"fable has {len(words)} words, outside 140..170")
        final_sentence = re.split(r"(?<=[.!?])\s+", stripped)[-1].casefold()
        if not any(
            term in final_sentence
            for term in ("credit", "share", "together", "team", "fair", "both")
        ):
            issues.append("response does not end with a moral about sharing credit")
    elif case_id == "hello":
        if len(stripped) > 512:
            issues.append("short greeting response is unexpectedly long")
    elif case_id == "topic":
        bullets = [
            line
            for line in stripped.splitlines()
            if re.match(r"^\s*(?:[-*•]|[1-5][.)])\s+", line)
        ]
        if len(bullets) != 5:
            issues.append(f"response has {len(bullets)} bullets, expected five")
        lowered = stripped.casefold()
        for term in ("paging", "page fault", "tlb"):
            if term not in lowered:
                issues.append(f"response omits {term}")
    elif case_id == "structured-json":
        json_text = stripped
        fenced = re.fullmatch(
            r"```(?:json)?\s*\n(?P<json>.*)\n```",
            stripped,
            flags=re.DOTALL | re.IGNORECASE,
        )
        if fenced is not None:
            # GLM sometimes wraps an otherwise exact object in a JSON fence.
            # Treat that as a presentation quirk, not a semantic failure.
            json_text = fenced.group("json")
        try:
            value = json.loads(json_text)
        except json.JSONDecodeError:
            issues.append("response is not bare valid JSON")
        else:
            expected = {"path", "operation", "line_start", "line_end", "rationale"}
            if not isinstance(value, dict) or set(value) != expected:
                issues.append("JSON object has the wrong key set")
            elif (
                value["path"] != "src/cache.rs"
                or value["operation"] != "replace"
                or value["line_start"] != 41
                or value["line_end"] != 47
                or not isinstance(value["rationale"], str)
                or not value["rationale"].strip()
            ):
                issues.append("JSON object does not preserve the requested edit")
    elif case_id == "multilingual":
        bullets = [
            line
            for line in stripped.splitlines()
            if re.match(r"^\s*(?:[-*•]|[1-4][.)、])\s*", line)
        ]
        if len(bullets) != 4:
            issues.append(f"response has {len(bullets)} bullets, expected four")
        lowered = stripped.casefold()
        if not ("寫入時複製" in stripped or "copy-on-write" in lowered):
            issues.append("response omits copy-on-write")
        if "fork" not in lowered or "頁" not in stripped:
            issues.append("response omits the requested fork/page example")
    return {
        "quality_contract_passed": not issues,
        "quality_contract_issues": issues,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument(
        "--model", default="glm-5.3-flash-local"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=900)
    parser.add_argument("--seed", type=int, default=20260901)
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=0,
        help="Optional override for each contract's visible completion budget.",
    )
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    def post(path: str, payload: dict) -> dict:
        request = urllib.request.Request(
            args.base_url.rstrip("/") + path,
            data=json.dumps(payload, ensure_ascii=False).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=args.timeout) as response:
            return json.load(response)

    close_think = post(
        "/tokenize",
        {"model": args.model, "prompt": "</think>", "add_special_tokens": False},
    )["tokens"]
    results = []
    for case_id in CASES:
        case = CASES[case_id]
        render = post(
            "/v1/chat/completions/render",
            {
                "model": args.model,
                "messages": [{"role": "user", "content": case.prompt}],
                "reasoning_effort": "low",
            },
        )
        payload = {
            "model": args.model,
            # The published template opens <think> unconditionally. Closing it
            # here makes this a visible-answer quality test, not a reasoning-
            # token budget test.
            "prompt": render["token_ids"] + close_think,
            "temperature": 0,
            "seed": args.seed,
            "max_tokens": args.max_tokens or case.max_tokens,
            "add_special_tokens": False,
        }
        started = time.perf_counter()
        body = post("/v1/completions", payload)
        seconds = time.perf_counter() - started
        choice = body["choices"][0]
        content = choice.get("text") or ""
        validation = validate_case_content(case_id, content)
        result = {
            "case": case_id,
            "category": case.category,
            "passed": validation["quality_contract_passed"],
            "issues": validation["quality_contract_issues"],
            "seconds": round(seconds, 3),
            "prompt_tokens": body["usage"]["prompt_tokens"],
            "completion_tokens": body["usage"]["completion_tokens"],
            "finish_reason": choice["finish_reason"],
            "preview": content[:240].replace("\n", "\\n"),
        }
        results.append(result)
        print(json.dumps(result, ensure_ascii=False), flush=True)

    summary = {
        "summary": True,
        "passed": sum(item["passed"] for item in results),
        "total": len(results),
        "request_seconds": round(sum(item["seconds"] for item in results), 3),
        "completion_tokens": sum(item["completion_tokens"] for item in results),
    }
    print(json.dumps(summary), flush=True)
    with args.output.open("w", encoding="utf-8") as output:
        for result in results:
            output.write(json.dumps(result, ensure_ascii=False) + "\n")
        output.write(json.dumps(summary) + "\n")
    if summary["passed"] != summary["total"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
