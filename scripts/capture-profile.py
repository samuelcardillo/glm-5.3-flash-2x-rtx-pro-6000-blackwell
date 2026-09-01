#!/usr/bin/env python3
"""Create a deterministic, privacy-safe runtime reconstruction profile."""
import argparse
import hashlib
import json
import pathlib
import re
import sys
from typing import Any

REDACTED = "<redacted>"
SENSITIVE_KEYS = re.compile(r"(?:authorization|api[_-]?key|token|secret|password|request|prompt|content|host(?:name)?|gpu|uuid|pci)", re.I)
PATH_RE = re.compile(r"^(?:/|~[/\\]|[A-Za-z]:[\\/])")
GPU_RE = re.compile(r"(?:GPU-[0-9a-f-]{8,}|\b[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]\b)", re.I)
SECRET_FLAGS = {"--api-key", "--token", "--password", "--authorization"}
PATH_FLAGS = {"--model", "--download-dir", "--chat-template", "--served-model-name-path"}
HOST_FLAGS = {"--host", "--hostname", "--node-address"}
METRIC_PREFIXES = ("vllm:", "vllm_")


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def redact(value: Any, key: str = "") -> Any:
    if key and SENSITIVE_KEYS.search(key):
        return REDACTED
    if isinstance(value, dict):
        return {str(k): redact(v, str(k)) for k, v in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, list):
        return [redact(item) for item in value]
    if isinstance(value, str):
        if PATH_RE.match(value):
            return "<redacted-path>"
        if GPU_RE.search(value):
            return REDACTED
    return value


def redact_args(args: list[str]) -> list[str]:
    result: list[str] = []
    redact_next = None
    for arg in args:
        if redact_next:
            result.append("<redacted-path>" if redact_next == "path" else REDACTED)
            redact_next = None
        elif arg in SECRET_FLAGS:
            result.append(arg)
            redact_next = "secret"
        elif arg in PATH_FLAGS:
            result.append(arg)
            redact_next = "path"
        elif arg in HOST_FLAGS:
            result.append(arg)
            redact_next = "secret"
        elif any(arg.startswith(flag + "=") for flag in SECRET_FLAGS):
            result.append(arg.split("=", 1)[0] + "=" + REDACTED)
        elif any(arg.startswith(flag + "=") for flag in PATH_FLAGS):
            result.append(arg.split("=", 1)[0] + "=<redacted-path>")
        elif any(arg.startswith(flag + "=") for flag in HOST_FLAGS):
            result.append(arg.split("=", 1)[0] + "=" + REDACTED)
        else:
            result.append(redact(arg))
    return result


def parse_metrics(text: str) -> dict[str, float]:
    selected: dict[str, float] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        name = fields[0].split("{", 1)[0]
        if len(fields) >= 2 and name.startswith(METRIC_PREFIXES):
            try:
                selected[name] = float(fields[-1])
            except ValueError:
                continue
    return dict(sorted(selected.items()))


def build_profile(source: dict[str, Any]) -> dict[str, Any]:
    required = ("image", "checkpoint_revision", "runtime_revision", "template_hash", "model_alias", "max_model_len")
    missing = [key for key in required if not source.get(key)]
    if missing:
        raise ValueError("missing reconstruction fields: " + ", ".join(missing))
    return {
        "schema_version": 1,
        "artifacts": {
            "checkpoint_revision": source["checkpoint_revision"],
            "image": source["image"],
            "runtime_revision": source["runtime_revision"],
            "template_hash": source["template_hash"],
        },
        "launch_args": redact_args(list(source.get("launch_args", []))),
        "metrics": parse_metrics(source.get("metrics", "")),
        "model": {"alias": source["model_alias"], "max_context_tokens": int(source["max_model_len"])},
        "service": redact(source.get("service_state", {})),
    }


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, indent=2, ensure_ascii=True, allow_nan=False) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=pathlib.Path, required=True, help="local JSON source receipt")
    parser.add_argument("--output", type=pathlib.Path, help="profile destination (stdout if omitted)")
    args = parser.parse_args(argv)
    profile = build_profile(json.loads(args.input.read_text(encoding="utf-8")))
    output = canonical_json(profile)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output, encoding="utf-8")
    else:
        sys.stdout.write(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
