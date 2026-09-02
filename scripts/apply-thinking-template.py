#!/usr/bin/env python3
"""Derive a thinking-controllable chat template from the pinned K3 template."""
from __future__ import annotations

import argparse
import hashlib
import os
import tempfile
from pathlib import Path

BEFORE_SHA256 = "34d5ee66b12fa6446cdae131c352b8f68cd85369e0e6fda115583805fada3891"
AFTER_SHA256 = "5bcdf9be4e5b4a6cf2017f74f7e0b5c7f91bb814a275438dc678dd48da1f81b5"
OLD_REASONING = "{%- set effective_reasoning_effort = reasoning_effort if reasoning_effort is defined and reasoning_effort in ['low', 'high'] else 'max' -%}"
NEW_REASONING = "{%- set effective_reasoning_effort = none if (enable_thinking is defined and not enable_thinking) else (reasoning_effort if reasoning_effort is defined and reasoning_effort in ['low', 'high'] else 'max') -%}"
OLD_GENERATION = """{%- if add_generation_prompt -%}
    <|assistant|>{{- '<think>' -}}
{%- endif -%}"""
NEW_GENERATION = """{%- if add_generation_prompt -%}
    <|assistant|>{{- '<think></think>' if effective_reasoning_effort is none else '<think>' -}}
{%- endif -%}"""


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def derive(raw: bytes) -> bytes:
    text = raw.decode("utf-8")
    if text.count(OLD_REASONING) != 1 or text.count(OLD_GENERATION) != 1:
        raise SystemExit("Expected exactly one reasoning and generation block")
    return text.replace(OLD_REASONING, NEW_REASONING).replace(
        OLD_GENERATION, NEW_GENERATION
    ).encode()


def atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink() or (path.exists() and not path.is_file()):
        raise SystemExit(f"Refusing unsafe output: {path}")
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o644)
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--remove", action="store_true")
    args = parser.parse_args()
    source = args.model_dir / "chat_template.jinja"
    output = args.output

    if args.remove:
        if not output.exists():
            return
        if output.is_symlink() or not output.is_file():
            raise SystemExit(f"Refusing unsafe output: {output}")
        current = digest(output.read_bytes())
        if current != AFTER_SHA256:
            raise SystemExit(f"Refusing to remove unknown output: {current}")
        output.unlink()
        return

    if not source.is_file():
        raise SystemExit(f"Pinned source template is missing: {source}")
    raw = source.read_bytes()
    current = digest(raw)
    if current != BEFORE_SHA256:
        raise SystemExit(f"Refusing unknown source template: {current}")
    derived = derive(raw)
    if digest(derived) != AFTER_SHA256:
        raise SystemExit("Derived template hash mismatch")
    if output.exists():
        if output.is_symlink() or not output.is_file():
            raise SystemExit(f"Refusing unsafe output: {output}")
        existing = digest(output.read_bytes())
        if existing == AFTER_SHA256:
            print(f"already_derived={output} sha256={existing}")
            return
        raise SystemExit(f"Refusing unknown existing output: {existing}")
    atomic_write(output, derived)
    print(f"derived={output} sha256={AFTER_SHA256}")


if __name__ == "__main__":
    main()
