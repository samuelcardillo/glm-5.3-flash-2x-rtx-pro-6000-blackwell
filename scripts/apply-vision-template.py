#!/usr/bin/env python3
"""Reversibly enable vision and make enable_thinking=false effective."""
from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import tempfile
from pathlib import Path

BEFORE_SHA256 = "41cff9af7b3a86c96751b107a8444f245fbda0bd5320b636a5bb1f7f4ba1a5c3"
LEGACY_AFTER_SHA256 = "15e397141077cea6619b82d0cf4b0f9419668580bebf9a82a9d1b3ad2fea3df5"
AFTER_SHA256 = "0a3c8768eb6b309ef077163b045b4240d5609a9c84ea1b7019aa63d76c9301bc"

OLD = """{%- macro visible_text(content) -%}
    {%- if content is string -%}
        {{- content }}
    {%- elif content is iterable and content is not mapping -%}
        {%- for item in content -%}
            {%- if item is mapping and item.type == 'text' -%}
                {{- item.text }}
            {%- elif item is string -%}
                {{- item }}
            {%- elif item is mapping and item.type in ['image', 'image_url', 'video', 'video_url', 'audio', 'audio_url', 'input_audio'] -%}
                {%- set media_type = item.type | replace('_url', '') | replace('input_', '') -%}
                {{- "<reminder>You are unable to process this " ~ media_type ~ " because you don't have multi-modal input ability. Try different methods.</reminder>" }}
            {%- endif -%}
        {%- endfor -%}
    {%- else -%}
        {{- content }}
    {%- endif -%}
{%- endmacro -%}"""
NEW = """{%- macro emit_image() -%}<|begin_of_image|><|image|><|end_of_image|>{%- endmacro -%}
{%- macro emit_video() -%}<|begin_of_video|><|video|><|end_of_video|>{%- endmacro -%}
{%- macro emit_audio() -%}<|begin_of_audio|><|end_of_audio|>{%- endmacro -%}
{%- macro visible_text(content) -%}
    {%- if content is string -%}
        {{- content }}
    {%- elif content is iterable and content is not mapping -%}
        {%- for item in content -%}
            {%- if item is mapping and item.type == 'text' -%}
                {{- item.text }}
            {%- elif item is string -%}
                {{- item }}
            {%- elif item is mapping and item.type in ['image', 'image_url'] -%}
                {{- emit_image() -}}
            {%- elif item is mapping and item.type in ['video', 'video_url'] -%}
                {{- emit_video() -}}
            {%- elif item is mapping and item.type in ['audio', 'audio_url', 'input_audio'] -%}
                {{- emit_audio() -}}
            {%- endif -%}
        {%- endfor -%}
    {%- else -%}
        {{- content }}
    {%- endif -%}
{%- endmacro -%}"""
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


def read_bytes_preserving_atime(path: Path) -> tuple[bytes, os.stat_result]:
    source_stat = path.stat(follow_symlinks=False)
    flags = os.O_RDONLY | getattr(os, "O_NOATIME", 0)
    restored_atime = False
    try:
        fd = os.open(path, flags)
    except PermissionError:
        fd = os.open(path, os.O_RDONLY)
        restored_atime = True
    with os.fdopen(fd, "rb") as handle:
        data = handle.read()
    if restored_atime:
        os.utime(
            path,
            ns=(source_stat.st_atime_ns, source_stat.st_mtime_ns),
            follow_symlinks=False,
        )
    return data, source_stat


def atomic_write(
    target: Path,
    data: bytes,
    expected_hash: str,
    metadata_source: Path,
    source_stat: os.stat_result | None = None,
) -> None:
    """Durably stage verified bytes, then atomically replace target."""
    source_stat = source_stat or metadata_source.stat(follow_symlinks=False)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    temporary = Path(temporary_name)
    try:
        try:
            handle = os.fdopen(fd, "wb")
        except BaseException:
            os.close(fd)
            raise
        with handle:
            handle.write(data)
            handle.flush()
            got = digest(os.pread(handle.fileno(), len(data) + 1, 0))
            if got != expected_hash:
                raise RuntimeError(
                    f"Refusing atomic replacement with hash {got}; expected {expected_hash}"
                )
            try:
                os.chown(temporary, source_stat.st_uid, source_stat.st_gid)
            except PermissionError as error:
                raise PermissionError(
                    "Refusing replacement because original ownership/group "
                    "cannot be preserved; run as an identity authorized to "
                    "retain them"
                ) from error
            shutil.copystat(metadata_source, temporary, follow_symlinks=False)
            os.utime(
                temporary,
                ns=(source_stat.st_atime_ns, source_stat.st_mtime_ns),
                follow_symlinks=False,
            )
            os.fsync(handle.fileno())
        os.replace(temporary, target)
        directory_fd = os.open(target.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def transform(text: str) -> str:
    if text.count(OLD) == 1:
        text = text.replace(OLD, NEW)
    elif text.count(NEW) != 1:
        raise SystemExit("Expected exactly one known media-template block")
    if text.count(OLD_REASONING) != 1:
        raise SystemExit("Expected exactly one legacy reasoning-default expression")
    if text.count(OLD_GENERATION) != 1:
        raise SystemExit("Expected exactly one legacy generation prompt")
    return text.replace(OLD_REASONING, NEW_REASONING).replace(
        OLD_GENERATION, NEW_GENERATION
    )


def ensure_backup(
    target: Path,
    backup: Path,
    current: str,
    text: str,
    raw: bytes,
    target_stat: os.stat_result,
) -> tuple[bytes, os.stat_result]:
    if backup.is_symlink():
        raise SystemExit(f"Refusing symlink backup path: {backup}")
    if backup.is_file():
        backup_raw, backup_stat = read_bytes_preserving_atime(backup)
        got = digest(backup_raw)
        if got != BEFORE_SHA256:
            raise SystemExit(
                f"Refusing invalid backup: {got}\nExpected: {BEFORE_SHA256}"
            )
        return backup_raw, backup_stat
    if current == BEFORE_SHA256:
        atomic_write(backup, raw, BEFORE_SHA256, target, target_stat)
        return raw, target_stat
    if current == LEGACY_AFTER_SHA256 and text.count(NEW) == 1:
        reconstructed = text.replace(NEW, OLD)
        if digest(reconstructed.encode()) != BEFORE_SHA256:
            raise SystemExit("Could not reconstruct the tested original template")
        reconstructed_raw = reconstructed.encode()
        atomic_write(backup, reconstructed_raw, BEFORE_SHA256, target, target_stat)
        return reconstructed_raw, target_stat
    raise SystemExit("The tested original backup is missing; refusing irreversible patch")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("--restore", action="store_true")
    args = parser.parse_args()
    target = args.model_dir / "chat_template.jinja"
    backup = args.model_dir / "chat_template.text-only.bak.jinja"
    if target.is_symlink():
        raise SystemExit(f"Refusing symlink template target: {target}")
    if not target.is_file():
        raise SystemExit(f"No template found: {target}")
    raw, target_stat = read_bytes_preserving_atime(target)
    current = digest(raw)
    text = raw.decode()

    if args.restore:
        if current not in {LEGACY_AFTER_SHA256, AFTER_SHA256}:
            raise SystemExit(
                f"Refusing restore over unknown target: {current}\n"
                f"Expected: {LEGACY_AFTER_SHA256} or {AFTER_SHA256}"
            )
        backup_raw, backup_stat = ensure_backup(
            target, backup, current, text, raw, target_stat
        )
        atomic_write(target, backup_raw, BEFORE_SHA256, backup, backup_stat)
        print(f"restored={target} sha256={BEFORE_SHA256}")
        return

    if current == AFTER_SHA256:
        ensure_backup(target, backup, current, text, raw, target_stat)
        print(
            f"already_patched={target} sha256={current} "
            f"backup_sha256={BEFORE_SHA256}"
        )
        return
    if current not in {BEFORE_SHA256, LEGACY_AFTER_SHA256}:
        raise SystemExit(
            f"Refusing unknown template revision: {current}\n"
            f"Expected: {BEFORE_SHA256} or {LEGACY_AFTER_SHA256}"
        )

    ensure_backup(target, backup, current, text, raw, target_stat)
    final_text = transform(text)
    final_raw = final_text.encode()
    final = digest(final_raw)
    if final != AFTER_SHA256:
        raise SystemExit(f"Unexpected transformed hash {final}; target unchanged")
    atomic_write(target, final_raw, AFTER_SHA256, target, target_stat)
    print(f"patched={target} sha256={final} backup={backup}")


if __name__ == "__main__":
    main()
