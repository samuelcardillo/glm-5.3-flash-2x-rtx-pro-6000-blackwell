#!/usr/bin/env python3
"""Source-exact GLM-5.3 DFlash/DCP block-table correction."""
from __future__ import annotations

import argparse
import hashlib
import os
import pathlib
import stat
import tempfile

ORIGINAL_SHA256 = "87d359028d57eb883849b8d8b03a1f7486c8e1ed5473328564629f8146ac56ed"
PATCHED_SHA256 = "84aaa80af64c91076422f0f8aa1e859d61c797c4212095fc0a2d73bde873f781"

LOGGER_ANCHOR = b"logger = init_logger(__name__)\n"
DIVISOR_ANCHOR = b"spec.block_size * self.dcp_size"
HELPER = b'''\n\ndef _effective_cache_parallel_width(cache_spec, configured_width):\n    \"\"\"Return a unanimous explicit cache width, otherwise the configured width.\"\"\"\n    nested_specs = getattr(cache_spec, "kv_cache_specs", None)\n    cache_specs = tuple(nested_specs.values()) if nested_specs else (cache_spec,)\n    requested_widths = tuple(\n        getattr(item, "dcp_shard_count_override", None) for item in cache_specs\n    )\n    if (\n        requested_widths\n        and requested_widths[0] is not None\n        and all(width == requested_widths[0] for width in requested_widths)\n    ):\n        return requested_widths[0]\n    return configured_width\n'''
REPLACEMENT_DIVISOR = (
    b"spec.block_size * "
    b"_effective_cache_parallel_width(spec, self.dcp_size)"
)


def transform(source: bytes, *, expected_original_sha256: str = ORIGINAL_SHA256) -> bytes:
    actual = hashlib.sha256(source).hexdigest()
    if actual != expected_original_sha256:
        raise ValueError(
            f"source hash mismatch: expected {expected_original_sha256}, got {actual}"
        )
    anchor_count = source.count(LOGGER_ANCHOR)
    divisor_count = source.count(DIVISOR_ANCHOR)
    if anchor_count != 1 or divisor_count != 1:
        raise ValueError(
            "patch anchors must each occur exactly once: "
            f"logger={anchor_count}, divisor={divisor_count}"
        )
    with_helper = source.replace(LOGGER_ANCHOR, LOGGER_ANCHOR + HELPER, 1)
    return with_helper.replace(DIVISOR_ANCHOR, REPLACEMENT_DIVISOR, 1)


def verify_file(
    path: pathlib.Path, *, expected_patched_sha256: str = PATCHED_SHA256
) -> None:
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected_patched_sha256:
        raise ValueError(
            f"patched source hash mismatch: expected {expected_patched_sha256}, got {actual}"
        )


def apply_file(
    path: pathlib.Path,
    *,
    expected_original_sha256: str = ORIGINAL_SHA256,
    expected_patched_sha256: str = PATCHED_SHA256,
) -> str:
    path = pathlib.Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"target must be a regular non-symlink file: {path}")
    source = path.read_bytes()
    actual = hashlib.sha256(source).hexdigest()
    if actual == expected_patched_sha256:
        return "already-patched"
    if actual != expected_original_sha256:
        raise ValueError(f"unknown source state: {actual}")
    candidate = transform(
        source, expected_original_sha256=expected_original_sha256
    )
    candidate_sha = hashlib.sha256(candidate).hexdigest()
    if candidate_sha != expected_patched_sha256:
        raise ValueError(
            f"candidate hash mismatch: expected {expected_patched_sha256}, got {candidate_sha}"
        )
    compile(candidate, str(path), "exec")
    metadata = path.stat(follow_symlinks=False)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb", closefd=True) as handle:
            handle.write(candidate)
            handle.flush()
            os.fsync(handle.fileno())
            os.fchown(handle.fileno(), metadata.st_uid, metadata.st_gid)
            os.fchmod(handle.fileno(), stat.S_IMODE(metadata.st_mode))
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
    verify_file(path, expected_patched_sha256=expected_patched_sha256)
    return "patched"


def main() -> int:
    parser = argparse.ArgumentParser()
    operation = parser.add_mutually_exclusive_group(required=True)
    operation.add_argument("--apply", action="store_true")
    operation.add_argument("--verify", action="store_true")
    parser.add_argument("path", type=pathlib.Path)
    args = parser.parse_args()
    if args.apply:
        print(apply_file(args.path))
    else:
        verify_file(args.path)
        print("verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
