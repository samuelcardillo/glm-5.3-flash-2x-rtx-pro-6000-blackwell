#!/usr/bin/env python3
"""Apply a fail-closed mixed-prefill policy to one exact vLLM scheduler.

The overlay recognizes only the scheduler shipped by the pinned local
vLLM 0.1.dev20051+g487ecf187 runtime and its exact transformed result.
Unknown, partial, missing, or symlinked source is never modified.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import tempfile
from pathlib import Path
from typing import Callable, NamedTuple

SCHEDULER_PATH = "vllm/v1/core/sched/scheduler.py"
POLICY_ENV = "GLM53_MIXED_PREFILL_CHUNK"
MARKER = b"# [glm53-mixed-prefill-policy]"


class Edit(NamedTuple):
    old: bytes
    new: bytes
    label: str


def parse_policy(raw: str, max_cap: int) -> int | None:
    """Parse exact `off`, `skip`, or a canonical positive bounded integer."""
    if not isinstance(max_cap, int) or isinstance(max_cap, bool) or max_cap <= 0:
        raise ValueError("mixed-prefill maximum cap must be a positive integer")
    if raw == "off":
        return None
    if raw == "skip":
        return 0
    if not raw or not raw.isascii() or not raw.isdecimal() or raw[0] == "0":
        raise ValueError(
            "mixed-prefill policy must be exactly off, skip, or a positive integer"
        )
    cap = int(raw)
    if cap > max_cap:
        raise ValueError(f"mixed-prefill cap {cap} exceeds scheduler bound {max_cap}")
    return cap


IMPORT_OLD = b"import itertools\nimport time\n"
IMPORT_NEW = b"import itertools\nimport os\nimport time\n"
HELPER_INSERT = b"from vllm.compilation.cuda_graph import CUDAGraphStat\n"

POLICY_HELPER = b'''def _glm53_parse_mixed_prefill_policy(raw, max_cap):
    if not isinstance(max_cap, int) or isinstance(max_cap, bool) or max_cap <= 0:
        raise ValueError("mixed-prefill maximum cap must be a positive integer")
    if raw == "off":
        return None
    if raw == "skip":
        return 0
    if not raw or not raw.isascii() or not raw.isdecimal() or raw[0] == "0":
        raise ValueError(
            "GLM53_MIXED_PREFILL_CHUNK must be exactly off, skip, "
            "or a positive integer"
        )
    cap = int(raw)
    if cap > max_cap:
        raise ValueError(
            f"GLM53_MIXED_PREFILL_CHUNK cap {cap} exceeds scheduler bound {max_cap}"
        )
    return cap


def _glm53_mixed_prefill_policy(running, current, max_cap):
    """Return None for stock behavior, zero to skip, or a bounded chunk cap."""
    cap = _glm53_parse_mixed_prefill_policy(
        os.environ.get("GLM53_MIXED_PREFILL_CHUNK", "off"), max_cap
    )
    if cap is None:
        return None
    current_id = getattr(current, "request_id", None)
    for peer in running:
        if peer is current or getattr(peer, "request_id", None) == current_id:
            continue
        if peer.num_computed_tokens >= peer.num_prompt_tokens:
            return cap
    return None


'''
HELPER_NEW = POLICY_HELPER + HELPER_INSERT

RUNNING_OLD = b'''            if 0 < self.scheduler_config.long_prefill_token_threshold < num_new_tokens:
                num_new_tokens = self.scheduler_config.long_prefill_token_threshold
            num_new_tokens = min(
                num_new_tokens, token_budget, input_budget - draft_slots
            )

            # Make sure the input position does not exceed the max model len.
'''
RUNNING_NEW = b'''            if 0 < self.scheduler_config.long_prefill_token_threshold < num_new_tokens:
                num_new_tokens = self.scheduler_config.long_prefill_token_threshold
            num_new_tokens = min(
                num_new_tokens, token_budget, input_budget - draft_slots
            )
            if request.num_computed_tokens < request.num_prompt_tokens:
                mixed_prefill_cap = _glm53_mixed_prefill_policy(
                    self.running,
                    request,
                    self.scheduler_config.max_num_batched_tokens,
                )  # [glm53-mixed-prefill-policy]
                if mixed_prefill_cap is not None:
                    num_new_tokens = min(num_new_tokens, mixed_prefill_cap)

            # Make sure the input position does not exceed the max model len.
'''

WAITING_OLD = b'''                    threshold = self.scheduler_config.long_prefill_token_threshold
                    if 0 < threshold < num_new_tokens:
                        num_new_tokens = threshold

                    # chunked prefill has to be enabled explicitly to allow
'''
WAITING_NEW = b'''                    threshold = self.scheduler_config.long_prefill_token_threshold
                    if 0 < threshold < num_new_tokens:
                        num_new_tokens = threshold
                    if num_computed_tokens < request.num_prompt_tokens:
                        mixed_prefill_cap = _glm53_mixed_prefill_policy(
                            self.running,
                            request,
                            self.scheduler_config.max_num_batched_tokens,
                        )  # [glm53-mixed-prefill-policy]
                        if mixed_prefill_cap == 0:
                            request_queue.pop_request()
                            step_skipped_waiting.prepend_request(request)
                            continue
                        if mixed_prefill_cap is not None:
                            num_new_tokens = min(
                                num_new_tokens, mixed_prefill_cap
                            )

                    # chunked prefill has to be enabled explicitly to allow
'''

PATCHES = {
    SCHEDULER_PATH: (
        Edit(IMPORT_OLD, IMPORT_NEW, "os import"),
        Edit(HELPER_INSERT, HELPER_NEW, "policy helper"),
        Edit(RUNNING_OLD, RUNNING_NEW, "running prefill path"),
        Edit(WAITING_OLD, WAITING_NEW, "waiting prefill path"),
    )
}

# Exact scheduler installed in the pinned A2 runtime parent image.
ORIGINAL_SHA256 = {
    SCHEDULER_PATH: "a509e42c4b9b6ed388f1f5372a7b50d0aeab58f959113812f817f014febef103"
}
# Frozen after applying only PATCHES above to that exact installed source.
PATCHED_SHA256 = {
    SCHEDULER_PATH: "4758a9f50e1234f872b017f3ce766f4493d87a1ce775c705dff87ee9988fb87e"
}


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def transform(relative: str, data: bytes) -> bytes:
    if relative not in PATCHES:
        raise SystemExit(f"unsupported source path: {relative}")
    result = data
    for edit in PATCHES[relative]:
        old_count = result.count(edit.old)
        new_count = result.count(edit.new)
        if old_count != 1 or new_count != 0:
            raise SystemExit(
                f"source anchor mismatch: {relative} {edit.label}: "
                f"old_count={old_count} new_count={new_count}"
            )
        result = result.replace(edit.old, edit.new, 1)
    return result


def validate_python(data: bytes, relative: str) -> None:
    try:
        compile(data, relative, "exec")
    except (SyntaxError, ValueError, TypeError) as error:
        raise SystemExit(f"transformed source does not compile: {relative}: {error}") from error


def _atomic_write(path: Path, data: bytes) -> None:
    source_stat = path.stat(follow_symlinks=False)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fchmod(handle.fileno(), source_stat.st_mode)
            try:
                os.fchown(handle.fileno(), source_stat.st_uid, source_stat.st_gid)
            except PermissionError as error:
                raise SystemExit(f"cannot preserve ownership for {path}: {error}") from error
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def apply_overlay(
    root: Path,
    *,
    original_hashes: dict[str, str] = ORIGINAL_SHA256,
    patched_hashes: dict[str, str] = PATCHED_SHA256,
    compiler: Callable[[bytes, str], None] = validate_python,
    writer: Callable[[Path, bytes], None] = _atomic_write,
    verify_only: bool = False,
) -> str:
    """Preflight and compile the exact source before one atomic replacement."""
    relative = SCHEDULER_PATH
    path = root / relative
    if path.is_symlink():
        raise SystemExit(f"refusing symlink source: {path}")
    if not path.is_file():
        raise SystemExit(f"missing source: {path}")

    original = path.read_bytes()
    current = digest(original)
    if current == original_hashes[relative]:
        state = "original"
        candidate = transform(relative, original)
        try:
            compiler(candidate, relative)
        except SystemExit:
            raise
        except (SyntaxError, ValueError, TypeError) as error:
            raise SystemExit(
                f"transformed source does not compile: {relative}: {error}"
            ) from error
        if digest(candidate) != patched_hashes[relative]:
            raise SystemExit(f"transformed hash mismatch: {relative}")
    elif current == patched_hashes[relative]:
        state = "patched"
        candidate = original
        try:
            compiler(candidate, relative)
        except SystemExit:
            raise
        except (SyntaxError, ValueError, TypeError) as error:
            raise SystemExit(f"patched source does not compile: {relative}: {error}") from error
    else:
        if MARKER in original or any(edit.new in original for edit in PATCHES[relative]):
            raise SystemExit(
                f"partial mixed-prefill overlay state: {relative} sha256={current}; "
                "refusing patch"
            )
        raise SystemExit(
            f"unknown source revision: {relative} sha256={current}; refusing patch"
        )

    if verify_only:
        if state != "patched":
            raise SystemExit(f"installed source is not patched: {relative}")
        return "verified"
    if state == "patched":
        return "already_patched"

    attempted = False
    try:
        attempted = True
        writer(path, candidate)
        if digest(path.read_bytes()) != patched_hashes[relative]:
            raise RuntimeError(f"post-commit hash mismatch: {relative}")
    except BaseException as error:
        if attempted:
            try:
                _atomic_write(path, original)
            except BaseException as rollback_error:
                raise SystemExit(
                    f"mixed-prefill commit failed ({error}); rollback also failed: "
                    f"{rollback_error}"
                ) from error
        raise SystemExit(
            f"mixed-prefill commit failed and was rolled back: {error}"
        ) from error
    return "patched"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("site_packages", type=Path)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    result = apply_overlay(args.site_packages, verify_only=args.verify)
    installed_hash = digest((args.site_packages / SCHEDULER_PATH).read_bytes())
    print(
        f"mixed_prefill_policy={result} env={POLICY_ENV} "
        f"{SCHEDULER_PATH}={installed_hash}"
    )


if __name__ == "__main__":
    main()
