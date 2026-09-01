#!/usr/bin/env python3
"""Apply two official vLLM XGrammar fixes to an exact, pinned source tree.

This backport intentionally recognizes only the source shipped in the pinned base
image and the exact result of applying the two upstream hunks. Any mixed,
missing, symlinked, or drifted state is rejected before either file is written.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import stat
import tempfile
from pathlib import Path
from typing import Callable, NamedTuple

COMMIT_IDS = (
    "12f64b39d29282437e35be9aa5db432fb2a1a6e6",
    "c6e19b3be24338759a443e03c8325d76da9ee202",
)


class Edit(NamedTuple):
    old: bytes
    new: bytes


BACKEND_OLD = b'''    def accept_tokens(self, request_id: str, tokens: list[int]) -> bool:
        """Accepts a list of tokens and advances the FSM.

        Returns True if the FSM was advanced successfully.
        Returns False if the FSM failed to advance.
        """
        if self._is_terminated:
            return False
        for token in tokens:
            if not self.matcher.accept_token(token):
                logger.error(
                    "Failed to advance FSM for request %s "
                    "for tokens %s. Please file an issue.",
                    request_id,
                    token,
                )
                return False
            self.num_processed_tokens += 1
        self._is_terminated = self.matcher.is_terminated()
        return True

    def validate_tokens(self, tokens: list[int]) -> list[int]:
        """Checks if the list of tokens are accepted by the FSM in sequence.
        Will not advance the FSM.

        Returns the prefix list of tokens that are accepted by the FSM.
        """
        accepted_tokens = []
        for token in tokens:
            if self.matcher.accept_token(token):
                accepted_tokens.append(token)
            else:
                break
        if len(accepted_tokens) > 0:
            # Rollback the FSM to the initial state
            self.matcher.rollback(len(accepted_tokens))
        return accepted_tokens
'''

BACKEND_NEW = b'''    def accept_tokens(self, request_id: str, tokens: list[int]) -> bool:
        """Accepts a list of tokens and advances the FSM.

        Returns True if all grammar-constrained tokens were accepted.
        Tokens after termination are ignored. Returns False if the FSM
        failed to advance.
        """
        if self._is_terminated:
            return True
        for token in tokens:
            if not self.matcher.accept_token(token):
                logger.error(
                    "Failed to advance FSM for request %s "
                    "for tokens %s. Please file an issue.",
                    request_id,
                    token,
                )
                return False
            self.num_processed_tokens += 1
            self._is_terminated = self.matcher.is_terminated()
            if self._is_terminated:
                break
        return True

    def validate_tokens(self, tokens: list[int]) -> list[int]:
        """Checks if the list of tokens are accepted by the FSM in sequence.
        Will not advance the FSM.

        Returns the prefix list of tokens that are accepted by the FSM.
        """
        if self._is_terminated:
            return []

        accepted_tokens = []
        for token in tokens:
            if self.matcher.accept_token(token):
                accepted_tokens.append(token)
                if self.matcher.is_terminated():
                    break
            else:
                break
        if len(accepted_tokens) > 0:
            # Rollback the FSM to the initial state
            self.matcher.rollback(len(accepted_tokens))
        return accepted_tokens
'''

RESET_OLD = b'''    def reset(self):
        self.num_processed_tokens = 0
        self.matcher.reset()
'''
RESET_NEW = b'''    def reset(self):
        self.matcher.reset()
        self.num_processed_tokens = 0
        self._is_terminated = False
'''

REASONING_OLD = b'''                    if advance_grammar and not grammar.is_terminated():
                        accepted = grammar.accept_tokens(req_id, [token])
                        if accepted:
'''
REASONING_NEW = b'''                    if advance_grammar and not grammar.is_terminated():
                        if post_reasoning_end_in_window:
                            accepted = bool(grammar.validate_tokens([token]))
                            if accepted:
                                accepted = grammar.accept_tokens(req_id, [token])
                        else:
                            accepted = grammar.accept_tokens(req_id, [token])
                        if accepted:
'''

PATCHES: dict[str, tuple[Edit, ...]] = {
    "vllm/v1/structured_output/backend_xgrammar.py": (
        Edit(BACKEND_OLD, BACKEND_NEW),
        Edit(RESET_OLD, RESET_NEW),
    ),
    "vllm/v1/structured_output/__init__.py": (
        Edit(REASONING_OLD, REASONING_NEW),
    ),
}

# Exact files in vLLM 0.1.dev20051+g487ecf187 from the pinned base image.
ORIGINAL_SHA256 = {
    "vllm/v1/structured_output/backend_xgrammar.py":
        "3fd606dc2b8e950fe9b49f28cf1c030be78beaaf7c78b457b5942a0909d3457f",
    "vllm/v1/structured_output/__init__.py":
        "355f6f1193c15d5d6901a0f567e2e16005e3681f04f70079c6ba11e020b4d33a",
}
# Filled with the source-exact result of the official hunks (not whole-file
# snapshots from a later vLLM revision).
PATCHED_SHA256 = {
    "vllm/v1/structured_output/backend_xgrammar.py":
        "906b24eae8ca3cdd9425a87f9e2dfae9ef9840cfd2f3be647d7b1f4ba72cbab4",
    "vllm/v1/structured_output/__init__.py":
        "003b090b3182e377dff48561050ba86b6f671e0e5c60cc35984e96df8699c386",
}


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def transform(relative: str, data: bytes) -> bytes:
    result = data
    for edit in PATCHES[relative]:
        if result.count(edit.old) != 1 or result.count(edit.new) != 0:
            raise SystemExit(f"official source anchor mismatch: {relative}")
        result = result.replace(edit.old, edit.new)
    return result


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
            try:
                os.fchown(handle.fileno(), source_stat.st_uid, source_stat.st_gid)
            except PermissionError as error:
                raise SystemExit(f"cannot preserve ownership for {path}: {error}") from error
            # Ownership changes can clear set-ID bits, so restore the complete
            # permission mode only after ownership is in place.
            os.fchmod(handle.fileno(), stat.S_IMODE(source_stat.st_mode))
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def apply_fixes(
    root: Path,
    *,
    original_hashes: dict[str, str] = ORIGINAL_SHA256,
    patched_hashes: dict[str, str] = PATCHED_SHA256,
    writer: Callable[[Path, bytes], None] = _atomic_write,
    verify_only: bool = False,
) -> str:
    """Preflight both exact files, then patch both or verify both."""
    states: dict[str, str] = {}
    originals: dict[str, bytes] = {}
    transformed: dict[str, bytes] = {}

    # Complete two-file preflight occurs before any replacement.
    for relative in PATCHES:
        path = root / relative
        if path.is_symlink():
            raise SystemExit(f"refusing symlink source: {path}")
        if not path.is_file():
            raise SystemExit(f"missing source: {path}")
        data = path.read_bytes()
        current = digest(data)
        if current == original_hashes[relative]:
            states[relative] = "original"
            originals[relative] = data
            candidate = transform(relative, data)
            if digest(candidate) != patched_hashes[relative]:
                raise SystemExit(f"official transformed hash mismatch: {relative}")
            transformed[relative] = candidate
        elif current == patched_hashes[relative]:
            states[relative] = "patched"
        else:
            raise SystemExit(
                f"unknown source revision: {relative} sha256={current}; refusing patch"
            )

    state_set = set(states.values())
    if len(state_set) != 1:
        raise SystemExit(f"partial XGrammar backport state: {states}; refusing patch")
    state = next(iter(state_set))
    if verify_only:
        if state != "patched":
            raise SystemExit(f"installed source is not patched: {states}")
        return "verified"
    if state == "patched":
        return "already_patched"

    attempted: list[str] = []
    try:
        for relative in PATCHES:
            # Record the path before calling the writer: replacement can
            # succeed and a later durability operation can still fail.
            attempted.append(relative)
            target = root / relative
            writer(target, transformed[relative])
            if digest(target.read_bytes()) != patched_hashes[relative]:
                raise RuntimeError(f"post-commit hash mismatch: {relative}")
    except BaseException as error:
        rollback_errors = []
        for relative in reversed(attempted):
            try:
                _atomic_write(root / relative, originals[relative])
            except BaseException as rollback_error:
                rollback_errors.append(f"{relative}: {rollback_error}")
        if rollback_errors:
            raise SystemExit(
                f"XGrammar commit failed ({error}); rollback also failed: "
                + "; ".join(rollback_errors)
            ) from error
        raise SystemExit(
            f"XGrammar commit failed and was rolled back: {error}"
        ) from error
    return "patched"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("site_packages", type=Path)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    result = apply_fixes(args.site_packages, verify_only=args.verify)
    hashes = ",".join(
        f"{relative}={digest((args.site_packages / relative).read_bytes())}"
        for relative in PATCHES
    )
    print(f"xgrammar_fixes={result} commits={','.join(COMMIT_IDS)} {hashes}")


if __name__ == "__main__":
    main()
