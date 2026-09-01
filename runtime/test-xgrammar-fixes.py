#!/usr/bin/env python3
"""Fail-closed tests for the source-exact official XGrammar backport."""
from __future__ import annotations

import hashlib
import importlib.util
import os
import stat
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "runtime" / "apply-xgrammar-fixes.py"
SPEC = importlib.util.spec_from_file_location("apply_xgrammar_fixes", SCRIPT)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"could not load {SCRIPT}")
mod = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mod)


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class XGrammarFixTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="xgrammar-fixes-")
        self.root = Path(self.temporary.name)
        self.original: dict[str, bytes] = {}
        self.patched: dict[str, bytes] = {}
        for relative, edits in mod.PATCHES.items():
            source = b"# exact-anchor fixture\n" + b"\n# between official edits\n".join(
                edit.old for edit in edits
            ) + b"\n# fixture end\n"
            target = source
            for edit in edits:
                target = target.replace(edit.old, edit.new)
            self.original[relative] = source
            self.patched[relative] = target
            path = self.root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(source)
        self.original_hashes = {name: sha(data) for name, data in self.original.items()}
        self.patched_hashes = {name: sha(data) for name, data in self.patched.items()}

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def apply(self, **kwargs):
        return mod.apply_fixes(
            self.root,
            original_hashes=self.original_hashes,
            patched_hashes=self.patched_hashes,
            **kwargs,
        )

    def test_backport_is_limited_to_the_two_official_commits_and_files(self):
        self.assertEqual(
            mod.COMMIT_IDS,
            (
                "12f64b39d29282437e35be9aa5db432fb2a1a6e6",
                "c6e19b3be24338759a443e03c8325d76da9ee202",
            ),
        )
        self.assertEqual(
            set(mod.PATCHES),
            {
                "vllm/v1/structured_output/backend_xgrammar.py",
                "vllm/v1/structured_output/__init__.py",
            },
        )
        combined = b"\n".join(
            edit.new for edits in mod.PATCHES.values() for edit in edits
        )
        self.assertIn(b"Tokens after termination are ignored", combined)
        self.assertIn(b"if self.matcher.is_terminated():\n                    break", combined)
        self.assertIn(b"accepted = bool(grammar.validate_tokens([token]))", combined)

    def test_original_pair_is_patched_source_exactly(self):
        result = self.apply()
        self.assertEqual(result, "patched")
        for relative, expected in self.patched.items():
            self.assertEqual((self.root / relative).read_bytes(), expected)

    def test_patched_pair_is_accepted_as_already_upstream(self):
        for relative, data in self.patched.items():
            (self.root / relative).write_bytes(data)
        self.assertEqual(self.apply(), "already_patched")

    def test_second_application_is_idempotent(self):
        self.apply()
        before = {
            relative: (self.root / relative).stat().st_mtime_ns
            for relative in self.patched
        }
        self.assertEqual(self.apply(), "already_patched")
        after = {
            relative: (self.root / relative).stat().st_mtime_ns
            for relative in self.patched
        }
        self.assertEqual(after, before)

    def test_partial_pair_is_rejected_without_writes(self):
        first, second = tuple(mod.PATCHES)
        (self.root / first).write_bytes(self.patched[first])
        before = {name: (self.root / name).read_bytes() for name in mod.PATCHES}
        with self.assertRaisesRegex(SystemExit, "partial"):
            self.apply()
        self.assertEqual(
            {name: (self.root / name).read_bytes() for name in mod.PATCHES}, before
        )
        self.assertEqual((self.root / second).read_bytes(), self.original[second])

    def test_drifted_source_is_rejected(self):
        relative = next(iter(mod.PATCHES))
        (self.root / relative).write_bytes(self.original[relative] + b"# drift\n")
        with self.assertRaisesRegex(SystemExit, "unknown source revision"):
            self.apply()

    def test_two_file_preflight_prevents_first_file_mutation(self):
        first, second = tuple(mod.PATCHES)
        (self.root / second).write_bytes(b"drifted")
        first_before = (self.root / first).read_bytes()
        with self.assertRaises(SystemExit):
            self.apply()
        self.assertEqual((self.root / first).read_bytes(), first_before)

    def test_missing_and_symlink_sources_are_rejected(self):
        first, second = tuple(mod.PATCHES)
        (self.root / second).unlink()
        with self.assertRaisesRegex(SystemExit, "missing source"):
            self.apply()
        (self.root / second).symlink_to(self.root / first)
        with self.assertRaisesRegex(SystemExit, "symlink"):
            self.apply()

    def test_failure_after_second_replacement_rolls_back_both_files(self):
        calls = 0

        def replace_then_fail(path, data):
            nonlocal calls
            calls += 1
            mod._atomic_write(path, data)
            if calls == 2:
                raise OSError("injected durability failure")

        with self.assertRaisesRegex(SystemExit, "rolled back"):
            self.apply(writer=replace_then_fail)
        for relative, expected in self.original.items():
            self.assertEqual((self.root / relative).read_bytes(), expected)

    def test_post_commit_hash_mismatch_rolls_back(self):
        def corrupting_writer(path, data):
            mod._atomic_write(path, data + b"# corruption\n")

        with self.assertRaisesRegex(SystemExit, "post-commit hash mismatch"):
            self.apply(writer=corrupting_writer)
        for relative, expected in self.original.items():
            self.assertEqual((self.root / relative).read_bytes(), expected)

    def test_atomic_write_preserves_mode_and_ownership(self):
        relative = next(iter(mod.PATCHES))
        path = self.root / relative
        path.chmod(0o6750)
        before = path.stat(follow_symlinks=False)
        mod._atomic_write(path, self.patched[relative])
        after = path.stat(follow_symlinks=False)
        self.assertEqual(stat.S_IMODE(after.st_mode), stat.S_IMODE(before.st_mode))
        self.assertEqual((after.st_uid, after.st_gid), (before.st_uid, before.st_gid))


if __name__ == "__main__":
    unittest.main(verbosity=2)
