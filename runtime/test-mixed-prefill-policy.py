#!/usr/bin/env python3
"""Fail-closed stdlib tests for the mixed-prefill scheduler overlay."""
from __future__ import annotations

import hashlib
import importlib.util
import os
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "runtime" / "apply-mixed-prefill-policy.py"
SPEC = importlib.util.spec_from_file_location("apply_mixed_prefill_policy", SCRIPT)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"could not load {SCRIPT}")
mod = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mod)

EXPECTED_ORIGINAL = "a509e42c4b9b6ed388f1f5372a7b50d0aeab58f959113812f817f014febef103"
EXPECTED_PATCHED = "4758a9f50e1234f872b017f3ce766f4493d87a1ce775c705dff87ee9988fb87e"


def synthetic_source() -> bytes:
    """Build a compilable scheduler-shaped source containing every exact anchor."""
    return (
        b"from __future__ import annotations\n"
        + mod.IMPORT_OLD
        + b"from vllm.compilation.cuda_graph import CUDAGraphStat\n\n"
        + b"class SyntheticScheduler:\n"
        + b"    def schedule(self):\n"
        + b"        token_budget = input_budget = draft_slots = num_new_tokens = 1\n"
        + b"        request = None\n"
        + b"        while True:\n"
        + mod.RUNNING_OLD
        + b"            num_new_tokens = num_new_tokens\n"
        + b"            while True:\n"
        + b"                if True:\n"
        + mod.WAITING_OLD
        + b"                    num_new_tokens = num_new_tokens\n"
        + b"                break\n"
        + b"            break\n"
    )


class MixedPrefillPolicyTests(unittest.TestCase):
    def test_exact_policy_enum_is_fail_closed_and_bounded(self):
        self.assertEqual(mod.parse_policy("off", 1024), None)
        self.assertEqual(mod.parse_policy("skip", 1024), 0)
        self.assertEqual(mod.parse_policy("1", 1024), 1)
        self.assertEqual(mod.parse_policy("1024", 1024), 1024)
        invalid_values = (
            "", "OFF", "Skip", " skip", "skip ", "0", "-1", "+1",
            "01", "1025", "no", "1.0", "١",
        )
        for invalid in invalid_values:
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                mod.parse_policy(invalid, 1024)
        for invalid_bound in (0, -1, True, 1.5):
            with self.subTest(bound=invalid_bound), self.assertRaises(ValueError):
                mod.parse_policy("off", invalid_bound)

    def test_embedded_production_helper_implements_off_skip_cap_and_peer_detection(self):
        namespace = {"os": os}
        exec(mod.POLICY_HELPER, namespace)
        policy = namespace["_glm53_mixed_prefill_policy"]
        current = types.SimpleNamespace(
            request_id="prefill", num_computed_tokens=10, num_prompt_tokens=100
        )
        decoding = types.SimpleNamespace(
            request_id="decode", num_computed_tokens=100, num_prompt_tokens=100
        )
        prefilling = types.SimpleNamespace(
            request_id="peer-prefill", num_computed_tokens=99, num_prompt_tokens=100
        )
        for raw, expected in (("off", None), ("skip", 0), ("128", 128)):
            with self.subTest(raw=raw), mock.patch.dict(
                os.environ, {mod.POLICY_ENV: raw}, clear=False
            ):
                self.assertEqual(policy([current, decoding], current, 1024), expected)
        with mock.patch.dict(os.environ, {mod.POLICY_ENV: "skip"}, clear=False):
            self.assertIsNone(policy([current, prefilling], current, 1024))
            same_id = types.SimpleNamespace(
                request_id="prefill", num_computed_tokens=100, num_prompt_tokens=100
            )
            self.assertIsNone(policy([current, same_id], current, 1024))
        with mock.patch.dict(os.environ, {mod.POLICY_ENV: "garbage"}, clear=False):
            with self.assertRaisesRegex(ValueError, mod.POLICY_ENV):
                policy([decoding], current, 1024)

    def test_missing_environment_defaults_to_off_and_preserves_stock(self):
        namespace = {"os": os}
        exec(mod.POLICY_HELPER, namespace)
        policy = namespace["_glm53_mixed_prefill_policy"]
        decoding = types.SimpleNamespace(
            request_id="decode", num_computed_tokens=2, num_prompt_tokens=1
        )
        current = types.SimpleNamespace(
            request_id="prefill", num_computed_tokens=0, num_prompt_tokens=100
        )
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertIsNone(policy([decoding], current, 1024))

    def test_transform_covers_running_and_waiting_without_breaking_requeue_fairness(self):
        transformed = mod.transform(mod.SCHEDULER_PATH, synthetic_source())
        self.assertIn(b"num_new_tokens = min(num_new_tokens, mixed_prefill_cap)", transformed)
        self.assertIn(b"request_queue.pop_request()", transformed)
        self.assertIn(b"step_skipped_waiting.prepend_request(request)", transformed)
        self.assertIn(b"continue", transformed)
        self.assertNotIn(b"break  # [glm53-mixed-prefill-policy]", transformed)
        compile(transformed, mod.SCHEDULER_PATH, "exec")

    def test_transform_rejects_anchor_drift_and_partial_application(self):
        source = synthetic_source()
        with self.assertRaisesRegex(SystemExit, "anchor mismatch"):
            mod.transform(mod.SCHEDULER_PATH, source.replace(mod.RUNNING_OLD, b"drifted\n"))
        partial = source.replace(mod.RUNNING_OLD, mod.RUNNING_NEW)
        with self.assertRaisesRegex(SystemExit, "anchor mismatch"):
            mod.transform(mod.SCHEDULER_PATH, partial)

    def test_exact_local_runtime_hashes_and_cli_are_frozen(self):
        self.assertEqual(mod.SCHEDULER_PATH, "vllm/v1/core/sched/scheduler.py")
        self.assertEqual(mod.ORIGINAL_SHA256, {mod.SCHEDULER_PATH: EXPECTED_ORIGINAL})
        self.assertEqual(mod.PATCHED_SHA256, {mod.SCHEDULER_PATH: EXPECTED_PATCHED})
        result = subprocess.run(
            [sys.executable, str(SCRIPT), "--help"],
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("site_packages", result.stdout)
        self.assertIn("--verify", result.stdout)


class MixedPrefillTransactionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="mixed-prefill-")
        self.root = Path(self.temporary.name)
        self.original = synthetic_source()
        self.patched = mod.transform(mod.SCHEDULER_PATH, self.original)
        self.path = self.root / mod.SCHEDULER_PATH
        self.path.parent.mkdir(parents=True)
        self.path.write_bytes(self.original)
        self.original_hashes = {
            mod.SCHEDULER_PATH: hashlib.sha256(self.original).hexdigest()
        }
        self.patched_hashes = {
            mod.SCHEDULER_PATH: hashlib.sha256(self.patched).hexdigest()
        }

    def tearDown(self):
        self.temporary.cleanup()

    def apply(self, **kwargs):
        return mod.apply_overlay(
            self.root,
            original_hashes=self.original_hashes,
            patched_hashes=self.patched_hashes,
            **kwargs,
        )

    def test_original_patches_verify_only_requires_patched_and_apply_is_idempotent(self):
        with self.assertRaisesRegex(SystemExit, "not patched"):
            self.apply(verify_only=True)
        self.assertEqual(self.apply(), "patched")
        self.assertEqual(self.path.read_bytes(), self.patched)
        self.assertEqual(self.apply(verify_only=True), "verified")
        before_mtime = self.path.stat().st_mtime_ns
        self.assertEqual(self.apply(), "already_patched")
        self.assertEqual(self.path.stat().st_mtime_ns, before_mtime)

    def test_drift_and_partial_source_are_rejected_without_writes(self):
        for data, message in (
            (b"unrecognized\n", "unknown source revision"),
            (self.original.replace(mod.RUNNING_OLD, mod.RUNNING_NEW), "partial"),
        ):
            with self.subTest(message=message):
                self.path.write_bytes(data)
                before = self.path.read_bytes()
                with self.assertRaisesRegex(SystemExit, message):
                    self.apply()
                self.assertEqual(self.path.read_bytes(), before)

    def test_missing_and_symlink_sources_are_rejected(self):
        self.path.unlink()
        with self.assertRaisesRegex(SystemExit, "missing source"):
            self.apply()
        target = self.root / "real.py"
        target.write_bytes(self.original)
        self.path.symlink_to(target)
        with self.assertRaisesRegex(SystemExit, "symlink"):
            self.apply()

    def test_candidate_is_compiled_before_write(self):
        writes = []
        with self.assertRaisesRegex(SystemExit, "does not compile"):
            self.apply(
                compiler=lambda data, relative: (_ for _ in ()).throw(
                    SyntaxError("injected compile failure")
                ),
                writer=lambda path, data: writes.append(path),
            )
        self.assertEqual(writes, [])
        self.assertEqual(self.path.read_bytes(), self.original)

    def test_post_replace_failure_rolls_back_the_attempted_file(self):
        def replace_then_fail(path, data):
            mod._atomic_write(path, data)
            raise OSError("injected post-replace durability failure")

        with self.assertRaisesRegex(SystemExit, "rolled back"):
            self.apply(writer=replace_then_fail)
        self.assertEqual(self.path.read_bytes(), self.original)

    def test_post_commit_hash_mismatch_rolls_back(self):
        def corrupting_writer(path, data):
            mod._atomic_write(path, data + b"# corruption\n")

        with self.assertRaisesRegex(SystemExit, "rolled back"):
            self.apply(writer=corrupting_writer)
        self.assertEqual(self.path.read_bytes(), self.original)

    def test_atomic_writer_preserves_mode_and_ownership(self):
        self.path.chmod(0o640)
        before = self.path.stat(follow_symlinks=False)
        mod._atomic_write(self.path, self.patched)
        after = self.path.stat(follow_symlinks=False)
        self.assertEqual(after.st_mode & 0o7777, before.st_mode & 0o7777)
        self.assertEqual((after.st_uid, after.st_gid), (before.st_uid, before.st_gid))


if __name__ == "__main__":
    unittest.main(verbosity=2)
