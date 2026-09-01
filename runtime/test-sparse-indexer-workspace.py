#!/usr/bin/env python3
"""Fail-closed tests for the DCP-aware sparse-indexer A2 overlay."""
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

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "runtime" / "apply-sparse-indexer-workspace.py"
SPEC = importlib.util.spec_from_file_location("apply_sparse_indexer_workspace", SCRIPT)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"could not load {SCRIPT}")
mod = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mod)


def synthetic_source(relative, edits):
    source = b"\n".join(
        edit.old for edit in edits for _ in range(edit.expected_count)
    )
    if relative == "vllm/v1/attention/backends/mla/indexer.py":
        source += b"\n" + mod.KPOOL_REMAP_ANCHOR
    return source


class SparseIndexerOverlayTests(unittest.TestCase):
    def test_cli_is_wired_for_apply_and_verify(self):
        result = subprocess.run(
            [sys.executable, str(SCRIPT), "--help"],
            text=True, capture_output=True, check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("site_packages", result.stdout)
        self.assertIn("--verify", result.stdout)

    def test_embedded_production_helper_matches_frozen_caps_and_rejects_interleave(self):
        namespace = {"VllmConfig": object}
        exec(mod.CAP_HELPER_NEW, namespace)
        helper = namespace["get_sparse_indexer_prefill_caps"]
        config = types.SimpleNamespace(
            model_config=types.SimpleNamespace(max_model_len=262144),
            scheduler_config=types.SimpleNamespace(max_num_seqs=16),
            parallel_config=types.SimpleNamespace(
                decode_context_parallel_size=2,
                cp_kv_cache_interleave_size=1,
            ),
        )
        self.assertEqual(helper(config, 4), (1_048_576, 524_288))
        config.parallel_config.cp_kv_cache_interleave_size = 2
        with self.assertRaisesRegex(ValueError, "interleave"):
            helper(config, 4)

    def test_transformed_bounds_do_not_use_optimizable_asserts_and_preserve_kpool_remap(self):
        transformed = mod.transform(
            "vllm/v1/attention/backends/mla/indexer.py",
            b"\n".join(
                edit.old
                for edit in mod.PATCHES["vllm/v1/attention/backends/mla/indexer.py"]
                for _ in range(edit.expected_count)
            ) + b"\n" + mod.KPOOL_REMAP_ANCHOR,
        )
        self.assertNotIn(b"assert local_total_seq_lens", transformed)
        self.assertIn(mod.KPOOL_REMAP_ANCHOR, transformed)

    def test_overlay_is_pinned_to_the_three_exact_base_image_sources(self):
        self.assertEqual(
            set(mod.PATCHES),
            {
                "vllm/v1/attention/backends/mla/indexer.py",
                "vllm/models/glm5next/nvidia/attention.py",
                "vllm/model_executor/layers/sparse_attn_indexer_kpool.py",
            },
        )
        self.assertTrue(all(mod.PATCHES[path] for path in mod.PATCHES))
        self.assertEqual(
            mod.ORIGINAL_SHA256,
            {
                "vllm/v1/attention/backends/mla/indexer.py":
                    "ef32db87f3735cf72adfac7cd8906a8951c69178e7eb1c0903965f2a1fca796c",
                "vllm/models/glm5next/nvidia/attention.py":
                    "7d42294c63ea52a7845a6ad0289ae3596098f5aee857718cfed34d6dcf0418f2",
                "vllm/model_executor/layers/sparse_attn_indexer_kpool.py":
                    "e7935f2a9b8ecbc75410defff571d9940ad2abf511286bf0936d0c324936df39",
            },
        )
        self.assertEqual(
            mod.PATCHED_SHA256,
            {
                "vllm/v1/attention/backends/mla/indexer.py":
                    "8d310b4782c3c9fbcf2d58e6f14236cbf3176eb0759289602a863754c1e5006c",
                "vllm/models/glm5next/nvidia/attention.py":
                    "085d856ee518cc5ad0aebb3e3f69803d250b19e6fff3741b2390f178b08d0e92",
                "vllm/model_executor/layers/sparse_attn_indexer_kpool.py":
                    "0324447527369e9958f30136826c7704d4dbfe62aa82957ae7fb1f29f5bec68d",
            },
        )

    def test_transformed_anchors_keep_global_chunking_separate_from_local_allocation(self):
        transformed = {
            path: mod.transform(path, synthetic_source(path, edits))
            for path, edits in mod.PATCHES.items()
        }
        indexer = transformed["vllm/v1/attention/backends/mla/indexer.py"]
        model = transformed["vllm/models/glm5next/nvidia/attention.py"]
        op = transformed[
            "vllm/model_executor/layers/sparse_attn_indexer_kpool.py"
        ]
        self.assertIn(b"def get_max_prefill_buffer_size", indexer)
        self.assertIn(b"def get_sparse_indexer_prefill_caps", indexer)
        self.assertIn(b"self.max_prefill_buffer_size,", indexer)
        self.assertIn(
            b"max_local_total_seq_lens=self.max_prefill_local_buffer_size",
            indexer,
        )
        self.assertIn(b"self.max_local_total_seq_len,", model)
        self.assertNotIn(
            b"self.max_total_seq_len,\n            self.topk_indices_buffer", model
        )
        self.assertIn(b"self.max_local_total_seq_len", op)
        self.assertNotIn(b"own_block * index_kpool", b"\n".join(transformed.values()))


class SparseIndexerTransactionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="sparse-indexer-a2-")
        self.root = Path(self.temporary.name)
        self.original = {}
        self.patched = {}
        for relative, edits in mod.PATCHES.items():
            source = synthetic_source(relative, edits)
            target = mod.transform(relative, source)
            self.original[relative] = source
            self.patched[relative] = target
            path = self.root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(source)
        self.original_hashes = {
            path: hashlib.sha256(data).hexdigest()
            for path, data in self.original.items()
        }
        self.patched_hashes = {
            path: hashlib.sha256(data).hexdigest()
            for path, data in self.patched.items()
        }

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def apply(self, **kwargs):
        return mod.apply_overlay(
            self.root,
            original_hashes=self.original_hashes,
            patched_hashes=self.patched_hashes,
            compiler=lambda data, relative: None,
            **kwargs,
        )

    def snapshot(self):
        return {path: (self.root / path).read_bytes() for path in mod.PATCHES}

    def test_original_triplet_is_patched_and_second_application_is_idempotent(self):
        with self.assertRaisesRegex(SystemExit, "not patched"):
            self.apply(verify_only=True)
        self.assertEqual(self.apply(), "patched")
        self.assertEqual(self.snapshot(), self.patched)
        self.assertEqual(self.apply(verify_only=True), "verified")
        mtimes = {path: (self.root / path).stat().st_mtime_ns for path in mod.PATCHES}
        self.assertEqual(self.apply(), "already_patched")
        self.assertEqual(
            {path: (self.root / path).stat().st_mtime_ns for path in mod.PATCHES},
            mtimes,
        )

    def test_unknown_and_partial_states_are_rejected_before_writes(self):
        first, second, _ = tuple(mod.PATCHES)
        (self.root / first).write_bytes(b"unknown source")
        before = self.snapshot()
        with self.assertRaisesRegex(SystemExit, "unknown source revision"):
            self.apply()
        self.assertEqual(self.snapshot(), before)
        (self.root / first).write_bytes(self.patched[first])
        before = self.snapshot()
        with self.assertRaisesRegex(SystemExit, "partial"):
            self.apply()
        self.assertEqual(self.snapshot(), before)
        self.assertEqual((self.root / second).read_bytes(), self.original[second])

    def test_missing_and_symlink_sources_are_rejected(self):
        first, second, _ = tuple(mod.PATCHES)
        (self.root / second).unlink()
        with self.assertRaisesRegex(SystemExit, "missing source"):
            self.apply()
        (self.root / second).symlink_to(self.root / first)
        with self.assertRaisesRegex(SystemExit, "symlink"):
            self.apply()

    def test_commit_failure_rolls_back_files_already_replaced(self):
        calls = 0

        def failing_writer(path, data):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise OSError("injected write failure")
            mod._atomic_write(path, data)

        before = self.snapshot()
        with self.assertRaisesRegex(SystemExit, "rolled back"):
            self.apply(writer=failing_writer)
        self.assertEqual(self.snapshot(), before)

    def test_post_replace_failure_rolls_back_replaced_file(self):
        calls = 0

        def replace_then_fail(path, data):
            nonlocal calls
            calls += 1
            mod._atomic_write(path, data)
            if calls == 2:
                raise OSError("injected post-replace fsync failure")

        before = self.snapshot()
        with self.assertRaisesRegex(SystemExit, "rolled back"):
            self.apply(writer=replace_then_fail)
        self.assertEqual(self.snapshot(), before)

    def test_post_commit_hash_mismatch_rolls_back_all_files(self):
        def corrupting_writer(path, data):
            mod._atomic_write(path, data + b"# corruption\n")

        before = self.snapshot()
        with self.assertRaisesRegex(SystemExit, "rolled back"):
            self.apply(writer=corrupting_writer)
        self.assertEqual(self.snapshot(), before)

    def test_atomic_writer_preserves_mode_and_ownership(self):
        relative = next(iter(mod.PATCHES))
        path = self.root / relative
        path.chmod(0o640)
        before = path.stat(follow_symlinks=False)
        mod._atomic_write(path, b"replacement\n")
        after = path.stat(follow_symlinks=False)
        self.assertEqual(after.st_mode & 0o7777, before.st_mode & 0o7777)
        self.assertEqual((after.st_uid, after.st_gid), (before.st_uid, before.st_gid))

    def test_python_compile_validation_fails_closed(self):
        mod.validate_python(b"value = 1\n", "valid.py")
        with self.assertRaisesRegex(SystemExit, "does not compile"):
            mod.validate_python(b"def broken(:\n", "broken.py")


class SparseIndexerCapTests(unittest.TestCase):
    def test_frozen_configuration_derives_global_and_rank_local_caps(self):
        self.assertEqual(
            mod.derive_workspace_caps(
                max_model_len=262144,
                max_num_seqs=16,
                index_kpool=4,
                dcp_world_size=2,
            ),
            (1_048_576, 524_288),
        )

    def test_compression_floors_before_dcp_rounds_up(self):
        self.assertEqual(
            mod.derive_workspace_caps(
                max_model_len=10,
                max_num_seqs=3,
                index_kpool=4,
                dcp_world_size=4,
            ),
            (6, 3),
        )

    def test_divisible_and_single_rank_boundaries(self):
        self.assertEqual(
            mod.derive_workspace_caps(
                max_model_len=16,
                max_num_seqs=2,
                index_kpool=4,
                dcp_world_size=2,
            ),
            (8, 4),
        )
        self.assertEqual(
            mod.derive_workspace_caps(
                max_model_len=17,
                max_num_seqs=2,
                index_kpool=4,
                dcp_world_size=1,
            ),
            (8, 8),
        )

    def test_interleave_greater_than_one_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "interleave"):
            mod.derive_workspace_caps(
                max_model_len=262144,
                max_num_seqs=16,
                index_kpool=4,
                dcp_world_size=2,
                cp_kv_cache_interleave_size=2,
            )

    def test_non_positive_and_zero_row_inputs_fail_closed(self):
        cases = (
            dict(max_model_len=0, max_num_seqs=1, index_kpool=1, dcp_world_size=1),
            dict(max_model_len=1, max_num_seqs=0, index_kpool=1, dcp_world_size=1),
            dict(max_model_len=1, max_num_seqs=1, index_kpool=0, dcp_world_size=1),
            dict(max_model_len=1, max_num_seqs=1, index_kpool=1, dcp_world_size=0),
            dict(max_model_len=3, max_num_seqs=1, index_kpool=4, dcp_world_size=1),
        )
        for case in cases:
            with self.subTest(case=case), self.assertRaisesRegex(ValueError, "positive"):
                mod.derive_workspace_caps(**case)


if __name__ == "__main__":
    unittest.main(verbosity=2)
