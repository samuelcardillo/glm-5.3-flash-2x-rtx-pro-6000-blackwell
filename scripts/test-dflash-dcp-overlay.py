#!/usr/bin/env python3
import hashlib
import importlib.util
import pathlib
import stat
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
PATCHER = ROOT / "overlays/dflash-dcp-block-table/patch-model-runner.py"

FIXTURE = b'''logger = init_logger(__name__)\n\nclass Fixture:\n    def method(self, spec):\n        max_num_blocks = cdiv(\n            block_table_max_model_len, spec.block_size * self.dcp_size\n        )\n'''


class DFlashDcpOverlayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location("dflash_overlay", PATCHER)
        if spec is None or spec.loader is None:
            raise AssertionError(f"missing patcher: {PATCHER}")
        cls.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.module)

    def test_pins_exact_live_source_hash(self):
        self.assertEqual(
            self.module.ORIGINAL_SHA256,
            "87d359028d57eb883849b8d8b03a1f7486c8e1ed5473328564629f8146ac56ed",
        )
        self.assertEqual(
            self.module.PATCHED_SHA256,
            "84aaa80af64c91076422f0f8aa1e859d61c797c4212095fc0a2d73bde873f781",
        )

    def test_transforms_replicated_cache_to_effective_dcp_width(self):
        source = FIXTURE
        patched = self.module.transform(
            source, expected_original_sha256=hashlib.sha256(source).hexdigest()
        )
        self.assertIn(b'dcp_shard_count_override', patched)
        self.assertIn(b'def _effective_cache_parallel_width', patched)
        self.assertIn(
            b'spec.block_size * _effective_cache_parallel_width(spec, self.dcp_size)',
            patched,
        )
        self.assertNotIn(b'spec.block_size * self.dcp_size', patched)

    def test_rejects_source_hash_drift(self):
        with self.assertRaisesRegex(ValueError, "source hash"):
            self.module.transform(FIXTURE, expected_original_sha256="0" * 64)

    def test_rejects_duplicate_anchor(self):
        source = FIXTURE.replace(
            b"spec.block_size * self.dcp_size",
            b"spec.block_size * self.dcp_size + spec.block_size * self.dcp_size",
        )
        with self.assertRaisesRegex(ValueError, "exactly once"):
            self.module.transform(
                source, expected_original_sha256=hashlib.sha256(source).hexdigest()
            )

    def test_apply_is_atomic_mode_preserving_and_idempotent(self):
        source = FIXTURE
        original_sha = hashlib.sha256(source).hexdigest()
        patched = self.module.transform(
            source, expected_original_sha256=original_sha
        )
        patched_sha = hashlib.sha256(patched).hexdigest()
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "model_runner.py"
            path.write_bytes(source)
            path.chmod(0o640)
            self.assertEqual(
                self.module.apply_file(
                    path,
                    expected_original_sha256=original_sha,
                    expected_patched_sha256=patched_sha,
                ),
                "patched",
            )
            self.assertEqual(path.read_bytes(), patched)
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o640)
            self.assertEqual(
                self.module.apply_file(
                    path,
                    expected_original_sha256=original_sha,
                    expected_patched_sha256=patched_sha,
                ),
                "already-patched",
            )
            self.module.verify_file(path, expected_patched_sha256=patched_sha)

    def test_apply_rejects_unknown_file_without_modifying_it(self):
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "model_runner.py"
            path.write_bytes(b"unknown")
            with self.assertRaisesRegex(ValueError, "unknown source state"):
                self.module.apply_file(path)
            self.assertEqual(path.read_bytes(), b"unknown")

    def test_child_image_recipe_applies_and_verifies_exact_patch(self):
        recipe = (PATCHER.parent / "Dockerfile").read_text()
        self.assertIn("ARG BASE_IMAGE", recipe)
        self.assertIn("FROM ${BASE_IMAGE}", recipe)
        self.assertIn("--apply", recipe)
        self.assertIn("--verify", recipe)
        self.assertIn("org.nous.glm53.base-image-id", recipe)
        self.assertIn("org.nous.glm53.overlay-recipe-sha256", recipe)

    def test_builder_pins_parent_and_verifies_effective_image(self):
        builder = (ROOT / "scripts/build-dflash-dcp-overlay.sh").read_text()
        self.assertIn(
            "sha256:fe249b88d091430d8a88cd987d087d556053f0f067a649f2e9ca95895129e82b",
            builder,
        )
        self.assertIn("docker image inspect", builder)
        self.assertIn("docker pull", builder)
        self.assertIn("--network none", builder)
        self.assertIn("--cap-drop ALL", builder)
        self.assertIn("--verify", builder)


if __name__ == "__main__":
    unittest.main(verbosity=2)
