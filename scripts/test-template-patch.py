#!/usr/bin/env python3
import hashlib
import importlib.util
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "apply-vision-template.py"
SPEC = importlib.util.spec_from_file_location("template_patch", SCRIPT)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"could not load {SCRIPT}")
mod = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mod)


def digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def read_noatime(path: Path) -> bytes:
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOATIME", 0))
    with os.fdopen(fd, "rb") as handle:
        return handle.read()


class TemplatePatchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original = "\n".join(
            ["prefix", mod.OLD_REASONING, mod.OLD, mod.OLD_GENERATION, "suffix"]
        )
        cls.legacy = cls.original.replace(mod.OLD, mod.NEW)
        cls.final = mod.transform(cls.original)
        cls.hashes = {
            "BEFORE_SHA256": digest(cls.original),
            "LEGACY_AFTER_SHA256": digest(cls.legacy),
            "AFTER_SHA256": digest(cls.final),
        }

    def patched_constants(self):
        return [
            mock.patch.object(mod, name, value) for name, value in self.hashes.items()
        ]

    def run_main(self, model: Path, restore: bool = False) -> None:
        argv = [str(SCRIPT)]
        if restore:
            argv.append("--restore")
        argv.append(str(model))
        patches = self.patched_constants()
        with patches[0], patches[1], patches[2], mock.patch.object(sys, "argv", argv):
            mod.main()

    def test_transform_honors_disable_thinking(self):
        patched = mod.transform(self.original)
        self.assertIn("enable_thinking is defined", patched)
        self.assertIn("<think></think>", patched)
        self.assertIn("<|begin_of_image|>", patched)
        self.assertNotIn("unable to process this", patched)

    def test_fresh_original_patch_and_restore(self):
        with tempfile.TemporaryDirectory(prefix="glm53-template-test-") as td:
            model = Path(td)
            target = model / "chat_template.jinja"
            target.write_text(self.original)
            target.chmod(0o640)
            now_ns = time.time_ns()
            expected_atime_ns = now_ns - 10_000_000_000_000
            expected_mtime_ns = now_ns - 100_000_000_000
            os.utime(target, ns=(expected_atime_ns, expected_mtime_ns))
            original_stat = target.stat()
            self.run_main(model)
            patched_stat = target.stat()
            self.assertEqual(patched_stat.st_uid, original_stat.st_uid)
            self.assertEqual(patched_stat.st_gid, original_stat.st_gid)
            self.assertEqual(patched_stat.st_mode & 0o7777, 0o640)
            self.assertEqual(patched_stat.st_atime_ns, expected_atime_ns)
            self.assertEqual(patched_stat.st_mtime_ns, expected_mtime_ns)
            backup = model / "chat_template.text-only.bak.jinja"
            backup_stat = backup.stat()
            self.assertEqual(backup_stat.st_atime_ns, expected_atime_ns)
            self.assertEqual(backup_stat.st_mtime_ns, expected_mtime_ns)
            self.assertEqual(read_noatime(target).decode(), self.final)
            self.assertEqual(
                read_noatime(backup).decode(),
                self.original,
            )
            self.run_main(model, restore=True)
            restored_stat = target.stat()
            self.assertEqual(restored_stat.st_uid, original_stat.st_uid)
            self.assertEqual(restored_stat.st_gid, original_stat.st_gid)
            self.assertEqual(restored_stat.st_mode & 0o7777, 0o640)
            self.assertEqual(restored_stat.st_atime_ns, expected_atime_ns)
            self.assertEqual(restored_stat.st_mtime_ns, expected_mtime_ns)
            self.assertEqual(read_noatime(target).decode(), self.original)
            final_backup_stat = backup.stat()
            self.assertEqual(final_backup_stat.st_atime_ns, expected_atime_ns)
            self.assertEqual(final_backup_stat.st_mtime_ns, expected_mtime_ns)

    def test_legacy_patch_upgrades_without_existing_backup(self):
        with tempfile.TemporaryDirectory(prefix="glm53-template-test-") as td:
            model = Path(td)
            target = model / "chat_template.jinja"
            target.write_text(self.legacy)
            self.run_main(model)
            self.assertEqual(target.read_text(), self.final)
            self.assertEqual(
                (model / "chat_template.text-only.bak.jinja").read_text(),
                self.original,
            )

    def test_failed_atomic_replace_leaves_target_intact(self):
        with tempfile.TemporaryDirectory(prefix="glm53-template-test-") as td:
            model = Path(td)
            target = model / "chat_template.jinja"
            target.write_text(self.original)
            real_replace = os.replace

            def fail_only_target(source, destination):
                if Path(destination) == target:
                    raise OSError("simulated interrupted replacement")
                return real_replace(source, destination)

            with mock.patch.object(mod.os, "replace", side_effect=fail_only_target):
                with self.assertRaisesRegex(OSError, "simulated interrupted"):
                    self.run_main(model)
            self.assertEqual(target.read_text(), self.original)
            self.assertEqual(
                (model / "chat_template.text-only.bak.jinja").read_text(),
                self.original,
            )
            self.assertEqual(list(model.glob(".*.tmp")), [])

    def test_failed_metadata_copy_leaves_target_intact(self):
        with tempfile.TemporaryDirectory(prefix="glm53-template-test-") as td:
            model = Path(td)
            target = model / "chat_template.jinja"
            target.write_text(self.original)
            with mock.patch.object(
                mod.shutil, "copystat", side_effect=OSError("simulated metadata failure")
            ):
                with self.assertRaisesRegex(OSError, "simulated metadata failure"):
                    self.run_main(model)
            self.assertEqual(target.read_text(), self.original)
            self.assertEqual(list(model.glob(".*.tmp")), [])

    def test_atomic_write_preserves_atime_and_mtime(self):
        with tempfile.TemporaryDirectory(prefix="glm53-template-test-") as td:
            directory = Path(td)
            source = directory / "metadata-source"
            target = directory / "target"
            source.write_bytes(b"source")
            target.write_bytes(b"old")
            now_ns = time.time_ns()
            expected_atime_ns = now_ns - 10_000_000_000_000
            expected_mtime_ns = now_ns - 100_000_000_000
            os.utime(source, ns=(expected_atime_ns, expected_mtime_ns))
            replacement = b"replacement"

            mod.atomic_write(target, replacement, mod.digest(replacement), source)

            replaced_stat = target.stat()
            self.assertEqual(replaced_stat.st_atime_ns, expected_atime_ns)
            self.assertEqual(replaced_stat.st_mtime_ns, expected_mtime_ns)
            self.assertEqual(target.read_bytes(), replacement)

    def test_symlink_target_is_rejected(self):
        with tempfile.TemporaryDirectory(prefix="glm53-template-test-") as td:
            model = Path(td)
            actual = model / "actual-template.jinja"
            actual.write_text(self.original)
            (model / "chat_template.jinja").symlink_to(actual)
            with self.assertRaisesRegex(SystemExit, "Refusing symlink template"):
                self.run_main(model)
            self.assertEqual(actual.read_text(), self.original)

    def test_unknown_revision_is_rejected(self):
        with tempfile.TemporaryDirectory(prefix="glm53-template-test-") as td:
            model = Path(td)
            (model / "chat_template.jinja").write_text("unknown")
            patches = self.patched_constants()
            argv = [str(SCRIPT), str(model)]
            with patches[0], patches[1], patches[2], mock.patch.object(sys, "argv", argv):
                with self.assertRaisesRegex(SystemExit, "Refusing unknown template revision"):
                    mod.main()


if __name__ == "__main__":
    unittest.main(verbosity=2)
