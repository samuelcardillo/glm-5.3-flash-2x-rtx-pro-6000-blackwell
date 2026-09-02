#!/usr/bin/env python3
import importlib.util
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATCH = ROOT / "scripts" / "apply-thinking-template.py"


def load_module():
    spec = importlib.util.spec_from_file_location("thinking_template", PATCH)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import template derivation")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ThinkingTemplateTests(unittest.TestCase):
    def test_derivation_changes_only_reasoning_and_generation_blocks(self):
        module = load_module()
        source = f"prefix\n{module.OLD_REASONING}\nmiddle\n{module.OLD_GENERATION}\nsuffix\n".encode()
        derived = module.derive(source).decode()
        self.assertNotIn(module.OLD_REASONING, derived)
        self.assertNotIn(module.OLD_GENERATION, derived)
        self.assertIn(module.NEW_REASONING, derived)
        self.assertIn(module.NEW_GENERATION, derived)
        self.assertTrue(derived.startswith("prefix\n"))
        self.assertTrue(derived.endswith("\nsuffix\n"))

    def test_unknown_or_duplicate_blocks_are_rejected(self):
        module = load_module()
        for text in ("unknown", module.OLD_REASONING, module.OLD_GENERATION,
                     module.OLD_REASONING * 2 + module.OLD_GENERATION):
            with self.subTest(text_length=len(text)), self.assertRaises(SystemExit):
                module.derive(text.encode())

    def test_atomic_output_is_regular_and_unknown_existing_output_is_preserved(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "derived" / "chat_template.jinja"
            module.atomic_write(output, b"known\n")
            self.assertTrue(output.is_file())
            self.assertFalse(output.is_symlink())
            self.assertEqual(output.read_bytes(), b"known\n")
            link = Path(directory) / "link"
            link.symlink_to(output)
            with self.assertRaises(SystemExit):
                module.atomic_write(link, b"replacement\n")
            self.assertEqual(output.read_bytes(), b"known\n")

    def test_frozen_public_hashes(self):
        module = load_module()
        self.assertEqual(module.BEFORE_SHA256, "34d5ee66b12fa6446cdae131c352b8f68cd85369e0e6fda115583805fada3891")
        self.assertEqual(module.AFTER_SHA256, "5bcdf9be4e5b4a6cf2017f74f7e0b5c7f91bb814a275438dc678dd48da1f81b5")


if __name__ == "__main__":
    unittest.main(verbosity=2)
