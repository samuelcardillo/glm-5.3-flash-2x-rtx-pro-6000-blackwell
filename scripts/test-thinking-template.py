#!/usr/bin/env python3
import importlib.util
import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

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
    def test_pinned_official_flash_template_is_vendored(self):
        module = load_module()
        source = module.SOURCE_TEMPLATE.read_bytes()
        self.assertEqual(
            hashlib.sha256(source).hexdigest(),
            "0c4099f3382d6c92700dfb99725025360966fd73032f0ecf32377c0d9e6309c5",
        )
        self.assertEqual(
            module.OFFICIAL_REVISION,
            "a5b45eb41df6402735dedc900be14a42e8d5e538",
        )

    def test_official_tool_result_early_exits_survive_derivation(self):
        module = load_module()
        derived = module.derive(module.SOURCE_TEMPLATE.read_bytes()).decode()
        self.assertEqual(derived.count("if not ns_chk.can_sort -%}{%- break"), 3)
        self.assertIn("if id_of(ns_a.tool_calls[j]) == tc_id", derived)
        self.assertIn("set ns_chk.can_sort = false -%}\n                    {%- break", derived)
        self.assertIn("{{- '<tool_call>' ~ tc.name -}}", derived)

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

    def test_known_previous_output_is_migrated_atomically(self):
        module = load_module()
        source_bytes = (
            f"prefix\n{module.OLD_REASONING}\nmiddle\n"
            f"{module.OLD_GENERATION}\nsuffix\n"
        ).encode()
        derived = module.derive(source_bytes)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.jinja"
            output = root / "derived.jinja"
            source.write_bytes(source_bytes)
            output.write_bytes(b"previous-qualified-output")
            setattr(module, "SOURCE_TEMPLATE", source)
            setattr(module, "BEFORE_SHA256", module.digest(source_bytes))
            setattr(module, "AFTER_SHA256", module.digest(derived))
            setattr(
                module,
                "REPLACEABLE_OUTPUT_SHA256S",
                {module.digest(output.read_bytes())},
            )
            with patch("sys.argv", ["apply-thinking-template.py", str(output)]):
                module.main()
            self.assertEqual(output.read_bytes(), derived)
            self.assertFalse(output.is_symlink())

    def test_frozen_public_hashes(self):
        module = load_module()
        self.assertEqual(module.BEFORE_SHA256, "0c4099f3382d6c92700dfb99725025360966fd73032f0ecf32377c0d9e6309c5")
        self.assertEqual(module.AFTER_SHA256, "058ef635c21b51eebb8abe880319c9186bdb54387c51114c88e9750f1015a8cf")
        self.assertIn(
            "5bcdf9be4e5b4a6cf2017f74f7e0b5c7f91bb814a275438dc678dd48da1f81b5",
            module.REPLACEABLE_OUTPUT_SHA256S,
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
