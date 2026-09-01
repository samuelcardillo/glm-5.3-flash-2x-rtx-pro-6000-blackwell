#!/usr/bin/env python3
"""Recipe tests for the pinned derived runtime image build."""
from __future__ import annotations

import hashlib
import os
import re
import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCKERFILE = ROOT / "Dockerfile.runtime-fixes"
DOCKERFILE_A2 = ROOT / "Dockerfile.runtime-fixes-a2"
DOCKERFILE_A3 = ROOT / "Dockerfile.runtime-fixes-a3"
BUILD_SCRIPT = ROOT / "scripts" / "build-runtime-image.sh"
BUILD_SCRIPT_A2 = ROOT / "scripts" / "build-runtime-image-a2.sh"
BUILD_SCRIPT_A3 = ROOT / "scripts" / "build-runtime-image-a3.sh"
PATCH_SCRIPT = ROOT / "runtime" / "apply-xgrammar-fixes.py"
WORKSPACE_PATCH_SCRIPT = ROOT / "runtime" / "apply-sparse-indexer-workspace.py"
MIXED_PREFILL_PATCH_SCRIPT = ROOT / "runtime" / "apply-mixed-prefill-policy.py"
SERVE_SCRIPT = ROOT / "scripts" / "serve.sh"
BASE = (
    "ghcr.io/tpurtell/glm-5.3-flash-exl3-4bpw-2x-rtx@"
    "sha256:da5cec95778bf6996660b52e28a6e51737fec69cfc3d508bf298c8a89f273ac5"
)
COMMITS = (
    "12f64b39d29282437e35be9aa5db432fb2a1a6e6",
    "c6e19b3be24338759a443e03c8325d76da9ee202",
)
A1_RECIPE = "e91aebecd2907d9905c6f4520c30d49fa57f4272e9e738d46c0d3edccf3d35fc"
A1_IMAGE = f"local/glm53-runtime-fixes:{A1_RECIPE}"
A1_DIGEST = "sha256:51279269e9deb57082d186c07eddfac1567d533d942ce5b651d9c27a0acfbb7f"
A1_IMMUTABLE_IMAGE = f"local/glm53-runtime-fixes@{A1_DIGEST}"
A2_RECIPE = "4ef38e761892c69e7c8e90748dbc362dca405cd6cde756b4afafb68cc0babd39"
A2_IMAGE = f"local/glm53-runtime-fixes:{A2_RECIPE}"
A2_DIGEST = "sha256:ca68a67e14b77c4291a19925d7ff262ff63805cdd90834250ab7a6d7438a54a6"
A2_IMMUTABLE_IMAGE = f"local/glm53-runtime-fixes@{A2_DIGEST}"


def framed_recipe_hash(items):
    digest = hashlib.sha256()
    for name, path in items:
        name_bytes = name.encode()
        data = path.read_bytes()
        digest.update(len(name_bytes).to_bytes(8, "big"))
        digest.update(name_bytes)
        digest.update(len(data).to_bytes(8, "big"))
        digest.update(data)
    return digest.hexdigest()


class RuntimeImageRecipeTests(unittest.TestCase):
    def test_dockerfile_uses_only_the_exact_digest_pinned_base(self):
        text = DOCKERFILE.read_text()
        from_lines = re.findall(r"(?m)^FROM\s+(\S+)", text)
        self.assertEqual(from_lines, [BASE])
        self.assertNotRegex(from_lines[0], r":(?:latest|main|master)(?:@|$)")

    def test_dockerfile_does_not_require_buildkit_only_copy_flags(self):
        text = DOCKERFILE.read_text()
        self.assertNotIn("COPY --chmod", text)

    def test_recipe_declares_commits_labels_and_installed_source_verification(self):
        text = DOCKERFILE.read_text()
        for commit in COMMITS:
            self.assertIn(commit, text)
        self.assertIn("org.opencontainers.image.base.digest", text)
        self.assertIn("io.github.glm53.runtime.recipe.sha256", text)
        self.assertIn("--verify", text)
        self.assertIn("apply-xgrammar-fixes.py", text)

    def test_recipe_hash_is_deterministic_and_covers_both_recipe_inputs(self):
        first = subprocess.check_output(
            [str(BUILD_SCRIPT), "--print-recipe-hash"], text=True, cwd=ROOT
        ).strip()
        second = subprocess.check_output(
            [str(BUILD_SCRIPT), "--print-recipe-hash"], text=True, cwd=ROOT
        ).strip()
        expected = hashlib.sha256(
            b"Dockerfile.runtime-fixes\0"
            + DOCKERFILE.read_bytes()
            + b"\0runtime/apply-xgrammar-fixes.py\0"
            + PATCH_SCRIPT.read_bytes()
        ).hexdigest()
        self.assertEqual(first, expected)
        self.assertEqual(second, expected)
        self.assertRegex(first, r"^[0-9a-f]{64}$")

    def test_build_path_fails_closed_and_verifies_labels_and_source(self):
        text = BUILD_SCRIPT.read_text()
        self.assertIn("set -euo pipefail", text)
        self.assertIn("docker image inspect", text)
        self.assertIn("--network none", text)
        self.assertIn("--cap-drop ALL", text)
        self.assertIn("--security-opt no-new-privileges", text)
        self.assertIn("--verify", text)
        self.assertNotIn("docker push", text)
        self.assertNotIn("systemctl", text)

    def test_serve_uses_validated_runtime_image_and_checks_derived_label(self):
        text = SERVE_SCRIPT.read_text()
        self.assertNotIn('IMAGE="ghcr.io/', text)
        self.assertIn('IMAGE="$RUNTIME_IMAGE"', text)
        self.assertIn("io.github.glm53.runtime.recipe.sha256", text)
        self.assertIn("${RUNTIME_IMAGE##*:}", text)

    def test_a2_recipe_layers_workspace_on_exact_a1_parent(self):
        text = DOCKERFILE_A2.read_text()
        self.assertEqual(
            re.findall(r"(?m)^FROM\s+(\S+)", text), [A1_IMMUTABLE_IMAGE]
        )
        self.assertNotIn("COPY runtime/apply-xgrammar-fixes.py", text)
        self.assertNotIn("python3 /usr/local/share/runtime-fixes/apply-xgrammar-fixes.py", text)
        self.assertIn("apply-sparse-indexer-workspace.py", text)
        self.assertEqual(text.count("--verify"), 1)
        self.assertIn("io.github.glm53.runtime.parent.recipe.sha256", text)
        self.assertIn(A1_RECIPE, text)
        self.assertIn("io.github.glm53.runtime.workspace", text)

    def test_a2_recipe_hash_covers_parent_recipe_dockerfile_overlay_and_build_procedure(self):
        actual = subprocess.check_output(
            [str(BUILD_SCRIPT_A2), "--print-recipe-hash"], text=True, cwd=ROOT
        ).strip()
        expected = framed_recipe_hash((
            ("Dockerfile.runtime-fixes-a2", DOCKERFILE_A2),
            ("runtime/apply-sparse-indexer-workspace.py", WORKSPACE_PATCH_SCRIPT),
            ("scripts/build-runtime-image-a2.sh", BUILD_SCRIPT_A2),
        ))
        self.assertEqual(actual, expected)
        self.assertRegex(actual, r"^[0-9a-f]{64}$")
        build = BUILD_SCRIPT_A2.read_text()
        self.assertIn(A1_DIGEST, build)
        self.assertIn("--format '{{.Id}}'", build)
        for required in ("docker image inspect", "--network none", "--cap-drop ALL", "--security-opt no-new-privileges", "apply-xgrammar-fixes.py", "apply-sparse-indexer-workspace.py"):
            self.assertIn(required, build)
        self.assertNotIn("docker push", build)
        self.assertNotIn("systemctl", build)

    def test_a3_recipe_layers_mixed_prefill_only_on_exact_a2_parent(self):
        text = DOCKERFILE_A3.read_text()
        self.assertEqual(
            re.findall(r"(?m)^FROM\s+(\S+)", text), [A2_IMMUTABLE_IMAGE]
        )
        self.assertIn("apply-mixed-prefill-policy.py", text)
        self.assertNotIn("COPY runtime/apply-xgrammar-fixes.py", text)
        self.assertNotIn("COPY runtime/apply-sparse-indexer-workspace.py", text)
        self.assertEqual(text.count("--verify"), 1)
        self.assertIn("io.github.glm53.runtime.parent.recipe.sha256", text)
        self.assertIn(A2_RECIPE, text)
        self.assertIn("io.github.glm53.runtime.mixed-prefill", text)

    def test_a3_recipe_hash_covers_parent_dockerfile_overlay_and_build_procedure(self):
        actual = subprocess.check_output(
            [str(BUILD_SCRIPT_A3), "--print-recipe-hash"], text=True, cwd=ROOT
        ).strip()
        expected = framed_recipe_hash((
            ("Dockerfile.runtime-fixes-a3", DOCKERFILE_A3),
            ("runtime/apply-mixed-prefill-policy.py", MIXED_PREFILL_PATCH_SCRIPT),
            ("scripts/build-runtime-image-a3.sh", BUILD_SCRIPT_A3),
        ))
        self.assertEqual(actual, expected)
        self.assertRegex(actual, r"^[0-9a-f]{64}$")
        build = BUILD_SCRIPT_A3.read_text()
        self.assertIn(A2_DIGEST, build)
        self.assertIn("--format '{{.Id}}'", build)
        for required in (
            "docker image inspect", "--network none", "--cap-drop ALL",
            "--security-opt no-new-privileges", "apply-xgrammar-fixes.py",
            "apply-sparse-indexer-workspace.py", "apply-mixed-prefill-policy.py",
        ):
            self.assertIn(required, build)
        self.assertNotIn("docker push", build)
        self.assertNotIn("systemctl", build)

    def test_serve_passes_validated_mixed_prefill_policy_to_runtime(self):
        text = SERVE_SCRIPT.read_text()
        self.assertIn('--env GLM53_MIXED_PREFILL_CHUNK="$MIXED_PREFILL_CHUNK"', text)

    def test_scripts_are_executable(self):
        self.assertTrue(os.access(BUILD_SCRIPT, os.X_OK))
        self.assertTrue(os.access(BUILD_SCRIPT_A2, os.X_OK))
        self.assertTrue(os.access(BUILD_SCRIPT_A3, os.X_OK))
        self.assertTrue(os.access(PATCH_SCRIPT, os.X_OK))
        self.assertTrue(os.access(WORKSPACE_PATCH_SCRIPT, os.X_OK))
        self.assertTrue(os.access(MIXED_PREFILL_PATCH_SCRIPT, os.X_OK))
        self.assertTrue(os.access(Path(__file__), os.X_OK))


if __name__ == "__main__":
    unittest.main(verbosity=2)
