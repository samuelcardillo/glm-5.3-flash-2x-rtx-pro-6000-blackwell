#!/usr/bin/env python3
import importlib.util
import io
import json
import pathlib
import tempfile
import unittest

HERE = pathlib.Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("capture_profile", HERE / "capture-profile.py")


def load_module():
    module = importlib.util.module_from_spec(SPEC)
    SPEC.loader.exec_module(module)
    return module


class CaptureProfileTests(unittest.TestCase):
    def test_redacts_secrets_hosts_gpu_ids_paths_and_request_content(self):
        mod = load_module()
        value = {
            "Authorization": "Bearer super-secret",
            "api_key": "sk-private",
            "host": "privatebox.local",
            "gpu_uuid": "GPU-" + "12345678-abcd",
            "pci_bus_id": "0000" + ":01:00.0",
            "checkpoint_path": "/home/alice/private/model",
            "request_body": {"prompt": "private prompt"},
            "safe": "overlord-testing",
        }
        redacted = mod.redact(value)
        rendered = json.dumps(redacted)
        for forbidden in ("super-secret", "sk-private", "privatebox", "GPU-123", "01:00.0", "/home/alice", "private prompt"):
            self.assertNotIn(forbidden, rendered)
        self.assertEqual(redacted["safe"], "overlord-testing")

    def test_profile_is_deterministic_and_reconstruction_complete(self):
        mod = load_module()
        source = {
            "image": "repo/image@sha256:" + "a" * 64,
            "checkpoint_revision": "0123456789abcdef",
            "runtime_revision": "fedcba9876543210",
            "template_hash": "sha256:" + "b" * 64,
            "launch_args": ["--model", "/private/checkpoint", "--api-key", "secret", "--max-model-len", "262144"],
            "model_alias": "overlord-testing",
            "max_model_len": 262144,
            "service_state": {"active": True, "restarts": 0, "host": "private"},
            "metrics": "vllm:num_requests_running 2\nvllm:gpu_cache_usage_perc 0.5\nprocess_cpu_seconds_total 9\n",
        }
        first = mod.build_profile(source)
        second = mod.build_profile(dict(reversed(list(source.items()))))
        self.assertEqual(mod.canonical_json(first), mod.canonical_json(second))
        self.assertEqual(first["schema_version"], 1)
        self.assertEqual(first["model"]["alias"], "overlord-testing")
        self.assertEqual(first["model"]["max_context_tokens"], 262144)
        self.assertEqual(first["artifacts"]["image"], source["image"])
        self.assertEqual(first["launch_args"][1], "<redacted-path>")
        self.assertEqual(first["launch_args"][3], "<redacted>")
        self.assertEqual(first["metrics"], {"vllm:gpu_cache_usage_perc": 0.5, "vllm:num_requests_running": 2.0})

    def test_launch_args_redact_host_and_gpu_identifiers(self):
        mod = load_module()
        args = ["--host", "privatebox.local", "--worker-id", "GPU-" + "12345678-abcd", "--port", "8000"]
        self.assertEqual(mod.redact_args(args), ["--host", "<redacted>", "--worker-id", "<redacted>", "--port", "8000"])

    def test_launch_args_redact_embedded_urls_and_paths(self):
        mod = load_module()
        args = [
            "--endpoint=https://privatehost.example/v1",
            "DATABASE_URL=postgres://alice:secret@privatehost/db",
            "note=/home/alice/private/model",
        ]
        redacted = mod.redact_args(args)
        rendered = json.dumps(redacted)
        for forbidden in ("privatehost", "alice:secret", "/home/alice"):
            self.assertNotIn(forbidden, rendered)
        self.assertEqual(redacted[0], "--endpoint=<redacted-url>")
        self.assertEqual(redacted[1], "DATABASE_URL=<redacted-url>")
        self.assertEqual(redacted[2], "note=<redacted-path>")

    def test_file_hash_is_content_based(self):
        mod = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "template.jinja"
            path.write_bytes(b"hello\n")
            self.assertEqual(mod.sha256_file(path), "5891b5b522d5df086d0ff0b110fbd9d21bb4fc7163af34d08286a2e846f6be03")


if __name__ == "__main__":
    unittest.main(verbosity=2)
