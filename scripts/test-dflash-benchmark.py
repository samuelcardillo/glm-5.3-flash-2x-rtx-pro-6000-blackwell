#!/usr/bin/env python3
import importlib.util
import json
import pathlib
import types
import unittest

HERE = pathlib.Path(__file__).resolve().parent


def load_module(filename="benchmark-dflash2.py"):
    spec = importlib.util.spec_from_file_location("benchmark_under_test", HERE / filename)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import benchmark")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def command(*, tokens=5, target_kv="fp8_ds_mla"):
    speculative = json.dumps({"method": "dflash", "model": "/draft", "num_speculative_tokens": tokens, "kv_cache_dtype": "bfloat16"})
    return ["/model", "--served-model-name", "overlord-testing", "--max-model-len", "1048576", "--kv-cache-dtype", target_kv, "--speculative-config", speculative]


class ProfileIdentityTests(unittest.TestCase):
    def test_exact_container_profile_passes(self):
        module = load_module()
        document = [{"Config": {"Cmd": command()}}]
        module.subprocess.run = lambda *a, **k: types.SimpleNamespace(returncode=0, stdout=json.dumps(document), stderr="")
        observed = module.verify_container_profile("candidate", "overlord-testing", 5)
        self.assertEqual(observed["dflash_tokens"], 5)
        self.assertEqual(observed["kv_cache_dtype"], "fp8_ds_mla")

    def test_wrong_k_or_target_kv_fails(self):
        for document in (
            [{"Config": {"Cmd": command(tokens=4)}}],
            [{"Config": {"Cmd": command(target_kv="bfloat16")}}],
        ):
            with self.subTest(document=document):
                module = load_module()
                module.subprocess.run = lambda *a, **k: types.SimpleNamespace(returncode=0, stdout=json.dumps(document), stderr="")
                with self.assertRaises(RuntimeError):
                    module.verify_container_profile("candidate", "overlord-testing", 5)

    def test_prefill_requires_exact_live_fp8_profile(self):
        module = load_module("benchmark-prefill-v06.py")
        document = [{"Config": {"Cmd": command()}}]
        module.subprocess.run = lambda *a, **k: types.SimpleNamespace(returncode=0, stdout=json.dumps(document), stderr="")
        observed = module.verify_container_profile("candidate", "overlord-testing")
        self.assertEqual(observed["max_model_len"], "1048576")
        bad = [{"Config": {"Cmd": command(target_kv="bfloat16")}}]
        module.subprocess.run = lambda *a, **k: types.SimpleNamespace(returncode=0, stdout=json.dumps(bad), stderr="")
        with self.assertRaises(RuntimeError):
            module.verify_container_profile("candidate", "overlord-testing")


if __name__ == "__main__":
    unittest.main(verbosity=2)
