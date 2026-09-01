#!/usr/bin/env python3
import importlib.util
import pathlib
import unittest

HERE = pathlib.Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("bench_prefill", HERE / "bench-prefill.py")


def load_module():
    module = importlib.util.module_from_spec(SPEC)
    SPEC.loader.exec_module(module)
    return module


class PrefillBenchmarkTests(unittest.TestCase):
    def test_calibrates_synthetic_prompt_with_tokenize_to_exact_target(self):
        mod = load_module()
        def tokenize(text):
            return len(text.split())
        prompt, count = mod.calibrate_prompt(25, "salt-1", tokenize)
        self.assertEqual(count, 25)
        self.assertEqual(tokenize(prompt), 25)
        self.assertIn("salt-1", prompt)

    def test_run_salts_are_unique_and_reproducible(self):
        mod = load_module()
        salts = [mod.run_salt("public-32k-v1", index) for index in range(4)]
        self.assertEqual(len(set(salts)), 4)
        self.assertEqual(salts, [mod.run_salt("public-32k-v1", index) for index in range(4)])

    def test_cache_metric_delta_uses_only_complete_aligned_blocks(self):
        mod = load_module()
        before = "vllm:prefix_cache_hits_total 100\nvllm:prefix_cache_queries_total 200\n"
        after = "vllm:prefix_cache_hits_total 165\nvllm:prefix_cache_queries_total 280\n"
        result = mod.cache_metric_delta(before, after, block_size=16)
        self.assertEqual(result["hit_tokens_raw"], 65.0)
        self.assertEqual(result["hit_tokens_block_aligned"], 64)
        self.assertEqual(result["queried_tokens"], 80.0)
        self.assertEqual(result["block_aligned_hit_rate"], 0.8)

    def test_unavailable_reset_fails_closed_for_cold_claim(self):
        mod = load_module()
        self.assertEqual(mod.classify_reset(204), {"available": True, "cold_claim_allowed": True})
        for status in (404, 405, 501):
            with self.subTest(status=status):
                self.assertEqual(mod.classify_reset(status), {"available": False, "cold_claim_allowed": False})
        with self.assertRaises(mod.ResetError):
            mod.require_cold_reset(404)

    def test_clean_process_requires_zero_prompt_and_cache_query_counters(self):
        mod = load_module()
        zero = (
            'vllm:prompt_tokens_total{engine="0"} 0\n'
            'vllm:prefix_cache_queries_total{engine="0"} 0\n'
        )
        self.assertEqual(mod.require_clean_process(zero), {"method": "clean-process", "cold_claim_allowed": True})
        for text in (
            'vllm:prompt_tokens_total{engine="0"} 1\nvllm:prefix_cache_queries_total{engine="0"} 0\n',
            'vllm:prompt_tokens_total{engine="0"} 0\nvllm:prefix_cache_queries_total{engine="0"} 1\n',
            'vllm:prefix_cache_queries_total{engine="0"} 0\n',
        ):
            with self.subTest(text=text), self.assertRaises(mod.ResetError):
                mod.require_clean_process(text)

    def test_targets_include_required_exact_context_rungs(self):
        mod = load_module()
        self.assertEqual(mod.PUBLIC_TARGETS, (8192, 32768, 131072, 261900))

    def test_health_gate_fails_closed(self):
        mod = load_module()
        mod.require_health(True, True)
        for before, after in ((False, True), (True, False), (False, False)):
            with self.subTest(before=before, after=after), self.assertRaisesRegex(RuntimeError, "health"):
                mod.require_health(before, after)

    def test_usage_verification_rejects_token_mismatch(self):
        mod = load_module()
        mod.verify_usage(expected_prompt_tokens=100, usage={"prompt_tokens": 100, "completion_tokens": 1})
        with self.assertRaisesRegex(ValueError, "prompt token mismatch"):
            mod.verify_usage(expected_prompt_tokens=100, usage={"prompt_tokens": 99, "completion_tokens": 1})


if __name__ == "__main__":
    unittest.main(verbosity=2)
