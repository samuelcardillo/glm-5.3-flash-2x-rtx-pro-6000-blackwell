#!/usr/bin/env python3
import importlib.util
import pathlib
import unittest

HERE = pathlib.Path(__file__).resolve().parent


def load(filename, name):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class RuntimeProbeTests(unittest.TestCase):
    def test_mixed_timeline_requires_a_first_token_before_b_submission(self):
        mixed = load("bench-mixed-prefill.py", "mixed")
        events = [
            ("a_submitted", 0.0), ("a_token", 1.0), ("a_token", 1.2),
            ("b_submitted", 1.1), ("a_token", 1.5), ("b_first_token", 3.1),
            ("a_finished", 2.0), ("b_finished", 3.2),
        ]
        result = mixed.analyze_timeline(events, a_completion_tokens=4)
        self.assertEqual(result["a_ttft_seconds"], 1.0)
        self.assertEqual(result["b_ttft_seconds"], 2.0)
        self.assertEqual(result["b_queue_seconds"], 2.0)
        self.assertAlmostEqual(result["a_overlap_itl_seconds"], 0.3)
        self.assertEqual(result["a_overlap_tokens_per_second"], 1 / 0.3)
        self.assertEqual(result["a_active_decode_tokens_per_second"], 3.0)
        self.assertEqual(result["combined_makespan_seconds"], 3.2)
        with self.assertRaisesRegex(ValueError, "first token"):
            mixed.analyze_timeline([("a_submitted", 0), ("b_submitted", .5), ("a_token", 1)], 1)

    def test_distinct_prefix_mode_rejects_identical_inputs(self):
        mixed = load("bench-mixed-prefill.py", "mixed_distinct")
        mixed.require_distinct_prefixes(b"decoder", b"prefill")
        with self.assertRaisesRegex(ValueError, "distinct"):
            mixed.require_distinct_prefixes(b"same", b"same")

    def test_repetition_detector_finds_bounded_periodic_loops(self):
        probe = load("verify-long-generation.py", "long_probe")
        loop = list("abcdefgh") * 4
        finding = probe.detect_repetition(loop, min_period=2, max_period=16, repeats=4)
        self.assertEqual(finding["period"], 8)
        self.assertIsNone(probe.detect_repetition(list("abcdefghijklmnop"), 2, 8, 3))

    def test_long_generation_requires_over_2048_tokens_strict_finish_and_health(self):
        probe = load("verify-long-generation.py", "long_validation")
        parsed = {"usage": {"completion_tokens": 2049}, "finish_reason": "length", "content": "ok", "reasoning": "", "reasoning_content": ""}
        probe.validate_long_result(parsed, requested_max_tokens=2049, health_after=True)
        for bad, requested, health in [
            ({**parsed, "usage": {"completion_tokens": 2048}}, 2049, True),
            (parsed, 2048, True),
            (parsed, 2049, False),
            ({**parsed, "finish_reason": None}, 2049, True),
        ]:
            with self.subTest(bad=bad, requested=requested, health=health), self.assertRaises(ValueError):
                probe.validate_long_result(bad, requested, health)


if __name__ == "__main__":
    unittest.main(verbosity=2)
