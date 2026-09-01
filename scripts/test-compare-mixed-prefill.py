#!/usr/bin/env python3
import importlib.util,json,pathlib,tempfile,unittest
ROOT=pathlib.Path(__file__).resolve().parents[1]
SCRIPT=ROOT/'scripts/compare-mixed-prefill.py'

def load():
 s=importlib.util.spec_from_file_location('compare_mixed',SCRIPT)
 if s is None or s.loader is None: raise RuntimeError('cannot load comparator')
 m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

class CompareMixedTests(unittest.TestCase):
 def test_accepts_half_recovery_without_aggregate_regression(self):
  m=load(); off={'isolated_decode_tokens_per_second_median':100,'active_decode_tokens_per_second_median':20,'combined_makespan_seconds_median':50,'ordering_valid':True,'prefill_fixture_sha256':['a','b'],'prefill_prompt_tokens':[160025]}; cand={**off,'active_decode_tokens_per_second_median':70,'combined_makespan_seconds_median':54}
  result=m.compare(off,cand);self.assertTrue(result['accepted']);self.assertEqual(result['decode_loss_recovery_fraction'],0.625);self.assertAlmostEqual(result['aggregate_throughput_loss_fraction'],1-50/54)
 def test_rejects_low_recovery_starvation_fixture_drift_and_overhead(self):
  m=load();base={'isolated_decode_tokens_per_second_median':100,'active_decode_tokens_per_second_median':20,'combined_makespan_seconds_median':50,'ordering_valid':True,'prefill_fixture_sha256':['a'],'prefill_prompt_tokens':[1]}
  cases=[{**base,'active_decode_tokens_per_second_median':50},{**base,'active_decode_tokens_per_second_median':70,'combined_makespan_seconds_median':56},{**base,'active_decode_tokens_per_second_median':70,'ordering_valid':False},{**base,'active_decode_tokens_per_second_median':70,'prefill_fixture_sha256':['x']}]
  for c in cases:self.assertFalse(m.compare(base,c)['accepted'])
if __name__=='__main__':unittest.main(verbosity=2)
