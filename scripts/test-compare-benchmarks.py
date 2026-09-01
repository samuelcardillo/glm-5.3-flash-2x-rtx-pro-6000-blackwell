#!/usr/bin/env python3
import importlib.util, pathlib, unittest
HERE=pathlib.Path(__file__).resolve().parent
SPEC=importlib.util.spec_from_file_location('compare_benchmarks',HERE/'compare-benchmarks.py')

def load():
 m=importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(m); return m

class CompareTests(unittest.TestCase):
 def test_matched_comparison_and_three_percent_gate(self):
  m=load()
  control={'runs':[{'fixture_id':'f','fixture_sha256':'a','warmup_count':2,'ttft_seconds':1.0,'decode_tokens_per_second':100.0,'mtp':{'acceptance_rate':0.6,'accepted_tokens_per_step':3.0}} for _ in range(5)]}
  candidate={'runs':[{'fixture_id':'f','fixture_sha256':'a','warmup_count':2,'ttft_seconds':1.02,'decode_tokens_per_second':98.0,'mtp':{'acceptance_rate':0.59,'accepted_tokens_per_step':2.95}} for _ in range(5)]}
  result=m.compare(control,candidate,regression_limit=0.03)
  self.assertTrue(result['matched_fixture'])
  self.assertTrue(result['gate_pass'])
  candidate['runs'][0]['fixture_sha256']='wrong'
  with self.assertRaisesRegex(ValueError,'fixture'):
   m.compare(control,candidate,regression_limit=0.03)
 def test_rejects_decode_regression(self):
  m=load()
  def receipt(rate):
   return {'runs':[{'fixture_id':'f','fixture_sha256':'a','warmup_count':2,'ttft_seconds':1.0,'decode_tokens_per_second':rate,'mtp':{'acceptance_rate':0.6,'accepted_tokens_per_step':3.0}} for _ in range(5)]}
  result=m.compare(receipt(100),receipt(95),regression_limit=0.03)
  self.assertFalse(result['gate_pass'])
  self.assertIn('decode_tokens_per_second',result['failed_metrics'])

if __name__=='__main__': unittest.main(verbosity=2)
