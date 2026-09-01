#!/usr/bin/env python3
"""Compare matched privacy-safe benchmark receipts against explicit gates."""
import argparse, json, statistics

def _median(runs, path):
 values=[]
 for run in runs:
  value=run
  for key in path: value=value.get(key) if isinstance(value,dict) else None
  if isinstance(value,(int,float)): values.append(float(value))
 return statistics.median(values) if values else None

def compare(control,candidate,regression_limit=0.03):
 c_runs=control.get('runs'); a_runs=candidate.get('runs')
 if not isinstance(c_runs,list) or not isinstance(a_runs,list) or not c_runs or len(c_runs)!=len(a_runs):
  raise ValueError('matched non-empty run counts are required')
 identity=('fixture_id','fixture_sha256','warmup_count')
 expected=tuple(c_runs[0].get(k) for k in identity)
 if any(tuple(r.get(k) for k in identity)!=expected for r in c_runs+a_runs):
  raise ValueError('fixture identity or warmup policy mismatch')
 paths={
  'ttft_seconds':('ttft_seconds',),
  'decode_tokens_per_second':('decode_tokens_per_second',),
  'mtp_acceptance_rate':('mtp','acceptance_rate'),
  'mtp_accepted_tokens_per_step':('mtp','accepted_tokens_per_step'),
 }
 metrics={}; failed=[]
 for name,path in paths.items():
  before=_median(c_runs,path); after=_median(a_runs,path)
  change=(after-before)/before if before not in (None,0) and after is not None else None
  metrics[name]={'control_median':before,'candidate_median':after,'relative_change':change}
  if name=='ttft_seconds' and (change is None or change>regression_limit): failed.append(name)
  if name=='decode_tokens_per_second' and (change is None or change < -regression_limit): failed.append(name)
 return {'schema_version':1,'matched_fixture':True,'fixture':dict(zip(identity,expected)),'run_count':len(c_runs),'regression_limit':regression_limit,'metrics':metrics,'failed_metrics':failed,'gate_pass':not failed}

def main(argv=None):
 p=argparse.ArgumentParser(description=__doc__); p.add_argument('--control',required=True); p.add_argument('--candidate',required=True); p.add_argument('--regression-limit',type=float,default=0.03); args=p.parse_args(argv)
 if not 0<=args.regression_limit<1: p.error('--regression-limit must be in [0,1)')
 with open(args.control) as f: control=json.load(f)
 with open(args.candidate) as f: candidate=json.load(f)
 result=compare(control,candidate,args.regression_limit)
 print(json.dumps(result,sort_keys=True,indent=2,allow_nan=False))
 return 0 if result['gate_pass'] else 1

if __name__=='__main__': raise SystemExit(main())
