#!/usr/bin/env python3
"""Compare matched mixed-prefill receipts against explicit acceptance gates."""
import argparse,json,pathlib,sys

def number(data,key):
    value=data.get(key)
    if not isinstance(value,(int,float)) or isinstance(value,bool): raise ValueError(f'{key} must be numeric')
    return float(value)

def compare(off,candidate):
    isolated=number(off,'isolated_decode_tokens_per_second_median')
    off_active=number(off,'active_decode_tokens_per_second_median')
    candidate_active=number(candidate,'active_decode_tokens_per_second_median')
    off_makespan=number(off,'combined_makespan_seconds_median')
    candidate_makespan=number(candidate,'combined_makespan_seconds_median')
    if isolated<=off_active or off_makespan<=0 or candidate_makespan<=0: raise ValueError('invalid baseline loss or makespan')
    recovery=(candidate_active-off_active)/(isolated-off_active)
    throughput_loss=1-off_makespan/candidate_makespan
    fixtures_match=(candidate.get('prefill_fixture_sha256')==off.get('prefill_fixture_sha256') and candidate.get('prefill_prompt_tokens')==off.get('prefill_prompt_tokens'))
    ordering=off.get('ordering_valid') is True and candidate.get('ordering_valid') is True
    gates={'decode_loss_recovery':recovery>=0.5,'aggregate_throughput':throughput_loss<=0.10,'no_starvation':ordering,'fixtures_match':fixtures_match}
    return {'schema_version':1,'decode_loss_recovery_fraction':recovery,'aggregate_throughput_loss_fraction':throughput_loss,'gates':gates,'accepted':all(gates.values())}

def main():
    p=argparse.ArgumentParser();p.add_argument('--off',required=True);p.add_argument('--candidate',required=True);p.add_argument('--output');a=p.parse_args()
    result=compare(json.loads(pathlib.Path(a.off).read_text()),json.loads(pathlib.Path(a.candidate).read_text()))
    text=json.dumps(result,sort_keys=True,indent=2)+'\n'
    if a.output:pathlib.Path(a.output).write_text(text)
    sys.stdout.write(text);return 0 if result['accepted'] else 1
if __name__=='__main__':raise SystemExit(main())
