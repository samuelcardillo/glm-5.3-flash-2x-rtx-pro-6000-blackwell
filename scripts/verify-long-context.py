#!/usr/bin/env python3
"""Build exact-token chat prompts through vLLM /tokenize and test retrieval."""
from __future__ import annotations
import argparse,json,time,urllib.request

def post(base,path,payload,timeout):
    req=urllib.request.Request(base.rstrip('/')+path,data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(req,timeout=timeout) as response:return json.load(response)

def token_count(base,model,content,timeout):
    data=post(base,'/tokenize',{'model':model,'messages':[{'role':'user','content':content}]},timeout)
    return int(data['count'])

def exact_prompt(base,model,target,needle,timeout):
    prefix='A passphrase appears exactly once below. Ignore all filler and reply with only the passphrase.\nBEGIN FILLER'
    marker=f'\nPASSCODE: {needle}\n'
    suffix='\nEND FILLER\nWhat is the passphrase? Reply with only the passphrase.'
    filler=max(0,target-token_count(base,model,prefix+marker+suffix,timeout))
    for _ in range(8):
        left=filler//2; content=prefix+(' x'*left)+marker+(' x'*(filler-left))+suffix
        count=token_count(base,model,content,timeout); delta=target-count
        if delta==0:return content,count
        filler+=delta
        if filler<0:raise RuntimeError('Target is smaller than fixed prompt overhead')
    raise RuntimeError(f'Could not converge to exactly {target} tokens; last count was {count}')

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--base-url',default='http://127.0.0.1:8000')
    ap.add_argument('--model',default='glm-5.3-flash-local')
    ap.add_argument('--targets',default='128000,261900',help='comma-separated exact chat prompt token counts')
    ap.add_argument('--timeout',type=int,default=600)
    ap.add_argument('--needle',default='GLM53_NEEDLE_739184')
    args=ap.parse_args(); results=[]
    for target in [int(x) for x in args.targets.split(',') if x]:
        content,count=exact_prompt(args.base_url,args.model,target,args.needle,args.timeout)
        payload={'model':args.model,'messages':[{'role':'user','content':content}],'max_tokens':64,'temperature':0,'chat_template_kwargs':{'enable_thinking':False}}
        started=time.time(); data=post(args.base_url,'/v1/chat/completions',payload,args.timeout); elapsed=round(time.time()-started,3)
        answer=(data['choices'][0]['message'].get('content')or'').strip(); usage=data.get('usage')or{}
        row={'target_prompt_tokens':target,'tokenize_count':count,'server_prompt_tokens':usage.get('prompt_tokens'),'answer':answer,'seconds':elapsed,'pass':answer==args.needle and usage.get('prompt_tokens')==target}
        results.append(row); print(json.dumps(row),flush=True)
    print(json.dumps({'results':results},indent=2))
    raise SystemExit(0 if all(row['pass'] for row in results) else 1)
if __name__=='__main__':main()
