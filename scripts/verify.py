#!/usr/bin/env python3
"""No-dependency health, text, tool-call, and semantic-vision test."""
from __future__ import annotations
import argparse,base64,binascii,json,struct,time,urllib.request,zlib
def chunk(kind,data):return struct.pack('>I',len(data))+kind+data+struct.pack('>I',binascii.crc32(kind+data)&0xffffffff)
def make_png():
    w,h=640,384; pix=bytearray([255]*(w*h*3))
    def setp(x,y,c):
        if 0<=x<w and 0<=y<h: pix[(y*w+x)*3:(y*w+x)*3+3]=bytes(c)
    for y in range(65,316):
        for x in range(45,226):setp(x,y,(220,30,30))
    for y in range(65,246):
        for x in range(415,596):
            if (x-505)**2+(y-155)**2<=90**2:setp(x,y,(30,80,220))
    glyphs={'7':['11111','00001','00010','00100','01000','01000','01000'],'3':['11110','00001','00001','01110','00001','00001','11110']}
    ox,oy,s=260,240,14
    for ch in '73':
        for gy,row in enumerate(glyphs[ch]):
            for gx,on in enumerate(row):
                if on=='1':
                    for yy in range(s):
                        for xx in range(s):setp(ox+gx*s+xx,oy+gy*s+yy,(0,0,0))
        ox+=6*s
    raw=b''.join(b'\0'+bytes(pix[y*w*3:(y+1)*w*3]) for y in range(h))
    return b'\x89PNG\r\n\x1a\n'+chunk(b'IHDR',struct.pack('>IIBBBBB',w,h,8,2,0,0,0))+chunk(b'IDAT',zlib.compress(raw,9))+chunk(b'IEND',b'')
def req(base,path,payload=None,timeout=300):
    data=None if payload is None else json.dumps(payload).encode(); r=urllib.request.Request(base.rstrip('/')+path,data=data,headers={'Content-Type':'application/json'}); t=time.time()
    with urllib.request.urlopen(r,timeout=timeout) as x: body=json.load(x) if x.headers.get_content_type()=='application/json' else x.read().decode()
    return body,round(time.time()-t,3)
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--base-url',default='http://127.0.0.1:8000'); ap.add_argument('--model',default='glm-5.3-flash-local'); a=ap.parse_args(); out={}
    _,dt=req(a.base_url,'/health'); out['health']={'pass':True,'seconds':dt}
    no_thinking={'chat_template_kwargs':{'enable_thinking':False}}
    p={'model':a.model,'messages':[{'role':'user','content':'Reply with exactly TEXT_OK'}],'max_tokens':64,'temperature':0,**no_thinking}
    d,dt=req(a.base_url,'/v1/chat/completions',p); ans=(d['choices'][0]['message'].get('content')or'').strip(); out['text']={'pass':ans=='TEXT_OK','answer':ans,'seconds':dt}
    p={'model':a.model,'messages':[{'role':'user','content':'Call record_value immediately with value TOOL_OK.'}],'tools':[{'type':'function','function':{'name':'record_value','description':'Record a value','parameters':{'type':'object','properties':{'value':{'type':'string'}},'required':['value']}}}],'max_tokens':256,'temperature':0,**no_thinking}
    d,dt=req(a.base_url,'/v1/chat/completions',p); calls=d['choices'][0]['message'].get('tool_calls')or[]; ok=bool(calls) and json.loads(calls[0]['function']['arguments']).get('value')=='TOOL_OK'; out['tool']={'pass':ok,'tool_calls':calls,'seconds':dt}
    image=base64.b64encode(make_png()).decode(); p={'model':a.model,'messages':[{'role':'user','content':[{'type':'text','text':'Read the large black number in this image. Answer with only that number.'},{'type':'image_url','image_url':{'url':'data:image/png;base64,'+image}}]}],'max_tokens':128,'temperature':0,**no_thinking}
    d,dt=req(a.base_url,'/v1/chat/completions',p); ans=(d['choices'][0]['message'].get('content')or'').strip(); out['vision']={'pass':ans=='73','answer':ans,'seconds':dt}
    print(json.dumps(out,indent=2)); raise SystemExit(0 if all(v['pass'] for v in out.values()) else 1)
if __name__=='__main__':main()
