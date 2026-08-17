#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, requests

ap=argparse.ArgumentParser()
ap.add_argument('--base',default='http://127.0.0.1:8000')
ap.add_argument('--model',default='qwen3.8-27b-w8a8')
a=ap.parse_args()
base=a.base.rstrip('/')
models=requests.get(base+'/v1/models',timeout=10); models.raise_for_status()
print('models:',[x['id'] for x in models.json().get('data',[])])
r=requests.post(base+'/v1/chat/completions',json={
    'model':a.model,
    'messages':[{'role':'user','content':'Reply with exactly: Q38_OK'}],
    'temperature':0,'max_tokens':16,
},timeout=120)
r.raise_for_status(); d=r.json(); text=d['choices'][0]['message']['content']
print('response:',repr(text))
print('usage:',json.dumps(d.get('usage',{}),ensure_ascii=False))
if 'Q38_OK' not in text: raise SystemExit('unexpected smoke response')
print('PASS')
