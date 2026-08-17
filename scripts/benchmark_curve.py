#!/usr/bin/env python3
from __future__ import annotations
import argparse, fcntl, json, os, random, re, threading, time, hashlib, requests

ap=argparse.ArgumentParser()
ap.add_argument('--port',type=int,default=8040)
ap.add_argument('--model',default='qwen3.8-27b-w8a8')
ap.add_argument('--lengths',default='512,2048,4096,8192,12288,16384,32768,65536,131072,257900')
ap.add_argument('--output-tokens',type=int,default=256)
ap.add_argument('--out',required=True)
ap.add_argument('--resume',action='store_true')
a=ap.parse_args()
base=f'http://127.0.0.1:{a.port}'
# Fail closed across parallel Qwen3.8 research sessions. Keep the fd alive for
# the whole benchmark so another standard scored run cannot overlap this port.
_lock_path=f'/tmp/qwen38_port{a.port}.score.lock'
_lock_fd=os.open(_lock_path, os.O_CREAT | os.O_RDWR, 0o644)
try:
 fcntl.flock(_lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
except BlockingIOError:
 raise SystemExit(f'ERROR: scored benchmark lock busy: {_lock_path}')
S=requests.Session()
words='kernel memory attention quantization scheduler latency throughput profile tensor cache decode prefill context vector matrix runtime compiler device token model system data reasoning optimization exact benchmark'.split()

def post(path,body,timeout=7200):
 r=S.post(base+path,json=body,timeout=timeout); r.raise_for_status(); return r.json()

def metrics():
 try:
  s=requests.get(base+'/metrics',timeout=2).text
 except Exception:
  return {}
 out={}
 for n in ['num_requests_running','num_requests_waiting','spec_decode_num_draft_tokens_total','spec_decode_num_accepted_tokens_total']:
  m=re.search(rf'^vllm:{n}\{{[^\n]+\}} ([0-9.e+-]+)$',s,re.M)
  out[n]=float(m.group(1)) if m else 0.0
 return out

def wait_idle(limit=180):
 end=time.time()+limit
 while time.time()<end:
  m=metrics()
  if m.get('num_requests_running')==0 and m.get('num_requests_waiting')==0:return True
  time.sleep(.2)
 return False

def make_prompt(target,seed):
 rng=random.Random(seed)
 chunks=[f'Q38-BENCH-{target}-{seed} ']
 count=0
 while count < target+64:
  chunks.append(' '.join(words[rng.randrange(len(words))] for _ in range(512)))
  if len(chunks)%8==0:
   d=post('/tokenize',{'prompt':' '.join(chunks)},120); count=int(d['count'])
 ids=post('/tokenize',{'prompt':' '.join(chunks)},300)['tokens'][:target]
 prompt=post('/detokenize',{'tokens':ids},300)['prompt']
 chk=post('/tokenize',{'prompt':prompt},300)
 if int(chk['count']) != target:
  raise RuntimeError(f'roundtrip {target}->{chk["count"]}')
 return prompt

def run(target):
 prompt=make_prompt(target,380000+target)
 if not wait_idle(): raise RuntimeError('not idle')
 before=metrics(); stop=False; maxr=maxw=0.0
 def mon():
  nonlocal maxr,maxw,stop
  while not stop:
   m=metrics(); maxr=max(maxr,m.get('num_requests_running',0)); maxw=max(maxw,m.get('num_requests_waiting',0)); time.sleep(.05)
 th=threading.Thread(target=mon,daemon=True); th.start()
 body={'model':a.model,'prompt':prompt,'temperature':0,'max_tokens':a.output_tokens,'stream':True,'stream_options':{'include_usage':True},'ignore_eos':True}
 t0=time.perf_counter(); first=None; text=''; usage={}
 try:
  with requests.post(base+'/v1/completions',json=body,stream=True,timeout=7200) as r:
   r.raise_for_status()
   for raw in r.iter_lines():
    if not raw: continue
    line=raw.decode('utf-8','replace')
    if not line.startswith('data: '): continue
    payload=line[6:]
    if payload=='[DONE]': break
    d=json.loads(payload)
    if d.get('usage'): usage=d['usage']
    for ch in d.get('choices') or []:
     piece=ch.get('text') or ''
     if piece and first is None: first=time.perf_counter()
     text += piece
 finally:
  stop=True; th.join(timeout=1)
 end=time.perf_counter(); after=metrics()
 comp=int((usage or {}).get('completion_tokens') or a.output_tokens)
 ttft=(first-t0) if first else None; dec=(end-first) if first else None
 drafted=after.get('spec_decode_num_draft_tokens_total',0)-before.get('spec_decode_num_draft_tokens_total',0)
 accepted=after.get('spec_decode_num_accepted_tokens_total',0)-before.get('spec_decode_num_accepted_tokens_total',0)
 return {
  'requested_prompt_tokens':target,'prompt_tokens':(usage or {}).get('prompt_tokens'),'completion_tokens':comp,
  'ttft_s':ttft,'prefill_proxy_tps':target/ttft if ttft else None,'decode_tps':max(comp-1,1)/dec if dec and dec>0 else None,
  'total_s':end-t0,'sha256':hashlib.sha256(text.encode()).hexdigest(),'max_running':maxr,'max_waiting':maxw,
  'contaminated':bool(maxr>1 or maxw>0),'drafted':drafted,'accepted':accepted,
  'accept_rate':accepted/drafted if drafted>0 else None,'mean_accept_len':1+(accepted/drafted*a.output_tokens/(a.output_tokens/max(drafted,1))) if False else None,
 }
targets=[int(x) for x in a.lengths.split(',') if x.strip()]
rows=[]
if a.resume and os.path.exists(a.out):
 try:
  old=json.load(open(a.out))
  if int(old.get('port',a.port))!=a.port or old.get('model',a.model)!=a.model or int(old.get('output_tokens',a.output_tokens))!=a.output_tokens:
   raise RuntimeError('resume metadata mismatch')
  # Successful rows are immutable scored evidence. Failed/incomplete rows are
  # retried so a transient timeout does not become a permanent skipped point.
  rows=[r for r in old.get('rows',[]) if 'error' not in r and r.get('requested_prompt_tokens') in targets]
 except Exception as e:
  raise SystemExit(f'ERROR: cannot resume {a.out}: {e!r}')

def write_state(status,error=None):
 order={t:i for i,t in enumerate(targets)}
 ordered=sorted(rows,key=lambda r:order.get(int(r.get('requested_prompt_tokens',-1)),len(order)))
 doc={'port':a.port,'model':a.model,'output_tokens':a.output_tokens,'status':status,'rows':ordered}
 if error is not None: doc['error']=error
 tmp=a.out+'.tmp'
 with open(tmp,'w') as f:
  json.dump(doc,f,indent=2,ensure_ascii=False); f.write('\n'); f.flush(); os.fsync(f.fileno())
 os.replace(tmp,a.out)

completed={int(r['requested_prompt_tokens']) for r in rows if 'error' not in r}
write_state('in_progress')
for target in targets:
 if target in completed:
  print(json.dumps({'requested_prompt_tokens':target,'resume':'skip_completed'}),flush=True)
  continue
 try:
  row=run(target); rows.append(row); print(json.dumps(row,ensure_ascii=False),flush=True); write_state('in_progress')
 except Exception as e:
  row={'requested_prompt_tokens':target,'error':repr(e)}; rows.append(row); print(json.dumps(row),flush=True); write_state('in_progress',repr(e))
write_state('complete')
