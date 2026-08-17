#!/usr/bin/env python3
"""Crash-tolerant single-service repeated decode curve for sequential same-GPU brackets.

Prompts are deterministic from (length, seed) and therefore reusable across a
candidate -> parent -> candidate bracket even though only one service occupies
the port at a time. Each length gets one warmup plus N measured streaming runs.
The result is fsync+rename persisted after every run and retains output SHA and
speculative counter deltas so speed is never separated from trajectory.
"""
from __future__ import annotations
import argparse, fcntl, hashlib, json, os, random, re, statistics, time
from pathlib import Path
import requests

WORDS="kernel memory attention quantization scheduler latency throughput profile tensor cache decode prefill context vector matrix runtime compiler device token model system data reasoning optimization exact benchmark".split()

def lock_port(port:int):
    fd=os.open(f"/tmp/qwen38_port{port}.score.lock",os.O_CREAT|os.O_RDWR,0o644)
    try:fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:raise SystemExit(f"ERROR scored benchmark lock busy port={port}")
    return fd

def post(base,path,body,timeout=7200):
    r=requests.post(base+path,json=body,timeout=timeout);r.raise_for_status();return r.json()

def metrics(base):
    try:s=requests.get(base+"/metrics",timeout=5).text
    except Exception:return {}
    out={}
    for n in ["num_requests_running","num_requests_waiting","spec_decode_num_draft_tokens_total","spec_decode_num_accepted_tokens_total"]:
        m=re.search(rf"^vllm:{n}\{{[^\n]+\}} ([0-9.e+-]+)$",s,re.M);out[n]=float(m.group(1)) if m else 0.0
    return out

def idle(base,limit=300):
    end=time.time()+limit
    while time.time()<end:
        m=metrics(base)
        if m.get("num_requests_running",0)==0 and m.get("num_requests_waiting",0)==0:return True
        time.sleep(.25)
    return False

def exact_prompt(base,target,seed):
    rng=random.Random(seed+target);chunks=[f"Q38-SEQ-BRACKET-{target}-{seed}"];count=0
    while count<target+64:
        chunks.append(" ".join(WORDS[rng.randrange(len(WORDS))] for _ in range(512)))
        if len(chunks)%8==0:count=int(post(base,"/tokenize",{"prompt":" ".join(chunks)},300)["count"])
    ids=post(base,"/tokenize",{"prompt":" ".join(chunks)},600)["tokens"][:target]
    prompt=post(base,"/detokenize",{"tokens":ids},600)["prompt"]
    got=int(post(base,"/tokenize",{"prompt":prompt},600)["count"])
    if got!=target:raise RuntimeError(f"roundtrip {target}->{got}")
    return prompt

def stream_one(base,model,prompt,n):
    if not idle(base):raise RuntimeError("service not idle")
    before=metrics(base);body={"model":model,"prompt":prompt,"max_tokens":n,"temperature":0,"ignore_eos":True,"stream":True,"stream_options":{"include_usage":True}}
    t0=time.perf_counter();first=None;end=None;usage={};parts=[]
    with requests.post(base+"/v1/completions",json=body,stream=True,timeout=7200) as r:
        r.raise_for_status()
        for raw in r.iter_lines():
            if not raw:continue
            line=raw.decode("utf-8","replace")
            if not line.startswith("data: "):continue
            data=line[6:]
            if data=="[DONE]":end=time.perf_counter();break
            d=json.loads(data)
            if d.get("usage"):usage=d["usage"]
            for c in d.get("choices") or []:
                txt=c.get("text") or ""
                if txt:
                    if first is None:first=time.perf_counter()
                    parts.append(txt)
    if end is None:end=time.perf_counter()
    if first is None:first=end
    after=metrics(base);comp=int(usage.get("completion_tokens") or n);dec=max(end-first,1e-9);text="".join(parts)
    return {"prompt_tokens":int(usage.get("prompt_tokens") or 0),"completion_tokens":comp,"ttft_s":first-t0,"decode_s":dec,"decode_tps":max(comp-1,1)/dec,"total_s":end-t0,"sha256":hashlib.sha256(text.encode()).hexdigest(),"drafted":after.get("spec_decode_num_draft_tokens_total",0)-before.get("spec_decode_num_draft_tokens_total",0),"accepted":after.get("spec_decode_num_accepted_tokens_total",0)-before.get("spec_decode_num_accepted_tokens_total",0)}

def persist(path,doc):
    p=Path(path);tmp=p.with_suffix(p.suffix+".tmp")
    with tmp.open("w") as f:json.dump(doc,f,indent=2,ensure_ascii=False);f.write("\n");f.flush();os.fsync(f.fileno())
    os.replace(tmp,p)

def summarize(row):
    vals=[r["decode_tps"] for r in row["runs"]]
    row["median_decode_tps"]=statistics.median(vals);row["min_decode_tps"]=min(vals);row["max_decode_tps"]=max(vals)
    row["sha_consistent"]=len({r["sha256"] for r in row["runs"]})==1
    row["trajectory_consistent"]=len({(r["drafted"],r["accepted"]) for r in row["runs"]})==1

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--port",type=int,default=8041);ap.add_argument("--model",default="qwen3.8-27b-w8a8");ap.add_argument("--lengths",default="512,42000,48000");ap.add_argument("--output",type=int,default=256);ap.add_argument("--repeats",type=int,default=3);ap.add_argument("--seed",type=int,default=20260816);ap.add_argument("--label",required=True);ap.add_argument("--out",required=True);a=ap.parse_args()
    fd=lock_port(a.port);base=f"http://127.0.0.1:{a.port}";lengths=[int(x) for x in a.lengths.split(",") if x]
    doc={"status":"in_progress","classification":"sequential same-GPU repeated decode curve","label":a.label,"port":a.port,"model":a.model,"lengths":lengths,"output_tokens":a.output,"repeats":a.repeats,"seed":a.seed,"rows":[]};persist(a.out,doc)
    for L in lengths:
        prompt=exact_prompt(base,L,a.seed)
        warm=stream_one(base,a.model,prompt,8)
        row={"length":L,"prompt_sha256":hashlib.sha256(prompt.encode()).hexdigest(),"warmup":warm,"runs":[]};doc["rows"].append(row);persist(a.out,doc)
        for i in range(a.repeats):
            r=stream_one(base,a.model,prompt,a.output);r["rep"]=i;row["runs"].append(r);summarize(row);persist(a.out,doc);print(json.dumps({"length":L,"rep":i,**r},ensure_ascii=False),flush=True)
        print("ROW",json.dumps({k:v for k,v in row.items() if k not in ("runs","warmup")},ensure_ascii=False),flush=True)
    doc["status"]="complete";persist(a.out,doc);print("SUMMARY",json.dumps({"label":a.label,"rows":[{"length":r["length"],"median_decode_tps":r["median_decode_tps"],"sha_consistent":r["sha_consistent"],"trajectory_consistent":r["trajectory_consistent"]} for r in doc["rows"]]},ensure_ascii=False),flush=True)
if __name__=="__main__":main()
