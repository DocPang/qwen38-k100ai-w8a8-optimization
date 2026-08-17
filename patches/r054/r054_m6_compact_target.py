"""R054: extend the frozen compact target/verifier lm_head to physical M6.

Frozen algorithm (unchanged from promoted R024/R025):
  uniform2048 full-row-scale W8A8 selector -> Top512(sorted=False)
  -> BF16 direct row-reduction over the original full lm_head
  -> dense-sparse BF16 logits (-inf outside candidate ids).

Eligible paths are single-request plain-greedy target sampling only:
- speculative verifier M=5 with exactly four scheduled speculative tokens;
- speculative verifier M=6 with exactly five scheduled speculative tokens,
  authorized only after the independent R053 physical-M6 full-BF16 holdout;
- true-M1 with no scheduled speculative tokens AND the request already past
  prompt prefill (`num_computed_tokens >= num_prompt_tokens`).

The original BF16 target compute_logits remains mandatory fallback for logprobs,
structured outputs/grammar, penalties, token masks, bad words, active/nontrivial
logits processors, non-greedy requests, multi-request batches, partial/chunked
prefill and unsupported/shape-inconsistent M. Draft proposal semantics are never
patched by this module.
"""
from __future__ import annotations

import contextvars
import hashlib
import time
import types

import torch
from vllm import _custom_ops as ops
from vllm.v1.spec_decode.eagle import EagleProposer
from vllm_hcu.v1.hcu_model_runner import GPUModelRunner
from vllm.model_executor.layers.quantization.compressed_tensors.triton_scaled_mm import scaled_mm_kernel
from vllm.triton_utils import triton, tl

VOCAB=248320
K_FULL=5120
K_SELECTOR=2048
TOPK=512
FROZEN_INDEX_SHA='1d1487248e97511306e6ba1192304f01ec44deee5930c1b992bf3c5e3d0330a2'
_SEL={'bm':32,'bn':64,'bk':512,'warps':8,'waves':2,'kp':2,'lat':'mmac5-ds6'}
_RR={'bk':1024,'warps':4,'waves':1}
_INSTALLED=False
_SEEN=set()
_ELIGIBLE_COUNTS={'fast':0,'fallback':0}
_RUNNER_CTX:contextvars.ContextVar[GPUModelRunner|None]=contextvars.ContextVar('k100_r024_runner',default=None)
_SCHED_CTX:contextvars.ContextVar[object|None]=contextvars.ContextVar('k100_r024_sched',default=None)


def _frozen_indices(device=None)->torch.Tensor:
    idx=torch.linspace(0,K_FULL-1,K_SELECTOR,dtype=torch.float64).round().to(torch.long).unique(sorted=True)
    if int(idx.numel())!=K_SELECTOR:
        raise RuntimeError(f'R054 uniform2048 cardinality drift {idx.numel()}')
    digest=hashlib.sha256(idx.numpy().tobytes()).hexdigest()
    if digest!=FROZEN_INDEX_SHA:
        raise RuntimeError(f'R054 uniform2048 index SHA drift {digest}')
    return idx.to(device).contiguous() if device is not None else idx.contiguous()


def _build_selector(head)->None:
    if hasattr(head,'k100_r024_selector_weight'):
        return
    w=getattr(head,'weight',None)
    if not isinstance(w,torch.Tensor) or w.dtype not in (torch.bfloat16,torch.float16) or tuple(w.shape)!=(VOCAB,K_FULL):
        raise RuntimeError(f'R054 unexpected BF16 target lm_head {type(w)} {getattr(w,"dtype",None)} {getattr(w,"shape",None)}')
    idx=_frozen_indices(w.device)
    q=torch.empty((VOCAB,K_SELECTOR),device=w.device,dtype=torch.int8)
    s=torch.empty((VOCAB,1),device=w.device,dtype=torch.float32)
    t=time.perf_counter();chunk=2048
    with torch.no_grad():
        for a in range(0,VOCAB,chunk):
            b=min(a+chunk,VOCAB)
            wf=w[a:b].float()
            sc=wf.abs().amax(1).clamp_min_(1e-12).div_(127.0)
            sub=torch.index_select(wf,1,idx)
            qq=torch.round(sub/sc[:,None]).clamp_(-127,127).to(torch.int8)
            q[a:b].copy_(qq);s[a:b,0].copy_(sc)
            del wf,sc,sub,qq
    head.register_buffer('k100_r024_selector_idx',idx,persistent=False)
    head.register_buffer('k100_r024_selector_weight',q,persistent=False)
    head.register_buffer('k100_r024_selector_scale',s,persistent=False)
    print('[K100 Q38 R054 M6 compact target] selector ready '
          f'shape={tuple(q.shape)} index_sha={FROZEN_INDEX_SHA} topk={TOPK} '
          f'bytes={q.numel()*q.element_size()+s.numel()*s.element_size()} build_s={time.perf_counter()-t:.3f}; BF16 head preserved',flush=True)


@triton.jit
def _rowreduce_kernel(x_ptr,w_ptr,ids_ptr,out_ptr,
                      SXM:tl.constexpr,SIM:tl.constexpr,SOM:tl.constexpr,
                      M_:tl.constexpr,C_:tl.constexpr,K_:tl.constexpr,BK_:tl.constexpr):
    pid=tl.program_id(0)
    row=pid//C_
    col=pid-row*C_
    valid=row<M_
    vid=tl.load(ids_ptr+row*SIM+col,mask=valid,other=0).to(tl.int64)
    o=tl.arange(0,BK_)
    acc=0.0
    for k0 in range(0,K_,BK_):
        kk=k0+o
        km=kk<K_
        x=tl.load(x_ptr+row*SXM+kk,mask=valid&km,other=0.0).to(tl.float32)
        ww=tl.load(w_ptr+vid*K_+kk,mask=valid&km,other=0.0).to(tl.float32)
        acc+=tl.sum(x*ww,axis=0)
    tl.store(out_ptr+row*SOM+col,acc.to(tl.bfloat16),mask=valid)


def _selector_logits(x2:torch.Tensor,idx:torch.Tensor,wq:torch.Tensor,ws:torch.Tensor)->torch.Tensor:
    m=int(x2.shape[0])
    xq,xs,zp=ops.scaled_int8_quant(x2,None,None,symmetric=True)
    if zp is not None:
        raise RuntimeError('R054 symmetric activation quant returned zero-point')
    xsub=torch.index_select(xq,1,idx).contiguous()
    wkn=wq.t()
    out=torch.empty((m,VOCAB),device=x2.device,dtype=torch.bfloat16)
    c=_SEL
    grid=(triton.cdiv(m,c['bm'])*triton.cdiv(VOCAB,c['bn']),)
    scaled_mm_kernel[grid](xsub,wkn,xs,ws,out,None,m,VOCAB,K_SELECTOR,
        xsub.stride(0),xsub.stride(1),wkn.stride(0),wkn.stride(1),
        out.stride(0),out.stride(1),tl.int32,
        BLOCK_SIZE_M=c['bm'],BLOCK_SIZE_N=c['bn'],BLOCK_SIZE_K=c['bk'],
        BLOCK_SIZE_SCALE_A=c['bm'],BLOCK_SIZE_SCALE_B=c['bn'],
        num_warps=c['warps'],num_stages=2,waves_per_eu=c['waves'],
        matrix_instr_nonkdim=16,kpack=c['kp'],mmac_layout_force=1,
        sched_latency=c['lat'])
    return out


def _rerank(x2:torch.Tensor,w:torch.Tensor,ids:torch.Tensor)->torch.Tensor:
    m=int(x2.shape[0]);out=torch.empty((m,TOPK),device=x2.device,dtype=torch.bfloat16);c=_RR
    _rowreduce_kernel[(m*TOPK,)](x2,w,ids,out,x2.stride(0),ids.stride(0),out.stride(0),M_=m,C_=TOPK,K_=K_FULL,BK_=c['bk'],num_warps=c['warps'],num_stages=1,waves_per_eu=c['waves'])
    return out


@torch.library.custom_op('k100_q38::r054_m6_compact_target_dense',mutates_args=(),device_types='cuda')
def _compact_dense(x:torch.Tensor,w:torch.Tensor,idx:torch.Tensor,wq:torch.Tensor,ws:torch.Tensor)->torch.Tensor:
    x2=x.reshape(-1,x.shape[-1]).contiguous();m=int(x2.shape[0])
    if m not in (1,5,6):
        raise RuntimeError(f'R054 compact target supports only M=1/5/6, got {m}')
    if tuple(w.shape)!=(VOCAB,K_FULL) or w.dtype is not torch.bfloat16:
        raise RuntimeError(f'R054 BF16 weight contract drift dtype={w.dtype} shape={tuple(w.shape)}')
    approx=_selector_logits(x2,idx,wq,ws)
    ids=torch.topk(approx,TOPK,dim=-1,largest=True,sorted=False).indices.contiguous()
    scores=_rerank(x2,w,ids)
    dense=torch.full((m,VOCAB),-float('inf'),device=x.device,dtype=torch.bfloat16)
    dense.scatter_(1,ids,scores)
    return dense


@_compact_dense.register_fake
def _compact_dense_fake(x:torch.Tensor,w:torch.Tensor,idx:torch.Tensor,wq:torch.Tensor,ws:torch.Tensor)->torch.Tensor:
    del w,idx,wq,ws
    return x.new_empty((*x.shape[:-1],VOCAB),dtype=torch.bfloat16)


def _processor_safe(md)->bool:
    try:procs=md.logitsprocs.non_argmax_invariant
    except Exception:return False
    for proc in procs:
        if proc.__class__.__name__=='MinTokensLogitsProcessor' and not bool(getattr(proc,'min_toks',{})):
            continue
        return False
    return True


def _eligible(m:int)->tuple[bool,str]:
    runner=_RUNNER_CTX.get();sched=_SCHED_CTX.get()
    if runner is None or sched is None:return False,'no_runner_context'
    if m not in (1,5,6):return False,f'M{m}'
    if int(getattr(runner.input_batch,'num_reqs',0))!=1:return False,'multi_request'
    # These SchedulerOutput/InputBatch fields are part of the exact HCU image
    # contract audited for R025/R053. Missing fields fail closed.
    if not hasattr(sched,'has_structured_output_requests'):
        return False,'missing_structured_output_state'
    if bool(sched.has_structured_output_requests):return False,'structured_output'
    if not hasattr(sched,'scheduled_spec_decode_tokens'):
        return False,'missing_spec_state'
    spec_map=sched.scheduled_spec_decode_tokens
    if m in (5,6):
        if not isinstance(spec_map,dict) or len(spec_map)!=1:
            return False,f'M{m}_spec_map_shape'
        try:
            scheduled_k=len(next(iter(spec_map.values())))
        except Exception:
            return False,f'M{m}_missing_scheduled_k'
        if scheduled_k!=m-1:
            return False,f'M{m}_scheduled_k{scheduled_k}'
    else:
        if bool(spec_map):return False,'M1_with_scheduled_spec'
        try:
            num_computed=int(runner.input_batch.num_computed_tokens_cpu[0])
            num_prompt=int(runner.input_batch.num_prompt_tokens[0])
        except Exception:
            return False,'missing_prompt_progress_state'
        # No-spec M1 is also used for partial/chunked prefills. Enable only once
        # the request was already fully prefetched before this model step.
        if num_computed<num_prompt:return False,'prefill_or_partial_prefill'
    md=runner.input_batch.sampling_metadata
    if not bool(md.all_greedy):return False,'non_greedy'
    if md.max_num_logprobs is not None:return False,'logprobs'
    if not bool(md.no_penalties):return False,'penalties'
    if md.allowed_token_ids_mask is not None:return False,'allowed_mask'
    if bool(md.bad_words_token_ids):return False,'bad_words'
    if not _processor_safe(md):return False,'logits_processor'
    return True,'eligible'


def install()->None:
    global _INSTALLED
    if _INSTALLED:return

    # Context must wrap the *current* execute_model after earlier R004/R304/R308
    # installers have already patched the runner class.
    orig_execute=GPUModelRunner.execute_model
    def execute_wrapped(self,scheduler_output,*args,**kwargs):
        rtok=_RUNNER_CTX.set(self);stok=_SCHED_CTX.set(scheduler_output)
        try:
            return orig_execute(self,scheduler_output,*args,**kwargs)
        finally:
            _SCHED_CTX.reset(stok);_RUNNER_CTX.reset(rtok)
    GPUModelRunner.execute_model=execute_wrapped

    # candidate_head.py has already wrapped EagleProposer.load_model to build
    # the accepted R004 draft selector. Chain after it, build target-only R025
    # selector buffers, and bind only the target language model compute_logits.
    orig_load=EagleProposer.load_model
    def load_wrapped(self,target_model):
        result=orig_load(self,target_model)
        tlm=target_model.get_language_model() if hasattr(target_model,'get_language_model') else target_model
        head=getattr(tlm,'lm_head',None)
        if head is None:raise RuntimeError('R054 target language model has no lm_head')
        _build_selector(head)
        # R047 draft and R025 target use the exact same frozen uniform2048
        # feature index and full-row-scale W8A8 selector. Alias the target
        # selector tensors onto the draft lm_head rather than allocating a
        # second ~0.475GiB copy. register_buffer receives the same Tensor
        # objects/storage; it does not clone them.
        draft_head=getattr(self.model,'lm_head',None)
        if draft_head is None:
            raise RuntimeError('R047 draft model has no lm_head for selector sharing')
        aliases=(
            ('k100_r047_selector_idx',head.k100_r024_selector_idx),
            ('k100_r047_selector_weight',head.k100_r024_selector_weight),
            ('k100_r047_selector_scale',head.k100_r024_selector_scale),
        )
        for name,tensor in aliases:
            if hasattr(draft_head,name):
                existing=getattr(draft_head,name)
                if existing.data_ptr()!=tensor.data_ptr():
                    raise RuntimeError(f'R047 shared selector alias conflict {name}')
            else:
                draft_head.register_buffer(name,tensor,persistent=False)
        if draft_head.k100_r047_selector_weight.data_ptr()!=head.k100_r024_selector_weight.data_ptr() or draft_head.k100_r047_selector_scale.data_ptr()!=head.k100_r024_selector_scale.data_ptr():
            raise RuntimeError('R047 target/draft selector storage is not shared')
        print('[K100 Q38 R047 draft K1024] shared R025 uniform2048 selector storage onto draft head; no duplicate selector allocation',flush=True)
        if not hasattr(tlm,'_k100_r024_orig_compute_logits'):
            orig_compute=tlm.compute_logits
            tlm._k100_r024_orig_compute_logits=orig_compute
            def target_compute(this,hidden_states):
                m=int(hidden_states.reshape(-1,hidden_states.shape[-1]).shape[0])
                ok,reason=_eligible(m)
                if ok:
                    _ELIGIBLE_COUNTS['fast']+=1
                    if m not in _SEEN:
                        _SEEN.add(m);print(f'[K100 Q38 R054 M6 compact target] ACTIVE M={m} Top{TOPK} dense-sparse eligible',flush=True)
                    h=this.lm_head
                    return torch.ops.k100_q38.r054_m6_compact_target_dense(hidden_states,h.weight,h.k100_r024_selector_idx,h.k100_r024_selector_weight,h.k100_r024_selector_scale)
                _ELIGIBLE_COUNTS['fallback']+=1
                return orig_compute(hidden_states)
            tlm.compute_logits=types.MethodType(target_compute,tlm)
            print(f'[K100 Q38 R054 M6 compact target] target compute_logits bound class={tlm.__class__.__name__}; draft untouched; exact scheduled-K4/M5 + scheduled-K5/M6 + true-M1 single-request plain-greedy only',flush=True)
        return result
    EagleProposer.load_model=load_wrapped
    _INSTALLED=True
    print('[K100 Q38 R054 M6 compact target] fail-closed install complete',flush=True)
