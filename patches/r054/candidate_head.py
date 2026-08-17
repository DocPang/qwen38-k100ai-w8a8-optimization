from __future__ import annotations

import os
import time
import hashlib

import torch
import torch.nn.functional as F
from vllm import _custom_ops as ops
from vllm.model_executor.layers.quantization.compressed_tensors.triton_scaled_mm import (
    scaled_mm_kernel,
)
from vllm.triton_utils import triton, tl
from vllm.v1.spec_decode.eagle import EagleProposer

K_FULL = 5120
K_SELECTOR = 2048
VOCAB = 248320
FROZEN_INDEX_SHA = "1d1487248e97511306e6ba1192304f01ec44deee5930c1b992bf3c5e3d0330a2"
CANDIDATE_K = int(os.getenv("K100_DRAFT_CANDIDATE_K", "1024"))
if CANDIDATE_K != 1024:
    raise RuntimeError(f"R047 frozen draft candidate requires K=1024, got {CANDIDATE_K}")
_INSTALLED = False
_DRAFT_SHADOW = os.getenv("K100_DRAFT_ONLY_LMHEAD_INT8", "0") == "1"
_DRAFT_BF16_VERIFY = os.getenv("K100_DRAFT_BF16_VERIFY", "0") == "1"
# Diagnostic-only clean oracle: keep the R003 selector/TopK shortlist, but
# rerank those ids using the exact original full-vocabulary BF16 compute_logits
# result. This deliberately gives up speed so R004 can distinguish selector
# recall from W8A8 rerank arithmetic without introducing a different-N BF16
# GEMM as a second numerical variable.
_DRAFT_FULL_BF16_ORACLE = os.getenv("K100_DRAFT_FULL_BF16_ORACLE", "0") == "1"
# Optional observer only. Keep OFF for the first quality-causality run: reading
# a GPU hit/miss predicate back into Python would synchronize the async path.
_DRAFT_ORACLE_LOG_RECALL = os.getenv("K100_DRAFT_ORACLE_LOG_RECALL", "0") == "1"
_DRAFT_SHADOW_CHUNK = int(os.getenv("K100_DRAFT_ONLY_LMHEAD_CHUNK", "2048"))
_ORACLE_STATS = {"total": 0, "miss": 0}

_CFG = {
    "bm": 32,
    "bn": 32,
    "bk": 512,
    "warps": 8,
    "waves": 2,
    "kp": 1,
    "lat": "mmac5-ds6",
}

# R067: the selector is a very different shape from the full-K candidate
# GEMM: M=1, K=1024, N=248320. Dedicated K100AI/gfx928 search found that
# doubling N tile width plus kpack=2 lifts effective weight bandwidth from
# ~589 to ~741 GB/s while remaining bitwise exact against R064.
_SELECTOR_CFG = {
    "bm": 32,
    "bn": 64,
    "bk": 512,
    "warps": 8,
    "waves": 2,
    "kp": 2,
    "lat": "mmac5-ds6",
}


def _scaled_logits_quantized(
    xq: torch.Tensor,
    xs: torch.Tensor,
    weight_nk: torch.Tensor,
    weight_scale: torch.Tensor,
    n: int,
    k: int,
) -> torch.Tensor:
    """Exact INT32 W8A8 scaled-mm for already-quantized one-token input."""
    w_nk = weight_nk[:n]
    ws = weight_scale[:n]
    w_kn = w_nk.t()
    out = torch.empty((1, n), dtype=torch.bfloat16, device=xq.device)
    c = _SELECTOR_CFG if k == K_SELECTOR else _CFG
    grid = (triton.cdiv(n, c["bn"]),)
    scaled_mm_kernel[grid](
        xq,
        w_kn,
        xs,
        ws,
        out,
        None,
        1,
        n,
        k,
        xq.stride(0),
        xq.stride(1),
        w_kn.stride(0),
        w_kn.stride(1),
        out.stride(0),
        out.stride(1),
        tl.int32,
        BLOCK_SIZE_M=c["bm"],
        BLOCK_SIZE_N=c["bn"],
        BLOCK_SIZE_K=c["bk"],
        BLOCK_SIZE_SCALE_A=c["bm"],
        BLOCK_SIZE_SCALE_B=c["bn"],
        num_warps=c["warps"],
        num_stages=2,
        waves_per_eu=c["waves"],
        matrix_instr_nonkdim=16,
        kpack=c.get("kp", 1),
        mmac_layout_force=1,
        sched_latency=c["lat"],
    )
    return out


def _quantize_one(hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    x = hidden_states.reshape(-1, hidden_states.shape[-1]).contiguous()
    assert int(x.shape[0]) == 1 and int(x.shape[1]) == K_FULL
    xq, xs, zp = ops.scaled_int8_quant(x, None, None, symmetric=True)
    assert zp is None
    return xq, xs


def _ensure_draft_shadow(lm_head) -> tuple[torch.Tensor, torch.Tensor] | None:
    if hasattr(lm_head, "k100_draft_weight") and hasattr(lm_head, "k100_draft_scale"):
        return lm_head.k100_draft_weight, lm_head.k100_draft_scale
    if not _DRAFT_SHADOW:
        return None
    weight = getattr(lm_head, "weight", None)
    if (
        weight is None
        or weight.dtype not in (torch.bfloat16, torch.float16)
        or tuple(weight.shape) != (VOCAB, K_FULL)
    ):
        return None

    started = time.perf_counter()
    # Quantize in row chunks so the BF16 target head remains untouched and we
    # never materialize the full ~5 GiB FP32 copy used by the legacy all-head
    # quantizer.  The shadow is used only inside Eagle draft proposal; target
    # verification keeps the original BF16 lm_head and therefore defines final
    # sampling semantics.
    qweight = torch.empty_like(weight, dtype=torch.int8)
    scale = torch.empty((VOCAB, 1), dtype=torch.float32, device=weight.device)
    with torch.no_grad():
        for start in range(0, VOCAB, _DRAFT_SHADOW_CHUNK):
            end = min(start + _DRAFT_SHADOW_CHUNK, VOCAB)
            wf = weight[start:end].float()
            sc = wf.abs().amax(dim=1).clamp_min_(1e-12).div_(127.0)
            q = torch.round(wf / sc[:, None]).clamp_(-127, 127).to(torch.int8)
            qweight[start:end].copy_(q)
            scale[start:end, 0].copy_(sc)
            del wf, sc, q
    lm_head.register_buffer("k100_draft_weight", qweight, persistent=False)
    lm_head.register_buffer("k100_draft_scale", scale, persistent=False)
    print(
        f"[K100 Q38 R003 draft-shadow] INT8 shadow ready shape={tuple(qweight.shape)} "
        f"chunk={_DRAFT_SHADOW_CHUNK} build_s={time.perf_counter() - started:.3f}",
        flush=True,
    )
    return qweight, scale


def _ensure_bf16_selector(lm_head) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
    """Resolve the R025 uniform2048 selector shared onto the draft head.

    R047 deliberately does not allocate a second ~0.475GiB selector. The outer
    R025 load wrapper builds the frozen target selector then aliases the same
    storage onto the draft lm_head under these R047 names. Missing aliases are
    a fail-closed integration error on the eligible draft path.
    """
    names=("k100_r047_selector_idx","k100_r047_selector_weight","k100_r047_selector_scale")
    if not all(hasattr(lm_head,n) for n in names):
        return None
    idx=lm_head.k100_r047_selector_idx
    q=lm_head.k100_r047_selector_weight
    sc=lm_head.k100_r047_selector_scale
    if tuple(idx.shape)!=(K_SELECTOR,) or tuple(q.shape)!=(VOCAB,K_SELECTOR) or tuple(sc.shape)!=(VOCAB,1):
        raise RuntimeError(f"R047 shared selector shape drift idx={tuple(idx.shape)} q={tuple(q.shape)} sc={tuple(sc.shape)}")
    return idx,q,sc


def _candidate_weight_scale(lm_head) -> tuple[torch.Tensor, torch.Tensor] | None:
    weight = getattr(lm_head, "weight", None)
    if (
        weight is not None
        and weight.dtype is torch.int8
        and hasattr(lm_head, "k100_27b_weight_scale")
        and tuple(weight.shape) == (VOCAB, K_FULL)
    ):
        return weight, lm_head.k100_27b_weight_scale
    return _ensure_draft_shadow(lm_head)


def _ensure_selector_prepack(lm_head, weight_nk: torch.Tensor) -> None:
    if hasattr(lm_head, "k100_r062_selector_weight"):
        return
    # Distribution-neutral static subset: 1024 dimensions evenly cover all
    # 5120 hidden dimensions.  R061 held-out real MTP hidden states showed
    # 100% true-full-head-top1 recall inside approximate Top16K (and Top2K).
    idx_cpu = torch.linspace(0, K_FULL - 1, K_SELECTOR, dtype=torch.float64).round().to(torch.long)
    assert int(idx_cpu.unique().numel()) == K_SELECTOR
    idx = idx_cpu.to(lm_head.weight.device).contiguous()
    with torch.no_grad():
        wsub = torch.index_select(weight_nk, 1, idx).contiguous()
    lm_head.register_buffer("k100_r062_selector_idx", idx, persistent=False)
    lm_head.register_buffer("k100_r062_selector_weight", wsub, persistent=False)
    print(
        f"[K100 R067 selector] prepacked uniform K={K_SELECTOR} weight shape={tuple(wsub.shape)}",
        flush=True,
    )


def install() -> None:
    global _INSTALLED
    if _INSTALLED:
        return

    orig_load_model = EagleProposer.load_model
    orig_propose = EagleProposer.propose
    orig_greedy = EagleProposer._greedy_sample

    def load_model(self, target_model):
        result = orig_load_model(self, target_model)
        if _DRAFT_SHADOW:
            lm_head = getattr(self.model, "lm_head", None)
            if lm_head is None:
                raise RuntimeError("draft-head optimization: MTP model has no shared lm_head after load_model")
            if _DRAFT_BF16_VERIFY:
                weight=getattr(lm_head,"weight",None)
                if weight is None or weight.dtype not in (torch.bfloat16,torch.float16) or tuple(weight.shape)!=(VOCAB,K_FULL):
                    raise RuntimeError(
                        "R047 frozen draft K1024: expected shared BF16/FP16 lm_head shape "
                        f"({VOCAB}, {K_FULL}), got dtype={getattr(weight, 'dtype', None)} shape={getattr(weight, 'shape', None)}"
                    )
                print(
                    "[K100 Q38 R047 draft K1024] BF16 draft head validated; uniform2048 selector allocation deferred to R025 shared target selector",
                    flush=True,
                )
            else:
                packed = _candidate_weight_scale(lm_head)
                if packed is None:
                    raise RuntimeError(
                        "R003 draft shadow: expected shared BF16/FP16 lm_head shape "
                        f"({VOCAB}, {K_FULL}), got dtype={getattr(getattr(lm_head, 'weight', None), 'dtype', None)} "
                        f"shape={getattr(getattr(lm_head, 'weight', None), 'shape', None)}"
                    )
                weight_nk, _ = packed
                _ensure_selector_prepack(lm_head, weight_nk)
                print(
                    "[K100 Q38 R003 draft-shadow] eager load-time shadow+selector prebuild complete; "
                    "memory will be included in subsequent KV profiling",
                    flush=True,
                )
        return result

    def propose(self, *args, **kwargs):
        self._k100_candidate_step = 0
        return orig_propose(self, *args, **kwargs)

    def greedy(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.use_local_argmax_reduction:
            return orig_greedy(self, hidden_states)

        step = int(getattr(self, "_k100_candidate_step", 0))
        self._k100_candidate_step = step + 1

        if int(hidden_states.reshape(-1, hidden_states.shape[-1]).shape[0]) != 1:
            return orig_greedy(self, hidden_states)

        lm_head = getattr(self.model, "lm_head", None)
        if lm_head is None:
            return orig_greedy(self, hidden_states)
        if _DRAFT_BF16_VERIFY:
            selector = _ensure_bf16_selector(lm_head)
            if selector is None:
                raise RuntimeError("R047 frozen draft K1024 eligible path missing shared R025 uniform2048 selector")
            selector_idx, selector_weight, selector_scale = selector
            x = hidden_states.reshape(-1, hidden_states.shape[-1]).contiguous()
            xq, xs = _quantize_one(hidden_states)
            xsub = torch.index_select(xq, 1, selector_idx).contiguous()
            approx = _scaled_logits_quantized(
                xsub,
                xs,
                selector_weight,
                selector_scale,
                VOCAB,
                K_SELECTOR,
            )
            ids = torch.topk(
                approx,
                CANDIDATE_K,
                dim=-1,
                largest=True,
                sorted=True,
            ).indices[0].contiguous()
            if _DRAFT_FULL_BF16_ORACLE:
                # Match upstream Eagle's original arithmetic exactly first:
                #   self.model.compute_logits(hidden_states).argmax(-1)
                # Then restrict that same full-BF16 result to the selector's
                # TopK ids. If the global BF16 top1 is present, the returned
                # draft token is guaranteed to equal upstream R001; if it is
                # absent, any difference is a selector-recall miss, not rerank
                # quantization or a smaller-N GEMM reduction-order artifact.
                with torch.no_grad():
                    full_logits = self.model.compute_logits(hidden_states)
                    full_token = full_logits.argmax(dim=-1).to(torch.long)
                    # Selector recall must be measured by membership, not by
                    # candidate-local argmax equality.  If exact BF16 logits
                    # contain a tie for the global maximum, upstream full-head
                    # argmax uses global vocabulary order while ``ids`` is
                    # ordered by approximate selector score. Candidate-local
                    # argmax could therefore choose a different tied token even
                    # when upstream ``full_token`` is present, falsely counting
                    # a selector miss.  On a membership hit return full_token
                    # directly so R004D is a true selector-only oracle.
                    hit_gpu = (ids == full_token.reshape(-1)[0]).any()
                    cand_logits = torch.index_select(full_logits, -1, ids)
                    local_idx = cand_logits.argmax(dim=-1).to(torch.long)
                    candidate_selected = ids.index_select(0, local_idx)
                    # Device-side selection keeps the strict oracle path free
                    # of GPU->CPU synchronization. On a shortlist hit this is
                    # exactly upstream full-BF16 top1; on a miss it returns the
                    # best exact-BF16 token available inside the same TopK.
                    selected = torch.where(
                        hit_gpu.reshape(1), full_token, candidate_selected
                    )
                if _DRAFT_ORACLE_LOG_RECALL:
                    # Observer mode only: this sync is intentionally excluded
                    # from the first causal quality run.
                    hit = bool(hit_gpu.item())
                    _ORACLE_STATS["total"] += 1
                    if not hit:
                        _ORACLE_STATS["miss"] += 1
                        # Observer-only rank bounds for choosing the next K
                        # from Qwen3.8 evidence rather than old-model priors.
                        # topk tie order is backend-dependent, so report both
                        # strictly-better+1 (best possible rank) and >=score
                        # (worst rank within an exact selector-score tie).
                        full_id = full_token.reshape(-1)[0]
                        full_selector_score = approx[0].index_select(
                            0, full_id.reshape(1)
                        )[0]
                        rank_lo = int(
                            (approx[0] > full_selector_score).sum().item()
                        ) + 1
                        rank_hi = int(
                            (approx[0] >= full_selector_score).sum().item()
                        )
                        token_id = int(full_id.item())
                        print(
                            f"[K100 Q38 R004 full-BF16 oracle] selector miss-rank "
                            f"total={_ORACLE_STATS['total']} miss={_ORACLE_STATS['miss']} "
                            f"rank_lo={rank_lo} rank_hi={rank_hi} token={token_id} "
                            f"candidate_k={CANDIDATE_K}",
                            flush=True,
                        )
                    # Observer service is intentionally synchronization-heavy
                    # and is never used for timing. Emit every step so a
                    # fail-closed external collector can recover exact per-case
                    # cumulative totals/misses without adding another API or
                    # guessing where a request ended. Strict R004D has this
                    # entire block disabled, so its execution path is unchanged.
                    print(
                        f"[K100 Q38 R004 full-BF16 oracle] recall progress "
                        f"total={_ORACLE_STATS['total']} miss={_ORACLE_STATS['miss']} "
                        f"candidate_k={CANDIDATE_K}",
                        flush=True,
                    )
                return selected
            with torch.no_grad():
                cand_w = torch.index_select(lm_head.weight, 0, ids).contiguous()
                exact = F.linear(x.to(cand_w.dtype), cand_w)
                # Full-vocabulary torch.argmax breaks exact-logit ties by the
                # lowest global vocabulary index. ``ids`` is ordered by the
                # approximate selector score, so candidate-local argmax would
                # use a different tie order and can diverge even when every
                # subset BF16 logit is bitwise identical to the full head and
                # the full token is present. Preserve the full-head contract by
                # choosing the smallest global token id among all candidate
                # entries tied at the exact maximum.
                max_logit = exact.max(dim=-1, keepdim=True).values
                is_max = exact == max_logit
                candidate_ids = ids.reshape(1, -1).expand_as(exact)
                sentinel = torch.full_like(candidate_ids, VOCAB)
                selected = torch.where(is_max, candidate_ids, sentinel).min(
                    dim=-1
                ).values
            return selected.to(torch.long)

        packed = _candidate_weight_scale(lm_head)
        if packed is None:
            return orig_greedy(self, hidden_states)
        weight_nk, weight_scale = packed

        _ensure_selector_prepack(lm_head, weight_nk)

        # R064: every draft position independently runs the cheap 1024-D
        # full-vocabulary selector, then chooses its token with an exact
        # full-5120-D W8A8 GEMM on only Top2K.  No candidate list is shared
        # across steps, avoiding R062's trajectory degradation from a low-set-
        # recall approximate Top16K.  On 748 real R053 draft hidden states the
        # true full-head token was inside Top2K 99.20% overall (100% at steps
        # 0/1, 98.40% at steps 2/3).
        xq, xs = _quantize_one(hidden_states)
        xsub = torch.index_select(
            xq, 1, lm_head.k100_r062_selector_idx
        ).contiguous()
        approx = _scaled_logits_quantized(
            xsub,
            xs,
            lm_head.k100_r062_selector_weight,
            weight_scale,
            VOCAB,
            K_SELECTOR,
        )
        ids = torch.topk(
            approx,
            CANDIDATE_K,
            dim=-1,
            largest=True,
            sorted=True,
        ).indices[0].contiguous()
        cand_w = torch.index_select(weight_nk, 0, ids).contiguous()
        cand_s = torch.index_select(weight_scale, 0, ids).contiguous()
        exact = _scaled_logits_quantized(
            xq, xs, cand_w, cand_s, CANDIDATE_K, K_FULL
        )
        local_idx = exact.argmax(dim=-1).to(torch.long)
        return ids.index_select(0, local_idx)

    EagleProposer.load_model = load_model
    EagleProposer.propose = propose
    EagleProposer._greedy_sample = greedy
    _INSTALLED = True
    if _DRAFT_BF16_VERIFY:
        if _DRAFT_FULL_BF16_ORACLE:
            mode = (
                f" + INT8 selector -> full-BF16 Top{CANDIDATE_K} oracle"
                f" recall_log={int(_DRAFT_ORACLE_LOG_RECALL)}"
            )
        else:
            mode = f" + INT8 selector -> BF16 Top{CANDIDATE_K} verify"
    else:
        mode = " + draft-only INT8 shadow" if _DRAFT_SHADOW else ""
    print(
        f"[K100 Q38 R047 draft K1024] frozen uniform2048 BN64/kpack2 -> {mode}; selector storage shared with R025 target",
        flush=True,
    )
