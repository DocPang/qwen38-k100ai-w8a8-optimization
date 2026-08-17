# Docker Image and R054 Optimization Summary

Date: 2026-08-17

This document summarizes the validated image, pinned checkpoint, and retained R054 optimization stack. It separates the **validated upstream runtime image**, the **pinned upstream checkpoint**, and the **project-owned K100AI runtime optimizations**.

## 1. Validated Docker image

Exact tested image tag:

```text
harbor.sourcefind.cn:5443/dcu/admin/base/custom:vllm018-ubuntu22.04-dtk26.04-qwen3.6-20260423
```

Immutable repository digest used by the validated deployment:

```text
sha256:13ce550647063a7fe76e87fd173986175946e5046bd36980c4289c60a4bdd811
```

Local Docker image ID observed on the tested host:

```text
sha256:ebbd1414b977e91775b925bd0a4151dda2b8089d3d649b2e459d2949cc0a9f59
```

For reproducible public instructions, prefer the **repo digest** rather than the mutable tag or host-local image ID:

```text
harbor.sourcefind.cn:5443/dcu/admin/base/custom@sha256:13ce550647063a7fe76e87fd173986175946e5046bd36980c4289c60a4bdd811
```

Public pull command:

```bash
docker pull harbor.sourcefind.cn:5443/dcu/admin/base/custom@sha256:13ce550647063a7fe76e87fd173986175946e5046bd36980c4289c60a4bdd811
```

Validated package metadata inside the current GPU2 production container:

```text
vLLM   0.18.1+das.fa71803.dtk2604
Torch  2.10.0+das.opt1.dtk2604.20260325.g6b060a
Triton 3.6.0+gitc73250c4.staging
DTK    DTK-26.04-DCC2602-0317
DCC    26.02.0-0
HIP    6.3.26093
```

Note: importing Python modules reports normalized upstream-style versions (`vllm 0.18.1`, `torch 2.10.0`, and a normalized Triton version) while `pip show` preserves the Hygon/DTK build suffixes. Public environment checks should therefore record `pip show`, the image digest and DTK path together.

Hardware authority:

```text
Hygon K100_AI
architecture: gfx928:sramecc+:xnack-
VRAM reported by hy-smi: 65,520 MiB per card
TP=1 for this release
```

## 2. Pinned upstream W8A8 checkpoint

The runtime optimization release starts from the already-public HuggingFace checkpoint:

```text
Freaksterz/Qwen3.8-27B-SmoothQuant-W8A8-INT8
```

Validated immutable revision:

```text
417ede1e4524c8fdbb586ebdabc9cfc5d0760b3e
```

This identity was recovered from the original 2026-08-15 Free Download Manager HTTP records on the Mac, which contain direct `huggingface.co/<repo>/resolve/<revision>/...` requests for the model shards, metadata and quantization helper scripts.

The GitHub release should therefore **not re-host the ~30 GiB checkpoint**. It should download the pinned HuggingFace revision and verify the hashes in `model_metadata/SHA256SUMS.quantized.txt`.

## 3. What R054 actually optimizes

The R054 release is a cumulative runtime stack. It is not one isolated kernel tweak. The main retained optimizations are grouped below by bottleneck.

### 3.1 Shape-aware small-M W8A8 GEMM dispatch

Problem:

Single-request decode and speculative verification repeatedly execute very small `M` W8A8 GEMMs. Generic Triton heuristics are not optimal on K100AI/gfx928.

Optimization:

- specialize real packed runtime `(M,K,N)` shapes rather than checkpoint-side logical shapes;
- provide K100AI/gfx928 Triton launch geometry for `M=1/3/4/5/6` where independently validated;
- preserve stock vLLM fallback for unsupported shapes;
- tune block sizes, warps, waves, kpack and HCU scheduling-latency hints for the actual model shapes.

The current patch has 33 shape-aware configurations and the R054 launcher allows these paths through physical M6.

### 3.2 Physical-M6 verifier body repair for speculative depth K=5

Problem:

Moving from K4 to K5 changes target verification from physical M5 to M6. The first K5 experiment reduced verifier cycle count but fell off optimized W8A8 dispatch paths, causing a severe body-performance cliff.

Optimization:

R052 added exact tuned M6 dispatch for five expensive families:

```text
gate/up          (6, 5120, 34816)
down             (6, 17408, 5120)
GDN input        (6, 5120, 16384)
full-attn input  (6, 5120, 14336)
output           (6, 6144, 5120)
```

Each selected M6 configuration passed independent bitwise equality gates before integration.

Causal full-model evidence:

```text
R050 K5 + stock M6 fallback:   ~41.70 tok/s fixed512
R052 K5 + repaired M6 body:    ~58.5 tok/s fixed512
```

The fixed-workload output SHA and the 485 drafted / 414 accepted trajectory remained unchanged across that repair, isolating the gain to body dispatch rather than speculative behavior.

### 3.3 Native INT8 output GEMV

Problem:

One hot single-row output projection shape is poorly served by a general GEMM path.

Optimization:

A project-owned HIP native kernel is used for the validated `M=1, K=6144, N=5120` output GEMV path:

```text
native_ext/k100_int8_gemv_v7.hip
native_ext/build_k100_int8_gemv_v7.py
```

The public release should compile the `.so` from source inside the pinned image rather than distribute only a prebuilt binary.

### 3.4 Gated DeltaNet QKVZ + BA W8A8 fusion

Problem:

The GDN path performs two adjacent W8A8 input projections with redundant launch/quantization/writeback overhead.

Optimization:

- fuse QKVZ and BA projection handling for the exact Qwen3.8 runtime shape/dtype contract;
- share the quantized input work where possible;
- fall back to the original vLLM implementation when layout or dtype assumptions are not met.

### 3.5 SwiGLU -> INT8 fusion

Problem:

The standard MLP path can materialize BF16 SiLU×up output and then launch a separate dynamic INT8 quantization step before the INT8 down projection.

Optimization (`r041_swiglu_int8.py`):

- compute FP32 SiLU and multiply with the same Inductor reduction/materialization semantics;
- derive the per-row absmax/scale;
- directly emit the BF16-equivalent result as INT8 plus scale;
- match HCU dynamic INT8 ties-to-even behavior;
- feed the result directly into the shape-aware W8A8 down projection.

The special 17,408-wide path is implemented as an exact fused custom op; unsupported shapes use stock behavior.

### 3.6 RMSNorm -> dynamic INT8 fusion and exact `nearbyint`

Problem:

A common path executes RMSNorm to BF16 and then performs a second dynamic per-token INT8 quantization operation. Earlier custom quantizers could also silently differ from the HCU runtime at `.5` rounding boundaries.

Optimization (`r210_norm_int8.py`, mature R224 semantics):

- use the HCU native `rms_norm_dynamic_per_token_quant` operation through the vLLM Inductor norm-quant fusion pass;
- use exact ROCm `nearbyint`/ties-to-even semantics in the custom dynamic INT8 kernel;
- specialize only validated hot row-count/K combinations and fall back elsewhere.

This preserves the exact INT8 values/scales on the validated boundary suite while removing an intermediate norm->quant step.

### 3.7 MTP control-flow / metadata D2H reductions

Two exact runtime-control optimizations are retained:

**R143 valid-count D2H**

The GPU already produces `valid_sampled_tokens_count` as int32, but the HCU runner previously mirrored it into a pinned int64 CPU buffer. R143 changes the CPU mirror to int32 and avoids the unnecessary dtype conversion before D2H.

**R171 shape-[1,1] accepted-count fast path**

For the common single-request non-align shape, the stock accepted-count expression algebraically reduces to `token != -1`. R171 writes that exact 0/1 result directly into the persistent accepted-count GPU buffer, eliminating a temporary allocation and an extra conversion/copy while keeping the original pinned-CPU copy/event ordering.

### 3.8 Exact long-prefill attention geometry tuning

Problem:

Cold TTFT at long context is dominated increasingly by full-attention prefill. One geometry is not optimal across query length and KV length on gfx928.

Optimization (`r113_prefill_bm32_w8.py`, evolved R134 policy):

- single-request BF16 full-attention prefill only;
- Qwen3.8 exact attention shape: 24 query heads, 4 KV heads, head_dim 256;
- adaptive BM64/BQ10/T32 launch geometry;
- 4 or 8 warps selected by measured q×K regions;
- conservative feature/shape gates;
- decode, unsupported multimodal/prefix-LM/alibi/sinks/other layouts fall back to stock.

The intent is to improve TTFT without changing decode semantics or using a blanket attention replacement.

### 3.9 Adaptive MTP -> true-M1 scheduling

Problem:

Speculative decoding does not remain beneficial forever as context grows. A fixed speculative depth can become slower than ordinary one-token decode even when speculative acceptance remains high.

Qwen3.8-specific measured crossover policy:

```text
cutoff = 41,216 total sequence tokens
```

Runtime pieces:

```text
r304_force_drafter_cutoff.py
r305_adaptive_scheduler.py
r308_dual_cudagraph.py
```

Deployment behavior:

- below the cutoff: R054 uses speculative depth **K=5**, physical verifier M6;
- at/above the cutoff: stop drafter work and clear future speculative placeholders;
- continue as true-M1 decode without reloading the model;
- maintain FULL graph coverage for the relevant short and long decode shapes.

The cutoff is based on Qwen3.8 measurements; it is not inherited from the old Qwen3.6 value.

### 3.10 R047 Eagle draft-head shortlist acceleration

Problem:

Each speculative proposal originally pays an expensive full-vocabulary draft-head operation.

Optimization:

```text
uniform2048 selector
-> Top1024 candidates
-> exact BF16 candidate rerank
```

Important implementation detail:

The uniform2048 selector storage is shared/aliased with the target compact-head selector rather than allocating a second large selector tensor.

Frozen R047 serving evidence showed 15/15 equality to the strict reference in text, tokens, common-prefix logprobs and speculative trajectory for its validation suite.

This reduction in proposal cost is a major reason K5 becomes economically viable after the M6 body is repaired.

### 3.11 R054 compact target/verifier head

Problem:

The full BF16 vocabulary head is extremely expensive at physical M6 and remains first-order even after the verifier body is repaired.

R054 target path:

```text
uniform2048 full-row-scale W8A8 selector
-> Top512(sorted=False)
-> original BF16 lm_head row-reduce rerank
-> dense-sparse BF16 logits
```

Frozen selector identity:

```text
1d1487248e97511306e6ba1192304f01ec44deee5930c1b992bf3c5e3d0330a2
```

Independent physical-M6 semantic authority:

```text
3,216 real BF16 hidden rows
Top512 membership misses:          0 / 3,216
final BF16-rerank top1 mismatch:  0 / 3,216
full BF16 target head:             3,725.54 us
compact target head:               1,180.96 us
saved per verifier step:           2,544.58 us
local head speedup:                3.155x
```

Full-model R054 fixed512 warm authority:

```text
R052 K5/M6 body:  58.2143 tok/s median
R054 + compact M6 target: 64.5343 tok/s median
```

R054 is fail-closed: unsupported sampler/API/prefill/shape cases use the original BF16 target head.

### 3.12 Prefix Cache and long-lived Agent deployment

Prefix Caching is a deployment optimization rather than the core R054 math change, but it is important for the actual Hermes/Claude use case because consecutive Agent turns share most of their prefix.

Current long service:

```text
max_model_len = 262144
gpu_memory_utilization = 0.95
prefix caching = enabled
MTP cutoff = 41216
speculative depth = 5
```

Measured repeated 32K prompt:

```text
cold TTFT: 98.06 s
hot TTFT:   7.65 s
second hot: 7.64 s
cached tokens: 28,944 / 32,768 (~88.3%)
```

Cleaner 32K/1-token cold-hot check:

```text
79.46 s -> 7.39 s (~10.75x TTFT reduction)
output SHA identical
```

The long-lived service also enables OpenAI-compatible tool calling and a Claude-compatible served-model alias; these are integration features, not performance claims.

## 4. Performance progression to present publicly

Use this table to explain the cumulative research path without publishing every failed experiment:

| Stage | Main change | fixed512 authority | Exactness role |
|---|---|---:|---|
| R001 | formal practical W8A8 base, BF16 target head | 34.25 tok/s | strict/base reference |
| R047 | accelerated Eagle draft shortlist, K4 | ~53.74 tok/s fresh same-GPU | exact/reference branch |
| R052 | K5 + exact physical-M6 W8A8 body repair | ~58.21 tok/s | relaxed K5 branch |
| **R054** | R052 + compact physical-M6 target head | **64.53 tok/s** | accepted fast branch |

The current user-facing 0.95 + Prefix Cache service measures a dedicated hot512 median of about **62.70 tok/s**; do not mix that deployment measurement with the 64.53 tok/s R054 promotion authority without labeling the different service configuration.

## 5. Correct public wording

Recommended summary:

> R054 combines K100AI/gfx928 shape-aware W8A8 dispatch, native/fused INT8 runtime paths, exact HCU control-flow reductions, Qwen3.8-specific adaptive speculative scheduling, an accelerated Eagle draft shortlist, a repaired physical-M6 verifier body, and a fail-closed compact physical-M6 target head. Prefix Caching is enabled in the long-lived Agent profile to reuse repeated conversation prefixes. R054 is the accepted high-performance branch; R047 remains the stricter exact reference branch.

Do **not** describe R054 as globally bitwise exact to R001/R047. K5/M6 speculative execution changes finite-precision trajectories for some prompts even though the individual M6 body repair and compact-head candidate gates were exact in their frozen scopes.
