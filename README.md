# Qwen3.8-27B W8A8 optimization for Hygon K100AI

Reproducible single-GPU inference optimization for **Qwen3.8-27B SmoothQuant W8A8/INT8** on **Hygon K100AI (gfx928)**.

The release is built around the accepted **R054 K5/M6 fast branch** and includes:

- pinned HuggingFace checkpoint revision;
- pinned Hygon/DTK Docker image digest;
- source for all runtime patches and the native HIP GEMV;
- one-command Agent and benchmark launch profiles;
- deterministic benchmark scripts;
- machine-readable raw/summary results;
- explicit correctness and non-exactness boundaries.

> **Important:** R054 is a high-performance **relaxed/non-exact** branch. The older R047 K4 branch is the exact/reference branch. R054 preserves the tested R052 K5 behavior and passes bounded semantic/quality gates, but it is not globally byte/token/logprob exact to R001/R047 for every prompt.

## 1. Validated stack

| Component | Validated value |
|---|---|
| Accelerator | Hygon K100AI |
| GPU arch | `gfx928:sramecc+:xnack-` |
| VRAM | 65,520 MiB |
| Tensor parallelism | TP=1 |
| vLLM | `0.18.1+das.fa71803.dtk2604` |
| PyTorch | `2.10.0+das.opt1.dtk2604.20260325.g6b060a` |
| Triton | `3.6.0+gitc73250c4.staging` |
| DTK | `DTK-26.04-DCC2602-0317` |
| HIP runtime | `6.3.26093` |
| Max model length | 262,144 |

Pinned image:

```text
harbor.sourcefind.cn:5443/dcu/admin/base/custom@sha256:13ce550647063a7fe76e87fd173986175946e5046bd36980c4289c60a4bdd811
```

Pull it directly by digest:

```bash
docker pull harbor.sourcefind.cn:5443/dcu/admin/base/custom@sha256:13ce550647063a7fe76e87fd173986175946e5046bd36980c4289c60a4bdd811
```

## 2. Exact model input

The published measurements use this already-public HuggingFace checkpoint:

```text
repo:     Freaksterz/Qwen3.8-27B-SmoothQuant-W8A8-INT8
revision: 417ede1e4524c8fdbb586ebdabc9cfc5d0760b3e
```

Download the immutable revision:

```bash
python3 -m pip install -r requirements.txt
python3 scripts/download_model.py --model-dir "$HOME/models/Qwen3.8-27B-SmoothQuant-W8A8-INT8"
```

Verify metadata quickly:

```bash
python3 scripts/verify_model.py \
  --model-dir "$HOME/models/Qwen3.8-27B-SmoothQuant-W8A8-INT8" \
  --metadata-only
```

Or verify all ~30 GiB of weight shards:

```bash
python3 scripts/verify_model.py \
  --model-dir "$HOME/models/Qwen3.8-27B-SmoothQuant-W8A8-INT8"
```

The complete expected SHA256 list is in `model_metadata/SHA256SUMS.quantized.txt`.

## 3. Build the native K100AI kernel

The repository ships HIP source, not a precompiled `.so`.

```bash
bash scripts/build_native_in_container.sh
```

Expected artifact:

```text
native_ext/k100_int8_gemv_v7.so
```

The build defaults to `PYTORCH_ROCM_ARCH=gfx928` inside the pinned image.

## 4. Start the same profile used for real deployment

This is the practical Agent profile: **0.95 memory utilization, Prefix Caching, 262K context, MTP5 below the adaptive cutoff, true-M1 above it, OpenAI + Claude-compatible model aliases, and tool-call parsing**.

```bash
MODEL_DIR="$HOME/models/Qwen3.8-27B-SmoothQuant-W8A8-INT8" \
GPU_ID=0 PORT=8000 \
bash scripts/quickstart.sh
```

Equivalent explicit entry point:

```bash
MODEL_DIR="$HOME/models/Qwen3.8-27B-SmoothQuant-W8A8-INT8" \
GPU_ID=0 PORT=8000 \
bash scripts/serve_r054_agent.sh
```

Defaults:

```text
max_model_len           262144
gpu_memory_utilization  0.95
prefix_caching          enabled
speculative depth       K=5
physical verifier M     6
adaptive MTP cutoff     41,216 total sequence tokens
max_num_batched_tokens  4096
max_num_seqs            32
```

The service exposes `qwen3.8-27b-w8a8`; with Claude compatibility enabled it also exposes `claude-sonnet-4-6` as an alias to the same underlying model.

Follow startup:

```bash
docker logs -f qwen38-k100ai-r054-agent
```

Health check:

```bash
curl http://127.0.0.1:8000/v1/models
```

## 5. Reproduce the benchmark profile

Historical R054 promotion measurements used the same runtime stack at 0.92 memory utilization with Prefix Caching disabled. Keep this separate from the real deployment profile:

```bash
MODEL_DIR="$HOME/models/Qwen3.8-27B-SmoothQuant-W8A8-INT8" \
GPU_ID=0 PORT=8000 \
bash scripts/serve_r054_benchmark.sh
```

Fixed 512 -> 512 repeated benchmark:

```bash
python3 scripts/benchmark_repeated.py \
  --port 8000 \
  --model qwen3.8-27b-w8a8 \
  --lengths 512 \
  --output 512 \
  --repeats 5 \
  --seed 20260817 \
  --label reproduce-fixed512 \
  --out reproduce-fixed512.json
```

Ten-level context curve:

```bash
python3 scripts/benchmark_curve.py \
  --port 8000 \
  --model qwen3.8-27b-w8a8 \
  --lengths 512,2048,4096,8192,12288,16384,32768,65536,131072,257900 \
  --output-tokens 256 \
  --out reproduce-10level.json
```

The benchmark scripts record TTFT, prefill proxy throughput, decode throughput, output SHA256, speculative draft/accept counters, running/waiting overlap, and a contamination flag.

## 6. Current deployed-profile measurements

The final publication measurements below are from the **same 0.95 + Prefix Cache profile used for real Agent deployment**, not a synthetic stripped-down service.

### 6.1 Short hot throughput

Fixed prompt 512, output 512, one warmup + five scored repeats:

| Repeat | Decode tok/s |
|---:|---:|
| 0 | 69.105 |
| 1 | 69.170 |
| 2 | 69.179 |
| 3 | 69.146 |
| 4 | 69.325 |
| **Median** | **69.170** |

All five scored runs produced the same output SHA256 and the same speculative trajectory (`440 drafted / 423 accepted`).

### 6.2 Ten-level context curve

All rows use output=256, one request at a time, and are rejected from authority if waiting/overlap contamination is observed.

| Prompt | TTFT | Prefill proxy | Decode | MTP mode | Draft / accept |
|---:|---:|---:|---:|---|---:|
| 512 | 0.928 s | 551.7 tok/s | 35.04 tok/s | MTP5 | 435 / 168 |
| 2K | 2.772 s | 738.8 tok/s | 59.75 tok/s | MTP5 | 235 / 213 |
| 4K | 5.830 s | 702.5 tok/s | 52.45 tok/s | MTP5 | 235 / 212 |
| 8K | 11.673 s | 701.8 tok/s | 43.47 tok/s | MTP5 | 230 / 213 |
| 12K | 17.808 s | 690.0 tok/s | 36.63 tok/s | MTP5 | 235 / 208 |
| 16K | 24.248 s | 675.7 tok/s | 31.69 tok/s | MTP5 | 235 / 212 |
| 32K | 52.793 s | 620.7 tok/s | 19.91 tok/s | MTP5 | 245 / 209 |
| 64K | 121.829 s | 537.9 tok/s | 15.04 tok/s | true-M1 | 0 / 0 |
| 128K | 310.403 s | 422.3 tok/s | 12.83 tok/s | true-M1 | 0 / 0 |
| 257.9K | 873.079 s | 295.4 tok/s | 9.93 tok/s | true-M1 | 0 / 0 |

The 512/256 row intentionally shows a low-acceptance prompt and therefore must not be confused with the 69.17 tok/s 512/512 hot workload above. Speculative decoding throughput is prompt/acceptance sensitive.

### 6.3 Real-world Agent average

A separate anonymized aggregate of **1,517 real interactive Agent API calls over the previous 30 days** was mapped onto this deployed-profile curve. No prompts, conversation text, user identifiers, or session identifiers are published.

Call-weighted session-average context distribution:

| Average context per call | Share of calls |
|---:|---:|
| <=8K | 0.66% |
| 8–16K | 1.98% |
| 16–32K | 3.10% |
| 32–64K | 44.63% |
| 64–128K | 49.64% |

Therefore more than 94% of observed calls were in the 32K–128K region.

Using piecewise interpolation of the measured decode curve:

- call-weighted arithmetic estimate: **16.17 tok/s**;
- call-weighted harmonic/effective-time estimate: **15.57 tok/s**;
- weighting by the actually generated historical output-token counts: **14.84 tok/s**.

For practical planning, **~15 tok/s decode** is the recommended single-number description of this real Agent workload. This is a workload-weighted estimate, not a claim that every request runs at 15 tok/s.

Prefix Caching affects TTFT much more than decode speed, so cold full-context TTFT should not be used as the steady-state Agent turn latency.

### 6.4 Prefix Cache Agent behavior

Repeated 32K prefix measurements on the deployed profile:

- cold 32K TTFT: about 98.06 s;
- repeated-prefix TTFT: about 7.65 s;
- measured prefix hit: 28,944 / 32,768 tokens (88.3%).

A cleaner 32K, 1-output-token cold/hot pair measured:

- 79.46 s cold;
- 7.39 s hot;
- about **10.75x TTFT reduction**;
- identical cold/hot output SHA256.

See `results/prefix_cache_summary.json` and the raw artifacts under `results/raw/`.

## 7. What was optimized

### Shape-aware small-M W8A8 GEMM

K100AI/gfx928-specific Triton launch configurations cover the real M=1/3/4/5/6 runtime shapes. Unsupported shapes fail closed to the original vLLM path.

### Physical-M6 verifier repair

MTP5 creates a physical verifier M=6. Five high-budget W8A8 families were independently proven bitwise-equal to their stock fallback and then specialized:

- gate/up `(6,5120,34816)`;
- down `(6,17408,5120)`;
- GDN input `(6,5120,16384)`;
- full-attention input `(6,5120,14336)`;
- output `(6,6144,5120)`.

This repaired the major K5 body dispatch cliff: a diagnostic K5 path around 41.70 tok/s rose to about 58.5 tok/s without changing the fixed-workload SHA or `485/414` speculative trajectory.

### Native HIP INT8 output GEMV

`native_ext/k100_int8_gemv_v7.hip` specializes the hot single-token output-projection family. It is compiled from source on the target machine.

### Gated DeltaNet projection fusion

QKVZ and BA W8A8 projections are fused under exact shape/dtype guards to reduce redundant quantization, launch, and intermediate-memory overhead.

### SwiGLU -> INT8 fusion

The runtime can produce the quantized down-projection input directly while preserving the Inductor BF16 materialization/rounding semantics used by the validated graph.

### RMSNorm -> INT8 and exact dynamic quantization

The patch uses the HCU native `rms_norm_dynamic_per_token_quant` path and matches ROCm/vLLM dynamic INT8 `nearbyint` ties-to-even behavior for the specialized speculative shapes.

### MTP control-flow cleanup

Small but repeated metadata overhead is reduced without changing model math, including the valid-sampled-token CPU mirror and the common single-request accepted-count path.

### Long-context prefill attention geometry

The Qwen3.8 24Q/4KV/head_dim=256 prefill path uses guarded gfx928 launch geometry selected by query/KV region; unsupported feature combinations fall back to stock.

### Adaptive speculative decoding

The cutoff is measured for this Qwen3.8 stack rather than copied from another model:

```text
41,216 total sequence tokens
```

Below the cutoff: K5 speculative decoding. Above the cutoff: the scheduler stops drafting and uses true-M1 decode.

### Draft shortlist

The Eagle proposal head uses a frozen uniform2048 selector, Top1024 candidate shortlist and BF16 rerank, with selector storage shared with the target compact-head path.

### Compact target/verifier head

The R054 target path is:

```text
uniform2048 W8A8 selector
-> Top512
-> original BF16 lm_head row rerank
-> sparse dense logits
```

Independent physical-M6 validation covered 3,216 real BF16 hidden rows:

- Top512 membership misses: 0 / 3,216;
- final BF16-rerank top1 mismatches: 0 / 3,216;
- full BF16 head: 3725.54 us;
- compact head: 1180.96 us;
- local head speedup: **3.155x**.

## 8. Correctness boundary

R054 versus the strict R001 reference on the extended 15-case API suite:

- 15/15 requests completed;
- 11/15 final text equal;
- 11/15 token sequence equal;
- divergent cases: 0, 2, 4, 6.

On a separate 20-item objective arithmetic suite:

- both sides produced scorable answers for 20/20;
- answers were identical 20/20;
- both scored 19/20 and missed the same item.

R054 also inherited R052 fast-branch behavior 15/15 in the recorded comparison. See `results/quality_summary.json` and raw evidence.

Do **not** describe R054 as globally exact. Use R047 if strict exact-branch semantics are required.

## 9. Reproduction checklist

A clean reproduction should satisfy all of the following:

1. Confirm accelerator is **K100AI / gfx928**, not merely a different K100-family product.
2. Pull the Docker image by the pinned digest.
3. Download the HuggingFace checkpoint at the pinned revision.
4. Verify at least checkpoint metadata SHA256; preferably verify all shards.
5. Build `k100_int8_gemv_v7.so` from source in the pinned container.
6. Start one of the provided profiles on one otherwise-idle GPU.
7. Wait for `/v1/models` to become healthy.
8. Run the deterministic repeated 512 benchmark and ten-level curve.
9. Keep `num_requests_waiting=0`; do not score overlapped/contaminated requests as single-request authority.
10. Compare output hashes and speculative counters, not only tokens/s.

Absolute throughput can vary with firmware, host CPU, clocks, thermals, cache state and runtime build. Reproduction should focus on the same software/model identity, clean single-request methodology, and the same performance class rather than claiming every machine must match the last decimal place.

## 10. Repository layout

```text
patches/r054/        minimal runtime patch closure
native_ext/          native HIP source
scripts/             download, verify, build, serve and benchmark tools
model_metadata/      pinned upstream identity and SHA256 manifests
results/             normalized summaries and selected raw evidence
docs/                technical details
```

## License

Original code in this repository is released under Apache-2.0 unless otherwise noted. Upstream models, runtimes, images and libraries retain their own licenses and terms. See `NOTICE.md`.
