# Qwen3.8-27B W8A8 海光 K100AI 推理优化

[中文（当前）](README.md) | [English](README.en.md)

面向 **海光 K100AI（gfx928）** 的 **Qwen3.8-27B SmoothQuant W8A8/INT8** 单卡可复现推理优化方案。

本次发布以已验收的 **R054 K5/M6 高性能分支**为核心，包含：

- 固定到具体 revision 的 HuggingFace 上游模型；
- 固定到 digest 的海光/DTK Docker 镜像；
- 全部运行时补丁源码和原生 HIP GEMV 源码；
- Agent 实际部署与 benchmark 两套一键启动配置；
- 确定性的测速脚本；
- 机器可读的原始结果与汇总结果；
- 明确的正确性边界与 non-exact 边界。

> **重要说明：** R054 是高性能 **relaxed / non-exact** 分支。较早的 R047 K4 是 exact/reference 分支。R054 在测试范围内保持了 R052 K5 的行为并通过了限定语义/质量门禁，但它**不能保证对所有 Prompt 都与 R001/R047 在 byte/token/logprob 层面完全一致**。

## 1. 已验证运行环境

| 组件 | 已验证版本/参数 |
|---|---|
| 加速卡 | Hygon K100AI |
| GPU 架构 | `gfx928:sramecc+:xnack-` |
| 显存 | 65,520 MiB |
| Tensor Parallel | TP=1 |
| vLLM | `0.18.1+das.fa71803.dtk2604` |
| PyTorch | `2.10.0+das.opt1.dtk2604.20260325.g6b060a` |
| Triton | `3.6.0+gitc73250c4.staging` |
| DTK | `DTK-26.04-DCC2602-0317` |
| HIP runtime | `6.3.26093` |
| 最大模型上下文 | 262,144 |

固定 Docker 镜像：

```text
harbor.sourcefind.cn:5443/dcu/admin/base/custom@sha256:13ce550647063a7fe76e87fd173986175946e5046bd36980c4289c60a4bdd811
```

建议直接按 digest 拉取：

```bash
docker pull harbor.sourcefind.cn:5443/dcu/admin/base/custom@sha256:13ce550647063a7fe76e87fd173986175946e5046bd36980c4289c60a4bdd811
```

## 2. 精确的上游模型输入

本文所有公开测速使用以下已经公开的 HuggingFace checkpoint：

```text
repo:     Freaksterz/Qwen3.8-27B-SmoothQuant-W8A8-INT8
revision: 417ede1e4524c8fdbb586ebdabc9cfc5d0760b3e
```

下载固定 revision：

```bash
python3 -m pip install -r requirements.txt
python3 scripts/download_model.py --model-dir "$HOME/models/Qwen3.8-27B-SmoothQuant-W8A8-INT8"
```

快速校验 metadata：

```bash
python3 scripts/verify_model.py \
  --model-dir "$HOME/models/Qwen3.8-27B-SmoothQuant-W8A8-INT8" \
  --metadata-only
```

或者完整校验约 30 GiB 的全部权重 shard：

```bash
python3 scripts/verify_model.py \
  --model-dir "$HOME/models/Qwen3.8-27B-SmoothQuant-W8A8-INT8"
```

完整预期 SHA256 位于 `model_metadata/SHA256SUMS.quantized.txt`。

## 3. 编译 K100AI 原生 Kernel

仓库提供 HIP 源码，不提供预编译 `.so` 作为唯一依赖。

```bash
bash scripts/build_native_in_container.sh
```

预期生成：

```text
native_ext/k100_int8_gemv_v7.so
```

在固定镜像中默认使用：

```text
PYTORCH_ROCM_ARCH=gfx928
```

发布前已经在服务器上做过一次独立验证：将公开 release bundle 拷贝到新目录、删除已有 `.so`，仅使用本仓库 HIP 源码、构建脚本和固定镜像，从零编译成功。

## 4. 启动与实际长期部署相同的配置

这是推荐的 Agent 实际使用配置：**0.95 显存利用率、Prefix Caching、262K 上下文、41,216 token 以下使用 MTP5、超过 cutoff 自动切换到 true-M1、同时提供 OpenAI/Claude 兼容模型别名并支持工具调用解析**。

本次发布有意固定为 **纯文本模式**（`--language-model-only`）；Qwen3.8 的视觉能力不属于本版本已验证范围。

```bash
MODEL_DIR="$HOME/models/Qwen3.8-27B-SmoothQuant-W8A8-INT8" \
GPU_ID=0 PORT=8000 \
bash scripts/quickstart.sh
```

等价的显式入口：

```bash
MODEL_DIR="$HOME/models/Qwen3.8-27B-SmoothQuant-W8A8-INT8" \
GPU_ID=0 PORT=8000 \
bash scripts/serve_r054_agent.sh
```

默认参数：

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

服务默认暴露 `qwen3.8-27b-w8a8`；开启 Claude 兼容后，还会额外暴露 `claude-sonnet-4-6` 作为同一底层模型的兼容别名。

查看启动日志：

```bash
docker logs -f qwen38-k100ai-r054-agent
```

健康检查：

```bash
curl http://127.0.0.1:8000/v1/models
```

## 5. 复现 Benchmark 配置

历史 R054 promotion 测试使用相同运行时栈，但显存利用率为 0.92，且关闭 Prefix Caching。请不要把它与实际长期部署配置混为一谈：

```bash
MODEL_DIR="$HOME/models/Qwen3.8-27B-SmoothQuant-W8A8-INT8" \
GPU_ID=0 PORT=8000 \
bash scripts/serve_r054_benchmark.sh
```

固定 512 → 512 重复测速：

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

十档上下文测速：

```bash
python3 scripts/benchmark_curve.py \
  --port 8000 \
  --model qwen3.8-27b-w8a8 \
  --lengths 512,2048,4096,8192,12288,16384,32768,65536,131072,257900 \
  --output-tokens 256 \
  --out reproduce-10level.json
```

测速脚本会记录：TTFT、Prefill proxy 吞吐、Decode 吞吐、输出 SHA256、speculative drafted/accepted 计数、running/waiting 重叠情况，以及 contamination 标记。

## 6. 当前实际部署版本测速

下面的最终发布数据来自**当前实际 Agent 使用的 0.95 + Prefix Cache 长期服务配置**，不是为了刷分而裁剪过的 synthetic benchmark 服务。

### 6.1 短上下文热态吞吐

固定输入 512 token、输出 512 token，先 warmup，再正式测 5 次：

| 次数 | Decode tok/s |
|---:|---:|
| 0 | 69.105 |
| 1 | 69.170 |
| 2 | 69.179 |
| 3 | 69.146 |
| 4 | 69.325 |
| **中位数** | **69.170** |

5 次正式测速的输出 SHA256 完全一致，speculative trajectory 也完全一致：`440 drafted / 423 accepted`。

因此当前部署版在短上下文、高 MTP 接受率场景下，实际热态 Decode 可达到约 **69.17 tok/s**。

### 6.2 十档上下文曲线

所有档位固定输出 256 token、单请求执行；如果发现 waiting 或请求重叠污染，则不计入正式结果。本次最终十档全部 `contaminated=false`。

| 输入上下文 | TTFT | Prefill proxy | Decode | MTP 模式 | Draft / accept |
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

512/256 这一行刻意保留了一个 MTP 接受率较低的 Prompt，因此不能把它与上一节 69.17 tok/s 的 512/512 热态 workload 混为一谈。Speculative decoding 的吞吐会明显受 Prompt 和接受率影响。

最后一档 **257,900 输入 + 256 输出**完整通过，没有 OOM，测试期间 `max_running=1`、`max_waiting=0`。

### 6.3 真实 Agent 日常平均速度

为了避免“十档测速很完整，但日常到底多快仍然抽象”，额外统计了过去 30 天 **1,517 次真实交互式 Agent API 调用**，并将匿名聚合后的上下文分布映射到本次实际部署曲线。

**没有公开任何 Prompt、对话正文、用户标识或 session 标识。**

按调用数加权后的 session-average 上下文分布：

| 单次调用平均上下文 | 调用占比 |
|---:|---:|
| <=8K | 0.66% |
| 8–16K | 1.98% |
| 16–32K | 3.10% |
| 32–64K | 44.63% |
| 64–128K | 49.64% |

因此，实际观察到的调用中，**超过 94% 位于 32K–128K 区间**。

将测得的 Decode 曲线做分段插值后：

- 按调用次数算术加权：**16.17 tok/s**；
- 按总生成耗时折算的调和/有效速度：**15.57 tok/s**；
- 按历史实际生成 output token 数加权：**14.84 tok/s**。

因此用于日常规划时，推荐用一个更贴近实际体验的单值：

> **真实 Hermes / Agent 工作负载平均 Decode ≈ 15 tok/s**

这是根据实际 workload 分布得到的加权估算，并不代表每一个请求都会固定运行在 15 tok/s。

Prefix Caching 对 TTFT 的改善远大于对 Decode tok/s 的影响，因此也不应该拿完整冷启动上下文的 TTFT 直接代表 Agent 稳态每一轮等待时间。

### 6.4 Prefix Cache 在 Agent 场景中的效果

当前实际部署配置上，对重复 32K 前缀的测试结果：

- 冷 32K TTFT：约 98.06 s；
- 重复前缀 TTFT：约 7.65 s；
- 实测 Prefix 命中：28,944 / 32,768 token（88.3%）。

另一组更干净的 32K、只输出 1 token 的 cold/hot A/B：

- cold：79.46 s；
- hot：7.39 s；
- TTFT 约降低 **10.75×**；
- cold/hot 输出 SHA256 完全一致。

详细数据见 `results/prefix_cache_summary.json` 和 `results/raw/` 下原始文件。

## 7. 具体优化了什么

R054 不是单一 Kernel 优化，而是一套针对 Qwen3.8-27B W8A8 在 K100AI/gfx928 上真实 decode/prefill/MTP 热点逐步叠加形成的运行时优化栈。

### 7.1 Shape-aware 小 M W8A8 GEMM

针对 K100AI/gfx928 的真实 runtime `(M,K,N)` shape 做专用 Triton launch geometry，而不是只依赖通用 heuristic。

当前覆盖实际运行中的 M=1/3/4/5/6 热点 shape；未验证或不满足条件的 shape 会 fail closed，自动回退原始 vLLM 路径。

### 7.2 Physical-M6 verifier 修复

MTP5 会让 target verifier 进入 physical M=6。早期 K5 实验虽然减少了 verifier cycle，但 M6 掉出了已经优化的 W8A8 dispatch 路径，导致明显性能断崖。

因此对 5 类高成本 M6 W8A8 shape 做了独立 bitwise-equal 验证后专项优化：

- gate/up `(6,5120,34816)`；
- down `(6,17408,5120)`；
- GDN input `(6,5120,16384)`；
- full-attention input `(6,5120,14336)`；
- output `(6,6144,5120)`。

该修复把一个诊断性 K5 路径从约 **41.70 tok/s 提升到约 58.5 tok/s**，同时固定 workload 的输出 SHA 和 `485 drafted / 414 accepted` speculative trajectory 不变，因此可以把提升归因于 verifier body dispatch，而不是 speculative 行为变化。

### 7.3 原生 HIP INT8 Output GEMV

`native_ext/k100_int8_gemv_v7.hip` 针对热点单 token output projection family 做原生 HIP 专项实现。

仓库只发布源码，并要求在目标机器的固定 Docker 镜像中重新编译，避免把预编译 `.so` 当成不可验证的黑盒依赖。

### 7.4 Gated DeltaNet Projection Fusion

在严格 shape/dtype guard 下融合 QKVZ 与 BA 两组 W8A8 projection，减少重复 dynamic quant、kernel launch 和中间显存写回开销。

不满足已验证条件时自动使用原实现。

### 7.5 SwiGLU → INT8 Fusion

传统路径会先物化 BF16 `SiLU(gate) * up`，然后再单独做一次 dynamic INT8 quant，最后进入 W8A8 down projection。

优化路径在保持已验证 Inductor BF16 materialization/rounding 语义的前提下，直接生成量化后的 down-projection 输入与 scale，减少中间 BF16 materialization 和独立 quant kernel。

### 7.6 RMSNorm → INT8 与精确 Dynamic Quant

补丁利用 HCU 原生 `rms_norm_dynamic_per_token_quant` 路径，将 RMSNorm 和动态 INT8 quant 融合。

同时针对 speculative 热点 shape 匹配 ROCm/vLLM 真正使用的 `nearbyint` / ties-to-even 动态 INT8 舍入语义，避免普通 `round()` 在 `.5` 边界上产生数值偏差。

### 7.7 MTP 控制流 / Metadata 开销优化

不改变模型数学，只削减频繁的小型控制流开销，包括：

- `valid_sampled_tokens_count` 从 GPU int32 直接复制到 pinned CPU int32，避免无意义的 int64 dtype conversion；
- 常见单请求 `[1,1]` accepted-count 路径直接把精确 0/1 结果写入 persistent GPU buffer，减少临时 tensor 和额外 copy/conversion。

### 7.8 长上下文 Prefill Attention Geometry

针对 Qwen3.8 的真实 attention 结构：

```text
24 query heads
4 KV heads
head_dim = 256
```

根据 query/KV 区域选择经过验证的 gfx928 launch geometry；不满足 feature/shape 条件的组合回退 stock。

目标是降低长上下文冷 Prefill/TTFT，而不是用一个全局 attention replacement 覆盖所有场景。

### 7.9 自适应 Speculative Decoding

MTP 并不是上下文越长越划算。本项目针对 Qwen3.8 实测得到切换阈值：

```text
41,216 total sequence tokens
```

行为：

- cutoff 以下：K5 speculative decoding，physical verifier M6；
- cutoff 及以上：停止 drafter，清理后续 speculative placeholder；
- 不重新加载模型，直接切换为 true-M1 decode。

该阈值来自本 Qwen3.8 栈的实际测量，不是从其它模型直接照搬。

### 7.10 Draft Shortlist

Eagle proposal head 使用冻结的：

```text
uniform2048 selector
-> Top1024 candidate shortlist
-> BF16 rerank
```

并让 draft selector 与 target compact-head selector 共享存储，避免额外分配一份大型 selector tensor。

### 7.11 Compact Target / Verifier Head

R054 target 路径：

```text
uniform2048 W8A8 selector
-> Top512
-> original BF16 lm_head row rerank
-> sparse dense logits
```

独立 physical-M6 验证覆盖了 **3,216 个真实 BF16 hidden rows**：

- Top512 membership miss：0 / 3,216；
- 最终 BF16-rerank top1 mismatch：0 / 3,216；
- 完整 BF16 head：3725.54 us；
- compact head：1180.96 us；
- 局部 head 加速：**3.155×**。

## 8. 正确性边界

R054 相对于严格 R001 reference，在 extended 15-case API suite 上：

- 15/15 请求正常完成；
- 11/15 最终文本完全相同；
- 11/15 token sequence 完全相同；
- 分叉 case：0、2、4、6。

在另一套独立 20 题客观算术测试中：

- 两边 20/20 都能解析出答案；
- 两边答案 20/20 完全一致；
- 两边均为 19/20，并且错的是同一道题。

记录中的 R054 对 R052 fast-branch inheritance 比较也为 15/15 一致。详见 `results/quality_summary.json` 以及原始 evidence。

**不要把 R054 描述为 globally exact。** 如果业务要求严格 exact/reference 语义，应使用 R047 分支。

## 9. 完整复现检查清单

一个干净的复现实验至少应满足：

1. 确认硬件是 **K100AI / gfx928**，不能只确认“属于 K100 系列”。
2. 按固定 digest 拉取 Docker 镜像。
3. 按固定 HuggingFace revision 下载 checkpoint。
4. 至少校验 checkpoint metadata SHA256；建议完整校验全部 shard。
5. 在固定容器中从源码编译 `k100_int8_gemv_v7.so`。
6. 在一张原本空闲的 GPU 上启动仓库提供的 profile。
7. 等待 `/v1/models` 健康检查通过。
8. 运行 deterministic 512 repeated benchmark 与十档 curve。
9. 保持 `num_requests_waiting=0`；存在重叠/污染的请求不能作为单请求正式成绩。
10. 除 tok/s 外，同时比较输出 hash 与 speculative counter。

绝对吞吐仍会受到 firmware、宿主 CPU、频率、温度、缓存状态和 runtime build 等因素影响。复现目标应优先保证：

- 软件/模型身份一致；
- 单请求测速方法一致；
- 无 waiting/overlap 污染；
- 输出与 speculative 行为可检查；
- 性能落在同一等级。

而不是要求不同机器最后一个小数位完全一致。

## 10. 仓库结构

```text
patches/r054/        R054 最小运行时补丁闭包
native_ext/          原生 HIP 源码
scripts/             下载、校验、编译、部署、测速工具
model_metadata/      固定上游身份与 SHA256 manifest
results/             标准化汇总与精选原始实验数据
docs/                技术细节
```

## License

本仓库原创代码除特别说明外使用 Apache-2.0 License。上游模型、运行时、Docker 镜像及第三方库仍遵循各自原有许可和使用条款，详见 `NOTICE.md`。
