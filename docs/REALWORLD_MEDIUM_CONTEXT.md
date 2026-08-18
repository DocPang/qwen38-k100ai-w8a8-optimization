# Qwen3.8-27B W8A8 单卡：8K–16K 真实使用区间性能总结

这份文档专门回答一个容易被短上下文 benchmark 掩盖的问题：

> **如果不看 512/1K/2K 的峰值，而看更接近真实使用的 8K–16K 中等上下文，当前单卡最快、证据最完整的版本是什么？**

结论：当前可公开、可复现的 **单卡 TP=1 中上下文冠军配置仍然是 R054 K5/M6 compact-target fast branch**。

它不是靠 512 token 的峰值夺冠，而是同时具备：

- 8K / 12K 的 matched repeated 证据；
- 一条完整的 8K / 12K / 16K 公开发行曲线；
- 一条独立的实际 0.95 + Prefix Cache 长期部署确认曲线；
- 16K 明确锚点；
- MTP 与 true-M1 分开记录；
- 原始 JSON、SHA256、固定运行环境与复现脚本。

> **语义边界：** R054 是 relaxed / non-exact 高性能分支。它不能被描述为相对 R001/R047 的 globally exact 版本。需要严格 reference 语义时仍应使用 exact/reference 分支。

## 1. 我们如何定义 Decode tok/s

公开测速脚本 `scripts/benchmark_curve.py` 的定义是：

```text
decode_tps = (completion_tokens - 1) / (request_end_time - first_streamed_token_time)
```

中上下文曲线固定输出 **256 token**。

每个正式单请求结果同时检查：

- `max_running <= 1`
- `max_waiting == 0`
- `contaminated == false`
- speculative drafted / accepted 计数
- 输出 SHA256

因此这里的 Decode tok/s 不是 profiler kernel 吞吐，也不是 isolated GEMM microbenchmark。

## 2. 冠军配置

```text
Model                 Qwen3.8-27B SmoothQuant W8A8/INT8
Hardware              Hygon K100AI / gfx928
Tensor Parallel       TP=1
Branch                 R054 K5/M6 compact-target fast branch
Speculative depth     K=5
Physical verifier     M=6
Adaptive cutoff       41,216 total sequence tokens
Public context limit  262,144
```

8K–16K 全部位于 cutoff 以下，因此下表中的冠军成绩都是 **MTP5 / speculative decode**，不是 true-M1。

## 3. 主要中上下文冠军曲线

首要 benchmark 采用公开的 R054 `gpu_memory_utilization=0.95` 单请求发布曲线。它是一条完整的同配置 8K / 12K / 16K 曲线，不能和其它 run 按档位取最大值后重新拼接。

| Prompt | Output | Decode | MTP | Draft / accept | 污染 |
|---:|---:|---:|---|---:|---|
| 8K | 256 | **44.46 tok/s** | MTP5 | 230 / 212 | false |
| 12K | 256 | **35.99 tok/s** | MTP5 | 235 / 211 | false |
| 16K | 256 | **31.96 tok/s** | MTP5 | 235 / 212 | false |

**16K 锚点：31.9644 tok/s。**

原始证据：

- `results/raw/r054_mem095_10level_curve.json`
- SHA256: `ac906b3c8e9619419a4de8653e8cf440f9b3f94ec3170b0d11bdb8a6a0089619`

机器可读汇总：`results/realworld_medium_context_summary.json`。

### 可复现性标签

这条曲线属于 **published single-curve authority**：每个上下文档位是一条无污染正式请求，并且公开仓库固定了模型 revision、Docker digest、启动脚本、benchmark 脚本和原始结果。

它不是“同一档位重复 N 次取中位”的统计型结果，因此 8K/12K 的重复性还要看下一节的 promotion 证据。

## 4. 8K / 12K 的 matched repeated 支撑

R054 promotion 阶段还保留了一条更严格的同 prompt、双重复曲线：

| Prompt | Repeat 1 | Repeat 2 | Median | SHA | Spec trajectory |
|---:|---:|---:|---:|---|---|
| 8K | 43.0184 | 43.0589 | **43.0386 tok/s** | consistent | 240 / 209，两次一致 |
| 12K | 37.2309 | 37.1494 | **37.1901 tok/s** | consistent | 235 / 212，两次一致 |

这组证据的价值不是“把 12K 的 37.19 拼到上一节的 8K/16K 曲线中”，而是证明 **R054 在中上下文不是一次性的偶然高分**。

原始证据：

- `results/raw/r054_repeated_curve_8k_12k.json`
- SHA256: `d6f25c11fed5422412123c955f52eb186b90b1af084d521d03342c492dec11c5`

## 5. 实际 Prefix Cache 长期部署确认

当前长期部署 profile 开启：

```text
gpu_memory_utilization = 0.95
Prefix Caching         = on
max_model_len          = 262144
```

在这套更接近实际服务的 profile 上，同样得到完整的 8K / 12K / 16K MTP5 曲线：

| Prompt | Output | Decode | MTP | Draft / accept | 污染 |
|---:|---:|---:|---|---:|---|
| 8K | 256 | **43.47 tok/s** | MTP5 | 230 / 213 | false |
| 12K | 256 | **36.63 tok/s** | MTP5 | 235 / 208 | false |
| 16K | 256 | **31.69 tok/s** | MTP5 | 235 / 212 | false |

16K 与上一节 benchmark authority 的 **31.96 tok/s** 处于同一性能等级，说明中上下文结论并不依赖一个只用于刷分的短生命周期服务。

原始证据：

- `results/raw/r054_deployed_mem095_prefixcache_10level_final.json`
- SHA256: `1ba5c09c06d72e5eb426e6c57a7ed0d2cfe100e7fcd7312e28599bd86927f869`

## 6. 为什么不能挑每档最高值拼成一条曲线

历史上存在多条 R054 测量：

- 0.92 promotion matched/repeated；
- 0.95 release benchmark；
- 0.95 + Prefix Cache deployed profile。

不同 run 的 Prompt trajectory、MTP acceptance、服务状态会有小幅变化。Speculative decoding 对接受率非常敏感，因此：

- 8K 可能某条曲线更快；
- 12K 可能另一条曲线更快；
- 16K 又可能回到前一条曲线。

正确做法是**保留每条曲线的完整性**，而不是用 `max(8K) + max(12K) + max(16K)` 生成一个从未真实执行过的 synthetic best-of 曲线。

## 7. true-M1 / no-MTP 必须单独看

中上下文冠军表里的 8K/12K/16K 全是 MTP5，所以不能把 31.96 tok/s 说成“基础模型单 token decode”。

历史审计后，最终 SmoothQuant/R054 线路里**没有一条可直接比较的 true-M1 8K/12K/16K 完整曲线**。因此这里不人为制造一个 no-MTP 冠军。

同一 SmoothQuant checkpoint 家族里最近的 formal true-M1 锚点是 R001 的 **24K**：

| Prompt | Output | Decode | Draft / accept | 污染 |
|---:|---:|---:|---:|---|
| 24K | 256 | **15.0425 tok/s** | 0 / 0 | false |

原始证据：

- `results/raw/sq_r001_true_m1_24k.json`
- SHA256: `3392d2330995c518e3547b7d814b92b5b690432b438795f8e2d7023eb7eb0b1c`

另外确实存在更早 RTN 线路的 16K true-M1 单点，例如 R001 约 15.50 tok/s、R193/R224 约 17.56 tok/s。但这些不属于最终公开 SmoothQuant/R054 主线，后者还是一个已知整体 non-exact 的 migration bundle，因此这里只作为历史参考，不拿来冠名。

## 8. 为什么后续 512 更高的版本没有夺走这个冠军

R054 之后继续出现了 R064 / R077 / R193 / R224 等 kernel、控制流或 exact 工作分支，也出现过明显更高的 fixed-512 热态成绩。

但本次历史结果索引没有发现这些后续版本拥有已经 promotion 的完整 **8K / 12K / 16K full-model curve**。

所以不能做这样的推断：

```text
512 更快
=> 8K–16K 一定也更快
=> 自动替换 R054 的中上下文冠军
```

这种外推正是本次总结刻意避免的。

## 9. 512 热态成绩应该怎么看

当前部署版的 512 -> 512 热态 5 次中位数约 **69.17 tok/s**，这是有效的短上下文回归/上限数据，但不是本页的 headline。

原因很简单：

- 上下文太短；
- speculative acceptance 对 Prompt 极其敏感；
- 它不能代表 8K–16K 的真实使用速度；
- 更不能代表 32K–128K 的长期 Agent workload。

因此仓库首页现在优先展示 8K–16K 曲线，512 峰值保留在后面的短上下文章节。

## 10. 最终建议引用方式

如果只需要一句可公开引用的结论：

> **Qwen3.8-27B SmoothQuant W8A8 在单张 Hygon K100AI、TP=1、R054 K5/M6 上，公开 0.95 单请求曲线在 8K / 12K / 16K prompt、output256 时分别约为 44.46 / 35.99 / 31.96 tok/s；实际 0.95 + Prefix Cache 长期部署曲线为 43.47 / 36.63 / 31.69 tok/s。以上均为 MTP5 speculative decode，不是 true-M1。**

若需要强调重复性，可追加：

> **R054 promotion 的同 prompt 双重复结果在 8K / 12K 分别为 43.0386 / 37.1901 tok/s 中位数，并保持输出 SHA 与 speculative trajectory 稳定。**

不要把不同曲线按档位取最大值后合并成一个从未真实运行过的数字表。
