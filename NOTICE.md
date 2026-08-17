# NOTICE

This repository contains original K100AI runtime tuning work for Qwen3.8-27B.
Upstream projects, checkpoints, images, libraries and trademarks retain their
own licenses and terms.

Validated upstream components:

- Qwen3.8-27B: Qwen / Alibaba Cloud, Apache-2.0 model card.
- `Freaksterz/Qwen3.8-27B-SmoothQuant-W8A8-INT8`: distributed separately on HuggingFace; pinned by revision in this repository.
- vLLM: Apache License 2.0.
- PyTorch / Triton: their upstream licenses.
- Hygon DTK / HCU runtime image: distributed by its vendor/community under the applicable terms.

No model weight shards, Docker layers, private IP addresses, credentials, SSH
configuration, or private conversation content are included here.

K100AI is the validated accelerator. Do not assume these tuned launch
geometries are optimal or correct for a different accelerator.
