#!/usr/bin/env python3
"""Build the K100AI native INT8 output GEMV extension from source.

Run this inside the pinned Hygon/DTK container. The resulting .so is a local
build artifact; the repository intentionally ships the HIP source instead of a
precompiled binary.
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path

from torch.utils.cpp_extension import load

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "native_ext" / "k100_int8_gemv_v7.hip"
BUILD = REPO / "native_ext" / "build_gemv_v7"
DST = REPO / "native_ext" / "k100_int8_gemv_v7.so"

if not SRC.is_file():
    raise SystemExit(f"missing source: {SRC}")

BUILD.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("PYTORCH_ROCM_ARCH", "gfx928")
os.environ.setdefault("TORCH_EXTENSIONS_DIR", str(BUILD))

mod = load(
    name="k100_int8_gemv_v7",
    sources=[str(SRC)],
    extra_cflags=["-O3", "-std=c++20"],
    extra_cuda_cflags=["-O3", "-std=c++20", "-fno-gpu-rdc"],
    with_cuda=True,
    verbose=True,
)
shutil.copy2(mod.__file__, DST)
print(DST)
