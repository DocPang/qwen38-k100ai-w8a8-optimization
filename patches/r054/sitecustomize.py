"""K100AI/gfx928 shape-aware W8A8 runtime patch for Qwen3.6-27B.

R002 ports the proven 35B optimization method to the dense 27B model:
- exact small-M Triton configs for the real Qwen3.6-27B decode shapes;
- load-time per-channel INT8 lm_head plus K100AI AITER A8W8 decode kernel.
Unsupported shapes and prefill fall back to stock vLLM.
"""
from __future__ import annotations

import importlib
import os
from typing import Final

import torch


_LM_HEAD_ENABLED: Final[bool] = os.getenv("K100_27B_LM_HEAD_W8A8", "1") == "1"
_LM_HEAD_MAX_M: Final[int] = int(os.getenv("K100_27B_LM_HEAD_W8A8_MAX_M", "4"))
_GDN_FUSED: Final[bool] = os.getenv("K100_27B_GDN_FUSED", "0") == "1"

if _LM_HEAD_ENABLED:
    import time as _time

    from aiter.ops.triton.gemm_a8w8 import gemm_a8w8 as _lm_gemm_a8w8
    from vllm import _custom_ops as _lm_ops
    from vllm.model_executor.layers.quantization.base_config import (
        QuantizeMethodBase as _QuantizeMethodBase,
    )
    from vllm.model_executor.layers.quantization.compressed_tensors.triton_scaled_mm import (
        triton_scaled_mm as _lm_stock_scaled_mm,
        scaled_mm_kernel as _lm_scaled_mm_kernel,
    )
    from vllm.triton_utils import triton as _lm_triton, tl as _lm_tl
    from vllm.model_executor.layers.vocab_parallel_embedding import (
        ParallelLMHead as _ParallelLMHead,
        UnquantizedEmbeddingMethod as _UnquantizedEmbeddingMethod,
    )

    # R022: M=1 was re-searched specifically for no-MTP decode after the
    # full-int8 R018 profile. It is materially faster than the earlier
    # conservative config while remaining bit-identical to the current INT8
    # lm_head output in the isolated kernel gate. M=2/3/4 keep the proven
    # earlier settings because R022 is a no-MTP specialization.
    _LM_HEAD_CONFIGS = {
        1: {
            "BLOCK_SIZE_M": 32,
            "BLOCK_SIZE_N": 32,
            "BLOCK_SIZE_K": 256,
            "GROUP_SIZE_M": 4,
            "num_warps": 8,
            "num_stages": 2,
            "waves_per_eu": 2,
            "matrix_instr_nonkdim": 16,
            "kpack": 1,
        },
        2: {
            "BLOCK_SIZE_M": 16,
            "BLOCK_SIZE_N": 32,
            "BLOCK_SIZE_K": 256,
            "GROUP_SIZE_M": 4,
            "num_warps": 4,
            "num_stages": 2,
            "waves_per_eu": 1,
            "matrix_instr_nonkdim": 16,
            "kpack": 2,
        },
        3: {
            "BLOCK_SIZE_M": 16,
            "BLOCK_SIZE_N": 32,
            "BLOCK_SIZE_K": 256,
            "GROUP_SIZE_M": 4,
            "num_warps": 4,
            "num_stages": 2,
            "waves_per_eu": 1,
            "matrix_instr_nonkdim": 16,
            "kpack": 2,
        },
        4: {
            "BLOCK_SIZE_M": 16,
            "BLOCK_SIZE_N": 32,
            "BLOCK_SIZE_K": 256,
            "GROUP_SIZE_M": 4,
            "num_warps": 4,
            "num_stages": 2,
            "waves_per_eu": 1,
            "matrix_instr_nonkdim": 16,
            "kpack": 1,
        },
    }
    # R044: draft M=1 only.  The INT32 scaled_mm path is ~10% faster than
    # AITER's FP32-segmented accumulation.  It can differ by a few BF16 ULPs,
    # but actual 27B lm_head weights showed 0/84 argmax changes across a broad
    # hidden-state stress set.  Target/verifier M=5 remains on the exact path.
    _LM_HEAD_M1_FAST_CFG = {
        "bm": 32, "bn": 32, "bk": 512,
        "warps": 8, "waves": 2, "lat": "mmac5-ds6",
    }
    # R035 MTP4 verifier uses M=5.  AITER's M5 path is faster but changes
    # accumulation semantics (FP32 partial accumulation) and is not bitwise
    # identical.  Use vLLM's INT32 scaled_mm kernel with the exact M5 config
    # measured at ~1.76ms vs ~4.10ms stock for the 248320x5120 lm_head.
    _LM_HEAD_M5_CFG = {
        "bm": 32, "bn": 32, "bk": 512,
        "warps": 8, "waves": 2, "lat": "mmac5-ds10",
    }

    @torch.library.custom_op(
        "k100_27b::lm_head_w8a8", mutates_args=(), device_types="cuda"
    )
    def _k100_27b_lm_head_w8a8(
        x: torch.Tensor, weight_nk: torch.Tensor, weight_scale: torch.Tensor
    ) -> torch.Tensor:
        original_shape = tuple(x.shape[:-1])
        x2 = x.reshape(-1, x.shape[-1]).contiguous()
        xq, xs, xzp = _lm_ops.scaled_int8_quant(x2, None, None, symmetric=True)
        assert xzp is None
        m = int(x2.shape[0])
        cfg = _LM_HEAD_CONFIGS.get(m) if m <= _LM_HEAD_MAX_M else None
        if m == 1 and m <= _LM_HEAD_MAX_M:
            w_kn = weight_nk.t()
            out = torch.empty(
                (1, int(weight_nk.shape[0])), dtype=torch.bfloat16, device=x.device
            )
            c = _LM_HEAD_M1_FAST_CFG
            grid = (_lm_triton.cdiv(int(weight_nk.shape[0]), c["bn"]),)
            _lm_scaled_mm_kernel[grid](
                xq, w_kn, xs, weight_scale, out, None,
                1, int(weight_nk.shape[0]), int(xq.shape[1]),
                xq.stride(0), xq.stride(1),
                w_kn.stride(0), w_kn.stride(1),
                out.stride(0), out.stride(1),
                _lm_tl.int32,
                BLOCK_SIZE_M=c["bm"], BLOCK_SIZE_N=c["bn"], BLOCK_SIZE_K=c["bk"],
                BLOCK_SIZE_SCALE_A=c["bm"], BLOCK_SIZE_SCALE_B=c["bn"],
                num_warps=c["warps"], num_stages=2, waves_per_eu=c["waves"],
                matrix_instr_nonkdim=16, kpack=1, mmac_layout_force=1,
                sched_latency=c["lat"],
            )
        elif cfg is not None:
            out = _lm_gemm_a8w8(
                xq,
                weight_nk,
                xs,
                weight_scale,
                None,
                torch.bfloat16,
                config=cfg,
            )
        elif m == 5 and m <= _LM_HEAD_MAX_M:
            w_kn = weight_nk.t()
            out = torch.empty(
                (m, int(weight_nk.shape[0])), dtype=torch.bfloat16, device=x.device
            )
            c = _LM_HEAD_M5_CFG
            grid = (
                _lm_triton.cdiv(m, c["bm"])
                * _lm_triton.cdiv(int(weight_nk.shape[0]), c["bn"]),
            )
            _lm_scaled_mm_kernel[grid](
                xq, w_kn, xs, weight_scale, out, None,
                m, int(weight_nk.shape[0]), int(xq.shape[1]),
                xq.stride(0), xq.stride(1),
                w_kn.stride(0), w_kn.stride(1),
                out.stride(0), out.stride(1),
                _lm_tl.int32,
                BLOCK_SIZE_M=c["bm"], BLOCK_SIZE_N=c["bn"], BLOCK_SIZE_K=c["bk"],
                BLOCK_SIZE_SCALE_A=c["bm"], BLOCK_SIZE_SCALE_B=c["bn"],
                num_warps=c["warps"], num_stages=2, waves_per_eu=c["waves"],
                matrix_instr_nonkdim=16, kpack=1, mmac_layout_force=1,
                sched_latency=c["lat"],
            )
        else:
            out = _lm_stock_scaled_mm(
                xq, weight_nk.t(), xs, weight_scale, torch.bfloat16
            )
        return out.reshape(*original_shape, weight_nk.shape[0])

    @_k100_27b_lm_head_w8a8.register_fake
    def _k100_27b_lm_head_w8a8_fake(
        x: torch.Tensor, weight_nk: torch.Tensor, weight_scale: torch.Tensor
    ) -> torch.Tensor:
        del weight_scale
        return x.new_empty(
            (*x.shape[:-1], weight_nk.shape[0]), dtype=torch.bfloat16
        )

    class _K10027BLMHeadW8A8Method(_QuantizeMethodBase):
        def create_weights(self, *args, **kwargs):
            raise RuntimeError("K100AI 27B lm_head method is installed after load")

        def apply(
            self,
            layer: torch.nn.Module,
            x: torch.Tensor,
            bias: torch.Tensor | None = None,
        ) -> torch.Tensor:
            out = _k100_27b_lm_head_w8a8(
                x, layer.weight, layer.k100_27b_weight_scale
            )
            return out if bias is None else out + bias

    _orig_embedding_process = _UnquantizedEmbeddingMethod.process_weights_after_loading

    def _k100_27b_process_embedding(self, layer: torch.nn.Module) -> None:
        _orig_embedding_process(self, layer)
        if not isinstance(layer, _ParallelLMHead):
            return
        if int(getattr(layer, "tp_size", 1)) != 1:
            return
        weight = layer.weight.data
        if weight.dtype not in (torch.bfloat16, torch.float16):
            return
        if weight.ndim != 2 or int(weight.shape[1]) != 5120:
            return
        started = _time.perf_counter()
        with torch.no_grad():
            scale = weight.float().abs().amax(dim=1).clamp_min(1e-12).div_(127.0)
            qweight = torch.round(weight.float() / scale[:, None]).clamp_(
                -127, 127
            ).to(torch.int8)
        layer.weight = torch.nn.Parameter(qweight, requires_grad=False)
        layer.register_buffer(
            "k100_27b_weight_scale",
            scale[:, None].contiguous(),
            persistent=False,
        )
        layer.quant_method = _K10027BLMHeadW8A8Method()
        del weight, qweight, scale
        torch.cuda.empty_cache()
        print(
            "[K100 27B lm_head W8A8] enabled "
            f"shape={tuple(layer.weight.shape)} "
            f"quantize_s={_time.perf_counter() - started:.3f}",
            flush=True,
        )

    _UnquantizedEmbeddingMethod.process_weights_after_loading = (
        _k100_27b_process_embedding
    )
    print(
        f"[K100 27B lm_head W8A8] hook installed max_m={_LM_HEAD_MAX_M}",
        flush=True,
    )


_LINEAR_ENABLED: Final[bool] = os.getenv("K100_27B_SC_LINEAR_A8W8", "1") == "1"
_LINEAR_MAX_M: Final[int] = int(os.getenv("K100_27B_SC_LINEAR_A8W8_MAX_M", "3"))
_NATIVE_OUT_GEMV: Final[bool] = os.getenv("K100_27B_NATIVE_OUT_GEMV", "0") == "1"
_NATIVE_OUT_GEMV_SO: Final[str] = os.getenv(
    "K100_27B_NATIVE_OUT_GEMV_SO",
    "/opt/q38-release/native_ext/k100_int8_gemv_v7.so",
)

# Exact configs searched on K100AI/gfx928 with vLLM 0.18.1 compressed-tensors
# TritonInt8ScaledMMLinearKernel. Every listed candidate matched stock with
# relative-L2=0 in the isolated microbenchmark.
_TRITON_CONFIGS: dict[tuple[int, int, int], dict[str, int]] = {
    # Real vLLM packed runtime shapes. Checkpoint gate/up, q/k/v, qkv/z and
    # b/a weights are merged before execution, so these—not shard shapes—are
    # the configurations that matter to end-to-end decode.
    # M=1: no-MTP target decode and serial MTP draft forwards.
    # R025: HCU/gfx928 LLVM scheduling model tuned against the native MMAC
    # instruction (v_mmac_i32_16x16x32_i8).  These latency hints are bitwise
    # exact but materially improve the LDS-heavy M=1 decode kernels.
    (1, 5120, 34816): {"bm": 32, "bn": 32, "bk": 512, "warps": 8, "waves": 2, "lat": "mmac5-ds6"},
    (1, 17408, 5120): {"bm": 64, "bn": 64, "bk": 256, "warps": 4, "waves": 1, "lat": "mmac5-ds10"},
    (1, 5120, 16384): {"bm": 32, "bn": 32, "bk": 512, "warps": 8, "waves": 2, "lat": "mmac5-ds6"},
    (1, 5120, 96): {"bm": 32, "bn": 32, "bk": 512, "warps": 4, "waves": 1, "lat": "mmac5-ds6"},
    (1, 6144, 5120): {"bm": 16, "bn": 32, "bk": 128, "warps": 4, "waves": 1},
    (1, 5120, 14336): {"bm": 32, "bn": 32, "bk": 256, "warps": 8, "waves": 2},
    (1, 10240, 5120): {"bm": 32, "bn": 64, "bk": 128, "warps": 8, "waves": 2},
    # M=3: target verification shape for MTP2.
    (3, 5120, 34816): {"bm": 32, "bn": 32, "bk": 256, "warps": 8, "waves": 2},
    (3, 17408, 5120): {"bm": 64, "bn": 64, "bk": 256, "warps": 4, "waves": 2},
    (3, 5120, 16384): {"bm": 32, "bn": 32, "bk": 256, "warps": 8, "waves": 2},
    (3, 5120, 96): {"bm": 32, "bn": 32, "bk": 512, "warps": 4, "waves": 1},
    (3, 6144, 5120): {"bm": 32, "bn": 32, "bk": 128, "warps": 4, "waves": 1},
    (3, 5120, 14336): {"bm": 32, "bn": 32, "bk": 512, "warps": 8, "waves": 2},
    (3, 10240, 5120): {"bm": 32, "bn": 64, "bk": 128, "warps": 8, "waves": 1},
    # M=4: target verification shape for MTP3.
    (4, 5120, 34816): {"bm": 32, "bn": 32, "bk": 512, "warps": 8, "waves": 2},
    (4, 17408, 5120): {"bm": 64, "bn": 64, "bk": 256, "warps": 4, "waves": 2},
    (4, 5120, 16384): {"bm": 32, "bn": 32, "bk": 512, "warps": 8, "waves": 2},
    (4, 5120, 96): {"bm": 32, "bn": 32, "bk": 256, "warps": 4, "waves": 1},
    (4, 6144, 5120): {"bm": 32, "bn": 32, "bk": 512, "warps": 8, "waves": 2},
    (4, 5120, 14336): {"bm": 32, "bn": 32, "bk": 256, "warps": 8, "waves": 2},
    (4, 10240, 5120): {"bm": 32, "bn": 64, "bk": 128, "warps": 8, "waves": 2},
    # R035 M=5: target verification shape for MTP4.  All configs were
    # microbench-gated bitwise exact against stock on K100AI/gfx928.
    (5, 5120, 34816): {"bm": 32, "bn": 32, "bk": 512, "warps": 8, "waves": 2, "lat": "mmac5-ds6"},
    (5, 17408, 5120): {"bm": 64, "bn": 64, "bk": 256, "warps": 4, "waves": 2, "lat": "mmac5-ds10"},
    (5, 5120, 16384): {"bm": 32, "bn": 32, "bk": 512, "warps": 8, "waves": 2, "lat": "mmac5-ds6"},
    (5, 5120, 96): {"bm": 32, "bn": 32, "bk": 256, "warps": 4, "waves": 1, "lat": "mmac5-ds6"},
    (5, 6144, 5120): {"bm": 32, "bn": 16, "bk": 512, "warps": 8, "waves": 2, "lat": "mmac5-ds10", "kp": 2},
    (5, 5120, 14336): {"bm": 32, "bn": 32, "bk": 256, "warps": 8, "waves": 2},
    (5, 10240, 5120): {"bm": 32, "bn": 64, "bk": 128, "warps": 8, "waves": 2, "lat": "mmac5-ds6"},
    # R052 M=6: K5 verifier body.  These five high-budget families were
    # isolated in R051b against the actual stock M6 fallback and were bitwise
    # equal on four independent real-sized rotated matrices.  Untested M6
    # shapes deliberately remain absent and therefore fail closed to stock.
    (6, 5120, 34816): {"bm": 32, "bn": 32, "bk": 512, "warps": 8, "waves": 2, "lat": "mmac5-ds6"},
    (6, 17408, 5120): {"bm": 64, "bn": 64, "bk": 256, "warps": 4, "waves": 2, "lat": "mmac5-ds10"},
    (6, 5120, 16384): {"bm": 32, "bn": 32, "bk": 512, "warps": 8, "waves": 2, "lat": "mmac5-ds6"},
    (6, 5120, 14336): {"bm": 32, "bn": 32, "bk": 256, "warps": 8, "waves": 2},
    (6, 6144, 5120): {"bm": 32, "bn": 16, "bk": 512, "warps": 8, "waves": 2, "lat": "mmac5-ds10", "kp": 2},
}

if _LINEAR_ENABLED:
    _ct_mod = importlib.import_module(
        "vllm.model_executor.layers.quantization.compressed_tensors.triton_scaled_mm"
    )
    _kernel_mod = importlib.import_module(
        "vllm.model_executor.kernels.linear.scaled_mm.triton"
    )
    from vllm.triton_utils import tl as _tl, triton as _triton

    _orig_scaled_mm = _ct_mod.triton_scaled_mm
    _scaled_mm_kernel = _ct_mod.scaled_mm_kernel
    _native_out_mod = None
    if _NATIVE_OUT_GEMV:
        import importlib.util as _importlib_util
        _native_spec = _importlib_util.spec_from_file_location(
            "k100_int8_gemv_v7", _NATIVE_OUT_GEMV_SO
        )
        if _native_spec is None or _native_spec.loader is None:
            raise RuntimeError(f"cannot load native GEMV: {_NATIVE_OUT_GEMV_SO}")
        _native_out_mod = _importlib_util.module_from_spec(_native_spec)
        _native_spec.loader.exec_module(_native_out_mod)
        print(f"[K100 27B native out GEMV] loaded {_NATIVE_OUT_GEMV_SO}", flush=True)

    @torch.library.custom_op(
        "k100_27b::shapeaware_w8a8_linear", mutates_args=(), device_types="cuda"
    )
    def _shapeaware_w8a8_linear(
        x: torch.Tensor,
        weight: torch.Tensor,
        scale_a: torch.Tensor,
        scale_b: torch.Tensor,
    ) -> torch.Tensor:
        m, k = (int(v) for v in x.shape)
        n = int(weight.shape[1])
        cfg = _TRITON_CONFIGS.get((m, k, n))
        if cfg is None or m > _LINEAR_MAX_M:
            return _orig_scaled_mm(x, weight, scale_a, scale_b, torch.bfloat16)
        if (
            _native_out_mod is not None
            and m == 1
            and k == 6144
            and n == 5120
        ):
            # compressed-tensors exposes [K,N] as a transpose view of the
            # underlying contiguous [N,K] allocation, so weight.t() is a
            # zero-copy contiguous view accepted by the native GEMV.
            # R026: buffer-resource + hand-managed vmcnt + two accumulator
            # chains. mode=1 alternates complete 16B chunks between chains.
            return _native_out_mod.gemv(x, weight.t(), scale_a, scale_b, 2, 3)
        out = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
        bm, bn, bk = cfg["bm"], cfg["bn"], cfg["bk"]
        grid = (_triton.cdiv(m, bm) * _triton.cdiv(n, bn),)
        _scaled_mm_kernel[grid](
            x,
            weight,
            scale_a,
            scale_b,
            out,
            None,
            m,
            n,
            k,
            x.stride(0),
            x.stride(1),
            weight.stride(0),
            weight.stride(1),
            out.stride(0),
            out.stride(1),
            _tl.int32,
            BLOCK_SIZE_M=bm,
            BLOCK_SIZE_N=bn,
            BLOCK_SIZE_K=bk,
            BLOCK_SIZE_SCALE_A=bm,
            BLOCK_SIZE_SCALE_B=bn,
            num_warps=cfg["warps"],
            num_stages=2,
            waves_per_eu=cfg["waves"],
            kpack=cfg.get("kp", 1),
            sched_latency=cfg.get("lat", "none"),
        )
        return out

    @_shapeaware_w8a8_linear.register_fake
    def _shapeaware_w8a8_linear_fake(
        x: torch.Tensor,
        weight: torch.Tensor,
        scale_a: torch.Tensor,
        scale_b: torch.Tensor,
    ) -> torch.Tensor:
        del scale_a, scale_b
        return x.new_empty((x.shape[0], weight.shape[1]), dtype=torch.bfloat16)

    _PATCH_SHAPES = {(k, n) for (_, k, n) in _TRITON_CONFIGS}

    def _patched_scaled_mm(
        input: torch.Tensor,
        weight: torch.Tensor,
        scale_a: torch.Tensor,
        scale_b: torch.Tensor,
        out_dtype: type[torch.dtype],
        bias: torch.Tensor | None = None,
        block_size_m: int = 32,
        block_size_n: int = 32,
        block_size_k: int = 32,
        use_heuristic=True,
    ) -> torch.Tensor:
        # Keep the dynamic token dimension M out of the outer Python branch.
        # The custom op handles M=1/3 at runtime and stock-falls back otherwise.
        shape = (int(input.shape[1]), int(weight.shape[1]))
        if (
            bias is None
            and out_dtype is torch.bfloat16
            and shape in _PATCH_SHAPES
        ):
            return _shapeaware_w8a8_linear(input, weight, scale_a, scale_b)
        return _orig_scaled_mm(
            input,
            weight,
            scale_a,
            scale_b,
            out_dtype,
            bias=bias,
            block_size_m=block_size_m,
            block_size_n=block_size_n,
            block_size_k=block_size_k,
            use_heuristic=use_heuristic,
        )

    _ct_mod.triton_scaled_mm = _patched_scaled_mm
    _kernel_mod.triton_scaled_mm = _patched_scaled_mm
    print(
        f"[K100 27B shape-aware W8A8] installed configs={len(_TRITON_CONFIGS)} max_m={_LINEAR_MAX_M}",
        flush=True,
    )

    import r041_swiglu_int8 as _r041_swiglu_int8
    _r041_swiglu_int8.install(_shapeaware_w8a8_linear)

    if _GDN_FUSED:
        from einops import rearrange as _rearrange
        from vllm import _custom_ops as _gdn_ops
        from vllm.model_executor.models.qwen3_5 import (
            Qwen3_5GatedDeltaNet as _Qwen3_5GatedDeltaNet,
        )

        @_triton.jit
        def _gdn_qkvz_ba_fused_kernel(
            a_ptr,
            qkvz_ptr,
            ba_ptr,
            scale_a_ptr,
            qkvz_scale_ptr,
            ba_scale_ptr,
            c_ptr,
            M: _tl.constexpr,
            N_QKVZ: _tl.constexpr,
            N_BA: _tl.constexpr,
            K: _tl.constexpr,
            stride_am: _tl.constexpr,
            stride_ak: _tl.constexpr,
            stride_qkvz_k: _tl.constexpr,
            stride_qkvz_n: _tl.constexpr,
            stride_ba_k: _tl.constexpr,
            stride_ba_n: _tl.constexpr,
            stride_cm: _tl.constexpr,
            stride_cn: _tl.constexpr,
            BM: _tl.constexpr,
            BN: _tl.constexpr,
            BK: _tl.constexpr,
        ):
            pid = _tl.program_id(0)
            total_n = N_QKVZ + N_BA
            num_pid_n = _tl.cdiv(total_n, BN)
            pid_m = pid // num_pid_n
            pid_n = pid % num_pid_n
            offsets_m = pid_m * BM + _tl.arange(0, BM)
            offsets_n = pid_n * BN + _tl.arange(0, BN)
            offsets_k = _tl.arange(0, BK)
            mask_m = offsets_m < M
            mask_n = offsets_n < total_n
            a_ptrs = a_ptr + offsets_m[:, None] * stride_am + offsets_k[None, :] * stride_ak
            acc = _tl.zeros((BM, BN), dtype=_tl.int32)
            output_n_base = pid_n * BN
            if output_n_base < N_QKVZ:
                weight_n = offsets_n
                weight_ptrs = qkvz_ptr + offsets_k[:, None] * stride_qkvz_k + weight_n[None, :] * stride_qkvz_n
                for _ in range(0, _tl.cdiv(K, BK)):
                    mask_k = offsets_k < K
                    av = _tl.load(a_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0)
                    wv = _tl.load(weight_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0)
                    acc = _tl.dot(av, wv, acc, out_dtype=_tl.int32)
                    offsets_k += BK
                    a_ptrs += BK * stride_ak
                    weight_ptrs += BK * stride_qkvz_k
                scale_b = _tl.load(qkvz_scale_ptr + weight_n, mask=mask_n, other=0.0)[None, :]
            else:
                weight_n = offsets_n - N_QKVZ
                mask_weight_n = weight_n < N_BA
                weight_ptrs = ba_ptr + offsets_k[:, None] * stride_ba_k + weight_n[None, :] * stride_ba_n
                for _ in range(0, _tl.cdiv(K, BK)):
                    mask_k = offsets_k < K
                    av = _tl.load(a_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0)
                    wv = _tl.load(weight_ptrs, mask=mask_k[:, None] & mask_weight_n[None, :], other=0)
                    acc = _tl.dot(av, wv, acc, out_dtype=_tl.int32)
                    offsets_k += BK
                    a_ptrs += BK * stride_ak
                    weight_ptrs += BK * stride_ba_k
                scale_b = _tl.load(ba_scale_ptr + weight_n, mask=mask_weight_n, other=0.0)[None, :]
            scale_a = _tl.load(scale_a_ptr + offsets_m, mask=mask_m, other=0.0)[:, None]
            result = (acc.to(_tl.float32) * scale_a * scale_b).to(_tl.bfloat16)
            c_ptrs = c_ptr + offsets_m[:, None] * stride_cm + offsets_n[None, :] * stride_cn
            _tl.store(c_ptrs, result, mask=mask_m[:, None] & mask_n[None, :])

        @torch.library.custom_op(
            "k100_27b::gdn_qkvz_ba_fused_w8a8",
            mutates_args=(),
            device_types="cuda",
        )
        def _gdn_qkvz_ba_fused_w8a8(
            x: torch.Tensor,
            qkvz_weight: torch.Tensor,
            qkvz_scale: torch.Tensor,
            ba_weight: torch.Tensor,
            ba_scale: torch.Tensor,
        ) -> torch.Tensor:
            original_shape = tuple(x.shape[:-1])
            x2 = x.reshape(-1, x.shape[-1]).contiguous()
            xq, xs, xzp = _gdn_ops.scaled_int8_quant(x2, None, None, symmetric=True)
            assert xzp is None
            m = int(x2.shape[0])
            k = int(x2.shape[1])
            nq = int(qkvz_weight.shape[1])
            nb = int(ba_weight.shape[1])
            # R034: the fused kernel is numerically exact at M=4 too.  MTP3
            # verification uses M=4, so let verifier reuse the same one-quant
            # QKVZ+BA path instead of falling back to two separate linears.
            if m in (1, 4, 5) and k == 5120 and nq == 16384 and nb == 96:
                out = torch.empty((m, nq + nb), dtype=torch.bfloat16, device=x.device)
                grid = (_triton.cdiv(m, 16) * _triton.cdiv(nq + nb, 32),)
                _gdn_qkvz_ba_fused_kernel[grid](
                    xq,
                    qkvz_weight,
                    ba_weight,
                    xs,
                    qkvz_scale,
                    ba_scale,
                    out,
                    m,
                    nq,
                    nb,
                    k,
                    xq.stride(0),
                    xq.stride(1),
                    qkvz_weight.stride(0),
                    qkvz_weight.stride(1),
                    ba_weight.stride(0),
                    ba_weight.stride(1),
                    out.stride(0),
                    out.stride(1),
                    BM=16,
                    BN=32,
                    BK=512,
                    num_warps=4,
                    num_stages=2,
                    waves_per_eu=1,
                    kpack=2,
                    mmac_layout_force=1,
                    sched_latency="mmac5-ds6",
                )
            else:
                qkvz = _patched_scaled_mm(xq, qkvz_weight, xs, qkvz_scale, torch.bfloat16)
                ba = _patched_scaled_mm(xq, ba_weight, xs, ba_scale, torch.bfloat16)
                out = torch.cat((qkvz, ba), dim=-1)
            return out.reshape(*original_shape, nq + nb)

        @_gdn_qkvz_ba_fused_w8a8.register_fake
        def _gdn_qkvz_ba_fused_w8a8_fake(
            x: torch.Tensor,
            qkvz_weight: torch.Tensor,
            qkvz_scale: torch.Tensor,
            ba_weight: torch.Tensor,
            ba_scale: torch.Tensor,
        ) -> torch.Tensor:
            del qkvz_scale, ba_scale
            return x.new_empty(
                (*x.shape[:-1], qkvz_weight.shape[1] + ba_weight.shape[1]),
                dtype=torch.bfloat16,
            )

        _orig_gdn_forward = _Qwen3_5GatedDeltaNet.forward

        def _k100_27b_gdn_forward(self, hidden_states: torch.Tensor, output: torch.Tensor):
            qkvz_layer = self.in_proj_qkvz
            ba_layer = self.in_proj_ba
            if (
                not hasattr(qkvz_layer, "weight_scale")
                or not hasattr(ba_layer, "weight_scale")
                or qkvz_layer.weight.dtype is not torch.int8
                or ba_layer.weight.dtype is not torch.int8
                or tuple(qkvz_layer.weight.shape) != (5120, 16384)
                or tuple(ba_layer.weight.shape) != (5120, 96)
            ):
                return _orig_gdn_forward(self, hidden_states, output)

            num_tokens = hidden_states.size(0)
            projected = _gdn_qkvz_ba_fused_w8a8(
                hidden_states,
                qkvz_layer.weight,
                qkvz_layer.weight_scale,
                ba_layer.weight,
                ba_layer.weight_scale,
            )
            mixed_qkvz, ba = projected.split([16384, 96], dim=-1)
            qkv_size = (self.key_dim * 2 + self.value_dim) // self.tp_size
            z_size = self.value_dim // self.tp_size
            mixed_qkv, z = mixed_qkvz.split([qkv_size, z_size], dim=-1)
            z = z.reshape(z.size(0), -1, self.head_v_dim)
            b, a = ba.chunk(2, dim=-1)
            b = b.contiguous()
            a = a.contiguous()
            core_attn_out = torch.zeros(
                (num_tokens, self.num_v_heads // self.tp_size, self.head_v_dim),
                dtype=hidden_states.dtype,
                device=hidden_states.device,
            )
            torch.ops.vllm.gdn_attention_core(
                mixed_qkv, b, a, core_attn_out, self.prefix
            )
            z_shape = z.shape
            core_attn_out = core_attn_out.reshape(-1, core_attn_out.shape[-1])
            z = z.reshape(-1, z.shape[-1])
            core_attn_out = self.norm(core_attn_out, z)
            core_attn_out = core_attn_out.reshape(z_shape)
            core_attn_out = _rearrange(core_attn_out, "... h d -> ... (h d)")
            output[:num_tokens], _ = self.out_proj(core_attn_out)

        _Qwen3_5GatedDeltaNet.forward = _k100_27b_gdn_forward
        print("[K100 27B fused GDN QKVZ+BA W8A8] hook installed", flush=True)

# R047: draft-only alternating full-vocab / Top-16K candidate lm_head.
# Target/verifier execution is untouched, so final sampling remains target exact.
import candidate_head as _candidate_head
_candidate_head.install()

# R071 long-context adaptive path. These hooks are model-agnostic HCU/vLLM
# scheduler/runner patches proven on 35B and reused here for 27B MTP4:
# - force the effective drafter cutoff to a configurable sequence length;
# - above the cutoff, stop AsyncScheduler from re-inserting fake draft placeholders;
# - keep native M=5 FULL CUDAGraph and add a true M=1 FULL graph for long decode.
if os.getenv("K100_R304_DRAFTER_MAX_MODEL_LEN"):
    try:
        import r304_force_drafter_cutoff as _r304_force_drafter_cutoff
        _r304_force_drafter_cutoff.install()
    except Exception as _exc:
        print(f"[K100 27B R071] R304 cutoff disabled: {_exc!r}", flush=True)

if os.getenv("K100_R305_SPEC_CUTOFF"):
    try:
        import r305_adaptive_scheduler as _r305_adaptive_scheduler
        _r305_adaptive_scheduler.install()
    except Exception as _exc:
        print(f"[K100 27B R071] R305 adaptive scheduler disabled: {_exc!r}", flush=True)

if os.getenv("K100_R308_DUAL_CUDAGRAPH", "0") == "1":
    try:
        import r308_dual_cudagraph as _r308_dual_cudagraph
        _r308_dual_cudagraph.install()
    except Exception as _exc:
        print(f"[K100 27B R071] R308 dual CG disabled: {_exc!r}", flush=True)


# R113: exact K100AI full-attention prefill launch-geometry optimization.
if os.getenv("K100_27B_PREFILL_BM32_W8", "0") == "1":
    try:
        import r113_prefill_bm32_w8 as _r113_prefill_bm32_w8
        _r113_prefill_bm32_w8.install()
    except Exception as _exc:
        print(f"[K100 27B R113] prefill BM32/W8 disabled: {_exc!r}", flush=True)

# R143: GPU valid_sampled_tokens_count is already int32; keep the pinned CPU
# mirror int32 too, avoiding an otherwise unnecessary int32->int64 conversion
# before D2H. Semantics/MTP trajectory were previously validated exact.
if os.getenv("K100_27B_VALID_COUNT_INT32", "0") == "1":
    try:
        import r143_valid_count_int32 as _r143_valid_count_int32
        _r143_valid_count_int32.install()
    except Exception as _exc:
        print(f"[K100 27B R143] valid-count int32 disabled: {_exc!r}", flush=True)

# R155: exact non-align single-request shape-[1,1] accepted-count algebra fast path.
if os.getenv("K100_27B_SHAPE1_EXACT_ACCEPT", "0") == "1":
    try:
        import r155_shape1_exact_accept as _r155_shape1_exact_accept
        _r155_shape1_exact_accept.install()
    except Exception as _exc:
        print(f"[K100 27B R155 none] disabled: {_exc!r}", flush=True)

# Qwen3.8 migration: inherit the bitwise-exact R224 dynamic INT8 + R210
# RMSNorm->INT8 fusion from the mature 35B line. The fused op is model-shape
# agnostic for hidden sizes exercised here; unsupported rows/K values fall back
# to the stock vLLM _C dynamic quant kernel.
if os.getenv("K100_RMSNORM_INT8_FUSION", "0") == "1":
    try:
        import r210_norm_int8 as _k100_r210_norm_int8
        _k100_r210_norm_int8.install()
    except Exception as _exc:
        print(f"[K100 Q38 R224] RMS/dynamic INT8 fusion disabled: {_exc!r}", flush=True)

# R054: retain the promoted R024/R025 frozen uniform2048 + Top512 contract,
# and extend it to physical M6 only after the independent R053 real-hidden gate.
# Unsupported sampler/API/prefill or scheduled-K shapes fail closed to BF16.
if os.getenv("K100_Q38_R025_M1_COMPACT_TARGET", "0") == "1":
    try:
        import r054_m6_compact_target as _r054_m6_compact_target
        _r054_m6_compact_target.install()
    except Exception as _exc:
        print(f"[K100 Q38 R054 M6 compact target] install failed: {_exc!r}", flush=True)
        raise
