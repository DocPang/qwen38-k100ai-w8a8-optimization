from __future__ import annotations

import os
import torch

_INSTALLED = False


def install() -> None:
    """Exact single-request Qwen3.6-27B full-attention prefill geometry tune.

    R120 keeps the R118 exact BM32/BQ5/T32/8w path for long KV sequences, but
    uses BM64/BQ10/T32/4w when max_seqlen_k <= 8192.  Microbench coverage at
    q=2048/3072/4096 showed this short-KV geometry is bitwise identical and
    materially faster, while it becomes slower beyond the 8K region.  The
    conservative 3072 query-token gate remains unchanged for end-to-end safety.

    Decode/speculative verification, multi-request batches, sliding-window,
    multimodal/prefix-LM, FP8, sinks, alibi and QQ-bias all fall back to stock.
    """
    global _INSTALLED
    if _INSTALLED or os.getenv("K100_27B_PREFILL_BM32_W8", "0") != "1":
        return

    import vllm.v1.attention.ops.triton_unified_attention as ua
    from vllm.triton_utils import triton

    if getattr(ua, "_k100_r120_installed", False):
        _INSTALLED = True
        return

    original = ua.unified_attention
    hits = {"n": 0}
    fallbacks = {"n": 0}

    def tuned(
        q, k, v, out, cu_seqlens_q, max_seqlen_q, seqused_k,
        max_seqlen_k, softmax_scale, causal, window_size, block_table,
        softcap, q_descale, k_descale, v_descale,
        seq_threshold_3D=None, num_par_softmax_segments=None,
        softmax_segm_output=None, softmax_segm_max=None,
        softmax_segm_expsum=None, alibi_slopes=None, output_scale=None,
        qq_bias=None, sinks=None, mm_prefix_range=None, use_alibi_sqrt=False,
    ):
        full_attention = bool(window_size[0] < 0)
        single_req = int(seqused_k.shape[0]) == 1
        exact_shape = (
            q.dtype == torch.bfloat16
            and q.ndim == 3 and k.ndim == 4 and v.ndim == 4
            and int(q.shape[1]) == 24
            and int(k.shape[2]) == 4
            and int(q.shape[2]) == 256
            and int(k.shape[3]) == 256
            and int(v.shape[3]) == 256
        )
        safe_features = (
            bool(causal)
            and full_attention
            and float(softcap) == 0.0
            and q_descale is None
            and output_scale is None
            and alibi_slopes is None
            and qq_bias is None
            and sinks is None
            and mm_prefix_range is None
            and not bool(use_alibi_sqrt)
        )
        gate = int(os.getenv("K100_27B_PREFILL_GATE", "3072"))
        is_prefill = int(max_seqlen_q) >= gate and int(q.shape[0]) >= gate

        if not (is_prefill and single_req and exact_shape and safe_features):
            fallbacks["n"] += 1
            if fallbacks["n"] <= 3:
                print(
                    f"[K100 27B R118 prefill] fallback={fallbacks['n']} "
                    f"qshape={tuple(q.shape)} maxq={max_seqlen_q} maxk={max_seqlen_k} "
                    f"single={single_req} causal={causal} window={window_size} "
                    f"softcap={softcap} dtype={q.dtype} alibi={alibi_slopes is not None} "
                    f"outscale={output_scale is not None} sinks={sinks is not None} "
                    f"mm={mm_prefix_range is not None}",
                    flush=True,
                )
            return original(
                q, k, v, out, cu_seqlens_q, max_seqlen_q, seqused_k,
                max_seqlen_k, softmax_scale, causal, window_size, block_table,
                softcap, q_descale, k_descale, v_descale,
                seq_threshold_3D, num_par_softmax_segments,
                softmax_segm_output, softmax_segm_max, softmax_segm_expsum,
                alibi_slopes, output_scale, qq_bias, sinks, mm_prefix_range,
                use_alibi_sqrt,
            )

        num_query_heads = 24
        num_kv_heads = 4
        num_queries_per_kv = 6
        max_k = int(max_seqlen_k)
        qlen = min(int(max_seqlen_q), int(q.shape[0]))
        w8_q3072_k = int(os.getenv("K100_27B_PREFILL_W8_Q3072_K", "5120"))
        w8_q2032_k = int(os.getenv("K100_27B_PREFILL_W8_Q2032_K", "7168"))
        w8_q1536_k = int(os.getenv("K100_27B_PREFILL_W8_Q1536_K", "16384"))
        w8_q1280_k = int(os.getenv("K100_27B_PREFILL_W8_Q1280_K", "32768"))
        w8_q1024_k = int(os.getenv("K100_27B_PREFILL_W8_Q1024_K", "49152"))
        use_w8 = (
            (qlen >= 3072 and max_k >= w8_q3072_k)
            or (qlen >= 2032 and max_k >= w8_q2032_k)
            or (qlen >= 1536 and max_k >= w8_q1536_k)
            or (qlen >= 1280 and max_k >= w8_q1280_k)
            or (qlen >= 1024 and max_k >= w8_q1024_k)
        )
        block_m = 64
        block_q = 10
        num_warps = 8 if use_w8 else 4
        geometry = f"BM64/BQ10/T32/{num_warps}w/stage1/wave1/qxk"
        tile_size = 32  # keep the verified BF16 reduction order exactly
        num_seqs = 1
        total_num_q_blocks = int(q.shape[0]) // block_q + num_seqs
        block_size = int(v.shape[1])

        ua.kernel_unified_attention_2d[(total_num_q_blocks, num_kv_heads)](
            output_ptr=out,
            query_ptr=q,
            key_cache_ptr=k,
            value_cache_ptr=v,
            sink_ptr=None,
            block_tables_ptr=block_table,
            seq_lens_ptr=seqused_k,
            alibi_slopes_ptr=None,
            qq_bias_ptr=None,
            scale=softmax_scale,
            k_scale=k_descale,
            v_scale=v_descale,
            out_scale=1.0,
            softcap=0.0,
            num_query_heads=num_query_heads,
            num_queries_per_kv=num_queries_per_kv,
            block_table_stride=block_table.stride(0),
            query_stride_0=q.stride(0),
            query_stride_1=q.stride(1),
            output_stride_0=out.stride(0),
            output_stride_1=out.stride(1),
            qq_bias_stride_0=0,
            BLOCK_SIZE=block_size,
            TILE_SIZE=tile_size,
            HEAD_SIZE=256,
            HEAD_SIZE_PADDED=256,
            USE_ALIBI_SLOPES=False,
            USE_ALIBI_SQRT=False,
            USE_QQ_BIAS=False,
            USE_SOFTCAP=False,
            USE_SINKS=False,
            USE_MM_PREFIX=False,
            MAX_MM_RANGES=0,
            mm_prefix_range_ptr=None,
            SLIDING_WINDOW=0,
            stride_k_cache_0=k.stride(0),
            stride_k_cache_1=k.stride(1),
            stride_k_cache_2=k.stride(2),
            stride_k_cache_3=k.stride(3),
            stride_v_cache_0=v.stride(0),
            stride_v_cache_1=v.stride(1),
            stride_v_cache_2=v.stride(2),
            stride_v_cache_3=v.stride(3),
            query_start_len_ptr=cu_seqlens_q,
            BLOCK_Q=block_q,
            num_seqs=num_seqs,
            BLOCK_M=block_m,
            USE_FP8=False,
            num_warps=num_warps,
            num_stages=1,
            waves_per_eu=1,
        )
        hits["n"] += 1
        if hits["n"] <= 3:
            print(
                f"[K100 27B R120 prefill] hit={hits['n']} q={int(q.shape[0])} "
                f"maxk={max_k} BS={block_size} {geometry}",
                flush=True,
            )
        return None

    ua.unified_attention = tuned

    # The core Triton backend can bind unified_attention while this module is
    # being imported through HCU plugin initialization.  Replace that early
    # module-global alias as well; its forward() resolves this global at runtime.
    import sys
    rebound = []
    for _name in (
        "vllm.v1.attention.backends.triton_attn",
        "vllm.v1.attention.backends.tree_attn",
    ):
        _mod = sys.modules.get(_name)
        if _mod is not None and hasattr(_mod, "unified_attention"):
            setattr(_mod, "unified_attention", tuned)
            rebound.append(_name.rsplit(".", 1)[-1])

    ua._k100_r120_installed = True
    ua._k100_r120_original = original
    _INSTALLED = True
    print(
        f"[K100 27B R134 prefill] exact BM64/BQ10/T32 qxK adaptive 4w/8w "
        f"gate={os.getenv('K100_27B_PREFILL_GATE','3072')} rebound={rebound}",
        flush=True,
    )
