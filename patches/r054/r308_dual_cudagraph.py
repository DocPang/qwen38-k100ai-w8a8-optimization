"""R308: add a single-request M=1 FULL CUDAGraph to an MTP3 engine.

With speculative decoding configured, vLLM sets uniform_decode_query_len=4 and
only captures the M=4 FULL decode graph.  R305 can turn long-context requests
into true one-token/no-draft steps, but those M=1 steps then miss FULL graph
replay and fall back to a slower path.

R308 keeps the existing M=4 graph and adds one exact BatchDescriptor for
single-request M=1 decode.  During capture of that descriptor only, the runner
and dispatcher temporarily use uniform_decode_query_len=1.  Runtime dispatch
recognizes the exact M=1 single-request case and replays that graph.
"""
from __future__ import annotations

import os

_INSTALLED = False


def install() -> None:
    global _INSTALLED
    if _INSTALLED or os.getenv("K100_R308_DUAL_CUDAGRAPH", "0") != "1":
        return

    from vllm.config import CUDAGraphMode
    from vllm.forward_context import BatchDescriptor
    from vllm.v1.cudagraph_dispatcher import CudagraphDispatcher
    from vllm_hcu.v1.hcu_model_runner import GPUModelRunner

    if getattr(CudagraphDispatcher, "_k100_r308_installed", False):
        _INSTALLED = True
        return

    original_init_keys = CudagraphDispatcher.initialize_cudagraph_keys
    original_dispatch = CudagraphDispatcher.dispatch
    original_is_uniform = GPUModelRunner._is_uniform_decode
    original_warmup_capture = GPUModelRunner._warmup_and_capture

    def init_keys_r308(self, cudagraph_mode, uniform_decode_query_len=1):
        original_init_keys(self, cudagraph_mode, uniform_decode_query_len)
        # Only relevant to an MTP engine whose native FULL graph is M>1.
        if (
            uniform_decode_query_len > 1
            and cudagraph_mode.decode_mode() == CUDAGraphMode.FULL
            and cudagraph_mode.separate_routine()
        ):
            desc = BatchDescriptor(
                num_tokens=1,
                num_reqs=1,
                uniform=True,
                has_lora=False,
                num_active_loras=0,
            )
            self.add_cudagraph_key(CUDAGraphMode.FULL, desc)
            print(
                "[K100 R308 dual CG] added FULL M1 key alongside "
                f"native M{uniform_decode_query_len}",
                flush=True,
            )

    def dispatch_r308(
        self,
        num_tokens: int,
        uniform_decode: bool = False,
        has_lora: bool = False,
        num_active_loras: int = 0,
        valid_modes=None,
        invalid_modes=None,
    ):
        # Exact single-request M=1 path.  Avoid the stock descriptor builder,
        # which assumes the engine-wide MTP query length (4) and cannot build
        # a uniform descriptor for num_tokens=1.
        if uniform_decode and num_tokens == 1 and not has_lora:
            allowed = valid_modes or CUDAGraphMode.valid_runtime_modes()
            if invalid_modes:
                allowed -= invalid_modes
            desc = BatchDescriptor(
                num_tokens=1,
                num_reqs=1,
                uniform=True,
                has_lora=False,
                num_active_loras=0,
            )
            if (
                CUDAGraphMode.FULL in allowed
                and desc in self.cudagraph_keys[CUDAGraphMode.FULL]
            ):
                return CUDAGraphMode.FULL, desc
        return original_dispatch(
            self,
            num_tokens=num_tokens,
            uniform_decode=uniform_decode,
            has_lora=has_lora,
            num_active_loras=num_active_loras,
            valid_modes=valid_modes,
            invalid_modes=invalid_modes,
        )

    @staticmethod
    def is_uniform_r308(
        max_num_scheduled_tokens: int,
        uniform_decode_query_len: int,
        num_tokens: int,
        num_reqs: int,
        force_uniform_decode=None,
    ) -> bool:
        if force_uniform_decode is not None:
            return force_uniform_decode
        # Accept both the native MTP verifier width and true one-token decode.
        if max_num_scheduled_tokens == 1 and num_tokens == num_reqs:
            return True
        return original_is_uniform(
            max_num_scheduled_tokens,
            uniform_decode_query_len,
            num_tokens,
            num_reqs,
            force_uniform_decode,
        )

    def warmup_capture_r308(
        self,
        desc,
        cudagraph_runtime_mode,
        profile_seq_lens=None,
        allow_microbatching=False,
        num_warmups=None,
    ):
        is_m1_full = (
            cudagraph_runtime_mode == CUDAGraphMode.FULL
            and desc.uniform
            and desc.num_tokens == 1
            and desc.num_reqs == 1
            and self.uniform_decode_query_len > 1
        )
        if not is_m1_full:
            return original_warmup_capture(
                self,
                desc,
                cudagraph_runtime_mode,
                profile_seq_lens=profile_seq_lens,
                allow_microbatching=allow_microbatching,
                num_warmups=num_warmups,
            )

        old_runner_q = self.uniform_decode_query_len
        old_dispatch_q = self.cudagraph_dispatcher.uniform_decode_query_len
        self.uniform_decode_query_len = 1
        self.cudagraph_dispatcher.uniform_decode_query_len = 1
        try:
            print("[K100 R308 dual CG] capturing FULL M1 graph", flush=True)
            return original_warmup_capture(
                self,
                desc,
                cudagraph_runtime_mode,
                profile_seq_lens=profile_seq_lens,
                allow_microbatching=allow_microbatching,
                num_warmups=num_warmups,
            )
        finally:
            self.uniform_decode_query_len = old_runner_q
            self.cudagraph_dispatcher.uniform_decode_query_len = old_dispatch_q

    CudagraphDispatcher.initialize_cudagraph_keys = init_keys_r308
    CudagraphDispatcher.dispatch = dispatch_r308
    CudagraphDispatcher._k100_r308_installed = True
    GPUModelRunner._is_uniform_decode = is_uniform_r308
    GPUModelRunner._warmup_and_capture = warmup_capture_r308
    _INSTALLED = True
    print("[K100 R308 dual CG] hooks installed", flush=True)
