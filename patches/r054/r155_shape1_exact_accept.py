from __future__ import annotations

_INSTALLED = False


def install():
    global _INSTALLED
    if _INSTALLED:
        return

    import torch
    from vllm_hcu.v1.hcu_model_runner import GPUModelRunner

    if getattr(GPUModelRunner, "_k100_r155_shape1_exact_accept", False):
        _INSTALLED = True
        return

    fallback = GPUModelRunner._update_states_after_model_execute

    def update(self, output_token_ids, scheduler_output):
        # Exact algebraic simplification of the stock accepted-count path for
        # the common single-request/non-align shape [1, 1]. Stock computes:
        #   cat([token, -1]) == -1 -> int -> argmax
        # which is exactly 1 for token != -1 and 0 for token == -1.
        # Keep the same GPU->pinned-CPU mirror copy and event ordering.
        if (
            not self.speculative_config
            or not self.model_config.is_hybrid
            or self.cache_config.mamba_cache_mode == "align"
            or output_token_ids.size(0) != 1
            or output_token_ids.size(1) != 1
            or len(self.input_batch.req_ids) != 1
        ):
            return fallback(self, output_token_ids, scheduler_output)

        # R171 production-form fix: eliminate R155's temporary GPU source
        # allocation entirely. Stock argmax ultimately writes int64 accepted
        # counts; torch.ne supports an int64 `out`, so write the exact 0/1
        # result directly into the persistent accepted-count GPU buffer.
        # This removes both the temporary lifetime hazard proven by R170 and
        # R155's extra int32->int64 GPU copy/conversion.
        torch.ne(
            output_token_ids[:, 0],
            -1,
            out=self.num_accepted_tokens.gpu[:1],
        )
        self.input_batch.num_accepted_tokens_cpu_tensor[:1].copy_(
            self.num_accepted_tokens.gpu[:1], non_blocking=True
        )
        assert self.num_accepted_tokens_event is not None
        self.num_accepted_tokens_event.record()
        return None

    GPUModelRunner._update_states_after_model_execute = update
    GPUModelRunner._k100_r155_shape1_exact_accept = True
    _INSTALLED = True
    print(
        "[K100 27B R171] exact shape1 accepted-count direct persistent-buffer write installed",
        flush=True,
    )
