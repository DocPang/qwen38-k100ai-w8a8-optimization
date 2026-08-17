from __future__ import annotations

import torch

_INSTALLED = False


def install() -> None:
    global _INSTALLED
    if _INSTALLED:
        return

    from vllm_hcu.v1.hcu_model_runner import GPUModelRunner

    if getattr(GPUModelRunner, "_k100_r143_valid_count_int32", False):
        _INSTALLED = True
        return

    original_init = GPUModelRunner.__init__

    def _r143_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        buf = getattr(self, "valid_sampled_token_count_cpu", None)
        if buf is not None and buf.dtype is torch.int64:
            self.valid_sampled_token_count_cpu = torch.empty(
                buf.shape,
                dtype=torch.int32,
                device="cpu",
                pin_memory=bool(getattr(self, "pin_memory", False)),
            )
            print(
                "[K100 27B R143 valid-count D2H] CPU mirror int64->int32; semantics unchanged",
                flush=True,
            )

    GPUModelRunner.__init__ = _r143_init
    GPUModelRunner._k100_r143_valid_count_int32 = True
    _INSTALLED = True
    print("[K100 27B R143 valid-count D2H] init hook installed", flush=True)
