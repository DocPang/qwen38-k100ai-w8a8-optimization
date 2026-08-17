from __future__ import annotations

import torch
from vllm import _custom_ops as ops
from vllm.triton_utils import triton, tl
from vllm.model_executor.models.qwen2_moe import Qwen2MoeMLP

D = 17408
BA = 512
BQ = 256
NP = triton.cdiv(D, BA)
NQ = triton.cdiv(D, BQ)
_INSTALLED = False


@triton.jit
def _partial_amax_rows(
    x_ptr,
    partial_ptr,
    D: tl.constexpr,
    NP: tl.constexpr,
    BS: tl.constexpr,
):
    pid = tl.program_id(0)
    row = pid // NP
    part = pid - row * NP
    offs = part * BS + tl.arange(0, BS)
    mask = offs < D
    base = row * (2 * D)

    gate = tl.load(x_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(x_ptr + base + D + offs, mask=mask, other=0.0).to(tl.float32)
    # Match Inductor compiled graph semantics:
    # FP32 SiLU -> FP32 multiply -> final BF16 materialization.
    # This differs from eager silu_and_mul and is required for exact MTP.
    y_bf16 = ((gate * tl.sigmoid(gate)) * up).to(tl.bfloat16)
    amax = tl.max(tl.abs(y_bf16.to(tl.float32)), axis=0)
    tl.store(partial_ptr + row * NP + part, amax)


@triton.jit
def _quant_rows_exact(
    x_ptr,
    q_ptr,
    scale_ptr,
    partial_ptr,
    D: tl.constexpr,
    NP: tl.constexpr,
    NQ: tl.constexpr,
    BS: tl.constexpr,
):
    pid = tl.program_id(0)
    row = pid // NQ
    qpart = pid - row * NQ

    poffs = tl.arange(0, 128)
    partial = tl.load(
        partial_ptr + row * NP + poffs,
        mask=poffs < NP,
        other=0.0,
    )
    amax = tl.max(partial, axis=0)
    scale = amax / 127.0
    # Exact K100AI HCU dynamic INT8 semantics recovered by differential tests:
    # the reciprocal is derived directly from amax, not from rounded scale.
    inv_scale = tl.where(amax > 0.0, 127.0 / amax, 0.0)
    if qpart == 0:
        tl.store(scale_ptr + row, scale)

    offs = qpart * BS + tl.arange(0, BS)
    mask = offs < D
    base = row * (2 * D)
    gate = tl.load(x_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(x_ptr + base + D + offs, mask=mask, other=0.0).to(tl.float32)
    # Match Inductor compiled graph semantics: FP32 SiLU -> FP32 multiply -> BF16.
    y_bf16 = ((gate * tl.sigmoid(gate)) * up).to(tl.bfloat16)

    value = y_bf16.to(tl.float32) * inv_scale
    av = tl.abs(value)
    floored = tl.floor(av)
    frac = av - floored
    floor_i = floored.to(tl.int32)
    inc = (frac > 0.5) | ((frac == 0.5) & ((floor_i & 1) != 0))
    rounded = floored + inc.to(tl.float32)
    rounded = tl.where(value < 0.0, -rounded, rounded)
    tl.store(q_ptr + row * D + offs, rounded.to(tl.int8), mask=mask)


@torch.library.custom_op(
    "k100_27b::r041_swiglu_int8_quant",
    mutates_args=(),
    device_types="cuda",
)
def swiglu_int8_quant(gate_up: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    x = gate_up.contiguous()
    m = int(x.shape[0])
    d2 = int(x.shape[1])
    if d2 == 2 * D:
        partial = torch.empty((m, NP), device=x.device, dtype=torch.float32)
        q = torch.empty((m, D), device=x.device, dtype=torch.int8)
        scale = torch.empty((m, 1), device=x.device, dtype=torch.float32)
        _partial_amax_rows[(m * NP,)](
            x,
            partial,
            D=D,
            NP=NP,
            BS=BA,
            num_warps=4,
        )
        _quant_rows_exact[(m * NQ,)](
            x,
            q,
            scale,
            partial,
            D=D,
            NP=NP,
            NQ=NQ,
            BS=BQ,
            num_warps=4,
        )
        return q, scale

    d = d2 // 2
    activated = torch.empty((m, d), device=x.device, dtype=x.dtype)
    torch.ops._C.silu_and_mul(activated, x)
    q, scale, zp = ops.scaled_int8_quant(
        activated.contiguous(), None, None, symmetric=True
    )
    assert zp is None
    return q, scale


@swiglu_int8_quant.register_fake
def _swiglu_int8_quant_fake(
    gate_up: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    d = gate_up.shape[-1] // 2
    q = gate_up.new_empty((*gate_up.shape[:-1], d), dtype=torch.int8)
    scale = gate_up.new_empty((*gate_up.shape[:-1], 1), dtype=torch.float32)
    return q, scale


def install(shapeaware_w8a8_linear) -> None:
    global _INSTALLED
    if _INSTALLED:
        return

    original_forward = Qwen2MoeMLP.forward

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        down = self.down_proj
        if (
            self.expert_gate is not None
            or int(getattr(down, "tp_size", 1)) != 1
            or down.bias is not None
            or not hasattr(down, "weight_scale")
            or down.weight.dtype is not torch.int8
            or tuple(down.weight.shape) != (D, 5120)
        ):
            return original_forward(self, x)

        gate_up, _ = self.gate_up_proj(x)
        if int(gate_up.shape[-1]) != 2 * D:
            return original_forward(self, x)
        q, scale = swiglu_int8_quant(gate_up)
        return shapeaware_w8a8_linear(q, down.weight, scale, down.weight_scale)

    Qwen2MoeMLP.forward = _forward
    _INSTALLED = True
    print(
        "[K100 27B R042 SwiGLU->INT8] Inductor-exact all-M fusion installed",
        flush=True,
    )
