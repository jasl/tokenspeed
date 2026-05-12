# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Native CUDA MXFP4 MoE kernels for SM12x."""

from __future__ import annotations

import functools
import os
from pathlib import Path

import torch


@functools.cache
def _load_sm12x_mxfp4_moe_module():
    import tvm_ffi

    objs_dir = Path(__file__).parent / "objs" / "sm12x_mxfp4_moe"
    so_path = objs_dir / "sm12x_mxfp4_moe.so"
    if not so_path.exists():
        raise RuntimeError(
            f"tokenspeed_kernel sm12x_mxfp4_moe library not found at {so_path}. "
            "Run: pip install -e tokenspeed-kernel/python/"
        )
    return tvm_ffi.load_module(str(so_path))


def _empty_float32_like(x: torch.Tensor) -> torch.Tensor:
    return torch.empty((0,), dtype=torch.float32, device=x.device)


def _empty_int32_like(x: torch.Tensor) -> torch.Tensor:
    return torch.empty((0,), dtype=torch.int32, device=x.device)


def _validate_work_buffer(
    name: str,
    tensor: torch.Tensor,
    *,
    shape: tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    if tuple(tensor.shape) != shape:
        raise ValueError(f"{name} must have shape {shape}, got {tuple(tensor.shape)}")
    if tensor.dtype != dtype:
        raise ValueError(f"{name} must use {dtype}, got {tensor.dtype}")
    if tensor.device != device:
        raise ValueError(f"{name} must be on {device}, got {tensor.device}")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")
    return tensor


def _sm12x_mxfp4_moe_forward_impl(
    op_name: str,
    hidden_states: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    w13_weight: torch.Tensor,
    w13_scale: torch.Tensor,
    w2_weight: torch.Tensor,
    w2_scale: torch.Tensor,
    *,
    w13_bias: torch.Tensor | None = None,
    w2_bias: torch.Tensor | None = None,
    activation: str = "swiglu",
    swiglu_alpha: float | None = None,
    swiglu_limit: float | None = None,
    swiglu_beta: float | None = None,
    ep_rank: int = 0,
    ep_size: int = 1,
    output: torch.Tensor | None = None,
    intermediate: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run one native SM12x MXFP4 MoE implementation.

    This path keeps the checkpoint's canonical packed layout and performs
    on-the-fly MXFP4 dequantization. It is intentionally layout-native: no
    whole-weight Triton/FlashInfer swizzle is required at load time.
    """
    if hidden_states.dim() != 2:
        raise ValueError(f"hidden_states must be 2-D, got {hidden_states.dim()}")
    if topk_ids.shape != topk_weights.shape:
        raise ValueError(
            f"topk_ids shape {tuple(topk_ids.shape)} must match topk_weights "
            f"shape {tuple(topk_weights.shape)}"
        )
    if activation not in {"silu", "swiglu"}:
        raise ValueError(f"Unsupported SM12x MXFP4 activation: {activation}")

    hidden_states = hidden_states.contiguous()
    topk_weights = topk_weights.to(torch.float32).contiguous()
    topk_ids = topk_ids.to(torch.int32).contiguous()
    w13_weight = w13_weight.contiguous()
    w13_scale = w13_scale.contiguous()
    w2_weight = w2_weight.contiguous()
    w2_scale = w2_scale.contiguous()

    num_tokens = hidden_states.shape[0]
    if num_tokens == 0:
        return torch.empty_like(hidden_states)

    intermediate_size = w2_weight.shape[2] * 2
    intermediate_shape = (num_tokens, topk_ids.shape[1], intermediate_size)
    if intermediate is None:
        intermediate = torch.empty(
            intermediate_shape,
            dtype=torch.float32,
            device=hidden_states.device,
        )
    else:
        intermediate = _validate_work_buffer(
            "intermediate",
            intermediate,
            shape=intermediate_shape,
            dtype=torch.float32,
            device=hidden_states.device,
        )
    if output is None:
        output = torch.empty_like(hidden_states)
    else:
        output = _validate_work_buffer(
            "output",
            output,
            shape=tuple(hidden_states.shape),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )

    if w13_bias is None:
        w13_bias = _empty_float32_like(hidden_states)
    else:
        w13_bias = w13_bias.to(torch.float32).contiguous()
    if w2_bias is None:
        w2_bias = _empty_float32_like(hidden_states)
    else:
        w2_bias = w2_bias.to(torch.float32).contiguous()

    mod = _load_sm12x_mxfp4_moe_module()
    getattr(mod, op_name)(
        output,
        intermediate,
        hidden_states,
        topk_weights,
        topk_ids,
        w13_weight,
        w13_scale,
        w2_weight,
        w2_scale,
        w13_bias,
        w2_bias,
        int(ep_rank),
        int(ep_size),
        bool(activation == "swiglu"),
        float(0.0 if swiglu_alpha is None else swiglu_alpha),
        bool(swiglu_alpha is not None),
        float(0.0 if swiglu_limit is None else swiglu_limit),
        bool(swiglu_limit is not None),
        float(0.0 if swiglu_beta is None else swiglu_beta),
        bool(swiglu_beta is not None),
    )
    return output


def sm12x_mxfp4_moe_forward_scalar(
    hidden_states: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    w13_weight: torch.Tensor,
    w13_scale: torch.Tensor,
    w2_weight: torch.Tensor,
    w2_scale: torch.Tensor,
    **kwargs,
) -> torch.Tensor:
    return _sm12x_mxfp4_moe_forward_impl(
        "sm12x_mxfp4_moe_forward_scalar",
        hidden_states,
        topk_weights,
        topk_ids,
        w13_weight,
        w13_scale,
        w2_weight,
        w2_scale,
        **kwargs,
    )


def sm12x_mxfp4_moe_forward_warp(
    hidden_states: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    w13_weight: torch.Tensor,
    w13_scale: torch.Tensor,
    w2_weight: torch.Tensor,
    w2_scale: torch.Tensor,
    **kwargs,
) -> torch.Tensor:
    return _sm12x_mxfp4_moe_forward_impl(
        "sm12x_mxfp4_moe_forward_warp",
        hidden_states,
        topk_weights,
        topk_ids,
        w13_weight,
        w13_scale,
        w2_weight,
        w2_scale,
        **kwargs,
    )


def sm12x_mxfp4_moe_forward(
    hidden_states: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    w13_weight: torch.Tensor,
    w13_scale: torch.Tensor,
    w2_weight: torch.Tensor,
    w2_scale: torch.Tensor,
    **kwargs,
) -> torch.Tensor:
    impl = os.getenv("TOKENSPEED_SM12X_MXFP4_MOE_IMPL", "warp").strip().lower()
    if impl == "warp":
        forward_impl = sm12x_mxfp4_moe_forward_warp
    elif impl == "scalar":
        forward_impl = sm12x_mxfp4_moe_forward_scalar
    else:
        raise ValueError(
            "TOKENSPEED_SM12X_MXFP4_MOE_IMPL must be 'warp' or 'scalar', "
            f"got {impl!r}"
        )
    return forward_impl(
        hidden_states,
        topk_weights,
        topk_ids,
        w13_weight,
        w13_scale,
        w2_weight,
        w2_scale,
        **kwargs,
    )


def sm12x_mxfp8_mxfp4_mma_tile(
    activations: torch.Tensor,
    activation_scale: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
) -> torch.Tensor:
    if activations.shape != (16, 32):
        raise ValueError(
            f"activations must have shape (16, 32), got {activations.shape}"
        )
    if activation_scale.shape != (16, 1):
        raise ValueError(
            f"activation_scale must have shape (16, 1), got {activation_scale.shape}"
        )
    if weight.shape != (8, 16):
        raise ValueError(f"weight must have shape (8, 16), got {weight.shape}")
    if weight_scale.shape != (8, 1):
        raise ValueError(
            f"weight_scale must have shape (8, 1), got {weight_scale.shape}"
        )
    if activations.dtype != torch.float8_e4m3fn:
        raise ValueError(
            f"activations must use torch.float8_e4m3fn, got {activations.dtype}"
        )
    if activation_scale.dtype != torch.uint8:
        raise ValueError(
            f"activation_scale must use torch.uint8, got {activation_scale.dtype}"
        )
    if weight.dtype != torch.uint8:
        raise ValueError(f"weight must use torch.uint8, got {weight.dtype}")
    if weight_scale.dtype != torch.uint8:
        raise ValueError(f"weight_scale must use torch.uint8, got {weight_scale.dtype}")

    output = torch.empty((16, 8), dtype=torch.float32, device=activations.device)
    mod = _load_sm12x_mxfp4_moe_module()
    mod.sm12x_mxfp8_mxfp4_mma_tile(
        output,
        activations.contiguous(),
        activation_scale.contiguous(),
        weight.contiguous(),
        weight_scale.contiguous(),
    )
    return output


def sm12x_mxfp4_mxfp8_mma_tile(
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    activations: torch.Tensor,
    activation_scale: torch.Tensor,
) -> torch.Tensor:
    if weight.shape != (16, 16):
        raise ValueError(f"weight must have shape (16, 16), got {weight.shape}")
    if weight_scale.shape != (16, 1):
        raise ValueError(
            f"weight_scale must have shape (16, 1), got {weight_scale.shape}"
        )
    if activations.shape != (8, 32):
        raise ValueError(
            f"activations must have shape (8, 32), got {activations.shape}"
        )
    if activation_scale.shape != (8, 1):
        raise ValueError(
            f"activation_scale must have shape (8, 1), got {activation_scale.shape}"
        )
    if weight.dtype != torch.uint8:
        raise ValueError(f"weight must use torch.uint8, got {weight.dtype}")
    if weight_scale.dtype != torch.uint8:
        raise ValueError(f"weight_scale must use torch.uint8, got {weight_scale.dtype}")
    if activations.dtype != torch.float8_e4m3fn:
        raise ValueError(
            f"activations must use torch.float8_e4m3fn, got {activations.dtype}"
        )
    if activation_scale.dtype != torch.uint8:
        raise ValueError(
            f"activation_scale must use torch.uint8, got {activation_scale.dtype}"
        )

    output = torch.empty((16, 8), dtype=torch.float32, device=activations.device)
    mod = _load_sm12x_mxfp4_moe_module()
    mod.sm12x_mxfp4_mxfp8_mma_tile(
        output,
        weight.contiguous(),
        weight_scale.contiguous(),
        activations.contiguous(),
        activation_scale.contiguous(),
    )
    return output


def sm12x_mxfp4_mxfp8_dense(
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    activations: torch.Tensor,
    activation_scale: torch.Tensor,
) -> torch.Tensor:
    if weight.dim() != 2:
        raise ValueError(f"weight must be 2-D, got {weight.dim()}")
    if weight_scale.dim() != 2:
        raise ValueError(f"weight_scale must be 2-D, got {weight_scale.dim()}")
    if activations.dim() != 2:
        raise ValueError(f"activations must be 2-D, got {activations.dim()}")
    if activation_scale.dim() != 2:
        raise ValueError(f"activation_scale must be 2-D, got {activation_scale.dim()}")
    if weight.dtype != torch.uint8:
        raise ValueError(f"weight must use torch.uint8, got {weight.dtype}")
    if weight_scale.dtype != torch.uint8:
        raise ValueError(f"weight_scale must use torch.uint8, got {weight_scale.dtype}")
    if activations.dtype != torch.float8_e4m3fn:
        raise ValueError(
            f"activations must use torch.float8_e4m3fn, got {activations.dtype}"
        )
    if activation_scale.dtype != torch.uint8:
        raise ValueError(
            f"activation_scale must use torch.uint8, got {activation_scale.dtype}"
        )

    n, packed_k = weight.shape
    m, k = activations.shape
    if m % 8 != 0:
        raise ValueError(f"activations rows must be a multiple of 8, got {m}")
    if n % 16 != 0:
        raise ValueError(f"weight rows must be a multiple of 16, got {n}")
    if k % 32 != 0:
        raise ValueError(f"activations columns must be a multiple of 32, got {k}")
    if packed_k != k // 2:
        raise ValueError(
            f"weight must have {k // 2} packed columns for K={k}, got {packed_k}"
        )
    expected_weight_scale_shape = (n, k // 32)
    if weight_scale.shape != expected_weight_scale_shape:
        raise ValueError(
            "weight_scale must have shape "
            f"{expected_weight_scale_shape}, got {tuple(weight_scale.shape)}"
        )
    expected_activation_scale_shape = (m, k // 32)
    if activation_scale.shape != expected_activation_scale_shape:
        raise ValueError(
            "activation_scale must have shape "
            f"{expected_activation_scale_shape}, got {tuple(activation_scale.shape)}"
        )

    output = torch.empty((m, n), dtype=torch.float32, device=activations.device)
    mod = _load_sm12x_mxfp4_moe_module()
    mod.sm12x_mxfp4_mxfp8_dense(
        output,
        weight.contiguous(),
        weight_scale.contiguous(),
        activations.contiguous(),
        activation_scale.contiguous(),
    )
    return output


def sm12x_mxfp8_mxfp4_dense(
    activations: torch.Tensor,
    activation_scale: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
) -> torch.Tensor:
    if activations.dim() != 2:
        raise ValueError(f"activations must be 2-D, got {activations.dim()}")
    if activation_scale.dim() != 2:
        raise ValueError(f"activation_scale must be 2-D, got {activation_scale.dim()}")
    if weight.dim() != 2:
        raise ValueError(f"weight must be 2-D, got {weight.dim()}")
    if weight_scale.dim() != 2:
        raise ValueError(f"weight_scale must be 2-D, got {weight_scale.dim()}")
    if activations.dtype != torch.float8_e4m3fn:
        raise ValueError(
            f"activations must use torch.float8_e4m3fn, got {activations.dtype}"
        )
    if activation_scale.dtype != torch.uint8:
        raise ValueError(
            f"activation_scale must use torch.uint8, got {activation_scale.dtype}"
        )
    if weight.dtype != torch.uint8:
        raise ValueError(f"weight must use torch.uint8, got {weight.dtype}")
    if weight_scale.dtype != torch.uint8:
        raise ValueError(f"weight_scale must use torch.uint8, got {weight_scale.dtype}")

    m, k = activations.shape
    n = weight.shape[0]
    if m % 16 != 0:
        raise ValueError(f"activations rows must be a multiple of 16, got {m}")
    if n % 8 != 0:
        raise ValueError(f"weight rows must be a multiple of 8, got {n}")
    if k % 32 != 0:
        raise ValueError(f"activations columns must be a multiple of 32, got {k}")
    if weight.shape[1] != k // 2:
        raise ValueError(
            f"weight must have {k // 2} packed columns for K={k}, got {weight.shape[1]}"
        )
    expected_scale_shape = (m, k // 32)
    if activation_scale.shape != expected_scale_shape:
        raise ValueError(
            "activation_scale must have shape "
            f"{expected_scale_shape}, got {tuple(activation_scale.shape)}"
        )
    expected_weight_scale_shape = (n, k // 32)
    if weight_scale.shape != expected_weight_scale_shape:
        raise ValueError(
            "weight_scale must have shape "
            f"{expected_weight_scale_shape}, got {tuple(weight_scale.shape)}"
        )

    output = torch.empty((m, n), dtype=torch.float32, device=activations.device)
    mod = _load_sm12x_mxfp4_moe_module()
    mod.sm12x_mxfp8_mxfp4_dense(
        output,
        activations.contiguous(),
        activation_scale.contiguous(),
        weight.contiguous(),
        weight_scale.contiguous(),
    )
    return output


def sm12x_mxfp4_mxfp8_quantize(
    values: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if values.dim() != 2:
        raise ValueError(f"values must be 2-D, got {values.dim()}")
    if values.dtype not in {
        torch.float32,
        torch.float16,
        torch.bfloat16,
    }:
        raise ValueError(
            "values must use float32, float16, or bfloat16, " f"got {values.dtype}"
        )
    if not values.is_cuda:
        raise ValueError("values must be a CUDA tensor")

    rows, hidden_dim = values.shape
    if hidden_dim % 32 != 0:
        raise ValueError(f"values columns must be divisible by 32, got {hidden_dim}")

    output = torch.empty(
        (rows, hidden_dim), dtype=torch.float8_e4m3fn, device=values.device
    )
    output_scale = torch.empty(
        (rows, hidden_dim // 32), dtype=torch.uint8, device=values.device
    )
    if rows == 0:
        return output, output_scale

    mod = _load_sm12x_mxfp4_moe_module()
    mod.sm12x_mxfp4_mxfp8_quantize(
        output,
        output_scale,
        values.contiguous(),
    )
    return output, output_scale


def sm12x_mxfp4_swiglu_mxfp8_quantize(
    gate_up: torch.Tensor,
    expert_ids: torch.Tensor,
    *,
    w13_bias: torch.Tensor | None = None,
    activation: str = "swiglu",
    swiglu_alpha: float | None = None,
    swiglu_limit: float | None = None,
    swiglu_beta: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if activation not in {"silu", "swiglu"}:
        raise ValueError(f"Unsupported SM12x MXFP4 activation: {activation}")
    if gate_up.dim() != 2:
        raise ValueError(f"gate_up must be 2-D, got {gate_up.dim()}")
    if expert_ids.dim() != 1:
        raise ValueError(f"expert_ids must be 1-D, got {expert_ids.dim()}")
    if gate_up.dtype != torch.float32:
        raise ValueError(f"gate_up must use torch.float32, got {gate_up.dtype}")
    if expert_ids.dtype != torch.int32:
        raise ValueError(f"expert_ids must use torch.int32, got {expert_ids.dtype}")
    if not gate_up.is_cuda:
        raise ValueError("gate_up must be a CUDA tensor")
    if expert_ids.device != gate_up.device:
        raise ValueError("expert_ids must be on the gate_up device")

    rows, gate_up_width = gate_up.shape
    if expert_ids.shape != (rows,):
        raise ValueError(
            f"expert_ids must have shape ({rows},), got {expert_ids.shape}"
        )
    if gate_up_width % 2 != 0:
        raise ValueError(f"gate_up width must be even, got {gate_up_width}")
    intermediate_size = gate_up_width // 2
    if intermediate_size % 32 != 0:
        raise ValueError(
            f"intermediate_size must be divisible by 32, got {intermediate_size}"
        )

    if rows == 0:
        return (
            torch.empty(
                (0, intermediate_size),
                dtype=torch.float8_e4m3fn,
                device=gate_up.device,
            ),
            torch.empty(
                (0, intermediate_size // 32),
                dtype=torch.uint8,
                device=gate_up.device,
            ),
        )

    if w13_bias is None:
        w13_bias = _empty_float32_like(gate_up)
    else:
        if w13_bias.dim() != 2:
            raise ValueError(f"w13_bias must be 2-D, got {w13_bias.dim()}")
        if w13_bias.shape[1] != gate_up_width:
            raise ValueError(
                "w13_bias must have second dimension "
                f"{gate_up_width}, got {w13_bias.shape[1]}"
            )
        if w13_bias.device != gate_up.device:
            raise ValueError("w13_bias must be on the gate_up device")
        w13_bias = w13_bias.to(torch.float32).contiguous()

    output = torch.empty(
        (rows, intermediate_size), dtype=torch.float8_e4m3fn, device=gate_up.device
    )
    output_scale = torch.empty(
        (rows, intermediate_size // 32), dtype=torch.uint8, device=gate_up.device
    )
    mod = _load_sm12x_mxfp4_moe_module()
    mod.sm12x_mxfp4_swiglu_mxfp8_quantize(
        output,
        output_scale,
        gate_up.contiguous(),
        expert_ids.contiguous(),
        w13_bias,
        bool(activation == "swiglu"),
        float(0.0 if swiglu_alpha is None else swiglu_alpha),
        bool(swiglu_alpha is not None),
        float(0.0 if swiglu_limit is None else swiglu_limit),
        bool(swiglu_limit is not None),
        float(0.0 if swiglu_beta is None else swiglu_beta),
        bool(swiglu_beta is not None),
    )
    return output, output_scale


def sm12x_mxfp4_moe_w13_tensorcore(
    hidden_fp8: torch.Tensor,
    hidden_scale: torch.Tensor,
    topk_ids: torch.Tensor,
    w13_weight: torch.Tensor,
    w13_scale: torch.Tensor,
    *,
    ep_rank: int = 0,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Tensorcore MoE W13 GEMM (no SwiGLU, no bias).

    Inputs are FP8 e4m3 hidden activations with per-32 ue8m0 scales (produced
    by ``sm12x_mxfp4_mxfp8_quantize``) and packed MXFP4 W13 weights. The output
    is the raw gate/up logits in float32 layout ``[num_tokens, top_k,
    gate_up_dim]``; the downstream ``sm12x_mxfp4_swiglu_mxfp8_quantize`` helper
    is responsible for adding ``w13_bias``, applying SwiGLU, and quantising
    back to FP8 for the W2 step.
    """
    if hidden_fp8.dim() != 2:
        raise ValueError(f"hidden_fp8 must be 2-D, got {hidden_fp8.dim()}")
    if hidden_scale.dim() != 2:
        raise ValueError(f"hidden_scale must be 2-D, got {hidden_scale.dim()}")
    if topk_ids.dim() != 2:
        raise ValueError(f"topk_ids must be 2-D, got {topk_ids.dim()}")
    if w13_weight.dim() != 3:
        raise ValueError(f"w13_weight must be 3-D, got {w13_weight.dim()}")
    if w13_scale.dim() != 3:
        raise ValueError(f"w13_scale must be 3-D, got {w13_scale.dim()}")
    if hidden_fp8.dtype != torch.float8_e4m3fn:
        raise ValueError(f"hidden_fp8 must use float8_e4m3fn, got {hidden_fp8.dtype}")
    if hidden_scale.dtype != torch.uint8:
        raise ValueError(f"hidden_scale must use torch.uint8, got {hidden_scale.dtype}")
    if w13_weight.dtype != torch.uint8:
        raise ValueError(f"w13_weight must use torch.uint8, got {w13_weight.dtype}")
    if w13_scale.dtype != torch.uint8:
        raise ValueError(f"w13_scale must use torch.uint8, got {w13_scale.dtype}")
    if not hidden_fp8.is_cuda:
        raise ValueError("hidden_fp8 must be a CUDA tensor")

    num_tokens, hidden = hidden_fp8.shape
    if topk_ids.shape[0] != num_tokens:
        raise ValueError(
            f"topk_ids first dim {topk_ids.shape[0]} must equal "
            f"num_tokens {num_tokens}"
        )
    top_k = topk_ids.shape[1]
    num_local_experts, gate_up_dim, weight_packed_k = w13_weight.shape
    if weight_packed_k * 2 != hidden:
        raise ValueError(
            f"w13_weight packed_k {weight_packed_k} must equal hidden/2 "
            f"({hidden // 2})"
        )
    if gate_up_dim % 16 != 0:
        raise ValueError(f"gate_up_dim must be a multiple of 16, got {gate_up_dim}")
    if hidden % 32 != 0:
        raise ValueError(f"hidden must be a multiple of 32, got {hidden}")
    if hidden_scale.shape != (num_tokens, hidden // 32):
        raise ValueError(
            f"hidden_scale must have shape ({num_tokens}, {hidden // 32}), "
            f"got {tuple(hidden_scale.shape)}"
        )
    if w13_scale.shape != (num_local_experts, gate_up_dim, hidden // 32):
        raise ValueError(
            f"w13_scale must have shape ({num_local_experts}, {gate_up_dim}, "
            f"{hidden // 32}), got {tuple(w13_scale.shape)}"
        )

    out_shape = (num_tokens, top_k, gate_up_dim)
    if out is None:
        out = torch.empty(out_shape, dtype=torch.float32, device=hidden_fp8.device)
    else:
        if out.shape != out_shape:
            raise ValueError(f"out must have shape {out_shape}, got {tuple(out.shape)}")
        if out.dtype != torch.float32:
            raise ValueError(f"out must use torch.float32, got {out.dtype}")
        if out.device != hidden_fp8.device:
            raise ValueError("out must be on the hidden_fp8 device")
        if not out.is_contiguous():
            raise ValueError("out must be contiguous")

    if num_tokens == 0 or top_k == 0:
        return out

    mod = _load_sm12x_mxfp4_moe_module()
    mod.sm12x_mxfp4_moe_w13_tensorcore(
        out,
        hidden_fp8.contiguous(),
        hidden_scale.contiguous(),
        topk_ids.to(torch.int32).contiguous(),
        w13_weight.contiguous(),
        w13_scale.contiguous(),
        int(ep_rank),
    )
    return out


def sm12x_mxfp4_moe_w2_tensorcore(
    intermediate_fp8: torch.Tensor,
    intermediate_scale: torch.Tensor,
    topk_ids: torch.Tensor,
    w2_weight: torch.Tensor,
    w2_scale: torch.Tensor,
    *,
    ep_rank: int = 0,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Tensorcore MoE W2 GEMM (per-pair output, no weighted reduce yet).

    Inputs are FP8 e4m3 post-SwiGLU intermediates (as produced by
    ``sm12x_mxfp4_swiglu_mxfp8_quantize``) and packed MXFP4 W2 weights. The
    output is a per-pair float32 tensor ``[num_tokens, top_k, hidden]``; the
    caller is expected to invoke ``sm12x_mxfp4_moe_weighted_reduce`` to fold
    in topk_weights and cast back to the model dtype.
    """
    if intermediate_fp8.dim() != 2:
        raise ValueError(f"intermediate_fp8 must be 2-D, got {intermediate_fp8.dim()}")
    if intermediate_scale.dim() != 2:
        raise ValueError(
            f"intermediate_scale must be 2-D, got {intermediate_scale.dim()}"
        )
    if topk_ids.dim() != 2:
        raise ValueError(f"topk_ids must be 2-D, got {topk_ids.dim()}")
    if w2_weight.dim() != 3:
        raise ValueError(f"w2_weight must be 3-D, got {w2_weight.dim()}")
    if w2_scale.dim() != 3:
        raise ValueError(f"w2_scale must be 3-D, got {w2_scale.dim()}")
    if intermediate_fp8.dtype != torch.float8_e4m3fn:
        raise ValueError(
            "intermediate_fp8 must use float8_e4m3fn, " f"got {intermediate_fp8.dtype}"
        )
    if intermediate_scale.dtype != torch.uint8:
        raise ValueError(
            f"intermediate_scale must use torch.uint8, "
            f"got {intermediate_scale.dtype}"
        )
    if w2_weight.dtype != torch.uint8:
        raise ValueError(f"w2_weight must use torch.uint8, got {w2_weight.dtype}")
    if w2_scale.dtype != torch.uint8:
        raise ValueError(f"w2_scale must use torch.uint8, got {w2_scale.dtype}")
    if not intermediate_fp8.is_cuda:
        raise ValueError("intermediate_fp8 must be a CUDA tensor")

    num_tokens, top_k = topk_ids.shape
    num_pairs = num_tokens * top_k
    if intermediate_fp8.shape[0] != num_pairs:
        raise ValueError(
            f"intermediate_fp8 first dim {intermediate_fp8.shape[0]} must "
            f"equal num_tokens*top_k ({num_pairs})"
        )
    intermediate = intermediate_fp8.shape[1]
    num_local_experts, hidden, weight_packed_k = w2_weight.shape
    if weight_packed_k * 2 != intermediate:
        raise ValueError(
            f"w2_weight packed_k {weight_packed_k} must equal intermediate/2"
            f" ({intermediate // 2})"
        )
    if hidden % 16 != 0:
        raise ValueError(f"hidden must be a multiple of 16, got {hidden}")
    if intermediate % 32 != 0:
        raise ValueError(f"intermediate must be a multiple of 32, got {intermediate}")
    if intermediate_scale.shape != (num_pairs, intermediate // 32):
        raise ValueError(
            "intermediate_scale must have shape "
            f"({num_pairs}, {intermediate // 32}), got "
            f"{tuple(intermediate_scale.shape)}"
        )
    if w2_scale.shape != (num_local_experts, hidden, intermediate // 32):
        raise ValueError(
            f"w2_scale must have shape ({num_local_experts}, {hidden}, "
            f"{intermediate // 32}), got {tuple(w2_scale.shape)}"
        )

    out_shape = (num_tokens, top_k, hidden)
    if out is None:
        out = torch.empty(
            out_shape, dtype=torch.float32, device=intermediate_fp8.device
        )
    else:
        if out.shape != out_shape:
            raise ValueError(f"out must have shape {out_shape}, got {tuple(out.shape)}")
        if out.dtype != torch.float32:
            raise ValueError(f"out must use torch.float32, got {out.dtype}")
        if out.device != intermediate_fp8.device:
            raise ValueError("out must be on the intermediate_fp8 device")
        if not out.is_contiguous():
            raise ValueError("out must be contiguous")

    if num_pairs == 0 or hidden == 0:
        return out

    mod = _load_sm12x_mxfp4_moe_module()
    mod.sm12x_mxfp4_moe_w2_tensorcore(
        out,
        intermediate_fp8.contiguous(),
        intermediate_scale.contiguous(),
        topk_ids.to(torch.int32).contiguous(),
        w2_weight.contiguous(),
        w2_scale.contiguous(),
        int(ep_rank),
    )
    return out


def sm12x_mxfp4_moe_weighted_reduce(
    per_pair: torch.Tensor,
    topk_weights: torch.Tensor,
    *,
    out: torch.Tensor | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Reduce ``[num_tokens, top_k, hidden]`` per-pair outputs with topk_weights.

    Computes ``out[t, c] = sum_k topk_weights[t, k] * per_pair[t, k, c]`` and
    casts to ``dtype`` (or ``out.dtype`` if ``out`` is provided). The output
    is the final per-token hidden_dim tensor matching the model dtype.
    """
    if per_pair.dim() != 3:
        raise ValueError(f"per_pair must be 3-D, got {per_pair.dim()}")
    if topk_weights.dim() != 2:
        raise ValueError(f"topk_weights must be 2-D, got {topk_weights.dim()}")
    if per_pair.dtype != torch.float32:
        raise ValueError(f"per_pair must use torch.float32, got {per_pair.dtype}")
    if topk_weights.dtype != torch.float32:
        raise ValueError(
            f"topk_weights must use torch.float32, got {topk_weights.dtype}"
        )
    if not per_pair.is_cuda:
        raise ValueError("per_pair must be a CUDA tensor")

    num_tokens, top_k, hidden = per_pair.shape
    if topk_weights.shape != (num_tokens, top_k):
        raise ValueError(
            f"topk_weights must have shape ({num_tokens}, {top_k}), "
            f"got {tuple(topk_weights.shape)}"
        )

    target_dtype = dtype if out is None else out.dtype
    if target_dtype is None:
        target_dtype = torch.bfloat16
    if target_dtype not in {torch.float32, torch.float16, torch.bfloat16}:
        raise ValueError(
            f"output dtype must be float32, float16, or bfloat16, "
            f"got {target_dtype}"
        )

    if out is None:
        out = torch.empty(
            (num_tokens, hidden), dtype=target_dtype, device=per_pair.device
        )
    else:
        if out.shape != (num_tokens, hidden):
            raise ValueError(
                f"out must have shape ({num_tokens}, {hidden}), "
                f"got {tuple(out.shape)}"
            )
        if out.device != per_pair.device:
            raise ValueError("out must be on the per_pair device")
        if not out.is_contiguous():
            raise ValueError("out must be contiguous")

    if num_tokens == 0 or hidden == 0:
        return out

    mod = _load_sm12x_mxfp4_moe_module()
    mod.sm12x_mxfp4_moe_weighted_reduce(
        out, per_pair.contiguous(), topk_weights.contiguous()
    )
    return out


__all__ = [
    "sm12x_mxfp4_mxfp8_dense",
    "sm12x_mxfp4_mxfp8_mma_tile",
    "sm12x_mxfp8_mxfp4_dense",
    "sm12x_mxfp8_mxfp4_mma_tile",
    "sm12x_mxfp4_mxfp8_quantize",
    "sm12x_mxfp4_swiglu_mxfp8_quantize",
    "sm12x_mxfp4_moe_forward",
    "sm12x_mxfp4_moe_forward_scalar",
    "sm12x_mxfp4_moe_forward_warp",
    "sm12x_mxfp4_moe_w13_tensorcore",
    "sm12x_mxfp4_moe_w2_tensorcore",
    "sm12x_mxfp4_moe_weighted_reduce",
]
