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

from __future__ import annotations

import torch
import torch.nn.functional as F

MXFP4_BLOCK = 32


def _e2m1_values(nibbles: torch.Tensor) -> torch.Tensor:
    table = nibbles.new_tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32
    )
    magnitude = table[(nibbles & 0x7).long()]
    sign = torch.where((nibbles & 0x8) != 0, -1.0, 1.0)
    return magnitude * sign


def mxfp4_dequantize_packed(
    packed: torch.Tensor,
    scale: torch.Tensor,
    *,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Dequantize packed E2M1 MXFP4 values with UE8M0 block scales.

    `packed` stores two E2M1 nibbles per byte along the last dimension. `scale`
    stores one biased exponent per 32 dequantized values.
    """
    if packed.dtype is not torch.uint8:
        raise TypeError(f"packed MXFP4 weights must be uint8, got {packed.dtype}")
    if scale.dtype is not torch.uint8:
        raise TypeError(f"MXFP4 scales must be uint8, got {scale.dtype}")

    dequantized_last_dim = packed.shape[-1] * 2
    if dequantized_last_dim % MXFP4_BLOCK != 0:
        raise ValueError(
            f"MXFP4 dequantized last dim must be divisible by {MXFP4_BLOCK}, "
            f"got {dequantized_last_dim}"
        )
    expected_scale_last_dim = dequantized_last_dim // MXFP4_BLOCK
    if (
        scale.shape[:-1] != packed.shape[:-1]
        or scale.shape[-1] != expected_scale_last_dim
    ):
        raise ValueError(
            f"scale shape {tuple(scale.shape)} is incompatible with packed shape "
            f"{tuple(packed.shape)}"
        )

    lo = packed & 0x0F
    hi = (packed >> 4) & 0x0F
    nibbles = torch.empty(
        (*packed.shape[:-1], dequantized_last_dim),
        dtype=torch.uint8,
        device=packed.device,
    )
    nibbles[..., 0::2] = lo
    nibbles[..., 1::2] = hi

    values = _e2m1_values(nibbles)
    block_scale = torch.pow(2.0, scale.float() - 127.0).repeat_interleave(
        MXFP4_BLOCK, dim=-1
    )
    return (values * block_scale).to(dtype)


def mxfp8_dequantize(
    values: torch.Tensor,
    scale: torch.Tensor,
    *,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Dequantize block-scaled FP8 activations with UE8M0 scales."""
    if values.dtype not in {torch.float8_e4m3fn, torch.float8_e5m2}:
        raise TypeError(
            "block-scaled activations must be torch.float8_e4m3fn or "
            f"torch.float8_e5m2, got {values.dtype}"
        )
    if scale.dtype is not torch.uint8:
        raise TypeError(f"MXFP8 scales must be uint8, got {scale.dtype}")
    if values.shape[-1] % MXFP4_BLOCK != 0:
        raise ValueError(
            f"MXFP8 last dim must be divisible by {MXFP4_BLOCK}, "
            f"got {values.shape[-1]}"
        )
    expected_scale_last_dim = values.shape[-1] // MXFP4_BLOCK
    if (
        scale.shape[:-1] != values.shape[:-1]
        or scale.shape[-1] != expected_scale_last_dim
    ):
        raise ValueError(
            f"scale shape {tuple(scale.shape)} is incompatible with values shape "
            f"{tuple(values.shape)}"
        )

    block_scale = torch.pow(2.0, scale.float() - 127.0).repeat_interleave(
        MXFP4_BLOCK, dim=-1
    )
    return (values.float() * block_scale).to(dtype)


def mxfp8_mxfp4_dense_reference(
    activations: torch.Tensor,
    activation_scale: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    *,
    out_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Reference dense GEMM for the future SM120 FP8xMXFP4 tensor-core path.

    Shapes follow the TokenSpeed canonical orientation used by DeepSeek V4
    weights: ``activations=[M,K]``, ``weight=[N,K/2]``, and output ``[M,N]``.
    """
    if activations.dim() != 2:
        raise ValueError(f"activations must be 2-D, got {activations.dim()}")
    if weight.dim() != 2:
        raise ValueError(f"weight must be 2-D, got {weight.dim()}")

    a = mxfp8_dequantize(activations, activation_scale)
    b = mxfp4_dequantize_packed(weight, weight_scale)
    if a.shape[1] != b.shape[1]:
        raise ValueError(f"activations K={a.shape[1]} must match weight K={b.shape[1]}")
    return (a @ b.T).to(out_dtype)


def _swiglu(
    gate: torch.Tensor,
    up: torch.Tensor,
    *,
    alpha: float | None,
    limit: float | None,
    beta: float | None,
) -> torch.Tensor:
    if limit is not None:
        gate = torch.clamp(gate, max=limit)
        up = torch.clamp(up, min=-limit, max=limit)
    if alpha is None:
        activated = F.silu(gate)
    else:
        activated = gate * torch.sigmoid(alpha * gate)
    up_term = up if beta is None else up + beta
    return activated * up_term


def _local_expert_id(
    expert_id: int,
    *,
    num_local_experts: int,
    ep_rank: int,
    ep_size: int,
) -> int | None:
    if expert_id < 0:
        return None
    if ep_size <= 1:
        if expert_id >= num_local_experts:
            return None
        return expert_id

    local_start = ep_rank * num_local_experts
    local_end = local_start + num_local_experts
    if expert_id < local_start or expert_id >= local_end:
        return None
    return expert_id - local_start


def sm12x_mxfp4_moe_reference_forward(
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
) -> torch.Tensor:
    """Small-shape correctness oracle for the SM12x MXFP4 backend.

    This intentionally dequantizes only selected experts. It is not a production
    path for full DeepSeek V4 checkpoints.
    """
    if hidden_states.dim() != 2:
        raise ValueError(f"hidden_states must be rank-2, got {hidden_states.dim()}")
    if topk_ids.shape != topk_weights.shape:
        raise ValueError(
            f"topk_ids shape {tuple(topk_ids.shape)} must match topk_weights "
            f"shape {tuple(topk_weights.shape)}"
        )

    num_tokens = hidden_states.shape[0]
    num_local_experts = w13_weight.shape[0]
    output = torch.zeros_like(hidden_states, dtype=torch.float32)
    expert_cache: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    compute_hidden = hidden_states.float()

    for token_idx in range(num_tokens):
        for choice_idx in range(topk_ids.shape[1]):
            expert_id = int(topk_ids[token_idx, choice_idx])
            local_id = _local_expert_id(
                expert_id,
                num_local_experts=num_local_experts,
                ep_rank=ep_rank,
                ep_size=ep_size,
            )
            if local_id is None:
                continue

            weights = expert_cache.get(local_id)
            if weights is None:
                weights = (
                    mxfp4_dequantize_packed(w13_weight[local_id], w13_scale[local_id]),
                    mxfp4_dequantize_packed(w2_weight[local_id], w2_scale[local_id]),
                )
                expert_cache[local_id] = weights
            w13, w2 = weights
            gate_up = F.linear(
                compute_hidden[token_idx],
                w13,
                None if w13_bias is None else w13_bias[local_id].float(),
            )
            gate, up = gate_up.chunk(2, dim=-1)
            if activation == "swiglu":
                intermediate = _swiglu(
                    gate,
                    up,
                    alpha=swiglu_alpha,
                    limit=swiglu_limit,
                    beta=swiglu_beta,
                )
            elif activation == "silu":
                intermediate = F.silu(gate) * up
            else:
                raise ValueError(f"Unsupported SM12x MXFP4 activation: {activation}")

            expert_out = F.linear(
                intermediate,
                w2,
                None if w2_bias is None else w2_bias[local_id].float(),
            )
            output[token_idx] += (
                expert_out * topk_weights[token_idx, choice_idx].float()
            )

    return output.to(hidden_states.dtype)


__all__ = [
    "mxfp4_dequantize_packed",
    "mxfp8_dequantize",
    "mxfp8_mxfp4_dense_reference",
    "sm12x_mxfp4_moe_reference_forward",
]
