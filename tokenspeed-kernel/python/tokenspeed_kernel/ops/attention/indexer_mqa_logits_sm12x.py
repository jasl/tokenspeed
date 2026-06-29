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

"""Consumer-Blackwell (sm_120/sm_121) DeepSeek-V4 indexer MQA-logits (torch).

deep_gemm's ``fp8_fp4_mqa_logits`` (the non-paged / prefill indexer scoring
kernel) has no consumer-Blackwell binary (cudaErrorNoKernelImageForDevice). This
is a portable torch fallback with identical semantics, used to unblock + validate
the DeepSeek-V4 sparse indexer prefill scoring on sm12x and to serve as the
numerical reference for the fused Triton kernel.

Lightning-indexer logits (deep_gemm ``ref_fp8_mqa_logits``):
    logits[m, n] = sum_h ReLU(<q[m, h, :], k[n, :]>) * weights[m, h]
with a per-query causal window [cu_start[m], cu_end[m]); the output is the
COMPACT, 0-based layout the kernel produces: ``logits[m, 0:L]`` holds the scores
over ``k[cu_start[m] : cu_start[m]+L]`` (L = cu_end[m] - cu_start[m]).

MXFP4 dequant (e2m1 values + e8m0 block-32 scales, ``DEEPSEEK_V4_MXFP4_BLOCK_SIZE``):
two e2m1 nibbles per byte (lo = even index), int32-packed e8m0 scales (4 bytes,
little-endian; one per 32-element block).
"""

from __future__ import annotations

import torch

# e2m1 magnitude lookup for the 3-bit code (sign is bit 3).
_E2M1_MAG = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
_MXFP4_BLOCK = 32


def _e2m1_lut(device: torch.device) -> torch.Tensor:
    return torch.tensor(_E2M1_MAG, dtype=torch.float32, device=device)


def dequant_indexer_mxfp4(
    values_int8: torch.Tensor,
    scales_int32: torch.Tensor,
    head_dim: int,
) -> torch.Tensor:
    """Dequantize packed MXFP4 indexer values to fp32.

    Args:
        values_int8: Packed e2m1 nibbles, ``[..., head_dim // 2]`` (int8 view of
            uint8; two values per byte, low nibble = even index).
        scales_int32: int32-packed e8m0 block scales, ``[...]`` with the same
            leading dims; each int32 holds ``head_dim // 32`` e8m0 bytes
            (little-endian), one per 32-element block.
        head_dim: Per-head vector length (128 for DeepSeek-V4 indexer).

    Returns:
        Dequantized values ``[..., head_dim]`` fp32.
    """
    device = values_int8.device
    lut = _e2m1_lut(device)
    v = values_int8.to(torch.int32) & 0xFF  # treat as uint8
    lo = v & 0xF
    hi = (v >> 4) & 0xF
    codes = torch.stack((lo, hi), dim=-1).reshape(*values_int8.shape[:-1], head_dim)
    sign = (codes >> 3) & 1
    mag = lut[(codes & 0x7).long()]
    vals = mag * (1.0 - 2.0 * sign.to(torch.float32))  # [..., head_dim]

    n_blocks = head_dim // _MXFP4_BLOCK
    s = scales_int32.to(torch.int64) & 0xFFFFFFFF
    shifts = torch.arange(n_blocks, device=device, dtype=torch.int64) * 8
    sbytes = (s.unsqueeze(-1) >> shifts) & 0xFF  # [..., n_blocks]
    scale = torch.exp2(sbytes.to(torch.float32) - 127.0)  # [..., n_blocks]
    scale_full = scale.repeat_interleave(_MXFP4_BLOCK, dim=-1)  # [..., head_dim]
    return vals * scale_full


def indexer_mqa_logits_sm12x(
    q_values: torch.Tensor,
    q_scales: torch.Tensor,
    k_values: torch.Tensor,
    k_scales: torch.Tensor,
    weights: torch.Tensor,
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
    max_seqlen_k: int,
    head_dim: int = 128,
) -> torch.Tensor:
    """DeepSeek-V4 indexer lightning-MQA logits, drop-in for fp8_fp4_mqa_logits.

    Args:
        q_values: Packed fp4 query, ``[num_q, num_heads, head_dim // 2]`` int8.
        q_scales: int32-packed e8m0 q scales, ``[num_q, num_heads]``.
        k_values: Packed fp4 gathered KV, ``[num_kv, head_dim // 2]`` int8.
        k_scales: int32-packed e8m0 KV scales, ``[num_kv]``.
        weights: Per-head indexer weights, ``[num_q, num_heads]`` (fp32).
        cu_seqlen_ks: Per-query KV window start into the gathered KV, ``[num_q]``.
        cu_seqlen_ke: Per-query KV window end (exclusive), ``[num_q]``.
        max_seqlen_k: Width of the output logits (max window length).
        head_dim: Indexer per-head dim (128 for DeepSeek-V4).

    Returns:
        Compact logits ``[num_q, max_seqlen_k]`` fp32: ``out[m, 0:L]`` are the
        scores over ``k[cu_seqlen_ks[m] : cu_seqlen_ke[m]]`` (L = ke - ks);
        columns >= L are -inf.
    """
    num_q = q_values.shape[0]
    device = q_values.device
    out = torch.full(
        (num_q, max_seqlen_k), float("-inf"), dtype=torch.float32, device=device
    )
    num_kv = k_values.shape[0]
    if num_q == 0 or num_kv == 0 or max_seqlen_k <= 0:
        return out

    q = dequant_indexer_mxfp4(q_values, q_scales, head_dim)  # [num_q, H, D]
    k = dequant_indexer_mxfp4(k_values, k_scales, head_dim)  # [num_kv, D]
    w = weights.to(torch.float32)  # [num_q, H]

    # logits[m, n] = sum_h relu(<q[m,h], k[n]>) * w[m,h]
    score = torch.einsum("mhd,nd->mhn", q, k)  # [num_q, H, num_kv]
    full = (torch.relu(score) * w.unsqueeze(-1)).sum(dim=1)  # [num_q, num_kv]

    ks = cu_seqlen_ks.to(torch.int64)
    ke = cu_seqlen_ke.to(torch.int64)
    for m in range(num_q):
        s = int(ks[m].clamp_(0, num_kv))
        e = int(ke[m].clamp_(0, num_kv))
        length = min(e - s, max_seqlen_k)
        if length > 0:
            out[m, :length] = full[m, s : s + length]
    return out
