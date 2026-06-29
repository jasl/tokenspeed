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

"""Consumer-Blackwell (sm_120/sm_121) DeepSeek-V4 indexer MQA-logits (Triton).

A tiled, fused Triton implementation of the DeepSeek-V4 lightning indexer
``fp8_fp4_mqa_logits`` (the non-paged / prefill scoring kernel), which has no
consumer-Blackwell binary in deep_gemm. Unlike the torch reference in
``ops/attention/indexer_mqa_logits_sm12x.py`` (which materializes the full
``[num_q, num_heads, num_kv]`` score tensor and OOMs on long context), this
kernel tiles over the KV axis and never materializes more than one
``[num_heads, BLOCK_N]`` score tile per program.

Lightning-indexer logits (deep_gemm ``ref_fp8_mqa_logits``):
    logits[m, n] = sum_h ReLU(<q[m, h, :], k[n, :]>) * weights[m, h]
with a per-query window ``[cu_seqlen_ks[m], cu_seqlen_ke[m])``; the output is the
COMPACT, 0-based layout the kernel produces: ``logits[m, 0:L]`` holds the scores
over ``k[cu_seqlen_ks[m] : cu_seqlen_ks[m]+L]`` (L = ke - ks); columns >= L are
``-inf``. Feeds ``_deepseek_v4_indexer_topk_from_logits``.

MXFP4 layout (matches ``dequant_indexer_mxfp4`` / ``DEEPSEEK_V4_MXFP4_BLOCK_SIZE``):
two e2m1 nibbles per byte (low nibble = even dim index, high = odd), int32-packed
e8m0 block scales (4 bytes little-endian, one per 32-element block). The dot is
split over the low/high nibbles of each byte -- both nibbles of byte ``b`` share
the scale block ``b // (32 // 2) = b // 16`` -- so the head_dim contraction needs
no nibble interleave.
"""

from __future__ import annotations

import torch

from tokenspeed_kernel._triton import tl, triton

# e2m1 magnitude lookup (sign is bit 3). value = LUT[code & 7] * (1 - 2*(code>>3)).
# LUT = [0, 0.5, 1, 1.5, 2, 3, 4, 6] reproduced arithmetically below.
_MXFP4_BLOCK = 32
_INDEXER_HEAD_DIM = 128
_INDEXER_NUM_HEADS = 64


@triton.jit
def _e2m1_decode(code):
    """Decode a 4-bit e2m1 code (0..15) tensor to float32 (LUT-exact)."""
    sign = (code >> 3) & 1
    exp = (code >> 1) & 3
    man = code & 1
    exp_f = exp.to(tl.float32)
    man_f = man.to(tl.float32)
    # exp == 0 -> subnormal (man * 0.5); else 2^(exp-1) * (1 + man*0.5).
    mag = tl.where(exp == 0, man_f * 0.5, tl.exp2(exp_f - 1.0) * (1.0 + man_f * 0.5))
    return mag * (1.0 - 2.0 * sign.to(tl.float32))


@triton.jit
def _block_scale(scale_i32, block_idx):
    """e8m0 block scale: byte = (scale_i32 >> 8*block_idx) & 0xFF -> 2^(byte-127)."""
    byte = (scale_i32 >> (block_idx * 8)) & 0xFF
    return tl.exp2(byte.to(tl.float32) - 127.0)


@triton.jit
def _indexer_mqa_logits_kernel(
    q_values_ptr,  # int8  [num_q, H, VALUE_BYTES]
    q_scales_ptr,  # int32 [num_q, H]
    k_values_ptr,  # int8  [num_kv, VALUE_BYTES]
    k_scales_ptr,  # int32 [num_kv]
    weights_ptr,  # fp32  [num_q, H]
    ks_ptr,  # int32 [num_q]   per-query window start into kv
    ke_ptr,  # int32 [num_q]   per-query window end (exclusive)
    out_ptr,  # fp32  [num_q, max_seqlen_k]
    num_kv,
    max_seqlen_k,
    stride_qv_m,
    stride_qv_h,
    stride_qs_m,
    stride_kv_n,
    stride_w_m,
    stride_o_m,
    H: tl.constexpr,
    VALUE_BYTES: tl.constexpr,  # head_dim // 2
    SCALE_BLOCKS: tl.constexpr,  # head_dim // 32
    BYTES_PER_BLOCK: tl.constexpr,  # 32 // 2 = 16
    BLOCK_N: tl.constexpr,
    INPUT_PRECISION: tl.constexpr,
):
    m = tl.program_id(0)
    nb = tl.program_id(1)
    m_off = m.to(tl.int64)  # 64-bit row base: guards int32 offset overflow at long context

    # Clamp the window to [0, num_kv] to match the torch reference contract
    # (defensive: the prefill planner already keeps ks>=0 and ke<=num_kv).
    ks = tl.maximum(tl.load(ks_ptr + m), 0)
    ke = tl.minimum(tl.load(ke_ptr + m), num_kv)
    row_len = tl.maximum(ke - ks, 0)

    n_rel = nb * BLOCK_N + tl.arange(0, BLOCK_N)  # 0-based output columns
    valid_n = (n_rel < row_len) & (n_rel < max_seqlen_k)
    k_idx = ks + n_rel  # actual kv row in [0, num_kv), valid only where valid_n

    hd = tl.arange(0, H)  # [H]
    db = tl.arange(0, VALUE_BYTES)  # byte index along head_dim
    blk = db // BYTES_PER_BLOCK  # scale block for each byte -> [VALUE_BYTES]

    # ---- dequant q[m]: [H, VALUE_BYTES] low/high nibbles ----
    q_ptr = q_values_ptr + m_off * stride_qv_m + hd[:, None] * stride_qv_h + db[None, :]
    q_b = tl.load(q_ptr).to(tl.int32) & 0xFF  # [H, VALUE_BYTES]
    q_lo = _e2m1_decode(q_b & 0xF)
    q_hi = _e2m1_decode((q_b >> 4) & 0xF)

    q_sc = tl.load(q_scales_ptr + m * stride_qs_m + hd)  # [H] int32
    q_scale = tl.zeros((H, VALUE_BYTES), dtype=tl.float32)
    for b in tl.static_range(SCALE_BLOCKS):
        sval = _block_scale(q_sc, b)  # [H]
        q_scale = tl.where(blk[None, :] == b, sval[:, None], q_scale)
    q_lo = q_lo * q_scale
    q_hi = q_hi * q_scale

    # ---- dequant k[k_idx]: transposed [VALUE_BYTES, BLOCK_N] ----
    k_ptr = k_values_ptr + k_idx[None, :].to(tl.int64) * stride_kv_n + db[:, None]
    k_b = tl.load(k_ptr, mask=valid_n[None, :], other=0).to(tl.int32) & 0xFF
    k_lo = _e2m1_decode(k_b & 0xF)
    k_hi = _e2m1_decode((k_b >> 4) & 0xF)

    k_sc = tl.load(k_scales_ptr + k_idx, mask=valid_n, other=0)  # [BLOCK_N] int32
    k_scale = tl.zeros((VALUE_BYTES, BLOCK_N), dtype=tl.float32)
    for b in tl.static_range(SCALE_BLOCKS):
        sval = _block_scale(k_sc, b)  # [BLOCK_N]
        k_scale = tl.where(blk[:, None] == b, sval[None, :], k_scale)
    k_lo = k_lo * k_scale
    k_hi = k_hi * k_scale

    # ---- scores[h, n] = <q[m,h], k[n]> = dot(q_lo, k_lo) + dot(q_hi, k_hi) ----
    scores = tl.dot(q_lo, k_lo, input_precision=INPUT_PRECISION)
    scores += tl.dot(q_hi, k_hi, input_precision=INPUT_PRECISION)  # [H, BLOCK_N]

    # ---- logits[n] = sum_h ReLU(scores[h, n]) * weights[m, h] ----
    w = tl.load(weights_ptr + m * stride_w_m + hd).to(tl.float32)  # [H]
    logits = tl.sum(tl.maximum(scores, 0.0) * w[:, None], axis=0)  # [BLOCK_N]

    out_val = tl.where(n_rel < row_len, logits, float("-inf"))
    tl.store(out_ptr + m_off * stride_o_m + n_rel, out_val, mask=n_rel < max_seqlen_k)


def indexer_mqa_logits_sm12x_triton(
    q_values: torch.Tensor,
    q_scales: torch.Tensor,
    k_values: torch.Tensor,
    k_scales: torch.Tensor,
    weights: torch.Tensor,
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
    max_seqlen_k: int,
    head_dim: int = 128,
    block_n: int = 128,
    input_precision: str = "ieee",
) -> torch.Tensor:
    """DeepSeek-V4 indexer lightning-MQA logits (Triton), drop-in for the torch ref.

    Args:
        q_values: Packed fp4 query, ``[num_q, num_heads, head_dim // 2]`` int8.
        q_scales: int32-packed e8m0 q scales, ``[num_q, num_heads]``.
        k_values: Packed fp4 gathered KV, ``[num_kv, head_dim // 2]`` int8.
        k_scales: int32-packed e8m0 KV scales, ``[num_kv]``.
        weights: Per-head indexer weights, ``[num_q, num_heads]`` (cast to fp32).
        cu_seqlen_ks: Per-query KV window start into the gathered KV, ``[num_q]``.
        cu_seqlen_ke: Per-query KV window end (exclusive), ``[num_q]``.
        max_seqlen_k: Width of the output logits (max window length).
        head_dim: Indexer per-head dim (128 for DeepSeek-V4).
        block_n: KV tile width (program granularity along the KV axis).
        input_precision: ``tl.dot`` precision -- ``"ieee"`` (default, matches the
            fp32 torch reference) or ``"tf32"`` (faster, ~1e-2 relative).

    Returns:
        Compact logits ``[num_q, max_seqlen_k]`` fp32: ``out[m, 0:L]`` are the
        scores over ``k[cu_seqlen_ks[m] : cu_seqlen_ke[m]]`` (L = ke - ks);
        columns >= L are -inf.
    """
    num_q, num_heads, value_bytes = q_values.shape
    device = q_values.device
    out = torch.full(
        (num_q, max_seqlen_k), float("-inf"), dtype=torch.float32, device=device
    )
    num_kv = k_values.shape[0]
    if num_q == 0 or num_kv == 0 or max_seqlen_k <= 0:
        return out

    if value_bytes != head_dim // 2:
        raise ValueError(
            f"q_values last dim {value_bytes} != head_dim//2 {head_dim // 2}"
        )
    if head_dim % _MXFP4_BLOCK != 0:
        raise ValueError(f"head_dim {head_dim} must be a multiple of {_MXFP4_BLOCK}")

    q_values = q_values.contiguous()
    q_scales = q_scales.contiguous()
    k_values = k_values.contiguous()
    k_scales = k_scales.contiguous()
    weights = weights.contiguous()
    cu_seqlen_ks = cu_seqlen_ks.contiguous()
    cu_seqlen_ke = cu_seqlen_ke.contiguous()

    grid = (num_q, triton.cdiv(max_seqlen_k, block_n))
    _indexer_mqa_logits_kernel[grid](
        q_values,
        q_scales,
        k_values,
        k_scales,
        weights,
        cu_seqlen_ks,
        cu_seqlen_ke,
        out,
        num_kv,
        max_seqlen_k,
        q_values.stride(0),
        q_values.stride(1),
        q_scales.stride(0),
        k_values.stride(0),
        weights.stride(0),
        out.stride(0),
        H=num_heads,
        VALUE_BYTES=value_bytes,
        SCALE_BLOCKS=head_dim // _MXFP4_BLOCK,
        BYTES_PER_BLOCK=_MXFP4_BLOCK // 2,
        BLOCK_N=block_n,
        INPUT_PRECISION=input_precision,
    )
    return out
