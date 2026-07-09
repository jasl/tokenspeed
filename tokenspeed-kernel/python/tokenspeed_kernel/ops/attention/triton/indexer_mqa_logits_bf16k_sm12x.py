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

"""DeepSeek-V4 indexer MQA-logits with a dequantized-K prepass (Triton, sm12x).

The per-query scoring kernel in ``indexer_mqa_logits_sm12x.py`` re-runs the
MXFP4 e2m1 decode ladder for every KV tile it touches — the gathered K is
decoded once per (query, kv-tile) pair, ~``num_q`` times per row on long
prefill (nsys 07-10: the K-side ALU makes the kernel ~2x slower per scored
cell than the vLLM-fork scorer despite an identical dot structure).

This variant splits the work:

1. ``_indexer_k_dequant_bf16_kernel``: decode the gathered MXFP4 K ONCE into
   a bf16 workspace (``[num_kv, head_dim]``, ~2 MB at 7.6K compressed keys) —
   EXACT: every e2m1 magnitude times an e8m0 block scale is representable in
   bf16 (<= 3 mantissa bits, power-of-two scales).
2. ``_indexer_mqa_logits_bf16k_kernel``: the same per-query compact scoring
   kernel, with the K nibble ladders replaced by two strided bf16 loads
   (even/odd dims), preserving the lo/hi nibble-plane dot structure so the Q
   path is untouched.

Output contract is identical to ``indexer_mqa_logits_sm12x_triton`` (compact
0-based ``[num_q, max_seqlen_k]`` fp32, -inf past each row's window).
"""

from __future__ import annotations

import torch

from tokenspeed_kernel._triton import tl, triton

_MXFP4_BLOCK = 32
_INDEXER_HEAD_DIM = 128


@triton.jit
def _e2m1_decode(code):
    """Decode a 4-bit e2m1 code (0..15) tensor to float32 (LUT-exact)."""
    sign = (code >> 3) & 1
    exp = (code >> 1) & 3
    man = code & 1
    exp_f = exp.to(tl.float32)
    man_f = man.to(tl.float32)
    mag = tl.where(exp == 0, man_f * 0.5, tl.exp2(exp_f - 1.0) * (1.0 + man_f * 0.5))
    return mag * (1.0 - 2.0 * sign.to(tl.float32))


@triton.jit
def _block_scale(scale_i32, block_idx):
    """e8m0 block scale: byte = (scale_i32 >> 8*block_idx) & 0xFF -> 2^(byte-127)."""
    byte = (scale_i32 >> (block_idx * 8)) & 0xFF
    return tl.exp2(byte.to(tl.float32) - 127.0)


@triton.jit
def _indexer_k_dequant_bf16_kernel(
    k_values_ptr,  # int8  [num_kv, VALUE_BYTES]
    k_scales_ptr,  # int32 [num_kv]
    out_ptr,  # bf16 [num_kv, D] with D = 2 * VALUE_BYTES
    num_kv,
    stride_kv_n,
    stride_o_n,
    VALUE_BYTES: tl.constexpr,
    SCALE_BLOCKS: tl.constexpr,
    BYTES_PER_BLOCK: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    nb = tl.program_id(0)
    n = nb * BLOCK_N + tl.arange(0, BLOCK_N)
    valid = n < num_kv
    db = tl.arange(0, VALUE_BYTES)
    blk = db // BYTES_PER_BLOCK

    b = tl.load(
        k_values_ptr + n[:, None].to(tl.int64) * stride_kv_n + db[None, :],
        mask=valid[:, None],
        other=0,
    ).to(tl.int32) & 0xFF
    lo = _e2m1_decode(b & 0xF)
    hi = _e2m1_decode((b >> 4) & 0xF)

    sc = tl.load(k_scales_ptr + n, mask=valid, other=0)
    scale = tl.zeros((BLOCK_N, VALUE_BYTES), dtype=tl.float32)
    for i in tl.static_range(SCALE_BLOCKS):
        sval = _block_scale(sc, i)
        scale = tl.where(blk[None, :] == i, sval[:, None], scale)
    lo = lo * scale
    hi = hi * scale

    # Packed layout: low nibble = even dim, high nibble = odd dim.
    out_base = out_ptr + n[:, None].to(tl.int64) * stride_o_n
    tl.store(out_base + db[None, :] * 2, lo.to(tl.bfloat16), mask=valid[:, None])
    tl.store(out_base + db[None, :] * 2 + 1, hi.to(tl.bfloat16), mask=valid[:, None])


@triton.jit
def _indexer_mqa_logits_bf16k_kernel(
    q_values_ptr,  # int8  [num_q, H, VALUE_BYTES]
    q_scales_ptr,  # int32 [num_q, H]
    k_bf16_ptr,  # bf16 [num_kv, D]
    weights_ptr,  # fp32  [num_q, H]
    ks_ptr,  # int32 [num_q]
    ke_ptr,  # int32 [num_q]
    out_ptr,  # fp32  [num_q, max_seqlen_k]
    num_kv,
    max_seqlen_k,
    stride_qv_m,
    stride_qv_h,
    stride_qs_m,
    stride_k_n,
    stride_w_m,
    stride_o_m,
    H: tl.constexpr,
    VALUE_BYTES: tl.constexpr,
    SCALE_BLOCKS: tl.constexpr,
    BYTES_PER_BLOCK: tl.constexpr,
    BLOCK_N: tl.constexpr,
    INPUT_PRECISION: tl.constexpr,
):
    m = tl.program_id(0)
    nb = tl.program_id(1)
    m_off = m.to(tl.int64)

    ks = tl.maximum(tl.load(ks_ptr + m), 0)
    ke = tl.minimum(tl.load(ke_ptr + m), num_kv)
    row_len = tl.maximum(ke - ks, 0)

    n_rel = nb * BLOCK_N + tl.arange(0, BLOCK_N)
    if nb * BLOCK_N >= row_len:
        neg = tl.full((BLOCK_N,), float("-inf"), tl.float32)
        tl.store(out_ptr + m_off * stride_o_m + n_rel, neg, mask=n_rel < max_seqlen_k)
        return
    valid_n = (n_rel < row_len) & (n_rel < max_seqlen_k)
    k_idx = ks + n_rel

    hd = tl.arange(0, H)
    db = tl.arange(0, VALUE_BYTES)
    blk = db // BYTES_PER_BLOCK

    # ---- dequant q[m]: [H, VALUE_BYTES] low/high nibbles (unchanged) ----
    q_ptr = q_values_ptr + m_off * stride_qv_m + hd[:, None] * stride_qv_h + db[None, :]
    q_b = tl.load(q_ptr).to(tl.int32) & 0xFF
    q_lo = _e2m1_decode(q_b & 0xF)
    q_hi = _e2m1_decode((q_b >> 4) & 0xF)

    q_sc = tl.load(q_scales_ptr + m * stride_qs_m + hd)
    q_scale = tl.zeros((H, VALUE_BYTES), dtype=tl.float32)
    for b in tl.static_range(SCALE_BLOCKS):
        sval = _block_scale(q_sc, b)
        q_scale = tl.where(blk[None, :] == b, sval[:, None], q_scale)
    q_lo = q_lo * q_scale
    q_hi = q_hi * q_scale

    # ---- k[k_idx]: two strided bf16 loads (even/odd dims), no decode ----
    k_base = k_bf16_ptr + k_idx[None, :].to(tl.int64) * stride_k_n + db[:, None] * 2
    k_lo = tl.load(k_base, mask=valid_n[None, :], other=0.0).to(tl.float32)
    k_hi = tl.load(k_base + 1, mask=valid_n[None, :], other=0.0).to(tl.float32)

    scores = tl.dot(q_lo, k_lo, input_precision=INPUT_PRECISION)
    scores += tl.dot(q_hi, k_hi, input_precision=INPUT_PRECISION)

    w = tl.load(weights_ptr + m * stride_w_m + hd).to(tl.float32)
    logits = tl.sum(tl.maximum(scores, 0.0) * w[:, None], axis=0)

    out_val = tl.where(n_rel < row_len, logits, float("-inf"))
    tl.store(out_ptr + m_off * stride_o_m + n_rel, out_val, mask=n_rel < max_seqlen_k)


def indexer_mqa_logits_sm12x_bf16k_triton(
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
    """Drop-in for ``indexer_mqa_logits_sm12x_triton`` with a dequantized-K
    prepass — same arguments, same compact output contract, exact numerics
    (MXFP4 dequant is bf16-representable). ~2 MB bf16 workspace per call.
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

    scale_blocks = head_dim // _MXFP4_BLOCK
    bytes_per_block = _MXFP4_BLOCK // 2

    k_bf16 = torch.empty(num_kv, head_dim, dtype=torch.bfloat16, device=device)
    dequant_block = 64
    _indexer_k_dequant_bf16_kernel[(triton.cdiv(num_kv, dequant_block),)](
        k_values,
        k_scales,
        k_bf16,
        num_kv,
        k_values.stride(0),
        k_bf16.stride(0),
        VALUE_BYTES=value_bytes,
        SCALE_BLOCKS=scale_blocks,
        BYTES_PER_BLOCK=bytes_per_block,
        BLOCK_N=dequant_block,
        num_warps=4,
    )

    grid = (num_q, triton.cdiv(max_seqlen_k, block_n))
    _indexer_mqa_logits_bf16k_kernel[grid](
        q_values,
        q_scales,
        k_bf16,
        weights,
        cu_seqlen_ks,
        cu_seqlen_ke,
        out,
        num_kv,
        max_seqlen_k,
        q_values.stride(0),
        q_values.stride(1),
        q_scales.stride(0),
        k_bf16.stride(0),
        weights.stride(0),
        out.stride(0),
        H=num_heads,
        VALUE_BYTES=value_bytes,
        SCALE_BLOCKS=scale_blocks,
        BYTES_PER_BLOCK=bytes_per_block,
        BLOCK_N=block_n,
        INPUT_PRECISION=input_precision,
        num_warps=4,
    )
    return out
