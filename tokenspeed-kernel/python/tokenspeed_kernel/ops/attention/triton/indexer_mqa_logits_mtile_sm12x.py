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

"""M-tiled DeepSeek-V4 indexer MQA-logits (Triton, sm12x).

The per-query scorer (``indexer_mqa_logits_sm12x.py``) launches one program
per (query row, KV tile): the packed K tile is re-fetched (and re-decoded)
``num_q`` times per layer and the Q row ``num_kv/BLOCK_N`` times — ~7 GB of
L2 traffic per prefill layer at 17K rows. Per-launch nsys forensics (07-10)
shows the vLLM-fork scorer runs the same dot structure M-TILED (grid
``(rows/16, kv/BLOCK_N)``) at ~2x the per-cell rate; a bf16-dequantized-K
prepass experiment confirmed the decode ALU is NOT the bottleneck — traffic
is. This kernel tiles ``BLOCK_M`` query rows per program, so the K tile is
fetched and decoded ONCE per program and amortized across the row block.

Contract matches ``indexer_mqa_logits_sm12x_triton`` (compact 0-based
``[num_q, max_seqlen_k]`` fp32 logits, -inf outside each row's window) with
one structural difference: the launcher pre-fills the output with -inf and
the kernel writes only in-window cells (global-column tiles, per-row masks,
compact scatter store at ``col = global - ks[row]``). Windows may differ
per row (``ks`` is per-request-constant in prefill), handled by masking.
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
def _indexer_mqa_logits_mtile_kernel(
    q_values_ptr,  # int8  [num_q, H, VALUE_BYTES]
    q_scales_ptr,  # int32 [num_q, H]
    k_values_ptr,  # int8  [num_kv, VALUE_BYTES]
    k_scales_ptr,  # int32 [num_kv]
    weights_ptr,  # fp32  [num_q, H]
    ks_ptr,  # int32 [num_q]   window start (global kv index)
    ke_ptr,  # int32 [num_q]   window end (exclusive)
    out_ptr,  # fp32  [num_q, max_seqlen_k], PRE-FILLED with -inf
    num_q,
    num_kv,
    max_seqlen_k,
    stride_qv_m,
    stride_qv_h,
    stride_qs_m,
    stride_kv_n,
    stride_w_m,
    stride_o_m,
    H: tl.constexpr,
    VALUE_BYTES: tl.constexpr,
    SCALE_BLOCKS: tl.constexpr,
    BYTES_PER_BLOCK: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    INPUT_PRECISION: tl.constexpr,
):
    mb = tl.program_id(0)
    nb = tl.program_id(1)

    offs_m = mb * BLOCK_M + tl.arange(0, BLOCK_M)
    valid_m = offs_m < num_q
    # other= values chosen so min/max over padding rows are inert.
    ks = tl.load(ks_ptr + offs_m, mask=valid_m, other=2147483647)
    ke = tl.load(ke_ptr + offs_m, mask=valid_m, other=0)
    ks = tl.maximum(ks, 0)
    ke = tl.minimum(ke, num_kv)

    g = nb * BLOCK_N + tl.arange(0, BLOCK_N)  # global kv columns
    # Whole tile outside every row's window: nothing to write (out is -inf).
    if nb * BLOCK_N >= tl.max(ke):
        return
    if (nb + 1) * BLOCK_N <= tl.min(ks):
        return
    valid_n = g < num_kv
    row_mask = (g[None, :] >= ks[:, None]) & (g[None, :] < ke[:, None])

    db = tl.arange(0, VALUE_BYTES)
    blk = db // BYTES_PER_BLOCK

    # ---- k tile: decoded ONCE per program, amortized over BLOCK_M rows ----
    k_b = tl.load(
        k_values_ptr + g[None, :].to(tl.int64) * stride_kv_n + db[:, None],
        mask=valid_n[None, :],
        other=0,
    ).to(tl.int32) & 0xFF
    k_lo = _e2m1_decode(k_b & 0xF)
    k_hi = _e2m1_decode((k_b >> 4) & 0xF)
    k_sc = tl.load(k_scales_ptr + g, mask=valid_n, other=0)
    k_scale = tl.zeros((VALUE_BYTES, BLOCK_N), dtype=tl.float32)
    for b in tl.static_range(SCALE_BLOCKS):
        sval = _block_scale(k_sc, b)
        k_scale = tl.where(blk[:, None] == b, sval[None, :], k_scale)
    k_lo = k_lo * k_scale
    k_hi = k_hi * k_scale

    m_off = offs_m.to(tl.int64)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for h in tl.range(0, H):
        q_b = tl.load(
            q_values_ptr + m_off[:, None] * stride_qv_m + h * stride_qv_h + db[None, :],
            mask=valid_m[:, None],
            other=0,
        ).to(tl.int32) & 0xFF
        q_lo = _e2m1_decode(q_b & 0xF)
        q_hi = _e2m1_decode((q_b >> 4) & 0xF)
        q_sc = tl.load(q_scales_ptr + offs_m * stride_qs_m + h, mask=valid_m, other=0)
        q_scale = tl.zeros((BLOCK_M, VALUE_BYTES), dtype=tl.float32)
        for qb in tl.static_range(SCALE_BLOCKS):
            q_sval = _block_scale(q_sc, qb)
            q_scale = tl.where(blk[None, :] == qb, q_sval[:, None], q_scale)
        q_lo = q_lo * q_scale
        q_hi = q_hi * q_scale

        s = tl.dot(q_lo, k_lo, input_precision=INPUT_PRECISION)
        s += tl.dot(q_hi, k_hi, input_precision=INPUT_PRECISION)
        w_h = tl.load(weights_ptr + offs_m * stride_w_m + h, mask=valid_m, other=0.0)
        acc += tl.maximum(s, 0.0) * w_h[:, None].to(tl.float32)

    c = g[None, :] - ks[:, None]  # compact output column
    st_mask = row_mask & valid_m[:, None] & (c < max_seqlen_k)
    tl.store(out_ptr + m_off[:, None] * stride_o_m + c, acc, mask=st_mask)


def indexer_mqa_logits_sm12x_mtile_triton(
    q_values: torch.Tensor,
    q_scales: torch.Tensor,
    k_values: torch.Tensor,
    k_scales: torch.Tensor,
    weights: torch.Tensor,
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
    max_seqlen_k: int,
    head_dim: int = 128,
    block_m: int = 32,
    block_n: int = 128,
    input_precision: str = "ieee",
) -> torch.Tensor:
    """M-tiled drop-in for ``indexer_mqa_logits_sm12x_triton`` (same contract)."""
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

    grid = (triton.cdiv(num_q, block_m), triton.cdiv(num_kv, block_n))
    _indexer_mqa_logits_mtile_kernel[grid](
        q_values,
        q_scales,
        k_values,
        k_scales,
        weights,
        cu_seqlen_ks,
        cu_seqlen_ke,
        out,
        num_q,
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
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        INPUT_PRECISION=input_precision,
        num_warps=4,
    )
    return out
