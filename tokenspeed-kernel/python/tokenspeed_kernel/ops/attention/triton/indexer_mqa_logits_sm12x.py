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
    m_off = m.to(
        tl.int64
    )  # 64-bit row base: guards int32 offset overflow at long context

    # Clamp the window to [0, num_kv] to match the torch reference contract
    # (defensive: the prefill planner already keeps ks>=0 and ke<=num_kv).
    ks = tl.maximum(tl.load(ks_ptr + m), 0)
    ke = tl.minimum(tl.load(ke_ptr + m), num_kv)
    row_len = tl.maximum(ke - ks, 0)

    n_rel = nb * BLOCK_N + tl.arange(0, BLOCK_N)  # 0-based output columns
    # Whole tile is past this row's window: write -inf and skip the dequant+dot.
    # (Decode captures an over-sized grid, so most tiles are empty at short ctx.)
    if nb * BLOCK_N >= row_len:
        neg = tl.full((BLOCK_N,), float("-inf"), tl.float32)
        tl.store(out_ptr + m_off * stride_o_m + n_rel, neg, mask=n_rel < max_seqlen_k)
        return
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


# ============================ MISA-dagger fast path ============================
# Two-pass sparse indexer prefill (see project_ds4_indexer_subquadratic_research):
#   pass 1 (_hhead): score all N keys using only each query's top-H_SEL heads (by
#     |weight|) -> cheap candidate ranking. FLOP = H_SEL/H of the full scorer.
#     Output is the SAME compact [num_q, max_seqlen_k] layout as the full kernel,
#     so torch.topk(C) over it yields the C candidate columns per query.
#   pass 2 (_gather): re-score ONLY those C candidates with ALL H heads (exact) ->
#     [num_q, C]. FLOP = C/N of the full scorer -> amortizes at long context.
# Selecting the top-H_SEL heads is bit-exact vs zeroing the other heads' weights
# (logits = sum_h ReLU(dot)*w, and a zero-weight head contributes 0), so the fast
# path reproduces the MVP masked-weight path it replaces (the validated oracle).


@triton.jit
def _indexer_mqa_logits_hhead_kernel(
    q_values_ptr,  # int8  [num_q, H, VALUE_BYTES]
    q_scales_ptr,  # int32 [num_q, H]
    k_values_ptr,  # int8  [num_kv, VALUE_BYTES]
    k_scales_ptr,  # int32 [num_kv]
    weights_ptr,  # fp32  [num_q, H]
    head_idx_ptr,  # int32 [num_q, H_SEL]   per-query selected head indices (top-|w|)
    ks_ptr,  # int32 [num_q]
    ke_ptr,  # int32 [num_q]
    out_ptr,  # fp32  [num_q, max_seqlen_k]
    num_kv,
    max_seqlen_k,
    stride_qv_m,
    stride_qv_h,
    stride_qs_m,
    stride_kv_n,
    stride_w_m,
    stride_hi_m,
    stride_o_m,
    H_SEL: tl.constexpr,  # number of heads scored per query (<= H)
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

    # Gathered head indices for this query (top-|w| heads) -> [H_SEL].
    hsel = tl.arange(0, H_SEL)
    hd = tl.load(head_idx_ptr + m_off * stride_hi_m + hsel)  # [H_SEL] int32
    db = tl.arange(0, VALUE_BYTES)
    blk = db // BYTES_PER_BLOCK

    # ---- dequant q[m] for the selected heads: [H_SEL, VALUE_BYTES] ----
    q_ptr = q_values_ptr + m_off * stride_qv_m + hd[:, None] * stride_qv_h + db[None, :]
    q_b = tl.load(q_ptr).to(tl.int32) & 0xFF
    q_lo = _e2m1_decode(q_b & 0xF)
    q_hi = _e2m1_decode((q_b >> 4) & 0xF)
    q_sc = tl.load(q_scales_ptr + m * stride_qs_m + hd)  # [H_SEL] int32
    q_scale = tl.zeros((H_SEL, VALUE_BYTES), dtype=tl.float32)
    for b in tl.static_range(SCALE_BLOCKS):
        sval = _block_scale(q_sc, b)  # [H_SEL]
        q_scale = tl.where(blk[None, :] == b, sval[:, None], q_scale)
    q_lo = q_lo * q_scale
    q_hi = q_hi * q_scale

    # ---- dequant k[k_idx]: [VALUE_BYTES, BLOCK_N] (identical to the full kernel) ----
    k_ptr = k_values_ptr + k_idx[None, :].to(tl.int64) * stride_kv_n + db[:, None]
    k_b = tl.load(k_ptr, mask=valid_n[None, :], other=0).to(tl.int32) & 0xFF
    k_lo = _e2m1_decode(k_b & 0xF)
    k_hi = _e2m1_decode((k_b >> 4) & 0xF)
    k_sc = tl.load(k_scales_ptr + k_idx, mask=valid_n, other=0)  # [BLOCK_N]
    k_scale = tl.zeros((VALUE_BYTES, BLOCK_N), dtype=tl.float32)
    for b in tl.static_range(SCALE_BLOCKS):
        sval = _block_scale(k_sc, b)  # [BLOCK_N]
        k_scale = tl.where(blk[:, None] == b, sval[None, :], k_scale)
    k_lo = k_lo * k_scale
    k_hi = k_hi * k_scale

    scores = tl.dot(q_lo, k_lo, input_precision=INPUT_PRECISION)
    scores += tl.dot(q_hi, k_hi, input_precision=INPUT_PRECISION)  # [H_SEL, BLOCK_N]

    w = tl.load(weights_ptr + m * stride_w_m + hd).to(tl.float32)  # [H_SEL]
    logits = tl.sum(tl.maximum(scores, 0.0) * w[:, None], axis=0)  # [BLOCK_N]

    out_val = tl.where(n_rel < row_len, logits, float("-inf"))
    tl.store(out_ptr + m_off * stride_o_m + n_rel, out_val, mask=n_rel < max_seqlen_k)


def indexer_mqa_logits_hhead_sm12x_triton(
    q_values: torch.Tensor,
    q_scales: torch.Tensor,
    k_values: torch.Tensor,
    k_scales: torch.Tensor,
    weights: torch.Tensor,
    head_idx: torch.Tensor,
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
    max_seqlen_k: int,
    head_dim: int = 128,
    block_n: int = 128,
    input_precision: str = "ieee",
) -> torch.Tensor:
    """MISA-dagger pass 1: compact indexer logits using each query's top-``H_SEL``
    heads only (``head_idx``), a drop-in for ``indexer_mqa_logits_sm12x_triton``
    restricted to a per-query head subset.

    Args:
        head_idx: Per-query selected head indices, ``[num_q, H_SEL]`` int32
            (e.g. ``weights.abs().topk(H_SEL, dim=1).indices``). All other args
            match ``indexer_mqa_logits_sm12x_triton``.

    Returns:
        Compact logits ``[num_q, max_seqlen_k]`` fp32 == the full-head kernel run
        with the non-selected heads' weights zeroed.
    """
    num_q, num_heads, value_bytes = q_values.shape
    h_sel = head_idx.shape[1]
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
    if h_sel > num_heads:
        raise ValueError(f"head_idx H_SEL {h_sel} > num_heads {num_heads}")
    # H_SEL is a constexpr block dim: tl.arange(0, H_SEL) needs a power of 2 and the
    # tl.dot M dim needs >= 16. Fail loudly here rather than as a raw Triton error.
    if h_sel < 16 or (h_sel & (h_sel - 1)) != 0:
        raise ValueError(
            f"head_idx H_SEL {h_sel} must be a power of 2 in [16, {num_heads}]"
        )

    q_values = q_values.contiguous()
    q_scales = q_scales.contiguous()
    k_values = k_values.contiguous()
    k_scales = k_scales.contiguous()
    weights = weights.contiguous()
    head_idx = head_idx.contiguous().to(torch.int32)
    cu_seqlen_ks = cu_seqlen_ks.contiguous()
    cu_seqlen_ke = cu_seqlen_ke.contiguous()

    grid = (num_q, triton.cdiv(max_seqlen_k, block_n))
    _indexer_mqa_logits_hhead_kernel[grid](
        q_values,
        q_scales,
        k_values,
        k_scales,
        weights,
        head_idx,
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
        head_idx.stride(0),
        out.stride(0),
        H_SEL=h_sel,
        VALUE_BYTES=value_bytes,
        SCALE_BLOCKS=head_dim // _MXFP4_BLOCK,
        BYTES_PER_BLOCK=_MXFP4_BLOCK // 2,
        BLOCK_N=block_n,
        INPUT_PRECISION=input_precision,
    )
    return out


@triton.jit
def _indexer_mqa_logits_gather_kernel(
    q_values_ptr,  # int8  [num_q, H, VALUE_BYTES]
    q_scales_ptr,  # int32 [num_q, H]
    k_values_ptr,  # int8  [num_kv, VALUE_BYTES]
    k_scales_ptr,  # int32 [num_kv]
    weights_ptr,  # fp32  [num_q, H]
    cand_ptr,  # int32 [num_q, C]   per-query candidate columns (compact, 0-based)
    ks_ptr,  # int32 [num_q]
    ke_ptr,  # int32 [num_q]
    out_ptr,  # fp32  [num_q, C]
    num_kv,
    C,
    stride_qv_m,
    stride_qv_h,
    stride_qs_m,
    stride_kv_n,
    stride_w_m,
    stride_cand_m,
    stride_o_m,
    H: tl.constexpr,
    VALUE_BYTES: tl.constexpr,
    SCALE_BLOCKS: tl.constexpr,
    BYTES_PER_BLOCK: tl.constexpr,
    BLOCK_N: tl.constexpr,
    INPUT_PRECISION: tl.constexpr,
):
    m = tl.program_id(0)
    cb = tl.program_id(1)
    m_off = m.to(tl.int64)

    ks = tl.maximum(tl.load(ks_ptr + m), 0)
    ke = tl.minimum(tl.load(ke_ptr + m), num_kv)
    row_len = tl.maximum(ke - ks, 0)

    c_rel = cb * BLOCK_N + tl.arange(0, BLOCK_N)  # candidate slots for this tile
    valid_c = c_rel < C
    # Candidate compact column for each slot; a column past this row's window
    # (row_len) is a -inf pad from topk on a short row -> score it -inf.
    cand_col = tl.load(cand_ptr + m_off * stride_cand_m + c_rel, mask=valid_c, other=0)
    valid = valid_c & (cand_col < row_len) & (cand_col >= 0)
    k_idx = ks + cand_col  # actual kv row, valid only where ``valid``

    hd = tl.arange(0, H)
    db = tl.arange(0, VALUE_BYTES)
    blk = db // BYTES_PER_BLOCK

    # ---- dequant q[m], all H heads: [H, VALUE_BYTES] ----
    q_ptr = q_values_ptr + m_off * stride_qv_m + hd[:, None] * stride_qv_h + db[None, :]
    q_b = tl.load(q_ptr).to(tl.int32) & 0xFF
    q_lo = _e2m1_decode(q_b & 0xF)
    q_hi = _e2m1_decode((q_b >> 4) & 0xF)
    q_sc = tl.load(q_scales_ptr + m * stride_qs_m + hd)  # [H]
    q_scale = tl.zeros((H, VALUE_BYTES), dtype=tl.float32)
    for b in tl.static_range(SCALE_BLOCKS):
        sval = _block_scale(q_sc, b)
        q_scale = tl.where(blk[None, :] == b, sval[:, None], q_scale)
    q_lo = q_lo * q_scale
    q_hi = q_hi * q_scale

    # ---- dequant gathered k[k_idx]: [VALUE_BYTES, BLOCK_N] (scattered gather) ----
    k_ptr = k_values_ptr + k_idx[None, :].to(tl.int64) * stride_kv_n + db[:, None]
    k_b = tl.load(k_ptr, mask=valid[None, :], other=0).to(tl.int32) & 0xFF
    k_lo = _e2m1_decode(k_b & 0xF)
    k_hi = _e2m1_decode((k_b >> 4) & 0xF)
    k_sc = tl.load(k_scales_ptr + k_idx, mask=valid, other=0)  # [BLOCK_N]
    k_scale = tl.zeros((VALUE_BYTES, BLOCK_N), dtype=tl.float32)
    for b in tl.static_range(SCALE_BLOCKS):
        sval = _block_scale(k_sc, b)
        k_scale = tl.where(blk[:, None] == b, sval[None, :], k_scale)
    k_lo = k_lo * k_scale
    k_hi = k_hi * k_scale

    scores = tl.dot(q_lo, k_lo, input_precision=INPUT_PRECISION)
    scores += tl.dot(q_hi, k_hi, input_precision=INPUT_PRECISION)  # [H, BLOCK_N]

    w = tl.load(weights_ptr + m * stride_w_m + hd).to(tl.float32)  # [H]
    logits = tl.sum(tl.maximum(scores, 0.0) * w[:, None], axis=0)  # [BLOCK_N]

    out_val = tl.where(valid, logits, float("-inf"))
    tl.store(out_ptr + m_off * stride_o_m + c_rel, out_val, mask=valid_c)


def indexer_mqa_logits_gather_sm12x_triton(
    q_values: torch.Tensor,
    q_scales: torch.Tensor,
    k_values: torch.Tensor,
    k_scales: torch.Tensor,
    weights: torch.Tensor,
    cand: torch.Tensor,
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
    head_dim: int = 128,
    block_n: int = 128,
    input_precision: str = "ieee",
) -> torch.Tensor:
    """MISA-dagger pass 2: re-score each query's ``C`` candidate columns with ALL
    heads, returning ``[num_q, C]`` exact logits over the candidate axis.

    ``cand`` holds compact (0-based, per-query window) candidate columns, e.g. from
    ``torch.topk(pass1_logits, C, dim=1).indices``. Candidate ``c`` maps to kv row
    ``cu_seqlen_ks[m] + cand[m, c]``; a candidate column >= the row window length is
    a -inf pad and is scored ``-inf``. The output aligns with ``cand`` column-wise,
    so ``full(-inf).scatter_(1, cand, out)`` rebuilds a compact logits tile for the
    unchanged top-k selector.

    Returns:
        Logits ``[num_q, C]`` fp32 (``-inf`` in padded candidate slots).
    """
    num_q, num_heads, value_bytes = q_values.shape
    c_tokens = cand.shape[1]
    device = q_values.device
    out = torch.full(
        (num_q, c_tokens), float("-inf"), dtype=torch.float32, device=device
    )
    num_kv = k_values.shape[0]
    if num_q == 0 or num_kv == 0 or c_tokens == 0:
        return out
    if value_bytes != head_dim // 2:
        raise ValueError(
            f"q_values last dim {value_bytes} != head_dim//2 {head_dim // 2}"
        )

    q_values = q_values.contiguous()
    q_scales = q_scales.contiguous()
    k_values = k_values.contiguous()
    k_scales = k_scales.contiguous()
    weights = weights.contiguous()
    cand = cand.contiguous().to(torch.int32)
    cu_seqlen_ks = cu_seqlen_ks.contiguous()
    cu_seqlen_ke = cu_seqlen_ke.contiguous()

    grid = (num_q, triton.cdiv(c_tokens, block_n))
    _indexer_mqa_logits_gather_kernel[grid](
        q_values,
        q_scales,
        k_values,
        k_scales,
        weights,
        cand,
        cu_seqlen_ks,
        cu_seqlen_ke,
        out,
        num_kv,
        c_tokens,
        q_values.stride(0),
        q_values.stride(1),
        q_scales.stride(0),
        k_values.stride(0),
        weights.stride(0),
        cand.stride(0),
        out.stride(0),
        H=num_heads,
        VALUE_BYTES=value_bytes,
        SCALE_BLOCKS=head_dim // _MXFP4_BLOCK,
        BYTES_PER_BLOCK=_MXFP4_BLOCK // 2,
        BLOCK_N=block_n,
        INPUT_PRECISION=input_precision,
    )
    return out


@triton.jit
def _indexer_mqa_logits_paged_kernel(
    q_values_ptr,  # int8  [num_tokens, H, VALUE_BYTES]
    q_scales_ptr,  # int32 [num_tokens, H]
    cache_ptr,  # int8  [num_pages, CACHE_BLOCK_SIZE * (VALUE_BYTES + SCALE_BLOCKS)];
    #             structure-of-arrays per page: a value region of
    #             CACHE_BLOCK_SIZE * VALUE_BYTES bytes then a scale region of
    #             CACHE_BLOCK_SIZE * SCALE_BLOCKS e8m0 bytes (one per 32-dim block).
    block_table_ptr,  # int32 [num_tokens, max_blocks]
    context_lens_ptr,  # int32 [num_tokens]
    weights_ptr,  # fp32  [num_tokens, H]
    out_ptr,  # fp32  [num_tokens, max_context_len]
    max_context_len,
    stride_qv_m,
    stride_qv_h,
    stride_qs_m,
    stride_cache_page,
    stride_bt_m,
    stride_w_m,
    stride_o_m,
    H: tl.constexpr,
    VALUE_BYTES: tl.constexpr,  # head_dim // 2
    SCALE_BLOCKS: tl.constexpr,  # head_dim // 32
    BYTES_PER_BLOCK: tl.constexpr,  # value bytes per block = 32 // 2 = 16
    CACHE_BLOCK_SIZE: tl.constexpr,  # slots per page
    BLOCK_N: tl.constexpr,
    INPUT_PRECISION: tl.constexpr,
):
    t = tl.program_id(0)
    nb = tl.program_id(1)
    t_off = t.to(tl.int64)

    ctx_len = tl.load(context_lens_ptr + t)
    j = nb * BLOCK_N + tl.arange(0, BLOCK_N)  # KV positions for this token
    # Whole tile is past this token's context: write -inf and skip dequant+dot.
    # The decode cudagraph captures an over-sized grid; at short context most
    # tiles are empty, so this early-out is the dominant decode speedup.
    if nb * BLOCK_N >= ctx_len:
        neg = tl.full((BLOCK_N,), float("-inf"), tl.float32)
        tl.store(out_ptr + t_off * stride_o_m + j, neg, mask=j < max_context_len)
        return
    valid = (j < ctx_len) & (j < max_context_len)

    # Paged location: page = block_table[t, j // CACHE_BLOCK_SIZE], slot = j % size.
    # The MXFP4 indexer cache is laid out structure-of-arrays PER PAGE (matching
    # the production writer and the deep_gemm paged reader): a contiguous value
    # region of CACHE_BLOCK_SIZE * VALUE_BYTES bytes (slot s value byte db at
    # s * VALUE_BYTES + db), followed by a scale region (slot s block-b e8m0 byte
    # at VALUE_REGION + s * SCALE_BLOCKS + b). It is NOT per-slot interleaved.
    blk_idx = j // CACHE_BLOCK_SIZE
    slot = j % CACHE_BLOCK_SIZE
    page = tl.load(block_table_ptr + t * stride_bt_m + blk_idx, mask=valid, other=0)
    page_off = page.to(tl.int64) * stride_cache_page  # [BLOCK_N] byte base of the page
    slot64 = slot.to(tl.int64)
    slot_val_off = page_off + slot64 * VALUE_BYTES  # [BLOCK_N] value-region base
    slot_scale_off = (
        page_off + CACHE_BLOCK_SIZE * VALUE_BYTES + slot64 * SCALE_BLOCKS
    )  # [BLOCK_N] scale-region base

    hd = tl.arange(0, H)
    db = tl.arange(0, VALUE_BYTES)
    blk = db // BYTES_PER_BLOCK  # value-byte -> 32-dim block index

    # ---- dequant q[t]: [H, VALUE_BYTES] (identical to the contiguous kernel) ----
    q_ptr = q_values_ptr + t_off * stride_qv_m + hd[:, None] * stride_qv_h + db[None, :]
    q_b = tl.load(q_ptr).to(tl.int32) & 0xFF
    q_lo = _e2m1_decode(q_b & 0xF)
    q_hi = _e2m1_decode((q_b >> 4) & 0xF)
    q_sc = tl.load(q_scales_ptr + t_off * stride_qs_m + hd)
    q_scale = tl.zeros((H, VALUE_BYTES), dtype=tl.float32)
    for b in tl.static_range(SCALE_BLOCKS):
        sval = _block_scale(q_sc, b)
        q_scale = tl.where(blk[None, :] == b, sval[:, None], q_scale)
    q_lo = q_lo * q_scale
    q_hi = q_hi * q_scale

    # ---- dequant paged k: [VALUE_BYTES, BLOCK_N], SoA value region ----
    # value byte db of slot s lives at page_off + s * VALUE_BYTES + db.
    k_ptr = cache_ptr + slot_val_off[None, :] + db[:, None]
    k_b = tl.load(k_ptr, mask=valid[None, :], other=0).to(tl.int32) & 0xFF
    k_lo = _e2m1_decode(k_b & 0xF)
    k_hi = _e2m1_decode((k_b >> 4) & 0xF)
    # scale byte for block b of slot s lives at page_off + VALUE_REGION
    # + s * SCALE_BLOCKS + b (SCALE_BLOCKS contiguous e8m0 bytes per slot).
    k_scale = tl.zeros((VALUE_BYTES, BLOCK_N), dtype=tl.float32)
    for b in tl.static_range(SCALE_BLOCKS):
        sb = tl.load(cache_ptr + slot_scale_off + b, mask=valid, other=0)
        sval = tl.exp2((sb.to(tl.int32) & 0xFF).to(tl.float32) - 127.0)  # [BLOCK_N]
        k_scale = tl.where(blk[:, None] == b, sval[None, :], k_scale)
    k_lo = k_lo * k_scale
    k_hi = k_hi * k_scale

    scores = tl.dot(q_lo, k_lo, input_precision=INPUT_PRECISION)
    scores += tl.dot(q_hi, k_hi, input_precision=INPUT_PRECISION)  # [H, BLOCK_N]

    w = tl.load(weights_ptr + t_off * stride_w_m + hd).to(tl.float32)
    logits = tl.sum(tl.maximum(scores, 0.0) * w[:, None], axis=0)  # [BLOCK_N]

    out_val = tl.where(j < ctx_len, logits, float("-inf"))
    tl.store(out_ptr + t_off * stride_o_m + j, out_val, mask=j < max_context_len)


def indexer_mqa_logits_paged_sm12x_triton(
    q_values: torch.Tensor,
    q_scales: torch.Tensor,
    cache_2d: torch.Tensor,
    block_table: torch.Tensor,
    context_lens: torch.Tensor,
    weights: torch.Tensor,
    cache_block_size: int,
    max_context_len: int,
    head_dim: int = 128,
    block_n: int = 128,
    input_precision: str = "ieee",
) -> torch.Tensor:
    """Paged DeepSeek-V4 indexer MQA-logits (Triton), drop-in for the decode
    ``fp8_fp4_paged_mqa_logits``.

    The KV is read straight from the paged MXFP4 indexer cache (no gather): each
    query token ``t`` scores its ``context_lens[t]`` keys, gathered from
    ``cache_2d`` via ``block_table[t]``. The per-slot head_dim is block-interleaved
    (``[16 value bytes | 1 e8m0 scale byte]`` per 32-dim block).

    Args:
        q_values: Packed fp4 query, ``[num_tokens, num_heads, head_dim // 2]`` int8.
        q_scales: int32-packed e8m0 q scales, ``[num_tokens, num_heads]``.
        cache_2d: Paged MXFP4 indexer cache, ``[num_pages, cache_block_size * row_bytes]`` int8.
        block_table: Per-token page table, ``[num_tokens, max_blocks]`` int32.
        context_lens: Per-token KV length, ``[num_tokens]`` int32.
        weights: Per-head indexer weights, ``[num_tokens, num_heads]`` (cast to fp32).
        cache_block_size: Slots per page.
        max_context_len: Width of the output logits.
        head_dim: Indexer per-head dim (128 for DeepSeek-V4).
        block_n: KV tile width.
        input_precision: ``tl.dot`` precision (``"ieee"`` default).

    Returns:
        Logits ``[num_tokens, max_context_len]`` fp32: ``out[t, 0:context_lens[t]]``
        are the scores; columns >= context_lens[t] are -inf.
    """
    num_tokens, num_heads, value_bytes = q_values.shape
    device = q_values.device
    out = torch.full(
        (num_tokens, max_context_len), float("-inf"), dtype=torch.float32, device=device
    )
    if num_tokens == 0 or max_context_len <= 0:
        return out
    if value_bytes != head_dim // 2:
        raise ValueError(
            f"q_values last dim {value_bytes} != head_dim//2 {head_dim // 2}"
        )

    row_bytes = cache_2d.shape[1] // cache_block_size
    scale_blocks = head_dim // _MXFP4_BLOCK
    bytes_per_block = _MXFP4_BLOCK // 2  # 16
    # SoA per-page layout: CACHE_BLOCK_SIZE value rows (scale_blocks * 16 bytes)
    # followed by CACHE_BLOCK_SIZE scale rows (scale_blocks e8m0 bytes).
    value_bytes_per_slot = scale_blocks * bytes_per_block  # 64
    scale_bytes_per_slot = scale_blocks  # one e8m0 byte per 32-dim block
    if row_bytes != value_bytes_per_slot + scale_bytes_per_slot:
        raise ValueError(
            f"cache row_bytes {row_bytes} != "
            f"{value_bytes_per_slot + scale_bytes_per_slot} (expected SoA per page: "
            f"{value_bytes_per_slot} value + {scale_bytes_per_slot} scale bytes/slot)"
        )

    q_values = q_values.contiguous()
    q_scales = q_scales.contiguous()
    cache_2d = cache_2d.contiguous()
    block_table = block_table.contiguous()
    context_lens = context_lens.contiguous()
    weights = weights.contiguous()

    grid = (num_tokens, triton.cdiv(max_context_len, block_n))
    _indexer_mqa_logits_paged_kernel[grid](
        q_values,
        q_scales,
        cache_2d,
        block_table,
        context_lens,
        weights,
        out,
        max_context_len,
        q_values.stride(0),
        q_values.stride(1),
        q_scales.stride(0),
        cache_2d.stride(0),
        block_table.stride(0),
        weights.stride(0),
        out.stride(0),
        H=num_heads,
        VALUE_BYTES=value_bytes,
        SCALE_BLOCKS=scale_blocks,
        BYTES_PER_BLOCK=bytes_per_block,
        CACHE_BLOCK_SIZE=cache_block_size,
        BLOCK_N=block_n,
        INPUT_PRECISION=input_precision,
    )
    return out
