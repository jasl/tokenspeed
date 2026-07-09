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

"""Consumer-Blackwell DeepSeek-V4 indexer MQA-logits over FP8 inputs (Triton).

The FP8 sibling of ``indexer_mqa_logits_sm12x.py``: same compact output layout
and per-query window contract, but queries and keys are fp8e4m3 with fp32
scales instead of packed MXFP4. FP8 elements upconvert with a single hardware
instruction, so the kernel spends its time in ``tl.dot`` instead of the e2m1
shift/mask/exp2 decode ladders that make the MXFP4 kernel ALU-bound (the same
trade the vLLM-fork ``_fp8_mqa_logits_kernel`` makes; that kernel scores its
prefill 1.5x faster than the MXFP4 path on the identical GB10 workload).

Scale conventions:
- ``k_scales``: fp32 per KV token; ``k[n, :] = k_fp8[n, :] * k_scales[n]``.
- ``q_scales``: optional fp32 per (query, head). Rather than rescaling the
  ``[H, D]`` query tile, the kernel folds it into the per-head weights --
  ``logits[m, n] = sum_h ReLU(<q_fp8[m,h], k[n]>) * (w[m,h] * q_scales[m,h])``
  -- which is exact because ReLU is positively homogeneous and the q scale is
  per-head constant along the contraction. Pass ``None`` when the producer
  already folded the scale into ``weights``.

Output is the COMPACT, 0-based layout (identical to the MXFP4 kernel and the
torch reference): ``out[m, 0:L]`` holds scores over
``k[cu_seqlen_ks[m] : cu_seqlen_ks[m]+L]`` (L = ke - ks); columns >= L are
``-inf``. Feeds ``_deepseek_v4_indexer_topk_from_logits`` unchanged.
"""

from __future__ import annotations

import torch

from tokenspeed_kernel._triton import tl, triton

_INDEXER_HEAD_DIM = 128


@triton.jit
def _indexer_fp8_mqa_logits_kernel(
    q_ptr,  # fp8e4m3 [num_q, H, D]
    q_scales_ptr,  # fp32 [num_q, H] (ignored when HAS_Q_SCALES == False)
    k_ptr,  # fp8e4m3 [num_kv, D]
    k_scales_ptr,  # fp32 [num_kv]
    weights_ptr,  # fp32 [num_q, H]
    ks_ptr,  # int32 [num_q]   per-query window start into kv
    ke_ptr,  # int32 [num_q]   per-query window end (exclusive)
    out_ptr,  # fp32 [num_q, max_seqlen_k]
    num_kv,
    max_seqlen_k,
    stride_q_m,
    stride_q_h,
    stride_qs_m,
    stride_k_n,
    stride_w_m,
    stride_o_m,
    H: tl.constexpr,
    D: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HAS_Q_SCALES: tl.constexpr,
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
    # Whole tile is past this row's window: write -inf and skip the dot.
    if nb * BLOCK_N >= row_len:
        neg = tl.full((BLOCK_N,), float("-inf"), tl.float32)
        tl.store(out_ptr + m_off * stride_o_m + n_rel, neg, mask=n_rel < max_seqlen_k)
        return
    valid_n = (n_rel < row_len) & (n_rel < max_seqlen_k)
    k_idx = ks + n_rel  # actual kv row in [0, num_kv), valid only where valid_n

    hd = tl.arange(0, H)  # [H]
    dd = tl.arange(0, D)  # [D]

    # ---- q[m]: [H, D] fp8 -> fp32 (hardware convert; scale folded into w) ----
    q = tl.load(q_ptr + m_off * stride_q_m + hd[:, None] * stride_q_h + dd[None, :]).to(
        tl.float32
    )

    # ---- k[k_idx]: transposed [D, BLOCK_N] fp8 -> fp32, per-token scale ----
    k = tl.load(
        k_ptr + k_idx[None, :].to(tl.int64) * stride_k_n + dd[:, None],
        mask=valid_n[None, :],
        other=0.0,
    ).to(tl.float32)
    k_sc = tl.load(k_scales_ptr + k_idx, mask=valid_n, other=0.0)  # [BLOCK_N] fp32
    k = k * k_sc[None, :]

    # ---- scores[h, n] = <q[m,h], k[n]> ----
    scores = tl.dot(q, k, input_precision=INPUT_PRECISION)  # [H, BLOCK_N]

    # ---- logits[n] = sum_h ReLU(scores[h, n]) * w_eff[m, h] ----
    w = tl.load(weights_ptr + m * stride_w_m + hd).to(tl.float32)  # [H]
    if HAS_Q_SCALES:
        q_sc = tl.load(q_scales_ptr + m * stride_qs_m + hd)  # [H] fp32
        w = w * q_sc
    logits = tl.sum(tl.maximum(scores, 0.0) * w[:, None], axis=0)  # [BLOCK_N]

    out_val = tl.where(n_rel < row_len, logits, float("-inf"))
    tl.store(out_ptr + m_off * stride_o_m + n_rel, out_val, mask=n_rel < max_seqlen_k)


def indexer_fp8_mqa_logits_sm12x_triton(
    q_fp8: torch.Tensor,
    q_scales: torch.Tensor | None,
    k_fp8: torch.Tensor,
    k_scales: torch.Tensor,
    weights: torch.Tensor,
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
    max_seqlen_k: int,
    block_n: int = 128,
    input_precision: str = "tf32",
) -> torch.Tensor:
    """DeepSeek-V4 indexer lightning-MQA logits over FP8 inputs (Triton).

    Args:
        q_fp8: Query, ``[num_q, num_heads, head_dim]`` fp8e4m3.
        q_scales: Optional per-head fp32 dequant scales ``[num_q, num_heads]``;
            folded into ``weights`` inside the kernel (exact under ReLU).
            Pass ``None`` if already folded by the producer.
        k_fp8: Gathered KV, ``[num_kv, head_dim]`` fp8e4m3.
        k_scales: Per-token fp32 dequant scales, ``[num_kv]``.
        weights: Per-head indexer weights, ``[num_q, num_heads]`` (cast to fp32).
        cu_seqlen_ks: Per-query KV window start into the gathered KV, ``[num_q]``.
        cu_seqlen_ke: Per-query KV window end (exclusive), ``[num_q]``.
        max_seqlen_k: Width of the output logits (max window length).
        block_n: KV tile width (program granularity along the KV axis).
        input_precision: ``tl.dot`` precision -- ``"tf32"`` (default; the
            indexer is a top-k selector, tf32 is accuracy-safe) or ``"ieee"``.

    Returns:
        Compact logits ``[num_q, max_seqlen_k]`` fp32: ``out[m, 0:L]`` are the
        scores over ``k[cu_seqlen_ks[m] : cu_seqlen_ke[m]]`` (L = ke - ks);
        columns >= L are -inf.
    """
    num_q, num_heads, head_dim = q_fp8.shape
    device = q_fp8.device
    out = torch.full(
        (num_q, max_seqlen_k), float("-inf"), dtype=torch.float32, device=device
    )
    num_kv = k_fp8.shape[0]
    if num_q == 0 or num_kv == 0 or max_seqlen_k <= 0:
        return out

    if k_fp8.shape[-1] != head_dim:
        raise ValueError(
            f"k_fp8 last dim {k_fp8.shape[-1]} != q head_dim {head_dim}"
        )
    if head_dim != _INDEXER_HEAD_DIM:
        raise ValueError(f"head_dim {head_dim} != {_INDEXER_HEAD_DIM}")

    q_fp8 = q_fp8.contiguous()
    k_fp8 = k_fp8.contiguous()
    k_scales = k_scales.contiguous().to(torch.float32)
    weights = weights.contiguous().to(torch.float32)
    cu_seqlen_ks = cu_seqlen_ks.contiguous()
    cu_seqlen_ke = cu_seqlen_ke.contiguous()
    has_q_scales = q_scales is not None
    if has_q_scales:
        q_scales = q_scales.contiguous().to(torch.float32)
        qs_arg, qs_stride = q_scales, q_scales.stride(0)
    else:
        qs_arg, qs_stride = weights, 0  # unused dummy pointer

    grid = (num_q, triton.cdiv(max_seqlen_k, block_n))
    _indexer_fp8_mqa_logits_kernel[grid](
        q_fp8,
        qs_arg,
        k_fp8,
        k_scales,
        weights,
        cu_seqlen_ks,
        cu_seqlen_ke,
        out,
        num_kv,
        max_seqlen_k,
        q_fp8.stride(0),
        q_fp8.stride(1),
        qs_stride,
        k_fp8.stride(0),
        weights.stride(0),
        out.stride(0),
        H=num_heads,
        D=head_dim,
        BLOCK_N=block_n,
        HAS_Q_SCALES=has_q_scales,
        INPUT_PRECISION=input_precision,
        num_warps=4,
    )
    return out
