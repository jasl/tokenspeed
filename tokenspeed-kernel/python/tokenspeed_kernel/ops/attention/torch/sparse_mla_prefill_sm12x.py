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

"""Consumer-Blackwell (sm_120/sm_121) MLA sparse-attention prefill (torch).

FlashMLA's ``sparse_prefill_fwd`` is sm90a/sm100f only and raises on consumer
Blackwell. This is a portable torch fallback with identical semantics, used to
unblock DeepSeek-V4 prefill on sm12x. It is correctness-first (chunked over query
tokens to bound the gathered-KV memory); a fused Triton kernel is the perf
follow-up.

MLA absorbed form: the key is the full ``d_qk`` latent vector and the value is the
first ``d_v`` (512) dims of the SAME vector; ``h_kv == 1`` (one latent KV head
shared across all query heads).
"""

from __future__ import annotations

import torch

# Query-token chunk size: bounds the gathered-KV tensor to
# [chunk, topk, d_qk] so prefill of long prompts does not allocate multi-GB.
_PREFILL_Q_CHUNK = 512


def sparse_mla_prefill_sm12x(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    sm_scale: float,
    d_v: int = 512,
    attn_sink: torch.Tensor | None = None,
    topk_length: torch.Tensor | None = None,
) -> tuple[torch.Tensor, None, None]:
    """MLA sparse-attention prefill, drop-in for ``flash_mla_sparse_fwd``.

    Args:
        q: Query, ``[s_q, h_q, d_qk]`` bf16.
        kv: Gathered latent KV, ``[s_kv, 1, d_qk]`` bf16 (already dequantized).
        indices: Top-k KV indices, ``[s_q, 1, topk]`` int32; entries < 0 or
            >= s_kv are invalid (masked out).
        sm_scale: Attention softmax scale.
        d_v: Value head dim (512 for DeepSeek-V4); the value is ``kv[..., :d_v]``.
        attn_sink: Optional per-head sink ``[h_q]`` float32; output is scaled by
            ``exp(lse) / (exp(lse) + exp(attn_sink))`` (a +inf sink zeros the
            output, a -inf sink is a no-op).
        topk_length: Optional valid top-k count per query ``[s_q]`` int32; query
            ``i`` attends only to ``indices[i, 0, :topk_length[i]]``.

    Returns:
        ``(output, None, None)`` where output is ``[s_q, h_q, d_v]`` bf16. The
        two trailing ``None`` mirror FlashMLA's ``(output, max_logits, lse)``
        signature; callers in this codebase use only the output.
    """
    if q.dim() != 3:
        raise ValueError(f"q must be [s_q, h_q, d_qk], got {tuple(q.shape)}")
    s_q, h_q, d_qk = q.shape
    s_kv = kv.shape[0]
    out_dtype = q.dtype
    kv_flat = kv.reshape(s_kv, kv.shape[-1])  # [s_kv, d_qk]
    idx = indices.reshape(s_q, -1)  # [s_q, topk]
    topk = idx.shape[1]
    device = q.device

    output = torch.empty((s_q, h_q, d_v), dtype=out_dtype, device=device)
    if s_q == 0:
        return output, None, None

    rank = torch.arange(topk, device=device)
    sink = attn_sink.to(torch.float32) if attn_sink is not None else None

    for start in range(0, s_q, _PREFILL_Q_CHUNK):
        end = min(start + _PREFILL_Q_CHUNK, s_q)
        q_b = q[start:end].to(torch.float32)  # [b, h_q, d_qk]
        idx_b = idx[start:end]  # [b, topk]

        valid = (idx_b >= 0) & (idx_b < s_kv)  # [b, topk]
        if topk_length is not None:
            valid = valid & (rank[None, :] < topk_length[start:end, None])

        gathered = kv_flat[idx_b.clamp_(0, max(s_kv - 1, 0)).long()].to(
            torch.float32
        )  # [b, topk, d_qk]

        scores = torch.einsum("bhd,btd->bht", q_b, gathered) * sm_scale
        scores = scores.masked_fill(~valid[:, None, :], float("-inf"))

        weights = torch.softmax(scores, dim=-1)  # [b, h_q, topk]
        # All-invalid rows softmax to NaN; force them to zero contribution.
        weights = torch.nan_to_num(weights, nan=0.0)
        out_b = torch.einsum("bht,btd->bhd", weights, gathered[:, :, :d_v])

        if sink is not None:
            lse = torch.logsumexp(scores, dim=-1)  # [b, h_q]
            # exp(lse)/(exp(lse)+exp(sink)) == sigmoid(lse - sink), computed in
            # log space so large lse cannot overflow fp32 into inf/NaN. The
            # nan_to_num keeps all-invalid rows (lse == sink == -inf) at zero.
            scale = torch.sigmoid(lse - sink[None, :])
            out_b = out_b * torch.nan_to_num(scale, nan=0.0)[:, :, None]

        output[start:end] = out_b.to(out_dtype)

    return output, None, None
