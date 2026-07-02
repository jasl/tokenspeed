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

"""FlashInfer SM120 packed sparse-MLA for DeepSeek-V4 (consumer Blackwell).

The bundled FlashMLA wheel ships no sm_120/sm_121 sparse-MLA kernels:
``flash_mla_with_kvcache`` / ``flash_mla_sparse_fwd`` raise "Unsupported
architecture" on consumer Blackwell. FlashInfer's official SM120 packed
sparse-MLA kernels (PR3395 lineage, flashinfer main / v0.6.14rc1+) fill the
gap. This module wraps the low-level ``_SparseMLAPagedAttentionRunner`` behind
a ``flash_mla``-compatible surface so the DeepSeek-V4 attention backend can
dispatch to it on sm12x without reaching outside the kernel-package boundary.

The runner serves BOTH phases from one call surface: FlashInfer
auto-dispatches ``num_tokens <= 64`` to its split-K decode kernels and larger
batches to its SM120 prefill orchestrator (``sparse_mla_sm120_prefill.cu``),
so a whole prefill chunk goes down in a single call with per-token top-k
indices.
"""

from __future__ import annotations

import functools

import torch

# DeepSeek-V4 value head dim; the SM120 sparse-MLA kernel requires d_v == 512.
_DSV4_VALUE_HEAD_DIM = 512


@functools.cache
def sparse_mla_sm120_available() -> bool:
    """Return True iff FlashInfer's SM120 packed sparse-MLA runner is importable.

    Cached: the import probe (and any first-time JIT module discovery) runs once.
    """
    try:
        from flashinfer.mla._sparse_mla_sm120 import (  # noqa: F401
            _SparseMLAPagedAttentionRunner,
        )
    except Exception:
        return False
    return True


@functools.cache
def _paged_runner(device_index: int, d_v: int):
    """Build and cache one runner per device.

    Constructed without ``max_num_tokens``/``max_num_heads`` so its LSE buffer
    and split-K decode scratch grow lazily — no global worst-case sizing needed
    for first-serve correctness. The underlying kernel JIT-compiles on first use.
    """
    from flashinfer.mla._sparse_mla_sm120 import _SparseMLAPagedAttentionRunner

    return _SparseMLAPagedAttentionRunner(
        d_v=d_v,
        kv_scale_format="auto",
        device=torch.device("cuda", device_index),
    )


def _as_uint8_cache(cache: torch.Tensor) -> torch.Tensor:
    """View a packed fp8_ds_mla cache as raw bytes (the kernel reads uint8)."""
    if cache.dtype == torch.uint8:
        return cache
    return cache.view(torch.uint8)


def _as_decode_indices(indices: torch.Tensor) -> torch.Tensor:
    """Normalize sparse top-k indices to ``[num_tokens, topk]`` int32, contiguous.

    The DeepSeek-V4 backend may carry a singleton query dim (``[T, 1, topk]``)
    from the FlashMLA call convention; squeeze it to match the SM120 kernel,
    which reads the last dim as ``topk``.
    """
    if indices.dim() == 3 and indices.shape[1] == 1:
        indices = indices.squeeze(1)
    if indices.dtype != torch.int32:
        indices = indices.to(torch.int32)
    return indices.contiguous()


def sparse_mla_sm120_paged_attention(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    indices: torch.Tensor,
    softmax_scale: float,
    *,
    head_dim_v: int = _DSV4_VALUE_HEAD_DIM,
    attn_sink: torch.Tensor | None = None,
    extra_k_cache: torch.Tensor | None = None,
    extra_indices: torch.Tensor | None = None,
    topk_length: torch.Tensor | None = None,
    extra_topk_length: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run FlashInfer's SM120 packed sparse-MLA for DeepSeek-V4.

    Drop-in replacement for ``flash_mla_with_kvcache`` (decode) and, with
    per-token indices for a whole chunk, ``flash_mla_sparse_fwd`` (prefill):
    FlashInfer auto-dispatches ``num_tokens <= 64`` to its split-K decode
    kernels and larger batches to its SM120 prefill orchestrator. The
    SWA/compressed caches are already packed fp8_ds_mla; this upcasts ``q`` to
    bf16 (the kernel consumes a bf16 query), views the caches as raw bytes, and
    drives the low-level runner, which allocates its split-K scratch lazily.

    Constraints (FlashInfer SM120 dispatch tables): the main-cache page size
    must be 64 and the extra-cache page size 64 or 2; dual-cache calls require
    main ``topk == 128`` (the DSv4 sliding window); prefill-sized calls
    (``num_tokens > 64``) require ``num_heads`` in {16, 32, 64, 128}.

    Args:
        q: Query, ``[num_tokens, num_heads, d_qk]`` or
            ``[num_tokens, 1, num_heads, d_qk]`` (fp8 or bf16). ``d_qk`` is 512
            for DeepSeek-V4; fp8 inputs are upcast to bf16.
        k_cache: Packed fp8_ds_mla SWA cache,
            ``[num_blocks, page_size, 1, bytes_per_token]``.
        indices: SWA top-k slot indices, ``[num_tokens, topk]`` int32, ``-1``
            marks invalid slots (skipped by the kernel).
        softmax_scale: Attention softmax scale.
        head_dim_v: Value head dim (512 for DeepSeek-V4).
        attn_sink: Optional per-head attention sink, ``[num_heads]`` float32.
        extra_k_cache: Optional packed compressed cache (same layout as
            ``k_cache``).
        extra_indices: Optional compressed top-k slot indices,
            ``[num_tokens, extra_topk]`` int32.
        topk_length: Optional valid SWA top-k length per token, ``[num_tokens]``.
        extra_topk_length: Optional valid compressed top-k length per token.

    Returns:
        Attention output ``[num_tokens, num_heads, head_dim_v]`` bf16.
    """
    if q.dim() == 4:
        if q.shape[1] != 1:
            raise ValueError(
                f"4-D q requires a singleton query dim, got shape {tuple(q.shape)}"
            )
        q = q.squeeze(1)
    if q.dtype != torch.bfloat16:
        q = q.to(torch.bfloat16)
    q = q.contiguous()

    num_tokens, num_heads, _ = q.shape
    output = torch.empty(
        (num_tokens, num_heads, head_dim_v),
        dtype=torch.bfloat16,
        device=q.device,
    )

    runner = _paged_runner(q.device.index, head_dim_v)
    runner.run(
        q,
        _as_uint8_cache(k_cache),
        _as_decode_indices(indices),
        output,
        softmax_scale,
        topk_length=topk_length,
        attn_sink=attn_sink,
        extra_kv_cache=(
            _as_uint8_cache(extra_k_cache) if extra_k_cache is not None else None
        ),
        extra_indices=(
            _as_decode_indices(extra_indices) if extra_indices is not None else None
        ),
        extra_topk_length=extra_topk_length,
    )
    return output


# Established decode call-site name; the runner underneath serves both phases.
sparse_mla_sm120_decode = sparse_mla_sm120_paged_attention
