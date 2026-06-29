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

"""Numerics tests for the consumer-Blackwell DeepSeek-V4 indexer MQA-logits.

The Triton prefill (``indexer_mqa_logits_sm12x_triton``) and paged-decode
(``indexer_mqa_logits_paged_sm12x_triton``) scoring kernels replace deep_gemm's
``fp8_fp4_mqa_logits`` / ``fp8_fp4_paged_mqa_logits`` on sm_120/sm_121. Both are
checked against the portable torch reference (``indexer_mqa_logits_sm12x``): the
top-k *selection* (what feeds the downstream sparse attention) must be identical,
the -inf padding must match, and the finite logits must be ~exactly equal at the
``ieee`` precision used by the reference. A separate check confirms the ``tf32``
serving precision stays close (its top-k may differ only on near-ties).
"""

from __future__ import annotations

import pytest
import torch

from tokenspeed_kernel.ops.attention.indexer_mqa_logits_sm12x import (
    indexer_mqa_logits_sm12x,
)
from tokenspeed_kernel.ops.attention.triton.indexer_mqa_logits_sm12x import (
    indexer_mqa_logits_paged_sm12x_triton,
    indexer_mqa_logits_sm12x_triton,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="indexer MQA-logits kernels require CUDA",
)

# DeepSeek-V4 indexer dims.
H, D, VB = 64, 128, 64
_SCALE_BLOCKS = D // 32
_BYTES_PER_BLOCK = 32 // 2  # 16 value bytes per 32-dim block
_BLOCK_BYTES = _BYTES_PER_BLOCK + 1  # 16 value + 1 e8m0 scale
_ROW_BYTES = _SCALE_BLOCKS * _BLOCK_BYTES  # 68


def _scales(shape: tuple[int, ...], gen: torch.Generator, device: str) -> torch.Tensor:
    """int32-packed e8m0 scales near exponent 0 (finite, scale ~1)."""
    b = torch.randint(120, 135, shape + (4,), dtype=torch.int64, device=device, generator=gen)
    packed = b[..., 0] | (b[..., 1] << 8) | (b[..., 2] << 16) | (b[..., 3] << 24)
    return packed.to(torch.int32)


def _cos(ref: torch.Tensor, other: torch.Tensor) -> float:
    mask = torch.isfinite(ref) & torch.isfinite(other)
    if not bool(mask.any()):
        return 1.0
    return torch.nn.functional.cosine_similarity(
        ref[mask].flatten(), other[mask].flatten(), dim=0
    ).item()


def _topk_sets_match(
    ref: torch.Tensor, other: torch.Tensor, lengths: list[int], topk: int = 512
) -> bool:
    for m, length in enumerate(lengths):
        if length <= 0:
            continue
        kk = min(topk, length)
        ri = set(ref[m, :length].topk(kk).indices.tolist())
        oi = set(other[m, :length].topk(kk).indices.tolist())
        if ri != oi:
            return False
    return True


# --------------------------------------------------------------------------- #
# Prefill (non-paged, contiguous gathered KV)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "num_q,num_kv,max_len,mode",
    [
        pytest.param(5, 8, 8, "dense", id="dense-tiny"),
        pytest.param(50, 40, 40, "dense", id="dense-40"),
        pytest.param(128, 512, 512, "dense", id="dense-512"),
        pytest.param(64, 300, 128, "ragged", id="ragged-ks>0"),
        pytest.param(48, 64, 128, "overflow", id="clamp-ke>num_kv"),
    ],
)
def test_indexer_prefill_matches_reference(device, num_q, num_kv, max_len, mode):
    gen = torch.Generator(device=device).manual_seed(num_q * 131 + num_kv)
    qv = torch.randint(0, 256, (num_q, H, VB), dtype=torch.uint8, device=device, generator=gen).view(torch.int8)
    qs = _scales((num_q, H), gen, device)
    kv = torch.randint(0, 256, (num_kv, VB), dtype=torch.uint8, device=device, generator=gen).view(torch.int8)
    ksc = _scales((num_kv,), gen, device)
    w = torch.randn(num_q, H, dtype=torch.float32, device=device, generator=gen)

    if mode == "ragged":
        ks = torch.randint(0, max(1, num_kv // 2), (num_q,), dtype=torch.int32, device=device, generator=gen)
        win = torch.randint(1, max_len + 1, (num_q,), dtype=torch.int32, device=device, generator=gen)
        ke = torch.minimum(ks + win, torch.full_like(ks, num_kv))
    elif mode == "overflow":  # ke may exceed num_kv -> exercises the clamp
        ks = torch.randint(0, num_kv, (num_q,), dtype=torch.int32, device=device, generator=gen)
        ke = ks + torch.randint(1, max_len + 8, (num_q,), dtype=torch.int32, device=device, generator=gen)
    else:
        ks = torch.zeros(num_q, dtype=torch.int32, device=device)
        ke = torch.randint(1, num_kv + 1, (num_q,), dtype=torch.int32, device=device, generator=gen)

    ref = indexer_mqa_logits_sm12x(qv, qs, kv, ksc, w, ks, ke, max_len, head_dim=D)
    tri = indexer_mqa_logits_sm12x_triton(
        qv, qs, kv, ksc, w, ks, ke, max_len, head_dim=D, input_precision="ieee"
    )

    # Effective compact length per row matches the reference's clamp to [0, num_kv].
    eff = [
        min(max(min(int(ke[m]), num_kv) - min(int(ks[m]), num_kv), 0), max_len)
        for m in range(num_q)
    ]
    assert bool((torch.isinf(ref) == torch.isinf(tri)).all()), "-inf padding mismatch"
    assert _cos(ref, tri) > 0.9999
    assert _topk_sets_match(ref, tri, eff), "prefill top-k selection differs from reference"


def test_indexer_prefill_tf32_topk_close(device):
    """The tf32 serving precision keeps the selection essentially identical."""
    num_q, num_kv, max_len = 96, 512, 512
    gen = torch.Generator(device=device).manual_seed(7)
    qv = torch.randint(0, 256, (num_q, H, VB), dtype=torch.uint8, device=device, generator=gen).view(torch.int8)
    qs = _scales((num_q, H), gen, device)
    kv = torch.randint(0, 256, (num_kv, VB), dtype=torch.uint8, device=device, generator=gen).view(torch.int8)
    ksc = _scales((num_kv,), gen, device)
    w = torch.randn(num_q, H, dtype=torch.float32, device=device, generator=gen)
    ks = torch.zeros(num_q, dtype=torch.int32, device=device)
    ke = torch.randint(1, num_kv + 1, (num_q,), dtype=torch.int32, device=device, generator=gen)

    ref = indexer_mqa_logits_sm12x(qv, qs, kv, ksc, w, ks, ke, max_len, head_dim=D)
    tf32 = indexer_mqa_logits_sm12x_triton(
        qv, qs, kv, ksc, w, ks, ke, max_len, head_dim=D, input_precision="tf32"
    )
    assert bool((torch.isinf(ref) == torch.isinf(tf32)).all())
    assert _cos(ref, tf32) > 0.999


# --------------------------------------------------------------------------- #
# Paged decode (block-interleaved MXFP4 cache + block_table)
# --------------------------------------------------------------------------- #
def _pack_paged_cache(
    kv_values_i8: torch.Tensor, kv_scales_i32: torch.Tensor, num_pages: int, bs: int, device: str
) -> torch.Tensor:
    """Pack logical [N, VB] values + [N] int32 scales into the block-interleaved
    paged cache layout: per 32-dim block ``[16 value bytes | 1 e8m0 scale byte]``."""
    n = kv_values_i8.shape[0]
    kvv = kv_values_i8.view(torch.uint8)
    cache = torch.zeros(num_pages, bs * _ROW_BYTES, dtype=torch.uint8, device=device)
    pos = torch.arange(n, device=device)
    page, slot = pos // bs, pos % bs
    ii = torch.arange(_BYTES_PER_BLOCK, device=device)
    for b in range(_SCALE_BLOCKS):
        dst = slot * _ROW_BYTES + b * _BLOCK_BYTES
        cache[page[:, None], dst[:, None] + ii[None, :]] = kvv[:, b * _BYTES_PER_BLOCK : (b + 1) * _BYTES_PER_BLOCK]
        cache[page, dst + _BYTES_PER_BLOCK] = ((kv_scales_i32 >> (8 * b)) & 0xFF).to(torch.uint8)
    return cache.view(torch.int8)


@pytest.mark.parametrize(
    "num_tokens,num_kv,block_size",
    [
        pytest.param(4, 40, 16, id="nt4-kv40"),
        pytest.param(16, 200, 64, id="nt16-kv200"),
        pytest.param(8, 130, 64, id="nt8-straddle"),
        pytest.param(160, 1024, 64, id="nt160-kv1024"),
    ],
)
def test_indexer_paged_decode_matches_reference(device, num_tokens, num_kv, block_size):
    gen = torch.Generator(device=device).manual_seed(num_tokens * 977 + num_kv)
    qv = torch.randint(0, 256, (num_tokens, H, VB), dtype=torch.uint8, device=device, generator=gen).view(torch.int8)
    qs = _scales((num_tokens, H), gen, device)
    kvv = torch.randint(0, 256, (num_kv, VB), dtype=torch.uint8, device=device, generator=gen).view(torch.int8)
    kvs = _scales((num_kv,), gen, device)
    w = torch.randn(num_tokens, H, dtype=torch.float32, device=device, generator=gen)
    ctx = torch.randint(1, num_kv + 1, (num_tokens,), dtype=torch.int32, device=device, generator=gen)
    max_ctx = int(ctx.max())

    num_pages = (num_kv + block_size - 1) // block_size
    cache = _pack_paged_cache(kvv, kvs, num_pages, block_size, device)
    block_table = torch.arange(num_pages, dtype=torch.int32, device=device)[None, :].repeat(num_tokens, 1)

    # Each token scores [0, ctx[t]) of the shared pool: reference window = [0, ctx).
    ref = indexer_mqa_logits_sm12x(
        qv, qs, kvv, kvs, w,
        torch.zeros(num_tokens, dtype=torch.int32, device=device), ctx, max_ctx, head_dim=D,
    )
    tri = indexer_mqa_logits_paged_sm12x_triton(
        qv, qs, cache, block_table, ctx, w, block_size, max_ctx, head_dim=D, input_precision="ieee"
    )

    eff = [min(int(ctx[m]), max_ctx) for m in range(num_tokens)]
    assert bool((torch.isinf(ref) == torch.isinf(tri)).all()), "-inf padding mismatch"
    assert _cos(ref, tri) > 0.9999
    assert _topk_sets_match(ref, tri, eff), "paged decode top-k selection differs from reference"
