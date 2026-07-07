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
from tokenspeed_kernel.ops.attention.torch.indexer_mqa_logits_sm12x import (
    indexer_mqa_logits_sm12x,
)
from tokenspeed_kernel.ops.attention.triton.indexer_mqa_logits_sm12x import (
    indexer_mqa_logits_gather_sm12x_triton,
    indexer_mqa_logits_hhead_sm12x_triton,
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
    b = torch.randint(
        120, 135, shape + (4,), dtype=torch.int64, device=device, generator=gen
    )
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
    qv = torch.randint(
        0, 256, (num_q, H, VB), dtype=torch.uint8, device=device, generator=gen
    ).view(torch.int8)
    qs = _scales((num_q, H), gen, device)
    kv = torch.randint(
        0, 256, (num_kv, VB), dtype=torch.uint8, device=device, generator=gen
    ).view(torch.int8)
    ksc = _scales((num_kv,), gen, device)
    w = torch.randn(num_q, H, dtype=torch.float32, device=device, generator=gen)

    if mode == "ragged":
        ks = torch.randint(
            0,
            max(1, num_kv // 2),
            (num_q,),
            dtype=torch.int32,
            device=device,
            generator=gen,
        )
        win = torch.randint(
            1, max_len + 1, (num_q,), dtype=torch.int32, device=device, generator=gen
        )
        ke = torch.minimum(ks + win, torch.full_like(ks, num_kv))
    elif mode == "overflow":  # ke may exceed num_kv -> exercises the clamp
        ks = torch.randint(
            0, num_kv, (num_q,), dtype=torch.int32, device=device, generator=gen
        )
        ke = ks + torch.randint(
            1, max_len + 8, (num_q,), dtype=torch.int32, device=device, generator=gen
        )
    else:
        ks = torch.zeros(num_q, dtype=torch.int32, device=device)
        ke = torch.randint(
            1, num_kv + 1, (num_q,), dtype=torch.int32, device=device, generator=gen
        )

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
    assert _topk_sets_match(
        ref, tri, eff
    ), "prefill top-k selection differs from reference"


def test_indexer_prefill_tf32_topk_close(device):
    """The tf32 serving precision keeps the selection essentially identical."""
    num_q, num_kv, max_len = 96, 512, 512
    gen = torch.Generator(device=device).manual_seed(7)
    qv = torch.randint(
        0, 256, (num_q, H, VB), dtype=torch.uint8, device=device, generator=gen
    ).view(torch.int8)
    qs = _scales((num_q, H), gen, device)
    kv = torch.randint(
        0, 256, (num_kv, VB), dtype=torch.uint8, device=device, generator=gen
    ).view(torch.int8)
    ksc = _scales((num_kv,), gen, device)
    w = torch.randn(num_q, H, dtype=torch.float32, device=device, generator=gen)
    ks = torch.zeros(num_q, dtype=torch.int32, device=device)
    ke = torch.randint(
        1, num_kv + 1, (num_q,), dtype=torch.int32, device=device, generator=gen
    )

    ref = indexer_mqa_logits_sm12x(qv, qs, kv, ksc, w, ks, ke, max_len, head_dim=D)
    tf32 = indexer_mqa_logits_sm12x_triton(
        qv, qs, kv, ksc, w, ks, ke, max_len, head_dim=D, input_precision="tf32"
    )
    assert bool((torch.isinf(ref) == torch.isinf(tf32)).all())
    assert _cos(ref, tf32) > 0.999


# --------------------------------------------------------------------------- #
# MISA-dagger fast path (pass-1 h-head prefilter + pass-2 candidate re-score)
# --------------------------------------------------------------------------- #
def _dense_qkvw(num_q: int, num_kv: int, seed: int, device: str):
    """Random MXFP4-packed q/k/w with a dense window ks=0, ke=num_kv."""
    gen = torch.Generator(device=device).manual_seed(seed)
    qv = torch.randint(
        0, 256, (num_q, H, VB), dtype=torch.uint8, device=device, generator=gen
    ).view(torch.int8)
    qs = _scales((num_q, H), gen, device)
    kv = torch.randint(
        0, 256, (num_kv, VB), dtype=torch.uint8, device=device, generator=gen
    ).view(torch.int8)
    ksc = _scales((num_kv,), gen, device)
    w = torch.randn(num_q, H, dtype=torch.float32, device=device, generator=gen)
    ks = torch.zeros(num_q, dtype=torch.int32, device=device)
    ke = torch.full((num_q,), num_kv, dtype=torch.int32, device=device)
    return qv, qs, kv, ksc, w, ks, ke


@pytest.mark.parametrize(
    "num_q,num_kv,h_sel",
    [
        pytest.param(64, 512, 32, id="h32-512"),
        pytest.param(128, 2048, 32, id="h32-2048"),
        pytest.param(96, 1024, 16, id="h16-1024"),
    ],
)
def test_indexer_hhead_matches_masked_full(device, num_q, num_kv, h_sel):
    """Pass-1: scoring each query's top-h heads == the full kernel with the
    non-selected heads' weights zeroed (a zero-weight head contributes 0 to
    logits = SUM_h ReLU(dot)*w, so this is exact up to fp add-order)."""
    qv, qs, kv, ksc, w, ks, ke = _dense_qkvw(
        num_q, num_kv, num_q + num_kv + h_sel, device
    )
    max_len = num_kv
    head_idx = w.abs().topk(h_sel, dim=1).indices.to(torch.int32)
    w_mask = torch.zeros_like(w).scatter_(
        1, head_idx.long(), w.gather(1, head_idx.long())
    )

    masked = indexer_mqa_logits_sm12x_triton(
        qv, qs, kv, ksc, w_mask, ks, ke, max_len, head_dim=D, input_precision="ieee"
    )
    hhead = indexer_mqa_logits_hhead_sm12x_triton(
        qv,
        qs,
        kv,
        ksc,
        w,
        head_idx,
        ks,
        ke,
        max_len,
        head_dim=D,
        input_precision="ieee",
    )
    # Scoring the top-h heads is exact vs zeroing the rest (masked heads add 0);
    # only tensor-core fp add-order differs -> near-identical logits + same top-k.
    assert bool(
        (torch.isinf(masked) == torch.isinf(hhead)).all()
    ), "-inf padding mismatch"
    assert _cos(masked, hhead) > 0.9999
    assert _topk_sets_match(masked, hhead, [num_kv] * num_q)


def test_indexer_gather_matches_full_at_candidates(device):
    """Pass-2: re-scoring C candidate columns == the full kernel's logits gathered
    at those columns (same per-element arithmetic -> bit-identical)."""
    num_q, num_kv, C = 128, 4096, 1024
    qv, qs, kv, ksc, w, ks, ke = _dense_qkvw(num_q, num_kv, 99, device)
    full = indexer_mqa_logits_sm12x_triton(
        qv, qs, kv, ksc, w, ks, ke, num_kv, head_dim=D, input_precision="ieee"
    )
    cand = full.topk(C, dim=1).indices.to(torch.int32)  # distinct valid columns
    gathered = indexer_mqa_logits_gather_sm12x_triton(
        qv, qs, kv, ksc, w, cand, ks, ke, head_dim=D, input_precision="ieee"
    )
    ref = full.gather(1, cand.long())
    # Same per-element arithmetic as the full kernel -> near-identical (fp add-order).
    assert _cos(ref, gathered) > 0.9999
    fin = torch.isfinite(ref)
    rel = (gathered[fin] - ref[fin]).abs() / ref[fin].abs().clamp_min(1.0)
    assert float(rel.max()) < 1e-3


def test_indexer_misa_fast_equals_masked_topk(device):
    """The full 2-pass fast path (hhead -> topk C -> gather -> scatter) selects the
    SAME top-512 as the masked-weight reference (misa_dagger) it replaces."""
    num_q, num_kv, h_sel, C, topk = 128, 4096, 32, 1024, 512
    qv, qs, kv, ksc, w, ks, ke = _dense_qkvw(num_q, num_kv, 7, device)
    max_len = num_kv
    head_idx = w.abs().topk(h_sel, dim=1).indices.to(torch.int32)
    w_mask = torch.zeros_like(w).scatter_(
        1, head_idx.long(), w.gather(1, head_idx.long())
    )
    full = indexer_mqa_logits_sm12x_triton(
        qv, qs, kv, ksc, w, ks, ke, max_len, head_dim=D, input_precision="ieee"
    )

    # masked reference (misa_dagger)
    masked = indexer_mqa_logits_sm12x_triton(
        qv, qs, kv, ksc, w_mask, ks, ke, max_len, head_dim=D, input_precision="ieee"
    )
    cand_ref = masked.topk(C, dim=1).indices
    ref_logits = torch.full_like(full, float("-inf")).scatter_(
        1, cand_ref, full.gather(1, cand_ref)
    )

    # fast path (dedicated kernels)
    hhead = indexer_mqa_logits_hhead_sm12x_triton(
        qv,
        qs,
        kv,
        ksc,
        w,
        head_idx,
        ks,
        ke,
        max_len,
        head_dim=D,
        input_precision="ieee",
    )
    cand_fast = hhead.topk(C, dim=1).indices.to(torch.int32)
    l_cand = indexer_mqa_logits_gather_sm12x_triton(
        qv, qs, kv, ksc, w, cand_fast, ks, ke, head_dim=D, input_precision="ieee"
    )
    fast_logits = torch.full_like(full, float("-inf")).scatter_(
        1, cand_fast.long(), l_cand
    )

    assert _topk_sets_match(ref_logits, fast_logits, [num_kv] * num_q, topk=topk)


@pytest.mark.parametrize("bad_h", [8, 24, 48])
def test_indexer_hhead_rejects_unsupported_h_sel(device, bad_h):
    """The wrapper rejects H_SEL that is <16 or not a power of 2 (tl.arange /
    tl.dot M constraints) with a clear ValueError, not a raw Triton error."""
    qv, qs, kv, ksc, w, ks, ke = _dense_qkvw(8, 64, 1, device)
    head_idx = torch.zeros(8, bad_h, dtype=torch.int32, device=device)
    with pytest.raises(ValueError):
        indexer_mqa_logits_hhead_sm12x_triton(
            qv, qs, kv, ksc, w, head_idx, ks, ke, 64, head_dim=D
        )


# --------------------------------------------------------------------------- #
# Paged decode (block-interleaved MXFP4 cache + block_table)
# --------------------------------------------------------------------------- #
_VALUE_BYTES = _SCALE_BLOCKS * _BYTES_PER_BLOCK  # 64 value bytes per slot
_SCALE_BYTES = _SCALE_BLOCKS  # one e8m0 byte per 32-dim block


def _pack_paged_cache(
    kv_values_i8: torch.Tensor,
    kv_scales_i32: torch.Tensor,
    num_pages: int,
    bs: int,
    device: str,
) -> torch.Tensor:
    """Pack logical [N, VB] values + [N] int32 scales into the paged cache's
    structure-of-arrays layout (what the production writer emits and deep_gemm
    reads): per page a value region ``bs x VB`` bytes followed by a scale region
    ``bs x SCALE_BLOCKS`` e8m0 bytes."""
    n = kv_values_i8.shape[0]
    kvv = kv_values_i8.view(torch.uint8)
    cache = torch.zeros(num_pages, bs * _ROW_BYTES, dtype=torch.uint8, device=device)
    pos = torch.arange(n, device=device)
    page, slot = pos // bs, pos % bs
    vb_idx = torch.arange(_VALUE_BYTES, device=device)
    cache[page[:, None], (slot * _VALUE_BYTES)[:, None] + vb_idx[None, :]] = kvv
    scale_base = bs * _VALUE_BYTES
    for b in range(_SCALE_BLOCKS):
        cache[page, scale_base + slot * _SCALE_BYTES + b] = (
            (kv_scales_i32 >> (8 * b)) & 0xFF
        ).to(torch.uint8)
    return cache.view(torch.int8)


def _soa_gather(
    cache_2d: torch.Tensor, num_kv: int, bs: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """Inverse of ``_pack_paged_cache``: read logical values/scales back out of
    the SoA cache (used to build the reference for a production-writer cache)."""
    raw = cache_2d.view(torch.uint8)
    pos = torch.arange(num_kv, device=device)
    page, slot = pos // bs, pos % bs
    vb_idx = torch.arange(_VALUE_BYTES, device=device)
    kv = raw[page[:, None], (slot * _VALUE_BYTES)[:, None] + vb_idx[None, :]]
    scale_base = bs * _VALUE_BYTES
    packed = torch.zeros(num_kv, dtype=torch.int64, device=device)
    for b in range(_SCALE_BYTES):
        packed |= raw[page, scale_base + slot * _SCALE_BYTES + b].to(torch.int64) << (
            8 * b
        )
    return kv.contiguous().view(torch.int8), packed.to(torch.int32)


@pytest.mark.parametrize(
    "num_tokens,num_kv,block_size",
    [
        pytest.param(4, 40, 16, id="nt4-kv40"),
        pytest.param(16, 200, 64, id="nt16-kv200"),
        pytest.param(8, 130, 64, id="nt8-straddle"),
        pytest.param(160, 1024, 64, id="nt160-kv1024"),
        pytest.param(8, 28000, 64, id="nt8-kv28000-longctx"),
    ],
)
def test_indexer_paged_decode_matches_reference(device, num_tokens, num_kv, block_size):
    gen = torch.Generator(device=device).manual_seed(num_tokens * 977 + num_kv)
    qv = torch.randint(
        0, 256, (num_tokens, H, VB), dtype=torch.uint8, device=device, generator=gen
    ).view(torch.int8)
    qs = _scales((num_tokens, H), gen, device)
    kvv = torch.randint(
        0, 256, (num_kv, VB), dtype=torch.uint8, device=device, generator=gen
    ).view(torch.int8)
    kvs = _scales((num_kv,), gen, device)
    w = torch.randn(num_tokens, H, dtype=torch.float32, device=device, generator=gen)
    ctx = torch.randint(
        1, num_kv + 1, (num_tokens,), dtype=torch.int32, device=device, generator=gen
    )
    max_ctx = int(ctx.max())

    num_pages = (num_kv + block_size - 1) // block_size
    cache = _pack_paged_cache(kvv, kvs, num_pages, block_size, device)
    block_table = torch.arange(num_pages, dtype=torch.int32, device=device)[
        None, :
    ].repeat(num_tokens, 1)

    # Each token scores [0, ctx[t]) of the shared pool: reference window = [0, ctx).
    ref = indexer_mqa_logits_sm12x(
        qv,
        qs,
        kvv,
        kvs,
        w,
        torch.zeros(num_tokens, dtype=torch.int32, device=device),
        ctx,
        max_ctx,
        head_dim=D,
    )
    tri = indexer_mqa_logits_paged_sm12x_triton(
        qv,
        qs,
        cache,
        block_table,
        ctx,
        w,
        block_size,
        max_ctx,
        head_dim=D,
        input_precision="ieee",
    )

    eff = [min(int(ctx[m]), max_ctx) for m in range(num_tokens)]
    assert bool((torch.isinf(ref) == torch.isinf(tri)).all()), "-inf padding mismatch"
    assert _cos(ref, tri) > 0.9999
    assert _topk_sets_match(
        ref, tri, eff
    ), "paged decode top-k selection differs from reference"


@pytest.mark.parametrize("num_kv", [1024, 20000], ids=["kv1024", "kv20000-longctx"])
def test_indexer_paged_decode_matches_production_writer(device, num_kv):
    """Lock the paged-decode kernel to the SoA cache layout emitted by the
    production MXFP4 writer (the layout deep_gemm reads). This is the guard the
    original kernel lacked: it assumed a per-slot block-interleaved layout, so a
    self-consistent hand-packed cache passed while the real serve dropped the
    early-context needles. Writes the cache with the real writer, then checks the
    Triton kernel against the torch reference on the same keys, at long context.
    """
    write_deepseek_v4_indexer_mxfp4_cache_cuda = pytest.importorskip(
        "tokenspeed_kernel.ops.attention.triton.deepseek_v4"
    ).write_deepseek_v4_indexer_mxfp4_cache_cuda

    num_tokens, bs = 8, 64
    gen = torch.Generator(device=device).manual_seed(num_kv * 13 + 5)
    index_k = torch.randn(num_kv, D, dtype=torch.bfloat16, device=device, generator=gen)
    num_pages = (num_kv + bs - 1) // bs
    cache = torch.zeros(
        num_pages, bs * _ROW_BYTES, dtype=torch.uint8, device=device
    ).view(torch.int8)
    slot_mapping = torch.arange(num_kv, dtype=torch.int64, device=device)
    valid = torch.ones(num_kv, dtype=torch.int32, device=device)
    write_deepseek_v4_indexer_mxfp4_cache_cuda(index_k, cache, slot_mapping, valid, bs)

    kv_i8, ks_i32 = _soa_gather(cache, num_kv, bs, device)
    qv = torch.randint(
        0, 256, (num_tokens, H, VB), dtype=torch.uint8, device=device, generator=gen
    ).view(torch.int8)
    qs = _scales((num_tokens, H), gen, device)
    w = torch.randn(num_tokens, H, dtype=torch.float32, device=device, generator=gen)
    ctx = torch.randint(
        1, num_kv + 1, (num_tokens,), dtype=torch.int32, device=device, generator=gen
    )
    max_ctx = int(ctx.max())
    block_table = torch.arange(num_pages, dtype=torch.int32, device=device)[
        None, :
    ].repeat(num_tokens, 1)

    ref = indexer_mqa_logits_sm12x(
        qv,
        qs,
        kv_i8,
        ks_i32,
        w,
        torch.zeros(num_tokens, dtype=torch.int32, device=device),
        ctx,
        max_ctx,
        head_dim=D,
    )
    tri = indexer_mqa_logits_paged_sm12x_triton(
        qv,
        qs,
        cache,
        block_table,
        ctx,
        w,
        bs,
        max_ctx,
        head_dim=D,
        input_precision="ieee",
    )

    eff = [min(int(ctx[m]), max_ctx) for m in range(num_tokens)]
    assert bool((torch.isinf(ref) == torch.isinf(tri)).all()), "-inf padding mismatch"
    assert _cos(ref, tri) > 0.9999
    assert _topk_sets_match(
        ref, tri, eff
    ), "paged decode top-k selection differs from the production-writer cache"
