# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.

"""GPU parity: FlashInfer SM120 sparse-MLA prefill vs the torch sm12x fallback.

Op-level numerical parity between the NEW packed-cache path
(``sparse_mla_sm120_paged_attention``, FlashInfer's SM120 sparse-MLA runner:
prefill orchestrator for ``num_tokens > 64``, split-K decode kernels at or
below 64) and the EXISTING dequant-gather torch fallback
(``sparse_mla_prefill_sm12x``).

Both sides consume byte-identical packed fp8_ds_mla rows: the FI path reads
the paged caches directly while the reference reads the production
dequantization (``dequantize_deepseek_v4_fp8_ds_mla_cache``) of the very same
rows, so fp8 quantization error cancels and the comparison isolates kernel
numerics (bf16 MMA vs fp32-accumulated einsum softmax). The FI call takes the
SWA window as the main segment and the compressed rows as the extra segment;
the reference attends over the union (compressed-then-SWA concatenated into
one combined index row) -- mathematically the same softmax.

Requires an sm12x GPU with FlashInfer's SM120 sparse-MLA runner (GB10 /
sm_121, RTX PRO 6000 / sm_120); skips cleanly anywhere else, CPU CI included.
"""

from __future__ import annotations

import unittest
from dataclasses import dataclass

import torch
from tokenspeed_kernel.ops.attention.flashinfer.sparse_mla_sm120 import (
    sparse_mla_sm120_available,
    sparse_mla_sm120_paged_attention,
)
from tokenspeed_kernel.ops.attention.torch.sparse_mla_prefill_sm12x import (
    sparse_mla_prefill_sm12x,
)

from tokenspeed.runtime.configs.deepseek_v4_cache_spec import (
    deepseek_v4_swa_scale_dim,
    deepseek_v4_swa_token_stride,
)
from tokenspeed.runtime.layers.attention.deepseek_v4_ops import (
    dequantize_deepseek_v4_fp8_ds_mla_cache,
    fused_qnorm_rope_kv_insert,
)

HEAD_DIM = 512
ROPE_DIM = 64
SWA_TOKEN_STRIDE = deepseek_v4_swa_token_stride(HEAD_DIM, ROPE_DIM)
SWA_SCALE_DIM = deepseek_v4_swa_scale_dim(HEAD_DIM, ROPE_DIM)
ROW_BYTES = SWA_TOKEN_STRIDE + SWA_SCALE_DIM
# FlashInfer SM120 dispatch envelope: main-cache page size must be 64 and the
# dual-cache main topk must be 128 (the DSv4 sliding window). See
# `_prefill_fi_sm120_eligible` in the DeepSeek-V4 attention backend.
MAIN_PAGE_SIZE = 64
WINDOW = 128
# TP=2 serving shape; the >64-token prefill dispatch allows {16, 32, 64, 128}.
NUM_HEADS = 64
SM_SCALE = HEAD_DIM**-0.5
RMS_EPS = 1.0e-6
COS_SIN_ROWS = 512
# bf16 tensor-core kernel vs fp32-softmax reference: closeness, not equality.
MIN_ROW_COSINE = 0.99
# Worst-row tolerance: the SM120 kernel computes the compressed/extra segment
# (and topk >= 512 main segments) in FP8 compute mode, while the reference
# consumes a full-precision q -- that q-quantization error does not cancel in
# the parity and shows up as a few-percent worst-row delta (measured 3.5-6.9%
# on GB10 with min cosine > 0.99 and an exact end-to-end 2k-token
# continuation). The decode-seam case pins the envelope: the SAME kernel has
# served production decode (GSM8K 0.97) since June with ~2% mean row error vs
# this fp32 reference, so ~2% mean is the kernel's native numerics, not an
# indexing bug (those corrupt whole rows: cosine collapse + mean explosion).
MAX_REL_ROW_NORM_ERR = 1.0e-1
MAX_MEAN_REL_ROW_NORM_ERR = 3.0e-2


def _fi_sm120_parity_supported() -> bool:
    if not torch.cuda.is_available():
        return False
    if torch.cuda.get_device_capability()[0] != 12:
        return False
    return sparse_mla_sm120_available()


def _fp8_ds_mla_cache_view(cache_2d: torch.Tensor, page_size: int) -> torch.Tensor:
    """[pages, page_size * row_bytes] uint8 -> [pages, page_size, 1, row_bytes].

    Mirrors the backend's `_fp8_ds_mla_cache_view` (the shape the FlashInfer
    runner expects over the footer-scale packed pages).
    """
    row_bytes = cache_2d.shape[1] // page_size
    return torch.as_strided(
        cache_2d,
        (cache_2d.shape[0], page_size, 1, row_bytes),
        (cache_2d.stride(0), row_bytes, row_bytes, 1),
    )


def _page_bases(rows_per_request: list[int], page_size: int) -> tuple[list[int], int]:
    """Identity block table: request r owns consecutive whole pages.

    Returns each request's base slot (page_base * page_size, so global slot =
    base + local position) and the total page count.
    """
    bases: list[int] = []
    next_page = 0
    for rows in rows_per_request:
        bases.append(next_page * page_size)
        next_page += max(1, -(-rows // page_size))
    return bases, next_page


def _insert_random_packed_rows(
    cache_2d: torch.Tensor,
    slots: torch.Tensor,
    page_size: int,
    cos_sin: torch.Tensor,
) -> None:
    """Fill packed fp8_ds_mla rows with the production quantize+insert op.

    Row content is random bf16 latent (RoPE-rotated/quantized by the op); the
    exact transform is irrelevant because both paths under test read back the
    SAME packed bytes. The dummy q is mutated in place by the op and dropped.
    """
    num_rows = slots.numel()
    device = cache_2d.device
    latent = torch.randn(num_rows, HEAD_DIM, device=device, dtype=torch.bfloat16)
    q_dummy = torch.randn(num_rows, 1, HEAD_DIM, device=device, dtype=torch.bfloat16)
    positions = torch.arange(num_rows, device=device, dtype=torch.int64) % COS_SIN_ROWS
    fused_qnorm_rope_kv_insert(
        q=q_dummy,
        kv=latent,
        swa_kv_cache_2d=cache_2d,
        slot_mapping=slots,
        positions=positions,
        cos_sin_cache=cos_sin,
        rms_norm_eps=RMS_EPS,
        block_size=page_size,
    )


@dataclass(frozen=True)
class _ParityCase:
    q: torch.Tensor
    attn_sink: torch.Tensor | None
    swa_cache_2d: torch.Tensor
    swa_indices: torch.Tensor
    swa_lens: torch.Tensor
    comp_cache_2d: torch.Tensor | None
    comp_page_size: int
    extra_indices: torch.Tensor | None
    extra_lens: torch.Tensor | None
    kv_flat: torch.Tensor
    combined_indices: torch.Tensor
    combined_lens: torch.Tensor


def _build_parity_case(
    *,
    device: torch.device,
    seq_lens: tuple[int, ...],
    compress_ratio: int | None,
    comp_page_size: int = MAIN_PAGE_SIZE,
    extra_width: int = 64,
    with_sink: bool = True,
    seed: int = 2026,
) -> _ParityCase:
    """Fabricate one packed-cache parity case (full prefill of each request).

    Per-token index rows keep every valid slot in a contiguous prefix (valid
    count == topk_length, `-1` padding strictly after), so the prefix and
    count-of-valid interpretations of ``topk_length`` agree on both paths.
    """
    torch.manual_seed(seed)
    cpu_rng = torch.Generator().manual_seed(seed + 1)
    cos_sin = (
        torch.randn(COS_SIN_ROWS, ROPE_DIM, device=device, dtype=torch.float32) * 0.05
    )

    # Main (SWA) packed cache: one row per token position, identity page map.
    swa_bases, num_swa_pages = _page_bases(list(seq_lens), MAIN_PAGE_SIZE)
    swa_cache = torch.zeros(
        num_swa_pages, MAIN_PAGE_SIZE * ROW_BYTES, device=device, dtype=torch.uint8
    )
    swa_slots = torch.cat(
        [
            base + torch.arange(seq, dtype=torch.int64)
            for base, seq in zip(swa_bases, seq_lens)
        ]
    ).to(device)
    _insert_random_packed_rows(swa_cache, swa_slots, MAIN_PAGE_SIZE, cos_sin)

    # Optional compressed (extra) packed cache: seq//ratio rows per request.
    comp_cache = None
    comp_counts: list[int] = []
    comp_bases: list[int] = []
    total_comp_slots = 0
    if compress_ratio is not None:
        comp_counts = [seq // compress_ratio for seq in seq_lens]
        comp_bases, num_comp_pages = _page_bases(comp_counts, comp_page_size)
        total_comp_slots = num_comp_pages * comp_page_size
        comp_cache = torch.zeros(
            num_comp_pages, comp_page_size * ROW_BYTES, device=device, dtype=torch.uint8
        )
        comp_slots = torch.cat(
            [
                base + torch.arange(count, dtype=torch.int64)
                for base, count in zip(comp_bases, comp_counts)
            ]
        ).to(device)
        _insert_random_packed_rows(comp_cache, comp_slots, comp_page_size, cos_sin)

    # Per-token GLOBAL slot indices. FI call: SWA main segment + compressed
    # extra segment. Reference: the same ids translated into the flat
    # dequantized buffer (compressed rows first, then SWA rows) and
    # concatenated compressed-then-SWA into one combined row.
    combined_width = WINDOW + (extra_width if compress_ratio is not None else 0)
    swa_rows: list[list[int]] = []
    swa_len_rows: list[int] = []
    extra_rows: list[list[int]] = []
    extra_len_rows: list[int] = []
    combined_rows: list[list[int]] = []
    combined_len_rows: list[int] = []
    for req, seq in enumerate(seq_lens):
        for pos in range(seq):
            swa_len = min(pos + 1, WINDOW)
            swa_ids = [swa_bases[req] + p for p in range(pos - swa_len + 1, pos + 1)]
            swa_rows.append(swa_ids + [-1] * (WINDOW - swa_len))
            swa_len_rows.append(swa_len)
            extra_ids: list[int] = []
            if compress_ratio is not None:
                # Causal compressed subset in random (topk-like) order.
                count = min((pos + 1) // compress_ratio, comp_counts[req])
                if count > 0:
                    order = torch.randperm(count, generator=cpu_rng).tolist()
                    extra_ids = [comp_bases[req] + j for j in order]
                extra_rows.append(extra_ids + [-1] * (extra_width - len(extra_ids)))
                extra_len_rows.append(len(extra_ids))
            combined = extra_ids + [total_comp_slots + s for s in swa_ids]
            combined_rows.append(combined + [-1] * (combined_width - len(combined)))
            combined_len_rows.append(len(combined))

    # Reference flat KV: production dequant of EVERY slot of both caches (the
    # bytes the FI kernel reads); unwritten page-padding slots are never
    # indexed.
    swa_all = dequantize_deepseek_v4_fp8_ds_mla_cache(
        swa_cache,
        torch.arange(num_swa_pages * MAIN_PAGE_SIZE, device=device),
        MAIN_PAGE_SIZE,
        head_dim=HEAD_DIM,
        rope_dim=ROPE_DIM,
    )
    parts = [swa_all]
    if comp_cache is not None:
        comp_all = dequantize_deepseek_v4_fp8_ds_mla_cache(
            comp_cache,
            torch.arange(total_comp_slots, device=device),
            comp_page_size,
            head_dim=HEAD_DIM,
            rope_dim=ROPE_DIM,
        )
        parts = [comp_all, swa_all]
    kv_flat = torch.cat(parts, dim=0).view(-1, 1, HEAD_DIM)

    num_tokens = sum(seq_lens)
    q = torch.randn(
        num_tokens, NUM_HEADS, HEAD_DIM, device=device, dtype=torch.bfloat16
    )
    attn_sink = (
        torch.randn(NUM_HEADS, device=device, dtype=torch.float32) * 0.5
        if with_sink
        else None
    )

    def _i32(rows: list) -> torch.Tensor:
        return torch.tensor(rows, device=device, dtype=torch.int32)

    return _ParityCase(
        q=q,
        attn_sink=attn_sink,
        swa_cache_2d=swa_cache,
        swa_indices=_i32(swa_rows),
        swa_lens=_i32(swa_len_rows),
        comp_cache_2d=comp_cache,
        comp_page_size=comp_page_size,
        extra_indices=_i32(extra_rows) if compress_ratio is not None else None,
        extra_lens=_i32(extra_len_rows) if compress_ratio is not None else None,
        kv_flat=kv_flat,
        combined_indices=_i32(combined_rows),
        combined_lens=_i32(combined_len_rows),
    )


@unittest.skipUnless(
    _fi_sm120_parity_supported(),
    "requires an sm12x CUDA GPU with FlashInfer's SM120 sparse-MLA runner",
)
class DeepseekV4SparseMlaPrefillFiSm120ParityTest(unittest.TestCase):
    def _build_case(self, **kwargs) -> _ParityCase:
        try:
            return _build_parity_case(device=torch.device("cuda"), **kwargs)
        except RuntimeError as exc:
            if "fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert" in str(exc):
                self.skipTest(str(exc))
            raise

    def _run_fi_sm120(self, case: _ParityCase) -> torch.Tensor:
        extra_cache = (
            _fp8_ds_mla_cache_view(case.comp_cache_2d, case.comp_page_size)
            if case.comp_cache_2d is not None
            else None
        )
        out = sparse_mla_sm120_paged_attention(
            case.q,
            _fp8_ds_mla_cache_view(case.swa_cache_2d, MAIN_PAGE_SIZE),
            case.swa_indices,
            SM_SCALE,
            head_dim_v=HEAD_DIM,
            attn_sink=case.attn_sink,
            extra_k_cache=extra_cache,
            extra_indices=case.extra_indices,
            topk_length=case.swa_lens,
            extra_topk_length=case.extra_lens,
        )
        torch.cuda.synchronize()
        return out

    def _run_reference(self, case: _ParityCase) -> torch.Tensor:
        # Clone: the fallback clamps its index view in place.
        out, _, _ = sparse_mla_prefill_sm12x(
            case.q,
            case.kv_flat,
            case.combined_indices.clone().unsqueeze(1),
            SM_SCALE,
            attn_sink=case.attn_sink,
            topk_length=case.combined_lens,
        )
        torch.cuda.synchronize()
        return out

    def _assert_parity(self, actual: torch.Tensor, expected: torch.Tensor) -> None:
        self.assertEqual(actual.shape, expected.shape)
        self.assertEqual(actual.dtype, torch.bfloat16)
        a = actual.float().reshape(-1, HEAD_DIM)
        e = expected.float().reshape(-1, HEAD_DIM)
        self.assertFalse(bool(torch.isnan(a).any()), "FI output has NaNs")
        self.assertFalse(bool(torch.isnan(e).any()), "reference output has NaNs")
        e_norm = e.norm(dim=-1)
        denom = (a.norm(dim=-1) * e_norm).clamp_min(1.0e-12)
        min_cos = float(((a * e).sum(dim=-1) / denom).min())
        rel = (a - e).norm(dim=-1) / e_norm.clamp_min(1.0e-6)
        max_rel = float(rel.max())
        mean_rel = float(rel.mean())
        self.assertGreater(
            min_cos,
            MIN_ROW_COSINE,
            f"min per-(token,head) cosine {min_cos:.6f} <= {MIN_ROW_COSINE}",
        )
        self.assertLess(
            max_rel,
            MAX_REL_ROW_NORM_ERR,
            f"max relative row-norm error {max_rel:.6f} >= {MAX_REL_ROW_NORM_ERR}",
        )
        self.assertLess(
            mean_rel,
            MAX_MEAN_REL_ROW_NORM_ERR,
            f"mean relative row-norm error {mean_rel:.6f} >= "
            f"{MAX_MEAN_REL_ROW_NORM_ERR}",
        )

    def test_prefill_dual_cache_c4_page64_parity_with_sink(self):
        """432 tokens > 64 -> prefill orchestrator; SWA + C4A extra, page 64."""
        case = self._build_case(
            seq_lens=(200, 232),
            compress_ratio=4,
            comp_page_size=64,
            with_sink=True,
            seed=2026,
        )
        self.assertGreater(case.q.shape[0], 64)
        self._assert_parity(self._run_fi_sm120(case), self._run_reference(case))

    def test_prefill_dual_cache_c4_compressed_page2_parity_with_sink(self):
        """Dual-cache with extra-cache page size 2 (the C128A page layout)."""
        case = self._build_case(
            seq_lens=(200, 232),
            compress_ratio=4,
            comp_page_size=2,
            with_sink=True,
            seed=2027,
        )
        self.assertGreater(case.q.shape[0], 64)
        self._assert_parity(self._run_fi_sm120(case), self._run_reference(case))

    def test_prefill_single_cache_topk128_parity_no_sink(self):
        """Single-cache prefill (topk 128, no extra segment), attn_sink=None."""
        case = self._build_case(
            seq_lens=(200, 232),
            compress_ratio=None,
            with_sink=False,
            seed=2028,
        )
        self.assertGreater(case.q.shape[0], 64)
        self.assertIsNone(case.extra_indices)
        self._assert_parity(self._run_fi_sm120(case), self._run_reference(case))

    def test_decode_seam_dual_cache_parity_with_sink(self):
        """32 tokens <= 64 -> split-K decode kernels through the same surface."""
        case = self._build_case(
            seq_lens=(20, 12),
            compress_ratio=4,
            comp_page_size=64,
            with_sink=True,
            seed=2029,
        )
        self.assertLessEqual(case.q.shape[0], 64)
        self._assert_parity(self._run_fi_sm120(case), self._run_reference(case))


if __name__ == "__main__":
    unittest.main()
