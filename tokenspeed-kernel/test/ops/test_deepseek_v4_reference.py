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

from __future__ import annotations

import pytest
import torch
from tokenspeed_kernel.ops.attention.deepseek_v4 import (
    deepseek_v4_sparse_mla_reference,
)

_HEAD_DIM = 512
_NOPE_DIM = 448
_QUANT_BLOCK = 64
_ROPE_BYTES = 128
_TOKEN_STRIDE = _NOPE_DIM + _ROPE_BYTES
_SCALE_DIM = _NOPE_DIM // _QUANT_BLOCK + 1
_FP8_MAX = 448.0


def _cache_row_bytes(rows: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    quant_rows = rows.to(torch.bfloat16).float()
    nope = quant_rows[:, :_NOPE_DIM].reshape(
        -1, _NOPE_DIM // _QUANT_BLOCK, _QUANT_BLOCK
    )
    absmax = nope.abs().amax(dim=-1).clamp_min(1.0e-4)
    exponent = torch.ceil(torch.log2(absmax / _FP8_MAX))
    scaled = torch.clamp(
        nope * torch.pow(2.0, -exponent).unsqueeze(-1),
        -_FP8_MAX,
        _FP8_MAX,
    )
    token_bytes = torch.zeros(
        rows.shape[0], _TOKEN_STRIDE, device=rows.device, dtype=torch.uint8
    )
    token_bytes[:, :_NOPE_DIM] = (
        scaled.to(torch.float8_e4m3fn)
        .view(torch.uint8)
        .reshape(
            rows.shape[0],
            _NOPE_DIM,
        )
    )
    token_bytes[:, _NOPE_DIM:] = (
        quant_rows[:, _NOPE_DIM:].to(torch.bfloat16).contiguous().view(torch.uint8)
    )
    scale_bytes = torch.zeros(
        rows.shape[0], _SCALE_DIM, device=rows.device, dtype=torch.uint8
    )
    scale_bytes[:, : _NOPE_DIM // _QUANT_BLOCK] = torch.clamp(
        exponent.to(torch.int32) + 127, 0, 255
    ).to(torch.uint8)
    return token_bytes, scale_bytes


def _write_cache_rows(
    rows: torch.Tensor,
    slots: torch.Tensor,
    *,
    block_size: int,
    num_blocks: int,
) -> torch.Tensor:
    cache = torch.zeros(
        num_blocks,
        block_size * (_TOKEN_STRIDE + _SCALE_DIM),
        device=rows.device,
        dtype=torch.uint8,
    )
    token_bytes, scale_bytes = _cache_row_bytes(rows)
    flat = cache.reshape(-1)
    slots_i64 = slots.to(torch.int64)
    pages = torch.div(slots_i64, block_size, rounding_mode="floor")
    pos = slots_i64 % block_size
    page_base = pages * cache.stride(0)
    token_base = page_base + pos * _TOKEN_STRIDE
    scale_base = page_base + block_size * _TOKEN_STRIDE + pos * _SCALE_DIM
    token_offsets = (
        token_base[:, None]
        + torch.arange(_TOKEN_STRIDE, device=rows.device, dtype=torch.int64)[None, :]
    )
    scale_offsets = (
        scale_base[:, None]
        + torch.arange(_SCALE_DIM, device=rows.device, dtype=torch.int64)[None, :]
    )
    flat[token_offsets] = token_bytes
    flat[scale_offsets] = scale_bytes
    return cache


def _runtime_cache_view(cache: torch.Tensor, block_size: int) -> torch.Tensor:
    row_bytes = _TOKEN_STRIDE + _SCALE_DIM
    return torch.as_strided(
        cache,
        (cache.shape[0], block_size, 1, row_bytes),
        (cache.stride(0), row_bytes, row_bytes, 1),
    )


def test_sm12x_mhc_pre_split_count_targets_deepseek_decode_shape():
    from tokenspeed_kernel.thirdparty.cuda.deepseek_v4_attention import (
        _sm12x_mhc_pre_split_count,
    )

    assert _sm12x_mhc_pre_split_count(hc_mult=4, hidden_size=128) == 1
    assert _sm12x_mhc_pre_split_count(hc_mult=4, hidden_size=4096) == 16


def _dequant_cache_rows(
    cache: torch.Tensor,
    slots: torch.Tensor,
    *,
    block_size: int,
) -> torch.Tensor:
    slots_i64 = slots.to(torch.int64)
    pages = torch.div(slots_i64, block_size, rounding_mode="floor")
    pos = slots_i64 % block_size
    page_base = pages * cache.stride(0)
    token_base = page_base + pos * _TOKEN_STRIDE
    scale_base = page_base + block_size * _TOKEN_STRIDE + pos * _SCALE_DIM
    flat = cache.reshape(-1)
    token_offsets = (
        token_base[:, None]
        + torch.arange(_TOKEN_STRIDE, device=cache.device, dtype=torch.int64)[None, :]
    )
    scale_offsets = (
        scale_base[:, None]
        + torch.arange(
            _NOPE_DIM // _QUANT_BLOCK, device=cache.device, dtype=torch.int64
        )[None, :]
    )
    row_bytes = flat[token_offsets]
    nope = row_bytes[:, :_NOPE_DIM].contiguous().view(torch.float8_e4m3fn)
    scales = torch.pow(2.0, flat[scale_offsets].to(torch.int32) - 127)
    scales = scales.float().repeat_interleave(_QUANT_BLOCK, dim=1)
    rope = row_bytes[:, _NOPE_DIM:].contiguous().view(torch.bfloat16).float()
    return torch.cat([nope.float() * scales, rope], dim=1).to(torch.bfloat16)


def _workspace_from_caches(
    *,
    swa_cache: torch.Tensor,
    swa_indices: torch.Tensor,
    swa_lens: torch.Tensor,
    swa_block_size: int,
    compressed_cache: torch.Tensor,
    extra_indices: torch.Tensor,
    extra_lens: torch.Tensor,
    compressed_block_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rows = []
    per_token = []
    cursor = 0
    max_width = 0
    for token_idx in range(swa_indices.shape[0]):
        token_rows = []
        extra_count = int(extra_lens[token_idx].item())
        if extra_count:
            slots = extra_indices[token_idx, 0, :extra_count]
            token_rows.append(
                _dequant_cache_rows(
                    compressed_cache,
                    slots,
                    block_size=compressed_block_size,
                )
            )
        swa_count = int(swa_lens[token_idx].item())
        if swa_count:
            slots = swa_indices[token_idx, :swa_count]
            token_rows.append(
                _dequant_cache_rows(swa_cache, slots, block_size=swa_block_size)
            )
        if token_rows:
            joined = torch.cat(token_rows, dim=0)
            rows.append(joined)
            token_indices = torch.arange(
                cursor,
                cursor + joined.shape[0],
                device=swa_indices.device,
                dtype=torch.int32,
            )
            cursor += joined.shape[0]
        else:
            token_indices = torch.empty(0, device=swa_indices.device, dtype=torch.int32)
        per_token.append(token_indices)
        max_width = max(max_width, token_indices.numel())

    indices = torch.full(
        (swa_indices.shape[0], max(max_width, 1)),
        -1,
        device=swa_indices.device,
        dtype=torch.int32,
    )
    lens = torch.zeros(
        swa_indices.shape[0], device=swa_indices.device, dtype=torch.int64
    )
    for token_idx, token_indices in enumerate(per_token):
        if token_indices.numel():
            indices[token_idx, : token_indices.numel()] = token_indices
            lens[token_idx] = token_indices.numel()
    return torch.cat(rows, dim=0), indices, lens


def _slot_from_local(
    block_table: torch.Tensor,
    req_idx: int,
    local: int,
    block_size: int,
) -> int:
    page = int(block_table[req_idx, local // block_size].item())
    return page * block_size + local % block_size


def _expected_decode_indices(
    *,
    positions: torch.Tensor,
    token_to_req_indices: torch.Tensor,
    block_table: torch.Tensor,
    topk_indices: torch.Tensor,
    window_size: int,
    swa_block_size: int,
    compressed_block_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    swa_indices = torch.full(
        (positions.numel(), window_size),
        -1,
        device=positions.device,
        dtype=torch.int32,
    )
    swa_lens = torch.zeros(
        positions.numel(), device=positions.device, dtype=torch.int32
    )
    extra_indices = torch.full(
        (positions.numel(), 1, 64),
        -1,
        device=positions.device,
        dtype=torch.int32,
    )
    extra_lens = torch.zeros(
        positions.numel(), device=positions.device, dtype=torch.int32
    )
    for token_idx, position_tensor in enumerate(positions):
        position = int(position_tensor.item())
        req_idx = int(token_to_req_indices[token_idx].item())
        start = max(0, position - window_size + 1)
        swa_local = list(range(start, position + 1))
        swa_lens[token_idx] = len(swa_local)
        for rank, local in enumerate(swa_local):
            swa_indices[token_idx, rank] = _slot_from_local(
                block_table,
                req_idx,
                local,
                swa_block_size,
            )
        compact = [
            int(value.item())
            for value in topk_indices[token_idx]
            if int(value.item()) >= 0
        ]
        extra_lens[token_idx] = len(compact)
        for rank, local in enumerate(compact):
            extra_indices[token_idx, 0, rank] = _slot_from_local(
                block_table,
                req_idx,
                local,
                compressed_block_size,
            )
    return swa_indices, swa_lens, extra_indices, extra_lens


def _apply_rope_tail(
    rows: torch.Tensor,
    positions: torch.Tensor,
    cos_sin: torch.Tensor,
) -> torch.Tensor:
    out = rows.float().clone()
    cos = cos_sin[positions.long(), : _ROPE_BYTES // 4]
    sin = cos_sin[positions.long(), _ROPE_BYTES // 4 : _ROPE_BYTES // 2]
    even = out[:, _NOPE_DIM::2].clone()
    odd = out[:, _NOPE_DIM + 1 :: 2].clone()
    out[:, _NOPE_DIM::2] = even * cos - odd * sin
    out[:, _NOPE_DIM + 1 :: 2] = even * sin + odd * cos
    return out


def _expected_compressed_kv_rows(
    *,
    state_cache: torch.Tensor,
    token_to_req_indices: torch.Tensor,
    positions: torch.Tensor,
    block_table: torch.Tensor,
    state_block_size: int,
    rms_norm_weight: torch.Tensor,
    rms_norm_eps: float,
    kv_slot_mapping: torch.Tensor,
    compress_ratio: int,
    overlap: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    state_width = state_cache.shape[-1] // 2
    window = (2 if overlap else 1) * compress_ratio
    rows = []
    row_positions = []
    slots = []
    for token_idx, position_tensor in enumerate(positions):
        position = int(position_tensor.item())
        kv_slot = int(kv_slot_mapping[token_idx].item())
        if kv_slot < 0 or (position + 1) % compress_ratio != 0:
            continue
        req_idx = int(token_to_req_indices[token_idx].item())
        kv_rows = []
        score_rows = []
        start = position - window + 1
        for offset in range(window):
            local_pos = start + offset
            if local_pos < 0:
                continue
            table_idx = local_pos // state_block_size
            if table_idx >= block_table.shape[1]:
                continue
            page = int(block_table[req_idx, table_idx].item())
            if page < 0:
                continue
            head_offset = _HEAD_DIM if overlap and offset >= compress_ratio else 0
            state_row = state_cache[page, local_pos % state_block_size]
            kv_rows.append(state_row[head_offset : head_offset + _HEAD_DIM].float())
            score_rows.append(
                state_row[
                    state_width + head_offset : state_width + head_offset + _HEAD_DIM
                ].float()
            )
        if not kv_rows:
            continue
        kv_stack = torch.stack(kv_rows)
        score_stack = torch.stack(score_rows)
        weights = torch.softmax(score_stack, dim=0)
        compressed = torch.sum(kv_stack * weights, dim=0)
        normed = compressed * torch.rsqrt(
            compressed.square().sum() / float(_HEAD_DIM) + rms_norm_eps
        )
        rows.append(normed * rms_norm_weight.float())
        row_positions.append((position // compress_ratio) * compress_ratio)
        slots.append(kv_slot)
    if not rows:
        return (
            torch.empty(0, _HEAD_DIM, device=state_cache.device, dtype=torch.float32),
            torch.empty(0, device=state_cache.device, dtype=torch.int64),
            torch.empty(0, device=state_cache.device, dtype=torch.int64),
        )
    return (
        torch.stack(rows),
        torch.tensor(row_positions, device=state_cache.device, dtype=torch.int64),
        torch.tensor(slots, device=state_cache.device, dtype=torch.int64),
    )


def test_deepseek_v4_sparse_mla_reference_matches_direct_softmax():
    q = torch.tensor(
        [
            [[1.0, 0.0], [0.5, 1.0]],
            [[0.0, 1.0], [1.0, 1.0]],
        ],
        dtype=torch.float32,
    )
    kv = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ],
        dtype=torch.float32,
    )
    indices = torch.tensor([[0, 1, -1], [1, 2, 0]], dtype=torch.int32)
    lengths = torch.tensor([2, 3], dtype=torch.int32)
    sink = torch.tensor([-float("inf"), -float("inf")], dtype=torch.float32)

    out = deepseek_v4_sparse_mla_reference(
        q=q,
        kv=kv,
        indices=indices,
        sm_scale=1.0,
        attn_sink=sink,
        topk_length=lengths,
    )

    expected_rows = []
    for token_idx in range(q.shape[0]):
        token_rows = kv[indices[token_idx, : lengths[token_idx]]]
        scores = torch.einsum("hd,kd->hk", q[token_idx], token_rows)
        probs = torch.softmax(scores, dim=1)
        expected_rows.append(torch.matmul(probs, token_rows))
    expected = torch.stack(expected_rows, dim=0)

    torch.testing.assert_close(out, expected)


def test_deepseek_v4_sparse_mla_reference_applies_attention_sink():
    q = torch.tensor([[[1.0, 0.0]]], dtype=torch.float32)
    kv = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
    indices = torch.tensor([[[0]]], dtype=torch.int32)
    lengths = torch.tensor([1], dtype=torch.int32)
    sink = torch.tensor([0.0], dtype=torch.float32)

    out = deepseek_v4_sparse_mla_reference(
        q=q,
        kv=kv,
        indices=indices,
        sm_scale=1.0,
        attn_sink=sink,
        topk_length=lengths,
    )

    weight = torch.softmax(torch.tensor([1.0, 0.0]), dim=0)[0]
    torch.testing.assert_close(out[0, 0], kv[0] * weight)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU required")
def test_deepseek_v4_decode_indices_cuda_matches_python_slot_mapping():
    from tokenspeed_kernel.ops.attention.deepseek_v4 import (
        deepseek_v4_decode_indices_cuda,
    )

    device = torch.device("cuda")
    positions = torch.tensor([1, 7, 10], device=device, dtype=torch.int64)
    token_to_req_indices = torch.tensor([0, 1, 1], device=device, dtype=torch.int32)
    block_table = torch.tensor(
        [
            [3, 4, 5, 6],
            [11, 12, 13, 14],
        ],
        device=device,
        dtype=torch.int32,
    )
    topk_indices = torch.tensor(
        [
            [0, -1, 1, -1],
            [0, 2, -1, 3],
            [-1, 4, 5, 6],
        ],
        device=device,
        dtype=torch.int32,
    )

    actual = deepseek_v4_decode_indices_cuda(
        positions=positions,
        token_to_req_indices=token_to_req_indices,
        block_table=block_table,
        topk_indices=topk_indices,
        window_size=4,
        swa_block_size=4,
        compress_ratio=4,
        compressed_block_size=4,
    )
    torch.cuda.synchronize()
    expected = _expected_decode_indices(
        positions=positions,
        token_to_req_indices=token_to_req_indices,
        block_table=block_table,
        topk_indices=topk_indices,
        window_size=4,
        swa_block_size=4,
        compressed_block_size=4,
    )

    for actual_tensor, expected_tensor in zip(actual, expected, strict=True):
        assert actual_tensor is not None
        torch.testing.assert_close(actual_tensor, expected_tensor)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU required")
def test_deepseek_v4_decode_indices_cuda_generates_full_candidate_slots():
    from tokenspeed_kernel.ops.attention.deepseek_v4 import (
        deepseek_v4_decode_indices_cuda,
    )

    device = torch.device("cuda")
    positions = torch.tensor([0, 3, 7, 15], device=device, dtype=torch.int64)
    token_to_req_indices = torch.tensor([0, 0, 1, 1], device=device, dtype=torch.int32)
    block_table = torch.tensor(
        [
            [3, 4, 5, 6],
            [11, 12, 13, 14],
        ],
        device=device,
        dtype=torch.int32,
    )
    topk_indices = torch.empty((positions.numel(), 0), device=device, dtype=torch.int32)

    swa_indices, swa_lens, extra_indices, extra_lens = deepseek_v4_decode_indices_cuda(
        positions=positions,
        token_to_req_indices=token_to_req_indices,
        block_table=block_table,
        topk_indices=topk_indices,
        window_size=4,
        swa_block_size=4,
        compress_ratio=4,
        compressed_block_size=4,
        full_candidate_max_seq_len=16,
    )
    torch.cuda.synchronize()

    expected = _expected_decode_indices(
        positions=positions,
        token_to_req_indices=token_to_req_indices,
        block_table=block_table,
        topk_indices=torch.tensor(
            [
                [-1, -1, -1, -1],
                [0, -1, -1, -1],
                [0, 1, -1, -1],
                [0, 1, 2, 3],
            ],
            device=device,
            dtype=torch.int32,
        ),
        window_size=4,
        swa_block_size=4,
        compressed_block_size=4,
    )

    torch.testing.assert_close(swa_indices, expected[0])
    torch.testing.assert_close(swa_lens, expected[1])
    assert extra_indices is not None
    assert extra_lens is not None
    torch.testing.assert_close(extra_indices, expected[2])
    torch.testing.assert_close(extra_lens, expected[3])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU required")
def test_deepseek_v4_decode_indices_cuda_generates_hca_dense_compressed_slots():
    from tokenspeed_kernel.ops.attention.deepseek_v4 import (
        deepseek_v4_decode_indices_cuda,
    )

    device = torch.device("cuda")
    positions = torch.tensor([0, 127, 255, 383], device=device, dtype=torch.int64)
    token_to_req_indices = torch.tensor([0, 0, 1, 1], device=device, dtype=torch.int32)
    block_table = torch.tensor(
        [
            [3, 4, 5, 6],
            [11, 12, 13, 14],
        ],
        device=device,
        dtype=torch.int32,
    )

    swa_indices, swa_lens, extra_indices, extra_lens = deepseek_v4_decode_indices_cuda(
        positions=positions,
        token_to_req_indices=token_to_req_indices,
        block_table=block_table,
        topk_indices=None,
        window_size=4,
        swa_block_size=128,
        compress_ratio=128,
        compressed_block_size=2,
        full_candidate_max_seq_len=384,
    )
    torch.cuda.synchronize()

    expected = _expected_decode_indices(
        positions=positions,
        token_to_req_indices=token_to_req_indices,
        block_table=block_table,
        topk_indices=torch.tensor(
            [
                [-1, -1, -1],
                [0, -1, -1],
                [0, 1, -1],
                [0, 1, 2],
            ],
            device=device,
            dtype=torch.int32,
        ),
        window_size=4,
        swa_block_size=128,
        compressed_block_size=2,
    )

    torch.testing.assert_close(swa_indices, expected[0])
    torch.testing.assert_close(swa_lens, expected[1])
    assert extra_indices is not None
    assert extra_lens is not None
    torch.testing.assert_close(extra_indices, expected[2])
    torch.testing.assert_close(extra_lens, expected[3])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU required")
def test_deepseek_v4_full_candidate_topk_cuda_matches_python():
    from tokenspeed_kernel.ops.attention.deepseek_v4 import (
        deepseek_v4_full_candidate_topk_cuda,
    )

    device = torch.device("cuda")
    positions = torch.tensor([0, 3, 7, 15], device=device, dtype=torch.int64)

    actual = deepseek_v4_full_candidate_topk_cuda(
        positions=positions,
        compress_ratio=4,
        max_seq_len=16,
        topk_tokens=8,
    )
    torch.cuda.synchronize()

    expected = torch.tensor(
        [
            [-1, -1, -1, -1],
            [0, -1, -1, -1],
            [0, 1, -1, -1],
            [0, 1, 2, 3],
        ],
        device=device,
        dtype=torch.int32,
    )
    torch.testing.assert_close(actual, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU required")
def test_deepseek_v4_save_compressor_state_cuda_matches_python():
    from tokenspeed_kernel.ops.attention.deepseek_v4 import (
        deepseek_v4_save_compressor_state_cuda,
    )

    torch.manual_seed(777)
    device = torch.device("cuda")
    compress_ratio = 4
    block_size = 4
    state_width = 6
    kv = torch.randn(3, state_width, device=device, dtype=torch.float32)
    score = torch.randn(3, state_width, device=device, dtype=torch.float32)
    ape = torch.randn(compress_ratio, state_width, device=device, dtype=torch.float32)
    positions = torch.tensor([0, 1, 6], device=device, dtype=torch.int64)
    slots = torch.tensor([0, -1, 5], device=device, dtype=torch.int64)
    actual = torch.zeros(
        2,
        block_size,
        state_width * 2,
        device=device,
        dtype=torch.float32,
    )
    expected = actual.clone()

    deepseek_v4_save_compressor_state_cuda(
        kv=kv,
        score=score,
        ape=ape,
        state_cache=actual,
        slot_mapping=slots,
        positions=positions,
        block_size=block_size,
        compress_ratio=compress_ratio,
    )
    torch.cuda.synchronize()

    ape_slots = ape.view(-1, state_width // 2)
    for token_idx, slot_tensor in enumerate(slots):
        slot = int(slot_tensor.item())
        if slot < 0:
            continue
        row = expected[slot // block_size, slot % block_size]
        row[:state_width] = kv[token_idx]
        slot_idx = int(positions[token_idx].item()) % compress_ratio
        scored = score[token_idx].clone()
        scored[: state_width // 2] += ape_slots[slot_idx]
        scored[state_width // 2 :] += ape_slots[slot_idx + compress_ratio]
        row[state_width:] = scored

    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU required")
def test_deepseek_v4_compressed_kv_cache_insert_cuda_matches_python_csa():
    from tokenspeed_kernel.ops.attention.deepseek_v4 import (
        deepseek_v4_compressed_kv_cache_insert_cuda,
    )

    torch.manual_seed(780)
    device = torch.device("cuda")
    compress_ratio = 4
    state_block_size = 4
    kv_block_size = 4
    state_width = _HEAD_DIM * 2
    state_cache = torch.randn(
        6,
        state_block_size,
        state_width * 2,
        device=device,
        dtype=torch.float32,
    )
    positions = torch.tensor([3, 4, 7, 11], device=device, dtype=torch.int64)
    token_to_req_indices = torch.tensor([0, 0, 1, 1], device=device, dtype=torch.int32)
    compressor_slot_mapping = torch.tensor(
        [3, 4, 7, 11], device=device, dtype=torch.int64
    )
    kv_slot_mapping = torch.tensor([0, 1, 2, -1], device=device, dtype=torch.int64)
    block_table = torch.tensor(
        [
            [0, 1, 2],
            [3, 4, 5],
        ],
        device=device,
        dtype=torch.int32,
    )
    rms_norm_weight = torch.randn(_HEAD_DIM, device=device, dtype=torch.float32)
    rms_norm_eps = 1.0e-6
    cos_sin = (
        torch.randn(16, _ROPE_BYTES // 2, device=device, dtype=torch.float32) * 0.1
    )
    actual_cache = torch.zeros(
        2,
        kv_block_size * (_TOKEN_STRIDE + _SCALE_DIM),
        device=device,
        dtype=torch.uint8,
    )

    # ``compressor_block_size`` is intentionally not passed -- the
    # thirdparty wrapper derives it from ``state_cache.shape[1]`` and
    # the (now-removed) ``cuda.py`` register_kernel wrapper that used
    # to forward it silently dropped it (pre-existing latent bug
    # observed during the 2026-05-13 dead-code cleanup).
    _ = state_block_size
    deepseek_v4_compressed_kv_cache_insert_cuda(
        state_cache=state_cache,
        token_to_req_indices=token_to_req_indices,
        positions=positions,
        compressor_slot_mapping=compressor_slot_mapping,
        block_table=block_table,
        rms_norm_weight=rms_norm_weight,
        rms_norm_eps=rms_norm_eps,
        cos_sin_cache=cos_sin,
        kv_cache_2d=actual_cache,
        kv_slot_mapping=kv_slot_mapping,
        kv_cache_block_size=kv_block_size,
        compress_ratio=compress_ratio,
    )
    torch.cuda.synchronize()

    rows, row_positions, slots = _expected_compressed_kv_rows(
        state_cache=state_cache,
        token_to_req_indices=token_to_req_indices,
        positions=positions,
        block_table=block_table,
        state_block_size=state_block_size,
        rms_norm_weight=rms_norm_weight,
        rms_norm_eps=rms_norm_eps,
        kv_slot_mapping=kv_slot_mapping,
        compress_ratio=compress_ratio,
        overlap=True,
    )
    expected_cache = _write_cache_rows(
        _apply_rope_tail(rows, row_positions, cos_sin),
        slots,
        block_size=kv_block_size,
        num_blocks=actual_cache.shape[0],
    )

    torch.testing.assert_close(
        _dequant_cache_rows(actual_cache, slots, block_size=kv_block_size).float(),
        _dequant_cache_rows(expected_cache, slots, block_size=kv_block_size).float(),
        atol=1.0e-1,
        rtol=1.0e-1,
    )


def test_deepseek_v4_sparse_mla_reference_matches_direct_softmax_with_chunks():
    q = torch.tensor(
        [
            [[1.0, 0.5], [0.0, 1.0]],
            [[0.5, 0.25], [1.0, 0.0]],
            [[0.0, 1.0], [1.0, 1.0]],
        ],
        dtype=torch.float32,
    )
    kv = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [0.5, 1.5],
        ],
        dtype=torch.float32,
    )
    indices = torch.tensor(
        [
            [0, 1, 2, -1],
            [2, 3, -1, -1],
            [1, 0, 3, 2],
        ],
        dtype=torch.int32,
    )
    lengths = torch.tensor([3, 2, 4], dtype=torch.int32)
    sink = torch.tensor([0.0, -float("inf")], dtype=torch.float32)

    out = deepseek_v4_sparse_mla_reference(
        q=q,
        kv=kv,
        indices=indices,
        sm_scale=0.5,
        attn_sink=sink,
        topk_length=lengths,
        query_chunk_size=2,
        topk_chunk_size=2,
    )

    expected_rows = []
    for token_idx in range(q.shape[0]):
        token_rows = kv[indices[token_idx, : lengths[token_idx]]]
        scores = torch.einsum("hd,kd->hk", q[token_idx], token_rows) * 0.5
        scores = torch.cat([scores, sink.reshape(q.shape[1], 1)], dim=1)
        probs = torch.softmax(scores, dim=1)[:, :-1]
        expected_rows.append(torch.matmul(probs, token_rows))
    expected = torch.stack(expected_rows, dim=0)

    torch.testing.assert_close(out, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU required")
def test_deepseek_v4_sparse_mla_cuda_matches_reference():
    from tokenspeed_kernel.ops.attention.deepseek_v4 import (
        deepseek_v4_sparse_mla_cuda,
    )

    torch.manual_seed(123)
    device = torch.device("cuda")
    q = torch.randn(2, 3, 512, device=device, dtype=torch.bfloat16) * 0.1
    kv = torch.randn(11, 512, device=device, dtype=torch.bfloat16) * 0.1
    indices = torch.tensor(
        [
            [0, 2, 5, -1, 8, -1],
            [10, 3, 1, 7, 4, 6],
        ],
        device=device,
        dtype=torch.int32,
    )
    lengths = torch.tensor([5, 6], device=device, dtype=torch.int64)
    sink = torch.tensor([0.0, -float("inf"), -0.5], device=device)

    expected = deepseek_v4_sparse_mla_reference(
        q=q,
        kv=kv,
        indices=indices,
        sm_scale=0.2,
        attn_sink=sink,
        topk_length=lengths,
    )
    actual = deepseek_v4_sparse_mla_cuda(
        q=q,
        kv=kv,
        indices=indices,
        sm_scale=0.2,
        attn_sink=sink,
        topk_length=lengths,
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(
        actual.float(),
        expected.float(),
        atol=2.5e-2,
        rtol=2.5e-2,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU required")
def test_deepseek_v4_sparse_mla_fp8_cache_cuda_matches_workspace_reference():
    from tokenspeed_kernel.ops.attention.deepseek_v4 import (
        deepseek_v4_sparse_mla_fp8_cache_cuda,
    )

    torch.manual_seed(321)
    device = torch.device("cuda")
    q = torch.randn(2, 2, _HEAD_DIM, device=device, dtype=torch.bfloat16) * 0.05
    swa_rows = torch.randn(8, _HEAD_DIM, device=device, dtype=torch.bfloat16) * 0.1
    compressed_rows = (
        torch.randn(6, _HEAD_DIM, device=device, dtype=torch.bfloat16) * 0.1
    )
    swa_slots = torch.tensor([0, 1, 3, 4, 5, 7, 8, 9], device=device, dtype=torch.int32)
    compressed_slots = torch.tensor(
        [0, 2, 3, 5, 6, 7], device=device, dtype=torch.int32
    )
    swa_cache = _write_cache_rows(swa_rows, swa_slots, block_size=4, num_blocks=3)
    compressed_cache = _write_cache_rows(
        compressed_rows,
        compressed_slots,
        block_size=4,
        num_blocks=2,
    )
    swa_indices = torch.tensor(
        [
            [0, 1, 4, -1],
            [5, 7, 8, 9],
        ],
        device=device,
        dtype=torch.int32,
    )
    swa_lens = torch.tensor([3, 4], device=device, dtype=torch.int32)
    extra_indices = torch.tensor(
        [
            [[0, 3, -1]],
            [[2, 5, 7]],
        ],
        device=device,
        dtype=torch.int32,
    )
    extra_lens = torch.tensor([2, 3], device=device, dtype=torch.int32)
    sink = torch.tensor([0.0, -0.25], device=device, dtype=torch.float32)

    kv, workspace_indices, workspace_lens = _workspace_from_caches(
        swa_cache=swa_cache,
        swa_indices=swa_indices,
        swa_lens=swa_lens,
        swa_block_size=4,
        compressed_cache=compressed_cache,
        extra_indices=extra_indices,
        extra_lens=extra_lens,
        compressed_block_size=4,
    )
    expected = deepseek_v4_sparse_mla_reference(
        q=q,
        kv=kv,
        indices=workspace_indices,
        sm_scale=0.17,
        attn_sink=sink,
        topk_length=workspace_lens,
    )
    actual = deepseek_v4_sparse_mla_fp8_cache_cuda(
        q=q,
        swa_cache=swa_cache,
        swa_indices=swa_indices,
        swa_lens=swa_lens,
        swa_block_size=4,
        compressed_cache=compressed_cache,
        extra_indices=extra_indices,
        extra_lens=extra_lens,
        compressed_block_size=4,
        sm_scale=0.17,
        attn_sink=sink,
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(
        actual.float(),
        expected.float(),
        atol=2.5e-2,
        rtol=2.5e-2,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU required")
def test_deepseek_v4_sparse_mla_fp8_cache_cuda_online_softmax_matches_reference():
    from tokenspeed_kernel.ops.attention.deepseek_v4 import (
        deepseek_v4_sparse_mla_fp8_cache_cuda,
    )

    torch.manual_seed(987)
    device = torch.device("cuda")
    q = torch.randn(2, 3, _HEAD_DIM, device=device, dtype=torch.bfloat16) * 0.05
    swa_rows = torch.randn(12, _HEAD_DIM, device=device, dtype=torch.bfloat16) * 0.1
    compressed_rows = (
        torch.randn(10, _HEAD_DIM, device=device, dtype=torch.bfloat16) * 0.1
    )
    swa_slots = torch.arange(12, device=device, dtype=torch.int32)
    compressed_slots = torch.arange(10, device=device, dtype=torch.int32)
    swa_cache = _write_cache_rows(swa_rows, swa_slots, block_size=4, num_blocks=3)
    compressed_cache = _write_cache_rows(
        compressed_rows,
        compressed_slots,
        block_size=4,
        num_blocks=3,
    )
    swa_indices = torch.tensor(
        [
            [0, 1, 2, 4, 8, -1],
            [3, 5, 6, 7, 9, 11],
        ],
        device=device,
        dtype=torch.int32,
    )
    swa_lens = torch.tensor([5, 6], device=device, dtype=torch.int32)
    extra_indices = torch.tensor(
        [
            [[0, 2, 3, 4, 8]],
            [[1, 5, 6, 7, 9]],
        ],
        device=device,
        dtype=torch.int32,
    )
    extra_lens = torch.tensor([5, 5], device=device, dtype=torch.int32)
    sink = torch.tensor([0.0, -0.25, -float("inf")], device=device, dtype=torch.float32)

    kv, workspace_indices, workspace_lens = _workspace_from_caches(
        swa_cache=swa_cache,
        swa_indices=swa_indices,
        swa_lens=swa_lens,
        swa_block_size=4,
        compressed_cache=compressed_cache,
        extra_indices=extra_indices,
        extra_lens=extra_lens,
        compressed_block_size=4,
    )
    expected = deepseek_v4_sparse_mla_reference(
        q=q,
        kv=kv,
        indices=workspace_indices,
        sm_scale=0.17,
        attn_sink=sink,
        topk_length=workspace_lens,
    )
    actual = deepseek_v4_sparse_mla_fp8_cache_cuda(
        q=q,
        swa_cache=swa_cache,
        swa_indices=swa_indices,
        swa_lens=swa_lens,
        swa_block_size=4,
        compressed_cache=compressed_cache,
        extra_indices=extra_indices,
        extra_lens=extra_lens,
        compressed_block_size=4,
        sm_scale=0.17,
        attn_sink=sink,
        online_softmax=True,
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(
        actual.float(),
        expected.float(),
        atol=2.5e-2,
        rtol=2.5e-2,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU required")
@pytest.mark.parametrize("k_split", [2, 4, 8, 16])
def test_deepseek_v4_sparse_mla_fp8_cache_cuda_k_split_matches_default(k_split):
    """K-axis split (T2-α-1) must agree with the single-block kernel.

    The K-split path runs ``k_split`` blocks per (token, head) on contiguous
    chunks of the candidate axis and merges via a small reduce kernel. The
    final output must be identical (modulo fp accumulation order noise) to
    the single-block ``k_split=1`` path.

    ``k_split=16`` with this small ``total_len`` also exercises the empty
    chunk path (``chunk_start == chunk_end``); the reduce kernel must
    ignore those slots via the ``d_k > 0`` guard.
    """
    from tokenspeed_kernel.ops.attention.deepseek_v4 import (
        deepseek_v4_sparse_mla_fp8_cache_cuda,
    )

    torch.manual_seed(20260513 + k_split)
    device = torch.device("cuda")
    q = torch.randn(2, 3, _HEAD_DIM, device=device, dtype=torch.bfloat16) * 0.05
    swa_rows = torch.randn(12, _HEAD_DIM, device=device, dtype=torch.bfloat16) * 0.1
    compressed_rows = (
        torch.randn(10, _HEAD_DIM, device=device, dtype=torch.bfloat16) * 0.1
    )
    swa_slots = torch.arange(12, device=device, dtype=torch.int32)
    compressed_slots = torch.arange(10, device=device, dtype=torch.int32)
    swa_cache = _write_cache_rows(swa_rows, swa_slots, block_size=4, num_blocks=3)
    compressed_cache = _write_cache_rows(
        compressed_rows, compressed_slots, block_size=4, num_blocks=3
    )
    swa_indices = torch.tensor(
        [
            [0, 1, 2, 4, 8, -1],
            [3, 5, 6, 7, 9, 11],
        ],
        device=device,
        dtype=torch.int32,
    )
    swa_lens = torch.tensor([5, 6], device=device, dtype=torch.int32)
    extra_indices = torch.tensor(
        [
            [[0, 2, 3, 4, 8]],
            [[1, 5, 6, 7, 9]],
        ],
        device=device,
        dtype=torch.int32,
    )
    extra_lens = torch.tensor([5, 5], device=device, dtype=torch.int32)
    sink = torch.tensor([0.0, -0.25, -float("inf")], device=device, dtype=torch.float32)

    common_kwargs = dict(
        q=q,
        swa_cache=swa_cache,
        swa_indices=swa_indices,
        swa_lens=swa_lens,
        swa_block_size=4,
        compressed_cache=compressed_cache,
        extra_indices=extra_indices,
        extra_lens=extra_lens,
        compressed_block_size=4,
        sm_scale=0.17,
        attn_sink=sink,
        online_softmax=True,
    )
    baseline = deepseek_v4_sparse_mla_fp8_cache_cuda(**common_kwargs)
    split = deepseek_v4_sparse_mla_fp8_cache_cuda(**common_kwargs, k_split=k_split)
    torch.cuda.synchronize()
    torch.testing.assert_close(split.float(), baseline.float(), atol=5e-3, rtol=5e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU required")
@pytest.mark.parametrize("k_split", [2, 8])
def test_deepseek_v4_sparse_mla_fp8_cache_cuda_k_split_swa_only(k_split):
    """K-split must also work when ``extra_indices`` is ``None`` (SWA-only).

    Some decode layers run without the extra/CSA branch -- the wrapper
    replaces ``extra_indices`` with an empty tensor and the kernel sees
    ``extra_width == 0`` and ``extra_len[token] == 0``. Each chunk then
    walks only the SWA range, but the chunk arithmetic still applies.
    The K-split output must match the single-block SWA-only path.
    """
    from tokenspeed_kernel.ops.attention.deepseek_v4 import (
        deepseek_v4_sparse_mla_fp8_cache_cuda,
    )

    torch.manual_seed(70000 + k_split)
    device = torch.device("cuda")
    q = torch.randn(2, 4, _HEAD_DIM, device=device, dtype=torch.bfloat16) * 0.05
    swa_rows = torch.randn(16, _HEAD_DIM, device=device, dtype=torch.bfloat16) * 0.1
    swa_slots = torch.arange(16, device=device, dtype=torch.int32)
    swa_cache = _write_cache_rows(swa_rows, swa_slots, block_size=4, num_blocks=4)
    swa_indices = torch.tensor(
        [
            [0, 1, 2, 3, 4, 5, 6, 7],
            [8, 9, 10, 11, 12, 13, 14, 15],
        ],
        device=device,
        dtype=torch.int32,
    )
    swa_lens = torch.tensor([7, 8], device=device, dtype=torch.int32)
    sink = torch.tensor(
        [0.0, -0.25, -float("inf"), -0.5], device=device, dtype=torch.float32
    )

    common_kwargs = dict(
        q=q,
        swa_cache=swa_cache,
        swa_indices=swa_indices,
        swa_lens=swa_lens,
        swa_block_size=4,
        compressed_cache=None,
        extra_indices=None,
        extra_lens=None,
        compressed_block_size=4,
        sm_scale=0.17,
        attn_sink=sink,
        online_softmax=True,
    )
    baseline = deepseek_v4_sparse_mla_fp8_cache_cuda(**common_kwargs)
    split = deepseek_v4_sparse_mla_fp8_cache_cuda(**common_kwargs, k_split=k_split)
    torch.cuda.synchronize()
    torch.testing.assert_close(split.float(), baseline.float(), atol=5e-3, rtol=5e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU required")
def test_deepseek_v4_sparse_mla_fp8_cache_cuda_k_split_long_context():
    """K-split must produce correct chunks at realistic decode shape.

    Plan-doc K-split bench targets ``swa=128`` + ``extra<=8192`` for
    DSv4-Flash CSA decode. The small unit tests above only have
    ``total_len <= 11``, so chunks are 1-2 candidates wide and don't
    exercise meaningful chunk arithmetic. This test scales up to
    ``swa=64`` + ``extra=192`` (``total_len=256``) so ``k_split=8``
    chunks are ~32 candidates each, validating that the contiguous
    chunk boundary calc holds at scale and that all chunks contribute
    proportionally to the merged output.
    """
    from tokenspeed_kernel.ops.attention.deepseek_v4 import (
        deepseek_v4_sparse_mla_fp8_cache_cuda,
    )

    torch.manual_seed(20260514)
    device = torch.device("cuda")
    num_tokens = 2
    num_heads = 4
    swa_width = 64
    extra_width = 192
    swa_block_size = 16
    compressed_block_size = 16
    swa_total_rows = 128
    compressed_total_rows = 384

    q = (
        torch.randn(
            num_tokens, num_heads, _HEAD_DIM, device=device, dtype=torch.bfloat16
        )
        * 0.05
    )
    swa_rows = (
        torch.randn(swa_total_rows, _HEAD_DIM, device=device, dtype=torch.bfloat16)
        * 0.1
    )
    compressed_rows = (
        torch.randn(
            compressed_total_rows, _HEAD_DIM, device=device, dtype=torch.bfloat16
        )
        * 0.1
    )
    swa_slots = torch.arange(swa_total_rows, device=device, dtype=torch.int32)
    compressed_slots = torch.arange(
        compressed_total_rows, device=device, dtype=torch.int32
    )
    swa_cache = _write_cache_rows(
        swa_rows,
        swa_slots,
        block_size=swa_block_size,
        num_blocks=swa_total_rows // swa_block_size,
    )
    compressed_cache = _write_cache_rows(
        compressed_rows,
        compressed_slots,
        block_size=compressed_block_size,
        num_blocks=compressed_total_rows // compressed_block_size,
    )

    # Variable per-token lengths so chunks have to handle uneven splits.
    swa_lens_py = [64, 48]
    extra_lens_py = [192, 144]
    swa_indices = torch.full(
        (num_tokens, swa_width), -1, device=device, dtype=torch.int32
    )
    extra_indices = torch.full(
        (num_tokens, 1, extra_width), -1, device=device, dtype=torch.int32
    )
    rng = torch.Generator(device=device).manual_seed(123)
    for t in range(num_tokens):
        sw_n = swa_lens_py[t]
        ex_n = extra_lens_py[t]
        swa_perm = torch.randperm(swa_total_rows, device=device, generator=rng)[:sw_n]
        ex_perm = torch.randperm(compressed_total_rows, device=device, generator=rng)[
            :ex_n
        ]
        swa_indices[t, :sw_n] = swa_perm.to(torch.int32)
        extra_indices[t, 0, :ex_n] = ex_perm.to(torch.int32)
    swa_lens = torch.tensor(swa_lens_py, device=device, dtype=torch.int32)
    extra_lens = torch.tensor(extra_lens_py, device=device, dtype=torch.int32)
    sink = torch.tensor(
        [0.0, -0.25, -float("inf"), -0.5], device=device, dtype=torch.float32
    )

    common_kwargs = dict(
        q=q,
        swa_cache=swa_cache,
        swa_indices=swa_indices,
        swa_lens=swa_lens,
        swa_block_size=swa_block_size,
        compressed_cache=compressed_cache,
        extra_indices=extra_indices,
        extra_lens=extra_lens,
        compressed_block_size=compressed_block_size,
        sm_scale=0.0625,
        attn_sink=sink,
        online_softmax=True,
    )
    baseline = deepseek_v4_sparse_mla_fp8_cache_cuda(**common_kwargs)
    for k_split in (2, 4, 8):
        split = deepseek_v4_sparse_mla_fp8_cache_cuda(**common_kwargs, k_split=k_split)
        torch.cuda.synchronize()
        try:
            torch.testing.assert_close(
                split.float(), baseline.float(), atol=5e-3, rtol=5e-3
            )
        except AssertionError as exc:
            raise AssertionError(
                f"k_split={k_split} differs from k_split=1: {exc}"
            ) from exc


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU required")
def test_deepseek_v4_sparse_mla_fp8_cache_cuda_k_split_ignores_online_softmax_flag():
    """The K-split path always uses the online-softmax formulation.

    ``launch_sparse_mla_fp8_cache`` routes ``k_split > 1`` to the
    K-split + reduce kernels regardless of the ``online_softmax`` flag.
    The single-block path's ``online_softmax`` switch picks between
    the legacy two-pass kernel and the streaming kernel. With K-split
    on, the flag is intentionally ignored; assert that explicitly so a
    future refactor does not silently re-introduce a flag-sensitive
    dispatch.
    """
    from tokenspeed_kernel.ops.attention.deepseek_v4 import (
        deepseek_v4_sparse_mla_fp8_cache_cuda,
    )

    torch.manual_seed(20260515)
    device = torch.device("cuda")
    q = torch.randn(2, 3, _HEAD_DIM, device=device, dtype=torch.bfloat16) * 0.05
    swa_rows = torch.randn(12, _HEAD_DIM, device=device, dtype=torch.bfloat16) * 0.1
    compressed_rows = (
        torch.randn(10, _HEAD_DIM, device=device, dtype=torch.bfloat16) * 0.1
    )
    swa_slots = torch.arange(12, device=device, dtype=torch.int32)
    compressed_slots = torch.arange(10, device=device, dtype=torch.int32)
    swa_cache = _write_cache_rows(swa_rows, swa_slots, block_size=4, num_blocks=3)
    compressed_cache = _write_cache_rows(
        compressed_rows, compressed_slots, block_size=4, num_blocks=3
    )
    swa_indices = torch.tensor(
        [
            [0, 1, 2, 4, 8, -1],
            [3, 5, 6, 7, 9, 11],
        ],
        device=device,
        dtype=torch.int32,
    )
    swa_lens = torch.tensor([5, 6], device=device, dtype=torch.int32)
    extra_indices = torch.tensor(
        [
            [[0, 2, 3, 4, 8]],
            [[1, 5, 6, 7, 9]],
        ],
        device=device,
        dtype=torch.int32,
    )
    extra_lens = torch.tensor([5, 5], device=device, dtype=torch.int32)
    sink = torch.tensor([0.0, -0.25, -float("inf")], device=device, dtype=torch.float32)

    common_kwargs = dict(
        q=q,
        swa_cache=swa_cache,
        swa_indices=swa_indices,
        swa_lens=swa_lens,
        swa_block_size=4,
        compressed_cache=compressed_cache,
        extra_indices=extra_indices,
        extra_lens=extra_lens,
        compressed_block_size=4,
        sm_scale=0.17,
        attn_sink=sink,
    )
    out_with = deepseek_v4_sparse_mla_fp8_cache_cuda(
        **common_kwargs, online_softmax=True, k_split=8
    )
    out_without = deepseek_v4_sparse_mla_fp8_cache_cuda(
        **common_kwargs, online_softmax=False, k_split=8
    )
    torch.cuda.synchronize()
    # K-split is always online-softmax internally; the flag must be ignored,
    # producing bit-identical output (we still allow a tiny epsilon for any
    # future scheduler-induced reordering).
    torch.testing.assert_close(
        out_with.float(), out_without.float(), atol=1e-6, rtol=1e-6
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU required")
def test_deepseek_v4_sparse_mla_fp8_cache_cuda_accepts_runtime_cache_view():
    from tokenspeed_kernel.ops.attention.deepseek_v4 import (
        deepseek_v4_sparse_mla_fp8_cache_cuda,
    )

    torch.manual_seed(988)
    device = torch.device("cuda")
    q = torch.randn(1, 2, _HEAD_DIM, device=device, dtype=torch.bfloat16) * 0.05
    swa_rows = torch.randn(4, _HEAD_DIM, device=device, dtype=torch.bfloat16) * 0.1
    compressed_rows = (
        torch.randn(4, _HEAD_DIM, device=device, dtype=torch.bfloat16) * 0.1
    )
    slots = torch.arange(4, device=device, dtype=torch.int32)
    swa_cache = _write_cache_rows(swa_rows, slots, block_size=4, num_blocks=1)
    compressed_cache = _write_cache_rows(
        compressed_rows,
        slots,
        block_size=4,
        num_blocks=1,
    )
    swa_indices = torch.tensor([[0, 1, 3]], device=device, dtype=torch.int32)
    swa_lens = torch.tensor([3], device=device, dtype=torch.int32)
    extra_indices = torch.tensor([[[0, 2, 3]]], device=device, dtype=torch.int32)
    extra_lens = torch.tensor([3], device=device, dtype=torch.int32)
    sink = torch.tensor([0.0, -0.25], device=device, dtype=torch.float32)

    kv, workspace_indices, workspace_lens = _workspace_from_caches(
        swa_cache=swa_cache,
        swa_indices=swa_indices,
        swa_lens=swa_lens,
        swa_block_size=4,
        compressed_cache=compressed_cache,
        extra_indices=extra_indices,
        extra_lens=extra_lens,
        compressed_block_size=4,
    )
    expected = deepseek_v4_sparse_mla_reference(
        q=q,
        kv=kv,
        indices=workspace_indices,
        sm_scale=0.17,
        attn_sink=sink,
        topk_length=workspace_lens,
    )
    actual = deepseek_v4_sparse_mla_fp8_cache_cuda(
        q=q,
        swa_cache=_runtime_cache_view(swa_cache, block_size=4),
        swa_indices=swa_indices,
        swa_lens=swa_lens,
        swa_block_size=4,
        compressed_cache=_runtime_cache_view(compressed_cache, block_size=4),
        extra_indices=extra_indices,
        extra_lens=extra_lens,
        compressed_block_size=4,
        sm_scale=0.17,
        attn_sink=sink,
        online_softmax=True,
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(
        actual.float(),
        expected.float(),
        atol=2.5e-2,
        rtol=2.5e-2,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU required")
def test_deepseek_v4_sparse_mla_fp8_cache_cuda_supports_swa_only():
    from tokenspeed_kernel.ops.attention.deepseek_v4 import (
        deepseek_v4_sparse_mla_fp8_cache_cuda,
    )

    torch.manual_seed(654)
    device = torch.device("cuda")
    q = torch.randn(2, 2, _HEAD_DIM, device=device, dtype=torch.bfloat16) * 0.05
    swa_rows = torch.randn(4, _HEAD_DIM, device=device, dtype=torch.bfloat16) * 0.1
    swa_slots = torch.tensor([0, 1, 2, 3], device=device, dtype=torch.int32)
    swa_cache = _write_cache_rows(swa_rows, swa_slots, block_size=4, num_blocks=1)
    swa_indices = torch.tensor(
        [
            [0, 2, 3],
            [1, 3, -1],
        ],
        device=device,
        dtype=torch.int32,
    )
    swa_lens = torch.tensor([3, 2], device=device, dtype=torch.int32)
    sink = torch.tensor([0.0, -0.25], device=device, dtype=torch.float32)

    kv = torch.cat(
        [
            _dequant_cache_rows(swa_cache, swa_indices[0], block_size=4),
            _dequant_cache_rows(swa_cache, swa_indices[1, :2], block_size=4),
        ],
        dim=0,
    )
    expected = deepseek_v4_sparse_mla_reference(
        q=q,
        kv=kv,
        indices=torch.tensor(
            [
                [0, 1, 2],
                [3, 4, -1],
            ],
            device=device,
            dtype=torch.int32,
        ),
        sm_scale=0.17,
        attn_sink=sink,
        topk_length=torch.tensor([3, 2], device=device, dtype=torch.int64),
    )
    actual = deepseek_v4_sparse_mla_fp8_cache_cuda(
        q=q,
        swa_cache=swa_cache,
        swa_indices=swa_indices,
        swa_lens=swa_lens,
        swa_block_size=4,
        compressed_cache=None,
        extra_indices=None,
        extra_lens=None,
        compressed_block_size=4,
        sm_scale=0.17,
        attn_sink=sink,
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(
        actual.float(),
        expected.float(),
        atol=2.5e-2,
        rtol=2.5e-2,
    )
