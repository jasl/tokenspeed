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

from types import SimpleNamespace

import torch

from tokenspeed.runtime.layers.attention.backends import deepseek_v4 as backend_module


class _Mode:
    def __init__(self, *, decode: bool = False, extend: bool = False) -> None:
        self._decode = decode
        self._extend = extend

    def is_decode(self) -> bool:
        return self._decode

    def is_extend(self) -> bool:
        return self._extend


class _Config:
    device = "cpu"
    dtype = torch.bfloat16
    head_dim = 4
    num_attention_heads = 1
    num_kv_heads = 1
    attn_tp_size = 1
    page_size = 4
    context_len = 16


class _Pool:
    swa_block_size = 4

    def get_compressed_block_size(self, layer_id: int) -> int:
        del layer_id
        return 4

    def get_swa_kv_buffer(self, layer_id: int) -> torch.Tensor:
        del layer_id
        return torch.empty((1, 4), dtype=torch.uint8)

    def get_compressed_kv_buffer_2d(self, layer_id: int) -> torch.Tensor:
        del layer_id
        return torch.empty((1, 4), dtype=torch.uint8)


def _metadata(mode: _Mode):
    return SimpleNamespace(forward_mode=mode)


def test_sm12x_decode_uses_fallback_before_flashmla(monkeypatch):
    monkeypatch.setattr(
        backend_module,
        "current_platform",
        lambda: SimpleNamespace(is_sm12x=True),
    )
    backend = backend_module.DeepseekV4AttentionBackend(_Config())
    backend.forward_metadata = _metadata(_Mode(decode=True))

    q = torch.ones((2, 1, 4), dtype=torch.bfloat16)
    expected = torch.full_like(q, 7.0)
    calls = []

    monkeypatch.setattr(
        backend,
        "_get_decode_swa_metadata",
        lambda *args, **kwargs: (
            torch.zeros((2, 1), dtype=torch.int32),
            torch.ones((2,), dtype=torch.int32),
        ),
    )
    monkeypatch.setattr(
        backend,
        "_decode_compressed_indices_and_lens",
        lambda *args, **kwargs: (None, None),
    )
    monkeypatch.setattr(
        backend,
        "_fp8_ds_mla_cache_view",
        lambda cache, block_size: cache,
    )

    def fake_cache_fallback(**kwargs):
        calls.append(kwargs)
        return expected

    monkeypatch.setattr(
        backend, "_forward_sparse_mla_fp8_cache_cuda", fake_cache_fallback
    )

    actual = backend.forward_deepseek_v4_decode(
        q=q,
        positions=torch.tensor([3, 5], dtype=torch.int64),
        token_to_kv_pool=_Pool(),
        layer_id=0,
        kind="hca",
        compress_ratio=1,
        num_local_heads=1,
        padded_heads=1,
        head_dim=4,
        window_size=4,
        softmax_scale=1.0,
        attn_sink=torch.zeros(1),
        topk_indices=None,
    )

    torch.testing.assert_close(actual, expected)
    assert len(calls) == 1


def test_sm12x_enables_cuda_sparse_mla_cache_by_default(monkeypatch):
    monkeypatch.delenv("TOKENSPEED_DEEPSEEK_V4_CUDA_SPARSE_MLA_CACHE", raising=False)
    monkeypatch.delenv(
        "TOKENSPEED_DEEPSEEK_V4_CUDA_SPARSE_MLA_ONLINE_SOFTMAX",
        raising=False,
    )
    monkeypatch.setattr(
        backend_module,
        "current_platform",
        lambda: SimpleNamespace(is_sm12x=True),
    )
    backend = backend_module.DeepseekV4AttentionBackend(_Config())

    assert backend._use_sparse_mla_fp8_cache_cuda()
    assert backend._use_sparse_mla_online_softmax()


def test_sparse_mla_cache_env_override_disables_sm12x_default(monkeypatch):
    monkeypatch.setenv("TOKENSPEED_DEEPSEEK_V4_CUDA_SPARSE_MLA_CACHE", "0")
    monkeypatch.setenv("TOKENSPEED_DEEPSEEK_V4_CUDA_SPARSE_MLA_ONLINE_SOFTMAX", "0")
    monkeypatch.setattr(
        backend_module,
        "current_platform",
        lambda: SimpleNamespace(is_sm12x=True),
    )
    backend = backend_module.DeepseekV4AttentionBackend(_Config())

    assert not backend._use_sparse_mla_fp8_cache_cuda()
    assert not backend._use_sparse_mla_online_softmax()


def test_non_sm12x_keeps_cuda_sparse_mla_cache_opt_in(monkeypatch):
    monkeypatch.delenv("TOKENSPEED_DEEPSEEK_V4_CUDA_SPARSE_MLA_CACHE", raising=False)
    monkeypatch.delenv(
        "TOKENSPEED_DEEPSEEK_V4_CUDA_SPARSE_MLA_ONLINE_SOFTMAX",
        raising=False,
    )
    monkeypatch.setattr(
        backend_module,
        "current_platform",
        lambda: SimpleNamespace(is_sm12x=False),
    )
    backend = backend_module.DeepseekV4AttentionBackend(_Config())

    assert not backend._use_sparse_mla_fp8_cache_cuda()
    assert not backend._use_sparse_mla_online_softmax()

    monkeypatch.setenv("TOKENSPEED_DEEPSEEK_V4_CUDA_SPARSE_MLA_CACHE", "1")
    monkeypatch.setenv("TOKENSPEED_DEEPSEEK_V4_CUDA_SPARSE_MLA_ONLINE_SOFTMAX", "1")

    assert backend._use_sparse_mla_fp8_cache_cuda()
    assert backend._use_sparse_mla_online_softmax()


def test_sm12x_prefill_uses_sparse_reference_before_flashmla(monkeypatch):
    monkeypatch.setattr(
        backend_module,
        "current_platform",
        lambda: SimpleNamespace(is_sm12x=True),
    )
    backend = backend_module.DeepseekV4AttentionBackend(_Config())
    backend.forward_metadata = _metadata(_Mode(extend=True))

    q = torch.ones((2, 1, 4), dtype=torch.bfloat16)
    expected = torch.full_like(q, 3.0)
    calls = []

    monkeypatch.setattr(
        backend,
        "_prefill_workspace",
        lambda *args, **kwargs: (
            torch.ones((1, 2, 4), dtype=torch.bfloat16),
            torch.zeros((2, 1), dtype=torch.int32),
            torch.ones((2,), dtype=torch.int32),
        ),
    )

    def fake_sparse_reference(**kwargs):
        calls.append(kwargs)
        return expected

    monkeypatch.setattr(backend, "_forward_sparse_mla_reference", fake_sparse_reference)

    actual = backend.forward_deepseek_v4_prefill(
        q=q,
        positions=torch.tensor([0, 1], dtype=torch.int64),
        token_to_kv_pool=_Pool(),
        layer_id=0,
        kind="hca",
        compress_ratio=1,
        num_local_heads=1,
        padded_heads=1,
        head_dim=4,
        window_size=4,
        softmax_scale=1.0,
        attn_sink=torch.zeros(1),
        topk_indices=None,
    )

    torch.testing.assert_close(actual, expected)
    assert len(calls) == 1
