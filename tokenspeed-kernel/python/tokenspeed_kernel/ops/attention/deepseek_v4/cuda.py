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

import torch
from tokenspeed_kernel.platform import CapabilityRequirement
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.thirdparty.cuda.deepseek_v4_attention import (
    compressed_kv_cache_insert_cuda as _compressed_kv_cache_insert_cuda,
)
from tokenspeed_kernel.thirdparty.cuda.deepseek_v4_attention import (
    csa_indexer_cache_insert_fp8_cuda as _csa_indexer_cache_insert_fp8_cuda,
)
from tokenspeed_kernel.thirdparty.cuda.deepseek_v4_attention import (
    decode_indices_cuda as _decode_indices_cuda,
)
from tokenspeed_kernel.thirdparty.cuda.deepseek_v4_attention import (
    full_candidate_topk_cuda as _full_candidate_topk_cuda,
)
from tokenspeed_kernel.thirdparty.cuda.deepseek_v4_attention import (
    inv_rope_grouped_cuda as _inv_rope_grouped_cuda,
)
from tokenspeed_kernel.thirdparty.cuda.deepseek_v4_attention import (
    save_compressor_state_cuda as _save_compressor_state_cuda,
)
from tokenspeed_kernel.thirdparty.cuda.deepseek_v4_attention import (
    sparse_mla_cuda,
    sparse_mla_fp8_cache_cuda,
)


@register_kernel(
    "attention",
    "deepseek_v4_sparse_mla",
    name="deepseek_v4_sparse_mla_cuda",
    solution="cuda",
    capability=CapabilityRequirement(vendors=frozenset({"nvidia"})),
    dtypes={torch.bfloat16, torch.float16},
    priority=Priority.PERFORMANT,
    tags={"cuda", "deepseek_v4"},
)
def deepseek_v4_sparse_mla_cuda(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    sm_scale: float,
    attn_sink: torch.Tensor | None,
    topk_length: torch.Tensor,
) -> torch.Tensor:
    return sparse_mla_cuda(
        q=q,
        kv=kv,
        indices=indices,
        sm_scale=sm_scale,
        attn_sink=attn_sink,
        topk_length=topk_length,
    )


@register_kernel(
    "attention",
    "deepseek_v4_sparse_mla_fp8_cache",
    name="deepseek_v4_sparse_mla_fp8_cache_cuda",
    solution="cuda",
    capability=CapabilityRequirement(vendors=frozenset({"nvidia"})),
    dtypes={torch.bfloat16, torch.float16},
    priority=Priority.PERFORMANT,
    tags={"cuda", "deepseek_v4", "fp8_cache"},
)
def deepseek_v4_sparse_mla_fp8_cache_cuda(
    q: torch.Tensor,
    swa_cache: torch.Tensor,
    swa_indices: torch.Tensor,
    swa_lens: torch.Tensor,
    swa_block_size: int,
    compressed_cache: torch.Tensor | None,
    extra_indices: torch.Tensor | None,
    extra_lens: torch.Tensor | None,
    compressed_block_size: int,
    sm_scale: float,
    attn_sink: torch.Tensor | None,
    online_softmax: bool = False,
) -> torch.Tensor:
    return sparse_mla_fp8_cache_cuda(
        q=q,
        swa_cache=swa_cache,
        swa_indices=swa_indices,
        swa_lens=swa_lens,
        swa_block_size=swa_block_size,
        compressed_cache=compressed_cache,
        extra_indices=extra_indices,
        extra_lens=extra_lens,
        compressed_block_size=compressed_block_size,
        sm_scale=sm_scale,
        attn_sink=attn_sink,
        online_softmax=online_softmax,
    )


@register_kernel(
    "attention",
    "deepseek_v4_csa_indexer_cache_insert_fp8",
    name="csa_indexer_cache_insert_fp8_cuda",
    solution="cuda",
    capability=CapabilityRequirement(vendors=frozenset({"nvidia"})),
    dtypes={torch.float32},
    priority=Priority.PERFORMANT,
    tags={"cuda", "deepseek_v4", "indexer", "fp8_cache"},
)
def csa_indexer_cache_insert_fp8_cuda(
    *,
    state_cache: torch.Tensor,
    token_to_req_indices: torch.Tensor,
    positions: torch.Tensor,
    compressor_slot_mapping: torch.Tensor,
    block_table: torch.Tensor,
    compressor_block_size: int,
    rms_norm_weight: torch.Tensor,
    rms_norm_eps: float,
    cos_sin_cache: torch.Tensor,
    kv_cache_2d: torch.Tensor,
    kv_slot_mapping: torch.Tensor,
    kv_cache_block_size: int,
) -> None:
    _csa_indexer_cache_insert_fp8_cuda(
        state_cache=state_cache,
        token_to_req_indices=token_to_req_indices,
        positions=positions,
        compressor_slot_mapping=compressor_slot_mapping,
        block_table=block_table,
        compressor_block_size=compressor_block_size,
        rms_norm_weight=rms_norm_weight,
        rms_norm_eps=rms_norm_eps,
        cos_sin_cache=cos_sin_cache,
        kv_cache_2d=kv_cache_2d,
        kv_slot_mapping=kv_slot_mapping,
        kv_cache_block_size=kv_cache_block_size,
    )


@register_kernel(
    "attention",
    "deepseek_v4_compressed_kv_cache_insert",
    name="deepseek_v4_compressed_kv_cache_insert_cuda",
    solution="cuda",
    capability=CapabilityRequirement(vendors=frozenset({"nvidia"})),
    dtypes={torch.float32},
    priority=Priority.PERFORMANT,
    tags={"cuda", "deepseek_v4", "compressor", "fp8_cache"},
)
def deepseek_v4_compressed_kv_cache_insert_cuda(
    *,
    state_cache: torch.Tensor,
    token_to_req_indices: torch.Tensor,
    positions: torch.Tensor,
    compressor_slot_mapping: torch.Tensor,
    block_table: torch.Tensor,
    compressor_block_size: int,
    rms_norm_weight: torch.Tensor,
    rms_norm_eps: float,
    cos_sin_cache: torch.Tensor,
    kv_cache_2d: torch.Tensor,
    kv_slot_mapping: torch.Tensor,
    kv_cache_block_size: int,
    compress_ratio: int,
) -> None:
    _compressed_kv_cache_insert_cuda(
        state_cache=state_cache,
        token_to_req_indices=token_to_req_indices,
        positions=positions,
        compressor_slot_mapping=compressor_slot_mapping,
        block_table=block_table,
        rms_norm_weight=rms_norm_weight,
        rms_norm_eps=rms_norm_eps,
        cos_sin_cache=cos_sin_cache,
        kv_cache_2d=kv_cache_2d,
        kv_slot_mapping=kv_slot_mapping,
        kv_cache_block_size=kv_cache_block_size,
        compress_ratio=compress_ratio,
    )


@register_kernel(
    "attention",
    "deepseek_v4_decode_indices",
    name="deepseek_v4_decode_indices_cuda",
    solution="cuda",
    capability=CapabilityRequirement(vendors=frozenset({"nvidia"})),
    dtypes={torch.int32, torch.int64},
    priority=Priority.PERFORMANT,
    tags={"cuda", "deepseek_v4", "decode_indices"},
)
def deepseek_v4_decode_indices_cuda(
    *,
    positions: torch.Tensor,
    token_to_req_indices: torch.Tensor,
    block_table: torch.Tensor,
    topk_indices: torch.Tensor | None,
    window_size: int,
    swa_block_size: int,
    compress_ratio: int,
    compressed_block_size: int,
    full_candidate_max_seq_len: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    return _decode_indices_cuda(
        positions=positions,
        token_to_req_indices=token_to_req_indices,
        block_table=block_table,
        topk_indices=topk_indices,
        window_size=window_size,
        swa_block_size=swa_block_size,
        compress_ratio=compress_ratio,
        compressed_block_size=compressed_block_size,
        full_candidate_max_seq_len=full_candidate_max_seq_len,
    )


@register_kernel(
    "attention",
    "deepseek_v4_full_candidate_topk",
    name="deepseek_v4_full_candidate_topk_cuda",
    solution="cuda",
    capability=CapabilityRequirement(vendors=frozenset({"nvidia"})),
    dtypes={torch.int32, torch.int64},
    priority=Priority.PERFORMANT,
    tags={"cuda", "deepseek_v4", "indexer", "topk"},
)
def deepseek_v4_full_candidate_topk_cuda(
    *,
    positions: torch.Tensor,
    compress_ratio: int,
    max_seq_len: int,
    topk_tokens: int,
) -> torch.Tensor | None:
    return _full_candidate_topk_cuda(
        positions=positions,
        compress_ratio=compress_ratio,
        max_seq_len=max_seq_len,
        topk_tokens=topk_tokens,
    )


@register_kernel(
    "attention",
    "deepseek_v4_save_compressor_state",
    name="deepseek_v4_save_compressor_state_cuda",
    solution="cuda",
    capability=CapabilityRequirement(vendors=frozenset({"nvidia"})),
    dtypes={torch.float32},
    priority=Priority.PERFORMANT,
    tags={"cuda", "deepseek_v4", "compressor"},
)
def deepseek_v4_save_compressor_state_cuda(
    *,
    kv: torch.Tensor,
    score: torch.Tensor,
    ape: torch.Tensor,
    state_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    positions: torch.Tensor,
    block_size: int,
    compress_ratio: int,
) -> None:
    _save_compressor_state_cuda(
        kv=kv,
        score=score,
        ape=ape,
        state_cache=state_cache,
        slot_mapping=slot_mapping,
        positions=positions,
        block_size=block_size,
        compress_ratio=compress_ratio,
    )


@register_kernel(
    "attention",
    "deepseek_v4_inv_rope_grouped",
    name="deepseek_v4_inv_rope_grouped_cuda",
    solution="cuda",
    capability=CapabilityRequirement(vendors=frozenset({"nvidia"})),
    dtypes={torch.bfloat16, torch.float16},
    priority=Priority.PERFORMANT,
    tags={"cuda", "deepseek_v4", "projection"},
)
def deepseek_v4_inv_rope_grouped_cuda(
    *,
    o: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    n_groups: int,
    heads_per_group: int,
    nope_dim: int,
    rope_dim: int,
) -> torch.Tensor:
    return _inv_rope_grouped_cuda(
        o=o,
        positions=positions,
        cos_sin_cache=cos_sin_cache,
        n_groups=n_groups,
        heads_per_group=heads_per_group,
        nope_dim=nope_dim,
        rope_dim=rope_dim,
    )
