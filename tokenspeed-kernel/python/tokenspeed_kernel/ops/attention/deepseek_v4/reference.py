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


def _flatten_sparse_mla_kv(kv: torch.Tensor) -> torch.Tensor:
    if kv.dim() == 2:
        return kv
    if kv.dim() == 3:
        return kv.reshape(-1, kv.shape[-1])
    if kv.dim() == 4 and kv.shape[-2] == 1:
        return kv.reshape(-1, kv.shape[-1])
    raise ValueError(f"kv must be [N, D], [B, N, D], or [N, 1, D], got {kv.shape}")


def _flatten_sparse_mla_indices(indices: torch.Tensor) -> torch.Tensor:
    if indices.dim() == 2:
        return indices
    if indices.dim() == 3 and indices.shape[1] == 1:
        return indices[:, 0, :]
    raise ValueError(f"indices must be [T, K] or [T, 1, K], got {indices.shape}")


def _resolve_chunk_size(
    chunk_size: int | None,
    *,
    limit: int,
    default: int,
    name: str,
) -> int:
    if limit <= 0:
        return 1
    if chunk_size is None:
        chunk_size = default
    chunk_size = int(chunk_size)
    if chunk_size <= 0:
        raise ValueError(f"{name} must be positive, got {chunk_size}")
    return min(chunk_size, limit)


def _new_attention_state(
    q_chunk: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    q_bhd = q_chunk.to(torch.float32)
    num_tokens, num_heads, head_dim = q_bhd.shape
    max_score = torch.full(
        (num_tokens, num_heads),
        float("-inf"),
        dtype=torch.float32,
        device=q_chunk.device,
    )
    denom = torch.zeros_like(max_score)
    acc = torch.zeros(
        (num_tokens, num_heads, head_dim),
        dtype=torch.float32,
        device=q_chunk.device,
    )
    return q_bhd, max_score, denom, acc


def _accumulate_attention_chunk(
    *,
    q_bhd: torch.Tensor,
    kv_btd: torch.Tensor,
    valid_tokens: torch.Tensor,
    max_score: torch.Tensor,
    denom: torch.Tensor,
    acc: torch.Tensor,
    sm_scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    zero = torch.zeros((), dtype=torch.float32, device=q_bhd.device)
    kv_btd = kv_btd.to(torch.float32)
    kv_btd = torch.where(valid_tokens[:, :, None], kv_btd, zero)
    scores = torch.einsum("bhd,btd->bht", q_bhd, kv_btd) * sm_scale
    scores = scores.masked_fill(~valid_tokens[:, None, :], float("-inf"))

    chunk_max = scores.amax(dim=-1)
    next_max = torch.maximum(max_score, chunk_max)
    previous_scale = torch.exp(max_score - next_max)
    previous_scale = torch.nan_to_num(previous_scale)

    weights = torch.exp(scores - next_max[:, :, None])
    weights = torch.where(valid_tokens[:, None, :], weights, zero)
    weights = torch.nan_to_num(weights)

    acc = acc * previous_scale[:, :, None]
    denom = denom * previous_scale
    acc = acc + torch.einsum("bht,btd->bhd", weights, kv_btd)
    denom = denom + weights.sum(dim=-1)
    return next_max, denom, acc


def _finish_attention_no_sink(
    max_score: torch.Tensor,
    denom: torch.Tensor,
    acc: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    valid = denom > 0
    safe_denom = torch.where(valid, denom, torch.ones_like(denom))
    output = acc / safe_denom[:, :, None]
    output = torch.where(
        valid[:, :, None],
        output,
        torch.zeros((), dtype=output.dtype, device=output.device),
    )
    lse = torch.where(
        valid,
        max_score + torch.log(safe_denom),
        torch.full_like(max_score, float("-inf")),
    )
    return output, lse


def _write_attention_output(
    *,
    output_no_sink: torch.Tensor,
    lse_no_sink: torch.Tensor,
    attn_sink: torch.Tensor | None,
    output: torch.Tensor,
) -> None:
    if attn_sink is None:
        output.copy_(output_no_sink.to(output.dtype))
        return

    sink = attn_sink[: output.shape[1]].to(device=output.device, dtype=torch.float32)
    sink = sink.reshape(1, output.shape[1])
    merge_max = torch.maximum(sink, lse_no_sink)
    safe_merge_max = torch.where(
        torch.isfinite(merge_max),
        merge_max,
        torch.zeros_like(merge_max),
    )
    sink_weight = torch.exp(sink - safe_merge_max)
    sink_weight = torch.nan_to_num(sink_weight)
    subset_weight = torch.exp(lse_no_sink - safe_merge_max)
    subset_weight = torch.nan_to_num(subset_weight)
    merged_denom = sink_weight + subset_weight
    safe_denom = torch.where(
        merged_denom > 0,
        merged_denom,
        torch.ones_like(merged_denom),
    )
    merged_output = output_no_sink * subset_weight[:, :, None]
    merged_output = merged_output / safe_denom[:, :, None]
    merged_output = torch.where(
        (merged_denom > 0)[:, :, None],
        merged_output,
        torch.zeros((), dtype=merged_output.dtype, device=merged_output.device),
    )
    output.copy_(merged_output.to(output.dtype))


@register_kernel(
    "attention",
    "deepseek_v4_sparse_mla",
    name="deepseek_v4_sparse_mla_reference",
    solution="reference",
    capability=CapabilityRequirement(vendors=frozenset({"nvidia", "amd"})),
    dtypes={torch.bfloat16, torch.float16, torch.float32},
    priority=Priority.REFERENCE,
    tags={"reference", "deepseek_v4"},
)
def deepseek_v4_sparse_mla_reference(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    sm_scale: float,
    attn_sink: torch.Tensor | None,
    topk_length: torch.Tensor,
    *,
    query_chunk_size: int | None = None,
    topk_chunk_size: int | None = None,
) -> torch.Tensor:
    """Reference DeepSeek V4 sparse MLA attention.

    This mirrors the FlashMLA sparse-attention contract used by TokenSpeed's
    DeepSeek V4 path: `q` is `[tokens, heads, dim]`, `kv` is flattened cache rows
    or a flattenable workspace, `indices` selects candidate rows for each token,
    and `topk_length` gives the valid candidate count per token. `attn_sink`
    contributes one extra softmax logit with zero value.
    """

    if q.dim() != 3:
        raise ValueError(f"q must be [tokens, heads, dim], got {q.shape}")
    kv_flat = _flatten_sparse_mla_kv(kv)
    indices_2d = _flatten_sparse_mla_indices(indices).to(
        device=q.device,
        dtype=torch.int64,
    )
    if topk_length.dim() != 1 or topk_length.shape[0] != q.shape[0]:
        raise ValueError(
            "topk_length must be [tokens], " f"got {topk_length.shape} for q={q.shape}"
        )
    if indices_2d.shape[0] != q.shape[0]:
        raise ValueError(f"indices token count {indices_2d.shape[0]} != {q.shape[0]}")
    if kv_flat.shape[-1] != q.shape[-1]:
        raise ValueError(f"kv dim {kv_flat.shape[-1]} != q dim {q.shape[-1]}")
    if kv_flat.shape[0] == 0:
        if bool((topk_length > 0).any().item()):
            raise ValueError("kv must contain rows when topk_length has candidates")
        return torch.zeros_like(q)

    out = torch.zeros_like(q)
    num_heads = q.shape[1]
    sink = None
    if attn_sink is not None:
        sink = attn_sink.to(device=q.device, dtype=torch.float32)
        if sink.numel() < num_heads:
            raise ValueError(
                f"attn_sink has {sink.numel()} heads, expected at least {num_heads}"
            )

    query_chunk_size = _resolve_chunk_size(
        query_chunk_size,
        limit=q.shape[0],
        default=64,
        name="query_chunk_size",
    )
    topk_chunk_size = _resolve_chunk_size(
        topk_chunk_size,
        limit=indices_2d.shape[1],
        default=256,
        name="topk_chunk_size",
    )
    topk_length = topk_length.to(device=q.device, dtype=torch.int64)

    for token_start in range(0, q.shape[0], query_chunk_size):
        token_end = min(token_start + query_chunk_size, q.shape[0])
        q_chunk = q[token_start:token_end]
        lengths_chunk = topk_length[token_start:token_end]
        indices_chunk_full = indices_2d[token_start:token_end]
        q_bhd, max_score, denom, acc = _new_attention_state(q_chunk)

        for index_start in range(0, indices_2d.shape[1], topk_chunk_size):
            index_end = min(index_start + topk_chunk_size, indices_2d.shape[1])
            indices_chunk = indices_chunk_full[:, index_start:index_end]
            index_offsets = torch.arange(
                index_start,
                index_end,
                dtype=torch.int64,
                device=q.device,
            )
            valid_tokens = (index_offsets[None, :] < lengths_chunk[:, None]) & (
                indices_chunk >= 0
            )
            safe_indices = torch.where(
                valid_tokens,
                indices_chunk,
                torch.zeros((), dtype=torch.int64, device=q.device),
            )
            gathered_kv = kv_flat[safe_indices]
            max_score, denom, acc = _accumulate_attention_chunk(
                q_bhd=q_bhd,
                kv_btd=gathered_kv,
                valid_tokens=valid_tokens,
                max_score=max_score,
                denom=denom,
                acc=acc,
                sm_scale=sm_scale,
            )

        output_no_sink, lse_no_sink = _finish_attention_no_sink(
            max_score=max_score,
            denom=denom,
            acc=acc,
        )
        _write_attention_output(
            output_no_sink=output_no_sink,
            lse_no_sink=lse_no_sink,
            attn_sink=sink,
            output=out[token_start:token_end],
        )
    return out
