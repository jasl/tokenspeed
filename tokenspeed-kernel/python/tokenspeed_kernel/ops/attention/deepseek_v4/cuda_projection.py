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

"""SM12x DeepSeek V4 attention output projection kernels (CUDA)."""

from __future__ import annotations

import torch
from tokenspeed_kernel.platform import ArchVersion, CapabilityRequirement
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.thirdparty.cuda.sm12x_deepseek_v4_output_proj import (
    sm12x_deepseek_v4_fused_inv_rope_fp8_quant as _sm12x_deepseek_v4_fused_inv_rope_fp8_quant,
)
from tokenspeed_kernel.thirdparty.cuda.sm12x_deepseek_v4_output_proj import (
    sm12x_deepseek_v4_grouped_fp8_gemv as _sm12x_deepseek_v4_grouped_fp8_gemv,
)


@register_kernel(
    "attention",
    "deepseek_v4_fused_inv_rope_fp8_quant",
    name="deepseek_v4_fused_inv_rope_fp8_quant_sm12x_cuda",
    solution="cuda",
    capability=CapabilityRequirement(
        min_arch_version=ArchVersion(12, 0),
        max_arch_version=ArchVersion(12, 1),
        vendors=frozenset({"nvidia"}),
    ),
    dtypes={torch.bfloat16},
    priority=Priority.PERFORMANT,
    tags={"cuda", "deepseek_v4", "fp8", "projection", "sm12x"},
)
def deepseek_v4_fused_inv_rope_fp8_quant_sm12x_cuda(
    o: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    *,
    n_groups: int,
    heads_per_group: int,
    nope_dim: int = 448,
    rope_dim: int = 64,
    quant_group_size: int = 128,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute the DSv4 inverse-RoPE + block-128 FP8 quant on SM12x (CUDA).

    Output: ``[n_groups, T_aligned, hidden]`` storage for the FP8 buffer
    and a strided ``[n_groups, num_tokens, scale_blocks]`` view for the
    scales, both returned as ``.transpose(0, 1)`` so the consumer sees
    ``[num_tokens, n_groups, ...]``.
    """
    return _sm12x_deepseek_v4_fused_inv_rope_fp8_quant(
        o,
        positions,
        cos_sin_cache,
        n_groups=n_groups,
        heads_per_group=heads_per_group,
        nope_dim=nope_dim,
        rope_dim=rope_dim,
        quant_group_size=quant_group_size,
    )


@register_kernel(
    "attention",
    "deepseek_v4_fp8_einsum",
    name="deepseek_v4_fp8_einsum_sm12x_cuda",
    solution="cuda",
    capability=CapabilityRequirement(
        min_arch_version=ArchVersion(12, 0),
        max_arch_version=ArchVersion(12, 1),
        vendors=frozenset({"nvidia"}),
    ),
    dtypes={torch.float8_e4m3fn},
    priority=Priority.PERFORMANT,
    tags={"cuda", "deepseek_v4", "fp8", "projection", "sm12x"},
)
def deepseek_v4_fp8_einsum_sm12x_cuda(
    a: torch.Tensor,
    a_scale: torch.Tensor,
    b: torch.Tensor,
    b_scale: torch.Tensor,
    out: torch.Tensor,
) -> None:
    """Compute DeepSeek V4 ``bhr,hdr->bhd`` for SM12x FP8 tensors (CUDA).

    Validation is delegated to the underlying CUDA entry point, which
    also handles the wo_a weight reshape conventions described below.
    """

    num_tokens, num_groups, hidden_size = a.shape
    if num_tokens == 0:
        return
    out_tokens, out_groups, out_rank = out.shape
    if (out_tokens, out_groups) != (num_tokens, num_groups):
        raise ValueError(
            f"out shape {tuple(out.shape)} does not match a shape {tuple(a.shape)}"
        )

    b_reshaped, b_scale_reshaped = _reshape_wo_a_weight(
        b,
        b_scale,
        num_groups=num_groups,
        out_rank=out_rank,
        hidden_size=hidden_size,
    )

    _sm12x_deepseek_v4_grouped_fp8_gemv(
        out,
        a,
        a_scale,
        b_reshaped,
        b_scale_reshaped,
    )


def _reshape_wo_a_weight(
    b: torch.Tensor,
    b_scale: torch.Tensor,
    *,
    num_groups: int,
    out_rank: int,
    hidden_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reshape the flat wo_a weight + scales into per-group layout.

    The DeepSeek V4 ``wo_a`` weight ships as a flat
    ``[num_local_groups * out_rank, hidden]`` FP8 tensor with scales
    ``[num_local_groups * out_rank/128, hidden/128]``. Reshape both into
    explicit grouped layouts so the kernel can index a single group via
    ``b[g, n, k]`` and ``b_scale[g, n/128, k/128]``.
    """
    if b.dim() == 2:
        if b.shape[0] % out_rank != 0:
            raise ValueError(
                f"wo_a rows={b.shape[0]} must be divisible by out_rank={out_rank}"
            )
        weight_groups = b.shape[0] // out_rank
        if weight_groups != num_groups:
            raise ValueError(
                f"expected {num_groups} local wo_a groups, got {weight_groups}"
            )
        b = b.view(num_groups, out_rank, hidden_size)
    elif b.shape != (num_groups, out_rank, hidden_size):
        raise ValueError(
            f"expected wo_a [{num_groups}, {out_rank}, {hidden_size}], "
            f"got {tuple(b.shape)}"
        )

    if b_scale.dim() == 2:
        scale_out_blocks = (out_rank + 127) // 128
        scale_hidden_blocks = hidden_size // 128
        if b_scale.shape[0] != num_groups * scale_out_blocks:
            raise ValueError(
                "wo_a scale rows do not match local groups: "
                f"expected {num_groups * scale_out_blocks}, "
                f"got {b_scale.shape[0]}"
            )
        b_scale = b_scale.view(num_groups, scale_out_blocks, scale_hidden_blocks)
    elif b_scale.shape != (num_groups, (out_rank + 127) // 128, hidden_size // 128):
        raise ValueError(f"unexpected wo_a scale shape {tuple(b_scale.shape)}")
    return b, b_scale
