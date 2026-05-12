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

"""SM12x DeepSeek V4 attention output projection kernels (CUDA).

This wrapper is targeted at the SM12x family validated by this project --
SM120 (RTX Pro / GeForce 50-series) and SM121 (GB10). It calls
:func:`tokenspeed_kernel.platform.ensure_sm12x_supported_device` on the
output tensor's device before launching, so accidental dispatch on
Hopper / Blackwell / Ampere will fail loudly instead of hitting an opaque
``no kernel image available`` CUDA error at launch time.

Two ops are exported by the underlying ``.so``:

- :func:`sm12x_deepseek_v4_fused_inv_rope_fp8_quant` — applies inverse
  RoPE to the last ``rope_dim`` elements of each attention head and
  quantises the full ``head_dim`` into FP8 e4m3 with one UE8M0-style
  fp32 scale per 128-element block. Output uses
  ``[n_groups, T_aligned, hidden]`` storage with the scale view strided
  for fp8x4 vectorised reads.
- :func:`sm12x_deepseek_v4_grouped_fp8_gemv` — the ``bhr,hdr->bhd``
  per-group batched FP8 GEMV that follows the inverse-RoPE / FP8 quant
  step; ported from upstream DeepGEMM's SM120 einsum kernel.
"""

from __future__ import annotations

import functools
from pathlib import Path

import torch
from tokenspeed_kernel.platform import ensure_sm12x_supported_device


@functools.cache
def _load_module():
    import tvm_ffi

    so_path = (
        Path(__file__).resolve().parent
        / "objs"
        / "sm12x_deepseek_v4_output_proj"
        / "sm12x_deepseek_v4_output_proj.so"
    )
    if not so_path.exists():
        raise RuntimeError(
            "tokenspeed_kernel sm12x_deepseek_v4_output_proj library not found "
            f"at {so_path}. Run: pip install -e tokenspeed-kernel/python/"
        )
    return tvm_ffi.load_module(str(so_path))


def sm12x_deepseek_v4_grouped_fp8_gemv(
    output: torch.Tensor,
    a: torch.Tensor,
    a_scales: torch.Tensor,
    b: torch.Tensor,
    b_scales: torch.Tensor,
) -> None:
    """In-place per-group batched FP8 GEMV for DeepSeek V4 attention output.

    Computes ``output[t, g, n] = sum_k (a[t,g,k] * b[g,n,k] * a_scales[t,g,k/128] * b_scales[g,n/128,k/128])``
    where each ``k/128`` block of activations and ``(n/128, k/128)`` block of
    weights carries its own scale.

    Shapes:
        a       : ``[num_tokens, num_groups, hidden]``         FP8 e4m3
        a_scales: ``[num_tokens, num_groups, hidden/128]``     fp32
        b       : ``[num_groups, out_rank, hidden]``           FP8 e4m3, contig
        b_scales: ``[num_groups, out_rank/128, hidden/128]``   UE8M0 or fp32
        output  : ``[num_tokens, num_groups, out_rank]``       bf16

    ``a`` and ``a_scales`` may be non-contiguous strided views (e.g. the
    transposed outputs of the fused inverse-RoPE + FP8 quant kernel);
    ``b`` and ``b_scales`` must be contiguous.
    """

    if a.dim() != 3 or a_scales.dim() != 3 or b.dim() != 3 or b_scales.dim() != 3:
        raise ValueError("a, a_scales, b, b_scales must all be 3-D tensors")
    if output.dim() != 3:
        raise ValueError("output must be a 3-D tensor")

    if a.dtype != torch.float8_e4m3fn:
        raise ValueError(f"a must use float8_e4m3fn, got {a.dtype}")
    if a_scales.dtype != torch.float32:
        raise ValueError(f"a_scales must use float32, got {a_scales.dtype}")
    if b.dtype != torch.float8_e4m3fn:
        raise ValueError(f"b must use float8_e4m3fn, got {b.dtype}")
    if output.dtype != torch.bfloat16:
        raise ValueError(f"output must use bfloat16, got {output.dtype}")

    for name, tensor in (
        ("output", output),
        ("a", a),
        ("a_scales", a_scales),
        ("b", b),
        ("b_scales", b_scales),
    ):
        if not tensor.is_cuda:
            raise ValueError(f"{name} must be a CUDA tensor")

    if a.stride(2) != 1:
        raise ValueError("a hidden dim must be contiguous (stride==1)")

    num_tokens, num_groups, hidden = a.shape
    out_groups, out_rank, weight_hidden = b.shape
    if out_groups != num_groups:
        raise ValueError(f"b groups ({out_groups}) must match a groups ({num_groups})")
    if weight_hidden != hidden:
        raise ValueError(f"b hidden ({weight_hidden}) must match a hidden ({hidden})")
    if hidden % 128 != 0:
        raise ValueError(f"hidden ({hidden}) must be divisible by 128")
    if out_rank % 128 != 0:
        raise ValueError(f"out_rank ({out_rank}) must be divisible by 128")

    if a_scales.shape != (num_tokens, num_groups, hidden // 128):
        raise ValueError(
            "a_scales shape must be "
            f"{(num_tokens, num_groups, hidden // 128)}, "
            f"got {tuple(a_scales.shape)}"
        )
    expected_b_scales = (num_groups, (out_rank + 127) // 128, hidden // 128)
    if b_scales.shape != expected_b_scales:
        raise ValueError(
            f"b_scales shape must be {expected_b_scales}, "
            f"got {tuple(b_scales.shape)}"
        )
    if output.shape != (num_tokens, num_groups, out_rank):
        raise ValueError(
            "output shape must be "
            f"{(num_tokens, num_groups, out_rank)}, "
            f"got {tuple(output.shape)}"
        )

    e8m0_dtype = getattr(torch, "float8_e8m0fnu", None)
    if b_scales.dtype == e8m0_dtype:
        scales_arg = b_scales.view(torch.uint8)
    elif b_scales.dtype in {torch.uint8, torch.float32}:
        scales_arg = b_scales
    else:
        raise ValueError(
            "b_scales must use float8_e8m0fnu, uint8, or float32, "
            f"got {b_scales.dtype}"
        )

    if num_tokens == 0:
        return

    ensure_sm12x_supported_device(output.device)
    _load_module().sm12x_deepseek_v4_grouped_fp8_gemv(
        output,
        a,
        a_scales,
        b.contiguous(),
        scales_arg.contiguous(),
    )


def _aligned_tokens(num_tokens: int) -> int:
    return ((num_tokens + 3) // 4) * 4


def sm12x_deepseek_v4_fused_inv_rope_fp8_quant(
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
    """Inverse-RoPE attention output + grouped block-128 FP8 quant.

    Returns ``(o_fp8, o_scale)`` with logical shapes
    ``[tokens, groups, heads_per_group * head_dim]`` (FP8 e4m3) and
    ``[tokens, groups, hidden / quant_group_size]`` (fp32), both as
    non-contiguous strided views. The underlying storage is
    ``[groups, T_aligned, hidden]`` and a strided scale buffer chosen so
    the downstream FP8 einsum can walk the per-token row with stride 1.

    Hard-codes the DSv4-Flash head shape (``head_dim = nope_dim +
    rope_dim = 512``); larger or smaller heads need their own kernel
    instance.
    """

    if o.dim() != 3:
        raise ValueError(f"o must be 3-D, got {o.dim()}")
    if positions.dim() != 1:
        raise ValueError(f"positions must be 1-D, got {positions.dim()}")
    if cos_sin_cache.dim() != 2:
        raise ValueError(f"cos_sin_cache must be 2-D, got {cos_sin_cache.dim()}")
    if o.dtype != torch.bfloat16:
        raise ValueError(f"o must be bfloat16, got {o.dtype}")
    if positions.dtype != torch.int64:
        raise ValueError(f"positions must be int64, got {positions.dtype}")
    if cos_sin_cache.dtype != torch.float32:
        raise ValueError(f"cos_sin_cache must be float32, got {cos_sin_cache.dtype}")

    for name, tensor in (
        ("o", o),
        ("positions", positions),
        ("cos_sin_cache", cos_sin_cache),
    ):
        if not tensor.is_cuda:
            raise ValueError(f"{name} must be a CUDA tensor")

    num_tokens, num_heads, head_dim = o.shape
    if num_heads != n_groups * heads_per_group:
        raise ValueError(
            f"expected {n_groups * heads_per_group} heads, got {num_heads}"
        )
    if head_dim != nope_dim + rope_dim:
        raise ValueError(f"expected head_dim={nope_dim + rope_dim}, got {head_dim}")
    if head_dim % quant_group_size != 0:
        raise ValueError(f"head_dim={head_dim} must be divisible by {quant_group_size}")
    if nope_dim % quant_group_size != quant_group_size - rope_dim:
        raise ValueError("DeepSeek V4 inverse-RoPE block layout is unsupported")
    if rope_dim % 2 != 0:
        raise ValueError(f"rope_dim must be even, got {rope_dim}")
    if cos_sin_cache.shape[-1] < rope_dim:
        raise ValueError(
            f"cos_sin_cache last dim must be >= {rope_dim}, "
            f"got {cos_sin_cache.shape[-1]}"
        )
    if quant_group_size != 128:
        raise ValueError(
            f"quant_group_size must be 128 for the SM12x CUDA kernel, "
            f"got {quant_group_size}"
        )
    if o.stride(2) != 1:
        raise ValueError("o hidden dim must be contiguous (stride==1)")
    if positions.shape[0] != num_tokens:
        raise ValueError(
            f"positions length ({positions.shape[0]}) must match num_tokens ({num_tokens})"
        )

    hidden = heads_per_group * head_dim
    scale_blocks = hidden // quant_group_size
    aligned_tokens = _aligned_tokens(num_tokens)

    fp8_buf = torch.empty(
        (n_groups, aligned_tokens, hidden),
        dtype=torch.float8_e4m3fn,
        device=o.device,
    )
    scale_storage = torch.empty(
        n_groups * scale_blocks * aligned_tokens,
        dtype=torch.float32,
        device=o.device,
    )
    scale_buf = scale_storage.as_strided(
        (n_groups, num_tokens, scale_blocks),
        (scale_blocks * aligned_tokens, 1, aligned_tokens),
    )

    if num_tokens == 0:
        return fp8_buf[:, :0, :].transpose(0, 1), scale_buf.transpose(0, 1)

    ensure_sm12x_supported_device(o.device)
    _load_module().sm12x_deepseek_v4_inv_rope_fp8_quant(
        fp8_buf,
        scale_buf,
        o,
        positions,
        cos_sin_cache.contiguous(),
        heads_per_group,
        nope_dim,
        rope_dim,
    )
    # `fp8_buf` was allocated with the T_aligned slot count; slice back to
    # the live `num_tokens` rows before exposing it. The CUDA kernel zero
    # the scale for padded tokens; the data rows for padded tokens are
    # never read by the downstream einsum.
    fp8_live = fp8_buf[:, :num_tokens, :]
    return fp8_live.transpose(0, 1), scale_buf.transpose(0, 1)


__all__ = [
    "sm12x_deepseek_v4_fused_inv_rope_fp8_quant",
    "sm12x_deepseek_v4_grouped_fp8_gemv",
]
