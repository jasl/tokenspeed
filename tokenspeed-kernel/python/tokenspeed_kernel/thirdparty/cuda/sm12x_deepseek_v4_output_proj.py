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

"""SM12x DeepSeek V4 attention output projection FP8 einsum (CUDA)."""

from __future__ import annotations

import functools
from pathlib import Path

import torch


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
    transposed outputs of the Triton fused inverse-RoPE + FP8 quant kernel);
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
        raise ValueError(
            f"b groups ({out_groups}) must match a groups ({num_groups})"
        )
    if weight_hidden != hidden:
        raise ValueError(
            f"b hidden ({weight_hidden}) must match a hidden ({hidden})"
        )
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

    _load_module().sm12x_deepseek_v4_grouped_fp8_gemv(
        output,
        a,
        a_scales,
        b.contiguous(),
        scales_arg.contiguous(),
    )


__all__ = ["sm12x_deepseek_v4_grouped_fp8_gemv"]
