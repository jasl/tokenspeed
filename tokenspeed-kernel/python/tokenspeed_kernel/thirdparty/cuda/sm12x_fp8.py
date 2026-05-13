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

"""SM12x FP8 helper kernels."""

from __future__ import annotations

import functools
from pathlib import Path

import torch


@functools.cache
def _load_sm12x_fp8_module():
    import tvm_ffi

    so_path = (
        Path(__file__).resolve().parent
        / "objs"
        / "sm12x_fp8_quantize"
        / "sm12x_fp8_quantize.so"
    )
    if not so_path.exists():
        raise RuntimeError(
            f"tokenspeed_kernel sm12x_fp8_quantize library not found at {so_path}. "
            "Run: pip install -e tokenspeed-kernel/python/"
        )
    return tvm_ffi.load_module(str(so_path))


def sm12x_mxfp8_block128_quantize(
    values: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if values.dim() != 2:
        raise ValueError(f"values must be 2-D, got {values.dim()}")
    if values.dtype not in {torch.float32, torch.float16, torch.bfloat16}:
        raise ValueError(
            "values must use float32, float16, or bfloat16, " f"got {values.dtype}"
        )
    if not values.is_cuda:
        raise ValueError("values must be a CUDA tensor")

    rows, hidden_dim = values.shape
    if hidden_dim % 128 != 0:
        raise ValueError(f"values columns must be divisible by 128, got {hidden_dim}")

    output = torch.empty_like(values, dtype=torch.float8_e4m3fn)
    output_scales = torch.empty(
        (rows, hidden_dim // 128), dtype=torch.float32, device=values.device
    )
    if rows == 0:
        return output, output_scales

    _load_sm12x_fp8_module().sm12x_mxfp8_block128_quantize(
        output,
        output_scales,
        values.contiguous(),
    )
    return output, output_scales


def sm12x_mxfp8_block128_quant_dequant_ue8m0(
    values: torch.Tensor,
) -> torch.Tensor:
    if values.dim() != 2:
        raise ValueError(f"values must be 2-D, got {values.dim()}")
    if values.dtype not in {torch.float32, torch.float16, torch.bfloat16}:
        raise ValueError(
            "values must use float32, float16, or bfloat16, " f"got {values.dtype}"
        )
    if not values.is_cuda:
        raise ValueError("values must be a CUDA tensor")

    rows, hidden_dim = values.shape
    if hidden_dim % 128 != 0:
        raise ValueError(f"values columns must be divisible by 128, got {hidden_dim}")

    output = torch.empty(values.shape, dtype=torch.float32, device=values.device)
    if rows == 0:
        return output

    _load_sm12x_fp8_module().sm12x_mxfp8_block128_quant_dequant_ue8m0(
        output,
        values.contiguous(),
    )
    return output


def sm12x_fp8_weight_gemv_ue8m0(
    values: torch.Tensor,
    weight: torch.Tensor,
    weight_scales: torch.Tensor,
) -> torch.Tensor:
    if values.dim() != 2:
        raise ValueError(f"values must be 2-D, got {values.dim()}")
    if values.dtype != torch.float32:
        raise ValueError(f"values must use float32, got {values.dtype}")
    if weight.dim() != 2:
        raise ValueError(f"weight must be 2-D, got {weight.dim()}")
    if weight.dtype != torch.float8_e4m3fn:
        raise ValueError(f"weight must use float8_e4m3fn, got {weight.dtype}")
    if not values.is_cuda or not weight.is_cuda or not weight_scales.is_cuda:
        raise ValueError("values, weight, and weight_scales must be CUDA tensors")
    if values.device != weight.device or values.device != weight_scales.device:
        raise ValueError("values, weight, and weight_scales must be on the same device")

    rows, hidden_dim = values.shape
    out_dim, weight_hidden = weight.shape
    if hidden_dim != weight_hidden:
        raise ValueError(
            f"values columns must match weight columns, got {hidden_dim} and "
            f"{weight_hidden}"
        )
    if hidden_dim % 128 != 0:
        raise ValueError(f"values columns must be divisible by 128, got {hidden_dim}")
    if weight_scales.shape != ((out_dim + 127) // 128, hidden_dim // 128):
        raise ValueError(
            "weight_scales shape must be "
            f"{((out_dim + 127) // 128, hidden_dim // 128)}, "
            f"got {tuple(weight_scales.shape)}"
        )

    e8m0_dtype = getattr(torch, "float8_e8m0fnu", None)
    if weight_scales.dtype == e8m0_dtype:
        scales_arg = weight_scales.view(torch.uint8)
    elif weight_scales.dtype in {torch.uint8, torch.float32}:
        scales_arg = weight_scales
    else:
        raise ValueError(
            "weight_scales must use float8_e8m0fnu, uint8, or float32, "
            f"got {weight_scales.dtype}"
        )

    output = torch.empty((rows, out_dim), dtype=torch.bfloat16, device=values.device)
    if rows == 0:
        return output

    _load_sm12x_fp8_module().sm12x_fp8_weight_gemv_ue8m0(
        output,
        values.contiguous(),
        weight.contiguous(),
        scales_arg.contiguous(),
    )
    return output


__all__ = [
    "sm12x_fp8_weight_gemv_ue8m0",
    "sm12x_mxfp8_block128_quant_dequant_ue8m0",
    "sm12x_mxfp8_block128_quantize",
]
