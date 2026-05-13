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

from __future__ import annotations

import pytest
import torch


def _has_sm12x() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 12


def _torch_mxfp8_block128_quantize(
    values: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    groups = values.float().reshape(values.shape[0], values.shape[1] // 128, 128)
    scales = groups.abs().amax(dim=-1).clamp_min(1.0e-10)
    scales = scales / torch.finfo(torch.float8_e4m3fn).max
    quantized = torch.clamp(
        groups / scales.unsqueeze(-1),
        torch.finfo(torch.float8_e4m3fn).min,
        torch.finfo(torch.float8_e4m3fn).max,
    ).to(torch.float8_e4m3fn)
    return quantized.reshape_as(values).contiguous(), scales.contiguous()


def _torch_deepseek_v4_fp8_quant_dequant(values: torch.Tensor) -> torch.Tensor:
    groups = values.float().reshape(values.shape[0], values.shape[1] // 128, 128)
    amax = groups.abs().amax(dim=-1).clamp_min(1.0e-4)
    scale = torch.pow(2.0, torch.ceil(torch.log2(amax / 448.0)))
    scale = scale.to(torch.float8_e8m0fnu).float()
    quantized = (
        (groups / scale.unsqueeze(-1)).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    )
    return (quantized.float() * scale.unsqueeze(-1)).flatten(-2).reshape_as(values)


@pytest.mark.skipif(not _has_sm12x(), reason="SM12x CUDA GPU required")
def test_sm12x_mxfp8_block128_quantize_matches_triton_fallback_quantization():
    from tokenspeed_kernel.ops.gemm.sm12x_fp8 import (
        sm12x_mxfp8_block128_quantize,
    )

    torch.manual_seed(0)
    values = torch.randn(3, 256, device="cuda", dtype=torch.bfloat16) * 3.0
    values[0, :128] = 0

    actual_q, actual_scales = sm12x_mxfp8_block128_quantize(values)
    expected_q, expected_scales = _torch_mxfp8_block128_quantize(values)

    torch.cuda.synchronize()
    assert actual_q.dtype == torch.float8_e4m3fn
    assert actual_scales.dtype == torch.float32
    assert actual_q.shape == values.shape
    assert actual_scales.shape == (3, 2)
    torch.testing.assert_close(actual_scales, expected_scales, rtol=1e-6, atol=1e-8)
    torch.testing.assert_close(actual_q.float(), expected_q.float(), rtol=0, atol=0)


@pytest.mark.skipif(not _has_sm12x(), reason="SM12x CUDA GPU required")
def test_sm12x_mxfp8_block128_quant_dequant_matches_deepseek_v4_reference():
    from tokenspeed_kernel.ops.gemm.sm12x_fp8 import (
        sm12x_mxfp8_block128_quant_dequant_ue8m0,
    )

    torch.manual_seed(1)
    values = torch.randn(5, 256, device="cuda", dtype=torch.bfloat16) * 3.0
    values[0, :128] = 0

    actual = sm12x_mxfp8_block128_quant_dequant_ue8m0(values)
    expected = _torch_deepseek_v4_fp8_quant_dequant(values)

    torch.cuda.synchronize()
    assert actual.dtype == torch.float32
    assert actual.shape == values.shape
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.skipif(not _has_sm12x(), reason="SM12x CUDA GPU required")
def test_sm12x_mxfp8_mm_defaults_to_triton_fallback():
    import tokenspeed_kernel
    from tokenspeed_kernel.profiling import ShapeCapture

    torch.manual_seed(0)
    capture = ShapeCapture.get()
    capture.clear()
    capture.enabled = True
    try:
        activations = torch.randn(1, 128, device="cuda", dtype=torch.bfloat16)
        weights = torch.randn(128, 128, device="cuda", dtype=torch.bfloat16).to(
            torch.float8_e4m3fn
        )
        weight_scales = torch.ones(1, 1, device="cuda", dtype=torch.float32)

        output = tokenspeed_kernel.mm(
            activations,
            weights,
            B_scales=weight_scales,
            out_dtype=torch.bfloat16,
            quant="mxfp8",
            block_size=[128, 128],
        )
        torch.cuda.synchronize()
    finally:
        capture.enabled = False

    assert output.shape == (1, 128)
    assert output.dtype == torch.bfloat16
    assert torch.isfinite(output).all()
    assert capture._records[-1].kernel_name == "triton_mm_fp8_blockscale"


@pytest.mark.skipif(not _has_sm12x(), reason="SM12x CUDA GPU required")
def test_sm12x_mxfp8_mm_uses_cuda_quantize_when_enabled(monkeypatch):
    import tokenspeed_kernel
    import tokenspeed_kernel.ops.gemm.sm12x_fp8 as sm12x_fp8

    calls = 0

    def fake_quantize(values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        nonlocal calls
        calls += 1
        return _torch_mxfp8_block128_quantize(values)

    monkeypatch.setenv("TOKENSPEED_SM12X_MXFP8_CUDA_QUANTIZE", "1")
    monkeypatch.setattr(
        sm12x_fp8,
        "sm12x_mxfp8_block128_quantize",
        fake_quantize,
    )

    activations = torch.randn(1, 128, device="cuda", dtype=torch.bfloat16)
    weights = torch.randn(128, 128, device="cuda", dtype=torch.bfloat16).to(
        torch.float8_e4m3fn
    )
    weight_scales = torch.ones(1, 1, device="cuda", dtype=torch.float32)

    output = tokenspeed_kernel.mm(
        activations,
        weights,
        B_scales=weight_scales,
        out_dtype=torch.bfloat16,
        quant="mxfp8",
        block_size=[128, 128],
    )

    torch.cuda.synchronize()
    assert calls == 1
    assert output.shape == (1, 128)
    assert output.dtype == torch.bfloat16


@pytest.mark.skipif(not _has_sm12x(), reason="SM12x CUDA GPU required")
def test_sm12x_fp8_weight_gemv_ue8m0_matches_reference():
    from tokenspeed_kernel.ops.gemm.sm12x_fp8 import (
        sm12x_fp8_weight_gemv_ue8m0,
    )

    torch.manual_seed(20260513)
    m, k, n = 2, 256, 384
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16) * 2.0
    x_eff = _torch_deepseek_v4_fp8_quant_dequant(x).contiguous()
    weight = (torch.randn(n, k, device="cuda", dtype=torch.bfloat16) * 0.25).to(
        torch.float8_e4m3fn
    )
    raw_scales = torch.randn(
        (n + 127) // 128,
        k // 128,
        device="cuda",
        dtype=torch.float32,
    ).abs()
    weight_scales = torch.pow(
        2.0,
        torch.ceil(torch.log2(raw_scales.clamp_min(1.0e-4))),
    ).to(torch.float8_e8m0fnu)

    actual = sm12x_fp8_weight_gemv_ue8m0(x_eff, weight, weight_scales)
    expanded_scales = weight_scales.float().repeat_interleave(128, dim=0)
    expanded_scales = expanded_scales.repeat_interleave(128, dim=1)[:n, :k]
    expected = (x_eff @ (weight.float() * expanded_scales).T).to(torch.bfloat16)

    torch.cuda.synchronize()
    assert actual.shape == (m, n)
    assert actual.dtype == torch.bfloat16
    torch.testing.assert_close(actual.float(), expected.float(), rtol=0, atol=0)
