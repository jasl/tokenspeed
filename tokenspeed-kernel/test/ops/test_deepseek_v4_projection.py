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


def _inverse_rope_grouped_reference(
    o: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    *,
    n_groups: int,
    heads_per_group: int,
    nope_dim: int,
    rope_dim: int,
) -> torch.Tensor:
    out = o.float().clone()
    cos = cos_sin_cache[positions, : rope_dim // 2]
    sin = cos_sin_cache[positions, rope_dim // 2 : rope_dim]
    even = out[..., nope_dim::2].clone()
    odd = out[..., nope_dim + 1 :: 2].clone()
    while cos.ndim < even.ndim:
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
    out[..., nope_dim::2] = even * cos + odd * sin
    out[..., nope_dim + 1 :: 2] = odd * cos - even * sin
    return out.reshape(o.shape[0], n_groups, heads_per_group, -1).flatten(2)


def _fp8_quant_reference(
    values: torch.Tensor,
    block_size: int = 128,
) -> tuple[torch.Tensor, torch.Tensor]:
    groups = values.float().reshape(*values.shape[:-1], values.shape[-1] // 128, 128)
    scale = groups.abs().amax(dim=-1).clamp_min(1.0e-10)
    scale = torch.pow(
        2.0,
        torch.ceil(torch.log2(scale / torch.finfo(torch.float8_e4m3fn).max)),
    )
    quantized = torch.clamp(
        groups / scale.unsqueeze(-1),
        torch.finfo(torch.float8_e4m3fn).min,
        torch.finfo(torch.float8_e4m3fn).max,
    ).to(torch.float8_e4m3fn)
    return quantized.reshape_as(values).contiguous(), scale.contiguous()


@pytest.mark.skipif(not _has_sm12x(), reason="SM12x CUDA GPU required")
def test_deepseek_v4_fused_inv_rope_fp8_quant_matches_reference():
    from tokenspeed_kernel.ops.attention.deepseek_v4 import (
        deepseek_v4_fused_inv_rope_fp8_quant_triton,
    )

    torch.manual_seed(20260509)
    tokens = 3
    n_groups = 2
    heads_per_group = 2
    nope_dim = 448
    rope_dim = 64
    head_dim = nope_dim + rope_dim
    max_position = 32
    positions = torch.tensor([1, 7, 17], device="cuda", dtype=torch.int64)
    theta = torch.randn(max_position, rope_dim // 2, device="cuda", dtype=torch.float32)
    cos_sin_cache = torch.cat([torch.cos(theta), torch.sin(theta)], dim=-1)
    o = torch.randn(
        tokens,
        n_groups * heads_per_group,
        head_dim,
        device="cuda",
        dtype=torch.bfloat16,
    )

    actual_fp8, actual_scale = deepseek_v4_fused_inv_rope_fp8_quant_triton(
        o,
        positions,
        cos_sin_cache,
        n_groups=n_groups,
        heads_per_group=heads_per_group,
        nope_dim=nope_dim,
        rope_dim=rope_dim,
    )
    grouped = _inverse_rope_grouped_reference(
        o,
        positions,
        cos_sin_cache,
        n_groups=n_groups,
        heads_per_group=heads_per_group,
        nope_dim=nope_dim,
        rope_dim=rope_dim,
    )
    expected_fp8, expected_scale = _fp8_quant_reference(grouped)

    torch.cuda.synchronize()
    assert actual_fp8.shape == expected_fp8.shape
    assert actual_scale.shape == expected_scale.shape
    torch.testing.assert_close(actual_scale, expected_scale, rtol=0, atol=0)
    torch.testing.assert_close(actual_fp8.float(), expected_fp8.float(), rtol=0, atol=0)


@pytest.mark.skipif(not _has_sm12x(), reason="SM12x CUDA GPU required")
def test_deepseek_v4_fp8_einsum_matches_reference():
    from tokenspeed_kernel.ops.attention.deepseek_v4 import (
        deepseek_v4_fp8_einsum_sm12x_triton,
    )

    torch.manual_seed(20260510)
    tokens = 4
    groups = 2
    hidden = 512
    out_rank = 256
    a_float = torch.randn(tokens, groups, hidden, device="cuda") * 0.75
    a, a_scale = _fp8_quant_reference(a_float)
    b_float = torch.randn(groups * out_rank, hidden, device="cuda") * 0.25
    b = b_float.to(torch.float8_e4m3fn)
    b_scale = torch.pow(
        2.0,
        torch.ceil(
            torch.log2(
                torch.rand(
                    groups * (out_rank // 128), hidden // 128, device="cuda"
                ).clamp_min(1.0e-4)
            )
        ),
    ).float()
    out = torch.empty(tokens, groups, out_rank, device="cuda", dtype=torch.bfloat16)

    deepseek_v4_fp8_einsum_sm12x_triton(a, a_scale, b, b_scale, out)

    a_effective = a.float() * a_scale.repeat_interleave(128, dim=-1)
    b_scale_view = b_scale.view(groups, out_rank // 128, hidden // 128)
    b_effective = b.view(groups, out_rank, hidden).float()
    b_effective = b_effective * b_scale_view.repeat_interleave(
        128, dim=1
    ).repeat_interleave(128, dim=2)
    expected = torch.einsum("tgh,grh->tgr", a_effective, b_effective).to(torch.bfloat16)

    torch.cuda.synchronize()
    assert out.shape == expected.shape
    assert out.dtype == torch.bfloat16
    torch.testing.assert_close(out.float(), expected.float(), rtol=2e-2, atol=0.5)
