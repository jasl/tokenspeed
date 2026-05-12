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
import torch.nn.functional as F
from tokenspeed_kernel.ops.moe.sm12x_mxfp4 import (
    mxfp4_dequantize_packed,
    mxfp8_dequantize,
    mxfp8_mxfp4_dense_reference,
    sm12x_mxfp4_moe_reference_forward,
)

_NIBBLE_TO_VALUE = {
    0x0: 0.0,
    0x1: 0.5,
    0x2: 1.0,
    0x3: 1.5,
    0x4: 2.0,
    0x5: 3.0,
    0x6: 4.0,
    0x7: 6.0,
    0x8: 0.0,
    0x9: -0.5,
    0xA: -1.0,
    0xB: -1.5,
    0xC: -2.0,
    0xD: -3.0,
    0xE: -4.0,
    0xF: -6.0,
}
_VALUE_TO_NIBBLE = {value: nibble for nibble, value in _NIBBLE_TO_VALUE.items()}


def _pack_dense(values: torch.Tensor) -> torch.Tensor:
    flat = values.reshape(-1, values.shape[-1])
    packed = torch.empty(
        (*flat.shape[:-1], flat.shape[-1] // 2),
        dtype=torch.uint8,
        device=values.device,
    )
    for row_idx, row in enumerate(flat):
        nibbles = torch.tensor(
            [_VALUE_TO_NIBBLE[float(value)] for value in row.tolist()],
            dtype=torch.uint8,
            device=values.device,
        )
        packed[row_idx].copy_(nibbles[0::2] | (nibbles[1::2] << 4))
    return packed.reshape(*values.shape[:-1], values.shape[-1] // 2)


def _has_sm12x() -> bool:
    if not torch.cuda.is_available():
        return False
    props = torch.cuda.get_device_properties(0)
    return props.major == 12


def _quantize_mxfp8_ue8m0(values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if values.shape[-1] % 32 != 0:
        raise ValueError("last dim must be divisible by 32")
    blocks = values.float().reshape(*values.shape[:-1], values.shape[-1] // 32, 32)
    max_abs = blocks.abs().amax(dim=-1)
    safe = torch.clamp(max_abs / 448.0, min=2.0**-126)
    exponent = torch.where(
        max_abs > 0,
        torch.ceil(torch.log2(safe)),
        torch.zeros_like(max_abs),
    ).clamp(-127, 127)
    scale = torch.pow(2.0, exponent).repeat_interleave(32, dim=-1)
    encoded = (exponent.to(torch.int16) + 127).to(torch.uint8)
    return (values.float() / scale).to(torch.float8_e4m3fn), encoded


def test_mxfp4_dequantize_packed_uses_e2m1_and_ue8m0_scale():
    nibbles = torch.tensor(
        [0x1, 0xA, 0x3, 0xC, *([0x0] * 28)],
        dtype=torch.uint8,
    )
    packed = (nibbles[0::2] | (nibbles[1::2] << 4)).reshape(1, 1, 16)
    scale = torch.tensor([[[128]]], dtype=torch.uint8)

    actual = mxfp4_dequantize_packed(packed, scale)

    expected = torch.zeros(1, 1, 32)
    expected[0, 0, :4] = torch.tensor([1.0, -2.0, 3.0, -4.0])
    assert torch.equal(actual, expected)


def test_sm12x_mxfp4_reference_forward_matches_dense_oracle():
    torch.manual_seed(0)
    hidden_states = torch.randn(2, 32, dtype=torch.float32)
    topk_weights = torch.tensor([[0.75, 0.25], [0.4, 0.6]], dtype=torch.float32)
    topk_ids = torch.tensor([[0, 1], [1, 0]], dtype=torch.int32)

    values = torch.tensor([0.0, 0.5, -0.5, 1.0, -1.0, 1.5, -1.5, 2.0])
    w13_dense = values[
        torch.arange(2 * 64 * 32).reshape(2, 64, 32) % values.numel()
    ].float()
    w2_dense = values[
        (torch.arange(2 * 32 * 32).reshape(2, 32, 32) + 3) % values.numel()
    ].float()
    w13_scale = torch.full((2, 64, 1), 127, dtype=torch.uint8)
    w2_scale = torch.full((2, 32, 1), 127, dtype=torch.uint8)
    w13_bias = torch.linspace(-0.25, 0.25, 64).repeat(2, 1)
    w2_bias = torch.linspace(-0.1, 0.1, 32).repeat(2, 1)

    actual = sm12x_mxfp4_moe_reference_forward(
        hidden_states,
        topk_weights,
        topk_ids,
        _pack_dense(w13_dense),
        w13_scale,
        _pack_dense(w2_dense),
        w2_scale,
        w13_bias=w13_bias,
        w2_bias=w2_bias,
        activation="swiglu",
    )

    expected = torch.zeros_like(hidden_states)
    for token_idx in range(hidden_states.shape[0]):
        for choice_idx in range(topk_ids.shape[1]):
            expert_id = int(topk_ids[token_idx, choice_idx])
            gate_up = F.linear(
                hidden_states[token_idx],
                w13_dense[expert_id],
                w13_bias[expert_id],
            )
            gate, up = gate_up.chunk(2, dim=-1)
            intermediate = F.silu(gate) * up
            expert_out = F.linear(intermediate, w2_dense[expert_id], w2_bias[expert_id])
            expected[token_idx] += expert_out * topk_weights[token_idx, choice_idx]

    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)


def test_mxfp8_mxfp4_dense_reference_matches_dequantized_matmul():
    torch.manual_seed(0)
    a_values = torch.tensor(
        [
            [1.0, -2.0, 0.5, -0.5, *([0.0] * 28)],
            [1.5, -1.0, 2.0, -3.0, *([0.0] * 28)],
        ],
        dtype=torch.float32,
    )
    b_values = torch.tensor(
        [
            [0.5, -1.0, 1.5, -2.0, *([0.0] * 28)],
            [1.0, 0.5, -0.5, 2.0, *([0.0] * 28)],
            [2.0, -1.5, 0.5, -0.5, *([0.0] * 28)],
        ],
        dtype=torch.float32,
    )
    a = a_values.to(torch.float8_e4m3fn)
    a_scale = torch.full((2, 1), 127, dtype=torch.uint8)
    b = _pack_dense(b_values)
    b_scale = torch.full((3, 1), 127, dtype=torch.uint8)

    actual = mxfp8_mxfp4_dense_reference(a, a_scale, b, b_scale)

    expected = a_values @ b_values.T
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)


def test_sm12x_mxfp4_forward_dispatches_warp_impl_by_default(monkeypatch):
    import tokenspeed_kernel.thirdparty.cuda.sm12x_mxfp4 as sm12x_native

    monkeypatch.delenv("TOKENSPEED_SM12X_MXFP4_MOE_IMPL", raising=False)

    calls = []

    def fake_warp_forward(*args, **kwargs):
        calls.append((args, kwargs))
        return torch.full_like(args[0], 11.0)

    monkeypatch.setattr(
        sm12x_native,
        "sm12x_mxfp4_moe_forward_warp",
        fake_warp_forward,
    )

    hidden_states = torch.zeros(2, 32)
    topk_weights = torch.ones(2, 1)
    topk_ids = torch.zeros(2, 1, dtype=torch.int32)
    w13 = torch.zeros(1, 64, 16, dtype=torch.uint8)
    w13_scale = torch.full((1, 64, 1), 127, dtype=torch.uint8)
    w2 = torch.zeros(1, 32, 16, dtype=torch.uint8)
    w2_scale = torch.full((1, 32, 1), 127, dtype=torch.uint8)

    actual = sm12x_native.sm12x_mxfp4_moe_forward(
        hidden_states,
        topk_weights,
        topk_ids,
        w13,
        w13_scale,
        w2,
        w2_scale,
        activation="swiglu",
        ep_rank=1,
        ep_size=2,
    )

    assert torch.equal(actual, torch.full_like(hidden_states, 11.0))
    assert len(calls) == 1
    assert calls[0][1]["activation"] == "swiglu"
    assert calls[0][1]["ep_rank"] == 1
    assert calls[0][1]["ep_size"] == 2


@pytest.mark.parametrize("impl", ["tensorcore", "tile4", "grouped_tc"])
def test_sm12x_mxfp4_forward_rejects_retired_impls(monkeypatch, impl):
    import tokenspeed_kernel.thirdparty.cuda.sm12x_mxfp4 as sm12x_native

    hidden_states = torch.zeros(2, 32)
    topk_weights = torch.ones(2, 1)
    topk_ids = torch.zeros(2, 1, dtype=torch.int32)
    w13 = torch.zeros(1, 64, 16, dtype=torch.uint8)
    w13_scale = torch.full((1, 64, 1), 127, dtype=torch.uint8)
    w2 = torch.zeros(1, 32, 16, dtype=torch.uint8)
    w2_scale = torch.full((1, 32, 1), 127, dtype=torch.uint8)

    monkeypatch.setenv("TOKENSPEED_SM12X_MXFP4_MOE_IMPL", impl)
    with pytest.raises(ValueError, match="must be 'warp' or 'scalar'"):
        sm12x_native.sm12x_mxfp4_moe_forward(
            hidden_states,
            topk_weights,
            topk_ids,
            w13,
            w13_scale,
            w2,
            w2_scale,
        )


@pytest.mark.skipif(not _has_sm12x(), reason="SM12x CUDA GPU required")
def test_sm12x_mxfp8_mxfp4_mma_tile_matches_dense_reference():
    import tokenspeed_kernel.ops.moe.sm12x_mxfp4 as sm12x_mxfp4

    values = torch.tensor(
        [0.0, 0.5, -0.5, 1.0, -1.0, 1.5, -1.5, 2.0],
        dtype=torch.float32,
        device="cuda",
    )
    a_values = values[torch.arange(16 * 32, device="cuda").reshape(16, 32) % 8]
    b_values = values[(torch.arange(8 * 32, device="cuda").reshape(8, 32) + 3) % 8]
    a = a_values.to(torch.float8_e4m3fn)
    a_scale = torch.full((16, 1), 127, dtype=torch.uint8, device="cuda")
    b = _pack_dense(b_values)
    b_scale = torch.full((8, 1), 127, dtype=torch.uint8, device="cuda")

    actual = sm12x_mxfp4.sm12x_mxfp8_mxfp4_mma_tile(a, a_scale, b, b_scale)
    expected = mxfp8_mxfp4_dense_reference(a, a_scale, b, b_scale)

    torch.cuda.synchronize()
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)


@pytest.mark.skipif(not _has_sm12x(), reason="SM12x CUDA GPU required")
def test_sm12x_mxfp8_mxfp4_mma_tile_applies_block_scales():
    import tokenspeed_kernel.ops.moe.sm12x_mxfp4 as sm12x_mxfp4

    values = torch.tensor(
        [0.0, 0.5, -0.5, 1.0, -1.0, 1.5, -1.5, 2.0],
        dtype=torch.float32,
        device="cuda",
    )
    a_values = values[(torch.arange(16 * 32, device="cuda").reshape(16, 32) + 1) % 8]
    b_values = values[(torch.arange(8 * 32, device="cuda").reshape(8, 32) + 5) % 8]
    a = a_values.to(torch.float8_e4m3fn)
    a_scale = torch.tensor(
        [
            [126],
            [127],
            [128],
            [125],
            [129],
            [126],
            [128],
            [127],
            [125],
            [128],
            [126],
            [129],
            [127],
            [126],
            [128],
            [125],
        ],
        dtype=torch.uint8,
        device="cuda",
    )
    b = _pack_dense(b_values)
    b_scale = torch.tensor(
        [[128], [125], [127], [129], [126], [128], [125], [127]],
        dtype=torch.uint8,
        device="cuda",
    )

    actual = sm12x_mxfp4.sm12x_mxfp8_mxfp4_mma_tile(a, a_scale, b, b_scale)
    expected = mxfp8_mxfp4_dense_reference(a, a_scale, b, b_scale)

    torch.cuda.synchronize()
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)


@pytest.mark.skipif(not _has_sm12x(), reason="SM12x CUDA GPU required")
def test_sm12x_mxfp4_mxfp8_mma_tile_matches_weight_major_reference():
    import tokenspeed_kernel.ops.moe.sm12x_mxfp4 as sm12x_mxfp4

    values = torch.tensor(
        [0.0, 0.5, -0.5, 1.0, -1.0, 1.5, -1.5, 2.0],
        dtype=torch.float32,
        device="cuda",
    )
    w_values = values[(torch.arange(16 * 32, device="cuda").reshape(16, 32) + 3) % 8]
    a_values = values[(torch.arange(8 * 32, device="cuda").reshape(8, 32) + 1) % 8]
    weight = _pack_dense(w_values)
    weight_scale = torch.tensor(
        [
            [127],
            [126],
            [128],
            [125],
            [129],
            [127],
            [126],
            [128],
            [125],
            [129],
            [127],
            [126],
            [128],
            [125],
            [129],
            [127],
        ],
        dtype=torch.uint8,
        device="cuda",
    )
    activations = a_values.to(torch.float8_e4m3fn)
    activation_scale = torch.tensor(
        [[128], [125], [127], [129], [126], [128], [125], [127]],
        dtype=torch.uint8,
        device="cuda",
    )

    actual = sm12x_mxfp4.sm12x_mxfp4_mxfp8_mma_tile(
        weight, weight_scale, activations, activation_scale
    )
    expected = (
        mxfp4_dequantize_packed(weight, weight_scale)
        @ mxfp8_dequantize(activations, activation_scale).T
    )

    torch.cuda.synchronize()
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)


@pytest.mark.skipif(not _has_sm12x(), reason="SM12x CUDA GPU required")
def test_sm12x_mxfp4_mxfp8_dense_matches_weight_major_reference_across_tiles():
    import tokenspeed_kernel.ops.moe.sm12x_mxfp4 as sm12x_mxfp4

    m, n, k = 16, 32, 64
    values = torch.tensor(
        [0.0, 0.5, -0.5, 1.0, -1.0, 1.5, -1.5, 2.0],
        dtype=torch.float32,
        device="cuda",
    )
    w_values = values[
        (torch.arange(n * k, device="cuda").reshape(n, k) + 3) % values.numel()
    ]
    a_values = values[
        (torch.arange(m * k, device="cuda").reshape(m, k) + 1) % values.numel()
    ]
    weight = _pack_dense(w_values)
    weight_scale = (
        125 + (torch.arange(n * (k // 32), device="cuda").reshape(n, -1) % 5)
    ).to(torch.uint8)
    activations = a_values.to(torch.float8_e4m3fn)
    activation_scale = (
        125 + ((torch.arange(m * (k // 32), device="cuda").reshape(m, -1) + 2) % 5)
    ).to(torch.uint8)

    actual = sm12x_mxfp4.sm12x_mxfp4_mxfp8_dense(
        weight, weight_scale, activations, activation_scale
    )
    expected = (
        mxfp4_dequantize_packed(weight, weight_scale)
        @ mxfp8_dequantize(activations, activation_scale).T
    ).T

    torch.cuda.synchronize()
    assert actual.shape == (m, n)
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)


@pytest.mark.skipif(not _has_sm12x(), reason="SM12x CUDA GPU required")
def test_sm12x_mxfp8_mxfp4_dense_matches_reference_across_tiles():
    import tokenspeed_kernel.ops.moe.sm12x_mxfp4 as sm12x_mxfp4

    m, n, k = 32, 16, 64
    values = torch.tensor(
        [0.0, 0.5, -0.5, 1.0, -1.0, 1.5, -1.5, 2.0],
        dtype=torch.float32,
        device="cuda",
    )
    a_values = values[
        (torch.arange(m * k, device="cuda").reshape(m, k) + 2) % values.numel()
    ]
    b_values = values[
        (torch.arange(n * k, device="cuda").reshape(n, k) + 5) % values.numel()
    ]
    a = a_values.to(torch.float8_e4m3fn)
    a_scale = (
        125 + (torch.arange(m * (k // 32), device="cuda").reshape(m, -1) % 5)
    ).to(torch.uint8)
    b = _pack_dense(b_values)
    b_scale = (
        125 + ((torch.arange(n * (k // 32), device="cuda").reshape(n, -1) + 3) % 5)
    ).to(torch.uint8)

    actual = sm12x_mxfp4.sm12x_mxfp8_mxfp4_dense(a, a_scale, b, b_scale)
    expected = mxfp8_mxfp4_dense_reference(a, a_scale, b, b_scale)

    torch.cuda.synchronize()
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)


@pytest.mark.skipif(not _has_sm12x(), reason="SM12x CUDA GPU required")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_sm12x_mxfp4_mxfp8_quantize_matches_torch_reference(dtype: torch.dtype):
    import tokenspeed_kernel.ops.moe.sm12x_mxfp4 as sm12x_mxfp4

    torch.manual_seed(20260508)
    values = (torch.randn(16, 64, dtype=torch.float32, device="cuda") * 3.0).to(dtype)

    actual_fp8, actual_scale = sm12x_mxfp4.sm12x_mxfp4_mxfp8_quantize(values)
    expected_fp8, expected_scale = _quantize_mxfp8_ue8m0(values)

    torch.cuda.synchronize()
    assert actual_fp8.dtype == torch.float8_e4m3fn
    assert actual_scale.dtype == torch.uint8
    assert actual_fp8.shape == values.shape
    assert actual_scale.shape == (values.shape[0], values.shape[1] // 32)
    torch.testing.assert_close(actual_scale, expected_scale)
    torch.testing.assert_close(actual_fp8.float(), expected_fp8.float(), rtol=0, atol=0)


@pytest.mark.skipif(not _has_sm12x(), reason="SM12x CUDA GPU required")
def test_sm12x_mxfp4_swiglu_mxfp8_quantize_matches_torch_reference():
    import tokenspeed_kernel.ops.moe.sm12x_mxfp4 as sm12x_mxfp4

    torch.manual_seed(20260508)
    rows, intermediate, experts = 16, 64, 3
    gate_up = (
        torch.randn(
            rows,
            2 * intermediate,
            dtype=torch.float32,
            device="cuda",
        )
        * 4.0
    )
    expert_ids = torch.randint(0, experts, (rows,), dtype=torch.int32, device="cuda")
    w13_bias = (
        torch.randn(
            experts,
            2 * intermediate,
            dtype=torch.float32,
            device="cuda",
        )
        * 0.25
    )

    actual_fp8, actual_scale = sm12x_mxfp4.sm12x_mxfp4_swiglu_mxfp8_quantize(
        gate_up,
        expert_ids,
        w13_bias=w13_bias,
        swiglu_alpha=1.702,
        swiglu_limit=7.0,
        swiglu_beta=0.125,
    )

    biased = gate_up + w13_bias[expert_ids.long()]
    gate, up = biased.chunk(2, dim=-1)
    gate = torch.clamp(gate, max=7.0)
    up = torch.clamp(up, min=-7.0, max=7.0)
    intermediate_values = gate * torch.sigmoid(1.702 * gate) * (up + 0.125)
    expected_fp8, expected_scale = _quantize_mxfp8_ue8m0(intermediate_values)

    torch.cuda.synchronize()
    assert actual_fp8.dtype == torch.float8_e4m3fn
    assert actual_scale.dtype == torch.uint8
    assert actual_fp8.shape == (rows, intermediate)
    assert actual_scale.shape == (rows, intermediate // 32)
    torch.testing.assert_close(actual_scale, expected_scale)
    torch.testing.assert_close(actual_fp8.float(), expected_fp8.float(), rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize(
    "forward_name",
    [
        "sm12x_mxfp4_moe_forward_scalar",
        "sm12x_mxfp4_moe_forward_warp",
    ],
)
def test_sm12x_mxfp4_native_forward_impls_match_reference_with_ep_masking(
    forward_name,
):
    torch.manual_seed(1)
    device = torch.device("cuda")
    hidden_states = torch.randn(2, 64, dtype=torch.float32, device=device)
    topk_weights = torch.tensor(
        [[0.75, 0.25], [0.4, 0.6]], dtype=torch.float32, device=device
    )
    topk_ids = torch.tensor([[0, 2], [3, 1]], dtype=torch.int32, device=device)

    values = torch.tensor(
        [0.0, 0.5, -0.5, 1.0, -1.0, 1.5, -1.5, 2.0],
        dtype=torch.float32,
        device=device,
    )
    w13_dense = values[
        torch.arange(2 * 64 * 64, device=device).reshape(2, 64, 64) % values.numel()
    ]
    w2_dense = values[
        (torch.arange(2 * 64 * 32, device=device).reshape(2, 64, 32) + 3)
        % values.numel()
    ]
    w13_scale = (
        125 + (torch.arange(2 * 64 * 2, device=device).reshape(2, 64, 2) % 5)
    ).to(torch.uint8)
    w2_scale = (126 + (torch.arange(2 * 64, device=device).reshape(2, 64, 1) % 3)).to(
        torch.uint8
    )
    w13_bias = torch.linspace(-0.25, 0.25, 64, device=device).repeat(2, 1)
    w2_bias = torch.linspace(-0.1, 0.1, 64, device=device).repeat(2, 1)

    import tokenspeed_kernel.ops.moe.sm12x_mxfp4 as sm12x_mxfp4

    forward_impl = getattr(sm12x_mxfp4, forward_name)

    actual = forward_impl(
        hidden_states,
        topk_weights,
        topk_ids,
        _pack_dense(w13_dense),
        w13_scale,
        _pack_dense(w2_dense),
        w2_scale,
        w13_bias=w13_bias,
        w2_bias=w2_bias,
        activation="swiglu",
        ep_rank=1,
        ep_size=2,
    )
    expected = sm12x_mxfp4_moe_reference_forward(
        hidden_states,
        topk_weights,
        topk_ids,
        _pack_dense(w13_dense),
        w13_scale,
        _pack_dense(w2_dense),
        w2_scale,
        w13_bias=w13_bias,
        w2_bias=w2_bias,
        activation="swiglu",
        ep_rank=1,
        ep_size=2,
    )

    torch.cuda.synchronize()
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_sm12x_mxfp4_native_forward_handles_ue8m0_min_subnormal_scale():
    device = torch.device("cuda")
    hidden_states = torch.full((1, 32), 0.125, dtype=torch.float32, device=device)
    topk_weights = torch.ones(1, 1, dtype=torch.float32, device=device)
    topk_ids = torch.zeros(1, 1, dtype=torch.int32, device=device)

    w13_dense = torch.full((1, 64, 32), 0.5, dtype=torch.float32, device=device)
    w2_dense = torch.full((1, 32, 32), 0.5, dtype=torch.float32, device=device)
    w13_scale = torch.full((1, 64, 1), 127, dtype=torch.uint8, device=device)
    w2_scale = torch.zeros((1, 32, 1), dtype=torch.uint8, device=device)

    import tokenspeed_kernel.ops.moe.sm12x_mxfp4 as sm12x_mxfp4

    actual = sm12x_mxfp4.sm12x_mxfp4_moe_forward_warp(
        hidden_states,
        topk_weights,
        topk_ids,
        _pack_dense(w13_dense),
        w13_scale,
        _pack_dense(w2_dense),
        w2_scale,
        activation="swiglu",
    )
    expected = sm12x_mxfp4_moe_reference_forward(
        hidden_states,
        topk_weights,
        topk_ids,
        _pack_dense(w13_dense),
        w13_scale,
        _pack_dense(w2_dense),
        w2_scale,
        activation="swiglu",
    )

    torch.cuda.synchronize()
    assert torch.count_nonzero(expected).item() == expected.numel()
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=0.0)


@pytest.mark.skipif(not _has_sm12x(), reason="SM12x CUDA GPU required")
def test_sm12x_mxfp4_native_forward_matches_reference_for_deepseek_v4_shape():
    torch.manual_seed(20260509)
    device = torch.device("cuda")
    hidden_states = (
        torch.randn(1, 4096, dtype=torch.float32, device=device) * 0.01
    ).to(torch.bfloat16)
    topk_weights = torch.tensor(
        [[0.20, 0.18, 0.17, 0.16, 0.15, 0.14]],
        dtype=torch.float32,
        device=device,
    )
    topk_ids = torch.tensor([[0, 1, 0, 1, 0, 1]], dtype=torch.int32, device=device)
    w13 = torch.randint(0, 256, (1, 4096, 2048), dtype=torch.uint8, device=device)
    w13_scale = torch.full((1, 4096, 128), 120, dtype=torch.uint8, device=device)
    w2 = torch.randint(0, 256, (1, 4096, 1024), dtype=torch.uint8, device=device)
    w2_scale = torch.full((1, 4096, 64), 120, dtype=torch.uint8, device=device)

    import tokenspeed_kernel.ops.moe.sm12x_mxfp4 as sm12x_mxfp4

    actual = sm12x_mxfp4.sm12x_mxfp4_moe_forward_warp(
        hidden_states,
        topk_weights,
        topk_ids,
        w13,
        w13_scale,
        w2,
        w2_scale,
        activation="swiglu",
        swiglu_alpha=1.702,
        swiglu_limit=7.0,
        ep_rank=0,
        ep_size=2,
    )
    expected = sm12x_mxfp4_moe_reference_forward(
        hidden_states,
        topk_weights,
        topk_ids,
        w13,
        w13_scale,
        w2,
        w2_scale,
        activation="swiglu",
        swiglu_alpha=1.702,
        swiglu_limit=7.0,
        ep_rank=0,
        ep_size=2,
    )

    torch.cuda.synchronize()
    torch.testing.assert_close(actual.float(), expected.float(), rtol=2e-3, atol=0.25)
