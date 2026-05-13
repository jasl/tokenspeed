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


def test_sm12x_mxfp4_forward_dispatches_auto_by_default(monkeypatch):
    """The SM12x MoE forward defaults to ``auto`` (M-aware dispatch).

    At the bench's bs=1 * top_k=1 decode shape (M=2), ``auto`` falls
    below the default M-threshold of 16 and routes to the warp kernel.
    A separate test exercises the prefill-sized M path that should
    pick tensorcore.
    """
    import tokenspeed_kernel.thirdparty.cuda.sm12x_mxfp4 as sm12x_native

    assert sm12x_native._SM12X_MXFP4_MOE_DEFAULT_IMPL == "auto"
    monkeypatch.delenv("TOKENSPEED_SM12X_MXFP4_MOE_IMPL", raising=False)
    monkeypatch.delenv("TOKENSPEED_SM12X_MXFP4_MOE_M_THRESHOLD", raising=False)

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


def test_sm12x_mxfp4_forward_explicit_warp_override(monkeypatch):
    """``TOKENSPEED_SM12X_MXFP4_MOE_IMPL=warp`` is the kill switch back to
    the legacy global-warp behaviour (used as the SM12x default before the
    M-aware dispatch landed)."""
    sm12x_native, calls = _install_dispatch_spies(monkeypatch)
    monkeypatch.setenv("TOKENSPEED_SM12X_MXFP4_MOE_IMPL", "warp")
    monkeypatch.delenv("TOKENSPEED_SM12X_MXFP4_MOE_M_THRESHOLD", raising=False)

    # Pick a large M that would normally route to tensorcore under ``auto``.
    hs, tw, ti, w13, w13s, w2, w2s = _dispatch_dummy_kwargs(num_tokens=64, top_k=6)
    out = sm12x_native.sm12x_mxfp4_moe_forward(hs, tw, ti, w13, w13s, w2, w2s)

    assert torch.equal(out, torch.full_like(hs, 21.0))  # warp marker
    assert len(calls["warp"]) == 1
    assert calls["tensorcore"] == []


@pytest.mark.parametrize("impl", ["scalar", "tensorcore", "tile4", "grouped_tc"])
def test_sm12x_mxfp4_forward_rejects_retired_impls(monkeypatch, impl):
    """The legacy names from rejected SM12x MoE experiments stay rejected.

    ``scalar`` joined this set after the per-call ``auto`` dispatch
    obsoleted it -- the warp kernel handles every shape correctly and
    the scalar reference path was never reached in production. The
    current ``persistent`` impl is intentionally not in this set; it
    is the production-targeted name for the tensorcore MoE forward.
    """
    import tokenspeed_kernel.thirdparty.cuda.sm12x_mxfp4 as sm12x_native

    hidden_states = torch.zeros(2, 32)
    topk_weights = torch.ones(2, 1)
    topk_ids = torch.zeros(2, 1, dtype=torch.int32)
    w13 = torch.zeros(1, 64, 16, dtype=torch.uint8)
    w13_scale = torch.full((1, 64, 1), 127, dtype=torch.uint8)
    w2 = torch.zeros(1, 32, 16, dtype=torch.uint8)
    w2_scale = torch.full((1, 32, 1), 127, dtype=torch.uint8)

    monkeypatch.setenv("TOKENSPEED_SM12X_MXFP4_MOE_IMPL", impl)
    with pytest.raises(
        ValueError,
        match="must be 'warp', 'persistent', or 'auto'",
    ):
        sm12x_native.sm12x_mxfp4_moe_forward(
            hidden_states,
            topk_weights,
            topk_ids,
            w13,
            w13_scale,
            w2,
            w2_scale,
        )


def _install_dispatch_spies(monkeypatch):
    """Wire fake warp/tensorcore implementations onto sm12x_native.

    Each fake returns a constant tensor and records its kwargs; the
    dispatch test asserts on the kwargs + the recorded constants.
    """
    import tokenspeed_kernel.thirdparty.cuda.sm12x_mxfp4 as sm12x_native

    calls: dict[str, list] = {"warp": [], "tensorcore": []}

    def _make_fake(label: str, fill: float):
        def fn(*args, **kwargs):
            calls[label].append((args, kwargs))
            return torch.full_like(args[0], fill)

        return fn

    monkeypatch.setattr(
        sm12x_native, "sm12x_mxfp4_moe_forward_warp", _make_fake("warp", 21.0)
    )
    monkeypatch.setattr(
        sm12x_native,
        "sm12x_mxfp4_moe_forward_tensorcore",
        _make_fake("tensorcore", 23.0),
    )
    return sm12x_native, calls


def _dispatch_dummy_kwargs(num_tokens: int = 2, top_k: int = 1):
    hidden_states = torch.zeros(num_tokens, 32)
    topk_weights = torch.ones(num_tokens, top_k)
    topk_ids = torch.zeros(num_tokens, top_k, dtype=torch.int32)
    w13 = torch.zeros(1, 64, 16, dtype=torch.uint8)
    w13_scale = torch.full((1, 64, 1), 127, dtype=torch.uint8)
    w2 = torch.zeros(1, 32, 16, dtype=torch.uint8)
    w2_scale = torch.full((1, 32, 1), 127, dtype=torch.uint8)
    return hidden_states, topk_weights, topk_ids, w13, w13_scale, w2, w2_scale


def test_sm12x_mxfp4_forward_auto_picks_warp_for_small_m(monkeypatch):
    """``auto`` dispatch routes small ``M`` to warp.

    The DSv4-Flash decode shape (``bs=1 * top_k=6 -> M=6``) sits well
    below the kernel's tile-utilisation knee; tensorcore wastes ~10/16
    of the M axis at ``M < 16``. ``auto`` keeps decode on warp.
    """
    sm12x_native, calls = _install_dispatch_spies(monkeypatch)
    monkeypatch.setenv("TOKENSPEED_SM12X_MXFP4_MOE_IMPL", "auto")
    monkeypatch.delenv("TOKENSPEED_SM12X_MXFP4_MOE_M_THRESHOLD", raising=False)

    hs, tw, ti, w13, w13s, w2, w2s = _dispatch_dummy_kwargs(num_tokens=1, top_k=6)
    out = sm12x_native.sm12x_mxfp4_moe_forward(hs, tw, ti, w13, w13s, w2, w2s)

    assert torch.equal(out, torch.full_like(hs, 21.0))  # warp marker
    assert len(calls["warp"]) == 1
    assert calls["tensorcore"] == []


def test_sm12x_mxfp4_forward_auto_picks_tensorcore_for_large_m(monkeypatch):
    """``auto`` dispatch routes large ``M`` (prefill chunk-sized) to tensorcore.

    With ``w2_bias=None`` the persistent contract is satisfied and the
    tensorcore tiles are well-utilised; ``auto`` picks the tensorcore
    forward.
    """
    sm12x_native, calls = _install_dispatch_spies(monkeypatch)
    monkeypatch.setenv("TOKENSPEED_SM12X_MXFP4_MOE_IMPL", "auto")
    monkeypatch.delenv("TOKENSPEED_SM12X_MXFP4_MOE_M_THRESHOLD", raising=False)

    # num_tokens=64 * top_k=6 = M=384 >> 16 default threshold.
    hs, tw, ti, w13, w13s, w2, w2s = _dispatch_dummy_kwargs(num_tokens=64, top_k=6)
    out = sm12x_native.sm12x_mxfp4_moe_forward(hs, tw, ti, w13, w13s, w2, w2s)

    assert torch.equal(out, torch.full_like(hs, 23.0))  # tensorcore marker
    assert calls["warp"] == []
    assert len(calls["tensorcore"]) == 1


def test_sm12x_mxfp4_forward_auto_routes_large_m_with_w2_bias(
    monkeypatch,
):
    """After the W2 bias fuse landed, ``auto`` no longer needs to avoid
    the tensorcore path when ``w2_bias`` is present. Large-M calls go to
    tensorcore regardless of bias; the tensorcore kernel handles the bias
    add at the final write of ``per_pair_out``.

    This used to be ``test_..._falls_back_to_warp_when_w2_bias_present``
    (when the persistent kernel silently dropped w2_bias). The
    bias-aware regression guard is the kernel-level test
    ``test_sm12x_mxfp4_moe_w2_tensorcore_fuses_w2_bias``.
    """
    sm12x_native, calls = _install_dispatch_spies(monkeypatch)
    monkeypatch.setenv("TOKENSPEED_SM12X_MXFP4_MOE_IMPL", "auto")
    monkeypatch.delenv("TOKENSPEED_SM12X_MXFP4_MOE_M_THRESHOLD", raising=False)

    hs, tw, ti, w13, w13s, w2, w2s = _dispatch_dummy_kwargs(num_tokens=64, top_k=6)
    fake_w2_bias = torch.zeros(1, 32, dtype=torch.float32)

    out = sm12x_native.sm12x_mxfp4_moe_forward(
        hs, tw, ti, w13, w13s, w2, w2s, w2_bias=fake_w2_bias
    )

    assert torch.equal(out, torch.full_like(hs, 23.0))  # tensorcore marker
    assert calls["warp"] == []
    assert len(calls["tensorcore"]) == 1


def test_sm12x_mxfp4_forward_auto_threshold_env_override(monkeypatch):
    """``TOKENSPEED_SM12X_MXFP4_MOE_M_THRESHOLD`` shifts the dispatch knee.

    ``M=12`` with a default threshold of ``16`` would route to warp;
    setting the threshold to ``8`` should flip it to tensorcore. Garbage
    values fall back to the default threshold rather than raising.
    """
    sm12x_native, calls = _install_dispatch_spies(monkeypatch)
    monkeypatch.setenv("TOKENSPEED_SM12X_MXFP4_MOE_IMPL", "auto")

    hs, tw, ti, w13, w13s, w2, w2s = _dispatch_dummy_kwargs(num_tokens=2, top_k=6)
    # M=12 with threshold=16 (default) -> warp
    monkeypatch.delenv("TOKENSPEED_SM12X_MXFP4_MOE_M_THRESHOLD", raising=False)
    sm12x_native.sm12x_mxfp4_moe_forward(hs, tw, ti, w13, w13s, w2, w2s)
    assert len(calls["warp"]) == 1
    assert calls["tensorcore"] == []

    # M=12 with threshold=8 -> tensorcore
    monkeypatch.setenv("TOKENSPEED_SM12X_MXFP4_MOE_M_THRESHOLD", "8")
    sm12x_native.sm12x_mxfp4_moe_forward(hs, tw, ti, w13, w13s, w2, w2s)
    assert len(calls["warp"]) == 1
    assert len(calls["tensorcore"]) == 1

    # Garbage threshold -> default 16 -> warp
    monkeypatch.setenv("TOKENSPEED_SM12X_MXFP4_MOE_M_THRESHOLD", "not_a_number")
    sm12x_native.sm12x_mxfp4_moe_forward(hs, tw, ti, w13, w13s, w2, w2s)
    assert len(calls["warp"]) == 2
    assert len(calls["tensorcore"]) == 1

    # Zero / negative threshold -> default 16 -> warp
    monkeypatch.setenv("TOKENSPEED_SM12X_MXFP4_MOE_M_THRESHOLD", "0")
    sm12x_native.sm12x_mxfp4_moe_forward(hs, tw, ti, w13, w13s, w2, w2s)
    assert len(calls["warp"]) == 3


def test_sm12x_mxfp4_forward_dispatches_persistent_to_tensorcore(monkeypatch):
    """`TOKENSPEED_SM12X_MXFP4_MOE_IMPL=persistent` routes to the tensorcore path."""
    import tokenspeed_kernel.thirdparty.cuda.sm12x_mxfp4 as sm12x_native

    monkeypatch.setenv("TOKENSPEED_SM12X_MXFP4_MOE_IMPL", "persistent")

    calls = []

    def fake_tensorcore_forward(*args, **kwargs):
        calls.append((args, kwargs))
        return torch.full_like(args[0], 13.0)

    monkeypatch.setattr(
        sm12x_native,
        "sm12x_mxfp4_moe_forward_tensorcore",
        fake_tensorcore_forward,
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
        ep_rank=0,
        ep_size=1,
    )
    assert torch.equal(actual, torch.full_like(hidden_states, 13.0))
    assert len(calls) == 1
    assert calls[0][1]["ep_rank"] == 0


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


def _moe_w13_tensorcore_reference(
    hidden_fp8: torch.Tensor,
    hidden_scale: torch.Tensor,
    topk_ids: torch.Tensor,
    w13_weight: torch.Tensor,
    w13_scale: torch.Tensor,
    *,
    ep_rank: int,
    num_local_experts: int,
) -> torch.Tensor:
    """Pure-PyTorch reference for the tensorcore W13 GEMM.

    Matches the kernel contract: per (token, top_k) pair, look up the
    selected expert, dequantize FP4 weights with their per-32 ue8m0 scale,
    dequantize FP8 activations with their per-32 ue8m0 scale, and run an
    fp32 matmul. Non-local experts (filtered by ep_rank) write zeros.
    """
    num_tokens, hidden = hidden_fp8.shape
    top_k = topk_ids.shape[1]
    gate_up_dim = w13_weight.shape[1]
    out = torch.zeros(
        (num_tokens, top_k, gate_up_dim),
        dtype=torch.float32,
        device=hidden_fp8.device,
    )
    expert_lo = ep_rank * num_local_experts
    expert_hi = expert_lo + num_local_experts

    hidden_f32 = mxfp8_dequantize(hidden_fp8, hidden_scale)
    for token_idx in range(num_tokens):
        for choice_idx in range(top_k):
            expert_id = int(topk_ids[token_idx, choice_idx])
            if expert_id < expert_lo or expert_id >= expert_hi:
                continue
            local_id = expert_id - expert_lo
            w = mxfp4_dequantize_packed(
                w13_weight[local_id : local_id + 1],
                w13_scale[local_id : local_id + 1],
            )[0]
            out[token_idx, choice_idx] = (
                hidden_f32[token_idx].unsqueeze(0).to(torch.float32)
                @ w.T.to(torch.float32)
            ).squeeze(0)
    return out


@pytest.mark.skipif(not _has_sm12x(), reason="SM12x CUDA GPU required")
def test_sm12x_mxfp4_moe_w13_tensorcore_matches_reference_small():
    """Small-shape correctness: 2 tokens, top_k=4, 8 experts, hidden=64, gate_up=32."""
    import tokenspeed_kernel.thirdparty.cuda.sm12x_mxfp4 as sm12x_native

    torch.manual_seed(20260513)
    device = torch.device("cuda")
    num_tokens = 2
    top_k = 4
    num_local_experts = 8
    hidden = 64
    gate_up_dim = 32

    hidden_states = (
        torch.randn(num_tokens, hidden, dtype=torch.float32, device=device) * 0.5
    )
    hidden_fp8, hidden_scale = _quantize_mxfp8_ue8m0(hidden_states)

    topk_ids = torch.tensor(
        [[0, 3, 1, 5], [2, 7, 4, 6]], dtype=torch.int32, device=device
    )

    values = torch.tensor([0.0, 0.5, -0.5, 1.0, -1.0, 1.5, -1.5, 2.0], device=device)
    w13_dense = values[
        torch.arange(num_local_experts * gate_up_dim * hidden, device=device).reshape(
            num_local_experts, gate_up_dim, hidden
        )
        % values.numel()
    ].float()
    w13_weight = _pack_dense(w13_dense)
    w13_scale = torch.full(
        (num_local_experts, gate_up_dim, hidden // 32),
        127,
        dtype=torch.uint8,
        device=device,
    )

    actual = sm12x_native.sm12x_mxfp4_moe_w13_tensorcore(
        hidden_fp8,
        hidden_scale,
        topk_ids,
        w13_weight,
        w13_scale,
        ep_rank=0,
    )
    expected = _moe_w13_tensorcore_reference(
        hidden_fp8,
        hidden_scale,
        topk_ids,
        w13_weight,
        w13_scale,
        ep_rank=0,
        num_local_experts=num_local_experts,
    )
    torch.cuda.synchronize()
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)


@pytest.mark.skipif(not _has_sm12x(), reason="SM12x CUDA GPU required")
def test_sm12x_mxfp4_moe_w13_tensorcore_filters_non_local_experts():
    """When topk_ids picks out-of-rank experts, those pair outputs must be zero."""
    import tokenspeed_kernel.thirdparty.cuda.sm12x_mxfp4 as sm12x_native

    torch.manual_seed(20260514)
    device = torch.device("cuda")
    num_tokens = 1
    top_k = 4
    num_local_experts = 4
    hidden = 32
    gate_up_dim = 16

    hidden_fp8, hidden_scale = _quantize_mxfp8_ue8m0(
        torch.randn(num_tokens, hidden, dtype=torch.float32, device=device)
    )
    # ep_rank=1, ep_size=2 -> local experts = [4..7]. Two of the four picks are
    # in-range, two are out-of-range -- both must be respected.
    topk_ids = torch.tensor([[3, 5, 0, 6]], dtype=torch.int32, device=device)
    w13_weight = torch.randint(
        0,
        256,
        (num_local_experts, gate_up_dim, hidden // 2),
        dtype=torch.uint8,
        device=device,
    )
    w13_scale = torch.full(
        (num_local_experts, gate_up_dim, hidden // 32),
        120,
        dtype=torch.uint8,
        device=device,
    )

    actual = sm12x_native.sm12x_mxfp4_moe_w13_tensorcore(
        hidden_fp8,
        hidden_scale,
        topk_ids,
        w13_weight,
        w13_scale,
        ep_rank=1,
    )
    torch.cuda.synchronize()
    # Pairs 0 and 2 -> experts 3 and 0 are out of range -> zeros.
    assert torch.all(actual[0, 0] == 0.0)
    assert torch.all(actual[0, 2] == 0.0)
    # Pairs 1 and 3 -> experts 5 and 6 are in range -> non-trivial result.
    assert torch.any(actual[0, 1] != 0.0)
    assert torch.any(actual[0, 3] != 0.0)


def _moe_w2_tensorcore_reference(
    intermediate_fp8: torch.Tensor,
    intermediate_scale: torch.Tensor,
    topk_ids: torch.Tensor,
    w2_weight: torch.Tensor,
    w2_scale: torch.Tensor,
    *,
    ep_rank: int,
    num_local_experts: int,
) -> torch.Tensor:
    """Pure-PyTorch reference for the tensorcore W2 GEMM (per-pair output)."""
    num_tokens, top_k = topk_ids.shape
    hidden = w2_weight.shape[1]
    out = torch.zeros(
        (num_tokens, top_k, hidden),
        dtype=torch.float32,
        device=intermediate_fp8.device,
    )
    expert_lo = ep_rank * num_local_experts
    expert_hi = expert_lo + num_local_experts

    intermediate_f32 = mxfp8_dequantize(intermediate_fp8, intermediate_scale)
    for token_idx in range(num_tokens):
        for choice_idx in range(top_k):
            pair_idx = token_idx * top_k + choice_idx
            expert_id = int(topk_ids[token_idx, choice_idx])
            if expert_id < expert_lo or expert_id >= expert_hi:
                continue
            local_id = expert_id - expert_lo
            w = mxfp4_dequantize_packed(
                w2_weight[local_id : local_id + 1],
                w2_scale[local_id : local_id + 1],
            )[0]
            out[token_idx, choice_idx] = (
                intermediate_f32[pair_idx].unsqueeze(0).to(torch.float32)
                @ w.T.to(torch.float32)
            ).squeeze(0)
    return out


@pytest.mark.skipif(not _has_sm12x(), reason="SM12x CUDA GPU required")
def test_sm12x_mxfp4_moe_w2_tensorcore_matches_reference_small():
    """Small-shape correctness: 2 tokens, top_k=4, 8 experts, intermediate=32, hidden=64."""
    import tokenspeed_kernel.thirdparty.cuda.sm12x_mxfp4 as sm12x_native

    torch.manual_seed(20260516)
    device = torch.device("cuda")
    num_tokens = 2
    top_k = 4
    num_local_experts = 8
    intermediate = 32
    hidden = 64

    intermediate_fp32 = (
        torch.randn(num_tokens * top_k, intermediate, device=device) * 0.5
    )
    intermediate_fp8, intermediate_scale = _quantize_mxfp8_ue8m0(intermediate_fp32)

    topk_ids = torch.tensor(
        [[0, 3, 1, 5], [2, 7, 4, 6]], dtype=torch.int32, device=device
    )

    values = torch.tensor([0.0, 0.5, -0.5, 1.0, -1.0, 1.5, -1.5, 2.0], device=device)
    w2_dense = values[
        torch.arange(num_local_experts * hidden * intermediate, device=device).reshape(
            num_local_experts, hidden, intermediate
        )
        % values.numel()
    ].float()
    w2_weight = _pack_dense(w2_dense)
    w2_scale = torch.full(
        (num_local_experts, hidden, intermediate // 32),
        127,
        dtype=torch.uint8,
        device=device,
    )

    actual = sm12x_native.sm12x_mxfp4_moe_w2_tensorcore(
        intermediate_fp8,
        intermediate_scale,
        topk_ids,
        w2_weight,
        w2_scale,
        ep_rank=0,
    )
    expected = _moe_w2_tensorcore_reference(
        intermediate_fp8,
        intermediate_scale,
        topk_ids,
        w2_weight,
        w2_scale,
        ep_rank=0,
        num_local_experts=num_local_experts,
    )
    torch.cuda.synchronize()
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)


@pytest.mark.skipif(not _has_sm12x(), reason="SM12x CUDA GPU required")
def test_sm12x_mxfp4_moe_weighted_reduce_matches_torch():
    """Weighted reduce: sum top_k contributions weighted by topk_weights."""
    import tokenspeed_kernel.thirdparty.cuda.sm12x_mxfp4 as sm12x_native

    torch.manual_seed(20260517)
    device = torch.device("cuda")
    num_tokens = 4
    top_k = 6
    hidden = 192
    per_pair = torch.randn(num_tokens, top_k, hidden, device=device)
    topk_weights = torch.rand(num_tokens, top_k, device=device)
    topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

    expected_bf16 = (
        (topk_weights.unsqueeze(-1) * per_pair).sum(dim=1).to(torch.bfloat16)
    )
    actual_bf16 = sm12x_native.sm12x_mxfp4_moe_weighted_reduce(
        per_pair, topk_weights, dtype=torch.bfloat16
    )
    torch.cuda.synchronize()
    torch.testing.assert_close(
        actual_bf16.float(), expected_bf16.float(), rtol=1e-3, atol=1e-3
    )

    # fp32 path: exact match modulo accumulation order.
    expected_f32 = (topk_weights.unsqueeze(-1) * per_pair).sum(dim=1)
    actual_f32 = sm12x_native.sm12x_mxfp4_moe_weighted_reduce(
        per_pair, topk_weights, dtype=torch.float32
    )
    torch.testing.assert_close(actual_f32, expected_f32, rtol=1e-5, atol=1e-5)


@pytest.mark.skipif(not _has_sm12x(), reason="SM12x CUDA GPU required")
def test_sm12x_mxfp4_moe_forward_tensorcore_matches_warp_zero_bias():
    """End-to-end tensorcore forward vs warp baseline at DSv4-Flash decode shape.

    Uses zero w13/w2 biases since the tensorcore prototype does not yet fuse
    in a non-zero w2_bias. Comparison tolerance is loose to allow for FP4
    quant + tensorcore FP32 accumulation differences vs the warp's scalar
    dequant + manual GEMV path.
    """
    import tokenspeed_kernel.ops.moe.sm12x_mxfp4 as sm12x_mxfp4

    torch.manual_seed(20260518)
    device = torch.device("cuda")
    num_tokens = 2
    top_k = 6
    num_local_experts = 8
    hidden = 4096
    intermediate = 2048

    hidden_states = (
        torch.randn(num_tokens, hidden, dtype=torch.float32, device=device) * 0.01
    ).to(torch.bfloat16)
    topk_weights = torch.rand(num_tokens, top_k, dtype=torch.float32, device=device)
    topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    topk_ids = torch.randint(
        0, num_local_experts, (num_tokens, top_k), dtype=torch.int32, device=device
    )

    w13_weight = torch.randint(
        0,
        256,
        (num_local_experts, 2 * intermediate, hidden // 2),
        dtype=torch.uint8,
        device=device,
    )
    w13_scale = torch.full(
        (num_local_experts, 2 * intermediate, hidden // 32),
        120,
        dtype=torch.uint8,
        device=device,
    )
    w2_weight = torch.randint(
        0,
        256,
        (num_local_experts, hidden, intermediate // 2),
        dtype=torch.uint8,
        device=device,
    )
    w2_scale = torch.full(
        (num_local_experts, hidden, intermediate // 32),
        120,
        dtype=torch.uint8,
        device=device,
    )

    warp_out = sm12x_mxfp4.sm12x_mxfp4_moe_forward_warp(
        hidden_states,
        topk_weights,
        topk_ids,
        w13_weight,
        w13_scale,
        w2_weight,
        w2_scale,
        activation="swiglu",
        swiglu_alpha=1.702,
        swiglu_limit=7.0,
        ep_rank=0,
        ep_size=1,
    )
    tc_out = sm12x_mxfp4.sm12x_mxfp4_moe_forward_tensorcore(
        hidden_states,
        topk_weights,
        topk_ids,
        w13_weight,
        w13_scale,
        w2_weight,
        w2_scale,
        activation="swiglu",
        swiglu_alpha=1.702,
        swiglu_limit=7.0,
        ep_rank=0,
        ep_size=1,
    )
    torch.cuda.synchronize()
    torch.testing.assert_close(tc_out.float(), warp_out.float(), rtol=5e-2, atol=5e-2)


@pytest.mark.skipif(not _has_sm12x(), reason="SM12x CUDA GPU required")
def test_sm12x_mxfp4_moe_w2_tensorcore_fuses_w2_bias():
    """The W2 tensorcore kernel fuses ``w2_bias`` at the final write.

    Two checks:

    1. Passing a non-zero ``w2_bias`` flips the tensorcore output by the
       same per-expert offset as the reference computes.
    2. ``w2_bias=None`` matches the older no-bias behaviour (regression
       guard for the optional parameter).
    """
    import tokenspeed_kernel.thirdparty.cuda.sm12x_mxfp4 as sm12x_native

    torch.manual_seed(20260520)
    device = torch.device("cuda")
    num_tokens = 2
    top_k = 4
    num_local_experts = 8
    intermediate = 32
    hidden = 64

    intermediate_fp32 = (
        torch.randn(num_tokens * top_k, intermediate, device=device) * 0.5
    )
    intermediate_fp8, intermediate_scale = _quantize_mxfp8_ue8m0(intermediate_fp32)
    topk_ids = torch.tensor(
        [[0, 3, 1, 5], [2, 7, 4, 6]], dtype=torch.int32, device=device
    )
    values = torch.tensor([0.0, 0.5, -0.5, 1.0, -1.0, 1.5, -1.5, 2.0], device=device)
    w2_dense = values[
        torch.arange(num_local_experts * hidden * intermediate, device=device).reshape(
            num_local_experts, hidden, intermediate
        )
        % values.numel()
    ].float()
    w2_weight = _pack_dense(w2_dense)
    w2_scale = torch.full(
        (num_local_experts, hidden, intermediate // 32),
        127,
        dtype=torch.uint8,
        device=device,
    )

    # Per-(local_expert, channel) bias with a deterministic non-zero pattern.
    w2_bias = (
        torch.arange(
            num_local_experts * hidden, device=device, dtype=torch.float32
        ).reshape(num_local_experts, hidden)
        * 0.01
    )
    w2_bias = w2_bias - w2_bias.mean(dim=1, keepdim=True)  # zero-mean per expert

    no_bias = sm12x_native.sm12x_mxfp4_moe_w2_tensorcore(
        intermediate_fp8,
        intermediate_scale,
        topk_ids,
        w2_weight,
        w2_scale,
        ep_rank=0,
    )
    with_bias = sm12x_native.sm12x_mxfp4_moe_w2_tensorcore(
        intermediate_fp8,
        intermediate_scale,
        topk_ids,
        w2_weight,
        w2_scale,
        w2_bias=w2_bias,
        ep_rank=0,
    )
    torch.cuda.synchronize()

    # Expected: each pair's per_pair_out gets ``+ w2_bias[expert(pair), :]``
    # for the local expert it routed to. Non-local pairs see no W2 work and
    # no bias either (the kernel writes 0 in the non-local branch, with no
    # bias added; the reference uses the same convention).
    expected = no_bias.clone()
    for token in range(num_tokens):
        for k in range(top_k):
            expert = int(topk_ids[token, k])
            # In this single-rank ep config, every expert id is local.
            expected[token, k] += w2_bias[expert]
    torch.testing.assert_close(with_bias, expected, rtol=1e-5, atol=1e-5)


@pytest.mark.skipif(not _has_sm12x(), reason="SM12x CUDA GPU required")
def test_sm12x_mxfp4_moe_forward_tensorcore_matches_warp_with_nonzero_bias():
    """The W2 bias fuse lets the tensorcore forward match warp on a checkpoint
    with a real non-zero ``w2_bias``. This is the contract the M-aware
    ``auto`` dispatch relies on -- before the fuse, only zero-bias
    checkpoints could safely reach the tensorcore path.
    """
    import tokenspeed_kernel.ops.moe.sm12x_mxfp4 as sm12x_mxfp4

    torch.manual_seed(20260521)
    device = torch.device("cuda")
    num_tokens = 2
    top_k = 6
    num_local_experts = 8
    hidden = 4096
    intermediate = 2048

    hidden_states = (
        torch.randn(num_tokens, hidden, dtype=torch.float32, device=device) * 0.01
    ).to(torch.bfloat16)
    topk_weights = torch.rand(num_tokens, top_k, dtype=torch.float32, device=device)
    topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    topk_ids = torch.randint(
        0, num_local_experts, (num_tokens, top_k), dtype=torch.int32, device=device
    )

    w13_weight = torch.randint(
        0,
        256,
        (num_local_experts, 2 * intermediate, hidden // 2),
        dtype=torch.uint8,
        device=device,
    )
    w13_scale = torch.full(
        (num_local_experts, 2 * intermediate, hidden // 32),
        120,
        dtype=torch.uint8,
        device=device,
    )
    w2_weight = torch.randint(
        0,
        256,
        (num_local_experts, hidden, intermediate // 2),
        dtype=torch.uint8,
        device=device,
    )
    w2_scale = torch.full(
        (num_local_experts, hidden, intermediate // 32),
        120,
        dtype=torch.uint8,
        device=device,
    )
    # Real non-zero W2 bias to exercise the fuse.
    w2_bias = (
        torch.randn(num_local_experts, hidden, dtype=torch.float32, device=device) * 0.1
    )

    common = dict(
        hidden_states=hidden_states,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        w13_weight=w13_weight,
        w13_scale=w13_scale,
        w2_weight=w2_weight,
        w2_scale=w2_scale,
        w2_bias=w2_bias,
        activation="swiglu",
        swiglu_alpha=1.702,
        swiglu_limit=7.0,
        ep_rank=0,
        ep_size=1,
    )
    warp_out = sm12x_mxfp4.sm12x_mxfp4_moe_forward_warp(**common)
    tc_out = sm12x_mxfp4.sm12x_mxfp4_moe_forward_tensorcore(**common)
    torch.cuda.synchronize()
    torch.testing.assert_close(tc_out.float(), warp_out.float(), rtol=5e-2, atol=5e-2)


@pytest.mark.skipif(not _has_sm12x(), reason="SM12x CUDA GPU required")
def test_sm12x_mxfp4_moe_w13_tensorcore_matches_reference_ds4_decode_shape():
    """DSv4-Flash decode shape: 2 tokens, top_k=6, 128 local experts, hidden=4096, gate_up=4096.

    Uses random uint8 weights -- this exercises the same code paths as the
    production shape, while keeping the matrix entries bounded so the
    pure-PyTorch reference stays in the FP4 representable range.
    """
    import tokenspeed_kernel.thirdparty.cuda.sm12x_mxfp4 as sm12x_native

    torch.manual_seed(20260515)
    device = torch.device("cuda")
    num_tokens = 2
    top_k = 6
    num_local_experts = 8  # small to keep reference time reasonable
    hidden = 4096
    gate_up_dim = 4096

    hidden_fp8, hidden_scale = _quantize_mxfp8_ue8m0(
        torch.randn(num_tokens, hidden, dtype=torch.float32, device=device) * 0.5
    )
    topk_ids = torch.randint(
        0, num_local_experts, (num_tokens, top_k), dtype=torch.int32, device=device
    )
    w13_weight = torch.randint(
        0,
        256,
        (num_local_experts, gate_up_dim, hidden // 2),
        dtype=torch.uint8,
        device=device,
    )
    w13_scale = torch.full(
        (num_local_experts, gate_up_dim, hidden // 32),
        125,
        dtype=torch.uint8,
        device=device,
    )

    actual = sm12x_native.sm12x_mxfp4_moe_w13_tensorcore(
        hidden_fp8,
        hidden_scale,
        topk_ids,
        w13_weight,
        w13_scale,
        ep_rank=0,
    )
    expected = _moe_w13_tensorcore_reference(
        hidden_fp8,
        hidden_scale,
        topk_ids,
        w13_weight,
        w13_scale,
        ep_rank=0,
        num_local_experts=num_local_experts,
    )
    torch.cuda.synchronize()
    torch.testing.assert_close(actual, expected, rtol=2e-3, atol=2e-3)
