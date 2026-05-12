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

import os

import pytest
import torch


def _has_sm12x() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 12


def _cuda_einsum_available() -> bool:
    if not _has_sm12x():
        return False
    try:
        from tokenspeed_kernel.ops.attention.deepseek_v4 import (
            deepseek_v4_fp8_einsum_sm12x_cuda,
        )
    except Exception:
        return False
    return deepseek_v4_fp8_einsum_sm12x_cuda is not None


def _cuda_inv_rope_quant_available() -> bool:
    if not _has_sm12x():
        return False
    try:
        from tokenspeed_kernel.ops.attention.deepseek_v4 import (
            deepseek_v4_fused_inv_rope_fp8_quant_sm12x_cuda,
        )
    except Exception:
        return False
    return deepseek_v4_fused_inv_rope_fp8_quant_sm12x_cuda is not None


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


def _make_inv_rope_inputs(
    *,
    tokens: int,
    n_groups: int,
    heads_per_group: int,
    nope_dim: int = 448,
    rope_dim: int = 64,
    max_position: int = 32,
    seed: int = 20260513,
    device: str = "cuda",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(seed)
    head_dim = nope_dim + rope_dim
    positions = torch.randint(
        low=0, high=max_position, size=(tokens,), device=device, dtype=torch.int64
    )
    theta = torch.randn(max_position, rope_dim // 2, device=device, dtype=torch.float32)
    cos_sin_cache = torch.cat([torch.cos(theta), torch.sin(theta)], dim=-1)
    o = torch.randn(
        tokens,
        n_groups * heads_per_group,
        head_dim,
        device=device,
        dtype=torch.bfloat16,
    )
    return o, positions, cos_sin_cache


@pytest.mark.skipif(
    not _cuda_inv_rope_quant_available(),
    reason="SM12x CUDA inv-rope+quant required",
)
@pytest.mark.parametrize("tokens", [1, 2, 3, 5, 8])
def test_deepseek_v4_fused_inv_rope_fp8_quant_cuda_matches_reference(tokens):
    from tokenspeed_kernel.ops.attention.deepseek_v4 import (
        deepseek_v4_fused_inv_rope_fp8_quant_sm12x_cuda,
    )

    n_groups = 2
    heads_per_group = 2
    nope_dim = 448
    rope_dim = 64
    o, positions, cos_sin_cache = _make_inv_rope_inputs(
        tokens=tokens, n_groups=n_groups, heads_per_group=heads_per_group
    )

    actual_fp8, actual_scale = deepseek_v4_fused_inv_rope_fp8_quant_sm12x_cuda(
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


@pytest.mark.skipif(
    not _cuda_inv_rope_quant_available(),
    reason="SM12x CUDA inv-rope+quant required",
)
@pytest.mark.parametrize("tokens", [1, 2, 4, 8, 16])
def test_deepseek_v4_fused_inv_rope_fp8_quant_cuda_matches_triton(tokens):
    """CUDA and Triton implementations must produce byte-exact output.

    The dispatcher swaps between them based on env (build availability /
    launch failure recovery); the downstream einsum walks identical
    strides either way, so any drift would silently corrupt decoded
    tokens after the fallback. Use exact equality.
    """
    from tokenspeed_kernel.ops.attention.deepseek_v4 import (
        deepseek_v4_fused_inv_rope_fp8_quant_sm12x_cuda,
        deepseek_v4_fused_inv_rope_fp8_quant_triton,
    )

    if deepseek_v4_fused_inv_rope_fp8_quant_triton is None:
        pytest.skip("Triton sibling unavailable in this environment")

    n_groups = 2
    heads_per_group = 2
    o, positions, cos_sin_cache = _make_inv_rope_inputs(
        tokens=tokens, n_groups=n_groups, heads_per_group=heads_per_group
    )

    cuda_fp8, cuda_scale = deepseek_v4_fused_inv_rope_fp8_quant_sm12x_cuda(
        o,
        positions,
        cos_sin_cache,
        n_groups=n_groups,
        heads_per_group=heads_per_group,
    )
    triton_fp8, triton_scale = deepseek_v4_fused_inv_rope_fp8_quant_triton(
        o,
        positions,
        cos_sin_cache,
        n_groups=n_groups,
        heads_per_group=heads_per_group,
    )

    torch.cuda.synchronize()
    assert cuda_fp8.shape == triton_fp8.shape
    assert cuda_scale.shape == triton_scale.shape
    # CUDA-vs-Triton at exact equality: both walk the same UE8M0 ceil-log2
    # path on the same FP32 intermediate values, so quantised bytes match.
    torch.testing.assert_close(cuda_scale, triton_scale, rtol=0, atol=0)
    torch.testing.assert_close(cuda_fp8.float(), triton_fp8.float(), rtol=0, atol=0)


@pytest.mark.skipif(
    not _cuda_inv_rope_quant_available(),
    reason="SM12x CUDA inv-rope+quant required",
)
def test_deepseek_v4_fused_inv_rope_fp8_quant_cuda_handles_zero_tokens():
    from tokenspeed_kernel.ops.attention.deepseek_v4 import (
        deepseek_v4_fused_inv_rope_fp8_quant_sm12x_cuda,
    )

    n_groups = 2
    heads_per_group = 2
    nope_dim = 448
    rope_dim = 64
    head_dim = nope_dim + rope_dim
    positions = torch.empty((0,), device="cuda", dtype=torch.int64)
    cos_sin_cache = torch.empty((32, rope_dim), device="cuda", dtype=torch.float32)
    o = torch.empty(
        0,
        n_groups * heads_per_group,
        head_dim,
        device="cuda",
        dtype=torch.bfloat16,
    )
    fp8, scale = deepseek_v4_fused_inv_rope_fp8_quant_sm12x_cuda(
        o,
        positions,
        cos_sin_cache,
        n_groups=n_groups,
        heads_per_group=heads_per_group,
    )
    torch.cuda.synchronize()
    assert fp8.shape == (0, n_groups, heads_per_group * head_dim)
    assert scale.shape == (0, n_groups, head_dim * heads_per_group // 128)


@pytest.mark.skipif(
    not _cuda_inv_rope_quant_available(),
    reason="SM12x CUDA inv-rope+quant required",
)
@pytest.mark.skipif(
    os.environ.get("TOKENSPEED_ENABLE_PROJECTION_MICROBENCH") != "1",
    reason="Set TOKENSPEED_ENABLE_PROJECTION_MICROBENCH=1 to run the microbench",
)
@pytest.mark.parametrize("tokens", [1, 2, 8])
def test_deepseek_v4_fused_inv_rope_fp8_quant_cuda_decode_microbench(tokens):
    """Decode-shape microbench. Opt-in via env var."""
    from tokenspeed_kernel.ops.attention.deepseek_v4 import (
        deepseek_v4_fused_inv_rope_fp8_quant_sm12x_cuda,
        deepseek_v4_fused_inv_rope_fp8_quant_triton,
    )

    # DSv4-Flash shape: G=8 local groups × heads_per_group=4.
    n_groups = 8
    heads_per_group = 4
    o, positions, cos_sin_cache = _make_inv_rope_inputs(
        tokens=tokens, n_groups=n_groups, heads_per_group=heads_per_group
    )

    def _time(fn, *, iters=200, warmup=20):
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        for s, e in zip(starts, ends):
            s.record()
            fn()
            e.record()
        torch.cuda.synchronize()
        times = [s.elapsed_time(e) for s, e in zip(starts, ends)]
        times.sort()
        return times[0], sum(times) / len(times), times[-1]

    cuda_min, cuda_mean, cuda_max = _time(
        lambda: deepseek_v4_fused_inv_rope_fp8_quant_sm12x_cuda(
            o,
            positions,
            cos_sin_cache,
            n_groups=n_groups,
            heads_per_group=heads_per_group,
        )
    )
    print(
        f"[t={tokens}] cuda_inv_rope_quant   min={cuda_min:.3f}ms "
        f"mean={cuda_mean:.3f}ms max={cuda_max:.3f}ms"
    )
    if deepseek_v4_fused_inv_rope_fp8_quant_triton is not None:
        tri_min, tri_mean, tri_max = _time(
            lambda: deepseek_v4_fused_inv_rope_fp8_quant_triton(
                o,
                positions,
                cos_sin_cache,
                n_groups=n_groups,
                heads_per_group=heads_per_group,
            )
        )
        print(
            f"[t={tokens}] triton_inv_rope_quant min={tri_min:.3f}ms "
            f"mean={tri_mean:.3f}ms max={tri_max:.3f}ms"
        )
        print(
            f"[t={tokens}] speedup mean={tri_mean / cuda_mean:.2f}x "
            f"min={tri_min / cuda_min:.2f}x"
        )


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


def _make_einsum_inputs(
    *,
    tokens: int,
    groups: int,
    hidden: int,
    out_rank: int,
    transpose_activation: bool,
    device: str = "cuda",
    seed: int = 20260512,
    b_scales_dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, ...]:
    """Build matched (a, a_scale, b, b_scale) pairs at decode-relevant shapes.

    When ``transpose_activation`` is True, returns ``a`` / ``a_scale`` as
    non-contiguous views laid out as ``[groups, tokens, ...]`` and transposed
    to ``[tokens, groups, ...]`` -- this matches the buffer the Triton fused
    inverse-RoPE + FP8 quant kernel emits at runtime.
    """
    torch.manual_seed(seed)
    a_float = torch.randn(tokens, groups, hidden, device=device) * 0.75
    a, a_scale = _fp8_quant_reference(a_float)
    if transpose_activation:
        # Reproduce the [G, T, H] storage + transpose(0, 1) view path.
        a = a.permute(1, 0, 2).contiguous().permute(1, 0, 2)
        a_scale = a_scale.permute(1, 0, 2).contiguous().permute(1, 0, 2)

    b_float = torch.randn(groups * out_rank, hidden, device=device) * 0.25
    b = b_float.to(torch.float8_e4m3fn)
    b_scale_fp32 = torch.pow(
        2.0,
        torch.ceil(
            torch.log2(
                torch.rand(
                    groups * (out_rank // 128), hidden // 128, device=device
                ).clamp_min(1.0e-4)
            )
        ),
    ).float()
    if b_scales_dtype == torch.float32:
        b_scale = b_scale_fp32
    elif b_scales_dtype == torch.uint8:
        # Encode each scale as a UE8M0 byte: exponent_field = log2(scale) + 127.
        log_scale = torch.log2(b_scale_fp32.clamp_min(1.0e-38))
        encoded = (log_scale.round() + 127).clamp(0, 255).to(torch.uint8)
        b_scale = encoded
    else:
        raise ValueError(f"unsupported b_scales_dtype={b_scales_dtype}")
    return a, a_scale, b, b_scale, b_scale_fp32


def _einsum_reference(
    a: torch.Tensor,
    a_scale: torch.Tensor,
    b: torch.Tensor,
    b_scale_fp32: torch.Tensor,
    *,
    groups: int,
    out_rank: int,
    hidden: int,
) -> torch.Tensor:
    a_effective = a.float() * a_scale.repeat_interleave(128, dim=-1)
    b_scale_view = b_scale_fp32.view(groups, out_rank // 128, hidden // 128)
    b_effective = b.view(groups, out_rank, hidden).float()
    b_effective = b_effective * b_scale_view.repeat_interleave(
        128, dim=1
    ).repeat_interleave(128, dim=2)
    return torch.einsum("tgh,grh->tgr", a_effective, b_effective).to(torch.bfloat16)


@pytest.mark.skipif(not _cuda_einsum_available(), reason="SM12x CUDA einsum required")
def test_deepseek_v4_fp8_einsum_cuda_matches_reference():
    from tokenspeed_kernel.ops.attention.deepseek_v4 import (
        deepseek_v4_fp8_einsum_sm12x_cuda,
    )

    tokens, groups, hidden, out_rank = 4, 2, 512, 256
    a, a_scale, b, b_scale, b_scale_fp32 = _make_einsum_inputs(
        tokens=tokens,
        groups=groups,
        hidden=hidden,
        out_rank=out_rank,
        transpose_activation=False,
    )
    out = torch.empty(tokens, groups, out_rank, device="cuda", dtype=torch.bfloat16)

    deepseek_v4_fp8_einsum_sm12x_cuda(a, a_scale, b, b_scale, out)
    expected = _einsum_reference(
        a,
        a_scale,
        b,
        b_scale_fp32,
        groups=groups,
        out_rank=out_rank,
        hidden=hidden,
    )

    torch.cuda.synchronize()
    assert out.shape == expected.shape
    assert out.dtype == torch.bfloat16
    torch.testing.assert_close(out.float(), expected.float(), rtol=2e-2, atol=0.5)


@pytest.mark.skipif(not _cuda_einsum_available(), reason="SM12x CUDA einsum required")
def test_deepseek_v4_fp8_einsum_cuda_handles_strided_activation():
    """The Triton inverse-RoPE + quant kernel returns non-contig [T, G, ...]
    views over [G, T, ...] storage; the CUDA einsum must agree."""
    from tokenspeed_kernel.ops.attention.deepseek_v4 import (
        deepseek_v4_fp8_einsum_sm12x_cuda,
    )

    tokens, groups, hidden, out_rank = 4, 2, 512, 256
    a, a_scale, b, b_scale, b_scale_fp32 = _make_einsum_inputs(
        tokens=tokens,
        groups=groups,
        hidden=hidden,
        out_rank=out_rank,
        transpose_activation=True,
    )
    assert not a.is_contiguous()
    assert not a_scale.is_contiguous()
    out = torch.empty(tokens, groups, out_rank, device="cuda", dtype=torch.bfloat16)

    deepseek_v4_fp8_einsum_sm12x_cuda(a, a_scale, b, b_scale, out)
    expected = _einsum_reference(
        a,
        a_scale,
        b,
        b_scale_fp32,
        groups=groups,
        out_rank=out_rank,
        hidden=hidden,
    )

    torch.cuda.synchronize()
    torch.testing.assert_close(out.float(), expected.float(), rtol=2e-2, atol=0.5)


@pytest.mark.skipif(not _cuda_einsum_available(), reason="SM12x CUDA einsum required")
def test_deepseek_v4_fp8_einsum_cuda_accepts_ue8m0_weight_scales():
    from tokenspeed_kernel.ops.attention.deepseek_v4 import (
        deepseek_v4_fp8_einsum_sm12x_cuda,
    )

    tokens, groups, hidden, out_rank = 2, 2, 512, 256
    a, a_scale, b, b_scale_u8, b_scale_fp32 = _make_einsum_inputs(
        tokens=tokens,
        groups=groups,
        hidden=hidden,
        out_rank=out_rank,
        transpose_activation=False,
        b_scales_dtype=torch.uint8,
    )
    # Mirror what the runtime computes from the encoded byte.
    decoded = (
        (b_scale_u8.to(torch.int32) << 23)
        .view(torch.float32)
        .reshape(b_scale_fp32.shape)
    )

    out = torch.empty(tokens, groups, out_rank, device="cuda", dtype=torch.bfloat16)
    deepseek_v4_fp8_einsum_sm12x_cuda(a, a_scale, b, b_scale_u8, out)
    expected = _einsum_reference(
        a,
        a_scale,
        b,
        decoded,
        groups=groups,
        out_rank=out_rank,
        hidden=hidden,
    )

    torch.cuda.synchronize()
    torch.testing.assert_close(out.float(), expected.float(), rtol=2e-2, atol=0.5)


@pytest.mark.skipif(
    not (_cuda_einsum_available() and _has_sm12x()),
    reason="CUDA + Triton einsum both required",
)
@pytest.mark.parametrize("tokens", [1, 2, 8])
def test_deepseek_v4_fp8_einsum_cuda_matches_triton_at_decode_shape(tokens):
    """At the DSv4-Flash decode shape the CUDA path must agree with Triton
    within FP8 quant noise (rtol=2e-2, atol=0.5 -- same as the reference test)."""
    from tokenspeed_kernel.ops.attention.deepseek_v4 import (
        deepseek_v4_fp8_einsum_sm12x_cuda,
        deepseek_v4_fp8_einsum_sm12x_triton,
    )

    if deepseek_v4_fp8_einsum_sm12x_triton is None:
        pytest.skip("Triton einsum unavailable in this environment")

    groups = 8
    hidden = 2048
    out_rank = 1024
    a, a_scale, b, b_scale, _ = _make_einsum_inputs(
        tokens=tokens,
        groups=groups,
        hidden=hidden,
        out_rank=out_rank,
        transpose_activation=True,
    )

    out_cuda = torch.empty(
        tokens, groups, out_rank, device="cuda", dtype=torch.bfloat16
    )
    out_triton = torch.empty_like(out_cuda)
    deepseek_v4_fp8_einsum_sm12x_cuda(a, a_scale, b, b_scale, out_cuda)
    deepseek_v4_fp8_einsum_sm12x_triton(a, a_scale, b, b_scale, out_triton)

    torch.cuda.synchronize()
    # Both implementations share an FP8 quant origin; the residual gap is the
    # FMA-ordering difference between the two reduction schedules (Triton
    # block-dot vs CUDA scalar-accumulate). At hidden=2048 the bf16 output
    # magnitude can reach ~1e2, where bf16 ULP is ~0.4, so a few-ULP slack is
    # expected.
    torch.testing.assert_close(
        out_cuda.float(), out_triton.float(), rtol=5e-3, atol=0.2
    )


@pytest.mark.skipif(not _cuda_einsum_available(), reason="SM12x CUDA einsum required")
def test_deepseek_v4_fp8_einsum_cuda_handles_zero_tokens():
    from tokenspeed_kernel.ops.attention.deepseek_v4 import (
        deepseek_v4_fp8_einsum_sm12x_cuda,
    )

    groups, hidden, out_rank = 2, 256, 128
    a = torch.empty(0, groups, hidden, device="cuda", dtype=torch.float8_e4m3fn)
    a_scale = torch.empty(0, groups, hidden // 128, device="cuda", dtype=torch.float32)
    b = torch.empty(groups, out_rank, hidden, device="cuda", dtype=torch.float8_e4m3fn)
    b_scale = torch.ones(
        groups, out_rank // 128, hidden // 128, device="cuda", dtype=torch.float32
    )
    out = torch.empty(0, groups, out_rank, device="cuda", dtype=torch.bfloat16)
    # Must complete without launching a kernel and without raising.
    deepseek_v4_fp8_einsum_sm12x_cuda(a, a_scale, b, b_scale, out)
    torch.cuda.synchronize()


@pytest.mark.skipif(not _cuda_einsum_available(), reason="SM12x CUDA einsum required")
@pytest.mark.skipif(
    os.environ.get("TOKENSPEED_ENABLE_PROJECTION_MICROBENCH") != "1",
    reason="Set TOKENSPEED_ENABLE_PROJECTION_MICROBENCH=1 to run the microbench",
)
@pytest.mark.parametrize("tokens", [1, 2, 8])
def test_deepseek_v4_fp8_einsum_cuda_decode_microbench(tokens):
    """Decode-shape microbench. Opt-in via env var.

    Prints min/mean/max ms/call so it can be captured into the workstation
    runbook's microbench bundle. The bench loops in-process; no Python-level
    fixtures pollute the timed window.
    """
    from tokenspeed_kernel.ops.attention.deepseek_v4 import (
        deepseek_v4_fp8_einsum_sm12x_cuda,
        deepseek_v4_fp8_einsum_sm12x_triton,
    )

    groups = 8
    hidden = 2048
    out_rank = 1024
    a, a_scale, b, b_scale, _ = _make_einsum_inputs(
        tokens=tokens,
        groups=groups,
        hidden=hidden,
        out_rank=out_rank,
        transpose_activation=True,
    )
    out = torch.empty(tokens, groups, out_rank, device="cuda", dtype=torch.bfloat16)

    def _time(fn, *, iters=200, warmup=20):
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        for s, e in zip(starts, ends):
            s.record()
            fn()
            e.record()
        torch.cuda.synchronize()
        times = [s.elapsed_time(e) for s, e in zip(starts, ends)]
        times.sort()
        return times[0], sum(times) / len(times), times[-1]

    cuda_min, cuda_mean, cuda_max = _time(
        lambda: deepseek_v4_fp8_einsum_sm12x_cuda(a, a_scale, b, b_scale, out)
    )
    print(
        f"[t={tokens}] cuda_einsum   min={cuda_min:.3f}ms "
        f"mean={cuda_mean:.3f}ms max={cuda_max:.3f}ms"
    )

    if deepseek_v4_fp8_einsum_sm12x_triton is not None:
        tri_min, tri_mean, tri_max = _time(
            lambda: deepseek_v4_fp8_einsum_sm12x_triton(a, a_scale, b, b_scale, out)
        )
        print(
            f"[t={tokens}] triton_einsum min={tri_min:.3f}ms "
            f"mean={tri_mean:.3f}ms max={tri_max:.3f}ms"
        )
        print(
            f"[t={tokens}] speedup mean={tri_mean / cuda_mean:.2f}x "
            f"min={tri_min / cuda_min:.2f}x"
        )
