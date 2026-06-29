# Copyright (c) 2026 LightSeek Foundation
#
# Portions adapted from the vLLM project (Apache-2.0): the SM12x Triton FP8
# einsum kernel for the DeepSeek-V4 output projection.

"""Consumer-Blackwell (sm_120/sm_121) Triton FP8 einsum for DeepSeek-V4 o_proj.

deep_gemm's grouped fp8 einsum (`bhr,hdr->bhd`) and its UE8M0 scale-factor
layout transform are datacenter-Blackwell-only; on consumer Blackwell they
assert "Unsupported architecture" / "Unknown SF transformation". This module
keeps the on-disk FP32 block-[128,128] scales and runs the grouped einsum in a
portable Triton kernel instead.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


# num_tokens and the token-count-dependent strides are RUNTIME args marked
# do_not_specialize, so the kernel compiles to a single cubin for every prefill
# chunk size (a fresh JIT mid-CUDA-graph-capture raises cudaErrorNotPermitted).
@triton.jit(
    do_not_specialize=[
        "num_tokens",
        "a_stride_group",
        "a_scale_stride_group",
        "a_scale_stride_hidden",
    ]
)
def _deepseek_v4_sm12x_fp8_einsum_kernel(
    a_ptr,
    a_scale_ptr,
    b_ptr,
    b_scale_ptr,
    out_ptr,
    num_tokens,
    num_groups: tl.constexpr,
    out_rank: tl.constexpr,
    hidden_size: tl.constexpr,
    a_stride_token: tl.constexpr,
    a_stride_group,
    a_stride_hidden: tl.constexpr,
    a_scale_stride_token: tl.constexpr,
    a_scale_stride_group,
    a_scale_stride_hidden,
    b_stride_group: tl.constexpr,
    b_stride_out: tl.constexpr,
    b_stride_hidden: tl.constexpr,
    b_scale_stride_group: tl.constexpr,
    b_scale_stride_out: tl.constexpr,
    b_scale_stride_hidden: tl.constexpr,
    out_stride_token: tl.constexpr,
    out_stride_group: tl.constexpr,
    out_stride_rank: tl.constexpr,
    BLOCK_TOKENS: tl.constexpr,
    BLOCK_OUT: tl.constexpr,
    BLOCK_HIDDEN: tl.constexpr,
) -> None:
    token_block = tl.program_id(0)
    out_block = tl.program_id(1)
    group = tl.program_id(2)

    token_offsets = token_block * BLOCK_TOKENS + tl.arange(0, BLOCK_TOKENS)
    out_offsets = out_block * BLOCK_OUT + tl.arange(0, BLOCK_OUT)
    hidden_offsets = tl.arange(0, BLOCK_HIDDEN)
    accum = tl.zeros((BLOCK_TOKENS, BLOCK_OUT), dtype=tl.float32)

    for hidden_start in range(0, hidden_size, BLOCK_HIDDEN):
        hidden = hidden_start + hidden_offsets
        a = tl.load(
            a_ptr
            + token_offsets[:, None] * a_stride_token
            + group * a_stride_group
            + hidden[None, :] * a_stride_hidden,
            mask=(token_offsets[:, None] < num_tokens)
            & (hidden[None, :] < hidden_size),
            other=0.0,
        )
        b = tl.load(
            b_ptr
            + group * b_stride_group
            + out_offsets[None, :] * b_stride_out
            + hidden[:, None] * b_stride_hidden,
            mask=(out_offsets[None, :] < out_rank) & (hidden[:, None] < hidden_size),
            other=0.0,
        )
        raw = tl.dot(a, b, out_dtype=tl.float32)
        hidden_scale_block = hidden_start // BLOCK_HIDDEN
        a_scale = tl.load(
            a_scale_ptr
            + token_offsets * a_scale_stride_token
            + group * a_scale_stride_group
            + hidden_scale_block * a_scale_stride_hidden,
            mask=token_offsets < num_tokens,
            other=0.0,
        )
        b_scale = tl.load(
            b_scale_ptr
            + group * b_scale_stride_group
            + (out_offsets // 128) * b_scale_stride_out
            + hidden_scale_block * b_scale_stride_hidden,
            mask=out_offsets < out_rank,
            other=0.0,
        )
        accum += raw * a_scale[:, None] * b_scale[None, :]

    tl.store(
        out_ptr
        + token_offsets[:, None] * out_stride_token
        + group * out_stride_group
        + out_offsets[None, :] * out_stride_rank,
        accum,
        mask=(token_offsets[:, None] < num_tokens) & (out_offsets[None, :] < out_rank),
    )


def _to_fp32_scale(s: torch.Tensor) -> torch.Tensor:
    if s.dtype == torch.float32:
        return s
    e8m0 = getattr(torch, "float8_e8m0fnu", None)
    if e8m0 is not None and s.dtype == e8m0:
        return s.to(torch.float32)
    return s.to(torch.float32)


def deepseek_v4_sm12x_fp8_einsum(
    a: torch.Tensor,
    a_scale: torch.Tensor,
    b: torch.Tensor,
    b_scale: torch.Tensor,
    out: torch.Tensor,
) -> None:
    """Compute ``bhr,hdr->bhd`` with FP32 block scales on consumer Blackwell.

    Args:
        a: inv-rope FP8 output, shape ``[tokens, groups, hidden]``.
        a_scale: FP32 per-token block scales ``[tokens, groups, hidden/128]``.
        b: ``wo_a`` reshaped to ``[groups, out_rank, hidden]`` (FP8).
        b_scale: FP32 weight block scales ``[groups, out_rank/128, hidden/128]``.
        out: bf16 output ``[tokens, groups, out_rank]``.
    """
    num_tokens, num_groups, hidden_size = a.shape
    b_groups, out_rank, b_hidden_size = b.shape
    assert b_groups == num_groups
    assert b_hidden_size == hidden_size
    assert out.shape == (num_tokens, num_groups, out_rank)
    assert hidden_size % 128 == 0
    assert out_rank % 128 == 0
    assert a.dtype == torch.float8_e4m3fn
    assert b.dtype == torch.float8_e4m3fn
    a_scale = _to_fp32_scale(a_scale)
    b_scale = _to_fp32_scale(b_scale)

    if num_tokens == 0:
        return

    block_tokens = 16
    block_out = 128
    block_hidden = 128
    grid = (
        triton.cdiv(num_tokens, block_tokens),
        triton.cdiv(out_rank, block_out),
        num_groups,
    )
    _deepseek_v4_sm12x_fp8_einsum_kernel[grid](
        a,
        a_scale,
        b,
        b_scale,
        out,
        num_tokens,
        num_groups,
        out_rank,
        hidden_size,
        a.stride(0),
        a.stride(1),
        a.stride(2),
        a_scale.stride(0),
        a_scale.stride(1),
        a_scale.stride(2),
        b.stride(0),
        b.stride(1),
        b.stride(2),
        b_scale.stride(0),
        b_scale.stride(1),
        b_scale.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        BLOCK_TOKENS=block_tokens,
        BLOCK_OUT=block_out,
        BLOCK_HIDDEN=block_hidden,
        num_warps=4,
        num_stages=3,
    )


def deepseek_v4_o_proj_fp8_einsum_config(capability_major: int) -> tuple[tuple[int, int, int], bool]:
    """Return ``(recipe, tma_aligned_scales)`` for the o_proj einsum.

    Datacenter Blackwell (major 10) keeps the deep_gemm INT32 UE8M0 path
    ``(1, 1, 128)``; consumer Blackwell / Thor / Hopper use the FP32 block-scale
    layout ``(1, 128, 128)`` consumed by the Triton kernel above.
    """
    if capability_major == 10:
        return (1, 1, 128), True
    return (1, 128, 128), False


def deepseek_v4_o_proj_einsum(
    o_fp8: torch.Tensor,
    o_scale: torch.Tensor,
    weight_3d: torch.Tensor,
    weight_scale: torch.Tensor,
    z: torch.Tensor,
    recipe: tuple[int, int, int],
    deep_gemm,
) -> None:
    """o_proj ``bhr,hdr->bhd``. Triton on consumer Blackwell, deep_gemm on DC.

    ``o_fp8``/``o_scale`` are tokenspeed inv-rope outputs laid out logically as
    ``[tokens, groups, hidden]`` / ``[tokens, groups, hidden/128]`` (physically
    group-major; the einsum kernel indexes them via strides, so no transpose is
    needed). ``weight_3d`` is ``wo_a`` ``[groups, out_rank, hidden]``; ``z`` is
    ``[tokens, groups, out_rank]``.
    """
    if tuple(recipe) == (1, 128, 128):
        g, out_rank, hidden = weight_3d.shape
        b_scale = weight_scale.reshape(g, out_rank // 128, hidden // 128)
        deepseek_v4_sm12x_fp8_einsum(o_fp8, o_scale, weight_3d, b_scale, z)
    else:
        deep_gemm.fp8_einsum(
            "bhr,hdr->bhd",
            (o_fp8, o_scale),
            (weight_3d, weight_scale),
            z,
            recipe=tuple(recipe),
        )
