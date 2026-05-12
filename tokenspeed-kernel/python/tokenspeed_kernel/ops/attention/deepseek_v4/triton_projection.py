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

"""SM12x DeepSeek V4 attention output projection kernels."""

from __future__ import annotations

import torch
from tokenspeed_kernel._triton import tl, triton
from tokenspeed_kernel.platform import ArchVersion, CapabilityRequirement
from tokenspeed_kernel.registry import Priority, register_kernel


@triton.jit
def _fused_inv_rope_fp8_quant_kernel(
    o_ptr,
    positions_ptr,
    cos_sin_cache_ptr,
    fp8_ptr,
    scale_ptr,
    num_tokens,
    heads_per_group: tl.constexpr,
    o_stride_token: tl.constexpr,
    o_stride_head: tl.constexpr,
    cache_stride_pos: tl.constexpr,
    fp8_stride_group: tl.constexpr,
    fp8_stride_token: tl.constexpr,
    scale_stride_group: tl.constexpr,
    scale_stride_block: tl.constexpr,
    fp8_max: tl.constexpr,
    eps: tl.constexpr,
    QUANT_GROUP_SIZE: tl.constexpr,
    CHUNKS_PER_HEAD: tl.constexpr,
    ROPE_START: tl.constexpr,
    HALF_ROPE: tl.constexpr,
) -> None:
    token_id = tl.program_id(0).to(tl.int64)
    group_head = tl.program_id(1).to(tl.int64)

    group = group_head // heads_per_group
    head_in_group = group_head % heads_per_group
    scale_block_start = head_in_group * CHUNKS_PER_HEAD

    block_offsets = tl.arange(0, CHUNKS_PER_HEAD)
    scale_blocks = scale_block_start + block_offsets
    if token_id >= num_tokens:
        scale_addrs = (
            scale_ptr
            + group * scale_stride_group
            + token_id
            + scale_blocks * scale_stride_block
        )
        tl.store(scale_addrs, tl.zeros((CHUNKS_PER_HEAD,), dtype=tl.float32))
        return

    HEAD_DIM: tl.constexpr = CHUNKS_PER_HEAD * QUANT_GROUP_SIZE
    offsets = tl.arange(0, HEAD_DIM)
    input_base = o_ptr + token_id * o_stride_token + group_head * o_stride_head
    x = tl.load(input_base + offsets).to(tl.float32)

    rope_abs_start: tl.constexpr = (CHUNKS_PER_HEAD - 1) * QUANT_GROUP_SIZE + ROPE_START
    pos = tl.load(positions_ptr + token_id)
    cache_base = cos_sin_cache_ptr + pos * cache_stride_pos
    is_rope = offsets >= rope_abs_start
    rope_local = offsets - rope_abs_start

    partner = tl.load(input_base + (offsets ^ 1), mask=is_rope, other=0.0).to(
        tl.float32
    )
    cs_idx = tl.maximum(rope_local >> 1, 0)
    cos_v = tl.load(cache_base + cs_idx, mask=is_rope, other=1.0)
    sin_v = tl.load(cache_base + HALF_ROPE + cs_idx, mask=is_rope, other=0.0)
    even_rot = x * cos_v + partner * sin_v
    odd_rot = x * cos_v - partner * sin_v
    x = tl.where(is_rope, tl.where((rope_local & 1) == 0, even_rot, odd_rot), x)

    x_blocks = tl.reshape(tl.abs(x), (CHUNKS_PER_HEAD, QUANT_GROUP_SIZE))
    block_absmax = tl.maximum(tl.max(x_blocks, axis=1), eps)
    scale = tl.math.exp2(tl.ceil(tl.log2(block_absmax * (1.0 / fp8_max))))
    scale_matrix = tl.reshape(
        tl.broadcast_to(
            tl.reshape(scale, (CHUNKS_PER_HEAD, 1)),
            (CHUNKS_PER_HEAD, QUANT_GROUP_SIZE),
        ),
        (HEAD_DIM,),
    )
    x_quant = tl.clamp(x / scale_matrix, -fp8_max, fp8_max).to(tl.float8e4nv)

    fp8_base = (
        fp8_ptr
        + group * fp8_stride_group
        + token_id * fp8_stride_token
        + scale_block_start * QUANT_GROUP_SIZE
    )
    tl.store(fp8_base + offsets, x_quant)

    scale_addrs = (
        scale_ptr
        + group * scale_stride_group
        + token_id
        + scale_blocks * scale_stride_block
    )
    tl.store(scale_addrs, scale)


def _aligned_tokens(num_tokens: int) -> int:
    return ((num_tokens + 3) // 4) * 4


@register_kernel(
    "attention",
    "deepseek_v4_fused_inv_rope_fp8_quant",
    name="deepseek_v4_fused_inv_rope_fp8_quant_triton",
    solution="triton",
    capability=CapabilityRequirement(
        min_arch_version=ArchVersion(12, 0),
        max_arch_version=ArchVersion(12, 1),
        vendors=frozenset({"nvidia"}),
    ),
    dtypes={torch.bfloat16, torch.float16},
    priority=Priority.PERFORMANT,
    tags={"triton", "deepseek_v4", "fp8", "projection", "sm12x"},
)
def deepseek_v4_fused_inv_rope_fp8_quant_triton(
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
    """Inverse-RoPE attention output and quantize it for grouped FP8 einsum.

    Returns ``o_fp8`` and ``o_scale`` with logical shapes
    ``[tokens, groups, heads_per_group * head_dim]`` and
    ``[tokens, groups, hidden / quant_group_size]``.
    """

    num_tokens, num_heads, head_dim = o.shape
    if num_tokens == 0:
        hidden = heads_per_group * head_dim
        return (
            torch.empty(
                (0, n_groups, hidden),
                dtype=torch.float8_e4m3fn,
                device=o.device,
            ),
            torch.empty(
                (0, n_groups, hidden // quant_group_size),
                dtype=torch.float32,
                device=o.device,
            ),
        )
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
    if cos_sin_cache.dtype != torch.float32:
        raise ValueError("cos_sin_cache must be float32")

    hidden = heads_per_group * head_dim
    scale_blocks = hidden // quant_group_size
    chunks_per_head = head_dim // quant_group_size
    aligned_tokens = _aligned_tokens(num_tokens)
    fp8_buf = torch.empty(
        (n_groups, num_tokens, hidden),
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
    grid = (aligned_tokens, num_heads)
    _fused_inv_rope_fp8_quant_kernel[grid](
        o,
        positions,
        cos_sin_cache,
        fp8_buf,
        scale_buf,
        num_tokens,
        heads_per_group=heads_per_group,
        o_stride_token=o.stride(0),
        o_stride_head=o.stride(1),
        cache_stride_pos=cos_sin_cache.stride(0),
        fp8_stride_group=fp8_buf.stride(0),
        fp8_stride_token=fp8_buf.stride(1),
        scale_stride_group=scale_buf.stride(0),
        scale_stride_block=scale_buf.stride(2),
        fp8_max=torch.finfo(torch.float8_e4m3fn).max,
        eps=1.0e-10,
        QUANT_GROUP_SIZE=quant_group_size,
        CHUNKS_PER_HEAD=chunks_per_head,
        ROPE_START=nope_dim % quant_group_size,
        HALF_ROPE=rope_dim // 2,
        num_warps=1,
        num_stages=1,
    )
    return fp8_buf.transpose(0, 1), scale_buf.transpose(0, 1)


def _upcast_e8m0_to_fp32(scale: torch.Tensor) -> torch.Tensor:
    exponent = scale.view(torch.uint8).to(torch.int32)
    return (exponent << 23).view(torch.float32)


@triton.jit
def _fp8_einsum_bhr_hdr_to_bhd_kernel(
    a_ptr,
    a_scale_ptr,
    b_ptr,
    b_scale_ptr,
    out_ptr,
    num_tokens: tl.constexpr,
    num_groups: tl.constexpr,
    out_rank: tl.constexpr,
    hidden_size: tl.constexpr,
    a_stride_token: tl.constexpr,
    a_stride_group: tl.constexpr,
    a_stride_hidden: tl.constexpr,
    a_scale_stride_token: tl.constexpr,
    a_scale_stride_group: tl.constexpr,
    a_scale_stride_hidden: tl.constexpr,
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
            + (out_offsets // BLOCK_OUT) * b_scale_stride_out
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


def _reshape_wo_a_weight(
    b: torch.Tensor,
    b_scale: torch.Tensor,
    *,
    num_groups: int,
    out_rank: int,
    hidden_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
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


@register_kernel(
    "attention",
    "deepseek_v4_fp8_einsum",
    name="deepseek_v4_fp8_einsum_sm12x_triton",
    solution="triton",
    capability=CapabilityRequirement(
        min_arch_version=ArchVersion(12, 0),
        max_arch_version=ArchVersion(12, 1),
        vendors=frozenset({"nvidia"}),
    ),
    dtypes={torch.float8_e4m3fn},
    priority=Priority.PERFORMANT,
    tags={"triton", "deepseek_v4", "fp8", "projection", "sm12x"},
)
def deepseek_v4_fp8_einsum_sm12x_triton(
    a: torch.Tensor,
    a_scale: torch.Tensor,
    b: torch.Tensor,
    b_scale: torch.Tensor,
    out: torch.Tensor,
) -> None:
    """Compute DeepSeek V4 ``bhr,hdr->bhd`` for SM12x FP8 tensors."""

    num_tokens, num_groups, hidden_size = a.shape
    if num_tokens == 0:
        return
    out_tokens, out_groups, out_rank = out.shape
    if (out_tokens, out_groups) != (num_tokens, num_groups):
        raise ValueError(
            f"out shape {tuple(out.shape)} does not match a shape {tuple(a.shape)}"
        )
    if hidden_size % 128 != 0:
        raise ValueError(f"hidden_size={hidden_size} must be divisible by 128")
    if out_rank % 128 != 0:
        raise ValueError(f"out_rank={out_rank} must be divisible by 128")
    if a.dtype != torch.float8_e4m3fn or b.dtype != torch.float8_e4m3fn:
        raise ValueError("a and b must use torch.float8_e4m3fn")
    if a_scale.dtype != torch.float32:
        raise ValueError("a_scale must use torch.float32")

    e8m0_dtype = getattr(torch, "float8_e8m0fnu", None)
    if b_scale.dtype == e8m0_dtype:
        b_scale = _upcast_e8m0_to_fp32(b_scale)
    if b_scale.dtype != torch.float32:
        raise ValueError("b_scale must use torch.float32 or torch.float8_e8m0fnu")

    b, b_scale = _reshape_wo_a_weight(
        b,
        b_scale,
        num_groups=num_groups,
        out_rank=out_rank,
        hidden_size=hidden_size,
    )
    block_tokens = 16
    block_out = 128
    block_hidden = 128
    grid = (
        triton.cdiv(num_tokens, block_tokens),
        triton.cdiv(out_rank, block_out),
        num_groups,
    )
    _fp8_einsum_bhr_hdr_to_bhd_kernel[grid](
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
