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

"""Activation kernel entry points."""

from __future__ import annotations

import functools

import torch


@functools.cache
def _resolve_silu_and_mul(dtype: torch.dtype):
    """Pick the fused ``silu_and_mul`` backend for ``dtype`` once.

    Prefers FlashInfer's kernel for the 16-bit activation dtypes it supports
    (bf16/fp16) — the production DSv4 path — benchmarked ~6x over the unfused
    ``F.silu(gate) * up`` path at SM120 shapes and bit-exact against its fp32
    reference. Falls back to the in-package Triton kernel (which also
    accumulates in fp32 and additionally supports fp32 input) otherwise, or when
    FlashInfer is unavailable or exposes a surprising signature. Both backends
    interpret the input as ``[..., 2 * D]`` (gate first, up second), take the
    input as the first positional argument, and return a tensor of the input
    dtype. The FlashInfer signature is probed so an out-buffer-first arg order
    falls back to Triton rather than silently producing wrong output.
    """
    import inspect

    from tokenspeed_kernel.ops.activation import flashinfer as _flashinfer
    from tokenspeed_kernel.ops.activation import triton as _triton
    from tokenspeed_kernel.registry import error_fn

    if dtype in (torch.bfloat16, torch.float16):
        fi = _flashinfer.silu_and_mul
        if fi is not error_fn:
            try:
                first = next(iter(inspect.signature(fi).parameters))
                if first not in ("out", "output", "output_tensor"):
                    return fi
            except (ValueError, TypeError, StopIteration):
                pass
    return _triton.silu_and_mul


def fused_silu_and_mul(
    gate_up: torch.Tensor, *, output_dtype: torch.dtype
) -> torch.Tensor:
    """Fused SwiGLU ``SiLU(gate_up[..., :D]) * gate_up[..., D:]`` -> ``[..., D]``.

    Single-pass fused replacement for the unfused ``(F.silu(gate) * up)`` path.
    ``gate_up`` is ``[..., 2 * D]`` with gate values first and up values second.
    Accumulation happens in fp32 in both backends, matching the reference
    ``F.silu(gate.float()) * up.float()``. The result is cast to ``output_dtype``
    only when the backend output dtype differs, avoiding a redundant copy on the
    common bf16-in/bf16-out path.
    """
    out = _resolve_silu_and_mul(gate_up.dtype)(gate_up)
    return out if out.dtype == output_dtype else out.to(output_dtype)
