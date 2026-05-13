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

_KERNEL_EXPORTS = {
    "mha_decode_with_kvcache": (
        "tokenspeed_kernel.ops.attention",
        "mha_decode_with_kvcache",
    ),
    "mha_prefill": ("tokenspeed_kernel.ops.attention", "mha_prefill"),
    "mha_prefill_with_kvcache": (
        "tokenspeed_kernel.ops.attention",
        "mha_prefill_with_kvcache",
    ),
    "mm": ("tokenspeed_kernel.ops.gemm", "mm"),
    "moe_combine": ("tokenspeed_kernel.ops.moe", "moe_combine"),
    "moe_dispatch": ("tokenspeed_kernel.ops.moe", "moe_dispatch"),
    "moe_experts": ("tokenspeed_kernel.ops.moe", "moe_experts"),
    "moe_fused": ("tokenspeed_kernel.ops.moe", "moe_fused"),
    "moe_route": ("tokenspeed_kernel.ops.moe", "moe_route"),
}


def _missing_optional_triton(exc: ModuleNotFoundError) -> bool:
    return exc.name in {"tokenspeed_triton", "triton"}


try:
    from tokenspeed_kernel.profiling import bootstrap_profiling_from_env
except ModuleNotFoundError as exc:
    if not _missing_optional_triton(exc):
        raise
else:
    bootstrap_profiling_from_env()


def __getattr__(name: str):
    if name not in _KERNEL_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    import importlib

    module_name, attr_name = _KERNEL_EXPORTS[name]
    module = importlib.import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


__all__ = [
    "mm",
    "moe_route",
    "moe_dispatch",
    "moe_experts",
    "moe_combine",
    "moe_fused",
    "mha_prefill",
    "mha_prefill_with_kvcache",
    "mha_decode_with_kvcache",
]
