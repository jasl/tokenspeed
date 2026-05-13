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

from dataclasses import dataclass


@dataclass(frozen=True)
class _FakeArch:
    major: int
    minor: int


@dataclass(frozen=True)
class _FakePlatform:
    arch_version: _FakeArch
    is_nvidia: bool = True
    is_blackwell: bool = True


def test_sm120_mxfp4_ep_selects_sm12x_backend(monkeypatch):
    from tokenspeed.runtime.layers.moe import backends
    from tokenspeed.runtime.layers.moe.backends.mxfp4 import flashinfer, sm12x
    from tokenspeed.runtime.layers.moe.core import registry, selector
    from tokenspeed.runtime.layers.moe.core.types import MoELayerSpec
    from tokenspeed.runtime.layers.quantization import Mxfp4Config

    monkeypatch.setattr(registry, "_REGISTRY", {})
    monkeypatch.setattr(backends, "_REGISTERED", set())
    monkeypatch.setattr(
        selector,
        "current_platform",
        lambda: _FakePlatform(arch_version=_FakeArch(major=12, minor=0)),
    )
    monkeypatch.setattr(flashinfer, "_is_nvidia", True)

    spec = MoELayerSpec(
        top_k=8,
        num_experts=256,
        num_local_experts=128,
        hidden_size=7168,
        intermediate_size=2048,
        activation="silu",
        tp_rank=0,
        tp_size=1,
        ep_rank=0,
        ep_size=2,
        prefix="model.layers.3.mlp",
    )

    backend = selector.select_backend(
        spec,
        Mxfp4Config(is_checkpoint_mxfp4_serialized=True),
    )

    assert isinstance(backend, sm12x.Mxfp4Sm12xBackend)
    assert backend.key.arch == "sm120"
    assert backend.key.impl == "sm12x_mxfp4"


def test_sm121_mxfp4_tp_selects_sm12x_backend(monkeypatch):
    from tokenspeed.runtime.layers.moe import backends
    from tokenspeed.runtime.layers.moe.backends.mxfp4 import flashinfer, sm12x
    from tokenspeed.runtime.layers.moe.core import registry, selector
    from tokenspeed.runtime.layers.moe.core.types import MoELayerSpec
    from tokenspeed.runtime.layers.quantization import Mxfp4Config

    monkeypatch.setattr(registry, "_REGISTRY", {})
    monkeypatch.setattr(backends, "_REGISTERED", set())
    monkeypatch.setattr(
        selector,
        "current_platform",
        lambda: _FakePlatform(arch_version=_FakeArch(major=12, minor=1)),
    )
    monkeypatch.setattr(flashinfer, "_is_nvidia", True)

    spec = MoELayerSpec(
        top_k=8,
        num_experts=256,
        num_local_experts=256,
        hidden_size=7168,
        intermediate_size=2048,
        activation="silu",
        tp_rank=0,
        tp_size=2,
        ep_rank=0,
        ep_size=1,
        prefix="model.layers.3.mlp",
    )

    backend = selector.select_backend(
        spec,
        Mxfp4Config(is_checkpoint_mxfp4_serialized=True),
    )

    assert isinstance(backend, sm12x.Mxfp4Sm12xBackend)
    assert backend.key.arch == "sm121"
    assert backend.key.impl == "sm12x_mxfp4"


def test_sm12x_mxfp4_keeps_concatenated_weights_in_place(monkeypatch):
    import torch

    from tokenspeed.runtime.layers.moe.backends.mxfp4.sm12x import (
        Mxfp4Sm12xBackend,
    )
    from tokenspeed.runtime.layers.moe.core.types import BackendKey, MoELayerSpec
    from tokenspeed.runtime.layers.quantization import Mxfp4Config

    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)

    spec = MoELayerSpec(
        top_k=2,
        num_experts=2,
        num_local_experts=2,
        hidden_size=64,
        intermediate_size=32,
        activation="swiglu",
        tp_rank=0,
        tp_size=1,
        ep_rank=0,
        ep_size=1,
        prefix="model.layers.3.mlp",
    )
    backend = Mxfp4Sm12xBackend(
        key=BackendKey("sm120", "mxfp4", "sm12x_mxfp4"),
        spec=spec,
        quant_config=Mxfp4Config(is_checkpoint_mxfp4_serialized=True),
    )
    layer = torch.nn.Module()
    layer.activation = "swiglu"
    layer.swiglu_arg = None
    layer.swiglu_beta = None
    layer.w13_input_layout = "concatenated"
    backend.create_layer_weights(layer, with_bias=True)

    w13_ptr = layer.w13_weight.data_ptr()
    w2_ptr = layer.w2_weight.data_ptr()
    layer.w13_weight.data.copy_(
        torch.arange(layer.w13_weight.numel(), dtype=torch.uint8).reshape_as(
            layer.w13_weight
        )
    )
    expected_w13 = layer.w13_weight.detach().clone()

    backend.process_weights_after_loading(layer)

    assert layer.w13_weight.data_ptr() == w13_ptr
    assert layer.w2_weight.data_ptr() == w2_ptr
    assert torch.equal(layer.w13_weight, expected_w13)
    assert layer.w13_weight_bias.dtype == torch.float32
    assert layer.w2_weight_bias.dtype == torch.float32
    assert layer.sm12x_mxfp4_layout.w13_input_layout == "concatenated"


def test_sm12x_mxfp4_canonicalizes_interleaved_w13_rows(monkeypatch):
    import torch

    from tokenspeed.runtime.layers.moe.backends.mxfp4.sm12x import (
        Mxfp4Sm12xBackend,
    )
    from tokenspeed.runtime.layers.moe.core.types import BackendKey, MoELayerSpec
    from tokenspeed.runtime.layers.quantization import Mxfp4Config

    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)

    spec = MoELayerSpec(
        top_k=2,
        num_experts=1,
        num_local_experts=1,
        hidden_size=64,
        intermediate_size=32,
        activation="swiglu",
        tp_rank=0,
        tp_size=1,
        ep_rank=0,
        ep_size=1,
        prefix="model.layers.3.mlp",
    )
    backend = Mxfp4Sm12xBackend(
        key=BackendKey("sm120", "mxfp4", "sm12x_mxfp4"),
        spec=spec,
        quant_config=Mxfp4Config(is_checkpoint_mxfp4_serialized=True),
    )
    layer = torch.nn.Module()
    layer.activation = "swiglu"
    layer.swiglu_arg = None
    layer.swiglu_beta = 1.0
    layer.w13_input_layout = "interleaved"
    backend.create_layer_weights(layer, with_bias=True)

    row_ids = torch.arange(layer.w13_weight.shape[1], dtype=torch.uint8)
    layer.w13_weight.data[0].copy_(row_ids[:, None])
    layer.w13_weight_scale.data[0].copy_(row_ids[:, None])
    layer.w13_weight_bias.data[0].copy_(row_ids.to(torch.bfloat16))
    w13_ptr = layer.w13_weight.data_ptr()

    backend.process_weights_after_loading(layer)

    expected_rows = torch.cat([row_ids[0::2], row_ids[1::2]])
    assert layer.w13_weight.data_ptr() == w13_ptr
    assert torch.equal(layer.w13_weight[0, :, 0], expected_rows)
    assert torch.equal(layer.w13_weight_scale[0, :, 0], expected_rows)
    assert torch.equal(layer.w13_weight_bias[0], expected_rows.to(torch.float32))
    assert layer.sm12x_mxfp4_layout.w13_input_layout == "concatenated"


def test_sm12x_mxfp4_forward_uses_native_op_by_default(monkeypatch):
    import tokenspeed_kernel.ops.moe.sm12x_mxfp4 as sm12x_ops
    import torch

    from tokenspeed.runtime.layers.moe.backends.mxfp4.sm12x import (
        Mxfp4Sm12xBackend,
    )
    from tokenspeed.runtime.layers.moe.core.types import BackendKey, MoELayerSpec
    from tokenspeed.runtime.layers.moe.topk import StandardTopKOutput
    from tokenspeed.runtime.layers.quantization import Mxfp4Config

    monkeypatch.delenv("TOKENSPEED_SM12X_MXFP4_REFERENCE_FORWARD", raising=False)

    spec = MoELayerSpec(
        top_k=2,
        num_experts=4,
        num_local_experts=2,
        hidden_size=64,
        intermediate_size=32,
        activation="swiglu",
        tp_rank=0,
        tp_size=1,
        ep_rank=1,
        ep_size=2,
        prefix="model.layers.3.mlp",
    )
    backend = Mxfp4Sm12xBackend(
        key=BackendKey("sm120", "mxfp4", "sm12x_mxfp4"),
        spec=spec,
        quant_config=Mxfp4Config(is_checkpoint_mxfp4_serialized=True),
    )
    layer = torch.nn.Module()
    layer.activation = "swiglu"
    layer.swiglu_arg = None
    layer.swiglu_beta = None
    backend.create_layer_weights(layer, with_bias=True)

    calls = []

    def fake_native_forward(*args, **kwargs):
        calls.append((args, kwargs))
        return torch.full_like(args[0], 7.0)

    monkeypatch.setattr(sm12x_ops, "sm12x_mxfp4_moe_forward", fake_native_forward)

    hidden_states = torch.ones(3, 64)
    topk_output = StandardTopKOutput(
        torch.ones(3, 2),
        torch.tensor([[2, 3], [3, 2], [1, -1]], dtype=torch.int32),
        torch.empty(3, 4),
    )

    actual = backend.forward(layer, hidden_states, topk_output, 3, 3)

    assert torch.equal(actual, torch.full_like(hidden_states, 7.0))
    assert len(calls) == 1
    assert calls[0][1]["activation"] == "swiglu"
    assert calls[0][1]["ep_rank"] == 1
    assert calls[0][1]["ep_size"] == 2


def test_flashinfer_mxfp4_stacks_permuted_experts_without_materializing_list():
    import torch

    from tokenspeed.runtime.layers.moe.backends.mxfp4.flashinfer import (
        _stack_permuted_expert_rows,
    )

    experts = torch.arange(2 * 4 * 3, dtype=torch.uint8).reshape(2, 4, 3)
    permute_indices = torch.tensor([3, 1, 2, 0])

    def row_transform(expert):
        return torch.cat([expert[2:], expert[:2]], dim=0)

    def after_permute(expert):
        return expert + 7

    actual = _stack_permuted_expert_rows(
        experts,
        permute_indices,
        row_transform=row_transform,
        after_permute=after_permute,
    )

    expected = torch.stack(
        [
            after_permute(row_transform(experts[idx])[permute_indices].contiguous())
            for idx in range(experts.shape[0])
        ]
    )
    assert torch.equal(actual, expected)


def test_flashinfer_mxfp4_copies_permuted_experts_in_place():
    import torch

    from tokenspeed.runtime.layers.moe.backends.mxfp4.flashinfer import (
        _copy_permuted_expert_rows_,
    )

    experts = torch.arange(2 * 4 * 3, dtype=torch.uint8).reshape(2, 4, 3)
    storage_ptr = experts.data_ptr()
    permute_indices = torch.tensor([3, 1, 2, 0])

    def row_transform(expert):
        return torch.cat([expert[2:], expert[:2]], dim=0)

    expected = torch.stack(
        [
            row_transform(experts[idx])[permute_indices].contiguous()
            for idx in range(experts.shape[0])
        ]
    )

    actual = _copy_permuted_expert_rows_(
        experts,
        permute_indices,
        row_transform=row_transform,
    )

    assert actual.data_ptr() == storage_ptr
    assert torch.equal(actual, expected)


def test_triton_mxfp4_releases_w13_before_swizzling_w2(monkeypatch):
    import sys
    import types

    import torch

    from tokenspeed.runtime.layers.moe.backends.mxfp4 import triton_kernel
    from tokenspeed.runtime.layers.moe.core.types import MoELayerSpec
    from tokenspeed.runtime.layers.quantization import Mxfp4Config

    class FlexCtx:
        def __init__(self, rhs_data):
            self.rhs_data = rhs_data

    class PrecisionConfig:
        def __init__(self, *, b_mx_scale, b_microblock_size, flex_ctx):
            self.b_mx_scale = b_mx_scale
            self.b_microblock_size = b_microblock_size
            self.flex_ctx = flex_ctx

    fake_triton_kernels = types.ModuleType("tokenspeed_kernel.ops.moe.triton_kernels")
    fake_triton_kernels.FlexCtx = FlexCtx
    fake_triton_kernels.PrecisionConfig = PrecisionConfig
    monkeypatch.setitem(
        sys.modules,
        "tokenspeed_kernel.ops.moe.triton_kernels",
        fake_triton_kernels,
    )
    monkeypatch.setattr(triton_kernel.torch.cuda, "empty_cache", lambda: None)

    spec = MoELayerSpec(
        top_k=8,
        num_experts=256,
        num_local_experts=256,
        hidden_size=128,
        intermediate_size=64,
        activation="silu",
        tp_rank=0,
        tp_size=2,
        ep_rank=0,
        ep_size=1,
        prefix="model.layers.3.mlp",
    )
    backend = triton_kernel.Mxfp4TritonKernelBackend(
        key=None,
        spec=spec,
        quant_config=Mxfp4Config(is_checkpoint_mxfp4_serialized=True),
    )
    layer = torch.nn.Module()
    layer.w13_weight = torch.nn.Parameter(torch.ones(1, 2), requires_grad=False)
    layer.w13_weight_scale = torch.nn.Parameter(torch.ones(1, 2), requires_grad=False)
    layer.w2_weight = torch.nn.Parameter(torch.ones(1, 2), requires_grad=False)
    layer.w2_weight_scale = torch.nn.Parameter(torch.ones(1, 2), requires_grad=False)
    layer.w13_weight_bias = torch.nn.Parameter(
        torch.ones(1, 2, dtype=torch.bfloat16), requires_grad=False
    )
    layer.w2_weight_bias = torch.nn.Parameter(
        torch.ones(1, 2, dtype=torch.bfloat16), requires_grad=False
    )

    calls = []

    def fake_swizzle(weight, scale, num_warps):
        calls.append((weight, scale, num_warps))
        if len(calls) == 2:
            assert not hasattr(layer, "w13_weight")
            assert not hasattr(layer, "w13_weight_scale")
            assert hasattr(layer, "w13_weight_triton_tensor")
            assert hasattr(layer, "w13_precision_config")
        return torch.empty(0), object(), torch.empty(0)

    monkeypatch.setattr(triton_kernel, "swizzle_mxfp4", fake_swizzle)

    backend.process_weights_after_loading(layer)

    assert len(calls) == 2
    assert not hasattr(layer, "w2_weight")
    assert not hasattr(layer, "w2_weight_scale")
    assert hasattr(layer, "w2_weight_triton_tensor")
    assert hasattr(layer, "w2_precision_config")
