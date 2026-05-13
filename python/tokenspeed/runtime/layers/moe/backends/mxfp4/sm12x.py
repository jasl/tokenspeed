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

import os
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn.parameter import Parameter

from tokenspeed.runtime.layers.moe.backends.base import MoEBackend
from tokenspeed.runtime.layers.moe.backends.mxfp4.weights import (
    MXFP4_BLOCK,
    create_mxfp4_weights,
)
from tokenspeed.runtime.layers.moe.core.types import MoELayerSpec
from tokenspeed.runtime.layers.moe.topk import TopKOutputFormat
from tokenspeed.runtime.layers.quantization import Mxfp4Config


@dataclass(frozen=True)
class Sm12xMxfp4Layout:
    num_local_experts: int
    hidden_size_padded: int
    intermediate_size_padded: int
    w13_input_layout: str


def _canonicalize_interleaved_w13_rows_(rows: torch.Tensor) -> torch.Tensor:
    if rows.shape[-2] % 2 != 0:
        raise ValueError(f"expected even w13 row count, got {rows.shape[-2]}")

    half = rows.shape[-2] // 2
    for expert_idx in range(rows.shape[0]):
        expert = rows[expert_idx]
        w1 = expert[0::2].clone()
        w3 = expert[1::2].clone()
        expert[:half].copy_(w1)
        expert[half:].copy_(w3)
    return rows


def _as_float32_parameter(
    param: torch.nn.Parameter | None,
) -> torch.nn.Parameter | None:
    if param is None:
        return None
    return Parameter(param.data.to(torch.float32), requires_grad=False)


class Mxfp4Sm12xBackend(MoEBackend):
    supported_arches = frozenset({"sm120", "sm121"})

    def __init__(
        self,
        key,
        spec: MoELayerSpec,
        quant_config: object,
        routing_config: dict | None = None,
    ):
        del routing_config
        self.key = key
        self.spec = spec
        self.quant_config = quant_config
        self._activation: str | None = None
        self._swiglu_arg = None
        self._swiglu_beta = None
        self._hidden_padded = 0
        self._ispp_padded = 0

    @classmethod
    def supports(cls, spec: MoELayerSpec, quant_config: object) -> bool:
        return (
            isinstance(quant_config, Mxfp4Config)
            and not spec.use_deepep
            and spec.activation in {"silu", "swiglu"}
        )

    @property
    def topk_output_format(self) -> TopKOutputFormat:
        return TopKOutputFormat.STANDARD

    def create_layer_weights(
        self, layer: nn.Module, *, with_bias: bool = False
    ) -> None:
        from tokenspeed.runtime.utils import round_up

        hidden = self.spec.hidden_size
        ispp = self.spec.intermediate_size // self.spec.tp_size
        self._hidden_padded = round_up(hidden, MXFP4_BLOCK)
        self._ispp_padded = round_up(ispp, MXFP4_BLOCK)

        create_mxfp4_weights(
            self,
            layer,
            self.spec.num_local_experts,
            self._hidden_padded,
            self._ispp_padded,
            with_bias=with_bias,
        )

        self._activation = layer.activation
        self._swiglu_arg = getattr(layer, "swiglu_arg", None)
        self._swiglu_beta = getattr(layer, "swiglu_beta", None)

    def process_weights_after_loading(self, layer: nn.Module) -> None:
        w13_layout = getattr(layer, "w13_input_layout", "concatenated")
        if w13_layout == "interleaved":
            _canonicalize_interleaved_w13_rows_(layer.w13_weight.data)
            _canonicalize_interleaved_w13_rows_(layer.w13_weight_scale.data)
            if getattr(layer, "w13_weight_bias", None) is not None:
                _canonicalize_interleaved_w13_rows_(
                    layer.w13_weight_bias.data[:, :, None]
                )
            w13_layout = "concatenated"
        elif w13_layout != "concatenated":
            raise ValueError(f"unknown w13_input_layout: {w13_layout!r}")

        if getattr(layer, "w13_weight_bias", None) is not None:
            layer.w13_weight_bias = _as_float32_parameter(layer.w13_weight_bias)
        if getattr(layer, "w2_weight_bias", None) is not None:
            layer.w2_weight_bias = _as_float32_parameter(layer.w2_weight_bias)

        layer.sm12x_mxfp4_layout = Sm12xMxfp4Layout(
            num_local_experts=self.spec.num_local_experts,
            hidden_size_padded=self._hidden_padded,
            intermediate_size_padded=self._ispp_padded,
            w13_input_layout=w13_layout,
        )
        torch.cuda.empty_cache()

    def forward(
        self,
        layer: nn.Module,
        hidden_states: torch.Tensor,
        topk_output: object,
        num_global_tokens: int,
        max_num_tokens_per_gpu: int,
    ) -> torch.Tensor:
        del num_global_tokens, max_num_tokens_per_gpu
        if os.getenv("TOKENSPEED_SM12X_MXFP4_REFERENCE_FORWARD") == "1":
            from tokenspeed_kernel.ops.moe.sm12x_mxfp4 import (
                sm12x_mxfp4_moe_reference_forward,
            )

            forward_impl = sm12x_mxfp4_moe_reference_forward
        else:
            from tokenspeed_kernel.ops.moe.sm12x_mxfp4 import sm12x_mxfp4_moe_forward

            forward_impl = sm12x_mxfp4_moe_forward

        topk_weights = topk_output.topk_weights
        topk_ids = topk_output.topk_ids
        swiglu_alpha = None
        swiglu_limit = None
        if self._swiglu_arg is not None:
            swiglu_alpha = self._swiglu_arg.alpha
            swiglu_limit = self._swiglu_arg.limit

        return forward_impl(
            hidden_states,
            topk_weights,
            topk_ids,
            layer.w13_weight,
            layer.w13_weight_scale,
            layer.w2_weight,
            layer.w2_weight_scale,
            w13_bias=getattr(layer, "w13_weight_bias", None),
            w2_bias=getattr(layer, "w2_weight_bias", None),
            activation=self._activation or layer.activation,
            swiglu_alpha=swiglu_alpha,
            swiglu_limit=swiglu_limit,
            swiglu_beta=self._swiglu_beta,
            ep_rank=self.spec.ep_rank,
            ep_size=self.spec.ep_size,
        )


__all__ = [
    "Mxfp4Sm12xBackend",
    "Sm12xMxfp4Layout",
]
