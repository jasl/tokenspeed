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

from functools import partial


def test_expert_e8m0_weight_scale_load_preserves_raw_bytes():
    import pytest
    import torch

    e8m0_dtype = getattr(torch, "float8_e8m0fnu", None)
    if e8m0_dtype is None:
        pytest.skip("torch.float8_e8m0fnu is not available")

    from tokenspeed.runtime.layers.moe.backends.weight_loaders import (
        load_model_weight,
    )
    from tokenspeed.runtime.layers.moe.checkpoint import (
        ExpertCheckpointSchema,
        build_moe_checkpoint_loader,
    )
    from tokenspeed.runtime.utils import set_weight_attrs

    param = torch.nn.Parameter(
        torch.zeros(1, 4, 2, dtype=torch.uint8),
        requires_grad=False,
    )
    set_weight_attrs(
        param,
        {
            "weight_loader": partial(
                load_model_weight,
                tp_rank=0,
                is_bias=False,
                use_presharded_weights=False,
                do_transpose=False,
            )
        },
    )
    loader = build_moe_checkpoint_loader(
        params_dict={"model.layers.0.ffn.experts.w13_weight_scale": param},
        expert_schema=ExpertCheckpointSchema(
            gate_proj_name="w1",
            down_proj_name="w2",
            up_proj_name="w3",
        ),
        num_experts=1,
    )
    raw_bytes = torch.tensor(
        [[123, 124], [125, 126]],
        dtype=torch.uint8,
    )

    loader.load(
        "model.layers.0.ffn.experts.0.w1.weight_scale",
        raw_bytes.view(e8m0_dtype),
    )

    assert torch.equal(param[0, :2], raw_bytes)
